"""One robot HAL composed from part HALs that each own a disjoint set of its joints.

A mobile manipulator is often an existing arm on an existing base: OpenAMRobot 2.0
is the Enactic OpenArm (16 joints, ``OpenArmRealHAL``) on a mobile platform with a
lift. ``ComposedHAL`` serves the robot's manifest by delegating to those parts
instead of re-implementing them. Each part is built from its own manifest (the arm
part from ``robots/openarm/robot.yaml``) and owns the robot joints of the same name.

Routing, all by name, never by position:

* ``read_state``: each part's ``JointState`` is merged into the robot's joint order.
* ``JOINT_POSITION``: the targets are split per part. A whole-robot row reaches every
  part as that part's own row.
* ADR-0102 slot groups: the group is staged here, then each part receives only its
  own slots, as a group of that size. So the arm part stages the same four-slot
  OpenArm tick it would get on the OpenArm itself.
* ``GRIPPER_*``: the part that has the end effector (``ee_name``).
* ``BODY_TWIST``: the part that declares ``base_joints``.
* Before any part is commanded, every part that can refuse its share
  (``check_action``: a held lift, a base that takes only twists) is asked, so a
  refusal never lands after another part already moved.
* ``estop``: every part is stopped. Then ``ROSEStopRequested`` is raised, unless a
  part's stop failed with another error: that error propagates instead (the
  others noted on it), so a failed base stop is never reported as a clean stop.

The composite also forwards the optional surfaces the lifecycle node probes
structurally: the ros2_control transport seam (delegated to its one
``RosControlDrivable`` part; with none or several those calls raise
``ROSConfigError``, while ``last_stop_report`` reads ``None`` and ``estop_recovery``
is absent), ``attach_ros_node``, and ``base_pose``/``base_twist``.
"""

from __future__ import annotations

import dataclasses
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import structlog
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError, ROSEStopRequested
from openral_core.schemas import Action, ControlMode, JointState
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from openral_hal._slot_group import SlotGroupStager, slot_group_targets

if TYPE_CHECKING:
    from openral_hal.protocol import HAL

__all__ = ["ComposedHAL", "ComposedPartSpec", "HeldJointsHAL"]

log = structlog.get_logger(__name__)

_JOINT_MODES = frozenset(
    {
        ControlMode.JOINT_POSITION.value,
        ControlMode.JOINT_VELOCITY.value,
        ControlMode.JOINT_TORQUE.value,
    }
)
# A held joint accepts a target within this of its declared position (float round-trip).
_HELD_TOLERANCE = 1e-6
_GRIPPER_MODES = frozenset({ControlMode.GRIPPER_POSITION.value, ControlMode.GRIPPER_BINARY.value})


def _repo_robots_dir() -> Path:
    # python/hal/src/openral_hal/composed.py -> repo root (the panda_mobile convention).
    return Path(__file__).resolve().parents[4] / "robots"


def _import(entry: str) -> Callable[..., Any]:
    from openral_hal.resolver import _import_object  # noqa: PLC0415  # reason: avoid cycle

    obj = _import_object(entry)
    if not callable(obj):
        raise ROSConfigError(f"ComposedHAL part entry {entry!r} is not callable.")
    return obj


class ComposedPartSpec(BaseModel):
    """One ``hal.parameters.defaults.parts`` entry of a ``ComposedHAL`` manifest.

    Unknown keys are refused, so a typo (``param:``) fails at load instead of
    silently starting a part on its default topics or buses.

    Example:
        >>> ComposedPartSpec(entry="openral_hal.composed:HeldJointsHAL", joints=["lift_joint"])
        ... # doctest: +ELLIPSIS
        ComposedPartSpec(entry='openral_hal.composed:HeldJointsHAL', robot=None, ...)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: str
    """``"module:Class"`` of the part HAL."""
    robot: str | None = None
    """Build the part from ``robots/<robot>/robot.yaml``."""
    joints: list[str] | None = None
    """Without ``robot``: the robot joints the part owns."""
    params: dict[str, Any] = {}
    """The part constructor's keyword arguments."""

    @model_validator(mode="after")
    def _robot_or_joints(self) -> ComposedPartSpec:
        if not self.robot and not self.joints:
            raise ValueError(f"part {self.entry!r} needs 'robot' or 'joints'")
        return self


class HeldJointsHAL:
    """A part for joints the robot has but no interface drives or reads.

    OpenAMRobot 2.0's real lift has no ros2_control interface (its firmware owns
    it), so the only knowable height is the one the operator declares. This part
    reports the declared ``positions`` and refuses any command that would move
    them: a declared joint cannot be commanded, and a silently ignored command
    would let a skill believe the lift moved.

    Args:
        description: The part's description; its joints are the held joints.
        positions: Declared position of every held joint (metres / radians).

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openamr_v2/robot.yaml")
        >>> part = d.model_copy(update={"joints": [j for j in d.joints if j.name == "lift_joint"]})
        >>> hal = HeldJointsHAL(description=part, positions={"lift_joint": 0.0})
        >>> hal.connect()
        >>> hal.read_state().position
        [0.0]
    """

    def __init__(self, *, description: RobotDescription, positions: Mapping[str, float]) -> None:
        """Validate that ``positions`` names exactly the description's joints."""
        names = [j.name for j in description.joints]
        if sorted(positions) != sorted(names):
            raise ROSConfigError(
                f"HeldJointsHAL positions {sorted(positions)} must name exactly {sorted(names)}."
            )
        self.description = description
        self._positions = [float(positions[n]) for n in names]
        self._connected = False

    def connect(self) -> None:
        """Nothing to open; the declared positions are the state."""
        self._connected = True

    def disconnect(self) -> None:
        """Idempotent."""
        self._connected = False

    def read_state(self) -> JointState:
        """The declared positions, stamped now."""
        return JointState(
            stamp_ns=time.time_ns(),  # wall clock, as every HAL stamps JointState
            name=[j.name for j in self.description.joints],
            position=list(self._positions),
            velocity=[0.0] * len(self._positions),
            effort=[],  # nothing measures it (JointState: never zero-filled)
        )

    def send_action(self, action: Action) -> None:
        """Accept only a joint target equal to the declared position."""
        self.check_action(action)

    def check_action(self, action: Action) -> None:
        """Refuse anything but a joint target equal to the declared position."""
        if action.control_mode != ControlMode.JOINT_POSITION.value or not action.joint_targets:
            raise ROSConfigError(
                f"HeldJointsHAL: {action.control_mode} cannot drive held joints "
                f"{[j.name for j in self.description.joints]}."
            )
        target = list(action.joint_targets[-1])
        if any(abs(t - p) > _HELD_TOLERANCE for t, p in zip(target, self._positions, strict=True)):
            raise ROSConfigError(
                f"HeldJointsHAL: {[j.name for j in self.description.joints]} have no command "
                f"interface; held at {self._positions}, refusing a move to {target}."
            )

    def estop(self) -> None:
        """Nothing to de-energise here; report the stop like every HAL."""
        raise ROSEStopRequested(f"Emergency stop on held joints of '{self.description.name}'.")


class ComposedHAL:
    """A robot HAL that delegates to part HALs, each owning a disjoint set of joints.

    ``parts`` comes from ``hal.parameters.defaults.parts`` in the robot manifest.
    Each entry has these keys:

    * ``entry``: ``"module:Class"`` of the part HAL.
    * ``robot`` (optional): build the part from ``robots/<robot>/robot.yaml``. Its
      joint names must exist on this robot. Without it, the part gets this robot's
      description narrowed to ``joints``.
    * ``joints`` (required without ``robot``): the robot joints the part owns.
    * ``params`` (optional): the part constructor's keyword arguments. ``description``
      is threaded when the constructor names it.

    (``ComposedPartSpec`` types an entry.) Every robot joint must be owned by exactly
    one part. ``staleness_limit_s`` (the manifest's
    ``safety.joint_state_staleness_limit_s``, threaded by ``build_hal``) and any other
    keyword (a ``deploy run`` transport override such as a CAN interface) reach every
    part whose constructor takes them, over that part's ``params``.

    Raises:
        ROSConfigError: Malformed ``parts``, overlapping or missing joint ownership,
            or a part manifest whose joints this robot does not have.
    """

    # The vendor joint_state_broadcaster covers only the ros2_control part's joints,
    # so the HAL node publishes the URDF-named state of every joint itself
    # (``resolver.vendor_publishes_urdf_joint_states``).
    urdf_joint_states_from_hal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        description: RobotDescription,
        parts: Sequence[Mapping[str, Any]],
        staleness_limit_s: float | None = None,
        **overrides: Any,  # noqa: ANN401  # reason: per-part constructor kwargs
    ) -> None:
        """Build every part; nothing connects until ``connect``."""
        if not parts:
            raise ROSConfigError(f"ComposedHAL for '{description.name}' needs at least one part.")
        try:
            specs = [ComposedPartSpec.model_validate(dict(p)) for p in parts]
        except ValidationError as exc:
            raise ROSConfigError(f"ComposedHAL '{description.name}' parts: {exc}") from exc
        if staleness_limit_s is not None:
            overrides["staleness_limit_s"] = staleness_limit_s
        self.description = description
        robot_joints = [j.name for j in description.joints]
        self._parts: list[HAL] = []
        self._owner: dict[str, int] = {}
        for i, spec in enumerate(specs):
            part_desc = self._part_description(description, spec)
            for name in (j.name for j in part_desc.joints):
                if name not in robot_joints:
                    raise ROSConfigError(
                        f"ComposedHAL part {spec.entry!r} owns {name!r}, which "
                        f"'{description.name}' does not have."
                    )
                if name in self._owner:
                    raise ROSConfigError(f"ComposedHAL: joint {name!r} is owned by two parts.")
                self._owner[name] = i
            ctor = _import(spec.entry)
            accepted = inspect.signature(ctor).parameters
            kwargs = {
                **spec.params,
                **{k: v for k, v in overrides.items() if k in accepted},
            }
            if "description" in accepted:
                kwargs["description"] = part_desc
            self._parts.append(ctor(**kwargs))
        unowned = [n for n in robot_joints if n not in self._owner]
        if unowned:
            raise ROSConfigError(f"ComposedHAL: no part owns {unowned}.")
        self._ee_owner = {
            ee.name: i for i, p in enumerate(self._parts) for ee in p.description.end_effectors
        }
        bases = [i for i, p in enumerate(self._parts) if p.description.base_joints]
        if len(bases) > 1:
            raise ROSConfigError("ComposedHAL: more than one part declares base_joints.")
        self._base: int | None = bases[0] if bases else None
        self._slot_group = SlotGroupStager()
        # Last read position of every joint (read_state), for last_applied_action.
        self._last_positions: dict[str, float] = {}

    @staticmethod
    def _part_description(robot: RobotDescription, spec: ComposedPartSpec) -> RobotDescription:
        if spec.robot:
            path = _repo_robots_dir() / spec.robot / "robot.yaml"
            if not path.is_file():
                raise ROSConfigError(f"ComposedHAL part manifest not found: {path}")
            return RobotDescription.from_yaml(str(path))
        names = list(spec.joints or [])
        by_name = {j.name: j for j in robot.joints}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ROSConfigError(f"ComposedHAL part {spec.entry!r}: unknown joints {missing}.")
        base = robot.base_joints if set(robot.base_joints or []) <= set(names) else None
        return robot.model_copy(
            update={
                "joints": [by_name[n] for n in names],
                "base_joints": base,
                "end_effectors": [],
                "sim": None,
                "collision_geometry": [],
                "allowed_collision_pairs": [],
                "mimic_joints": [],
                "fixed_attachments": [],
                "force_observer": None,
            }
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect every part; a failure disconnects the ones already up."""
        done: list[HAL] = []
        try:
            for part in self._parts:
                part.connect()
                done.append(part)
        except BaseException:
            for part in reversed(done):
                part.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect every part (idempotent per the protocol)."""
        self._slot_group.reset()
        for part in reversed(self._parts):
            part.disconnect()

    def estop(self) -> None:
        """Stop every part, then raise.

        Each part's ``estop`` raises ``ROSEStopRequested`` by contract. Every part
        is stopped whatever an earlier one raised. If they all stopped,
        ``ROSEStopRequested`` is raised. If any part's stop failed with another
        error, the first such error is re-raised (any later failures noted on it),
        so the caller's "hardware estop failed" path runs.
        """
        self._slot_group.discard()
        failures: list[BaseException] = []
        for part in self._parts:
            try:
                part.estop()
            except ROSEStopRequested:
                continue  # the contract's success signal, re-raised once below
            except (
                BaseException
            ) as exc:  # reason: every part must still be stopped; re-raised below
                exc.add_note(f"ComposedHAL: e-stop of part {type(part).__name__} failed.")
                failures.append(exc)
        if failures:
            for later in failures[1:]:
                failures[0].add_note(f"also failed: {type(later).__name__}: {later}")
            log.error(
                "hal.composed.estop_failed",
                robot=self.description.name,
                failures=[f"{type(e).__name__}: {e}" for e in failures],
            )
            raise failures[0]
        raise ROSEStopRequested(f"Emergency stop on robot '{self.description.name}'.")

    # ── state ────────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Every part's state merged into ``description.joints`` order, by name.

        ``effort`` is all joints or none (``JointState``): it is merged only when
        every part measures it, else empty. A part without effort (odometry, a held
        lift) is never zero-filled, since a torque observer reads 0.0 as unloaded.
        """
        pos: dict[str, float] = {}
        vel: dict[str, float] = {}
        eff: dict[str, float] = {}
        stamps: list[int] = []
        for part in self._parts:
            js = part.read_state()
            names = js.name or [j.name for j in part.description.joints]
            for k, name in enumerate(names):
                pos[name] = js.position[k]
                vel[name] = js.velocity[k] if js.velocity else 0.0
                if js.effort:
                    eff[name] = js.effort[k]
            stamps.append(js.stamp_ns)
        names = [j.name for j in self.description.joints]
        self._last_positions = pos
        return JointState(
            stamp_ns=min(stamps),  # the oldest part bounds the state's freshness
            name=names,
            position=[pos[n] for n in names],
            velocity=[vel[n] for n in names],
            effort=[eff[n] for n in names] if all(n in eff for n in names) else [],
        )

    # ── actions ──────────────────────────────────────────────────────────────

    def send_action(self, action: Action) -> None:
        """Route ``action`` to the part(s) that own what it commands (see module doc)."""
        if int(action.tick_group_size) > 1:
            group = self._slot_group.stage(action)
            if group is None:
                return
            # The tick's whole-robot command, for consumers of last_applied_action
            # (the vision attachment bridge): the slots' targets by name, every joint
            # they leave uncommanded (base, held lift) at its last read position. Not
            # re-read here: an arms-only tick must not fail on base odometry.
            names = [j.name for j in self.description.joints]
            targets = slot_group_targets(group, names)
            if any(
                t is None and n not in self._last_positions
                for t, n in zip(targets, names, strict=True)
            ):
                self.read_state()
            row = [
                self._last_positions[n] if t is None else t
                for t, n in zip(targets, names, strict=True)
            ]
            applied = Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[row],
                tick_index=action.tick_index,
                runner_session_id=action.runner_session_id,
            )
            by_part: dict[int, list[Action]] = {}
            for slot in group:
                for i, sub in self._split(slot):
                    by_part.setdefault(i, []).append(sub)
            self._dispatch(
                [
                    (i, sub.model_copy(update={"tick_group_size": len(subs)}))
                    for i, subs in by_part.items()
                    for sub in subs
                ]
            )
            self._slot_group.commit(group, applied=applied)
            return
        self._slot_group.admit(action)
        self._dispatch(self._split(action))
        self._slot_group.commit_tick(action)

    def _dispatch(self, shares: list[tuple[int, Action]]) -> None:
        """Ask every part that can refuse (``check_action``) first, then send all.

        A held lift or a twist-only base then refuses before any part has moved.
        """
        for i, sub in shares:
            check = getattr(self._parts[i], "check_action", None)
            if check is not None:
                check(sub)
        for i, sub in shares:
            self._parts[i].send_action(sub)

    def _split(self, action: Action) -> list[tuple[int, Action]]:
        mode = action.control_mode
        if mode == ControlMode.BODY_TWIST.value:
            if self._base is None:
                raise ROSConfigError(f"'{self.description.name}': no part drives the base.")
            return [(self._base, action)]
        if mode in _GRIPPER_MODES:
            if action.ee_name not in self._ee_owner:
                raise ROSConfigError(
                    f"ComposedHAL: no part has end effector {action.ee_name!r} "
                    f"(have {sorted(self._ee_owner)})."
                )
            return [(self._ee_owner[action.ee_name], action)]
        if mode not in _JOINT_MODES:
            raise ROSConfigError(f"ComposedHAL cannot route control mode {mode!r}.")
        field = {
            ControlMode.JOINT_POSITION.value: "joint_targets",
            ControlMode.JOINT_VELOCITY.value: "joint_velocities",
            ControlMode.JOINT_TORQUE.value: "joint_torques",
        }[mode]
        rows: list[list[float]] = [list(r) for r in (getattr(action, field) or [])]
        robot_names = [j.name for j in self.description.joints]
        names = list(action.joint_names or robot_names)
        robot_idx = {n: k for k, n in enumerate(robot_names)}
        # Rows are padded to the robot's dof with values at each joint's robot index
        # (what the rSkill runner sends), or compact, one value per joint name.
        if rows and all(len(r) == len(robot_names) for r in rows):
            src = robot_idx
        elif rows and all(len(r) == len(names) for r in rows):
            src = {n: k for k, n in enumerate(names)}
        else:
            raise ROSConfigError(
                f"ComposedHAL: {mode} rows must have {len(robot_names)} (padded) or "
                f"{len(names)} (one per joint name) values."
            )
        unknown = [n for n in names if n not in robot_idx]
        if unknown:
            raise ROSConfigError(f"ComposedHAL: {mode} names unknown joints {unknown}.")
        commanded = set(names)
        out: list[tuple[int, Action]] = []
        for i, part in enumerate(self._parts):
            part_names = [j.name for j in part.description.joints]
            owned = [n for n in part_names if n in commanded]
            if not owned:
                continue
            # Re-pad into the part's own dof: the part places values by its index.
            sub_rows = [[r[src[n]] if n in commanded else 0.0 for n in part_names] for r in rows]
            whole = action.joint_names is None and len(owned) == len(part_names)
            sub = action.model_copy(
                update={field: sub_rows, "joint_names": None if whole else owned}
            )
            out.append((i, sub))
        return out

    # ── optional surfaces the lifecycle node probes ─────────────────────────

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """The base part's ``(x, y, yaw)``."""
        if self._base is None:
            raise AttributeError("base_pose")
        pose: tuple[float, float, float] = self._parts[self._base].base_pose  # type: ignore[attr-defined]  # reason: base part contract
        return pose

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """The base part's last body twist."""
        if self._base is None:
            raise AttributeError("base_twist")
        twist: tuple[float, float, float, float, float, float] = self._parts[self._base].base_twist  # type: ignore[attr-defined]  # reason: base part contract
        return twist

    @property
    def last_committed_tick(self) -> int:
        """Tick of the last slot group applied here (all parts)."""
        return self._slot_group.last_committed_tick

    @property
    def last_committed_session(self) -> int:
        """Runner session of that tick."""
        return self._slot_group.last_committed_session

    @property
    def last_applied_action(self) -> Action | None:
        """The last whole-robot command applied (a slot group composed, see send_action)."""
        return self._slot_group.last_applied_action

    def discard_staged_slots(self) -> int:
        """Drop half-staged slots here and in every part that stages its own."""
        n = self._slot_group.discard()
        for part in self._parts:
            fn = getattr(part, "discard_staged_slots", None)
            if fn is not None:
                n += int(fn())
        return n

    def attach_ros_node(self, node: object) -> None:
        """Hand the lifecycle node to every part that needs one (e.g. a ROS base)."""
        for part in self._parts:
            fn = getattr(part, "attach_ros_node", None)
            if fn is not None:
                fn(node)

    # ros2_control transport seam (RosControlDrivable / ControllerStoppable). The
    # lifecycle node checks these protocols with static lookup, so they are real
    # members, delegated to the one ros2_control part (the OpenArm arm).

    def _ros_part(self) -> Any:  # noqa: ANN401  # reason: structural ros2_control HAL
        part = self._ros_part_or_none()
        if part is None:
            raise ROSConfigError(
                f"ComposedHAL '{self.description.name}': the ros2_control seam needs exactly "
                "one ros2_control part."
            )
        return part

    def _ros_part_or_none(self) -> Any:  # noqa: ANN401  # reason: structural ros2_control HAL
        from openral_hal.ros_control_transport import RosControlDrivable  # noqa: PLC0415

        parts = [p for p in self._parts if isinstance(p, RosControlDrivable)]
        return parts[0] if len(parts) == 1 else None

    def command_bindings(self) -> Any:  # noqa: ANN401  # reason: delegated
        """The arm part's controller command topics."""
        return self._ros_part().command_bindings()

    def ros2_control_joint_names(self) -> list[str]:
        """The arm part's URDF joint names on ``/joint_states``."""
        names: list[str] = self._ros_part().ros2_control_joint_names()
        return names

    @property
    def joint_state_topic(self) -> str:
        """The arm part's ``JointState`` topic."""
        topic: str = self._ros_part().joint_state_topic
        return topic

    def attach_transport(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401  # reason: delegated
        """Wire the arm part to the live ros2_control transport."""
        self._ros_part().attach_transport(*args, **kwargs)

    def controller_names(self) -> list[str]:
        """Controllers ``/openral/estop`` deactivates (the arm part's)."""
        names: list[str] = self._ros_part().controller_names()
        return names

    def vendor_stop_services(self) -> list[str]:
        """The arm part's vendor stop services."""
        services: list[str] = self._ros_part().vendor_stop_services()
        return services

    def vendor_stop_topics(self) -> list[str]:
        """The arm part's vendor stop topics."""
        topics: list[str] = self._ros_part().vendor_stop_topics()
        return topics

    def attach_controller_stop(self, seam: Any) -> None:  # noqa: ANN401  # reason: delegated
        """Give the arm part the controller-stop seam."""
        self._ros_part().attach_controller_stop(seam)

    @property
    def estop_recovery(self) -> Any:  # noqa: ANN401  # reason: EStopRecovery of the arm part
        """How the arm part recovers from an e-stop (the only part with motors to re-arm).

        Raises ``AttributeError`` without exactly one ros2_control part, so
        ``getattr(hal, "estop_recovery", default)`` falls back as for any HAL without it.
        """
        part = self._ros_part_or_none()
        if part is None:
            raise AttributeError("estop_recovery")
        return part.estop_recovery

    def reset_estop(self) -> None:
        """Re-arm the arm part after an e-stop."""
        self._ros_part().reset_estop()

    @property
    def last_stop_report(self) -> Any:  # noqa: ANN401  # reason: DownstreamStopReport | None
        """The arm part's downstream stop report, marked as covering the arm only.

        Only the ros2_control part can acknowledge a stop. The other parts' stops
        (a zero twist to the platform, a lift nothing drives) have no
        acknowledgement, so ``detail`` names them instead of claiming them.
        """
        part = self._ros_part_or_none()
        report = None if part is None else part.last_stop_report
        if report is None:
            return None  # "unproven", read inside the lifecycle node's e-stop handler
        others = [p for p in self._parts if p is not part]
        note = "covers the ros2_control arm only; unacknowledged parts: " + ", ".join(
            type(p).__name__ for p in others
        )
        return dataclasses.replace(report, detail=f"{report.detail}; {note}".lstrip("; "))
