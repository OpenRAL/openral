"""Real-hardware HAL for the Unitree hg humanoid family (G1 now; H2 shares the stack).

S0 is NVIDIA's, not OpenRAL's. On real hardware the legs are commanded by a whole-body
ONNX policy running *inside* ``ros2_control`` (NVIDIA ``isaac_ros_deploy``'s
``InferenceController``), behind NVIDIA's chainable ``SafetyController``, on NVIDIA's
``unitree_hg_ros2_control`` hardware interface (``isaac_ros_robots``) — all C++, all in
the 200 Hz controller-manager loop. This adapter never commands a leg. It proposes; the
C++ graph disposes (CLAUDE.md §3 "Python proposes; C++ disposes"):

    Skill → Action → OpenRAL safety kernel → UnitreeHgRealHAL
        JOINT_POSITION (upper body only) ─► /upper_body_forward_joint_command_controller/
                                             joint_commands_trajectory
                                             (isaac_ros_deploy_interfaces/JointCommandTrajectory)
        BODY_TWIST                       ─► /inference_controller/cmd_vel (geometry_msgs/Twist)
                                                │ both write into safety_controller's
                                                ▼ `<joint>/<if>_raw` reference interfaces
        safety_controller (blend / rate clamp / out-of-domain → freeze_controller)
                                                ▼
        UnitreeHgSystemInterface (CRC, 500 ms state freshness) → unitree_sdk2 → motors

The bringup that starts that graph is ``openral_hal_unitree``'s ``real_bringup.launch.py``
(``hal.real_bringup`` in ``robots/g1/robot.yaml``).

**Joint ownership comes from the selected S0 policy.** ``s0_policy`` (manifest
``hal.parameters.defaults.s0_policy``; GR00T Decoupled WBC for the G1) names the policy
file the launch also configures the C++ graph from (``openral_hal.unitree_s0_policy``).
The policy owns its ``s0_joints`` (GR00T: 12 legs + all three waist axes; AGILE: 12 legs
+ waist roll/pitch); this HAL owns the upper-body controller's ``joints`` (GR00T: both
7-DoF arms; AGILE: + waist yaw). The two must be a disjoint cover of the manifest. A
``JOINT_POSITION`` action that names, or — as a whole-vector action with no
``Action.joint_names`` — implies, any S0 joint is refused with ``ROSCapabilityMismatch``
before anything is published: on this robot a leg target from Python is a capability the
real stack does not offer, not a value to drop silently.

**Every upper-body message is whole.** NVIDIA's ``ForwardJointCommandController`` holds
the pose captured *at activation* for any joint a message omits. A command for one arm
would therefore snap the other back to its activation pose, so every message carries all
upper-body joints: the ones an action does not own keep their last commanded target (or
their measured position, before the first command).

**The velocity command is a heartbeat.** The policy's ``topic_input_timeout_ms`` (500 ms
in each ``policies/*.yaml``) blanks its output when ``cmd_vel`` goes stale, and
``SafetyController`` then holds the measured pose — what stops the robot walking on its
last command if this process dies. So every ``send_action`` publishes a twist: the
commanded one for ``BODY_TWIST``, zero otherwise (the G1 twin's own rule: a non-twist
action zeroes the base twist). Inside an ADR-0102 slot group, a sibling ``BODY_TWIST`` of
the same tick is re-published instead of zero.

**Base height is not commandable.** ``Action`` has no base-height field, and upstream's
only height-carrying converter (``command/body/velocity_height`` at isaac_ros_deploy
f3aa6b89) hardcodes 0.7 m — there is nothing to forward a height to.

**E-stop.** ``UnitreeHgSystemInterface`` sends zero torque (motor mode 0) for any joint no
controller claims, so the base ``RosControlHAL`` stop — deactivate the controllers — would
drop a standing humanoid. ``estop`` instead swaps the S0 chain for ``freeze_controller`` in
**one** ``switch_controller`` request (``ControllerSwapSeam``), the same hand-over NVIDIA's
``SafetyController`` performs on an out-of-domain detection. ``freeze_controller`` holds
the measured pose with low gains; it does not balance. ``estop_recovery`` is
``RESTART_REQUIRED``: re-arming a walking policy on a robot that may have fallen is an
operator decision.

Example:
    >>> from openral_hal.unitree_real import UnitreeHgRealHAL
    >>> hal = UnitreeHgRealHAL()
    >>> hal.controller_names()
    ['safety_controller', 'inference_controller', 'upper_body_forward_joint_command_controller']
    >>> len(hal.upper_body_joints()), len(hal.s0_joints())
    (14, 15)
    >>> "waist_yaw_joint" in hal.s0_joints()  # GR00T WBC owns the whole waist
    True
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from openral_core import BODY_TWIST_DIM
from openral_core.exceptions import (
    ROSCapabilityMismatch,
    ROSConfigError,
    ROSRuntimeError,
)
from openral_core.schemas import Action, ControlMode, RobotDescription

from openral_hal._real_description import make_real_description
from openral_hal.g1 import G1_DESCRIPTION
from openral_hal.protocol import EStopRecovery, HALHealthReport
from openral_hal.ros_control import (
    ControllerKind,
    ControllerStopSeam,
    ControllerSwapSeam,
    ControllerSwitchReport,
    RosControlHAL,
)
from openral_hal.unitree_s0_policy import load_s0_policy, s0_policy_path

__all__ = [
    "G1_REAL_DESCRIPTION",
    "UnitreeHgRealHAL",
]

log = structlog.get_logger(__name__)


# ── Real-HW RobotDescription ──────────────────────────────────────────────────
# `closed_with_api`: NVIDIA's isaac_ros_robots / isaac_ros_deploy are Apache-2.0, but
# `unitree_hg_ros2_control` links unitree_sdk2's prebuilt `libunitree_sdk2.a` (BSD-3
# wrapper, closed core) — the posture robots/g1/robot.yaml already documented for any
# real G1 path.
G1_REAL_DESCRIPTION = make_real_description(G1_DESCRIPTION, sdk_kind="closed_with_api")

# Controller names from packages/openral_hal_unitree/config/<model>/controller_manager.yaml.
_SAFETY_CONTROLLER = "safety_controller"
_INFERENCE_CONTROLLER = "inference_controller"
_UPPER_BODY_CONTROLLER = "upper_body_forward_joint_command_controller"
_FREEZE_CONTROLLER = "freeze_controller"

#: Budget for the e-stop's pre-swap listing: it must never hold the stop back long. A
#: listing that does not answer in time goes straight to the swap.
_PRECHECK_TIMEOUT_S = 0.2


def _with_notes(report: ControllerSwitchReport, notes: list[str]) -> ControllerSwitchReport:
    """``report`` with ``notes`` prepended to its detail."""
    if not notes:
        return report
    return ControllerSwitchReport(
        ok=report.ok,
        controllers=report.controllers,
        states=report.states,
        detail="; ".join([*notes, report.detail]),
    )


#: `joint_command_trajectory_topic: "~/joint_commands_trajectory"` in that file.
_UPPER_BODY_TOPIC_SUFFIX = "joint_commands_trajectory"

# Twist components the S0 velocity policy does not take from OpenRAL (linear-z,
# angular-x, angular-y) — refused when non-zero, as the G1 twin refuses them.
_PLANAR_TWIST_EPS = 1e-9


@dataclass(frozen=True)
class _UnitreeHgModel:
    """Per-model defaults of the hg family; one row per robot the bringup supports."""

    description: RobotDescription
    default_s0_policy: str


#: Models the bringup ships a config/<model>/ for. H2 joins here with its own manifest.
_MODELS: dict[str, _UnitreeHgModel] = {
    "g1": _UnitreeHgModel(
        description=G1_REAL_DESCRIPTION,
        # The G1 S0 bake-off winner (robustness); see policies/gr00t_decoupled_wbc.yaml.
        default_s0_policy="gr00t_decoupled_wbc",
    ),
}


class UnitreeHgRealHAL(RosControlHAL):
    """ros2_control HAL for a real Unitree hg humanoid whose legs belong to an S0 policy.

    Args:
        description: Manifest to publish. Defaults to the model's real description
            (``G1_REAL_DESCRIPTION``); ``build_hal`` passes the loaded
            ``robots/<id>/robot.yaml``.
        robot_model: Which hg-family model's defaults apply (``"g1"``).
        s0_policy: Stem of ``config/<robot_model>/policies/<stem>.yaml``; the joint split
            and the velocity-command topic come from it. ``None`` takes the model's
            default (``gr00t_decoupled_wbc`` for the G1). Must be the policy the bringup
            launch runs (its ``policy:=``).
        upper_body_controller: The ``ForwardJointCommandController`` commanding
            ``upper_body_joints``.
        inference_controller: The S0 policy controller.
        safety_controller: The chainable safety controller both write through.
        freeze_controller: The holding controller ``estop`` hands every joint to.
        joint_state_topic: Topic publishing ``sensor_msgs/JointState``.
        publish_fn: Transport callable (the lifecycle node attaches the real one).
        state_fn: Latest raw joint state as a dict.
        staleness_limit_s: ``read_state`` age limit; ``None`` reads the manifest's
            ``safety.joint_state_staleness_limit_s``.
        stop_timeout_s: Bound on the e-stop's controller swap.

    Raises:
        ROSConfigError: Unknown ``robot_model``; a missing, malformed or placeholder
            policy file; a policy whose ``s0_joints`` and upper-body joints are not a
            disjoint cover of the manifest's joints.

    Example:
        >>> from openral_hal.unitree_real import UnitreeHgRealHAL
        >>> hal = UnitreeHgRealHAL()
        >>> sorted(k.value for k in hal.command_bindings().values())
        ['joint_command_trajectory', 'twist']
    """

    #: Re-arming a walking policy after a freeze is an operator decision (module docstring).
    estop_recovery: EStopRecovery = EStopRecovery.RESTART_REQUIRED

    def __init__(
        self,
        description: RobotDescription | None = None,
        *,
        robot_model: str = "g1",
        s0_policy: str | None = None,
        upper_body_controller: str = _UPPER_BODY_CONTROLLER,
        inference_controller: str = _INFERENCE_CONTROLLER,
        safety_controller: str = _SAFETY_CONTROLLER,
        freeze_controller: str = _FREEZE_CONTROLLER,
        joint_state_topic: str = "/joint_states",
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float | None = None,
        stop_timeout_s: float = 5.0,
    ) -> None:
        """Initialise the adapter; opens no ROS connection yet."""
        model = _MODELS.get(robot_model)
        if model is None:
            raise ROSConfigError(
                f"UnitreeHgRealHAL: no hg-family model {robot_model!r}; supported: "
                f"{sorted(_MODELS)}. A new model needs its config/<model>/ in "
                "openral_hal_unitree and a row in openral_hal.unitree_real._MODELS."
            )
        desc = description or model.description
        upper_body_topic = f"/{upper_body_controller}/{_UPPER_BODY_TOPIC_SUFFIX}"
        super().__init__(
            desc,
            controller_name=upper_body_controller,
            joint_state_topic=joint_state_topic,
            command_topic=upper_body_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
            stop_timeout_s=stop_timeout_s,
        )
        policy_name = s0_policy or model.default_s0_policy
        policy = load_s0_policy(s0_policy_path(robot_model, policy_name))
        names = [j.name for j in desc.joints]
        upper = tuple(policy.upper_body_joints)
        s0 = tuple(policy.openral_policy.s0_joints)
        unknown = [n for n in (*upper, *s0) if n not in names]
        overlap = sorted(set(upper) & set(s0))
        uncovered = [n for n in names if n not in upper and n not in s0]
        if unknown or overlap or uncovered or len(set(upper)) != len(upper):
            raise ROSConfigError(
                f"UnitreeHgRealHAL: S0 policy {policy_name!r} does not split "
                f"'{desc.name}'s joints cleanly — unknown {unknown}, owned twice {overlap}, "
                f"owned by nobody {uncovered}. s0_joints and the upper-body controller's "
                "joints must be a disjoint cover of the manifest."
            )
        if not s0:
            raise ROSConfigError(
                f"UnitreeHgRealHAL: S0 policy {policy_name!r} owns no joint — the legs must "
                "belong to S0."
            )
        self._s0_policy = policy_name
        self._robot_model = robot_model
        self._upper: tuple[str, ...] = upper
        self._s0: frozenset[str] = frozenset(s0)
        self._index = {n: i for i, n in enumerate(names)}
        self._upper_body_topic = upper_body_topic
        self._velocity_topic = policy.openral_policy.velocity_command_topic
        self._upper_body_controller = upper_body_controller
        self._inference_controller = inference_controller
        self._safety_controller = safety_controller
        self._freeze_controller = freeze_controller
        # Last commanded target per upper-body joint (see "Every upper-body message is whole").
        self._held: dict[str, float] = {}
        # (tick_index, twist) of the last BODY_TWIST, re-published by same-tick siblings.
        self._twist_tick: tuple[int, tuple[float, ...]] | None = None

    # ── Introspection ─────────────────────────────────────────────────────────

    def upper_body_joints(self) -> list[str]:
        """Joints this HAL commands, in the order every upper-body message carries them.

        Example:
            >>> from openral_hal.unitree_real import UnitreeHgRealHAL
            >>> UnitreeHgRealHAL().upper_body_joints()[0]
            'left_shoulder_pitch_joint'
        """
        return list(self._upper)

    def s0_joints(self) -> list[str]:
        """Joints the S0 policy owns (manifest order); a ``JOINT_POSITION`` on one is refused.

        Example:
            >>> from openral_hal.unitree_real import UnitreeHgRealHAL
            >>> "left_knee_joint" in UnitreeHgRealHAL().s0_joints()
            True
        """
        return [j.name for j in self.description.joints if j.name in self._s0]

    def command_bindings(self) -> dict[str, ControllerKind]:
        """The upper-body trajectory topic and the S0 velocity-command topic, with kinds.

        Example:
            >>> from openral_hal.unitree_real import UnitreeHgRealHAL
            >>> list(UnitreeHgRealHAL().command_bindings())[1]
            '/inference_controller/cmd_vel'
        """
        return {
            self._upper_body_topic: ControllerKind.JOINT_COMMAND_TRAJECTORY,
            self._velocity_topic: ControllerKind.TWIST,
        }

    def controller_names(self) -> list[str]:
        """The S0 chain ``estop`` swaps out for ``freeze_controller``.

        Example:
            >>> from openral_hal.unitree_real import UnitreeHgRealHAL
            >>> UnitreeHgRealHAL().controller_names()[0]
            'safety_controller'
        """
        return [
            self._safety_controller,
            self._inference_controller,
            self._upper_body_controller,
        ]

    @property
    def s0_policy(self) -> str:
        """Stem of the S0 policy file the joint split was read from."""
        return self._s0_policy

    @property
    def freeze_controller(self) -> str:
        """The holding controller ``estop`` activates in the same switch."""
        return self._freeze_controller

    # ── Stop seam ─────────────────────────────────────────────────────────────

    def attach_controller_stop(self, seam: ControllerStopSeam) -> None:
        """Bind the stop seam, refusing one that cannot swap in a single request.

        Raises:
            ROSConfigError: ``seam`` is not a ``ControllerStopSeam``, or cannot
                ``swap_controllers`` — plain deactivation would release every joint of a
                standing humanoid (see module docstring), so such a seam fails at wire-up,
                never at e-stop time.
        """
        super().attach_controller_stop(seam)
        if not isinstance(seam, ControllerSwapSeam):
            self._stop_seam = None
            raise ROSConfigError(
                f"UnitreeHgRealHAL needs a stop seam with swap_controllers(); "
                f"{type(seam).__name__} can only deactivate, which would leave every joint "
                "unclaimed (zero torque) on a standing humanoid."
            )

    def _stop_switch(
        self, seam: ControllerStopSeam, names: tuple[str, ...]
    ) -> ControllerSwitchReport:
        """Swap the S0 chain for ``freeze_controller``: pre-check, one swap, retry (#566).

        ``freeze_controller`` cannot be activated *before* the chain lets go — the two
        claim the same command interfaces and ``controller_manager`` refuses the
        activation — so the single swap request is the tightest hand-over. Around it:

        * **pre-check** — the swap is BEST_EFFORT and the manager drops the activation
          of a controller that is not ``inactive`` while still applying the
          deactivations, so an ``unconfigured`` freeze controller is configured first.
          Bounded by ``_PRECHECK_TIMEOUT_S`` so it never holds the stop back long; an
          unanswered listing goes straight to the swap.
        * **retry** — if the swap did not confirm ``freeze_controller`` active (its
          ``on_activate`` failed, or it was dropped), the chain is already out, so it
          is activated alone at once; nothing conflicts any more.
        """
        if not isinstance(seam, ControllerSwapSeam):  # attach_controller_stop refuses these
            raise ROSConfigError(f"{type(seam).__name__} cannot swap controllers")
        freeze = self._freeze_controller
        timeout = self._stop_timeout_s
        notes: list[str] = []
        before = seam.controller_states(timeout_s=min(timeout, _PRECHECK_TIMEOUT_S)).get(freeze)
        if before == "unconfigured":
            configured = seam.configure_controller(freeze, timeout_s=timeout)
            log.critical(
                "hal.estop.freeze_configured",
                robot=self.description.name,
                controller=freeze,
                ok=configured,
            )
            notes.append(f"{freeze} was unconfigured; configure {'ok' if configured else 'FAILED'}")
        swap = seam.swap_controllers(deactivate=names, activate=(freeze,), timeout_s=timeout)
        if swap.ok or swap.states.get(freeze) == "active":
            return _with_notes(swap, notes)
        log.critical(
            "hal.estop.freeze_retry",
            robot=self.description.name,
            controller=freeze,
            swap_detail=swap.detail,
        )
        retry = seam.activate_controllers((freeze,), timeout_s=timeout)
        states = {**swap.states, **retry.states}
        held = retry.ok and all(states.get(n) == "inactive" for n in names)
        notes.append(f"swap: {swap.detail}; {freeze} activated alone: {retry.detail}")
        return ControllerSwitchReport(
            ok=held, controllers=swap.controllers, states=states, detail="; ".join(notes)
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def disconnect(self) -> None:
        """Close the connection and forget held upper-body targets."""
        self._held.clear()
        self._twist_tick = None
        super().disconnect()

    def estop(self) -> None:
        """Forget held targets, then swap the S0 chain for ``freeze_controller``.

        Raises:
            ROSEStopRequested: Always, from the base implementation.
        """
        self._held.clear()
        self._twist_tick = None
        super().estop()

    # ── Hot path ──────────────────────────────────────────────────────────────

    def send_action(self, action: Action) -> None:
        """Publish an upper-body chunk and/or the S0 velocity command.

        ``JOINT_POSITION``: ``joint_targets`` rows are full-dof in manifest order (the
        ADR-0102 convention); ``Action.joint_names`` says which joints the action owns,
        ``None`` meaning all of them. Owning any S0 joint is refused. The message carries
        every upper-body joint (unowned ones at their held target), then the twist
        heartbeat follows. ``BODY_TWIST``: row 0 of ``body_twist`` (the G1 twin's rule)
        is published on the velocity topic; arms keep their last trajectory.

        Raises:
            ROSRuntimeError: Not connected, or no measured pose yet for an upper-body
                joint that has never been commanded.
            ROSConfigError: Unsupported control mode, wrong dimensions, a non-planar or
                non-finite twist.
            ROSCapabilityMismatch: The action owns an S0 (leg/waist-roll/pitch) joint.
        """
        self._require_connected("send_action")
        self._validate_action(action)
        if action.control_mode == ControlMode.BODY_TWIST.value:
            twist = self._twist_of(action)
            self._twist_tick = (int(action.tick_index), twist)
            self._publish_fn(self._velocity_topic, {"twist": list(twist)})
            log.debug("hal.send_action", robot=self.description.name, body_twist=twist)
            return

        upper_msg = self._upper_body_message(action)
        if upper_msg is not None:
            self._publish_fn(self._upper_body_topic, upper_msg)
        self._publish_fn(self._velocity_topic, {"twist": list(self._heartbeat_twist(action))})
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _validate_action(self, action: Action) -> None:
        supported = self.description.capabilities.supported_control_modes
        if supported and action.control_mode not in supported:
            raise ROSConfigError(
                f"Action control_mode '{action.control_mode}' is not in "
                f"supported_control_modes {supported} for robot '{self.description.name}'."
            )
        if action.control_mode != ControlMode.BODY_TWIST.value:
            self._validate_action_dims(action, len(self._joint_names))

    def _twist_of(self, action: Action) -> tuple[float, ...]:
        if not action.body_twist:
            raise ROSConfigError("UnitreeHgRealHAL: empty Action.body_twist for BODY_TWIST.")
        row = tuple(float(v) for v in action.body_twist[0])
        if len(row) != BODY_TWIST_DIM:
            raise ROSConfigError(
                f"UnitreeHgRealHAL: BODY_TWIST expects {BODY_TWIST_DIM} floats "
                f"(vx, vy, vz, wx, wy, wz); got {len(row)}."
            )
        if not all(math.isfinite(v) for v in row):
            raise ROSConfigError(f"UnitreeHgRealHAL: non-finite BODY_TWIST {row}.")
        if any(abs(row[i]) > _PLANAR_TWIST_EPS for i in (2, 3, 4)):
            raise ROSConfigError(
                "UnitreeHgRealHAL: BODY_TWIST carries non-zero linear-z / angular-x / "
                "angular-y; the S0 velocity policy takes only vx, vy, wz."
            )
        return (row[0], row[1], 0.0, 0.0, 0.0, row[5])

    def _heartbeat_twist(self, action: Action) -> tuple[float, ...]:
        """Zero, unless a BODY_TWIST sibling of this action's slot group already set one."""
        held = self._twist_tick
        if held is not None and int(action.tick_group_size) > 1:
            tick, twist = held
            if tick == int(action.tick_index):
                return twist
        return (0.0,) * BODY_TWIST_DIM

    def _upper_body_message(self, action: Action) -> dict[str, object] | None:
        if action.joint_targets is None:
            return None
        owned = list(action.joint_names) if action.joint_names is not None else self._joint_names
        legs = [n for n in owned if n in self._s0]
        if legs:
            raise ROSCapabilityMismatch(
                f"UnitreeHgRealHAL: the action targets {legs}, which the S0 whole-body "
                f"policy owns on the real {self.description.name}. Python commands only "
                f"{list(self._upper)}; set Action.joint_names to those (a whole-vector "
                "action with joint_names=None claims every joint, legs included)."
            )
        unknown = [n for n in owned if n not in self._index]
        if unknown:
            raise ROSConfigError(f"UnitreeHgRealHAL: unknown joint name(s) {unknown}.")
        owned_set = set(owned)
        base = self._held_or_measured([n for n in self._upper if n not in owned_set])
        rows: list[list[float]] = []
        for step in action.joint_targets:
            row = [float(step[self._index[n]]) if n in owned_set else base[n] for n in self._upper]
            if not all(math.isfinite(v) for v in row):
                raise ROSConfigError(f"UnitreeHgRealHAL: non-finite upper-body target {row}.")
            rows.append(row)
        self._held.update(zip(self._upper, rows[-1], strict=True))
        return {
            "control_mode": action.control_mode,
            "horizon": action.horizon,
            "joint_names": list(self._upper),
            "joint_targets": rows,
            "stamp_ns": action.stamp_ns,
            "time_from_start_s": self.time_from_start_s(action),
        }

    def _held_or_measured(self, names: list[str]) -> dict[str, float]:
        """Held target per joint, falling back to its measured position before any command."""
        out = {n: self._held[n] for n in names if n in self._held}
        missing = [n for n in names if n not in out]
        if not missing:
            return out
        raw = self._state_fn() if self._state_fn is not None else {}
        state_names = raw.get("name")
        positions = raw.get("position")
        if isinstance(state_names, list) and isinstance(positions, list):
            measured = dict(zip((str(n) for n in state_names), positions, strict=False))
        elif isinstance(positions, list) and len(positions) == len(self._joint_names):
            measured = dict(zip(self._joint_names, positions, strict=True))
        else:
            measured = {}
        unheard = [n for n in missing if n not in measured]
        if unheard:
            raise ROSRuntimeError(
                f"UnitreeHgRealHAL: no measured position for {unheard} yet; an upper-body "
                "command must carry every upper-body joint, and these have neither a held "
                "target nor a reading to hold."
            )
        out.update({n: float(measured[n]) for n in missing})
        return out

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def health(self) -> HALHealthReport:
        """Cached wiring summary; performs no I/O.

        Example:
            >>> from openral_hal.unitree_real import UnitreeHgRealHAL
            >>> UnitreeHgRealHAL().health().fields["robot_model"]
            'g1'
        """
        state = "connected" if self._connected else "disconnected"
        return HALHealthReport(
            message=f"Unitree {self._robot_model} ({self.description.name}) — {state}",
            fields={
                "connected": str(self._connected),
                "robot_model": self._robot_model,
                "s0_policy": self._s0_policy,
                "upper_body_topic": self._upper_body_topic,
                "velocity_command_topic": self._velocity_topic,
                "estop_holding_controller": self._freeze_controller,
            },
        )
