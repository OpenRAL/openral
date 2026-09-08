"""Production ``rclpy`` transport for every ``RosControlHAL`` robot.

`RosControlHAL` is deliberately wire-format-free: it turns an `Action` into a
plain dict and hands it to an injected `publish_fn`. Something has to be that
callable on real hardware. Until this module existed nothing was — the default
was `_default_publish`, a `log.debug` no-op — so `openral deploy run` against a
real ros2_control arm sent commands into a logger and read back zeros, for every
robot in that family (OpenArm, UR5e, UR10e, Franka, Sawyer). Only the serial
arms (SO-100/SO-101, whose HAL owns its own bus) actually moved, which is why
the gap went unnoticed.

This transport is robot-agnostic on purpose. It is built entirely from what the
HAL already reports about itself:

* `command_bindings()` — one publisher per entry, typed by the entry's
  `ControllerKind`, so a single-controller UR and the four-controller bimanual
  OpenArm are the same code path.
* `ros2_control_joint_names()` — the names `/joint_states` is keyed by.
* `joint_state_topic` — the aggregated state topic.

Adding a robot therefore needs no transport code at all: give its HAL those
three (the base class already answers all of them) and the lifecycle node wires
this automatically. A robot whose controllers speak a format not yet in
`ControllerKind` is the one case that needs a change here — and it fails loudly
at wire-up rather than publishing a type DDS will silently drop.

Joint state is merged **by name, never by index**. Independently publishing
controllers give no ordering guarantee, and on a robot whose controllers are
split per limb each message carries only that limb's joints.

Arrival time is tracked per message and surfaced via `last_arrival`, which is
what lets `RosControlHAL.read_state` measure real staleness instead of time
since `connect()`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from openral_hal.ros_control import ControllerKind

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from rclpy.node import Node

__all__ = ["RosControlDrivable", "RosControlTransport"]


@runtime_checkable
class RosControlDrivable(Protocol):
    """What a HAL must expose to be driven by `RosControlTransport`.

    Membership is **structural, not by ancestry**, so a HAL that reimplements
    the ros2_control fan-out on `HALBase` rather than inheriting still gets
    wired. Gating the lifecycle node on `isinstance(..., RosControlHAL)` would
    leave such a robot with no transport and no error — the failure this whole
    surface exists to prevent.

    A HAL opts in by answering these; the base class answers all four, so for
    most robots that is automatic.
    """

    def command_bindings(self) -> dict[str, ControllerKind]:
        """Controller topics this HAL publishes to, each with its wire format."""
        ...

    def ros2_control_joint_names(self) -> list[str]:
        """Joint names in ros2_control's namespace, in action-vector order."""
        ...

    @property
    def joint_state_topic(self) -> str:
        """Aggregated ``sensor_msgs/JointState`` topic to read."""
        ...

    def attach_transport(
        self,
        publish_fn: Callable[[str, dict[str, object]], None],
        state_fn: Callable[[], dict[str, object]],
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind the live transport's callables to this HAL."""
        ...


log = structlog.get_logger(__name__)

#: Command QoS per CLAUDE.md §2: control is RELIABLE, VOLATILE, shallow. A
#: trajectory that arrives late is worse than one that is dropped and resent.
_COMMAND_DEPTH = 1

#: State QoS. `joint_state_broadcaster` publishes RELIABLE, and a BEST_EFFORT
#: subscriber is compatible with a RELIABLE publisher, so this reads a live
#: broadcaster while still tolerating a lossy link.
_STATE_DEPTH = 10


class RosControlTransport:
    """Bridges one `RosControlHAL` to live ros2_control topics.

    Args:
        node: A live `rclpy` node the publishers/subscription are created on.
            Ownership stays with the caller (the lifecycle node).
        command_topics: Controller topics, from `hal.command_topics()`.
        joint_names: ros2_control joint names, from
            `hal.ros2_control_joint_names()`.
        joint_state_topic: Aggregated state topic, from `hal.joint_state_topic`.
        command_kinds: Wire format per topic, from `hal.command_bindings()`.
            A topic absent from the mapping defaults to
            `ControllerKind.JOINT_TRAJECTORY`, which is what every
            ros2_control robot in this repo runs.

    Raises:
        ROSConfigError: If no command topic or no joint name was given — a HAL
            that reports neither cannot be driven, and failing here is far
            cheaper than a silent no-op on hardware.
    """

    def __init__(
        self,
        node: Node,
        *,
        command_topics: list[str],
        joint_names: list[str],
        joint_state_topic: str = "/joint_states",
        command_kinds: dict[str, ControllerKind] | None = None,
    ) -> None:
        """Create one publisher per command topic and subscribe to the state topic."""
        # `openral_hal` must import without a ROS 2 install (unit tests, docs
        # builds, CI lanes with no rclpy), so every ROS symbol below is imported
        # at call time rather than at module scope — hence the PLC0415 waivers.
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        if not command_topics:
            raise ROSConfigError(
                "RosControlTransport needs at least one command topic; the HAL "
                "reported none, so nothing it publishes could reach a controller."
            )
        if not joint_names:
            raise ROSConfigError(
                "RosControlTransport needs the ros2_control joint names; the HAL "
                "reported none, so incoming /joint_states could not be matched."
            )

        kinds = dict(command_kinds or {})
        unknown = sorted(set(kinds) - set(command_topics))
        if unknown:
            raise ROSConfigError(
                f"RosControlTransport was given controller kinds for topic(s) {unknown} "
                f"that are not in command_topics {sorted(command_topics)}. A kind that "
                "names no topic silently governs nothing, which is how a controller ends "
                "up commanded with the wrong message type."
            )
        self._kinds = {t: kinds.get(t, ControllerKind.JOINT_TRAJECTORY) for t in command_topics}

        from rclpy.qos import (  # noqa: PLC0415  # reason: rclpy is a ROS-only dep; the HAL package must import without it
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import JointState as RosJointState  # noqa: PLC0415

        self._node = node
        self._joint_names = list(joint_names)
        self._joint_state_topic = joint_state_topic
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_arrival: float = 0.0

        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_COMMAND_DEPTH,
        )
        # One publisher per topic, typed by the kind the HAL declared. Getting
        # this wrong is invisible at runtime — DDS drops a mismatched type
        # without delivering it and without erroring — so the type is chosen
        # here, once, from the declaration rather than assumed at publish time.
        self._pubs = {
            topic: node.create_publisher(_message_type(self._kinds[topic]), topic, command_qos)
            for topic in command_topics
        }

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_STATE_DEPTH,
        )
        self._sub = node.create_subscription(
            RosJointState, joint_state_topic, self._on_joint_state, state_qos
        )
        log.info(
            "hal.transport.ready",
            command_topics=list(self._pubs),
            joint_state_topic=joint_state_topic,
            joints=len(self._joint_names),
        )

    # ── Injected callables ────────────────────────────────────────────────────

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Send one HAL command dict to its controller in that controller's format.

        The message type follows the `ControllerKind` the HAL declared for this
        topic in `command_bindings` — `trajectory_msgs/JointTrajectory` for a
        `JointTrajectoryController`, `std_msgs/Float64MultiArray` for the
        forward-command family.

        The message's own `joint_names` are used verbatim, so this cannot
        misroute a value the HAL placed correctly; it falls back to the full
        joint list only for a HAL that omits them.

        Only the chunk's **last** step is published. A `JointTrajectory` point
        carries an absolute target plus a deadline, and the controller
        interpolates from where the robot actually is — replaying every
        intermediate step of an action chunk would fight that interpolation.

        Args:
            topic: One of `command_topics`.
            msg: The HAL's command dict.

        Raises:
            ROSConfigError: The topic is not one this transport owns, or the
                payload is a shape this transport cannot express — both are
                contract drift between HAL and transport, which must be loud.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        publisher = self._pubs.get(topic)
        if publisher is None:
            raise ROSConfigError(
                f"RosControlTransport has no publisher for {topic!r}; the HAL "
                f"publishes to topics it never declared in command_topics(). "
                f"Declared: {sorted(self._pubs)}"
            )

        # A payload this transport cannot express must be loud. Returning
        # quietly would drop a command on the actuation path with nothing above
        # debug in the logs — the exact silent no-op this transport exists to
        # end. A HAL whose controllers take something else (a scalar gripper
        # position, say) needs that message type implemented here, not skipped
        # at runtime.
        if "joint_targets" not in msg:
            raise ROSConfigError(
                f"RosControlTransport cannot express the command {sorted(msg)} sent to "
                f"{topic!r}: every ControllerKind it publishes is commanded from "
                f"'joint_targets', and this one ({self._kinds[topic].value}) is no "
                "exception. Add the kind that controller speaks to ControllerKind rather "
                "than letting the command be dropped."
            )
        targets = msg.get("joint_targets")
        if not isinstance(targets, list) or not targets:
            # Distinct from the above: the HAL had nothing to send this tick.
            log.debug("hal.transport.empty_command", topic=topic)
            return
        raw_names = msg.get("joint_names")
        names = (
            [str(n) for n in raw_names]
            if isinstance(raw_names, list) and raw_names
            else list(self._joint_names)
        )
        last_step = targets[-1]
        if not isinstance(last_step, list) or len(last_step) != len(names):
            raise ROSConfigError(
                f"RosControlTransport: {topic!r} names {len(names)} joint(s) but the "
                f"chunk's final step has "
                f"{len(last_step) if isinstance(last_step, list) else 'a non-list'}."
            )

        kind = self._kinds[topic]
        if kind is ControllerKind.FORWARD_COMMAND:
            from std_msgs.msg import Float64MultiArray  # noqa: PLC0415  # reason: ROS-only dep

            # A forward controller takes a bare vector in the order its own
            # `joints` parameter declares — there is no room in the message for
            # names, so the HAL's ordering is the whole contract. The length
            # check above is therefore the only guard available.
            command = Float64MultiArray()
            command.data = [float(v) for v in last_step]
            publisher.publish(command)
            return

        from trajectory_msgs.msg import (  # noqa: PLC0415  # reason: ROS-only dep
            JointTrajectory,
            JointTrajectoryPoint,
        )

        traj = JointTrajectory()
        traj.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in last_step]
        seconds = self._time_from_start_s(msg)
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
        traj.points.append(point)
        publisher.publish(traj)

    def state(self) -> dict[str, object]:
        """Return the newest joint state, projected onto the HAL's joint order.

        Includes `name` so a HAL that reorders by name (OpenArm) sees what
        actually arrived rather than an assumed layout.
        """
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self._joint_names:
            p, v, e = self._latest.get(name, (0.0, 0.0, 0.0))
            positions.append(p)
            velocities.append(v)
            efforts.append(e)
        return {
            "name": list(self._joint_names),
            "position": positions,
            "velocity": velocities,
            "effort": efforts,
        }

    def last_arrival(self) -> float:
        """`time.monotonic()` of the newest joint state; 0.0 if none has arrived."""
        return self._last_arrival

    # ── Introspection ─────────────────────────────────────────────────────────

    def seen_joints(self) -> set[str]:
        """Joint names that have actually appeared on the state topic."""
        return set(self._latest)

    def missing_joints(self) -> list[str]:
        """Expected joints the state topic has never reported."""
        return [n for n in self._joint_names if n not in self._latest]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _time_from_start_s(self, msg: dict[str, object]) -> float:
        """Deadline for the published point.

        Taken from the action's own `chunk_dt * horizon` when the HAL supplies
        it, because the controller's speed is `(target - measured) / deadline`
        — a deadline shorter than the chunk it represents asks the arm to cover
        the whole chunk in one step.
        """
        raw = msg.get("time_from_start_s")
        if isinstance(raw, int | float) and float(raw) > 0.0:
            return float(raw)
        return _DEFAULT_TIME_FROM_START_S

    def _on_joint_state(self, msg: Any) -> None:  # noqa: ANN401  # reason: duck-typed sensor_msgs/JointState keeps this module ROS-free at import
        """Merge one `sensor_msgs/JointState` into the cache, by name."""
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))
        for i, name in enumerate(names):
            self._latest[str(name)] = (
                positions[i] if i < len(positions) else 0.0,
                velocities[i] if i < len(velocities) else 0.0,
                efforts[i] if i < len(efforts) else 0.0,
            )
        self._last_arrival = time.monotonic()


#: Default trajectory deadline. Matches the 100 ms the production HALs assume
#: for a single-step command; overridden per message via `time_from_start_s`.
_DEFAULT_TIME_FROM_START_S = 0.1


def _message_type(kind: ControllerKind) -> type:
    """Return the ROS message class one `ControllerKind` is commanded with.

    Imported at call time so this module still imports without a ROS 2
    installation (unit tests, docs builds, CI lanes with no rclpy).

    Raises:
        ROSConfigError: For a kind with no message mapping. Unreachable while
            the enum and this function agree; it exists so that adding a member
            without teaching this function fails at wire-up rather than
            publishing a plausible-but-wrong type onto the actuation path.
    """
    from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

    if kind is ControllerKind.JOINT_TRAJECTORY:
        from trajectory_msgs.msg import JointTrajectory  # noqa: PLC0415  # reason: ROS-only dep

        return JointTrajectory
    if kind is ControllerKind.FORWARD_COMMAND:
        from std_msgs.msg import Float64MultiArray  # noqa: PLC0415  # reason: ROS-only dep

        return Float64MultiArray
    raise ROSConfigError(  # pragma: no cover - guarded by test_every_controller_kind_maps
        f"RosControlTransport has no message type for ControllerKind {kind!r}. "
        "Add one here in the same change that adds the enum member."
    )
