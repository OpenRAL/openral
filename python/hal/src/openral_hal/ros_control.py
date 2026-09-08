"""RosControlHAL — ros2_control-backed Hardware Abstraction Layer adapter.

This adapter forwards ``Action`` chunks to a ``ros2_control`` joint trajectory
controller and reads ``JointState`` from the controller's state topic.  In
unit tests the underlying transport is replaced by a mock; the adapter itself
contains no ROS 2 imports so it can be tested without a live ROS 2 installation.

Example:
    >>> from openral_hal.ros_control import RosControlHAL
    >>> from openral_core import (
    ...     RobotDescription,
    ...     EmbodimentKind,
    ...     JointSpec,
    ...     JointType,
    ...     RobotCapabilities,
    ...     SafetyEnvelope,
    ...     ControlMode,
    ... )
    >>> desc = RobotDescription(
    ...     name="test_robot",
    ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
    ...     joints=[
    ...         JointSpec(
    ...             name="j1", joint_type=JointType.REVOLUTE, parent_link="base", child_link="link1"
    ...         ),
    ...     ],
    ...     capabilities=RobotCapabilities(
    ...         supported_control_modes=[ControlMode.JOINT_POSITION],
    ...     ),
    ...     safety=SafetyEnvelope(),
    ... )
    >>> hal = RosControlHAL(desc, controller_name="joint_trajectory_controller")
    >>> hal.connect()
    >>> hal.disconnect()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import Action, JointState, RobotDescription

from openral_hal._base import HALBase, _raw_floats

__all__ = ["ControllerKind", "RosControlHAL"]


class ControllerKind(StrEnum):
    """Which ros2_control controller sits behind one command topic.

    A command topic is not self-describing: the topic name says nothing about
    the message type the controller expects, and publishing the wrong type is
    *silent* — DDS simply never delivers it, so the arm does not move and
    nothing anywhere logs an error. That is the same class of failure as the
    no-op transport this layer already had to fix once, so the wire format is
    declared rather than assumed.

    A HAL names the kind per topic in `command_bindings`; `RosControlTransport`
    builds the matching publisher and message from it. Adding a robot whose
    controllers differ therefore means declaring a kind, not writing transport
    code.

    Only kinds this transport can actually build a message for belong here —
    an unlisted controller must fail loudly at wire-up rather than be
    approximated by a neighbouring one.
    """

    #: `joint_trajectory_controller/JointTrajectoryController`, commanded via
    #: `trajectory_msgs/JointTrajectory` on `~/joint_trajectory`. Every
    #: ros2_control robot in this repo uses this, grippers included (OpenArm's
    #: are 1-DoF instances of it).
    JOINT_TRAJECTORY = "joint_trajectory"

    #: The `forward_command_controller` family — `ForwardCommandController`,
    #: `position_controllers/JointGroupPositionController` and siblings —
    #: commanded via `std_msgs/Float64MultiArray` on `~/commands`. No robot in
    #: this repo uses one yet; it is here because it is the other type a
    #: ros2_control arm is commonly configured with, and because two kinds are
    #: what make the dispatch real rather than a rename of an assumption.
    FORWARD_COMMAND = "forward_command"


log = structlog.get_logger(__name__)

# Type alias for the injectable transport callable used in tests.
# Signature: (topic: str, msg: dict[str, object]) -> None
_PublishFn = Callable[[str, dict[str, object]], None]


def _default_publish(topic: str, msg: dict[str, object]) -> None:  # pragma: no cover
    """No-op publish used when no real ROS 2 node is available.

    In production this is replaced by the actual ``rclpy`` publisher at
    ``connect()`` time.  The no-op is only reached in fully isolated unit tests
    that do not inject a custom transport.
    """
    log.debug("hal.publish", topic=topic, fields=list(msg.keys()))


class RosControlHAL(HALBase):
    """ros2_control-backed HAL adapter.

    The adapter does not import ``rclpy`` directly so it can be unit-tested
    without a live ROS 2 installation.  In integration / HIL tests, inject a
    real publisher via the ``publish_fn`` parameter.

    Args:
        description: Normative ``RobotDescription`` for the target robot.
        controller_name: Name of the ``ros2_control`` joint trajectory
            controller, e.g. ``"joint_trajectory_controller"``.
        joint_state_topic: ROS 2 topic that publishes ``sensor_msgs/JointState``.
            Defaults to ``"/joint_states"``.
        command_topic: ROS 2 topic for joint trajectory commands.
            Defaults to ``"/<controller_name>/joint_trajectory"``.
        publish_fn: Callable used to send messages to ROS 2 topics.  Defaults
            to a no-op logger; replace with a real publisher in integration
            tests.
        state_fn: Callable that returns the latest raw joint state as a dict.
            Defaults to ``None``, in which case the adapter returns a zeroed
            ``JointState``.  Replace with a real subscriber callback in
            integration tests.
        staleness_limit_s: Maximum age (seconds) of a ``read_state()`` reading
            before ``ROSPerceptionStale`` is raised.  Defaults to ``0.5 s``.

    Raises:
        ROSConfigError: If ``description.joints`` is empty.
    """

    def __init__(
        self,
        description: RobotDescription,
        controller_name: str,
        *,
        joint_state_topic: str = "/joint_states",
        command_topic: str | None = None,
        publish_fn: _PublishFn | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the adapter; does not open any connection yet."""
        if not description.joints:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no joints; "
                "cannot initialise RosControlHAL."
            )
        self.description = description
        self._controller_name = controller_name
        self._joint_state_topic = joint_state_topic
        self._command_topic = command_topic or f"/{controller_name}/joint_trajectory"
        self._publish_fn: _PublishFn = publish_fn or _default_publish
        self._state_fn = state_fn
        self._staleness_limit_s = staleness_limit_s

        self._connected: bool = False
        self._last_state_time: float = 0.0
        # Set by `attach_transport` to the transport's real per-message arrival
        # clock. While it is None the HAL has no way to know how old its state
        # is (see `read_state`).
        self._stamp_fn: Callable[[], float] | None = None
        self._joint_names: list[str] = [j.name for j in description.joints]

    # ── Transport wiring ───────────────────────────────────────────────────────

    def attach_transport(
        self,
        publish_fn: _PublishFn,
        state_fn: Callable[[], dict[str, object]],
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind this HAL to a live transport after construction.

        ``build_hal`` constructs the HAL from the manifest alone, before any ROS
        node exists, so a real deployment cannot pass ``publish_fn``/``state_fn``
        to ``__init__``. The lifecycle node calls this once it has a node to
        create publishers on — which is what makes a real ros2_control robot
        work without any per-robot wiring code.

        Args:
            publish_fn: Sends one command dict to a controller topic.
            state_fn: Returns the newest joint state as a raw dict.
            stamp_fn: Returns the ``time.monotonic()`` timestamp of the newest
                joint state. Without it ``read_state`` can only measure age from
                ``connect()``, which is not a freshness check at all.

        Raises:
            ROSConfigError: If ``stamp_fn`` is given but is not callable.
        """
        # Checked here rather than at first use: `stamp_fn` feeds the staleness
        # watchdog, so a caller that passes a *value* (a property read once at
        # wire-up, say) instead of a callable would otherwise fail deep in the
        # hot path, on hardware, with the robot already connected.
        if stamp_fn is not None and not callable(stamp_fn):
            raise ROSConfigError(
                f"attach_transport(stamp_fn=...) needs a callable returning the newest "
                f"joint state's time.monotonic() timestamp, got {type(stamp_fn).__name__}. "
                "A property must be wrapped, e.g. `lambda: transport.last_stamp`."
            )
        self._publish_fn = publish_fn
        self._state_fn = state_fn
        self._stamp_fn = stamp_fn
        self._last_state_time = time.monotonic()

    def command_bindings(self) -> dict[str, ControllerKind]:
        """Return each controller topic this HAL publishes to, with its kind.

        One entry for a single-controller arm; robots whose controllers are
        split (the bimanual OpenArm's arm+gripper per side) override this.
        Insertion order is the fan-out order `send_action` uses, so it is
        load-bearing.

        This is the override point for a new robot: declaring the topic *and*
        the message type its controller speaks is what lets the transport
        publish without guessing, so adding a robot needs no transport code.
        """
        return {self._command_topic: ControllerKind.JOINT_TRAJECTORY}

    def command_topics(self) -> list[str]:
        """Return every controller topic this HAL publishes to, in fan-out order.

        Derived from `command_bindings` so the topic list and the declared wire
        formats cannot drift apart. Override `command_bindings`, not this.
        """
        return list(self.command_bindings())

    def ros2_control_joint_names(self) -> list[str]:
        """Return the joint names in ros2_control's namespace, in action order.

        Defaults to the manifest's own names; robots whose URDF names differ
        from the manifest's (again, OpenArm) override it. ``/joint_states`` is
        keyed by these, so a transport matches on them.
        """
        return list(self._joint_names)

    @property
    def joint_state_topic(self) -> str:
        """The aggregated ``sensor_msgs/JointState`` topic this HAL reads."""
        return self._joint_state_topic

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the connection to the robot hardware or simulator.

        Raises:
            ROSRuntimeError: If already connected.
        """
        if self._connected:
            raise ROSRuntimeError(f"RosControlHAL('{self.description.name}') is already connected.")
        log.info(
            "hal.connect",
            robot=self.description.name,
            controller=self._controller_name,
            joint_state_topic=self._joint_state_topic,
            command_topic=self._command_topic,
        )
        self._connected = True
        self._last_state_time = time.monotonic()

    # ── Hot path ───────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state snapshot.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.

        Returns:
            Latest ``JointState`` for all joints in ``description.joints``.
        """
        self._require_connected("read_state")
        # With a transport attached, age is measured from the arrival of the
        # newest joint state — the only thing that actually says whether the
        # robot is still talking to us. Without one, the best available
        # reference is `connect()`; that is a liveness floor, not a freshness
        # check, and it is why a transport must supply `stamp_fn` on real
        # hardware. A transport that has received nothing yet reports 0.0,
        # which must read as "infinitely stale", never as "fresh at epoch".
        if self._stamp_fn is not None:
            last = self._stamp_fn()
            age = float("inf") if last <= 0.0 else time.monotonic() - last
        else:
            age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        raw: dict[str, object] = {}
        if self._state_fn is not None:
            raw = self._state_fn()

        return JointState(
            name=self._joint_names,
            position=_raw_floats(raw, "position", n),
            velocity=_raw_floats(raw, "velocity", n),
            effort=_raw_floats(raw, "effort", n),
            stamp_ns=int(time.time_ns()),
        )

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the ros2_control joint trajectory controller.

        Args:
            action: The ``Action`` produced by a Skill.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not in the robot's
                ``supported_control_modes``, or if the joint target dimensions
                do not match the robot's joint count.
        """
        self._require_connected("send_action")
        self._validate_action(action)

        # `joint_names` travels with the command so the transport forwards what
        # the HAL chose rather than keeping a second copy of the mapping that
        # can drift from it (the rule ADR-0102 set for the OpenArm fan-out).
        msg: dict[str, object] = {
            "control_mode": action.control_mode,
            "horizon": action.horizon,
            "joint_names": self.ros2_control_joint_names(),
            "joint_targets": action.joint_targets,
            "stamp_ns": action.stamp_ns,
        }
        self._publish_fn(self._command_topic, msg)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
        )

    # ── Safety ─────────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop.

        Sets the connection state to False before raising so that subsequent
        calls to ``read_state`` or ``send_action`` also fail fast.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name)
        self._connected = False
        raise ROSEStopRequested(f"Emergency stop triggered on robot '{self.description.name}'.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate_action(self, action: Action) -> None:
        supported = self.description.capabilities.supported_control_modes
        if supported and action.control_mode not in supported:
            raise ROSConfigError(
                f"Action control_mode '{action.control_mode}' is not in "
                f"supported_control_modes {supported} for robot "
                f"'{self.description.name}'."
            )
        self._validate_action_dims(action, len(self._joint_names))
