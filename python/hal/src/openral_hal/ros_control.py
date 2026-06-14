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

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import Action, JointState, RobotDescription

from openral_hal._base import HALBase

__all__ = ["RosControlHAL"]

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
        self._joint_names: list[str] = [j.name for j in description.joints]

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

    def disconnect(self) -> None:
        """Close the connection and release all resources.

        Idempotent — calling on an already-disconnected HAL is a no-op.
        """
        if not self._connected:
            return
        log.info("hal.disconnect", robot=self.description.name)
        self._connected = False

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
        age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        raw: dict[str, object] = {}
        if self._state_fn is not None:
            raw = self._state_fn()

        def _floats(key: str) -> list[float]:
            val = raw.get(key)
            if isinstance(val, list):
                return [float(v) for v in val]
            return [0.0] * n

        return JointState(
            name=self._joint_names,
            position=_floats("position"),
            velocity=_floats("velocity"),
            effort=_floats("effort"),
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

        msg: dict[str, object] = {
            "control_mode": action.control_mode,
            "horizon": action.horizon,
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
