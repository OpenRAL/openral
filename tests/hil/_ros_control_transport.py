"""Lab-runner-only ``rclpy`` bridge for single-controller real-HW HALs.

The real-HW HAL adapters (``UR5eRealHAL`` / ``UR10eRealHAL`` /
``FrankaPandaRealHAL`` / ``SawyerRealHAL``) do not import ``rclpy``
themselves — the HAL Protocol is wire-format-free.  The transport is
injected at construction time via ``publish_fn`` / ``state_fn`` callables.

In production the transport is the per-HAL ROS 2 lifecycle node
(``packages/openral_hal_<robot>/``).  Inside a HIL test we bring up a
minimal ``rclpy`` node locally so the test can drive the real vendor
``ros2_control`` controller end-to-end.

This module is HIL-only — it is imported by ``tests/hil/test_<robot>.py``
after the test has confirmed both ``rclpy`` and the live driver are
present.  The unit-lane conformance tests use
:class:`openral_hal.sim_transport.SimTransport` instead and never
touch ``rclpy``.

Defaults match ``ur_robot_driver``
(``/scaled_joint_trajectory_controller/joint_trajectory``,
``/joint_states``); pass explicit topics for other controllers.

Per CLAUDE.md §5.4: real component or ``pytest.skip`` — nothing in
between.  The helper raises ``RuntimeError`` if it is imported on a host
without ``rclpy`` so the caller has to guard with
``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from typing import Any

if importlib.util.find_spec("rclpy") is None:  # pragma: no cover
    raise RuntimeError(
        "rclpy is not installed; this transport may only be imported by HIL "
        "tests after they've confirmed rclpy is available."
    )

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState as RosJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

__all__ = ["RosControlHILTransport", "make_hil_transport"]


_CONTROL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    depth=10,
)


def _make_trajectory_publisher(node: Node, command_topic: str) -> Publisher:
    """Create a ``trajectory_msgs/JointTrajectory`` publisher with the HIL QoS.

    Shared with :mod:`tests.hil._aloha_ros_transport` so the per-arm and
    per-gripper publishers in the bimanual bridge use the same QoS profile
    as the single-controller bridge.
    """
    return node.create_publisher(JointTrajectory, command_topic, _CONTROL_QOS)


class RosControlHILTransport:
    """Tiny ``rclpy`` bridge for HIL tests against a live ``ros2_control`` driver.

    Subscribes to ``joint_state_topic`` and re-publishes commanded
    trajectories on ``command_topic``.  The most recent joint state is
    cached and surfaced via :meth:`state` so the HAL's ``read_state()`` can
    return it without blocking.

    The class is intended to be used as a fixture; call :meth:`spin_once`
    inside the test loop to drain incoming joint-state messages.

    Args:
        node: A live ``rclpy`` node owned by the test (cleanup is the test's
            responsibility).
        joint_names: Canonical joint names in the order the HAL exposes
            them.  Used to project incoming ``sensor_msgs/JointState`` (which
            may contain extra topics or a different order) onto the HAL's
            joint vector.
        command_topic: ``trajectory_msgs/JointTrajectory`` topic the
            controller subscribes to (default
            ``"/scaled_joint_trajectory_controller/joint_trajectory"`` —
            the ``ur_robot_driver`` convention; override for other
            controllers).
        joint_state_topic: ``sensor_msgs/JointState`` topic the driver
            publishes on (default ``"/joint_states"``).
    """

    def __init__(
        self,
        node: Node,
        joint_names: list[str],
        *,
        command_topic: str = "/scaled_joint_trajectory_controller/joint_trajectory",
        joint_state_topic: str = "/joint_states",
    ) -> None:
        self._node = node
        self._joint_names = joint_names
        self._publisher = _make_trajectory_publisher(node, command_topic)
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_stamp = 0.0
        node.create_subscription(
            RosJointState,
            joint_state_topic,
            self._on_joint_state,
            _CONTROL_QOS,
        )

    # -- Transport callables (injected into the HAL) --------------------------

    def publish(self, _topic: str, msg: dict[str, object]) -> None:
        targets = msg.get("joint_targets")
        if not isinstance(targets, list) or not targets:
            return
        last_step = targets[-1]
        if not isinstance(last_step, list):
            return
        traj = JointTrajectory()
        traj.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in last_step]
        # 100 ms forward — short enough to feel responsive, long enough to
        # avoid trajectory-controller "time in the past" warnings.
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100_000_000
        traj.points.append(point)
        self._publisher.publish(traj)

    def state(self) -> dict[str, object]:
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self._joint_names:
            p, v, e = self._latest.get(name, (0.0, 0.0, 0.0))
            positions.append(p)
            velocities.append(v)
            efforts.append(e)
        return {"position": positions, "velocity": velocities, "effort": efforts}

    # -- Helpers --------------------------------------------------------------

    def spin_once(self, timeout_sec: float = 0.05) -> None:
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    @property
    def last_stamp(self) -> float:
        return self._last_stamp

    def wait_for_first_state(self, deadline_s: float = 2.0) -> bool:
        """Return True once at least one joint-state message has been received."""
        start = time.monotonic()
        while time.monotonic() - start < deadline_s:
            self.spin_once(timeout_sec=0.05)
            if self._latest:
                return True
        return False

    # -- Internal callbacks ---------------------------------------------------

    def _on_joint_state(self, msg: Any) -> None:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))
        for i, name in enumerate(names):
            p = positions[i] if i < len(positions) else 0.0
            v = velocities[i] if i < len(velocities) else 0.0
            e = efforts[i] if i < len(efforts) else 0.0
            self._latest[name] = (p, v, e)
        self._last_stamp = time.monotonic()


def make_hil_transport(
    node_name: str,
    joint_names: list[str],
    *,
    command_topic: str = "/scaled_joint_trajectory_controller/joint_trajectory",
    joint_state_topic: str = "/joint_states",
) -> tuple[Node, RosControlHILTransport, Callable[[], None]]:
    """Initialise rclpy and return ``(node, transport, cleanup)`` for a HIL test.

    The cleanup callable destroys the node and shuts ``rclpy`` down — call
    it from the fixture's teardown branch.
    """
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(node_name)
    transport = RosControlHILTransport(
        node,
        joint_names,
        command_topic=command_topic,
        joint_state_topic=joint_state_topic,
    )

    def cleanup() -> None:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return node, transport, cleanup
