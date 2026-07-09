"""Integration smoke tests for the HIL transport bridges.

Exercises the round-trip publish + state-cache contract that the real-HW
HIL tests rely on, without needing any vendor hardware.  Brings up
in-process ``rclpy`` nodes, drives the bridge end-to-end, and asserts:

1. :class:`tests.hil._ros_control_transport.RosControlHILTransport` caches
   incoming ``sensor_msgs/JointState`` so ``state()`` reflects the latest
   message.
2. The same bridge publishes ``trajectory_msgs/JointTrajectory`` on the
   command topic when ``publish()`` is called.
3. :class:`tests.hil._aloha_ros_transport.AlohaHILTransport` dispatches
   arm and gripper publishes to the matching topic with the right joint
   layout.

These tests guard against regressions in the bridge wiring itself; the
per-robot HIL tests assume the bridge already works.

Skip if ``ROS_DISTRO`` is not set (matches the gating pattern in
``tests/integration/test_world_state_integration.py``).
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


@contextmanager
def _rclpy_session() -> Iterator[Any]:
    """Initialise rclpy for the test and shut down cleanly."""
    import rclpy  # type: ignore[import-untyped]

    rclpy.init()
    try:
        yield rclpy
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_ros_control_bridge_caches_published_joint_state() -> None:
    """A ``JointState`` published on the bridge's topic must surface in ``state()``."""
    from sensor_msgs.msg import JointState as RosJointState  # type: ignore[import-untyped]

    from tests.hil._ros_control_transport import RosControlHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_bridge_test_state")
        helper = rclpy.create_node("openral_hil_bridge_test_state_helper")
        try:
            transport = RosControlHILTransport(
                node,
                joint_names=["a", "b"],
                command_topic="/test/cmd",
                joint_state_topic="/test/states",
            )
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy  # type: ignore[import-untyped]

            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            publisher = helper.create_publisher(RosJointState, "/test/states", qos)

            # Publish until the bridge sees it (publishers + subscribers may
            # take a tick to discover each other under DDS).
            deadline = time.monotonic() + 2.0
            saw_state = False
            while time.monotonic() < deadline:
                msg = RosJointState()
                msg.name = ["a", "b"]
                msg.position = [0.1, 0.2]
                msg.velocity = [0.0, 0.0]
                msg.effort = [0.0, 0.0]
                publisher.publish(msg)
                rclpy.spin_once(helper, timeout_sec=0.05)
                transport.spin_once(timeout_sec=0.05)
                if transport.state()["position"] == [0.1, 0.2]:
                    saw_state = True
                    break
            assert saw_state, "RosControlHILTransport.state() never reflected the published message"
        finally:
            helper.destroy_node()
            node.destroy_node()


def test_ros_control_bridge_publishes_joint_trajectory() -> None:
    """``transport.publish`` must emit a ``JointTrajectory`` on the command topic."""
    from trajectory_msgs.msg import JointTrajectory  # type: ignore[import-untyped]

    from tests.hil._ros_control_transport import RosControlHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_bridge_test_pub")
        helper = rclpy.create_node("openral_hil_bridge_test_pub_helper")
        try:
            transport = RosControlHILTransport(
                node,
                joint_names=["a", "b"],
                command_topic="/test/cmd",
                joint_state_topic="/test/states",
            )
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy  # type: ignore[import-untyped]

            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            received: list[Any] = []
            helper.create_subscription(
                JointTrajectory,
                "/test/cmd",
                received.append,
                qos,
            )

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not received:
                transport.publish("/test/cmd", {"joint_targets": [[1.0, 2.0]]})
                rclpy.spin_once(helper, timeout_sec=0.05)
                transport.spin_once(timeout_sec=0.05)

            assert received, "JointTrajectory publish never reached the helper subscriber"
            traj = received[-1]
            assert list(traj.joint_names) == ["a", "b"]
            assert len(traj.points) == 1
            assert list(traj.points[0].positions) == [1.0, 2.0]
        finally:
            helper.destroy_node()
            node.destroy_node()


def test_aloha_bridge_dispatches_arm_and_gripper() -> None:
    """``AlohaHILTransport.publish`` must route by topic to the right publisher."""
    from trajectory_msgs.msg import JointTrajectory  # type: ignore[import-untyped]

    from tests.hil._aloha_ros_transport import AlohaHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_aloha_bridge_test")
        helper = rclpy.create_node("openral_hil_aloha_bridge_test_helper")
        try:
            joint_names = [
                "left_waist",
                "left_shoulder",
                "left_elbow",
                "left_forearm_roll",
                "left_wrist_angle",
                "left_wrist_rotate",
                "left_gripper",
                "right_waist",
                "right_shoulder",
                "right_elbow",
                "right_forearm_roll",
                "right_wrist_angle",
                "right_wrist_rotate",
                "right_gripper",
            ]
            transport = AlohaHILTransport(
                node,
                joint_names=joint_names,
                left_arm_command_topic="/test/left_arm/joint_trajectory",
                right_arm_command_topic="/test/right_arm/joint_trajectory",
                left_gripper_command_topic="/test/left_gripper/command",
                right_gripper_command_topic="/test/right_gripper/command",
                joint_state_topic="/test/aloha_states",
            )
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy  # type: ignore[import-untyped]

            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            arm_msgs: list[Any] = []
            gripper_msgs: list[Any] = []
            helper.create_subscription(
                JointTrajectory,
                "/test/left_arm/joint_trajectory",
                arm_msgs.append,
                qos,
            )
            helper.create_subscription(
                JointTrajectory,
                "/test/left_gripper/command",
                gripper_msgs.append,
                qos,
            )

            arm_chunk = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and (not arm_msgs or not gripper_msgs):
                transport.publish(
                    "/test/left_arm/joint_trajectory",
                    {"joint_targets": arm_chunk},
                )
                transport.publish(
                    "/test/left_gripper/command",
                    {"position": 0.025, "stamp_ns": time.time_ns()},
                )
                rclpy.spin_once(helper, timeout_sec=0.05)
                transport.spin_once(timeout_sec=0.05)

            assert arm_msgs, "left-arm JointTrajectory never received"
            assert gripper_msgs, "left-gripper JointTrajectory never received"

            arm_traj = arm_msgs[-1]
            assert list(arm_traj.joint_names) == joint_names[0:6]
            assert list(arm_traj.points[0].positions) == arm_chunk[0]

            grip_traj = gripper_msgs[-1]
            assert list(grip_traj.joint_names) == [joint_names[6]]
            assert list(grip_traj.points[0].positions) == [0.025]
        finally:
            helper.destroy_node()
            node.destroy_node()


def test_aloha_bridge_rejects_unknown_topic() -> None:
    """An unknown topic must raise ``ValueError`` so HAL drift fails loudly."""
    from tests.hil._aloha_ros_transport import AlohaHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_aloha_bridge_unknown_topic")
        try:
            joint_names = [f"j{i}" for i in range(14)]
            transport = AlohaHILTransport(
                node,
                joint_names=joint_names,
                left_arm_command_topic="/test/left_arm/joint_trajectory",
                right_arm_command_topic="/test/right_arm/joint_trajectory",
                left_gripper_command_topic="/test/left_gripper/command",
                right_gripper_command_topic="/test/right_gripper/command",
                joint_state_topic="/test/aloha_states",
            )
            with pytest.raises(ValueError, match="unknown topic"):
                transport.publish("/test/bogus", {"joint_targets": [[0.0] * 6]})
        finally:
            node.destroy_node()
