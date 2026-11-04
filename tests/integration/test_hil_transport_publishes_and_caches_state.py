"""Integration smoke tests for the HIL transport bridges.

Exercises the round-trip publish + state-cache contract the real-HW HIL tests rely on,
without vendor hardware. Brings up in-process ``rclpy`` nodes, drives the bridge end-to-end,
and asserts:

1. ``RosControlHILTransport`` caches incoming ``sensor_msgs/JointState`` so ``state()``
   reflects the latest message.
2. The same bridge publishes ``trajectory_msgs/JointTrajectory`` on the command topic when
   ``publish()`` is called.
3. ``AlohaHILTransport`` turns ``AlohaHAL``'s ``{name, cmd}`` payloads into
   ``interbotix_xs_msgs`` ``JointGroupCommand`` / ``JointSingleCommand`` messages, and merges
   the two per-namespace ``xs_sdk`` joint-state streams into manifest order (skips without
   ``interbotix_xs_msgs``).

These guard against regressions in the bridge wiring itself; per-robot HIL tests assume the
bridge already works. Skips if ``ROS_DISTRO`` is not set (matches
``test_world_state_integration.py``'s gating pattern).
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
    import rclpy

    rclpy.init()
    try:
        yield rclpy
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_ros_control_bridge_caches_published_joint_state() -> None:
    """A ``JointState`` published on the bridge's topic must surface in ``state()``."""
    from sensor_msgs.msg import JointState as RosJointState

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
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy

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
    from trajectory_msgs.msg import JointTrajectory

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
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy

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


_needs_interbotix = pytest.mark.skipif(
    importlib.util.find_spec("interbotix_xs_msgs") is None,
    reason="interbotix_xs_msgs not installed — the ALOHA bridge speaks the xs_sdk messages.",
)

_ALOHA_JOINTS = [
    f"{side}_{j}"
    for side in ("left", "right")
    for j in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
]


@_needs_interbotix
def test_aloha_bridge_publishes_xs_sdk_messages_and_merges_state() -> None:
    """Arm -> ``JointGroupCommand``, gripper -> ``JointSingleCommand``; state merged by side."""
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState as RosJointState

    from tests.hil._aloha_ros_transport import AlohaHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_aloha_bridge_test")
        helper = rclpy.create_node("openral_hil_aloha_bridge_test_helper")
        try:
            transport = AlohaHILTransport(
                node,
                _ALOHA_JOINTS,
                arm_namespaces=("t_left", "t_right"),
                gripper_cmd_range=(1.0, 2.0),
            )
            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
            groups: list[Any] = []
            singles: list[Any] = []
            helper.create_subscription(
                JointGroupCommand, "/t_left/commands/joint_group", groups.append, qos
            )
            helper.create_subscription(
                JointSingleCommand, "/t_left/commands/joint_single", singles.append, qos
            )
            state_pubs = {
                ns: helper.create_publisher(RosJointState, f"/{ns}/joint_states", qos)
                for ns in ("t_left", "t_right")
            }
            arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not (
                groups and singles and {"left_gripper", "right_gripper"} <= transport._latest.keys()
            ):
                transport.publish("/t_left/commands/joint_group", {"name": "arm", "cmd": arm})
                transport.publish("/t_left/commands/joint_single", {"name": "gripper", "cmd": 1.5})
                for offset, pub in enumerate(state_pubs.values()):
                    msg = RosJointState()
                    msg.name = ["waist", "gripper"]
                    msg.position = [float(offset), 1.25 + offset * 0.5]
                    pub.publish(msg)
                rclpy.spin_once(helper, timeout_sec=0.05)
                transport.spin_once(timeout_sec=0.05)
                transport.spin_once(timeout_sec=0.05)

            assert groups and singles, "xs_sdk command messages never received"
            assert groups[-1].name == "arm"
            assert list(groups[-1].cmd) == pytest.approx(arm)
            assert singles[-1].name == "gripper"
            assert singles[-1].cmd == pytest.approx(1.5)

            position = transport.state()["position"]
            assert isinstance(position, list)
            by_name = dict(zip(_ALOHA_JOINTS, position, strict=True))
            assert by_name["right_waist"] == pytest.approx(1.0)
            assert by_name["left_gripper"] == pytest.approx(0.25)  # (1.25 - 1) / (2 - 1)
            assert by_name["right_gripper"] == pytest.approx(0.75)
        finally:
            helper.destroy_node()
            node.destroy_node()


@_needs_interbotix
def test_aloha_bridge_rejects_unknown_topic() -> None:
    """An unknown topic must raise ``ValueError`` so HAL drift fails loudly."""
    from tests.hil._aloha_ros_transport import AlohaHILTransport

    with _rclpy_session() as rclpy:
        node = rclpy.create_node("openral_hil_aloha_bridge_unknown_topic")
        try:
            transport = AlohaHILTransport(
                node,
                _ALOHA_JOINTS,
                arm_namespaces=("t_left", "t_right"),
                gripper_cmd_range=(1.0, 2.0),
            )
            with pytest.raises(ValueError, match="unknown topic"):
                transport.publish("/t_left/commands/joint_trajectory", {"name": "arm", "cmd": []})
        finally:
            node.destroy_node()
