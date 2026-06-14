"""Live exercise of the reflective ``ResetToPose`` service (issue #191 Phase 2).

``ManifestHALLifecycleNode`` opens ``/openral/<robot>/reset_to_pose`` **only**
when the HAL it built exposes ``reset_to_pose`` — generalising the service that
previously lived, hand-wired, only in the bespoke openarm node (ADR-0029
blocker #4). Every ``MujocoArmHAL`` sim arm gains it for free; a HAL without the
method (panda_mobile's ``PandaMobileHAL``) gets no service.

These tests bring the node up for real and assert:

* franka_panda (sim → ``MujocoArmHAL``) exposes the service, and calling it with
  a target pose actually snaps the simulator (the streamed ``/joint_states``
  reflect the new joint angle);
* panda_mobile (sim → ``PandaMobileHAL``, no ``reset_to_pose``) gets no service.

Real ``RobotDescription`` + real HAL + real rclpy service round-trip — no mocks
(CLAUDE.md §1.11). Skips cleanly without ROS / rclpy / mujoco.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — this test requires a sourced ROS 2 install.",
)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("mujoco")

try:
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from openral_msgs.srv import ResetToPose
except ImportError:  # pragma: no cover - no-rclpy / unbuilt-msgs hosts
    pytest.skip(
        "rclpy / openral_msgs unavailable; build the workspace + source install/setup.bash.",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_YAML = REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
PANDA_MOBILE_YAML = REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"


def _make_node(robot_yaml: Path) -> Any:
    from rclpy.parameter import Parameter

    node = ManifestHALLifecycleNode("test_reset_to_pose")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(robot_yaml)),
            Parameter("hal_mode", value="sim"),
            Parameter("publish_rate_hz", value=50.0),
        ]
    )
    return node


def test_reset_to_pose_present_and_snaps_simulator() -> None:
    """franka sim exposes the reflective service; a call snaps the qpos."""
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = _make_node(FRANKA_YAML)
    listener = Node("test_reset_to_pose_listener")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(listener)

    latest: dict[str, list[float]] = {"pos": []}
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    listener.create_subscription(
        JointState, "/joint_states", lambda m: latest.update(pos=list(m.position)), qos
    )
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS")
        assert str(node.trigger_activate()).endswith("SUCCESS")
        # Service was opened by the reflection (HAL exposes reset_to_pose).
        assert node._reset_to_pose_srv is not None
        time.sleep(0.5)
        initial = list(latest["pos"])
        assert initial, "no /joint_states observed before reset"

        # Target: nudge joint 0 by +0.15 rad (small, within limits), keep the rest.
        target = list(initial)
        target[0] = initial[0] + 0.15

        client = listener.create_client(ResetToPose, "/openral/franka_panda/reset_to_pose")
        assert client.wait_for_service(timeout_sec=5.0), "reset_to_pose service never advertised"
        req = ResetToPose.Request()
        req.pose = target
        future = client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        assert future.done(), "reset_to_pose call did not return"
        resp = future.result()
        assert resp.success, f"reset_to_pose failed: {resp.failure_reason}"

        time.sleep(0.5)
        after = list(latest["pos"])
        assert abs(after[0] - target[0]) < 0.05, (
            f"joint 0 did not snap: target={target[0]:.3f} after={after[0]:.3f}"
        )
        assert str(node.trigger_deactivate()).endswith("SUCCESS")
        assert str(node.trigger_cleanup()).endswith("SUCCESS")
    finally:
        executor.shutdown()
        node.destroy_node()
        listener.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2.0)


def test_reset_to_pose_absent_when_hal_lacks_it() -> None:
    """panda_mobile sim (PandaMobileHAL has no reset_to_pose) gets no service."""
    rclpy.init()
    node = _make_node(PANDA_MOBILE_YAML)
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS")
        # No reflective service — the HAL does not expose reset_to_pose.
        assert node._reset_to_pose_srv is None
        assert str(node.trigger_cleanup()).endswith("SUCCESS")
    finally:
        node.destroy_node()
        rclpy.shutdown()
