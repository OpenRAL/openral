"""ADR-0024 / ADR-0025 — live exercise of ``_PandaMobileLifecycleNode``.

Brings up the real panda_mobile HAL lifecycle node (built and
installed via ``just ros2-build``), drives it through
``UNCONFIGURED → INACTIVE → ACTIVE → INACTIVE → UNCONFIGURED``, and
asserts that the three publishers fire at their declared rates:

* ``/joint_states`` at 30 Hz (base-class joint-state publisher).
* ``/odom`` at 20 Hz (mobile-base extras).
* ``/scan`` at 10 Hz (mobile-base extras).

The TF broadcast on ``odom -> base_link`` is verified by subscribing
to ``/tf`` and checking at least one frame transform arrives within
the same window. No mocks per CLAUDE.md §1.11 — real
``PandaMobileHAL`` (in-process digital twin), real rclpy executor,
real ROS publishers + subscribers.

Gates: ``ROS_DISTRO`` env + ``rclpy`` import + a colcon install
exposing ``openral_hal_panda_mobile``. When any of those are
missing the test ``pytest.skip(reason=...)`` cleanly per §1.11.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from collections.abc import Iterator
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

try:
    from openral_hal.lifecycle import ManifestHALLifecycleNode
except ImportError:
    pytest.skip(
        "rclpy unavailable; ManifestHALLifecycleNode not importable.",
        allow_module_level=True,
    )

_PANDA_MOBILE_YAML = Path(__file__).resolve().parents[2] / "robots" / "panda_mobile" / "robot.yaml"


# ── Per-publisher counter subscriber ─────────────────────────────────────


_COUNT_WINDOW_S = 2.0
"""Sampling window. 2 s × 30 Hz = 60 joint-state messages; we assert
``>= 25`` to give startup + lifecycle-transition slack."""

# Per-topic minimum counts at 2 s. Floors are loose to absorb startup
# + executor scheduling slack; ceiling assertions would be brittle.
_MIN_JOINT_STATES = 25  # 30 Hz × 2 s = 60 nominal
_MIN_ODOM = 15  # 20 Hz × 2 s = 40 nominal
_MIN_SCAN = 7  # 10 Hz × 2 s = 20 nominal
_MIN_TF = 1  # at least one TF broadcast


@pytest.fixture
def node_and_executor() -> Iterator[tuple[Any, Any]]:
    """Construct the manifest-driven node + start a SingleThreadedExecutor in a thread."""
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter

    rclpy.init()
    # issue #191 Phase 3 — panda_mobile runs on the generic node now. Digital-twin
    # sim (no sim_env_yaml) → PandaMobileHAL; MobileBaseBridge (gated on the
    # manifest's base_joints) publishes /odom + TF + /cmd_vel; SimSensorBridge
    # publishes the /scan no-hit fan.
    node: Any = ManifestHALLifecycleNode("openral_hal_panda_mobile")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(_PANDA_MOBILE_YAML)),
            Parameter("hal_mode", value="sim"),
            Parameter("viewer_enabled", value=False),
        ]
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        yield node, executor
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def test_lifecycle_publishes_joint_states_odom_scan_and_tf(
    node_and_executor: tuple[Any, Any],
) -> None:
    """Configure → activate → assert per-topic rates → deactivate → cleanup."""
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState, LaserScan
    from tf2_msgs.msg import TFMessage

    node, executor = node_and_executor

    # Drive lifecycle into ACTIVE.
    assert str(node.trigger_configure()).endswith("SUCCESS"), "configure failed"
    assert str(node.trigger_activate()).endswith("SUCCESS"), "activate failed"

    # Subscribe everything we want to count, with QoS that matches
    # the node's publishers (sensor /scan is BEST_EFFORT, control
    # /joint_states + /odom are RELIABLE, TF default uses RELIABLE).
    qos_re = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=20,
    )
    qos_be = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=20,
    )

    sub_node = Node("test_panda_mobile_smoke_subscriber")
    counts = {"joint_states": 0, "odom": 0, "scan": 0, "tf": 0}

    def make_cb(key: str) -> Any:
        def cb(_msg: Any) -> None:
            counts[key] += 1

        return cb

    sub_node.create_subscription(JointState, "/joint_states", make_cb("joint_states"), qos_re)
    sub_node.create_subscription(Odometry, "/odom", make_cb("odom"), qos_re)
    sub_node.create_subscription(LaserScan, "/scan", make_cb("scan"), qos_be)
    sub_node.create_subscription(TFMessage, "/tf", make_cb("tf"), qos_re)
    executor.add_node(sub_node)

    time.sleep(_COUNT_WINDOW_S)

    # Drive lifecycle back to UNCONFIGURED so the fixture teardown
    # doesn't race the timers.
    assert str(node.trigger_deactivate()).endswith("SUCCESS"), "deactivate failed"
    assert str(node.trigger_cleanup()).endswith("SUCCESS"), "cleanup failed"
    sub_node.destroy_node()

    # Assertions — floors absorb startup + transition slack.
    assert counts["joint_states"] >= _MIN_JOINT_STATES, (
        f"/joint_states should hit >= {_MIN_JOINT_STATES} messages in "
        f"{_COUNT_WINDOW_S}s @ 30 Hz; got {counts['joint_states']}"
    )
    assert counts["odom"] >= _MIN_ODOM, (
        f"/odom should hit >= {_MIN_ODOM} messages in "
        f"{_COUNT_WINDOW_S}s @ 20 Hz; got {counts['odom']}"
    )
    assert counts["scan"] >= _MIN_SCAN, (
        f"/scan should hit >= {_MIN_SCAN} messages in "
        f"{_COUNT_WINDOW_S}s @ 10 Hz; got {counts['scan']}"
    )
    assert counts["tf"] >= _MIN_TF, (
        f"/tf should carry the odom->base_link broadcast (>= {_MIN_TF} message); got {counts['tf']}"
    )


def test_lifecycle_odom_pose_advances_under_body_twist(
    node_and_executor: tuple[Any, Any],
) -> None:
    """Sending a /openral/safe_action body_twist advances the published /odom pose.

    Construct an ActionChunk on /openral/safe_action with a 1 m/s
    forward velocity, integrate for ~0.5 s of in-process digital-twin
    HAL ticks, then assert the latest /odom message reports
    pose.pose.position.x > 0 (rough — the digital twin's Euler step
    is calibrated to 50 ms per safe_action send, so even a single
    send should advance the pose).
    """
    import math

    from nav_msgs.msg import Odometry
    from openral_msgs.msg import ActionChunk
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    node, executor = node_and_executor
    assert str(node.trigger_configure()).endswith("SUCCESS")
    assert str(node.trigger_activate()).endswith("SUCCESS")

    chunk_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    odom_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    sub_node = Node("test_panda_mobile_pose_subscriber")
    latest_odom: list[Odometry] = []

    def _on_odom(msg: Odometry) -> None:
        latest_odom.append(msg)

    sub_node.create_subscription(Odometry, "/odom", _on_odom, odom_qos)
    pub = sub_node.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)
    executor.add_node(sub_node)

    # Build an ActionChunk for BODY_TWIST. The HAL's
    # _apply_body_twist expects a 6-vec (vx, vy, vz, wx, wy, wz); only
    # the planar components (vx, vy, wz) are actuated. The wire-format
    # uint8 comes from `openral_core.CONTROL_MODE_TO_UINT8` (shared
    # producer/consumer table).
    from openral_core import CONTROL_MODE_TO_UINT8
    from openral_core.schemas import ControlMode

    body_twist_dim = 6
    chunk = ActionChunk()
    chunk.control_mode = CONTROL_MODE_TO_UINT8[ControlMode.BODY_TWIST]
    chunk.horizon = 1
    chunk.n_dof = body_twist_dim
    chunk.flat = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # row-major [horizon=1][n_dof=6]

    # Burst-send several action chunks so the in-process digital
    # twin steps multiple times. Each send advances by dt_s = 0.05 s,
    # so 10 sends ≈ 0.5 m at 1 m/s.
    for _ in range(10):
        pub.publish(chunk)
        time.sleep(0.06)

    # Wait one extra publish cycle so the next /odom timer fires.
    time.sleep(0.2)

    assert str(node.trigger_deactivate()).endswith("SUCCESS")
    assert str(node.trigger_cleanup()).endswith("SUCCESS")
    sub_node.destroy_node()

    assert latest_odom, "no /odom messages observed at all"
    final = latest_odom[-1]
    x = float(final.pose.pose.position.x)
    # Loose floor: the HAL's integrator does 0.05 m per safe_action
    # at 1 m/s × dt_s=0.05; 10 sends → 0.5 m nominal. Floor at 0.15 m
    # absorbs the race between safe_action arrival and /odom timer.
    assert x >= 0.15, f"base_x should advance >= 0.15 m after 10 body_twist sends; got x={x}"
    # The orientation should still be identity (no yaw command).
    qz = float(final.pose.pose.orientation.z)
    qw = float(final.pose.pose.orientation.w)
    assert math.isclose(qz, 0.0, abs_tol=1e-6)
    assert math.isclose(qw, 1.0, abs_tol=1e-6)
