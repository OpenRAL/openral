"""``/openral/estop`` freezes a sim HAL's physics; ``/openral/estop_cleared`` resumes it (#683).

The lifecycle node latches ``/openral/estop`` itself for a HAL that does not opt into
the hardware stop (``LifecycleEStopHAL``) — every sim HAL — and drops commands. Its
sim idle stepper used to keep stepping physics through the latch: the twin tracked its
last ``ctrl`` and the sim clock ran on while the graph read "e-stopped". The bridge's
idle stepper now holds while the node is latched, and resumes on ``estop_cleared``;
the twin is not torn down (it keeps answering ``read_state`` with the frozen pose).

Real ``ManifestHALLifecycleNode`` on ``robots/so100_follower`` (the derived
``MujocoArmHAL`` twin), real DDS topics, no mocks (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_sim_estop_freezes_physics_live.py
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_SO100_YAML = Path(__file__).resolve().parents[2] / "robots" / "so100_follower" / "robot.yaml"


def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_estop_freezes_sim_physics_and_estop_cleared_resumes_it() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("openral_msgs")
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    rclpy.init()
    node: Any = ManifestHALLifecycleNode(
        "test_sim_estop_freeze",
        parameter_overrides=[
            Parameter("robot_yaml", value=str(_SO100_YAML)),
            Parameter("hal_mode", value="sim"),
            Parameter("viewer_enabled", value=False),
        ],
    )
    helper = rclpy.create_node("test_sim_estop_freeze_helper")
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    estop_pub = helper.create_publisher(Empty, "/openral/estop", qos)
    cleared_pub = helper.create_publisher(Empty, "/openral/estop_cleared", qos)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.add_node(helper)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS"), "configure failed"
        assert str(node.trigger_activate()).endswith("SUCCESS"), "activate failed"
        hal = node._hal

        def sim_ns() -> int:
            value = hal.sim_time_ns()
            assert value is not None, "the twin lost its clock"
            return int(value)

        start = sim_ns()
        assert _wait_until(lambda: sim_ns() > start + 50_000_000), "idle stepper never ran"

        assert _wait_until(lambda: helper.count_subscribers("/openral/estop") > 0)
        estop_pub.publish(Empty())
        assert _wait_until(lambda: bool(node._estopped)), "the node never latched"
        time.sleep(0.3)  # an idle tick already dequeued may still land
        frozen = sim_ns()
        time.sleep(1.0)
        assert sim_ns() == frozen, "sim physics kept stepping through /openral/estop"
        # Not torn down: the frozen twin still answers reads.
        assert len(hal.read_state().position) == len(hal.description.joints)

        assert _wait_until(lambda: helper.count_subscribers("/openral/estop_cleared") > 0)
        cleared_pub.publish(Empty())
        assert _wait_until(lambda: not node._estopped), "estop_cleared never cleared the latch"
        assert _wait_until(lambda: sim_ns() > frozen + 50_000_000), "physics did not resume"
    finally:
        with suppress(Exception):
            node.trigger_deactivate()
            node.trigger_cleanup()
        executor.shutdown()
        node.destroy_node()
        helper.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=2.0)
