# SPDX-License-Identifier: Apache-2.0
"""HAL + deploy-runtime spins end quietly on shutdown, even mid-publish; live errors propagate.

rclpy's SIGINT handler invalidates the context first; a timer callback already
dequeued (the sim bridge's depth publish) then publishes on the dead context, raises
``RCLError: publisher's context is invalid`` out of ``rclpy.spin`` and the HAL exited
1 — on Ctrl-C of an OpenArm ``deploy sim`` once its head depth camera published
(2026-10-04, ``lifecycle_node.py`` exit code 1). Both HAL mains spin through
``openral_hal.lifecycle.spin_until_shutdown``, which treats exactly that as teardown.

Real rclpy context, node, publisher and timer; the race is made deterministic by
shutting the context down inside the callback, before it publishes.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_hal_sigint_teardown_live.py
"""

from __future__ import annotations

import os

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy context — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)


def test_a_publish_racing_the_shutdown_ends_the_spin_quietly() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_hal.lifecycle import spin_until_shutdown
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("test_spin_until_shutdown_race")
    pub = node.create_publisher(Empty, "/test_spin_until_shutdown", 1)

    def tick() -> None:
        rclpy.shutdown()  # what the SIGINT handler does first
        pub.publish(Empty())  # the dequeued sensor publish: RCLError, context invalid

    node.create_timer(0.01, tick)
    try:
        spin_until_shutdown(node)  # must return, not raise
        assert not rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_an_rcl_error_on_a_live_context_still_propagates() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_hal.lifecycle import spin_until_shutdown
    from rclpy._rclpy_pybind11 import RCLError

    rclpy.init()
    node = rclpy.create_node("test_spin_until_shutdown_live_error")

    def tick() -> None:
        raise RCLError("a real failure with the context still up")

    node.create_timer(0.01, tick)
    try:
        with pytest.raises(RCLError, match="real failure"):
            spin_until_shutdown(node)
        assert rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_the_deploy_runtime_executor_ends_quietly_on_the_same_race() -> None:
    """``runtime_node`` spins a ``MultiThreadedExecutor``; it exited 1 on this race too."""
    rclpy = pytest.importorskip("rclpy")
    from openral_rskill_ros.compose import spin_until_shutdown
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("test_runtime_spin_until_shutdown_race")
    pub = node.create_publisher(Empty, "/test_runtime_spin_until_shutdown", 1)

    def tick() -> None:
        rclpy.shutdown()
        pub.publish(Empty())

    node.create_timer(0.01, tick)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        spin_until_shutdown(executor)  # must return, not raise
        assert not rclpy.ok()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
