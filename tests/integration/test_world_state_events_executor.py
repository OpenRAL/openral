"""``start_world_state_executor``: world state served by rclpy's C++ events executor.

Real ``_WorldStateLifecycleNode``, real ``sensor_msgs/JointState`` over DDS. The
helper publisher spins on its own ``SingleThreadedExecutor``; the world-state
node is on the events executor thread, and its aggregator must see the joint
state that thread ingested. Skips where ``openral_msgs``/rclpy or the
``rclpy.experimental`` executor (Jazzy 7.1+) are absent.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openral_msgs") is None or importlib.util.find_spec("rclpy") is None,
    reason="openral_msgs / rclpy not importable — ROS 2 workspace not sourced",
)


def test_world_state_ingests_on_the_events_executor_thread() -> None:
    import rclpy  # type: ignore[import-untyped]
    from openral_rskill_ros.compose import start_world_state_executor
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn  # type: ignore[import-untyped]
    from sensor_msgs.msg import JointState as RosJointState  # type: ignore[import-untyped]

    if importlib.util.find_spec("rclpy.experimental") is None:
        pytest.skip("this rclpy ships no EventsExecutor (pre-Jazzy 7.1)")

    rclpy.init()
    node = _WorldStateLifecycleNode()
    topic = "/test_events_executor/joint_states"
    node.set_parameters([rclpy.parameter.Parameter("joint_states_topic", value=topic)])
    stop = start_world_state_executor(node)
    assert stop is not None
    helper = rclpy.create_node("test_events_executor_helper")
    helper_executor = rclpy.executors.SingleThreadedExecutor()
    helper_executor.add_node(helper)
    pub = helper.create_publisher(RosJointState, topic, 5)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        msg = RosJointState()
        msg.name = ["j0", "j1"]
        msg.position = [0.25, -0.5]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pub.publish(msg)
            helper_executor.spin_once(timeout_sec=0.05)
            js = node._aggregator.snapshot().joint_state if node._aggregator else None
            if js is not None and list(js.position) == [0.25, -0.5]:
                break
        else:
            pytest.fail("world state never ingested the joint state on the events executor")
    finally:
        stop()
        helper.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()
