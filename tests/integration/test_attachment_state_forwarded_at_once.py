"""A new attachment revision reaches the kernel's input at once, not on the next timer tick.

Real ``_WorldStateLifecycleNode`` and real ``openral_msgs`` over DDS, with the fast timer
at its slowest (1 Hz). The node's timer runs on the node clock, which stands still while a
sim twin holds a tick for its attachment perception barrier: a revision ingested in that
window (a support witness retired 40 ms after the attach, seen live on Spark) waited for
the clock to move again, ~2 s, and the barrier waits on the kernel's ack of that revision.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openral_msgs") is None or importlib.util.find_spec("rclpy") is None,
    reason="openral_msgs / rclpy not importable — ROS 2 workspace not sourced",
)


def test_an_attachment_revision_is_published_before_the_next_tick() -> None:
    import rclpy
    from openral_msgs.msg import AttachmentState, WorldStateStamped
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    node = _WorldStateLifecycleNode()
    node.set_parameters(
        [rclpy.parameter.Parameter("publish_rate_hz_fast", value=1.0)]  # one tick per second
    )
    helper = rclpy.create_node("test_attachment_forward_helper")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    seen: list[tuple[float, int]] = []
    helper.create_subscription(
        WorldStateStamped,
        "/openral/world_state_fast",
        lambda m: seen.append((time.monotonic(), int(m.attachment_revision))),
        QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        ),
    )
    pub = helper.create_publisher(
        AttachmentState,
        "/openral/attachment_state",
        QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        ),
    )
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # Wait for a timer tick, then publish right after it: the next tick is ~1 s away.
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert seen, "the world state never ticked"
        msg = AttachmentState()
        msg.revision = 7
        pub.publish(msg)
        sent = time.monotonic()
        deadline = sent + 0.6
        while time.monotonic() < deadline and not any(r == 7 for _, r in seen):
            executor.spin_once(timeout_sec=0.02)
        got = [t for t, r in seen if r == 7]
        assert got, "revision 7 was not forwarded within 0.6 s (waited for the 1 Hz timer?)"
        assert got[0] - sent < 0.5
    finally:
        executor.shutdown()
        helper.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()
