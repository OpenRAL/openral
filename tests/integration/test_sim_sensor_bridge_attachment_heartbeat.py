"""A sim HAL without attach mechanics still reports "nothing attached, fresh".

The deploy launch enables the C++ kernel's attached-payload check for every
sim robot with collision capsules, and that check is fail-closed on a payload
it cannot verify — including one it has never heard about: a world state whose
``attachment_stamp_ns`` is still 0 drops every JOINT chunk as
``attached_overflow``. Until 2026-09-22 the bridge only published
``/openral/attachment_state`` for a HAL with the attachment API, so the OpenArm
MuJoCo twin (a plain ``MujocoArmHAL``, no attach mechanics) had zero publishers
on that topic and not one joint chunk ever reached the arm.

Real bridge, real rclpy node, real ``OpenArmMujocoHAL``; lives in
``tests/integration/`` because test-selective has no ``rclpy``::

    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_sim_sensor_bridge_attachment_heartbeat.py
"""

from __future__ import annotations

import os
import time

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_a_hal_without_the_attachment_api_still_heartbeats_an_empty_fresh_set() -> None:
    rclpy = pytest.importorskip("rclpy")

    from openral_hal.openarm import OPENARM_DESCRIPTION, OpenArmMujocoHAL
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from openral_msgs.msg import AttachmentState
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    hal = OpenArmMujocoHAL(gravity_enabled=False)
    assert not hasattr(hal, "update_attached_objects"), "fixture must lack the attachment API"

    rclpy.init()
    try:
        node = Node("test_attachment_heartbeat")
        try:
            received: list[AttachmentState] = []
            node.create_subscription(
                AttachmentState,
                "/openral/attachment_state",
                received.append,
                QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    depth=1,
                ),
            )
            # Same order as the HAL lifecycle node: connect, then activate the
            # bridge's streams via ``setup()``.
            hal.connect()
            bridge = SimSensorBridge(node, hal, OPENARM_DESCRIPTION, viewer_enabled=False)
            try:
                bridge.setup()
                # Publisher and heartbeat exist; the staging path (which needs
                # the API to have anything to stage) does not.
                assert bridge._attachment_pub is not None
                assert bridge._attachment_timer is not None
                assert bridge._attachment_sub is None

                deadline = time.monotonic() + 2.0
                while not received and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                assert received, "no /openral/attachment_state heartbeat within 2 s"
                msg = received[-1]
                assert list(msg.objects) == []
                assert msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0, (
                    "an unstamped snapshot is exactly what the kernel fails closed on"
                )
                assert msg.place_declaration_valid is False
            finally:
                bridge.teardown()
                hal.disconnect()
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
