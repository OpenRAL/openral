"""Every sim HAL heartbeats a fresh, empty attachment set — not only the ones with attach mechanics.

The deploy launch enables the kernel's attached-payload check for every
sim robot with collision capsules, and that check is fail-closed on a payload
it cannot verify — including one it has never heard about: a world state
whose ``attachment_stamp_ns`` is still 0 is unverifiable, so every joint
chunk is dropped as ``attached_overflow``. A ``MujocoArmHAL`` twin has no
attachment API at all, so "nothing attached, fresh" is the exact truth for
it, and saying so is what lets the kernel certify its motion. Seen on the
OpenArm twin (qorin1, 2026-09-22): zero publishers on
``/openral/attachment_state`` and not one joint chunk reached the arm.

Real ``SimSensorBridge`` on a real rclpy node against real MuJoCo twins of
three different robots, so the guarantee is the bridge's, not one HAL's.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)


def _openarm() -> Any:
    from openral_hal.openarm import OpenArmMujocoHAL

    return OpenArmMujocoHAL(gravity_enabled=False)


def _so100() -> Any:
    from openral_hal.so100_mujoco import SO100MujocoHAL

    return SO100MujocoHAL(gravity_enabled=False)


def _franka() -> Any:
    from openral_hal.franka_panda import FrankaPandaHAL

    return FrankaPandaHAL(gravity_enabled=False)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
@pytest.mark.parametrize(
    "make_hal", [_openarm, _so100, _franka], ids=["openarm_v2", "so100", "franka_panda"]
)
def test_a_hal_without_the_attachment_api_still_heartbeats_an_empty_fresh_set(
    make_hal: Callable[[], Any],
) -> None:
    rclpy = pytest.importorskip("rclpy")

    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from openral_msgs.msg import AttachmentState
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    hal = make_hal()
    assert not hasattr(hal, "update_attached_objects"), "fixture must lack the attachment API"

    rclpy.init()
    try:
        node = Node(f"test_attachment_heartbeat_{hal.description.name}")
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
            bridge = SimSensorBridge(node, hal, hal.description, viewer_enabled=False)
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
