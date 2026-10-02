# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the vision attachment leg heartbeats its set, only while effort evidence is live.

The kernel's attached-payload check fails closed on an attachment snapshot it
has never heard (``attachment_stamp_ns == 0``) or one older than its deadline.
The vision leg used to publish only on events, so with that check on every chunk
dropped before the first grasp and again a deadline after each event; and its
revision restarted at 0 on every HAL activate, which the world-state aggregator
rejects as moving backwards.

The heartbeat is a claim about the jaws, so it is gated: nothing is published
until every gripper's effort channel has reported for
``GraspTriggerConfig.consecutive_ticks`` samples in a row, and it stops within
``evidence_timeout_s`` of the channel going quiet — a dead channel becomes a
kernel drop, never a stale "nothing attached".

Real rclpy, real ``VisionAttachmentBridge`` on the bimanual OpenArm manifest,
real ``openral_msgs`` on the wire, real tf2 and a real depth frame. The
segmenter is a ``SegmentInView`` service offered and never answered — a real DDS
peer at a process boundary (CLAUDE.md §1.11), the wedged-perception case the
attach deadline exists for. Every received snapshot is fed into a real
``WorldStateAggregator``, which raises on a revision that moves backwards.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_vision_attachment_heartbeat_live.py
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from openral_core import CameraTopicKind, camera_topic

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_ROBOT_YAML = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_CAMERA = "head_zed"
_SERVICE = "/openral/perception/segment_in_view_heartbeat_itest"
_ATTACH_DEADLINE_S = 0.5
_EVIDENCE_TIMEOUT_S = 0.5
_PERIOD_S = 0.2
_FEED_PERIOD_S = 1.0 / 30.0


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    """Poll a live-graph predicate (the executor spins on its own thread)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_vision_leg_heartbeats_only_on_live_effort_with_a_monotonic_revision() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    import numpy as np
    from geometry_msgs.msg import TransformStamped
    from openral_core import AttachedCollisionObject, JointState, RobotDescription
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import AttachmentState
    from openral_msgs.srv import SegmentInView
    from openral_world_state import WorldStateAggregator
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image
    from tf2_ros import StaticTransformBroadcaster

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    left = next(j for j in description.joints if j.name == "left_gripper")
    camera = next(spec for spec in description.sensors if spec.name == _CAMERA)
    depth_topic = camera_topic(_CAMERA, CameraTopicKind.DEPTH_IMAGE)
    config = VisionAttachmentConfig(
        camera=_CAMERA,
        depth_topic=depth_topic,
        service_name=_SERVICE,
        deadline_s=_ATTACH_DEADLINE_S,
        evidence_timeout_s=_EVIDENCE_TIMEOUT_S,
    )

    rclpy.init()
    node = Node("test_vision_attachment_heartbeat_hal")
    peer = Node("test_vision_attachment_heartbeat_peer")
    # Offered so the client is ready, and NEVER spun: the attach deadline settles it.
    unanswered = Node("test_vision_attachment_heartbeat_segmenter")
    unanswered.create_service(SegmentInView, _SERVICE, lambda _req, resp: resp)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)

    aggregator = WorldStateAggregator(description)
    aggregator_errors: list[str] = []
    # (receipt monotonic s, samples fed so far, revision, stamp ns, object count)
    received: list[tuple[float, int, int, int, int]] = []
    feed: dict[str, Any] = {"on": False, "left": 0.01, "count": 0, "bridge": None}

    def on_state(msg: Any) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        received.append(
            (time.monotonic(), feed["count"], int(msg.revision), stamp_ns, len(msg.objects))
        )
        try:
            aggregator.update_attached_objects(
                [AttachedCollisionObject.from_idl(item) for item in msg.objects],
                revision=int(msg.revision),
                stamp_ns=stamp_ns,
            )
        except ValueError as exc:
            aggregator_errors.append(str(exc))

    peer.create_subscription(
        AttachmentState,
        "/openral/attachment_state",
        on_state,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )

    # The HAL feeds observe_joint_state from its own executor; so does this
    # feeder, so the bridge's callbacks never race the test thread.
    def on_feed() -> None:
        bridge = feed["bridge"]
        if not feed["on"] or bridge is None:
            return
        feed["count"] += 1
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[0.0, 0.0],
                effort=[feed["left"], 0.01],
                stamp_ns=time.time_ns(),
            )
        )

    node.create_timer(_FEED_PERIOD_S, on_feed)

    def since(index: int) -> list[tuple[float, int, int, int, int]]:
        return received[index:]

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    first = second = None
    try:
        first = VisionAttachmentBridge(node, description, config=config)
        feed["bridge"] = first
        first.setup()

        # ── (a) No effort yet: nothing is claimed, not even an empty set. ─────
        time.sleep(4 * _PERIOD_S)
        assert received == [], "the heartbeat spoke before any effort evidence"
        feed["on"] = True
        assert _wait_until(lambda: bool(received)), "no heartbeat once effort was live"
        assert received[0][1] >= 3, (
            f"published after {received[0][1]} samples; the debounce needs "
            "consecutive_ticks=3 complete samples first"
        )

        # ── (b) ~5 Hz, empty, stamp advancing, revision constant. ────────────
        start = len(received)
        time.sleep(6 * _PERIOD_S)
        window = since(start)
        assert 4 <= len(window) <= 8, f"{len(window)} heartbeats in {6 * _PERIOD_S:.1f} s"
        assert all(count == 0 for *_, count in window)
        assert len({rev for _, _, rev, _, _ in window}) == 1, "a heartbeat bumped the revision"
        stamps = [stamp for _, _, _, stamp, _ in window]
        assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)

        # ── (c) The effort channel goes quiet: the heartbeat stops. ──────────
        feed["on"] = False
        quiet_at = time.monotonic()
        time.sleep(_EVIDENCE_TIMEOUT_S + 4 * _PERIOD_S)
        last_heard = received[-1][0]
        assert last_heard <= quiet_at + _EVIDENCE_TIMEOUT_S + _PERIOD_S + 0.1, (
            f"still heartbeating {last_heard - quiet_at:.2f} s after effort went quiet"
        )

        # ── (d) ATTACH resolves to the fallback box on the attach deadline,
        #        and the heartbeat keeps republishing it at that revision. ────
        broadcaster = StaticTransformBroadcaster(peer)
        hop = TransformStamped()
        hop.header.frame_id = left.parent_link
        hop.child_frame_id = str(camera.frame_id)
        hop.transform.translation.z = 0.3
        hop.transform.rotation.w = 1.0
        broadcaster.sendTransform([hop])
        depth_pub = peer.create_publisher(
            Image,
            depth_topic,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
        depth = Image()
        depth.header.frame_id = str(camera.frame_id)
        depth.height, depth.width = 8, 8
        depth.encoding = "32FC1"
        depth.step = 4 * depth.width
        depth.data = np.full((8, 8), 0.25, dtype=np.float32).tobytes()

        def depth_cached() -> bool:
            depth.header.stamp = peer.get_clock().now().to_msg()
            depth_pub.publish(depth)
            return first._depth is not None

        assert _wait_until(depth_cached), "the bridge never cached a depth frame"
        assert _wait_until(first._client.service_is_ready), "SegmentInView never discovered"

        empty_revision = received[-1][2]
        start = len(received)
        feed["left"] = 0.9 * float(left.effort_limit)
        loaded_at = time.monotonic()
        feed["on"] = True
        assert _wait_until(lambda: any(count == 1 for *_, count in since(start))), (
            "the attach never published the fallback box"
        )
        attach = next(row for row in since(start) if row[4] == 1)
        assert attach[0] - loaded_at >= _ATTACH_DEADLINE_S, (
            "the box arrived before the attach deadline — the request never went to "
            "the (unanswered) service"
        )
        assert attach[2] == empty_revision + 1
        time.sleep(5 * _PERIOD_S)
        after = [row for row in since(start) if row[0] >= attach[0]]
        assert len(after) >= 4, f"only {len(after)} snapshots after the attach"
        assert all(row[4] == 1 and row[2] == attach[2] for row in after), (
            "the heartbeat dropped or re-revisioned the held box"
        )

        # ── (e) Re-activate: the next bridge's revision is greater. ──────────
        feed["on"] = False
        feed["left"] = 0.01
        first.teardown()
        last_revision = max(row[2] for row in received)
        start = len(received)
        second = VisionAttachmentBridge(node, description, config=config)
        feed["bridge"] = second
        second.setup()
        feed["on"] = True
        assert _wait_until(lambda: bool(since(start))), "the second activation never published"
        assert since(start)[0][2] > last_revision, (
            f"revision moved backwards across a re-activate: {since(start)[0][2]} "
            f"<= {last_revision}"
        )

        assert aggregator_errors == [], aggregator_errors
    finally:
        feed["on"] = False
        for bridge in (first, second):
            if bridge is not None:
                with suppress(Exception):
                    bridge.teardown()
        executor.shutdown()
        spin.join(timeout=5.0)
        unanswered.destroy_node()
        peer.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
