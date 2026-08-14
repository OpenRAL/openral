"""Live ROS coverage for the dashboard's camera perception overlays.

Two properties that only a real DDS graph can prove, because both are about
**QoS matching and the spin thread** rather than about the decoding code the
unit tests already cover:

1. **The subscription actually matches the detector.**
   `ros_image_detector_node` publishes BEST_EFFORT / VOLATILE / KEEP_LAST=5.
   A RELIABLE subscriber never matches a BEST_EFFORT publisher, so a drifted
   profile would leave the overlay permanently blank with nothing anywhere to
   explain it — the same silent-mismatch trap `/openral/safety_status`
   documents from the durability side. Only a real graph catches it: every
   in-process test drives the callback directly and passes either way.

2. **The message survives the whole path into `/api/state`.**
   Real `PromptStamped` on the real topic → the real `PerceptionOverlaySubscriber`
   spin thread → the real `TelemetryStore` → a real ASGI app instance. That is
   the path the browser reads, and it crosses a thread boundary the unit tests
   do not.

The segmenter half is covered here too, from a real `sensor_msgs/Image` mono8
mask built exactly as `openral_msgs/srv/SegmentInView` returns them. That
service does not stream, so there is no topic to subscribe to yet — but the
decode + store seam it will land on is real, and this pins it to the real ROS
message rather than to a bytes literal.

Real components throughout (CLAUDE.md §1.11): the production subscriber, the
production store, the production ASGI app, real generated messages. No mocks.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the sibling live tests, and listed in
``scripts/ros_live_tests.sh``. CI runs it inside ``openral:x86`` (the
``docker-build`` workflow). Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live            # whole suite; `-k perception_overlay` narrows it
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_MODEL_ID = "OpenRAL/rskill-omdet_turbo-any-indoor-fp16"
_SEGMENTER_ID = "OpenRAL/rskill-sam2_1-any-grasped_object_mask-bf16"


def _detector_qos() -> Any:
    """The detector node's publish profile: BEST_EFFORT / VOLATILE / KEEP_LAST=5.

    Sensor-class QoS per CLAUDE.md §2. Written out here rather than imported so
    a drift in the dashboard's subscriber shows up as a live failure instead of
    both sides moving together.
    """
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    """Poll until ``predicate()`` is true or the timeout expires.

    The subscriber spins on its OWN daemon thread (that is its production
    shape), so this waits rather than spinning an executor here.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _objects_metadata_json() -> str:
    """A real `ObjectsMetadata` document, as the detector puts on the wire."""
    from openral_core.schemas import ObjectDetection2D, ObjectsMetadata

    return ObjectsMetadata(
        sensor_id="top",
        detections=[
            ObjectDetection2D(
                label="mug", confidence=0.87, bbox_xyxy=(210, 140, 318, 266), det_id=4
            ),
            ObjectDetection2D(
                label="plate", confidence=0.62, bbox_xyxy=(20, 300, 190, 410), det_id=7
            ),
        ],
        model_id=_MODEL_ID,
        frame_width=640,
        frame_height=480,
    ).model_dump_json()


@contextmanager
def _live_overlay_subscriber() -> Iterator[tuple[Any, Any]]:
    """Bring up the real subscriber + store; yield (store, subscriber)."""
    from openral_observability.dashboard import TelemetryStore
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PerceptionOverlaySubscriber,
    )

    store = TelemetryStore()
    subscriber = PerceptionOverlaySubscriber(store)
    try:
        assert subscriber.available, (
            "the subscriber must come up on a live ROS host — an inert one would "
            "make every assertion below vacuous"
        )
        yield store, subscriber
    finally:
        subscriber.close()


def _api_state(store: Any) -> dict[str, Any]:
    """Read `/api/state` from a REAL app instance, the way the browser does."""
    import httpx
    from openral_observability.dashboard import create_app

    async def _get() -> dict[str, Any]:
        app = create_app(store)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/state")
        assert resp.status_code == 200
        return dict(resp.json())

    return asyncio.run(_get())


def test_real_detections_reach_api_state_over_a_live_topic() -> None:
    """A real `PromptStamped` on the real topic paints a real overlay.

    The QoS-match proof: this only passes if the dashboard's subscriber profile
    is compatible with the detector node's BEST_EFFORT publisher.
    """
    import rclpy
    from openral_msgs.msg import PromptStamped
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PERCEPTION_OBJECTS_TOPIC,
    )

    with _live_overlay_subscriber() as (store, _subscriber):
        publisher_node = rclpy.create_node("openral_perception_overlay_itest_pub")
        pub = publisher_node.create_publisher(
            PromptStamped, PERCEPTION_OBJECTS_TOPIC, _detector_qos()
        )
        try:
            # Let DDS discovery settle, then publish until it lands. BEST_EFFORT
            # has no retransmit, so a single pre-discovery publish is simply lost.
            assert _wait_until(lambda: pub.get_subscription_count() >= 1, timeout_s=5.0), (
                "the dashboard subscriber never matched the BEST_EFFORT publisher — "
                "check the QoS profile in perception_overlay_subscriber.py"
            )
            msg = PromptStamped()
            msg.text = "2 objects"
            msg.metadata_json = _objects_metadata_json()
            msg.header.frame_id = "top"
            msg.header.stamp.sec = 1717171717
            msg.header.stamp.nanosec = 250_000_000

            def _published_and_received() -> bool:
                pub.publish(msg)
                overlays = store.snapshot()["topics"]["perception"].get("overlays") or {}
                return "top" in overlays

            assert _wait_until(_published_and_received, timeout_s=5.0), (
                "the detections never reached the store over the live topic"
            )
        finally:
            publisher_node.destroy_node()

        # ...and the browser's view of it.
        state = _api_state(store)
        det = state["topics"]["perception"]["overlays"]["top"]["detections"]
        assert det["model_id"] == _MODEL_ID
        assert [b["label"] for b in det["boxes"]] == ["mug", "plate"]
        assert det["boxes"][0]["bbox_xyxy"] == [210, 140, 318, 266]
        assert det["boxes"][0]["confidence"] == pytest.approx(0.87)
        # Source-frame size, without which the renderer cannot scale the boxes
        # onto the aspect-preserved thumbnail the tile actually displays.
        assert det["frame_width"] == 640
        assert det["frame_height"] == 480
        # The source stamp survived the whole path, not the receipt time.
        assert det["stamp_unix"] == pytest.approx(1717171717.25)


def test_undecodable_payload_leaves_the_live_subscription_alive() -> None:
    """A bad message must not end overlays for the rest of the session.

    `ObjectsMetadata` is ``extra="forbid"``, so a producer newer than this
    dashboard raises in the callback. If that escaped, the spin thread would
    die and every later frame would be lost silently — worse than the one
    dropped frame, because the tile keeps looking healthy.
    """
    import rclpy
    from openral_msgs.msg import PromptStamped
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PERCEPTION_OBJECTS_TOPIC,
    )

    with _live_overlay_subscriber() as (store, _subscriber):
        publisher_node = rclpy.create_node("openral_perception_overlay_itest_bad_pub")
        pub = publisher_node.create_publisher(
            PromptStamped, PERCEPTION_OBJECTS_TOPIC, _detector_qos()
        )
        try:
            assert _wait_until(lambda: pub.get_subscription_count() >= 1, timeout_s=5.0)
            bad = PromptStamped()
            bad.metadata_json = '{"sensor_id":"top","kind":"objects","from_the_future":1}'
            for _ in range(5):
                pub.publish(bad)
                time.sleep(0.05)
            # Nothing rendered from the bad frame...
            assert (store.snapshot()["topics"]["perception"].get("overlays") or {}) == {}

            # ...and a GOOD frame still gets through afterwards.
            good = PromptStamped()
            good.metadata_json = _objects_metadata_json()
            good.header.stamp.sec = 42

            def _recovered() -> bool:
                pub.publish(good)
                return "top" in (store.snapshot()["topics"]["perception"].get("overlays") or {})

            assert _wait_until(_recovered, timeout_s=5.0), (
                "the subscription did not survive an undecodable payload"
            )
        finally:
            publisher_node.destroy_node()


def test_a_real_mono8_image_mask_renders_through_to_api_state() -> None:
    """The segmenter seam, driven by a real `sensor_msgs/Image` mono8 mask.

    `SegmentInView` returns masks as full-frame mono8 `sensor_msgs/Image`s —
    255 where the pixel belongs to the prompted object — plural and ordered
    area-ascending, with a parallel advisory score array. The service does not
    stream, so the dashboard has nothing to subscribe to yet; what this pins is
    that the decode + store path is written against the real message, so a mask
    publisher lands on a working renderer rather than on a guess.
    """
    from openral_observability.dashboard.perception_overlay_subscriber import (
        mono8_mask_to_png_b64,
    )
    from sensor_msgs.msg import Image

    width, height = 64, 48
    # A real mask: the centre quarter set, exactly as the producer writes it.
    payload = bytearray(width * height)
    for y in range(height // 4, (3 * height) // 4):
        for x in range(width // 4, (3 * width) // 4):
            payload[y * width + x] = 255

    mask = Image()
    mask.header.frame_id = "wrist_camera_optical_frame"
    mask.header.stamp.sec = 1717171717
    mask.header.stamp.nanosec = 250_000_000
    mask.height = height
    mask.width = width
    mask.encoding = "mono8"
    mask.is_bigendian = 0
    mask.step = width
    mask.data = bytes(payload)

    # The service's own invariants, asserted on the real message.
    assert mask.encoding == "mono8"
    assert mask.step == mask.width
    assert len(bytes(mask.data)) == mask.width * mask.height

    with _live_overlay_subscriber() as (store, _subscriber):
        store.set_perception_masks(
            camera="wrist",
            masks=[
                {
                    "png_b64": mono8_mask_to_png_b64(bytes(mask.data), mask.width, mask.height),
                    "width": int(mask.width),
                    "height": int(mask.height),
                    "score_advisory": 0.977,
                }
            ],
            rskill_id=_SEGMENTER_ID,
            stamp_unix=float(mask.header.stamp.sec) + float(mask.header.stamp.nanosec) * 1e-9,
        )
        state = _api_state(store)
        lane = state["topics"]["perception"]["overlays"]["wrist"]["masks"]
        assert lane["rskill_id"] == _SEGMENTER_ID
        assert lane["stamp_unix"] == pytest.approx(1717171717.25)
        assert len(lane["masks"]) == 1
        entry = lane["masks"][0]
        assert (entry["width"], entry["height"]) == (width, height)
        # The advisory score is carried for DISPLAY and never used to rank or
        # filter — 0.977 is the score the service's own docs record for a
        # measured whole-tablecloth mask.
        assert entry["score_advisory"] == pytest.approx(0.977)

        # The decoded PNG's alpha IS the mask, at the source frame's own size.
        import base64
        import io

        from PIL import Image as PilImage

        decoded = PilImage.open(io.BytesIO(base64.b64decode(entry["png_b64"])))
        assert decoded.mode == "LA"
        assert decoded.size == (width, height)
        alpha = decoded.split()[1]
        assert alpha.getpixel((width // 2, height // 2)) == 255
        assert alpha.getpixel((0, 0)) == 0
