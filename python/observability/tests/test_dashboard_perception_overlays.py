"""The dashboard's camera perception overlays — detector boxes + segmenter masks.

The camera tiles already show what the robot sees; these overlays show what it
made of it. Two producers, two very different upstream shapes:

* **Detectors** stream. `ros_image_detector_node` publishes
  `openral_msgs/PromptStamped` on `/openral/perception/objects` carrying an
  `openral_core.ObjectsMetadata` JSON document, and `PerceptionOverlaySubscriber`
  decodes it into the store.
* **Segmenters** do not stream. `openral_msgs/srv/SegmentInView` returns mono8
  masks one-shot per attach event, so the store API and the mask decoder are
  built to that convention and wait for a publisher.

Split the way `test_dashboard_safety_status.py` splits: the rclpy half needs a
ROS workspace CI's cheap env does not have, so it is import-gated; the store +
snapshot contract the frontend reads is covered unconditionally. The UI contract
is pinned by asserting the exact snapshot keys `dashboard.js`'s `drawOverlay`
reads, which is how the rest of this suite pins card behaviour (no test parses
the JS).

No mocks (CLAUDE.md §1.11): the store and the ASGI app are the production
classes, the detection payloads are real `ObjectsMetadata` documents, and the
mask bytes are a real mono8 buffer in the service's own encoding.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time

import httpx
import pytest
from openral_core.schemas import ObjectDetection2D, ObjectsMetadata
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.perception_overlay_subscriber import (
    PERCEPTION_OBJECTS_TOPIC,
    dashboard_flip_180,
    mono8_mask_to_png_b64,
)

# The keys dashboard.js's drawOverlay() reads off
# ``snapshot()["topics"]["perception"]["overlays"][camera]``. Renaming one
# without touching the JS silently blanks the overlay while the camera tile
# keeps looking healthy — the operator then believes perception saw nothing.
_DETECTION_UI_KEYS = {
    "boxes",
    "model_id",
    "frame_width",
    "frame_height",
    "stamp_unix",
    "flip_180",
    "ts_unix",
}
_BOX_UI_KEYS = {"label", "confidence", "bbox_xyxy", "det_id"}
_MASK_UI_KEYS = {"masks", "rskill_id", "stamp_unix", "flip_180", "ts_unix"}

# A real detector payload: the omdet-turbo indoor rSkill's manifest name is what
# `build_manifest_detector` puts on the wire as `model_id`.
_MODEL_ID = "OpenRAL/rskill-omdet_turbo-any-indoor-fp16"
_SEGMENTER_ID = "OpenRAL/rskill-sam2_1-any-grasped_object_mask-bf16"


def _objects_metadata() -> ObjectsMetadata:
    """A real `ObjectsMetadata` as the detector node would publish it."""
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
    )


def _set_detections(store: TelemetryStore, **overrides: object) -> None:
    """Write one detector result into the store as the subscriber would."""
    meta = _objects_metadata()
    payload: dict[str, object] = {
        "camera": meta.sensor_id,
        "detections": [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "bbox_xyxy": list(d.bbox_xyxy),
                "det_id": int(d.det_id),
            }
            for d in meta.detections
        ],
        "model_id": meta.model_id,
        "frame_width": meta.frame_width,
        "frame_height": meta.frame_height,
        "stamp_unix": time.time(),
    }
    payload.update(overrides)
    store.set_perception_detections(**payload)  # type: ignore[arg-type]  # reason: kwargs built dynamically


def _mono8_mask(width: int, height: int) -> bytes:
    """A real mono8 mask in the `SegmentInView` encoding: 255 set, 0 clear."""
    rows = []
    for y in range(height):
        row = bytearray(width)
        if height // 4 <= y < (3 * height) // 4:
            for x in range(width // 4, (3 * width) // 4):
                row[x] = 255
        rows.append(bytes(row))
    return b"".join(rows)


# --------------------------------------------------------------------------
# Store contract
# --------------------------------------------------------------------------


def test_overlays_start_empty_not_absent_detections() -> None:
    """With no detector seen, there are no overlays — never fabricated boxes.

    "We have not heard from the detector" and "the detector found nothing" are
    different facts. The card draws the bare camera tile for the first, and the
    store must never hand it the second by default (CLAUDE.md §1.2).
    """
    store = TelemetryStore()
    assert store.snapshot()["topics"]["perception"].get("overlays") is None


def test_detections_reach_the_snapshot_with_every_ui_key() -> None:
    """One detector result surfaces with the full field set the renderer reads."""
    store = TelemetryStore()
    _set_detections(store)
    overlays = store.snapshot()["topics"]["perception"]["overlays"]
    assert set(overlays) == {"top"}
    det = overlays["top"]["detections"]
    assert set(det) == _DETECTION_UI_KEYS
    assert det["model_id"] == _MODEL_ID
    # Source-frame dimensions are load-bearing: the tile displays an
    # aspect-preserving 320x240 THUMBNAIL of a 640x480 frame, so the renderer
    # scales by ratio. Without these the boxes would be drawn at source pixel
    # offsets on a half-size image — visibly plausible and wrong.
    assert det["frame_width"] == 640
    assert det["frame_height"] == 480
    assert set(det["boxes"][0]) == _BOX_UI_KEYS
    assert det["boxes"][0]["label"] == "mug"
    assert det["boxes"][0]["bbox_xyxy"] == [210, 140, 318, 266]
    # Receipt stamp on the dashboard's own clock — the source stamp rides sim
    # time in a sim deploy, so it cannot drive the staleness fade.
    assert det["ts_unix"] > 0


def test_overlay_camera_key_matches_the_camera_tile_key() -> None:
    """Overlays key off the same camera name the tiles do, or nothing lines up.

    ``ObjectsMetadata.sensor_id`` is a ``SensorSpec`` name, and so is the
    ``openral.sensors.source`` attribute the camera tiles are keyed by. That
    shared namespace is the entire join between a frame and its boxes.
    """
    store = TelemetryStore()
    store._topics["perception"].setdefault("cameras", {})["top"] = {"ts_unix": time.time()}
    _set_detections(store)
    perception = store.snapshot()["topics"]["perception"]
    assert set(perception["overlays"]) <= set(perception["cameras"])


def test_a_newer_result_replaces_the_previous_boxes() -> None:
    """Detections are current state, not a log — the old boxes must not linger."""
    store = TelemetryStore()
    _set_detections(store)
    _set_detections(
        store,
        detections=[{"label": "bowl", "confidence": 0.4, "bbox_xyxy": [1, 2, 3, 4], "det_id": -1}],
    )
    boxes = store.snapshot()["topics"]["perception"]["overlays"]["top"]["detections"]["boxes"]
    assert [b["label"] for b in boxes] == ["bowl"]


def test_an_empty_detection_set_is_recorded_not_dropped() -> None:
    """A detector that reports nothing must be able to CLEAR the overlay.

    The node does not publish empty results, but `locate_in_view` misses and a
    future producer can, and "no boxes" has to be expressible — otherwise the
    last non-empty result would stay painted over a scene that no longer has
    the object in it.
    """
    store = TelemetryStore()
    _set_detections(store)
    _set_detections(store, detections=[])
    assert store.snapshot()["topics"]["perception"]["overlays"]["top"]["detections"]["boxes"] == []


def test_two_cameras_keep_independent_overlays() -> None:
    """A second camera's boxes must not overwrite the first camera's."""
    store = TelemetryStore()
    _set_detections(store)
    _set_detections(store, camera="wrist", frame_width=320, frame_height=240)
    overlays = store.snapshot()["topics"]["perception"]["overlays"]
    assert set(overlays) == {"top", "wrist"}
    assert overlays["top"]["detections"]["frame_width"] == 640
    assert overlays["wrist"]["detections"]["frame_width"] == 320


def test_flip_180_rides_with_the_overlay() -> None:
    """The display flip is carried per-overlay, not assumed by the renderer.

    Detectors consume the RAW camera topic while `OPENRAL_DASHBOARD_FLIP_180`
    rotates only the dashboard's display copy. An overlay drawn without the
    same correction lands point-mirrored on a picture that looks right.
    """
    store = TelemetryStore()
    _set_detections(store, flip_180=True)
    assert store.snapshot()["topics"]["perception"]["overlays"]["top"]["detections"]["flip_180"]


def test_masks_reach_the_snapshot_with_every_ui_key() -> None:
    """The segmenter lane carries decoded masks + the advisory scores."""
    store = TelemetryStore()
    png = mono8_mask_to_png_b64(_mono8_mask(64, 48), 64, 48)
    store.set_perception_masks(
        camera="wrist",
        masks=[
            {"png_b64": png, "width": 64, "height": 48, "score_advisory": 0.61},
            {"png_b64": png, "width": 64, "height": 48, "score_advisory": 0.98},
        ],
        rskill_id=_SEGMENTER_ID,
        stamp_unix=time.time(),
    )
    lane = store.snapshot()["topics"]["perception"]["overlays"]["wrist"]["masks"]
    assert set(lane) == _MASK_UI_KEYS
    assert lane["rskill_id"] == _SEGMENTER_ID
    assert len(lane["masks"]) == 2
    assert lane["masks"][0]["width"] == 64


def test_mask_order_is_preserved_exactly_as_the_producer_sent_it() -> None:
    """`SegmentInView` orders masks area-ascending and NEVER by score.

    The service documents its own IoU estimate as advisory — a mis-aimed prompt
    was measured returning a 59.8%-of-frame mask at that model's top score of
    0.977. Re-sorting here by score would quietly promote exactly that mask to
    the front of the render order, so the store keeps the producer's order.
    """
    store = TelemetryStore()
    png = mono8_mask_to_png_b64(_mono8_mask(8, 8), 8, 8)
    store.set_perception_masks(
        camera="wrist",
        masks=[
            {"png_b64": png, "width": 8, "height": 8, "score_advisory": 0.30},
            {"png_b64": png, "width": 8, "height": 8, "score_advisory": 0.97},
            {"png_b64": png, "width": 8, "height": 8, "score_advisory": 0.55},
        ],
        rskill_id=_SEGMENTER_ID,
        stamp_unix=time.time(),
    )
    lane = store.snapshot()["topics"]["perception"]["overlays"]["wrist"]["masks"]
    assert [m["score_advisory"] for m in lane["masks"]] == [0.30, 0.97, 0.55]


def test_detections_and_masks_coexist_on_one_camera() -> None:
    """Both lanes render together; writing one must not erase the other."""
    store = TelemetryStore()
    _set_detections(store)
    store.set_perception_masks(
        camera="top",
        masks=[
            {
                "png_b64": mono8_mask_to_png_b64(_mono8_mask(8, 8), 8, 8),
                "width": 8,
                "height": 8,
                "score_advisory": None,
            }
        ],
        rskill_id=_SEGMENTER_ID,
        stamp_unix=time.time(),
    )
    entry = store.snapshot()["topics"]["perception"]["overlays"]["top"]
    assert set(entry) == {"detections", "masks"}
    _set_detections(store)
    assert "masks" in store.snapshot()["topics"]["perception"]["overlays"]["top"]


def test_overlay_update_reaches_sse_subscribers() -> None:
    """Overlays are pushed, not polled — they must track the frame, not a timer."""

    async def _run() -> dict[str, object]:
        store = TelemetryStore()
        queue = store.subscribe()
        try:
            _set_detections(store)
            payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        finally:
            store.unsubscribe(queue)
        return dict(payload)

    payload = asyncio.run(_run())
    overlays = payload["topics"]["perception"]["overlays"]  # type: ignore[index]  # reason: snapshot is a plain nested dict
    assert overlays["top"]["detections"]["boxes"][0]["label"] == "mug"


# --------------------------------------------------------------------------
# The mono8 mask decoder — the segmenter seam
# --------------------------------------------------------------------------


def test_mono8_mask_encodes_to_a_tintable_png() -> None:
    """The mask becomes an LA PNG whose ALPHA is the mask.

    That is what lets the frontend tint it with one `source-in` composite
    instead of decoding pixels in JS. Verified by decoding the PNG back and
    checking alpha, not by trusting the encoder's word for it.
    """
    from PIL import Image

    data = _mono8_mask(16, 12)
    png = base64.b64decode(mono8_mask_to_png_b64(data, 16, 12))
    img = Image.open(io.BytesIO(png))
    assert img.mode == "LA"
    assert img.size == (16, 12)
    alpha = img.split()[1]
    # Inside the mask rectangle (rows 3..8, cols 4..11) alpha is set; outside clear.
    assert alpha.getpixel((8, 6)) == 255
    assert alpha.getpixel((0, 0)) == 0


def test_mono8_mask_treats_any_non_zero_pixel_as_set() -> None:
    """Tolerant like the HAL-side decoder: the producer writes 255, but a mask
    that survived a lossy hop must still render rather than vanish."""
    from PIL import Image

    png = base64.b64decode(mono8_mask_to_png_b64(bytes([0, 1, 200, 255]), 2, 2))
    alpha = Image.open(io.BytesIO(png)).split()[1]
    assert [alpha.getpixel((x, y)) for y in range(2) for x in range(2)] == [0, 255, 255, 255]


def test_truncated_mono8_mask_fails_loudly() -> None:
    """A short buffer must raise, not render a plausible but wrong shape."""
    with pytest.raises(ValueError, match="expected 48"):
        mono8_mask_to_png_b64(bytes(10), 8, 6)


def test_dashboard_flip_180_reads_the_repo_wide_env_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same parsing as `openral_rskill_ros.sensor_leg` and the world-state node."""
    monkeypatch.delenv("OPENRAL_DASHBOARD_FLIP_180", raising=False)
    assert dashboard_flip_180() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("OPENRAL_DASHBOARD_FLIP_180", truthy)
        assert dashboard_flip_180() is True
    monkeypatch.setenv("OPENRAL_DASHBOARD_FLIP_180", "0")
    assert dashboard_flip_180() is False


# --------------------------------------------------------------------------
# HTTP surface — a real app instance
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlays_are_served_on_api_state() -> None:
    """The frontend's only fetch path carries the overlays end to end."""
    store = TelemetryStore()
    _set_detections(store)
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/state")
    assert resp.status_code == 200
    overlays = resp.json()["topics"]["perception"]["overlays"]
    assert overlays["top"]["detections"]["model_id"] == _MODEL_ID
    assert overlays["top"]["detections"]["boxes"][1]["label"] == "plate"


@pytest.mark.asyncio
async def test_app_starts_without_an_overlay_subscriber() -> None:
    """No ROS workspace → no subscriber, and the dashboard is unaffected.

    The camera tiles must keep working on a laptop with nothing but a trace
    file; the overlay is an addition to that surface, never a dependency of it.
    """
    app = create_app(TelemetryStore())
    assert app.state.perception_overlay is None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/state")
    assert resp.status_code == 200
    assert resp.json()["topics"]["perception"] == {}


# --------------------------------------------------------------------------
# The rclpy half — needs the colcon `openral_msgs` overlay
# --------------------------------------------------------------------------


def test_subscriber_writes_a_real_detection_message_into_the_store() -> None:
    """A real `PromptStamped` carrying real `ObjectsMetadata` lands as overlays.

    Drives `PerceptionOverlaySubscriber._on_objects` with the real generated
    message (no ROS graph needed for the store-write path, which is the half
    that can break silently). Skips without the `openral_msgs` overlay.
    """
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import PromptStamped
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PerceptionOverlaySubscriber,
    )

    store = TelemetryStore()
    subscriber = PerceptionOverlaySubscriber.__new__(PerceptionOverlaySubscriber)
    subscriber._store = store  # reason: exercising the callback without a ROS graph

    meta = _objects_metadata()
    msg = PromptStamped()
    msg.text = f"{len(meta.detections)} objects"
    msg.metadata_json = meta.model_dump_json()
    msg.header.frame_id = meta.sensor_id
    msg.header.stamp.sec = 1234
    msg.header.stamp.nanosec = 500_000_000
    subscriber._on_objects(msg)  # reason: the subscription callback is the unit under test

    det = store.snapshot()["topics"]["perception"]["overlays"]["top"]["detections"]
    assert det["model_id"] == _MODEL_ID
    assert [b["label"] for b in det["boxes"]] == ["mug", "plate"]
    assert det["boxes"][0]["det_id"] == 4
    assert det["stamp_unix"] == pytest.approx(1234.5)


def test_subscriber_drops_an_undecodable_payload_without_dying() -> None:
    """`ObjectsMetadata` is extra="forbid"; a newer producer must not kill the spin.

    The alternative — letting the exception escape the callback — takes the
    whole subscription down, so one unknown field would silently end overlays
    for the rest of the session.
    """
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import PromptStamped
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PerceptionOverlaySubscriber,
    )

    store = TelemetryStore()
    subscriber = PerceptionOverlaySubscriber.__new__(PerceptionOverlaySubscriber)
    subscriber._store = store  # reason: exercising the callback without a ROS graph

    msg = PromptStamped()
    msg.metadata_json = '{"sensor_id":"top","kind":"objects","unknown_future_field":1}'
    subscriber._on_objects(msg)  # must not raise

    assert store.snapshot()["topics"]["perception"].get("overlays") is None


def test_objects_topic_name_matches_the_detector_node() -> None:
    """The topic is the detector node's `output_topic` default, not a guess."""
    assert PERCEPTION_OBJECTS_TOPIC == "/openral/perception/objects"
