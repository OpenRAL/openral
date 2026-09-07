"""Live ROS 2 subscriber feeding the dashboard's camera perception overlays.

Camera tiles already show what the robot sees (OTel `sensors.read_latest`
spans carry a JPEG thumbnail, re-served as MJPEG by
`/api/camera/<src>/stream`); this module adds a `kind: detector` rSkill's
boxes and a `kind: segmenter` rSkill's masks, drawn over those tiles instead
of shown only as event-log text.

Follows `safety_status_subscriber.py`'s shape: one node created at launch,
spun on a daemon thread, inert when rclpy / `openral_msgs` is unavailable (no
overlays drawn, nothing else breaks).

* **Detector boxes** — `openral_msgs/PromptStamped` on
  `/openral/perception/objects`, an `openral_core.ObjectsMetadata` JSON doc.
* **Segmenter masks** — `openral_msgs/SegmentMasks` on
  `/openral/perception/masks`: the segmenter node's diagnostic
  re-publication of its `SegmentInView` service replies (off by default
  behind `publish_debug_masks`, so subscribing is free when unused). Decoded
  with :func:`mono8_mask_to_png_b64` into `TelemetryStore.set_perception_masks`.

Read-only: subscribes and writes to the store only — no publisher, no
service client, no robot authority. Overlays are advisory display only.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from openral_observability.dashboard.store import TelemetryStore

__all__ = [
    "PERCEPTION_MASKS_TOPIC",
    "PERCEPTION_OBJECTS_TOPIC",
    "PerceptionOverlaySubscriber",
    "dashboard_flip_180",
    "mono8_mask_to_png_b64",
]

_logger = structlog.get_logger(__name__)

PERCEPTION_OBJECTS_TOPIC = "/openral/perception/objects"
PERCEPTION_MASKS_TOPIC = "/openral/perception/masks"

# `ros_image_detector_node` publishes BEST_EFFORT / VOLATILE / KEEP_LAST=5. A
# RELIABLE subscriber would never match it at all — CLAUDE.md §2's sensor-class
# QoS, and the same "must match the publisher" trap the safety-status subscriber
# documents from the other direction.
_OBJECTS_QOS_DEPTH = 5

# `segmenter_node` publishes BEST_EFFORT / VOLATILE / KEEP_LAST=1: masks arrive
# one shot per attach event and only the newest is worth drawing. Same matching
# trap as above — the depth may differ from the publisher's, the reliability and
# durability may not.
_MASKS_QOS_DEPTH = 1


def dashboard_flip_180() -> bool:
    """Whether this host rotates the dashboard's display copy of camera frames.

    ``OPENRAL_DASHBOARD_FLIP_180`` is a repo-wide convention: HAL frames
    (LIBERO/MuJoCo) publish bottom-up and only the dashboard thumbnail is
    rotated, so an overlay must carry the same flip or it renders
    point-mirrored (CLAUDE.md §1.2).

    Returns:
        True when the env var is set to a truthy value, matching
        ``openral_rskill_ros.sensor_leg`` and the world-state node's parsing.

    Example:
        >>> import os
        >>> os.environ["OPENRAL_DASHBOARD_FLIP_180"] = "1"
        >>> dashboard_flip_180()
        True
        >>> del os.environ["OPENRAL_DASHBOARD_FLIP_180"]
        >>> dashboard_flip_180()
        False
    """
    return os.environ.get("OPENRAL_DASHBOARD_FLIP_180", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def mono8_mask_to_png_b64(data: bytes, width: int, height: int) -> str:
    """Encode one `sensor_msgs/Image` mono8 mask as a tintable base64 PNG.

    `SegmentInView` returns each candidate as a full-frame mono8 image: 255
    where the pixel belongs to the prompted object, 0 elsewhere (any non-zero
    pixel counts as set, tolerating a lossy hop).

    Emitted as an **LA** PNG whose alpha channel *is* the mask: the frontend
    draws the PNG then fills with the instance colour using
    ``globalCompositeOperation = "source-in"``, landing on exactly the set
    pixels. Binary alpha also compresses best — a 640x480 mask costs a few kB,
    the same order as the JPEG thumbnails already riding in the snapshot.

    Args:
        data: The mono8 payload, row-major, ``width`` bytes per row.
        width: Mask width in pixels; must match the source frame's own width.
        height: Mask height in pixels.

    Returns:
        The PNG bytes, base64-encoded ASCII, ready for a ``data:`` URI.

    Raises:
        ValueError: If ``data`` is not exactly ``width * height`` bytes — a
            truncated mask must fail loudly rather than render as a plausible
            but wrong shape.

    Example:
        >>> png = mono8_mask_to_png_b64(bytes([0, 255, 255, 0]), 2, 2)
        >>> png.startswith("iVBOR")
        True
    """
    expected = int(width) * int(height)
    if len(data) != expected:
        raise ValueError(
            f"mono8 mask is {len(data)} bytes, expected {expected} for {width}x{height}"
        )
    from PIL import Image

    alpha = Image.frombytes("L", (int(width), int(height)), bytes(data))
    # Any non-zero pixel is set; normalise to a clean binary alpha so a mask
    # that arrived with soft edges tints uniformly.
    alpha = alpha.point(lambda v: 255 if v else 0)
    luma = Image.new("L", alpha.size, 255)
    buf = io.BytesIO()
    Image.merge("LA", (luma, alpha)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class PerceptionOverlaySubscriber:
    """Feeds the store's camera overlays from the two perception topics."""

    def __init__(self, store: TelemetryStore) -> None:
        """Create the node + subscriptions now (at launch) so DDS discovers early.

        Args:
            store: The dashboard's telemetry store; every decoded detection set
                is written to it via
                :meth:`TelemetryStore.set_perception_detections`, and every
                decoded mask set via
                :meth:`TelemetryStore.set_perception_masks`.
        """
        self._store = store
        self._node: Any = None
        self._subscription: Any = None
        self._masks_subscription: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_rclpy = False
        try:
            self._start()
        except Exception:  # reason: a ROS hiccup must never break the dashboard
            self.close()

    def _start(self) -> None:
        import rclpy
        from openral_msgs.msg import PromptStamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = Node("openral_dashboard_perception_overlay")
        # Must MATCH the detector node's sensor-class profile (see above).
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=_OBJECTS_QOS_DEPTH,
        )
        self._subscription = self._node.create_subscription(
            PromptStamped, PERCEPTION_OBJECTS_TOPIC, self._on_objects, qos
        )
        # The mask half. Its own try/except: `SegmentMasks` is newer than
        # `PromptStamped`, so a dashboard running against an older `openral_msgs`
        # overlay must keep drawing boxes rather than lose both legs.
        try:
            from openral_msgs.msg import SegmentMasks

            self._masks_subscription = self._node.create_subscription(
                SegmentMasks,
                PERCEPTION_MASKS_TOPIC,
                self._on_masks,
                QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=_MASKS_QOS_DEPTH,
                ),
            )
        except ImportError:
            _logger.info(
                "dashboard.perception_mask_overlay_unavailable",
                topic=PERCEPTION_MASKS_TOPIC,
                reason="openral_msgs/msg/SegmentMasks is not in the sourced overlay",
            )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="openral_dashboard_perception_overlay_spin",
            daemon=True,
        )
        self._thread.start()

    @property
    def available(self) -> bool:
        """True when the subscription is live (rclpy + `openral_msgs` present)."""
        return self._subscription is not None

    @property
    def masks_available(self) -> bool:
        """True when the mask subscription is live (needs `SegmentMasks` built).

        Separate from :attr:`available` because the two legs can differ: an
        older `openral_msgs` overlay carries `PromptStamped` but not
        `SegmentMasks`, and losing masks must not read as losing overlays.
        """
        return self._masks_subscription is not None

    def _on_objects(self, msg: Any) -> None:  # reason: ROS message is untyped
        """Write one decoded detection set into the store (on the spin thread)."""
        from openral_core.schemas import ObjectsMetadata

        try:
            meta = ObjectsMetadata.model_validate_json(msg.metadata_json)
        except Exception as exc:  # reason: a future producer's extra key must not kill the spin
            # `ObjectsMetadata` is extra="forbid", so a newer producer field
            # raises here. Drop the frame and say so — never render a partially
            # decoded overlay, and never take the dashboard down for it.
            _logger.warning(
                "dashboard.perception_overlay_decode_failed",
                topic=PERCEPTION_OBJECTS_TOPIC,
                error=str(exc),
            )
            return
        stamp = msg.header.stamp
        self._store.set_perception_detections(
            camera=meta.sensor_id,
            detections=[
                {
                    "label": d.label,
                    "confidence": float(d.confidence),
                    "bbox_xyxy": list(d.bbox_xyxy),
                    "det_id": int(d.det_id),
                }
                for d in meta.detections
            ],
            model_id=meta.model_id,
            frame_width=meta.frame_width,
            frame_height=meta.frame_height,
            stamp_unix=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            flip_180=dashboard_flip_180(),
        )

    def _on_masks(self, msg: Any) -> None:  # reason: ROS message is untyped
        """Write one decoded mask set into the store (on the spin thread).

        Masks arrive full-frame ``mono8``, area-ascending, with a parallel
        advisory-score array (display-only, never used to rank/filter — a
        59.8%-of-frame mask was measured at that model's top score of 0.977).

        A malformed mask (length != ``width * height``) fails the whole set
        rather than render a wrong shape, and is logged; nothing here may
        escape and kill the spin thread.
        """
        try:
            masks = [
                {
                    "png_b64": mono8_mask_to_png_b64(
                        bytes(mask.data), int(mask.width), int(mask.height)
                    ),
                    "width": int(mask.width),
                    "height": int(mask.height),
                    "score_advisory": (
                        float(msg.mask_scores_advisory[i])
                        if i < len(msg.mask_scores_advisory)
                        else None
                    ),
                }
                for i, mask in enumerate(msg.masks)
            ]
        except Exception as exc:  # reason: a malformed mask must not kill the spin thread
            _logger.warning(
                "dashboard.perception_mask_decode_failed",
                topic=PERCEPTION_MASKS_TOPIC,
                error=str(exc),
            )
            return
        stamp = msg.header.stamp
        self._store.set_perception_masks(
            camera=str(msg.camera),
            masks=masks,
            rskill_id=str(msg.rskill_id),
            stamp_unix=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            flip_180=dashboard_flip_180(),
        )

    def close(self) -> None:
        """Tear down the node + executor; shut rclpy down only if we started it."""
        import contextlib

        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown()
        if self._node is not None:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
        if self._owns_rclpy:
            with contextlib.suppress(Exception):
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
        self._node = None
        self._subscription = None
        self._masks_subscription = None
        self._executor = None
