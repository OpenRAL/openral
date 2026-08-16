"""Live ROS 2 subscriber feeding the dashboard's camera perception overlays.

The camera tiles already show what the robot sees (OTel `sensors.read_latest`
spans carry a JPEG thumbnail, re-served as MJPEG by `/api/camera/<src>/stream`).
What they could not show is what the robot *made of it*: a `kind: detector`
rSkill's boxes and a `kind: segmenter` rSkill's masks were only visible as text
in the event log, so an operator debugging a mis-grasp had to correlate a label
list against a picture by eye.

This module closes that gap for both halves. It follows
`safety_status_subscriber.py`'s shape exactly — one node created at launch, spun
on a daemon thread, inert-but-harmless when rclpy / the `openral_msgs` overlay
is unavailable, so a standalone dashboard with no ROS workspace sourced keeps
working and simply draws no overlays.

* **Detector boxes** — `openral_msgs/PromptStamped` on
  `/openral/perception/objects`, carrying an `openral_core.ObjectsMetadata` JSON
  document.
* **Segmenter masks** — `openral_msgs/SegmentMasks` on
  `/openral/perception/masks`. The segmenter's own contract is a *service*
  (`openral_msgs/srv/SegmentInView`, one shot per attach event), so this topic
  is not that contract: it is the segmenter node's **diagnostic** re-publication
  of its latest reply, off by default behind that node's `publish_debug_masks`
  parameter. Subscribing to it is therefore free when nobody enabled it, and a
  dashboard that never sees a message simply draws no masks. Decoded here with
  :func:`mono8_mask_to_png_b64`, written against the service's exact mono8
  convention, into `TelemetryStore.set_perception_masks`.

Read-only by construction: it subscribes and writes into the store. It holds no
publisher, no service client, and no authority over the robot. Overlays are
advisory *display* only — nothing here is a safety input, and nothing on the
robot reads back what this module renders.
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

    ``OPENRAL_DASHBOARD_FLIP_180`` is an existing repo-wide convention: the HAL
    publishes LIBERO/MuJoCo frames bottom-up, the camera *topic* stays raw, and
    only the dashboard's thumbnail is rotated so an operator sees an upright
    picture. Detectors consume the raw topic, so an overlay drawn over a flipped
    tile without the same correction lands point-mirrored — right-looking enough
    to be believed and wrong (CLAUDE.md §1.2). Read here and carried on the
    overlay so the renderer applies exactly the flip the image got.

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

    The `SegmentInView` service returns each candidate as a full-frame mono8
    image: 255 where the pixel belongs to the prompted object, 0 elsewhere. The
    producer writes 255, but any non-zero pixel counts as set — the same
    tolerance the HAL-side decoder documents, so a mask that survived a lossy
    hop still renders.

    Emitted as an **LA** PNG whose alpha channel *is* the mask, which is what
    makes the frontend's job one composite: draw the PNG, then flood the canvas
    with the instance colour under ``globalCompositeOperation = "source-in"``
    and the fill lands on exactly the set pixels. A binary alpha channel is also
    what PNG compresses best, so a 640x480 mask costs a few kB in the snapshot —
    the same order as the JPEG thumbnails already riding there.

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

        The wire carries the segmenter's own full-frame ``mono8`` candidates in
        **area-ascending** order with a parallel advisory-score array; both are
        preserved as-is. The scores are passed through for display and are never
        used to rank or filter — the service records a 59.8%-of-frame mask
        measured at that model's top score of 0.977.

        A malformed mask (a length that does not match its own ``width``
        times its ``height``) fails the whole set rather than rendering a plausible but
        wrong shape, and is logged. Like the detector leg, nothing here may
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
