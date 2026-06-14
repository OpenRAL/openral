"""Perception event tee for :class:`GStreamerSensorReader` (ADR-0018 F6).

When a sensor's :class:`~openral_core.SensorReaderConfig` enables the
event leg via :attr:`PipelineSpec.enable_event_tee`, the pipeline builder
adds a third ``tee`` branch terminating in
``appsink name=event_sink`` (default). Frames pulled from that appsink
are fed to a list of :class:`EventDetector` instances; whenever a
detector emits a :data:`~openral_core.PerceptionEventMetadata`, the
publisher fans it out as a ``openral_msgs/PromptStamped`` on the
per-kind topic ``/openral/perception/<kind>``.

The contract is intentionally narrow:

* Three legs (policy / observability / event) share the same upstream
  GStreamer pipeline and the same
  :func:`openral_runner.backends.gstreamer.cuda_context.get_shared_cuda_context`
  singleton (ADR-0011 §"Shared CUDA context"). The event leg lifts
  frames to system memory before the appsink — Python detectors consume
  numpy arrays, never NVMM handles.
* Per-kind topics, per ADR-0018 §3 / capability review §3 (F6). The
  topology is symmetric with :mod:`openral_observability.failure_bus`'s
  ``/openral/failure/<source>`` layout.
* Token-bucket rate-limit at each detector (default 5 Hz), so a noisy
  motion source can't storm the reasoner. Dropped events are counted
  but not summarised — the reasoner does not steer on perception
  events the way it steers on failures (FailureBus owns the
  ``KIND_SUPPRESSED_SUMMARY`` roll-up).
* :mod:`rclpy` is lazy-imported inside :meth:`PerceptionEventPublisher.start`
  so this module is import-safe on hosts without a sourced ROS env.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any, Final, Protocol

import structlog
from openral_core import (
    MotionMetadata,
    PerceptionEventMetadata,
    SceneChangeMetadata,
)

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.publisher import Publisher

__all__ = [
    "TOPIC_PREFIX",
    "EventDetector",
    "MotionDetector",
    "PerceptionEventPublisher",
    "SceneChangeDetector",
]

log = structlog.get_logger(__name__)

# Topic prefix locked by ADR-0018 §1 / capability review §3 (F6).
# New kinds = new topics under the same prefix; no IDL bump.
TOPIC_PREFIX: Final[str] = "/openral/perception"

# Default QoS depth for the per-kind PromptStamped publisher. Matches the
# /openral/perception/* QoS specified in ADR-0018 §1
# (BEST_EFFORT + VOLATILE + KEEP_LAST = 10).
_DEFAULT_QOS_DEPTH: Final[int] = 10

# Default token-bucket rate-limit per detector, in Hz. Matches the
# reasoner tick cap so a single detector can fully load the reasoner
# but not the broker.
_DEFAULT_RATE_HZ: Final[float] = 5.0


class EventDetector(Protocol):
    """A pluggable per-frame detector that maps a BGR frame to an event.

    Implementations live in this module (:class:`MotionDetector`,
    :class:`SceneChangeDetector`) or in a downstream package (e.g. an
    ``ObjectsDetector`` wrapping ``yolov8n``).

    Args:
        sensor_id: Sensor name forwarded to the emitted metadata.

    Returns:
        ``None`` when the frame does not trip the detector's threshold.
        Otherwise a :data:`~openral_core.PerceptionEventMetadata`
        variant whose ``kind`` matches :attr:`EventDetector.kind`.
    """

    kind: str
    """One of ``motion`` / ``objects`` / ``ocr`` / ``scene_change``."""

    def detect(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
    ) -> PerceptionEventMetadata | None:
        """Run one detection pass. ``None`` means "no event this frame"."""

    def summarise(self, metadata: PerceptionEventMetadata) -> str:
        """Return the human-readable ``PromptStamped.text`` for ``metadata``."""


class MotionDetector:
    """Frame-difference motion detector — pure-Python over a BGR appsink.

    Computes the mean absolute per-pixel difference between consecutive
    frames in the luma channel (BT.601). Emits :class:`MotionMetadata`
    when the magnitude crosses :attr:`threshold`.

    Numpy is lazy-imported inside :meth:`detect` so the module stays
    import-safe on hosts that haven't pulled the openral-rskill ML
    stack — the detector itself requires numpy at runtime.

    Args:
        threshold: Magnitude threshold in ``[0, 1]``. The default of
            ``0.02`` corresponds to ~5/255 mean abs delta — sensitive
            enough to fire on hand motion, quiet on lighting drift.
        downsample: Per-axis decimation factor applied before the diff.
            ``1`` (no decimation) at 320x240 BGR is ~5 ms per frame on
            a modern CPU. Increase for higher resolutions; the detector
            is not pixel-accurate so a 2-4x decimation is usually fine.

    Example:
        >>> det = MotionDetector(threshold=0.02)
        >>> det.kind
        'motion'
    """

    kind: str = "motion"

    def __init__(self, *, threshold: float = 0.02, downsample: int = 1) -> None:
        """Validate thresholds and reset the internal previous-frame slot."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"MotionDetector.threshold must be in [0, 1]; got {threshold!r}")
        if downsample < 1:
            raise ValueError(f"MotionDetector.downsample must be >= 1; got {downsample!r}")
        self.threshold = threshold
        self.downsample = downsample
        self._prev_luma: Any | None = None

    def detect(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
    ) -> PerceptionEventMetadata | None:
        """Compute mean abs delta vs the previous frame; emit on threshold cross."""
        import numpy as np  # noqa: PLC0415  # reason: lazy — see module docstring

        try:
            arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        except ValueError:
            # frame size != width*height*3 → caps mismatch upstream; drop sample.
            return None
        step = self.downsample
        bgr = arr[::step, ::step, :]
        # BT.601 luma; integer math keeps the per-frame cost predictable.
        luma = (
            bgr[..., 0].astype(np.int32) * 29
            + bgr[..., 1].astype(np.int32) * 150
            + bgr[..., 2].astype(np.int32) * 77
        ) >> 8

        prev = self._prev_luma
        self._prev_luma = luma
        if prev is None or prev.shape != luma.shape:
            return None

        delta = np.abs(luma - prev)
        magnitude = float(delta.mean()) / 255.0
        if magnitude < self.threshold:
            return None

        # Localise: tight axis-aligned bbox around moving pixels. Using
        # ``> 0`` rather than ``> mean`` keeps the bbox sensible when the
        # delta is uniform (whole-frame motion) — the mean-mask would be
        # all-False for a constant delta and erase the localisation.
        mask = delta > 0
        if mask.any():
            ys, xs = mask.nonzero()
            x_min = int(xs.min()) * step
            x_max = int(xs.max() + 1) * step
            y_min = int(ys.min()) * step
            y_max = int(ys.max() + 1) * step
            region: tuple[int, int, int, int] | None = (x_min, y_min, x_max, y_max)
        else:
            region = None

        return MotionMetadata(
            sensor_id=sensor_id,
            magnitude=min(magnitude, 1.0),
            threshold=self.threshold,
            region_bbox=region,
        )

    def summarise(self, metadata: PerceptionEventMetadata) -> str:
        """Render a one-line ``PromptStamped.text`` for a motion event."""
        if not isinstance(metadata, MotionMetadata):  # pragma: no cover — typed call sites
            raise TypeError(f"MotionDetector.summarise: wrong kind {metadata.kind!r}")
        region = metadata.region_bbox
        if region is None:
            return f"motion magnitude={metadata.magnitude:.3f} on {metadata.sensor_id}"
        x_min, y_min, x_max, y_max = region
        return (
            f"motion magnitude={metadata.magnitude:.3f} "
            f"bbox=({x_min},{y_min},{x_max},{y_max}) on {metadata.sensor_id}"
        )


class SceneChangeDetector:
    """Grayscale-histogram scene-change detector — pure numpy, no cv2.

    Builds a 32-bin grayscale histogram per frame and emits
    :class:`SceneChangeMetadata` when the chi-square distance to the
    previous frame's histogram exceeds :attr:`threshold`. Cheap,
    illumination-tolerant, and good enough to wake a reasoner on a
    new scene without firing on per-pixel jitter the way
    :class:`MotionDetector` does.

    Args:
        threshold: Distance threshold in the detector's native units
            (``chisqr_alt`` over a normalised histogram → values are
            typically ``0`` for "identical frames", ``> 0.5`` for
            "different scene"). Default ``0.5``.

    Example:
        >>> det = SceneChangeDetector(threshold=0.5)
        >>> det.kind
        'scene_change'
    """

    kind: str = "scene_change"
    metric: str = "chisqr_alt"

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Validate the threshold and reset the previous-histogram slot."""
        if threshold < 0.0:
            raise ValueError(
                f"SceneChangeDetector.threshold must be >= 0; got {threshold!r}",
            )
        self.threshold = threshold
        self._prev_hist: Any | None = None

    def detect(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
    ) -> PerceptionEventMetadata | None:
        """Compare grayscale histograms across consecutive frames."""
        import numpy as np  # noqa: PLC0415  # reason: lazy — see module docstring

        try:
            arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        except ValueError:
            return None
        luma = (
            arr[..., 0].astype(np.int16) * 29
            + arr[..., 1].astype(np.int16) * 150
            + arr[..., 2].astype(np.int16) * 77
        ) >> 8
        hist, _ = np.histogram(luma, bins=32, range=(0, 256))
        hist_norm = hist.astype(np.float64) / max(1, hist.sum())

        prev = self._prev_hist
        self._prev_hist = hist_norm
        if prev is None:
            return None

        # chi-square alternative (OpenCV's HISTCMP_CHISQR_ALT):
        # 2 * sum( (h1 - h2)^2 / (h1 + h2) ), masked over non-zero bins.
        denom = hist_norm + prev
        mask = denom > 0
        diff_sq = (hist_norm - prev) ** 2
        distance = float((2 * diff_sq[mask] / denom[mask]).sum())
        if distance < self.threshold:
            return None

        return SceneChangeMetadata(
            sensor_id=sensor_id,
            distance=distance,
            threshold=self.threshold,
            metric=self.metric,
        )

    def summarise(self, metadata: PerceptionEventMetadata) -> str:
        """Render a one-line ``PromptStamped.text`` for a scene-change event."""
        if not isinstance(metadata, SceneChangeMetadata):  # pragma: no cover
            raise TypeError(f"SceneChangeDetector.summarise: wrong kind {metadata.kind!r}")
        return (
            f"scene_change distance={metadata.distance:.3f} "
            f"({metadata.metric}) on {metadata.sensor_id}"
        )


class _TokenBucket:
    """Lock-free-on-the-fast-path token bucket; one per (sensor, kind) pair.

    Mirrors :class:`openral_observability.failure_bus._TokenBucket` in
    spirit — independent implementation to keep ``perception_tee`` free
    of an observability-package dependency.
    """

    def __init__(self, *, rate_hz: float) -> None:
        """Stash the per-detector rate cap; arm the bucket as empty."""
        if rate_hz <= 0:
            raise ValueError(f"_TokenBucket.rate_hz must be > 0; got {rate_hz!r}")
        self._min_interval_ns = int(1e9 / rate_hz)
        self._last_emit_ns = 0
        self._dropped = 0
        self._lock = threading.Lock()

    def try_consume(self, now_ns: int) -> bool:
        """Return ``True`` and arm the next slot when one is due; else drop."""
        with self._lock:
            if now_ns - self._last_emit_ns < self._min_interval_ns:
                self._dropped += 1
                return False
            self._last_emit_ns = now_ns
            return True

    @property
    def dropped(self) -> int:
        """Number of events dropped by this bucket since construction."""
        return self._dropped


class PerceptionEventPublisher:
    """Publishes detector outputs as ``PromptStamped`` on ``/openral/perception/<kind>``.

    One publisher owns the ``event_sink`` appsink for a single sensor;
    it fans out to one :class:`rclpy.publisher.Publisher` per registered
    detector ``kind``. Per-detector token buckets cap the per-kind topic
    rate at :attr:`rate_hz` (default 5 Hz).

    Args:
        sensor_id: Sensor name; embedded in the ROS node name and the
            emitted metadata's ``sensor_id`` field.
        appsink: The ``event_sink`` :class:`Gst.Element` (typed ``Any``
            here to avoid importing ``gi`` at module load).
        detectors: Ordered list of detectors to run on every frame.
        rate_hz: Per-detector topic rate cap in Hz. Defaults to
            ``_DEFAULT_RATE_HZ`` (5 Hz).
        node_name: Optional override for the ROS node name; defaults to
            ``openral_perception_tee_<sensor_id>``.
        qos_depth: KEEP_LAST depth on each publisher.
        topic_prefix: ROS topic prefix. Defaults to :data:`TOPIC_PREFIX`
            (``/openral/perception``). The full topic is
            ``f"{topic_prefix}/{detector.kind}"``.

    Raises:
        ValueError: When two detectors declare the same ``kind`` (the
            per-kind topic would be ambiguous), or when ``rate_hz``
            is not positive.
        RuntimeError: When :meth:`start` is called but ``rclpy`` is
            not importable.

    Example:
        >>> # End-to-end exercise lives in tests/unit/test_gstreamer_perception_tee.py
        >>> import contextlib
        >>> with contextlib.suppress(ImportError):
        ...     from openral_runner.backends.gstreamer.perception_tee import (
        ...         PerceptionEventPublisher,
        ...     )
    """

    def __init__(
        self,
        *,
        sensor_id: str,
        appsink: Any,  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        detectors: list[EventDetector],
        rate_hz: float = _DEFAULT_RATE_HZ,
        node_name: str | None = None,
        qos_depth: int = _DEFAULT_QOS_DEPTH,
        topic_prefix: str = TOPIC_PREFIX,
    ) -> None:
        """Validate detectors and stash configuration; no ROS I/O until :meth:`start`."""
        if rate_hz <= 0:
            raise ValueError(
                f"PerceptionEventPublisher.rate_hz must be > 0; got {rate_hz!r}",
            )
        if not detectors:
            raise ValueError(
                "PerceptionEventPublisher: at least one detector is required",
            )
        kinds = [d.kind for d in detectors]
        if len(set(kinds)) != len(kinds):
            raise ValueError(
                f"PerceptionEventPublisher: duplicate detector kinds {kinds!r} — "
                f"each kind owns one /openral/perception/<kind> topic",
            )
        if not topic_prefix.startswith("/"):
            raise ValueError(
                f"PerceptionEventPublisher.topic_prefix must be absolute; got {topic_prefix!r}",
            )

        self.sensor_id = sensor_id
        self._appsink = appsink
        self._detectors = list(detectors)
        self._rate_hz = rate_hz
        self._node_name = node_name or f"openral_perception_tee_{sensor_id}"
        self._qos_depth = qos_depth
        self._topic_prefix = topic_prefix

        self._buckets: dict[str, _TokenBucket] = {
            d.kind: _TokenBucket(rate_hz=rate_hz) for d in detectors
        }

        # Populated by start().
        self._node: Node | None = None
        self._publishers: dict[str, Publisher] = {}
        self._signal_handler_id: int | None = None
        self._we_initialised_rclpy = False
        self._is_started = False

    @property
    def is_started(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._is_started

    @property
    def dropped_counts(self) -> dict[str, int]:
        """Per-kind count of detections suppressed by the rate-limit gate."""
        return {kind: bucket.dropped for kind, bucket in self._buckets.items()}

    def start(self) -> None:
        """Initialise rclpy (if needed), create publishers, hook the appsink."""
        if self._is_started:
            return
        try:
            import rclpy  # noqa: PLC0415  # reason: optional ROS dep
            from openral_msgs.msg import (  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415  # reason: rclpy-generated
                PromptStamped,
            )
            from rclpy.qos import (  # noqa: PLC0415
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PerceptionEventPublisher.start() requires rclpy + openral_msgs. "
                "Source a ROS 2 install (e.g. `source /opt/ros/jazzy/setup.bash`) "
                "before invoking the GStreamer reader with enable_event_tee=True.",
            ) from exc

        if not rclpy.ok():
            rclpy.init()
            self._we_initialised_rclpy = True

        from rclpy.node import Node  # noqa: PLC0415

        self._node = Node(self._node_name)
        # /openral/perception/* uses BEST_EFFORT + VOLATILE + KEEP_LAST per ADR-0018 §1.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self._qos_depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        for detector in self._detectors:
            topic = f"{self._topic_prefix}/{detector.kind}"
            self._publishers[detector.kind] = self._node.create_publisher(
                PromptStamped,
                topic,
                qos,
            )

        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._signal_handler_id = self._appsink.connect("new-sample", self._on_new_sample)
        self._is_started = True
        log.debug(
            "perception_tee.started",
            sensor_id=self.sensor_id,
            kinds=[d.kind for d in self._detectors],
            rate_hz=self._rate_hz,
        )

    def stop(self) -> None:
        """Disconnect the signal, destroy publishers, shut down rclpy (if we own it)."""
        if not self._is_started:
            return
        if self._signal_handler_id is not None and self._appsink is not None:
            with contextlib.suppress(Exception):  # reason: defensive cleanup
                self._appsink.disconnect(self._signal_handler_id)
        self._signal_handler_id = None
        if self._node is not None:
            for publisher in self._publishers.values():
                self._node.destroy_publisher(publisher)
            self._node.destroy_node()
        self._publishers = {}
        self._node = None
        if self._we_initialised_rclpy:
            import rclpy  # noqa: PLC0415

            if rclpy.ok():
                rclpy.shutdown()
            self._we_initialised_rclpy = False
        self._is_started = False
        log.debug("perception_tee.stopped", sensor_id=self.sensor_id)

    # ── GStreamer callback ──────────────────────────────────────────────────

    def _on_new_sample(self, appsink: Any) -> int:  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        """Pull a sample, run every detector, publish on the matching topic."""
        from gi.repository import Gst  # noqa: PLC0415

        ok_flow = int(Gst.FlowReturn.OK)
        if not self._is_started or self._node is None:
            return ok_flow
        extracted = self._pull_bgr_sample(appsink, Gst)
        if extracted is None:
            return ok_flow
        self._dispatch_detectors(*extracted)
        return ok_flow

    def _pull_bgr_sample(
        self,
        appsink: Any,  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        gst: Any,  # noqa: ANN401  # reason: gi.repository.Gst — duck-typed
    ) -> tuple[bytes, int, int] | None:
        """Pull one ``BGR`` sample; return ``None`` on caps mismatch / EOS / map failure."""
        sample = appsink.emit("pull-sample")
        if sample is None:  # pragma: no cover — EOS edge
            return None
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None:  # pragma: no cover — caps always present
            return None
        gst_format = structure.get_string("format") or ""
        if gst_format != "BGR":
            # The event leg's capsfilter pins format=BGR; an unexpected
            # caps mismatch means upstream changed mid-stream — log and drop.
            log.warning(
                "perception_tee.unsupported_format",
                sensor_id=self.sensor_id,
                format=gst_format,
            )
            return None
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h):  # pragma: no cover
            return None
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(gst.MapFlags.READ)
        if not ok:  # pragma: no cover
            return None
        try:
            payload = bytes(map_info.data)
        finally:
            buffer.unmap(map_info)
        return payload, int(width), int(height)

    def _dispatch_detectors(self, payload: bytes, width: int, height: int) -> None:
        """Run every detector on ``payload`` and publish any emitted events."""
        from openral_msgs.msg import PromptStamped  # noqa: PLC0415

        assert self._node is not None  # caller guards
        now_ns = time.monotonic_ns()
        stamp = self._node.get_clock().now().to_msg()
        for detector in self._detectors:
            metadata = detector.detect(payload, width, height, self.sensor_id)
            if metadata is None:
                continue
            bucket = self._buckets[detector.kind]
            if not bucket.try_consume(now_ns):
                continue
            publisher = self._publishers.get(detector.kind)
            if publisher is None:  # pragma: no cover — start() builds the dict
                continue
            msg = PromptStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.sensor_id
            msg.text = detector.summarise(metadata)
            msg.metadata_json = metadata.model_dump_json()
            publisher.publish(msg)
