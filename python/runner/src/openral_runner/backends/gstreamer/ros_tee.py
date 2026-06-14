"""ROS 2 image-publisher tee for :class:`GStreamerSensorReader`.

When a sensor's :class:`~openral_core.SensorReaderConfig` sets
``publish_to_ros = True``, the pipeline builder splits the GStreamer
pipeline at a ``tee`` element with two named appsink branches:

* ``bh_sink`` — the inference-path appsink the reader uses.
* ``ros_sink`` — the ROS-side appsink fed into this module's
  :class:`RosImagePublisher`, which republishes frames as
  ``sensor_msgs/Image`` on a configurable topic.

The ROS branch always lifts frames to system memory before this
appsink (see ``pipeline._build_ros_tee_branch``), so the publisher and
the inference path never share NVMM buffer ownership and the
publisher is identical on x86, Tegra, and Spark.

Rate-limiting is independent of the inference loop — the user picks a
``publish_rate_hz`` for rosbag2 / rqt that does not have to track the
30+ Hz inference cadence. Frames in between are simply dropped at the
publisher's gate (the appsink's ``max-buffers=1 drop=true`` keeps the
queue from growing).

``rclpy`` is lazy-imported inside :meth:`RosImagePublisher.start` so
this module can be imported on hosts without a sourced ROS env (the
reader's CPU path then carries on; the publisher just never starts).
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any, Final

import structlog
from openral_core import FrameEncoding

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.publisher import Publisher

__all__ = ["RosImagePublisher"]

log = structlog.get_logger(__name__)

# Default QoS depth for the image publisher. Matches gscam2's default
# (``sensor_data``-style: shallow, BEST_EFFORT-friendly).
_DEFAULT_QOS_DEPTH: Final[int] = 5

# Map our :class:`FrameEncoding` to the ROS string encoding (`sensor_msgs/Image.encoding`).
# Only the CPU-side encodings make sense here — the NVMM path never
# reaches the ROS publisher (ROS path is always lifted to system memory).
_OPENRAL_TO_ROS_ENCODING: Final[dict[FrameEncoding, str]] = {
    FrameEncoding.BGR8: "bgr8",
    FrameEncoding.RGB8: "rgb8",
    FrameEncoding.MONO8: "mono8",
}


class RosImagePublisher:
    """Republishes frames from a GStreamer ``ros_sink`` appsink onto a ROS topic.

    Args:
        sensor_id: Sensor name; embedded in the ROS node name to keep
            multi-camera processes from clashing.
        appsink: The ``ros_sink`` :class:`Gst.Element` (typed ``Any``
            here to avoid importing ``gi`` at module load).
        topic: ROS topic to publish on (e.g. ``/cameras/wrist_rgb/image_raw``).
        rate_hz: Maximum publish rate. Frames that arrive faster than
            this are dropped. ``None`` means "publish every frame".
        node_name: Optional override for the ROS node name; defaults to
            ``bh_ros_tee_<sensor_id>``.
        qos_depth: Depth of the publisher's QoS history queue.

    Raises:
        RuntimeError: When :meth:`start` is called but ``rclpy`` is
            not importable. Construction itself is safe without ROS.

    Example:
        >>> # End-to-end exercised in tests/unit/test_gstreamer_ros_tee.py
        >>> pass
    """

    def __init__(
        self,
        *,
        sensor_id: str,
        appsink: Any,  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        topic: str,
        rate_hz: float | None = None,
        node_name: str | None = None,
        qos_depth: int = _DEFAULT_QOS_DEPTH,
    ) -> None:
        """Stash configuration; no ROS I/O until :meth:`start`."""
        if not topic.startswith("/"):
            raise ValueError(
                f"RosImagePublisher: topic must be absolute (start with '/'); got {topic!r}"
            )
        if rate_hz is not None and rate_hz <= 0:
            raise ValueError(f"RosImagePublisher: rate_hz must be > 0 or None; got {rate_hz!r}")

        self.sensor_id = sensor_id
        self._appsink = appsink
        self._topic = topic
        self._rate_hz = rate_hz
        self._node_name = node_name or f"bh_ros_tee_{sensor_id}"
        self._qos_depth = qos_depth

        # Populated by start().
        self._node: Node | None = None
        self._publisher: Publisher | None = None
        self._signal_handler_id: int | None = None
        self._last_publish_monotonic_ns: int = 0
        self._publish_lock = threading.Lock()
        self._we_initialised_rclpy = False
        self._is_started = False

    @property
    def is_started(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._is_started

    def start(self) -> None:
        """Initialise rclpy (if needed), create the publisher, hook the appsink."""
        if self._is_started:
            return
        try:
            import rclpy  # noqa: PLC0415  # reason: optional ROS dep
            from rclpy.qos import (  # noqa: PLC0415
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import Image  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "RosImagePublisher.start() requires rclpy + sensor_msgs. "
                "Source a ROS 2 install (e.g. `source /opt/ros/jazzy/setup.bash`) "
                "before invoking the GStreamer reader with publish_to_ros=True."
            ) from exc

        # Initialise rclpy lazily and remember whether we did so we don't
        # tear down a context the user might own.
        if not rclpy.ok():
            rclpy.init()
            self._we_initialised_rclpy = True

        from rclpy.node import Node  # noqa: PLC0415

        self._node = Node(self._node_name)
        # Camera streams: BEST_EFFORT + VOLATILE + KEEP_LAST per CLAUDE.md §5.3.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self._qos_depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._publisher = self._node.create_publisher(Image, self._topic, qos)

        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._signal_handler_id = self._appsink.connect("new-sample", self._on_new_sample)
        self._is_started = True
        log.debug(
            "ros_tee.started",
            sensor_id=self.sensor_id,
            topic=self._topic,
            rate_hz=self._rate_hz,
        )

    def stop(self) -> None:
        """Disconnect the signal, destroy the publisher, shut down rclpy (if we own it)."""
        if not self._is_started:
            return
        if self._signal_handler_id is not None and self._appsink is not None:
            with contextlib.suppress(Exception):  # reason: defensive cleanup
                self._appsink.disconnect(self._signal_handler_id)
        self._signal_handler_id = None
        if self._publisher is not None and self._node is not None:
            self._node.destroy_publisher(self._publisher)
        self._publisher = None
        if self._node is not None:
            self._node.destroy_node()
        self._node = None
        if self._we_initialised_rclpy:
            import rclpy  # noqa: PLC0415

            if rclpy.ok():
                rclpy.shutdown()
            self._we_initialised_rclpy = False
        self._is_started = False
        log.debug("ros_tee.stopped", sensor_id=self.sensor_id)

    # ── GStreamer callback ──────────────────────────────────────────────────

    def _on_new_sample(self, appsink: Any) -> int:  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        """Pull a sample and publish it as a sensor_msgs/Image.

        Runs on a GStreamer streaming thread; the work is bounded:
        rate-gate → map → numpy view → ``sensor_msgs.msg.Image`` →
        ``publisher.publish``. ``rclpy`` publishers are thread-safe.
        """
        from gi.repository import Gst  # noqa: PLC0415
        from sensor_msgs.msg import Image  # noqa: PLC0415

        ok_flow = int(Gst.FlowReturn.OK)
        if not self._is_started or self._publisher is None or self._node is None:
            # Defensive: stop() may have been called between this callback
            # being scheduled and run.
            return ok_flow
        if self._rate_hz is not None and not self._claim_rate_slot():
            return ok_flow

        extracted = self._extract_image_payload(appsink, Gst)
        if extracted is None:
            return ok_flow
        payload, width, height, encoding = extracted

        msg = Image()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.sensor_id
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = encoding
        msg.is_bigendian = 0
        channels = 1 if encoding == "mono8" else 3
        msg.step = int(width) * channels
        msg.data = payload
        self._publisher.publish(msg)
        return ok_flow

    def _claim_rate_slot(self) -> bool:
        """Return ``True`` and update the monotonic gate when a publish slot is due.

        When :attr:`_rate_hz` is set, throttles publishes to at most
        ``rate_hz`` per second; otherwise the caller should not call this.
        """
        assert self._rate_hz is not None  # caller guards
        min_interval_ns = int(1e9 / self._rate_hz)
        now_ns = time.monotonic_ns()
        with self._publish_lock:
            if now_ns - self._last_publish_monotonic_ns < min_interval_ns:
                return False
            self._last_publish_monotonic_ns = now_ns
            return True

    def _extract_image_payload(
        self,
        appsink: Any,  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed
        gst: Any,  # noqa: ANN401  # reason: gi.repository.Gst — duck-typed
    ) -> tuple[bytes, int, int, str] | None:
        """Pull the latest sample and return (data, width, height, ros_encoding).

        Returns ``None`` when the sample is malformed, the format is
        unsupported, or the buffer cannot be mapped — the caller treats
        this as "no message this tick" and returns ``FlowReturn.OK``.
        """
        sample = appsink.emit("pull-sample")
        if sample is None:  # pragma: no cover — EOS edge
            return None
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None:  # pragma: no cover — samples always carry caps
            return None
        gst_format = structure.get_string("format") or ""
        encoding = _gst_format_to_ros_encoding(gst_format)
        if encoding is None:
            log.warning(
                "ros_tee.unsupported_format",
                sensor_id=self.sensor_id,
                format=gst_format,
            )
            return None
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h):  # pragma: no cover — caps always carry w/h
            return None
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(gst.MapFlags.READ)
        if not ok:  # pragma: no cover — mapping failure is rare
            return None
        try:
            payload = bytes(map_info.data)
        finally:
            buffer.unmap(map_info)
        return payload, int(width), int(height), encoding


def _gst_format_to_ros_encoding(gst_format: str) -> str | None:
    """Map a GStreamer caps ``format`` string to a ROS Image encoding."""
    return {
        "BGR": "bgr8",
        "RGB": "rgb8",
        "GRAY8": "mono8",
    }.get(gst_format)
