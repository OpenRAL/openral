"""Generalised sensor → ROS 2 image publisher (PR2 of ADR-0019).

The existing :class:`openral_runner.backends.gstreamer.ros_tee.RosImagePublisher`
is GStreamer-coupled — it republishes frames pulled from a tee'd
``appsink``, which only works on the GStreamer backend. This module
generalises that pattern to **any** :class:`SensorReader`: a background
thread polls :meth:`SensorReader.read_latest` at a configurable rate and
republishes each frame as a ``sensor_msgs/Image`` on a configurable
topic.

Used by the new ``packages/openral_sensors_ros/`` lifecycle node so the
OpenCV / RealSense / mock readers — which today never reach a ROS
topic — feed the same dashboard, rosbag2 recorder (PR3), and converter
(PR4) as the GStreamer reader does via its zero-copy tee.

The publisher is a **parallel** consumer of the reader, not an
in-line interceptor. The inference hot path keeps calling
``reader.read_latest()`` directly on every tick; the publisher polls
from its own thread, so adding ROS publication does not slow the
runner. When the GStreamer backend is in use, that backend's tee
(``RosImagePublisher``) is the zero-copy path; instances of this class
are for the non-zero-copy fallback.

``rclpy`` is lazy-imported inside :meth:`SensorRosPublisher.start` so
this module is importable on hosts without a sourced ROS env. The
publisher just never starts there (and unit tests
``pytest.importorskip("rclpy")``).

Per CLAUDE.md §5.3 (QoS profiles by data class):

* image streams → ``BEST_EFFORT``, ``VOLATILE``, ``KEEP_LAST=5``
* ``CameraInfo`` → ``RELIABLE``, ``VOLATILE``, ``KEEP_LAST=1``
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Final, cast

import structlog
from openral_core import FrameEncoding
from openral_core.exceptions import ROSPerceptionStale

if TYPE_CHECKING:
    # Use a Protocol shim to avoid importing openral_runner at module load
    # time (sensors/runner depend on openral_core; we don't want a cycle).
    from openral_core import IntrinsicsPinhole
    from rclpy.node import Node
    from rclpy.publisher import Publisher

    from openral_sensors._reader_protocol import SensorReaderLike

__all__ = ["SensorRosPublisher"]

_log = structlog.get_logger(__name__)

# Default QoS depth for the image publisher. Matches gscam2's default
# (``sensor_data``-style: shallow, BEST_EFFORT-friendly).
_DEFAULT_QOS_DEPTH: Final[int] = 5

# Default join timeout when stopping the background thread. Long enough
# that a slow rclpy publish or a sleep can finish; short enough that a
# hung publisher doesn't block the runner indefinitely.
_THREAD_JOIN_TIMEOUT_S: Final[float] = 2.0

# Map :class:`FrameEncoding` to the ROS string encoding (``sensor_msgs/Image.encoding``).
# Only the CPU-side encodings make sense — the NVMM / DMA-BUF handles
# the GStreamer publisher owns never reach this fallback path.
_OPENRAL_TO_ROS_ENCODING: Final[dict[FrameEncoding, str]] = {
    FrameEncoding.BGR8: "bgr8",
    FrameEncoding.RGB8: "rgb8",
    FrameEncoding.MONO8: "mono8",
    FrameEncoding.DEPTH16: "16UC1",
}


class SensorRosPublisher:
    """Republish frames from a :class:`SensorReader` onto a ROS topic.

    Args:
        reader: Any backend that satisfies the :class:`SensorReader`
            structural protocol (OpenCV, RealSense, mock — anything
            with ``open / close / read_latest``). The publisher does
            **not** call ``reader.open`` / ``reader.close``; the
            caller (typically the ``openral_sensors_ros`` lifecycle
            node) owns the reader's lifecycle so the publisher can be
            attached to an already-open reader.
        topic: ROS topic to publish on (must be absolute, e.g.
            ``/cameras/wrist_rgb/image_raw``).
        rate_hz: Publish cadence in Hz. Frames are pulled from the
            reader at this rate; intermediate frames the reader
            buffers are dropped at the ``read_latest`` call (which is
            "latest only" by contract).
        node_name: Optional override for the ROS node name; defaults
            to ``openral_sensor_publisher_<sensor_id>``.
        frame_id: ``Image.header.frame_id`` to stamp on outgoing
            messages. Defaults to ``reader.sensor_id``.
        qos_depth: Depth of the image publisher's QoS history queue
            (CLAUDE.md §5.3 calls for ``KEEP_LAST=5–10`` for sensor
            streams; defaults to 5).
        camera_info: Optional :class:`IntrinsicsPinhole`. When
            provided, a companion ``CameraInfo`` topic is published at
            ``<topic>/camera_info`` with ``RELIABLE`` QoS — matches
            the ROS 2 ``camera_info_manager`` convention. The
            publisher rebuilds the message once and re-publishes at
            the same cadence as the image stream.

    Example:
        >>> # End-to-end exercised in tests/unit/test_sensor_ros_publisher.py
        >>> pass
    """

    def __init__(
        self,
        *,
        reader: SensorReaderLike,
        topic: str,
        rate_hz: float,
        node_name: str | None = None,
        frame_id: str | None = None,
        qos_depth: int = _DEFAULT_QOS_DEPTH,
        camera_info: IntrinsicsPinhole | None = None,
    ) -> None:
        """Stash configuration; no ROS I/O until :meth:`start`."""
        if not topic.startswith("/"):
            raise ValueError(
                f"SensorRosPublisher: topic must be absolute (start with '/'); got {topic!r}"
            )
        if rate_hz <= 0:
            raise ValueError(f"SensorRosPublisher: rate_hz must be > 0; got {rate_hz!r}")
        if qos_depth <= 0:
            raise ValueError(f"SensorRosPublisher: qos_depth must be > 0; got {qos_depth!r}")

        self._reader = reader
        self._topic = topic
        self._info_topic = f"{topic}/camera_info"
        self._rate_hz = float(rate_hz)
        self._node_name = node_name or f"openral_sensor_publisher_{reader.sensor_id}"
        self._frame_id = frame_id or reader.sensor_id
        self._qos_depth = qos_depth
        self._camera_info_spec = camera_info

        # Populated by start().
        self._node: Node | None = None
        self._image_publisher: Publisher | None = None
        self._info_publisher: Publisher | None = None
        self._we_initialised_rclpy = False
        self._is_started = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Stats — useful for the live test + the lifecycle node's
        # diagnostics topic.
        self._n_published = 0
        self._n_stale_skipped = 0

    @property
    def is_started(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._is_started

    @property
    def n_published(self) -> int:
        """Number of image messages successfully published since :meth:`start`."""
        return self._n_published

    @property
    def n_stale_skipped(self) -> int:
        """Number of ticks the reader had no fresh frame and we skipped publish."""
        return self._n_stale_skipped

    @property
    def topic(self) -> str:
        """The configured image topic (read-only)."""
        return self._topic

    @property
    def info_topic(self) -> str:
        """The configured ``CameraInfo`` companion topic (read-only)."""
        return self._info_topic

    def start(self) -> None:
        """Init rclpy (if needed), create publishers, start the pump thread.

        Raises:
            RuntimeError: When ``rclpy`` / ``sensor_msgs`` are not
                importable. Source a ROS 2 install before instantiating
                the lifecycle node that owns this publisher.
        """
        if self._is_started:
            return
        try:
            import rclpy
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "SensorRosPublisher.start() requires rclpy + sensor_msgs. "
                "Source a ROS 2 install (e.g. `source /opt/ros/jazzy/setup.bash`) "
                "before starting an openral_sensors_ros node."
            ) from exc

        if not rclpy.ok():
            rclpy.init()
            self._we_initialised_rclpy = True

        from rclpy.node import Node

        self._node = Node(self._node_name)
        # Image stream QoS — sensor_data-style per CLAUDE.md §5.3.
        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self._qos_depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._image_publisher = self._node.create_publisher(Image, self._topic, image_qos)
        # CameraInfo: RELIABLE, KEEP_LAST=1 (camera_info_manager convention).
        if self._camera_info_spec is not None:
            info_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._info_publisher = self._node.create_publisher(
                CameraInfo, self._info_topic, info_qos
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._pump_loop,
            name=f"sensor_ros_publisher:{self._reader.sensor_id}",
            daemon=True,
        )
        self._thread.start()
        self._is_started = True
        _log.debug(
            "sensor_ros_publisher.started",
            sensor_id=self._reader.sensor_id,
            topic=self._topic,
            rate_hz=self._rate_hz,
            has_camera_info=self._camera_info_spec is not None,
        )

    def stop(self) -> None:
        """Signal the pump thread to exit, then tear down ROS resources.

        Idempotent. The thread is given :data:`_THREAD_JOIN_TIMEOUT_S`
        to finish its current publish; a stuck thread is logged but
        not awaited indefinitely.
        """
        if not self._is_started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                _log.warning(
                    "sensor_ros_publisher.thread_join_timeout",
                    sensor_id=self._reader.sensor_id,
                    timeout_s=_THREAD_JOIN_TIMEOUT_S,
                )
        self._thread = None
        if self._image_publisher is not None and self._node is not None:
            self._node.destroy_publisher(self._image_publisher)
        self._image_publisher = None
        if self._info_publisher is not None and self._node is not None:
            self._node.destroy_publisher(self._info_publisher)
        self._info_publisher = None
        if self._node is not None:
            self._node.destroy_node()
        self._node = None
        if self._we_initialised_rclpy:
            import rclpy

            if rclpy.ok():
                with contextlib.suppress(Exception):  # defensive teardown
                    rclpy.shutdown()
            self._we_initialised_rclpy = False
        self._is_started = False
        _log.debug(
            "sensor_ros_publisher.stopped",
            sensor_id=self._reader.sensor_id,
            n_published=self._n_published,
            n_stale_skipped=self._n_stale_skipped,
        )

    # ── Pump thread ─────────────────────────────────────────────────────────

    def _pump_loop(self) -> None:
        """Poll ``reader.read_latest`` at ``rate_hz`` and publish each frame.

        Runs on the background thread until :attr:`_stop_event` fires.
        Stale frames raise :class:`ROSPerceptionStale` from the reader;
        we log + count + continue (the next-tick frame may be fresh).
        """
        period_s = 1.0 / self._rate_hz
        next_deadline = time.monotonic() + period_s
        while not self._stop_event.is_set():
            try:
                frame = self._reader.read_latest()
            except ROSPerceptionStale:
                self._n_stale_skipped += 1
                frame = None
            except RuntimeError as exc:
                # Reader was closed under us, or a real backend error.
                # Log and exit the loop — stop() will tidy up.
                _log.warning(
                    "sensor_ros_publisher.reader_error",
                    sensor_id=self._reader.sensor_id,
                    error=str(exc),
                )
                break
            if frame is not None:
                try:
                    self._publish_frame(frame)
                except Exception as exc:  # defensive: never kill the thread
                    _log.warning(
                        "sensor_ros_publisher.publish_failed",
                        sensor_id=self._reader.sensor_id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
            # Sleep until the next tick deadline. monotonic-based so
            # clock-skew doesn't drift the cadence.
            remaining = next_deadline - time.monotonic()
            # Use the stop_event.wait so stop() can wake us early.
            if remaining > 0 and self._stop_event.wait(timeout=remaining):
                return
            next_deadline += period_s
            # If we're way behind (the reader blocked for many periods),
            # snap to "now + one period" so we don't fire a burst of
            # publishes to catch up.
            if time.monotonic() > next_deadline + period_s:
                next_deadline = time.monotonic() + period_s

    def _publish_frame(self, frame: object) -> None:
        """Convert a :class:`SensorFrame` to ``sensor_msgs/Image`` + publish.

        Frames with no inline ``data`` (e.g. a topic-ref-only frame from
        a ROS subscriber backend) are skipped silently — there is nothing
        to republish; the existing ROS topic the frame points at is
        already on the bus.
        """
        from sensor_msgs.msg import Image

        # Avoid importing openral_core.SensorFrame at module load (cycle
        # risk through openral_sensors); duck-type via attribute access.
        data = getattr(frame, "data", None)
        if data is None:
            return  # topic-ref or handle-only frame; nothing to publish here
        encoding_enum = getattr(frame, "encoding", None)
        # Duck-typed frame: encoding may be any FrameEncoding member.
        # Cast for mypy; the .get() returns None for any non-key value.
        ros_encoding = _OPENRAL_TO_ROS_ENCODING.get(cast("FrameEncoding", encoding_enum))
        if ros_encoding is None:
            _log.warning(
                "sensor_ros_publisher.unsupported_encoding",
                sensor_id=self._reader.sensor_id,
                encoding=str(encoding_enum),
            )
            return

        assert self._image_publisher is not None  # type narrowing
        assert self._node is not None

        width = int(getattr(frame, "width", 0))
        height = int(getattr(frame, "height", 0))
        channels = 1 if ros_encoding in {"mono8", "16UC1"} else 3

        msg = Image()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = height
        msg.width = width
        msg.encoding = ros_encoding
        msg.is_bigendian = 0
        msg.step = width * channels * (2 if ros_encoding == "16UC1" else 1)
        msg.data = bytes(data)
        self._image_publisher.publish(msg)
        self._n_published += 1

        if self._info_publisher is not None and self._camera_info_spec is not None:
            self._publish_camera_info(width, height, msg.header.stamp)

    def _publish_camera_info(self, width: int, height: int, stamp: object) -> None:
        """Publish a companion ``CameraInfo`` at the same cadence as the image."""
        from sensor_msgs.msg import CameraInfo

        assert self._info_publisher is not None
        assert self._camera_info_spec is not None
        spec = self._camera_info_spec

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.width = width
        info.height = height
        info.distortion_model = spec.distortion_model
        # ROS expects flat lists; pad/truncate to spec sizes.
        info.d = list(spec.distortion_coeffs)
        info.k = [spec.fx, 0.0, spec.cx, 0.0, spec.fy, spec.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            spec.fx,
            0.0,
            spec.cx,
            0.0,
            0.0,
            spec.fy,
            spec.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        self._info_publisher.publish(info)
