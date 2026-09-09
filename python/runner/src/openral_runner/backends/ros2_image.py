"""``ros2_image`` sensor reader — bridge a vendor driver's ROS 2 image topic.

The other real-hardware backends open a device themselves: ``opencv_thread``
takes a ``/dev/video*`` node, ``gstreamer`` runs a pipeline. Some cameras
cannot be opened that way at all, because the useful stream is *computed* by a
vendor SDK and only ever published as a ROS topic:

* **StereoLabs ZED** — a passive stereo pair. Over USB it presents ONE UVC node
  carrying both eyes side by side; depth does not exist on the device. The ZED
  SDK rectifies and stereo-matches on the host GPU and ``zed_wrapper``
  publishes the result on ``/<name>/depth/depth_registered``. Without this
  backend the catalog's ``stereolabs/zed_mini`` depth stream
  (``openral_sensors.stereolabs.zed_mini_bundle``) is undeliverable — the
  bundle declares it, and nothing could subscribe.
* Any other driver-owned stream: RealSense's aligned depth, Orbbec, a
  GMSL/Isaac driver, a rectified rig from a calibration node.

So this reader owns no device. It subscribes, keeps the newest message in a
one-slot buffer, and serves it through the same non-blocking
``read_latest`` staleness contract every other backend honours — a frame
older than the budget raises ``ROSPerceptionStale`` rather than being
returned quietly.

Node ownership mirrors ``openral_sensors.ros_publisher.SensorRosPublisher``:
a composed runtime injects its existing node, and a standalone caller gets a
private node plus a background executor thread. ``rclpy`` is lazy-imported
inside ``open`` so the runner stays importable on hosts without ROS.

QoS defaults to the **sensor data** class of CLAUDE.md §2 (``BEST_EFFORT``,
``VOLATILE``, ``KEEP_LAST=5``). That is also the compatible-in-both-directions
choice: a ``BEST_EFFORT`` subscriber matches a ``RELIABLE`` publisher, while a
``RELIABLE`` subscriber would silently receive **nothing** from a
``BEST_EFFORT`` one — a QoS mismatch reads exactly like a dead camera.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog
from openral_core import FrameEncoding, SensorFrame
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Ros2ImageSensorReader"]

log = structlog.get_logger(__name__)

# ROS `sensor_msgs/Image.encoding` → (FrameEncoding, numpy dtype, channels).
# Only the layouts a camera driver actually publishes; anything else is
# refused by name at open() rather than misread as pixels.
_DIRECT_ENCODINGS: Final[dict[str, tuple[FrameEncoding, str, int]]] = {
    "rgb8": (FrameEncoding.RGB8, "uint8", 3),
    "bgr8": (FrameEncoding.BGR8, "uint8", 3),
    "mono8": (FrameEncoding.MONO8, "uint8", 1),
    "8UC1": (FrameEncoding.MONO8, "uint8", 1),
    "8UC3": (FrameEncoding.BGR8, "uint8", 3),
    # Depth already in the DEPTH16 layout: uint16 millimetres, 0 = no reading.
    "mono16": (FrameEncoding.DEPTH16, "uint16", 1),
    "16UC1": (FrameEncoding.DEPTH16, "uint16", 1),
}

# Four-channel colour, mapped onto its three-channel sibling by dropping the
# alpha byte. The ZED wrapper publishes `bgra8` on
# `rgb/color/rect/image` — its default colour layout — and the alpha plane is
# a constant 255 that carries nothing. Refusing it left the context camera's
# `/openral/cameras/context/image` permanently empty on the real OpenArm cell
# while the driver was healthy and streaming: every frame raised, `_on_image`
# downgraded it to a WARN, and the panel read "no data" for a reason visible
# only in the log.
_ALPHA_ENCODINGS: Final[dict[str, str]] = {"bgra8": "bgr8", "rgba8": "rgb8"}

# Float depth in METRES, which is what the ZED SDK and several other drivers
# publish. Converted to DEPTH16 millimetres on the way in — see
# `_depth32f_to_depth16`.
_FLOAT_DEPTH_ENCODINGS: Final[frozenset[str]] = frozenset({"32FC1"})

# DEPTH16 is uint16 millimetres, so it tops out here. 0 is the ROS convention
# for "no reading", which is also where non-finite and out-of-range samples go.
_DEPTH16_MAX_MM: Final[int] = 65535


def _depth32f_to_depth16(metres: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Convert float32 metre depth to the uint16-millimetre DEPTH16 layout.

    ``FrameEncoding`` has no float-depth member, and the
    consumers downstream (nvblox, octomap, the world-cloud bridge) all speak
    the uint16-millimetre convention, so the conversion happens once here
    rather than at every reader.

    Non-finite samples (``NaN`` marks "no match" in a stereo matcher, ``inf``
    marks "beyond range") and anything outside ``[0, 65.535] m`` become **0**,
    the ROS "no reading" value. That is the fail-safe direction: a wrapped
    ``uint16`` would report a confident, wrong, *near* distance for something
    far away.

    Args:
        metres: Depth image in metres.

    Returns:
        The same image as uint16 millimetres.

    Example:
        >>> import numpy as np
        >>> _depth32f_to_depth16(np.array([[1.5, np.nan, -1.0, 1e6]], np.float32)).tolist()
        [[1500, 0, 0, 0]]
    """
    mm = np.asarray(metres, dtype=np.float64) * 1000.0
    valid = np.isfinite(mm) & (mm >= 0.0) & (mm <= _DEPTH16_MAX_MM)
    return np.where(valid, mm, 0.0).astype(np.uint16)


class Ros2ImageSensorReader:
    """Reader that serves frames from a ROS 2 ``sensor_msgs/Image`` topic.

    The constructor only records configuration; ``open`` creates the
    subscription (and, when no node was injected, the node and its executor
    thread).

    Args:
        sensor_id: Sensor name, matching
            ``SensorReaderConfig.sensor_id``.
        topic: Topic the vendor driver publishes on, e.g.
            ``/zed/depth/depth_registered``.
        default_max_age_ms: Staleness budget applied when ``read_latest``
            is called with ``max_age_ms=None``.
        reliability: ``"best_effort"`` (default) or ``"reliable"``. Leave it at
            the default unless a driver publishes RELIABLE *and* you need
            delivery guarantees — see the module docstring on why the default
            is the compatible-in-both-directions choice.
        qos_depth: Subscription queue depth (sensor class: 5-10).
        node: An existing ``rclpy`` node to attach the subscription to. When
            ``None`` the reader creates and spins its own.

    Example:
        >>> reader = Ros2ImageSensorReader(sensor_id="head_depth", topic="/zed/depth")
        >>> reader.sensor_id, reader.is_open
        ('head_depth', False)
    """

    def __init__(
        self,
        *,
        sensor_id: str,
        topic: str,
        default_max_age_ms: int = 100,
        reliability: str = "best_effort",
        qos_depth: int = 5,
        node: Any = None,
    ) -> None:
        """Record configuration; ``open`` does the work."""
        if not topic:
            raise ROSConfigError(
                f"Ros2ImageSensorReader({sensor_id!r}) requires a non-empty "
                "`topic` backend param — the driver topic to subscribe to."
            )
        if reliability not in ("best_effort", "reliable"):
            raise ROSConfigError(
                f"Ros2ImageSensorReader({sensor_id!r}).reliability must be "
                f"'best_effort' or 'reliable', got {reliability!r}."
            )
        self.sensor_id = sensor_id
        self.is_open = False
        self._topic = topic
        self._default_max_age_ms = default_max_age_ms
        self._reliability = reliability
        self._qos_depth = qos_depth

        self._node: Any = node
        self._owns_node = node is None
        self._we_initialised_rclpy = False
        self._subscription: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None

        self._frame_lock = threading.Lock()
        self._latest: SensorFrame | None = None
        self._latest_stamp_monotonic_ns: int | None = None
        self._frames_received = 0
        self._frames_dropped = 0

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def open(self) -> None:
        """Subscribe to the topic, spinning a private node when needed.

        Idempotent. Returns as soon as the subscription exists — no frame is
        buffered yet, so ``read_latest`` raises until the driver publishes
        one.

        Raises:
            ROSConfigError: ``rclpy`` / ``sensor_msgs`` are unavailable.
        """
        if self.is_open:
            return
        try:
            import rclpy
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise ROSConfigError(
                f"Ros2ImageSensorReader({self.sensor_id!r}) requires rclpy + "
                "sensor_msgs. Source a ROS 2 install (and the workspace overlay) "
                "before opening a `ros2_image` sensor."
            ) from exc

        if self._owns_node:
            if not rclpy.ok():
                rclpy.init()
                self._we_initialised_rclpy = True
            from rclpy.node import Node

            self._node = Node(f"openral_ros2_image_{_node_safe(self.sensor_id)}")

        qos = QoSProfile(
            depth=self._qos_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self._reliability == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
        )
        self._subscription = self._node.create_subscription(Image, self._topic, self._on_image, qos)

        if self._owns_node:
            from rclpy.executors import SingleThreadedExecutor

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name=f"ros2-image-{self.sensor_id}",
                daemon=True,
            )
            self._spin_thread.start()

        self.is_open = True
        log.info(
            "sensor.ros2_image.open",
            sensor_id=self.sensor_id,
            topic=self._topic,
            reliability=self._reliability,
            owns_node=self._owns_node,
        )

    def close(self) -> None:
        """Tear down the subscription (and the private node, if we made one).

        Idempotent, and safe to call after a partially-failed ``open``.

        No ``is_open`` early return: ``open`` sets that flag *last*, so an
        ``open`` that raised after creating the node (or after calling
        ``rclpy.init``) leaves both alive with the flag still false. Every step
        below is guarded on the resource it releases, which is what makes the
        repeat call a no-op — the flag never was.
        """
        self.is_open = False
        if self._subscription is not None and self._node is not None:
            self._node.destroy_subscription(self._subscription)
        self._subscription = None
        if self._owns_node:
            if self._executor is not None:
                self._executor.shutdown()
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=2.0)
            if self._node is not None:
                self._node.destroy_node()
            self._node = None
            if self._we_initialised_rclpy:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
                self._we_initialised_rclpy = False
        self._executor = None
        self._spin_thread = None
        log.info(
            "sensor.ros2_image.close",
            sensor_id=self.sensor_id,
            frames_received=self._frames_received,
            frames_dropped=self._frames_dropped,
        )

    def __enter__(self) -> Ros2ImageSensorReader:
        """Open the reader."""
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the reader (idempotent)."""
        self.close()

    # ── Hot path ────────────────────────────────────────────────────────────

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the newest buffered frame.

        Non-blocking — the subscription callback does the conversion, so this
        only snapshots the one-slot buffer.

        Args:
            max_age_ms: Staleness budget; ``None`` uses the configured default.

        Returns:
            The buffered ``SensorFrame``.

        Raises:
            RuntimeError: The reader is closed.
            ROSPerceptionStale: Nothing has arrived yet, or the newest frame is
                older than the budget.
        """
        if not self.is_open:
            raise RuntimeError(
                f"Ros2ImageSensorReader({self.sensor_id!r}).read_latest called on a closed reader"
            )
        budget_ms = self._default_max_age_ms if max_age_ms is None else max_age_ms
        with self._frame_lock:
            frame = self._latest
            mono_ns = self._latest_stamp_monotonic_ns
        if frame is None or mono_ns is None:
            raise ROSPerceptionStale(
                f"Ros2ImageSensorReader({self.sensor_id!r}): no frame received yet "
                f"on {self._topic!r}. Is the driver running, and does its QoS "
                "match (a RELIABLE subscriber gets nothing from a BEST_EFFORT "
                "publisher)?"
            )
        age_ms = (time.monotonic_ns() - mono_ns) / 1e6
        if age_ms > budget_ms:
            raise ROSPerceptionStale(
                f"Ros2ImageSensorReader({self.sensor_id!r}): freshest frame from "
                f"{self._topic!r} is {age_ms:.1f} ms old (budget {budget_ms} ms)"
            )
        return frame

    # ── Internal ────────────────────────────────────────────────────────────

    def _on_image(self, msg: Any) -> None:
        """Convert an incoming ``sensor_msgs/Image`` into the buffer slot.

        Conversion failures are counted and logged, never raised — this runs on
        the executor thread, where an exception would kill the spin loop and
        silently stop the camera. A reader that stops receiving surfaces
        through ``read_latest``'s staleness contract instead, which is the
        channel the runner already watches.
        """
        try:
            frame = self._to_frame(msg)
        except Exception as exc:  # reason: never kill the spin thread
            self._frames_dropped += 1
            log.warning(
                "sensor.ros2_image.decode_failed",
                sensor_id=self.sensor_id,
                topic=self._topic,
                encoding=getattr(msg, "encoding", "?"),
                error=str(exc),
            )
            return
        with self._frame_lock:
            self._latest = frame
            self._latest_stamp_monotonic_ns = time.monotonic_ns()
        self._frames_received += 1

    def _to_frame(self, msg: Any) -> SensorFrame:
        """Build a ``SensorFrame`` from a ``sensor_msgs/Image``.

        ``stamp_monotonic_ns`` is stamped at receipt by the caller because the
        staleness contract is monotonic; the message header carries the
        driver's own capture time, which is what lands in ``stamp_wall_ns``.

        ``msg.step`` is honoured rather than assumed equal to the packed row
        length: an Isaac/NITROS-aligned buffer or an ROI crop pads every row,
        and reshaping such a buffer to ``height x width`` raises — which
        ``_on_image`` downgrades to a WARN, so the sensor would go
        permanently stale and read as a dead camera.
        """
        encoding = str(msg.encoding)
        height, width = int(msg.height), int(msg.width)
        raw = bytes(msg.data)

        if encoding in _FLOAT_DEPTH_ENCODINGS:
            metres = _rows(raw, _byte_order(msg) + "f4", msg, height, width, 1)
            array: NDArray[Any] = _depth32f_to_depth16(metres)
            frame_encoding, channels = FrameEncoding.DEPTH16, 1
        else:
            without_alpha = _ALPHA_ENCODINGS.get(encoding)
            mapped = _DIRECT_ENCODINGS.get(without_alpha or encoding)
            if mapped is None:
                raise ROSConfigError(
                    f"unsupported sensor_msgs/Image encoding {encoding!r} on "
                    f"{self._topic!r}; supported: "
                    f"{sorted(_DIRECT_ENCODINGS) + sorted(_ALPHA_ENCODINGS)}"
                    f" + {sorted(_FLOAT_DEPTH_ENCODINGS)}"
                )
            frame_encoding, dtype, channels = mapped
            # Unpack all four planes so `step` and the row stride still line
            # up, then drop alpha. `tobytes()` re-packs the non-contiguous
            # slice, so the frame stays a tight `height x width x 3` buffer.
            array = _rows(
                raw,
                _byte_order(msg) + _dtype_code(dtype),
                msg,
                height,
                width,
                channels + 1 if without_alpha is not None else channels,
            )
            if without_alpha is not None:
                array = array[..., :channels]

        return SensorFrame(
            sensor_id=self.sensor_id,
            stamp_monotonic_ns=time.monotonic_ns(),
            stamp_wall_ns=_header_stamp_ns(msg),
            encoding=frame_encoding,
            width=width,
            height=height,
            channels=channels,
            data=bytes(array.tobytes()),
        )


def _rows(raw: bytes, dtype: str, msg: Any, height: int, width: int, channels: int) -> NDArray[Any]:
    """Unpack an ``Image`` payload into pixels, honouring the row stride.

    ``sensor_msgs/Image.step`` is the byte length of a row *as published*,
    which is only ``width * channels * itemsize`` when the publisher packs
    rows tightly. Isaac/NITROS hand out pitch-aligned buffers and an ROI crop
    keeps the parent image's stride, so the extra bytes at the end of each row
    have to be sliced off rather than folded into the picture.

    ``step`` is trusted only when it is at least the packed row length and
    the payload really holds ``height`` such rows; anything else falls back to
    the packed stride so a publisher with a wrong ``step`` fails the same way
    it did before rather than silently yielding a skewed image.

    Args:
        raw: The message payload.
        dtype: Byte-order-prefixed numpy dtype code (e.g. ``"<u2"``).
        msg: The ``sensor_msgs/Image``, read for ``step``.
        height: Image height in rows.
        width: Image width in pixels.
        channels: Values per pixel.

    Returns:
        ``(height, width)`` for single-channel images, else
        ``(height, width, channels)``.
    """
    itemsize = np.dtype(dtype).itemsize
    packed = width * channels * itemsize
    step = int(getattr(msg, "step", 0) or 0)
    if step <= packed or step % itemsize or len(raw) < step * height:
        step = packed
    flat = np.frombuffer(raw[: step * height], dtype=dtype)
    pixels = flat.reshape(height, step // itemsize)[:, : width * channels]
    return pixels if channels == 1 else pixels.reshape(height, width, channels)


def _dtype_code(dtype: str) -> str:
    """Numpy dtype name → its short code, for byte-order prefixing."""
    return {"uint8": "u1", "uint16": "u2"}[dtype]


def _byte_order(msg: Any) -> str:
    """``>`` when the publisher marked the image big-endian, else ``<``.

    Single-byte layouts are unaffected; a 16-bit depth image from a big-endian
    publisher read as little-endian is byte-swapped garbage, so the header flag
    is honoured rather than assumed.
    """
    return ">" if bool(getattr(msg, "is_bigendian", False)) else "<"


def _header_stamp_ns(msg: Any) -> int:
    """The driver's capture time in nanoseconds, or 0 when unstamped."""
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0
    return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))


def _node_safe(sensor_id: str) -> str:
    """Coerce a sensor name into a legal ROS node-name suffix."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in sensor_id)
