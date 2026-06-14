"""GStreamer-backed :class:`SensorReader` (CPU appsink path — ADR-0010 PR I/2).

The :class:`GStreamerSensorReader` runs a user-supplied (or
:class:`PipelineSpec`-generated) GStreamer pipeline that terminates
in an ``appsink``. It mirrors the latest-only contract of
:class:`~openral_runner.backends.opencv_thread.OpenCVThreadSensorReader`:

* :meth:`open` parses the pipeline, sets it to PLAYING, connects the
  appsink's ``new-sample`` signal to a Python callback, and spawns a
  daemon thread that drains the GStreamer bus for ERROR / EOS messages.
* The callback latches the latest mapped frame into a ``Lock``-guarded
  slot and updates monotonic + wall-clock timestamps.
* :meth:`read_latest` is non-blocking: returns the freshest slot, or
  raises :class:`ROSPerceptionStale` when no frame has arrived or the
  freshest is older than ``max_age_ms``.

This module imports ``gi.repository`` at module load and therefore
requires the ``gstreamer`` optional-extra (``pip install
openral-runner[gstreamer]``). The
:mod:`openral_runner.backends.gstreamer.pipeline` module — which
the reader's factory uses to build pipeline strings — has no such
requirement and is import-safe everywhere.

The CPU path here delivers system-memory frames as
:class:`~openral_core.SensorFrame` with ``data=bytes`` and
``encoding`` ∈ {BGR8, RGB8, MONO8}. The NVMM / CUDA zero-copy path
(commit #3) populates ``handle`` + ``encoding=CUDA_NV12`` instead and
is grafted into :meth:`_on_new_sample` without changing the Protocol
surface.
"""

from __future__ import annotations

import contextlib
import ctypes
import threading
import time
from typing import TYPE_CHECKING, Any, Final

import gi
import structlog

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402  # gi requires version-pin before import
from openral_core import FrameEncoding, SensorFrame  # noqa: E402
from openral_core.exceptions import (  # noqa: E402
    ROSConfigError,
    ROSPerceptionStale,
    ROSRuntimeError,
)

# Initialise GStreamer at module load time, NOT inside open().
#
# Why eager: when ``rclpy`` is imported into the same interpreter before
# ``Gst.init()`` runs, ``rclpy.Node()`` segfaults inside Fast DDS thread
# setup (observed reliably inside the x86-ros image and reproducible with
# a minimal probe — see PR I/8 investigation). Calling ``Gst.init()`` here
# guarantees that any later ``import rclpy`` (lazy or eager) sees an
# already-initialised GStreamer process state, which is the only ordering
# we have found that does not segfault. ``Gst.init()`` is idempotent and
# safe to call from a non-main thread, so re-imports / fork-safe wrappers
# elsewhere remain valid.
Gst.init(None)

from openral_runner.backends.gstreamer.pipeline import (  # noqa: E402
    PipelineSpec,
    Platform,
    build_pipeline_string,
    ensure_appsink_name,
)

if TYPE_CHECKING:
    from types import TracebackType

__all__ = ["GStreamerSensorReader"]

log = structlog.get_logger(__name__)

# Bus poll timeout when listening for ERROR / EOS. Short enough that
# close() returns promptly; long enough not to thrash.
_BUS_POLL_TIMEOUT_NS: Final[int] = 100_000_000  # 100 ms in ns (Gst timing is ns)

# Default staleness budget; ~3 frames at 30 Hz, matches OpenCV reader default.
_DEFAULT_MAX_AGE_MS: Final[int] = 100

# Map GStreamer caps ``format=...`` to our FrameEncoding for the CPU path.
# NV12 is intentionally absent from the CPU path: a system-memory NV12 buffer
# can be delivered but our SensorFrame consumers (Skill.step) expect BGR/RGB
# per-pixel today. Tegra / desktop NVMM NV12 → CUDA_NV12 is handled in commit #3.
_GST_FORMAT_TO_ENCODING: Final[dict[str, FrameEncoding]] = {
    "BGR": FrameEncoding.BGR8,
    "RGB": FrameEncoding.RGB8,
    "GRAY8": FrameEncoding.MONO8,
}


class GStreamerSensorReader:
    """:class:`SensorReader` backed by a GStreamer pipeline.

    Two construction modes:

    1. *Explicit pipeline string*. Pass ``pipeline=`` with a full
       GStreamer string terminating in an ``appsink``. The reader
       ensures the appsink is named (default ``bh_sink``) so it can
       look it up.
    2. *Generated from spec*. Pass ``spec=`` (a :class:`PipelineSpec`)
       and optionally ``platform=`` to override platform detection.
       The reader materialises the pipeline string itself.

    Exactly one of ``pipeline`` / ``spec`` must be supplied.

    Args:
        sensor_id: Sensor name used by the runner to correlate frames
            with :class:`~openral_core.SensorReaderConfig`.
        pipeline: Full GStreamer pipeline string (explicit mode).
        spec: :class:`PipelineSpec` to materialise (generated mode).
        platform: Override platform detection for generated mode.
        appsink_name: Name of the openral appsink. Defaults to
            ``bh_sink``; only override when supplying an explicit
            ``pipeline`` whose appsink uses a different name.
        default_max_age_ms: Default staleness budget applied when
            :meth:`read_latest` is called with ``max_age_ms=None``.

    Raises:
        ROSConfigError: When both / neither of ``pipeline`` / ``spec``
            are provided, or when the pipeline string is malformed.

    Example:
        >>> # Doctest exercised in tests/unit/test_gstreamer_sensor_reader.py
        >>> # to avoid requiring videotestsrc + Gst.init at doctest time.
        >>> pass
    """

    sensor_id: str
    is_open: bool

    def __init__(
        self,
        *,
        sensor_id: str,
        pipeline: str | None = None,
        spec: PipelineSpec | None = None,
        platform: Platform | None = None,
        appsink_name: str = "bh_sink",
        ros_appsink_name: str = "ros_sink",
        ros_topic: str | None = None,
        ros_rate_hz: float | None = None,
        default_max_age_ms: int = _DEFAULT_MAX_AGE_MS,
    ) -> None:
        """Stash configuration; no GStreamer I/O until :meth:`open`."""
        if (pipeline is None) == (spec is None):
            raise ROSConfigError(
                f"GStreamerSensorReader({sensor_id!r}) requires exactly one of "
                "(pipeline, spec); got "
                f"pipeline={'set' if pipeline else 'None'}, "
                f"spec={'set' if spec else 'None'}"
            )
        if default_max_age_ms <= 0:
            raise ROSConfigError(
                f"GStreamerSensorReader({sensor_id!r}).default_max_age_ms "
                f"must be > 0; got {default_max_age_ms}"
            )

        if spec is not None:
            self._pipeline_string = build_pipeline_string(spec, platform=platform)
            self._appsink_name = spec.appsink_name
            self._ros_appsink_name = spec.ros_appsink_name
            self._has_ros_tee = spec.enable_ros_tee
        else:
            assert pipeline is not None  # narrowed by the XOR above
            self._pipeline_string = ensure_appsink_name(pipeline, name=appsink_name)
            self._appsink_name = appsink_name
            self._ros_appsink_name = ros_appsink_name
            # In explicit-pipeline mode we can't introspect whether the string
            # has a tee branch without re-parsing it. We rely on ``ros_topic``
            # being set as the user's signal that they wired a ros_sink.
            self._has_ros_tee = ros_topic is not None
        if self._has_ros_tee and ros_topic is None:
            raise ROSConfigError(
                f"GStreamerSensorReader({sensor_id!r}): ROS tee is enabled by the "
                "pipeline spec but no ``ros_topic`` was supplied."
            )
        self._ros_topic = ros_topic
        self._ros_rate_hz = ros_rate_hz

        self.sensor_id = sensor_id
        self._default_max_age_ms = default_max_age_ms

        # Populated by open().
        self._pipeline: Gst.Pipeline | None = None
        self._appsink: Any | None = None  # GstApp.AppSink — duck-typed to keep gi-app optional
        self._ros_publisher: Any | None = None  # RosImagePublisher when tee is active
        self._bus_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_data: bytes | None = None
        self._latest_handle: int | None = None
        # The NvBufSurfaceHandle / Gst.Buffer / Gst.MapInfo are kept alive
        # for the duration of the latest-frame slot so a downstream CUDA
        # consumer doesn't read freed GPU memory. Released in close() and
        # whenever a new NVMM frame replaces the slot.
        self._latest_handle_descriptor: Any | None = None
        self._latest_buffer_ref: Any | None = None
        self._latest_map_info: Any | None = None
        self._latest_stamp_monotonic_ns: int | None = None
        self._latest_stamp_wall_ns: int | None = None
        self._latest_width: int | None = None
        self._latest_height: int | None = None
        self._latest_channels: int = 3
        self._latest_encoding: FrameEncoding | None = None
        self._bus_error: str | None = None  # latched from bus thread
        self.is_open = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def open(self) -> None:
        """Initialise GStreamer, parse the pipeline, transition to PLAYING."""
        if self.is_open:
            return
        _ensure_gst_initialised()

        try:
            element = Gst.parse_launch(self._pipeline_string)
        except GLib.Error as exc:
            raise ROSConfigError(
                f"GStreamerSensorReader({self.sensor_id!r}): failed to parse pipeline "
                f"{self._pipeline_string!r}: {exc.message}"
            ) from exc
        # parse_launch returns a Gst.Element; for our multi-element strings it is
        # actually a Gst.Pipeline. Narrow defensively.
        if not isinstance(element, Gst.Pipeline):
            element = self._wrap_in_pipeline(element)
        self._pipeline = element

        appsink = self._pipeline.get_by_name(self._appsink_name)
        if appsink is None:
            raise ROSConfigError(
                f"GStreamerSensorReader({self.sensor_id!r}): pipeline does not "
                f"contain an appsink named {self._appsink_name!r}"
            )
        self._appsink = appsink
        # Even when the caller pre-set emit-signals via the pipeline string,
        # force the latest-only properties — defensive against truncated strings.
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.connect("new-sample", self._on_new_sample)

        # ROS publisher MUST be started before transitioning the pipeline to
        # PLAYING. rclpy's DDS init spawns threads that segfault if any
        # GStreamer streaming thread is already running on the same
        # interpreter when rclpy.init()/create_node() runs — observed as
        # SIGSEGV inside the x86-ros container with ros-jazzy + Fast DDS.
        # See probe in PR I/8.
        if self._has_ros_tee:
            self._start_ros_publisher()

        self._stop_event.clear()
        self._bus_thread = threading.Thread(
            target=self._bus_loop,
            name=f"GStreamerSensorReader[{self.sensor_id}].bus",
            daemon=True,
        )
        self._bus_thread.start()

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            # Clean up partially-opened pipeline before raising.
            self._teardown_pipeline()
            raise ROSConfigError(
                f"GStreamerSensorReader({self.sensor_id!r}): pipeline failed to "
                f"transition to PLAYING (pipeline={self._pipeline_string!r})"
            )

        self.is_open = True
        log.debug(
            "gstreamer_reader.opened",
            sensor_id=self.sensor_id,
            pipeline=self._pipeline_string,
            ros_tee=self._has_ros_tee,
        )

    def _start_ros_publisher(self) -> None:
        """Look up the ros_sink appsink and start the ROS publisher branch.

        Tears the pipeline back down with an actionable ROSConfigError when
        rclpy isn't available — the caller asked for a ROS tee and we can't
        deliver it.
        """
        assert self._pipeline is not None  # invariant in open() flow
        assert self._ros_topic is not None  # validated in __init__
        ros_appsink = self._pipeline.get_by_name(self._ros_appsink_name)
        if ros_appsink is None:
            self._teardown_pipeline()
            raise ROSConfigError(
                f"GStreamerSensorReader({self.sensor_id!r}): ROS tee enabled but "
                f"pipeline does not contain an appsink named {self._ros_appsink_name!r}"
            )
        from openral_runner.backends.gstreamer.ros_tee import (  # noqa: PLC0415
            RosImagePublisher,
        )

        publisher = RosImagePublisher(
            sensor_id=self.sensor_id,
            appsink=ros_appsink,
            topic=self._ros_topic,
            rate_hz=self._ros_rate_hz,
        )
        try:
            publisher.start()
        except RuntimeError as exc:
            self._teardown_pipeline()
            raise ROSConfigError(
                f"GStreamerSensorReader({self.sensor_id!r}): ROS tee start failed: {exc}"
            ) from exc
        self._ros_publisher = publisher

    def close(self) -> None:
        """Stop the pipeline, join the bus thread, release the appsink."""
        if not self.is_open:
            return
        if self._ros_publisher is not None:
            with contextlib.suppress(Exception):  # reason: defensive ROS shutdown
                self._ros_publisher.stop()
            self._ros_publisher = None
        self._teardown_pipeline()
        with self._frame_lock:
            buffer_ref = self._latest_buffer_ref
            map_info = self._latest_map_info
            self._latest_data = None
            self._latest_handle = None
            self._latest_handle_descriptor = None
            self._latest_buffer_ref = None
            self._latest_map_info = None
            self._latest_stamp_monotonic_ns = None
            self._latest_stamp_wall_ns = None
            self._latest_width = None
            self._latest_height = None
            self._latest_encoding = None
            self._bus_error = None
        # Release the held NVMM buffer ref outside the lock — unmap can be slow.
        if buffer_ref is not None and map_info is not None:
            with contextlib.suppress(Exception):  # reason: defensive cleanup
                buffer_ref.unmap(map_info)
        self.is_open = False

    def __enter__(self) -> GStreamerSensorReader:
        """Open the reader and return ``self``."""
        self.open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Close the reader (idempotent)."""
        self.close()

    # ── Hot path ────────────────────────────────────────────────────────────

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the most recent buffered frame as a :class:`SensorFrame`.

        Args:
            max_age_ms: Maximum acceptable frame age. ``None`` falls back to
                the constructor's ``default_max_age_ms``.

        Returns:
            A populated :class:`SensorFrame` whose ``data`` field holds
            the pixel bytes and ``encoding`` reflects the negotiated
            caps format.

        Raises:
            RuntimeError: When called on a closed reader.
            ROSRuntimeError: When the GStreamer bus has surfaced an error
                or EOS-as-error (bus thread latches the error message).
            ROSPerceptionStale: When no frame has arrived yet, or the
                freshest frame is older than ``max_age_ms``.
        """
        if not self.is_open:
            raise RuntimeError(
                f"GStreamerSensorReader({self.sensor_id!r}).read_latest called on a closed reader"
            )
        budget_ms = self._default_max_age_ms if max_age_ms is None else max_age_ms

        with self._frame_lock:
            bus_error = self._bus_error
            data = self._latest_data
            handle = self._latest_handle
            handle_descriptor = self._latest_handle_descriptor
            mono_ns = self._latest_stamp_monotonic_ns
            wall_ns = self._latest_stamp_wall_ns
            width = self._latest_width
            height = self._latest_height
            encoding = self._latest_encoding
            channels = self._latest_channels
        if bus_error is not None:
            raise ROSRuntimeError(
                f"GStreamerSensorReader({self.sensor_id!r}): GStreamer bus reported "
                f"error: {bus_error}"
            )
        if (
            (data is None and handle is None)
            or mono_ns is None
            or wall_ns is None
            or width is None
            or height is None
            or encoding is None
        ):
            raise ROSPerceptionStale(
                f"GStreamerSensorReader({self.sensor_id!r}): no frame captured yet"
            )
        age_ms = (time.monotonic_ns() - mono_ns) / 1e6
        if age_ms > budget_ms:
            raise ROSPerceptionStale(
                f"GStreamerSensorReader({self.sensor_id!r}): freshest frame is "
                f"{age_ms:.1f} ms old (budget {budget_ms} ms)"
            )
        frame_metadata: dict[str, object] = {}
        if handle_descriptor is not None:
            frame_metadata["nvbufsurface"] = handle_descriptor.model_dump()
        kwargs: dict[str, object] = {
            "sensor_id": self.sensor_id,
            "stamp_monotonic_ns": mono_ns,
            "stamp_wall_ns": wall_ns,
            "encoding": encoding,
            "width": width,
            "height": height,
            "channels": channels,
            "metadata": frame_metadata,
        }
        # Exactly one of (data, handle) is populated — SensorFrame's
        # invariant enforces that on construction.
        if handle is not None:
            kwargs["handle"] = handle
        else:
            kwargs["data"] = data
        return SensorFrame(**kwargs)

    # ── Internal ────────────────────────────────────────────────────────────

    def _on_new_sample(self, appsink: Any) -> int:  # noqa: ANN401  # reason: GstApp.AppSink — duck-typed to keep the gst-app introspection optional
        """Callback fired by GStreamer when a new buffer hits the appsink.

        Runs on a GStreamer streaming thread; must remain non-blocking.
        Branches on the negotiated caps: ``memory:NVMM`` features take
        the zero-copy GPU path (populates ``handle``); plain
        ``video/x-raw`` takes the CPU path (copies pixel bytes into
        ``data``).

        Returns:
            ``Gst.FlowReturn.OK`` (as an int — gi maps the enum) on
            success; any other value would tell upstream to error out.
        """
        sample = appsink.emit("pull-sample")
        if sample is None:  # pragma: no cover — only happens after EOS
            return int(Gst.FlowReturn.OK)
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None:  # pragma: no cover — defensive; samples carry caps
            return int(Gst.FlowReturn.OK)

        # Detect NVMM caps feature (``video/x-raw(memory:NVMM)``).
        features = caps.get_features(0) if caps is not None else None
        is_nvmm = bool(features is not None and features.contains("memory:NVMM"))
        buffer = sample.get_buffer()
        if is_nvmm:
            return self._handle_nvmm_buffer(buffer, structure)
        return self._handle_cpu_buffer(buffer, structure)

    def _handle_cpu_buffer(self, buffer: Any, structure: Any) -> int:  # noqa: ANN401  # reason: Gst.Buffer / Gst.Structure — duck-typed
        """System-memory CPU path: map → copy → latch ``data``."""
        gst_format = structure.get_string("format") or ""
        encoding = _GST_FORMAT_TO_ENCODING.get(gst_format)
        if encoding is None:
            with self._frame_lock:
                self._bus_error = (
                    f"unsupported negotiated format {gst_format!r}; "
                    "CPU path supports BGR / RGB / GRAY8 only "
                    "(NVMM NV12 takes the zero-copy path)"
                )
            return int(Gst.FlowReturn.ERROR)
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h):  # pragma: no cover — caps always carry w/h
            return int(Gst.FlowReturn.OK)
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:  # pragma: no cover — mapping failure signals a Gst bug
            return int(Gst.FlowReturn.OK)
        try:
            # bytes(map_info.data) copies so the buffer can be unmapped
            # immediately and upstream is free to recycle.
            payload = bytes(map_info.data)
        finally:
            buffer.unmap(map_info)

        mono_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        channels = 1 if encoding is FrameEncoding.MONO8 else 3
        with self._frame_lock:
            self._latest_data = payload
            self._latest_handle = None
            self._latest_handle_descriptor = None
            self._latest_buffer_ref = None
            self._latest_stamp_monotonic_ns = mono_ns
            self._latest_stamp_wall_ns = wall_ns
            self._latest_width = int(width)
            self._latest_height = int(height)
            self._latest_channels = channels
            self._latest_encoding = encoding
        return int(Gst.FlowReturn.OK)

    def _handle_nvmm_buffer(self, buffer: Any, structure: Any) -> int:  # noqa: ANN401  # reason: Gst.Buffer / Gst.Structure — duck-typed
        """NVMM zero-copy path: map → wrap as :class:`NvBufSurfaceHandle` → latch ``handle``.

        Holds a reference to the GStreamer buffer for the lifetime of
        the latched slot (released when the next frame arrives) so the
        underlying GPU memory isn't recycled while a consumer reads
        through the handle.
        """
        # Lazy import: keeps the CPU path independent of libnvbufsurface.
        from openral_runner.backends.gstreamer.nvbufsurface import (  # noqa: PLC0415
            NvBufSurfaceLibraryError,
            load,
            wrap_buffer,
        )

        try:
            load()
        except NvBufSurfaceLibraryError as exc:
            with self._frame_lock:
                self._bus_error = (
                    f"NVMM caps negotiated but libnvbufsurface.so is unavailable: {exc}"
                )
            return int(Gst.FlowReturn.ERROR)
        # Width / height for the NVMM path come from the NvBufSurface itself,
        # not the caps — keeping the source-of-truth consistent with the GPU
        # buffer rather than the negotiated caps which may lag.
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:  # pragma: no cover — mapping failure signals a Gst bug
            return int(Gst.FlowReturn.OK)
        try:
            # map_info.data is a memoryview onto the GstBuffer's contents;
            # for NVMM that's the NvBufSurface struct. NVMM buffers map READ-only,
            # so ctypes.from_buffer (which needs a writable buffer) raises
            # "underlying buffer is not writable"; from_buffer_copy copies only the
            # small surface-descriptor struct (NOT the GPU frame — dataPtr still
            # points at device memory, so this stays zero-copy for the frame).
            # ``struct_bytes`` must outlive the wrap_buffer call: wrap_buffer derefs
            # surface_list, whose pointer indexes back into the still-mapped buffer
            # (this handler holds the map alive across frames, below). Validated for
            # the detector NVMM path against real DeepStream buffers in the ds-on
            # container (see DetectorRunner._on_sample_nvmm).
            struct_bytes = (ctypes.c_uint8 * map_info.size).from_buffer_copy(map_info.data)
            buffer_address = ctypes.cast(struct_bytes, ctypes.c_void_p).value
            if buffer_address is None:
                raise ValueError("NVMM mapped buffer has NULL base address")
            handle = wrap_buffer(buffer_address)
        except (ValueError, OSError) as exc:
            with self._frame_lock:
                self._bus_error = f"NVMM buffer unwrap failed: {exc}"
            buffer.unmap(map_info)
            return int(Gst.FlowReturn.ERROR)
        finally:
            # Keep the NvBufSurface readable until the next frame replaces
            # us — unmap below in the slot swap.
            pass

        mono_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        with self._frame_lock:
            # Release the previous buffer's map BEFORE overwriting the slot.
            prev_buffer = self._latest_buffer_ref
            prev_map = self._latest_map_info
            self._latest_data = None
            self._latest_handle = handle.gpu_ptr
            self._latest_handle_descriptor = handle
            self._latest_buffer_ref = buffer
            self._latest_map_info = map_info
            self._latest_stamp_monotonic_ns = mono_ns
            self._latest_stamp_wall_ns = wall_ns
            self._latest_width = handle.width
            self._latest_height = handle.height
            # NV12 is semi-planar Y + UV interleaved → 1.5 bytes/pixel; reported
            # as 3 channels because consumers typically want a 3-channel CUDA view.
            self._latest_channels = 3
            self._latest_encoding = FrameEncoding.CUDA_NV12
        if prev_buffer is not None and prev_map is not None:
            with contextlib.suppress(Exception):  # reason: defensive cleanup
                prev_buffer.unmap(prev_map)
        return int(Gst.FlowReturn.OK)

    def _bus_loop(self) -> None:
        """Drain the GStreamer bus for ERROR / EOS while the reader is open."""
        assert self._pipeline is not None  # invariant: thread only runs while open
        bus = self._pipeline.get_bus()
        mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while not self._stop_event.is_set():
            msg = bus.timed_pop_filtered(_BUS_POLL_TIMEOUT_NS, mask)
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                error_text = f"{err.message} (debug: {debug or 'none'})"
                with self._frame_lock:
                    # Preserve a more-specific error already latched by the
                    # callback (e.g. "unsupported negotiated format"); the
                    # upstream bus error is then a cascaded effect.
                    if self._bus_error is None:
                        self._bus_error = error_text
                log.error(
                    "gstreamer_reader.bus_error",
                    sensor_id=self.sensor_id,
                    error=err.message,
                    debug=debug,
                )
                return
            if msg.type == Gst.MessageType.EOS:
                # EOS is normal for finite sources (num-buffers, file replay).
                # We keep the last frame in the slot — read_latest will continue
                # serving until it ages out under max_age_ms. No error latched.
                log.debug("gstreamer_reader.eos", sensor_id=self.sensor_id)
                return

    def _teardown_pipeline(self) -> None:
        """Drop the pipeline and join the bus thread. Used by close() and open() rollback."""
        self._stop_event.set()
        if self._bus_thread is not None and self._bus_thread.is_alive():
            self._bus_thread.join(timeout=2.0)
        self._bus_thread = None
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._appsink = None

    @staticmethod
    def _wrap_in_pipeline(element: Gst.Element) -> Gst.Pipeline:
        """Wrap a bare element returned by parse_launch in a Pipeline bin.

        ``Gst.parse_launch`` returns a single :class:`Gst.Element` only when
        the string is a single element (e.g. ``"appsink"``). For our multi-
        element pipelines it returns a :class:`Gst.Pipeline` already, so this
        helper is hit only by edge-case tests / single-element specs.
        """
        pipeline = Gst.Pipeline.new(None)
        pipeline.add(element)
        return pipeline


_GST_INIT_LOCK = threading.Lock()
_GST_INITIALISED = False


def _ensure_gst_initialised() -> None:
    """Call ``Gst.init`` exactly once per process, thread-safely."""
    global _GST_INITIALISED  # noqa: PLW0603  # reason: one-shot init flag
    if _GST_INITIALISED:
        return
    with _GST_INIT_LOCK:
        if _GST_INITIALISED:
            return
        Gst.init(None)
        _GST_INITIALISED = True
