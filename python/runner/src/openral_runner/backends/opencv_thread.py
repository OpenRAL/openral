"""OpenCV-thread :class:`SensorReader` backend.

This backend is the default :class:`~openral_runner.SensorReader`
implementation. It mirrors the pattern lerobot uses in
``src/lerobot/cameras/opencv/camera_opencv.py``: one daemon
:class:`threading.Thread` per camera continuously calls
``cv2.VideoCapture.read()`` and posts each frame into a single
``latest_frame`` slot guarded by a ``Lock`` + ``Event``. The foreground
reader calls :meth:`read_latest` to peek at that slot without blocking.

When the freshest frame is older than the caller's ``max_age_ms`` budget,
:meth:`read_latest` raises :class:`~openral_core.exceptions.ROSPerceptionStale`
— this is the staleness contract the inference runner's `sensors.read`
span uses to short-circuit a tick.

OpenCV is gated as the ``opencv`` optional extra on
``openral-runner`` (``pip install openral-runner[opencv]``);
this module imports ``cv2`` lazily inside :meth:`open` so the runner
remains importable on hosts without it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import structlog
from openral_core import FrameEncoding, SensorFrame
from openral_core.exceptions import ROSPerceptionStale

__all__ = ["OpenCVThreadSensorReader"]

log = structlog.get_logger(__name__)

# Number of dimensions for a colour frame returned by ``cv2.VideoCapture``:
# ``(H, W, C)``. Mono frames are ``(H, W)``.
_COLOR_NDIM = 3


class OpenCVThreadSensorReader:
    """Per-camera background-thread reader wrapping ``cv2.VideoCapture``.

    The constructor only records configuration. :meth:`open` performs the
    expensive work (opens the device, configures FPS / dims, spawns the
    background capture thread). :meth:`read_latest` is non-blocking and
    returns the most recently captured frame.

    Args:
        sensor_id: Sensor name used by the inference runner to correlate
            frames with :class:`~openral_core.SensorReaderConfig`.
        device: Camera index (``int``, e.g. ``0`` for ``/dev/video0``) or a
            file path / RTSP URL accepted by ``cv2.VideoCapture``.
        fps: Requested capture rate (best-effort; hardware may run slower).
        width: Optional requested frame width in pixels.
        height: Optional requested frame height in pixels.
        encoding: Pixel encoding of the captured frames. OpenCV decodes to
            BGR8 by default; choose another value when the backend post-
            processes frames before they reach this reader.
        default_max_age_ms: Default staleness budget applied when
            :meth:`read_latest` is called with ``max_age_ms=None``.
            Defaults to ~3 frames at 30 Hz.

    Example:
        >>> # Doctest requires a real video device; exercised in
        >>> # tests/unit/test_opencv_thread_sensor_reader.py.
        >>> pass
    """

    sensor_id: str
    is_open: bool

    def __init__(
        self,
        *,
        sensor_id: str,
        device: int | str,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        encoding: FrameEncoding = FrameEncoding.BGR8,
        default_max_age_ms: int = 100,
    ) -> None:
        """Stash configuration; no I/O until :meth:`open`."""
        if fps <= 0:
            raise ValueError(f"OpenCVThreadSensorReader.fps must be > 0; got {fps}")
        if default_max_age_ms <= 0:
            raise ValueError(
                f"OpenCVThreadSensorReader.default_max_age_ms must be > 0; got {default_max_age_ms}"
            )
        self.sensor_id = sensor_id
        self._device = device
        self._fps = fps
        self._req_width = width
        self._req_height = height
        self._encoding = encoding
        self._default_max_age_ms = default_max_age_ms

        # Populated by open().
        self._cap: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._latest_stamp_monotonic_ns: int | None = None
        self._latest_stamp_wall_ns: int | None = None
        self.is_open = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the capture device and start the background thread."""
        if self.is_open:
            return
        import cv2  # lazy: opencv optional-extra

        # Mirror lerobot: pin cv2 to single-threaded mode so per-camera
        # threads don't fight inside libstdc++. Idempotent.
        cv2.setNumThreads(1)

        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCVThreadSensorReader({self.sensor_id!r}): "
                f"cv2.VideoCapture({self._device!r}) failed to open"
            )
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        if self._req_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        if self._req_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        self._cap = cap

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"OpenCVThreadSensorReader[{self.sensor_id}]",
            daemon=True,
        )
        self._thread.start()
        self.is_open = True
        log.debug(
            "opencv_thread_reader.opened",
            sensor_id=self.sensor_id,
            device=self._device,
            fps=self._fps,
        )

    def close(self) -> None:
        """Stop the background thread and release the capture device."""
        if not self.is_open:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_stamp_monotonic_ns = None
            self._latest_stamp_wall_ns = None
        self.is_open = False

    # Make the reader usable as a context manager for tidy test setups.
    def __enter__(self) -> OpenCVThreadSensorReader:
        """Open the reader and return ``self``."""
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the reader (idempotent)."""
        self.close()

    # ── Hot path ────────────────────────────────────────────────────────────

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the most recent buffered frame as a :class:`SensorFrame`.

        Non-blocking. Snapshots the current ``latest_frame`` slot under a
        lock and returns a fresh :class:`SensorFrame` with the captured
        bytes inlined as ``data`` (encoded per :attr:`_encoding`).

        Args:
            max_age_ms: Maximum acceptable frame age. ``None`` falls back to
                :attr:`_default_max_age_ms`.

        Raises:
            RuntimeError: When the reader is not open.
            ROSPerceptionStale: When no frame has been captured yet, or the
                freshest frame is older than ``max_age_ms``.
        """
        if not self.is_open:
            raise RuntimeError(
                f"OpenCVThreadSensorReader({self.sensor_id!r}).read_latest "
                f"called on a closed reader"
            )
        budget_ms = self._default_max_age_ms if max_age_ms is None else max_age_ms

        with self._frame_lock:
            frame = self._latest_frame
            mono_ns = self._latest_stamp_monotonic_ns
            wall_ns = self._latest_stamp_wall_ns
        if frame is None or mono_ns is None or wall_ns is None:
            raise ROSPerceptionStale(
                f"OpenCVThreadSensorReader({self.sensor_id!r}): no frame captured yet"
            )
        age_ms = (time.monotonic_ns() - mono_ns) / 1e6
        if age_ms > budget_ms:
            raise ROSPerceptionStale(
                f"OpenCVThreadSensorReader({self.sensor_id!r}): freshest "
                f"frame is {age_ms:.1f} ms old (budget {budget_ms} ms)"
            )

        height, width = frame.shape[:2]
        # ``cv2.VideoCapture.read`` returns ``(H, W, 3)`` for color or ``(H, W)``
        # for mono; surface that to ``SensorFrame.channels`` accurately.
        channels = frame.shape[2] if frame.ndim == _COLOR_NDIM else 1
        return SensorFrame(
            sensor_id=self.sensor_id,
            stamp_monotonic_ns=mono_ns,
            stamp_wall_ns=wall_ns,
            encoding=self._encoding,
            width=int(width),
            height=int(height),
            channels=int(channels),
            data=bytes(frame.tobytes()),
        )

    # ── Internal ────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Background thread: read frames as fast as the device allows."""
        assert self._cap is not None  # invariant: thread only runs while open
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Source exhausted (file EOF) or transient device error.
                # Sleep briefly and keep the latest-frame slot intact so
                # read_latest can still serve the most recent good frame
                # within max_age_ms. A persistent read failure surfaces as
                # ROSPerceptionStale on the foreground side.
                time.sleep(1.0 / self._fps)
                continue
            mono_ns = time.monotonic_ns()
            wall_ns = time.time_ns()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_stamp_monotonic_ns = mono_ns
                self._latest_stamp_wall_ns = wall_ns
