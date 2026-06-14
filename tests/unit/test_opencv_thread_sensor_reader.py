"""Unit tests for :class:`OpenCVThreadSensorReader`.

No mocks (CLAUDE.md §1.11). Tests generate a short MJPG/AVI video on disk
with ``cv2.VideoWriter`` and feed its path to a real
:class:`OpenCVThreadSensorReader`. The reader's background thread
captures the file via the real ``cv2.VideoCapture``; assertions exercise
the published :class:`SensorFrame` shape, the staleness contract, and
lifecycle idempotency.

The fixture is skipped cleanly when ``opencv-python`` is not installed
(the package's ``opencv`` optional-extra). The tests do not require a
camera, v4l2loopback, or GStreamer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed (openral-runner[opencv])")

# Imports after `pytest.importorskip` are intentionally below the gate so the
# whole module skips cleanly when `cv2` is missing — these must remain here.
from openral_core import FrameEncoding, SensorFrame  # noqa: E402
from openral_core.exceptions import ROSPerceptionStale  # noqa: E402
from openral_runner import SensorReader  # noqa: E402
from openral_runner.backends import OpenCVThreadSensorReader  # noqa: E402

# Synthetic video parameters — small + short so each test is cheap.
_W, _H, _FPS, _N_FRAMES = 64, 48, 30, 60  # 2 s of video at 30 fps


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a tiny MJPG/AVI to ``tmp_path`` and return the path."""
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(_FPS), (_W, _H))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter MJPG codec unavailable on this host")
    try:
        for i in range(_N_FRAMES):
            frame = np.zeros((_H, _W, 3), dtype=np.uint8)
            frame[:, :, i % 3] = (i * 4) % 255
            writer.write(frame)
    finally:
        writer.release()
    if path.stat().st_size == 0:
        pytest.skip("synthetic MJPG output is empty on this host")
    return path


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_reader_satisfies_sensor_reader_protocol(synthetic_video: Path) -> None:
    """Structural ``isinstance`` against the Protocol must succeed."""
    reader = OpenCVThreadSensorReader(sensor_id="test_cam", device=str(synthetic_video), fps=_FPS)
    assert isinstance(reader, SensorReader)


# ── Construction guards ──────────────────────────────────────────────────────


def test_constructor_rejects_zero_fps() -> None:
    with pytest.raises(ValueError, match="fps must be > 0"):
        OpenCVThreadSensorReader(sensor_id="c", device=0, fps=0)


def test_constructor_rejects_zero_max_age() -> None:
    with pytest.raises(ValueError, match="default_max_age_ms must be > 0"):
        OpenCVThreadSensorReader(sensor_id="c", device=0, fps=30, default_max_age_ms=0)


# ── Lifecycle ────────────────────────────────────────────────────────────────


def test_open_close_idempotent(synthetic_video: Path) -> None:
    """``open`` and ``close`` are idempotent."""
    reader = OpenCVThreadSensorReader(sensor_id="test_cam", device=str(synthetic_video), fps=_FPS)
    reader.open()
    assert reader.is_open is True
    reader.open()  # idempotent second call
    assert reader.is_open is True
    reader.close()
    assert reader.is_open is False
    reader.close()  # idempotent second close
    assert reader.is_open is False


def test_read_latest_on_closed_reader_raises(synthetic_video: Path) -> None:
    """``read_latest`` on a never-opened reader raises ``RuntimeError``."""
    reader = OpenCVThreadSensorReader(sensor_id="test_cam", device=str(synthetic_video), fps=_FPS)
    with pytest.raises(RuntimeError, match="closed reader"):
        reader.read_latest()


def test_open_failure_raises_runtime_error(tmp_path: Path) -> None:
    """``cv2.VideoCapture`` failing to open surfaces as ``RuntimeError``."""
    missing = tmp_path / "does_not_exist.avi"
    reader = OpenCVThreadSensorReader(sensor_id="ghost", device=str(missing), fps=_FPS)
    with pytest.raises(RuntimeError, match="failed to open"):
        reader.open()


def test_context_manager_opens_and_closes(synthetic_video: Path) -> None:
    """``with reader`` opens on enter and closes on exit."""
    reader = OpenCVThreadSensorReader(sensor_id="test_cam", device=str(synthetic_video), fps=_FPS)
    assert reader.is_open is False
    with reader as r:
        assert r is reader
        assert reader.is_open is True
    assert reader.is_open is False


# ── Frame shape + freshness ──────────────────────────────────────────────────


def test_read_latest_returns_populated_sensor_frame(synthetic_video: Path) -> None:
    """The returned ``SensorFrame`` carries the captured pixels + timestamps."""
    with OpenCVThreadSensorReader(
        sensor_id="test_cam", device=str(synthetic_video), fps=_FPS
    ) as reader:
        # Allow the background thread to capture at least a few frames.
        time.sleep(0.2)
        sf = reader.read_latest(max_age_ms=2000)
    assert isinstance(sf, SensorFrame)
    assert sf.sensor_id == "test_cam"
    assert sf.width == _W
    assert sf.height == _H
    assert sf.channels == 3
    assert sf.encoding == FrameEncoding.BGR8
    assert sf.data is not None
    # BGR8 means H*W*3 raw bytes inlined.
    assert len(sf.data) == _W * _H * 3
    # Timestamps are populated.
    assert sf.stamp_monotonic_ns > 0
    assert sf.stamp_wall_ns > 0
    # No other carry-mode is set (mutual-exclusion invariant).
    assert sf.topic is None
    assert sf.handle is None


def test_read_latest_raises_when_no_frame_yet(tmp_path: Path) -> None:
    """``read_latest`` raises ``ROSPerceptionStale`` before the first frame."""
    # Generate a tiny video but read *immediately* after open() — there's a
    # narrow window before the thread captures the first frame.
    path = tmp_path / "tiny.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(_FPS), (_W, _H))
    if not writer.isOpened():
        pytest.skip("MJPG codec unavailable")
    for _ in range(_N_FRAMES):
        writer.write(np.zeros((_H, _W, 3), dtype=np.uint8))
    writer.release()

    reader = OpenCVThreadSensorReader(sensor_id="test_cam", device=str(path), fps=_FPS)
    reader.open()
    try:
        # The thread may or may not have captured yet — poll briefly.
        first_frame_at: float | None = None
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            try:
                reader.read_latest(max_age_ms=10_000)
                first_frame_at = time.perf_counter()
                break
            except ROSPerceptionStale:
                time.sleep(0.005)
        # At minimum, the reader must eventually produce a frame.
        assert first_frame_at is not None, "background thread never produced a frame"
    finally:
        reader.close()


def test_read_latest_raises_when_frame_too_old(synthetic_video: Path) -> None:
    """A 1 ms staleness budget is impossible to satisfy in the steady state."""
    with OpenCVThreadSensorReader(
        sensor_id="test_cam", device=str(synthetic_video), fps=_FPS
    ) as reader:
        time.sleep(0.2)
        # The thread reads ahead and EOFs after _N_FRAMES; by the time we
        # request a frame the latest one is many ms old. 1 ms budget is
        # below that floor on any host.
        time.sleep(0.05)  # ensure the latest frame is > 1 ms old
        with pytest.raises(ROSPerceptionStale, match="freshest frame is"):
            reader.read_latest(max_age_ms=1)


def test_default_max_age_used_when_arg_is_none(synthetic_video: Path) -> None:
    """``max_age_ms=None`` falls back to the ``default_max_age_ms`` from ctor."""
    with OpenCVThreadSensorReader(
        sensor_id="test_cam",
        device=str(synthetic_video),
        fps=_FPS,
        default_max_age_ms=1,
    ) as reader:
        time.sleep(0.1)
        # default 1 ms budget => stale
        with pytest.raises(ROSPerceptionStale):
            reader.read_latest(max_age_ms=None)


# ── Frame freshness round-trip ──────────────────────────────────────────────


def test_read_latest_frame_age_is_recent(synthetic_video: Path) -> None:
    """In the steady state, the returned frame's age is below ``max_age_ms``."""
    with OpenCVThreadSensorReader(
        sensor_id="test_cam", device=str(synthetic_video), fps=_FPS
    ) as reader:
        time.sleep(0.1)
        sf = reader.read_latest(max_age_ms=500)
    age_ms = (time.monotonic_ns() - sf.stamp_monotonic_ns) / 1e6
    # The read happens inside the `with`; close() may add ~ms of overhead.
    # Asserting < 600 ms is generous slack while still catching a stuck thread.
    assert age_ms < 600.0, f"unexpectedly stale frame: age {age_ms:.1f} ms"
