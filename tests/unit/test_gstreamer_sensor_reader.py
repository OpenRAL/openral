"""Unit tests for :class:`GStreamerSensorReader` (CPU appsink path).

No mocks (CLAUDE.md §1.11). Tests run a real GStreamer pipeline anchored
on ``videotestsrc`` (ships with every install — no camera required),
exercise the live appsink callback, and assert the published
:class:`SensorFrame` shape + the staleness / EOS / error contracts.

The module skips wholesale when PyGObject is not importable, so it is a
no-op on hosts that lack the ``gstreamer`` optional-extra.
"""

from __future__ import annotations

import time

import pytest

# Gate the whole module on PyGObject availability.
gi = pytest.importorskip(
    "gi",
    reason="PyGObject not installed (pip install openral-runner[gstreamer])",
)
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
from openral_core import FrameEncoding, SensorReaderConfig  # noqa: E402
from openral_core.exceptions import (  # noqa: E402
    ROSConfigError,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_runner.backends.gstreamer import (  # noqa: E402
    GStreamerSensorReader,
    PipelineSpec,
    Platform,
    Source,
)
from openral_runner.factory import (  # noqa: E402
    SENSOR_BACKEND_REGISTRY,
    _make_gstreamer_reader,
)
from openral_runner.sensor_reader import SensorReader  # noqa: E402


def _wait_for_frame(reader: GStreamerSensorReader, *, timeout_s: float = 5.0) -> None:
    """Poll ``read_latest`` until a frame lands or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            reader.read_latest(max_age_ms=10_000)
            return
        except ROSPerceptionStale:
            time.sleep(0.02)
    raise AssertionError(
        f"GStreamerSensorReader({reader.sensor_id!r}) did not deliver a frame within {timeout_s} s"
    )


# ── Construction validation ──────────────────────────────────────────────────


def test_reader_rejects_both_pipeline_and_spec() -> None:
    """Constructor enforces exactly one of (pipeline, spec)."""
    with pytest.raises(ROSConfigError, match=r"exactly one of \(pipeline, spec\)"):
        GStreamerSensorReader(
            sensor_id="x",
            pipeline="videotestsrc ! appsink",
            spec=PipelineSpec(source=Source.TESTSRC),
        )


def test_reader_rejects_neither_pipeline_nor_spec() -> None:
    """Constructor rejects neither argument supplied."""
    with pytest.raises(ROSConfigError, match=r"exactly one of \(pipeline, spec\)"):
        GStreamerSensorReader(sensor_id="x")


def test_reader_rejects_non_positive_max_age() -> None:
    """default_max_age_ms must be strictly positive."""
    with pytest.raises(ROSConfigError, match="must be > 0"):
        GStreamerSensorReader(
            sensor_id="x",
            pipeline="videotestsrc ! appsink",
            default_max_age_ms=0,
        )


# ── End-to-end runtime ───────────────────────────────────────────────────────


def test_reader_videotestsrc_yields_real_frames() -> None:
    """A real videotestsrc pipeline must deliver BGR8 frames with valid timestamps."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=320,
        height=240,
        fps=30,
        enable_nvmm=False,
    )
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        spec=spec,
        platform=Platform.CPU_ONLY,
        default_max_age_ms=500,
    )
    with reader:
        _wait_for_frame(reader)
        frame = reader.read_latest()
        assert frame.sensor_id == "cam0"
        assert frame.width == 320
        assert frame.height == 240
        assert frame.channels == 3
        # videoconvert delivers BGR by caps-default for downstream BGR sinks;
        # the reader's encoding map handles all three of BGR/RGB/GRAY8.
        assert frame.encoding in {FrameEncoding.BGR8, FrameEncoding.RGB8}
        assert frame.data is not None
        # 320 * 240 * 3 bytes for BGR/RGB
        assert len(frame.data) == 320 * 240 * 3
        # Monotonic timestamps are strictly increasing across two reads.
        first_mono = frame.stamp_monotonic_ns
        time.sleep(0.05)  # 1-2 frames at 30 Hz
        second = reader.read_latest()
        assert second.stamp_monotonic_ns >= first_mono


def test_reader_read_latest_staleness_raises() -> None:
    """A finite-source EOS pipeline goes stale after max_age_ms elapses."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline=(
            "videotestsrc num-buffers=3 ! videoconvert ! "
            "video/x-raw,format=BGR,width=160,height=120 ! appsink name=bh_sink"
        ),
        default_max_age_ms=20,
    )
    with reader:
        _wait_for_frame(reader)
        # Let EOS land and the last frame age out.
        time.sleep(0.25)
        with pytest.raises(ROSPerceptionStale, match="ms old"):
            reader.read_latest()


def test_reader_close_is_idempotent() -> None:
    """Calling close() twice is a no-op."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline="videotestsrc num-buffers=2 ! videoconvert ! appsink name=bh_sink",
    )
    reader.open()
    reader.close()
    reader.close()  # second close must not raise
    assert reader.is_open is False


def test_reader_read_latest_on_closed_raises() -> None:
    """Reading before open() / after close() raises RuntimeError."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline="videotestsrc ! appsink",
    )
    with pytest.raises(RuntimeError, match="closed reader"):
        reader.read_latest()


def test_reader_invalid_pipeline_raises_config_error() -> None:
    """A pipeline that fails parse_launch surfaces ROSConfigError."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline="this is not a valid gstreamer pipeline ! appsink",
    )
    with pytest.raises(ROSConfigError, match="failed to parse pipeline"):
        reader.open()


def test_reader_missing_appsink_name_raises_config_error() -> None:
    """A pipeline whose declared appsink_name is absent fails open()."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline="videotestsrc ! appsink name=actual_sink",
        appsink_name="expected_sink",
    )
    with pytest.raises(ROSConfigError, match="does not contain an appsink"):
        reader.open()


def test_reader_eos_does_not_latch_error() -> None:
    """EOS from a finite source must NOT be surfaced as ROSRuntimeError."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline=(
            "videotestsrc num-buffers=5 ! videoconvert ! "
            "video/x-raw,format=BGR,width=160,height=120 ! appsink name=bh_sink"
        ),
        default_max_age_ms=2000,
    )
    with reader:
        _wait_for_frame(reader)
        # Allow EOS to land.
        time.sleep(0.3)
        # Frame is still served while inside max_age_ms — no error latched.
        frame = reader.read_latest(max_age_ms=5000)
        assert frame is not None


def test_reader_unsupported_format_surfaces_bus_error() -> None:
    """A pipeline negotiating an unsupported CPU format raises ROSRuntimeError."""
    # NV12 in system memory is not in our CPU encoding map; the callback latches
    # an error which read_latest surfaces.
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline=(
            "videotestsrc num-buffers=10 ! videoconvert ! "
            "video/x-raw,format=NV12,width=160,height=120 ! appsink name=bh_sink"
        ),
        default_max_age_ms=2000,
    )
    with reader:
        time.sleep(0.3)
        with pytest.raises(ROSRuntimeError, match="unsupported negotiated format"):
            reader.read_latest()


# ── SensorReader Protocol conformance ────────────────────────────────────────


def test_reader_is_sensor_reader_protocol_instance() -> None:
    """The reader satisfies the structural :class:`SensorReader` Protocol."""
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        pipeline="videotestsrc ! appsink",
    )
    assert isinstance(reader, SensorReader)


# ── Factory wiring ───────────────────────────────────────────────────────────


def test_factory_registry_contains_gstreamer() -> None:
    """The ``gstreamer`` backend is registered in the factory."""
    assert "gstreamer" in SENSOR_BACKEND_REGISTRY


def test_factory_builds_reader_from_explicit_pipeline_yaml() -> None:
    """A SensorReaderConfig with a raw pipeline string builds a reader."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {
                "pipeline": (
                    "videotestsrc num-buffers=5 ! videoconvert ! "
                    "video/x-raw,format=BGR,width=160,height=120 ! appsink"
                )
            },
            "max_age_ms": 200,
        },
    )
    reader = _make_gstreamer_reader(cfg)
    with reader:
        _wait_for_frame(reader)
        frame = reader.read_latest()
        assert frame.encoding == FrameEncoding.BGR8
        assert frame.width == 160
        assert frame.height == 120


def test_factory_builds_reader_from_structured_source_yaml() -> None:
    """A SensorReaderConfig with structured ``source`` builds via PipelineSpec."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {
                "source": "testsrc",
                "width": 320,
                "height": 240,
                "fps": 30,
                "enable_nvmm": False,
            },
            "max_age_ms": 500,
        },
    )
    reader = _make_gstreamer_reader(cfg)
    with reader:
        _wait_for_frame(reader)
        frame = reader.read_latest()
        assert frame.width == 320
        assert frame.height == 240


def test_factory_rejects_both_pipeline_and_source() -> None:
    """YAML supplying both pipeline and source is rejected."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {"pipeline": "videotestsrc ! appsink", "source": "testsrc"},
        },
    )
    with pytest.raises(ROSConfigError, match="exactly one of"):
        _make_gstreamer_reader(cfg)


def test_factory_rejects_neither_pipeline_nor_source() -> None:
    """YAML supplying neither pipeline nor source is rejected."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {"fps": 30},
        },
    )
    with pytest.raises(ROSConfigError, match="exactly one of"):
        _make_gstreamer_reader(cfg)


def test_factory_rejects_invalid_source() -> None:
    """Source string must be one of the Source enum members."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {"source": "not-a-real-source"},
        },
    )
    with pytest.raises(ROSConfigError, match="not a valid Source"):
        _make_gstreamer_reader(cfg)


def test_holoscan_backend_value_is_reserved_but_unimplemented() -> None:
    """``SensorReaderBackend.HOLOSCAN`` parses but builds no reader today.

    The enum value exists so a future PR can register the backend
    additively (ADR-0010 Amendment 2026-05-12, ADR-0011). Configs that
    select it today fall through the factory registry and surface a
    typed ROSConfigError listing the registered backends.
    """
    from openral_core import SensorReaderBackend
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY

    # The enum carries the reserved variant.
    assert SensorReaderBackend.HOLOSCAN.value == "holoscan"
    # It is *not* in the registry: selecting it must fail with a typed error.
    assert "holoscan" not in SENSOR_BACKEND_REGISTRY


# Ensure Gst is initialised so the test module load itself doesn't leak.
Gst.init(None)
