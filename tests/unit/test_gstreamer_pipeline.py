"""Unit tests for the GStreamer pipeline-string builder + platform detect.

No mocks (CLAUDE.md §1.11). The tests do not import ``gi`` and do not
launch a real pipeline; they only exercise the pure-Python string
builder, the regex / Pydantic validation rules, and the real
``gst-inspect-1.0`` probe (skipped when GStreamer is not installed).

Runtime behaviour of the actual reader (open / read_latest / staleness)
is covered by ``tests/unit/test_gstreamer_sensor_reader.py`` (commit #2),
which does require the ``gstreamer`` optional-extra.
"""

from __future__ import annotations

import shutil

import pytest
from openral_runner.backends.gstreamer import (
    LEAKY_BRANCH_QUEUE,
    TEE_NAME,
    PipelineSpec,
    Platform,
    Source,
    build_pipeline_string,
    detect_platform,
    ensure_appsink_name,
    inspect_element_present,
    leaky_branch,
    nvmm_convert_element,
)

# ── detect_platform ──────────────────────────────────────────────────────────


def test_detect_platform_returns_a_known_value() -> None:
    """Detect must return one of the three enum members on any host."""
    assert detect_platform() in {
        Platform.TEGRA,
        Platform.NVIDIA_DESKTOP,
        Platform.CPU_ONLY,
    }


def test_detect_platform_is_cached() -> None:
    """``lru_cache`` returns the same instance across calls."""
    assert detect_platform() is detect_platform()


# ── inspect_element_present ───────────────────────────────────────────────────


def test_inspect_element_present_returns_false_for_unknown_element() -> None:
    """A made-up element name must report absent."""
    assert inspect_element_present("definitely_not_a_real_gst_element_xyz_42") is False


@pytest.mark.skipif(
    shutil.which("gst-inspect-1.0") is None,
    reason="gst-inspect-1.0 not on PATH (GStreamer not installed)",
)
def test_inspect_element_present_finds_videotestsrc() -> None:
    """``videotestsrc`` ships with every GStreamer install; must be detected."""
    assert inspect_element_present("videotestsrc") is True


def test_inspect_element_present_no_gst_inspect_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``gst-inspect-1.0`` is missing entirely, probe returns False, not raises."""
    monkeypatch.setattr(
        "openral_runner.backends.gstreamer.pipeline.shutil.which",
        lambda _name: None,
    )
    assert inspect_element_present("videotestsrc") is False


# ── nvmm_convert_element ──────────────────────────────────────────────────────


def test_nvmm_convert_element_prefers_nvvideoconvert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both NVMM converters are registered, DeepStream's ``nvvideoconvert`` wins."""
    import openral_runner.backends.gstreamer.pipeline as p

    monkeypatch.setattr(p, "inspect_element_present", lambda _name: True)
    assert p.nvmm_convert_element() == "nvvideoconvert"


def test_nvmm_convert_element_falls_back_to_nvvidconv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Tegra/L4T (no ``nvvideoconvert``), the probe resolves ``nvvidconv``."""
    import openral_runner.backends.gstreamer.pipeline as p

    monkeypatch.setattr(p, "inspect_element_present", lambda name: name == "nvvidconv")
    assert p.nvmm_convert_element() == "nvvidconv"


def test_nvmm_convert_element_none_when_neither_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with no NVMM converter (e.g. CPU-only) yields ``None``."""
    import openral_runner.backends.gstreamer.pipeline as p

    monkeypatch.setattr(p, "inspect_element_present", lambda _name: False)
    assert p.nvmm_convert_element() is None


def test_nvmm_convert_element_reexported_from_package() -> None:
    """The package re-export is the same object as the module-level helper."""
    import openral_runner.backends.gstreamer.pipeline as p

    assert nvmm_convert_element is p.nvmm_convert_element


# ── ensure_appsink_name ──────────────────────────────────────────────────────


def test_ensure_appsink_name_injects_default_name() -> None:
    """Pipeline ending in unnamed appsink gets ``name=bh_sink`` injected."""
    result = ensure_appsink_name("videotestsrc ! videoconvert ! appsink")
    assert result == "videotestsrc ! videoconvert ! appsink name=bh_sink"


def test_ensure_appsink_name_preserves_existing_name() -> None:
    """Pipeline whose appsink already has ``name=...`` is returned unchanged."""
    pipeline = "v4l2src ! appsink name=already_named"
    assert ensure_appsink_name(pipeline) == pipeline


def test_ensure_appsink_name_custom_name() -> None:
    """Caller-supplied name lands on the trailing appsink."""
    result = ensure_appsink_name("videotestsrc ! appsink", name="my_sink")
    assert result == "videotestsrc ! appsink name=my_sink"


def test_ensure_appsink_name_with_existing_properties() -> None:
    """Property string before name= is preserved; name appended to the end."""
    result = ensure_appsink_name("videotestsrc ! appsink emit-signals=true sync=false")
    assert "emit-signals=true" in result
    assert "sync=false" in result
    assert "name=bh_sink" in result


def test_ensure_appsink_name_raises_when_no_trailing_appsink() -> None:
    """Pipeline that doesn't end in an appsink is rejected."""
    with pytest.raises(ValueError, match="terminate in an 'appsink'"):
        ensure_appsink_name("videotestsrc ! videoconvert ! filesink location=out.raw")


# ── PipelineSpec validation ──────────────────────────────────────────────────


def test_pipeline_spec_rejects_invalid_appsink_name() -> None:
    """GStreamer element names must match a strict identifier regex."""
    with pytest.raises(ValueError, match="not a valid GStreamer element name"):
        PipelineSpec(source=Source.TESTSRC, appsink_name="bad name with spaces")


def test_pipeline_spec_rejects_non_positive_fps() -> None:
    """``fps`` must be a positive integer."""
    with pytest.raises(ValueError):
        PipelineSpec(source=Source.TESTSRC, fps=0)


def test_pipeline_spec_is_frozen() -> None:
    """Specs are immutable so they can be cached / hashed safely."""
    spec = PipelineSpec(source=Source.TESTSRC)
    with pytest.raises(ValueError):
        spec.fps = 60  # type: ignore[misc]


# ── build_pipeline_string ────────────────────────────────────────────────────


def test_build_pipeline_string_testsrc_cpu_only() -> None:
    """videotestsrc → videoconvert → appsink on CPU-only platform."""
    spec = PipelineSpec(source=Source.TESTSRC, width=320, height=240, fps=30)
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert result.startswith("videotestsrc is-live=true")
    assert "video/x-raw,format=BGR,width=320,height=240,framerate=30/1" in result
    assert " ! videoconvert ! " in result
    assert result.endswith(
        "appsink name=bh_sink emit-signals=true max-buffers=1 drop=true sync=false"
    )


def test_build_pipeline_string_testsrc_nvmm_dropped_on_cpu_only() -> None:
    """``enable_nvmm=True`` is silently ignored on CPU-only (no NVMM caps emitted)."""
    spec = PipelineSpec(source=Source.TESTSRC, enable_nvmm=True)
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert "memory:NVMM" not in result


def test_build_pipeline_string_usb_tegra_uses_nvvidconv_and_nvmm() -> None:
    """USB on Tegra: v4l2src → nvvidconv → NVMM caps → appsink."""
    spec = PipelineSpec(source=Source.USB, device=0, width=1280, height=720, fps=30)
    result = build_pipeline_string(spec, platform=Platform.TEGRA)
    assert "v4l2src device=/dev/video0" in result
    assert " ! nvvidconv ! " in result
    assert "video/x-raw(memory:NVMM)" in result


def test_build_pipeline_string_usb_desktop_nvidia_uses_videoconvert_no_nvmm() -> None:
    """USB on desktop NVIDIA: v4l2src → videoconvert → system-memory caps → appsink.

    The open-core image does not bundle NVIDIA DeepStream (ADR-0010
    Amendment 2026-05-12), so ``nvvideoconvert`` is unavailable and
    no element produces NVMM caps after the convert step. The
    pipeline must therefore stay on system-memory BGR / videoconvert
    on this branch; ``nvh264dec`` etc. still GPU-accelerate the
    decode side.
    """
    spec = PipelineSpec(source=Source.USB, device="/dev/video2", fps=60)
    result = build_pipeline_string(spec, platform=Platform.NVIDIA_DESKTOP)
    assert "v4l2src device=/dev/video2" in result
    assert " ! videoconvert ! " in result
    assert "nvvideoconvert" not in result, (
        "nvvideoconvert is DeepStream-only; must not appear in open-core builds"
    )
    assert "video/x-raw(memory:NVMM)" not in result, (
        "NVMM caps require nvvidconv (Tegra) or nvvideoconvert (DeepStream); "
        "open-core x86 must use system memory"
    )


def test_build_pipeline_string_csi_requires_tegra() -> None:
    """Source.CSI on a non-Tegra platform must fail loudly, not silently swap elements."""
    spec = PipelineSpec(source=Source.CSI, device=0)
    with pytest.raises(ValueError, match=r"Source\.CSI requires Platform\.TEGRA"):
        build_pipeline_string(spec, platform=Platform.NVIDIA_DESKTOP)


def test_build_pipeline_string_csi_tegra_uses_nvarguscamerasrc() -> None:
    """Source.CSI on Tegra emits nvarguscamerasrc with sensor-id."""
    spec = PipelineSpec(source=Source.CSI, device=1)
    result = build_pipeline_string(spec, platform=Platform.TEGRA)
    assert result.startswith("nvarguscamerasrc sensor-id=1")


def test_build_pipeline_string_rtsp_desktop_nvidia_uses_nvh264dec() -> None:
    """RTSP on desktop NVIDIA: rtspsrc → rtph264depay → h264parse → nvh264dec → ..."""
    spec = PipelineSpec(source=Source.RTSP, device="rtsp://10.0.0.1:8554/cam0")
    result = build_pipeline_string(spec, platform=Platform.NVIDIA_DESKTOP)
    assert "rtspsrc location=rtsp://10.0.0.1:8554/cam0 latency=0" in result
    assert "rtph264depay ! h264parse ! nvh264dec" in result


def test_build_pipeline_string_rtsp_cpu_only_uses_avdec() -> None:
    """RTSP on CPU-only falls back to avdec_h264."""
    spec = PipelineSpec(source=Source.RTSP, device="rtsp://host/stream")
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert "rtph264depay ! h264parse ! avdec_h264" in result
    assert "nvh264dec" not in result


def test_build_pipeline_string_rtsp_without_device_raises() -> None:
    """Source.RTSP needs a location."""
    spec = PipelineSpec(source=Source.RTSP)
    with pytest.raises(ValueError, match=r"Source\.RTSP requires"):
        build_pipeline_string(spec, platform=Platform.CPU_ONLY)


def test_build_pipeline_string_file_uses_decodebin() -> None:
    """File source uses decodebin (auto-decoder picker)."""
    spec = PipelineSpec(source=Source.FILE, device="/tmp/foo.mp4")
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert "filesrc location=/tmp/foo.mp4" in result
    assert " ! decodebin ! " in result


def test_build_pipeline_string_ros_tee_emits_tee_with_two_branches() -> None:
    """``enable_ros_tee=True`` inserts the named bus tee with two branches."""
    spec = PipelineSpec(source=Source.TESTSRC, fps=30, enable_ros_tee=True, enable_nvmm=False)
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    # The single per-camera tee is the named perception-bus attach point (ADR-0037).
    assert f"tee name={TEE_NAME}" in result
    # ral appsink and ros appsink are both present
    assert "appsink name=bh_sink" in result
    assert "appsink name=ros_sink" in result
    # ROS branch forces BGR caps so the publisher gets system memory
    assert "video/x-raw,format=BGR" in result
    # Every branch references the named tee and carries the shared leaky queue.
    assert result.count(f"{TEE_NAME}. ! {LEAKY_BRANCH_QUEUE}") == 2


def test_leaky_branch_builds_isolated_tee_branch() -> None:
    """``leaky_branch`` prefixes a branch body with the named tee + leaky queue.

    This is the single isolation primitive shared by the static builder and the
    runtime TeeManager (ADR-0037), so a dynamically attached consumer carries
    the same backpressure isolation as the static legs.
    """
    branch = leaky_branch("appsink name=det_sink")
    assert branch == f"{TEE_NAME}. ! {LEAKY_BRANCH_QUEUE} ! appsink name=det_sink"
    # A custom tee name (e.g. a second camera) threads through.
    assert leaky_branch("fakesink", tee_name="cam1_tee").startswith("cam1_tee. ! ")
    # Every branch the static builder emits is exactly a leaky_branch.
    spec = PipelineSpec(source=Source.TESTSRC, enable_ros_tee=True, enable_nvmm=False)
    result = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    policy_appsink = "appsink name=bh_sink emit-signals=true max-buffers=1 drop=true sync=false"
    assert leaky_branch(policy_appsink) in result


def test_build_pipeline_string_always_ends_in_appsink_named() -> None:
    """Every emitted string includes an appsink with a name attribute."""
    for source in (Source.TESTSRC, Source.USB, Source.FILE):
        spec = PipelineSpec(source=source, device="/tmp/x.mp4" if source is Source.FILE else 0)
        for platform in Platform:
            if source is Source.CSI and platform is not Platform.TEGRA:
                continue
            result = build_pipeline_string(spec, platform=platform)
            assert "appsink name=" in result, (
                f"missing appsink name= for {source} on {platform}: {result}"
            )


def test_build_pipeline_string_is_parseable_by_gst_parse_launch_smoke() -> None:
    """Sanity: the emitted string is syntactically valid GStreamer.

    Skipped when ``gi`` is not importable; this is a syntactic check only
    (we don't set the pipeline to PLAYING), so it does not require the
    NVIDIA plugins or any camera.
    """
    gi = pytest.importorskip("gi", reason="PyGObject not installed (openral-runner[gstreamer])")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    spec = PipelineSpec(source=Source.TESTSRC, width=320, height=240, fps=30)
    pipeline_str = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    pipeline = Gst.parse_launch(pipeline_str)
    assert pipeline is not None
    assert pipeline.get_by_name("bh_sink") is not None
