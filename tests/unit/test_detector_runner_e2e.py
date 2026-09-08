"""End-to-end tests for ``DetectorRunner`` — live, no mocks (CLAUDE.md §1.11).

Live e2e: real ``rskills/rtdetr-coco-r18/rskill.yaml``, a live ``videotestsrc``
GStreamer pipeline with a named bus tee, a deterministic 4-class ONNX (shared
with ``test_objects_detector.py``); asserts detections, model_id, tee
attach/detach, and that the pipeline survives ``stop()``.

Non-live: a ``kind: vla`` manifest raises ``ROSConfigError`` at
construction, before any GStreamer call.

Gates: skips if ``gi``, ``onnxruntime``, or ``onnx`` is absent.
"""

from __future__ import annotations

import pathlib
import time

import pytest
import yaml

gi = pytest.importorskip("gi")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
from openral_core import ObjectsMetadata  # noqa: E402
from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_core.schemas import RSkillManifest  # noqa: E402
from openral_runner.backends.gstreamer.detector_runner import DetectorRunner  # noqa: E402
from openral_runner.backends.gstreamer.objects_detector import DetectorTier  # noqa: E402
from openral_runner.backends.gstreamer.pipeline import (  # noqa: E402
    PipelineSpec,
    Platform,
    Source,
    build_pipeline_string,
)

from tests.unit.conftest import _write_rtdetr_like_onnx  # noqa: E402

Gst.init(None)

# ── Repo-root path helper ──────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


# 4-class labels that match tests.unit.conftest._write_rtdetr_like_onnx's fixture.
# COCO indices 0=person, 2=car — so the 4-class ["person","bicycle","car","dog"]
# slice aligns with COCO: q0 cls-2 → "car", q1 cls-0 → "person".
_LABELS_4 = ["person", "bicycle", "car", "dog"]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Write the deterministic 4-class ONNX once per module and return the path."""
    p = tmp_path_factory.mktemp("onnx_e2e") / "rtdetr_e2e.onnx"
    _write_rtdetr_like_onnx(p)
    return p


@pytest.fixture(scope="module")
def rtdetr_manifest() -> RSkillManifest:
    """Load and validate the real rtdetr-coco-r18 rskill.yaml manifest."""
    fixture_path = _REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
    with open(fixture_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    manifest = RSkillManifest.model_validate(data)
    assert manifest.kind == "detector"
    assert manifest.detector is not None
    return manifest


# ── Live end-to-end test ───────────────────────────────────────────────────────


class TestDetectorRunnerE2E:
    """Live end-to-end tests requiring GStreamer, onnxruntime, and onnx."""

    def test_detector_runner_live_pipeline(
        self,
        onnx_path: pathlib.Path,
        rtdetr_manifest: RSkillManifest,
    ) -> None:
        """Full pipeline: videotestsrc → tee → BGR branch → ObjectsDetector → callback.

        4-class ONNX emits ``car`` (conf ≈0.953, COCO idx 2) and ``person`` (conf
        ≈0.881, idx 0) every frame, above the manifest's 0.5 threshold. Manifest uses
        ``score_threshold: 0.7`` / 80 COCO labels; only classes 0,2 activate here.
        """
        # Build a live CPU pipeline with the named bus tee.
        spec = PipelineSpec(
            source=Source.TESTSRC,
            fps=30,
            width=640,
            height=480,
            enable_ros_tee=True,  # inserts tee name=openral_cam_tee
            enable_nvmm=False,
        )
        pipeline_str = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
        pipeline = Gst.parse_launch(pipeline_str)
        assert pipeline is not None, "Gst.parse_launch returned None"

        pipeline.set_state(Gst.State.PLAYING)
        ret, _state, _pending = pipeline.get_state(Gst.SECOND)
        assert ret == Gst.StateChangeReturn.SUCCESS, f"Pipeline did not reach PLAYING: {ret}"

        collected: list[ObjectsMetadata] = []

        runner = DetectorRunner(
            pipeline,
            rtdetr_manifest,
            onnx_path=onnx_path,
            sensor_id="cam0",
            on_detection=collected.append,
        )

        try:
            runner.start()

            # Assert the branch was attached and the appsink is reachable.
            assert pipeline.get_by_name("cam0_det_sink") is not None, (
                "appsink 'cam0_det_sink' not found after runner.start()"
            )

            # Pump: wait up to 5 s for at least one detection to arrive.
            # The streaming thread fires the callback asynchronously; videotestsrc
            # delivers a frame every ~33 ms so we expect the first hit in <200 ms.
            deadline = time.monotonic() + 5.0
            while not collected and time.monotonic() < deadline:
                time.sleep(0.05)

            assert collected, "No ObjectsMetadata received within 5 s — callback never fired"

            # Validate the latest metadata.
            md = collected[-1]
            assert isinstance(md, ObjectsMetadata)
            assert md.model_id == rtdetr_manifest.name
            assert md.sensor_id == "cam0"

            # Exactly 2 detections from the 4-class fixture (q0=car, q1=person; q2 filtered).
            labels = {d.label for d in md.detections}
            assert len(md.detections) == 2, (
                f"Expected 2 detections; got {len(md.detections)}: {labels}"
            )
            assert labels == {"car", "person"}, f"Unexpected labels: {labels}"

        finally:
            runner.stop()

            # After stop the branch bin is torn down; the appsink should be gone.
            assert pipeline.get_by_name("cam0_det_sink") is None, (
                "appsink still present after runner.stop() — branch was not torn down"
            )

            # The main pipeline must still be PLAYING.
            ret2, state2, _pending2 = pipeline.get_state(Gst.SECOND)
            assert ret2 == Gst.StateChangeReturn.SUCCESS
            assert state2 == Gst.State.PLAYING, (
                f"Pipeline NOT in PLAYING after stop(): state={state2}"
            )

            pipeline.set_state(Gst.State.NULL)


# ── Non-live unit test: kind guard ────────────────────────────────────────────


class TestDetectorRunnerKindGuard:
    """DetectorRunner rejects manifests whose ``kind`` is not ``'detector'``."""

    def test_kind_vla_raises_ros_config_error(
        self,
        onnx_path: pathlib.Path,
    ) -> None:
        """``kind: vla`` manifest raises ``ROSConfigError`` from ``DetectorRunner``.

        Guard fires in ``__init__`` before any GStreamer call, so a minimal pipeline
        (``videotestsrc ! fakesink``) suffices — ``TeeManager`` is never reached.
        """
        vla_fixture = _REPO_ROOT / "rskills" / "pi05-libero-int8" / "rskill.yaml"
        assert vla_fixture.exists(), f"vla fixture not found: {vla_fixture}"
        with open(vla_fixture, encoding="utf-8") as fh:
            vla_data = yaml.safe_load(fh)
        vla_manifest = RSkillManifest.model_validate(vla_data)
        assert vla_manifest.kind == "vla"

        # Minimal pipeline: no tee required because the kind guard fires first.
        pipeline = Gst.parse_launch("videotestsrc ! fakesink")
        assert pipeline is not None

        with pytest.raises(ROSConfigError, match="kind='vla'"):
            DetectorRunner(
                pipeline,
                vla_manifest,
                onnx_path=onnx_path,
                sensor_id="cam0",
                on_detection=lambda md: None,
            )

        pipeline.set_state(Gst.State.NULL)


# ── Non-live unit test: tier default ─────────────────────────────────────────


class TestDetectorRunnerTierDefault:
    """DetectorRunner resolves to CPU_ONNX on a host without DeepStream/Tegra."""

    def test_tier_default_is_cpu_onnx(
        self,
        onnx_path: pathlib.Path,
        rtdetr_manifest: RSkillManifest,
    ) -> None:
        """No explicit tier → resolves to CPU_ONNX (no DeepStream/Tegra on this host).

        ``select_detector_tier()`` sets it in ``DetectorRunner.__init__``; no
        ``start()`` call needed to verify ``_tier``.
        """
        spec = PipelineSpec(
            source=Source.TESTSRC,
            fps=30,
            width=640,
            height=480,
            enable_ros_tee=True,
            enable_nvmm=False,
        )
        pipeline_str = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
        pipeline = Gst.parse_launch(pipeline_str)
        assert pipeline is not None

        runner = DetectorRunner(
            pipeline,
            rtdetr_manifest,
            onnx_path=onnx_path,
            sensor_id="head_rgb",
            on_detection=lambda md: None,
        )
        assert runner._tier is DetectorTier.CPU_ONNX, (
            f"Expected CPU_ONNX on this host; got {runner._tier!r}"
        )

        pipeline.set_state(Gst.State.NULL)


# ── Non-live unit test: non-square input_size handoff ────────────────────────


class TestDetectorRunnerInputSizeHandoff:
    """DetectorRunner converts ``input_size`` (width, height) → detector (height, width)."""

    def test_nonsquare_input_size_passed_as_height_width(
        self,
        onnx_path: pathlib.Path,
    ) -> None:
        """``DetectorContract.input_size`` (width,height) must reach the detector as
        (height,width) — a latent transpose masked by the square 640×640 fixture.

        ``(640,480)`` in → ``(480,640)`` out. ``ObjectsDetector.__init__`` only stores
        ``input_size`` (no ONNX validation), so the existing fixture works unmodified.
        """
        fixture = _REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"
        assert fixture.exists(), f"Fixture not found: {fixture}"
        data = yaml.safe_load(fixture.read_text(encoding="utf-8"))
        data["detector"]["input_size"] = [640, 480]  # (width, height), non-square
        manifest = RSkillManifest.model_validate(data)
        assert manifest.detector is not None
        assert manifest.detector.input_size == (640, 480)

        spec = PipelineSpec(
            source=Source.TESTSRC,
            fps=30,
            width=640,
            height=480,
            enable_ros_tee=True,
            enable_nvmm=False,
        )
        pipeline_str = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
        pipeline = Gst.parse_launch(pipeline_str)
        assert pipeline is not None

        runner = DetectorRunner(
            pipeline,
            manifest,
            onnx_path=onnx_path,
            sensor_id="head_rgb",
            on_detection=lambda md: None,
        )
        # The detector must receive (height, width) = (480, 640).
        assert runner._detector._input_size == (480, 640), (
            f"Expected detector input_size (height, width)=(480, 640); "
            f"got {runner._detector._input_size!r}"
        )
        # And the runner caches explicit width/height for the NVMM caps.
        assert (runner._net_w, runner._net_h) == (640, 480)

        pipeline.set_state(Gst.State.NULL)
