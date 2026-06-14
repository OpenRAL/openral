"""End-to-end tests for :class:`DetectorRunner` (ADR-0037 integration).

These are **live** tests — no mocks, no stubs, per CLAUDE.md §1.11.

The e2e test:
- Builds a deterministic 4-class ONNX (copied from ``test_objects_detector.py``).
- Loads the **real** ``rskills/rtdetr-coco-r18/rskill.yaml`` manifest.
- Constructs a live ``videotestsrc`` GStreamer pipeline with the named bus tee.
- Creates a :class:`DetectorRunner` and calls ``start()``.
- Pumps until the ``new-sample`` callback fires and a detection lands in
  ``collected``.
- Asserts that the 4-class ONNX emits exactly the expected 2 objects (car +
  person), the model_id matches the manifest name, and the tee branch is live.
- Calls ``stop()`` and asserts the branch is torn down and the pipeline is still
  PLAYING.

The non-live unit test verifies that passing a ``kind: vla`` manifest raises
:exc:`~openral_core.exceptions.ROSConfigError` at construction time — the kind
guard fires before any GStreamer call, so the test can use a minimal pipeline
with no tee.

Gates:
    ``gi``, ``onnxruntime``, ``onnx`` — skipped if any is absent.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
import yaml

gi = pytest.importorskip("gi")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnx  # noqa: E402
import onnx.helper as h  # noqa: E402
import onnx.numpy_helper as nph  # noqa: E402

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

Gst.init(None)

# ── Repo-root path helper ──────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


# ── Deterministic ONNX fixture (mirrors test_objects_detector._write_rtdetr_like_onnx) ──

# 4-class labels that match the ONNX fixture constants below.
# COCO indices 0=person, 2=car — so the 4-class ["person","bicycle","car","dog"]
# slice aligns with COCO: q0 cls-2 → "car", q1 cls-0 → "person".
_LABELS_4 = ["person", "bicycle", "car", "dog"]


def _write_rtdetr_like_onnx(path: pathlib.Path) -> None:
    """Write a deterministic 4-class RT-DETR-like ONNX to *path*.

    Identical to the fixture in :mod:`tests.unit.test_objects_detector`:
    - q0 logits ``[-5, -5, 3.0, -5]``  → car (idx 2),   sigmoid(3)  ≈ 0.953
    - q1 logits ``[2.0, -5, -5, -5]``  → person (idx 0), sigmoid(2) ≈ 0.881
    - q2 logits ``[-5, -5, -5, -5]``   → max ≈ 0.007 (below 0.5 threshold)
    """
    logits_data = np.array(
        [[[-5.0, -5.0, 3.0, -5.0], [2.0, -5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]],
        dtype=np.float32,
    )
    boxes_data = np.array(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]],
        dtype=np.float32,
    )

    logits_tensor = nph.from_array(logits_data, name="logits_const")
    boxes_tensor = nph.from_array(boxes_data, name="boxes_const")

    images_input = h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])
    logits_out = h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 3, 4])
    boxes_out = h.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 3, 4])
    passthrough_out = h.make_tensor_value_info(
        "images_passthrough", onnx.TensorProto.FLOAT, [1, 3, 640, 640]
    )

    id_node = h.make_node("Identity", inputs=["images"], outputs=["images_passthrough"])
    logits_node = h.make_node("Constant", inputs=[], outputs=["logits"], value=logits_tensor)
    boxes_node = h.make_node("Constant", inputs=[], outputs=["boxes"], value=boxes_tensor)

    graph = h.make_graph(
        nodes=[id_node, logits_node, boxes_node],
        name="rtdetr_test",
        inputs=[images_input],
        outputs=[logits_out, boxes_out, passthrough_out],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


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

        The 4-class ONNX fixture emits ``car`` (confidence ≈ 0.953) and ``person``
        (confidence ≈ 0.881) on every frame, above the manifest's 0.5 threshold.
        COCO index 0 = ``person``, index 2 = ``car`` — confirmed by the rtdetr-coco-r18
        label list.

        The ``rtdetr-coco-r18`` manifest uses ``score_threshold: 0.7`` and 80 COCO labels,
        but the 4-class ONNX only activates classes 0 and 2.  The ``DetectorRunner``
        passes ``manifest.detector.labels`` to ``ObjectsDetector``, so the label names
        are ``"person"`` (idx 0) and ``"car"`` (idx 2) from the COCO 80-class list.
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
        """Passing a ``kind: vla`` manifest to ``DetectorRunner`` raises ``ROSConfigError``.

        The kind guard fires in ``__init__`` before any GStreamer call, so we
        can use a minimal pipeline (``videotestsrc ! fakesink``) with no tee —
        the ``TeeManager`` constructor is never reached.
        """
        vla_fixture = _REPO_ROOT / "rskills" / "pi05-libero" / "rskill.yaml"
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
        """Constructing DetectorRunner without an explicit tier resolves to CPU_ONNX.

        On this host (no DeepStream, no Tegra) ``select_detector_tier()`` returns
        :attr:`~openral_runner.backends.gstreamer.objects_detector.DetectorTier.CPU_ONNX`.
        No call to ``start()`` is made — construction is sufficient to verify ``_tier``.
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
        """``DetectorContract.input_size`` is (width, height); detector gets (height, width).

        Regression for a latent transpose masked by the square 640×640 fixture:
        a non-square ``(640, 480)`` (= width 640, height 480) must reach
        ``ObjectsDetector`` as ``(480, 640)`` = (height, width).
        ``ObjectsDetector.__init__`` only stores ``input_size`` (no ONNX validation),
        so the existing ``onnx_path`` fixture works without a non-square model.
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
