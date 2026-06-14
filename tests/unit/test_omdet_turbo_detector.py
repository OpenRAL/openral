"""Tests for the in-process OmDet-Turbo zero-shot detector backend (ADR-0037).

Two tiers:

* **Pure conversion + manifest/dispatch** (always run, no GPU/torch):
  ``build_objects_metadata_from_results`` and the ``engine: zeroshot_hf``
  manifest → :class:`OmDetTurboDetector` dispatch. Construction is lazy (the
  model loads on first ``detect()``), so the dispatch path builds the backend
  without torch, transformers, or a GPU.
* **Real end-to-end** (gated): load the real ``omlab/omdet-turbo-swin-tiny-hf``
  Apache-2.0 weights and detect indoor classes on the real ``coco_sample.jpg``
  fixture. Skipped unless a GPU is present — the legitimate CI skip path
  (CLAUDE.md §12).
"""

from __future__ import annotations

import pathlib
import shutil

import pytest
from openral_core import ObjectsMetadata
from openral_core.exceptions import ROSConfigError
from openral_runner.backends.gstreamer.omdet_turbo_detector import (
    OmDetTurboDetector,
    build_objects_metadata_from_results,
    query_to_classes,
)

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "rskills" / "omdet-turbo-indoor" / "rskill.yaml"
_RTDETR_MANIFEST = _REPO / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"


# --------------------------------------------------------------------------
# Pure conversion (no torch / transformers / GPU).
# --------------------------------------------------------------------------


def test_build_keeps_above_threshold_drops_below() -> None:
    md = build_objects_metadata_from_results(
        labels=["mug", "drawer"],
        scores=[0.9, 0.2],
        boxes_xyxy=[(10.0, 10.0, 50.0, 60.0), (0.0, 0.0, 100.0, 100.0)],
        width=640,
        height=480,
        model_id="omdet-turbo-indoor",
        sensor_id="front_cam",
        score_threshold=0.3,
    )
    assert isinstance(md, ObjectsMetadata)
    assert [d.label for d in md.detections] == ["mug"]
    assert md.detections[0].bbox_xyxy == (10, 10, 50, 60)
    assert md.detections[0].confidence == pytest.approx(0.9)


def test_build_sorts_by_descending_confidence() -> None:
    md = build_objects_metadata_from_results(
        labels=["cup", "bowl", "plate"],
        scores=[0.4, 0.95, 0.7],
        boxes_xyxy=[(0, 0, 10, 10), (20, 20, 40, 40), (50, 50, 70, 70)],
        width=100,
        height=100,
        model_id="m",
        sensor_id="s",
        score_threshold=0.3,
    )
    assert md is not None
    assert [d.label for d in md.detections] == ["bowl", "plate", "cup"]


def test_build_clips_boxes_to_frame_and_orders_corners() -> None:
    # x2<x1 corner order + out-of-frame coords get corner-ordered and clipped.
    md = build_objects_metadata_from_results(
        labels=["kettle"],
        scores=[0.8],
        boxes_xyxy=[(700.0, -5.0, 50.0, 500.0)],
        width=640,
        height=480,
        model_id="m",
        sensor_id="s",
        score_threshold=0.3,
    )
    assert md is not None
    assert md.detections[0].bbox_xyxy == (50, 0, 640, 480)


def test_build_drops_near_full_image_box() -> None:
    # Box covering ~full frame (>= 98%) is a background mis-fire; dropped.
    assert (
        build_objects_metadata_from_results(
            labels=["wall"],
            scores=[0.9],
            boxes_xyxy=[(0.0, 0.0, 100.0, 100.0)],
            width=100,
            height=100,
            model_id="m",
            sensor_id="s",
            score_threshold=0.3,
        )
        is None
    )


def test_build_none_when_all_below_threshold() -> None:
    assert (
        build_objects_metadata_from_results(
            labels=["mug"],
            scores=[0.1],
            boxes_xyxy=[(10, 10, 50, 50)],
            width=640,
            height=480,
            model_id="m",
            sensor_id="s",
            score_threshold=0.3,
        )
        is None
    )


def test_build_rejects_length_mismatch() -> None:
    with pytest.raises(ROSConfigError, match="length"):
        build_objects_metadata_from_results(
            labels=["a", "b"],
            scores=[0.9],
            boxes_xyxy=[(0, 0, 10, 10)],
            width=64,
            height=64,
            model_id="m",
            sensor_id="s",
            score_threshold=0.3,
        )


# --------------------------------------------------------------------------
# On-demand query path (ADR-0051): pure parsing + lazy set_query (no GPU).
# --------------------------------------------------------------------------


def test_query_to_classes_splits_multi_and_keeps_single_phrase() -> None:
    assert query_to_classes("stapler, scissors") == ["stapler", "scissors"]
    assert query_to_classes("red mug</c>blue bowl") == ["red mug", "blue bowl"]
    assert query_to_classes("the red mug") == ["the red mug"]
    # whitespace-only fragments dropped
    assert query_to_classes("cup, , bowl") == ["cup", "bowl"]


def test_query_to_classes_rejects_empty() -> None:
    with pytest.raises(ROSConfigError, match="non-empty"):
        query_to_classes("  ,  ")


def test_set_query_retargets_vocabulary_lazily() -> None:
    # Construction + set_query are side-effect-free (no model load): the on-demand
    # retarget hook updates the persistent class list without a GPU.
    det = OmDetTurboDetector(
        labels=["cup"], model_id="m", weights_source="omlab/omdet-turbo-swin-tiny-hf"
    )
    det.set_query("stapler, scissors")
    assert det._labels == ["stapler", "scissors"]
    with pytest.raises(ROSConfigError, match="non-empty"):
        det.set_query("   ")


def test_on_demand_locator_backend_exposes_query_methods() -> None:
    # The detector node wires locate_in_view / detector_query by hasattr; the
    # backend must expose both for the on-demand role to function.
    det = OmDetTurboDetector(
        labels=["cup"], model_id="m", weights_source="omlab/omdet-turbo-swin-tiny-hf"
    )
    assert hasattr(det, "set_query") and hasattr(det, "detect_with_query")


# --------------------------------------------------------------------------
# Manifest validation + dispatch (no GPU/torch).
# --------------------------------------------------------------------------


def test_omdet_manifest_validates_as_zeroshot_detector() -> None:
    from openral_core.schemas import DetectorEngine, RSkillManifest, RSkillRuntime

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    assert m.kind == "detector"
    assert m.runtime is RSkillRuntime.PYTORCH
    assert m.detector is not None
    assert m.detector.engine is DetectorEngine.ZEROSHOT_HF
    # Curated indoor vocabulary: far more than the 80 COCO classes.
    assert len(m.detector.labels) > 200
    assert "mug" in m.detector.labels and "drawer" in m.detector.labels
    assert m.actuators_required == []


def test_build_manifest_detector_dispatches_zeroshot_in_process() -> None:
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.gstreamer.detector_factory import (
        build_manifest_detector,
        weights_source_from_manifest,
    )
    from openral_runner.backends.gstreamer.objects_detector import DetectorTier
    from openral_runner.backends.gstreamer.omdet_turbo_detector import OmDetTurboDetector

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    # No onnx_path: the zero-shot backend needs none, and construction is lazy
    # so this builds without torch/transformers or a model load.
    det, tier = build_manifest_detector(m)
    try:
        assert tier is DetectorTier.ZEROSHOT_HF
        assert isinstance(det, OmDetTurboDetector)
        assert det._labels == m.detector.labels  # fixed vocabulary
        assert weights_source_from_manifest(m) == "omlab/omdet-turbo-swin-tiny-hf"
        # ADR-0051: the "unprompted" intent is declared by the manifest mode, not
        # by the backend lacking the capability (one backend serves both modes).
        from openral_core.schemas import DetectorMode

        assert m.detector.mode is DetectorMode.CONTINUOUS
    finally:
        det.close()


def test_omdet_group_declares_torch_and_transformers() -> None:
    """The in-process backend's torch + transformers deps ship in a real group.

    Regression guard: ``OmDetTurboDetector`` lazily imports ``torch`` +
    ``transformers``; those must ship in the ``omdet`` optional-dependency group
    so ``uv sync --group omdet`` (and the ``deploy sim`` omdet leg) resolve,
    rather than failing on first ``detect()`` with a bare ``No module named
    'torch'``.
    """
    import tomllib

    with (_REPO / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    groups = pyproject["dependency-groups"]
    assert "omdet" in groups, "missing `omdet` dependency group"
    group = " ".join(groups["omdet"])
    assert "transformers" in group and "torch" in group


def test_on_demand_locator_manifest_dispatches_to_omdet_backend() -> None:
    from openral_core.schemas import DetectorMode, RSkillManifest
    from openral_runner.backends.gstreamer.detector_factory import build_manifest_detector
    from openral_runner.backends.gstreamer.objects_detector import DetectorTier

    m = RSkillManifest.from_yaml(str(_REPO / "rskills" / "omdet-turbo-locator" / "rskill.yaml"))
    assert m.detector is not None and m.detector.mode is DetectorMode.ON_DEMAND
    det, tier = build_manifest_detector(m)  # same backend, regardless of mode
    try:
        assert tier is DetectorTier.ZEROSHOT_HF
        assert isinstance(det, OmDetTurboDetector)
        assert det._labels == m.detector.labels  # static default query set
        # On-demand role: the query methods the detector node binds by hasattr.
        assert hasattr(det, "set_query") and hasattr(det, "detect_with_query")
    finally:
        det.close()


def test_zeroshot_engine_does_not_disturb_onnx_dispatch() -> None:
    # Regression guard: an onnx manifest (engine unset) still needs an onnx_path
    # — the new engine branch must not shadow the legacy runtime dispatch.
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.gstreamer.detector_factory import build_manifest_detector

    m = RSkillManifest.from_yaml(str(_RTDETR_MANIFEST))
    assert m.detector is not None and m.detector.engine is None
    with pytest.raises(ROSConfigError, match="onnx_path"):
        build_manifest_detector(m, onnx_path=None)


# --------------------------------------------------------------------------
# Real end-to-end with the Apache-2.0 weights (GPU-gated).
# --------------------------------------------------------------------------


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(
    not _gpu_present(),
    reason="needs a local GPU to load omlab/omdet-turbo-swin-tiny-hf "
    "(the legitimate CI skip path, CLAUDE.md §12).",
)
def test_e2e_detects_indoor_objects_on_coco_sample() -> None:
    # coco_sample.jpg is the canonical COCO image of two cats and two remotes on
    # a couch — a real indoor scene. The fixed vocabulary includes
    # "remote control", "sofa", "blanket" etc., so the detector should fire on
    # indoor classes without any prompting.
    import numpy as np
    from openral_runner.backends.gstreamer.omdet_turbo_detector import OmDetTurboDetector
    from PIL import Image

    img = Image.open(_FIXTURES / "coco_sample.jpg").convert("RGB")
    w, h = img.size
    bgr = np.asarray(img)[:, :, ::-1].tobytes()  # RGB -> BGR bytes

    det = OmDetTurboDetector(
        labels=["remote control", "sofa", "cat", "blanket", "cup"],
        model_id="OpenRAL/rskill-omdet-turbo-indoor",
        weights_source="omlab/omdet-turbo-swin-tiny-hf",
        score_threshold=0.3,
    )
    try:
        md = det.detect(bgr, w, h, sensor_id="coco_cam")
    finally:
        det.close()

    assert md is not None, "expected indoor detections on coco_sample.jpg"
    assert md.model_id == "OpenRAL/rskill-omdet-turbo-indoor"
    for d in md.detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert 0.0 < d.confidence <= 1.0
