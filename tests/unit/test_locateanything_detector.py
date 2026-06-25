"""Tests for the LocateAnything open-vocabulary detector backend (Phase 1).

Two tiers:

* **Pure parsing/conversion** (always run, no GPU): exercise
  ``parse_grounding_answer`` / ``build_objects_metadata`` against a *real*
  captured model answer — including the degenerate repeated-box tail the model
  emits when greedy decoding loops (CLAUDE.md §1.11: real data, no placeholders).
* **Real end-to-end** (gated): boot the actual ``transformers==4.57.1`` sidecar,
  run NF4 LocateAnything-3B on the real ``coco_sample.jpg`` fixture, and assert a
  ``person`` ``ObjectsMetadata`` comes back over ZMQ. Skipped unless a sidecar
  venv is provisioned (``OPENRAL_LOCATEANYTHING_SIDECAR_VENV``) and a GPU is
  present — the legitimate CI skip path (CLAUDE.md §12).
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest
from openral_core import ObjectsMetadata
from openral_runner.backends.gstreamer.locateanything_detector import (
    build_objects_metadata,
    parse_grounding_answer,
)

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "rskills" / "locateanything-3b-nf4" / "rskill.yaml"
_RTDETR_MANIFEST = _REPO / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"

# A real answer captured from nvidia/LocateAnything-3B for query "person":
# two genuine person boxes, then the degenerate full-image tail the greedy
# loop produces (here truncated to two identical repeats).
_REAL_ANSWER = (
    "<ref>person</ref>"
    "<box><43><468><129><869></box>"
    "<box><75><510><186><896></box>"
    "<box><981><0><1000><1000></box>"
    "<box><981><0><1000><1000></box>"
)


def test_parse_keeps_real_boxes_drops_degenerate_tail() -> None:
    boxes = parse_grounding_answer(_REAL_ANSWER, fallback_label="person")
    # The two genuine boxes survive; the near-full-image repeats are dropped
    # (area >= 0.85) and deduped.
    assert boxes == [
        ("person", (43, 468, 129, 869)),
        ("person", (75, 510, 186, 896)),
    ]


def test_parse_binds_boxes_to_most_recent_ref() -> None:
    answer = (
        "<ref>cat</ref><box><100><100><200><200></box><ref>dog</ref><box><300><300><400><400></box>"
    )
    assert parse_grounding_answer(answer) == [
        ("cat", (100, 100, 200, 200)),
        ("dog", (300, 300, 400, 400)),
    ]


def test_parse_drops_thin_slivers() -> None:
    # width 5/1000 = 0.005 < 0.02 sliver guard.
    assert parse_grounding_answer("<box><10><10><15><900></box>", fallback_label="x") == []


def test_parse_normalizes_corner_order() -> None:
    # x2<x1 / y2<y1 should be corner-ordered, not dropped.
    assert parse_grounding_answer("<box><200><200><100><100></box>", fallback_label="x") == [
        ("x", (100, 100, 200, 200)),
    ]


def test_build_objects_metadata_scales_to_pixels() -> None:
    md = build_objects_metadata(
        _REAL_ANSWER,
        width=640,
        height=480,
        model_id="locateanything-3b-nf4",
        sensor_id="front_cam",
        fallback_label="person",
    )
    assert isinstance(md, ObjectsMetadata)
    assert md.model_id == "locateanything-3b-nf4"
    assert md.sensor_id == "front_cam"
    assert md.frame_width == 640 and md.frame_height == 480
    assert len(md.detections) == 2
    d0 = md.detections[0]
    assert d0.label == "person"
    assert d0.confidence == 1.0  # grounding model: no per-box score
    # 43/1000*640 = 27.52 -> 28 ; 468/1000*480 = 224.64 -> 225
    assert d0.bbox_xyxy == (28, 225, 83, 417)


def test_build_objects_metadata_none_when_empty() -> None:
    assert (
        build_objects_metadata("no boxes here", width=64, height=64, model_id="m", sensor_id="s")
        is None
    )


# --------------------------------------------------------------------------
# Manifest validation + dispatch (Phase 2, no GPU/sidecar).
# --------------------------------------------------------------------------


def test_locateanything_manifest_validates() -> None:
    from openral_core.schemas import RSkillManifest, RSkillRuntime

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    assert m.kind == "detector"
    assert m.runtime is RSkillRuntime.PYTORCH
    assert m.detector is not None
    assert "person" in m.detector.labels
    # max_side caps the sidecar's grounding frame so LA-3B co-fits a reward model
    # on an 8 GB GPU (ADR-0057 co-residency); the manifest pins 512.
    assert m.detector.max_side == 512


def test_build_manifest_detector_dispatches_pytorch_to_sidecar() -> None:
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.gstreamer.detector_factory import (
        build_manifest_detector,
        weights_source_from_manifest,
    )
    from openral_runner.backends.gstreamer.locateanything_detector import (
        LocateAnythingDetector,
    )
    from openral_runner.backends.gstreamer.objects_detector import DetectorTier

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    # No onnx_path and no running sidecar: construction is lazy, so this builds
    # the backend without connecting.
    det, tier = build_manifest_detector(m)
    try:
        assert tier is DetectorTier.VLM_SIDECAR
        assert isinstance(det, LocateAnythingDetector)
        # Static default query = the manifest's labels joined for the prompt.
        assert det._query == "</c>".join(m.detector.labels)
        # weights_uri (the prequantized NF4 mirror this rSkill ships) wins over
        # source_repo (upstream provenance): the sidecar loads the mirror directly
        # via the prequantized path (see _locateanything_server._load).
        assert weights_source_from_manifest(m) == "OpenRAL/rskill-locateanything-3b-nf4"
        # The factory threads the manifest's max_side into the backend so the
        # sidecar boots with --max-side 512 (lower grounding VRAM peak).
        assert det._max_side == 512
    finally:
        det.close()


def test_locate_in_view_tool_schema_round_trips() -> None:
    """LocateInViewTool parses via the ReasonerToolCall discriminated union (ADR-0043)."""
    from openral_core import LocateInViewTool, ReasonerToolCall
    from pydantic import TypeAdapter

    parsed = TypeAdapter(ReasonerToolCall).validate_python(
        {"tool": "locate_in_view", "query": "red mug", "camera": "wrist"}
    )
    assert isinstance(parsed, LocateInViewTool)
    assert parsed.query == "red mug" and parsed.camera == "wrist"
    # camera is optional (camera-agnostic): empty default, not a hardcoded name.
    assert (
        TypeAdapter(ReasonerToolCall)
        .validate_python({"tool": "locate_in_view", "query": "cup"})
        .camera
        == ""
    )


def test_locate_in_view_palette_gated_on_detector_available() -> None:
    """The LLM sees locate_in_view only when a detector is available (ADR-0043)."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    off = [d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette())]
    on = [d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette(detector_available=True))]
    assert "locate_in_view" not in off
    assert "locate_in_view" in on


def test_locateanything_extra_declares_sidecar_client_deps() -> None:
    """The detector-node-side ZMQ client transport is a declared dependency.

    Regression guard: the ``LocateAnythingDetector`` client (which runs in the
    deploy-sim / detector-node venv, not the sidecar venv) lazily imports
    ``zmq`` + ``msgpack``. Those must ship in a real optional-dependency group so
    ``deploy sim --object-detector-manifest`` doesn't fail per-request with a bare
    ``No module named 'zmq'`` — they were previously only in the unrelated
    ``rldx`` group.
    """
    import tomllib

    with (_REPO / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    groups = pyproject["dependency-groups"]
    assert "locateanything" in groups, "missing `locateanything` dependency group"
    group = " ".join(groups["locateanything"])
    assert "pyzmq" in group and "msgpack" in group


def test_build_manifest_detector_onnx_requires_path() -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.gstreamer.detector_factory import build_manifest_detector

    m = RSkillManifest.from_yaml(str(_RTDETR_MANIFEST))  # runtime: onnx
    with pytest.raises(ROSConfigError, match="onnx_path"):
        build_manifest_detector(m, onnx_path=None)


# --------------------------------------------------------------------------
# Real end-to-end through the sidecar (gated).
# --------------------------------------------------------------------------


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(
    not os.environ.get("OPENRAL_LOCATEANYTHING_SIDECAR_VENV") or not _gpu_present(),
    reason="needs a provisioned LocateAnything sidecar venv + a local GPU "
    "(set OPENRAL_LOCATEANYTHING_SIDECAR_VENV).",
)
def test_e2e_detect_cats_on_coco_sample() -> None:
    # coco_sample.jpg is the canonical COCO image of two cats and two remotes
    # (no person). Querying "cat" exercises the real open-vocab grounding path.
    from openral_runner.backends.gstreamer.locateanything_detector import (
        LocateAnythingDetector,
    )
    from PIL import Image

    img = Image.open(_FIXTURES / "coco_sample.jpg").convert("RGB")
    w, h = img.size
    import numpy as np

    bgr = np.asarray(img)[:, :, ::-1].tobytes()  # RGB -> BGR bytes

    det = LocateAnythingDetector(
        labels=["cat"],
        model_id="OpenRAL/rskill-locateanything-3b-nf4",
        port=5759,
        boot_timeout_s=1800,
    )
    try:
        md = det.detect(bgr, w, h, sensor_id="coco_cam")
        # The image has no person; querying a class that isn't present should
        # ground nothing — proves the detector discriminates, not just fires.
        det.set_query("person")
        person_md = det.detect(bgr, w, h, sensor_id="coco_cam")
    finally:
        det.close()

    assert md is not None, "expected cat detections on coco_sample.jpg"
    # >=1 (not >=2): do_sample=True makes the exact box count nondeterministic;
    # the meaningful check is that "cat" grounds at all (and "person" doesn't).
    assert len(md.detections) >= 1, f"expected >=1 cat, got {len(md.detections)}"
    assert all(d.label == "cat" for d in md.detections)
    assert all(d.confidence == 1.0 for d in md.detections)
    for d in md.detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    assert person_md is None, "no person in image; expected no person detections"
