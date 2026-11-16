"""Tests for the Open Vision Banana (OVB) on-demand locator backend.

Two tiers:

* **Pure prompt/decode/dispatch** (always run, no GPU): the referring-segmentation
  prompt, the paint-colour decode and the connected-region split, driven by a
  real SAM 2.1 mask fixture painted the way OVB paints its answer; plus the real
  ``rskills/open-vision-banana-4b`` manifest through the real factory.
* **Real end-to-end** (gated): boot the actual sidecar, run OVB 4B on the real
  ``coco_sample.jpg`` fixture and assert a ``person`` box comes back over ZMQ.
  Skipped unless a sidecar venv is provisioned (``OPENRAL_OVB_SIDECAR_VENV``) and
  a GPU is present (CLAUDE.md §1.12).
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import shutil
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from openral_core import ObjectsMetadata
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    DetectorContract,
    DetectorEngine,
    DetectorMode,
    OvbDetectorConfig,
    RSkillLicensePosture,
    RSkillManifest,
)
from openral_runner.backends.gstreamer.ovb_detector import (
    REFSEG_PAINT_RGB,
    build_objects_metadata_from_paint,
    build_refseg_prompt,
    decode_painted_mask,
    mask_regions_xyxy,
)
from PIL import Image

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_MANIFEST = _REPO / "rskills" / "open-vision-banana-4b" / "rskill.yaml"
_MASK = _FIXTURES / "sam2_front_tablecloth_mask.png"


def _paint(
    mask: npt.NDArray[np.bool_], *, edge_rgb: tuple[int, int, int] | None = None
) -> npt.NDArray[np.uint8]:
    """Paint ``mask`` the way OVB answers: paint colour on black.

    ``edge_rgb`` recolours the mask's boundary pixels, standing in for the
    anti-aliased edge a diffusion decoder produces.
    """
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[mask] = REFSEG_PAINT_RGB
    if edge_rgb is not None:
        inner = mask.copy()
        inner[1:, :] &= mask[:-1, :]
        inner[:-1, :] &= mask[1:, :]
        inner[:, 1:] &= mask[:, :-1]
        inner[:, :-1] &= mask[:, 1:]
        rgb[mask & ~inner] = edge_rgb
    return rgb


def test_prompt_matches_upstream_quick_start() -> None:
    # Verbatim the OpenVisionBanana PROMPTS.md referring-segmentation example,
    # with its "table in the room" description.
    assert build_refseg_prompt("table in the room") == (
        "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this "
        'description: "table in the room". Paint the matching region red [220, 30, 30] and '
        "leave everything else black [0, 0, 0]. Output only the segmentation mask image."
    )


@pytest.mark.parametrize("expr", ["", "   ", 'the "red" mug'])
def test_prompt_refuses_empty_or_quoted_expression(expr: str) -> None:
    with pytest.raises(ROSConfigError):
        build_refseg_prompt(expr)


def test_decode_recovers_a_real_mask_including_soft_edges() -> None:
    mask = np.asarray(Image.open(_MASK)).astype(bool)
    # Edge pixels a bit nearer the paint colour than black stay foreground.
    painted = _paint(mask, edge_rgb=(130, 20, 20))
    assert np.array_equal(decode_painted_mask(painted), mask)
    # Edge pixels nearer black drop out — the midpoint rule, not a hue match.
    dark_edges = decode_painted_mask(_paint(mask, edge_rgb=(90, 10, 10)))
    assert dark_edges.sum() < mask.sum() and not (dark_edges & ~mask).any()


def test_regions_box_each_connected_component() -> None:
    mask = np.zeros((40, 60), bool)
    mask[2:10, 3:12] = True  # region A
    mask[20:35, 40:58] = True  # region B
    mask[0, 59] = True  # one-pixel speckle, dropped by the area floor
    assert mask_regions_xyxy(mask) == [(3, 2, 12, 10), (40, 20, 58, 35)]
    # Diagonal neighbours are not 4-connected: two regions, not one.
    diag = np.zeros((4, 4), bool)
    diag[0, 0] = diag[1, 1] = True
    assert len(mask_regions_xyxy(diag, min_region_frac=0.0)) == 2


def test_metadata_from_real_mask_boxes_the_mask_extent() -> None:
    mask = np.asarray(Image.open(_MASK)).astype(bool)
    meta = build_objects_metadata_from_paint(
        _paint(mask), label="tablecloth", model_id="ovb-test", sensor_id="front"
    )
    assert isinstance(meta, ObjectsMetadata)
    assert (meta.frame_width, meta.frame_height) == (mask.shape[1], mask.shape[0])
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = meta.detections[0].bbox_xyxy
    # The largest region's box lies inside the mask's overall extent.
    assert xs.min() <= x1 < x2 <= xs.max() + 1 and ys.min() <= y1 < y2 <= ys.max() + 1
    assert {d.label for d in meta.detections} == {"tablecloth"}
    assert all(d.confidence == 1.0 for d in meta.detections)


def test_metadata_is_none_when_nothing_is_painted() -> None:
    black = np.zeros((48, 64, 3), np.uint8)
    assert build_objects_metadata_from_paint(black, label="x", model_id="m", sensor_id="s") is None


def test_detector_contract_needs_ovb_block_iff_ovb_engine() -> None:
    cfg = OvbDetectorConfig(
        base_pipeline_uri="hf://black-forest-labs/FLUX.2-klein-base-4B@"
        "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        transformer_subfolder="final_models/4B/zeroshot/checkpoint-87500-95000-w25/transformer",
    )
    DetectorContract(labels=["object"], engine=DetectorEngine.OVB_SIDECAR, ovb=cfg)
    with pytest.raises(ValueError, match="needs an `ovb` block"):
        DetectorContract(labels=["object"], engine=DetectorEngine.OVB_SIDECAR)
    with pytest.raises(ValueError, match="only valid with engine 'ovb_sidecar'"):
        DetectorContract(labels=["object"], engine=DetectorEngine.ZEROSHOT_HF, ovb=cfg)
    # The base pipeline must be pinned to a full commit sha (CLAUDE.md §1.8).
    with pytest.raises(ValueError):
        OvbDetectorConfig(
            base_pipeline_uri="hf://black-forest-labs/FLUX.2-klein-base-4B@main",
            transformer_subfolder="transformer",
        )


def test_ovb_manifest_is_on_demand_and_non_commercial() -> None:
    m = RSkillManifest.from_yaml(str(_MANIFEST))
    assert m.detector is not None and m.detector.ovb is not None
    assert m.detector.engine is DetectorEngine.OVB_SIDECAR
    assert m.detector.mode is DetectorMode.ON_DEMAND
    assert m.license is RSkillLicensePosture.FLUX_NON_COMMERCIAL
    assert not m.is_commercial_use_allowed
    assert m.quantization is not None and m.quantization.dtype == "bf16"


def test_build_manifest_detector_dispatches_to_ovb_sidecar() -> None:
    from openral_runner.backends.gstreamer.detector_factory import build_manifest_detector
    from openral_runner.backends.gstreamer.objects_detector import DetectorTier
    from openral_runner.backends.gstreamer.ovb_detector import OpenVisionBananaDetector

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    det, tier = build_manifest_detector(m)  # lazy: no sidecar, no GPU
    try:
        assert tier is DetectorTier.OVB_SIDECAR
        assert isinstance(det, OpenVisionBananaDetector)
        assert det._weights_source == (
            "FunAILab/Open-Vision-Banana@be5da66250cc385ff9c75ebbe4114e13ec305848"
        )
        assert det._base_pipeline == (
            "black-forest-labs/FLUX.2-klein-base-4B@a3b4f4849157f664bdbc776fd7453c2783562f4d"
        )
        assert det._transformer_subfolder.endswith(
            "4B/zeroshot/checkpoint-87500-95000-w25/transformer"
        )
        assert (det._steps, det._guidance, det._max_side) == (28, 3.5, 1024)
    finally:
        det.close()


@pytest.mark.parametrize("ref", ["FunAILab/ovb", "FunAILab/ovb@main", "FunAILab/ovb@be5da66"])
def test_sidecar_server_refuses_a_movable_revision(ref: str) -> None:
    """The server must load exactly the pinned weights; only a commit sha pins them.

    The server imports torch/diffusers at the top (it runs in its own venv), so
    this compiles only the real ``_split_ref`` definition from the file.
    """
    path = _REPO / "tools" / "_ovb_server.py"
    tree = ast.parse(path.read_text())
    keep: list[ast.stmt] = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_split_ref")
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_SHA" for t in n.targets))
    ]
    ns: dict[str, object] = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    split_ref = ns["_split_ref"]
    assert callable(split_ref)
    sha = "be5da66250cc385ff9c75ebbe4114e13ec305848"
    assert split_ref(f"FunAILab/ovb@{sha}") == ("FunAILab/ovb", sha)
    with pytest.raises(ValueError, match="40-hex sha"):
        split_ref(ref)


def test_ovb_group_declares_sidecar_client_deps(
    expand_dependency_group: Callable[..., list[str]],
) -> None:
    import tomllib

    with (_REPO / "pyproject.toml").open("rb") as handle:
        groups = tomllib.load(handle)["dependency-groups"]
    group = " ".join(expand_dependency_group(groups, "ovb"))
    assert "pyzmq" in group and "msgpack" in group


@pytest.mark.skipif(
    not os.environ.get("OPENRAL_OVB_SIDECAR_VENV") or shutil.which("nvidia-smi") is None,
    reason="needs a provisioned OVB sidecar venv (OPENRAL_OVB_SIDECAR_VENV) and a GPU",
)
def test_real_ovb_locates_person_in_coco_sample() -> None:
    from openral_runner.backends.gstreamer.detector_factory import build_manifest_detector

    os.environ.setdefault("OPENRAL_ALLOW_NONCOMMERCIAL", "1")
    image = Image.open(_FIXTURES / "coco_sample.jpg").convert("RGB")
    frame_bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1]).tobytes()
    det, _ = build_manifest_detector(RSkillManifest.from_yaml(str(_MANIFEST)))
    try:
        meta = det.detect_with_query(frame_bgr, image.width, image.height, "front", "person")
    finally:
        det.close()
    assert meta is not None and meta.detections
    assert all(d.label == "person" for d in meta.detections)
