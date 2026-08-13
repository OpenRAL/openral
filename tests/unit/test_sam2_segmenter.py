"""SAM 2.1 segmenter backend: pure helpers, manifest dispatch, and real inference.

The inference tests load the real ``facebook/sam2.1-hiera-small`` checkpoint and
run it on real in-tree frames — no mocked model, no synthetic image (CLAUDE.md
§1.11). They skip only when the opt-in ``sam2`` dependency group is absent
(``just sync --group sam2``).

CPU is the correct execution target on this host and not a fallback of
convenience: the dev box's GTX 1060 is sm_61, below the sm_70 floor of the
installed torch CUDA wheels, so there is no GPU path to prefer here
(CLAUDE.md §1.12).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import IntrinsicsPinhole, RSkillManifest
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import SegmenterEngine
from openral_runner.backends.gstreamer.sam2_segmenter import (
    Sam2Segmenter,
    build_mask_candidates,
    project_point_to_pixel,
)
from openral_runner.backends.gstreamer.segmenter_factory import build_manifest_segmenter

_RSKILL = Path("rskills/rskill-sam2_1-any-grasped_object_mask-bf16/rskill.yaml")
_WRIST_FRAME = Path("rskills/rskill-smolvla-so101-eraser_place-bf16/media/wrist_mid.png")
_FRONT_FRAME = Path("rskills/gr00t-n17-so101-fruit/assets/so101_fruit_ep100_front.png")

# The prompt convention the design used: a point at (0.52*W, 0.68*H), standing in
# for the TF-projected tool center point on a wrist view.
_PROMPT_FRAC = (0.52, 0.68)


def _manifest() -> RSkillManifest:
    return RSkillManifest.model_validate(yaml.safe_load(_RSKILL.read_text()))


def _frame_bgr(path: Path) -> tuple[bytes, int, int]:
    """Load a real in-tree PNG as raw BGR bytes, the GStreamer appsink shape."""
    image_module = pytest.importorskip("PIL.Image")
    img = image_module.open(path).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR
    return arr.tobytes(), int(img.width), int(img.height)


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_project_point_to_pixel_maps_optical_axis_to_principal_point() -> None:
    """A point on the optical axis lands exactly on (cx, cy)."""
    k = IntrinsicsPinhole(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    assert project_point_to_pixel((0.0, 0.0, 0.5), k) == (320.0, 240.0)


def test_project_point_to_pixel_rejects_behind_camera_and_out_of_frame() -> None:
    """Behind the optical plane or outside the image is ``None``, never a guess."""
    k = IntrinsicsPinhole(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    assert project_point_to_pixel((0.0, 0.0, -0.5), k) is None
    assert project_point_to_pixel((0.0, 0.0, 0.0), k) is None
    # 1 m off-axis at 0.1 m range projects thousands of pixels outside the frame.
    assert project_point_to_pixel((1.0, 0.0, 0.1), k) is None


def test_build_mask_candidates_orders_by_area_not_by_score() -> None:
    """Ordering is geometric; the highest-scoring candidate gets no privilege."""
    small = np.zeros((8, 8), dtype=bool)
    small[0, :4] = True  # 4 px
    large = np.ones((8, 8), dtype=bool)  # 64 px
    # The LARGE mask carries the top score, exactly as in the tablecloth failure.
    out = build_mask_candidates([large, small], scores=[0.98, 0.11], min_mask_area_px=1)
    assert [c.area_px for c in out] == [4, 64]
    assert [c.score_advisory for c in out] == [0.11, 0.98]


def test_build_mask_candidates_drops_degenerate_masks() -> None:
    """A handful of pixels cannot bound a payload, so it is dropped outright."""
    tiny = np.zeros((8, 8), dtype=bool)
    tiny[0, 0] = True
    big = np.ones((8, 8), dtype=bool)
    out = build_mask_candidates([tiny, big], scores=[0.9, 0.5], min_mask_area_px=64)
    assert [c.area_px for c in out] == [64]


def test_build_mask_candidates_rejects_length_mismatch() -> None:
    """Parallel arrays that disagree are a config error, not a silent truncation."""
    with pytest.raises(ROSConfigError, match="length mismatch"):
        build_mask_candidates([np.ones((2, 2), dtype=bool)], scores=[], min_mask_area_px=1)


# ── Manifest dispatch ──────────────────────────────────────────────────────────


def test_real_manifest_builds_the_sam2_backend() -> None:
    """The shipped rSkill manifest dispatches to the in-process SAM 2.1 backend."""
    manifest = _manifest()
    assert manifest.kind == "segmenter"
    assert manifest.segmenter is not None
    assert manifest.segmenter.engine is SegmenterEngine.SAM2_HF

    backend = build_manifest_segmenter(manifest)
    assert isinstance(backend, Sam2Segmenter)
    assert backend.kind == "segmenter"
    assert backend.model_id == "OpenRAL/rskill-sam2_1-any-grasped_object_mask-bf16"


def test_build_manifest_segmenter_rejects_a_detector_manifest() -> None:
    """A detector manifest is not a segmenter, and saying so is not optional."""
    detector = RSkillManifest.model_validate(
        yaml.safe_load(Path("rskills/omdet-turbo-locator/rskill.yaml").read_text())
    )
    with pytest.raises(ROSConfigError, match="is not a"):
        build_manifest_segmenter(detector)


def test_segmenter_rejects_over_budget_prompts_before_loading_a_model() -> None:
    """The prompt-point bound is enforced by the contract, not by the model."""
    backend = Sam2Segmenter(
        model_id="test", weights_source="facebook/sam2.1-hiera-small", max_prompt_points=2
    )
    with pytest.raises(ROSConfigError, match="max_prompt_points"):
        backend.segment(
            b"", 2, 2, positive_points=[(0.0, 0.0)], negative_points=[(1.0, 1.0), (1.0, 0.0)]
        )
    with pytest.raises(ROSConfigError, match="at least one positive point"):
        backend.segment(b"", 2, 2, positive_points=[])


# ── Real inference on real frames ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpu_segmenter() -> Sam2Segmenter:
    """The real SAM 2.1 backend, pinned to CPU (this host has no usable CUDA)."""
    pytest.importorskip(
        "transformers",
        reason="SAM 2.1 backend needs the opt-in group: just sync --group sam2",
    )
    pytest.importorskip("torch", reason="SAM 2.1 backend needs torch (just sync --group sam2)")
    manifest = _manifest()
    assert manifest.segmenter is not None
    return Sam2Segmenter(
        model_id=manifest.name,
        weights_source="facebook/sam2.1-hiera-small",
        max_prompt_points=manifest.segmenter.max_prompt_points,
        multimask=manifest.segmenter.multimask,
        min_mask_area_px=manifest.segmenter.min_mask_area_px,
        device="cpu",
    )


def test_wrist_frame_point_prompt_isolates_the_grasped_object(
    cpu_segmenter: Sam2Segmenter,
) -> None:
    """A well-aimed TCP prompt on a real wrist frame masks only the held object.

    The good case: on the SO-101 wrist view holding an eraser, a point prompt
    returns a compact mask — a few percent of the frame — rather than the
    gripper or the table.
    """
    frame, width, height = _frame_bgr(_WRIST_FRAME)
    u, v = _PROMPT_FRAC[0] * width, _PROMPT_FRAC[1] * height
    candidates = cpu_segmenter.segment(frame, width, height, positive_points=[(u, v)])

    assert candidates, "SAM 2.1 returned no candidate above min_mask_area_px"
    frame_px = float(width * height)
    coverage = [c.area_px / frame_px for c in candidates]
    # Every candidate on this frame is a small object, not the scene.
    assert min(coverage) < 0.10, coverage
    # Ordering is area-ascending regardless of score.
    assert coverage == sorted(coverage)


def test_mis_aimed_prompt_returns_a_huge_mask_at_the_models_top_score(
    cpu_segmenter: Sam2Segmenter,
) -> None:
    """The measurement that forbids score-based gating, re-run for real.

    On a third-person frame the "grasp point" projection is meaningless, so the
    prompt lands on background. SAM 2.1 returns essentially the whole tablecloth
    — and gives it the HIGHEST score of its three candidates.

    This test asserts the *failure*, deliberately. If a future model or
    transformers release makes the score trustworthy here, this test breaks and
    someone re-examines the gate design on purpose rather than by accident. Until
    then, `openral_hal._vision_attachment_evidence` may not gate on this number.
    """
    frame, width, height = _frame_bgr(_FRONT_FRAME)
    u, v = _PROMPT_FRAC[0] * width, _PROMPT_FRAC[1] * height
    candidates = cpu_segmenter.segment(frame, width, height, positive_points=[(u, v)])

    assert candidates
    frame_px = float(width * height)
    biggest = candidates[-1]  # area-ascending
    assert biggest.area_px / frame_px > 0.5, "expected the tablecloth-scale mask"
    # The damning part: the biggest, most wrong mask also carries the top score.
    assert biggest.score_advisory == max(c.score_advisory for c in candidates)
    assert biggest.score_advisory > 0.9
