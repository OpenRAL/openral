"""Vision attachment evidence: geometric gates, primitive contract, fail-closed fallback.

Every mask here is **real** SAM 2.1 output on a real in-tree SO-101 frame,
committed under ``tests/unit/fixtures/`` with provenance in
``sam2_masks.SOURCE.txt`` — nothing is hand-drawn (CLAUDE.md §1.11). The robot
comes from the real ``robots/so101_follower/robot.yaml`` manifest, and the
camera model from that manifest's own wrist intrinsics.

Depth is the one thing the fixtures cannot carry: the two source frames are RGB
captures with no registered depth channel. The tests therefore pair each real
mask with an **analytic** depth field of the surface it actually masked — a
plane at wrist range for the eraser, a perspective-foreshortened tilted plane
for the tablecloth. That is a description of the geometry in the frame, not a
stand-in for a component, and it is what makes the gate arithmetic checkable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachmentEvidenceKind,
    IntrinsicsPinhole,
    RobotDescription,
    scale_intrinsics_to,
)
from openral_core.exceptions import ROSConfigError
from openral_hal._vision_attachment_evidence import (
    VisionAttachmentEvidenceProducer,
    VisionGateConfig,
    backproject_masked_depth,
    clustered_obb_primitives,
    jaw_span_primitive,
)
from PIL import Image

_ROBOT_YAML = "robots/so101_follower/robot.yaml"
_ERASER_MASK = Path("tests/unit/fixtures/sam2_wrist_eraser_mask.png")
_TABLECLOTH_MASK = Path("tests/unit/fixtures/sam2_front_tablecloth_mask.png")

# The SO-101's wrist camera mounted so its optical +z runs along the link's +x
# (REP-103 optical frame vs. a body frame), 2 cm forward and 3 cm up the link.
# A real mount, not identity, so the transform is genuinely exercised.
_R_LINK_FROM_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
_T_LINK_FROM_CAM = np.eye(4, dtype=np.float64)
_T_LINK_FROM_CAM[:3, :3] = _R_LINK_FROM_CAM
_T_LINK_FROM_CAM[:3, 3] = (0.02, 0.0, 0.03)

# Tool center point of the SO-101 gripper in its `gripper_base` frame.
_TCP_IN_LINK = (0.14, 0.0, 0.01)


def _robot() -> RobotDescription:
    return RobotDescription.from_yaml(_ROBOT_YAML)


def _wrist_intrinsics(width: int, height: int) -> IntrinsicsPinhole:
    """The manifest's own wrist intrinsics, rescaled to the fixture resolution."""
    base = next(s.intrinsics for s in _robot().sensors if s.name == "wrist")
    assert base is not None
    return scale_intrinsics_to(base, width, height)


def _mask(path: Path) -> NDArray[np.bool_]:
    return np.asarray(Image.open(path).convert("1"), dtype=bool)


def _plane_depth(
    shape: tuple[int, int], *, distance_m: float, tilt: float = 0.0
) -> NDArray[np.float64]:
    """Depth of a flat surface at ``distance_m``, optionally foreshortened.

    ``tilt`` is the fractional depth change from the top row to the bottom row —
    what a camera sees looking across a table rather than straight down at it.
    """
    height, width = shape
    rows = np.arange(height, dtype=np.float64)[:, None] / max(height - 1, 1)
    return np.broadcast_to(distance_m * (1.0 + tilt * (rows - 0.5)), (height, width)).astype(
        np.float64
    )


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_backproject_rejects_mismatched_shapes() -> None:
    """Mask and depth must come from the same frame; disagreement is an error."""
    k = _wrist_intrinsics(320, 240)
    with pytest.raises(ROSConfigError, match="mask shape"):
        backproject_masked_depth(
            np.ones((240, 320), dtype=bool),
            np.ones((480, 640), dtype=np.float64),
            k,
            min_depth_m=0.02,
            max_depth_m=1.0,
        )


def test_backproject_rejects_intrinsics_at_a_different_resolution() -> None:
    """Unscaled intrinsics would silently skew every back-projected point."""
    unscaled = next(s.intrinsics for s in _robot().sensors if s.name == "wrist")
    assert unscaled is not None  # calibrated at 640x480
    with pytest.raises(ROSConfigError, match="scale_intrinsics_to"):
        backproject_masked_depth(
            np.ones((240, 320), dtype=bool),
            np.ones((240, 320), dtype=np.float64),
            unscaled,
            min_depth_m=0.02,
            max_depth_m=1.0,
        )


def test_backproject_reports_the_depth_valid_fraction() -> None:
    """Out-of-range readings are dropped and counted, not silently back-projected."""
    mask = _mask(_ERASER_MASK)
    k = _wrist_intrinsics(mask.shape[1], mask.shape[0])
    depth = _plane_depth(mask.shape, distance_m=0.12)
    # Blank the top half of the frame, as a sensor's minimum-range hole would.
    depth = depth.copy()
    depth[: mask.shape[0] // 2] = 0.0

    points, fraction = backproject_masked_depth(mask, depth, k, min_depth_m=0.02, max_depth_m=1.0)
    masked_total = int(mask.sum())
    expected = float((mask[mask.shape[0] // 2 :]).sum()) / masked_total
    assert fraction == pytest.approx(expected)
    assert points.shape == (round(fraction * masked_total), 3)


def test_empty_mask_back_projects_to_nothing_without_raising() -> None:
    """ "Nothing was masked" is a gate result, not an exception."""
    k = _wrist_intrinsics(320, 240)
    points, fraction = backproject_masked_depth(
        np.zeros((240, 320), dtype=bool),
        np.full((240, 320), 0.12),
        k,
        min_depth_m=0.02,
        max_depth_m=1.0,
    )
    assert points.shape == (0, 3)
    assert fraction == 0.0


def test_clustered_primitives_respect_the_sixteen_box_contract_and_cover_the_cloud() -> None:
    """The union of the emitted boxes contains every observed point, and there are <= 16."""
    rng = np.random.default_rng(20260813)
    cloud = rng.normal(scale=(0.01, 0.03, 0.02), size=(4000, 3)) + np.array([0.14, 0.0, 0.01])

    primitives, centre, rotation = clustered_obb_primitives(
        cloud, object_id="payload", max_primitives=16
    )
    assert 1 <= len(primitives) <= 16
    local = (cloud - centre) @ rotation
    inside = np.zeros(len(cloud), dtype=bool)
    for prim in primitives:
        half = np.asarray(prim.shape.half_extents_m, dtype=np.float64)
        centre_local = np.asarray(prim.pose_in_object.xyz, dtype=np.float64)
        inside |= np.all(np.abs(local - centre_local) <= half, axis=1)
    assert inside.all(), f"{(~inside).sum()} observed points fell outside every box"
    assert all(p.pose_in_object.frame_id == "payload" for p in primitives)


def test_view_ray_inflation_grows_the_boxes() -> None:
    """The unobserved far side is padded along the view ray, not ignored."""
    rng = np.random.default_rng(7)
    cloud = rng.normal(scale=0.01, size=(500, 3)) + np.array([0.14, 0.0, 0.01])
    bare, _, _ = clustered_obb_primitives(cloud, object_id="p", max_primitives=1)
    padded, _, _ = clustered_obb_primitives(
        cloud,
        object_id="p",
        max_primitives=1,
        inflation_m=0.02,
        inflation_axis_in_link=(1.0, 0.0, 0.0),
    )
    assert sum(padded[0].shape.half_extents_m) > sum(bare[0].shape.half_extents_m)


def test_clustered_primitives_reject_degenerate_input() -> None:
    """An empty cloud or a zero primitive budget is a config error."""
    with pytest.raises(ROSConfigError, match="non-empty point cloud"):
        clustered_obb_primitives(np.zeros((0, 3)), object_id="p")
    with pytest.raises(ROSConfigError, match="max_primitives"):
        clustered_obb_primitives(np.ones((10, 3)), object_id="p", max_primitives=0)


def test_jaw_span_primitive_rejects_a_non_positive_span() -> None:
    """A zero-size fallback box would be no fallback at all."""
    with pytest.raises(ROSConfigError, match="jaw_span_m"):
        jaw_span_primitive(object_id="p", jaw_span_m=0.0)


# ── Producer wiring ────────────────────────────────────────────────────────────


def test_producer_resolves_attach_and_touch_links_from_the_real_manifest() -> None:
    """Attach link and touch links come from the manifest's gripper-role joints."""
    producer = VisionAttachmentEvidenceProducer(_robot())
    assert producer.attach_link == "gripper_base"
    assert producer.on_release() == []


def test_producer_requires_gripper_role_joints() -> None:
    """A robot with no gripper cannot produce attachment evidence, and says so."""
    robot = _robot()
    armless = robot.model_copy(update={"joints": [j for j in robot.joints if j.role != "gripper"]})
    with pytest.raises(ROSConfigError, match="gripper-role joints"):
        VisionAttachmentEvidenceProducer(armless)


def test_on_grasp_rejects_a_malformed_transform() -> None:
    """The caller owns TF2; a transform of the wrong shape is not silently coerced."""
    producer = VisionAttachmentEvidenceProducer(_robot())
    mask = _mask(_ERASER_MASK)
    with pytest.raises(ROSConfigError, match=r"\(4, 4\)"):
        producer.on_grasp(
            mask=mask,
            depth_m=_plane_depth(mask.shape, distance_m=0.12),
            intrinsics=_wrist_intrinsics(mask.shape[1], mask.shape[0]),
            t_link_from_cam=np.eye(3),
            tcp_in_link=_TCP_IN_LINK,
            object_id="payload",
            stamp_ns=1,
        )


# ── The good case ──────────────────────────────────────────────────────────────


def test_real_eraser_mask_produces_a_gated_vision_attachment() -> None:
    """A real wrist mask of a held eraser clears every gate and yields real geometry."""
    producer = VisionAttachmentEvidenceProducer(_robot())
    mask = _mask(_ERASER_MASK)
    attachment, report = producer.on_grasp(
        mask=mask,
        depth_m=_plane_depth(mask.shape, distance_m=0.12),
        intrinsics=_wrist_intrinsics(mask.shape[1], mask.shape[0]),
        t_link_from_cam=_T_LINK_FROM_CAM,
        tcp_in_link=_TCP_IN_LINK,
        object_id="payload",
        stamp_ns=1_700_000_000_000_000_000,
        mask_score_advisory=0.8416,
    )

    assert report.accepted, report.rejections
    assert report.rejections == ()
    assert report.depth_valid_fraction == 1.0
    assert report.centroid_distance_m < 0.02
    # ~2.5 x 4.8 cm of eraser, comfortably inside every extent cap.
    assert max(report.extents_m) < 0.06

    assert attachment.evidence_kind is AttachmentEvidenceKind.VISION_SEGMENTATION
    assert attachment.attach_link == "gripper_base"
    assert attachment.touch_links == ["moving_jaw"]
    assert 1 <= len(attachment.primitives) <= 16
    assert attachment.pose_in_link.frame_id == "gripper_base"
    assert all(p.pose_in_object.frame_id == "payload" for p in attachment.primitives)
    # Vision gives geometry only — no mass, no CoG, no inertia (unlike the sim producer).
    assert attachment.mass_kg is None
    assert attachment.center_of_mass_m is None
    assert attachment.inertia_kg_m2 is None
    assert attachment.stamp_ns == 1_700_000_000_000_000_000


# ── The regression that drives the whole design ────────────────────────────────


def test_tablecloth_mask_is_rejected_on_geometry_despite_its_top_score() -> None:
    """The design-driving regression: a confidently-wrong mask must not be trusted.

    This is the real 59.85%-of-frame mask SAM 2.1 returned at its own **highest**
    score, 0.9781 (see ``tests/unit/fixtures/sam2_masks.SOURCE.txt``). Paired
    with the depth of the table it actually masked, it must be rejected — and
    rejected on geometry: it is far too big to be in the jaws and far too far
    from the tool center point.

    The score is passed in truthfully and must appear in the report **without**
    influencing the verdict. If a future change lets this mask through, a robot
    would plan as if it were carrying a half-metre slab of tablecloth.
    """
    producer = VisionAttachmentEvidenceProducer(_robot())
    mask = _mask(_TABLECLOTH_MASK)
    attachment, report = producer.on_grasp(
        mask=mask,
        depth_m=_plane_depth(mask.shape, distance_m=0.50, tilt=0.35),
        intrinsics=_wrist_intrinsics(mask.shape[1], mask.shape[0]),
        t_link_from_cam=_T_LINK_FROM_CAM,
        tcp_in_link=_TCP_IN_LINK,
        object_id="payload",
        stamp_ns=42,
        mask_score_advisory=0.9781,
    )

    assert not report.accepted
    # Rejected on SIZE and on POSITION — both purely geometric.
    assert "payload_extent" in report.rejections
    assert "jaw_containment" in report.rejections
    # The model's confidence is recorded, and it is high, and it changed nothing.
    assert report.mask_score_advisory == pytest.approx(0.9781)
    assert report.depth_valid_fraction == 1.0
    assert max(report.extents_m) > 0.4

    # Fail-closed, not fail-silent: a conservative box still reaches the kernel.
    assert attachment.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE
    assert len(attachment.primitives) == 1
    assert attachment.primitives[0].shape.half_extents_m == (0.05, 0.05, 0.05)
    assert attachment.pose_in_link.xyz == _TCP_IN_LINK
    assert attachment.confidence < 0.5
    assert "vision_gate_rejected" in (attachment.evidence_ref or "")


def test_insufficient_depth_falls_back_conservatively() -> None:
    """A mask over mostly-invalid depth cannot bound anything, so it is not trusted."""
    producer = VisionAttachmentEvidenceProducer(_robot())
    mask = _mask(_ERASER_MASK)
    depth = _plane_depth(mask.shape, distance_m=0.12)
    depth = depth.copy()
    depth[: int(mask.shape[0] * 0.95)] = 0.0  # sensor returns almost nothing

    attachment, report = producer.on_grasp(
        mask=mask,
        depth_m=depth,
        intrinsics=_wrist_intrinsics(mask.shape[1], mask.shape[0]),
        t_link_from_cam=_T_LINK_FROM_CAM,
        tcp_in_link=_TCP_IN_LINK,
        object_id="payload",
        stamp_ns=7,
    )
    assert not report.accepted
    assert "depth_validity" in report.rejections
    assert report.depth_valid_fraction < 0.30
    assert attachment.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE


def test_empty_mask_falls_back_conservatively() -> None:
    """The segmenter returning nothing must not leave the payload invisible."""
    producer = VisionAttachmentEvidenceProducer(_robot())
    attachment, report = producer.on_grasp(
        mask=np.zeros((240, 320), dtype=bool),
        depth_m=np.full((240, 320), 0.12),
        intrinsics=_wrist_intrinsics(320, 240),
        t_link_from_cam=_T_LINK_FROM_CAM,
        tcp_in_link=_TCP_IN_LINK,
        object_id="payload",
        stamp_ns=9,
    )
    # Named for what it was: nothing was masked, not "the depth sensor failed".
    assert report.rejections == ("empty_mask",)
    assert report.point_count == 0
    assert attachment.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE
    assert len(attachment.primitives) == 1


def test_volume_backstop_fires_on_a_thick_over_large_payload() -> None:
    """The scalar volume cap catches what three per-axis caps individually let through."""
    producer = VisionAttachmentEvidenceProducer(
        _robot(),
        # Per-axis caps deliberately loosened so only the volume backstop can fire.
        config=VisionGateConfig(
            max_payload_extents_m=(1.0, 1.0, 1.0),
            containment_radius_m=10.0,
            max_payload_volume_m3=0.004,
        ),
    )
    mask = _mask(_TABLECLOTH_MASK)
    attachment, report = producer.on_grasp(
        mask=mask,
        depth_m=_plane_depth(mask.shape, distance_m=0.50, tilt=0.35),
        intrinsics=_wrist_intrinsics(mask.shape[1], mask.shape[0]),
        t_link_from_cam=_T_LINK_FROM_CAM,
        tcp_in_link=_TCP_IN_LINK,
        object_id="payload",
        stamp_ns=11,
    )
    assert report.rejections == ("payload_volume",)
    assert report.volume_m3 > 0.004
    assert attachment.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE
