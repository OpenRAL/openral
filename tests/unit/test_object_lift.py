"""Unit tests for ADR-0035 lift geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest
from openral_core.schemas import DetectedObject, IntrinsicsPinhole, ObjectDetection2D, Pose6D
from openral_world_state.object_lift import (
    VoxelFrustumLifter,
    aabb_iou_3d,
    build_in_fov_predicate,
    decode_occupied_centers,
    depth_cloud_to_centers_base,
    homogeneous_from_quat_xyz,
)


def test_homogeneous_identity_quat_is_translation():
    m = homogeneous_from_quat_xyz((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(m[:3, :3], np.eye(3))
    assert np.allclose(m[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(m[3], [0.0, 0.0, 0.0, 1.0])


def test_homogeneous_90deg_about_z():
    m = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    p = m @ np.array([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(p[:3], [0.0, 1.0, 0.0], atol=1e-9)


def test_homogeneous_rejects_degenerate_quat():
    with pytest.raises(ValueError):
        homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))


def test_homogeneous_scale_invariant_to_quat_norm():
    unit = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    scaled = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0))
    assert np.allclose(unit, scaled)


def test_decode_occupied_centers_row_major_x_fastest():
    occ = bytes([0, 1, 0, 0])
    centers = decode_occupied_centers(
        origin=(0.0, 0.0, 0.0), resolution=0.1, size_xyz=(2, 2, 1), occupancy=occ
    )
    assert centers.shape == (1, 3)
    assert np.allclose(centers[0], [0.15, 0.05, 0.05])


def test_decode_occupied_centers_origin_offset_and_empty():
    empty = decode_occupied_centers(
        origin=(1.0, 1.0, 1.0), resolution=0.5, size_xyz=(1, 1, 1), occupancy=bytes([0])
    )
    assert empty.shape == (0, 3)
    one = decode_occupied_centers(
        origin=(1.0, 1.0, 1.0), resolution=0.5, size_xyz=(1, 1, 1), occupancy=bytes([7])
    )
    assert np.allclose(one[0], [1.25, 1.25, 1.25])


def test_decode_occupied_centers_length_mismatch_raises():
    with pytest.raises(ValueError):
        decode_occupied_centers(
            origin=(0.0, 0.0, 0.0), resolution=0.1, size_xyz=(2, 2, 2), occupancy=bytes([1, 0])
        )


def test_aabb_iou_3d_identical_disjoint_partial():
    box = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert aabb_iou_3d(box, box) == pytest.approx(1.0)
    far = (5.0, 5.0, 5.0, 6.0, 6.0, 6.0)
    assert aabb_iou_3d(box, far) == 0.0
    half = (0.5, 0.0, 0.0, 1.5, 1.0, 1.0)
    assert aabb_iou_3d(box, half) == pytest.approx(1.0 / 3.0)


# Camera looking down +z (optical convention), 100x100, fx=fy=100, principal centre.
_INTR = IntrinsicsPinhole(
    width=100,
    height=100,
    fx=100.0,
    fy=100.0,
    cx=50.0,
    cy=50.0,
    distortion_model="none",
    distortion_coeffs=[],
)


def _cluster(center_xyz, n=5, spread=0.02):
    offs = np.linspace(-spread, spread, n)
    pts = np.array([[cx, cy, cz] for cx in offs for cy in offs for cz in offs])
    return pts + np.asarray(center_xyz)


def test_lift_single_object_center_at_cluster_centroid():
    lifter = VoxelFrustumLifter(k_nearest=200, min_voxels=3)
    voxels = _cluster((0.0, 0.0, 2.0), n=5, spread=0.02)
    eye = np.eye(4)
    det = ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(40, 40, 60, 60))
    objs = lifter.lift(
        detections=[det],
        occupied_centers_base=voxels,
        intrinsics=_INTR,
        frame_size=(100, 100),
        t_cam_from_base=eye,
        t_map_from_base=eye,
        map_frame="map",
    )
    assert len(objs) == 1
    o = objs[0]
    assert o.label == "cup"
    assert o.pose.frame_id == "map"
    assert o.pose.quat_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert o.pose.xyz == pytest.approx((0.0, 0.0, 2.0), abs=0.03)
    assert o.bbox_3d is not None


def test_lift_skips_when_too_few_voxels():
    lifter = VoxelFrustumLifter(k_nearest=10, min_voxels=5)
    voxels = _cluster((0.0, 0.0, 2.0), n=1)
    det = ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(40, 40, 60, 60))
    objs = lifter.lift(
        detections=[det],
        occupied_centers_base=voxels,
        intrinsics=_INTR,
        frame_size=(100, 100),
        t_cam_from_base=np.eye(4),
        t_map_from_base=np.eye(4),
    )
    assert objs == []


def test_lift_k_nearest_to_box_center_rejects_offcenter_background():
    lifter = VoxelFrustumLifter(k_nearest=125, min_voxels=3)
    near = _cluster((0.0, 0.0, 2.0), n=5, spread=0.02)
    far = _cluster((1.2, 0.0, 6.0), n=5, spread=0.02)
    voxels = np.concatenate([near, far])
    det = ObjectDetection2D(label="box", confidence=0.8, bbox_xyxy=(30, 30, 80, 70))
    objs = lifter.lift(
        detections=[det],
        occupied_centers_base=voxels,
        intrinsics=_INTR,
        frame_size=(100, 100),
        t_cam_from_base=np.eye(4),
        t_map_from_base=np.eye(4),
    )
    assert len(objs) == 1
    assert objs[0].pose.xyz[2] == pytest.approx(2.0, abs=0.2)
    assert objs[0].pose.xyz[0] == pytest.approx(0.0, abs=0.15)


def test_lift_excludes_behind_camera_voxels():
    lifter = VoxelFrustumLifter(k_nearest=200, min_voxels=3)
    front = _cluster((0.0, 0.0, 2.0), n=5, spread=0.02)
    behind = np.array([[0.0, 0.0, -1.0]])  # behind the camera plane
    voxels = np.concatenate([front, behind])
    det = ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(40, 40, 60, 60))
    objs = lifter.lift(
        detections=[det],
        occupied_centers_base=voxels,
        intrinsics=_INTR,
        frame_size=(100, 100),
        t_cam_from_base=np.eye(4),
        t_map_from_base=np.eye(4),
    )
    assert len(objs) == 1
    assert objs[0].pose.xyz == pytest.approx((0.0, 0.0, 2.0), abs=0.03)


def test_lifter_rejects_bad_config():
    with pytest.raises(ValueError):
        VoxelFrustumLifter(k_nearest=0)
    with pytest.raises(ValueError):
        VoxelFrustumLifter(min_voxels=0)


def test_lift_scales_box_when_frame_differs_from_intrinsics():
    lifter = VoxelFrustumLifter(k_nearest=200, min_voxels=3)
    voxels = _cluster((0.0, 0.0, 2.0), n=5, spread=0.02)
    det = ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(80, 80, 120, 120))
    objs = lifter.lift(
        detections=[det],
        occupied_centers_base=voxels,
        intrinsics=_INTR,
        frame_size=(200, 200),
        t_cam_from_base=np.eye(4),
        t_map_from_base=np.eye(4),
    )
    assert len(objs) == 1
    assert objs[0].pose.xyz == pytest.approx((0.0, 0.0, 2.0), abs=0.03)


def test_lift_empty_inputs():
    lifter = VoxelFrustumLifter()
    assert (
        lifter.lift(
            detections=[],
            occupied_centers_base=np.empty((0, 3)),
            intrinsics=_INTR,
            frame_size=(100, 100),
            t_cam_from_base=np.eye(4),
            t_map_from_base=np.eye(4),
        )
        == []
    )


def test_in_fov_predicate():
    pred = build_in_fov_predicate(intrinsics=_INTR, t_cam_from_map=np.eye(4))
    in_view = DetectedObject(
        label="x",
        confidence=0.5,
        pose=Pose6D(xyz=(0.0, 0.0, 2.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    behind = DetectedObject(
        label="x",
        confidence=0.5,
        pose=Pose6D(xyz=(0.0, 0.0, -2.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    far_side = DetectedObject(
        label="x",
        confidence=0.5,
        pose=Pose6D(xyz=(100.0, 0.0, 2.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    assert pred(in_view) is True
    assert pred(behind) is False
    assert pred(far_side) is False


def test_public_reexports():
    import openral_world_state as ws

    assert ws.VoxelFrustumLifter is not None
    assert ws.ObjectMemory is not None
    assert ws.aabb_iou_3d is not None
    assert ws.build_in_fov_predicate is not None
    assert ws.decode_occupied_centers is not None
    assert ws.depth_cloud_to_centers_base is not None


# ── #11 depth-cloud fallback (octomap-free lift depth source) ──────────────────


def test_depth_cloud_transforms_to_base_frame():
    # Cloud in an optical frame translated +1 m in base x; identity rotation.
    t_base_from_cloud = homogeneous_from_quat_xyz((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    pts = np.array([[0.0, 0.0, 2.0], [0.5, -0.5, 3.0]], dtype=np.float64)
    out = depth_cloud_to_centers_base(pts, t_base_from_cloud)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1.0, 0.0, 2.0])
    assert np.allclose(out[1], [1.5, -0.5, 3.0])


def test_depth_cloud_drops_non_finite_points():
    t = np.eye(4, dtype=np.float64)
    pts = np.array(
        [[1.0, 1.0, 1.0], [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [2.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    out = depth_cloud_to_centers_base(pts, t)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])


def test_depth_cloud_all_non_finite_returns_empty():
    out = depth_cloud_to_centers_base(
        np.full((4, 3), np.nan, dtype=np.float64), np.eye(4, dtype=np.float64)
    )
    assert out.shape == (0, 3)


def test_depth_cloud_subsamples_to_max_points():
    pts = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (1000, 1))
    out = depth_cloud_to_centers_base(pts, np.eye(4, dtype=np.float64), max_points=100)
    # Uniform stride keeps it at or below the cap (never silently unbounded).
    assert 0 < out.shape[0] <= 100
