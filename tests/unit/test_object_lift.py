"""Unit tests for lift geometry helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core.schemas import DetectedObject, IntrinsicsPinhole, ObjectDetection2D, Pose6D
from openral_world_state.object_lift import (
    MissEvidence,
    ObjectsLiftError,
    VoxelFrustumLifter,
    aabb_iou_3d,
    build_miss_classifier,
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


def test_decode_occupied_centers_applies_the_grids_orientation():
    """The grid's lattice is the OctoMap's, so its axes are not base_frame's.

    ``OccupancyVoxels`` carries the rotation precisely so the bridge can publish
    one cell per map voxel instead of dilating the map onto a base-aligned
    lattice (issue #173). A consumer that ignores it lifts every object into a
    place the obstacle is not, whenever the base is not map-aligned.
    """
    # 90 degrees about +z: the grid's +x runs along the base frame's +y.
    quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    centers = decode_occupied_centers(
        origin=(0.0, 0.0, 0.0),
        resolution=0.1,
        size_xyz=(2, 1, 1),
        occupancy=bytes([0, 1]),
        orientation_xyzw=quat,
    )
    assert np.allclose(centers[0], [-0.05, 0.15, 0.05])


def test_decode_occupied_centers_refuses_an_unset_orientation():
    """All-zeros is what an unset field carries, and it is not identity.

    Reading it as identity would place obstacles somewhere the robot is not,
    which is a fail-OPEN misread of the world — so it is refused, not defaulted.
    """
    with pytest.raises(ObjectsLiftError, match="unit quaternion"):
        decode_occupied_centers(
            origin=(0.0, 0.0, 0.0),
            resolution=0.1,
            size_xyz=(1, 1, 1),
            occupancy=bytes([1]),
            orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
        )


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


def _cluster(
    center_xyz: tuple[float, float, float], n: int = 5, spread: float = 0.02
) -> NDArray[np.float64]:
    offs = np.linspace(-spread, spread, n)
    pts = np.array([[cx, cy, cz] for cx in offs for cy in offs for cz in offs])
    out: NDArray[np.float64] = pts + np.asarray(center_xyz)
    return out


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


# A 0.2 m cube 2 m in front of the camera, standing on a "shelf" at map z = 1.9.
_CUBE = DetectedObject(
    label="carton",
    confidence=0.9,
    pose=Pose6D(xyz=(0.0, 0.0, 2.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    bbox_3d=(-0.1, -0.1, 1.9, 0.1, 0.1, 2.1),
)


def _miss(
    points: NDArray[np.float64],
    obj: DetectedObject = _CUBE,
    *,
    max_range_m: float = 5.0,
    margin_m: float = 0.03,
) -> MissEvidence:
    return build_miss_classifier(
        occupied_centers_map=points,
        intrinsics=_INTR,
        t_cam_from_map=np.eye(4),
        max_range_m=max_range_m,
        margin_m=margin_m,
    )(obj)


def _plane(z: float, half: float = 0.08, n: int = 5) -> NDArray[np.float64]:
    offs = np.linspace(-half, half, n)
    return np.array([[x, y, z] for x in offs for y in offs])


def test_miss_is_free_when_the_camera_sees_through_the_box():
    """The object is gone: depth lands on the wall behind where it stood."""
    assert _miss(_plane(3.0)) is MissEvidence.FREE


def test_miss_is_occupied_while_the_box_still_holds_points():
    """Undetected but still there (a missed detection): never a strong miss."""
    assert _miss(np.vstack([_cluster((0.0, 0.0, 2.0)), _plane(3.0)])) is MissEvidence.OCCUPIED


def test_miss_is_none_when_occluded():
    """Issue #636: the arm in front of the object hides it — no evidence either way."""
    assert _miss(_plane(1.0, half=0.04)) is MissEvidence.NONE
    # A persistent map still remembers the wall behind; a full occluder outweighs it.
    assert _miss(np.vstack([_plane(1.0, half=0.04, n=7), _plane(3.0, n=4)])) is MissEvidence.NONE


def test_miss_ignores_the_support_the_object_stood_on():
    """A removed carton's shelf keeps the box's floor occupied; that is not the carton."""
    shelf_in_box = np.array([[x, y, 1.9 + 0.01] for x in (-0.05, 0.0, 0.05) for y in (-0.05, 0.05)])
    points = np.vstack([shelf_in_box, _plane(3.0)])
    assert _miss(points) is MissEvidence.FREE
    assert _miss(points, margin_m=0.0) is MissEvidence.OCCUPIED
    # A box no taller than twice the margin cannot be told from its support: counted whole.
    flat = _CUBE.model_copy(update={"bbox_3d": (-0.1, -0.1, 1.9, 0.1, 0.1, 1.92)})
    assert _miss(points, obj=flat) is MissEvidence.OCCUPIED


def test_a_partial_occluder_is_no_evidence_but_stray_points_are_ignored():
    """The gripper covers part of the box and background shows beside it: not a clear view."""
    wall = _plane(3.0)
    front = np.array([[x, 0.0, 1.0] for x in (-0.02, 0.0, 0.02)])
    assert _miss(np.vstack([front, wall])) is MissEvidence.NONE
    assert _miss(np.vstack([front[:2], wall])) is MissEvidence.FREE  # < min_points: noise


def test_one_voxel_layer_lifts_to_a_box_with_volume_that_associates():
    """A head-on carton face is one layer of voxel centres: a zero-volume AABB, whose IoU
    is 0 even with itself — every batch would mint a track. Padded by half a cell it is
    the cells' real extent, and matches itself and the two-layer lift of the same face."""
    from openral_world_state import ObjectMemory

    res = 0.015
    xs = (np.arange(5) - 2) * res + 1e-9  # float noise the mean of the centres never cancels
    one = np.array([[x, y, 2.0] for x in xs for y in xs])
    two = np.vstack([one, one + np.array([0.0, 0.0, res])])
    det = ObjectDetection2D(label="carton", confidence=0.9, bbox_xyxy=(40, 40, 60, 60))

    def lift(points: NDArray[np.float64], pad_m: float) -> list[DetectedObject]:
        return VoxelFrustumLifter(k_nearest=50, min_voxels=3).lift(
            detections=[det],
            occupied_centers_base=points,
            intrinsics=_INTR,
            frame_size=(100, 100),
            t_cam_from_base=np.eye(4),
            t_map_from_base=np.eye(4),
            pad_m=pad_m,
        )

    flat = lift(one, 0.0)[0].bbox_3d
    assert flat is not None and flat[2] == flat[5]  # the trap: no thickness at all
    padded = lift(one, res / 2)[0].bbox_3d
    assert padded is not None and padded[5] - padded[2] == pytest.approx(res)

    mem = ObjectMemory()
    for t in range(1, 21):
        out = mem.tick(
            lift(one if t % 2 else two, res / 2),
            stamp_ns=t,
            miss_evidence=lambda o: MissEvidence.NONE,
        )
    assert [o.track_id for o in out] == [0]


def test_a_minimal_lift_is_confirmed_by_the_points_that_made_it():
    """Three voxels in a column: the lowest lies in the support band, yet the box is occupied."""
    column = np.array([[0.0, 0.0, 1.95], [0.0, 0.0, 2.0], [0.0, 0.0, 2.05]])
    # Camera looking along +y so map z is "up" in the image: y forward.
    t = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 2.0], [0, 1.0, 0, 2.0], [0, 0, 0, 1.0]])
    bottle = _CUBE.model_copy(update={"bbox_3d": (-0.02, -0.02, 1.95, 0.02, 0.02, 2.05)})
    wall = np.array([[x, 1.0, z] for x in (-0.01, 0.0, 0.01) for z in (1.97, 2.0, 2.03)])

    def classify(points: NDArray[np.float64]) -> MissEvidence:
        return build_miss_classifier(
            occupied_centers_map=points, intrinsics=_INTR, t_cam_from_map=t, max_range_m=5.0
        )(bottle)

    assert classify(np.vstack([column, wall])) is MissEvidence.OCCUPIED
    # Taken away: only the surface it stood on is left in the box's floor band.
    shelf = np.array([[x, 0.0, 1.95] for x in (-0.01, 0.0, 0.01)])
    assert classify(np.vstack([shelf, wall])) is MissEvidence.FREE


def test_a_flat_box_keeps_its_own_points_under_jitter():
    """A box lifted from one voxel layer has zero thickness; a millimetre of map drift
    must not empty it and turn its own surface into "background seen through"."""
    face = _CUBE.model_copy(update={"bbox_3d": (-0.1, -0.1, 2.0, 0.1, 0.1, 2.0)})
    for drift in (0.0, 1e-13, 0.001, -0.001, 0.02):
        assert _miss(_plane(2.0 + drift), obj=face) is MissEvidence.OCCUPIED, drift
    assert _miss(_plane(3.0), obj=face) is MissEvidence.FREE


def test_miss_is_none_when_not_wholly_in_view_or_out_of_range():
    wall = _plane(3.0)
    at_edge = _CUBE.model_copy(update={"bbox_3d": (0.9, -0.1, 1.9, 1.1, 0.1, 2.1)})
    behind = _CUBE.model_copy(update={"bbox_3d": (-0.1, -0.1, -2.1, 0.1, 0.1, -1.9)})
    straddling = _CUBE.model_copy(update={"bbox_3d": (-0.1, -0.1, -0.1, 0.1, 0.1, 0.1)})
    assert _miss(wall, obj=at_edge) is MissEvidence.NONE
    assert _miss(wall, obj=behind) is MissEvidence.NONE
    assert _miss(wall, obj=straddling) is MissEvidence.NONE
    assert _miss(wall, max_range_m=2.0) is MissEvidence.NONE
    assert _miss(wall, obj=_CUBE.model_copy(update={"bbox_3d": None})) is MissEvidence.NONE


def test_miss_is_none_without_depth_behind_the_box():
    """A depth hole, or nothing mapped behind: no ray is known to have passed through."""
    assert _miss(np.empty((0, 3))) is MissEvidence.NONE
    assert _miss(_plane(3.0, n=1)) is MissEvidence.NONE  # 1 point < min_points
    assert _miss(_plane(3.0) + np.array([1.0, 0.0, 0.0])) is MissEvidence.NONE  # off to the side


def test_public_reexports():
    import openral_world_state as ws

    assert ws.VoxelFrustumLifter is not None
    assert ws.ObjectMemory is not None
    assert ws.aabb_iou_3d is not None
    assert ws.build_miss_classifier is not None
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
