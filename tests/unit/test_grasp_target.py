"""Pre-grasp target geometry (design note §2.2): seed, prompt, region, map cross-check, tracking.

The scene is the real OpenArm cell's, numerically: the ``deploy_e2e`` octomap lattice
(20 mm cells, coverage ball r = 1.05 m centred (0, 0, 0.5) in ``openarm_base``), the
floor 0.698 m below the base (``robots/openarm/robot.yaml`` ``base_to_root_xyz_rpy``),
the Thor ZED's measured mount (``robots/openarm/units/thor.yaml``) and its live
``camera_info`` K (fx = fy = 1498.18, cx = 936.11, cy = 541.81 at 1920x1080).

Masks are the **real** SAM 2.1 outputs committed for ``test_vision_attachment_evidence``
(provenance in ``tests/unit/fixtures/sam2_masks.SOURCE.txt``), upsampled by an integer
factor into the ZED raster. As in that file, depth is the one thing the fixtures cannot
carry, so each mask is paired with an analytic depth field: the ray-cast distance to a
horizontal slab (the target's top face) seen through the real mount. No mocks
(CLAUDE.md §1.11).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, PlaceRegion, Pose6D, RobotUnit
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import yaw_to_quat_xyzw
from openral_hal._grasp_target import (
    TargetRefusal,
    TargetRegionFit,
    VoxelLattice,
    occupied_centers_in_box,
    project_point,
    region_covers_occupied,
    target_region_from_mask,
    target_seed_from_voxels,
    track_region,
)
from PIL import Image

_RES = 0.02
_COVERAGE_RADIUS = 1.05  # deploy_e2e.launch.py _octomap_coverage_radius()
_COVERAGE_CENTRE = (0.0, 0.0, 0.5)  # deploy_e2e.launch.py _OCTOMAP_COVERAGE_CENTRE
_FLOOR_Z = -0.698  # robots/openarm/robot.yaml base_to_root_xyz_rpy
_SUPPORT_Z = _FLOOR_Z + 0.40  # a 40 cm table top
_BOX_HEIGHT = 0.12
_TOP_Z = _SUPPORT_Z + _BOX_HEIGHT
_FRAME = "openarm_base"
# The octomap's lattice is not base-aligned in general; a yaw exercises that.
_LATTICE_YAW = 0.2

_ZED_K = IntrinsicsPinhole(width=1920, height=1080, fx=1498.18, fy=1498.18, cx=936.11, cy=541.81)
_ERASER_MASK = Path("tests/unit/fixtures/sam2_wrist_eraser_mask.png")
_TABLECLOTH_MASK = Path("tests/unit/fixtures/sam2_front_tablecloth_mask.png")
_BUCKET2 = Path("packages/openral_foxglove_bringup/openral_foxglove_bringup/bucket2_markers.py")


# ── Scene ──────────────────────────────────────────────────────────────────────


def _rpy(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """URDF/ROS fixed-axis rpy: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


@cache
def _t_base_from_optical() -> NDArray[np.float64]:
    """``openarm_base -> zed_camera_link`` (thor.yaml) composed with REP-103 body→optical."""
    unit = RobotUnit.from_yaml("robots/openarm/units/thor.yaml")
    head = next(s for s in unit.sensors if s.name == "head_zed")
    assert head.static_transform_xyz_rpy is not None
    x, y, z, roll, pitch, yaw = head.static_transform_xyz_rpy
    body_from_optical = np.array(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    t = np.eye(4)
    t[:3, :3] = _rpy(roll, pitch, yaw) @ body_from_optical
    t[:3, 3] = (x, y, z)
    return t


def _box_centre_xy() -> tuple[float, float]:
    """Where the ZED's optical axis meets the box's top face: the target sits mid-image."""
    t = _t_base_from_optical()
    origin, axis = t[:3, 3], t[:3, 2]
    s = (_TOP_Z - origin[2]) / axis[2]
    return float(origin[0] + s * axis[0]), float(origin[1] + s * axis[1])


def _lattice_rotation() -> NDArray[np.float64]:
    return _rpy(0.0, 0.0, _LATTICE_YAW)


def _empty_lattice() -> tuple[VoxelLattice, NDArray[np.float64]]:
    """The coverage ball's bounding lattice and every cell centre in it."""
    n = int(2.0 * _COVERAGE_RADIUS / _RES) + 1
    rot = _lattice_rotation()
    origin = np.asarray(_COVERAGE_CENTRE) - rot @ np.full(3, n * _RES / 2.0)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ijk = np.stack((i.ravel(order="F"), j.ravel(order="F"), k.ravel(order="F")), axis=1)
    centres = (ijk + 0.5) * _RES @ rot.T + origin
    grid = VoxelLattice(
        frame_id=_FRAME,
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        orientation_xyzw=yaw_to_quat_xyzw(_LATTICE_YAW),
        resolution=_RES,
        size=(n, n, n),
        occupancy=np.zeros(n**3, dtype=np.uint8),
    )
    return grid, centres


def _in_obb(
    p: NDArray[np.float64], centre: tuple[float, float, float], half: tuple[float, ...], yaw: float
) -> NDArray[np.bool_]:
    local = (p - np.asarray(centre)) @ _rpy(0.0, 0.0, yaw)
    return np.asarray(np.all(np.abs(local) <= np.asarray(half), axis=1))


def _scene_lattice(boxes: list[tuple[float, float, float, float, float]]) -> VoxelLattice:
    """Floor + table + boxes ``(cx, cy, half_x, half_y, yaw)`` standing on it, in the ball."""
    grid, c = _empty_lattice()
    in_ball = np.linalg.norm(c - np.asarray(_COVERAGE_CENTRE), axis=1) <= _COVERAGE_RADIUS
    # The floor lies below the ball's lowest point (z = -0.55): the real map never sees it
    # either, so this term rasterises nothing — kept so the scene says what the cell holds.
    floor = np.abs(c[:, 2] - _FLOOR_Z) <= _RES / 2.0
    table = (
        (c[:, 2] <= _SUPPORT_Z)
        & (c[:, 2] >= _SUPPORT_Z - 0.04)
        & (c[:, 0] >= 0.0)
        & (c[:, 0] <= 0.8)
        & (np.abs(c[:, 1]) <= 0.6)
    )
    occ = floor | table
    zc = _SUPPORT_Z + _BOX_HEIGHT / 2.0
    for bx, by, hx, hy, yaw in boxes:
        occ |= _in_obb(c, (bx, by, zc), (hx, hy, _BOX_HEIGHT / 2.0), yaw)
    return VoxelLattice(
        grid.frame_id,
        grid.origin,
        grid.orientation_xyzw,
        grid.resolution,
        grid.size,
        (occ & in_ball).astype(np.uint8),
    )


def _search_box() -> PlaceRegion:
    bx, by = _box_centre_xy()
    return PlaceRegion(
        frame_id=_FRAME,
        pose=Pose6D(xyz=(bx, by, _SUPPORT_Z + 0.10), quat_xyzw=(0, 0, 0, 1), frame_id=_FRAME),
        half_extents=(0.25, 0.30, 0.20),
        evidence_ref="scene:pick_declaration.search_box",
    )


# ── Masks and depth ────────────────────────────────────────────────────────────


def _mask_in_zed(path: Path, factor: int) -> NDArray[np.bool_]:
    """A real SAM mask, nearest-upsampled by ``factor``, centred on the ZED principal point
    (rows/cols beyond the raster cropped)."""
    small = np.asarray(Image.open(path).convert("1"), dtype=bool)
    big = np.kron(small, np.ones((factor, factor), dtype=bool))
    canvas = np.zeros((_ZED_K.height, _ZED_K.width), dtype=bool)
    rows, cols = np.nonzero(big)
    rows = rows - round(rows.mean()) + round(_ZED_K.cy)
    cols = cols - round(cols.mean()) + round(_ZED_K.cx)
    keep = (rows >= 0) & (rows < _ZED_K.height) & (cols >= 0) & (cols < _ZED_K.width)
    canvas[rows[keep], cols[keep]] = True
    return canvas


def _slab_depth(top_z: float) -> NDArray[np.float64]:
    """Optical-z depth of the horizontal plane ``z = top_z`` at every ZED pixel centre."""
    t = _t_base_from_optical()
    v, u = np.mgrid[0 : _ZED_K.height, 0 : _ZED_K.width].astype(np.float64) + 0.5
    rays = np.stack(((u - _ZED_K.cx) / _ZED_K.fx, (v - _ZED_K.cy) / _ZED_K.fy, np.ones_like(u)))
    dz = np.tensordot(t[2, :3], rays, axes=1)  # base-z change per unit optical z
    with np.errstate(divide="ignore"):
        depth = (top_z - t[2, 3]) / dz
    return np.where(depth > 0.0, depth, 0.0)


def _footprint_lattice(mask: NDArray[np.bool_], depth: NDArray[np.float64]) -> VoxelLattice:
    """Rasterise the same slab into the lattice: every cell under the masked top face, from
    the support plane to the top — what octomap holds for a solid object."""
    grid, _ = _empty_lattice()
    t = _t_base_from_optical()
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    cam = np.stack(
        (((cols + 0.5) - _ZED_K.cx) * z / _ZED_K.fx, ((rows + 0.5) - _ZED_K.cy) * z / _ZED_K.fy, z),
        axis=1,
    )
    xy = (cam @ t[:3, :3].T + t[:3, 3])[:, :2]
    occ = grid.occupancy.copy()
    n = grid.size[0]
    for zl in np.arange(_SUPPORT_Z + _RES / 4.0, _TOP_Z, _RES / 2.0):
        p = np.column_stack((xy, np.full(len(xy), zl)))
        ijk = np.floor((p - np.asarray(grid.origin)) @ _lattice_rotation() / _RES).astype(int)
        occ[ijk[:, 0] + n * (ijk[:, 1] + n * ijk[:, 2])] = 1
    return VoxelLattice(
        grid.frame_id, grid.origin, grid.orientation_xyzw, grid.resolution, grid.size, occ
    )


def _fit(mask: NDArray[np.bool_]) -> tuple[TargetRegionFit, NDArray[np.float64]]:
    depth = _slab_depth(_TOP_Z)
    return target_region_from_mask(
        mask,
        depth,
        _ZED_K,
        _t_base_from_optical(),
        support_z=_SUPPORT_Z,
        resolution=_RES,
        frame_id=_FRAME,
        evidence_ref="head_zed:sam2.1:test",
    ), depth


# ── Seed from voxels ───────────────────────────────────────────────────────────


def test_seed_is_the_box_top_centre() -> None:
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.05, 0.035, 0.4)])
    centres = occupied_centers_in_box(grid, _search_box())
    seed = target_seed_from_voxels(grid, centres, support_z=_SUPPORT_Z, min_cells=20)
    assert seed.refusal is None and seed.point is not None
    assert len(seed.cluster_sizes) == 1  # the table layer is support, not a cluster
    x, y, z = seed.point
    assert math.hypot(x - bx, y - by) <= _RES
    assert abs(z - _TOP_Z) <= _RES


def test_seed_refuses_two_equal_objects() -> None:
    """HZ-01xx-2: two comparable candidates in the search box — no guessing."""
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by - 0.12, 0.04, 0.04, 0.0), (bx, by + 0.12, 0.04, 0.04, 0.0)])
    centres = occupied_centers_in_box(grid, _search_box())
    seed = target_seed_from_voxels(grid, centres, support_z=_SUPPORT_Z, min_cells=20)
    assert seed.point is None
    assert seed.refusal is TargetRefusal.AMBIGUOUS
    assert len(seed.cluster_sizes) == 2


def test_seed_refuses_too_few_cells() -> None:
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.015, 0.015, 0.0)])
    centres = occupied_centers_in_box(grid, _search_box())
    seed = target_seed_from_voxels(grid, centres, support_z=_SUPPORT_Z, min_cells=20)
    assert seed.refusal is TargetRefusal.TOO_FEW_CELLS


def test_search_box_in_another_frame_is_refused() -> None:
    grid = _scene_lattice([])
    box = _search_box().model_copy(update={"frame_id": "world"})
    with pytest.raises(ROSConfigError, match="frame"):
        occupied_centers_in_box(grid, box)


# ── Prompt projection ──────────────────────────────────────────────────────────


def test_seed_projects_into_the_zed_image_and_back() -> None:
    bx, by = _box_centre_xy()
    t_cam_from_base = np.asarray(np.linalg.inv(_t_base_from_optical()), dtype=np.float64)
    uv = project_point((bx, by, _TOP_Z), t_cam_from_base, _ZED_K)
    assert uv is not None
    # The box was placed on the optical axis: the prompt lands on the principal point.
    assert uv == pytest.approx((_ZED_K.cx, _ZED_K.cy), abs=1e-6)
    # Behind the camera: no prompt.
    assert project_point((-1.0, 0.0, 0.5), t_cam_from_base, _ZED_K) is None


# ── Region from a real mask ────────────────────────────────────────────────────


def test_region_from_real_mask_sits_above_support_within_caps_and_matches_the_map() -> None:
    mask = _mask_in_zed(_ERASER_MASK, factor=4)
    fit, depth = _fit(mask)
    assert fit.refusal is None and fit.region is not None
    region = fit.region
    hx, hy, hz = region.half_extents
    # HZ-01xx-6: the lower face is one voxel above the support plane, never below.
    assert region.pose.xyz[2] - hz == pytest.approx(_SUPPORT_Z + _RES)
    assert region.pose.xyz[2] - hz >= _SUPPORT_Z
    assert max(hx, hy, hz) <= 0.20
    assert region.volume_m3() <= 0.03
    assert region.frame_id == _FRAME and region.pose.frame_id == _FRAME
    # The region reaches past the measured top by the pad, not further.
    pad = math.sqrt(3.0) * _RES / 2.0 + 0.01
    assert region.pose.xyz[2] + hz == pytest.approx(_TOP_Z + pad, abs=1e-3)

    count, ok = region_covers_occupied(_footprint_lattice(mask, depth), region)
    assert ok and count > 0
    # The same region against a map with nothing there: refused.
    count_empty, ok_empty = region_covers_occupied(_scene_lattice([]), region)
    assert (count_empty, ok_empty) == (0, False)


def test_whole_view_mask_is_refused_by_the_caps() -> None:
    """The committed mis-aimed SAM mask (59.85 % of its frame, at score 0.977) spread over the
    ZED view describes a tabletop, not one graspable object."""
    fit, _ = _fit(_mask_in_zed(_TABLECLOTH_MASK, factor=3))
    assert fit.region is None
    assert fit.refusal is TargetRefusal.HALF_EXTENT_CAP
    assert max(fit.half_extents) > 0.20


def test_mask_with_no_depth_is_refused() -> None:
    mask = _mask_in_zed(_ERASER_MASK, factor=4)
    fit = target_region_from_mask(
        mask,
        np.zeros((_ZED_K.height, _ZED_K.width)),
        _ZED_K,
        _t_base_from_optical(),
        support_z=_SUPPORT_Z,
        resolution=_RES,
        frame_id=_FRAME,
        evidence_ref="head_zed:sam2.1:test",
    )
    assert fit.refusal is TargetRefusal.TOO_FEW_POINTS and fit.point_count == 0


def test_surface_at_support_height_is_refused() -> None:
    """A mask on the table itself has no height above the support: nothing to exempt."""
    mask = _mask_in_zed(_ERASER_MASK, factor=4)
    fit = target_region_from_mask(
        mask,
        _slab_depth(_SUPPORT_Z),
        _ZED_K,
        _t_base_from_optical(),
        support_z=_SUPPORT_Z,
        resolution=_RES,
        frame_id=_FRAME,
        evidence_ref="head_zed:sam2.1:test",
    )
    assert fit.refusal is TargetRefusal.NO_HEIGHT_ABOVE_SUPPORT


# ── Tracking gate ──────────────────────────────────────────────────────────────


def test_track_accepts_one_voxel_and_refuses_five() -> None:
    fit, _ = _fit(_mask_in_zed(_ERASER_MASK, factor=4))
    assert fit.region is not None
    prev = fit.region
    x, y, z = prev.pose.xyz

    def shifted(dx: float) -> PlaceRegion:
        pose = prev.pose.model_copy(update={"xyz": (x + dx, y, z)})
        out: PlaceRegion = prev.model_copy(update={"pose": pose})
        return out

    kw = {"max_centroid_shift_m": _RES, "extents_tol_m": _RES}
    assert track_region(prev, shifted(_RES), **kw)
    assert not track_region(prev, shifted(5 * _RES), **kw)
    assert not track_region(prev, prev.model_copy(update={"frame_id": "world"}), **kw)


# ── Lattice convention cross-check ─────────────────────────────────────────────


def _load_bucket2() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_bucket2_markers_gt", _BUCKET2)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bucket2_markers_gt"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lattice_centres_agree_with_bucket2_markers_on_a_rotated_lattice() -> None:
    """The HAL re-derives the OccupancyVoxels convention; it must match the viewer's."""
    bucket2 = _load_bucket2()
    axis = np.array((0.3, -0.5, 0.8)) / np.linalg.norm((0.3, -0.5, 0.8))
    half = 0.7 / 2.0
    q = (*(axis * math.sin(half)), math.cos(half))
    rng = np.random.default_rng(7)
    size = (7, 5, 4)
    occ = (rng.random(int(np.prod(size))) < 0.3).astype(np.uint8)
    origin = (0.31, -0.42, 0.05)
    grid = VoxelLattice(_FRAME, origin, q, _RES, size, occ)
    ours = grid.occupied_centers()
    theirs = np.asarray(bucket2.occupied_voxel_centers(origin, _RES, size, occ.tolist(), q))
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
    # A search box that spans the whole lattice returns every occupied centre.
    box = PlaceRegion(
        frame_id=_FRAME,
        pose=Pose6D(xyz=tuple(ours.mean(axis=0)), quat_xyzw=(0, 0, 0, 1), frame_id=_FRAME),
        half_extents=(0.5, 0.5, 0.5),
    )
    assert len(occupied_centers_in_box(grid, box)) == len(ours)


def test_lattice_rejects_an_unset_orientation() -> None:
    with pytest.raises(ROSConfigError, match="unit quaternion"):
        VoxelLattice(_FRAME, (0, 0, 0), (0, 0, 0, 0), _RES, (1, 1, 1), np.ones(1, np.uint8))
