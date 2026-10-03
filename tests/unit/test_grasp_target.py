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
    support_top_from_voxels,
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
    return _fit_on(mask, _SUPPORT_Z)


def _fit_on(
    mask: NDArray[np.bool_], support_z: float
) -> tuple[TargetRegionFit, NDArray[np.float64]]:
    depth = _slab_depth(_TOP_Z)
    return target_region_from_mask(
        mask,
        depth,
        _ZED_K,
        _t_base_from_optical(),
        support_z=support_z,
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


def _column(bottom_z: float) -> PlaceRegion:
    """The search box's footprint, from ``bottom_z`` up to the box top."""
    bx, by = _box_centre_xy()
    top = _TOP_Z + 0.05
    return PlaceRegion(
        frame_id=_FRAME,
        pose=Pose6D(xyz=(bx, by, (top + bottom_z) / 2), quat_xyzw=(0, 0, 0, 1), frame_id=_FRAME),
        half_extents=(0.25, 0.30, (top - bottom_z) / 2),
    )


def test_support_below_the_search_box_bottom_is_measured_not_assumed() -> None:
    """HZ-01xx-6: a detection bbox whose min-z sits 5 cm *below* the table top must not
    lower the region into the table — the region stands on the measured table top."""
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.05, 0.035, 0.4)])
    box_bottom = _SUPPORT_Z - 0.05
    measured = support_top_from_voxels(
        grid,
        occupied_centers_in_box(grid, _column(box_bottom - 0.15)),
        near_xy=(bx, by),
        min_cells=8,
    )
    assert measured is not None
    assert abs(measured - _SUPPORT_Z) <= _RES / 2 + 1e-9  # the table top, to the lattice
    seed = target_seed_from_voxels(
        grid, occupied_centers_in_box(grid, _search_box()), support_z=measured, min_cells=20
    )
    assert seed.refusal is None and seed.cluster_sizes[0] < 200  # the box alone, no table
    fit, _ = _fit_on(_mask_in_zed(_ERASER_MASK, factor=4), measured)
    assert fit.region is not None
    lower = fit.region.pose.xyz[2] - fit.region.half_extents[2]
    assert lower == pytest.approx(measured + _RES)
    assert lower > _SUPPORT_Z + _RES / 2  # every table cell stays outside the region
    # The bbox's own bottom would have dropped the lower face into the table.
    assert box_bottom + _RES < _SUPPORT_Z - _RES / 2


def test_no_support_layer_is_refused() -> None:
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.05, 0.035, 0.4)])
    centres = grid.occupied_centers()
    no_table = grid.occupancy.copy()
    occupied = np.flatnonzero(no_table)
    no_table[occupied[centres[:, 2] <= _SUPPORT_Z]] = 0
    floating = VoxelLattice(  # the box alone: no table under it
        grid.frame_id, grid.origin, grid.orientation_xyzw, grid.resolution, grid.size, no_table
    )
    column = occupied_centers_in_box(floating, _column(_SUPPORT_Z - 0.20))
    # Only the box's own layers remain, and none of them reaches past its own footprint.
    assert support_top_from_voxels(floating, column, near_xy=(bx, by), min_cells=1) is None
    empty = np.empty((0, 3))
    assert support_top_from_voxels(floating, empty, near_xy=(bx, by), min_cells=1) is None


def test_a_table_beside_the_target_is_not_support_under_it() -> None:
    """The column holds a table, but none of it around the target (a gap past the edge)."""
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.05, 0.035, 0.4)])
    centres = grid.occupied_centers()
    holed = grid.occupancy.copy()
    occupied = np.flatnonzero(holed)
    near = np.all(np.abs(centres[:, :2] - (bx, by)) <= 0.18, axis=1)
    holed[occupied[near & (centres[:, 2] <= _SUPPORT_Z)]] = 0
    grid = VoxelLattice(
        grid.frame_id, grid.origin, grid.orientation_xyzw, grid.resolution, grid.size, holed
    )
    column = occupied_centers_in_box(grid, _column(_SUPPORT_Z - 0.20))
    assert len(column[column[:, 2] <= _SUPPORT_Z]) > 100  # the table is in the column
    assert support_top_from_voxels(grid, column, near_xy=(bx, by), min_cells=8) is None


def test_a_tilted_lattice_has_no_support_layer() -> None:
    occ = np.ones(8, dtype=np.uint8)
    pitched = (0.0, math.sin(0.15), 0.0, math.cos(0.15))
    tilted = VoxelLattice("base", (0.0, 0.0, 0.0), pitched, 0.1, (2, 2, 2), occ)
    with pytest.raises(ROSConfigError, match="z-up"):
        support_top_from_voxels(tilted, tilted.occupied_centers(), near_xy=(0, 0), min_cells=1)


# ── Support under the target, octomap-realistic ──────────────────────────────────
#
# A base-aligned 20 mm lattice built cell by cell, the way a head camera's octomap
# holds a scene: surfaces only (shells, not solids), the table hidden under the
# target and in its shadow behind it (the camera looks along +x).

_CELLS = (24, 24, 20)


def _cells_lattice(cells: set[tuple[int, int, int]]) -> VoxelLattice:
    occ = np.zeros(int(np.prod(_CELLS)), dtype=np.uint8)
    for i, j, k in cells:
        occ[i + _CELLS[0] * (j + _CELLS[1] * k)] = 1
    return VoxelLattice(_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), _RES, _CELLS, occ)


def _slab(i: range, j: range, k: int) -> set[tuple[int, int, int]]:
    return {(a, b, k) for a in i for b in j}


def _shell(i: range, j: range, k: range) -> set[tuple[int, int, int]]:
    """A box's visible surface from a camera at -x above: top and every side but +x."""
    out = _slab(i, j, k[-1])
    for kk in k:
        out |= {(i[0], b, kk) for b in j} | {(a, j[0], kk) for a in i}
        out |= {(a, j[-1], kk) for a in i}
    return out


def _seen_table(k: int, item_i: range, item_j: range, shadow: int) -> set[tuple[int, int, int]]:
    """The whole table layer minus the cells under the item and in its +x shadow."""
    hidden = _slab(range(item_i[0], item_i[-1] + 1 + shadow), item_j, k)
    return _slab(range(_CELLS[0]), range(_CELLS[1]), k) - hidden


def _centre_xy(i: range, j: range) -> tuple[float, float]:
    return ((i[0] + i[-1] + 1) * _RES / 2, (j[0] + j[-1] + 1) * _RES / 2)


def _column_of(grid: VoxelLattice, i: range, j: range) -> NDArray[np.float64]:
    """Occupied centres in a tight search column over cells ``i`` x ``j``, all heights."""
    c = grid.occupied_centers()
    lo = np.array((i[0], j[0])) * _RES
    hi = np.array((i[-1] + 1, j[-1] + 1)) * _RES
    return c[np.all((c[:, :2] >= lo) & (c[:, :2] <= hi), axis=1)]


def test_a_dense_target_top_over_an_occluded_table_ring_is_not_the_support() -> None:
    """A 10x10 cm item in a tight detection column: the table there is mostly hidden
    (under the item, in its shadow), so the item's top layer (25 cells) holds more than
    half the table layer's — "the densest layer's highest near-equal layer" picked it."""
    item_i, item_j = range(8, 13), range(8, 13)
    cells = _seen_table(5, item_i, item_j, shadow=3) | _shell(item_i, item_j, range(6, 11))
    grid = _cells_lattice(cells)
    column = _column_of(grid, range(6, 14), range(6, 14))
    layers, counts = np.unique(np.floor(column[:, 2] / _RES).astype(int), return_counts=True)
    top, table = counts[layers == 10][0], counts[layers == 5][0]
    assert top >= 0.5 * table, "the scene must hold the trap the old rule fell into"
    support = support_top_from_voxels(grid, column, near_xy=_centre_xy(item_i, item_j), min_cells=8)
    assert support == pytest.approx(6 * _RES)  # the table's top face, not the item's


def test_a_small_target_finds_the_table_under_it() -> None:
    item_i, item_j = range(10, 12), range(10, 12)
    cells = _seen_table(5, item_i, item_j, shadow=2) | _shell(item_i, item_j, range(6, 8))
    grid = _cells_lattice(cells)
    column = _column_of(grid, range(7, 15), range(7, 15))
    support = support_top_from_voxels(grid, column, near_xy=_centre_xy(item_i, item_j), min_cells=8)
    assert support == pytest.approx(6 * _RES)


def test_a_target_on_a_shelf_board_edge_stands_on_the_board_not_the_bench() -> None:
    """A board (k=10) whose front edge the item overhangs; a bench 10 cm lower (k=5)."""
    item_i, item_j = range(6, 11), range(8, 13)
    board = _slab(range(8, 24), range(24), 10) - _slab(range(8, 14), item_j, 10)
    bench = _slab(range(24), range(24), 5) - _slab(range(6, 24), range(24), 5)
    cells = board | bench | _shell(item_i, item_j, range(11, 16))
    grid = _cells_lattice(cells)
    column = _column_of(grid, range(3, 16), range(5, 16))
    support = support_top_from_voxels(grid, column, near_xy=_centre_xy(item_i, item_j), min_cells=8)
    assert support == pytest.approx(11 * _RES)  # the board's top face


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


def test_a_25_by_30_cm_target_finds_the_table_around_it() -> None:
    """The fixed 0.10 m probe square lay wholly under such a target, where the head camera
    sees no table; the ring around the footprint reaches the visible table past it."""
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.125, 0.15, 0.0)])
    centres = grid.occupied_centers()
    hidden = grid.occupancy.copy()
    under = np.all(np.abs(centres[:, :2] - (bx, by)) <= (0.125, 0.15), axis=1)
    hidden[np.flatnonzero(hidden)[under & (centres[:, 2] <= _SUPPORT_Z)]] = 0
    grid = VoxelLattice(
        grid.frame_id, grid.origin, grid.orientation_xyzw, grid.resolution, grid.size, hidden
    )
    column = occupied_centers_in_box(grid, _column(_SUPPORT_Z - 0.20))
    support = support_top_from_voxels(grid, column, near_xy=(bx, by), min_cells=8)
    assert support is not None
    assert abs(support - _SUPPORT_Z) <= _RES / 2 + 1e-9
