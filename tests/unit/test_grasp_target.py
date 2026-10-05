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
    _in_region,
    cell_closed_region,
    map_completed_region,
    mask_without_removed_points,
    occupied_centers_in_box,
    occupied_touching_outside,
    project_point,
    region_covers_occupied,
    region_within,
    support_top_from_voxels,
    target_region_from_mask,
    target_region_from_masks,
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


def _prism_view(
    top: NDArray[np.bool_], top_z: float, bottom_z: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """The solid under a masked top face as the ZED sees it: SAM masks the whole visible
    object — the real top-face mask plus the prism's near sides down to ``bottom_z`` —
    ray-marched through the real mount in 0.5 mm steps; depth 0 off the object."""
    t = _t_base_from_optical()
    slab = _slab_depth(top_z)
    rows, cols = np.nonzero(top)
    z = slab[rows, cols]
    cam = np.stack(
        (((cols + 0.5) - _ZED_K.cx) * z / _ZED_K.fx, ((rows + 0.5) - _ZED_K.cy) * z / _ZED_K.fy, z),
        axis=1,
    )
    cell = 0.002  # the footprint raster; the top face samples it at < 1 mm
    ij = np.floor((cam @ t[:3, :3].T + t[:3, 3])[:, :2] / cell).astype(int)
    lo = ij.min(axis=0) - 2
    foot = np.zeros(tuple(ij.max(axis=0) - lo + 3), dtype=bool)
    foot[ij[:, 0] - lo[0], ij[:, 1] - lo[1]] = True
    # The image window holding the prism: its footprint box projected at both heights.
    box = [(lo + np.array((a, b))) * cell for a in (0, foot.shape[0]) for b in (0, foot.shape[1])]
    corners = np.array([(*xy, h, 1.0) for xy in box for h in (top_z, bottom_z)])
    opt = (np.linalg.inv(t) @ corners.T)[:3]
    us, vs = _ZED_K.fx * opt[0] / opt[2] + _ZED_K.cx, _ZED_K.fy * opt[1] / opt[2] + _ZED_K.cy
    v, u = np.mgrid[
        max(0, int(vs.min())) : min(_ZED_K.height, int(vs.max()) + 1),
        max(0, int(us.min())) : min(_ZED_K.width, int(us.max()) + 1),
    ]
    rays = np.stack(((u + 0.5 - _ZED_K.cx) / _ZED_K.fx, (v + 0.5 - _ZED_K.cy) / _ZED_K.fy))
    d = np.tensordot(t[:3, :2], rays, axes=1) + t[:3, 2:3, None]  # base dir per optical z
    t_top, t_bot = (top_z - t[2, 3]) / d[2], (bottom_z - t[2, 3]) / d[2]
    hit = np.zeros(u.shape)
    for s in np.linspace(0.0, 1.0, int((top_z - bottom_z) / 0.0005) + 1):
        tt = t_top + s * (t_bot - t_top)
        fi = np.floor((t[0, 3] + tt * d[0]) / cell).astype(int) - lo[0]
        fj = np.floor((t[1, 3] + tt * d[1]) / cell).astype(int) - lo[1]
        ok = (fi >= 0) & (fi < foot.shape[0]) & (fj >= 0) & (fj < foot.shape[1]) & (hit == 0)
        ok[ok] = foot[fi[ok], fj[ok]]
        hit[ok] = tt[ok]
    depth = np.zeros((_ZED_K.height, _ZED_K.width))
    depth[v, u] = hit
    depth[top] = slab[top]
    return top | (depth > 0.0), depth


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
    mask: NDArray[np.bool_], support_z: float, *, bottom_z: float = _SUPPORT_Z
) -> tuple[TargetRegionFit, NDArray[np.float64]]:
    """Fit the box under the real top-face ``mask`` standing on ``bottom_z``."""
    seen, depth = _prism_view(mask, _TOP_Z, bottom_z)
    return target_region_from_mask(
        seen,
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
    seed = target_seed_from_voxels(
        grid, centres, near_xy=(bx, by), support_z=_SUPPORT_Z, min_cells=20
    )
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
    """HZ-0115-6: a detection bbox whose min-z sits 5 cm *below* the table top must not
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
        grid,
        occupied_centers_in_box(grid, _search_box()),
        near_xy=(bx, by),
        support_z=measured,
        min_cells=20,
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


def test_a_taller_neighbour_never_makes_the_targets_top_its_support() -> None:
    """HZ-0115-2: a flat box (the target, search box centred on it) beside a taller bottle.
    Anchored on whatever stood above each layer, the scan took the bottle as "the target"
    at the box's top layer, found the box's top face ringing the bottle, called it the
    support and seeded on the bottle. Anchored on the box at every layer, the support is
    the table, and the bigger bottle standing beside the target makes the seed ambiguous."""
    box_i, box_j = range(6, 11), range(8, 13)
    bottle_i, bottle_j = range(12, 15), range(9, 12)
    table = _seen_table(5, box_i, box_j, shadow=2)
    table -= _slab(range(12, 20), bottle_j, 5)  # under the bottle and its long shadow
    cells = table | _shell(box_i, box_j, range(6, 8)) | _shell(bottle_i, bottle_j, range(6, 16))
    grid = _cells_lattice(cells)
    column = _column_of(grid, range(4, 17), range(6, 15))
    near = _centre_xy(box_i, box_j)
    support = support_top_from_voxels(grid, column, near_xy=near, min_cells=8)
    assert support is not None and support == pytest.approx(6 * _RES)  # the table, not its top
    seed = target_seed_from_voxels(grid, column, near_xy=near, support_z=support, min_cells=8)
    assert seed.point is None and seed.refusal is TargetRefusal.AMBIGUOUS
    assert seed.cluster_sizes[0] < seed.cluster_sizes[1]  # the anchored box comes first


@pytest.mark.parametrize("under", ["same_footprint_box", "narrower_riser"])
def test_a_target_standing_on_another_object_is_refused_not_on_support(under: str) -> None:
    """HZ-0115-6. Voxels: a 6 cm item on a 6 cm box of its footprint, or on a riser hidden
    under it, clusters with what it stands on — one object whose bottom is on the table, so
    the seed's contact check passes. Mask: SAM names the item alone, whose cloud ends 6 cm
    above the table; a region pinned to the table would exempt the lower object's cells."""
    item_i, item_j = range(8, 13), range(8, 13)
    below = (item_i, item_j) if under == "same_footprint_box" else (range(9, 12), range(9, 12))
    cells = _seen_table(5, item_i, item_j, shadow=3) | _shell(*below, range(6, 9))
    grid = _cells_lattice(cells | _shell(item_i, item_j, range(9, 12)))
    column = _column_of(grid, range(4, 17), range(4, 17))
    near = _centre_xy(item_i, item_j)
    support = support_top_from_voxels(grid, column, near_xy=near, min_cells=8)
    assert support is not None and support == pytest.approx(6 * _RES)
    seed = target_seed_from_voxels(grid, column, near_xy=near, support_z=support, min_cells=8)
    assert seed.point is not None and seed.bottom_z is not None
    assert seed.bottom_z - support <= 2.0 * _RES + 1e-9, "the voxel check is fooled"

    mask = _mask_in_zed(_ERASER_MASK, factor=4)
    fit, _ = _fit_on(mask, _SUPPORT_Z, bottom_z=_SUPPORT_Z + 0.06)
    assert (fit.region, fit.refusal) == (None, TargetRefusal.NOT_ON_SUPPORT)
    on_table, _ = _fit_on(mask, _SUPPORT_Z)  # the same item standing on the table
    assert on_table.region is not None and on_table.box is None
    # The refused fit still reports the visible part's box (the leg's occlusion test),
    # its lower face pinned one voxel above the support like a region's.
    assert fit.box is not None and fit.box.half_extents == pytest.approx(fit.half_extents)
    assert fit.box.pose.xyz[2] - fit.box.half_extents[2] == pytest.approx(_SUPPORT_Z + _RES)


def test_seed_refuses_two_equal_objects() -> None:
    """HZ-0115-2: two comparable candidates in the search box — no guessing."""
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by - 0.12, 0.04, 0.04, 0.0), (bx, by + 0.12, 0.04, 0.04, 0.0)])
    centres = occupied_centers_in_box(grid, _search_box())
    seed = target_seed_from_voxels(
        grid, centres, near_xy=(bx, by), support_z=_SUPPORT_Z, min_cells=20
    )
    assert seed.point is None
    assert seed.refusal is TargetRefusal.AMBIGUOUS
    assert len(seed.cluster_sizes) == 2


def test_seed_refuses_too_few_cells() -> None:
    bx, by = _box_centre_xy()
    grid = _scene_lattice([(bx, by, 0.015, 0.015, 0.0)])
    centres = occupied_centers_in_box(grid, _search_box())
    seed = target_seed_from_voxels(
        grid, centres, near_xy=(bx, by), support_z=_SUPPORT_Z, min_cells=20
    )
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
    # HZ-0115-6: the lower face is one voxel above the support plane, never below.
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
    fit = target_region_from_mask(
        _mask_in_zed(_TABLECLOTH_MASK, factor=3),
        _slab_depth(_TOP_Z),
        _ZED_K,
        _t_base_from_optical(),
        support_z=_SUPPORT_Z,
        resolution=_RES,
        frame_id=_FRAME,
        evidence_ref="head_zed:sam2.1:test",
    )
    assert fit.region is None
    assert fit.refusal is TargetRefusal.HALF_EXTENT_CAP
    assert max(fit.half_extents) > 0.20


def _base_z(depth: NDArray[np.float64]) -> NDArray[np.float64]:
    """Base-frame height of every ZED pixel's depth point (``nan`` where there is none)."""
    t = _t_base_from_optical()
    v, u = np.mgrid[0 : _ZED_K.height, 0 : _ZED_K.width].astype(np.float64) + 0.5
    rays = np.stack(((u - _ZED_K.cx) / _ZED_K.fx, (v - _ZED_K.cy) / _ZED_K.fy, np.ones_like(u)))
    z = np.tensordot(t[2, :3], rays, axes=1) * depth + t[2, 3]
    return np.where(depth > 0.0, z, np.nan)


def _masks_fit(masks: list[NDArray[np.bool_]], depth: NDArray[np.float64]) -> TargetRegionFit:
    return target_region_from_masks(
        masks,
        depth,
        _ZED_K,
        _t_base_from_optical(),
        support_z=_SUPPORT_Z,
        resolution=_RES,
        frame_id=_FRAME,
        evidence_ref="head_zed:sam2.1:test",
    )


def test_a_whole_stack_mask_never_exempts_what_a_refused_part_stands_on() -> None:
    """HZ-0115-6 under SAM's multimask: the item (top 6 cm) is refused ``not_on_support``,
    the item + its same-footprint box (the whole 12 cm stack) reaches the table. Taking
    the first candidate that fits would pin the region to the table over the box."""
    whole, depth = _prism_view(_mask_in_zed(_ERASER_MASK, factor=4), _TOP_Z, _SUPPORT_Z)
    item = whole & (_base_z(depth) >= _SUPPORT_Z + 0.06)
    assert _masks_fit([item], depth).refusal is TargetRefusal.NOT_ON_SUPPORT
    assert _masks_fit([whole], depth).region is not None  # alone, the stack fits
    for masks in ([item, whole], [whole, item]):
        fit = _masks_fit(masks, depth)
        assert (fit.region, fit.refusal) == (None, TargetRefusal.STACKED)


def test_a_top_face_subpart_does_not_veto_the_whole_object() -> None:
    """An ordinary can: SAM's subpart is its lid (the top face, no height of its own), which
    never reaches the table — the whole-object mask still fits."""
    top = _mask_in_zed(_ERASER_MASK, factor=4)
    whole, depth = _prism_view(top, _TOP_Z, _SUPPORT_Z)
    assert _masks_fit([top], depth).refusal is TargetRefusal.NOT_ON_SUPPORT
    fit = _masks_fit([top, whole], depth)
    assert fit.refusal is None and fit.region is not None


def test_an_all_refused_candidate_set_reports_its_most_severe_refusal() -> None:
    """A contradiction beats a lost view whatever the segmenter's order: a mask spread over
    the table (caps) and an empty one (too few points) is the caps refusal."""
    cloth = _mask_in_zed(_TABLECLOTH_MASK, factor=3)
    depth = _slab_depth(_TOP_Z)
    empty = np.zeros_like(cloth)
    for masks in ([cloth, empty], [empty, cloth]):
        assert _masks_fit(masks, depth).refusal is TargetRefusal.HALF_EXTENT_CAP


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


def test_a_probe_margin_under_two_cells_is_refused_not_widened() -> None:
    """The ring starts one cell out; a margin under two cells used to be silently raised."""
    item_i, item_j = range(10, 12), range(10, 12)
    cells = _seen_table(5, item_i, item_j, shadow=2) | _shell(item_i, item_j, range(6, 8))
    grid = _cells_lattice(cells)
    column = _column_of(grid, range(7, 15), range(7, 15))
    xy = _centre_xy(item_i, item_j)
    with pytest.raises(ROSConfigError, match="under two cells"):
        support_top_from_voxels(grid, column, near_xy=xy, min_cells=8, probe_margin_m=0.03)
    exact = support_top_from_voxels(grid, column, near_xy=xy, min_cells=8, probe_margin_m=0.04)
    assert exact == pytest.approx(6 * _RES)


def test_the_robots_own_pixels_leave_the_target_mask() -> None:
    """Closing on the target, the fingers enter its SAM mask; their depth points are the
    robot's, which the robot self-filter removed from the same capture. Only masked pixels
    whose point survives that filter are kept (Isaac 2026-10-04: the fingers in the mask
    grew the re-fit to the hand, or failed it outright as ``not_on_support``)."""
    from openral_core import IntrinsicsPinhole

    k = IntrinsicsPinhole(width=64, height=64, fx=64.0, fy=64.0, cx=32.0, cy=32.0)
    t = np.diag([1.0, -1.0, -1.0, 1.0])  # camera 1 m up, looking straight down
    t[2, 3] = 1.0
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    depth = np.full((64, 64), 0.9)  # the target's top, z = 0.1
    depth[16:48, 16:24] = 0.7  # a finger over its left edge, z = 0.3
    target_only = mask.copy()
    target_only[16:48, 16:24] = False
    rows, cols = np.nonzero(target_only)
    z = depth[rows, cols]
    pts_cam = np.stack([(cols + 0.5 - 32.0) / 64.0 * z, (rows + 0.5 - 32.0) / 64.0 * z, z], axis=1)
    kept = pts_cam @ t[:3, :3].T + t[:3, 3] + 0.002  # what the self-filter let through, 2 mm off

    out = mask_without_removed_points(mask, depth, k, t, kept)
    assert out.dtype == np.bool_ and out.shape == mask.shape
    assert not out[16:48, 16:24].any(), "the finger's pixels stay in the mask"
    assert out[16:48, 24:48].all(), "the target's own pixels must survive"
    # Nothing survived the filter (the hand covers the whole target): nothing is kept.
    assert not mask_without_removed_points(mask, depth, k, t, np.zeros((0, 3))).any()


def _yawed(
    xyz: tuple[float, float, float], half: tuple[float, float, float], yaw_deg: float
) -> PlaceRegion:
    return PlaceRegion(
        frame_id=_FRAME,
        half_extents=half,
        pose=Pose6D(xyz=xyz, quat_xyzw=yaw_to_quat_xyzw(math.radians(yaw_deg)), frame_id=_FRAME),
    )


def _z_up_lattice(resolution: float) -> VoxelLattice:
    """A z-up lattice with a cell corner at the origin (the closure needs only r and axes)."""
    return VoxelLattice(
        _FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), resolution, (1, 1, 1), np.zeros(1, np.uint8)
    )


@pytest.mark.parametrize(("yaw_deg", "grow"), [(0.0, _RES / 2.0), (45.0, _RES / math.sqrt(2.0))])
def test_the_cell_closure_grows_sideways_by_the_cells_reach_and_up_only(
    yaw_deg: float, grow: float
) -> None:
    """r/2·(|cos θ|+|sin θ|) horizontally (r/2 at 0°, r/√2 at 45°), the top up r/2, the
    bottom exactly where the fit put it."""
    tight = _yawed((0.3, 0.1, 0.2), (0.05, 0.03, 0.04), yaw_deg)
    closed, clamped = cell_closed_region(tight, _z_up_lattice(_RES), max_half_extent_m=0.2)
    assert not clamped
    hx, hy, hz = closed.half_extents
    assert (hx, hy) == (pytest.approx(0.05 + grow), pytest.approx(0.03 + grow))
    assert closed.pose.xyz[:2] == tight.pose.xyz[:2]
    assert closed.pose.quat_xyzw == tight.pose.quat_xyzw
    assert closed.pose.xyz[2] - hz == pytest.approx(0.2 - 0.04), "the bottom moved"
    assert closed.pose.xyz[2] + hz == pytest.approx(0.2 + 0.04 + _RES / 2.0)


def test_the_cell_closure_is_clamped_at_the_cap_with_the_bottom_kept() -> None:
    tight = _yawed((0.3, 0.1, 0.2), (0.195, 0.03, 0.198), 0.0)
    closed, clamped = cell_closed_region(tight, _z_up_lattice(_RES), max_half_extent_m=0.2)
    assert clamped
    assert closed.half_extents == (0.2, pytest.approx(0.04), 0.2)
    assert closed.pose.xyz[2] - 0.2 == pytest.approx(0.2 - 0.198)


def test_the_cell_closure_refuses_another_frame_or_a_tilted_region() -> None:
    lattice = _z_up_lattice(_RES)
    with pytest.raises(ROSConfigError, match="frame"):
        cell_closed_region(
            _yawed((0, 0, 0), (0.05, 0.05, 0.05), 0.0).model_copy(update={"frame_id": "other"}),
            lattice,
            max_half_extent_m=0.2,
        )
    tilted = _yawed((0, 0, 0), (0.05, 0.05, 0.05), 0.0)
    pitched = tilted.pose.model_copy(update={"quat_xyzw": (0.0, 0.3826834, 0.0, 0.9238795)})
    tilted = tilted.model_copy(update={"pose": pitched})
    with pytest.raises(ROSConfigError, match="gravity-aligned"):
        cell_closed_region(tilted, lattice, max_half_extent_m=0.2)


def test_isaac_i37_the_targets_top_edge_cell_is_exempt_only_once_cell_closed() -> None:
    """Isaac i36/i37: the kernel stopped ``openarm_right_finger_pair`` on the cell centred
    (0.2925, -0.2325, -0.4275) with the grasp exemption active. The true surface (x≈0.289)
    lies inside that cell (x 0.285-0.300), its centre half a cell outside the tight fit.
    Cell-closed, the target's own cell is exempt; the support cell straight under the
    region (centre half a cell below its bottom) stays outside."""
    res = 0.015
    tight = _yawed((0.2430, -0.1966, -0.4387), (0.0615, 0.0453, 0.0263), -96.0)
    closed, _ = cell_closed_region(tight, _z_up_lattice(res), max_half_extent_m=0.2)
    edge = np.array([[0.2925, -0.2325, -0.4275]])
    bottom = tight.pose.xyz[2] - tight.half_extents[2]
    support = np.array([[0.2430, -0.1966, bottom - res / 2.0]])
    assert not _in_region(edge, tight).any()
    assert _in_region(edge, closed).all()
    assert not _in_region(support, closed).any()
    assert closed.pose.xyz[2] - closed.half_extents[2] == pytest.approx(bottom)


def test_every_cell_the_fit_touches_has_its_centre_in_the_closure() -> None:
    """The bound itself, sampled: every lattice cell that intersects a yawed box (cube vs
    box separating-axis test) has its centre inside the closed box, for many yaws — except
    the cells whose centre lies below the fit's bottom, which the closure never reaches."""
    rng = np.random.default_rng(7)
    lattice = _z_up_lattice(_RES)
    ijk = np.stack(np.meshgrid(*[np.arange(-12, 12)] * 3, indexing="ij"), -1).reshape(-1, 3)
    centres = (ijk + 0.5) * _RES
    straddling = 0
    for _ in range(20):
        tight = _yawed(
            tuple(rng.uniform(-0.02, 0.02, 3)),
            tuple(rng.uniform(0.02, 0.08, 3)),
            float(rng.uniform(-180.0, 180.0)),
        )
        closed, _ = cell_closed_region(tight, lattice, max_half_extent_m=0.2)
        yaw = 2.0 * math.atan2(tight.pose.quat_xyzw[2], tight.pose.quat_xyzw[3])
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        d = centres - np.asarray(tight.pose.xyz)
        half = np.asarray(tight.half_extents)
        # Separating axes for two boxes with a shared z axis: the cube's x, y, the box's
        # x, y, and z. A cell meets the box iff no axis separates them.
        cube_axes, box_axes = np.eye(3), rot
        meets = np.ones(len(centres), dtype=bool)
        for axis in (*cube_axes[:2], *box_axes.T[:2], np.array([0.0, 0.0, 1.0])):
            reach = _RES / 2.0 * np.abs(cube_axes @ axis).sum() + np.abs(box_axes.T @ axis) @ half
            meets &= np.abs(d @ axis) <= reach
        bottom = tight.pose.xyz[2] - tight.half_extents[2]
        below = centres[:, 2] < bottom
        assert (meets & ~below).any()
        assert _in_region(centres[meets & ~below], closed).all()
        # Never down: a cell straddling the bottom face (the support side) stays outside.
        assert not _in_region(centres[meets & below], closed).any()
        straddling += int((meets & below).sum())
    assert straddling, "no sample straddled a bottom face"


# ── the region must hold the target's whole map component (Isaac i40/i43) ────────

_I4X_RES = 0.015
_I4X_CELLS = (36, 20, 14)
#: The Isaac i41/i42 potted-meat can as the octomap holds it (15 mm cells, origin
#: (0, -0.3, -0.6)): centres x 0.2175-0.3225, y -0.2325..-0.1575, z -0.4725..-0.4275 on a
#: table whose top layer (k=7, centres -0.4875) puts the measured support at z=-0.48.
_I4X_TARGET = {(i, j, k) for i in range(14, 22) for j in range(4, 10) for k in range(8, 12)}
_I4X_TABLE = {(i, j, 7) for i in range(4, 32) for j in range(20)}
_I4X_SUPPORT_Z = -0.48


def _i4x_lattice(
    *extra: set[tuple[int, int, int]], cells: tuple[int, int, int] = _I4X_CELLS
) -> VoxelLattice:
    occ = np.zeros(int(np.prod(cells)), dtype=np.uint8)
    for i, j, k in _I4X_TARGET.union(_I4X_TABLE, *extra):
        occ[i + cells[0] * (j + cells[1] * k)] = 1
    return VoxelLattice(_FRAME, (0.0, -0.3, -0.6), (0.0, 0.0, 0.0, 1.0), _I4X_RES, cells, occ)


def _touching(grid: VoxelLattice, tight: PlaceRegion) -> NDArray[np.float64]:
    closed, _ = cell_closed_region(tight, grid, max_half_extent_m=0.2)
    return occupied_touching_outside(grid, closed, support_z=_I4X_SUPPORT_Z)


#: The fits the Isaac leg accepted (centre, half-extents, yaw), from the trials' graph.log.
_I40_PARTIAL = ((0.242, -0.197, -0.439), (0.06, 0.043, 0.026), -90.8)
_I41_FULL = ((0.259, -0.2, -0.439), (0.068, 0.06, 0.026), 163.6)
_I42_FULL = ((0.257, -0.199, -0.439), (0.063, 0.064, 0.026), 76.0)


def test_isaac_i40_a_fit_of_the_near_part_leaves_the_far_edge_cells_outside() -> None:
    """Isaac i40/i43: the hovering hand hid the can's far side from the head camera; the
    fit (x 0.199-0.285) passed the half-footprint map cover and was armed, and the finger
    hull stopped on the can's own far-edge cells outside the kernel's region. Those cells
    are occupied, touch the region's cells and lie outside it: the fit is partial."""
    grid = _i4x_lattice()
    assert region_covers_occupied(grid, _yawed(*_I40_PARTIAL), min_fraction=0.5)[1]
    left = _touching(grid, _yawed(*_I40_PARTIAL))
    assert len(left), "the partial fit's region holds the whole can"
    assert (left[:, 0] > 0.29).all() and (left[:, 2] > _I4X_SUPPORT_Z + _I4X_RES).all()
    far_x = {round(float(x), 4) for x in left[:, 0]}
    assert 0.3075 in far_x, far_x


@pytest.mark.parametrize("fit", [_I41_FULL, _I42_FULL], ids=["i41", "i42"])
def test_isaac_i41_i42_a_fit_of_the_whole_can_holds_its_whole_component(
    fit: tuple[tuple[float, float, float], tuple[float, float, float], float],
) -> None:
    """The good views (both armed, i42 attached from the region): nothing of the can
    touches the kernel's region from outside — the table layer under it never counts."""
    assert len(_touching(_i4x_lattice(), _yawed(*fit))) == 0


def test_a_neighbour_touching_the_target_is_part_of_it_one_cell_apart_is_not() -> None:
    """The map cannot separate bodies that touch (26-connected above the support): a full
    fit of the can with a box against its +x side is refused; the same box one empty cell
    away is another object and the can's fit stands."""
    against = {(i, j, k) for i in range(22, 26) for j in range(4, 10) for k in range(8, 12)}
    apart = {(i + 1, j, k) for i, j, k in against}
    assert len(_touching(_i4x_lattice(against), _yawed(*_I41_FULL)))
    assert len(_touching(_i4x_lattice(apart), _yawed(*_I41_FULL))) == 0


# ── completing a partial fit from the map component (Isaac i45) ─────────────────

#: Isaac i45's hover fit (graph.log; the yaw is i40's, the same view): the hand hides the
#: can's far side from the head camera on every capture at the hover pose.
_I45_PARTIAL = ((0.242, -0.196, -0.439), (0.06, 0.043, 0.026), -90.8)
#: The search column the approach-armed leg vouches for at that hover: ``approach_box`` of
#: the right TCP ~12 cm over the can (half-extents capped at 0.20 m) reaching 0.15 m below.
_I45_COLUMN = _yawed((0.27, -0.195, -0.365), (0.2, 0.2, 0.275), 0.0)


def _complete(
    grid: VoxelLattice, fit: PlaceRegion, column: PlaceRegion = _I45_COLUMN
) -> tuple[PlaceRegion | None, int, str]:
    return map_completed_region(
        grid,
        fit,
        column,
        support_z=_I4X_SUPPORT_Z,
        max_half_extent_m=0.20,
        max_volume_m3=0.03,
    )


def _cell_centres_of(grid: VoxelLattice, cells: set[tuple[int, int, int]]) -> NDArray[np.float64]:
    ijk = np.asarray(sorted(cells), dtype=np.float64)
    return np.asarray((ijk + 0.5) * grid.resolution + np.asarray(grid.origin), np.float64)


def test_isaac_i45_a_fit_of_the_near_part_is_completed_to_the_whole_can() -> None:
    """i45: every hover capture saw the can's near part only (5-6 far cells outside, e.g.
    (0.2925, -0.2325, -0.4425)) and was refused ``partial_fit``. Completed from the map:
    the box holds the fit and every cell of the can's component, its bottom is the fit's
    own, the kernel's closure of it leaves nothing of the can outside and no table cell
    inside — and the good views' fits (i41/i42) track it within one voxel."""
    grid = _i4x_lattice()
    fit = _yawed(*_I45_PARTIAL)
    assert len(_touching(grid, fit)), "the i45 fit already holds the whole can"
    done, added, why = _complete(grid, fit)
    assert done is not None, why
    assert why == "" and added > 0
    # The component: the can's cells more than a voxel above the support. Its lowest layer
    # (k=8, centres half a voxel above the table top) stays outside, as for any fit.
    can = _cell_centres_of(grid, {c for c in _I4X_TARGET if c[2] > 8})
    assert _in_region(can, done).all(), "a can cell's centre is outside the completed box"
    assert not _in_region(_cell_centres_of(grid, {c for c in _I4X_TARGET if c[2] == 8}), done).any()
    assert region_within(fit, done, tol_m=1e-9), "the completed box dropped part of the fit"
    bottom = done.pose.xyz[2] - done.half_extents[2]
    assert bottom == pytest.approx(fit.pose.xyz[2] - fit.half_extents[2], abs=1e-12)
    assert done.pose.quat_xyzw == fit.pose.quat_xyzw, "completed in another yaw"
    assert len(_touching(grid, done)) == 0
    closed, _ = cell_closed_region(done, grid, max_half_extent_m=0.2)
    assert not _in_region(_cell_centres_of(grid, _I4X_TABLE), closed).any(), "table exempt"
    for full in (_I41_FULL, _I42_FULL):
        tol = {"max_centroid_shift_m": _I4X_RES, "extents_tol_m": _I4X_RES}
        assert track_region(_yawed(*full), done, **tol)
    # A fit of the whole can holds its component already: nothing to add.
    whole, none_added, _ = _complete(grid, _yawed(*_I41_FULL))
    assert whole is not None and none_added == 0


def test_a_neighbour_within_one_voxel_merges_into_the_completed_target() -> None:
    """The map cannot separate bodies 26-connected above the support: a box against the
    can's +x side becomes part of the completed region (the Safety-WG residual,
    HZ-0115-30), one empty cell away it does not. Merged into a blob over the
    declaration's caps — a long box beside the can — the fit is not completed."""
    against = {(i, j, k) for i in range(22, 26) for j in range(4, 10) for k in range(8, 12)}
    apart = {(i + 1, j, k) for i, j, k in against}
    fit = _yawed(*_I45_PARTIAL)

    merged, _, _ = _complete(_i4x_lattice(against), fit)
    assert merged is not None
    above = {c for c in against if c[2] > 8}  # the layer at the support never counts
    assert _in_region(_cell_centres_of(_i4x_lattice(), above), merged).all()

    alone, _, _ = _complete(_i4x_lattice(apart), fit)
    assert alone is not None
    assert not _in_region(_cell_centres_of(_i4x_lattice(), apart), alone).any()

    long_box = {(i, j, k) for i in range(22, 50) for j in range(4, 10) for k in range(8, 12)}
    wide_column = _yawed((0.45, -0.195, -0.365), (0.42, 0.2, 0.275), 0.0)
    over, _, why = _complete(_i4x_lattice(long_box, cells=(60, 20, 14)), fit, wide_column)
    assert over is None and "over the 0.2 m / 0.03 m^3 caps" in why


def test_a_component_touching_the_search_column_edge_is_not_completed() -> None:
    """The leg vouches for its search column only: a component that reaches the column's
    edge may continue past it, so it is not completed (``partial_fit`` in the leg)."""
    cut = _yawed((0.15, -0.195, -0.365), (0.16, 0.2, 0.275), 0.0)  # x <= 0.31: cuts the can
    done, _, why = _complete(_i4x_lattice(), _yawed(*_I45_PARTIAL), cut)
    assert done is None and "touches the search column's edge" in why


def test_another_body_inside_the_completed_box_is_not_completed() -> None:
    """An L-shaped target's box takes in its bounding corner: a separate body standing
    there (two empty cells from either arm, so not of the component) would be exempted
    with it — refused, never merged."""
    arms = {(i, 1, 1) for i in range(1, 5)} | {(4, j, 1) for j in range(2, 5)}
    corner = {(1, 4, 1)}

    def grid(cells: set[tuple[int, int, int]]) -> VoxelLattice:
        occ = np.zeros(6 * 6 * 3, dtype=np.uint8)
        for i, j, k in cells:
            occ[i + 6 * (j + 6 * k)] = 1
        return VoxelLattice(_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (6, 6, 3), occ)

    fit = _yawed((0.2, 0.15, 0.15), (0.09, 0.05, 0.05), 0.0)  # the camera saw (1,1)-(2,1)
    column = _yawed((0.3, 0.3, 0.15), (0.3, 0.3, 0.15), 0.0)
    kwargs = {"support_z": 0.0, "max_half_extent_m": 0.2, "max_volume_m3": 0.03}
    done, added, _ = map_completed_region(grid(arms), fit, column, **kwargs)
    assert done is not None and added == 5
    refused, _, why = map_completed_region(grid(arms | corner), fit, column, **kwargs)
    assert refused is None and "another body inside the completed region" in why


def test_another_body_in_a_yawed_completed_boxs_corner_is_refused() -> None:
    """The completed box is in the fit's yaw, the search column in the base frame's: at 45°
    the box's corners reach past the column (and the seed box) by more than the cells the
    flood looks at. A separate body standing in such a corner — two voxels past the column
    grown by two, outside the seed box grown by two — lies inside the kernel's closure, so
    it must be refused like any foreign cell, not exempted with the target."""
    res, n = 0.02, 30
    bar = {(i, 15, 1) for i in range(8, 22)}  # x -0.13..0.13 at y 0.01, z 0.03
    corner = {(14, 21, 1)}  # (-0.01, 0.13, 0.03): six cells from the bar

    def grid(cells: set[tuple[int, int, int]]) -> VoxelLattice:
        occ = np.zeros(n * n * 4, dtype=np.uint8)
        for i, j, k in cells:
            occ[i + n * (j + n * k)] = 1
        return VoxelLattice(_FRAME, (-0.3, -0.3, 0.0), (0.0, 0.0, 0.0, 1.0), res, (n, n, 4), occ)

    fit = _yawed((0.0, 0.01, 0.04), (0.015, 0.015, 0.02), 45.0)  # the camera saw the middle
    column = _yawed((0.0, 0.01, 0.04), (0.17, 0.05, 0.04), 0.0)
    kwargs = {"support_z": 0.0, "max_half_extent_m": 0.2, "max_volume_m3": 0.03}
    done, added, why = map_completed_region(grid(bar), fit, column, **kwargs)
    assert done is not None and added > 0, why
    closed, _ = cell_closed_region(done, grid(bar), max_half_extent_m=0.2)
    centre = np.asarray([[-0.01, 0.13, 0.03]])
    assert _in_region(centre, closed).all(), "the corner cell is not in the kernel's region"
    refused, _, why = map_completed_region(grid(bar | corner), fit, column, **kwargs)
    assert refused is None and "another body inside the completed region" in why
