"""Declared static geometry is no foreign body for the one-body (SPLIT) check (#418 restock).

Real data: ``fixtures/restock_target_wall.npz`` -- the OpenArm restock deploy sim (r8rs02_on2,
2026-10-17): the 15 mm world grid at sim 8.82 s, the target carton's tracked box (trk:3, half-x
9.1 cm for a 4.1 cm carton: the head camera's fit overshoots toward the source tote's left wall)
and the declared static cells in the crop. Two components of 12 and 7 cells lie wholly within one
cell of declared cells -- the tote wall as the camera sees it -- and made ``map_split``. The fit
keeps every measured point; only the SPLIT test excuses a body that is declared geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_core import PlaceRegion, Pose6D
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker import ObjectTracker, cover_verdict, static_cell_keys

_F = np.load(Path(__file__).parent / "fixtures" / "restock_target_wall.npz")
_RES = 0.015


def _grid(extra: NDArray[np.float64] | None = None) -> VoxelLattice:
    pts = np.vstack(
        [_F["voxels"].astype(float), _F["static_cells"].astype(float)]
        + ([extra] if extra is not None else [])
    )
    lo = np.floor(pts.min(axis=0) / _RES) * _RES - 2 * _RES
    ijk = np.floor(
        (np.vstack([_F["voxels"].astype(float)] + ([extra] if extra is not None else [])) - lo)
        / _RES
    ).astype(int)
    dims = np.floor((pts - lo) / _RES).astype(int).max(axis=0) + 3
    occ = np.zeros(int(np.prod(dims)), dtype=np.uint8)
    occ[ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])] = 1
    return VoxelLattice("base", tuple(lo.tolist()), (0.0, 0.0, 0.0, 1.0), _RES,
                        (int(dims[0]), int(dims[1]), int(dims[2])), occ)  # fmt: skip


def _box() -> PlaceRegion:
    b = [float(v) for v in _F["box"]]
    return PlaceRegion(frame_id="base", half_extents=(b[7], b[8], b[9]),
                       pose=Pose6D(xyz=(b[0], b[1], b[2]),
                                   quat_xyzw=(b[3], b[4], b[5], b[6]),
                                   frame_id="base"))  # fmt: skip


def _verdict(grid: VoxelLattice, *, static: bool) -> str:
    box = _box()
    occ = grid.occupied_centers()
    inside = occ[
        np.all(np.abs(occ - np.asarray(box.pose.xyz)) <= np.asarray(box.half_extents), axis=1)
    ]
    local = (inside - np.asarray(grid.origin)) / _RES
    cells = frozenset(map(tuple, np.floor(local).astype(int).tolist()))
    keys = (
        static_cell_keys(grid, _F["static_cells"].astype(float), pad_cells=1)
        if static
        else frozenset()
    )
    return cover_verdict(grid, cells, box, support_z=float(_F["support_z"]), static=keys).kind


def test_the_tote_wall_inside_an_overshooting_box_splits_it_without_the_declaration() -> None:
    assert _verdict(_grid(), static=False) == "map_split"


def test_declared_static_geometry_is_not_a_foreign_body() -> None:
    assert _verdict(_grid(), static=True) == ""


def test_a_real_second_body_in_the_box_still_splits() -> None:
    """A 4.5 x 4.5 x 1.5 cm block floating one empty layer above the carton (cells 15 mm apart,
    lattice-aligned, nowhere near a declared cell) is a foreign body even with the declaration:
    the excuse is only for components wholly inside declared geometry."""
    block = np.array(
        [[x, y, -0.4575] for x in (0.2475, 0.2625, 0.2775) for y in (0.0675, 0.0825, 0.0975)]
    )
    assert _verdict(_grid(block), static=False) == "map_split"
    assert _verdict(_grid(block), static=True) == "map_split"


def test_the_fit_is_untouched_by_the_declaration() -> None:
    """``static`` reaches only the one-body check: with a box that really lacks cover the verdict
    is still ``map_disagrees`` (cover counts every occupied cell)."""
    grid = _grid()
    far = PlaceRegion(frame_id="base", half_extents=(0.03, 0.03, 0.03),
                      pose=Pose6D(xyz=(0.0, 0.9, 0.0), quat_xyzw=(0, 0, 0, 1),
                                  frame_id="base"))  # fmt: skip
    keys = static_cell_keys(grid, _F["static_cells"].astype(float), pad_cells=1)
    v = cover_verdict(grid, frozenset({(1, 1, 1), (1, 1, 2)}), far, support_z=0.0, static=keys)
    assert v.kind == "map_disagrees"


def _tracker_with_the_overshooting_target(declare: bool) -> tuple[ObjectTracker, str, VoxelLattice]:
    from openral_core import FitKind, ObjectRole, TrackStatus

    grid = _grid()
    tr = ObjectTracker("base")
    box = _box()
    t = tr.store.mint(
        label="box", role=ObjectRole.TARGET, declaration_key="deploy@0", box=box, tight=(),
        fit_kind=FitKind.SILHOUETTE_CONE, support_z=float(_F["support_z"]), support_ref="map",
        floor_z=float(_F["support_z"]) + _RES, visible_cells=33, occupied_cells=33,
        last_fit_stamp_ns=1, last_confirmed_stamp_ns=1,
    )  # fmt: skip
    tr.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
    occ = grid.occupied_centers()
    inside = occ[
        np.all(np.abs(occ - np.asarray(box.pose.xyz)) <= np.asarray(box.half_extents), axis=1)
    ]
    tr._cells[t.object_id] = frozenset(
        map(tuple, np.floor((inside - np.asarray(grid.origin)) / _RES).astype(int).tolist())
    )
    if declare:
        tr.set_static_geometry(_F["static_cells"].astype(float))
    return tr, t.object_id, grid


def test_a_confirmed_target_is_not_split_by_the_declared_tote_wall() -> None:
    from openral_core import TrackStatus

    for declare, want in ((False, TrackStatus.SPLIT), (True, TrackStatus.CONFIRMED)):
        tr, oid, grid = _tracker_with_the_overshooting_target(declare)
        t = tr.store.get(oid)
        assert t is not None
        tr._confirm(t, grid, 10, ())
        after = tr.store.get(oid)
        assert after is not None
        assert after.status is want, (declare, after.status)
