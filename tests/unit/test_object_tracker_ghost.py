"""A box over two tracked oranges never confirms as a third (#418 PickOrange ghost trk:5, OT-13).

Real grid: ``fixtures/pick_orange_ghost_grid.npz``, the PickOrange world voxels (15 mm, the
voxel bridge's 2 Hz debug cloud at sim 3.52 s) and the tracked boxes at sim 3.47 s of trial
r6po_dbg4 (Spark, 2026-10-17). The locator boxed both orange_1 and orange_2 as one "orange"
(score 0.171), SAM's mask took both, and the containing fit (trk:5, 21 x 13 x 9 cm) confirmed
at once: its two bodies lay inside the CONFIRMED trk:2 / trk:4 boxes, which the one-body check
excuses as "that object". trk:5 then held cells by the hand at the grasp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_core import FitKind, ObjectQuery, ObjectRole, PlaceRegion, Pose6D, TrackStatus
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker import ObjectTracker, TrackerParams, surface_cells

_FIX = np.load(Path(__file__).parent / "fixtures" / "pick_orange_ghost_grid.npz")
# orange_1 / orange_2's side only (y < -0.235). Nearer the arm the grid holds orange_3 and a
# low row of cells under the descending hand, which the live tracker's one-body check left to
# orange_3's box and the hand's own space; this rig models neither.
_VOX: NDArray[np.float64] = np.asarray(_FIX["voxels"], dtype=np.float64)
_VOX = _VOX[_VOX[:, 1] < -0.235]
_RES = 0.015
_SUPPORT_Z = 0.015  # the tracker's measured counter top (tracked.jsonl)
_FRAME = "base"


def _grid() -> VoxelLattice:
    pts = _VOX
    # The cloud's points are cell centres: centre the lattice on them.
    lo = pts.min(axis=0) - 1.5 * _RES
    ijk = np.rint((pts - lo) / _RES - 0.5).astype(int)
    dims = ijk.max(axis=0) + 2
    occ = np.zeros(int(np.prod(dims)), dtype=np.uint8)
    occ[ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])] = 1
    return VoxelLattice(
        _FRAME,
        tuple(lo.tolist()),
        (0.0, 0.0, 0.0, 1.0),
        _RES,
        (int(dims[0]), int(dims[1]), int(dims[2])),
        occ,
    )


def _box(name: str) -> PlaceRegion:
    x, y, z, hx, hy, hz = (float(v) for v in _FIX[name])
    return PlaceRegion(
        frame_id=_FRAME,
        half_extents=(hx, hy, hz),
        pose=Pose6D(xyz=(x, y, z), quat_xyzw=(0, 0, 0, 1), frame_id=_FRAME),
    )


def _cells_in(grid: VoxelLattice, box: PlaceRegion) -> frozenset[tuple[int, int, int]]:
    """``E`` for a fit filling ``box``: the occupied voxel centres in it above the support."""
    pts = _VOX
    lo = np.asarray(box.pose.xyz) - np.asarray(box.half_extents)
    hi = np.asarray(box.pose.xyz) + np.asarray(box.half_extents)
    inside = pts[np.all((pts >= lo) & (pts <= hi), axis=1) & (pts[:, 2] > _SUPPORT_Z + _RES)]
    return surface_cells(grid, inside)


def _tracker(low_score_confirm_grids: int = 3) -> ObjectTracker:
    tr = ObjectTracker(
        _FRAME, params=TrackerParams(low_score_confirm_grids=low_score_confirm_grids)
    )
    q = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)
    tr.open_query("g", "q0", q, "deploy@0", stamp_ns=0)
    return tr


def _mint(tr: ObjectTracker, name: str, grid: VoxelLattice, *, confirmed: bool) -> str:
    box = _box(name)
    cells = _cells_in(grid, box)
    t = tr.store.mint(
        label="orange", role=ObjectRole.TARGET, declaration_key="deploy@0", box=box, tight=(),
        fit_kind=FitKind.SILHOUETTE_CONE, support_z=_SUPPORT_Z, support_ref="map",
        floor_z=_SUPPORT_Z + _RES, visible_cells=len(cells), occupied_cells=0,
        last_fit_stamp_ns=1, last_confirmed_stamp_ns=0,
    )  # fmt: skip
    if confirmed:
        tr.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
    tr._cells[t.object_id] = cells
    return t.object_id


def _status(tr: ObjectTracker, oid: str) -> TrackStatus:
    t = tr.store.get(oid)
    assert t is not None
    return t.status


def test_a_box_over_two_tracked_oranges_never_confirms() -> None:
    tr, grid = _tracker(), _grid()
    _mint(tr, "box_trk2", grid, confirmed=True)  # orange_2
    _mint(tr, "box_trk4", grid, confirmed=True)  # orange_1
    ghost = _mint(tr, "box_trk5", grid, confirmed=False)
    for stamp in (10, 20, 30, 40):
        t = tr.store.get(ghost)
        assert t is not None
        tr._confirm(t, grid, stamp, ())
    assert _status(tr, ghost) is TrackStatus.TENTATIVE
    assert tr.refusal_counts[("orange@" + ghost, "cover_already_tracked")] == 4


def test_an_orange_of_its_own_still_confirms_beside_another() -> None:
    """The control: orange_2's cells lie outside orange_1's box, so it confirms at once."""
    tr, grid = _tracker(), _grid()
    _mint(tr, "box_trk4", grid, confirmed=True)
    own = _mint(tr, "box_trk2", grid, confirmed=False)
    t = tr.store.get(own)
    assert t is not None
    tr._confirm(t, grid, 10, ())
    assert _status(tr, own) is TrackStatus.CONFIRMED


def test_a_low_score_prompt_needs_consecutive_confirming_grids() -> None:
    tr, grid = _tracker(low_score_confirm_grids=3), _grid()
    oid = _mint(tr, "box_trk2", grid, confirmed=False)
    tr._note_low_score(oid, 0.17)
    for stamp, want in ((10, TrackStatus.TENTATIVE), (10, TrackStatus.TENTATIVE),
                        (20, TrackStatus.TENTATIVE), (30, TrackStatus.CONFIRMED)):  # fmt: skip
        t = tr.store.get(oid)
        assert t is not None
        tr._confirm(t, grid, stamp, ())
        assert _status(tr, oid) is want, stamp
    # A refuting grid resets the run.
    tr2 = _tracker(low_score_confirm_grids=2)
    oid2 = _mint(tr2, "box_trk2", grid, confirmed=False)
    tr2._note_low_score(oid2, 0.17)
    empty = VoxelLattice(grid.frame_id, grid.origin, grid.orientation_xyzw, grid.resolution,
                         grid.size, np.zeros_like(grid.occupancy))  # fmt: skip
    for g, stamp in ((grid, 10), (empty, 20), (grid, 30)):
        t = tr2.store.get(oid2)
        assert t is not None
        tr2._confirm(t, g, stamp, ())
    assert _status(tr2, oid2) is TrackStatus.TENTATIVE


def test_a_target_lying_in_its_destinations_box_confirms() -> None:
    """Spark R278 (2026-10-23): the banana dropped into the bowl never re-confirmed: its cells
    lay inside the bowl's containing box, read as another object's. A destination's box holds
    its cavity; the target in it is the goal state, not a ghost (the two-oranges ghost above
    still never confirms)."""
    tr, grid = _tracker(), _grid()
    plate = _mint(tr, "box_trk2", grid, confirmed=True)
    track = tr.store.get(plate)
    assert track is not None
    tr.store.replace(track.model_copy(update={"label": "plate", "role": ObjectRole.DESTINATION}))
    inside = _mint(tr, "box_trk2", grid, confirmed=False)
    for stamp in (10, 20):
        t = tr.store.get(inside)
        assert t is not None
        tr._confirm(t, grid, stamp, ())
    assert _status(tr, inside) is TrackStatus.CONFIRMED
