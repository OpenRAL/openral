"""Size-aware map confirmation and map_split (issue #418 steps 4-5), on real camera renders."""

from __future__ import annotations

import math

import numpy as np
from openral_core import ObjectQuery, ObjectRole, PlaceRegion, Pose6D, TrackStatus
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_world_state.object_tracker import ObjectTracker, cover_verdict, surface_cells

from tests.unit._tracker_scenes import (
    RES,
    Solid,
    cylinder,
    grid_from_depth,
    render,
    so101_top,
    sphere,
    step,
)

_RIG = so101_top()
_R = 0.024  # the ∅4.8 cm PickOrange orange
_ORANGE = sphere((*_RIG.centre_xy, _RIG.support_z + _R), _R)
# A can: its containing box reaches cells two clear of what the camera sees of it (a 4.8 cm
# orange's box does not — every cell in it touches the orange, so a body there merges).
_CAN = cylinder(_RIG.centre_xy, 0.035, _RIG.support_z, _RIG.support_z + 0.10)
_QUERY = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)


def _confirmed_tracker(solid: Solid = _ORANGE) -> tuple[ObjectTracker, float]:
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    out, wall = step(tr, _RIG, [solid], stamp_ns=0, wall_s=0.0)
    assert [o.status for o in out.tracked.objects] == [TrackStatus.CONFIRMED]
    return tr, wall


def test_size_aware_cover_confirms_a_48mm_orange_at_15mm_cells() -> None:
    tr, _ = _confirmed_tracker()
    track = tr.store.all()[0]
    e, hit = track.visible_cells, track.occupied_cells
    assert 2 <= e < 40  # cells the camera's surface points fall in, not a padded footprint
    assert hit >= max(2, math.ceil(0.5 * e))
    # The same rule, straight on the grid: the expected cells are what the map must hold.
    depth, labels = render([_ORANGE], _RIG)
    grid = grid_from_depth(depth, _RIG)
    rows, cols = np.nonzero(labels == 1)
    z = depth[rows, cols]
    k = _RIG.k
    cam = np.stack(((cols + 0.5 - k.cx) / k.fx * z, (rows + 0.5 - k.cy) / k.fy * z, z), axis=1)
    cells = surface_cells(grid, cam @ _RIG.t[:3, :3].T + _RIG.t[:3, 3])
    verdict = cover_verdict(grid, cells, track.box, support_z=track.support_z)
    assert verdict.kind == "" and verdict.visible == len(cells)
    # An empty map at the same cells refutes it.
    empty = grid_from_depth(render([], _RIG)[0], _RIG)
    assert cover_verdict(empty, cells, track.box, support_z=track.support_z).kind == (
        "map_disagrees"
    )


def _foreign_cell(tr: ObjectTracker) -> tuple[int, int, int]:
    """A cell deep inside the can's box, not touching the can (a second body)."""
    track = tr.store.all()[0]
    depth, _ = render([_CAN], _RIG)
    grid = grid_from_depth(depth, _RIG)
    occ = {tuple(c) for c in np.floor((grid.occupied_centers() - grid.origin) / RES).astype(int)}
    t = homogeneous_from_quat_xyz(track.box.pose.xyz, track.box.pose.quat_xyzw)
    pad = 0.5 * RES  # map_confirms' sliver band: a body must reach inside the box less it
    half = np.asarray(track.box.half_extents) - (pad, pad, pad / 2)
    for i in range(40):
        for j in range(40):
            for kk in range(4, 20):
                centre = np.asarray(grid.origin) + (np.array((i, j, kk)) + 0.5) * RES
                local = (centre - t[:3, 3]) @ t[:3, :3] + (0.0, 0.0, pad / 2)
                near = any(
                    (i + a, j + b, kk + c) in occ
                    for a in (-1, 0, 1)
                    for b in (-1, 0, 1)
                    for c in (-1, 0, 1)
                )
                if np.all(np.abs(local) <= half) and not near:
                    return i, j, kk
    raise AssertionError("no free cell inside the can's box")


def test_map_split_on_a_foreign_body_makes_the_track_ineligible() -> None:
    tr, wall = _confirmed_tracker(_CAN)
    cell = _foreign_cell(tr)
    out, _ = step(tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=[cell])
    assert out.tracked.objects[0].status is TrackStatus.SPLIT


def test_map_split_inside_the_hand_is_the_hands_own_and_ages_out() -> None:
    tr, wall = _confirmed_tracker(_CAN)
    before = tr.store.all()[0].last_confirmed_stamp_ns
    cell = _foreign_cell(tr)
    centre = tuple(
        float(o + (c + 0.5) * RES)
        for o, c in zip(grid_from_depth(render([], _RIG)[0], _RIG).origin, cell, strict=True)
    )
    hand = PlaceRegion(
        frame_id=_RIG.frame,
        half_extents=(RES, RES, RES),
        pose=Pose6D(xyz=centre, quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
    )
    out, _ = step(
        tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=[cell], hands=(hand,)
    )
    track = out.tracked.objects[0]
    assert track.status is TrackStatus.CONFIRMED
    assert track.last_confirmed_stamp_ns == before  # not re-stamped: it ages out
