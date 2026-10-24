"""Size-aware map confirmation and map_split (issue #418 steps 4-5), on real camera renders."""

from __future__ import annotations

import math

import numpy as np
from openral_core import ObjectQuery, ObjectRole, PlaceRegion, Pose6D, TrackStatus
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker import (
    CameraFrame,
    ObjectTracker,
    TrackerResult,
    cover_verdict,
    surface_cells,
)
from openral_world_state.object_tracker.cover import MIN_SPLIT_CELLS, box_cells

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


def _confirmed_tracker(
    solid: Solid = _ORANGE, query: ObjectQuery = _QUERY
) -> tuple[ObjectTracker, float]:
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", query, "obj:orange_1@0", stamp_ns=0)
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


def _foreign_body(tr: ObjectTracker, n: int = MIN_SPLIT_CELLS) -> list[tuple[int, int, int]]:
    """``n`` stacked cells deep inside the can's box, not touching the can (a second body)."""
    track = tr.store.all()[0]
    depth, _ = render([_CAN], _RIG)
    grid = grid_from_depth(depth, _RIG)
    occ = {tuple(c) for c in np.floor((grid.occupied_centers() - grid.origin) / RES).astype(int)}
    t = homogeneous_from_quat_xyz(track.box.pose.xyz, track.box.pose.quat_xyzw)
    pad = 0.5 * RES  # map_confirms' sliver band: a body must reach inside the box less it
    half = np.asarray(track.box.half_extents) - (pad, pad, pad / 2)

    def free(i: int, j: int, kk: int) -> bool:
        centre = np.asarray(grid.origin) + (np.array((i, j, kk)) + 0.5) * RES
        local = (centre - t[:3, 3]) @ t[:3, :3] + (0.0, 0.0, pad / 2)
        near = any(
            (i + a, j + b, kk + c) in occ
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
        )
        return bool(np.all(np.abs(local) <= half)) and not near

    for i in range(40):
        for j in range(40):
            for kk in range(4, 20):
                if all(free(i, j, kk + d) for d in range(n)):
                    return [(i, j, kk + d) for d in range(n)]
    raise AssertionError(f"no {n} free stacked cells inside the can's box")


def test_map_split_on_a_foreign_body_makes_the_track_ineligible() -> None:
    tr, wall = _confirmed_tracker(_CAN)
    body = _foreign_body(tr)
    out, _ = step(tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body)
    assert out.tracked.objects[0].status is TrackStatus.SPLIT


def test_a_fragment_smaller_than_a_body_does_not_split_the_track() -> None:
    """Isaac R28/R31: 1-2 cells of the banana itself, cut off from its main body by the
    hand's occlusion or its own curve, made the target SPLIT mid-grasp. Under
    ``MIN_SPLIT_CELLS`` a second body is a fragment, and the track stays confirmed."""
    tr, wall = _confirmed_tracker(_CAN)
    body = _foreign_body(tr, 2)  # R31's fragment
    out, _ = step(tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body)
    assert out.tracked.objects[0].status is TrackStatus.CONFIRMED


def test_map_split_inside_the_hand_is_the_hands_own() -> None:
    """A second body inside the hand's own space is the hand's (leaked points), never a split.
    A context object then ages out (not re-stamped); at a TARGET the hand is at, the visible
    cover still confirms it (2026-10-21 decision: the piece by the hand is the object's)."""
    for role, restamped in ((ObjectRole.CONTEXT, False), (ObjectRole.TARGET, True)):
        query = ObjectQuery(label="orange", count="all", role=role)
        tr, wall = _confirmed_tracker(_CAN, query)
        before = tr.store.all()[0].last_confirmed_stamp_ns
        body = _foreign_body(tr)
        centre = tuple(
            float(o + (c + 0.5) * RES)
            for o, c in zip(grid_from_depth(render([], _RIG)[0], _RIG).origin, body[1], strict=True)
        )
        hand = PlaceRegion(
            frame_id=_RIG.frame,
            half_extents=(RES, RES, 2 * RES),
            pose=Pose6D(xyz=centre, quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
        )
        out, _ = step(
            tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body, hands=(hand,)
        )
        track = out.tracked.objects[0]
        assert track.status is TrackStatus.CONFIRMED, role
        assert (track.last_confirmed_stamp_ns == 100_000_000) is restamped, role
        assert restamped or track.last_confirmed_stamp_ns == before  # context: it ages out


def _hand_on_top(tr: ObjectTracker) -> PlaceRegion:
    """A finger-sized hand box resting on the target's box top, outside it (contact)."""
    box = tr.store.all()[0].box
    x, y, z = box.pose.xyz
    return PlaceRegion(
        frame_id=_RIG.frame,
        half_extents=(RES, RES, 0.5 * RES),
        pose=Pose6D(xyz=(x, y, z + box.half_extents[2] + 0.5 * RES), quat_xyzw=(0, 0, 0, 1),
                    frame_id=_RIG.frame),
    )  # fmt: skip


def test_a_hand_at_the_target_puts_its_refresh_first() -> None:
    """2026-10-22, Spark R118 / R122: sharing the refresh slots, the banana's fit was over
    max_age_s at the grasp; a hand at a TARGET orders its refresh first among refreshes."""
    tr, _ = _confirmed_tracker()
    (track,) = tr.store.all()
    depth, _ = render([_ORANGE], _RIG)
    frame = CameraFrame(_RIG.name, 5, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG)

    def served(hands: tuple[PlaceRegion, ...]) -> int:
        (req,) = [r for r in tr._candidates(frame, grid, hands) if r.object_id == track.object_id]
        return req.served_ns

    assert served(()) >= 0
    assert served((_hand_on_top(tr),)) == -1


def test_a_bound_track_is_prompted_where_its_payload_rides() -> None:
    """2026-10-22, Spark R150-R153: prompted at its grasp-site box the segmenter masked the
    table; a BOUND track's request goes where the session says the hand carries it."""
    tr, _ = _confirmed_tracker()
    (track,) = tr.store.all()
    tr.store.replace(
        track.model_copy(
            update={
                "status": TrackStatus.BOUND,
                "bound_hand": "gripper",
                "attach_link": "jaw",
                "pose_in_link": Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="jaw"),
            }
        )
    )
    depth, _ = render([_ORANGE], _RIG)
    frame = CameraFrame(_RIG.name, 5, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG)
    x, y, z = track.box.pose.xyz
    carried = track.box.model_copy(
        update={"pose": track.box.pose.model_copy(update={"xyz": (x + 0.06, y, z + 0.05)})}
    )

    def prompt() -> tuple[float, float]:
        (req,) = [r for r in tr._candidates(frame, grid) if r.object_id == track.object_id]
        assert req.point_uv is not None
        return req.point_uv

    at_box = prompt()
    tr.bound_prompts = {track.object_id: carried}
    at_hand = prompt()
    assert np.hypot(at_hand[0] - at_box[0], at_hand[1] - at_box[1]) > 5.0  # pixels


def test_a_split_under_contact_confirms_and_splits_once_the_hand_leaves() -> None:
    """#418 decisions (2026-10-21, Isaac R33/R38/R59): fingers at the target fragment its map
    (they bisect it; a thin banana's own pieces part on the lowest layer). While a hand is at
    the TARGET a split inside its box does not block a confirmation the visible cover earns;
    with the hand gone the same body splits it again (fail-closed). The DESTINATION (a bowl
    the hand places into, R63) is held the same way; a CONTEXT neighbour never is."""
    for role in ObjectRole:
        query = ObjectQuery(label="orange", count="all", role=role)
        tr, wall = _confirmed_tracker(_CAN, query)
        body, hand = _foreign_body(tr), _hand_on_top(tr)
        out, _ = step(
            tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body, hands=(hand,)
        )
        track = out.tracked.objects[0]
        if role is ObjectRole.CONTEXT:
            assert track.status is TrackStatus.SPLIT, role
            continue
        assert track.status is TrackStatus.CONFIRMED, role
        assert track.last_confirmed_stamp_ns == 100_000_000, role
        far = hand.model_copy(
            update={"pose": hand.pose.model_copy(update={"xyz": (5.0, 5.0, 5.0)})}
        )
        out, _ = step(
            tr, _RIG, [_CAN], stamp_ns=200_000_000, wall_s=wall + 2, extra=body, hands=(far,)
        )
        assert out.tracked.objects[0].status is TrackStatus.SPLIT, role  # hand gone: fail-closed


def test_a_piece_beside_the_hand_at_the_target_confirms_it() -> None:
    """The fingers cut the target's map in two (Isaac R33/R38): at the target, a second body
    within a cell of the hand is the piece they cut off, so the track confirms and re-stamps.
    Far from the hand the same body still splits it (the test above)."""
    tr, wall = _confirmed_tracker(_CAN)
    body = _foreign_body(tr)
    origin = grid_from_depth(render([], _RIG)[0], _RIG).origin
    x, y, z = (float(o + (c + 0.5) * RES) for o, c in zip(origin, body[1], strict=True))
    hand = PlaceRegion(
        frame_id=_RIG.frame,
        half_extents=(0.5 * RES, 0.5 * RES, 1.5 * RES),
        pose=Pose6D(xyz=(x + 1.2 * RES, y, z), quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
    )  # beside the body, not around it: only the grown hand holds it
    out, _ = step(
        tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body, hands=(hand,)
    )
    track = out.tracked.objects[0]
    assert track.status is TrackStatus.CONFIRMED
    assert track.last_confirmed_stamp_ns == 100_000_000


def test_a_hand_at_the_target_holds_a_disagreeing_map_and_lets_it_age_out() -> None:
    """The same hold for ``map_disagrees`` (a push leaves the target's old cells empty):
    with the hand at it the track is never LOST, only unconfirmed; without, it is. The same
    for a DESTINATION; a CONTEXT neighbour the hand meets is lost straight away."""
    n = ObjectTracker(_RIG.frame).params.disagree_grids
    for role in ObjectRole:
        tr, wall = _confirmed_tracker(_CAN, ObjectQuery(label="orange", count="all", role=role))
        before = tr.store.all()[0].last_confirmed_stamp_ns
        hand = _hand_on_top(tr)
        for k in range(n + 1):
            out, wall = step(tr, _RIG, [], stamp_ns=(k + 1) * 100_000_000, wall_s=wall + 1,
                             hands=(hand,))  # fmt: skip
        track = out.tracked.objects[0]
        if role is ObjectRole.CONTEXT:
            assert track.status is TrackStatus.LOST, role
            continue
        assert track.status is TrackStatus.CONFIRMED, role
        assert track.last_confirmed_stamp_ns == before, role
        for k in range(n + 1):
            out, wall = step(tr, _RIG, [], stamp_ns=(n + k + 2) * 100_000_000, wall_s=wall + 1)
        assert out.tracked.objects[0].status is TrackStatus.LOST, role


def test_a_push_inside_the_box_re_anchors_a_touched_target_on_where_the_map_holds_it() -> None:
    """#418 decision (2026-10-21; Isaac R68: a banana pushed 66 mm inside its box held 12 of
    its 26 expected cells, the contact hold aged it out mid-grasp). With the hand at it, the
    expected cells move to what the map holds inside the box, and the track re-stamps; with
    no hand the same push is no confirmation, and a push out of the box is none either."""

    def pushed(dx: float, *, hand: bool) -> int:
        tr, wall = _confirmed_tracker(_CAN)
        x, y = _RIG.centre_xy
        depth, _ = render(
            [cylinder((x + dx, y), 0.035, _RIG.support_z, _RIG.support_z + 0.1)], _RIG
        )
        frame = CameraFrame(
            _RIG.name, 100_000_000, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8)
        )
        out = tr.feed(
            frame,
            grid=grid_from_depth(depth, _RIG),
            grid_stamp_ns=100_000_000,
            wall_s=wall + 1,
            hand_spaces=(_hand_on_top(tr),) if hand else (),
        )
        return out.tracked.objects[0].last_confirmed_stamp_ns  # the grid alone: no refit

    assert pushed(0.03, hand=True) == 100_000_000  # 29 of 80 expected cells left: re-anchored
    assert pushed(0.03, hand=False) != 100_000_000
    assert pushed(0.15, hand=True) != 100_000_000  # out of the box: nothing to re-anchor on


def _box_layer(tr: ObjectTracker, k: int) -> list[tuple[int, int, int]]:
    """Every cell of lattice layer ``k`` centred in the track's box footprint."""
    box = tr.store.all()[0].box
    origin = np.asarray(grid_from_depth(render([], _RIG)[0], _RIG).origin)
    (x, y, _), (hx, hy, _) = box.pose.xyz, box.half_extents
    return [
        (i, j, k)
        for i in range(200)
        for j in range(200)
        if abs(origin[0] + (i + 0.5) * RES - x) <= hx and abs(origin[1] + (j + 0.5) * RES - y) <= hy
    ]


def test_box_cells_leave_out_the_support_layer_on_the_isaac_lattice() -> None:
    """Isaac R79's lattice (origin z -0.705 m, 15 mm cells) puts a cell boundary on the table
    top at z = 0: the top's own layer is centred a rounding error above half a cell over the
    support, and must still be left out."""
    nx, ny, nz = 4, 4, 50
    occ = np.zeros(nx * ny * nz, np.uint8)
    for k in (47, 48):  # the table top's layer, then the one an object on it fills
        occ[1 + nx * (1 + ny * k)] = 1
    grid = VoxelLattice("base", (0.0, 0.0, -0.705), (0, 0, 0, 1.0), 0.015, (nx, ny, nz), occ)
    box = PlaceRegion(
        frame_id="base",
        half_extents=(0.03, 0.03, 0.03),
        pose=Pose6D(xyz=(0.03, 0.03, 0.03), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
    )
    assert box_cells(grid, box, support_z=0.0) == frozenset({(1, 1, 48)})


def test_a_re_anchor_counts_neither_the_support_layer_nor_more_than_the_object() -> None:
    """Isaac R79: 212 of a re-anchor's 235 cells were the table top's own layer, so a banana
    pushed out of its box still re-stamped. The support layer is never the object, and a box
    holding more than ``_REANCHOR_MAX_RATIO`` times the expected cells holds something else
    too: neither re-anchors (2026-10-21 decisions)."""
    tr, _ = _confirmed_tracker(_CAN)
    t0 = tr.store.all()[0]
    origin_z = grid_from_depth(render([], _RIG)[0], _RIG).origin[2]
    support_layer = math.floor((t0.support_z - origin_z) / RES)  # the cell the top lies in
    before = grid_from_depth(render([_CAN], _RIG)[0], _RIG)
    was = {
        tuple(c) for c in np.floor((before.occupied_centers() - before.origin) / RES).astype(int)
    }

    def touched(solids: list[Solid], extra: list[tuple[int, int, int]]) -> int:
        tr, wall = _confirmed_tracker(_CAN)
        depth, _ = render(solids, _RIG)
        frame = CameraFrame(
            _RIG.name, 100_000_000, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8)
        )
        out = tr.feed(
            frame,
            grid=grid_from_depth(depth, _RIG, extra=extra),
            grid_stamp_ns=100_000_000,
            wall_s=wall + 1,
            hand_spaces=(_hand_on_top(tr),),
        )
        return out.tracked.objects[0].last_confirmed_stamp_ns

    x, y = _RIG.centre_xy
    gone = [cylinder((x + 0.15, y), 0.035, _RIG.support_z, _RIG.support_z + 0.10)]
    assert touched(gone, _box_layer(tr, support_layer)) != 100_000_000  # the table is no can
    pushed = [cylinder((x + 0.03, y), 0.035, _RIG.support_z, _RIG.support_z + 0.10)]
    assert touched(pushed, []) == 100_000_000
    crowd = [  # off the can's old cells, so only a re-anchor could confirm on it
        c
        for k in range(support_layer + 2, support_layer + 7)
        for c in _box_layer(tr, k)
        if c not in was
    ]
    assert len(crowd) > 2 * t0.visible_cells
    assert touched(pushed, crowd) != 100_000_000  # far more than the can: not re-anchored


def test_a_refit_never_shrinks_a_touched_targets_box() -> None:
    """#418 decision (2026-10-21; Isaac R70: a clear refit while the finger was at the banana
    shrank its box from 26 to 19 cm and the finger met the part it dropped). With the hand
    at it, a refit's box encloses the old box too; with no hand, it is the new fit alone."""

    def refit_after_push(*, hand: bool, push: float = 0.05) -> tuple[PlaceRegion, PlaceRegion]:
        tr, wall = _confirmed_tracker(_CAN)
        old = tr.store.all()[0].box
        x, y = _RIG.centre_xy
        can = cylinder((x + push, y), 0.035, _RIG.support_z, _RIG.support_z + 0.10)
        out, _ = step(tr, _RIG, [can], stamp_ns=100_000_000, wall_s=wall + 1,
                      hands=(_hand_on_top(tr),) if hand else ())  # fmt: skip
        return old, out.tracked.objects[0].box

    def encloses(outer: PlaceRegion, inner: PlaceRegion) -> bool:
        return all(
            abs(o - i) + hi <= ho + 1e-6
            for o, i, ho, hi in zip(
                outer.pose.xyz, inner.pose.xyz, outer.half_extents, inner.half_extents, strict=True
            )
        )  # both gravity-aligned and unrotated on this rig

    old, new = refit_after_push(hand=False)
    assert not encloses(new, old)  # the push moved the fit: the old place dropped
    old, new = refit_after_push(hand=True)  # inside its box grown a cell: unmoved
    assert new == old, "touched, an unmoved refit neither shrinks nor grows the box"
    # Spark R266 (2026-10-23): a union per touched refit ratcheted the banana's box to
    # 32 x 39 cm; every refit prompt on it masked the table and the bind went stale. Moved
    # out of its box, the old box no longer holds it: the new fit replaces it, hand or not.
    old, new = refit_after_push(hand=True, push=0.07)
    assert not encloses(new, old)


def test_the_cells_the_robot_hides_are_no_evidence_the_object_is_gone() -> None:
    """#418 decision (2026-10-21; Isaac baseline: a successful grasp keeps the fingers on the
    banana for 255-466 steps). An arm in front of most of the can hides its expected cells;
    the self-filtered map no longer holds them. Visible-only cover still confirms the track
    and re-stamps it, publishing the smaller visible count (the kernel's cover bar)."""
    tr, wall = _confirmed_tracker(_CAN)
    before = tr.store.all()[0]
    depth, labels = render([_CAN], _RIG)
    rows, cols = np.nonzero(labels == 1)
    cut = cols.min() + int(0.7 * (cols.max() - cols.min()))
    arm = (slice(rows.min() - 2, rows.max() + 3), slice(cols.min() - 2, cut))
    robot = np.zeros(depth.shape, np.uint8)
    robot[arm] = 255
    seen = depth.copy()
    seen[arm] = float(depth[labels == 1].min()) - 0.10  # the arm, nearer than the can
    filtered = depth.copy()
    filtered[arm] = 0.0  # the self-filter drops the arm's returns: no ray, no hit
    frame = CameraFrame(_RIG.name, 200_000_000, seen, _RIG.k, _RIG.t, robot)
    grid = grid_from_depth(filtered, _RIG)
    out = tr.feed(frame, grid=grid, grid_stamp_ns=200_000_000, wall_s=wall + 1)
    track = next(o for o in out.tracked.objects if o.object_id == before.object_id)
    assert track.status is TrackStatus.CONFIRMED
    assert track.last_confirmed_stamp_ns == 200_000_000  # confirmed on what the camera sees
    assert 2 <= track.visible_cells < before.visible_cells


def test_a_hidden_target_under_the_hand_re_stamps_on_what_the_map_holds_in_its_box() -> None:
    """#418 decision (2026-10-21; Isaac R83/R85/R92: the hand hid the banana, 3-7 cells in
    view, while the map held 33-52 of its cells in the box, and the confirmation aged out
    mid-grasp). The map keeps what no ray clears: with the hand at it, a hidden track
    re-anchors on its box's cells and re-stamps; with no hand it is held, never re-stamped."""

    def hidden(*, hand: bool) -> int:
        tr, wall = _confirmed_tracker(_CAN)
        depth, labels = render([_CAN], _RIG)
        rows, cols = np.nonzero(labels == 1)
        arm = (slice(rows.min() - 2, rows.max() + 3), slice(cols.min() - 2, cols.max() + 3))
        robot = np.zeros(depth.shape, np.uint8)
        robot[arm] = 255
        seen = depth.copy()
        seen[arm] = float(depth[labels == 1].min()) - 0.10  # the arm in front of all of it
        frame = CameraFrame(_RIG.name, 200_000_000, seen, _RIG.k, _RIG.t, robot)
        grid = grid_from_depth(depth, _RIG)  # the map still holds the can: no ray cleared it
        out = tr.feed(frame, grid=grid, grid_stamp_ns=200_000_000, wall_s=wall + 1,
                      hand_spaces=(_hand_on_top(tr),) if hand else ())  # fmt: skip
        return out.tracked.objects[0].last_confirmed_stamp_ns

    assert hidden(hand=True) == 200_000_000
    assert hidden(hand=False) != 200_000_000


def test_a_mask_wider_than_any_fit_is_refused_before_the_support_scan() -> None:
    """Isaac R89/R94: the segmenter answered "bowl" with the whole table (0.6-1.1 m), and each
    such mask cost ~2 s of the support scan before ``no_support``, stalling the tracker until
    the workspace region went stale. A mask whose span no capped box can hold is refused as
    ``half_extent_cap`` up front, never scanned."""
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    depth, _ = render([_ORANGE], _RIG)
    frame = CameraFrame(_RIG.name, 0, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG)
    table = depth > 0  # the whole view: the orange and all the table round it
    out = tr.feed(frame, grid=grid, grid_stamp_ns=0, wall_s=0.0)
    wall = 0.0
    for _ in range(6):
        if not out.requests:
            break
        req = out.requests[0]
        wall += 0.6
        res = (
            TrackerResult(
                req.request_id,
                boxes_xyxy=((0.0, 0.0, float(depth.shape[1]), float(depth.shape[0])),),
            )
            if req.service == "locate_in_view"
            else TrackerResult(req.request_id, mask=table)
        )
        out = tr.feed(frame, grid=grid, grid_stamp_ns=0, wall_s=wall, results=[res])
    # Refused up front: never reaching the support scan or the fit's own gates.
    assert {why for _, why in tr.refusal_counts} == {"half_extent_cap"}
    assert not out.tracked.objects


def test_a_body_inside_another_confirmed_track_is_not_a_split() -> None:
    """Design step 4: the closure cells minus other tracks. A neighbour the tracker already
    holds (a plate's cone reaching over an orange) does not make the box SPLIT; the same
    cell with no track over it still does (the test above)."""
    from openral_core import FitKind

    tr, wall = _confirmed_tracker(_CAN)
    body = _foreign_body(tr)
    centre = tuple(
        float(o + (c + 0.5) * RES)
        for o, c in zip(grid_from_depth(render([], _RIG)[0], _RIG).origin, body[1], strict=True)
    )
    neighbour = PlaceRegion(
        frame_id=_RIG.frame,
        half_extents=(RES, RES, 2 * RES),
        pose=Pose6D(xyz=centre, quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
    )
    other = tr.store.mint(
        label="orange", role=ObjectRole.TARGET, declaration_key="obj:orange_1@0",
        box=neighbour, tight=(), fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=_RIG.support_z, support_ref="map", floor_z=_RIG.support_z + RES,
        visible_cells=1, occupied_cells=1, last_fit_stamp_ns=0, last_confirmed_stamp_ns=0,
    )  # fmt: skip
    tr.store.replace(other.model_copy(update={"status": TrackStatus.CONFIRMED}))
    out, _ = step(tr, _RIG, [_CAN], stamp_ns=100_000_000, wall_s=wall + 1, extra=body)
    can = next(o for o in out.tracked.objects if o.object_id != other.object_id)
    assert can.status is TrackStatus.CONFIRMED
    assert can.last_confirmed_stamp_ns == 100_000_000


def test_one_refuting_grid_counts_once_however_often_it_is_confirmed_on() -> None:
    """``disagree_grids`` counts distinct grids (#425 review): an accepted fit and the
    per-grid pass both confirm on one stamp, and the same empty grid must not count twice."""
    tr, _ = _confirmed_tracker(_CAN)
    track = tr.store.all()[0]
    empty = grid_from_depth(render([], _RIG)[0], _RIG)
    for _ in range(tr.params.disagree_grids):
        tr._confirm(tr.store.all()[0], empty, 100_000_000, ())
    now = tr.store.get(track.object_id)
    assert now is not None and now.status is TrackStatus.CONFIRMED
    for k in range(2, tr.params.disagree_grids + 1):
        tr._confirm(tr.store.all()[0], empty, k * 100_000_000, ())
    now = tr.store.get(track.object_id)
    assert now is not None and now.status is TrackStatus.LOST


def test_per_track_state_leaves_with_the_track() -> None:
    """Track ids are never reused (#425 review): a row the store drops takes its per-track
    state with it at the next frame."""
    tr, wall = _confirmed_tracker(_CAN)
    oid = tr.store.all()[0].object_id
    assert oid in tr._cells
    tr.store.drop(oid)
    step(tr, _RIG, [], stamp_ns=100_000_000, wall_s=wall + 1)
    for per_track in (tr._views, tr._cells, tr._centroid, tr._disagree, tr._served):
        assert oid not in per_track
