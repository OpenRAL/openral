"""Tracks: association by continuity, ties, inferred identity, source-time ageing (#418 5-6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    ObjectQuery,
    ObjectQueryStatus,
    ObjectRole,
    Pose6D,
    TrackedObject,
    TrackStatus,
)
from openral_core.exceptions import ROSConfigError
from openral_world_state.object_tracker import (
    CameraFrame,
    ObjectTracker,
    TrackerResult,
    associate,
    containing_fit,
    visible_fraction,
)
from openral_world_state.object_tracker.release import release_track

from tests.unit._tracker_scenes import (
    RES,
    Rig,
    Solid,
    answer,
    banana,
    droid_exterior,
    grid_from_depth,
    render,
    so101_top,
    sphere,
    step,
)

_RIG = so101_top()
_R = 0.024
_QUERY = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)
_PUSH_M = 0.02  # inside the box grown by a cell: not a moved refit


def _get(tr: ObjectTracker, object_id: str) -> TrackedObject:
    track = tr.store.get(object_id)
    assert track is not None
    return track


def _oranges(spacing: float) -> list[Solid]:
    x, y = _RIG.centre_xy
    return [sphere((x + d, y, _RIG.support_z + _R), _R) for d in (-spacing, 0.0, spacing)]


def _grounded(oranges: list[Solid]) -> tuple[ObjectTracker, float]:
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    out, wall = step(tr, _RIG, oranges, stamp_ns=0, wall_s=0.0)
    assert [o.status for o in out.tracked.objects] == [TrackStatus.CONFIRMED] * 3
    return tr, wall


def test_three_identical_oranges_tie_updates_nothing() -> None:
    oranges = _oranges(0.05)
    tr, wall = _grounded(oranges)
    a, b = _get(tr, "trk:1"), _get(tr, "trk:2")
    mid = tuple((p + q) / 2 for p, q in zip(a.box.pose.xyz, b.box.pose.xyz, strict=True))
    assert associate(tr.store.all(), mid, voxel_m=RES) == (None, "tie")
    # A segmenter mask that spans two oranges: its fit sits between them and moves neither.
    depth, labels = render(oranges, _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG)
    out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=wall + 1)
    req = out.requests[0]
    boxes, rev = {o.object_id: o.box for o in tr.store.all()}, tr.store.revision
    both = (labels == 1) | (labels == 2)
    union = TrackerResult(req.request_id, mask=both)
    # That fit passes every gate and lands on the tie (what feed then does with it).
    fit = containing_fit(
        both, depth, frame.robot_mask, _RIG.k, _RIG.t, support_z=a.support_z,
        resolution=RES, frame_id=_RIG.frame, stamp_ns=1,
    )  # fmt: skip
    assert fit.refusal == "" and fit.box is not None
    vis = visible_fraction(fit.box, depth, frame.robot_mask, _RIG.k, _RIG.t)
    assert vis is not None and vis >= 0.5
    centroid = tuple(fit.surface.mean(axis=0))
    assert associate(tr.store.all(), centroid, voxel_m=RES) == (None, "tie")
    out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=wall + 1, results=[union])
    assert {o.object_id: o.box for o in out.tracked.objects} == boxes
    assert tr.store.revision == rev


def test_refound_lost_track_inherits_its_id_as_inferred() -> None:
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    lost = _get(tr, "trk:2")
    tr.store.replace(lost.model_copy(update={"status": TrackStatus.LOST}))
    for n in range(1, 4):
        out, wall = step(tr, _RIG, oranges, stamp_ns=n * 100_000_000, wall_s=wall + 1)
    back = out.tracked.objects[1]
    assert back.object_id == "trk:2" and back.identity_inferred
    assert back.status is TrackStatus.CONFIRMED
    assert [o.object_id for o in out.tracked.objects] == ["trk:1", "trk:2", "trk:3"]


def test_ageing_runs_on_the_source_clock_not_the_wall() -> None:
    """A 0.1x real-time sim: 1 s of wall per 0.1 s of source. Lifetimes follow the source."""
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    statuses = {}
    for n in range(1, 71):  # source 0.1 .. 7.0 s; wall 1 .. 70 s
        wall += 1.0
        out, wall = step(tr, _RIG, oranges, stamp_ns=n * 100_000_000, wall_s=wall, with_grid=False)
        statuses[n] = out.tracked.objects[0].status
    assert statuses[19] is TrackStatus.CONFIRMED  # 1.9 s of source, 19 s of wall: still fresh
    assert statuses[21] is TrackStatus.TENTATIVE  # past max_age_s = 2 s of source: ineligible
    assert statuses[59] is TrackStatus.TENTATIVE
    assert statuses[61] is TrackStatus.LOST  # past lost_s = 6 s of source
    query = tr.queries.get("g", "q0")
    assert query is not None and query.status is ObjectQueryStatus.UNRESOLVED  # search restarts


def test_track_store_contract() -> None:
    tr, _ = _grounded(_oranges(0.09))
    store = tr.store
    rev = store.revision
    t = _get(tr, "trk:1")
    store.replace(t.model_copy(update={"epoch": 1}))
    assert store.revision == rev + 1 and _get(tr, "trk:1").epoch == 1
    with pytest.raises(ROSConfigError, match="epoch"):
        store.replace(t)  # epoch 0 < 1
    with pytest.raises(ROSConfigError, match="unknown"):
        store.replace(t.model_copy(update={"object_id": "trk:99"}))
    assert [o.object_id for o in store.all()] == ["trk:1", "trk:2", "trk:3"]


def test_a_candidate_that_splits_on_the_map_hands_over_to_the_next() -> None:
    """SAM's "part" of a point on one of two separate oranges can hold both: its box splits
    on the map, so the next (smaller) candidate is fitted instead of minting a SPLIT track."""
    oranges = _oranges(0.08)[:2]
    depth, labels = render(oranges, _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    # The counter's top mid-cell, as a real octree has it (on a layer boundary the counter's
    # own returns would round into the layer above and join the two oranges).
    grid = grid_from_depth(depth, _RIG, shift_z=0.5 * RES)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=0.0)
    (locate,) = out.requests
    rows, cols = np.nonzero(labels == 1)  # the detector boxed the first orange
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    out = tr.feed(
        frame, grid=grid, grid_stamp_ns=1, wall_s=1.0,
        results=[TrackerResult(locate.request_id, boxes_xyxy=(box,))],
    )  # fmt: skip
    (seg,) = out.requests  # the detector box's prompt, on the same frame
    both, one = labels > 0, labels == 1
    out = tr.feed(
        frame, grid=grid, grid_stamp_ns=1, wall_s=2.1,
        results=[TrackerResult(seg.request_id, mask=both, alternatives=(one,))],
    )  # fmt: skip
    (track,) = out.tracked.objects
    assert track.status is TrackStatus.CONFIRMED
    x, y = _RIG.centre_xy  # labels == 1 is the first orange, at x - 0.08
    assert np.linalg.norm(np.asarray(track.box.pose.xyz[:2]) - (x - 0.08, y)) < 0.02


def _arm_beside(
    labels: NDArray[np.int64], depth: NDArray[np.float64]
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """A robot region (an arm 10 cm nearer) touching the first object's mask and reaching
    past the occluder band: what makes a fit PROVISIONAL."""
    rows, cols = np.nonzero(labels == 1)
    robot = np.zeros(depth.shape, np.uint8)
    arm = (slice(rows.min(), rows.max() + 1), slice(cols.max() + 1, cols.max() + 25))
    robot[arm] = 255
    hidden = depth.copy()
    hidden[arm] = float(depth[labels == 1].min()) - 0.10
    return robot, hidden


def _ground(tr: ObjectTracker, depth: NDArray[np.float64], labels: NDArray[np.int64],
            robot: NDArray[np.uint8],
            stamp: int, wall: float) -> float:  # fmt: skip
    """One query -> locate -> segment round on a frame with ``robot`` as its robot mask."""
    frame = CameraFrame(_RIG.name, stamp, depth, _RIG.k, _RIG.t, robot)
    grid = grid_from_depth(depth, _RIG, shift_z=0.5 * RES)
    out = tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall)
    for _ in range(4):
        if not out.requests:
            break
        req = out.requests[0]
        rows, cols = np.nonzero(labels == 1)
        box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
        res = (
            TrackerResult(req.request_id, boxes_xyxy=(box,))
            if req.service == "locate_in_view"
            else TrackerResult(req.request_id, mask=labels == 1)
        )
        wall += 1.0
        out = tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall, results=[res])
    for req in out.requests:  # leave nothing in flight (same wall: nothing new goes out)
        tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall,
                results=[TrackerResult(req.request_id)])  # fmt: skip
    return wall


def test_a_fit_through_the_robot_is_provisional_until_a_clear_refit() -> None:
    """#418 decision (2026-10-17): an arm hiding more than the occluder band makes the fit
    PROVISIONAL, yet the track confirms (K2 may exempt over it); a provisional refit never
    overwrites a clear one, and a clear refit clears the flag."""
    orange = _oranges(0.05)[1:2]
    depth, labels = render(orange, _RIG)
    robot, hidden = _arm_beside(labels, depth)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    wall = _ground(tr, hidden, labels, robot, stamp=1, wall=0.0)
    (track,) = tr.store.all()
    assert track.provisional and track.status is TrackStatus.CONFIRMED  # exemption-eligible
    wall = _ground(tr, depth, labels, np.zeros_like(robot), stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert not track.provisional and track.last_fit_stamp_ns == 2  # the clear refit clears it
    (clear,) = tr.store.all()
    _ground(tr, hidden, labels, robot, stamp=3, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert not track.provisional and track.box == clear.box  # never overwritten ...
    # ... but re-dated (2026-10-22, Spark R107-R116): the object was measured inside the box
    # at 3, so the grasp binder sees a fresh fit, not one aged by a hand hiding the target.
    assert track.last_fit_stamp_ns == 3


def test_a_pushed_object_takes_the_fit_through_the_robot_as_provisional() -> None:
    """#418 decision (2026-10-21, Spark R17 banana / R19 bowl): when the hand pushes an
    object, the stored clear box no longer contains it; a refit whose measured surface
    leaves that box replaces it even through the robot, kept PROVISIONAL (never a payload,
    still K2 / K7 geometry). Unmoved, the clear box stays (the test above)."""
    orange = _oranges(0.05)[1:2]
    depth, labels = render(orange, _RIG)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    wall = _ground(tr, depth, labels, np.zeros(depth.shape, np.uint8), stamp=1, wall=0.0)
    (clear,) = tr.store.all()
    assert not clear.provisional
    x, y = _RIG.centre_xy
    pushed = [sphere((x + 0.045, y, _RIG.support_z + _R), _R)]  # still this track's gate
    depth2, labels2 = render(pushed, _RIG)
    robot, hidden = _arm_beside(labels2, depth2)
    _ground(tr, hidden, labels2, robot, stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert track.object_id == clear.object_id
    assert track.provisional and track.last_fit_stamp_ns == 2
    assert track.box.pose.xyz[0] - x > 0.04  # the box followed the orange


def test_a_bound_tracks_box_follows_the_in_hand_refit() -> None:
    """2026-10-22, Spark R127: the bind-time box stayed where the banana was grasped while the
    hand carried it, so the kernel read the binding off the hand. A camera refit of a BOUND
    track moves its box to where the object is seen; status and the bind's flags stay."""
    orange = _oranges(0.05)[1:2]
    depth, labels = render(orange, _RIG)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    wall = _ground(tr, depth, labels, np.zeros(depth.shape, np.uint8), stamp=1, wall=0.0)
    (clear,) = tr.store.all()
    tr.store.replace(
        clear.model_copy(
            update={
                "status": TrackStatus.BOUND,
                "bound_hand": "gripper",
                "attach_link": "jaw",
                "pose_in_link": Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="jaw"),
            }
        )
    )
    x, y = _RIG.centre_xy
    carried = [sphere((x + 0.045, y, _RIG.support_z + _R), _R)]
    depth2, labels2 = render(carried, _RIG)
    _ground(tr, depth2, labels2, np.zeros(depth.shape, np.uint8), stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert track.status is TrackStatus.BOUND and not track.provisional
    assert track.last_fit_stamp_ns == 2
    assert track.box.pose.xyz[0] - x > 0.04  # the box followed the orange


def _ground_mask(tr: ObjectTracker, rig: Rig, depth: NDArray[np.float64], mask: NDArray[np.bool_],
                 stamp: int, wall: float) -> float:  # fmt: skip
    """``_ground`` on any rig, answering every segment request with ``mask``."""
    frame = CameraFrame(rig.name, stamp, depth, rig.k, rig.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, rig, shift_z=0.5 * RES)
    out = tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall)
    rows, cols = np.nonzero(mask)
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    for _ in range(4):
        if not out.requests:
            break
        req = out.requests[0]
        res = (
            TrackerResult(req.request_id, boxes_xyxy=(box,))
            if req.service == "locate_in_view"
            else TrackerResult(req.request_id, mask=mask)
        )
        wall += 1.0
        out = tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall, results=[res])
    for req in out.requests:
        tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall,
                results=[TrackerResult(req.request_id)])  # fmt: skip
    return wall


def test_a_partial_mask_never_shrinks_an_unmoved_objects_box() -> None:
    """#418 decision (2026-10-21, Spark R37/R39): a clear refit from a partial mask halved
    an unmoved banana's box and moved it 5-6 cm; the finger met the uncovered half, outside
    every explanation. A refit that leaves out surface cells the last fit measured, which
    the map still holds, is ignored. A full-mask refit is still taken."""
    rig = droid_exterior()
    fruit = banana(rig.centre_xy, rig.support_z, 0.08, 0.018, 1.2, math.pi / 2)
    depth, labels = render([fruit], rig)
    query = ObjectQuery(label="banana", count="all", role=ObjectRole.TARGET)
    tr = ObjectTracker(rig.frame)
    tr.open_query("g", "q0", query, "obj:banana@0", stamp_ns=0)
    wall = _ground_mask(tr, rig, depth, labels == 1, stamp=1, wall=0.0)
    (full,) = tr.store.all()
    assert full.status is TrackStatus.CONFIRMED and not full.provisional
    _, cols = np.nonzero(labels == 1)
    half = (labels == 1) & (np.arange(labels.shape[1])[None, :] < np.median(cols))
    wall = _ground_mask(tr, rig, depth, half, stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert track.object_id == full.object_id
    # Ignored for geometry, but it measured the banana inside the kept box: re-dated (2026-10-31)
    assert track.last_fit_stamp_ns == 2
    assert track.box == full.box
    _ground_mask(tr, rig, depth, labels == 1, stamp=3, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert track.last_fit_stamp_ns == 3  # a full refit still replaces it


def test_an_ignored_refit_still_moves_the_cover_to_the_surface_it_measured() -> None:
    """#418 decision (2026-10-21, Isaac R54): a target pushed inside its own box stays in it,
    so its refits through the hand are ignored for geometry, and the first fit's cells emptied
    as it moved: never re-confirmed, it dropped out mid-push. The ignored refit's surface now
    replaces the expected cells (the box stays), and the next grid confirms it there."""
    orange = _oranges(0.05)[1:2]
    depth, labels = render(orange, _RIG)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    wall = _ground(tr, depth, labels, np.zeros(depth.shape, np.uint8), stamp=1, wall=0.0)
    (clear,) = tr.store.all()
    x, y = _RIG.centre_xy
    pushed = [sphere((x + _PUSH_M, y, _RIG.support_z + _R), _R)]
    depth2, labels2 = render(pushed, _RIG)
    robot, hidden = _arm_beside(labels2, depth2)
    wall = _ground(tr, hidden, labels2, robot, stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert track.box == clear.box and not track.provisional  # geometry untouched
    frame = CameraFrame(_RIG.name, 3, depth2, _RIG.k, _RIG.t, np.zeros(depth2.shape, np.uint8))
    tr.feed(frame, grid=grid_from_depth(depth2, _RIG, shift_z=0.5 * RES), grid_stamp_ns=3,
            wall_s=wall + 1.0)  # fmt: skip
    (track,) = tr.store.all()
    assert track.last_confirmed_stamp_ns == 3  # confirmed where the surface is now


def test_a_split_track_never_takes_a_neighbours_fit() -> None:
    """A SPLIT box holds two bodies; a fit centred inside it is a new object, never the
    split track's continuation (PickOrange r2po_shadow3: a swap among static oranges)."""
    tr, _ = _grounded(_oranges(0.05))
    a = _get(tr, "trk:1")
    big = a.model_copy(
        update={
            "status": TrackStatus.SPLIT,
            "box": a.box.model_copy(update={"half_extents": (0.12, 0.05, 0.05)}),
        }
    )
    others = [t for t in tr.store.all() if t.object_id != "trk:1"]
    inside = (a.box.pose.xyz[0] + 0.08, a.box.pose.xyz[1], a.box.pose.xyz[2])
    assert associate([big], inside, voxel_m=RES) == (None, "new")
    assert associate([big.model_copy(update={"status": TrackStatus.CONFIRMED})], inside,
                     voxel_m=RES)[0] == "trk:1"  # fmt: skip
    assert others  # the scene still holds the neighbours


def test_a_two_object_refit_never_replaces_a_confirmed_box() -> None:
    """PickOrange r3po_on1: a mask over orange_1 and orange_3 refit the CONFIRMED orange_1
    track, which went SPLIT, so orange_1's next fit took a new id. A refit that splits on
    the map is ignored: the confirmed box and id stay."""
    oranges = _oranges(0.08)[:2]
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    depth, labels = render(oranges, _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG, shift_z=0.5 * RES)
    (locate,) = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=0.0).requests
    rows, cols = np.nonzero(labels == 1)
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=1.0,
                  results=[TrackerResult(locate.request_id, boxes_xyxy=(box,))])  # fmt: skip
    (seg,) = out.requests
    out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=1.1,
                  results=[TrackerResult(seg.request_id, mask=labels == 1)])  # fmt: skip
    (fitted,) = out.tracked.objects
    assert fitted.status is TrackStatus.CONFIRMED
    # Its box as an earlier, wider (provisional-era) fit left it: wide enough to hold the
    # centroid of a mask over both oranges.
    wide = fitted.box.model_copy(update={"half_extents": (0.06, 0.06, fitted.box.half_extents[2])})
    tr.store.replace(fitted.model_copy(update={"box": wide}))
    before = _get(tr, fitted.object_id)
    # A refresh prompt answered with a mask over both oranges.
    t2 = 6_000_000_000  # past the count-all search window: refreshes compete again
    frame2 = CameraFrame(_RIG.name, t2, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    req = None
    for n in range(40):  # cycle the detector's turns until the track's refresh comes up
        out = tr.feed(frame2, grid=grid, grid_stamp_ns=t2, wall_s=3.0 + n)
        if not out.requests:
            continue
        if out.requests[0].object_id == before.object_id:
            req = out.requests[0]
            break
        tr.feed(frame2, grid=grid, grid_stamp_ns=t2, wall_s=3.0 + n,
                results=[TrackerResult(out.requests[0].request_id)])  # fmt: skip
    assert req is not None
    out = tr.feed(frame2, grid=grid, grid_stamp_ns=t2, wall_s=3.0 + n,
                  results=[TrackerResult(req.request_id, mask=labels > 0)])  # fmt: skip
    after = _get(tr, before.object_id)
    assert after.status is TrackStatus.CONFIRMED and after.box == before.box


def test_a_released_orange_is_not_followed_and_is_found_afresh() -> None:
    """A released object is never re-associated (2026-10-22, Spark R177-R185).

    trk:2 is bound, released 10 cm above its spot and lands 3 cm aside. Its track stays
    RELEASED where it was let go, takes no fit and is never prompted; the landed orange is
    found afresh under a new id. The identical neighbours keep their ids and epochs.
    """
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    before = {o.object_id: o for o in tr.store.all()}
    two = before["trk:2"]
    x, y, z = two.box.pose.xyz
    tr.store.replace(
        TrackedObject.model_validate(
            {
                **two.model_dump(),
                "status": TrackStatus.BOUND,
                "bound_hand": "gripper",
                "attach_link": "jaw",
                "pose_in_link": Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="jaw"),
            }
        )
    )
    released = release_track(
        tr.store,
        "trk:2",
        object_pose=Pose6D(xyz=(x, y, z + 0.10), quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
        support_z=_RIG.support_z,
        cell_m=RES,
        stamp_ns=50_000_000,
    )
    assert released is not None and released.status is TrackStatus.RELEASED
    cx, cy = _RIG.centre_xy
    landed = [oranges[0], sphere((cx, cy + 0.03, _RIG.support_z + _R), _R), oranges[2]]
    for n in range(1, 6):
        out, wall = step(tr, _RIG, landed, stamp_ns=n * 100_000_000, wall_s=wall + 1)
    by_id = {o.object_id: o for o in out.tracked.objects}
    gone = by_id["trk:2"]
    assert gone.status is TrackStatus.RELEASED and gone.epoch == 0
    assert gone.box == released.box, "a released track is never refit"
    [fresh] = [o for o in by_id.values() if o.object_id not in before]
    assert fresh.status is TrackStatus.CONFIRMED and not fresh.identity_inferred
    assert abs(fresh.box.pose.xyz[1] - (cy + 0.03)) < RES
    for other in ("trk:1", "trk:3"):
        assert by_id[other].epoch == 0 and not by_id[other].identity_inferred
        np.testing.assert_allclose(
            by_id[other].box.pose.xyz[:2], before[other].box.pose.xyz[:2], atol=RES
        )


def test_a_retired_release_seeds_its_query_where_it_was_let_go() -> None:
    """Spark R242-R261 (2026-10-22): a retired banana's query went back to the locator,
    whose table-sized boxes the fit refused 186 times; it was never re-found. The query is
    first prompted on the map's raised tops around where the object was let go."""
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    two = _get(tr, "trk:2")
    tr.store.replace(
        TrackedObject.model_validate(
            {
                **two.model_dump(),
                "status": TrackStatus.BOUND,
                "bound_hand": "gripper",
                "attach_link": "jaw",
                "pose_in_link": Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="jaw"),
            }
        )
    )
    x, y, z = two.box.pose.xyz  # let go 10 cm up (R268: the in-hand box held no support)
    released = release_track(
        tr.store, "trk:2",
        object_pose=Pose6D(xyz=(x, y, z + 0.10), quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
        support_z=_RIG.support_z, cell_m=RES, stamp_ns=50_000_000,
    )  # fmt: skip
    assert released is not None
    tr.queries.note_released(released)  # the session's retire, at the end of the window
    tr.store.drop("trk:2")
    q = tr.queries.get("g", "q0")
    assert q is not None and q.prior is not None
    bottom = q.prior.pose.xyz[2] - q.prior.half_extents[2]
    assert bottom <= released.support_z + 1e-9, "the column reaches down to the support"
    cx, cy = _RIG.centre_xy
    landed = [oranges[0], sphere((cx, cy + 0.03, _RIG.support_z + _R), _R), oranges[2]]
    depth, _ = render(landed, _RIG)
    frame = CameraFrame(_RIG.name, 10**8, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG)
    served = []  # q0's prompts, in order; the earlier detector boxes still queued go first
    for n in range(30):
        out = tr.feed(frame, grid=grid, grid_stamp_ns=10**8, wall_s=wall + 1 + n)
        if not out.requests:
            continue
        r = out.requests[0]
        if r.query_key == ("g", "q0") and r.box_xyxy is None:
            served.append(r)
        if r.seed_cell is not None or r.service == "locate_in_view":
            break
        tr.feed(frame, grid=grid, grid_stamp_ns=10**8, wall_s=wall + 1 + n,
                results=[TrackerResult(r.request_id)])  # fmt: skip
    assert served and served[-1].seed_cell is not None, served
    assert served[-1].within_xyxy is not None, "the prior's image rectangle bounds the mask"


def test_a_fit_is_never_taken_by_a_track_of_another_label() -> None:
    """Spark R269 (2026-10-23): the banana let go into the bowl lay inside the bowl's box,
    and by position alone every banana fit refit the bowl; it was never found again."""
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    two = _get(tr, "trk:2")
    tr.store.replace(two.model_copy(update={"label": "plate"}))  # a plate where it stands
    for n in range(1, 8):
        out, wall = step(tr, _RIG, oranges, stamp_ns=n * 400_000_000, wall_s=wall + 1)
    oranges_now = [o for o in out.tracked.objects if o.label == "orange"]
    assert len(oranges_now) == 3, [(o.object_id, o.label) for o in out.tracked.objects]
    fresh = [o for o in oranges_now if o.object_id not in ("trk:1", "trk:3")]
    np.testing.assert_allclose(fresh[0].box.pose.xyz[:2], two.box.pose.xyz[:2], atol=RES)


def test_a_neighbours_refresh_yields_to_the_targets() -> None:
    """Spark R275 (2026-10-23): three tote cartons shared the refresh slot round-robin with
    the target, whose fit was 1.5 s old at the grasp; the bind went untracked."""
    oranges = _oranges(0.09)
    tr, wall = _grounded(oranges)
    for oid in ("trk:1", "trk:3"):
        tr.store.replace(_get(tr, oid).model_copy(update={"role": ObjectRole.CONTEXT}))
    depth, labels = render(oranges, _RIG)
    grid = grid_from_depth(depth, _RIG)
    issued: list[str | None] = []
    results: list[TrackerResult] = []
    for n in range(1, 25):
        stamp = 6_000_000_000 + n * 300_000_000  # past the count-all search window
        frame = CameraFrame(
            _RIG.name, stamp, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8)
        )
        wall += 1.0
        out = tr.feed(frame, grid=grid, grid_stamp_ns=stamp, wall_s=wall, results=results)
        results = []
        if out.requests:
            req = out.requests[0]
            issued.append(req.object_id)
            frame2 = CameraFrame(
                _RIG.name, stamp, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8)
            )
            tr.feed(frame2, grid=grid, grid_stamp_ns=stamp, wall_s=wall,
                    results=[answer(req, labels)])  # fmt: skip
    refits = {oid: issued.count(oid) for oid in ("trk:1", "trk:2", "trk:3")}
    assert refits["trk:2"] > max(refits["trk:1"], refits["trk:3"]), (refits, issued)


def test_a_held_candidate_is_never_aged() -> None:
    """Spark banana R335 (2026-10-23): a provisional hold carried the banana; its track's box on
    the table emptied, no grid re-confirmed it, and ageing made it TENTATIVE mid-carry. The
    kernel dropped the payload's own cells with it and stopped the banana against its far end.
    A candidate of a live hold ages no more than a BOUND track does; any other track still does."""
    from openral_world_state.object_tracker.tracks import age_tracks

    tr, _ = _grounded(_oranges(0.09))
    ids = [t.object_id for t in tr.store.all()]
    now = max(t.last_confirmed_stamp_ns for t in tr.store.all()) + 5 * 1_000_000_000
    age_tracks(tr.store, now, max_age_s=2.0, lost_s=6.0, keep_lost_s=30.0, held=frozenset(ids[:1]))
    assert _get(tr, ids[0]).status is TrackStatus.CONFIRMED
    assert all(_get(tr, i).status is TrackStatus.TENTATIVE for i in ids[1:])


def test_a_fit_through_the_robot_rising_above_the_clear_box_is_no_move() -> None:
    """Spark banana R343 (2026-10-23): fits of the bowl through the robot took in what the
    hand held over it; every surface point outside the clear box lay above it (115 of 115),
    the bowl read as moved and its box was replaced, provisional. A resting object pushed
    along its support does not rise above its own clear ceiling; only points beside or
    below that box evidence a move for a fit through the robot."""
    from tests.unit._tracker_scenes import cylinder

    x, y = _RIG.centre_xy
    z0 = _RIG.support_z
    bowl = cylinder((x, y), 0.07, z0, z0 + 0.05)
    depth, labels = render([bowl], _RIG)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY, "obj:orange_1@0", stamp_ns=0)
    wall = _ground(tr, depth, labels, np.zeros(depth.shape, np.uint8), stamp=1, wall=0.0)
    (clear,) = tr.store.all()
    assert not clear.provisional
    held = sphere((x, y, z0 + 0.07), 0.02)  # resting on it, the mask takes it in
    depth2, labels2 = render([bowl, held], _RIG)
    both = (labels2 > 0).astype(labels2.dtype)
    robot, hidden = _arm_beside(both, depth2)
    _ground(tr, hidden, both, robot, stamp=2, wall=wall + 1.0)
    (track,) = tr.store.all()
    assert not track.provisional and track.box == clear.box, "held above: not moved"
    assert track.last_fit_stamp_ns == 2  # measured inside its box: re-dated
