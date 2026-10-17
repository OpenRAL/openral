"""Tracks: association by continuity, ties, inferred identity, source-time ageing (#418 5-6)."""

from __future__ import annotations

import numpy as np
import pytest
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

from tests.unit._tracker_scenes import RES, Solid, grid_from_depth, render, so101_top, sphere, step

_RIG = so101_top()
_R = 0.024
_QUERY = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)


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


def test_a_released_orange_among_three_identical_reassociates_to_itself() -> None:
    """Release re-association with identical neighbours: only the fall column decides.

    trk:2 is bound, released 10 cm above its spot and lands 3 cm aside; its neighbours
    (identical oranges 9 cm away) keep their ids and epochs, and the landed one comes back
    as trk:2, epoch 1, ``identity_inferred`` — never as a neighbour, never a new id.
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
    assert sorted(by_id) == ["trk:1", "trk:2", "trk:3"], "no new id for the landed orange"
    back = by_id["trk:2"]
    assert back.status is TrackStatus.CONFIRMED and back.epoch == 1 and back.identity_inferred
    assert abs(back.box.pose.xyz[1] - (cy + 0.03)) < RES
    for other in ("trk:1", "trk:3"):
        assert by_id[other].epoch == 0 and not by_id[other].identity_inferred
        np.testing.assert_allclose(
            by_id[other].box.pose.xyz[:2], before[other].box.pose.xyz[:2], atol=RES
        )
