"""Standing queries (issue #418 Addendum G): keep looking, escalate, never fail silently."""

from __future__ import annotations

import math

import numpy as np
from openral_core import ObjectQuery, ObjectQueryStatus, ObjectRole, TrackStatus
from openral_world_state.object_tracker import (
    CameraFrame,
    ObjectTracker,
    QueryEscalation,
    TrackerParams,
    TrackerResult,
)

from tests.unit._tracker_scenes import banana, droid_exterior, grid_from_depth, render, step

_RIG = droid_exterior()
_BANANA = banana(_RIG.centre_xy, _RIG.support_z, 0.08, 0.018, 1.2, math.pi / 2)
_QUERY = ObjectQuery(label="banana", count="one", role=ObjectRole.TARGET)
_S = 1_000_000_000


def _status(tr: ObjectTracker) -> ObjectQueryStatus:
    query = tr.queries.get("g", "q0")
    assert query is not None
    return query.status


def _tracker() -> ObjectTracker:
    tr = ObjectTracker(_RIG.frame, TrackerParams(search_start_s=5.0, search_max_s=60.0))
    tr.open_query("g", "q0", _QUERY, "obj:banana@0", stamp_ns=0)
    return tr


def test_late_appearance_resolves_without_failing_the_goal() -> None:
    """Hidden at goal start, in view 5 s later: the passive search finds it."""
    tr, wall = _tracker(), 0.0
    escalations: list[QueryEscalation] = []
    for n in range(10):  # 0 .. 4.5 s: nothing to see; the detector keeps being asked
        out, wall = step(tr, _RIG, [], stamp_ns=n * _S // 2, wall_s=wall + 1)
        escalations += out.escalations
        assert _status(tr) is ObjectQueryStatus.UNRESOLVED
    for n in range(10, 13):  # 5 s on: the banana is in view
        out, wall = step(tr, _RIG, [_BANANA], stamp_ns=n * _S // 2, wall_s=wall + 1)
        escalations += out.escalations
    assert _status(tr) is ObjectQueryStatus.RESOLVED
    assert tr.queries.ready("g")
    [track] = out.tracked.objects
    assert track.label == "banana" and track.status is TrackStatus.CONFIRMED
    assert track.declaration_key == "obj:banana@0"
    assert all(e.kind != "exhausted" for e in escalations)


def test_never_appears_escalates_then_exhausts() -> None:
    tr, wall, kinds = _tracker(), 0.0, []
    for n in range(0, 62):  # one frame per source second
        out, wall = step(tr, _RIG, [], stamp_ns=n * _S, wall_s=wall + 1)
        kinds += [(n, e.kind, e.label) for e in out.escalations]
    assert kinds == [(5, "unresolved", "banana"), (60, "exhausted", "banana")]
    assert _status(tr) is ObjectQueryStatus.EXHAUSTED
    assert not tr.queries.ready("g")


def test_scheduler_serves_unresolved_queries_fairly() -> None:
    """Two unresolved queries share the detector turn by turn (G.2), never one starved."""
    tr = _tracker()
    plate = ObjectQuery(label="bowl", count="one", role=ObjectRole.DESTINATION)
    tr.open_query("g", "q1", plate, "obj:banana@0", stamp_ns=0)
    depth, _ = render([], _RIG)
    grid = grid_from_depth(depth, _RIG)
    labels, wall = [], 0.0
    for n in range(6):
        frame = CameraFrame(_RIG.name, n, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
        out = tr.feed(frame, grid=grid, grid_stamp_ns=n, wall_s=wall)
        req = out.requests[0]
        labels.append(req.label)
        done = [TrackerResult(req.request_id)]  # nothing detected
        tr.feed(frame, grid=grid, grid_stamp_ns=n, wall_s=wall, results=done)
        wall += 1.0
    assert labels == ["banana", "bowl"] * 3


def test_tf_or_mask_missing_ages_only() -> None:
    tr = _tracker()
    depth, _ = render([_BANANA], _RIG)
    frame = CameraFrame(_RIG.name, 0, depth, _RIG.k, None, np.zeros(depth.shape, np.uint8))
    out = tr.feed(frame, grid=grid_from_depth(depth, _RIG), grid_stamp_ns=0, wall_s=0.0)
    assert out.requests == () and out.tracked.objects == []


def test_a_seeded_prompt_is_bounded_by_the_seed_rectangle() -> None:
    """A map-seeded point prompt carries its seed's image rectangle, so the mask pick keeps
    only candidates inside it (SAM's "whole" of a packed carton is the tote's contents)."""
    from openral_core import PlaceRegion, Pose6D
    from openral_world_state.object_tracker.fit import project_box

    x, y = _RIG.centre_xy
    seed = PlaceRegion(
        frame_id=_RIG.frame,
        pose=Pose6D(xyz=(x, y, _RIG.support_z + 0.05), quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
        half_extents=(0.12, 0.12, 0.07),
    )
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", _QUERY.model_copy(update={"seed": seed}), "obj:banana@0", stamp_ns=0)
    depth, _ = render([_BANANA], _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    out = tr.feed(frame, grid=grid_from_depth(depth, _RIG), grid_stamp_ns=1, wall_s=0.0)
    (req,) = out.requests
    rect = project_box(seed, _RIG.k, _RIG.t)
    assert req.service == "segment_in_view" and req.point_uv is not None and rect is not None
    assert req.box_xyxy is None  # a bound on the pick, never SAM's box prompt
    assert req.within_xyxy == tuple(float(v) for v in rect[0])
    x0, y0, x1, y1 = req.within_xyxy
    assert x0 <= req.point_uv[0] <= x1 and y0 <= req.point_uv[1] <= y1


def test_a_low_body_grounds_when_the_support_tops_out_mid_cell() -> None:
    """A bowl on a table whose top face lies mid-cell: the object's bottom shares the table's
    top layer, and that layer must still count as its support (DROID, 2026-10-17)."""
    from tests.unit._tracker_scenes import RES, cylinder

    x, y = _RIG.centre_xy
    bowl = cylinder((x, y), 0.08, _RIG.support_z, _RIG.support_z + 0.05)
    q = ObjectQuery(label="bowl", count="one", role=ObjectRole.DESTINATION)
    for shift in (0.0, 0.25 * RES, 0.5 * RES, 0.75 * RES):
        tr = ObjectTracker(_RIG.frame)
        tr.open_query("g", "q0", q, "obj:bowl@0", stamp_ns=0)
        out, _ = step(tr, _RIG, [bowl], stamp_ns=0, wall_s=0.0, shift_z=shift)
        assert [o.label for o in out.tracked.objects] == ["bowl"], shift


def test_a_hollow_body_grounds_on_the_support_around_it() -> None:
    """A bowl seen from above is a ring: the table shows through its middle, so the support
    search is anchored on the body's highest seen point, not its centroid (DROID bowl)."""
    from tests.unit._tracker_scenes import RES, sphere

    x, y = _RIG.centre_xy
    ring = [
        sphere((x + 0.06 * math.cos(a), y + 0.06 * math.sin(a), _RIG.support_z + 0.015), 0.015)
        for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)
    ]
    depth, labels = render(ring, _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG, shift_z=0.5 * RES)
    tr = ObjectTracker(_RIG.frame)
    q = ObjectQuery(label="bowl", count="one", role=ObjectRole.DESTINATION)
    tr.open_query("g", "q0", q, "obj:bowl@0", stamp_ns=0)
    (locate,) = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=0.0).requests
    rows, cols = np.nonzero(labels > 0)
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    out = tr.feed(
        frame, grid=grid, grid_stamp_ns=1, wall_s=1.0,
        results=[TrackerResult(locate.request_id, boxes_xyxy=(box,))],
    )  # fmt: skip
    (seg,) = out.requests
    out = tr.feed(
        frame, grid=grid, grid_stamp_ns=1, wall_s=1.1,
        results=[TrackerResult(seg.request_id, mask=labels > 0)],
    )  # fmt: skip
    assert [o.label for o in out.tracked.objects] == ["bowl"]


def test_a_count_all_query_keeps_its_turn_while_another_stays_unresolved() -> None:
    """Three oranges and a plate: one orange resolves "all", but within the search window
    the oranges still get detector turns beside the unresolved plate (PickOrange)."""
    from tests.unit._tracker_scenes import sphere

    x, y = _RIG.centre_xy
    orange = sphere((x, y, _RIG.support_z + 0.024), 0.024)
    tr = ObjectTracker(_RIG.frame, TrackerParams(search_start_s=5.0, search_max_s=60.0))
    q_all = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)
    tr.open_query("g", "q0", q_all, "k@0", stamp_ns=0)
    _, wall = step(tr, _RIG, [orange], stamp_ns=0, wall_s=0.0)
    assert tr.queries.get("g", "q0").status is ObjectQueryStatus.RESOLVED
    plate = ObjectQuery(label="plate", count="one", role=ObjectRole.DESTINATION)
    tr.open_query("g", "q1", plate, "k@0", stamp_ns=0)
    depth, _ = render([orange], _RIG)
    grid = grid_from_depth(depth, _RIG)
    labels = []
    for n in range(1, 7):
        wall += 1.0
        frame = CameraFrame(_RIG.name, n, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
        req = tr.feed(frame, grid=grid, grid_stamp_ns=n, wall_s=wall).requests[0]
        labels.append(req.label)
        tr.feed(frame, grid=grid, grid_stamp_ns=n, wall_s=wall,
                results=[TrackerResult(req.request_id)])  # fmt: skip
    assert labels.count("orange") >= 2 and labels.count("plate") >= 2, labels


def test_a_served_seed_prompt_moves_on_to_the_next_raised_top() -> None:
    """A map-seeded prompt that does not ground (here: answered empty) is not re-issued on
    the same cell; the next raised top in the seed gets it, then the detector (PickOrange:
    one seed cell monopolised every turn)."""
    from openral_core import PlaceRegion, Pose6D

    from tests.unit._tracker_scenes import sphere

    x, y = _RIG.centre_xy
    solids = [sphere((x + d, y, _RIG.support_z + 0.024), 0.024) for d in (-0.06, 0.06)]
    seed = PlaceRegion(
        frame_id=_RIG.frame,
        pose=Pose6D(xyz=(x, y, _RIG.support_z + 0.05), quat_xyzw=(0, 0, 0, 1), frame_id=_RIG.frame),
        half_extents=(0.12, 0.08, 0.07),
    )
    tr = ObjectTracker(_RIG.frame)
    q = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, seed=seed)
    tr.open_query("g", "q0", q, "k@0", stamp_ns=0)
    depth, _ = render(solids, _RIG)
    frame = CameraFrame(_RIG.name, 1, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    grid = grid_from_depth(depth, _RIG, shift_z=0.5 * 0.015)
    served = []
    for n in range(60):
        out = tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=float(n))
        req = out.requests[0]
        served.append(req.seed_cell if req.service == "segment_in_view" else "locate")
        tr.feed(frame, grid=grid, grid_stamp_ns=1, wall_s=float(n),
                results=[TrackerResult(req.request_id)])  # fmt: skip
    assert "locate" in served, served
    first = served[: served.index("locate")]
    assert first and len(first) == len(set(first)), served  # no seed cell twice in a round
