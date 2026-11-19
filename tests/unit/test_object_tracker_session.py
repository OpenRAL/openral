"""``TrackerSession``: goals, grounding wait, binding, release and ambiguous pruning (#418 D1).

Real manifests (``robots/droid_franka_robotiq``, ``robots/so101_follower``), the DROID
``exterior_depth`` rig ray-cast from the committed scene, the real tracker core and store.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachmentEvidenceKind,
    FitKind,
    GraspDeclaration,
    ObjectQuery,
    ObjectQueryStatus,
    ObjectRole,
    PlaceRegion,
    PlaceZone,
    Pose6D,
    RobotDescription,
    TrackStatus,
)
from openral_world_state.object_tracker import CameraFrame, TrackerParams
from openral_world_state.object_tracker.binding import Binding, BindOutcome
from openral_world_state.object_tracker.session import (
    GraspEventKind,
    TrackerSession,
    TrackingMode,
)

from tests.unit._object_tracker_fixtures import ROBOTS, arm_at_hold
from tests.unit._tracker_scenes import RES, answer, banana, droid_exterior, grid_from_depth, render

_S = 1_000_000_000
_RIG = droid_exterior()
_DX = np.array([0.004, 0.0, 0.0])
_BANANA = banana(_RIG.centre_xy, _RIG.support_z, 0.08, 0.018, 1.2, math.pi / 2)


def _decl(
    stamp_ns: int,
    *,
    label: str = "banana",
    active: bool = True,
    invocation_id: str = "",
    robot: str = "droid_franka_robotiq",
) -> GraspDeclaration:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    return GraspDeclaration(
        target_id=label,
        contact_links=d.grasp_contact_links(),
        timeout_s=60.0,
        stamp_ns=stamp_ns,
        active=active,
        invocation_id=invocation_id,
        target_queries=(ObjectQuery(label=label, count="one", role=ObjectRole.TARGET),),
    )


def _session(robot: str, wait_s: float = 3.0) -> TrackerSession:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    return TrackerSession(
        d, cell_m=RES, grounding_wait_s=wait_s, params=TrackerParams(), require_hands=True
    )


def _frame(s: TrackerSession, solids: list, stamp_ns: int, wall: float) -> float:  # type: ignore[type-arg]  # reason: Solid is the test rig's private type
    """One frame through the session, answering every request it raises on it."""
    depth, labels = render(solids, _RIG)
    frame = CameraFrame(_RIG.name, stamp_ns, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    s.on_grid(grid_from_depth(depth, _RIG), stamp_ns)
    out = s.on_frame(frame, wall_s=wall)
    for _ in range(8):
        if not out.requests:
            break
        wall += 0.6
        out = s.on_frame(frame, wall_s=wall, results=[answer(out.requests[0], labels)])
    return wall


def test_goal_grounds_logs_ready_and_rekeys_on_the_next_goal(cap: Any) -> None:
    s, wall = _session("droid_franka_robotiq"), 0.0
    s.on_grasp_declaration(_decl(100))
    for n in range(4):
        wall = _frame(s, [_BANANA], n * _S // 2, wall + 1)
    [track] = s.snapshot().objects
    assert track.status is TrackStatus.CONFIRMED and track.declaration_key == "banana@100"
    ready = cap.named("object_tracker.grounding_ready")
    assert [e["goal_id"] for e in ready] == ["grasp:banana@100"]
    revision = s.snapshot().revision
    s.on_grasp_declaration(_decl(200))  # the next goal: the persisting track follows it
    [track] = s.snapshot().objects
    assert track.declaration_key == "banana@200" and s.snapshot().revision > revision
    assert s.tracker.queries.get("grasp:banana@100", "q0") is None
    s.on_grasp_declaration(_decl(200, active=False))  # retraction closes the goal
    assert s.grasp_declaration is None and not s.tracker.queries.queries


def test_live_grasp_scope_rekeys_a_later_track_minted_by_deploy_query() -> None:
    """A declaration opened before perception owns the matching track once it is minted."""
    s = _session("droid_franka_robotiq")
    deploy_query = ObjectQuery(label="banana", count="one", role=ObjectRole.TARGET)
    s.open_standing_queries([deploy_query])
    depth, labels = render([_BANANA], _RIG)
    stamp_ns = _S
    frame = CameraFrame(_RIG.name, stamp_ns, depth, _RIG.k, _RIG.t, np.zeros(depth.shape, np.uint8))
    s.on_grid(grid_from_depth(depth, _RIG), stamp_ns)
    out = s.on_frame(frame, wall_s=1.0)
    [deploy_request] = out.requests
    assert deploy_request.query_key == ("deploy:deploy@0", "q0")

    # The grasp goal opens after the deploy request is in flight, before a track exists.
    invocation_a = "a" * 32
    s.on_grasp_declaration(_decl(100, invocation_id=invocation_a))
    wall_s = 2.0
    for _ in range(8):
        if s.snapshot().objects:
            break
        assert out.requests
        request = out.requests[0]
        out = s.on_frame(
            frame,
            wall_s=wall_s,
            results=[answer(request, labels)],
        )
        wall_s += 1.0
    [track] = s.snapshot().objects
    assert track.label == "banana" and track.role is ObjectRole.TARGET
    assert track.declaration_key == "banana@100"
    goal_a = f"grasp:banana@100#{invocation_a}"
    assert s.tracker.queries.ready(goal_a)

    # A replacement adopts that identity; closing it cannot leave the old scope live.
    invocation_b = "b" * 32
    s.on_grasp_declaration(_decl(200, invocation_id=invocation_b))
    [track] = s.snapshot().objects
    assert track.declaration_key == "banana@200"
    goal_b = f"grasp:banana@200#{invocation_b}"
    assert s.tracker.queries.get(goal_a, "q0") is None
    assert s.tracker.queries.get(goal_b, "q0") is not None
    s.on_grasp_declaration(_decl(200, active=False, invocation_id=invocation_b))
    [track] = s.snapshot().objects
    assert track.declaration_key == "deploy@0"


def test_same_goal_id_with_changed_queries_replaces_scope_and_clears_old_matches() -> None:
    s = _session("droid_franka_robotiq")
    centre = np.array([0.0, 0.0, 0.1])
    banana_id = _mint(s, centre, "banana")
    orange_id = _mint(s, centre + np.array([0.1, 0.0, 0.0]), "orange")
    invocation_id = "c" * 32

    s.on_grasp_declaration(_decl(100, invocation_id=invocation_id))
    banana = s.tracker.store.get(banana_id)
    orange = s.tracker.store.get(orange_id)
    assert banana is not None and banana.declaration_key == "banana@100"
    assert orange is not None and orange.declaration_key == ""

    changed = _decl(100, invocation_id=invocation_id).model_copy(
        update={
            "target_queries": (ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET),)
        }
    )
    s.on_grasp_declaration(changed)

    banana = s.tracker.store.get(banana_id)
    orange = s.tracker.store.get(orange_id)
    assert banana is not None and banana.declaration_key == ""
    assert orange is not None and orange.declaration_key == "banana@100"
    goal = s._goals["grasp"]
    assert goal is not None and goal.goal_id == f"grasp:banana@100#{invocation_id}"
    assert goal.queries[0].label == "orange"


def test_active_goal_without_matching_live_scope_clears_track_key() -> None:
    s = _session("droid_franka_robotiq")
    track_id = _mint(s, np.array([0.0, 0.0, 0.1]), "orange")

    s.on_grasp_declaration(_decl(100))

    track = s.tracker.store.get(track_id)
    assert track is not None and track.declaration_key == ""


def test_same_goal_id_with_empty_queries_retires_previous_scope() -> None:
    s = _session("droid_franka_robotiq")
    track_id = _mint(s, np.array([0.0, 0.0, 0.1]), "banana")
    invocation_id = "d" * 32
    declaration = _decl(100, invocation_id=invocation_id)
    s.on_grasp_declaration(declaration)
    track = s.tracker.store.get(track_id)
    assert track is not None and track.declaration_key == "banana@100"

    s.on_grasp_declaration(declaration.model_copy(update={"target_queries": ()}))

    track = s.tracker.store.get(track_id)
    assert track is not None and track.declaration_key == ""
    assert s._goals["grasp"] is None


def test_grounding_incomplete_is_logged_once_after_the_wait(cap: Any) -> None:
    s, wall = _session("droid_franka_robotiq", wait_s=1.0), 0.0
    s.on_grasp_declaration(_decl(1))
    for n in range(6):
        wall = _frame(s, [], n * _S // 2, wall + 1)
    [event] = cap.named("object_tracker.grounding_incomplete")
    assert event["grounding_required"] is False


def _mint(
    s: TrackerSession,
    xyz: NDArray[np.float64],
    oid_label: str = "orange",
    *,
    half: tuple[float, float, float] = (0.024, 0.024, 0.024),
    provisional: bool = False,
    fit_ns: int = 1,
) -> str:
    frame = s.frame_id
    pose = Pose6D(xyz=tuple(float(v) for v in xyz), quat_xyzw=(0, 0, 0, 1), frame_id=frame)
    t = s.tracker.store.mint(
        label=oid_label,
        role=ObjectRole.TARGET,
        declaration_key="orange@1",
        box=PlaceRegion(frame_id=frame, pose=pose, half_extents=half),
        tight=(),
        fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=float(xyz[2]) - half[2] - 0.001,
        support_ref="map",
        floor_z=float(xyz[2]) - half[2],
        visible_cells=6,
        occupied_cells=5,
        last_fit_stamp_ns=fit_ns,
        last_confirmed_stamp_ns=fit_ns,
        provisional=provisional,
    )
    s.tracker.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
    return t.object_id


def test_attach_binds_then_the_jaw_opening_releases() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    oid = _mint(s, centre)
    s.on_joint_state(q, 10)
    binding = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    assert binding is not None and binding.outcome is BindOutcome.TRACKED
    [att] = s.attachments()
    assert att.object_id == oid and att.evidence_kind is AttachmentEvidenceKind.TRACKED_OBJECT
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]
    changes = s.attachment_changes
    # The jaw opens past hold + release_gap and stays open for the dwell: released.
    hand = s.binders[joint].hand
    lo, hi = hand.joint.position_limits  # type: ignore[misc]  # reason: the SO-101 jaw has limits
    far = hi if abs(hi - hand.closed) > abs(lo - hand.closed) else lo
    s.on_joint_state({joint: far}, 30)
    assert s.attachments()  # not yet: the dwell runs on the source clock
    s.on_joint_state({joint: far}, 30 + int(0.31 * _S))
    assert s.binders[joint].held is None and s.attachment_changes > changes
    assert s.tracker.store.get(oid).status is TrackStatus.RELEASED  # type: ignore[union-attr]


def _released(s: TrackerSession) -> tuple[str, str, int]:
    """Bind the orange at the hold, then DETACH it: ``(joint, object_id, release stamp)``."""
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    oid = _mint(s, centre)
    s.on_joint_state(q, 10)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    s.on_grasp_event(GraspEventKind.DETACH, joint, jaw=jaw, hold=jaw, stamp_ns=_S)
    return joint, oid, _S


def test_a_released_binding_stays_on_the_wire_until_its_window_runs_out() -> None:
    s = _session("so101_follower")
    joint, oid, t0 = _released(s)
    [att] = s.attachments()  # frozen by the kernel, its box cleared by the voxel bridge
    assert (att.object_id, att.epoch) == (oid, 0) and s.binders[joint].held is None
    changes = s.attachment_changes
    s.on_joint_state({}, t0 + int(2.9 * _S))
    assert s.attachments() == [att]
    s.on_joint_state({}, t0 + 3 * _S)  # released_clear_timeout_s (3.0 s): dropped
    assert s.attachments() == [] and s.attachment_changes == changes + 1


def test_leaving_the_station_retires_released_records_and_world_tracks() -> None:
    """Station mode: a frozen released record and the world tracks are base-frame state; once
    the base drives they describe the wrong place. ``leave_station`` drops them at once (no
    3 s window) and records the change so the next ``attachment_state`` carries it."""
    s = _session("so101_follower")
    _, oid, _ = _released(s)
    other = _mint(s, np.array([0.3, 0.1, 0.05]))
    assert s.attachments() and s.tracker.store.get(other) is not None
    changes = s.attachment_changes
    dropped, released = s.leave_station()
    assert released == [oid] and set(dropped) == {oid, other}
    assert s.attachments() == [] and s.tracker.store.all() == []
    assert s.grid is None and s.attachment_changes == changes + 1


def test_leaving_the_station_forgets_where_targets_were_placed() -> None:
    """A placed record is a base-frame box: at the next station it would excuse a foreign
    body inside it from splitting that station's destination."""
    s = _session("so101_follower")
    pose = Pose6D(xyz=(0.3, 0.1, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="base_link")
    s.tracker._placed["placed"] = PlaceRegion(
        frame_id="base_link", pose=pose, half_extents=(0.02, 0.02, 0.02)
    )
    s.leave_station()
    assert s.tracker._placed == {}


def test_leaving_the_station_keeps_the_payload_the_hand_still_holds() -> None:
    """A held payload rides the hand, not the base: dropping it would publish "nothing carried"
    for a payload the robot still holds (the unsafe direction)."""
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    oid = _mint(s, centre)
    s.on_joint_state(q, 10)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    [held] = s.attachments()
    _mint(s, np.array([0.3, 0.1, 0.05]), "apple")
    dropped, released = s.leave_station()
    assert released == [] and oid not in dropped and len(dropped) == 1
    assert s.attachments() == [held] and s.binders[joint].held is not None
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def test_a_released_payload_is_frozen_where_it_was_let_go() -> None:
    """Spark restock 2026-11-01: a released record left on the hand rode the retreating hand
    1.0-1.5 s (up to 20 cm) in ``tracking``; it is re-expressed on the base frame at the
    release pose, the hand and jaws (its old touch links) still exempt."""
    s = _session("so101_follower")
    _, q, _ = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    oid = _mint(s, arm_at_hold("so101_follower")[2])
    s.on_joint_state(q, 10)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    [held] = s.attachments()
    binder = s.binders[joint]
    at_release = binder.object_pose(q)
    s.on_grasp_event(GraspEventKind.DETACH, joint, jaw=jaw, hold=jaw, stamp_ns=_S)
    [frozen] = s.attachments()
    assert frozen.object_id == oid and frozen.attach_link == s.frame_id
    assert frozen.released_by_link == held.attach_link, "the kernel's release key"
    assert held.attach_link in frozen.touch_links
    assert np.allclose(frozen.pose_in_link.xyz, at_release.xyz, atol=1e-9)
    assert frozen.primitives == held.primitives


def test_under_on_the_released_record_stays_as_held() -> None:
    """``on``: the exemption provider releases the hand-attached record inside the kernel,
    earlier, keyed on its attach link; a re-attached record would restart that release."""
    d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    s = TrackerSession(
        d,
        cell_m=RES,
        grounding_wait_s=3.0,
        params=TrackerParams(),
        require_hands=True,
        mode=TrackingMode.ON,
    )
    _, q, _ = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    _mint(s, arm_at_hold("so101_follower")[2])
    s.on_joint_state(q, 10)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    [held] = s.attachments()
    s.on_grasp_event(GraspEventKind.DETACH, joint, jaw=jaw, hold=jaw, stamp_ns=_S)
    assert s.attachments() == [held]


def test_a_jaw_open_release_freezes_where_the_jaw_opened_not_where_it_was_decided() -> None:
    """The jaw-open rule decides a dwell (0.3 s) after the opening; by then the retreat may
    have begun (review of #481, 2026-11-02). The record and the released track are frozen at
    the hand pose of the opening."""
    from tests.unit.test_object_tracker_binding import _hand_step

    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    oid = _mint(s, centre)
    s.on_joint_state(q, 10)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    binder = s.binders[joint]
    hand = binder.hand
    lo, hi = hand.joint.position_limits  # type: ignore[misc]  # reason: the SO-101 jaw has limits
    far = hi if abs(hi - hand.closed) > abs(lo - hand.closed) else lo
    opened = {**q, joint: far}
    at_opening = binder.object_pose(opened)
    s.on_joint_state({joint: far}, 30)
    s.on_joint_state({**_hand_step(binder, "so101_follower", q, 0.05), joint: far}, 30 + _S // 10)
    s.on_joint_state({joint: far}, 30 + int(0.31 * _S))  # decided, the hand 5 cm up
    [frozen] = s.attachments()
    assert frozen.released_by_link is not None
    assert np.allclose(frozen.pose_in_link.xyz, at_opening.xyz, atol=1e-9)
    track = s.tracker.store.get(oid)
    assert track is not None and track.status is TrackStatus.RELEASED
    assert np.allclose(track.box.pose.xyz, at_opening.xyz, atol=1e-6)


def test_a_released_track_retires_once_every_hand_is_clear_of_it() -> None:
    """Spark R281 (2026-10-23): the source-clock window ran 14 s of wall time and the frozen
    box blocked the regrasp. Once the hand is clear of the released box the window ends."""
    from tests.unit.test_object_tracker_binding import _hand_step

    s = _session("so101_follower")
    _, oid, t0 = _released(s)
    binder, q, _ = arm_at_hold("so101_follower")
    s.on_joint_state(q, t0 + int(0.5 * _S))
    assert s.tracker.store.get(oid) is not None, "the hand still at it: the window holds"
    s.on_joint_state(_hand_step(binder, "so101_follower", q, 0.20), t0 + _S)
    assert s.tracker.store.get(oid) is None and s.attachments() == []


def test_a_released_track_retires_at_the_kernels_separation() -> None:
    """Spark orange R370 (2026-10-23): the kernel closed the release window by FK separation
    (every hand link 4 cm clear) and held the frozen box as a plain obstacle; the tracker
    waited for its pinch volume to be 8 cm clear, so the release stayed on the wire and the
    regrasp of the same, never-lifted orange met the frozen box. The release retires at the
    first hand pose whose every collision box is ``tracked_release_clear_m`` (0.04) clear."""
    from openral_core.kinematics import box_gap_lower_bound_m
    from openral_world_state.object_tracker.session import _box

    from tests.unit.test_object_tracker_binding import _hand_step

    s = _session("so101_follower")
    _, oid, t0 = _released(s)
    binder, q, _ = arm_at_hold("so101_follower")
    box = _box(s.tracker.store.get(oid).box)  # type: ignore[union-attr]
    hand = s.binders[ROBOTS["so101_follower"][0]].hand
    for n in range(1, 40):
        qq = _hand_step(binder, "so101_follower", q, 0.005 * n)
        gap = min(box_gap_lower_bound_m(box, hb) for hb in hand.hand_boxes(s.frame_id, qq))
        s.on_joint_state(qq, t0 + n * _S // 100)
        if gap < 0.04:
            assert s.tracker.store.get(oid) is not None, (n, gap)
            continue
        assert s.tracker.store.get(oid) is None and s.attachments() == [], (n, gap)
        # The old rule: the pinch centre's distance to the box less the pinch half diagonal.
        centre, _, half = s.binders[ROBOTS["so101_follower"][0]].pinch_in_frame(qq)
        local = np.abs((centre - box[0]) @ box[1])
        old = np.linalg.norm(np.maximum(local - box[2], 0.0)) - np.linalg.norm(half)
        assert old < 0.08, "the 8 cm pinch rule would retire here too: no discrimination"
        return
    pytest.fail("the hand never got 4 cm clear")


@pytest.mark.parametrize("case", ["reacquired", "old neighbour", "other label"])
def test_a_released_track_retires_once_it_is_tracked_again_at_rest(case: str) -> None:
    """Spark R369 (2026-10-23): the orange dropped 3 cm onto the table and was re-tracked
    there as a new CONFIRMED track 1 s later; the box frozen in the air at the release pose
    stopped the arm carrying the next orange through it. A new track of the label touching
    the released box retires it. A track that was there before the release, or another
    label, is a neighbour, not the object: the window holds (the hand stays at it)."""
    s = _session("so101_follower")
    _, q, centre = arm_at_hold("so101_follower")
    below = centre - np.array([0.0, 0.0, 0.03])
    if case == "old neighbour":  # beside the held orange, its box touching, the pinch outside
        _mint(s, centre + np.array([0.046, 0.0, 0.0]))
    _, oid, t0 = _released(s)
    if case != "old neighbour":
        _mint(s, below, "orange" if case == "reacquired" else "plate")
    s.on_joint_state(q, t0 + int(0.5 * _S))
    gone = s.tracker.store.get(oid) is None
    assert gone is (case == "reacquired"), case
    assert (s.attachments() == []) is gone


def test_a_released_track_is_retired_with_its_window() -> None:
    """Never followed after release (Spark R177-R185): RELEASED for the window, then gone."""
    s = _session("so101_follower")
    _, oid, t0 = _released(s)
    track = s.tracker.store.get(oid)
    assert track is not None and track.status is TrackStatus.RELEASED
    s.on_joint_state({}, t0 + int(2.9 * _S))
    assert s.tracker.store.get(oid) is not None  # the kernel still freezes the payload
    s.on_joint_state({}, t0 + 3 * _S)
    assert s.tracker.store.get(oid) is None and oid not in {
        o.object_id for o in s.snapshot().objects
    }


def test_a_retired_release_leaves_its_box_as_the_querys_prior() -> None:
    """Spark R242-R261: retired, the released object is looked for where it was let go."""
    s = _session("so101_follower")
    s.open_standing_queries([ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET)])
    _, oid, t0 = _released(s)
    box = s.tracker.store.get(oid).box  # type: ignore[union-attr]
    [q] = s.tracker.queries.queries
    assert q.prior is None
    s.on_joint_state({}, t0 + 3 * _S)
    assert s.tracker.store.get(oid) is None and q.prior is not None
    for c, h, bc, bh in zip(
        q.prior.pose.xyz, q.prior.half_extents, box.pose.xyz, box.half_extents, strict=True
    ):
        assert c - h <= bc - bh + 1e-9 and bc + bh <= c + h + 1e-9  # the column holds the box


def test_leaving_the_station_clears_the_queries_release_priors() -> None:
    """A release prior is a base-frame column: at the next station it would aim the search at
    where the object lay relative to the old base pose. Neither a prior already retired at this
    station nor a still-frozen released record survives ``leave_station``."""
    for retired in (True, False):
        s = _session("so101_follower")
        s.open_standing_queries([ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET)])
        _, _, t0 = _released(s)
        [q] = s.tracker.queries.queries
        if retired:
            s.on_joint_state({}, t0 + 3 * _S)  # retired at this station: the prior is seeded
            assert q.prior is not None
        s.leave_station()
        assert q.prior is None, retired


def test_deploy_queries_ground_without_any_declaration(cap: Any) -> None:
    s, wall = _session("droid_franka_robotiq"), 0.0
    s.open_standing_queries([ObjectQuery(label="banana", count="one", role=ObjectRole.TARGET)])
    for n in range(4):
        wall = _frame(s, [_BANANA], n * _S // 2, wall + 1)
    [track] = s.snapshot().objects
    assert track.status is TrackStatus.CONFIRMED and track.declaration_key == "deploy@0"
    s.on_grasp_declaration(_decl(300))  # dispatch's goal re-keys it for the kernel
    assert s.snapshot().objects[0].declaration_key == "banana@300"


def test_ambiguous_union_drops_the_candidate_that_stayed_behind() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    a, b = _mint(s, centre + _DX), _mint(s, centre - _DX)
    s.on_joint_state(q, 10)
    binding = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    assert binding is not None and binding.outcome is BindOutcome.AMBIGUOUS
    [att] = s.attachments()
    assert att.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK and len(att.primitives) == 3
    # Post-lift re-fit: b is re-fitted where it was, 30 cm away from the lifted hand.
    stayed = s.tracker.store.get(b)
    assert stayed is not None
    moved = stayed.box.model_copy(
        update={"pose": stayed.box.pose.model_copy(update={"xyz": tuple(centre + 75 * _DX)})}
    )
    s.tracker.store.replace(stayed.model_copy(update={"box": moved, "last_fit_stamp_ns": 30}))
    s._prune_ambiguous()
    [att] = s.attachments()
    assert len(att.primitives) == 2 and att.geometry_revision == 1
    assert s.binders[joint].held.candidates == (a,)  # type: ignore[union-attr]


def test_untracked_grasp_restarts_the_live_search() -> None:
    _, q, _ = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    s.on_grasp_declaration(_decl(1, label="orange", robot="so101_follower"))
    assert s.tracker.queries.get("grasp:orange@1", "q0") is None  # no source stamp yet
    s.on_joint_state(q, 5 * _S)  # the first stamp starts the search, not stamp 0
    query = s.tracker.queries.get("grasp:orange@1", "q0")
    assert query is not None and query.since_ns == 5 * _S
    binding = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=6 * _S)
    assert binding is not None and binding.outcome is BindOutcome.UNTRACKED and binding.reground
    query = s.tracker.queries.get("grasp:orange@1", "q0")
    assert query is not None and query.since_ns == 6 * _S
    assert query.status is ObjectQueryStatus.UNRESOLVED


def test_unposed_grasp_event_binds_nothing() -> None:
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    assert s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=1) is None
    assert not s.attachments()


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_hand_spaces_cover_the_measured_hand(robot: str) -> None:
    _, q, _ = arm_at_hold(robot)
    s = _session(robot)
    assert s.hand_spaces() == []  # unposed: no hand space claimed
    s.on_joint_state(q, 1)
    spaces = s.hand_spaces()
    assert spaces and all(r.frame_id == s.frame_id for r in spaces)


def test_a_grasp_event_without_a_measured_jaw_uses_the_newest_sample() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, _ = ROBOTS["so101_follower"]
    s = _session("so101_follower")
    oid = _mint(s, centre)
    nan = float("nan")
    assert s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=nan, hold=nan, stamp_ns=5) is None
    s.on_joint_state(q, 10)  # the measured jaw at the hold
    binding = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=nan, hold=nan, stamp_ns=20)
    assert binding is not None and binding.attachment.object_id == oid


def test_tracking_relays_the_workspace_box_while_the_map_confirms_it() -> None:
    """With the HAL producer stood down (tracking/on) the tracker confirms the declaration's
    workspace_box on its own grid, the producer's rule, and stamps it with the grid."""
    s = _session("droid_franka_robotiq")
    x, y = _RIG.centre_xy
    ws = PlaceRegion(
        frame_id=s.frame_id,
        pose=Pose6D(xyz=(x, y, _RIG.support_z + 0.05), quat_xyzw=(0, 0, 0, 1), frame_id=s.frame_id),
        half_extents=(0.15, 0.15, 0.08),
    )
    s.on_grasp_declaration(_decl(1).model_copy(update={"workspace_box": ws}))
    assert s.workspace_region() is None  # no grid yet
    depth, _ = render([_BANANA], _RIG)
    s.on_grid(grid_from_depth(depth, _RIG), 7 * _S)
    held = s.workspace_region()
    assert held is not None and held.stamp_ns == 7 * _S and held.half_extents == ws.half_extents
    s.on_grid(grid_from_depth(render([], _RIG)[0], _RIG, shift_z=10.0), 8 * _S)  # map empty
    assert s.workspace_region() is None


def _search_box(frame: str, centre: NDArray[np.float64], half: float = 0.06) -> PlaceRegion:
    pose = Pose6D(
        xyz=tuple(float(v) for v in centre), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame
    )
    return PlaceRegion(frame_id=frame, pose=pose, half_extents=(half, half, half))


def _grasp_region_rig(
    mode: TrackingMode = TrackingMode.TRACKING, box: bool = True
) -> tuple[TrackerSession, str, dict[str, float], NDArray[np.float64]]:
    """SO-101 at its hold, one CONFIRMED orange there, a grid, the orange declared at 1
    (with a ``search_box`` around it unless ``box`` is False)."""
    from tests.unit.test_object_tracker_binding import _grid

    _, q, centre = arm_at_hold("so101_follower")
    s = _session("so101_follower")
    s.mode = mode
    oid = _mint(s, centre)
    s.on_grid(_grid(s.frame_id, centre[None, :], centre - 0.3, 60), 5)
    s.on_joint_state(q, 6)
    decl = _decl(1, label="orange", robot="so101_follower")
    if box:
        decl = decl.model_copy(update={"search_box": _search_box(s.frame_id, centre)})
    s.on_grasp_declaration(decl)
    return s, oid, q, centre


def test_the_grasp_region_is_the_target_box_only_while_it_lies_inside_the_search_box() -> None:
    """`tracking`: a track narrows the declaration, never widens it. The region is the one
    target's box while all of it lies inside the declared ``search_box`` (the declared box
    itself would allow more than `off`, whose HAL leg measures a region inside it)."""
    s, oid, _, _ = _grasp_region_rig()
    track = s.tracker.store.get(oid)
    assert track is not None and track.declaration_key == "orange@1"
    decl = s.grasp_declaration
    assert decl is not None and decl.search_box is not None
    region = s.grasp_region()
    assert region is not None
    assert region.pose == track.box.pose and region.half_extents == track.box.half_extents
    assert max(region.half_extents) <= max(decl.search_box.half_extents)
    assert region.stamp_ns == track.last_confirmed_stamp_ns
    assert region.evidence_ref == f"tracked:{oid}|epoch={track.epoch}|within:search_box"
    GraspDeclaration.model_validate(  # within the declaration's caps: the kernel takes it
        {**decl.model_dump(), "region": region}
    )
    s.on_grasp_declaration(_decl(1, label="orange", robot="so101_follower", active=False))
    assert s.grasp_region() is None, "no declaration, no region"


def test_no_declared_search_box_is_no_grasp_region_in_any_mode() -> None:
    """A target track alone never makes a region: `tracking` fails closed without a declared
    box; `on` publishes none (its provider keys on track ids)."""
    for mode in TrackingMode:
        s, _, _, _ = _grasp_region_rig(mode, box=False)
        assert s.grasp_region() is None, mode


def test_on_publishes_no_grasp_region_even_with_a_search_box() -> None:
    s, _, _, _ = _grasp_region_rig(TrackingMode.ON)
    assert s.grasp_region() is None


def test_the_grasp_region_fails_closed_on_doubt() -> None:
    s, oid, _, centre = _grasp_region_rig()
    store = s.tracker.store
    track = store.get(oid)
    assert track is not None
    for status in (TrackStatus.TENTATIVE, TrackStatus.SPLIT, TrackStatus.LOST):
        store.replace(track.model_copy(update={"status": status}))
        assert s.grasp_region() is None, status
    store.replace(track.model_copy(update={"status": TrackStatus.OCCLUDED}))
    assert s.grasp_region() is not None, "the map holds an occluded target"
    other = _mint(s, centre + np.array([0.03, 0.0, 0.0]))
    s.on_grasp_declaration(s.grasp_declaration)  # type: ignore[arg-type]  # rekeys it
    assert store.get(other).declaration_key == "orange@1"  # type: ignore[union-attr]
    assert s.grasp_region() is None, "two targets in the box: which is not the tracker's to guess"
    store.drop(other)
    assert s.grasp_region() is not None
    store.replace(track.model_copy(update={"status": TrackStatus.CONFIRMED}))
    grid = s.grid
    assert grid is not None and s.grasp_region() is not None
    s.grid = dataclasses.replace(grid, frame_id="world")  # the box is in the old grid frame
    assert s.grasp_region() is None, "a box off the grid frame"
    s.grid = grid
    # Nothing in the box: a track outside it confirms nothing.
    far = track.box.model_copy(
        update={
            "pose": track.box.pose.model_copy(
                update={
                    "xyz": (
                        track.box.pose.xyz[0] + 0.5,
                        track.box.pose.xyz[1],
                        track.box.pose.xyz[2],
                    )
                }
            )
        }
    )
    store.replace(track.model_copy(update={"box": far}))
    assert s.grasp_region() is None, "the target is not in the declared box"
    store.replace(track)
    # A target box reaching past the declared box: the track would widen it, so no region.
    decl = s.grasp_declaration
    assert decl is not None and decl.search_box is not None
    h = decl.search_box.half_extents
    wide = track.box.model_copy(update={"half_extents": (h[0] + 0.01, 0.01, 0.01)})
    store.replace(track.model_copy(update={"box": wide}))
    assert s.grasp_region() is None, "the track box reaches outside the search_box"
    # Past the caps (a model_copy skips the cap checks): never the kernel's.
    big_box = decl.search_box.model_copy(update={"half_extents": (0.19, 0.19, 0.19)})
    s.grasp_declaration = decl.model_copy(update={"search_box": big_box})
    big = track.box.model_copy(update={"half_extents": (0.18, 0.18, 0.18)})
    store.replace(track.model_copy(update={"box": big}))
    assert s.grasp_region() is None, "past the volume cap"


def test_the_grasp_region_rides_through_the_hold_for_the_kernels_latch() -> None:
    """The region keeps being published once the target is held, as a HAL-measured one is:
    the kernel latches the box the first attachment snapshot carries, ignores later moves,
    and retires the declaration once the payload leaves it (ADR-0115's handover)."""
    s, oid, _, _ = _grasp_region_rig()
    joint, _, jaw = ROBOTS["so101_follower"]
    before = s.grasp_region()
    assert before is not None
    binding = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=20)
    assert binding is not None and binding.outcome is BindOutcome.TRACKED
    held = s.grasp_region()
    assert held is not None and f"tracked:{oid}|" in held.evidence_ref


def test_several_targets_are_narrowed_to_the_one_in_the_search_box() -> None:
    """`tracking`: the declaration's own ``search_box`` picks the declared orange; one
    outside it is not the declaration's (AMR aisle, 2026-11-16). Two inside stays
    fail-closed."""
    s, oid, _, centre = _grasp_region_rig()
    other = _mint(s, centre + np.array([0.12, 0.0, 0.0]))
    s.on_grasp_declaration(s.grasp_declaration)  # type: ignore[arg-type]  # rekeys
    assert s.tracker.store.get(other).declaration_key == ""  # type: ignore[union-attr]
    assert s.tracker.store.get(oid).declaration_key == "orange@1"  # type: ignore[union-attr]
    region = s.grasp_region()
    assert region is not None and f"tracked:{oid}|" in region.evidence_ref
    decl = s.grasp_declaration
    assert decl is not None and decl.search_box is not None
    wide = decl.search_box.model_copy(update={"half_extents": (0.15, 0.15, 0.15)})
    s.on_grasp_declaration(decl.model_copy(update={"search_box": wide}))
    assert s.grasp_region() is None, "both oranges inside the search box"


def test_a_named_memory_node_binds_only_its_instance_through_tf2() -> None:
    """Issue #631: a goal naming one memory node seeds its TARGET query with the node's box in
    ``map``. With two same-label oranges tracked, only the one centred in it, re-expressed in
    the grid frame through tf2, is the goal's, never the neighbour. The scope follows tf2 per
    grid (the base moves under ``map``); a lookup that misses keeps the last resolved box (one
    late tf never resets the query); a scope that never resolved matches nothing."""
    s, oid, _, centre = _grasp_region_rig(box=False)
    other = _mint(s, centre + np.array([0.12, 0.0, 0.0]))
    offset = np.array([1.0, -0.5, 0.0])  # grid <- map: a pure translation
    shift = np.eye(4)
    shift[:3, 3] = offset
    s.zone_transform = lambda frame, _ns: shift if frame == "map" else None
    decl = s.grasp_declaration
    assert decl is not None
    (query,) = decl.target_queries
    seeded = query.model_copy(update={"seed": _search_box("map", centre - offset)})
    named = decl.model_copy(update={"target_queries": (seeded,)})
    s.on_grasp_declaration(named)
    store = s.tracker.store
    assert store.get(oid).declaration_key == "orange@1"  # type: ignore[union-attr]
    assert store.get(other).declaration_key == ""  # type: ignore[union-attr]
    (standing,) = s.tracker.queries.queries
    assert standing.matches(store.get(oid))  # type: ignore[arg-type]
    assert not standing.matches(store.get(other))  # type: ignore[arg-type]

    # The base drives 0.12 m: the same map box now lies over the other orange.
    shift[:3, 3] = offset + np.array([0.12, 0.0, 0.0])
    s.on_grid(s.grid, s.grid_stamp_ns + 1)  # type: ignore[arg-type]
    assert standing.matches(store.get(other)) and not standing.matches(store.get(oid))  # type: ignore[arg-type]

    s.zone_transform = lambda _frame, _ns: None  # tf2 lost
    s.on_grid(s.grid, s.grid_stamp_ns + 1)  # type: ignore[arg-type]
    assert standing.matches(store.get(other)) and not standing.matches(store.get(oid))  # type: ignore[arg-type]

    # A new goal whose scope never resolves (no tf2 yet) matches neither: fail closed.
    renamed = named.model_copy(update={"stamp_ns": named.stamp_ns + 1})
    s.on_grasp_declaration(renamed)
    (fresh,) = s.tracker.queries.queries
    assert not fresh.matches(store.get(oid)) and not fresh.matches(store.get(other))  # type: ignore[arg-type]


def test_a_base_move_discards_the_station_so_a_memory_node_never_binds_a_stale_track() -> None:
    """Issue #651: tracks are base-frame boxes from where the base stood at their fit, while a
    memory node's scope is re-expressed where the base stands now. The base moves between
    the tracks' birth and the match, by the two oranges' spacing: the node's box then lies
    over the NEIGHBOUR's stale box. Past the kernel's station threshold the session discards
    what the station measured, so no stale track is matched (fail closed), and the tracks
    measured from the new pose bind the node's own orange. Under the threshold they stay."""
    from openral_world_state.object_tracker.session import STATION_MOTION_M, STATION_ROTATION_RAD

    from tests.unit.test_object_tracker_binding import _grid

    d = RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml")
    assert d.base_joints is not None
    forward, side, yaw = d.base_joints[:3]
    s = TrackerSession(d, cell_m=RES, grounding_wait_s=3.0, params=TrackerParams())
    s.mode = TrackingMode.TRACKING
    centre, spacing = np.array([0.5, 0.0, 0.4]), np.array([0.12, 0.0, 0.0])
    base = {forward: 0.0, side: 0.0, yaw: 0.0}
    offset = np.array([1.0, -0.5, 0.0])  # grid <- map with the base at its first pose

    def tf(frame: str, _stamp_ns: int) -> NDArray[np.float64] | None:
        t = np.eye(4)  # the base drives along its own x: map points move back in its frame
        t[:3, 3] = offset - np.array([base[forward], 0.0, 0.0])
        return t if frame == "map" else None

    def grid(stamp_ns: int) -> None:
        s.on_grid(_grid(s.frame_id, centre[None, :], centre - 0.3, 60), stamp_ns)

    s.zone_transform = tf
    assert s.on_joint_state(base, 1) is False
    oid, other = _mint(s, centre), _mint(s, centre + spacing)
    grid(5)
    seed = _search_box("map", centre - offset)  # the node: the first orange, in map
    query = ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET, seed=seed)
    named = _decl(1, label="orange", robot="openamrobot_v2").model_copy(
        update={"target_queries": (query,)}
    )
    s.on_grasp_declaration(named)
    (standing,) = s.tracker.queries.queries
    store = s.tracker.store
    assert standing.matches(store.get(oid))  # type: ignore[arg-type]
    assert not standing.matches(store.get(other))  # type: ignore[arg-type]

    # Settle noise under the thresholds: the station holds.
    base[forward], base[yaw] = 0.9 * STATION_MOTION_M, 0.9 * STATION_ROTATION_RAD
    assert s.on_joint_state(base, 6) is False
    assert {t.object_id for t in store.all()} == {oid, other}

    # The base backs up by the spacing: the node's box now lies over the neighbour's old box.
    base[forward], base[yaw] = -float(spacing[0]), 0.0
    assert s.on_joint_state(base, 7) is True
    assert store.all() == [], "no track from before the move is left to match"
    grid(8)
    assert standing.within is not None
    assert standing.within.pose.xyz == pytest.approx(tuple(centre + spacing))

    # Measured again from the new pose, the node's own orange is the goal's; its neighbour is not.
    again, neighbour = _mint(s, centre + spacing), _mint(s, centre + 2 * spacing)
    assert s.on_joint_state(base, 9) is False
    s.on_grasp_declaration(named)  # rekeys
    assert standing.matches(store.get(again))  # type: ignore[arg-type]
    assert not standing.matches(store.get(neighbour))  # type: ignore[arg-type]
    assert store.get(again).declaration_key == "orange@1"  # type: ignore[union-attr]
    assert store.get(neighbour).declaration_key == ""  # type: ignore[union-attr]

    # A turn past the rotation threshold leaves the station as well.
    base[yaw] = 1.5 * STATION_ROTATION_RAD
    assert s.on_joint_state(base, 10) is True and store.all() == []


def test_a_fixed_base_never_leaves_its_station() -> None:
    """No ``base_joints``: joint states never discard the tracks (issue #651's rule is a
    mobile base's)."""
    s, oid, q, _ = _grasp_region_rig()
    assert s.on_joint_state(q, 7) is False
    assert s.tracker.store.get(oid) is not None


def test_the_runners_search_box_seed_does_not_scope_the_goal_to_an_instance() -> None:
    """Review of #638: the runner seeds an unseeded TARGET query with the scene's search box.
    That box is the declaration's own ``search_box`` (scoped, or logged unscoped, as such), not
    a named instance: it never becomes a tf2-gated instance scope."""
    s, oid, _, centre = _grasp_region_rig(box=False)
    decl = s.grasp_declaration
    assert decl is not None
    box = _search_box("map", centre)
    (query,) = decl.target_queries
    seeded = query.model_copy(update={"seed": box})
    s.on_grasp_declaration(decl.model_copy(update={"search_box": box, "target_queries": (seeded,)}))
    (standing,) = s.tracker.queries.queries
    assert standing.within is None  # the off-frame search box is logged unscoped
    assert standing.matches(s.tracker.store.get(oid))  # type: ignore[arg-type]


def test_a_count_all_declaration_makes_every_orange_eligible_and_nothing_else() -> None:
    """r7po_on1..3: the policy grasps whichever orange it picks, while the declaration names
    ``obj:orange_1``. Every live track of the declaration's own query (label + role) carries its
    key, so the kernel's K2 (``tracked_grasp_eligible n=3 of=5``) exempts the one the hand
    reaches; a plate (another role) and a lost track do not follow it."""
    _, _, centre = arm_at_hold("so101_follower")
    s = _session("so101_follower")
    oranges = [_mint(s, centre + np.array([0.06 * k, 0.0, 0.0])) for k in range(3)]
    plate = _mint(s, centre + np.array([0.0, 0.3, 0.0]), "plate")
    store = s.tracker.store
    store.replace(store.get(plate).model_copy(update={"role": ObjectRole.DESTINATION}))  # type: ignore[union-attr]
    d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    decl = GraspDeclaration(
        target_id="obj:orange_1",
        contact_links=d.grasp_contact_links(),
        timeout_s=60.0,
        stamp_ns=300,
        active=True,
        target_queries=(ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),),
    )
    s.on_grasp_declaration(decl)
    keys = {t.object_id: t.declaration_key for t in store.all()}
    assert all(keys[o] == "obj:orange_1@300" for o in oranges), keys
    assert keys[plate] != "obj:orange_1@300", keys


# ── The tracker's own grasp event (tracking/on, sim only; #418 2026-10-17) ────────────────────────


def _derived_session(robot: str, derive: bool = True, *, max_age_s: float = 2.0) -> TrackerSession:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    return TrackerSession(
        d,
        cell_m=RES,
        grounding_wait_s=3.0,
        params=TrackerParams(max_age_s=max_age_s),
        require_hands=True,
        derive_attach=derive,
    )


def _hold_for(
    s: TrackerSession,
    joint: str,
    q: dict[str, float],
    jaw: float,
    *,
    command: float,
    t0: int = 10 * _S,
    seconds: float = 2.0,
    hz: int = 15,
) -> None:
    """Joint states at ``hz`` with the jaw chattering ±0.02 about ``jaw`` and a steady command."""
    s.on_jaw_command(joint, command, t0)
    for i in range(int(seconds * hz)):
        wobble = 0.02 * (1 if i % 2 else -1)
        s.on_joint_state({**q, joint: jaw + wobble}, t0 + int(i * _S / hz))


def test_a_stalled_chattering_jaw_binds_exactly_as_an_attach_does() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower")
    oid = _mint(s, centre)
    track = s.tracker.store.get(oid)
    assert track is not None
    s.tracker.store.replace(
        track.model_copy(update={"last_fit_stamp_ns": 10 * _S, "last_confirmed_stamp_ns": 10 * _S})
    )
    _hold_for(s, joint, q, jaw, command=0.10)
    binding = s.binders[joint].held
    assert binding is not None and binding.outcome is BindOutcome.TRACKED
    [att] = s.attachments()
    assert att.object_id == oid and att.evidence_kind is AttachmentEvidenceKind.TRACKED_OBJECT
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def test_a_derived_bind_is_not_doubled_by_the_hals_attach_for_the_same_close() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower")
    oid = _mint(s, centre)
    track = s.tracker.store.get(oid)
    assert track is not None
    s.tracker.store.replace(
        track.model_copy(update={"last_fit_stamp_ns": 10 * _S, "last_confirmed_stamp_ns": 10 * _S})
    )
    _hold_for(s, joint, q, jaw, command=0.10)
    changes, held = s.attachment_changes, s.binders[joint].held
    again = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=13 * _S)
    assert again is held and s.attachment_changes == changes and len(s.attachments()) == 1
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def test_the_hals_attach_first_leaves_nothing_for_the_derived_rule_to_do() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower")
    _mint(s, centre)
    s.on_joint_state(q, 9 * _S)
    s.on_jaw_command(joint, 0.10, 9 * _S)
    first = s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=10 * _S)
    changes = s.attachment_changes
    _hold_for(s, joint, q, jaw, command=0.10, t0=10 * _S + 1)
    assert s.binders[joint].held is first and s.attachment_changes == changes


def test_a_jaw_that_closes_on_nothing_binds_nothing() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, _ = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower")
    _mint(s, centre)
    _hold_for(s, joint, q, 0.10, command=0.10)  # measured at the command: nothing between the pads
    assert s.binders[joint].held is None and not s.attachments()


def test_the_derived_rule_is_off_unless_asked() -> None:
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower", derive=False)  # every real HAL (object_tracking off)
    _mint(s, centre)
    _hold_for(s, joint, q, jaw, command=0.10)
    assert s.binders[joint].held is None and not s.attachments()


def test_hands_are_clear_of_targets_only_away_from_every_target_and_holding_nothing() -> None:
    """Spark banana R336 / R338 / R340 (2026-10-23): asked with the open fingers still on the
    banana in the bowl, the completion judge said done; the rubric needs the robot clear of it.
    The completion check asks only once ``hands_clear_of_targets`` (published on
    ``/openral/object_tracker/hands_clear``) says so: no hold, every hand clear of every
    target, and a target to be clear of."""
    _, q, centre = arm_at_hold("droid_franka_robotiq")
    joint, _, jaw = ROBOTS["droid_franka_robotiq"]
    s = _derived_session("droid_franka_robotiq")
    s.on_joint_state({**q, joint: jaw}, 1)
    assert not s.hands_clear_of_targets(), "no target: nothing says the hand let go"
    _mint(s, centre, "banana")
    assert not s.hands_clear_of_targets(), "the pinch is at the target"
    far = _mint(s, centre + np.array([0.0, 0.0, 0.15]), "banana")  # 15 cm: off the hand
    s.tracker.store.drop(next(t.object_id for t in s.tracker.store.all() if t.object_id != far))
    assert s.hands_clear_of_targets(), "the only target is 15 cm away: off the hand"


def test_the_session_clock_follows_a_source_clock_reset() -> None:
    """Review of #479: jaw commands are stamped no earlier than ``now_ns``; a running maximum
    kept across a sim reset put every command ahead of every jaw sample, so the stall-derived
    ATTACH never fired. A late message (under a second back) does not move it."""
    s = _session("so101_follower")
    _, q, _ = arm_at_hold("so101_follower")
    s.on_joint_state(q, 100 * _S)
    s.on_joint_state(q, 100 * _S - _S // 2)  # late, not a reset
    assert s.now_ns == 100 * _S
    s.on_joint_state(q, 2 * _S)  # the sim reset: source time restarts
    assert s.now_ns == 2 * _S


def test_input_queued_before_a_sim_reset_neither_lifts_the_clock_nor_blocks_new_grids() -> None:
    """Review of #479: after the joint stamps go back, a frame or grid still queued from before
    the reset carries the old (larger) time. It must not lift ``now_ns`` back into the old
    epoch, and the newest-grid gate must not refuse every new grid until sim time passes the
    old one."""
    s = _session("droid_franka_robotiq")
    _, q, _ = arm_at_hold("droid_franka_robotiq")
    depth, _ = render([], _RIG)
    grid = grid_from_depth(depth, _RIG)
    s.on_joint_state(q, 100 * _S)
    s.on_grid(grid, 100 * _S)
    s.on_joint_state(q, 2 * _S)  # the sim reset
    assert s.grid is None and s.grid_stamp_ns == 0
    s.on_grid(grid, 99 * _S)  # queued before the reset
    assert s.now_ns == 2 * _S
    assert s.grid is None
    s.on_grid(grid, 2 * _S)  # the new epoch's first grid
    assert s.grid is grid and s.grid_stamp_ns == 2 * _S
    assert s.now_ns == 2 * _S


def test_a_grid_a_second_behind_the_joint_states_does_not_move_the_session_clock_back() -> None:
    """Review of #479: only a joint state resets the clock. A frame or grid that lags (the node
    holds a frame up to a second for its transform) is late, not a sim reset: read as one, the
    rest of that step ran, and published its snapshot, at the older time."""
    s = _session("droid_franka_robotiq")
    _, q, _ = arm_at_hold("droid_franka_robotiq")
    s.on_joint_state(q, 100 * _S)
    depth, _ = render([], _RIG)
    s.on_grid(grid_from_depth(depth, _RIG), 98 * _S)
    assert s.now_ns == 100 * _S


def test_instance_cells_are_applied_only_on_an_axis_aligned_lattice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #521: ``InstanceCells`` carries no orientation and ``_vision_fit`` reads the
    lattice's z as the frame's, so on a rotated OccupancyVoxels lattice the cells would land on
    the wrong voxels. The session refuses them there, and applies them on an aligned one."""
    from openral_world_state.object_tracker import session as session_mod

    events: list[str] = []

    class _Rec:
        def __getattr__(self, level: str) -> Any:  # reason: any log level
            return lambda event, **_: events.append(event)

    monkeypatch.setattr(session_mod, "log", _Rec())
    s = _session("droid_franka_robotiq")
    depth, _ = render([], _RIG)
    aligned = grid_from_depth(depth, _RIG)
    half = math.radians(10.0) / 2.0
    rotated = dataclasses.replace(
        aligned, orientation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half))
    )
    for grid, refused in ((rotated, True), (aligned, False)):
        events.clear()
        s.on_grid(grid, 1)
        s.on_instance_cells(1, grid.frame_id, grid.origin, grid.resolution, {})
        s._apply_instance_cells(())
        assert ("object_tracker.instance_cells_lattice_rotated" in events) is refused, events


def test_held_attachments_are_the_hands_payloads_without_released_records() -> None:
    """Station perception off publishes ``held_attachments``: a released record (frozen in the
    base frame) is not among them; a payload still in a hand is."""
    s = _session("so101_follower")
    _released(s)
    assert s.attachments() and s.held_attachments() == []
    _, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    _mint(s, centre, "apple")
    s.on_joint_state(q, 2 * _S)
    s.on_grasp_event(GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=2 * _S + 10)
    held = s.held_attachments()
    assert len(held) == 1 and held[0] in s.attachments()
    assert held[0].released_by_link is None


_SHELF_CARTON = np.array([0.801, 0.199, 0.809])  # object_tracker.mint, AMR aisle 2026-11-16
_TOTE_CARTON = np.array([0.504, 0.085, 0.457])  # the scene's search_box centre (the tote)


@pytest.mark.parametrize("mode", [TrackingMode.TRACKING, TrackingMode.ON])
def test_only_the_carton_in_the_search_box_grounds_the_goals_target_query(
    mode: TrackingMode,
) -> None:
    """The runner's merged declaration (the reasoner's ``box`` query, the scene's tote box):
    under `tracking` a same-label carton on the shelf neither resolves the query nor is keyed
    to the declaration; the one in the tote does both, and is the grasp region. `on` keeps
    every carton of the class keyed (its provider exempts the one the policy picks)."""
    s = TrackerSession(
        RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml"),
        cell_m=RES,
        grounding_wait_s=3.0,
        params=TrackerParams(),
        mode=mode,
    )
    assert s.frame_id == "base_footprint"
    s.on_joint_state({}, 1)  # a source stamp: the goal's queries open at it
    shelf = _mint(s, _SHELF_CARTON, "box")
    decl = GraspDeclaration(
        target_id="obj:box",
        contact_links=("openarm_left_ee_link1",),
        timeout_s=70.0,
        stamp_ns=5,
        search_box=_search_box(s.frame_id, _TOTE_CARTON, half=0.08),
        target_queries=(ObjectQuery(label="box", count="one", role=ObjectRole.TARGET),),
    )
    s.on_grasp_declaration(decl)
    book, store = s.tracker.queries, s.tracker.store
    [query] = book.queries
    book.update(store.all(), now_ns=6)
    if mode is TrackingMode.ON:
        assert store.get(shelf).declaration_key == "obj:box@5"  # type: ignore[union-attr]
        assert query.status is ObjectQueryStatus.RESOLVED
        return
    assert store.get(shelf).declaration_key == ""  # type: ignore[union-attr]
    assert query.status is ObjectQueryStatus.UNRESOLVED, "the shelf carton ended the search"
    tote = _mint(s, _TOTE_CARTON, "box")
    s.on_grasp_declaration(decl)  # rekeys, as every fed frame does
    book.update(store.all(), now_ns=7)
    assert store.get(tote).declaration_key == "obj:box@5"  # type: ignore[union-attr]
    assert store.get(shelf).declaration_key == ""  # type: ignore[union-attr]
    assert book.ready(query.goal_id), "the tote carton resolves the query"


# AMR aisle 2026-11-16 (`on`, Pro provider): the scene's initial pose, the left jaw at the logged
# hold (`object_tracker.bound ... hand=left_gripper jaw=0.110`), the base docked (base joints 0).
_AMR_JAW = 0.110
_AMR_HOLD = {
    "base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0, "lift_joint": 0.0,
    "left_joint1": 0.53, "left_joint2": -0.122, "left_joint3": -0.13, "left_joint4": 1.797,
    "left_joint5": -0.13, "left_joint6": -0.127, "left_joint7": -0.574, "left_gripper": _AMR_JAW,
    "right_joint1": -0.661, "right_joint2": 0.2, "right_joint3": 0.159, "right_joint4": 1.917,
    "right_joint5": 0.074, "right_joint6": 0.089, "right_joint7": 0.642, "right_gripper": -0.768,
}  # fmt: skip
_CARTON_HALF = (0.03, 0.03, 0.04)


def _amr_tote_grasp(
    mode: TrackingMode, *, provisional: bool
) -> tuple[TrackerSession, str, Binding | None]:
    """The AMR case: station perception switched on (the runner's SetBool edge), one carton in
    the left pinch, the HAL's ATTACH at the logged jaw. ``(session, track id, binding)``."""
    s = TrackerSession(
        RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml"),
        cell_m=RES,
        grounding_wait_s=20.0,
        params=TrackerParams(),
        require_hands=True,
        mode=mode,
    )
    s.leave_station()  # `_on_station_perception(True)`: the previous station is discarded
    s.on_joint_state(_AMR_HOLD, _S)
    binder = s.binders["left_gripper"]
    centre = binder.pinch_in_frame(_AMR_HOLD)[0]
    oid = _mint(s, centre, "box", half=_CARTON_HALF, provisional=provisional, fit_ns=_S)
    binding = s.on_grasp_event(
        GraspEventKind.ATTACH, "left_gripper", jaw=_AMR_JAW, hold=_AMR_JAW, stamp_ns=_S + 10
    )
    return s, oid, binding


def test_amr_provisional_sole_candidate_binds_the_pinch_stand_in_in_either_mode() -> None:
    """AMR aisle 2026-11-16: ``candidates=('trk:1',) ... outcome=provisional`` under `on`. The
    outcome is the candidate's own flag (``GraspBinder.on_attach``), never the mode: `tracking`
    binds the same carton the same way, the same record. A provisional box may under-contain
    what the robot hid (A4), so it is never the payload alone: the record is the pinch stand-in
    unioned with the carton's box (``geometry_source="measured"``, one sole candidate)."""
    records = []
    for mode in (TrackingMode.ON, TrackingMode.TRACKING):
        s, oid, binding = _amr_tote_grasp(mode, provisional=True)
        assert binding is not None and binding.outcome is BindOutcome.PROVISIONAL
        assert binding.candidates == (oid,) and binding.reground
        [att] = s.attachments()
        assert att.object_id.startswith("pinch:left_gripper@")
        assert att.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
        assert att.evidence_ref == f"provisional:{oid}"
        assert [p.geometry_source for p in att.primitives][:1] == ["fallback"]
        assert "measured" in [p.geometry_source for p in att.primitives[1:]]
        # Never BOUND: the store keeps the carton free, a held candidate (never aged).
        assert s.tracker.store.get(oid).status is TrackStatus.CONFIRMED  # type: ignore[union-attr]
        records.append(att.model_dump(exclude={"object_id"}))
    assert records[0] == records[1], "the bind is mode-independent"


def test_amr_provisional_hold_is_never_promoted_after_the_bind() -> None:
    """No post-bind confirmation exists: a clear fit of the candidate after the ATTACH (in the
    hand every fit sees the gripper past the occluder band, so A4 keeps it provisional anyway)
    leaves the hold PROVISIONAL until release. The station-off edge (the runner's goal end)
    keeps the held stand-in and its candidate."""
    s, oid, _ = _amr_tote_grasp(TrackingMode.ON, provisional=True)
    [held] = s.attachments()
    store = s.tracker.store
    store.replace(store.get(oid).model_copy(update={"provisional": False}))  # type: ignore[union-attr]
    s.on_joint_state(_AMR_HOLD, 2 * _S)
    assert s.binders["left_gripper"].held.outcome is BindOutcome.PROVISIONAL  # type: ignore[union-attr]
    assert s.attachments() == [held]
    dropped, _ = s.leave_station()  # `_on_station_perception(False)`
    assert oid not in dropped and s.held_attachments() == [held]


def test_amr_clear_carton_binds_its_measured_box_under_on() -> None:
    """The same grasp with the carton's last accepted fit clear of the robot (the earlier AMR
    `tracking` run: ``outcome=tracked object_id=trk:3``) binds the track itself under `on`: the
    record the kernel sees is the carton's measured box, no pinch stand-in."""
    s, oid, binding = _amr_tote_grasp(TrackingMode.ON, provisional=False)
    assert binding is not None and binding.outcome is BindOutcome.TRACKED
    [att] = s.attachments()
    assert att.object_id == oid
    assert att.evidence_kind is AttachmentEvidenceKind.TRACKED_OBJECT
    assert {p.geometry_source for p in att.primitives} == {"measured"}
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def _zone(frame: str, name: str = "slot") -> PlaceZone:
    pose = Pose6D(xyz=(0.5, 0.0, 0.4), quat_xyzw=(0, 0, 0, 1), frame_id=frame)
    box = PlaceRegion(frame_id=frame, pose=pose, half_extents=(0.1, 0.1, 0.1))
    return PlaceZone(name=name, region=box)


def test_place_zones_resolve_only_in_the_robots_fixed_frames() -> None:
    """Review of #617: the tracker, not only the CLI, whitelists a zone's frame (a launch arg
    bypasses the CLI): map / odom on any robot, the base frame on a fixed base only."""
    amr = TrackerSession(
        RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml"),
        cell_m=RES,
        grounding_wait_s=1.0,
    )
    arm = TrackerSession(
        RobotDescription.from_yaml("robots/so101_follower/robot.yaml"),
        cell_m=RES,
        grounding_wait_s=1.0,
    )
    d = amr.description
    assert amr.zone_frame_refusal(_zone(d.map_frame)) is None
    assert amr.zone_frame_refusal(_zone(d.odom_frame)) is None
    assert amr.zone_frame_refusal(_zone(d.base_frame)) is not None, "rides the moving base"
    assert amr.zone_frame_refusal(_zone("camera_link")) is not None
    assert arm.zone_frame_refusal(_zone(arm.frame_id)) is None
    amr.zone_transform = lambda frame, stamp_ns: np.eye(4)  # any tf2 at all
    amr.grid_stamp_ns = 1
    amr.place_zones = {z.target_id: z for z in (_zone(d.base_frame), _zone("camera_link", "c"))}
    assert amr._zone_region("zone:slot", amr.frame_id) is None
    assert amr._zone_region("zone:c", amr.frame_id) is None


def test_a_zone_resolve_state_logs_again_after_the_base_leaves_the_station() -> None:
    """Review of #617: ``no_tf`` logged at one station is logged again at the next."""
    from structlog.testing import capture_logs

    s = TrackerSession(
        RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml"),
        cell_m=RES,
        grounding_wait_s=1.0,
    )
    s.zone_transform = lambda frame, stamp_ns: None
    s.place_zones = {"zone:slot": _zone(s.description.map_frame)}
    with capture_logs() as logs:
        for _ in range(2):
            s.grid_stamp_ns = 1
            assert s._zone_region("zone:slot", s.frame_id) is None
            assert s._zone_region("zone:slot", s.frame_id) is None
            s.leave_station()
    assert [e["event"] for e in logs].count("object_tracker.place_zone_no_tf") == 2
