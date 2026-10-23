"""A late ATTACH binds the track at the hand pose of the hold's start (#418, WP C).

The policy lifts within a fraction of a second of the jaw settling; the ATTACH (the HAL's
trigger or the stalled-jaw dwell) lands after the object has begun to move and its track has
been split or lost. Real manifests, the real tracker core, the SO-101 hold configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import TrackStatus
from openral_world_state.object_tracker.binding import BindOutcome, GraspBinder
from openral_world_state.object_tracker.session import TrackerSession

from tests.unit._object_tracker_fixtures import ROBOTS, arm_at_hold
from tests.unit.test_object_tracker_binding import _grid, _hand_step
from tests.unit.test_object_tracker_session import _derived_session, _mint

# The session logs through structlog; ``cap`` (tests/unit/conftest.py) pins the module logger to
# the capture pipeline so a later test that asserts on those logs still sees them.
pytestmark = pytest.mark.usefixtures("cap")

_S = 1_000_000_000
_HZ = 15


def _cube(
    centre: NDArray[np.float64], half: float = 0.024, step: float = 0.005
) -> NDArray[np.float64]:
    g = np.arange(-half, half + 1e-9, step)
    return np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) + centre


def _run(
    robot: str,
    *,
    lift_m: float,
    follows: bool,
    status: TrackStatus = TrackStatus.SPLIT,
    confirmed_ago_s: float = 0.0,
    fit_ago_s: float = 0.0,
    provisional: bool = False,
    max_age_s: float = 2.0,
    lift_at: int = 6,
    box_half: tuple[float, float, float] | None = None,
) -> tuple[TrackerSession, GraspBinder, NDArray[np.float64], dict[str, float]]:
    """Hold, lift by ``lift_m`` from sample ``lift_at``, keep the jaw stalled until the read."""
    binder, q, centre = arm_at_hold(robot)
    joint, _, jaw = ROBOTS[robot]
    s = _derived_session(robot, max_age_s=max_age_s)
    oid = _mint(s, centre)
    t0 = 10 * _S
    track = s.tracker.store.get(oid)
    assert track is not None
    box = track.box.model_copy(update={"half_extents": box_half}) if box_half else track.box
    s.tracker.store.replace(
        track.model_copy(
            update={
                "box": box,
                "status": status,
                "provisional": provisional,
                "last_fit_stamp_ns": t0 - int(fit_ago_s * _S),
                "last_confirmed_stamp_ns": t0 - int(confirmed_ago_s * _S),
            }
        )
    )
    q_up = _hand_step(binder, robot, q, lift_m) if lift_m > 0 else q
    t_hold = binder.hand.t_frame_from_attach(binder.frame_id, q)
    t_now = binder.hand.t_frame_from_attach(binder.frame_id, q_up)
    carried = (t_now @ np.linalg.inv(t_hold))[:3, :3] @ centre + (t_now @ np.linalg.inv(t_hold))[
        :3, 3
    ]
    where = carried if follows else centre
    lo = np.floor((centre - 0.3) / 0.015) * 0.015
    s.on_grid(_grid(binder.frame_id, _cube(where), lo, 60), t0)
    s.on_jaw_command(joint, 0.10, t0)
    closed = s._stall[joint].closed
    opened = jaw + (0.5 if jaw >= closed else -0.5)  # the jaw closes into the hold
    s.on_joint_state({**q, joint: opened}, t0 - int(_S / _HZ))
    for i in range(14):
        wobble = 0.02 * (1 if i % 2 else -1)
        s.on_joint_state(
            {**(q if i < lift_at else q_up), joint: jaw + wobble}, t0 + int(i * _S / _HZ)
        )
    return s, s.binders[joint], carried, q_up


@pytest.mark.parametrize("status", [TrackStatus.SPLIT, TrackStatus.LOST, TrackStatus.OCCLUDED])
def test_a_late_attach_binds_the_track_at_the_hold_pose_and_carries_it(status: TrackStatus) -> None:
    s, binder, carried, q_up = _run("so101_follower", lift_m=0.05, follows=True, status=status)
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED
    pose = binder.object_pose(q_up)  # the payload follows the hand to where the object went
    assert np.allclose(pose.xyz, carried, atol=1e-6), (pose.xyz, carried)
    [oid] = binder.held.candidates
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def test_a_lift_before_the_jaw_settles_binds_from_the_last_pose_in_the_box(cap: Any) -> None:
    """2026-10-22, Spark R117 / R121: the policy lifts while the jaw still settles, so at the
    stall the pinch is above the box; the grasp pose is the newest one before it in the box."""
    s, binder, carried, q_up = _run("so101_follower", lift_m=0.05, follows=True, lift_at=2)
    hold_ns = s._stall[ROBOTS["so101_follower"][0]].hold_since_ns
    assert hold_ns is not None and hold_ns >= 10 * _S + int(2 * _S / _HZ)  # stalled after lift
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED
    assert not cap.named("object_tracker.late_attach_skipped")
    np.testing.assert_allclose(binder.object_pose(q_up).xyz, carried, atol=1e-6)


def test_a_loose_containing_box_binds_on_the_objects_own_cells() -> None:
    """2026-10-22, Spark R129-R146: a banana's 28 x 18 cm containing box needs 113 occupied
    cells under a footprint rule, a carried banana shows ~30; the carried box is checked
    against the cells the track's last fit saw, above its support."""
    _, binder, carried, q_up = _run(
        "so101_follower", lift_m=0.05, follows=True, box_half=(0.12, 0.08, 0.024)
    )
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED
    np.testing.assert_allclose(binder.object_pose(q_up).xyz, carried, atol=1e-6)


def test_a_late_bind_does_not_wait_on_the_map() -> None:
    """2026-10-22, Spark R129-R156: the self-filter hides a held object and the map clears its
    old cells late, so the map is no evidence either way at the bind; the camera's in-hand fits
    release a wrong one (``test_two_in_hand_fits_away_from_the_hand_release_the_payload``)."""
    _, binder, _, _ = _run("so101_follower", lift_m=0.05, follows=False)
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED


def test_a_hand_that_travelled_far_since_the_hold_binds_nothing_late() -> None:
    _, binder, _, _ = _run("so101_follower", lift_m=0.2, follows=True)
    assert (
        binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED
    )  # a fallback bind


def test_an_ordinary_lift_before_a_late_attach_still_binds_late() -> None:
    """Spark R302 (2026-10-23): a 15 cm lift before the ATTACH skipped the late bind at the
    old 12 cm cap and bound the pinch fallback."""
    _, binder, _, _ = _run("so101_follower", lift_m=0.11, follows=True)  # ~0.14 m of pinch
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED


def test_fresh_fit_can_late_bind_even_when_map_confirmation_is_old() -> None:
    s, binder, carried, q_up = _run(
        "so101_follower",
        lift_m=0.05,
        follows=True,
        confirmed_ago_s=5.0,
        fit_ago_s=0.0,
    )
    [track] = s.tracker.store.all()
    assert track.last_confirmed_stamp_ns < 10 * _S
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED
    np.testing.assert_allclose(binder.object_pose(q_up).xyz, carried, atol=1e-6)


def test_fresh_map_confirmation_does_not_make_an_old_fit_eligible_for_late_binding(
    cap: Any,
) -> None:
    _, binder, _, _ = _run(
        "droid_franka_robotiq",
        lift_m=0.05,
        follows=True,
        status=TrackStatus.SPLIT,
        confirmed_ago_s=0.0,
        fit_ago_s=5.0,
    )
    assert binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED
    [event] = cap.named("object_tracker.late_attach_skipped")
    assert event["why"] == "stale_fit"


def test_stale_late_fit_is_excluded_from_normal_attach_fallback(cap: Any) -> None:
    _, binder, _, _ = _run(
        "droid_franka_robotiq",
        lift_m=0.0,
        follows=True,
        status=TrackStatus.SPLIT,
        fit_ago_s=1.6,
        max_age_s=5.0,
    )
    assert binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED
    [event] = cap.named("object_tracker.late_attach_skipped")
    assert event["why"] == "stale_fit"


def test_a_provisional_track_binds_late_as_a_provisional_payload_carried_with_the_hand() -> None:
    """Restock tracks are fitted through the robot (provisional): the late bind keeps that rule."""
    s, binder, carried, q_up = _run("so101_follower", lift_m=0.05, follows=True, provisional=True)
    assert binder.held is not None and binder.held.outcome is BindOutcome.PROVISIONAL
    assert set(binder.held.candidates) <= s._held_ids()  # never aged
    boxes = binder._held_boxes(q_up)
    near = min(np.linalg.norm(b[0] - carried) for b in boxes)
    assert near < 0.05, (
        near
    )  # the measured box rides with the hand: a primitive sits at the carried object


def _fit_at(binder: Any, xyz: NDArray[np.float64]) -> Any:
    from openral_core import PlaceRegion, Pose6D
    from openral_world_state.object_tracker.fit import ContainingFit

    pose = Pose6D(
        xyz=tuple(float(v) for v in xyz), quat_xyzw=(0, 0, 0, 1), frame_id=binder.frame_id
    )
    box = PlaceRegion(frame_id=binder.frame_id, pose=pose, half_extents=(0.02, 0.02, 0.02))
    return ContainingFit(box=box, tight=None, surface=np.zeros((0, 3)), support_z=0.0)


def test_two_in_hand_fits_away_from_the_hand_release_the_payload(cap: Any) -> None:
    """2026-10-22, Spark R150: the banana slipped out with the jaw shut and rode the hand
    into the bowl. Two consecutive in-hand fits meeting no hand box release it."""
    s, binder, carried, _ = _run("so101_follower", lift_m=0.05, follows=True)
    [oid] = binder.held.candidates  # type: ignore[union-attr]
    away = _fit_at(binder, carried + np.array([0.0, 0.0, -0.5]))
    hands = s.hand_spaces()
    s._check_drop(oid, away, hands)
    assert binder.held is not None, "one stray fit never releases"
    s._check_drop(oid, _fit_at(binder, carried), hands)
    s._check_drop(oid, away, hands)
    assert binder.held is not None, "a fit at the hand resets the count"
    s._check_drop(oid, away, hands)
    assert binder.held is None
    assert s.tracker.store.get(oid).status is TrackStatus.RELEASED
    assert len(cap.named("object_tracker.drop_seen")) == 1


def test_a_chattering_command_does_not_rebind_the_held_payload(cap: Any) -> None:
    """2026-10-22, Spark R157: an opening command unlatched the stall detector, the same hold
    read again as a derived ATTACH, and each one released and rebound the payload (~2 s)."""
    s, binder, _, q_up = _run("so101_follower", lift_m=0.05, follows=True)
    joint, _, jaw = ROBOTS["so101_follower"]
    held = binder.held
    assert held is not None and held.outcome is BindOutcome.TRACKED
    t = 12 * _S
    s.on_jaw_command(joint, 1.5, t)  # the policy's chatter: open ...
    s.on_jaw_command(joint, 0.10, t + 1)  # ... and closed again, the jaw never moved
    for i in range(20):
        wobble = 0.02 * (1 if i % 2 else -1)
        s.on_joint_state({**q_up, joint: jaw + wobble}, t + int(i * _S / _HZ))
    assert binder.held is held, "the same hold, not a rebind"
    assert not cap.named("object_tracker.released")


def test_the_place_region_is_the_one_grounded_destination_while_a_payload_is_held() -> None:
    """2026-10-22, Spark R148-R156: the HAL's flat-support producer never armed over a bowl;
    in tracking the tracker publishes the declaration, so its region is the DESTINATION
    track's box, stamped with its last confirmation, failing closed on doubt."""
    from openral_core import (
        FitKind,
        ObjectQuery,
        ObjectRole,
        PlaceDeclaration,
        PlaceRegion,
        Pose6D,
    )

    s, binder, carried, _ = _run("so101_follower", lift_m=0.05, follows=True)
    joint = ROBOTS["so101_follower"][0]

    def bowl(dx: float) -> str:
        frame = s.frame_id
        pose = Pose6D(xyz=(float(carried[0]) + dx, float(carried[1]), 0.05), quat_xyzw=(0, 0, 0, 1),
                      frame_id=frame)  # fmt: skip
        t = s.tracker.store.mint(
            label="bowl", role=ObjectRole.DESTINATION, declaration_key="plate@1",
            box=PlaceRegion(frame_id=frame, pose=pose, half_extents=(0.08, 0.08, 0.04)),
            tight=(), fit_kind=FitKind.SILHOUETTE_CONE, support_z=0.01, support_ref="map",
            floor_z=0.01, visible_cells=20, occupied_cells=18, last_fit_stamp_ns=5,
            last_confirmed_stamp_ns=11 * _S,
        )  # fmt: skip
        s.tracker.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
        return t.object_id

    query = ObjectQuery(label="bowl", count="one", role=ObjectRole.DESTINATION)
    s.on_place_declaration(
        PlaceDeclaration(
            target_id="obj:bowl", timeout_s=60.0, stamp_ns=10 * _S, destination_query=query
        )
    )
    oid = bowl(0.3)
    region = s.place_region()
    assert region is not None and region.stamp_ns == 11 * _S
    assert region.pose == s.tracker.store.get(oid).box.pose  # type: ignore[union-attr]
    assert region.evidence_ref.startswith(f"tracked:{oid}|")
    bowl(0.6)
    assert s.place_region() is None, "two bowls: which one is not the tracker's to guess"
    s.tracker.store.drop(oid)
    assert s.place_region() is not None
    s._release(joint, 12 * _S)
    assert binder.held is None and s.place_region() is None, "no payload, no place"


def test_a_prompt_attach_after_the_lift_binds_from_the_grasp_pose() -> None:
    """2026-10-22, Spark R167: the HAL's ATTACH came 0.067 s after the hold read, the hand
    already lifted; the plain bind found no box at the pinch and bound the pinch fallback."""
    from openral_world_state.object_tracker.session import GraspEventKind

    binder, q, centre = arm_at_hold("so101_follower")
    joint, _, jaw = ROBOTS["so101_follower"]
    s = _derived_session("so101_follower", max_age_s=2.0)
    oid = _mint(s, centre)
    t0 = 10 * _S
    track = s.tracker.store.get(oid)
    assert track is not None
    s.tracker.store.replace(track.model_copy(update={"last_fit_stamp_ns": t0}))
    q_up = _hand_step(binder, "so101_follower", q, 0.05)
    lo = np.floor((centre - 0.3) / 0.015) * 0.015
    s.on_grid(_grid(binder.frame_id, _cube(centre), lo, 60), t0)
    s.on_jaw_command(joint, 0.10, t0)
    detector = s._stall[joint]
    i = 0
    while detector.hold_since_ns is None and i < 14:
        wobble = 0.02 * (1 if i % 2 else -1)
        s.on_joint_state({**(q if i < 2 else q_up), joint: jaw + wobble}, t0 + int(i * _S / _HZ))
        i += 1
    hold_ns = detector.hold_since_ns
    assert hold_ns is not None
    s.on_grasp_event(
        GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=hold_ns + int(_S / _HZ)
    )
    held = s.binders[joint].held
    assert held is not None and held.outcome is BindOutcome.TRACKED


def test_an_attach_with_no_hold_read_still_binds_where_the_pinch_held_the_track() -> None:
    """Spark banana R347 (2026-10-23): the HAL's ATTACH came with no stalled-jaw hold read
    (``no_hold_read``), so the bind read the pinch where the hand already was, 9.6 cm up, above
    the banana's 5 cm box: an untracked pinch payload the kernel then stopped on the banana's
    own cells. With no hold read, the ATTACH's own time is the hold: the lookback finds the
    pose of the last second whose pinch the fresh track holds, under the same travel guard."""
    from openral_world_state.object_tracker.session import GraspEventKind

    robot = "droid_franka_robotiq"
    binder, q, centre = arm_at_hold(robot)
    joint, _, jaw = ROBOTS[robot]
    s = _derived_session(robot, derive=False)  # no stall detector: no hold read
    oid = _mint(s, centre, "banana")
    t0 = 10 * _S
    track = s.tracker.store.get(oid)
    assert track is not None
    s.tracker.store.replace(track.model_copy(update={"last_fit_stamp_ns": t0}))
    q_up = _hand_step(binder, robot, q, 0.08)
    lo = np.floor((centre - 0.3) / 0.015) * 0.015
    s.on_grid(_grid(binder.frame_id, _cube(centre), lo, 60), t0)
    for i in range(12):  # 0.5 s at the hold, then lifted 8 cm
        s.on_joint_state({**(q if i < 8 else q_up), joint: jaw}, t0 + int(i * _S / _HZ))
    s.on_grasp_event(
        GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=t0 + int(11 * _S / _HZ)
    )
    held = s.binders[joint].held
    assert held is not None and held.outcome is BindOutcome.TRACKED, held
    assert held.candidates == (oid,)


def test_an_attach_stamped_just_before_the_hold_read_still_binds_from_the_lookback() -> None:
    """Spark banana R352 (2026-10-23): the stalled-jaw hold was read 67 ms after the HAL's
    ATTACH stamp, so the lag came out negative (``lag_out_of_range``) and the bind read the
    pinch where the hand already was, 12.9 cm up: untracked. A hold read after the ATTACH
    means the ATTACH is not late; its own stamp stands in, and the lookback finds the pose
    whose pinch the fresh track holds."""
    from openral_world_state.object_tracker.session import GraspEventKind

    robot = "droid_franka_robotiq"
    binder, q, centre = arm_at_hold(robot)
    joint, _, jaw = ROBOTS[robot]
    s = _derived_session(robot)
    oid = _mint(s, centre, "banana")
    t0 = 10 * _S
    track = s.tracker.store.get(oid)
    assert track is not None
    s.tracker.store.replace(track.model_copy(update={"last_fit_stamp_ns": t0}))
    q_up = _hand_step(binder, robot, q, 0.08)
    lo = np.floor((centre - 0.3) / 0.015) * 0.015
    s.on_grid(_grid(binder.frame_id, _cube(centre), lo, 60), t0)
    s.on_jaw_command(joint, 0.10, t0)
    closed = s._stall[joint].closed
    s.on_joint_state({**q, joint: jaw + (0.5 if jaw >= closed else -0.5)}, t0 - int(_S / _HZ))
    since = None
    i = 0
    while since is None:
        s.on_joint_state({**q, joint: jaw}, t0 + int(i * _S / _HZ))
        since = s._stall[joint].hold_since_ns
        i += 1
        assert i < 15
    s.on_joint_state({**q_up, joint: jaw}, t0 + int(i * _S / _HZ))  # lifted, before the dwell
    assert s.binders[joint].held is None, "the stall has not fired yet"
    s.on_grasp_event(
        GraspEventKind.ATTACH, joint, jaw=jaw, hold=jaw, stamp_ns=since - int(_S / _HZ)
    )
    held = s.binders[joint].held
    assert held is not None and held.outcome is BindOutcome.TRACKED, held
    assert held.candidates == (oid,)
