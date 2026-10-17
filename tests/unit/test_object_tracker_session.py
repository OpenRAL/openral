"""``TrackerSession``: goals, grounding wait, binding, release and ambiguous pruning (#418 D1).

Real manifests (``robots/droid_franka_robotiq``, ``robots/so101_follower``), the DROID
``exterior_depth`` rig ray-cast from the committed scene, the real tracker core and store.
"""

from __future__ import annotations

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
    Pose6D,
    RobotDescription,
    TrackStatus,
)
from openral_world_state.object_tracker import CameraFrame, TrackerParams
from openral_world_state.object_tracker.binding import BindOutcome
from openral_world_state.object_tracker.session import GraspEventKind, TrackerSession

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
    robot: str = "droid_franka_robotiq",
) -> GraspDeclaration:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    return GraspDeclaration(
        target_id=label,
        contact_links=d.grasp_contact_links(),
        timeout_s=60.0,
        stamp_ns=stamp_ns,
        active=active,
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


def test_grounding_incomplete_is_logged_once_after_the_wait(cap: Any) -> None:
    s, wall = _session("droid_franka_robotiq", wait_s=1.0), 0.0
    s.on_grasp_declaration(_decl(1))
    for n in range(6):
        wall = _frame(s, [], n * _S // 2, wall + 1)
    [event] = cap.named("object_tracker.grounding_incomplete")
    assert event["grounding_required"] is False


def _mint(s: TrackerSession, xyz: NDArray[np.float64], oid_label: str = "orange") -> str:
    frame = s.frame_id
    half = (0.024, 0.024, 0.024)
    pose = Pose6D(xyz=tuple(float(v) for v in xyz), quat_xyzw=(0, 0, 0, 1), frame_id=frame)
    t = s.tracker.store.mint(
        label=oid_label,
        role=ObjectRole.TARGET,
        declaration_key="orange@1",
        box=PlaceRegion(frame_id=frame, pose=pose, half_extents=half),
        tight=(),
        fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=float(xyz[2]) - 0.025,
        support_ref="map",
        floor_z=float(xyz[2]) - 0.024,
        visible_cells=6,
        occupied_cells=5,
        last_fit_stamp_ns=1,
        last_confirmed_stamp_ns=1,
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


def test_a_released_binding_leaves_with_the_reconfirmation() -> None:
    s = _session("so101_follower")
    _, oid, t0 = _released(s)
    track = s.tracker.store.get(oid)
    assert track is not None and track.status is TrackStatus.RELEASED
    s.tracker.store.replace(
        track.model_copy(update={"status": TrackStatus.CONFIRMED, "epoch": track.epoch + 1})
    )
    s.on_joint_state({}, t0 + 1)  # the very next update takes (id, 0) off the wire
    assert s.attachments() == []


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


def test_bind_relays_the_workspace_box_while_the_map_confirms_it() -> None:
    """With the HAL producer stood down (bind/on) the tracker confirms the declaration's
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


# ── The tracker's own grasp event (bind/on, sim only; #418 2026-10-17) ────────────────────────


def _derived_session(robot: str, derive: bool = True) -> TrackerSession:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    return TrackerSession(
        d, cell_m=RES, grounding_wait_s=3.0, params=TrackerParams(), require_hands=True,
        derive_attach=derive,
    )  # fmt: skip


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
    s = _derived_session("so101_follower", derive=False)  # shadow, and every real HAL
    _mint(s, centre)
    _hold_for(s, joint, q, jaw, command=0.10)
    assert s.binders[joint].held is None and not s.attachments()
