"""Grasp binding and in-hand growth of tracked objects (issue #418, WP C2).

Real manifests (``robots/so101_follower``, ``robots/droid_franka_robotiq``), poses by the
manifest FK, a protocol-conforming in-memory track store (a real component, not a mock).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachmentEvidenceKind,
    ObjectRole,
    RobotDescription,
    TrackStatus,
)
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.kinematics import (
    Box,
    bounding_half_extents,
    box_gap_lower_bound_m,
    link_transform,
    transform_from_xyz_rpy,
)
from openral_core.perception_geometry import VoxelLattice
from openral_world_state.object_tracker.binding import (
    MAX_PRIMITIVES,
    Binding,
    BindOutcome,
    GraspBinder,
    _box,
)
from openral_world_state.object_tracker.pinch import HandModel

from tests.unit._object_tracker_fixtures import (
    CELL,
    ROBOTS,
    DictStore,
    arm_at_hold,
    track,
)


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_one_candidate_binds_the_tracked_id(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    c_axis = binder.hand.t_frame_from_attach(frame, q)[:3, :3] @ binder.hand.axes[:, 0]
    held = track("trk:1", frame, centre + 0.008 * c_axis, (0.024, 0.024, 0.024))  # off-axis 8 mm
    far = track(
        "trk:2",
        frame,
        centre + np.array([0.3, 0.3, 0.0]),
        (0.024, 0.024, 0.024),
        role=ObjectRole.DESTINATION,
    )
    store = DictStore([held, far])
    b = binder.on_attach(store, q, stamp_ns=10)
    a = b.attachment
    assert b.outcome is BindOutcome.TRACKED and b.candidates == ("trk:1",)
    assert a.object_id == "trk:1" and a.evidence_kind is AttachmentEvidenceKind.TRACKED_OBJECT
    assert (a.epoch, a.geometry_revision) == (0, 0)
    assert a.attach_link == binder.hand.attach_link
    assert tuple(a.touch_links) == binder.hand.description.grasp_contact_links([ROBOTS[robot][0]])
    # pose_in_link = T_link^-1 . T_obj: composing back gives the fitted box pose.
    t_fa = binder.hand.t_frame_from_attach(frame, q)
    t_fo = t_fa @ homogeneous_from_quat_xyz(a.pose_in_link.xyz, a.pose_in_link.quat_xyzw)
    np.testing.assert_allclose(t_fo[:3, 3], held.box.pose.xyz, atol=1e-9)
    # The fitted shape plus the same shape slid onto the pinch axis.
    assert len(a.primitives) == 2
    slid = t_fo @ homogeneous_from_quat_xyz(
        *(a.primitives[1].pose_in_object.xyz, a.primitives[1].pose_in_object.quat_xyzw)
    )
    assert abs(float((slid[:3, 3] - centre) @ c_axis)) < 1e-9
    bound = store.get("trk:1")
    assert bound is not None and bound.status is TrackStatus.BOUND
    assert bound.bound_hand == ROBOTS[robot][0] and bound.pose_in_link == a.pose_in_link
    assert store.get("trk:2") == far


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_fresh_map_confirmation_does_not_make_an_old_fit_bindable(robot: str, cap: Any) -> None:
    binder, q, centre = arm_at_hold(robot)
    stale = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={"last_fit_stamp_ns": 1, "last_confirmed_stamp_ns": 10_000_000_000}
    )
    store = DictStore([stale])

    result = binder.on_attach(store, q, stamp_ns=10_000_000_000)

    assert result.outcome is BindOutcome.UNTRACKED
    assert result.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    assert result.reground
    assert store.get("trk:1") == stale
    [event] = cap.named("object_tracker.stale_attach_candidate")
    assert event["candidates"][0]["reason"] == "stale_fit"


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_map_agreement_keeps_an_occluded_tracks_fit_bindable(robot: str) -> None:
    """Under the hand the camera cannot refit; the map's agreement keeps the box current.

    Restock 2026-10-31 (arms-out start): the clear target box was map-confirmed 0.17 s
    before the grasp but 4 s past its last fit, and bound the pinch fallback.
    """
    binder, q, centre = arm_at_hold(robot)
    hidden = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={
            "status": TrackStatus.OCCLUDED,
            "role": ObjectRole.TARGET,
            "last_fit_stamp_ns": 1,
            "last_confirmed_stamp_ns": 10_000_000_000,
        }
    )
    result = binder.on_attach(DictStore([hidden]), q, stamp_ns=10_500_000_000)
    assert result.outcome is BindOutcome.TRACKED


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_an_occluded_neighbour_is_not_kept_bindable_by_the_map(robot: str) -> None:
    """2026-10-31 restock seed 20: an occluded context carton kept fresh this way met the
    pinch beside the target and made the grasp AMBIGUOUS."""
    binder, q, centre = arm_at_hold(robot)
    hidden = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={
            "status": TrackStatus.OCCLUDED,
            "role": ObjectRole.CONTEXT,
            "last_fit_stamp_ns": 1,
            "last_confirmed_stamp_ns": 10_000_000_000,
        }
    )
    result = binder.on_attach(DictStore([hidden]), q, stamp_ns=10_500_000_000)
    assert result.outcome is BindOutcome.UNTRACKED


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_an_occluded_track_the_map_has_not_confirmed_is_still_stale(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    hidden = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={"status": TrackStatus.OCCLUDED, "last_fit_stamp_ns": 1,
                "last_confirmed_stamp_ns": 1}
    )  # fmt: skip
    result = binder.on_attach(DictStore([hidden]), q, stamp_ns=10_000_000_000)
    assert result.outcome is BindOutcome.UNTRACKED


def test_a_future_fit_is_not_eligible_for_normal_tracked_binding() -> None:
    binder, q, centre = arm_at_hold("so101_follower")
    future = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={"last_fit_stamp_ns": 11, "last_confirmed_stamp_ns": 10}
    )

    result = binder.on_attach(DictStore([future]), q, stamp_ns=10)

    assert result.outcome is BindOutcome.UNTRACKED


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_two_candidates_bind_the_union_without_identity(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    t_fa = binder.hand.t_frame_from_attach(frame, q)
    b_axis = t_fa[:3, :3] @ binder.hand.axes[:, 2]
    pair = [
        track(f"trk:{i}", frame, centre + s * 0.012 * b_axis, (0.012, 0.012, 0.012))
        for i, s in ((1, -1.0), (2, 1.0))
    ]
    store = DictStore(pair)
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.AMBIGUOUS and set(b.candidates) == {"trk:1", "trk:2"}
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    assert not b.attachment.object_id.startswith("trk:")  # nothing is cleared by identity
    assert len(b.attachment.primitives) == 3  # pinch box + both candidates
    assert all(t.status is TrackStatus.CONFIRMED for t in store.all())


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_no_candidate_is_an_untracked_pinch_fallback(robot: str) -> None:
    binder, q, centre = arm_at_hold(robot)
    store = DictStore([track("trk:1", binder.frame_id, centre + 0.25, (0.02, 0.02, 0.02))])
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED and b.reground and b.candidates == ()
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    assert b.attachment.evidence_ref == "untracked_grasp"
    kept = store.get("trk:1")
    assert kept is not None and kept.status is TrackStatus.CONFIRMED


def _corners(box: Box) -> NDArray[np.float64]:
    c, r, h = box
    unit = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return np.asarray((unit * h) @ r.T + c, dtype=np.float64)


def test_pinch_fallback_never_past_the_palm_on_the_so101() -> None:
    """Over the stroke and every wrist roll: in front of the palm, clear of the wrist (g20)."""
    d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hand = HandModel(d, "gripper")
    a = hand.axes[:, 1]
    palm_plane = float(np.asarray(hand.joint.origin_xyz) @ a)  # the jaw joint's origin
    wrist = [g for g in d.collision_geometry if g.link_name == "wrist"]
    for jaw in np.linspace(0.0, 1.0, 21):
        fb = hand.fallback_in_link(float(jaw))
        assert float((_corners(fb.box()) @ a).min()) >= palm_plane - 1e-12
        for roll in np.linspace(-2.79, 2.79, 15):
            t_aw = np.linalg.inv(link_transform(d, "wrist", "gripper", {"wrist_roll": roll}))
            for g in wrist:
                t = t_aw @ transform_from_xyz_rpy(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:])
                box = (t[:3, 3], t[:3, :3], bounding_half_extents(g.shape))
                assert box_gap_lower_bound_m(fb.box(), box) > 0.0, (jaw, roll)
        # ...and it covers the pads' gap and reaches past the fingertips.
        pinch = hand.pinch_in_link(float(jaw))
        assert float((_corners(fb.box()) @ a).max()) > float((_corners(pinch.box()) @ a).max())


def _grid(
    frame: str, points: NDArray[np.float64], lo: NDArray[np.float64], size: int = 40
) -> VoxelLattice:
    occ = np.zeros(size**3, dtype=np.uint8)
    ijk = np.floor((points - lo) / CELL).astype(int)
    ok = np.all((ijk >= 0) & (ijk < size), axis=1)
    i, j, k = ijk[ok].T
    occ[i + size * (j + size * k)] = 1
    return VoxelLattice(frame, tuple(lo.tolist()), (0.0, 0.0, 0.0, 1.0), CELL, (size,) * 3, occ)


def test_regrasp_unions_until_the_cap_then_refuses() -> None:
    """A re-stall re-centres the shape on the new pinch axis; the union only grows."""
    binder, q, centre = arm_at_hold("so101_follower")
    store = DictStore([track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024))])
    before = binder.on_attach(store, q, stamp_ns=10).attachment
    refused = False
    for i, jaw in enumerate(np.linspace(0.05, 0.6, 24)):
        nxt = binder.on_regrasp({**q, "gripper": float(jaw)}, stamp_ns=20 + i)
        if nxt is None:
            refused = True
            assert binder.held is not None and binder.held.attachment == before
            continue
        assert nxt.attachment.primitives[: len(before.primitives)] == before.primitives
        assert nxt.attachment.geometry_revision == before.geometry_revision + 1
        before = nxt.attachment
    assert refused and len(before.primitives) == MAX_PRIMITIVES


def test_a_neighbour_whose_geometry_misses_the_pinch_centre_is_not_bound() -> None:
    """p2po_on2 (Spark, on-mode): the SO-101 closed on the ungrounded orange_3 while the
    tracked orange_1's padded box still met the swept pinch volume, and the tracker bound
    orange_1. A track binds only if its measured geometry holds the pinch centre within one
    cell: here its k-DOP misses it by 4 cm, so the grasp is an untracked PINCH_FALLBACK."""
    import json
    from pathlib import Path

    from openral_core import TrackedObject

    rec = json.loads((Path(__file__).parent / "_p2po_on2_attach.json").read_text())
    d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    binder = GraspBinder(
        HandModel(d, "gripper"),
        frame_id=d.base_frame,
        cell_m=0.015,
        max_track_fit_age_s=2.0,
    )
    orange_1 = TrackedObject.model_validate(rec["track"])
    store = DictStore([orange_1])
    b = binder.on_attach(store, rec["positions"], stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED and b.candidates == ()
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    kept = store.get(orange_1.object_id)
    assert kept is not None and kept.status is TrackStatus.CONFIRMED


def _payload_boxes(binder: GraspBinder, q: dict[str, float], b: Binding) -> list[Box]:
    """The attachment's primitives as boxes in the grid frame at ``q``."""
    att = b.attachment
    t_fa = binder.hand.t_frame_from_attach(binder.frame_id, q)
    t_fo = t_fa @ homogeneous_from_quat_xyz(att.pose_in_link.xyz, att.pose_in_link.quat_xyzw)
    out: list[Box] = []
    for p in att.primitives:
        t = t_fo @ homogeneous_from_quat_xyz(p.pose_in_object.xyz, p.pose_in_object.quat_xyzw)
        out.append((t[:3, 3], t[:3, :3], bounding_half_extents(p.shape)))
    return out


def _outside(points: NDArray[np.float64], boxes: list[Box]) -> float:
    """Worst distance of ``points`` outside the union of ``boxes`` (m)."""
    best = np.full(len(points), np.inf)
    for c, r, h in boxes:
        local = np.abs((points - c) @ r)
        best = np.minimum(best, np.linalg.norm(np.maximum(local - h, 0.0), axis=1))
    return float(best.max())


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_provisional_track_is_never_the_payload_alone(robot: str) -> None:
    """#418 r4: a grasp whose only candidate is provisional binds no identity, but its payload
    is the union of the provisional box and the pinch fallback, cut at the palm and at the
    candidate's support: never smaller than either in front of the palm and above the support.
    (The 2026-10-17 rule dropped the box and left a pad-width fallback that
    under-covered every restock carton by 4-16 cm.)"""
    binder, q, centre = arm_at_hold(robot)
    held = track("trk:1", binder.frame_id, centre, (0.05, 0.05, 0.05))
    store = DictStore([held.model_copy(update={"provisional": True})])
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.PROVISIONAL and b.reground and b.candidates == ("trk:1",)
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    assert not b.attachment.object_id.startswith("trk:")  # nothing is cleared by identity
    kept = store.get("trk:1")
    assert kept is not None and kept.status is TrackStatus.CONFIRMED and kept.provisional
    boxes = _payload_boxes(binder, q, b)
    a = binder.hand.t_frame_from_attach(binder.frame_id, q)[:3, :3] @ binder.hand.axes[:, 1]
    palm = binder.hand.palm_m + float(
        binder.hand.t_frame_from_attach(binder.frame_id, q)[:3, 3] @ a
    )
    corners = _corners(_box(held.box))
    in_front = corners[corners @ a >= palm + 1e-6]
    assert _outside(in_front, boxes) < 1e-6  # the provisional box, in front of the palm
    fb = DictStore([])
    untracked = _payload_boxes(binder, q, binder.on_attach(fb, q, stamp_ns=11))
    # ...and the untracked fallback above the candidate's support (nothing held lies under
    # the surface it rested on: `_support_clipped`, Spark restock R284/R296).
    fb_corners = _corners(untracked[0])
    above = fb_corners[fb_corners[:, 2] >= held.support_z + 1e-6]
    assert _outside(above, boxes) < 1e-6
    clear = DictStore([held])
    assert binder.on_attach(clear, q, stamp_ns=12).outcome is BindOutcome.TRACKED


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_the_untracked_fallback_bounds_the_opening_and_the_stroke(robot: str) -> None:
    """OT-10 bounded: along the closing axis the jaw opening seen before closure (an upper
    bound on the held width there), along the other two the gripper's full stroke from the
    manifest; never behind the palm."""
    binder, q, centre = arm_at_hold(robot)
    hand = binder.hand
    joint = hand.joint.name
    opening = q[joint] + 0.6 * (hand.open_end - q[joint])
    for jaw in (q[joint], opening, q[joint]):  # opened, then closed onto the object
        binder.observe_jaw(jaw)
    b = binder.on_attach(DictStore([]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED
    (box,) = _payload_boxes(binder, q, b)
    t = hand.t_frame_from_attach(binder.frame_id, q)
    axes = t[:3, :3] @ hand.axes
    local = (_corners(box) - box[0]) @ axes
    stroke_half = float(hand.pinch_in_link(hand.open_end).half[0])
    gap = hand.pinch_in_link(opening)
    gap_world = (t[:3, :3] @ gap.centre + t[:3, 3], axes, gap.half)
    assert _outside(_corners(gap_world), [box]) < 1e-6  # the pre-closure gap, whole
    assert local[:, 0].max() < gap.half[0] + 1e-6  # ...and no wider along the closing axis
    assert local[:, 2].max() >= stroke_half - 1e-6  # the stroke across the pads
    jh_centre = (centre - box[0]) @ axes
    assert local[:, 1].max() - jh_centre[1] >= stroke_half - 1e-6  # ...and past the pinch
    palm = hand.palm_m + float(t[:3, 3] @ axes[:, 1])
    assert float((_corners(box) @ axes[:, 1]).min()) >= palm - 1e-6
    # Without a pre-closure sample the closing axis takes the whole stroke.
    b2 = binder.on_attach(DictStore([]), q, stamp_ns=11)
    (box2,) = _payload_boxes(binder, q, b2)
    assert ((_corners(box2) - box2[0]) @ axes)[:, 0].max() >= stroke_half - 1e-6


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_the_fallback_width_is_the_closing_stroke_that_grasped(robot: str) -> None:
    """Spark bkA_on_rs_10 (2026-11-02): a regrasp closed from 0.47 onto the carton, but the
    jaw had stood fully open earlier in the hold, so the fallback took the whole stroke
    (21 cm across a 5 cm carton) and met the torso. The width is the opening the grasping
    closure started from; jitter under the jaw's release gap never starts a new stroke."""
    binder, q, _ = arm_at_hold(robot)
    hand = binder.hand
    joint = hand.joint.name
    cal = hand.joint.closure_calibration
    assert cal is not None
    tol = cal.stall_gap / 2.0
    sign = 1.0 if hand.open_end > hand.closed else -1.0

    def at(frac: float) -> float:
        return q[joint] + frac * (hand.open_end - q[joint])

    # Wide open, closed onto nothing, reopened only part way, then closed onto the object
    # with a reading that ticks back open by less than the release gap on the way.
    for jaw in (at(1.0), at(0.2), at(0.5), at(0.3), at(0.3) + sign * 0.4 * tol, q[joint]):
        binder.observe_jaw(jaw)
    b = binder.on_attach(DictStore([]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED
    (box,) = _payload_boxes(binder, q, b)
    t = hand.t_frame_from_attach(binder.frame_id, q)
    axes = t[:3, :3] @ hand.axes
    width = float(((_corners(box) - box[0]) @ axes)[:, 0].max())
    assert width == pytest.approx(float(hand.pinch_in_link(at(0.5)).half[0]), abs=1e-6)


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_bound_payload_never_reaches_behind_the_palm(robot: str) -> None:
    """DROID r2dr_on1: a 17 cm banana's containing box, bound at the pinch volume, reached the
    Franka's wrist (payload x panda_link7). Nothing held lies behind the palm, so the bound
    primitives are cut at the palm plane; the part in front of it is still held."""
    binder, q, centre = arm_at_hold(robot)
    big = track("trk:1", binder.frame_id, centre, (0.12, 0.12, 0.12))
    b = binder.on_attach(DictStore([big]), q, stamp_ns=10)
    assert b.outcome is BindOutcome.TRACKED
    pinch = binder.hand.pinch_in_link(q[ROBOTS[robot][0]])
    a = pinch.axes[:, 1]
    t_ao = homogeneous_from_quat_xyz(
        b.attachment.pose_in_link.xyz, b.attachment.pose_in_link.quat_xyzw
    )
    reach = []
    for p in b.attachment.primitives:
        t = t_ao @ homogeneous_from_quat_xyz(p.pose_in_object.xyz, p.pose_in_object.quat_xyzw)
        corners = _corners((t[:3, 3], t[:3, :3], bounding_half_extents(p.shape)))
        assert (corners @ a).min() >= pinch.palm_m - 1e-6
        reach.append((corners @ a).max())
    assert max(reach) > pinch.palm_m + 0.1  # what lies in front of the palm is still held


def _lift(robot: str, q: dict[str, float]) -> dict[str, float]:
    """The same hold, the arm lifted (the hand rises and moves, the jaw unchanged)."""
    lifted = dict(q)
    if robot == "droid_franka_robotiq":
        lifted["panda_joint2"] -= 0.15
        lifted["panda_joint4"] += 0.10
    else:
        lifted["shoulder_lift"] -= 0.25
    return lifted


# ── In-hand growth of a TRACKED payload (#418 r6po_dbg8, 2026-10-17) ────────────────────
#
# The bound orange_3 was no longer where the payload said: in the gripper frame the orange
# drifted 0.9 cm in 80 ms (inhand.py on the bag), then fell out 1.5 cm at 8.03 s. The payload
# is fixed in the link at bind and only rotation-about-the-pinch-axis growth ran for tracked
# binds; the cells that moved with the hand but stuck out of the payload were never taken.


def _tracked_hold(robot: str) -> tuple[GraspBinder, dict[str, float], NDArray[np.float64]]:
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    r = binder.hand.t_frame_from_attach(frame, q)[:3, :3] @ binder.hand.axes  # c, a, b
    held = track("trk:1", frame, centre, (0.02, 0.02, 0.02))
    store = DictStore([held])
    assert binder.on_attach(store, q, stamp_ns=10).outcome is BindOutcome.TRACKED
    return binder, q, r


def _body(
    centre: NDArray[np.float64], r: NDArray[np.float64], lo: float, hi: float
) -> NDArray[np.float64]:
    """A 3 x 3 cm (closing, width) body long along the pinch approach axis, ``lo..hi`` about
    ``centre``, sampled at 2 mm: the real orange is wider than its fit by a few mm."""
    half = np.array([0.015, 0.015, 0.015])
    axes = (
        np.arange(-half[0], half[0] + 1e-9, 0.002),
        np.arange(lo, hi + 1e-9, 0.002),
        np.arange(-half[2], half[2] + 1e-9, 0.002),
    )
    g = np.stack(np.meshgrid(*axes), -1).reshape(-1, 3)
    return g @ r.T + centre


def _hand_step(
    binder: GraspBinder, robot: str, q: dict[str, float], step_m: float
) -> dict[str, float]:
    """``q`` with the hand lifted by about ``step_m`` (``_lift`` scaled; small-angle linear)."""
    t0 = binder.hand.t_frame_from_attach(binder.frame_id, q)
    full = binder.hand.t_frame_from_attach(binder.frame_id, _lift(robot, q))
    k = step_m / float(np.linalg.norm(full[:3, 3] - t0[:3, 3]))
    lifted = _lift(robot, q)
    return {n: q[n] + k * (lifted[n] - q[n]) for n in q}


_PAN = {"so101_follower": "shoulder_pan", "droid_franka_robotiq": "panda_joint1"}


@pytest.mark.parametrize("robot", list(ROBOTS))
@pytest.mark.parametrize("status", [TrackStatus.TENTATIVE, TrackStatus.SPLIT])
def test_an_unconfirmed_cluster_track_never_makes_a_grounded_object_ambiguous(
    robot: str, status: TrackStatus
) -> None:
    """r6po_dbg9: orange_2 was grasped with trk:2 (OCCLUDED, its own box) and trk:5, a
    never-confirmed fit over orange_1 + orange_2 (refused ``cover_already_tracked`` 44 times,
    then SPLIT) whose 21 x 12 cm box held the pinch centre: AMBIGUOUS, a PINCH_FALLBACK with no
    identity. A measured, grounded object beats an unconfirmed fit of the same cells."""
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    held = track("trk:2", frame, centre, (0.024, 0.024, 0.024), status=TrackStatus.OCCLUDED)
    cluster = track("trk:5", frame, centre + np.array([0.0, 0.0, 0.0]), (0.1, 0.06, 0.04),
                    status=status)  # fmt: skip
    store = DictStore([held, cluster])
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.TRACKED and b.candidates == ("trk:2",)
    assert b.attachment.object_id == "trk:2"
    kept = store.get("trk:5")
    assert kept is not None and kept.status is status  # untouched: not bound, not cleared
    # With no grounded object, the unconfirmed track is still the (only) candidate.
    alone = DictStore([cluster])
    assert binder.on_attach(alone, q, stamp_ns=11).candidates == ("trk:5",)


def test_the_hand_boxes_carry_the_kernels_wrist_links() -> None:
    """Isaac R87: ``panda_link6`` at the bowl rim was no contact to the tracker, so the bowl
    went SPLIT while the kernel took the link as the hand's. The tracker's hand boxes now
    carry the same wrist links as the kernel's (``wrist_links_of``): ``panda_link7`` by
    default (DROID's carrier is welded on, so it was missing too), ``panda_link6`` with 2."""
    d = RobotDescription.from_yaml("robots/droid_franka_robotiq/robot.yaml")
    q = {j.name: 0.0 for j in d.joints}

    def at_link(hand: HandModel, link: str) -> bool:
        t = link_transform(d, d.base_frame, link, q)
        return any(_covers(c, h, t, d, link) for c, _, h in hand.hand_boxes(d.base_frame, q))

    one, two = HandModel(d, "finger_joint"), HandModel(d, "finger_joint", wrist_links=2)
    assert at_link(one, "panda_link7")
    assert not at_link(one, "panda_link6")
    assert at_link(two, "panda_link6")


def _covers(
    c: NDArray[np.float64],
    h: NDArray[np.float64],
    t: NDArray[np.float64],
    d: RobotDescription,
    link: str,
) -> bool:
    """Whether the box at ``c``, half-extents ``h``, is a primitive of ``link`` posed by ``t``."""
    for g in d.collision_geometry:
        if g.link_name != link:
            continue
        p = t @ transform_from_xyz_rpy(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:])
        if np.allclose(p[:3, 3], c, atol=1e-9) and np.allclose(
            bounding_half_extents(g.shape), h, atol=1e-9
        ):
            return True
    return False


def test_a_fallback_payload_never_reaches_below_its_candidates_support() -> None:
    """Spark restock R284/R296 (2026-10-23): the pinch fallback's full stroke, centred on a
    carton in a tote, reached a cell under the tote floor and stopped the lift. Nothing the
    hand holds lies below the surface it rested on at the grasp."""
    from openral_core.kinematics import bounding_half_extents
    from openral_world_state.object_tracker.binding import _t

    binder, q, centre = arm_at_hold("droid_franka_robotiq")
    cand = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02)).model_copy(
        update={"provisional": True, "last_fit_stamp_ns": 10}
    )
    result = binder.on_attach(DictStore([cand]), q, stamp_ns=10)
    assert result.outcome is BindOutcome.PROVISIONAL
    t_fo = binder.hand.t_frame_from_attach(binder.frame_id, q) @ _t(result.attachment.pose_in_link)
    for p in result.attachment.primitives:
        t = t_fo @ _t(p.pose_in_object)
        low = t[2, 3] - np.abs(t[2, :3]) @ np.asarray(bounding_half_extents(p.shape))
        assert low >= cand.support_z - 1e-6, (low, cand.support_z)


def test_an_untracked_payload_never_reaches_below_its_neighbours_support() -> None:
    """Spark restock R305 (2026-10-23): an untracked pinch box reached under the tote while
    lifting. Its tracked neighbours give the support the grasp happened over."""
    from openral_core.kinematics import bounding_half_extents
    from openral_world_state.object_tracker.binding import _t

    binder, q, centre = arm_at_hold("droid_franka_robotiq")
    # A confirmed neighbour 8 cm aside (it does not hold the pinch centre); its support is
    # just under the pinch, so the full-stroke fallback reaches below it.
    beside = track("trk:9", binder.frame_id, centre + np.array([0.08, 0.0, 0.0]), (0.02,) * 3)

    def lowest(store: DictStore) -> float:
        result = binder.on_attach(store, q, stamp_ns=10)
        assert result.outcome is BindOutcome.UNTRACKED
        t_fo = binder.hand.t_frame_from_attach(binder.frame_id, q) @ _t(
            result.attachment.pose_in_link
        )
        lows = []
        for p in result.attachment.primitives:
            t = t_fo @ _t(p.pose_in_object)
            lows.append(t[2, 3] - np.abs(t[2, :3]) @ np.asarray(bounding_half_extents(p.shape)))
        return float(min(lows))

    assert lowest(DictStore([])) < beside.support_z - 0.01, "no neighbour: the full fallback"
    assert lowest(DictStore([beside])) >= beside.support_z - 1e-6


def test_a_support_cut_only_ever_shrinks_the_payload() -> None:
    """Spark restock R315 (2026-10-23): re-boxing the cut fallback in gravity's axes grew it
    sideways into the robot's own forearm. On a tilted hold (SO-101, ~20 deg) and a level
    one (DROID) every cut box lies inside the box it was cut from, and reaches no lower."""
    from openral_core.kinematics import bounding_half_extents
    from openral_world_state.object_tracker import binding as b

    for robot in ("so101_follower", "droid_franka_robotiq"):
        binder, q, centre = arm_at_hold(robot)
        cand = track("trk:1", binder.frame_id, centre, (0.02, 0.02, 0.02)).model_copy(
            update={"provisional": True, "last_fit_stamp_ns": 10}
        )
        fb = binder.hand.fallback_in_link(q[binder.hand.joint.name], None)
        t_ao = b._frame(fb.centre, fb.axes)
        t_fo = binder.hand.t_frame_from_attach(binder.frame_id, q) @ t_ao
        prims = [b._prim(np.eye(4), fb.half, "pinch:x")]
        [cut] = b._support_clipped(prims, t_fo, cand.support_z, "pinch:x")
        t_cut = b._t(cut.pose_in_object)
        corners = (b._UNIT * np.asarray(bounding_half_extents(cut.shape))) @ t_cut[:3, :3].T
        corners += t_cut[:3, 3]
        assert np.all(np.abs(corners) <= np.asarray(fb.half) + 1e-9), robot  # inside

        def low(t: NDArray[np.float64], h: NDArray[np.float64]) -> float:
            return float(t[2, 3] - np.abs(t[2, :3]) @ h)

        assert (
            low(t_fo @ t_cut, np.asarray(bounding_half_extents(cut.shape)))
            >= low(t_fo, np.asarray(fb.half)) - 1e-9
        ), robot


def test_a_thin_object_below_the_pad_centre_still_binds_but_a_neighbour_beside_does_not() -> None:
    """Spark banana R347 / R352 / R355 (2026-10-23): grasped from above, the 3.7 cm banana's
    tight box topped out 1.8-4.6 cm under the Robotiq's pinch centre, so every such grasp bound an
    untracked pinch payload. The pinch centre lowered to the pinch volume's bottom (the
    fingertips' level) meets it; a box beside the hand meets neither point."""
    robot = "droid_franka_robotiq"
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    jh = binder.pinch_in_frame(q)
    reach = float(np.abs(jh[1][2, :]) @ jh[2])
    under = centre - np.array([0.0, 0.0, 0.5 * reach + 0.03])  # its top 3 cm under the centre
    thin = track("trk:1", frame, under, (0.04, 0.04, 0.5 * reach - 0.03 + 0.03))
    beside = track("trk:2", frame, centre + np.array([0.0, 0.2, 0.0]), (0.03, 0.03, 0.03))
    binding = binder.on_attach(DictStore([thin, beside]), q, stamp_ns=10)
    assert binding.outcome is BindOutcome.TRACKED, binding
    assert binding.candidates == ("trk:1",)
