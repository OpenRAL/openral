"""Grasp binding and in-hand growth of tracked objects (issue #418, WP C2).

Real manifests (``robots/so101_follower``, ``robots/droid_franka_robotiq``), poses by the
manifest FK, a protocol-conforming in-memory track store (a real component, not a mock).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachmentEvidenceKind,
    ObjectRole,
    RobotDescription,
    TrackStatus,
)
from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
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


def test_bound_geometry_only_grows_regrasp_and_protrusion() -> None:
    """DROID t8: the banana rotates in the Robotiq; the payload follows by union only."""
    binder, q, centre = arm_at_hold("droid_franka_robotiq")
    frame = binder.frame_id
    t_fa = binder.hand.t_frame_from_attach(frame, q)
    r_pinch = t_fa[:3, :3] @ binder.hand.axes  # columns c, a, b in the base frame
    # A long body along the pad width (b), its box axes = the pinch axes (x=c, y=a, z=b).
    w, x, y, z = rotation_to_quat_wxyz(r_pinch)
    half = (0.012, 0.012, 0.07)
    banana = track("trk:1", frame, centre, half, quat=(x, y, z, w))
    store = DictStore([banana])
    first = binder.on_attach(store, q, stamp_ns=10).attachment
    assert first.geometry_revision == 0

    # The map: the banana rotated 30 deg about the pinch axis (c), sampled at 2 mm.
    theta = math.radians(30.0)
    c = r_pinch[:, 0]
    k = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    rot = np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)
    g = np.stack(np.meshgrid(*[np.arange(-h, h + 1e-9, 0.002) for h in half]), -1).reshape(-1, 3)
    body = g @ (rot @ r_pinch).T + centre
    lo = np.floor((centre - 0.2) / CELL) * CELL
    grid = _grid(frame, body, lo)
    grown = binder.grow_by_protrusion(store, grid, q, stamp_ns=20)
    assert grown is not None
    att = grown.attachment
    assert (
        att.geometry_revision == 1 and att.primitives[: len(first.primitives)] == first.primitives
    )
    assert len(att.primitives) <= MAX_PRIMITIVES
    assert abs(binder._theta - theta) <= math.radians(2.0) or abs(
        binder._theta + theta
    ) <= math.radians(2.0)
    # Every occupied cell of the banana outside the hand is now within a cell of the payload.
    held = binder._held_boxes(q)
    hand = binder.hand.hand_boxes(frame, q)
    occ = grid.occupied_centers()
    from openral_world_state.object_tracker.binding import _outside_m

    loose = occ[_outside_m(occ, hand) > CELL / 2]
    assert float(_outside_m(loose, held).max()) < CELL
    # Nothing new: no growth.
    assert binder.grow_by_protrusion(store, grid, q, stamp_ns=30) is None


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
    binder = GraspBinder(HandModel(d, "gripper"), frame_id=d.base_frame, cell_m=0.015)
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
    is the union of the provisional box and the pinch fallback, cut at the palm: never smaller
    than either. (The 2026-10-17 rule dropped the box and left a pad-width fallback that
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
    assert _outside(_corners(untracked[0]), boxes) < 1e-6  # ...and the untracked fallback
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


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_an_untracked_payload_grows_by_cells_that_move_with_the_hand(robot: str) -> None:
    """#418 r4: a fallback's revision stayed 0 while 2-10 cm of the held object stuck out
    (C2's in-hand growth ran for TRACKED bindings only). Cells that hold their place in the
    hand across a hand motion, and left their old place in the world, are the payload; the
    table the object stood on is not."""
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    b0 = binder.on_attach(DictStore([]), q, stamp_ns=10)
    assert b0.outcome is BindOutcome.UNTRACKED
    t_a = binder.hand.t_frame_from_attach(frame, q)
    r_pinch = t_a[:3, :3] @ binder.hand.axes
    # A body much longer than any stroke along the pad width, through the pinch centre.
    half = np.array([0.01, 0.01, 0.15])
    g = np.stack(np.meshgrid(*[np.arange(-h, h + 1e-9, 0.004) for h in half]), -1).reshape(-1, 3)
    body_a = g @ r_pinch.T + centre
    floor = float(body_a[:, 2].min()) - CELL
    xy = np.stack(np.meshgrid(*[np.arange(-0.25, 0.25, CELL)] * 2), -1).reshape(-1, 2)
    table = np.column_stack([xy + centre[:2], np.full(len(xy), floor)])
    lo = np.floor((centre - 0.3) / CELL) * CELL
    q_b = _lift(robot, q)
    t_b = binder.hand.t_frame_from_attach(frame, q_b)
    move = t_b @ np.linalg.inv(t_a)
    assert np.linalg.norm(move[:3, 3]) > 2 * CELL  # the hand really moved
    body_b = body_a @ move[:3, :3].T + move[:3, 3]
    store = DictStore([])
    assert (
        binder.grow_untracked(store, _grid(frame, np.vstack([body_a, table]), lo, 60), q, 20)
        is None
    )
    grown = binder.grow_untracked(store, _grid(frame, np.vstack([body_b, table]), lo, 60), q_b, 30)
    assert grown is not None and grown.attachment.geometry_revision == 1
    assert grown.attachment.primitives[0] == b0.attachment.primitives[0]  # a union only
    from openral_world_state.object_tracker.binding import _outside_m

    held = binder._held_boxes(q_b)
    hand = binder.hand.hand_boxes(frame, q_b)
    grid_b = _grid(frame, body_b, lo, 60)
    cells = grid_b.occupied_centers()
    cells = cells[_outside_m(cells, hand) > CELL / 2]
    assert float(_outside_m(cells, held).max()) < CELL  # the whole body is now payload
    table_cells = _grid(frame, table, lo, 60).occupied_centers()
    assert float(_outside_m(table_cells, held).min()) > CELL / 2  # the table is not
    # Seen again where it is: nothing new.
    assert (
        binder.grow_untracked(store, _grid(frame, np.vstack([body_b, table]), lo, 60), q_b, 40)
        is None
    )


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_static_cells_never_grow_an_untracked_payload(robot: str) -> None:
    """The hand moves over a table and a wall with nothing in it: the cells it passes keep
    their place in the world, so none of them is ever the payload."""
    binder, q, centre = arm_at_hold(robot)
    frame = binder.frame_id
    binder.on_attach(DictStore([]), q, stamp_ns=10)
    xy = np.stack(np.meshgrid(*[np.arange(-0.25, 0.25, CELL)] * 2), -1).reshape(-1, 2)
    table = np.column_stack([xy + centre[:2], np.full(len(xy), centre[2] - 0.03)])
    wall = np.column_stack(
        [np.full(len(xy), centre[0] + 0.04), xy[:, 0] + centre[1], xy[:, 1] + centre[2]]
    )
    world = np.vstack([table, wall])
    lo = np.floor((centre - 0.3) / CELL) * CELL
    store = DictStore([])
    for i, qq in enumerate((q, _lift(robot, q), q)):
        assert binder.grow_untracked(store, _grid(frame, world, lo, 60), qq, 20 + i) is None


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


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_tracked_payload_grows_by_cells_that_move_with_the_hand_within_a_cell_of_travel(
    robot: str,
) -> None:
    """The payload (a 4 cm cube) stops 1.5 cm short of the real body along the approach axis;
    the hand lifts one cell (a clear bind needs half a cell; an untracked payload waits for
    two) and the cells that came along are payload."""
    from openral_world_state.object_tracker.binding import _outside_m

    binder, q, r = _tracked_hold(robot)
    frame = binder.frame_id
    t_a = binder.hand.t_frame_from_attach(frame, q)
    pc = t_a[:3, :3] @ binder.hand.pinch_in_link(q[binder.hand.joint.name]).centre + t_a[:3, 3]
    body_a = _body(pc, r, -0.02, 0.035)  # reaches 1.5 cm past the cube along the approach axis
    q_b = _hand_step(binder, robot, q, 1.0 * CELL)  # the lift: one cell
    t_b = binder.hand.t_frame_from_attach(frame, q_b)
    move = t_b @ np.linalg.inv(t_a)
    assert abs(float(np.linalg.norm(move[:3, 3])) - CELL) < 0.2 * CELL
    body_b = body_a @ move[:3, :3].T + move[:3, 3]
    floor = float(body_a[:, 2].min()) - 2 * CELL
    xy = np.stack(np.meshgrid(*[np.arange(-0.25, 0.25, CELL)] * 2), -1).reshape(-1, 2)
    table = np.column_stack([xy + pc[:2], np.full(len(xy), floor)])
    lo = np.floor((pc - 0.3) / CELL) * CELL
    store = DictStore([])
    assert binder.grow_untracked(
        store, _grid(frame, np.vstack([body_a, table]), lo, 60), q, 20) is None  # fmt: skip
    grown = binder.grow_untracked(store, _grid(frame, np.vstack([body_b, table]), lo, 60), q_b, 30)
    assert grown is not None and grown.attachment.geometry_revision == 1
    assert grown.attachment.object_id == "trk:1" and grown.outcome is BindOutcome.TRACKED
    hand = binder.hand.hand_boxes(frame, q_b)
    cells = _grid(frame, body_b, lo, 60).occupied_centers()
    cells = cells[_outside_m(cells, hand) > CELL / 2]
    assert float(_outside_m(cells, binder._held_boxes(q_b)).max()) < CELL  # all payload now
    table_cells = _grid(frame, table, lo, 60).occupied_centers()
    assert float(_outside_m(table_cells, binder._held_boxes(q_b)).min()) > CELL / 2


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_tracked_payload_follows_a_body_that_slips_out_along_the_approach_axis(
    robot: str,
) -> None:
    """r6po_dbg8: the held orange drifted 0.87 cm along the pinch APPROACH axis (and 0.12 cm
    along the closing axis) in 80 ms, towards the fingertips, then fell out (inhand.py,
    slipaxis.py on the bag). Sightings at four hand poses: the body, shifted 2 cm against the travel
    between the second and the third, is taken once it has been seen to travel with the hand."""
    from openral_world_state.object_tracker.binding import _outside_m

    binder, q, r = _tracked_hold(robot)
    frame = binder.frame_id
    t_a = binder.hand.t_frame_from_attach(frame, q)
    pc = t_a[:3, :3] @ binder.hand.pinch_in_link(q[binder.hand.joint.name]).centre + t_a[:3, 3]
    q_b = _hand_step(binder, robot, q, 1.0 * CELL)
    q_c = _hand_step(binder, robot, q_b, 1.0 * CELL)
    q_d = _hand_step(binder, robot, q_c, 1.0 * CELL)
    store = DictStore([])
    lo = np.floor((pc - 0.3) / CELL) * CELL
    # The slab a slip adds is told from the world by the place it left: it shows on the side of
    # the body the hand travels away from (occupancy alone cannot tell a moving interior cell
    # from a static one), so the slip goes against the hand's travel, along the pinch axis most
    # parallel to it (the approach axis in dbg8).
    t_step = binder.hand.t_frame_from_attach(frame, q_b)[:3, 3] - t_a[:3, 3]
    k_axis = int(np.argmax(np.abs(r.T @ t_step)))
    slip = -0.02 * float(np.sign(r[:, k_axis] @ t_step)) * r[:, k_axis]
    poses = []
    for qq, shifted in ((q, False), (q_b, False), (q_c, True), (q_d, True)):
        t = binder.hand.t_frame_from_attach(frame, qq)
        move = t @ np.linalg.inv(t_a)
        body = _body(pc + (slip if shifted else 0.0), r, -0.02, 0.02) @ move[:3, :3].T + move[:3, 3]
        poses.append((qq, body))
    assert binder.held is not None
    before = binder.held.attachment
    stick: list[float] = []
    for n, (qq, body) in enumerate(poses):
        grid = _grid(frame, body, lo, 60)
        grown = binder.grow_untracked(store, grid, qq, 20 + n)
        cells = grid.occupied_centers()
        cells = cells[_outside_m(cells, binder.hand.hand_boxes(frame, qq)) > CELL / 2]
        stick.append(float(_outside_m(cells, binder._held_boxes(qq)).max()))
        if grown is not None:
            assert n >= 2, "nothing sticks out before the slip"
            assert grown.attachment.primitives[: len(before.primitives)] == before.primitives
    # The cells seen to leave their place are payload now; the slab's interior, which
    # occupancy alone cannot tell from a static cell, is not taken. The stick-out shrinks.
    assert binder.held is not None and binder.held.attachment.geometry_revision >= 1
    assert stick[3] < stick[2] - 1e-6, stick


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_static_neighbour_touching_a_tracked_payload_is_never_taken(robot: str) -> None:
    """A body that keeps its place in the world while the hand moves is not held: the 0.5-cell
    threshold must not make the static world payload."""
    binder, q, r = _tracked_hold(robot)
    frame = binder.frame_id
    t_a = binder.hand.t_frame_from_attach(frame, q)
    pc = t_a[:3, :3] @ binder.hand.pinch_in_link(q[binder.hand.joint.name]).centre + t_a[:3, 3]
    neighbour = _body(pc + 0.045 * r[:, 1], r, 0.0, 0.03)  # touches the cube's approach face
    lo = np.floor((pc - 0.3) / CELL) * CELL
    store = DictStore([])
    q_b = _hand_step(binder, robot, q, 1.0 * CELL)
    for n, qq in enumerate((q, q_b, q_b)):
        assert binder.grow_untracked(store, _grid(frame, neighbour, lo, 60), qq, 20 + n) is None
