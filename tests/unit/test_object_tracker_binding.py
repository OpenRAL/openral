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
    BindOutcome,
    GraspBinder,
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


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_provisional_track_never_becomes_the_payload(robot: str) -> None:
    """#418 decision (2026-10-17): a track fitted through the robot may be exempted over,
    but the grasp that meets it falls back to PINCH_FALLBACK and asks for a re-ground (the
    clear refit after lift); the track itself is not bound."""
    binder, q, centre = arm_at_hold(robot)
    held = track("trk:1", binder.frame_id, centre, (0.024, 0.024, 0.024))
    store = DictStore([held.model_copy(update={"provisional": True})])
    b = binder.on_attach(store, q, stamp_ns=10)
    assert b.outcome is BindOutcome.UNTRACKED and b.reground and b.candidates == ()
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    kept = store.get("trk:1")
    assert kept is not None and kept.status is TrackStatus.CONFIRMED and kept.provisional
    clear = DictStore([held])
    assert binder.on_attach(clear, q, stamp_ns=11).outcome is BindOutcome.TRACKED


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
        corners = _corners((t[:3, 3], t[:3, :3], np.asarray(p.shape.half_extents_m)))
        assert (corners @ a).min() >= pinch.palm_m - 1e-6
        reach.append((corners @ a).max())
    assert max(reach) > pinch.palm_m + 0.1  # what lies in front of the palm is still held
