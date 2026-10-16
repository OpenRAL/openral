"""Release of bound tracked objects and their return to tracking (issue #418, WP C2).

Real manifests, poses by the manifest FK, a protocol-conforming in-memory track store.
"""

from __future__ import annotations

import numpy as np
from openral_core import (
    AttachmentEvidenceKind,
    FitKind,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    TrackedObject,
    TrackStatus,
)
from openral_world_state.object_tracker.binding import BindOutcome
from openral_world_state.object_tracker.pinch import HandModel
from openral_world_state.object_tracker.release import (
    JawReleaseMonitor,
    fall_column,
    reassociate_released,
    release_track,
)

from tests.unit._object_tracker_fixtures import CELL, DictStore, arm_at_hold, track


def test_jaw_open_releases_on_a_t9_like_trace() -> None:
    """DROID t9: no DETACH ever came; the jaw opened to 0.988 and stayed there."""
    hand = HandModel(
        RobotDescription.from_yaml("robots/droid_franka_robotiq/robot.yaml"), "finger_joint"
    )
    m = JawReleaseMonitor.for_hand(hand, hold=0.45, release_dwell_s=0.2)
    assert m.release_gap == 0.075  # half the manifest's stall_gap
    hz, fired = 15.0, []
    jaw = [0.45 + 0.01 * np.sin(i) for i in range(15)]  # 1 s held, encoder noise
    jaw += [0.62, 0.45, 0.45]  # one-sample slip spike: shorter than the dwell
    jaw += list(np.linspace(0.45, 0.988, 6)) + [0.988] * 15  # opens and stays open
    for i, v in enumerate(jaw):
        if m.observe(float(v), int(i / hz * 1e9)):
            fired.append(i)
    first_open = next(i for i, v in enumerate(jaw) if i > 17 and v - 0.45 > 0.075)
    assert fired and fired[0] == first_open + 3  # 3 samples at 15 Hz = 0.2 s dwell
    assert all(i >= first_open for i in fired)


def test_g20_drop_reassociates_in_the_fall_column_with_a_new_epoch() -> None:
    """SO-101 g20: the orange drops mid-air; it lands 4 cm aside and is the same object."""
    binder, q, centre = arm_at_hold("so101_follower")
    store = DictStore([track("trk:3", binder.frame_id, centre, (0.024, 0.024, 0.024))])
    assert binder.on_attach(store, q, stamp_ns=10).outcome is BindOutcome.TRACKED
    pose = binder.object_pose(q)
    table = pose.xyz[2] - 0.12  # released 12 cm above the table
    released = release_track(
        store, "trk:3", object_pose=pose, support_z=table, cell_m=CELL, stamp_ns=20
    )
    binder.on_release()
    assert released is not None and released.status is TrackStatus.RELEASED
    assert released.bound_hand is None and released.pose_in_link is None
    np.testing.assert_allclose(released.box.pose.xyz, pose.xyz, atol=1e-9)
    column = fall_column(released, CELL)
    assert column.half_extents[0] > released.box.half_extents[0] + 0.09  # gate grew with height

    def landed(dx: float) -> PlaceRegion:
        return PlaceRegion(
            frame_id=binder.frame_id,
            pose=Pose6D(
                xyz=(pose.xyz[0] + dx, pose.xyz[1], table + 0.025),
                quat_xyzw=(0, 0, 0, 1),
                frame_id=binder.frame_id,
            ),
            half_extents=(0.026, 0.026, 0.025),
        )

    def reassociate(fit: PlaceRegion) -> TrackedObject | None:
        return reassociate_released(
            store,
            fit,
            fit_kind=FitKind.SILHOUETTE_CONE,
            support_z=table,
            support_ref="map",
            visible_cells=6,
            occupied_cells=5,
            cell_m=CELL,
            stamp_ns=30,
        )

    assert reassociate(landed(0.40)) is None  # outside every fall column: a new object
    back = reassociate(landed(0.04))
    assert back is not None and back.object_id == "trk:3"
    assert back.status is TrackStatus.CONFIRMED and back.epoch == 1 and back.identity_inferred
    assert back.support_z == table and back.floor_z == table + CELL
    assert store.get("trk:3") == back


def test_regrasp_of_a_released_track_binds_it_recentred_with_a_new_epoch() -> None:
    """g20 timing: the policy re-grasps the dropped orange before it is re-grounded."""
    binder, q, centre = arm_at_hold("so101_follower")
    store = DictStore([track("trk:3", binder.frame_id, centre, (0.024, 0.024, 0.024))])
    binder.on_attach(store, q, stamp_ns=10)
    pose = binder.object_pose(q)
    lifted = Pose6D(
        xyz=(pose.xyz[0], pose.xyz[1], pose.xyz[2] + 0.05),
        quat_xyzw=pose.quat_xyzw,
        frame_id=pose.frame_id,
    )
    release_track(
        store, "trk:3", object_pose=lifted, support_z=pose.xyz[2] - 0.2, cell_m=CELL, stamp_ns=20
    )
    binder.on_release()
    # The hand comes back to where the orange was (inside its fall column).
    b = binder.on_attach(store, q, stamp_ns=30)
    assert b.outcome is BindOutcome.TRACKED and b.candidates == ("trk:3",)
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.TRACKED_OBJECT
    assert b.attachment.epoch == 1  # the kernel retired (trk:3, 0) at the release
    pinch = binder.hand.pinch_in_link(q["gripper"])
    np.testing.assert_allclose(b.attachment.pose_in_link.xyz, pinch.centre, atol=1e-9)
    assert len(b.attachment.primitives) == 1  # its known shape, re-centred; no stale pose
    bound = store.get("trk:3")
    assert bound is not None and bound.status is TrackStatus.BOUND and bound.epoch == 1
