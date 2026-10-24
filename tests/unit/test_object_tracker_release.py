"""Release of bound tracked objects (issue #418, WP C2): a released object is not followed.

Real manifests, poses by the manifest FK, a protocol-conforming in-memory track store.
"""

from __future__ import annotations

import numpy as np
from openral_core import AttachmentEvidenceKind, Pose6D, RobotDescription, TrackStatus
from openral_world_state.object_tracker.binding import BindOutcome
from openral_world_state.object_tracker.pinch import HandModel
from openral_world_state.object_tracker.release import JawReleaseMonitor, release_track

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


def test_a_drop_marks_the_track_released_at_the_release_pose() -> None:
    """SO-101 g20: the orange drops mid-air; its track is RELEASED where the hand let go."""
    binder, q, centre = arm_at_hold("so101_follower")
    store = DictStore([track("trk:3", binder.frame_id, centre, (0.024, 0.024, 0.024))])
    assert binder.on_attach(store, q, stamp_ns=10).outcome is BindOutcome.TRACKED
    pose = binder.object_pose(q)
    released = release_track(
        store, "trk:3", object_pose=pose, support_z=pose.xyz[2] - 0.12, cell_m=CELL, stamp_ns=20
    )
    binder.on_release()
    assert released is not None and released.status is TrackStatus.RELEASED
    assert released.bound_hand is None and released.pose_in_link is None
    np.testing.assert_allclose(released.box.pose.xyz, pose.xyz, atol=1e-9)


def test_regrasp_of_a_released_track_never_binds_it() -> None:
    """g20 timing: the policy re-grasps the dropped orange before it is found again.

    A released object is not followed (2026-10-22, Spark R177-R185): its track's box is
    where it was let go, not where it fell, so it is no candidate; the grasp is untracked.
    """
    binder, q, centre = arm_at_hold("so101_follower")
    attach_ns = 10_000_000_000
    released_track = track("trk:3", binder.frame_id, centre, (0.024, 0.024, 0.024)).model_copy(
        update={"last_fit_stamp_ns": attach_ns, "last_confirmed_stamp_ns": attach_ns}
    )
    store = DictStore([released_track])
    binder.on_attach(store, q, stamp_ns=attach_ns)
    pose = binder.object_pose(q)
    lifted = Pose6D(
        xyz=(pose.xyz[0], pose.xyz[1], pose.xyz[2] + 0.05),
        quat_xyzw=pose.quat_xyzw,
        frame_id=pose.frame_id,
    )
    release_track(
        store,
        "trk:3",
        object_pose=lifted,
        support_z=pose.xyz[2] - 0.2,
        cell_m=CELL,
        stamp_ns=attach_ns + 20,
    )
    binder.on_release()
    # The hand comes back to where the orange was.
    b = binder.on_attach(store, q, stamp_ns=attach_ns + 3_000_000_000)
    assert b.outcome is BindOutcome.UNTRACKED and b.candidates == ()
    assert b.attachment.evidence_kind is AttachmentEvidenceKind.PINCH_FALLBACK
    kept = store.get("trk:3")
    assert kept is not None and kept.status is TrackStatus.RELEASED and kept.epoch == 0
