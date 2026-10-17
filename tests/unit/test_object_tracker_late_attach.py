"""A late ATTACH binds the track at the hand pose of the hold's start (#418, WP C).

The policy lifts within a fraction of a second of the jaw settling; the ATTACH (the HAL's
trigger or the stalled-jaw dwell) lands after the object has begun to move and its track has
been split or lost. Real manifests, the real tracker core, the SO-101 hold configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import TrackStatus
from openral_world_state.object_tracker.binding import BindOutcome

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
) -> tuple[object, object, NDArray[np.float64], NDArray[np.float64]]:
    """Hold 0.35 s, lift by ``lift_m``, keep the jaw stalled until the derived read fires."""
    binder, q, centre = arm_at_hold(robot)
    joint, _, jaw = ROBOTS[robot]
    s = _derived_session(robot)
    oid = _mint(s, centre)
    t0 = 10 * _S
    s.tracker.store.replace(
        s.tracker.store.get(oid).model_copy(  # type: ignore[union-attr]
            update={"status": status, "last_confirmed_stamp_ns": t0 - int(confirmed_ago_s * _S)}
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
    for i in range(14):
        wobble = 0.02 * (1 if i % 2 else -1)
        s.on_joint_state({**(q if i < 6 else q_up), joint: jaw + wobble}, t0 + int(i * _S / _HZ))
    return s, s.binders[joint], carried, q_up


@pytest.mark.parametrize("status", [TrackStatus.SPLIT, TrackStatus.LOST, TrackStatus.OCCLUDED])
def test_a_late_attach_binds_the_track_at_the_hold_pose_and_carries_it(status: TrackStatus) -> None:
    s, binder, carried, q_up = _run("so101_follower", lift_m=0.05, follows=True, status=status)
    assert binder.held is not None and binder.held.outcome is BindOutcome.TRACKED
    pose = binder.object_pose(q_up)  # the payload follows the hand to where the object went
    assert np.allclose(pose.xyz, carried, atol=1e-6), (pose.xyz, carried)
    [oid] = binder.held.candidates
    assert s.tracker.store.get(oid).status is TrackStatus.BOUND  # type: ignore[union-attr]


def test_the_old_box_is_not_bound_when_the_hand_lifted_and_the_object_stayed() -> None:
    _, binder, _, _ = _run("so101_follower", lift_m=0.05, follows=False)
    assert binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED


def test_a_hand_that_travelled_far_since_the_hold_binds_nothing_late() -> None:
    _, binder, _, _ = _run("so101_follower", lift_m=0.2, follows=True)
    assert (
        binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED
    )  # a fallback bind


def test_a_track_last_confirmed_long_before_the_hold_is_not_revived() -> None:
    _, binder, _, _ = _run("so101_follower", lift_m=0.05, follows=True, confirmed_ago_s=5.0)
    assert (
        binder.held is not None and binder.held.outcome is not BindOutcome.TRACKED
    )  # a fallback bind
