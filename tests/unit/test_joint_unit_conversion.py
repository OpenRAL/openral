"""Joint deg<->rad conversion at the policy boundary is independent of reordering.

openral's ``Action`` / ``JointState`` contract is radians; a degrees-trained
joint-position checkpoint must be converted at the runner's policy boundary.
That conversion MUST run whether or not the checkpoint's joint order matches the
robot's (i.e. whether ``robot_to_policy`` is a permutation or ``None``). A prior
bug nested it inside the reorder branch, so a matching-order checkpoint
(``robot_to_policy is None``) passed degree actions through raw — ~57x too large,
slamming the arm into its limits.

These tests are robot-agnostic: they pin the shared helpers
``_policy_action_to_robot`` / ``_robot_state_to_policy`` / ``_effective_perm``
for any joint-position VLA, not one checkpoint.
"""

from __future__ import annotations

import math

import numpy as np
from openral_rskill_ros.rskill_runner_node import (
    _effective_perm,
    _policy_action_to_robot,
    _robot_state_to_policy,
)


def test_effective_perm_is_identity_when_no_reorder() -> None:
    assert _effective_perm(None, 4) == [0, 1, 2, 3]
    # A wrong-length perm also falls back to identity (defensive).
    assert _effective_perm([0, 1], 4) == [0, 1, 2, 3]
    # A valid perm is passed through.
    assert _effective_perm([2, 0, 1], 3) == [2, 0, 1]


def test_action_converts_deg_to_rad_without_reorder() -> None:
    """perm=None + degrees: every non-gripper channel is deg->rad (the bug)."""
    deg = np.array([90.0, -45.0, 30.0], dtype=np.float32)
    out = _policy_action_to_robot(
        deg, robot_to_policy=None, joint_units_are_degrees=True, policy_is_gripper=[]
    )
    np.testing.assert_allclose(out, np.radians(deg), rtol=1e-5)
    assert float(np.abs(out).max()) < math.pi  # radians-scale, not the raw ~90


def test_state_converts_rad_to_deg_without_reorder() -> None:
    rad = np.array([1.5708, -0.7854, 0.5236], dtype=np.float32)
    out = _robot_state_to_policy(
        rad, robot_to_policy=None, joint_units_are_degrees=True, policy_is_gripper=[]
    )
    np.testing.assert_allclose(out, np.degrees(rad), rtol=1e-4)


def test_radians_checkpoint_passes_through() -> None:
    """joint_units_are_degrees=False: no conversion, any robot."""
    vec = np.array([0.1, -0.4, 1.2], dtype=np.float32)
    out = _policy_action_to_robot(
        vec, robot_to_policy=None, joint_units_are_degrees=False, policy_is_gripper=[]
    )
    np.testing.assert_allclose(out, vec, rtol=1e-6)


def test_gripper_channel_not_converted() -> None:
    """A flagged gripper channel is a motor range, not an angle — left as-is."""
    vec = np.array([45.0, 60.0], dtype=np.float32)
    out = _policy_action_to_robot(
        vec, robot_to_policy=None, joint_units_are_degrees=True, policy_is_gripper=[False, True]
    )
    assert abs(float(out[0]) - math.radians(45.0)) < 1e-5
    assert float(out[1]) == 60.0


def test_reorder_and_convert_together() -> None:
    """With a real permutation, the helper both reorders AND converts."""
    deg = np.array([10.0, 20.0, 30.0], dtype=np.float32)  # policy order
    # robot_to_policy[i] = policy index feeding robot slot i
    out = _policy_action_to_robot(
        deg, robot_to_policy=[2, 0, 1], joint_units_are_degrees=True, policy_is_gripper=[]
    )
    expected = np.radians(np.array([30.0, 10.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(out, expected, rtol=1e-5)
