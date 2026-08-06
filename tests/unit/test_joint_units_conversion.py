"""Joint-units conversion at the policy boundary (issue #135).

The runner no longer guesses the checkpoint's joint units — every
joint-position rSkill declares ``action_contract.joint_units`` and the
runner converts deg<->rad only when it is ``degrees``. These tests pin the
two actuation-critical helpers that apply that conversion:

* :func:`_robot_state_to_policy` — robot-order radians state → policy order,
  converting rad->deg when the policy is a degrees checkpoint (so a
  degrees-trained policy is never fed raw radians, ~57x too small → OOD).
* :func:`_policy_action_to_robot` — policy-order action → robot order,
  converting deg->rad so a degrees checkpoint's output reaches the radians
  ``Action`` contract instead of passing through raw (~57x too large → the
  arm slams its limits).

Gripper channels carry a custom motor unit, not an angle. ``gripper_scale``
maps the HAL's normalized [0, 1] surface to a checkpoint's [0, 100] dataset
surface when declared.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from openral_core import RobotDescription
from openral_rskill_ros.rskill_runner_node import (
    _build_joint_permutation,
    _policy_action_to_robot,
    _robot_state_to_policy,
)


def test_degrees_state_converts_rad_to_deg() -> None:
    """A degrees policy is fed rad->deg-converted state; the gripper is untouched."""
    robot_state = np.array([math.pi / 2, -math.pi, 0.5])  # radians (arm, arm, gripper)
    policy_is_gripper = [False, False, True]
    out = _robot_state_to_policy(
        robot_state, None, joint_units_are_degrees=True, policy_is_gripper=policy_is_gripper
    )
    assert out[0] == 90.0
    assert out[1] == -180.0
    assert out[2] == 0.5  # gripper left as-is


def test_degrees_action_converts_deg_to_rad() -> None:
    """A degrees policy's action is deg->rad-converted before the HAL; gripper untouched."""
    policy_action = np.array([90.0, -180.0, 42.0])  # degrees (arm, arm, gripper)
    policy_is_gripper = [False, False, True]
    out = _policy_action_to_robot(
        policy_action, None, joint_units_are_degrees=True, policy_is_gripper=policy_is_gripper
    )
    assert math.isclose(out[0], math.pi / 2)
    assert math.isclose(out[1], -math.pi)
    assert out[2] == 42.0  # gripper left as-is


def test_radians_is_identity() -> None:
    """A radians checkpoint (joint_units_are_degrees=False) → no conversion."""
    state = np.array([1.0, -2.0, 0.3])
    action = np.array([0.7, -1.1, 0.9])
    gripper = [False, False, True]
    np.testing.assert_array_equal(
        _robot_state_to_policy(
            state, None, joint_units_are_degrees=False, policy_is_gripper=gripper
        ),
        state,
    )
    np.testing.assert_array_equal(
        _policy_action_to_robot(
            action, None, joint_units_are_degrees=False, policy_is_gripper=gripper
        ),
        action,
    )


def test_conversion_round_trips_through_reorder() -> None:
    """deg->rad->deg round-trips even with a robot<->policy permutation."""
    robot_state = np.array([0.1, 0.2, 0.3])
    perm = [2, 0, 1]  # robot index i maps to policy index perm[i]
    gripper = [False, False, False]
    policy = _robot_state_to_policy(
        robot_state, perm, joint_units_are_degrees=True, policy_is_gripper=gripper
    )
    back = _policy_action_to_robot(
        policy, perm, joint_units_are_degrees=True, policy_is_gripper=gripper
    )
    np.testing.assert_allclose(back, robot_state)


def test_gripper_scale_maps_normalized_hal_to_lerobot_motor_units() -> None:
    """SO-101 policy sees [0,100], while the HAL remains normalized [0,1]."""
    robot_state = np.array([0.5])
    policy = _robot_state_to_policy(
        robot_state,
        None,
        joint_units_are_degrees=True,
        policy_is_gripper=[True],
        policy_gripper_scale=100.0,
    )
    assert policy[0] == 50.0

    robot = _policy_action_to_robot(
        policy,
        None,
        joint_units_are_degrees=True,
        policy_is_gripper=[True],
        policy_gripper_scale=100.0,
    )
    assert robot[0] == 0.5


def test_missing_policy_feature_names_falls_back_to_robot_gripper_role() -> None:
    """SmolVLA config has only action shape; the robot manifest still identifies gripper."""
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    adapter = SimpleNamespace(
        _policy=SimpleNamespace(
            config=SimpleNamespace(
                output_features={"action": SimpleNamespace(shape=(6,))},
            )
        )
    )

    permutation, grippers = _build_joint_permutation(
        adapter=adapter,
        description=description,
    )

    assert permutation is None
    assert grippers == [False, False, False, False, False, True]
