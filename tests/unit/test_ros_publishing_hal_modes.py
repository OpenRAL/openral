"""Per-control-mode serialisation tests for ``ROSPublishingHAL._flatten_action_payload``
(ADR-0028b step 4).

Before this ADR the serialiser rejected every non-joint mode with
``ROSConfigError``. After: joint position / velocity / torque /
trajectory + cartesian delta / twist + body twist + gripper
position / binary all flatten onto the ActionChunk wire format. The
remaining unsupported modes (cartesian_pose carrying a Pose6D,
foot_placement, dex_hand_joint) still raise — tracked but unwired.

These tests cover the pure function ``_flatten_action_payload`` so we
don't need a live ROS 2 publisher to exercise the dispatch.
"""

from __future__ import annotations

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSConfigError
from openral_runner.ros_publishing_hal import ROSPublishingHAL

_flatten = ROSPublishingHAL._flatten_action_payload


# ─── Joint modes ─────────────────────────────────────────────────────────────


def test_joint_position_single_step() -> None:
    a = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.1, 0.2, 0.3, 0.4]],
    )
    flat, n_dof, horizon = _flatten(a)
    assert flat == [0.1, 0.2, 0.3, 0.4]
    assert n_dof == 4
    assert horizon == 1


def test_joint_position_horizon_3_chunk() -> None:
    a = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=3,
        joint_targets=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )
    flat, n_dof, horizon = _flatten(a)
    assert flat == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert n_dof == 2
    assert horizon == 3


def test_joint_velocity_and_torque_route_to_their_fields() -> None:
    av = Action(
        control_mode=ControlMode.JOINT_VELOCITY,
        horizon=1,
        joint_velocities=[[1.0, 2.0, 3.0]],
    )
    assert _flatten(av) == ([1.0, 2.0, 3.0], 3, 1)
    at = Action(
        control_mode=ControlMode.JOINT_TORQUE,
        horizon=1,
        joint_torques=[[5.0, -5.0]],
    )
    assert _flatten(at) == ([5.0, -5.0], 2, 1)


def test_joint_trajectory_routes_to_joint_targets() -> None:
    """JOINT_TRAJECTORY shares the joint_targets field with JOINT_POSITION."""
    a = Action(
        control_mode=ControlMode.JOINT_TRAJECTORY,
        horizon=2,
        joint_targets=[[0.0, 0.0], [1.0, 1.0]],
    )
    assert _flatten(a) == ([0.0, 0.0, 1.0, 1.0], 2, 2)


def test_joint_position_empty_payload_rejected() -> None:
    a = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=None)
    with pytest.raises(ROSConfigError, match="empty payload"):
        _flatten(a)


# ─── Cartesian delta / twist ─────────────────────────────────────────────────


def test_cartesian_delta_flat_6d() -> None:
    a = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.01, 0.02, 0.03, 0.1, 0.2, 0.3)],
        ee_name="panda_hand",
        frame_id="panda_link0",
    )
    flat, n_dof, horizon = _flatten(a)
    assert flat == [0.01, 0.02, 0.03, 0.1, 0.2, 0.3]
    assert n_dof == 6
    assert horizon == 1


def test_cartesian_twist_flat_6d() -> None:
    a = Action(
        control_mode=ControlMode.CARTESIAN_TWIST,
        horizon=1,
        cartesian_twist=[(0.5, 0.0, 0.0, 0.0, 0.0, 0.1)],
        ee_name="ee",
        frame_id="f",
    )
    assert _flatten(a) == ([0.5, 0.0, 0.0, 0.0, 0.0, 0.1], 6, 1)


def test_cartesian_wrong_width_rejected() -> None:
    a = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.01, 0.02, 0.03)],  # 3-D instead of 6-D
        ee_name="ee",
        frame_id="f",
    )
    with pytest.raises(ROSConfigError, match="width 6"):
        _flatten(a)


# ─── Body twist ──────────────────────────────────────────────────────────────


def test_body_twist_flat_6d() -> None:
    a = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[(1.0, 0.0, 0.0, 0.0, 0.0, 0.5)],
        frame_id="base_link",
    )
    flat, n_dof, horizon = _flatten(a)
    assert flat == [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    assert n_dof == 6
    assert horizon == 1


# ─── Gripper ─────────────────────────────────────────────────────────────────


def test_gripper_position_flat_1d() -> None:
    a = Action(
        control_mode=ControlMode.GRIPPER_POSITION,
        horizon=1,
        gripper=[-0.989],
        ee_name="panda_gripper",
    )
    flat, n_dof, horizon = _flatten(a)
    assert flat == [-0.989]
    assert n_dof == 1
    assert horizon == 1


def test_gripper_binary_flat_1d() -> None:
    a = Action(
        control_mode=ControlMode.GRIPPER_BINARY,
        horizon=1,
        gripper=[1.0],
        ee_name="g",
    )
    assert _flatten(a) == ([1.0], 1, 1)


def test_gripper_empty_payload_rejected() -> None:
    a = Action(control_mode=ControlMode.GRIPPER_POSITION, horizon=1, gripper=None, ee_name="g")
    with pytest.raises(ROSConfigError, match="empty gripper payload"):
        _flatten(a)


# ─── Modes still unsupported (tracked) ───────────────────────────────────────


def test_cartesian_pose_still_rejected() -> None:
    """CARTESIAN_POSE carries a :class:`Pose6D` not a flat tuple — not yet wired."""
    from openral_core import Pose6D

    a = Action(
        control_mode=ControlMode.CARTESIAN_POSE,
        horizon=1,
        cartesian_pose=[
            Pose6D(
                xyz=(0.5, 0.0, 0.3),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="ee",
            )
        ],
        ee_name="ee",
        frame_id="f",
    )
    with pytest.raises(ROSConfigError, match="does not serialise"):
        _flatten(a)
