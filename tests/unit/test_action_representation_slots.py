"""Unit tests for the representation → ControlMode + canonical-slot helpers (ADR-0036).

Covers :func:`openral_core.schemas.control_modes_for_representation` and
:func:`openral_core.schemas.canonical_slots_for_representation`, the single
source of truth that both the skill_runner (action dispatch) and the
reasoner (palette gate) use to map a VLA's declared
:class:`ActionRepresentation` onto control modes + a typed slot layout.

Validated against the real ``robots/franka_panda/robot.yaml`` fixture —
no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    ActionRepresentation,
    ControlMode,
    canonical_slots_for_representation,
    control_modes_for_representation,
)

_REPO = Path(__file__).resolve().parents[2]


def _franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / "franka_panda" / "robot.yaml"))


def test_delta_ee_6d_plus_gripper_canonical_slots() -> None:
    slots = canonical_slots_for_representation(
        ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER, dim=7, description=_franka()
    )
    assert slots is not None and len(slots) == 2
    cart, grip = slots
    assert cart.control_mode is ControlMode.CARTESIAN_DELTA and cart.range == (0, 5)
    assert cart.ee == "panda_hand" and cart.frame
    assert grip.control_mode is ControlMode.GRIPPER_POSITION and grip.range == (6, 6)


def test_delta_ee_6d_canonical_slots() -> None:
    slots = canonical_slots_for_representation(
        ActionRepresentation.DELTA_EE_6D, dim=6, description=_franka()
    )
    assert slots is not None and len(slots) == 1
    (cart,) = slots
    assert cart.control_mode is ControlMode.CARTESIAN_DELTA and cart.range == (0, 5)
    assert cart.ee == "panda_hand" and cart.frame == "panda_hand"


def test_cartesian_pose_canonical_slots() -> None:
    slots = canonical_slots_for_representation(
        ActionRepresentation.CARTESIAN_POSE, dim=6, description=_franka()
    )
    assert slots is not None and len(slots) == 1
    (cart,) = slots
    assert cart.control_mode is ControlMode.CARTESIAN_POSE and cart.range == (0, 5)
    assert cart.ee == "panda_hand" and cart.frame == "panda_hand"


def test_joint_positions_returns_none_canonical_slots() -> None:
    assert (
        canonical_slots_for_representation(
            ActionRepresentation.JOINT_POSITIONS, dim=8, description=_franka()
        )
        is None
    )


def test_joint_velocities_returns_none_canonical_slots() -> None:
    assert (
        canonical_slots_for_representation(
            ActionRepresentation.JOINT_VELOCITIES, dim=7, description=_franka()
        )
        is None
    )


def test_control_modes_for_representation() -> None:
    assert control_modes_for_representation(ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER) == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
    }
    assert control_modes_for_representation(ActionRepresentation.JOINT_POSITIONS) == {
        ControlMode.JOINT_POSITION
    }
    assert control_modes_for_representation(ActionRepresentation.JOINT_VELOCITIES) == {
        ControlMode.JOINT_VELOCITY
    }
    assert control_modes_for_representation(ActionRepresentation.DELTA_EE_6D) == {
        ControlMode.CARTESIAN_DELTA
    }
    assert control_modes_for_representation(ActionRepresentation.CARTESIAN_POSE) == {
        ControlMode.CARTESIAN_POSE
    }


def test_dim_too_small_raises_config_error() -> None:
    # DELTA_EE_6D_PLUS_GRIPPER needs dim >= 7 (6 cartesian + >=1 gripper).
    with pytest.raises(ROSConfigError, match="dim"):
        canonical_slots_for_representation(
            ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER, dim=6, description=_franka()
        )
    # DELTA_EE_6D needs dim >= 6.
    with pytest.raises(ROSConfigError, match="dim"):
        canonical_slots_for_representation(
            ActionRepresentation.DELTA_EE_6D, dim=5, description=_franka()
        )
    # CARTESIAN_POSE needs dim >= 6.
    with pytest.raises(ROSConfigError, match="dim"):
        canonical_slots_for_representation(
            ActionRepresentation.CARTESIAN_POSE, dim=5, description=_franka()
        )


def test_empty_end_effectors_raises_config_error() -> None:
    franka = _franka()
    no_ee = franka.model_copy(update={"end_effectors": []})
    with pytest.raises(ROSConfigError, match="end_effector"):
        canonical_slots_for_representation(
            ActionRepresentation.DELTA_EE_6D, dim=6, description=no_ee
        )
