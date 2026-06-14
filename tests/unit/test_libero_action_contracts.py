"""LIBERO worked-example skills declare a cartesian (OSC) action contract.

`pi05-libero-nf4` and `smolvla-libero` emit a 7-D LIBERO action that is a 6-D
OSC end-effector delta plus a 1-D gripper command
(:class:`ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER`). Declaring the
representation in ``rskill.yaml`` makes deploy-sim expand the vector into
``cartesian_delta`` + ``gripper_position`` slots (via
:func:`canonical_slots_for_representation`) instead of defaulting the whole
vector to ``JOINT_POSITION``, which the joint-space franka envelope rejects.
"""

from pathlib import Path

import pytest
from openral_core import RobotDescription, RSkillManifest
from openral_core.schemas import (
    ActionRepresentation,
    ControlMode,
    canonical_slots_for_representation,
)

_REPO = Path(__file__).resolve().parents[2]


def _franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / "franka_panda" / "robot.yaml"))


@pytest.mark.parametrize("skill", ["pi05-libero-nf4", "smolvla-libero"])
def test_libero_skill_declares_cartesian_action_contract(skill: str) -> None:
    m = RSkillManifest.from_yaml(str(_REPO / "rskills" / skill / "rskill.yaml"))
    assert m.action_contract is not None
    assert m.action_contract.representation is ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER
    assert m.action_contract.dim == 7
    slots = canonical_slots_for_representation(
        m.action_contract.representation,
        dim=m.action_contract.dim,
        description=_franka(),
    )
    assert [s.control_mode for s in slots] == [
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
    ]
