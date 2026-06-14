"""Remaining cartesian rSkills declare an explicit OSC action contract (ADR-0036).

These checkpoints all emit a 7-D action that is a 6-D OSC end-effector delta
plus a 1-D gripper command
(:class:`ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER`), even though their
action dim (7) differs from their embodiment's actuated-joint count — the
tell-tale sign the vector is cartesian, not joint-space. Declaring the
representation in ``rskill.yaml`` makes deploy-sim expand the vector into
``cartesian_delta`` + ``gripper_position`` slots (via
:func:`canonical_slots_for_representation`) instead of defaulting the whole
vector to ``JOINT_POSITION``, which the joint-space envelope rejects.

This sweep follows the two LIBERO worked examples already covered by
``test_libero_action_contracts.py``; ``rldx1-ft-simpler-widowx-nf4`` was the
first to declare the representation and is asserted here as the reference.
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

# Each swept skill mapped to the robot fixture for its first embodiment tag.
# The skill emits a 7-D cartesian (6-D OSC EE delta + gripper) action; the
# robot fixture supplies the primary end-effector that names the cartesian /
# gripper slots. The embodiment tag matches the robots/<tag>/robot.yaml dir
# name directly (franka_panda, google_robot, widowx).
_SWEPT_SKILLS: dict[str, str] = {
    "act-libero": "franka_panda",
    "xvla-libero": "franka_panda",
    "molmoact2-libero-nf4": "franka_panda",
    "rldx1-ft-libero-nf4": "franka_panda",
    "rldx1-pt-nf4": "franka_panda",
}

# Already declared before this sweep — the reference manifest.
_REFERENCE_SKILL = "rldx1-ft-simpler-widowx-nf4"
_REFERENCE_ROBOT = "widowx"


def _robot(robot_dir: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / robot_dir / "robot.yaml"))


def _assert_cartesian_contract(skill: str, robot_dir: str) -> None:
    m = RSkillManifest.from_yaml(str(_REPO / "rskills" / skill / "rskill.yaml"))
    assert m.kind == "vla"
    assert m.action_contract is not None
    assert m.action_contract.dim == 7
    assert m.action_contract.representation is ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER
    # No conflicting explicit slots — the canonical layout is derived.
    assert m.action_contract.slots is None
    slots = canonical_slots_for_representation(
        m.action_contract.representation,
        dim=m.action_contract.dim,
        description=_robot(robot_dir),
    )
    assert slots is not None
    assert [s.control_mode for s in slots] == [
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
    ]


@pytest.mark.parametrize(
    "skill,robot_dir", sorted(_SWEPT_SKILLS.items()), ids=sorted(_SWEPT_SKILLS)
)
def test_swept_skill_declares_cartesian_action_contract(skill: str, robot_dir: str) -> None:
    _assert_cartesian_contract(skill, robot_dir)


def test_reference_widowx_skill_already_declares_cartesian_contract() -> None:
    """``rldx1-ft-simpler-widowx-nf4`` is the pre-existing reference."""
    _assert_cartesian_contract(_REFERENCE_SKILL, _REFERENCE_ROBOT)
