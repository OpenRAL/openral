"""TaskSpace cross-layer contract (ADR-0071).

Exercises the layer-neutral ``TaskSpace`` view + ``task_space_compatible``
against **real** ``robots/`` and ``rskills/`` fixtures — no mocks (CLAUDE.md
§1.11). The motivating task-space audit found the three asset layers do not
share a contract today; these tests pin the new shared object's behavior so the
staged migration in ADR-0071 can build on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    ControlMode,
    RobotDescription,
    RSkillManifest,
    TaskSpace,
    TaskSpaceFamily,
    TaskSpaceSegment,
    task_space_compatible,
)
from openral_core.schemas import _FAMILY_FOR_MODE

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = REPO_ROOT / "robots"
RSKILLS_DIR = REPO_ROOT / "rskills"


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(ROBOTS_DIR / name / "robot.yaml"))


def _rskill(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(RSKILLS_DIR / name / "rskill.yaml"))


def test_every_mode_has_family() -> None:
    """Lockstep: every ControlMode maps to exactly one TaskSpaceFamily."""
    for mode in ControlMode:
        assert mode in _FAMILY_FOR_MODE, f"ControlMode {mode} missing a family"


def test_segment_family_must_match_mode() -> None:
    with pytest.raises(ValueError, match="does not match control_mode"):
        TaskSpaceSegment(
            family=TaskSpaceFamily.JOINT,
            control_mode=ControlMode.CARTESIAN_DELTA,
            width=6,
        )


def test_task_space_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TaskSpace(segments=[])


def test_libero_skill_expands_via_representation() -> None:
    """act-libero declares representation=delta_ee_6d_plus_gripper, no slots.

    from_action_contract must expand it (ADR-0036) into a 6-D cartesian-delta
    segment + a 1-D gripper segment addressed at the Franka's EE — total 7.
    """
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, robot)

    assert space.total_dim == 7
    assert [s.family for s in space.segments] == [
        TaskSpaceFamily.CARTESIAN,
        TaskSpaceFamily.GRIPPER,
    ]
    assert space.control_modes == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
    }
    # The gripper is an explicit 1-D dimension, not folded into the arm slice.
    gripper = next(s for s in space.segments if s.family is TaskSpaceFamily.GRIPPER)
    assert gripper.width == 1
    assert gripper.target == "panda_hand"


def test_joint_skill_falls_back_to_whole_vector() -> None:
    """act-aloha has no slots and a joint representation → one JOINT segment."""
    robot = _robot("aloha_bimanual")
    skill = _rskill("act-aloha")
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, robot)

    assert len(space.segments) == 1
    seg = space.segments[0]
    assert seg.family is TaskSpaceFamily.JOINT
    assert seg.control_mode is ControlMode.JOINT_POSITION
    assert seg.width == skill.action_contract.dim == 14


def test_libero_skill_incompatible_on_real_joint_only_franka() -> None:
    """The audit's representation-vs-actuators contradiction, made executable.

    act-libero's representation implies {cartesian_delta, gripper_position}, but
    franka_panda advertises only joint_position. On REAL hardware the gate must
    report BOTH missing modes — surfacing the contradiction instead of silently
    passing on embodiment-tag match.
    """
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, robot)

    match = task_space_compatible(space, robot, hal_mode="real")
    assert match.ok is False
    joined = " ".join(match.reasons)
    assert "cartesian_delta" in joined
    assert "gripper_position" in joined


def test_libero_skill_compatible_in_sim_via_osc() -> None:
    """The same act-libero skill IS executable on franka in SIM.

    The robosuite OSC packer synthesises cartesian_delta + gripper_position from
    joint commands (SIM_EXECUTABLE_CONTROL_MODES) — matching how it actually runs
    under `openral deploy sim`. The sim/real split mirrors the reasoner gate's
    `_action_executable` (ADR-0036).
    """
    robot = _robot("franka_panda")
    skill = _rskill("act-libero")
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, robot)

    match = task_space_compatible(space, robot, hal_mode="sim")
    assert match.ok is True, match.reasons


def test_compatible_when_robot_advertises_modes() -> None:
    """so101 advertises joint_position + gripper_position; a matching space passes."""
    robot = _robot("so101_follower")
    space = TaskSpace(
        segments=[
            TaskSpaceSegment(
                family=TaskSpaceFamily.JOINT,
                control_mode=ControlMode.JOINT_POSITION,
                width=5,
            ),
            TaskSpaceSegment(
                family=TaskSpaceFamily.GRIPPER,
                control_mode=ControlMode.GRIPPER_POSITION,
                width=1,
                target="gripper",
            ),
        ]
    )
    match = task_space_compatible(space, robot)
    assert match.ok is True, match.reasons


def test_unknown_end_effector_target_is_reported() -> None:
    robot = _robot("franka_panda")
    space = TaskSpace(
        segments=[
            TaskSpaceSegment(
                family=TaskSpaceFamily.GRIPPER,
                control_mode=ControlMode.GRIPPER_POSITION,
                width=1,
                target="nonexistent_hand",
            )
        ]
    )
    match = task_space_compatible(space, robot)
    assert match.ok is False
    assert "nonexistent_hand" in " ".join(match.reasons)
