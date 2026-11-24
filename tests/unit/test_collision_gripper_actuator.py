"""`_gripper_actuator` picks the actuator the MuJoCo twin writes a gripper to.

`MujocoTwinHAL` resolves a gripper's actuator as its own `actuator_index`, else
the joint's `sim.actuator_index` entry (else the joint's list position when no
map exists). The lowering's normalised-gripper check must inspect that same
actuator: on r1pro the right gripper sits at list position 20 but drives
actuator 19, and reading the position made the check refuse a correct twin.
Real manifests only (CLAUDE.md §1.11); no MJCF or asset download needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")

from openral_core import RobotDescription
from openral_safety.urdf_lowering import _gripper_actuator

REPO = Path(__file__).resolve().parents[2]


def _robot(robot_id: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(REPO / "robots" / robot_id / "robot.yaml"))


def _grippers(robot: RobotDescription) -> dict[str, int]:
    assert robot.sim is not None
    return {g.joint or "": _gripper_actuator(robot, g) for g in robot.sim.grippers}


def test_joint_map_wins_over_list_position() -> None:
    # r1pro: no per-gripper actuator_index; the sim.actuator_index map names 11 and 19
    # while the joints list puts the grippers at 11 and 20.
    assert _grippers(_robot("r1pro")) == {
        "left_gripper_finger_joint1": 11,
        "right_gripper_finger_joint1": 19,
    }


def test_gripper_actuator_index_wins_over_joint_map() -> None:
    assert _grippers(_robot("makermods_metal_station")) == {
        "left_gripper": 6,
        "right_gripper": 13,
    }


def test_map_without_the_gripper_joint_has_no_actuator() -> None:
    robot = _robot("r1pro")
    assert robot.sim is not None and robot.sim.actuator_index is not None
    amap = {k: v for k, v in robot.sim.actuator_index.items() if k != "right_gripper_finger_joint1"}
    robot = robot.model_copy(update={"sim": robot.sim.model_copy(update={"actuator_index": amap})})
    assert _grippers(robot)["right_gripper_finger_joint1"] == -1
