"""OpenArm arm joints must not declare a joint torque sensor (issue #375).

OpenArm's joint torque is a current-derived estimate from the motor driver,
not a dedicated joint torque sensor. ``JointSpec.has_torque_sensor`` must say
so in both the in-code HAL description and the committed ``robot.yaml``, so
anything that trusts the flag for force decisions does not over-trust the
estimate. ``anvil_openarm_v2`` is the same hardware and carries the same
declaration.

Pure description checks against the real committed manifests — no hardware,
no simulator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import RobotDescription
from openral_hal.anvil_openarm_v2 import ANVIL_OPENARM_V2_DESCRIPTION
from openral_hal.openarm import OPENARM_DESCRIPTION

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ARM_JOINTS = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]

_DESCRIPTIONS: dict[str, RobotDescription] = {
    "openarm": OPENARM_DESCRIPTION,
    "anvil_openarm_v2": ANVIL_OPENARM_V2_DESCRIPTION,
}


def _arm_torque_flags(desc: RobotDescription) -> dict[str, bool]:
    by_name = {j.name: j for j in desc.joints}
    assert set(_ARM_JOINTS) <= set(by_name), "manifest lost an OpenArm arm joint"
    return {name: by_name[name].has_torque_sensor for name in _ARM_JOINTS}


@pytest.mark.parametrize("robot_id", sorted(_DESCRIPTIONS))
def test_hal_description_arm_joints_declare_no_torque_sensor(robot_id: str) -> None:
    flags = _arm_torque_flags(_DESCRIPTIONS[robot_id])
    assert not any(flags.values()), flags


@pytest.mark.parametrize("robot_id", sorted(_DESCRIPTIONS))
def test_robot_yaml_arm_joints_declare_no_torque_sensor(robot_id: str) -> None:
    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    flags = _arm_torque_flags(desc)
    assert not any(flags.values()), flags


@pytest.mark.parametrize("robot_id", sorted(_DESCRIPTIONS))
def test_no_openarm_joint_declares_a_torque_sensor(robot_id: str) -> None:
    """Gripper joints included: nothing on OpenArm has a joint torque sensor."""
    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    assert not any(j.has_torque_sensor for j in desc.joints)
    assert not any(j.has_torque_sensor for j in _DESCRIPTIONS[robot_id].joints)
