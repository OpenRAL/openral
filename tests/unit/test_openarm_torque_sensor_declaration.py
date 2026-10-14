"""OpenArm joints must not declare a joint torque sensor (issue #375).

OpenArm's joint torque is a current-derived estimate from the motor driver,
not a dedicated joint torque sensor, so ``JointSpec.has_torque_sensor`` is
``False`` on every joint. ``anvil_openarm_v2`` is the same hardware. The HAL
constants are held to these manifests by
``test_robot_manifests_match_hal_constants.py``; this pins the value itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import RobotDescription

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("robot_id", ["anvil_openarm_v2", "openarm"])
def test_no_openarm_joint_declares_a_torque_sensor(robot_id: str) -> None:
    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    assert not any(j.has_torque_sensor for j in desc.joints)
