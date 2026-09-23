# SPDX-License-Identifier: Apache-2.0
"""The skill runner ticks at the manifest's control rate unless told otherwise.

Issue #303: the real ros2_control HAL derives every trajectory deadline from
`action_spec.control_freq_hz`, so the runner must tick at that same value or
the controller is again asked to cover each step in the wrong time.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from openral_core.schemas import RobotDescription

REPO_ROOT = Path(__file__).resolve().parents[2]

runner = importlib.import_module("openral_rskill_ros.rskill_runner_node")


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [("robots/openarm/robot.yaml", 30.0), ("robots/aloha_bimanual/robot.yaml", 50.0)],
)
def test_an_unset_param_reads_the_manifest(manifest: str, expected: float) -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / manifest))
    assert runner.resolve_control_rate_hz(0.0, desc) == expected


def test_an_explicit_param_overrides_the_manifest() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    assert runner.resolve_control_rate_hz(15.0, desc) == 15.0


def test_a_manifest_without_a_control_rate_is_reported_not_guessed() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    rateless = desc.model_copy(update={"action_spec": None})
    assert runner.resolve_control_rate_hz(0.0, rateless) is None
    assert runner.resolve_control_rate_hz(0.0, None) is None


@pytest.mark.parametrize("robot", ["ur5e", "ur10e", "franka_panda", "sawyer", "openarm"])
def test_every_ros2_control_robot_declares_its_rate(robot: str) -> None:
    """Their real HALs refuse to build without it, so the committed manifests must carry it."""
    desc = RobotDescription.from_yaml(str(REPO_ROOT / f"robots/{robot}/robot.yaml"))
    assert runner.resolve_control_rate_hz(0.0, desc) == 30.0


# ── Starting-pose ramp: declared in the manifest, never derived ───────────────


def test_the_ramp_comes_from_the_manifest_not_from_rated_joint_limits() -> None:
    """Half the slowest rated velocity_limit would put the OpenArm at 2 rad/s (issue #303)."""
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    assert runner.resolve_starting_pose_ramp(0.0, 0.0, desc) == (0.5, 0.05)
    slowest = min(j.velocity_limit for j in desc.joints if j.velocity_limit)
    assert slowest * desc.safety.max_joint_speed_factor > 0.5


@pytest.mark.parametrize(
    "robot", ["so101_follower", "ur5e", "franka_panda", "sawyer", "aloha_bimanual", "galaxea_a1"]
)
def test_every_real_robot_declares_its_approach(robot: str) -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / f"robots/{robot}/robot.yaml"))
    speed, tolerance = runner.resolve_starting_pose_ramp(0.0, 0.0, desc)
    assert speed > 0.0 and tolerance > 0.0


def test_explicit_ramp_params_win() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    assert runner.resolve_starting_pose_ramp(0.2, 0.01, desc) == (0.2, 0.01)
    assert runner.resolve_starting_pose_ramp(0.2, 0.0, desc) == (0.2, 0.05)


def test_a_manifest_without_the_approach_values_is_refused_not_guessed() -> None:
    from openral_core.exceptions import ROSConfigError

    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    bare = desc.model_copy(
        update={
            "safety": desc.safety.model_copy(
                update={
                    "starting_pose_max_joint_speed_rad_s": None,
                    "starting_pose_tolerance_rad": None,
                }
            )
        }
    )
    with pytest.raises(ROSConfigError, match="starting_pose_max_joint_speed_rad_s"):
        runner.resolve_starting_pose_ramp(0.0, 0.0, bare)
    with pytest.raises(ROSConfigError, match="starting_pose_tolerance_rad"):
        runner.resolve_starting_pose_ramp(0.3, 0.0, bare)
    with pytest.raises(ROSConfigError):
        runner.resolve_starting_pose_ramp(0.0, 0.0, None)
