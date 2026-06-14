"""ADR-0020 PR-H — exercise envelope_loader against every in-tree manifest.

This is the production envelope-loading flow:

* every ``robots/<id>/robot.yaml`` is a real ``RobotDescription``;
* every ``rskills/<id>/rskill.yaml`` is a real ``RSkillManifest``;
* the loader intersects them, validates the result, and writes a flat
  YAML the C++ safety kernel slurps at ``on_configure``.

By cross-product-testing every (robot, skill) pair we catch envelope
drift before the C++ kernel sees a stale or unmatched envelope on a
real run. CLAUDE.md §1.11: no mocks — real fixtures only.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import yaml
from openral_core import RobotDescription, RSkillManifest
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import (
    EnvelopeIntersection,
    compute_intersection,
    kernel_params_from_envelope,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _real_robots() -> list[pathlib.Path]:
    paths = sorted((_REPO_ROOT / "robots").glob("*/robot.yaml"))
    if not paths:
        pytest.skip("no real robot manifests under robots/")
    return paths


def _real_skills() -> list[pathlib.Path]:
    paths = sorted((_REPO_ROOT / "rskills").glob("*/rskill.yaml"))
    if not paths:
        pytest.skip("no real rskill manifests under rskills/")
    return paths


@pytest.mark.parametrize("robot_path", _real_robots(), ids=lambda p: p.parent.name)
def test_intersection_with_no_skill_uses_real_robot(robot_path: pathlib.Path) -> None:
    """Each robot's safety: block intersects cleanly when no skill is loaded."""
    robot = RobotDescription.from_yaml(str(robot_path))
    intersection = compute_intersection(robot, None)
    assert isinstance(intersection, EnvelopeIntersection)
    assert intersection.robot_name == robot.name
    assert intersection.rskill_id == ""
    # n_dof must equal the number of revolute/prismatic/continuous joints.
    actuated = sum(
        1 for j in robot.joints if j.joint_type.value in {"revolute", "prismatic", "continuous"}
    )
    assert intersection.n_dof == actuated
    assert len(intersection.joint_position_min) == actuated
    assert len(intersection.joint_torque_max) == actuated
    # max_* fields finite or +inf — never NaN.
    for name in (
        "max_ee_speed_m_s",
        "max_force_n",
        "max_torque_nm",
        "max_ee_accel_m_s2",
        "contact_force_threshold_n",
    ):
        v = getattr(intersection, name)
        assert math.isfinite(v) or math.isinf(v), f"{name} = {v}"


@pytest.mark.parametrize("skill_path", _real_skills(), ids=lambda p: p.parent.name)
def test_in_tree_skills_load_without_envelope_field(skill_path: pathlib.Path) -> None:
    """Pre-existing skill manifests (no envelope:) still parse — backwards-compat."""
    skill = RSkillManifest.from_yaml(str(skill_path))
    assert skill.envelope is None, (
        f"{skill_path} has envelope set; should be None unless explicitly added"
    )


def test_cross_product_so100_against_compatible_skills(tmp_path: pathlib.Path) -> None:
    """For each so100-targeted skill, the envelope intersects cleanly."""
    so100_path = _REPO_ROOT / "robots/so100_follower/robot.yaml"
    robot = RobotDescription.from_yaml(str(so100_path))
    skills = [
        s for s in _real_skills() if "so100" in RSkillManifest.from_yaml(str(s)).embodiment_tags
    ]
    if not skills:
        pytest.skip("no so100-targeted in-tree skills found")
    for s in skills:
        skill = RSkillManifest.from_yaml(str(s))
        intersection = compute_intersection(robot, skill)
        assert intersection.robot_name == "so100_follower"
        assert intersection.rskill_id == skill.name


def test_real_so100_envelope_round_trips_to_kernel_params() -> None:
    """End-to-end: load SO-100 manifest → EnvelopeIntersection → ROS param dict.

    Replaces the old write_envelope_file path (deleted in ADR-0020
    PR-K): the kernel reads each canonical field as a ROS parameter,
    not as a YAML file.
    """
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/so100_follower/robot.yaml"))
    intersection = compute_intersection(robot, None)
    params = kernel_params_from_envelope(intersection)
    assert params["robot_name"] == "so100_follower"
    # n_dof matches the SO-100 manifest (5 arm + 1 gripper = 6).
    assert params["n_dof"] == 6
    assert len(params["joint_position_min"]) == 6  # type: ignore[arg-type]
    assert intersection.n_dof == 6


def test_synthetic_looser_skill_is_rejected_against_so100(tmp_path: pathlib.Path) -> None:
    """A skill that loosens the robot ceiling must be rejected at intersect."""
    # Build a synthetic loose skill (max_force_n=999 vs robot's 10).
    skill_dict = {
        "schema_version": "0.1",
        "name": "openral/rskill-loose-attacker",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "vla",
        "model_family": "smolvla",
        "embodiment_tags": ["so100_follower"],
        "runtime": "pytorch",
        "weights_uri": "hf://lerobot/smolvla_base@main",
        "chunk_size": 16,
        "latency_budget": {"per_chunk_ms": 100.0},
        # description + actions are required by RSkillManifest (ADR-0022,
        # surfaced to the reasoner tool palette); this synthetic fixture
        # predates that requirement and must carry them to validate.
        "description": "Synthetic attacker skill that loosens the force ceiling.",
        "actions": ["reach"],
        "actuators_required": [
            {"kind": "joint_position", "control_mode_semantics": {"mode": "absolute"}}
        ],
        "processors": {
            "preprocessor_uri": "hf://lerobot/smolvla_base/policy_preprocessor.json",
            "postprocessor_uri": "hf://lerobot/smolvla_base/policy_postprocessor.json",
        },
        "envelope": {"max_force_n": 999.0},
    }
    skill_path = tmp_path / "loose.yaml"
    skill_path.write_text(yaml.safe_dump(skill_dict), encoding="utf-8")
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/so100_follower/robot.yaml"))
    skill = RSkillManifest.from_yaml(str(skill_path))
    with pytest.raises(ROSConfigError, match="max_force_n"):
        compute_intersection(robot, skill)
