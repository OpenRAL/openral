"""Tests for ``openral_safety.envelope_loader``.

Exercises the robot ⨯ skill envelope intersection algebra, the
loosening-rejection contract (ADR-0018 §5), and the flat-YAML kernel
bridge format. No mocks (CLAUDE.md §1.11): every fixture is a real
:class:`openral_core.RobotDescription` / :class:`RSkillManifest`
constructed from real in-tree YAMLs.
"""

from __future__ import annotations

import math
import pathlib

import pytest
from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    RSkillManifest,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import (
    compute_intersection,
    kernel_params_from_envelope,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _so100() -> RobotDescription:
    """Load the real SO-100 follower manifest."""
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots/so100_follower/robot.yaml"))


def _minimal_skill_dict() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "name": "openral/rskill-test-skill",
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
        "actuators_required": [
            {
                "kind": "joint_position",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ],
        "processors": {
            "preprocessor_uri": "hf://lerobot/smolvla_base/policy_preprocessor.json",
            "postprocessor_uri": "hf://lerobot/smolvla_base/policy_postprocessor.json",
        },
        "description": "Envelope-loader test rSkill fixture.",
        "actions": ["generalist"],
    }


def _skill_with_envelope(envelope: dict[str, object]) -> RSkillManifest:
    d = _minimal_skill_dict()
    d["envelope"] = envelope
    return RSkillManifest.model_validate(d)


def _toy_robot(
    *,
    workspace_min: tuple[float, float, float] | None = (-0.4, -0.4, 0.0),
    workspace_max: tuple[float, float, float] | None = (0.4, 0.4, 0.6),
    max_force_n: float = 10.0,
    max_ee_speed: float = 0.5,
    deadman: bool = True,
    n_joints: int = 3,
    pos_limits: tuple[float, float] | None = (-1.0, 1.0),
    velocity_limit: float | None = 4.5,
    effort_limit: float | None = 5.0,
) -> RobotDescription:
    """Build a toy RobotDescription for envelope-intersection unit tests.

    Real fixtures (so100, franka_panda) cover the integration paths; this
    toy builder makes per-joint and box-shape edge cases tractable.
    """
    joints = [
        JointSpec(
            name=f"j{i}",
            joint_type=JointType.REVOLUTE,
            parent_link="base" if i == 0 else f"link_{i - 1}",
            child_link=f"link_{i}",
            position_limits=pos_limits,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
            actuator_kind="servo",
        )
        for i in range(n_joints)
    ]
    return RobotDescription(
        name="toy_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=joints,
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["toy"],
        ),
        safety=SafetyEnvelope(
            workspace_box_min_xyz=workspace_min,
            workspace_box_max_xyz=workspace_max,
            max_force_n=max_force_n,
            max_ee_speed_m_s=max_ee_speed,
            deadman_required=deadman,
        ),
    )


# ── compute_intersection: no skill ──────────────────────────────────────────


class TestIntersectionNoSkill:
    def test_robot_alone_yields_robot_ceiling(self) -> None:
        robot = _toy_robot()
        intersection = compute_intersection(robot, None)
        assert intersection.robot_name == "toy_robot"
        assert intersection.rskill_id == ""
        assert intersection.rskill_revision == ""
        assert intersection.max_force_n == 10.0
        assert intersection.max_ee_speed_m_s == 0.5
        assert intersection.workspace_box_min_xyz == (-0.4, -0.4, 0.0)
        assert intersection.workspace_box_max_xyz == (0.4, 0.4, 0.6)
        assert intersection.deadman_required is True

    def test_n_dof_counts_only_actuated_joints(self) -> None:
        robot = _toy_robot(n_joints=5)
        intersection = compute_intersection(robot, None)
        assert intersection.n_dof == 5
        assert len(intersection.joint_position_min) == 5
        assert len(intersection.joint_position_max) == 5
        assert len(intersection.joint_velocity_max) == 5
        assert len(intersection.joint_torque_max) == 5

    def test_joint_velocity_is_pre_multiplied_by_speed_factor(self) -> None:
        # max_joint_speed_factor defaults to 0.7; toy robot's velocity_limit=4.5.
        robot = _toy_robot(velocity_limit=4.5)
        intersection = compute_intersection(robot, None)
        for v in intersection.joint_velocity_max:
            assert math.isclose(v, 4.5 * 0.7)

    def test_real_so100_intersection_no_skill(self) -> None:
        robot = _so100()
        intersection = compute_intersection(robot, None)
        assert intersection.robot_name == "so100_follower"
        # SO-100 has 6 actuated joints (5 arm + 1 gripper).
        assert intersection.n_dof == 6
        # From robots/so100_follower/robot.yaml: max_force_n=10.0.
        assert intersection.max_force_n == 10.0
        # Workspace box from the manifest.
        assert intersection.workspace_box_min_xyz == (-0.4, -0.4, 0.0)
        assert intersection.workspace_box_max_xyz == (0.4, 0.4, 0.6)

    def test_missing_joint_limits_become_inf(self) -> None:
        robot = _toy_robot(pos_limits=None, velocity_limit=None, effort_limit=None)
        intersection = compute_intersection(robot, None)
        for lo, hi, v, t in zip(
            intersection.joint_position_min,
            intersection.joint_position_max,
            intersection.joint_velocity_max,
            intersection.joint_torque_max,
            strict=True,
        ):
            assert math.isinf(lo) and lo < 0
            assert math.isinf(hi) and hi > 0
            assert math.isinf(v)
            assert math.isinf(t)


# ── compute_intersection: tighter skill ─────────────────────────────────────


class TestIntersectionTighterSkill:
    def test_tighter_max_force_wins(self) -> None:
        robot = _toy_robot(max_force_n=50.0)
        skill = _skill_with_envelope({"max_force_n": 20.0})
        intersection = compute_intersection(robot, skill)
        assert intersection.max_force_n == 20.0
        assert intersection.rskill_id == "openral/rskill-test-skill"
        assert intersection.rskill_revision == "0.1.0"

    def test_tighter_max_ee_speed_wins(self) -> None:
        robot = _toy_robot(max_ee_speed=1.0)
        skill = _skill_with_envelope({"max_ee_speed_m_s": 0.2})
        intersection = compute_intersection(robot, skill)
        assert intersection.max_ee_speed_m_s == 0.2

    def test_tighter_workspace_box_wins(self) -> None:
        robot = _toy_robot(
            workspace_min=(-1.0, -1.0, 0.0),
            workspace_max=(1.0, 1.0, 1.0),
        )
        skill = _skill_with_envelope(
            {
                "workspace_box_min_xyz": [-0.3, -0.3, 0.1],
                "workspace_box_max_xyz": [0.3, 0.3, 0.8],
            }
        )
        intersection = compute_intersection(robot, skill)
        assert intersection.workspace_box_min_xyz == (-0.3, -0.3, 0.1)
        assert intersection.workspace_box_max_xyz == (0.3, 0.3, 0.8)

    def test_tighter_speed_factor_is_propagated_to_per_joint_velocity(self) -> None:
        # Robot factor 0.7, skill 0.4 → joint_velocity_max scaled by 0.4.
        robot = _toy_robot(velocity_limit=4.5)
        skill = _skill_with_envelope({"max_joint_speed_factor": 0.4})
        intersection = compute_intersection(robot, skill)
        for v in intersection.joint_velocity_max:
            assert math.isclose(v, 4.5 * 0.4)

    def test_skill_setting_deadman_only_tightens_logical_or(self) -> None:
        # robot=False, skill=True → required (OR).
        robot = _toy_robot(deadman=False)
        skill = _skill_with_envelope({"deadman_required": True})
        intersection = compute_intersection(robot, skill)
        assert intersection.deadman_required is True


# ── compute_intersection: loosening rejected ────────────────────────────────


class TestLooseningRejected:
    def test_max_force_loosening_rejected(self) -> None:
        robot = _toy_robot(max_force_n=10.0)
        skill = _skill_with_envelope({"max_force_n": 100.0})
        with pytest.raises(ROSConfigError, match="max_force_n"):
            compute_intersection(robot, skill)

    def test_max_ee_speed_loosening_rejected(self) -> None:
        robot = _toy_robot(max_ee_speed=0.5)
        skill = _skill_with_envelope({"max_ee_speed_m_s": 5.0})
        with pytest.raises(ROSConfigError, match="max_ee_speed_m_s"):
            compute_intersection(robot, skill)

    def test_workspace_loosening_on_min_rejected(self) -> None:
        robot = _toy_robot(
            workspace_min=(-0.4, -0.4, 0.0),
            workspace_max=(0.4, 0.4, 0.6),
        )
        skill = _skill_with_envelope(
            {
                "workspace_box_min_xyz": [-1.0, -0.4, 0.0],  # Loose on x.
                "workspace_box_max_xyz": [0.4, 0.4, 0.6],
            }
        )
        with pytest.raises(ROSConfigError, match="workspace_box_min_xyz"):
            compute_intersection(robot, skill)

    def test_workspace_loosening_on_max_rejected(self) -> None:
        robot = _toy_robot(
            workspace_min=(-0.4, -0.4, 0.0),
            workspace_max=(0.4, 0.4, 0.6),
        )
        skill = _skill_with_envelope(
            {
                "workspace_box_min_xyz": [-0.4, -0.4, 0.0],
                "workspace_box_max_xyz": [0.4, 0.4, 99.0],  # Loose on z-max.
            }
        )
        with pytest.raises(ROSConfigError, match="workspace_box_max_xyz"):
            compute_intersection(robot, skill)

    def test_deadman_clear_rejected(self) -> None:
        # robot=True, skill=False → loosening.
        robot = _toy_robot(deadman=True)
        skill = _skill_with_envelope({"deadman_required": False})
        with pytest.raises(ROSConfigError, match="deadman_required"):
            compute_intersection(robot, skill)

    def test_workspace_half_set_rejected(self) -> None:
        robot = _toy_robot()
        skill = _skill_with_envelope(
            {"workspace_box_min_xyz": [-0.3, -0.3, 0.0]}  # No max.
        )
        with pytest.raises(ROSConfigError, match="both must be set together"):
            compute_intersection(robot, skill)


# ── kernel_params_from_envelope ─────────────────────────────────────────────


class TestKernelParamsFromEnvelope:
    """The Python → C++-kernel-ROS-params converter (ADR-0020 PR-K).

    The legacy ``write_envelope_file`` / ``load_envelope_files`` helpers
    that flattened the envelope to a YAML file the kernel slurped were
    removed in the PR that landed this contract. There is exactly one
    transport: per-field ROS parameters.
    """

    def test_round_trips_robot_ceiling(self) -> None:
        robot = _toy_robot()
        intersection = compute_intersection(robot, None)
        params = kernel_params_from_envelope(intersection)
        assert params["n_dof"] == 3
        assert params["robot_name"] == "toy_robot"
        assert params["joint_position_min"] == list(intersection.joint_position_min)
        assert params["joint_position_max"] == list(intersection.joint_position_max)
        assert params["joint_velocity_max"] == list(intersection.joint_velocity_max)
        assert params["joint_torque_max"] == list(intersection.joint_torque_max)
        assert params["max_force_n"] == 10.0
        assert params["workspace_box_min_xyz"] == [-0.4, -0.4, 0.0]
        assert params["workspace_box_max_xyz"] == [0.4, 0.4, 0.6]

    def test_inf_joint_limits_pass_through(self) -> None:
        """Unbounded joints (no manifest limit) become ±inf — kernel reads as
        ``no enforcement on this joint``."""
        robot = _toy_robot(pos_limits=None, velocity_limit=None, effort_limit=None)
        intersection = compute_intersection(robot, None)
        params = kernel_params_from_envelope(intersection)
        for v in params["joint_position_max"]:  # type: ignore[union-attr]
            assert math.isinf(v) and v > 0
        for v in params["joint_position_min"]:  # type: ignore[union-attr]
            assert math.isinf(v) and v < 0

    def test_workspace_box_omitted_when_unset(self) -> None:
        """No workspace box → keys absent; launch_ros rejects empty
        ``double_array`` parameters, so omission is the contract."""
        robot = _toy_robot(workspace_min=None, workspace_max=None)
        intersection = compute_intersection(robot, None)
        params = kernel_params_from_envelope(intersection)
        assert "workspace_box_min_xyz" not in params
        assert "workspace_box_max_xyz" not in params
