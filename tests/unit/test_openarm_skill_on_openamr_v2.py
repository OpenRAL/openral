"""A 16-D OpenArm rSkill runs unchanged on the 20-joint OpenAMR v2.

``robots/openamr_v2/robot.yaml`` leads with three base joints and the lift, then the
exact OpenArm 16 by logical name. The policy must read exactly its 16 joints, by name
(``PolicyIOCodec`` gathers the slot-named joints), never the base and lift as arm
joints; on ``robots/openarm`` (where the slot order is the robot order) nothing changes.

No OpenArm π0.5 rSkill manifest is in-tree (the checkpoints are private), so
``openarm_pi05_manifest`` builds one through the real ``RSkillManifest`` schema,
mirroring the shipped restock skill's contract field for field.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from openral_core import RobotDescription, ROSConfigError, RSkillManifest
from openral_rskill._policy_io import PolicyIOCodec, slot_state_joint_names

_REPO = Path(__file__).resolve().parents[2]
OPENARM_JOINTS = [
    *(f"left_joint{i}" for i in range(1, 8)),
    "left_gripper",
    *(f"right_joint{i}" for i in range(1, 8)),
    "right_gripper",
]


def openarm_pi05_manifest(**overrides: Any) -> RSkillManifest:
    """The OpenArm π0.5 restock skill's manifest (16-D state, four action slots)."""
    raw: dict[str, Any] = {
        "schema_version": "0.1",
        "name": "OpenRAL/rskill-pi05-openarm-restock_shelf-bf16",
        "version": "0.1.0",
        "license": "permissive_research",
        "role": "s1",
        "kind": "vla",
        "model_family": "pi05",
        "embodiment_tags": ["openarm"],
        "sensors_required": [
            {"modality": "rgb", "vla_feature_key": f"observation.images.{cam}"}
            for cam in ("top", "wrist_left", "wrist_right")
        ],
        "actuators_required": [
            {"kind": "joint_position", "control_mode_semantics": {"mode": "delta"}},
            {
                "kind": "gripper_position",
                "control_mode_semantics": {
                    "mode": "absolute",
                    "gripper_convention": "raw_joint_rad",
                },
            },
        ],
        "runtime": "pytorch",
        "quantization": {"dtype": "bf16", "backend": "pytorch"},
        "weights_uri": "hf://OpenRAL/rskill-pi05-openarm-restock_shelf-bf16",
        "description": (
            "π0.5 fine-tuned for the OpenArm v2 bimanual station to restock boxes onto "
            "a shelf: three RGB views and 16-D joint state in, 16-D actions out."
        ),
        "actions": ["pick", "place", "pick_and_place"],
        "objects": ["box", "shelf"],
        "scenes": ["shelf"],
        "state_contract": {"dim": 16},
        "chunk_size": 52,
        "latency_budget": {"per_chunk_ms": 600.0},
        "min_vram_gb": {"bf16": 9.0},
        "processors": {
            "preprocessor_uri": (
                "hf://OpenRAL/rskill-pi05-openarm-restock_shelf-bf16/policy_preprocessor.json"
            ),
            "postprocessor_uri": (
                "hf://OpenRAL/rskill-pi05-openarm-restock_shelf-bf16/policy_postprocessor.json"
            ),
        },
        "image_preprocessing": {"flip_180": False, "aliases": {"top": "context"}},
        "policy_extras": {"chunk_prefetch_at": 30},
        "n_action_steps": 52,
        "action_contract": {
            "dim": 16,
            "joint_units": "radians",
            "slots": [
                {
                    "range": [0, 6],
                    "control_mode": "joint_position",
                    "joint_names": OPENARM_JOINTS[0:7],
                },
                {"range": [7, 7], "control_mode": "gripper_position", "ee": "left_gripper"},
                {
                    "range": [8, 14],
                    "control_mode": "joint_position",
                    "joint_names": OPENARM_JOINTS[8:15],
                },
                {"range": [15, 15], "control_mode": "gripper_position", "ee": "right_gripper"},
            ],
        },
    }
    raw.update(overrides)
    return RSkillManifest.model_validate(raw)


@pytest.fixture(scope="module")
def amr() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots/openamr_v2/robot.yaml"))


@pytest.fixture(scope="module")
def openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml"))


@pytest.fixture(scope="module")
def runner_mod() -> ModuleType:
    """``rskill_runner_node`` loaded past the ROS-gated package init."""
    src = _REPO / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_amr", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── (a) state: the policy reads exactly its 16 joints, by name ──


def test_amr_state_maps_to_the_openarm_16_by_name(amr: RobotDescription) -> None:
    names = [j.name for j in amr.joints]
    # Distinct per-joint values so any shift (base/lift read as an arm joint) shows.
    robot_state = np.arange(len(names), dtype=np.float32) * 0.01 + 0.5
    codec = PolicyIOCodec.from_manifest(openarm_pi05_manifest(), amr)
    policy_state = codec.to_policy_state(robot_state)
    want = [robot_state[names.index(n)] for n in OPENARM_JOINTS]
    assert policy_state.shape == (16,)
    np.testing.assert_array_equal(policy_state, want)
    assert codec.policy_joint_names == OPENARM_JOINTS
    assert [i for i, g in enumerate(codec.policy_is_gripper) if g] == [7, 15]
    assert not codec.is_identity


def test_openarm_state_stays_identity(openarm: RobotDescription) -> None:
    robot_state = np.arange(16, dtype=np.float32) * 0.01 + 0.5
    codec = PolicyIOCodec.from_manifest(openarm_pi05_manifest(), openarm)
    assert codec.state_index is None
    assert codec.is_identity
    np.testing.assert_array_equal(codec.to_policy_state(robot_state), robot_state)


def test_amr_degrees_gather_converts_arms_not_grippers(amr: RobotDescription) -> None:
    """The gathered channels keep their gripper flags: arms deg-converted, grippers not."""
    manifest = openarm_pi05_manifest()
    assert manifest.action_contract is not None
    ac = manifest.action_contract.model_dump(mode="json") | {"joint_units": "degrees"}
    codec = PolicyIOCodec.from_manifest(openarm_pi05_manifest(action_contract=ac), amr)
    state = codec.to_policy_state(np.full(len(amr.joints), np.pi / 2, dtype=np.float32))
    assert state[0] == pytest.approx(90.0)
    assert state[7] == pytest.approx(np.pi / 2)
    assert state[15] == pytest.approx(np.pi / 2)


def test_a_slot_joint_the_robot_lacks_is_refused() -> None:
    so101 = RobotDescription.from_yaml(str(_REPO / "robots/so101_follower/robot.yaml"))
    with pytest.raises(ROSConfigError, match="left_joint1"):
        PolicyIOCodec.from_manifest(openarm_pi05_manifest(), so101)


def test_a_short_robot_state_is_refused(amr: RobotDescription) -> None:
    codec = PolicyIOCodec.from_manifest(openarm_pi05_manifest(), amr)
    with pytest.raises(ROSConfigError, match="robot state has 16 values"):
        codec.to_policy_state(np.zeros(16, dtype=np.float32))


# ── (c) palette admission: the slot-named joints, not the robot's joint count ──


def test_slot_state_joints_admit_the_openarm_skill_on_the_amr(amr: RobotDescription) -> None:
    assert slot_state_joint_names(openarm_pi05_manifest(), amr) == OPENARM_JOINTS


def test_a_skill_without_joint_slots_is_not_admitted(amr: RobotDescription) -> None:
    """A state_contract.dim the slots do not account for gives no slot-derived state."""
    skill = RSkillManifest.from_yaml(str(_REPO / "rskills/pi05-libero-int8/rskill.yaml"))
    assert slot_state_joint_names(skill, amr) is None
    mismatched = openarm_pi05_manifest(state_contract={"dim": 14})
    assert slot_state_joint_names(mismatched, amr) is None


# ── (d) starting_pose over the slot-named joints holds the rest ──


def test_a_16d_starting_pose_expands_holding_base_and_lift(
    runner_mod: ModuleType, amr: RobotDescription
) -> None:
    names = [j.name for j in amr.joints]
    measured = [0.0] * len(names)
    measured[names.index("base_yaw")] = 0.3
    measured[names.index("lift_joint")] = 0.25
    pose = [0.1 * (k + 1) for k in range(16)]
    full = runner_mod.expand_starting_pose(pose, openarm_pi05_manifest(), amr, measured)
    assert len(full) == len(names)
    assert full[:4] == [0.0, 0.0, 0.3, 0.25]
    assert [full[names.index(n)] for n in OPENARM_JOINTS] == pytest.approx(pose)


def test_a_full_width_starting_pose_passes_through(
    runner_mod: ModuleType, amr: RobotDescription, openarm: RobotDescription
) -> None:
    pose = [0.01 * k for k in range(16)]
    assert runner_mod.expand_starting_pose(pose, openarm_pi05_manifest(), openarm, []) == pose
    pose20 = [0.01 * k for k in range(20)]
    assert runner_mod.expand_starting_pose(pose20, openarm_pi05_manifest(), amr, []) == pose20


def test_a_starting_pose_of_neither_width_is_refused(
    runner_mod: ModuleType, amr: RobotDescription
) -> None:
    with pytest.raises(ROSConfigError, match="starting_pose has 15 values"):
        runner_mod.expand_starting_pose([0.0] * 15, openarm_pi05_manifest(), amr, [0.0] * 20)


def test_a_held_joint_past_its_limit_does_not_refuse_the_pose(
    runner_mod: ModuleType, amr: RobotDescription
) -> None:
    """A lift read 2 mm below 0 is held, not commanded: only named joints are range-checked."""
    names = [j.name for j in amr.joints]
    measured = [0.0] * len(names)
    measured[names.index("lift_joint")] = -0.002
    full = runner_mod.expand_starting_pose([0.0] * 16, openarm_pi05_manifest(), amr, measured)
    held = frozenset(names) - frozenset(slot_state_joint_names(openarm_pi05_manifest(), amr) or ())
    assert "lift_joint" in held
    runner_mod.validate_starting_pose(amr, full, held_joints=held)
    with pytest.raises(ROSConfigError, match="lift_joint"):
        runner_mod.validate_starting_pose(amr, full)  # named, it would be out of range
    named = list(full)
    named[names.index("left_joint1")] = 99.0
    with pytest.raises(ROSConfigError, match="left_joint1"):
        runner_mod.validate_starting_pose(amr, named, held_joints=held)
