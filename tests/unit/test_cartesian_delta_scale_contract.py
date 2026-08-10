"""Cartesian-delta controller scaling is explicit and replayable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_core import Action, ActionContract, ControlMode, RSkillManifest
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_ROBOSUITE_OSC_SCALE_VALUES = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
_ROBOSUITE_OSC_SCALE = pytest.approx(_ROBOSUITE_OSC_SCALE_VALUES)
_NORMALIZED_OSC_SKILLS = (
    "act-libero",
    "gr00t-n17-libero",
    "molmoact2-libero-nf4",
    "pi05-libero-int8",
    "rldx1-ft-libero-nf4",
    "rldx1-ft-rc365-nf4",
    "smolvla-libero",
    "xr1-robocasa",
    "xr1-robocasa365",
    "xvla-libero",
)
_PHYSICAL_DELTA_SKILLS = (
    "openvla-oft-simpler-widowx-nf4",
    "rldx1-ft-simpler-widowx-nf4",
)


@pytest.mark.parametrize("skill", _NORMALIZED_OSC_SKILLS)
def test_normalized_robosuite_skills_declare_physical_safety_scale(skill: str) -> None:
    manifest = RSkillManifest.from_yaml(str(_REPO / "rskills" / skill / "rskill.yaml"))
    assert manifest.action_contract is not None
    assert manifest.action_contract.cartesian_delta_scale == _ROBOSUITE_OSC_SCALE


@pytest.mark.parametrize("skill", _PHYSICAL_DELTA_SKILLS)
def test_physical_delta_skills_keep_identity_safety_scale(skill: str) -> None:
    """ManiSkill WidowX has normalize_action=False; its deltas are already physical."""
    manifest = RSkillManifest.from_yaml(str(_REPO / "rskills" / skill / "rskill.yaml"))
    assert manifest.action_contract is not None
    assert manifest.action_contract.cartesian_delta_scale is None


def test_installed_robosuite_osc_config_matches_manifest_scale() -> None:
    robosuite = pytest.importorskip("robosuite")
    root = Path(robosuite.__file__).resolve().parent
    composite = root / "controllers" / "config" / "default" / "composite" / "basic.json"
    part = root / "controllers" / "config" / "default" / "parts" / "osc_pose.json"
    if composite.is_file():
        config = json.loads(composite.read_text(encoding="utf-8"))["body_parts"]["arms"]["right"]
    elif part.is_file():
        config = json.loads(part.read_text(encoding="utf-8"))
    else:
        pytest.skip("installed RoboSuite has no known OSC_POSE config path")
    input_min = config["input_min"]
    input_max = config["input_max"]
    assert (
        all(value == -1 for value in input_min) if isinstance(input_min, list) else input_min == -1
    )
    assert all(value == 1 for value in input_max) if isinstance(input_max, list) else input_max == 1
    assert tuple(config["output_min"]) == pytest.approx(
        tuple(-value for value in _ROBOSUITE_OSC_SCALE_VALUES)
    )
    assert tuple(config["output_max"]) == pytest.approx(_ROBOSUITE_OSC_SCALE_VALUES)


def test_action_preserves_raw_delta_and_carries_scale_separately() -> None:
    raw = (-0.176, 0.095, -0.309, 0.0, 0.013, 0.104)
    scale = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    action = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        cartesian_delta=[raw],
        cartesian_delta_scale=scale,
    )
    assert action.cartesian_delta == [raw]
    assert action.cartesian_delta_scale == scale


def test_scale_rejects_wrong_width_nonpositive_and_non_cartesian_use() -> None:
    with pytest.raises(ValidationError, match="length must match"):
        Action(
            control_mode=ControlMode.CARTESIAN_DELTA,
            cartesian_delta=[(0.0,) * 6],
            cartesian_delta_scale=(0.05,) * 3,
        )
    with pytest.raises(ValidationError, match="finite and > 0"):
        ActionContract(
            dim=7,
            representation="delta_ee_6d_plus_gripper",
            cartesian_delta_scale=(0.05, 0.05, 0.0, 0.5, 0.5, 0.5),
        )
    with pytest.raises(ValidationError, match="requires a CARTESIAN_DELTA"):
        ActionContract(dim=7, representation="joint_positions", cartesian_delta_scale=(1.0,) * 7)
