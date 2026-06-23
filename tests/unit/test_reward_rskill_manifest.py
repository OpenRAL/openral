"""Tests for the ``kind: "reward"`` rSkill manifest variant (ADR-0057).

Covers:
- :class:`~openral_core.schemas.RewardContract` Hypothesis round-trip +
  JSON-Schema validation.
- Contract test: load ``rskills/robometer-4b/rskill.yaml`` via
  :meth:`~openral_core.schemas.RSkillManifest.model_validate` and assert
  the key guarantees.
- Validator boundary tests: each required/forbidden field rule raises
  :exc:`pydantic.ValidationError` exactly.

Run with:
    uv run pytest tests/unit/test_reward_rskill_manifest.py -v
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core.schemas import (
    RewardContract,
    RSkillAction,
    RSkillManifest,
)
from pydantic import ValidationError

# ─── Fixture path ──────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_ROBOMETER_FIXTURE = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"

# ─── Fuzz strategy helpers ────────────────────────────────────────────────────

_FUZZ_SETTINGS = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_prob = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0)
_pos = st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=120.0)
_pos_int = st.integers(min_value=1, max_value=1000)


@st.composite
def _ordered_range(draw: st.DrawFn) -> tuple[float, float]:
    lo = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0))
    span = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=10.0))
    return (lo, lo + span)


_reward_contract_st = st.builds(
    RewardContract,
    progress_range=_ordered_range(),
    success_threshold=_prob,
    preference=st.booleans(),
    frame_window_s=_pos,
    target_fps=_pos,
    num_bins=_pos_int,
    instruction_required=st.booleans(),
)


# ─── RewardContract fuzz ──────────────────────────────────────────────────────


@_FUZZ_SETTINGS
@given(_reward_contract_st)
def test_fuzz_reward_contract(instance: RewardContract) -> None:
    """RewardContract round-trips through JSON and validates against its schema."""
    import json

    import jsonschema

    serialized = instance.model_dump_json()
    reloaded = RewardContract.model_validate_json(serialized)
    assert reloaded == instance, "Round-trip failed for RewardContract"

    data = json.loads(serialized)
    schema = RewardContract.model_json_schema()
    jsonschema.validate(data, schema)


# ─── Contract test: real fixture ──────────────────────────────────────────────


def test_robometer_fixture_validates() -> None:
    """Load rskills/robometer-4b/rskill.yaml and assert key invariants."""
    assert _ROBOMETER_FIXTURE.exists(), f"Fixture not found: {_ROBOMETER_FIXTURE}"

    with open(_ROBOMETER_FIXTURE, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    manifest = RSkillManifest.model_validate(data)

    assert manifest.kind == "reward"
    assert manifest.reward is not None
    assert manifest.reward.progress_range == (0.0, 1.0)
    assert manifest.reward.frame_window_s > 0.0
    assert manifest.reward.target_fps > 0.0
    assert manifest.weights_uri is not None, "reward requires weights_uri"
    assert manifest.actuators_required == [], "reward must have no actuators"
    assert manifest.embodiment_tags == ["any"], "reward is embodiment-agnostic (wildcard)"
    assert RSkillAction.MONITOR in manifest.actions
    # A reward monitor has no VLA identity / perception-producer fields
    assert manifest.model_family is None
    assert manifest.action_contract is None
    assert manifest.state_contract is None
    assert manifest.ros_integration is None
    assert manifest.detector is None


# ─── Validator boundary tests ─────────────────────────────────────────────────

_VALID_REWARD: dict = {
    "schema_version": "0.1",
    "name": "openral/test-reward",
    "version": "0.1.0",
    "license": "apache-2.0",
    "role": "s2",
    "kind": "reward",
    "embodiment_tags": ["any"],
    "sensors_required": [{"modality": "rgb", "min_width": 224, "min_height": 224}],
    "actuators_required": [],
    "runtime": "pytorch",
    "weights_uri": "hf://OpenRAL/rskill-robometer-4b-nf4",
    "chunk_size": 1,
    "latency_budget": {"per_chunk_ms": 3000.0},
    "description": "Test reward monitor for unit tests.",
    "actions": ["monitor"],
    "reward": {
        "progress_range": [0.0, 1.0],
        "success_threshold": 0.5,
        "frame_window_s": 8.0,
        "target_fps": 3.0,
        "num_bins": 100,
    },
}

_JOINT_POS_ACTUATOR = {
    "kind": "joint_position",
    "control_mode_semantics": {"mode": "absolute"},
}


def test_reward_valid_baseline() -> None:
    """The baseline reward dict validates without errors."""
    m = RSkillManifest.model_validate(copy.deepcopy(_VALID_REWARD))
    assert m.kind == "reward"
    assert m.reward is not None
    assert m.reward.target_fps == 3.0


def test_reward_missing_reward_block_raises() -> None:
    """A ``kind: "reward"`` manifest without a ``reward`` block is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    del bad["reward"]
    with pytest.raises(ValidationError, match="requires a `reward` block"):
        RSkillManifest.model_validate(bad)


def test_reward_missing_weights_uri_raises() -> None:
    """A ``kind: "reward"`` manifest without ``weights_uri`` is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    del bad["weights_uri"]
    with pytest.raises(ValidationError, match=r"weights_uri"):
        RSkillManifest.model_validate(bad)


def test_reward_with_actuators_raises() -> None:
    """A ``kind: "reward"`` manifest that lists actuators is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    bad["actuators_required"] = [_JOINT_POS_ACTUATOR]
    with pytest.raises(ValidationError, match=r"actuates nothing"):
        RSkillManifest.model_validate(bad)


def test_reward_with_model_family_raises() -> None:
    """A ``kind: "reward"`` manifest with ``model_family`` set is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    bad["model_family"] = "smolvla"
    with pytest.raises(ValidationError, match="model_family"):
        RSkillManifest.model_validate(bad)


def test_reward_with_detector_block_raises() -> None:
    """A ``kind: "reward"`` manifest with a ``detector`` block is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    bad["detector"] = {"labels": ["x"], "input_size": [640, 640], "score_threshold": 0.5}
    with pytest.raises(ValidationError, match="detector"):
        RSkillManifest.model_validate(bad)


def test_reward_with_action_contract_raises() -> None:
    """A ``kind: "reward"`` manifest with ``action_contract`` set is rejected."""
    bad = copy.deepcopy(_VALID_REWARD)
    bad["action_contract"] = {"dim": 7}
    with pytest.raises(ValidationError, match="action_contract"):
        RSkillManifest.model_validate(bad)


_REWARD_BLOCK = {
    "progress_range": [0.0, 1.0],
    "success_threshold": 0.5,
    "frame_window_s": 8.0,
    "target_fps": 3.0,
}


def test_non_reward_with_reward_block_raises_vla() -> None:
    """A ``kind: "vla"`` manifest with a ``reward`` block is rejected."""
    bad = {
        "schema_version": "0.1",
        "name": "openral/test-vla",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "vla",
        "model_family": "smolvla",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": [_JOINT_POS_ACTUATOR],
        "runtime": "pytorch",
        "weights_uri": "hf://openral/test",
        "chunk_size": 16,
        "latency_budget": {"per_chunk_ms": 100.0},
        "description": "Test VLA for unit tests.",
        "actions": ["pick"],
        "processors": {
            "preprocessor_uri": "hf://openral/test/policy_preprocessor.json",
            "postprocessor_uri": "hf://openral/test/policy_postprocessor.json",
        },
        "reward": _REWARD_BLOCK,
    }
    with pytest.raises(ValidationError, match="reward"):
        RSkillManifest.model_validate(bad)


# ─── RewardContract edge-case tests ───────────────────────────────────────────


def test_reward_contract_progress_range_must_be_ordered() -> None:
    """RewardContract progress_range with max <= min raises ValidationError."""
    with pytest.raises(ValidationError, match="max > min"):
        RewardContract(progress_range=(1.0, 1.0), frame_window_s=8.0, target_fps=3.0)
    with pytest.raises(ValidationError, match="max > min"):
        RewardContract(progress_range=(1.0, 0.0), frame_window_s=8.0, target_fps=3.0)


def test_reward_contract_window_and_fps_must_be_positive() -> None:
    """RewardContract frame_window_s and target_fps must be > 0."""
    with pytest.raises(ValidationError):
        RewardContract(frame_window_s=0.0, target_fps=3.0)
    with pytest.raises(ValidationError):
        RewardContract(frame_window_s=8.0, target_fps=0.0)


def test_reward_contract_success_threshold_bounds() -> None:
    """RewardContract success_threshold must be in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        RewardContract(frame_window_s=8.0, target_fps=3.0, success_threshold=-0.1)
    with pytest.raises(ValidationError):
        RewardContract(frame_window_s=8.0, target_fps=3.0, success_threshold=1.1)
    ok = RewardContract(frame_window_s=8.0, target_fps=3.0, success_threshold=1.0)
    assert ok.success_threshold == 1.0


def test_reward_contract_num_bins_must_be_positive() -> None:
    """RewardContract num_bins must be > 0."""
    with pytest.raises(ValidationError):
        RewardContract(frame_window_s=8.0, target_fps=3.0, num_bins=0)
