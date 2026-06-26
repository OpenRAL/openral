"""Unit tests for RewardContract progress calibration fields + ExecuteRskillTool overrides.

ADR-0074 extends :class:`RewardContract` with four calibration fields
(``check_floor``, ``plateau_window_s``, ``plateau_tolerance``,
``default_patience_s``) and :class:`ExecuteRskillTool` with two optional
per-dispatch overrides (``patience_s``, ``progress_tolerance``).

Validates against the REAL robometer-4b fixture (no mocks).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import ExecuteRskillTool, RewardContract, RSkillManifest
from pydantic import ValidationError

# ── repo root (same pattern used by test_all_manifests_validate.py) ───────────

_HERE = Path(__file__).resolve()
_REPO_ROOT: Path | None = None
for _p in _HERE.parents:
    if (_p / "robots").is_dir() and (_p / "rskills").is_dir():
        _REPO_ROOT = _p
        break


def _require_root() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("No repo root found (wheel install?)")
    return _REPO_ROOT  # type: ignore[return-value]


# ── RewardContract default values ─────────────────────────────────────────────


def test_reward_contract_defaults_for_calibration_fields() -> None:
    """Four calibration fields carry documented defaults when not supplied."""
    c = RewardContract(frame_window_s=8.0, target_fps=3.0)
    assert c.check_floor == 0.4
    assert c.plateau_window_s == 3.0
    assert c.plateau_tolerance == 0.05
    assert c.default_patience_s == 30.0


def test_reward_contract_existing_defaults_unchanged() -> None:
    """Existing fields keep their defaults after the extension."""
    c = RewardContract(frame_window_s=8.0, target_fps=3.0)
    assert c.progress_range == (0.0, 1.0)
    assert c.success_threshold == 0.5
    assert c.preference is False
    assert c.num_bins == 100
    assert c.instruction_required is True


# ── check_floor ≤ success_threshold cross-validator ───────────────────────────


def test_reward_contract_validator_rejects_floor_above_threshold() -> None:
    """check_floor > success_threshold must raise ValidationError."""
    with pytest.raises(ValidationError):
        RewardContract(
            frame_window_s=8,
            target_fps=3,
            check_floor=0.9,
            success_threshold=0.5,
        )


def test_reward_contract_validator_accepts_floor_equal_threshold() -> None:
    """check_floor == success_threshold is valid (boundary case)."""
    c = RewardContract(
        frame_window_s=8.0,
        target_fps=3.0,
        check_floor=0.5,
        success_threshold=0.5,
    )
    assert c.check_floor == c.success_threshold


def test_reward_contract_validator_accepts_floor_below_threshold() -> None:
    """Normal case: check_floor < success_threshold must not raise."""
    c = RewardContract(
        frame_window_s=8.0,
        target_fps=3.0,
        check_floor=0.3,
        success_threshold=0.6,
    )
    assert c.check_floor == 0.3


# ── Real robometer-4b manifest fixture ────────────────────────────────────────


def test_robometer_manifest_carries_calibrated_reward_contract() -> None:
    """The real robometer-4b rskill.yaml loads and its reward contract
    carries the model-calibrated values added in ADR-0074."""
    root = _require_root()
    manifest_path = root / "rskills" / "robometer-4b" / "rskill.yaml"
    manifest = RSkillManifest.from_yaml(str(manifest_path))

    assert manifest.reward is not None
    rc = manifest.reward
    assert rc.check_floor == 0.4
    assert rc.plateau_window_s == 3.0
    assert rc.plateau_tolerance == 0.06
    assert rc.default_patience_s == 30.0
    # Existing fields unchanged
    assert rc.success_threshold == 0.5
    assert rc.frame_window_s == 8.0
    assert rc.target_fps == 3.0


# ── ExecuteRskillTool optional overrides ─────────────────────────────────────


def test_execute_rskill_tool_accepts_patience_and_tolerance_overrides() -> None:
    """patience_s and progress_tolerance are accepted when supplied."""
    tool = ExecuteRskillTool(
        rskill_id="OpenRAL/some-skill",
        patience_s=12.0,
        progress_tolerance=0.1,
    )
    assert tool.patience_s == 12.0
    assert tool.progress_tolerance == 0.1


def test_execute_rskill_tool_defaults_overrides_to_none() -> None:
    """patience_s and progress_tolerance default to None (use model calibration)."""
    tool = ExecuteRskillTool(rskill_id="OpenRAL/some-skill")
    assert tool.patience_s is None
    assert tool.progress_tolerance is None


def test_execute_rskill_tool_existing_fields_intact() -> None:
    """Adding the new fields does not disturb existing ExecuteRskillTool fields."""
    tool = ExecuteRskillTool(
        rskill_id="OpenRAL/smolvla-libero",
        prompt="pick the bread",
        deadline_s=10.0,
    )
    assert tool.rskill_id == "OpenRAL/smolvla-libero"
    assert tool.prompt == "pick the bread"
    assert tool.deadline_s == 10.0
    assert tool.goal_params_json == ""
    assert tool.tool == "execute_rskill"
