"""Tests for the VLA↔reward pairing + GPU VRAM fit check (ADR-0077).

A VLA emits no success signal of its own, so it runs with a reward model resident
alongside it (ADR-0074). ADR-0077 records that pairing on the VLA manifest
(`reward_rskill_name`) and refuses to launch a pair that does not fit GPU VRAM.

Fixture-backed (CLAUDE.md §1.11): real `rskills/smolvla-libero` (the VLA) +
`rskills/robometer-4b` (the reward model).

Run with:
    uv run pytest tests/unit/test_vla_reward_pairing.py -v
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import RSkillManifest, assert_vla_reward_fits
from openral_core.exceptions import ROSConfigError, ROSGPUMemoryError
from openral_core.schemas import QuantizationDtype
from pydantic import ValidationError

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_SMOLVLA = _REPO_ROOT / "rskills" / "smolvla-libero" / "rskill.yaml"
_ROBOMETER = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"


def _vla() -> RSkillManifest:
    return RSkillManifest.from_yaml(_SMOLVLA)


def _reward() -> RSkillManifest:
    return RSkillManifest.from_yaml(_ROBOMETER)


def test_smolvla_manifest_declares_its_reward_pairing_and_size() -> None:
    """ADR-0077 — the VLA manifest names its reward model and declares bf16 VRAM."""
    vla = _vla()
    assert vla.kind == "vla"
    assert vla.reward_rskill_name == "OpenRAL/rskill-robometer-4b-nf4"
    assert vla.active_min_vram_gb() == pytest.approx(1.2)


def test_robometer_active_vram_is_its_nf4_int4_footprint() -> None:
    """The reward model's active dtype is int4 (nf4) → 3.6 GB, not the bf16 9.0."""
    reward = _reward()
    assert reward.kind == "reward"
    assert reward.quantization.dtype == QuantizationDtype.INT4
    assert reward.active_min_vram_gb() == pytest.approx(3.6)


def test_pair_fits_8gb_card() -> None:
    """smolvla (1.2) + robometer (3.6) = 4.8 GB fits a 7.62 GB usable 8 GB card."""
    combined = assert_vla_reward_fits(_vla(), _reward(), gpu_total_gb=7.62)
    assert combined == pytest.approx(4.8)


def test_pair_does_not_fit_4gb_card_raises_gpu_memory_error() -> None:
    """The pair (4.8 GB + margin) exceeds a 4 GB card → fail fast, do not run blind."""
    with pytest.raises(ROSGPUMemoryError) as exc:
        assert_vla_reward_fits(_vla(), _reward(), gpu_total_gb=4.0)
    msg = str(exc.value)
    assert "robometer" in msg.lower() and "smolvla" in msg.lower()
    assert "4.80 GB" in msg  # the combined footprint is reported


def test_margin_is_enforced() -> None:
    """4.8 GB just under a 5.0 GB card still fails once the default 0.5 GB margin
    is added (4.8 + 0.5 = 5.3 > 5.0) — the headroom is real, not cosmetic."""
    with pytest.raises(ROSGPUMemoryError):
        assert_vla_reward_fits(_vla(), _reward(), gpu_total_gb=5.0)
    # With a generous card and no margin it passes.
    assert assert_vla_reward_fits(
        _vla(), _reward(), gpu_total_gb=5.0, margin_gb=0.0
    ) == pytest.approx(4.8)


def test_undeclared_vram_cannot_be_verified_raises_config_error() -> None:
    """A VLA without min_vram_gb can't have its co-residency verified → ROSConfigError
    (the operator must declare the size we are about to require fits)."""
    vla_no_size = _vla().model_copy(update={"min_vram_gb": None})
    with pytest.raises(ROSConfigError) as exc:
        assert_vla_reward_fits(vla_no_size, _reward(), gpu_total_gb=7.62)
    assert "min_vram_gb" in str(exc.value)


def test_reward_rskill_name_forbidden_on_non_vla_kind() -> None:
    """`reward_rskill_name` is a reference FROM a VLA; a reward-kind manifest that
    sets it is rejected (ADR-0077 validator guard)."""
    reward_dict = _reward().model_dump(mode="json")
    reward_dict["reward_rskill_name"] = "OpenRAL/rskill-something"
    with pytest.raises(ValidationError) as exc:
        RSkillManifest.model_validate(reward_dict)
    assert "reward_rskill_name" in str(exc.value)
