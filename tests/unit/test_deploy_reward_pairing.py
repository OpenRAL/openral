"""Deploy-side VLA↔reward resolution + VRAM preflight (ADR-0077 §4).

`openral deploy sim` does not preselect a VLA — the reasoner picks one at runtime
from the capability-matched palette. So the deploy resolves the reward model from
the *palette's* `reward_rskill_name` pairing and runs a pre-LAUNCH VRAM feasibility
check over the whole candidate set before bringing up ROS.

Fixture-backed (CLAUDE.md §1.11): the real `robots/franka_panda` manifest (whose
palette includes `rskills/smolvla-libero`, the only in-tree VLA that names a reward
model) + `rskills/robometer-4b` (the reward model). The franka palette mixes VLAs
that declare `min_vram_gb` (smolvla 1.2, molmoact/pi05 4.0, rldx 7.0, 3dda 2.0) with
ones that do not (act / gr00t / xvla / smolvla-maniskill), so a single robot fixture
exercises the fit / OOM / undeclared branches.

These are CLI-layer helpers — no ROS. Run with the main venv:
    PYTHONPATH=python/cli/src:python/core/src:python/reasoner/src:python/sim/src \
        .venv/bin/python -m pytest tests/unit/test_deploy_reward_pairing.py -v
"""

from __future__ import annotations

import pathlib

import pytest
import typer
from openral_cli.deploy_sim import (
    _DEFAULT_REWARD_RSKILL_DIR,
    _capability_matched_manifests,
    _preflight_reward_vram_fit,
    _resolve_reward_monitor_manifest,
)
from openral_core import RobotDescription

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_FRANKA = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
_ROBOMETER = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"


def _franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_FRANKA))


def test_palette_contains_smolvla_which_names_its_reward_model() -> None:
    """The franka palette is the runtime candidate set: it includes the VLA that
    declares `reward_rskill_name` (the pairing the deploy must honour)."""
    vlas = [m for m in _capability_matched_manifests(_REPO_ROOT, _franka()) if m.kind == "vla"]
    by_name = {m.name: m for m in vlas}
    assert "OpenRAL/rskill-smolvla-libero" in by_name
    assert by_name["OpenRAL/rskill-smolvla-libero"].reward_rskill_name == (
        "OpenRAL/rskill-robometer-4b-nf4"
    )


def test_explicit_reward_manifest_wins() -> None:
    """An operator `--reward-monitor-manifest` is returned verbatim (override)."""
    resolved = _resolve_reward_monitor_manifest(
        repo_root=_REPO_ROOT,
        description=_franka(),
        explicit_manifest="/some/operator/reward.yaml",
    )
    assert resolved == "/some/operator/reward.yaml"


def test_default_resolves_from_vla_pairing() -> None:
    """With no explicit override, the reward model is derived from the palette VLAs'
    `reward_rskill_name` — here smolvla → robometer-4b's in-tree manifest path."""
    resolved = _resolve_reward_monitor_manifest(
        repo_root=_REPO_ROOT,
        description=_franka(),
        explicit_manifest=None,
    )
    assert resolved == str(_ROBOMETER.resolve())
    assert _DEFAULT_REWARD_RSKILL_DIR in resolved


def test_preflight_skipped_when_gpu_total_unreadable() -> None:
    """gpu_total_gb <= 0 (no nvidia-smi) → defer to the reasoner's runtime check, no exit."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_total_gb=0.0,
    )  # returns None / does not raise


def test_preflight_skipped_when_no_reward_active() -> None:
    """No reward manifest path → nothing to pair against, no exit."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path="",
        gpu_total_gb=8.0,
    )


def test_preflight_passes_on_8gb_card() -> None:
    """On an 8 GB card at least one VLA (smolvla 1.2 + robometer 3.6 = 4.8) fits, so the
    preflight proceeds (some larger VLAs are warned about, but the deploy is runnable)."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_total_gb=8.0,
    )  # does not raise


def test_preflight_hard_exits_when_no_vla_fits() -> None:
    """On a 4 GB card no franka VLA can co-reside with robometer (smolvla 1.2 + 3.6 =
    4.8 > 4.0), so the deploy could dispatch nothing → fail fast BEFORE ROS."""
    with pytest.raises(typer.Exit) as exc:
        _preflight_reward_vram_fit(
            repo_root=_REPO_ROOT,
            description=_franka(),
            reward_manifest_path=str(_ROBOMETER),
            gpu_total_gb=4.0,
        )
    assert exc.value.exit_code == 1
