"""Deploy-side VLA↔reward resolution + VRAM preflight.

`openral deploy sim` resolves the reward model from the reasoner-picked palette VLA's
`reward_rskill_name` pairing, then runs a pre-launch VRAM feasibility check over the
candidate set before ROS comes up.

Fixture: real `robots/franka_panda` manifest (palette includes `rskills/smolvla-libero`,
the only in-tree VLA naming a reward model) + `rskills/robometer-4b`. Franka palette
min_vram_gb: smolvla 1.2, molmoact/pi05 4.0, rldx 7.0, 3dda 2.0; undeclared: act/gr00t/
xvla/smolvla-maniskill — one fixture exercises the fit/OOM/undeclared branches.

CLI-layer helpers, no ROS. Run:
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
from openral_core import POLICY_KINDS, RobotDescription

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_FRANKA = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
_ROBOMETER = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"


def _franka() -> RobotDescription:
    """The real ``robots/franka_panda`` manifest."""
    return RobotDescription.from_yaml(str(_FRANKA))


def test_palette_contains_smolvla_which_names_its_reward_model() -> None:
    """The franka palette is the runtime candidate set: it includes the VLA that
    declares `reward_rskill_name` (the pairing the deploy must honour)."""
    vlas = [m for m in _capability_matched_manifests(_REPO_ROOT, _franka()) if m.kind == "vla"]
    by_name = {m.name: m for m in vlas}
    skill_name = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"
    assert skill_name in by_name
    assert by_name[skill_name].reward_rskill_name == "OpenRAL/rskill-robometer_4b-any-general-nf4"


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


def test_preflight_skipped_when_gpu_budget_unreadable() -> None:
    """gpu_budget_gb <= 0 (no nvidia-smi) → defer to the reasoner's runtime check, no exit."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=0.0,
    )  # returns None / does not raise


def test_preflight_skipped_when_no_reward_active() -> None:
    """No reward manifest path → nothing to pair against, no exit."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path="",
        gpu_budget_gb=8.0,
    )


def test_preflight_passes_on_8gb_card() -> None:
    """On an 8 GB card at least one VLA (smolvla 1.2 + robometer 5.5 = 6.7) fits, so the
    preflight proceeds; the larger and undeclared VLAs come back as refused, the fitting
    one does not."""
    refused = _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=8.0,
    )
    assert "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16" not in refused
    assert "OpenRAL/rskill-rldx1_ft-franka_panda-libero_spatial-nf4" in refused  # 7.0 GB
    assert "OpenRAL/rskill-act-franka_panda-libero-fp32" in refused  # undeclared


def test_preflight_hard_exits_when_no_vla_fits(tmp_path: pathlib.Path) -> None:
    """On a 4 GB card the franka smolvla cannot co-reside with robometer (1.2 + 5.5 =
    6.7 > 4.0); with no wrapped-ROS skill beside it the deploy could dispatch nothing →
    fail fast BEFORE ROS."""
    smolvla = _REPO_ROOT / "rskills" / "smolvla-libero" / "rskill.yaml"
    dst = tmp_path / "rskills" / "smolvla-libero" / "rskill.yaml"
    dst.parent.mkdir(parents=True)
    dst.write_text(smolvla.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(typer.Exit) as exc:
        _preflight_reward_vram_fit(
            repo_root=tmp_path,
            description=_franka(),
            reward_manifest_path=str(_ROBOMETER),
            gpu_budget_gb=4.0,
        )
    assert exc.value.exit_code == 1


# ── Wrapped-ROS skills keep the gate open (#475) ────────────────────────────────────
# `panda_mobile`'s palette: rldx1 + internvla (7.0 GB each) and two undeclared xr1 VLAs,
# plus the Nav2 `ros_action` skill. On an 8 GB card no VLA fits beside robometer
# (7.0 + 5.5 > 8.0).
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_RLDX_RC365 = _REPO_ROOT / "rskills" / "rldx1-ft-rc365-nf4" / "rskill.yaml"
_NAV2 = _REPO_ROOT / "rskills" / "rskill-nav2-navigate-to-pose" / "rskill.yaml"
_MOVEIT_JOINTS = _REPO_ROOT / "rskills" / "rskill-moveit-joints" / "rskill.yaml"


def _panda_mobile() -> RobotDescription:
    """The real ``robots/panda_mobile`` manifest (mobile base + lidar, so Nav2 matches)."""
    return RobotDescription.from_yaml(str(_PANDA_MOBILE))


def test_preflight_passes_when_nav2_wrapper_runnable(capsys: pytest.CaptureFixture[str]) -> None:
    """No VLA fits, but the Nav2 `ros_action` skill is runnable: launch proceeds, every
    policy — too large or undeclared — is reported and returned as refused, and the
    message names the server the wrapper needs."""
    refused = _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_panda_mobile(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=8.0,
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "cannot fit beside the reward model" in out
    assert "do not declare min_vram_gb" in out
    assert "rskill-nav2-mobile_base-navigate_to_pose → ros_action /navigate_to_pose" in out
    assert "preflight failed" not in out
    policies = {
        m.name
        for m in _capability_matched_manifests(_REPO_ROOT, _panda_mobile())
        if m.kind in POLICY_KINDS
    }
    assert set(refused) == policies


def test_preflight_passes_when_moveit_wrappers_runnable() -> None:
    """franka on a 4 GB card: no VLA fits, but its three MoveIt `ros_action` skills need no
    VRAM, so launch proceeds (move_group is the reasoner's graph check, at on_configure)."""
    _preflight_reward_vram_fit(
        repo_root=_REPO_ROOT,
        description=_franka(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=4.0,
    )


def _copy_with_kind(src: pathlib.Path, dst_root: pathlib.Path, kind: str) -> None:
    """Copy a real in-tree manifest into ``dst_root/rskills/`` with only ``kind`` changed.

    The tree ships no WAM or ``ros_service`` manifest; swapping the discriminator on a
    real VLA / Nav2 manifest keeps every other field a real fixture.
    """
    text = src.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("kind:"))
    dst = dst_root / "rskills" / src.parent.name / "rskill.yaml"
    dst.parent.mkdir(parents=True)
    dst.write_text(text.replace(old, f'kind: "{kind}"', 1), encoding="utf-8")


@pytest.mark.parametrize("policy_kind", ["vla", "wam"])
@pytest.mark.parametrize("wrapper_kind", ["ros_action", "ros_service"])
def test_preflight_policy_and_wrapper_kind_matrix(
    tmp_path: pathlib.Path, policy_kind: str, wrapper_kind: str
) -> None:
    """A non-fitting VLA or WAM beside a runnable `ros_action` or `ros_service` skill
    does not fail the gate."""
    _copy_with_kind(_RLDX_RC365, tmp_path, policy_kind)
    _copy_with_kind(_NAV2, tmp_path, wrapper_kind)
    kinds = {m.kind for m in _capability_matched_manifests(tmp_path, _panda_mobile())}
    assert kinds == {policy_kind, wrapper_kind}
    _preflight_reward_vram_fit(
        repo_root=tmp_path,
        description=_panda_mobile(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=8.0,
    )


@pytest.mark.parametrize("policy_kind", ["vla", "wam"])
@pytest.mark.parametrize("wrapper_kind", ["ros_action", "ros_service"])
def test_preflight_ineligible_wrapper_does_not_bypass_gate(
    tmp_path: pathlib.Path, policy_kind: str, wrapper_kind: str
) -> None:
    """A MoveIt wrapper targets arm embodiments, not `panda_mobile`, so the reasoner palette
    rejects it — it must not keep the gate open for a non-fitting policy."""
    _copy_with_kind(_RLDX_RC365, tmp_path, policy_kind)
    _copy_with_kind(_MOVEIT_JOINTS, tmp_path, wrapper_kind)
    with pytest.raises(typer.Exit) as exc:
        _preflight_reward_vram_fit(
            repo_root=tmp_path,
            description=_panda_mobile(),
            reward_manifest_path=str(_ROBOMETER),
            gpu_budget_gb=8.0,
        )
    assert exc.value.exit_code == 1


def test_preflight_policy_only_palette_without_fit_fails(tmp_path: pathlib.Path) -> None:
    """Policies only, none fits → hard fail."""
    _copy_with_kind(_RLDX_RC365, tmp_path, "wam")
    with pytest.raises(typer.Exit):
        _preflight_reward_vram_fit(
            repo_root=tmp_path,
            description=_panda_mobile(),
            reward_manifest_path=str(_ROBOMETER),
            gpu_budget_gb=8.0,
        )


def test_preflight_no_policy_in_palette_skips(tmp_path: pathlib.Path) -> None:
    """A wrapper-only palette has no policy to pair, so the gate stays a no-op."""
    _copy_with_kind(_NAV2, tmp_path, "ros_action")
    refused = _preflight_reward_vram_fit(
        repo_root=tmp_path,
        description=_panda_mobile(),
        reward_manifest_path=str(_ROBOMETER),
        gpu_budget_gb=8.0,
    )
    assert refused == []
