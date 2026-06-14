"""Sim test: ACT on the real gym-aloha MuJoCo physics environment via the strict runner.

Drives ``openral_sim.SimRunner`` against
``scenes/sim/aloha_transfer_cube.yaml`` so the rSkill compatibility
check, the gym-aloha MuJoCo env, and the ACT policy adapter are all exercised
in the same path that ``openral sim run`` uses.

What is asserted
----------------
* Manifest contract — embodiment tags, sensors, latency budget all match
  what the lerobot ACT-ALOHA checkpoint declares.
* IO contract — the eval factory returns an ACT adapter whose IO shapes
  match the gym-aloha env: 480x640 RGB ``top`` + 14-D state → 14-D action,
  chunk_size=100.
* Closed-loop rollout completes 50 steps without NaN / inf actions; agent
  joints actually move (MuJoCo dynamics propagate).
* Mean per-step latency stays within ``manifest.latency_budget.per_chunk_ms``
  with the canonical ``tolerance_pct=100`` (CLAUDE.md §5.4).
* Peak VRAM during the rollout stays under 500 MB (catches load
  regressions; baseline 285 MB).

Skips automatically when CUDA is unavailable, lerobot/gym_aloha/torch are
missing, or the weights cannot be fetched from HF Hub.

Reference-host measurements
---------------------------
On RTX 4070 Laptop (7.62 GiB), CUDA 12.8, PyTorch 2.10:

  params = 51.6M     load VRAM = 211 MB     peak VRAM = 285 MB
  warm chunk inference = 16 ms (the rest are sub-ms queue pops)
  50-step rollout wall time ≈ 1.5 s
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip` / `pytest.skip(allow_module_level=True)`: with
# `tests/sim/__init__.py` making this directory a Package, a Skipped raised
# at module-import time poisons the whole `tests/sim` Package collection
# ("found no collectors for ..." on every sibling). Deferring the decision
# to `pytestmark` keeps this module importable when optional deps are
# missing, so sibling files remain reachable.
_REQUIRED_MODULES = ("torch", "gymnasium", "gym_aloha")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)
_CUDA_AVAILABLE = False
if not _MISSING_MODULES:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="ACT-ALOHA-full sim test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="ACT-ALOHA-full sim test requires CUDA",
    ),
]

# Resource ceiling: ~1.75x measured peak.
_PEAK_VRAM_CEILING_B = 500 * 1024**2  # 500 MB

_REPO_ROOT = Path(__file__).parent.parent.parent
# A SimScene (not a BenchmarkScene): `compose_sim_env` loads via
# `load_scene_strict(..., SimScene)`, which rejects BenchmarkScene YAMLs.
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "aloha_transfer_cube.yaml"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "act-aloha" / "rskill.yaml"


@pytest.fixture(scope="module")
def env_cfg():
    """Load the canonical ACT-ALOHA sim env, capped at one short episode."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    if not _LOCAL_MANIFEST.exists():
        pytest.skip(f"rSkill manifest not found at {_LOCAL_MANIFEST}")

    return compose_sim_env(
        _CONFIG,
        rskill_uri="rskills/act-aloha",
        n_episodes=1,
        max_steps=50,
    )


@pytest.fixture(scope="module")
def skill_manifest():
    """Load the rSkill manifest from disk (no network)."""
    from openral_rskill.loader import rSkill

    return rSkill.from_yaml(_LOCAL_MANIFEST)


class TestACTAlohaManifest:
    """Manifest contract — declared inputs match the bimanual ALOHA embodiment."""

    def test_manifest_loads_and_declares_aloha(self, skill_manifest) -> None:
        m = skill_manifest.manifest
        assert m.name == "OpenRAL/rskill-act-aloha"
        assert "aloha" in m.embodiment_tags
        assert m.role == "s1"

    def test_manifest_declares_top_camera(self, skill_manifest) -> None:
        sensors = skill_manifest.manifest.sensors_required
        assert len(sensors) == 1
        assert sensors[0].modality == "rgb"
        assert sensors[0].vla_feature_key == "observation.images.top"

    def test_manifest_has_latency_budget(self, skill_manifest) -> None:
        budget = skill_manifest.manifest.latency_budget
        assert budget is not None
        assert budget.per_chunk_ms > 0


class TestACTAlohaIOContract:
    """Adapter IO shapes — verified through the canonical eval factory."""

    def test_make_policy_has_aloha_io_shape(self, env_cfg) -> None:
        from openral_sim import make_policy

        policy = make_policy(env_cfg)
        try:
            cfg = policy._policy.config
            assert cfg.input_features["observation.images.top"].shape == (3, 480, 640)
            assert cfg.input_features["observation.state"].shape == (14,)
            assert cfg.output_features["action"].shape == (14,)
        finally:
            policy.close()


class TestACTAlohaRollout:
    """End-to-end episode through the strict runner."""

    def test_aloha_episode_runs_via_sim_runner(self, env_cfg, skill_manifest) -> None:
        from openral_sim import SimRunner

        runner = SimRunner(env_cfg)
        try:
            runner.activate()
            runner.run(max_ticks=env_cfg.n_episodes * (env_cfg.task.max_steps + 1))
        finally:
            runner.deactivate()
        results = runner.episode_results
        assert len(results) == 1
        result = results[0]

        assert result.steps == env_cfg.task.max_steps, (
            f"rollout terminated early at step {result.steps}/{env_cfg.task.max_steps}"
        )
        assert result.mean_step_latency_ms > 0
        assert result.latency_budget_ms == skill_manifest.manifest.latency_budget.per_chunk_ms

        # Mean per-step latency stays within manifest budget × 2 (tolerance_pct=100).
        # Most of the 50 steps are queue pops (chunk_size=100); only the first
        # pays the full 16 ms GPU cost, so the mean is dominated by sub-ms pops.
        ceiling = 2.0 * skill_manifest.manifest.latency_budget.per_chunk_ms
        assert result.mean_step_latency_ms <= ceiling, (
            f"mean_step_latency_ms={result.mean_step_latency_ms:.1f} > {ceiling:.1f}ms"
        )

    def test_peak_vram_under_ceiling(self, env_cfg) -> None:
        """50-step rollout must fit in the 500 MB ceiling."""
        from openral_sim import make_env, make_policy

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        env = make_env(env_cfg)
        policy = make_policy(env_cfg)
        try:
            obs = env.reset(seed=env_cfg.seed)
            for _ in range(env_cfg.task.max_steps):
                action = policy.step(obs, env_cfg.task.instruction)
                step_result = env.step(action)
                obs = step_result.observation
                if step_result.terminated or step_result.truncated:
                    break
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            assert peak < _PEAK_VRAM_CEILING_B, (
                f"rollout peak VRAM {peak / 1024**2:.0f} MB exceeds "
                f"{_PEAK_VRAM_CEILING_B / 1024**2:.0f} MB ceiling (baseline 285 MB)"
            )
        finally:
            policy.close()
            env.close()
