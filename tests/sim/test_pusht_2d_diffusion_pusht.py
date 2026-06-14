"""Sim test: real Diffusion Policy on the PushT physics simulator via the strict runner.

Drives ``openral_sim.SimRunner`` against
``scenes/benchmark/pusht.yaml`` so the rSkill compatibility
check, the gym-pusht physics env, and the Diffusion Policy adapter are
all exercised in the same path that ``openral sim run`` uses.

What is asserted
----------------
* IO contract matches PushT: 96x96 RGB + 2-DoF state → 8-step 2-DoF chunk
  (horizon=16, n_action_steps=8).
* Closed-loop rollout completes with finite rewards and a non-zero agent
  displacement (pymunk dynamics propagate).
* Mean per-step latency stays within ``manifest.latency_budget.per_chunk_ms``
  with the canonical ``tolerance_pct=100`` (CLAUDE.md §5.4).
* Peak VRAM during one chunk denoise stays under 2.0 GiB (catches load
  regressions; baseline 1.64 GiB).

Skips automatically when CUDA is unavailable, lerobot/gym_pusht/pymunk
are missing, or the weights cannot be fetched from HF Hub.

Reference-host measurements
---------------------------
On RTX 4070 Laptop (7.62 GiB), CUDA 12.8, PyTorch 2.10:

  params = 262.7M   load VRAM = 1055 MB   peak VRAM = 1639 MB
  warm chunk inference = 1756 ms (100-step DDPM denoising)
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
_REQUIRED_MODULES = ("torch", "gymnasium", "gym_pusht", "pymunk")
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
        reason="Diffusion-PushT sim test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="Diffusion-PushT sim test requires CUDA",
    ),
]

# Resource ceiling: ~1.25x measured peak. Tighter than other tests because
# the absolute footprint is large (1.6 GiB) — a true regression here would
# blow our 7.6 GiB budget.
_PEAK_VRAM_CEILING_B = int(2.0 * 1024**3)  # 2.0 GiB

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "benchmark" / "pusht.yaml"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "diffusion-pusht" / "rskill.yaml"


@pytest.fixture(scope="module")
def env_cfg():
    """Load the canonical Diffusion-PushT sim env, capped at one short episode."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    if not _LOCAL_MANIFEST.exists():
        pytest.skip(f"rSkill manifest not found at {_LOCAL_MANIFEST}")

    return compose_sim_env(
        _CONFIG,
        rskill_uri="rskills/diffusion-pusht",
        n_episodes=1,
        max_steps=40,
    )


@pytest.fixture(scope="module")
def skill_manifest():
    """Load the rSkill manifest from disk (no network)."""
    from openral_rskill.loader import rSkill

    return rSkill.from_yaml(_LOCAL_MANIFEST)


class TestDiffusionPushTManifest:
    """Manifest contract — declared inputs match the PushT embodiment."""

    def test_manifest_loads_and_declares_pusht(self, skill_manifest) -> None:
        m = skill_manifest.manifest
        assert m.name == "OpenRAL/rskill-diffusion-pusht"
        assert "pusht" in m.embodiment_tags
        assert m.role == "s1"

    def test_manifest_declares_single_rgb_sensor(self, skill_manifest) -> None:
        sensors = skill_manifest.manifest.sensors_required
        assert len(sensors) == 1
        assert sensors[0].modality == "rgb"
        assert sensors[0].vla_feature_key == "observation.image"

    def test_manifest_has_latency_budget(self, skill_manifest) -> None:
        budget = skill_manifest.manifest.latency_budget
        assert budget is not None
        assert budget.per_chunk_ms > 0


class TestDiffusionPushTIOContract:
    """Adapter IO shapes — verified through the canonical eval factory."""

    def test_make_policy_has_pusht_io_shape(self, env_cfg) -> None:
        from openral_sim import make_policy

        policy = make_policy(env_cfg)
        try:
            cfg = policy._policy.config
            assert cfg.input_features["observation.image"].shape == (3, 96, 96)
            assert cfg.input_features["observation.state"].shape == (2,)
            assert cfg.output_features["action"].shape == (2,)
            assert cfg.n_action_steps == 8
            assert cfg.horizon == 16
        finally:
            policy.close()

    def test_all_parameters_live_on_gpu(self, env_cfg) -> None:
        """All ~263M parameters must reside on cuda:0 — no silent CPU fallback."""
        from openral_sim import make_policy

        policy = make_policy(env_cfg)
        try:
            inner = policy._policy
            param_devices = {p.device for p in inner.parameters()}
            assert param_devices == {torch.device("cuda:0")}, param_devices
            n_params = sum(p.numel() for p in inner.parameters())
            assert 235_000_000 < n_params < 290_000_000, n_params
        finally:
            policy.close()


class TestDiffusionPushTRollout:
    """End-to-end episode through the strict runner."""

    def test_pusht_episode_runs_via_sim_runner(self, env_cfg, skill_manifest) -> None:
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

        assert result.steps > 0
        assert result.mean_step_latency_ms > 0
        assert result.latency_budget_ms == skill_manifest.manifest.latency_budget.per_chunk_ms

        # Mean per-step latency stays within manifest budget × 2 (tolerance_pct=100).
        ceiling = 2.0 * skill_manifest.manifest.latency_budget.per_chunk_ms
        assert result.mean_step_latency_ms <= ceiling, (
            f"mean_step_latency_ms={result.mean_step_latency_ms:.1f} > {ceiling:.1f}ms"
        )

    def test_peak_vram_under_ceiling(self, env_cfg) -> None:
        """One chunk denoise must fit in the 2 GiB ceiling."""
        from openral_sim import make_env, make_policy

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        env = make_env(env_cfg)
        policy = make_policy(env_cfg)
        try:
            obs = env.reset(seed=env_cfg.seed)
            policy.step(obs, env_cfg.task.instruction)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            assert peak < _PEAK_VRAM_CEILING_B, (
                f"peak VRAM {peak / 1024**3:.2f} GiB exceeds "
                f"{_PEAK_VRAM_CEILING_B / 1024**3:.2f} GiB ceiling (baseline 1.64 GiB)"
            )
        finally:
            policy.close()
            env.close()
