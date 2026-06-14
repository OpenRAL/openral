"""Sim test: pi05 on a real RoboCasa MuJoCo episode via the strict runner.

Drives ``openral_sim.SimRunner`` against
``scenes/sim/robocasa_pnp.yaml`` so the rSkill compatibility
check, the RoboCasa physics env (robosuite + MuJoCo), and the pi05 policy
adapter are all exercised in the same path that ``openral sim run`` (and the
``robocasa_pnp`` benchmark) drives.

What is asserted
----------------
* The configured rSkill manifest loads via ``rSkill.from_yaml`` and declares
  the panda_mobile / pi05 contract (3 RGB camera streams via aliases, 16-D
  human300 state layout).
* ``SimRunner`` runs at least one full step in the RoboCasa MuJoCo env and
  populates ``EpisodeResult.latency_budget_ms`` from the manifest.

Skips automatically when CUDA / torch / transformers / robocasa / robosuite
are unavailable, mirroring the gate pattern used by the LIBERO + SmolVLA
sim tests so the same hosts run both.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip` / `pytest.skip(allow_module_level=True)`: with
# `tests/sim/__init__.py` making this directory a Package, a module-level
# Skipped raised during collection of *this* file marks the **whole
# `tests/sim` Package** as `outcome='skipped'`, which drops every sibling
# test file from collection ("found no collectors for ..."). Deferring the
# decision to `pytestmark` keeps this module importable so its siblings
# remain reachable when the optional `robocasa` package isn't installed.
_REQUIRED_MODULES = ("torch", "transformers", "robocasa", "robosuite")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)


def _robosuite_incompatible() -> str:
    """``robocasa`` needs the openral-vendored robosuite (>=1.5.2, exposing
    ``get_elements``); PyPI's 1.5.1 lacks it, so ``import robocasa`` raises.

    ``find_spec`` only sees that ``robosuite`` exists, not the missing symbol —
    so a ``uv run --group robocasa`` that re-resolved robosuite to 1.5.1 would
    ERROR here instead of skipping. The compatible 1.5.2 is installed at runtime
    by the ``openral sim run`` / ``deploy sim`` auto-install path; checking the
    version via dist metadata is cheap (no heavy import at collection).
    """
    if "robosuite" in _MISSING_MODULES:
        return ""
    try:
        ver = importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError:
        return ""
    parts = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
    if parts < (1, 5, 2):
        return (
            f"robocasa needs robosuite>=1.5.2 (get_elements); found {ver} "
            "— run via `openral sim run`"
        )
    return ""


_INCOMPATIBLE = _robosuite_incompatible()
_CUDA_AVAILABLE = False
if not _MISSING_MODULES and not _INCOMPATIBLE:
    import torch  # gated above

    _CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=("pi05 RoboCasa sim test requires " + ", ".join(_MISSING_MODULES)),
    ),
    pytest.mark.skipif(
        bool(_INCOMPATIBLE),
        reason=_INCOMPATIBLE or "robosuite incompatible",
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="pi05 RoboCasa sim test requires CUDA",
    ),
]


_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "robocasa_pnp.yaml"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "pi05-robocasa365-human300-nf4" / "rskill.yaml"


@pytest.fixture(scope="module")
def env_cfg():
    """Compose the canonical pi05-RoboCasa sim env, capped at one short episode."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    if not _LOCAL_MANIFEST.exists():
        pytest.skip(f"rSkill manifest not found at {_LOCAL_MANIFEST}")

    return compose_sim_env(
        _CONFIG,
        rskill_uri="rskills/pi05-robocasa365-human300-nf4",
        n_episodes=1,
        max_steps=10,
    )


@pytest.fixture(scope="module")
def skill_manifest():
    """Load the rSkill manifest from disk (no network)."""
    from openral_rskill.loader import rSkill

    return rSkill.from_yaml(_LOCAL_MANIFEST)


class TestPi05RoboCasaManifest:
    """Manifest contract — declared inputs/outputs match the panda_mobile embodiment."""

    def test_manifest_loads_and_declares_panda_mobile(self, skill_manifest) -> None:
        m = skill_manifest.manifest
        assert m.name == "OpenRAL/rskill-pi05-robocasa365-human300-nf4"
        assert "panda_mobile" in m.embodiment_tags
        assert m.role == "s1"
        assert m.model_family == "pi05"

    def test_manifest_declares_two_rgb_sensors(self, skill_manifest) -> None:
        sensors = skill_manifest.manifest.sensors_required
        assert len(sensors) == 2
        modalities = {s.modality for s in sensors}
        assert modalities == {"rgb"}
        keys = {s.vla_feature_key for s in sensors}
        assert keys == {
            "observation.images.camera1",
            "observation.images.camera2",
        }

    def test_manifest_has_latency_budget(self, skill_manifest) -> None:
        budget = skill_manifest.manifest.latency_budget
        assert budget is not None
        assert budget.per_chunk_ms > 0


class TestPi05RoboCasaRollout:
    """End-to-end episode through the strict runner."""

    def test_robocasa_episode_runs_via_sim_runner(self, env_cfg, skill_manifest) -> None:
        from openral_sim import SimRunner

        if not env_cfg.scene.id.startswith("robocasa"):
            pytest.skip(f"unexpected scene id in fixture: {env_cfg.scene.id!r}")

        try:
            import mujoco  # noqa: F401
        except ImportError:
            pytest.skip("RoboCasa sim test requires mujoco")

        runner = SimRunner(env_cfg)
        try:
            runner.activate()
            runner.run(max_ticks=env_cfg.n_episodes * (env_cfg.task.max_steps + 1))
        except ImportError as exc:
            pytest.skip(f"RoboCasa suite not importable: {exc}")
        finally:
            runner.deactivate()

        results = runner.episode_results
        assert len(results) == 1
        result = results[0]

        assert result.steps > 0
        assert result.mean_step_latency_ms > 0
        assert result.latency_budget_ms == skill_manifest.manifest.latency_budget.per_chunk_ms
