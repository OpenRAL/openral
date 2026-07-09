"""Sim test: MolmoAct2 (NF4) on a real LIBERO observation via the migrated adapter.

Loads ``rskills/molmoact2-libero-nf4`` against ``scenes/sim/libero_spatial.yaml``
through the eval factory (``make_env`` / ``make_policy``) and drives one real
inference step on a real LIBERO MuJoCo observation — no mocks.

What is asserted
----------------
* The adapter builds its model graph from lerobot's **in-tree**
  ``MolmoAct2ForConditionalGeneration`` (``lerobot.policies.molmoact2.*``),
  i.e. the migration off the ``trust_remote_code`` ``AutoModel`` path landed —
  and it loaded WITHOUT ``OPENRAL_ALLOW_REMOTE_CODE`` being set (no remote code).
* One ``policy.step`` on a real LIBERO reset observation returns a finite
  action vector of the manifest's declared action dim (7 for LIBERO Franka:
  6-DoF eef delta + gripper).
* The NF4 8 GiB fit holds: peak VRAM stays under 8.0 GiB.

Skips automatically when CUDA / lerobot / transformers / the LIBERO suite are
unavailable (a GPU-less CI runner is the legitimate skip path). Runs for real
on a dev host with a GPU.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Deferred skip via pytestmark (see the sibling SmolVLA-LIBERO test for why a
# module-level Skipped would poison tests/sim package collection).
_REQUIRED_MODULES = ("torch", "transformers", "lerobot", "bitsandbytes", "accelerate")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)
_CUDA_AVAILABLE = False
if not _MISSING_MODULES:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()


def _libero_robosuite_conflict() -> bool:
    """True when an installed robosuite (>=1.5) blocks the LIBERO 1.4.x runtime."""
    import importlib.metadata as _md

    if importlib.util.find_spec("robosuite") is None:
        return False
    try:
        return not _md.version("robosuite").startswith("1.4")
    except _md.PackageNotFoundError:
        return False


pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="MolmoAct2 LIBERO sim test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="MolmoAct2 NF4 requires CUDA"),
    pytest.mark.skipif(
        _libero_robosuite_conflict(),
        reason="LIBERO needs robosuite 1.4.x; a newer robosuite is installed",
    ),
]

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
_RSKILL = "rskills/molmoact2-libero-nf4"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "molmoact2-libero-nf4" / "rskill.yaml"
_PEAK_VRAM_CEILING_B = int(8.0 * 1024**3)  # NF4 8 GiB fit


@pytest.fixture(scope="module")
def env_cfg():
    """Compose the MolmoAct2-LIBERO sim env, capped at one short episode."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists() or not _LOCAL_MANIFEST.exists():
        pytest.skip("MolmoAct2-LIBERO scene/manifest not found")
    # LIBERO's MuJoCo reset needs a GL backend on a headless host.
    os.environ.setdefault("MUJOCO_GL", "egl")
    return compose_sim_env(_CONFIG, rskill_uri=_RSKILL, n_episodes=1, max_steps=5)


class TestMolmoAct2LiberoMigratedAdapter:
    """The migrated adapter loads the in-tree class and emits a valid action."""

    def test_loads_intree_class_and_steps(self, env_cfg) -> None:
        import numpy as np
        from openral_sim import make_env, make_policy

        # The migration must NOT require the remote-code acknowledgement env any
        # more (the in-tree class executes no remote code).
        os.environ.pop("OPENRAL_ALLOW_REMOTE_CODE", None)

        try:
            import mujoco  # noqa: F401
        except ImportError:
            pytest.skip("LIBERO sim test requires mujoco")

        torch.cuda.reset_peak_memory_stats()
        env = make_env(env_cfg)
        policy = make_policy(env_cfg)
        try:
            # The model graph must be lerobot's in-tree MolmoAct2 class, not a
            # transformers_modules.* remote-code class.
            model_module = type(policy._model).__module__
            assert model_module.startswith("lerobot.policies.molmoact2"), model_module

            try:
                obs = env.reset(seed=env_cfg.seed)
            except ImportError as exc:
                pytest.skip(f"LIBERO suite not importable: {exc}")

            instruction = str(obs.get("task", "") or env_cfg.task.instruction or "")
            action = policy.step(obs, instruction)

            # Manifest declares a 7-D LIBERO Franka action (6-DoF eef delta + gripper).
            assert action.shape == (7,), action.shape
            assert np.all(np.isfinite(action)), action

            peak = torch.cuda.max_memory_allocated()
            assert peak <= _PEAK_VRAM_CEILING_B, (
                f"peak VRAM {peak / 1024**3:.2f} GiB > 8.0 GiB NF4 ceiling"
            )
        finally:
            policy.close()
            env.close()
