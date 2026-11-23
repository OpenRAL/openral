"""Sim test: NVIDIA Cosmos 3 Edge (bf16) on a real LIBERO observation, in-process on lerobot.

Loads ``rskills/cosmos3-edge-libero-bf16`` against ``scenes/sim/libero_spatial.yaml``
through the eval factory (``make_env`` / ``make_policy``): the real 9.1 GB
``lerobot/Cosmos3-Edge-Policy-LIBERO`` checkpoint and its own processor pipeline, run by
lerobot>=0.7's ``Cosmos3Policy`` — no sidecar, no mocks.

What is asserted
----------------
* One closed-loop episode of ``libero_spatial/0`` (seed 0) succeeds within the model
  card's 210-step budget. Finite, in-range actions are not evidence: a wrong 180-degree
  flip or a wrong rot6d -> axis-angle decode still yields them, but does not solve the
  task (seed 0 took 75 steps on an RTX 5090).
* Every action is a finite 7-D LIBERO command, and the 16-step chunk is replayed in
  full through the chunk executor.

Skip policy
-----------
The checkpoint is 9.1 GB, so a Hub download is not a skip-free path: the test skips
unless it is already in the local HF cache, a CUDA GPU with >= 16 GiB is present, and
lerobot ships the policy (>=0.7) with diffusers. A GPU-less CI runner is the legitimate
skip path.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from tests.sim.conftest import _libero_robosuite_conflict

if TYPE_CHECKING:
    from openral_core import SimEnvironment
    from openral_sim.policies.cosmos3 import _Cosmos3Adapter

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
_RSKILL = "rskills/cosmos3-edge-libero-bf16"
_REPO = "lerobot/Cosmos3-Edge-Policy-LIBERO"
_REV = "9d4de4350e416c12babc9a42db01ab0c5df9f82f"
_MIN_VRAM_GIB = 16


def _unavailable() -> str:
    """Empty string when the test can run, else the skip reason."""
    try:
        # find_spec imports the parent packages, so a missing lerobot raises here.
        if any(
            importlib.util.find_spec(m) is None for m in ("lerobot.policies.cosmos3", "diffusers")
        ):
            return "lerobot>=0.7 with cosmos3 + diffusers (lerobot[cosmos3]) is not installed"
        import torch
    except ImportError:
        return "lerobot>=0.7 / torch are not installed"
    if not torch.cuda.is_available():
        return "Cosmos 3 Edge bf16 requires a CUDA GPU"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total < _MIN_VRAM_GIB:
        return (
            f"Cosmos 3 Edge bf16 requires a CUDA GPU with >= {_MIN_VRAM_GIB} GiB (have {total:.1f})"
        )
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(_REPO, revision=_REV, local_files_only=True)
    except Exception:  # reason: any cache miss is the skip, not a failure
        return f"{_REPO}@{_REV[:8]} is not in the local HF cache (9.1 GB)"
    return ""


_SKIP = _unavailable()
pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_SKIP), reason=_SKIP),
    pytest.mark.skipif(
        _libero_robosuite_conflict(),
        reason="LIBERO needs robosuite 1.4.x; a newer robosuite is installed",
    ),
]


_BUDGET = 210  # the model card's LIBERO-Spatial policy budget


@pytest.fixture(scope="module")
def env_cfg() -> SimEnvironment:
    from tests.sim.conftest import compose_sim_env

    # LIBERO's MuJoCo reset needs a GL backend on a headless host.
    os.environ.setdefault("MUJOCO_GL", "egl")
    return compose_sim_env(_CONFIG, rskill_uri=_RSKILL, n_episodes=1, max_steps=_BUDGET)


def test_closed_loop_episode_succeeds(env_cfg: SimEnvironment) -> None:
    import torch
    from openral_sim import make_env, make_policy

    torch.manual_seed(0)  # the UniPC sampler draws its noise from the global generator
    env = make_env(env_cfg)
    try:
        policy = cast("_Cosmos3Adapter", make_policy(env_cfg))
        try:
            obs = env.reset(seed=env_cfg.seed)
            instruction = str(obs.get("task", "") or env_cfg.task.instruction or "")
            policy.reset()
            success_key = env_cfg.task.success_key or "is_success"
            success = False
            for step in range(_BUDGET):
                action = policy.step(obs, instruction)
                assert action.shape == (7,), action.shape
                assert action.dtype == np.float32
                assert np.isfinite(action).all(), action
                if step == 0:
                    # n_action_steps=16 routes steps through the chunk executor, which
                    # keeps the rest of the chunk in its own buffer.
                    assert policy._policy.config.n_action_steps == 16
                    assert policy._chunk_executor is not None
                    assert len(policy._chunk_executor._buffer) == 15
                result = env.step(action)
                obs = result.observation
                if result.info.get(success_key):
                    success = True
                    break
                if result.terminated or result.truncated:
                    break
            assert success, f"libero_spatial/0 seed {env_cfg.seed} not solved in {_BUDGET} steps"
        finally:
            policy.close()
    finally:
        env.close()
