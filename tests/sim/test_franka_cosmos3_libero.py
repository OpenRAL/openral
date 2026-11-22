"""Sim test: NVIDIA Cosmos 3 Edge (bf16) on a real LIBERO observation, in-process on lerobot.

Loads ``rskills/cosmos3-edge-libero-bf16`` against ``scenes/sim/libero_spatial.yaml``
through the eval factory (``make_env`` / ``make_policy``): the real 9.1 GB
``lerobot/Cosmos3-Edge-Policy-LIBERO`` checkpoint and its own processor pipeline, run by
lerobot>=0.7's ``Cosmos3Policy`` — no sidecar, no mocks.

What is asserted
----------------
* One ``policy.step`` on a real LIBERO reset observation returns a finite 7-D action:
  the 10-D rot6d chunk decoded by lerobot's ``Cosmos3LiberoActionProcessorStep`` into the
  LIBERO OSC command, with the gripper in ``[-1, 1]``.
* The 16-step chunk is replayed in full: after the first step the chunk executor holds
  the other 15.

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


@pytest.fixture(scope="module")
def env_cfg() -> SimEnvironment:
    from tests.sim.conftest import compose_sim_env

    # LIBERO's MuJoCo reset needs a GL backend on a headless host.
    os.environ.setdefault("MUJOCO_GL", "egl")
    return compose_sim_env(_CONFIG, rskill_uri=_RSKILL, n_episodes=1, max_steps=20)


def test_step_returns_finite_libero_action_and_replays_the_chunk(env_cfg: SimEnvironment) -> None:
    from openral_sim import make_env, make_policy

    env = make_env(env_cfg)
    policy = cast("_Cosmos3Adapter", make_policy(env_cfg))
    try:
        obs = env.reset(seed=env_cfg.seed)
        instruction = str(obs.get("task", "") or env_cfg.task.instruction or "")
        policy.reset()
        action = policy.step(obs, instruction)

        assert action.shape == (7,), action.shape
        assert action.dtype == np.float32
        assert np.isfinite(action).all(), action
        assert -1.0 <= float(action[6]) <= 1.0, action

        # n_action_steps=16 routes steps through the chunk executor, which keeps the rest
        # of the chunk in its own buffer (the policy's queue stays unused).
        chunk = policy._policy.config.n_action_steps
        assert chunk == 16
        executor = policy._chunk_executor
        assert executor is not None
        assert len(executor._buffer) == chunk - 1
    finally:
        policy.close()
        env.close()
