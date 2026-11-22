"""Sim test: LingBot-VLA 2.0 in-process on lerobot's ``LingbotVLAV2Policy``.

Drives ``POLICIES['lingbot_vla_v2']`` (openral_sim.policies.lingbot_vla_v2) with the
in-tree ``rskills/lingbot-vla-v2-robotwin-bf16`` manifest: the real 6.38 B
checkpoint (``lerobot/lingbot_vla_v2_robotwin``) and its own processor pipeline,
loaded in bf16 in the workspace interpreter — no sidecar.

What is asserted
----------------
* One ``step`` on a RoboTwin-shaped observation (three 256x256 RGB views + 14-D
  proprio) returns a **finite 14-D** action — a real forward pass, the checkpoint's
  pre/post pipeline included (55-D canonical slots back to 14-D).
* The 50-step chunk replays ``n_action_steps`` (25): after the first step the
  policy's queue holds the remaining 24 steps.

Skip policy
-----------
The checkpoint is 25.5 GB, so a Hub download is not a skip-free path: the test
skips unless it is already in the local HF cache, a CUDA GPU with ≥16 GB is
present (bf16 is ~12.8 GB resident), and lerobot ships the policy (>=0.7).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.sim.conftest import _robotwin_obs

_RSKILL_DIR = Path(__file__).resolve().parents[2] / "rskills" / "lingbot-vla-v2-robotwin-bf16"
_REPO = "lerobot/lingbot_vla_v2_robotwin"
_REV = "54f79a4b6e9166bda91b5010088097ff3191bca2"
_ACTION_DIM = 14
_MIN_VRAM_GIB = 16


def _unavailable() -> str:
    """Empty string when the test can run, else the skip reason."""
    if importlib.util.find_spec("lerobot.policies.lingbot_vla_v2") is None:
        return "lerobot>=0.7 with lingbot_vla_v2 is not installed"
    import torch

    if not torch.cuda.is_available():
        return "LingBot-VLA 2.0 bf16 needs a CUDA GPU"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total < _MIN_VRAM_GIB:
        return f"LingBot-VLA 2.0 bf16 needs a >= {_MIN_VRAM_GIB} GiB GPU (have {total:.1f})"
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(_REPO, revision=_REV, local_files_only=True)
    except Exception:  # reason: any cache miss is the skip, not a failure
        return f"{_REPO}@{_REV[:8]} is not in the local HF cache (25.5 GB)"
    return ""


_SKIP = _unavailable()
pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.skipif(bool(_SKIP), reason=_SKIP)]


@pytest.fixture(scope="module")
def adapter() -> Any:
    import openral_sim.policies  # noqa: F401 — registers lingbot_vla_v2
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_sim.registry import POLICIES

    env_cfg = SimEnvironment(
        robot_id="aloha_agilex",
        scene=SceneSpec(
            id="robotwin",
            backend=PhysicsBackend.SAPIEN,
            observation_height=256,
            observation_width=256,
            cameras=["camera1", "camera2", "camera3"],
        ),
        task=TaskSpec(
            id="robotwin/beat_block_hammer",
            scene_id="robotwin",
            instruction="pick up the hammer and beat the block",
            success_key="is_success",
            max_steps=300,
        ),
        vla=VLASpec(id="lingbot_vla_v2", weights_uri=str(_RSKILL_DIR)),
    )
    built = POLICIES.get("lingbot_vla_v2")(env_cfg)
    yield built
    built.close()


def test_step_returns_finite_14d_action(adapter: Any) -> None:
    adapter.reset()
    action = adapter.step(_robotwin_obs(), "pick up the hammer and beat the block")
    assert action.shape == (_ACTION_DIM,), action.shape
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def test_chunk_replays_n_action_steps(adapter: Any) -> None:
    adapter.reset()
    adapter.step(_robotwin_obs(), "pick up the hammer and beat the block")
    assert len(adapter._policy._queues["action"]) == adapter._policy.config.n_action_steps - 1
