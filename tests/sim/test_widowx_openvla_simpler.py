"""Sim test: OpenVLA-OFT reaches non-zero success on SimplerEnv WidowX.

This is the issue #55 reproduction path for
``rskills/openvla-oft-simpler-widowx-nf4``. It is intentionally opt-in because
it executes a transformers custom-code checkpoint from Hugging Face, requires a
CUDA GPU, and needs the OpenVLA-compatible 4.40-era transformers runtime rather
than the default OpenRAL lerobot runtime.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openral_core import SimEnvironment

_REQUIRED_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "mani_skill",
    "simpler_env",
    "gymnasium",
)
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)
_CUDA_AVAILABLE = False
_TRANSFORMERS_COMPATIBLE = False
if not _MISSING_MODULES:
    import torch
    import transformers

    _CUDA_AVAILABLE = torch.cuda.is_available()
    _TRANSFORMERS_COMPATIBLE = int(transformers.__version__.split(".", maxsplit=1)[0]) < 5

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("OPENRAL_RUN_OPENVLA_SIM") != "1",
        reason="set OPENRAL_RUN_OPENVLA_SIM=1 to run the remote-code OpenVLA rollout",
    ),
    pytest.mark.skipif(
        os.environ.get("OPENRAL_ALLOW_REMOTE_CODE") != "1",
        reason="OpenVLA custom-code model load requires OPENRAL_ALLOW_REMOTE_CODE=1",
    ),
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="OpenVLA SimplerEnv sim test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="OpenVLA SimplerEnv sim test requires CUDA"),
    pytest.mark.skipif(
        not _TRANSFORMERS_COMPATIBLE,
        reason="RLinf OpenVLA custom code requires transformers<5",
    ),
]

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "benchmark" / "widowx_carrot_on_plate.yaml"
_RSKILL = "rskills/openvla-oft-simpler-widowx-nf4"


@pytest.fixture(scope="module")
def env_cfg() -> SimEnvironment:
    """Compose the validated five-episode carrot-on-plate benchmark config."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    env = compose_sim_env(_CONFIG, rskill_uri=_RSKILL, n_episodes=5, max_steps=60)
    return env.model_copy(update={"vla": env.vla.model_copy(update={"device": "cuda:0"})})


def test_openvla_oft_widowx_carrot_nonzero_success(env_cfg: SimEnvironment) -> None:
    from openral_sim.sim_runner import SimRunner

    runner = SimRunner(env_cfg)
    try:
        runner.activate()
        runner.run(max_ticks=env_cfg.n_episodes * ((env_cfg.task.max_steps or 60) + 1))
        successes = sum(1 for episode in runner.episode_results if episode.success)
    finally:
        runner.deactivate()
    assert successes > 0
