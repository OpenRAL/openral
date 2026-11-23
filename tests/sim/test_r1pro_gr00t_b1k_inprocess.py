"""Sim-tier test: the BEHAVIOR-1K GR00T checkpoint in-process on lerobot ``groot``.

Real ``rskills/gr00t-n17-b1k-turning-on-radio`` manifest, real
``@POLICIES.register("gr00t_b1k")`` factory, real organizer checkpoint, real
lerobot GR00T processors, loaded fp32 (the open path). The only double is at
the sensor boundary: seeded camera frames and a seeded 61-D proprio.

Skips unless CUDA, lerobot's groot policy, and the organizer checkpoint
(``OPENRAL_BEHAVIOR_GROOT_CHECKPOINT``, a manual Google Drive download) are
present. ~12 GB VRAM.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILL = _REPO_ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"
_CHECKPOINT = os.environ.get("OPENRAL_BEHAVIOR_GROOT_CHECKPOINT", "")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


pytestmark = [
    pytest.mark.skipif(not _cuda(), reason="needs CUDA"),
    pytest.mark.skipif(
        importlib.util.find_spec("lerobot") is None, reason="needs `just sync --group gr00t`"
    ),
    pytest.mark.skipif(
        not (_CHECKPOINT and Path(_CHECKPOINT).is_dir()),
        reason="needs the organizer checkpoint at OPENRAL_BEHAVIOR_GROOT_CHECKPOINT",
    ),
]


def test_r1pro_gr00t_b1k_inprocess_emits_ensembled_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_core import VLASpec
    from openral_sim import _behavior_wire
    from openral_sim.policies import gr00t
    from openral_sim.policies.behavior_groot_inprocess import _BehaviorGrootInprocessAdapter

    monkeypatch.setenv("OPENRAL_BEHAVIOR_GROOT_RUNTIME", "inprocess")
    monkeypatch.setenv("OPENRAL_QUANTIZATION_DTYPE", "fp32")
    env = SimpleNamespace(
        vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)),
        scene=SimpleNamespace(cameras=["head", "left_wrist", "right_wrist"]),
    )
    adapter = gr00t._build_gr00t_b1k(env)
    assert isinstance(adapter, _BehaviorGrootInprocessAdapter)

    rng = np.random.default_rng(0)
    images = {
        "head": rng.integers(0, 256, size=(360, 640, 4), dtype=np.uint8),
        "left_wrist": rng.integers(0, 256, size=(240, 240, 4), dtype=np.uint8),
        "right_wrist": rng.integers(0, 256, size=(240, 240, 4), dtype=np.uint8),
    }
    try:
        adapter.reset()
        actions = []
        for _ in range(6):  # past the 5-chunk ensemble window
            obs = {"images": images, "state": rng.normal(0, 0.1, 61).astype(np.float32)}
            actions.append(adapter.step(obs, "turn on the radio"))
        for action in actions:
            assert action.shape == (_behavior_wire.ACTION_DIM,)
            assert np.isfinite(action).all()
        frame = adapter.last_input_frame()
        assert frame is not None and frame.shape == (224, 224, 3)
    finally:
        adapter.close()
    adapter.reset()  # a reset after close is a no-op, not an AttributeError
