"""Sim test: NaT VLA (pretrained, zero-shot) runs in-process on the SO-101 MuJoCo box scene.

Drives ``POLICIES['nat_vla']`` with the in-tree ``rskills/nat-vla-so101-pretrained-bf16``
manifest: the real 450M checkpoint (``ahad-j/nat-vla-450m``), the real NAT tokenizer and an
SO-101 dataset's normalisation stats. One step on a two-camera 640x480 observation returns a
finite 6-D joint action in LeRobot servo degrees; the 16-step replay holds the rest of the
streamed chunk in the policy's queue.

Skips unless the checkpoint, tokenizer and dataset stats are in the local HF cache (a Hub
download is not a skip-free path) and lerobot >= 0.7 is installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_RSKILL_DIR = Path(__file__).resolve().parents[2] / "rskills" / "nat-vla-so101-pretrained-bf16"


def _unavailable() -> str:
    try:
        if importlib.util.find_spec("lerobot.policies.pretrained") is None:
            return "lerobot is not installed"
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        return "lerobot / huggingface_hub are not installed"
    try:
        snapshot_download(
            "ahad-j/nat-vla-450m",
            revision="a5c2e13d45832050e364265962f6ad536460e955",
            allow_patterns=["model.safetensors", "vlm/*"],
            local_files_only=True,
        )
        snapshot_download(
            "ahad-j/nat-tokenizer-h32-l4-c6400",
            revision="daee61dc36d74129d7bf3e823871d3e8b3c5fa85",
            allow_patterns=["config.json", "model.safetensors"],
            local_files_only=True,
        )
        hf_hub_download(
            "makermods/eraser_place_unblurry_real",
            "meta/stats.json",
            repo_type="dataset",
            local_files_only=True,
        )
    except Exception:  # reason: any cache miss is the skip, not a failure
        return "nat-vla-450m, its NAT tokenizer or SO-101 stats are not in the HF cache"
    return ""


_SKIP = _unavailable()
pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.skipif(bool(_SKIP), reason=_SKIP)]


@pytest.fixture(scope="module")
def adapter() -> Any:
    import openral_sim.policies  # noqa: F401 — registers nat_vla
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_sim.registry import POLICIES

    env_cfg = SimEnvironment(
        robot_id="so101_follower",
        scene=SceneSpec(
            id="so101_box",
            observation_height=480,
            observation_width=640,
            cameras=["camera1", "camera2"],
        ),
        task=TaskSpec(
            id="so101_box/tube_insertion",
            scene_id="so101_box",
            instruction="insert the orange tube into the slotted block",
            success_key="is_success",
            max_steps=300,
        ),
        vla=VLASpec(id="nat_vla", weights_uri=str(_RSKILL_DIR)),
    )
    built = POLICIES.get("nat_vla")(env_cfg)
    yield built
    built.close()


def _obs() -> dict[str, object]:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    state = np.array([0.0, -100.0, 95.0, 55.0, 5.0, 2.0], dtype=np.float32)
    return {"images": {"camera1": frame, "camera2": frame.copy()}, "state": state}


def test_step_returns_a_finite_6d_joint_action(adapter: Any) -> None:
    adapter.reset()
    action = adapter.step(_obs(), "insert the orange tube into the slotted block")
    assert action.shape == (6,), action.shape
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def test_the_streamed_chunk_replays_n_action_steps(adapter: Any) -> None:
    adapter.reset()
    adapter.step(_obs(), "insert the orange tube into the slotted block")
    assert adapter._policy.config.n_action_steps == 16
