"""Sim test: LingBot-VLA 2.0 (NF4) real inference via the out-of-process sidecar.

Drives ``POLICIES['lingbot_vla2']`` (openral_sim.policies.lingbot_vla2) through the
real ZMQ sidecar: the boot helper provisions/execs the upstream ``LingbotVLAv2Server``
(NF4 Qwen3-VL backbone + bf16 sparse-MoE expert) in its torch-2.8 venv, and the
openral-side adapter connects and drives ``ping`` / ``reset`` / ``get_action`` /
``close`` on a RoboTwin-shaped observation (three 256x256 RGB views + 14-D proprio).

What is asserted
----------------
* The ``lingbot_vla2`` policy is registered and its adapter connects to the sidecar
  (ping on connect).
* One ``policy.step`` on a RoboTwin-shaped observation returns a **finite 14-D**
  action (arm.position 12 + effector.position 2) — a real forward pass through the
  NF4-quantized 6.38 B model, not a mock.
* The sidecar returns a **multi-step** chunk: after one ``step`` the adapter's replay
  queue is non-empty (regression guard for the ``use_length`` truncation that would
  otherwise emit a single 14-D step per inference).
* The 8 GB fit holds: total GPU memory (nvidia-smi) stays under 8.0 GiB.

Why a constructed observation (not a live SAPIEN frame)
-------------------------------------------------------
The NF4 model is ~6.9 GiB resident and the RoboTwin SAPIEN env is ~1.9 GiB (both
measured on the 8 GiB reference host), so they cannot co-reside on an 8 GiB card
(a full ``openral benchmark run --suite robotwin`` needs >=12 GiB or CPU-rendered
SAPIEN). The model, adapter, msgpack wire, and NF4 quantization are all real; only
the pixel/state input is synthesized to the verified RoboTwin contract.

Skip policy
-----------
The sidecar is an externally-provisioned torch-2.8 venv (multi-GB, CUDA-pinned;
CLAUDE.md §1.9/§1.12). The test skips unless BOTH the openral-side wire
(pyzmq/msgpack) is importable AND the sidecar interpreter + upstream checkout +
a local checkpoint are resolvable via env overrides
(``OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON`` / ``OPENRAL_LINGBOT_VLA2_REPO`` /
``OPENRAL_LINGBOT_VLA2_CKPT``) or their cache defaults. A GPU-less CI runner is the
legitimate skip path; a provisioned dev host with a GPU runs it for real.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]


def _sidecar_python() -> Path | None:
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON")
    if override:
        return Path(override) if Path(override).is_file() else None
    cache = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar"
    default = cache / ".venv" / "bin" / "python"
    return default if default.is_file() else None


def _repo_checkout() -> Path | None:
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_REPO")
    if override:
        p = Path(override)
        return p if (p / "lingbotvla").is_dir() else None
    default = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar" / "source"
    return default if (default / "lingbotvla").is_dir() else None


def _checkpoint() -> str | None:
    """Local checkpoint dir. HF auto-download is not a skip-free path (25 GB)."""
    ckpt = os.environ.get("OPENRAL_LINGBOT_VLA2_CKPT")
    return ckpt if ckpt and Path(ckpt).is_dir() else None


def _gpu_available() -> bool:
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _gpu_mem_used_mib() -> int:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return int(out)


_SIDECAR_PY = _sidecar_python()
_REPO = _repo_checkout()
_CKPT = _checkpoint()

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="lingbot_vla2 wire needs " + ", ".join(_WIRE_MISSING) + " (pyzmq/msgpack)",
    ),
    pytest.mark.skipif(not _gpu_available(), reason="LingBot-VLA 2.0 NF4 requires a CUDA GPU"),
    pytest.mark.skipif(
        _SIDECAR_PY is None or _REPO is None or _CKPT is None,
        reason=(
            "LingBot sidecar not provisioned: set OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON, "
            "OPENRAL_LINGBOT_VLA2_REPO, OPENRAL_LINGBOT_VLA2_CKPT (or their cache defaults)"
        ),
    ),
]

_VRAM_CEILING_MIB = 8000
_STATE_DIM = 14
_ACTION_DIM = 14


@pytest.fixture(scope="module")
def lingbot_adapter():
    import openral_sim.policies  # noqa: F401 — registers lingbot_vla2
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_sim.registry import POLICIES

    assert "lingbot_vla2" in POLICIES

    scene = SceneSpec(
        id="robotwin",
        backend=PhysicsBackend.SAPIEN,
        observation_height=256,
        observation_width=256,
        cameras=["camera1", "camera2", "camera3"],
    )
    task = TaskSpec(
        id="robotwin/pick_place",
        scene_id="robotwin",
        instruction="pick up the block and place it on the plate",
        success_key="is_success",
        max_steps=300,
    )
    env_cfg = SimEnvironment(
        robot_id="aloha_agilex",
        scene=scene,
        task=task,
        vla=VLASpec(id="lingbot_vla2", weights_uri=str(_CKPT), extra={"model_id": str(_CKPT)}),
    )
    adapter = POLICIES.get("lingbot_vla2")(env_cfg)  # connect() -> ping
    yield adapter
    adapter.close()


def _robotwin_obs() -> dict[str, object]:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return {
        "images": {"camera1": img, "camera2": img.copy(), "camera3": img.copy()},
        "state": np.zeros(_STATE_DIM, dtype=np.float32),
    }


def test_step_returns_finite_14d_action(lingbot_adapter) -> None:
    lingbot_adapter.reset()
    action = lingbot_adapter.step(_robotwin_obs(), "pick up the block and place it on the plate")
    assert action.shape == (_ACTION_DIM,), action.shape
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def test_sidecar_returns_multistep_chunk(lingbot_adapter) -> None:
    """Regression guard: use_length must not truncate the chunk to a single step."""
    lingbot_adapter.reset()
    lingbot_adapter.step(_robotwin_obs(), "pick up the block and place it on the plate")
    # After popping the first step, the replay queue must still hold the rest of
    # the predicted chunk (empty here == the use_length=1 truncation bug).
    assert len(lingbot_adapter._queue) >= 1


def test_nf4_fit_under_8gib(lingbot_adapter) -> None:
    lingbot_adapter.reset()
    lingbot_adapter.step(_robotwin_obs(), "pick up the block and place it on the plate")
    assert _gpu_mem_used_mib() < _VRAM_CEILING_MIB
