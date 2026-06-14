"""Sim test: real Isaac Sim Franka lift-cube via the out-of-process sidecar (ADR-0045).

Exercises ``openral_sim.backends.isaac_sim`` end-to-end: the openral (py3.12)
process auto-spawns the Isaac Lab sidecar (py3.11 venv), which launches a real
Omniverse Kit app, builds a PhysX Franka + cube scene, RTX-renders a camera, and
answers ``reset`` / ``step`` / ``render`` over ZMQ. A ``random`` mock policy
supplies actions so no VLA checkpoint is needed.

What is asserted
----------------
* The ``isaac_sim`` scene factory connects to the auto-spawned sidecar.
* ``reset(seed)`` returns an eval-shaped Observation with a populated RGB frame
  and a non-empty proprioception state.
* A random-action ``step`` returns a finite reward and a typed ``info`` dict
  carrying the success key.
* ``render`` returns a non-trivial (non-all-zero) RTX frame.

Skip policy
-----------
Isaac Sim is an externally-provisioned dependency (~50 GB, RTX GPU, separate
license; CLAUDE.md §1.9). The test skips unless BOTH pyzmq/msgpack are importable
on the openral venv AND the sidecar interpreter is resolvable via
``OPENRAL_ISAAC_SIDECAR_PYTHON`` (or the cache default exists). CI runners without
an RTX GPU + provisioned sidecar skip — the legitimate skip path (§1.12).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def _sidecar_python_available() -> bool:
    override = os.environ.get("OPENRAL_ISAAC_SIDECAR_PYTHON")
    if override:
        return Path(override).is_file()
    default = Path.home() / ".cache" / "openral" / "isaac-sidecar" / ".venv" / "bin" / "python"
    return default.is_file()


_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (uv sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]


@pytest.fixture(scope="module")
def isaac_rollout():
    import openral_sim.backends  # noqa: F401 — registers the scene factory
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_sim.registry import SCENES

    scene = SceneSpec(
        id="isaac_sim",
        backend=PhysicsBackend.ISAACSIM,
        observation_height=128,
        observation_width=128,
        backend_options={"headless": True, "boot_timeout_s": 1200, "control_mode": "joint"},
    )
    task = TaskSpec(
        id="isaac_sim/lift_cube",
        scene_id="isaac_sim",
        instruction="lift the cube",
        success_key="is_success",
        max_steps=50,
    )
    env_cfg = SimEnvironment(
        robot_id="franka_panda",
        scene=scene,
        task=task,
        vla=VLASpec(id="random", weights_uri="none"),
    )
    rollout = SCENES.get("isaac_sim")(env_cfg)
    yield rollout
    rollout.close()


def test_reset_returns_populated_observation(isaac_rollout) -> None:
    obs = isaac_rollout.reset(seed=0)
    assert set(obs) >= {"images", "state", "task"}
    frame = obs["images"]["camera1"]
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    # Franka 9 joints + cube xyz = 12-D proprioception.
    assert obs["state"].shape == (12,)
    assert obs["task"] == "lift the cube"


def test_lift_scene_exposes_agent_and_wrist_cameras(isaac_rollout) -> None:
    """IsaacLiftScene renders both camera1 (agent-view) and camera2 (eye-in-hand
    wrist) so a two-camera LIBERO-shaped rSkill (gr00t / rldx) clears the camera
    contract — it used to expose camera1 only."""
    obs = isaac_rollout.reset(seed=0)
    images = obs["images"]
    assert {"camera1", "camera2"} <= set(images), f"got camera keys {sorted(images)}"
    for key in ("camera1", "camera2"):
        frame = images[key]
        assert frame.shape == (128, 128, 3)
        assert frame.dtype == np.uint8


def test_random_step_returns_finite_reward_and_success_key(isaac_rollout) -> None:
    isaac_rollout.reset(seed=1)
    rng = np.random.default_rng(0)
    action = (rng.standard_normal(8) * 0.2).astype(np.float32)
    result = isaac_rollout.step(action)
    assert np.isfinite(result.reward)
    assert "is_success" in result.info
    assert isinstance(result.info["is_success"], bool)


def test_render_returns_non_trivial_rtx_frame(isaac_rollout) -> None:
    isaac_rollout.reset(seed=2)
    # A few steps so the RTX renderer has warmed past the first (blank) frame.
    for _ in range(4):
        isaac_rollout.step(np.zeros(8, dtype=np.float32))
    frame = isaac_rollout.render()
    assert frame is not None
    assert frame.shape == (128, 128, 3)
    # Real render carries signal — not an all-zero placeholder.
    assert np.count_nonzero(frame) > 0
