"""Sim test: real RoboTwin 2.0 dual-arm SAPIEN env via the out-of-process sidecar (ADR-0061).

Exercises ``openral_sim.backends.robotwin`` end-to-end: the openral (py3.12) process
auto-spawns the RoboTwin sidecar venv, which builds LeRobot's native
``robotwin`` SAPIEN env for the requested task and answers ``reset`` / ``step`` /
``render`` over ZMQ. A random 14-D action source is used so no VLA checkpoint is
needed for the wiring check.

What is asserted
----------------
* The ``robotwin`` scene factory connects to the auto-spawned sidecar.
* ``reset(seed)`` returns an eval-shaped Observation with the three re-keyed RGB
  cameras (camera1/camera2/camera3) and a non-empty proprioception state.
* A random-action ``step`` returns a finite reward and a typed ``info`` dict
  carrying the success key.

Skip policy
-----------
RoboTwin's SAPIEN + lerobot-main + asset stack is an externally-provisioned sidecar
venv (multi-GB, CUDA-pinned, Linux-only; CLAUDE.md §1.9 / ADR-0061). The test skips
unless BOTH pyzmq/msgpack are importable on the openral venv AND the sidecar
interpreter is resolvable via ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` (or the cache
default exists). Hosts without the provisioned venv skip — the legitimate skip path
(§1.12); the SAPIEN engine itself is independently verified on the reference host
(ADR-0061 §Live verification).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def _sidecar_python_available() -> bool:
    override = os.environ.get("OPENRAL_ROBOTWIN_SIDECAR_PYTHON")
    if override:
        return Path(override).is_file()
    default = Path.home() / ".cache" / "openral" / "robotwin-sidecar" / ".venv" / "bin" / "python"
    return default.is_file()


_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="robotwin wire needs " + ", ".join(_WIRE_MISSING) + " (uv sync --group robotwin)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="RoboTwin sidecar venv not provisioned (set OPENRAL_ROBOTWIN_SIDECAR_PYTHON)",
    ),
]


@pytest.fixture(scope="module")
def robotwin_rollout():
    import openral_sim.backends  # noqa: F401 — registers the scene factory
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_sim.registry import SCENES

    scene = SceneSpec(
        id="robotwin",
        backend=PhysicsBackend.SAPIEN,
        observation_height=256,
        observation_width=256,
        cameras=["camera1", "camera2", "camera3"],
        backend_options={"boot_timeout_s": 900},
    )
    task = TaskSpec(
        id="robotwin/lift_pot",
        scene_id="robotwin",
        instruction="lift the pot with both arms",
        success_key="is_success",
        max_steps=300,
    )
    env_cfg = SimEnvironment(
        robot_id="aloha_agilex",
        scene=scene,
        task=task,
        vla=VLASpec(id="random", weights_uri="none"),
    )
    rollout = SCENES.get("robotwin")(env_cfg)
    yield rollout
    rollout.close()


def test_reset_returns_populated_observation(robotwin_rollout) -> None:
    obs = robotwin_rollout.reset(seed=0)
    assert set(obs) >= {"images", "state", "task"}
    # All three RoboTwin cameras re-keyed to the scene's camera1/2/3.
    assert {"camera1", "camera2", "camera3"} <= set(obs["images"])
    frame = obs["images"]["camera1"]
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[:2] == (256, 256)
    assert obs["state"].size > 0


def test_random_step_returns_finite_reward_and_success_key(robotwin_rollout) -> None:
    robotwin_rollout.reset(seed=1)
    rng = np.random.default_rng(0)
    action = (rng.standard_normal(robotwin_rollout.action_dim) * 0.1).astype(np.float32)
    result = robotwin_rollout.step(action)
    assert np.isfinite(result.reward)
    assert "is_success" in result.info
    assert isinstance(result.info["is_success"], bool)
