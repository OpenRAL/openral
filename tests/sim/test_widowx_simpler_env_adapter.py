"""Sim test: real SimplerEnv WidowX carrot-on-plate via the openral scene adapter.

Exercises ``openral_sim.backends.simpler_env._build_simpler_env_scene``
end-to-end against the upstream ``mani_skill`` Bridge-data digital twin
so the adapter's friendly-name → MS3 env-id resolution, version bump,
and shared MS3 obs-extraction helpers all run on the real path.

What is asserted
----------------
* ``SimScene.from_yaml`` accepts (and ``BenchmarkScene.from_yaml`` validates)
  ``scenes/benchmark/widowx_carrot_on_plate.yaml`` with
  ``robot_id: widowx`` (free-axis scene, no fixed_robot).
* The friendly task name ``widowx_carrot_on_plate`` resolves to
  ``PutCarrotOnPlateInScene-v1`` (auto-bumped from the upstream
  ENVIRONMENT_MAP's ``-v0`` suffix).
* The adapter resets the env and emits a populated :class:`Observation`
  (RGB from ``3rd_view_camera`` + agent.qpos/qvel state vector).
* A canonical zero-action step propagates with finite reward and the
  Bridge-task ``success`` info field.

Skips automatically when ``mani_skill`` or ``simpler_env`` are missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REQUIRED_MODULES = ("mani_skill", "simpler_env", "gymnasium")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="SimplerEnv sim test requires " + ", ".join(_MISSING_MODULES),
    ),
]

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "benchmark" / "widowx_carrot_on_plate.yaml"


@pytest.fixture(scope="module")
def scene_env():
    from openral_core import BenchmarkScene, load_scene_strict

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    return load_scene_strict(str(_CONFIG), BenchmarkScene)


@pytest.fixture(scope="module")
def sim(scene_env):
    from openral_core import SimEnvironment, VLASpec
    from openral_sim.registry import SCENES

    env_cfg = SimEnvironment(
        robot_id=scene_env.robot_id or "widowx",
        scene=scene_env.scene,
        task=scene_env.task,
        vla=VLASpec(id="zero", weights_uri="stub"),
        seed=scene_env.seed,
        n_episodes=scene_env.n_episodes,
    )
    built = SCENES.get("simpler_env")(env_cfg)
    yield built
    built.close()


class TestSimplerEnvAdapter:
    def test_friendly_name_resolves_with_version_bump(self) -> None:
        from openral_sim.backends.simpler_env import _resolve_friendly_name

        env_id, kwargs = _resolve_friendly_name("widowx_carrot_on_plate")
        # ENVIRONMENT_MAP carries -v0 in upstream simpler-env; MS3 v3.0.x
        # registers -v1, so the adapter must bump.
        assert env_id == "PutCarrotOnPlateInScene-v1"
        assert kwargs == {}

    def test_reset_returns_populated_observation(self, sim) -> None:
        obs = sim.reset(seed=0)
        assert "images" in obs and "camera1" in obs["images"]
        rgb = obs["images"]["camera1"]
        # The Bridge env renders at its native 480x640; the adapter
        # forwards what MS3 emits without resizing.
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8
        # WidowX-250s has 8 joints (6 arm + 2 finger); state = qpos + qvel.
        assert obs["state"].shape == (16,)
        assert obs["state"].dtype == np.float32

    def test_step_propagates_action(self, sim) -> None:
        sim.reset(seed=0)
        t0 = sim.sim_time_ns()
        assert t0 is not None
        result = sim.step(np.zeros(7, dtype=np.float32))
        t1 = sim.sim_time_ns()
        assert np.isfinite(result.reward)
        assert "success" in result.info
        assert isinstance(result.info["success"], (bool, np.bool_))
        assert t1 is not None
        assert t1 > t0
