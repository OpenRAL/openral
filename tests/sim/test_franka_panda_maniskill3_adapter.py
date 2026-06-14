"""Sim test: real ManiSkill3 PickCube-v1 via the openral scene adapter.

Exercises ``openral_sim.backends.maniskill3._build_maniskill3_scene``
end-to-end against the upstream ``mani_skill`` SAPIEN backend so the
adapter's obs / state / action plumbing is validated without dragging
in a (non-existent) MS3-specific rSkill.

What is asserted
----------------
* ``SimScene.from_yaml`` accepts (and ``BenchmarkScene.from_yaml`` validates)
  ``scenes/benchmark/maniskill_pick_cube.yaml``
  with ``robot_id: franka_panda`` (free-axis scene, no fixed_robot).
* The adapter resets the env from a seed and emits a populated
  :class:`Observation` (non-empty RGB + non-empty state vector under the
  ``state_dict+rgb`` obs mode).
* A canonical zero-action step propagates through the env and returns
  finite reward / a typed info dict.

Skips automatically when ``mani_skill`` is missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REQUIRED_MODULES = ("mani_skill", "gymnasium")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="ManiSkill3 sim test requires " + ", ".join(_MISSING_MODULES),
    ),
]

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "benchmark" / "maniskill_pick_cube.yaml"


@pytest.fixture(scope="module")
def scene_env():
    from openral_core import SimScene, load_scene_strict

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    return load_scene_strict(str(_CONFIG), SimScene)


@pytest.fixture(scope="module")
def sim(scene_env):
    """Build the ManiSkill3 scene exactly the way ``openral sim run`` does."""
    from openral_core import SimEnvironment, VLASpec
    from openral_sim.registry import SCENES

    # The adapter cares about scene + task only; supply a placeholder VLASpec
    # so SimEnvironment validates (the scene factory never reads vla).
    env_cfg = SimEnvironment(
        robot_id=scene_env.robot_id or "franka_panda",
        scene=scene_env.scene,
        task=scene_env.task,
        vla=VLASpec(id="zero", weights_uri="stub"),
        seed=scene_env.seed,
        n_episodes=scene_env.n_episodes,
    )
    built = SCENES.get("maniskill3")(env_cfg)
    yield built
    built.close()


class TestManiSkill3Adapter:
    def test_reset_returns_populated_observation(self, sim) -> None:
        obs = sim.reset(seed=0)
        assert "images" in obs and "camera1" in obs["images"]
        rgb = obs["images"]["camera1"]
        # Resolution is driven by the scene config (sensor_configs), not hardcoded.
        expected_hw = (sim.scene.observation_height, sim.scene.observation_width, 3)
        assert rgb.shape == expected_hw
        assert rgb.dtype == np.uint8
        # state_dict+rgb yields agent.qpos + agent.qvel; Panda has 9 joints
        # (7 arm + 2 finger), so the concatenation is 18-D.
        assert obs["state"].shape == (18,)
        assert obs["state"].dtype == np.float32

    def test_step_propagates_action(self, sim) -> None:
        sim.reset(seed=0)
        # Action dim follows the env's controller, which is version-dependent
        # (ManiSkill 3.0.x panda_wristcam = 7 arm + 1 gripper = 8); derive it
        # from the action space rather than hardcoding.
        action_dim = int(sim._env.action_space.shape[-1])
        result = sim.step(np.zeros(action_dim, dtype=np.float32))
        # PickCube-v1 reward is shaped; a zero-action step yields a small
        # positive value at the initial pose. The contract we care about is
        # "finite and machine-comparable", not a specific magnitude.
        assert np.isfinite(result.reward)
        assert "success" in result.info
        assert isinstance(result.info["success"], (bool, np.bool_))
