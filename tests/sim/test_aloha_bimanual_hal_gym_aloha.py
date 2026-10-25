"""Sim test for the gym-aloha scene adapter's observation rendering.

``openral_sim.backends.aloha`` swaps gym-aloha's three-camera
``get_observation`` for one that renders only ``top`` through a cached
dm_control ``Camera``. This pins that the swap is invisible to the policy:
the ``top`` frame matches a fresh ``physics.render`` of the same camera, on
reset and after real physics steps.

"Matches" allows a handful of pixels off by one intensity level: two fresh
EGL renders of the same post-step state already differ that way on an RTX
5070 Laptop (measured before the swap too — e.g. 3 / 21 of 921 600 pixels),
so exact equality would test the GPU driver, not the adapter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from openral_core import SimScene, load_scene_strict

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("gym_aloha") is None, reason="gym-aloha not installed"
    ),
]

_CONFIG = Path(__file__).parent.parent.parent / "scenes" / "sim" / "aloha_transfer_cube.yaml"


def test_top_frame_matches_fresh_render_and_side_cameras_are_skipped() -> None:
    from openral_core import SimEnvironment, VLASpec
    from openral_sim.factory import make_env

    scene = load_scene_strict(str(_CONFIG), SimScene)
    env_cfg = SimEnvironment(
        robot_id="aloha_bimanual",
        scene=scene.scene,
        task=scene.task,
        vla=VLASpec(id="zero", weights_uri="none"),
        n_episodes=1,
    )
    env = make_env(env_cfg)
    try:
        # reason: reach the dm_control Environment behind the SimRollout protocol.
        inner = env._env.unwrapped._env  # type: ignore[attr-defined]
        physics = inner.physics
        # The unread ``angle`` / ``front_close`` views are no longer rendered.
        assert list(inner.task.get_observation(physics)["images"]) == ["top"]
        obs = env.reset(seed=0)
        frames = [obs["images"]["top"]]
        expected = [physics.render(height=480, width=640, camera_id="top")]
        rng = np.random.default_rng(0)
        for _ in range(5):
            action = rng.uniform(-0.2, 0.2, size=14).astype(np.float32)
            result = env.step(action)
            frames.append(result.observation["images"]["top"])
            expected.append(physics.render(height=480, width=640, camera_id="top"))
        for got, want in zip(frames, expected, strict=True):
            assert got.shape == (480, 640, 3)
            diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
            assert int(diff.max()) <= 1
            assert int(np.count_nonzero(diff)) < 100
        # Frames are owned copies, not views of the reused camera buffer.
        assert not np.array_equal(frames[0], frames[-1])
    finally:
        env.close()
