"""Sim test: MakerMods pen-holder episodes replayed open loop in the Isaac example scene.

Boots ``scenes/sim/isaac_makermods_metal_pen_holder.yaml`` through the real factory on a
real Isaac Sim, puts the cup and the pen at each episode's own start layout and feeds the
recorded actions of ``makermods/pen_holder_bi_20260723_161553`` through the rSkill's codec
units (degrees -> rad, stroke / 100): the left arm lifts the cup, the right arm drops the
pen into it, and the scene must grade ``complete``. Same fixture and episodes as the MuJoCo
twin's test (``tests/sim/test_makermods_metal_station_hal_metal_pen_holder.py``).

``OPENRAL_TEST_ARTIFACT_DIR`` (optional) receives ``isaac_pen_holder_replay_ep<N>.mp4`` (top
view above the two wrist views, 30 fps).

Skip policy: no Isaac sidecar, or no pyzmq / msgpack (the isaac_sim wire).
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tests.sim.conftest import _repo_root, _sidecar_python_available

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]

_EPISODES = np.load(_repo_root() / "tests/unit/fixtures/metal_pen_holder_episodes.npz")
_SHELF_TOP_Z, _PEN_RADIUS = 0.09, 0.01


@pytest.mark.parametrize("episode", [0])
def test_replayed_episode_drops_the_pen_into_the_cup(episode: int) -> None:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_core import SimEnvironment, SimScene, VLASpec, load_scene_strict
    from openral_sim.factory import make_env

    hx, hy, hyaw, px, py, pyaw = (float(v) for v in _EPISODES[f"layout_{episode}"])
    scene = load_scene_strict(
        str(_repo_root() / "scenes/sim/isaac_makermods_metal_pen_holder.yaml"), SimScene
    )
    for o in cast(dict[str, Any], scene.scene.backend_options)["objects"]:
        if o["name"] == "pen_holder":
            o["xyz"], o["yaw"] = [hx, hy, _SHELF_TOP_Z + 0.0005], math.radians(hyaw)
        if o["name"] == "pen":
            o["xyz"], o["yaw"] = [px, py, _PEN_RADIUS + 0.0005], math.radians(pyaw)
    env = make_env(
        SimEnvironment(
            robot_id=scene.robot_id,
            scene=scene.scene,
            task=scene.task,
            base_pose=scene.base_pose,
            vla=VLASpec(id="zero", weights_uri="placeholder"),
        )
    )
    frames = []
    try:
        env.reset(seed=0)
        result = None
        for a in _EPISODES[f"action_{episode}"].astype(np.float64):
            robot = np.radians(a)
            robot[[6, 13]] = np.clip(a[[6, 13]] / 100.0, 0.0, 1.0)  # stroke -> [0, 1]
            result = env.step(robot.astype(np.float32))
            im = result.observation["images"]
            wrists = np.concatenate([im["wrist_left"], im["wrist_right"]], axis=1)[::2, ::2]
            frames.append(np.concatenate([im["top"], wrists], axis=0).astype(np.uint8))
            if result.terminated:
                break
        assert result is not None
        assert result.info["placement_grade"] == "complete", result.info["object_positions"]
        assert result.terminated and result.reward == 1.0
    finally:
        out = os.environ.get("OPENRAL_TEST_ARTIFACT_DIR")
        if out and frames and importlib.util.find_spec("imageio") is not None:
            import imageio.v2 as imageio

            Path(out).mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(Path(out) / f"isaac_pen_holder_replay_ep{episode}.mp4", frames, fps=30)
        env.close()
