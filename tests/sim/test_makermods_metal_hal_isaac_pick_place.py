"""Sim test: a MakerMods pick-and-place episode replayed open loop in the Isaac example scene.

Boots ``scenes/sim/isaac_makermods_metal_pick_place.yaml`` through the real factory on a real
Isaac Sim, puts the cube and the marker at the recorded episode's own layout and feeds its
actions (``HappyEthan/metal_pick_place`` episode 0, the MuJoCo twin's fixture) through the
rSkill's codec units (degrees -> rad, stroke / 100): the arm carries the cube onto the
marker and the scene must grade ``complete``.

``OPENRAL_TEST_ARTIFACT_DIR`` (optional) receives ``isaac_pick_place_replay_ep00.mp4`` (top
view beside the wrist view, 30 fps).

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

_DESK_TOP_Z, _HALF_EDGE = 0.003, 0.0125
_HALF_COMPLETE, _HALF_PARTIAL = 0.012, 0.031  # the scene's region half widths


def test_replayed_episode_puts_the_cube_on_the_marker() -> None:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_core import SimEnvironment, SimScene, VLASpec, load_scene_strict
    from openral_sim.factory import make_env

    ep = np.load(_repo_root() / "tests/unit/fixtures/metal_pick_place_ep00.npz")
    (cx, cy), (mx, my) = ep["cube_xy"].tolist(), ep["target_xy"].tolist()
    scene = load_scene_strict(
        str(_repo_root() / "scenes/sim/isaac_makermods_metal_pick_place.yaml"), SimScene
    )
    opts = cast(dict[str, Any], scene.scene.backend_options)
    for o in opts["objects"]:
        if o["name"] == "cube":
            o["xyz"] = [cx, cy, _DESK_TOP_Z + _HALF_EDGE + 0.0005]
            o["yaw"] = math.radians(float(ep["cube_yaw_deg"]))
        if o["name"] == "marker":
            o["xyz"] = [mx, my, _DESK_TOP_Z + 0.0006]
    regions = opts["placement_success"]["regions"]
    for name, half in (("on_marker", _HALF_COMPLETE), ("touching_marker", _HALF_PARTIAL)):
        regions[name]["min_xyz"] = [mx - half, my - half, _DESK_TOP_Z]
        regions[name]["max_xyz"] = [mx + half, my + half, 0.03]
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
        for a in ep["action"].astype(np.float64):
            robot = np.radians(a)
            robot[6] = np.clip(a[6] / 100.0, 0.0, 1.0)  # stroke -> [0, 1]
            result = env.step(robot.astype(np.float32))
            im = result.observation["images"]
            frames.append(np.concatenate([im["top"], im["wrist"]], axis=1).astype(np.uint8))
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
            imageio.mimwrite(Path(out) / "isaac_pick_place_replay_ep00.mp4", frames, fps=30)
        env.close()
