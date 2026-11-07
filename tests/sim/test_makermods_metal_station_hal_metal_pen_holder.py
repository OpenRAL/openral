# SPDX-License-Identifier: Apache-2.0
"""MakerMods Metal Arm Station in the ``metal_pen_holder`` MuJoCo scene (HAL-only sim test).

Builds the real scene from ``scenes/sim/makermods_metal_pen_holder.yaml`` around the MJCF
that ``makermods:metal_station`` derives from the pinned upstream URDF, and checks that

* the three cameras render 640x480 RGB and the top view is the real background plate
  wherever MuJoCo draws no robot / cup / pen;
* replaying real episodes of ``makermods/pen_holder_bi_20260723_161553`` open loop (their
  recorded 30 Hz actions in degrees + gripper stroke, through the scene's own unit
  conversion, from each episode's own start layout) has the left arm lift the cup, the
  right arm drop the pen into it, and grades ``complete``;
* the world render carries the cup's rim outline that the top observation does not.

Fixture: ``tests/unit/fixtures/metal_pen_holder_episodes.npz`` (see its SOURCE.txt). No
mocks (CLAUDE.md §1.11): skips only when ``mujoco`` / a GL renderer is missing or the pinned
MakerMods clone cannot be fetched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco")

from openral_core import RobotDescription, SimScene  # noqa: E402
from openral_core.assets import AssetFetchError, resolve_asset  # noqa: E402
from openral_core.schemas import SimEnvironment, VLASpec  # noqa: E402

from tests.sim.conftest import mujoco_renderer_probe_error  # noqa: E402

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_RENDER_ERROR = mujoco_renderer_probe_error()
_EPISODES = np.load(_ROOT / "tests" / "unit" / "fixtures" / "metal_pen_holder_episodes.npz")


@pytest.fixture(scope="module", autouse=True)
def _assets_available() -> None:
    if _RENDER_ERROR is not None:
        pytest.skip(f"mujoco renderer unavailable: {_RENDER_ERROR}")
    desc = RobotDescription.from_yaml(
        str(_ROOT / "robots" / "makermods_metal_station" / "robot.yaml")
    )
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"pinned MakerMods Metal clone not fetchable on this host: {exc}")


def _rollout(**opts: Any) -> Any:
    from openral_sim.factory import make_env

    sc = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pen_holder.yaml"))
    scene = sc.scene.model_copy(
        update={"backend_options": {**(sc.scene.backend_options or {}), **opts}}
    )
    return make_env(
        SimEnvironment(
            robot_id="makermods_metal_station",
            scene=scene,
            task=sc.task.model_copy(update={"max_steps": 2000}),
            vla=VLASpec(id="zero", weights_uri="zero", device="cpu"),
            seed=0,
        )
    )


def test_cameras_render_and_the_top_view_is_the_plate_behind_the_robot() -> None:
    env = _rollout()
    try:
        obs = env.reset(seed=0)
        assert {k: v.shape for k, v in obs["images"].items()} == {
            "top": (480, 640, 3),
            "wrist_left": (480, 640, 3),
            "wrist_right": (480, 640, 3),
        }
        assert obs["state"].shape == (14,)
        plate = env._plate
        assert plate is not None
        # The wall band at the top of the frame is pure plate (nothing composited there).
        assert np.array_equal(obs["images"]["top"][:60], plate[:60])
        world = env.render()
        assert world is not None and not np.array_equal(world, obs["images"]["top"])
    finally:
        env.close()


@pytest.mark.parametrize("episode", [0, 7, 30])
def test_replayed_episode_drops_the_pen_into_the_cup(episode: int) -> None:
    hx, hy, hyaw, px, py, pyaw = (float(v) for v in _EPISODES[f"layout_{episode}"])
    env = _rollout(holder_start=[hx, hy, hyaw], pen_start=[px, py, pyaw])
    try:
        env.reset(seed=0)
        z0 = float(env._data.xpos[env._holder_body][2])
        cup_lift = 0.0
        result = None
        for a in _EPISODES[f"action_{episode}"]:
            result = env.step(a)
            cup_lift = max(cup_lift, float(env._data.xpos[env._holder_body][2]) - z0)
            if result.terminated:
                break
        assert result is not None
        assert cup_lift > 0.02, "the left arm never lifted the cup"
        assert result.info["lifted"], "the right arm never lifted the pen"
        assert result.info["placement_grade"] == "complete", result.info
        assert result.terminated and result.reward == 1.0
        assert result.info["is_success"] is True
    finally:
        env.close()
