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
mocks (CLAUDE.md §1.11): skips when ``mujoco`` / a GL renderer is missing or the pinned
MakerMods clone cannot be fetched; the replays also skip under ``MUJOCO_GL=osmesa`` (the per-PR
CI shards) and run nightly in ``.github/workflows/metal-replays.yml`` instead (#563).
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


def _rollout(size: tuple[int, int] | None = None, **opts: Any) -> Any:
    from openral_sim.factory import make_env

    sc = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pen_holder.yaml"))
    update: dict[str, Any] = {"backend_options": {**(sc.scene.backend_options or {}), **opts}}
    if size is not None:
        update |= {"observation_width": size[0], "observation_height": size[1]}
    scene = sc.scene.model_copy(update=update)
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
        # the wrist views' grid of shadow-casting lights (the replays below render without it)
        assert int(env._model.light_castshadow.sum()) == env.options.light_grid**2 > 1
    finally:
        env.close()


def test_large_requested_render_size_grows_the_offscreen_buffer() -> None:
    from openral_sim.backends.metal_pen_holder._assets import compose_metal_pen_holder

    desc = RobotDescription.from_yaml(
        str(_ROOT / "robots" / "makermods_metal_station" / "robot.yaml")
    )
    model = compose_metal_pen_holder(desc, width=1280, height=960)
    assert model.vis.global_.offwidth >= 1280
    assert model.vis.global_.offheight >= 960


def test_camera_offset_moves_and_turns_the_wrist_camera_in_its_own_frame() -> None:
    from openral_sim.backends.metal_pen_holder._assets import (
        MetalPenHolderOptions,
        compose_metal_pen_holder,
    )

    desc = RobotDescription.from_yaml(
        str(_ROOT / "robots" / "makermods_metal_station" / "robot.yaml")
    )
    moved = compose_metal_pen_holder(desc)  # default offsets: wrist_left is refitted
    bare = compose_metal_pen_holder(desc, MetalPenHolderOptions(camera_offsets={}))
    cam = mujoco.mj_name2id(bare, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_left")
    q0, q1 = bare.cam_quat[cam], moved.cam_quat[cam]
    q0_inv = np.zeros(4)
    mujoco.mju_negQuat(q0_inv, q0)
    rel, shift = np.zeros(4), np.zeros(3)
    mujoco.mju_mulQuat(rel, q0_inv, q1)
    mujoco.mju_rotVecQuat(shift, moved.cam_pos[cam] - bare.cam_pos[cam], q0_inv)
    np.testing.assert_allclose(shift, (0.0007, 0.0154, 0.0076), atol=1e-9)
    # scipy Rotation.from_euler("xyz", (-3.85, -2.22, -2.25), degrees=True), wxyz.
    ref = np.array([0.99904271555633, -0.03395861841986, -0.01869788151546, -0.02026953269524])
    assert min(np.abs(rel - ref).max(), np.abs(rel + ref).max()) < 1e-9


# Each step renders three views. At 640x480 under the default 6 x 6 grid of shadow-casting
# lights (4096 px shadow maps) a ~700-step replay did not finish in 15 min on a software
# renderer (OSMesa, 16-core host). The replay grades physics only, never pixels, so it renders
# small views with one unshadowed light and a rendered (not plate-composited) top view; the
# full-size shadowed path stays covered by the camera test above. The 205k-face meshes still
# cost ~100 s per replay on OSMesa on a GitHub-hosted runner, too much for the shared 15-min
# CI shard (#563).
@pytest.mark.skipif(
    os.environ.get("MUJOCO_GL") == "osmesa",
    reason=(
        "Metal replay renders every step: too slow for the OSMesa CI shard (#563); "
        "runs nightly in .github/workflows/metal-replays.yml, or locally with MUJOCO_GL=egl"
    ),
)
@pytest.mark.parametrize("episode", [0, 7, 30])
def test_replayed_episode_drops_the_pen_into_the_cup(episode: int) -> None:
    hx, hy, hyaw, px, py, pyaw = (float(v) for v in _EPISODES[f"layout_{episode}"])
    env = _rollout(
        size=(64, 48),
        holder_start=[hx, hy, hyaw],
        pen_start=[px, py, pyaw],
        light_grid=1,
        light_castshadow=False,
        top_background="render",
    )
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
