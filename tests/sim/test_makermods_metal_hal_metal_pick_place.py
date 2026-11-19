# SPDX-License-Identifier: Apache-2.0
"""MakerMods Metal in the ``metal_pick_place`` MuJoCo scene (HAL-only sim test).

Builds the real scene from ``scenes/sim/makermods_metal_pick_place.yaml`` around
the MJCF that ``makermods:metal`` derives from the pinned upstream URDF, and
checks that

* both cameras render 640x480 RGB, the top view is the real background plate
  everywhere MuJoCo draws no robot / cube / marker, and the leader arm adds no
  actuator in any mode;
* replaying one real episode per layout of ``HappyEthan/metal_pick_place``
  open-loop (its recorded 30 Hz actions, degrees + gripper stroke, through the
  scene's own unit conversion) grasps the cube, carries it onto the marker and
  is graded ``complete`` (cube released, at rest, >= 50 % of its footprint over
  the marker; restock's ``placement_grade`` contract);
* the world render carries a white ring around the marker that the top
  observation does not;
* ``SimRunner`` on a suite scene keeps ``frames`` / ``frame_grades`` aligned and
  reports the final grade (``final_info``, ``grade_counts``).

No mocks (CLAUDE.md §1.11): skips only when ``mujoco`` / a GL renderer is
missing or the pinned MakerMods clone cannot be fetched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco")

from openral_core import RobotDescription, SimScene  # noqa: E402
from openral_core.assets import AssetFetchError, resolve_asset  # noqa: E402
from openral_core.schemas import SimEnvironment, VLASpec  # noqa: E402

from tests.sim.conftest import mujoco_renderer_probe_error  # noqa: E402

if TYPE_CHECKING:
    from openral_sim.backends.metal_pick_place.env import _MetalPickPlaceRollout

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_RENDER_ERROR = mujoco_renderer_probe_error()


@pytest.fixture(scope="module", autouse=True)
def _assets_available() -> None:
    if _RENDER_ERROR is not None:
        pytest.skip(f"mujoco renderer unavailable: {_RENDER_ERROR}")
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "makermods_metal" / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"pinned MakerMods Metal clone not fetchable on this host: {exc}")


def _rollout(size: tuple[int, int] = (640, 480), **opts: Any) -> _MetalPickPlaceRollout:
    from openral_sim import SCENES

    sc = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pick_place.yaml"))
    scene = sc.scene.model_copy(
        update={
            "backend_options": {**(sc.scene.backend_options or {}), **opts},
            "observation_width": size[0],
            "observation_height": size[1],
        }
    )
    env = SimEnvironment(
        robot_id="makermods_metal",
        scene=scene,
        task=sc.task,
        vla=VLASpec(id="zero", weights_uri="zero", device="cpu"),
        seed=0,
    )
    return cast("_MetalPickPlaceRollout", SCENES.get("metal_pick_place")(env))


def test_cameras_render_and_top_view_is_the_real_plate() -> None:
    r = _rollout()
    try:
        obs = r.reset(seed=0)
        top, wrist = obs["images"]["top"], obs["images"]["wrist"]
        assert top.shape == wrist.shape == (480, 640, 3)
        assert top.dtype == wrist.dtype == np.uint8
        assert obs["state"].shape == (7,)
        np.testing.assert_allclose(obs["state"][6], 2.5, atol=0.5)  # closed, dataset start
        assert r._plate is not None
        # The wall (top band) is untouched plate; the cube's orange is composited in.
        np.testing.assert_array_equal(top[:120], r._plate[:120])
        orange = (top[..., 0] > 200) & (top[..., 2] < 120) & (top[..., 1] < 180)
        assert orange.sum() > 50
        assert float(wrist.std()) > 20.0  # textured, lit scene, not a flat fill
        model, _ = r.mujoco_handles()
        for name in ("table", "cart_photo", "marker"):  # real-photo textures load
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0 and model.mat_texid[model.geom_matid[gid]].max() >= 0, name
        # the soft gripper shadow the policy keys on: a grid of shadow-casting lights
        assert int(model.light_castshadow.sum()) == r.options.light_grid**2 > 1
    finally:
        r.close()


@pytest.mark.parametrize("leader", ["mirror_action", "static", "hidden"])
def test_leader_arm_never_adds_actuators(leader: str) -> None:
    r = _rollout(size=(64, 48), leader_arm=leader)
    try:
        model, _ = r.mujoco_handles()
        assert model.nu == 7
        assert model.njnt == 9  # 6 arm + 2 fingers + cube freejoint
        assert (len(r._leader_mocap) == 0) == (leader == "hidden")
    finally:
        r.close()


# One real episode per layout (fixtures: tests/unit/fixtures/metal_pick_place_epNN.npz, see
# its SOURCE.txt): recorded actions + states, cube xy / yaw and marker xy measured from the
# episode's own top-camera frames. ep42/62/91 are held out of the wrist-camera fit.
_REPLAYS = {0: "A", 42: "B", 62: "C", 91: "D"}


# Each step renders the manifest cameras; one replay took 594 s under software OSMesa (16-core
# host) against ~4 s on EGL, so four of them outlast the 15-min CI shard.
@pytest.mark.skipif(
    os.environ.get("MUJOCO_GL") == "osmesa",
    reason="replays render every step: too slow on OSMesa; runs with a hardware renderer",
)
@pytest.mark.parametrize("episode", sorted(_REPLAYS), ids=lambda e: f"ep{e:02d}-{_REPLAYS[e]}")
def test_replaying_a_real_episode_places_the_cube_on_the_marker(episode: int) -> None:
    ep = np.load(_ROOT / "tests" / "unit" / "fixtures" / f"metal_pick_place_ep{episode:02d}.npz")
    cube, target = (tuple(float(v) for v in ep[k]) for k in ("cube_xy", "target_xy"))
    r = _rollout(
        size=(64, 48),
        layout="ep",
        layouts={"ep": [cube, target]},
        spawn_jitter_m=0.0,
        cube_yaw_jitter_deg=0.0,
        cube_yaw_deg=float(ep["cube_yaw_deg"]),
        leader_arm="hidden",
        top_background="render",
        initial_joint_positions=[float(v) for v in ep["state"][0]],
    )
    try:
        r.reset(seed=0)
        for action in ep["action"]:
            res = r.step(action)
            assert res.info["placement_grade"] in ("fail", "complete"), res.info
            if res.terminated:
                break
        else:
            pytest.fail(f"ep{episode}: not graded in {len(ep['action'])} steps; info {res.info}")
        assert res.info["is_success"] and res.reward == 1.0
        assert res.info["lifted"], "success without the cube ever leaving the table"
        assert res.info["placement_overlap"] >= 0.5
        assert res.info["cube_marker_distance_m"] < 0.02
    finally:
        r.close()


# Step of episode 0 at which the cube is lifted and carried (grasped by step 474, released
# at 631 on this replay).
_CARRY_STEP = 550


def _judge_replay(episode: int, *, budget_steps: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Replay a real episode and judge it by the scene's ``SuccessCondition`` (issue #635).

    The condition is the one the scene file declares for the task's prompt, re-centred on
    the marker this episode's own frames measured (as the replay's layout is). The judge is
    the runner's (``openral_core.success.SuccessJudge``), fed one ``TrackedObjectSet`` per
    step, stamped with sim time. This MuJoCo scene has no object tracker in the loop, so
    the set is written from the simulator's state: the cube's pose in ``base_link``,
    ``bound`` while a finger touches it. (The real tracker feeding the judge:
    ``tests/integration/test_object_tracker_node_live.py``.) Stops as a deployed run does:
    when the condition has settled, or on ``budget_steps``.

    Returns:
        The final ``TaskOutcome`` and the simulator's own grading ``info`` at that step.
    """
    from openral_core import (
        FitKind,
        ObjectRole,
        PlaceRegion,
        Pose6D,
        TaskVerdict,
        TrackedObject,
        TrackedObjectSet,
        TrackStatus,
        success_condition_for_prompt,
    )
    from openral_core.success import SuccessJudge

    ep = np.load(_ROOT / "tests" / "unit" / "fixtures" / f"metal_pick_place_ep{episode:02d}.npz")
    cube, target = (tuple(float(v) for v in ep[k]) for k in ("cube_xy", "target_xy"))
    scene = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pick_place.yaml"))
    declared = success_condition_for_prompt(scene.task.instruction, scene.success_conditions)
    assert declared is not None
    centre = (target[0], target[1], declared.region.pose.xyz[2])
    pose = declared.region.pose.model_copy(update={"xyz": centre})
    judge = SuccessJudge(
        declared.model_copy(update={"region": declared.region.model_copy(update={"pose": pose})})
    )
    frame = declared.region.frame_id
    r = _rollout(
        size=(64, 48),
        layout="ep",
        layouts={"ep": [cube, target]},
        spawn_jitter_m=0.0,
        cube_yaw_jitter_deg=0.0,
        cube_yaw_deg=float(ep["cube_yaw_deg"]),
        leader_arm="hidden",
        top_background="render",
        initial_joint_positions=[float(v) for v in ep["state"][0]],
    )

    def tracked() -> TrackedObjectSet:
        _, data = r.mujoco_handles()
        held = r._touches_fingers()
        stamp = r.sim_time_ns() or 0
        xyz = tuple(float(v) for v in data.xpos[r._cube_body])
        link_pose = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="link6")
        track = TrackedObject(
            object_id="trk:1",
            epoch=0,
            label="cube",
            role=ObjectRole.TARGET,
            declaration_key="obj:cube@0",
            status=TrackStatus.BOUND if held else TrackStatus.CONFIRMED,
            box=PlaceRegion(
                frame_id=frame,
                pose=Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame),
                half_extents=(0.0125, 0.0125, 0.0125),
            ),
            fit_kind=FitKind.SILHOUETTE_CONE,
            support_z=0.003,
            support_ref="map",
            floor_z=0.03,
            visible_cells=1,
            occupied_cells=1,
            last_fit_stamp_ns=stamp,
            last_confirmed_stamp_ns=stamp,
            bound_hand="joint8" if held else None,
            attach_link="link6" if held else None,
            pose_in_link=link_pose if held else None,
        )
        return TrackedObjectSet(frame_id=frame, stamp_ns=stamp, revision=stamp, objects=[track])

    try:
        r.reset(seed=0)
        for action in ep["action"][:budget_steps]:
            res = r.step(action)
            judge.observe(tracked())
            if judge.outcome().verdict is TaskVerdict.SUCCEEDED:
                break
        return judge.outcome(), dict(res.info)
    finally:
        r.close()


@pytest.mark.skipif(
    os.environ.get("MUJOCO_GL") == "osmesa",
    reason="replays render every step: too slow on OSMesa; runs with a hardware renderer",
)
def test_a_run_is_judged_succeeded_or_failed_from_world_state() -> None:
    """Issue #635: one run graded success and one graded failure, from world-state evidence."""
    from openral_core import TaskVerdict

    # The whole episode: the cube is placed and released; the condition settles and the
    # run ends early, a success the simulator's own grade agrees with.
    placed, info = _judge_replay(0)
    assert placed.verdict is TaskVerdict.SUCCEEDED, placed
    assert placed.source == "success_condition" and placed.object_ids == ("trk:1",)
    assert (placed.resting, placed.required) == (1, 1) and placed.held_s >= 1.0
    assert info["placement_grade"] == "complete" and info["cube_marker_distance_m"] < 0.02

    # The same policy cut off mid-carry (a run that ends on its budget): the cube is in
    # the gripper, so the run failed -- and the simulator grades it "fail" too.
    carried, info = _judge_replay(0, budget_steps=_CARRY_STEP)
    assert carried.verdict is TaskVerdict.FAILED and "trk:1 is held" in carried.reason, carried
    assert info["lifted"] and info["placement_grade"] == "fail"

    # Cut off before the grasp: the cube still lies where it started, outside the region.
    untouched, info = _judge_replay(0, budget_steps=30)
    assert untouched.verdict is TaskVerdict.FAILED, untouched
    assert "trk:1 is outside it" in untouched.reason and not info["lifted"]


def test_world_render_rings_the_marker_but_the_policy_view_does_not() -> None:
    r = _rollout()
    try:
        top = r.reset(seed=0)["images"]["top"]
        world = r.render()
        assert world is not None and world.shape == top.shape
        changed = np.any(world != top, axis=-1)
        assert changed.any()
        # every changed pixel is the white ring, on a circle around the marker
        np.testing.assert_array_equal(world[changed], 255)
        ys, xs = np.nonzero(changed)
        rad = np.hypot(xs - xs.mean(), ys - ys.mean())
        assert rad.min() > 0.6 * rad.max()  # a ring, not a disc
        assert r.set_task("Pick up the cube")["grade_scores"]["partial"] == 0.5
    finally:
        r.close()


def test_sim_runner_records_the_grade_of_every_world_frame() -> None:
    """SimRunner on a suite scene: frames, frame_grades and final_info stay aligned."""
    from openral_core import load_benchmark_suite
    from openral_sim import SimRunner
    from openral_sim.benchmark import episode_records, grade_counts

    sc = load_benchmark_suite(str(_ROOT / "benchmarks" / "makermods_metal_pick_place.yaml"))[0]
    cfg = SimEnvironment(
        robot_id="makermods_metal",
        scene=sc.scene.model_copy(update={"observation_width": 64, "observation_height": 48}),
        task=sc.task.model_copy(update={"max_steps": 4}),
        vla=VLASpec(id="zero", weights_uri="placeholder", extra={"action_dim": 7}),
        n_episodes=1,
        record_video=True,
    )
    runner = SimRunner(cfg)
    runner.activate()
    try:
        runner.run(max_ticks=10)
    finally:
        runner.deactivate()
    (ep,) = runner.episode_results
    # One frame before each step, plus the ending step's own frame.
    assert len(ep.frames) == len(ep.frame_grades) == ep.steps + 1 == 5
    assert ep.frame_grades[0] is None  # rendered before the first step
    assert ep.frame_grades[1:] == [("fail", 0.0)] * 4  # the cube never moved
    # The marker ring turns with the grade on that last frame (red: a fail at the limit).
    from openral_sim._website_video import grade_colour

    red = np.all(ep.frames[-1] == np.array(grade_colour(0.0), np.uint8), axis=-1)
    assert red.sum() > 10 and not np.all(ep.frames[0] == np.array(grade_colour(0.0)), axis=-1).any()
    assert ep.final_info["placement_grade"] == "fail" and ep.final_info["lifted"] is False
    assert grade_counts([ep]) == {"fail": 1}
    assert float(str(episode_records([ep])[0]["cube_marker_distance_m"])) > 0.1
