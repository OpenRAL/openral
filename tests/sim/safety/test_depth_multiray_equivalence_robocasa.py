# SPDX-License-Identifier: Apache-2.0
"""The batched ``mj_multiRay`` still misses geometry a robot can collide with.

#111 made the depth synth cast one ``mj_ray`` per strided pixel instead of the
batched ``mj_multiRay``, at a measured cost premium, because the batched call's
broad-phase culling reports free space where a surface is. #180 then filtered
non-collidable geometry out of the cast *before* it runs and conjectured that
the premium was now recoverable, "since the ``mj_multiRay`` body cull only ever
mis-skipped non-collidable geoms".

#195 measured that conjecture instead of adopting it, and it is **false**. On
the four validation-matrix scenes at the deploy stride, with #180's filter
applied to both casters so they see the identical world, the batched cast
disagrees on 4 113 of 65 536 rays — and **every geom it skips is collidable**
(``contype`` or ``conaffinity`` set). Not one is the intangible decoration #180
removed. The failure is entirely in the unsafe direction: no return ever comes
back nearer, they come back up to 480 mm *farther*, which is the perception
path telling the safety kernel's world grid that a countertop is not there.

What #195 did **not** establish is the mechanism. #111's account — a body BVH
built from collision geoms only, so a visual geom outside the collision extent
is skipped — cannot be it: every geom skipped here is collidable, and the
batched cast still returns 184 rays on the very body whose countertop it walks
through. Nor could the skip be reproduced in a hand-written MJCF (far-outlier
collision geoms in the same body, body yaw, geom yaw — all agree ray for ray),
which is why this gate runs on real scenes and has no unit-tier twin.

So these two tests are the gate that keeps the swap from happening quietly:

* the first adjudicates the **shipped** cast against an independent analytic
  ray/box intersection — it fails if anyone swaps the caster;
* the second is the #195 measurement itself, and trips if the disagreement
  ever disappears (a fixed upstream MuJoCo would be a reason to re-open #195,
  not a silent licence to batch).

Real scenes and a real compiled ``MjModel`` throughout (CLAUDE.md §1.11): the
four RoboCasa scenes the validation matrix runs, built through the same
``build_sim_env_from_yaml`` path ``openral deploy sim`` uses, cast through the
same ``panda_mobile`` ``front_depth`` intrinsics at the same ``stride=4``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("openral_sim")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

from openral_core import RobotDescription  # noqa: E402  # reason: after importorskip
from openral_hal.depth_cloud import (  # noqa: E402
    depth_synth_kwargs,
    is_depth_sensor,
    robot_self_body_ids,
)
from openral_hal.sim_bringup import build_sim_env_from_yaml  # noqa: E402
from openral_sim.backends.depth_camera import (  # noqa: E402
    _body_geom_ids,
    _transparent_geoms,
    noncollidable_geom_ids,
)

#: The validation matrix's four scenes — the same set #180 measured its filter
#: on, so the two results are directly comparable — each with the #195 verdict
#: for that scene: does the batched cast lose collidable geometry here?
#:
#: The fridge is `False` and that is not a reprieve. Its camera is on the fridge
#: front at close range — median return 1.28 m, and `..._freezer_door_main`
#: alone takes 4 711 of the 16 384 rays — with **no** counter geom in view at
#: all, so the cull has nothing here to fire on. The same caster loses 3 815
#: rays one scene over. A per-scene expectation is what makes this a
#: measurement rather than a slogan.
_SCENES = (
    ("scenes/deploy/robocasa_baguette.yaml", True),
    ("scenes/deploy/robocasa_sink_cup.yaml", True),
    ("scenes/deploy/robocasa_fridge_drawer.yaml", False),
    ("scenes/deploy/robocasa_drawer_utensil.yaml", True),
)

#: `SimSensorBridge` defaults (`sim_sensor_bridge.py`): stride 4, 5 m range,
#: intrinsics rescaled to the scene's 512² render. 512/4 = 128² = 16 384 rays.
_STRIDE = 4
_MAX_RANGE_M = 5.0
_RENDER_SIZE = (512, 512)


@pytest.fixture(scope="module", params=_SCENES, ids=lambda p: p[0].split("/")[-1])
def scene_cast(request: pytest.FixtureRequest) -> tuple[str, bool, tuple[Any, ...]]:
    """One built scene + its ray bundle, shared by both tests in this module.

    Building a RoboCasa kitchen costs ~8 s; both tests cast the same rays
    through it, so it is built once per scene.
    """
    scene_yaml, expects_loss = request.param
    return scene_yaml, expects_loss, _cast_setup(scene_yaml)


def _cast_setup(scene_yaml: str) -> tuple[Any, Any, NDArray[np.float64], NDArray[np.float64], int]:
    """Build the scene and derive the exact ray bundle the deploy synth casts.

    Returns ``(model, data, origin, dir_world, bodyexclude)``.
    """
    env, seed = build_sim_env_from_yaml(scene_yaml)
    env.reset(seed=seed)
    model, data = env.mujoco_handles()

    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    spec = next(s for s in desc.sensors if is_depth_sensor(s) and s.name == "front_depth")
    kwargs = depth_synth_kwargs(spec, max_range_default=_MAX_RANGE_M, render_size=_RENDER_SIZE)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, kwargs["camera_name"])
    assert cam_id >= 0, f"{kwargs['camera_name']!r} is not a camera in {scene_yaml}"

    # The pinhole grid of `_cast_depth_rays`, verbatim.
    us = np.arange(0, kwargs["width"], _STRIDE, dtype=np.float64)
    vs = np.arange(0, kwargs["height"], _STRIDE, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    dir_opt = np.empty((grid_u.size, 3), dtype=np.float64)
    dir_opt[:, 0] = (grid_u.ravel() - kwargs["cx"]) / kwargs["fx"]
    dir_opt[:, 1] = (grid_v.ravel() - kwargs["cy"]) / kwargs["fy"]
    dir_opt[:, 2] = 1.0
    dir_opt /= np.linalg.norm(dir_opt, axis=1, keepdims=True)
    dir_cam = dir_opt * np.array([1.0, -1.0, -1.0], dtype=np.float64)
    rot = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    dir_world = np.ascontiguousarray((rot @ dir_cam.T).T)
    origin = np.ascontiguousarray(np.asarray(data.cam_xpos[cam_id], dtype=np.float64))

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base")
    return model, data, origin, dir_world, int(base_id)


def _hidden_geoms(model: Any) -> NDArray[np.int64]:
    """The filter the shipped cast applies: intangible geometry + the robot."""
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    self_bodies = robot_self_body_ids(model, [j.sim_joint_name for j in desc.joints])
    return np.concatenate((noncollidable_geom_ids(model), _body_geom_ids(model, self_bodies)))


def _per_pixel(
    model: Any,
    data: Any,
    origin: NDArray[np.float64],
    dirs: NDArray[np.float64],
    groups: NDArray[np.uint8],
    bodyexclude: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """The shipped caster: one ``mj_ray`` per ray."""
    n = dirs.shape[0]
    geomids = np.full(n, -1, dtype=np.int32)
    distances = np.full(n, -1.0, dtype=np.float64)
    out = np.zeros(1, dtype=np.int32)
    for i in range(n):
        out[0] = -1
        distances[i] = mujoco.mj_ray(model, data, origin, dirs[i], groups, 1, bodyexclude, out)
        geomids[i] = out[0]
    return geomids, distances


def _batched(
    model: Any,
    data: Any,
    origin: NDArray[np.float64],
    dirs: NDArray[np.float64],
    groups: NDArray[np.uint8],
    bodyexclude: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """The candidate: one ``mj_multiRay`` for the whole bundle.

    ``cutoff`` is ``mjMAXVAL`` — effectively none — so the only difference from
    :func:`_per_pixel` is the batched call's body cull, not a range gate.
    """
    n = dirs.shape[0]
    geomids = np.full(n, -1, dtype=np.int32)
    distances = np.full(n, -1.0, dtype=np.float64)
    mujoco.mj_multiRay(
        model,
        data,
        origin,
        dirs.ravel(),
        groups,
        1,
        bodyexclude,
        geomids,
        distances,
        None,
        n,
        float(mujoco.mjMAXVAL),
    )
    return geomids, distances


def _nearest_box_hit(
    model: Any,
    data: Any,
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    candidates: NDArray[np.int64],
) -> tuple[int, float]:
    """Independent oracle: nearest ray/box intersection by the slab method.

    Deliberately shares no code with MuJoCo's caster — it is what decides
    *which* of the two disagreeing answers is the surface actually there.
    Returns ``(geom_id, distance)``, or ``(-1, inf)`` when nothing is struck.
    """
    best_id, best_t = -1, np.inf
    for gid in candidates:
        rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        half = np.asarray(model.geom_size[gid], dtype=np.float64)
        o = rot.T @ (origin - np.asarray(data.geom_xpos[gid], dtype=np.float64))
        v = rot.T @ direction
        t_near, t_far = -np.inf, np.inf
        for axis in range(3):
            if abs(v[axis]) < 1e-12:
                if abs(o[axis]) > half[axis]:
                    t_near = np.inf
                    break
                continue
            a = (-half[axis] - o[axis]) / v[axis]
            b = (half[axis] - o[axis]) / v[axis]
            t_near, t_far = max(t_near, min(a, b)), min(t_far, max(a, b))
        if t_far >= max(t_near, 0.0) and 0.0 <= t_near < best_t:
            best_id, best_t = int(gid), float(t_near)
    return best_id, best_t


@pytest.mark.sim
def test_the_shipped_depth_cast_reports_the_true_nearest_collidable_surface(
    scene_cast: tuple[str, bool, tuple[Any, ...]],
) -> None:
    """Adjudicate the shipped caster against geometry, not against MuJoCo.

    This is the assertion that fails the moment someone swaps
    ``_cast_depth_rays`` onto ``mj_multiRay``: on three of the four scenes
    here the batched call answers with a surface *behind* the one the slab
    test finds (the fourth, the fridge, has nothing in view for its body cull
    to fire on — see ``_SCENES``).
    """
    scene_yaml, _, (model, data, origin, dirs, bodyexclude) = scene_cast
    with _transparent_geoms(model, _hidden_geoms(model)) as groups:
        geomids, distances = _per_pixel(model, data, origin, dirs, groups, bodyexclude)
        visible_boxes = np.flatnonzero(
            (np.asarray(model.geom_type) == mujoco.mjtGeom.mjGEOM_BOX)
            & (groups[np.asarray(model.geom_group)] != 0)
            & (np.asarray(model.geom_bodyid) != bodyexclude)
        ).astype(np.int64)

        # Adjudicating every ray against every box is O(rays x geoms) in
        # Python; a fixed pseudo-random sample of the rays that struck a box
        # is enough to catch a swapped caster, which moves thousands of them.
        box_rays = np.flatnonzero(np.isin(geomids, visible_boxes))
        assert box_rays.size >= 100, f"{scene_yaml}: too few box returns to adjudicate"
        sample = np.random.default_rng(0).choice(box_rays, size=100, replace=False)

    wrong = [
        (int(i), int(geomids[i]), float(distances[i]), gid, t)
        for i in sample
        for gid, t in [_nearest_box_hit(model, data, origin, dirs[i], visible_boxes)]
        if gid != int(geomids[i]) or abs(t - float(distances[i])) > 1e-6
    ]
    assert not wrong, (
        f"{scene_yaml}: the depth cast disagrees with an independent ray/box "
        f"intersection on {len(wrong)} of {sample.size} sampled rays "
        f"(first: {wrong[0]}). The shipped per-pixel `mj_ray` matches it "
        "exactly; `mj_multiRay` does not (see #195)."
    )


@pytest.mark.sim
def test_batched_mj_multiray_still_skips_collidable_geometry(
    scene_cast: tuple[str, bool, tuple[Any, ...]],
) -> None:
    """#195's measurement: the premium #180 hoped to recover is not recoverable.

    Both casters see the identical filtered world. Every geom the batched one
    fails to see is collidable — the intangible geometry #180 blamed is not
    even in the cast any more — and every disagreement is a surface reported
    too far away, never too near.
    """
    scene_yaml, expects_loss, (model, data, origin, dirs, bodyexclude) = scene_cast
    with _transparent_geoms(model, _hidden_geoms(model)) as groups:
        ref_ids, ref_dist = _per_pixel(model, data, origin, dirs, groups, bodyexclude)
        new_ids, new_dist = _batched(model, data, origin, dirs, groups, bodyexclude)

    disagree = (ref_ids != new_ids) & (ref_ids >= 0)
    skipped = np.unique(ref_ids[disagree])
    contype = np.asarray(model.geom_contype)[skipped]
    conaffinity = np.asarray(model.geom_conaffinity)[skipped]
    intangible = skipped[(contype == 0) & (conaffinity == 0)]
    both = (ref_ids >= 0) & (new_ids >= 0)
    delta = np.where(both, new_dist - ref_dist, 0.0)

    assert intangible.size == 0, (
        f"{scene_yaml}: {intangible.size} of the {skipped.size} geoms the batched "
        "cast skips are non-collidable — #180's filter should already have "
        "removed them from both casts."
    )
    assert np.count_nonzero(delta < -1e-9) == 0, (
        f"{scene_yaml}: the batched cast reported a NEARER surface on "
        f"{np.count_nonzero(delta < -1e-9)} rays, which the body cull cannot "
        "explain — investigate before trusting either number."
    )
    assert bool(np.any(disagree)) == expects_loss, (
        f"{scene_yaml}: the batched cast was measured in #195 to "
        f"{'lose' if expects_loss else 'keep'} collidable geometry here and "
        f"now does the opposite ({int(np.count_nonzero(disagree))} rays "
        f"disagree). That is not licence to swap the caster either way — the "
        f"cull behaviour changed under mujoco {mujoco.__version__}. Re-run "
        "#195's measurement on all four scenes and re-open the issue."
    )
