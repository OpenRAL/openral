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

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
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
    _cast_depth_rays,
    _transparent_geoms,
    noncollidable_geom_ids,
)


@dataclass(frozen=True)
class _SceneCast:
    """One built scene and the deploy synth's ray bundle through it."""

    scene_yaml: str
    expected_disagreeing: int
    model: Any
    data: Any
    origin: NDArray[np.float64]
    dirs: NDArray[np.float64]
    bodyexclude: int
    self_bodies: frozenset[int]
    synth_kwargs: dict[str, Any]


#: The validation matrix's four scenes — the same set #180 measured its filter
#: on, so the two results are directly comparable — each with the exact number
#: of rays #195 measured the batched cast losing on it.
#:
#: The counts are asserted exactly, not as "more than zero". A release that
#: narrows the cull to one stray ray on baguette would leave a `> 0` gate green
#: while the table in docs/reference/world-map-fidelity.md silently rots.
#:
#: The fridge's 0 is not a reprieve. Its camera is on the fridge front at close
#: range — median return 1.28 m, and `..._freezer_door_main` alone takes 4 711
#: of the 16 384 rays — with **no** counter geom in view at all, so nothing
#: there fires the skip. The same caster loses 3 815 rays one scene over.
_SCENES = (
    ("scenes/deploy/robocasa_baguette.yaml", 3815),
    ("scenes/deploy/robocasa_sink_cup.yaml", 39),
    ("scenes/deploy/robocasa_fridge_drawer.yaml", 0),
    ("scenes/deploy/robocasa_drawer_utensil.yaml", 259),
)

#: `SimSensorBridge` defaults (`sim_sensor_bridge.py:1847`, `lifecycle.py:1406`).
#: The render size is NOT a constant here: the bridge passes `self._render_size()`,
#: the live rendered frame shape, so the ray bundle follows the scene. Read it
#: off the scene the same way, or a scene dropped to 256² would leave this gate
#: measuring a bundle deploy no longer casts.
_STRIDE = 4
_MAX_RANGE_M = 5.0
_MANIFEST = "robots/panda_mobile/robot.yaml"
_SENSOR = "front_depth"


@pytest.fixture(scope="module", params=_SCENES, ids=lambda p: p[0].split("/")[-1])
def scene_cast(request: pytest.FixtureRequest) -> Iterator[_SceneCast]:
    """One built scene + the exact ray bundle the deploy synth casts on it.

    Building a RoboCasa kitchen costs ~8 s and both tests cast the same rays
    through it, so it is built once per scene and closed on teardown.
    """
    scene_yaml, expected_disagreeing = request.param
    env, seed = build_sim_env_from_yaml(scene_yaml)
    try:
        env.reset(seed=seed)
        yield _build_cast(scene_yaml, expected_disagreeing, env)
    finally:
        env.close()


def _build_cast(scene_yaml: str, expected_disagreeing: int, env: Any) -> _SceneCast:
    """Derive the deploy synth's ray bundle from a built scene."""
    model, data = env.mujoco_handles()

    desc = RobotDescription.from_yaml(_MANIFEST)
    spec = next(s for s in desc.sensors if is_depth_sensor(s) and s.name == _SENSOR)
    kwargs = depth_synth_kwargs(
        spec, max_range_default=_MAX_RANGE_M, render_size=_render_size(scene_yaml)
    )

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

    base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base"))
    self_bodies = robot_self_body_ids(model, [j.sim_joint_name for j in desc.joints])
    return _SceneCast(
        scene_yaml=scene_yaml,
        expected_disagreeing=expected_disagreeing,
        model=model,
        data=data,
        origin=origin,
        dirs=dir_world,
        bodyexclude=base_id,
        self_bodies=self_bodies,
        synth_kwargs=kwargs,
    )


def _render_size(scene_yaml: str) -> tuple[int, int]:
    """The scene's rendered frame size, which is what `stride` subsamples.

    `SimSensorBridge` feeds `depth_synth_kwargs` the live rendered shape
    (`sim_sensor_bridge.py:3288`), so the ray count is a property of the scene,
    not of the manifest: the RoboCasa deploy scenes span 128² to 512², a 16x
    spread in rays. Read it off the same YAML the env was built from.
    """
    scene = yaml.safe_load(Path(scene_yaml).read_text())["scene"]
    return int(scene["observation_width"]), int(scene["observation_height"])


def _shipped_cast(cast: _SceneCast) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The SHIPPED synth, called as the deploy bridge calls it.

    This is the arm that makes these tests a gate rather than a demonstration:
    swap ``_cast_depth_rays`` onto ``mj_multiRay`` and the distances it returns
    stop matching the per-pixel reference below.
    """
    _, distances, hit, _, _, _ = _cast_depth_rays(
        model=cast.model,
        data=cast.data,
        stride=_STRIDE,
        exclude_body_id=cast.bodyexclude,
        exclude_body_ids=cast.self_bodies,
        **cast.synth_kwargs,
    )
    return distances, hit


def _hidden_geoms(cast: _SceneCast) -> NDArray[np.int64]:
    """The filter the shipped cast applies: intangible geometry + the robot."""
    return np.concatenate(
        (noncollidable_geom_ids(cast.model), _body_geom_ids(cast.model, cast.self_bodies))
    )


def _visible(cast: _SceneCast, groups: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Per-geom visibility under ``groups``, with MuJoCo's own group clamping.

    MuJoCo clamps a geom's group into ``[0, mjNGROUP-1]`` before testing it
    against the mask. Indexing a length-6 array with the raw value instead
    raises on a group of 7 and silently wraps a negative one — and RoboCasa
    ships third-party MJCF, so neither is hypothetical.
    """
    return groups[np.clip(np.asarray(cast.model.geom_group), 0, groups.size - 1)] != 0


def _per_pixel(
    model: Any,
    data: Any,
    origin: NDArray[np.float64],
    dirs: NDArray[np.float64],
    groups: NDArray[np.uint8],
    bodyexclude: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """A local copy of the shipped caster, used to recover per-ray geom ids.

    ``_cast_depth_rays`` returns distances but not geom ids, and naming the
    struck geom is what turns "the two casters differ" into "the batched one
    walks through a collidable countertop". Every test that uses this first
    pins it to the shipped path via :func:`_shipped_cast`, so it can never
    drift into being the thing under test.
    """
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
    scene_cast: _SceneCast,
) -> None:
    """Adjudicate the SHIPPED caster against geometry, not against MuJoCo.

    Two links in one chain, and both have to hold:

    1. ``_cast_depth_rays`` — the function the deploy bridge calls — returns
       the same distances as a per-pixel ``mj_ray`` reference. Swap it onto
       ``mj_multiRay`` and this fails on three of the four scenes.
    2. That reference is itself the true nearest collidable surface, by an
       analytic ray/box intersection sharing no code with MuJoCo's caster.

    Together they say the shipped cloud is correct, which is the claim the
    per-pixel cost is being paid for.
    """
    cast = scene_cast
    shipped_dist, shipped_hit = _shipped_cast(cast)

    with _transparent_geoms(cast.model, _hidden_geoms(cast)) as groups:
        geomids, distances = _per_pixel(
            cast.model, cast.data, cast.origin, cast.dirs, groups, cast.bodyexclude
        )
        visible_boxes = np.flatnonzero(
            (np.asarray(cast.model.geom_type) == mujoco.mjtGeom.mjGEOM_BOX)
            & _visible(cast, groups)
            & (np.asarray(cast.model.geom_bodyid) != cast.bodyexclude)
        ).astype(np.int64)

    # Link 1: the shipped path IS this reference. `_cast_depth_rays` runs the
    # same two passes with the same filters, so the distances must be equal
    # bit for bit, not merely close.
    assert np.array_equal(shipped_dist, distances), (
        f"{cast.scene_yaml}: `_cast_depth_rays` no longer returns what a "
        f"per-pixel `mj_ray` cast returns "
        f"({int(np.count_nonzero(shipped_dist != distances))} of "
        f"{distances.size} rays differ). If the caster was swapped onto "
        "`mj_multiRay`, #195 measured that swap walking through collidable "
        "countertops — see the module docstring."
    )

    # Link 2: adjudicate against geometry. Only rays the synth ACCEPTS matter
    # — nothing else reaches the published cloud — and only box returns, so
    # "nearest box" and "nearest geom" are the same question.
    box_rays = np.flatnonzero(shipped_hit & np.isin(geomids, visible_boxes))
    assert box_rays.size >= 100, f"{cast.scene_yaml}: too few box returns to adjudicate"
    sample = np.random.default_rng(0).choice(box_rays, size=100, replace=False)

    wrong = [
        (int(i), int(geomids[i]), float(distances[i]), gid, t)
        for i in sample
        for gid, t in [
            _nearest_box_hit(cast.model, cast.data, cast.origin, cast.dirs[i], visible_boxes)
        ]
        if gid != int(geomids[i]) or abs(t - float(distances[i])) > 1e-6
    ]
    assert not wrong, (
        f"{cast.scene_yaml}: the depth cast disagrees with an independent "
        f"ray/box intersection on {len(wrong)} of {sample.size} sampled rays "
        f"(first: {wrong[0]}). The shipped per-pixel `mj_ray` matches it "
        "exactly; `mj_multiRay` does not (see #195)."
    )


@pytest.mark.sim
def test_batched_mj_multiray_still_skips_collidable_geometry(scene_cast: _SceneCast) -> None:
    """#195's measurement: the premium #180 hoped to recover is not recoverable.

    Both casters see the identical filtered world, and only the rays the synth
    accepts are counted, so every number here is a difference in the cloud the
    world grid is actually built from. Every geom the batched cast fails to see
    is collidable — the intangible geometry #180 blamed is not in the cast any
    more — and every disagreement is a surface reported too far away, never too
    near and never lost outright.
    """
    cast = scene_cast
    _, accepted = _shipped_cast(cast)

    with _transparent_geoms(cast.model, _hidden_geoms(cast)) as groups:
        ref_ids, ref_dist = _per_pixel(
            cast.model, cast.data, cast.origin, cast.dirs, groups, cast.bodyexclude
        )
        new_ids, new_dist = _batched(
            cast.model, cast.data, cast.origin, cast.dirs, groups, cast.bodyexclude
        )

    disagree = accepted & (ref_ids != new_ids)
    skipped = np.unique(ref_ids[disagree])
    contype = np.asarray(cast.model.geom_contype)[skipped]
    conaffinity = np.asarray(cast.model.geom_conaffinity)[skipped]
    intangible = skipped[(contype == 0) & (conaffinity == 0)]
    both = accepted & (new_ids >= 0)
    delta = np.where(both, new_dist - ref_dist, 0.0)
    nearer = int(np.count_nonzero(delta < -1e-9))
    lost = int(np.count_nonzero(accepted & (new_ids < 0)))

    assert intangible.size == 0, (
        f"{cast.scene_yaml}: {intangible.size} of the {skipped.size} geoms the "
        "batched cast skips are non-collidable — #180's filter should already "
        "have removed them from both casts."
    )
    assert nearer == 0, (
        f"{cast.scene_yaml}: the batched cast reported a NEARER surface on "
        f"{nearer} rays, which broad-phase culling cannot explain — "
        "investigate before trusting either number."
    )
    # A lost return is the one outcome worse than a late one: the cloud says
    # "nothing here" rather than "something, further away", so OctoMap clears
    # the cell instead of moving it. #195 measured zero of them; keep it that
    # way, and never let this test's other numbers stand in for it.
    assert lost == 0, (
        f"{cast.scene_yaml}: the batched cast LOST {lost} returns the "
        "per-pixel cast accepted — a surface reported as free space outright, "
        "which is worse than the push-back #195 measured."
    )
    assert int(np.count_nonzero(disagree)) == cast.expected_disagreeing, (
        f"{cast.scene_yaml}: #195 measured the batched cast losing "
        f"{cast.expected_disagreeing} of the accepted rays here; it now loses "
        f"{int(np.count_nonzero(disagree))}. That is not licence to swap the "
        f"caster either way — the cull behaviour changed under mujoco "
        f"{mujoco.__version__}. Re-run #195's measurement on all four scenes, "
        "update docs/reference/world-map-fidelity.md, and re-open the issue."
    )
