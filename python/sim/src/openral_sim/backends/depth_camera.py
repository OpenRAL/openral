# SPDX-License-Identifier: Apache-2.0
"""Simulated depth camera → point cloud, via MuJoCo CPU ray-casting.

3-D analogue of ``openral_sim.backends.robocasa.synthesize_laser_scan_2d``.
Casts one ``mj_ray`` per (strided) pixel through a pinhole model anchored on a
named MJCF camera; returns hits in the camera *optical* frame (REP-103: +x
right, +y down, +z forward). Uses MuJoCo's analytic ray-caster, not a GL
renderer, so no display/EGL and deterministic in CI; robot-agnostic (any MJCF
camera works), which lets deploy-sim feed ``octomap_server`` from any robot,
not just panda_mobile.

Cost scales with rays x geoms; ``stride`` is the lever. 1200-geom
RoboCasa-kitchen-scale clutter scene, 256x256 camera: ~60 ms/cast at
stride=4 (4096 rays), ~240 ms at stride=2 (16384 rays). Deploy-sim depth timer
runs on single-threaded ``rclpy.spin`` at ``depth_publish_rate_hz`` (default
10 Hz, 100 ms period; manifest ``SensorSpec.rate_hz`` is not read). ``stride``
subsamples the *rendered*, not nominal, resolution
(``openral_hal.depth_cloud.depth_synth_kwargs`` rescales intrinsics to the
scene's ``observation_width``/``height``) — RoboCasa scenes vary 16x: the five
task scenes (baguette, sink_cup, fridge_drawer, drawer_utensil, deliver_straw)
render 512x512 (stride=4 -> 16384 rays); ``robocasa_vslam*`` render 256x256
(4096 rays); ``robocasa_pnp``/``robocasa_navigate`` render 128x128 (1024
rays). Measured on the four validation-matrix scenes at those settings
(q-laptop, CPU): 83-129 ms/pass, up to the whole 100 ms budget; a payload
frame casts the bundle twice.

Budget one cast per camera per frame — derive cloud and image from a single
raster (``openral_hal.sim_sensor_bridge`` publishes ``PointCloud2`` via
``openral_hal.depth_cloud.points_from_depth_grid``) rather than double-casting.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

# A ray's "no hit" sentinel from mj_ray is a negative distance; a hit
# also reports the struck geom id (>= 0). We require both to accept a point.

_GEOMGROUP_ALL = np.ones(6, dtype=np.uint8)


def noncollidable_geom_ids(model: Any) -> NDArray[np.int64]:
    """Geom ids MuJoCo can never collide with — decoration, markers, sites.

    A geom with neither ``contype`` nor ``conaffinity`` is excluded from every
    contact pair, so no physics touches it, yet it is still *visible*: the
    depth synth would strike it and the safety kernel could E-stop on a cell
    no body occupies (#174), so the ground-truth probe is required (#149) to
    ignore it. MuJoCo's own collision predicate, not a scene convention —
    robosuite/RoboCasa's group-0/group-1 split is a rendering convention only.
    """
    contype = np.asarray(model.geom_contype, dtype=np.int64)
    conaffinity = np.asarray(model.geom_conaffinity, dtype=np.int64)
    return np.flatnonzero((contype == 0) & (conaffinity == 0)).astype(np.int64)


def _body_geom_ids(model: Any, body_ids: frozenset[int] | None) -> NDArray[np.int64]:
    """Every geom id belonging to ``body_ids`` (empty when nothing is selected)."""
    if not body_ids:
        return np.empty(0, dtype=np.int64)
    geom_body_ids = np.asarray(model.geom_bodyid)
    return np.flatnonzero(np.isin(geom_body_ids, np.fromiter(body_ids, dtype=np.int64))).astype(
        np.int64
    )


@contextmanager
def _transparent_geoms(
    model: Any,  # reason: optional MuJoCo pybind type
    geom_ids: NDArray[np.int64],
) -> Iterator[NDArray[np.uint8]]:
    """Temporarily hide selected geoms from MuJoCo rays, then restore them."""
    from openral_core.exceptions import ROSConfigError

    transparent_geom_ids = np.unique(geom_ids)
    if transparent_geom_ids.size == 0:
        yield _GEOMGROUP_ALL
        return
    geom_groups = np.asarray(model.geom_group)
    opaque_groups = set(
        int(group)
        for group in geom_groups[
            ~np.isin(
                np.arange(int(model.ngeom)),
                transparent_geom_ids,
            )
        ]
    )
    hidden_group = next((group for group in range(6) if group not in opaque_groups), None)
    if hidden_group is None:
        raise ROSConfigError(
            "Depth ray filter cannot make geoms transparent: all six MuJoCo geom groups "
            "are used by non-filtered geometry."
        )
    original_groups = geom_groups[transparent_geom_ids].copy()
    ray_groups = _GEOMGROUP_ALL.copy()
    ray_groups[hidden_group] = 0
    try:
        model.geom_group[transparent_geom_ids] = hidden_group
        yield ray_groups
    finally:
        model.geom_group[transparent_geom_ids] = original_groups


def _cast_depth_rays(
    *,
    model: Any,
    data: Any,
    camera_name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_range_m: float,
    min_range_m: float,
    stride: int,
    exclude_body_id: int | None,
    exclude_body_ids: frozenset[int] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    int,
    int,
]:
    """Shared pinhole MuJoCo ray-cast behind the cloud + image synths.

    Casts one ray per (strided) pixel ``(u, v)`` — ``u in range(0, width,
    stride)``, ``v in range(0, height, stride)``, row-major (``v`` outer) — and
    returns the per-ray geometry both public synths derive their output from.

    Returns:
        ``(dir_opt, distances, hit, clearing, n_cols, n_rows)``:

        * ``dir_opt`` — ``(R, 3)`` float64 unit ray directions in the camera
          optical frame (REP-103).
        * ``distances`` — ``(R,)`` float64 Euclidean ranges from ``mj_ray``
          (``-1`` where no geom was struck; out-of-range values are masked by
          ``hit``, not cleared here).
        * ``hit`` — ``(R,)`` bool accept mask (genuine hit, in ``[min, max]``
          range, not a self-filtered body).
        * ``clearing`` — rays that originally struck a transparent body and
          found no farther surface; point clouds emit a max-range endpoint so
          OctoMap clears the ray without adding an occupied cell.
        * ``n_cols`` / ``n_rows`` — the strided pixel-grid width / height, so a
          dense raster reshapes as ``ray_index = row * n_cols + col``.

    Raises:
        ROSConfigError: ``camera_name`` is not a camera in ``model``.
    """
    import mujoco  # reason: defer optional sim dep
    from openral_core.exceptions import ROSConfigError  # reason: defer core import

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ROSConfigError(
            f"camera {camera_name!r} not found in the MuJoCo model "
            "(declare it in the robot's MJCF or fix the SensorSpec name)."
        )

    # Pixel grid (row-major: v outer, u inner) so ray i corresponds to
    # the i-th raster pixel.
    us = np.arange(0, width, stride, dtype=np.float64)
    vs = np.arange(0, height, stride, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)  # each (len_vs, len_us)
    u_flat = grid_u.ravel()
    v_flat = grid_v.ravel()

    # Pinhole rays in the optical frame, normalised to unit length so the
    # mj_ray distances come back as Euclidean ranges.
    dir_opt = np.empty((u_flat.size, 3), dtype=np.float64)
    dir_opt[:, 0] = (u_flat - cx) / fx
    dir_opt[:, 1] = (v_flat - cy) / fy
    dir_opt[:, 2] = 1.0
    dir_opt /= np.linalg.norm(dir_opt, axis=1, keepdims=True)

    # Optical (x right, y down, z forward) → MuJoCo camera frame
    # (x right, y up, z back): flip y and z.
    dir_cam = dir_opt * np.array([1.0, -1.0, -1.0], dtype=np.float64)

    # Rotate into world: cam_xmat is the row-major world<-camera rotation.
    rot = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    dir_world = np.ascontiguousarray((rot @ dir_cam.T).T)

    origin = np.ascontiguousarray(np.asarray(data.cam_xpos[cam_id], dtype=np.float64))

    n_rays = dir_world.shape[0]
    geomids = np.full(n_rays, -1, dtype=np.int32)
    distances = np.full(n_rays, -1.0, dtype=np.float64)
    transparent_hits: NDArray[np.bool_] = np.zeros(n_rays, dtype=np.bool_)

    # One `mj_ray` per pixel, NOT batched `mj_multiRay`: multiRay's broad-phase
    # culling drops real surfaces on real scenes; `mj_ray` linear-scans every
    # visible geom, matching the GL depth renderer.
    #
    # #180 conjectured the cull only mis-skips `contype=0 conaffinity=0`
    # decoration (filtered below as `intangible`) — filtering it out would
    # make multiRay safe. #195 measured FALSE on all four validation-matrix
    # scenes at deploy stride, both casters over the identical filtered world
    # (tests/sim/safety/test_depth_multiray_equivalence_robocasa.py; table in
    # docs/reference/world-map-fidelity.md): multiRay disagrees on 4113/65536
    # rays, every skipped geom COLLIDABLE (baguette scene: walks through
    # `counter_1_left_group_top_left_0`/`_1`, contype=1/conaffinity=1, on
    # 3815 rays). Every disagreement is unsafe — up to 480 mm too far, never
    # too near (an independent ray/box check agrees with `mj_ray` 200/200,
    # `mj_multiRay` 0/200). Mechanism unestablished: #111's "BVH from
    # collision geoms only" theory fails — skipped geoms ARE collidable,
    # multiRay still returns 184 rays on `counter_1_left_group_main`; #195
    # could not reproduce it in a hand-written MJCF. Premium 4.2-6.2x here vs
    # #111's ~1.9x on synthetic clutter — budget from measurement, not a
    # multiplier. The depth stream is a
    # strided, rate-limited sim sensor, and a wrong surface is worse than a
    # slower one.
    geomid_out = np.zeros(1, dtype=np.int32)
    bodyexclude = -1 if exclude_body_id is None else int(exclude_body_id)

    # Intangible geometry filtered from EVERY pass: #174 measured 713/1714
    # geoms in the RoboCasa fridge scene as non-collidable decoration, backing
    # 18.8% of one live map's occupied cells — obstacles no policy trains or
    # evaluates against, phantom E-stops the ground-truth probe must call
    # unbacked. Filtering can only move a return farther along its ray or
    # remove it — a collidable surface stays hittable, so no touchable
    # geometry leaves the map. On hardware a visible object is solid; this
    # makes sim match that.
    intangible = noncollidable_geom_ids(model)

    if exclude_body_ids:
        # First pass, every TOUCHABLE thing visible: which rays land on a body
        # we are about to make transparent? Those are the ones whose
        # second-pass result is "the world behind the payload" rather than a
        # real return, so they clear their ray in OctoMap instead of marking a
        # cell. Both passes must see the same world, or a ray blocked by
        # decoration here would go unrecognised as a payload return there.
        initial_geomids = np.full(n_rays, -1, dtype=np.int32)
        initial_distances = np.full(n_rays, -1.0, dtype=np.float64)
        with _transparent_geoms(model, intangible) as solid_groups:
            for i in range(n_rays):
                geomid_out[0] = -1
                initial_distances[i] = mujoco.mj_ray(
                    model, data, origin, dir_world[i], solid_groups, 1, bodyexclude, geomid_out
                )
                initial_geomids[i] = geomid_out[0]
        safe_geom = np.where(initial_geomids >= 0, initial_geomids, 0)
        initial_bodies = np.asarray(model.geom_bodyid)[safe_geom]
        transparent_hits = (
            (initial_geomids >= 0)
            & (initial_distances <= max_range_m)
            & np.isin(initial_bodies, np.fromiter(exclude_body_ids, dtype=np.int64))
        )

    hidden = np.concatenate((intangible, _body_geom_ids(model, exclude_body_ids)))
    with _transparent_geoms(model, hidden) as geom_groups:
        for i in range(n_rays):
            geomid_out[0] = -1
            distances[i] = mujoco.mj_ray(
                model, data, origin, dir_world[i], geom_groups, 1, bodyexclude, geomid_out
            )
            geomids[i] = geomid_out[0]

    # Accept only genuine hits within [min_range, max_range]. `mj_ray` has no
    # range cutoff at all — it always reports the true nearest visible
    # surface — so the max-range clamp is this mask, not a caster argument.
    hit = (geomids >= 0) & (distances >= min_range_m) & (distances <= max_range_m)
    clearing = transparent_hits & ~hit
    return dir_opt, distances, hit, clearing, int(us.size), int(vs.size)


def synthesize_depth_pointcloud(
    *,
    model: Any,
    data: Any,
    camera_name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_range_m: float,
    min_range_m: float = 0.0,
    stride: int = 1,
    exclude_body_id: int | None = None,
    exclude_body_ids: frozenset[int] | None = None,
) -> NDArray[np.float32]:
    """Ray-cast a depth point cloud from a named MJCF camera.

    One ray per pixel ``(u, v)`` for ``u in range(0, width, stride)`` and
    ``v in range(0, height, stride)`` (row-major: ``v`` outer, ``u`` inner),
    cast through the pinhole model ``((u - cx) / fx, (v - cy) / fy, 1)`` and
    anchored on the camera's live world pose (``data.cam_xpos`` /
    ``data.cam_xmat``). Hit points are returned in the camera optical frame,
    so the caller publishes them with ``frame_id`` = the camera's optical
    frame and lets TF place them in the world.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (caller must have stepped / forwarded it).
        camera_name: Name of the ``<camera>`` in the MJCF.
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Pinhole focal length in x (pixels).
        fy: Pinhole focal length in y (pixels).
        cx: Pinhole principal point x (pixels).
        cy: Pinhole principal point y (pixels).
        max_range_m: Rays returning farther than this (or no hit) are dropped.
        min_range_m: Rays returning nearer than this are dropped (e.g. to
            reject the robot's own gripper in view).
        stride: Pixel subsample step. ``stride=2`` casts a quarter of the rays.
        exclude_body_id: Single MuJoCo body id passed to ``mj_ray``'s
            ``bodyexclude`` (so rays don't immediately strike the camera's own
            mount body at range ~0); ``None`` excludes nothing.
        exclude_body_ids: Body ids made transparent to the ray-cast — the
            robot's own links and acknowledged attached payloads. Rays continue
            to the next world surface so OctoMap receives clearing rays instead
            of permanent holes where a carried object used to be.

    Returns:
        ``(N, 3)`` float32 array of hit points in the camera optical frame
        (REP-103). ``N`` is the number of in-range hits (``<=`` the cast ray
        count); an empty ``(0, 3)`` array when nothing is in range.

    Raises:
        ROSConfigError: ``camera_name`` is not a camera in ``model``.

    Example:
        >>> # points = synthesize_depth_pointcloud(
        >>> #     model=m, data=d, camera_name="head_depth",
        >>> #     width=64, height=48, fx=40, fy=40, cx=32, cy=24,
        >>> #     max_range_m=5.0, stride=2)
    """
    dir_opt, distances, hit, clearing, _, _ = _cast_depth_rays(
        model=model,
        data=data,
        camera_name=camera_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
        stride=stride,
        exclude_body_id=exclude_body_id,
        exclude_body_ids=exclude_body_ids,
    )
    accepted = hit | clearing
    if not np.any(accepted):
        return np.zeros((0, 3), dtype=np.float32)

    ranges = distances.copy()
    ranges[clearing] = max_range_m
    points = ranges[accepted, None] * dir_opt[accepted]
    return points.astype(np.float32)


def synthesize_depth_frame(
    *,
    model: Any,
    data: Any,
    camera_name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_range_m: float,
    min_range_m: float = 0.0,
    stride: int = 1,
    exclude_body_id: int | None = None,
    exclude_body_ids: frozenset[int] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """One ray-cast, both of its products: the depth raster **and** its clearing mask.

    A pixel whose only return is a self-filtered body (robot's own link, an
    acknowledged payload) has no depth (``0.0``, else nvblox integrates a
    false surface) but the ray is free to ``max_range_m``, which OctoMap needs
    to clear cells the robot occludes — collapsing both onto ``0.0`` would
    make the robot's silhouette write-only, leaving stale voxels forever. The
    raster keeps the sensor's ``0.0 = no measurement`` contract; the mask
    feeds ``openral_hal.depth_cloud.points_from_depth_grid``, which
    ``synthesize_depth_pointcloud`` already emits as max-range endpoints.
    Still one ``mj_ray`` per pixel per frame.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (caller must have stepped / forwarded it).
        camera_name: Name of the ``<camera>`` in the MJCF.
        width: Image width in pixels (full, pre-stride).
        height: Image height in pixels (full, pre-stride).
        fx: Pinhole focal length in x (pixels).
        fy: Pinhole focal length in y (pixels).
        cx: Pinhole principal point x (pixels).
        cy: Pinhole principal point y (pixels).
        max_range_m: Rays returning farther than this (or no hit) read ``0.0``.
        min_range_m: Rays returning nearer than this read ``0.0``.
        stride: Pixel subsample step.
        exclude_body_id: ``mj_ray`` ``bodyexclude`` (camera's own mount).
        exclude_body_ids: Body ids made transparent to the cast.

    Returns:
        ``(depth, clearing)`` — both ``(n_rows, n_cols)``. ``depth`` is the
        float32 optical-Z raster in metres (``0.0`` = no measurement);
        ``clearing`` is the bool mask of pixels whose only return was a
        self-filtered body and which found no farther surface, i.e. the rays
        OctoMap should clear to ``max_range_m``. The two are disjoint by
        construction.

    Raises:
        ROSConfigError: ``camera_name`` is not a camera in ``model``.

    Example:
        >>> # depth, clearing = synthesize_depth_frame(
        >>> #     model=m, data=d, camera_name="front_depth",
        >>> #     width=128, height=128, fx=92, fy=92, cx=64, cy=64,
        >>> #     max_range_m=5.0, exclude_body_ids=robot_bodies)
    """
    dir_opt, distances, hit, clearing, n_cols, n_rows = _cast_depth_rays(
        model=model,
        data=data,
        camera_name=camera_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
        stride=stride,
        exclude_body_id=exclude_body_id,
        exclude_body_ids=exclude_body_ids,
    )
    # Perpendicular optical-Z = Euclidean range · ẑ. dir_opt is unit-norm, so
    # dir_opt[:, 2] is cos(angle off the optical axis): exactly the z-component
    # of the back-projected point (``points[:, 2]`` in the cloud synth).
    depth = np.zeros(dir_opt.shape[0], dtype=np.float32)
    depth[hit] = (distances[hit] * dir_opt[hit, 2]).astype(np.float32)
    return depth.reshape(n_rows, n_cols), clearing.reshape(n_rows, n_cols)


def synthesize_depth_image(
    *,
    model: Any,
    data: Any,
    camera_name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_range_m: float,
    min_range_m: float = 0.0,
    stride: int = 1,
    exclude_body_id: int | None = None,
    exclude_body_ids: frozenset[int] | None = None,
) -> NDArray[np.float32]:
    """Ray-cast a dense ``32FC1`` depth image from a named MJCF camera.

    Image counterpart of ``synthesize_depth_pointcloud``, sharing the same
    ray-cast (``_cast_depth_rays``) but keeping every pixel — nvblox's
    projective integrator needs a dense raster, not the sparse hit-only cloud.
    Each pixel is *perpendicular optical-Z* depth in metres (``range · ẑ``,
    the ROS depth-image convention, not Euclidean range); ``0.0`` = no
    measurement (miss, out of ``[min_range_m, max_range_m]``, or self-filtered
    body).

    **Do not build an OctoMap cloud from this alone** — ``0.0`` can't
    distinguish "no return" from "free to ``max_range_m``" (self-filtered
    body), which OctoMap needs to clear occluded cells; use
    ``synthesize_depth_frame`` for both raster and clearing mask.

    Raster is at the **strided** resolution: ``(ceil(height/stride),
    ceil(width/stride))``; matching ``CameraInfo`` must scale intrinsics by
    ``1/stride`` (``openral_hal.depth_cloud.camera_info_from_intrinsics``).

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (caller must have stepped / forwarded it).
        camera_name: Name of the ``<camera>`` in the MJCF.
        width: Image width in pixels (full, pre-stride).
        height: Image height in pixels (full, pre-stride).
        fx: Pinhole focal length in x (pixels).
        fy: Pinhole focal length in y (pixels).
        cx: Pinhole principal point x (pixels).
        cy: Pinhole principal point y (pixels).
        max_range_m: Rays returning farther than this (or no hit) read ``0.0``.
        min_range_m: Rays returning nearer than this read ``0.0``.
        stride: Pixel subsample step. ``stride=2`` rasterises a quarter of the
            pixels (the ``CameraInfo`` intrinsics scale to match).
        exclude_body_id: ``mj_ray`` ``bodyexclude`` (camera's own mount).
        exclude_body_ids: Body ids made transparent so the next world surface
            supplies depth (robot/attached-object self-filter).

    Returns:
        ``(n_rows, n_cols)`` float32 depth raster in metres (optical-Z), ``0.0``
        = no measurement. All-zero when nothing is in range.

    Raises:
        ROSConfigError: ``camera_name`` is not a camera in ``model``.

    Example:
        >>> # depth = synthesize_depth_image(
        >>> #     model=m, data=d, camera_name="front_depth",
        >>> #     width=128, height=128, fx=92, fy=92, cx=64, cy=64,
        >>> #     max_range_m=8.0)  # -> (128, 128) float32, metres
    """
    depth, _clearing = synthesize_depth_frame(
        model=model,
        data=data,
        camera_name=camera_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
        stride=stride,
        exclude_body_id=exclude_body_id,
        exclude_body_ids=exclude_body_ids,
    )
    return depth
