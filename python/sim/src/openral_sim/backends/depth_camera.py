# SPDX-License-Identifier: Apache-2.0
"""Simulated depth camera → point cloud, via MuJoCo CPU ray-casting.

The 3-D analogue of
:func:`openral_sim.backends.robocasa.synthesize_laser_scan_2d`. Casts one
``mj_ray`` per (strided) pixel through a pinhole model anchored on
a named MJCF camera, and returns the hit points in the camera *optical*
frame (REP-103: ``+x`` right, ``+y`` down, ``+z`` forward — the ROS camera
convention).

Like the 2-D lidar synth this uses MuJoCo's analytic ray-caster, **not** a
GL renderer — so it needs no display / EGL context and runs deterministically
in CI. It is robot-agnostic: any camera declared in any robot's MJCF works,
which is what lets the deploy-sim HAL feed an ``octomap_server`` (and thus the
world-collision kernel check) from any robot, not just panda_mobile.

The returned cloud is the dense, bounded input perception lowers into an
OctoMap; the kernel never sees it directly ("perception proposes, the kernel
disposes").

Cost scales with rays x geoms, and the caller's ``stride`` is the lever.
Measured on a synthetic 1200-geom clutter scene (RoboCasa-kitchen scale) with a
256x256 camera: **one** cast costs ~60 ms at the deploy default ``stride=4``
(4096 rays) and ~240 ms at ``stride=2`` (16384 rays). The deploy-sim depth timer
runs on the single-threaded ``rclpy.spin`` at ``depth_publish_rate_hz`` —
default **10 Hz**, a 100 ms period (the manifest ``SensorSpec.rate_hz`` is not
read by the bridge) — so ``stride=4`` already spends ~60% of the period inside
the cast and ``stride=2`` cannot hold the rate at all.

Budget one cast per camera per frame: derive every output from that single
raster rather than calling both synths, which casts each ray twice for numbers
that are equal by construction (``openral_hal.sim_sensor_bridge`` publishes the
``PointCloud2`` via ``openral_hal.depth_cloud.points_from_depth_grid``).
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


@contextmanager
def _transparent_body_geoms(
    model: Any,  # reason: optional MuJoCo pybind type
    body_ids: frozenset[int] | None,
) -> Iterator[NDArray[np.uint8]]:
    """Temporarily hide selected bodies from MuJoCo rays, then restore them."""
    from openral_core.exceptions import ROSConfigError

    if not body_ids:
        yield _GEOMGROUP_ALL
        return
    geom_body_ids = np.asarray(model.geom_bodyid)
    transparent_geom_ids = np.flatnonzero(
        np.isin(geom_body_ids, np.fromiter(body_ids, dtype=np.int64))
    )
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
            "Depth self-filter cannot make bodies transparent: all six MuJoCo geom groups "
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

    # One `mj_ray` per pixel, NOT the batched `mj_multiRay`. `mj_multiRay`
    # culls whole bodies against the bounding sphere of the body's BVH root
    # AABB (`mju_singleRay`), and MuJoCo's compiler builds that BVH from
    # COLLISION geoms only — so a `contype=0 conaffinity=0` visual geom lying
    # outside its body's collision extent is silently skipped and the ray
    # reports whatever solid sits behind it. On RoboCasa that hides counter
    # tops (world z 0.920) behind the cabinet carcasses under them (0.890):
    # a 30 mm systematic underestimate, in the unsafe direction, on the cloud
    # the world-collision voxel grid is built from. `mj_ray` runs the plain
    # linear scan over every visible geom and has no such cull, matching what
    # the GL depth renderer draws. It costs more than the batched call — how
    # much depends entirely on how often that body cull fires, i.e. on the
    # scene (~1.9x on the 1200-geom clutter scene the module docstring's budget
    # is measured on), so size the budget from a measurement, not a multiplier.
    # The depth stream is a strided, rate-limited sim sensor, and a wrong
    # surface is worse than a slower one.
    geomid_out = np.zeros(1, dtype=np.int32)
    bodyexclude = -1 if exclude_body_id is None else int(exclude_body_id)

    if exclude_body_ids:
        # First pass, everything visible: which rays land on a body we are
        # about to make transparent? Those are the ones whose second-pass
        # result is "the world behind the payload" rather than a real return,
        # so they clear their ray in OctoMap instead of marking a cell.
        initial_geomids = np.full(n_rays, -1, dtype=np.int32)
        initial_distances = np.full(n_rays, -1.0, dtype=np.float64)
        for i in range(n_rays):
            geomid_out[0] = -1
            initial_distances[i] = mujoco.mj_ray(
                model, data, origin, dir_world[i], _GEOMGROUP_ALL, 1, bodyexclude, geomid_out
            )
            initial_geomids[i] = geomid_out[0]
        safe_geom = np.where(initial_geomids >= 0, initial_geomids, 0)
        initial_bodies = np.asarray(model.geom_bodyid)[safe_geom]
        transparent_hits = (
            (initial_geomids >= 0)
            & (initial_distances <= max_range_m)
            & np.isin(initial_bodies, np.fromiter(exclude_body_ids, dtype=np.int64))
        )

    with _transparent_body_geoms(model, exclude_body_ids) as geom_groups:
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

    The image counterpart of :func:`synthesize_depth_pointcloud`, sharing the
    same pinhole ray-cast (:func:`_cast_depth_rays`) but keeping **every** pixel
    — a dense raster nvblox's projective depth integrator consumes
    directly (it rejects the sparse, hit-only cloud, whose unorganised layout
    matches no camera/lidar intrinsic model). Each pixel holds the *perpendicular
    optical-Z* depth in metres (``range · ẑ`` — the ROS depth-image convention,
    **not** the Euclidean range), with ``0.0`` where the ray missed, fell out of
    ``[min_range_m, max_range_m]``, or struck a self-filtered body (the standard
    "no measurement" sentinel nvblox skips).

    The raster is at the **strided** resolution: shape ``(ceil(height / stride),
    ceil(width / stride))``. The matching ``CameraInfo`` must scale the
    intrinsics by ``1 / stride`` (see
    ``openral_hal.depth_cloud.camera_info_from_intrinsics``).

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
    dir_opt, distances, hit, _clearing, n_cols, n_rows = _cast_depth_rays(
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
    return depth.reshape(n_rows, n_cols)
