# SPDX-License-Identifier: Apache-2.0
"""Simulated depth camera → point cloud, via MuJoCo CPU ray-casting.

ADR-0030 — the 3-D analogue of
:func:`openral_sim.backends.robocasa.synthesize_laser_scan_2d`. Casts one
``mj_multiRay`` ray per (strided) pixel through a pinhole model anchored on
a named MJCF camera, and returns the hit points in the camera *optical*
frame (REP-103: ``+x`` right, ``+y`` down, ``+z`` forward — the ROS camera
convention).

Like the 2-D lidar synth this uses MuJoCo's analytic ray-caster, **not** a
GL renderer — so it needs no display / EGL context and runs deterministically
in CI. It is robot-agnostic: any camera declared in any robot's MJCF works,
which is what lets the deploy-sim HAL feed an ``octomap_server`` (and thus the
ADR-0030 world-collision kernel check) from any robot, not just panda_mobile.

The returned cloud is the dense, bounded input perception lowers into an
OctoMap; the kernel never sees it directly ("perception proposes, the kernel
disposes").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# A ray's "no hit" sentinel from mj_multiRay is a negative distance; a hit
# also reports the struck geom id (>= 0). We require both to accept a point.

_GEOMGROUP_ALL = np.ones(6, dtype=np.uint8)


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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], int, int]:
    """Shared pinhole MuJoCo ray-cast behind the cloud + image synths.

    Casts one ray per (strided) pixel ``(u, v)`` — ``u in range(0, width,
    stride)``, ``v in range(0, height, stride)``, row-major (``v`` outer) — and
    returns the per-ray geometry both public synths derive their output from.

    Returns:
        ``(dir_opt, distances, hit, n_cols, n_rows)``:

        * ``dir_opt`` — ``(R, 3)`` float64 unit ray directions in the camera
          optical frame (REP-103).
        * ``distances`` — ``(R,)`` float64 Euclidean ranges from ``mj_multiRay``
          (``-1`` where no geom was struck; out-of-range values are masked by
          ``hit``, not cleared here).
        * ``hit`` — ``(R,)`` bool accept mask (genuine hit, in ``[min, max]``
          range, not a self-filtered body).
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
    # mj_multiRay distances come back as Euclidean ranges.
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

    # Positional pybind signature: (m, d, pnt, vec, geomgroup, flg_static,
    # bodyexclude, geomid, dist, normal, nray, cutoff). normal=None (we only
    # need ranges).
    mujoco.mj_multiRay(
        model,
        data,
        origin,
        dir_world.ravel(),
        _GEOMGROUP_ALL,
        1,
        -1 if exclude_body_id is None else int(exclude_body_id),
        geomids,
        distances,
        None,
        n_rays,
        float(max_range_m),
    )

    # Accept only genuine hits within [min_range, max_range]. mj_multiRay's
    # cutoff is a bounding-sphere prefilter, so a geom that passes it can
    # still report a distance > max_range — clamp explicitly (same caveat
    # the 2-D lidar synth documents).
    hit = (geomids >= 0) & (distances >= min_range_m) & (distances <= max_range_m)
    if exclude_body_ids:
        # Drop rays that struck one of the robot's own bodies (self-filter), so
        # the robot isn't voxelised into its own world map. geom_bodyid[-1] is
        # invalid, so index only the genuine hits.
        safe_geom = np.where(geomids >= 0, geomids, 0)
        hit_body = np.asarray(model.geom_bodyid)[safe_geom]
        excluded = np.isin(hit_body, np.fromiter(exclude_body_ids, dtype=np.int64))
        hit &= ~excluded
    return dir_opt, distances, hit, int(us.size), int(vs.size)


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
        exclude_body_id: Single MuJoCo body id passed to ``mj_multiRay``'s
            ``bodyexclude`` (so rays don't immediately strike the camera's own
            mount body at range ~0); ``None`` excludes nothing.
        exclude_body_ids: Body ids whose hits are dropped after casting — the
            robot's own links/gripper, so a base-mounted depth camera that sees
            the arm does NOT voxelise the robot into the world map (which would
            make the kernel's world-collision check flag the arm against
            itself). ``None``/empty drops nothing.

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
    dir_opt, distances, hit, _, _ = _cast_depth_rays(
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
    if not np.any(hit):
        return np.zeros((0, 3), dtype=np.float32)

    points = distances[hit, None] * dir_opt[hit]
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
    — a dense raster nvblox's projective depth integrator (ADR-0064) consumes
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
        exclude_body_id: ``mj_multiRay`` ``bodyexclude`` (camera's own mount).
        exclude_body_ids: Body ids whose hits read ``0.0`` (robot self-filter).

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
    dir_opt, distances, hit, n_cols, n_rows = _cast_depth_rays(
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
