# SPDX-License-Identifier: Apache-2.0
"""Reusable, robot-agnostic depth-camera → PointCloud2 plumbing (ADR-0030).

A deploy-sim HAL node turns each depth ``SensorSpec`` on its robot into a
``sensor_msgs/PointCloud2`` that ``octomap_server`` lifts into the 3-D
OctoMap feeding the safety kernel's world-collision check. This module holds
the pieces shared across robots so a node only has to wire publishers/timers:

* :func:`is_depth_sensor` / :func:`mjcf_camera_name` / :func:`depth_synth_kwargs`
  — pure SensorSpec adapters (no ROS / MuJoCo import).
* :func:`camera_optical_tf_to_base` — the live camera-optical-frame → base
  transform, from the MuJoCo camera/body poses (so TF can place the cloud).
* :func:`pointcloud2_from_points_xyz` — pack an ``(N, 3)`` array into a
  ``sensor_msgs/PointCloud2`` (``sensor_msgs`` imported lazily).

The synth itself lives in
:func:`openral_sim.backends.depth_camera.synthesize_depth_pointcloud`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# REP-103 optical (x right, y down, z forward) expressed in the MuJoCo
# camera frame (x right, y up, z back): flip y and z.
_OPTICAL_IN_MJCAM = np.diag([1.0, -1.0, -1.0])


def is_depth_sensor(spec: Any) -> bool:
    """True when ``spec`` is a depth/point-cloud camera with intrinsics.

    Intrinsics are required to back-project pixels, so a depth ``SensorSpec``
    without them is not usable by the synth and is skipped.
    """
    return spec.modality in ("depth", "point_cloud") and spec.intrinsics is not None


def mjcf_camera_name(spec: Any) -> str:
    """Resolve the MJCF ``<camera>`` name backing a depth ``SensorSpec``.

    Prefers ``spec.metadata['mjcf_camera']`` (the sim camera name, which can
    differ from the ROS-facing sensor name), falling back to ``spec.name``.
    """
    meta = getattr(spec, "metadata", {}) or {}
    name = meta.get("mjcf_camera")
    if isinstance(name, str) and name:
        return name
    return str(spec.name)


def depth_synth_kwargs(
    spec: Any,
    *,
    max_range_default: float,
    render_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Map a depth ``SensorSpec`` to ``synthesize_depth_pointcloud`` kwargs.

    Pulls width/height/fx/fy/cx/cy from the pinhole intrinsics and the range
    gates from ``range_min_m`` / ``range_max_m`` (falling back to
    ``max_range_default`` when ``range_max_m`` is unset).

    When ``render_size`` is given (the scene's
    ``observation_width``/``height``), the intrinsics are first rescaled to that
    resolution via :func:`openral_core.scale_intrinsics_to`. The depth synth
    ray-casts a pixel grid sized by ``width``/``height`` through the
    ``(u-cx)/fx, (v-cy)/fy`` pinhole model, so for the back-projected cloud to
    match the RGB the env rendered (same MuJoCo camera, possibly at a non-default
    resolution), the focal length and principal point must track the render
    resolution. Leaving ``render_size`` ``None`` keeps the manifest's nominal
    intrinsics unchanged.
    """
    from openral_core import scale_intrinsics_to

    intr = spec.intrinsics
    if render_size is not None:
        intr = scale_intrinsics_to(intr, render_size[0], render_size[1])
    return {
        "camera_name": mjcf_camera_name(spec),
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "min_range_m": float(spec.range_min_m) if spec.range_min_m is not None else 0.0,
        "max_range_m": (
            float(spec.range_max_m) if spec.range_max_m is not None else float(max_range_default)
        ),
    }


def robot_self_body_ids(model: Any, sim_joint_names: Any) -> frozenset[int]:
    """Resolve the robot's own MJCF body ids, for depth self-filtering.

    A base-mounted depth camera sees the arm, so without this the robot is
    voxelised into its own world map and the kernel's world-collision check
    flags the arm against itself. Returns every body whose name shares a prefix
    (first ``_``-delimited token) with one of the robot's ``sim_joint_name``s —
    e.g. ``mobilebase0`` / ``robot0`` / ``gripper0`` in a robosuite/robocasa
    scene — so the synth can drop hits on them.
    """
    import mujoco  # reason: defer optional sim dep

    prefixes = {n.split("_", 1)[0] for n in sim_joint_names if n}
    if not prefixes:
        return frozenset()
    out: set[int] = set()
    for i in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.split("_", 1)[0] in prefixes:
            out.add(i)
    return frozenset(out)


def camera_optical_tf_to_base(
    *,
    model: Any,
    data: Any,
    camera_name: str,
    base_body_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Live transform of a camera's optical frame, expressed in the base body.

    Returns ``(translation_xyz, quaternion_xyzw)`` mapping the camera optical
    frame (REP-103) into ``base_body_name``'s frame, so a node can broadcast
    ``base_frame -> <camera>_optical_frame`` from the current MuJoCo poses and
    octomap_server can resolve the published cloud.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData``.
        camera_name: MJCF camera name.
        base_body_name: MJCF body whose frame the robot's ``base_frame`` tracks.

    Raises:
        ROSConfigError: camera or base body absent from the model.
    """
    import mujoco  # reason: defer optional sim dep
    from openral_core.exceptions import ROSConfigError  # reason: defer core import

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ROSConfigError(f"camera {camera_name!r} not found in the MuJoCo model.")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_id < 0:
        raise ROSConfigError(f"base body {base_body_name!r} not found in the MuJoCo model.")

    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    rot_world_cam = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    base_pos = np.asarray(data.xpos[base_id], dtype=np.float64)
    rot_world_base = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)

    rot_world_opt = rot_world_cam @ _OPTICAL_IN_MJCAM
    rot_base_world = rot_world_base.T
    translation = rot_base_world @ (cam_pos - base_pos)
    rot_base_opt = np.ascontiguousarray(rot_base_world @ rot_world_opt)

    quat_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, rot_base_opt.ravel())
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (qx, qy, qz, qw),
    )


def pointcloud2_from_points_xyz(
    points: NDArray[np.float32],
    *,
    frame_id: str,
    stamp: Any = None,
) -> Any:
    """Pack an ``(N, 3)`` float32 array into a ``sensor_msgs/PointCloud2``.

    The cloud is an unordered list (``height=1``, ``width=N``) of XYZ float32
    points — the standard layout octomap_server's ``cloud_in`` expects.

    Args:
        points: ``(N, 3)`` float32 array of XYZ in the ``frame_id`` frame.
        frame_id: tf2 frame the points live in (the camera optical frame).
        stamp: ``builtin_interfaces/Time`` for the header; ``None`` leaves the
            default (zero) stamp — callers normally pass ``node.get_clock().now()``.
    """
    from sensor_msgs.msg import PointCloud2, PointField  # reason: defer ROS dep

    pts = np.ascontiguousarray(points, dtype="<f4")
    n = int(pts.shape[0])

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.data = pts.reshape(-1).tobytes()
    msg.is_dense = True
    return msg
