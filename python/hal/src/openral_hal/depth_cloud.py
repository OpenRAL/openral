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


def resolve_base_body_name(model: Any, *, description: Any = None) -> str | None:
    """Resolve the MJCF body backing a robot's ``base_frame``, or ``None``.

    Robosuite/RoboCasa scenes name the base body after the first base joint's
    prefix with a ``_base`` tail (``mobilebase0_base`` under a composed
    kitchen), so when a ``RobotDescription`` is given we derive that candidate
    first (mirroring the depth/TF base resolution). We then try the common bare
    names, returning the first that exists in ``model``:

    * ``mobilebase0_base`` — the real mobile base of a robosuite/RoboCasa mobile
      manipulator. Tried **before** ``robot0_base`` because in those composed
      scenes ``robot0_base`` is a placeholder mount left at a fixed offset
      (e.g. ``(10, 10, 0)``) — locking the camera onto it frames empty space.
    * ``base`` — synthetic / single-body twins.
    * ``robot0_base`` — fixed-arm robosuite (LIBERO etc.), where it *is* the base.
    * ``base_link`` — generic fallback.

    Returns ``None`` when no candidate body is present, so callers can fall back
    (e.g. the viewer camera centres on the model bounds instead).
    """
    import mujoco  # reason: defer optional sim dep
    from openral_core import extract_base_sim_joint_names

    candidates: list[str] = []
    if description is not None:
        base_names = extract_base_sim_joint_names(description)
        if base_names:
            first = base_names[0]
            prefix = first.split("_joint_")[0] if "_joint_" in first else ""
            if prefix:
                candidates.append(f"{prefix}_base")
    candidates += ["mobilebase0_base", "base", "robot0_base", "base_link"]
    for name in candidates:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    return None


# Substrings of a 3rd-person "workspace overview" camera, in preference order:
# robosuite/RoboCasa ``robot0_agentview_*``, gym-aloha ``top``, then the generic
# ``frontview`` / ``front`` 3rd-person cams. ``top`` is ranked above bare
# ``front`` so aloha picks its top-down overview rather than its ``front_close``
# zoom. Matched case-insensitively as substrings of the model's camera names.
_VIEWER_CAMERA_PREFS: tuple[str, ...] = ("agentview", "top", "frontview", "front")


def preferred_viewer_camera_id(
    model: Any, *, prefer: tuple[str, ...] = _VIEWER_CAMERA_PREFS
) -> int:
    """Pick a named MJCF camera for the viewer to open on; ``-1`` if none.

    Scene cameras are authored to frame the action, so opening the viewer on one
    avoids the free orbit camera's occlusion problems in cluttered scenes (a
    base-centred orbit in a RoboCasa kitchen ends up staring at a wall). Returns:

    * the id of the first camera whose name contains a ``prefer`` substring — a
      3rd-person workspace view (``robot0_agentview_left``, ``top``, ``agentview``);
    * else the first declared camera (e.g. a wrist / eye-in-hand cam), so the
      viewer still opens on an authored vantage when no overview cam exists;
    * else ``-1`` when the model declares no cameras, so the caller falls back to
      the base-aligned free camera (:func:`base_aligned_free_camera`).

    Example:
        >>> import mujoco
        >>> m = mujoco.MjModel.from_xml_string(
        ...     "<mujoco><worldbody>"
        ...     "<camera name='robot0_eye_in_hand'/><camera name='robot0_agentview_left'/>"
        ...     "<body name='b'><geom type='box' size='.1 .1 .1'/></body>"
        ...     "</worldbody></mujoco>"
        ... )
        >>> cid = preferred_viewer_camera_id(m)
        >>> mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        'robot0_agentview_left'
    """
    import mujoco  # reason: defer optional sim dep

    ncam = int(model.ncam)
    if ncam == 0:
        return -1
    names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or "").lower() for i in range(ncam)
    ]
    for pref in prefer:
        for i, name in enumerate(names):
            if pref in name:
                return i
    return 0  # any authored camera beats the occlusion-prone free orbit


def apply_robosuite_visual_geomgroups(opt: Any, model: Any) -> bool:
    """Hide collision shells in a robosuite/RoboCasa model so textures show.

    Robosuite/RoboCasa put **collision** geoms in group 0 (rendered as flat
    colours — RoboCasa's dark-red kitchen, the green robot capsules) and the
    **textured visual** geoms in group 1; their offscreen renderer shows only
    group 1, but ``mujoco.viewer`` shows every group by default, so the viewer
    looks like a red collision box. This sets ``opt.geomgroup`` to hide group 0
    and show group 1.

    Gated on a robosuite signature — a ``robot0_`` / ``gripper0_`` /
    ``mobilebase0_`` body, **or** an ``agentview`` / ``frontview`` camera (which
    catches custom robosuite compositions that don't use the ``robot0_`` prefix)
    — **not** on geom counts, because dm_control / gym scenes (gym-aloha) put
    their *visual* geoms in group 0, so blindly hiding it would blank them.
    Returns ``True`` when it acted, ``False`` (no-op) otherwise.
    """
    import mujoco  # reason: defer optional sim dep

    prefixes = ("robot0_", "gripper0_", "mobilebase0_")
    is_robosuite = any(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(prefixes)
        for i in range(int(model.nbody))
    ) or any(
        sig in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or "")
        for i in range(int(model.ncam))
        for sig in ("agentview", "frontview")
    )
    if not is_robosuite:
        return False
    opt.geomgroup[0] = 0  # collision shells — hide
    opt.geomgroup[1] = 1  # textured visual geoms — show
    return True


def base_aligned_free_camera(
    *,
    model: Any,
    data: Any,
    base_body_name: str | None = None,
    azimuth_offset_deg: float = 135.0,
    elevation_deg: float = -25.0,
    distance_scale: float = 2.0,
    max_distance_m: float = 3.5,
) -> tuple[tuple[float, float, float], float, float, float]:
    """Free-camera framing centred on the robot base, aligned to its frame.

    Returns ``(lookat_xyz, distance, azimuth_deg, elevation_deg)`` for a MuJoCo
    free camera (an ``MjvCamera`` with ``type = mjCAMERA_FREE``).

    MuJoCo's world frame is immutable and the orbit camera's azimuth/elevation
    are world-relative, so the viewer cannot be re-rooted onto ``base_link``.
    Instead this points the camera at the base body's world origin
    (``lookat`` = base position) and offsets the azimuth by the base frame's
    world yaw, so the opening view is framed identically relative to the robot's
    own forward (+X) axis no matter where or how the base is placed in the world
    — i.e. "centred on and aligned with the base reference frame".

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (read for the base body's current pose).
        base_body_name: MJCF body backing the robot's ``base_frame``. When
            ``None`` or absent from the model, the camera falls back to the
            model's bounding centre (``model.stat.center``) with no yaw offset,
            so the helper is safe on any model.
        azimuth_offset_deg: Bearing of the camera relative to the base +X axis.
            Only the fallback when a scene has no authored camera (see
            :func:`preferred_viewer_camera_id`), so it targets open single-robot
            twins rather than cluttered scenes.
        elevation_deg: Camera elevation (negative looks down).
        distance_scale: Orbit distance as a multiple of ``model.stat.extent``.
        max_distance_m: Hard cap on the orbit distance. ``model.stat.extent`` is
            the whole-model bound, which for a composed scene (a RoboCasa
            kitchen is ~20 m across) would push the camera tens of metres away
            and shrink the robot to a speck. Since the camera frames the *robot*
            (not the scene), the distance is capped to keep the robot ~screen-
            filling; small scenes (a tabletop ~1-2 m) stay below the cap and are
            unaffected.

    Returns:
        ``(lookat_xyz, distance, azimuth_deg, elevation_deg)``.

    Example:
        >>> import mujoco
        >>> m = mujoco.MjModel.from_xml_string(
        ...     "<mujoco><worldbody><body name='base'>"
        ...     "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
        ... )
        >>> d = mujoco.MjData(m)
        >>> mujoco.mj_forward(m, d)
        >>> lookat, dist, az, el = base_aligned_free_camera(model=m, data=d, base_body_name="base")
        >>> lookat
        (0.0, 0.0, 0.0)
        >>> round(az, 1)
        135.0
    """
    import mujoco  # reason: defer optional sim dep

    extent = float(getattr(model.stat, "extent", 1.0)) or 1.0
    distance = min(extent * float(distance_scale), float(max_distance_m))

    bid = -1
    if base_body_name:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        center = np.asarray(model.stat.center, dtype=np.float64)
        return (
            (float(center[0]), float(center[1]), float(center[2])),
            distance,
            float(azimuth_offset_deg),
            float(elevation_deg),
        )

    base_pos = np.asarray(data.xpos[bid], dtype=np.float64)
    rot_world_base = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    base_yaw_deg = float(np.degrees(np.arctan2(rot_world_base[1, 0], rot_world_base[0, 0])))
    return (
        (float(base_pos[0]), float(base_pos[1]), float(base_pos[2])),
        distance,
        base_yaw_deg + float(azimuth_offset_deg),
        float(elevation_deg),
    )


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
