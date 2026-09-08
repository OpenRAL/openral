# SPDX-License-Identifier: Apache-2.0
"""Reusable, robot-agnostic depth-camera → PointCloud2 plumbing.

A deploy-sim HAL node turns each depth ``SensorSpec`` into a
``sensor_msgs/PointCloud2`` that ``octomap_server`` lifts into the OctoMap
feeding the safety kernel's world-collision check. Shared pieces so a node
only wires publishers/timers:

* ``is_depth_sensor`` / ``mjcf_camera_name`` / ``depth_synth_kwargs``
  — pure SensorSpec adapters (no ROS / MuJoCo import).
* ``camera_optical_tf_to_base`` — camera-optical-frame → base transform
  from the live MuJoCo poses.
* ``points_from_depth_grid`` — back-project a depth raster into an
  ``(N, 3)`` optical-frame cloud (one ray-cast feeds both depth image and
  cloud; see ``openral_hal.sim_sensor_bridge``).
* ``pointcloud2_from_points_xyz`` — pack an ``(N, 3)`` array into a
  ``sensor_msgs/PointCloud2``.

Synth: ``openral_sim.backends.depth_camera.synthesize_depth_image``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# REP-103 optical (x right, y down, z forward) expressed in the MuJoCo
# camera frame (x right, y up, z back): flip y and z.
_OPTICAL_IN_MJCAM = np.diag([1.0, -1.0, -1.0])

# Degenerate eye→lookat distance below which the azimuth/elevation are undefined.
_MIN_EYE_DISTANCE_M = 1e-6

# Deploy-viewer framing on an authored scene camera: lift the orbit pivot off the
# floor onto the robot body, and pull the eye back, so cluttered mobile-manip
# scenes show the whole robot instead of a floor-filling counter close-up.
_VIEWER_LOOKAT_LIFT_M = 0.7
_VIEWER_PULLBACK_M = 2.0


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
    ``max_range_default`` when ``range_max_m`` is unset). Ray-cast pinhole
    model: ``(u-cx)/fx, (v-cy)/fy``.

    When ``render_size`` is given (the scene's ``observation_width``/``height``),
    intrinsics are rescaled to it first via ``openral_core.scale_intrinsics_to``
    so the back-projected cloud matches the RGB the env rendered at that
    resolution. ``None`` keeps the manifest's nominal intrinsics.
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

    Returns every body whose name shares a prefix (first ``_``-delimited
    token) with one of the robot's ``sim_joint_name``s — e.g. ``mobilebase0``
    / ``robot0`` / ``gripper0`` in a robosuite/RoboCasa scene — plus every
    descendant of those matched roots (needed because robosuite names some
    nested arm bodies generically).

    Why: without this a base-mounted depth camera voxelises the robot's own
    arm into the world map and the kernel's world-collision check flags it
    against itself.
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
    # MuJoCo body ids are parent-before-child. Include generic / unnamed
    # descendants below any matched robot root without pulling in sibling
    # kitchen fixtures.
    for i in range(1, int(model.nbody)):
        if int(model.body_parentid[i]) in out:
            out.add(i)
    return frozenset(out)


def resolve_base_body_name(model: Any, *, description: Any = None) -> str | None:
    """Resolve the MJCF body backing a robot's ``base_frame``, or ``None``.

    When a ``RobotDescription`` is given, derives ``<prefix>_base`` from the
    first base joint's prefix first, then tries these bare names in order,
    returning the first that exists in ``model``:

    * ``mobilebase0_base`` — real mobile base of a robosuite/RoboCasa mobile
      manipulator. Tried before ``robot0_base``: in those composed scenes
      ``robot0_base`` is a placeholder mount at a fixed offset (e.g.
      ``(10, 10, 0)``), so locking onto it frames empty space.
    * ``base`` — synthetic / single-body twins.
    * ``robot0_base`` — fixed-arm robosuite (LIBERO etc.), where it *is* the base.
    * ``base_link`` — generic fallback.

    Returns ``None`` when no candidate exists, so callers (e.g. the viewer
    camera) can fall back to the model bounds.
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


def resolve_base_frame_body_name(model: Any, *, description: Any = None) -> str | None:
    """Resolve the MJCF body whose pose the robot's ``base_frame`` TF carries.

    Not always ``resolve_base_body_name`` (ADR-0095). That resolves the
    chassis *root* — the right anchor for the depth self-filter's
    ``mj_multiRay`` body-exclude and the viewer's follow camera. This resolves
    the body ``base_frame`` denotes on ``/tf``, which any extrinsic published
    as ``base_frame -> <child>`` must be measured against. Tries
    ``<prefix>_support`` first, then falls back to
    ``resolve_base_body_name``'s chassis candidates. Fixed-base arms
    (LIBERO franka, ur5e, …) have no ``_support`` body and resolve unchanged.

    Why they differ: on robosuite/RoboCasa mobile manipulators, OmronMobileBase
    stacks a geomless ground-level root (``mobilebase0_base``, world z 0) under
    a 0.70 m pedestal whose top plate (``mobilebase0_support``) carries the arm
    and cameras. ``base_link`` on ``/tf`` is that pedestal top —
    ``MobileBaseBridge`` publishes
    ``odom -> base_link`` from ``base_pose_6dof()`` (RoboCasa's
    ``robot0_base_pos``, z = 0.70 m), the convention the pi05 / rldx / XR-1
    state assemblers were trained against. Measuring against the ground-level
    root instead put the depth cloud, the OctoMap and the kernel's world-voxel
    grid 0.70 m out for every TF consumer.

    Returns ``None`` when no candidate body is present.
    """
    import mujoco  # reason: defer optional sim dep
    from openral_core import extract_base_sim_joint_names

    if description is not None:
        base_names = extract_base_sim_joint_names(description)
        if base_names:
            first = base_names[0]
            prefix = first.split("_joint_")[0] if "_joint_" in first else ""
            mount = f"{prefix}_support"
            if prefix and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount) >= 0:
                return mount
    return resolve_base_body_name(model, description=description)


# Substrings of a 3rd-person "workspace overview" camera, preference order:
# robosuite/RoboCasa robot0_agentview_*, gym-aloha top, then frontview/front.
# `top` ranks above bare `front` so aloha picks its top-down overview over
# `front_close`. Matched case-insensitively against model camera names.
_VIEWER_CAMERA_PREFS: tuple[str, ...] = ("agentview", "top", "frontview", "front")


def preferred_viewer_camera_id(
    model: Any, *, prefer: tuple[str, ...] = _VIEWER_CAMERA_PREFS
) -> int:
    """Pick a named MJCF camera for the viewer to open on; ``-1`` if none.

    Returns the id of the first camera whose name contains a ``prefer``
    substring, else the first declared camera, else ``-1`` (caller falls back
    to ``base_aligned_free_camera``).

    Why: scene cameras are authored to frame the action, avoiding the free
    orbit camera's occlusion in cluttered scenes (a base-centred orbit in a
    RoboCasa kitchen ends up staring at a wall).

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

    Sets ``opt.geomgroup`` to hide group 0 (collision) and show group 1
    (textured visual). Gated on a robosuite signature — a ``robot0_`` /
    ``gripper0_`` / ``mobilebase0_`` body, or an ``agentview`` / ``frontview``
    camera — not on geom counts, since gym/dm_control scenes (gym-aloha) put
    their *visual* geoms in group 0. Returns ``True`` when it acted.

    Why: robosuite/RoboCasa's offscreen renderer shows only group 1, but
    ``mujoco.viewer`` shows every group by default, so the viewer otherwise
    looks like a flat-coloured collision box.
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

    Points the camera at the base body's world origin (``lookat`` = base
    position) and offsets the azimuth by the base frame's world yaw, so the
    opening view is framed the same relative to the robot's forward (+X) axis
    regardless of world placement.

    Why: MuJoCo's world frame is immutable and the orbit camera's
    azimuth/elevation are world-relative, so the viewer cannot be re-rooted
    onto ``base_link``.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (read for the base body's current pose).
        base_body_name: MJCF body backing the robot's ``base_frame``. When
            ``None`` or absent, falls back to ``model.stat.center`` with no
            yaw offset.
        azimuth_offset_deg: Bearing of the camera relative to the base +X axis.
            Only used as the fallback when a scene has no authored camera (see
            ``preferred_viewer_camera_id``).
        elevation_deg: Camera elevation (negative looks down).
        distance_scale: Orbit distance as a multiple of ``model.stat.extent``.
        max_distance_m: Hard cap on the orbit distance, since ``model.stat.extent``
            is the whole-model bound (a RoboCasa kitchen is ~20 m across) and
            would otherwise shrink the robot to a speck; small scenes (a
            tabletop ~1-2 m) stay below the cap.

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


def initial_viewer_camera(
    *, model: Any, data: Any, description: Any = None
) -> tuple[tuple[float, float, float], float, float, float]:
    """Opening **free-camera** pose for the viewer ``(lookat, distance, az, el)``.

    The viewer always uses a *free* camera (``mjCAMERA_FREE``, not
    ``mjCAMERA_FIXED``) so the user keeps mouse control (drag to orbit, scroll
    to zoom); this only sets the initial viewpoint. When the scene ships an
    authored overview camera (``preferred_viewer_camera_id``), the eye is
    placed at that camera's world position with the orbit pivot on the robot
    base, so the opening view matches the authored vantage while staying
    interactive. Otherwise falls back to ``base_aligned_free_camera``.

    MuJoCo places the eye at ``lookat - distance · f`` where the unit forward
    ``f = (cos el cos az, cos el sin az, sin el)``; the returned tuple recovers
    it via ``distance = ‖lookat - eye‖``, ``azimuth = atan2(fy, fx)``,
    ``elevation = asin(fz)``.
    """
    import mujoco  # reason: defer optional sim dep

    base_body = resolve_base_body_name(model, description=description)
    cam_id = preferred_viewer_camera_id(model)
    if cam_id >= 0:
        eye = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        bid = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            if base_body is not None
            else -1
        )
        lookat = (
            np.asarray(data.xpos[bid], dtype=np.float64)
            if bid >= 0
            else np.asarray(model.stat.center, dtype=np.float64)
        )
        forward = lookat - eye
        distance = float(np.linalg.norm(forward))
        if distance > _MIN_EYE_DISTANCE_M:
            forward /= distance
            azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
            elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
            # Frame the robot *body* (not the floor its base sits on) and pull the
            # eye back so cluttered deploy scenes (RoboCasa) show the whole robot
            # rather than a floor-filling close-up of the authored counter cam.
            lookat = lookat + np.array([0.0, 0.0, _VIEWER_LOOKAT_LIFT_M])
            distance += _VIEWER_PULLBACK_M
            return (
                (float(lookat[0]), float(lookat[1]), float(lookat[2])),
                distance,
                azimuth,
                elevation,
            )
    return base_aligned_free_camera(model=model, data=data, base_body_name=base_body)


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


def depth_image_from_grid(
    depth: NDArray[np.float32],
    *,
    frame_id: str,
    stamp: Any = None,
) -> Any:
    """Pack an ``(H, W)`` metric-depth raster into a ``32FC1 sensor_msgs/Image``.

    The dense, organised depth image nvblox's projective depth integrator
    consumes — produced by
    ``openral_sim.backends.depth_camera.synthesize_depth_image``. Pixels are
    perpendicular optical-Z metres, ``0.0`` = no measurement (nvblox skips them).

    Args:
        depth: ``(H, W)`` float32 depth raster in metres (optical-Z).
        frame_id: tf2 frame the image lives in (the camera optical frame); must
            match the companion ``CameraInfo``.
        stamp: ``builtin_interfaces/Time`` header stamp; ``None`` leaves zero.
    """
    from sensor_msgs.msg import Image  # reason: defer ROS dep

    grid = np.ascontiguousarray(depth, dtype="<f4")
    h, w = int(grid.shape[0]), int(grid.shape[1])

    msg = Image()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = h
    msg.width = w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = 4 * w
    msg.data = grid.reshape(-1).tobytes()
    return msg


def points_from_depth_grid(
    depth: NDArray[np.float32],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    clearing: NDArray[np.bool_] | None = None,
    max_range_m: float | None = None,
) -> NDArray[np.float32]:
    """Back-project an ``(H, W)`` metric-depth raster into an ``(N, 3)`` cloud.

    The inverse of the pinhole projection the depth synth casts: a pixel
    ``(col, row)`` holding perpendicular optical-Z ``z`` becomes
    ``((col - cx) / fx · z, (row - cy) / fy · z, z)`` in the camera optical
    frame (REP-103). Pixels reading exactly ``0.0`` (the "no measurement"
    sentinel) are dropped.

    ``clearing`` marks pixels whose only return was a self-filtered body (own
    link, an acknowledged payload): no depth, but the ray behind is free, so
    a ``max_range_m`` endpoint is emitted there, letting OctoMap clear
    occluded cells instead of leaving them frozen. Passing the mask
    ``openral_sim.backends.depth_camera.synthesize_depth_frame`` returns
    reproduces what
    ``synthesize_depth_pointcloud``
    would cast separately, from one ray-cast — one cast per camera per frame
    instead of two (depth image + cloud).

    Args:
        depth: ``(H, W)`` float32 depth raster in metres (optical-Z), ``0.0``
            where there is no measurement — i.e. what
            ``openral_sim.backends.depth_camera.synthesize_depth_image``
            returns and ``depth_image_from_grid`` packs.
        fx: Focal length x **of this raster** (pixels) — for a strided synth the
            stride-scaled value, the same one the companion ``CameraInfo``
            advertises (see ``camera_info_from_intrinsics``).
        fy: Focal length y of this raster (pixels).
        cx: Principal point x of this raster (pixels).
        cy: Principal point y of this raster (pixels).
        clearing: ``(H, W)`` bool mask of self-filtered rays with no farther
            surface, from
            ``openral_sim.backends.depth_camera.synthesize_depth_frame``.
            ``None`` (the default) emits measured returns only.
        max_range_m: Euclidean range the clearing endpoints are placed at.
            Required when ``clearing`` marks any pixel.

    Returns:
        ``(N, 3)`` float32 XYZ in the camera optical frame, row-major in pixel
        order; ``(0, 3)`` when no pixel is measured or cleared.

    Raises:
        ROSConfigError: ``clearing`` marks a pixel but ``max_range_m`` is unset,
            or its shape disagrees with ``depth``.

    Example:
        >>> import numpy as np
        >>> # (1, 2) raster: pixel (row 0, col 0) has no measurement.
        >>> grid = np.array([[0.0, 2.0]], dtype=np.float32)
        >>> points_from_depth_grid(grid, fx=4.0, fy=4.0, cx=1.0, cy=0.5).tolist()
        [[0.0, -0.25, 2.0]]
        >>> # …and now pixel (0, 0) is the robot's own arm: free to 4 m.
        >>> free = np.array([[True, False]])
        >>> [
        ...     [round(v, 4) for v in p]
        ...     for p in points_from_depth_grid(
        ...         grid, fx=4.0, fy=4.0, cx=1.0, cy=0.5, clearing=free, max_range_m=4.0
        ...     ).tolist()
        ... ]
        [[-0.9631, -0.4815, 3.8523], [0.0, -0.25, 2.0]]
    """
    from openral_core.exceptions import ROSConfigError  # reason: defer core import

    grid = np.asarray(depth, dtype=np.float32)
    measured = grid > 0.0
    if clearing is None:
        selected = measured
    else:
        free = np.asarray(clearing, dtype=np.bool_)
        if free.shape != grid.shape:
            raise ROSConfigError(
                f"clearing mask {free.shape} does not match the depth raster {grid.shape}."
            )
        if free.any() and max_range_m is None:
            raise ROSConfigError(
                "points_from_depth_grid needs max_range_m to place the self-filter's "
                "clearing endpoints; pass the same value the cast used."
            )
        selected = measured | free
    rows, cols = np.nonzero(selected)  # row-major order: matches the ray order
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Pinhole ray through each selected pixel, in the camera optical frame.
    ray = np.empty((rows.size, 3), dtype=np.float64)
    ray[:, 0] = (cols.astype(np.float64) - cx) / fx
    ray[:, 1] = (rows.astype(np.float64) - cy) / fy
    ray[:, 2] = 1.0
    # Measured pixels scale the ray by optical-Z; cleared pixels are placed at
    # max_range along the UNIT ray, exactly as the cloud synth does.
    z = grid[rows, cols].astype(np.float64)
    points = ray * z[:, None]
    if clearing is not None:
        free_sel = np.asarray(clearing, dtype=np.bool_)[rows, cols] & ~measured[rows, cols]
        if np.any(free_sel):
            unit = ray[free_sel] / np.linalg.norm(ray[free_sel], axis=1, keepdims=True)
            points[free_sel] = unit * float(max_range_m)  # type: ignore[arg-type]  # reason: guarded above
    return points.astype(np.float32)


def depth_grid_from_image(msg: Any) -> NDArray[np.float64]:
    """Decode a ``sensor_msgs/Image`` depth frame into an ``(H, W)`` metre raster.

    The inverse of ``depth_image_from_grid``, and the reader half the HAL's
    vision attachment bridge needs: on real hardware the wrist depth arrives
    from a camera driver, not from the simulator that produced it here. Both
    REP-118 depth encodings are accepted, because both are shipped by drivers
    OpenRAL already supports (``openral_sensors.realsense`` /
    ``openral_sensors.luxonis`` publish ``16UC1``; the sim bridge and
    ``depth_provider_node`` publish ``32FC1``).

    Args:
        msg: A ``sensor_msgs/Image`` with encoding ``32FC1`` (metres) or
            ``16UC1`` (millimetres, the RealSense/OAK-D convention).

    Returns:
        An ``(H, W)`` float64 raster in **metres**. ``0`` stays ``0`` — the
        REP-118 "no measurement" value, which every consumer here treats as
        invalid depth rather than as a surface at the optical centre.

    Raises:
        ROSConfigError: On an unsupported encoding, or a payload whose length
            does not match ``height * width``.
    """
    from openral_core.exceptions import ROSConfigError

    height, width = int(msg.height), int(msg.width)
    encoding = str(msg.encoding)
    if encoding == "32FC1":
        dtype: str = "<f4"
        scale = 1.0
    elif encoding == "16UC1":
        dtype, scale = "<u2", 1e-3
    else:
        raise ROSConfigError(
            f"depth_grid_from_image: unsupported depth encoding {encoding!r}; "
            "expected '32FC1' (metres) or '16UC1' (millimetres)."
        )
    flat = np.frombuffer(bytes(msg.data), dtype=dtype)
    if flat.size != height * width:
        raise ROSConfigError(
            f"depth_grid_from_image: {encoding} payload has {flat.size} samples "
            f"for a {height}x{width} frame."
        )
    return np.asarray(flat, dtype=np.float64).reshape(height, width) * scale


def camera_info_from_intrinsics(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_id: str,
    stamp: Any = None,
) -> Any:
    """Build a pinhole ``sensor_msgs/CameraInfo`` for a synthesised depth image.

    The intrinsics are those of the **rasterised** image — for a strided depth
    synth (``openral_sim.backends.depth_camera.synthesize_depth_image``) the
    caller passes the stride-scaled values (``fx / stride`` … ``cy / stride``,
    ``width``/``height`` = the strided raster dims) so the model is consistent
    with the image nvblox receives.

    Distortion is zero (``plumb_bob``, ``D = [0,0,0,0,0]``) — the MuJoCo pinhole
    ray-cast has no lens distortion — with an identity rectification ``R`` and a
    ``P`` that mirrors ``K`` (no stereo baseline).

    Args:
        width: Rasterised image width (pixels).
        height: Rasterised image height (pixels).
        fx: Focal length x of the rasterised image (pixels).
        fy: Focal length y of the rasterised image (pixels).
        cx: Principal point x of the rasterised image (pixels).
        cy: Principal point y of the rasterised image (pixels).
        frame_id: tf2 frame; must match the companion depth image.
        stamp: ``builtin_interfaces/Time`` header stamp; ``None`` leaves zero.
    """
    from sensor_msgs.msg import CameraInfo  # reason: defer ROS dep

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = int(height)
    msg.width = int(width)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg
