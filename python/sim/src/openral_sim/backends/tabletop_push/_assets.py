"""Compose the ``tabletop_push`` MJCF — robot-agnostic, MjSpec-based (ADR-0033).

Unlike :mod:`openral_sim.backends.so101_box._assets` (which regex-splices its
task world into the SO-ARM101 MJCF and is therefore coupled to that robot's
``<body name="base">`` / ``<body name="gripper">`` / ``"1"``..``"6"`` schema),
this composer is **robot as a flag**: it works against any arm whose manifest
declares an ``assets.mjcf``.

How it stays robot-agnostic
---------------------------
* The robot's own MJCF is loaded into a :class:`mujoco.MjSpec`; the task world
  (table, cube, goal marker, cameras, light) is **appended** to that spec's
  ``worldbody``. Appending never reorders the robot's joints/actuators, so the
  composed model's actuator and qpos indices stay 1:1 with ``description.joints``
  in declaration order — exactly the contract
  :meth:`openral_hal.MujocoArmHAL._sim_kwargs_for` relies on. The free objects'
  qpos land *after* the robot's, so driving the robot by its low actuator
  indices is correct regardless of which robot is loaded.
* The robot base is re-anchored by mutating the spec's **root body**
  (``worldbody.bodies[0]``) — no body-name lookup, so an SO-ARM ``base``, a
  Franka ``link0`` and a UR ``base`` are all handled identically.
* The robot's base MJCF is resolved from the manifest via
  :func:`openral_core.assets.resolve_asset` (the same source
  ``build_hal(mode="sim")`` uses) — the manifest is the single robot-MJCF
  contract across ``sim run`` / ``deploy sim`` / ``deploy run``.

The composed model is returned directly (no sibling XML file): MjSpec embeds the
resolved meshes at compile time, so there is nothing to resolve relative to a
written path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz

if TYPE_CHECKING:
    import mujoco
    from openral_core import Pose6D, RobotDescription

__all__ = [
    "TabletopOptions",
    "compose_tabletop_mjcf",
]


@dataclass(frozen=True)
class TabletopOptions:
    """Every dimension and pose the ``tabletop_push`` scene exposes.

    Lengths are metres, angles are radians unless suffixed ``_deg``. Defaults
    suit a small tabletop arm (the SO-101 proof-of-concept robot); a larger arm
    (Franka, UR) typically wants the table and spawn ranges tuned via the
    YAML ``scene.backend_options`` — the composition mechanism is identical
    either way.

    Attributes:
        table_size_xy: Full ``(X, Y)`` extent of the tabletop slab in metres.
        table_top_z: World-Z of the table's top surface in metres. The robot
            base is mounted at this height by default (``robot_base_xyz`` z).
        table_thickness: Thickness of the table slab in metres.
        table_center_xy: World ``(X, Y)`` centre of the table slab.
        robot_base_xyz: Fallback world-frame position of the robot root body,
            in metres, used only when the scene YAML omits ``base_pose:``.
        robot_base_yaw_deg: Fallback rotation about world +Z applied to the
            robot root body, degrees. Used only when ``base_pose:`` is omitted.
        cube_size: Half-extent ``(X, Y, Z)`` of the manipuland cube, metres.
        cube_mass: Cube mass in kg.
        cube_spawn_xy_range: Cube spawn area in world frame, given as
            ``((x_min, x_max), (y_min, y_max))``. The cube spawns resting on
            the table at a random ``(x, y)`` within this range each ``reset``.
        goal_spawn_xy_range: Goal-marker spawn area, same shape. The goal is a
            flat visual disc on the table surface; success is geometric (cube
            centre within ``goal_radius`` of it).
        goal_radius: Success XY tolerance — the cube centre must be within this
            radius of the goal centre, metres.
        goal_min_separation: Minimum cube↔goal centre separation at spawn,
            metres (so the task always requires a non-trivial push).
        off_table_z_tol: Success also requires the cube to still be resting on
            the table — its centre Z within this tolerance of the resting
            height ``table_top_z + cube_size_z``. Guards against "cube knocked
            off the table happens to pass over the goal XY" false positives.
        overhead_camera_pos: World-frame position of the overhead camera.
        overhead_camera_fovy: Overhead camera vertical FoV, degrees.
        front_camera_pos: World-frame position of the front camera.
        front_camera_fovy: Front camera vertical FoV, degrees.
        wrist_camera_mount_body: When set, the composed model parents a
            ``wrist`` camera to the named MJCF body (the robot's end-effector
            link, e.g. ``"gripper"`` for SO-101 or ``"hand"`` for Franka). Left
            ``None``, callers may infer the mount from the robot manifest's
            ``sensors[].sim_placement.parent_body`` for the ``wrist`` camera.
            A name absent from the loaded MJCF fails loudly.
        wrist_camera_pos_local: Wrist-camera position in the mount-body frame.
        wrist_camera_fovy: Wrist-camera vertical FoV, degrees.
        settle_steps: ``mj_step`` calls after each action write — the scene's
            position actuators are advanced this many steps per ``step()``.
        initial_joint_positions: Optional reset pose in the same convention as
            ``joint_units`` plus the calibration affine. Empty means use the
            robot MJCF's default qpos. Values are clipped to the robot joint
            limits before writing qpos/ctrl.
        joint_units: Unit convention for the robot joint state/action
            boundary. ``"radians"`` is sim-native; ``"degrees"`` is for
            LeRobot SO-100/SO-101 checkpoints that record servo degrees.
        joint_offsets_deg: Per-actuator affine offset used only when
            ``joint_units == "degrees"``:
            ``policy_deg = sign * scale * mujoco_deg + offset``.
        joint_signs: Per-actuator affine sign, each ``+1`` or ``-1``.
        joint_scales: Per-actuator affine scale (policy degrees per MuJoCo
            degree), positive values only. Empty means identity scale.
        ambient_light: Headlight ambient RGB lifted into the scene so the
            table top renders with even illumination.
        instruction: Default natural-language task instruction (the YAML
            ``task.instruction`` overrides it).
        extra_metadata: Free-form string map echoed into scene metadata.
    """

    table_size_xy: tuple[float, float] = (0.80, 0.80)
    table_top_z: float = 0.0
    table_thickness: float = 0.02
    table_center_xy: tuple[float, float] = (0.0, 0.30)

    robot_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base_yaw_deg: float = 0.0

    cube_size: tuple[float, float, float] = (0.025, 0.025, 0.025)
    cube_mass: float = 0.05
    cube_spawn_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.15, 0.15),
        (0.18, 0.30),
    )
    goal_spawn_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.15, 0.15),
        (0.32, 0.44),
    )
    goal_radius: float = 0.05
    goal_min_separation: float = 0.12
    off_table_z_tol: float = 0.04

    overhead_camera_pos: tuple[float, float, float] = (0.0, 0.30, 0.90)
    overhead_camera_fovy: float = 58.0
    front_camera_pos: tuple[float, float, float] = (0.0, -0.45, 0.45)
    front_camera_fovy: float = 58.0

    wrist_camera_mount_body: str | None = None
    wrist_camera_pos_local: tuple[float, float, float] = (0.0, 0.0, -0.05)
    wrist_camera_fovy: float = 75.0

    settle_steps: int = 5
    initial_joint_positions: tuple[float, ...] = ()
    joint_units: str = "radians"
    joint_offsets_deg: tuple[float, ...] = ()
    joint_signs: tuple[float, ...] = ()
    joint_scales: tuple[float, ...] = ()
    ambient_light: tuple[float, float, float] = (0.4, 0.4, 0.4)

    instruction: str = "push the red cube onto the green goal marker"
    extra_metadata: dict[str, str] = field(default_factory=dict)


def _resolve_robot_mjcf(description: RobotDescription) -> str:
    """Resolve a robot's base MJCF path from its manifest ``assets.mjcf``.

    Reuses :func:`openral_core.assets.resolve_asset` — the single resolver that
    ``build_hal(mode="sim")`` / ``MujocoArmHAL.from_description`` use — so every
    ref scheme (``rd:``, ``gym_aloha:``, ``openarm:``, ``menagerie:``, ``file:``)
    is honoured identically across the sim/real paths.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if not description.assets.mjcf:
        raise ROSConfigError(
            f"tabletop_push: robot {description.name!r} has no `assets.mjcf` in its "
            "manifest, so there is no base MJCF to build the tabletop scene around.",
        )
    try:
        path = resolve_asset(description.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(f"tabletop_push: {exc}") from exc
    if path is None:
        raise ROSConfigError(
            f"tabletop_push: assets.mjcf={description.assets.mjcf!r} did not resolve to a file.",
        )
    return str(path)


def infer_wrist_camera_mount_body(description: RobotDescription) -> str | None:
    """Return the wrist camera's MJCF parent body from the robot manifest.

    Free-axis scenes can request a logical ``wrist`` camera without hardcoding
    a robot-specific MJCF body name. This keeps the robot manifest as the
    source of truth for the camera mount.
    """
    for sensor in description.sensors:
        if sensor.name != "wrist" or sensor.sim_placement is None:
            continue
        if sensor.sim_placement.parent_body:
            return sensor.sim_placement.parent_body
    return None


def _base_pos_quat(
    options: TabletopOptions,
    base_pose: Pose6D | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the robot root-body ``(pos, quat_wxyz)`` for the MJCF.

    ``base_pose`` (honoured by free-axis scenes) wins when set — full 6-DOF,
    so a robot can be tilted/rotated without a per-scene helper. When the YAML
    omits it, the ``robot_base_xyz`` / ``robot_base_yaw_deg`` options provide a
    yaw-only fallback so a minimal config still composes.

    Returns the position and a MuJoCo ``(w, x, y, z)`` quaternion.
    """
    if base_pose is not None:
        x, y, z = base_pose.xyz
        qx, qy, qz, qw = base_pose.quat_xyzw  # tf2 / Pydantic order is (x, y, z, w)
        return (float(x), float(y), float(z)), (float(qw), float(qx), float(qy), float(qz))
    half = math.radians(options.robot_base_yaw_deg) * 0.5
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    return options.robot_base_xyz, quat


def compose_tabletop_mjcf(
    description: RobotDescription,
    options: TabletopOptions | None = None,
    *,
    base_pose: Pose6D | None = None,
) -> mujoco.MjModel:
    """Compose and compile the ``tabletop_push`` model around ``description``.

    The robot's base MJCF (resolved from ``description.assets.mjcf``) is loaded
    into an :class:`mujoco.MjSpec`; the table, cube, goal marker, two world
    cameras and a light are appended to its ``worldbody``; the robot root body
    is re-anchored to the requested base pose; and the spec is compiled. Because
    the task world is *appended*, the robot keeps the low actuator / qpos
    indices the manifest contract assumes.

    Args:
        description: Robot whose ``assets.mjcf`` provides the base arm MJCF.
        options: Scene options; ``None`` uses :class:`TabletopOptions` defaults.
        base_pose: Optional full 6-DOF robot mount pose (free-axis scenes honour
            it); falls back to ``options.robot_base_xyz`` / ``robot_base_yaw_deg``.

    Returns:
        A compiled :class:`mujoco.MjModel` ready for an :class:`mujoco.MjData`.

    Raises:
        ROSConfigError: If ``mujoco`` is missing, the manifest lacks
            ``assets.mjcf``, the loaded MJCF declares no root body, or a
            requested ``wrist_camera_mount_body`` is absent from it.
    """
    opts = options if options is not None else TabletopOptions()

    try:
        import mujoco as mj  # reason: optional sim-only dep
    except ModuleNotFoundError as exc:
        raise ROSConfigError(
            "mujoco is not installed. Install the sim extras with: "
            "just sync --all-packages --group sim",
        ) from exc

    mjcf_path = _resolve_robot_mjcf(description)
    try:
        spec = mj.MjSpec.from_file(mjcf_path)
    except (OSError, ValueError) as exc:
        raise ROSConfigError(
            f"tabletop_push: could not load robot MJCF {mjcf_path!r} for "
            f"{description.name!r}: {exc}",
        ) from exc

    if not spec.worldbody.bodies:
        raise ROSConfigError(
            f"tabletop_push: robot MJCF {mjcf_path!r} declares no <body> under "
            "<worldbody>; cannot anchor the robot into the tabletop scene.",
        )

    # Re-anchor the robot: its root body is the first worldbody child for every
    # menagerie arm. Mutating pos/quat rigidly moves the whole kinematic chain —
    # no body-name lookup, so this is identical across SO-ARM / Franka / UR.
    pos, quat = _base_pos_quat(opts, base_pose)
    root = spec.worldbody.bodies[0]
    root.pos = list(pos)
    root.quat = list(quat)

    # Even fill light so the table renders cleanly from every camera.
    spec.visual.headlight.ambient = list(opts.ambient_light)

    _append_table(spec, opts)
    _append_cube(spec, opts)
    _append_goal_marker(spec, opts)
    _append_world_cameras(spec, opts)
    _append_overhead_light(spec, opts)
    if opts.wrist_camera_mount_body is not None:
        _append_wrist_camera(spec, opts)

    try:
        return spec.compile()
    except ValueError as exc:
        raise ROSConfigError(
            f"tabletop_push: composed model for {description.name!r} failed to compile: {exc}",
        ) from exc


def _append_table(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Append the tabletop slab + a ground plane to the worldbody."""
    import mujoco as mj

    sx, sy = opts.table_size_xy
    cx, cy = opts.table_center_xy
    half_t = opts.table_thickness / 2.0
    table = spec.worldbody.add_geom()
    table.name = "table_top"
    table.type = mj.mjtGeom.mjGEOM_BOX
    table.size = [sx / 2.0, sy / 2.0, half_t]
    table.pos = [cx, cy, opts.table_top_z - half_t]
    table.rgba = [0.85, 0.80, 0.70, 1.0]
    table.friction = [1.0, 0.01, 0.001]

    floor = spec.worldbody.add_geom()
    floor.name = "ground"
    floor.type = mj.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.pos = [cx, cy, opts.table_top_z - opts.table_thickness - 0.30]
    floor.rgba = [0.30, 0.30, 0.32, 1.0]


def _append_cube(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Append the free-floating manipuland cube (a body with a freejoint)."""
    import mujoco as mj

    body = spec.worldbody.add_body()
    body.name = "cube"
    # Spawn pose is overwritten every reset(); a benign on-table default keeps
    # the compiled keyframe valid.
    cx, cy = opts.table_center_xy
    body.pos = [cx, cy, opts.table_top_z + opts.cube_size[2]]
    body.add_freejoint()
    geom = body.add_geom()
    geom.name = "cube_geom"
    geom.type = mj.mjtGeom.mjGEOM_BOX
    geom.size = list(opts.cube_size)
    geom.rgba = [0.85, 0.20, 0.20, 1.0]
    geom.mass = opts.cube_mass
    geom.friction = [1.0, 0.01, 0.001]


def _append_goal_marker(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Append the flat goal disc as a contact-free visual site on the table."""
    import mujoco as mj

    site = spec.worldbody.add_site()
    site.name = "goal"
    site.type = mj.mjtGeom.mjGEOM_CYLINDER
    site.size = [opts.goal_radius, 0.001, 0.0]
    cx, cy = opts.table_center_xy
    site.pos = [cx, cy, opts.table_top_z + 0.001]
    site.rgba = [0.10, 0.80, 0.20, 0.6]


def _append_world_cameras(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Append the overhead + front world-frame cameras (look at the table centre)."""
    import mujoco as mj

    cx, cy = opts.table_center_xy
    target = (cx, cy, opts.table_top_z)
    for name, pos, fovy in (
        ("overhead", opts.overhead_camera_pos, opts.overhead_camera_fovy),
        ("front", opts.front_camera_pos, opts.front_camera_fovy),
    ):
        cam = spec.worldbody.add_camera()
        cam.name = name
        cam.pos = list(pos)
        cam.mode = mj.mjtCamLight.mjCAMLIGHT_FIXED
        cam.quat = list(_look_at_quat(pos, target))
        cam.fovy = fovy


def _append_overhead_light(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Append a soft directional light above the table."""
    cx, cy = opts.table_center_xy
    light = spec.worldbody.add_light()
    light.name = "ceiling"
    light.pos = [cx, cy, opts.table_top_z + 1.0]
    light.dir = [0.0, 0.0, -1.0]
    light.diffuse = [0.7, 0.7, 0.7]
    light.specular = [0.1, 0.1, 0.1]
    light.castshadow = False


def _append_wrist_camera(spec: mujoco.MjSpec, opts: TabletopOptions) -> None:
    """Parent a ``wrist`` camera to the configured end-effector body.

    Robot-specific (the body name differs per MJCF), so this is opt-in via
    ``wrist_camera_mount_body``. Looks down the body-local -Z (the usual
    approach direction for a wrist-mounted camera).
    """
    import mujoco as mj

    name = opts.wrist_camera_mount_body
    mount = next((b for b in spec.bodies if b.name == name), None)
    if mount is None:
        available = sorted(b.name for b in spec.bodies if b.name)
        raise ROSConfigError(
            f"tabletop_push: wrist_camera_mount_body={name!r} is not a body in the "
            f"robot MJCF. Available bodies: {available}.",
        )
    cam = mount.add_camera()
    cam.name = "wrist"
    cam.pos = list(opts.wrist_camera_pos_local)
    cam.mode = mj.mjtCamLight.mjCAMLIGHT_FIXED
    cam.fovy = opts.wrist_camera_fovy


# MuJoCo (w, x, y, z) look-at quaternion — promoted to the shared gaze-geometry
# helper in ADR-0044 Phase 1; the "-z" default is the MuJoCo camera convention.
_look_at_quat = look_at_quat_wxyz
