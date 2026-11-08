"""Compose the ``metal_pen_holder`` MJCF — the MakerMods Metal Arm Station pen-holder rig.

The scene reproduces the real station the ``makermods/pen_holder_bi_20260723_161553``
LeRobot dataset was recorded on (``makermods/act_pen_holder_bi`` was trained on it): two
black Metal arms 0.612 m apart on a grey-brown wood desk, a dark wood riser shelf behind
them, an off-white wall, a white mesh pen cup on the shelf and a black marker on the desk.
The left arm lifts the cup to mid-desk, the right arm drops the marker into it, the left
arm puts the cup back. Sizes, poses and colours come from the dataset analysis (top-camera
fit, arm forward kinematics at grasp and release, crops of the objects); every one is a
``backend_options`` field.

The robot MJCF is the manifest's ``assets.mjcf`` (``makermods:metal_station``), loaded into
an ``mujoco.MjSpec``; the task world is appended, so the arms keep the qpos / actuator
indices the manifest's ``sim`` block declares (left 0..7 / 0..6, right 8..15 / 7..13).
There is no leader arm: in this dataset both arms are followers and no leader is in view.

The cameras are the manifest's own sensors (``sim_placement`` + pinhole ``intrinsics``,
principal point included): the measured top camera on ``station_base`` and the
single-arm wrist refit on each ``gripper_base`` (corrected by ``camera_offsets``, fitted to
this dataset's wrist frames). The static top camera is composited over
a real background plate (``assets/top_plate.jpg``, the dataset's temporal median) — only
the robot, the cup and the pen come from MuJoCo; the wrist cameras are fully rendered.
Shared pieces (light grid, textured boxes, principal pixel, the arm's body list) come from
``openral_sim.backends.metal_pick_place._assets``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openral_sim.backends.metal_pick_place._assets import (
    ARM_BODIES,
    ClutterItem,
    _static_geom,
    _textured_box,
    add_light_grid,
    mujoco_principal_pixel,
)

if TYPE_CHECKING:
    import mujoco
    import numpy as np
    from numpy.typing import NDArray
    from openral_core import RobotDescription

__all__ = [
    "ARMS",
    "ASSET_DIR",
    "MetalPenHolderOptions",
    "compose_metal_pen_holder",
    "holder_hex_texture",
    "holder_inner_half_width",
]

#: Bundled image assets (background plate, textures) + their SOURCE.txt.
ASSET_DIR: Path = Path(__file__).resolve().parent / "assets"
#: The two arm prefixes, in the manifest's joint order (left first).
ARMS: tuple[str, str] = ("left_", "right_")

_POLICY_DIM = 14  # per arm: joint1..6 + gripper, left then right
_SRGB_LINEAR_KNEE = 0.04045  # the sRGB transfer function's linear segment
_Vec2 = tuple[float, float]
_Vec3 = tuple[float, float, float]
_Vec4 = tuple[float, float, float, float]

# Dataset median first frame (degrees x6 + gripper stroke per arm, left then right).
_INITIAL = (-6.58, -0.16, 0.07, 1.98, -2.75, 1.49, 0.0, -1.21, 0.0, 4.79, -16.91, -1.67, -0.52, 0.0)


def _srgb(r: int, g: int, b: int) -> _Vec4:
    """An 8-bit sRGB colour as a linear MuJoCo rgba."""

    def ch(c: int) -> float:
        v = c / 255.0
        return round(v / 12.92 if v <= _SRGB_LINEAR_KNEE else ((v + 0.055) / 1.055) ** 2.4, 4)

    return (ch(r), ch(g), ch(b), 1.0)


class MetalPenHolderOptions(BaseModel):
    """Every dimension, pose, colour and unit the ``metal_pen_holder`` scene exposes.

    Lengths are metres in ``station_base`` (the midpoint between the arm bases, x forward
    toward the wall, y left, z up, z = 0 the desk the arms stand on); angles are degrees
    where suffixed ``_deg``. Defaults are the measured station (dataset analysis).

    Units: the policy-facing ``state`` / action are 14-D (left joint1..6, left gripper,
    then the right arm). ``joint_units="degrees"`` with ``gripper_scale=100`` is the
    dataset's convention (degrees with the URDF's sign and zero; gripper stroke ~0 closed
    .. ~105 open, joint8 metres = stroke / 2000 - ``gripper_offset_m``); the scene converts
    once, here (``converts_policy_units=True``).

    Example:
        >>> MetalPenHolderOptions().holder_start
        (0.475, 0.294, 0.0)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- units / control
    joint_units: str = Field(default="degrees", pattern=r"^(radians|degrees)$")
    gripper_scale: float = Field(default=100.0, gt=0.0)
    gripper_offset_m: float = Field(default=0.00125, ge=0.0)
    control_hz: float = Field(default=30.0, gt=0.0)
    initial_joint_positions: tuple[float, ...] = _INITIAL
    settle_steps: int = Field(default=50, ge=0)

    # ---- desk / shelf / wall / floor
    table_top_z: float = 0.0
    table_x_range: _Vec2 = (-0.35, 0.60)
    # The desk ends ~0.47 m right of the station's centre (its right edge back-projected from
    # the top plate: y -0.43 at x 0.11 to -0.55 at x 0.52); beyond it, the floor and the tray.
    table_y_range: _Vec2 = (-0.47, 1.0)
    table_thickness: float = 0.025
    # Region means matched to the real wrist frames (see the parity check in the README).
    table_rgba: _Vec4 = (0.85, 0.83, 0.8, 1.0)  # multiplies the wood texture
    table_emission: float = Field(default=0.3, ge=0.0)
    table_texture: str | None = "desk_wood.png"  # file under assets/, or absolute
    table_texture_size_m: _Vec2 = (0.13, 0.13)  # one texture tile along (y, x)
    # The riser shelf the cup starts and ends on (left_base x 0.385-0.59, y +0.18..-0.70). The
    # front edge sits 2 cm behind the plate estimate (0.382): there the replayed pens, at their
    # true centres, hit its front face while being lifted (3 of 8 replays); the left wrist's
    # view of the front face agrees it is further away than 0.382.
    shelf_x_range: _Vec2 = (0.402, 0.587)
    shelf_y_range: _Vec2 = (-0.394, 0.486)
    shelf_top_z: float = 0.09
    shelf_rgba: _Vec4 = (0.22, 0.19, 0.15, 1.0)  # the board's sides and front edge
    shelf_emission: float = Field(default=0.3, ge=0.0)
    # The board's top: the top plate's shelf pixels, rows = world y, columns = world x.
    shelf_texture: str | None = "shelf_wood.jpg"
    shelf_texture_gain: float = Field(default=1.35, gt=0.0)  # right wrist read 0.74x at 1.2
    wall_x: float = 0.60
    wall_rgba: _Vec4 = (0.78, 0.78, 0.77, 1.0)
    wall_emission: float = 0.26
    floor_z: float = -0.74
    floor_rgba: _Vec4 = (0.35, 0.34, 0.33, 1.0)
    # The white tray unit at the far right edge of the top view (out of reach).
    clutter: tuple[ClutterItem, ...] = (
        # The white plastic drawer unit right of the desk (top plate x 560-640 px, right
        # wrist frames at rest): floor to ~5 cm above the desk, its drawers' dark gaps on
        # the face toward the station.
        ClutterItem(
            pos=(0.25, -0.62, -0.33),
            size=(0.20, 0.12, 0.38),
            rgba=(0.85, 0.86, 0.85, 1.0),
            emission=0.3,
        ),
        *(
            ClutterItem(
                pos=(0.25, -0.499, z), size=(0.19, 0.002, 0.006), rgba=(0.2, 0.2, 0.21, 1.0)
            )
            for z in (-0.02, -0.17, -0.32)
        ),
    )

    # ---- the pen cup: a tapered square cup, origin at the centre of its bottom face
    # Sized against the real cup in the top frames at t = 0 (episodes 0, 15) and the left
    # grip (79-81 mm at 3.5 cm up): the analysis' 0.10 m top / 0.10 m height read ~15 % large.
    holder_bottom_width_m: float = Field(default=0.075, gt=0.0)
    holder_top_width_m: float = Field(default=0.085, gt=0.0)
    holder_height_m: float = Field(default=0.095, gt=0.0)
    holder_wall_m: float = Field(default=0.003, gt=0.0)
    holder_floor_m: float = Field(default=0.004, gt=0.0)
    holder_corner_radius_m: float = Field(default=0.015, ge=0.0)  # at the rim; scales down
    holder_mass_kg: float = Field(default=0.05, gt=0.0)
    # Torsional friction 0.03 m: the flat cup sides against the finger pads resist turning;
    # at 0.01 the held cup pivoted about the closing axis and tilted away from the pen.
    holder_friction: _Vec3 = (1.0, 0.03, 0.0005)
    holder_rgba: _Vec4 = _srgb(220, 222, 230)
    holder_emission: float = 0.25
    # Start (x, y, yaw_deg) on the shelf; the dataset mean.
    holder_start: _Vec3 = (0.475, 0.294, 0.0)

    # ---- the pen: a black marker lying on the desk, origin at its centre
    pen_length_m: float = Field(default=0.153, gt=0.0)
    # 20 mm: 55-57 px across in the right wrist view, as the real marker; the twin's fingers
    # settle at stroke 22.2 on it (real holds: p10 19.1 / median 21.2 / p90 23.2).
    pen_diameter_m: float = Field(default=0.020, gt=0.0)
    pen_mass_kg: float = Field(default=0.015, gt=0.0)
    pen_friction: _Vec3 = (1.2, 0.01, 0.0005)
    pen_rgba: _Vec4 = (0.012, 0.012, 0.014, 1.0)  # glossy black barrel
    # The printed sleeve (both white rings, the label text, the grey tip section), unwrapped
    # from a real right-wrist frame; None leaves the barrel plain black.
    pen_label_texture: str | None = "pen_label.png"
    pen_label_emission: float = Field(default=0.5, ge=0.0)  # the camera-baked print, self-lit
    pen_specular: float = Field(default=0.9, ge=0.0, le=1.0)
    pen_shininess: float = Field(default=0.85, ge=0.0, le=1.0)
    pen_reflectance: float = Field(default=0.08, ge=0.0, le=1.0)
    # Start (x, y, yaw_deg) on the desk; yaw 0 = the pen along x. The dataset mean.
    pen_start: _Vec3 = (0.27, -0.318, -1.7)
    # Per-reset uniform jitter of both starts (x / y +-spawn_jitter_m, yaw +-yaw_jitter_deg),
    # drawn from the episode seed; 0 keeps the declared starts.
    spawn_jitter_m: float = Field(default=0.0, ge=0.0)
    yaw_jitter_deg: float = Field(default=0.0, ge=0.0)

    # ---- graded success
    rest_speed_m_s: float = Field(default=0.02, gt=0.0)
    settle_s: float = Field(default=0.3, ge=0.0)

    # ---- arm look / drives (the single-arm twin's fits; the two arms are the same kit)
    # Dark anodised grey, a touch green as the real links read (top (32, 35, 33), wrist
    # (38-44, 41-48, 38-48)); lit, it reads ~40 in the wrists.
    arm_rgba: _Vec4 = (0.30, 0.33, 0.32, 1.0)  # +15 %: right wrist 0.72x, left 0.89x of real
    arm_specular: float = 0.5
    arm_shininess: float = 0.55
    arm_gravcomp: bool = True
    finger_mesh_scale: _Vec3 = (0.676, 1.0, 0.93)
    finger_kp: float = Field(default=3000.0, gt=0.0)
    finger_kv: float = Field(default=30.0, ge=0.0)
    elliptic_friction: bool = True
    impratio: float = Field(default=10.0, gt=0.0)

    # ---- camera corrections on top of the manifest's sim_placement: name -> (dx, dy, dz) m,
    # (rx, ry, rz) deg in the camera frame. Fitted per wrist on the pen-holder dataset (both
    # arms posed at the recorded state, edge-map correlation over 4 episodes x 9-11 ticks from
    # rest to grasp): left 0.59 -> 0.77, right 0.56 -> 0.64. The manifest's wrist mount is the
    # single-arm rig's refit.
    camera_offsets: dict[str, tuple[float, float, float, float, float, float]] = Field(
        default_factory=lambda: {
            "wrist_left": (0.0088, 0.0073, -0.0169, -3.32, -1.96, -2.40),
            "wrist_right": (-0.0035, 0.0005, 0.0120, 1.13, -2.46, 1.25),
        }
    )

    # ---- top camera background
    top_background: str = Field(default="plate", pattern=r"^(plate|render)$")
    top_plate: str = "top_plate.jpg"  # file under assets/, or absolute
    top_mask_feather_px: int = Field(default=1, ge=0)

    # ---- lighting (wrist views: shadowed grid; top: brighter, shadow-free)
    headlight_ambient: _Vec3 = (0.36, 0.36, 0.36)
    headlight_diffuse: _Vec3 = (0.0, 0.0, 0.0)
    light_pos: _Vec3 = (0.25, 0.0, 1.0)
    light_diffuse: _Vec3 = (2.0, 2.0, 2.0)
    light_castshadow: bool = True
    light_bulb_radius_m: float = 0.25
    light_grid: int = Field(default=6, ge=1, le=8)
    light_spread_m: float = Field(default=0.8, ge=0.0)
    light_cutoff_deg: float = Field(default=80.0, gt=0.0, le=90.0)
    light_exponent: float = Field(default=0.0, ge=0.0)
    top_headlight_ambient: _Vec3 = (0.55, 0.55, 0.55)
    top_headlight_diffuse: _Vec3 = (0.30, 0.30, 0.30)
    top_light_gain: float = Field(default=0.27, ge=0.0)
    # Per-material colour gains for the top render only (the region means of the composited
    # arms / cup against the real top frames: arms 50 vs 33, cup 166 vs 140).
    top_material_gain: dict[str, float] = Field(
        default_factory=lambda: {"anodised_black": 0.565, "holder": 0.85}
    )
    # Per-camera material gains for the wrist renders: the real left wrist sees the desk at
    # 0.65x the right wrist's brightness (arm shadow + exposure; region means 65 vs 96).
    wrist_material_gain: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {"wrist_left": {"wood": 0.65}}
    )

    @model_validator(mode="after")
    def _check(self) -> MetalPenHolderOptions:
        if len(self.initial_joint_positions) != _POLICY_DIM:
            raise ValueError("initial_joint_positions must be 14-D (2 x (6 joints + gripper))")
        if self.holder_top_width_m < self.holder_bottom_width_m:
            raise ValueError("holder_top_width_m must be >= holder_bottom_width_m (it tapers out)")
        return self

    def asset_path(self, name: str) -> Path:
        """Resolve an asset option: absolute path as is, else under ``ASSET_DIR``."""
        p = Path(name)
        return p if p.is_absolute() else ASSET_DIR / p


def holder_inner_half_width(opts: MetalPenHolderOptions, z: float) -> float:
    """Inner half-width of the cup at height ``z`` above its bottom (the walls lean out).

    Example:
        >>> o = MetalPenHolderOptions()
        >>> round(holder_inner_half_width(o, 0.0), 4), round(holder_inner_half_width(o, 0.1), 4)
        (0.031, 0.047)
    """
    hb, ht = opts.holder_bottom_width_m / 2.0, opts.holder_top_width_m / 2.0
    frac = min(max(z / opts.holder_height_m, 0.0), 1.0)
    return hb + (ht - hb) * frac - opts.holder_wall_m


def _resolve_robot_mjcf(description: RobotDescription) -> str:
    from openral_core.assets import AssetRefError, resolve_asset

    if not description.assets.mjcf:
        raise ROSConfigError(f"metal_pen_holder: robot {description.name!r} has no assets.mjcf.")
    try:
        path = resolve_asset(description.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(f"metal_pen_holder: {exc}") from exc
    if path is None:
        raise ROSConfigError(
            f"metal_pen_holder: assets.mjcf={description.assets.mjcf!r} did not resolve."
        )
    return str(path)


def compose_metal_pen_holder(
    description: RobotDescription,
    options: MetalPenHolderOptions | None = None,
    *,
    width: int = 640,
    height: int = 480,
) -> mujoco.MjModel:
    """Compose and compile the scene around the station of ``description``.

    Args:
        description: The ``makermods_metal_station`` manifest (its ``assets.mjcf`` and
            camera sensors).
        options: Scene options; ``None`` uses the measured defaults.
        width: Render width the cameras' intrinsics are scaled to.
        height: Render height.

    Raises:
        ROSConfigError: ``mujoco`` missing, the station MJCF lacks an expected body, a
            camera sensor lacks ``sim_placement`` / ``intrinsics``, or the model fails to
            compile.
    """
    opts = options if options is not None else MetalPenHolderOptions()
    try:
        import mujoco as mj  # reason: optional sim-only dep
    except ModuleNotFoundError as exc:
        raise ROSConfigError(
            "mujoco is not installed; install the sim extras: just sync --group sim"
        ) from exc

    path = _resolve_robot_mjcf(description)
    spec = mj.MjSpec.from_file(path)
    bodies = {b.name: b for b in spec.bodies}
    want = [p + b for p in ARMS for b in ARM_BODIES]
    missing = [b for b in want if b not in bodies]
    if missing:
        raise ROSConfigError(f"metal_pen_holder: station MJCF {path} lacks bodies {missing}.")

    for mesh in spec.meshes:
        if mesh.name in ("link7", "link8"):
            mesh.scale = list(opts.finger_mesh_scale)
    for act in spec.actuators:
        if act.name.endswith("joint8"):
            act.gainprm[0] = opts.finger_kp
            act.biasprm[1] = -opts.finger_kp
            act.biasprm[2] = -opts.finger_kv
    if opts.elliptic_friction:
        # Elliptic friction cones + a high impratio: grasps that do not creep (a held pen
        # or cup slid out of the 3 N grip under the default pyramidal cones).
        spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = opts.impratio
    spec.visual.headlight.ambient = list(opts.headlight_ambient)
    spec.visual.headlight.diffuse = list(opts.headlight_diffuse)
    spec.visual.quality.shadowsize = 4096

    arm_mat = spec.add_material()
    arm_mat.name = "anodised_black"
    arm_mat.rgba = list(opts.arm_rgba)
    arm_mat.specular = opts.arm_specular
    arm_mat.shininess = opts.arm_shininess
    for name in want:
        if opts.arm_gravcomp:
            bodies[name].gravcomp = 1.0
        for geom in bodies[name].geoms:
            if geom.type == mj.mjtGeom.mjGEOM_MESH:
                geom.name = geom.name or f"{name}_mesh"
                geom.material = "anodised_black"
                geom.rgba = list(opts.arm_rgba)

    _add_station(spec, opts)
    _add_holder(spec, opts)
    _add_pen(spec, opts)
    _add_cameras(spec, description, width, height, opts.camera_offsets)
    add_light_grid(
        spec,
        pos=opts.light_pos,
        diffuse=opts.light_diffuse,
        grid=opts.light_grid,
        spread_m=opts.light_spread_m,
        bulb_radius_m=opts.light_bulb_radius_m,
        cutoff_deg=opts.light_cutoff_deg,
        exponent=opts.light_exponent,
        castshadow=opts.light_castshadow,
    )
    try:
        return spec.compile()
    except ValueError as exc:
        raise ROSConfigError(f"metal_pen_holder: composed model failed to compile: {exc}") from exc


def _material(spec: mujoco.MjSpec, name: str, rgba: _Vec4, emission: float) -> str:
    mat = spec.add_material()
    mat.name = name
    mat.rgba = list(rgba)
    mat.emission = emission
    return name


def _add_station(spec: mujoco.MjSpec, opts: MetalPenHolderOptions) -> None:
    import mujoco as mj

    (x0, x1), (y0, y1) = opts.table_x_range, opts.table_y_range
    half_t = opts.table_thickness / 2.0
    table = _static_geom(
        spec,
        "table",
        mj.mjtGeom.mjGEOM_BOX,
        ((x0 + x1) / 2, (y0 + y1) / 2, opts.table_top_z - half_t),
        ((x1 - x0) / 2, (y1 - y0) / 2, half_t),
        opts.table_rgba,
    )
    table.friction = [1.0, 0.01, 0.001]
    if opts.table_texture is not None:
        tex = spec.add_texture()
        tex.name = "wood"
        tex.type = mj.mjtTexture.mjTEXTURE_2D
        tex.file = str(opts.asset_path(opts.table_texture))
        mat = spec.add_material()
        mat.name = "wood"
        mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "wood"
        mat.texuniform = True  # one period spans 2 / texrepeat metres
        mat.texrepeat = [2.0 / opts.table_texture_size_m[1], 2.0 / opts.table_texture_size_m[0]]
        mat.specular = 0.1
        mat.shininess = 0.2
        mat.rgba = list(opts.table_rgba)
        mat.emission = opts.table_emission
        table.material = "wood"

    (sx0, sx1), (sy0, sy1) = opts.shelf_x_range, opts.shelf_y_range
    shelf_h = (opts.shelf_top_z - opts.table_top_z) / 2.0
    shelf = _static_geom(
        spec,
        "shelf",
        mj.mjtGeom.mjGEOM_BOX,
        ((sx0 + sx1) / 2, (sy0 + sy1) / 2, opts.table_top_z + shelf_h),
        ((sx1 - sx0) / 2, (sy1 - sy0) / 2, shelf_h),
        opts.shelf_rgba,
    )
    shelf.friction = [1.0, 0.01, 0.001]
    shelf.material = _material(spec, "shelf", opts.shelf_rgba, opts.shelf_emission)
    if opts.shelf_texture is not None:  # the real board, cut from the top plate
        _textured_box(
            spec,
            "shelf_top",
            opts.asset_path(opts.shelf_texture),
            ((sx0 + sx1) / 2, (sy0 + sy1) / 2, opts.shelf_top_z + 0.0002),
            ((sx1 - sx0) / 2, (sy1 - sy0) / 2),
            opts.shelf_texture_gain,
            opts.shelf_emission,
        )

    wall = _static_geom(
        spec,
        "wall",
        mj.mjtGeom.mjGEOM_BOX,
        (opts.wall_x + 0.01, 0.0, 0.3),
        (0.01, 1.5, 1.1),
        opts.wall_rgba,
    )
    wall.material = _material(spec, "wall", opts.wall_rgba, opts.wall_emission)
    floor = _static_geom(
        spec,
        "floor",
        mj.mjtGeom.mjGEOM_PLANE,
        (0.0, 0.0, opts.floor_z),
        (0, 0, 0.05),
        opts.floor_rgba,
    )
    props = [wall, floor]
    for k, item in enumerate(opts.clutter):
        kind = mj.mjtGeom.mjGEOM_BOX if item.kind == "box" else mj.mjtGeom.mjGEOM_CYLINDER
        geom = _static_geom(spec, f"clutter_{k}", kind, item.pos, item.size, item.rgba)
        if item.emission:
            geom.material = _material(spec, f"clutter_{k}", item.rgba, item.emission)
        props.append(geom)
    for geom in props:  # visual only: the arms never need to touch these
        geom.contype = 0
        geom.conaffinity = 0


def _add_holder(spec: mujoco.MjSpec, opts: MetalPenHolderOptions) -> None:
    """The cup: a square floor + four walls leaning out to the wider rim, one free body.

    Colliders are invisible (group 3); the closed tapered tube of ``_add_holder_visual`` is
    the rendered cup. An upright 79 mm collider (the grip width) caught 5 of 13 replayed
    pens against 9 of 13 for this tapered one: the pen drops near the rim, where the real
    cup is 94 mm wide inside.
    """
    import mujoco as mj

    hb, ht = opts.holder_bottom_width_m / 2.0, opts.holder_top_width_m / 2.0
    t, h = opts.holder_wall_m, opts.holder_height_m
    tilt = math.atan2(ht - hb, h)
    slant = math.hypot(h, ht - hb)
    x, y, yaw = opts.holder_start
    body = spec.worldbody.add_body()
    body.name = "pen_holder"
    body.pos = [x, y, opts.shelf_top_z]
    body.quat = [math.cos(math.radians(yaw) / 2), 0.0, 0.0, math.sin(math.radians(yaw) / 2)]
    body.add_freejoint().name = "pen_holder_joint"
    mat = _material(spec, "holder", opts.holder_rgba, opts.holder_emission)
    n_geoms = 5
    parts: list[tuple[str, _Vec3, _Vec3, tuple[float, float, float, float]]] = [
        (
            "holder_floor",
            (0.0, 0.0, opts.holder_floor_m / 2),
            (hb, hb, opts.holder_floor_m / 2),
            (1.0, 0.0, 0.0, 0.0),
        )
    ]
    for i in range(4):
        a = math.pi / 2 * i
        r = (hb + ht) / 2.0 - t / 2.0  # the wall's mid-plane, halfway up
        # Rz(a) * Ry(tilt): the wall's top leans out, away from the cup's axis.
        qz = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
        qy = (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0)
        # As wide as the bottom: the corners close there and open to a slit of
        # (top - bottom) / 2 at the rim, narrower than the pen.
        parts.append(
            (
                f"holder_wall_{i}",
                (r * math.cos(a), r * math.sin(a), h / 2),
                (t / 2, hb + t, slant / 2),
                _quat_mul(qz, qy),
            )
        )
    for name, pos, size, quat in parts:
        g = body.add_geom()
        g.name = name
        g.type = mj.mjtGeom.mjGEOM_BOX
        g.pos = list(pos)
        g.size = list(size)
        g.quat = list(quat)
        g.mass = opts.holder_mass_kg / n_geoms
        g.friction = list(opts.holder_friction)
        g.condim = 4
        g.group = 3  # not rendered: the closed visual tube below is the cup's look
    _add_holder_visual(spec, body, opts, mat)


def holder_hex_texture(size: int = 256) -> NDArray[np.uint8]:
    """The cup's look: white plastic perforated by rows of hexagonal holes (RGB, size^2).

    u runs across a wall, v up it (row 0 = the rim). Solid bands at the rim (15 %) and the
    bottom (12 %) as on the real cup; ~7.5 holes across with thin webs (the wrist crop of
    the real cup). A hole shows the cup's inside: light grey high up, darker low down.

    Example:
        >>> img = holder_hex_texture(64)
        >>> img.shape, int(img[0, 0, 0]), int(img.min()) < 200
        ((64, 64, 3), 255, True)
    """
    import numpy as np

    v, u = np.mgrid[0:size, 0:size] / float(size)
    dx = 1.0 / 7.5
    dy = dx * math.sqrt(3.0) / 2.0
    row = np.floor(v / dy)
    cx = (np.floor((u - (row % 2) * dx / 2) / dx) + 0.5) * dx + (row % 2) * dx / 2
    cy = (row + 0.5) * dy
    ox, oy = np.abs(u - cx), np.abs(v - cy)
    r = 0.42 * dx  # hole radius (corner), the rest is the web between holes
    hole = (oy <= r * math.sqrt(3) / 2) & (ox * math.sqrt(3) / 2 + oy / 2 <= r * math.sqrt(3) / 2)
    rim_band, bottom_band = 0.15, 0.12  # solid plastic at the rim and the bottom
    hole &= (v > rim_band) & (v < 1.0 - bottom_band)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    shade = np.clip((v - rim_band) / (1.0 - rim_band - bottom_band), 0.0, 1.0)[..., None]
    inside = (1.0 - shade) * np.array([165.0, 165.0, 172.0]) + shade * np.array([92.0, 92.0, 98.0])
    img[hole] = inside[hole].astype(np.uint8)
    return img


def _texture_holder_material(spec: mujoco.MjSpec, material: str) -> None:
    """Put the hex-perforation texture (``holder_hex_texture``) on the cup's material."""
    import mujoco as mj

    img = holder_hex_texture()
    tex = spec.add_texture()
    tex.name = "holder_hex"
    tex.type = mj.mjtTexture.mjTEXTURE_2D
    tex.height, tex.width, tex.nchannel = img.shape
    tex.data = img.tobytes()
    mat = next(m for m in spec.materials if m.name == material)
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "holder_hex"
    mat.specular = 0.3
    mat.shininess = 0.3


def _rounded_square(half: float, radius: float, n_corner: int) -> list[tuple[float, float]]:
    """A rounded square's outline (counter-clockwise), ``n_corner`` + 1 points per corner."""
    pts: list[tuple[float, float]] = []
    c = half - radius
    for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        for i in range(n_corner + 1):
            a = math.pi / 2 * (k + i / n_corner)
            pts.append((sx * c + radius * math.cos(a), sy * c + radius * math.sin(a)))
    return pts


def _add_holder_visual(
    spec: mujoco.MjSpec, body: mujoco.MjsBody, opts: MetalPenHolderOptions, material: str
) -> None:
    """A closed tapered rounded-square tube (outer + inner walls, rim, floor), visual only.

    The collision walls stand apart at the corners (slits narrower than the pen); the
    rendered cup is the real cup's closed shape with rounded corners
    (``holder_corner_radius_m`` at the rim), its walls textured with the hexagonal
    perforation (``holder_hex_texture``, once per side). MuJoCo draws a visual mesh's own
    faces, with smooth normals around the ring.
    """
    import mujoco as mj

    hb, ht = opts.holder_bottom_width_m / 2.0, opts.holder_top_width_m / 2.0
    t, h, f = opts.holder_wall_m, opts.holder_height_m, opts.holder_floor_m
    nc = 6
    rr = opts.holder_corner_radius_m

    def ring(half: float, z: float) -> list[tuple[float, float, float]]:
        return [(x, y, z) for x, y in _rounded_square(half, rr * half / ht, nc)]

    rings = [ring(hb, 0.0), ring(ht, h), ring(ht - t, h), ring(hb - t, f)]
    n = len(rings[0])
    # u: perimeter fraction x 4 (the texture once per side), seam vertex repeated.
    seg = [math.dist(rings[1][i][:2], rings[1][(i + 1) % n][:2]) for i in range(n)]
    u = [0.0]
    for d in seg:
        u.append(u[-1] + 4.0 * d / sum(seg))
    vert: list[float] = []
    uv: list[float] = []
    face: list[int] = []

    def band(
        lo: list[tuple[float, float, float]],
        hi: list[tuple[float, float, float]],
        v: tuple[float, float],
    ) -> None:
        base = len(vert) // 3
        for pts, vv in ((lo, v[0]), (hi, v[1])):
            for i in range(n + 1):
                vert.extend(pts[i % n])
                uv.extend((u[i], vv))
        for i in range(n):
            a, b = base + i, base + n + 1 + i
            face.extend((a, a + 1, b + 1, a, b + 1, b))

    band(rings[0], rings[1], (1.0, 0.0))  # outer wall (texture row 0 = the rim)
    band(rings[1], rings[2], (0.0, 0.0))  # rim: the solid band
    band(rings[2], rings[3], (0.0, 1.0))  # inner wall
    for pts, z_up in ((rings[0], False), (rings[3], True)):  # bottom (down), inner floor (up)
        base = len(vert) // 3
        cz = pts[0][2]
        vert.extend((0.0, 0.0, cz))
        uv.extend((0.0, 0.95))
        for p in pts:
            vert.extend(p)
            uv.extend((0.0, 0.95))
        for i in range(n):
            a, b = base + 1 + i, base + 1 + (i + 1) % n
            face.extend((base, a, b) if z_up else (base, b, a))
    mesh = spec.add_mesh()
    mesh.name = "holder_visual"
    mesh.uservert = vert
    mesh.usertexcoord = uv
    mesh.userface = face
    mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
    _texture_holder_material(spec, material)
    g = body.add_geom()
    g.name = "holder_visual"
    g.type = mj.mjtGeom.mjGEOM_MESH
    g.meshname = "holder_visual"
    g.material = material
    g.contype = 0
    g.conaffinity = 0
    g.density = 0.0


def _add_pen(spec: mujoco.MjSpec, opts: MetalPenHolderOptions) -> None:
    """The marker: a square-section box collider (it does not roll) + a visual cylinder."""
    import mujoco as mj

    half_l, r = opts.pen_length_m / 2.0, opts.pen_diameter_m / 2.0
    x, y, yaw = opts.pen_start
    body = spec.worldbody.add_body()
    body.name = "pen"
    body.pos = [x, y, opts.table_top_z + r]
    body.quat = [math.cos(math.radians(yaw) / 2), 0.0, 0.0, math.sin(math.radians(yaw) / 2)]
    body.add_freejoint().name = "pen_joint"
    col = body.add_geom()
    col.name = "pen"
    col.type = mj.mjtGeom.mjGEOM_BOX
    col.size = [half_l, r, r]
    col.mass = opts.pen_mass_kg
    col.friction = list(opts.pen_friction)
    col.condim = 4
    col.group = 3  # not rendered: the cylinder below is the pen's look
    along_x = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
    mat = spec.add_material()
    mat.name = "pen_body"
    mat.rgba = list(opts.pen_rgba)
    mat.specular = opts.pen_specular
    mat.shininess = opts.pen_shininess
    mat.reflectance = opts.pen_reflectance
    g = body.add_geom()
    g.name = "pen_body"
    g.type = mj.mjtGeom.mjGEOM_CYLINDER
    g.quat = along_x
    g.size = [r, half_l, 0.0]
    g.material = "pen_body"
    g.contype = 0
    g.conaffinity = 0
    g.density = 0.0
    if opts.pen_label_texture is not None:
        _add_pen_sleeve(spec, body, opts)


# The sleeve spans 27 %..81.5 % of the length from the cap end (+x); its texture covers the
# +-60 deg face the camera saw, so it repeats 3x around the barrel.
_PEN_SLEEVE_SPAN = (0.27, 0.815)
_PEN_SLEEVE_WRAPS = 3.0


def _add_pen_sleeve(spec: mujoco.MjSpec, body: mujoco.MjsBody, opts: MetalPenHolderOptions) -> None:
    """A textured open cylinder just outside the barrel: the real marker's print, visual only."""
    import mujoco as mj

    half_l, r = opts.pen_length_m / 2.0, opts.pen_diameter_m / 2.0 * 1.01
    x_hi, x_lo = (half_l - f * opts.pen_length_m for f in _PEN_SLEEVE_SPAN)
    n = 32
    vert: list[float] = []
    uv: list[float] = []
    face: list[int] = []
    for i in range(n + 1):  # seam vertex repeated
        a = 2.0 * math.pi * i / n
        for x, v in ((x_hi, 0.0), (x_lo, 1.0)):
            vert.extend((x, r * math.cos(a), r * math.sin(a)))
            uv.extend((_PEN_SLEEVE_WRAPS * i / n, v))
    for i in range(n):
        a, b = 2 * i, 2 * i + 2
        face.extend((a, a + 1, b + 1, a, b + 1, b))
    mesh = spec.add_mesh()
    mesh.name = "pen_sleeve"
    mesh.uservert = vert
    mesh.usertexcoord = uv
    mesh.userface = face
    mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
    tex = spec.add_texture()
    tex.name = "pen_label"
    tex.type = mj.mjtTexture.mjTEXTURE_2D
    tex.file = str(opts.asset_path(opts.pen_label_texture))
    mat = spec.add_material()
    mat.name = "pen_label"
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "pen_label"
    mat.specular = 0.3
    mat.shininess = 0.5
    mat.emission = opts.pen_label_emission
    g = body.add_geom()
    g.name = "pen_label"
    g.type = mj.mjtGeom.mjGEOM_MESH
    g.meshname = "pen_sleeve"
    g.material = "pen_label"
    g.contype = 0
    g.conaffinity = 0
    g.density = 0.0


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _add_cameras(
    spec: mujoco.MjSpec,
    description: RobotDescription,
    width: int,
    height: int,
    offsets: dict[str, tuple[float, float, float, float, float, float]] | None = None,
) -> None:
    """Every manifest RGB sensor as a MuJoCo camera: its ``sim_placement`` + pinhole.

    ``offsets`` (``MetalPenHolderOptions.camera_offsets``) moves a named camera by
    ``(dx, dy, dz)`` metres and turns it by ``(rx, ry, rz)`` degrees (XYZ), both in its own
    frame (MuJoCo: x right, y up, looking down -z).
    """
    from scipy.spatial.transform import Rotation

    offsets = offsets or {}
    bodies = {b.name: b for b in spec.bodies}
    pixel_m = 3e-6  # nominal pixel pitch; only the pixel ratios matter
    for s in description.sensors:
        sp, k = s.sim_placement, s.intrinsics
        if getattr(s.modality, "value", s.modality) != "rgb":
            continue
        if sp is None or k is None or sp.parent_body not in bodies:
            raise ROSConfigError(
                f"metal_pen_holder: camera {s.name!r} needs sim_placement on a station body "
                "and pinhole intrinsics."
            )
        cam = bodies[sp.parent_body].add_camera()
        cam.name = s.name
        w, x, y, z = look_at_quat_wxyz(sp.pos, sp.target, up=sp.up)
        rot = Rotation.from_quat([x, y, z, w])
        pos = list(sp.pos)
        if s.name in offsets:
            o = offsets[s.name]
            pos = [float(v) for v in sp.pos + rot.apply(o[:3])]
            rot = rot * Rotation.from_euler("xyz", o[3:], degrees=True)
        qx, qy, qz, qw = rot.as_quat()
        cam.pos = pos
        cam.quat = [qw, qx, qy, qz]
        sx, sy = width / k.width, height / k.height
        cam.resolution = [width, height]
        cam.sensor_size = [width * pixel_m, height * pixel_m]
        cam.focal_pixel = [k.fx * sx, k.fy * sy]
        cam.principal_pixel = list(mujoco_principal_pixel(k.cx * sx, k.cy * sy, width, height))
