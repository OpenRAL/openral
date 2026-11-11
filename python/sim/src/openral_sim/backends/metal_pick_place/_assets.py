"""Compose the ``metal_pick_place`` MJCF — the MakerMods Metal pick-and-place rig.

The scene reproduces the real rig the ``HappyEthan/metal_pick_place`` LeRobot
dataset was recorded on (and that ``makermods/pi05_metal_pick_place_lora`` was
trained on): a black Metal arm on a dark wood desk against a light-grey wall, a
3 cm orange cube, a white Ø3.9 cm marker sticker, a black desk grommet, a white
rolling cart to the right and the teleoperation leader arm to the left. Every
pose, size and colour below was calibrated against that dataset's top and wrist
videos (camera fit residual ~3 px); they are all ``backend_options`` fields so a
re-measured rig is a YAML edit.

The robot MJCF comes from the manifest's ``assets.mjcf`` (``makermods:metal``)
and is loaded into an ``mujoco.MjSpec``; the task world is appended, so the arm
keeps actuator / qpos indices 0..6 / 0..7 the manifest's ``sim`` block declares.
The leader arm is a set of contact-free *mocap* bodies carrying copies of the
arm's meshes: it adds no joints and no actuators, so it can never perturb the
follower's index contract.

Photorealism: the static top camera is rendered with the real camera intrinsics
(principal point included) and, by default, composited over a real background
plate (``assets/top_plate.jpg``) — only robot / leader / cube / marker pixels
come from MuJoCo. The wrist camera moves, so it is fully rendered, with a table
texture cut from the same plate (``assets/table_wood.png``). See
``assets/SOURCE.txt`` for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import mujoco
    from openral_core import RobotDescription

__all__ = [
    "ARM_BODIES",
    "ASSET_DIR",
    "ClutterItem",
    "MetalPickPlaceOptions",
    "PhotoDecal",
    "add_light_grid",
    "compose_metal_pick_place",
    "mujoco_principal_pixel",
]

#: Bundled image assets (background plate, table texture) + their SOURCE.txt.
ASSET_DIR: Path = Path(__file__).resolve().parent / "assets"

#: Bodies of the Metal arm MJCF that carry a mesh, root first (the leader copy
#: and the top-camera composite mask are built from these).
ARM_BODIES: tuple[str, ...] = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "link7",
    "link8",
)

_POLICY_DIM = 7  # joint1..6 + gripper
_Vec2 = tuple[float, float]
_Vec3 = tuple[float, float, float]
_Vec4 = tuple[float, float, float, float]

# Cube start -> marker target per real-rig layout (follower base frame, metres),
# fit on the dataset's top video. Episode ranges: A 0-29, B 30-49, C 50-68,
# D 69-99.
_DEFAULT_LAYOUTS: dict[str, tuple[_Vec2, _Vec2]] = {
    "A": ((0.267, -0.069), (0.278, 0.076)),
    "B": ((0.270, 0.187), (0.276, 0.068)),
    "C": ((0.210, 0.073), (0.297, 0.064)),
    "D": ((0.290, 0.024), (0.285, -0.081)),
}


class ClutterItem(BaseModel):
    """A static, contact-free prop (the white cart / wall pipes beside the desk).

    Attributes:
        kind: ``box`` (``size`` = half extents) or ``cylinder`` (``size`` =
            ``(radius, half_length, unused)``, axis vertical).
        pos: Centre in the follower base frame, metres.
        size: See ``kind``.
        rgba: Colour.
        emission: Self-illumination (the overhead lights graze vertical props).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["box", "cylinder"] = "box"
    pos: _Vec3
    size: _Vec3
    rgba: _Vec4 = (0.86, 0.87, 0.86, 1.0)
    emission: float = 0.0


class PhotoDecal(BaseModel):
    """A real photo laid on a horizontal plane (contact-free, visual only).

    The image is a rectified mosaic of the dataset's wrist frames (see
    ``assets/SOURCE.txt``): texel row = world y descending, column = world x
    ascending (MuJoCo's 2D-texture convention on a box top). Any imageio format.

    Attributes:
        file: Image under ``assets/`` or an absolute path.
        x_range: World x extent the image covers, metres.
        y_range: World y extent, metres.
        z: Height of the plane, metres.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    x_range: _Vec2
    y_range: _Vec2
    z: float


_DEFAULT_CLUTTER: tuple[ClutterItem, ...] = (
    # Two white pipes (Ø 28 mm) standing on the floor against the wall right of
    # the desk, tops 0.139 m above the desk; located in the rest-pose wrist frames.
    ClutterItem(
        kind="cylinder",
        pos=(0.531, -0.213, -0.30),
        size=(0.014, 0.44, 0.0),
        rgba=(0.67, 0.70, 0.69, 1.0),
        emission=0.05,
    ),
    ClutterItem(
        kind="cylinder",
        pos=(0.531, -0.262, -0.30),
        size=(0.014, 0.44, 0.0),
        rgba=(0.67, 0.70, 0.69, 1.0),
        emission=0.05,
    ),
)


class MetalPickPlaceOptions(BaseModel):
    """Every dimension, pose, colour and unit the ``metal_pick_place`` scene exposes.

    Lengths are metres in the follower ``base_link`` frame (x forward toward the
    wall, z up), angles degrees where suffixed ``_deg``. Defaults are the
    calibrated real rig.

    Units: the policy-facing ``state`` / action are 7-D (joint1..joint6 then the
    gripper). ``joint_units="degrees"`` with ``gripper_scale=100`` is the
    dataset's convention (arm joints in degrees with the URDF's sign and zero,
    gripper = stroke ~0..100, joint8 metres = stroke / 2000); the scene converts
    exactly once, here, so a degree rSkill runs unconverted through ``sim run``
    (the scene registers ``converts_policy_units=True``). ``radians`` + ``1.0``
    is the manifest's own unit surface (gripper normalised ``[0, 1]``).

    Example:
        >>> MetalPickPlaceOptions().layout
        'A'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- units / control
    joint_units: Literal["radians", "degrees"] = "degrees"
    gripper_scale: float = Field(default=100.0, gt=0.0)
    # The fully closed real gripper reads stroke ~2.5, not 0: joint8 = stroke/2000
    # - this, so a commanded stroke squeezes the 25 mm cube as hard as on the real
    # arm (real holds read 25-30) instead of stopping short of it.
    gripper_offset_m: float = Field(default=0.00125, ge=0.0)
    control_hz: float = Field(default=30.0, gt=0.0)
    # Dataset median episode-start state (policy units, 7-D incl. gripper).
    initial_joint_positions: tuple[float, ...] = (0.74, 0.04, -0.05, 0.49, -0.10, -1.38, 2.5)
    settle_steps: int = Field(default=50, ge=0)

    # ---- desk / wall / floor
    table_top_z: float = 0.003
    table_x_range: _Vec2 = (-0.30, 0.535)
    table_y_range: _Vec2 = (-0.149, 0.87)
    table_thickness: float = 0.025
    table_rgba: _Vec4 = (1.0, 1.0, 1.0, 1.0)  # multiplies the wood texture
    # Shadow-free share of the desk brightness: the rest comes from the shadow
    # lights, so this sets how dark the soft gripper shadow gets.
    table_emission: float = Field(default=0.3, ge=0.0)
    table_texture: str | None = "table_wood.png"  # file under assets/, or absolute
    # One seamless tile of real desk wood, 150 x 150 mm at 0.4 mm/texel: four
    # near-nadir wrist-frame patches rectified with the fitted camera, de-shaded
    # and cross-faded (see assets/SOURCE.txt).
    table_texture_size_m: _Vec2 = (0.1504, 0.1504)  # one texture tile along (y, x)
    # Optional real-photo decal over the desk (a rectified mosaic); off by
    # default: the uniform tile above reads cleaner than a blotchy mosaic.
    table_photo: PhotoDecal | None = None
    # The white cart right of the desk: wrist frames rectified onto a plane over it.
    cart_photo: PhotoDecal | None = PhotoDecal(
        file="cart_photo.jpg", x_range=(0.04, 0.60), y_range=(-0.50, -0.149), z=-0.02
    )
    # Material gain / shadow-free emission of the photo textures (marker sticker,
    # cart): rendered ~= photo pixel under the lights below.
    photo_gain: float = Field(default=1.0, gt=0.0)
    photo_emission: float = Field(default=0.45, ge=0.0)
    wall_x: float = 0.545
    wall_rgba: _Vec4 = (0.76, 0.79, 0.76, 1.0)
    wall_emission: float = 0.26
    floor_z: float = -0.74
    floor_rgba: _Vec4 = (0.35, 0.34, 0.33, 1.0)

    # ---- objects
    # 25 mm: the wrist fit puts a 30 mm cube 1.23x too large in frame (25 mm: 1.00).
    cube_edge_m: float = Field(default=0.025, gt=0.0)
    cube_mass_kg: float = Field(default=0.01, gt=0.0)
    cube_friction: _Vec3 = (1.2, 0.01, 0.0005)
    # Matched to the real wrist frames (cube region mean RGB ~ (207, 125, 84)).
    cube_rgba: _Vec4 = (1.0, 0.60, 0.41, 1.0)
    cube_emission: float = 0.29
    marker_diameter_m: float = 0.039
    marker_rgba: _Vec4 = (0.93, 0.93, 0.92, 1.0)
    # Real sticker, rectified from a wrist frame and cropped to the disc; the
    # marker becomes a textured disc. None: plain white disc.
    marker_texture: str | None = "marker_sticker.png"
    grommet_xy: _Vec2 = (0.446, 0.004)
    grommet_diameter_m: float = 0.060
    clutter: tuple[ClutterItem, ...] = _DEFAULT_CLUTTER

    # ---- task layout + randomisation
    layout: str = "A"
    layouts: dict[str, tuple[_Vec2, _Vec2]] = Field(default_factory=lambda: dict(_DEFAULT_LAYOUTS))
    spawn_jitter_m: float = Field(default=0.015, ge=0.0)
    cube_yaw_deg: float = 0.0  # cube yaw before jitter (world z)
    cube_yaw_jitter_deg: float = Field(default=10.0, ge=0.0)
    # Graded success (``env.placement_grade``): the cube is judged once released and
    # below this speed for ``settle_s``.
    rest_speed_m_s: float = Field(default=0.02, gt=0.0)
    settle_s: float = Field(default=0.3, ge=0.0)

    # ---- arm look
    arm_rgba: _Vec4 = (0.12, 0.17, 0.14, 1.0)  # real wrist finger pixels ~ (14, 20, 16)
    arm_specular: float = 0.35
    arm_shininess: float = 0.4
    # Gravity compensation on the arm links. The derived MJCF's sim-only servo
    # gains sag ~0.5-1 deg on joint2/3 under gravity, which puts the fingertips
    # into the desk at the dataset's grasp poses; the real Damiao loop holds them.
    arm_gravcomp: bool = True
    # Scale of the two finger meshes (link7 / link8) in their STL frame: x = finger
    # thickness along the opening axis (inner face at x ~ 0, so it stays put),
    # y = finger width, z = length. Applies to collision and visuals.
    # From the real finger silhouettes with the wrist camera held fixed: the
    # upstream finger STL is ~1.5x too thick for the fingers on the dataset's arm
    # (thickness 0.676; edge chamfer 14 -> 6 px, train and held-out). The
    # silhouette-only optimum for the length (0.855) leaves the tips 11 px short
    # and drops the cube during real placements in replay; 0.93 keeps the tips
    # within a few px and holds the cube through all eight replayed episodes.
    finger_mesh_scale: _Vec3 = (0.676, 1.0, 0.93)
    # Finger position-servo stiffness (N/m) and damping; the force stays clamped at
    # the URDF's 3 N. The derived MJCF's 400 N/m lets a blocked finger reach a
    # target 1 mm inside the cube with only 0.4 N; the real servo is stiff.
    finger_kp: float = Field(default=3000.0, gt=0.0)
    finger_kv: float = Field(default=30.0, ge=0.0)

    # ---- leader (teleop) arm: visual-only mocap copy, never actuated
    leader_arm: Literal["mirror_action", "static", "hidden"] = "mirror_action"
    leader_base_xyz: _Vec3 = (0.05, 0.65, 0.0)
    leader_base_yaw_deg: float = 0.0

    # ---- top camera (static, real intrinsics incl. principal point)
    top_camera_pos: _Vec3 = (0.024, 0.371, 0.705)
    top_camera_target: _Vec3 = (0.705, 0.353, -0.027)  # pos + optical axis
    top_camera_up: _Vec3 = (0.732, 0.017, 0.681)
    top_camera_focal_px: float = 373.0
    top_camera_principal_px: _Vec2 = (287.8, 195.0)  # (cx, cy), pinhole convention
    top_camera_resolution: tuple[int, int] = (640, 480)
    # "plate": composite MuJoCo robot/cube/marker pixels over the real plate;
    # "render": the full MuJoCo render.
    top_background: Literal["plate", "render"] = "plate"
    top_plate: str = "top_plate.jpg"  # file under assets/, or absolute
    top_mask_feather_px: int = Field(default=1, ge=0)

    # ---- wrist camera (in gripper_base: +z approach, +x top of gripper)
    # Offline fit on 120 real wrist frames (20 episodes, layouts A-D, rest ->
    # release) against appearance-free features only: desk far/right edges, the
    # marker and cube (back-projected from the top camera) and the grommet.
    # Held-out 24 frames: desk edges 1 / 3 px median, marker 4.4 px, cube 7.7 px
    # (cube position error dominated). Radial distortion was not supported by
    # the data (k1 0.018 +/- 0.007, no held-out gain), so the model is a pinhole.
    wrist_camera_pos: _Vec3 = (0.0677, -0.0004, -0.0134)
    wrist_camera_target: _Vec3 = (0.0056, 0.0003, 0.1768)  # pos + 0.2 * axis
    wrist_camera_up: _Vec3 = (0.9507, -0.0028, 0.3102)
    wrist_camera_fovy_deg: float = 80.1  # only used when wrist_camera_focal_px is None
    # Pinhole intrinsics at ``top_camera_resolution``; when set they replace
    # ``wrist_camera_fovy_deg`` (MuJoCo renders the principal-point offset).
    wrist_camera_focal_px: float | None = 285.6
    wrist_camera_principal_px: _Vec2 | None = (325.8, 249.9)

    # ---- lighting
    # Overhead lighting, matched per region to the real wrist frames (desk mean
    # RGB within 1 %, see the scene README). A 4 x 4 grid of shadow-casting lights
    # gives the soft gripper shadow on the desk -- the cue the pi0.5 LoRA keys on
    # before it opens the gripper (no shadow: 0/30 openings in the four-mix test).
    headlight_ambient: _Vec3 = (0.36, 0.36, 0.36)
    headlight_diffuse: _Vec3 = (0.0, 0.0, 0.0)
    light_pos: _Vec3 = (0.25, 0.0, 1.0)
    light_diffuse: _Vec3 = (2.0, 2.0, 2.0)
    light_castshadow: bool = True
    light_bulb_radius_m: float = 0.25
    light_grid: int = Field(default=6, ge=1, le=8)
    light_spread_m: float = Field(default=0.5, ge=0.0)
    light_cutoff_deg: float = Field(default=80.0, gt=0.0, le=90.0)  # spot cone half-angle
    light_exponent: float = Field(default=0.0, ge=0.0)  # spot falloff (0 = uniform cone)
    # The top view is composited over a real plate and was matched with brighter,
    # shadow-free lighting; it is rendered with these instead of the above.
    top_headlight_ambient: _Vec3 = (0.55, 0.55, 0.55)
    top_headlight_diffuse: _Vec3 = (0.30, 0.30, 0.30)
    top_light_gain: float = Field(default=0.27, ge=0.0)

    extra_metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> MetalPickPlaceOptions:
        if self.layout not in self.layouts:
            raise ValueError(f"layout {self.layout!r} not in layouts {sorted(self.layouts)}")
        if len(self.initial_joint_positions) != _POLICY_DIM:
            raise ValueError("initial_joint_positions must be 7-D (6 joints + gripper)")
        return self

    def asset_path(self, name: str) -> Path:
        """Resolve an asset option: absolute path as is, else under ``ASSET_DIR``."""
        p = Path(name)
        return p if p.is_absolute() else ASSET_DIR / p


def mujoco_principal_pixel(cx: float, cy: float, width: int, height: int) -> tuple[float, float]:
    """Pinhole principal point ``(cx, cy)`` -> MuJoCo ``camera.principal_pixel``.

    MuJoCo's principal pixel is an offset from the image centre with the sign
    flipped on both axes (measured: a point on the optical axis lands at
    ``(W/2 - px, H/2 - py)``), so the pinhole ``cx, cy`` maps to
    ``(W/2 - cx, H/2 - cy)``.

    Example:
        >>> mujoco_principal_pixel(287.8, 195.0, 640, 480)
        (32.19999999999999, 45.0)
    """
    return (width / 2.0 - cx, height / 2.0 - cy)


def _resolve_robot_mjcf(description: RobotDescription) -> str:
    from openral_core.assets import AssetRefError, resolve_asset

    if not description.assets.mjcf:
        raise ROSConfigError(f"metal_pick_place: robot {description.name!r} has no assets.mjcf.")
    try:
        path = resolve_asset(description.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(f"metal_pick_place: {exc}") from exc
    if path is None:
        raise ROSConfigError(
            f"metal_pick_place: assets.mjcf={description.assets.mjcf!r} did not resolve."
        )
    return str(path)


def compose_metal_pick_place(
    description: RobotDescription,
    options: MetalPickPlaceOptions | None = None,
    *,
    width: int = 640,
    height: int = 480,
) -> tuple[mujoco.MjModel, str]:
    """Compose and compile the scene around the Metal arm of ``description``.

    Args:
        description: The ``makermods_metal`` manifest (its ``assets.mjcf``).
        options: Scene options; ``None`` uses the calibrated defaults.
        width: Render width the top camera's intrinsics are scaled to.
        height: Render height.

    Returns:
        ``(model, arm_mjcf_path)`` — the compiled scene and the bare arm MJCF
        (the leader arm's forward kinematics run on a model built from it).

    Raises:
        ROSConfigError: ``mujoco`` missing, the arm MJCF lacks an expected body,
            or the composed model fails to compile.
    """
    opts = options if options is not None else MetalPickPlaceOptions()
    try:
        import mujoco as mj  # reason: optional sim-only dep
    except ModuleNotFoundError as exc:
        raise ROSConfigError(
            "mujoco is not installed; install the sim extras: just sync --group sim"
        ) from exc

    arm_path = _resolve_robot_mjcf(description)
    spec = mj.MjSpec.from_file(arm_path)
    bodies = {b.name: b for b in spec.bodies}
    missing = [b for b in ARM_BODIES if b not in bodies]
    if missing:
        raise ROSConfigError(f"metal_pick_place: arm MJCF {arm_path} lacks bodies {missing}.")

    for mesh in spec.meshes:
        if mesh.name in ("link7", "link8"):
            mesh.scale = list(opts.finger_mesh_scale)
    for act in spec.actuators:
        if act.name == "joint8":
            act.gainprm[0] = opts.finger_kp
            act.biasprm[1] = -opts.finger_kp
            act.biasprm[2] = -opts.finger_kv
    spec.visual.headlight.ambient = list(opts.headlight_ambient)
    spec.visual.headlight.diffuse = list(opts.headlight_diffuse)
    spec.visual.quality.shadowsize = 4096

    arm_mat = spec.add_material()
    arm_mat.name = "anodised_black"
    arm_mat.rgba = list(opts.arm_rgba)
    arm_mat.specular = opts.arm_specular
    arm_mat.shininess = opts.arm_shininess
    meshes: dict[str, str] = {}
    for name in ARM_BODIES:
        if opts.arm_gravcomp:
            bodies[name].gravcomp = 1.0
        for geom in bodies[name].geoms:
            if geom.type == mj.mjtGeom.mjGEOM_MESH:
                geom.name = geom.name or f"{name}_mesh"
                geom.material = "anodised_black"
                geom.rgba = list(opts.arm_rgba)  # URDF rgba would override the material
                meshes[name] = geom.meshname

    _add_desk(spec, opts)
    _add_objects(spec, opts)
    if opts.leader_arm != "hidden":
        _add_leader(spec, opts, meshes)
    _add_cameras(spec, opts, width, height)

    _add_lights(spec, opts)

    try:
        return spec.compile(), arm_path
    except ValueError as exc:
        raise ROSConfigError(f"metal_pick_place: composed model failed to compile: {exc}") from exc


def _add_lights(spec: mujoco.MjSpec, opts: MetalPickPlaceOptions) -> None:
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


def add_light_grid(
    spec: mujoco.MjSpec,
    *,
    pos: _Vec3,
    diffuse: _Vec3,
    grid: int,
    spread_m: float,
    bulb_radius_m: float,
    cutoff_deg: float,
    exponent: float,
    castshadow: bool,
) -> None:
    """Add a ``grid`` x ``grid`` of downward spot lights centred on ``pos``, sharing ``diffuse``.

    Their overlapping hard shadows add up to the soft contact shadow the real overhead
    lighting gives (MuJoCo's own shadows are hard-edged). Shared by the Metal scenes.
    """
    n = grid
    for i in range(n):
        for j in range(n):
            light = spec.worldbody.add_light()
            light.name = f"ceiling_{i}_{j}"
            off = [(k / (n - 1) - 0.5) * spread_m if n > 1 else 0.0 for k in (i, j)]
            light.pos = [pos[0] + off[0], pos[1] + off[1], pos[2]]
            light.dir = [0.0, 0.0, -1.0]
            light.diffuse = [c / (n * n) for c in diffuse]
            light.specular = [0.15 / (n * n)] * 3
            light.castshadow = castshadow
            light.bulbradius = bulb_radius_m
            light.cutoff = cutoff_deg
            light.exponent = exponent


def _static_geom(
    spec: mujoco.MjSpec, name: str, kind: int, pos: _Vec3, size: _Vec3, rgba: _Vec4
) -> mujoco.MjsGeom:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = kind
    geom.pos = list(pos)
    geom.size = list(size)
    geom.rgba = list(rgba)
    return geom


def _add_desk(spec: mujoco.MjSpec, opts: MetalPickPlaceOptions) -> None:
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
        # texuniform: one texture period spans 2 / texrepeat metres (measured), with
        # texel rows along world y and columns along world x.
        mat.texuniform = True
        mat.texrepeat = [2.0 / opts.table_texture_size_m[1], 2.0 / opts.table_texture_size_m[0]]
        mat.specular = 0.1
        mat.shininess = 0.2
        mat.rgba = list(opts.table_rgba)
        mat.emission = opts.table_emission
        table.material = "wood"

    wall = _static_geom(
        spec,
        "wall",
        mj.mjtGeom.mjGEOM_BOX,
        (opts.wall_x + 0.01, 0.3, 0.3),
        (0.01, 1.5, 1.1),
        opts.wall_rgba,
    )
    wall_mat = spec.add_material()
    wall_mat.name = "wall"
    wall_mat.emission = opts.wall_emission  # the overhead lights graze the wall
    wall_mat.rgba = list(opts.wall_rgba)
    wall.material = "wall"
    floor = _static_geom(
        spec,
        "floor",
        mj.mjtGeom.mjGEOM_PLANE,
        (0.0, 0.0, opts.floor_z),
        (0, 0, 0.05),
        opts.floor_rgba,
    )
    r = opts.grommet_diameter_m / 2.0
    gx, gy = opts.grommet_xy
    ring = _static_geom(
        spec,
        "grommet",
        mj.mjtGeom.mjGEOM_CYLINDER,
        (gx, gy, opts.table_top_z + 0.0005),
        (r, 0.0005, 0.0),
        (0.03, 0.03, 0.03, 1.0),
    )
    hole = _static_geom(
        spec,
        "grommet_hole",
        mj.mjtGeom.mjGEOM_CYLINDER,
        (gx, gy, opts.table_top_z + 0.0008),
        (r * 0.78, 0.0005, 0.0),
        (0.12, 0.10, 0.08, 1.0),
    )
    props = [wall, floor, ring, hole]
    for k, item in enumerate(opts.clutter):
        kind = mj.mjtGeom.mjGEOM_BOX if item.kind == "box" else mj.mjtGeom.mjGEOM_CYLINDER
        geom = _static_geom(spec, f"clutter_{k}", kind, item.pos, item.size, item.rgba)
        if item.emission:
            mat = spec.add_material()
            mat.name = f"clutter_{k}"
            mat.rgba = list(item.rgba)
            mat.emission = item.emission
            geom.material = mat.name
        props.append(geom)
    for name, decal in (("table_photo", opts.table_photo), ("cart_photo", opts.cart_photo)):
        if decal is not None:
            (x0, x1), (y0, y1) = decal.x_range, decal.y_range
            props.append(
                _textured_box(
                    spec,
                    name,
                    opts.asset_path(decal.file),
                    ((x0 + x1) / 2, (y0 + y1) / 2, decal.z),
                    ((x1 - x0) / 2, (y1 - y0) / 2),
                    opts.photo_gain,
                    opts.photo_emission,
                )
            )
    for geom in props:  # visual only: the arm never needs to touch these
        geom.contype = 0
        geom.conaffinity = 0


def _textured_box(
    spec: mujoco.MjSpec,
    name: str,
    file: Path,
    pos: _Vec3,
    half_xy: _Vec2,
    gain: float,
    emission: float,
    parent: mujoco.MjsBody | None = None,
    *,
    disc: bool = False,
) -> mujoco.MjsGeom:
    """A 0.2 mm thick box (or disc of radius ``half_xy[0]``) whose top shows ``file`` once.

    Texel row = -y, column = +x; a disc maps the image onto its bounding square.
    """
    import imageio.v3 as iio
    import mujoco as mj
    import numpy as np

    try:  # any imageio format (the big desk photo ships as JPEG); MuJoCo itself reads PNG only
        img = np.ascontiguousarray(np.asarray(iio.imread(file), dtype=np.uint8)[..., :3])
    except OSError as exc:
        raise ROSConfigError(f"metal_pick_place: cannot read texture {file}: {exc}") from exc
    tex = spec.add_texture()
    tex.name = name
    tex.type = mj.mjtTexture.mjTEXTURE_2D
    tex.height, tex.width, tex.nchannel = img.shape
    tex.data = img.tobytes()
    mat = spec.add_material()
    mat.name = name
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = name
    mat.texuniform = False
    mat.texrepeat = [1.0, 1.0]
    mat.specular = 0.0
    mat.shininess = 0.0
    mat.rgba = [gain, gain, gain, 1.0]
    mat.emission = emission
    geom = (parent or spec.worldbody).add_geom()
    geom.name = name
    geom.type = mj.mjtGeom.mjGEOM_CYLINDER if disc else mj.mjtGeom.mjGEOM_BOX
    geom.pos = list(pos)
    geom.size = [half_xy[0], 0.0001, 0.0] if disc else [half_xy[0], half_xy[1], 0.0001]
    geom.material = name
    geom.contype = 0
    geom.conaffinity = 0
    return geom


def _add_objects(spec: mujoco.MjSpec, opts: MetalPickPlaceOptions) -> None:
    import mujoco as mj

    half = opts.cube_edge_m / 2.0
    (cx, cy), (tx, ty) = opts.layouts[opts.layout]
    cube = spec.worldbody.add_body()
    cube.name = "cube"
    cube.pos = [cx, cy, opts.table_top_z + half]
    cube.add_freejoint().name = "cube_joint"
    geom = cube.add_geom()
    geom.name = "cube"
    geom.type = mj.mjtGeom.mjGEOM_BOX
    geom.size = [half, half, half]
    geom.mass = opts.cube_mass_kg
    geom.friction = list(opts.cube_friction)
    geom.condim = 4
    cube_mat = spec.add_material()
    cube_mat.name = "cube"
    cube_mat.rgba = list(opts.cube_rgba)
    cube_mat.emission = opts.cube_emission
    geom.material = "cube"

    marker = spec.worldbody.add_body()
    marker.name = "marker"
    marker.mocap = True
    marker.pos = [tx, ty, opts.table_top_z + 0.0004]
    if opts.marker_texture is not None:
        _textured_box(
            spec,
            "marker",
            opts.asset_path(opts.marker_texture),
            (0.0, 0.0, 0.0),
            (opts.marker_diameter_m / 2.0, opts.marker_diameter_m / 2.0),
            opts.photo_gain,
            opts.photo_emission,
            parent=marker,
            disc=True,
        )
        return
    disc = marker.add_geom()
    disc.name = "marker"
    disc.type = mj.mjtGeom.mjGEOM_CYLINDER
    disc.size = [opts.marker_diameter_m / 2.0, 0.0003, 0.0]
    disc.rgba = list(opts.marker_rgba)
    disc.contype = 0
    disc.conaffinity = 0


def _add_leader(spec: mujoco.MjSpec, opts: MetalPickPlaceOptions, meshes: dict[str, str]) -> None:
    import mujoco as mj

    for name in ARM_BODIES:
        body = spec.worldbody.add_body()
        body.name = f"leader_{name}"
        body.mocap = True
        body.pos = list(opts.leader_base_xyz)
        geom = body.add_geom()
        geom.name = f"leader_{name}"
        geom.type = mj.mjtGeom.mjGEOM_MESH
        geom.meshname = meshes[name]
        geom.material = "anodised_black"
        geom.rgba = list(opts.arm_rgba)
        geom.contype = 0
        geom.conaffinity = 0
        geom.mass = 0.0
        geom.density = 0.0


def _add_cameras(spec: mujoco.MjSpec, opts: MetalPickPlaceOptions, width: int, height: int) -> None:
    top = spec.worldbody.add_camera()
    top.name = "top"
    top.pos = list(opts.top_camera_pos)
    top.quat = list(
        look_at_quat_wxyz(opts.top_camera_pos, opts.top_camera_target, up=opts.top_camera_up)
    )
    res_w, res_h = opts.top_camera_resolution
    sx, sy = width / res_w, height / res_h
    top.resolution = [width, height]
    pixel_m = 3e-6  # nominal pixel pitch; only the pixel ratios matter
    top.sensor_size = [width * pixel_m, height * pixel_m]
    top.focal_pixel = [opts.top_camera_focal_px * sx, opts.top_camera_focal_px * sy]
    top.principal_pixel = list(
        mujoco_principal_pixel(
            opts.top_camera_principal_px[0] * sx,
            opts.top_camera_principal_px[1] * sy,
            width,
            height,
        )
    )

    mount = next(b for b in spec.bodies if b.name == "gripper_base")
    wrist = mount.add_camera()
    wrist.name = "wrist"
    wrist.pos = list(opts.wrist_camera_pos)
    wrist.quat = list(
        look_at_quat_wxyz(opts.wrist_camera_pos, opts.wrist_camera_target, up=opts.wrist_camera_up)
    )
    wrist.fovy = opts.wrist_camera_fovy_deg
    if opts.wrist_camera_focal_px is not None:
        cx, cy = opts.wrist_camera_principal_px or (res_w / 2.0, res_h / 2.0)
        wrist.resolution = [width, height]
        wrist.sensor_size = [width * pixel_m, height * pixel_m]
        wrist.focal_pixel = [opts.wrist_camera_focal_px * sx, opts.wrist_camera_focal_px * sy]
        wrist.principal_pixel = list(mujoco_principal_pixel(cx * sx, cy * sy, width, height))
