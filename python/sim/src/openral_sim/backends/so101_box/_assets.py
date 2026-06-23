"""Compose the ``so101_box`` MJCF.

Everything that controls scene geometry — box dimensions, robot base
pose, OAK-D Pro overhead camera placement, wrist camera placement,
slotted block + tube dimensions — is sourced from a single typed
``BoxSceneOptions`` dataclass.  The dataclass is filled from the
YAML's ``scene.backend_options`` block in :mod:`.env`, so any future
"SO-101 in a box" variant is a pure YAML edit.

The composer reads the upstream
``robot_descriptions:so_arm101_mj_description`` MJCF
(``TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.xml``),
re-anchors its ``<body name="base">`` to the configured robot pose,
splices a wrist camera into the ``<body name="gripper">`` body, and
appends the arena + camera + tube + slotted-block bodies to the
worldbody.  The result is written next to the upstream MJCF so
``meshdir="assets"`` resolves at compile time without copying any
mesh STLs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz

__all__ = [
    "BoxSceneOptions",
    "compose_so101_box_mjcf",
]


# Sentinel for "not provided": lets None mean "use a typed default".
_UNSET: tuple[float, ...] = ()


@dataclass(frozen=True)
class BoxSceneOptions:
    """Every dimension and pose the ``so101_box`` scene exposes.

    Lengths are metres, angles are radians unless suffixed ``_deg``.

    Attributes:
        box_size_xyz: Inside dimensions of the box arena in metres
            ``(X, Y, Z)``. Default ``(1.00, 0.615, 0.75)`` matches the
            user-specified scene (100 cm wide × 61.5 cm deep × 75 cm
            tall).
        wall_thickness: Half-thickness of the floor / wall geoms in
            metres. The walls extend OUTWARD from the inside surface,
            i.e. the inside dimension is ``box_size_xyz``.
        robot_base_xyz: World-frame position of the SO-101 ``base``
            body, in metres. Defaults to ``(0.50, 0.50, 0.0)`` —
            back-centre on the floor, leaving ~11 cm to the back wall.
        robot_base_yaw_deg: Rotation about world +Z applied to the
            SO-101 base body, in degrees. Default ``0.0`` keeps the
            arm pointing toward decreasing Y (i.e. toward the front of
            the box, away from the back wall the base sits against).
        wrist_camera_pos_local: Wrist-camera position in the SO-101
            ``gripper`` body frame, in metres. Default
            ``(0.03, -0.02, -0.06)`` sits the camera just above/behind
            the fingertips.
        wrist_camera_target_local: Wrist-camera look-at target in the
            same local frame. Default looks along the gripper APPROACH
            axis (gripper-local -X = world-down at the home pose, i.e.
            where the fingers point / what gets grasped), so the jaw and
            the object ahead stay in frame at every arm pose.
        wrist_camera_fovy: Vertical field of view of the wrist camera,
            degrees. Default ``75`` gives a wide wrist-cam view.
        oak_top_camera_pos: World-frame position of the overhead
            OAK-D Pro, in metres. Default
            ``(0.50, 0.3075, 0.75)`` — centred on the box, sitting on
            the ceiling.
        oak_top_camera_target: World-frame look-at target. Default
            looks straight down at the floor centre.
        oak_top_camera_fovy: Vertical field of view of the OAK-D Pro,
            degrees. Default ``54`` mirrors the real OAK-D Pro's
            RGB FoV when configured for 4:3 imaging.
        slot_block_size: Slotted block outside dimensions ``(X, Y, Z)``,
            in metres. Default ``(0.0445, 0.0445, 0.020)`` matches the
            sketch (44.5 × 44.5 × 20 mm).
        slot_block_hole_diameter: Diameter of the hole through the
            block, in metres. The hole is modelled as a square of the
            same edge length (so the cylindrical tube clears the
            walls; the corners contribute extra clearance that does
            not affect insertion success). Default ``0.023`` m.
        slot_block_slot_width: Width of the slot connecting the hole
            to one edge of the block, metres. Default ``0.005``.
        slot_block_mass: Mass of the slotted block, kg. Default
            ``0.05``.
        tube_radius: Tube radius in metres. Default ``0.01095`` (Ø 21.9 mm).
        tube_length: Tube length in metres. Default ``0.090`` (90 mm).
        tube_mass: Tube mass in kg. Default ``0.020``.
        block_spawn_xy_range: Block spawn area in world frame, given
            as ``((x_min, x_max), (y_min, y_max))``. Both block and
            tube spawn lying flat on the floor at random (x, y) and a
            random yaw within ``[-pi, pi]``. The block z is fixed at
            ``block_size_z / 2`` (resting on the floor).
        tube_spawn_xy_range: Tube spawn area in world frame; same
            shape as ``block_spawn_xy_range``. Tube spawns lying on
            its side (z = ``tube_radius``).
        spawn_min_separation: Minimum centre-to-centre separation
            between the block and the tube at spawn, metres. The
            spawn sampler retries the tube draw until this is met (up
            to 32 attempts).
    """

    box_size_xyz: tuple[float, float, float] = (1.00, 0.615, 0.75)
    wall_thickness: float = 0.01

    robot_base_xyz: tuple[float, float, float] = (0.50, 0.50, 0.0)
    robot_base_yaw_deg: float = 0.0

    # Wrist camera, parented to the terminal gripper body and tuned to look in
    # the direction orthogonal to the jaw open/close motion while keeping about
    # 20% of the frame on the gripper and most of the rest on the workspace.
    wrist_camera_pos_local: tuple[float, float, float] = (-0.02, 0.02, 0.0)
    wrist_camera_target_local: tuple[float, float, float] = (-0.02, -0.08, 0.0)
    wrist_camera_up_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    wrist_camera_fovy: float = 58.0

    oak_top_camera_pos: tuple[float, float, float] = (0.50, 0.3075, 0.749)
    oak_top_camera_target: tuple[float, float, float] = (0.50, 0.3075, 0.0)
    oak_top_camera_fovy: float = 54.0

    slot_block_size: tuple[float, float, float] = (0.0445, 0.0445, 0.020)
    slot_block_hole_diameter: float = 0.023
    slot_block_slot_width: float = 0.005
    slot_block_mass: float = 0.05

    tube_radius: float = 0.01095
    tube_length: float = 0.090
    tube_mass: float = 0.020

    block_spawn_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (0.10, 0.40),
    )
    tube_spawn_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (0.10, 0.40),
    )
    spawn_min_separation: float = 0.10

    # Insertion-success thresholds — read by :func:`_check_insertion`.
    # ``insertion_xy_tol_m`` is the lateral slack on the tube tip's XY
    # position relative to the hole centre at success time. The physical
    # fit constraint (Ø 21.9 mm tube into a Ø 23 mm hole) is ±0.55 mm at
    # the inscribed square's midpoints — but the depth check is what
    # actually enforces "in the hole" (the tube cannot descend
    # ``insertion_depth_m`` past the block top unless it geometrically
    # fits). The XY tolerance therefore guards against false positives
    # where the tube happens to be at the right height beside the block;
    # 3 mm is well-aligned with "above the hole" intuition while
    # absorbing the discrete-physics-step transient noise that a 0.55 mm
    # threshold would clip out.
    insertion_depth_m: float = 0.010
    insertion_axis_tol_deg: float = 10.0
    insertion_xy_tol_m: float = 0.003

    # Joint-units convention for the scene's proprio state + action contract.
    # ``"radians"`` (default) keeps MuJoCo-native units; ``"degrees"`` makes
    # the env emit state and accept actions in degrees — the convention
    # LeRobot-trained SO-100/101 checkpoints (for example MolmoAct2-SO100_101)
    # were recorded in. Consumed by the env factory, not the MJCF composer.
    joint_units: str = "radians"

    # Per-joint calibration affine bridging the MuJoCo URDF joint convention to
    # the checkpoint's LeRobot servo-degree convention (only applied when
    # ``joint_units == "degrees"``):
    #     lerobot_deg = joint_signs * mujoco_deg + joint_offsets_deg
    # The MuJoCo ``so101_new_calib`` zero does not share a per-joint zero with
    # the LeRobot calibration a checkpoint was recorded in (most visibly
    # shoulder_lift / elbow_flex sit ~120° from URDF zero in LeRobot data), and
    # different checkpoints used different calibrations — so this is left
    # IDENTITY by default and set per-checkpoint in the run config. Each is a
    # 6-vector in the robot's joint order (shoulder_pan, shoulder_lift,
    # elbow_flex, wrist_flex, wrist_roll, gripper); ``joint_signs`` entries must
    # be +1 or -1.
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    extra_metadata: dict[str, str] = field(default_factory=dict)


def _resolve_so101_mjcf() -> Path:
    """Return the path to the upstream SO-101 MJCF.

    Imports ``robot_descriptions`` lazily so loading this module does
    not trigger the ~300 MB upstream fetch on light CLI paths
    (``openral sim list``).
    """
    from robot_descriptions import so_arm101_mj_description

    return Path(so_arm101_mj_description.MJCF_PATH)


def _resolve_robot_mjcf(description: RobotDescription) -> Path:
    """Resolve a robot's base MJCF from its manifest ``assets.mjcf`` (ADR-0033/0057).

    The same ``assets.mjcf`` source ``build_hal(mode="sim")`` /
    ``MujocoArmHAL.from_description`` consume — so the manifest is the single
    robot-MJCF source across sim run, deploy sim, and deploy run. Resolved by the
    one :func:`openral_core.assets.resolve_asset` grammar (``rd:`` for the SO-ARM
    family, ``file:`` / ``gym_aloha:`` / ``openarm:`` / ``menagerie:``).

    The composed scene splices its task world onto this robot MJCF via the
    ``<body name="base">`` + ``<body name="gripper">`` anchors, so the robot
    must share the SO-ARM body naming (so100 / so101). A robot with different
    base/end-effector body names needs those anchors parameterised — a
    follow-up beyond the so101_box PoC.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if not description.assets.mjcf:
        raise ROSConfigError(
            f"so101_box: robot {description.name!r} has no `assets.mjcf` in its manifest, "
            "so there is no base MJCF to compose the box scene around.",
        )
    try:
        path = resolve_asset(description.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(f"so101_box: {exc}") from exc
    if path is None:
        raise ROSConfigError(
            f"so101_box: assets.mjcf={description.assets.mjcf!r} did not resolve to a file.",
        )
    return path


def _yaw_quat_z(yaw_deg: float) -> tuple[float, float, float, float]:
    """Return the (w, x, y, z) MuJoCo quaternion for a rotation about world +Z."""
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# MuJoCo (w, x, y, z) look-at quaternion — promoted to the shared gaze-geometry
# helper in ADR-0044 Phase 1; the "-z" default is the MuJoCo camera convention.
_look_at_quat = look_at_quat_wxyz


def _reanchor_robot_base(xml: str, pos: tuple[float, float, float], yaw_deg: float) -> str:
    """Rewrite the ``<body name="base" pos=... quat=...>`` line.

    The upstream SO-101 MJCF declares its base at ``pos="0 0 0"
    quat="1 0 0 0"``. We rewrite both attributes in one regex pass so
    the entire kinematic chain rigidly translates + yaws to the
    configured pose.
    """
    quat = _yaw_quat_z(yaw_deg)
    pattern = re.compile(
        r'(<body[^>]*\bname="base"[^>]*\bpos=")([^"]+)("[^>]*\bquat=")([^"]+)("[^>]*>)',
    )
    new_pos = f"{pos[0]} {pos[1]} {pos[2]}"
    new_quat = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
    xml, n = pattern.subn(rf"\g<1>{new_pos}\g<3>{new_quat}\g<5>", xml, count=1)
    if n != 1:
        raise ROSConfigError(
            'so101_box: cannot find <body name="base" pos=... quat=...> in the robot '
            "MJCF (so_arm101 schema). The base body must declare pos= + quat=; a robot "
            "with a different base-body schema needs the splice anchors parameterised "
            "(ADR-0033 follow-up).",
        )
    return xml


def _splice_wrist_camera(
    xml: str,
    pos_local: tuple[float, float, float],
    target_local: tuple[float, float, float],
    up_local: tuple[float, float, float],
    fovy: float,
) -> str:
    """Insert a ``<camera name="wrist">`` element inside the terminal gripper body.

    The camera is parented to the SO-101's ``gripper`` body, matching the
    direct MJCF snippet used to tune the pose. ``pos_local`` / ``target_local``
    are expressed in that body's local frame.
    """
    quat = _look_at_quat(pos_local, target_local, up=up_local)
    cam = (
        f'<camera name="wrist" pos="{pos_local[0]} {pos_local[1]} {pos_local[2]}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" fovy="{fovy}" '
        f'mode="fixed"/>'
    )
    pattern = re.compile(r'(<body[^>]*\bname="gripper"[^>]*>)')
    xml, n = pattern.subn(rf"\g<1>\n        {cam}", xml, count=1)
    if n != 1:
        raise ROSConfigError(
            'so101_box: cannot find <body name="gripper"> in the '
            "upstream SO-101 MJCF — wrist camera cannot be parented.",
        )
    return xml


def _inject_fill_light(xml: str) -> str:
    """Add a moderate ambient ``<visual><headlight>`` to the scene.

    The arena has a single directional ceiling light, so surfaces that face
    away from it — everything the gripper-mounted wrist camera sees when it
    looks down into the box — render nearly black. A moderate ambient term
    lifts those shadows without washing out the materials (the policy still
    sees realistic, coloured frames). No-op if the upstream MJCF already
    declares a ``<visual>`` block (don't clobber an explicit choice).
    """
    if "<visual" in xml:
        return xml
    headlight = (
        "\n  <visual>\n"
        '    <headlight ambient="0.4 0.4 0.4" diffuse="0.4 0.4 0.4" '
        'specular="0.1 0.1 0.1"/>\n'
        "  </visual>"
    )
    body, n = re.subn(r"(<mujoco\b[^>]*>)", r"\1" + headlight, xml, count=1)
    if n != 1:
        raise ROSConfigError(
            "so101_box: cannot find the opening <mujoco> tag to inject scene "
            "lighting — upstream MJCF structure changed.",
        )
    return body


def _render_arena_geoms(box: tuple[float, float, float], wall_t: float) -> str:
    """Return the floor + 4 walls + ceiling sites as a worldbody XML snippet.

    Each wall is a thin box geom whose inside face flushes with the
    box interior. The floor is opaque white; walls are translucent
    light grey so the overhead camera framing is clean and the
    interactive viewer can still see into the box.
    """
    bx, by, bz = box
    t = wall_t
    floor = (
        f'<geom name="floor" type="box" pos="{bx / 2.0} {by / 2.0} {-t}" '
        f'size="{bx / 2.0 + t} {by / 2.0 + t} {t}" '
        f'rgba="0.92 0.92 0.92 1" friction="1 0.01 0.001"/>'
    )
    wall_xn = (
        f'<geom name="wall_xn" type="box" pos="{-t} {by / 2.0} {bz / 2.0}" '
        f'size="{t} {by / 2.0 + t} {bz / 2.0}" '
        f'rgba="0.85 0.85 0.85 0.3" contype="0" conaffinity="0"/>'
    )
    wall_xp = (
        f'<geom name="wall_xp" type="box" pos="{bx + t} {by / 2.0} {bz / 2.0}" '
        f'size="{t} {by / 2.0 + t} {bz / 2.0}" '
        f'rgba="0.85 0.85 0.85 0.3" contype="0" conaffinity="0"/>'
    )
    wall_yn = (
        f'<geom name="wall_yn" type="box" pos="{bx / 2.0} {-t} {bz / 2.0}" '
        f'size="{bx / 2.0} {t} {bz / 2.0}" '
        f'rgba="0.85 0.85 0.85 0.3" contype="0" conaffinity="0"/>'
    )
    wall_yp = (
        f'<geom name="wall_yp" type="box" pos="{bx / 2.0} {by + t} {bz / 2.0}" '
        f'size="{bx / 2.0} {t} {bz / 2.0}" '
        f'rgba="0.85 0.85 0.85 0.3" contype="0" conaffinity="0"/>'
    )
    light = (
        '<light name="ceiling" mode="targetbody" target="slot_block" '
        f'pos="{bx / 2.0} {by / 2.0} {bz}" dir="0 0 -1" diffuse="0.8 0.8 0.8" '
        'specular="0.1 0.1 0.1" castshadow="false"/>'
    )
    return "\n        ".join([floor, wall_xn, wall_xp, wall_yn, wall_yp, light])


def _render_overhead_camera(opts: BoxSceneOptions) -> str:
    """Build the ``<camera name="oak_top">`` element."""
    pos = opts.oak_top_camera_pos
    quat = _look_at_quat(pos, opts.oak_top_camera_target)
    return (
        f'<camera name="oak_top" pos="{pos[0]} {pos[1]} {pos[2]}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
        f'fovy="{opts.oak_top_camera_fovy}" mode="fixed"/>'
    )


def _render_slot_block(opts: BoxSceneOptions) -> str:
    """Build the slotted block as 5 box geoms in a single body.

    Layout (block local frame, X-right, Y-front, Z-up; slot opens in
    +Y direction):

        +-------+---+-------+
        | LL    |   |    LR |   ← top strip split by the slot
        +-------+   +-------+
        |       |   |       |
        |  L    |   |   R   |   ← side strips around the square hole
        |       |   |       |
        +-------+---+-------+
        |        B          |   ← bottom strip (opposite the slot)
        +-------------------+

    The square hole's edge length equals
    ``slot_block_hole_diameter`` so a Ø21.9 mm cylindrical tube clears
    the inscribed square (0.55 mm radial clearance at the midpoints,
    more at the corners). The slot
    is ``slot_block_slot_width`` wide and runs from the hole to the
    +Y edge.

    All geoms share the parent ``slot_block`` body so a single freejoint
    pose translates the entire shape.
    """
    bx, by, bz = opts.slot_block_size
    hd = opts.slot_block_hole_diameter
    sw = opts.slot_block_slot_width
    if hd >= bx or hd >= by:
        raise ROSConfigError(
            f"so101_box: slot_block_hole_diameter ({hd}) must be < both XY "
            f"edges of slot_block_size ({bx}, {by}).",
        )
    if sw >= hd:
        raise ROSConfigError(
            f"so101_box: slot_block_slot_width ({sw}) must be < slot_block_hole_diameter ({hd}).",
        )

    # Block extents (centred at body origin):
    #   X ∈ [-bx/2, +bx/2], Y ∈ [-by/2, +by/2], Z ∈ [-bz/2, +bz/2]
    # Square hole: X ∈ [-hd/2, +hd/2], Y ∈ [-hd/2, +hd/2].
    # Slot opens toward +Y: X ∈ [-sw/2, +sw/2], Y ∈ [+hd/2, +by/2].

    half_x = bx / 2.0
    half_y = by / 2.0
    half_z = bz / 2.0
    hr = hd / 2.0  # hole half-edge
    sr = sw / 2.0  # slot half-width

    # Block-local frame: centred at origin. Each strip is a box geom
    # whose ``pos`` is the strip's centre and ``size`` is the half-extent.
    #
    # Left wall  — X ∈ [-half_x, -hr], Y ∈ [-half_y, +half_y]
    left = {
        "pos": (-(half_x + hr) / 2.0, 0.0, 0.0),
        "size": ((half_x - hr) / 2.0, half_y, half_z),
        "name": "slot_block_left",
    }
    # Right wall — X ∈ [+hr, +half_x], Y ∈ [-half_y, +half_y]
    right = {
        "pos": ((half_x + hr) / 2.0, 0.0, 0.0),
        "size": ((half_x - hr) / 2.0, half_y, half_z),
        "name": "slot_block_right",
    }
    # Bottom strip (opposite the slot — closed side) — X ∈ [-hr, +hr], Y ∈ [-half_y, -hr]
    bottom = {
        "pos": (0.0, -(half_y + hr) / 2.0, 0.0),
        "size": (hr, (half_y - hr) / 2.0, half_z),
        "name": "slot_block_bottom",
    }
    # Top-left strip — X ∈ [-hr, -sr], Y ∈ [+hr, +half_y]
    top_left = {
        "pos": (-(hr + sr) / 2.0, (half_y + hr) / 2.0, 0.0),
        "size": ((hr - sr) / 2.0, (half_y - hr) / 2.0, half_z),
        "name": "slot_block_top_left",
    }
    # Top-right strip — X ∈ [+sr, +hr], Y ∈ [+hr, +half_y]
    top_right = {
        "pos": ((sr + hr) / 2.0, (half_y + hr) / 2.0, 0.0),
        "size": ((hr - sr) / 2.0, (half_y - hr) / 2.0, half_z),
        "name": "slot_block_top_right",
    }

    geoms = [left, right, bottom, top_left, top_right]
    geom_xml = "\n          ".join(
        '<geom name="{name}" type="box" pos="{px} {py} {pz}" size="{sx} {sy} {sz}" '
        'rgba="0.20 0.55 0.85 1" friction="1 0.01 0.001"/>'.format(
            name=g["name"],
            px=g["pos"][0],
            py=g["pos"][1],
            pz=g["pos"][2],
            sx=g["size"][0],
            sy=g["size"][1],
            sz=g["size"][2],
        )
        for g in geoms
    )

    # Insertion target site at the centre of the hole, slightly below the top
    # surface — the success check reads this site to know where the hole is in
    # world frame after the block has been pushed around.
    hole_site = (
        '<site name="slot_block_hole" type="cylinder" pos="0 0 0" '
        f'size="{hr * 0.95} {half_z}" rgba="1 0 0 0.05" group="3"/>'
    )

    return (
        f'<body name="slot_block" pos="0 0 {half_z}">\n'
        '          <freejoint name="slot_block_joint"/>\n'
        f'          <inertial pos="0 0 0" mass="{opts.slot_block_mass}" '
        f'diaginertia="{opts.slot_block_mass * (by**2 + bz**2) / 12.0} '
        f"{opts.slot_block_mass * (bx**2 + bz**2) / 12.0} "
        f'{opts.slot_block_mass * (bx**2 + by**2) / 12.0}"/>\n'
        f"          {geom_xml}\n"
        f"          {hole_site}\n"
        f"        </body>"
    )


def _render_tube(opts: BoxSceneOptions) -> str:
    """Build the cylindrical tube body."""
    r = opts.tube_radius
    half_l = opts.tube_length / 2.0
    m = opts.tube_mass
    # Tube spawns lying on its side: its long axis is body-local +X.
    # MuJoCo's cylinder primitive is axis-aligned along the body's local
    # +Z by default; we rotate the tube 90° about world +Y so its long
    # axis points along body +X, then spawn with the freejoint placing
    # it on the floor (z = tube_radius).
    return (
        '<body name="tube" pos="0 0 0">\n'
        '          <freejoint name="tube_joint"/>\n'
        f'          <inertial pos="0 0 0" mass="{m}" '
        f'diaginertia="{m * (3 * r * r + opts.tube_length**2) / 12.0} '
        f"{m * (3 * r * r + opts.tube_length**2) / 12.0} "
        f'{m * r * r / 2.0}"/>\n'
        f'          <geom name="tube_geom" type="cylinder" '
        f'size="{r} {half_l}" rgba="0.95 0.65 0.10 1" friction="1 0.01 0.001"/>\n'
        '          <site name="tube_tip_lo" pos="0 0 ' + str(-half_l) + '" '
        'type="sphere" size="0.003" rgba="0 1 0 0.5" group="3"/>\n'
        '          <site name="tube_tip_hi" pos="0 0 ' + str(half_l) + '" '
        'type="sphere" size="0.003" rgba="0 1 0 0.5" group="3"/>\n'
        "        </body>"
    )


def compose_so101_box_mjcf(
    options: BoxSceneOptions | None = None,
    robot_description: RobotDescription | None = None,
) -> tuple[str, Path]:
    """Return ``(composed_xml, output_mjcf_path)`` for the so101_box scene.

    The output path is a sibling of the upstream SO-101 MJCF so its
    ``meshdir="assets"`` resolves at compile time without copying
    STL meshes.  The XML string is also returned for inspection /
    unit-tests that don't want to touch the filesystem.

    Args:
        options: Scene options. ``None`` falls back to all defaults
            (matches :class:`BoxSceneOptions` field defaults).
        robot_description: Robot whose ``assets.mjcf`` provides the base arm
            MJCF (ADR-0033). ``None`` falls back to the SO-101 MJCF, keeping the
            legacy call path byte-for-byte unchanged. The robot must share the
            so_arm101 splice anchors (``<body name="base">`` + ``gripper``).

    Returns:
        ``(xml, output_path)`` — ``xml`` is the composed MJCF; the
        same string is written to ``output_path`` next to the
        upstream MJCF.

    Raises:
        ROSConfigError: If any required upstream MJCF landmark
            (the ``<body name="base">`` declaration, the
            ``<body name="gripper">`` body, the closing
            ``</worldbody>``) is missing — those are the splice
            anchors and a future upstream rename would surface here
            loudly.
    """
    opts = options if options is not None else BoxSceneOptions()

    # ADR-0033 — the robot is a flag: resolve its base MJCF from the manifest
    # (`assets.mjcf`) when a description is given; default to SO-101 so the
    # legacy call path is byte-for-byte unchanged.
    upstream_path = (
        _resolve_robot_mjcf(robot_description)
        if robot_description is not None
        else _resolve_so101_mjcf()
    )
    raw = upstream_path.read_text()
    meshdir = upstream_path.parent / "assets"
    if not meshdir.is_dir():
        raise ROSConfigError(
            f"SO-101 mesh dir not found at {meshdir}; upstream MJCF assumes "
            "meshdir='assets' next to the XML.",
        )

    body = _inject_fill_light(raw)
    body = _reanchor_robot_base(body, opts.robot_base_xyz, opts.robot_base_yaw_deg)
    body = _splice_wrist_camera(
        body,
        opts.wrist_camera_pos_local,
        opts.wrist_camera_target_local,
        opts.wrist_camera_up_local,
        opts.wrist_camera_fovy,
    )

    arena_geoms = _render_arena_geoms(opts.box_size_xyz, opts.wall_thickness)
    top_camera = _render_overhead_camera(opts)
    block_xml = _render_slot_block(opts)
    tube_xml = _render_tube(opts)

    scene_snippet = (
        f"        {arena_geoms}\n        {top_camera}\n        {block_xml}\n        {tube_xml}\n"
    )

    body, n = re.subn(r"(</worldbody>)", f"{scene_snippet}      \\1", body, count=1)
    if n != 1:
        raise ROSConfigError(
            "so101_box: cannot find </worldbody> in the upstream SO-101 MJCF — "
            "composer cannot splice the arena into the scene.",
        )

    output_path = upstream_path.parent / "so101_box_generated.xml"
    output_path.write_text(body)
    return body, output_path
