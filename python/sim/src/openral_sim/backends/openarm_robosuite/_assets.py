"""Compose the OpenArm v2 tabletop pick-and-place MJCF.

The upstream ``enactic/openarm_mujoco`` v2 bimanual MJCF ships with
``<position>`` actuators (internal PD, gains tuned per joint class).
That works for behaviour-cloning rollouts where the policy commands
joint angles directly — but a robosuite OSC controller produces
**torques**, so the scene we hand it must expose direct ``<motor>``
actuators on every joint.

This module produces a single composite XML string that:

1. Wraps the upstream MJCF as a ``<worldbody>`` snippet by reading it
   and splicing out (a) its outer ``<mujoco>`` tag, (b) the
   ``<actuator>`` block, and (c) the trailing ``</worldbody>``;
2. Re-anchors the OpenArm base in a known frame above a robosuite
   ``TableArena``-style table top;
3. Adds the manipulation scene: three coloured cubes
   (red / green / blue), a passive parallel-jaw drawer (prismatic
   joint), and three RGB cameras (``top``, ``wrist_left``,
   ``wrist_right``) at the resolutions the pi05 LoRA was trained on;
4. Emits ``<motor>`` actuators for all 16 joints (7 arm + 1 finger per
   side), inheriting the upstream ``<equality>`` constraint that
   yokes the second finger to the first;
5. Preserves the upstream ``meshdir="assets"`` so the cached MJCF's
   sibling ``assets/`` directory still resolves at compile time.

The resulting string is fed to :func:`mujoco.MjModel.from_xml_string`
with an ``asset_root`` callback so the OpenArm meshes load from the
upstream cache without copying them.

The actuator inventory + per-joint ctrlrange / forcerange numbers are
**derived from a loaded** :class:`openral_core.RobotDescription` (see
:func:`actuator_specs_from_description`) — there is no module-level
copy of the openarm joint table anymore. Same for the "top" overview
camera placement: the default ``pos`` / ``target`` / ``fovy`` come from
:attr:`RobotDescription.scene_defaults.top_camera` on the supplied
description, with the YAML's ``scene.backend_options.top_camera_*``
keys still overriding when set.

Honest scope note
-----------------
Only the **structural** composition is exercised today (smoke-tested
by ``tests/sim/test_openarm_scene_pnp.py``). The motor-actuator
control-range numbers (``effort_limit`` in
``robots/openarm/robot.yaml``) are passed through as ``forcerange`` /
``ctrlrange``; whether they need per-joint tuning to keep robosuite
OSC stable on this rig will surface the first time we close the loop
with a real VLA chunk.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz

if TYPE_CHECKING:
    from openral_core import RobotDescription

# `openral_hal._openarm_v2_assets` is imported lazily inside
# `compose_openarm_tabletop_mjcf` — the parent `openral_hal` package
# eagerly imports every HAL adapter, which transitively pulls torch /
# mujoco / lerobot. Keeping the import deferred lets
# `import openral_cli.main` stay light (gated by
# `tests/unit/test_cli_sim_run.py::test_bh_cli_import_is_light`).

__all__ = [
    "ActuatorSpec",
    "actuator_specs_from_description",
    "compose_openarm_tabletop_mjcf",
    "load_openarm_description",
    "motor_actuator_names_from_description",
]


# (actuator_name, mjcf_joint_name, ctrl_lo, ctrl_hi, effort)
ActuatorSpec = tuple[str, str, float, float, float]


def load_openarm_description() -> RobotDescription:
    """Return the OpenArm v2 :class:`RobotDescription` HAL constant.

    The HAL constant ``openral_hal.openarm.OPENARM_DESCRIPTION`` is the
    in-code source of truth for the OpenArm v2 manifest; the YAML at
    ``robots/openarm/robot.yaml`` mirrors it (drift guarded by
    ``tests/unit/test_robot_manifests_match_hal_constants.py``).

    The import is deferred (CLAUDE.md §1.5 fast-CLI rule) — pulling
    ``openral_hal`` at module-import time would drag torch + lerobot +
    mujoco into ``openral_cli.main``'s import path.
    """
    from openral_hal.openarm import OPENARM_DESCRIPTION

    return OPENARM_DESCRIPTION


def _mjcf_joint_name(joint_name: str) -> str:
    """Map a :class:`JointSpec` name onto the upstream MJCF joint name.

    The OpenArm v2 manifest uses logical joint names like
    ``left_joint1`` / ``left_gripper`` (no ``openarm_`` prefix); the
    upstream MJCF qualifies them with ``openarm_`` and the gripper
    expands to the driven finger joint.
    """
    if joint_name.endswith("_gripper"):
        side = joint_name[: -len("_gripper")]
        return f"openarm_{side}_finger_joint1"
    return f"openarm_{joint_name}"


def _actuator_name(joint_name: str) -> str:
    """Map a :class:`JointSpec` name onto the upstream MJCF actuator name."""
    if joint_name.endswith("_gripper"):
        side = joint_name[: -len("_gripper")]
        return f"{side}_finger1_ctrl"
    return f"{joint_name}_ctrl"


def actuator_specs_from_description(desc: RobotDescription) -> list[ActuatorSpec]:
    """Derive the OpenArm v2 motor-actuator table from a loaded manifest.

    Replaces the previous module-level ``_JOINT_SPECS`` constant whose
    own docstring conceded it "Mirrors robots/openarm/robot.yaml" —
    that was two sources of truth. Now the table is computed at use
    time from :attr:`RobotDescription.joints`, so the only place a
    limit can drift is the manifest itself.

    Args:
        desc: A loaded :class:`RobotDescription`. ``desc.joints`` must
            be the OpenArm v2 16-joint inventory (7 revolute arm + 1
            revolute gripper per side); each joint must carry
            ``position_limits`` and ``effort_limit``.

    Returns:
        A list of ``(actuator_name, mjcf_joint_name, ctrl_lo, ctrl_hi,
        effort)`` tuples in the manifest's joint order.

    Raises:
        ROSConfigError: If any required joint field is missing.
    """
    specs: list[ActuatorSpec] = []
    for joint in desc.joints:
        if joint.position_limits is None:
            raise ROSConfigError(
                f"openarm_robosuite: joint {joint.name!r} is missing position_limits; "
                "cannot derive ctrlrange for the composed MJCF actuator.",
            )
        if joint.effort_limit is None:
            raise ROSConfigError(
                f"openarm_robosuite: joint {joint.name!r} is missing effort_limit; "
                "cannot derive forcerange for the composed MJCF actuator.",
            )
        lo, hi = joint.position_limits
        specs.append(
            (
                _actuator_name(joint.name),
                _mjcf_joint_name(joint.name),
                float(lo),
                float(hi),
                float(joint.effort_limit),
            ),
        )
    return specs


def motor_actuator_names_from_description(desc: RobotDescription) -> list[str]:
    """Return the actuator names from the manifest, in joint order."""
    return [s[0] for s in actuator_specs_from_description(desc)]


_SCENE_BODIES = dedent(
    """\
    <!-- Table — flat tabletop centred 60 cm in front of the OpenArm base.
         Matches a typical bimanual workspace; the OpenArm's ~70 cm reach
         envelope reaches the back row of cubes. -->
    <body name="table" pos="0.55 0 0.0">
      <geom name="table_top" type="box" size="0.45 0.6 0.02" rgba="0.7 0.5 0.3 1" friction="1 0.005 0.0001"/>
      <geom name="table_legFL" type="box" pos="0.40 0.55 -0.40" size="0.02 0.02 0.40" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_legFR" type="box" pos="0.40 -0.55 -0.40" size="0.02 0.02 0.40" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_legBL" type="box" pos="-0.40 0.55 -0.40" size="0.02 0.02 0.40" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_legBR" type="box" pos="-0.40 -0.55 -0.40" size="0.02 0.02 0.40" rgba="0.4 0.3 0.2 1"/>
    </body>

    <!-- Three small cubes — pick targets. Sized for OpenArm gripper jaw. -->
    <body name="cube_red" pos="0.45 -0.20 0.05">
      <freejoint/>
      <geom name="cube_red_geom" type="box" size="0.022 0.022 0.022" rgba="0.85 0.10 0.10 1" mass="0.05"/>
    </body>
    <body name="cube_green" pos="0.55 0.00 0.05">
      <freejoint/>
      <geom name="cube_green_geom" type="box" size="0.022 0.022 0.022" rgba="0.10 0.75 0.20 1" mass="0.05"/>
    </body>
    <body name="cube_blue" pos="0.45 0.20 0.05">
      <freejoint/>
      <geom name="cube_blue_geom" type="box" size="0.022 0.022 0.022" rgba="0.10 0.25 0.85 1" mass="0.05"/>
    </body>

    <!-- Drawer — a simple sliding tray with one prismatic joint along +x.
         No friction tuning yet; just a kinematic placeholder so the
         policy can put a cube inside and we can read drawer_pos for
         a success signal. -->
    <body name="drawer_frame" pos="0.85 0 0.04" euler="0 0 0">
      <geom name="drawer_frame_top" type="box" pos="0 0 0.06" size="0.08 0.12 0.005" rgba="0.5 0.4 0.3 1"/>
      <geom name="drawer_frame_floor" type="box" pos="0 0 -0.005" size="0.08 0.12 0.005" rgba="0.5 0.4 0.3 1"/>
      <geom name="drawer_frame_left"  type="box" pos="0 0.12 0.03" size="0.08 0.005 0.04" rgba="0.5 0.4 0.3 1"/>
      <geom name="drawer_frame_right" type="box" pos="0 -0.12 0.03" size="0.08 0.005 0.04" rgba="0.5 0.4 0.3 1"/>
      <body name="drawer_tray" pos="0 0 0.01">
        <joint name="drawer_slide" type="slide" axis="-1 0 0" range="-0.16 0.0" damping="2.0"/>
        <geom name="drawer_tray_floor" type="box" size="0.07 0.105 0.003" rgba="0.6 0.5 0.4 1"/>
        <geom name="drawer_tray_front" type="box" pos="-0.07 0 0.02" size="0.005 0.105 0.02" rgba="0.6 0.5 0.4 1"/>
        <geom name="drawer_tray_back"  type="box" pos="0.07 0 0.02" size="0.005 0.105 0.02" rgba="0.6 0.5 0.4 1"/>
      </body>
    </body>

    <!-- ``top`` (aka ``base`` in the pi05 manifest's alias map) —
         scene-overview camera. The placement is NOT baked into this
         block; ``compose_openarm_tabletop_mjcf`` injects a
         ``<camera name="top" pos=... quat=... fovy=.../>`` line just
         before ``</worldbody>`` from per-call parameters. Default
         placement matches the mddoai/openarm_2026-05-14_clean POV
         (camera between the two bases, looking forward + down at the
         tabletop). Per-checkpoint overrides flow through the YAML's
         ``scene.backend_options.top_camera_*`` fields. The two upstream
         wrist camera tags are renamed here so the HAL + pi05 manifest find
         the dataset-facing names; the runtime renderer re-aims those named
         cameras at the tabletop because the physical hand-mounted pose is
         occluded by the OpenArm gripper shell in this layout. -->
    """
)


# The default "top" camera placement is sourced from
# :attr:`RobotDescription.scene_defaults.top_camera` on the robot
# manifest (see ``robots/openarm/robot.yaml`` and
# ``OPENARM_DESCRIPTION`` in ``openral_hal.openarm``). The previous
# module-level ``_DEFAULT_TOP_CAMERA_*`` constants — baked to the
# ``mddoai/openarm_2026-05-14_clean`` dataset POV — moved onto the
# manifest via the ``SceneDefaults`` / ``TopCameraDefaults`` schemas.
# Per-rollout overrides still flow through
# ``compose_openarm_tabletop_mjcf`` kwargs.

# Hard fallback when a description has no ``scene_defaults.top_camera``
# at all — kept identical to the historical openarm POV so any caller
# that constructs a bare ``RobotDescription`` (e.g. an early test
# fixture without the new submodel) still gets a working camera.
_FALLBACK_TOP_CAMERA_POS: tuple[float, float, float] = (0.20, 0.0, 0.95)
_FALLBACK_TOP_CAMERA_TARGET: tuple[float, float, float] = (0.65, 0.0, 0.05)
_FALLBACK_TOP_CAMERA_FOVY: float = 65.0

# Fallback for the operator-facing ``front`` camera — mirrors the upstream
# enactic/openarm_mujoco ``front_camera`` (pos=[0, 2, 0.5], look=-Y) at a
# more practical 1.5 m distance aimed at the tabletop workspace centre.
_FALLBACK_FRONT_CAMERA_POS: tuple[float, float, float] = (0.45, 1.5, 0.5)
_FALLBACK_FRONT_CAMERA_TARGET: tuple[float, float, float] = (0.45, 0.0, 0.4)
_FALLBACK_FRONT_CAMERA_FOVY: float = 65.0


# MuJoCo (w, x, y, z) look-at quaternion — promoted to the shared gaze-geometry
# helper in ADR-0044 Phase 1; the "-z" default is the MuJoCo camera convention
# (and gains the zero-norm / parallel-up guards this copy lacked).
_look_at_quat = look_at_quat_wxyz


def _render_actuator_block(specs: list[ActuatorSpec]) -> str:
    """Render the ``<actuator>`` block for the OSC composer path.

    Each entry becomes a torque-driven ``<motor>`` actuator clipped by
    the manifest's ``position_limits`` (ctrlrange) and
    ``effort_limit`` (symmetric forcerange).
    """
    lines = [
        f'    <motor name="{name}" joint="{joint}" gear="1" '
        f'ctrllimited="true" ctrlrange="{lo} {hi}" '
        f'forcelimited="true" forcerange="{-effort} {effort}"/>'
        for name, joint, lo, hi, effort in specs
    ]
    return "<actuator>\n" + "\n".join(lines) + "\n  </actuator>"


_BASE_CENTER_SITES = (
    '<site name="openarm_left_base_link_center" pos="0 0 0" '
    'group="3" rgba="0 0 0 0" type="sphere" size="0.001"/>',
    '<site name="openarm_right_base_link_center" pos="0 0 0" '
    'group="3" rgba="0 0 0 0" type="sphere" size="0.001"/>',
)
"""Two zero-size invisible sites at each arm's base body.

robosuite's OSC controller reads ``{naming_prefix}{part_name}_center``
to compute the base's linear / angular velocity (so it can de-bias the
EE jacobian when the base moves). Our OpenArm bases are fixed, so the
sites carry no kinematic content — but the controller still queries
the names, so they must exist.
"""


def _inject_base_center_sites(xml: str) -> str:
    """Attach the per-arm ``_center`` sites inside their base bodies.

    Adds the site as the first child of each ``<body name="openarm_{side}_base_link">``
    declaration so it inherits that body's frame.
    """
    for side, site in zip(("left", "right"), _BASE_CENTER_SITES, strict=True):
        body_pattern = rf'(<body[^>]*\bname="openarm_{side}_base_link"[^>]*>)'
        new, count = re.subn(
            body_pattern,
            r"\1\n      " + site,
            xml,
            count=1,
        )
        if count != 1:
            raise ROSConfigError(
                f'Cannot find <body name="openarm_{side}_base_link"> in the upstream '
                "OpenArm v2 MJCF — composer must be updated."
            )
        xml = new
    return xml


def _lift_robot_bases(xml: str, z_offset: float, x_offset: float = 0.0) -> str:
    """Shift both OpenArm base bodies' ``pos`` attributes.

    The upstream MJCF mounts the bases at ``pos="0 ±0.031 0"`` (floor
    level, x=0). With the table also at z=0 in the scene splice, the
    arms naturally hang straight down below the table top and never
    reach the objects sitting on it. Lifting the bases by ~0.5 m parks
    the robot above the table edge so the arms can sweep down onto the
    cubes / drawer.

    The ``x_offset`` knob is the horizontal equivalent: the cubes sit at
    x≈0.45–0.55 m, which is at the limit of the 0.6 m arm reach from a
    base at x=0. Sliding the bases forward by ~0.2 m brings the picking
    targets to the centre of the workspace where the policy has the
    most demonstrations.
    """
    if z_offset == 0.0 and x_offset == 0.0:
        return xml
    pattern = re.compile(
        r'(<body[^>]*\bname="openarm_(?:left|right)_base_link"[^>]*\bpos=")'
        r"([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)"
        r'("[^>]*>)'
    )
    matches = list(pattern.finditer(xml))
    if len(matches) != 2:
        raise ROSConfigError(
            f"Expected exactly 2 OpenArm base bodies to lift, found {len(matches)}; "
            "composer must be updated.",
        )

    def _repl(match: re.Match[str]) -> str:
        x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
        x += x_offset
        z += z_offset
        return f"{match.group(1)}{x} {y} {z}{match.group(5)}"

    return pattern.sub(_repl, xml)


_WHITE_SKYBOX_ASSET = (
    '<texture name="openral_skybox_white" type="skybox" builtin="flat" '
    'rgb1="1 1 1" rgb2="1 1 1" width="512" height="512"/>'
)


def _rename_upstream_wrist_cameras(xml: str) -> str:
    """Rename upstream ``camera_wrist_{left,right}`` → ``wrist_{left,right}``.

    The upstream OpenArm v2 MJCF already provides wrist-mounted camera tags
    parented inside each ``openarm_*_ee_base_link`` body. Per ADR-0069 the
    canonical HAL/sensor name is ``wrist_left`` / ``wrist_right``, so the
    composer preserves the upstream camera IDs and only renames them. The
    rollout renderer may re-aim those named cameras at runtime when the
    physical hand-mounted view is occluded by the tabletop reset pose.

    No-op if the upstream camera names ever change — the caller still
    gets a syntactically valid MJCF, the wrist render path just won't
    expose ``wrist_left`` / ``wrist_right`` and the world-state
    aggregator will surface that as a stale-sensor warning.
    """
    xml = xml.replace('name="camera_wrist_left"', 'name="wrist_left"')
    xml = xml.replace('name="camera_wrist_right"', 'name="wrist_right"')
    return xml


def _inject_white_skybox(xml: str) -> str:
    """Splice a flat-white skybox texture into the MJCF's ``<asset>`` block.

    MuJoCo's passive viewer paints whatever skybox is declared. The
    upstream OpenArm v2 MJCF ships none, so the viewer falls back to
    its default dark grey — the table, robot, and cubes blend into
    the background and the user cannot tell the scene apart from a
    crash. A flat-white skybox replaces that with a clean studio
    backdrop.
    """
    if "<asset>" in xml:
        new = xml.replace(
            "<asset>",
            f"<asset>\n    {_WHITE_SKYBOX_ASSET}",
            1,
        )
        return new
    # No <asset> block — splice one in just before <worldbody>.
    return xml.replace(
        "<worldbody>",
        f"<asset>\n    {_WHITE_SKYBOX_ASSET}\n  </asset>\n  <worldbody>",
        1,
    )


def _strip_position_actuators(xml: str) -> str:
    """Drop the upstream ``<actuator>...</actuator>`` block.

    We replace with ``<motor>`` actuators of our own further down. The
    regex is single-line and pinned to the exact tag boundary the
    upstream emits, so a future MJCF reformat would surface here loudly
    rather than silently leave double actuators in place.
    """
    new, count = re.subn(
        r"<actuator>.*?</actuator>",
        "",
        xml,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ROSConfigError(
            "Did not find exactly one <actuator>...</actuator> block in the upstream "
            "OpenArm v2 MJCF — composer must be updated before motor actuators can be "
            "safely added (would leave double actuators on every joint)."
        )
    return new


def compose_openarm_tabletop_mjcf(
    *,
    strip_actuators: bool = True,
    robot_lift_z: float = 0.0,
    robot_forward_x: float = 0.0,
    white_background: bool = False,
    top_camera_pos: tuple[float, float, float] | None = None,
    top_camera_target: tuple[float, float, float] | None = None,
    top_camera_fovy: float | None = None,
    front_camera_pos: tuple[float, float, float] | None = None,
    front_camera_target: tuple[float, float, float] | None = None,
    front_camera_fovy: float | None = None,
    robot_description: RobotDescription | None = None,
) -> tuple[str, Path]:
    """Return ``(xml_string, meshdir)`` for the tabletop scene.

    The ``meshdir`` second element is the absolute path to the upstream
    MJCF's sibling ``assets/`` directory; callers pass it to
    :func:`mujoco.MjModel.from_xml_string` so meshes resolve at compile
    time without copying.

    Args:
        strip_actuators: When ``True`` (default — the robosuite OSC
            path), drop the upstream ``<position>`` actuators and
            append a fresh ``<motor>`` block so robosuite can layer its
            own OSC controllers on top. When ``False`` (the
            ``OpenArmMujocoHAL`` path used by the ROS lifecycle node),
            preserve the upstream position actuators so the HAL's
            ``data.ctrl[act_idx] = target_position`` writes still
            drive the joints — the scene bodies (table / cubes /
            drawer) are spliced in but the actuator contract is left
            alone.
        robot_lift_z: Additive offset on the OpenArm base bodies'
            z component. The upstream MJCF mounts the bases at the
            origin alongside the table, which leaves the arms hanging
            below the workspace. Pass a positive value (~0.5 m) for
            an over-the-table tabletop layout; default 0.0 preserves
            the upstream geometry.
        white_background: When ``True``, splice a flat-white skybox
            texture into the MJCF's ``<asset>`` block so the passive
            viewer shows the scene against a studio backdrop instead
            of MuJoCo's default dark grey. Default ``False``.
        top_camera_pos: ``(x, y, z)`` world position of the ``top``
            (aka ``base``) camera. ``None`` falls back to
            ``robot_description.scene_defaults.top_camera.pos`` (or the
            hard-coded openarm POV when neither is provided).
        top_camera_target: ``(x, y, z)`` world point the ``top`` camera
            aims at. ``None`` falls back to
            ``robot_description.scene_defaults.top_camera.target``.
        top_camera_fovy: Vertical field-of-view in degrees for the
            ``top`` camera. ``None`` falls back to
            ``robot_description.scene_defaults.top_camera.fovy``.
        front_camera_pos: ``(x, y, z)`` world position of the ``front``
            camera (operator-facing view from the +Y side). ``None``
            falls back to ``_FALLBACK_FRONT_CAMERA_POS``.
        front_camera_target: ``(x, y, z)`` world point the ``front``
            camera aims at. ``None`` falls back to
            ``_FALLBACK_FRONT_CAMERA_TARGET``.
        front_camera_fovy: Vertical field-of-view for the ``front``
            camera. ``None`` falls back to ``_FALLBACK_FRONT_CAMERA_FOVY``.
        robot_description: Loaded :class:`RobotDescription`. Defaults
            to :func:`load_openarm_description` (the in-tree OpenArm
            HAL constant). Drives both the actuator inventory and the
            per-robot scene defaults so this composer no longer carries
            its own copy of either.
    """
    from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

    desc = robot_description if robot_description is not None else load_openarm_description()
    actuator_specs = actuator_specs_from_description(desc)

    upstream_path = Path(ensure_openarm_v2_mjcf())
    raw = upstream_path.read_text()
    meshdir = upstream_path.parent / "assets"
    if not meshdir.is_dir():
        raise ROSConfigError(
            f"OpenArm v2 mesh dir not found at {meshdir}; the upstream MJCF assumes "
            "meshdir='assets' next to the XML — composer cannot resolve mesh paths."
        )

    body = _strip_position_actuators(raw) if strip_actuators else raw
    body = _inject_base_center_sites(body)
    body = _lift_robot_bases(body, robot_lift_z, x_offset=robot_forward_x)
    body = _rename_upstream_wrist_cameras(body)
    if white_background:
        body = _inject_white_skybox(body)

    # Defaults precedence: explicit kwarg → manifest scene_defaults →
    # hard fallback. The manifest's scene_defaults is the single source
    # of truth for the per-robot baseline; the hard fallback only
    # exists so callers handing in a bare description (no
    # ``scene_defaults`` populated) still get a working camera.
    sd_camera = desc.scene_defaults.top_camera if desc.scene_defaults else None
    if sd_camera is not None:
        default_pos = sd_camera.pos
        default_target = sd_camera.target
        default_fovy = sd_camera.fovy
    else:
        default_pos = _FALLBACK_TOP_CAMERA_POS
        default_target = _FALLBACK_TOP_CAMERA_TARGET
        default_fovy = _FALLBACK_TOP_CAMERA_FOVY

    pos = top_camera_pos or default_pos
    tgt = top_camera_target or default_target
    fovy = top_camera_fovy if top_camera_fovy is not None else default_fovy
    quat = _look_at_quat(pos, tgt)
    top_camera_xml = (
        f'<camera name="top" pos="{pos[0]} {pos[1]} {pos[2]}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
        f'fovy="{fovy}"/>'
    )

    f_pos = front_camera_pos or _FALLBACK_FRONT_CAMERA_POS
    f_tgt = front_camera_target or _FALLBACK_FRONT_CAMERA_TARGET
    f_fovy = front_camera_fovy if front_camera_fovy is not None else _FALLBACK_FRONT_CAMERA_FOVY
    f_quat = _look_at_quat(f_pos, f_tgt)
    front_camera_xml = (
        f'<camera name="front" pos="{f_pos[0]} {f_pos[1]} {f_pos[2]}" '
        f'quat="{f_quat[0]} {f_quat[1]} {f_quat[2]} {f_quat[3]}" '
        f'fovy="{f_fovy}"/>'
    )

    # Splice the scene bodies + the per-call top camera into the existing <worldbody>.
    indented = "\n    ".join(
        (_SCENE_BODIES.strip() + "\n" + top_camera_xml + "\n" + front_camera_xml).splitlines()
    )
    body, n_subs = re.subn(
        r"(</worldbody>)",
        f"    {indented}\n  \\1",
        body,
        count=1,
    )
    if n_subs != 1:
        raise ROSConfigError(
            "Could not find </worldbody> in the upstream OpenArm v2 MJCF; "
            "composer cannot splice the table/cubes/drawer into the scene."
        )

    if strip_actuators:
        # Append the new <actuator> block before </mujoco>.
        body = body.replace(
            "</mujoco>",
            f"  {_render_actuator_block(actuator_specs)}\n</mujoco>",
            1,
        )

    return body, meshdir
