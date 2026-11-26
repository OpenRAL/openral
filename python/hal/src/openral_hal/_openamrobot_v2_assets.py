"""Build the OpenAMRobot 2.0 MuJoCo twin from its vendored URDF.

Upstream (``openAMRobot/openamr-upperbody-sw``) ships one combined URDF, plus
MoveIt and Gazebo configs, and no MJCF. ``robots/openamrobot_v2/openamrobot_v2.urdf``
is upstream's own expansion of it with OpenRAL's camera overlay
(``tools/openamrobot_v2_urdf.py``). This module turns it into an MJCF
once per URDF revision and caches the result under
``$OPENRAL_CACHE_DIR/openamrobot_v2/<sha>/``. Steps:

* resolve ``package://`` meshes via ``openral_core.assets.resolve_package_uri``
  (the pinned ``openarm_description`` / ``openamrobot_description`` clones). The
  OpenArm visual ``.dae`` meshes become their collision ``.stl`` twins, because
  MuJoCo reads no COLLADA.
* import the URDF with MuJoCo's native compiler, then free the base in the floor
  plane: ``base_x``/``base_y`` slides, a ``base_yaw`` hinge and a ``base_z`` slide on
  ``base_footprint``. None is actuated: the base moves on upstream's two drive wheels
  (``left/right_wheel_joint``, continuous), each with a ``<velocity>`` actuator, against
  a colliding floor, on four frictionless caster spheres (a swivel caster resists no
  direction). ``base_z`` lets gravity load the wheels; roll and pitch stay rigid.
* add one ``<position>`` actuator per manifest joint but the base, in manifest order. Arm and
  finger gains come from ``enactic/openarm_mujoco`` v2 (``openarm_bimanual.xml``
  @ 1daab06: DM8009 kp 230 kv 2.7, DM4340 kp 190 kv 2.2, DM4310 kp 30 kv 1.5,
  finger kp 30 kv 0.2). Joint damping and frictionloss come from the URDF's
  ``<dynamics>`` (upstream's ``sim_dynamics:=on`` expansion) and armature from the
  vendored ``sim_dynamics_openarm_v2.yaml``. The lift takes that file's ``position_lifter`` kv and
  forcerange with a stiffer kp (see ``_LIFT_KP``). The wheel gain (``_WHEEL_KV``) is a
  velocity servo, not motor data. Gravity is on: everything that rides the lift
  carriage is gravity-compensated (``gravcomp``), as a real arm's controller does, so
  the OpenArm v2 gains hold the arms while the base and column load the wheels.
* add the mimic-finger equality, the wrist cameras on the URDF optical frames
  (OpenArm's wrist mount),
  and the floor. The head ZED is spliced by the generic camera rig
  from the manifest's ``sim_placement``, on ``openarm_left_base_link`` at the OpenArm
  shelf cell's head-camera pose. The URDF's ``head_camera_link`` (upstream's ZED body
  box) sits 2.5 mm behind that lens; the rig's camera is not a URDF frame.

Robot geoms collide with the world but not with each other (``contype`` 2 /
``conaffinity`` 1), as in the bare OpenArm twin. Self-collision is the safety
kernel's job, not the twin's.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from openral_core.exceptions import ROSConfigError

from openral_hal import _openamrobot_platform_assets, _openarm_description_assets

__all__ = [
    "BASE_Z_JOINT",
    "OPENAMROBOT_V2_URDF",
    "UPSTREAM_URDF",
    "WHEEL_JOINTS",
    "compose_openamrobot_v2_walls_mjcf",
    "diff_drive_accel_limits",
    "ensure_openamrobot_v2_mjcf",
]

#: The vendored, xacro-expanded URDF (repo checkout layout, as ``panda_mobile``).
OPENAMROBOT_V2_URDF: Path = (
    Path(__file__).resolve().parents[4] / "robots/openamrobot_v2/openamrobot_v2.urdf"
)
#: Upstream's own expansion, vendored unmodified (``tools/openamrobot_v2_urdf.py``).
UPSTREAM_URDF: Path = OPENAMROBOT_V2_URDF.with_name("upstream") / "combined_robot.urdf"

# Upstream's simulator joint dynamics (openamr-upperbody-sw
# config/sim_dynamics/openarm_v2.yaml, vendored unchanged). The URDF is expanded with
# sim_dynamics:=on, so each OpenArm joint's damping and friction arrive in its
# <dynamics> element; armature has no URDF element and is read from this file.
_SIM_DYNAMICS: Path = OPENAMROBOT_V2_URDF.with_name("sim_dynamics_openarm_v2.yaml")
# (kp, kv) position-actuator gains: openarm_mujoco v2 (openarm_bimanual.xml @ 1daab06,
# unchanged since 45d4c29). Upstream's profile carries no actuator gains.
_DM8009 = (230.0, 2.7)
_DM4340 = (190.0, 2.2)
_DM4310 = (30.0, 1.5)
_FINGER = (30.0, 0.2)
_ARM_GAINS = (_DM8009, _DM8009, _DM4340, _DM4340, _DM4310, _DM4310, _DM4310)
#: The passive slide on ``base_footprint`` (after the manifest's planar ``base_joints``)
#: that lets gravity load the wheels.
BASE_Z_JOINT: str = "base_z"
#: Upstream's drive wheels (continuous joints on ``base_link``), left then right.
WHEEL_JOINTS: tuple[str, str] = ("left_wheel_joint", "right_wheel_joint")
# Wheel servo (MuJoCo intvelocity: a speed command integrated into a position target),
# N*m per rad and N*m per rad/s. Tuning values, not ZLAC8015D motor data.
_WHEEL_KP, _WHEEL_KV = 500.0, 50.0
# Bound on the integrated wheel angle MuJoCo requires; 1e6 rad is ~27 h at full speed.
_WHEEL_ACT_LIMIT_RAD = 1.0e6
# Caster spheres (the URDF's fixed `*_caster_link` / `*_wheel_link` pairs): MuJoCo has
# no swivel caster, so their contact is frictionless and they only carry load.
_CASTER_LINKS: tuple[str, ...] = (
    "fl_wheel_link",
    "fr_wheel_link",
    "bl_wheel_link",
    "br_wheel_link",
)
#: Every body on (or below) this link rides the lift and is gravity-compensated.
_GRAVCOMP_ROOT = "lift_carriage_link"
# openarm_mujoco's position_lifter kp (8000) sags 24 mm under the arms with
# gravity on; 1e5 holds within 2.3 mm (upstream's 8 Nov carriage masses). Its kv
# and forcerange are kept.
_LIFT_KP, _LIFT_KV, _LIFT_FORCE = 1.0e5, 4000.0, 3000.0


def diff_drive_accel_limits(urdf_path: Path = UPSTREAM_URDF) -> tuple[float, float]:
    """``(linear m/s^2, angular rad/s^2)`` of upstream's DiffDrive plugin (its ``max_*``).

    Upstream configures its Gazebo diff drive in the vendored, unmodified
    ``upstream/combined_robot.urdf``; the twin ramps a wheel command by these limits,
    as that controller does, so a step command does not spin the wheels.

    Raises:
        ROSConfigError: The file has no DiffDrive plugin or it lacks a limit.

    Example:
        >>> diff_drive_accel_limits()
        (0.5, 1.0)
    """
    root = ET.parse(urdf_path).getroot()
    for plugin in root.iter("plugin"):
        if plugin.get("name", "").endswith("DiffDrive"):
            try:
                return (
                    float(plugin.findtext("max_linear_acceleration", "")),
                    float(plugin.findtext("max_angular_acceleration", "")),
                )
            except ValueError as exc:
                raise ROSConfigError(f"{urdf_path}: DiffDrive acceleration limits: {exc}") from exc
    raise ROSConfigError(f"{urdf_path}: no DiffDrive plugin")


def _actuated_joints() -> list[tuple[str, tuple[float, float]]]:
    """The MJCF joint each actuator drives, in ``robots/openamrobot_v2/robot.yaml`` order."""
    out: list[tuple[str, tuple[float, float]]] = []
    for side in ("left", "right"):
        out += [(f"openarm_{side}_joint{i + 1}", g) for i, g in enumerate(_ARM_GAINS)]
        out.append((f"openarm_{side}_finger_joint1", _FINGER))
    return out


def _mujoco_urdf(urdf_xml: str, alias_dir: Path) -> str:
    """``urdf_xml`` with every mesh made an absolute STL path MuJoCo can load.

    MuJoCo's URDF compiler names a mesh asset after its file and ignores ``scale``
    when it reuses one. The OpenArm macro draws the left arm from the same STLs as the
    right, mirrored with ``scale="1 -1 1"``, so both arms would get whichever mirror
    came first (the right arm's collision meshes ended up as the left's). Each
    distinct (file, scale) pair therefore gets its own file name: a symlink under
    ``alias_dir`` that lives with the cached MJCF.
    """
    from openral_core.assets import AssetRefError, resolve_package_uri  # noqa: PLC0415

    aliases: dict[tuple[str, str], str] = {}

    def _sub(match: re.Match[str]) -> str:
        uri, rest = match.group(1), match.group(2)
        if uri.endswith(".dae"):
            uri = uri.replace("/visual/", "/collision/")[: -len(".dae")] + ".stl"
        try:
            path = resolve_package_uri(uri)
        except AssetRefError as exc:
            raise ROSConfigError(f"openamrobot_v2 MJCF build: {exc}") from exc
        scale = re.search(r'scale="([^"]+)"', rest)
        key = (str(path), scale.group(1) if scale else "1 1 1")
        if key not in aliases:
            alias = alias_dir / f"{path.stem}_{len(aliases)}{path.suffix}"
            alias.symlink_to(path)
            aliases[key] = str(alias)
        return f'<mesh filename="{aliases[key]}"{rest}'

    xml = re.sub(r'<mesh filename="(package://[^"]+)"([^>]*>)', _sub, urdf_xml)
    compiler = (
        '<mujoco><compiler discardvisual="false" fusestatic="false" '
        'strippath="false" angle="radian"/></mujoco>'
    )
    return re.sub(r"(<robot\b[^>]*>)", r"\1" + compiler, xml, count=1)


def _build(urdf_xml: str, alias_dir: Path) -> str:
    import mujoco  # noqa: PLC0415  # reason: optional sim-only dep

    spec = mujoco.MjSpec.from_string(_mujoco_urdf(urdf_xml, alias_dir))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Room for a full-HD offscreen render (the default framebuffer is 640x480).
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080

    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:  # collision geom (visuals import as 0/0)
            geom.contype, geom.conaffinity = 2, 1
            # Group 3 is not rendered by default, so cameras and viewers draw the
            # visual meshes instead of the coarse collision shells drawn over them
            # (the chassis box hid the base meshes). Contacts and mj_ray (lidar,
            # depth ray-cast) ignore groups unless asked, so they still hit these.
            geom.group = 3

    base = spec.body("base_footprint")
    slide, hinge = mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE
    base_joints = [("base_x", slide, (1, 0, 0)), ("base_y", slide, (0, 1, 0))]
    base_joints += [("base_yaw", hinge, (0, 0, 1)), (BASE_Z_JOINT, slide, (0, 0, 1))]
    # base_footprint is the URDF root and the first body, so its joints lead qpos
    # (0..3) in this order, ahead of the wheels, lift_joint and the arms. Unactuated:
    # the wheels move the base, and these joints measure where it is.
    for name, jtype, axis in base_joints:
        base.add_joint(name=name, type=jtype, axis=axis)
    for link in _CASTER_LINKS:
        for geom in spec.body(link).geoms:
            if geom.contype or geom.conaffinity:
                geom.friction = [0.0, 0.0, 0.0]

    def _gravcomp(body: object) -> None:
        body.gravcomp = 1.0  # type: ignore[attr-defined]  # reason: mujoco MjsBody
        for child in body.bodies:  # type: ignore[attr-defined]  # reason: mujoco MjsBody
            _gravcomp(child)

    _gravcomp(spec.body(_GRAVCOMP_ROOT))

    def _position(joint: str, kp: float, kv: float, force: float | None = None) -> None:
        act = spec.add_actuator(name=f"{joint}_ctrl", target=joint)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        if force is not None:
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-force, force]

    _position("lift_joint", _LIFT_KP, _LIFT_KV, _LIFT_FORCE)
    for joint_name, (kp, kv) in _actuated_joints():
        _position(joint_name, kp, kv)
    # Damping and frictionloss came in with the URDF's <dynamics>; armature has no URDF
    # element, so it is set here from upstream's profile (fingers included).
    profile = yaml.safe_load(_SIM_DYNAMICS.read_text())["joints"]
    for joint_name, entry in profile.items():
        spec.joint(joint_name).armature = float(entry["armature"]["value"])
    # Every position actuator clamps to its joint range.
    for act in spec.actuators:
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.inheritrange = 1.0
    # The wheels last (actuators 17, 18), after the manifest's position actuators.
    for wheel in WHEEL_JOINTS:
        act = spec.add_actuator(name=f"{wheel}_vel", target=wheel)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        # Integrated velocity: ctrl is a wheel speed, the actuator's activation integrates
        # it into a position target, and a PD servo tracks that. A zero speed then holds
        # the wheel where it is (a driver's brake), instead of letting the base creep.
        act.set_to_intvelocity(kp=_WHEEL_KP, kv=_WHEEL_KV)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.actlimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.actrange = [-_WHEEL_ACT_LIMIT_RAD, _WHEEL_ACT_LIMIT_RAD]

    for side in ("left", "right"):
        spec.add_equality(
            name=f"openarm_{side}_finger_mimic",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=f"openarm_{side}_finger_joint1",
            name2=f"openarm_{side}_finger_joint2",
            data=[0, 1, 0, 0, 0] + [0] * 6,
            solref=[0.005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )

    # Optical frames are REP-103 (z forward, y down); a MuJoCo camera looks
    # down -z with +y up, so a half-turn about x maps one onto the other.
    for camera, frame_body in (
        ("wrist_left", "left_wrist_camera_optical_frame"),
        ("wrist_right", "right_wrist_camera_optical_frame"),
    ):
        spec.body(frame_body).add_camera(name=camera, pos=[0, 0, 0], quat=[0, 1, 0, 0], fovy=55.4)

    # A 1 m checker so camera frames show the base moving.
    spec.add_texture(
        name="floor_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.42, 0.44, 0.47],
        rgb2=[0.55, 0.57, 0.6],
        width=512,
        height=512,
    )
    floor_mat = spec.add_material(name="floor_grid", texrepeat=[1, 1], texuniform=True)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="floor_grid",
        # Touches the robot's geoms (contype 2) only: world geoms (walls, the walls
        # scene's sinking boxes) pass through it, as before.
        contype=0,
        conaffinity=2,
    )
    spec.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.compile()
    return str(spec.to_xml())


def ensure_openamrobot_v2_mjcf(urdf_path: Path = OPENAMROBOT_V2_URDF) -> str:
    """Return the cached OpenAMRobot 2.0 MJCF path, building it on first use.

    The cache key hashes the URDF, this module's source and both mesh-clone pins,
    so an edited URDF or builder, or a bumped pin, never serves a stale twin. The
    entry (MJCF plus mesh aliases) is built in a staging dir and renamed into place,
    so concurrent workers never read a partial one.

    Raises:
        ROSConfigError: The URDF is missing, a mesh cannot be resolved, or
            MuJoCo rejects the model.

    Example:
        >>> ensure_openamrobot_v2_mjcf(Path("/nonexistent.urdf"))
        Traceback (most recent call last):
        ...
        openral_core.exceptions.ROSConfigError: openamrobot_v2 URDF not found: /nonexistent.urdf
    """
    if not urdf_path.is_file():
        raise ROSConfigError(f"openamrobot_v2 URDF not found: {urdf_path}")
    urdf_xml = urdf_path.read_text()
    # An edited builder or dynamics profile never serves a stale twin.
    builder = Path(__file__).read_bytes() + _SIM_DYNAMICS.read_bytes()
    # The mesh aliases point into the pinned clones: a bumped pin is a new entry.
    pins = f"{_openarm_description_assets._PINNED_SHA}:{_openamrobot_platform_assets._PINNED_SHA}"
    key = hashlib.sha256(builder + urdf_xml.encode() + pins.encode()).hexdigest()[:16]
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    out = base / "openamrobot_v2" / key / "openamrobot_v2.xml"
    if out.is_file():
        return str(out)
    # Build in a private staging dir (the MJCF plus its mesh aliases), point the
    # MJCF at the final dir, then rename the dir into place whole: a concurrent
    # worker sees the finished entry or none, never a partial one.
    staging = out.parent.with_name(f".{key}.{uuid.uuid4().hex}")
    (staging / "meshes").mkdir(parents=True)
    try:
        xml = _build(urdf_xml, staging / "meshes")
        (staging / out.name).write_text(xml.replace(str(staging), str(out.parent)))
        try:
            staging.rename(out.parent)
        except OSError:
            if not out.is_file():  # not a concurrent winner: a real failure
                raise
    except ValueError as exc:  # mujoco's spec/compile errors
        raise ROSConfigError(f"openamrobot_v2 MJCF build failed: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(out)


# (name, centre xy, half-extents xy): walls of a 20 m x 20 m room around the
# spawn pose, plus pillars that break its symmetry so scan matching has one fit.
_ROOM_WALLS: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
    ("wall_east", (10.0, 0.0), (0.1, 10.1)),
    ("wall_west", (-10.0, 0.0), (0.1, 10.1)),
    ("wall_north", (0.0, 10.0), (10.1, 0.1)),
    ("wall_south", (0.0, -10.0), (10.1, 0.1)),
    ("pillar_a", (6.0, 3.0), (0.3, 0.3)),
    ("pillar_b", (-4.0, 6.5), (0.5, 0.25)),
    ("pillar_c", (3.0, -7.0), (0.25, 0.6)),
)
_WALL_HEIGHT_M = 1.5


# A transient box: 2 m tall on a vertical slide, pulled by a joint spring towards a
# rest 3 m below the floor through heavy damping (time constant damping / stiffness
# = 90 s), so it stands in the lidar plane while the graph comes up and leaves it
# ~80 s of sim time after connect: a costmap obstacle that disappears. A spring, not
# gravity: the box is gravity-compensated, so its fall stays the spring's.
_TRANSIENT_HALF = (0.1, 0.1, 1.0)
_TRANSIENT_REST_M, _TRANSIENT_STIFFNESS, _TRANSIENT_DAMPING = -3.0, 10.0, 900.0


def compose_openamrobot_v2_walls_mjcf(
    sinking_boxes: list[list[float]] | None = None,
) -> tuple[str, Path]:
    """The twin inside a 20 m walled room with three pillars (``composition`` hook).

    A ground-truth world for odometry, ``/scan`` and SLAM: every wall and pillar
    is at a known pose (``_ROOM_WALLS``), so a scan or a map can be checked against it.

    Args:
        sinking_boxes: ``[x, y]`` centres of 0.2 m x 0.2 m x 2 m boxes that start
            standing and sink through the floor, leaving the lidar plane ~80 s of
            sim time after connect: obstacles a costmap must clear.

    Returns:
        ``(xml, meshdir)``: the scene MJCF and the twin's mesh-alias dir.
    """
    import mujoco  # noqa: PLC0415  # reason: optional sim-only dep

    twin = Path(ensure_openamrobot_v2_mjcf())
    spec = mujoco.MjSpec.from_file(str(twin))
    for name, (x, y), (hx, hy) in _ROOM_WALLS:
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, _WALL_HEIGHT_M / 2],
            size=[hx, hy, _WALL_HEIGHT_M / 2],
            rgba=[0.75, 0.62, 0.45, 1.0],
        )
    for i, (x, y) in enumerate(sinking_boxes or ()):
        body = spec.worldbody.add_body(
            name=f"sinking_box_{i}", pos=[x, y, _TRANSIENT_HALF[2]], gravcomp=1.0
        )
        body.add_joint(
            name=f"sinking_box_{i}_z",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            stiffness=[_TRANSIENT_STIFFNESS, 0.0, 0.0],
            springref=_TRANSIENT_REST_M,
            damping=[_TRANSIENT_DAMPING, 0.0, 0.0],
            range=[_TRANSIENT_REST_M, 0.0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(_TRANSIENT_HALF),
            mass=1.0,
            rgba=[0.85, 0.25, 0.2, 1.0],
        )
    spec.compile()
    return str(spec.to_xml()), twin.parent / "meshes"
