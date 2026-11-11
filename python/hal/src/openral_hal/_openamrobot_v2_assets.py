"""Build the OpenAMRobot 2.0 MuJoCo twin from its vendored URDF.

Upstream (``openAMRobot/openamr-upperbody-sw``) ships one combined URDF, plus
MoveIt and Gazebo configs, and no MJCF. ``robots/openamrobot_v2/openamrobot_v2.urdf``
is that URDF with xacro already expanded. This module turns it into an MJCF
once per URDF revision and caches the result under
``$OPENRAL_CACHE_DIR/openamrobot_v2/<sha>/``. Steps:

* resolve ``package://`` meshes via ``openral_core.assets.resolve_package_uri``
  (the pinned ``openarm_description`` / ``openamrobot_description`` clones). The
  OpenArm visual ``.dae`` meshes become their collision ``.stl`` twins, because
  MuJoCo reads no COLLADA.
* import the URDF with MuJoCo's native compiler, then add a planar base:
  ``base_x``/``base_y`` slides and a ``base_yaw`` hinge on ``base_footprint``.
* add one ``<position>`` actuator per manifest joint, in manifest order. Arm and
  finger gains come from ``enactic/openarm_mujoco`` v2 (``openarm_v20_bimanual.xml``
  @ 45d4c29: DM8009 kp 230 kv 2.7, DM4340 kp 190 kv 2.2, DM4310 kp 30 kv 1.5,
  finger kp 30 kv 0.2). The lift takes that file's ``position_lifter`` kv and
  forcerange with a stiffer kp (see ``_LIFT_KP``). The base gains are stiff
  tracking gains for a kinematic base, not motor data.
* add the mimic-finger equality, the wrist cameras on the URDF optical frames
  (OpenArm's wrist mount),
  and a visual-only floor. The head ZED is spliced by the generic camera rig
  from the manifest's ``sim_placement``, on upstream's ``head_camera_link`` (not
  OpenArm's measured mount, which lands inside the Getac tablet here).

Robot geoms collide with the world but not with each other (``contype`` 2 /
``conaffinity`` 1), as in the bare OpenArm twin. Self-collision is the safety
kernel's job, not the twin's.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import uuid
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal import _openamrobot_platform_assets, _openarm_description_assets

__all__ = ["OPENAMROBOT_V2_URDF", "compose_openamrobot_v2_walls_mjcf", "ensure_openamrobot_v2_mjcf"]

#: The vendored, xacro-expanded URDF (repo checkout layout, as ``panda_mobile``).
OPENAMROBOT_V2_URDF: Path = (
    Path(__file__).resolve().parents[4] / "robots/openamrobot_v2/openamrobot_v2.urdf"
)

# (joint, kp, kv, damping, armature, frictionloss): OpenArm v2 motor classes.
_DM8009 = (230.0, 2.7, 1.0, 0.0081, 0.2)
_DM4340 = (190.0, 2.2, 0.9, 0.16, 0.1)
_DM4310 = (30.0, 1.5, 0.9, 0.01, 0.04)
_FINGER = (30.0, 0.2, 2.0, 0.0, 0.0)
_ARM_GAINS = (_DM8009, _DM8009, _DM4340, _DM4340, _DM4310, _DM4310, _DM4310)
# Kinematic base: a critically damped tracker at a fixed natural frequency, so its
# lag does not change when upstream's masses do (87 kg since openamr-platform-sw
# #126). sqrt(500) rad/s is the old kp 20000 on a 40 kg body. Not motor data.
_BASE_OMEGA_RAD_S = math.sqrt(500.0)
# openarm_mujoco's position_lifter kp (8000) sags 24 mm under the arms with
# gravity on; 1e5 holds within 2.3 mm (upstream's 8 Nov carriage masses). Its kv
# and forcerange are kept.
_LIFT_KP, _LIFT_KV, _LIFT_FORCE = 1.0e5, 4000.0, 3000.0


def _srgb(hex_rgb: str) -> list[float]:
    return [int(hex_rgb[i : i + 2], 16) / 255.0 for i in (1, 3, 5)] + [1.0]


# Visual finishes (presentation only: no geometry, mass or contact change). Upstream's
# URDF draws the whole upper body in one placeholder colour; these are the colours its
# own roadmap model renders (openAMRobot/openamrobot-docs, docs/assets/roadmap/model:
# finish.json RAL references, else model.js materials alu #C9CDD2, anod #D7DBDE, steel
# #4A4F55, black #15181C). Links the roadmap model draws only as placeholder envelopes
# (Getac, head light, second E-stop) keep upstream's placeholder colour.
_LINK_FINISH: dict[str, list[float]] = {
    "lift_column_base_link": _srgb("#D7DBDE"),  # anodised column body (model.js anod)
    "lift_carriage_link": _srgb("#C9CDD2"),  # lift stage (model.js alu)
    "adapter_plate_link": _srgb("#0A0A0A"),  # RAL 9005, satin (finish.json)
    "arm_profile_link": _srgb("#383E42"),  # RAL 7016, satin (finish.json)
    "head_camera_link": _srgb("#15181C"),  # ZED Mini body (model.js black)
    "head_camera_bracket_link": _srgb("#C9CDD2"),
    "lift_flag_rod_link": _srgb("#C9CDD2"),
    "lift_flag_link": _srgb("#4A4F55"),
    "lift_stop_sensor_l1000_link": _srgb("#4A4F55"),
    "lift_stop_sensor_l1175_link": _srgb("#4A4F55"),
    "lift_stop_sensor_l1350_link": _srgb("#4A4F55"),
}
# base_link carries three visual meshes: the steel shell (side, front and rear panels,
# RAL 7035) and the top cover (RAL 7016), as finish.json; the equipment keeps its own.
_BASE_MESH_FINISH: dict[str, list[float]] = {
    "base_chassis": _srgb("#CBD0CC"),
    "base_covers": _srgb("#383E42"),
}


def _apply_finish(geom: object) -> None:
    """Colour one visual geom from ``_LINK_FINISH`` / ``_BASE_MESH_FINISH`` (else unchanged)."""
    body = geom.parent.name  # type: ignore[attr-defined]  # reason: mujoco MjsGeom
    if body in _LINK_FINISH:
        geom.rgba = _LINK_FINISH[body]  # type: ignore[attr-defined]  # reason: mujoco MjsGeom
    elif body == "base_link":
        for stem, rgba in _BASE_MESH_FINISH.items():
            if geom.meshname.startswith(f"{stem}_"):  # type: ignore[attr-defined]  # reason: mujoco MjsGeom
                geom.rgba = rgba  # type: ignore[attr-defined]  # reason: mujoco MjsGeom


def _actuated_joints() -> list[tuple[str, tuple[float, float, float, float, float]]]:
    """The MJCF joint each actuator drives, in ``robots/openamrobot_v2/robot.yaml`` order."""
    out: list[tuple[str, tuple[float, float, float, float, float]]] = []
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
        else:
            _apply_finish(geom)

    base = spec.body("base_footprint")
    slide, hinge = mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE
    base_joints = [("base_x", slide, (1, 0, 0)), ("base_y", slide, (0, 1, 0))]
    base_joints.append(("base_yaw", hinge, (0, 0, 1)))
    # base_footprint is the URDF root and the first body, so its joints lead
    # qpos (0..2) in this order, ahead of lift_joint and the arms.
    for name, jtype, axis in base_joints:
        base.add_joint(name=name, type=jtype, axis=axis)

    def _position(joint: str, kp: float, kv: float, force: float | None = None) -> None:
        act = spec.add_actuator(name=f"{joint}_ctrl", target=joint)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        if force is not None:
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-force, force]

    # Gains from the mass the slides actually carry (the whole robot), so the base
    # stays critically damped whatever upstream's masses are.
    compiled = spec.compile()
    base_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")
    base_mass = float(compiled.body_subtreemass[base_id])
    base_kp = base_mass * _BASE_OMEGA_RAD_S**2
    base_kv = 2.0 * base_mass * _BASE_OMEGA_RAD_S
    for name, _jtype, _axis in base_joints:
        _position(name, base_kp, base_kv)
    _position("lift_joint", _LIFT_KP, _LIFT_KV, _LIFT_FORCE)
    for joint_name, (kp, kv, damping, armature, friction) in _actuated_joints():
        joint = spec.joint(joint_name)
        joint.damping = [damping, 0.0, 0.0]  # mujoco >=3.8: polynomial damping
        joint.armature, joint.frictionloss = armature, friction
        _position(joint_name, kp, kv)
    # Every non-base actuator clamps to its joint range (the base is unbounded).
    for act in spec.actuators:
        if act.target not in {"base_x", "base_y", "base_yaw"}:
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.inheritrange = 1.0

    for side in ("left", "right"):
        spec.joint(f"openarm_{side}_finger_joint2").damping = [2.0, 0.0, 0.0]
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
        contype=0,
        conaffinity=0,
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
    builder = Path(__file__).read_bytes()  # an edited builder never serves a stale twin
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
# gravity: the deploy twin runs with gravity off (robot.yaml hal.sim params).
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
        body = spec.worldbody.add_body(name=f"sinking_box_{i}", pos=[x, y, _TRANSIENT_HALF[2]])
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
