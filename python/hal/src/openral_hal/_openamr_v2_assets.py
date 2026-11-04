"""Build the OpenAMRobot 2.0 MuJoCo twin from its vendored URDF.

Upstream (``openAMRobot/openamr-upperbody-sw``) ships one combined URDF, plus
MoveIt and Gazebo configs, and no MJCF. ``robots/openamr_v2/openamr_v2.urdf``
is that URDF with xacro already expanded. This module turns it into an MJCF
once per URDF revision and caches the result under
``$OPENRAL_CACHE_DIR/openamr_v2/<sha>/``. Steps:

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
* add the mimic-finger equality, the head and wrist cameras on the URDF optical
  frames, and a visual-only floor.

Robot geoms collide with the world but not with each other (``contype`` 2 /
``conaffinity`` 1), as in the bare OpenArm twin. Self-collision is the safety
kernel's job, not the twin's.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path

from openral_core.exceptions import ROSConfigError

__all__ = ["OPENAMR_V2_URDF", "ensure_openamr_v2_mjcf"]

#: The vendored, xacro-expanded URDF (repo checkout layout, as ``panda_mobile``).
OPENAMR_V2_URDF: Path = Path(__file__).resolve().parents[4] / "robots/openamr_v2/openamr_v2.urdf"

# (joint, kp, kv, damping, armature, frictionloss): OpenArm v2 motor classes.
_DM8009 = (230.0, 2.7, 1.0, 0.0081, 0.2)
_DM4340 = (190.0, 2.2, 0.9, 0.16, 0.1)
_DM4310 = (30.0, 1.5, 0.9, 0.01, 0.04)
_FINGER = (30.0, 0.2, 2.0, 0.0, 0.0)
_ARM_GAINS = (_DM8009, _DM8009, _DM4340, _DM4340, _DM4310, _DM4310, _DM4310)
# Kinematic base: stiff and critically damped, enough for a 38 kg body.
_BASE_KP = 20000.0
# openarm_mujoco's position_lifter kp (8000) sags 24 mm under the arms with
# gravity on; 1e5 holds within 2 mm. Its kv and forcerange are kept.
_LIFT_KP, _LIFT_KV, _LIFT_FORCE = 1.0e5, 4000.0, 3000.0


def _actuated_joints() -> list[tuple[str, tuple[float, float, float, float, float]]]:
    """The MJCF joint each actuator drives, in ``robots/openamr_v2/robot.yaml`` order."""
    out: list[tuple[str, tuple[float, float, float, float, float]]] = []
    for side in ("left", "right"):
        out += [(f"openarm_{side}_joint{i + 1}", g) for i, g in enumerate(_ARM_GAINS)]
        out.append((f"openarm_{side}_finger_joint1", _FINGER))
    return out


def _mujoco_urdf(urdf_xml: str) -> str:
    """``urdf_xml`` with every mesh made an absolute STL path MuJoCo can load."""
    from openral_core.assets import AssetRefError, resolve_package_uri  # noqa: PLC0415

    def _sub(match: re.Match[str]) -> str:
        uri = match.group(1)
        if uri.endswith(".dae"):
            uri = uri.replace("/visual/", "/collision/")[: -len(".dae")] + ".stl"
        try:
            return f'filename="{resolve_package_uri(uri)}"'
        except AssetRefError as exc:
            raise ROSConfigError(f"openamr_v2 MJCF build: {exc}") from exc

    xml = re.sub(r'filename="(package://[^"]+)"', _sub, urdf_xml)
    compiler = (
        '<mujoco><compiler discardvisual="false" fusestatic="false" '
        'strippath="false" angle="radian"/></mujoco>'
    )
    return re.sub(r"(<robot\b[^>]*>)", r"\1" + compiler, xml, count=1)


def _build(urdf_xml: str) -> str:
    import mujoco  # noqa: PLC0415  # reason: optional sim-only dep

    spec = mujoco.MjSpec.from_string(_mujoco_urdf(urdf_xml))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Room for a full-HD offscreen render (the default framebuffer is 640x480).
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080

    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:  # collision geom (visuals import as 0/0)
            geom.contype, geom.conaffinity = 2, 1

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

    base_kv = 2.0 * math.sqrt(_BASE_KP * 40.0)
    for name, _jtype, _axis in base_joints:
        _position(name, _BASE_KP, base_kv)
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
        ("head", "head_camera_optical_frame"),
        ("wrist_left", "left_wrist_camera_optical_frame"),
        ("wrist_right", "right_wrist_camera_optical_frame"),
    ):
        # 3 cm out along the optical axis, clear of the mount's own marker box.
        spec.body(frame_body).add_camera(
            name=camera, pos=[0, 0, 0.03], quat=[0, 1, 0, 0], fovy=55.4
        )

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


def ensure_openamr_v2_mjcf(urdf_path: Path = OPENAMR_V2_URDF) -> str:
    """Return the cached OpenAMRobot 2.0 MJCF path, building it on first use.

    The cache key hashes the URDF and this module's source, so an edited
    URDF or builder never serves a stale twin. The write is atomic (a unique
    temp file renamed into place), so concurrent workers never read a partial file.

    Raises:
        ROSConfigError: The URDF is missing, a mesh cannot be resolved, or
            MuJoCo rejects the model.

    Example:
        >>> ensure_openamr_v2_mjcf(Path("/nonexistent.urdf"))
        Traceback (most recent call last):
        ...
        openral_core.exceptions.ROSConfigError: openamr_v2 URDF not found: /nonexistent.urdf
    """
    if not urdf_path.is_file():
        raise ROSConfigError(f"openamr_v2 URDF not found: {urdf_path}")
    urdf_xml = urdf_path.read_text()
    builder = Path(__file__).read_bytes()  # an edited builder never serves a stale twin
    key = hashlib.sha256(builder + urdf_xml.encode()).hexdigest()[:16]
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    out = base / "openamr_v2" / key / "openamr_v2.xml"
    if out.is_file():
        return str(out)
    try:
        xml = _build(urdf_xml)
    except ValueError as exc:  # mujoco's spec/compile errors
        raise ROSConfigError(f"openamr_v2 MJCF build failed: {exc}") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{uuid.uuid4().hex}.xml")
    tmp.write_text(xml)
    tmp.replace(out)
    return str(out)
