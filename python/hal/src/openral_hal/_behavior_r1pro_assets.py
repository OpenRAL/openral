"""The BEHAVIOR-1K Galaxea R1 Pro URDF, fetched on demand and made loadable.

``robots/r1pro/robot.yaml`` describes the R1 Pro that the 2026 BEHAVIOR-1K
challenge simulates; its joint limits were transcribed from
``behavior-1k/omnigibson-robot-assets`` (a Hugging Face dataset, MIT) at
``BEHAVIOR_ASSETS_REVISION``, ``models/r1pro/urdf/r1pro.urdf``. This module
downloads that URDF and the visual meshes beside it at that exact revision and
writes one derived copy into ``$OPENRAL_CACHE_DIR/behavior_r1pro`` with:

* every mesh ``filename`` made absolute, so the derived file can live outside
  the Hugging Face snapshot; and
* every ``<collision>`` whose mesh is not in the dataset dropped. Upstream
  references ``meshes/collision/*-col-*.obj`` files it does not ship, so no
  loader (MuJoCo, yourdfpy) can open the file as published. The result has
  visual geometry only; collision lowering has no collision meshes to read.

Nothing is vendored: the dataset revision pins the content.

``ensure_r1pro_mjcf`` builds the deploy-sim MuJoCo twin from that URDF (cached
under ``$OPENRAL_CACHE_DIR/behavior_r1pro/<rev12>/mjcf/<sha>/``, keyed on the URDF,
this module's source and the mujoco version):

* **Collision geometry.** The URDF ships none, so every visual mesh is also a
  collision geom. MuJoCo collides a mesh as its convex hull, so each link is
  as large as the hull of its visual mesh (a conservative envelope, never
  smaller than the part). The collision copies sit in group 3 (not rendered)
  and collide with the world but not with each other (``contype`` 2 /
  ``conaffinity`` 1): self-collision is the safety kernel's job, not the twin's.
* **Base.** The three swerve steer/wheel pairs become fixed (no wheel model),
  and ``base_link`` gets planar ``base_x`` / ``base_y`` slides and a
  ``base_yaw`` hinge, qpos 0..2. The twin places the base kinematically per
  ``BODY_TWIST`` (``mujoco_twin._PlanarBaseTwin``); the base has no actuators.
* **Actuators.** One ``<position>`` actuator per manifest joint except each
  hand's ``finger_joint2``, in manifest order (torso, left arm, left finger,
  right arm, right finger: 20). ``finger_joint2`` follows ``finger_joint1``
  through a joint equality, as OmniGibson drives both fingers from one gripper
  command. Gains are a stiff position loop clamped to the URDF effort limits,
  not motor data (BEHAVIOR publishes none for MuJoCo).
* A visual-only checker floor and a light. Cameras are spliced by the generic
  camera rig from the manifest's ``sim_placement``s, on the URDF camera links.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from openral_core.exceptions import ROSConfigError

__all__ = ["BEHAVIOR_ASSETS_REVISION", "ensure_r1pro_mjcf", "ensure_r1pro_urdf"]

#: ``behavior-1k/omnigibson-robot-assets`` commit the r1pro manifest cites.
BEHAVIOR_ASSETS_REVISION: str = "c325bd3b6d1560d3145f3c9c0e885e79e47f180c"
_REPO_ID = "behavior-1k/omnigibson-robot-assets"
_URDF_REL = "models/r1pro/urdf/r1pro.urdf"
_MESH_GLOB = "models/r1pro/urdf/meshes/*"
_COLLISION = re.compile(r"<collision\b.*?</collision>", re.S)
_FILENAME = re.compile(r'filename="([^"]+)"')


def _derived_path() -> Path:
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "behavior_r1pro" / BEHAVIOR_ASSETS_REVISION[:12] / "r1pro.urdf"


def derive_urdf(urdf_xml: str, urdf_dir: Path) -> str:
    """``urdf_xml`` with absolute mesh paths and its unshipped collisions removed.

    Paths are joined, not ``resolve()``-d: a Hugging Face snapshot file is a
    symlink to an extension-less blob, and the loaders pick a mesh decoder by
    the extension.
    """

    def keep(collision: re.Match[str]) -> str:
        files = _FILENAME.findall(collision.group(0))
        return collision.group(0) if all((urdf_dir / f).is_file() for f in files) else ""

    xml = _COLLISION.sub(keep, urdf_xml)

    def absolute(m: re.Match[str]) -> str:
        return m.group(0) if m.group(1).startswith("/") else f'filename="{urdf_dir / m.group(1)}"'

    return _FILENAME.sub(absolute, xml)


def ensure_r1pro_urdf() -> Path:
    """Return the derived R1 Pro URDF, downloading the pinned dataset files if needed.

    Raises:
        ROSConfigError: ``huggingface_hub`` is missing, or the download failed
            (``.__cause__`` carries the hub error).
    """
    out = _derived_path()
    if out.is_file() and all(
        Path(f).is_file() for f in _FILENAME.findall(out.read_text(encoding="utf-8"))
    ):
        return out  # a cleaned Hugging Face cache (meshes gone) re-downloads below
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415  # reason: optional dep
    except ImportError as exc:
        raise ROSConfigError(f"behavior:r1pro needs huggingface_hub: {exc}") from exc
    try:
        root = snapshot_download(
            _REPO_ID,
            repo_type="dataset",
            revision=BEHAVIOR_ASSETS_REVISION,
            allow_patterns=[_URDF_REL, _MESH_GLOB],
        )
    except Exception as exc:  # every hub/network failure is a fetch failure
        raise ROSConfigError(
            f"Failed to fetch {_REPO_ID}@{BEHAVIOR_ASSETS_REVISION[:10]} {_URDF_REL}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    urdf = Path(root) / _URDF_REL
    xml = derive_urdf(urdf.read_text(encoding="utf-8"), urdf.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".r1pro.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(xml)
    os.replace(tmp, out)
    return out


_SWERVE_JOINTS = 6  # three steer + three wheel joints
_SWERVE = re.compile(
    r'(<joint name="(?:steer|wheel)_motor_joint\d" type=)"(?:revolute|continuous)"'
)
# (joint-name part, kp, kv): a stiff position loop per joint class; forcerange is the
# URDF effort limit. Not motor data: sized so the gravity-on twin holds a pose.
_GAINS: tuple[tuple[str, float, float], ...] = (
    ("torso_joint", 3000.0, 150.0),
    ("arm_joint1", 1500.0, 60.0),
    ("arm_joint2", 1500.0, 60.0),
    ("arm_joint3", 600.0, 20.0),
    ("arm_joint4", 600.0, 20.0),
    ("arm_joint5", 300.0, 8.0),
    ("arm_joint6", 300.0, 8.0),
    ("arm_joint7", 300.0, 8.0),
    ("gripper_finger_joint1", 1000.0, 20.0),
)


def _actuated() -> list[str]:
    """The MJCF joints the actuators drive, in ``robots/r1pro/robot.yaml`` order."""
    out = [f"torso_joint{i}" for i in range(1, 5)]
    for side in ("left", "right"):
        out += [f"{side}_arm_joint{i}" for i in range(1, 8)]
        out.append(f"{side}_gripper_finger_joint1")
    return out


def _build(urdf_xml: str) -> str:
    import mujoco  # noqa: PLC0415  # reason: optional sim-only dep

    xml, swerve = _SWERVE.subn(r'\1"fixed"', urdf_xml)
    if swerve != _SWERVE_JOINTS:  # else every later qpos shifts and the manifest map lies
        raise ValueError(f"expected {_SWERVE_JOINTS} swerve joints to fix, found {swerve}")
    compiler = (
        '<mujoco><compiler discardvisual="false" fusestatic="false" '
        'strippath="false" angle="radian"/></mujoco>'
    )
    spec = mujoco.MjSpec.from_string(re.sub(r"(<robot\b[^>]*>)", r"\1" + compiler, xml, count=1))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080

    for geom in list(spec.geoms):  # visual meshes import as contype/conaffinity 0
        body = geom.parent
        hull = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=geom.meshname)
        hull.pos, hull.quat = geom.pos, geom.quat
        hull.contype, hull.conaffinity, hull.group = 2, 1, 3
        hull.mass = 0.0  # the URDF <inertial> sets the body's mass

    base = spec.body("base_link")
    slide, hinge = mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE
    for name, jtype, axis in (
        ("base_x", slide, (1, 0, 0)),
        ("base_y", slide, (0, 1, 0)),
        ("base_yaw", hinge, (0, 0, 1)),
    ):
        base.add_joint(name=name, type=jtype, axis=axis)

    for joint_name in _actuated():
        kp, kv = next((kp, kv) for part, kp, kv in _GAINS if part in joint_name)
        joint = spec.joint(joint_name)
        joint.armature = 0.05
        act = spec.add_actuator(name=f"{joint_name}_ctrl", target=joint_name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.inheritrange = 1.0
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = joint.actfrcrange  # the URDF <limit effort>

    for side in ("left", "right"):
        spec.add_equality(
            name=f"{side}_gripper_finger_mimic",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=f"{side}_gripper_finger_joint2",
            name2=f"{side}_gripper_finger_joint1",
            data=[0, 1, 0, 0, 0] + [0] * 6,
            solref=[0.005, 1.0],
        )

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


def ensure_r1pro_mjcf() -> Path:
    """Return the cached R1 Pro MuJoCo twin MJCF, building it on first use.

    Raises:
        ROSConfigError: The URDF cannot be fetched (``ensure_r1pro_urdf``),
            ``mujoco`` is missing or rejects the model, the URDF's swerve layout
            changed, or the cache cannot be written.
    """
    urdf_xml = ensure_r1pro_urdf().read_text(encoding="utf-8")
    try:
        import mujoco  # noqa: PLC0415  # reason: optional sim-only dep
    except ImportError as exc:
        raise ROSConfigError(f"behavior:r1pro MJCF needs mujoco: {exc}") from exc
    # The writer's mujoco version is in the key: another venv's mujoco may not parse it.
    material = Path(__file__).read_bytes() + urdf_xml.encode() + mujoco.__version__.encode()
    out = _derived_path().parent / "mjcf" / hashlib.sha256(material).hexdigest()[:16] / "r1pro.xml"
    if out.is_file():
        return out
    try:
        xml = _build(urdf_xml)
    except ValueError as exc:  # mujoco's spec/compile errors, or a URDF layout change
        raise ROSConfigError(f"behavior:r1pro MJCF build failed: {exc}") from exc
    tmp: str | None = None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".r1pro.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(xml)
        os.replace(tmp, out)
    except OSError as exc:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)
        raise ROSConfigError(f"behavior:r1pro MJCF cache write failed: {exc}") from exc
    return out
