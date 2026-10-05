"""Robot-agnostic, URDF-driven Isaac Sim scene for the sidecar.

Runs under the Isaac Sim py3.11 venv only; imported by ``isaac_sidecar.py`` AFTER
``SimulationApp`` is live (every import here needs a running Kit app).

The only sidecar scene: it honours the forwarded ``--robot`` — it imports the manifest robot's URDF (``isaacsim.asset.importer.urdf``)
and wires joints/sensors/control from a plain-JSON "isaac robot spec" the
openral-side backend marshals across the venv boundary (the sidecar cannot
import ``openral_core``).

Control is by ABSOLUTE targets, the same meaning ``SimAttachedHAL`` sends: one
slot per non-base manifest joint in manifest order (arm in rad, grippers in the
manifest's units, mapped onto the URDF finger by ``manifest_to_urdf_gripper``;
NaN = hold), then the base twist (vx, vy, wz) for a planar base. The openral
side packs typed actions into it by name
(``openral_sim.backends.isaac_sim.pack_isaac_action``).

The world around the robot is either the built-in bring-up stage (ground plane,
plus a few obstacle boxes when the robot has a lidar) or an **environment USD**
(``environment_usd``, from the deploy scene's ``scene.assets_uri``) referenced
under ``/World/environment`` — a warehouse, an office, any stage that carries
its own floor collider and lights. The robot is placed at ``spawn_pose``
(x, y, z, yaw — from the deploy scene's ``base_pose``); a mobile base's
odometry is relative to that spawn, as on a real robot.

Robot-spec contract (built by ``openral_sim.backends.isaac_sim._build_robot_spec``)::

    {
      "robot_id": str,
      "urdf_path": str,                  # resolved absolute path to the URDF file
      "fix_base": bool,                  # pin the root (True for a fixed arm)
      "ros_package_paths": [{"name", "path"}],      # package:// roots for the meshes
      "joints": [{"name", "role", "joint_type", "urdf_name"}],  # manifest order
      "grippers": [{"name", "leader", "closed", "open", "manifest_closed",
                    "manifest_open", "followers": [{"dof", "multiplier", "offset"}]}],
      "base_joints": [str] | null,       # [forward, side, yaw] for a planar base
      "base_kinematics": str | null,
      "action": {"dim": int, "control_mode": "joint_position", "has_base": bool},
      "sensors": [{"name", "modality", "vla_feature_key", "frame_id",
                   "parent_frame", "intrinsics": {...}, "range_min_m",
                   "range_max_m", "n_channels"}],
      "camera_mounts": {sensor_name: {"link", "xyz", "quat_wxyz", "look_at", "axes"}},
    }                                  # optional; openral_sim...isaac_sim.IsaacCameraMount
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from _isaac_scene_base import IsaacSceneBase
from numpy.typing import NDArray

# Joint drive gains applied when the Isaac >= 6 URDF importer converts the robot
# (URDFs carry no drive gains; without them the arm sags under gravity within a
# few steps). Revolute: Nm/rad, Nm*s/rad; prismatic: N/m, N*s/m.
# ponytail: one gain for every joint; per-joint gains from the manifest if a
# heavy arm sags or a light one rings.
_DRIVE_STIFFNESS = 1000.0
_DRIVE_DAMPING = 100.0
# Coulomb friction of the robot's finger colliders. The URDF importer authors no physics
# material, so the fingers get PhysX's default (~0.5) and a lifted object slips out of a
# closed, stalled jaw (Isaac i58/i59: jaw squeezing at its 7 Nm effort cap, can sliding).
# Rubber-pad order of magnitude; combined with the object's material by max, so a default
# 0.5 prop is gripped at this value.
_FINGER_FRICTION = 1.0
# Every camera's clipping range (m). Isaac's default near plane is 1 stage unit,
# 1 m here: it blanked every surface closer than a metre, in RGB and in depth (an
# all-inf depth image, so no cloud at all at tabletop range).
_CAM_NEAR_M = 0.01
_CAM_FAR_M = 100.0
# Self-occlusion handling for the lidar fan, mirroring the MuJoCo sibling
# (`openral_sim.backends.robocasa._LASER_MAX_SELF_SKIPS` / `_LASER_SELF_SKIP_EPS_M`).
# A beam may terminate on the robot's own body (chassis / wheels / arm column);
# we re-cast past each such hit. This caps how many self-layers one beam steps
# through before giving up and reading `range_max_m`. 8 covers chassis + arm
# links with margin.
_SCAN_MAX_SELF_SKIPS = 8
# Nudge (m) added past a self-hit before re-casting, so the next ray does not
# re-detect the surface it just exited.
_SCAN_SELF_SKIP_EPS_M = 1e-3


def map_dof_to_manifest(
    values: NDArray[np.float32],
    *,
    dof_index: dict[str, int],
    manifest_joints: list[dict[str, Any]],
    grippers: list[dict[str, Any]],
    base_values: list[float] | None = None,
    base_joints: list[str] | None = None,
    rates: bool = False,
) -> NDArray[np.float32]:
    """Map an Isaac articulation DOF vector to the full manifest joint order.

    Per manifest joint: a base joint reads ``base_values`` (not a URDF DOF); an
    arm joint reads its URDF DOF (``urdf_name``); a gripper reads its leader
    finger DOF, mapped linearly from the URDF ``closed``/``open`` targets onto
    the manifest's own range (a normalised ``[0, 1]`` Panda width, the OpenArm
    jaw angle) so ``/joint_states`` stays inside the manifest limits; anything
    unresolved is ``0.0``. ``rates=True`` maps velocities (scale only, no offset).

    Returns:
        Vector in ``manifest_joints`` order, for ``SimAttachedHAL.read_state``
        to index against ``description.joints``.
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    base_idx = {name: i for i, name in enumerate(base_joints or [])}
    by_gripper = {str(g["name"]): g for g in grippers}
    bv = base_values or []
    out: list[float] = []
    for j in manifest_joints:
        name = str(j.get("name", ""))
        if name in base_idx and base_idx[name] < len(bv):
            out.append(float(bv[base_idx[name]]))
            continue
        idx = dof_index.get(str(j.get("urdf_name")))
        if idx is None or idx >= v.shape[0]:
            out.append(0.0)
            continue
        g = by_gripper.get(name)
        if g is None:
            out.append(float(v[idx]))
            continue
        span = float(g["open"]) - float(g["closed"])
        scale = (float(g["manifest_open"]) - float(g["manifest_closed"])) / span if span else 0.0
        if rates:
            out.append(float(v[idx]) * scale)
        else:
            out.append(float(g["manifest_closed"]) + (float(v[idx]) - float(g["closed"])) * scale)
    return np.asarray(out, dtype=np.float32)


def _yaw_quat(yaw: float) -> NDArray[np.float64]:
    """``(w, x, y, z)`` quaternion of a rotation by ``yaw`` about world z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def manifest_to_urdf_gripper(gripper: dict[str, Any], value: float) -> float:
    """A gripper target in manifest units → its URDF leader joint target.

    Linear between the gripper's manifest closed/open ends and its URDF
    closed/open targets (``_gripper_spec``), clamped to that travel — e.g.
    panda_mobile's normalised width 1.0 → 0.04 m, OpenArm's right jaw -0.785 →
    the URDF finger's -0.785.

    Example:
        >>> g = {"closed": 0.0, "open": 0.04, "manifest_closed": 0.0, "manifest_open": 1.0}
        >>> round(manifest_to_urdf_gripper(g, 0.5), 3), manifest_to_urdf_gripper(g, 2.0)
        (0.02, 0.04)
    """
    span = float(gripper["manifest_open"]) - float(gripper["manifest_closed"])
    frac = (float(value) - float(gripper["manifest_closed"])) / span if span else 0.0
    frac = min(max(frac, 0.0), 1.0)
    closed, opened = float(gripper["closed"]), float(gripper["open"])
    return closed + frac * (opened - closed)


@dataclass(frozen=True)
class FingerFrictionResult:
    """What :func:`apply_finger_friction` actually did, so the caller can log it honestly.

    Attributes:
        bound: Prim paths the pad material was bound to (finger links + instanced roots).
        missing_joints: Requested joint names that resolved to no USD joint with a ``body1``
            link (renamed or dropped by the importer) — their finger keeps PhysX's default.
        displaced: Descendant prims under a finger link that already carried their own
            ``physics``-purpose material binding; the pad overrides it.
        combine_max: ``True`` when the ``max`` friction combine was authored; ``False`` when
            the PhysX schema is unavailable and PhysX falls back to average combine.
    """

    bound: list[str]
    missing_joints: list[str]
    displaced: list[str]
    combine_max: bool


def apply_finger_friction(
    stage: Any,
    robot_prim: str,
    finger_joints: Iterable[str],
    friction: float = _FINGER_FRICTION,
) -> FingerFrictionResult:
    """Bind a high-friction physics material to the links the robot's finger joints drive.

    The URDF importer authors no physics material, so every collider gets PhysX's default
    friction and a gripped prop slips out when lifted. The finger links are found through
    the gripper joints (``grippers[].leader`` and ``followers[].dof`` in the robot spec),
    not by name: the OpenArm's finger colliders are ``openarm_*_ee_link1/2``, its
    ``finger_*`` prims are visuals. One shared material (static = dynamic = ``friction``,
    no restitution, ``max`` friction combine when the PhysX schema is available so a
    low-friction prop does not average the pad down) is bound for the ``physics`` purpose
    on each finger link and on every instanced root under it (the importer instances its
    collision geometry, and an instance proxy cannot take a binding).

    The binding is ``strongerThanDescendants``: the pad wins over any physics material
    already authored on a finger collider, so the friction the caller logs is the friction
    PhysX uses (``weakerThanDescendants`` would let a descendant's material silently win
    while the prim is still reported as bound). Descendants whose own binding is thereby
    replaced are returned in ``displaced`` for the caller to surface.

    Args:
        stage: The USD stage (``pxr.Usd.Stage``).
        robot_prim: The robot's root prim path.
        finger_joints: URDF names of the gripper joints (leaders and mimic followers).
        friction: Static and dynamic friction coefficient, ``> 0``.

    Returns:
        A :class:`FingerFrictionResult`. A partial match (``missing_joints`` non-empty)
        grips asymmetrically; no match at all leaves ``bound`` empty and creates no material.

    Raises:
        ValueError: On a non-positive or non-finite ``friction``.
    """
    import math

    if not (math.isfinite(friction) and friction > 0.0):
        raise ValueError(f"apply_finger_friction: friction {friction!r} must be finite and > 0")
    from pxr import Usd, UsdPhysics, UsdShade

    names = set(finger_joints)
    root = stage.GetPrimAtPath(robot_prim)
    links: set[str] = set()
    found: set[str] = set()
    for p in Usd.PrimRange(root):
        if p.GetName() in names and p.IsA(UsdPhysics.Joint):
            targets = UsdPhysics.Joint(p).GetBody1Rel().GetTargets()
            if targets:
                found.add(p.GetName())
                links.update(str(t) for t in targets)
    missing = sorted(names - found)
    pads: list[Any] = []
    displaced: list[str] = []
    for link in sorted(links):
        link_prim = stage.GetPrimAtPath(link)
        pads.append(link_prim)
        pads.extend(p for p in Usd.PrimRange(link_prim) if p.IsInstance())
        displaced.extend(
            str(p.GetPath())
            for p in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
            if UsdShade.MaterialBindingAPI(p).GetDirectBinding("physics").GetMaterialPath()
        )
    if not pads:
        return FingerFrictionResult([], missing, [], combine_max=False)
    material = UsdShade.Material.Define(stage, "/World/Physics_Materials/finger_pad")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.0)
    try:
        from pxr import PhysxSchema
    except ImportError:
        combine_max = False  # reported, not hidden: PhysX averages the pad with the prop
    else:
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("max")
        combine_max = True
    for pad in pads:
        UsdShade.MaterialBindingAPI.Apply(pad).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics"
        )
    return FingerFrictionResult(
        [str(p.GetPath()) for p in pads], missing, sorted(displaced), combine_max
    )


def strip_urdf_mimics(urdf_text: str, source_dir: str) -> str:
    """A URDF without ``<mimic>`` tags, its relative mesh paths made absolute.

    Isaac 6's converter turns ``<mimic>`` into a Newton mimic constraint that
    destabilises PhysX articulations — OpenArm's mirrored (x -1) jaw drags its
    wrist into spinning even with gravity off. The scene drives every mimic
    follower itself (``grippers[].followers``), so the engine never needs the
    constraint. The copy lives elsewhere, hence the absolute mesh paths
    (``package://`` and absolute refs are left alone).

    Example:
        >>> text = (
        ...     '<robot name="r"><joint name="b" type="revolute">'
        ...     '<mimic joint="a" multiplier="-1"/></joint>'
        ...     '<link name="l"><visual><geometry><mesh filename="m/x.stl"/>'
        ...     "</geometry></visual></link></robot>"
        ... )
        >>> out = strip_urdf_mimics(text, "/robots/r")
        >>> "mimic" in out, 'filename="/robots/r/m/x.stl"' in out
        (False, True)
    """
    import os
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_text)
    for joint in root.iter("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
    for mesh in root.iter("mesh"):
        ref = mesh.attrib.get("filename", "")
        if ref and not ref.startswith(("package://", "file://", "/")):
            mesh.attrib["filename"] = os.path.normpath(os.path.join(source_dir, ref))
    return ET.tostring(root, encoding="unicode")


def compose_planar(
    spawn: tuple[float, float, float, float], odom: list[float]
) -> tuple[float, float, float, float]:
    """World ``(x, y, z, yaw)`` of the base: the spawn pose composed with the odom pose.

    ``spawn`` is the world placement ``(x, y, z, yaw)``; ``odom`` the planar
    ``(x, y, yaw)`` the kinematic base has integrated since that spawn (the
    odometry frame starts at the spawn, like a real base's wheel odometry).

    Example:
        >>> import math
        >>> x, y, z, yaw = compose_planar((1.0, 2.0, 0.5, math.pi / 2), [1.0, 0.0, 0.0])
        >>> round(x, 6), round(y, 6), z, round(yaw, 6)
        (1.0, 3.0, 0.5, 1.570796)
    """
    sx, sy, sz, syaw = spawn
    ox, oy, oyaw = (float(v) for v in odom)
    c, s = float(np.cos(syaw)), float(np.sin(syaw))
    return (sx + c * ox - s * oy, sy + s * ox + c * oy, sz, syaw + oyaw)


def look_at_matrix(eye: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation whose columns are a camera's x (forward), y (left), z (up) axes aimed at ``target``.

    Isaac's ``world`` camera-axes convention (x forward, z up). Looking straight
    down or up keeps y on the frame's +y.

    Example:
        >>> look_at_matrix(np.zeros(3), np.array([0.0, 0.0, -1.0]))[:, 0].tolist()
        [0.0, 0.0, -1.0]
    """
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(left) < 1e-9 else left / np.linalg.norm(left)
    return np.column_stack([fwd, left, np.cross(fwd, left)])


def camera_hfov_deg(meta: dict[str, Any]) -> float | None:
    """A planned camera's horizontal FOV: its mount's ``hfov_deg``, else the manifest intrinsics.

    ``None`` when neither is given (Isaac's default lens stays).

    Example:
        >>> camera_hfov_deg({"intrinsics": {"width": 640, "fx": 320.0}})
        90.0
        >>> camera_hfov_deg(
        ...     {"mount": {"hfov_deg": 70.0}, "intrinsics": {"width": 640, "fx": 320.0}}
        ... )
        70.0
    """
    mount = meta.get("mount") or {}
    if mount.get("hfov_deg"):
        return float(mount["hfov_deg"])
    k = meta.get("intrinsics") or {}
    if k.get("width") and k.get("fx"):
        return float(np.degrees(2.0 * np.arctan(float(k["width"]) / (2.0 * float(k["fx"])))))
    return None


def points_in_frame(
    points: NDArray[np.float32], world_from_frame: NDArray[np.float64]
) -> NDArray[np.float32]:
    """World points ``(N, 3)`` in a frame given by its 4x4 world pose; non-finite rows dropped.

    A depth pixel that hits nothing deprojects to inf/NaN; those rows are not points.

    Example:
        >>> pose = np.eye(4)
        >>> pose[:3, 3] = (1.0, 0.0, 0.5)
        >>> points_in_frame(
        ...     np.array([[1.0, 0.0, 0.0], [np.inf, 0, 0]], dtype=np.float32), pose
        ... ).tolist()
        [[0.0, 0.0, -0.5]]
    """
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = raw[np.isfinite(raw).all(axis=1)]
    rot, trans = world_from_frame[:3, :3], world_from_frame[:3, 3]
    return ((pts - trans) @ rot).astype(np.float32)


def resolve_beam_range(
    raycast_closest: Any,
    *,
    robot_prim: str,
    origin_xy: tuple[float, float],
    angle_rad: float,
    z: float,
    range_min_m: float,
    range_max_m: float,
) -> float:
    """One lidar beam's range, with the robot's own body skipped by IDENTITY.

    Pure: ``raycast_closest(start, dir, span)`` is a PhysX scene-query callable,
    so the walk is testable without a Kit app. Returns Isaac's hit dict
    (``hit``/``distance``/``rigidBody``) or a falsy value.

    The beam starts at ``range_min_m`` (the sensor's own minimum) and walks
    outward; a hit whose ``rigidBody`` prim is under ``robot_prim`` (the
    imported articulation's root prim) is the robot itself, so the beam
    re-casts from just past it — same resolution as
    ``openral_sim.backends.robocasa.synthesize_laser_scan_2d`` (which compares
    ``model.body_rootid`` instead of a prim path). #194: ``range_min_m`` was
    lowered from chassis-sized (panda_mobile: 0.55 m vs 0.43 m circumscribed
    radius) to the sensor minimum, so self-hits must be skipped by identity
    rather than assumed clear.

    Returns ``range_max_m`` for a miss, an out-of-range hit, or a beam still on
    the robot after ``_SCAN_MAX_SELF_SKIPS`` layers.

    Example:
        >>> def cast(start, _dir, span):  # a wall 3 m out, nothing else
        ...     d = 3.0 - start[0]
        ...     return (
        ...         {"hit": True, "distance": d, "rigidBody": "/wall"} if 0 <= d <= span else None
        ...     )
        >>> round(
        ...     resolve_beam_range(
        ...         cast,
        ...         robot_prim="/panda",
        ...         origin_xy=(0.0, 0.0),
        ...         angle_rad=0.0,
        ...         z=0.3,
        ...         range_min_m=0.05,
        ...         range_max_m=12.0,
        ...     ),
        ...     3,
        ... )
        3.0
    """
    cx, cy = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    travelled = float(range_min_m)
    own_prefix = robot_prim.rstrip("/") + "/"
    for _ in range(_SCAN_MAX_SELF_SKIPS):
        span = float(range_max_m) - travelled
        if span <= 0.0:
            break
        hit = raycast_closest(
            (ox + travelled * cx, oy + travelled * cy, float(z)), (cx, cy, 0.0), span
        )
        if not (isinstance(hit, dict) and hit.get("hit")):
            break  # nothing out there
        distance = float(hit.get("distance", span))
        body = str(hit.get("rigidBody", ""))
        if body == robot_prim or body.startswith(own_prefix):
            travelled += distance + _SCAN_SELF_SKIP_EPS_M
            continue
        return min(travelled + distance, float(range_max_m))
    return float(range_max_m)


class IsaacManifestScene(IsaacSceneBase):
    """A URDF-imported, manifest-driven Isaac Sim scene."""

    warmup_steps = 4
    physics_substeps = 1

    def __init__(
        self,
        *,
        robot_spec: dict[str, Any],
        environment_usd: str | None = None,
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        objects: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._spec = robot_spec
        self._environment_usd = environment_usd
        self._spawn = spawn_pose  # world x, y, z, yaw
        # Actuated manifest joints (the planar base is handled separately in M3;
        # it has no URDF DOF). Order is the manifest order.
        self._manifest_joints: list[dict[str, Any]] = list(robot_spec.get("joints", []))
        self._arm_names = [
            str(j["urdf_name"]) for j in self._manifest_joints if j.get("role") == "arm"
        ]
        # One entry per manifest gripper: URDF leader DOF, open/closed targets,
        # mimic followers (see openral_sim.backends.isaac_sim._gripper_spec).
        self._grippers: list[dict[str, Any]] = list(robot_spec.get("grippers") or [])
        self._base_joints: list[str] = list(robot_spec.get("base_joints") or [])
        action = robot_spec.get("action", {}) or {}
        self._has_base = bool(action.get("has_base", False))
        # Action layout (openral_sim.backends.isaac_sim.IsaacActionLayout): one
        # ABSOLUTE target per non-base manifest joint in manifest order (arm in
        # rad, gripper in manifest units; NaN = hold), then the base twist.
        by_gripper = {str(g["name"]): g for g in self._grippers}
        self._slot_plan: list[tuple[str, Any]] = [
            ("gripper", by_gripper[str(j["name"])])
            if j.get("role") == "gripper"
            else ("arm", str(j["urdf_name"]))
            for j in self._manifest_joints
            if j.get("role") in ("arm", "gripper")
        ]
        self.action_dim = len(self._slot_plan) + (3 if self._has_base else 0)
        self._objects: list[dict[str, Any]] = list(objects or [])
        self._object_prims: dict[str, Any] = {}

        # Kinematic planar base: the arm imports fix_base=True (pinned) and the
        # whole articulation root is teleported each step from an integrated
        # (x, y, yaw) pose driven by the action's last 3 base-twist channels
        # (vx, vy, wyaw, base frame) — no PhysX base joints. Integrated by the
        # BODY_TWIST command interval (one env.step = one /cmd_vel command), not
        # the physics dt, matching SimAttachedHAL's body_twist_dt_s.
        self._base_pose = [0.0, 0.0, 0.0]  # odom x, y, yaw (relative to the spawn)
        self._mount_z = float(robot_spec.get("mount_z", 0.0))
        # base_frame link pose in the root frame, resolved at build (see _depth_clouds).
        self._root_to_base: NDArray[np.float64] = np.eye(4)
        self._base_dt = float(action.get("body_twist_dt_s", 0.05))

        self._robot: Any = None
        self._robot_prim = ""
        # Isaac >= 6: the importer pins the base with a fixed joint whose body0 is
        # not a rigid body — its localPos0/localRot0 are a WORLD-frame anchor, so
        # placing (or kinematically driving) the robot moves that anchor rather
        # than fighting it. None on the < 6 command importer.
        self._anchor_joint: Any = None
        self._ArticulationAction: Any = None
        self._euler_to_quat: Any = None
        # One Isaac Camera per manifest RGB/depth sensor, base-relative so they
        # ride the kinematic base. `_cam_meta` is the ordered plan; `_cameras`
        # holds the live Camera objects keyed by sensor name (built in `build`).
        self._cam_meta: list[dict[str, Any]] = self._plan_cameras()
        self._cameras: dict[str, Any] = {}
        # 2-D lidar: a PhysX raycast fan from the base produces real /scan ranges
        # against the scene's obstacles (built in `build` when the manifest
        # declares a lidar_2d sensor). `_scan_z` is the beam height above the floor
        # the robot stands on (the spawn z).
        self._lidar: dict[str, Any] | None = next(
            (s for s in robot_spec.get("sensors", []) if s.get("modality") == "lidar_2d"), None
        )
        self._scan_query: Any = None
        self._scan_z = 0.30
        # DOF mapping, resolved post-reset once dof_names is populated.
        self._dof_index: dict[str, int] = {}
        self._arm_dof_idx: list[int] = []
        # Commanded joint targets (full DOF vector). A NaN action slot keeps its
        # entry here, so holding never re-targets wherever gravity has dragged
        # the arm. Re-seeded from the articulation at every reset.
        self._target: NDArray[np.float32] | None = None

    def _plan_cameras(self) -> list[dict[str, Any]]:
        """Plan one base-relative camera per manifest RGB/depth sensor.

        Each entry: ``{name, key, modality, offset_pos (base frame xyz), offset_euler
        (deg)}``. RGB sensors are keyed by their ``vla_feature_key`` suffix
        (``camera1``…) so ``SimSensorBridge`` finds them; depth sensors are keyed by
        the sensor ``name`` (the bridge's depth obs key). Offsets give each camera a
        distinct, workspace-facing viewpoint (the manifest carries no Isaac-frame
        extrinsics) — robot-mounted, so they translate/rotate with the base.
        """
        # base frame: x forward, y left, z up. (pos, euler_deg=[roll,pitch,yaw]).
        rgb_offsets = [
            ((1.5, 0.0, 1.0), (0.0, 35.0, 180.0)),  # front, look back + down
            ((1.3, -0.7, 1.1), (0.0, 38.0, 205.0)),  # front-right
            ((0.5, 0.0, 1.7), (0.0, 75.0, 180.0)),  # top-down over the workspace
        ]
        # Depth faces FORWARD (+x) and down — obstacle/ground sensing ahead of the
        # base, where a mobile manipulator drives (yaw 0, unlike the back-facing
        # manipulation agentviews above).
        depth_offset = ((0.30, 0.0, 0.75), (0.0, 20.0, 0.0))
        plan: list[dict[str, Any]] = []
        mounts: dict[str, Any] = self._spec.get("camera_mounts") or {}
        rgb_i = 0
        for s in self._spec.get("sensors", []):
            modality = s.get("modality")
            if modality == "rgb":
                pos, euler = rgb_offsets[min(rgb_i, len(rgb_offsets) - 1)]
                rgb_i += 1
                key = (
                    str(s["vla_feature_key"]).rsplit(".", 1)[-1]
                    if s.get("vla_feature_key")
                    else s.get("name", "camera1")  # sensor name (canonical form)
                )
                plan.append(
                    {
                        "name": s["name"],
                        "key": key,
                        "modality": "rgb",
                        "offset_pos": pos,
                        "offset_euler": euler,
                        "mount": mounts.get(s["name"]),
                        "intrinsics": s.get("intrinsics"),
                    }
                )
            elif modality == "depth":
                pos, euler = depth_offset
                plan.append(
                    {
                        "name": s["name"],
                        "key": s["name"],
                        "modality": "depth",
                        "offset_pos": pos,
                        "offset_euler": euler,
                        "range_max_m": s.get("range_max_m"),
                        "mount": mounts.get(s["name"]),
                        "intrinsics": s.get("intrinsics"),
                    }
                )
        return plan

    # ── build ────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Import the manifest robot's URDF and stand up cameras + the world."""
        import isaacsim.core.utils.numpy.rotations as rot_utils
        import omni.kit.commands
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera

        self._ArticulationAction = ArticulationAction

        # Isaac >= 6 converts the URDF to a USD file first — its importer opens
        # its own stage, so this must run before World() owns a fresh one.
        robot_usd = self._convert_urdf()
        if robot_usd is not None:
            from isaacsim.core.utils.stage import create_new_stage

            create_new_stage()

        # NOTE: no device="cuda:0" — forcing GPU PhysX hangs the first warmup for
        # minutes on an 8 GB laptop GPU (per the PoC notes). Default device
        # renders the same scene in ~15 s.
        self._world = World(stage_units_in_meters=1.0)
        if self._environment_usd:
            self._add_environment(self._environment_usd)
        else:
            self._world.scene.add_default_ground_plane()

        self._euler_to_quat = rot_utils.euler_angles_to_quats

        prim_path = (
            self._reference_robot(robot_usd)
            if robot_usd is not None
            else self._import_urdf(omni.kit.commands)
        )
        self._robot_prim = prim_path
        from isaacsim.core.utils.stage import get_current_stage

        finger_joints = [
            name
            for g in self._spec.get("grippers") or []
            for name in (g["leader"], *(f["dof"] for f in g["followers"]))
        ]
        result = apply_finger_friction(get_current_stage(), prim_path, finger_joints)
        print(
            f"[isaac_manifest_scene] finger friction {_FINGER_FRICTION} bound on "
            f"{len(result.bound)} prim(s) for joints {finger_joints}, combine="
            + (
                "max"
                if result.combine_max
                else "n/a (no pad)"
                if not result.bound
                else "average (PhysxSchema unavailable)"
            ),
            flush=True,
        )
        if not result.bound:
            print(
                "[isaac_manifest_scene] WARNING: NO finger link found, the fingers keep "
                "PhysX's default friction",
                flush=True,
            )
        elif result.missing_joints:
            print(
                f"[isaac_manifest_scene] WARNING: finger joints {result.missing_joints} not "
                "found, those fingers keep PhysX's default friction (asymmetric grip)",
                flush=True,
            )
        if result.bound and not result.combine_max:
            print(
                "[isaac_manifest_scene] WARNING: PhysxSchema unavailable, friction combine is "
                "average (a low-friction prop drags the pad below its nominal value)",
                flush=True,
            )
        if result.displaced:
            print(
                "[isaac_manifest_scene] WARNING: finger colliders already carried a physics "
                f"material, replaced by the pad: {result.displaced}",
                flush=True,
            )
        wx, wy, wz, wyaw = self._world_pose()
        self._set_anchor(wx, wy, wz, wyaw)
        self._robot = self._world.scene.add(
            Robot(
                prim_path=prim_path,
                name="robot",
                position=np.array([wx, wy, wz]),
                orientation=_yaw_quat(wyaw),
            )
        )

        # On the bare bring-up stage a lidar robot gets a few static obstacles so
        # /scan, slam, and Nav2 have real geometry to map + avoid (a bare ground
        # plane returns no hits). An environment USD brings its own geometry.
        if self._lidar is not None and not self._environment_usd:
            self._add_obstacles()
        self._add_objects()

        # One Camera per planned sensor. A bare deploy scene with no RGB sensor
        # still gets a default camera1 so the obs always carries a frame.
        if not any(m["modality"] == "rgb" for m in self._cam_meta):
            self._cam_meta.insert(
                0,
                {
                    "name": "camera1",
                    "key": "camera1",
                    "modality": "rgb",
                    "offset_pos": (1.5, 0.0, 1.0),
                    "offset_euler": (0.0, 35.0, 180.0),
                },
            )
        for meta in self._cam_meta:
            parent = self._mount_parent(meta["mount"])[0] if meta.get("mount") else "/World"
            cam = Camera(
                prim_path=f"{parent}/cam_{meta['name']}",
                resolution=(self.obs_width, self.obs_height),
            )
            self._cameras[meta["name"]] = cam

        # Pose of the manifest base_frame link in the robot root's frame: depth
        # clouds are published in base_frame, which need not be the URDF root
        # (OpenArm's openarm_base sits 0.698 m above it).
        self._root_to_base = self._link_offset(self._spec.get("base_frame"))
        self._world.reset()
        for meta in self._cam_meta:
            cam = self._cameras[meta["name"]]
            cam.initialize()
            cam.set_clipping_range(_CAM_NEAR_M, _CAM_FAR_M)
            hfov = camera_hfov_deg(meta)
            if hfov is not None:
                aperture = 2.0 * cam.get_focal_length() * float(np.tan(np.radians(hfov) / 2.0))
                cam.set_horizontal_aperture(aperture)
                cam.set_vertical_aperture(aperture * self.obs_height / self.obs_width)
            if meta.get("mount"):
                self._mount_camera(cam, meta["mount"])
            if meta["modality"] == "depth":
                # distance_to_image_plane → the depth array we deproject to a cloud.
                cam.add_distance_to_image_plane_to_frame()
        self._place_robot()
        self._resolve_dof_mapping()
        if self._lidar is not None:
            from omni.physx import get_physx_scene_query_interface

            self._scan_query = get_physx_scene_query_interface()

    def _add_environment(self, usd: str) -> None:
        """Reference an environment USD under ``/World/environment``.

        ``usd`` is a local path, an ``http(s)://`` / ``omniverse://`` URL, or
        ``isaac:<path>`` — a path under the installed Isaac Sim's own asset root
        (``get_assets_root_path()``, e.g.
        ``isaac:Isaac/Environments/Simple_Warehouse/warehouse.usd``), so the
        asset version always matches the sidecar's Isaac release. Isaac's own
        assets are licensed for use inside Isaac Sim, which is exactly where they
        are loaded here — never converted or vendored.

        The environment replaces the default ground plane: it must carry its own
        floor collider (Isaac's environments do).
        """
        from isaacsim.core.utils.stage import add_reference_to_stage

        usd = self._resolve_usd(usd)
        print(f"[isaac_manifest_scene] environment: {usd}", flush=True)
        add_reference_to_stage(usd_path=usd, prim_path="/World/environment")

    @staticmethod
    def _resolve_usd(usd: str) -> str:
        """Expand an ``isaac:<path>`` ref against the installed Isaac asset root."""
        if not usd.startswith("isaac:"):
            return usd
        try:
            from isaacsim.storage.native import get_assets_root_path
        except ImportError:  # Isaac Sim < 5.0
            from isaacsim.core.utils.nucleus import get_assets_root_path
        root = get_assets_root_path()
        if not root:
            raise RuntimeError(
                f"{usd!r}: Isaac Sim reports no asset root (get_assets_root_path() is "
                "empty) — offline host without a local asset pack? Use a local path or "
                "URL instead."
            )
        return f"{str(root).rstrip('/')}/{usd[len('isaac:') :].lstrip('/')}"

    def _add_objects(self) -> None:
        """Place the scene's extra USD objects; dynamic ones are graspable rigid bodies.

        Each object is ``{"usd", "name", "xyz", "yaw", "dynamic"}`` (validated
        openral-side). A dynamic object without physics gets a rigid body on its
        root and convex-hull colliders on its meshes; one that already carries a
        rigid body (Isaac's ``Axis_Aligned_Physics`` YCB props) is used as is.
        Registered with the World scene, so ``world.reset()`` puts every object
        back at its declared pose.
        """
        if not self._objects:
            return
        from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = get_current_stage()
        for obj in self._objects:
            path = f"/World/objects/{obj['name']}"
            add_reference_to_stage(usd_path=self._resolve_usd(str(obj["usd"])), prim_path=path)
            yaw = float(obj.get("yaw", 0.0))
            quat = _yaw_quat(yaw)
            pos = np.asarray(obj["xyz"], dtype=np.float64)
            name = str(obj["name"])
            prim: Any
            if not obj.get("dynamic", True):
                prim = SingleXFormPrim(prim_path=path, name=name, position=pos, orientation=quat)
            else:
                subtree = list(Usd.PrimRange(stage.GetPrimAtPath(path)))
                body = next((p for p in subtree if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
                if body is None:
                    UsdPhysics.RigidBodyAPI.Apply(subtree[0])
                if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in subtree):
                    for mesh in (p for p in subtree if p.IsA(UsdGeom.Mesh)):
                        UsdPhysics.CollisionAPI.Apply(mesh)
                        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(
                            "convexHull"
                        )
                if body is not None and body != subtree[0]:
                    # The rigid body is a child prim: place the referencing root
                    # and let the body ride it.
                    SingleXFormPrim(prim_path=path, position=pos, orientation=quat)
                    prim = SingleRigidPrim(prim_path=str(body.GetPath()), name=name)
                else:
                    prim = SingleRigidPrim(
                        prim_path=path, name=name, position=pos, orientation=quat
                    )
            self._object_prims[name] = self._world.scene.add(prim)
            print(f"[isaac_manifest_scene] object {obj['name']} at {pos.tolist()}", flush=True)

    def _add_obstacles(self) -> None:
        """Add a few static boxes around the robot for lidar/slam/Nav2 to see."""
        from isaacsim.core.api.objects import FixedCuboid

        # (center xyz, half-ish scale) — a partial enclosure + scattered boxes.
        boxes = [
            ((3.0, 0.0, 0.5), (0.3, 4.0, 1.0)),  # wall ahead
            ((-3.0, 0.0, 0.5), (0.3, 4.0, 1.0)),  # wall behind
            ((1.6, 1.8, 0.5), (0.6, 0.6, 1.0)),  # box front-left
            ((-1.5, -2.0, 0.5), (0.8, 0.8, 1.0)),  # box back-right
        ]
        for i, (pos, scale) in enumerate(boxes):
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array(pos),
                    scale=np.array(scale),
                    color=np.array([0.4, 0.4, 0.45]),
                )
            )

    def _scan_ranges(self) -> NDArray[np.float32] | None:
        """A 2-D LaserScan range fan via PhysX raycasts from the base.

        ``n_channels`` beams over ``[-π, π]`` (bridge convention) in the
        base_link frame, rotated to world by base yaw; each ray starts at
        ``range_min_m``. Self-hits (the robot's own body) are skipped by
        identity per beam — see ``resolve_beam_range`` for the resolution
        and the #194 rationale. ``None`` when the manifest declares no lidar.
        """
        if self._scan_query is None or self._lidar is None:
            return None
        n = int(self._lidar.get("n_channels") or 360)
        rmin = float(self._lidar.get("range_min_m") or 0.0)
        rmax = float(self._lidar.get("range_max_m") or 12.0)
        bx, by, bz, byaw = compose_planar(self._spawn, self._base_pose)
        z = bz + float(self._scan_z)
        out = np.empty(n, dtype=np.float32)
        step = 2.0 * np.pi / n
        for i in range(n):
            ang = -np.pi + i * step + byaw  # base beam angle, rotated to world
            out[i] = resolve_beam_range(
                self._scan_query.raycast_closest,
                robot_prim=self._robot_prim,
                origin_xy=(bx, by),
                angle_rad=ang,
                z=z,
                range_min_m=rmin,
                range_max_m=rmax,
            )
        return out

    def _link_prim_path(self, link: str | None) -> str:
        """The imported robot's prim for URDF ``link`` (``None`` / ``""``: the robot root)."""
        if not link:
            return self._robot_prim
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd

        root = get_current_stage().GetPrimAtPath(self._robot_prim)
        for prim in Usd.PrimRange(root):
            if prim.GetName() == link:
                return str(prim.GetPath())
        raise ValueError(f"camera mount / base_frame link {link!r} is not in the imported robot")

    def _link_offset(self, link: str | None) -> NDArray[np.float64]:
        """4x4 pose of ``link`` in the robot root frame (identity for the root or no link)."""
        if not link:
            return np.eye(4)
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom

        stage = get_current_stage()
        try:
            path = self._link_prim_path(link)
        except ValueError:
            if link != self._spec.get("base_frame"):
                raise
            # base_frame is no URDF link (OpenArm's openarm_base): the manifest's
            # base_to_root transform, marshalled openral-side.
            return np.asarray(self._spec.get("root_to_base", np.eye(4)), dtype=np.float64)
        rel, _reset = UsdGeom.XformCache().ComputeRelativeTransform(
            stage.GetPrimAtPath(path), stage.GetPrimAtPath(self._robot_prim)
        )
        return np.asarray(rel, dtype=np.float64).T  # Gf is row-vector: transpose to column

    def _mount_parent(self, mount: dict[str, Any]) -> tuple[str, NDArray[np.float64]]:
        """``(parent prim, parent-from-link pose)`` for a camera mount.

        The mount's own link, or for a ``base_frame`` that is no URDF link the
        robot root plus the base offset.
        """
        link = mount.get("link")
        try:
            return self._link_prim_path(link), np.eye(4)
        except ValueError:
            if link != self._spec.get("base_frame"):
                raise
            return self._robot_prim, self._link_offset(link)

    def _mount_camera(self, cam: Any, mount: dict[str, Any]) -> None:
        """Set a link-mounted camera's local pose from its scene mount."""
        from isaacsim.core.utils.numpy.rotations import (
            quats_to_rot_matrices,
            rot_matrices_to_quats,
        )

        xyz = np.asarray(mount["xyz"], dtype=np.float64)
        if mount.get("look_at") is not None:
            rot = look_at_matrix(xyz, np.asarray(mount["look_at"], dtype=np.float64))
        else:
            rot = quats_to_rot_matrices(np.asarray(mount["quat_wxyz"], dtype=np.float64)[None])[0]
        _parent, offset = self._mount_parent(mount)
        cam.set_local_pose(
            translation=offset[:3, :3] @ xyz + offset[:3, 3],
            orientation=rot_matrices_to_quats((offset[:3, :3] @ rot)[None])[0],
            camera_axes=mount.get("axes", "world"),
        )

    def _update_camera_poses(self) -> None:
        """Place every camera at its base-relative offset for the current base pose.

        The cameras are robot-mounted: as the kinematic base translates/rotates the
        viewpoints follow. base frame is planar (x, y, yaw), so the offset position
        rotates by yaw about z and the camera's euler yaw adds the base yaw.
        """
        bx, by, bz, byaw = compose_planar(self._spawn, self._base_pose)
        cos_y, sin_y = float(np.cos(byaw)), float(np.sin(byaw))
        byaw_deg = float(np.degrees(byaw))
        for meta in self._cam_meta:
            if meta.get("mount"):
                continue  # rides its link
            ox, oy, oz = meta["offset_pos"]
            wx = bx + cos_y * ox - sin_y * oy
            wy = by + sin_y * ox + cos_y * oy
            roll, pitch, yaw = meta["offset_euler"]
            quat = self._euler_to_quat(np.array([roll, pitch, yaw + byaw_deg]), degrees=True)
            self._cameras[meta["name"]].set_world_pose(np.array([wx, wy, bz + oz]), quat)

    def _convert_urdf(self) -> str | None:
        """Isaac >= 6: convert the URDF to a USD file; ``None`` on older Isaac.

        The 6.x importer (``URDFImporter``, built on ``urdf-usd-converter``)
        replaced the ``URDFParseAndImportFile`` command. It resolves
        ``package://`` meshes from the spec's ``ros_package_paths`` and authors
        position drives with explicit gains (``_DRIVE_STIFFNESS``). It converts
        a mimic-free copy (``strip_urdf_mimics``).
        """
        try:
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        except ImportError:
            return None
        import os
        import tempfile

        source = str(self._spec["urdf_path"])
        work = tempfile.mkdtemp(prefix="openral_isaac_urdf_")
        urdf = os.path.join(work, os.path.basename(source))
        with open(source, encoding="utf-8") as src, open(urdf, "w", encoding="utf-8") as dst:
            dst.write(strip_urdf_mimics(src.read(), os.path.dirname(source)))
        config = URDFImporterConfig(
            urdf_path=urdf,
            usd_path=os.path.join(work, "usd"),
            # Keep the kinematic tree faithful to the manifest: DOFs map by URDF
            # joint name.
            merge_fixed_joints=False,
            fix_base=bool(self._spec.get("fix_base", True)),
            joint_target_type="position",
            override_joint_stiffness=_DRIVE_STIFFNESS,
            override_joint_damping=_DRIVE_DAMPING,
            ros_package_paths=list(self._spec.get("ros_package_paths") or []),
        )
        return str(URDFImporter(config).import_urdf())

    def _reference_robot(self, robot_usd: str) -> str:
        """Reference a converted robot USD into the stage; return its prim path.

        The converter's asset keeps its physics behind a ``Physics`` variant set
        (``none`` / ``physics`` / ``physx`` / ``mujoco``) with no default — PhysX
        is selected, else the robot has no articulation.
        """
        import re

        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Usd, UsdPhysics

        prim_path = "/" + re.sub(r"\W", "_", str(self._spec.get("robot_id", "robot")))
        add_reference_to_stage(usd_path=robot_usd, prim_path=prim_path)
        stage = get_current_stage()
        variants = stage.GetPrimAtPath(prim_path).GetVariantSets()
        if "Physics" in variants.GetNames():
            variants.GetVariantSet("Physics").SetVariantSelection("physx")
        self._anchor_joint = next(
            (
                UsdPhysics.FixedJoint(p)
                for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path))
                if p.IsA(UsdPhysics.FixedJoint)
                and [str(t) for t in UsdPhysics.FixedJoint(p).GetBody0Rel().GetTargets()]
                in ([], [prim_path])
            ),
            None,
        )
        return prim_path

    def _set_anchor(self, x: float, y: float, z: float, yaw: float) -> None:
        """Move the world-frame base anchor (Isaac >= 6) to a world pose."""
        if self._anchor_joint is None:
            return
        from pxr import Gf

        self._anchor_joint.GetLocalPos0Attr().Set(Gf.Vec3f(x, y, z))
        self._anchor_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
        )

    def _import_urdf(self, commands: Any) -> str:
        """Isaac < 6: run the command-based URDF importer; return the prim path."""
        urdf_path = str(self._spec["urdf_path"])
        _status, import_config = commands.execute("URDFCreateImportConfig")
        # Keep the kinematic tree faithful to the manifest: do NOT merge fixed
        # joints (we map DOFs by URDF joint name), import as position-drive.
        import_config.merge_fixed_joints = False
        import_config.fix_base = bool(self._spec.get("fix_base", True))
        import_config.make_default_prim = False
        # World() already owns the PhysX scene + ground plane.
        import_config.create_physics_scene = False
        import_config.import_inertia_tensor = True
        import_config.distance_scale = 1.0
        result = commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )
        # The command returns (success, prim_path); tolerate either a 2-tuple or a
        # bare path across Isaac point releases.
        prim_path = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else result
        if not prim_path:
            raise RuntimeError(f"URDF import returned no prim path for {urdf_path!r}")
        return str(prim_path)

    def _resolve_dof_mapping(self) -> None:
        """Index the imported articulation's DOFs by name (post-reset)."""
        dof_names = [str(n) for n in (self._robot.dof_names or [])]
        self._dof_index = {n: i for i, n in enumerate(dof_names)}
        missing = [n for n in self._arm_names if n not in self._dof_index]
        missing += [
            name
            for g in self._grippers
            for name in [g["leader"], *(f["dof"] for f in g["followers"])]
            if name not in self._dof_index
        ]
        if missing:
            raise RuntimeError(f"URDF joints {missing} are not DOFs of the imported robot.")
        self._arm_dof_idx = [self._dof_index[n] for n in self._arm_names]

    # ── IsaacSceneBase template methods ──────────────────────────────────────

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        """Set absolute joint targets (NaN slots hold) and advance the base twist."""
        if self._target is None:
            self._target = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
        target = self._target
        for k, (kind, ref) in enumerate(self._slot_plan):
            value = float(action[k]) if k < action.shape[0] else float("nan")
            if np.isnan(value):
                continue  # HOLD — 0.0 is a legal target, so only NaN means "untouched"
            if kind == "arm":
                target[self._dof_index[ref]] = value
                continue
            lead = manifest_to_urdf_gripper(ref, value)
            target[self._dof_index[ref["leader"]]] = lead
            # Mimic followers (the second finger) — the URDF <mimic> relation.
            for f in ref["followers"]:
                target[self._dof_index[f["dof"]]] = float(f["multiplier"]) * lead + float(
                    f["offset"]
                )
        self._robot.get_articulation_controller().apply_action(
            self._ArticulationAction(joint_positions=target)
        )
        if self._has_base:
            base_i = len(self._slot_plan)
            if action.shape[0] >= base_i + 3:
                vx, vy, wz = (float(action[base_i + i]) for i in range(3))
                self._integrate_base(*(0.0 if np.isnan(v) else v for v in (vx, vy, wz)))

    def _integrate_base(self, vx: float, vy: float, wyaw: float) -> None:
        """Advance the kinematic base by a base-frame twist and teleport the root.

        ``(vx, vy)`` are body-frame linear velocities (m/s), ``wyaw`` the yaw rate
        (rad/s). Integrated to an odom ``(x, y, yaw)`` and applied (composed with
        the spawn) as the articulation's root world pose — the arm rides along,
        giving real base motion + a ``base_pose`` for ``/odom`` without PhysX
        base joints.
        """
        x, y, yaw = self._base_pose
        dt = self._base_dt
        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        x += (vx * cos_y - vy * sin_y) * dt
        y += (vx * sin_y + vy * cos_y) * dt
        yaw += wyaw * dt
        self._base_pose = [x, y, yaw]
        self._place_robot()

    def _world_pose(self) -> tuple[float, float, float, float]:
        """World ``(x, y, z, yaw)`` of the robot root: spawn ∘ odom, plus the mount height."""
        wx, wy, wz, wyaw = compose_planar(self._spawn, self._base_pose)
        return wx, wy, wz + self._mount_z, wyaw

    def _place_robot(self) -> None:
        """Move the pinned root to spawn ∘ odom; the mounted cameras follow.

        Moves the base anchor and teleports the root together, so the pinning
        joint is satisfied at the new pose instead of yanking the robot back.
        """
        wx, wy, wz, wyaw = self._world_pose()
        self._set_anchor(wx, wy, wz, wyaw)
        self._robot.set_world_pose(np.array([wx, wy, wz]), _yaw_quat(wyaw))
        self._update_camera_poses()

    def _on_reset(self, rng: np.random.Generator) -> None:
        # Odometry restarts at the spawn each episode.
        self._base_pose = [0.0, 0.0, 0.0]

    def _after_world_reset(self) -> None:
        # world.reset() puts the pinned root back at its import pose (the
        # origin); move it to the spawn before the warmup steps settle physics.
        self._place_robot()
        # Hold the reset pose until the first command arrives.
        self._target = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
        self._robot.get_articulation_controller().apply_action(
            self._ArticulationAction(joint_positions=self._target)
        )

    def _observe(self) -> dict[str, Any]:
        obs = super()._observe()
        if self._has_base:
            obs["base_pose"] = np.asarray(self._base_pose, dtype=np.float32)  # [x, y, yaw]
        clouds = self._depth_clouds()
        if clouds:
            obs["depth_points"] = clouds
        scan = self._scan_ranges()
        if scan is not None:
            obs["scan"] = scan
        return obs

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        out: dict[str, NDArray[np.uint8]] = {}
        for meta in self._cam_meta:
            if meta["modality"] == "rgb":
                out[meta["key"]] = self._grab(self._cameras[meta["name"]])
        return out

    def _depth_clouds(self) -> dict[str, NDArray[np.float32]]:
        """``{sensor_name: (N, 3) base_frame points}`` from each depth camera.

        Uses Isaac's ``Camera.get_pointcloud(world_frame=True)`` — Isaac owns the
        camera intrinsics + frame convention, so we never guess the
        Isaac-camera↔REP-103-optical rotation — then transforms world→base_frame by
        the robot root's pose and the base_frame link's offset from the root
        (``points_in_frame``; misses are dropped). The openral-side
        ``SimSensorBridge`` publishes these as ``PointCloud2`` in the manifest's
        ``base_frame`` (already on /tf), so
        octomap gets a geometrically-correct cloud. Range-filtered (planar distance
        from the base) per the depth ``SensorSpec``. Empty when the manifest
        declares no depth sensor — never a fabricated cloud.
        """
        bx, by, bz, byaw = self._world_pose()
        world_from_root = np.eye(4)
        world_from_root[:3, :3] = [
            [np.cos(byaw), -np.sin(byaw), 0.0],
            [np.sin(byaw), np.cos(byaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
        world_from_root[:3, 3] = (bx, by, bz)
        world_from_base = world_from_root @ self._root_to_base
        out: dict[str, NDArray[np.float32]] = {}
        for meta in self._cam_meta:
            if meta["modality"] != "depth":
                continue
            pts = self._cameras[meta["name"]].get_pointcloud(world_frame=True)
            if pts is None:
                continue
            pb = points_in_frame(np.asarray(pts, dtype=np.float32), world_from_base)
            rmax = float(meta.get("range_max_m") or 0.0)
            if rmax > 0.0:
                pb = pb[(pb[:, 0] ** 2 + pb[:, 1] ** 2) <= rmax * rmax]
            if pb.size:
                out[meta["name"]] = pb
        return out

    def _state(self) -> NDArray[np.float32]:
        # No task object in a bare deploy scene — proprioception is the manifest
        # joint vector (what a JOINT_POSITION policy's state head expects).
        joints = self._joint_positions()
        return joints if joints is not None else np.zeros(0, dtype=np.float32)

    def _reward_terminated(self) -> tuple[float, bool]:
        # Bare bring-up scene: no task reward / termination.
        return 0.0, False

    def _extra_info(self) -> dict[str, Any]:
        """Simulator ground truth in the step ``info`` — never a policy observation.

        ``robot_position``: the articulation root's world ``[x, y, z]`` as PhysX
        holds it (where the robot physically is, vs. the integrated odometry);
        ``object_positions``: ``{name: [x, y, z]}`` per scene object, for
        checking a grasp or a placement.
        """
        info: dict[str, Any] = {
            "robot_position": [float(v) for v in self._robot.get_world_pose()[0]]
        }
        if self._object_prims:
            info["object_positions"] = {
                name: [float(v) for v in prim.get_world_pose()[0]]
                for name, prim in self._object_prims.items()
            }
        return info

    def _joint_positions(self) -> NDArray[np.float32] | None:
        if self._robot is None:
            return None
        return map_dof_to_manifest(
            np.asarray(self._robot.get_joint_positions(), dtype=np.float32),
            dof_index=self._dof_index,
            manifest_joints=self._manifest_joints,
            grippers=self._grippers,
            base_values=list(self._base_pose),  # base joints = kinematic (x, y, yaw)
            base_joints=self._base_joints,
        )

    def _joint_velocities(self) -> NDArray[np.float32] | None:
        if self._robot is None:
            return None
        # Base joint velocities are left at 0 (the kinematic base tracks pose, not
        # per-axis velocity); arm/gripper come from the articulation.
        return map_dof_to_manifest(
            np.asarray(self._robot.get_joint_velocities(), dtype=np.float32),
            dof_index=self._dof_index,
            manifest_joints=self._manifest_joints,
            grippers=self._grippers,
            base_values=[0.0 for _ in self._base_joints],
            base_joints=self._base_joints,
            rates=True,
        )
