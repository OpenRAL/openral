"""Robot-agnostic, URDF-driven Isaac Sim scene for the sidecar.

Runs under the Isaac Sim py3.11 venv only; imported by ``isaac_sidecar.py`` AFTER
``SimulationApp`` is live (every import here needs a running Kit app).

The only sidecar scene: it honours the forwarded ``--robot`` — it imports the manifest robot's URDF (``isaacsim.asset.importer.urdf``)
and wires joints/sensors/control from a plain-JSON "isaac robot spec" the
openral-side backend marshals across the venv boundary (the sidecar cannot
import ``openral_core``).

Control is by ABSOLUTE targets, the same meaning ``SimAttachedHAL`` sends: one
slot per non-base manifest joint in manifest order (arm in rad, grippers in their
end effector's ``command_convention``, mapped onto the URDF finger by
``manifest_to_urdf_gripper``;
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
                    "manifest_open", "command_closed", "command_open",
                    "followers": [{"dof", "multiplier", "offset"}]}],
      "base_joints": [str] | null,       # [forward, side, yaw] for a planar base
      "base_kinematics": str | null,
      "action": {"dim": int, "control_mode": "joint_position", "has_base": bool},
      "sensors": [{"name", "modality", "vla_feature_key", "frame_id",
                   "parent_frame", "intrinsics": {...}, "range_min_m",
                   "range_max_m", "n_channels"}],
      "camera_mounts": {sensor_name: {"link", "xyz", "quat_wxyz", "look_at", "axes"}},
                                       # optional; openral_sim...isaac_sim.IsaacCameraMount
      "initial_joint_positions": {manifest_joint: value},  # optional start pose
    }
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


def drive_gain_overrides(
    joint_drive_gains: dict[str, list[float]] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """The importer's per-joint ``(stiffness, damping)`` pattern dicts.

    ``joint_drive_gains`` is the spec's URDF joint-name regex -> ``[kp, kd]``
    (``IsaacSimOptions.joint_drive_gains``). The importer applies patterns in
    order, later matches winning, so a catch-all default goes first and every
    joint no pattern names keeps the stiff default.

    Example:
        >>> s, d = drive_gain_overrides({"joint[12]$": [70.0, 2.75]})
        >>> s
        {'.*': 1000.0, 'joint[12]$': 70.0}
        >>> d
        {'.*': 100.0, 'joint[12]$': 2.75}
    """
    gains = joint_drive_gains or {}
    stiffness = {".*": _DRIVE_STIFFNESS, **{p: float(kp) for p, (kp, _kd) in gains.items()}}
    damping = {".*": _DRIVE_DAMPING, **{p: float(kd) for p, (_kp, kd) in gains.items()}}
    return stiffness, damping


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


def _rpy_quat(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """``(w, x, y, z)`` of fixed-axis roll about x, then pitch about y, then yaw about z.

    The URDF ``rpy`` convention, ``R = Rz(yaw)·Ry(pitch)·Rx(roll)``.

    Example:
        >>> import numpy as np
        >>> np.allclose(_rpy_quat(0.0, 0.0, 0.3), _yaw_quat(0.3))
        True
        >>> np.round(_rpy_quat(np.pi / 2, 0.0, 0.0), 4).tolist()
        [0.7071, 0.7071, 0.0, 0.0]
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product ``a ⊗ b`` of ``(w, x, y, z)`` quaternions.

    Example:
        >>> import numpy as np
        >>> np.allclose(_quat_mul(_yaw_quat(0.2), _yaw_quat(0.3)), _yaw_quat(0.5))
        True
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _footprint(prim: Any, obj: dict[str, Any]) -> tuple[float, float, float, float]:
    """``(cx, cy, hx, hy)`` of a scene object's USD bounds after its roll/pitch, in its yawed frame."""
    from pxr import Usd, UsdGeom

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng_ = box.ComputeUntransformedBound(prim).ComputeAlignedRange()
    lo, hi = np.array(rng_.GetMin()), np.array(rng_.GetMax())
    corners = np.array(
        [
            [(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    cr, sr = np.cos(float(obj.get("roll", 0.0))), np.sin(float(obj.get("roll", 0.0)))
    cp, sp = np.cos(float(obj.get("pitch", 0.0))), np.sin(float(obj.get("pitch", 0.0)))
    rot = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    )
    xy = (corners @ rot.T)[:, :2]
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    return float((x0 + x1) / 2), float((y0 + y1) / 2), float((x1 - x0) / 2), float((y1 - y0) / 2)


def _rect_corners(
    x: float, y: float, yaw: float, rect: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    """World ``(4, 2)`` corners of a footprint ``(cx, cy, hx, hy)`` (object frame) at ``x, y, yaw``."""
    cx, cy, hx, hy = rect
    local = np.array(
        [[cx + sx * hx, cy + sy * hy] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + (x, y)


def _rects_overlap(a: NDArray[np.float64], b: NDArray[np.float64], margin: float) -> bool:
    """Separating-axis test of two convex quads ``(4, 2)``; ``margin`` (m) counts as contact."""
    for quad in (a, b):
        for i in range(4):
            edge = quad[(i + 1) % 4] - quad[i]
            axis = np.array([-edge[1], edge[0]]) / np.hypot(*edge)
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


def sample_object_poses(
    objects: list[dict[str, Any]],
    footprints: list[tuple[float, float, float, float]],
    rng: np.random.Generator,
    *,
    margin_m: float = 0.002,
) -> list[tuple[float, float, float, str]]:
    """Per-reset ``(x, y, yaw, status)`` of each scene object, in order.

    ``objects`` are the sidecar's object dicts; one with a ``pose_noise`` block
    (``IsaacObjectPoseNoise``) draws truncated-normal offsets around its declared
    ``xyz``/``yaw`` until its ``footprints`` rectangle (object frame, ``(cx, cy,
    hx, hy)``) clears every object placed before it and stays inside
    ``keep_inside_xy`` by ``margin_m``. ``status``: ``"declared"`` (no noise),
    ``"sampled"``, or ``"declared_fallback"`` (``max_tries`` draws all rejected).
    Deterministic for a given ``rng`` state.

    Example:
        >>> import numpy as np
        >>> objs = [
        ...     {
        ...         "xyz": [0, 0, 0],
        ...         "yaw": 0.0,
        ...         "pose_noise": {
        ...             "xy_sigma_m": [0.01, 0.01],
        ...             "yaw_sigma_deg": 5.0,
        ...             "clip_sigma": 2.0,
        ...             "keep_inside_xy": None,
        ...             "max_tries": 10,
        ...         },
        ...     }
        ... ]
        >>> x, y, yaw, status = sample_object_poses(
        ...     objs, [(0, 0, 0.02, 0.02)], np.random.default_rng(0)
        ... )[0]
        >>> status, bool(abs(x) <= 0.02 and abs(y) <= 0.02 and abs(yaw) <= np.radians(10))
        ('sampled', True)
    """
    placed: list[NDArray[np.float64]] = []
    out: list[tuple[float, float, float, str]] = []
    for obj, rect in zip(objects, footprints, strict=True):
        x0, y0, yaw0 = float(obj["xyz"][0]), float(obj["xyz"][1]), float(obj.get("yaw", 0.0))
        noise = obj.get("pose_noise")
        pose = (x0, y0, yaw0, "declared")
        if noise:
            sigma = np.array(
                [*noise["xy_sigma_m"], np.radians(noise["yaw_sigma_deg"])], dtype=np.float64
            )
            clip, box = float(noise["clip_sigma"]), noise.get("keep_inside_xy")
            pose = (x0, y0, yaw0, "declared_fallback")
            for _ in range(int(noise["max_tries"])):
                z = rng.standard_normal(3)
                while np.any(np.abs(z) > clip):  # truncate by redrawing, never by clamping
                    bad = np.abs(z) > clip
                    z[bad] = rng.standard_normal(int(bad.sum()))
                x, y, yaw = np.array([x0, y0, yaw0]) + z * sigma
                quad = _rect_corners(x, y, yaw, rect)
                if box is not None and (
                    np.any(quad < np.add(box[0], margin_m))
                    or np.any(quad > np.subtract(box[1], margin_m))
                ):
                    continue
                if any(_rects_overlap(quad, q, margin_m) for q in placed):
                    continue
                pose = (float(x), float(y), float(yaw), "sampled")
                break
        placed.append(_rect_corners(pose[0], pose[1], pose[2], rect))
        out.append(pose)
    return out


def pose_matrix(position: Any, quat_wxyz: Any) -> NDArray[np.float64]:
    """4x4 rigid transform from a position and a ``(w, x, y, z)`` quaternion (normalised).

    Example:
        >>> import numpy as np
        >>> np.round(pose_matrix((1.0, 2.0, 3.0), _yaw_quat(np.pi / 2)), 6).tolist()[0]
        [0.0, -1.0, 0.0, 1.0]
    """
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64) / np.linalg.norm(quat_wxyz)
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def manifest_to_urdf_gripper(gripper: dict[str, Any], value: float) -> float:
    """A gripper command (its end effector's ``command_convention``) → its URDF leader target.

    Linear between the gripper's command closed/open ends and its URDF
    closed/open targets (``_gripper_spec``), clamped to that travel — e.g.
    panda_mobile's ``normalized_close_symmetric`` +1.0 → closed (0.0 m), -1.0 →
    open (0.04 m); OpenArm's raw right jaw -0.785 → the URDF finger's -0.785.

    Example:
        >>> g = {"closed": 0.0, "open": 0.04, "command_closed": 1.0, "command_open": -1.0}
        >>> manifest_to_urdf_gripper(g, 1.0), round(manifest_to_urdf_gripper(g, -1.0), 3)
        (0.0, 0.04)
    """
    span = float(gripper["command_open"]) - float(gripper["command_closed"])
    frac = (float(value) - float(gripper["command_closed"])) / span if span else 0.0
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


def recolor_robot_materials(
    stage: Any, robot_prim: str, colors: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Set the diffuse colour of the robot's visual materials whose name matches a pattern.

    Upstream meshes carry their own materials (OpenArm's `.dae` "matte_black" is diffuse
    0.247, which renders mid-grey; the real plastic is near-black), so the colour has to be
    set after the URDF import. Only materials bound under ``robot_prim`` are touched; each
    ``fnmatch`` pattern is tested against the material prim's name, first match wins.
    Handles ``UsdPreviewSurface`` (``diffuseColor``) and OmniPBR MDL
    (``diffuse_color_constant``) shaders. Returns ``{material path: colour}``.
    """
    import fnmatch

    from pxr import Gf, Sdf, Usd, UsdShade

    def match(name: str) -> list[float] | None:
        return next((c for pat, c in colors.items() if fnmatch.fnmatch(name, pat)), None)

    root = stage.GetPrimAtPath(robot_prim)
    # The importer instances the visuals, so the bound materials are instance proxies,
    # which cannot be edited: walk the proxies, then de-instance only the instances that
    # hold a matching material; the rest (e.g. the collision geometry the finger pad binds)
    # stay instanced.
    paths = set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        mat, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat and match(mat.GetPrim().GetName()) is not None:
            paths.add(mat.GetPath())
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        while prim.IsInstanceProxy():
            anc = prim.GetParent()
            while not anc.IsInstance():
                anc = anc.GetParent()
            anc.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
    changed: dict[str, list[float]] = {}
    for path in sorted(paths, key=str):
        mat = UsdShade.Material(stage.GetPrimAtPath(path))
        rgb = match(mat.GetPrim().GetName())
        if rgb is None:
            continue
        # The importer's materials expose the colour as a Material interface input that
        # their UsdPreviewSurface reads; set it there when present.
        iface = mat.GetInput("diffuseColor")
        if iface:
            iface.Set(Gf.Vec3f(*rgb))
            changed[str(path)] = list(rgb)
        for child in Usd.PrimRange(mat.GetPrim()):
            shader = UsdShade.Shader(child)
            if not shader:
                continue
            for input_name in ("diffuseColor", "diffuse_color_constant"):
                inp = shader.GetInput(input_name) or (
                    shader.CreateInput(input_name, Sdf.ValueTypeNames.Color3f)
                    if shader.GetIdAttr().Get() == "UsdPreviewSurface"
                    and input_name == "diffuseColor"
                    else None
                )
                if not inp:
                    continue
                # A connected input reads its source (Isaac's URDF importer exposes the
                # colour as a Material interface input): setting the shader's own value
                # would be silently ignored, so write the source instead.
                for source in inp.GetConnectedSources()[0] if inp.HasConnectedSource() else []:
                    UsdShade.ConnectableAPI(source.source.GetPrim()).GetInput(
                        source.sourceName
                    ).Set(Gf.Vec3f(*rgb))
                if not inp.HasConnectedSource():
                    inp.Set(Gf.Vec3f(*rgb))
                changed[str(path)] = list(rgb)
    return changed


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


def camera_resolution(meta: dict[str, Any], obs_width: int, obs_height: int) -> tuple[int, int]:
    """A planned camera's render ``(width, height)``: its manifest intrinsics' raster.

    A policy camera renders at the raster its checkpoint was trained on, so its aspect
    survives the policy's resize. A depth camera keeps its aspect at no more than the
    scene's width: its cloud and registered frames cross the wire every step. A camera
    with no intrinsics takes the scene's ``observation_width``/``height``.

    Example:
        >>> camera_resolution(
        ...     {"modality": "rgb", "intrinsics": {"width": 672, "height": 376}}, 512, 384
        ... )
        (672, 376)
        >>> camera_resolution(
        ...     {"modality": "depth", "intrinsics": {"width": 1920, "height": 1080}}, 512, 384
        ... )
        (512, 288)
        >>> camera_resolution({"modality": "rgb"}, 512, 384)
        (512, 384)
    """
    k = meta.get("intrinsics") or {}
    w, h = int(k.get("width") or 0), int(k.get("height") or 0)
    if w <= 0 or h <= 0:
        return obs_width, obs_height
    if meta.get("modality") == "depth" and w > obs_width:
        return obs_width, max(1, round(h * obs_width / w))
    return w, h


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


def radial_fold_radius(coeffs: list[float]) -> float:
    """The largest distorted normalised radius a Brown/rational radial model reaches.

    ``r_d = r * (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)`` must
    increase with the undistorted radius ``r`` for a pixel to have a ray. Beyond its
    first turning point (or a pole) a distorted pixel has no preimage, so a renderer that
    inverts the model per output pixel draws garbage there. Tangential terms are ignored.
    Pure.

    Example:
        >>> round(radial_fold_radius([-0.358038981, 0.187982349, 0, 0, -0.056267709]), 3)
        0.858
        >>> radial_fold_radius([0.0] * 5) > 3.9
        True
    """
    k1, k2, _p1, _p2, k3, k4, k5, k6 = (list(coeffs) + [0.0] * 8)[:8]
    r = np.linspace(0.0, 4.0, 40001)[1:]
    r2 = r * r
    den = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = r * (1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / den
    bad = np.flatnonzero(~np.isfinite(rd[1:]) | (den[1:] <= 0) | (np.diff(rd) <= 0))
    rd = rd[: bad[0] + 1] if bad.size else rd
    return float(np.max(rd))


def check_radial_reaches_corners(name: str, k: dict[str, Any], coeffs: list[float]) -> None:
    """Refuse an RGB lens model that folds before the image corners.

    Isaac renders a distorted camera by inverting its model for every output pixel, so a
    calibration whose radial curve turns over inside the frame renders swirls at the
    edges (the Oct-2 ZED-M fit folds at 0.858 while its corners sit at 0.995). Extend it
    with ``rational_polynomial`` terms that stay monotonic instead. Pure.

    Example:
        >>> k = {"width": 672, "height": 376, "fx": 388.5, "fy": 388.5, "cx": 336.7, "cy": 190.1}
        >>> check_radial_reaches_corners("top", k, [-0.103, -0.002, 0.0, 0.0, 0.043])
        >>> check_radial_reaches_corners("top", k, [-0.358, 0.188, 0.0, 0.0, -0.0563])
        Traceback (most recent call last):
        ...
        ValueError: top: radial model folds at normalised radius 0.858, inside the image (corners at 0.995); extend it with rational_polynomial k4-k6
    """
    corner = max(
        float(np.hypot((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"]))
        for u in (0.0, float(k["width"]))
        for v in (0.0, float(k["height"]))
    )
    fold = radial_fold_radius(coeffs)
    if fold < corner:
        raise ValueError(
            f"{name}: radial model folds at normalised radius {fold:.3f}, inside the image "
            f"(corners at {corner:.3f}); extend it with rational_polynomial k4-k6"
        )


def blur_rgb(image: NDArray[np.uint8], sigma_px: float) -> NDArray[np.uint8]:
    """Separable Gaussian blur of an HWC uint8 frame; ``sigma_px <= 0`` returns it unchanged.

    RTX renders are sharper than a real low-cost camera after compression; a sub-pixel
    sigma matches their measured sharpness (scene option ``image_blur_sigma_px``). Pure.

    Example:
        >>> img = np.zeros((9, 9, 3), dtype=np.uint8)
        >>> img[4, 4] = 255
        >>> out = blur_rgb(img, 1.0)
        >>> int(out[4, 4, 0]) < 255 and int(out[4, 3, 0]) > 0
        True
        >>> blur_rgb(img, 0.0) is img
        True
    """
    if sigma_px <= 0:
        return image
    import cv2

    # Same normalised Gaussian (radius ceil(3 sigma), reflect-101 border) as a hand-rolled
    # separable convolution, but ~10x faster: this runs on every camera, every sim step.
    radius = max(1, int(np.ceil(3.0 * sigma_px)))
    out = cv2.GaussianBlur(
        image.astype(np.float32),
        (2 * radius + 1, 2 * radius + 1),
        sigmaX=sigma_px,
        sigmaY=sigma_px,
        borderType=cv2.BORDER_REFLECT_101,
    )  # float, not cv2's fixed-point uint8 path: same DN as the reference convolution
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# RTX tonemap op 6 (what exposure_ev selects) is the Narkowicz ACES fit followed by the
# sRGB encode: re-exposing a frame through its inverse matches a re-render at another
# exposure_ev to ~1 DN, where a display-space gain misses by ~11 DN.
_ACES_GRID: NDArray[np.float64] = np.linspace(0.0, 20.0, 200001, dtype=np.float64)


def _aces(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _srgb_decode(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(v: NDArray[np.float64]) -> NDArray[np.float64]:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


_ACES_Y = _aces(_ACES_GRID)
# uint8 display value -> scene-linear radiance under the op-6 tonemap.
_SCENE_FROM_DN = np.interp(
    np.clip(_srgb_decode(np.arange(256) / 255.0), 0.0, _ACES_Y[-1]), _ACES_Y, _ACES_GRID
)


def relight_rgb(
    image: NDArray[np.uint8], gain_rgb: tuple[float, float, float]
) -> NDArray[np.uint8]:
    """Re-expose an op-6-tonemapped RGB frame by scene-linear per-channel gains. Pure.

    Example:
        >>> img = np.full((2, 2, 3), 100, dtype=np.uint8)
        >>> int(relight_rgb(img, (2.0, 2.0, 2.0))[0, 0, 0]) > 100
        True
        >>> bool((relight_rgb(img, (1.0, 1.0, 1.0)) == img).all())
        True
    """
    # Each output DN depends only on its own channel's input DN: one 256-entry table per
    # channel, applied with cv2.LUT (exact; ~20x faster than the per-pixel float path).
    import cv2

    lut = np.stack([_relight_lut(float(g)) for g in gain_rgb], axis=-1)  # (256, 3)
    out: NDArray[np.uint8] = np.asarray(
        cv2.LUT(np.ascontiguousarray(image), lut.reshape(1, 256, 3)), dtype=np.uint8
    )
    return out


def _relight_lut(gain: float) -> NDArray[np.uint8]:
    """256-entry DN -> DN table of ``relight_rgb`` for one channel's scene-linear gain."""
    scene = _SCENE_FROM_DN * gain
    return np.clip(np.rint(_srgb_encode(_aces(scene)) * 255.0), 0, 255).astype(np.uint8)


def auto_exposure_gain(
    image: NDArray[np.uint8], target: float, wb_rgb: tuple[float, float, float]
) -> float:
    """Scene-linear gain putting the frame's mean BT.601 luma at ``target`` (metered 1:8). Pure.

    Example:
        >>> img = np.full((16, 16, 3), 60, dtype=np.uint8)
        >>> g = auto_exposure_gain(img, 120.0, (1.0, 1.0, 1.0))
        >>> abs(float(relight_rgb(img, (g, g, g)).mean()) - 120.0) < 1.5
        True
    """
    small = image[::8, ::8]
    lo, hi = 1 / 16, 16.0
    for _ in range(24):
        g = (lo * hi) ** 0.5
        y = relight_rgb(small, (g * wb_rgb[0], g * wb_rgb[1], g * wb_rgb[2])).astype(np.float64)
        if float((y @ np.array([0.299, 0.587, 0.114])).mean()) < target:
            lo = g
        else:
            hi = g
    return (lo * hi) ** 0.5


def _configure_rgb_projection(cam: Any, meta: dict[str, Any]) -> None:
    """Apply calibrated RGB K/D through Isaac's native OmniLensDistortion schema.

    Explicit mount FOVs keep their existing ideal pinhole model. Depth retains its
    existing pinhole path: its pointcloud API does not support distorted deprojection.
    """
    k = meta.get("intrinsics")
    if meta["modality"] != "rgb" or not k or (meta.get("mount") or {}).get("hfov_deg"):
        return
    width, height = cam.get_resolution()
    sx, sy = width / k["width"], height / k["height"]
    params = {"fx": k["fx"] * sx, "fy": k["fy"] * sy, "cx": k["cx"] * sx, "cy": k["cy"] * sy}
    model = k.get("distortion_model", "none")
    coeffs = list(k.get("distortion_coeffs") or [])
    if model == "equidistant":
        if len(coeffs) != 4:
            raise ValueError(f"{meta['name']}: equidistant requires four coefficients")
        cam.set_opencv_fisheye_properties(**params, fisheye=coeffs)
        readback = cam.get_opencv_fisheye_properties()
    elif model in ("none", "plumb_bob", "rational_polynomial"):
        if model == "none" and any(coeffs):
            raise ValueError(f"{meta['name']}: distortion_model none has nonzero coefficients")
        counts = {"none": (0,), "plumb_bob": (0, 5), "rational_polynomial": (8,)}[model]
        if len(coeffs) not in counts:
            raise ValueError(
                f"{meta['name']}: {model} requires {' or '.join(map(str, counts))} coefficients"
            )
        check_radial_reaches_corners(meta["name"], k, coeffs)
        # Isaac's OpenCV pinhole takes [k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4]: the rational
        # terms are positions 6-8, the thin-prism ones stay zero.
        coeffs += [0.0] * (12 - len(coeffs))
        cam.set_opencv_pinhole_properties(**params, pinhole=coeffs)
        readback = cam.get_opencv_pinhole_properties()
    else:
        raise ValueError(f"{meta['name']}: unsupported distortion model {model!r}")
    expected = [params[key] for key in ("cx", "cy", "fx", "fy")]
    if not np.allclose(readback[:4], expected) or not np.allclose(readback[4], coeffs):
        raise RuntimeError(f"{meta['name']}: camera calibration readback differs from request")
    print(
        f"[isaac_manifest_scene] {meta['name']} {width}x{height} K={params} "
        f"D={coeffs} model={model} (native lens schema readback OK)",
        flush=True,
    )


def _configure_rendering(spec: dict[str, Any]) -> None:
    """Opt-in process settings required by solid clear-plastic environment assets."""
    import carb

    settings = carb.settings.get_settings()
    values: dict[str, bool | int | float] = {}
    if spec.get("translucent_materials"):
        values.update(
            {
                "/rtx/material/enableRefraction": True,
                "/rtx/material/translucencyAsOpacity": False,
                "/rtx/sceneDb/translucencyAsOpacity": False,
                "/rtx/translucency/enabled": True,
                "/rtx/translucency/sampleRoughness": True,
                "/rtx/translucency/reflectAtAllBounce": True,
                "/rtx/translucency/maxRefractionBounces": 8,
                # Isaac 6 defaults to Real-Time 2.0; legacy translucency limits alone
                # do not change its three-bounce budget for a thick transparent shell.
                "/rtx/rtpt/maxBounces": 8,
                "/rtx/rtpt/maxSpecularAndTransmissionBounces": 12,
                "/rtx/pathtracing/maxBounces": 8,
                "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 12,
                "/rtx/indirectDiffuse/enabled": True,
                "/rtx/indirectDiffuse/maxBounces": 4,
                "/rtx/indirectDiffuse/scalingFactor": 1.0,
                # Low-resolution policy cameras need the quality DLSS profile.
                "/rtx/post/dlss/execMode": 2,
            }
        )
    ev = spec.get("exposure_ev")
    if ev is not None:
        values.update(
            {
                "/rtx/post/histogram/enabled": False,
                "/rtx/post/tonemap/op": 6,
                "/rtx/post/tonemap/colorMode": 0,
                "/rtx/post/tonemap/enableSrgbToGamma": True,
                "/rtx/post/tonemap/filmIso": 100.0,
                "/rtx/post/tonemap/exposureTime": 0.02,
                "/rtx/post/tonemap/fNumber": 5.0 * 2.0 ** (-float(ev) / 2.0),
                "/rtx/post/tonemap/responsivity": 1.1026709,
            }
        )
    for name, value in values.items():
        settings.set(name, value)
    if values:
        print(
            f"[isaac_manifest_scene] rendering settings: "
            f"{ {name: settings.get(name) for name in values} }",
            flush=True,
        )


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
        # Physics steps per env step; cameras render once, on the last (spec physics_substeps).
        self.physics_substeps = max(1, int(robot_spec.get("physics_substeps") or 1))
        # Auto-exposure state per camera: (scene-linear gain, sim time ns of that frame).
        self._ae_state: dict[str, tuple[float, int | None]] = {}
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
        # rad, gripper in its end effector's command_convention; NaN = hold), then the
        # base twist.
        by_gripper = {str(g["name"]): g for g in self._grippers}
        self._slot_plan: list[tuple[str, Any]] = [
            ("gripper", by_gripper[str(j["name"])])
            if j.get("role") == "gripper"
            else ("arm", str(j["urdf_name"]))
            for j in self._manifest_joints
            if j.get("role") in ("arm", "gripper")
        ]
        self.action_dim = len(self._slot_plan) + (3 if self._has_base else 0)
        # The scene's start pose, as one action (NaN = the URDF's own zero pose): each
        # reset teleports the joints there and holds it.
        initial: dict[str, float] = dict(robot_spec.get("initial_joint_positions") or {})
        self._initial_slots = np.array(
            [
                float(initial.get(str(j["name"]), np.nan))
                for j in self._manifest_joints
                if j.get("role") in ("arm", "gripper")
            ],
            dtype=np.float32,
        )
        self._objects: list[dict[str, Any]] = list(objects or [])
        # Per object: footprint (cx, cy, hx, hy) in its yawed frame, and the default
        # (position, wxyz) its prim was registered with -- the pose_noise anchor.
        self._object_footprints: dict[str, tuple[float, float, float, float]] = {}
        self._object_nominal: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
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
        # (prim path, 4x4 pose in the root frame at import) of the root rigid body.
        self._root_body: tuple[str, NDArray[np.float64]] = ("", np.eye(4))
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
        # The root rigid body and its pose in the root frame, read NOW — at the import
        # pose. Once the robot is placed, PhysX writes the body's world pose back while
        # the pinned root Xform stays at the origin, so the same read later returns the
        # spawn offset (and a camera mounted through it lands back at the origin).
        root_body = self._root_rigid_body_prim()
        if root_body:
            self._root_body = (root_body, self._relative_to_root(root_body))
        from isaacsim.core.utils.stage import get_current_stage

        # World.reset() initializes physics before the post-reset teleport.
        # Seed the imported joints too, so its first contact pass uses the
        # requested pose rather than the URDF zero pose among the scene props.
        from pxr import PhysxSchema, Usd, UsdPhysics

        initial_dofs: dict[str, float] = {}
        for (kind, ref), value in zip(self._slot_plan, self._initial_slots, strict=True):
            if np.isnan(value):
                continue
            if kind == "arm":
                initial_dofs[ref] = float(value)
            else:
                lead = manifest_to_urdf_gripper(ref, float(value))
                initial_dofs[ref["leader"]] = lead
                for follower in ref["followers"]:
                    initial_dofs[follower["dof"]] = float(follower["multiplier"]) * lead + float(
                        follower["offset"]
                    )
        for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(prim_path)):
            if prim.GetName() not in initial_dofs:
                continue
            angular = prim.IsA(UsdPhysics.RevoluteJoint)
            axis = UsdPhysics.Tokens.angular if angular else UsdPhysics.Tokens.linear
            value = initial_dofs[prim.GetName()]
            PhysxSchema.JointStateAPI.Apply(prim, axis).CreatePositionAttr(
                float(np.degrees(value)) if angular else value
            )

        finger_joints = [
            name
            for g in self._spec.get("grippers") or []
            for name in (g["leader"], *(f["dof"] for f in g["followers"]))
        ]
        colors = self._spec.get("robot_material_colors") or {}
        if colors:
            changed = recolor_robot_materials(get_current_stage(), prim_path, colors)
            print(
                f"[isaac_manifest_scene] robot material colours {colors}: "
                f"{len(changed)} material(s) recoloured {sorted(changed)}"
                + ("" if changed else " — WARNING: no robot material matched"),
                flush=True,
            )
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
                resolution=camera_resolution(meta, self.obs_width, self.obs_height),
            )
            self._cameras[meta["name"]] = cam

        # Pose of the manifest base_frame link in the robot root's frame: depth
        # clouds are published in base_frame, which need not be the URDF root
        # (OpenArm's openarm_base sits 0.698 m above it).
        self._root_to_base = self._link_offset(self._spec.get("base_frame"))
        self._world.reset()
        # Opening/importing a stage can restore renderer defaults. Apply the
        # requested settings after those operations and before the first image.
        _configure_rendering(self._spec)
        for meta in self._cam_meta:
            cam = self._cameras[meta["name"]]
            cam.initialize()
            cam.set_clipping_range(_CAM_NEAR_M, _CAM_FAR_M)
            hfov = camera_hfov_deg(meta)
            if hfov is not None:
                aperture = 2.0 * cam.get_focal_length() * float(np.tan(np.radians(hfov) / 2.0))
                cam.set_horizontal_aperture(aperture)
                width, height = camera_resolution(meta, self.obs_width, self.obs_height)
                cam.set_vertical_aperture(aperture * height / width)
            _configure_rgb_projection(cam, meta)
            if meta.get("mount"):
                self._mount_camera(cam, meta["mount"])
            if meta["modality"] == "depth":
                # distance_to_image_plane → the depth array we deproject to a cloud.
                cam.add_distance_to_image_plane_to_frame()
        self._place_robot()
        self._resolve_dof_mapping()
        self._after_world_reset()
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

        Each object is ``{"usd", "name", "xyz", "roll", "pitch", "yaw", "dynamic"}`` (validated
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
            quat = _rpy_quat(
                float(obj.get("roll", 0.0)),
                float(obj.get("pitch", 0.0)),
                float(obj.get("yaw", 0.0)),
            )
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
            self._object_footprints[name] = _footprint(stage.GetPrimAtPath(path), obj)
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
        try:
            path = self._link_prim_path(link)
        except ValueError:
            if link != self._spec.get("base_frame"):
                raise
            # base_frame is no URDF link (OpenArm's openarm_base): the manifest's
            # base_to_root transform, marshalled openral-side.
            return np.asarray(self._spec.get("root_to_base", np.eye(4)), dtype=np.float64)
        return self._relative_to_root(path)

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
            # Not the robot root prim: Isaac >= 6 places a pinned robot by its world
            # anchor, so the root Xform stays at the import origin while PhysX moves
            # the links — a camera parented to the root renders (and reports its pose)
            # from the origin, not from the robot (2026-10-04: the OpenArm head camera
            # looked at the warehouse floor 5 m from the robot). The first rigid body
            # under the root is fixed to it and is moved with it; its offset from the
            # root is the import-time one (``self._root_body``).
            body, root_from_body = self._root_body
            if not body:
                raise ValueError(
                    f"no rigid body under {self._robot_prim!r} to mount {link!r} on"
                ) from None
            return body, np.linalg.inv(root_from_body) @ self._link_offset(link)

    def _root_rigid_body_prim(self) -> str:
        """The first rigid body under the robot root (a fixed-base robot's root link), or ``""``."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdPhysics

        root = get_current_stage().GetPrimAtPath(self._robot_prim)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        return ""

    def _relative_to_root(self, prim_path: str) -> NDArray[np.float64]:
        """4x4 pose of ``prim_path`` in the robot root frame (the import pose, as USD holds it)."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom

        stage = get_current_stage()
        rel, _reset = UsdGeom.XformCache().ComputeRelativeTransform(
            stage.GetPrimAtPath(prim_path), stage.GetPrimAtPath(self._robot_prim)
        )
        return np.asarray(rel, dtype=np.float64).T  # Gf is row-vector: transpose to column

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
        position drives with explicit gains (``drive_gain_overrides``). It converts
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
        stiffness, damping = drive_gain_overrides(self._spec.get("joint_drive_gains"))
        config = URDFImporterConfig(
            urdf_path=urdf,
            usd_path=os.path.join(work, "usd"),
            # Keep the kinematic tree faithful to the manifest: DOFs map by URDF
            # joint name.
            merge_fixed_joints=False,
            fix_base=bool(self._spec.get("fix_base", True)),
            joint_target_type="position",
            override_joint_stiffness=stiffness,
            override_joint_damping=damping,
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
        if self._anchor_joint is not None and self._spec.get("fix_base", True):
            # The converter puts the root API on a body, creating a floating
            # articulation restrained by a solver joint. Mark the world joint
            # instead so PhysX creates a genuinely fixed-base articulation.
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            self._anchor_joint.GetBody0Rel().ClearTargets(True)
            UsdPhysics.ArticulationRootAPI.Apply(self._anchor_joint.GetPrim())
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

    def _write_slots(self, target: NDArray[np.float32], action: NDArray[np.float32]) -> None:
        """Write an action's joint slots into the DOF ``target`` vector (NaN slots hold)."""
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

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        """Set absolute joint targets (NaN slots hold) and advance the base twist."""
        if self._target is None:
            self._target = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
        target = self._target
        self._write_slots(target, action)
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
        self._jitter_objects(rng)

    def _jitter_objects(self, rng: np.random.Generator) -> None:
        """Redraw every ``pose_noise`` object's default pose; ``world.reset()`` then applies it."""
        if not any(o.get("pose_noise") for o in self._objects):
            return
        foot = [self._object_footprints[str(o["name"])] for o in self._objects]
        for obj, (x, y, yaw, status) in zip(
            self._objects, sample_object_poses(self._objects, foot, rng), strict=True
        ):
            if not obj.get("pose_noise"):
                continue
            name = str(obj["name"])
            prim = self._object_prims[name]
            if name not in self._object_nominal:
                st = prim.get_default_state()
                self._object_nominal[name] = (
                    np.asarray(st.position, dtype=np.float64),
                    np.asarray(st.orientation, dtype=np.float64),
                )
            p0, q0 = self._object_nominal[name]
            # Planar move of the declared root pose, applied to the registered prim
            # (the root, or a rigid child that rides it).
            x0, y0, yaw0 = float(obj["xyz"][0]), float(obj["xyz"][1]), float(obj.get("yaw", 0.0))
            d = yaw - yaw0
            c, s_ = np.cos(d), np.sin(d)
            rx, ry = p0[0] - x0, p0[1] - y0
            pos = np.array([x + c * rx - s_ * ry, y + s_ * rx + c * ry, p0[2]])
            prim.set_default_state(position=pos, orientation=_quat_mul(_yaw_quat(d), q0))
            print(
                f"[isaac_manifest_scene] object {name} {status}: x={x:.4f} y={y:.4f} "
                f"yaw={np.degrees(yaw):.1f}deg (declared {x0:.4f} {y0:.4f} {np.degrees(yaw0):.1f}deg)",
                flush=True,
            )

    def _after_world_reset(self) -> None:
        # world.reset() puts the pinned root back at its import pose (the
        # origin); move it to the spawn before the warmup steps settle physics.
        self._place_robot()
        # Hold the reset pose until the first command arrives: the URDF's zero pose, with
        # the scene's initial_joint_positions teleported in (no motion to get there).
        self._target = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
        if not np.all(np.isnan(self._initial_slots)):
            self._write_slots(self._target, self._initial_slots)
            self._robot.set_joint_positions(self._target)
            self._robot.set_joint_velocities(np.zeros_like(self._target))
            self._robot.set_joints_default_state(
                positions=self._target.copy(), velocities=np.zeros_like(self._target)
            )
        self._robot.get_articulation_controller().apply_action(
            self._ArticulationAction(joint_positions=self._target)
        )

    # Hold-correction passes after the warmup and physics steps per pass: the softest
    # real OpenArm joint (kp 10, kd 0.7) settles within ~0.5 s.
    _HOLD_PASSES = 3
    _HOLD_SETTLE_STEPS = 30

    def _after_warmup(self) -> None:
        """Raise the hold target until the drives hold the reset pose under gravity.

        A position drive rests gravity torque / stiffness below its target. With real
        motor gains (``joint_drive_gains``) the teleported start pose would sag by
        centimetres, and the policy would open on an arm lower than the real robot's
        measured start state, which is already a sagged pose held by a higher command.
        Iterating ``target += pose - measured`` converges on that command; stiff
        drives make it a no-op.
        """
        if self._target is None:
            return
        pose = self._target.copy()
        for _ in range(self._HOLD_PASSES):
            err = pose - np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
            err[np.isnan(err)] = 0.0
            if float(np.abs(err).max()) < 1e-4:
                break
            self._target += err
            self._robot.get_articulation_controller().apply_action(
                self._ArticulationAction(joint_positions=self._target)
            )
            for _ in range(self._HOLD_SETTLE_STEPS):
                self._world.step(render=False)
        print(
            "[isaac_manifest_scene] hold target raised by up to "
            f"{float(np.abs(self._target - pose).max()):.4f} rad to hold the reset pose "
            "under gravity",
            flush=True,
        )

    def _observe(self) -> dict[str, Any]:
        obs = super()._observe()
        if self._has_base:
            obs["base_pose"] = np.asarray(self._base_pose, dtype=np.float32)  # [x, y, yaw]
        clouds = self._depth_clouds()
        if clouds:
            obs["depth_points"] = clouds
        frames = self._depth_frames()
        if frames:
            obs["depth_frames"] = frames
        scan = self._scan_ranges()
        if scan is not None:
            obs["scan"] = scan
        return obs

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        out: dict[str, NDArray[np.uint8]] = {}
        posts = self._spec.get("camera_image_post") or {}
        now = self.sim_time_ns()
        for meta in self._cam_meta:
            if meta["modality"] != "rgb":
                continue
            post = posts.get(meta["name"], {})
            sigma = post.get("blur_sigma_px")
            if sigma is None:
                sigma = self._spec.get("image_blur_sigma_px") or 0
            # The blur is in manifest pixels: a camera rendered smaller blurs proportionally.
            scale = float((self._spec.get("camera_render_scale") or {}).get(meta["name"], 1.0))
            frame = blur_rgb(self._grab(self._cameras[meta["name"]]), float(sigma) * scale)
            wb = tuple(post.get("white_balance_rgb") or (1.0, 1.0, 1.0))
            target = post.get("auto_exposure_target")
            gain = 1.0
            if target is not None:
                want = auto_exposure_gain(frame, float(target), wb)
                prev = self._ae_state.get(meta["name"])
                if prev is None or now is None or now <= prev[1]:  # first frame / reset
                    gain = want
                else:
                    dt_s = (now - prev[1]) * 1e-9
                    alpha = 1.0 - float(np.exp(-dt_s / post["auto_exposure_tau_s"]))
                    gain = prev[0] * (want / prev[0]) ** alpha  # first-order, in log exposure
                self._ae_state[meta["name"]] = (gain, now)
            if gain != 1.0 or wb != (1.0, 1.0, 1.0):
                frame = relight_rgb(frame, (gain * wb[0], gain * wb[1], gain * wb[2]))
            out[meta["key"]] = frame
        return out

    def _world_from_base(self) -> NDArray[np.float64]:
        """4x4 pose of the manifest ``base_frame`` in the world, where PhysX holds the robot.

        From the articulation root's measured pose, not the commanded spawn: the two
        differ when PhysX does not honour the spawn (a spawn below the ground plane
        stays at ground level), and a cloud expressed against the commanded pose then
        disagrees with the robot's own links by that offset — the self-filter misses
        the arm, the kernel stops it against its own surface.
        """
        pos, quat_wxyz = self._robot.get_world_pose()
        return pose_matrix(pos, quat_wxyz) @ self._root_to_base

    def _depth_frames(self) -> dict[str, dict[str, Any]]:
        """Registered colour + metric depth per depth camera — what a real RGB-D driver publishes.

        ``{sensor_name: {"depth", "rgb", "k", "optical_in_base"}}``: the
        ``distance_to_image_plane`` raster (metres, ``0.0`` = no return or past the
        sensor's ``range_max_m``) and the RGB of the SAME Isaac camera in the same
        render, so a mask on one is a mask on the other; ``k`` = ``[fx, fy, cx, cy]``
        from Isaac's own intrinsics matrix; ``optical_in_base`` = the camera's
        REP-103 optical pose (``camera_axes="ros"``) in the manifest ``base_frame``
        as a 4x4. The openral-side ``SimSensorBridge`` publishes these as Image +
        CameraInfo pairs and the optical frame on /tf — the four streams the vision
        attachment leg (segmenter + HAL bridge) consumes. Empty when the manifest
        declares no depth sensor.
        """
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

        base_from_world = np.linalg.inv(self._world_from_base())
        out: dict[str, dict[str, Any]] = {}
        for meta in self._cam_meta:
            if meta["modality"] != "depth":
                continue
            cam = self._cameras[meta["name"]]
            depth = cam.get_depth()
            if depth is None or np.asarray(depth).size == 0:
                continue
            raster = np.nan_to_num(
                np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            raster[raster < 0.0] = 0.0
            rmax = float(meta.get("range_max_m") or 0.0)
            if rmax > 0.0:
                raster[raster > rmax] = 0.0
            k = np.asarray(cam.get_intrinsics_matrix(), dtype=np.float64)
            pos, quat_wxyz = cam.get_world_pose(camera_axes="ros")
            world_from_optical = np.eye(4)
            world_from_optical[:3, :3] = quats_to_rot_matrices(
                np.asarray(quat_wxyz, dtype=np.float64)[None]
            )[0]
            world_from_optical[:3, 3] = np.asarray(pos, dtype=np.float64)
            out[meta["name"]] = {
                "depth": raster,
                "rgb": self._grab(cam),
                "k": np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64),
                "optical_in_base": base_from_world @ world_from_optical,
            }
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
        world_from_base = self._world_from_base()
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
        checking a grasp or a placement; ``object_orientations_wxyz``: their
        orientations (``pose_noise`` draws a new yaw each reset).
        """
        info: dict[str, Any] = {
            "robot_position": [float(v) for v in self._robot.get_world_pose()[0]]
        }
        if self._object_prims:
            info["object_positions"] = {
                name: [float(v) for v in prim.get_world_pose()[0]]
                for name, prim in self._object_prims.items()
            }
            info["object_orientations_wxyz"] = {
                name: [float(v) for v in prim.get_world_pose()[1]]
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
