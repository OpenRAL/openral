"""Fit a depth camera's mount to the robot it looks at: the library behind the extrinsic tools.

The camera sees the robot, and the robot's shape at a joint configuration is known exactly
from its model. So the robot is the calibration target: record depth points at several
poses together with the REAL joint readings, pose the robot's meshes at those readings,
and solve for the one ``parent_frame -> frame_id`` mount that puts the robot's returns on
its surfaces in every pose at once. Nothing is measured by hand.

Robot-agnostic by construction. Everything comes from the robot manifest (the depth
sensor, its ``parent_frame``, ``frame_id`` and intrinsics, the base frame, the joints)
and from the robot's own model: its MJCF when it has one (MuJoCo), else its URDF
(yourdfpy + trimesh, the ``lowering`` group). The mount's parent may be any link: it is
posed by forward kinematics at each recorded configuration, so a head camera on a torso
fits the same way as a camera on a fixed base. The poses are per robot type, in
``robots/<id>/calibration/extrinsic_fit_poses.yaml``.

Why several poses: the fit only sees an axis if moving the camera along it moves the
robot's returns off the robot. With an arm hanging straight down, sliding the camera
along the arm or turning it about the vertical changes almost nothing, so height and yaw
are unobservable (measured on the OpenArm cells, 2026-09-26). Poses with the links at
different angles make every axis visible; :func:`fit_mount` measures that per axis
(``axis_unrecovered``: restarted a full limit off, does the fit come back?) and
cross-checks the result by refitting with each pose left out (``heldout_spread``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from openral_core import RobotDescription, SensorSpec
from openral_core.depth_extrinsic import (
    ERROR_LIMITS,
    FIT_AXES,
    MAX_HEIGHT_ERR_M,
    MAX_PLANAR_ERR_M,
    MAX_TILT_DEG,
    MAX_YAW_DEG,
    MIN_FIT_POSES,
)
from openral_core.exceptions import ROSConfigError
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

Mat = NDArray[np.float64]
Points = NDArray[np.float64]

#: Surface sample density on the robot's meshes (points per square metre).
_SAMPLES_PER_M2 = 200_000.0
#: Coarse-to-fine match gates (m): a point further than this from the robot is not the robot.
_GATES_M = (0.05, 0.03, 0.02)
#: Trimmed ICP: the worst fifth of matches per iteration is ignored (scene points near
#: the robot, depth-edge flying pixels).
_KEEP_FRACTION = 0.8
_ITERATIONS = 25
#: Held-out refits start from the full fit and use the finest gate only.
_HELDOUT_GATE_M = 0.02
#: Recovery refits start a full limit (<= 15 mm) off, so they need the wider gate first.
_RECOVERY_GATES_M = (0.03, 0.02)


# ── The committed pose list ───────────────────────────────────────────────────


class FitPose(BaseModel):
    """One configuration to record at; joints not listed are commanded to 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    joints: dict[str, float]


class FitPoseFile(BaseModel):
    """``robots/<id>/calibration/extrinsic_fit_poses.yaml``.

    Example:
        >>> FitPoseFile.model_validate(
        ...     {"schema_version": "0.1", "robot_id": "r", "poses": [{"name": "a", "joints": {}}]}
        ... ).poses[0].name
        'a'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"]
    robot_id: str
    poses: list[FitPose] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> FitPoseFile:
        """Parse and validate the YAML at ``path``."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def targets(
        self, robot_id: str, description: RobotDescription
    ) -> list[tuple[str, dict[str, float]]]:
        """Each pose as a full, limit-checked ``{joint: position}`` over the manifest's joints.

        Args:
            robot_id: The robot's directory name (``robots/<robot_id>/``).
            description: Its manifest.

        Raises:
            ROSConfigError: another robot's file, fewer poses than a fit needs, an
                unknown joint, or a value outside the joint's limits.
        """
        if self.robot_id != robot_id:
            raise ROSConfigError(f"pose file is for robot {self.robot_id!r}, not {robot_id!r}")
        if len(self.poses) < MIN_FIT_POSES:
            raise ROSConfigError(f"{len(self.poses)} poses; a fit needs {MIN_FIT_POSES}")
        known = {j.name: j for j in description.joints}
        out = []
        for pose in self.poses:
            unknown = sorted(set(pose.joints) - set(known))
            if unknown:
                raise ROSConfigError(f"pose {pose.name!r}: unknown joints {unknown}")
            q = {name: float(pose.joints.get(name, 0.0)) for name in known}
            for name, value in q.items():
                limits = known[name].position_limits
                if not math.isfinite(value) or (
                    limits is not None and not limits[0] <= value <= limits[1]
                ):
                    raise ROSConfigError(f"pose {pose.name!r}: {name}={value} outside {limits}")
            out.append((pose.name, q))
        return out


# ── Geometry helpers ──────────────────────────────────────────────────────────


def xyzrpy_to_matrix(v: Sequence[float]) -> Mat:
    """``static_transform_publisher``'s convention: fixed-axis XYZ, R = Rz Ry Rx."""
    x, y, z, roll, pitch, yaw = (float(a) for a in v)
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    out = np.eye(4)
    out[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    out[:3, 3] = [x, y, z]
    return out


def matrix_to_xyzrpy(m: Mat) -> list[float]:
    """Inverse of :func:`xyzrpy_to_matrix`."""
    r = m[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return [float(m[0, 3]), float(m[1, 3]), float(m[2, 3]), roll, pitch, yaw]


def _rot(axis: int, angle: float) -> Mat:
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    out = np.eye(4)
    out[i, i], out[i, j], out[j, i], out[j, j] = c, -s, s, c
    return out


def _apply(t: Mat, pts: Points) -> Points:
    out: Points = pts @ t[:3, :3].T + t[:3, 3]
    return out


def _sample_triangles(
    verts: Points, faces: NDArray[np.int64], rng: np.random.Generator
) -> tuple[Points, Points]:
    """Area-weighted surface samples of a triangle mesh, with their face normals."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    k = max(1, int(total * _SAMPLES_PER_M2))
    tri = rng.choice(len(faces), size=k, p=area / total)
    u, w = rng.random(k), rng.random(k)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + w[:, None] * (c[tri] - a[tri])
    normals = cross[tri] / np.maximum(np.linalg.norm(cross[tri], axis=1), 1e-12)[:, None]
    return pts, normals


# ── The robot model: MJCF or URDF ─────────────────────────────────────────────


class RobotSurface:
    """The robot's surfaces and frames at a joint configuration, in its base frame.

    Built from the robot's MJCF (preferred: it resolves its own meshes) or, failing that,
    its URDF. ``set(q)`` poses it; ``frame(name)`` is ``base_frame -> name``;
    ``surface()`` is every mesh of the robot as (points, normals).
    """

    def __init__(self, description: RobotDescription, manifest_dir: Path | None) -> None:
        from openral_core.assets import resolve_asset

        self.description = description
        self._rng = np.random.default_rng(0)
        mjcf = (
            resolve_asset(description.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
            if description.assets.mjcf
            else None
        )
        if mjcf is not None:
            self._init_mjcf(Path(mjcf))
            return
        urdf = (
            resolve_asset(description.assets.urdf.ref, "urdf", manifest_dir=manifest_dir)
            if description.assets.urdf
            else None
        )
        if urdf is None:
            raise ROSConfigError(
                f"robot {description.name!r} has neither an MJCF nor a URDF with meshes; the "
                "extrinsic fit needs the robot's shape"
            )
        self._init_urdf(Path(urdf))

    # MJCF ---------------------------------------------------------------------

    def _init_mjcf(self, path: Path) -> None:
        import mujoco

        # Any: MuJoCo's pybind objects carry no usable stubs for mypy --strict.
        self._m: Any = mujoco.MjModel.from_xml_path(str(path))
        self._d: Any = mujoco.MjData(self._m)
        m = self._m
        self._qadr: dict[str, int] = {}
        joint_of: dict[str, int] = {}
        for j in self.description.joints:
            for name in (j.sim_joint_name, j.name):
                ji = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)) if name else -1
                if ji >= 0:
                    self._qadr[j.name] = int(m.jnt_qposadr[ji])
                    joint_of[j.name] = ji
                    break
        missing = [j.name for j in self.description.joints if j.name not in self._qadr]
        if missing:
            raise ROSConfigError(f"MJCF {path.name} has no joint for manifest joints {missing}")
        # Equality-coupled followers (a second finger), set from their leader like the sim.
        self._followers: list[tuple[int, int, float, float]] = []
        leaders = {ji: self._qadr[n] for n, ji in joint_of.items()}
        for e in range(m.neq):
            if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT) or int(m.eq_obj2id[e]) < 0:
                continue
            a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            c0, c1 = float(m.eq_data[e][0]), float(m.eq_data[e][1])
            if b in leaders and a not in leaders:
                self._followers.append((int(m.jnt_qposadr[a]), leaders[b], c0, c1))
            elif a in leaders and b not in leaders and c1 != 0.0:
                self._followers.append((int(m.jnt_qposadr[b]), leaders[a], -c0 / c1, 1.0 / c1))
        # Frame name -> MJCF body: a body of that name, else the body a manifest joint
        # moves (its child link), else the world for the manifest's base frame.
        self._body: dict[str, int] = {}
        for j in self.description.joints:
            if j.name in joint_of:
                self._body.setdefault(j.child_link, int(m.jnt_bodyid[joint_of[j.name]]))
        # Surface samples in each geom's own frame (every mesh of a robot body).
        self._geoms: list[tuple[int, Points, Points]] = []
        for g in range(m.ngeom):
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH) or int(m.geom_bodyid[g]) == 0:
                continue
            mi = int(m.geom_dataid[g])
            verts = np.asarray(
                m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]],
                dtype=np.float64,
            )
            faces = np.asarray(
                m.mesh_face[m.mesh_faceadr[mi] : m.mesh_faceadr[mi] + m.mesh_facenum[mi]],
                dtype=np.int64,
            )
            pts, normals = _sample_triangles(verts, faces, self._rng)
            self._geoms.append((g, pts, normals))
        self._kind = "mjcf"
        self.set({j.name: 0.0 for j in self.description.joints})

    def _mj_body_pose(self, name: str) -> Mat:
        import mujoco

        bid = int(mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, name))
        if bid < 0:
            bid = self._body.get(name, -1)
        if bid < 0:
            if name != self.description.base_frame:
                raise ROSConfigError(f"frame {name!r} is not a body of the robot's MJCF")
            bid = 0  # the manifest's base frame is the MJCF world
        out = np.eye(4)
        out[:3, :3] = np.asarray(self._d.xmat[bid], dtype=np.float64).reshape(3, 3)
        out[:3, 3] = self._d.xpos[bid]
        return out

    # URDF ---------------------------------------------------------------------

    def _init_urdf(self, path: Path) -> None:
        import yourdfpy

        # Any: yourdfpy ships no type information.
        self._urdf: Any = yourdfpy.URDF.load(str(path), load_meshes=True, build_scene_graph=True)
        self._kind = "urdf"
        self.set({j.name: 0.0 for j in self.description.joints})

    # Common -------------------------------------------------------------------

    def set(self, q: dict[str, float]) -> None:
        """Pose the robot at ``q`` (manifest joint names, radians / metres)."""
        if self._kind == "mjcf":
            import mujoco

            self._d.qpos[:] = self._m.qpos0
            for name, value in q.items():
                self._d.qpos[self._qadr[name]] = value
            for adr, leader, c0, c1 in self._followers:
                self._d.qpos[adr] = c0 + c1 * self._d.qpos[leader]
            mujoco.mj_kinematics(self._m, self._d)
            self._base_inv = np.linalg.inv(self._mj_body_pose(self.description.base_frame))
        else:
            self._urdf.update_cfg(
                {k: v for k, v in q.items() if k in self._urdf.actuated_joint_names}
            )

    def self_contact_depth(
        self, qa: dict[str, float], qb: dict[str, float], samples: int
    ) -> float | None:
        """Deepest robot-on-robot mesh interpenetration along the straight ramp ``qa -> qb``.

        What the capture's joint-space ramps sweep through (the runner's starting-pose
        ramp is linear in joint space too). ``None`` for a URDF-only robot: there the
        safety kernel's own self-collision check, live, is the only guard. At runtime the
        kernel is the authority either way; this is the offline preview.
        """
        if self._kind != "mjcf":
            return None
        import mujoco

        worst = 0.0
        for s in np.linspace(0.0, 1.0, samples):
            self.set({k: qa[k] + s * (qb[k] - qa[k]) for k in qa})
            mujoco.mj_forward(self._m, self._d)
            for c in self._d.contact[: self._d.ncon]:
                if int(self._m.geom_bodyid[c.geom1]) and int(self._m.geom_bodyid[c.geom2]):
                    worst = max(worst, -float(c.dist))
        return worst

    def frame(self, name: str) -> Mat:
        """``base_frame -> name`` at the current configuration."""
        if self._kind == "mjcf":
            out: Mat = self._base_inv @ self._mj_body_pose(name)
            return out
        base = self.description.base_frame
        return np.asarray(
            self._urdf.get_transform(frame_to=name, frame_from=base), dtype=np.float64
        )

    def surface(self) -> tuple[Points, Points]:
        """Every robot mesh sampled at the current configuration, in the base frame."""
        if self._kind == "mjcf":
            parts, norms = [], []
            for g, pts, normals in self._geoms:
                t = np.eye(4)
                t[:3, :3] = np.asarray(self._d.geom_xmat[g], dtype=np.float64).reshape(3, 3)
                t[:3, 3] = self._d.geom_xpos[g]
                t = self._base_inv @ t
                parts.append(_apply(t, pts))
                norms.append(normals @ t[:3, :3].T)
            return np.vstack(parts), np.vstack(norms)
        import trimesh

        mesh = self._urdf.scene.to_geometry()
        base = self.description.base_frame
        root_in_base = np.asarray(
            self._urdf.get_transform(frame_to=self._urdf.base_link, frame_from=base),
            dtype=np.float64,
        )
        pts, face = trimesh.sample.sample_surface(
            mesh, max(1, int(mesh.area * _SAMPLES_PER_M2)), seed=0
        )
        return _apply(root_in_base, np.asarray(pts, dtype=np.float64)), np.asarray(
            mesh.face_normals[face], dtype=np.float64
        ) @ root_in_base[:3, :3].T


# ── Recording format ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseCapture:
    """One recorded pose: depth points in the camera's MOUNT frame, and the joints."""

    name: str
    points: Points
    joints: dict[str, float]


def write_capture(
    directory: Path, meta: dict[str, object], captures: Sequence[PoseCapture]
) -> None:
    """``capture.json`` plus one ``pose_NN.npz`` per pose."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for i, c in enumerate(captures):
        names = sorted(c.joints)
        np.savez_compressed(
            directory / f"pose_{i:02d}.npz",
            name=np.array(c.name),
            points=c.points.astype(np.float32),
            joint_names=np.array(names),
            joint_positions=np.array([c.joints[n] for n in names], dtype=np.float64),
        )


def read_capture(directory: Path) -> tuple[dict[str, Any], list[PoseCapture]]:
    """Inverse of :func:`write_capture`."""
    meta_path = directory / "capture.json"
    if not meta_path.is_file():
        raise ROSConfigError(f"{directory} is not a capture (no capture.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = []
    for f in sorted(directory.glob("pose_*.npz")):
        with np.load(f) as z:
            joints = dict(
                zip(
                    (str(n) for n in z["joint_names"]),
                    (float(v) for v in z["joint_positions"]),
                    strict=True,
                )
            )
            out.append(
                PoseCapture(str(z["name"]), np.asarray(z["points"], dtype=np.float64), joints)
            )
    return meta, out


# ── Simulated capture (pose planning, tests) ──────────────────────────────────


def camera_axes(frame_id: str) -> Mat:
    """Rotation taking optical coordinates (z forward, x right, y down) into ``frame_id``.

    ROS REP 103/105: a ``*_optical_frame`` is optical; any other camera frame is a body
    frame (x forward, y left, z up).
    """
    if frame_id.endswith(("optical_frame", "_optical")):
        return np.eye(3)
    return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def simulate_capture(
    robot: RobotSurface,
    spec: SensorSpec,
    true_mount: Mat,
    q: dict[str, float],
    *,
    noise_m: float = 0.002,
    rng: np.random.Generator,
    max_points: int = 6000,
) -> Points:
    """The depth points the camera would return from the robot at ``q``, in its mount frame.

    Robot surfaces facing the camera, inside its field of view (from the manifest
    intrinsics) and 0.1-2.0 m away, with Gaussian range noise. Self-occlusion beyond
    back-face culling is not modelled, which only makes a pose look better than it is;
    the real recording is what the gate judges.
    """
    assert spec.parent_frame is not None and spec.intrinsics is not None
    robot.set(q)
    cam_in_base = robot.frame(spec.parent_frame) @ true_mount
    pts, normals = robot.surface()
    local = _apply(np.asarray(np.linalg.inv(cam_in_base), dtype=np.float64), pts)
    optical = local @ camera_axes(spec.frame_id)  # rows: mount -> optical
    n_local = normals @ cam_in_base[:3, :3]
    facing = (n_local * -local).sum(axis=1) > 0
    z = optical[:, 2]
    k = spec.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k.fx * optical[:, 0] / z + k.cx
        v = k.fy * optical[:, 1] / z + k.cy
    keep = facing & (z > 0.1) & (z < 2.0) & (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    seen = local[keep]
    if len(seen) > max_points:
        seen = seen[rng.choice(len(seen), max_points, replace=False)]
    ray = seen / np.maximum(np.linalg.norm(seen, axis=1), 1e-9)[:, None]
    noisy: Points = seen + ray * rng.normal(0.0, noise_m, size=(len(seen), 1))
    return noisy


# ── The fit ───────────────────────────────────────────────────────────────────


@dataclass
class _Pose:
    name: str
    points: Points  # mount frame
    parent: Mat  # base -> parent at this pose
    surface: Points  # robot surface, base frame
    tree: cKDTree


def _prepare(robot: RobotSurface, spec: SensorSpec, captures: Sequence[PoseCapture]) -> list[_Pose]:
    assert spec.parent_frame is not None
    out = []
    for c in captures:
        q = {j.name: c.joints.get(j.name, 0.0) for j in robot.description.joints}
        missing = [j.name for j in robot.description.joints if j.name not in c.joints]
        if missing:
            raise ROSConfigError(f"pose {c.name!r} recorded no reading for joints {missing}")
        robot.set(q)
        surf, _ = robot.surface()
        out.append(_Pose(c.name, c.points, robot.frame(spec.parent_frame), surf, cKDTree(surf)))
    return out


def _distances(poses: Sequence[_Pose], mount: Mat, picks: Sequence[NDArray[np.bool_]]) -> Points:
    return np.concatenate(
        [
            p.tree.query(_apply(p.parent @ mount, p.points[k]))[0]
            for p, k in zip(poses, picks, strict=True)
        ]
    )


def _icp(
    poses: Sequence[_Pose], mount: Mat, gates: Sequence[float]
) -> tuple[Mat, list[NDArray[np.bool_]]]:
    picks: list[NDArray[np.bool_]] = []
    for gate in gates:
        picks = [p.tree.query(_apply(p.parent @ mount, p.points))[0] <= gate for p in poses]
        for _ in range(_ITERATIONS):
            src, dst = [], []
            for p, k in zip(poses, picks, strict=True):
                in_parent = _apply(mount, p.points[k])
                d, i = p.tree.query(_apply(p.parent, in_parent))
                if len(d) == 0:
                    continue
                keep = d <= np.quantile(d, _KEEP_FRACTION)
                src.append(in_parent[keep])
                dst.append(
                    _apply(
                        np.asarray(np.linalg.inv(p.parent), dtype=np.float64), p.surface[i[keep]]
                    )
                )
            if not src:
                raise ROSConfigError(
                    f"no recorded point lies within {gate} m of the robot: is the camera "
                    "looking at the robot, and is the starting mount within ~5 cm?"
                )
            a, b = np.vstack(src), np.vstack(dst)
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
            r = vt.T @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))]) @ u.T
            step = np.eye(4)
            step[:3, :3], step[:3, 3] = r, cb - r @ ca
            mount = step @ mount
            angle = math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))
            if angle < 1e-6 and float(np.linalg.norm(step[:3, 3])) < 1e-6:
                break  # converged: another iteration would not move the mount
    return mount, picks


def mount_error(a: Mat, b: Mat, parent: Mat) -> dict[str, float]:
    """How far mount ``b`` puts the camera from mount ``a``, in the base frame.

    ``height_m``/``planar_m`` split the camera's displacement into vertical and
    horizontal; ``tilt_deg`` is the relative rotation's tilt of the vertical;
    ``yaw_deg`` its turn about the vertical.
    """
    ca, cb = parent @ a, parent @ b
    dt = cb[:3, 3] - ca[:3, 3]
    dr = cb[:3, :3] @ ca[:3, :3].T
    return {
        "height_m": float(abs(dt[2])),
        "planar_m": float(math.hypot(dt[0], dt[1])),
        "tilt_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, float(dr[2, 2])))))),
        "yaw_deg": float(abs(math.degrees(math.atan2(float(dr[1, 0]), float(dr[0, 0]))))),
    }


def _perturbed(mount: Mat, parent: Mat, axis: str) -> Mat:
    """``mount`` moved by one axis limit in the base frame (rotations about the camera)."""
    cam = parent @ mount
    step = np.eye(4)
    if axis in ("x", "y"):
        step[["x", "y"].index(axis), 3] = MAX_PLANAR_ERR_M
        moved = step @ cam
    elif axis == "z":
        step[2, 3] = MAX_HEIGHT_ERR_M
        moved = step @ cam
    else:
        angle = math.radians(MAX_YAW_DEG if axis == "yaw" else MAX_TILT_DEG)
        rot = _rot(["roll", "pitch", "yaw"].index(axis), angle)
        about = np.eye(4)
        about[:3, 3] = cam[:3, 3]
        moved = about @ rot @ np.linalg.inv(about) @ cam
    out: Mat = np.linalg.inv(parent) @ moved
    return out


def fit_mount(
    robot: RobotSurface,
    spec: SensorSpec,
    captures: Sequence[PoseCapture],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Mat, dict[str, Any]]:
    """Fit ``spec``'s mount to the robot's returns; return it and the report residuals.

    Starts from the manifest mount (``static_transform_xyz_rpy``), which must be within a
    few centimetres. Residuals: the median robot residual, the manifest mount's error
    against the fit, the held-out spread (worst refit with one pose left out, relative
    to the full fit) and, per axis, how much of a one-limit displacement a restarted fit
    leaves unrecovered — all judged by
    ``openral_core.depth_extrinsic.residual_failures``.
    """
    assert spec.static_transform_xyz_rpy is not None
    if len(captures) < 2:
        raise ROSConfigError(f"{len(captures)} pose(s) recorded; a fit needs several")
    manifest = xyzrpy_to_matrix(spec.static_transform_xyz_rpy)
    poses = _prepare(robot, spec, captures)
    mount, picks = _icp(poses, manifest, _GATES_M)
    base_dist = _distances(poses, mount, picks)
    median = float(np.median(base_dist))
    ref = poses[0].parent
    # Observability: restart one limit off along each axis; a visible axis pulls back.
    unrecovered = {}
    for axis, metric in FIT_AXES.items():
        if progress:
            progress(f"recovery along {axis}")
        back, _ = _icp(poses, _perturbed(mount, ref, axis), _RECOVERY_GATES_M)
        unrecovered[axis] = mount_error(mount, back, ref)[metric] / ERROR_LIMITS[metric]
    spread = dict.fromkeys(ERROR_LIMITS, 0.0)
    for i in range(len(poses)):
        if progress:
            progress(f"held-out refit {i + 1}/{len(poses)}")
        rest = [p for j, p in enumerate(poses) if j != i]
        held, _ = _icp(rest, mount, (_HELDOUT_GATE_M,))
        for key, value in mount_error(mount, held, ref).items():
            spread[key] = max(spread[key], value)
    residuals = {
        "poses": len(poses),
        "points": int(sum(int(k.sum()) for k in picks)),
        "per_pose": [
            {
                "name": p.name,
                "points": int(k.sum()),
                "median_residual_m": float(
                    np.median(p.tree.query(_apply(p.parent @ mount, p.points[k]))[0])
                )
                if k.any()
                else None,
            }
            for p, k in zip(poses, picks, strict=True)
        ],
        "median_residual_m": median,
        "mount_error": mount_error(manifest, mount, ref),
        "heldout_spread": spread,
        "axis_unrecovered": unrecovered,
    }
    return mount, residuals
