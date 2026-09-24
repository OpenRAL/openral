"""Every collision-mesh vertex lies inside its link's collision primitive, at any pose.

Hazard-log Entry 045: the OpenArm's hand-written primitives were smaller than the
links they stood for (the finger meshes reached 83.7 mm outside the finger sphere),
so the kernel's self- and world-collision checks were not conservative. The
geometry is now fitted to the MJCF meshes (`openral collision lower
--fit-mjcf-geometry`); this pins the property that fit exists to guarantee.

The two sides are placed independently, which is the point:

- mesh vertices by MuJoCo's own forward kinematics, with every equality-coupled
  follower joint (the OpenArm's second finger) set from its leader, exactly as
  the simulator moves it;
- primitives by the SAFETY KERNEL's collision model
  (`collision_params_from_description`, the parameters the deploy launch hands
  the kernel) and the kernel's joint convention.

The two root frames are related by one fixed transform, fitted once from link
origins at q = 0; its residual is asserted, so a kinematic disagreement between
the manifest and the MJCF fails here rather than hiding in the fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the MJCF meshes")
pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")

from openral_core import RobotDescription  # noqa: E402
from openral_core.assets import resolve_asset  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# Robots whose collision geometry is fitted to MJCF meshes. A robot joins by being
# re-lowered with `--fit-mjcf-geometry`.
FITTED = ["openarm"]
N_POSES = 300
SEED = 20260924
# Nine stroke samples bound a follower finger between samples; this is the most
# any vertex may sit outside at a pose between them.
TOLERANCE_M = 1e-3


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _tf(xyzrpy: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rpy(*xyzrpy[3:])
    t[:3, 3] = xyzrpy[:3]
    return t


def _axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = np.cos(q), np.sin(q), 1 - np.cos(q)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _kernel_fk(params: dict, q: np.ndarray) -> list[np.ndarray]:
    """The kernel's `forward_kinematics`: origin · joint motion, parent before child."""
    n = int(params["collision_n_links"])
    origin = np.asarray(params["collision_origin_xyzrpy"], dtype=float).reshape(n, 6)
    axis = np.asarray(params["collision_axis"], dtype=float).reshape(n, 3)
    world: list[np.ndarray] = []
    for i in range(n):
        motion = np.eye(4)
        dof = int(params["collision_dof_index"][i])
        if dof >= 0:
            kind = int(params["collision_joint_kind"][i])
            if kind == 1:
                motion[:3, :3] = _axis_angle(axis[i], q[dof])
            elif kind == 2:
                motion[:3, 3] = axis[i] * q[dof]
        local = _tf(origin[i]) @ motion
        parent = int(params["collision_parent"][i])
        world.append(local if parent < 0 else world[parent] @ local)
    return world


def _mesh_vertices_world(model: object, data: object, body: int) -> np.ndarray | None:
    parts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body or int(model.geom_type[g]) != int(
            mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        m = int(model.geom_dataid[g])
        v = model.mesh_vert[
            int(model.mesh_vertadr[m]) : int(model.mesh_vertadr[m]) + int(model.mesh_vertnum[m])
        ]
        parts.append((data.geom_xmat[g].reshape(3, 3) @ v.T).T + data.geom_xpos[g])
    return np.vstack(parts) if parts else None


@pytest.mark.parametrize("robot_id", FITTED)
def test_every_mesh_vertex_is_inside_its_links_primitive(robot_id: str) -> None:
    from openral_safety.urdf_lowering import _mjcf_link_bodies, mjcf_coupled_joints

    manifest = REPO / "robots" / robot_id / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    params = collision_params_from_description(robot)
    names: list[str] = list(params["collision_link_names"])
    model = mujoco.MjModel.from_xml_path(
        str(resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest.parent))
    )
    data = mujoco.MjData(model)
    link_body = _mjcf_link_bodies(robot, model)

    # Which MJCF bodies each link's primitive has to hold: its own body, plus the
    # bodies of joints equality-coupled to the joint that drives it (followers).
    coupled = mjcf_coupled_joints(model)
    driven: dict[int, int] = {}  # MJCF joint the manifest drives -> its qpos address
    bodies_of: dict[str, list[int]] = {ln: [b] for ln, b in link_body.items()}
    for j in robot.joints:
        ji = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or ""))
        driven[ji] = int(model.jnt_qposadr[ji])
        for follower, _c0, _c1 in coupled.get(ji, []):
            bodies_of.setdefault(j.child_link, []).append(int(model.jnt_bodyid[follower]))

    # Manifest dof -> MJCF qpos address, in the kernel's dof order.
    dof_adr: list[int] = []
    limits: list[tuple[float, float]] = []
    for j in robot.joints:
        ji = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or ""))
        assert ji >= 0, f"{j.name}: sim_joint_name {j.sim_joint_name!r} not in the MJCF"
        dof_adr.append(int(model.jnt_qposadr[ji]))
        lo, hi = j.position_limits if j.position_limits else (-np.pi, np.pi)
        limits.append((float(lo), float(hi)))

    def pose(q: np.ndarray) -> list[np.ndarray]:
        mujoco.mj_resetData(model, data)
        for d, adr in enumerate(dof_adr):
            data.qpos[adr] = q[d]
        # Followers track the joint the manifest drives, never the other way round.
        for leader, adr in driven.items():
            for follower, c0, c1 in coupled.get(leader, []):
                if follower not in driven:
                    data.qpos[int(model.jnt_qposadr[follower])] = c0 + c1 * data.qpos[adr]
        mujoco.mj_kinematics(model, data)
        return _kernel_fk(params, q)

    # One fixed transform from the kernel's root frame to MuJoCo's world, fitted at q = 0.
    q0 = np.zeros(len(dof_adr))
    for d, (lo, hi) in enumerate(limits):
        q0[d] = min(max(0.0, lo), hi)
    kw = pose(q0)
    common = [ln for ln in names if ln in link_body]
    a = np.array([kw[names.index(ln)][:3, 3] for ln in common])
    b = np.array([data.xpos[link_body[ln]] for ln in common])
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rot = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    trans = b.mean(0) - rot @ a.mean(0)
    residual = np.linalg.norm((rot @ a.T).T + trans - b, axis=1).max()
    assert residual < 1e-6, f"manifest and MJCF kinematics disagree at q=0 by {residual:.2e} m"

    # Every primitive the kernel gets: capsules/spheres (half-extents None) and
    # boxes (radius/half-length None), each with its link-frame origin.
    prims: list[tuple[int, float | None, float | None, np.ndarray | None, np.ndarray]] = [
        (int(link), float(r), float(h), None, o)
        for link, r, h, o in zip(
            params["collision_capsule_link"],
            params["collision_capsule_radius"],
            params["collision_capsule_half_length"],
            np.asarray(params["collision_capsule_origin_xyzrpy"], dtype=float).reshape(-1, 6),
            strict=True,
        )
    ]
    box_links = params.get("collision_box_link", [])
    box_he = np.asarray(params.get("collision_box_half_extents", []), dtype=float).reshape(-1, 3)
    box_o = np.asarray(params.get("collision_box_origin_xyzrpy", []), dtype=float).reshape(-1, 6)
    prims += [(int(link), None, None, box_he[i], box_o[i]) for i, link in enumerate(box_links)]
    covered = {names[p[0]] for p in prims}
    uncovered = [
        ln
        for ln, bs in bodies_of.items()
        if ln not in covered and any(_mesh_vertices_world(model, data, bb) is not None for bb in bs)
    ]
    assert not uncovered, f"links with collision meshes but no primitive: {uncovered}"

    rng = np.random.default_rng(SEED)
    worst: dict[str, float] = {}
    for sample in range(N_POSES):
        q = q0 if sample == 0 else np.array([rng.uniform(lo, hi) for lo, hi in limits])
        kw = pose(q)
        for link, radius, half, half_extents, origin in prims:
            ln = names[link]
            verts = [_mesh_vertices_world(model, data, bb) for bb in bodies_of.get(ln, [])]
            verts = [v for v in verts if v is not None]
            if not verts:
                continue
            pts = np.vstack(verts)
            prim = kw[link] @ _tf(origin)
            prim_rot = rot @ prim[:3, :3]
            c = rot @ prim[:3, 3] + trans
            local = (pts - c) @ prim_rot  # vertices in the primitive's own frame
            if half_extents is not None:
                # Outside distance of a box; 0 anywhere inside it.
                over = np.maximum(np.abs(local) - half_extents, 0.0)
                outside = float(np.linalg.norm(over, axis=1).max())
            else:
                assert radius is not None and half is not None
                z = np.clip(local[:, 2], -half, half)
                axial = local - np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
                outside = float((np.linalg.norm(axial, axis=1) - radius).max())
            worst[ln] = max(worst.get(ln, -np.inf), outside)
    bad = {ln: round(d * 1000, 2) for ln, d in worst.items() if d > TOLERANCE_M}
    assert not bad, f"{robot_id}: mesh vertices outside their primitive (mm): {bad}"
