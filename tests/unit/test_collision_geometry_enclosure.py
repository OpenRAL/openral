"""Every MJCF-twin mesh vertex lies inside its link's primitive set, at any pose.

Hazard-log Entries 045/046: the hand-written OpenArm primitives were smaller than
the links they stood for (the finger meshes reached 83.7 mm outside the finger
sphere), and the URDF-lowered capsules under-covered H1's torso by 549 mm. The
geometry is now fitted to each link's whole geometry (collision and visual) by
one fitter for both lowering paths; this pins the property that fit exists to
guarantee, on every robot whose manifest declares an MJCF twin.

The two sides are placed independently, which is the point:

- mesh vertices (every geom of the twin body, collision and visual) by MuJoCo's
  own forward kinematics, with every equality-coupled follower joint (the
  OpenArm's second finger) set from its leader, exactly as the simulator moves it;
- primitives by the SAFETY KERNEL's collision model
  (`collision_params_from_description`, the parameters the deploy launch hands
  the kernel) and the kernel's joint convention, folded per link as the kernel
  folds them (a link may carry several).

The two root frames are related by one fixed transform, fitted once from link
origins at q = 0; its residual is asserted, so a kinematic disagreement between
the manifest and the MJCF fails here rather than hiding in the fit. A twin whose
body names match none of the manifest's links (SO-100's menagerie twin) is
skipped with the unmapped links named; `test_collision_geometry_enclosure_urdf.py`
covers it from the URDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the MJCF meshes")
pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
pytest.importorskip("trimesh")

from openral_core import RobotDescription  # noqa: E402
from openral_core.assets import AssetRefError, resolve_asset  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

from tests.unit._collision_kernel_verdict import tight_escape_m  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# Every manifest with collision geometry and an MJCF twin.
TWINNED = sorted(
    p.parent.name
    for p in REPO.glob("robots/*/robot.yaml")
    if "collision_geometry:" in p.read_text() and "\n  mjcf:" in p.read_text()
)
N_POSES = 300
# A URDF-lowered robot's twin links are rigid (no follower sweep), so extra poses
# only re-test the two FKs' agreement; fewer keep the CAD-heavy twins affordable.
N_POSES_RIGID = 40
SEED = 20260924
# Nine stroke samples bound a follower finger between samples; this is the most
# any vertex of a swept link may sit outside at a pose between them.
SWEPT_TOLERANCE_M = 1e-3
# Every other link: each vertex at least half the declared 1 mm headroom inside
# (the 4-dp rendering moves a primitive by ~0.1 mm at most), as the URDF test.
HEADROOM_FLOOR_M = 0.5e-3
# The manifest-vs-MJCF kinematic residual this test asserts at q = 0.
FK_RESIDUAL_M = 1e-6
# A refined box's DOP and hull are exact (no headroom), so the two independent
# FKs' disagreement away from q = 0 shows up directly: the manifest writes joint
# origins and axes at 4 dp, which moves H1's elbow by ~1 um at random poses.
HULL_TOLERANCE_M = 1e-5


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
    x, y, z = axis / np.linalg.norm(axis)
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


_GEOM_POINTS: dict[tuple[int, int, bool], np.ndarray] = {}


def _geom_points(model: object, geom: int, *, hull_only: bool) -> np.ndarray:
    """A geom's vertices in its own frame (or just their convex hull's), cached per model."""
    import trimesh
    from openral_safety.urdf_lowering import _geom_local_mesh

    key = (id(model), geom, hull_only)
    if key not in _GEOM_POINTS:
        v = np.unique(np.asarray(_geom_local_mesh(model, geom).vertices, dtype=float), axis=0)
        if hull_only and len(v) > 4:
            v = np.asarray(trimesh.convex.convex_hull(v).vertices, dtype=float)
        _GEOM_POINTS[key] = v
    return _GEOM_POINTS[key]


def _body_vertices_world(
    model: object, data: object, body: int, *, meshes_only: bool = False, hull_only: bool = False
) -> np.ndarray | None:
    """Every geom's bounding vertices of `body` (collision and visual), in the world frame.

    ``meshes_only`` keeps the mesh geoms (the CAD) and drops MuJoCo primitive
    geoms, which on a URDF-lowered robot's twin are the simulator's contact
    proxies, not the robot (H1's arm capsules reach 145 mm past its CAD).
    ``hull_only`` keeps each geom's convex-hull vertices: exact for a link with
    ONE convex primitive (it holds a geom iff it holds the geom's hull), and
    what keeps 300 poses of CAD meshes affordable.
    """
    parts = []
    for g in range(model.ngeom):  # type: ignore[attr-defined]  # reason: MjModel
        if int(model.geom_bodyid[g]) != body:  # type: ignore[attr-defined]  # reason: MjModel
            continue
        if meshes_only and int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):  # type: ignore[attr-defined]  # reason: MjModel
            continue
        v = _geom_points(model, g, hull_only=hull_only)
        parts.append((data.geom_xmat[g].reshape(3, 3) @ v.T).T + data.geom_xpos[g])  # type: ignore[attr-defined]  # reason: MjData
    return np.vstack(parts) if parts else None


def _signed_distance(local: np.ndarray, prim: tuple) -> np.ndarray:
    """Signed distance of primitive-frame points to one primitive (negative inside)."""
    _link, radius, half, half_extents, _origin = prim
    if half_extents is not None:
        q = np.abs(local) - half_extents
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    z = np.clip(local[:, 2], -half, half)
    axial = local - np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return np.linalg.norm(axial, axis=1) - radius


# Twins whose manifest leaves a twin body with geometry uncovered. Strict xfails
# so covering them flips green (docs/reference/collision-geometry-review.md §12).
_UNCOVERED_TWIN_BODIES: dict[str, str] = {
    # The manifest's synthetic `panda_finger_pair` (prismatic off the hand) has
    # no URDF link, so no primitive: the twin's fingers (~50 mm past the hand)
    # are not in the kernel's model. Pre-existing; needs the MJCF finger sweep
    # the OpenArm uses, on a URDF-lowered robot.
    "franka_panda": "panda_finger_pair (the fingers) carries no primitive",
    # Hand-authored manifest (no GENERATED header, not re-lowered here): its
    # `gripper_base` / `moving_jaw` bodies carry no primitive.
    "so101_follower": "gripper_base / moving_jaw carry no primitive (hand-authored manifest)",
}


@pytest.mark.parametrize(
    "robot_id",
    [
        pytest.param(
            r,
            marks=(
                pytest.mark.xfail(strict=True, reason=_UNCOVERED_TWIN_BODIES[r])
                if r in _UNCOVERED_TWIN_BODIES
                else ()
            ),
        )
        for r in TWINNED
    ],
)
def test_every_twin_mesh_vertex_is_inside_its_links_primitives(robot_id: str) -> None:
    from openral_safety.urdf_lowering import (
        _mjcf_link_bodies,
        mjcf_coupled_joints,
        select_lowering,
    )

    manifest = REPO / "robots" / robot_id / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    params = collision_params_from_description(robot)
    names: list[str] = list(params["collision_link_names"])
    try:
        mjcf = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest.parent)
    except AssetRefError as exc:
        pytest.skip(f"{robot_id}: MJCF twin unavailable on this host: {exc}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    link_body = _mjcf_link_bodies(robot, model)
    # A robot lowered FROM its MJCF is held to every geom the fit used; a
    # URDF-lowered robot's twin to its meshes (the CAD), not its contact proxies.
    mjcf_lowered = select_lowering(robot, manifest_dir=manifest.parent) == "mjcf"
    declared = {g.link_name for g in robot.collision_geometry}
    unmapped = sorted(declared - set(link_body))
    if unmapped:
        pytest.skip(
            f"{robot_id}: the twin names no body for links {unmapped}; the URDF enclosure "
            "test covers them"
        )

    # Which MJCF bodies each link's primitives have to hold: its own body, plus
    # the bodies of joints equality-coupled to the joint that drives it (followers).
    coupled = mjcf_coupled_joints(model)
    driven: dict[int, int] = {}  # MJCF joint the manifest drives -> its qpos address
    bodies_of: dict[str, list[int]] = {ln: [b] for ln, b in link_body.items()}
    dof_adr: list[int] = []
    limits: list[tuple[float, float]] = []
    for j in robot.joints:
        ji = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or j.name))
        assert ji >= 0, f"{j.name}: neither sim_joint_name nor the joint name is in the MJCF"
        driven[ji] = int(model.jnt_qposadr[ji])
        for follower, _c0, _c1 in coupled.get(ji, []):
            bodies_of.setdefault(j.child_link, []).append(int(model.jnt_bodyid[follower]))
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
    # Links that carry primitives only: a synthetic manifest link (franka's
    # finger pair, mapped to the twin's left finger body) has no frame to agree on.
    common = [ln for ln in names if ln in link_body and ln in declared]
    a = np.array([kw[names.index(ln)][:3, 3] for ln in common])
    b = np.array([data.xpos[link_body[ln]] for ln in common])
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rot = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    trans = b.mean(0) - rot @ a.mean(0)
    residual = np.linalg.norm((rot @ a.T).T + trans - b, axis=1).max()
    if not mjcf_lowered and residual >= FK_RESIDUAL_M:
        # A twin in another frame convention (the menagerie UR arms put their
        # body origins off the URDF joint origins) says nothing about a URDF
        # fit; test_collision_geometry_enclosure_urdf.py checks that one.
        pytest.skip(
            f"{robot_id}: the twin's body frames sit {residual * 1e3:.1f} mm off the manifest's "
            "link frames at q = 0; the URDF enclosure test covers the geometry it was lowered from"
        )
    assert residual < FK_RESIDUAL_M, (
        f"manifest and MJCF kinematics disagree at q=0 by {residual:.2e} m"
    )

    # Every primitive the kernel gets, grouped by link: capsules/spheres
    # (half-extents None) and boxes (radius/half-length None), with origins.
    prims: dict[int, list[tuple]] = {}
    for link, r, h, o in zip(
        params.get("collision_capsule_link", []),
        params.get("collision_capsule_radius", []),
        params.get("collision_capsule_half_length", []),
        np.asarray(params.get("collision_capsule_origin_xyzrpy", []), dtype=float).reshape(-1, 6),
        strict=True,
    ):
        prims.setdefault(int(link), []).append((int(link), float(r), float(h), None, o))
    box_he = np.asarray(params.get("collision_box_half_extents", []), dtype=float).reshape(-1, 3)
    box_o = np.asarray(params.get("collision_box_origin_xyzrpy", []), dtype=float).reshape(-1, 6)
    for i, link in enumerate(params.get("collision_box_link", [])):
        prims.setdefault(int(link), []).append((int(link), None, None, box_he[i], box_o[i]))
    uncovered = [
        ln
        for ln, bs in bodies_of.items()
        if names.index(ln) not in prims
        and any(
            _body_vertices_world(model, data, bb, meshes_only=not mjcf_lowered) is not None
            for bb in bs
        )
    ]
    assert not uncovered, f"links with twin geometry but no primitive: {uncovered}"

    refined: dict[str, list] = {}
    for g in robot.collision_geometry:
        if g.tight_geometry is not None:
            refined.setdefault(g.link_name, []).append(g)
    rng = np.random.default_rng(SEED)
    worst: dict[str, float] = {}
    escape: dict[str, float] = {}
    for sample in range(N_POSES if mjcf_lowered else N_POSES_RIGID):
        q = q0 if sample == 0 else np.array([rng.uniform(lo, hi) for lo, hi in limits])
        kw = pose(q)
        for link, link_prims in prims.items():
            ln = names[link]
            # Every vertex at q = 0; the hull vertices of a one-primitive link
            # elsewhere (containment is rigid, so later poses test the two FKs).
            verts = [
                _body_vertices_world(
                    model,
                    data,
                    bb,
                    meshes_only=not mjcf_lowered,
                    hull_only=sample > 0 and len(link_prims) == 1,
                )
                for bb in bodies_of.get(ln, [])
            ]
            verts = [v for v in verts if v is not None]
            if not verts:
                continue
            pts = np.vstack(verts)
            outside = np.full(len(pts), np.inf)
            for prim in link_prims:
                placed = kw[link] @ _tf(prim[4])
                prim_rot = rot @ placed[:3, :3]
                c = rot @ placed[:3, 3] + trans
                outside = np.minimum(outside, _signed_distance((pts - c) @ prim_rot, prim))
            worst[ln] = max(worst.get(ln, -np.inf), float(outside.max()))
            for g in refined.get(ln, []):  # a refined box: its DOP and hull hold the mesh too
                placed = kw[link] @ _tf(np.asarray(g.origin_xyz_rpy, dtype=float))
                local = (pts - (rot @ placed[:3, 3] + trans)) @ (rot @ placed[:3, :3])
                escape[ln] = max(escape.get(ln, -np.inf), tight_escape_m(g, local))
    # A link that folds in an equality follower (the OpenArm's second finger) is
    # fitted to STROKE_SAMPLES of the stroke, so a pose between two samples may
    # poke out by up to SWEPT_TOLERANCE_M; every other link keeps its headroom.
    swept = {ln for ln, bs in bodies_of.items() if len(bs) > 1}
    bad_tight = {
        ln: round(d * 1000, 6)
        for ln, d in escape.items()
        if d > (SWEPT_TOLERANCE_M if ln in swept else HULL_TOLERANCE_M)
    }
    assert not bad_tight, f"{robot_id}: twin vertices outside tight_geometry (mm): {bad_tight}"
    bad = {
        ln: round(d * 1000, 2)
        for ln, d in worst.items()
        if d > (SWEPT_TOLERANCE_M if ln in swept else -HEADROOM_FLOOR_M)
    }
    assert not bad, f"{robot_id}: twin mesh vertices outside their primitives (mm): {bad}"
