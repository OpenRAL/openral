"""Contact sweep: the kernel must trip on every configuration where the real links touch.

For each URDF-lowered robot, random configurations inside the manifest's joint
limits are checked two independent ways:

- **the kernel's verdict** — the exact parameters the deploy launch hands the C++
  kernel (``collision_params_from_description``), placed by the kernel's own
  forward-kinematics convention and measured with the kernel's own predicates
  (``kernel_predicates.shape_distance``, folded over each link's primitives and
  re-asked of the exact hulls for a refined box pair, as ``check_self_collision``
  does — ``_collision_kernel_verdict.kernel_link_gap``), tripping at
  ``d <= margin`` on every pair the ACM does not exempt;
- **ground truth** — each link's ``<collision>`` + ``<visual>`` geometry placed
  by the URDF's own forward kinematics (yourdfpy). A pair is *clear* only when a
  separating plane between the two convex hulls is proven (GJK witness +
  separating-axis bound, or an exact LP when GJK does not certify); otherwise it
  is a contact. Hull overlap is a superset of mesh overlap, so a false negative
  counted here bounds the mesh-level count from above.

A false negative — truth in contact, kernel clear, pair not exempted — fails the
test. False positives are the price of a bounding primitive and are only
reported (``docs/reference/collision-geometry-review.md`` has the 3000-pose
numbers per robot). Real fixtures only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("trimesh")
pytest.importorskip("robot_descriptions")
pytest.importorskip("scipy")

import trimesh
from openral_core import RobotDescription
from openral_core.assets import resolve_asset
from openral_core.schemas import LinkCollisionGeometry
from openral_hal.convex_distance import _gjk, separating_axis_bound
from openral_safety.envelope_loader import collision_params_from_description
from openral_safety.urdf_lowering import (
    _collision_local_mesh,
    _load_urdf,
    _origin_matrix,
)

from tests.unit._collision_kernel_verdict import kernel_link_gap, surface_pieces

_REPO = Path(__file__).resolve().parents[2]
URDF_LOWERED = (
    "droid_franka_robotiq",
    "franka_panda",
    "g1",
    "h1",
    "rizon4",
    "so100_follower",
    "ur10e",
    "ur5e",
)
N_POSES = 120
SEED = 20260924


def _rpy(r: float, p: float, y: float) -> npt.NDArray[np.float64]:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot: npt.NDArray[np.float64] = rz @ ry @ rx
    return rot


def _tf(xyzrpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    t = np.eye(4)
    t[:3, :3] = _rpy(*xyzrpy[3:])
    t[:3, 3] = xyzrpy[:3]
    return t


def _axis_angle(axis: npt.NDArray[np.float64], q: float) -> npt.NDArray[np.float64]:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rot: npt.NDArray[np.float64] = np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)
    return rot


def _kernel_fk(params: dict[str, Any], q: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
    """The kernel's ``forward_kinematics``: origin · joint motion, parent before child."""
    n = int(params["collision_n_links"])
    origin = np.asarray(params["collision_origin_xyzrpy"], dtype=float).reshape(n, 6)
    axis = np.asarray(params["collision_axis"], dtype=float).reshape(n, 3)
    world: list[npt.NDArray[np.float64]] = []
    mult = params.get("collision_mimic_multiplier", [1.0] * n)
    off = params.get("collision_mimic_offset", [0.0] * n)
    for i in range(n):
        motion = np.eye(4)
        dof = int(params["collision_dof_index"][i])
        kind = int(params["collision_joint_kind"][i])
        # A mimic kind (3/4) reads its leader's column: multiplier * q + offset.
        qi = float(mult[i]) * q[dof] + float(off[i]) if dof >= 0 and kind in (3, 4) else q[dof]
        if dof >= 0 and kind in (1, 3):
            motion[:3, :3] = _axis_angle(axis[i] / np.linalg.norm(axis[i]), qi)
        elif dof >= 0 and kind in (2, 4):
            motion[:3, 3] = axis[i] * qi
        local = _tf(origin[i]) @ motion
        parent = int(params["collision_parent"][i])
        world.append(local if parent < 0 else world[parent] @ local)
    return world


def _hulls_overlap(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    """Exact: do conv(a) and conv(b) intersect? (GJK proof when separated, else an LP.)"""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ra = np.linalg.norm(a - ca, axis=1).max()
    rb = np.linalg.norm(b - cb, axis=1).max()
    if np.linalg.norm(ca - cb) > ra + rb:
        return False
    d, pa, pb = _gjk(a, b)
    if d > 0.0 and pa is not None and separating_axis_bound(a, b, pb - pa) > 0.0:
        return False
    na, nb = len(a), len(b)
    eq = np.zeros((5, na + nb))
    eq[:3, :na], eq[:3, na:], eq[3, :na], eq[4, na:] = a.T, -b.T, 1.0, 1.0
    from scipy import optimize

    res = optimize.linprog(
        np.zeros(na + nb), A_eq=eq, b_eq=[0, 0, 0, 1, 1], bounds=(0, None), method="highs"
    )
    return bool(res.status == 0)


@pytest.mark.parametrize("robot_id", URDF_LOWERED)
def test_kernel_trips_on_every_sampled_real_contact(robot_id: str) -> None:
    manifest = _REPO / "robots" / robot_id / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    params = collision_params_from_description(robot)
    names: list[str] = list(params["collision_link_names"])
    margin = float(params["self_collision_margin_m"])
    urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest.parent)  # type: ignore[union-attr]  # reason: every URDF_LOWERED robot declares assets.urdf
    model = _load_urdf(str(urdf))
    handler = model._filename_handler

    # Every primitive the kernel gets, by link index (a link may carry several).
    prims: dict[int, list[LinkCollisionGeometry]] = {}
    for g in robot.collision_geometry:
        prims.setdefault(names.index(g.link_name), []).append(g)
    allowed_flat = [int(v) for v in params["collision_allowed_pairs"]]
    allowed = {frozenset(allowed_flat[i : i + 2]) for i in range(0, len(allowed_flat), 2)}

    # Ground truth (link frame) for every link that carries a primitive: its
    # convex hull, or — for a link with several primitives, whose union is not
    # convex — the hulls of its surface split among them (`surface_pieces`),
    # which contain the surface without filling the concavities a capsule chain
    # leaves out (the link's hull there would report contacts no mesh makes).
    hull: dict[int, list[npt.NDArray[np.float64]]] = {}
    for li in sorted(prims):
        link = model.link_map[names[li]]
        mesh = trimesh.util.concatenate(
            [
                _collision_local_mesh(el, handler).apply_transform(_origin_matrix(el.origin))
                for el in list(link.collisions or []) + list(link.visuals or [])
            ]
        )
        if len(prims[li]) == 1:
            hull[li] = [np.asarray(trimesh.convex.convex_hull(mesh.vertices).vertices)]
            continue
        pieces, uncovered = surface_pieces(
            np.asarray(mesh.vertices), np.asarray(mesh.faces), prims[li]
        )
        assert uncovered == 0, f"{robot_id}/{names[li]}: surface leaves its primitives"
        hull[li] = [
            np.asarray(trimesh.convex.convex_hull(p).vertices) for p in pieces if len(p) >= 4
        ]

    child_to_urdf = {j.child: j.name for j in model.robot.joints}
    actuated = set(model.actuated_joint_names)
    lo = np.array([j.position_limits[0] if j.position_limits else -np.pi for j in robot.joints])
    hi = np.array([j.position_limits[1] if j.position_limits else np.pi for j in robot.joints])
    root = names[int(np.flatnonzero(np.asarray(params["collision_parent"]) < 0)[0])]
    pairs = [
        (a, b)
        for a in sorted(hull)
        for b in sorted(hull)
        if a < b and frozenset((a, b)) not in allowed
    ]
    rng = np.random.default_rng(SEED)
    contacts, misses, fk_err = 0, [], 0.0
    for _ in range(N_POSES):
        q = rng.uniform(lo, hi)
        kernel_world = _kernel_fk(params, q)
        cfg = {
            child_to_urdf[j.child_link]: float(v)
            for j, v in zip(robot.joints, q, strict=True)
            if child_to_urdf.get(j.child_link) in actuated
        }
        # A mimic row whose child is driven by an actuated URDF joint (the DROID's
        # Robotiq leader `finger_joint` under the normalised jaw) sets that joint
        # from the manifest relation; the URDF's own <mimic> rows pose the rest.
        leader_q = {j.name: float(v) for j, v in zip(robot.joints, q, strict=True)}
        for mim in robot.mimic_joints:
            if child_to_urdf.get(mim.child_link) in actuated:
                cfg[child_to_urdf[mim.child_link]] = mim.value_at(leader_q[mim.mimics])
        model.update_cfg(cfg)
        base = np.linalg.inv(model.get_transform(root))
        truth = {}
        for li, hs in hull.items():
            t = base @ model.get_transform(names[li])
            fk_err = max(fk_err, float(np.abs(t - kernel_world[li]).max()))
            truth[li] = [h @ t[:3, :3].T + t[:3, 3] for h in hs]
        for a, b in pairs:
            if not any(_hulls_overlap(ha, hb) for ha in truth[a] for hb in truth[b]):
                continue
            contacts += 1
            # Hull-refined where the kernel refines: a refined box pair is judged
            # by its hulls, which is exactly where a miss would hide.
            gap = kernel_link_gap(prims[a], kernel_world[a], prims[b], kernel_world[b], margin)
            if gap > margin:
                misses.append((names[a], names[b], round(gap, 4), np.round(q, 4).tolist()))
    assert fk_err < 1e-6, f"{robot_id}: manifest FK disagrees with the URDF by {fk_err:.2e} m"
    assert not misses, (
        f"{robot_id}: {len(misses)} of {contacts} sampled real contacts the kernel does not "
        f"trip on (pair, kernel gap m, q): {misses[:5]}"
    )
