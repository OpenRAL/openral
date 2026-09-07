#!/usr/bin/env python3
"""Derive a robot manifest's ``tight_geometry`` blocks from its real collision meshes.

The safety kernel's arm-link-vs-world-voxel check runs a staged 26-DOP → exact
convex hull narrow phase for links that declare ``tight_geometry`` (see
``docs/reference/collision-hull-narrow-phase.md``); this tool derives that
geometry, not fits it: the 26-DOP is the intersection of 26 tangent halfspaces
``u·x <= h_mesh(u)`` (mesh inside by construction, no optimiser tolerance) and
the hull is ``conv(mesh vertices)`` (containment definitional); both are in the
manifest box's own frame, and the tool refuses to emit anything outside that box.

Mesh placement follows ``docs/reference/collision-tight-geometry.md`` §11:
MuJoCo folds mesh recentring into the geom frame, so vertices are placed by the
GEOM transform only — applying ``mesh_pos`` too double-counts it (PR #158).

Usage::

    # print the YAML fragment to paste into robots/<name>/robot.yaml
    python tools/generate_tight_geometry.py emit --robot robots/panda_mobile/robot.yaml

    # re-derive from the mesh and verify the manifest still matches (CI / tests)
    python tools/generate_tight_geometry.py check --robot robots/panda_mobile/robot.yaml

``check`` exits 0 when every declared block is reproduced and contains its mesh,
and 3 otherwise — the same fail-closed exit ``openral collision lower --write``
uses when it refuses to loosen a shipped envelope.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

    Points = npt.NDArray[Any]

REPO_ROOT = Path(__file__).resolve().parent.parent

# The MuJoCo geom that carries each manifest link's collision mesh. Manifest
# link names are the URDF/TF names; robosuite's MJCF uses bare `linkN`.
PANDA_GEOM_OF_LINK = {f"panda_link{i}": f"link{i}_collision" for i in range(1, 8)}

_PANDA_XML = "robosuite/models/assets/robots/panda/robot.xml"

ROBOT_MESH_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "panda_mobile": (_PANDA_XML, PANDA_GEOM_OF_LINK),
    # Same arm, same links, same geometry -- and it must stay that way.
    # `tests/unit/test_collision_geometry_contracts.py` asserts the two
    # manifests' `panda_link*` entries are equal, so a mesh or fit change has to
    # land on both or fail.
    "panda_mobile_vslam": (_PANDA_XML, PANDA_GEOM_OF_LINK),
}


def _dop_axes() -> Points:
    import numpy as np
    from openral_core.schemas import DOP_AXES

    axes: Points = np.asarray(DOP_AXES, dtype=float)
    return axes


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Points:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` -- the kernel's ``transform_from_xyz_rpy``."""
    import numpy as np

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rot: Points = rz @ ry @ rx
    return rot


def _robosuite_asset_root() -> Path:
    import robosuite

    return Path(robosuite.__file__).resolve().parent.parent


def link_mesh_in_box_frame(
    xml_path: Path, geom_name: str, origin_xyz_rpy: tuple[float, ...]
) -> Points:
    """Collision-mesh vertices of ``geom_name``, expressed in the manifest box's frame."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        msg = f"geom {geom_name!r} not found in {xml_path}"
        raise ValueError(msg)
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        msg = f"geom {geom_name!r} is not a mesh geom"
        raise ValueError(msg)
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start : start + count].astype(float)
    # MuJoCo folds mesh recentring into the geom frame; applying both would
    # double-count it. Assert the premise rather than trusting it.
    if not np.allclose(model.mesh_pos[mesh_id], model.geom_pos[gid], atol=1e-9):
        msg = (
            f"{geom_name}: mesh_pos != geom_pos, so the geom-only placement this tool "
            "relies on does not hold for this asset"
        )
        raise ValueError(msg)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    in_link = verts @ rot.reshape(3, 3).T + model.geom_pos[gid]
    translation = np.asarray(origin_xyz_rpy[:3], dtype=float)
    rotation = _rpy_to_matrix(*origin_xyz_rpy[3:6])
    in_box: Points = (in_link - translation) @ rotation
    return in_box


def link_mesh_faces(xml_path: Path, geom_name: str) -> Points:
    """Triangle indices for ``geom_name``'s mesh, local to its own vertex block.

    Indexes the same vertex order :func:`link_mesh_in_box_frame` returns, so
    ``faces`` from this function and ``points`` from that one describe one
    consistent triangle mesh.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_faceadr[mesh_id]
    count = model.mesh_facenum[mesh_id]
    faces: Points = model.mesh_face[start : start + count].copy()
    return faces


#: Samples per `closest_point_naive` call. The query allocates
#: `samples x mesh-faces x 3` floats, so this bounds peak memory independently
#: of how large a link's mesh is.
_OVERHANG_BATCH = 512

#: Cap on total barycentric samples across all facets, so overhang cost is
#: bounded by mesh size rather than by hull complexity.
_OVERHANG_MAX_SAMPLES = 12_000


def hull_overhang_m(
    hull_points: Points, mesh_points: Points, mesh_faces: Points, *, samples_per_edge: int = 24
) -> float:
    """How far the hull's surface reaches past the real source mesh, in metres.

    Containment (``mesh ⊆ hull``) is exact and definitional. This is the other
    direction: the hull's faces bridge over mesh concavities, and the worst
    point can fall anywhere on a facet (not just at a hull vertex, which sits
    ON the mesh). So this samples a barycentric grid per hull facet and
    measures each sample's distance to the real mesh surface.

    Args:
        hull_points: The hull's own vertices, box-frame, the same array
            ``scipy.spatial.ConvexHull`` was built from.
        mesh_points: Real mesh vertices, box-frame (:func:`link_mesh_in_box_frame`).
        mesh_faces: Real mesh triangle indices into ``mesh_points``
            (:func:`link_mesh_faces`).
        samples_per_edge: Barycentric grid resolution per hull facet (24 gives
            325 samples/facet). A sampled lower bound on the true continuous
            supremum (measured max creeps up a couple % per doubling), which
            is why the caller pads it (see ``_emit``) rather than shipping it raw.

    Returns:
        The sampled maximum distance, in metres, with NO margin applied.
        ``_check`` re-samples independently, at a different resolution, to
        guard against a shipped number a denser grid would exceed.
    """
    import numpy as np
    import trimesh

    # reason: scipy ships no type stubs and scipy-stubs is not a workspace dependency.
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

    mesh = _trimesh(mesh_points, mesh_faces)
    hull = ConvexHull(hull_points)
    # Keep TOTAL samples bounded, not samples-per-facet: cost is
    # (facets x samples-per-facet) x mesh-faces, and a 320-vertex envelope has
    # ~636 facets vs ~300 for a 152-vertex hull, so a fixed `samples_per_edge`
    # makes the largest link minutes slower than the rest.
    #
    # Coarsening only lowers the sampled maximum (a LOWER bound on the true
    # supremum), and `_check` fails when the declared overhang is *below* a
    # fresh resample — so a coarser resample can only make that gate more
    # permissive, never pass a manifest that is actually wrong.
    n = samples_per_edge
    while n > 4 and len(hull.simplices) * (n + 1) * (n + 2) // 2 > _OVERHANG_MAX_SAMPLES:
        n -= 1
    bary = np.array(
        [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    )
    tris = hull_points[hull.simplices]  # (n_facets, 3, 3)
    samples = np.einsum("fvc,sv->fsc", tris, bary).reshape(-1, 3)
    # `closest_point` (`mesh.nearest`) needs `rtree`, an optional trimesh dep
    # this workspace does not pin, so query naive brute-force in batches: the
    # largest panda link is ~207k samples x ~12k triangles = 57.8 GiB in one
    # call (raised `numpy._core._exceptions._ArrayMemoryError`). Batching bounds
    # peak memory at `_OVERHANG_BATCH x faces x 3` regardless of link size.
    worst = 0.0
    for start in range(0, len(samples), _OVERHANG_BATCH):
        batch = samples[start : start + _OVERHANG_BATCH]
        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, batch)
        worst = max(worst, float(distances.max()))
    return worst


def _trimesh(points: Points, faces: Points) -> Any:
    import trimesh

    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


def refine_dop_to_budget(points: Points, dop_lo: Points, dop_hi: Points, budget: int) -> Points:
    """A ≤``budget``-vertex convex envelope strictly tighter than the 26-DOP.

    ``panda_link1``: exact hull is 1588 vertices vs a 320-vertex kernel budget,
    so it ships stage-1 only (26-DOP: median 4.52 mm / max 25.68 mm support gap
    vs the real mesh) — "over budget" need not mean "fall back to the DOP".
    Not worth it for ``link1`` on the evidence: a live battery on 2026-09-07
    moved the refined envelope's predicted stops by only 0.0003 mm (deficit was
    against the box, already collected by the DOP — see
    ``docs/reference/collision-validation-evidence.md``), so no manifest ships
    a refined envelope without a fresh measurement.

    Construction: greedy halfspace refinement, not hull-vertex subsetting,
    because containment must stay definitional. Every candidate plane is a
    face of ``conv(mesh)`` (tangent, can never cut the mesh); the starting
    polytope is the DOP so the result is ``⊆ DOP`` by construction (what
    ``TightCollisionGeometry`` requires; ``link1``'s DOP has only 0.083 mm of
    room inside its manifest box); planes are added worst-violation-first and
    skipped if they'd push the vertex count over budget. So
    ``mesh ⊆ result ⊆ DOP ⊆ box`` holds throughout (CLAUDE.md §3). For
    ``link1``: 162 planes, 0.18 mm median / 0.65 mm max support gap.

    Args:
        points: every mesh vertex, in the manifest box's frame.
        dop_lo: per-axis minima already computed for the 26-DOP.
        dop_hi: per-axis maxima already computed for the 26-DOP.
        budget: the kernel's ``kMaxTightHullVertices``.

    Returns:
        The refined polytope's vertices, at most ``budget`` of them.
    """
    import numpy as np
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    axes = _dop_axes()
    halfspaces = np.vstack(
        [
            np.hstack([axes, -np.asarray(dop_hi)[:, None]]),
            np.hstack([-axes, np.asarray(dop_lo)[:, None]]),
        ]
    )
    candidates = ConvexHull(points).equations
    interior = points.mean(axis=0)

    def vertices_of(planes: Points) -> Points:
        found: Points = HalfspaceIntersection(planes, interior).intersections
        return found

    used = np.zeros(len(candidates), dtype=bool)
    while True:
        current = vertices_of(halfspaces)
        if len(current) > budget:
            break
        # Score each unused tangent plane by how far the current polytope pokes
        # past it: the plane that trims the most is the one worth spending
        # vertices on.
        overshoot = current @ candidates[:, :3].T + candidates[:, 3]
        score = overshoot.max(axis=0)
        score[used] = -np.inf
        best = int(np.argmax(score))
        if score[best] <= 1e-9:
            break  # nothing left to trim; this IS the hull within tolerance
        used[best] = True
        trial = np.vstack([halfspaces, candidates[best]])
        if len(vertices_of(trial)) > budget:
            continue  # would overrun the budget; try the next-worst plane
        halfspaces = trial

    refined = vertices_of(halfspaces)
    # Refuse rather than emit an envelope that does not contain its mesh.
    residual = float(
        (
            points @ ConvexHull(refined).equations[:, :3].T + ConvexHull(refined).equations[:, 3]
        ).max()
    )
    if residual > 1e-9:
        msg = (
            f"refined envelope cuts the mesh by {residual * 1e3:.9f} mm; refusing to "
            "emit an envelope smaller than the geometry it must contain"
        )
        raise ValueError(msg)
    return refined


def derive_tight_geometry(points: Points, half_extents: tuple[float, ...]) -> dict[str, Any]:
    """Build the DOP slabs and (when it fits the budget) the exact hull.

    Returns a mapping ready for :class:`openral_core.schemas.TightCollisionGeometry`,
    plus the diagnostics a reviewer needs: vertex counts and the achieved inward
    margin of the DOP inside the shipped box.
    """
    import numpy as np
    from openral_core.schemas import MAX_TIGHT_HULL_VERTICES

    # mypy only requires the `import-untyped` ignore on this module's FIRST
    # import site in the file (hull_overhang_m, above); a second one here is
    # flagged as unused.
    from scipy.spatial import ConvexHull

    axes = _dop_axes()
    proj = points @ axes.T
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)

    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    exact_count = int(len(hull_vertices))
    fits_budget = exact_count <= MAX_TIGHT_HULL_VERTICES
    decimated = False
    if not fits_budget:
        # Over budget is not a reason to fall back to the DOP — it is a reason to
        # refine the DOP toward the hull. See `refine_dop_to_budget`.
        hull_vertices = refine_dop_to_budget(points, lo, hi, MAX_TIGHT_HULL_VERTICES)
        fits_budget = True
        decimated = True

    he = np.asarray(half_extents, dtype=float)
    inward = float(np.minimum(he - hi[:3], he + lo[:3]).min())
    return {
        "dop_lo_m": [float(v) for v in lo],
        "dop_hi_m": [float(v) for v in hi],
        "hull_vertices_m": ([[float(c) for c in v] for v in hull_vertices] if fits_budget else []),
        "_hull_vertex_count": int(len(hull_vertices)),
        "_exact_hull_vertex_count": exact_count,
        "_decimated": decimated,
        "_stage2": bool(fits_budget),
        "_dop_inward_margin_m": inward,
    }


def _load_robot(path: Path) -> Any:
    from openral_core.schemas import RobotDescription

    return RobotDescription.from_yaml(str(path))


def _sources_for(robot_name: str) -> tuple[Path, dict[str, str]]:
    if robot_name not in ROBOT_MESH_SOURCES:
        msg = (
            f"no mesh source registered for robot {robot_name!r}; add one to "
            "ROBOT_MESH_SOURCES with the MJCF path and the per-link geom names"
        )
        raise KeyError(msg)
    rel, geoms = ROBOT_MESH_SOURCES[robot_name]
    return _robosuite_asset_root() / rel, geoms


def _emit(robot_path: Path) -> int:
    robot = _load_robot(robot_path)
    xml, geoms = _sources_for(robot.name)
    print(f"# tight_geometry for {robot.name}, derived from {xml}")
    print("# GENERATED by tools/generate_tight_geometry.py -- do not hand-edit.")
    for geom in robot.collision_geometry:
        geom_name = geoms.get(geom.link_name)
        if geom_name is None or geom.shape.shape != "box":
            continue
        pts = link_mesh_in_box_frame(xml, geom_name, geom.origin_xyz_rpy)
        derived = derive_tight_geometry(pts, geom.shape.half_extents_m)
        overhang = None
        if derived["hull_vertices_m"]:
            import numpy as np

            faces = link_mesh_faces(xml, geom_name)
            hull_pts = np.asarray(derived["hull_vertices_m"], dtype=float)
            sampled = hull_overhang_m(hull_pts, pts, faces)
            overhang = _round_up_m(sampled * _HULL_OVERHANG_SAFETY_MARGIN)
        print(f"\n# --- {geom.link_name} ---")
        print(
            f"#   mesh vertices {len(pts)}, hull vertices {derived['_hull_vertex_count']}, "
            f"stage 2 {'on' if derived['_stage2'] else 'OFF (over the vertex budget)'}, "
            f"DOP inward margin {derived['_dop_inward_margin_m'] * 1e3:.4f} mm"
            + (f", hull overhang {overhang * 1e3:.4f} mm" if overhang is not None else "")
        )
        print("    tight_geometry:")
        print("      dop_lo_m: [" + ", ".join(repr(v) for v in derived["dop_lo_m"]) + "]")
        print("      dop_hi_m: [" + ", ".join(repr(v) for v in derived["dop_hi_m"]) + "]")
        if derived["hull_vertices_m"]:
            print("      hull_vertices_m:")
            for v in derived["hull_vertices_m"]:
                print("        - [" + ", ".join(repr(c) for c in v) + "]")
            print(f"      hull_overhang_m: {overhang!r}")
        else:
            print("      hull_vertices_m: []")
    return 0


# The sampled max keeps creeping up a couple of percent per grid doubling
# (measured up to samples_per_edge=32 on every panda link with a stage-2
# hull); this pads the shipped number well past that residual drift instead
# of chasing convergence with an ever-finer, ever-slower grid.
_HULL_OVERHANG_SAFETY_MARGIN = 1.2


def _round_up_m(value: float, precision_m: float = 1e-6) -> float:
    """Round a sampled distance up to the next ``precision_m``, never down.

    The sampled maximum is a lower bound on the true continuous supremum; this
    keeps the shipped number from ever quietly under-stating it by less than a
    micron of rounding.
    """
    return math.ceil(value / precision_m) * precision_m


def _check(robot_path: Path) -> int:
    import numpy as np

    robot = _load_robot(robot_path)
    xml, geoms = _sources_for(robot.name)
    axes = _dop_axes()
    failures: list[str] = []
    checked = 0
    for geom in robot.collision_geometry:
        if geom.tight_geometry is None:
            continue
        geom_name = geoms.get(geom.link_name)
        if geom_name is None:
            failures.append(f"{geom.link_name}: declares tight_geometry but has no mesh source")
            continue
        checked += 1
        pts = link_mesh_in_box_frame(xml, geom_name, geom.origin_xyz_rpy)
        tight = geom.tight_geometry
        # The obligation the kernel cannot check: the true mesh is inside the DOP.
        proj = pts @ axes.T
        over = float((proj - np.asarray(tight.dop_hi_m)).max())
        under = float((np.asarray(tight.dop_lo_m) - proj).max())
        worst = max(over, under)
        if worst > 1e-9:
            failures.append(
                f"{geom.link_name}: mesh escapes its declared DOP by {worst * 1e3:.6f} mm"
            )
        if tight.hull_vertices_m:
            hull_pts = np.asarray(tight.hull_vertices_m, dtype=float)
            # And the mesh is inside the declared hull -- checked against every
            # facet, so a stale or truncated vertex list cannot pass.
            from scipy.spatial import ConvexHull

            hull = ConvexHull(hull_pts)
            normals = hull.equations[:, :3]
            offsets = hull.equations[:, 3]
            slack = float(((pts @ normals.T + offsets) / np.linalg.norm(normals, axis=1)).max())
            if slack > 1e-9:
                failures.append(
                    f"{geom.link_name}: mesh escapes its declared hull by {slack * 1e3:.6f} mm"
                )
            # The other direction: re-sample the hull's overhang past the real
            # mesh, at a FINER grid than generation used, and refuse a shipped
            # number a denser sample would exceed. A generator checking its own
            # output at the same resolution would just reproduce it -- the
            # margin in `_emit` is what makes this independent re-sample a
            # real check rather than a coin flip against grid placement.
            faces = link_mesh_faces(xml, geom_name)
            fresh = hull_overhang_m(hull_pts, pts, faces, samples_per_edge=32)
            declared = tight.hull_overhang_m
            if declared is None:
                failures.append(f"{geom.link_name}: ships a hull but no hull_overhang_m")
            elif fresh > declared + 1e-6:
                failures.append(
                    f"{geom.link_name}: hull_overhang_m={declared * 1e3:.4f} mm understates the "
                    f"resampled overhang of {fresh * 1e3:.4f} mm"
                )
        elif tight.hull_overhang_m is not None:
            failures.append(
                f"{geom.link_name}: hull_overhang_m is set but hull_vertices_m is empty"
            )
        print(
            f"{geom.link_name}: mesh {len(pts)} vtx, declared hull "
            f"{len(tight.hull_vertices_m)} vtx, mesh-outside-DOP {worst * 1e3:+.9f} mm"
        )
    if checked == 0:
        print(f"{robot.name}: no link declares tight_geometry")
    for line in failures:
        print(f"FAIL {line}", file=sys.stderr)
    return 3 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("emit", "check"))
    parser.add_argument(
        "--robot",
        type=Path,
        default=REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml",
        help="path to the robot manifest",
    )
    args = parser.parse_args(argv)
    if args.mode == "emit":
        return _emit(args.robot)
    return _check(args.robot)


if __name__ == "__main__":
    raise SystemExit(main())
