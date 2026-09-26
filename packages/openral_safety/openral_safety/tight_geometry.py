"""Derive a box's ``tight_geometry`` (26-DOP + exact hull) from the geometry it bounds.

The safety kernel refines a ``BoxShape`` with a staged 26-DOP → exact convex
hull narrow phase when the manifest declares ``LinkCollisionGeometry.
tight_geometry`` (``docs/reference/collision-hull-narrow-phase.md``): the voxel
check runs the DOP then the hull on every box that carries one, and a
self-collision box pair whose two boxes both carry a hull is re-asked of the
two hulls (``refine_self_pair``, issue #191). A capsule cannot carry one.

Containment is **definitional, not fitted**: the slabs are tangent halfspaces
``u·x <= max over the points of u·x`` and the hull is ``conv(points)`` — or,
over the kernel's ``MAX_TIGHT_HULL_VERTICES`` budget, a polytope built only
from those tangent planes (``refine_dop_to_budget``). So
``mesh ⊆ hull ⊆ DOP ⊆ box`` holds with no optimiser tolerance anywhere, and the
kernel's broad-phase window (sized from the box alone) never moves.

Two producers use it: ``tools/generate_tight_geometry.py`` (the panda_mobile
MJCF meshes) and ``urdf_lowering`` (every link the lowering is asked to refine,
from the same collision + visual cloud the primitive fit uses).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt
    from openral_core.schemas import TightCollisionGeometry

    Points = npt.NDArray[Any]

__all__ = [
    "HULL_OVERHANG_SAFETY_MARGIN",
    "derive_tight_geometry",
    "hull_overhang_m",
    "hull_overhang_upper_bound_m",
    "refine_dop_to_budget",
    "round_up_m",
    "tight_geometry_for_box",
]


def _dop_axes() -> Points:
    import numpy as np
    from openral_core.schemas import DOP_AXES

    axes: Points = np.asarray(DOP_AXES, dtype=float)
    return axes


#: Samples per `closest_point_naive` call. The query allocates
#: `samples x mesh-faces x 3` floats, so this bounds peak memory independently
#: of how large a link's mesh is.
_OVERHANG_BATCH = 512

#: Cap on total barycentric samples across all facets, so overhang cost is
#: bounded by mesh size rather than by hull complexity.
_OVERHANG_MAX_SAMPLES = 12_000

# The sampled max keeps creeping up a couple of percent per grid doubling
# (measured up to samples_per_edge=32 on every panda link with a stage-2
# hull); this pads the shipped number well past that residual drift instead
# of chasing convergence with an ever-finer, ever-slower grid.
HULL_OVERHANG_SAFETY_MARGIN = 1.2


def round_up_m(value: float, precision_m: float = 1e-6) -> float:
    """Round a sampled distance up to the next ``precision_m``, never down.

    The sampled maximum is a lower bound on the true continuous supremum; this
    keeps the shipped number from ever quietly under-stating it by less than a
    micron of rounding.

    Example:
        >>> round_up_m(0.0012341)
        0.001235
    """
    return math.ceil(value / precision_m) * precision_m


def _facet_samples(hull_points: Points, samples_per_edge: int) -> Points:
    """A barycentric grid over every facet of ``conv(hull_points)``, total bounded."""
    import numpy as np

    # reason: scipy ships no type stubs and scipy-stubs is not a workspace dependency.
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

    hull = ConvexHull(hull_points)
    # Keep TOTAL samples bounded, not samples-per-facet: cost is
    # (facets x samples-per-facet) x mesh-faces, and a 320-vertex envelope has
    # ~636 facets vs ~300 for a 152-vertex hull. Coarsening only lowers the
    # sampled maximum (a LOWER bound on the true supremum), which the caller's
    # safety margin absorbs.
    n = samples_per_edge
    while n > 4 and len(hull.simplices) * (n + 1) * (n + 2) // 2 > _OVERHANG_MAX_SAMPLES:
        n -= 1
    bary = np.array(
        [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    )
    tris = hull_points[hull.simplices]  # (n_facets, 3, 3)
    samples: Points = np.einsum("fvc,sv->fsc", tris, bary).reshape(-1, 3)
    return samples


def hull_overhang_m(
    hull_points: Points, mesh_points: Points, mesh_faces: Points, *, samples_per_edge: int = 24
) -> float:
    """How far the hull's surface reaches past the real source mesh, in metres.

    Containment (``mesh ⊆ hull``) is exact and definitional. This is the other
    direction: the hull's faces bridge over mesh concavities, and the worst
    point can fall anywhere on a facet (not just at a hull vertex, which sits
    ON the mesh). So this samples a barycentric grid per hull facet and
    measures each sample's exact distance to the real mesh surface.

    Args:
        hull_points: The hull's own vertices, box-frame, the same array
            ``scipy.spatial.ConvexHull`` was built from.
        mesh_points: Real mesh vertices, box-frame.
        mesh_faces: Real mesh triangle indices into ``mesh_points``.
        samples_per_edge: Barycentric grid resolution per hull facet (24 gives
            325 samples/facet). A sampled lower bound on the true continuous
            supremum (measured max creeps up a couple % per doubling), which
            is why a producer pads it (``HULL_OVERHANG_SAFETY_MARGIN``) rather
            than shipping it raw.

    Returns:
        The sampled maximum distance, in metres, with NO margin applied.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=mesh_points, faces=mesh_faces, process=False)
    samples = _facet_samples(hull_points, samples_per_edge)
    # `closest_point` (`mesh.nearest`) needs `rtree`, an optional trimesh dep
    # this workspace does not pin, so query naive brute-force in batches: the
    # largest panda link is ~207k samples x ~12k triangles = 57.8 GiB in one
    # call. Batching bounds peak memory at `_OVERHANG_BATCH x faces x 3`.
    worst = 0.0
    for start in range(0, len(samples), _OVERHANG_BATCH):
        batch = samples[start : start + _OVERHANG_BATCH]
        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, batch)
        worst = max(worst, float(distances.max()))
    return worst


#: Decimal places (metres) ``tight_geometry_for_box`` writes a refinement at: 1 nm.
_WRITE_DP = 9

#: Surface samples ``hull_overhang_upper_bound_m`` draws from the source mesh
#: (plus every vertex), with a fixed seed so a re-lowering is reproducible.
_SURFACE_SAMPLES = 60_000
_SURFACE_SEED = 20260925


def hull_overhang_upper_bound_m(
    hull_points: Points, mesh: Any, *, samples_per_edge: int = 24
) -> float:
    """``hull_overhang_m`` for meshes too large for the brute-force query, from above.

    The distance from a facet sample to its nearest point of a finite sample
    of the mesh surface is never below its distance to the surface itself, so
    the maximum over facet samples is an upper bound on what
    ``hull_overhang_m`` measures at the same samples — the over-stating side,
    the side the shipped number is padded toward anyway. A KD-tree makes it
    O(samples · log n) where the exact query is O(samples · faces) in memory,
    which a 100k-triangle CAD visual mesh cannot afford on a lowering host.

    Args:
        hull_points: The hull's vertices, box-frame.
        mesh: The source ``trimesh.Trimesh``, box-frame.
        samples_per_edge: Barycentric grid resolution per hull facet.

    Returns:
        The bound in metres, with NO margin applied.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    surface, _ = trimesh.sample.sample_surface(mesh, _SURFACE_SAMPLES, seed=_SURFACE_SEED)
    cloud = np.vstack([np.asarray(mesh.vertices, dtype=float), np.asarray(surface, dtype=float)])
    distances, _ = cKDTree(cloud).query(_facet_samples(hull_points, samples_per_edge))
    return float(np.max(distances))


def refine_dop_to_budget(points: Points, dop_lo: Points, dop_hi: Points, budget: int) -> Points:
    """A ≤``budget``-vertex convex envelope strictly tighter than the 26-DOP.

    ``panda_link1``: exact hull is 1588 vertices vs a 320-vertex kernel budget,
    so it ships stage-1 only (26-DOP: median 4.52 mm / max 25.68 mm support gap
    vs the real mesh) — "over budget" need not mean "fall back to the DOP".

    Construction: greedy halfspace refinement, not hull-vertex subsetting,
    because containment must stay definitional. Every candidate plane is a
    face of ``conv(mesh)`` (tangent, can never cut the mesh); the starting
    polytope is the DOP so the result is ``⊆ DOP`` by construction (what
    ``TightCollisionGeometry`` requires); planes are added worst-violation-first
    and skipped if they'd push the vertex count over budget. So
    ``mesh ⊆ result ⊆ DOP ⊆ box`` holds throughout (CLAUDE.md §3).

    Args:
        points: every mesh vertex, in the box's frame.
        dop_lo: per-axis minima already computed for the 26-DOP.
        dop_hi: per-axis maxima already computed for the 26-DOP.
        budget: the kernel's ``kMaxTightHullVertices``.

    Returns:
        The refined polytope's vertices, at most ``budget`` of them.

    Raises:
        ValueError: If the refined envelope would cut the mesh (never emitted).
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
    current = vertices_of(halfspaces)
    while len(current) <= budget:
        # Score each unused tangent plane by how far the current polytope pokes
        # past it: the plane that trims the most is the one worth spending
        # vertices on. A rejected plane leaves the polytope — and so every
        # score — unchanged, so the planes are tried in this one ranking
        # (stable: ties go to the lower index, as `argmax` would) until one is
        # accepted, and only then re-scored.
        score = (current @ candidates[:, :3].T + candidates[:, 3]).max(axis=0)
        score[used] = -np.inf
        accepted = False
        for best in np.argsort(-score, kind="stable"):
            if score[best] <= 1e-9:
                break  # nothing left to trim; this IS the hull within tolerance
            used[best] = True
            trial = np.vstack([halfspaces, candidates[best]])
            trial_vertices = vertices_of(trial)
            if len(trial_vertices) > budget:
                continue  # would overrun the budget; try the next-worst plane
            halfspaces, current, accepted = trial, trial_vertices, True
            break
        if not accepted:
            break

    refined = current
    # Refuse rather than emit an envelope that does not contain its mesh.
    eq = ConvexHull(refined).equations
    residual = float((points @ eq[:, :3].T + eq[:, 3]).max())
    if residual > 1e-9:
        msg = (
            f"refined envelope cuts the mesh by {residual * 1e3:.9f} mm; refusing to "
            "emit an envelope smaller than the geometry it must contain"
        )
        raise ValueError(msg)
    return refined


def derive_tight_geometry(points: Points, half_extents: tuple[float, ...]) -> dict[str, Any]:
    """Build the DOP slabs and (when it fits the budget) the exact hull.

    Returns a mapping ready for ``openral_core.schemas.TightCollisionGeometry``,
    plus the diagnostics a reviewer needs: vertex counts and the achieved inward
    margin of the DOP inside the shipped box.
    """
    import numpy as np
    from openral_core.schemas import MAX_TIGHT_HULL_VERTICES
    from scipy.spatial import ConvexHull

    axes = _dop_axes()
    proj = points @ axes.T
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)

    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    exact_count = len(hull_vertices)
    decimated = False
    if exact_count > MAX_TIGHT_HULL_VERTICES:
        # Over budget is not a reason to fall back to the DOP — it is a reason to
        # refine the DOP toward the hull. See `refine_dop_to_budget`.
        hull_vertices = refine_dop_to_budget(points, lo, hi, MAX_TIGHT_HULL_VERTICES)
        decimated = True

    he = np.asarray(half_extents, dtype=float)
    inward = float(np.minimum(he - hi[:3], he + lo[:3]).min())
    return {
        "dop_lo_m": [float(v) for v in lo],
        "dop_hi_m": [float(v) for v in hi],
        "hull_vertices_m": [[float(c) for c in v] for v in hull_vertices],
        "_hull_vertex_count": len(hull_vertices),
        "_exact_hull_vertex_count": exact_count,
        "_decimated": decimated,
        "_stage2": True,
        "_dop_inward_margin_m": inward,
    }


def tight_geometry_for_box(mesh: Any, half_extents: tuple[float, ...]) -> TightCollisionGeometry:
    """The ``TightCollisionGeometry`` refining a box around ``mesh`` (box frame).

    DOP + hull from ``derive_tight_geometry`` over the mesh's vertices, and the
    hull's overhang past the mesh from ``hull_overhang_upper_bound_m``, padded
    by ``HULL_OVERHANG_SAFETY_MARGIN`` and rounded up. The schema re-checks
    that the hull sits inside the DOP; ``LinkCollisionGeometry`` re-checks that
    the DOP sits inside the box.

    Args:
        mesh: The link's geometry as a ``trimesh.Trimesh`` in the box's frame.
        half_extents: The box it refines.

    Returns:
        The refinement.
    """
    import numpy as np
    from openral_core.schemas import TightCollisionGeometry

    points = np.unique(np.asarray(mesh.vertices, dtype=float), axis=0)
    derived = derive_tight_geometry(points, half_extents)
    hull_pts = np.asarray(derived["hull_vertices_m"], dtype=float)
    overhang = round_up_m(hull_overhang_upper_bound_m(hull_pts, mesh) * HULL_OVERHANG_SAFETY_MARGIN)
    # Written at 1 nm so a re-lowering on another host reproduces the manifest
    # byte for byte (the last float bits of a BLAS product are not portable).
    # The slabs round OUTWARD, so the mesh stays inside the DOP exactly; a hull
    # vertex moves by at most 0.87 nm, under TIGHT_CONTAINMENT_EPSILON_M.
    scale = 10.0**_WRITE_DP
    return TightCollisionGeometry(
        dop_lo_m=tuple(math.floor(v * scale) / scale for v in derived["dop_lo_m"]),
        dop_hi_m=tuple(math.ceil(v * scale) / scale for v in derived["dop_hi_m"]),
        hull_vertices_m=tuple(
            (round(v[0], _WRITE_DP), round(v[1], _WRITE_DP), round(v[2], _WRITE_DP))
            for v in derived["hull_vertices_m"]
        ),
        hull_overhang_m=overhang,
    )
