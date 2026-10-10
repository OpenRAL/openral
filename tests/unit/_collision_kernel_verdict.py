"""The safety kernel's self-collision verdict for one link pair, in Python.

``check_self_collision`` (``cpp/openral_safety_kernel/src/collision.cpp``) folds a
link pair as the minimum gap over the two links' primitives, and re-asks a box
pair of the two boxes' exact hulls (``refine_self_pair`` → ``hull_hull_distance``)
when both boxes carry ``tight_geometry`` hull vertices and the boxes are already
within the margin. The hull answer replaces the box bound only when it proves
separation; overlapping hulls keep the box bound, as the kernel does.

Shared by the tests that decide "would the kernel trip here" on real manifests
(the rest-pose test and the contact sweep), so a refined link is judged by its
hull, not by the corner slop of the box around it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openral_core.schemas import LinkCollisionGeometry
from openral_hal.convex_distance import ConvexBody, _signed_distance
from openral_safety.kernel_predicates import shape_distance
from openral_safety.urdf_lowering import _xyzrpy_matrix


def _hull_world(geom: LinkCollisionGeometry, placed: np.ndarray) -> np.ndarray | None:
    tight = geom.tight_geometry
    if tight is None or not tight.hull_vertices_m:
        return None
    hull = np.asarray(tight.hull_vertices_m, dtype=float)
    return hull @ placed[:3, :3].T + placed[:3, 3]


def kernel_link_gap(
    geoms_a: Sequence[LinkCollisionGeometry],
    pose_a: np.ndarray,
    geoms_b: Sequence[LinkCollisionGeometry],
    pose_b: np.ndarray,
    margin: float,
) -> float:
    """The gap the kernel measures between two links at one pose (``<= margin`` trips).

    Args:
        geoms_a: Every primitive of link ``a``.
        pose_a: Link ``a``'s ``(4, 4)`` world pose.
        geoms_b: Every primitive of link ``b``.
        pose_b: Link ``b``'s ``(4, 4)`` world pose.
        margin: The robot's ``self_collision_margin_m``.

    Returns:
        The minimum over primitive pairs, hull-refined where the kernel refines.
    """
    best = np.inf
    for ga in geoms_a:
        ta = pose_a @ _xyzrpy_matrix(ga.origin_xyz_rpy)
        for gb in geoms_b:
            tb = pose_b @ _xyzrpy_matrix(gb.origin_xyz_rpy)
            gap = float(shape_distance(ga.shape, ta[None], gb.shape, tb[None])[0])
            if gap <= margin:
                hull_a, hull_b = _hull_world(ga, ta), _hull_world(gb, tb)
                if hull_a is not None and hull_b is not None:
                    refined = _signed_distance(
                        ConvexBody(core=hull_a, radius=0.0, faces=None),
                        ConvexBody(core=hull_b, radius=0.0, faces=None),
                    )[0]
                    if refined > 0.0:
                        gap = float(refined)
            best = min(best, gap)
    return float(best)


def tight_escape_m(geom: LinkCollisionGeometry, local: np.ndarray) -> float:
    """How far box-frame points reach outside ``geom``'s DOP or hull (``<= 0``: inside).

    The refinement contains nothing by construction unless it contains the
    mesh: ``mesh ⊆ hull ⊆ DOP`` is the whole safety argument of the kernel's
    staged narrow phase, so the enclosure tests check it on every refined box.

    Args:
        geom: A primitive carrying ``tight_geometry``.
        local: Points in the box's own frame.

    Returns:
        The largest excursion past a DOP slab or a hull facet, in metres.
    """
    from openral_core.schemas import DOP_AXES
    from scipy.spatial import ConvexHull

    tight = geom.tight_geometry
    assert tight is not None, f"{geom.link_name}: no tight_geometry to check"
    proj = local @ np.asarray(DOP_AXES, dtype=float).T
    worst = float(
        max(
            (proj - np.asarray(tight.dop_hi_m)).max(),
            (np.asarray(tight.dop_lo_m) - proj).max(),
        )
    )
    if tight.hull_vertices_m:
        key = id(tight)
        if key not in _FACETS:  # the hull is fixed per primitive; pose-independent
            eq = ConvexHull(np.asarray(tight.hull_vertices_m, dtype=float)).equations
            norm = np.linalg.norm(eq[:, :3], axis=1)
            _FACETS[key] = (eq[:, :3] / norm[:, None], eq[:, 3] / norm, tight)
        normals, offsets, _ = _FACETS[key]
        worst = max(worst, float((local @ normals.T + offsets).max()))
    return worst


# Unit facet planes per refinement, keyed by object identity (the refinement is
# kept alive in the value so the id cannot be reused).
_FACETS: dict[int, tuple[np.ndarray, np.ndarray, object]] = {}


def _signed_distance_to(geom: LinkCollisionGeometry, pts: np.ndarray) -> np.ndarray:
    """Signed distance of link-frame points to one primitive (negative inside)."""
    from openral_core.schemas import BoxShape, CapsuleShape

    tf = _xyzrpy_matrix(geom.origin_xyz_rpy)
    local = (pts - tf[:3, 3]) @ tf[:3, :3]
    shape = geom.shape
    if isinstance(shape, BoxShape):
        q = np.abs(local) - np.asarray(shape.half_extents_m)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    half = shape.length_m / 2.0 if isinstance(shape, CapsuleShape) else 0.0
    z = np.clip(local[:, 2], -half, half)
    axial = local - np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return np.linalg.norm(axial, axis=1) - shape.radius_m  # type: ignore[union-attr]  # reason: capsule or sphere


def surface_pieces(
    vertices: np.ndarray,
    faces: np.ndarray,
    geoms: Sequence[LinkCollisionGeometry],
    *,
    max_depth: int = 10,
) -> tuple[list[np.ndarray], int]:
    """Split a link's surface among its primitives, proving each piece is inside one.

    A triangle whose three corners lie inside one CONVEX primitive lies inside
    it entirely, so every triangle is assigned to the first primitive holding
    all three corners, and a triangle no single primitive holds is split into
    four at its edge midpoints and retried (a capsule chain's seam triangles).
    What comes back is, per primitive, the points of the triangles assigned to
    it — so ``conv(points_k) ⊆ primitive_k`` and the union of the pieces'
    hulls contains the whole surface — plus how many (sub)triangles no
    primitive held after ``max_depth`` splits (``0`` proves the surface is
    inside the union, which checking the vertices alone does not for a
    non-convex union).

    That union of piece hulls is also the ground truth the contact sweep uses
    for a link with several primitives: it contains the real surface, and
    unlike the link's single convex hull it does not fill the concavities a
    capsule chain deliberately leaves out.

    Args:
        vertices: The link's mesh vertices, link frame.
        faces: Its triangles (indices into ``vertices``).
        geoms: The link's primitives.
        max_depth: Midpoint splits tried before a triangle counts as uncovered.

    Returns:
        ``(points per primitive, uncovered triangle count)``.
    """
    tris = np.asarray(vertices, dtype=float)[np.asarray(faces)]
    pieces: list[list[np.ndarray]] = [[] for _ in geoms]
    for _ in range(max_depth + 1):
        if len(tris) == 0:
            break
        inside = np.stack(
            [_signed_distance_to(g, tris.reshape(-1, 3)) <= 0.0 for g in geoms], axis=1
        ).reshape(len(tris), 3, len(geoms))
        held = inside.all(axis=1)  # (T, P): primitive p holds all three corners
        accepted = held.any(axis=1)
        owner = held.argmax(axis=1)
        for k in range(len(geoms)):
            chosen = tris[accepted & (owner == k)]
            if len(chosen):
                pieces[k].append(chosen.reshape(-1, 3))
        rest = tris[~accepted]
        m01, m12, m20 = (
            (rest[:, 0] + rest[:, 1]) / 2,
            (rest[:, 1] + rest[:, 2]) / 2,
            (rest[:, 2] + rest[:, 0]) / 2,
        )
        tris = np.concatenate(
            [
                np.stack([rest[:, 0], m01, m20], axis=1),
                np.stack([m01, rest[:, 1], m12], axis=1),
                np.stack([m20, m12, rest[:, 2]], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ]
        )
    return [np.unique(np.vstack(p), axis=0) if p else np.empty((0, 3)) for p in pieces], len(tris)
