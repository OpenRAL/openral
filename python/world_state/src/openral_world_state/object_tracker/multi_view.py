"""Multi-view containing fit (issue #418 phase 4): intersect, never shrink below the solid.

Each camera's ``containing_fit`` ``C_i`` contains the solid under A1-A4. A second camera
whose silhouette is complete (A1: not cut by its image border) contains the solid in its
own **cone** ``cone_j`` — the rays through its dilated silhouette's 16-direction outer
polygon — whatever it hides behind it. So

    K = conv(C_p) ∩ cone_j ∩ cone_k ∩ …

contains the solid too, where ``C_p`` is the primary (first fitted) view. Only the second
views' cones are used, never their front cut or prism: those hold only for ``C_j`` as a whole
(a union), so cutting with them could cut the solid.

A second view qualifies only when (design "Containing fit", open question 6):

* its own fit was not refused (``a1_border_truncated`` among others);
* its cone holds **every** measured surface point ``P`` of every view — a silhouette too
  small to cover what another camera measured would cut the solid, so it is refused whole.

``K`` is computed exactly (every plane triple, the feasible points) and reduced *outwards*
to the kernel's 64-vertex hull cap: as is when it has at most 64 vertices, else the hull of
two horizontal slabs' 16-direction outer polygons at each slab's bottom and top. The result
falls back to the primary fit (``fit_kind`` ``silhouette_cone``, reason logged) when no
second view qualifies, when a measured point of any view falls outside the result, or when
the result's box is not smaller than the primary's.

Pure numpy, no ROS, no scipy (the ROS tracker node's interpreter need not have it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import FitKind
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_world_state.object_tracker.fit import ContainingFit, _boxed, _outer_polygon

__all__ = ["MultiViewFit", "intersect_views"]

log = structlog.get_logger(__name__)

_RING = 16  # fit.py's 16-direction rings: front, support, footprint at support, at top
_MAX_VERTS = 4 * _RING  # TrackedObject.MAX_HULL_VERTICES
_TOL_M = 1e-9  # feasibility slack; only ever widens the set
_VERTEX_TOL_M = 1e-7  # a solved plane-triple point this far outside a plane is no vertex
_DEGENERATE = 1e-12  # a cross product / determinant this small spans no plane / vertex


@dataclass(frozen=True)
class MultiViewFit:
    """Result of ``intersect_views``.

    Attributes:
        fit: The multi-view fit, or the primary single-view fit on any fallback.
        fit_kind: ``MULTI_VIEW_CONE`` when an intersection was applied, else
            ``SILHOUETTE_CONE``.
        refusals: Per view index: ``"contributed"`` (its cone cut the result), ``"primary"``
            (the view the others cut; ``"fallback:<why>"`` when the whole result fell back),
            else why it did not contribute.
    """

    fit: ContainingFit
    fit_kind: FitKind
    refusals: dict[int, str] = field(default_factory=dict)


Planes = tuple[NDArray[np.float64], NDArray[np.float64]]  # (normals (m, 3), offsets (m,))


def _hull_base(fit: ContainingFit) -> NDArray[np.float64]:
    """The fit's hull vertices in the base frame, in ``fit.py``'s order."""
    assert fit.tight is not None and fit.tight.tight_geometry is not None
    pose = fit.tight.pose_in_object
    t = homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)
    local = np.asarray(fit.tight.tight_geometry.hull_vertices_m, dtype=np.float64)
    return np.asarray(local @ t[:3, :3].T + t[:3, 3])


def _hull_planes(v: NDArray[np.float64]) -> Planes:
    """Supporting planes ``n·x <= c`` of ``conv(v)``: every point triple, kept if one-sided.

    Every facet is among them; a near-degenerate triple can only add a valid outer plane.
    """
    i, j, k = np.asarray(list(combinations(range(len(v)), 3))).T
    n = np.cross(v[j] - v[i], v[k] - v[i])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > _DEGENERATE
    n = n[ok] / norm[ok, None]
    c = np.einsum("ij,ij->i", n, v[i[ok]])
    side = v @ n.T - c
    below, above = (side <= _TOL_M).all(axis=0), (side >= -_TOL_M).all(axis=0)
    normals = np.concatenate((n[below], -n[above]))
    offsets = np.concatenate((c[below], -c[above])) + _TOL_M
    key = np.round(np.column_stack((normals, offsets)), 9)
    _, first = np.unique(key, axis=0, return_index=True)
    return normals[first], offsets[first]


def _cone_planes(rays: NDArray[np.float64], origin: NDArray[np.float64]) -> Planes:
    """The planes of the cone over consecutive ``rays`` from ``origin``, interior ``<=``."""
    inward = (rays / np.linalg.norm(rays, axis=1, keepdims=True)).mean(axis=0)
    n = np.cross(rays, np.roll(rays, -1, axis=0))
    norm = np.linalg.norm(n, axis=1)
    keep = norm > _DEGENERATE  # coincident polygon vertices: no plane (cuts less, never more)
    n = n[keep] / norm[keep, None]
    n = np.where((n @ inward)[:, None] > 0.0, -n, n)
    return np.asarray(n, np.float64), np.asarray(n @ origin + _TOL_M, np.float64)


def _vertices(planes: Planes) -> NDArray[np.float64]:
    """Vertices of ``{x: n·x <= c}``: every non-degenerate plane triple's feasible point."""
    n, c = planes
    idx = np.asarray(list(combinations(range(len(n)), 3)))
    a, b = n[idx], c[idx]
    ok = np.abs(np.linalg.det(a)) > _DEGENERATE
    x = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
    feasible = (x @ n.T - c <= _VERTEX_TOL_M).all(axis=1)
    return np.asarray(np.unique(np.round(x[feasible], 12), axis=0))


def _inside(points: NDArray[np.float64], planes: Planes) -> bool:
    n, c = planes
    return bool((points @ n.T - c <= _TOL_M).all())


def _outward_64(planes: Planes, verts: NDArray[np.float64]) -> NDArray[np.float64]:
    """At most 64 points whose hull contains ``{n·x <= c}`` (``verts`` are its vertices)."""
    if len(verts) <= _MAX_VERTS:
        return verts
    z_mid = 0.5 * (verts[:, 2].min() + verts[:, 2].max())
    rings = []
    for sign in (1.0, -1.0):  # z <= z_mid, then z >= z_mid
        slab = _vertices(
            (
                np.vstack((planes[0], (0.0, 0.0, sign))),
                np.append(planes[1], sign * z_mid + _TOL_M),
            )
        )
        poly = _outer_polygon(slab[:, :2])
        for z in (slab[:, 2].min(), slab[:, 2].max()):
            rings.append(np.column_stack((poly, np.full(len(poly), z))))
    return np.concatenate(rings)


def _volume(fit: ContainingFit) -> float:
    assert fit.box is not None
    hx, hy, hz = fit.box.half_extents
    return 8.0 * hx * hy * hz


def intersect_views(
    views: Sequence[tuple[ContainingFit, NDArray[np.float64]]],
    *,
    resolution: float,
    frame_id: str,
    stamp_ns: int,
) -> MultiViewFit:
    """Intersect one object's per-camera containing fits (module docstring).

    Args:
        views: ``(containing_fit result, T_base_from_optical)`` per depth camera that saw
            the object at about the same source time, all in the grid frame.
        resolution: Grid cell edge (the box pads half of it).
        frame_id: The grid frame.
        stamp_ns: Stamp carried on the box.

    Returns:
        A ``MultiViewFit``; ``fit_kind`` says whether an intersection was applied.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> from openral_world_state.object_tracker import containing_fit
        >>> k = IntrinsicsPinhole(width=64, height=48, fx=60.0, fy=60.0, cx=32.0, cy=24.0)
        >>> t = np.eye(4)
        >>> t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # looking straight down
        >>> t[2, 3] = 0.5
        >>> depth = np.full((48, 64), 0.5)
        >>> depth[20:28, 28:36] = 0.45
        >>> f = containing_fit(
        ...     depth < 0.49,
        ...     depth,
        ...     None,
        ...     k,
        ...     t,
        ...     support_z=0.0,
        ...     resolution=0.01,
        ...     frame_id="b",
        ...     stamp_ns=1,
        ... )
        >>> intersect_views([(f, t)], resolution=0.01, frame_id="b", stamp_ns=1).fit_kind
        <FitKind.SILHOUETTE_CONE: 'silhouette_cone'>
    """
    refusals: dict[int, str] = {}
    primary = next((i for i, (f, _) in enumerate(views) if not f.refusal), None)
    if primary is None:
        return MultiViewFit(views[0][0], FitKind.SILHOUETTE_CONE, {0: "no_fitted_view"})
    base = views[primary][0]
    refusals[primary] = "primary"
    measured = np.concatenate([f.surface for f, _ in views if not f.refusal])
    n, c = _hull_planes(_hull_base(base))
    for i, (f, t) in enumerate(views):
        if i == primary:
            continue
        if f.refusal:
            refusals[i] = f.refusal
            continue
        origin = t[:3, 3]
        hull = _hull_base(f)
        rays = hull[_RING : 2 * _RING] - origin  # fit.py: support ring, on the cone's rays
        front = hull[:_RING] - origin
        parallel = np.linalg.norm(np.cross(rays, front), axis=1) <= 1e-6 * (
            np.linalg.norm(rays, axis=1) * np.linalg.norm(front, axis=1) + 1e-12
        )
        if len(hull) != _MAX_VERTS or not parallel.all():
            refusals[i] = "cone_layout"
            continue
        cone = _cone_planes(rays, origin)
        if not _inside(measured, cone):
            refusals[i] = "silhouette_too_small"  # would cut what another camera measured
            continue
        n, c = np.vstack((n, cone[0])), np.append(c, cone[1])
        refusals[i] = "contributed"

    def fallback(why: str) -> MultiViewFit:
        refusals[primary] = f"fallback:{why}"
        log.info("object_tracker.multi_view_fallback", reason=why, views=refusals)
        return MultiViewFit(base, FitKind.SILHOUETTE_CONE, refusals)

    if "contributed" not in refusals.values():
        return fallback("no_second_view")
    verts = _vertices((n, c))
    if len(verts) < 4:  # noqa: PLR2004  # reason: a solid needs 4 vertices
        return fallback("empty_intersection")
    pts = _outward_64((n, c), verts)
    # conv(C_p) stands on the support; only the feasibility slack dips under it.
    pts[:, 2] = np.maximum(pts[:, 2], base.support_z)
    if not _inside(measured, _hull_planes(pts)):
        return fallback("views_disagree")
    cones = float(sum(r == "contributed" for r in refusals.values()))
    notes = {**base.assumptions, "multi_view_cones": cones}
    fit = _boxed(pts, base.surface, base.support_z, resolution, frame_id, stamp_ns, notes)
    if fit.refusal or _volume(fit) >= _volume(base):
        return fallback(fit.refusal or "no_gain")
    assert fit.box is not None
    box = fit.box.model_copy(update={"evidence_ref": "object_tracker:multi_view_cone"})
    log.info("object_tracker.multi_view", views=refusals, vertices=len(pts))
    return MultiViewFit(
        ContainingFit(box, fit.tight, fit.surface, fit.support_z, fit.assumptions),
        FitKind.MULTI_VIEW_CONE,
        refusals,
    )
