"""Offline URDF(+SRDF) → manifest collision-model lowering tool.

Produces the hand-reviewable ``collision_geometry`` + ``allowed_collision_pairs``
that ``robot.yaml`` carries and ``collision_params_from_description`` consumes:

* **Geometry** — fit one conservative capsule/sphere per link to the URDF
  ``<collision>`` and ``<visual>`` geometry together (minimum-volume capsule
  holding every vertex plus a declared headroom, so the safety check never
  under-covers the part).
* **ACM** — adjacent pairs, plus pairs *proved* always-colliding over their own
  relative-DoF subspace, plus the hand-reviewed rows of the SRDF
  ``disable_collisions`` block where one exists. Every verdict is taken with the
  **kernel's own** predicates (``openral_safety.kernel_predicates``) at the
  robot's own ``self_collision_margin_m``, so the generated matrix is about the
  robot the kernel actually checks. No RNG: the result is reproducible.

Heavy deps (``yourdfpy``, ``trimesh``) are imported lazily — install the
optional ``[lowering]`` group. Pure: no ROS, no I/O beyond reading the source
files passed in.

An ACM entry *removes* a self-collision check, so every rule here is written to
fail toward *fewer* entries: a missing entry costs a false E-stop, an unearned
one hides a real self-collision (issue #155).
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from openral_core import (
    BoxShape,
    CapsuleShape,
    CollisionShape,
    LinkCollisionGeometry,
    RobotDescription,
    SphereShape,
)
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    _Arr = NDArray[np.float64]

__all__ = [
    "CAPSULE_HEADROOM_M",
    "LinkFit",
    "LoweredCollisionModel",
    "LoweringSource",
    "acm_for_geometry",
    "fit_capsule_to_vertices",
    "fit_collision_geometry_from_mjcf",
    "fit_link_primitives",
    "fit_link_with_tight_geometry",
    "fit_obb_to_vertices",
    "lower_joint_fk",
    "lower_link_geometry",
    "lower_robot",
    "lower_robot_auto",
    "lower_robot_from_mjcf",
    "mjcf_coupled_joints",
    "parse_srdf_disabled_pairs",
    "sample_acm_from_urdf",
    "select_lowering",
]

_AcmPairs = set[frozenset[str]]
_Origin = tuple[float, float, float, float, float, float]
_Vec3 = tuple[float, float, float]


def parse_srdf_disabled_pairs(srdf_path: str) -> _AcmPairs:
    """Parse ``<disable_collisions link1 link2/>`` rows into unordered link pairs.

    Args:
        srdf_path: Filesystem path to a MoveIt SRDF.

    Returns:
        A set of two-element frozensets (symmetric, dedup'd). Self-pairs and
        rows missing a link attribute are skipped.

    Example:
        >>> # parse_srdf_disabled_pairs("panda.srdf")
        >>> # -> {frozenset({"panda_link1", "panda_link2"}), ...}
    """
    root = ET.parse(srdf_path).getroot()  # reason: trusted local SRDF
    pairs: _AcmPairs = set()
    for el in root.iter("disable_collisions"):
        a = el.get("link1")
        b = el.get("link2")
        if a and b and a != b:
            pairs.add(frozenset({a, b}))
    return pairs


# ── Geometry: URDF <collision> → conservative capsule / sphere per link ────────


def _mat_to_rpy(r: _Arr) -> tuple[float, float, float]:
    """Row-major 3×3 → fixed-axis XYZ (roll, pitch, yaw); the kernel's convention.

    Inverse of ``mjcf_lowering._rpy_to_mat`` (R = Rz(yaw)·Ry(pitch)·Rx(roll)), so
    a capsule placed by ``origin_xyz_rpy`` lands where the cloud was fitted.
    """
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:  # gimbal lock
        roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


# Headroom every fitted primitive carries beyond the geometry it bounds: the
# radius is grown by this much after the fit. It is a declared margin, not a fit
# tolerance: it absorbs the manifest's 4-dp rendering of radius/length/origin
# (at most ~0.1 mm of displacement over a link) with room to spare, so the
# committed primitive — not just the in-memory fit — still contains every
# vertex. `tests/unit/test_collision_geometry_enclosure_urdf.py` asserts that on
# the written manifests.
CAPSULE_HEADROOM_M = 0.001
# Axis candidates for the capsule fit: the cloud's three principal axes plus a
# Fibonacci hemisphere, then a local refinement. The search only changes how
# TIGHT the capsule is — containment holds by construction for every axis.
_AXIS_SEARCH_DIRS = 256
_AXIS_REFINE_MIN_STEP = 1e-4
# Clouds at least this large are reduced to their convex-hull vertices first.
_HULL_REDUCE_MIN_POINTS = 64


def _min_enclosing_circle(pts: _Arr) -> tuple[_Arr, float]:
    """Smallest circle holding every 2-D point (Welzl, iterative).

    Each loop only scans forward, so it terminates whatever floating point does
    at the boundary. The capsule built on the centre takes its radius as the
    true max distance (``_capsule_for_axis``), so a last-ulp miss here can only
    loosen the fit, never let a point out. The fixed-seed shuffle keeps the
    expected cost linear and the result reproducible.

    Plain Python floats, not numpy: this runs ~40k times per robot on clouds of
    a few hundred points, where numpy's per-call overhead dominated the whole
    lowering. The arithmetic is the one numpy did — ``sqrt(dx*dx + dy*dy)`` is
    what ``np.linalg.norm`` computes for a 2-vector — so the circle is the same
    to the last bit.
    """
    import numpy as np

    p = pts[np.random.default_rng(0).permutation(len(pts))]
    xs: list[float] = p[:, 0].tolist()
    ys: list[float] = p[:, 1].tolist()
    sqrt = math.sqrt

    def next_outside(start: int, stop: int, cx: float, cy: float, r: float) -> int:
        """The first index in ``[start, stop)`` outside the circle, or -1."""
        limit = r * (1.0 + 1e-12) + 1e-15
        for q in range(start, stop):
            dx = xs[q] - cx
            dy = ys[q] - cy
            if sqrt(dx * dx + dy * dy) > limit:
                return q
        return -1

    def circle_2(i: int, j: int) -> tuple[float, float, float]:
        cx = (xs[i] + xs[j]) / 2.0
        cy = (ys[i] + ys[j]) / 2.0
        dx, dy = xs[i] - cx, ys[i] - cy
        return cx, cy, sqrt(dx * dx + dy * dy)

    def circle_3(i: int, j: int, k: int) -> tuple[float, float, float]:
        """Circumcircle of three points (the widest pair's circle if collinear)."""
        ax, ay, bx, by, qx, qy = xs[i], ys[i], xs[j], ys[j], xs[k], ys[k]
        d = 2.0 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
        if abs(d) < 1e-18:

            def span(pair: tuple[int, int]) -> float:
                dx, dy = xs[pair[0]] - xs[pair[1]], ys[pair[0]] - ys[pair[1]]
                return sqrt(dx * dx + dy * dy)

            return circle_2(*max([(i, j), (i, k), (j, k)], key=span))
        sa, sb, sc = ax * ax + ay * ay, bx * bx + by * by, qx * qx + qy * qy
        ux = (sa * (by - qy) + sb * (qy - ay) + sc * (ay - by)) / d
        uy = (sa * (qx - bx) + sb * (ax - qx) + sc * (bx - ax)) / d
        dx, dy = ax - ux, ay - uy
        return ux, uy, sqrt(dx * dx + dy * dy)

    n = len(xs)
    cx, cy, radius = xs[0], ys[0], 0.0
    i = next_outside(1, n, cx, cy, radius)
    while i >= 0:
        cx, cy, radius = xs[i], ys[i], 0.0
        j = next_outside(0, i, cx, cy, radius)
        while j >= 0:
            cx, cy, radius = circle_2(i, j)
            k = next_outside(0, j, cx, cy, radius)
            while k >= 0:
                cx, cy, radius = circle_3(i, j, k)
                k = next_outside(k + 1, j, cx, cy, radius)
            j = next_outside(j + 1, i, cx, cy, radius)
        i = next_outside(i + 1, n, cx, cy, radius)
    return np.array([cx, cy]), radius


def _capsule_for_axis(pts: _Arr, axis: _Arr) -> tuple[float, _Arr, _Arr, float]:
    """Tightest capsule with direction ``axis`` holding every point.

    The axis line goes through the centre of the minimal enclosing circle of the
    points projected on the plane normal to ``axis``; the radius is the largest
    distance of any point from that line; the segment is the shortest one on the
    line whose capsule still holds every point (a point at axial position ``t``
    and line distance ``d`` is inside iff the nearer end is within
    ``sqrt(r² - d²)`` of ``t``). Returns ``(volume, end0, end1, radius)``.
    """
    import numpy as np

    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    planar = np.stack([pts @ e1, pts @ e2], axis=1)
    c2, _ = _min_enclosing_circle(planar)
    d = np.linalg.norm(planar - c2, axis=1)
    radius = max(float(d.max()), 1e-4)
    t = pts @ a
    slack = np.sqrt(np.maximum(radius * radius - d * d, 0.0))
    lo = float((t + slack).min())
    hi = float((t - slack).max())
    if lo > hi:  # a short, fat cloud: one sphere anywhere in [hi, lo] holds it
        lo = hi = (lo + hi) / 2.0
    base = c2[0] * e1 + c2[1] * e2
    volume = math.pi * radius * radius * (hi - lo) + 4.0 / 3.0 * math.pi * radius**3
    return volume, base + a * lo, base + a * hi, radius


def fit_capsule_to_vertices(
    vertices: _Arr, *, headroom_m: float = CAPSULE_HEADROOM_M
) -> tuple[CapsuleShape, _Origin]:
    """Fit a conservative, minimum-volume bounding capsule (segment along +Z) to a cloud.

    The axis direction is searched (principal axes, a 256-direction Fibonacci
    hemisphere, then a local refinement) for the capsule of least volume; for
    each direction the axis line passes through the centre of the minimal
    enclosing circle of the cloud's projection, and the segment is the shortest
    whose capsule still holds every vertex (``_capsule_for_axis``). Every vertex
    is inside the result by construction, whichever axis wins — the search only
    decides how tight it is. The radius then grows by ``headroom_m`` so the
    capsule keeps that clearance around every vertex after the manifest's 4-dp
    rendering. A short, fat cloud gets a zero-length capsule (a sphere).

    The principal axes the previous fitter used are always candidates. Measured
    on the seven URDF-lowered robots' real link meshes, the result is 1.5–2.2×
    the convex-hull volume against the PCA fit's 2.6–4.1×
    (``docs/reference/collision-geometry-review.md``).

    Args:
        vertices: ``(N, 3)`` point cloud (N ≥ 1) in the link frame.
        headroom_m: Declared clearance added to the fitted radius.

    Returns:
        ``(CapsuleShape, origin_xyz_rpy)`` in the same frame as ``vertices``:
        the segment midpoint and the rotation taking local +Z onto the axis.
    """
    import numpy as np

    pts = _hull_points(vertices)
    _, _, vh = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    n = _AXIS_SEARCH_DIRS
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - i / n)  # upper hemisphere: a capsule axis has no sign
    azim = math.pi * (1.0 + 5.0**0.5) * i
    fib = np.stack(
        [np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)], axis=1
    )
    best = min((_capsule_for_axis(pts, d) for d in [*vh, *fib]), key=lambda c: c[0])
    seg = best[2] - best[1]
    axis = seg / np.linalg.norm(seg) if np.linalg.norm(seg) > 1e-12 else vh[0]
    step = 0.1
    while step >= _AXIS_REFINE_MIN_STEP:
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        improved = False
        for delta in (e1, -e1, e2, -e2):
            trial_axis = (axis + step * delta) / np.linalg.norm(axis + step * delta)
            trial = _capsule_for_axis(pts, trial_axis)
            if trial[0] < best[0] - 1e-15:
                best, axis, improved = trial, trial_axis, True
        if not improved:
            step /= 2.0
    _, end0, end1, radius = best
    seg = end1 - end0
    length = float(np.linalg.norm(seg))
    axis = seg / length if length > 1e-12 else axis
    center = (end0 + end1) / 2.0
    # Rotation taking local +Z onto `axis` (Rodrigues; handle the antiparallel case).
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    c = float(np.dot(z, axis))
    if float(np.linalg.norm(v)) < 1e-9:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        rot = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    roll, pitch, yaw = _mat_to_rpy(rot)
    origin: _Origin = (
        float(center[0]),
        float(center[1]),
        float(center[2]),
        roll,
        pitch,
        yaw,
    )
    return CapsuleShape(radius_m=radius + headroom_m, length_m=length), origin


def _hull_points(vertices: _Arr) -> _Arr:
    """The convex-hull vertices of a cloud (the cloud itself when it is small).

    A convex primitive holds the cloud iff it holds the cloud's hull vertices,
    so every fitter below works on these: exact, and 10-100x cheaper on CAD
    meshes.
    """
    import numpy as np

    pts = np.unique(np.asarray(vertices, dtype=np.float64).reshape(-1, 3), axis=0)
    if len(pts) >= _HULL_REDUCE_MIN_POINTS:
        import trimesh

        pts = np.asarray(trimesh.convex.convex_hull(pts).vertices, dtype=np.float64)
    return pts


def fit_obb_to_vertices(
    vertices: _Arr, *, headroom_m: float = CAPSULE_HEADROOM_M
) -> tuple[BoxShape, _Origin]:
    """Minimum-volume oriented bounding box of a cloud, grown by ``headroom_m``.

    The frame comes from ``trimesh.bounds.oriented_bounds`` (rotating calipers
    over the hull); the half-extents are re-measured from the cloud in that
    frame, so containment is by construction, and then each grows by
    ``headroom_m`` for the same reason a capsule's radius does.

    Returns:
        ``(BoxShape, origin_xyz_rpy)`` in the cloud's frame.
    """
    import numpy as np
    import trimesh

    pts = _hull_points(vertices)
    to_box, _ = trimesh.bounds.oriented_bounds(pts)
    box_to_link = np.linalg.inv(np.asarray(to_box, dtype=np.float64))
    rot = np.asarray(box_to_link[:3, :3], dtype=np.float64)
    if np.linalg.det(rot) < 0.0:
        rot = rot @ np.diag([1.0, 1.0, -1.0])  # a proper rotation; the box is symmetric
    local = (pts - box_to_link[:3, 3]) @ rot
    lo, hi = local.min(axis=0), local.max(axis=0)
    half = (hi - lo) / 2.0 + headroom_m
    center = box_to_link[:3, 3] + rot @ ((lo + hi) / 2.0)
    roll, pitch, yaw = _mat_to_rpy(rot)
    origin: _Origin = (float(center[0]), float(center[1]), float(center[2]), roll, pitch, yaw)
    return BoxShape(half_extents_m=(float(half[0]), float(half[1]), float(half[2]))), origin


def _capsule_axis(shape: CapsuleShape, origin: _Origin) -> _Arr:
    """The unit axis of a placed capsule (its local +Z) in the cloud's frame."""
    import numpy as np

    return np.asarray(_xyzrpy_matrix(origin)[:3, 2], dtype=np.float64)


def _capsule_chain(
    mesh: Any, axis: _Arr, k: int, *, headroom_m: float
) -> list[tuple[CollisionShape, _Origin]] | None:
    """``k`` capsules holding a mesh: the mesh cut into ``k`` equal slabs along ``axis``.

    Each slab piece keeps its cut points (``trimesh.slice_plane``), so the piece
    of the solid inside the slab lies in the convex hull of the piece's vertices,
    which ``fit_capsule_to_vertices`` holds; the union of the ``k`` capsules
    therefore holds the whole mesh. Cutting the mesh, not its hull, is what buys
    anything: a hull slab bulges between the fat and thin ends of a link, the
    mesh slab does not. ``None`` when a slab is degenerate.
    """
    import numpy as np

    t = np.asarray(mesh.vertices) @ axis
    cuts = np.linspace(float(t.min()), float(t.max()), k + 1)
    out: list[tuple[CollisionShape, _Origin]] = []
    for i in range(k):
        piece = mesh
        if i > 0:
            piece = piece.slice_plane(axis * cuts[i], axis, cap=False)
        if i < k - 1 and piece is not None:
            piece = piece.slice_plane(axis * cuts[i + 1], -axis, cap=False)
        if piece is None or len(piece.vertices) < 4:
            return None
        out.append(fit_capsule_to_vertices(np.asarray(piece.vertices), headroom_m=headroom_m))
    return out


_PROTRUSION_DIRS = 1024
# Candidates whose mean protrusion is within this fraction of the least are
# ties; the tie goes to the cheaper kernel kind (capsules over boxes, fewer over more).
_PROTRUSION_TIE_FRACTION = 0.03
_UNION_VOLUME_SAMPLES = 60_000


def _support(prims: list[tuple[CollisionShape, _Origin]], dirs: _Arr) -> _Arr:
    """Support function of a primitive union along each row of ``dirs``."""
    import numpy as np

    best = np.full(len(dirs), -np.inf)
    for shape, origin in prims:
        tf = _xyzrpy_matrix(origin)
        c = tf[:3, 3]
        if isinstance(shape, BoxShape):
            h = np.asarray(shape.half_extents_m)
            s = dirs @ c + np.abs(dirs @ tf[:3, :3]) @ h
        else:
            half = shape.length_m / 2.0 if isinstance(shape, CapsuleShape) else 0.0
            a = tf[:3, 2] * half
            s = np.maximum(dirs @ (c + a), dirs @ (c - a)) + shape.radius_m
        best = np.maximum(best, s)
    return np.asarray(best, dtype=np.float64)


def _primitive_volume(shape: CollisionShape) -> float:
    if isinstance(shape, BoxShape):
        hx, hy, hz = shape.half_extents_m
        return 8.0 * hx * hy * hz
    r = shape.radius_m
    length = shape.length_m if isinstance(shape, CapsuleShape) else 0.0
    return math.pi * r * r * length + 4.0 / 3.0 * math.pi * r**3


def _union_volume(prims: list[tuple[CollisionShape, _Origin]]) -> float:
    """Volume of a primitive union: closed form for one, fixed-seed Monte Carlo for more."""
    import numpy as np

    if len(prims) == 1:
        return _primitive_volume(prims[0][0])
    eye = np.eye(3)
    hi = _support(prims, eye)
    lo = -_support(prims, -eye)
    x = np.random.default_rng(0).uniform(lo, hi, (_UNION_VOLUME_SAMPLES, 3))
    inside = np.zeros(len(x), dtype=bool)
    for shape, origin in prims:
        tf = _xyzrpy_matrix(origin)
        local = (x - tf[:3, 3]) @ tf[:3, :3]
        if isinstance(shape, BoxShape):
            inside |= np.all(np.abs(local) <= np.asarray(shape.half_extents_m), axis=1)
        else:
            half = shape.length_m / 2.0 if isinstance(shape, CapsuleShape) else 0.0
            z = np.clip(local[:, 2], -half, half)
            inside |= np.linalg.norm(local - np.column_stack([0 * z, 0 * z, z]), axis=1) <= (
                shape.radius_m
            )
    return float(np.prod(hi - lo) * inside.mean())


@dataclass(frozen=True)
class LinkFit:
    """One link's fitted primitive set and how far it protrudes past the link.

    Attributes:
        primitives: ``(shape, origin_xyz_rpy)`` per primitive; every cloud vertex
            lies inside their union by at least the declared headroom.
        kind: ``"capsule"``, ``"box"`` or ``"capsule×k"``.
        volume_ratio: Union volume ÷ the cloud's convex-hull volume (1 = exact).
        max_protrusion_m: Largest support-function excess of the union over the
            hull across ``_PROTRUSION_DIRS`` directions — the worst clearance the
            kernel under-reports for this link.
        mean_protrusion_m: The same excess averaged over the directions — the
            typical clearance the kernel under-reports.
        candidates: ``{kind: (volume_ratio, max_protrusion_m, mean_protrusion_m)}``
            for every candidate that was fitted, for the review tables.
        fitted: ``{kind: primitives}`` — every candidate's primitives, so a
            review can re-pick under another rule without refitting.
    """

    primitives: list[tuple[CollisionShape, _Origin]]
    kind: str
    volume_ratio: float
    max_protrusion_m: float
    mean_protrusion_m: float
    candidates: dict[str, tuple[float, float, float]]
    fitted: dict[str, list[tuple[CollisionShape, _Origin]]]


def _fibonacci_sphere(n: int) -> _Arr:
    import numpy as np

    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azim = math.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)], axis=1
    )


def fit_link_primitives(
    mesh: Any,
    *,
    headroom_m: float = CAPSULE_HEADROOM_M,
    max_capsules: int = 3,
    force_box: bool = False,
) -> LinkFit:
    """The primitive set that protrudes least past a mesh: one capsule, one box, or a chain.

    Geometry-source agnostic: the URDF and MJCF readers only collect each link's
    geometry into one ``trimesh.Trimesh`` (collision and visual, in the link
    frame) and hand it here. Every candidate contains every vertex by
    construction with ``headroom_m`` of clearance. The choice between them is
    the **mean protrusion**: the support-function excess of the candidate over
    the mesh's convex hull, averaged over ``_PROTRUSION_DIRS`` directions — how
    far, on average, the kernel's surface stands off the real part, which is
    the clearance a false stop is made of. Candidates within
    ``_PROTRUSION_TIE_FRACTION`` of the least are ties, and a tie goes to the
    cheaper kernel kind: a capsule↔capsule check costs 11 ns against 139 ns
    box↔box and 669 ns box↔capsule, so capsules first, then fewer capsules,
    then a box. The excess volume over the hull and the max protrusion are
    measured for every candidate and reported (``LinkFit.candidates``), not
    optimised: on the shipped robots the least-volume pick refused MORE poses
    than the least-mean-protrusion pick (UR10e 498 vs 340 per 1000,
    ``docs/reference/collision-geometry-review.md`` §3). ``force_box``
    restricts the choice to the box (a link that will carry ``tight_geometry``,
    which the kernel refines a box with).

    Args:
        mesh: The link's geometry as one ``trimesh.Trimesh`` in the link frame
            (a triangle soup is fine; only the capsule chains use the faces).
        headroom_m: Declared clearance every primitive keeps around the cloud.
        max_capsules: Longest capsule chain tried.
        force_box: Fit only the box candidate.

    Returns:
        A ``LinkFit``.
    """
    import numpy as np
    import trimesh

    pts = _hull_points(np.asarray(mesh.vertices, dtype=np.float64))
    hull = trimesh.convex.convex_hull(pts)
    hull_volume = max(float(hull.volume), 1e-12)
    dirs = _fibonacci_sphere(_PROTRUSION_DIRS)
    hull_support = (dirs @ np.asarray(hull.vertices).T).max(axis=1)

    fitted: list[
        tuple[str, int, list[tuple[CollisionShape, _Origin]]]
    ] = []  # (kind, cost rank, prims)
    fitted.append(("box", 10, [fit_obb_to_vertices(pts, headroom_m=headroom_m)]))
    if not force_box:
        cap = fit_capsule_to_vertices(pts, headroom_m=headroom_m)
        fitted.append(("capsule", 0, [cap]))
        axis = _capsule_axis(*cap)
        for k in range(2, max_capsules + 1):
            chain = _capsule_chain(mesh, axis, k, headroom_m=headroom_m)
            if chain is not None:
                fitted.append((f"capsule×{k}", k - 1, chain))

    scored: dict[str, tuple[float, float, float]] = {}
    for kind, _rank, prims in fitted:
        excess = _support(prims, dirs) - hull_support
        scored[kind] = (
            _union_volume(prims) / hull_volume,
            float(excess.max()),
            float(excess.mean()),
        )
    least = min(v[2] for v in scored.values())
    kind, _rank, prims = min(
        (c for c in fitted if scored[c[0]][2] <= least * (1.0 + _PROTRUSION_TIE_FRACTION)),
        key=lambda c: (c[1], scored[c[0]][2]),
    )
    return LinkFit(
        primitives=prims,
        kind=kind,
        volume_ratio=scored[kind][0],
        max_protrusion_m=scored[kind][1],
        mean_protrusion_m=scored[kind][2],
        candidates=scored,
        fitted={c[0]: c[2] for c in fitted},
    )


#: Decimal places a manifest renders a primitive at (``openral_cli.collision.
#: render_blocks``). A ``tight_geometry`` is expressed in its box's frame, so it
#: is derived in the box AS RENDERED — otherwise the 4-dp rounding of the box
#: origin would move the hull off the mesh it was built to contain.
_RENDER_DP = 4


def fit_link_with_tight_geometry(
    link_name: str, mesh: Any, *, headroom_m: float = CAPSULE_HEADROOM_M
) -> LinkCollisionGeometry:
    """One link as a box plus the ``tight_geometry`` (26-DOP + exact hull) refining it.

    The box is ``fit_obb_to_vertices`` (minimum volume, ``headroom_m``), with its
    origin rounded and its half-extents rounded **up** to the precision the
    manifest is written at, and the refinement is derived in that rendered
    frame (``openral_safety.tight_geometry.tight_geometry_for_box``). So
    ``mesh ⊆ hull ⊆ DOP ⊆ box`` holds for the box the kernel loads, not for an
    unrounded twin of it.

    The kernel uses it for a box only: every voxel check on this box runs the
    DOP then the hull, and a self-collision pair whose two boxes both carry a
    hull is re-asked of the two hulls (``refine_self_pair``). Against a capsule
    the box itself is what is checked, so the refinement pays off for pairs of
    refined links (``docs/reference/collision-geometry-review.md`` §8).

    Args:
        link_name: The manifest link.
        mesh: The link's geometry as one ``trimesh.Trimesh`` in the link frame.
        headroom_m: Declared clearance the box keeps around the mesh.

    Returns:
        The refined box.
    """
    import numpy as np

    from openral_safety.tight_geometry import tight_geometry_for_box

    box, origin = fit_obb_to_vertices(
        np.asarray(mesh.vertices, dtype=np.float64), headroom_m=headroom_m
    )
    scale = 10.0**_RENDER_DP
    rendered: _Origin = tuple(round(float(v), _RENDER_DP) for v in origin)  # type: ignore[assignment]  # reason: six floats in, six out
    half = tuple(math.ceil(float(h) * scale) / scale for h in box.half_extents_m)
    in_box = mesh.copy()
    in_box.apply_transform(np.linalg.inv(_xyzrpy_matrix(rendered)))
    return LinkCollisionGeometry(
        link_name=link_name,
        shape=BoxShape(half_extents_m=(half[0], half[1], half[2])),
        origin_xyz_rpy=rendered,
        tight_geometry=tight_geometry_for_box(in_box, half),
    )


def _rendered(shape: CollisionShape, origin: _Origin) -> tuple[CollisionShape, _Origin]:
    """A primitive at the precision the manifest writes it (``_RENDER_DP``).

    The lowering hands the ACM certificate and the manifest the SAME numbers the
    kernel will load: a pair proven always-colliding on the unrounded fit could
    come out of the 4-dp manifest clear at a pose, and the proof would be about
    a model the kernel never runs. The declared headroom absorbs the ≤ 0.05 mm
    shift (the enclosure tests hold every vertex ≥ 0.5 mm inside).
    """
    r = _RENDER_DP
    placed: _Origin = tuple(round(float(v), r) for v in origin)  # type: ignore[assignment]  # reason: six floats in, six out
    if isinstance(shape, BoxShape):
        hx, hy, hz = (round(float(h), r) for h in shape.half_extents_m)
        return BoxShape(half_extents_m=(hx, hy, hz)), placed
    if isinstance(shape, CapsuleShape):
        return CapsuleShape(
            radius_m=round(float(shape.radius_m), r), length_m=round(float(shape.length_m), r)
        ), placed
    return SphereShape(radius_m=round(float(shape.radius_m), r)), placed


def _fit_link(
    link_name: str, mesh: Any, tight_links: frozenset[str]
) -> list[LinkCollisionGeometry]:
    """A link's entries at manifest precision: box + hull when asked for, else the fit."""
    if link_name in tight_links:
        return [fit_link_with_tight_geometry(link_name, mesh)]
    out = []
    for shape, origin in fit_link_primitives(mesh).primitives:
        rendered_shape, rendered_origin = _rendered(shape, origin)
        out.append(
            LinkCollisionGeometry(
                link_name=link_name, shape=rendered_shape, origin_xyz_rpy=rendered_origin
            )
        )
    return out


def _tight_links_of(robot: RobotDescription, extra: Iterable[str] | None) -> frozenset[str]:
    """Links to lower as box + hull: the manifest's refined links plus ``extra``.

    Sticky on purpose: a link the manifest already refines stays refined on a
    re-lower, so ``openral collision check`` reproduces it byte for byte, and
    dropping a refinement is a deliberate manifest edit, never a side effect.
    """
    return frozenset(
        {g.link_name for g in robot.collision_geometry if g.tight_geometry is not None}
        | set(extra or ())
    )


def _origin_matrix(origin: object) -> _Arr:
    """A yourdfpy collision ``origin`` (4×4 or ``None``) as a 4×4 numpy array."""
    import numpy as np

    if origin is None:
        return np.eye(4)
    return np.asarray(origin, dtype=np.float64)


def _box_vertices(size: object) -> _Arr:
    """The 8 corners of a centred box with full extents ``size`` (sx, sy, sz)."""
    import numpy as np

    sx, sy, sz = (float(v) / 2.0 for v in size)  # type: ignore[attr-defined]  # reason: yourdfpy box.size is a float triple
    return np.array(
        [[ex, ey, ez] for ex in (-sx, sx) for ey in (-sy, sy) for ez in (-sz, sz)],
        dtype=np.float64,
    )


def _cylinder_vertices(radius: float, length: float) -> _Arr:
    """Cap rims of a polygonal prism CIRCUMSCRIBING a +Z cylinder (radius, length).

    The 24-gon's apothem is ``radius``, so the prism — and any convex primitive
    holding its vertices — contains the whole cylinder, not just sampled rim
    points (an inscribed rim would leave ``r·(1 - cos 7.5°)`` uncovered between
    samples).
    """
    import numpy as np

    n = 24
    h = length / 2.0
    circ = radius / math.cos(math.pi / n)
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([circ * np.cos(ang), circ * np.sin(ang), np.zeros_like(ang)], axis=1)
    return np.vstack([ring + np.array([0.0, 0.0, h]), ring + np.array([0.0, 0.0, -h])])


def _sphere_vertices(radius: float) -> _Arr:
    """Vertices of a polyhedron CIRCUMSCRIBING a sphere (a scaled icosphere).

    Scaled so the nearest face plane sits at ``radius``: the polyhedron, hence any
    convex primitive holding these vertices, contains the whole sphere.
    """
    import numpy as np
    import trimesh

    ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    normals = np.asarray(ico.face_normals, dtype=np.float64)
    first = np.asarray(ico.vertices, dtype=np.float64)[np.asarray(ico.faces)[:, 0]]
    inradius = float(np.einsum("ij,ij->i", normals, first).min())
    return np.asarray(ico.vertices, dtype=np.float64) * (radius / inradius)


def _apply(transform: _Arr, pts: _Arr) -> _Arr:
    """Apply a 4×4 homogeneous transform to an ``(N, 3)`` cloud."""
    import numpy as np

    return np.asarray((pts @ transform[:3, :3].T) + transform[:3, 3], dtype=np.float64)


def lower_link_geometry(
    urdf_path: str,
    *,
    tight_links: Iterable[str] = (),
    extra_meshes: Mapping[str, Any] | None = None,
) -> list[LinkCollisionGeometry]:
    """The conservative primitive set of every URDF link with geometry.

    The primitives bound the link's ``<collision>`` **and** ``<visual>``
    geometry together. A vendor's ``<collision>`` block is a simplification, and
    measured on the shipped robots it is often *smaller* than the part: H1's is a
    set of placeholder spheres/cylinders the real torso reaches 549 mm past, G1's
    shoulder cylinders miss the shoulder by 61 mm, SO-100's meshes leave out the
    servo housings (26 mm), Flexiv's 30-vertex meshes cut 3 mm into the CAD. The
    visual mesh is the CAD of the part the robot is made of, so the fit holds
    both (``docs/reference/collision-geometry-review.md``).

    Boxes map to their 8 corners, cylinders and spheres to circumscribing
    polytopes, meshes to their vertices (``trimesh``); everything is placed in
    the link frame by its ``<origin>`` and handed to ``fit_link_primitives`` —
    the same fitter the MJCF path uses — which returns one capsule, one box or a
    capsule chain, whichever has the least excess volume over the link's convex
    hull (every vertex inside with the declared headroom). A link may therefore
    yield several entries. A link named in ``tight_links`` is lowered instead
    as one box plus the exact hull refining it
    (``fit_link_with_tight_geometry``). ``extra_meshes`` adds geometry per link,
    already in the link frame, to the cloud (``lower_robot`` passes the robot's
    MJCF twin meshes, ``_twin_link_meshes``). A link whose only geometry is one
    sphere ``<collision>`` emits that exact ``SphereShape``.

    Fails closed on partial resolution: a link some of whose geometry loads and
    some does not raises, because fitting the part that loaded would under-cover
    the link. A link none of whose geometry resolves is skipped with a warning
    per missing mesh (the openarm URDF's deliberately unresolvable refs, which
    route it to the MJCF lowering).

    Raises:
        ROSConfigError: If a link's geometry resolves only partially.
    """
    tight = frozenset(tight_links)
    out: list[LinkCollisionGeometry] = []
    for link_name, item in _link_clouds(urdf_path, tight, extra_meshes):
        if isinstance(item, LinkCollisionGeometry):
            out.append(item)
        else:
            out.extend(_fit_link(link_name, item, tight))
    return out


def _link_clouds(
    urdf_path: str, tight: frozenset[str], extra_meshes: Mapping[str, Any] | None
) -> Iterator[tuple[str, LinkCollisionGeometry | Any]]:
    """Each URDF link's geometry, ready to fit: an exact sphere entry or a link-frame mesh.

    The resolution half of ``lower_link_geometry`` (which documents the rules),
    without the fit, so ``_urdf_has_collision_geometry`` can ask whether a URDF
    yields geometry at the cost of loading it rather than fitting it.

    Raises:
        ROSConfigError: If a link's geometry resolves only partially.
    """
    import trimesh

    model = _load_urdf(urdf_path)
    handler = getattr(model, "_filename_handler", None)
    extra = dict(extra_meshes or {})

    for link_name, link in model.link_map.items():  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        collisions = list(getattr(link, "collisions", None) or [])
        visuals = list(getattr(link, "visuals", None) or [])
        elements = collisions + visuals
        if not collisions:
            continue
        if (
            link_name not in tight
            and link_name not in extra
            and len(elements) == 1
            and getattr(collisions[0].geometry, "sphere", None) is not None
        ):
            sph = collisions[0].geometry.sphere
            tf = _origin_matrix(collisions[0].origin)
            cx, cy, cz = (float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3]))
            sphere, placed = _rendered(
                SphereShape(radius_m=float(sph.radius)), (cx, cy, cz, 0.0, 0.0, 0.0)
            )
            yield (
                link_name,
                LinkCollisionGeometry(link_name=link_name, shape=sphere, origin_xyz_rpy=placed),
            )
            continue

        parts: list[Any] = []
        missing: list[str] = []
        for el in elements:
            part = _collision_local_mesh(el, handler)
            if part is None or len(part.vertices) == 0:
                mesh = getattr(el.geometry, "mesh", None)
                missing.append(getattr(mesh, "filename", "<unsupported geometry>"))
                continue
            parts.append(part.apply_transform(_origin_matrix(el.origin)))
        if not parts:
            continue
        if missing:
            raise ROSConfigError(
                f"link {link_name!r}: geometry {missing} did not load while the rest of "
                "the link did; a primitive fitted to the part that loaded would "
                "under-cover the link. Resolve the refs (vendored URDFs use "
                "`rd:<module>:<relpath>`) before lowering."
            )
        if link_name in extra:
            parts.append(extra[link_name])
        cloud = trimesh.util.concatenate(parts)
        if len(cloud.vertices) < 4:
            continue
        yield link_name, cloud


def _collision_local_mesh(col: object, handler: object) -> Any | None:
    """One ``<collision>``/``<visual>`` geometry as a ``trimesh.Trimesh`` in its own frame.

    Mesh → loaded × scale; box → the box; cylinder / sphere → a circumscribing
    polytope, so a convex primitive holding its vertices holds the whole solid.
    ``None`` when a mesh file is missing (warned, never silent) or the geometry
    kind is unknown.
    """
    import os

    import numpy as np
    import trimesh

    geom = col.geometry  # type: ignore[attr-defined]  # reason: yourdfpy Collision, no stubs
    box = getattr(geom, "box", None)
    cyl = getattr(geom, "cylinder", None)
    sph = getattr(geom, "sphere", None)
    mesh = getattr(geom, "mesh", None)
    if box is not None:
        return trimesh.convex.convex_hull(_box_vertices(box.size))
    if cyl is not None:
        return trimesh.convex.convex_hull(_cylinder_vertices(float(cyl.radius), float(cyl.length)))
    if sph is not None:
        return trimesh.convex.convex_hull(_sphere_vertices(float(sph.radius)))
    if mesh is not None:
        path = handler(mesh.filename) if callable(handler) else mesh.filename
        if not os.path.isfile(path):
            # Never skip a collision link silently — an absent mesh means the link
            # would carry no geometry and go unchecked by the kernel (§1.4).
            warnings.warn(
                f"mesh not found: {mesh.filename!r}",
                stacklevel=2,
            )
            return None
        loaded = trimesh.load(path, force="mesh")
        scale = getattr(mesh, "scale", None)
        if scale is not None:
            loaded.apply_scale(np.asarray(scale, dtype=np.float64))
        return loaded
    return None


def _collision_local_vertices(col: object, handler: object) -> _Arr | None:
    """Vertices of one ``<collision>``/``<visual>`` geometry, in its own local frame."""
    import numpy as np

    mesh = _collision_local_mesh(col, handler)
    return None if mesh is None else np.asarray(mesh.vertices, dtype=np.float64)


# ── ACM: certified "always-colliding" over each pair's relative-DoF subspace ───
# A pair enters the ACM here only under the **always-colliding** justification (see
# ``acm_for_geometry``): the kernel's trip condition holds at *every* reachable
# configuration — a proof, not a sample, since "every" must mean every.
# 1. Relative pose of two links depends only on the joints between them (``panda_link5``
#    <-> ``panda_link7`` moves with ``panda_joint6``+``panda_joint7`` only, a 2-D subspace of
#    the arm's 7-D one), so it can be enumerated exhaustively rather than sampled. (The old
#    sweep drew 2000 uniform 7-D points — too sparse for a 13%-measure separated region, and
#    RNG-draw-order-dependent, so not reproducible.)
# 2. A grid + Lipschitz bound certifies the continuum: turning joint *j* by δ moves a point at
#    distance *R* from its axis by at most *R·δ*, so over a grid cell of half-width ``h_j/2``
#    the far link moves at most ``ε = Σ_j R_j·h_j/2`` relative to the near one. If the near
#    shape eroded by ε still trips against the far shape at the cell's centre node, the
#    untouched shapes trip everywhere in that cell — certify every cell and the whole subspace
#    is certified.
# Anything uncertifiable (too many relative DoF, an erosion that eats the shape, a joint the
# URDF doesn't pin down) is simply not an ACM entry — fails toward fewer exemptions, the safe
# direction: a missing entry costs a false E-stop, an unearned one hides a real collision.

# Refinement budget for the branch-and-bound in `_certified_always_colliding`.
# None of these is a soundness knob: hitting any of them makes that function
# return False, i.e. *withhold* an ACM entry. Raising them can only certify more
# pairs, never certify a wrong one — so they are tuned purely for cost.
#
# `_CERTIFY_MAX_DOF` bounds how many joints may separate a pair. g1's
# `hip_pitch`↔`torso` needs 4 (the hip pitch plus the three waist joints); nothing
# shipped needs more, and each extra DoF makes ε harder to shrink.
_CERTIFY_MAX_DOF = 5
# Live cells allowed at once. g1's `hip_pitch`↔`torso` — the hardest pair on any
# shipped robot — peaks at ~246k before pruning takes over and it certifies at
# level 23, so this leaves real headroom rather than sitting on the measurement.
_CERTIFY_MAX_CELLS = 400_000
# Depth ceiling. Each level halves one axis, so shrinking ε by 2^n across k axes
# takes about k·n levels; 96 covers a 4-DoF pair over four orders of magnitude.
_CERTIFY_MAX_LEVELS = 96
# Nodes per axis for the cheap rejection pass that runs before refinement.
_COARSE_NODES = 7

# The MJCF backend (`lower_robot_from_mjcf`) still decides always-colliding by a
# seeded random sweep: its FK comes from mujoco, so the URDF joint-tree walk the
# certificate is built on does not apply to it. It now at least asks the kernel's
# real predicate for the real shape (`shape_distance`), which is what #155 was
# about; the *criterion* there remains sampled and is tracked as residual risk in
# the hazard-log entry. Only `openarm` uses this path, and only with capsules.
_MJCF_RNG_SEED = 20260610
_MJCF_N_SAMPLES = 2000
# Gripper stroke positions a follower-coupled body is sampled at when fitting a
# leader link's capsule (``fit_collision_geometry_from_mjcf``). The capsule has
# to hold the follower at every opening; nine evenly spaced positions over a
# hinge's range leave at most range/16 between a sample and the true extreme,
# which the fit's own radius absorbs for fingers of this size.
_MJCF_STROKE_SAMPLES = 9


_GeomsByLink = Mapping[str, Sequence[LinkCollisionGeometry]]


def _by_link(
    geoms: Mapping[str, LinkCollisionGeometry | Sequence[LinkCollisionGeometry]]
    | Iterable[LinkCollisionGeometry],
) -> dict[str, list[LinkCollisionGeometry]]:
    """Group primitives by link, accepting one-per-link mappings and flat lists alike."""
    out: dict[str, list[LinkCollisionGeometry]] = {}
    items: Iterable[LinkCollisionGeometry | Sequence[LinkCollisionGeometry]] = (
        geoms.values() if isinstance(geoms, Mapping) else geoms
    )
    for entry in items:
        for g in [entry] if isinstance(entry, LinkCollisionGeometry) else entry:
            out.setdefault(g.link_name, []).append(g)
    return out


def _link_pair_distance(
    geoms_a: Sequence[LinkCollisionGeometry],
    link_a: _Arr,
    geoms_b: Sequence[LinkCollisionGeometry],
    link_b: _Arr,
) -> _Arr:
    """Kernel gap between two links: the minimum over their primitive pairs.

    ``link_a`` / ``link_b`` are ``(n, 4, 4)`` link poses; each primitive is
    placed by its own origin. This is exactly how ``check_self_collision``
    folds a link pair, so an ACM decided on it is about the kernel's check.
    """
    import numpy as np

    from openral_safety.kernel_predicates import shape_distance

    best: _Arr | None = None
    for ga in geoms_a:
        t_a = link_a @ _xyzrpy_matrix(ga.origin_xyz_rpy)
        for gb in geoms_b:
            t_b = link_b @ _xyzrpy_matrix(gb.origin_xyz_rpy)
            d = shape_distance(ga.shape, t_a, gb.shape, t_b)
            best = d if best is None else np.minimum(best, d)
    assert best is not None, "a link with no primitives cannot be paired"
    return best


def _parent_joint_map(model: object) -> dict[str, object]:
    """``child_link -> the URDF joint that drives it``. One parent per link (a tree)."""
    return {j.child: j for j in model.robot.joints}  # type: ignore[attr-defined]  # reason: yourdfpy URDF


def _ancestor_joints(model: object, link: str) -> list[object]:
    """The joints from the kinematic root down to ``link``, root-first."""
    parent_of = _parent_joint_map(model)
    chain: list[object] = []
    seen: set[str] = set()
    cur = link
    while cur in parent_of and cur not in seen:
        seen.add(cur)
        joint = parent_of[cur]
        chain.append(joint)
        cur = joint.parent  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    chain.reverse()
    return chain


def _relative_chains(
    model: object, link_a: str, link_b: str
) -> tuple[list[object], list[object]] | None:
    """Split the two ancestor chains at their common ancestor.

    Returns ``(chain_a, chain_b)`` — the joints below the deepest shared ancestor
    on each side, root-first. The relative transform ``A → B`` is
    ``inv(∏ chain_a) · (∏ chain_b)`` and depends on **no other joint**. ``None``
    when the two links are not in one tree (a disconnected graph, which
    ``envelope_loader`` refuses separately).
    """
    anc_a, anc_b = _ancestor_joints(model, link_a), _ancestor_joints(model, link_b)
    shared = 0
    while (
        shared < len(anc_a) and shared < len(anc_b) and anc_a[shared].name == anc_b[shared].name  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    ):
        shared += 1
    root_a = anc_a[0].parent if anc_a else link_a  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    root_b = anc_b[0].parent if anc_b else link_b  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    if root_a != root_b:
        return None
    return anc_a[shared:], anc_b[shared:]


_MOVABLE_JOINT_TYPES = ("revolute", "continuous", "prismatic")


def _joint_span(joint: object) -> tuple[float, float]:
    """A movable joint's ``(lower, upper)``; an unlimited revolute spans ``[-π, π]``."""
    limit = getattr(joint, "limit", None)
    lower = getattr(limit, "lower", None) if limit is not None else None
    upper = getattr(limit, "upper", None) if limit is not None else None
    if lower is None or upper is None or lower == upper:
        return -math.pi, math.pi
    return float(lower), float(upper)


def _chain_transforms(chain: list[object], values: dict[str, _Arr], n: int) -> _Arr:
    """Compose a joint chain into ``(n, 4, 4)`` transforms.

    Each joint contributes its fixed ``origin`` followed by its own motion: a
    rotation about ``axis`` for revolute/continuous, a translation along ``axis``
    for prismatic. Joints absent from ``values`` are held at zero — correct
    because ``_relative_chains`` guarantees every joint that can change the
    pair's relative transform is present.
    """
    import numpy as np

    out = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for joint in chain:
        step = np.broadcast_to(_origin_matrix(joint.origin), (n, 4, 4)).copy()  # type: ignore[attr-defined]  # reason: yourdfpy Joint
        jtype = str(joint.type)  # type: ignore[attr-defined]  # reason: yourdfpy Joint
        name = str(joint.name)  # type: ignore[attr-defined]  # reason: yourdfpy Joint
        if jtype in _MOVABLE_JOINT_TYPES and name in values:
            q = values[name]
            axis = np.asarray(joint.axis, dtype=np.float64)  # type: ignore[attr-defined]  # reason: yourdfpy Joint
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if jtype == "prismatic":
                motion[:, :3, 3] = axis[None, :] * q[:, None]
            else:  # Rodrigues rotation about the joint axis
                kx = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                c, s = np.cos(q)[:, None, None], np.sin(q)[:, None, None]
                motion[:, :3, :3] = np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)
            step = step @ motion
        out = out @ step
    return out


def _axis_radius_bound(
    chain: list[object], joint: object, geoms: Sequence[LinkCollisionGeometry]
) -> float:
    """Bound the distance from ``joint``'s axis to any point of the link's ``geoms``.

    Configuration-independent, so it is valid over the whole grid cell: rotations
    preserve lengths, so the farthest a point of the distal link's shape can lie
    from the joint's own frame origin is the sum of the link offsets below it plus
    the shape's own offset and extent. Distance to the *axis line* is never more
    than distance to a point on it, so this over-bounds — which makes ``ε`` larger
    and the certificate stricter, never looser.
    """
    import numpy as np

    from openral_safety.kernel_predicates import shape_max_extent_m

    total = 0.0
    seen = False
    for link_joint in chain:
        if seen:
            total += float(np.linalg.norm(np.asarray(_origin_matrix(link_joint.origin))[:3, 3]))  # type: ignore[attr-defined]  # reason: yourdfpy Joint
        if link_joint is joint:
            seen = True
    return total + max(
        float(np.linalg.norm(np.asarray(g.origin_xyz_rpy[:3], dtype=np.float64)))
        + shape_max_extent_m(g.shape)
        for g in geoms
    )


def _pair_relative_dofs(
    model: object, link_a: str, link_b: str, geoms: _GeomsByLink
) -> list[tuple[object, float, tuple[float, float]]] | None:
    """The joints that move ``link_a`` relative to ``link_b``, with their radius bounds.

    Returns ``[(joint, radius_bound_m, (lower, upper))]`` — everything the
    certificate needs — or ``None`` if the pair's relative pose cannot be pinned
    down (links in different trees). An **empty** list is a meaningful answer: the
    two links are rigidly related, so a single evaluation decides the pair
    exactly.
    """
    chains = _relative_chains(model, link_a, link_b)
    if chains is None:
        return None
    chain_a, chain_b = chains
    out: list[tuple[object, float, tuple[float, float]]] = []
    for chain, distal in ((chain_a, link_a), (chain_b, link_b)):
        for joint in chain:
            if str(joint.type) not in _MOVABLE_JOINT_TYPES:  # type: ignore[attr-defined]  # reason: yourdfpy Joint
                continue
            radius = 1.0  # prismatic: 1 m of travel moves the link 1 m
            if str(joint.type) != "prismatic":  # type: ignore[attr-defined]  # reason: yourdfpy Joint
                radius = _axis_radius_bound(chain, joint, geoms[distal])
            out.append((joint, radius, _joint_span(joint)))
    return out


def _certified_always_colliding(  # noqa: PLR0911  # reason: one early-out per way the proof can fail; each is a distinct, documented safe refusal
    model: object,
    geoms: _GeomsByLink,
    link_a: str,
    link_b: str,
    *,
    margin_m: float,
) -> bool:
    """Is the kernel's trip condition **provably** true at every reachable pose?

    The always-colliding justification for an ACM entry (``acm_for_geometry``)
    is only sound when the check it removes is a constant. Decided over the
    pair's relative-DoF subspace, using the kernel's own predicates at the
    robot's own margin, by branch-and-bound over boxes of joint space (not by
    sampling poses).

    Each cell is judged by one evaluation at its centre plus a Lipschitz bound:
    turning joint *j* by δ moves a point at distance *R* from its axis by at
    most *R·δ*, so within a cell every point of the far link moves at most
    ``ε = Σ_j R_j · w_j / 2`` relative to the near one, and
    ``gap(q) <= gap(centre) + ε`` for every configuration ``q`` in the cell.
    Three verdicts follow: ``gap(centre) > margin`` is a witness the pair is
    NOT always-colliding (reject immediately); ``gap(centre) + ε <= margin``
    certifies the cell (drop it); otherwise undecided — split across the axis
    that shrinks ε fastest and revisit.

    Always-colliding requires every cell to certify. Returns ``False`` on:
    exhausted refinement budget, exceeding ``_CERTIFY_MAX_DOF``, an
    undetermined relative pose, or both links declaring ``tight_geometry``
    (the kernel checks those at exact-hull fidelity, which this function's
    box-based ``shape_distance`` cannot bound in the certifying direction).
    Every failure path withholds the ACM entry: withheld-in-error costs a
    false E-stop, granted-in-error hides a real self-collision. Deterministic
    — no RNG, no dependence on evaluation order.
    """
    import numpy as np

    dofs = _pair_relative_dofs(model, link_a, link_b, geoms)
    if dofs is None or len(dofs) > _CERTIFY_MAX_DOF:
        return False

    geoms_a, geoms_b = geoms[link_a], geoms[link_b]
    if any(g.tight_geometry is not None for g in geoms_a) and any(
        g.tight_geometry is not None for g in geoms_b
    ):
        # The kernel re-asks a box pair it cannot clear of the two exact hulls
        # (`hull_hull_distance`, issue #191), and the hull gap is >= the box gap
        # everywhere. So "the box gap is <= margin at every pose" — all this
        # function can prove with `shape_distance` — no longer implies the kernel
        # trips, and certifying on it would grant an ACM entry that hides a live
        # check. Withhold instead; the pair stays checked, which is the direction
        # every other refusal here also fails in.
        return False
    chains = _relative_chains(model, link_a, link_b)
    if chains is None:  # pragma: no cover — _pair_relative_dofs already returned non-None
        return False
    chain_a, chain_b = chains
    names = [str(j.name) for j, _, _ in dofs]  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    radii = np.asarray([r for _, r, _ in dofs], dtype=np.float64)
    lo = np.asarray([span[0] for _, _, span in dofs], dtype=np.float64)
    hi = np.asarray([span[1] for _, _, span in dofs], dtype=np.float64)

    def gaps_at(centres: _Arr) -> _Arr:
        """Kernel surface gap at each row of ``centres`` (one joint vector per row).

        The minimum over the two links' primitive pairs, as the kernel folds it.
        """
        n = int(centres.shape[0])
        values = {name: centres[:, i] for i, name in enumerate(names)}
        t_a = _chain_transforms(chain_a, values, n)
        t_b = _chain_transforms(chain_b, values, n)
        return _link_pair_distance(geoms_a, t_a, geoms_b, t_b)

    if not dofs:  # rigidly related links — one evaluation settles it exactly
        return bool(gaps_at(np.zeros((1, 0)))[0] <= margin_m)

    # Cheap rejection first: most pairs on a real arm separate somewhere obvious,
    # and finding that costs one small batch instead of a refinement run.
    coarse = np.meshgrid(
        *[np.linspace(lo[i], hi[i], _COARSE_NODES) for i in range(len(dofs))], indexing="ij"
    )
    if bool(np.any(gaps_at(np.stack([g.ravel() for g in coarse], axis=1)) > margin_m)):
        return False

    centres = ((lo + hi) / 2.0)[None, :]
    widths = (hi - lo)[None, :]
    for _ in range(_CERTIFY_MAX_LEVELS):
        gaps = gaps_at(centres)
        if bool(np.any(gaps > margin_m)):
            return False  # witness: a reachable pose the kernel would not trip on
        undecided = gaps + 0.5 * (widths @ radii) > margin_m
        if not bool(undecided.any()):
            return True  # every cell certified
        cell_c, cell_w = centres[undecided], widths[undecided]
        # Split the axis whose remaining width buys the most ε reduction.
        axis = np.argmax(cell_w * radii, axis=1)
        rows = np.arange(cell_c.shape[0])
        half = cell_w[rows, axis] / 2.0
        split_w = cell_w.copy()
        split_w[rows, axis] = half
        low, high = cell_c.copy(), cell_c.copy()
        low[rows, axis] -= half / 2.0
        high[rows, axis] += half / 2.0
        centres = np.concatenate([low, high])
        widths = np.concatenate([split_w, split_w])
        if centres.shape[0] > _CERTIFY_MAX_CELLS:
            return False  # out of budget — withhold the exemption
    return False


def _xyzrpy_matrix(origin: _Origin) -> _Arr:
    """A manifest ``(x, y, z, roll, pitch, yaw)`` origin as a 4×4 matrix."""
    import numpy as np

    from openral_safety.mjcf_lowering import _rpy_to_mat

    out = np.eye(4)
    out[:3, :3] = np.asarray(_rpy_to_mat(origin[3], origin[4], origin[5])).reshape(3, 3)
    out[:3, 3] = origin[:3]
    return out


def acm_for_geometry(
    urdf_path: str,
    geoms: Mapping[str, LinkCollisionGeometry | Sequence[LinkCollisionGeometry]]
    | Iterable[LinkCollisionGeometry],
    *,
    srdf_path: str | None = None,
    margin_m: float = 0.0,
) -> _AcmPairs:
    """The self-collision ACM for a specific per-link primitive geometry ``geoms``.

    The kernel checks collisions with ``geoms``, so the ACM is decided against the
    *same* primitives, with the *same* predicates
    (``openral_safety.kernel_predicates``), at the *same* ``margin_m``. A pair
    is exempted under exactly one of three justifications: **adjacent** (directly
    joint-connected); **always-colliding** (the kernel's trip condition holds at
    every reachable configuration — a proof over the pair's relative-DoF
    subspace via ``_certified_always_colliding``, never a sample); or
    **never-able-to-collide** (hand-reviewed SRDF ``disable_collisions`` rows,
    when ``srdf_path`` is given). So with an SRDF: ``ACM = adjacent ∪ always ∪
    SRDF``; without one: ``ACM = adjacent ∪ always`` — every other pair stays
    **checked**, since nothing short of mesh ground truth or a human can retire
    a sometimes-colliding pair.

    .. warning::
       The SRDF term is **not** self-evidently "never collides". MoveIt's own
       ``reason="Never"`` rows come from its mesh sweep, but a ``reason="User"``
       row is whatever a human put there. Both are trusted here, so both are a
       safety-WG surface. ``robots/panda_mobile/panda_mobile.srdf`` carries one
       such row under protest — see its comment and issue #155.

    Deterministic: no RNG is involved anywhere in this function.

    Args:
        urdf_path: Concrete on-disk URDF path (see ``_load_urdf``).
        geoms: The primitives the kernel will load: a flat list, or a mapping by
            link name to one primitive or to the link's list of them.
        srdf_path: Optional SRDF whose ``disable_collisions`` rows are unioned in.
        margin_m: The robot's ``safety.self_collision_margin_m``. The kernel trips
            a pair at ``distance <= margin_m``, so the always-colliding proof must
            use the same threshold — a sweep pinned at ``0.0`` against a kernel
            running a negative margin would exempt pairs the kernel never trips
            on, silently deleting a live check.

    Returns:
        The disabled pairs, as unordered two-element frozensets.
    """
    model = _load_urdf(urdf_path)
    by_link = _by_link(geoms)
    links = [ln for ln in model.link_map if ln in by_link]  # type: ignore[attr-defined]  # reason: yourdfpy URDF

    disabled: _AcmPairs = set()
    for joint in model.robot.joints:  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        if joint.parent in by_link and joint.child in by_link and joint.parent != joint.child:
            disabled.add(frozenset({joint.parent, joint.child}))  # adjacent
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            if frozenset({a, b}) in disabled:
                continue  # already adjacent; no need to prove anything
            if _certified_always_colliding(model, by_link, a, b, margin_m=margin_m):
                disabled.add(frozenset({a, b}))

    if srdf_path is not None:
        disabled |= parse_srdf_disabled_pairs(srdf_path)  # hand-reviewed exemptions
    return disabled


def sample_acm_from_urdf(
    urdf_path: str,
    *,
    margin_m: float = 0.0,
) -> _AcmPairs:
    """The ACM from a URDF alone (the no-SRDF fallback).

    Lowers the URDF's own collision geometry and runs ``acm_for_geometry``
    without an SRDF, so the result is ``adjacent ∪ always-colliding`` and nothing
    else: with no mesh ground truth and no human in the loop, a pair that is only
    *sometimes* colliding stays checked.

    Args:
        urdf_path: Concrete on-disk URDF path.
        margin_m: The robot's ``safety.self_collision_margin_m`` (see
            ``acm_for_geometry``).

    Returns:
        The disabled pairs, as unordered two-element frozensets.
    """
    return acm_for_geometry(
        urdf_path, lower_link_geometry(urdf_path), srdf_path=None, margin_m=margin_m
    )


# ── Top-level entry: URDF/SRDF → manifest collision model ──────────────────────


@dataclass(frozen=True)
class LoweredCollisionModel:
    """The two manifest blocks the lowering tool produces, plus provenance.

    Attributes:
        collision_geometry: Per-link capsule/sphere (empty when ``acm_only``).
        allowed_collision_pairs: The ACM as sorted ``(link_a, link_b)`` tuples
            (empty when ``geometry_only``).
        acm_source: ``"srdf"`` when derived from an SRDF, else ``"sampling"``.
        srdf_path: The SRDF used, if any.
        joint_fk: Per-manifest-joint forward-kinematics lowered from the URDF —
            ``{joint_name: (origin_xyz, origin_rpy, axis_xyz)}`` — for joints that
            matched a URDF joint by ``child_link``. The kernel needs these to place
            the link capsules; empty when ``acm_only`` or no URDF joint matched.
    """

    collision_geometry: list[LinkCollisionGeometry] = field(default_factory=list)
    allowed_collision_pairs: list[tuple[str, str]] = field(default_factory=list)
    acm_source: str = "sampling"
    srdf_path: str | None = None
    joint_fk: dict[str, tuple[_Vec3, _Vec3, _Vec3]] = field(default_factory=dict)


def _rd_mesh_filename_handler(urdf_path: str) -> object:
    """A yourdfpy filename handler that also expands ``rd:<module>:<relpath>`` refs.

    Vendored URDFs (``openral robot vendor-urdf``) reference their meshes as
    ``rd:<robot_descriptions module>:<path relative to the upstream repository>``
    so the committed file carries no machine-specific absolute path. Expanding
    imports the module, which clones the pinned upstream into the shared
    ``robot_descriptions`` cache on first use — the same mechanism CI pre-warms.
    Every other ref falls through to yourdfpy's stock resolution (absolute paths,
    relative-to-URDF, ``package://`` heuristics) unchanged; in particular
    openarm's unresolvable ``package://openarm_description`` refs must KEEP
    failing so ``select_lowering`` keeps routing openarm to its MJCF path.
    """
    import functools
    import importlib
    import os
    from pathlib import Path

    from yourdfpy.urdf import filename_handler_magic  # reason: yourdfpy ships no stubs

    fallback = functools.partial(filename_handler_magic, dir=os.path.dirname(urdf_path))

    def handler(fname: str) -> str:
        if fname.startswith("rd:"):
            module, _, rel = fname[len("rd:") :].partition(":")
            mod = importlib.import_module(f"robot_descriptions.{module}")
            return str(Path(mod.REPOSITORY_PATH) / rel)
        return str(fallback(fname))

    return handler


def _load_urdf(urdf_path: str) -> object:
    """Load a yourdfpy model from a concrete on-disk URDF file path.

    The asset grammar is resolved upstream by
    ``openral_core.assets.resolve_asset`` (``rd:`` modules download their
    pre-expanded URDF, ``file:`` refs resolve against the manifest dir), so this
    helper only loads a real file — no URI dispatch beyond the vendored-mesh
    ``rd:<module>:<relpath>`` refs ``_rd_mesh_filename_handler`` expands.
    Collision-scene-graph build + collision meshes on, visual meshes off,
    identical to the previous loader.
    """
    import yourdfpy  # reason: yourdfpy ships no stubs; mypy.ini ignores its imports

    return yourdfpy.URDF.load(
        urdf_path,
        build_collision_scene_graph=True,
        load_meshes=False,
        load_collision_meshes=True,
        filename_handler=_rd_mesh_filename_handler(urdf_path),
    )


def lower_joint_fk(robot: RobotDescription, urdf_ref: str) -> dict[str, tuple[_Vec3, _Vec3, _Vec3]]:
    """Per-manifest-joint FK (``origin_xyz``, ``origin_rpy``, ``axis_xyz``) from the URDF.

    The kernel computes link poses from the manifest joints' fixed parent→joint
    transform + axis; a manifest that only declares the chain topology
    (parent/child) needs these populated. For each manifest joint the fixed
    ``origin`` is the URDF transform from the manifest ``parent_link`` to its
    ``child_link`` at the zero configuration — computed via the URDF's own forward
    kinematics, so it is correct even when the URDF inserts intermediate links
    between them (e.g. UR's non-identity ``base_link_inertia``). The ``axis`` is the
    matching URDF joint's axis (in the child frame). Returns ``{joint_name: (xyz,
    rpy, axis)}`` for joints whose ``parent_link`` AND ``child_link`` both exist in
    the URDF; unmatched joints (a synthetic gripper, a base DoF the URDF lacks) are
    omitted and keep their manifest defaults.
    """
    import numpy as np

    model = _load_urdf(urdf_ref)
    urdf_links: set[str] = set(model.link_map)  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    by_child: dict[str, object] = {j.child: j for j in model.robot.joints}  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    model.update_cfg(np.zeros(model.num_actuated_joints))  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    out: dict[str, tuple[_Vec3, _Vec3, _Vec3]] = {}
    for joint in robot.joints:
        if joint.parent_link not in urdf_links or joint.child_link not in urdf_links:
            continue
        t_parent = np.asarray(model.get_transform(joint.parent_link), dtype=np.float64)  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        t_child = np.asarray(model.get_transform(joint.child_link), dtype=np.float64)  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        tf = np.linalg.inv(t_parent) @ t_child  # fixed parent→child transform at q=0
        xyz: _Vec3 = (float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3]))
        roll, pitch, yaw = _mat_to_rpy(tf[:3, :3])
        uj = by_child.get(joint.child_link)
        axis_raw = getattr(uj, "axis", None) if uj is not None else None
        if axis_raw is None:
            axis: _Vec3 = (0.0, 0.0, 1.0)
        else:
            a = np.asarray(axis_raw, dtype=np.float64)
            axis = (float(a[0]), float(a[1]), float(a[2]))
        out[joint.name] = (xyz, (roll, pitch, yaw), axis)
    return out


def _mjcf_link_bodies(robot: RobotDescription, model: object) -> dict[str, int]:
    """Manifest link name → MJCF body id.

    The two naming schemes can diverge (openarm's manifest ``link0`` / ``link7``
    are the MJCF's ``base_link`` / ``ee_base_link``), so map via the joint
    correspondence (``sim_joint_name``, else the manifest joint name → MJCF
    joint → its child body), which is unambiguous, and fall back to a direct
    name match for anything else. A link that maps neither way is absent from
    the result.
    """
    import mujoco

    m: Any = model
    link_body: dict[str, int] = {}
    for j in robot.joints:
        # `sim_joint_name` when the manifest wires a twin; else a joint of the
        # manifest's own name (the vendored menagerie twins keep URDF names).
        ji = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or j.name))
        if ji < 0:
            continue
        child_b = int(m.jnt_bodyid[ji])
        link_body[j.child_link] = child_b
        # The parent link maps to the MJCF body's parent (root link of the chain).
        if j.parent_link not in link_body:
            link_body[j.parent_link] = int(m.body_parentid[child_b])
    for ln in {g.link_name for g in robot.collision_geometry} | {
        j.parent_link for j in robot.joints
    }:
        bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln))
        if ln not in link_body and bid >= 0:
            link_body[ln] = bid
    return link_body


def _geom_local_mesh(model: object, geom: int) -> Any:
    """One MJCF geom as a ``trimesh.Trimesh`` in the geom's own frame.

    A mesh gives its vertices and faces; a box / sphere / cylinder / capsule
    the hull of a circumscribing polytope (the same constructions the URDF
    reader uses), so a convex primitive holding it holds the solid. Visual-only
    geoms count: the collision meshes under-cover the parts
    (``lower_link_geometry``).

    Raises:
        ROSConfigError: On a geom type this reader cannot bound (a plane, an
            hfield, an sdf) — fitting the rest would under-cover the link.
    """
    import mujoco
    import numpy as np
    import trimesh

    m: Any = model
    kind = int(m.geom_type[geom])
    size = np.asarray(m.geom_size[geom], dtype=np.float64)
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(m.geom_dataid[geom])
        v0, nv = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        f0, nf = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
        return trimesh.Trimesh(
            vertices=np.asarray(m.mesh_vert[v0 : v0 + nv], dtype=np.float64),
            faces=np.asarray(m.mesh_face[f0 : f0 + nf]),
            process=False,
        )
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return trimesh.convex.convex_hull(_box_vertices(2.0 * size))
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return trimesh.convex.convex_hull(_sphere_vertices(float(size[0])))
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return trimesh.convex.convex_hull(_cylinder_vertices(float(size[0]), 2.0 * float(size[1])))
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = float(size[0]), float(size[1])
        ends = _sphere_vertices(r)
        lift = np.array([0.0, 0.0, h])
        return trimesh.convex.convex_hull(
            np.vstack([ends + lift, ends - lift, _cylinder_vertices(r, 2.0 * h)])
        )
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
    raise ROSConfigError(
        f"MJCF geom {name!r} has type {mujoco.mjtGeom(kind).name}, which the collision "
        "fit cannot bound; give the body a mesh or primitive geom or exclude it"
    )


def _body_mesh(model: object, data: object, body_id: int) -> Any | None:
    """Every geom of MJCF body ``body_id`` as one ``trimesh.Trimesh`` in the body frame.

    Collision **and** visual geoms: the visual mesh is the CAD of the part, and
    the shipped collision meshes are simplifications that under-cover it
    (``docs/reference/collision-geometry-review.md`` §3).
    """
    import numpy as np
    import trimesh

    m: Any = model
    d: Any = data
    body_rot = d.xmat[body_id].reshape(3, 3)
    body_pos = d.xpos[body_id]
    parts = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != body_id:
            continue
        part = _geom_local_mesh(m, g)
        world = (d.geom_xmat[g].reshape(3, 3) @ np.asarray(part.vertices).T).T + d.geom_xpos[g]
        parts.append(
            trimesh.Trimesh(vertices=(world - body_pos) @ body_rot, faces=part.faces, process=False)
        )
    return trimesh.util.concatenate(parts) if parts else None


def mjcf_coupled_joints(model: object) -> dict[int, list[tuple[int, float, float]]]:
    """Joint equalities as a two-way map: joint id → ``[(other joint, c0, c1)]``.

    MuJoCo's joint equality is ``q[obj1] = c0 + c1·q[obj2]`` (higher terms
    zero). Which side the manifest drives is arbitrary — the OpenArm's is
    ``obj1`` (``finger_joint1 = finger_joint2``) — so each coupling is returned
    from both ends: driving ``obj2`` gives ``obj1 = c0 + c1·q``, driving
    ``obj1`` gives ``obj2 = (q - c0) / c1``. A non-linear equality (``c2..c4``
    non-zero) or ``c1 == 0`` cannot be inverted and raises ``ROSConfigError``
    rather than being silently ignored.
    """
    import mujoco

    m: Any = model
    coupled: dict[int, list[tuple[int, float, float]]] = {}
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if b < 0:
            continue  # joint pinned to a constant: nothing to follow
        c = [float(v) for v in m.eq_data[e][:5]]
        if any(abs(v) > 0.0 for v in c[2:]) or c[1] == 0.0:
            raise ROSConfigError(
                f"joint equality {e} ({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, a)} = "
                f"poly({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, b)})) is not an "
                "invertible linear coupling; the collision fit cannot sweep it"
            )
        coupled.setdefault(b, []).append((a, c[0], c[1]))
        coupled.setdefault(a, []).append((b, -c[0] / c[1], 1.0 / c[1]))
    return coupled


def _stroke_swept_meshes(
    model: object,
    data: object,
    body: int,
    leader: int,
    followers: list[tuple[int, float, float]],
    stroke_samples: int,
) -> list[Any]:
    """Meshes of ``body`` and its equality followers across ``leader``'s stroke.

    ``stroke_samples`` evenly spaced leader positions over the joint range, each
    follower set to ``c0 + c1·q``; every mesh expressed in ``body``'s frame.
    Leaves ``data`` posed at the last sample — the caller restores it.
    """
    import mujoco
    import numpy as np
    import trimesh

    m: Any = model
    d: Any = data
    lo, hi = (float(v) for v in m.jnt_range[leader])
    parts: list[Any] = []
    for s in np.linspace(lo, hi, max(stroke_samples, 2)):
        mujoco.mj_resetData(m, d)
        d.qpos[int(m.jnt_qposadr[leader])] = s
        for follower, c0, c1 in followers:
            d.qpos[int(m.jnt_qposadr[follower])] = c0 + c1 * s
        mujoco.mj_kinematics(m, d)
        inner = _body_mesh(m, d, body)
        if inner is not None:
            parts.append(inner)
        rot_b = d.xmat[body].reshape(3, 3)
        for follower, _c0, _c1 in followers:
            fbody = int(m.jnt_bodyid[follower])
            fmesh = _body_mesh(m, d, fbody)
            if fmesh is None:
                continue
            # follower body frame -> leader's child body frame
            world = (d.xmat[fbody].reshape(3, 3) @ np.asarray(fmesh.vertices).T).T + d.xpos[fbody]
            parts.append(
                trimesh.Trimesh(
                    vertices=(world - d.xpos[body]) @ rot_b, faces=fmesh.faces, process=False
                )
            )
    return parts


def fit_collision_geometry_from_mjcf(
    robot: RobotDescription,
    *,
    manifest_dir: Path | None = None,
    stroke_samples: int = _MJCF_STROKE_SAMPLES,
    tight_links: Iterable[str] = (),
) -> list[LinkCollisionGeometry]:
    """Fit each manifest link's primitive set to its MJCF geometry.

    For robots whose MJCF collision geometry is meshes, which the primitive
    lowering (``mjcf_lowering``) skips. Each manifest link maps to an MJCF body
    by the same joint correspondence the ACM lowering uses; the body's geoms —
    collision **and** visual, meshes and primitives — are collected in the body
    frame (``_body_mesh``) and handed to ``fit_link_primitives``, the same
    fitter the URDF path uses; a link in ``tight_links`` becomes one box plus the
    exact hull refining it (``fit_link_with_tight_geometry``). A link with no
    geometry gets no primitive (e.g. openarm's ``openarm_base``, which is
    MuJoCo's ``world``).

    A body driven by an equality-coupled FOLLOWER joint (openarm's second
    finger, ``finger_joint2 = finger_joint1``) is not in the manifest's
    kinematic chain, so its vertices are folded into the LEADER joint's child
    link: sampled at ``stroke_samples`` evenly spaced leader positions across
    the joint's range, with the follower set by the equality's polynomial, and
    expressed in the leader's child body frame. The primitive therefore encloses
    both fingers at every opening the gripper can reach.

    Raises:
        ROSConfigError: If the robot has no resolvable MJCF.
    """
    import mujoco
    import trimesh
    from openral_core.assets import AssetRefError, resolve_asset

    if not robot.assets.mjcf:
        raise ROSConfigError(f"{robot.name}: no assets.mjcf to fit collision geometry from")
    try:
        mjcf_path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(f"{robot.name}: {exc}") from exc
    if mjcf_path is None:
        raise ROSConfigError(f"{robot.name}: assets.mjcf={robot.assets.mjcf!r} did not resolve")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    link_body = _mjcf_link_bodies(robot, model)

    # Equality-coupled followers of each joint, from whichever side it is driven.
    followers = mjcf_coupled_joints(model)

    def at_rest() -> None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_kinematics(model, data)

    at_rest()
    geometry: list[LinkCollisionGeometry] = []
    # Parents too: a chain's root link (each OpenArm arm's ``link0``) is only
    # ever a parent, and a bimanual robot has one root per arm.
    ordered_links = [ln for j in robot.joints for ln in (j.parent_link, j.child_link)]
    seen: set[str] = set()
    for link in ordered_links:
        if link in seen or link not in link_body:
            continue
        seen.add(link)
        body = link_body[link]
        mesh = _body_mesh(model, data, body)
        leader_joint = next(
            (
                j
                for j in robot.joints
                if j.child_link == link
                and j.sim_joint_name
                and int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name))
                in followers
            ),
            None,
        )
        if leader_joint is not None and leader_joint.sim_joint_name:
            leader = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, leader_joint.sim_joint_name)
            )
            parts = [] if mesh is None else [mesh]
            parts += _stroke_swept_meshes(
                model, data, body, leader, followers[leader], stroke_samples
            )
            at_rest()
            mesh = trimesh.util.concatenate(parts) if parts else None
        if mesh is None:
            continue
        geometry.extend(_fit_link(link, mesh, frozenset(tight_links)))
    return geometry


def lower_robot_from_mjcf(  # noqa: PLR0912, PLR0915  # reason: one cohesive MJCF lowering pass (load → link-map → FK → sweep → ACM)
    robot: RobotDescription,
    *,
    n_samples: int = _MJCF_N_SAMPLES,
    seed: int = _MJCF_RNG_SEED,
    margin_m: float = 0.0,
    manifest_dir: Path | None = None,
    acm_only: bool = False,
    tight_links: Iterable[str] | None = None,
) -> LoweredCollisionModel:
    """Lower geometry, joint FK and the sampling ACM from a robot's MJCF.

    For MJCF-native robots with **no URDF** whose collision geoms are meshes (which
    ``mjcf_lowering``'s primitive path skips) — e.g. the bimanual ``openarm``.
    Geometry is fitted to the MJCF meshes by the same fitter the URDF path uses
    (``fit_collision_geometry_from_mjcf``), or kept from the manifest under
    ``acm_only`` exactly as ``lower_robot`` keeps it; FK is the MJCF transform
    from each joint's ``parent_link`` to its ``child_link`` at the rest pose
    (matched to the MJCF by ``sim_joint_name``), and the ACM is the primitive
    sweep run with **mujoco forward kinematics** over that geometry.
    ``acm_source = "mjcf"``. ``tight_links`` adds links to lower as box + hull
    on top of those the manifest already refines (``_tight_links_of``).

    The sweep never auto-disables a pair whose two links both carry a hull:
    the kernel re-asks that pair of the exact hulls, which the box overlap it
    measures says nothing about (the same withholding
    ``_certified_always_colliding`` applies on the URDF path).

    ``manifest_dir`` resolves a ``file:`` MJCF ref against the manifest's own
    directory (no in-tree robot uses one today, but the resolver honours it).

    Raises:
        ROSConfigError: If the robot has no ``assets.mjcf`` ref, or it cannot be
            resolved to a file.
    """
    import mujoco
    import numpy as np
    from openral_core.assets import AssetRefError, resolve_asset

    if not robot.assets.mjcf:
        raise ROSConfigError(f"{robot.name}: no urdf and no assets.mjcf to lower from")
    try:
        mjcf_path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(f"{robot.name}: {exc}") from exc
    if mjcf_path is None:  # mjcf never yields the ros2:// dynamic marker
        raise ROSConfigError(f"{robot.name}: assets.mjcf={robot.assets.mjcf!r} did not resolve")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))

    def jid(name: str | None) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) if name else -1

    def body_tf(i: int) -> _Arr:
        tf = np.eye(4)
        tf[:3, :3] = data.xmat[i].reshape(3, 3)
        tf[:3, 3] = data.xpos[i]
        return tf

    link_body = _mjcf_link_bodies(robot, model)
    # The joint FK and the ACM sweep below run over the geometry the kernel will
    # load: the manifest's under ``acm_only``, else the fresh fit.
    geometry_in = (
        list(robot.collision_geometry)
        if acm_only
        else fit_collision_geometry_from_mjcf(
            robot, manifest_dir=manifest_dir, tight_links=_tight_links_of(robot, tight_links)
        )
    )

    # Joint FK: parent→child transform at the rest pose, axis from the MJCF joint.
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    joint_fk: dict[str, tuple[_Vec3, _Vec3, _Vec3]] = {}
    for j in robot.joints:
        if j.parent_link not in link_body or j.child_link not in link_body:
            continue
        tf = np.linalg.inv(body_tf(link_body[j.parent_link])) @ body_tf(link_body[j.child_link])
        roll, pitch, yaw = _mat_to_rpy(tf[:3, :3])
        ji = jid(j.sim_joint_name)
        if ji >= 0 and int(model.jnt_type[ji]) in hinge_slide:
            ax = model.jnt_axis[ji]
            axis: _Vec3 = (float(ax[0]), float(ax[1]), float(ax[2]))
        else:
            axis = (0.0, 0.0, 1.0)
        xyz: _Vec3 = (float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3]))
        joint_fk[j.name] = (xyz, (roll, pitch, yaw), axis)

    # ACM sweep over the manifest geometry, using mujoco FK for link placement.
    geoms = _by_link(g for g in geometry_in if g.link_name in link_body)
    links = list(geoms)
    # Each link's primitives are placed by mujoco FK and the pair gap is the
    # kernel's own predicate on the actual shapes, folded over the two links'
    # primitives as `check_self_collision` folds it (issue #155).
    sweep: list[tuple[int, float, float]] = []
    for j in robot.joints:
        ji = jid(j.sim_joint_name)
        if ji < 0 or int(model.jnt_type[ji]) not in hinge_slide:
            continue
        adr = int(model.jnt_qposadr[ji])
        if int(model.jnt_limited[ji]):
            lo, hi = float(model.jnt_range[ji][0]), float(model.jnt_range[ji][1])
        else:
            lo, hi = -math.pi, math.pi
        sweep.append((adr, lo, hi))

    disabled: _AcmPairs = set()
    for j in robot.joints:
        if j.parent_link in geoms and j.child_link in geoms and j.parent_link != j.child_link:
            disabled.add(frozenset({j.parent_link, j.child_link}))  # adjacent
    # Deliberate hand exemptions come in through the manifest SRDF — the same
    # explicit, reviewable channel URDF robots use — never as hand edits to the
    # generated ACM block. The sweep below can only prove "always-colliding";
    # exemptions for poses the robot must reach (e.g. openarm's folded gripper
    # resting beside the forearm) are a safety-WG judgment the SRDF records with
    # per-pair ``reason`` attributes.
    if robot.assets.srdf:
        try:
            resolved_srdf = resolve_asset(robot.assets.srdf, "srdf", manifest_dir=manifest_dir)
        except AssetRefError as exc:
            raise ROSConfigError(f"{robot.name}: {exc}") from exc
        if resolved_srdf is not None:
            disabled |= parse_srdf_disabled_pairs(str(resolved_srdf))

    hulled = {
        ln
        for ln, gs in geoms.items()
        if any(g.tight_geometry is not None and g.tight_geometry.hull_vertices_m for g in gs)
    }
    rng = np.random.default_rng(seed)
    # Pose every link at every sample first, then ask each pair's gap once over
    # the whole (n_samples, 4, 4) batch: the predicates are row-wise, so this is
    # the per-sample answer at a fraction of the per-call overhead.
    world = {ln: np.empty((n_samples, 4, 4)) for ln in links}
    for s in range(n_samples):
        mujoco.mj_resetData(model, data)
        for adr, lo, hi in sweep:
            data.qpos[adr] = lo + (hi - lo) * rng.random()
        mujoco.mj_kinematics(model, data)
        for ln in links:
            world[ln][s] = body_tf(link_body[ln])
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            # Conservative (no SRDF ground truth): disable only ALWAYS-colliding
            # capsule junctions. Never-collide pairs stay CHECKED — a sweep can't
            # prove a cross-branch bimanual pair never collides (it can miss the
            # tail), so we never auto-disable one.
            if a in hulled and b in hulled:
                continue  # a pair the kernel re-asks of hulls
            gap = _link_pair_distance(geoms[a], world[a], geoms[b], world[b])
            if bool(np.all(gap <= margin_m)):  # always-colliding
                disabled.add(frozenset({a, b}))

    return LoweredCollisionModel(
        collision_geometry=list(geometry_in),
        allowed_collision_pairs=_scoped_sorted_pairs(disabled, set(links)),
        acm_source="mjcf",
        joint_fk=joint_fk,
    )


def _scoped_sorted_pairs(pairs: _AcmPairs, links: set[str]) -> list[tuple[str, str]]:
    """Filter to pairs whose both links carry geometry; deterministic sorted output."""
    out: list[tuple[str, str]] = []
    for p in pairs:
        a, b = sorted(p)
        if a in links and b in links:
            out.append((a, b))
    return sorted(out)


#: How far (metres, and radians) a twin body frame may sit from its manifest
#: link frame for ``_twin_link_meshes`` to fold the twin's meshes in.
_TWIN_FRAME_TOL = 1e-6


def _twin_link_meshes(
    robot: RobotDescription, urdf_model: object, manifest_dir: Path | None
) -> dict[str, Any]:
    """The robot's MJCF twin meshes per manifest link, in the link frame — when frames agree.

    A URDF-lowered robot is also run as its MJCF twin (``deploy sim``), and the
    twin's CAD is not always the URDF's: the menagerie H1's forearm reaches
    46 mm past the h1_description one, its ankle 25 mm wider. The kernel checks
    the twin with the same primitives, so they have to hold both. Only mesh
    geoms count (the CAD); a twin's primitive geoms are the simulator's
    contact proxies, not the robot.

    Folded in only when every mapped twin body frame coincides with its link
    frame at the zero configuration (``_TWIN_FRAME_TOL``, relative to the first
    mapped link): a twin in another frame convention (the menagerie UR arms put
    body origins off the URDF joint origins) would be misplaced, so it is left
    out and logged — the URDF geometry still bounds the robot.

    Returns:
        ``{link_name: trimesh.Trimesh}`` in the link frame; empty when the robot
        has no twin, ``mujoco`` is not installed, or the frames disagree.
    """
    import numpy as np
    import structlog
    import trimesh
    from openral_core.assets import AssetRefError, resolve_asset

    log = structlog.get_logger(__name__)
    if not robot.assets.mjcf:
        return {}
    try:
        import mujoco

        mjcf_path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
    except (ImportError, AssetRefError) as exc:
        log.warning("collision.twin_unavailable", robot=robot.name, reason=str(exc))
        return {}
    if mjcf_path is None:
        return {}
    model: Any = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data: Any = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    link_body = _mjcf_link_bodies(robot, model)
    urdf: Any = urdf_model
    mapped = [ln for ln in link_body if ln in urdf.link_map]
    if not mapped:
        return {}

    def body_tf(b: int) -> _Arr:
        tf = np.eye(4)
        tf[:3, :3] = np.asarray(data.xmat[b]).reshape(3, 3)
        tf[:3, 3] = data.xpos[b]
        return tf

    ref = mapped[0]
    mj_ref = np.linalg.inv(body_tf(link_body[ref]))
    urdf_ref = np.linalg.inv(np.asarray(urdf.get_transform(ref), dtype=np.float64))
    worst = 0.0
    for ln in mapped:
        rel_mj = mj_ref @ body_tf(link_body[ln])
        rel_urdf = urdf_ref @ np.asarray(urdf.get_transform(ln), dtype=np.float64)
        worst = max(worst, float(np.abs(rel_mj - rel_urdf).max()))
    if worst > _TWIN_FRAME_TOL:
        log.info(
            "collision.twin_frames_differ",
            robot=robot.name,
            max_frame_error=round(worst, 6),
            note="twin meshes not folded into the fit; the URDF geometry bounds the robot",
        )
        return {}
    out: dict[str, Any] = {}
    for ln in mapped:
        body = link_body[ln]
        rot_b = np.asarray(data.xmat[body]).reshape(3, 3)
        parts = []
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != body or int(model.geom_type[g]) != int(
                mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            part = _geom_local_mesh(model, g)
            world = (np.asarray(data.geom_xmat[g]).reshape(3, 3) @ np.asarray(part.vertices).T).T
            world = world + data.geom_xpos[g]
            parts.append(
                trimesh.Trimesh(
                    vertices=(world - data.xpos[body]) @ rot_b, faces=part.faces, process=False
                )
            )
        if parts:
            out[ln] = trimesh.util.concatenate(parts)
    return out


def lower_robot(
    robot: RobotDescription,
    *,
    srdf_path: str | None = None,
    acm_only: bool = False,
    geometry_only: bool = False,
    manifest_dir: Path | None = None,
    tight_links: Iterable[str] | None = None,
) -> LoweredCollisionModel:
    """Lower a robot's URDF/SRDF into the manifest collision blocks.

    ACM source precedence: an explicit ``srdf_path`` → the manifest's
    ``assets.srdf`` → the URDF random-pose sampling fallback. The ACM is scoped to
    links that carry geometry, so an SRDF's hand/finger rows don't leak into an
    arm-only model. ``acm_only`` / ``geometry_only`` restrict the output so
    hand-tuned geometry on an existing safety robot isn't churned when only the ACM
    needs refreshing.

    Args:
        robot: The robot manifest (must declare ``assets.urdf`` or a sim MJCF).
        srdf_path: Override SRDF (a resolved file path); falls back to the
            manifest's ``assets.srdf`` ref then sampling.
        acm_only: Emit only ``allowed_collision_pairs`` (keep existing geometry).
        geometry_only: Emit only ``collision_geometry`` (skip the ACM).
        manifest_dir: Directory the manifest was loaded from; ``file:`` URDF /
            SRDF refs (the vendored arms, every in-tree SRDF) resolve against it.
        tight_links: Links to lower as box + exact hull, on top of the links
            the manifest already refines (``_tight_links_of``).

    Returns:
        A ``LoweredCollisionModel``.

    Raises:
        ROSConfigError: If ``assets.urdf`` is unset (and no sim MJCF) or a
            declared asset ref does not resolve.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if robot.assets.urdf is None:
        # MJCF-native robots (no URDF; mesh collision) lower from their sim MJCF.
        if robot.assets.mjcf:
            return lower_robot_from_mjcf(
                robot, manifest_dir=manifest_dir, acm_only=acm_only, tight_links=tight_links
            )
        raise ROSConfigError(f"{robot.name}: assets.urdf is required to lower a collision model")
    try:
        urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(f"{robot.name}: {exc}") from exc
    if urdf is None:  # ros2://robot_description — no static file to lower from
        raise ROSConfigError(
            f"{robot.name}: assets.urdf.ref={robot.assets.urdf.ref!r} is the dynamic "
            "robot_description marker (no file); cannot lower a collision model from it."
        )
    urdf_ref = str(urdf)

    # Links the manifest actually models (its kinematic chain). Generated geometry
    # is scoped to these so an orphan URDF link (e.g. panda_leftfinger, absent from
    # a manifest that models a single panda_finger_pair) can't reach the kernel.
    chain_links = {j.parent_link for j in robot.joints} | {j.child_link for j in robot.joints}

    geometry: list[LinkCollisionGeometry] = []
    joint_fk: dict[str, tuple[_Vec3, _Vec3, _Vec3]] = {}
    if not acm_only:
        geometry = [
            g
            for g in lower_link_geometry(
                urdf_ref,
                tight_links=_tight_links_of(robot, tight_links),
                extra_meshes=_twin_link_meshes(robot, _load_urdf(urdf_ref), manifest_dir),
            )
            if g.link_name in chain_links
        ]
        if not geometry:
            # Refuse to emit an empty collision model. Zero fitted links means the
            # URDF's <collision> meshes did not resolve on this host (each one
            # warned above) — proceeding would cascade into an empty ACM and a
            # kernel that checks nothing, silently (§1.4). A robot without a
            # usable URDF lowers from its MJCF via select_lowering, never here.
            raise ROSConfigError(
                f"{robot.name}: URDF produced zero collision geometry — its "
                f"<collision> meshes likely did not resolve on this host "
                f"(urdf={urdf_ref}). Refusing to lower an empty collision model."
            )
        joint_fk = lower_joint_fk(robot, urdf_ref)

    pairs: list[tuple[str, str]] = []
    source = "sampling"
    # ACM source precedence: explicit srdf_path override → manifest assets.srdf →
    # sampling. Only resolve the SRDF ref when the ACM is actually produced
    # (geometry_only emits no ACM, so a missing/absent SRDF must not block it).
    used_srdf: str | None = srdf_path
    if not geometry_only:
        if used_srdf is None and robot.assets.srdf:
            try:
                resolved_srdf = resolve_asset(robot.assets.srdf, "srdf", manifest_dir=manifest_dir)
            except AssetRefError as exc:
                raise ROSConfigError(f"{robot.name}: {exc}") from exc
            used_srdf = str(resolved_srdf) if resolved_srdf is not None else None
        # The kernel checks collisions with the SAME capsules it will load: the
        # existing manifest geometry under acm_only, else the freshly lowered set.
        # The ACM is computed against that geometry so a mesh-based SRDF's omitted
        # capsule-junction pairs (always-colliding under the conservative capsules)
        # are added — otherwise the kernel would false-E-stop every step.
        geom_list = list(robot.collision_geometry if acm_only else geometry)
        # The kernel trips at `safety.self_collision_margin_m`, so the
        # always-colliding proof must use that same threshold — see
        # `acm_for_geometry`. so101_follower runs -0.06 m; a sweep hardcoded at
        # 0.0 would have called pairs always-colliding that the kernel never
        # trips on, and exempted a check that was doing real work.
        disabled = acm_for_geometry(
            urdf_ref,
            geom_list,
            srdf_path=used_srdf,
            margin_m=float(getattr(robot.safety, "self_collision_margin_m", 0.0) or 0.0),
        )
        source = "srdf" if used_srdf else "sampling"
        pairs = _scoped_sorted_pairs(disabled, {g.link_name for g in geom_list})

    return LoweredCollisionModel(
        collision_geometry=geometry,
        allowed_collision_pairs=pairs,
        acm_source=source,
        srdf_path=used_srdf,
        joint_fk=joint_fk,
    )


# ── Provenance-correct dispatch: pick the lowering source per robot ─────────────

LoweringSource = Literal["srdf", "sampling", "mjcf"]
"""Which lowering path a robot resolves to (matches ``LoweredCollisionModel.acm_source``)."""


def _resolved_urdf_path(robot: RobotDescription, manifest_dir: Path | None) -> str | None:
    """The robot's URDF as a concrete on-disk path, or ``None`` if it has no static URDF.

    Returns ``None`` for a robot with no ``assets.urdf`` and for the
    ``ros2://robot_description`` dynamic marker (a runtime topic, not a file) —
    in both cases there is no URDF file to lower geometry from.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if robot.assets.urdf is None:
        return None
    try:
        urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(f"{robot.name}: {exc}") from exc
    return None if urdf is None else str(urdf)


def _urdf_has_collision_geometry(urdf_path: str) -> bool:
    """True iff the URDF yields at least one collision capsule/sphere.

    The discriminator between the URDF-sampling path and the MJCF path for a
    robot that declares both but no SRDF. A URDF whose ``<collision>`` meshes do
    not resolve on disk (e.g. ``openarm``'s ``package://`` refs) lowers to *zero*
    geometry; such a robot must lower from its MJCF instead, where its
    hand-authored manifest capsules are kept. ``lower_link_geometry`` warns per
    missing mesh, so the unusable URDF is never silently dropped.

    Resolves every link (so a partially resolving link raises here exactly as
    in ``lower_link_geometry``) but fits none: the answer is whether any link
    yields geometry, which does not depend on the fit.
    """
    return len(list(_link_clouds(urdf_path, frozenset(), None))) > 0


def select_lowering(robot: RobotDescription, *, manifest_dir: Path | None = None) -> LoweringSource:
    """Pick the provenance-correct lowering source for ``robot``.

    Deterministic routing that reproduces each robot's *committed* collision
    source exactly — a drift here changes what the C++ safety kernel checks, so
    the choice is explicit, not the old ``urdf if assets.urdf else mjcf`` guess:

    * ``"srdf"`` — an SRDF **and** a URDF with usable collision geometry: the
      SRDF's ``disable_collisions`` is the mesh-proven ACM ground truth
      (franka_panda, panda_mobile, rizon4, ur5e, ur10e).
    * ``"sampling"`` — a URDF (no SRDF) whose ``<collision>`` meshes resolve to
      usable geometry: the MoveIt-style random-pose ACM sweep over the
      URDF-fitted capsules (g1, h1, so100_follower, so101_follower).
    * ``"mjcf"`` — no usable URDF geometry but an MJCF exists: keep the
      manifest's hand-authored capsules and sweep the ACM with mujoco FK
      (openarm, whose vendored URDF's collision meshes are ``package://`` refs
      that don't resolve). An SRDF on such a robot does NOT flip it to the URDF
      path (there is no geometry to lower there); instead
      ``lower_robot_from_mjcf`` unions the SRDF's ``disable_collisions``
      into its sweep, so deliberate hand exemptions carry an explicit,
      reviewable paper trail.

    Raises:
        ROSConfigError: If the robot declares no lowerable asset (no URDF/SRDF
            file and no MJCF), or a declared ref does not resolve.
    """
    urdf_path = _resolved_urdf_path(robot, manifest_dir)
    urdf_usable = urdf_path is not None and _urdf_has_collision_geometry(urdf_path)
    if robot.assets.srdf and urdf_usable:
        return "srdf"
    if urdf_usable:
        return "sampling"
    if robot.assets.mjcf:
        return "mjcf"
    if urdf_path is not None:
        # A URDF with no usable collision geometry and no MJCF: still the URDF
        # path (it will raise/emit empty geometry, surfacing the missing meshes)
        # rather than silently producing nothing.
        return "sampling"
    raise ROSConfigError(
        f"{robot.name}: no lowerable asset — needs assets.urdf (with collision "
        "meshes) or assets.mjcf"
    )


def lower_robot_auto(
    robot: RobotDescription,
    *,
    acm_only: bool = False,
    geometry_only: bool = False,
    manifest_dir: Path | None = None,
    tight_links: Iterable[str] | None = None,
) -> LoweredCollisionModel:
    """Lower ``robot`` via the provenance-correct source (``select_lowering``).

    The single dispatch the CLI (``openral collision lower``/``check``) and the
    byte-identical regression test both call, so routing can never diverge
    between "what we commit" and "what we verify". ``acm_only`` keeps the
    manifest geometry on both paths; ``geometry_only`` applies to the URDF path
    (the MJCF path always emits both blocks); ``tight_links`` names links to
    lower as box + exact hull on top of those the manifest already refines.

    Raises:
        ROSConfigError: Propagated from ``select_lowering`` /
            ``lower_robot`` / ``lower_robot_from_mjcf``.
    """
    if select_lowering(robot, manifest_dir=manifest_dir) == "mjcf":
        return lower_robot_from_mjcf(
            robot, manifest_dir=manifest_dir, acm_only=acm_only, tight_links=tight_links
        )
    return lower_robot(
        robot,
        acm_only=acm_only,
        geometry_only=geometry_only,
        manifest_dir=manifest_dir,
        tight_links=tight_links,
    )
