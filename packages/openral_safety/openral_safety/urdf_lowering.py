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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from openral_core import (
    CapsuleShape,
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
    "LoweredCollisionModel",
    "LoweringSource",
    "acm_for_geometry",
    "fit_capsule_to_vertices",
    "lower_joint_fk",
    "lower_link_geometry",
    "lower_robot",
    "lower_robot_auto",
    "lower_robot_from_mjcf",
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


def _circle_2(a: _Arr, b: _Arr) -> tuple[_Arr, float]:
    import numpy as np

    c = (a + b) / 2.0
    return c, float(np.linalg.norm(a - c))


def _circle_3(a: _Arr, b: _Arr, c: _Arr) -> tuple[_Arr, float]:
    """Circumcircle of three 2-D points (the widest pair's circle if collinear)."""
    import numpy as np

    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-18:
        pairs = [(a, b), (a, c), (b, c)]
        return _circle_2(*max(pairs, key=lambda p: float(np.linalg.norm(p[0] - p[1]))))
    sa, sb, sc = a @ a, b @ b, c @ c
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(a - center))


def _next_outside(pts: _Arr, start: int, stop: int, center: _Arr, radius: float) -> int:
    """Index of the first point in ``pts[start:stop]`` outside the circle, or -1."""
    import numpy as np

    d = np.linalg.norm(pts[start:stop] - center, axis=1)
    out = np.flatnonzero(d > radius * (1.0 + 1e-12) + 1e-15)
    return start + int(out[0]) if out.size else -1


def _min_enclosing_circle(pts: _Arr) -> tuple[_Arr, float]:
    """Smallest circle holding every 2-D point (Welzl, iterative, vectorised scans).

    Each loop only scans forward, so it terminates whatever floating point does
    at the boundary. The capsule built on the centre takes its radius as the
    true max distance (``_capsule_for_axis``), so a last-ulp miss here can only
    loosen the fit, never let a point out. The fixed-seed shuffle keeps the
    expected cost linear and the result reproducible.
    """
    import numpy as np

    p = pts[np.random.default_rng(0).permutation(len(pts))]
    n = len(p)
    center, radius = p[0].copy(), 0.0
    i = _next_outside(p, 1, n, center, radius)
    while i >= 0:
        center, radius = p[i].copy(), 0.0
        j = _next_outside(p, 0, i, center, radius)
        while j >= 0:
            center, radius = _circle_2(p[i], p[j])
            k = _next_outside(p, 0, j, center, radius)
            while k >= 0:
                center, radius = _circle_3(p[i], p[j], p[k])
                k = _next_outside(p, k + 1, j, center, radius)
            j = _next_outside(p, j + 1, i, center, radius)
        i = _next_outside(p, i + 1, n, center, radius)
    return center, radius


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

    pts = np.unique(np.asarray(vertices, dtype=np.float64).reshape(-1, 3), axis=0)
    if len(pts) >= _HULL_REDUCE_MIN_POINTS:
        import trimesh

        # A convex capsule holds the cloud iff it holds the cloud's hull
        # vertices, so fitting those is exact and 10-100x cheaper on CAD meshes.
        pts = np.asarray(trimesh.convex.convex_hull(pts).vertices, dtype=np.float64)
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


def lower_link_geometry(urdf_path: str) -> list[LinkCollisionGeometry]:
    """One conservative ``LinkCollisionGeometry`` per URDF link with geometry.

    The primitive bounds the link's ``<collision>`` **and** ``<visual>``
    geometry together. A vendor's ``<collision>`` block is a simplification, and
    measured on the shipped robots it is often *smaller* than the part: H1's is a
    set of placeholder spheres/cylinders the real torso reaches 549 mm past, G1's
    shoulder cylinders miss the shoulder by 61 mm, SO-100's meshes leave out the
    servo housings (26 mm), Flexiv's 30-vertex meshes cut 3 mm into the CAD. The
    visual mesh is the CAD of the part the robot is made of, so the fit holds
    both (``docs/reference/collision-geometry-review.md``).

    Boxes map to their 8 corners, cylinders and spheres to circumscribing
    polytopes, meshes to their vertices (``trimesh``); everything is placed in
    the link frame by its ``<origin>`` and fitted with
    ``fit_capsule_to_vertices`` (minimum volume, declared headroom). A link whose
    only geometry is one sphere ``<collision>`` emits that exact ``SphereShape``.

    Fails closed on partial resolution: a link some of whose geometry loads and
    some does not raises, because fitting the part that loaded would under-cover
    the link. A link none of whose geometry resolves is skipped with a warning
    per missing mesh (the openarm URDF's deliberately unresolvable refs, which
    route it to the MJCF lowering).

    Raises:
        ROSConfigError: If a link's geometry resolves only partially.
    """
    import numpy as np

    model = _load_urdf(urdf_path)
    handler = getattr(model, "_filename_handler", None)

    out: list[LinkCollisionGeometry] = []
    for link_name, link in model.link_map.items():  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        collisions = list(getattr(link, "collisions", None) or [])
        visuals = list(getattr(link, "visuals", None) or [])
        elements = collisions + visuals
        if not collisions:
            continue
        if len(elements) == 1 and getattr(collisions[0].geometry, "sphere", None) is not None:
            sph = collisions[0].geometry.sphere
            tf = _origin_matrix(collisions[0].origin)
            cx, cy, cz = (float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3]))
            out.append(
                LinkCollisionGeometry(
                    link_name=link_name,
                    shape=SphereShape(radius_m=float(sph.radius)),
                    origin_xyz_rpy=(cx, cy, cz, 0.0, 0.0, 0.0),
                )
            )
            continue

        clouds: list[_Arr] = []
        missing: list[str] = []
        for el in elements:
            verts = _collision_local_vertices(el, handler)
            if verts is None or len(verts) == 0:
                mesh = getattr(el.geometry, "mesh", None)
                missing.append(getattr(mesh, "filename", "<unsupported geometry>"))
                continue
            clouds.append(_apply(_origin_matrix(el.origin), verts))
        if not clouds:
            continue
        if missing:
            raise ROSConfigError(
                f"link {link_name!r}: geometry {missing} did not load while the rest of "
                "the link did; a primitive fitted to the part that loaded would "
                "under-cover the link. Resolve the refs (vendored URDFs use "
                "`rd:<module>:<relpath>`) before lowering."
            )
        cloud = np.vstack(clouds)
        if len(cloud) < 4:
            continue
        shape, origin = fit_capsule_to_vertices(cloud)
        out.append(LinkCollisionGeometry(link_name=link_name, shape=shape, origin_xyz_rpy=origin))
    return out


def _collision_local_vertices(col: object, handler: object) -> _Arr | None:
    """Vertices of one ``<collision>``/``<visual>`` geometry, in its own local frame.

    Mesh → loaded vertices × scale; box → corners; cylinder / sphere → the
    vertices of a circumscribing polytope, so a convex primitive holding them
    holds the whole solid.
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
        return _box_vertices(box.size)
    if cyl is not None:
        return _cylinder_vertices(float(cyl.radius), float(cyl.length))
    if sph is not None:
        return _sphere_vertices(float(sph.radius))
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
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        scale = getattr(mesh, "scale", None)
        if scale is not None:
            verts = verts * np.asarray(scale, dtype=np.float64)
        return np.asarray(verts, dtype=np.float64)
    return None


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


def _axis_radius_bound(chain: list[object], joint: object, geom: LinkCollisionGeometry) -> float:
    """Bound the distance from ``joint``'s axis to any point of ``geom``.

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
    origin_xyz = np.asarray(geom.origin_xyz_rpy[:3], dtype=np.float64)
    return total + float(np.linalg.norm(origin_xyz)) + shape_max_extent_m(geom.shape)


def _pair_relative_dofs(
    model: object, link_a: str, link_b: str, geoms: dict[str, LinkCollisionGeometry]
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
    geoms: dict[str, LinkCollisionGeometry],
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

    from openral_safety.kernel_predicates import shape_distance

    dofs = _pair_relative_dofs(model, link_a, link_b, geoms)
    if dofs is None or len(dofs) > _CERTIFY_MAX_DOF:
        return False

    geom_a, geom_b = geoms[link_a], geoms[link_b]
    if geom_a.tight_geometry is not None and geom_b.tight_geometry is not None:
        # The kernel re-asks a box pair it cannot clear of the two exact hulls
        # (`hull_hull_distance`, issue #191), and the hull gap is >= the box gap
        # everywhere. So "the box gap is <= margin at every pose" — all this
        # function can prove with `shape_distance` — no longer implies the kernel
        # trips, and certifying on it would grant an ACM entry that hides a live
        # check. Withhold instead; the pair stays checked, which is the direction
        # every other refusal here also fails in.
        return False
    origin_a = _xyzrpy_matrix(geom_a.origin_xyz_rpy)
    origin_b = _xyzrpy_matrix(geom_b.origin_xyz_rpy)
    chains = _relative_chains(model, link_a, link_b)
    if chains is None:  # pragma: no cover — _pair_relative_dofs already returned non-None
        return False
    chain_a, chain_b = chains
    names = [str(j.name) for j, _, _ in dofs]  # type: ignore[attr-defined]  # reason: yourdfpy Joint
    radii = np.asarray([r for _, r, _ in dofs], dtype=np.float64)
    lo = np.asarray([span[0] for _, _, span in dofs], dtype=np.float64)
    hi = np.asarray([span[1] for _, _, span in dofs], dtype=np.float64)

    def gaps_at(centres: _Arr) -> _Arr:
        """Kernel surface gap at each row of ``centres`` (one joint vector per row)."""
        n = int(centres.shape[0])
        values = {name: centres[:, i] for i, name in enumerate(names)}
        t_a = _chain_transforms(chain_a, values, n) @ origin_a
        t_b = _chain_transforms(chain_b, values, n) @ origin_b
        return shape_distance(geom_a.shape, t_a, geom_b.shape, t_b)

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
    geoms: dict[str, LinkCollisionGeometry],
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
        geoms: The per-link primitives the kernel will load, by link name.
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
    links = [ln for ln in model.link_map if ln in geoms]  # type: ignore[attr-defined]  # reason: yourdfpy URDF

    disabled: _AcmPairs = set()
    for joint in model.robot.joints:  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        if joint.parent in geoms and joint.child in geoms and joint.parent != joint.child:
            disabled.add(frozenset({joint.parent, joint.child}))  # adjacent
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            if frozenset({a, b}) in disabled:
                continue  # already adjacent; no need to prove anything
            if _certified_always_colliding(model, geoms, a, b, margin_m=margin_m):
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
    geoms = {g.link_name: g for g in lower_link_geometry(urdf_path)}
    return acm_for_geometry(urdf_path, geoms, srdf_path=None, margin_m=margin_m)


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


def lower_robot_from_mjcf(  # noqa: PLR0912, PLR0915  # reason: one cohesive MJCF lowering pass (load → link-map → FK → sweep → ACM)
    robot: RobotDescription,
    *,
    n_samples: int = _MJCF_N_SAMPLES,
    seed: int = _MJCF_RNG_SEED,
    margin_m: float = 0.0,
    manifest_dir: Path | None = None,
) -> LoweredCollisionModel:
    """Lower joint FK + sampling ACM from a robot's MJCF, keeping its manifest geometry.

    For MJCF-native robots with **no URDF** whose collision geoms are meshes (which
    ``mjcf_lowering``'s primitive path skips) — e.g. the bimanual ``openarm``. The
    hand-authored manifest capsules are kept; FK is the MJCF transform from each
    joint's ``parent_link`` to its ``child_link`` at the rest pose (matched to the
    MJCF by ``sim_joint_name``), and the ACM is the capsule sweep run with **mujoco
    forward kinematics** over the manifest geometry. ``acm_source = "mjcf"``.

    ``manifest_dir`` resolves a ``file:`` MJCF ref against the manifest's own
    directory (no in-tree robot uses one today, but the resolver honours it).

    Raises:
        ROSConfigError: If the robot has no ``assets.mjcf`` ref, or it cannot be
            resolved to a file.
    """
    import mujoco
    import numpy as np
    from openral_core.assets import AssetRefError, resolve_asset

    from openral_safety.kernel_predicates import shape_distance

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

    def name_bid(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    def jid(name: str | None) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) if name else -1

    def body_tf(i: int) -> _Arr:
        tf = np.eye(4)
        tf[:3, :3] = data.xmat[i].reshape(3, 3)
        tf[:3, 3] = data.xpos[i]
        return tf

    # Resolve manifest link names → MJCF body ids. The two naming schemes can
    # diverge (openarm's manifest ``link0`` / ``link7`` are the MJCF's
    # ``base_link`` / ``ee_base_link``), so map via the joint correspondence
    # (``sim_joint_name`` → MJCF joint → its child body), which is unambiguous.
    link_body: dict[str, int] = {}
    for j in robot.joints:
        ji = jid(j.sim_joint_name)
        if ji < 0:
            continue
        child_b = int(model.jnt_bodyid[ji])
        link_body[j.child_link] = child_b
        # The parent link maps to the MJCF body's parent (root link of the chain).
        if j.parent_link not in link_body:
            link_body[j.parent_link] = int(model.body_parentid[child_b])
    # Fall back to a direct name match for any link the joint map didn't cover.
    for ln in {g.link_name for g in robot.collision_geometry} | {
        j.parent_link for j in robot.joints
    }:
        if ln not in link_body and name_bid(ln) >= 0:
            link_body[ln] = name_bid(ln)

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
    geoms = {g.link_name: g for g in robot.collision_geometry if g.link_name in link_body}
    links = list(geoms)
    # Each link's primitive origin in its own body frame; the sweep places it by
    # mujoco FK and asks `shape_distance` — the kernel's own predicate for the
    # actual shape, not a stand-in for it (issue #155).
    local_origin = {ln: _xyzrpy_matrix(geoms[ln].origin_xyz_rpy) for ln in links}
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

    rng = np.random.default_rng(seed)
    counts: dict[frozenset[str], int] = {}
    for _ in range(n_samples):
        mujoco.mj_resetData(model, data)
        for adr, lo, hi in sweep:
            data.qpos[adr] = lo + (hi - lo) * rng.random()
        mujoco.mj_kinematics(model, data)
        world = {ln: (body_tf(link_body[ln]) @ local_origin[ln])[None] for ln in links}
        for i, a in enumerate(links):
            for b in links[i + 1 :]:
                gap = shape_distance(geoms[a].shape, world[a], geoms[b].shape, world[b])
                if float(gap[0]) <= margin_m:
                    counts[frozenset({a, b})] = counts.get(frozenset({a, b}), 0) + 1
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            # Conservative (no SRDF ground truth): disable only ALWAYS-colliding
            # capsule junctions. Never-collide pairs stay CHECKED — a sweep can't
            # prove a cross-branch bimanual pair never collides (it can miss the
            # tail), so we never auto-disable one.
            if counts.get(frozenset({a, b}), 0) == n_samples:  # always-colliding
                disabled.add(frozenset({a, b}))

    return LoweredCollisionModel(
        collision_geometry=list(robot.collision_geometry),
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


def lower_robot(
    robot: RobotDescription,
    *,
    srdf_path: str | None = None,
    acm_only: bool = False,
    geometry_only: bool = False,
    manifest_dir: Path | None = None,
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

    Returns:
        A ``LoweredCollisionModel``.

    Raises:
        ROSConfigError: If ``assets.urdf`` is unset (and no sim MJCF) or a
            declared asset ref does not resolve.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if robot.assets.urdf is None:
        # MJCF-native robots (no URDF; mesh collision) lower from their sim MJCF —
        # geometry stays the manifest's hand-authored capsules.
        if robot.assets.mjcf:
            return lower_robot_from_mjcf(robot, manifest_dir=manifest_dir)
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
        geometry = [g for g in lower_link_geometry(urdf_ref) if g.link_name in chain_links]
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
        geom_list = robot.collision_geometry if acm_only else geometry
        geom_by_link = {g.link_name: g for g in geom_list}
        # The kernel trips at `safety.self_collision_margin_m`, so the
        # always-colliding proof must use that same threshold — see
        # `acm_for_geometry`. so101_follower runs -0.06 m; a sweep hardcoded at
        # 0.0 would have called pairs always-colliding that the kernel never
        # trips on, and exempted a check that was doing real work.
        disabled = acm_for_geometry(
            urdf_ref,
            geom_by_link,
            srdf_path=used_srdf,
            margin_m=float(getattr(robot.safety, "self_collision_margin_m", 0.0) or 0.0),
        )
        source = "srdf" if used_srdf else "sampling"
        pairs = _scoped_sorted_pairs(disabled, set(geom_by_link))

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
    """
    return len(lower_link_geometry(urdf_path)) > 0


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
) -> LoweredCollisionModel:
    """Lower ``robot`` via the provenance-correct source (``select_lowering``).

    The single dispatch the CLI (``openral collision lower``/``check``) and the
    byte-identical regression test both call, so routing can never diverge
    between "what we commit" and "what we verify". ``acm_only`` / ``geometry_only``
    are forwarded to the URDF path; the MJCF path always emits both blocks (it
    keeps the manifest geometry and recomputes the ACM).

    Raises:
        ROSConfigError: Propagated from ``select_lowering`` /
            ``lower_robot`` / ``lower_robot_from_mjcf``.
    """
    if select_lowering(robot, manifest_dir=manifest_dir) == "mjcf":
        return lower_robot_from_mjcf(robot, manifest_dir=manifest_dir)
    return lower_robot(
        robot, acm_only=acm_only, geometry_only=geometry_only, manifest_dir=manifest_dir
    )
