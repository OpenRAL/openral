# python/hal/src/openral_hal/convex_distance.py
"""Certified signed distance between two MuJoCo convex geoms.

``mujoco.mj_geomDistance`` is not usable as an adjudication instrument for the
pairs the collision-evidence path measures. Under mujoco 3.8.0 it fails in two
distinct ways on a RoboCasa fixture geom against a ``panda_mobile`` collision
mesh, and both failures are silent — a wrong number, no error, no flag:

* **The default (native CCD) path is knife-edge.** On ``robocasa_fridge_drawer``
  layout 9 it returns ``+0.000000`` for ``robot0_link7_collision`` vs
  ``fridge_right_group_freezer_door_main``, writing a ``fromto`` witness
  126.264 mm long whose endpoints lie outside *both* geoms. The true gap is
  ``+0.148512 mm``. Displacing the link by **1 picometre** — 1e-12 m, ten
  orders of magnitude below the answer — makes the same call return
  ``+0.1485 mm`` with a 0.149 mm witness. It is a degenerate *configuration*,
  not a distance regime, so no choice of ``distmax`` avoids it; and a scene's
  reset pose is exactly where degenerate configurations live, because fixtures
  are placed on exact axis-aligned numbers.
* **The libccd path (``mjDSBL_NATIVECCD``) is robustly wrong, and unbounded in
  ``distmax``.** The same pair reports ``-2.168 / -46.372 / -57.032 / -339.690
  / -351.570 / -361.890 / -367.604 mm`` at ``distmax`` ``0.02 / 0.05 / 0.1 /
  0.2 / 0.3 / 0.6 / 1.0`` — a monotone function of the probe window, through a
  48 mm-thick door panel, against a true gap of ``+0.15 mm``.

Both reproduce in a **two-geom standalone MJCF** carrying nothing but that mesh
and that box at those world poses, so neither is a robosuite, RoboCasa or
model-size artifact: it is ``mj_geomDistance`` itself.

This module replaces it there. It answers the same question — the signed
distance between the two *convex* bodies MuJoCo would actually collide — and
answers it with a proof:

* Every geom is represented as ``conv(core) ⊕ ball(radius)``: a box and a mesh
  are their hull vertices at radius 0, a sphere is one point, a capsule two.
  Signed distance is then ``signed(core_a, core_b) - r_a - r_b`` — exact on
  both branches, because inflating two convex bodies by balls shifts their
  signed distance by exactly the sum of the radii.
* **Separated** cores are solved by GJK, and the answer carries a
  **separating-axis certificate**: for the unit witness direction ``u``,
  ``min_B u·b - max_A u·a`` is a lower bound on the true distance (weak
  duality) and ``||p_b - p_a||`` is an upper bound. Coinciding bounds prove
  optimality. The residual is reported as ``duality_gap_m``; it is ~1e-14 m in
  practice.
* **Overlapping** cores are solved by exact SAT over face normals and
  edge-edge cross products — the direction set that provably contains the
  minimum-translational-distance axis of two convex polytopes, and the same
  construction the safety kernel's own ``box_box_distance`` uses on its 6 + 9
  axes.
* **Round types with no ball form** (cylinder, ellipsoid) are bracketed by an
  inscribed and a circumscribed polytope, so the answer is an interval that
  provably contains the truth rather than a number that might not.

Nothing here is a fallback (CLAUDE.md §1.4). Every result states whether it is
certified, and an uncertified result carries the reason instead of a plausible
number. A caller that cannot use an uncertified distance must refuse the
measurement.

Cost: ~1 ms per mesh↔box pair on this dev host, against ``mj_geomDistance``'s
~0.8 µs. Three orders of magnitude, affordable only because this runs once at a
terminal event in evidence collection. Nothing on the 100 Hz path calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

# Inscribed/circumscribed polygon resolution for the round types that have no
# exact ball form. The bracket a pair of them opens is
# ``(r_a + r_b) * (1/cos(pi/n) - 1)``, so 256 segments hold a pair of half-metre
# cylinders inside 0.04 mm — an order below the 1 mm any collision verdict in
# this repo is stated at. 64 was tried first and misses: two 50/60 mm cylinders
# bracket to 0.13 mm there, which this module then refuses to certify rather
# than round away. The bracket is reported either way, so nobody has to trust
# this number.
_ARC_SEGMENTS: Final[int] = 256
# Certify a bracket this tight or tighter. Exact types bracket at width 0.0.
_BRACKET_TOL_M: Final[float] = 1e-4
# Certify a separating-axis duality gap this tight or tighter.
_DUALITY_TOL_M: Final[float] = 1e-9
# Below this, GJK's answer is "the cores touch or overlap" rather than a
# distance: its terminal simplex has collapsed onto the origin and the witness
# it would report is arbitrary. Overlap is SAT's branch, not GJK's.
_OVERLAP_TOL_M: Final[float] = 1e-12
# Refuse an exact penetration depth beyond this many SAT axes rather than
# spend unbounded time or silently subsample the axis set.
_MAX_SAT_AXES: Final[int] = 8_000_000


@dataclass(frozen=True)
class ConvexDistance:
    """One certified signed distance, with the proof that backs it.

    Attributes:
        distance_m: The reported signed distance — the **closest** end of the
            certified bracket, so a reader is never told a pair is further
            apart than it provably is. Negative is penetration depth.
        lower_m: Certified lower bound on the true signed distance.
        upper_m: Certified upper bound.
        certified: Whether the bracket and the duality gap both closed inside
            tolerance. ``False`` means the number must not be cited.
        uncertified_reason: Why not; empty when certified.
        witness_a: Nearest point on geom ``a``. For an overlapping pair solved
            by SAT it is the centroid of ``a``'s deepest face along the
            minimum-translation axis; ``None`` only when the depth itself was
            not measured.
        witness_b: Nearest point on geom ``b`` on the separated branch. For an
            **overlapping** pair it is the ``witness_a`` point lifted onto
            ``b``'s supporting *plane* along the minimum-translation axis —
            which is the right depth but need not lie within ``b``'s face, so
            it is NOT a point on ``b``'s surface and ``witness_clearance_m``
            will refute it as one. Read it as "where ``a``'s buried face meets
            ``b``'s supporting plane", and take the direction from
            :attr:`direction`, never by differencing the two witnesses.
        direction: Unit direction from ``b`` toward ``a`` — the separating
            direction on the GJK branch and the minimum-translation axis on
            the SAT branch. ``None`` when the pair was not solved (beyond the
            window, unsupported geom, or an overlap whose depth was not
            measured). This is the ONLY reliable source of a contact direction
            at a flush contact, where the two witnesses coincide and their
            difference carries no direction at all.
        duality_gap_m: Residual of the separating-axis certificate; ``0.0`` for
            an overlapping pair, which SAT solves without one.
        method: Which construction produced the answer.
    """

    distance_m: float
    lower_m: float
    upper_m: float
    certified: bool
    uncertified_reason: str
    witness_a: tuple[float, float, float] | None
    witness_b: tuple[float, float, float] | None
    direction: tuple[float, float, float] | None
    duality_gap_m: float
    method: str

    def as_record(self) -> dict[str, object]:
        """The JSON-safe form that goes into a stop record.

        A non-finite bound serialises as ``None``, never as ``Infinity``: the
        stop record is read back as strict JSON, and a bound that cannot be
        written is the same thing as a bound that is not known.
        """

        def finite(value: float) -> float | None:
            return round(value, 9) if float("-inf") < value < float("inf") else None

        record: dict[str, object] = {
            "distance_m": round(self.distance_m, 9),
            "distance_certified": self.certified,
            "distance_method": self.method,
        }
        if self.lower_m != self.upper_m:
            record["distance_bracket_m"] = [finite(self.lower_m), finite(self.upper_m)]
        if self.witness_a is not None and self.witness_b is not None:
            record["witness_a_xyz"] = [round(v, 9) for v in self.witness_a]
            record["witness_b_xyz"] = [round(v, 9) for v in self.witness_b]
        if not self.certified:
            record["uncertified_reason"] = self.uncertified_reason
        return record


@dataclass(frozen=True)
class ConvexBody:
    """A geom as ``conv(core) ⊕ ball(radius)``, plus what SAT needs.

    ``faces`` indexes ``core`` and is ``None`` when the geom's hull topology is
    not available — which costs nothing until the two cores actually overlap,
    and then produces an honest "not measured" rather than a guess.
    """

    core: Any
    radius: float
    faces: Any | None


def _geom_frame(data: Any, geom_id: int) -> tuple[Any, Any]:
    import numpy as np

    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    return rot, pos


def mesh_hull(model: Any, mesh_id: int) -> tuple[Any, Any | None]:
    """The hull MuJoCo itself collides: ``(vertices, faces)`` in mesh-local frame.

    MuJoCo compiles a convex-hull graph for every mesh it may have to collide
    and collides *that* hull, never the raw triangles. Reading the hull off the
    graph — rather than re-deriving one — is what keeps this module's answer the
    answer to MuJoCo's own question, and it needs no computational-geometry
    dependency. A mesh with no graph contributes all of its vertices and no
    faces: the vertex set still bounds the hull exactly (its convex hull *is*
    the hull), so distance stays exact; only penetration depth loses its axis
    set, and says so.

    Args:
        model: ``mjModel``.
        mesh_id: Index into the model's mesh arrays.

    Returns:
        ``(vertices, faces)`` — ``(n, 3)`` float64 and ``(f, 3)`` int64
        indexing into ``vertices``, or ``None`` faces when the mesh has no
        compiled hull graph.
    """
    import numpy as np

    adr = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    verts = np.asarray(model.mesh_vert[adr : adr + count], dtype=np.float64).reshape(count, 3)
    graph_adr = int(model.mesh_graphadr[mesh_id])
    if graph_adr < 0:
        return verts, None
    graph = np.asarray(model.mesh_graph, dtype=np.int64)
    n_vert = int(graph[graph_adr])
    n_face = int(graph[graph_adr + 1])
    vert_global = graph[graph_adr + 2 + n_vert : graph_adr + 2 + 2 * n_vert]
    face_base = graph_adr + 2 + 3 * n_vert + 3 * n_face
    face_global = graph[face_base : face_base + 3 * n_face].reshape(n_face, 3)
    remap = np.full(count, -1, dtype=np.int64)
    remap[vert_global] = np.arange(n_vert, dtype=np.int64)
    faces = remap[face_global]
    if bool((faces < 0).any()):  # pragma: no cover - a graph that does not index its own hull
        return verts[vert_global], None
    return verts[vert_global], faces


def _unit_circle(segments: int) -> Any:
    import numpy as np

    theta = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _prism(radius: float, half_height: float, segments: int) -> tuple[Any, Any]:
    """A ``segments``-gon prism as ``(vertices, faces)``, cap centres included."""
    import numpy as np

    circle = _unit_circle(segments) * radius
    top = np.column_stack([circle, np.full(segments, half_height)])
    bottom = np.column_stack([circle, np.full(segments, -half_height)])
    verts = np.concatenate([top, bottom])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + i])
        faces.append([j, segments + j, segments + i])
    faces.append([0, 1, 2])
    faces.append([segments, segments + 2, segments + 1])
    return verts, np.asarray(faces, dtype=np.int64)


def _lat_long_sphere(radius: float, segments: int) -> Any:
    """A latitude/longitude polytope **inscribed** in a sphere of ``radius``."""
    import numpy as np

    rings = max(2, segments // 2)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, rings + 1)
    circle = _unit_circle(segments)
    parts = [np.array([[0.0, 0.0, -radius], [0.0, 0.0, radius]], dtype=np.float64)]
    for phi in lat[1:-1]:
        parts.append(
            np.column_stack(
                [circle * (radius * np.cos(phi)), np.full(segments, radius * np.sin(phi))]
            )
        )
    return np.concatenate(parts)


def _apothem(segments: int) -> float:
    """Inradius of the regular ``segments``-gon of unit circumradius."""
    import numpy as np

    return float(np.cos(np.pi / segments))


def geom_convex_bracket(
    model: Any, data: Any, geom_id: int, *, arc_segments: int = _ARC_SEGMENTS
) -> tuple[ConvexBody, ConvexBody] | None:
    """World-frame ``(inscribed, circumscribed)`` bodies bracketing one geom.

    Both are ``conv(core) ⊕ ball(radius)`` forms with ``inner ⊆ geom ⊆ outer``.
    Box, mesh, sphere and capsule are represented **exactly** — the two returned
    bodies are the same object — because each is a polytope inflated by a ball.
    Cylinder and ellipsoid have no ball form, so they are sandwiched by
    polygonised bodies and the caller reports the resulting interval.

    Returns ``None`` for a geom with no bounded convex hull (plane, heightfield,
    SDF), so the caller refuses rather than guesses.

    Args:
        model: ``mjModel``.
        data: ``mjData``, already forwarded.
        geom_id: The geom.
        arc_segments: Polygon resolution for the bracketed types.

    Returns:
        ``(inner, outer)``, or ``None``.
    """
    import mujoco  # reason: optional sim dep
    import numpy as np

    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rot, pos = _geom_frame(data, geom_id)

    def world(local: Any) -> Any:
        return np.asarray(local, dtype=np.float64) @ rot.T + pos

    def exact(core: Any, radius: float, faces: Any | None) -> tuple[ConvexBody, ConvexBody]:
        body = ConvexBody(core=core, radius=radius, faces=faces)
        return body, body

    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float64,
        )
        # One triangle per face is enough: SAT needs the three face NORMALS and
        # the three edge directions, and these six triangles carry both.
        box_faces = np.asarray(
            [[0, 1, 2], [4, 6, 5], [0, 4, 1], [2, 3, 6], [0, 2, 4], [1, 5, 3]], dtype=np.int64
        )
        return exact(world(signs * size[:3]), 0.0, box_faces)
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        verts, mesh_faces = mesh_hull(model, int(model.geom_dataid[geom_id]))
        return exact(world(verts), 0.0, mesh_faces)
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return exact(pos.reshape(1, 3), float(size[0]), None)
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        axis = np.array([[0.0, 0.0, float(size[1])], [0.0, 0.0, -float(size[1])]])
        return exact(world(axis), float(size[0]), None)
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        inner_v, faces = _prism(float(size[0]), float(size[1]), arc_segments)
        outer_v, _ = _prism(float(size[0]) / _apothem(arc_segments), float(size[1]), arc_segments)
        return (
            ConvexBody(core=world(inner_v), radius=0.0, faces=faces),
            ConvexBody(core=world(outer_v), radius=0.0, faces=faces),
        )
    if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        unit = _lat_long_sphere(1.0, arc_segments)
        return (
            ConvexBody(core=world(unit * size[:3]), radius=0.0, faces=None),
            ConvexBody(
                core=world(unit / _apothem(arc_segments) * size[:3]), radius=0.0, faces=None
            ),
        )
    return None


def _closest_on_simplex(pts: Any) -> tuple[Any, Any, list[int]]:
    """Closest point of ``conv(pts)`` (at most four points) to the origin.

    Brute force over every face of the simplex — at most fifteen 5x5 solves.
    Being exhaustive is the point: a hand-rolled Johnson subalgorithm is where
    GJK implementations acquire the degenerate-simplex failures this module
    exists to stop trusting.

    Returns:
        ``(point, barycentric weights, kept indices)``.
    """
    from itertools import combinations

    import numpy as np

    best: tuple[float, Any, Any, list[int]] | None = None
    n = len(pts)
    for size in range(1, n + 1):
        for comb in combinations(range(n), size):
            face = pts[list(comb)]
            if size == 1:
                weights = np.array([1.0])
            else:
                system = np.zeros((size + 1, size + 1), dtype=np.float64)
                system[:size, :size] = 2.0 * (face @ face.T)
                system[:size, size] = 1.0
                system[size, :size] = 1.0
                rhs = np.zeros(size + 1, dtype=np.float64)
                rhs[size] = 1.0
                try:
                    solution = np.linalg.solve(system, rhs)
                except np.linalg.LinAlgError:
                    continue
                weights = solution[:size]
                if not bool(np.all(np.isfinite(weights))) or bool((weights < -1e-12).any()):
                    continue
                weights = np.clip(weights, 0.0, None)
                total = float(weights.sum())
                if total <= 0.0:
                    continue
                weights = weights / total
            point = weights @ face
            norm = float(point @ point)
            if best is None or norm < best[0]:
                best = (norm, point, weights, list(comb))
    if best is None:  # pragma: no cover - the single-point case always solves
        nearest = int(np.argmin((pts * pts).sum(axis=1)))
        return pts[nearest], np.array([1.0]), [nearest]
    return best[1], best[2], best[3]


def _gjk(points_a: Any, points_b: Any, *, max_iter: int = 128) -> tuple[float, Any, Any]:
    """Distance between two convex point clouds, with the nearest points.

    ``0.0`` with ``None`` witnesses means the hulls overlap: GJK proves overlap
    by writing the origin as a convex combination of Minkowski-difference
    points, but it cannot say how deep — that is SAT's job.
    """
    import numpy as np

    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)

    def support(direction: Any) -> tuple[Any, int, int]:
        ia = int(np.argmax(a @ direction))
        ib = int(np.argmin(b @ direction))
        return a[ia] - b[ib], ia, ib

    seed = a.mean(axis=0) - b.mean(axis=0)
    if float(np.linalg.norm(seed)) == 0.0:
        seed = np.array([1.0, 0.0, 0.0])
    simplex = [support(-seed)]
    v = simplex[0][0]
    for _ in range(max_iter):
        norm_sq = float(v @ v)
        if norm_sq <= 0.0:
            return 0.0, None, None
        w, ia, ib = support(-v)
        if norm_sq - float(v @ w) <= 1e-15 * max(norm_sq, 1.0):
            break
        if any(sa == ia and sb == ib for _, sa, sb in simplex):
            break
        simplex.append((w, ia, ib))
        point, _weights, keep = _closest_on_simplex(np.array([s[0] for s in simplex]))
        simplex = [simplex[k] for k in keep]
        v = point
        if float(v @ v) <= 0.0:
            return 0.0, None, None
    point, weights, keep = _closest_on_simplex(np.array([s[0] for s in simplex]))
    kept = [simplex[k] for k in keep]
    p_a = sum(weights[i] * a[kept[i][1]] for i in range(len(kept)))
    p_b = sum(weights[i] * b[kept[i][2]] for i in range(len(kept)))
    return float(np.linalg.norm(point)), np.asarray(p_a), np.asarray(p_b)


def separating_axis_bound(points_a: Any, points_b: Any, direction: Any) -> float:
    """Weak-duality lower bound on the distance, along one direction.

    For **any** unit ``u``, ``min_B u·b - max_A u·a`` is a lower bound on the
    distance between the two convex hulls. Evaluating it at the direction a
    witness segment claims is what turns a GJK answer into a proof: an upper
    bound that meets its own lower bound cannot be improved, so the number is
    optimal rather than merely plausible.

    Args:
        points_a: Convex point cloud ``a``.
        points_b: Convex point cloud ``b``.
        direction: Any non-zero direction; normalised internally.

    Returns:
        The bound in metres; ``-inf`` for a degenerate direction, which
        certifies nothing and is meant not to.
    """
    import numpy as np

    u = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        return float("-inf")
    u = u / norm
    return float((np.asarray(points_b) @ u).min() - (np.asarray(points_a) @ u).max())


def _sat_axes(body: ConvexBody) -> tuple[Any, Any] | None:
    """``(face normals, edge directions)`` of a body's core polytope."""
    import numpy as np

    core = np.asarray(body.core, dtype=np.float64)
    if len(core) == 1:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if len(core) == 2:
        return np.zeros((0, 3)), (core[1] - core[0]).reshape(1, 3)
    if body.faces is None:
        return None
    faces = np.asarray(body.faces, dtype=np.int64)
    tri = core[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = core[pairs[:, 1]] - core[pairs[:, 0]]
    return normals, edges


def _sat_penetration_depth(
    body_a: ConvexBody, body_b: ConvexBody
) -> tuple[float, Any, Any, Any] | None:
    """Exact minimum translational distance between two overlapping cores.

    The MTD axis of two convex polytopes is always a face normal of one, a face
    normal of the other, or the cross product of an edge from each, so the
    minimum overlap over that finite set *is* the penetration depth — no search,
    no iteration. Returns ``None`` when an axis set is unavailable or would
    exceed ``_MAX_SAT_AXES``, so the caller reports "not measured" instead of a
    subsampled guess.

    Returns:
        ``(depth, axis, p_a, p_b)``: the (negative) depth; the unit MTD axis
        oriented so that ``a`` lies on its positive side; the centroid of
        ``a``'s deepest face along it; and that point lifted onto ``b``'s
        supporting plane, so ``p_a - p_b == depth * axis``.
    """
    import numpy as np

    axes_a = _sat_axes(body_a)
    axes_b = _sat_axes(body_b)
    if axes_a is None or axes_b is None:
        return None
    normals_a, edges_a = axes_a
    normals_b, edges_b = axes_b
    if len(edges_a) * len(edges_b) > _MAX_SAT_AXES:
        return None
    cross = np.cross(edges_a[:, None, :], edges_b[None, :, :]).reshape(-1, 3)
    axes = np.concatenate([normals_a, normals_b, cross])
    norms = np.linalg.norm(axes, axis=1)
    keep = norms > 1e-12
    if not bool(keep.any()):
        return None
    axes = axes[keep] / norms[keep][:, None]
    a = np.asarray(body_a.core, dtype=np.float64)
    b = np.asarray(body_b.core, dtype=np.float64)
    proj_a = a @ axes.T
    proj_b = b @ axes.T
    a_past_b = proj_a.max(axis=0) - proj_b.min(axis=0)
    b_past_a = proj_b.max(axis=0) - proj_a.min(axis=0)
    overlap = np.minimum(a_past_b, b_past_a)
    k = int(np.argmin(overlap))
    depth = -float(overlap[k])
    # Orient the axis so ``a`` sits on its positive side: then ``a``'s minimum
    # projection is the face buried in ``b`` and ``b``'s maximum is the plane
    # it is buried under.
    axis = axes[k] if b_past_a[k] <= a_past_b[k] else -axes[k]
    heights = a @ axis
    deepest = a[np.isclose(heights, heights.min(), rtol=0.0, atol=1e-12)]
    p_a = deepest.mean(axis=0)
    p_b = p_a - depth * axis
    return depth, axis, p_a, p_b


def _signed_distance(
    body_a: ConvexBody, body_b: ConvexBody
) -> tuple[float, Any, Any, Any, float, str]:
    """``(signed distance, p_a, p_b, direction, duality gap, method)`` for one pair.

    ``direction`` is the unit ``b``→``a`` direction: the separating direction
    when the cores are apart, the minimum-translation axis when they overlap.
    It is carried out separately because at a flush contact the two witness
    points coincide and differencing them yields nothing.

    Inflating two convex bodies by balls shifts their signed distance by exactly
    the sum of the radii, on both the separated and the penetrating branch, so
    the ball radii come off the core answer with no special case.
    """
    import numpy as np

    inflate = body_a.radius + body_b.radius
    distance, p_a, p_b = _gjk(body_a.core, body_b.core)
    if distance > _OVERLAP_TOL_M and p_a is not None and p_b is not None:
        lower = separating_axis_bound(body_a.core, body_b.core, p_b - p_a)
        gap = abs(distance - lower)
        unit = (np.asarray(p_a) - np.asarray(p_b)) / distance
        if inflate > 0.0:
            p_a = np.asarray(p_a) - unit * body_a.radius
            p_b = np.asarray(p_b) + unit * body_b.radius
        return distance - inflate, p_a, p_b, unit, gap, "gjk"
    solved = _sat_penetration_depth(body_a, body_b)
    if solved is None:
        return -inflate, None, None, None, float("inf"), "gjk-overlap"
    depth, axis, p_a, p_b = solved
    # The balls push each witness further along the axis: ``a``'s deeper into
    # ``b``, ``b``'s further out, keeping ``p_a - p_b == (depth - inflate) * axis``.
    return depth - inflate, p_a - axis * body_a.radius, p_b + axis * body_b.radius, axis, 0.0, "sat"


def convex_geom_distance(
    model: Any,
    data: Any,
    geom_a: int,
    geom_b: int,
    *,
    distmax_m: float | None = None,
    arc_segments: int = _ARC_SEGMENTS,
) -> ConvexDistance:
    """Signed distance between two MuJoCo geoms, with a proof attached.

    The replacement for ``mujoco.mj_geomDistance`` on the collision-evidence
    path — see this module's docstring for what that call does wrong, and how
    it was measured. Positive is a gap, negative is penetration depth, and the
    reported ``distance_m`` is always the **closest** end of the certified
    bracket, so nothing here can make a pair look safer than it provably is.

    Args:
        model: ``mjModel``.
        data: ``mjData``, already forwarded to the state being measured.
        geom_a: First geom id.
        geom_b: Second geom id.
        distmax_m: Probe window. When given, a pair the centre-axis separating
            bound already **proves** to be further apart than this returns
            early with ``method="beyond-window"`` — certified, and three orders
            of magnitude cheaper than solving it. That rejection is a proof,
            not a heuristic: the bound is valid for every direction, so a pair
            it discards cannot be inside the window.
        arc_segments: Polygon resolution for the bracketed round types.

    Returns:
        A :class:`ConvexDistance`. ``certified`` is ``False`` — with a reason —
        when a geom has no bounded convex hull (plane, heightfield, SDF), when a
        round type's bracket is wider than 0.1 mm, when the separating-axis
        certificate does not close, or when an overlapping pair's exact axis set
        is unavailable or too large. In none of those cases is a plausible
        number substituted for a defensible one.

    Example:
        >>> result = convex_geom_distance(model, data, a, b)  # doctest: +SKIP
        >>> result.lower_m <= result.distance_m <= result.upper_m  # doctest: +SKIP
        True
    """
    import mujoco  # reason: optional sim dep

    bracket_a = geom_convex_bracket(model, data, geom_a, arc_segments=arc_segments)
    bracket_b = geom_convex_bracket(model, data, geom_b, arc_segments=arc_segments)
    if bracket_a is None or bracket_b is None:
        missing = geom_a if bracket_a is None else geom_b
        kind = mujoco.mjtGeom(int(model.geom_type[missing])).name
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, missing)
        return ConvexDistance(
            distance_m=0.0,
            lower_m=float("-inf"),
            upper_m=float("inf"),
            certified=False,
            uncertified_reason=(
                f"geom {name!r} is {kind}, which has no bounded convex hull; "
                "this probe does not measure it"
            ),
            witness_a=None,
            witness_b=None,
            direction=None,
            duality_gap_m=float("inf"),
            method="unsupported-geom-type",
        )

    inner_a, outer_a = bracket_a
    inner_b, outer_b = bracket_b
    if distmax_m is not None:
        import numpy as np

        axis = np.asarray(data.geom_xpos[geom_b], dtype=np.float64) - np.asarray(
            data.geom_xpos[geom_a], dtype=np.float64
        )
        bound = (
            separating_axis_bound(outer_a.core, outer_b.core, axis)
            - outer_a.radius
            - outer_b.radius
        )
        if bound > distmax_m:
            return ConvexDistance(
                distance_m=bound,
                lower_m=bound,
                upper_m=float("inf"),
                certified=True,
                uncertified_reason="",
                witness_a=None,
                witness_b=None,
                direction=None,
                duality_gap_m=float("inf"),
                method="beyond-window",
            )
    lower, p_a, p_b, direction, gap_outer, method = _signed_distance(outer_a, outer_b)
    if inner_a is outer_a and inner_b is outer_b:
        upper, gap_inner = lower, gap_outer
    else:
        upper, _pa, _pb, _dir, gap_inner, _method = _signed_distance(inner_a, inner_b)
    if method == "gjk-overlap":
        # Overlap is proved; the depth is not. ``-(r_a + r_b)`` is the LEAST
        # negative the answer can be, so it bounds from above and nothing
        # bounds it from below. Reported as such, and never certified.
        upper, lower = lower, float("-inf")

    reasons: list[str] = []
    if upper - lower > _BRACKET_TOL_M:
        reasons.append(
            f"the inscribed/circumscribed bracket is {(upper - lower) * 1e3:.4g} mm wide; "
            f"raise arc_segments above {arc_segments} to close it"
        )
    duality = max(gap_outer, gap_inner)
    if duality == float("inf"):
        reasons.append(
            "the cores overlap and their exact SAT axis set is unavailable or beyond the "
            "enumeration cap, so the penetration depth was not measured"
        )
    elif duality > _DUALITY_TOL_M:
        reasons.append(
            f"the separating-axis certificate did not close ({duality * 1e3:.3g} mm residual)"
        )
    return ConvexDistance(
        distance_m=upper if lower == float("-inf") else lower,
        lower_m=lower,
        upper_m=upper,
        certified=not reasons,
        uncertified_reason="; ".join(reasons),
        witness_a=None if p_a is None else (float(p_a[0]), float(p_a[1]), float(p_a[2])),
        witness_b=None if p_b is None else (float(p_b[0]), float(p_b[1]), float(p_b[2])),
        direction=(
            None
            if direction is None
            else (float(direction[0]), float(direction[1]), float(direction[2]))
        ),
        duality_gap_m=duality,
        method=method,
    )


def witness_clearance_m(
    model: Any, data: Any, geom_id: int, point: Any, *, arc_segments: int = _ARC_SEGMENTS
) -> float:
    """A certified lower bound on how far ``point`` lies **outside** ``geom_id``.

    Zero or below means the point may be on or inside the geom; positive is
    proof it is outside, by at least that much. This is the contradiction
    detector for a *claimed* nearest-point witness, whoever produced it: the
    endpoints of a nearest-pair segment lie on their own geoms by definition, so
    a positive clearance on both ends refutes the pair. That is exactly the tell
    that exposed ``mj_geomDistance``'s native-CCD failure — a ``+0.000000``
    reading whose 126.264 mm ``fromto`` had both ends over half a metre outside
    either body.

    For a geom with no bounded hull the bound falls back to the bounding sphere,
    which is still valid; a geom with neither (a plane) yields ``-inf``, which
    accuses nothing.

    Args:
        model: ``mjModel``.
        data: ``mjData``.
        geom_id: The geom the point is claimed to lie on.
        point: World-frame ``(3,)`` point.
        arc_segments: Polygon resolution for the bracketed round types.

    Returns:
        Metres. Positive proves the point is outside the geom.
    """
    import numpy as np

    p = np.asarray(point, dtype=np.float64).reshape(1, 3)
    rbound = float(model.geom_rbound[geom_id])
    centre = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    sphere_bound = (
        float(np.linalg.norm(p.ravel() - centre)) - rbound if rbound > 0.0 else float("-inf")
    )
    bracket = geom_convex_bracket(model, data, geom_id, arc_segments=arc_segments)
    if bracket is None:
        return sphere_bound
    outer = bracket[1]
    distance, _p_a, _p_b = _gjk(outer.core, p)
    return max(sphere_bound, distance - outer.radius)
