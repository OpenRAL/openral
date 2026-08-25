"""The Python mirror of the safety kernel's narrow-phase collision predicates.

``cpp/openral_safety_kernel/src/collision.cpp`` is normative: the C++ kernel is
what stops the robot. Anything offline that decides *what the kernel will do* —
the ACM always-colliding sweep in :mod:`openral_safety.urdf_lowering`, the
collision studies — must ask the same question with the same geometry, or it
reasons about a robot that does not exist. Issue #155 is exactly that failure:
the ACM sweep modelled a ``BoxShape`` as its inscribed sphere while the kernel
checked the true oriented box.

Each function here is a line-by-line port of its C++ counterpart:

===========================  ==========================================
this module                  ``collision.cpp``
===========================  ==========================================
:func:`box_box_distance`     ``box_box_distance`` (L327) — 15-axis SAT
:func:`box_capsule_distance` ``box_capsule_distance`` (L293) — ternary
:func:`capsule_distance`     ``capsule_distance`` (L252) — segment pair
===========================  ==========================================

All three return a **surface gap**: positive = disjoint by that much, ``<= 0`` =
overlap. The kernel trips a pair when the gap is ``<= self_collision_margin_m``.

Every function is batched over a leading ``(N, ...)`` axis so an offline sweep
can evaluate a whole grid of configurations at once; pass ``N = 1`` transforms
for the scalar case.

Example:
    >>> import numpy as np
    >>> a = np.eye(4)[None]
    >>> b = np.eye(4)[None]
    >>> b[0, 0, 3] = 1.0  # 1 m apart on +X
    >>> half = (0.1, 0.1, 0.1)
    >>> float(box_box_distance(a, half, b, half).round(3))
    0.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from openral_core import BoxShape, CapsuleShape, CollisionShape, SphereShape
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _Arr = NDArray[np.float64]

# `CollisionShape` is the canonical discriminated union (`openral_core.schemas`,
# #165). Aliasing it here rather than restating the member list keeps this
# module's dispatch and the schema in lockstep: a primitive added there cannot
# quietly fall through a branch here without mypy flagging the gap.
_Shape: TypeAlias = CollisionShape

__all__ = [
    "bounding_capsule_segment",
    "box_box_distance",
    "box_capsule_distance",
    "capsule_distance",
    "eroded_shape",
    "shape_distance",
    "shape_max_extent_m",
]

# Matches the C++ degenerate-axis guard (``collision.cpp`` L351).
_DEGENERATE_AXIS = 1e-9
# Matches the C++ ternary-search iteration count (``collision.cpp`` L306).
_TERNARY_ITERS = 48


def _unhandled(shape: object) -> ROSConfigError:
    """Fail closed on a primitive this module does not model.

    Every dispatch here is exhaustive over :data:`CollisionShape` and ends in
    this raise rather than in a bare ``else`` that would treat an unknown
    primitive as a capsule. A silently mis-modelled shape is exactly how the box
    under-approximation of issue #155 survived, and #165 closed the same pattern
    in ``envelope_loader``.
    """
    return ROSConfigError(
        f"collision primitive {type(shape).__name__!r} is not modelled by "
        "openral_safety.kernel_predicates; add it to every dispatch in this "
        "module before adding it to CollisionShape."
    )


def _half(shape: _Shape) -> tuple[float, float, float]:
    """Box half-extents as a plain 3-tuple."""
    return tuple(float(v) for v in shape.half_extents_m)  # type: ignore[return-value, union-attr]  # reason: BoxShape only, guarded by callers


def box_box_distance(a_tf: _Arr, a_half: object, b_tf: _Arr, b_half: object) -> _Arr:
    """Separating-axis surface gap between two oriented boxes, batched.

    A port of ``box_box_distance`` (``collision.cpp`` L327): project both boxes
    onto each of the 15 candidate axes (3 + 3 face normals, 9 edge-edge cross
    products) and return the **largest** gap. For two convex polytopes the
    optimal separating direction is always a face normal or an edge-edge cross
    product, so this maximum is the exact signed separation — the true distance
    when disjoint, and minus the penetration depth when overlapping.

    Args:
        a_tf: ``(N, 4, 4)`` world poses of box A.
        a_half: A's half-extents ``(3,)``.
        b_tf: ``(N, 4, 4)`` world poses of box B.
        b_half: B's half-extents ``(3,)``.

    Returns:
        ``(N,)`` surface gaps; ``<= 0`` means overlap.
    """
    ar, br = a_tf[:, :3, :3], b_tf[:, :3, :3]
    ah = np.asarray(a_half, dtype=np.float64)
    bh = np.asarray(b_half, dtype=np.float64)
    dc = b_tf[:, :3, 3] - a_tf[:, :3, 3]
    # Column k of the rotation is local axis k expressed in the world frame.
    a_ax = [ar[:, :, k] for k in range(3)]
    b_ax = [br[:, :, k] for k in range(3)]
    axes = [*a_ax, *b_ax] + [np.cross(a_ax[i], b_ax[j]) for i in range(3) for j in range(3)]
    best: _Arr = np.full(a_tf.shape[0], -np.inf)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        ok = norm >= _DEGENERATE_AXIS  # parallel edges — already covered by faces
        unit = np.where(ok[:, None], axis / np.where(ok, norm, 1.0)[:, None], 0.0)
        ra = sum(ah[k] * np.abs(np.einsum("ij,ij->i", a_ax[k], unit)) for k in range(3))
        rb = sum(bh[k] * np.abs(np.einsum("ij,ij->i", b_ax[k], unit)) for k in range(3))
        gap = np.abs(np.einsum("ij,ij->i", dc, unit)) - ra - rb
        best = np.where(ok, np.maximum(best, gap), best)
    return best


def _point_aabb_distance(p: _Arr, half: _Arr) -> _Arr:
    """Distance from batched points to the axis-aligned box ``[-half, +half]``."""
    out: _Arr = np.linalg.norm(np.clip(np.abs(p) - half, 0.0, None), axis=-1)
    return out


def _capsule_endpoints(tf: _Arr, half_length: float) -> tuple[_Arr, _Arr]:
    """Capsule segment endpoints: the local +Z axis scaled by ``half_length``."""
    z = tf[:, :3, 2] * half_length
    return tf[:, :3, 3] - z, tf[:, :3, 3] + z


def box_capsule_distance(
    box_tf: _Arr, box_half: object, cap_tf: _Arr, cap_radius: float, cap_half_length: float
) -> _Arr:
    """Surface gap between an oriented box and a capsule, batched.

    A port of ``box_capsule_distance`` (``collision.cpp`` L293): bring the
    capsule's central segment into the box's local frame (where the box is the
    axis-aligned ``[-h, +h]``), then ternary-search ``t`` along the segment. The
    point→AABB distance is convex along a segment, so the search converges to
    the global minimum; subtracting the capsule radius gives the gap.

    Args:
        box_tf: ``(N, 4, 4)`` world poses of the box.
        box_half: The box's half-extents ``(3,)``.
        cap_tf: ``(N, 4, 4)`` world poses of the capsule (segment along local +Z).
        cap_radius: Capsule radius (m).
        cap_half_length: Half the capsule's segment length (m); 0 for a sphere.

    Returns:
        ``(N,)`` surface gaps; ``<= 0`` means overlap.
    """
    half = np.asarray(box_half, dtype=np.float64)
    c0, c1 = _capsule_endpoints(cap_tf, cap_half_length)
    rot = box_tf[:, :3, :3]
    a = np.einsum("ikj,ik->ij", rot, c0 - box_tf[:, :3, 3])  # Rᵀ (p - t)
    b = np.einsum("ikj,ik->ij", rot, c1 - box_tf[:, :3, 3])
    ab = b - a
    lo: _Arr = np.zeros(box_tf.shape[0])
    hi: _Arr = np.ones(box_tf.shape[0])
    for _ in range(_TERNARY_ITERS):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = _point_aabb_distance(a + m1[:, None] * ab, half)
        f2 = _point_aabb_distance(a + m2[:, None] * ab, half)
        closer = f1 < f2
        hi = np.where(closer, m2, hi)
        lo = np.where(closer, lo, m1)
    tm = 0.5 * (lo + hi)
    return _point_aabb_distance(a + tm[:, None] * ab, half) - cap_radius


def capsule_distance(
    a_tf: _Arr, a_radius: float, a_half_length: float, b_tf: _Arr, b_radius: float, b_hl: float
) -> _Arr:
    """Surface gap between two capsules, batched.

    A port of ``capsule_distance`` (``collision.cpp`` L252): the minimum distance
    between the two central segments (Ericson, *Real-Time Collision Detection*
    §5.1.9) minus both radii. A sphere is a capsule with ``half_length = 0``,
    exactly as the kernel models it (``envelope_loader.py`` L709).

    Args:
        a_tf: ``(N, 4, 4)`` world poses of capsule A.
        a_radius: A's radius (m).
        a_half_length: Half A's segment length (m).
        b_tf: ``(N, 4, 4)`` world poses of capsule B.
        b_radius: B's radius (m).
        b_hl: Half B's segment length (m).

    Returns:
        ``(N,)`` surface gaps; ``<= 0`` means overlap.
    """
    p1, q1 = _capsule_endpoints(a_tf, a_half_length)
    p2, q2 = _capsule_endpoints(b_tf, b_hl)
    return _seg_seg_distance(p1, q1, p2, q2) - a_radius - b_radius


def _seg_seg_distance(p1: _Arr, q1: _Arr, p2: _Arr, q2: _Arr) -> _Arr:
    """Batched segment-segment minimum distance (Ericson §5.1.9).

    The same clamped-parametric solution as ``mjcf_lowering._seg_seg_distance``
    and ``collision.cpp`` L127, evaluated over a whole batch at once.
    """
    eps = 1e-12
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    safe_denom = np.where(denom > eps, denom, 1.0)
    s = np.where(denom > eps, np.clip((b * f - c * e) / safe_denom, 0, 1), 0.0)
    t = (b * s + f) / np.where(e > eps, e, 1.0)
    # Re-clamp t out of range, recomputing s (Ericson's two corrective branches).
    below = t < 0.0
    above = t > 1.0
    s = np.where(below, np.clip(-c / np.where(a > eps, a, 1.0), 0, 1), s)
    s = np.where(above, np.clip((b - c) / np.where(a > eps, a, 1.0), 0, 1), s)
    t = np.clip(t, 0.0, 1.0)
    # Degenerate segments: a point against a segment, or two points.
    a_pt, e_pt = a <= eps, e <= eps
    s = np.where(a_pt, 0.0, s)
    t = np.where(a_pt & ~e_pt, np.clip(f / np.where(e > eps, e, 1.0), 0, 1), t)
    t = np.where(e_pt, 0.0, t)
    s = np.where(e_pt & ~a_pt, np.clip(-c / np.where(a > eps, a, 1.0), 0, 1), s)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    out: _Arr = np.linalg.norm(c1 - c2, axis=1)
    return out


def shape_distance(a_shape: _Shape, a_tf: _Arr, b_shape: _Shape, b_tf: _Arr) -> _Arr:
    """Surface gap between two link primitives, dispatched exactly as the kernel.

    ``check_self_collision`` (``collision.cpp`` L527) routes each pair by type:
    box↔box to :func:`box_box_distance`, box↔capsule (either order) to
    :func:`box_capsule_distance`, and everything else — capsules and spheres — to
    :func:`capsule_distance`. This reproduces that routing so an offline sweep
    asks the kernel's question, not an approximation of it.

    Args:
        a_shape: First link's primitive.
        a_tf: ``(N, 4, 4)`` world poses of ``a_shape``'s own frame.
        b_shape: Second link's primitive.
        b_tf: ``(N, 4, 4)`` world poses of ``b_shape``'s own frame.

    Returns:
        ``(N,)`` surface gaps; ``<= 0`` means overlap.
    """
    a_box, b_box = isinstance(a_shape, BoxShape), isinstance(b_shape, BoxShape)
    if a_box and b_box:
        return box_box_distance(a_tf, _half(a_shape), b_tf, _half(b_shape))
    if a_box:
        r, hl = _capsule_rh(b_shape)
        return box_capsule_distance(a_tf, _half(a_shape), b_tf, r, hl)
    if b_box:
        r, hl = _capsule_rh(a_shape)
        return box_capsule_distance(b_tf, _half(b_shape), a_tf, r, hl)
    ar, ahl = _capsule_rh(a_shape)
    br, bhl = _capsule_rh(b_shape)
    return capsule_distance(a_tf, ar, ahl, b_tf, br, bhl)


def _capsule_rh(shape: _Shape) -> tuple[float, float]:
    """(radius, half_length) for a capsule or sphere, as the kernel stores it."""
    if isinstance(shape, SphereShape):
        return float(shape.radius_m), 0.0
    if isinstance(shape, CapsuleShape):
        return float(shape.radius_m), float(shape.length_m) / 2.0
    raise _unhandled(shape)


def shape_max_extent_m(shape: _Shape) -> float:
    """The farthest any point of ``shape`` lies from its own origin.

    Used to bound how far a link's geometry can be from a joint axis, which
    bounds how far it moves when that joint turns.
    """
    if isinstance(shape, BoxShape):
        return float(np.linalg.norm(np.asarray(shape.half_extents_m, dtype=np.float64)))
    if isinstance(shape, SphereShape):
        return float(shape.radius_m)
    if isinstance(shape, CapsuleShape):
        return float(shape.length_m) / 2.0 + float(shape.radius_m)
    raise _unhandled(shape)


def bounding_capsule_segment(
    shape: _Shape, origin: tuple[float, float, float, float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """A capsule that **contains** ``shape`` → ``(endpoint0, endpoint1, radius)``.

    For consumers that can only express a link as a capsule or a sphere sweep —
    today that is the cuRobo/cuMotion plan-time model
    (:mod:`openral_safety.cumotion_config`). The result always **over**-covers,
    which is the only safe direction for a planner: a planner that thinks the arm
    is thinner than it is emits trajectories the kernel then has to E-stop.

    * Sphere → a zero-length segment at its origin, same radius.
    * Capsule → itself: the segment along the local +Z axis, same radius.
    * Box → the segment along its **longest** axis, with radius equal to the
      half-diagonal of the other two half-extents. Every box corner then lies
      exactly on the capsule surface, so the box is contained. (The older
      ``_capsule_segment_radius`` used the *inscribed* sphere here — issue #155 —
      which under-covered by up to 3.1× in radius on ``panda_link5``.)

    Args:
        shape: The link primitive.
        origin: The primitive's ``(x, y, z, roll, pitch, yaw)`` in its link frame.

    Returns:
        ``(p0, p1, radius)`` in the link frame.

    Example:
        >>> from openral_core import BoxShape
        >>> p0, p1, r = bounding_capsule_segment(
        ...     BoxShape(half_extents_m=(0.03, 0.04, 0.10)), (0, 0, 0, 0, 0, 0)
        ... )
        >>> round(r, 4)  # half-diagonal of (0.03, 0.04)
        0.05
    """
    from openral_safety.mjcf_lowering import _rpy_to_mat

    cx, cy, cz, roll, pitch, yaw = origin
    centre = np.array([cx, cy, cz], dtype=np.float64)
    if isinstance(shape, SphereShape):
        return (cx, cy, cz), (cx, cy, cz), float(shape.radius_m)
    rot = np.asarray(_rpy_to_mat(roll, pitch, yaw), dtype=np.float64).reshape(3, 3)
    if isinstance(shape, BoxShape):
        half = np.asarray(shape.half_extents_m, dtype=np.float64)
        long_axis = int(np.argmax(half))
        others = [k for k in range(3) if k != long_axis]
        radius = float(np.hypot(half[others[0]], half[others[1]]))
        direction = rot[:, long_axis] * float(half[long_axis])
    elif isinstance(shape, CapsuleShape):  # the segment is the local +Z axis
        radius = float(shape.radius_m)
        direction = rot[:, 2] * (float(shape.length_m) / 2.0)
    else:
        raise _unhandled(shape)
    p0, p1 = centre - direction, centre + direction
    return (
        (float(p0[0]), float(p0[1]), float(p0[2])),
        (float(p1[0]), float(p1[1]), float(p1[2])),
        radius,
    )


def eroded_shape(shape: _Shape, eps_m: float) -> _Shape | None:
    """``shape`` shrunk by an ``eps_m`` ball — its Minkowski erosion, or ``None``.

    The erosion of a box by a ball of radius ``eps_m`` is the box with every
    half-extent reduced by ``eps_m``; of a capsule or sphere, the same primitive
    with the radius reduced. ``None`` when the erosion is empty (``eps_m`` is at
    least the smallest half-extent / the radius), which a caller must treat as
    "cannot certify" rather than as "no overlap".

    This is the certificate step of the always-colliding proof: if the eroded
    shape still overlaps its partner at a grid node, then every configuration
    within ``eps_m`` of that node also overlaps (see
    ``urdf_lowering._certified_always_colliding``).

    Args:
        shape: The primitive to erode.
        eps_m: Erosion radius (m); must be non-negative.

    Returns:
        The eroded primitive, or ``None`` if it would be empty.
    """
    if eps_m <= 0.0:
        return shape
    if isinstance(shape, BoxShape):
        half = [h - eps_m for h in shape.half_extents_m]
        if min(half) <= 0.0:
            return None
        return BoxShape(half_extents_m=(half[0], half[1], half[2]))
    if isinstance(shape, SphereShape):
        r = shape.radius_m - eps_m
        return SphereShape(radius_m=r) if r > 0.0 else None
    if isinstance(shape, CapsuleShape):
        r = shape.radius_m - eps_m
        return CapsuleShape(radius_m=r, length_m=shape.length_m) if r > 0.0 else None
    raise _unhandled(shape)
