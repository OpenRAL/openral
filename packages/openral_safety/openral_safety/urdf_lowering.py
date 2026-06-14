"""Offline URDF(+SRDF) → manifest collision-model lowering tool (ADR-0030).

Produces the hand-reviewable ``collision_geometry`` + ``allowed_collision_pairs``
that ``robot.yaml`` carries and ``collision_params_from_description`` consumes:

* **Geometry** — fit one conservative capsule/sphere per link from the URDF
  ``<collision>`` (primitive → direct map; mesh → PCA bounding capsule that
  contains every vertex, so the safety check never under-covers).
* **ACM** — from the SRDF ``disable_collisions`` block where one exists, else a
  MoveIt-Setup-Assistant-style random-pose sweep that disables adjacent /
  always-colliding / never-colliding pairs, tested with the **kernel's own**
  capsule distance (``mjcf_lowering._seg_seg_distance``) so the generated matrix
  matches what the kernel checks.

Heavy deps (``yourdfpy``, ``trimesh``) are imported lazily — install the
optional ``[lowering]`` group. Pure: no ROS, no I/O beyond reading the source
files passed in.

Later tasks extend this module with geometry fitting, the sampling ACM, and the
top-level ``lower_robot``. This file starts with the SRDF parser.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openral_core import CapsuleShape, LinkCollisionGeometry, RobotDescription, SphereShape
from openral_core.urdf_resolve import resolve_urdf_path

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _Arr = NDArray[np.float64]

__all__ = [
    "LoweredCollisionModel",
    "acm_for_geometry",
    "fit_capsule_to_vertices",
    "lower_joint_fk",
    "lower_link_geometry",
    "lower_robot",
    "lower_robot_from_mjcf",
    "parse_srdf_disabled_pairs",
    "sample_acm_from_urdf",
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


def fit_capsule_to_vertices(vertices: _Arr) -> tuple[CapsuleShape, _Origin]:
    """Fit a conservative bounding capsule (segment along +Z) to a vertex cloud.

    PCA via SVD: the dominant principal component is the capsule axis. ``length_m``
    is the span of the projections onto that axis; ``radius_m`` is the max distance
    of any vertex from the axis line. Every vertex therefore lies inside the result
    — a conservative over-approximation, so the safety check never under-covers
    (ADR-0030 §2). Returns the ``CapsuleShape`` plus its ``origin_xyz_rpy`` in the
    same frame as ``vertices``: the segment midpoint and the rotation taking local
    +Z onto the principal axis.

    Args:
        vertices: ``(N, 3)`` point cloud (N ≥ 1) in the link frame.

    Returns:
        ``(CapsuleShape, origin_xyz_rpy)``.
    """
    import numpy as np

    pts = np.asarray(vertices, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Principal axis = first right-singular vector of the centered cloud.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0] / np.linalg.norm(vh[0])
    proj = centered @ axis
    length = float(proj.max() - proj.min())
    perp = centered - np.outer(proj, axis)
    radius = max(float(np.linalg.norm(perp, axis=1).max()), 1e-4)
    # Segment centred on the projection midpoint (not the centroid).
    center = centroid + axis * float((proj.max() + proj.min()) / 2.0)
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
    return CapsuleShape(radius_m=radius, length_m=length), origin


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
    """Rim points at both caps of a +Z cylinder (radius, length) — bounds R and L."""
    import numpy as np

    h = length / 2.0
    ang = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros_like(ang)], axis=1)
    return np.vstack([ring + np.array([0.0, 0.0, h]), ring + np.array([0.0, 0.0, -h])])


def _sphere_vertices(radius: float) -> _Arr:
    """A coarse surface sampling of a sphere (so it bounds when mixed with geoms)."""
    import numpy as np

    u = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    v = np.linspace(0.0, math.pi, 6)
    uu, vv = np.meshgrid(u, v)
    return np.stack(
        [radius * np.cos(uu) * np.sin(vv), radius * np.sin(uu) * np.sin(vv), radius * np.cos(vv)],
        axis=-1,
    ).reshape(-1, 3)


def _apply(transform: _Arr, pts: _Arr) -> _Arr:
    """Apply a 4×4 homogeneous transform to an ``(N, 3)`` cloud."""
    import numpy as np

    return np.asarray((pts @ transform[:3, :3].T) + transform[:3, 3], dtype=np.float64)


def lower_link_geometry(urdf_path: str) -> list[LinkCollisionGeometry]:
    """One conservative ``LinkCollisionGeometry`` per URDF link with a ``<collision>``.

    Primitive collisions map by exact analytic bounds (box → 8 corners; cylinder →
    cap rims; sphere → an exact :class:`SphereShape`); mesh collisions load their
    vertices (``trimesh``) and PCA-fit a bounding capsule. All vertices are first
    transformed by the ``<collision><origin>`` into the link frame, so the emitted
    ``origin_xyz_rpy`` is link-relative (what the kernel's forward kinematics
    expects). Links with no collision element — or fewer than 4 cloud points and no
    sphere — are skipped.

    A link whose sole collision is a single sphere emits an exact ``SphereShape``;
    every other case (mesh / box / cylinder / multi-geom) emits a capsule that
    contains the union of all its collision vertices.
    """
    import numpy as np

    model = _load_urdf_model(urdf_path)
    handler = getattr(model, "_filename_handler", None)

    out: list[LinkCollisionGeometry] = []
    for link_name, link in model.link_map.items():  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        collisions = list(getattr(link, "collisions", None) or [])
        if not collisions:
            continue
        # Exact-sphere fast path: a single sphere collision → an exact SphereShape.
        if len(collisions) == 1 and getattr(collisions[0].geometry, "sphere", None) is not None:
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
        for col in collisions:
            verts = _collision_local_vertices(col, handler)
            if verts is None or len(verts) == 0:
                continue
            clouds.append(_apply(_origin_matrix(col.origin), verts))
        if not clouds:
            continue
        cloud = np.vstack(clouds)
        if len(cloud) < 4:
            continue
        shape, origin = fit_capsule_to_vertices(cloud)
        out.append(LinkCollisionGeometry(link_name=link_name, shape=shape, origin_xyz_rpy=origin))
    return out


def _collision_local_vertices(col: object, handler: object) -> _Arr | None:
    """Vertices of one ``<collision>`` geometry, in the geometry's own local frame.

    Mesh → loaded vertices × scale; box / cylinder → analytic samples; sphere →
    a coarse surface sampling (so a sphere mixed with other geoms still bounds).
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
                f"collision mesh not found, link will carry no geometry: {mesh.filename!r}",
                stacklevel=2,
            )
            return None
        loaded = trimesh.load(path, force="mesh")
        verts = np.asarray(loaded.vertices, dtype=np.float64)  # type: ignore[attr-defined]  # reason: force="mesh" yields a Trimesh with .vertices
        scale = getattr(mesh, "scale", None)
        if scale is not None:
            verts = verts * np.asarray(scale, dtype=np.float64)
        return np.asarray(verts, dtype=np.float64)
    return None


# ── ACM: MoveIt-Setup-Assistant-style random-pose sampling fallback ────────────

_RNG_SEED = 20260610
_N_SAMPLES = 2000
_SAMPLE_MARGIN_M = 0.0


def _capsule_segment_radius(
    shape: CapsuleShape | SphereShape, origin: _Origin
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Capsule (in its link frame) → (endpoint0, endpoint1, radius), local frame.

    A sphere degenerates to a zero-length segment at its origin.
    """
    import numpy as np

    from openral_safety.mjcf_lowering import _rpy_to_mat

    cx, cy, cz, roll, pitch, yaw = origin
    if isinstance(shape, SphereShape):
        return (cx, cy, cz), (cx, cy, cz), shape.radius_m
    rot = np.asarray(_rpy_to_mat(roll, pitch, yaw), dtype=np.float64).reshape(3, 3)
    z_axis = rot[:, 2]  # local +Z in the link frame
    half = shape.length_m / 2.0
    center = np.array([cx, cy, cz], dtype=np.float64)
    p0 = center - z_axis * half
    p1 = center + z_axis * half
    return (
        (float(p0[0]), float(p0[1]), float(p0[2])),
        (
            float(p1[0]),
            float(p1[1]),
            float(p1[2]),
        ),
        shape.radius_m,
    )


def _world_segment(
    link_tf: _Arr, p0: tuple[float, float, float], p1: tuple[float, float, float]
) -> tuple[list[float], list[float]]:
    """Transform a link-frame segment by the 4×4 link pose into the base frame."""
    import numpy as np

    rot, trans = link_tf[:3, :3], link_tf[:3, 3]
    w0 = rot @ np.asarray(p0, dtype=np.float64) + trans
    w1 = rot @ np.asarray(p1, dtype=np.float64) + trans
    return list(w0), list(w1)


def _joint_limit_arrays(model: object) -> tuple[_Arr, _Arr]:
    """(lower, upper) sampling bounds per actuated joint (continuous → [-π, π])."""
    import numpy as np

    lo: list[float] = []
    hi: list[float] = []
    for joint in model.actuated_joints:  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        limit = getattr(joint, "limit", None)
        lower = getattr(limit, "lower", None) if limit is not None else None
        upper = getattr(limit, "upper", None) if limit is not None else None
        if lower is None or upper is None or lower == upper:
            lo.append(-math.pi)
            hi.append(math.pi)
        else:
            lo.append(float(lower))
            hi.append(float(upper))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _pair_collision_counts(
    model: object,
    geoms: dict[str, LinkCollisionGeometry],
    *,
    n_samples: int,
    seed: int,
    margin_m: float,
) -> tuple[list[str], dict[frozenset[str], int]]:
    """Sweep random poses; count, per link pair, how many collide under the capsules.

    FK is the kernel's: each link's capsule (from ``geoms``) is placed by the URDF
    transform and tested with ``mjcf_lowering._seg_seg_distance``. Deterministic under
    ``seed`` / ``n_samples``. Returns the geometry-bearing link list and the counts.
    """
    import numpy as np

    from openral_safety.mjcf_lowering import _seg_seg_distance

    links = [ln for ln in model.link_map if ln in geoms]  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    local_seg = {
        ln: _capsule_segment_radius(geoms[ln].shape, geoms[ln].origin_xyz_rpy) for ln in links
    }
    rng = np.random.default_rng(seed)
    lo, hi = _joint_limit_arrays(model)
    counts: dict[frozenset[str], int] = {}
    for _ in range(n_samples):
        q = lo + (hi - lo) * rng.random(lo.shape)
        model.update_cfg(q)  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        world = {}
        for ln in links:
            tf = np.asarray(model.get_transform(ln), dtype=np.float64)  # type: ignore[attr-defined]  # reason: yourdfpy URDF
            p0l, p1l, r = local_seg[ln]
            world[ln] = (*_world_segment(tf, p0l, p1l), r)
        for i, a in enumerate(links):
            for b in links[i + 1 :]:
                (a0, a1, ra) = world[a]
                (b0, b1, rb) = world[b]
                if _seg_seg_distance(a0, a1, b0, b1) - ra - rb <= margin_m:
                    key = frozenset({a, b})
                    counts[key] = counts.get(key, 0) + 1
    return links, counts


def acm_for_geometry(
    urdf_path: str,
    geoms: dict[str, LinkCollisionGeometry],
    *,
    srdf_path: str | None = None,
    n_samples: int = _N_SAMPLES,
    seed: int = _RNG_SEED,
    margin_m: float = _SAMPLE_MARGIN_M,
) -> _AcmPairs:
    """The self-collision ACM for a specific per-link capsule geometry ``geoms``.

    The kernel checks collisions with ``geoms``, so the ACM is computed against the
    *same* capsules. A pair is disabled when it is:

    * **adjacent** — directly joint-connected;
    * **always-colliding** — overlaps in every sampled pose under ``geoms`` (the
      capsule-junction artifacts a mesh-based SRDF omits, e.g. a short link making
      its skip-one neighbours' capsules overlap — these MUST be disabled or the
      kernel false-E-stops every step);
    * **never-able-to-collide** — ONLY from the SRDF ``disable_collisions`` (mesh
      ground truth) when ``srdf_path`` is given.

    So with an SRDF: ``ACM = adjacent ∪ always(capsule) ∪ SRDF``. Without one (the
    sampling fallback): ``ACM = adjacent ∪ always(capsule)`` — every other pair stays
    **checked**. A random-pose sweep cannot *prove* a pair never collides (it can
    miss a rare config, especially between independent kinematic branches on a
    bimanual / humanoid robot), so sampling never auto-disables a "never-collide"
    pair; that's safe but mesh-authoritative SRDFs stay efficient. Deterministic
    under the pinned seed.
    """
    model = _load_urdf_model(urdf_path)
    links, counts = _pair_collision_counts(
        model, geoms, n_samples=n_samples, seed=seed, margin_m=margin_m
    )

    disabled: _AcmPairs = set()
    for joint in model.robot.joints:  # type: ignore[attr-defined]  # reason: yourdfpy URDF
        if joint.parent in geoms and joint.child in geoms and joint.parent != joint.child:
            disabled.add(frozenset({joint.parent, joint.child}))  # adjacent
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            if counts.get(frozenset({a, b}), 0) == n_samples:  # always-colliding (capsule)
                disabled.add(frozenset({a, b}))

    if srdf_path is not None:
        disabled |= parse_srdf_disabled_pairs(srdf_path)  # mesh-proven "never"
    return disabled


def sample_acm_from_urdf(
    urdf_path: str,
    *,
    n_samples: int = _N_SAMPLES,
    seed: int = _RNG_SEED,
    margin_m: float = _SAMPLE_MARGIN_M,
) -> _AcmPairs:
    """MoveIt-Setup-Assistant-style ACM from a URDF (the no-SRDF fallback).

    Lowers the URDF's own collision geometry and runs :func:`acm_for_geometry`
    without an SRDF: disables adjacent / always-colliding / never-colliding pairs,
    tested with the kernel's own capsule distance. Deterministic under the pinned
    ``seed`` / ``n_samples``.
    """
    geoms = {g.link_name: g for g in lower_link_geometry(urdf_path)}
    return acm_for_geometry(
        urdf_path, geoms, srdf_path=None, n_samples=n_samples, seed=seed, margin_m=margin_m
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


def _load_urdf_model(urdf_ref: str) -> object:
    """Load a yourdfpy model from a manifest ``urdf_path`` form OR a file path.

    Supports ``robot_descriptions:<module>`` (processes the module's xacro via
    ``xacrodoc`` — the UR / Flexiv arms ship only xacro), a ``python:…`` /
    repo-relative / absolute path resolved by :func:`resolve_urdf_path`, and a bare
    on-disk file path. Raises ``ValueError`` when the reference can't be loaded.
    """
    import yourdfpy  # type: ignore[import-untyped]  # reason: yourdfpy ships no stubs

    if urdf_ref.startswith("robot_descriptions:"):
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        module = urdf_ref.split(":", 1)[1]
        return load_robot_description(
            module,
            load_meshes=False,
            load_collision_meshes=True,
            build_collision_scene_graph=True,
        )
    import os

    path = urdf_ref if os.path.isfile(urdf_ref) else resolve_urdf_path(urdf_ref)
    if path is None:
        raise ValueError(f"urdf_path {urdf_ref!r} did not resolve")
    return yourdfpy.URDF.load(
        path, build_collision_scene_graph=True, load_meshes=False, load_collision_meshes=True
    )


def lower_joint_fk(robot: RobotDescription, urdf_ref: str) -> dict[str, tuple[_Vec3, _Vec3, _Vec3]]:
    """Per-manifest-joint FK (``origin_xyz``, ``origin_rpy``, ``axis_xyz``) from the URDF.

    The kernel computes link poses from the manifest joints' fixed parent→joint
    transform + axis (ADR-0030); a manifest that only declares the chain topology
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

    model = _load_urdf_model(urdf_ref)
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
    n_samples: int = _N_SAMPLES,
    seed: int = _RNG_SEED,
    margin_m: float = _SAMPLE_MARGIN_M,
) -> LoweredCollisionModel:
    """Lower joint FK + sampling ACM from a robot's MJCF, keeping its manifest geometry.

    For MJCF-native robots with **no URDF** whose collision geoms are meshes (which
    ``mjcf_lowering``'s primitive path skips) — e.g. the bimanual ``openarm``. The
    hand-authored manifest capsules are kept; FK is the MJCF transform from each
    joint's ``parent_link`` to its ``child_link`` at the rest pose (matched to the
    MJCF by ``sim_joint_name``), and the ACM is the capsule sweep run with **mujoco
    forward kinematics** over the manifest geometry. ``acm_source = "mjcf"``.

    Raises ``ValueError`` if the robot has no ``sim.mjcf_uri``.
    """
    import mujoco
    import numpy as np

    from openral_safety.mjcf_lowering import _seg_seg_distance

    if robot.sim is None or not robot.sim.mjcf_uri:
        raise ValueError(f"{robot.name}: no urdf_path and no sim.mjcf_uri to lower from")
    from openral_hal._mujoco_arm import resolve_mjcf_uri

    model = mujoco.MjModel.from_xml_path(resolve_mjcf_uri(robot.sim.mjcf_uri))
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
    local_seg = {
        ln: _capsule_segment_radius(geoms[ln].shape, geoms[ln].origin_xyz_rpy) for ln in links
    }
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

    rng = np.random.default_rng(seed)
    counts: dict[frozenset[str], int] = {}
    for _ in range(n_samples):
        mujoco.mj_resetData(model, data)
        for adr, lo, hi in sweep:
            data.qpos[adr] = lo + (hi - lo) * rng.random()
        mujoco.mj_kinematics(model, data)
        world = {}
        for ln in links:
            p0l, p1l, r = local_seg[ln]
            world[ln] = (*_world_segment(body_tf(link_body[ln]), p0l, p1l), r)
        for i, a in enumerate(links):
            for b in links[i + 1 :]:
                (a0, a1, ra) = world[a]
                (b0, b1, rb) = world[b]
                if _seg_seg_distance(a0, a1, b0, b1) - ra - rb <= margin_m:
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
) -> LoweredCollisionModel:
    """Lower a robot's URDF/SRDF into the manifest collision blocks (ADR-0030).

    ACM source precedence: an explicit ``srdf_path`` → the manifest's ``srdf_path``
    → the URDF random-pose sampling fallback. The ACM is scoped to links that carry
    geometry, so an SRDF's hand/finger rows don't leak into an arm-only model.
    ``acm_only`` / ``geometry_only`` restrict the output so hand-tuned geometry on
    an existing safety robot isn't churned when only the ACM needs refreshing.

    Args:
        robot: The robot manifest (must declare ``urdf_path``).
        srdf_path: Override SRDF; falls back to ``robot.srdf_path`` then sampling.
        acm_only: Emit only ``allowed_collision_pairs`` (keep existing geometry).
        geometry_only: Emit only ``collision_geometry`` (skip the ACM).

    Returns:
        A :class:`LoweredCollisionModel`.

    Raises:
        ValueError: If ``robot.urdf_path`` is unset or does not resolve.
    """
    if robot.urdf_path is None:
        # MJCF-native robots (no URDF; mesh collision) lower from their sim MJCF —
        # geometry stays the manifest's hand-authored capsules.
        if robot.sim is not None and robot.sim.mjcf_uri:
            return lower_robot_from_mjcf(robot)
        raise ValueError(f"{robot.name}: urdf_path is required to lower a collision model")
    if robot.urdf_path.startswith("robot_descriptions:"):
        # xacro-only robots (UR / Flexiv) load via the robot_descriptions loader;
        # don't run the file resolver (it would warn about a non-existent path).
        urdf_ref = robot.urdf_path
    else:
        urdf = resolve_urdf_path(robot.urdf_path)
        if urdf is None:
            raise ValueError(f"{robot.name}: urdf_path {robot.urdf_path!r} did not resolve")
        urdf_ref = urdf

    # Links the manifest actually models (its kinematic chain). Generated geometry
    # is scoped to these so an orphan URDF link (e.g. panda_leftfinger, absent from
    # a manifest that models a single panda_finger_pair) can't reach the kernel.
    chain_links = {j.parent_link for j in robot.joints} | {j.child_link for j in robot.joints}

    geometry: list[LinkCollisionGeometry] = []
    joint_fk: dict[str, tuple[_Vec3, _Vec3, _Vec3]] = {}
    if not acm_only:
        geometry = [g for g in lower_link_geometry(urdf_ref) if g.link_name in chain_links]
        joint_fk = lower_joint_fk(robot, urdf_ref)

    pairs: list[tuple[str, str]] = []
    source = "sampling"
    used_srdf = srdf_path or robot.srdf_path
    if not geometry_only:
        # The kernel checks collisions with the SAME capsules it will load: the
        # existing manifest geometry under acm_only, else the freshly lowered set.
        # The ACM is computed against that geometry so a mesh-based SRDF's omitted
        # capsule-junction pairs (always-colliding under the conservative capsules)
        # are added — otherwise the kernel would false-E-stop every step.
        geom_list = robot.collision_geometry if acm_only else geometry
        geom_by_link = {g.link_name: g for g in geom_list}
        disabled = acm_for_geometry(urdf_ref, geom_by_link, srdf_path=used_srdf)
        source = "srdf" if used_srdf else "sampling"
        pairs = _scoped_sorted_pairs(disabled, set(geom_by_link))

    return LoweredCollisionModel(
        collision_geometry=geometry,
        allowed_collision_pairs=pairs,
        acm_source=source,
        srdf_path=used_srdf,
        joint_fk=joint_fk,
    )
