"""Offline MJCF → kernel collision-params lowering tool (ADR-0030).

Reads a compiled MuJoCo model and produces the flat ROS-parameter arrays the
C++ safety kernel's ``load_collision_model`` consumes. This is the source
adapter for robots whose collision geometry lives in an MJCF (the sim-first
fleet) and whose manifest ``joints`` are only the *actuated* DoFs — the kernel
needs the **full kinematic tree** (including fixed mounts and the floating
base), which only the MJCF carries.

Self-collision is computed in the robot's own base frame, so a floating base
joint is treated as a fixed identity root: a rigid base transform applies to
every link equally and cannot change inter-link distances.

Scope (ADR-0030 phase 2): one representative primitive per body (the first
collidable capsule / sphere / cylinder). Cylinders lower to capsules and boxes
to a bounding sphere — both **conservative over-approximations** (the proxy
fully contains the real geom), so the safety check never misses a real
collision of the bounded volume. Bodies with multiple collidable geoms are a
known limitation; a future revision will carry several capsules per link.

This module imports ``mujoco`` lazily so the rest of ``openral_safety`` stays
import-light on hosts without it.
"""

from __future__ import annotations

import math
from typing import Any, cast

__all__ = ["lower_collision_params"]

_Vec = list[float]

# MuJoCo geom type codes (mjtGeom).
_MJ_GEOM_PLANE = 0
_MJ_GEOM_SPHERE = 2
_MJ_GEOM_CAPSULE = 3
_MJ_GEOM_CYLINDER = 5
_MJ_GEOM_BOX = 6


def _quat_wxyz_to_rpy(mj: Any, quat: Any) -> tuple[float, float, float]:
    """MuJoCo (w, x, y, z) quaternion → fixed-axis XYZ Euler (roll, pitch, yaw).

    Matches the kernel's ``transform_from_xyz_rpy`` convention
    (R = Rz(yaw)·Ry(pitch)·Rx(roll)).
    """
    import numpy as np

    mat = np.zeros(9, dtype=np.float64)
    mj.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))  # row-major 3x3
    pitch = math.asin(max(-1.0, min(1.0, -mat[6])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(mat[7], mat[8])
        yaw = math.atan2(mat[3], mat[0])
    else:  # gimbal lock
        roll = math.atan2(-mat[5], mat[4])
        yaw = 0.0
    return roll, pitch, yaw


def _capsule_from_geom(mj: Any, model: Any, geom_id: int) -> tuple[float, float]:
    """(radius, half_length) for a collidable primitive, conservatively.

    Capsule/cylinder → (size0, size1). Sphere → (size0, 0). Box → bounding
    sphere (radius = ‖half-extents‖, half_length 0).
    """
    gtype = int(model.geom_type[geom_id])
    size = [float(v) for v in model.geom_size[geom_id]]
    if gtype in (_MJ_GEOM_CAPSULE, _MJ_GEOM_CYLINDER):
        return size[0], size[1]
    if gtype == _MJ_GEOM_SPHERE:
        return size[0], 0.0
    if gtype == _MJ_GEOM_BOX:
        return math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2), 0.0
    msg = f"geom {geom_id} has unsupported collidable type {gtype}"
    raise ValueError(msg)


_LOWERABLE_GEOM_TYPES = frozenset(
    {_MJ_GEOM_SPHERE, _MJ_GEOM_CAPSULE, _MJ_GEOM_CYLINDER, _MJ_GEOM_BOX}
)


def _first_collidable_geom(model: Any, body_id: int) -> int | None:
    """First collidable *primitive* geom owned by ``body_id``, or None.

    Mesh and plane geoms are skipped — this lowering only emits convex analytic
    primitives, so a mesh-only body simply carries no capsule (it is not
    self-collision-checked) rather than crashing the lowering. That keeps the
    tool robust across the fleet's mixed mesh/primitive MJCFs; full mesh
    coverage is a future revision.
    """
    for gi in range(int(model.ngeom)):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_contype[gi]) == 0 and int(model.geom_conaffinity[gi]) == 0:
            continue  # visual-only
        if int(model.geom_type[gi]) in _LOWERABLE_GEOM_TYPES:
            return gi
    return None


def _collidable_geoms(model: Any, body_id: int) -> list[int]:
    """Every collidable primitive geom owned by ``body_id`` (mesh/plane skipped)."""
    out: list[int] = []
    for gi in range(int(model.ngeom)):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_contype[gi]) == 0 and int(model.geom_conaffinity[gi]) == 0:
            continue
        if int(model.geom_type[gi]) in _LOWERABLE_GEOM_TYPES:
            out.append(gi)
    return out


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ]  # fmt: skip


def _compose(ar: _Vec, at: _Vec, br: _Vec, bt: _Vec) -> tuple[_Vec, _Vec]:
    """(ar,at) ∘ (br,bt) → (r, t), row-major 3x3 + 3-vec."""
    r = [
        sum(ar[row * 3 + k] * br[k * 3 + col] for k in range(3))
        for row in range(3)
        for col in range(3)
    ]
    t = [at[i] + sum(ar[i * 3 + k] * bt[k] for k in range(3)) for i in range(3)]
    return r, t


def _seg_seg_distance(p1: _Vec, q1: _Vec, p2: _Vec, q2: _Vec) -> float:
    """Minimum distance between segments [p1,q1] and [p2,q2] (Ericson §5.1.9)."""

    def sub(a: _Vec, b: _Vec) -> _Vec:
        return [a[i] - b[i] for i in range(3)]

    def dot(a: _Vec, b: _Vec) -> float:
        return sum(a[i] * b[i] for i in range(3))

    d1, d2, r = sub(q1, p1), sub(q2, p2), sub(p1, p2)
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    eps = 1e-12
    s = t = 0.0
    if a <= eps and e <= eps:
        return math.sqrt(dot(r, r))
    if a <= eps:
        t = min(1.0, max(0.0, f / e))
    else:
        c = dot(d1, r)
        if e <= eps:
            s = min(1.0, max(0.0, -c / a))
        else:
            b = dot(d1, d2)
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if abs(denom) > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    c1 = [p1[i] + d1[i] * s for i in range(3)]
    c2 = [p2[i] + d2[i] * t for i in range(3)]
    return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))


def _neutral_pose_collisions(params: dict[str, object], threshold: float) -> list[tuple[int, int]]:
    """Pairs whose capsules already overlap at the neutral (all-zero) pose.

    Mirrors the MoveIt setup-assistant "disable always-in-collision pairs" step,
    using the kernel's own conservative capsule approximation so the resulting
    allowed-collision matrix matches what the kernel actually sees.
    """
    n = cast(int, params["collision_n_links"])
    parent = cast("list[int]", params["collision_parent"])
    origin = cast(_Vec, params["collision_origin_xyzrpy"])
    cap_link = cast("list[int]", params["collision_capsule_link"])
    cap_r = cast(_Vec, params["collision_capsule_radius"])
    cap_h = cast(_Vec, params["collision_capsule_half_length"])
    cap_o = cast(_Vec, params["collision_capsule_origin_xyzrpy"])
    # Forward kinematics at q=0 (joint motion is identity) → per-link frames.
    link_r: list[_Vec] = [[] for _ in range(n)]
    link_t: list[_Vec] = [[] for _ in range(n)]
    for i in range(n):
        o = origin[6 * i : 6 * i + 6]
        r, t = _rpy_to_mat(o[3], o[4], o[5]), list(o[:3])
        p = parent[i]
        if p >= 0:
            r, t = _compose(link_r[p], link_t[p], r, t)
        link_r[i], link_t[i] = r, t
    # Place each capsule's segment endpoints in the base frame.
    n_caps = len(cap_link)
    cap_endpoints: list[tuple[_Vec, _Vec]] = []
    for c in range(n_caps):
        li = cap_link[c]
        co = cap_o[6 * c : 6 * c + 6]
        cr, ct = _compose(link_r[li], link_t[li], _rpy_to_mat(co[3], co[4], co[5]), list(co[:3]))
        z_axis = [cr[2], cr[5], cr[8]]
        h = cap_h[c]
        cap_endpoints.append(
            ([ct[k] - z_axis[k] * h for k in range(3)], [ct[k] + z_axis[k] * h for k in range(3)])
        )
    # Any two capsules on different links overlapping at rest → disable that
    # whole link pair (dedup; the kernel's ACM is link-level).
    extra: set[tuple[int, int]] = set()
    for i in range(n_caps):
        for j in range(i + 1, n_caps):
            if cap_link[i] == cap_link[j]:
                continue
            a0, a1 = cap_endpoints[i]
            b0, b1 = cap_endpoints[j]
            dist = _seg_seg_distance(a0, a1, b0, b1) - cap_r[i] - cap_r[j]
            if dist <= threshold:
                extra.add((min(cap_link[i], cap_link[j]), max(cap_link[i], cap_link[j])))
    return sorted(extra)


def _body_joint(mj: Any, model: Any, body: int) -> tuple[int, _Vec]:
    """(joint_kind, axis) for the single hinge/slide joint on ``body``.

    Fixed/welded or free/ball bodies → (0, +Z): the kernel treats them as a
    fixed transform (a floating base is a rigid transform that can't change
    inter-link distances).

    The ``dof_index`` (the commanded-joint column this link tracks) is NOT
    decided here — it is assigned by movable-joint order in
    :func:`lower_collision_params`, because the MJCF's own joint names do not
    match the manifest's (e.g. ``Rotation`` vs ``shoulder_pan``); matching by
    name silently froze every link's FK at the rest pose (see
    ``tests/sim/safety/test_mjcf_lowering_dof_index.py``).
    """
    for ji in range(int(model.njnt)):
        if int(model.jnt_bodyid[ji]) != body:
            continue
        jtype = int(model.jnt_type[ji])
        if jtype == int(mj.mjtJoint.mjJNT_HINGE):
            kind = 1
        elif jtype == int(mj.mjtJoint.mjJNT_SLIDE):
            kind = 2
        else:
            break  # free / ball joint → fixed root
        return kind, [float(v) for v in model.jnt_axis[ji]]
    return 0, [0.0, 0.0, 1.0]


def _static_allowed_pairs(
    model: Any, parent: list[int], link_index: dict[int, int], n_bodies: int
) -> list[int]:
    """Parent↔child pairs (touch by design) + the MJCF's explicit contact excludes."""
    pairs: list[int] = []
    for body in range(1, n_bodies):
        p = parent[link_index[body]]
        if p >= 0:
            pairs.extend([p, link_index[body]])
    for ei in range(int(model.nexclude)):
        sig = int(model.exclude_signature[ei])
        b1, b2 = sig >> 16, sig & 0xFFFF
        if b1 in link_index and b2 in link_index:
            pairs.extend([link_index[b1], link_index[b2]])
    return pairs


def lower_collision_params(
    model: Any, joint_names: list[str], *, margin_m: float = 0.0
) -> dict[str, object]:
    """Lower a compiled MuJoCo model to safety-kernel collision ROS parameters.

    Args:
        model: A ``mujoco.MjModel`` (e.g. ``MujocoArmHAL._model``).
        joint_names: The robot manifest's actuated joint order — the same order
            the ``ActionChunk.flat`` joint vector uses. Each entry is matched to
            a MuJoCo hinge/slide joint by name to assign that link's ``dof_index``.
        margin_m: Clearance margin in metres (a pair closer than this fires).

    Returns:
        The ``collision_*`` ROS-parameter dict to merge with the scalar envelope
        params, with ``self_collision_enabled: True`` — **unless** the model has
        no lowerable collision geometry (every collidable geom is a mesh/plane,
        e.g. the SO-101 ``new_calib`` MJCF), in which case it returns just
        ``{"self_collision_enabled": False}`` (same disabled sentinel as
        :func:`collision_params_from_description`) so the kernel runs its scalar
        envelope check and the launch never forwards an empty-list ROS param.
    """
    import mujoco as mj

    n_bodies = int(model.nbody)
    # The manifest enumerates joints in the same order as the robot's MuJoCo
    # actuators (``python/hal/.../_mujoco_arm.py`` docstring), so the i-th movable
    # MJCF joint (in body order) maps to manifest/qpos column ``i``. ``n_cols``
    # caps that: a movable joint past the commanded vector (e.g. a robot's second,
    # mimic, gripper finger) gets ``dof_index = -1`` rather than indexing out of
    # bounds. ``next_dof`` is the running movable-joint ordinal.
    n_cols = len(joint_names)
    next_dof = 0

    # Body id → contiguous link index (skip the world body 0).
    link_index = {b: b - 1 for b in range(1, n_bodies)}

    parent: list[int] = []
    joint_kind: list[int] = []
    dof_index: list[int] = []
    origin_xyzrpy: list[float] = []
    axis: list[float] = []
    link_names: list[str] = []
    capsule_link: list[int] = []
    capsule_radius: list[float] = []
    capsule_half_length: list[float] = []
    capsule_origin_xyzrpy: list[float] = []

    for body in range(1, n_bodies):
        link_names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or f"body_{body}")
        parent_body = int(model.body_parentid[body])
        parent.append(link_index.get(parent_body, -1))

        # Fixed parent→body transform (joints rotate about the body origin).
        bpos = [float(v) for v in model.body_pos[body]]
        broll, bpitch, byaw = _quat_wxyz_to_rpy(mj, model.body_quat[body])
        origin_xyzrpy.extend([bpos[0], bpos[1], bpos[2], broll, bpitch, byaw])

        kind, jaxis = _body_joint(mj, model, body)
        joint_kind.append(kind)
        if kind == 0:
            # Fixed/welded or free/ball: no commanded column, no FK angle.
            dof_index.append(-1)
        else:
            dof_index.append(next_dof if next_dof < n_cols else -1)
            next_dof += 1
        axis.extend(jaxis)

        # Every collidable primitive on the body becomes one capsule tagged with
        # this link's index (multi-capsule per link).
        for geom in _collidable_geoms(model, body):
            radius, half_length = _capsule_from_geom(mj, model, geom)
            gpos = [float(v) for v in model.geom_pos[geom]]
            groll, gpitch, gyaw = _quat_wxyz_to_rpy(mj, model.geom_quat[geom])
            capsule_link.append(link_index[body])
            capsule_radius.append(radius)
            capsule_half_length.append(half_length)
            capsule_origin_xyzrpy.extend([gpos[0], gpos[1], gpos[2], groll, gpitch, gyaw])

    # A mesh-only MJCF (every collidable geom is a mesh/plane, which this
    # lowering doesn't approximate — see ``_first_collidable_geom``) yields zero
    # capsules. With no capsule geometry the self-collision check has nothing to
    # test, so claiming ``self_collision_enabled: True`` would be dishonest; and
    # emitting the empty ``capsule_*`` lists as ROS parameters crashes
    # ``ros2 launch`` (launch_ros normalises an empty list to ``()``, which
    # ``ensure_argument_type`` rejects with "got '()' of type tuple"). Fall back
    # to the same ``{self_collision_enabled: False}`` contract as the
    # manifest-geometry path (:func:`collision_params_from_description`): the
    # kernel runs its scalar envelope check exactly as before. Full mesh coverage
    # is a future revision (module docstring).
    if not capsule_link:
        return {"self_collision_enabled": False}

    allowed_pairs = _static_allowed_pairs(model, parent, link_index, n_bodies)

    params: dict[str, object] = {
        "self_collision_enabled": True,
        "self_collision_margin_m": float(margin_m),
        "collision_n_links": len(link_names),
        "collision_parent": parent,
        "collision_joint_kind": joint_kind,
        "collision_dof_index": dof_index,
        "collision_origin_xyzrpy": origin_xyzrpy,
        "collision_axis": axis,
        "collision_capsule_link": capsule_link,
        "collision_capsule_radius": capsule_radius,
        "collision_capsule_half_length": capsule_half_length,
        "collision_capsule_origin_xyzrpy": capsule_origin_xyzrpy,
        "collision_allowed_pairs": allowed_pairs,
        "collision_link_names": link_names,
    }
    # Disable pairs already overlapping (per the kernel's conservative capsules)
    # at the neutral pose — they carry no information and would only false-fire.
    existing = {
        (min(a, b), max(a, b)) for a, b in zip(allowed_pairs[::2], allowed_pairs[1::2], strict=True)
    }
    for a, b in _neutral_pose_collisions(params, margin_m):
        if (a, b) not in existing:
            allowed_pairs.extend([a, b])
            existing.add((a, b))
    params["collision_allowed_pairs"] = allowed_pairs
    return params
