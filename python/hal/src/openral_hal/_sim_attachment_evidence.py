"""MuJoCo ground-truth attachment evidence for deploy simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    CollisionShape,
    JointSpec,
    Pose6D,
    RobotDescription,
    SphereShape,
    SupportContactWitness,
)
from openral_core.exceptions import ROSConfigError

# Below this a summed support normal carries no direction: the probed normals
# oppose each other, so there is no support plane to attest.
_DEGENERATE_NORM = 1e-9

# -- Support-contact probe bounds --
# The producer measures support with signed geom distances, not with the
# solver's contact list. That is not a refinement, it is a correctness fix:
# MuJoCo's ``contype``/``conaffinity`` bitmasks suppress whole geom pairs, and a
# payload flush on a counter can produce ZERO contact records (2026-08-14
# acceptance: a cup resting on a RoboCasa island at 0.000 mm generated none,
# while a baguette on a counter generated six, purely because the second pair's
# bitmasks happened to meet). ``mj_geomDistance`` sees both.
#
# A payload separated by more than this is not resting on anything: the number
# is the safety kernel's own ``attached_contact_tolerance_m`` — physical slack
# for FK and pose noise, deliberately not the occupancy resolution.
_SUPPORT_PROBE_GAP_M = 0.001
# The safety kernel's ``support_witness_max_penetration_m`` and
# ``support_witness_max_patch_radius_m``. A claim past either fails the WHOLE
# attachment message closed on the kernel side, so the producer must never
# construct one; a payload deeper than the cap is a collision, not a support
# contact, and attesting a clamped depth would launder it into an exemption.
_SUPPORT_MAX_PENETRATION_M = 0.01
_SUPPORT_MAX_PATCH_RADIUS_M = 0.5
# Exact-distance call budget for one attestation. The payload contributes a
# handful of geoms and this runs once per attach, so the cost is ~0.4 ms at
# MuJoCo's ~0.8 us/call; the cap only bounds the pathological scene.
_SUPPORT_PROBE_MAX_CALLS = 1024
# Under this separation the probe's closest-point segment is degenerate and
# carries no direction — which is precisely the resting case (the field cup sat
# at 0.000 mm). The support geom's own surface normal is the primary source;
# the segment is only ever a cross-check or a last resort.
_SEGMENT_DIRECTION_MIN_M = 1e-5
# A surface normal and a closest-point segment more than 60 degrees apart do not
# describe the same plane. Rather than pick one, attest neither (fail closed).
_NORMAL_AGREEMENT_MIN = 0.5


class SimObjectMobility(str, Enum):
    """Kinematic mobility of a contacted non-robot MuJoCo body."""

    FREE = "free"
    HINGE = "hinge"
    SLIDE = "slide"
    ARTICULATED = "articulated"
    FIXED = "fixed"


@dataclass
class _ContactCandidate:
    contact_bodies: set[int] = field(default_factory=set)
    stable_ticks: int = 0
    missed_ticks: int = 0
    last_translation: NDArray[np.float64] | None = None
    last_rotation: NDArray[np.float64] | None = None


def _matrix_to_quat_xyzw(matrix: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def _relative_pose(
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    parent_body_id: int,
    child_body_id: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parent_rotation = np.asarray(data.xmat[parent_body_id], dtype=np.float64).reshape(3, 3)
    child_rotation = np.asarray(data.xmat[child_body_id], dtype=np.float64).reshape(3, 3)
    translation = parent_rotation.T @ (
        np.asarray(data.xpos[child_body_id], dtype=np.float64)
        - np.asarray(data.xpos[parent_body_id], dtype=np.float64)
    )
    return translation, parent_rotation.T @ child_rotation


def _root_motion_body(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    body_id: int,
) -> int:
    """Return the nearest ancestor owning a joint, or the fixed body itself."""
    current = int(body_id)
    while current > 0:
        if int(model.body_jntnum[current]) > 0:
            return current
        current = int(model.body_parentid[current])
    return int(body_id)


def classify_body_mobility(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    body_id: int,
) -> SimObjectMobility:
    """Classify the contacted body's root as free, hinge, slide, or fixed."""
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    root = _root_motion_body(model, body_id)
    joint_count = int(model.body_jntnum[root])
    if joint_count == 0:
        return SimObjectMobility.FIXED
    first = int(model.body_jntadr[root])
    types = {int(model.jnt_type[index]) for index in range(first, first + joint_count)}
    if int(mujoco.mjtJoint.mjJNT_FREE) in types:
        return SimObjectMobility.FREE
    if types == {int(mujoco.mjtJoint.mjJNT_HINGE)}:
        return SimObjectMobility.HINGE
    if types == {int(mujoco.mjtJoint.mjJNT_SLIDE)}:
        return SimObjectMobility.SLIDE
    return SimObjectMobility.ARTICULATED


def _body_subtree(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    root_body_id: int,
) -> set[int]:
    bodies = {int(root_body_id)}
    for body_id in range(1, int(model.nbody)):
        if int(model.body_parentid[body_id]) in bodies:
            bodies.add(body_id)
    return bodies


def _sim_joint_body_id(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    joint: JointSpec,
) -> int:
    """Resolve one manifest joint to its MuJoCo child body."""
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    if not joint.sim_joint_name:
        raise ROSConfigError(f"Joint {joint.name!r} needs sim_joint_name for contact evidence.")
    joint_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint.sim_joint_name,
        )
    )
    if joint_id < 0:
        raise ROSConfigError(f"Sim joint {joint.sim_joint_name!r} is missing.")
    return int(model.jnt_bodyid[joint_id])


def _resolve_gripper_body_groups(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    gripper_joints: list[JointSpec],
) -> dict[int, int]:
    """Map every physical finger body to its independently contacting jaw."""
    assemblies = {
        int(model.body_parentid[_sim_joint_body_id(model, joint)]) for joint in gripper_joints
    }
    groups: dict[int, int] = {}
    for assembly_body_id in assemblies:
        jaw_roots = [
            body_id
            for body_id in range(1, int(model.nbody))
            if int(model.body_parentid[body_id]) == assembly_body_id
            and any(
                int(model.body_jntnum[subtree_body_id]) > 0
                for subtree_body_id in _body_subtree(model, body_id)
            )
        ]
        for jaw_root in jaw_roots:
            for body_id in _body_subtree(model, jaw_root):
                groups[body_id] = jaw_root
    if not groups:
        raise ROSConfigError("Sim attachment evidence found no physical gripper branches.")
    return groups


def _resolve_robot_bodies(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    joints: list[JointSpec],
) -> set[int]:
    """Resolve all manifest-backed robot body subtrees present in MuJoCo."""
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    bodies: set[int] = set()
    for joint in joints:
        if not joint.sim_joint_name:
            continue
        joint_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint.sim_joint_name,
            )
        )
        if joint_id >= 0:
            bodies.update(_body_subtree(model, int(model.jnt_bodyid[joint_id])))
    return bodies


def _primitive_from_geom(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    geom_id: int,
    root_body_id: int,
    object_id: str,
) -> AttachedCollisionPrimitive:
    """Lower one live MuJoCo collision geom into a bounded analytic primitive."""
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    root_rotation = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    geom_rotation_world = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    rotation = root_rotation.T @ geom_rotation_world
    translation = root_rotation.T @ (
        np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        - np.asarray(data.xpos[root_body_id], dtype=np.float64)
    )
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        shape: CollisionShape = SphereShape(radius_m=float(size[0]))
    elif geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        shape = CapsuleShape(radius_m=float(size[0]), length_m=float(2.0 * size[1]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        shape = BoxShape(half_extents_m=tuple(float(value) for value in size[:3]))
    else:
        center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
        half_extents = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64) + 1e-4
        translation = translation + rotation @ center
        shape = BoxShape(half_extents_m=tuple(float(value) for value in half_extents))
    return AttachedCollisionPrimitive(
        shape=shape,
        pose_in_object=Pose6D(
            xyz=tuple(float(value) for value in translation),
            quat_xyzw=_matrix_to_quat_xyzw(rotation),
            frame_id=object_id,
        ),
    )


def _geom_corners_in_root(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    geom_id: int,
    root_body_id: int,
) -> NDArray[np.float64]:
    """Return one geom's conservative local-AABB corners in the root frame."""
    root_rotation = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    geom_rotation_world = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    rotation = root_rotation.T @ geom_rotation_world
    translation = root_rotation.T @ (
        np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        - np.asarray(data.xpos[root_body_id], dtype=np.float64)
    )
    local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
    half_extents = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64) + 1e-4
    signs = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float64,
    )
    local_corners = local_center + signs * half_extents
    return translation + local_corners @ rotation.T


def _clustered_box_primitives(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    geom_ids: list[int],
    root_body_id: int,
    object_id: str,
    max_primitives: int,
) -> list[AttachedCollisionPrimitive]:
    """Conservatively reduce many collision geoms into bounded local AABBs."""
    corners_by_geom = [
        _geom_corners_in_root(
            model,
            data,
            geom_id=geom_id,
            root_body_id=root_body_id,
        )
        for geom_id in geom_ids
    ]
    centers = np.asarray([corners.mean(axis=0) for corners in corners_by_geom])
    split_axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = sorted(range(len(geom_ids)), key=lambda index: float(centers[index, split_axis]))
    primitives: list[AttachedCollisionPrimitive] = []
    for cluster in np.array_split(np.asarray(ordered, dtype=np.int64), max_primitives):
        cluster_corners = np.concatenate(
            [corners_by_geom[int(index)] for index in cluster],
            axis=0,
        )
        lower = cluster_corners.min(axis=0)
        upper = cluster_corners.max(axis=0)
        center = 0.5 * (lower + upper)
        half_extents = 0.5 * (upper - lower) + 1e-4
        primitives.append(
            AttachedCollisionPrimitive(
                shape=BoxShape(
                    half_extents_m=(
                        float(half_extents[0]),
                        float(half_extents[1]),
                        float(half_extents[2]),
                    )
                ),
                pose_in_object=Pose6D(
                    xyz=(float(center[0]), float(center[1]), float(center[2])),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            )
        )
    return primitives


def extract_body_primitives(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    root_body_id: int,
    object_id: str,
    max_primitives: int = 16,
) -> list[AttachedCollisionPrimitive]:
    """Extract conservative collision primitives for one unknown sim object."""
    subtree = _body_subtree(model, root_body_id)
    geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in subtree and int(model.geom_contype[geom_id]) != 0
    ]
    if not geom_ids:
        raise ROSConfigError(f"MuJoCo body {root_body_id} has no collision geometry.")
    if len(geom_ids) > max_primitives:
        return _clustered_box_primitives(
            model,
            data,
            geom_ids=geom_ids,
            root_body_id=root_body_id,
            object_id=object_id,
            max_primitives=max_primitives,
        )
    return [
        _primitive_from_geom(
            model,
            data,
            geom_id=geom_id,
            root_body_id=root_body_id,
            object_id=object_id,
        )
        for geom_id in geom_ids
    ]


def _collision_geoms(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    bodies: set[int],
) -> list[int]:
    """The payload's collision geoms — exactly what is published as primitives."""
    return [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in bodies and int(model.geom_contype[geom_id]) != 0
    ]


def _support_candidate_geoms(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    payload_bodies: set[int],
    robot_body_ids: frozenset[int],
) -> tuple[list[int], dict[int, int]]:
    """Every geom that could legitimately be carrying this payload.

    Eligible supports are non-robot and non-free: the world, a counter, a
    cabinet. The gripper holding the payload is excluded (``touch_links``
    already covers it) and so is another free-floating object, which is not
    something the safety kernel may be told to ignore. Purely visual geometry
    (no ``contype`` *and* no ``conaffinity``) is excluded too — it is not solid,
    and a decorative shell coincident with a real surface would only add noise.

    Note that a *nonzero* ``contype``/``conaffinity`` is no guarantee the pair
    collides: the island that motivated this producer has solid bitmasks that
    simply do not meet the cup's. Suppression is a property of the pair, which
    is why membership here is decided per geom and adjudicated by distance.

    Returns:
        ``(geom_ids, support_root_of_geom)``.
    """
    geom_ids: list[int] = []
    support_root_of_geom: dict[int, int] = {}
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in payload_bodies or body_id in robot_body_ids:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        support_root = _root_motion_body(model, body_id)
        if classify_body_mobility(model, support_root) is SimObjectMobility.FREE:
            continue
        geom_ids.append(geom_id)
        support_root_of_geom[geom_id] = support_root
    return geom_ids, support_root_of_geom


def _support_surface_normal(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    geom_id: int,
    world_point: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Outward world normal of a support geom at a point on its surface.

    The closest-point segment ``mj_geomDistance`` returns is the obvious source
    for a normal, and it is the wrong one for exactly the case that matters: at
    a true resting contact the separation is ~0, the two closest points
    coincide, and the segment has no direction at all. So the primary source is
    the support's own analytic surface, which is well-defined however flush the
    payload sits on it.

    Returns:
        The unit outward normal, or ``None`` for geometry with no analytic
        surface normal (mesh, heightfield, SDF) — the caller then falls back to
        the segment, or attests nothing.
    """
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # A plane's surface normal is its frame's +z, everywhere.
        return np.asarray(rotation[:, 2], dtype=np.float64)

    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    local = rotation.T @ (world_point - np.asarray(data.geom_xpos[geom_id], dtype=np.float64))
    local_normal: NDArray[np.float64]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        # The face the point lies on is the one it is least far *inside*.
        axis = int(np.argmax(np.abs(local[:3]) - size[:3]))
        local_normal = np.zeros(3, dtype=np.float64)
        local_normal[axis] = math.copysign(1.0, float(local[axis]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local_normal = local
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        local_normal = local / np.square(size[:3])
    elif geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        radius, half_length = float(size[0]), float(size[1])
        radial = np.asarray([local[0], local[1], 0.0], dtype=np.float64)
        overshoot_axial = abs(float(local[2])) - half_length
        overshoot_radial = float(np.linalg.norm(radial)) - radius
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER) and overshoot_axial > overshoot_radial:
            local_normal = np.asarray(
                [0.0, 0.0, math.copysign(1.0, float(local[2]))], dtype=np.float64
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            # The capsule's normal points away from its spine, whose nearest
            # point is the axial coordinate clamped to the cylindrical section.
            local_normal = local - np.asarray(
                [0.0, 0.0, float(np.clip(local[2], -half_length, half_length))],
                dtype=np.float64,
            )
        else:
            local_normal = radial
    else:
        return None

    norm = float(np.linalg.norm(local_normal))
    if norm < _DEGENERATE_NORM:
        return None
    return np.asarray(rotation @ (local_normal / norm), dtype=np.float64)


@dataclass(frozen=True)
class _SupportProbeHit:
    """One payload↔support geom pair measured by signed distance."""

    support_root: int
    contact_point: NDArray[np.float64]
    normal: NDArray[np.float64]
    penetration_m: float


def _probe_support_hits(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    payload_geoms: list[int],
    candidate_geoms: list[int],
    support_root_of_geom: dict[int, int],
) -> list[_SupportProbeHit]:
    """Measure every payload↔environment pair that is within touching distance.

    Bounded exactly as the E-stop near-miss probe is: the same vectorised
    bounding-sphere prefilter and the same round-robin call budget, reused
    rather than re-derived, so a kitchen full of geometry costs a fixed number
    of exact calls and no payload geom can be starved out of the measurement.
    """
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    from openral_hal.sim_sensor_bridge import (  # noqa: PLC0415  # reason: bounded-probe reuse
        _pair_distance_lower_bound,
        _round_robin_candidates,
    )

    side = np.asarray(payload_geoms, dtype=np.int64)
    other = np.asarray(candidate_geoms, dtype=np.int64)
    gap = _pair_distance_lower_bound(model, data, side, other)
    candidates, _ = _round_robin_candidates(gap, _SUPPORT_PROBE_GAP_M, _SUPPORT_PROBE_MAX_CALLS)

    hits: list[_SupportProbeHit] = []
    segment = np.zeros(6, dtype=np.float64)
    for row, column in candidates:
        payload_geom = int(side[int(row)])
        support_geom = int(other[int(column)])
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                payload_geom,
                support_geom,
                _SUPPORT_PROBE_GAP_M,
                segment,
            )
        )
        if distance >= _SUPPORT_PROBE_GAP_M:
            continue  # not touching, and ``segment`` was left unwritten
        payload_point = np.asarray(segment[:3], dtype=np.float64).copy()
        support_point = np.asarray(segment[3:], dtype=np.float64).copy()

        surface_normal = _support_surface_normal(
            model,
            data,
            geom_id=support_geom,
            world_point=support_point,
        )
        # ``(payload_point - support_point) / distance`` is the support→payload
        # direction for both signs of ``distance``: the segment runs payload→
        # support when they overlap and support→payload when they do not, and
        # dividing by the signed distance folds that flip in. It is undefined at
        # a flush contact, which is why it is the fallback and not the source.
        segment_normal: NDArray[np.float64] | None = None
        if abs(distance) >= _SEGMENT_DIRECTION_MIN_M:
            segment_normal = (payload_point - support_point) / distance
        if surface_normal is None:
            if segment_normal is None:
                continue  # unanalysable surface, flush contact: measure nothing
            normal = segment_normal
        else:
            if segment_normal is not None and (
                float(np.dot(surface_normal, segment_normal)) < _NORMAL_AGREEMENT_MIN
            ):
                continue  # the two disagree about the plane; measure nothing
            normal = surface_normal

        hits.append(
            _SupportProbeHit(
                support_root=support_root_of_geom[support_geom],
                contact_point=0.5 * (payload_point + support_point),
                normal=normal,
                penetration_m=max(0.0, -distance),
            )
        )
    return hits


def _world_up(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
) -> NDArray[np.float64]:
    """The direction a support must push to be carrying anything."""
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    magnitude = float(np.linalg.norm(gravity))
    if magnitude < _DEGENERATE_NORM:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(-gravity / magnitude, dtype=np.float64)


def _dominant_support(
    hits: list[_SupportProbeHit],
    *,
    up: NDArray[np.float64],
    payload_com: NDArray[np.float64],
) -> tuple[int, list[_SupportProbeHit], NDArray[np.float64]] | None:
    """Pick the one surface that is actually carrying the payload.

    A payload can touch several fixed surfaces at once — a counter under it, a
    backsplash beside it, a trivet at its rim — and only one may be attested,
    because the witness names one plane.

    *Load-bearing first.* A support is what opposes gravity, so a surface whose
    mean normal has no upward component is not support at all (a wall, the
    underside of a shelf) and is discarded rather than ranked. Among those that
    do carry, the most anti-gravity normal wins: a flat seat beats a slanted
    rest, because the flat one takes the weight.

    *Then whichever is under the load.* Equally horizontal surfaces are
    separated by which one lies closest to the payload's centre of mass in the
    support plane — the seat is under the mass, a ledge merely catches the rim.
    Note what this deliberately does **not** use: the number of contacting geom
    pairs, and the spread of the probed points. A distance probe reports one
    closest point per geom pair, so both of those measure how finely a piece of
    furniture happens to be tessellated — a monolithic island slab yields a
    single point and a three-geom ledge yields three, whatever their real
    contact areas. That artefact is what the old contact-count rule was hostage
    to, and it is the reason this ranks by load path instead.

    Returns:
        ``(support_root, hits, unit mean normal)``, or ``None`` when nothing
        touching the payload is holding it up.
    """
    grouped: dict[int, list[_SupportProbeHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.support_root, []).append(hit)

    ranked: list[tuple[float, float, int, int, list[_SupportProbeHit], NDArray[np.float64]]] = []
    for support_root, group in grouped.items():
        mean_normal = np.mean(np.asarray([hit.normal for hit in group]), axis=0)
        norm = float(np.linalg.norm(mean_normal))
        if norm < _DEGENERATE_NORM:
            continue  # opposed normals: the payload is pinched, not supported
        mean_normal = mean_normal / norm
        alignment = float(np.dot(mean_normal, up))
        if alignment <= 0.0:
            continue
        offset = np.mean(np.asarray([hit.contact_point for hit in group]), axis=0) - payload_com
        lateral = float(np.linalg.norm(offset - float(np.dot(offset, mean_normal)) * mean_normal))
        # Round the alignment so two equally horizontal surfaces tie here and
        # are then separated by load path rather than by float noise.
        ranked.append(
            (-round(alignment, 6), lateral, -len(group), support_root, group, mean_normal)
        )
    if not ranked:
        return None
    ranked.sort(key=lambda entry: entry[:4])
    _, _, _, support_root, group, mean_normal = ranked[0]
    return support_root, group, mean_normal


def _plane_basis(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal in-plane axes for a unit normal."""
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    first = reference - float(np.dot(reference, normal)) * normal
    first = first / float(np.linalg.norm(first))
    return first, np.asarray(np.cross(normal, first), dtype=np.float64)


def _centred_patch(
    corners: NDArray[np.float64],
    *,
    plane_point: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Re-centre the attested point in-plane on the payload's own footprint.

    A plane is fixed by any point on it, and the probe hands back whichever
    surface point happened to be closest — an off-centre one, since a single
    geom pair yields a single point where the solver's contact set yielded a
    symmetric handful. Keeping that point would make the attested disc large
    enough to reach the payload's far corners *from the rim*, licensing a patch
    the payload does not occupy. Sliding the point along the plane to the centre
    of the payload's own in-plane footprint changes no geometry — same plane,
    same half-space — and shrinks the patch to the smallest disc that still
    covers the payload. The attestation stays bounded by the payload, tightly.

    Args:
        corners: Payload collision-AABB corners in the object frame.
        plane_point: A measured point on the support plane, object frame.
        normal: Unit support normal, object frame.

    Returns:
        ``(contact_point, patch_radius_m)`` in the object frame.
    """
    first, second = _plane_basis(normal)
    offsets = corners - plane_point
    lateral = offsets - np.outer(offsets @ normal, normal)
    along_first = lateral @ first
    along_second = lateral @ second
    centre_first = 0.5 * float(along_first.min() + along_first.max())
    centre_second = 0.5 * float(along_second.min() + along_second.max())
    radius = float(
        np.hypot(along_first - centre_first, along_second - centre_second).max(),
    )
    return plane_point + centre_first * first + centre_second * second, radius


def support_contact_witness(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    root_body_id: int,
    robot_body_ids: frozenset[int],
    stamp_ns: int,
) -> SupportContactWitness | None:
    """Attest one payload's bounded support contact from exact MuJoCo distances.

    A grasped object is routinely still resting on the counter it was picked
    from. That contact is real, legitimate, and — once the payload is checked
    as robot geometry — indistinguishable to the safety kernel from driving the
    payload through a wall. This produces the attestation that tells the two
    apart, from ground truth the simulator already has (ADR-0092 D6).

    Support is measured with ``mj_geomDistance``, **not** with the solver's
    contact list. The contact list is not a proximity oracle: ``contype`` /
    ``conaffinity`` suppression empties whole geom pairs, so on the 2026-08-14
    acceptance run a cup resting on an island at 0.000 mm produced no contact
    record at all — no attestation, no exemption, and an E-stop on the real
    support contact — while a baguette on a counter produced six, purely
    because that pair's bitmasks happened to meet. Signed distance sees both.

    Only a *non-free* environment body counts as a support: the world, a
    counter, a cabinet. Another free-floating object is not something the
    kernel may be told to ignore, and neither is the gripper holding the
    payload (that is what ``touch_links`` covers). ``None`` — nothing within
    touching distance, nothing load-bearing, or a claim the kernel's caps would
    not accept — is the honest answer and yields no exemption.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` after a step.
        root_body_id: Root body of the payload.
        robot_body_ids: Every body belonging to the robot.
        stamp_ns: Producer timestamp for the witness.

    Returns:
        The witness, or ``None`` when no eligible support contact exists.
    """
    payload_bodies = _body_subtree(model, root_body_id)
    payload_geoms = _collision_geoms(model, payload_bodies)
    if not payload_geoms:
        return None
    candidate_geoms, support_root_of_geom = _support_candidate_geoms(
        model,
        payload_bodies=payload_bodies,
        robot_body_ids=robot_body_ids,
    )
    if not candidate_geoms:
        return None

    dominant = _dominant_support(
        _probe_support_hits(
            model,
            data,
            payload_geoms=payload_geoms,
            candidate_geoms=candidate_geoms,
            support_root_of_geom=support_root_of_geom,
        ),
        up=_world_up(model),
        payload_com=np.asarray(data.xipos[root_body_id], dtype=np.float64),
    )
    if dominant is None:
        return None
    support_root, group, normal_world = dominant

    penetration = max(hit.penetration_m for hit in group)
    if penetration > _SUPPORT_MAX_PENETRATION_M:
        # Deeper than the safety kernel will accept as support. Clamping would
        # launder a real penetration into an exemption, so attest nothing and
        # let the kernel stop on it.
        return None

    rotation = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.xpos[root_body_id], dtype=np.float64)
    normal_in_object = rotation.T @ normal_world
    normal_in_object = normal_in_object / float(np.linalg.norm(normal_in_object))
    plane_point = np.mean(
        np.asarray([rotation.T @ (hit.contact_point - origin) for hit in group]),
        axis=0,
    )

    # The patch spans the payload's own supported footprint: its collision
    # geometry projected onto the support plane. Bounded by the payload, and
    # nothing outside it is ever exempt.
    corners = np.concatenate(
        [
            _geom_corners_in_root(model, data, geom_id=geom_id, root_body_id=root_body_id)
            for geom_id in payload_geoms
        ],
        axis=0,
    )
    contact_point, patch_radius = _centred_patch(
        corners,
        plane_point=plane_point,
        normal=normal_in_object,
    )
    if not math.isfinite(patch_radius) or not 0.0 < patch_radius <= _SUPPORT_MAX_PATCH_RADIUS_M:
        # Degenerate, or a payload too large to describe inside the kernel's
        # bound. Trimming the radius would attest a patch the payload does not
        # occupy, so attest nothing.
        return None

    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    support_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, support_root)
    if not support_name:
        raise ROSConfigError(f"Supporting MuJoCo body {support_root} has no name.")
    return SupportContactWitness(
        support_id=f"sim:{support_name}",
        contact_point_in_object=(
            float(contact_point[0]),
            float(contact_point[1]),
            float(contact_point[2]),
        ),
        contact_normal_in_object=(
            float(normal_in_object[0]),
            float(normal_in_object[1]),
            float(normal_in_object[2]),
        ),
        patch_radius_m=patch_radius,
        max_penetration_m=penetration,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_GEOM_DISTANCE,
        evidence_ref=f"mujoco_geom_distance:{support_name}",
        stamp_ns=stamp_ns,
    )


class SimAttachmentEvidenceTracker:
    """Confirm free-object grasp/release from exact MuJoCo contacts and motion."""

    def __init__(
        self,
        model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        description: RobotDescription,
        *,
        stable_ticks: int = 3,
        release_ticks: int = 3,
        translation_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.15,
    ) -> None:
        self._model = model
        self._stable_ticks = stable_ticks
        self._release_ticks = release_ticks
        self._translation_tolerance_m = translation_tolerance_m
        self._rotation_tolerance_rad = rotation_tolerance_rad
        self._candidates: dict[int, _ContactCandidate] = {}
        self._attached_root: int | None = None
        self._attached_translation: NDArray[np.float64] | None = None
        self._attached_rotation: NDArray[np.float64] | None = None

        gripper_joints = [joint for joint in description.joints if joint.role == "gripper"]
        if not gripper_joints:
            raise ROSConfigError("Sim attachment evidence requires gripper-role joints.")
        attach_links = {joint.parent_link for joint in gripper_joints}
        if len(attach_links) != 1:
            raise ROSConfigError(
                f"Sim attachment evidence needs one gripper parent link, got {attach_links}."
            )
        attach_link = next(iter(attach_links))
        child_to_joint = {joint.child_link: joint for joint in description.joints}
        attach_joint = child_to_joint.get(attach_link)
        if attach_joint is None or not attach_joint.sim_joint_name:
            raise ROSConfigError(f"Cannot resolve sim body for attach link {attach_link!r}.")
        self._attach_link = attach_link
        self._attach_body_id = _sim_joint_body_id(model, attach_joint)
        self._touch_links = [joint.child_link for joint in gripper_joints]

        self._gripper_body_groups = _resolve_gripper_body_groups(model, gripper_joints)

        end_effector = next(
            (spec for spec in description.end_effectors if spec.actuated),
            None,
        )
        if end_effector is None:
            raise ROSConfigError("Sim attachment evidence requires an actuated end effector.")
        self._required_contact_groups = (
            2 if end_effector.kind in {"parallel_gripper", "dexterous_hand"} else 1
        )
        if len(set(self._gripper_body_groups.values())) < self._required_contact_groups:
            raise ROSConfigError(
                f"Sim {end_effector.kind} needs {self._required_contact_groups} physical "
                f"contact branches, found {len(set(self._gripper_body_groups.values()))}."
            )

        robot_bodies = _resolve_robot_bodies(model, description.joints)
        robot_bodies.update(self._gripper_body_groups)
        self._robot_body_ids = frozenset(robot_bodies)

    def _contacts(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    ) -> dict[int, set[int]]:
        contacts: dict[int, set[int]] = {}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            body_a = int(self._model.geom_bodyid[int(contact.geom1)])
            body_b = int(self._model.geom_bodyid[int(contact.geom2)])
            if body_a in self._gripper_body_groups and body_b not in self._robot_body_ids:
                root = _root_motion_body(self._model, body_b)
                contacts.setdefault(root, set()).add(self._gripper_body_groups[body_a])
            elif body_b in self._gripper_body_groups and body_a not in self._robot_body_ids:
                root = _root_motion_body(self._model, body_a)
                contacts.setdefault(root, set()).add(self._gripper_body_groups[body_b])
        return contacts

    def update(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        stamp_ns: int,
    ) -> list[AttachedCollisionObject] | None:
        """Return a new complete attachment set only when attach/detach changes."""
        contacts = self._contacts(data)
        if self._attached_root is not None:
            candidate = self._candidates.setdefault(self._attached_root, _ContactCandidate())
            if self._attached_root in contacts:
                candidate.missed_ticks = 0
                self._attached_translation, self._attached_rotation = _relative_pose(
                    data,
                    parent_body_id=self._attach_body_id,
                    child_body_id=self._attached_root,
                )
                return None
            translation, rotation = _relative_pose(
                data,
                parent_body_id=self._attach_body_id,
                child_body_id=self._attached_root,
            )
            if self._attached_translation is not None and self._attached_rotation is not None:
                translation_error = float(np.linalg.norm(translation - self._attached_translation))
                relative_rotation = self._attached_rotation.T @ rotation
                angle_error = math.acos(
                    float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
                )
                if (
                    translation_error <= self._translation_tolerance_m
                    and angle_error <= self._rotation_tolerance_rad
                ):
                    candidate.missed_ticks = 0
                    return None
            candidate.missed_ticks += 1
            if candidate.missed_ticks < self._release_ticks:
                return None
            self._attached_root = None
            self._attached_translation = None
            self._attached_rotation = None
            self._candidates.clear()
            return []

        for root, contact_bodies in contacts.items():
            if classify_body_mobility(self._model, root) is not SimObjectMobility.FREE:
                continue
            candidate = self._candidates.setdefault(root, _ContactCandidate())
            candidate.contact_bodies = contact_bodies
            if len(contact_bodies) < self._required_contact_groups:
                candidate.stable_ticks = 0
                continue
            translation, rotation = _relative_pose(
                data,
                parent_body_id=self._attach_body_id,
                child_body_id=root,
            )
            if candidate.last_translation is None or candidate.last_rotation is None:
                candidate.stable_ticks = 1
            else:
                translation_error = float(np.linalg.norm(translation - candidate.last_translation))
                relative_rotation = candidate.last_rotation.T @ rotation
                angle_error = math.acos(
                    float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
                )
                candidate.stable_ticks = (
                    candidate.stable_ticks + 1
                    if translation_error <= self._translation_tolerance_m
                    and angle_error <= self._rotation_tolerance_rad
                    else 1
                )
            candidate.last_translation = translation
            candidate.last_rotation = rotation
            if candidate.stable_ticks < self._stable_ticks:
                continue

            import mujoco  # noqa: PLC0415  # reason: optional sim dependency

            body_name = mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root,
            )
            if not body_name:
                raise ROSConfigError(f"Attached MuJoCo body {root} has no name.")
            object_id = f"sim:{body_name}"
            object_bodies = _body_subtree(self._model, root)
            attachment = AttachedCollisionObject(
                object_id=object_id,
                attach_link=self._attach_link,
                touch_links=self._touch_links,
                primitives=extract_body_primitives(
                    self._model,
                    data,
                    root_body_id=root,
                    object_id=object_id,
                ),
                pose_in_link=Pose6D(
                    xyz=tuple(float(value) for value in translation),
                    quat_xyzw=_matrix_to_quat_xyzw(rotation),
                    frame_id=self._attach_link,
                ),
                mass_kg=sum(float(self._model.body_mass[body_id]) for body_id in object_bodies),
                confidence=1.0,
                evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
                evidence_ref=f"mujoco_body:{body_name}",
                stamp_ns=stamp_ns,
                # Attested once, at attach, from the contacts that exist right
                # now. The safety kernel latches it and kills it on separation;
                # re-attesting mid-carry would defeat that hysteresis.
                support_contact=support_contact_witness(
                    self._model,
                    data,
                    root_body_id=root,
                    robot_body_ids=self._robot_body_ids,
                    stamp_ns=stamp_ns,
                ),
            )
            self._attached_root = root
            self._attached_translation = translation.copy()
            self._attached_rotation = rotation.copy()
            return [attachment]
        return None
