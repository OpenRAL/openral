"""MuJoCo ground-truth attachment evidence for deploy simulation."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    CollisionShape,
    ContactForceWitness,
    JointSpec,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    SphereShape,
    SupportContactWitness,
)
from openral_core.exceptions import ROSConfigError

_LOGGER = structlog.get_logger(__name__)

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
# bitmasks happened to meet). A signed distance sees both.
#
# The distance is ``openral_hal.convex_distance.convex_geom_distance``, NOT
# ``mujoco.mj_geomDistance``. #170 measured that call returning confidently
# wrong values on RoboCasa-fixture-vs-``panda_mobile``-mesh pairs in two silent
# modes, always toward *closer* — which on this path would attest a contact
# that is not there, and a witness earns a kernel exemption. Only a certified
# measurement produces a hit; an uncertified pair (a plane, an over-budget
# hull) attests nothing, so the failure direction is a missing exemption, never
# a false one (#190).
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
# Exact-distance call budget for one attestation. This runs at attach, and
# then on EVERY tick of a live place declaration until one attests (the
# hysteresis in ``_place_witness`` short-circuits only after a success) — so
# the per-tick worst case, not the per-attach cost, is what this cap bounds. Measured on the shipped
# ``robocasa_baguette`` kitchen (2383 geoms, 16 payload geoms, 1358 support
# candidates): 353 pairs pass the 1 mm bounding-sphere prefilter, the certified
# window rejection discards 234 of them unsolved, 119 are solved, and the whole
# attestation costs ~0.21 s — ~1.7 ms per solved mesh pair against
# ``mj_geomDistance``'s ~0.8 us, the price of an answer that is proved. The
# place phase measures against the DECLARED TARGET's bodies only, which is what
# keeps a per-tick re-probe affordable.
# ponytail: hulls are re-decoded per call (~0.1 s of the 0.46 s); cache them
# per (model, mesh) if attach latency ever matters.
_SUPPORT_PROBE_MAX_CALLS = 1024
# ADR-0100 contact-force gate. MuJoCo reports a contact's force in the contact's
# own frame via ``mj_contactForce``; this scalar maps that magnitude to the
# number a place declaration's ``contact_force_threshold_n`` is compared against.
#
# IT IS A CALIBRATION KNOB, NOT AN EQUIVALENCE CLAIM (CLAUDE.md 1.2). No
# published work validates MuJoCo contact-force MAGNITUDES against real
# force-torque measurements (survey 21.7): MuJoCo documents its contact model as
# an approximation whose physical validity rests on ``solref`` / ``solimp``
# choices, FORGE (arXiv:2408.04587) re-tunes its threshold on hardware across
# >1000 real trials, and arXiv:2602.14174 argues from that same gap that only
# force DIRECTION survives sim-to-real. So the producer publishes
# ``magnitude_calibrated=False`` unless an operator has explicitly asserted a
# calibration, and the kernel then refuses to read the magnitude at all.
#
# 1.0 is the identity mapping, and deliberately not a claim that one MuJoCo
# force unit is one newton. Set both env vars together to arm the gate.
_CONTACT_FORCE_SCALE_ENV = "OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT"
_CONTACT_FORCE_CALIBRATION_REF_ENV = "OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF"
_CONTACT_FORCE_DEFAULT_SCALE = 1.0
# A surface normal and the instrument's own contact direction more than 60
# degrees apart do not describe the same plane. Rather than pick one, attest
# neither (fail closed). This runs at EVERY hit, flush ones included, because
# ``convex_geom_distance`` reports its direction directly rather than leaving
# it to be recovered from two witness points that coincide at a resting
# contact — the case that matters (the field cup sat at 0.000 mm).
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


def _quat_xyzw_to_matrix(quat_xyzw: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Unit quaternion ``(x, y, z, w)`` to a 3x3 rotation matrix.

    The inverse of :func:`_matrix_to_quat_xyzw`, and needed for the same reason
    that function is: the declared place target's primitives (ADR-0098) are
    measured once in the target body's own frame and re-posed into the robot
    base frame on every publication, which is a rotation composition and not a
    quaternion the producer can carry through unchanged.
    """
    qx, qy, qz, qw = (float(value) for value in quat_xyzw)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < _DEGENERATE_NORM:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
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


def subtree_region_box(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    root_body_id: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Bound one body's whole collision subtree by a box in that body's frame.

    The sim producer for a declared place target's region (ADR-0097's
    2026-08-14 amendment). A cabinet's declared identity covers the shelf, the
    walls and the door inside it, so the region is measured over the same
    subtree the place witness is allowed to attest against — computed once at
    declaration time from the model, never inferred from where the payload
    happens to be.

    The box is axis-aligned **in the declared body's own frame**, which makes it
    an oriented box once posed in the robot base frame. That is deliberately
    tighter than an axis-aligned world hull would be: a rotated cabinet's world
    AABB is strictly larger, i.e. strictly more permissive.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData`` (geom world poses are read from it).
        root_body_id: The declared target body.

    Returns:
        ``(centre_in_body_frame, half_extents)``, or ``None`` when the subtree
        has no collision geometry or the hull is degenerate — both of which mean
        no region, and therefore no allowance.
    """
    geom_ids = _collision_geoms(model, _body_subtree(model, root_body_id))
    if not geom_ids:
        return None
    corners = np.concatenate(
        [
            _geom_corners_in_root(model, data, geom_id=geom_id, root_body_id=root_body_id)
            for geom_id in geom_ids
        ]
    )
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    half_extents = 0.5 * (upper - lower)
    if not np.all(np.isfinite(half_extents)) or float(half_extents.min()) <= 0.0:
        return None
    return 0.5 * (lower + upper), half_extents


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
    support_roots: frozenset[int] | None = None,
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

    Args:
        model: Live ``mujoco.MjModel``.
        payload_bodies: Bodies belonging to the payload itself.
        robot_body_ids: Bodies belonging to the robot.
        support_roots: When given, the ONLY support roots eligible at all — the
            place-phase declaration's declared target and the bodies that are
            part of it (ADR-0097). ``None`` is the pick-phase case: any
            eligible environment surface may be attested.

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
        if support_roots is not None and support_root not in support_roots:
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

    The closest-point segment the distance instrument returns is the obvious
    source for a normal, and it is the wrong one for exactly the case that matters: at
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
    """One payload↔support geom pair measured by certified signed distance."""

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

    Measured with the same certified instrument as the evidence path,
    ``convex_geom_distance``: a pair the separating-axis bound proves to be
    beyond the window is rejected without being solved, and a pair whose
    distance could not be certified is skipped rather than guessed at — this
    probe's output licenses an exemption, so only a proved contact may count.
    """
    from openral_hal.convex_distance import (  # noqa: PLC0415  # reason: optional sim dependency
        convex_geom_distance,
    )
    from openral_hal.sim_sensor_bridge import (  # noqa: PLC0415  # reason: bounded-probe reuse
        _pair_distance_lower_bound,
        _round_robin_candidates,
    )

    side = np.asarray(payload_geoms, dtype=np.int64)
    other = np.asarray(candidate_geoms, dtype=np.int64)
    gap = _pair_distance_lower_bound(model, data, side, other)
    candidates, _ = _round_robin_candidates(gap, _SUPPORT_PROBE_GAP_M, _SUPPORT_PROBE_MAX_CALLS)

    hits: list[_SupportProbeHit] = []
    for row, column in candidates:
        payload_geom = int(side[int(row)])
        support_geom = int(other[int(column)])
        measured = convex_geom_distance(
            model, data, payload_geom, support_geom, distmax_m=_SUPPORT_PROBE_GAP_M
        )
        distance = measured.distance_m
        if measured.method == "beyond-window" or distance >= _SUPPORT_PROBE_GAP_M:
            continue  # provably not touching
        if not measured.certified or measured.witness_a is None or measured.witness_b is None:
            continue  # not measured: an unproved contact licenses nothing
        payload_point = np.asarray(measured.witness_a, dtype=np.float64)
        support_point = np.asarray(measured.witness_b, dtype=np.float64)
        if measured.direction is None:
            continue  # no contact direction: nothing to check a surface normal against
        probe_normal = np.asarray(measured.direction, dtype=np.float64)

        surface_normal = _support_surface_normal(
            model,
            data,
            geom_id=support_geom,
            world_point=support_point,
        )
        # The instrument reports the support→payload direction itself, on both
        # of its branches, and — unlike differencing the two witness points —
        # it is still defined at a flush contact, which is precisely the case
        # this module exists for. So the cross-check runs ALWAYS, and never
        # degrades to trusting one source alone.
        #
        # That matters because the SAT witness on ``b`` lies in the support's
        # supporting *plane*, not necessarily within its face: on a tessellated
        # counter a payload flush on one strip yields, for a neighbouring
        # strip, a support point metres outside it, where the box face normal
        # comes back LATERAL. Averaged into the group by ``_dominant_support``
        # that tilted the attested plane by up to 45 degrees — a wrong support
        # plane handed to the kernel as an exemption. Checked against the
        # instrument's own direction, those hits are dropped (#190).
        if surface_normal is None:
            continue  # unanalysable surface (mesh, heightfield, SDF): measure nothing
        if float(np.dot(surface_normal, probe_normal)) < _NORMAL_AGREEMENT_MIN:
            continue  # the surface and the measurement disagree; measure nothing
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
    support_roots: frozenset[int] | None = None,
) -> SupportContactWitness | None:
    """Attest one payload's bounded support contact from certified signed distances.

    A grasped object is routinely still resting on the counter it was picked
    from. That contact is real, legitimate, and — once the payload is checked
    as robot geometry — indistinguishable to the safety kernel from driving the
    payload through a wall. This produces the attestation that tells the two
    apart, from ground truth the simulator already has (ADR-0092 D6).

    Support is measured with ``openral_hal.convex_distance`` (the certified
    instrument #170 put on the evidence path — never ``mj_geomDistance``, see
    the module header), **not** with the solver's contact list. The contact
    list is not a proximity oracle: ``contype`` /
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
        support_roots: Restrict eligible supports to these roots. ``None`` (the
            pick-phase default) admits any eligible environment surface; the
            place phase passes the declared target's own body subtree, so the
            only contact that can ever be attested under a place declaration is
            contact **on the declared target** (ADR-0097). Contact with
            anything else measures the same and attests nothing.

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
        support_roots=support_roots,
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
        evidence_ref=f"openral_hal.convex_distance.convex_geom_distance:{support_name}",
        stamp_ns=stamp_ns,
    )


def contact_force_calibration() -> tuple[float, bool, str | None]:
    """Resolve the sim contact-force calibration from the environment.

    Returns:
        ``(scale, calibrated, reference)``. ``calibrated`` is ``True`` only when
        an operator has set **both** :data:`_CONTACT_FORCE_SCALE_ENV` to a
        finite positive scale and :data:`_CONTACT_FORCE_CALIBRATION_REF_ENV` to
        a non-empty name for it. Anything else — neither set, one set, an
        unparseable or non-positive scale — yields the identity scale,
        ``calibrated=False`` and no reference, which leaves the ADR-0100 force
        gate disarmed and geometry deciding alone.

        Requiring the *reference* as well as the number is the point. A scale
        with no name is a magnitude nobody can audit, and survey §21.7 is
        explicit that a sim magnitude is a calibration knob rather than a claim
        about newtons. An unnamed knob would be exactly the silent arming this
        design refuses.

    Example:
        >>> scale, calibrated, ref = contact_force_calibration()
        >>> (scale > 0.0, isinstance(calibrated, bool))
        (True, True)
    """
    raw_scale = os.environ.get(_CONTACT_FORCE_SCALE_ENV, "").strip()
    reference = os.environ.get(_CONTACT_FORCE_CALIBRATION_REF_ENV, "").strip()
    if not raw_scale or not reference:
        return _CONTACT_FORCE_DEFAULT_SCALE, False, None
    try:
        scale = float(raw_scale)
    except ValueError:
        _LOGGER.warning(
            "sim.contact_force_calibration_ignored",
            reason="unparseable_scale",
            value=raw_scale,
        )
        return _CONTACT_FORCE_DEFAULT_SCALE, False, None
    if not math.isfinite(scale) or scale <= 0.0:
        _LOGGER.warning(
            "sim.contact_force_calibration_ignored", reason="non_positive_scale", value=scale
        )
        return _CONTACT_FORCE_DEFAULT_SCALE, False, None
    return scale, True, reference


def probe_contact_force(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    payload_geoms: Sequence[int],
    target_root_body: int,
    target_name: str,
    stamp_ns: int,
) -> ContactForceWitness | None:
    """Attest the measured contact force between a payload and its place target.

    Walks MuJoCo's solver contact list for pairs with one geom on the payload
    and the other on the declared target's subtree, and reports the **total**
    normal load over them — a box resting on a shelf makes four corner contacts
    each carrying a quarter of its weight, so any single one understates the
    press by the size of the solver's contact manifold. The direction is the
    dominant contact's normal, expressed in the payload's own frame.

    **Absence of a return value is not evidence of absent contact.** This reads
    the solver's contact list, and ``contype`` / ``conaffinity`` exclusions can
    suppress a pair entirely — field-observed at 30 mm of interpenetration with
    ``ncon == 0``. A ``None`` here only ever means the ADR-0100 gate does not
    arm, and geometry decides exactly as it does today. That is why this
    producer feeds a check which can only *add* a refusal: a blind spot in it
    can never remove one.

    The magnitude is Newtons only under an explicit operator calibration
    (:func:`contact_force_calibration`); otherwise the witness carries
    ``magnitude_calibrated=False`` and the kernel does not read it.

    Args:
        model: MuJoCo ``MjModel``.
        data: MuJoCo ``MjData`` at the configuration to attest.
        payload_geoms: Geom ids belonging to the carried payload.
        target_root_body: Root body id of the declared place target.
        target_name: The target's ``sim:`` identity, without the prefix.
        stamp_ns: Producer timestamp.

    Returns:
        A witness, or ``None`` when no payload-to-target contact is in the
        solver's list at this configuration.
    """
    import mujoco  # noqa: PLC0415  # reason: optional sim dependency

    payload = frozenset(int(g) for g in payload_geoms)
    if not payload:
        return None
    target_bodies = frozenset(_body_subtree(model, target_root_body))
    scale, calibrated, reference = contact_force_calibration()

    # SUM of the normal components over every payload-to-target contact, not the
    # largest single one. A box resting on a shelf makes four corner contacts
    # each carrying a quarter of the load, so the maximum understates the press
    # by the contact count — a number that depends on the solver's manifold
    # rather than on how hard the payload is being pushed. The sum is the total
    # normal load, it is what "pressing too hard on the receptacle" means, and it
    # is the more conservative of the two (sum >= max), which is the direction
    # this whole gate is required to move in.
    total_normal = 0.0
    contacts_found = 0
    best_magnitude = -1.0
    best_normal_world: NDArray[np.float64] | None = None
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        a_is_payload = geom_a in payload
        b_is_payload = geom_b in payload
        if a_is_payload == b_is_payload:
            continue
        other_geom = geom_b if a_is_payload else geom_a
        if int(model.geom_bodyid[other_geom]) not in target_bodies:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        # The normal component is the first entry of the contact-frame wrench;
        # the tangential pair that follows is friction, which is not what a
        # power-and-force-limiting bound is about. Magnitude of the normal
        # component only, so a hard press registers and a shear graze does not.
        magnitude = abs(float(wrench[0]))
        if not math.isfinite(magnitude):
            continue
        total_normal += magnitude
        contacts_found += 1
        # Direction comes from the DOMINANT contact — the one carrying the most
        # normal load — rather than from an average, which on a four-corner
        # manifold would be the same vector anyway and on an edge contact would
        # be a direction no contact actually has.
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            # MuJoCo's contact frame is row-major with the normal first, pointing
            # from geom1 toward geom2. The message declares a direction pointing
            # INTO the payload, so the normal is taken as-is when the payload is
            # geom2 and negated when it is geom1. Verified against a settled
            # box-on-shelf rest in test_sim_contact_force_witness.py rather than
            # taken from the docs: a sign error here would attest a direction
            # opposite the physical press.
            normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0]
            best_normal_world = -normal if a_is_payload else normal
    if contacts_found == 0 or best_normal_world is None:
        return None

    # Into the payload's own frame, for the same reason the support witness's
    # geometry is there: a base-frame direction decorrelates as the mobile base
    # drives while the physical contact persists.
    payload_body = int(model.geom_bodyid[next(iter(sorted(payload)))])
    rotation = np.asarray(data.xmat[payload_body], dtype=np.float64).reshape(3, 3)
    direction = rotation.T @ best_normal_world
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    direction = direction / norm

    return ContactForceWitness(
        target_id=f"sim:{target_name}",
        direction_in_object=(float(direction[0]), float(direction[1]), float(direction[2])),
        magnitude_n=total_normal * scale,
        magnitude_calibrated=calibrated,
        calibration_ref=reference,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT_FORCE,
        evidence_ref=f"mujoco.mj_contactForce:{target_name}",
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
        # -- Place-phase declaration state (ADR-0097) --
        # The declaration itself, the bodies its target resolves to, and the
        # stamp of the declaration a place witness has already been attested
        # for. That last one is the hysteresis: one declaration licenses ONE
        # attestation, so a payload that separates from its place surface and
        # touches it again is a fresh violation, never silently re-forgiven.
        self._place_declaration: PlaceDeclaration | None = None
        self._place_target_bodies: frozenset[int] = frozenset()
        self._place_attested_stamp_ns: int | None = None
        # The declared target's body and its model-measured region box, in that
        # body's own frame (ADR-0097's 2026-08-14 amendment). Measured once at
        # declaration time; posed into the robot base frame on every
        # publication, because the base frame is the frame the occupancy grid —
        # and therefore the allowance — lives in.
        self._place_target_body_id: int | None = None
        self._place_target_body_name: str = ""
        self._place_region_local: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._place_geometry_local: tuple[AttachedCollisionPrimitive, ...] | None = None

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

        # The body the robot's ``base_frame`` TF denotes (ADR-0095), which is the
        # frame the octomap bridge publishes the occupancy grid in and therefore
        # the only frame a place region may be expressed in. Unresolvable → no
        # region is ever produced, which is the fail-closed direction.
        from openral_hal.depth_cloud import (  # noqa: PLC0415  # reason: optional sim dependency
            resolve_base_frame_body_name,
        )

        self._base_frame_id = str(getattr(description, "base_frame", "") or "base_link")
        base_body_name = resolve_base_frame_body_name(model, description=description)
        self._base_body_id: int | None = None
        if base_body_name:
            import mujoco  # noqa: PLC0415  # reason: optional sim dependency

            resolved = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name))
            self._base_body_id = resolved if resolved >= 0 else None

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
        """Return a new complete attachment set only when the attachment changes.

        "Changes" is attach, detach, and — while a place-phase declaration is
        active (ADR-0097) — the arming or disarming of the payload's place
        witness. Nothing else re-publishes; a heartbeat of the same attachment
        would re-arm an exemption the kernel had deliberately killed.
        """
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
                return self._place_phase_transition(
                    data,
                    root=self._attached_root,
                    stamp_ns=stamp_ns,
                )
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
                    # Still carried — the jaws lost their solver contact
                    # records but the payload is rigidly following the
                    # gripper. The place phase is live here too.
                    return self._place_phase_transition(
                        data,
                        root=self._attached_root,
                        stamp_ns=stamp_ns,
                    )
            candidate.missed_ticks += 1
            if candidate.missed_ticks < self._release_ticks:
                return None
            self._attached_root = None
            self._attached_translation = None
            self._attached_rotation = None
            self._candidates.clear()
            # Release ends the place phase for this payload: a subsequent grasp
            # starts a new one, and the declaration alone never carries over.
            self._place_attested_stamp_ns = None
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
            attachment = self._build_attachment(
                data,
                root=root,
                translation=translation,
                rotation=rotation,
                stamp_ns=stamp_ns,
                # Attested once, at attach, from the contacts that exist right
                # now. The safety kernel latches it and kills it on separation;
                # re-attesting mid-carry would defeat that hysteresis.
                witness=support_contact_witness(
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
            # A newly grasped payload is a new place-phase subject: whatever a
            # still-live declaration already licensed for the previous payload
            # does not carry over.
            self._place_attested_stamp_ns = None
            return [attachment]
        return None

    def _build_attachment(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        root: int,
        translation: NDArray[np.float64],
        rotation: NDArray[np.float64],
        stamp_ns: int,
        witness: SupportContactWitness | None,
    ) -> AttachedCollisionObject:
        """Lower one live MuJoCo payload into a complete attachment record."""
        import mujoco  # noqa: PLC0415  # reason: optional sim dependency

        body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, root)
        if not body_name:
            raise ROSConfigError(f"Attached MuJoCo body {root} has no name.")
        object_id = f"sim:{body_name}"
        object_bodies = _body_subtree(self._model, root)
        return AttachedCollisionObject(
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
                xyz=(float(translation[0]), float(translation[1]), float(translation[2])),
                quat_xyzw=_matrix_to_quat_xyzw(rotation),
                frame_id=self._attach_link,
            ),
            mass_kg=sum(float(self._model.body_mass[body_id]) for body_id in object_bodies),
            confidence=1.0,
            evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
            evidence_ref=f"mujoco_body:{body_name}",
            stamp_ns=stamp_ns,
            support_contact=witness,
        )

    # -- Place-phase declaration (ADR-0097) ----------------------------------

    def set_place_declaration(self, declaration: PlaceDeclaration | None) -> None:
        """Install, replace, or retract the active place-phase declaration.

        The declaration is dispatch's typed statement that this goal is placing
        its payload into a named target. It licenses nothing on its own: it
        only makes contact **on that target** attestable, and only once.

        Resolution is done here, at install time, rather than per tick, so an
        unresolvable target is an explicit, logged refusal at the dispatch
        boundary instead of a silent per-tick no-op.

        Args:
            declaration: The declaration to install. ``None``, or one that is
                already retracted or expired, clears the tracker's place state.

        Raises:
            ROSConfigError: The declared ``target_id`` does not name a body in
                this simulation. The declaration is refused (no place witness
                can arm), which is the fail-closed direction.
        """
        import mujoco  # noqa: PLC0415  # reason: optional sim dependency

        if declaration is None or not declaration.active:
            self._clear_place_declaration()
            return
        target_id = declaration.target_id
        body_name = target_id.removeprefix("sim:")
        body_id = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id < 0:
            self._clear_place_declaration()
            raise ROSConfigError(
                f"Place declaration target {target_id!r} names no MuJoCo body; refused."
            )
        # The declared target *and every body that is part of it*: a cabinet's
        # declared identity covers the shelf inside it, which is the surface
        # the payload actually comes to rest on. Nothing outside this subtree
        # can be attested while the declaration is live.
        self._place_declaration = declaration
        self._place_target_bodies = frozenset(_body_subtree(self._model, body_id))
        # The same subtree, measured (ADR-0097's 2026-08-14 amendment). The box
        # is computed here, once, from the model — never from the payload's
        # position and never per tick — so an unmeasurable target is a decided,
        # logged "no region" rather than a silently varying one. Only its POSE is
        # refreshed later, because the base frame moves and the region does not.
        self._place_target_body_id = body_id
        self._place_target_body_name = body_name
        self._place_region_local = None
        self._place_geometry_local = None

    def _clear_place_declaration(self) -> None:
        """Drop the declaration and everything scoped to it.

        One method so retraction, refusal and expiry can never diverge: the
        region is part of the declaration's lifetime (HZ-0097-4 mitigation 4), so
        anything that kills the declaration must kill the allowance with it.
        """
        self._place_declaration = None
        self._place_target_bodies = frozenset()
        self._place_target_body_id = None
        self._place_target_body_name = ""
        self._place_region_local = None
        self._place_geometry_local = None

    def place_declaration(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        stamp_ns: int,
    ) -> PlaceDeclaration | None:
        """The live declaration, with its region posed in the robot base frame.

        This is the producer half of the amendment's Condition 2: sim measures
        the declared body's model subtree, and the safety kernel consumes the
        resulting box without knowing or caring which producer measured it. Real
        hardware will fill the same field from the perception stack through a
        seam that does not exist yet, which is why no allowance is applied on
        real hardware today.

        Every path that yields no region — a dead declaration, an unresolved
        target, a subtree with no collision geometry, a degenerate hull, an
        unresolvable base frame, or a region the schema's own bounds reject —
        returns a declaration with ``region=None``, i.e. exactly the margins the
        kernel used before the amendment. That includes the case where the
        *incoming* declaration already carried a region: dispatch cannot measure
        geometry, so a region it supplied is overwritten when this producer can
        measure one and dropped when it cannot. The region the kernel reads is
        therefore always this producer's own measurement, never a relayed claim
        (HZ-0097-2/4). Dispatch is stripped upstream too — the rSkill runner
        drops the field before publishing — and this is the same rule enforced
        where it is load-bearing, on the message the kernel actually consumes.

        Args:
            data: Live ``mujoco.MjData``.
            stamp_ns: Consumer's current time, same clock as the declaration's.

        Returns:
            The live declaration (region attached when measurable), or ``None``
            when no declaration is in force at ``stamp_ns``.
        """
        declaration = self._place_declaration
        if declaration is None or not declaration.is_live(now_ns=stamp_ns):
            return None
        region = self._place_region(data, stamp_ns=stamp_ns)
        if region is None and declaration.region is None:
            return declaration
        return declaration.model_copy(update={"region": region})

    def _place_region(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        stamp_ns: int,
    ) -> PlaceRegion | None:
        """Pose the declared target's measured box in the robot base frame."""
        if self._place_target_body_id is None or self._base_body_id is None:
            return None
        if self._place_region_local is None:
            self._place_region_local = subtree_region_box(
                self._model,
                data,
                root_body_id=self._place_target_body_id,
            )
            if self._place_region_local is None:
                return None
        if self._place_geometry_local is None:
            self._place_geometry_local = self._place_target_geometry(data)
        centre_in_body, half_extents = self._place_region_local
        translation, rotation = _relative_pose(
            data,
            parent_body_id=self._base_body_id,
            child_body_id=self._place_target_body_id,
        )
        centre_in_base = translation + rotation @ centre_in_body
        try:
            return PlaceRegion(
                frame_id=self._base_frame_id,
                geometry=tuple(
                    self._primitive_in_base(primitive, translation, rotation)
                    for primitive in self._place_geometry_local
                ),
                pose=Pose6D(
                    xyz=(
                        float(centre_in_base[0]),
                        float(centre_in_base[1]),
                        float(centre_in_base[2]),
                    ),
                    quat_xyzw=_matrix_to_quat_xyzw(rotation),
                    frame_id=self._base_frame_id,
                ),
                half_extents=(
                    float(half_extents[0]),
                    float(half_extents[1]),
                    float(half_extents[2]),
                ),
                evidence_ref=f"mujoco_body_subtree:{self._place_target_body_name}",
                stamp_ns=stamp_ns,
            )
        except ValueError:
            # A degenerate or over-large hull. Refusing it restores the
            # pre-amendment margin, which is the fail-closed direction here —
            # unlike a malformed witness, a bad region can only ever make the
            # kernel more permissive.
            return None

    def _place_target_geometry(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    ) -> tuple[AttachedCollisionPrimitive, ...]:
        """Measure the declared target's own collision primitives (ADR-0098).

        The producer half of survey Path B. The region box says *where* the
        declared receptacle is; these say *what it is*, so the safety kernel can
        adjudicate a payload against the modelled shelf instead of against the
        25 mm cubes the shelf was quantised into.

        Measured **once**, from the model, on the same discipline and for the
        same reason as the box: an unmeasurable target is a decided, logged "no
        geometry" rather than a silently varying one. That does mean an
        articulated target measured with its door in one pose keeps that pose —
        and it is why the kernel caps what the geometry may buy at exactly the
        blanket allowance the box alone would have granted. A stale door can
        never make the kernel more permissive than it already was without any
        geometry at all; it can only fail to make it more accurate.

        Returns:
            The target's primitives in the target body's own frame, or an empty
            tuple when the subtree has no collision geometry — which leaves the
            region behaving exactly as it did before ADR-0098.
        """
        if self._place_target_body_id is None:
            return ()
        try:
            return tuple(
                extract_body_primitives(
                    self._model,
                    data,
                    root_body_id=self._place_target_body_id,
                    object_id=self._place_target_body_name,
                    max_primitives=PlaceRegion.MAX_GEOMETRY_PRIMITIVES,
                )
            )
        except ROSConfigError:
            # No collision geometry in the subtree. The box still stands; only
            # the mm-resolution refinement is unavailable.
            return ()

    def _primitive_in_base(
        self,
        primitive: AttachedCollisionPrimitive,
        translation: NDArray[np.float64],
        rotation: NDArray[np.float64],
    ) -> AttachedCollisionPrimitive:
        """Re-pose one target-frame primitive into the robot base frame.

        The shape is unchanged; only the pose moves, because the base frame does
        and the receptacle does not. This runs per publication for the same
        reason the region box's pose does.
        """
        pose = primitive.pose_in_object
        centre = translation + rotation @ np.asarray(pose.xyz, dtype=np.float64)
        return primitive.model_copy(
            update={
                "pose_in_object": Pose6D(
                    xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
                    quat_xyzw=_matrix_to_quat_xyzw(rotation @ _quat_xyzw_to_matrix(pose.quat_xyzw)),
                    frame_id=self._base_frame_id,
                )
            }
        )

    def _place_witness(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        root: int,
        object_id: str,
        stamp_ns: int,
    ) -> SupportContactWitness | None:
        """Attest place-phase support contact, or explain why there is none.

        Every gate here fails toward *no attestation*, which is the pre-ADR-0097
        behaviour (the kernel stops on the contact).
        """
        declaration = self._place_declaration
        if declaration is None or not declaration.is_live(now_ns=stamp_ns):
            return None
        if declaration.object_id and declaration.object_id != object_id:
            return None
        if self._place_attested_stamp_ns == declaration.stamp_ns:
            return None  # one declaration, one attestation — the hysteresis
        return support_contact_witness(
            self._model,
            data,
            root_body_id=root,
            robot_body_ids=self._robot_body_ids,
            stamp_ns=stamp_ns,
            support_roots=self._place_target_bodies,
        )

    def _place_phase_transition(
        self,
        data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
        *,
        root: int,
        stamp_ns: int,
    ) -> list[AttachedCollisionObject] | None:
        """Arm or disarm the place witness for the carried payload.

        Two transitions, both mid-carry, both re-publishing the whole
        attachment set (the atomic snapshot contract World State ingests):

        1. **Arm.** A live declaration, fresh bounded contact on the declared
           target, and no attestation yet for this declaration — attest.
        2. **Disarm.** A declaration this tracker already attested under has
           been retracted or has timed out — re-publish with no witness at
           all, which is what makes the kernel drop the exemption rather than
           carry it past the goal that justified it (HZ-0097-3).

        Returns:
            The new attachment set, or ``None`` when nothing changed.
        """
        if self._place_declaration is None and self._place_attested_stamp_ns is None:
            return None  # no place phase has ever been declared for this carry

        import mujoco  # noqa: PLC0415  # reason: optional sim dependency

        body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, root)
        if not body_name:
            raise ROSConfigError(f"Attached MuJoCo body {root} has no name.")
        object_id = f"sim:{body_name}"

        declaration = self._place_declaration
        attested_stamp_ns = self._place_attested_stamp_ns
        witness = self._place_witness(
            data,
            root=root,
            object_id=object_id,
            stamp_ns=stamp_ns,
        )
        if witness is not None and declaration is not None:
            self._place_attested_stamp_ns = declaration.stamp_ns
        elif attested_stamp_ns is not None and (
            declaration is None
            or declaration.stamp_ns != attested_stamp_ns
            or not declaration.is_live(now_ns=stamp_ns)
        ):
            self._place_attested_stamp_ns = None
        else:
            return None

        translation, rotation = _relative_pose(
            data,
            parent_body_id=self._attach_body_id,
            child_body_id=root,
        )
        return [
            self._build_attachment(
                data,
                root=root,
                translation=translation,
                rotation=rotation,
                stamp_ns=stamp_ns,
                witness=witness,
            )
        ]
