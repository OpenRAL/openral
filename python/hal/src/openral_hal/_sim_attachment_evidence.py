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

# Below this a summed contact normal carries no direction: the contacts oppose
# each other, so there is no support plane to attest.
_DEGENERATE_NORM = 1e-9


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


def _environment_contacts_by_support(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    payload_bodies: set[int],
    robot_body_ids: frozenset[int],
) -> dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64], float]]]:
    """Group the payload's environment contacts by supporting body root.

    Eligible supports are non-robot and non-free: the world, a counter, a
    cabinet. The gripper holding the payload is excluded (``touch_links``
    already covers it) and so is another free-floating object, which is not
    something the safety kernel may be told to ignore.
    """
    groups: dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64], float]]] = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        body_a = int(model.geom_bodyid[int(contact.geom1)])
        body_b = int(model.geom_bodyid[int(contact.geom2)])
        a_is_payload = body_a in payload_bodies
        b_is_payload = body_b in payload_bodies
        if a_is_payload == b_is_payload:
            continue
        other = body_b if a_is_payload else body_a
        if other in robot_body_ids:
            continue
        support_root = _root_motion_body(model, other)
        if classify_body_mobility(model, support_root) is SimObjectMobility.FREE:
            continue
        groups.setdefault(support_root, []).append(
            (
                np.asarray(contact.pos, dtype=np.float64).copy(),
                np.asarray(contact.frame, dtype=np.float64)[:3].copy(),
                float(contact.dist),
            )
        )
    return groups


def support_contact_witness(
    model: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    data: Any,  # noqa: ANN401  # reason: optional MuJoCo pybind type
    *,
    root_body_id: int,
    robot_body_ids: frozenset[int],
    stamp_ns: int,
) -> SupportContactWitness | None:
    """Attest one payload's bounded support contact from exact MuJoCo contacts.

    A grasped object is routinely still resting on the counter it was picked
    from. That contact is real, legitimate, and — once the payload is checked
    as robot geometry — indistinguishable to the safety kernel from driving the
    payload through a wall. This produces the attestation that tells the two
    apart, from ground truth the simulator already has (ADR-0092 D6).

    Only a *non-free* environment body counts as a support: the world, a
    counter, a cabinet. Another free-floating object is not something the
    kernel may be told to ignore, and neither is the gripper holding the
    payload (that is what ``touch_links`` covers). ``None`` — no contact, or
    only ineligible ones — is the honest answer and yields no exemption.

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
    groups = _environment_contacts_by_support(
        model,
        data,
        payload_bodies=payload_bodies,
        robot_body_ids=robot_body_ids,
    )
    if not groups:
        return None

    # One witness names one support surface: the one carrying this payload.
    support_root = max(groups, key=lambda key: len(groups[key]))
    contacts = groups[support_root]
    rotation = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.xpos[root_body_id], dtype=np.float64)
    points = np.asarray([rotation.T @ (position - origin) for position, _, _ in contacts])

    # Orient every contact normal from the supporting solid toward the payload,
    # so the attested half-space is unambiguous regardless of which geom MuJoCo
    # happened to list first.
    oriented = []
    for position, normal, _ in contacts:
        local_normal = rotation.T @ normal
        toward_payload = rotation.T @ (origin - position)
        if float(np.dot(toward_payload, local_normal)) < 0.0:
            local_normal = -local_normal
        oriented.append(local_normal)
    normal_in_object = np.mean(np.asarray(oriented), axis=0)
    norm = float(np.linalg.norm(normal_in_object))
    if norm < _DEGENERATE_NORM:
        # Opposed normals — the payload is pinched, not supported. Attesting a
        # plane here would be inventing geometry, so attest nothing.
        return None
    normal_in_object = normal_in_object / norm
    contact_point = points.mean(axis=0)

    # The patch spans the payload's own supported footprint: its collision
    # geometry projected onto the support plane. Bounded by the payload, and
    # nothing outside it is ever exempt.
    subtree_geoms = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in payload_bodies
        and int(model.geom_contype[geom_id]) != 0
    ]
    if not subtree_geoms:
        return None
    corners = np.concatenate(
        [
            _geom_corners_in_root(model, data, geom_id=geom_id, root_body_id=root_body_id)
            for geom_id in subtree_geoms
        ],
        axis=0,
    )
    offsets = corners - contact_point
    lateral = offsets - np.outer(offsets @ normal_in_object, normal_in_object)
    patch_radius = float(np.linalg.norm(lateral, axis=1).max())
    if not math.isfinite(patch_radius) or patch_radius <= 0.0:
        return None

    penetration = max(0.0, *(-distance for _, _, distance in contacts))

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
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"mujoco_body:{support_name}",
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
