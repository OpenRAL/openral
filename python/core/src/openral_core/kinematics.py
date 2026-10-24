"""Manifest forward kinematics and oriented-box helpers, below every layer that poses links.

Moved down from ``openral_hal.vision_attachment_bridge`` (the release window's FK) so the
Layer-2 object tracker (issue #418) poses the hand from joint states without importing the
HAL. The joint convention is URDF's and the safety kernel's: ``origin · motion(q)`` per
edge, fixed-axis roll-pitch-yaw ``R = Rz Ry Rx``, a ``MimicJoint`` reads its leader's
value through ``value_at``. ``tests/unit/test_urdf_joint_map.py`` pins it against the URDF.

Kept out of ``openral_core.__init__`` so importing the core schemas stays numpy-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    BoxShape,
    CapsuleShape,
    CollisionShape,
    FixedAttachment,
    JointSpec,
    JointType,
    MimicJoint,
    RobotDescription,
    SphereShape,
)

__all__ = [
    "Box",
    "bounding_half_extents",
    "box_gap_lower_bound_m",
    "edge_chain",
    "joint_motion",
    "link_transform",
    "transform_from_xyz_rpy",
]

#: An oriented box in one frame: ``(centre (3,), rotation (3, 3), half extents (3,))``.
Box = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
#: One kinematic-tree edge: a movable joint, a mimic follower or a rigid attachment.
Edge = JointSpec | MimicJoint | FixedAttachment

#: Below this a cross-product axis is degenerate (parallel edges).
_PARALLEL_EPS = 1e-9


def bounding_half_extents(shape: CollisionShape) -> NDArray[np.float64]:
    """Half extents of a box containing a primitive, in the primitive's frame.

    A box is itself, a sphere its cube, a capsule (segment along local +Z) its
    ``(r, r, L/2 + r)`` box. Containment makes every distance measured on the result a
    lower bound on the distance to the primitive.

    Example:
        >>> half = bounding_half_extents(CapsuleShape(radius_m=0.01, length_m=0.1))
        >>> [round(float(v), 6) for v in half]
        [0.01, 0.01, 0.06]
    """
    if isinstance(shape, BoxShape):
        return np.asarray(shape.half_extents_m, dtype=np.float64)
    if isinstance(shape, SphereShape):
        return np.full(3, float(shape.radius_m), dtype=np.float64)
    if isinstance(shape, CapsuleShape):
        r = float(shape.radius_m)
        return np.array([r, r, float(shape.length_m) / 2.0 + r], dtype=np.float64)
    raise ROSConfigError(f"cannot bound collision primitive {shape!r}.")


def transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> NDArray[np.float64]:
    """``(4, 4)`` from a translation and fixed-axis roll-pitch-yaw (``R = Rz Ry Rx``, URDF).

    Example:
        >>> transform_from_xyz_rpy((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))[:3, 3].tolist()
        [1.0, 0.0, 0.0]
    """
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def joint_motion(joint: JointSpec | MimicJoint, q: float) -> NDArray[np.float64]:
    """Child-in-parent motion of one joint at position ``q`` (applied after its origin).

    Raises:
        ROSConfigError: The joint is neither revolute, continuous nor prismatic.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> jaw = next(j for j in d.joints if j.name == "gripper")
        >>> round(float(joint_motion(jaw, np.pi / 2)[0, 1]), 6)
        -1.0
    """
    axis = np.asarray(joint.axis_xyz, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = np.eye(4, dtype=np.float64)
    if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
        k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        t[:3, :3] = np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)
    elif joint.joint_type is JointType.PRISMATIC:
        t[:3, 3] = q * axis
    else:
        raise ROSConfigError(
            f"joint {joint.name!r} is {joint.joint_type.value}; only revolute, continuous "
            "and prismatic joints are posed."
        )
    return t


def edge_chain(
    description: RobotDescription, root: str, link: str
) -> tuple[tuple[Edge, NDArray[np.float64]], ...] | None:
    """The ``(edge, origin)`` chain from ``root`` down to ``link``, or ``None`` off its subtree.

    Each link has one defining edge (``RobotDescription`` validates it): a joint, a
    mimic joint or a fixed attachment, each with its origin in the parent frame.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/droid_franka_robotiq/robot.yaml")
        >>> [e.name for e, _ in edge_chain(d, "robotiq_arg2f_base_link", "left_bar")]
        ['left_outer_knuckle_joint', 'left_bar_joint']
        >>> edge_chain(d, "robotiq_arg2f_base_link", "panda_link3") is None
        True
    """
    edge_of: dict[str, Edge] = {
        **{j.child_link: j for j in description.joints},
        **{m.child_link: m for m in description.mimic_joints},
        **{a.child_link: a for a in description.fixed_attachments},
    }
    chain: list[tuple[Edge, NDArray[np.float64]]] = []
    current = link
    while current != root:
        edge = edge_of.get(current)
        if edge is None:
            return None
        chain.append((edge, transform_from_xyz_rpy(edge.origin_xyz, edge.origin_rpy)))
        current = edge.parent_link
    return tuple(reversed(chain))


def link_transform(
    description: RobotDescription, root: str, link: str, positions: Mapping[str, float]
) -> NDArray[np.float64]:
    """``T_root<-link`` at joint ``positions`` (by ``JointSpec.name``), mimic-aware.

    A ``MimicJoint`` reads its leader's position through ``value_at``; never posed from a
    ``robot_state_publisher`` tf, which reads the SO-101's jaw fraction as radians.

    Raises:
        ROSConfigError: ``link`` is not in ``root``'s subtree, or a joint on the chain has
            no position.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> t = link_transform(d, "gripper", "jaw", {"gripper": 0.0})
        >>> [round(float(v), 4) for v in t[:3, 3]]
        [0.0202, 0.0188, -0.0234]
    """
    chain = edge_chain(description, root, link)
    if chain is None:
        raise ROSConfigError(f"link {link!r} is not in the subtree of {root!r}.")
    t = np.eye(4, dtype=np.float64)
    for edge, origin in chain:
        t = t @ origin
        if isinstance(edge, FixedAttachment):
            continue
        driver = edge.mimics if isinstance(edge, MimicJoint) else edge.name
        if driver not in positions:
            raise ROSConfigError(f"no position for joint {driver!r} posing {link!r}.")
        q = float(positions[driver])
        t = t @ joint_motion(edge, edge.value_at(q) if isinstance(edge, MimicJoint) else q)
    return t


def box_gap_lower_bound_m(a: Box, b: Box) -> float:
    """Separating-axis lower bound on the distance between two oriented boxes.

    The gap between the boxes' projections onto any unit axis never exceeds their
    distance (projection is 1-Lipschitz), so the largest gap over the 15 SAT axes is a
    sound lower bound — exact along a face normal, conservative across an edge-edge
    pair. ``<= 0`` exactly when the boxes intersect (the 15 axes decide OBB overlap).

    Args:
        a: ``(centre, rotation, half_extents)`` of the first box.
        b: The second box, same frame.

    Returns:
        A lower bound on the surface distance, metres.

    Example:
        >>> unit = (np.zeros(3), np.eye(3), np.full(3, 0.5))
        >>> moved = (np.array([1.05, 0.0, 0.0]), np.eye(3), np.full(3, 0.5))
        >>> round(box_gap_lower_bound_m(unit, moved), 6)
        0.05
    """
    (ca, ra, ha), (cb, rb, hb) = a, b
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [
        np.asarray(np.cross(ra[:, i], rb[:, j]), dtype=np.float64)
        for i in range(3)
        for j in range(3)
    ]
    offset = cb - ca
    gap = -np.inf
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm < _PARALLEL_EPS:  # parallel edges: their face normals already cover the pair
            continue
        n = axis / norm
        reach = float(np.sum(ha * np.abs(ra.T @ n)) + np.sum(hb * np.abs(rb.T @ n)))
        gap = max(gap, abs(float(offset @ n)) - reach)
    return float(gap)
