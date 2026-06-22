"""cuRobo robot-config emission from the lowered safety collision model (ADR-0065).

cuMotion's planner (cuRobo) represents each link's collision volume as a set of
**spheres**, while the OpenRAL safety kernel lowers links to **capsules/spheres**
(ADR-0030). To keep plan-time (cuMotion) and kernel-time collision geometry
consistent — a single source of truth — this module derives cuRobo collision
spheres directly from the same lowered capsule geometry, by sampling spheres
along each capsule's central segment.

Pure module: no ROS, no cuRobo import, no I/O. The only heavy dependency
(``numpy``) is pulled in lazily by the reused capsule→segment helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from openral_core import CapsuleShape, JointType, LinkCollisionGeometry, SphereShape

if TYPE_CHECKING:
    from openral_core import RobotDescription

    from openral_safety.urdf_lowering import LoweredCollisionModel

# Reuse the kernel's own capsule→segment lowering so plan-time spheres are fitted
# to the exact same geometry the safety kernel checks (ADR-0065 D4). Importing the
# private helper is deliberate: duplicating the capsule axis math here would risk
# the two collision models silently diverging, which is safety-relevant.
from openral_safety.urdf_lowering import _capsule_segment_radius

__all__ = [
    "CuMotionSphere",
    "actuated_joint_names",
    "capsule_to_spheres",
    "link_collision_spheres",
    "render_cumotion_config",
    "spheres_for_capsule",
]

# Single-DOF movable joints cuMotion plans over. FIXED/FLOATING/PLANAR are
# excluded — fixed joints don't move; floating/planar are multi-DOF and not part
# of an arm cspace.
_ACTUATED_JOINT_TYPES = frozenset({JointType.REVOLUTE, JointType.PRISMATIC, JointType.CONTINUOUS})

_Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CuMotionSphere:
    """One cuRobo collision sphere in its link frame.

    Attributes:
        center: ``(x, y, z)`` sphere centre in metres, in the link frame.
        radius: Sphere radius in metres (the capsule's radius).
    """

    center: _Vec3
    radius: float


def capsule_to_spheres(p0: _Vec3, p1: _Vec3, radius: float, *, count: int) -> list[CuMotionSphere]:
    """Sample ``count`` spheres of ``radius`` evenly along the segment ``p0``→``p1``.

    For ``count >= 2`` the spheres include both endpoints (parameters
    ``t = i / (count - 1)``). For ``count == 1`` a single sphere is placed at the
    segment midpoint, which collapses to the point itself for a zero-length
    (sphere) segment.

    Args:
        p0: Segment start in the link frame.
        p1: Segment end in the link frame.
        radius: Sphere radius in metres (must be > 0).
        count: Number of spheres to emit (must be >= 1).

    Returns:
        The sampled spheres, ordered from ``p0`` to ``p1``.

    Example:
        >>> [s.center for s in capsule_to_spheres((0, 0, 0), (1, 0, 0), 0.1, count=3)]
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count == 1:
        mid: _Vec3 = (
            (p0[0] + p1[0]) / 2.0,
            (p0[1] + p1[1]) / 2.0,
            (p0[2] + p1[2]) / 2.0,
        )
        return [CuMotionSphere(center=mid, radius=radius)]
    spheres: list[CuMotionSphere] = []
    for i in range(count):
        t = i / (count - 1)
        center: _Vec3 = (
            p0[0] + (p1[0] - p0[0]) * t,
            p0[1] + (p1[1] - p0[1]) * t,
            p0[2] + (p1[2] - p0[2]) * t,
        )
        spheres.append(CuMotionSphere(center=center, radius=radius))
    return spheres


def spheres_for_capsule(shape: CapsuleShape | SphereShape) -> int:
    """Number of spheres needed to tile a lowered ``shape`` with spacing <= radius.

    A sphere (or a zero-length capsule) needs exactly one. A capsule of length
    ``L`` and radius ``r`` needs ``ceil(L / r) + 1`` spheres so adjacent centres
    sit no more than one radius apart, covering the swept volume.

    Example:
        >>> spheres_for_capsule(CapsuleShape(radius_m=0.05, length_m=0.30))
        7
    """
    if isinstance(shape, SphereShape) or shape.length_m == 0.0:
        return 1
    return max(2, math.ceil(shape.length_m / shape.radius_m) + 1)


def link_collision_spheres(
    geom: LinkCollisionGeometry, *, count: int | None = None
) -> list[CuMotionSphere]:
    """Lower one link's collision volume to cuRobo spheres in the link frame.

    Reuses the kernel's capsule→segment lowering, then samples spheres along the
    resulting segment. ``count`` defaults to :func:`spheres_for_capsule`.

    Args:
        geom: A lowered link collision volume (capsule or sphere).
        count: Optional explicit sphere count; defaults to the radius-spacing
            heuristic.

    Returns:
        The link's cuRobo collision spheres.
    """
    p0, p1, radius = _capsule_segment_radius(geom.shape, geom.origin_xyz_rpy)
    n = spheres_for_capsule(geom.shape) if count is None else count
    return capsule_to_spheres(p0, p1, radius, count=n)


def actuated_joint_names(robot: RobotDescription) -> list[str]:
    """The robot's single-DOF movable joint names, in manifest order.

    These become the cuRobo ``cspace.joint_names`` — the joints the planner
    solves for. Fixed/floating/planar joints are excluded.
    """
    return [j.name for j in robot.joints if j.joint_type in _ACTUATED_JOINT_TYPES]


_SPHERE_DP = 6


def render_cumotion_config(robot: RobotDescription, model: LoweredCollisionModel) -> str:
    """Render a cuRobo ``robot_cfg`` fragment from a lowered collision model (ADR-0065).

    Emits the collision geometry cuMotion needs — per-link ``collision_spheres``
    sampled from the kernel's own lowered capsules, ``self_collision_ignore`` from
    the allowed-collision matrix, and ``cspace.joint_names`` — so plan-time and
    kernel-time collision geometry share one source of truth (ADR-0065 D4).

    ``retract_config`` and acceleration/jerk limits are planner tuning, not
    geometry; they are intentionally left out and added when validated against a
    live cuRobo install (ADR-0065 Phase 3). The header documents this.

    Args:
        robot: The robot manifest (provides ``base_frame`` and joints).
        model: The lowered collision model (capsule geometry + ACM).

    Returns:
        A YAML document string with a generated-provenance header.
    """
    collision_spheres: dict[str, list[dict[str, object]]] = {}
    for g in model.collision_geometry:
        collision_spheres[g.link_name] = [
            {
                "center": [round(c, _SPHERE_DP) for c in s.center],
                "radius": round(s.radius, _SPHERE_DP),
            }
            for s in link_collision_spheres(g)
        ]

    ignore: dict[str, set[str]] = {}
    for a, b in model.allowed_collision_pairs:
        ignore.setdefault(a, set()).add(b)
        ignore.setdefault(b, set()).add(a)
    self_collision_ignore = {k: sorted(v) for k, v in sorted(ignore.items())}

    cfg: dict[str, object] = {
        "robot_cfg": {
            "kinematics": {
                "base_link": robot.base_frame,
                "collision_link_names": [g.link_name for g in model.collision_geometry],
                "collision_spheres": collision_spheres,
                "self_collision_ignore": self_collision_ignore,
                "cspace": {"joint_names": actuated_joint_names(robot)},
            }
        }
    }

    header = (
        "# GENERATED by `openral collision lower --emit-cumotion` (ADR-0065) — do not hand-edit.\n"
        f"# Source ACM: {model.acm_source}. collision_spheres + self_collision_ignore derive\n"
        "# from the same lowered geometry the OpenRAL safety kernel checks (ADR-0030), so\n"
        "# cuMotion plan-time and kernel-time collision stay consistent. retract_config and\n"
        "# accel/jerk limits are planner tuning — add + validate vs cuRobo (ADR-0065 Phase 3).\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False)
