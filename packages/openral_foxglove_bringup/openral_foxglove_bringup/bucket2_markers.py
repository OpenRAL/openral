#!/usr/bin/env python3
"""Bucket-2 converter node: OpenRAL custom msgs → standard ROS viz types.

Subscribes to two ``openral_msgs`` topics and re-publishes them as
standard ROS visualization types that Foxglove renders natively without
any TypeScript extension (ADR-0059 §Decision 5):

- ``/openral/world_collisions`` (``openral_msgs/WorldCollision``)
  → ``/openral/world_collisions_markers`` (``visualization_msgs/MarkerArray``)

  Each capsule obstacle is approximated as a CYLINDER marker of the same
  radius and length 2 × half_length.  A sphere (half_length == 0) is
  emitted as a CYLINDER of length 0; Foxglove renders it visually as a
  squashed disc — operators should note the approximation.  The exact
  capsule geometry (hemispherical end-caps) is not representable as a
  single standard Marker type; a two-marker approach (cylinder + two
  spheres) was considered and rejected as noisy in the panel.

- ``/openral/world_voxels`` (``openral_msgs/OccupancyVoxels``)
  → ``/openral/world_voxels_cloud`` (``sensor_msgs/PointCloud2``)

  One point per occupied voxel, placed at the voxel CENTRE.

The conversion math lives in **pure, ROS-free functions** (``capsule_markers``
and ``occupied_voxel_centers``) that operate on plain Python data so the
unit tests can exercise them without a ROS context (CLAUDE.md §1.11).
All ``rclpy`` / ``openral_msgs`` / ROS message imports are deferred
inside node methods — the repo's standard PLC0415 pattern, which is
ruff-exempt for ``packages/**``.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

__all__ = [
    "Bucket2MarkersNode",
    "MarkerSpec",
    "capsule_markers",
    "main",
    "occupied_voxel_centers",
]

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure data types and conversion functions (no ROS imports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerSpec:
    """Plain data representing one ``visualization_msgs/Marker`` to emit.

    All lengths are in metres; angles in radians.  ``q`` is a
    (qx, qy, qz, qw) quaternion.  ``marker_type`` matches the
    ``visualization_msgs/Marker`` integer constants (CYLINDER = 3).
    """

    marker_id: int
    ns: str
    pos_x: float
    pos_y: float
    pos_z: float
    q_x: float
    q_y: float
    q_z: float
    q_w: float
    scale_x: float  # diameter for CYLINDER
    scale_y: float  # diameter for CYLINDER
    scale_z: float  # length for CYLINDER
    marker_type: int = 3  # CYLINDER


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw (radians) to a (qx, qy, qz, qw) quaternion.

    Uses the ZYX (yaw-pitch-roll) intrinsic convention matching ROS TF2.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def capsule_markers(
    radius: list[float],
    half_length: list[float],
    origin_xyzrpy: list[float],
    object_id: list[str],
) -> list[MarkerSpec]:
    """Convert parallel capsule arrays from ``WorldCollision`` to marker specs.

    Args:
        radius: Per-obstacle capsule radius (metres). Length N.
        half_length: Per-obstacle half-length of the cylinder shaft (metres).
            0 → sphere (emitted as zero-length CYLINDER).  Length N.
        origin_xyzrpy: Flat array of 6 floats per obstacle: x, y, z (metres),
            roll, pitch, yaw (radians).  Length 6N.
        object_id: Per-obstacle label string.  Length N (may be empty strings).

    Returns:
        One :class:`MarkerSpec` per obstacle, index-parallel with the inputs.

    Raises:
        ValueError: If ``len(radius) != len(half_length)``, the 6N invariant
            is violated, or ``len(object_id) != len(radius)``.
    """
    n = len(radius)
    if len(half_length) != n:
        raise ValueError(
            f"radius and half_length must have the same length: {n} vs {len(half_length)}"
        )
    if len(origin_xyzrpy) != 6 * n:
        raise ValueError(f"origin_xyzrpy must have length 6*N={6 * n}, got {len(origin_xyzrpy)}")
    if len(object_id) != n:
        raise ValueError(f"object_id must have the same length as radius: {n} vs {len(object_id)}")

    specs: list[MarkerSpec] = []
    for i in range(n):
        base = 6 * i
        x, y, z = origin_xyzrpy[base], origin_xyzrpy[base + 1], origin_xyzrpy[base + 2]
        roll, pitch, yaw = (
            origin_xyzrpy[base + 3],
            origin_xyzrpy[base + 4],
            origin_xyzrpy[base + 5],
        )
        qx, qy, qz, qw = _rpy_to_quaternion(roll, pitch, yaw)

        r = radius[i]
        length = 2.0 * half_length[i]
        ns = object_id[i] if object_id[i] else f"obstacle_{i}"

        specs.append(
            MarkerSpec(
                marker_id=i,
                ns=ns,
                pos_x=x,
                pos_y=y,
                pos_z=z,
                q_x=qx,
                q_y=qy,
                q_z=qz,
                q_w=qw,
                scale_x=2.0 * r,  # CYLINDER scale_x/y = diameter
                scale_y=2.0 * r,
                scale_z=length,
            )
        )
    return specs


def occupied_voxel_centers(
    origin: tuple[float, float, float],
    resolution: float,
    size: tuple[int, int, int],
    occupancy: Sequence[int],
) -> list[tuple[float, float, float]]:
    """Return the centre positions of all occupied voxels.

    Voxel indexing (row-major, x fastest):
        ``idx = x + size_x * (y + size_y * z)``

    Centre of voxel (x, y, z):
        ``(origin_x + (x + 0.5) * resolution,
           origin_y + (y + 0.5) * resolution,
           origin_z + (z + 0.5) * resolution)``

    Args:
        origin: (ox, oy, oz) minimum corner of voxel (0, 0, 0) in metres.
        resolution: Edge length of one voxel in metres.
        size: (size_x, size_y, size_z) grid dimensions.
        occupancy: Flat occupancy array (length size_x*size_y*size_z).
            Non-zero → occupied.

    Returns:
        List of (cx, cy, cz) centre coordinates for occupied voxels.
    """
    ox, oy, oz = origin
    sx, sy, sz = size
    expected = sx * sy * sz
    if len(occupancy) != expected:
        raise ValueError(f"occupancy length {len(occupancy)} != size_x*size_y*size_z={expected}")

    centers: list[tuple[float, float, float]] = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                idx = x + sx * (y + sy * z)
                if occupancy[idx]:
                    cx = ox + (x + 0.5) * resolution
                    cy = oy + (y + 0.5) * resolution
                    cz = oz + (z + 0.5) * resolution
                    centers.append((cx, cy, cz))
    return centers


# ---------------------------------------------------------------------------
# ROS node (all rclpy / message imports deferred to methods — PLC0415)
# ---------------------------------------------------------------------------


class Bucket2MarkersNode:
    """Read-only converter node for Bucket-2 custom message types.

    Subscribes to ``/openral/world_collisions`` and
    ``/openral/world_voxels`` and re-publishes them as standard ROS
    visualization types.  Never commands the robot.
    """

    def __init__(self) -> None:
        """Initialise subscriptions and publishers, then log readiness."""
        import rclpy
        import rclpy.node
        import rclpy.qos

        self._node: rclpy.node.Node = rclpy.node.Node("bucket2_markers")

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        from openral_msgs.msg import OccupancyVoxels, WorldCollision
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import MarkerArray

        self._pub_markers = self._node.create_publisher(
            MarkerArray, "/openral/world_collisions_markers", qos
        )
        self._pub_cloud = self._node.create_publisher(
            PointCloud2, "/openral/world_voxels_cloud", qos
        )

        self._sub_collisions = self._node.create_subscription(
            WorldCollision,
            "/openral/world_collisions",
            self._on_world_collisions,
            qos,
        )
        self._sub_voxels = self._node.create_subscription(
            OccupancyVoxels,
            "/openral/world_voxels",
            self._on_world_voxels,
            qos,
        )

        log.info("bucket2_markers node ready")

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_world_collisions(self, msg: object) -> None:
        """Convert WorldCollision → MarkerArray and publish."""
        from visualization_msgs.msg import Marker, MarkerArray

        # Extract parallel arrays from the ROS message
        radius = list(msg.radius)  # type: ignore[union-attr]
        half_length = list(msg.half_length)  # type: ignore[union-attr]
        origin_xyzrpy = list(msg.origin_xyzrpy)  # type: ignore[union-attr]
        object_id = list(msg.object_id)  # type: ignore[union-attr]

        try:
            specs = capsule_markers(radius, half_length, origin_xyzrpy, object_id)
        except ValueError:
            log.exception("bucket2: malformed WorldCollision message — skipping")
            return

        array = MarkerArray()
        for spec in specs:
            m = Marker()
            m.header = msg.header  # type: ignore[union-attr]
            m.ns = spec.ns
            m.id = spec.marker_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = spec.pos_x
            m.pose.position.y = spec.pos_y
            m.pose.position.z = spec.pos_z
            m.pose.orientation.x = spec.q_x
            m.pose.orientation.y = spec.q_y
            m.pose.orientation.z = spec.q_z
            m.pose.orientation.w = spec.q_w
            m.scale.x = spec.scale_x
            m.scale.y = spec.scale_y
            m.scale.z = spec.scale_z if spec.scale_z > 0.0 else 0.001  # avoid zero scale
            # Semi-transparent cyan so capsules don't occlude the robot model
            m.color.r = 0.0
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = 0.4
            array.markers.append(m)

        self._pub_markers.publish(array)
        log.debug("bucket2: published world_collisions_markers", count=len(specs))

    def _on_world_voxels(self, msg: object) -> None:
        """Convert OccupancyVoxels → PointCloud2 and publish."""
        from sensor_msgs.msg import PointCloud2, PointField

        origin = (msg.origin.x, msg.origin.y, msg.origin.z)  # type: ignore[union-attr]
        size = (int(msg.size_x), int(msg.size_y), int(msg.size_z))  # type: ignore[union-attr]
        occupancy = list(msg.occupancy)  # type: ignore[union-attr]
        resolution = float(msg.resolution)  # type: ignore[union-attr]

        try:
            centers = occupied_voxel_centers(origin, resolution, size, occupancy)
        except ValueError:
            log.exception("bucket2: malformed OccupancyVoxels message — skipping")
            return

        cloud = PointCloud2()
        cloud.header = msg.header  # type: ignore[union-attr]
        cloud.height = 1
        cloud.width = len(centers)
        cloud.is_dense = True
        cloud.is_bigendian = False

        # xyz as three float32 fields (4 bytes each, 12 bytes per point)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width

        raw = bytearray(cloud.row_step)
        for i, (cx, cy, cz) in enumerate(centers):
            offset = i * 12
            struct.pack_into("fff", raw, offset, cx, cy, cz)
        cloud.data = bytes(raw)

        self._pub_cloud.publish(cloud)
        log.debug("bucket2: published world_voxels_cloud", points=len(centers))

    # ------------------------------------------------------------------

    def spin(self) -> None:
        """Block and spin the node until shutdown."""
        import rclpy

        rclpy.spin(self._node)

    def destroy(self) -> None:
        """Tear down the node."""
        self._node.destroy_node()


def main() -> None:
    """Entry point for the ``bucket2_markers`` console script."""
    import rclpy

    rclpy.init()
    node = Bucket2MarkersNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()
