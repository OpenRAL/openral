#!/usr/bin/env python3
"""Bucket-2 converter node: OpenRAL custom msgs → standard ROS viz types.

- ``/openral/world_voxels`` (``OccupancyVoxels``) → ``/openral/world_voxels_cloud``
  (``PointCloud2``): one point per occupied voxel, at the voxel centre.

Conversion math lives in a pure, ROS-free function (``occupied_voxel_centers``)
so unit tests exercise it without a ROS context
(CLAUDE.md §1.11). ``rclpy``/``openral_msgs`` imports are deferred inside node
methods (PLC0415, ruff-exempt for ``packages/**``).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import structlog

__all__ = [
    "Bucket2MarkersNode",
    "main",
    "occupied_voxel_centers",
]

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure data types and conversion functions (no ROS imports)
# ---------------------------------------------------------------------------


def occupied_voxel_centers(
    origin: tuple[float, float, float],
    resolution: float,
    size: tuple[int, int, int],
    occupancy: Sequence[int],
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> list[tuple[float, float, float]]:
    """Return the centre positions of all occupied voxels.

    Row-major indexing, x fastest: ``idx = x + size_x * (y + size_y * z)``.
    ``openral_msgs/OccupancyVoxels`` is an ORIENTED lattice — its cell axes
    are the source map's, not ``header.frame_id``'s — so dropping the
    rotation draws every voxel in the wrong place whenever the robot isn't
    map-aligned.

    Args:
        origin: (ox, oy, oz) minimum corner of voxel (0, 0, 0) in metres.
        resolution: Edge length of one voxel in metres.
        size: (size_x, size_y, size_z) grid dimensions.
        occupancy: Flat occupancy array (length size_x*size_y*size_z).
            Non-zero → occupied.
        orientation_xyzw: Grid orientation ``(x, y, z, w)``. Defaults to
            identity for callers building an axis-aligned grid themselves; a
            caller decoding a wire message passes the message's own. An all-zero
            quaternion (the unset default) is rejected, never read as identity.

    Returns:
        List of (cx, cy, cz) centre coordinates for occupied voxels.

    Raises:
        ValueError: On a length mismatch, or a non-unit ``orientation_xyzw``.
    """
    ox, oy, oz = origin
    sx, sy, sz = size
    expected = sx * sy * sz
    if len(occupancy) != expected:
        raise ValueError(f"occupancy length {len(occupancy)} != size_x*size_y*size_z={expected}")
    qx, qy, qz, qw = orientation_xyzw
    norm2 = qx * qx + qy * qy + qz * qz + qw * qw
    if not math.isfinite(norm2) or abs(norm2 - 1.0) > 1e-6:
        raise ValueError(
            f"orientation {orientation_xyzw!r} is not a unit quaternion "
            "(an unset OccupancyVoxels.orientation is all zeros, not identity)"
        )
    rot = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )

    centers: list[tuple[float, float, float]] = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                idx = x + sx * (y + sy * z)
                if occupancy[idx]:
                    lx = (x + 0.5) * resolution
                    ly = (y + 0.5) * resolution
                    lz = (z + 0.5) * resolution
                    centers.append(
                        (
                            ox + rot[0][0] * lx + rot[0][1] * ly + rot[0][2] * lz,
                            oy + rot[1][0] * lx + rot[1][1] * ly + rot[1][2] * lz,
                            oz + rot[2][0] * lx + rot[2][1] * ly + rot[2][2] * lz,
                        )
                    )
    return centers


# ---------------------------------------------------------------------------
# ROS node (all rclpy / message imports deferred to methods — PLC0415)
# ---------------------------------------------------------------------------


class Bucket2MarkersNode:
    """Read-only converter node for Bucket-2 custom message types.

    Subscribes to ``/openral/world_voxels`` and re-publishes it as a
    standard ROS visualization type.  Never commands the robot.
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

        from openral_msgs.msg import OccupancyVoxels
        from sensor_msgs.msg import PointCloud2

        self._pub_cloud = self._node.create_publisher(
            PointCloud2, "/openral/world_voxels_cloud", qos
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

    def _on_world_voxels(self, msg: object) -> None:
        """Convert OccupancyVoxels → PointCloud2 and publish."""
        from sensor_msgs.msg import PointCloud2, PointField

        origin = (msg.origin.x, msg.origin.y, msg.origin.z)  # type: ignore[union-attr]
        size = (int(msg.size_x), int(msg.size_y), int(msg.size_z))  # type: ignore[union-attr]
        occupancy = list(msg.occupancy)  # type: ignore[union-attr]
        resolution = float(msg.resolution)  # type: ignore[union-attr]
        # The grid's lattice is the OctoMap's; without its rotation every voxel
        # is drawn somewhere the obstacle is not.
        orientation = (
            float(msg.orientation.x),  # type: ignore[union-attr]
            float(msg.orientation.y),  # type: ignore[union-attr]
            float(msg.orientation.z),  # type: ignore[union-attr]
            float(msg.orientation.w),  # type: ignore[union-attr]
        )

        try:
            centers = occupied_voxel_centers(origin, resolution, size, occupancy, orientation)
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
