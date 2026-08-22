#!/usr/bin/env python3
"""Drop the carried payload's own returns out of the ``/scan`` Nav2 costmaps read.

The other half of the dynamic footprint. Once
:mod:`openral_nav2_bringup.payload_footprint_node` has folded the payload into
Nav2's *robot*, the same payload must stop being one of Nav2's *obstacles* —
otherwise the base is permanently 1.2 s from colliding with the thing it is
holding, which is the ``collision_monitor`` failure the panda_mobile config
already had to disable a polygon over.

**What actually feeds the costmaps.** Not ``/octomap_binary``. The lidar
profile's ``voxel_layer`` / ``obstacle_layer`` and the ``collision_monitor``
all take ``sensor_msgs/LaserScan`` on ``/scan``
(``config/nav2_panda_mobile.yaml``); the visual profile takes an
``OccupancyGrid`` on ``/map``. So the honest seam for "robot + payload
geometry must not reach the costmap" is the scan topic, and this node sits on
it: ``/scan`` in, ``/scan`` minus the payload out, Nav2 pointed at the output.

**The robot's own geometry is already gone before this node sees it**, which is
why this node only removes the payload. ``openral_sim.backends.robocasa.
synthesize_laser_scan_2d`` skips every hit whose body shares the robot's
kinematic-tree root and re-casts past it, so a chassis or arm return never
enters ``/scan``; a real 2-D lidar's own structure is masked in the driver.
The carried object is a different MJCF body with a different tree root, so it
is exactly the geometry that survives that filter and needs this one.

**And Nav2 clears its own footprint too** — ``obstacle_layer``'s
``footprint_clearing_enabled`` defaults to ``True`` (verified against the Jazzy
binary), so cells inside the published polygon are freed on every update. That
is a *second* line, not a substitute: it only reaches cells inside the current
polygon, it is a costmap-layer setting a future config change could flip, and
the ``collision_monitor`` reads the raw scan without any costmap in between.
Removing the returns at the source covers all three.

**Failure is pass-through.** Missing attach-link TF, a primitive the kernel
itself would reject, a world state older than ``attached_state_timeout_s``, or
no world state at all republishes the scan **unfiltered**. That leaves the
payload in the costmap, which makes Nav2 more cautious, never less — and it
keeps the output topic alive, because a costmap whose only observation source
went silent is a costmap that stops seeing the room.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from openral_nav2_bringup.payload_footprint_node import (
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_SPHERE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_OUTPUT_TOPIC",
    "filter_scan_ranges",
    "main",
    "points_in_primitive",
]

#: Where the filtered scan is published. ``config/nav2_panda_mobile.yaml``
#: points every costmap observation source and the collision monitor here.
DEFAULT_OUTPUT_TOPIC = "/openral/nav2/scan"


def points_in_primitive(
    points_xyz: Any,
    shape_type: int,
    shape_dimensions: Sequence[float],
    transform: Any,
    *,
    margin_m: float = 0.0,
) -> Any:
    """Boolean mask of which ``points_xyz`` lie inside one attached primitive.

    The containment test is the kernel's own
    (``openral_octomap_bridge``'s ``surface_distance``, at zero slack): sphere
    ``[radius]``, capsule ``[radius, central_segment_length]`` about the
    primitive's local +Z, box ``[half_x, half_y, half_z]``.

    Args:
        points_xyz: ``(N, 3)`` array of points in the frame ``transform`` maps
            *from* the primitive frame *into*.
        shape_type: One of the ``SHAPE_*`` tags.
        shape_dimensions: Dimensions in metres, per the convention above.
        transform: ``(4, 4)`` homogeneous transform placing the primitive in
            the points' frame.
        margin_m: Extra containment reach, for pose uncertainty. Every
            millimetre here removes sensor returns the payload cannot explain.

    Returns:
        ``(N,)`` boolean array, ``True`` where the point is inside.

    Raises:
        ValueError: On an unknown shape tag, too few or non-positive
            dimensions, a bad transform, or a negative margin.
    """
    import numpy as np

    if margin_m < 0.0 or not math.isfinite(margin_m):
        raise ValueError(f"margin_m must be finite and non-negative; got {margin_m}")
    m = np.asarray(transform, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"transform must be a finite (4, 4) matrix; got shape {m.shape}")
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3); got shape {pts.shape}")
    dims = [float(d) for d in shape_dimensions]

    # Rigid inverse: local = R^T (p - t).
    local = (pts - m[:3, 3]) @ m[:3, :3]

    if shape_type == SHAPE_SPHERE:
        if len(dims) < 1 or not (dims[0] > 0.0 and math.isfinite(dims[0])):
            raise ValueError(f"sphere needs one positive radius; got {dims}")
        return np.linalg.norm(local, axis=1) <= dims[0] + margin_m

    if shape_type == SHAPE_CAPSULE:
        capsule_dims = 2
        if len(dims) < capsule_dims or not all(d > 0.0 and math.isfinite(d) for d in dims[:2]):
            raise ValueError(f"capsule needs positive [radius, length]; got {dims}")
        half_length = 0.5 * dims[1]
        z = np.clip(local[:, 2], -half_length, half_length)
        radial = np.stack([local[:, 0], local[:, 1], local[:, 2] - z], axis=1)
        return np.linalg.norm(radial, axis=1) <= dims[0] + margin_m

    if shape_type == SHAPE_BOX:
        box_dims = 3
        if len(dims) < box_dims or not all(d > 0.0 and math.isfinite(d) for d in dims[:3]):
            raise ValueError(f"box needs three positive half-extents; got {dims}")
        half = np.asarray(dims[:3], dtype=np.float64) + margin_m
        return np.all(np.abs(local) <= half, axis=1)

    raise ValueError(f"unknown attached-primitive shape_type {shape_type!r}")


def filter_scan_ranges(
    ranges: Sequence[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    placements: Sequence[tuple[int, Sequence[float], Any]],
    margin_m: float = 0.0,
) -> list[float]:
    """Replace payload-explained scan returns with ``inf``.

    A 2-D scan's returns lie in its own plane, so beam *i*'s endpoint is
    ``(r cos(theta_i), r sin(theta_i), 0)`` in the scan frame; a beam is
    dropped when that endpoint is inside some attached primitive. Dropped beams
    become ``inf``, which Nav2 discards outright at the observation buffer
    (``inf_is_valid`` defaults to ``False``, verified against the Jazzy binary)
    — the payload is neither marked as an obstacle nor used to clear the space
    behind it. Readings already outside ``[range_min, range_max]`` are left
    untouched; they are the sensor's own "no return", not ours to reinterpret.

    Args:
        ranges: The scan's ranges, metres.
        angle_min: ``LaserScan.angle_min``, radians.
        angle_increment: ``LaserScan.angle_increment``, radians.
        range_min: ``LaserScan.range_min``, metres.
        range_max: ``LaserScan.range_max``, metres.
        placements: ``(shape_type, shape_dimensions, transform)`` per attached
            primitive, each transform placing it in the **scan** frame.
        margin_m: Containment margin passed to :func:`points_in_primitive`.

    Returns:
        A new range list, same length, with payload beams set to ``inf``.

    Raises:
        ValueError: If a placement is malformed (the caller must then publish
            the scan unfiltered rather than guess).

    Example:
        >>> import numpy as np
        >>> at_one_metre_ahead = np.eye(4)
        >>> at_one_metre_ahead[0, 3] = 1.0
        >>> filter_scan_ranges(
        ...     [1.0, 2.0],
        ...     angle_min=0.0,
        ...     angle_increment=math.pi / 2,
        ...     range_min=0.05,
        ...     range_max=12.0,
        ...     placements=[(1, (0.05,), at_one_metre_ahead)],
        ... )
        [inf, 2.0]
    """
    import numpy as np

    out = np.asarray(ranges, dtype=np.float64).copy()
    if out.size == 0 or not placements:
        return [float(v) for v in out]
    angles = float(angle_min) + float(angle_increment) * np.arange(out.size, dtype=np.float64)
    usable = np.isfinite(out) & (out >= float(range_min)) & (out <= float(range_max))
    # Zero the unusable readings *before* the projection: `inf * cos(pi/2)` is a
    # NaN, and a NaN endpoint would make the containment test's answer depend on
    # numpy's comparison semantics rather than on the mask.
    reach = np.where(usable, out, 0.0)
    pts = np.stack([reach * np.cos(angles), reach * np.sin(angles), np.zeros_like(out)], axis=1)

    drop = np.zeros(out.size, dtype=bool)
    for shape_type, dims, transform in placements:
        drop |= points_in_primitive(pts, int(shape_type), dims, transform, margin_m=margin_m)
    out[drop & usable] = math.inf
    return [float(v) for v in out]


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_nav2_bringup payload_scan_filter_node.py``."""
    import rclpy
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_msgs.msg import WorldStateStamped
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener

    def _pose_matrix(pose: Any) -> Any:
        q, p = pose.orientation, pose.position
        return homogeneous_from_quat_xyz(
            (float(p.x), float(p.y), float(p.z)), (float(q.x), float(q.y), float(q.z), float(q.w))
        )

    class PayloadScanFilterNode(Node):  # type: ignore[misc]
        """Republishes ``/scan`` with the attached payload's returns removed."""

        def __init__(self) -> None:
            super().__init__("openral_nav2_payload_scan_filter")
            self.declare_parameter("input_topic", "/scan")
            self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
            self.declare_parameter("world_state_topic", "/openral/world_state_fast")
            self.declare_parameter("attached_state_timeout_s", 0.5)
            self.declare_parameter("payload_margin_m", 0.0)
            self.declare_parameter("tf_timeout_ms", 50)

            gp = self.get_parameter
            self._margin_m = gp("payload_margin_m").get_parameter_value().double_value
            self._timeout_ns = int(
                gp("attached_state_timeout_s").get_parameter_value().double_value * 1e9
            )
            self._tf_timeout = Duration(
                seconds=gp("tf_timeout_ms").get_parameter_value().integer_value / 1000.0
            )
            self._state: WorldStateStamped | None = None
            self._state_ns: int | None = None
            self._warned_passthrough = False

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            sensor_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            state_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            out_topic = gp("output_topic").get_parameter_value().string_value
            in_topic = gp("input_topic").get_parameter_value().string_value
            self._pub = self.create_publisher(LaserScan, out_topic, sensor_qos)
            self._state_sub = self.create_subscription(
                WorldStateStamped,
                gp("world_state_topic").get_parameter_value().string_value,
                self._on_world_state,
                state_qos,
            )
            self._scan_sub = self.create_subscription(
                LaserScan, in_topic, self._on_scan, sensor_qos
            )
            self.get_logger().info(f"payload_scan_filter: {in_topic} -> {out_topic}")

        def _on_world_state(self, msg: WorldStateStamped) -> None:
            self._state = msg
            self._state_ns = self.get_clock().now().nanoseconds

        def _placements(self, scan_frame: str) -> list[tuple[int, list[float], Any]]:
            state = self._state
            if state is None or self._state_ns is None:
                return []
            if self.get_clock().now().nanoseconds - self._state_ns > self._timeout_ns:
                return []
            out: list[tuple[int, list[float], Any]] = []
            for obj in state.attached_objects:
                link_tf = self._tf_buffer.lookup_transform(
                    scan_frame, obj.attach_link, Time(), timeout=self._tf_timeout
                )
                t, q = link_tf.transform.translation, link_tf.transform.rotation
                scan_from_object = homogeneous_from_quat_xyz(
                    (float(t.x), float(t.y), float(t.z)),
                    (float(q.x), float(q.y), float(q.z), float(q.w)),
                ) @ _pose_matrix(obj.pose_in_link)
                for prim in obj.primitives:
                    out.append(
                        (
                            int(prim.shape_type),
                            [float(d) for d in prim.shape_dimensions],
                            scan_from_object @ _pose_matrix(prim.pose_in_object),
                        )
                    )
            return out

        def _on_scan(self, msg: LaserScan) -> None:
            try:
                placements = self._placements(msg.header.frame_id)
                ranges = (
                    filter_scan_ranges(
                        list(msg.ranges),
                        angle_min=float(msg.angle_min),
                        angle_increment=float(msg.angle_increment),
                        range_min=float(msg.range_min),
                        range_max=float(msg.range_max),
                        placements=placements,
                        margin_m=self._margin_m,
                    )
                    if placements
                    else list(msg.ranges)
                )
            except (TransformException, ValueError) as exc:
                if not self._warned_passthrough:
                    self.get_logger().warning(
                        f"republishing /scan unfiltered: cannot place attached geometry ({exc})"
                    )
                    self._warned_passthrough = True
                self._pub.publish(msg)
                return
            self._warned_passthrough = False
            msg.ranges = [float(r) for r in ranges]
            self._pub.publish(msg)

    rclpy.init(args=args)
    node = PayloadScanFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
