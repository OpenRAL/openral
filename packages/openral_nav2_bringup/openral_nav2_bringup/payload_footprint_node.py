#!/usr/bin/env python3
"""Publish Nav2's robot footprint, grown to include whatever the robot is carrying.

Nav2 models the base as one 2-D polygon. That polygon is the *only* thing
standing between a carried object and the furniture the base drives past: the
C++ safety kernel checks arm + payload capsules against
``/openral/world_voxels``, but it does not gate ``/cmd_vel`` at all (see
"The Nav2 <-> safety-kernel boundary" in this package's README). A baguette
sticking 30 cm past the chassis is, to Nav2, not there.

Nav2 supports exactly this case: ``nav2_costmap_2d::Costmap2DROS`` subscribes a
``geometry_msgs/Polygon`` on its own ``footprint`` topic and calls
``setRobotFootprintPolygon`` on every message, which re-derives the layered
costmap's inscribed / circumscribed radii and re-publishes the oriented result
on ``published_footprint`` (which is in turn what ``nav2_collision_monitor``,
``nav2_behaviors`` and ``opennav_docking`` read). Verified against the Jazzy
binaries rather than from memory — ``ros2 node info`` on a live
``nav2_costmap_2d`` shows ``footprint: geometry_msgs/msg/Polygon`` inbound and
``published_footprint: geometry_msgs/msg/PolygonStamped`` outbound. The
subscription exists whether the costmap was configured with ``robot_radius``
or with a ``footprint`` polygon, but only a polygon-configured costmap can
*express* an asymmetric payload, so ``config/nav2_panda_mobile.yaml`` declares
one.

This node is that publisher. It is the footprint sibling of
``openral_octomap_bridge``: same input (the attachment set on
``/openral/world_state_fast`` — the kernel's own source, so kernel, occupancy
grid and footprint can never act on different attachment sets), same job
(lower one World State fact into the representation one consumer needs), and
it likewise derives everything from the wire on every message rather than
latching state.

Geometry, in one line: the published polygon is the 2-D convex hull of the
manifest's base footprint together with the ground projection of every
attached collision primitive, each placed by
``TF(base_frame <- attach_link) . pose_in_link . pose_in_object`` — the same
composition the kernel and the octomap bridge use.

Conservative directions, all deliberate:

* A box projects **exactly** (the projection of a convex polytope is the hull
  of its projected vertices). Spheres and capsules project onto *circumscribed*
  N-gons, so their polygons contain the true circle/stadium rather than
  inscribing it.
* Height is ignored. A primitive held above the obstacle band still enters the
  footprint, because Nav2's world is 2-D and the alternative is deciding on
  the robot's behalf which carried geometry cannot hit anything.
* Anything that stops the node from placing a payload — missing
  ``base_frame <- attach_link`` TF, a primitive shape the kernel itself would
  reject, world state gone stale — makes it **hold the last polygon** rather
  than fall back to the bare chassis. Shrinking the footprint is the one
  direction that can let the base drive its payload into something, so it only
  ever happens on a fresh, well-formed message that says the payload is gone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_FOOTPRINT_TOPICS",
    "SHAPE_BOX",
    "SHAPE_CAPSULE",
    "SHAPE_SPHERE",
    "base_footprint_polygon",
    "convex_hull_2d",
    "footprint_with_payload",
    "main",
    "primitive_ground_points",
]

# Mirrors openral_msgs/AttachedCollisionPrimitive. Duplicated as plain ints so
# the pure geometry below imports without the colcon message overlay (the unit
# tests run outside it); the ROS half asserts the constants match at runtime.
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_BOX = 3

#: The costmap topics Nav2 subscribes footprints on, for the node names
#: ``nav2_bringup`` gives its costmaps. ``Costmap2DROS`` puts each costmap in a
#: namespace equal to its own name, and the subscription is the *relative*
#: topic ``footprint``.
DEFAULT_FOOTPRINT_TOPICS = ("/local_costmap/footprint", "/global_costmap/footprint")

_MIN_POLYGON_VERTICES = 3


def convex_hull_2d(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull of ``points`` (Andrew's monotone chain).

    Collinear points are dropped, so the result is a minimal vertex set. Fewer
    than three distinct points is a degenerate footprint and raises rather than
    handing Nav2 a polygon it will reject at ``setRobotFootprintPolygon``.

    Args:
        points: ``(x, y)`` pairs in metres. Duplicates are fine.

    Returns:
        Hull vertices, counter-clockwise, first vertex not repeated at the end.

    Raises:
        ValueError: If fewer than three non-collinear finite points are given.

    Example:
        >>> convex_hull_2d([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)])
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"convex_hull_2d got a non-finite vertex ({x}, {y})")
    if len(pts) < _MIN_POLYGON_VERTICES:
        raise ValueError(f"convex_hull_2d needs >= 3 distinct points; got {len(pts)}")

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0.0:
                out.pop()
            out.append(p)
        return out

    lower = _half(pts)
    upper = _half(pts[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < _MIN_POLYGON_VERTICES:
        raise ValueError(f"convex_hull_2d: {len(pts)} points are collinear; no polygon")
    return hull


def _circumscribed_circle_points(
    cx: float, cy: float, radius: float, *, circle_samples: int
) -> list[tuple[float, float]]:
    """N-gon vertices whose *inscribed* circle is the requested one.

    Sampling the circle itself would inscribe the polygon in it and lose up to
    ``radius * (1 - cos(pi/n))`` of the payload. Scaling by ``1/cos(pi/n)``
    puts the circle inside the polygon instead, which is the safe direction.
    """
    n = max(int(circle_samples), 4)
    scale = 1.0 / math.cos(math.pi / n)
    r = float(radius) * scale
    return [
        (cx + r * math.cos(2.0 * math.pi * i / n), cy + r * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]


def primitive_ground_points(
    shape_type: int,
    shape_dimensions: Sequence[float],
    transform: Any,
    *,
    margin_m: float = 0.0,
    circle_samples: int = 12,
) -> list[tuple[float, float]]:
    """Ground-plane support points of one attached collision primitive.

    The convex hull of the returned points contains the primitive's vertical
    projection. Shape conventions are the kernel's
    (``openral_msgs/AttachedCollisionPrimitive``): sphere ``[radius]``, capsule
    ``[radius, central_segment_length]`` with the segment along the primitive's
    local +Z, box ``[half_x, half_y, half_z]``.

    Args:
        shape_type: One of :data:`SHAPE_SPHERE` / :data:`SHAPE_CAPSULE` /
            :data:`SHAPE_BOX`.
        shape_dimensions: Shape dimensions in metres, per the convention above.
        transform: ``(4, 4)`` homogeneous transform placing the primitive frame
            in the frame the footprint is expressed in (``base_frame``).
        margin_m: Extra radius / half-extent added to the primitive before
            projection, for pose uncertainty. Non-negative.
        circle_samples: N-gon resolution for the round shapes.

    Returns:
        ``(x, y)`` support points in the footprint frame.

    Raises:
        ValueError: On an unknown shape tag, too few dimensions, a
            non-positive dimension, or a negative margin — the same inputs the
            kernel fails closed on.
    """
    import numpy as np

    if margin_m < 0.0 or not math.isfinite(margin_m):
        raise ValueError(f"margin_m must be finite and non-negative; got {margin_m}")
    dims = [float(d) for d in shape_dimensions]
    m = np.asarray(transform, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"transform must be a finite (4, 4) matrix; got shape {m.shape}")

    def _to_world(local: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
        pts = np.asarray(local, dtype=np.float64)
        world = pts @ m[:3, :3].T + m[:3, 3]
        return [(float(p[0]), float(p[1])) for p in world]

    origin_xy = (float(m[0, 3]), float(m[1, 3]))

    if shape_type == SHAPE_SPHERE:
        if len(dims) < 1 or not (dims[0] > 0.0 and math.isfinite(dims[0])):
            raise ValueError(f"sphere needs one positive radius; got {dims}")
        return _circumscribed_circle_points(
            origin_xy[0], origin_xy[1], dims[0] + margin_m, circle_samples=circle_samples
        )

    if shape_type == SHAPE_CAPSULE:
        capsule_dims = 2
        if len(dims) < capsule_dims or not all(d > 0.0 and math.isfinite(d) for d in dims[:2]):
            raise ValueError(f"capsule needs positive [radius, length]; got {dims}")
        radius, half_length = dims[0] + margin_m, 0.5 * dims[1]
        ends = _to_world([(0.0, 0.0, -half_length), (0.0, 0.0, half_length)])
        out: list[tuple[float, float]] = []
        for ex, ey in ends:
            out.extend(_circumscribed_circle_points(ex, ey, radius, circle_samples=circle_samples))
        return out

    if shape_type == SHAPE_BOX:
        box_dims = 3
        if len(dims) < box_dims or not all(d > 0.0 and math.isfinite(d) for d in dims[:3]):
            raise ValueError(f"box needs three positive half-extents; got {dims}")
        hx, hy, hz = (d + margin_m for d in dims[:3])
        corners = [
            (sx * hx, sy * hy, sz * hz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        return _to_world(corners)

    raise ValueError(f"unknown attached-primitive shape_type {shape_type!r}")


def base_footprint_polygon(
    description: Any, *, circle_samples: int = 12
) -> list[tuple[float, float]]:
    """The robot's nominal Nav2 footprint polygon, from its manifest.

    Prefers ``RobotDescription.footprint_polygon`` (the measured outline) and
    falls back to a circumscribed N-gon around ``footprint_radius``. A manifest
    with neither declares no mobile base and has no Nav2 footprint to publish.

    Args:
        description: A loaded ``openral_core.RobotDescription``.
        circle_samples: N-gon resolution used for the ``footprint_radius``
            fallback.

    Returns:
        Counter-clockwise ``(x, y)`` vertices in the robot's ``base_frame``.

    Raises:
        ValueError: If the manifest declares neither a footprint polygon nor a
            footprint radius.
    """
    poly = getattr(description, "footprint_polygon", None)
    if poly:
        return convex_hull_2d([(float(x), float(y)) for x, y in poly])
    radius = getattr(description, "footprint_radius", None)
    if radius is not None:
        return convex_hull_2d(
            _circumscribed_circle_points(0.0, 0.0, float(radius), circle_samples=circle_samples)
        )
    raise ValueError(
        f"robot {getattr(description, 'robot_id', '?')!r} declares neither footprint_polygon "
        "nor footprint_radius; it has no Nav2 footprint to publish"
    )


def footprint_with_payload(
    base_polygon: Sequence[tuple[float, float]],
    placements: Sequence[tuple[int, Sequence[float], Any]],
    *,
    margin_m: float = 0.0,
    circle_samples: int = 12,
) -> list[tuple[float, float]]:
    """Union the base footprint with the ground projection of carried geometry.

    Args:
        base_polygon: The nominal footprint, ``(x, y)`` in ``base_frame``.
        placements: ``(shape_type, shape_dimensions, transform)`` per attached
            collision primitive, each transform placing that primitive in
            ``base_frame``.
        margin_m: Pose-uncertainty margin added to each payload primitive (the
            base polygon is a measured chassis and is never inflated).
        circle_samples: N-gon resolution for round primitives.

    Returns:
        The counter-clockwise convex hull of the union.

    Raises:
        ValueError: If the base polygon is degenerate or any placement is
            malformed — the caller must not narrow the footprint on bad input.

    Example:
        >>> import numpy as np
        >>> base = [(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.35, -0.25)]
        >>> reach = np.eye(4)
        >>> reach[0, 3] = 0.60
        >>> hull = footprint_with_payload(base, [(3, (0.10, 0.03, 0.03), reach)])
        >>> round(max(x for x, _ in hull), 3)
        0.7
    """
    points = [(float(x), float(y)) for x, y in base_polygon]
    if len(points) < _MIN_POLYGON_VERTICES:
        raise ValueError(f"base_polygon needs >= 3 vertices; got {len(points)}")
    for shape_type, dims, transform in placements:
        points.extend(
            primitive_ground_points(
                int(shape_type),
                dims,
                transform,
                margin_m=margin_m,
                circle_samples=circle_samples,
            )
        )
    return convex_hull_2d(points)


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_nav2_bringup payload_footprint_node.py``.

    The node class is declared inline here, like its sibling bringup nodes, so
    the pure geometry above imports without ``rclpy`` or the colcon overlay.
    """
    import rclpy
    from geometry_msgs.msg import Point32, Polygon
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_msgs.msg import AttachedCollisionPrimitive, WorldStateStamped
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    # The pure half above hardcodes the wire tags so it imports without the
    # colcon overlay. If the IDL ever renumbers them, fail here rather than
    # publish a footprint that projects the wrong shape.
    if (
        AttachedCollisionPrimitive.SHAPE_SPHERE,
        AttachedCollisionPrimitive.SHAPE_CAPSULE,
        AttachedCollisionPrimitive.SHAPE_BOX,
    ) != (SHAPE_SPHERE, SHAPE_CAPSULE, SHAPE_BOX):
        raise RuntimeError("AttachedCollisionPrimitive shape tags no longer match this node")

    def _pose_matrix(pose: Any) -> Any:
        q = pose.orientation
        p = pose.position
        return homogeneous_from_quat_xyz(
            (float(p.x), float(p.y), float(p.z)), (float(q.x), float(q.y), float(q.z), float(q.w))
        )

    class PayloadFootprintNode(Node):  # type: ignore[misc]
        """Publishes the payload-inclusive Nav2 footprint polygon."""

        def __init__(self) -> None:
            super().__init__("openral_nav2_payload_footprint")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("base_frame", "")
            self.declare_parameter("world_state_topic", "/openral/world_state_fast")
            self.declare_parameter("footprint_topics", list(DEFAULT_FOOTPRINT_TOPICS))
            self.declare_parameter("publish_rate_hz", 2.0)
            self.declare_parameter("attached_state_timeout_s", 0.5)
            self.declare_parameter("payload_margin_m", 0.0)
            self.declare_parameter("circle_samples", 12)
            self.declare_parameter("tf_timeout_ms", 50)

            gp = self.get_parameter
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            if not robot_yaml:
                raise RuntimeError(
                    "payload_footprint_node needs `robot_yaml`: the nominal footprint is the "
                    "robot manifest's, and guessing one would put a made-up robot outline on "
                    "Nav2's collision surface."
                )
            from openral_core import RobotDescription

            description = RobotDescription.from_yaml(robot_yaml)
            self._base_polygon = base_footprint_polygon(
                description, circle_samples=gp("circle_samples").get_parameter_value().integer_value
            )
            self._base_frame = (
                gp("base_frame").get_parameter_value().string_value or description.base_frame
            )
            self._margin_m = gp("payload_margin_m").get_parameter_value().double_value
            self._circle_samples = gp("circle_samples").get_parameter_value().integer_value
            self._timeout_ns = int(
                gp("attached_state_timeout_s").get_parameter_value().double_value * 1e9
            )
            self._tf_timeout = Duration(
                seconds=gp("tf_timeout_ms").get_parameter_value().integer_value / 1000.0
            )
            self._polygon: list[tuple[float, float]] = list(self._base_polygon)
            self._carrying = 0
            self._last_state_ns: int | None = None
            self._warned_hold = False

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            control_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            topics = list(gp("footprint_topics").get_parameter_value().string_array_value)
            self._pubs = [self.create_publisher(Polygon, t, control_qos) for t in topics]
            self._state_sub = self.create_subscription(
                WorldStateStamped,
                gp("world_state_topic").get_parameter_value().string_value,
                self._on_world_state,
                control_qos,
            )
            rate = max(gp("publish_rate_hz").get_parameter_value().double_value, 0.1)
            self._timer = self.create_timer(1.0 / rate, self._publish)
            self.get_logger().info(
                f"payload_footprint: base polygon {len(self._base_polygon)} vertices in "
                f"{self._base_frame}, publishing on {topics} at {rate:.1f} Hz "
                f"(margin {self._margin_m * 1000:.0f} mm)"
            )

        def _placements(self, msg: WorldStateStamped) -> list[tuple[int, list[float], Any]]:
            """Place every attached primitive in ``base_frame``.

            Raises:
                TransformException: attach-link TF unavailable.
                ValueError: a primitive the kernel would itself reject.
            """
            out: list[tuple[int, list[float], Any]] = []
            for obj in msg.attached_objects:
                link_tf = self._tf_buffer.lookup_transform(
                    self._base_frame, obj.attach_link, Time(), timeout=self._tf_timeout
                )
                t, q = link_tf.transform.translation, link_tf.transform.rotation
                base_from_link = homogeneous_from_quat_xyz(
                    (float(t.x), float(t.y), float(t.z)),
                    (float(q.x), float(q.y), float(q.z), float(q.w)),
                )
                base_from_object = base_from_link @ _pose_matrix(obj.pose_in_link)
                for prim in obj.primitives:
                    placement = base_from_object @ _pose_matrix(prim.pose_in_object)
                    out.append(
                        (int(prim.shape_type), [float(d) for d in prim.shape_dimensions], placement)
                    )
            return out

        def _on_world_state(self, msg: WorldStateStamped) -> None:
            try:
                placements = self._placements(msg)
                polygon = footprint_with_payload(
                    self._base_polygon,
                    placements,
                    margin_m=self._margin_m,
                    circle_samples=self._circle_samples,
                )
            except (TransformException, ValueError) as exc:
                # Refuse to narrow. A payload we cannot place is a payload we
                # must keep assuming is out there.
                if not self._warned_hold:
                    self.get_logger().warning(
                        f"holding the last footprint: cannot place attached geometry ({exc})"
                    )
                    self._warned_hold = True
                return
            self._warned_hold = False
            self._last_state_ns = self.get_clock().now().nanoseconds
            if len(placements) != self._carrying or polygon != self._polygon:
                self.get_logger().info(
                    f"footprint: {len(polygon)} vertices, {len(placements)} attached "
                    f"primitive(s), reach {max(math.hypot(x, y) for x, y in polygon):.3f} m"
                )
            self._carrying = len(placements)
            self._polygon = polygon

        def _publish(self) -> None:
            if (
                self._last_state_ns is not None
                and self._carrying
                and self.get_clock().now().nanoseconds - self._last_state_ns > self._timeout_ns
                and not self._warned_hold
            ):
                self.get_logger().warning(
                    "world state is stale while carrying: holding the expanded footprint"
                )
                self._warned_hold = True
            msg = Polygon()
            msg.points = [Point32(x=float(x), y=float(y), z=0.0) for x, y in self._polygon]
            for pub in self._pubs:
                pub.publish(msg)

    rclpy.init(args=args)
    node = PayloadFootprintNode()
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
