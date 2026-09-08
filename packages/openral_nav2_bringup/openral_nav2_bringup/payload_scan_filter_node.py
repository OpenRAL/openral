#!/usr/bin/env python3
"""Drop the payload's and the robot's own returns from ``/scan`` before Nav2's costmaps read it.

Nav2 is base-only (bare-chassis footprint); arm + payload belong to the 3-D safety kernel.
Unfiltered payload returns look like a moving obstacle (the failure `collision_monitor` already
disables a polygon over); unfiltered self-returns mark the costmap and never clear.

Feeds: not ``/octomap_binary``. Lidar profile's ``voxel_layer``/``obstacle_layer``/
``collision_monitor`` take ``LaserScan`` on ``/scan`` (``config/nav2_panda_mobile.yaml``); visual
profile takes ``OccupancyGrid`` on ``/map``. Node: ``/scan`` in, minus payload/self, out.

Payload half: removed by its exact collision primitives (the kernel's own composition). Self half:
removed by the bare-chassis footprint polygon (never the payload-grown hull) — the same polygon
``obstacle_layer.footprint_clearing_enabled`` (default ``True``, verified against the Jazzy binary)
frees each update and ``collision_monitor`` reads; the kernel's per-link OBBs are conservative
over-approximations, unused here. ``footprint_clearing_enabled`` is a second, independent defense,
not a substitute (misses a cell straddling the polygon boundary; ``collision_monitor`` reads the
raw scan with no costmap at all).

Sim: self half is redundant — ``openral_sim.backends.robocasa.synthesize_laser_scan_2d``
(``mujoco.mj_ray`` + ``body_rootid`` comparison) already skips the robot's own kinematic tree; no
real-hardware counterpart exists (no lidar driver, launch file, or manifest field for one). Until
#194 the only real-hw knob was ``range_min_m: 0.55`` (deleted every obstacle inside 0.55 m); #194
lowered it to the sensor minimum (0.05 m), so this node is now the ONLY hardware self-exclusion.

Failure direction: every failure leaves MORE obstacles in the scan, never fewer — missing
attach-link TF, a kernel-rejected primitive, or stale/absent world state (older than
``attached_state_timeout_s``) republishes ``/scan`` unfiltered. Exception (#212): fail-open is not
conservative for the self half at startup — one unfiltered ring during TF warm-up marks the chassis
PERMANENTLY (measured: 32 cells survived 20 s of all-``inf`` filtered scans) — so while a
self-polygon is configured and ``base_frame <- scan_frame`` has never resolved, this node publishes
NOTHING, bounded by ``self_tf_grace_s``; the gate arms once, a later TF gap fails open.

Does not synthesize ``range_max`` for a dropped beam: Nav2 raytraces the whole ray out to
``raytrace_max_range`` (3.0 m shipped), permanently erasing real obstacles marked from other poses
on a permanently occluded bearing (measured in ``tests/integration/test_nav2_scan_filter_live.py``:
a real obstacle 0.25 m past the chassis edge deleted by one such beam).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from openral_nav2_bringup._footprint_geometry import (
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_SPHERE,
    base_footprint_polygon,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_OUTPUT_TOPIC",
    "filter_scan_ranges",
    "main",
    "points_in_convex_polygon",
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


_MIN_POLYGON_VERTICES = 3


def points_in_convex_polygon(
    points_xy: Any,
    polygon: Sequence[tuple[float, float]],
    *,
    margin_m: float = 0.0,
) -> Any:
    """Boolean mask of which ``points_xy`` lie inside a CCW convex polygon.

    Half-plane test: inside iff left of (or on) every directed edge. ``margin_m`` offsets each edge
    outward (mitres corners, a superset of the true offset region); default is zero, the safe
    direction for a self-filter. Convexity/winding is verified, not assumed — the caller's polygon
    comes from ``convex_hull_2d`` (always CCW convex),
    but a bad hand-supplied outline would misclassify concavities as robot; failing the check raises
    and the self-filter removes nothing.

    Args:
        points_xy: ``(N, 2)`` array of points in the polygon's own frame.
        polygon: Counter-clockwise convex vertices, first not repeated at the
            end.
        margin_m: Outward offset applied to every edge, metres. Non-negative.

    Returns:
        ``(N,)`` boolean array, ``True`` where the point is inside or on the
        boundary.

    Raises:
        ValueError: If the polygon has fewer than three vertices, is not
            finite, is not counter-clockwise convex, or the margin is negative.

    Example:
        >>> import numpy as np
        >>> chassis = [(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.35, -0.25)]
        >>> probes = np.array([[0.20, 0.0], [0.60, 0.0]])
        >>> [bool(v) for v in points_in_convex_polygon(probes, chassis)]
        [True, False]
    """
    import numpy as np

    if margin_m < 0.0 or not math.isfinite(margin_m):
        raise ValueError(f"margin_m must be finite and non-negative; got {margin_m}")
    verts = np.asarray([(float(x), float(y)) for x, y in polygon], dtype=np.float64)
    if verts.ndim != 2 or verts.shape[0] < _MIN_POLYGON_VERTICES or verts.shape[1] != 2:
        raise ValueError(f"polygon needs >= 3 (x, y) vertices; got shape {verts.shape}")
    if not np.all(np.isfinite(verts)):
        raise ValueError("polygon has a non-finite vertex")
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must be (N, 2); got shape {pts.shape}")

    edges = np.roll(verts, -1, axis=0) - verts
    lengths = np.linalg.norm(edges, axis=1)
    if not np.all(lengths > 0.0):
        raise ValueError("polygon has a zero-length edge")
    # CCW convex <=> every consecutive edge turns left.
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - (
        edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if not np.all(turns >= 0.0):
        raise ValueError("polygon is not counter-clockwise convex")

    # Signed distance left of each directed edge, for every point at once:
    # cross(edge, point - vertex) / |edge|, positive inside a CCW polygon.
    rel = pts[:, None, :] - verts[None, :, :]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    signed = cross / lengths[None, :]
    return np.all(signed >= -float(margin_m), axis=1)


def filter_scan_ranges(
    ranges: Sequence[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    placements: Sequence[tuple[int, Sequence[float], Any]],
    margin_m: float = 0.0,
    self_polygon: Sequence[tuple[float, float]] | None = None,
    base_from_scan: Any = None,
    self_margin_m: float = 0.0,
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
            primitive, each transform placing it in the **scan** frame. Empty
            disables the payload half.
        margin_m: Containment margin passed to ``points_in_primitive``.
        self_polygon: The robot's **bare chassis** outline, CCW convex, in the
            frame ``base_from_scan`` maps the scan into. ``None`` disables the
            self half, which is what the node passes on any failure to resolve
            it — a self-return the node cannot prove is the robot stays in the
            scan as an obstacle.
        base_from_scan: ``(4, 4)`` homogeneous transform placing the scan frame
            in ``self_polygon``'s frame. Required with ``self_polygon``.
        self_margin_m: Outward offset on the chassis polygon. Defaults to
            ``0.0`` and should stay there: the polygon is already the outline
            Nav2 treats as robot, so every millimetre past it deletes returns
            Nav2 *would* have acted on. It exists for a real lidar with
            characterised range noise, as a deliberate, measured choice.

    Returns:
        A new range list, same length, with removed beams set to ``inf``.

    Raises:
        ValueError: If a placement is malformed, or ``self_polygon`` is given
            without a usable ``base_from_scan`` / is not CCW convex. The caller
            must then drop the affected half rather than guess.

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
    if out.size == 0 or (not placements and self_polygon is None):
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

    if self_polygon is not None:
        base_m = np.asarray(base_from_scan, dtype=np.float64)
        if base_m.shape != (4, 4) or not np.all(np.isfinite(base_m)):
            raise ValueError(
                f"self_polygon needs a finite (4, 4) base_from_scan; got shape {base_m.shape}"
            )
        in_base = pts @ base_m[:3, :3].T + base_m[:3, 3]
        drop |= points_in_convex_polygon(
            in_base[:, :2], self_polygon, margin_m=float(self_margin_m)
        )

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
        """Republishes ``/scan`` with the payload's and the robot's returns removed."""

        def __init__(self) -> None:
            super().__init__("openral_nav2_payload_scan_filter")
            self.declare_parameter("input_topic", "/scan")
            self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
            self.declare_parameter("world_state_topic", "/openral/world_state_fast")
            self.declare_parameter("attached_state_timeout_s", 0.5)
            self.declare_parameter("payload_margin_m", 0.0)
            self.declare_parameter("tf_timeout_ms", 50)
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("base_frame", "")
            self.declare_parameter("self_margin_m", 0.0)
            self.declare_parameter("circle_samples", 12)
            self.declare_parameter("self_tf_grace_s", 5.0)

            gp = self.get_parameter
            self._margin_m = gp("payload_margin_m").get_parameter_value().double_value
            self._self_margin_m = gp("self_margin_m").get_parameter_value().double_value
            self._timeout_ns = int(
                gp("attached_state_timeout_s").get_parameter_value().double_value * 1e9
            )
            self._tf_timeout = Duration(
                seconds=gp("tf_timeout_ms").get_parameter_value().integer_value / 1000.0
            )
            self._state: WorldStateStamped | None = None
            self._state_ns: int | None = None
            self._warned_passthrough = False
            self._warned_no_self_filter = False

            # Startup readiness gate (#212). Until `base_frame <- scan_frame`
            # resolves once, a scan published here can mark the chassis into
            # the cost grid with nothing able to clear it again, so nothing is
            # published at all. Bounded, because a node that never publishes is
            # a Nav2 that never sees the room.
            self._self_tf_grace_ns = int(
                gp("self_tf_grace_s").get_parameter_value().double_value * 1e9
            )
            self._self_tf_ready = False
            self._first_scan_ns: int | None = None
            self._warned_grace_expired = False

            # The self-filter's region is the manifest's BARE chassis outline.
            # Without a manifest there is no outline to prove a return is the
            # robot, so the self half simply does not run — the payload half
            # still does. Inventing a shape here would delete returns off a
            # made-up robot.
            self._self_polygon: list[tuple[float, float]] | None = None
            self._base_frame = gp("base_frame").get_parameter_value().string_value
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            if robot_yaml:
                from openral_core import RobotDescription

                description = RobotDescription.from_yaml(robot_yaml)
                self._self_polygon = base_footprint_polygon(
                    description,
                    circle_samples=gp("circle_samples").get_parameter_value().integer_value,
                )
                self._base_frame = self._base_frame or description.base_frame
            else:
                self.get_logger().warning(
                    "no `robot_yaml`: the robot's OWN returns are not filtered. In sim the "
                    "MuJoCo ray-cast already skips the robot's kinematic tree, so this is "
                    "harmless there; on real hardware every chassis/mast/arm return reaches "
                    "the costmap and the collision monitor as a permanent obstacle."
                )

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
            self_desc = (
                f"self-filter on {len(self._self_polygon)}-vertex chassis in "
                f"{self._base_frame} (margin {self._self_margin_m * 1000:.0f} mm)"
                if self._self_polygon is not None
                else "self-filter OFF"
            )
            self.get_logger().info(f"payload_scan_filter: {in_topic} -> {out_topic}, {self_desc}")

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

        def _base_from_scan(self, scan_frame: str) -> Any:
            """``(4, 4)`` placing the scan frame in ``base_frame``.

            Raises:
                TransformException: the chain is unavailable, in which case the
                    caller runs no self-filter at all.
            """
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, scan_frame, Time(), timeout=self._tf_timeout
            )
            t, q = tf.transform.translation, tf.transform.rotation
            return homogeneous_from_quat_xyz(
                (float(t.x), float(t.y), float(t.z)),
                (float(q.x), float(q.y), float(q.z), float(q.w)),
            )

        def _gate_allows_publishing(self, self_filter_live: bool) -> bool:
            """Whether this scan may go out at all — the #212 startup gate.

            Returns ``True`` unless a self-polygon is configured, its TF has
            never yet resolved, and the grace window is still open. See the
            module docstring for why the self half cannot fail open the way the
            payload half can.
            """
            if self_filter_live:
                self._self_tf_ready = True
                return True
            if self._self_polygon is None or self._self_tf_ready:
                return True

            now = self.get_clock().now().nanoseconds
            if self._first_scan_ns is None:
                self._first_scan_ns = now
            if now - self._first_scan_ns < self._self_tf_grace_ns:
                return False
            if not self._warned_grace_expired:
                self._warned_grace_expired = True
                self.get_logger().error(
                    f"self-filter TF never resolved within {self._self_tf_grace_ns / 1e9:.1f}s; "
                    f"republishing /scan UNFILTERED so Nav2 is not blind. Every chassis return "
                    f"now reaching the cost grid is permanent (see #212) — check `base_frame` "
                    f"({self._base_frame!r}) and the TF chain to the scan frame."
                )
            return True

        def _on_scan(self, msg: LaserScan) -> None:
            # The two halves are resolved independently, because their unsafe
            # directions are opposite. A payload we cannot place must stay in
            # the scan (dropping it would make Nav2 less cautious); a robot
            # return we cannot prove is the robot must also stay in the scan
            # (dropping it could delete a real obstacle). Both failures
            # therefore ADD obstacles, and neither is allowed to suppress the
            # other half's filtering.
            scan_frame = msg.header.frame_id
            try:
                placements: list[tuple[int, list[float], Any]] = self._placements(scan_frame)
                self._warned_passthrough = False
            except (TransformException, ValueError) as exc:
                placements = []
                if not self._warned_passthrough:
                    self._warned_passthrough = True
                    self.get_logger().warning(
                        f"payload returns left in /scan: cannot place attached geometry ({exc})"
                    )

            self_polygon: list[tuple[float, float]] | None = None
            base_from_scan: Any = None
            if self._self_polygon is not None:
                try:
                    base_from_scan = self._base_from_scan(scan_frame)
                    self_polygon = self._self_polygon
                    self._warned_no_self_filter = False
                except (TransformException, ValueError) as exc:
                    if not self._warned_no_self_filter:
                        self._warned_no_self_filter = True
                        self.get_logger().warning(
                            f"self-returns left in /scan: cannot place {scan_frame!r} in "
                            f"{self._base_frame!r} ({exc})"
                        )

            if not self._gate_allows_publishing(self_polygon is not None):
                return

            if not placements and self_polygon is None:
                self._pub.publish(msg)
                return

            try:
                ranges = filter_scan_ranges(
                    list(msg.ranges),
                    angle_min=float(msg.angle_min),
                    angle_increment=float(msg.angle_increment),
                    range_min=float(msg.range_min),
                    range_max=float(msg.range_max),
                    placements=placements,
                    margin_m=self._margin_m,
                    self_polygon=self_polygon,
                    base_from_scan=base_from_scan,
                    self_margin_m=self._self_margin_m,
                )
            except ValueError as exc:
                # Neither half could be applied safely; the whole scan passes
                # through, which is the more-obstacles direction for both.
                if not self._warned_passthrough:
                    self._warned_passthrough = True
                    self.get_logger().warning(f"republishing /scan unfiltered: {exc}")
                self._pub.publish(msg)
                return
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
