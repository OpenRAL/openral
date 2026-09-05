#!/usr/bin/env python3
"""Drop the carried payload's own returns out of the ``/scan`` Nav2 costmaps read.

Two halves, and Nav2 being **base-only** is why both matter.

*The payload half.* A carried object is not part of Nav2's robot — the
costmaps' footprint is the manifest's bare chassis, and the arm and payload
belong to the 3-D safety kernel. So if the payload's own returns reach the
scan, Nav2 sees an obstacle that moves with the robot: one it can never drive
away from, leaving the base permanently 1.2 s from colliding with the thing it
is holding. That is the ``collision_monitor`` failure the panda_mobile config
already had to disable a polygon over. Removing the returns is what keeps the
payload out of Nav2's world entirely, which is the whole point of base-only.

*The self half.* Same argument, one layer in: a real 2-D lidar sees the robot's
own chassis, and those returns mark the costmap and never clear.

**What actually feeds the costmaps.** Not ``/octomap_binary``. The lidar
profile's ``voxel_layer`` / ``obstacle_layer`` and the ``collision_monitor``
all take ``sensor_msgs/LaserScan`` on ``/scan``
(``config/nav2_panda_mobile.yaml``); the visual profile takes an
``OccupancyGrid`` on ``/map``. So the honest seam for "robot + payload
geometry must not reach the costmap" is the scan topic, and this node sits on
it: ``/scan`` in, ``/scan`` minus the payload out, Nav2 pointed at the output.

**Two things get removed, for two different reasons.**

*The payload*, by its exact collision primitives — the geometry the kernel is
already tracking, placed by the kernel's own composition.

*The robot's own body*, by the chassis footprint polygon. In sim this half is
redundant: ``openral_sim.backends.robocasa.synthesize_laser_scan_2d`` skips
every hit whose body shares the robot's kinematic-tree root and re-casts past
it, so a chassis return never enters ``/scan`` at all. That mechanism is
``mujoco.mj_ray`` with a MuJoCo ``body_rootid`` comparison — it has no
real-hardware counterpart, and this repo has no lidar driver, no lidar launch
file and no manifest field that could carry one. On real hardware a 2-D lidar
mounted on the base *does* see the chassis, the mast and the arm, and an
unfiltered self-return becomes a costmap obstacle that never clears: the base
ends up believing it is surrounded by itself. Until #194 the only knob in the repo
was ``panda_mobile``'s ``range_min_m: 0.55``, a blunt radial cutoff that also
deleted every real obstacle inside 0.55 m in every direction. This node
replaced it with a shaped one, and #194 then lowered that field to the sensor
minimum (0.05 m) — so this filter is now the ONLY self-exclusion on the
hardware path, and its fail-closed behaviour is what the near field rests on.

**The two halves fail in opposite-looking directions, and that is the point.**
For the payload, the dangerous mistake is *not removing* — a payload left in
the costmap only makes Nav2 more cautious, so bad input keeps it. For a
self-return the dangerous mistake is the other one: dropping a real obstacle
because we mistook it for the robot. So the self-filter removes only returns it
can *prove* are the robot, and on a missing TF, an unreadable manifest or a
degenerate polygon it removes **nothing**.

Both statements are the same invariant seen from two sides: **every failure
mode in this node leaves more obstacles in the scan, never fewer.**

**Why the chassis polygon is a proof and the arm's link boxes would not be.**
A return whose endpoint lies inside the chassis outline is either the chassis
itself or an object standing where the chassis already is — which is not a
place an object can be. Nav2 agrees: that polygon is precisely what this
package publishes as *the robot*, ``obstacle_layer``'s
``footprint_clearing_enabled`` (default ``True``, verified against the Jazzy
binary) frees those cells on every update, and ``collision_monitor`` reads the
same polygon. So removing those returns takes away nothing Nav2 could have
acted on — it only stops ``collision_monitor``, which reads the raw scan with
no costmap in between, from braking for the robot's own chassis. The kernel's
per-link OBBs in ``link_collision`` are the opposite case: they are documented
as *conservative* bounds, i.e. deliberate over-approximations, and the air
between a link and its box is air a real obstacle can occupy. Over-bounding is
the right direction for a collision check and the wrong one for deleting sensor
returns, so this node does not use them.

The region is the **bare chassis** polygon, never the payload-grown one the
footprint publisher emits: the convex hull that joins chassis to payload spans
free air in between, and the payload's own primitives already cover the payload
exactly.

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

**Except at startup, where pass-through is not conservative** (#212). Fail-open
reasons about one scan at a time, and for the payload half that is the whole
story — a payload return that reaches the grid is cleared by the next scan's
ray along the same bearing. The *self* half has no such next scan: once the
filter starts working it removes exactly the beam whose ray would have cleared
the cell it let through, so a single unfiltered ring published during the TF
warm-up marks the chassis into the cost grid **permanently**. Measured: 32
cells survived 20 s of all-``inf`` filtered scans and cleared only when real
returns were put on the same bearings. Nav2's own ``footprint_clearing_enabled``
(default ``True``) frees the cells whose *centre* falls inside the published
polygon and is the standing mitigation in the shipped config, but it does not
reach a cell straddling the boundary, and ``collision_monitor`` has no costmap
at all.

So while a self-polygon is configured and ``base_frame <- scan_frame`` has
never yet resolved, this node publishes **nothing**: an observation source that
has not started is strictly better than one that starts by lying. That window
is bounded by ``self_tf_grace_s`` — after it the node reverts to pass-through
and says so loudly, because a permanently blind Nav2 (a mistyped
``base_frame``, say) is the worse failure of the two. The gate arms once: a TF
gap *after* the first successful resolve still fails open, which is the
one-scan-at-a-time case the paragraph above covers.

What this node deliberately does **not** do is make the map self-healing by
writing ``range_max`` for a dropped beam so Nav2 raytraces it clear. That
reverses the fail direction in a way that is worse than the phantom it removes:
Nav2 clears the whole ray out to ``raytrace_max_range`` (3.0 m in the shipped
config), and a bearing on which the chassis returns is a bearing the sensor is
*permanently* occluded on — so the erased cells are ones marked from other
robot poses, which nothing on that bearing can ever re-mark. Measured, in
``tests/integration/test_nav2_scan_filter_live.py``: a real obstacle 0.25 m
past the chassis edge is deleted from the cost grid by one such beam. A dropped
beam stays ``inf``.
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

    Half-plane test: for a counter-clockwise convex polygon a point is inside
    exactly when it is left of (or on) every directed edge. ``margin_m`` offsets
    each edge's half-plane outward by that distance, which mitres the corners
    rather than rounding them — the resulting region is a *superset* of the true
    offset polygon, i.e. it removes slightly more than asked near a corner. That
    is the dangerous direction for a self-filter, which is why the node's
    default margin is zero.

    Convexity and winding are **verified, not assumed**: the caller's polygon
    comes from :func:`~openral_nav2_bringup._footprint_geometry.convex_hull_2d`
    and so is always CCW convex, but a hand-supplied concave outline would make
    the half-plane test claim the concavities are robot. Anything that fails the
    check raises, and the node's self-filter then removes nothing.

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
        margin_m: Containment margin passed to :func:`points_in_primitive`.
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
