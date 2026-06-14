"""ADR-0025 — rclpy → OTLP bridge for slam_toolbox ``/map`` updates.

The OpenRAL dashboard is OTLP-only — it never subscribes to ROS topics
directly. This module ships :class:`SlamMapBridge`, a small
``rclpy.node.Node`` that subscribes to ``nav_msgs/OccupancyGrid`` (the
output of slam_toolbox once the Reasoner has driven it through
``LifecycleTransitionTool(node="openral_slam_toolbox", transition=
"activate")``), throttles to 1 Hz, rasterises the occupancy grid into
a PNG, base64-encodes it, and emits one ``slam.occupancy_grid`` OTel
span carrying the metadata + the PNG as attributes.

The dashboard's store has a matching handler that populates
``_topics["slam"]`` from the span (see
``python/observability/src/openral_observability/dashboard/store.py``).

Composed into the existing ``RskillRunnerNode`` via
``packages/openral_rskill_ros/openral_rskill_ros/compose.py`` so it
shares the runner's rclpy executor; constructing it manually outside
of compose is supported for tests.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    # `rclpy` / `nav_msgs` are not importable without a sourced ROS 2
    # workspace; deferring at type-check time keeps the module lightweight.
    pass

log = structlog.get_logger(__name__)

__all__ = [
    "SLAM_MAP_TOPIC_DEFAULT",
    "SlamMapBridge",
    "encode_occupancy_grid_png",
    "robot_pose_from_transform",
    "yaw_from_quaternion_xyzw",
]


SLAM_MAP_TOPIC_DEFAULT = "/map"
"""Default ``nav_msgs/OccupancyGrid`` topic slam_toolbox publishes on."""

# Throttle so a busy SLAM run doesn't flood the OTLP pipeline (the
# dashboard re-renders on every span; >1 Hz would just consume CPU
# without operator value).
_DEFAULT_PUBLISH_INTERVAL_S = 1.0

# Occupancy values per nav_msgs/OccupancyGrid:
#   -1 = unknown, 0 = free, 100 = occupied. Anything between is the
# probability the cell is occupied. We render to 8-bit greyscale via:
#   unknown → 128 (mid-grey),
#   free    → 255 (white),
#   occupied → 0 (black),
#   in-between scaled linearly between black and white.
_UNKNOWN_GREY = 128

# nav_msgs/OccupancyGrid occupancy probability bounds (0..100).
_OCC_FREE_THRESHOLD = 0
_OCC_OCCUPIED_THRESHOLD = 100

# How many oversize-grid warnings to emit verbosely before falling back
# to a 1-in-100 rate to keep log noise bounded.
_OVERSIZE_WARN_BURST = 3
_OVERSIZE_WARN_RATE = 100


def encode_occupancy_grid_png(
    *,
    width: int,
    height: int,
    data: list[int],
) -> str:
    """Render a ``nav_msgs/OccupancyGrid`` ``data`` array as a base64 PNG.

    Pure function — used directly by tests against synthetic grids so
    the dashboard contract can be exercised without spinning up
    slam_toolbox.

    Args:
        width: Cells across (``OccupancyGrid.info.width``).
        height: Cells down (``OccupancyGrid.info.height``).
        data: Row-major int array of length ``width * height``. Values
            are ``-1`` for unknown, ``0..100`` for occupancy probability
            (slam_toolbox convention).

    Returns:
        Base64-encoded PNG bytes (UTF-8 string), ready to drop into a
        ``<img src="data:image/png;base64,...">`` tag.

    Raises:
        ValueError: If ``len(data) != width * height``.
    """
    if len(data) != width * height:
        raise ValueError(
            f"encode_occupancy_grid_png: data len={len(data)} but width*height={width * height}"
        )

    # `Pillow` is already an OpenRAL dep (camera thumbnail rendering).
    from PIL import Image  # noqa: PLC0415

    pixels = bytearray(width * height)
    for i, v in enumerate(data):
        if v < 0:
            pixels[i] = _UNKNOWN_GREY
        elif v >= _OCC_OCCUPIED_THRESHOLD:
            pixels[i] = 0
        elif v <= _OCC_FREE_THRESHOLD:
            pixels[i] = 255
        else:
            # Linear ramp: 0 -> 255 (white), 100 -> 0 (black).
            pixels[i] = int(255 - (v / 100.0) * 255)

    # nav_msgs/OccupancyGrid stores row-major; the conventional image
    # representation has y flipped (origin at bottom-left for the grid,
    # top-left for PIL). Flip vertically so map north points up in the
    # dashboard card.
    img = Image.frombytes("L", (width, height), bytes(pixels)).transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw (rad, CCW about +Z) from a quaternion ``(x, y, z, w)``.

    Standard ZYX extraction reduced to the yaw term; the robot base is
    planar so roll/pitch are ignored.

    Args:
        x: Quaternion x component.
        y: Quaternion y component.
        z: Quaternion z component.
        w: Quaternion w (scalar) component.

    Returns:
        Yaw in radians in ``[-pi, pi]``.

    Example:
        >>> round(yaw_from_quaternion_xyzw(0.0, 0.0, 0.0, 1.0), 6)
        0.0
    """
    import math  # noqa: PLC0415

    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def robot_pose_from_transform(
    *,
    translation_xyz: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Planar ``(x, y, yaw)`` from a tf2 transform's translation + rotation.

    Args:
        translation_xyz: ``(x, y, z)`` translation in metres; ``z`` ignored.
        rotation_xyzw: Quaternion ``(x, y, z, w)``.

    Returns:
        ``(x_m, y_m, yaw_rad)`` — the base pose in the transform's target
        frame (the map frame at the call site).

    Example:
        >>> import math
        >>> x, y, yaw = robot_pose_from_transform(
        ...     translation_xyz=(1.5, -2.25, 0.0),
        ...     rotation_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        ... )
        >>> round(x, 2), round(y, 2), round(yaw, 4)
        (1.5, -2.25, 1.5708)
    """
    return (
        float(translation_xyz[0]),
        float(translation_xyz[1]),
        yaw_from_quaternion_xyzw(*rotation_xyzw),
    )


class SlamMapBridge:
    """rclpy → OTLP bridge for slam_toolbox ``/map``.

    Constructed against an existing ``rclpy.node.Node`` (typically the
    host ``RskillRunnerNode`` so the subscription shares the runner's
    executor). On each callback the bridge:

    1. Skips if less than ``publish_interval_s`` seconds have passed
       since the last emit (1 Hz default — matches slam_toolbox's
       canonical ``map_update_interval``).
    2. Rasterises the occupancy grid to a PNG via
       :func:`encode_occupancy_grid_png`.
    3. Emits a single ``slam.occupancy_grid`` OTel span carrying the
       metadata + the PNG as attributes (see store handler in
       ``openral_observability.dashboard.store``).

    Args:
        node: Host ``rclpy.node.Node``. The subscription is created on
            this node; destroy_subscription on
            :meth:`destroy` releases it.
        topic: Topic to subscribe to. Defaults to
            :data:`SLAM_MAP_TOPIC_DEFAULT` (``/map``).
        base_frame: TF2 child frame whose pose in the map frame is
            looked up on each /map callback and emitted as
            ``openral.slam.robot_x/y/yaw`` span attributes. Defaults
            to ``"base_link"``.
        footprint_radius_m: If provided, emitted as
            ``openral.slam.footprint_radius_m`` so the dashboard can
            draw the robot circle. ``None`` omits the attribute.
        footprint_polygon: Base-frame XY vertices in metres as a list of
            ``(x, y)`` tuples describing the robot's polygon footprint.
            Flattened to ``[x0, y0, x1, y1, …]`` and emitted as
            ``openral.slam.footprint_polygon_xy`` so the dashboard can
            draw the exact hull on the occupancy grid. ``None`` falls
            back to the ``footprint_radius_m`` circle (or no marker).
        source_node_name: Identifier for the emitting upstream node;
            surfaces on the dashboard card so deployments running
            multiple SLAM instances can disambiguate.
        publish_interval_s: Minimum wall-clock interval between
            emitted spans. Defaults to 1 s.
        max_cells: Refuse to encode grids larger than this. Defaults
            to 4 000 000 (a 2000x2000 map at 5 cm/cell = 100 m x 100 m
            covering area; bigger maps would explode the OTLP
            payload).
    """

    def __init__(
        self,
        node: Any,  # noqa: ANN401  # reason: rclpy.node.Node not importable without a sourced ROS 2 workspace
        *,
        topic: str = SLAM_MAP_TOPIC_DEFAULT,
        base_frame: str = "base_link",
        footprint_radius_m: float | None = None,
        footprint_polygon: list[tuple[float, float]] | None = None,
        source_node_name: str = "openral_slam_toolbox",
        publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
        max_cells: int = 4_000_000,
    ) -> None:
        """Subscribe to ``topic`` and prepare the OTel tracer."""
        from nav_msgs.msg import (  # noqa: PLC0415
            OccupancyGrid,  # type: ignore[import-untyped,unused-ignore]
        )
        from opentelemetry import trace  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        # slam_toolbox publishes /map with TRANSIENT_LOCAL durability so
        # late-joining subscribers get the latest map without having to
        # wait for the next update; mirror its QoS here.
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self._node = node
        self._topic = topic
        self._source_node_name = source_node_name
        self._publish_interval_s = publish_interval_s
        self._max_cells = max_cells
        self._last_publish_ts_s: float = 0.0
        self._tracer = trace.get_tracer("openral.slam_bridge")
        self._dropped_oversize = 0
        self._base_frame = base_frame
        self._footprint_radius_m = footprint_radius_m
        # Flatten the base-frame footprint to a homogeneous float list
        # [x0, y0, x1, y1, ...] for the OTel span attribute (OTel attrs are
        # scalars or homogeneous sequences). None when the robot declares no
        # polygon -> dashboard falls back to the footprint_radius circle.
        self._footprint_polygon_xy: list[float] | None = (
            [float(c) for pt in footprint_polygon for c in pt]
            if footprint_polygon is not None
            else None
        )
        # ADR-0025 — tf2 lookup for map→base_frame so the dashboard can
        # draw the robot on the occupancy grid. Lazy: built on first
        # /map callback (mirrors ADR-0027 in rskill_runner_node). Stays
        # None on non-ROS unit-test paths where tf2_ros isn't importable.
        self._tf_buffer: Any = None
        self._tf_listener: Any = None
        self._tf_init_attempted = False
        self._tf_warned = False

        self._subscription = node.create_subscription(
            OccupancyGrid,
            topic,
            self._on_map,
            map_qos,
        )

    def destroy(self) -> None:
        """Release the ROS subscription + tf2 listener. Safe to call multiple times."""
        if self._subscription is None:
            return
        try:
            self._node.destroy_subscription(self._subscription)
        finally:
            self._subscription = None
            # TransformListener owns a /tf + /tf_static subscription;
            # dropping the references is enough for GC to tear it down.
            self._tf_listener = None
            self._tf_buffer = None
            self._tf_init_attempted = False
            self._tf_warned = False

    def _ensure_tf(self) -> None:
        """Build the tf2 buffer + listener once. No-op without tf2_ros."""
        if self._tf_init_attempted:
            return
        self._tf_init_attempted = True
        try:
            import tf2_ros  # type: ignore[import-untyped,unused-ignore]  # noqa: PLC0415
        except ImportError:
            return
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

    def _lookup_robot_pose(self, map_frame: str) -> tuple[float, float, float] | None:
        """Return ``(x, y, yaw)`` of ``base_frame`` in ``map_frame``, or None.

        Uses "latest available" (``rclpy.time.Time()``); the buffer
        handles caching + interpolation. Any tf2 failure (missing/stale
        transform) returns None so the map still renders without a marker.
        """
        self._ensure_tf()
        if self._tf_buffer is None:
            return None
        try:
            import rclpy.time  # noqa: PLC0415

            tf = self._tf_buffer.lookup_transform(
                map_frame,
                self._base_frame,
                rclpy.time.Time(),
            )
        except Exception as exc:  # tf2 raises LookupException / ExtrapolationException /
            # ConnectivityException; any failure → render map without robot marker.
            if not self._tf_warned:
                self._tf_warned = True
                logging.getLogger(__name__).warning(
                    "SlamMapBridge: tf lookup %s→%s failed (%s); "
                    "rendering map without robot marker",
                    map_frame,
                    self._base_frame,
                    exc,
                )
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        self._tf_warned = False  # reset so the next failure episode logs again
        return robot_pose_from_transform(
            translation_xyz=(float(t.x), float(t.y), float(t.z)),
            rotation_xyzw=(float(r.x), float(r.y), float(r.z), float(r.w)),
        )

    def _on_map(self, msg: Any) -> None:  # noqa: ANN401  # reason: nav_msgs.msg.OccupancyGrid IDL
        """``/map`` callback — emit one OTel span per accepted message."""
        now = time.monotonic()
        if (now - self._last_publish_ts_s) < self._publish_interval_s:
            return

        info = msg.info
        width = int(info.width)
        height = int(info.height)
        n_cells = width * height
        if n_cells <= 0:
            return
        if n_cells > self._max_cells:
            # Don't crash the bridge; just warn and drop the oversize
            # grid. The operator can lower SLAM resolution or trim the
            # map's max range.
            self._dropped_oversize += 1
            if (
                self._dropped_oversize <= _OVERSIZE_WARN_BURST
                or self._dropped_oversize % _OVERSIZE_WARN_RATE == 0
            ):
                # rclpy loggers don't have structlog binding; standard
                # logger is enough for this rare path.
                logging.getLogger(__name__).warning(
                    "SlamMapBridge: dropping %dx%d grid (%d cells > %d limit); "
                    "lower slam_toolbox resolution or shrink the max range "
                    "(dropped=%d)",
                    width,
                    height,
                    n_cells,
                    self._max_cells,
                    self._dropped_oversize,
                )
            return

        try:
            png_b64 = encode_occupancy_grid_png(
                width=width,
                height=height,
                data=list(msg.data),
            )
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "SlamMapBridge: encode failed (%s); skipping this /map message", exc
            )
            return

        self._last_publish_ts_s = now
        with self._tracer.start_as_current_span("slam.occupancy_grid") as span:
            span.set_attribute("openral.slam.frame_id", str(msg.header.frame_id))
            span.set_attribute("openral.slam.width", width)
            span.set_attribute("openral.slam.height", height)
            span.set_attribute("openral.slam.resolution_m", float(info.resolution))
            span.set_attribute("openral.slam.origin_x", float(info.origin.position.x))
            span.set_attribute("openral.slam.origin_y", float(info.origin.position.y))
            span.set_attribute("openral.slam.png_b64", png_b64)
            span.set_attribute("openral.slam.source_node", self._source_node_name)
            pose = self._lookup_robot_pose(str(msg.header.frame_id))
            if pose is not None:
                robot_x, robot_y, robot_yaw = pose
                span.set_attribute("openral.slam.robot_x", robot_x)
                span.set_attribute("openral.slam.robot_y", robot_y)
                span.set_attribute("openral.slam.robot_yaw", robot_yaw)
                span.set_attribute("openral.slam.base_frame", self._base_frame)
                if self._footprint_radius_m is not None:
                    span.set_attribute(
                        "openral.slam.footprint_radius_m",
                        float(self._footprint_radius_m),
                    )
                if self._footprint_polygon_xy is not None:
                    span.set_attribute(
                        "openral.slam.footprint_polygon_xy",
                        self._footprint_polygon_xy,
                    )
