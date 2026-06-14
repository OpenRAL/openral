# SPDX-License-Identifier: Apache-2.0
"""ADR-0030 — rclpy → OTLP bridge for the octomap occupied-voxel cloud.

The OpenRAL dashboard is OTLP-only — it never subscribes to ROS topics
directly. This module ships :class:`WorldCloudBridge`, a small consumer
constructed against an existing ``rclpy.node.Node`` that subscribes to the
``sensor_msgs/PointCloud2`` octomap_server publishes on
``/octomap_point_cloud_centers`` (occupied voxel centers — the ADR-0030
world map the safety kernel gates on), transforms the points into the
robot ``base_link`` frame via TF2, crops them to a local box, renders an
oblique "chase-cam" perspective PNG colored by distance from the robot,
and emits one ``world.pointcloud`` OTel span carrying the metadata + PNG
as attributes.

The dashboard store has a matching handler that populates
``_topics["pointcloud"]`` from the span (see
``openral_observability.dashboard.store``).

The pure render functions (:func:`crop_points_to_box`,
:func:`encode_world_cloud_png`, :func:`world_cloud_span_attributes`) take
plain ``(N, 3)`` arrays so the dashboard contract is testable without ROS.

Composed into the existing ``RskillRunnerNode`` via
``packages/openral_rskill_ros/openral_rskill_ros/compose.py`` so it shares
the runner's rclpy executor; constructing it manually outside compose is
supported for tests.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)

__all__ = [
    "WORLD_CLOUD_TOPIC_DEFAULT",
    "WorldCloudBridge",
    "crop_points_to_box",
    "distance_to_rgb",
    "encode_world_cloud_png",
    "world_cloud_span_attributes",
]

WORLD_CLOUD_TOPIC_DEFAULT = "/octomap_point_cloud_centers"
"""Default ``sensor_msgs/PointCloud2`` of occupied voxel centers (octomap_server)."""

# Throttle so a busy octomap run doesn't flood the OTLP pipeline (the
# dashboard re-renders on every span; >1 Hz just burns CPU).
_DEFAULT_PUBLISH_INTERVAL_S = 1.0

# How many oversize-cloud warnings to emit verbosely before dropping to a
# 1-in-100 rate, mirroring SlamMapBridge's bounded warn policy.
_DROP_WARN_BURST = 3
_DROP_WARN_RATE = 100

# Oblique chase camera expressed in base_link: placed behind (-x) and above
# (+z) the robot, looking forward (+x) with a slight downward tilt. Focal
# length in pixels tuned so a ~2 m local box fills a 480x360 frame.
_CAM_BACK_M = 2.2
_CAM_UP_M = 1.6
_CAM_PITCH_DOWN_RAD = 0.45
_FOCAL_PX = 320.0
# Camera-forward depth below which a point is treated as behind the lens.
_MIN_CAM_DEPTH_M = 1e-3

_BG_RGB = (16, 20, 28)
_ORIGIN_RGB = (240, 240, 255)


def crop_points_to_box(
    points: NDArray[np.float32], *, xy_m: float, z_min: float, z_max: float
) -> NDArray[np.float32]:
    """Keep only points inside the local box around the ``base_link`` origin."""
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    p = np.ascontiguousarray(points, dtype=np.float32)
    mask = (
        (np.abs(p[:, 0]) <= xy_m)
        & (np.abs(p[:, 1]) <= xy_m)
        & (p[:, 2] >= z_min)
        & (p[:, 2] <= z_max)
    )
    return p[mask]


def distance_to_rgb(dist_m: float, *, range_max_m: float) -> tuple[int, int, int]:
    """Map a distance to an RGB triple via a near=warm → far=cool ramp."""
    t = 0.0 if range_max_m <= 0 else min(max(dist_m / range_max_m, 0.0), 1.0)
    r = round(255 * (1.0 - t))
    g = round(255 * (1.0 - 0.5 * t))
    b = round(255 * t)
    return (r, g, b)


def _project_chase_view(
    points_base: NDArray[np.float32], *, image_w: int, image_h: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], NDArray[np.intp]]:
    """Project base_link points through the fixed oblique chase camera.

    Returns ``(us, vs, cam_depth, order)`` where ``us``/``vs`` are pixel
    columns/rows (``-1`` for points behind the camera), ``cam_depth`` is
    the camera-forward distance, and ``order`` sorts far→near for
    painter's-algorithm drawing.
    """
    n = points_base.shape[0]
    if n == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        return empty_i, empty_i, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.intp)
    cp = float(np.cos(_CAM_PITCH_DOWN_RAD))
    sp = float(np.sin(_CAM_PITCH_DOWN_RAD))
    # Camera origin in base frame; translate then pitch-down about base y.
    # Pitch-down rotation about +y by _CAM_PITCH_DOWN_RAD: the look axis tilts
    # from +x toward -z, so fwd = cos*x - sin*z and up = sin*x + cos*z. (The
    # opposite signs pitch the camera *up*, projecting the whole scene below the
    # principal point where it clips onto the bottom row — the flattened map.)
    x = points_base[:, 0] + _CAM_BACK_M
    y = points_base[:, 1]
    z = points_base[:, 2] - _CAM_UP_M
    fwd = cp * x - sp * z  # camera +Z (into the scene)
    up = sp * x + cp * z  # camera +Y (image up before the row flip)
    valid = fwd > _MIN_CAM_DEPTH_M
    us = np.full(n, -1, dtype=np.int32)
    vs = np.full(n, -1, dtype=np.int32)
    u = (_FOCAL_PX * (y[valid] / fwd[valid])) + image_w / 2.0
    v = (_FOCAL_PX * (up[valid] / fwd[valid])) + image_h / 2.0
    us[valid] = np.clip(u, 0, image_w - 1).astype(np.int32)
    vs[valid] = np.clip(v, 0, image_h - 1).astype(np.int32)
    cam_depth = np.where(valid, fwd, np.inf).astype(np.float32)
    order = np.argsort(-cam_depth)  # far first
    return us, vs, cam_depth, order


def encode_world_cloud_png(
    points_base: NDArray[np.float32],
    *,
    range_max_m: float = 4.0,
    image_w: int = 480,
    image_h: int = 360,
    xy_m: float = 2.0,
    z_min: float = -0.2,
    z_max: float = 2.0,
) -> str:
    """Render base_link points as an oblique chase-view base64 PNG.

    Pure function — used directly by tests against synthetic ``(N, 3)``
    arrays so the dashboard contract is exercisable without ROS.

    Args:
        points_base: ``(N, 3)`` XYZ in ``base_link`` (metres).
        range_max_m: distance normalisation for the color ramp.
        image_w: output PNG width in pixels.
        image_h: output PNG height in pixels.
        xy_m: half-extent of the local crop box in x/y (metres).
        z_min: lower z bound of the crop box (metres).
        z_max: upper z bound of the crop box (metres).

    Returns:
        Base64 PNG (UTF-8), ready for ``<img src="data:image/png;base64,...">``.

    Example:
        >>> import numpy as np
        >>> png = encode_world_cloud_png(np.zeros((1, 3), dtype=np.float32))
        >>> png.startswith("iVBOR")  # PNG magic, base64-encoded
        True
    """
    from PIL import Image  # noqa: PLC0415  # reason: defer optional dep (already an OpenRAL dep)

    canvas = np.empty((image_h, image_w, 3), dtype=np.uint8)
    canvas[:, :] = _BG_RGB

    pts = crop_points_to_box(
        np.ascontiguousarray(points_base, dtype=np.float32),
        xy_m=xy_m,
        z_min=z_min,
        z_max=z_max,
    )
    if pts.shape[0] > 0:
        us, vs, _depth, order = _project_chase_view(pts, image_w=image_w, image_h=image_h)
        dists = np.linalg.norm(pts, axis=1)
        for i in order:
            if us[i] < 0:
                continue
            rgb = distance_to_rgb(float(dists[i]), range_max_m=range_max_m)
            yy = image_h - 1 - int(vs[i])  # flip so "up" is up in the image
            xx = int(us[i])
            canvas[yy, xx] = rgb
            if xx + 1 < image_w:  # 2x2 splat for visibility
                canvas[yy, xx + 1] = rgb
            if yy - 1 >= 0:
                canvas[yy - 1, xx] = rgb

    ou, ov, _d, _o = _project_chase_view(
        np.zeros((1, 3), dtype=np.float32), image_w=image_w, image_h=image_h
    )
    if ou[0] >= 0:  # robot origin marker
        oy = image_h - 1 - int(ov[0])
        ox = int(ou[0])
        canvas[max(oy - 2, 0) : oy + 3, max(ox - 2, 0) : ox + 3] = _ORIGIN_RGB

    img = Image.fromarray(canvas, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def world_cloud_span_attributes(
    *,
    points_base: NDArray[np.float32],
    frame_id: str,
    source_node: str,
    range_max_m: float,
    xy_m: float,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    """Assemble the ``world.pointcloud`` span attribute dict (pure)."""
    png_b64 = encode_world_cloud_png(
        points_base, range_max_m=range_max_m, xy_m=xy_m, z_min=z_min, z_max=z_max
    )
    return {
        "openral.world_cloud.frame_id": frame_id,
        "openral.world_cloud.n_points": int(points_base.shape[0]),
        "openral.world_cloud.png_b64": png_b64,
        "openral.world_cloud.source_node": source_node,
        "openral.world_cloud.crop_xy_m": float(xy_m),
        "openral.world_cloud.crop_z_m_min": float(z_min),
        "openral.world_cloud.crop_z_m_max": float(z_max),
        "openral.world_cloud.range_max_m": float(range_max_m),
    }


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> NDArray[np.float32]:
    """Rotation matrix from a quaternion ``(x, y, z, w)``."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _apply_transform(points: NDArray[np.float32], tf: Any) -> NDArray[np.float32]:  # noqa: ANN401  # reason: geometry_msgs/TransformStamped IDL
    """Apply a ``geometry_msgs/TransformStamped`` to ``(N, 3)`` points."""
    t = tf.transform.translation
    q = tf.transform.rotation
    rot = _quat_to_matrix(q.x, q.y, q.z, q.w)
    return (points @ rot.T) + np.array([t.x, t.y, t.z], dtype=np.float32)


class WorldCloudBridge:
    """rclpy → OTLP bridge for ``/octomap_point_cloud_centers`` (ADR-0030).

    Mirrors :class:`openral_runner.slam_bridge.SlamMapBridge`: constructed
    against an existing ``rclpy.node.Node`` so the PointCloud2 subscription
    shares the runner's executor. On each accepted message it reads the
    cloud, transforms it into ``base_frame`` via TF2, crops, renders the
    oblique chase view, and emits one ``world.pointcloud`` span.

    Args:
        node: Host ``rclpy.node.Node``; the subscription + TF listener are
            created on it. :meth:`destroy` releases the subscription.
        topic: PointCloud2 topic. Defaults to
            :data:`WORLD_CLOUD_TOPIC_DEFAULT`.
        base_frame: tf2 frame to express the cloud in (the robot frame).
        source_node_name: identifier surfaced on the dashboard card.
        publish_interval_s: minimum wall-clock interval between spans.
        max_points: drop clouds larger than this (warn-and-skip) to bound
            render cost / OTLP payload.
        xy_m / z_min / z_max: local crop box around the robot.
        range_max_m: distance normalisation for the color ramp.
    """

    def __init__(
        self,
        node: Any,  # noqa: ANN401  # reason: rclpy.node.Node not importable without a sourced ROS 2 workspace
        *,
        topic: str = WORLD_CLOUD_TOPIC_DEFAULT,
        base_frame: str = "base_link",
        source_node_name: str = "openral_octomap_server",
        publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
        max_points: int = 200_000,
        xy_m: float = 2.0,
        z_min: float = -0.2,
        z_max: float = 2.0,
        range_max_m: float = 4.0,
    ) -> None:
        """Subscribe to ``topic``, start a TF listener, prepare the tracer."""
        import tf2_ros  # type: ignore[import-untyped,unused-ignore]  # noqa: PLC0415
        from opentelemetry import trace  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import PointCloud2  # noqa: PLC0415

        # octomap_server latches its centers topic; mirror TRANSIENT_LOCAL so
        # a late-joining bridge still gets the latest map.
        cloud_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._node = node
        self._topic = topic
        self._base_frame = base_frame
        self._source_node_name = source_node_name
        self._publish_interval_s = publish_interval_s
        self._max_points = max_points
        self._xy_m = xy_m
        self._z_min = z_min
        self._z_max = z_max
        self._range_max_m = range_max_m
        self._last_publish_ts_s: float = 0.0
        self._dropped = 0
        self._tracer = trace.get_tracer("openral.world_cloud_bridge")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)
        self._subscription = node.create_subscription(PointCloud2, topic, self._on_cloud, cloud_qos)

    def destroy(self) -> None:
        """Release the ROS subscription. Safe to call multiple times."""
        if self._subscription is None:
            return
        try:
            self._node.destroy_subscription(self._subscription)
        finally:
            self._subscription = None

    def _on_cloud(self, msg: Any) -> None:  # noqa: ANN401  # reason: sensor_msgs.msg.PointCloud2 IDL
        """PointCloud2 callback — emit one ``world.pointcloud`` span per accept."""
        now = time.monotonic()
        if (now - self._last_publish_ts_s) < self._publish_interval_s:
            return
        pts = self._read_and_transform(msg)
        if pts is None:
            return
        if pts.shape[0] > self._max_points:
            self._dropped += 1
            if self._dropped <= _DROP_WARN_BURST or self._dropped % _DROP_WARN_RATE == 0:
                logging.getLogger(__name__).warning(
                    "WorldCloudBridge: dropping %d-point cloud (> %d limit); "
                    "raise octomap resolution or shrink the crop box (dropped=%d)",
                    pts.shape[0],
                    self._max_points,
                    self._dropped,
                )
            return
        self._last_publish_ts_s = now
        attrs = world_cloud_span_attributes(
            points_base=pts,
            frame_id=self._base_frame,
            source_node=self._source_node_name,
            range_max_m=self._range_max_m,
            xy_m=self._xy_m,
            z_min=self._z_min,
            z_max=self._z_max,
        )
        with self._tracer.start_as_current_span("world.pointcloud") as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)

    def _read_and_transform(self, msg: Any) -> NDArray[np.float32] | None:  # noqa: ANN401  # reason: PointCloud2 IDL
        """Read XYZ and express it in ``base_frame`` via TF2.

        Returns ``None`` (warn-and-skip) when TF is not yet available — the
        HAL publishes ``odom→base_link`` only after ``on_activate``.
        """
        import rclpy.time  # noqa: PLC0415
        import tf2_ros  # noqa: PLC0415
        from sensor_msgs_py import point_cloud2  # noqa: PLC0415

        raw = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        pts = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return pts
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                msg.header.frame_id,
                rclpy.time.Time(),  # latest available
            )
        except (tf2_ros.LookupException, tf2_ros.TransformException) as exc:
            logging.getLogger(__name__).warning(
                "WorldCloudBridge: TF %s<-%s unavailable (%s); skipping frame",
                self._base_frame,
                msg.header.frame_id,
                exc,
            )
            return None
        return _apply_transform(pts, tf)
