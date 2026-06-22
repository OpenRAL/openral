#!/usr/bin/env python3
"""Filter metric depth to a robot-derived height band before nvblox mapping.

ADR-0064 — nvblox's ``static_occupancy_grid`` is the backend-agnostic ``/map``
OpenRAL gives Nav2 and the dashboard. Isaac ROS nvblox 4.4 applies
``static_mapper.workspace_bounds_*`` to TSDF integration, but the static
occupancy camera integrator still projects all depth returns into 2D. A
forward/downward camera therefore marks the floor as occupied.

This node removes depth pixels outside the robot's navigation body height before
nvblox receives them. The band is derived from the loaded ``RobotDescription``
and shifted by the live ``global_frame -> base_frame`` TF, so a robot with the
same measurements works across scenes whose map-frame floor height differs. The
output remains a standard ``32FC1`` depth image plus ``CameraInfo``; nvblox then
builds a floor-excluded 2D obstacle grid.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RobotRelativeHeightBand",
    "derive_robot_relative_height_band",
    "filter_depth_by_global_height",
    "main",
    "quaternion_to_matrix_z_row",
]


@dataclass(frozen=True)
class RobotRelativeHeightBand:
    """Navigation-relevant vertical band in the robot ``base_frame``."""

    min_z_m: float
    max_z_m: float
    source: str


def quaternion_to_matrix_z_row(
    x: float, y: float, z: float, w: float
) -> tuple[float, float, float]:
    """Return the third row of the rotation matrix for quaternion ``(x, y, z, w)``.

    The height filter only needs the global-frame z coordinate of each projected
    depth point, so computing the full 3x3 matrix would be unnecessary.
    """
    return (
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_vec(
    r: tuple[tuple[float, float, float], ...],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    )


def _mat_mul(
    a: tuple[tuple[float, float, float], ...],
    b: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    cols = ((b[0][0], b[1][0], b[2][0]), (b[0][1], b[1][1], b[2][1]), (b[0][2], b[1][2], b[2][2]))
    return tuple(tuple(sum(row[i] * col[i] for i in range(3)) for col in cols) for row in a)  # type: ignore[return-value] # reason: tuple comprehension has fixed 3x3 shape.


def _vec_add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _compose(
    parent_r: tuple[tuple[float, float, float], ...],
    parent_t: tuple[float, float, float],
    child_r: tuple[tuple[float, float, float], ...],
    child_t: tuple[float, float, float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    return _mat_mul(parent_r, child_r), _vec_add(parent_t, _mat_vec(parent_r, child_t))


def _link_transforms_at_zero(
    description: Any,
) -> dict[str, tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]]:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    transforms: dict[
        str, tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]
    ] = {str(description.base_frame): (identity, (0.0, 0.0, 0.0))}
    pending = list(description.joints)
    progressed = True
    while pending and progressed:
        progressed = False
        rest = []
        for joint in pending:
            parent = transforms.get(str(joint.parent_link))
            if parent is None:
                rest.append(joint)
                continue
            jr = _rpy_to_matrix(*[float(v) for v in joint.origin_rpy])
            jt = tuple(float(v) for v in joint.origin_xyz)
            transforms[str(joint.child_link)] = _compose(parent[0], parent[1], jr, jt)  # type: ignore[arg-type] # reason: JointSpec origin_xyz is a fixed 3-tuple.
            progressed = True
        pending = rest
    return transforms


def _footprint_body_height_m(description: Any) -> float | None:
    if description.footprint_polygon:
        xs = [float(p[0]) for p in description.footprint_polygon]
        ys = [float(p[1]) for p in description.footprint_polygon]
        return max(max(xs) - min(xs), max(ys) - min(ys))
    if description.footprint_radius is not None:
        return 2.0 * float(description.footprint_radius)
    return None


def _collision_z_extent_m(description: Any) -> tuple[float, float] | None:
    transforms = _link_transforms_at_zero(description)
    z_values: list[float] = []
    for geom in description.collision_geometry:
        link_tf = transforms.get(str(geom.link_name))
        if link_tf is None:
            continue
        link_r, link_t = link_tf
        ox, oy, oz, roll, pitch, yaw = (float(v) for v in geom.origin_xyz_rpy)
        geom_r, geom_t = _compose(link_r, link_t, _rpy_to_matrix(roll, pitch, yaw), (ox, oy, oz))
        shape = geom.shape
        radius = float(shape.radius_m)
        if shape.shape == "sphere":
            z_values.extend([geom_t[2] - radius, geom_t[2] + radius])
            continue
        half_len = float(shape.length_m) * 0.5
        axis_z = (geom_r[0][2], geom_r[1][2], geom_r[2][2])
        end_a_z = geom_t[2] - axis_z[2] * half_len
        end_b_z = geom_t[2] + axis_z[2] * half_len
        z_values.extend([min(end_a_z, end_b_z) - radius, max(end_a_z, end_b_z) + radius])
    if not z_values:
        return None
    return min(z_values), max(z_values)


def derive_robot_relative_height_band(
    description: Any,
    *,
    floor_clearance_m: float = 0.10,
    min_body_height_m: float = 0.30,
) -> RobotRelativeHeightBand:
    """Derive the depth-retention band from robot measurements.

    The lower edge is the robot-relative floor plus a small clearance so floor
    pixels do not become occupied cells. The upper edge is the measured body
    height from the manifest footprint and collision/link geometry. The result
    is relative to ``base_frame``; the ROS node shifts it into ``global_frame``
    using live TF for every frame.
    """
    if floor_clearance_m < 0.0:
        raise ValueError(f"floor_clearance_m must be non-negative, got {floor_clearance_m}")
    if min_body_height_m <= floor_clearance_m:
        raise ValueError(
            "min_body_height_m must exceed floor_clearance_m "
            f"(got {min_body_height_m} <= {floor_clearance_m})"
        )

    collision_extent = _collision_z_extent_m(description)
    footprint_height = _footprint_body_height_m(description)

    floor_z = min(0.0, collision_extent[0]) if collision_extent is not None else 0.0
    measured_max = floor_z + min_body_height_m
    sources = ["minimum"]
    if footprint_height is not None:
        measured_max = max(measured_max, floor_z + footprint_height)
        sources.append("footprint")
    if collision_extent is not None:
        measured_max = max(measured_max, collision_extent[1])
        sources.append("collision_geometry")

    min_z = floor_z + floor_clearance_m
    if measured_max <= min_z:
        measured_max = min_z + min_body_height_m
    return RobotRelativeHeightBand(min_z_m=min_z, max_z_m=measured_max, source="+".join(sources))


def filter_depth_by_global_height(
    depth_m: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation_z_row: tuple[float, float, float],
    translation_z_m: float,
    min_height_m: float,
    max_height_m: float,
) -> Any:
    """Zero depth pixels whose back-projected point is outside a global height band.

    Args:
        depth_m: ``HxW`` float depth image in metres, optical-Z convention.
        fx: Camera focal length in x, pixels.
        fy: Camera focal length in y, pixels.
        cx: Camera principal point x, pixels.
        cy: Camera principal point y, pixels.
        rotation_z_row: Third row of the camera-optical-frame to global-frame
            rotation matrix.
        translation_z_m: Camera origin z in ``global_frame`` metres.
        min_height_m: Inclusive minimum allowed global z.
        max_height_m: Inclusive maximum allowed global z.

    Returns:
        A contiguous ``float32`` copy where out-of-band or invalid depths are 0.

    Raises:
        ValueError: If the depth array is not 2-D, intrinsics are invalid, or the
            height band is inverted.
    """
    import numpy as np

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"invalid pinhole intrinsics: fx={fx}, fy={fy}")
    if min_height_m > max_height_m:
        raise ValueError(
            f"invalid height band: min_height_m={min_height_m} > max_height_m={max_height_m}"
        )

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D HxW, got shape {depth.shape}")
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    valid = np.isfinite(depth) & (depth > 0.0)

    r20, r21, r22 = rotation_z_row
    x_opt = ((u - float(cx)) / float(fx)) * depth
    y_opt = ((v - float(cy)) / float(fy)) * depth
    z_global = (r20 * x_opt) + (r21 * y_opt) + (r22 * depth) + float(translation_z_m)
    keep = valid & (z_global >= float(min_height_m)) & (z_global <= float(max_height_m))

    filtered = np.zeros_like(depth, dtype=np.float32)
    filtered[keep] = depth[keep]
    return np.ascontiguousarray(filtered)


def _depth_array_from_image(msg: Any) -> Any:
    import numpy as np

    if msg.encoding != "32FC1":
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}; expected '32FC1'")
    expected_step = int(msg.width) * 4
    if int(msg.step) != expected_step:
        raise ValueError(
            f"unsupported padded depth rows: step={msg.step}, expected {expected_step}"
        )
    return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(int(msg.height), int(msg.width))


def _depth_image_from_array(depth_m: Any, template: Any) -> Any:
    import numpy as np
    from sensor_msgs.msg import Image

    arr = np.ascontiguousarray(depth_m, dtype=np.float32)
    msg = Image()
    msg.header = deepcopy(template.header)
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = arr.tobytes()
    return msg


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_slam_bringup depth_height_filter_node.py``."""
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformException, TransformListener

    class DepthHeightFilterNode(Node):  # type: ignore[misc]
        """ROS node wrapper around :func:`filter_depth_by_global_height`."""

        def __init__(self) -> None:
            super().__init__("openral_nvblox_depth_height_filter")
            self.declare_parameter("input_depth_topic", "/openral/cameras/front_depth/depth/image")
            self.declare_parameter(
                "input_camera_info_topic", "/openral/cameras/front_depth/depth/camera_info"
            )
            self.declare_parameter("output_depth_topic", "/openral/nvblox/depth_filtered/image")
            self.declare_parameter(
                "output_camera_info_topic", "/openral/nvblox/depth_filtered/camera_info"
            )
            self.declare_parameter("global_frame", "map")
            self.declare_parameter("base_frame", "base_link")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("floor_clearance_m", 0.10)
            self.declare_parameter("min_body_height_m", 0.30)
            self.declare_parameter("min_height_m", float("nan"))
            self.declare_parameter("max_height_m", float("nan"))
            self.declare_parameter("tf_timeout_ms", 50)

            gp = self.get_parameter
            self._global_frame = gp("global_frame").get_parameter_value().string_value
            self._base_frame = gp("base_frame").get_parameter_value().string_value
            self._floor_clearance = gp("floor_clearance_m").get_parameter_value().double_value
            self._min_body_height = gp("min_body_height_m").get_parameter_value().double_value
            self._override_min_height = gp("min_height_m").get_parameter_value().double_value
            self._override_max_height = gp("max_height_m").get_parameter_value().double_value
            self._tf_timeout = Duration(
                seconds=gp("tf_timeout_ms").get_parameter_value().integer_value / 1000.0
            )
            self._relative_band: RobotRelativeHeightBand | None = None
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            if robot_yaml:
                from openral_core import RobotDescription

                description = RobotDescription.from_yaml(robot_yaml)
                self._base_frame = description.base_frame
                self._global_frame = description.map_frame
                self._relative_band = derive_robot_relative_height_band(
                    description,
                    floor_clearance_m=self._floor_clearance,
                    min_body_height_m=self._min_body_height,
                )
            if math.isfinite(self._override_min_height) != math.isfinite(self._override_max_height):
                raise RuntimeError("min_height_m and max_height_m overrides must be set together")
            self._latest_info: CameraInfo | None = None
            self._warned_no_info = False

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            sensor_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            info_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._depth_pub = self.create_publisher(
                Image, gp("output_depth_topic").get_parameter_value().string_value, sensor_qos
            )
            self._info_pub = self.create_publisher(
                CameraInfo,
                gp("output_camera_info_topic").get_parameter_value().string_value,
                info_qos,
            )
            self._info_sub = self.create_subscription(
                CameraInfo,
                gp("input_camera_info_topic").get_parameter_value().string_value,
                self._on_camera_info,
                info_qos,
            )
            self._depth_sub = self.create_subscription(
                Image,
                gp("input_depth_topic").get_parameter_value().string_value,
                self._on_depth,
                sensor_qos,
            )
            band_desc = (
                f"override {self._global_frame} z="
                f"[{self._override_min_height:.2f}, {self._override_max_height:.2f}]"
                if math.isfinite(self._override_min_height)
                else (
                    f"robot-relative {self._base_frame} z="
                    f"[{self._relative_band.min_z_m:.2f}, {self._relative_band.max_z_m:.2f}] "
                    f"({self._relative_band.source})"
                    if self._relative_band is not None
                    else f"fallback live {self._base_frame}/camera height"
                )
            )
            self.get_logger().info(
                "depth_height_filter: "
                f"{gp('input_depth_topic').get_parameter_value().string_value} -> "
                f"{gp('output_depth_topic').get_parameter_value().string_value} "
                f"({band_desc})"
            )

        def _on_camera_info(self, msg: CameraInfo) -> None:
            self._latest_info = msg

        def _on_depth(self, msg: Image) -> None:
            if self._latest_info is None:
                if not self._warned_no_info:
                    self.get_logger().warning("skip depth frame: no CameraInfo received yet")
                    self._warned_no_info = True
                return

            info = self._latest_info
            try:
                depth = _depth_array_from_image(msg)
            except ValueError as exc:
                self.get_logger().warning(f"skip depth frame: {exc}")
                return

            try:
                transform = self._tf_buffer.lookup_transform(
                    self._global_frame,
                    msg.header.frame_id,
                    Time.from_msg(msg.header.stamp),
                    timeout=self._tf_timeout,
                )
            except TransformException as exc:
                self.get_logger().warning(f"skip depth frame: TF lookup failed: {exc}")
                return

            q = transform.transform.rotation
            t = transform.transform.translation
            try:
                min_height_m = self._override_min_height
                max_height_m = self._override_max_height
                if not math.isfinite(min_height_m):
                    base_transform = self._tf_buffer.lookup_transform(
                        self._global_frame,
                        self._base_frame,
                        Time.from_msg(msg.header.stamp),
                        timeout=self._tf_timeout,
                    )
                    base_z = float(base_transform.transform.translation.z)
                    if self._relative_band is not None:
                        min_height_m = base_z + self._relative_band.min_z_m
                        max_height_m = base_z + self._relative_band.max_z_m
                        max_height_m = max(max_height_m, float(t.z) + self._floor_clearance)
                    else:
                        min_height_m = base_z + self._floor_clearance
                        max_height_m = max(
                            base_z + self._min_body_height,
                            float(t.z) + self._floor_clearance,
                        )
                filtered = filter_depth_by_global_height(
                    depth,
                    fx=float(info.k[0]),
                    fy=float(info.k[4]),
                    cx=float(info.k[2]),
                    cy=float(info.k[5]),
                    rotation_z_row=quaternion_to_matrix_z_row(q.x, q.y, q.z, q.w),
                    translation_z_m=float(t.z),
                    min_height_m=min_height_m,
                    max_height_m=max_height_m,
                )
            except (TransformException, ValueError) as exc:
                self.get_logger().warning(f"skip depth frame: {exc}")
                return

            out_info = deepcopy(info)
            out_info.header.stamp = msg.header.stamp
            out_info.header.frame_id = msg.header.frame_id
            self._depth_pub.publish(_depth_image_from_array(filtered, msg))
            self._info_pub.publish(out_info)

    rclpy.init(args=args)
    node = DepthHeightFilterNode()
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
