#!/usr/bin/env python3
"""Stereo visual SLAM from the PyCuVSLAM wheel — no Isaac ROS apt stack.

The existing visual backend (``cuvslam.launch.py``) composes NVIDIA's
``isaac_ros_visual_slam`` C++ node, which requires the operator's full
Isaac ROS apt install (NITROS, VPI, ``nvsci`` — see the package README
for the x86 install pain). NVIDIA now also ships **PyCuVSLAM**
(https://github.com/nvidia-isaac/cuVSLAM): the same cuVSLAM engine as a
pip wheel with a Python API, supporting this workspace's Python 3.12 on
Ubuntu 24.04 x86_64/aarch64 (CUDA 12/13).

This node runs that engine in-process: it subscribes a synchronized
rectified stereo pair from the OpenRAL camera bus, tracks with
``cuvslam.Tracker``, and fills the same ``map → odom`` TF edge the other
SLAM backends fill (composing the tracker's ``map ← rig`` pose with the
live ``odom ← rig`` TF). The rig frame is the **left camera optical
frame**; images must be rectified (pinhole) with the stereo baseline
encoded in the right camera's projection matrix ``P`` (standard ROS
rectified-stereo convention, as produced by ``stereo_image_proc``,
RealSense rectified streams, and the OpenRAL sim camera bus).

License posture: the cuVSLAM engine is NVIDIA-proprietary (NVIDIA
Community License — commercial use OK, NVIDIA hardware only). OpenRAL
does **not** bundle or depend on the wheel; the operator installs it
(see README). Import failure raises ``ROSConfigError`` at node startup.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "compose_pose",
    "invert_pose",
    "main",
    "map_from_odom",
    "stereo_baseline_m",
]

# A pose is ((qx, qy, qz, qw), (tx, ty, tz)) — the cuVSLAM convention
# (xyzw quaternion), which also matches geometry_msgs field order.
_Quat = tuple[float, float, float, float]
_Vec3 = tuple[float, float, float]
_Pose = tuple[_Quat, _Vec3]


def stereo_baseline_m(right_p: Any) -> float:
    """Extract the stereo baseline from a rectified right camera's ``P`` matrix.

    In the ROS rectified-stereo convention the right camera's 3x4
    projection matrix carries ``P[0][3] = -fx * baseline`` (``Tx``).

    Args:
        right_p: The right ``CameraInfo.p`` — 12 row-major floats.

    Returns:
        The baseline in metres (> 0).

    Raises:
        ValueError: If ``P`` is malformed, ``fx`` is not positive, or the
            encoded baseline is not positive (unrectified / mono stream).

    Example:
        >>> p = [386.9, 0.0, 320.0, -19.30631, 0.0, 386.9, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        >>> round(stereo_baseline_m(p), 4)
        0.0499
    """
    p = [float(v) for v in right_p]
    if len(p) != 12:
        raise ValueError(f"CameraInfo.p must have 12 entries, got {len(p)}")
    fx, tx = p[0], p[3]
    if fx <= 0.0:
        raise ValueError(f"invalid rectified intrinsics: fx={fx}")
    baseline = -tx / fx
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(
            f"right P encodes no positive stereo baseline (Tx={tx}, fx={fx}); "
            "is this a rectified stereo right camera_info?"
        )
    return baseline


def _quat_mul(a: _Quat, b: _Quat) -> _Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_rotate(q: _Quat, v: _Vec3) -> _Vec3:
    # v' = q * (v, 0) * q⁻¹ for a unit quaternion.
    qx, qy, qz, qw = q
    ux, uy, uz = qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]
    uux, uuy, uuz = qy * uz - qz * uy, qz * ux - qx * uz, qx * uy - qy * ux
    return (
        v[0] + 2.0 * (qw * ux + uux),
        v[1] + 2.0 * (qw * uy + uuy),
        v[2] + 2.0 * (qw * uz + uuz),
    )


def compose_pose(a: _Pose, b: _Pose) -> _Pose:
    """Compose two poses: the transform applying ``b`` then ``a`` (``a ∘ b``)."""
    aq, at = a
    bq, bt = b
    rt = _quat_rotate(aq, bt)
    return _quat_mul(aq, bq), (at[0] + rt[0], at[1] + rt[1], at[2] + rt[2])


def invert_pose(p: _Pose) -> _Pose:
    """Invert a unit-quaternion pose."""
    (qx, qy, qz, qw), t = p
    inv_q: _Quat = (-qx, -qy, -qz, qw)
    it = _quat_rotate(inv_q, t)
    return inv_q, (-it[0], -it[1], -it[2])


def map_from_odom(map_from_rig: _Pose, odom_from_rig: _Pose) -> _Pose:
    """The ``map → odom`` correction from a tracked pose and the live odom TF.

    Example:
        >>> identity = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        >>> moved = ((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        >>> map_from_odom(moved, moved)  # odometry already accounts for the motion
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    """
    return compose_pose(map_from_rig, invert_pose(odom_from_rig))


def _image_to_array(msg: Any) -> Any:
    """Convert a ``sensor_msgs/Image`` to the ``HxW``/``HxWx3`` uint8 array cuVSLAM eats."""
    import numpy as np

    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}; expected mono8/rgb8/bgr8")
    expected_step = int(msg.width) * channels
    if int(msg.step) != expected_step:
        raise ValueError(f"unsupported padded rows: step={msg.step}, expected {expected_step}")
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = (
        arr.reshape(int(msg.height), int(msg.width), channels)
        if channels == 3
        else arr.reshape(int(msg.height), int(msg.width))
    )
    if msg.encoding == "bgr8":
        arr = np.ascontiguousarray(arr[:, :, ::-1])
    return arr


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_slam_bringup pycuvslam_node.py``."""
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

    class PyCuVSLAMNode(Node):  # type: ignore[misc]
        """Stereo cuVSLAM tracker publishing the ``map → odom`` TF edge."""

        def __init__(self) -> None:
            super().__init__("openral_pycuvslam")
            self.declare_parameter("left_image_topic", "/openral/cameras/left/image")
            self.declare_parameter("left_camera_info_topic", "/openral/cameras/left/camera_info")
            self.declare_parameter("right_image_topic", "/openral/cameras/right/image")
            self.declare_parameter("right_camera_info_topic", "/openral/cameras/right/camera_info")
            self.declare_parameter("map_frame", "map")
            self.declare_parameter("odom_frame", "odom")
            self.declare_parameter("odometry_topic", "/openral/visual_slam/odometry")
            self.declare_parameter("sync_slop_s", 0.01)
            self.declare_parameter("tf_timeout_ms", 50)
            self.declare_parameter("enable_slam", True)

            try:
                import cuvslam
            except ImportError as exc:
                from openral_core import ROSConfigError

                raise ROSConfigError(
                    "PyCuVSLAM is not installed. It is an NVIDIA-licensed engine OpenRAL "
                    "does not bundle — install the matching wheel from "
                    "https://github.com/nvidia-isaac/cuVSLAM/releases on this GPU host "
                    "(see packages/openral_slam_bringup/README.md)."
                ) from exc
            self._cuvslam = cuvslam

            gp = self.get_parameter
            self._map_frame = gp("map_frame").get_parameter_value().string_value
            self._odom_frame = gp("odom_frame").get_parameter_value().string_value
            self._tf_timeout = Duration(
                seconds=gp("tf_timeout_ms").get_parameter_value().integer_value / 1000.0
            )
            self._enable_slam = gp("enable_slam").get_parameter_value().bool_value
            self._tracker: Any = None
            self._infos: dict[str, Any] = {}
            self._lost_warned = False

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_broadcaster = TransformBroadcaster(self)

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
            self._odom_pub = self.create_publisher(
                Odometry, gp("odometry_topic").get_parameter_value().string_value, sensor_qos
            )
            for side in ("left", "right"):
                self.create_subscription(
                    CameraInfo,
                    gp(f"{side}_camera_info_topic").get_parameter_value().string_value,
                    lambda msg, side=side: self._infos.setdefault(side, msg),
                    info_qos,
                )
            self._sync = ApproximateTimeSynchronizer(
                [
                    Subscriber(
                        self,
                        Image,
                        gp("left_image_topic").get_parameter_value().string_value,
                        qos_profile=sensor_qos,
                    ),
                    Subscriber(
                        self,
                        Image,
                        gp("right_image_topic").get_parameter_value().string_value,
                        qos_profile=sensor_qos,
                    ),
                ],
                queue_size=5,
                slop=gp("sync_slop_s").get_parameter_value().double_value,
            )
            self._sync.registerCallback(self._on_stereo_pair)
            self.get_logger().info(
                f"pycuvslam: engine {cuvslam.get_version()[0]}, waiting for stereo pair on "
                f"{gp('left_image_topic').get_parameter_value().string_value} + "
                f"{gp('right_image_topic').get_parameter_value().string_value}"
            )

        def _build_tracker(self, left_info: Any, right_info: Any) -> Any:
            cuvslam = self._cuvslam
            baseline = stereo_baseline_m(right_info.p)

            def camera(info: Any, tx: float) -> Any:
                return cuvslam.Camera(
                    size=[int(info.width), int(info.height)],
                    principal=[float(info.k[2]), float(info.k[5])],
                    focal=[float(info.k[0]), float(info.k[4])],
                    rig_from_camera=cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[tx, 0, 0]),
                )

            rig = cuvslam.Rig()
            # Rig frame ≡ left camera optical frame; rectified pair → pure
            # x-baseline extrinsic, distortion left at the pinhole default.
            rig.cameras = [camera(left_info, 0.0), camera(right_info, baseline)]
            slam_config = cuvslam.Tracker.SlamConfig() if self._enable_slam else None
            self.get_logger().info(
                f"pycuvslam: stereo rig {left_info.width}x{left_info.height}, "
                f"baseline {baseline:.4f} m, slam={'on' if self._enable_slam else 'off'}"
            )
            return cuvslam.Tracker(rig, slam_config=slam_config)

        def _on_stereo_pair(self, left: Any, right: Any) -> None:
            if self._tracker is None:
                if "left" not in self._infos or "right" not in self._infos:
                    return  # camera_info not seen yet — RELIABLE, arrives promptly
                try:
                    self._tracker = self._build_tracker(self._infos["left"], self._infos["right"])
                except ValueError as exc:
                    # Bad calibration is fatal, not per-frame noise.
                    self.get_logger().error(f"pycuvslam: cannot build rig: {exc}")
                    raise

            try:
                images = [_image_to_array(left), _image_to_array(right)]
            except ValueError as exc:
                self.get_logger().warning(f"skip stereo pair: {exc}")
                return
            stamp_ns = int(left.header.stamp.sec) * 1_000_000_000 + int(left.header.stamp.nanosec)
            pose_est, _ = self._tracker.track(stamp_ns, images)
            if pose_est.world_from_rig is None:
                if not self._lost_warned:
                    self.get_logger().warning("pycuvslam: tracker lost — no pose this frame")
                    self._lost_warned = True
                return
            self._lost_warned = False

            tracked = pose_est.world_from_rig.pose
            rot, tr = tracked.rotation, tracked.translation
            map_from_rig: _Pose = (
                (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
                (float(tr[0]), float(tr[1]), float(tr[2])),
            )
            self._publish_odometry(left, map_from_rig)
            try:
                odom_tf = self._tf_buffer.lookup_transform(
                    self._odom_frame,
                    left.header.frame_id,
                    Time.from_msg(left.header.stamp),
                    timeout=self._tf_timeout,
                )
            except TransformException as exc:
                self.get_logger().warning(f"skip map→odom broadcast: TF lookup failed: {exc}")
                return
            oq = odom_tf.transform.rotation
            ot = odom_tf.transform.translation
            correction = map_from_odom(map_from_rig, ((oq.x, oq.y, oq.z, oq.w), (ot.x, ot.y, ot.z)))
            out = TransformStamped()
            out.header.stamp = left.header.stamp
            out.header.frame_id = self._map_frame
            out.child_frame_id = self._odom_frame
            (
                out.transform.rotation.x,
                out.transform.rotation.y,
                out.transform.rotation.z,
                out.transform.rotation.w,
            ) = correction[0]
            (
                out.transform.translation.x,
                out.transform.translation.y,
                out.transform.translation.z,
            ) = correction[1]
            self._tf_broadcaster.sendTransform(out)

        def _publish_odometry(self, left: Any, map_from_rig: _Pose) -> None:
            msg = Odometry()
            msg.header.stamp = left.header.stamp
            msg.header.frame_id = self._map_frame
            msg.child_frame_id = left.header.frame_id
            (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ) = map_from_rig[0]
            (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ) = map_from_rig[1]
            self._odom_pub.publish(msg)

    rclpy.init(args=args)
    node = PyCuVSLAMNode()
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
