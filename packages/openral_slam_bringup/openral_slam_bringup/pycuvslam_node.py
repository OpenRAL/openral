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
image pair from the OpenRAL camera bus, tracks with ``cuvslam.Tracker``,
and fills the same ``map → odom`` TF edge the other SLAM backends fill
(composing the tracker's ``map ← rig`` pose with the live ``odom ← rig``
TF). Two rig modes:

* **Multi-camera** (``rig_frame`` set, or derived from ``robot_yaml``'s
  ``base_frame``): each camera's ``rig_from_camera`` extrinsic is read
  from TF, so an arbitrary base-mounted rig — e.g. the toed-in sim
  cameras of a mobile robot — works without a rectified pair. This is
  cuVSLAM's default mode (``odometry_mode=Multicamera``).
* **Rectified baseline** (neither set): the rig is the left camera
  optical frame and the right camera sits at a pure x-baseline read from
  its projection matrix ``P`` — a standalone RealSense-style rectified
  pair.

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
    "depth_to_uint16_mm",
    "invert_pose",
    "main",
    "map_from_odom",
    "stereo_baseline_m",
    "transform_to_pose",
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


def transform_to_pose(transform: Any) -> _Pose:
    """Convert a ``geometry_msgs/Transform`` to the node's ``(quat, translation)`` pose.

    ``lookup_transform(rig, camera)`` returns the pose of ``camera`` expressed in
    ``rig`` — i.e. cuVSLAM's ``rig_from_camera`` — so this is how a multi-camera
    rig's per-camera extrinsics are read straight from TF (no rectified-baseline
    assumption). Both use the xyzw quaternion convention.

    Example:
        >>> class _T:  # minimal geometry_msgs/Transform stand-in
        ...     class rotation:
        ...         x, y, z, w = 0.0, 0.0, 0.0, 1.0
        ...
        ...     class translation:
        ...         x, y, z = 0.06, 0.0, 0.0
        >>> transform_to_pose(_T())
        ((0.0, 0.0, 0.0, 1.0), (0.06, 0.0, 0.0))
    """
    q = transform.rotation
    t = transform.translation
    return (
        (float(q.x), float(q.y), float(q.z), float(q.w)),
        (float(t.x), float(t.y), float(t.z)),
    )


def depth_to_uint16_mm(msg: Any, width: int, height: int, scale: float) -> Any:
    """Convert a ``32FC1`` metric-depth ``Image`` to the ``uint16`` cuVSLAM RGBD eats.

    cuVSLAM's RGBD odometry expects depth as ``uint16`` aligned pixel-for-pixel
    with camera-0 (the RGB image), and divides each raw value by
    ``depth_scale_factor`` to recover metres. The metric-depth provider (DA3)
    publishes ``32FC1`` metres at the model's own resolution, so this resizes to
    the RGB ``width``/``height`` (bilinear) and encodes ``metres * scale`` — the
    inverse of cuVSLAM's divide, so ``scale`` (e.g. ``1000`` for millimetres) is
    the single knob shared by both sides. Values are clipped to the ``uint16``
    range; the RealSense reference example uses the same ``1/depth_scale``
    convention.

    Example:
        >>> import numpy as np
        >>> class _D:  # minimal 32FC1 sensor_msgs/Image stand-in, 2x2 metres
        ...     encoding = "32FC1"
        ...     height, width = 2, 2
        ...     data = np.array([[0.5, 1.0], [1.5, 2.0]], np.float32).tobytes()
        >>> depth_to_uint16_mm(_D(), 2, 2, 1000.0).tolist()
        [[500, 1000], [1500, 2000]]
    """
    import numpy as np

    if msg.encoding != "32FC1":
        raise ValueError(f"depth must be 32FC1 metres, got encoding {msg.encoding!r}")
    depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
        int(msg.height), int(msg.width)
    )
    if (int(msg.height), int(msg.width)) != (height, width):
        # Register DA3's native-resolution depth onto the RGB grid so cuVSLAM's
        # per-pixel depth_camera_id=0 alignment holds. PIL "F" = float32 bilinear.
        from PIL import Image as _PILImage

        depth = np.asarray(
            _PILImage.fromarray(depth, mode="F").resize((width, height), _PILImage.BILINEAR),
            dtype=np.float32,
        )
    return np.clip(depth * scale, 0, 65535).astype(np.uint16)


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
            # Multi-camera rig frame. When set (or derived from robot_yaml's
            # base_frame), the rig is built from each camera's real
            # rig_from_camera extrinsic looked up from TF — cuVSLAM's default
            # multi-camera mode, which handles arbitrary (e.g. toed-in) rigs, not
            # just a rectified stereo pair. Empty → the standalone rectified path
            # (rig ≡ left camera, baseline from the right CameraInfo P matrix).
            self.declare_parameter("rig_frame", "")
            self.declare_parameter("robot_yaml", "")
            # Mono RGBD mode: when depth_image_topic is set, the node tracks a
            # SINGLE RGB camera (left_image_topic) fused with a metric-depth
            # stream (the DA3 depth provider) via cuVSLAM's RGBD odometry — the
            # one-camera path for lidar-less robots without a stereo rig. Empty →
            # the stereo/multi-camera path above. depth_scale_factor is the
            # shared metres↔uint16 knob (1000 = millimetres); see
            # depth_to_uint16_mm.
            self.declare_parameter("depth_image_topic", "")
            self.declare_parameter("depth_scale_factor", 1000.0)

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
            self._depth_topic = gp("depth_image_topic").get_parameter_value().string_value
            self._depth_scale = gp("depth_scale_factor").get_parameter_value().double_value
            # Resolve the rig frame: explicit param wins; else the robot
            # manifest's base_frame (the natural rig for a mobile robot whose
            # cameras are mounted on its base). Empty → rectified-baseline path.
            self._rig_frame = gp("rig_frame").get_parameter_value().string_value
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            if not self._rig_frame and robot_yaml:
                from openral_core import RobotDescription

                self._rig_frame = RobotDescription.from_yaml(robot_yaml).base_frame
            # The frame the tracked pose is expressed in (for the odom lookup +
            # odometry child frame): the rig frame in multi-camera mode, else the
            # left camera optical frame — set when the tracker is built.
            self._rig_child_frame = self._rig_frame
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
            self._setup_inputs(sensor_qos, info_qos)

        def _setup_inputs(self, sensor_qos: Any, info_qos: Any) -> None:
            """Subscribe camera_info + the synced image pair (mono RGBD or stereo)."""
            cuvslam = self._cuvslam
            gp = self.get_parameter
            left_image = gp("left_image_topic").get_parameter_value().string_value
            self.create_subscription(
                CameraInfo,
                gp("left_camera_info_topic").get_parameter_value().string_value,
                lambda msg: self._infos.setdefault("left", msg),
                info_qos,
            )
            if self._depth_topic:
                # Mono RGBD: one RGB camera (left) + a metric-depth stream. Sync
                # the RGB frame with its depth image; cuVSLAM fuses them for scale.
                self._sync = ApproximateTimeSynchronizer(
                    [
                        Subscriber(self, Image, left_image, qos_profile=sensor_qos),
                        Subscriber(self, Image, self._depth_topic, qos_profile=sensor_qos),
                    ],
                    queue_size=5,
                    slop=gp("sync_slop_s").get_parameter_value().double_value,
                )
                self._sync.registerCallback(self._on_rgbd)
                self.get_logger().info(
                    f"pycuvslam: engine {cuvslam.get_version()[0]}, mono RGBD, waiting for "
                    f"{left_image} + depth {self._depth_topic}"
                )
            else:
                self.create_subscription(
                    CameraInfo,
                    gp("right_camera_info_topic").get_parameter_value().string_value,
                    lambda msg: self._infos.setdefault("right", msg),
                    info_qos,
                )
                self._sync = ApproximateTimeSynchronizer(
                    [
                        Subscriber(self, Image, left_image, qos_profile=sensor_qos),
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
                    f"{left_image} + "
                    f"{gp('right_image_topic').get_parameter_value().string_value}"
                )

        def _intrinsics_camera(self, info: Any, rig_from_camera: Any) -> Any:
            cuvslam = self._cuvslam
            return cuvslam.Camera(
                size=[int(info.width), int(info.height)],
                principal=[float(info.k[2]), float(info.k[5])],
                focal=[float(info.k[0]), float(info.k[4])],
                rig_from_camera=rig_from_camera,
            )

        def _build_tracker(self, left_info: Any, right_info: Any, stamp: Any) -> Any:
            cuvslam = self._cuvslam
            if self._rig_frame:
                # Multi-camera mode: each camera's rig_from_camera is its real
                # pose in the rig frame, read from TF — handles the arbitrary
                # (toed-in, wide-baseline) rigs sim robots expose, not just a
                # rectified pair. Raises TransformException if TF isn't ready yet
                # (caller retries on the next frame).
                cameras = []
                for info in (left_info, right_info):
                    tf = self._tf_buffer.lookup_transform(
                        self._rig_frame, info.header.frame_id, stamp, timeout=self._tf_timeout
                    )
                    q, t = transform_to_pose(tf.transform)
                    pose = cuvslam.Pose(rotation=list(q), translation=list(t))
                    cameras.append(self._intrinsics_camera(info, pose))
                self._rig_child_frame = self._rig_frame
                mode = f"multicamera rig={self._rig_frame}"
            else:
                # Standalone rectified path: rig ≡ left camera; the right camera
                # sits at a pure x-baseline read from its CameraInfo P matrix.
                baseline = stereo_baseline_m(right_info.p)
                cameras = [
                    self._intrinsics_camera(
                        left_info, cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
                    ),
                    self._intrinsics_camera(
                        right_info,
                        cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[baseline, 0, 0]),
                    ),
                ]
                self._rig_child_frame = left_info.header.frame_id
                mode = f"rectified baseline={baseline:.4f}m"

            rig = cuvslam.Rig()
            rig.cameras = cameras
            slam_config = cuvslam.Tracker.SlamConfig() if self._enable_slam else None
            self.get_logger().info(
                f"pycuvslam: {left_info.width}x{left_info.height} {mode}, "
                f"slam={'on' if self._enable_slam else 'off'}"
            )
            return cuvslam.Tracker(rig, slam_config=slam_config)

        def _on_stereo_pair(self, left: Any, right: Any) -> None:
            if self._tracker is None:
                if "left" not in self._infos or "right" not in self._infos:
                    return  # camera_info not seen yet — RELIABLE, arrives promptly
                try:
                    self._tracker = self._build_tracker(
                        self._infos["left"], self._infos["right"], Time.from_msg(left.header.stamp)
                    )
                except TransformException as exc:
                    # Multi-camera rig needs the cameras' TF; it may not have
                    # buffered on the first frame — retry on the next one.
                    self.get_logger().warning(f"pycuvslam: rig TF not ready, retrying: {exc}")
                    return
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
            self._emit(left.header.stamp, map_from_rig)

        def _build_mono_tracker(self, rgb_info: Any) -> Any:
            """Single-camera RGBD tracker: rig ≡ the RGB camera, depth gives scale."""
            cuvslam = self._cuvslam
            rig = cuvslam.Rig()
            rig.cameras = [
                self._intrinsics_camera(
                    rgb_info, cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
                )
            ]
            self._rig_child_frame = rgb_info.header.frame_id
            # ponytail: odometry-only (no SlamConfig loop closure). Add slam_config
            # if mono drift over a long run matters — nvblox only needs the pose.
            odom_config = cuvslam.Tracker.OdometryConfig(
                odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
                rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(
                    depth_camera_id=0, depth_scale_factor=self._depth_scale
                ),
            )
            self.get_logger().info(
                f"pycuvslam: {rgb_info.width}x{rgb_info.height} mono RGBD "
                f"(depth scale={self._depth_scale:g})"
            )
            return cuvslam.Tracker(rig, odom_config=odom_config)

        def _on_rgbd(self, rgb: Any, depth: Any) -> None:
            if self._tracker is None:
                if "left" not in self._infos:
                    return  # camera_info not seen yet — RELIABLE, arrives promptly
                self._tracker = self._build_mono_tracker(self._infos["left"])
            try:
                rgb_arr = _image_to_array(rgb)
                depth_mm = depth_to_uint16_mm(
                    depth, int(rgb.width), int(rgb.height), self._depth_scale
                )
            except ValueError as exc:
                self.get_logger().warning(f"skip rgbd frame: {exc}")
                return
            stamp_ns = int(rgb.header.stamp.sec) * 1_000_000_000 + int(rgb.header.stamp.nanosec)
            pose_est, _ = self._tracker.track(stamp_ns, [rgb_arr], depths=[depth_mm])
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
            self._emit(rgb.header.stamp, map_from_rig)

        def _emit(self, stamp: Any, map_from_rig: _Pose) -> None:
            """Publish odometry and broadcast the ``map → odom`` TF for a tracked pose."""
            self._publish_odometry(stamp, map_from_rig)
            try:
                odom_tf = self._tf_buffer.lookup_transform(
                    self._odom_frame,
                    self._rig_child_frame,
                    Time.from_msg(stamp),
                    timeout=self._tf_timeout,
                )
            except TransformException as exc:
                self.get_logger().warning(f"skip map→odom broadcast: TF lookup failed: {exc}")
                return
            oq = odom_tf.transform.rotation
            ot = odom_tf.transform.translation
            correction = map_from_odom(map_from_rig, ((oq.x, oq.y, oq.z, oq.w), (ot.x, ot.y, ot.z)))
            out = TransformStamped()
            out.header.stamp = stamp
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

        def _publish_odometry(self, stamp: Any, map_from_rig: _Pose) -> None:
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = self._map_frame
            msg.child_frame_id = self._rig_child_frame
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
