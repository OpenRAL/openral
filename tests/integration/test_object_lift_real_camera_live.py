# SPDX-License-Identifier: Apache-2.0
"""Real-camera detections reach ``/openral/world_state_slow`` lifted in ``openarm_base``.

The Thor OpenArm's detector watches ``top``, whose manifest frame/K are the sim render's
stand-in (``world``, 640x480, fx 640) while the real image is the ZED's left rectified
1920x1080 in ``zed_left_camera_frame_optical`` at fx 1498.18. The detector stamps each batch
with its driver ``CameraInfo``'s frame and K (``stamp_camera_geometry``); the world-state
lift must project through those and publish the box in the fixed-base map frame.

Real components (CLAUDE.md §1.11): the real ``_WorldStateLifecycleNode`` on the real OpenArm
manifest with the Thor unit overlay, real tf2 carrying the overlay's calibrated mount plus the
driver's REP-103 ``zed_camera_link -> zed_left_camera_frame_optical`` hop, a real
``OccupancyVoxels`` grid, a real ``sensor_msgs/CameraInfo`` turned into the detector's
``ObjectsMetadata`` by the detector's own helper, and a real subscriber on the slow topic.
The detector *model* is not run (it would only produce the box this test places): the batch
is published exactly as the node publishes it, at the process boundary.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("tf2_ros")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ROS_DISTRO") and importlib.util.find_spec("openral_msgs")),
    reason="needs a sourced ROS 2 install with the colcon-built openral_msgs overlay",
)

_ROBOT_YAML = "robots/openarm/robot.yaml"
_BASE = "openarm_base"
_OPTICAL = "zed_left_camera_frame_optical"
_K = [1498.18, 0.0, 936.11, 0.0, 1498.18, 541.81, 0.0, 0.0, 1.0]
_OBJECT_BASE = np.array([0.25, 0.05, -0.40])
_RES = 0.02
_N = 5


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    return (
        (rot[2, 1] - rot[1, 2]) / (4 * w),
        (rot[0, 2] - rot[2, 0]) / (4 * w),
        (rot[1, 0] - rot[0, 1]) / (4 * w),
        w,
    )


def _tf(parent: str, child: str, rot: np.ndarray, xyz: Any) -> Any:
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = parent, child
    t = msg.transform
    t.translation.x, t.translation.y, t.translation.z = (float(v) for v in xyz)
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = _quat_xyzw(rot)
    return msg


def test_a_zed_detection_is_published_lifted_in_openarm_base() -> None:
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import Point, Quaternion
    from openral_core import (
        ObjectDetection2D,
        ObjectsMetadata,
        RobotDescription,
        apply_sensor_overlays,
        resolve_sensor_overlays,
    )
    from openral_msgs.msg import OccupancyVoxels, PromptStamped, WorldStateStamped
    from openral_perception_ros.ros_image_detector_node import (
        camera_geometry_from_info,
        stamp_camera_geometry,
    )
    from openral_world_state import WorldStateAggregator
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo

    arm = RobotDescription.from_yaml(_ROBOT_YAML)
    thor = resolve_sensor_overlays(_ROBOT_YAML, "thor", required=True)
    arm = arm.model_copy(update={"sensors": apply_sensor_overlays(arm.sensors, thor)})
    zed = next(s for s in arm.sensors if s.name == "head_zed")
    assert arm.base_frame == _BASE and zed.parent_frame == _BASE
    mount = zed.static_transform_xyz_rpy
    assert mount is not None
    r_base_body = _rot_rpy(*mount[3:])
    r_body_opt = _rot_rpy(-math.pi / 2, 0.0, -math.pi / 2)

    # Where the object appears in the 1920x1080 ZED image.
    r_base_opt = r_base_body @ r_body_opt
    p = r_base_opt.T @ (_OBJECT_BASE - np.asarray(mount[:3]))
    u, v = _K[0] * p[0] / p[2] + _K[2], _K[4] * p[1] / p[2] + _K[5]

    rclpy.init()
    node = _WorldStateLifecycleNode(WorldStateAggregator(arm))
    node.set_parameters(
        [
            rclpy.parameter.Parameter("publish_rate_hz_slow", value=10.0),
            rclpy.parameter.Parameter("object_lift_enabled", value=True),
            rclpy.parameter.Parameter("object_lift_memory_cadence_hz", value=20.0),
            rclpy.parameter.Parameter("object_lift_min_voxels", value=3),
            rclpy.parameter.Parameter("object_lift_max_misses", value=50),
            # What compose.py sets for a robot with no locomotion.
            rclpy.parameter.Parameter("object_lift_map_frame", value=_BASE),
        ]
    )
    helper = rclpy.create_node("test_object_lift_real_camera_helper")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    received: list[Any] = []
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        tf2_ros.StaticTransformBroadcaster(helper).sendTransform(
            [
                _tf(_BASE, "zed_camera_link", r_base_body, mount[:3]),
                _tf("zed_camera_link", _OPTICAL, r_body_opt, (0.0, 0.0, 0.0)),
            ]
        )
        helper.create_subscription(
            WorldStateStamped, "/openral/world_state_slow", received.append, 10
        )
        vox_pub = helper.create_publisher(
            OccupancyVoxels,
            "/openral/world_voxels",
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            ),
        )
        det_pub = helper.create_publisher(
            PromptStamped,
            "/openral/perception/objects",
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            ),
        )

        grid = OccupancyVoxels()
        grid.header.frame_id = _BASE
        lo = _OBJECT_BASE - _RES * _N / 2
        grid.origin = Point(x=float(lo[0]), y=float(lo[1]), z=float(lo[2]))
        grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        grid.resolution = _RES
        grid.size_x = grid.size_y = grid.size_z = _N
        grid.occupancy = bytes([1] * _N**3)

        info = CameraInfo()
        info.header.frame_id = _OPTICAL
        info.width, info.height = 1920, 1080
        info.k = _K
        md = stamp_camera_geometry(
            ObjectsMetadata(
                sensor_id="top",  # the manifest's stand-in: frame "world", fx 640
                detections=[
                    ObjectDetection2D(
                        label="cup",
                        confidence=0.9,
                        bbox_xyxy=(int(u - 60), int(v - 60), int(u + 60), int(v + 60)),
                    )
                ],
                model_id="rtdetr-coco-r18",
                frame_width=1920,
                frame_height=1080,
            ),
            camera_geometry_from_info(info),
        )
        det = PromptStamped()
        det.header.frame_id = "top"
        det.metadata_json = md.model_dump_json()

        def _lifted() -> Any:
            for msg in reversed(received):
                if "cup" in list(msg.detected_object_labels):
                    return msg
            return None

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and _lifted() is None:
            now = helper.get_clock().now().to_msg()
            grid.header.stamp = det.header.stamp = now
            vox_pub.publish(grid)
            det_pub.publish(det)
            for _ in range(10):
                executor.spin_once(timeout_sec=0.02)
        msg = _lifted()
        assert msg is not None, "no lifted 'cup' on /openral/world_state_slow within 8 s"
        i = list(msg.detected_object_labels).index("cup")
        assert msg.detected_object_frame == _BASE
        pos = msg.detected_object_positions[i]
        assert np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - _OBJECT_BASE) < 0.02
        assert msg.detected_object_bbox_valid[i]
        bmin, bmax = msg.detected_object_bbox_min[i], msg.detected_object_bbox_max[i]
        for axis, c in zip("xyz", _OBJECT_BASE, strict=True):
            assert getattr(bmin, axis) - 1e-3 <= c <= getattr(bmax, axis) + 1e-3
    finally:
        node.trigger_deactivate()
        node.trigger_cleanup()
        executor.remove_node(helper)
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
