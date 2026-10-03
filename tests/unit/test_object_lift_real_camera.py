# SPDX-License-Identifier: Apache-2.0
"""The object lift projects a real camera's detections through the driver's frame and K.

On the real OpenArm (Thor cell) the detector watches ``top``, which the manifest declares as
the sim render's stand-in (frame ``world``, 640x480, fx 640) while the real image is the ZED's
rectified left image: ``zed_left_camera_frame_optical``, 1920x1080, K = [1498.18, 0, 936.11;
0, 1498.18, 541.81] (measured 2026-10-02). The detector now stamps each batch with the
driver ``CameraInfo``'s frame and K; this pins that a 1920x1080 box lifted through them lands
on the right point in ``openarm_base`` under the Thor overlay's calibrated mount.

Real fixtures (CLAUDE.md §1.11): ``robots/openarm/robot.yaml`` + ``units/thor.yaml``, real
``sensor_msgs/CameraInfo`` (skipped without ROS), real ``VoxelFrustumLifter``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    IntrinsicsPinhole,
    ObjectDetection2D,
    ObjectsMetadata,
    RobotDescription,
    ROSConfigError,
    apply_sensor_overlays,
    resolve_sensor_overlays,
)
from openral_perception_ros.ros_image_detector_node import (
    camera_geometry_from_info,
    stamp_camera_geometry,
)
from openral_world_state import VoxelFrustumLifter
from pydantic import ValidationError

_ROBOT_YAML = "robots/openarm/robot.yaml"
_OPTICAL = "zed_left_camera_frame_optical"
_THOR_K = IntrinsicsPinhole(width=1920, height=1080, fx=1498.18, fy=1498.18, cx=936.11, cy=541.81)
# An object on the bench in front of the arms, near the ZED's optical axis.
_OBJECT_BASE = np.array([0.25, 0.05, -0.40])


def _rot_rpy(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Fixed-axis RPY (tf2 / URDF): Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
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
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def _t_base_optical() -> NDArray[np.float64]:
    """openarm_base <- zed_left_camera_frame_optical: Thor's calibrated mount, then REP-103.

    The driver's ``/tf_static`` also offsets the left lens from ``zed_camera_link``; that is
    a translation tf2 composes the same way, omitted here (not a value OpenRAL records).
    """
    arm = RobotDescription.from_yaml(_ROBOT_YAML)
    thor = resolve_sensor_overlays(_ROBOT_YAML, "thor", required=True)
    zed = next(s for s in apply_sensor_overlays(arm.sensors, thor) if s.name == "head_zed")
    assert zed.parent_frame == "openarm_base"
    mount = zed.static_transform_xyz_rpy
    assert mount is not None
    t_base_body = np.eye(4)
    t_base_body[:3, :3] = _rot_rpy(*mount[3:])
    t_base_body[:3, 3] = mount[:3]
    t_body_opt = np.eye(4)
    t_body_opt[:3, :3] = _rot_rpy(-math.pi / 2, 0.0, -math.pi / 2)
    return np.asarray(t_base_body @ t_body_opt, dtype=np.float64)


def _cube(center: NDArray[np.float64], res: float = 0.02, n: int = 5) -> NDArray[np.float64]:
    offs = (np.arange(n) - (n - 1) / 2) * res
    return np.array([center + np.array([x, y, z]) for x in offs for y in offs for z in offs])


@pytest.mark.parametrize("frame_size", [(1920, 1080), (960, 540)])
def test_a_zed_box_lifts_to_the_object_in_openarm_base(frame_size: tuple[int, int]) -> None:
    """Through the driver's frame + K the box lands on the object (also on a downscaled frame,
    which is what the sensor leg's republish can hand the detector)."""
    t_cam_from_base = np.asarray(np.linalg.inv(_t_base_optical()), dtype=np.float64)
    p = t_cam_from_base @ np.append(_OBJECT_BASE, 1.0)
    assert p[2] > 0.3, "the object must be in front of the ZED"
    u = _THOR_K.fx * p[0] / p[2] + _THOR_K.cx
    v = _THOR_K.fy * p[1] / p[2] + _THOR_K.cy
    assert 0 < u < 1920 and 0 < v < 1080, (u, v)
    sx, sy = frame_size[0] / 1920, frame_size[1] / 1080
    box = (int((u - 60) * sx), int((v - 60) * sy), int((u + 60) * sx), int((v + 60) * sy))

    (obj,) = VoxelFrustumLifter().lift(
        detections=[ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=box)],
        occupied_centers_base=_cube(_OBJECT_BASE),
        intrinsics=_THOR_K,
        frame_size=frame_size,
        t_cam_from_base=t_cam_from_base,
        t_map_from_base=np.eye(4),
        map_frame="openarm_base",
    )
    assert obj.pose.frame_id == "openarm_base"
    assert np.linalg.norm(np.array(obj.pose.xyz) - _OBJECT_BASE) < 0.02, obj.pose.xyz


def test_the_detector_stamps_the_driver_frame_and_k() -> None:
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    info = sensor_msgs.CameraInfo()
    info.header.frame_id = _OPTICAL
    info.width, info.height = 1920, 1080
    info.k = [1498.18, 0.0, 936.11, 0.0, 1498.18, 541.81, 0.0, 0.0, 1.0]
    geometry = camera_geometry_from_info(info)
    assert geometry == (_OPTICAL, geometry[1])
    assert (geometry[1].fx, geometry[1].cx, geometry[1].cy) == (1498.18, 936.11, 541.81)

    md = ObjectsMetadata(
        sensor_id="top", detections=[], model_id="rtdetr", frame_width=1920, frame_height=1080
    )
    wire = ObjectsMetadata.model_validate_json(
        stamp_camera_geometry(md, geometry).model_dump_json()
    )
    assert wire.camera_frame_id == _OPTICAL
    assert wire.camera_intrinsics == geometry[1]
    assert stamp_camera_geometry(md, None) == md  # no CameraInfo yet: manifest fallback

    info.header.frame_id = ""
    with pytest.raises(ROSConfigError, match=r"no header\.frame_id"):
        camera_geometry_from_info(info)


def test_a_frame_without_its_k_is_refused() -> None:
    with pytest.raises(ValidationError, match="set together"):
        ObjectsMetadata(
            sensor_id="top",
            detections=[],
            model_id="m",
            frame_width=1,
            frame_height=1,
            camera_frame_id=_OPTICAL,
        )
