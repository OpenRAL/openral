"""CameraInfo shared by both real-camera ROS paths (opencv_thread + GStreamer tee).

``openral_sensors.ros_publisher.build_camera_info_msg`` is the single builder the
polling ``SensorRosPublisher`` and the GStreamer reader's in-pipeline
``RosImagePublisher`` both call, so the two paths cannot drift. Intrinsics come
from a real manifest (``robots/so101_follower/robot.yaml``), per CLAUDE.md §1.11.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    IntrinsicsPinhole,
    RobotDescription,
    SensorReaderBackend,
    SensorReaderConfig,
)
from openral_sensors.ros_publisher import camera_info_topic_for

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "so101_follower" / "robot.yaml"


def _manifest_intrinsics() -> IntrinsicsPinhole:
    desc = RobotDescription.from_yaml(str(_MANIFEST))
    return next(s.intrinsics for s in desc.sensors if s.intrinsics is not None)


def test_camera_info_topic_is_the_image_topics_sibling() -> None:
    """OpenRAL layout → sibling; anything else → camera_info_manager's ``<topic>/camera_info``."""
    assert camera_info_topic_for("/openral/cameras/wrist/image") == (
        "/openral/cameras/wrist/camera_info"
    )
    assert camera_info_topic_for("/cameras/wrist_rgb/image_raw") == (
        "/cameras/wrist_rgb/camera_info"
    )
    assert camera_info_topic_for("/camera/color/image_rect_color") == "/camera/color/camera_info"
    # realsense2_camera's depth stream is image_rect_raw; its CameraInfo is the sibling.
    assert camera_info_topic_for("/head/depth/image_rect_raw") == "/head/depth/camera_info"
    assert camera_info_topic_for("/zed/left/rgb") == "/zed/left/rgb/camera_info"
    # Only a trailing ``/image`` segment is replaced, never an inner one.
    assert camera_info_topic_for("/image/cam/image") == "/image/cam/camera_info"


def test_build_camera_info_msg_scales_manifest_intrinsics() -> None:
    """k/p/d/width/height/frame match what ``SensorRosPublisher`` has always published."""
    pytest.importorskip("sensor_msgs", reason="rclpy / sensor_msgs not on PYTHONPATH")
    from openral_sensors.ros_publisher import build_camera_info_msg

    intr = _manifest_intrinsics()
    assert (intr.width, intr.height, intr.fx, intr.cx, intr.cy) == (640, 480, 480.0, 320.0, 240.0)

    full = build_camera_info_msg(intr, width=640, height=480, stamp=None, frame_id="cam_optical")
    assert (full.width, full.height) == (640, 480)
    assert full.header.frame_id == "cam_optical"
    assert full.distortion_model == intr.distortion_model
    assert list(full.d) == list(intr.distortion_coeffs)
    assert list(full.k) == [480.0, 0.0, 320.0, 0.0, 480.0, 240.0, 0.0, 0.0, 1.0]
    assert list(full.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert list(full.p) == [480.0, 0.0, 320.0, 0.0, 0.0, 480.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    half = build_camera_info_msg(intr, width=320, height=240, stamp=None, frame_id="cam_optical")
    assert (half.width, half.height) == (320, 240)
    assert list(half.k) == [240.0, 0.0, 160.0, 0.0, 240.0, 120.0, 0.0, 0.0, 1.0]


def test_sensor_reader_config_carries_tee_frame_and_intrinsics() -> None:
    """The GStreamer tee's frame id + intrinsics round-trip through the config."""
    intr = _manifest_intrinsics()
    cfg = SensorReaderConfig(
        sensor_id="wrist",
        backend=SensorReaderBackend.GSTREAMER,
        backend_params={"source": "testsrc"},
        publish_to_ros=True,
        publish_topic="/openral/cameras/wrist/image",
        publish_frame_id="wrist_camera_optical_frame",
        publish_camera_info=intr,
    )
    again = SensorReaderConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg
    assert again.publish_camera_info == intr
    # Absent → images only, frame falls back to the sensor id (backward compatible).
    bare = SensorReaderConfig(sensor_id="wrist")
    assert bare.publish_frame_id is None and bare.publish_camera_info is None


def test_sensor_reader_config_rejects_tee_fields_without_the_tee() -> None:
    """Frame id / intrinsics only configure the ROS tee; setting them without it is a bug."""
    with pytest.raises(ValueError, match="publish_to_ros is False"):
        SensorReaderConfig(sensor_id="wrist", publish_camera_info=_manifest_intrinsics())
    with pytest.raises(ValueError, match="publish_to_ros is False"):
        SensorReaderConfig(sensor_id="wrist", publish_frame_id="wrist_optical")


def test_ros_tee_is_refused_on_a_backend_without_one() -> None:
    """Only the gstreamer backend has an in-pipeline ROS tee; others must not accept the flag."""
    from openral_core import SensorReaderBackend

    with pytest.raises(ValueError, match="needs the gstreamer backend"):
        SensorReaderConfig(
            sensor_id="wrist",
            backend=SensorReaderBackend.OPENCV_THREAD,
            publish_to_ros=True,
            publish_topic="/openral/cameras/wrist/image",
        )


def test_ros_tee_releases_a_partial_start() -> None:
    """A tee whose start fails after its node and publishers exist leaves nothing behind.

    The appsink here cannot be hooked (a plain object has no ``set_property``), so
    ``start`` fails at its last step — after the node, the Image publisher and the
    CameraInfo publisher were created. They must be released and, when the tee
    initialised rclpy itself, rclpy shut down again.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("sensor_msgs")
    from openral_runner.backends.gstreamer.ros_tee import RosImagePublisher

    owned_before = rclpy.ok()
    tee = RosImagePublisher(
        sensor_id="wrist",
        appsink=object(),
        topic="/openral/cameras/wrist/image",
        camera_info=_manifest_intrinsics(),
    )
    with pytest.raises(AttributeError):
        tee.start()
    assert tee._node is None
    assert tee._publisher is None and tee._info_publisher is None
    assert not tee.is_started
    assert rclpy.ok() is owned_before
