"""ADR-0064 — metric-depth ndarray <-> sensor_msgs/Image (32FC1) conversion.

openral_perception_ros is a colcon-built ROS package; like the other ROS-package
unit tests, skip cleanly when the workspace overlay / sensor_msgs isn't sourced.
The ros2-test CI job sources install/setup.bash and runs this for real.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openral_perception_ros")
pytest.importorskip("sensor_msgs")

from openral_perception_ros.depth_convert import (
    DepthConvertError,
    camera_info_from_intrinsics,
    depth_array_to_image_msg,
    image_msg_to_depth_array,
)


def test_depth_round_trips_in_metres_with_nan_preserved():
    depth = np.array([[0.5, 1.25, np.nan], [2.0, 3.5, 0.0]], dtype=np.float32)
    msg = depth_array_to_image_msg(depth, frame_id="cam_optical")
    assert msg.encoding == "32FC1"
    assert (msg.width, msg.height) == (3, 2)
    assert msg.step == 3 * 4
    assert msg.header.frame_id == "cam_optical"
    out = image_msg_to_depth_array(msg)
    # Finite metres preserved exactly (float32, no scaling); NaN stays NaN.
    assert np.allclose(out[~np.isnan(out)], depth[~np.isnan(depth)])
    assert np.isnan(out[0, 2])


def test_depth_rejects_non_2d():
    with pytest.raises(DepthConvertError):
        depth_array_to_image_msg(np.zeros((2, 2, 3), dtype=np.float32), frame_id="c")


def test_decode_rejects_wrong_encoding():
    msg = depth_array_to_image_msg(np.zeros((2, 2), dtype=np.float32), frame_id="c")
    msg.encoding = "mono16"
    with pytest.raises(DepthConvertError):
        image_msg_to_depth_array(msg)


def test_decode_rejects_padded_rows():
    msg = depth_array_to_image_msg(np.zeros((2, 2), dtype=np.float32), frame_id="c")
    msg.step = 999  # not width*4
    with pytest.raises(DepthConvertError):
        image_msg_to_depth_array(msg)


def test_camera_info_pinhole_matrices():
    info = camera_info_from_intrinsics(
        fx=466.0, fy=467.2, cx=252.0, cy=189.0, width=504, height=378, frame_id="cam_optical"
    )
    assert (info.width, info.height) == (504, 378)
    assert info.distortion_model == "plumb_bob"
    assert list(info.d) == [0.0, 0.0, 0.0, 0.0, 0.0]
    # K = [fx 0 cx; 0 fy cy; 0 0 1]
    assert info.k[0] == 466.0 and info.k[4] == 467.2
    assert info.k[2] == 252.0 and info.k[5] == 189.0
    assert info.k[8] == 1.0
    # P focal/principal mirror K; translation column zero (monocular).
    assert info.p[0] == 466.0 and info.p[5] == 467.2 and info.p[3] == 0.0
