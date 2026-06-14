"""sensor_msgs/Image -> BGR bytes conversion (no cv_bridge)."""

from __future__ import annotations

import numpy as np
import pytest

# openral_perception_ros is a colcon-built ROS package (ament_cmake); like every
# other ROS-package unit test (test_skill_runner_*, test_reasoner_palette_*),
# skip cleanly when the workspace overlay isn't sourced. The ros2-test CI job
# sources install/setup.bash and runs this for real. image_convert itself is
# pure-Python (numpy only) — the guard is about the package being on the path.
pytest.importorskip("openral_perception_ros")

from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes


class _FakeImage:
    """Duck-typed sensor_msgs/Image stand-in (real msg used in integration)."""

    def __init__(self, data, width, height, encoding, step):
        self.data = data
        self.width = width
        self.height = height
        self.encoding = encoding
        self.step = step


def _rgb_frame(h, w):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_rgb8_to_bgr_reverses_channels():
    arr = _rgb_frame(2, 2)
    msg = _FakeImage(arr.tobytes(), 2, 2, "rgb8", 2 * 3)
    out, w, h = image_to_bgr_bytes(msg)
    assert (w, h) == (2, 2)
    bgr = np.frombuffer(out, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(bgr, arr[..., ::-1])


def test_bgr8_passthrough():
    arr = _rgb_frame(2, 2)
    msg = _FakeImage(arr.tobytes(), 2, 2, "bgr8", 2 * 3)
    out, _, _ = image_to_bgr_bytes(msg)
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(2, 2, 3), arr)


def test_rejects_unsupported_encoding():
    msg = _FakeImage(b"\x00" * 12, 2, 2, "mono8", 2)
    with pytest.raises(ImageConvertError):
        image_to_bgr_bytes(msg)


def test_rejects_row_padding():
    msg = _FakeImage(b"\x00" * 16, 2, 2, "rgb8", 8)
    with pytest.raises(ImageConvertError):
        image_to_bgr_bytes(msg)
