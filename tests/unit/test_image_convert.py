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


# The ZED wrapper's colour topic: 1920x1080 bgra8. Its buffers are pitch-aligned,
# so also model a padded row (step > width*4) — the frame that reached the
# segmenter on the real OpenArm cell (2026-10-04 twin pass) and was dropped.
_ZED_W, _ZED_H = 1920, 1080


def _padded(pixels, pad):
    """Lay ``pixels`` out with ``pad`` junk bytes after every row; return (data, step)."""
    h, w, c = pixels.shape
    rows = np.full((h, w * c + pad), 0xEE, dtype=np.uint8)
    rows[:, : w * c] = pixels.reshape(h, w * c)
    return rows.tobytes(), w * c + pad


def _random_frame(h, w, c):
    return np.random.default_rng(0).integers(0, 256, (h, w, c), dtype=np.uint8)


@pytest.mark.parametrize("pad", [0, 64])
def test_zed_bgra8_drops_alpha_honours_step(pad):
    bgra = _random_frame(_ZED_H, _ZED_W, 4)
    data, step = _padded(bgra, pad)
    out, w, h = image_to_bgr_bytes(_FakeImage(data, _ZED_W, _ZED_H, "bgra8", step))
    assert (w, h) == (_ZED_W, _ZED_H)
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(h, w, 3), bgra[..., :3])


@pytest.mark.parametrize("pad", [0, 64])
def test_zed_rgba8_drops_alpha_and_swaps(pad):
    rgba = _random_frame(_ZED_H, _ZED_W, 4)
    data, step = _padded(rgba, pad)
    out, w, h = image_to_bgr_bytes(_FakeImage(data, _ZED_W, _ZED_H, "rgba8", step))
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(h, w, 3), rgba[..., 2::-1])


def test_padded_rgb8_rows_are_sliced_off():
    rgb = _random_frame(4, 5, 3)
    data, step = _padded(rgb, 1)
    out, _, _ = image_to_bgr_bytes(_FakeImage(data, 5, 4, "rgb8", step))
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(4, 5, 3), rgb[..., ::-1])


@pytest.mark.parametrize("enc", ["mono8", "8UC1"])
def test_mono_replicated_to_three_channels(enc):
    mono = _random_frame(3, 4, 1)
    out, _, _ = image_to_bgr_bytes(_FakeImage(mono.tobytes(), 4, 3, enc, 4))
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(3, 4, 3), np.repeat(mono, 3, -1))


def test_8uc3_is_bgr_passthrough():
    arr = _rgb_frame(2, 2)
    out, _, _ = image_to_bgr_bytes(_FakeImage(arr.tobytes(), 2, 2, "8UC3", 6))
    assert np.array_equal(np.frombuffer(out, np.uint8).reshape(2, 2, 3), arr)


def test_rejects_unsupported_encoding():
    msg = _FakeImage(b"\x00" * 16, 2, 2, "32FC1", 8)
    with pytest.raises(ImageConvertError, match="32FC1"):
        image_to_bgr_bytes(msg)


def test_rejects_step_shorter_than_row():
    msg = _FakeImage(b"\x00" * 16, 2, 2, "bgra8", 4)
    with pytest.raises(ImageConvertError, match="step"):
        image_to_bgr_bytes(msg)


def test_rejects_truncated_payload():
    msg = _FakeImage(b"\x00" * 10, 2, 2, "rgb8", 6)
    with pytest.raises(ImageConvertError, match="payload"):
        image_to_bgr_bytes(msg)
