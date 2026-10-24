"""The locator detects on a crop of the frame and answers in full-frame pixels (#418)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openral_perception_ros", reason="the detector node package is not on the path")


def test_crop_then_uncrop_round_trips_a_box() -> None:
    from openral_core import ObjectDetection2D, ObjectsMetadata
    from openral_perception_ros.ros_image_detector_node import crop_bgr, uncrop_metadata

    w, h = 1920, 1080
    img = np.zeros((h, w, 3), np.uint8)
    img[500:540, 1000:1060] = 255
    crop = crop_bgr(img.tobytes(), w, h, (0.5, 0.4, 0.6, 0.6))
    assert crop is not None
    data, cw, ch, x0, y0 = crop
    assert (x0, y0, cw, ch) == (960, 432, 192, 216)
    sub = np.frombuffer(data, np.uint8).reshape(ch, cw, 3)
    assert sub[500 - y0, 1000 - x0].tolist() == [255, 255, 255]
    md = ObjectsMetadata(
        sensor_id="head_zed",
        detections=[ObjectDetection2D(label="box", confidence=0.6, bbox_xyxy=(40, 68, 100, 108))],
        model_id="omdet",
        frame_width=cw,
        frame_height=ch,
    )
    full = uncrop_metadata(md, x0, y0, w, h)
    assert full.detections[0].bbox_xyxy == (1000, 500, 1060, 540)
    assert (full.frame_width, full.frame_height) == (w, h)


@pytest.mark.parametrize("roi", [(), (0.2, 0.2, 0.2, 0.5), (0.9, 0.9, 1.5, 1.5)])
def test_an_empty_or_degenerate_roi_means_the_whole_frame(roi: tuple[float, ...]) -> None:
    from openral_perception_ros.ros_image_detector_node import crop_bgr

    w, h = 64, 48
    img = bytes(w * h * 3)
    out = crop_bgr(img, w, h, roi)
    if roi == (0.9, 0.9, 1.5, 1.5):  # clamped to the image: still a valid crop
        assert out is not None and out[3:] == (57, 43)
    else:
        assert out is None
