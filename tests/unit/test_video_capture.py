from __future__ import annotations

import numpy as np

from openral_sim.policies._video_capture import tile_input_frames, to_input_frame


def test_to_input_frame_flips_180() -> None:
    image = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3)
    frame = to_input_frame(image, flip_180=True)
    assert frame is not None
    assert np.array_equal(frame, image[::-1, ::-1])


def test_tile_input_frames_returns_single_frame_unchanged() -> None:
    image = np.full((4, 5, 3), 17, dtype=np.uint8)
    tiled = tile_input_frames([image])
    assert tiled is not None
    assert np.array_equal(tiled, image)


def test_tile_input_frames_concatenates_multiple_cameras() -> None:
    left = np.full((6, 4, 3), 10, dtype=np.uint8)
    right = np.full((3, 2, 3), 200, dtype=np.uint8)
    tiled = tile_input_frames([left, right])
    assert tiled is not None
    assert tiled.shape[0] == 3
    assert tiled.shape[1] > right.shape[1]
    assert int(tiled[:, :2].mean()) == 10
    assert int(tiled[:, -2:].mean()) == 200
