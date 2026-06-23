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


def test_tile_input_frames_arranges_multiple_cameras_into_square_grid() -> None:
    left = np.full((6, 4, 3), 10, dtype=np.uint8)
    right = np.full((6, 4, 3), 200, dtype=np.uint8)
    bottom = np.full((6, 4, 3), 120, dtype=np.uint8)
    tiled = tile_input_frames([left, right, bottom])
    assert tiled is not None
    assert tiled.shape[0] == tiled.shape[1]
    assert int(tiled[1:5, 3:5].mean()) == 10
    assert int(tiled[1:5, 7:9].mean()) == 200
    assert int(tiled[7:11, 3:5].mean()) == 120
