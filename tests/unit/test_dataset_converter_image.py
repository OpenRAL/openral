"""The bag converter refuses image dims its pixel bytes do not carry (CWE-789)."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from openral_core.exceptions import ROSConfigError
from openral_dataset import Rosbag2ToLeRobotConverter


def _payload(height: int, width: int, channels: int, data: bytes) -> dict[str, object]:
    return {
        "camera": "agentview",
        "height": height,
        "width": width,
        "channels": channels,
        "data_b64": base64.b64encode(data).decode("ascii"),
    }


def test_round_trips_a_real_frame() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    decoded = Rosbag2ToLeRobotConverter._decode_image(_payload(2, 3, 3, frame.tobytes()))
    np.testing.assert_array_equal(decoded, frame)


@pytest.mark.parametrize(
    ("height", "width", "channels", "nbytes"),
    [
        (100_000, 100_000, 0, 0),  # zero channels: an empty payload claims 10^10 pixels
        (0, 100_000, 3, 0),
        (2, 3, 3, 17),  # short payload
    ],
)
def test_dims_must_match_pixel_bytes(height: int, width: int, channels: int, nbytes: int) -> None:
    with pytest.raises(ROSConfigError, match="pixel bytes"):
        Rosbag2ToLeRobotConverter._decode_image(_payload(height, width, channels, b"\0" * nbytes))
