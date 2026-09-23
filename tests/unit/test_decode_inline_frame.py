"""``decode_inline_frame``: the payload length is checked before the reshape.

``SensorFrame`` validates the dimensions, not the byte count. A truncated or
row-padded payload used to raise ``ValueError`` out of ``reshape`` and abort the
recorder's tick; it is now a skipped frame (``None``), like a frame with no
inline pixels.
"""

from __future__ import annotations

import numpy as np
from openral_core.schemas import FrameEncoding, SensorFrame
from openral_runner.dataset_recorder_bridge import decode_inline_frame


def _frame(encoding: FrameEncoding, data: bytes, *, channels: int) -> SensorFrame:
    return SensorFrame(
        sensor_id="top",
        stamp_monotonic_ns=1,
        stamp_wall_ns=2,
        encoding=encoding,
        width=4,
        height=3,
        channels=channels,
        data=data,
    )


def test_exact_payloads_decode_with_the_encodings_dtype() -> None:
    rgb = decode_inline_frame(_frame(FrameEncoding.RGB8, bytes(range(36)), channels=3))
    assert rgb is not None and rgb.shape == (3, 4, 3) and rgb.dtype == np.uint8
    depth = decode_inline_frame(
        _frame(FrameEncoding.DEPTH16, np.arange(12, dtype=np.uint16).tobytes(), channels=1)
    )
    assert depth is not None and depth.shape == (3, 4, 1) and depth.dtype == np.uint16
    assert int(depth[2, 3, 0]) == 11


def test_truncated_or_padded_payloads_are_skipped_not_raised() -> None:
    short = decode_inline_frame(_frame(FrameEncoding.RGB8, bytes(35), channels=3))
    padded = decode_inline_frame(_frame(FrameEncoding.RGB8, bytes(40), channels=3))
    # A DEPTH16 payload sized for uint8 pixels is half a frame.
    half_depth = decode_inline_frame(_frame(FrameEncoding.DEPTH16, bytes(12), channels=1))
    assert short is None and padded is None and half_depth is None
