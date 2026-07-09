"""Regression tests for ``openral_rskill.gpu_passthrough`` frame helpers.

``_channels`` used to reference ``FrameEncoding.BGRA8`` — an enum member
that never existed — so any encoding that fell past the RGB checks
(e.g. ``DEPTH16``) raised ``AttributeError`` at runtime instead of
falling through to the defensive BGR default. Real enum members, no
mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import numpy as np
from openral_core.schemas import FrameEncoding
from openral_rskill.gpu_passthrough import _channels, _zero_frame


def test_channels_handles_every_frame_encoding() -> None:
    """Every member resolves to a positive channel count — no AttributeError."""
    for encoding in FrameEncoding:
        n = _channels(encoding)
        assert n in (1, 3), f"{encoding} -> {n}"


def test_channels_known_layouts() -> None:
    assert _channels(FrameEncoding.MONO8) == 1
    assert _channels(FrameEncoding.BGR8) == 3
    assert _channels(FrameEncoding.RGB8) == 3
    # Non-raster / opaque encodings fall through to the defensive BGR default.
    assert _channels(FrameEncoding.DEPTH16) == 3
    assert _channels(FrameEncoding.CUDA_NV12) == 3


def test_zero_frame_is_uint8_rgb_shaped() -> None:
    frame = _zero_frame()
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert not frame.any()
