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


def _encoded(fmt: str, rgb: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format=fmt)
    return buf.getvalue()


def test_png_decodes_losslessly_to_rgb() -> None:
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    out = decode_inline_frame(_frame(FrameEncoding.PNG, _encoded("PNG", rgb), channels=3))
    assert out is not None and out.dtype == np.uint8
    np.testing.assert_array_equal(out, rgb)


def test_jpeg_decodes_to_declared_shape_and_mono_when_one_channel() -> None:
    rgb = np.full((3, 4, 3), 128, dtype=np.uint8)
    data = _encoded("JPEG", rgb)
    colour = decode_inline_frame(_frame(FrameEncoding.JPEG, data, channels=3))
    mono = decode_inline_frame(_frame(FrameEncoding.JPEG, data, channels=1))
    assert colour is not None and colour.shape == (3, 4, 3)
    assert abs(int(colour.mean()) - 128) <= 2  # lossy, but a flat field survives
    assert mono is not None and mono.shape == (3, 4, 1)


def test_every_skip_is_logged_with_sensor_and_encoding() -> None:
    from structlog.testing import capture_logs

    wrong_size_png = _encoded("PNG", np.zeros((5, 5, 3), dtype=np.uint8))
    with capture_logs() as logs:
        assert decode_inline_frame(_frame(FrameEncoding.JPEG, b"\xff\xd8junk", channels=3)) is None
        assert decode_inline_frame(_frame(FrameEncoding.PNG, wrong_size_png, channels=3)) is None
        assert decode_inline_frame(_frame(FrameEncoding.CUDA_NV12, bytes(18), channels=1)) is None
        assert decode_inline_frame(_frame(FrameEncoding.RGB8, bytes(35), channels=3)) is None
    skips = [e for e in logs if e["event"] == "runner.frame_skipped"]
    assert [e["encoding"] for e in skips] == ["jpeg", "png", "cuda_nv12", "rgb8"]
    assert all(e["sensor"] == "top" and e["log_level"] == "warning" for e in skips)


def test_frames_without_inline_data_are_not_skips() -> None:
    from structlog.testing import capture_logs

    frame = SensorFrame(
        sensor_id="top",
        stamp_monotonic_ns=1,
        stamp_wall_ns=2,
        encoding=FrameEncoding.CUDA_NV12,
        width=4,
        height=3,
        handle=0xDEAD,
    )
    with capture_logs() as logs:
        assert decode_inline_frame(frame) is None
    assert logs == []
