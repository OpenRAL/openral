"""Convert a sensor_msgs/Image to contiguous BGR bytes (no cv_bridge dep)."""

from __future__ import annotations

from typing import Any, Final

__all__ = ["SUPPORTED_ENCODINGS", "ImageConvertError", "image_to_bgr_bytes"]

# encoding -> (channels, source-channel indices that make B, G, R). The colour
# set the runner's `Ros2ImageSensorReader` accepts. `bgra8` is the ZED
# wrapper's default colour layout (the deploy camera): refusing it left the SAM
# segmenter with no cached frame on the real OpenArm cell (2026-10-04 twin
# pass) — every segment_in_view raised ROSPerceptionStale.
_LAYOUTS: Final[dict[str, tuple[int, list[int]]]] = {
    "bgr8": (3, [0, 1, 2]),
    "8UC3": (3, [0, 1, 2]),
    "rgb8": (3, [2, 1, 0]),
    "bgra8": (4, [0, 1, 2]),
    "rgba8": (4, [2, 1, 0]),
    "mono8": (1, [0, 0, 0]),
    "8UC1": (1, [0, 0, 0]),
}

SUPPORTED_ENCODINGS: Final[frozenset[str]] = frozenset(_LAYOUTS)


class ImageConvertError(ValueError):
    """Raised when an Image can't be converted (encoding/stride unsupported)."""


def image_to_bgr_bytes(msg: Any) -> tuple[bytes, int, int]:
    """Return ``(bgr_bytes, width, height)`` from a colour or mono Image.

    Accepts ``bgr8``/``8UC3``, ``rgb8``, ``bgra8``, ``rgba8`` (alpha dropped)
    and ``mono8``/``8UC1`` (replicated to three channels). ``msg.step`` is
    honoured, so padded rows (pitch-aligned ZED / NITROS buffers) are sliced
    off rather than folded into the picture.

    Args:
        msg: A ``sensor_msgs/Image`` (or duck-typed stand-in) with ``data``,
            ``width``, ``height``, ``encoding``, ``step``.

    Returns:
        ``(bgr_bytes, width, height)`` — contiguous H*W*3 BGR uint8 bytes.

    Raises:
        ImageConvertError: On an unsupported encoding, a ``step`` shorter than
            a packed row, or a payload shorter than ``height * step``.

    Example:
        >>> from types import SimpleNamespace
        >>> m = SimpleNamespace(
        ...     encoding="bgra8", width=1, height=1, step=8, data=bytes([1, 2, 3, 255, 0, 0, 0, 0])
        ... )
        >>> bgr, w, h = image_to_bgr_bytes(m)
        >>> list(bgr), w, h
        ([1, 2, 3], 1, 1)
    """
    import numpy as np

    enc = str(msg.encoding)
    layout = _LAYOUTS.get(enc)
    if layout is None:
        raise ImageConvertError(
            f"unsupported encoding {enc!r}; supported: {sorted(SUPPORTED_ENCODINGS)}"
        )
    channels, order = layout
    w, h, step = int(msg.width), int(msg.height), int(msg.step)
    packed = w * channels
    if step < packed:
        raise ImageConvertError(f"step={step} shorter than a packed {enc} row ({packed} B)")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if raw.size < h * step:
        raise ImageConvertError(f"payload {raw.size} B < height*step = {h * step} B")
    pixels = raw[: h * step].reshape(h, step)[:, :packed].reshape(h, w, channels)
    return np.ascontiguousarray(pixels[..., order]).tobytes(), w, h
