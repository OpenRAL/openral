"""Convert a sensor_msgs/Image to contiguous BGR bytes (no cv_bridge dep)."""

from __future__ import annotations

from typing import Any

__all__ = ["ImageConvertError", "image_to_bgr_bytes"]


class ImageConvertError(ValueError):
    """Raised when an Image can't be converted (encoding/stride unsupported)."""


def image_to_bgr_bytes(msg: Any) -> tuple[bytes, int, int]:
    """Return ``(bgr_bytes, width, height)`` from an ``rgb8``/``bgr8`` Image.

    Only tightly-packed rows (``step == width * 3``) are supported; padded rows
    raise :class:`ImageConvertError`. ``rgb8`` channels are reversed to BGR (the
    order ``ObjectsDetector.detect`` expects); ``bgr8`` passes through.

    Args:
        msg: A ``sensor_msgs/Image`` (or duck-typed stand-in) with ``data``,
            ``width``, ``height``, ``encoding``, ``step``.

    Returns:
        ``(bgr_bytes, width, height)`` — contiguous H*W*3 BGR uint8 bytes.

    Raises:
        ImageConvertError: On an unsupported encoding or a padded row stride.
    """
    import numpy as np

    enc = msg.encoding
    if enc not in ("rgb8", "bgr8"):
        raise ImageConvertError(f"unsupported encoding {enc!r}; need rgb8/bgr8")
    w, h = int(msg.width), int(msg.height)
    if int(msg.step) != w * 3:
        raise ImageConvertError(f"padded rows unsupported: step={msg.step} != width*3={w * 3}")
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
    bgr = arr[..., ::-1] if enc == "rgb8" else arr
    return np.ascontiguousarray(bgr).tobytes(), w, h
