"""Tiny shared utility for VLA adapters to record what they fed the policy.

Every visuomotor adapter (SmolVLA, ACT, π0.5, xVLA, Diffusion Policy, …)
applies its own preprocessing — channel reorder, ImageNet normalize,
resize / crop, optional 180° flip — before the image reaches the
underlying policy. The eval-layer video helper wants to display the
*post-processing* image (i.e. what the policy actually saw), not the raw
env frame, so users can spot bugs like a flipped wrist camera or a
bottom-half crop. This utility gives every adapter a one-liner to record
that image without each having to reimplement uint8 conversion / flip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def to_input_frame(
    image: object | None,
    *,
    flip_180: bool = False,
) -> NDArray[np.uint8] | None:
    """Convert an adapter input image to an HWC uint8 RGB frame for video.

    Args:
        image: HWC uint8 ndarray (raw env image), or anything ``np.asarray``
            can coerce to such. ``None`` returns ``None``.
        flip_180: If True, apply the same 180° rotation the adapter applies
            before policy ingestion (so the frame in the video matches the
            policy's view).

    Returns:
        HWC uint8 RGB array, or ``None`` if ``image`` is ``None``.
    """
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        # Adapters generally pass uint8; clamp + cast as a safety net.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if flip_180:
        # Rotate 180° = flip on both spatial axes.
        arr = arr[::-1, ::-1].copy()
    return arr


def tile_input_frames(images: Sequence[object | None]) -> NDArray[np.uint8] | None:
    """Compose multiple adapter input images into one debug-preview frame.

    The MP4 helper stores one ``vla_input_frame`` per step. Multi-camera
    policies therefore need a stitched preview so the debug video can show
    every camera the adapter consumed, not only the last one seen.

    Args:
        images: Ordered per-camera HWC images after any adapter-specific
            orientation / flip has already been applied. ``None`` entries are
            ignored.

    Returns:
        A single HWC uint8 RGB frame. One camera returns unchanged; multiple
        cameras are arranged into a padded near-square grid so downstream video
        resizing keeps each stream readable.
    """
    frames = [
        frame
        for frame in (to_input_frame(image) for image in images if image is not None)
        if frame is not None
    ]
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    cols = 2
    rows = (len(frames) + cols - 1) // cols
    cell_h = min(frame.shape[0] for frame in frames)
    cell_w = min(frame.shape[1] for frame in frames)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        fitted = _fit_frame(frame, cell_w=cell_w, cell_h=cell_h)
        y0 = row * cell_h + (cell_h - fitted.shape[0]) // 2
        x0 = col * cell_w + (cell_w - fitted.shape[1]) // 2
        canvas[y0 : y0 + fitted.shape[0], x0 : x0 + fitted.shape[1]] = fitted
    return _pad_to_square(canvas)


def _fit_frame(
    frame: NDArray[np.uint8],
    *,
    cell_w: int,
    cell_h: int,
) -> NDArray[np.uint8]:
    """Resize a preview frame to fit within one grid cell, preserving aspect."""
    from PIL import Image

    if frame.shape[0] <= cell_h and frame.shape[1] <= cell_w:
        return frame
    scale = min(cell_w / frame.shape[1], cell_h / frame.shape[0])
    width = max(1, round(frame.shape[1] * scale))
    height = max(1, round(frame.shape[0] * scale))
    resized = Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _pad_to_square(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Center-pad a preview canvas to square so later resize is not misleading."""
    side = max(frame.shape[0], frame.shape[1])
    out = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - frame.shape[0]) // 2
    x0 = (side - frame.shape[1]) // 2
    out[y0 : y0 + frame.shape[0], x0 : x0 + frame.shape[1]] = frame
    return out
