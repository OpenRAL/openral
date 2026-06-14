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
