# SPDX-License-Identifier: Apache-2.0
"""The one registered RGB-D frame contract every sim backend publishes depth through.

A frame is what a real RGB-D driver publishes, as a plain dict (it crosses the Isaac
sidecar's msgpack wire and the HAL's observation dict, so it stays a dict)::

    {"depth": (H, W) float32 m, 0.0 = no return,
     "rgb": (H, W, 3) uint8 from the same render, or (0, 0, 3) when the camera has none,
     "k": [fx, fy, cx, cy] of that raster,
     "optical_in_base": 4x4 REP-103 optical pose in the manifest base_frame}

Producers: the Isaac scene (``obs["depth_frames"]``, rendered by the RTX camera) and the
MuJoCo sensor bridge (ray-cast raster + the same MJCF camera's colour). Both are ground
truth: no estimated depth is published through this contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from openral_core.exceptions import ROSConfigError

__all__ = ["depth_frame"]

_RGB_CHANNELS = 3


def depth_frame(depth: Any, rgb: Any, k: Any, optical_in_base: Any) -> dict[str, Any]:
    """Build (and check) one frame of the contract.

    Example:
        >>> f = depth_frame(np.ones((2, 3)), np.zeros((2, 3, 3)), [1, 1, 1, 1], np.eye(4))
        >>> f["depth"].dtype, f["rgb"].shape
        (dtype('float32'), (2, 3, 3))
        >>> depth_frame(np.ones((2, 3)), None, [1, 1, 1, 1], np.eye(4))["rgb"].shape
        (0, 0, 3)

    Raises:
        ROSConfigError: a shape is not the contract's.
    """
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0.0] = 0.0
    c = np.zeros((0, 0, _RGB_CHANNELS), np.uint8) if rgb is None else np.asarray(rgb, np.uint8)
    kk = np.asarray(k, dtype=np.float64).reshape(-1)
    pose = np.asarray(optical_in_base, dtype=np.float64)
    if d.ndim != 2 or kk.size != 4 or pose.shape != (4, 4):  # noqa: PLR2004  # reason: contract shapes
        raise ROSConfigError(
            f"depth frame: depth {d.shape} must be (H, W), k {kk.size} values must be 4, "
            f"optical_in_base {pose.shape} must be 4x4"
        )
    if c.shape not in {(0, 0, _RGB_CHANNELS), (*d.shape, _RGB_CHANNELS)}:
        raise ROSConfigError(f"depth frame: rgb {c.shape} is not registered to depth {d.shape}")
    return {"depth": d, "rgb": c, "k": kk, "optical_in_base": pose}
