# SPDX-License-Identifier: Apache-2.0
"""The one registered RGB-D frame contract every sim backend publishes depth through.

A frame is what a real RGB-D driver publishes, as a plain dict (it crosses the Isaac
sidecar's msgpack wire and the HAL's observation dict, so it stays a dict)::

    {"depth": (H, W) float32 m, 0.0 = no return,
     "rgb": (H, W, 3) uint8 from the same render, or (0, 0, 3) when the camera has none,
     "k": [fx, fy, cx, cy] of that raster,
     "optical_in_base": 4x4 REP-103 optical pose in the manifest base_frame,
     "source": "rendered" | "estimated:<model>"}

Producers: the Isaac scene (``obs["depth_frames"]``, rendered by the RTX camera), the
MuJoCo sensor bridge (ray-cast raster + the same MJCF camera's colour), and
``estimated_depth_frame`` — monocular depth from the RGB (Depth Anything V3), the
explicit opt-in fallback for a camera that renders colour but no depth. ``source``
says which: an estimated frame is never ground truth and a consumer that needs ground
truth checks it (``is_estimated``).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

__all__ = ["RENDERED", "DepthEstimator", "depth_frame", "estimated_depth_frame", "is_estimated"]

_log = structlog.get_logger(__name__)

#: ``source`` of a frame whose depth the simulator rendered or ray-cast (ground truth).
RENDERED = "rendered"
_ESTIMATED_PREFIX = "estimated:"
_RGB_CHANNELS = 3


class DepthEstimator(Protocol):
    """Monocular metric depth from colour (``openral_sim.da3_depth.Da3DepthClient``)."""

    def infer(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Metric depth (m) pixel-aligned with ``rgb``."""
        ...


def depth_frame(
    depth: Any, rgb: Any, k: Any, optical_in_base: Any, *, source: str = RENDERED
) -> dict[str, Any]:
    """Build (and check) one frame of the contract.

    Example:
        >>> f = depth_frame(np.ones((2, 3)), np.zeros((2, 3, 3)), [1, 1, 1, 1], np.eye(4))
        >>> f["depth"].dtype, f["rgb"].shape, f["source"]
        (dtype('float32'), (2, 3, 3), 'rendered')
        >>> depth_frame(np.ones((2, 3)), None, [1, 1, 1, 1], np.eye(4))["rgb"].shape
        (0, 0, 3)

    Raises:
        ROSConfigError: a shape is not the contract's, or ``source`` is neither
            ``"rendered"`` nor ``"estimated:<model>"``.
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
    if c.size and c.shape != (*d.shape, _RGB_CHANNELS):
        raise ROSConfigError(f"depth frame: rgb {c.shape} is not registered to depth {d.shape}")
    if source != RENDERED and not (
        source.startswith(_ESTIMATED_PREFIX) and len(source) > len(_ESTIMATED_PREFIX)
    ):
        raise ROSConfigError(f"depth frame source {source!r} is not 'rendered' or 'estimated:*'")
    return {"depth": d, "rgb": c, "k": kk, "optical_in_base": pose, "source": source}


def is_estimated(frame: dict[str, Any]) -> bool:
    """Whether ``frame``'s depth was estimated from colour rather than rendered.

    Example:
        >>> is_estimated({"source": "estimated:da3"}), is_estimated({})
        (True, False)
    """
    return str(frame.get("source", RENDERED)).startswith(_ESTIMATED_PREFIX)


def estimated_depth_frame(
    rgb: NDArray[np.uint8],
    k: Any,
    optical_in_base: Any,
    estimator: DepthEstimator,
    *,
    model: str = "da3",
) -> dict[str, Any]:
    """A frame whose depth ``estimator`` inferred from ``rgb`` (``source="estimated:<model>"``).

    The fallback for a camera that renders colour but no depth. Only ever built on
    an explicit per-camera opt-in (the sensor bridge's ``estimated_depth_cameras``);
    never substituted for a rendered raster.
    """
    depth = estimator.infer(np.ascontiguousarray(rgb, dtype=np.uint8))
    return depth_frame(depth, rgb, k, optical_in_base, source=f"{_ESTIMATED_PREFIX}{model}")
