"""DA3 metric depth for the world-collision map: canonical output to metres, then a cloud.

``DeployRuntime.estimated_depth_camera`` runs the depth provider on an RGB camera with
``depth-anything/DA3METRIC-LARGE``, whose output is *canonical* depth: the upstream
README's conversion is ``metric_depth = focal * net_output / 300`` with ``focal`` the
camera's focal length in pixels (mean of ``fx`` and ``fy``). The network predicts at its
own processing raster, so the focal used is the camera's ``CameraInfo`` focal scaled to
that raster — the image the network actually saw. The camera's real K is used, never the
K DA3 estimates, because the cloud is placed through the camera's calibrated mount.

Pure numpy: no ROS, no torch. The provider node (``depth_provider_node``) owns the I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DA3_CANONICAL_FOCAL_PX",
    "da3_metric_depth",
    "is_metric_model",
    "points_from_depth",
    "scale_intrinsics",
]

#: The focal length (pixels) DA3METRIC's canonical depth is expressed at.
DA3_CANONICAL_FOCAL_PX = 300.0


def is_metric_model(model_id: str) -> bool:
    """Whether a DA3 checkpoint predicts canonical *metric* depth (``DA3METRIC-*``).

    ``DA3-SMALL`` .. ``DA3-GIANT`` are not documented as metric, so their output must
    never be converted to metres and fed to collision checking.

    Example:
        >>> is_metric_model("depth-anything/DA3METRIC-LARGE"), is_metric_model("DA3-SMALL")
        (True, False)
    """
    return "DA3METRIC" in model_id.upper()


def scale_intrinsics(
    k: tuple[float, float, float, float], src_wh: tuple[int, int], dst_wh: tuple[int, int]
) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of a ``src_wh`` image resized (not cropped) to ``dst_wh``.

    Pixel centres map to pixel centres: ``c' = (c + 0.5) * s - 0.5``.

    Example:
        >>> scale_intrinsics((600.0, 600.0, 319.5, 239.5), (640, 480), (320, 240))
        (300.0, 300.0, 159.5, 119.5)
    """
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    fx, fy, cx, cy = k
    return (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)


def da3_metric_depth(canonical: Any, *, fx: float, fy: float) -> NDArray[np.float32]:
    """Metres from DA3METRIC's canonical output, ``fx``/``fy`` at the output's raster.

    Non-finite or non-positive outputs become ``0.0`` (no return).

    Example:
        >>> da3_metric_depth(np.array([[300.0, -1.0]]), fx=150.0, fy=150.0).tolist()
        [[150.0, 0.0]]
    """
    raw = np.asarray(canonical, dtype=np.float64)
    depth = 0.5 * (fx + fy) * raw / DA3_CANONICAL_FOCAL_PX
    depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    return depth.astype(np.float32)


def points_from_depth(
    depth: Any, *, fx: float, fy: float, cx: float, cy: float, max_range_m: float
) -> NDArray[np.float32]:
    """Back-project a metric depth raster to an ``(N, 3)`` optical-frame cloud.

    A pixel ``(col, row)`` at depth ``z`` is ``((col - cx) / fx * z, (row - cy) / fy * z,
    z)`` (REP-103 optical). Pixels with no return (``0.0``) or beyond ``max_range_m``
    (optical z) are dropped: a monocular estimate is least reliable far away, and the
    sky or a far wall must not become phantom obstacles at the map's edge.

    Example:
        >>> d = np.array([[0.0, 2.0], [9.0, 1.0]], dtype=np.float32)
        >>> points_from_depth(d, fx=2.0, fy=2.0, cx=0.0, cy=0.0, max_range_m=5.0).tolist()
        [[1.0, 0.0, 2.0], [0.5, 0.5, 1.0]]
    """
    z = np.asarray(depth, dtype=np.float32)
    rows, cols = np.nonzero((z > 0.0) & (z <= max_range_m))
    zz = z[rows, cols]
    pts = np.empty((rows.size, 3), dtype=np.float32)
    pts[:, 0] = (cols - cx) / fx * zz
    pts[:, 1] = (rows - cy) / fy * zz
    pts[:, 2] = zz
    return pts
