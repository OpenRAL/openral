"""Ray-cast boxes into the OpenArm head camera — the captures instance tests fit masks from.

The camera is the committed manifest's ``head_zed`` (``robots/openarm/robot.yaml``): its
``static_transform_xyz_rpy`` in ``openarm_base`` and REP-103 optical axes
(``depth_cloud.BODY_TO_OPTICAL_QUAT_XYZW``), at its intrinsics scaled by ``scale`` (a
quarter of HD1080 keeps a 7 cm prop above ``target_region_from_mask``'s 200-point floor and
the raster cheap). Each pixel's label is the box it hits first (0 = none): the per-instance
masks a segmenter would return for a prompt on each box. No mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, RobotDescription
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_hal.depth_cloud import BODY_TO_OPTICAL_QUAT_XYZW

_ROBOT = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"

#: ``(lo_xyz, hi_xyz)`` of an axis-aligned box in ``openarm_base``.
Box = tuple[tuple[float, float, float], tuple[float, float, float]]


def head_camera(scale: float = 0.25) -> tuple[NDArray[np.float64], IntrinsicsPinhole]:
    """``(T_base_from_optical, intrinsics)`` of the manifest's ``head_zed``."""
    cam = next(c for c in RobotDescription.from_yaml(str(_ROBOT)).sensors if c.name == "head_zed")
    assert cam.static_transform_xyz_rpy is not None and cam.intrinsics is not None
    x, y, z, roll, pitch, yaw = cam.static_transform_xyz_rpy
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    t_link = np.eye(4)
    t_link[:3, :3], t_link[:3, 3] = rz @ ry @ rx, (x, y, z)
    t = t_link @ homogeneous_from_quat_xyz((0.0, 0.0, 0.0), BODY_TO_OPTICAL_QUAT_XYZW)
    k = cam.intrinsics
    scaled = IntrinsicsPinhole(
        width=round(k.width * scale),
        height=round(k.height * scale),
        fx=k.fx * scale,
        fy=k.fy * scale,
        cx=k.cx * scale,
        cy=k.cy * scale,
    )
    return np.asarray(t, dtype=np.float64), scaled


def render(
    boxes: Sequence[Box], t_base_from_cam: NDArray[np.float64], k: IntrinsicsPinhole
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``(depth, labels)``: the z-depth raster and the 1-based index of the box each pixel hits."""
    origin = t_base_from_cam[:3, 3]
    rows, cols = np.mgrid[0 : k.height, 0 : k.width]
    rays = np.stack(((cols + 0.5 - k.cx) / k.fx, (rows + 0.5 - k.cy) / k.fy, np.ones(rows.shape)))
    rays = rays.reshape(3, -1).T @ t_base_from_cam[:3, :3].T  # parameter = optical z
    depth = np.full(rays.shape[0], np.inf)
    label = np.zeros(rays.shape[0], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n, (lo, hi) in enumerate(boxes, start=1):
            t1, t2 = (np.asarray(lo) - origin) / rays, (np.asarray(hi) - origin) / rays
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            closer = (near <= far) & (near > 0) & (near < depth)
            depth[closer], label[closer] = near[closer], n
    depth[~np.isfinite(depth)] = 0.0
    shape = (k.height, k.width)
    return depth.reshape(shape), label.reshape(shape)
