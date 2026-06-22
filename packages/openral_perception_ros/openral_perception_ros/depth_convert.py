"""Encode a metric-depth ndarray into a sensor_msgs/Image (+ CameraInfo).

ADR-0064 — the monocular metric-depth provider (DA3-Small by default) turns a
mono RGB stream into a metric depth image that **nvblox** fuses with cuVSLAM's
pose into a 2D costmap for Nav2, so lidar-less robots get a `/map`. This module
is the pure, ROS-message-boundary half: float32 metres → `32FC1` Image, and the
predicted/declared pinhole intrinsics → CameraInfo. The model itself runs
out-of-process (isolated venv sidecar); this code never imports torch.

`32FC1` carries depth in **metres** (the nvblox + REP-118 convention); invalid
pixels are NaN. Only tightly-packed rows are produced (`step == width * 4`).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DepthConvertError",
    "camera_info_from_intrinsics",
    "depth_array_to_image_msg",
    "image_msg_to_depth_array",
]


class DepthConvertError(ValueError):
    """Raised when a depth array / Image can't be converted."""


def depth_array_to_image_msg(depth_m: Any, *, frame_id: str, stamp: Any = None) -> Any:
    """Build a ``32FC1`` ``sensor_msgs/Image`` (metres) from a HxW float array.

    Args:
        depth_m: 2-D array-like of per-pixel depth in **metres**. Cast to
            ``float32``; non-finite entries are preserved as NaN (nvblox treats
            NaN as "no return").
        frame_id: TF frame the depth optical centre lives in (the camera's
            optical frame) — nvblox looks this up against cuVSLAM's pose.
        stamp: Optional ``builtin_interfaces/Time``; left zero when ``None``.

    Returns:
        A ``sensor_msgs/Image`` with ``encoding='32FC1'`` and tightly-packed rows.

    Raises:
        DepthConvertError: If ``depth_m`` is not 2-D.
    """
    import numpy as np
    from sensor_msgs.msg import Image

    arr = np.ascontiguousarray(np.asarray(depth_m, dtype=np.float32))
    if arr.ndim != 2:
        raise DepthConvertError(f"depth must be 2-D HxW, got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4
    msg.data = arr.tobytes()
    return msg


def image_msg_to_depth_array(msg: Any) -> Any:
    """Inverse of :func:`depth_array_to_image_msg` — ``32FC1`` Image → HxW float32.

    Args:
        msg: A ``sensor_msgs/Image`` with ``encoding='32FC1'`` and packed rows.

    Returns:
        A ``(height, width)`` float32 ``numpy`` array of metres.

    Raises:
        DepthConvertError: On a non-``32FC1`` encoding or a padded row stride.
    """
    import numpy as np

    if msg.encoding != "32FC1":
        raise DepthConvertError(f"unsupported encoding {msg.encoding!r}; need 32FC1")
    w, h = int(msg.width), int(msg.height)
    if int(msg.step) != w * 4:
        raise DepthConvertError(f"padded rows unsupported: step={msg.step} != width*4={w * 4}")
    return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, w)


def camera_info_from_intrinsics(
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    frame_id: str,
    stamp: Any = None,
) -> Any:
    """Build a plain pinhole ``sensor_msgs/CameraInfo`` (no distortion).

    nvblox pairs the depth Image with this to back-project pixels into the TSDF.
    The intrinsics come from the depth model's prediction (DA3 predicts them) or
    the robot's calibrated ``SensorSpec`` — the provider node chooses.

    Args:
        fx: Horizontal focal length, pixels.
        fy: Vertical focal length, pixels.
        cx: Principal-point x, pixels.
        cy: Principal-point y, pixels.
        width: Image width, pixels.
        height: Image height, pixels.
        frame_id: Optical frame (matches the depth Image's ``frame_id``).
        stamp: Optional ``builtin_interfaces/Time``.

    Returns:
        A ``sensor_msgs/CameraInfo`` with ``distortion_model='plumb_bob'`` and
        zero distortion coefficients.
    """
    from sensor_msgs.msg import CameraInfo

    info = CameraInfo()
    if stamp is not None:
        info.header.stamp = stamp
    info.header.frame_id = frame_id
    info.width = int(width)
    info.height = int(height)
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info
