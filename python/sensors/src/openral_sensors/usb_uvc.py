"""USB UVC RGB camera adapters — Logitech C920.

The C920 is the default camera on LeRobot SO-100 / SO-101 and most low-cost
research arms.  Driver: ``usb_cam`` (or ``v4l2_camera``) ROS 2 package; on
non-ROS hosts the LeRobot HAL grabs them via OpenCV.

Example:
    >>> from openral_sensors.usb_uvc import logitech_c920_spec
    >>> spec = logitech_c920_spec(name="scene_cam", parent_frame="base_link")
    >>> spec.modality
    'rgb'
    >>> spec.intrinsics.width
    1920
"""

from __future__ import annotations

import math

from openral_core.schemas import (
    IntrinsicsPinhole,
    SensorModality,
    SensorSpec,
)

from openral_sensors.catalog import CATALOG, SensorCatalogEntry, SensorSignature

__all__ = [
    "generic_uvc_rgb_spec",
    "logitech_c920_spec",
]


def _uvc_intrinsics(width: int, height: int, hfov_deg: float) -> IntrinsicsPinhole:
    """Compute nominal pinhole intrinsics from sensor dimensions and hFOV."""
    fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    return IntrinsicsPinhole(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        distortion_model="plumb_bob",
        distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )


def logitech_c920_spec(
    name: str = "usb_cam",
    parent_frame: str = "base_link",
    rate_hz: float = 30.0,
    width: int = 1920,
    height: int = 1080,
) -> SensorSpec:
    """Build a ``SensorSpec`` for a Logitech C920 / C920e (78° hFOV)."""
    return SensorSpec(
        name=name,
        modality=SensorModality.RGB,
        frame_id=f"{name}_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rate_hz,
        encoding="yuyv",
        intrinsics=_uvc_intrinsics(width, height, 78.0),
        fov_h_deg=78.0,
        fov_v_deg=43.3,
        ros2_topic=f"/{name}/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Logitech",
        model="C920",
        driver_pkg="usb_cam",
    )


def generic_uvc_rgb_spec(
    name: str = "usb_cam",
    parent_frame: str = "base_link",
    rate_hz: float = 30.0,
    width: int = 640,
    height: int = 480,
    hfov_deg: float = 70.0,
) -> SensorSpec:
    """Build a generic USB UVC RGB camera spec for calibrated per-robot overrides.

    Use this catalog entry when the robot manifest knows "USB UVC wrist camera"
    but not a stable vendor model. The manifest should override intrinsics with
    calibrated values and may override ``vendor`` / ``model`` when known.
    """
    return SensorSpec(
        name=name,
        modality=SensorModality.RGB,
        frame_id=f"{name}_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rate_hz,
        encoding="rgb8",
        intrinsics=_uvc_intrinsics(width, height, hfov_deg),
        fov_h_deg=hfov_deg,
        ros2_topic=f"/{name}/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        catalog_id="generic/usb_uvc_rgb",
        vendor="generic_usb",
        model="usb_uvc_rgb",
        driver_pkg="usb_cam",
    )


CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="generic/usb_uvc_rgb",
            vendor="generic",
            model="usb_uvc_rgb",
            kind="sensor",
            factory=generic_uvc_rgb_spec,
            modalities=(SensorModality.RGB,),
            description=(
                "Generic USB UVC RGB camera — use when the robot manifest has "
                "per-unit calibration but no stable vendor model id."
            ),
            signatures=(
                SensorSignature(kind="v4l2_name", value="USB Camera"),
                SensorSignature(kind="v4l2_name", value="UVC Camera"),
            ),
        ),
        SensorCatalogEntry(
            id="logitech/c920",
            vendor="logitech",
            model="c920",
            kind="sensor",
            factory=logitech_c920_spec,
            modalities=(SensorModality.RGB,),
            description="Logitech C920 / C920e — 1080p UVC, 78° hFOV; LeRobot SO-100/101 default.",
            docs_url="https://www.logitech.com/en-us/shop/p/c920-pro-hd-webcam",
            signatures=(
                # Logitech C920 (HD Pro Webcam C920) and C920e variants.
                SensorSignature(kind="usb_uvc", value="0x046d:0x082d"),
                SensorSignature(kind="usb_uvc", value="0x046d:0x0892"),
                SensorSignature(kind="v4l2_name", value="C920"),
            ),
        ),
    ]
)
