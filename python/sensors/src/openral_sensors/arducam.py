"""Arducam USB3 global-shutter camera adapters — B0495 (AR0234 colour).

Provides:

- ``arducam_b0495_spec`` — factory that builds a :class:`SensorSpec` for an
  Arducam B0495 (2.3 MP AR0234 global shutter over a Cypress FX3 USB 3.0
  UVC bridge).
- Catalog registration under ``arducam/b0495``.

Global shutter is the reason this module exists separately from
:mod:`openral_sensors.usb_uvc`: a rolling-shutter webcam smears the frame
during arm motion, which corrupts exactly the wrist / workspace views a VLA
conditions on.  The B0495 exposes every row simultaneously, so a frame
captured mid-trajectory is geometrically valid.

Optics are **not** a property of this device.  The B0495 ships as a bare
board with an M12 lens mount, so field of view and distortion depend on
whichever lens the integrator screwed on.  The factory therefore leaves
``intrinsics`` unset unless the caller supplies the mounted lens's
horizontal FOV — an uncalibrated pinhole model would be a fabricated number,
not a conservative default.

Example:
    >>> from openral_sensors.arducam import arducam_b0495_spec
    >>> spec = arducam_b0495_spec(name="cell_cam_left", parent_frame="base_link")
    >>> spec.modality
    'rgb'
    >>> spec.intrinsics is None  # no lens declared → no fabricated intrinsics
    True
    >>> calibrated = arducam_b0495_spec(name="cell_cam_left", hfov_deg=70.0)
    >>> calibrated.intrinsics.width
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

__all__ = ["arducam_b0495_spec"]

# ── Device facts ──────────────────────────────────────────────────────────────
# Read off the hardware with VIDIOC_ENUM_FMT / VIDIOC_ENUM_FRAMESIZES /
# VIDIOC_ENUM_FRAMEINTERVALS, not from a datasheet:
#   pixel format : YUYV 4:2:2 (single format)
#   1920×1200    : 50 / 30 / 15 fps
#    960×600     : 80 / 60 / 30 / 15 fps
# USB descriptor: 0x04b4:0x4950 (Cypress FX3), product "Arducam B0495 (USB3
# 2.3MP)", manufacturer "Arducam".
_B0495_NATIVE_WIDTH = 1920
_B0495_NATIVE_HEIGHT = 1200
_B0495_MAX_RATE_HZ = 50.0


def arducam_b0495_spec(
    name: str = "arducam",
    parent_frame: str = "base_link",
    rate_hz: float = 30.0,
    width: int = _B0495_NATIVE_WIDTH,
    height: int = _B0495_NATIVE_HEIGHT,
    hfov_deg: float | None = None,
    serial: str = "",
) -> SensorSpec:
    """Build a ``SensorSpec`` for an Arducam B0495 USB3 global-shutter camera.

    Args:
        name: Sensor name — also the topic prefix and frame-id stem.
        parent_frame: tf2 parent frame for the static transform.
        rate_hz: Publish rate.  The device supports up to 50 Hz at the native
            1920×1200 and up to 80 Hz at 960×600; values above the mode's
            maximum are recorded as-is and will be clamped by the driver.
        width: Stream width.  Native is 1920; 960 is the other supported mode.
        height: Stream height.  Native is 1200; 600 is the other supported mode.
        hfov_deg: Horizontal field of view of the **mounted M12 lens**, in
            degrees.  Supply it to materialise nominal pinhole intrinsics;
            leave it ``None`` (the default) to leave ``intrinsics`` unset
            rather than record a guess.  Either way, replace with values from
            ``openral calibrate camera`` before deployment.
        serial: Optional USB serial string, recorded in ``metadata`` so a rig
            with several identical B0495s can address a specific one.

    Returns:
        A fully-materialised :class:`SensorSpec`.

    Example:
        >>> spec = arducam_b0495_spec(name="wrist", rate_hz=50.0)
        >>> spec.rate_hz
        50.0
        >>> spec.metadata["shutter"]
        'global'
    """
    intrinsics: IntrinsicsPinhole | None = None
    fov_v_deg: float | None = None
    if hfov_deg is not None:
        fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        intrinsics = IntrinsicsPinhole(
            width=width,
            height=height,
            fx=fx,
            fy=fx,
            cx=width / 2.0,
            cy=height / 2.0,
            distortion_model="plumb_bob",
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        fov_v_deg = math.degrees(2.0 * math.atan(height / (2.0 * fx)))

    metadata: dict[str, object] = {
        "shutter": "global",
        "image_sensor": "AR0234",
        "usb_bridge": "Cypress FX3 (USB 3.0)",
        "native_resolution": [_B0495_NATIVE_WIDTH, _B0495_NATIVE_HEIGHT],
        "max_rate_hz": _B0495_MAX_RATE_HZ,
        "lens_mount": "M12",
        "needs_calibration": intrinsics is None,
    }
    if serial:
        metadata["serial_no"] = serial

    return SensorSpec(
        name=name,
        modality=SensorModality.RGB,
        frame_id=f"{name}_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rate_hz,
        encoding="yuyv",
        intrinsics=intrinsics,
        fov_h_deg=hfov_deg,
        fov_v_deg=fov_v_deg,
        ros2_topic=f"/{name}/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        catalog_id="arducam/b0495",
        vendor="Arducam",
        model="B0495",
        driver_pkg="usb_cam",
        metadata=metadata,
    )


CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="arducam/b0495",
            vendor="arducam",
            model="b0495",
            kind="sensor",
            factory=arducam_b0495_spec,
            modalities=(SensorModality.RGB,),
            description=(
                "Arducam B0495 — 2.3 MP AR0234 global-shutter colour over USB 3.0 "
                "UVC (Cypress FX3); 1920×1200 @ 50 fps, 960×600 @ 80 fps, YUYV. "
                "M12 lens mount, so optics are integrator-supplied."
            ),
            docs_url="https://www.arducam.com/product/arducam-2-3mp-ar0234-global-shutter-usb3-camera/",
            signatures=(SensorSignature(kind="usb_uvc", value="0x04b4:0x4950"),),
        ),
    ]
)
