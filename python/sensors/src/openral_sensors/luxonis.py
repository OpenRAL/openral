"""Luxonis OAK-D family sensor adapters.

Provides:

- ``oak_d_pro_bundle`` — factory that builds a :class:`SensorBundle` for a
  Luxonis OAK-D Pro (RGB + stereo depth + IMU), with nominal intrinsics
  from the Luxonis datasheet.
- Catalog registration under ``luxonis/oak_d_pro``.

The OAK-D Pro is the overhead RGB-D camera used by the ``so101_box``
sim scene (and the recommended pairing for the SO-101 follower in real
hardware). The data sheet values used here are at the default stream
resolutions (1920×1080 RGB, 1280×800 depth); replace with calibrated
values from ``openral calibrate camera`` before deployment.

The OAK-D Pro has an embedded RVC2 SoC that runs depth + AI on-device,
so the ROS-side driver (``depthai_ros_driver`` /
``depthai_ros_examples``) publishes RGB + depth + IMU streams without
needing host-side stereo processing.

Example:
    >>> from openral_sensors.luxonis import oak_d_pro_bundle
    >>> bundle = oak_d_pro_bundle(name="oak_top", parent_frame="box_ceiling")
    >>> bundle.bundle_name
    'oak_top'
    >>> len(bundle.sensors)
    3
    >>> bundle.sensors[0].modality
    'rgb'
"""

from __future__ import annotations

from openral_core.schemas import (
    IntrinsicsPinhole,
    SensorBundle,
    SensorModality,
    SensorSpec,
    scale_intrinsics_to,
)

from openral_sensors.catalog import CATALOG, SensorCatalogEntry, SensorSignature

__all__ = [
    "oak_d_pro_bundle",
]


# ── Nominal intrinsics ────────────────────────────────────────────────────────
# Source: Luxonis OAK-D Pro datasheet, RVC2 reference design (2024).
# Replace with calibrated values from `openral calibrate camera`.

# RGB (IMX378, 12 MP sensor) at the default 1920×1080 stream:
#   HFoV 95°, VFoV 70°.  fx = W / (2 * tan(HFoV / 2)).
_OAK_D_PRO_RGB_INTRINSICS = IntrinsicsPinhole(
    width=1920,
    height=1080,
    fx=873.0,  # 1920 / (2 * tan(95°/2)) ≈ 873
    fy=873.0,
    cx=960.0,
    cy=540.0,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)
# Stereo depth (OV9282 global-shutter, 1 MP) at native 1280×800:
#   HFoV 71.86°, VFoV 56°.  Baseline 7.5 cm.
_OAK_D_PRO_DEPTH_INTRINSICS = IntrinsicsPinhole(
    width=1280,
    height=800,
    fx=884.0,  # 1280 / (2 * tan(71.86°/2)) ≈ 884
    fy=884.0,
    cx=640.0,
    cy=400.0,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)


# ── Bundle factories ──────────────────────────────────────────────────────────


def oak_d_pro_bundle(
    name: str = "oak",
    parent_frame: str = "base_link",
    mxid: str = "",
    rgb_rate_hz: float = 30.0,
    depth_rate_hz: float = 30.0,
    imu_rate_hz: float = 400.0,
    rgb_width: int = 1920,
    rgb_height: int = 1080,
    depth_width: int = 1280,
    depth_height: int = 800,
) -> SensorBundle:
    """Build a :class:`SensorBundle` for a Luxonis OAK-D Pro.

    The bundle exposes three streams: RGB (IMX378), depth (OV9282
    global-shutter stereo) and the on-device BNO086 IMU. Nominal
    intrinsics are from the Luxonis datasheet at the default
    resolutions; replace with calibrated values before deploying to
    hardware.

    Args:
        name: Bundle name — used as a prefix for topic names and frame
            ids (e.g. ``"oak_top"`` → ``/oak_top/rgb/image_raw``).
        parent_frame: tf2 parent frame for the static transform.
        mxid: Optional Luxonis MXID for device targeting (multi-device
            hosts). Stored in ``metadata.mxid`` and forwarded to the
            ROS driver's ``i_mxid`` parameter.
        rgb_rate_hz: RGB stream publish rate.
        depth_rate_hz: Depth stream publish rate.
        imu_rate_hz: IMU publish rate (BNO086).
        rgb_width: RGB stream width. The factory rescales the nominal
            intrinsics linearly when this differs from the default
            1920.
        rgb_height: RGB stream height.
        depth_width: Depth stream width.
        depth_height: Depth stream height.

    Returns:
        A :class:`SensorBundle` with ``sync="hardware"`` and 5 ms
        tolerance (RVC2 syncs RGB + depth on-device).

    Example:
        >>> bundle = oak_d_pro_bundle(name="oak_top", parent_frame="box_ceiling")
        >>> bundle.sensors[1].modality
        'depth'
        >>> bundle.sensors[1].range_max_m
        19.0
    """
    meta: dict[str, object] = {"mxid": mxid} if mxid else {}

    rgb_intr = _scale_intrinsics(_OAK_D_PRO_RGB_INTRINSICS, rgb_width, rgb_height)
    depth_intr = _scale_intrinsics(_OAK_D_PRO_DEPTH_INTRINSICS, depth_width, depth_height)

    rgb = SensorSpec(
        name=f"{name}_color",
        modality=SensorModality.RGB,
        frame_id=f"{name}_rgb_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rgb_rate_hz,
        encoding="rgb8",
        intrinsics=rgb_intr,
        fov_h_deg=95.0,
        fov_v_deg=70.0,
        ros2_topic=f"/{name}/rgb/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Luxonis",
        model="OAK-D Pro",
        driver_pkg="depthai_ros_driver",
        metadata=meta,
    )
    depth = SensorSpec(
        name=f"{name}_depth",
        modality=SensorModality.DEPTH,
        frame_id=f"{name}_depth_optical_frame",
        parent_frame=parent_frame,
        rate_hz=depth_rate_hz,
        encoding="16UC1",
        intrinsics=depth_intr,
        fov_h_deg=71.86,
        fov_v_deg=56.0,
        range_min_m=0.20,
        range_max_m=19.0,
        ros2_topic=f"/{name}/stereo/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Luxonis",
        model="OAK-D Pro",
        driver_pkg="depthai_ros_driver",
        metadata=meta,
    )
    imu = SensorSpec(
        name=f"{name}_imu",
        modality=SensorModality.IMU,
        frame_id=f"{name}_imu_frame",
        parent_frame=parent_frame,
        rate_hz=imu_rate_hz,
        accel_noise_density=3.0e-3,
        gyro_noise_density=3.5e-3,
        ros2_topic=f"/{name}/imu/data",
        ros2_msg_type="sensor_msgs/Imu",
        vendor="Luxonis",
        model="OAK-D Pro (BNO086 IMU)",
        driver_pkg="depthai_ros_driver",
        metadata=meta,
    )
    return SensorBundle(
        bundle_name=name,
        sensors=[rgb, depth, imu],
        sync="hardware",
        sync_tolerance_ms=5.0,
    )


def _scale_intrinsics(base: IntrinsicsPinhole, width: int, height: int) -> IntrinsicsPinhole:
    """Rescale a pinhole intrinsics tuple linearly to a new (width, height).

    Used when a caller picks a non-default stream resolution. Thin wrapper over
    :func:`openral_core.scale_intrinsics_to` (the canonical linear-rescale
    helper); the distortion model is preserved and coefficients are assumed
    normalised.
    """
    return scale_intrinsics_to(base, width, height)


CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="luxonis/oak_d_pro",
            vendor="luxonis",
            model="oak_d_pro",
            kind="bundle",
            factory=oak_d_pro_bundle,
            modalities=(SensorModality.RGB, SensorModality.DEPTH, SensorModality.IMU),
            description=(
                "Luxonis OAK-D Pro — 12 MP IMX378 RGB + 1 MP OV9282 global-shutter "
                "stereo depth (0.20–19 m, 71.86°×56°), 850 nm IR flood + dot "
                "projector, on-device RVC2 SoC, BNO086 IMU."
            ),
            docs_url="https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9098/",
            signatures=(
                # Luxonis OAK family enumerates over USB; the upstream
                # ``depthai`` SDK identifies devices by ``mxid`` rather than
                # VID/PID, so the v4l2 product string is the practical
                # ``openral detect`` discriminator.
                SensorSignature(kind="v4l2_name", value="OAK-D Pro"),
                SensorSignature(kind="usb_uvc", value="0x03e7:0x2485"),
            ),
        ),
    ]
)
