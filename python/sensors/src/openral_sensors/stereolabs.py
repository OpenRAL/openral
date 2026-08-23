"""StereoLabs ZED camera adapters — ZED Mini (ZED-M).

Provides:

- ``zed_mini_bundle`` — factory that builds a :class:`SensorBundle` for a
  StereoLabs ZED Mini (left + right rectified RGB, stereo depth, 6-DoF IMU).
- Catalog registration under ``stereolabs/zed_mini``.

The ZED Mini is a *passive* stereo camera: it emits no structured light and
no IR pattern, so unlike a RealSense or an OAK-D Pro it degrades on
untextured surfaces but never interferes with a second depth camera pointed
at the same workspace.  That property is why it coexists on a cell with
active sensors.

Depth is computed **on the host**, not on the device.  Over USB the camera
presents a single UVC node streaming the two eyes side-by-side in one YUYV
frame (2560×720 = 2×1280×720 at HD720); the ZED SDK splits and rectifies
that frame and runs stereo matching on the GPU.  A host without the SDK sees
one wide RGB camera and no depth at all — which is exactly what
``openral detect`` reports, and why the bundle records ``sdk_required``.

Example:
    >>> from openral_sensors.stereolabs import zed_mini_bundle
    >>> bundle = zed_mini_bundle(name="head", parent_frame="base_link")
    >>> bundle.bundle_name
    'head'
    >>> len(bundle.sensors)
    4
    >>> bundle.sensors[2].modality
    'depth'
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

__all__ = ["zed_mini_bundle"]


# ── Nominal intrinsics ────────────────────────────────────────────────────────
# Source: StereoLabs ZED Mini datasheet, cross-checked against the modes the
# device actually advertises over UVC (VIDIOC_ENUM_FRAMESIZES):
#   4416×1242 (2×2208×1242, "2.2K"), 3840×1080 (2×1920×1080, "HD1080"),
#   2560×720  (2×1280×720,  "HD720"), 1344×376  (2×672×376,   "WVGA").
# Max FoV 90° (H) × 60° (V) × 100° (D); stereo baseline 63 mm.
# Per-eye intrinsics below are at the HD720 default: fx = W / (2·tan(H/2))
#   = 1280 / (2·tan(45°)) = 640.
# Replace with calibrated values from `openral calibrate camera` — or read the
# factory calibration the SDK ships per serial number.
_ZED_MINI_EYE_INTRINSICS = IntrinsicsPinhole(
    width=1280,
    height=720,
    fx=640.0,
    fy=640.0,
    cx=640.0,
    cy=360.0,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)

_ZED_MINI_BASELINE_M = 0.063
_ZED_MINI_DEPTH_MIN_M = 0.10
_ZED_MINI_DEPTH_MAX_M = 15.0
_ZED_MINI_FOV_H_DEG = 90.0
_ZED_MINI_FOV_V_DEG = 60.0


def zed_mini_bundle(
    name: str = "zed",
    parent_frame: str = "base_link",
    serial: str = "",
    rgb_rate_hz: float = 30.0,
    depth_rate_hz: float = 30.0,
    imu_rate_hz: float = 400.0,
    width: int = 1280,
    height: int = 720,
) -> SensorBundle:
    """Build a :class:`SensorBundle` for a StereoLabs ZED Mini.

    The bundle exposes four streams: left and right rectified RGB, host-computed
    stereo depth, and the on-device 6-DoF IMU.  Nominal intrinsics are the
    datasheet values at HD720; replace with calibrated values before deploying.

    Args:
        name: Bundle name — used as a prefix for topic names and frame ids
            (e.g. ``"zed_head"`` → ``/zed_head/left/image_rect_color``).
        parent_frame: tf2 parent frame for the static transform.
        serial: Optional ZED serial number for multi-camera hosts.  Stored in
            ``metadata.serial_no`` and forwarded to the ROS driver's
            ``serial_number`` parameter.
        rgb_rate_hz: Per-eye RGB publish rate.
        depth_rate_hz: Depth publish rate.
        imu_rate_hz: IMU publish rate.
        width: Per-eye stream width.  The factory rescales the nominal
            intrinsics linearly when this differs from the HD720 default.
        height: Per-eye stream height.

    Returns:
        A :class:`SensorBundle` with ``sync="hardware"`` — the two eyes come
        out of one sensor pair in a single UVC frame, so they are exposed
        simultaneously by construction.

    Example:
        >>> bundle = zed_mini_bundle(name="head")
        >>> bundle.sensors[2].range_max_m
        15.0
        >>> bundle.sensors[0].metadata["baseline_m"]
        0.063
    """
    intrinsics = scale_intrinsics_to(_ZED_MINI_EYE_INTRINSICS, width, height)
    meta: dict[str, object] = {
        "baseline_m": _ZED_MINI_BASELINE_M,
        "stereo": "passive",
        # Depth and rectification run on the host GPU via the ZED SDK; a bare
        # UVC host gets one side-by-side RGB stream and nothing else.
        "sdk_required": "zed-sdk",
        "uvc_side_by_side": [width * 2, height],
    }
    if serial:
        meta["serial_no"] = serial

    def _eye(side: str) -> SensorSpec:
        return SensorSpec(
            name=f"{name}_{side}",
            modality=SensorModality.RGB,
            frame_id=f"{name}_{side}_camera_optical_frame",
            parent_frame=parent_frame,
            rate_hz=rgb_rate_hz,
            encoding="rgb8",
            intrinsics=intrinsics,
            fov_h_deg=_ZED_MINI_FOV_H_DEG,
            fov_v_deg=_ZED_MINI_FOV_V_DEG,
            ros2_topic=f"/{name}/{side}/image_rect_color",
            ros2_msg_type="sensor_msgs/Image",
            vendor="StereoLabs",
            model="ZED Mini",
            driver_pkg="zed_wrapper",
            metadata=meta,
        )

    depth = SensorSpec(
        name=f"{name}_depth",
        modality=SensorModality.DEPTH,
        frame_id=f"{name}_left_camera_optical_frame",
        parent_frame=parent_frame,
        rate_hz=depth_rate_hz,
        encoding="32FC1",
        intrinsics=intrinsics,
        fov_h_deg=_ZED_MINI_FOV_H_DEG,
        fov_v_deg=_ZED_MINI_FOV_V_DEG,
        range_min_m=_ZED_MINI_DEPTH_MIN_M,
        range_max_m=_ZED_MINI_DEPTH_MAX_M,
        ros2_topic=f"/{name}/depth/depth_registered",
        ros2_msg_type="sensor_msgs/Image",
        vendor="StereoLabs",
        model="ZED Mini",
        driver_pkg="zed_wrapper",
        metadata=meta,
    )
    imu = SensorSpec(
        name=f"{name}_imu",
        modality=SensorModality.IMU,
        frame_id=f"{name}_imu_link",
        parent_frame=parent_frame,
        rate_hz=imu_rate_hz,
        ros2_topic=f"/{name}/imu/data",
        ros2_msg_type="sensor_msgs/Imu",
        vendor="StereoLabs",
        model="ZED Mini (integrated IMU)",
        driver_pkg="zed_wrapper",
        metadata=meta,
    )
    return SensorBundle(
        bundle_name=name,
        sensors=[_eye("left"), _eye("right"), depth, imu],
        sync="hardware",
        sync_tolerance_ms=1.0,
    )


CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="stereolabs/zed_mini",
            vendor="stereolabs",
            model="zed_mini",
            kind="bundle",
            factory=zed_mini_bundle,
            modalities=(SensorModality.RGB, SensorModality.DEPTH, SensorModality.IMU),
            description=(
                "StereoLabs ZED Mini — passive stereo, 63 mm baseline, up to "
                "2×2208×1242; host-side (GPU) depth 0.10–15 m, 90°×60° FoV, "
                "integrated 6-DoF IMU. Requires the ZED SDK for depth."
            ),
            docs_url="https://www.stereolabs.com/products/zed-mini",
            signatures=(
                SensorSignature(kind="usb_uvc", value="0x2b03:0xf682"),
                SensorSignature(kind="v4l2_name", value="ZED-M"),
            ),
        ),
    ]
)
