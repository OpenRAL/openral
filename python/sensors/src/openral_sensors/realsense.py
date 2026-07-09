"""RealSense sensor adapters for openral.

Provides:
- Factory functions that build ``SensorBundle`` objects for Intel RealSense
  cameras (D415, D435, D435i) using nominal intrinsics from the Intel data
  sheets.
- ``bundle_to_node_params`` — maps a bundle to ``realsense2_camera`` node
  parameters suitable for a ROS 2 launch file.
- ``generate_launch_py`` — emits a complete Python ROS 2 launch file as a
  string.  Does **not** import ``launch`` so it can run without ROS 2.
- ``calibrate_camera_cmd`` — builds the ``ros2 run camera_calibration
  cameracalibrator`` argument list for a given ``SensorSpec``.

Notes on nominal intrinsics
---------------------------
The intrinsics used here are *nominal* values from the Intel RealSense
data sheet at 640x480.  They vary by unit and should be replaced with
values from ``openral calibrate camera`` before deployment.

Example:
    >>> from openral_sensors.realsense import realsense_d435_bundle
    >>> bundle = realsense_d435_bundle(name="head", parent_frame="base_link")
    >>> bundle.bundle_name
    'head'
    >>> len(bundle.sensors)
    3
    >>> bundle.sensors[0].modality
    'rgb'
"""

from __future__ import annotations

import textwrap

from openral_core.schemas import (
    IntrinsicsPinhole,
    SensorBundle,
    SensorModality,
    SensorSpec,
)

__all__ = [
    "bundle_to_node_params",
    "calibrate_camera_cmd",
    "generate_launch_py",
    "realsense_d415_bundle",
    "realsense_d435_bundle",
    "realsense_d435i_bundle",
]

# Type alias for the node parameter dict accepted by realsense2_camera.
NodeParams = dict[str, str | int | float | bool]

# ── Nominal intrinsics ────────────────────────────────────────────────────────
# Source: Intel RealSense D400 Series Datasheet rev 9 (2023), §4.3.
# Replace with calibrated values from `openral calibrate camera`.

_D435_RGB_INTRINSICS = IntrinsicsPinhole(
    width=640,
    height=480,
    fx=617.3,
    fy=617.6,
    cx=321.0,
    cy=240.3,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)
_D435_DEPTH_INTRINSICS = IntrinsicsPinhole(
    width=640,
    height=480,
    fx=383.5,
    fy=383.5,
    cx=320.3,
    cy=240.1,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)
# D415 (rolling-shutter IR-stereo, 65°×40°): nominal at 640×480.
_D415_RGB_INTRINSICS = IntrinsicsPinhole(
    width=640,
    height=480,
    fx=617.0,
    fy=617.0,
    cx=320.0,
    cy=240.0,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)
_D415_DEPTH_INTRINSICS = IntrinsicsPinhole(
    width=640,
    height=480,
    fx=596.0,
    fy=596.0,
    cx=320.0,
    cy=240.0,
    distortion_model="plumb_bob",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
)


# ── Bundle factories ──────────────────────────────────────────────────────────


def realsense_d435_bundle(
    name: str = "realsense",
    parent_frame: str = "base_link",
    serial_no: str = "",
    rgb_rate_hz: float = 30.0,
    depth_rate_hz: float = 30.0,
    imu_rate_hz: float = 400.0,
) -> SensorBundle:
    """Build a ``SensorBundle`` for an Intel RealSense D435.

    The bundle contains three sensors: RGB, depth, and IMU.  Nominal
    intrinsics are from the Intel D400 Series data sheet at 640x480; replace
    them with calibrated values before deploying to hardware.

    Args:
        name: Sensor bundle name, used as a prefix for topic names and frame
            IDs (e.g. ``"head"`` → ``/head/color/image_raw``).
        parent_frame: tf2 parent frame for the static transform.
        serial_no: Optional serial number for the device.  Stored in metadata
            and forwarded to ``realsense2_camera`` as ``serial_no``.
        rgb_rate_hz: RGB stream publish rate.
        depth_rate_hz: Depth stream publish rate.
        imu_rate_hz: IMU publish rate.

    Returns:
        A ``SensorBundle`` with ``sync="hardware"`` and 5 ms tolerance.

    Example:
        >>> bundle = realsense_d435_bundle(name="wrist", parent_frame="ee_link")
        >>> bundle.sensors[0].frame_id
        'wrist_color_optical_frame'
        >>> bundle.sensors[1].modality
        'depth'
        >>> bundle.sensors[2].rate_hz
        400.0
    """
    meta: dict[str, object] = {"serial_no": serial_no} if serial_no else {}
    rgb = SensorSpec(
        name=f"{name}_color",
        modality=SensorModality.RGB,
        frame_id=f"{name}_color_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rgb_rate_hz,
        encoding="rgb8",
        intrinsics=_D435_RGB_INTRINSICS,
        fov_h_deg=69.4,
        fov_v_deg=42.5,
        ros2_topic=f"/{name}/color/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Intel",
        model="RealSense D435",
        driver_pkg="realsense2_camera",
        metadata=meta,
    )
    depth = SensorSpec(
        name=f"{name}_depth",
        modality=SensorModality.DEPTH,
        frame_id=f"{name}_depth_optical_frame",
        parent_frame=parent_frame,
        rate_hz=depth_rate_hz,
        encoding="16UC1",
        intrinsics=_D435_DEPTH_INTRINSICS,
        fov_h_deg=87.0,
        fov_v_deg=58.0,
        range_min_m=0.1,
        range_max_m=10.0,
        ros2_topic=f"/{name}/depth/image_rect_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Intel",
        model="RealSense D435",
        driver_pkg="realsense2_camera",
        metadata=meta,
    )
    imu = SensorSpec(
        name=f"{name}_imu",
        modality=SensorModality.IMU,
        frame_id=f"{name}_imu_optical_frame",
        parent_frame=parent_frame,
        rate_hz=imu_rate_hz,
        accel_noise_density=2.0e-3,
        gyro_noise_density=5.0e-3,
        ros2_topic=f"/{name}/imu",
        ros2_msg_type="sensor_msgs/Imu",
        vendor="Intel",
        model="RealSense D435",
        driver_pkg="realsense2_camera",
        metadata=meta,
    )
    return SensorBundle(
        bundle_name=name,
        sensors=[rgb, depth, imu],
        sync="hardware",
        sync_tolerance_ms=5.0,
    )


# ── Launch generator ──────────────────────────────────────────────────────────


def bundle_to_node_params(bundle: SensorBundle, serial_no: str = "") -> NodeParams:
    """Map a RealSense ``SensorBundle`` to ``realsense2_camera`` node parameters.

    Only bundles whose sensors carry ``driver_pkg="realsense2_camera"`` are
    supported.  Parameters are derived from the first RGB and depth
    ``SensorSpec`` found in the bundle.

    Args:
        bundle: A ``SensorBundle`` produced by :func:`realsense_d435_bundle`,
            :func:`realsense_d435i_bundle`, or :func:`realsense_d415_bundle`.
        serial_no: Device serial number.  Overrides any value stored in
            sensor metadata.

    Returns:
        A flat parameter dict suitable for the ``realsense2_camera``
        composable node.

    Raises:
        ValueError: If no RGB sensor is found in the bundle.

    Example:
        >>> bundle = realsense_d435_bundle(name="head")
        >>> params = bundle_to_node_params(bundle)
        >>> params["camera_name"]
        'head'
        >>> params["rgb_camera.color_profile"]
        '640x480x30'
    """
    rgb = next(
        (s for s in bundle.sensors if s.modality == SensorModality.RGB),
        None,
    )
    if rgb is None:
        raise ValueError(f"SensorBundle '{bundle.bundle_name}' has no RGB sensor.")

    depth = next(
        (s for s in bundle.sensors if s.modality == SensorModality.DEPTH),
        None,
    )
    imu = next(
        (s for s in bundle.sensors if s.modality == SensorModality.IMU),
        None,
    )

    # Resolve serial number: explicit arg > metadata > empty
    sn = serial_no or str(rgb.metadata.get("serial_no", ""))

    rgb_w = rgb.intrinsics.width if rgb.intrinsics else 640
    rgb_h = rgb.intrinsics.height if rgb.intrinsics else 480
    rgb_fps = int(rgb.rate_hz)

    params: NodeParams = {
        "camera_name": bundle.bundle_name,
        "camera_namespace": f"/{bundle.bundle_name}",
        "serial_no": sn,
        "enable_color": True,
        "rgb_camera.color_profile": f"{rgb_w}x{rgb_h}x{rgb_fps}",
        "enable_depth": depth is not None,
        "enable_gyro": imu is not None,
        "enable_accel": imu is not None,
        "unite_imu_method": "1",  # linear interpolation
        "align_depth.enable": True,
        "pointcloud.enable": False,
    }

    if depth is not None:
        d_w = depth.intrinsics.width if depth.intrinsics else 640
        d_h = depth.intrinsics.height if depth.intrinsics else 480
        d_fps = int(depth.rate_hz)
        params["depth_module.depth_profile"] = f"{d_w}x{d_h}x{d_fps}"

    return params


def generate_launch_py(bundle: SensorBundle, serial_no: str = "") -> str:
    """Generate a ROS 2 Python launch file string for a RealSense bundle.

    The generated file uses the ``realsense2_camera`` composable node and
    does **not** depend on ``launch`` at generation time — this function is
    fully testable without a ROS 2 installation.

    Args:
        bundle: A ``SensorBundle`` for the target camera.
        serial_no: Optional device serial number forwarded to the node.

    Returns:
        A Python string that, when written to ``<name>.launch.py`` and
        executed with ``ros2 launch``, starts the camera driver with the
        correct parameters.

    Example:
        >>> bundle = realsense_d435_bundle(name="head")
        >>> src = generate_launch_py(bundle)
        >>> "realsense2_camera" in src
        True
        >>> "head" in src
        True
        >>> src.startswith("# Generated by OpenRAL")
        True
    """
    params = bundle_to_node_params(bundle, serial_no=serial_no)

    # Format the parameter dict as Python literal lines for the launch file.
    param_lines = "\n".join(f"            {k!r}: {v!r}," for k, v in params.items())

    return textwrap.dedent(
        f"""\
        # Generated by OpenRAL ral sensors — do not edit by hand.
        # Re-generate with: ral launch sensor --bundle {bundle.bundle_name}
        from launch import LaunchDescription
        from launch_ros.actions import ComposableNodeContainer
        from launch_ros.descriptions import ComposableNode


        def generate_launch_description() -> LaunchDescription:
            container = ComposableNodeContainer(
                name="{bundle.bundle_name}_container",
                namespace="/{bundle.bundle_name}",
                package="rclcpp_components",
                executable="component_container",
                composable_node_descriptions=[
                    ComposableNode(
                        package="realsense2_camera",
                        plugin="realsense2_camera::RealSenseNodeFactory",
                        name="{bundle.bundle_name}",
                        parameters=[
                            {{
        {param_lines}
                            }}
                        ],
                        extra_arguments=[{{"use_intra_process_comms": True}}],
                    ),
                ],
                output="screen",
            )
            return LaunchDescription([container])
        """
    )


# ── Calibration helper ────────────────────────────────────────────────────────


def calibrate_camera_cmd(
    sensor: SensorSpec,
    chessboard_cols: int = 8,
    chessboard_rows: int = 6,
    square_size_m: float = 0.025,
) -> list[str]:
    """Build a ``ros2 run camera_calibration cameracalibrator`` command.

    The command can be passed to ``subprocess.run`` or printed with
    ``--dry-run`` so the user can inspect it before execution.

    Args:
        sensor: The RGB ``SensorSpec`` to calibrate.
        chessboard_cols: Number of internal corners along the long axis.
        chessboard_rows: Number of internal corners along the short axis.
        square_size_m: Physical size of one chessboard square in metres.

    Returns:
        A list of command tokens suitable for ``subprocess.run``.

    Raises:
        ValueError: If ``sensor.modality`` is not ``RGB``.

    Example:
        >>> from openral_sensors.realsense import realsense_d435_bundle
        >>> bundle = realsense_d435_bundle(name="head")
        >>> rgb_spec = bundle.sensors[0]
        >>> cmd = calibrate_camera_cmd(rgb_spec, chessboard_cols=8, chessboard_rows=6)
        >>> cmd[0]
        'ros2'
        >>> "--size" in cmd
        True
        >>> "8x6" in cmd
        True
    """
    if sensor.modality != SensorModality.RGB:
        raise ValueError(
            f"Sensor '{sensor.name}' has modality '{sensor.modality}', "
            "expected 'rgb' for camera calibration."
        )

    size_arg = f"{chessboard_cols}x{chessboard_rows}"
    square_arg = str(square_size_m)

    # Derive camera_info topic from image topic convention
    # e.g. /head/color/image_raw → /head/color/camera_info
    topic = sensor.ros2_topic
    if topic is None:
        raise ValueError(
            f"Sensor '{sensor.name}' has no ros2_topic; cannot derive camera_info remap."
        )
    info_topic = topic.replace("/image_raw", "/camera_info").replace(
        "/image_rect_raw", "/camera_info"
    )

    return [
        "ros2",
        "run",
        "camera_calibration",
        "cameracalibrator",
        "--size",
        size_arg,
        "--square",
        square_arg,
        "--ros-args",
        "-r",
        f"image:={topic}",
        "-r",
        f"camera_info:={info_topic}",
    ]


# ── D435i / D415 bundle factories ─────────────────────────────────────────────


def realsense_d435i_bundle(
    name: str = "realsense",
    parent_frame: str = "base_link",
    serial_no: str = "",
    rgb_rate_hz: float = 30.0,
    depth_rate_hz: float = 30.0,
    imu_rate_hz: float = 400.0,
) -> SensorBundle:
    """Build a ``SensorBundle`` for an Intel RealSense D435i.

    The D435i is a D435 with an integrated Bosch BMI085 IMU; the existing
    :func:`realsense_d435_bundle` already includes the IMU sensor, so this
    factory delegates to it and stamps ``model="RealSense D435i"`` on each
    constituent ``SensorSpec``.

    Example:
        >>> bundle = realsense_d435i_bundle(name="wrist")
        >>> bundle.sensors[0].model
        'RealSense D435i'
    """
    bundle = realsense_d435_bundle(
        name=name,
        parent_frame=parent_frame,
        serial_no=serial_no,
        rgb_rate_hz=rgb_rate_hz,
        depth_rate_hz=depth_rate_hz,
        imu_rate_hz=imu_rate_hz,
    )
    return bundle.model_copy(
        update={
            "sensors": [s.model_copy(update={"model": "RealSense D435i"}) for s in bundle.sensors],
        }
    )


def realsense_d415_bundle(
    name: str = "realsense",
    parent_frame: str = "base_link",
    serial_no: str = "",
    rgb_rate_hz: float = 30.0,
    depth_rate_hz: float = 30.0,
) -> SensorBundle:
    """Build a ``SensorBundle`` for an Intel RealSense D415.

    Rolling-shutter IR-stereo camera, 65°×40° FOV, 0.16 – 10 m range; no IMU.

    Example:
        >>> bundle = realsense_d415_bundle(name="head")
        >>> bundle.sensors[1].range_min_m
        0.16
    """
    meta: dict[str, object] = {"serial_no": serial_no} if serial_no else {}
    rgb = SensorSpec(
        name=f"{name}_color",
        modality=SensorModality.RGB,
        frame_id=f"{name}_color_optical_frame",
        parent_frame=parent_frame,
        rate_hz=rgb_rate_hz,
        encoding="rgb8",
        intrinsics=_D415_RGB_INTRINSICS,
        fov_h_deg=65.0,
        fov_v_deg=40.0,
        ros2_topic=f"/{name}/color/image_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Intel",
        model="RealSense D415",
        driver_pkg="realsense2_camera",
        metadata=meta,
    )
    depth = SensorSpec(
        name=f"{name}_depth",
        modality=SensorModality.DEPTH,
        frame_id=f"{name}_depth_optical_frame",
        parent_frame=parent_frame,
        rate_hz=depth_rate_hz,
        encoding="16UC1",
        intrinsics=_D415_DEPTH_INTRINSICS,
        fov_h_deg=65.0,
        fov_v_deg=40.0,
        range_min_m=0.16,
        range_max_m=10.0,
        ros2_topic=f"/{name}/depth/image_rect_raw",
        ros2_msg_type="sensor_msgs/Image",
        vendor="Intel",
        model="RealSense D415",
        driver_pkg="realsense2_camera",
        metadata=meta,
    )
    return SensorBundle(
        bundle_name=name,
        sensors=[rgb, depth],
        sync="hardware",
        sync_tolerance_ms=5.0,
    )


# ── Catalog registration ──────────────────────────────────────────────────────

from openral_sensors.catalog import (  # noqa: E402  # reason: avoid circular import at top
    CATALOG,
    SensorCatalogEntry,
    SensorSignature,
)

CATALOG.register_many(
    [
        SensorCatalogEntry(
            id="intel/realsense_d435",
            vendor="intel",
            model="realsense_d435",
            kind="bundle",
            factory=realsense_d435_bundle,
            modalities=(SensorModality.RGB, SensorModality.DEPTH, SensorModality.IMU),
            description="Intel RealSense D435 — 87°×58° active IR stereo, 0.1–10 m, with IMU.",
            docs_url="https://www.intelrealsense.com/depth-camera-d435/",
            signatures=(SensorSignature(kind="realsense", value="D435"),),
        ),
        SensorCatalogEntry(
            id="intel/realsense_d435i",
            vendor="intel",
            model="realsense_d435i",
            kind="bundle",
            factory=realsense_d435i_bundle,
            modalities=(SensorModality.RGB, SensorModality.DEPTH, SensorModality.IMU),
            description="Intel RealSense D435i — D435 + Bosch BMI085 IMU.",
            docs_url="https://www.intelrealsense.com/depth-camera-d435i/",
            signatures=(SensorSignature(kind="realsense", value="D435I"),),
        ),
        SensorCatalogEntry(
            id="intel/realsense_d415",
            vendor="intel",
            model="realsense_d415",
            kind="bundle",
            factory=realsense_d415_bundle,
            modalities=(SensorModality.RGB, SensorModality.DEPTH),
            description="Intel RealSense D415 — rolling-shutter IR stereo, 65°×40°, 0.16–10 m.",
            docs_url="https://www.intelrealsense.com/depth-camera-d415/",
            signatures=(SensorSignature(kind="realsense", value="D415"),),
        ),
    ]
)
