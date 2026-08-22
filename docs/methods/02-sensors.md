# Layer 2 — Sensors

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/sensors/src/openral_sensors/catalog.py`
_Sensor catalog — vendor-agnostic registry of `SensorSpec` / `SensorBundle` factories._

- `class SensorSignature` — Probe-side identifier (kind + canonical value) for catalog reverse-lookup. (L59)
  fields: `kind, value`
- `class SensorCatalogEntry` — One row in the catalog. (L88)
  fields: `id, vendor, model, kind, factory, modalities, description, docs_url, signatures`
- `class SensorCatalog` — In-memory registry. (L122)
  - `register(entry, *, replace=False) -> SensorCatalogEntry` (L152)
  - `unregister(sensor_id) -> None` (idempotent) (L175)
  - `get(sensor_id) -> SensorCatalogEntry` — Raises `KeyError` on miss. (L181)
  - `__contains__(sensor_id) -> bool` (L190)
  - `__len__() -> int` (L194)
  - `__iter__() -> object` (L198)
  - `list_ids() -> list[str]` — Sorted alphabetically. (L202)
  - `entries() -> list[SensorCatalogEntry]` — Sorted by id. (L206)
  - `filter(*, vendor=None, modality=None, kind=None) -> list[SensorCatalogEntry]` (L210)
  - `find_by_signature(signature) -> SensorCatalogEntry | None` — Reverse-lookup for `openral detect`. (L229)
  - `build(sensor_id, **kwargs) -> SensorSpec | SensorBundle` (L250)
- const `CATALOG = SensorCatalog()` — global singleton. (L262)

### Sensor `SensorSpec` factories — single-modality

> Only the factories used by an active HAL adapter remain.  Speculative
> vendor modules (orbbec, hokuyo, slamtec, livox, ouster, imu, tactile) were
> deleted; reintroduce them when a robot manifest needs them.

#### `python/sensors/src/openral_sensors/force_torque.py`
- `robotiq_ft300s_spec(name='wrist_ft', parent_frame='ee_link', rate_hz=100.0) -> SensorSpec` — Robotiq FT 300-S, 6-axis, 100 Hz, UR-native. (L25)

#### `python/sensors/src/openral_sensors/usb_uvc.py`
- `_uvc_intrinsics(width, height, hfov_deg) -> IntrinsicsPinhole` — Nominal pinhole intrinsics from sensor dims + hFOV. (L34)
- `logitech_c920_spec(name='usb_cam', parent_frame='base_link', rate_hz=30.0, width=1920, height=1080) -> SensorSpec` — Logitech C920 / C920e, 1080p UVC, 78° hFOV. (L49)
- `generic_uvc_rgb_spec(name='usb_cam', parent_frame='base_link', rate_hz=30.0, width=640, height=480, hfov_deg=70.0) -> SensorSpec` — Generic USB UVC RGB camera for calibrated robot-mounted cameras without stable vendor/model provenance; registered as `generic/usb_uvc_rgb`. (L75)

### Sensor `SensorBundle` factories — multi-modality

#### `python/sensors/src/openral_sensors/realsense.py`
- `realsense_d435_bundle(name='realsense', parent_frame='base_link', serial_no='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0) -> SensorBundle` (L104)
- `realsense_d435i_bundle(...) -> SensorBundle` — D435 + Bosch BMI085 IMU; delegates to `realsense_d435_bundle`. (L421)
- `realsense_d415_bundle(...) -> SensorBundle` — rolling-shutter IR stereo, 65°×40°, no IMU. (L456)
- `bundle_to_node_params(bundle, serial_no='') -> NodeParams` — Map to `realsense2_camera` node params. (L203)
- `generate_launch_py(bundle, serial_no='') -> str` — Auto-generated ROS 2 launch file. (L277)
- `calibrate_camera_cmd(sensor, chessboard_cols=8, chessboard_rows=6, square_size_m=0.025) -> list[str]` — Build `ros2 run camera_calibration cameracalibrator` argv. (L346)

#### `python/sensors/src/openral_sensors/luxonis.py`
- `oak_d_pro_bundle(name='oak', parent_frame='base_link', mxid='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0, rgb_width=1920, rgb_height=1080, depth_width=1280, depth_height=800) -> SensorBundle` — Luxonis OAK-D Pro RGB + global-shutter stereo depth (0.20–19 m, 71.86°×56°) + BNO086 IMU bundle, with nominal IMX378 / OV9282 intrinsics from the datasheet, linearly rescaled to non-default stream resolutions. Registered in the catalog as `luxonis/oak_d_pro`; recommended overhead RGB-D for the `so101_box` scene. (L82)
- `_scale_intrinsics(base, width, height) -> IntrinsicsPinhole` — Thin wrapper delegating to `openral_core.scale_intrinsics_to`; lets a caller pick a non-default stream resolution and still get a self-consistent (fx, fy, cx, cy). (L194)

#### `python/sensors/src/openral_sensors/stereolabs.py`
- `zed_mini_bundle(name='zed', parent_frame='base_link', serial='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0, width=1280, height=720) -> SensorBundle` — StereoLabs ZED Mini: left + right rectified RGB, host-computed stereo depth (0.10–15 m, 90°×60°), integrated 6-DoF IMU. Passive stereo — no IR pattern, so it degrades on untextured surfaces but never interferes with an active depth camera aimed at the same workspace. Over USB the camera is a **single** UVC node streaming both eyes side-by-side in one YUYV frame (2560×720 at HD720); rectification and depth run on the host GPU via the ZED SDK, recorded as `metadata.sdk_required`. Registered as `stereolabs/zed_mini`. (L76)

#### `python/sensors/src/openral_sensors/arducam.py`
- `arducam_b0495_spec(name='arducam', parent_frame='base_link', rate_hz=30.0, width=1920, height=1200, hfov_deg=None, serial='') -> SensorSpec` — Arducam B0495: 2.3 MP AR0234 **global-shutter** colour over USB 3.0 UVC (Cypress FX3); 1920×1200 @ 50 fps, 960×600 @ 80 fps, YUYV — read off the device with `VIDIOC_ENUM_*`, not a datasheet. Global shutter is why this is not a `usb_uvc` entry: a rolling-shutter webcam smears the frame during arm motion, corrupting exactly the wrist views a VLA conditions on. `intrinsics` is left **unset** unless the caller supplies `hfov_deg` — the board ships with an M12 mount, so optics belong to the integrator and a default pinhole model would be a fabricated number. Registered as `arducam/b0495`. (L62)

### `python/sensors/src/openral_sensors/ros_publisher.py`
_Generalised sensor → ROS 2 image publisher; non-GStreamer fallback to `RosImagePublisher`._

- `class SensorRosPublisher(*, reader, topic, rate_hz, node_name=None, frame_id=None, qos_depth=5, camera_info=None, info_topic=None, node=None, max_size=None)` — Background-thread publisher that polls any `SensorReader.read_latest()` and republishes as `sensor_msgs/Image`. Lazy-imports rclpy; raises `RuntimeError` at `start()` with install hint when ROS 2 isn't sourced. Optional `CameraInfo` companion (from an `IntrinsicsPinhole`) with RELIABLE QoS on `info_topic` — `None` derives `<topic>/camera_info` (camera_info_manager convention); the deploy sensor leg overrides it to the OpenRAL sibling layout `/openral/cameras/<name>/camera_info` so real cameras match the sim HAL (mono visual SLAM subscribes there). Optional `max_size=(w, h)` ceiling downscales the published image (aspect-preserving, BILINEAR) and rescales `CameraInfo` `k`/`p` by the same factor — every publish hands rclpy a full-resolution buffer whose Python→C conversion holds the GIL for the whole copy (~30 ms per 640x480 frame). `_publish_camera_info` always scales intrinsics from the spec's calibrated resolution to the published one, so a manifest whose declared geometry disagrees with the sensor no longer ships silently-wrong `k`/`p`. Reader lifecycle (open/close) is owned by the caller. (L79)
  - `start() -> None` — Init rclpy if needed, create publishers, spawn the pump thread. (L266)
  - `stop() -> None` — Signal the pump thread, tear down publishers + node; idempotent. (L287)
  - prop `is_started`, `n_published`, `n_stale_skipped`, `topic`, `info_topic` — Diagnostics surface consumed by the `openral_sensors_ros` lifecycle node. (L184)

### `python/sensors/src/openral_sensors/_reader_protocol.py`
_Internal Protocol shim mirroring `openral_runner.SensorReader` to avoid a sensors↔runner import cycle._

- `class SensorReaderLike(Protocol)` — Structural alias with `sensor_id`, `is_open`, `open`, `close`, `read_latest`. (L27)
