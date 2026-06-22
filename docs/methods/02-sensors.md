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
  - `keys() -> list[str]` — Insertion order. (L202)
  - `list_ids() -> list[str]` — Sorted alphabetically. (L206)
  - `entries() -> list[SensorCatalogEntry]` — Sorted by id. (L210)
  - `filter(*, vendor=None, modality=None, kind=None) -> list[SensorCatalogEntry]` (L214)
  - `find_by_signature(signature) -> SensorCatalogEntry | None` — Reverse-lookup for `openral detect`. (L233)
  - `build(sensor_id, **kwargs) -> SensorSpec | SensorBundle` (L254)
- const `CATALOG = SensorCatalog()` — global singleton. (L266)

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

### `python/sensors/src/openral_sensors/ros_publisher.py`
_ADR-0019 PR2 — generalised sensor → ROS 2 image publisher; non-GStreamer fallback to `RosImagePublisher`._

- `class SensorRosPublisher(*, reader, topic, rate_hz, node_name=None, frame_id=None, qos_depth=5, camera_info=None)` — Background-thread publisher that polls any `SensorReader.read_latest()` and republishes as `sensor_msgs/Image`. Lazy-imports rclpy; raises `RuntimeError` at `start()` with install hint when ROS 2 isn't sourced. Optional `CameraInfo` companion topic at `<topic>/camera_info` with RELIABLE QoS. Reader lifecycle (open/close) is owned by the caller. (L79)
  - `start() -> None` — Init rclpy if needed, create publishers, spawn the pump thread. (L183)
  - `stop() -> None` — Signal the pump thread, tear down publishers + node; idempotent. (L252)
  - prop `is_started`, `n_published`, `n_stale_skipped`, `topic`, `info_topic` — Diagnostics surface consumed by the `openral_sensors_ros` lifecycle node. (L159)

### `python/sensors/src/openral_sensors/_reader_protocol.py`
_Internal Protocol shim mirroring `openral_runner.SensorReader` to avoid a sensors↔runner import cycle._

- `class SensorReaderLike(Protocol)` — Structural alias with `sensor_id`, `is_open`, `open`, `close`, `read_latest`. (L27)
