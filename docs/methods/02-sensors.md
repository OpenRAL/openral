# Layer 2 — Sensors

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/sensors/src/openral_sensors/catalog.py`
_Sensor catalog — vendor-agnostic registry of `SensorSpec` / `SensorBundle` factories._

- `class SensorSignature` — Probe-side identifier (kind + canonical value) for catalog reverse-lookup. (L51)
  fields: `kind, value`
- `class SensorCatalogEntry` — One row in the catalog. (L80)
  fields: `id, vendor, model, kind, factory, modalities, description, docs_url, signatures`
- `class SensorCatalog` — In-memory registry. (L114)
  - `register(entry, *, replace=False) -> SensorCatalogEntry` (L144)
  - `register_many(entries, *, replace=True) -> None` — Bulk-register entries; used by vendor modules at import time to populate the global `CATALOG`. Defaults `replace=True` since side-effect imports may run more than once in tests. (L157)
  - `unregister(sensor_id) -> None` (idempotent) (L167)
  - `get(sensor_id) -> SensorCatalogEntry` — Raises `KeyError` on miss. (L173)
  - `__contains__(sensor_id) -> bool` (L182)
  - `__len__() -> int` (L186)
  - `__iter__() -> object` (L190)
  - `list_ids() -> list[str]` — Sorted alphabetically. (L194)
  - `entries() -> list[SensorCatalogEntry]` — Sorted by id. (L198)
  - `filter(*, vendor=None, modality=None, kind=None) -> list[SensorCatalogEntry]` (L202)
  - `find_by_signature(signature) -> SensorCatalogEntry | None` — Reverse-lookup for `openral detect`. (L221)
  - `build(sensor_id, **kwargs) -> SensorSpec | SensorBundle` (L242)
- const `CATALOG = SensorCatalog()` — global singleton. (L254)

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
- const `_D435_RGB_INTRINSICS = IntrinsicsPinhole(...)` — Nominal D435 RGB intrinsics at 640×480, from the vendor datasheet. (L58)
- const `_D435_DEPTH_INTRINSICS = IntrinsicsPinhole(...)` — Nominal D435 depth intrinsics at 640×480. (L68)
- const `_D415_RGB_INTRINSICS = IntrinsicsPinhole(...)` — Nominal D415 RGB intrinsics at 640×480 (rolling-shutter IR-stereo, 65°×40°). (L79)
- const `_D415_DEPTH_INTRINSICS = IntrinsicsPinhole(...)` — Nominal D415 depth intrinsics at 640×480. (L89)
- `realsense_d435_bundle(name='realsense', parent_frame='base_link', serial_no='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0) -> SensorBundle` (L104)
- `realsense_d435i_bundle(...) -> SensorBundle` — D435 + Bosch BMI085 IMU; delegates to `realsense_d435_bundle`. (L424)
- `realsense_d415_bundle(...) -> SensorBundle` — rolling-shutter IR stereo, 65°×40°, no IMU. (L459)
- `bundle_to_node_params(bundle, serial_no='') -> NodeParams` — Map to `realsense2_camera` node params. (L205)
- `generate_launch_py(bundle, serial_no='') -> str` — Auto-generated ROS 2 launch file. (L279)
- `calibrate_camera_cmd(sensor, chessboard_cols=8, chessboard_rows=6, square_size_m=0.025) -> list[str]` — Build `ros2 run camera_calibration cameracalibrator` argv. `sensor.ros2_topic` is the driver's image topic (ADR-0108); `camera_info` is its sibling in the same namespace. (L348)

#### `python/sensors/src/openral_sensors/luxonis.py`
- const `_OAK_D_PRO_RGB_INTRINSICS = IntrinsicsPinhole(...)` — Nominal RGB (IMX378) intrinsics at 1920×1080 (95° HFoV). (L48)
- const `_OAK_D_PRO_DEPTH_INTRINSICS = IntrinsicsPinhole(...)` — Nominal stereo-depth (OV9282) intrinsics at 1280×800, 71.86° HFoV, 7.5 cm baseline. (L60)
- `oak_d_pro_bundle(name='oak', parent_frame='base_link', mxid='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0, rgb_width=1920, rgb_height=1080, depth_width=1280, depth_height=800) -> SensorBundle` — Luxonis OAK-D Pro RGB + global-shutter stereo depth (0.20–19 m, 71.86°×56°) + BNO086 IMU bundle. Registered as `luxonis/oak_d_pro`, the recommended overhead RGB-D for the `so101_box` scene. (L75)
- `_scale_intrinsics(base, width, height) -> IntrinsicsPinhole` — Delegates to `openral_core.scale_intrinsics_to` so a caller can pick a non-default stream resolution and still get consistent intrinsics. (L189)

#### `python/sensors/src/openral_sensors/stereolabs.py`
- const `_ZED_MINI_EYE_INTRINSICS = IntrinsicsPinhole(...)` — Nominal per-eye intrinsics at the HD720 default (1280×720). (L51)
- const `_ZED_MINI_BASELINE_M = 0.063` — Stereo baseline, metres. (L62)
- const `_ZED_MINI_DEPTH_MIN_M = 0.10` — Minimum reported depth range, metres. (L63)
- const `_ZED_MINI_DEPTH_MAX_M = 15.0` — Maximum reported depth range, metres. (L64)
- const `_ZED_MINI_FOV_H_DEG = 90.0` — Horizontal FOV, degrees. (L65)
- const `_ZED_MINI_FOV_V_DEG = 60.0` — Vertical FOV, degrees. (L66)
- `zed_mini_bundle(name='zed', parent_frame='base_link', serial='', rgb_rate_hz=30.0, depth_rate_hz=30.0, imu_rate_hz=400.0, width=1280, height=720) -> SensorBundle` — StereoLabs ZED Mini: rectified stereo RGB, host-computed depth (0.10–15 m, 90°×60°), integrated IMU. Passive stereo — degrades on untextured surfaces but never conflicts with an active depth camera on the same workspace; ships as one USB UVC node needing the ZED SDK for depth. Registered as `stereolabs/zed_mini`. (L69)

#### `python/sensors/src/openral_sensors/arducam.py`
- const `_B0495_NATIVE_WIDTH = 1920` — Native stream width, read off the device with `VIDIOC_ENUM_FRAMESIZES`. (L47)
- const `_B0495_NATIVE_HEIGHT = 1200` — Native stream height. (L48)
- const `_B0495_MAX_RATE_HZ = 50.0` — Max frame rate at native resolution. (L49)
- `arducam_b0495_spec(name='arducam', parent_frame='base_link', rate_hz=30.0, width=1920, height=1200, hfov_deg=None, serial='') -> SensorSpec` — Arducam B0495: global-shutter USB3 UVC colour (1920×1200@50fps), avoiding the motion smear a rolling-shutter `usb_uvc` camera gives wrist views. `intrinsics` stays unset unless `hfov_deg` is given — the M12 mount has no default optics. Registered as `arducam/b0495`. (L52)

### `python/sensors/src/openral_sensors/ros_publisher.py`
_Generalised sensor → ROS 2 image publisher; non-GStreamer fallback to `RosImagePublisher`._

- const `_DEFAULT_QOS_DEPTH: Final[int] = 5` — Default image-publisher QoS depth; matches gscam2's `sensor_data`-style default. (L51)
- const `_THREAD_JOIN_TIMEOUT_S: Final[float] = 2.0` — Join timeout for the background pump thread on `stop()`. (L56)
- const `_OPENRAL_TO_ROS_ENCODING: Final[dict[FrameEncoding, str]] = {...}` — Maps `FrameEncoding` to the `sensor_msgs/Image.encoding` string; CPU-side encodings only. (L61)
- `class SensorRosPublisher(*, reader, topic, rate_hz, node_name=None, frame_id=None, qos_depth=5, camera_info=None, info_topic=None, node=None, max_size=None)` — Background-thread publisher that polls a `SensorReader` and republishes frames as `sensor_msgs/Image`, with an optional `CameraInfo` companion and an optional `max_size` downscale that rescales intrinsics to match. Lazy-imports rclpy, raises `RuntimeError` at `start()` if ROS 2 isn't sourced; reader lifecycle (open/close) is owned by the caller. (L69)
  - `prepare() -> None` — Create ROS resources without starting the pump thread; multi-camera callers prepare every publisher first to avoid concurrent rclpy setup. (L195)
  - `start() -> None` — Init rclpy if needed, create publishers, spawn the pump thread. (L253)
  - `stop() -> None` — Signal the pump thread, tear down publishers + node; idempotent. (L274)
  - `is_started -> bool` [@property] — `True` between `start` and `stop`. (L171)
  - `n_published -> int` [@property] — Number of image messages successfully published since `start`. (L176)
  - `n_stale_skipped -> int` [@property] — Number of ticks the reader had no fresh frame and publish was skipped. (L181)
  - `topic -> str` [@property] — The configured image topic (read-only). (L186)
  - `info_topic -> str` [@property] — The configured `CameraInfo` companion topic (read-only). (L191)
- `camera_info_topic_for(image_topic: str) -> str` — The `CameraInfo` topic beside an image topic. OpenRAL's own `/openral/cameras/<name>/image` and `…/depth/image` resolve through `openral_core.camera_topic(name, CAMERA_INFO | DEPTH_CAMERA_INFO)` (the one spelling of that layout); driver topics follow image_pipeline's sibling rule (`.../image_raw` → `.../camera_info`), anything else → `<topic>/camera_info`. Used by the GStreamer ROS tee and camera calibration. (L503)
- `build_camera_info_msg(spec: IntrinsicsPinhole, *, width, height, stamp, frame_id) -> CameraInfo` — The one `sensor_msgs/CameraInfo` builder both real-camera ROS paths use (`SensorRosPublisher` and the GStreamer `RosImagePublisher`): intrinsics rescaled to `width x height` via `openral_core.scale_intrinsics_to` (degenerate spec verbatim), identity `r`, monocular `p` (`Tx = 0` always: no manifest field declares a stereo baseline, so a right stereo camera's consumers take the baseline from TF). `sensor_msgs` lazy-imported. (L536)

### `python/sensors/src/openral_sensors/_reader_protocol.py`
_Internal Protocol shim mirroring `openral_runner.SensorReader` to avoid a sensors↔runner import cycle._

- `class SensorReaderLike(Protocol)` — Structural alias with `sensor_id`, `is_open`, `open`, `close`, `read_latest`. (L27)
  - `open() -> None` — Acquire the capture device and start any background workers. (L39)
  - `close() -> None` — Release the capture device. Idempotent. (L42)
  - `read_latest(max_age_ms=None) -> SensorFrame` — Return the most recent buffered `SensorFrame`. Non-blocking. (L45)
