# Inference Runner (executor of S1)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

The hardware-side counterpart to `openral_sim` — closes
`WorldState → Skill.step → SafetyClient.check → HAL.send_action` at the runner
tick rate. `DeployScene` is the deploy/workcell YAML used by the ROS deploy
graph; `openral_runner` remains the library executor behind runtime nodes.

### `python/runner/src/openral_runner/clock.py`
_High-precision cadence helpers for the inference runner._

- `precise_sleep(duration_s: float) -> None` — Hybrid sleep: `time.sleep` for the bulk, busy-wait on `perf_counter` for the final ~1 ms; non-positive `duration_s` is a no-op. (L29)
- `sleep_until(deadline_perf_counter_s: float) -> None` — Convenience wrapper taking an absolute `time.perf_counter` deadline. Used by `InferenceRunnerBase.run` to enforce cadence. (L60)
- module constant `_BUSY_LOOP_THRESHOLD_S = 1e-3` — Busy-wait threshold used by `precise_sleep`. (L26)

### `python/runner/src/openral_runner/protocol.py`
_Inference runner Protocol. The structural contract every runner shape satisfies._

- `class InferenceRunner(Protocol)` — `@runtime_checkable` Protocol. (L24)
  - attr `rate_hz: float` — foreground tick rate.
  - `activate() -> None` — open sensors / HAL / executor. (L38)
  - `tick() -> TickResult` — run one tick (record), no cadence enforcement here. (L42)
  - `run(max_ticks: int | None = None) -> RunResult` — rate-limited loop returning the aggregate. (L52)
  - `deactivate() -> None` — release resources opened by `activate`; idempotent. (L62)

### `python/runner/src/openral_runner/world_cloud_bridge.py`
_rclpy → OTLP bridge rendering the octomap occupied-voxel cloud (`/octomap_point_cloud_centers`) as a robot-frame oblique "chase-view" PNG for the dashboard `world.pointcloud` card. Pure render core is rclpy-free (tested without ROS)._

- module constant `WORLD_CLOUD_TOPIC_DEFAULT = "/octomap_point_cloud_centers"` — default occupied-voxel-centers PointCloud2 topic (octomap_server). (L48)
- module constant `_DEFAULT_PUBLISH_INTERVAL_S = 1.0` (L53) — throttle so a busy octomap run doesn't flood the OTLP pipeline.
- module constant `_DROP_WARN_BURST = 3` (L57) — oversize-cloud warnings emitted verbosely before dropping to a rate-limited cadence.
- module constant `_DROP_WARN_RATE = 100` (L58) — the 1-in-N rate after `_DROP_WARN_BURST` is exceeded.
- module constant `_CAM_BACK_M = 2.2` (L63) — oblique chase-camera offset behind the robot in `base_link`.
- module constant `_CAM_UP_M = 1.6` (L64) — oblique chase-camera offset above the robot in `base_link`.
- module constant `_CAM_PITCH_DOWN_RAD = 0.45` (L65) — oblique chase-camera downward tilt.
- module constant `_FOCAL_PX = 320.0` (L66) — chase-camera focal length in pixels, tuned so a ~2 m local box fills a 480x360 frame.
- module constant `_MIN_CAM_DEPTH_M = 1e-3` (L68) — camera-forward depth below which a point is treated as behind the lens.
- module constant `_BG_RGB = (16, 20, 28)` (L70) — chase-view canvas background colour.
- module constant `_ORIGIN_RGB = (240, 240, 255)` (L71) — chase-view origin-marker colour.
- `crop_points_to_box(points, *, xy_m, z_min, z_max) -> NDArray[float32]` — keep `(N,3)` points inside the local box around base_link. (L74)
- `distance_to_rgb(dist_m, *, range_max_m) -> tuple[int,int,int]` — near=warm→far=cool color ramp. (L90)
- `encode_world_cloud_png(points_base, *, range_max_m=4.0, image_w=480, image_h=360, xy_m=2.0, z_min=-0.2, z_max=2.0) -> str` — crop→oblique-pinhole project→rasterize→base64 PNG. Pure; PIL-only. (L137)
- `world_cloud_span_attributes(*, points_base, frame_id, source_node, range_max_m, xy_m, z_min, z_max) -> dict[str,Any]` — assemble the `openral.world_cloud.*` span attributes. (L210)
- `class WorldCloudBridge` — Subscribes the voxel cloud, TF2-transforms into `base_frame` (default `base_link`; pass the manifest's own `base_frame` for a fixed-base arm, or lookups silently drop every cloud), throttles to 1 Hz, emits a `world.pointcloud` span. `latched=True` matches octomap's TRANSIENT_LOCAL topic, `False` matches nvblox's VOLATILE ESDF cloud — mismatched durability receives nothing; an unreadable frame is warned and skipped rather than crashing the executor. (L256)
  - `destroy() -> None` (L333) — Releases the ROS subscription; safe to call multiple times.

### `python/runner/src/openral_runner/dataset_recorder_bridge.py`
_Bus-attached LeRobot/rosbag recorder for the deploy graph (mirrors `WorldCloudBridge`)._

- `decode_inline_frame(frame: SensorFrame) -> np.ndarray | None` (L85) — Decode one aggregator `SensorFrame` with inline `data` into an `HxWxC` array whose **dtype comes from `frame.encoding`** (`DEPTH16` → uint16 millimetres; `BGR8` / `RGB8` / `MONO8` / `RAW` → uint8); `None` for topic/handle delivery or a compressed / device-handle encoding (JPEG, PNG, CUDA_*) rather than a mis-reshape. Shared by the recorder's `_decode_images` — which additionally keeps only `(H, W, 3) uint8` frames, the contract of `DatasetRecorder.record_frame`, so a depth or mono frame in the same world state is skipped rather than rejected per frame — and the runner's `_decode_image_frames`, where the depth frame rides along under its own sensor name as `uint16 (H, W, 1)`. Reading everything as uint8 aborted the first real-hardware OpenArm dispatch (qorin1, 2026-09-22): the ZED depth sensor's `16UC1` 1280x720 frame is two bytes per pixel and `reshape(720, 1280, 1)` raised `ValueError: cannot reshape array of size 1843200`, killing the observation the policy's RGB slots were in.

- module constant `_PHASE_START = 0` (L61) — `Episode.phase` enum value; mirrors `packages/msgs/msg/Episode.msg`.
- module constant `_PHASE_END = 1` (L62) — `Episode.phase` enum value; mirrors `packages/msgs/msg/Episode.msg`.
- module constant `ACTION_TOPIC_DEFAULT = "/openral/candidate_action"` (L64) — default `ActionChunk` topic.
- module constant `EPISODE_TOPIC_DEFAULT = "/openral/episode"` (L65) — default `Episode` marker topic.
- `class DatasetRecorderBridge(node, *, robot, aggregator, recorder, output_path=None, action_topic="/openral/candidate_action", episode_topic="/openral/episode")` — Subscribes `Episode` (drives `recorder.episode_start/end`) and `ActionChunk`, joins each tick's action with the `WorldStateAggregator` snapshot, and writes frames via `Rosbag2Sink`. Logs `dataset_recorder.nothing_recorded` at `destroy()` if no episode marker ever fired, so an empty recording is never silent. (L115)
  - `destroy() -> None` (L212) — Flushes the pending tick, closes any open episode (marking it a failure), finalizes the recorder, releases the subscriptions; idempotent.

### `python/runner/src/openral_runner/sensor_reader.py`
_``SensorReader`` Protocol — seam between per-sensor capture backends and the inference runner._

- `class SensorReader(Protocol)` — `@runtime_checkable` Protocol; concrete backends live under `openral_runner.backends`. (L29)
  - attr `sensor_id: str` — matches `SensorReaderConfig.sensor_id`.
  - attr `is_open: bool` — True between `open()` and `close()`.
  - `open() -> None` — Acquire device, start background workers. Idempotent. (L55)
  - `close() -> None` — Release device, join workers. Idempotent. (L64)
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` — Non-blocking peek at the most recent buffered frame; raises `ROSPerceptionStale` if no frame yet or freshest exceeds budget. (L71)

### `python/runner/src/openral_runner/backends/opencv_thread.py`
_``OpenCVThreadSensorReader`` — default backend. Mirrors lerobot's per-camera-thread pattern._

- module constant `_COLOR_NDIM = 3` — Number of dims for an OpenCV colour frame (`(H, W, 3)`); mono is `(H, W)`. Used to derive `SensorFrame.channels`. (L39)
- module constant `_CROP_LEN = 4` (L41) — A crop is `(x, y, width, height)`.
- `_validated_crop(sensor_id, crop) -> tuple[int, int, int, int] | None` — Normalises a crop (tuple or YAML list) and rejects an invalid length, origin, or extent at construction, before any I/O.
- `class OpenCVThreadSensorReader` — Per-camera background-thread reader on top of `cv2.VideoCapture`. Imports `cv2` lazily inside `open()` (the `opencv` optional-extra). (L84)
  - `__init__(*, sensor_id, device, fps=30, width=None, height=None, encoding=BGR8, crop=None, default_max_age_ms=100)` — Stash config; rejects non-positive `fps` / `default_max_age_ms` and a malformed `crop`.
  - `crop` handles side-by-side stereo (e.g. a ZED Mini's doubled-width UVC frame): without it the extra width is silently out-of-distribution, not an error, since the shape is still a valid image. Applied in the capture thread at zero hot-path cost (the copy happens later, at `tobytes()`).
  - `_apply_crop(frame) -> frame | None` — Returns the cropped sub-rectangle, or `None` (frame dropped, logged) if the device changed mode mid-stream; never raises, so the reader stays recoverable instead of dying silently.
  - `open() -> None` (L161) — Opens the capture, validates any `crop` against the mode the device actually negotiated, and spawns the reader thread; idempotent. A crop that doesn't fit raises here, so a bad deploy refuses to start rather than run blind.
  - `close() -> None` — Stop event, join thread (2 s timeout), release capture. Idempotent. (L218)
  - `__enter__() / __exit__()` — Context-manager sugar; calls `open` / `close`. (L236)
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` — Lock-protected snapshot with inlined raw bytes; raises `ROSPerceptionStale` if no frame yet or stale, `RuntimeError` if closed. (L247)
  - `_read_loop()` — Background daemon: `cv2.VideoCapture.read` → `_latest_frame + _latest_stamp_*_ns` under lock; sleeps `1/fps` on read failure / EOF. (L330)

### `python/runner/src/openral_runner/backends/ros2_image.py`
_``Ros2ImageSensorReader`` — backend for a stream a device only publishes over ROS rather than emits over USB directly (e.g. ZED SDK depth, RealSense aligned depth). Subscribes rather than opening a device._

- module constant `_DIRECT_ENCODINGS` (L57) — Maps supported `sensor_msgs/Image` encodings to `(FrameEncoding, dtype, channels)`; anything unlisted is refused by name, not misread as pixels.
- module constant `_ALPHA_ENCODINGS: Final[dict[str, str]] = {"bgra8": "bgr8", "rgba8": "rgb8"}` (L76) — alpha-channel encodings accepted by dropping the alpha byte to their `_DIRECT_ENCODINGS` base.
- module constant `_FLOAT_DEPTH_ENCODINGS = {"32FC1"}` (L81) — float metre depth (what the ZED SDK publishes), converted on the way in.
- module constant `_DEPTH16_MAX_MM = 65535` (L85) — the uint16-millimetre ceiling of `FrameEncoding.DEPTH16`.
- `_depth32f_to_depth16(metres) -> NDArray[uint16]` (L88) — Converts float32 metres to uint16 millimetres, the format every downstream consumer expects. Non-finite or out-of-range samples become 0 (ROS's "no reading") rather than wrapping to a confident, wrong, near distance.
- `class Ros2ImageSensorReader` (L118) — Subscribes to a driver topic, keeps the newest message in a one-slot buffer, serves it through the same non-blocking `read_latest` staleness contract as every other backend. Owns no device.
  - `__init__(*, sensor_id, topic, default_max_age_ms=100, reliability="best_effort", qos_depth=5, node=None)` (L146) — Config only; rejects empty `topic` or unknown `reliability`. Defaults to `BEST_EFFORT`/`VOLATILE`/`KEEP_LAST=5` because a `RELIABLE` subscriber receives nothing from a `BEST_EFFORT` publisher — a mismatch that looks exactly like a dead camera.
  - `open() -> None` (L189) — Creates the subscription and, if it owns the node, spins an executor thread. Idempotent; sets `is_open` last so a failure mid-setup is still cleaned up.
  - `close() -> None` (L252) — Destroys the subscription; calls `rclpy.shutdown()` only if this reader initialised it. Idempotent, guarding per-resource rather than on `is_open`.
  - `read_latest(max_age_ms=None) -> SensorFrame` (L301) — Lock-protected snapshot; `ROSPerceptionStale` on no-frame-yet or staleness, `RuntimeError` on a closed reader. Frames carry inlined `data`, not a `topic` reference.
  - `_on_image(msg) -> None` (L343) — Subscription callback; conversion failures are counted and logged, never raised, since an exception here would kill the spin loop and silently stop the camera.
- `_rows(raw, dtype, msg, height, width, channels) -> NDArray` (L445) — Unpacks an `Image` payload honouring `msg.step` (row stride), needed when a publisher hands out pitch-aligned buffers (e.g. Isaac/NITROS) or a cropped ROI. An implausible `step` falls back to the packed stride, so a wrong value fails loud rather than yielding a skewed image.
- `_byte_order(msg) -> str` (L486) — Honours `Image.is_bigendian`; a 16-bit depth image from a big-endian publisher read little-endian is byte-swapped garbage.

### `python/runner/src/openral_runner/backends/galaxea_a1_camera_bridge.py`
_Real-deploy reader for the public A1 Runtime paired-frame bridge. It never
opens a camera device or imports the Runtime checkout; the A1 camera monitor
stays the only RealSense owner._

- module constant `_PROTOCOL_VERSION = 2` — wire version stamped on every request to the camera-bridge session. (L22)
- module constant `_SOCKET_NAME = "a1-camera-bridge.sock"` — the shared session's Unix-socket name under `runtime_socket_path`. (L23)
- module constant `_MAX_RESPONSE_BYTES = 128 * 1024 * 1024` — response-size ceiling passed to `RuntimeLocalClient`. (L24)
- module constant `_SHA256_HEX_LENGTH = 64` — expected length of the session's `contract_digest`. (L25)
- module constant `_COLOR_NDIM = 3` — expected rank of a `*_color_shape` tuple. (L26)
- module constant `_COLOR_CHANNELS = 3` — expected channel count of a `*_color_shape` tuple. (L27)
- `class GalaxeaA1CameraBridgeReader` (L187) — Reads either the `front` or `wrist`
  member of a native versioned Unix-socket session and returns an inline RGB8
  `SensorFrame`. The client discovers and pins the Runtime contract digest and
  camera shapes before reading; stale pairs, wrong shapes, and drift fail
  explicitly.
  - `open() -> None` — Acquire the shared Runtime camera connection; idempotent. (L207)
  - `close() -> None` — Release this view and close the connection after the last owner; idempotent. (L214)
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` — Return the latest source-timestamped RGB frame. (L223)

### `python/runner/src/openral_runner/backends/galaxea_a1_ipc.py`
_Shared native transport for public A1 Runtime local services._

- module constant `_PACKET_LENGTH = struct.Struct("!I")` — the length-prefix codec shared by send/receive. (L16)
- `runtime_socket_path(name) -> Path` (L19) — Resolves a private per-user Runtime
  endpoint from `A1_PROCESS_STATE_ROOT` or the standard runtime directory.
- `class RuntimeLocalClient` (L32) — Bounded, synchronous length-prefixed MessagePack
  client with typed connect-time and request-time failures.
  - `connect(*, timeout_s: float) -> None` — Connect to the Runtime-owned Unix socket. (L50)
  - `call(request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]` — Perform one bounded request and require an explicit success reply. (L68)
  - `close() -> None` — Close the local connection idempotently. (L99)
- `encode_array(value) -> dict[str, Any]` (L106) — Encode one contiguous ndarray with an explicit shape and dtype.
- `decode_array(value, *, shape, dtype, label) -> NDArray` (L116) — Decode an exact ndarray payload and reject protocol drift.

### `python/runner/src/openral_runner/backends/__init__.py`
_Per-backend `SensorReader` implementations. Default `OpenCVThreadSensorReader` is always available; `GStreamerSensorReader` gates on PyGObject and `Ros2ImageSensorReader` on a ROS 2 install (both lazy-imported at `open()`)._

- `OpenCVThreadSensorReader` — Lazy-exported via PEP 562 so importing the gstreamer backend doesn't eagerly pull in `cv2`, whose glib init can segfault a subsequent `rclpy.Node()` in some ROS-enabled processes.
- `__getattr__(name) -> Any` — PEP 562 attribute hook; resolves `OpenCVThreadSensorReader` on first access via `importlib.import_module`. (L28)

### `python/runner/src/openral_runner/backends/gstreamer/pipeline.py`
_GStreamer pipeline-string builder + platform detection. Pure-Python — does **not** import `gi` at module load._

- `TEE_NAME: Final[str]` (L56) — `"openral_cam_tee"`. Name of the per-camera `tee` — the perception-bus attach point `TeeManager` uses to add pads for reasoner-activated consumers at runtime.
- `LEAKY_BRANCH_QUEUE: Final[str]` (L63) — `"queue leaky=downstream max-size-buffers=2"`. The single definition of the per-branch isolation policy, shared by the static builder and the runtime `TeeManager`.
- `leaky_branch(elements, *, tee_name=TEE_NAME) -> str` (L66) — Returns one `tee` branch `<tee>. ! <leaky queue> ! <elements>`. The shared branch-construction primitive so the static builder and the dynamic `TeeManager` build branches identically.
- module constant `_DEFAULT_APPSINK_NAME: Final[str] = "bh_sink"` (L48) — default name attached to the trailing appsink so the reader can look it up via `Gst.Bin.get_by_name`.
- module constant `_TEGRA_RELEASE_PATH: Final[Path] = Path("/etc/nv_tegra_release")` (L92) — path read to identify a Tegra host (Jetson / Spark).
- module constant `_GST_INSPECT_TIMEOUT_S: Final[float] = 5.0` (L97) — timeout for the `gst-inspect-1.0` probe.
- module constant `_NVVIDEOCONVERT: Final[str] = "nvvideoconvert"` (L101) — the DeepStream NVMM colour-convert element.
- module constant `_NVVIDCONV: Final[str] = "nvvidconv"` (L102) — the Tegra / L4T multimedia-stack NVMM colour-convert element.
- module constant `_VIDEOCONVERT: Final[str] = "videoconvert"` (L103) — the stock system-memory / CPU colour-convert element.
- module constant `_NVVIDCONV_BGR_BRIDGE_FORMAT: Final[str] = "BGRx"` (L107) — the closest packed format `nvvidconv` advertises; bridged to `BGR` via `bgr_convert_chain`.
- `class PipelineSpec(BaseModel)` (L167) — Validated description of a GStreamer ingest pipeline; `jpeg` (MJPG UVC, USB-only) is exclusive with `encoded`, and `event_appsink_name` is validated as a legal GStreamer element name.
- `class Platform(str, Enum)` (L110) — GStreamer platform tier: `TEGRA` / `NVIDIA_DEEPSTREAM` / `NVIDIA_DESKTOP` / `CPU_ONLY`. `NVIDIA_DEEPSTREAM` is the x86 `ds-on` image, decoding and converting NVMM-native on GPU.
- `class Source(str, Enum)` (L143) — `USB | CSI | RTSP | FILE | TESTSRC`.
- `detect_platform() -> Platform` (L272) — Cached; detects the platform by reading `/etc/nv_tegra_release` then probing `gst-inspect-1.0` for DeepStream/nvcodec elements.
- `inspect_element_present(element_name) -> bool` (L307) — Generic `gst-inspect-1.0 --exists` probe with timeout.
- `nvmm_convert_element() -> str | None` (L337) — Probes for the host's NVMM colour-convert element: `nvvideoconvert` (DeepStream/x86) preferred, else `nvvidconv` (Tegra/L4T), else `None`.
- `bgr_convert_chain(convert) -> str` (L357) — Bridges a converter element to system-memory `BGR`: passthrough for `nvvideoconvert`/`videoconvert`, but `nvvidconv` needs a `BGRx`→`videoconvert` bridge since pinning `format=BGR` on it fails to link on a plain-L4T Jetson.
- `ensure_appsink_name(pipeline, name) -> str` (L401) — Rewrites a trailing `appsink` to carry `name=<name>`.
- `build_pipeline_string(spec, platform=None) -> str` (L453) — Builds the pipeline string, adding a `tee` leg per enabled `ros`/`event` branch via `leaky_branch` so a stalled observability or detector branch never backpressures the policy leg.
- `_build_event_tee_branch(spec, platform) -> str` (L726) — Returns the event leg of the `tee`: lifts NVMM to system memory, pins `format=BGR`, rate-caps via `videorate` to `event_rate_hz`, terminates in `appsink name=event_sink`.
- `_build_ros_tee_branch(spec, platform) -> str` (L710) — Returns the observability leg (system memory BGR `appsink name=ros_sink`).
- `_platform_convert_element(platform) -> str` (L613) — The bare colour-convert element for a platform: `nvvidconv` (Tegra) / `nvvideoconvert` (DeepStream) / `videoconvert` (else). Shared by the policy leg and the tee legs.
- `_use_nvmm(spec, platform) -> bool` (L637) — Single definition of "does the policy leg negotiate `memory:NVMM`", shared by `_build_convert` and `_build_caps` so converter and caps can never disagree.
- `_build_convert(spec, platform) -> str` (L648) — The policy leg's conversion stage: the bare element on the NVMM path (`NV12`/`RGBA` are on its own src template), `bgr_convert_chain(...)` on the system-memory `BGR` path.
- `_lift_convert(platform) -> str` (L751) — The chain a tee leg uses to lift NVMM → system-memory BGR: `_platform_convert_element` passed through `bgr_convert_chain`, so Tegra legs bridge via `BGRx` while DeepStream legs stay direct-to-`BGR`.

### `python/runner/src/openral_runner/backends/gstreamer/reader.py`
_GStreamer-backed `SensorReader` (CPU appsink path + NVMM zero-copy path). Mirrors the latest-only contract of `OpenCVThreadSensorReader`. `Gst.init` runs immediately after the `gi` import, before anything else, to avoid a Fast-DDS/GStreamer thread-init SIGSEGV._

- module constant `_BUS_POLL_TIMEOUT_NS: Final[int] = 100_000_000` (L69) — bus poll timeout (100 ms) when listening for ERROR / EOS.
- module constant `_DEFAULT_MAX_AGE_MS: Final[int] = 100` (L72) — default staleness budget; matches the OpenCV reader default.
- module constant `_GST_FORMAT_TO_ENCODING: Final[dict[str, FrameEncoding]]` (L78) — maps GStreamer caps `format=...` (`BGR`/`RGB`/`GRAY8`) to `FrameEncoding` for the CPU path; NV12 is deliberately absent (handled by the NVMM path instead).
- `class GStreamerSensorReader` (L85) — `SensorReader` backed by a GStreamer pipeline; construct from an explicit `pipeline=` string or a generated `spec=` (`PipelineSpec`). CPU path delivers `SensorFrame(data=bytes)`; NVMM/CUDA zero-copy populates `handle` + `encoding` ∈ `{CUDA_NV12, CUDA_RGBA}`.
  - `open() -> None` (L200) — Initialise GStreamer, parse the pipeline, start the ROS tee (if enabled) and the bus-drain thread, transition to PLAYING. Idempotent.
  - `close() -> None` (L301) — Stop the ROS publisher (if any), tear down the pipeline, join the bus thread, release the latched frame/handle. Idempotent.
  - `__enter__() / __exit__()` (L333) — Context-manager sugar; calls `open` / `close`.
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` (L349) — Non-blocking snapshot of the latched frame; raises `ROSRuntimeError` on a bus-reported error, `ROSPerceptionStale` on no-frame-yet or staleness, `RuntimeError` on a closed reader.
  - `_start_ros_publisher()` (L266) — Look up the `ros_sink` appsink and start a `RosImagePublisher`; tears the pipeline back down with an actionable `ROSConfigError` if `rclpy` is unavailable.
  - `_on_new_sample(appsink) -> int` (L430) — Streaming-thread callback; branches on `memory:NVMM` caps features to the zero-copy or CPU handler.
  - `_handle_cpu_buffer(buffer, structure) -> int` (L459) — Map → copy → latch `data`; unsupported format latches a bus error.
  - `_handle_nvmm_buffer(buffer, structure) -> int` (L501) — Map → wrap as an `NvBufSurfaceHandle` (lazy `openral_pro_trt.nvbufsurface` import) → DtoD-mirror into a reader-owned `StableSurfaceMirror` → latch `handle`; an absent/unloadable NVMM backend latches a bus error rather than silently falling back.
  - `_bus_loop()` (L608) — Background thread draining the GStreamer bus for ERROR / EOS.
  - `_teardown_pipeline()` (L640) — Drop the pipeline and join the bus thread; shared by `close()` and `open()`'s rollback path.
  - `_wrap_in_pipeline(element) -> Gst.Pipeline` (L652) [@staticmethod] — Wraps a bare `Gst.Element` from `Gst.parse_launch` in a `Pipeline` bin (single-element strings only).
- module constant `_GST_INIT_LOCK` (L665) — one-shot-init guard lock.
- module constant `_GST_INITIALISED` (L666) — one-shot-init guard flag.
- `_ensure_gst_initialised() -> None` (L669) — Calls `Gst.init` exactly once per process, thread-safely.

### `python/runner/src/openral_runner/backends/gstreamer/ros_tee.py`
_ROS 2 image-publisher tee for `GStreamerSensorReader`. Republishes the `ros_sink` appsink branch as `sensor_msgs/Image` on a configurable topic, independently rate-limited from the inference loop. `rclpy` is lazy-imported inside `start()` so the module is import-safe without a sourced ROS env._

- module constant `_DEFAULT_QOS_DEPTH: Final[int] = 5` (L47) — default QoS depth for the image publisher (mirrors gscam2's shallow, `BEST_EFFORT`-friendly default).
- `class RosImagePublisher` (L50) — `__init__(*, sensor_id, appsink, topic, rate_hz=None, node_name=None, qos_depth=_DEFAULT_QOS_DEPTH)` — validates `topic` is absolute and `rate_hz` is positive or `None`; no ROS I/O until `start`.
  - `is_started` [@property] (L105) — `True` between `start` and `stop`.
  - `start() -> None` (L109) — Initialise rclpy (if needed), create the `sensor_msgs/Image` publisher (`BEST_EFFORT`+`VOLATILE`+`KEEP_LAST`), hook the appsink; raises `RuntimeError` if `rclpy` is unavailable.
  - `stop() -> None` (L158) — Disconnect the signal, destroy the publisher, shut down rclpy if this instance initialised it. Idempotent.
  - `_on_new_sample(appsink) -> int` (L183) — Rate-gate → map → build `sensor_msgs/Image` → publish.
  - `_claim_rate_slot() -> bool` (L219) — Monotonic-clock token gate enforcing `rate_hz`.
  - `_extract_image_payload(appsink, gst) -> tuple[bytes, int, int, str] | None` (L234) — Pull the latest sample; `None` on malformed sample / unsupported format / map failure.
- `_gst_format_to_ros_encoding(gst_format) -> str | None` (L276) — Maps a GStreamer caps `format` (`BGR`/`RGB`/`GRAY8`) to a ROS `Image.encoding` (`bgr8`/`rgb8`/`mono8`).

### `python/runner/src/openral_runner/backends/gstreamer/perception_tee.py`
_Perception event tee for `GStreamerSensorReader`. Pulls frames from the event leg's `appsink`, runs `EventDetector`s, publishes `openral_msgs/PromptStamped` on `/openral/perception/<kind>`. `rclpy` lazy-imported in `start()` so the module stays import-safe on hosts without a sourced ROS env._

- module constant `TOPIC_PREFIX: Final[str] = "/openral/perception"` (L56) — Fixed value; full topic is `f"{TOPIC_PREFIX}/{detector.kind}"`.
- module constant `_DEFAULT_QOS_DEPTH: Final[int] = 10` (L61) — default QoS depth for the per-kind `PromptStamped` publisher (`BEST_EFFORT`+`VOLATILE`+`KEEP_LAST`).
- module constant `_DEFAULT_RATE_HZ: Final[float] = 5.0` (L66) — default per-detector token-bucket rate cap.
- `class EventDetector(Protocol)` (L69) — `kind: str`.
  - `detect(frame_bgr, width, height, sensor_id) -> PerceptionEventMetadata | None` (L88) — Run one detection pass; `None` means no event this frame.
  - `summarise(metadata) -> str` (L97) — Human-readable `PromptStamped.text` for `metadata`.
- `class MotionDetector` (L101) — Pure-Python frame-diff motion detector over a BGR appsink (BT.601 luma, mean abs delta). Numpy lazy-imported in `detect`. `__init__(*, threshold=0.02, downsample=1)`.
  - `detect(frame_bgr, width, height, sensor_id) -> PerceptionEventMetadata | None` (L139) — Mean abs luma delta vs the previous frame; emits `MotionMetadata` on threshold cross, with a tight axis-aligned bbox around moving pixels.
  - `summarise(metadata) -> str` (L195) — One-line motion-event summary.
- `class SceneChangeDetector` (L209) — Grayscale-histogram scene-change detector (`chisqr_alt` distance, 32 bins). `__init__(*, threshold=0.5)`.
  - `detect(frame_bgr, width, height, sensor_id) -> PerceptionEventMetadata | None` (L243) — Chi-square-alt distance between consecutive 32-bin grayscale histograms; emits `SceneChangeMetadata` on threshold cross.
  - `summarise(metadata) -> str` (L286) — One-line scene-change-event summary.
- `class _TokenBucket` (L296) — Per-`(sensor, kind)` rate-limit primitive; mirrors `openral_observability.failure_bus._TokenBucket` but independently implemented to keep the runner free of an observability-package dep.
- `class PerceptionEventPublisher` (L328) — Owns one event-sink appsink for one sensor; fans out to one `Publisher` per detector kind. Constructor enforces unique `kind`s, absolute `topic_prefix`, positive `rate_hz`. QoS: `BEST_EFFORT + VOLATILE + KEEP_LAST=10`.
  - `is_started` [@property] (L418) — `True` between `start` and `stop`.
  - `dropped_counts` [@property] (L423) — Per-kind count of detections suppressed by the rate-limit gate.
  - `start() -> None` (L427) — Initialise rclpy (if needed), create one publisher per detector kind, hook the appsink.
  - `stop() -> None` (L482) — Disconnect the signal, destroy publishers, shut down rclpy (if this instance owns it). Idempotent.

### `python/runner/src/openral_runner/backends/gstreamer/tee_manager.py`
_Runtime tee-branch manager for the GStreamer perception bus. Attaches / detaches consumer branches on a running pipeline's named `tee` (`pipeline.TEE_NAME`) via dynamic pad add/remove — the mechanism the S2 reasoner drives through `ExecuteRskill`. Imports `gi` at load (requires the `gstreamer` extra)._

- module constant `_REQUEST_PAD_TEMPLATE: Final[str] = "src_%u"` (L56) — tee request-pad template name passed to `request_pad_simple`.
- module constant `_DETACH_TIMEOUT_S: Final[float] = 5.0` (L61) — how long `detach` waits for the IDLE probe before raising (pipeline presumed stalled).
- `class BranchHandle` (L65) — Opaque dataclass handle to an attached branch (`name` + the private `tee` pad / branch bin); returned by `attach`, passed back to `detach`.
- `class TeeManager` (L81) — `__init__(pipeline, *, tee_name=TEE_NAME)` — raises `ROSConfigError` if the tee is absent.
  - `branch_count` [@property] (L115) — Number of currently attached branches.
  - `attach(elements, *, name) -> BranchHandle` (L120) — requests a tee pad, parses `LEAKY_BRANCH_QUEUE ! <elements>` into a bin, links + syncs it live; rolls back on link failure.
  - `detach(handle) -> None` (L190) — IDLE-probe unlink + release-pad + NULL teardown; idempotent, blocks until removed (bounded by `_DETACH_TIMEOUT_S`).

### `python/runner/src/openral_runner/backends/gstreamer/objects_detector.py`
_CPU-tier object detector for the perception event tee. Implements `EventDetector` via ONNXRuntime on system-memory BGR frames (RT-DETR / D-FINE ONNX signature). `onnxruntime` lazy-imported at construction time. Zero-copy NVMM tiers are a planned follow-up; requesting them raises `ROSConfigError`._

- module constant `_DETECTOR_TIERS_GROUP = "openral.detector_tiers"` (L74) — entry-point group name the `NVMM_AGGREGATOR` tier resolves against (registered by the private `openral-pro-trt` package).
- `class DetectorTier(str, Enum)` (L80) — Execution tier for the object detector: `CPU_ONNX`, `NVINFER`, `NVMM_AGGREGATOR`, `VLM_SIDECAR` (out-of-process open-vocab VLM), `ZEROSHOT_HF` (in-process Transformers zero-shot over a fixed vocabulary). The latter two reuse the `CPU_ONNX` BGR appsink branch.
- `select_detector_tier(platform=None) -> DetectorTier` (L125) — Probes `gst-inspect-1.0 nvinfer` (→ `NVINFER`), then checks for `Platform.TEGRA` (→ `NVMM_AGGREGATOR`), else `CPU_ONNX`. `nvinfer` probe always wins over explicit `platform`.
- `identify_rtdetr_outputs(named_shapes: list[tuple[str, tuple[Any, ...]]]) -> tuple[str, str]` (L213) — Tier-agnostic helper: from a list of `(name, shape)` output pairs, returns `(logits_name, boxes_name)`. Among 3-D outputs, the one with last-dim==4 is boxes; if both (or neither) end in 4, falls back to index order (0=logits, 1=boxes). Raises `ROSConfigError` if fewer than two 3-D outputs are present.
- `postprocess_rtdetr(logits, boxes, *, labels, model_id, sensor_id, score_threshold, frame_width, frame_height) -> ObjectsMetadata | None` (L251) — Tier-agnostic RT-DETR decode: sigmoid→argmax→threshold, cxcywh→xyxy pixels, degenerate-bbox guard, sorts descending by confidence, returns `None` on zero survivors.
- `class ObjectsDetector` (L350) — `EventDetector` implementation. `__init__(onnx_path, *, labels, model_id, input_size=(640,640), score_threshold=0.5, device="cpu")`. Delegates logits/boxes identification to `identify_rtdetr_outputs`.
  - `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` (L456) — BGR→RGB, NN-resize, float32/255, NCHW, ORT inference, delegates postprocessing to `postprocess_rtdetr`.
  - `summarise(metadata) -> str` (L512) — Aggregates label counts as `"Nx label"` string.
- `make_objects_detector(onnx_path, *, labels, model_id, tier=None, **kwargs) -> ObjectsDetector | object` (L553) — Resolves a detector by tier (auto-selected via `select_detector_tier` when unset): `CPU_ONNX` → `ObjectsDetector`; `NVMM_AGGREGATOR` → the `openral.detector_tiers` entry point (private `openral-pro-trt`); `NVINFER` and unknown tiers raise `ROSConfigError`.

> **Moved to OpenRAL Pro:** the NVMM zero-copy consumers — `nvbufsurface.py`, `cuda_context.py`, `trt_nvmm.py` (`TrtNvmmExecutor`), `nvmm_detector.py` (`NvmmObjectsDetector`), `nvmm_vision_encoder.py`, `act_nvmm.py` — now live in the private `openral-pro-trt` package as `openral_pro_trt.*`; the `NVMM_AGGREGATOR` tier resolves via the `openral.detector_tiers` entry-point group. The open perception bus (`pipeline.py`, `reader.py`, `tee_manager.py`, CPU/VLM detector tiers) is unchanged.

### `python/runner/src/openral_runner/backends/gstreamer/detector_factory.py`
_gi-free dispatch seam so the manifest→detector-backend selection is unit-testable without a live pipeline. `DetectorRunner` delegates construction here. No `gi`/`onnxruntime`/`zmq`/`torch` at import; the `pytorch` and `zeroshot_hf` branches lazy-import their detector classes._

- `weights_source_from_manifest(manifest) -> str` (L97) — Resolves the HF repo the backend loads: prefers `source_repo`, falls back to `weights_uri`, else `nvidia/LocateAnything-3B`; strips the `hf://` scheme and any `@revision` to a bare `org/name`.
- `build_manifest_detector(manifest, *, onnx_path=None, tier=None) -> tuple[Any, DetectorTier]` (L125) — Dispatches on `manifest.detector.engine` (then `manifest.runtime`) to the matching backend — the `zeroshot_hf`/`pytorch` sidecar/VLM tiers need no `onnx_path`, `onnx`/`tensorrt` do. Raises `ROSConfigError` for a non-detector manifest or a missing `onnx_path` when required.
- `class DetectorNodeWiring` (frozen dataclass) (L50) — `run_continuous_leg: bool`, `serve_on_demand: bool`; the perception node's wiring decision, pure and rclpy-free.
- `detector_node_wiring(mode: DetectorMode) -> DetectorNodeWiring` (L69) — pure (rclpy-free, unit-testable) policy the perception node consumes: `continuous` → `run_continuous_leg=True, serve_on_demand=False` (publish leg, no query service); `on_demand` → `run_continuous_leg=False, serve_on_demand=True` (locate_in_view service + `detector_query` topic, no continuous publishing).

### `python/runner/src/openral_runner/backends/gstreamer/omdet_turbo_detector.py`
_In-process Transformers open-vocabulary detector — `omlab/omdet-turbo-swin-tiny-hf` (Apache-2.0). One backend serves both detector modes (declared by the manifest's `detector.mode`): `continuous` (fixed `labels`, unprompted) or `on_demand` (prompted via `set_query`/`detect_with_query`). Same interface as `ObjectsDetector`, so it reuses the CPU BGR appsink branch; loads under the runtime's own `transformers>=5`, no sidecar._

- module constant `_MAX_AREA_FRAC = 0.98` (L55) — degenerate-box guard: drops detections covering ≥98% of the frame.
- `build_objects_metadata_from_results(*, labels, scores, boxes_xyxy, width, height, model_id, sensor_id, score_threshold) -> ObjectsMetadata | None` (L58) — Pure (no torch): from decoded per-detection `labels`/`scores`/pixel `boxes_xyxy`, drops sub-threshold + degenerate/near-full-image (≥98%) boxes, clips + corner-orders to frame, sorts descending by confidence; `None` on zero survivors. Raises `ROSConfigError` on length mismatch.
- `query_to_classes(query) -> list[str]` (L143) — Pure: parse a free-text on-demand query into OmDet's multi-label class list (split on commas / `</c>`; a single phrase is one class; whitespace dropped). Raises `ROSConfigError` if empty.
- `class OmDetTurboDetector` (L175) — `__init__(*, labels, model_id, weights_source, score_threshold=0.3, nms_threshold=0.5, device="auto")` — stores config; model/processor load deferred to first `detect()` (lazy, side-effect-free; `device="auto"` → CUDA when available else CPU).
  - `set_query(text) -> None` (L298) — Retarget the persistent vocabulary (the `detector_query` topic; on-demand), parsed via `query_to_classes`.
  - `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` (L310) — Detect over the current (persistent) vocabulary.
  - `detect_with_query(frame_bgr, width, height, sensor_id, query) -> ObjectsMetadata | None` (L327) — One-shot detect for `query` WITHOUT mutating the persistent vocabulary (the read-only `locate_in_view` service). Both `detect` and `detect_with_query` delegate to `_detect_classes` (BGR→RGB PIL, processor over the class list, `model(**inputs)` under `no_grad`, `post_process_grounded_object_detection`, → `build_objects_metadata_from_results`).
  - `close() -> None` (L383) — Releases the model + `cuda.empty_cache()` if loaded on GPU; idempotent.

### `python/runner/src/openral_runner/backends/gstreamer/segmenter_factory.py`
_gi-free dispatch seam for `kind: segmenter` rSkills — the sibling of `detector_factory.py`. Dispatch keys on the required `manifest.segmenter.engine` (no legacy `runtime`-keyed fallback for this kind). Reuses `weights_source_from_manifest` from the detector factory rather than reimplementing `hf://` stripping._

- `build_manifest_segmenter(manifest, *, device="auto") -> Sam2Segmenter` (L28) — `engine: sam2_hf` → `Sam2Segmenter` configured from the contract's `max_prompt_points` / `multimask` / `min_mask_area_px`. Raises `ROSConfigError` if the manifest is not a `kind:segmenter` with a segmenter block, or names an engine with no backend. `device` is host posture, **not** manifest contract (the same weights run on either): `"auto"` picks CUDA when available, and an explicit `"cpu"` is the honest setting on a pre-sm_70 dev GPU where the CUDA wheels ship no kernels for the card and `torch.cuda.is_available()` alone would mislead the auto path. `segmenter_node` forwards its `device` parameter here.

### `python/runner/src/openral_runner/backends/gstreamer/sam2_segmenter.py`
_In-process Transformers SAM 2.1 promptable segmenter — `facebook/sam2.1-hiera-small` (Apache-2.0), the backend for `engine: sam2_hf`. Answers a geometric prompt (a point) with binary masks, where a detector answers a semantic one (a label) with scored boxes — no label vocabulary, no thresholdable confidence. Held resident and warmed at node activate, since a cold call would not fit inside the HAL's deferred-ack barrier._

- module constant `_DEPTH_EPS = 1e-6` (L65) — a point closer than this to the camera's optical plane has no meaningful projection.
- `class MaskCandidate` (L69) — Frozen dataclass: one mask hypothesis — `mask` (`(H, W)` bool at the source frame's resolution), `score_advisory` (the model's own IoU estimate, **advisory only**, never thresholded), `area_px`.
- `project_point_to_pixel(point_xyz, intrinsics) -> tuple[float, float] | None` (L92) — Pure: project a camera **optical**-frame point (REP-103) to continuous pixels. `None` when behind/on the optical plane or outside the image — the caller then has no usable prompt rather than a guessed one. This is where intrinsics are applied; the `SegmentInView` prompt crosses the HAL boundary as a 3-D point precisely so they stay on this side.
- `build_mask_candidates(masks, *, scores, min_mask_area_px) -> list[MaskCandidate]` (L340) — Pure (no torch): drops masks below `min_mask_area_px` as degenerate and sorts survivors by **area ascending** (SAM 2's nested subpart → part → whole), **never** by score. Raises `ROSConfigError` on masks/scores length mismatch.
- `class Sam2Segmenter` (L133) — `__init__(*, model_id, weights_source, max_prompt_points=8, multimask=True, min_mask_area_px=64, device="auto")` — stores config; the `Sam2Processor` + `Sam2Model` load is deferred to first use (bf16 on CUDA, fp32 on CPU).
  - `model_id` [@property] (L191) — Identifier recorded alongside every emitted mask.
  - `warm_up(*, width=320, height=240) -> None` (L222) — Load and burn one forward pass at node activate, moving the ~742 ms cold call off the first real attach event; **not optional**.
  - `segment(frame_bgr, width, height, *, positive_points, negative_points=()) -> list[MaskCandidate]` (L246) — One BGR frame plus pixel prompts (positive at the TCP, negative typically at the jaw tips) → surviving candidates, area ascending. Returns `[]` on a caps-mismatched buffer; raises `ROSConfigError` with no positive point or over `max_prompt_points`, `ROSRuntimeError` without the wheels.
  - `close() -> None` (L331) — Releases the model + `cuda.empty_cache()` if loaded on GPU.

  Candidates are handed to the consumer rather than resolved here by score, because only geometry can pick between them: a mis-aimed point prompt on a real frame returned a mask covering 59.8% of the image at this model's **top** score of 0.977.

### `python/runner/src/openral_runner/backends/gstreamer/_zmq_sidecar.py`
_Private. Shared ZMQ REQ/REP transport mixin for the two sidecar-client backends below — `_connect`/`_rpc`/`_try_ping` were byte-identical between `LocateAnythingDetector` and `QwenSceneVlm`; consolidated here (see `docs/methods/14-duplication-watch.md`). `_spawn_and_wait`, `_ensure_ready` and `close` stay per-class — they genuinely differ (CLI args, error messages, ports)._

- `class ZmqSidecarMixin` (L17) — Shared transport mixin: `_connect()` opens a fresh `zmq.REQ` socket to `tcp://{host}:{port}`. `_rpc()` does a msgpack round-trip and recreates the socket on a timeout (REQ/REP can't recover from a missed reply); `_try_ping()` is a best-effort liveness check.

### `python/runner/src/openral_runner/backends/gstreamer/locateanything_detector.py`
_Open-vocabulary detector backend for the LocateAnything-3B sidecar. Same interface as `ObjectsDetector`, so it reuses the CPU BGR appsink branch. Connects lazily on first `detect()` and auto-spawns the sidecar (ping → `Popen` → poll → `close`); the model runs in an isolated `transformers==4.57.1` venv, this module is the ZMQ/msgpack client._

- module constant `_TOKEN_RE` (L45) — compiled regex matching `<ref>label</ref>` or a 4-coord `<box>` token (point boxes are ignored).
- module constant `_MIN_SIDE_FRAC = 0.02` (L50) — drops boxes thinner than 2% of the image in either axis (degenerate-box guard).
- module constant `_MAX_AREA_FRAC = 0.85` (L51) — drops boxes covering more than 85% of the image (degenerate-box guard).
- `parse_grounding_answer(answer, *, fallback_label="object", norm=1000) -> list[tuple[str, tuple[int,int,int,int]]]` (L54) — Parses `<ref>label</ref>` + `<box>` tokens in document order, binding each box to the most recent ref; coords stay normalized `[0,norm]`. Drops exact duplicates and degenerate boxes — the repeated-box tail a looping decode emits.
- `build_objects_metadata(answer, *, width, height, model_id, sensor_id, fallback_label="object", norm=1000) -> ObjectsMetadata | None` (L96) — Scales `parse_grounding_answer` boxes into `width`×`height` pixels (clipped), builds `ObjectDetection2D` at `confidence=1.0` (a grounding model has no per-box score); `None` if no valid detections.
- `class LocateAnythingDetector(ZmqSidecarMixin)` (L150) — `__init__(*, labels, model_id, weights_source="nvidia/LocateAnything-3B", host="127.0.0.1", port=5757, query=None, auto_spawn=True, boot_timeout_s=1200.0, request_timeout_s=180.0, max_side=1024, max_new_tokens=1024, mode="hybrid")` — stores config; static default `query = "</c>".join(labels)`; no connection (lazy).
  - `set_query(text) -> None` (L262) — Runtime open-vocab override for the continuous leg.
  - `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` (L268) — One-shot detect of the persistent query (delegates to `detect_with_query`).
  - `detect_with_query(frame_bgr, width, height, sensor_id, query) -> ObjectsMetadata | None` (L274) — One-shot detect for `query` WITHOUT mutating the persistent query; used by the `locate_in_view` service so an on-demand reasoner query doesn't change what the continuous leg grounds.
  - `close() -> None` (L314) — Closes the socket and terminates the sidecar if spawned; idempotent.

### `python/runner/src/openral_runner/backends/gstreamer/qwen_scene_vlm.py`

_Scene-VLM backend backed by the Qwen3.5-4B sidecar — the scene-reasoning counterpart of `LocateAnythingDetector`. Returns **text**, not `ObjectsMetadata` (a reasoning aid for task-progress / success verification, not a localizer). Same ZMQ lifecycle (lazy connect, auto-spawn, teardown only the child), transport shared via `ZmqSidecarMixin`. No `zmq`/`numpy`/`PIL` at import (all lazy)._

- `class QwenSceneVlm(ZmqSidecarMixin)` (L47) — `__init__(*, model_id, weights_source="Qwen/Qwen3.5-4B", host="127.0.0.1", port=5759, auto_spawn=True, boot_timeout_s=1200.0, request_timeout_s=180.0, max_side=1024, max_new_tokens=256)` — stores config; no connection (lazy).
  - `query(frame_bgr, width, height, question) -> str` (L143) — Encode BGR→PNG, RPC `{"op":"query",...}`, return the whitespace-stripped answer; raises `ROSConfigError` on empty question or sidecar error.
  - `close() -> None` (L184) — Closes the socket + terminates the spawned sidecar; idempotent.
- `build_scene_vlm(manifest, *, host="127.0.0.1", port=5759) -> QwenSceneVlm` (L200) — build from a `kind:"vlm"` manifest; `model_id=manifest.name`, `weights_source` from `weights_uri` (the deployable pre-quant checkpoint) stripped of `hf://`/`@rev`. Raises `ROSConfigError` if `manifest.kind != "vlm"`. Lazy.

### `python/runner/src/openral_runner/backends/reward/frame_source.py`
_Transport-agnostic rolling frame buffer for the reward monitor. Pure Python + stdlib only (no numpy/torch), so it unit-tests without ROS, torch, or a GPU._

- module constant `_NS_PER_S = 1_000_000_000` (L20) — nanosecond/second conversion factor.
- `class Frame` (frozen dataclass) (L24) — one buffered camera frame: `stamp_ns: int`, `bgr: bytes`, `width: int`, `height: int`.
- `class RollingFrameBuffer` (L40) — `__init__(*, window_s, max_frames=256, stale_after_s=3.0)` — transport-agnostic node-side ring of recent frames (sim + real).
  - `push(frame) -> None` (L79) — Append + evict frames older than `window_s` relative to the newest / over `max_frames`.
  - `window(seconds) -> list[Frame]` (L88) — Frames within the last `seconds` (capped to `window_s`).
  - `is_stale(now_ns) -> bool` (L99) — True if no fresh frame within `stale_after_s`.
  - `__len__() -> int` (L105) — Number of frames currently buffered.
- module constant `_MIN_POINTS_FOR_SLOPE = 2` (L111) — a least-squares slope needs at least two points.
- `trend(series: list[float]) -> float` (L114) — least-squares slope per sample (0.0 for < 2 points); used for progress/success trend + `stalled`.
- module constant `_STALL_TREND_EPS = 0.002` (L133) — |progress trend per sample| below this reads as "stalled"; shared by `RobometerInProcessReward.assess` and `TOPRewardMonitor.assess`.
- `assess_from_score(progress, success, *, success_threshold, frames_seen) -> dict` (L136) — Shared `assess()` dict builder, consolidated out of `RobometerInProcessReward.assess` and `TOPRewardMonitor.assess` (see `docs/methods/14-duplication-watch.md`).

### `python/runner/src/openral_runner/backends/reward/robometer_reward.py`
_Robometer reward-monitor backend — loads lerobot 0.6.0's native Robometer model inside `reward_monitor_node` and scores clips on demand. Nothing here imports torch / transformers / numpy at module load._

- `critic_score_from_assessment(assessment, *, threshold) -> tuple[float, float]` (L37) — Pure mapping from a reward assessment result to a generic `openral_msgs/CriticScore` `(score, threshold)`: uses `progress_now` (higher-is-better) as the score, clamped to `[0, 1]`, defaulting a missing/non-numeric/bool value to `0.0`. Lets `reward_monitor_node` feed the Tier-C critic producer. Pure, ROS-free, unit-tested.
- `class RobometerInProcessReward` (L147) — `__init__(*, model_id, weights_source="OpenRAL/rskill-robometer_4b-any-general-nf4", num_bins=100, success_threshold=0.5, max_frames=8, device="cuda")` — default Robometer backend for `reward_monitor_node`: lazily imports `tools/_robometer_scorer.py::_Scorer`, meta-loads the prequantized NF4 checkpoint in the reward-monitor process, and scores BGR frames via the same native lerobot 0.6.0 `_compute_rbm_logits` + `decode_progress_outputs` path.
  - `score(frames, task) -> (progress, success)` (L179) — Validates empty task/clip + frame sizes, evenly subsamples to `max_frames`, converts BGR→RGB, returns per-frame normalized arrays.
  - `assess(frames, task) -> dict` (L191) — Scores `frames` and summarizes the window for the Reasoner; mirrors the reasoner contract.
  - `close() -> None` (L200) — Releases the model + `cuda.empty_cache()` if loaded on GPU.
- `build_reward_monitor(manifest) -> RobometerInProcessReward | TOPRewardMonitor` (L209) — build from a `kind:"reward"` manifest; dispatches `reward.backend=="topreward"` to `TOPRewardMonitor` and defaults Robometer to `RobometerInProcessReward`. Raises `ROSConfigError` if `manifest.kind != "reward"`. Lazy.

### `python/runner/src/openral_runner/backends/reward/topreward_reward.py`
_In-process TOPReward (arXiv 2602.19313) reward monitor — a zero-shot reward reading `P("True" | video, instruction)` from an off-the-shelf Qwen3-VL VLM, no fine-tuned checkpoint and no ZMQ sidecar. Nothing here imports torch / transformers at module load._

- `class TOPRewardMonitor` (L36) — `__init__(*, model_id, weights_source, success_threshold=0.8, max_frames=8, num_samples=6, fps=2.0, device="cuda")` — stores config; the VLM is loaded lazily on first `score`.
  - `score(frames, task) -> (progress, success)` (L134) — Prefix-sweep: scores `frames[:L]` at `num_samples` anchor lengths, min-max normalizes raw log-probs within the queried window, then interpolates to one value per frame. `success` mirrors `progress` (TOPReward has a single head).
  - `assess(frames, task) -> dict` (L184) — Scores `frames` and summarizes the window for the Reasoner; same keys as `RobometerInProcessReward.assess`.
  - `close() -> None` (L197) — Releases the in-process model + frees CUDA memory.
- `build_topreward_monitor(manifest, *, device="cuda") -> TOPRewardMonitor` (L212) — Build a `TOPRewardMonitor` from a `reward.backend == "topreward"` manifest (weights source, success threshold, target fps). Raises `ROSConfigError` on a non-reward manifest.

### `python/runner/src/openral_runner/backends/gstreamer/detector_runner.py`
_Runtime glue that wires a ``kind: detector`` rSkill to a live camera pipeline — loads the `DetectorContract`, delegates backend construction to `build_manifest_detector` (ONNX CPU/NVMM tiers or the `VLM_SIDECAR` open-vocab tier), attaches the appropriate branch to the bus tee via `TeeManager`, and fires the `on_detection` callback for each non-`None` `ObjectsMetadata`. Imports `gi` + `DetectorTier`/`build_manifest_detector` + `nvmm_convert_element` eagerly at load._

- `class DetectorRunner` (L59) — `__init__(pipeline, manifest, *, onnx_path=None, sensor_id, on_detection, tee_name=TEE_NAME, tier=None)` — validates `manifest.kind == "detector"`, caches NVMM caps dims from `DetectorContract.input_size`, delegates backend construction to `build_manifest_detector`, and creates a `TeeManager`.
  - `start() -> None` (L156) — Attaches the tee branch matching the tier: `NVMM_AGGREGATOR` gets the NVMM RGBA appsink, every other tier a BGR appsink; raises `ROSRuntimeError` if the appsink isn't found after attach.
  - `_on_sample_bgr(appsink) -> int` (L220) — pulls BGR sample, format assert, buffer.map/unmap, calls `detector.detect`, fires `on_detection` on non-`None`; errors guarded.
  - `_on_sample_nvmm(appsink) -> int` (L279) — pulls NVMM sample, `wrap_buffer` (lazy import from the private `openral_pro_trt.nvbufsurface`), calls the entry-point-resolved NVMM detector's `detect_nvmm`, fires `on_detection`; always unmaps; errors guarded.
  - `stop() -> None` (L336) — disconnects signal + detaches branch + calls `detector.close()` if present; idempotent.

### `python/runner/src/openral_runner/__init__.py`
_Public surface of the inference runner. Imports are PEP 562 lazy: heavy symbols (`InferenceRunnerBase`, `factory.*`, `DeployRunner`, `safety.*`) are resolved on first attribute access so importing any subpackage does not eagerly drag in torch or trigger downstream glib conflicts._

- light eager imports: `precise_sleep`, `sleep_until`, `InferenceRunner` (Protocol), `SensorReader` (Protocol).
- `_LAZY_ATTRS: dict[str, tuple[str, str]]` — `attr → (module, name)` map driving the `__getattr__` resolver. (L70)
- `__getattr__(name) -> Any` — Resolves heavy symbols on first access (torch / glib-sensitive deferral). (L84)

### `python/runner/src/openral_runner/factory.py`
_Library deploy runner used by runtime nodes; the public deploy CLI now shells the ROS graph from a `DeployScene`._

- `SKILL_REGISTRY: dict[str, Callable[[dict[str, object]], rSkillBase]]` — `vla.id` → skill factory. Today: `hello`, `gpu_passthrough`. (L98)
- `SENSOR_BACKEND_REGISTRY: dict[str, Callable[[SensorReaderConfig], SensorReader]]` — `backend` id → reader factory. Today: `opencv_thread`, `ros2_image`, `gstreamer`, `galaxea_a1_camera_bridge`. (`ros2_image` was in the `SensorReaderBackend` enum but absent here, so selecting it raised `unknown sensor reader backend`.) (L383)
- `_to_int(value, *, field, sensor_id) -> int` — YAML `object` → `int` coercion helper used across factories; rejects bools explicitly. (L48)
- `_make_gpu_passthrough_skill(extra) -> rSkillBase` — Builds `GpuPassthroughSkill`; recognised `extra`: `sensor_id` (default `"wrist_rgb"`), `n_joints`, `horizon`, `device` (default `"cuda"`, raises if unavailable). (L75)
- `_make_opencv_thread_reader(cfg) -> SensorReader` — Builds `OpenCVThreadSensorReader` from a `SensorReaderConfig`; requires `backend_params.device`, forwards optional `fps`/`width`/`height`/`crop` (`[x, y, width, height]`); an invalid value raises `ROSConfigError`.
- `_make_ros2_image_reader(cfg) -> SensorReader` — Builds `Ros2ImageSensorReader`; requires `backend_params.topic` (e.g. `/zed/depth/depth_registered`), optional `reliability` (`best_effort` default / `reliable`) and `qos_depth` (default 5). `cfg.max_age_ms` becomes the reader's staleness budget. Imported lazily so the factory module stays importable without ROS. (L346)
- `_make_gstreamer_reader(cfg) -> SensorReader` — Builds `GStreamerSensorReader` from a `SensorReaderConfig`. Translates `publish_to_ros` / `publish_topic` / `publish_rate_hz` → `PipelineSpec.enable_ros_tee`. (L159)
- `_make_galaxea_a1_camera_bridge_reader(cfg) -> SensorReader` — Builds the
  native A1 Runtime paired-camera connector. Accepts only `camera`; unknown
  values are rejected.
- `make_sensor_readers(configs) -> list[SensorReader]` (L266) — Batch constructor that
  preserves config order and shares one A1 paired-camera session across both
  views. Other backends still dispatch through `SENSOR_BACKEND_REGISTRY`.

### `python/runner/src/openral_runner/deploy_runner.py`
_``DeployRunner`` — concrete `InferenceRunnerBase` subclass composing HAL + Skill + WorldStateAggregator + SensorReaders + SafetyClient._

- module constant `_THUMBNAIL_HZ = 25.0` (L67) — fixed cadence for the dashboard JPEG thumbnail emission.
- `class DeployRunner(InferenceRunnerBase)` — Closes the `WorldState → Skill → safety → HAL` loop on real hardware / digital twins. It is the safety-supervisor boundary: catches `ROSSafetyViolation` from the `SafetyClient`, records it on the `TickResult`, and withholds `HAL.send_action` rather than re-raising — withholding is today's mitigation. (L72)
  - `__init__(*, hal, skill, aggregator, sensor_readers=(), safety_client=None, recorder=None, **base_kwargs)` — Caller must pre-`configure()`+`activate()` the skill; runner manages HAL + reader open/close. Defaults `safety_client` to `NullSafetyClient`. Dashboard JPEG thumbnails are emitted at a private fixed cadence. Optional `recorder` is a `openral_dataset.RolloutRecorder`; when set, `episode_start` / `episode_end` drive its lifecycle and every tick fans out via `record_frame`. (L114)
  - `episode_start(task_string: str) -> int` — Open a new episode on the attached recorder; returns the new `episode_idx` (or `-1` when no recorder is attached). Raises `RuntimeError` if called twice without `episode_end`. (L173)
  - `episode_end(*, success: bool) -> None` — Close the current recorder episode with the success flag. No-op when no recorder is attached. Raises `RuntimeError` if called without `episode_start`. (L201)
  - `activate() -> None` — `super().activate()` + `hal.connect()` + open every `SensorReader`. (L227)
  - `deactivate() -> None` — Close every `SensorReader` (best-effort; logs + continues), `hal.disconnect()`, `super().deactivate()`. (L245)
  - `_tick_impl(tick_idx) -> TickResult` — Five-phase tick: sensors → world_state → inference → safety → hal, each phase timed onto the `TickResult`/OTel span. Catches `ROSPerceptionStale` per reader and `ROSSafetyViolation` at the supervisor boundary, recording each rather than letting it crash the tick. (L316)
  - `_tracer` [@property] — Per-call `trace.get_tracer("openral")` (never cached at `__init__`, would bind to the provider live at construction time). (L218)
  - `_hal_adapter_label` — Lower-cased class name of the HAL adapter, used as the closed-set `openral.hal.adapter` value on spans + metrics. (L157)

### `python/runner/src/openral_runner/safety.py`
_``SafetyClient`` stub — Python-side seam for the future C++ safety kernel._

- `class SafetyClient(Protocol)` — `@runtime_checkable` Protocol. `check_action(action)` returns `None` to allow or raises `ROSSafetyViolation` to reject; the inference runner catches it only at its supervisor boundary, never silently. (L40)
  - attr `envelope: SafetyEnvelope` — the envelope checked against.
  - `check_action(action: Action) -> None` (L56)
- `class NullSafetyClient` — no-op stub that always allows. Every call opens a `safety.check` OTel span at `severity="info"` carrying `control_mode`, `horizon`, `envelope_max_ee_speed_m_s`, `envelope_max_force_n`. Used by digital-twin runs and pre-hardware tests so traces show the seam is wired before the C++ kernel arrives. (L72)
  - `__init__(envelope: SafetyEnvelope | None = None)` — defaults to a stock `SafetyEnvelope`. (L97)
  - `check_action(action: Action) -> None` (L101)

### `python/runner/src/openral_runner/base.py`
_Shared base for inference runners. Subclasses override `_tick_impl`._

- module constant `_DEADLINE_LOG_PERIOD_S = 5.0` (L44) — minimum spacing between consecutive deadline-miss WARN log lines (rate-limit period consumed by `_deadline_log_due`).
- `_percentile(samples: list[float], q: float) -> float` — Linear-interpolation percentile (`0.0` for empty list). Used by `_build_run_result`. (L47)
- `class InferenceRunnerBase(ABC)` — Owns the rate-limited loop, `rskill.tick` OTel parent span, `RunResult` aggregation, deadline-overrun policy. (L65)
  - `__init__(*, rate_hz=30.0, deadline_overrun_policy=WARN, runner_name="inference_runner", latency_budget_ms=None, save_dir=None)` — Reject `rate_hz <= 0`. (L101)
  - `activate() -> None` — Reset tick counter; mark active. (L128)
  - `deactivate() -> None` — Stop ticking; idempotent. (L133)
  - `_tick_impl(tick_idx: int) -> TickResult` [@abstractmethod] — Subclass hook; the base wraps it in a `rskill.tick` span. (L140)
  - `episode_start(task_string: str) -> int` — Optional explicit episode boundary; default raises `NotImplementedError`. `DeployRunner` overrides to drive the recorder; `SimRunner` overrides as a no-op (sim derives episode boundaries from `env.step` flags). (L177)
  - `episode_end(*, success: bool) -> None` — Optional explicit episode boundary; default raises `NotImplementedError`. See `episode_start`. (L202)
  - `_should_terminate() -> bool` — Subclass early-exit hook (default False) consulted after each tick inside `run()`. `SimRunner` overrides to stop once `n_episodes` complete. (L154)
  - `tick() -> TickResult` — Span-wrapped single-tick entry; attaches per-stage timings as `skill.{tick_ms, inference_ms, sensors_ms, world_state_ms, safety_ms, hal_ms, action_applied, safety_violations}` attributes plus sim-only `skill.{step_idx, episode_idx, reward, terminated, truncated}` when set, plus `openral.tick.idx`. Records `openral.tick.duration` / `openral.inference.duration` histograms (label: `skill.id`) and increments `openral.safety.violations{check_name="runtime", severity="violation"}` for each violation on the tick. (L221)
  - `run(max_ticks: int | None = None) -> RunResult` — Rate-limited loop using `sleep_until`. Applies `DeadlineOverrunPolicy` (`warn` / `drop` / `raise`). Records `latency_budget_ms` violations and increments `openral.tick.budget_violations` per violation. Honors `_should_terminate()` after each tick. (L308)
  - `_on_deadline_overrun(result: TickResult) -> None` — Apply policy: structlog warn / drop / raise `ROSDeadlineMissed`. Always increments `openral.tick.deadline_misses` and emits `openral.event.deadline_missed` on the current parent span; on `RAISE`, also calls `record_exception` + `set_status(ERROR)` on the parent span before re-raising. The `WARN` / `DROP` **log lines** are rate-limited via `_deadline_log_due`; the metric and the span event are not. (L393)
  - `_deadline_log_due(tick_ms: float) -> tuple[bool, int, float]` — Rate-limits the deadline-miss WARN log to one line per `_DEADLINE_LOG_PERIOD_S` (5 s) per runner, since a genuinely slow host misses every tick and would otherwise flood the dashboard Event Log. No miss is lost — suppressed counts and the window's `worst_tick_ms` fold into the next logged line, and `openral.tick.deadline_misses` still counts every miss regardless.
  - `_build_run_result(results, *, budget_violations, trace_id) -> RunResult` — Aggregate per-tick records into `RunResult` (mean / p99). (L454)
