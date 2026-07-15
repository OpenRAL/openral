# Inference Runner (executor of S1)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

The hardware-side counterpart to `openral_sim` — closes
`WorldState → Skill.step → SafetyClient.check → HAL.send_action` at the runner
tick rate. `DeployScene` is the deploy/workcell YAML used by the ROS deploy
graph; `openral_runner` remains the library executor behind runtime nodes.

### `python/runner/src/openral_runner/clock.py`
_High-precision cadence helpers for the inference runner._

- `precise_sleep(duration_s: float) -> None` — Hybrid sleep: `time.sleep` for the bulk + busy-wait on `time.perf_counter` for the final ~1 ms. Mirrors lerobot's `precise_sleep` shape. Non-positive `duration_s` is a no-op. (L29)
- `sleep_until(deadline_perf_counter_s: float) -> None` — Convenience wrapper taking an absolute `time.perf_counter` deadline. Used by `InferenceRunnerBase.run` to enforce cadence. (L60)
- module constant `_BUSY_LOOP_THRESHOLD_S = 1e-3` — Busy-wait threshold (mirrors lerobot's private constant in `lerobot_record.py`). (L26)

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

- module constant `WORLD_CLOUD_TOPIC_DEFAULT = "/octomap_point_cloud_centers"` — default occupied-voxel-centers PointCloud2 topic (octomap_server). (L52)
- `crop_points_to_box(points, *, xy_m, z_min, z_max) -> NDArray[float32]` — keep `(N,3)` points inside the local box around base_link. (L78)
- `distance_to_rgb(dist_m, *, range_max_m) -> tuple[int,int,int]` — near=warm→far=cool color ramp. (L94)
- `encode_world_cloud_png(points_base, *, range_max_m=4.0, image_w=480, image_h=360, xy_m=2.0, z_min=-0.2, z_max=2.0) -> str` — crop→oblique-pinhole project→rasterize→base64 PNG. Pure; PIL-only. (L141)
- `world_cloud_span_attributes(*, points_base, frame_id, source_node, range_max_m, xy_m, z_min, z_max) -> dict[str,Any]` — assemble the `openral.world_cloud.*` span attributes. (L214)
- `class WorldCloudBridge` — constructed against a host `rclpy.node.Node`; subscribes the voxel cloud, TF2-transforms to `base_link`, throttles to 1 Hz, emits a `world.pointcloud` span. Mirrors `SlamMapBridge`. `latched: bool = True` picks the subscription durability: `True` mirrors octomap's TRANSIENT_LOCAL centers topic; `False` (the mono visual-SLAM deploy) is VOLATILE for nvblox's un-latched ESDF cloud — a TRANSIENT_LOCAL sub is incompatible with a VOLATILE pub and receives nothing. An unreadable frame (empty/field-less cloud → `read_points_numpy` asserts) is warned + skipped, never crashing the shared executor. `destroy()` releases the subscription. (L260)

### `python/runner/src/openral_runner/dataset_recorder_bridge.py`
_Bus-attached LeRobot/rosbag recorder for the deploy graph (mirrors `WorldCloudBridge`)._

- `class DatasetRecorderBridge(node, *, robot, aggregator, recorder, action_topic="/openral/candidate_action", episode_topic="/openral/episode")` — constructed against the shared runtime `rclpy.node.Node`; subscribes `Episode` (drives `recorder.episode_start/end`) + `ActionChunk` (RELIABLE depth 100). Per inference tick it joins the shared `WorldStateAggregator` snapshot (proprio + camera `image_frames`) with the tick's action, reassembling multi-slot chunks into one full action vector — grouped by `ActionChunk.tick_index` (1-based; slot-cycle on `(control_mode, ee_name)` is the fallback when `tick_index==0`). Writes via `Rosbag2Sink`. A reassembled shape the recorder rejects (vs a defined `action_spec.dim`) is logged, not raised. `destroy()` flushes the pending tick, closes the episode, finalizes the bag, releases the subscriptions. (L85)

### `python/runner/src/openral_runner/sensor_reader.py`
_:class:`SensorReader` Protocol — seam between per-sensor capture backends and the inference runner._

- `class SensorReader(Protocol)` — `@runtime_checkable` Protocol; concrete backends live under `openral_runner.backends`. (L29)
  - attr `sensor_id: str` — matches `SensorReaderConfig.sensor_id`.
  - attr `is_open: bool` — True between `open()` and `close()`.
  - `open() -> None` — Acquire device, start background workers. Idempotent. (L51)
  - `close() -> None` — Release device, join workers. Idempotent. (L60)
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` — Non-blocking peek at the most recent buffered frame; raises `ROSPerceptionStale` if no frame yet or freshest exceeds budget. (L67)

### `python/runner/src/openral_runner/backends/opencv_thread.py`
_:class:`OpenCVThreadSensorReader` — default backend. Mirrors lerobot's per-camera-thread pattern._

- module constant `_COLOR_NDIM = 3` — Number of dims for an OpenCV colour frame (`(H, W, 3)`); mono is `(H, W)`. Used to derive `SensorFrame.channels`. (L38)
- `class OpenCVThreadSensorReader` — Per-camera background-thread reader on top of `cv2.VideoCapture`. Imports `cv2` lazily inside `open()` (the `opencv` optional-extra). (L41)
  - `__init__(*, sensor_id, device, fps=30, width=None, height=None, encoding=BGR8, default_max_age_ms=100)` — Stash config; rejects non-positive `fps` / `default_max_age_ms`. (L68)
  - `open() -> None` — Open `cv2.VideoCapture`, pin `cv2.setNumThreads(1)` (lerobot parity), spawn daemon thread. Idempotent. (L106)
  - `close() -> None` — Stop event, join thread (2 s timeout), release capture. Idempotent. (L144)
  - `__enter__() / __exit__()` — Context-manager sugar; calls `open` / `close`. (L162)
  - `read_latest(max_age_ms: int | None = None) -> SensorFrame` — Lock-protected snapshot of the `_latest_frame` slot; constructs a `SensorFrame` with inlined raw bytes; raises `ROSPerceptionStale` on no-frame-yet or staleness, `RuntimeError` on closed reader. (L173)
  - `_read_loop()` — Background daemon: `cv2.VideoCapture.read` → `_latest_frame + _latest_stamp_*_ns` under lock; sleeps `1/fps` on read failure / EOF. (L228)

### `python/runner/src/openral_runner/backends/__init__.py`
_Per-backend `SensorReader` implementations. Default `OpenCVThreadSensorReader` is always available; `GStreamerSensorReader` (PR I) + `Ros2ImageSensorReader` gate on optional deps._

- `OpenCVThreadSensorReader` — lazy-exported via PEP 562 `__getattr__` (M8 PR I/8) so importing `openral_runner.backends.gstreamer` does NOT eagerly pull in `cv2`. cv2 initialises glib state that segfaults a subsequent `rclpy.Node()` inside the x86-ros Docker image; the lazy split keeps the gstreamer-only path importable in ROS-enabled processes.
- `__getattr__(name) -> Any` — PEP 562 attribute hook; resolves `OpenCVThreadSensorReader` on first access via `importlib.import_module`. (L27)

### `python/runner/src/openral_runner/backends/gstreamer/pipeline.py`
_GStreamer pipeline-string builder + platform detection. Pure-Python — does **not** import `gi` at module load._

- `TEE_NAME: Final[str]` (L60) — `"openral_cam_tee"`. Name of the per-camera `tee` — the **perception-bus attach point** the runtime `TeeManager` looks up via `Gst.Bin.get_by_name` to request pads for reasoner-activated consumers at runtime.
- `LEAKY_BRANCH_QUEUE: Final[str]` (L67) — `"queue leaky=downstream max-size-buffers=2"`. The single definition of the per-branch isolation policy, shared by the static builder and the runtime `TeeManager`.
- `leaky_branch(elements, *, tee_name=TEE_NAME) -> str` (L70) — Returns one `tee` branch `<tee>. ! <leaky queue> ! <elements>`. The shared branch-construction primitive so the static builder and the dynamic `TeeManager` build branches identically.
- `class PipelineSpec(BaseModel)` (L170) — Validated description of a GStreamer ingest pipeline. Fields: `source, device, width, height, fps, encoded, jpeg, enable_nvmm, enable_ros_tee, enable_event_tee, appsink_name, ros_appsink_name, event_appsink_name, event_rate_hz, max_buffers`. The three event-tee fields were added, along with `jpeg` (MJPG UVC cameras — USB-only, exclusive with `encoded`); the validator on `event_appsink_name` enforces valid GStreamer element names.
- `class Platform(str, Enum)` (L104) — `TEGRA | NVIDIA_DEEPSTREAM | NVIDIA_DESKTOP | CPU_ONLY`. `NVIDIA_DEEPSTREAM` is the x86 `ds-on` image: the main reader pipeline goes NVMM-native (`nvjpegdec` decodes MJPG straight into NVMM, `nvvideoconvert` converts on-GPU, appsink negotiates `memory:NVMM` RGBA).
- `class Source(str, Enum)` (L146) — `USB | CSI | RTSP | FILE | TESTSRC`.
- `detect_platform() -> Platform` (L275) — `lru_cache`d; reads `/etc/nv_tegra_release`, probes `gst-inspect-1.0` for `nvjpegdec`+`nvvideoconvert` (DeepStream) then `nvh264dec` (desktop nvcodec).
- `inspect_element_present(element_name) -> bool` (L310) — Generic `gst-inspect-1.0 --exists` probe with timeout.
- `nvmm_convert_element() -> str | None` (L340) — Probes for the host's NVMM colour-convert element: `nvvideoconvert` (DeepStream/x86) preferred, else `nvvidconv` (Tegra/L4T), else `None`.
- `ensure_appsink_name(pipeline, name) -> str` (L360) — Rewrites a trailing `appsink` to carry `name=<name>`.
- `build_pipeline_string(spec, platform=None) -> str` (L412) — Materialises the pipeline string; emits a 2- or 3-leg `tee name=openral_cam_tee` when `enable_ros_tee` / `enable_event_tee` are set, assembling each leg via `leaky_branch` so a stalled observability / detector branch never backpressures the policy.
- `_build_event_tee_branch(spec, platform) -> str` (L650) — Returns the event leg of the `tee`: lifts NVMM to system memory, pins `format=BGR`, rate-caps via `videorate` to `event_rate_hz`, terminates in `appsink name=event_sink`.
- `_build_ros_tee_branch(spec, platform) -> str` (L634) — Returns the observability leg (system memory BGR `appsink name=ros_sink`).
- `_lift_convert(platform) -> str` (L675) — The converter a tee leg uses to lift NVMM → system-memory BGR: `nvvidconv` (Tegra) / `nvvideoconvert` (DeepStream) / `videoconvert` (else).

### `python/runner/src/openral_runner/backends/gstreamer/perception_tee.py`
_Perception event tee for `GStreamerSensorReader`. Pulls frames from the event leg's `appsink`, runs `EventDetector`s, publishes `openral_msgs/PromptStamped` on `/openral/perception/<kind>`. `rclpy` lazy-imported in `start()` so the module stays import-safe on hosts without a sourced ROS env._

- module constant `TOPIC_PREFIX: Final[str] = "/openral/perception"` (L62) — Fixed value; full topic is `f"{TOPIC_PREFIX}/{detector.kind}"`.
- `class EventDetector(Protocol)` (L75) — `kind: str`, `detect(frame_bgr, width, height, sensor_id) -> PerceptionEventMetadata | None`, `summarise(metadata) -> str`.
- `class MotionDetector` (L107) — Pure-Python frame-diff motion detector over a BGR appsink (BT.601 luma, mean abs delta). Numpy lazy-imported in `detect`. `__init__(*, threshold=0.02, downsample=1)`.
- `class SceneChangeDetector` (L215) — Grayscale-histogram scene-change detector (`chisqr_alt` distance, 32 bins). `__init__(*, threshold=0.5)`.
- `class _TokenBucket` (L302) — Per-`(sensor, kind)` rate-limit primitive; mirrors `openral_observability.failure_bus._TokenBucket` but independently implemented to keep the runner free of an observability-package dep.
- `class PerceptionEventPublisher` (L334) — Owns one event-sink appsink for one sensor; fans out to one `Publisher` per detector kind. Constructor enforces unique `kind`s, absolute `topic_prefix`, positive `rate_hz`. QoS: `BEST_EFFORT + VOLATILE + KEEP_LAST=10`. Methods: `start()`, `stop()`, `is_started` [property], `dropped_counts` [property].

### `python/runner/src/openral_runner/backends/gstreamer/tee_manager.py`
_Runtime tee-branch manager for the GStreamer perception bus. Attaches / detaches consumer branches on a running pipeline's named `tee` (`pipeline.TEE_NAME`) via dynamic pad add/remove — the mechanism the S2 reasoner drives through `ExecuteRskill`. Imports `gi` at load (requires the `gstreamer` extra)._

- `class BranchHandle` (L67) — Opaque dataclass handle to an attached branch (`name` + the private `tee` pad / branch bin); returned by `attach`, passed back to `detach`.
- `class TeeManager` (L83) — `__init__(pipeline, *, tee_name=TEE_NAME)` (raises `ROSConfigError` if the tee is absent). `branch_count` [property] (L117). `attach(elements, *, name) -> BranchHandle` (L122) — requests a tee pad, parses `LEAKY_BRANCH_QUEUE ! <elements>` into a bin, links + syncs it live; rolls back on link failure. `detach(handle)` (L83) — IDLE-probe unlink + release-pad + NULL teardown; idempotent, blocks until removed (bounded by `_DETACH_TIMEOUT_S`).

### `python/runner/src/openral_runner/backends/gstreamer/objects_detector.py`
_CPU-tier object detector for the perception event tee. Implements `EventDetector` via ONNXRuntime on system-memory BGR frames (RT-DETR / D-FINE ONNX signature). `onnxruntime` lazy-imported at construction time. Zero-copy NVMM tiers are a planned follow-up; requesting them raises `ROSConfigError`._

- `class DetectorTier(str, Enum)` (L80) — `CPU_ONNX = "cpu_onnx"`, `NVINFER = "nvinfer"`, `NVMM_AGGREGATOR = "nvmm_aggregator"`, `VLM_SIDECAR = "vlm_sidecar"`, `ZEROSHOT_HF = "zeroshot_hf"`. Execution tier for the object detector; `VLM_SIDECAR` is the out-of-process open-vocab VLM tier (2026-06-09 amendment) and `ZEROSHOT_HF` is the in-process Transformers zero-shot tier run over a fixed vocabulary (2026-06-12 amendment) — both reuse the `CPU_ONNX` BGR appsink branch.
- `select_detector_tier(platform=None) -> DetectorTier` (L125) — Probes `gst-inspect-1.0 nvinfer` (→ `NVINFER`), then checks for `Platform.TEGRA` (→ `NVMM_AGGREGATOR`), else `CPU_ONNX`. `nvinfer` probe always wins over explicit `platform`.
- `identify_rtdetr_outputs(named_shapes: list[tuple[str, tuple[Any, ...]]]) -> tuple[str, str]` (L213) — Tier-agnostic helper: from a list of `(name, shape)` output pairs, returns `(logits_name, boxes_name)`. Among 3-D outputs, the one with last-dim==4 is boxes; if both (or neither) end in 4, falls back to index order (0=logits, 1=boxes). Raises `ROSConfigError` if fewer than two 3-D outputs are present.
- `postprocess_rtdetr(logits, boxes, *, labels, model_id, sensor_id, score_threshold, frame_width, frame_height) -> ObjectsMetadata | None` (L251) — Tier-agnostic decode (CLAUDE.md §13): sigmoid→argmax→threshold, cxcywh normalised→xyxy pixels, degenerate-bbox guard, label-index bounds check (warns), sorts descending by confidence, returns `None` on zero survivors. Accepts `(N,C)`/`(1,N,C)` logits and `(N,4)`/`(1,N,4)` boxes.
- `class ObjectsDetector` (L350) — `EventDetector` implementation. `__init__(onnx_path, *, labels, model_id, input_size=(640,640), score_threshold=0.5, device="cpu")`. Delegates logits/boxes identification to `identify_rtdetr_outputs`. `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` — BGR→RGB, NN-resize, float32/255, NCHW, ORT inference, delegates postprocessing to `postprocess_rtdetr`. `summarise(metadata) -> str` — aggregates label counts as `"Nx label"` string.
- `make_objects_detector(onnx_path, *, labels, model_id, tier=None, **kwargs) -> ObjectsDetector | object` (L558) — Auto-selects tier via `select_detector_tier()` when `tier=None`; returns `ObjectsDetector` for `CPU_ONNX`; for `NVMM_AGGREGATOR` resolves the factory registered under the `openral.detector_tiers` entry-point group (ships in the private `openral-pro-trt` package — miss raises `ROSConfigError` naming it); raises `ROSConfigError` for `NVINFER` and for unknown tiers.

> **Moved to OpenRAL Pro:** the NVMM zero-copy consumers — `nvbufsurface.py`, `cuda_context.py`, `trt_nvmm.py` (`TrtNvmmExecutor`), `nvmm_detector.py` (`NvmmObjectsDetector`), `nvmm_vision_encoder.py`, `act_nvmm.py` — now live in the private `openral-pro-trt` package as `openral_pro_trt.*`; the `NVMM_AGGREGATOR` tier resolves via the `openral.detector_tiers` entry-point group. The open perception bus (`pipeline.py`, `reader.py`, `tee_manager.py`, CPU/VLM detector tiers) is unchanged.

### `python/runner/src/openral_runner/backends/gstreamer/detector_factory.py`
_gi-free dispatch seam (2026-06-09 amendment) so the manifest→detector-backend selection is unit-testable without a live pipeline. `DetectorRunner` delegates construction here. No `gi`/`onnxruntime`/`zmq`/`torch` at import; the `pytorch` branch lazy-imports `LocateAnythingDetector` and the `zeroshot_hf` branch lazy-imports `OmDetTurboDetector`._

- `weights_source_from_manifest(manifest) -> str` (L97) — Resolves the HF repo the backend loads: prefers `source_repo`, falls back to `weights_uri`, else `nvidia/LocateAnything-3B`; strips the `hf://` scheme and any `@revision` to a bare `org/name`.
- `build_manifest_detector(manifest, *, onnx_path=None, tier=None) -> tuple[Any, DetectorTier]` (L116) — Dispatches on `manifest.detector.engine` first, then `manifest.runtime`: `engine: zeroshot_hf` → `OmDetTurboDetector` (lazy import) + `DetectorTier.ZEROSHOT_HF` (no `onnx_path`); else `runtime: pytorch` → `LocateAnythingDetector` (lazy import) + `DetectorTier.VLM_SIDECAR` (no `onnx_path`); `onnx`/`tensorrt` → `make_objects_detector(onnx_path, ..., input_size=(net_h,net_w), score_threshold=...)` + the resolved tier. Raises `ROSConfigError` if the manifest is not a `kind:detector` with a detector block, or an ONNX runtime is requested without an `onnx_path`.
- `class DetectorNodeWiring` (frozen dataclass) + `detector_node_wiring(mode: DetectorMode) -> DetectorNodeWiring` — pure (rclpy-free, unit-testable) policy the perception node consumes: `continuous` → `run_continuous_leg=True, serve_on_demand=False` (publish leg, no query service); `on_demand` → `run_continuous_leg=False, serve_on_demand=True` (locate_in_view service + `detector_query` topic, no continuous publishing).

### `python/runner/src/openral_runner/backends/gstreamer/omdet_turbo_detector.py`
_In-process Transformers open-vocabulary detector (2026-06-12 amendment) — `omlab/omdet-turbo-swin-tiny-hf` (Apache-2.0). One backend serves both detector modes (the manifest's `detector.mode` declares intent): `continuous` (fixed `labels`, unprompted background producer — `omdet-turbo-indoor`) or `on_demand` (prompted locator via `set_query`/`detect_with_query` — `omdet-turbo-locator`). Same `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` interface as `ObjectsDetector`, so it reuses the CPU BGR appsink branch (`DetectorTier.ZEROSHOT_HF`). Loads under the runtime's own `transformers>=5` (no sidecar). `torch`/`transformers`/`numpy`/`PIL` lazy-imported (the `omdet` group); conversion + query parsing are pure functions (unit-testable, no GPU)._

- `build_objects_metadata_from_results(*, labels, scores, boxes_xyxy, width, height, model_id, sensor_id, score_threshold) -> ObjectsMetadata | None` (L62) — Pure (no torch): from decoded per-detection `labels`/`scores`/pixel `boxes_xyxy`, drops sub-threshold + degenerate/near-full-image (≥98%) boxes, clips + corner-orders to frame, sorts descending by confidence; `None` on zero survivors. Raises `ROSConfigError` on length mismatch.
- `query_to_classes(query) -> list[str]` (L147) — Pure: parse a free-text on-demand query into OmDet's multi-label class list (split on commas / `</c>`; a single phrase is one class; whitespace dropped). Raises `ROSConfigError` if empty.
- `class OmDetTurboDetector` (L179) — `__init__(*, labels, model_id, weights_source, score_threshold=0.3, nms_threshold=0.5, device="auto")` — stores config; model/processor load deferred to first `detect()` (lazy, side-effect-free; `device="auto"` → CUDA when available else CPU). `set_query(text)` — retarget the persistent vocabulary (the `detector_query` topic; on-demand). `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` — over the current vocabulary. `detect_with_query(frame_bgr, width, height, sensor_id, query) -> ObjectsMetadata | None` — one-shot detect for `query` WITHOUT mutating the persistent vocabulary (the read-only `locate_in_view` service). Both delegate to `_detect_classes` (BGR→RGB PIL, processor over the class list, `model(**inputs)` under `no_grad`, `post_process_grounded_object_detection`, → `build_objects_metadata_from_results`). `close()` — releases the model + `cuda.empty_cache()` if loaded on GPU; idempotent.

### `python/runner/src/openral_runner/backends/gstreamer/locateanything_detector.py`
_Open-vocabulary detector backend (2026-06-09 amendment) backed by the LocateAnything-3B sidecar. Same `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata` interface as `ObjectsDetector`, so it reuses the CPU BGR appsink branch. Connects lazily on first `detect()`; auto-spawns the sidecar (ping → `Popen` → poll → `close`) mirroring the RLDX adapter. The model runs in an isolated `transformers==4.57.1` venv (`tools/locateanything_sidecar.py`); this is the ZMQ/msgpack client. Parsing is pure-function + main-env (unit-testable, no GPU). No `zmq`/`numpy`/`PIL` at import (all lazy)._

- `parse_grounding_answer(answer, *, fallback_label="object", norm=1000) -> list[tuple[str, tuple[int,int,int,int]]]` (L53) — Parses `<ref>label</ref>` + 4-coord `<box>` tokens in document order; each box binds to the most recent `<ref>`. Coords stay normalized `[0,norm]`, corner-ordered. Drops exact duplicates and degenerate boxes (side < 2% or area ≥ 85% of the image — the repeated-box tail a looping decode emits).
- `build_objects_metadata(answer, *, width, height, model_id, sensor_id, fallback_label="object", norm=1000) -> ObjectsMetadata | None` (L95) — Scales `parse_grounding_answer` boxes into `width`×`height` pixels (clipped), builds `ObjectDetection2D` at `confidence=1.0` (grounding model — no per-box score, CLAUDE.md §1.2); `None` if no valid detections.
- `class LocateAnythingDetector` (L151) — `__init__(*, labels, model_id, weights_source="nvidia/LocateAnything-3B", host="127.0.0.1", port=5757, query=None, auto_spawn=True, boot_timeout_s=1200.0, request_timeout_s=180.0, max_side=1024, max_new_tokens=1024, mode="hybrid")` (L149) — stores config; static default `query = "</c>".join(labels)`; no connection (lazy). `set_query(text)` — runtime open-vocab override for the continuous leg. `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` — one-shot detect of the persistent query (delegates to `detect_with_query`). `detect_with_query(frame_bgr, width, height, sensor_id, query) -> ObjectsMetadata | None` — one-shot detect for `query` WITHOUT mutating the persistent query; used by the `locate_in_view` service so an on-demand reasoner query doesn't change what the continuous leg grounds. `close()` — closes the socket and terminates the sidecar if spawned; idempotent.

### `python/runner/src/openral_runner/backends/gstreamer/qwen_scene_vlm.py`

_Scene-VLM backend backed by the Qwen3.5-4B sidecar — the scene-reasoning counterpart of `LocateAnythingDetector`. Returns **text**, not `ObjectsMetadata` (a reasoning aid for task-progress / success verification, not a localizer). Same ZMQ lifecycle (lazy connect, auto-spawn, teardown only the child). No `zmq`/`numpy`/`PIL` at import (all lazy)._

- `class QwenSceneVlm` — `__init__(*, model_id, weights_source="Qwen/Qwen3.5-4B", host="127.0.0.1", port=5759, auto_spawn=True, boot_timeout_s=1200.0, request_timeout_s=180.0, max_side=1024, max_new_tokens=256)` — stores config; no connection (lazy). `query(frame_bgr, width, height, question) -> str` — encode BGR→PNG, RPC `{"op":"query",...}`, return the whitespace-stripped answer; raises `ROSConfigError` on empty question or sidecar error. `close()` — closes the socket + terminates the spawned sidecar; idempotent.
- `build_scene_vlm(manifest, *, host="127.0.0.1", port=5759) -> QwenSceneVlm` — build from a `kind:"vlm"` manifest; `model_id=manifest.name`, `weights_source` from `weights_uri` (the deployable pre-quant checkpoint) stripped of `hf://`/`@rev`. Raises `ROSConfigError` if `manifest.kind != "vlm"`. Lazy.

### `openral_runner.backends.reward` (reward monitor)

- `class Frame` (frozen dataclass) — one buffered camera frame: `stamp_ns: int`, `bgr: bytes`, `width: int`, `height: int`.
- `class RollingFrameBuffer` — `__init__(*, window_s, max_frames=256, stale_after_s=3.0)` — transport-agnostic node-side ring of recent frames (sim + real). `push(frame)` — append + evict frames older than `window_s` relative to the newest / over `max_frames`. `window(seconds) -> list[Frame]` — frames within the last `seconds` (capped to `window_s`). `is_stale(now_ns) -> bool` — True if no fresh frame within `stale_after_s`. `__len__`. Pure stdlib (no numpy/torch); unit-tested without ROS.
- `trend(series: list[float]) -> float` — least-squares slope per sample (0.0 for < 2 points); used for progress/success trend + `stalled`.
- `class RobometerInProcessReward` — `__init__(*, model_id, weights_source="OpenRAL/rskill-robometer_4b-any-general-nf4", num_bins=100, success_threshold=0.5, max_frames=8, device="cuda")` — default Robometer backend for `reward_monitor_node`: lazily imports `tools/_robometer_scorer.py::_Scorer`, meta-loads the prequantized NF4 checkpoint in the reward-monitor process, and scores BGR frames via the same native lerobot 0.6.0 `_compute_rbm_logits` + `decode_progress_outputs` path. `score(frames, task) -> (progress, success)` validates empty task/clip + frame sizes, evenly subsamples to `max_frames`, converts BGR→RGB, returns per-frame normalized arrays. `assess(...) -> dict` mirrors the reasoner contract.
- `build_reward_monitor(manifest) -> RobometerInProcessReward | TOPRewardMonitor` — build from a `kind:"reward"` manifest; dispatches `reward.backend=="topreward"` to `TOPRewardMonitor` and defaults Robometer to `RobometerInProcessReward`. Raises `ROSConfigError` if `manifest.kind != "reward"`. Lazy.
- `critic_score_from_assessment(assessment, *, threshold) -> tuple[float, float]` — Pure mapping from a reward assessment result to a generic `openral_msgs/CriticScore` `(score, threshold)`: uses `progress_now` (higher-is-better) as the score, clamped to `[0, 1]`, defaulting a missing/non-numeric/bool value to `0.0`. Lets `reward_monitor_node` feed the Tier-C critic producer. Pure, ROS-free, unit-tested.

### `python/runner/src/openral_runner/backends/gstreamer/detector_runner.py`
_Runtime glue that wires a ``kind: detector`` rSkill to a live camera pipeline — loads the `DetectorContract`, delegates backend construction to `build_manifest_detector` (ONNX CPU/NVMM tiers or the `VLM_SIDECAR` open-vocab tier), attaches the appropriate branch to the bus tee via `TeeManager`, and fires the `on_detection` callback for each non-`None` `ObjectsMetadata`. Imports `gi` + `DetectorTier`/`build_manifest_detector` + `nvmm_convert_element` eagerly at load._

- `class DetectorRunner` (L60) — `__init__(pipeline, manifest, *, onnx_path=None, sensor_id, on_detection, tee_name=TEE_NAME, tier=None)` (L101) — validates `manifest.kind == "detector"` + `manifest.detector is not None` (raises `ROSConfigError`); caches `_net_w`/`_net_h` from `DetectorContract.input_size` for the NVMM caps; delegates to `build_manifest_detector(manifest, onnx_path=onnx_path, tier=tier)` → `(detector, tier)` (gi-free dispatch; `onnx_path` optional, `None` for the VLM sidecar tier); creates `TeeManager`. `start()` (L178) — selects branch string + handler by tier: NVMM_AGGREGATOR resolves the platform's NVMM converter (`nvvideoconvert`/`nvvidconv`) via `nvmm_convert_element()` (raises `ROSConfigError` if neither registered) and attaches the NVMM RGBA appsink + `_on_sample_nvmm`; every other tier (CPU_ONNX, VLM_SIDECAR, ZEROSHOT_HF) attaches `videoconvert ! video/x-raw,format=BGR ! appsink` + `_on_sample_bgr`; raises `ROSRuntimeError` if appsink not found after attach. `_on_sample_bgr(appsink) -> int` (L242) — pulls BGR sample, format assert, buffer.map/unmap, calls `detector.detect`, fires `on_detection` on non-`None`; errors guarded. `_on_sample_nvmm(appsink) -> int` (L301) — pulls NVMM sample, `wrap_buffer` (lazy import from the private `openral_pro_trt.nvbufsurface`), calls the entry-point-resolved NVMM detector's `detect_nvmm`, fires `on_detection`; always unmaps; errors guarded. `stop()` (L59) — disconnects signal + detaches branch + calls `detector.close()` if present; idempotent.

### `python/runner/src/openral_runner/__init__.py`
_Public surface of the inference runner. Imports are PEP 562 lazy (M8 PR I/8): heavy symbols (`InferenceRunnerBase`, `factory.*`, `DeployRunner`, `safety.*`) are resolved on first attribute access so importing any subpackage does not eagerly drag in torch (582 modules) or trigger downstream glib conflicts._

- light eager imports: `precise_sleep`, `sleep_until`, `InferenceRunner` (Protocol), `SensorReader` (Protocol).
- `_LAZY_ATTRS: dict[str, tuple[str, str]]` — `attr → (module, name)` map driving the `__getattr__` resolver. (L78)
- `__getattr__(name) -> Any` — Resolves heavy symbols on first access (torch / glib-sensitive deferral). (L92)

### `python/runner/src/openral_runner/factory.py`
_Library deploy runner used by runtime nodes; the public deploy CLI now shells the ROS graph from a `DeployScene`._

- `SKILL_REGISTRY: dict[str, Callable[[dict[str, object]], rSkillBase]]` — `vla.id` → skill factory. Today: `hello`, `gpu_passthrough` (M8 PR I/10). (L84)
- `SENSOR_BACKEND_REGISTRY: dict[str, Callable[[SensorReaderConfig], SensorReader]]` — `backend` id → reader factory. Today: `opencv_thread`, `gstreamer`. (L246)
- `_to_int(value, *, field, sensor_id) -> int` — YAML `object` → `int` coercion helper used across factories; rejects bools explicitly. (L40)
- `_make_gpu_passthrough_skill(extra) -> rSkillBase` — Builds `GpuPassthroughSkill`; recognised `extra`: `sensor_id` (default `"wrist_rgb"`), `n_joints`, `horizon`, `device` (default `"cuda"`, raises if unavailable). (L61)
- `_make_opencv_thread_reader(cfg) -> SensorReader` — Builds `OpenCVThreadSensorReader` from a `SensorReaderConfig`; requires `backend_params.device`. (L90)
- `_make_gstreamer_reader(cfg) -> SensorReader` — Builds `GStreamerSensorReader` from a `SensorReaderConfig`. Translates `publish_to_ros` / `publish_topic` / `publish_rate_hz` → `PipelineSpec.enable_ros_tee`. (M8 PR I/2 + I/4.) (L130)

### `python/runner/src/openral_runner/deploy_runner.py`
_:class:`DeployRunner` — concrete `InferenceRunnerBase` subclass composing HAL + Skill + WorldStateAggregator + SensorReaders + SafetyClient._

- `class DeployRunner(InferenceRunnerBase)` — First end-to-end closer of the `WorldState → Skill → safety → HAL` loop on real hardware / digital twins. The runner is the safety-supervisor boundary per CLAUDE.md §10: catches `ROSSafetyViolation` from the SafetyClient, records it on the `TickResult`, withholds the `HAL.send_action` call (does not re-raise because withholding IS the mitigation today). (L77)
  - `__init__(*, hal, skill, aggregator, sensor_readers=(), safety_client=None, recorder=None, **base_kwargs)` — Caller must pre-`configure()`+`activate()` the skill; runner manages HAL + reader open/close. Defaults `safety_client` to `NullSafetyClient`. Dashboard JPEG thumbnails are emitted at a private fixed cadence. Optional `recorder` is a `openral_dataset.RolloutRecorder`; when set, `episode_start` / `episode_end` drive its lifecycle and every tick fans out via `record_frame`. (L119)
  - `episode_start(task_string: str) -> int` — Open a new episode on the attached recorder; returns the new `episode_idx` (or `-1` when no recorder is attached). Raises `RuntimeError` if called twice without `episode_end`. (L178)
  - `episode_end(*, success: bool) -> None` — Close the current recorder episode with the success flag. No-op when no recorder is attached. Raises `RuntimeError` if called without `episode_start`. (L206)
  - `activate() -> None` — `super().activate()` + `hal.connect()` + open every `SensorReader`. (L232)
  - `deactivate() -> None` — Close every `SensorReader` (best-effort; logs + continues), `hal.disconnect()`, `super().deactivate()`. (L250)
  - `_tick_impl(tick_idx) -> TickResult` — Five-phase tick: sensors → world_state → inference → safety → hal. Per-phase `*_ms` populated on the `TickResult`; `InferenceRunnerBase.tick` lifts them onto the `rskill.tick` OTel parent span. Each sensor `read_latest` call is wrapped in a `sensors.read_latest` span that records `openral.sensors.age_ms` (frame age at read time) onto the `openral.sensors.age_ms` histogram. Wraps `HAL.read_state` in a `hal.read_state` span and `HAL.send_action` in a `hal.send_action` span (labels: `openral.hal.adapter`, `openral.hal.robot.model`, `openral.hal.control_mode`); records `openral.hal.read_state.duration` + `openral.hal.send_action.duration` histograms keyed by adapter. Catches `ROSPerceptionStale` per reader and emits `openral.event.sensor_stale` + `openral.sensors.stale_reads` counter. Catches `ROSSafetyViolation` at the supervisor boundary and emits `openral.event.safety_violation` + `record_exception` + `openral.safety.violations` counter (labeled by exception type and severity). (L321)
  - `_tracer` [@property] — Per-call `trace.get_tracer("openral")` (never cached at `__init__`, would bind to the provider live at construction time). (L223)
  - `_hal_adapter_label` — Lower-cased class name of the HAL adapter, used as the closed-set `openral.hal.adapter` value on spans + metrics. (L162)

### `python/runner/src/openral_runner/safety.py`
_:class:`SafetyClient` stub — Python-side seam for the future C++ safety kernel (CLAUDE.md §6 Layer 6)._

- `class SafetyClient(Protocol)` — `@runtime_checkable` Protocol. `check_action(action)` returns `None` to allow or raises `ROSSafetyViolation` to reject. The inference runner catches at its supervisor boundary; never silently caught per CLAUDE.md §10. (L48)
  - attr `envelope: SafetyEnvelope` — the envelope checked against.
  - `check_action(action: Action) -> None` (L64)
- `class NullSafetyClient` — no-op stub that always allows. Every call opens a `safety.check` OTel span at `severity="info"` carrying `control_mode`, `horizon`, `envelope_max_ee_speed_m_s`, `envelope_max_force_n`. Used by digital-twin runs and pre-hardware tests so traces show the seam is wired before the C++ kernel arrives. (L80)
  - `__init__(envelope: SafetyEnvelope | None = None)` — defaults to a stock `SafetyEnvelope`. (L107)
  - `check_action(action: Action) -> None` (L111)

### `python/runner/src/openral_runner/base.py`
_Shared base for inference runners. Subclasses override `_tick_impl`._

- `_percentile(samples: list[float], q: float) -> float` — Linear-interpolation percentile (`0.0` for empty list). Used by `_build_run_result`. (L39)
- `class InferenceRunnerBase(ABC)` — Owns the rate-limited loop, `rskill.tick` OTel parent span, `RunResult` aggregation, deadline-overrun policy. (L57)
  - `__init__(*, rate_hz=30.0, deadline_overrun_policy=WARN, runner_name="inference_runner", latency_budget_ms=None, save_dir=None)` — Reject `rate_hz <= 0`. (L93)
  - `activate() -> None` — Reset tick counter; mark active. (L115)
  - `deactivate() -> None` — Stop ticking; idempotent. (L120)
  - `_tick_impl(tick_idx: int) -> TickResult` [@abstractmethod] — Subclass hook; the base wraps it in a `rskill.tick` span. (L127)
  - `episode_start(task_string: str) -> int` — Optional explicit episode boundary; default raises `NotImplementedError`. `DeployRunner` overrides to drive the recorder; `SimRunner` overrides as a no-op (sim derives episode boundaries from `env.step` flags). (L164)
  - `episode_end(*, success: bool) -> None` — Optional explicit episode boundary; default raises `NotImplementedError`. See `episode_start`. (L189)
  - `_should_terminate() -> bool` — Subclass early-exit hook (default False) consulted after each tick inside `run()`. `SimRunner` overrides to stop once `n_episodes` complete. (L141)
  - `tick() -> TickResult` — Span-wrapped single-tick entry; attaches per-stage timings as `skill.{tick_ms, inference_ms, sensors_ms, world_state_ms, safety_ms, hal_ms, action_applied, safety_violations}` attributes plus sim-only `skill.{step_idx, episode_idx, reward, terminated, truncated}` when set, plus `openral.tick.idx`. Records `openral.tick.duration` / `openral.inference.duration` histograms (label: `skill.id`) and increments `openral.safety.violations{check_name="runtime", severity="violation"}` for each violation on the tick. (L208)
  - `run(max_ticks: int | None = None) -> RunResult` — Rate-limited loop using `sleep_until`. Applies `DeadlineOverrunPolicy` (`warn` / `drop` / `raise`). Records `latency_budget_ms` violations and increments `openral.tick.budget_violations` per violation. Honors `_should_terminate()` after each tick. (L293)
  - `_current_trace_id() -> str | None` [@staticmethod] — Active OTel trace id (hex) or None. (L354)
  - `_on_deadline_overrun(result: TickResult) -> None` — Apply policy: structlog warn / drop / raise `ROSDeadlineMissed`. Always increments `openral.tick.deadline_misses` and emits `openral.event.deadline_missed` on the current parent span; on `RAISE`, also calls `record_exception` + `set_status(ERROR)` on the parent span before re-raising. (L364)
  - `_build_run_result(results, *, budget_violations, trace_id) -> RunResult` — Aggregate per-tick records into `RunResult` (mean / p99). (L417)
