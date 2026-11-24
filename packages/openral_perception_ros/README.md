# `openral_perception_ros` (ROS 2)

Perception nodes that sit on plain ROS image topics (no GStreamer): an object-detection
producer feeding the spatial-memory object lift, a monocular depth provider, a SAM 2.1
segmenter for the vision attachment leg, and two on-demand VLM / reward query services.
An `ament_cmake` package whose nodes are Python programs.

The detector (`ros_image_detector_node`) subscribes a camera `sensor_msgs/Image`, runs the
GStreamer-free
[`ObjectsDetector`](../../python/runner/src/openral_runner/backends/gstreamer/objects_detector.py)
(or a manifest-driven backend) and publishes detections on `/openral/perception/objects`, the
topic the world-state lifecycle node lifts to 3D. It is the deploy-sim counterpart of the
on-robot GStreamer perception bus: same `ObjectsDetector` and `ObjectsMetadata` schema, only
the frame source differs. The deploy default backend is the open-vocabulary
`omdet-turbo-indoor` detector; it falls back to the fixed-label RT-DETR COCO ONNX
(`rtdetr-coco-r18`) when the omdet deps are not installed.

| Node (executable) | ROS node name | Kind | Serves / publishes |
| --- | --- | --- | --- |
| `ros_image_detector_node.py` | `openral_ros_image_detector` | lifecycle | `/openral/perception/objects`, `LocateInView` service |
| `depth_provider_node.py` | `openral_depth_provider` / `openral_estimated_depth_provider` | lifecycle | depth + `CameraInfo` (+ cloud in metric mode) |
| `segmenter_node.py` | `openral_segmenter` | lifecycle | `/openral/perception/segment_in_view` |
| `scene_vlm_node.py` | `openral_scene_vlm` | plain node | `/openral/perception/query_scene` |
| `reward_monitor_node.py` | `openral_reward_monitor` | plain node | `/openral/perception/query_task_progress` |

## Detector node

`ros_image_detector_node` (`RosImageObjectDetectorNode`). Best-effort producer: an Image it can't
convert (unsupported encoding, a `step` shorter than a row) is logged at
WARNING (throttled, naming camera and encoding) and skipped, as is a
detector error — it never crashes the graph.

The detection image comes from a high-resolution RGB camera (e.g.
`agentview_left`), but detections are attributed to `sensor_id` (default
`front_depth`), the co-located depth camera whose REP-103 optical frame
the world-state lift projects through. They share the sim viewpoint, so the geometry
is consistent.

### Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `cameras` | string[] | `[""]` | `"id=topic"` entries; empty = one camera `primary_camera` on `image_topic`. |
| `primary_camera` | string | `default` | Id of the continuously-detected camera. |
| `image_topic` | string | `""` | Single-camera fallback `sensor_msgs/Image` — `openral_core.camera_topic(<name>)`, passed by `deploy_e2e`. With no `cameras` entries either, configure raises `ROSConfigError` (ADR-0108). |
| `output_topic` | string | `/openral/perception/objects` | Perception topic to publish on. |
| `sensor_id` | string | `front_depth` | Sensor name stamped on the metadata + `header.frame_id`. |
| `camera_infos` | string[] | `[""]` | `"id=topic"` driver `CameraInfo` per camera. Once one arrives, every batch for that camera carries its `header.frame_id` + K (`ObjectsMetadata.camera_frame_id` / `camera_intrinsics`), which the world-state lift projects through. Without an entry the node warns at configure and the lift falls back to the manifest `SensorSpec` (logged once). `deploy_e2e` derives it (`_live_camera_info_topic`): the sim bridge's topic, or on real the driver's beside the unit overlay's `ros2_image` binding (Thor `top` → `/zed/zed_node/rgb/color/rect/camera_info`). |
| `manifest_path` | string | `""` | Detector rSkill manifest; when set the backend is manifest-driven, else the legacy RT-DETR ONNX path below. |
| `onnx_path` | string | `""` | RT-DETR ONNX model path (legacy path, or the ONNX for a manifest backend). |
| `model_id` | string | `rtdetr-coco-r18` | Id embedded in `ObjectsMetadata`. |
| `score_threshold` | double | `0.5` | Minimum sigmoid score. |
| `input_size` | int | `640` | Square model input edge. |
| `max_rate_hz` | double | `5.0` | Publish-rate cap. |
| `labels` | string[] | `[""]` | COCO-80 class names indexed by class-id (legacy ONNX path). |
| `query` | string | `""` | Initial open-vocab query override (VLM backends only). |
| `query_topic` | string | `/openral/perception/detector_query` | GStreamer perception-bus open-vocab retarget topic (on-demand). Namespaced per on-demand locator. |
| `locate_in_view_service` | string | `/openral/perception/locate_in_view` | On-demand-locator service name. The deploy launch sets it to `/openral/perception/<alias>/locate_in_view` per on-demand locator so several locators co-exist. |
| `detector_id` | string | `""` | This locator's alias, echoed in the `LocateInView` response so the reasoner records which model answered. |

### Topics

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `image_topic` (configurable) | BEST_EFFORT / VOLATILE / KEEP_LAST=1 | `sensor_msgs/Image` (`bgr8`/`rgb8`/`bgra8`/`rgba8`/`mono8`; padded rows OK) |
| Sub | each `camera_infos` topic | BEST_EFFORT / VOLATILE / KEEP_LAST=1 | `sensor_msgs/CameraInfo` |
| Pub | `/openral/perception/objects` (configurable) | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `openral_msgs/PromptStamped` (`metadata_json` = `ObjectsMetadata`; `header.frame_id` = `sensor_id`) |

## Launch

Wired into the generic `deploy_e2e.launch.py` graph behind the
`enable_object_detector` launch argument, driven by the CLI:

```bash
openral deploy sim --config scenes/deploy/<scene>.yaml   # detector ON by default
```

`openral deploy sim` brings the detector up **by default** (pass
`--no-object-detector` to turn the leg off). The default backend is the open-vocab
`omdet-turbo-indoor` manifest, falling back to the RT-DETR COCO ONNX
(`rskills/rtdetr-coco-r18/model.onnx`, gitignored; export it with
`tools/export_rtdetr_onnx.py`) when the omdet deps are absent. The leg downgrades to off
only when neither backend is available. Use `--object-detector-manifest PATH` to pick a specific detector
rSkill or `--object-detector-onnx PATH` to force the fixed-label RT-DETR path.
The launch forwards `onnx_path`/`manifest_path`, `labels`, `model_id`,
`input_size`, and the topics as ROS parameters on the node, and throttles by the
manifest's `DetectorEngine` (`vlm_sidecar` 0.5 Hz, `zeroshot_hf` 2 Hz, ONNX 5 Hz).

**On-demand locators.** Alongside the continuous detector, the deploy
launch can bring up one or more `mode: on_demand` open-vocab locators — each as
its **own** lifecycle node serving a namespaced
`/openral/perception/<alias>/locate_in_view`, so the reasoner picks a model via
`LocateInViewTool.detector`. Default = `omdet-turbo-locator` (when the omdet deps
import); add more with the repeatable `--object-detector-locator <manifest|alias>`.
LocateAnything is opt-in (NVIDIA non-commercial, 5 GB, needs the sidecar venv); so is
`open-vision-banana-4b`, a referring-segmentation locator (FLUX non-commercial, one
diffusion step per query, throttled to 0.3 Hz). Each locator is an independent lifecycle
and VRAM peer, so the reasoner can evict it before a co-resident VLA. Deploy starts
locators with a bounded one-shot state-poll helper that exits at ACTIVE, so a dropped
configure response cannot leave grounding unavailable; a later reasoner deactivation
still releases VRAM.

The `LocateInView` service follows the source-frame and failure contracts in
[`openral_msgs`](../msgs/README.md#services--srv): an empty `failure_reason` means the
detector completed the query, even when it found no detections. `detector_inactive`,
`no_frame`, `requested_frame_unavailable`, and bounded
`detector_exception:<type>:<detail>` values mean the query did not establish visibility;
clients must not treat those responses as an absence result. Zero `source_stamp_ns` asks
for the latest frame for reasoner tools. Nonzero requests select only that exact image
stamp from each camera's bounded 32-frame history; the response reports the actual source
stamp, and an evicted/unseen stamp returns no detections with a typed failure.

## What's in here

| Path | Role |
| --- | --- |
| `openral_perception_ros/ros_image_detector_node.py` | The node + its `main()` entry point. ROS imports are deferred into `main()` so the module stays import-safe on hosts without a sourced ROS env. |
| `openral_perception_ros/image_convert.py` | `image_to_bgr_bytes(msg)` — the one `sensor_msgs/Image` → contiguous BGR bytes converter every node here uses (no `cv_bridge`). Accepts `bgr8`/`8UC3`, `rgb8`, `bgra8`/`rgba8` (alpha dropped; the ZED wrapper's default layout), `mono8`/`8UC1`; honours `step`, so pitch-padded rows work. Raises `ImageConvertError` on anything else; every caller logs that at WARNING (throttled), never DEBUG-only. |
| `openral_perception_ros/depth_convert.py` | `depth_array_to_image_msg` / `image_msg_to_depth_array` (`32FC1` metres ↔ ndarray, NaN-preserving) + `camera_info_from_intrinsics` (pinhole `CameraInfo`). Pure message-boundary helpers, no torch. Unit-tested in `tests/unit/test_depth_convert.py`. |
| `openral_perception_ros/depth_provider_node.py` | `depth_provider_node` (`openral_depth_provider`): subscribes a mono RGB stream and calls the sidecar of a `kind: depth_estimator` rSkill (`manifest_path`; booted by `tools/depth_estimator_sidecar.py`) over ZMQ. **SLAM mode** (DA3-SMALL, `predicts_intrinsics`): republishes `32FC1` depth + the model's `CameraInfo` for **nvblox** (`openral_slam_bringup/nvblox.launch.py`). **Metric mode** (`rgb_camera_info_topic`, `DeployRuntime.estimated_depth_camera`; DA3METRIC by default, Depth Pro 2 via `estimated_depth_rskill`): rectifies the frame, converts by the skill's `DepthEstimatorContract` with the camera's own focal (`openral_rskill.depth_estimator.depth_skill`, through the rSkill license gate, so research-only Depth Pro 2 needs `OPENRAL_ALLOW_NONCOMMERCIAL=1`; `metres_from_output`), publishes depth + `CameraInfo` + the back-projected cloud. Drops any reply from another model than the skill's. A managed **lifecycle node**: `configure` checks the skill and opens the socket and topics, only ACTIVE calls the sidecar and publishes, `deactivate` stops both (the sidecar keeps its weights), `cleanup` closes the socket. `deploy_e2e` autostarts it ACTIVE; the estimated-depth instance (`openral_estimated_depth_provider`) is a reasoner lifecycle peer the LLM can deactivate / activate (no cloud while inactive: the kernel's world-voxel deadline stops motion); the SLAM instance (`openral_depth_provider`) is not, since cuVSLAM / nvblox must never lose depth. With `runtime.enable_octomap: true` and nothing else giving octomap a cloud, the deploy picks the estimated-depth camera itself (`auto_estimated_depth_camera`). |
| `openral_perception_ros/lens_rectify.py` | Lens → exact pinhole for metric mode (OpenCV): `build_rectifier` / `PinholeRectifier` (remap built once per calibration; valid pixels only, pixel aspect kept, stays in the optical frame), `distortion_problem`, `rectified_stream_k`. |
| `openral_perception_ros/metric_depth.py` | Pure numpy half of metric mode: `metres_from_output` (per `DepthEstimatorContract.output`: `metric` passes through, `canonical_focal` is `focal * out / canonical_focal_px` with the focal at the output raster, `relative` refused), `camera_info_problem`, `scale_intrinsics`, `points_from_depth`. |
| `openral_perception_ros/scene_vlm_node.py` | `scene_vlm_node`: serves `/openral/perception/query_scene` (`openral_msgs/srv/QueryScene`) from a `kind: "vlm"` rSkill in the out-of-process sidecar, answering free-text questions about a cached camera frame. On-demand only; publishes nothing. |
| `openral_perception_ros/reward_monitor_node.py` | `reward_monitor_node`: buffers recent camera frames and serves `/openral/perception/query_task_progress` (`openral_msgs/srv/QueryTaskProgress`) from a `kind: "reward"` rSkill. Advisory only. |
| `openral_perception_ros/segmenter_node.py` | `segmenter_node` (`SegmenterNode`, `openral_segmenter`): a managed lifecycle node serving `/openral/perception/segment_in_view` (`openral_msgs/srv/SegmentInView`) from an in-process SAM 2.1 hiera-small (`kind: segmenter` rSkill). See [Segmenter node](#segmenter-node) below. |
| `openral_perception_ros/camera_topics.py` | `parse_camera_entries(entries)` (the bare `"id=topic"` parse, no fallback) and `resolve_camera_topics(entries, *, primary_camera, image_topic)` — the one implementation of the `"id=topic"` → ordered map resolution every node here takes as parameters. Pure, ROS-free, unit-tested in `tests/unit/test_perception_camera_topics.py`. |
| `package.xml` / `CMakeLists.txt` | `ament_cmake` manifest (depends on `rclpy`, `sensor_msgs`, `openral_msgs`, plus `geometry_msgs` + `tf2_ros` for the segmenter's prompt transform; exec-depends on `python3-openral-runner` and `python3-openral-hal`, the latter for `intrinsics_from_camera_info`) + `ament_python_install_package` and `install(PROGRAMS … ros_image_detector_node.py)`. Installed as a program (not a setuptools `console_scripts` entry) so its `#!/usr/bin/env python3` shebang survives — a `console_scripts` entry is regenerated by colcon with a system-python shebang that can't see the `openral_runner` workspace package. Launch executable: `ros_image_detector_node.py`. |

## Segmenter node

`segmenter_node` (`SegmenterNode`, `openral_segmenter`) answers a **geometric**
question, where `ros_image_detector_node` answers a semantic one: *"which pixels
of camera Y's current view belong to the thing under **this 3-D point**?"* It
serves `/openral/perception/segment_in_view` (`openral_msgs/srv/SegmentInView`)
from an in-process SAM 2.1 hiera-small — the
`rskills/rskill-sam2_1-any-grasped_object_mask-bf16` rSkill — and its one caller
is the HAL's vision attachment-evidence bridge
(`openral_hal.vision_attachment_bridge`), at attach / detach / regrasp events.
One shot per event, never per frame.

It lives here, beside the detector rSkills, because **the HAL is kept
torch-free**: it asks over the service rather than importing a model.

Load-bearing parts of the contract:

- **The prompt is a 3-D point, not a pixel.** The node transforms it from the
  request's `frame_id` into the **cached image's `header.frame_id`** — the
  optical frame the camera driver declares — through tf2, and projects it with
  the driver's **live `CameraInfo` K** when `camera_infos` names one for that
  camera, rescaled to the frame actually received (`projection_geometry`). The
  manifest's `SensorSpec.frame_id` and `SensorSpec.intrinsics` are fallbacks
  only, each logged once at warning: on the OpenArm `head_zed` names the ZED
  *body* frame while the driver stamps its images in the optical frame, and the
  declared intrinsics are nominal (fx 960 declared vs 1498.18 published). A camera listed
  in `camera_infos` is refused (`ROSPerceptionStale`) until its `CameraInfo`
  arrives, never silently projected through the nominal K. Reply masks carry
  the segmented image's own header (frame + capture stamp). Intrinsics never
  leak across the service boundary, and nothing here hardcodes a pixel or a
  frame id.
- **The reply is plural.** With `segmenter.multimask` the model emits three
  nested hypotheses (subpart / part / whole) and only geometry can pick between
  them — this node has no depth, so it returns every candidate that cleared
  `min_mask_area_px`, **area ascending**, and the HAL selects against the wrist
  depth frame it already holds.
- **`mask_scores_advisory` is advisory, and named to say so.** A mis-aimed point
  prompt was measured returning a mask covering 59.8% of a real frame at this
  model's *top* score of 0.977. Nothing may threshold on it or rank by it.
- **The handler never raises.** The caller is holding an action-acknowledgement
  barrier open, so every failure is a typed `failure_reason` with empty `masks`,
  which the caller turns into a conservative attachment rather than a stall.

Lifecycle: cameras, tf2, subscriptions and the service live for the
configured→cleanup span; the model is built **and warmed** on `on_activate`
(the first forward pass is ~742 ms against ~53 ms warmed, and only the warmed
figure fits inside the HAL's ~100 ms barrier) and released on `on_deactivate`.
Activate also burns the service handler's own lazy imports (`openral_hal`'s package
`__init__` loads every HAL, which alone cost the first call about a second against the
HAL's 0.25 s segmentation deadline). With that done, the first call is ~0.10-0.15 s end
to end at 1920x1080 with three prompt points; the activate log line reports the total
warm-up time.

### The diagnostic mask topic

A service reply cannot be watched with `ros2 topic echo`, and the dashboard's
camera tiles render segmentation masks.
`publish_debug_masks` (**default false**) re-publishes each successful reply
as an `openral_msgs/SegmentMasks` on `debug_masks_topic`
(`/openral/perception/masks`), at sensor-class QoS — `BEST_EFFORT` / `VOLATILE`
/ `KEEP_LAST=1`, because only the newest mask set is worth drawing. Off means
*no publisher is created at all*, so it costs nothing when nobody asked for it.
`openral_observability.dashboard.perception_overlay_subscriber` is the consumer.

It is **strictly diagnostic**; keep it that way:

- nothing in the safety or attachment path reads it — the HAL producer takes its
  masks from its own service reply and gates them geometrically;
- it is published *after* the response is complete, so it cannot delay a caller
  that is holding an action-acknowledgement barrier open;
- a publish failure is logged and swallowed: a display path must never be able
  to degrade a grasp.

### Segmenter launch

`deploy_e2e.launch.py` composes it as the `openral_segmenter` lifecycle node
(executable `segmenter_node.py`, autostarted to ACTIVE) only when the deploy
scene enables the vision attachment leg, **off by default**:

```yaml
runtime:
  vision_attachment:
    enabled: false          # off by default
    camera: head_zed        # manifest sensor name
    rgb_topic: /zed/zed_node/rgb/color/rect/image
    rgb_camera_info_topic: /zed/zed_node/rgb/color/rect/camera_info
    depth_topic: /zed/zed_node/depth/depth_registered
    depth_camera_info_topic: /zed/zed_node/depth/camera_info
```

`openral deploy` forwards it as `enable_vision_attachment`, `vision_attachment_camera`,
`vision_attachment_rgb_topic`, `vision_attachment_rgb_camera_info_topic`,
`vision_attachment_segmenter_manifest` and `vision_attachment_segmenter_device`; the
launch sets `cameras` / `camera_infos` to `<camera>=<topic>`, `primary_camera`,
`robot_yaml`, `manifest_path`, `device` and `use_sim_time`. The depth pair goes to
the HAL's `vision_attachment_*` parameters, not here. Enabled, the launch **always**
turns the safety kernel's attached-payload check on too (1000 ms deadline on real):
the leg never runs alone, because the octomap bridge clears and the self-filter
removes a published payload whatever the kernel flag.

### Segmenter parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `cameras` | string[] | `[""]` | `"id=topic"` entries. Each id MUST be a `SensorSpec` name in `robot_yaml` — that is where its fallback intrinsics and frame come from. |
| `camera_infos` | string[] | `[""]` | Optional `"id=topic"` entries naming each camera's `sensor_msgs/CameraInfo` (sensor-data QoS). Listed → the driver's live K is used and a request waits for it; unlisted → the manifest's nominal intrinsics, warned once. Each id must also be in `cameras`, else configure fails. |
| `primary_camera` | string | `default` | Camera used when a request leaves `camera` empty. |
| `image_topic` | string | `""` | Single-camera fallback topic; with no `cameras` entries either, configure raises `ROSConfigError` (ADR-0108). |
| `robot_yaml` | string | — (required) | `RobotDescription` path, for camera intrinsics + optical frame. |
| `manifest_path` | string | — (required) | `kind: segmenter` rSkill manifest. |
| `segment_in_view_service` | string | `/openral/perception/segment_in_view` | Service name. |
| `publish_debug_masks` | bool | `false` | Re-publish each successful reply on `debug_masks_topic` for the dashboard's mask overlay. Diagnostic only; off creates no publisher. |
| `debug_masks_topic` | string | `/openral/perception/masks` | Topic for the above (`openral_msgs/SegmentMasks`). |
| `max_frame_skew_s` | double | `0.1` | A request with a non-zero stamp is served from the cached frame nearest that stamp (8 frames cached per camera) and refused (`ROSPerceptionStale`) when none is within this skew. A zero stamp is served from the newest frame. |
| `frame_wait_s` | double | `0.2` | When the newest cached frame is older than the request's stamp, wait up to this long for a frame at or after it. The wait is also capped by `request_deadline_s` minus the time already spent. |
| `request_deadline_s` | double | `0.25` | The requester's deadline per call (the HAL's `vision_attachment_deadline_s`). A reply that still lands late is dropped by the HAL, which falls back. |
| `camera_request_deadlines` | string[] | `[""]` | `"camera=seconds"`: per-camera frame wait and deadline, for requesters that allow more than the HAL's 0.25 s (the object tracker's cameras). |
| `camera_max_frame_skews` | string[] | `[""]` | `"camera=seconds"`: per-camera `max_frame_skew_s`; a 1080p BEST_EFFORT stream loses frames under load. |
| `device` | string | `auto` | `auto` picks CUDA when available; set `cpu` on a pre-sm_70 GPU, where the CUDA wheels ship no kernels for the card. |

## Tests

Unit tests live in `tests/unit/` (`test_perception_camera_topics.py`, `test_image_convert.py`,
`test_depth_convert.py`, `test_estimated_depth_camera.py`, `test_segmenter_projection_geometry.py`,
`test_objects_detector.py`, `test_rtdetr_onnx_detects.py`). Live-ROS tests
(`OPENRAL_TEST_ROS_LIVE=1`, listed in `scripts/ros_live_tests.sh`) are
`tests/integration/test_segment_in_view_service.py` (the real segmenter node, rSkill manifest, SO-101
manifest and static TF: plural area-ascending masks, the typed failure branches, the diagnostic
topic) and `test_estimated_depth_provider_live.py` (metric mode against a fake DA3 sidecar,
deactivate stops the cloud, reactivate resumes it).

## Related

- `packages/world_state/`: subscribes `/openral/perception/objects` and lifts each 2D
  detection to a 3D `map`-frame centre.
- `openral_core.ObjectsMetadata` and `ObjectsDetector`: the schema and detector reused here.
- CLAUDE.md §3 (layer discipline) and §2 (QoS).
