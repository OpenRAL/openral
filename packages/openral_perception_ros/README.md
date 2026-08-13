# `openral_perception_ros` (ROS 2)

> **Standalone ROS-Image object-detection producer (no
> GStreamer) feeding the perception → spatial-memory object lift.**

A single `ament_cmake` ROS 2 package (Python node) that subscribes a
camera `sensor_msgs/Image`, runs the GStreamer-free
[`openral_runner.backends.gstreamer.objects_detector.ObjectsDetector`](../../python/runner/src/openral_runner/backends/gstreamer/objects_detector.py)
(or a manifest-driven backend — see below), and publishes the
detections on `/openral/perception/objects` — the topic the world-state
lifecycle node lifts to 3D in deploy-sim. The deploy-sim **default** backend is
the open-vocabulary `omdet-turbo-indoor` continuous detector (grounds arbitrary
indoor/kitchen objects); it falls back to the fixed-label RT-DETR COCO ONNX
(`rtdetr-coco-r18`) when the omdet deps are not installed.

It exists so the perception → spatial-memory object lift can
run against a plain ROS image topic in `openral deploy sim`, without
standing up the GStreamer perception bus (the supervisor-graph's F6 leg) that
the on-robot path uses. It reuses the exact same `ObjectsDetector` and
`ObjectsMetadata` schema; only the frame source differs.

## Node

`ros_image_detector_node` (`RosImageObjectDetectorNode`,
`openral_ros_image_detector`). Best-effort producer: an Image it can't
convert (unsupported encoding / padded rows) or a detector error is
logged at debug and skipped — it never crashes the graph.

The detection image comes from a high-resolution RGB camera (e.g.
`agentview_left`), but detections are attributed to `sensor_id` (default
`front_depth`) — the co-located depth camera whose REP-103 optical frame
the world-state lift projects through. They share the MuJoCo viewpoint,
so the geometry is consistent (per the object-lift design).

### Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `image_topic` | string | `/openral/cameras/agentview_left/image` | Camera `sensor_msgs/Image` to detect on. |
| `output_topic` | string | `/openral/perception/objects` | Perception topic to publish on. |
| `sensor_id` | string | `front_depth` | Sensor name stamped on the metadata + `header.frame_id`. |
| `onnx_path` | string | — (required) | RT-DETR ONNX model path. |
| `model_id` | string | `rtdetr-coco-r18` | Id embedded in `ObjectsMetadata`. |
| `score_threshold` | double | `0.3` | Minimum sigmoid score. |
| `input_size` | int | `640` | Square model input edge. |
| `max_rate_hz` | double | `5.0` | Publish-rate cap. |
| `labels` | string[] | — (required) | COCO-80 class names indexed by class-id. |
| `query_topic` | string | `/openral/perception/detector_query` | GStreamer perception-bus open-vocab retarget topic (on-demand). Namespaced per on-demand locator. |
| `locate_in_view_service` | string | `/openral/perception/locate_in_view` | On-demand-locator service name. The deploy launch sets it to `/openral/perception/<alias>/locate_in_view` per on-demand locator so several locators co-exist. |
| `detector_id` | string | `""` | This locator's alias, echoed in the `LocateInView` response so the reasoner records which model answered. |

### Topics

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `image_topic` (configurable) | BEST_EFFORT / VOLATILE / KEEP_LAST=1 | `sensor_msgs/Image` (`rgb8`/`bgr8`) |
| Pub | `/openral/perception/objects` (configurable) | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `openral_msgs/PromptStamped` (`metadata_json` = `ObjectsMetadata`; `header.frame_id` = `sensor_id`) |

## Launch

Not invoked directly in practice — it's wired into the generic
`sim_e2e.launch.py` graph behind the `enable_object_detector` launch
argument, driven by the CLI:

```bash
openral deploy sim --config scenes/<SceneEnvironment>.yaml   # detector ON by default
```

`openral deploy sim` brings the detector up **by default** (pass
`--no-object-detector` to turn the leg off). With no explicit override the
default backend is the open-vocab `omdet-turbo-indoor` manifest, falling back to
the in-tree RT-DETR COCO ONNX (`rskills/rtdetr-coco-r18/model.onnx`) when the
omdet deps are absent; the leg auto-downgrades to off only when neither backend
is available. Use `--object-detector-manifest PATH` to pick a specific detector
rSkill or `--object-detector-onnx PATH` to force the fixed-label RT-DETR path.
The launch forwards `onnx_path`/`manifest_path`, `labels`, `model_id`,
`input_size`, and the topics as ROS parameters on the node, and throttles by the
manifest's `DetectorEngine` (`vlm_sidecar` 0.5 Hz, `zeroshot_hf` 2 Hz, ONNX 5 Hz). The detection camera can render at up to 640² with
resolution-consistent intrinsics so the lift scales `bbox_xyxy` to the
intrinsics correctly.

**On-demand locators.** Alongside the continuous detector, the deploy
launch can bring up one or more `mode: on_demand` open-vocab locators — each as
its **own** lifecycle node serving a namespaced
`/openral/perception/<alias>/locate_in_view`, so the reasoner picks a model via
`LocateInViewTool.detector`. Default = `omdet-turbo-locator` (when the omdet deps
import); add more with the repeatable `--object-detector-locator <manifest|alias>`
(LocateAnything is opt-in — NVIDIA non-commercial, 5 GB, needs the sidecar venv).
Each locator is an independent lifecycle + VRAM peer (per the single-resident-skill VRAM eviction policy), so the reasoner
can evict it before a co-resident VLA.

## What's in here

| Path | Role |
| --- | --- |
| `openral_perception_ros/ros_image_detector_node.py` | The node + its `main()` entry point. ROS imports are deferred into `main()` so the module stays import-safe on hosts without a sourced ROS env. |
| `openral_perception_ros/image_convert.py` | `image_to_bgr_bytes(msg)` — `sensor_msgs/Image` → contiguous BGR bytes (no `cv_bridge`); raises `ImageConvertError` on an unsupported encoding or padded rows. |
| `openral_perception_ros/depth_convert.py` | `depth_array_to_image_msg` / `image_msg_to_depth_array` (`32FC1` metres ↔ ndarray, NaN-preserving) + `camera_info_from_intrinsics` (pinhole `CameraInfo`). Pure message-boundary helpers, no torch. Unit-tested in `tests/unit/test_depth_convert.py`. |
| `openral_perception_ros/depth_provider_node.py` | `depth_provider_node` (`openral_depth_provider`): subscribes a mono RGB stream, calls the DA3 metric-depth sidecar (`tools/da3_depth_sidecar.py`, default `depth-anything/DA3-SMALL` — measured 0.27 GB / ~27 Hz on an 8 GB Ada) over ZMQ, and republishes a `32FC1` depth Image + `CameraInfo` for **nvblox** (`openral_slam_bringup/nvblox.launch.py`). Gives lidar-less robots a Nav2 cost map via cuVSLAM pose + nvblox. Best-effort; a sidecar hiccup skips the frame, never crashes the graph. Live bring-up is operator-run (sidecar venv + GPU). |
| `openral_perception_ros/segmenter_node.py` | `segmenter_node` (`SegmenterNode`, `openral_segmenter`): a managed lifecycle node serving `/openral/perception/segment_in_view` (`openral_msgs/srv/SegmentInView`) from an in-process SAM 2.1 hiera-small (`kind: segmenter` rSkill). See [Segmenter node](#segmenter-node) below. |
| `openral_perception_ros/camera_topics.py` | `resolve_camera_topics(entries, *, primary_camera, image_topic)` — the one implementation of the `"id=topic"` → ordered map resolution every node here takes as parameters. Pure, ROS-free, unit-tested in `tests/unit/test_perception_camera_topics.py`. |
| `package.xml` / `CMakeLists.txt` | `ament_cmake` manifest (depends on `rclpy`, `sensor_msgs`, `openral_msgs`, plus `geometry_msgs` + `tf2_ros` for the segmenter's prompt transform) + `ament_python_install_package` and `install(PROGRAMS … ros_image_detector_node.py)`. Installed as a program (not a setuptools `console_scripts` entry) so its `#!/usr/bin/env python3` shebang survives — a `console_scripts` entry is regenerated by colcon with a system-python shebang that can't see the `openral_runner` workspace package. Launch executable: `ros_image_detector_node.py`. |

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

It lives here, beside the detector rSkills, because **the HAL is deliberately
kept torch-free**: the HAL asks over the service rather than importing a model.

Four things about the contract are load-bearing:

- **The prompt is a 3-D point, not a pixel.** The node transforms it from the
  request's `frame_id` into the camera's optical frame through tf2 and projects
  it with the **manifest's** `SensorSpec.intrinsics`, rescaled to the frame
  actually received. Intrinsics never leak across the service boundary, and
  nothing here hardcodes a pixel or a frame id.
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

### Segmenter parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `cameras` | string[] | `[""]` | `"id=topic"` entries. Each id MUST be a `SensorSpec` name in `robot_yaml` — that is where its intrinsics and optical frame come from. |
| `primary_camera` | string | `default` | Camera used when a request leaves `camera` empty. |
| `image_topic` | string | `/openral/cameras/wrist/image` | Single-camera fallback topic. |
| `robot_yaml` | string | — (required) | `RobotDescription` path, for camera intrinsics + optical frame. |
| `manifest_path` | string | — (required) | `kind: segmenter` rSkill manifest. |
| `segment_in_view_service` | string | `/openral/perception/segment_in_view` | Service name. |
| `device` | string | `auto` | `auto` picks CUDA when available; set `cpu` on a pre-sm_70 GPU, where the CUDA wheels ship no kernels for the card. |

## Tests

- `tests/integration/test_segment_in_view_service.py` — live-ROS
  (`OPENRAL_TEST_ROS_LIVE=1`) round trip against the **real** node, the real
  rSkill manifest, the real SO-101 manifest, a real static TF and a real
  in-tree wrist frame: plural `mono8` masks at the source resolution, area
  ascending, parallel advisory scores, the caller's stamp echoed, plus the
  three typed failure branches (un-published camera, un-projectable prompt,
  deactivated node). Listed in `scripts/ros_live_tests.sh`.
- `tests/unit/test_perception_camera_topics.py` — the shared camera-id → topic
  resolution, including the `[""]`-for-unset rclpy quirk.
- `tests/unit/test_image_convert.py` — `rgb8`/`bgr8` → BGR byte
  conversion, channel order, and the rejection paths (bad encoding,
  padded stride).
- `tests/unit/test_depth_convert.py` — metric-depth ↔ `32FC1`
  Image round-trip (metres preserved, NaN preserved), the `CameraInfo`
  pinhole matrices, and the rejection paths (non-2D, wrong encoding,
  padded stride).
- `tests/unit/test_objects_detector.py` — the reused `ObjectsDetector`
  against a real deterministic `onnx.helper` fixture (no mocks, per
  CLAUDE.md §1.11).
- `tests/unit/test_rtdetr_onnx_detects.py` — gate that the exported
  RT-DETR ONNX produces real COCO detections via `ObjectsDetector`
  (skips when the ONNX or onnxruntime/PIL is absent).

## Related

- The perception → spatial-memory object lift; the deploy-sim integration this
  node serves.
- `packages/world_state/` — the consumer; subscribes
  `/openral/perception/objects` and lifts each 2D detection to a 3D
  `map`-frame centre.
- `openral_runner.backends.gstreamer.objects_detector.ObjectsDetector` /
  `openral_core.ObjectsMetadata` — the detector and metadata schema this
  node reuses.
- CLAUDE.md §3 (layer discipline) and §5.3 (QoS).
