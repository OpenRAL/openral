# `openral_perception_ros` (ROS 2)

> **ADR-0035 — standalone ROS-Image object-detection producer (no
> GStreamer).**

A single `ament_cmake` ROS 2 package (Python node) that subscribes a
camera `sensor_msgs/Image`, runs the GStreamer-free
[`openral_runner.backends.gstreamer.objects_detector.ObjectsDetector`](../../python/runner/src/openral_runner/backends/gstreamer/objects_detector.py)
(or a manifest-driven backend — see below), and publishes the
detections on `/openral/perception/objects` — the topic the world-state
lifecycle node lifts to 3D in deploy-sim. The deploy-sim **default** backend is
the open-vocabulary `omdet-turbo-indoor` continuous detector (grounds arbitrary
indoor/kitchen objects); it falls back to the fixed-label RT-DETR COCO ONNX
(`rtdetr-coco-r18`) when the omdet deps are not installed.

It exists so the perception → spatial-memory object lift (ADR-0035) can
run against a plain ROS image topic in `openral deploy sim`, without
standing up the GStreamer perception tee (ADR-0018 F6 / ADR-0037) that
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
so the geometry is consistent (ADR-0035 §3).

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
| `query_topic` | string | `/openral/perception/detector_query` | ADR-0037 — open-vocab retarget topic (on-demand). Namespaced per locator (ADR-0056). |
| `locate_in_view_service` | string | `/openral/perception/locate_in_view` | ADR-0056 — service name. The deploy launch sets it to `/openral/perception/<alias>/locate_in_view` per on-demand locator so several locators co-exist. |
| `detector_id` | string | `""` | ADR-0056 — this locator's alias, echoed in the `LocateInView` response so the reasoner records which model answered. |

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

**On-demand locators (ADR-0056).** Alongside the continuous detector, the deploy
launch can bring up one or more `mode: on_demand` open-vocab locators — each as
its **own** lifecycle node serving a namespaced
`/openral/perception/<alias>/locate_in_view`, so the reasoner picks a model via
`LocateInViewTool.detector`. Default = `omdet-turbo-locator` (when the omdet deps
import); add more with the repeatable `--object-detector-locator <manifest|alias>`
(LocateAnything is opt-in — NVIDIA non-commercial, 5 GB, needs the sidecar venv).
Each locator is an independent lifecycle + VRAM peer (ADR-0050), so the reasoner
can evict it before a co-resident VLA.

## What's in here

| Path | Role |
| --- | --- |
| `openral_perception_ros/ros_image_detector_node.py` | The node + its `main()` entry point. ROS imports are deferred into `main()` so the module stays import-safe on hosts without a sourced ROS env. |
| `openral_perception_ros/image_convert.py` | `image_to_bgr_bytes(msg)` — `sensor_msgs/Image` → contiguous BGR bytes (no `cv_bridge`); raises `ImageConvertError` on an unsupported encoding or padded rows. |
| `package.xml` / `CMakeLists.txt` | `ament_cmake` manifest (depends on `rclpy`, `sensor_msgs`, `openral_msgs`) + `ament_python_install_package` and `install(PROGRAMS … ros_image_detector_node.py)`. Installed as a program (not a setuptools `console_scripts` entry) so its `#!/usr/bin/env python3` shebang survives — a `console_scripts` entry is regenerated by colcon with a system-python shebang that can't see the `openral_runner` workspace package. Launch executable: `ros_image_detector_node.py`. |

## Tests

- `tests/unit/test_image_convert.py` — `rgb8`/`bgr8` → BGR byte
  conversion, channel order, and the rejection paths (bad encoding,
  padded stride).
- `tests/unit/test_objects_detector.py` — the reused `ObjectsDetector`
  against a real deterministic `onnx.helper` fixture (no mocks, per
  CLAUDE.md §1.11).
- `tests/unit/test_rtdetr_onnx_detects.py` — gate that the exported
  RT-DETR ONNX produces real COCO detections via `ObjectsDetector`
  (skips when the ONNX or onnxruntime/PIL is absent).

## Related

- [ADR-0035](../../docs/adr/0035-perception-spatial-memory-object-lift.md) —
  perception → spatial-memory object lift; the deploy-sim integration this
  node serves.
- `packages/world_state/` — the consumer; subscribes
  `/openral/perception/objects` and lifts each 2D detection to a 3D
  `map`-frame centre.
- `openral_runner.backends.gstreamer.objects_detector.ObjectsDetector` /
  `openral_core.ObjectsMetadata` — the detector and metadata schema this
  node reuses.
- CLAUDE.md §3 (layer discipline) and §5.3 (QoS).
