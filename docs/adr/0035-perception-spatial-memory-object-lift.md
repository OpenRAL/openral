# ADR-0035: Perception → spatial-memory object lift (2D→3D object-center lift)

- Status: **Accepted**
- Date: 2026-06-03
- Related: [ADR-0030](0030-geometric-safety-collision-checking.md) (OccupancyVoxels on
  `/openral/world_voxels`, base frame); [ADR-0037](0037-gstreamer-perception-bus-object-detection.md)
  (object-detection rSkill that produces `ObjectsMetadata` on `/openral/perception/objects` — the
  input this ADR consumes); [ADR-0018](0018-ros2-reasoner-supervisor.md) (reasoner consumes
  `WorldState.detected_objects` via the snapshot); CLAUDE.md §3 (layer boundaries — Sensors/rSkill
  → World State is a new cross-layer data flow).

## Context

`WorldState.detected_objects: list[DetectedObject]` has existed in the schema since the earliest
spatial-memory work (PRs #217–#220) but has always been an empty list — no node produced it.
ADR-0037 deliberately deferred the "2D→3D pose-lift" step, noting that the detector was the
missing producer but that the lift itself (needing depth, intrinsics, TF2, and a temporal
association policy) was a separate, later decision.

Two existing systems provide the raw material:

1. **ADR-0037 object-detector rSkill** publishes `ObjectsMetadata{ detections: list[ObjectDetection2D], model_id, sensor_id, frame_width, frame_height }` on `/openral/perception/objects`. Each `ObjectDetection2D` carries a pixel bounding box but no depth.
2. **ADR-0030 occupancy voxels** publishes a dense `OccupancyVoxels` grid on `/openral/world_voxels` (base frame, row-major x-fastest). These cells *are* in 3D, but they are associated with the robot's local environment as a whole, not with individual objects.

The lift gap: no node correlates the 2D pixel boxes against the 3D voxel grid and writes the
resulting object centers into the aggregator. The reasoner cannot reason spatially about objects
because `WorldState.detected_objects` is always empty.

## Decision

**New cross-layer data flow: Sensors/rSkill perception (Layer 1/3) → World State (Layer 2).**
This boundary crossing is what requires this ADR (CLAUDE.md §3).

### Lift algorithm

For each detected 2D box, lift to a single `(x, y, z)` center in `map` frame via
**frustum-projection + K-nearest-to-box-center** from the `/openral/world_voxels` grid:

1. Decode occupied voxel centers from `OccupancyVoxels` into base-frame 3D points.
2. Project all base-frame centers into the camera image plane using `SensorSpec.intrinsics`
   and the `base_link → <camera_optical>` TF2 transform. Discard `z ≤ 0`.
3. Scale each detection's `bbox_xyxy` from the detector's `frame_width × frame_height` to
   `intrinsics.width × intrinsics.height` (handles detector stream ≠ sensor native resolution).
4. Per detection: keep voxels whose projected `(u, v)` falls inside the scaled box; rank by
   2D distance from box center; take the K nearest (default K = 25). If fewer than `min_voxels`
   (default 3) survive → **skip** (insufficient 3D evidence; never fabricate a pose).
5. Transform the K selected centers to `map` frame using the `base_link → map` TF2 transform.
   `center_map = mean(K points)`. `bbox_3d = (min_xyz, max_xyz)` over the K points in map frame.
6. Emit `DetectedObject(label, confidence, pose=Pose6D(center_map, (0,0,0,1), frame_id="map"), bbox_3d=bbox_3d)`.

### Association policy — `ObjectMemory`

A per-run "tick" applies IoU-gated association (greedy, highest-confidence-first) to keep the
memory stable across frames:

- **Same-label + 3D AABB IoU ≥ `iou_threshold` (default 0.3) → freeze.** Leave stored pose and
  `bbox_3d` unchanged; bump `confidence = max(stored, candidate)`; reset miss counter. This
  prevents jitter from re-writing an object that is already well-remembered.
- **No match → create.** New tracked entry with a fresh monotonic `track_id`.
- **In-FOV + unmatched → evict after `max_misses` (default 1).** An object the camera should have
  seen but did not is presumed moved or removed.
- **Out-of-FOV → retain unchanged.** Objects outside the current camera view are never evicted
  by the absence of detections — the memory persists across camera pans.

The FOV predicate projects each `DetectedObject.pose.xyz` (map frame) back into the current
camera image; an object is "in FOV" iff its center projects within `(intrinsics.width × intrinsics.height)`.

### Hosting

The lift and memory tick are hosted **inside the existing `_WorldStateLifecycleNode`**, feeding
the shared `WorldStateAggregator` via a new typed setter `update_detected_objects()`. This is
consistent with the existing pattern (the node subscribes sensor topics and calls `update_*`
setters on the aggregator; the aggregator stays ROS-free).

All lift logic lives in **pure, ROS-free, unit-testable** Python modules in `openral_world_state`:
`object_lift.py` (`VoxelFrustumLifter`, geometry helpers, `ObjectsLiftError`) and
`object_memory.py` (`ObjectMemory`). The node wiring (subscriptions, TF2 buffer, timer) is in
`lifecycle_node.py` as usual.

### Best-effort contract (invariant)

> A missing, empty, or stale voxel grid is a **normal** condition. When there is no usable grid
> the world-state node behaves exactly as today: `WorldState.detected_objects == []`.
> No error, no warning spam, no degradation of the existing snapshot publish path.
> **Never fabricate a pose** (CLAUDE.md §1.2): any path without a truthful 3D lift — no `map`
> TF, no camera intrinsics, no in-frustum voxels, stale grid — skips the detection silently.

### Schema change

`ObjectsMetadata` gains two required fields:

```python
frame_width: int = Field(gt=0)   # pixel width of the detector's input frame
frame_height: int = Field(gt=0)  # pixel height of that frame
```

These make the pixel space of `bbox_xyxy` explicit so the lifter can scale to intrinsics
resolution rather than silently assuming stream resolution == sensor native resolution
(CLAUDE.md §1.4 "explicit beats implicit"). The detector backends (`ObjectsDetector`,
`NvmmObjectsDetector`) populate these at `detect()` time. Pre-publish: `schema_version`
stays `"0.1"` (CLAUDE.md §1.6, no migrators).

## Alternatives considered

- **Approach B: standalone `object_memory_node` + `openral_msgs/DetectedObjects` IDL.** A separate
  process would consume `/openral/perception/objects` and `/openral/world_voxels`, do the lift, and
  publish a new `DetectedObjects` IDL the world-state node subscribes to. Cleaner separation of
  concerns and allows the memory to be consumed by separate-process nodes (e.g. the reasoner node)
  without sharing a process. Deferred: adds a new IDL (a layer boundary), a new node binary, and
  inter-process latency. The v1 scope (in-process Pydantic memory for the collocated skill_runner /
  reasoner) does not require it; this ADR records it as the top documented follow-up.
- **Depth-image lift.** Use a registered `DEPTH16` sensor frame from `WorldState.image_frames`
  instead of the occupancy voxels. Rejected for v1: not every deployment carries a depth sensor
  alongside the RGB camera used by the detector; the voxel grid (ADR-0030) is an existing
  mandatory component. Depth-based lift is a future alternative.
- **K-nearest with depth tie-break.** Background voxels directly behind the object center can
  bias the mean center outward. Documented as a future tunable (v1 accepts the simplification).

## Consequences

### Layer boundary (new, authorized by this ADR)

The world-state lifecycle node now subscribes to `/openral/perception/objects` (a perception
output published by an rSkill/Layer 3 node) and `/openral/world_voxels` (Layer 2 ADR-0030
output). This is the first data-flow edge from the rSkill layer *into* the World State layer
(perception → World State). It is authorized here.

### Scope (v1) and top follow-ups

- **Detected objects now on the wire (landed).** The former top follow-up has shipped: the
  shared in-process `WorldStateAggregator` still owns `WorldState.detected_objects`, but
  `openral_msgs/WorldStateStamped` now also carries them as `detected_object_*` parallel arrays
  (`detected_object_labels` / `detected_object_confidences` / `detected_object_positions`
  (`geometry_msgs/Point[]`) / `detected_object_track_ids` (`int32[]`, `-1` = unset) /
  `detected_object_frame`). They are serialised by `_fill_detected_objects` inside
  `build_world_state_stamped_msg`, and read back by `world_state_from_idl`. Separate-process
  consumers reading `/openral/world_state_slow` (e.g. a standalone reasoner node) now **do** see
  the spatial memory. This was a deliberate IDL boundary crossing, authorized by this ADR.
- No orientation: `quat_xyzw = (0, 0, 0, 1)` (identity).
- Greedy (not Hungarian) data association.
- Single-camera FOV predicate.
- No safety-kernel feed: converting `detected_objects` into `WorldCollisionPrimitive`s is a
  separate downstream concern.
- Approach B (standalone process) is the documented path if a separate process is later required.

### Deploy-sim integration (landed)

To exercise the lift in `openral deploy sim` without standing up the GStreamer perception tee
(ADR-0018 F6 / ADR-0037) that the on-robot path uses, a standalone ROS-Image detector ships in
the new `openral_perception_ros` package. Its `ros_image_detector_node`
(`RosImageObjectDetectorNode`) subscribes the `panda_mobile` `agentview_left` RGB
`sensor_msgs/Image`, runs the GStreamer-free `openral_runner` `ObjectsDetector` (the
`rtdetr-coco-r18` rSkill ONNX), and publishes `openral_msgs/PromptStamped` (carrying
`ObjectsMetadata`) on `/openral/perception/objects` — the same topic and schema the world-state
node already consumes. Detections are attributed to `sensor_id` (default `front_depth`) so the
lift projects through the co-located depth camera's optical frame.

It is wired into the generic `sim_e2e.launch.py` behind the `enable_object_detector` launch
argument, driven by `openral deploy sim --enable-object-detector` (auto-enabled when
`rskills/rtdetr-coco-r18/model.onnx` is present). The detection camera can render at up to 640²
with resolution-consistent intrinsics (`scale_intrinsics_to` / the `depth_synth_kwargs`
`render_size`) so the lift scales `bbox_xyxy` to the intrinsics resolution correctly.

### New node parameters

See `packages/world_state/README.md` §Object-lift parameters for the full table.
Key parameters: `object_lift_enabled` (master toggle, default `True`), `object_lift_k_nearest`
(default 25), `object_lift_min_voxels` (default 3), `object_lift_iou_threshold` (default 0.3),
`object_lift_memory_cadence_hz` (default 2.0 Hz), `object_lift_max_misses` (default 1),
`object_lift_voxel_staleness_s` (default 1.0 s).

### New exception

`ObjectsLiftError(ValueError)` — raised by geometry helpers on degenerate inputs (e.g. a
near-zero quaternion or an occupancy buffer whose byte length does not match `size_x * size_y * size_z`).

## Amendment — 2026-06-15 (octomap-free depth fallback; GitHub #11)

The object lift previously required an octomap voxel grid (`/openral/world_voxels`) as its depth
source and silently dropped every detection when none was published — so with `--no-enable-octomap`
(common in dense scenes that false-positive the kernel's octomap collision check) spatial memory
never accumulated and `recall_object` always missed. The lift now **falls back to the depth camera
point cloud** when no fresh voxel grid is available: `_WorldStateLifecycleNode` subscribes the
configurable `object_depth_points_topic` (default `/openral/cameras/front_depth/points`,
`sensor_msgs/PointCloud2`, BEST_EFFORT) and, when `_latest_voxels` is stale/absent, decodes it via
`depth_cloud_to_centers_base` (drop non-finite returns, subsample to `object_lift_depth_max_points`
= 4000, transform the cloud's optical frame → base frame) and feeds the result to
`VoxelFrustumLifter.lift` as `occupied_centers_base` — interchangeable with the octomap path. The
octomap voxel grid remains preferred when fresh (filtered, persistent). Verified live on
`robocasa_baguette` with octomap **off**: `world_state_slow.detected_object_labels` populated and
`recall_object` returns matches. New public symbol: `openral_world_state.depth_cloud_to_centers_base`.
SLAM stays on-by-default for any robot that can run it (`capabilities.has_lidar`) so the `map` frame
the lift projects through is present (deploy-sim CLI, firm default).
