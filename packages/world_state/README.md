# `openral_world_state` (ROS 2)

Lifecycle-node wrapper for `openral_world_state.WorldStateAggregator`
(Layer 2). Subscribes to joint and
sensor topics, ticks the aggregator, and publishes a typed
`openral_msgs/WorldStateStamped` snapshot at two rates (fast 30 Hz,
slow 5 Hz) for downstream consumers (Skill, Reasoner, dashboards).

## Synopsis

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build
source install/setup.bash

ros2 run openral_world_state world_state_node \
    --ros-args -p robot_name:=so100 \
      -p publish_rate_hz_fast:=30.0 \
      -p publish_rate_hz_slow:=5.0
```

## What's in here

| Path | Role |
| --- | --- |
| `openral_world_state_ros/lifecycle_node.py` | The managed lifecycle node. Wraps `WorldStateAggregator`; stub `RobotDescription` is built at `configure` time from the `robot_name` parameter. Owns the fast/slow `WorldStateStamped` publishers and the snapshot-to-message translator (`build_world_state_stamped_msg`). |
| `scripts/world_state_node` | Console entry point for `ros2 run`. |
| `openral_world_state_ros/object_tracker_node.py` | The #418 object-tracker lifecycle node (see "Object tracker" below). Wraps `openral_world_state.object_tracker.TrackerSession`. |
| `scripts/object_tracker_node` | Its console entry point. |
| `CMakeLists.txt` | `ament_cmake_python` install for the Python package + script. |
| `package.xml` | ROS package manifest (depends on `openral_msgs` for the `WorldStateStamped` IDL). |

## Lifecycle contract

| Transition | Action |
| --- | --- |
| `configure` | Build the stub `RobotDescription` (unless one was injected via the compose factory), instantiate `WorldStateAggregator(staleness_limit_s=…)`, subscribe to `/joint_states`, `/openral/policy_state`, the attachment-state topic and one image topic per `camera_names` entry, create fast + slow `WorldStateStamped` publishers, wire the F8 diagnostics heartbeat. |
| `activate` | Start the fast publish timer at `publish_rate_hz_fast`; slow topic publishes every `round(fast/slow)` ticks from the same snapshot. |
| `deactivate` | Stop the timer. |
| `cleanup` | Destroy subscriptions, publishers, and aggregator (the latter only when this node owns it; the compose factory keeps the aggregator alive on its behalf). |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `robot_name` | string | `robot` | Short id used to build the stub `RobotDescription`. |
| `publish_rate_hz_fast` | double | `30.0` | Fast-topic snapshot rate (`/openral/world_state_fast`). |
| `publish_rate_hz_slow` | double | `5.0` | Slow-topic snapshot rate (`/openral/world_state_slow`). |
| `joint_states_topic` | string | `/joint_states` | JointState input topic. Override in tests or composed graphs that isolate robot buses. |
| `staleness_limit_s` | double | `0.5` | Age threshold for marking sensors stale (0.5 s clears 10 Hz camera flapping). |
| `policy_state_staleness_limit_s` | double | `5.0` | Separate staleness window for the step-locked `/openral/policy_state`. |
| `camera_names` | string[] | `[""]` (none) | One `sensor_msgs/Image` subscription on `camera_topic(<name>)` per entry. |
| `attachment_state_topic` | string | `/openral/attachment_state` | `AttachmentState` input relayed onto `WorldStateStamped`. |
| `tracked_objects_topic` | string | `/openral/tracked_objects` | The object tracker's `TrackedObjectSet`, relayed as `WorldStateStamped.tracked_objects`; `tracked_objects_valid` false once no set arrived for 1.5 s (`DEFAULT_TRACKED_OBJECTS_STALE_S`). |
| `direct_image_frame_sensors` | string[] | `[""]` (none) | Cameras the co-located sensor leg writes directly; their image callback keeps the thumbnail span but does not re-write the aggregator. |

## Topics

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `/joint_states` | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `sensor_msgs/JointState` |
| Sub | `/openral/policy_state` | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `std_msgs/Float32MultiArray` |
| Sub | `/openral/attachment_state` (configurable) | RELIABLE / TRANSIENT_LOCAL / KEEP_LAST=1 | `openral_msgs/AttachmentState` |
| Sub | `/openral/tracked_objects` (configurable) | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/TrackedObjectSet` |
| Sub | `camera_topic(<name>)` per `camera_names` entry | BEST_EFFORT / VOLATILE / KEEP_LAST=1 | `sensor_msgs/Image` |
| Pub | `/openral/world_state_fast` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/WorldStateStamped` |
| Pub | `/openral/world_state_slow` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/WorldStateStamped` |
| Pub | `/diagnostics` | default | `diagnostic_msgs/DiagnosticArray` (supervisor-graph F8 heartbeat) |

`WorldStateStamped` carries: a `sensor_msgs/JointState`, optional
base pose + twist, parallel arrays of EE names + poses, sensor image
topic refs, per-component diagnostic status (`DIAG_OK | DIAG_WARN |
DIAG_STALE | DIAG_ERROR`), per-component staleness in milliseconds,
battery percentage, and the tf2 `frame_ids[]` consumers should look up
themselves (per the reasoner/supervisor-graph design). There is no JSON fallback.

## Wiring

```
HAL (openral_hal_node) ──► /joint_states ─┐
camera_topic(<name>) images ──────────────┤
/openral/attachment_state ────────────────┼─► world_state_node ──► /openral/world_state_fast (30 Hz)
/openral/policy_state ────────────────────┘                    └──► /openral/world_state_slow (5 Hz)
```

The aggregator is the only place joint + sensor state are fused
into the snapshot consumed by S1 / S2.

## Tests

- Unit (Python only — no ROS 2 required):
  `tests/unit/test_world_state.py` covers freshness, staleness
  latching, 30 Hz clock injection, and thread-safety.
- Integration (`launch_testing` equivalent in-process):
  `tests/integration/test_world_state_integration.py` exercises the
  real lifecycle node end-to-end against the typed `WorldStateStamped`
  topics: fast/slow rate ratio (≥5:1), `DIAG_OK` under steady joint
  publication, `DIAG_STALE` on dropout, recovery to `DIAG_OK`, and
  parallel-array consistency under 8 concurrent publishers. CI runs
  this in docker-build's live-ROS step (`scripts/ros_live_tests.sh`).
- Package (`test/`): `test_lifecycle_node_image_qos.py` (AST guard: camera subscriptions
  stay BEST_EFFORT) and `test_lifecycle_node_sigint_shape.py`.

## Build

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build       # includes world_state
just ros2-test        # colcon test
just test-integration # PYTHONPATH-aware pytest run for the launch tests
```

## Object-lift — 2D→3D spatial memory

When `object_lift_enabled` is `True` (default), the node subscribes the object-detector
output and a depth source, lifts each 2D detection to a `map`-frame 3D centre via
`VoxelFrustumLifter`, and maintains a temporal `ObjectMemory`, writing results into
`WorldStateAggregator.update_detected_objects()`.

Depth source: `object_voxels_topic` when fresh, else — depth-fallback amendment #11 —
`object_depth_points_topic` decoded by `depth_cloud_to_centers_base`. Decouples the lift
from octomap, so spatial-memory ingest and `recall_object` work with `--no-enable-octomap`.

**Best-effort:** missing/empty/stale depth source → `WorldState` publishes with
`detected_objects == []`, no error/warning. Never fabricates a pose — any path lacking a
truthful lift (no `map` TF, no intrinsics, no in-frustum voxels, stale grid) silently skips.

**On the wire:** `openral_msgs/WorldStateStamped` carries `detected_object_*` parallel
arrays (labels / confidences / `geometry_msgs/Point[]` positions / `int32[]` track ids
(`-1` = unset) / frame), serialised by `_fill_detected_objects` in
`build_world_state_stamped_msg` and read back by `world_state_from_idl` — so separate-process
consumers (e.g. a reasoner reading `/openral/world_state_slow`) see the spatial memory too.

### Object-lift parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `object_lift_enabled` | `True` | Master toggle. `False` → feature fully inert; no subscriptions or timer are created. |
| `object_detections_topic` | `/openral/perception/objects` | `PromptStamped` topic carrying `ObjectsMetadata` detections. |
| `object_voxels_topic` | `/openral/world_voxels` | `OccupancyVoxels` topic (base frame, row-major x-fastest). Preferred depth source when fresh. |
| `object_depth_points_topic` | `""` (disabled) | Depth-fallback amendment (#11) — depth `PointCloud2` used as the lift's depth source when no fresh voxel grid exists (e.g. `--no-enable-octomap`). Empty disables the fallback. `deploy_e2e.launch.py` passes `openral_core.deploy_cloud_topic`: the scene-pinned `runtime.octomap_cloud_topic` (a real driver's cloud) when set, else in sim the manifest's one depth sensor's `camera_topic(<name>, CameraTopicKind.POINTS)`, else empty (a real deploy with nothing pinned, or no depth sensor) — ADR-0108: consumers derive, never guess. |
| `object_lift_depth_max_points` | `4000` | Cap on depth-cloud points fed to the lift (uniform subsample) so a dense cloud can't stall per-detection projection. |
| `object_lift_map_frame` | `map` | Fixed frame used to anchor the object memory. |
| `object_lift_k_nearest` | `25` | K voxels (nearest to box centre) used to estimate the 3D centre. |
| `object_lift_min_voxels` | `3` | Minimum in-frustum voxels required to lift a detection; below this the detection is skipped. |
| `object_lift_iou_threshold` | `0.3` | 3D AABB IoU threshold for freeze-on-match association. |
| `object_lift_memory_cadence_hz` | `2.0` | Rate of the associate+evict memory tick. |
| `object_lift_max_misses` | `1` | Number of consecutive in-FOV misses before an object is evicted. |
| `object_lift_voxel_staleness_s` | `1.0` | Voxel grid older than this (seconds) is treated as unavailable. |

### Additional topics (when lift is enabled)

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `/openral/perception/objects` (configurable) | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `openral_msgs/PromptStamped` (metadata_json = `ObjectsMetadata`) |
| Sub | `/openral/world_voxels` (configurable) | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/OccupancyVoxels` |

### Wiring (with lift)

```
/openral/perception/objects (PromptStamped → ObjectsMetadata)  ─┐
/openral/world_voxels       (OccupancyVoxels, base_link)        ─┤
TF2: base_link→<cam_optical>, base_link→map                     ─┤──► _WorldStateLifecycleNode
ObjectsMetadata.camera_frame_id + camera_intrinsics             ─┘          │
  (driver CameraInfo, stamped by the detector; absent → the
   SensorSpec named by sensor_id, logged once per camera)
                                                                        memory tick (cadence Hz)
                                                                             │
                                                           VoxelFrustumLifter + ObjectMemory
                                                                             │
                                                  aggregator.update_detected_objects(...)
                                                                             │
                            WorldState.detected_objects (frame_id="map") ──► WorldStateStamped.detected_object_*
```

## Object tracker (issue #418)

`object_tracker_node` grounds the objects a goal names (`GraspDeclaration.target_queries`,
`PlaceDeclaration.destination_query`) as standing queries, fits each one to a set that
contains it, confirms it on the voxel map, tracks its identity, binds it at the HAL
trigger's ATTACH and releases it at DETACH or when the measured jaw opens. Every decision is
in the ROS-free `TrackerSession` (`python/world_state/.../object_tracker/session.py`); the
node owns the wire. `deploy_e2e.launch.py` starts it when `object_tracking` is not `off`.

| Mode | What leaves the node |
| --- | --- |
| `shadow` | `/openral/tracked_objects` and the logged predicted bindings (`object_tracker.bound`); nothing consumes them. Allowed on `deploy run`. |
| `bind` | Also `/openral/attachment_state`: every hand's binding (`TRACKED_OBJECT` / `PINCH_FALLBACK`) plus the live declarations relayed **without** a region. The HAL vision bridge stops publishing that topic (`vision_attachment_publish_attachment_state:=false`). Sim-only until Safety-WG review. |
| `on` | As `bind`; the kernel additionally keys grasp exemptions on tracked ids (K2). Sim-only. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `robot_yaml` | — | Manifest; the tracks' frame is its `base_frame`. |
| `mode` | `shadow` | `shadow` / `bind` / `on` (`bind`/`on` refused with `hal_mode:=real`). |
| `hal_mode` | `sim` | |
| `cameras` | none | Depth cameras: `camera_topic(<cam>, DEPTH_IMAGE / DEPTH_CAMERA_INFO / ROBOT_MASK)`. |
| `cell_m` | `0.015` | The octomap resolution (the pinch volume is grown by one cell). |
| `grounding_wait_s` | — (required) | `DeployRuntime.grounding_wait_s`. |
| `grounding_required` | `false` | `grounding_incomplete` is logged at error level (refusing the goal is dispatch's). |
| `release_dwell_s` | `0.3` | Jaw-open release dwell (source seconds). Calibration point. |
| `released_clear_timeout_s` | `3.0` | How long a released `TRACKED_OBJECT` binding stays on `attachment_state` (frozen by the kernel, its box cleared by the voxel bridge) unless the track leaves RELEASED first. MUST equal the kernel's `tracked_release_timeout_s` and the voxel bridge's `released_clear_timeout_s`; the launch passes one `tracked_release_timeout_s` to all three. |
| `object_queries_json` | `""` | The deploy's prompt-time `ObjectQuery` list (`deploy_sim._object_queries_json`), grounded at start whatever declarations are in force; a declaration re-keys the tracks it matches. |
| `segment_service` / `locate_service` | `/openral/perception/segment_in_view` / `""` | `locate_service` empty: no open-vocabulary detector, seeds and refreshes only. |
| `mask_tf_wait_s` | `0.3` | Wall seconds a depth frame waits for its robot mask and tf before it is used for ageing only. |
| `heartbeat_hz` | `2.0` | `tracked_objects` (and, while joint states are fresh, `attachment_state`) heartbeat. |
| `process_hz` | `15.0` | Frame loop. |
| `attachment_evidence_timeout_s` | `1.0` | No joint state for this long (node clock): the `attachment_state` heartbeat stops and the kernel's copy ages out. |

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `camera_topic(<cam>, DEPTH_IMAGE)`, `…DEPTH_CAMERA_INFO`, `…ROBOT_MASK` | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `sensor_msgs/Image`, `CameraInfo`, `Image` (mono8) |
| Sub | `/openral/world_voxels` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/OccupancyVoxels` |
| Sub | `/joint_states` | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `sensor_msgs/JointState` |
| Sub | `/openral/grasp_events` | RELIABLE / VOLATILE / KEEP_LAST=10 | `openral_msgs/GraspEvent` |
| Sub | `/openral/grasp_declaration`, `/openral/place_declaration` | RELIABLE / TRANSIENT_LOCAL / KEEP_LAST=1 | `openral_msgs/GraspDeclaration`, `PlaceDeclaration` |
| Pub | `/openral/tracked_objects` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/TrackedObjectSet` |
| Pub | `/openral/attachment_state` (`bind`/`on`) | RELIABLE / TRANSIENT_LOCAL / KEEP_LAST=1 | `openral_msgs/AttachmentState` |
| Client | `segment_service`, `locate_service` | | `SegmentInView`, `LocateInView` |

Clocks: every lifetime runs on source stamps (depth, grid `source_stamp`, joint states,
grasp events); the segmenter rate and budget on wall time. Revisions are seeded from wall
time at every activate, so a restarted tracker never regresses one (the kernel refuses a
regressing `tracked_objects` revision until it is re-activated).

Covered by `tests/unit/test_object_tracker_session.py` and the live
`tests/integration/test_object_tracker_node_live.py` (real tracker + World State nodes:
grounding, relay, staleness, `bind`-mode `TRACKED_OBJECT` binding, restart revision).

## See also

- `python/world_state/README.md` and the package source under
  `python/world_state/src/openral_world_state/`.
- `openral_core.WorldState` / `JointState` / `Pose6D` /
  `RobotDescription` — Pydantic schemas this node produces and consumes.
- `openral_msgs/msg/WorldStateStamped.msg` — the typed wire format.
- The ROS 2 reasoner/supervisor-graph design and the
  capability review's F2 section for the full design rationale.
- The perception → spatial-memory object-lift design decisions and follow-ups.
- CLAUDE.md §3 (layer discipline) and §2 (QoS).
