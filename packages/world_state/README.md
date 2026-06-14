# `openral_world_state` (ROS 2)

Lifecycle-node wrapper for `openral_world_state.WorldStateAggregator`
— Layer 2 in the eight-layer architecture. Subscribes to joint and
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
| `CMakeLists.txt` | `ament_cmake_python` install for the Python package + script. |
| `package.xml` | ROS package manifest (depends on `openral_msgs` for the `WorldStateStamped` IDL). |

## Lifecycle contract

| Transition | Action |
| --- | --- |
| `configure` | Build the stub `RobotDescription` (unless one was injected via the compose factory), instantiate `WorldStateAggregator(staleness_limit_s=…)`, subscribe to `/joint_states`, create fast + slow `WorldStateStamped` publishers, wire the F8 diagnostics heartbeat. |
| `activate` | Start the fast publish timer at `publish_rate_hz_fast`; slow topic publishes every `round(fast/slow)` ticks from the same snapshot. |
| `deactivate` | Stop the timer. |
| `cleanup` | Destroy subscriptions, publishers, and aggregator (the latter only when this node owns it; the compose factory keeps the aggregator alive on its behalf). |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `robot_name` | string | `robot` | Short id used to build the stub `RobotDescription`. |
| `publish_rate_hz_fast` | double | `30.0` | Fast-topic snapshot rate (`/openral/world_state_fast`). |
| `publish_rate_hz_slow` | double | `5.0` | Slow-topic snapshot rate (`/openral/world_state_slow`). |
| `staleness_limit_s` | double | `0.5` | Age threshold for marking sensors stale (0.5 s clears 10 Hz camera flapping). |

## Topics

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `/joint_states` | BEST_EFFORT / VOLATILE / KEEP_LAST=5 | `sensor_msgs/JointState` |
| Pub | `/openral/world_state_fast` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/WorldStateStamped` |
| Pub | `/openral/world_state_slow` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/WorldStateStamped` |
| Pub | `/diagnostics` | default | `diagnostic_msgs/DiagnosticArray` (ADR-0018 F8 heartbeat) |

`WorldStateStamped` carries: a `sensor_msgs/JointState`, optional
base pose + twist, parallel arrays of EE names + poses, sensor image
topic refs, per-component diagnostic status (`DIAG_OK | DIAG_WARN |
DIAG_STALE | DIAG_ERROR`), per-component staleness in milliseconds,
battery percentage, and the tf2 `frame_ids[]` consumers should look up
themselves (ADR-0018 §2). The typed `WorldStateStamped` topics are the
only wire format — there is no JSON fallback.

## Wiring

```
HAL (e.g. openral_hal_so100) ──► /joint_states ─┐
                                                     ├─► world_state_node ──► /openral/world_state_fast (30 Hz)
sensors/ → ROS topics (planned per-sensor packages) ─┘                  └──► /openral/world_state_slow (5 Hz)
```

The HAL provides joints; sensor topics will be wired through additional
subscriptions as the per-sensor ROS packages land. The aggregator is
the only place where joint + sensor state are fused into the typed
snapshot consumed by S1 / S2.

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
  this via `hal.yml` after `colcon build`.

## Build

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build       # includes world_state alongside msgs + hal_so100
just ros2-test        # colcon test
just test-integration # PYTHONPATH-aware pytest run for the launch tests
```

## Object-lift — 2D→3D spatial memory (ADR-0035)

When `object_lift_enabled` is `True` (the default), the node also subscribes the
object-detector output and the 3D occupancy voxel grid, lifts each 2D detection to a
`map`-frame 3D centre via `VoxelFrustumLifter`, and maintains a temporal `ObjectMemory`.
Results are written into `WorldStateAggregator.update_detected_objects()` so
`WorldState.detected_objects` is non-empty for the first time.

**Best-effort contract:** a missing, empty, or stale voxel grid is a normal condition.
When there is no usable grid the node publishes `WorldState` with `detected_objects == []`
unchanged from today — no error, no warning spam, no degradation of the snapshot path.
The node **never fabricates a pose**: any path lacking a truthful 3D lift (no `map` TF,
no camera intrinsics, no in-frustum voxels, stale grid) silently skips the detection.

**On the wire (ADR-0035, landed):** the shared in-process `WorldStateAggregator` owns
`WorldState.detected_objects`, and `openral_msgs/WorldStateStamped` now also carries them as
`detected_object_*` parallel arrays (labels / confidences / `geometry_msgs/Point[]` positions /
`int32[]` track ids (`-1` = unset) / frame), serialised by `_fill_detected_objects` inside
`build_world_state_stamped_msg` and read back by `world_state_from_idl`. Separate-process
consumers (e.g. a standalone reasoner node reading `/openral/world_state_slow`) now **do** see
the spatial memory.

### Object-lift parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `object_lift_enabled` | `True` | Master toggle. `False` → feature fully inert; no subscriptions or timer are created. |
| `object_detections_topic` | `/openral/perception/objects` | `PromptStamped` topic carrying `ObjectsMetadata` detections. |
| `object_voxels_topic` | `/openral/world_voxels` | `OccupancyVoxels` topic (base frame, row-major x-fastest). |
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
RobotDescription.sensors[sensor_id].intrinsics                  ─┘          │
                                                                        memory tick (cadence Hz)
                                                                             │
                                                           VoxelFrustumLifter + ObjectMemory
                                                                             │
                                                  aggregator.update_detected_objects(...)
                                                                             │
                            WorldState.detected_objects (frame_id="map") ──► WorldStateStamped.detected_object_*
```

## See also

- `python/world_state/README.md` (planned) and the package source under
  `python/world_state/src/openral_world_state/`.
- `openral_core.WorldState` / `JointState` / `Pose6D` /
  `RobotDescription` — Pydantic schemas this node produces and consumes.
- `openral_msgs/msg/WorldStateStamped.msg` — the typed wire format.
- [ADR-0018](../../docs/adr/0018-ros2-reasoner-supervisor.md) §2 and the
  capability review's F2 section for the full design rationale.
- [ADR-0035](../../docs/adr/0035-perception-spatial-memory-object-lift.md) —
  perception→spatial-memory object lift design decisions and follow-ups.
- CLAUDE.md §6.1 (layer discipline) and §5.3 (QoS).
