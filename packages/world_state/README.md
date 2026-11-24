# `openral_world_state` (ROS 2)

ROS 2 wrapper (Layer 2) for `openral_world_state`. It holds three lifecycle nodes:

- `world_state_node`: wraps `WorldStateAggregator`, subscribes to joint and sensor topics and
  publishes a typed `openral_msgs/WorldStateStamped` at 30 Hz (fast) and 5 Hz (slow) for the
  skill, the reasoner and dashboards. It also lifts 2D detections to 3D object memory.
- `object_tracker_node`: grounds and tracks the objects a goal names (sim-only).
- `instance_mask_propagator_node`: an experimental SAM2 mask-to-voxel node (sim-only).

## Run it

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build && source install/setup.bash

ros2 run openral_world_state world_state_node \
    --ros-args -p robot_name:=so100 -p publish_rate_hz_fast:=30.0 -p publish_rate_hz_slow:=5.0
```

`deploy_e2e.launch.py` (in `openral_rskill_ros`) starts the nodes for a deploy and passes the
parameters below from the scene.

| Path | Role |
| --- | --- |
| `openral_world_state_ros/lifecycle_node.py` | `world_state_node`. Builds a stub `RobotDescription` from `robot_name` unless the compose factory injects one; owns the publishers and `build_world_state_stamped_msg`. |
| `openral_world_state_ros/object_tracker_node.py` | `object_tracker_node`: the ROS wire around the ROS-free `TrackerSession` (`python/world_state/.../object_tracker/session.py`). |
| `openral_world_state_ros/instance_mask_propagator_node.py` | `instance_mask_propagator_node` (experiment). |
| `scripts/` | `ros2 run` entry points for the three nodes. |

## `world_state_node`

| Transition | Action |
| --- | --- |
| `configure` | Build the stub `RobotDescription`, create `WorldStateAggregator(staleness_limit_s=...)`, subscribe, create the fast and slow publishers, wire the `/diagnostics` heartbeat. |
| `activate` | Start the fast timer; the slow topic publishes every `round(fast/slow)` ticks from the same snapshot. |
| `deactivate` | Stop the timer. |
| `cleanup` | Destroy subscriptions and publishers, and the aggregator when this node owns it. |

Transitions use `log_lifecycle_errors` and `refuse_unavailable_transitions`: a failed
callback is logged, and a transition the node's state lacks is refused rather than killing the
node.

| Parameter | Default | Notes |
| --- | --- | --- |
| `robot_name` | `robot` | Id for the stub `RobotDescription`. |
| `publish_rate_hz_fast` / `_slow` | `30.0` / `5.0` | Rates of `/openral/world_state_fast` / `_slow`. |
| `joint_states_topic` | `/joint_states` | Override to isolate robot buses. |
| `staleness_limit_s` | `0.5` | Sensor staleness threshold (clears 10 Hz camera flapping). |
| `policy_state_staleness_limit_s` | `5.0` | Window for the step-locked `/openral/policy_state`. |
| `camera_names` | none | One `sensor_msgs/Image` subscription on `camera_topic(<name>)` each. |
| `direct_image_frame_sensors` | none | Cameras the co-located sensor leg writes directly: the callback keeps the thumbnail span but does not re-write the aggregator. |
| `attachment_state_topic` | `/openral/attachment_state` | `AttachmentState` relayed onto `WorldStateStamped`. |
| `tracked_objects_topic` | `/openral/tracked_objects` | Relayed as `tracked_objects`; `tracked_objects_valid` is false after 1.5 s without a set. |

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `/joint_states` | BEST_EFFORT / VOLATILE / 5 | `sensor_msgs/JointState` |
| Sub | `/openral/policy_state` | BEST_EFFORT / VOLATILE / 5 | `std_msgs/Float32MultiArray` |
| Sub | `/openral/attachment_state` | RELIABLE / TRANSIENT_LOCAL / 1 | `openral_msgs/AttachmentState` |
| Sub | `/openral/tracked_objects` | RELIABLE / VOLATILE / 1 | `openral_msgs/TrackedObjectSet` |
| Sub | `camera_topic(<name>)` per camera | BEST_EFFORT / VOLATILE / 1 | `sensor_msgs/Image` |
| Sub | `/openral/perception/objects`, `/openral/world_voxels` | see Object lift | `PromptStamped`, `OccupancyVoxels` |
| Pub | `/openral/world_state_fast`, `/openral/world_state_slow` | RELIABLE / VOLATILE / 1 | `openral_msgs/WorldStateStamped` |
| Pub | `/diagnostics` | default | `diagnostic_msgs/DiagnosticArray` |

`WorldStateStamped` carries a `JointState`, optional base pose and twist, parallel arrays of
EE names and poses, sensor image refs, per-component status (`DIAG_OK | DIAG_WARN |
DIAG_STALE | DIAG_ERROR`) and staleness in ms, battery percentage, the tf2 `frame_ids[]`
consumers look up themselves, the attachment state, the tracked objects and the
`detected_object_*` arrays. There is no JSON fallback.

## Object lift: 2D detections to 3D memory

With `object_lift_enabled` (default `True`) the node lifts each 2D detection to a `map`-frame
3D centre (`VoxelFrustumLifter`), keeps a temporal `ObjectMemory`, and writes it to
`WorldStateAggregator.update_detected_objects()`. The result rides
`WorldStateStamped.detected_object_*` (labels, confidences, positions, track ids with `-1`
unset, frame, existence belief with NaN none), so separate-process readers such as the
reasoner see it too. It is best effort: a missing, empty or stale depth source publishes
`detected_objects == []`, and a lift with no `map` TF, intrinsics or in-frustum voxels is
skipped. The safety kernel does not read it.

Depth source: `object_voxels_topic` when fresh, else `object_depth_points_topic` (so spatial
memory works with `--no-enable-octomap`). The launch passes `openral_core.deploy_cloud_topic`:
the scene's `runtime.octomap_cloud_topic` when set, else in sim the manifest's one depth
sensor's points topic, else empty.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `object_lift_enabled` | `True` | Master toggle; `False` creates no subscription or timer. |
| `object_detections_topic` | `/openral/perception/objects` | `PromptStamped` carrying `ObjectsMetadata`. |
| `object_voxels_topic` | `/openral/world_voxels` | `OccupancyVoxels` (base frame); preferred depth source. |
| `object_depth_points_topic` | `""` (off) | Depth `PointCloud2` fallback. |
| `object_lift_depth_max_points` | `4000` | Cap on cloud points fed to the lift (uniform subsample). |
| `object_lift_map_frame` | `map` | Frame anchoring the memory. |
| `object_lift_k_nearest` | `25` | Voxels nearest the box centre used for the 3D centre. |
| `object_lift_min_voxels` | `3` | Fewer in-frustum voxels skips the detection. |
| `object_lift_iou_threshold` | `0.3` | 3D AABB IoU for freeze-on-match association. |
| `object_lift_memory_cadence_hz` | `2.0` | Associate and evict tick rate. |
| `object_lift_voxel_staleness_s` | `1.0` | An older grid counts as unavailable. |
| `object_memory_path` | `""` (off) | `SceneGraph` JSON restored at configure and saved when the object set changes. `openral deploy sim/run --memory-dir <bundle>` sets it when `<bundle>/object_memory.json` exists. A file that does not validate is neither restored nor overwritten. |

Existence-belief parameters (log-odds, see below):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `object_lift_hit_log_odds` | `0.85` | Added when a detection matches (once per tick). |
| `object_lift_miss_free_log_odds` | `-0.85` | Added when undetected and the depth source sees through the box. |
| `object_lift_miss_occupied_log_odds` | `-0.1` | Added when undetected, the box is still occupied and its camera's detector delivered a batch. `0` means no evidence, but then a false detection is never forgotten. |
| `object_lift_belief_max` / `_absent` | `3.5` / `-2.0` | Upper clamp; lower clamp and eviction threshold. |
| `object_lift_max_range_m` | `5.0` | A miss counts only for a box wholly in view within this range. Set it to the detector's working range. |
| `object_lift_miss_margin_m` | `0.03` | Tolerance of the miss test; a box no taller than twice this is counted whole. |

### Existence belief

Each remembered object carries a log-odds existence belief. A matched detection adds `hit`;
an unmatched tick adds what the depth source says (`build_miss_classifier`). A miss is weighed
once per new depth observation, and an object is evicted when its belief reaches
`object_lift_belief_absent`, never on one miss.

| Evidence | When | Effect |
| --- | --- | --- |
| none | no `bbox_3d`; box not wholly in a detection camera's view or beyond max range; an occluder in front; a depth hole behind; no fresh depth or TF | no change |
| occupied | box still occupied, and its camera's detector delivered a batch this tick | `miss_occupied` |
| free | box empty and the depth source sees something behind it | `miss_free` |

`DetectedObject.existence` is the belief as a probability: above 0.5 is *present*, else
*uncertain*. The reasoner's `SpatialMemory` marks a node the set drops as *absent* but keeps
its last pose for "where was X". With `object_memory_path` set, a restart restores every
object at belief 0 (*uncertain*): a fresh detection confirms it, a clear view from a camera
whose detector has delivered a batch removes it. Persist only where `map` is the same place
every deploy; under a SLAM map that starts where the robot is, restored objects land displaced.

Limits: an object no taller than twice the margin is never cleared by the depth source and
decays only through `miss_occupied`; the see-through test counts points, not rays, so mount
the depth sensor with the detection camera; a camera whose detector has delivered nothing is
never used to clear. A node restarted without persistence publishes an empty set, which the
reasoner reads as every followed object evicted until re-detected.

## `object_tracker_node`

Grounds the objects a goal names (`GraspDeclaration.target_queries`,
`PlaceDeclaration.destination_query`) as standing queries, fits each to a set that contains
it, confirms it on the voxel map, tracks its identity, binds it at the HAL trigger's ATTACH and
releases it at DETACH or when the measured jaw opens. `deploy_e2e.launch.py` starts it when
`object_tracking` is not `off`. Every mode's heartbeat also publishes `std_msgs/Bool`
`/openral/object_tracker/hands_clear` (no hand holds anything and every hand is clear of every
target track); the completion check asks its judge only while it is true.

**Sim-only.** The node refuses `tracking` and `on` unless `hal_mode:=sim` is set explicitly;
unset means `real`, as in the kernel. A real deploy runs `object_tracking=off`.

| Mode | What leaves the node |
| --- | --- |
| `tracking` | `/openral/tracked_objects` and `/openral/attachment_state`: every hand's binding (`TRACKED_OBJECT` / `PINCH_FALLBACK`) plus the live declarations with the regions the tracker measures: the grasp `region` (the one CONFIRMED / OCCLUDED TARGET track, `grasp_region`; none once a hold started), the `workspace_region` and the place `region`. The kernel applies its open box rules over them (ADR-0115, ADR-0097/0098, workspace contact). The HAL vision bridge then runs only its grasp trigger (`/openral/grasp_events`); an ATTACH holds the action ack until this node's next attachment revision. The octomap bridge clears a `TRACKED_OBJECT` binding's cells by identity; a `PINCH_FALLBACK` hold stays mapped. The former `shadow` / `bind` load as aliases of `tracking`. |
| `on` | As `tracking` without the grasp `region`; grasp exemptions key on tracked ids, with their rules in the exemption provider (OpenRAL Pro, ADR-0121), so `on` is refused without `exemption_provider`. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `robot_yaml` | none | Manifest; the tracks' frame is its `base_frame`. |
| `mode` | `tracking` | `tracking` or `on`. |
| `hal_mode` | `real` | Must be `sim` (see above). |
| `cameras` | none | Depth cameras: `camera_topic(<cam>, DEPTH_IMAGE / DEPTH_CAMERA_INFO / ROBOT_MASK)`. |
| `cell_m` | `0.015` | Octomap resolution (the pinch volume grows by one cell). |
| `grounding_wait_s` | required (> 0) | `DeployRuntime.grounding_wait_s`. |
| `grounding_required` | `false` | Logs `grounding_incomplete` at error; the skill runner refuses the goal with `ROSPerceptionStale` when a TARGET query is not grounded in time. |
| `search_start_s` / `search_max_s` | `5.0` / `60.0` | A query unresolved for the first escalates `unresolved`, for the second `exhausted` (a `ROSObjectNotInMemory` with `ObjectQueryEvidence`). The reasoner cancels the matching manipulation and recovers within the task dispatch budget. |
| `release_dwell_s` | `0.3` | Jaw-open release dwell (`RELEASE_DWELL_S`). |
| `released_clear_timeout_s` | `3.0` | Longest a released `TRACKED_OBJECT` binding stays on `attachment_state`; it leaves sooner once every hand box is 4 cm clear or a new track of its label is confirmed touching it. Must equal the kernel's `tracked_release_timeout_s` and the voxel bridge's; the launch passes one value to all three. |
| `wrist_links` | `1` | The kernel's hand-link set (`workspace_contact_wrist_links`). |
| `object_queries_json` | `""` | The deploy's prompt-time `ObjectQuery` list, grounded at start. |
| `place_zones_json` | `""` | Map / odom-frame place zones. |
| `zone_reach_center` / `zone_reach_radius_m` | `[0,0,0]` / `0.0` | Reach ball (the launch's world-map coverage) used to pick one of several same-slot zones. |
| `declared_object_catalog_json`, `object_surface_model_json` | `""` | Declared object geometry; the surface-model pilot (below). |
| `static_cells_file` | `""` | Static geometry cells. |
| `locate_service` | `""` | Empty: no open-vocabulary detector, seeds and refreshes only. See the [`LocateInView` contract](../msgs/README.md#services--srv). |
| `locate_contrast_labels` | `false` | Append the contrast labels to the locate query; the launch sets it only for a `zeroshot_hf` (OmDet) locator. |
| `joint_states_topic` | `/joint_states` | |
| `instance_cells` | `false` | Subscribe `/openral/instance_cells` (experiment, below). |
| `station_perception_initially_on` | `true` | The launch sets `false` on a mobile base (station mode). |

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Sub | `camera_topic(<cam>, DEPTH_IMAGE)`, `DEPTH_CAMERA_INFO`, `ROBOT_MASK` | BEST_EFFORT / VOLATILE / 5 | `Image`, `CameraInfo`, `Image` (mono8) |
| Sub | `/openral/world_voxels` | RELIABLE / VOLATILE / 1 | `openral_msgs/OccupancyVoxels` |
| Sub | `/joint_states` | BEST_EFFORT / VOLATILE / 5 | `sensor_msgs/JointState` |
| Sub | `/openral/grasp_events` | RELIABLE / VOLATILE / 10 | `openral_msgs/GraspEvent` |
| Sub | `/openral/safe_action` | RELIABLE / VOLATILE / 10 | `openral_msgs/ActionChunk` (the approved jaw targets, for the stalled-jaw read) |
| Sub | `/openral/grasp_declaration`, `/openral/place_declaration` | RELIABLE / TRANSIENT_LOCAL / 1 | `GraspDeclaration`, `PlaceDeclaration` |
| Pub | `/openral/tracked_objects` | RELIABLE / VOLATILE / 1 | `openral_msgs/TrackedObjectSet` |
| Pub | `/openral/attachment_state` | RELIABLE / TRANSIENT_LOCAL / 1 | `openral_msgs/AttachmentState` |
| Pub | `/openral/object_tracker/hands_clear` | RELIABLE / VOLATILE / 1 | `std_msgs/Bool` |
| Client | `/openral/perception/segment_in_view`, `locate_service` | | `SegmentInView`, `LocateInView` |
| Service | `/openral/object_tracker/station_perception` | | `std_srvs/SetBool` |

### Behaviour notes

- **Locate and segment evidence.** Only an empty `LocateInView.failure_reason` means the
  detector completed its check; a non-empty cause becomes a refusal observation on the query
  and is never read as the object being absent. A completed empty locate is
  `locate_found_nothing`. The tracker sends the depth frame's non-zero `source_stamp_ns` and
  discards boxes whose stamp is missing or different. Every `SegmentInView` mask must echo the
  requested camera, stamp and optical `frame_id` or it is refused before decoding. Frames
  lacking camera pose or robot mask count against the query as `no_camera_pose` /
  `no_robot_mask`, never as a detector attempt.
- **Binding.** A normal `TRACKED_OBJECT` ATTACH needs a non-future fit within
  `TrackerParams.max_age_s`; voxel confirmation does not refresh fit age. A late ATTACH uses
  the 1.5 s window on `last_fit_stamp_ns`. A RELEASED track is never a bind candidate.
- **Scheduling.** When an unresolved query and a visible track refresh both wait, the
  scheduler alternates their turns.
- **Fixed constants.** `_PROCESS_HZ` 15 frame loop; `_HEARTBEAT_HZ` 2 for `tracked_objects`
  (and `attachment_state` while joint states are fresh); no joint state for 1.0 s stops the
  `attachment_state` heartbeat so the kernel's copy ages out; the last 4 depth frames wait
  1.0 s for their robot mask and tf.
- **Clocks.** Lifetimes run on source stamps; the segmenter rate and budget on wall time.
  Revisions are seeded from wall time at every activate, so a restart never regresses one (the
  kernel refuses a regressing `tracked_objects` revision).
- **Backup fill.** A steady-clock timer in a separate callback group republishes cached sets
  with unchanged stamps and revisions, and stops after 5 s without main activity. It does not
  renew measurement evidence or change freshness limits.

### Station mode (mobile base, tracking on)

The node stays ACTIVE. The skill runner switches its *station perception* on for a skill that
actuates an arm or gripper and off for a base-only one (`station_perception_initially_on:
false`); the switch replies `success=False` while the node is not ACTIVE. Off, it consumes no
frame and runs no query, and `attachment_state` keeps its heartbeat with the hands' held
payloads only (`TrackerSession.held_attachments`): no declaration, region or tracked key
reaches the kernel, while a payload in a hand stays checked and the kernel's attachment stream
stays fresh (a stale one refuses every candidate, a base twist included). Either edge discards
the previous station (`leave_station`).

A base that moves keeps its tracks. Once the measured `base_joints` pose leaves the pose the
tracks were measured at by more than 2 cm or 0.02 rad (`STATION_MOTION_M` /
`STATION_ROTATION_RAD` in `tracking_thresholds.yaml`), the session re-expresses every unheld
base-frame record by the measured motion, once at the source. Held payloads ride the hand, and
so does a track centred in the grasp declaration's base-frame `search_box`. The carry is
odometry (base joints in `/joint_states`; a platform with only tf2 odom gets no carry). A
track carried past 0.75 m and 0.8 rad since its last fit is dropped (`CARRY_MAX_M` /
`CARRY_MAX_RAD`). The session fails closed: a base pose that stops being measured, a sim reset
or a non-planar motion drops every unheld track (`object_tracker.base_moved why=...`). While
the base moves continuously perception re-arms every 2 cm, so a whole-body skill sees few
fresh fits. A carried box feeds `grasp_region` like a measured one, odometry error included.

A place declaration naming same-slot map-frame zones (`zone_candidates_target_id`) resolves to
the one whose box centre lies in the robot's reach (`zone_reach_center` / `zone_reach_radius_m`);
none or several resolve to no region. A zone is accepted only in the robot's `map_frame` /
`odom_frame` (or its base frame on a fixed base); another frame is refused and logged
(`object_tracker.place_zones_refused`).

## `instance_mask_propagator_node` (experiment, sim-only)

`DeployRuntime.instance_cells: true` (refused with tracking `off`) starts it beside the
tracker. It follows each TARGET / DESTINATION track with a SAM2.1 video mask (seeded only
from a CONFIRMED track's projected box), lifts the mask minus robot pixels through aligned
depth into `/openral/world_voxels` cells, and publishes them per track on
`/openral/instance_cells` (`openral_msgs/InstanceCells`). `configure` loads SAM2, `activate`
starts it, `deactivate` / `cleanup` drop every mask. The tracker fits those tracks' boxes to
the cells (axis-aligned lattices only) and re-finds a released target once within `recent_s`.
Thresholds: the `vision` section of `tracking_thresholds.yaml`.

## Observed object-local surfaces (opt-in pilot)

`runtime.object_surface_model` in a deploy scene enables an observation-only PCL registration
pilot; every convergence and resource limit must be supplied (evaluation settings, not
calibrated safety bounds). Build this package with PCL and `pybind11_vendor` so
`_openral_pcl_registration` imports; without it the tracker refuses to configure.

Each identity keeps observed surface voxels in a canonical local frame; accepted
registrations relocate the model and rejected ones keep the last accepted one. Partial views
never carve unseen cells free, and one identity cannot delete another's cells or raw OctoMap
occupancy. The optional `free_space_calibration` block (`calibration_ref`, `sensor_ids`,
`depth_error_m`, `pose_error_m`, `pixel_error`, `max_age_s`) lets a fresh depth frame retire
local voxels only when every pixel covering the uncertainty-grown voxel has a valid return
behind it; occlusion, missing depth and exhausted budgets retain cells. The pose bound must
cover relative object/camera pose error (ICP residual is not that bound). Model status,
size, residual and overlap go to `/diagnostics` under `openral_object_tracker: observed
surface models`. These models never replace containing fits, change the world occupancy map,
authorize contact or shrink an attached envelope.

## Tests

- Unit (no ROS): `tests/unit/test_world_state.py`, `tests/unit/test_object_tracker_session.py`.
- Integration: `tests/integration/test_world_state_integration.py` (real lifecycle node:
  fast/slow ratio, `DIAG_OK` / `DIAG_STALE` and recovery, parallel-array consistency) and
  `tests/integration/test_object_tracker_node_live.py` (real tracker + world state nodes).
  CI runs both in docker-build's live-ROS step (`scripts/ros_live_tests.sh`).
- Package `test/`: `test_lifecycle_node_image_qos.py` (camera subscriptions stay
  BEST_EFFORT) and `test_lifecycle_node_sigint_shape.py`.

Build and test: `just ros2-build`, `just ros2-test`, `just test-integration`.

## See also

- [`python/world_state/README.md`](../../python/world_state/README.md): the ROS-free aggregator,
  tracker and memory.
- `openral_msgs/msg/WorldStateStamped.msg`: the wire format.
- CLAUDE.md §3 (layers) and §2 (QoS).
