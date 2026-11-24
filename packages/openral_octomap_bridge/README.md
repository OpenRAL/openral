# openral_octomap_bridge

Lowers a 3-D OctoMap into the safety kernel's dense `openral_msgs/OccupancyVoxels` grid for
its allocation-free capsule-vs-voxel world-collision check. The package ships three
executables:

| Executable | Role |
|---|---|
| `octomap_voxel_bridge` | OctoMap to `/openral/world_voxels` (this robot's world). |
| `robot_self_filter` | Removes the robot's own returns from a depth cloud before `octomap_server` inserts it; optionally publishes a robot mask. |
| `peer_voxels` | The other robots of the scene as `/openral/peer_voxels` (ADR-0137). |

The grid is published on the OctoMap's own lattice, one cell per octree cell, with the
rotation carried in `OccupancyVoxels.orientation` (not base-aligned). Field-run numbers are
catalogued in
[`docs/reference/collision-validation-evidence.md`](../../docs/reference/collision-validation-evidence.md).

## Why a bridge (and not the kernel)

The C++ safety kernel stays allocation-free on its hot path, so it does not parse a raw
octree (deserialisation allocates, queries are not time-bounded, `octomap` is heavy). This
Layer-2 node does the octree work off the real-time path and publishes a bounded dense grid
("perception proposes, the kernel disposes"):

```
octomap_msgs/Octomap (map frame)          openral_msgs/WorldStateStamped
   | msgToMap -> octomap::OcTree             | .attached_objects, .tracked_objects
   | tf2: octomap_frame <- base_frame        | tf2: base_frame <- attach_link
   | cover a ball around the robot,          |
   | on the octree's own lattice             |
   v                                         v
   +-----------> identity-validated clearing (only bound track cells)
openral_msgs/OccupancyVoxels (origin + orientation in base_frame, /openral/world_voxels)
   v
C++ safety kernel: check_voxel_collision
```

## Run it

Normally through the CLI, which wires octomap_server, this bridge and the kernel's
`world_voxel_enabled:=true`. It turns on by itself when the robot manifest declares a depth
`SensorSpec`:

```bash
uv run openral deploy sim --config scenes/deploy/robocasa_pnp.yaml   # panda_mobile: auto-on
uv run openral deploy sim --config <cfg> --enable-octomap            # force it
```

Needs `ros-${ROS_DISTRO}-octomap-server` and this package colcon-built (both are in the deploy
Docker images). Standalone:

```bash
ros2 launch openral_octomap_bridge octomap_voxel_bridge.launch.py \
    base_frame:=base_link octomap_topic:=/octomap_binary \
    coverage_radius_m:=<robot reach + margin>   # required: 0 publishes nothing
```

The standalone launch exposes `base_frame`, `octomap_topic`, `output_topic`, `resolution`,
`coverage_radius_m`, `coverage_center_z`, `publish_rate_hz` and `max_octree_age_s`; set other
parameters on the node directly. Launch the kernel with `world_voxel_enabled:=true` and
`world_voxel_max_cells` at least the grid's `size_x*size_y*size_z`. The bridge needs TF from
`base_frame` into the OctoMap's `header.frame_id` (usually `map`) from your SLAM or
localisation stack. The upstream producer is `octomap_server`:

```bash
ros2 run octomap_server octomap_server_node \
    --ros-args -p resolution:=0.05 -p frame_id:=map -r cloud_in:=/camera/points
```

## `octomap_voxel_bridge`

| Direction | Topic | Type |
|---|---|---|
| Sub | `octomap_topic` (`/octomap_binary`) | `octomap_msgs/Octomap` |
| Sub | `world_state_topic` (`/openral/world_state_fast`) | `WorldStateStamped` (attachment set and tracked objects: the kernel's own source) |
| Pub | `output_topic` (`/openral/world_voxels`) | `OccupancyVoxels` (RELIABLE; the kernel subscribes RELIABLE) |
| Client | `payload_sweep_clear_service` | octomap_server `~/clear_bbox` (`BoundingBoxQuery`) |

| Param | Default | Meaning |
|---|---|---|
| `base_frame` | `base_link` | Output frame; the kernel expects obstacles here. |
| `octomap_topic` / `output_topic` | `/octomap_binary` / `/openral/world_voxels` | Input and output topics. |
| `resolution` | `0.05` | Compatibility only; lattice and cell size come from the input octree. |
| `coverage_radius_m` | `0.0` | Radius of the covered ball around the robot. Unset: the node logs an error and publishes nothing. |
| `coverage_center_{x,y,z}` | `0, 0, 0.5` | Ball centre in `base_frame`. |
| `publish_rate_hz` | `10.0` | Republish rate (the grid follows the robot via TF). |
| `max_octree_age_s` | `1.0` | Stop publishing once the last octree was received longer ago than this. Must be in (0, `kMaxOctreeAgeS` = 2.0]; otherwise nothing is published. |
| `static_points_file` | `""` | Base-frame surface points of declared static geometry, marked occupied in every grid. Only adds occupancy; a file that fails to load is fatal. |
| `attached_clear_enabled` | `true` | Enable identity-validated tracked-object clearing. |
| `tracked_clear_enabled` | `false` | Fail closed: no track drives a clear unless set. The launch sets it for `object_tracking: tracking` and `on`. The attachment revision is acknowledged either way. |
| `attached_state_timeout_s` | `0.5` | Attachment state older than this clears nothing. |
| `payload_sweep_clear_service` | `""` (off) | octomap_server `~/clear_bbox`, used for identity-validated sweeps, vacated boxes and the release window only. |
| `payload_sweep_clear_period_s` | `0.2` | Period of those clear requests. |
| `released_clear_timeout_s` | `3.0` | Release window; must equal the tracker's and the kernel's `tracked_release_timeout_s` (the launch passes one value). |
| `attach_link_tf_frames` | `[]` | `"link=frame"` renames for the TF lookup of a payload's attach link, when the real TF tree names the body differently from the manifest (OpenArm: `openarm_left_link7` is `openarm_left_ee_base_link`). The launch passes the scene's `vision_attachment.tf_frames`. A malformed or conflicting entry refuses the whole mapping (ERROR) and the payload stays in the map. |

`size_x = size_y = size_z <= floor(2*coverage_radius_m / resolution + 1/2) + 1`. Keep the
product within `world_voxel_max_cells` (kernel default 614125 = 85^3; the launch derives it
from the robot's ball) or the kernel fails closed.

### The grid's lattice is the octree's

One published cell is one octree cell: same size, phase and axes. `origin` plus `orientation`
give the pose, in `header.frame_id`, of cell (0,0,0)'s minimum corner; the cell axes are
`orientation`'s, not the frame's. Consumers apply it; nobody re-rasterises. This replaced two
base-aligned rules: centre point-query (a single sample across two lattices, which put a
panel a full voxel nearer with no cells where it is) and overlap (sound, but dilated the
reach by 29-35 mm median on 25 mm cells and held 48% of live start-state e-stops). Cost is
proportional to occupied leaves. Pinned by `test_octree_to_grid` (real `octomap::OcTree`, no
ROS graph: no obstacle lost and no reach surrendered at any phase or yaw, a unit orientation,
absurd specs refused, a publish-budget bound of a quarter of the 10 Hz period).

The up-to-one-resolution inflation toward the sensor is octomap's (`insertPointCloud` marks
the cell holding the ray endpoint). It always reports nearer than true, never farther, and is
not corrected here.

### What the grid covers: a ball sized by the robot

`coverage_radius_m` / `coverage_center_*` name a ball in `base_frame` (a rotated box would
need up to 2x the cells). The radius must reach wherever the kernel's checked geometry can
go, which is a property of the robot manifest. `deploy_e2e.launch.py` measures it per robot
(`_coverage_ball`): over the joint limits, against the same collision model the kernel gets,
the centre is the middle of the moving links' reach box and the radius is the farthest reach
from it plus 50 mm (OpenArm 0.871 m about `(0.003, 0, 0)`, panda_mobile 1.04 m about
`(0, 0, 0.432)`). The kernel's cap derives from that radius; a ball this node would refuse
(`kMaxCells`) is refused at launch.

### A stopped octree stops the grid (fail closed on a dead camera)

The bridge re-rasterises the last octree on the `publish_rate_hz` timer, so between octrees
the grid follows the robot through TF. `octomap_server` publishes only when it inserts a
cloud, so a dead camera is a silent octree. The kernel times voxel freshness from receipt, so
a frozen map republished with a fresh stamp would hide that.

Republishing is bounded by `max_octree_age_s`, measured from the octree's receipt on this
node's clock. Past it the node publishes nothing (not even a payload-cleared grid) and logs a
throttled WARN; the kernel's `world_voxel_deadline_ms` then drops chunks as
`DROP_VOXEL_UNAVAILABLE` (a drop, not a latch). The next octree resumes publishing. Worst case
from the last cloud to the drop is `max_octree_age_s + world_voxel_deadline_ms` (1 s + 1 s at
the deploy defaults).

- The bound must exceed octomap's normal inter-publish gap (0.25-0.45 s measured on the Thor
  ZED path at 2.2-4 Hz) and not exceed the kernel's deadline, so the kernel fails closed. Too
  small only costs availability.
- Both are per-rig `DeployRuntime` fields: `world_voxel_deadline_s` (default 1.0 s, hard cap
  2.0 s) and `max_octree_age_s` (default the deadline; a larger value is refused by the schema
  and the launch). A source slower than about 1 Hz needs both raised in its scene.
- `header.stamp` is `now()` (when the grid was placed in `base_frame`; the HAL's evidence
  decoder picks "the grid current at time t" by it). The octree's own capture stamp rides on
  every grid as `source_stamp`, and the kernel budgets the world's age from it:
  `world_voxel_data_age_budget_ms` from `DeployRuntime.world_voxel_data_age_budget_s`
  (default 1.5 s, cap 3.0 s); past it the chunk drops as `voxel_stale`.
- The kernel enforces the caps itself: with `world_voxel_enabled` it refuses to configure on
  a deadline outside (0, 2000] ms or a data-age budget outside (0, 3000] ms.

Not covered: a camera that keeps publishing garbage, or a driver re-publishing a frozen
image, still reaches `octomap_server` as fresh inserts. Pinned by `test_bridge_staleness`
(real node in-process) and `tests/sim/safety/test_kernel_voxel_bridge_staleness.py` (against
the real kernel).

### Publish rate: measure it RELIABLE

At a 0.02 m, 1.05 m-radius deploy geometry the grid is 106^3 = 1.19 M cells, a 1.19 MB
sample. A BEST_EFFORT subscriber loses a fragmented sample whole if it misses one fragment,
so a best-effort `ros2 topic hz` under-reads it (7 Hz against 10 Hz reliable on the Orin
OpenArm cell). The kernel subscribes RELIABLE. Per-octree work is small: a 14k-leaf map
costs 0.7 ms to deserialise and 0.3 ms to rasterise; deserialising walks the whole map, so
it grows with mapped volume (about 60 ms at 437k leaves). `test_bridge_rate_budget` pins
both.

## Attached objects and world occupancy

The kernel checks every occupied world voxel against robot and attached-object geometry. The
bridge removes world occupancy only for an attachment with `evidence_kind == "tracked_object"`
when `identity_clearing_track` finds exactly one matching track with status `bound`, matching
`epoch` and `attach_link`, a finite `floor_z`, no conflicting track or payload on the link,
and a usable tracked-object set in `base_frame`. The clear uses the track's bound geometry and
tight hull when available; malformed or ambiguous geometry clears nothing. Identity sweeps,
vacated boxes and the release window use the same binding and `payload_sweep_clear_service`.

Attachments evidenced by `grasp_target_region`, `vision_segmentation`, `pinch_fallback`,
gripper closure, unknown evidence or an ambiguous binding stay occupied. Primitive geometry,
link transforms and support witnesses do not say which cells belong to an object. The
attachment revision in `OccupancyVoxels` acknowledges a fresh state revision and any
identity-clearing work; it does not certify that every payload cell was removed.

**Without the object tracker the posture is fail closed.** Only the tracker
(`DeployRuntime.object_tracking: tracking` or `on`, sim-only) publishes `tracked_object`
attachments. With `object_tracking: off` (every real arm, the MuJoCo twin, an Isaac sidecar
scene) the payload stays a mapped obstacle, so the kernel may stop the arm on the object in
its own hand during carry and place. That is the accepted cost; there is deliberately no
pose-based fallback clear. A run that carries an object with world collision on needs the
tracker. (A MuJoCo `SimAttachedHAL` scene masks payload bodies out of its rendered depth and
does not need this.)

Identity binding is not per-cell ownership: an untracked obstacle touching a tracked shape can
still be removed. The bridge limits clearing to a tight track shape and excludes the floor,
declared static points and other tracked objects. See `identity_clearing.hpp`,
`test_identity_clearing` and `BridgeIdentitySweep.*`.

## `robot_self_filter`

Runs on robot-visible camera paths (real depth and Isaac-rendered depth). It takes `cloud_in`
and publishes `cloud_out` (and `cloud_out_rgb` for viewers). It poses the kernel's collision
parameters at the cloud's capture stamp and drops returns within `padding_m` of those
primitives before `octomap_server` inserts the cloud. It filters attached-object returns only
when `identity_clearing_track` validates one unique `tracked_object` binding against the fresh
tracked set. Region, segmentation, pinch, unknown, stale or ambiguous geometry leaves the
returns in. No pose at the capture stamp drops the whole cloud. A foreign surface overlapping
a validated track shape may still have its returns suppressed.

| Parameter | Default | Meaning |
|---|---|---|
| `collision_*` | none | The kernel's collision model. With none, the node forwards nothing. |
| `padding_m` | `0.02` | Padding; must be in [0, 0.10] (hard cap) or nothing is forwarded. |
| `max_joint_state_skew_s` | `0.1` | Wall deadline for waiting on joint history. |
| `joint_state_period_s` | `0.05` | Joint-state period; the launch passes `1 / action_spec.control_freq_hz`. |
| `joint_state_period_from_hal` | `false` | Launch-owned, Isaac camera paths only: the filter drops clouds until the HAL reports its real simulation tick through `joint_state_period_s`. |
| `velocity_padding_dt_s` | `0.02` | The sync error interpolation cannot see; each primitive grows by `\|v\|*dt`. |
| `trailing_sweep_s` | `0.0` | Also pad by motion since `stamp - trailing_sweep_s` (max `kMaxTrailingSweepS` = 0.5). The deploy launch sets 0.2 s in sim only. |
| `tf_timeout_s`, `joint_state_history_s` | `0.05`, `2.0` | TF wait and joint history length. |
| `attached_filter_enabled`, `tracked_clear_enabled` | `true`, `false` | Drop a bound payload's points only when both are true. |
| `attached_state_timeout_s` | `0.5` | |
| `joint_states_topic`, `world_state_topic` | `/joint_states`, `/openral/world_state_fast` | |
| `base_frame`, `collision_base_dofs` | `""`, `[]` | A mobile manipulator's planar base dofs (the kernel's own), held at zero so the model is posed in `base_frame` and every TF lookup starts there. See below. |
| `robot_mask_*` | off | The robot mask (below). |

Joint positions are linearly interpolated at the capture stamp. A frame is refused (dropped,
counted, WARN rate-limited) when the samples around it are more than two
`joint_state_period_s` apart or it needs more than one period of extrapolation. Clouds and
mask depth images wait for joint history to cover their capture stamp rather than
extrapolating; retention is two buffers of at most 32 MiB each and a 5 ms retry timer within
`max_joint_state_skew_s`. Only a joint-state message carrying at least one of the model's
posed joints adds a history sample. Total padding stays under the 0.10 m cap: a faster link
is under-padded, which leaves its points in the map (a false stop) rather than erasing more
of the world.

Mobile bases: the model is posed in `base_frame`, so no `world` or fleet map need exist in TF
and each robot under its own `odom` filters like a fixed arm. Base dofs without a
`base_frame` forward nothing, and so does a `base_frame` that is not the end of the model's
planar base chain (identity-origin fixed or plain revolute / prismatic links, no calibration
curve, moving on exactly `collision_base_dofs`, with no other link using a base dof). The
launch refuses the same manifest first (`_collision_base_dofs`). Base joints are never read
from a joint state, and a moving base adds no velocity padding.

Boxes that carry the kernel's tight geometry (`collision_box_hull` / `collision_hull_*`) are
filtered against that geometry, not the slack box, using the kernel's staged narrow phase (a
26-DOP slab bound, then GJK on the hull vertices). Every value is a lower bound on the true
distance, so a return the hull explains is always removed. `test_self_filter` pins it against
the kernel's `hull_cell_distance`. A hull the kernel's `validate_tight_hull` would refuse
(vertex outside its DOP, over 320 vertices, bad arity) refuses the whole model.

**Recent sweep.** With `trailing_sweep_s > 0` each primitive is also padded by how far it has
moved since `stamp - trailing_sweep_s`, so a return inside the volume the robot just left is
read as its own trailing edge. The world is blind to new returns there for at most that long;
obstacles already mapped are never removed.

**Robot mask** (`robot_mask_enabled`, off by default; the launch sets
`robot_mask_cameras:=top,wrist` only where the self-filter runs). The node renders the
unpadded model into each named depth camera at the depth stamp and publishes
`/openral/cameras/<cam>/robot_mask` (mono8: 0 free, 128 uncertain, 255 robot; depth header;
BEST_EFFORT, KEEP_LAST 5). A pixel is robot when the rendered robot depth is at most the
measured depth + 5 mm and the return is not seen past the robot, or its back-projected return
is inside the unpadded hand (`robot_mask_hand_links`) or within 10 mm of another link. Pixels
within `robot_mask_uncertain_px` (3) of a boundary, and robot pixels with no return, are
uncertain. Parameters: `robot_mask_cameras`, `robot_mask_depth_topics`,
`robot_mask_camera_info_topics`, `robot_mask_topics` (parallel lists), `robot_mask_hand_links`.
RGB-only cameras get no mask. Depth that arrives before its camera TF is retried every 5 ms
for the same stamp; each camera holds at most two images, up to 16 cameras and 32 MiB / 8 MP
per image are admitted, and late TF after expiry cannot resurrect an image.

## `peer_voxels` (several robots in one scene, ADR-0137)

Publishes every other robot of the scene, from its own geometry, as `/openral/peer_voxels`
(`OccupancyVoxels`, RELIABLE, KEEP_LAST 1) for this robot's kernel. Each peer runs on its own
ROS domain: the node joins each peer's domain with a context of its own and reads only that
peer's `/joint_states` and `/tf` (`world -> map -> odom -> base`); it never publishes there.
The launch lowers each peer's collision model from its manifest as parameters under
`peer_<name>.`, from `OPENRAL_SCENE_PEERS`.

Fail closed: while any peer's joints (each on its own stamp) or base pose, or this robot's own
base pose, is missing, older than `max_input_age_s`, or stamped further ahead than that,
nothing is published, and the kernel's receipt deadline refuses every chunk. A peer is never
assumed gone. The launch refuses a several-robot scene whose kernel has no world-voxel check.

| Param | Default | Meaning |
|---|---|---|
| `peer_names`, `peer_domains` | `[]` | Parallel lists; empty or unequal lengths publish nothing. |
| `base_frame`, `world_frame` | `base_link`, `world` | Output frame; frame joining the robots. |
| `coverage_center_{x,y,z}`, `coverage_radius_m`, `resolution` | `0, 0, 0.5`; `1.05`; `0.025` | The grid ball and cell size. |
| `sweep_horizon_s`, `max_input_age_s` | `0.3`, `0.3` | Motion swept per peer, and input age bound; `sweep_horizon_s >= max_input_age_s > 0`. The launch sets both from the kernel's peer data-age budget. |
| `pad_m` | `0.02` | Padding on each peer primitive. |
| `publish_period_s` | `0.05` | Publish period. |

Tested by `test_peer_grid`.

## The frame contract (ADR-0095)

`OccupancyVoxels` carries no TF of its own. `header.frame_id` names the frame `origin` is
measured in, and the kernel rasterises FK'd link capsules against the grid with no transform.
So the grid's content and the kernel's collision FK root must share one frame, the one
`base_frame` denotes on `/tf`:

| Party | What it must use |
|---|---|
| whatever feeds `octomap_server` (sim HAL depth cloud, real RGB-D driver) | an extrinsic measured against the body `base_frame` denotes |
| this bridge | `base_frame` as `header.frame_id`, `origin` in that frame |
| the kernel | per-link `origin_xyz` measured from that same `base_frame` |

On robosuite / RoboCasa mobile manipulators `base_link` on `/tf` is the pedestal top, not the
MJCF chassis root; `openral_hal.depth_cloud.resolve_base_frame_body_name` resolves the
arm-mount body for every `base_frame -> ...` extrinsic. Both halves must land together: a
half-applied change leaves the kernel a full pedestal away from the obstacles it checks
(hazard HZ-0095-1). Pinned by
`VoxelCollision.BaseFrameAlignmentPreservesTheProtectiveEnvelope` and
`...HalfAppliedFrameAlignmentMovesTheKernelByThePedestal`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) and
`tests/unit/test_voxel_grid_frame_alignment.py`.

## Tests

C++ gtests under `test/`: `test_octree_to_grid`, `test_bridge_staleness`,
`test_bridge_rate_budget`, `test_occupancy_persistence`, `test_identity_clearing`,
`test_payload_clearing`, `test_bridge_payload_sweep`, `test_bridge_lifted_payload`,
`test_bridge_attach_link_tf`, `test_self_filter`, `test_robot_self_filter_node`,
`test_peer_grid`.

- `test_occupancy_persistence` pins the occupancy semantics the deploy launch tunes for
  (`occupancy_thres 0.8`, `sensor_model.max 0.85`, `hit 0.7`, `miss 0.4`): one hit is not a
  safety voxel and two are; a confirmed voxel survives when no ray crosses it; one clearing
  ray retires it and never marks; a cleared cell returns after about four 10 Hz frames.
- The removal boundary: `BridgePayloadSweep.*` (region, segmentation, unknown, pinch-fallback
  and gripper-closure evidence keep occupied cells and issue no `clear_bbox`, yet still
  acknowledge the revision), `BridgeAttachLinkTf.TheMappingDoesNotAuthorizeGeometryOnlyClearing`,
  `BridgeLiftedPayload.ASupportWitnessDoesNotAuthorizeGeometryOnlyClearing`, and
  `BridgeIdentitySweep.*` (identity-bound carry backfill, keep-outs, vacated boxes and the
  release window).
- Producer side: `tests/unit/test_depth_camera_synth.py`, `tests/unit/test_depth_cloud_helpers.py`.
  Kernel side: `tests/sim/safety/test_kernel_voxel_collision_synthetic.py` (feeds
  `OccupancyVoxels` directly) and `tests/sim/safety/test_panda_mobile_robocasa_collision_fk.py`.
  The self-filter fixtures in `test/data/` replay recorded joint-state arrival cases. A chain
  with a live camera and `octomap_server` is still HIL.
