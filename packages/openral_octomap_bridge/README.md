# openral_octomap_bridge

Lowers a **3-D OctoMap** into the safety kernel's dense
`openral_msgs/OccupancyVoxels` grid for the kernel's allocation-free
capsule-vs-voxel world-collision check.

The grid is published **on the OctoMap's own lattice**, one cell per octree
cell, with the rotation carried in `OccupancyVoxels.orientation` (not
base-aligned, since 2026-08-25 — see "How a cell becomes occupied").

Field-run numbers cited below (round-5's grid timeline, the 2026-08-13
lattice phase, `robocasa_drawer_utensil`) are catalogued in
[`docs/reference/collision-validation-evidence.md`](../../docs/reference/collision-validation-evidence.md).

## Why a bridge (and not the kernel)

The C++ safety kernel stays allocation-free on its hot path, so it does not
parse a raw OctoMap octree (deserialization allocates, querying isn't
time-bounded, `octomap` is a heavy dependency). This Layer-2 node does the
octree work off the real-time path and publishes a bounded dense grid the
kernel rasterizes capsules against ("perception proposes, the kernel
disposes"):

```
octomap_msgs/Octomap (map frame)          openral_msgs/WorldStateStamped
   │  msgToMap → octomap::OcTree             │  .attached_objects
   │  tf2: octomap_frame ← base_frame        │  tf2: base_frame ← attach_link
   │  cover a ball around the robot,         │
   │  on the octree's own lattice            │
   ▼                                         ▼
   │                                         │  .tracked_objects
   └───────────▶ identity-validated clearing (only bound track cells)
openral_msgs/OccupancyVoxels (origin+orientation in base_frame,
                              cell axes the octree's, /openral/world_voxels)
   ▼
C++ safety kernel  ──  check_voxel_collision (allocation-free)
```

## How a cell becomes occupied: the grid's lattice IS the octree's

One published cell is one octree cell — same size, same phase, same axes.
`origin` + `orientation` on the wire (`OccupancyVoxels`) give the pose, in
`header.frame_id`, of cell (0,0,0)'s minimum corner; **the cell axes are
`orientation`'s, not `header.frame_id`'s**. Consumers apply it; nobody
re-rasterizes.

### Why not a base-aligned grid

Two rejected rules, in order:

1. **Centre point-query** (until 2026-08-16): mark a base cell when its centre
   point-queried occupied — a single sample across two independently-phased
   lattices. On `robocasa_drawer_utensil` (25 mm cells, ~12.1 mm measured
   phase) a cabinet door face at base x = +0.0614 came out occupied at
   x ∈ [0.025, 0.050) — a full voxel closer to the robot, with **zero cells
   where the panel actually is**.
2. **Overlap**: mark every base cell whose cube shares volume with an
   occupied leaf's — the minimum sound cover for a base-aligned output. Sound,
   but costs a dilation: [issue #173](https://github.com/OpenRAL/openral/issues/173)
   measured **29–35 mm median extra reach on 25 mm cells, 40 mm worst case**,
   holding **48% of the live start-state E-stops**. Two mitigations were
   measured and rejected: snapping the grid origin to the octree's at yaw≈0
   (half a degree of yaw restores the full 25 mm — sim-only), and shrinking
   the effective leaf half-extent (0.75× buys 12% of the dilation but leaves
   9.9% of the leaf uncovered).

Since the dilation belongs to the format, the format changed: grid-on-octree-
lattice has no phase and no yaw to be exact about. Cost is proportional to
occupied leaves rather than cell count (integer-arithmetic marking vs. a
separating-axis test per cell), so it can only be cheaper than the overlap
rule's measured 0.26–0.30 ms.

Tests (`test_octree_to_grid`, no ROS graph, real `octomap::OcTree`):
- `TheGridIsTheOctreeCellForCellAtEveryPhaseAndYaw` — no obstacle lost, no
  reach surrendered, over swept phases/yaws.
- `TheFieldPanelIsNeverReportedNearerThanTheOctreeItselfSaysAtAnyPhase` —
  a face is never reported nearer than the octree places it, at any phase.
- `TheOrientationIsTheOctreeToBaseRotationAndIsAlwaysAUnitQuaternion`,
  `AnUnplaceableLatticePublishesNoCellsRatherThanWrongOnes`,
  `AnAbsurdSpecIsRefusedRatherThanAllocated`,
  `ACoarseLeafMarksEveryCellUnderIt`,
  `AGridStraddlingTheOctreesKeyRangeStillSeesEveryLeaf`.
- `RasterizationCostAcrossTreeResolutions`,
  `RasterizingTheKitchenStaysInsideThePublishBudget` — hold the publish
  budget to a quarter of the 10 Hz period.

## A stopped octree stops the grid (fail-closed on a dead camera)

The bridge re-rasterizes the **last** octree it received on a
`publish_rate_hz` timer, so between octrees the grid follows the robot
through TF and identity-validated clearing follows the tracked object pose.
`octomap_server` publishes `/octomap_binary` only when it inserts a cloud,
so a dead camera is a silent octree. Until hazard log Entry 033 the timer
kept republishing that frozen map, stamped `now()`, forever; the kernel times
voxel freshness from **receipt**, so its fail-closed `DROP_VOXEL_UNAVAILABLE`
never fired and anything entering the workspace afterwards was invisible
(Thor, 2026-09-24: ZED driver stopped, `/octomap_binary` silent,
`/openral/world_voxels` still arriving at 7.1 Hz).

The republish is now bounded by `max_octree_age_s`, measured from the
octree's **receipt** on this node's clock (the clock the kernel times voxel
freshness on; the octree's header stamp is the sensor's and may sit on
another clock domain). Past the bound the node publishes **nothing** — not
even a payload-cleared grid, which would be a fresh-looking stale map — and
logs a throttled WARN. The kernel's own `world_voxel_deadline_ms` then turns
the silence into `DROP_VOXEL_UNAVAILABLE` (a drop, not a latch). The next
octree resumes publication automatically (one INFO line). Worst case from
the last inserted cloud to the kernel's drop: `max_octree_age_s +
world_voxel_deadline_ms` (1.0 s + 1.0 s with the deploy defaults).

Sizing the bound:

- It must exceed octomap's normal inter-publish gap with margin (0.25–0.31 s
  measured on the Thor ZED path at 3.2–4.0 Hz, but ~0.45 s in one Thor run at
  2.2 Hz; ≤ 0.33 s in sim with the depth cast slowed to ~3 Hz) and must not
  exceed the kernel's deadline, so the kernel, not the bridge, fails closed.
- Both are per-rig `DeployRuntime` fields: `world_voxel_deadline_s` (default
  1.0 s) and `max_octree_age_s` (default equal to the deadline; a value above
  it is refused by the schema and by `deploy_e2e.launch.py`). An earlier
  half-deadline bound (0.5 s) silenced a healthy camera's grid at 2.2 Hz; a
  source slower than about 1 Hz needs both raised in its scene, up to the
  schema's hard cap of 2.0 s on the deadline (2x its pre-2026-09-25 default;
  above it the scene is refused at load).
- Too small only costs availability: silence shorter than the kernel's
  deadline is not a drop. A non-finite or non-positive bound publishes nothing
  (ERROR at start-up).

Stamps:

- A grid's `header.stamp` is still `now()`: when the grid was placed in
  `base_frame` (the TF it was rasterized with). Nothing reads it as data age
  (the kernel and World State time grids from receipt), and the HAL's evidence
  decoder (`openral_hal.sim_sensor_bridge`) picks "the grid current at time t"
  by it, which the octree's own stamp would break.
- The octree's own stamp rides on every grid as `source_stamp` (capture stamp
  of the newest cloud inserted), and the kernel budgets the world's age from
  it: `world_voxel_data_age_budget_ms`, from the per-rig
  `DeployRuntime.world_voxel_data_age_budget_s` (default 1.5 s, the Thor ZED-M
  tail; hard cap 3.0 s); past it the chunk drops as `voxel_stale`.
- The kernel enforces the caps itself: with `world_voxel_enabled` it refuses to
  configure on a `world_voxel_deadline_ms` outside (0, 2000] or a
  `world_voxel_data_age_budget_ms` outside (0, 3000] (its own default is 1500;
  0 does not disable the check), so a kernel started with `ros2 run` is held
  to the same limits as a validated scene.

`robot_self_filter` removes the robot's own returns on robot-visible camera paths
(real depth and Isaac-rendered depth)
before `octomap_server` inserts the cloud: it poses the kernel's collision
parameters at the cloud's capture stamp and drops returns within `padding_m`
of those robot primitives (from `DeployRuntime.robot_self_filter_padding_m`,
provisional 0.02 m, hard cap 0.10 m). It filters attached-object returns only
when `identity_clearing_track` validates one unique `tracked_object` binding
against the fresh tracked set in the robot root frame. Region, segmentation,
pinch-fallback, unknown, stale, or ambiguous attachment geometry leaves the
sensor returns in the cloud. No pose at the capture stamp drops the whole cloud.

The identity check ties the mask to one tracked object, but does not establish
per-point sensor ownership. A foreign surface overlapping a validated track
shape may still have its returns suppressed; typed occupancy provenance remains
unfinished.
The node enforces the cap itself: a `padding_m` outside [0, 0.10] (or
non-finite) logs an ERROR and forwards nothing.

Joint positions are linearly interpolated at the capture stamp between the two
samples around it. The frame is refused (dropped, counted, WARN rate-limited)
when those samples are more than two `joint_state_period_s` apart or the stamp
needs more than one period of extrapolation past the newest; a refused frame
inserts nothing. Each robot primitive's padding grows with its speed:
`padding_m + |v|·velocity_padding_dt_s`, `|v|·δt` the primitive's motion over
the interpolated joint rate within δt_sync (default 0.02 s: the stamp error
interpolation cannot see; 0.5 m/s adds 10 mm). The total stays under the 0.10 m
cap: a link faster than that is under-padded, which leaves its points in the
map (a false stop) rather than erasing more of the world.

**Recent sweep** (issue #418, `trailing_sweep_s`, default 0 = off; the deploy
launch sets 0.2 s in sim only, Safety-WG review pending, hazard row owed). Each
primitive is also padded by how far it has moved since `stamp − trailing_sweep_s`
(the same history and interpolation), taking the larger of that and its forward
motion, under the same 0.10 m cap. A return inside the volume the robot just left is
then the robot's own trailing or silhouette-edge return. Spark R16: two returns of the
Robotiq finger edge, 15–21 mm outside its hull, landed 0.13 s after the gripper left
that cell; OctoMap held them, and the finger stopped on that mid-air voxel 2.4 s later.
The world there is blind to new returns for at most `trailing_sweep_s` after the robot
leaves. Obstacles already in the map are never removed. With no joint state that far
back (a fresh start) the padding is the forward motion alone. A value outside
[0, `kMaxTrailingSweepS` = 0.5] logs an ERROR and forwards nothing.

**Robot mask** (issue #418, off by default). With `robot_mask_enabled`, the node
also renders the UNPADDED model into each named depth camera at the depth
stamp and publishes `/openral/cameras/<cam>/robot_mask` (mono8: 0 free, 128
uncertain, 255 robot; the depth image's header; BEST_EFFORT, KEEP_LAST 5). A
pixel is robot iff the rendered robot depth is at most the measured depth +
5 mm (so a held object in front of a finger stays free) and the return is not seen
past the robot (a return more than 3 cm behind the rendered front and more than 10 mm
outside every link is not robot: a convex hull or 26-DOP spans a gripper's open jaw,
#418), or its back-projected
return is inside the unpadded hand (`robot_mask_hand_links`) or within 10 mm of
any other link. Pixels within `robot_mask_uncertain_px` of a robot/non-robot
boundary, and robot-covered pixels with no return, are uncertain. Each
primitive is rasterised over its projected bounding box only; a hulled box is
rendered as its 26-DOP (which contains the hull). The deploy launch turns it on
with `robot_mask_cameras:=top,wrist` (only where the self-filter runs).
RGB-only cameras get no mask yet.

A box that carries the kernel's tight geometry (`collision_box_hull` /
`collision_hull_*`, lowered from `LinkCollisionGeometry.tight_geometry`) is
filtered against that geometry, not the box: the box's slack (12–27 mm mean on
the Panda links, more at the corners) would otherwise widen the blind shell.
The distance is the kernel's staged narrow phase with the voxel cube shrunk to
a point — the 26-DOP slab bound, then GJK on the exact hull's vertices with an
exhaustive support scan — so every value is a lower bound on the true distance
(never an over-report: a return the hull explains is always removed) and the
converged value is exact to 1e-9 m. `test_self_filter` pins it against the
kernel's `hull_cell_distance` on panda_mobile's real hulls. A hull the kernel's
`validate_tight_hull` chain would not prove (vertex outside its DOP, DOP outside
its box, over 320 vertices, bad arity) refuses the whole model, so the node
forwards nothing. Cost, dev laptop, panda_mobile, 230 400-point cloud with 10 %
of it within 8 cm of the arm: 2.4–2.6 ms box-only → 7.0–8.0 ms with hulls
(the extra is GJK on the points inside a box's shell; not yet measured on Thor).

Pinned by `test_bridge_staleness` (the real node in-process: publishes while
fresh, silent past the bound for as long as the silence lasts, resumes on
the next octree, keeps flowing at Thor's 3.2 Hz cadence, publishes nothing
under an unusable bound; plus the `octree_is_fresh` boundaries) and, end to
end against the real kernel,
`tests/sim/safety/test_kernel_voxel_bridge_staleness.py` (certifies while
octrees arrive, `DROP_VOXEL_UNAVAILABLE` with no latch or E-stop once they
stop, certifies again when they resume). It was checked by hand to
discriminate: with the bridge bound set to 1e9 (pre-fix behaviour) the stale
chunk is certified, reproducing the fail-open. The committed suite does not
carry that control.

**Not covered:** a camera that keeps publishing garbage, or a frozen image
re-published by a driver, still reaches `octomap_server` as fresh inserts.

## Publish rate: measure it RELIABLE

The grid goes out on the `publish_rate_hz` timer (10 Hz) whenever the octree
is fresh. At the then-fixed real-arm deploy geometry (0.02 m, radius 1.05 m) it is
106³ = 1 191 016 cells, a **1.19 MB** sample. A `BEST_EFFORT` subscriber
loses such a fragmented sample whole whenever it misses one fragment, so a
best-effort `ros2 topic hz` or rate probe under-reads it; the safety kernel
subscribes `RELIABLE` and does not. On the Orin OpenArm cell (mock hardware,
2026-09-26, 240 s) the same run read **10.01 Hz, max gap 0.36 s** on a reliable
subscriber, 7.0 Hz (max gap 1.1 s) on a best-effort one, and 5.7 Hz on a
best-effort all-topics probe (a 4.4–6.1 Hz reading of that probe the day
before was briefly mistaken for a slow bridge).

The per-octree work is small beside the period: the real Orin map (14k leaves,
~4.6k occupied cells in the ball, replayed from recorded ZED clouds) costs
0.7 ms to deserialize and 0.3 ms to rasterize on a dev laptop. Deserializing
walks the **whole** map, not the ball, so it grows with the mapped volume
(~60 ms at 437k leaves); rasterizing is bounded by the ball's bbx query.
`test_bridge_rate_budget` pins both: deserialize + rasterize of a ray-cast
map under half the period, and the real node reaching a reliable
`KEEP_LAST(1)` subscriber at more than 7 Hz with the deploy-sized grid.

## What the grid covers: a ball, sized by the robot

`coverage_radius_m` / `coverage_center_*` name a **ball** in `base_frame`,
not a box — a base-frame box would need its rotated bounding box covered
(up to 2× the cells at 45°). A ball costs `ceil(2·radius/resolution)³` cells
at every yaw.

The radius must reach wherever the safety kernel's checked geometry can go —
a property of the robot manifest, not this node: **unset, the node logs an
error and publishes nothing.**

The old sim box was 1.6 m (fit the kernel's then 262,144-cell cap).
Measured against `panda_mobile`'s manifest `collision_geometry`, its checked
arm reach is **1016 mm** from the grid centre — up to **124 mm** outside the
old grid. `deploy_e2e.launch.py` then covered a fixed 1.05 m about `(0, 0, 0.5)`,
which left the OpenArm's hanging arms reaching 108 mm below it (its `base_frame`
is the torso top). The launch now measures the ball per robot (`_coverage_ball`): over the
joint limits, against the same collision model the kernel is handed, the centre is
the middle of the moving links' reach box and the radius the farthest reach from
it plus 50 mm (OpenArm 0.871 m about `(0.003, 0, 0)`, panda_mobile 1.04 m about
`(0, 0, 0.432)`, franka_panda 1.089 m about `(0, 0.001, 0.432)`). The kernel's cap
is derived from that radius, and a ball this node would refuse (`kMaxCells`) is
refused at launch instead.

### The ≤1-resolution inflation toward the sensor is octomap's, and stays

`insertPointCloud` marks the cell **containing the ray endpoint**, up to one
tree resolution in front of the true surface — inherent to voxel storage,
upstream of this bridge (`octomap_server`'s octree, also read directly by
Nav2/SLAM/the dashboard). Not corrected here. The direction is always safe
(reports nearer than true, never farther); pinned by
`TheFieldPanelIsNeverReportedNearerThanTheOctreeItselfSaysAtAnyPhase`.

## Attached objects and world occupancy

The kernel continues checking every occupied world voxel against robot and
attached-object geometry. The bridge removes world occupancy only for an
attachment with `evidence_kind == "tracked_object"` when
`identity_clearing_track` finds exactly one matching track with status
`bound`, matching `epoch` and `attach_link`, a finite `floor_z`, no conflicting
track or payload on the link, and a usable tracked-object set in `base_frame`.
The current-pose grid clear uses the track's bound geometry and tight hull when
available; malformed or ambiguous geometry clears nothing. Identity sweeps,
vacated boxes, and the short release window use the same track binding and the
configured `payload_sweep_clear_service`.

Attachments evidenced by `grasp_target_region`, `vision_segmentation`,
`pinch_fallback`, gripper closure, unknown evidence, or an ambiguous tracked
binding remain occupied in the world grid. Primitive geometry, attachment-link
transforms, and support-contact witnesses do not establish which occupied cells
belong to an object, so they do not authorize removal. The attachment revision
in `OccupancyVoxels` acknowledges processing of a fresh state revision and any
required identity-clearing work; it does not certify that every attached object's
cells were removed or that asynchronous octree clear requests completed. The kernel
still checks remaining cells normally.

**Consequence for a deploy without the object tracker (decided posture: fail closed).**
Only the #418 object tracker (`DeployRuntime.object_tracking: tracking` or `on`, sim-only
until Safety-WG review) publishes `tracked_object` attachments, so identity clearing
applies to its bindings in both modes; its `pinch_fallback` holds stay mapped. With `object_tracking` `off`
— every real arm today, the MuJoCo digital twin (`MujocoArmHAL`) and an Isaac
sidecar scene — the attachment the HAL's vision leg or the sim bridge publishes is
geometry-only, so nothing the arm holds is cleared
from the world grid, nor filtered from the self-filter's cloud: the payload stays a
mapped obstacle and the kernel may stop the arm on the object in its own hand, during
the carry and at the place. That stop is the accepted cost. There is deliberately no
pose-based fallback clear, because a pose does not say which cells are the payload's.
A run that must carry an object with world collision on needs `object_tracking: tracking`
or `on` (sim) until a real-cell tracker posture is reviewed. (A MuJoCo `SimAttachedHAL` scene is
the exception that does not need the bridge: it masks the payload bodies out of its
rendered depth, so they never enter the map.)

Identity binding is not per-cell occupancy ownership. The bridge constrains
clearing to a tight track shape and excludes the floor, declared static points,
and other tracked objects, but an untracked obstacle touching or overlapping
that shape can still be removed. Typed per-cell contribution provenance is
unfinished; this residual remains part of the identity-clearing hazard. See
`identity_clearing.hpp`, `test_identity_clearing`, and
`BridgeIdentitySweep.*` for the current bounds and tests.

## The frame contract (ADR-0095)

`OccupancyVoxels` carries **no TF of its own**. `header.frame_id` names the
frame `origin` is measured in; the kernel rasterizes FK'd link capsules
against the grid directly, no transform applied. So the grid's content and
the kernel's collision FK root must share one frame — the one `base_frame`
denotes on `/tf`:

| Party | What it must use |
|---|---|
| whatever feeds `octomap_server` (sim HAL depth cloud, real RGB-D driver) | an extrinsic measured against the body `base_frame` denotes |
| this bridge | `base_frame` as `header.frame_id`, `origin` in that frame |
| the kernel | per-link `origin_xyz` measured from that same `base_frame` |

On robosuite/RoboCasa mobile manipulators that is not the MJCF chassis root:
the OmronMobileBase stacks a ground-level root (`mobilebase0_base`, world
z 0) under a 0.70 m pedestal whose top plate (`mobilebase0_support`) carries
the arm/cameras, and `base_link` on `/tf` is the **pedestal top**
(`MobileBaseBridge` publishes `odom -> base_link` from the HAL's
`base_pose_6dof()` / robosuite's `robot0_base_pos`). Measuring the sim
camera extrinsic against the chassis root instead made the grid content
world-referenced while labelled `base_link`; every TF consumer saw obstacles
a pedestal too high. The kernel agreed with the grid only because PR #103
had pushed the same 0.70 m into its FK root (cancelling the error for the
kernel alone).

ADR-0095 fix: `openral_hal.depth_cloud.resolve_base_frame_body_name`
resolves the arm-mount body for every `base_frame -> …` extrinsic;
`panda_mobile`'s `panda_joint1` origin is the plain Franka 0.333 m again.
**Both halves must land together** — a half-applied change leaves the
kernel a full pedestal away from the obstacles it's checked against (hazard
HZ-0095-1). Pinned by
`VoxelCollision.BaseFrameAlignmentPreservesTheProtectiveEnvelope` and
`…HalfAppliedFrameAlignmentMovesTheKernelByThePedestal`
(`cpp/openral_safety_kernel/test/test_collision.cpp`), and
`tests/unit/test_voxel_grid_frame_alignment.py` on the producer side.

## Run

### Integrated (recommended) — via `openral deploy sim`

`openral deploy sim --enable-octomap` brings up the whole world-collision leg:
octomap_server (from the HAL's depth `PointCloud2`), this bridge, the
kernel's capsule-vs-voxel check (`world_voxel_enabled:=true`). **Auto-enables**
when the robot manifest declares a depth `SensorSpec`:

```bash
openral deploy sim --config scenes/deploy/robocasa_pnp.yaml   # panda_mobile → auto-on
# or force it / point at a different depth topic:
openral deploy sim --config <cfg> --enable-octomap
```

Requires `ros-${ROS_DISTRO}-octomap-server` apt-installed and this package
colcon-built (both in the deploy Docker images).

### Standalone

```bash
ros2 launch openral_octomap_bridge octomap_voxel_bridge.launch.py \
    base_frame:=base_link octomap_topic:=/octomap_binary \
    coverage_radius_m:=<robot reach + margin>   # required: 0 publishes nothing
```

and launch the kernel with `world_voxel_enabled:=true` (plus
`world_voxel_max_cells` ≥ the grid's `size_x*size_y*size_z`).

Requires TF from `base_frame` into the OctoMap's `header.frame_id` (usually
`map`) — published by your SLAM / localization stack.

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `base_frame` | `base_link` | Output frame; the kernel expects obstacles here. |
| `octomap_topic` | `/octomap_binary` | Input `octomap_msgs/Octomap`. |
| `output_topic` | `/openral/world_voxels` | Output `OccupancyVoxels`. |
| `resolution` | `0.05` | Compatibility parameter; output lattice and cell size come from the input octree. |
| `coverage_radius_m` | `0.0` | Local volume radius around the robot (m). |
| `coverage_center_{x,y,z}` | `0.0, 0.0, 0.5` | Local volume centre in `base_frame`. |
| `publish_rate_hz` | `10.0` | Republish rate (the grid follows the robot via TF). |
| `max_octree_age_s` | `1.0` | Stop publishing once the last octree was received longer ago than this, so the kernel's `world_voxel_deadline_ms` fails closed on a dead camera. Well above octomap's normal gap, not above the kernel's deadline (`deploy_e2e.launch.py`: equal to it). Non-finite, ≤ 0, or above the kernel's 2.0 s deadline cap (`kMaxOctreeAgeS`) publishes nothing. |
| `static_points_file` | `""` | Base-frame surface points of declared static geometry, marked occupied in every published grid. Only adds occupancy; empty = off; a file that fails to load is fatal. |
| `attached_clear_enabled` | `true` | Enable identity-validated tracked-object clearing. Attachments with region, segmentation, pinch, unknown, or ambiguous evidence stay occupied. |
| `world_state_topic` | `/openral/world_state_fast` | Where the attachment set is read from — the kernel's own source. |
| `attached_state_timeout_s` | `0.5` | Attachment state older than this clears nothing. |
| `payload_sweep_clear_service` | `""` (off) | octomap_server `~/clear_bbox`; identity-validated track sweeps, vacated boxes, and release-window clearing only. Geometry-only attachment evidence never authorizes octree clearing. |
| `payload_sweep_clear_period_s` | `0.2` | Period for identity-validated octree clearing requests. |
| `attach_link_tf_frames` | `[]` (none) | `"link=frame"` renames for the TF lookup of a payload's attach link: the published `attach_link` is the manifest's name (the kernel's collision model uses manifest names), but a real cell's TF tree may name that body differently (OpenArm: `openarm_left_link7` is `openarm_left_ee_base_link`). Unmapped links are looked up as themselves. `deploy_e2e.launch.py` passes the scene's `vision_attachment.tf_frames` — the strings the HAL gets as `vision_attachment_tf_frames` — and nothing when the leg is off. A malformed or conflicting entry refuses the whole mapping (logged ERROR): every link is then looked up by its own name and, on such a cell, its payload stays in the map. An unmapped link whose lookup fails logs a one-time hint to set this. |

`size_x = size_y = size_z ≤ floor(2·coverage_radius_m / resolution + 1/2) + 1` (the half
cell is the snap onto the octree's lattice). Keep
`size_x*size_y*size_z ≤ world_voxel_max_cells` (kernel default 614125 = 85³, 1.05 m at 25 mm
cells; `deploy_e2e.launch.py` derives it from the robot's ball), or
the kernel fails closed.

### `robot_self_filter`

The package's second executable (`robot_self_filter`, robot-visible camera path) takes
`cloud_in` and publishes `cloud_out` (+ `cloud_out_rgb` for viewers). Parameters:
the kernel's collision model (`collision_*`), `padding_m` (`0.02`),
`max_joint_state_skew_s` (`0.1`), `joint_state_period_s` (`0.05`; the launch
normally passes `1 / action_spec.control_freq_hz`). On Isaac sim camera paths
the filter starts with no admitted period and drops clouds until the connected
HAL sends its actual simulation tick through `joint_state_period_s`; an invalid
or missing update leaves it closed. That accepted update clears joint history.
`joint_state_period_from_hal` is launch-owned and enabled only for this path.
Real and non-Isaac paths retain the manifest cadence. Nanosecond bounds round up
to the smallest integer count that represents the configured period; no timing
margin is added. `velocity_padding_dt_s` (`0.02`), `trailing_sweep_s` (`0.0`),
`tf_timeout_s` (`0.05`), `joint_state_history_s`
(`2.0`), `attached_filter_enabled` (`true`, unique tracked-object bindings only), `attached_state_timeout_s` (`0.5`),
`joint_states_topic` (`/joint_states`), `world_state_topic`
(`/openral/world_state_fast`), and the robot mask's `robot_mask_enabled`
(`false`), `robot_mask_cameras` / `robot_mask_depth_topics` /
`robot_mask_camera_info_topics` / `robot_mask_topics` (parallel, `[]`),
`robot_mask_hand_links` (`[]`), `robot_mask_uncertain_px` (`3`). With no
collision model, or a period / δt_sync out of range, it forwards nothing.

## Producing the upstream OctoMap

The canonical producer is `octomap_server`
(`ros-${ROS_DISTRO}-octomap-server`, bundled in dev + inference images with
`octomap` / `octomap-msgs` / `tf2-geometry-msgs`), built from a 3-D depth
point cloud (`sensor_msgs/PointCloud2`):

```bash
ros2 run octomap_server octomap_server_node \
    --ros-args -p resolution:=0.05 -p frame_id:=map -r cloud_in:=/camera/points
```

## Testing

`test_octree_to_grid` unit-tests `rasterize_octree_to_grid` against a real
`octomap::OcTree` — no ROS graph. Test list is in "How a cell becomes
occupied" above. `test_bridge_staleness` runs the real node in-process
(see "A stopped octree stops the grid"). The octree → bridge → kernel
chain's liveness is covered by
`tests/sim/safety/test_kernel_voxel_bridge_staleness.py`, which publishes
an all-free octree in octomap's binary encoding from Python; a chain with
a live OctoMap producer (camera → `octomap_server`) is still HIL.

`test_occupancy_persistence` pins the occupancy semantics the deploy-sim
launch tunes for, on a real `OcTree` seeded with `occupancy_thres 0.8`,
`sensor_model.max 0.85`, `hit 0.7`, `miss 0.4`:
- one hit is **not** a safety voxel (transient rejection), two hits are;
- a confirmed voxel survives indefinitely when **no ray ever crosses it**
  (the failure mode a lost self-filter clearing ray creates);
- one clearing ray retires it; a clearing ray never marks;
- a cleared cell still comes back when something real is there — four 10 Hz
  frames (~0.4 s) from the clamping floor.

The real-node bridge tests pin the current removal boundary:
- `BridgePayloadSweep.*` verifies region, segmentation, unknown, pinch-fallback,
  and gripper-closure evidence preserve occupied cells, issue no `clear_bbox`
  request, and still publish the processed attachment revision.
- `BridgeAttachLinkTf.TheMappingDoesNotAuthorizeGeometryOnlyClearing` and
  `BridgeLiftedPayload.ASupportWitnessDoesNotAuthorizeGeometryOnlyClearing`
  verify that a valid link mapping or support witness alone preserves occupancy.
- `BridgeIdentitySweep.*` verifies identity-bound carry backfill, keep-outs,
  revision acknowledgement, unavailable-state handling, vacated boxes, and the
  release window. This identity path still removes selected occupancy and retains
  the residual ownership risk described above.

**Sim status.** Target scene: `scenes/sim/robocasa_panda_mobile_kitchen.yaml`
(mobile manipulator, cluttered RoboCasa kitchen, only scene with real 3-D
obstacles + obstacle-avoidance task). Deploy-sim HAL publishes a depth
`PointCloud2` for it: the panda_mobile node ray-casts each depth `SensorSpec`
(`robots/panda_mobile/robot.yaml` → `front_depth`) from MuJoCo via
`openral_sim.backends.depth_camera.synthesize_depth_frame` (one cast/frame,
back-projected by `openral_hal.depth_cloud.points_from_depth_grid` — raster
**and** self-filter clearing mask), publishing
`/openral/cameras/front_depth/points` (camera optical frame) + a live
`base_link → front_depth_optical_frame` TF. Robot-agnostic: any robot
declaring a depth `SensorSpec` (with `metadata.mjcf_camera`) gets the same.

```
panda_mobile HAL  ──/openral/cameras/front_depth/points──▶  octomap_server
   (depth ray-cast)                                              │ /octomap_binary
                                                                 ▼
   openral_octomap_bridge  ──/openral/world_voxels──▶  C++ safety kernel
```

Run `octomap_server` against the published cloud (see *Producing the
upstream OctoMap*) and this bridge against its output. Booting the full
RoboCasa kitchen + octomap_server is a HIL / on-host integration step
(RoboCasa assets + GPU render); the synth, packing, and TF are unit-tested
(`tests/unit/test_depth_camera_synth.py`, `tests/unit/test_depth_cloud_helpers.py`),
the kernel-side voxel check by
`tests/sim/safety/test_kernel_voxel_collision_synthetic.py` (feeds
`OccupancyVoxels` directly). The frame contract is pinned by
`tests/unit/test_voxel_grid_frame_alignment.py` and, against the live
kitchen, by `tests/sim/safety/test_panda_mobile_robocasa_collision_fk.py`
(see *The frame contract* above).

Robot-mask depth arriving before its camera TF is retained for nonblocking
exact-source-stamp retries every 5 ms. Each camera holds at most two images;
oldest overflow and a steady-clock wait reaching `max_joint_state_skew_s` refuse
that mask with diagnostics. At most 16 cameras and 32 MiB / 8 megapixels per image
are admitted. CameraInfo is captured with the image. No latest-TF substitution,
extra joint extrapolation, or payload masking is used; late TF after expiry
cannot resurrect an image.

### Capture-time joint-state arrival

Clouds and robot-mask images wait for joint history to cover their capture stamp
(exact sample or interpolation), rather than extrapolating a removal pose. Cloud
retention is bounded to two buffers of at most 32 MiB each; a shared 5 ms timer
retries only within the existing `max_joint_state_skew_s` wall deadline. Overflow
and expiry refuse the sensor frame. Existing interpolation gap/skew checks,
TF requirements and padding are unchanged.

The recorded R9 fixture reproduces 3/16 gripper returns surviving forward
extrapolation; all 16 disappear when the matching state arrives 5.1 ms later.
World-point preservation and real ROS delayed-arrival/expiry/overflow cases
guard the bounded wait. This is not permission to remove unknown objects.
