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
   │                                         │  .support_contact (ADR-0092 D6)
   └───────────▶ clear_attached_payload_cells (the payload leaves the map,
   ▼                                          the attested support patch stays)
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
- `TheFieldPanelKeepsTheColumnItActuallyOccupies` — replays the
  `robocasa_drawer_utensil` geometry (face at x=+0.0614, 25 mm, 12.1 mm phase).
- `ASlabIsNeverMissedAndNeverFurtherForwardThanTheLatticesExplain` — sweeps
  all 25 phases of a 25 mm lattice.
- `EveryPhaseAndYawKeepsEveryCellTheCentreSampleWouldHaveMarked`,
  `APhaseAlignedGridRasterizesExactlyAsCentreSamplingDid`,
  `ACoarseLeafMarksEveryCellUnderIt`,
  `AnObliqueLeafMarksTheCellsItEntersAndNotTheirDiagonals`,
  `AGridOutsideTheOctreesKeyRangeStillSeesEveryLeaf`,
  `RasterizingTheKitchenStaysInsideThePublishBudget` — hold the publish
  budget to a quarter of the 10 Hz period.

## What the grid covers: a ball, sized by the robot

`coverage_radius_m` / `coverage_center_*` name a **ball** in `base_frame`,
not a box — a base-frame box would need its rotated bounding box covered
(up to 2× the cells at 45°). A ball costs `ceil(2·radius/resolution)³` cells
at every yaw.

The radius must reach wherever the safety kernel's checked geometry can go —
a property of the robot manifest, not this node: **unset, the node logs an
error and publishes nothing.**

History: the old sim box was 1.6 m (fit the kernel's then 262,144-cell cap).
Measured against `panda_mobile`'s manifest `collision_geometry`, its checked
arm reach is **1016 mm** from the grid centre — up to **124 mm** outside the
old grid. `deploy_e2e.launch.py` now covers 1.05 m and the kernel's cap is sized
to hold it.

### The ≤1-resolution inflation toward the sensor is octomap's, and stays

`insertPointCloud` marks the cell **containing the ray endpoint**, up to one
tree resolution in front of the true surface — inherent to voxel storage,
upstream of this bridge (`octomap_server`'s octree, also read directly by
Nav2/SLAM/the dashboard). Not corrected here. The direction is always safe
(reports nearer than true, never farther); pinned by
`TheFieldPanelIsNeverReportedNearerThanTheOctreeItselfSaysAtAnyPhase`.

## The attached payload leaves world occupancy here

Contract (`openral_msgs/AttachedCollisionObject`): *"The same object must be
absent from world occupancy while attached."* Before grasp, the object's
cells are honest sensor occupancy; at attach they become the robot's own
payload, already checked as collision-active attached geometry by the
kernel. Until PR #110 nothing removed them: on the 2026-08-14 acceptance run
the arm E-stopped 1.8 mm off the carried object's surface (26 cells inside
the payload's own primitives, +32 mm above the attested support plane) — the
ADR-0092 D6 support-contact witness correctly refused to exempt it (not
support contact — the payload's own stale silhouette).

**Why here, not elsewhere:**
- *Kernel*: could skip cells inside the payload's primitives, but that's an
  exemption, not a clearing — the map still carries the phantom for SLAM/the
  dashboard, and adds a per-cell escape hatch to the kernel's one
  unconditional rule.
- *HAL depth self-filter*: already makes the payload transparent to rays
  (`exclude_body_ids`), ray-clearing via OctoMap — but only cells **a ray
  still crosses**. An occluded/out-of-frustum/between-rays cell gets nothing;
  `OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt` pins
  it stays for the rest of the run. `test_depth_camera_synth.py::test_transparency_clears_only_the_payload_cells_a_ray_still_reaches`
  measures both halves against a real `MjModel` (visible half clears to a
  wall 1.9 m away; behind a counter, rays stop 0.95 m out).
- *This bridge*: already owns the lowering, re-derives the grid every frame,
  sits in the same layer as the attachment state. Clears.

**Mechanism.** Subscribes `/openral/world_state_fast` (the kernel's own
attachment source), places each object by
`FK(attach_link) · pose_in_link · pose_in_object`, zeroes every occupied cell
whose centre lies within the cell cube's **circumradius**
(`resolution·√3/2`, 21.7 mm at 25 mm cells) of a payload primitive.
`attached_clear_padding_m` adds to this reach on every frame, default **0**
(padding removes cells the payload cannot explain — protection given up).

### …and the attach transition reaches one voxel further than steady state

The circumradius reach is right for the volume the payload explains *now*,
not the pre-grasp real-object silhouette a fitted convex primitive doesn't
cover. Fit error + lattice quantization leave residue just outside one
circumradius, uncleared thereafter
(`OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt`).

Round-5 field measurement (2026-08-14): baguette r1's E-stop cell
`voxel_91633` sat **22.13 mm** from the payload primitive surface at
resolution 0.025 — **0.48 mm outside** the 21.65 mm reach.

| Grid state | Reach | Why |
|---|---|---|
| payload within `attach_sweep_padding_m` of where its **object revision** first appeared | `resolution·√3/2 + attached_clear_padding_m + attach_sweep_padding_m` (default one voxel → 46.65 mm at 25 mm cells) | stale pre-attach silhouette still under the payload |
| payload moved further | `resolution·√3/2 + attached_clear_padding_m` — unchanged | payload has left the silhouette; nearby occupancy is now evidence about the world |

The bridge re-rasterizes from the octree every tick, so a one-shot sweep
loses the residue back next tick. Round-5 timeline: attach sweep at
1786710428.38, E-stop at 1786710431.16 — **2.78 s, ~28 published grids at
10 Hz, later**, payload still where grasped.
`PayloadClearing.TheAttachWindowOutlivesTheFieldRunsTwentyEightGrids`
replays 29 consecutive grids; a one-shot sweep fails 28 of them.

Window closes once the payload has moved past `attach_sweep_padding_m`
(`TheAttachWindowClosesOnceThePayloadHasMovedAVoxel`, pinned to the
millimetre) and **latches** — does not reopen on return
(`AClosedAttachWindowDoesNotReopenWhenThePayloadComesBack`). Displacement is
measured in the **OctoMap's frame** (world-fixed), not `base_frame`.

Also pinned: `PayloadClearing.TheRoundSixFieldCellSurvivesTheSteadyReachAndGoesOnTheAttachSweep`,
`…StopsOneVoxelPastTheSteadyReach`,
`TheAttachPaddingNeverShrinksTheSteadyReach` (`attach_transition_padding`
floors negative/non-finite at 0).

**State kept.** `WorldStateStamped.attachment_revision` (producer's own
counter, bumped once per atomic `AttachmentState` change) makes
`(object_id, attachment_revision)` a payload's attachment-record identity.
`AttachSweepLedger` remembers only, per live object: that identity, the
window's anchor position, and whether it has shut — bounded by the current
attachment count. Not a cell latch; every grid is still derived fresh from
the wire. Answering and recording are the same call (`sweep`).

- Re-grasp / second payload / place-phase arm-disarm each open a window;
  carrying one payload does not (`ANewObjectRevisionOpensANewAttachWindow`).
- A release sweeps the empty set — next grasp opens a window even under an
  already-seen revision (same test).
- A frame that clears **nothing** (stale attachment state, missing TF, an
  unplaceable payload) never calls `sweep`, so it can't advance/close a
  window (`AFrameThatClearsNothingLeavesTheWindowExactlyAsItWas`).

The padded reach still withholds the attested support patch exactly as the
steady reach does
(`PayloadClearing.ThePaddedAttachSweepStillWithholdsTheAttestedSupportPatch`,
with the no-attestation counterfactual beside it).

### …except the attested support patch, which is the counter

A resting payload shares its bottom cell layer with the counter's own top
surface. Clearing by volume alone takes the counter with it:

- The kernel's support-contact witness (ADR-0092 D6,
  `update_support_contact_witnesses`) is kept alive by *occupancy*: clear the
  supporting cells and the witness declares separation from a payload that
  hasn't moved. Observed 2/2 on 2026-08-14 (baguette+counter, cup+island):
  `support_witness_separated live=0x0 was=0x1` 2.7 s after arming, ground
  truth +0.000 mm still touching, nearest surviving cell 21.77 mm out.
- SLAM/the dashboard read the same octree; the counter is real furniture.

Partition (never neither, never both):

| Cell | Owner | Outcome |
|---|---|---|
| Inside the attested patch laterally **and** in the support half-space — no higher above the attested plane than `resolution/2·(\|n.x\|+\|n.y\|+\|n.z\|) + attested depth + resolution`, unbounded below it | kernel's witness | withheld from clearing, exempted by the kernel |
| Anywhere else within a circumradius of a payload primitive | this clearing | zeroed |

The `+ resolution` term is the kernel's one voxel of co-planar headroom
(hazard log Entry 012, "Calibration 2026-08-15"), mirrored here **in
lockstep by obligation of that entry**: `withheld ⊆ exempt` holds because
both sides are the same inequality, this one with zero slack.

At 25 mm cells the withheld slab reaches at most
`21.65 mm + attested depth + 25 mm` above the plane; with the kernel's
`support_witness_max_penetration_m` of 10 mm, no acceptable attestation
withholds a cell more than 56.65 mm up
(`PayloadClearing.NoAttestationTheKernelAcceptsWithholdsACellAboveTheWidenedSlab`
pins +60 mm). `PayloadClearing.TheRoundFiveResidueCellIsWithheldInTheCoplanarBand`
pins the field's +35.8 mm cell (cleared before the calibration) as now
withheld.

`place_attached_object` lifts the wire `SupportContactWitness` through the
payload's live pose into a grid-frame `SupportPatch`; `support_patch_withholds`
mirrors the kernel's `support_contact_exempts` at **zero slack** (the kernel
adds `attached_contact_tolerance`, 1 mm), so withholding is a strict subset
of what the kernel exempts.
`PayloadClearing.WithholdingIsTheKernelsExemptionPredicateAtZeroSlack` and
`…WithholdingOnlyEverPutsOccupancyBack` pin it cell-for-cell. Two scope notes:

- **Per-object.** The kernel exempts per object
  (`check_attached_voxel_collision`); this bridge withholds per message, so
  one payload's attestation also guards a second payload's clearing
  (`PayloadClearing.OneObjectsAttestationGuardsEveryObjectsClearing`).
- **Acceptance bounds.** The kernel's lifecycle node caps acceptance —
  `support_witness_max_patch_radius_m` (0.5 m),
  `support_witness_max_penetration_m` (10 mm) — and fails the message closed
  past either; this bridge applies the geometry but not those caps
  (mirrored in the sim producer, `openral_hal._sim_attachment_evidence`).

**Phase-blind, so ADR-0097 is free here.** `place_attached_object` /
`support_patch_withholds` read only `AttachedCollisionObject.support_contact`
geometry, never `support_id`/`evidence_kind`, so ADR-0097's place-phase
witness rides this partition unchanged
(`PayloadClearing.APlacePhaseWitnessIsWithheldExactlyAsAPickPhaseOneIs`
asserts a bit-identical grid). ADR-0097's 2026-08-14 approach allowance and
its 2026-08-15 Second Amendment (`min(1.5 × voxel, 4 cm)` cap) are a kernel
margin on cells inside a declared `PlaceRegion`, not a change to which cells
exist — this bridge never reads `AttachmentState.place_declaration`.

The two predicates (bridge vs. kernel) are a deliberate cross-package
mirror — consolidating them would couple this Layer-2 node to the Layer-6
kernel's collision core — see `docs/methods/14-duplication-watch.md`, item 8.

**No latch beyond the attach window.** Withholding is derived from the wire
attestation every publish; the bridge's only memory is `AttachSweepLedger`'s
per-object attach window, which decides reach, never which cells are taken.
Hysteresis lives in the kernel: a witness that died stays dead until a new
attestation, so a lifted-and-set-down payload finds its support cells
unexempted (correct, new violation). On a genuine lift the patch (object
frame) rides with the payload and the counter cells are out of reach — the
kernel sees the separation geometrically.

**No attestation, no withholding.** `support_contact_valid == false` (honest
default for any producer that can't measure support contact, incl. the
vision attachment producer) appends no patch; cells clear as before the
witness existed. A *malformed* attestation is not downgraded to that — it
fails the whole object closed and clears nothing (same rule
`ingest_attached_objects` applies in the kernel).

**Conservative.** The payload stays collision-active attached geometry;
`check_attached_voxel_collision` keeps testing it against every remaining
occupied cell, plus `check_attached_world_collision` and
`check_attached_self_collision`. Only self-stops are removed.
`AttachedVoxelCollision.ClearingThePayloadsOwnCellsKeepsThePayloadVsWorldCheck`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) pins it: payload cells
cleared, a real obstacle 90 mm off its surface stays, kernel still stops and
reports true 40 mm clearance.

**Failure is refusal to clear.** No attachment message, one older than
`attached_state_timeout_s`, a missing `base_frame ← attach_link` TF, or a
primitive the kernel would itself reject (unknown shape tag, too few
dimensions, degenerate quaternion) — clears nothing and logs, which can only
stop the robot early.

**Detach.** Which cells clear is derived from the current attachment set on
every published grid, so the frame after detach already publishes the
payload as an obstacle again (the ledger's only detach role is forgetting
the revision). Re-marking a cell the transparency rays cleared costs
OctoMap's two-hit confirmation (~0.2 s at 10 Hz, pinned by
`test_occupancy_persistence`).

**What it does not do.** Clears this node's published grid, not
`octomap_server`'s own octree (ROS 2's `octomap_server` exposes `~/reset`
but no `clear_bbx`). A consumer reading `/octomap_binary` directly still
sees the payload's silhouette until transparency rays retire it.

**Who the residual reaches.** Issue #108 expected Nav2 to be a consumer — it
is not: Nav2's costmaps take a `sensor_msgs/LaserScan` (lidar profile) and
an `OccupancyGrid` on `/map` (visual profile,
`packages/openral_nav2_bringup/config/nav2_panda_mobile.yaml`); nothing in
the Nav2 graph subscribes `/octomap_binary`. The payload is kept out of
Nav2's world at *its* source by `openral_nav2_bringup`'s
`payload_scan_filter_node`, and into Nav2's robot by the footprint publisher
beside it. What remains on this octree is the dashboard and any future
direct octomap consumer — a display concern.

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
openral deploy sim --config scenes/sim/robocasa_panda_mobile_kitchen.yaml   # panda_mobile → auto-on
# or force it / point at a different depth topic:
openral deploy sim --config <cfg> --enable-octomap
```

Requires `ros-${ROS_DISTRO}-octomap-server` apt-installed and this package
colcon-built (both in the deploy Docker images).

### Standalone

```bash
ros2 launch openral_octomap_bridge octomap_voxel_bridge.launch.py \
    base_frame:=base_link octomap_topic:=/octomap_binary
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
| `resolution` | `0.05` | Output voxel edge length (m). |
| `box_size_{x,y,z}` | `2.0` | Local volume extent around the robot (m). |
| `box_center_{x,y,z}` | `0,0,0.5` | Local volume centre in `base_frame`. |
| `publish_rate_hz` | `10.0` | Republish rate (the grid follows the robot via TF). |
| `attached_clear_enabled` | `true` | Clear an attached payload's own cells out of the published grid. Off = pre-#110 behaviour (payload stays in the map and can stop the robot against itself). |
| `world_state_topic` | `/openral/world_state_fast` | Where the attachment set is read from — the kernel's own source. |
| `attached_clear_padding_m` | `0.0` | Extra reach beyond the cell circumradius **on every frame**, for pose uncertainty. |
| `attach_sweep_padding_m` | `resolution` (one voxel) | Extra reach **while a payload is within this distance of where its object revision first appeared** — for the stale pre-attach silhouette (round-5 `voxel_91633`, 22.13 mm out). Also the distance the payload must travel to close the window. Adds to `attached_clear_padding_m`; `0` restores the pre-#111 single reach; negative/non-finite contributes nothing. |
| `attached_state_timeout_s` | `0.5` | Attachment state older than this clears nothing. |

`size_{x,y,z} = ceil(box_size / resolution)`. Keep
`size_x*size_y*size_z ≤ world_voxel_max_cells` (kernel default 262144), or
the kernel fails closed.

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
`octomap::OcTree` — no ROS graph. A full octomap → bridge → kernel chain
needs a live OctoMap producer and is a HIL / on-robot test (`octomap`
Python bindings aren't in CI, so a synthetic OctoMap publisher would itself
need to be C++). Test list is in "How a cell becomes occupied" above.

`test_occupancy_persistence` pins the occupancy semantics the deploy-sim
launch tunes for, on a real `OcTree` seeded with `occupancy_thres 0.8`,
`sensor_model.max 0.85`, `hit 0.7`, `miss 0.4`:
- one hit is **not** a safety voxel (transient rejection), two hits are;
- a confirmed voxel survives indefinitely when **no ray ever crosses it**
  (the failure mode a lost self-filter clearing ray creates);
- one clearing ray retires it; a clearing ray never marks;
- a cleared cell still comes back when something real is there — four 10 Hz
  frames (~0.4 s) from the clamping floor.

`test_payload_clearing` pins the attached-payload half on a real `OcTree`
seeded through real rays:
- 20 cells a pre-grasp camera marked on the object reach the kernel's grid
  unchanged before the fix; the attach clears exactly those 20, leaves the
  real obstacle beside them;
- a cell 1.8 mm off the payload surface (the acceptance run's tripping cell)
  goes; one a millimetre past the circumradius stays; clearing never marks;
  it follows the payload's live pose, not a snapshot at attach; detach
  restores the object in the same frame; an unplaceable payload (unknown
  shape, short dimension list, no primitives, degenerate quaternion) clears
  nothing.
- **Partition half** (2026-08-13 baguette-run lattice phase, support cell
  centre 3.2 mm above the attested face, ~1 mm contact reads as 15.7 mm cube
  penetration): 12 payload-silhouette cells clear, 16 counter cells under
  the footprint survive; same payload with no attestation clears all 28
  (the 2026-08-14 defect mechanism); partitioned cleared-set is a strict
  subset of the un-partitioned one; depth bound holds to the millimetre
  either side of `resolution/2 + attested depth`, lateral bound either side
  of `patch_radius + circumradius`; a lift clears and withholds nothing; a
  payload rolled 90° carries its attested plane with it; a malformed
  attestation clears nothing.
- **Attach-transition window half**: the round-5 field cell at 22.13 mm
  survives the steady reach, clears at the attach reach; 29 grids replaying
  the field run's ~28-grid gap all clear it; the window closes to the
  millimetre once the payload has moved a voxel and does not re-open on
  return; a new revision / second payload / release-then-grasp each open a
  window, carrying does not; a frame clearing nothing cannot advance one;
  the padded sweep still withholds the attested patch; the widened reach
  stops one voxel past the steady one to the millimetre; no parameter value
  makes the transition clear less than an ordinary frame.

Kernel's half of the same partition:
`SupportContactWitness.ThePartitionedClearingLeavesTheWitnessItsEvidence`,
`…ClearingTheAttestedPatchKillsTheWitnessAndTheReturningCellStops`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) — change either
predicate and one of the two suites goes red.

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
