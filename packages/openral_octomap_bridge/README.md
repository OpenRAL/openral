# openral_octomap_bridge

Lowers a **3-D OctoMap** into the safety kernel's dense, base-frame
`openral_msgs/OccupancyVoxels` grid for the kernel's allocation-free
capsule-vs-voxel world-collision check.

## Why a bridge (and not the kernel)

The C++ safety kernel must stay small, auditable, and allocation-free on its
hot path, so it does **not** parse a raw OctoMap octree (octree deserialization
allocates, querying isn't time-bounded, and `octomap` is a heavy dependency).
Instead — "perception proposes, the kernel disposes" — this Layer-2 node does
the octree work off the real-time path and publishes a bounded dense grid the
kernel rasterizes capsules against.

```
octomap_msgs/Octomap (map frame)          openral_msgs/WorldStateStamped
   │  msgToMap → octomap::OcTree             │  .attached_objects
   │  tf2: octomap_frame ← base_frame        │  tf2: base_frame ← attach_link
   │  crop + rasterize the local box         │
   ▼                                         ▼
   │                                         │  .support_contact (ADR-0092 D6)
   └───────────▶ clear_attached_payload_cells (the payload leaves the map,
   ▼                                          the attested support patch stays)
openral_msgs/OccupancyVoxels (base frame, /openral/world_voxels)
   ▼
C++ safety kernel  ──  check_voxel_collision (allocation-free)
```

## The attached payload leaves world occupancy here

`openral_msgs/AttachedCollisionObject` states the contract: *"The same object
must be absent from world occupancy while attached."* Before the grasp the
object legitimately **is** world occupancy — its cells were marked by honest
sensor returns. At the attach transition they stop describing the world and
start describing the robot's own payload, which the kernel already re-checks as
collision-active attached geometry. Until PR #110 nothing removed them, and on
the 2026-08-14 acceptance run the arm E-stopped 1.8 mm off the surface of the
object it was carrying (26 further cells sat *inside* the payload's primitives),
+32 mm above the attested support plane — so the ADR-0092 D6 support-contact
witness correctly refused to exempt it. It was not support contact. It was the
payload's own stale silhouette.

**Why here.** Three places could do it:

* **The kernel** could skip cells inside the payload's primitives. That is an
  *exemption*, not a clearing: the map still carries the phantom for every other
  consumer (Nav2, SLAM, the dashboard — issue #108), and the kernel's one
  unconditional rule ("an occupied cell is an obstacle") acquires a per-cell
  escape hatch on the real-time path.
* **The HAL's depth self-filter** already makes the payload transparent to the
  rays (`exclude_body_ids`) and emits max-range clearing rays where nothing lies
  behind it, so OctoMap ray-clears the silhouette. That reaches exactly the
  cells **a ray still crosses**. A cell of the payload that is occluded by real
  geometry, outside the frustum, or simply between two rays gets no ray at all,
  and `OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt`
  pins what happens to it: nothing, for the rest of the run.
  `tests/unit/test_depth_camera_synth.py::test_transparency_clears_only_the_payload_cells_a_ray_still_reaches`
  measures both halves against a real `MjModel` — the visible half's rays reach
  the wall 1.9 m away and clear everything between; the half behind a counter
  stops 0.95 m out, half a metre short of the payload.
* **This bridge**, which already owns the lowering, re-derives the grid every
  frame, and sits in the same layer as the attachment state. It clears.

**What it does.** The bridge subscribes `/openral/world_state_fast` — the same
message the kernel ingests its attached geometry from, so bridge and kernel
always act on one attachment set — places each object by
`FK(attach_link) · pose_in_link · pose_in_object` (tf2 + the wire poses, the
kernel's own composition), and zeroes every occupied cell whose centre lies
within the cell cube's **circumradius** (`resolution·√3/2`, 21.7 mm at 25 mm
cells) of a payload primitive. That bound is the map's own discretisation slop —
the same one `support_contact_exempts` uses — so every cell the payload's volume
actually intersects goes, and the over-reach is at most one cell layer.
`attached_clear_padding_m` adds to it and is **0 by default**: padding removes
cells the payload cannot explain, which is protection given up.

### …except the attested support patch, which is the counter

A payload **resting** on a counter shares its bottom cell layer with the
counter's own top surface. A clearing that knows only the payload's volume takes
the counter with it, and that is destructive twice over:

* The kernel's support-contact witness (ADR-0092 D6) is kept alive by
  *occupancy*: `update_support_contact_witnesses` retains a witness only while
  some **occupied** cell it would exempt is still touching the payload. Clear
  those cells and the witness declares separation from a payload that has not
  moved — observed 2/2 on 2026-08-14 (baguette+counter, cup+island):
  `support_witness_separated live=0x0 was=0x1` 2.7 s after arming, ground truth
  +0.000 mm still touching, nearest surviving cell 21.77 mm out. The moment a
  support cell falls back outside the clearing reach and returns to the map, the
  unchanged physical contact E-stops with no exemption active
  (`sweep_min == min_distance`).
* Nav2, SLAM, and the dashboard read the same grid, and the counter is real
  furniture they are supposed to see.

So the two mechanisms **partition** the cells between them. Within the payload's
reach a cell is either cleared here or exempted by the kernel — never neither,
never both:

| Cell | Owner | Outcome |
|---|---|---|
| Inside the attested patch laterally, no higher above the attested plane than `resolution/2·(\|n.x\|+\|n.y\|+\|n.z\|) + attested depth` | the **kernel's witness** | withheld from the clearing, exempted by the kernel, and still there for the latch to measure |
| Anywhere else within a circumradius of a payload primitive — the payload's own silhouette, the residue above the support plane, the +32 mm class the acceptance round tripped on | this **clearing** | zeroed |

`place_attached_object` lifts the wire `SupportContactWitness` (stated in the
attached object's own frame) through the payload's live pose into a grid-frame
`SupportPatch`, and `support_patch_withholds` is the kernel's own
`support_contact_exempts` geometry with **zero slack**. That asymmetry is
deliberate: the kernel adds `attached_contact_tolerance` (1 mm of physical
FK/pose slack) to the same bound, so what this bridge withholds is a strict
subset of what the kernel exempts, and no withheld cell can ever be the one that
stops the robot. The two predicates are a deliberate cross-package mirror —
consolidating them would make this Layer-2 node link the Layer-6 kernel's
collision core — and must be changed in lockstep
(`docs/methods/14-duplication-watch.md`, item 8).

**Withholding only ever puts occupancy back.** It can only *skip* a clear, so
the cells removed with the partition are a strict subset of the cells removed
without it (`PayloadClearing.WithholdingOnlyEverPutsOccupancyBack`). Relative to
the un-partitioned clearing this change moves cases from no-stop to stop, never
the reverse.

**No latch here.** The bridge is stateless per frame, as the rest of the
clearing is: withholding is derived from the attestation on the wire every time
the grid is published. Hysteresis lives in the kernel, where it belongs — a
witness that died stays dead until World State attests a new contact, so a
payload lifted and set back down finds its support cells in the map and
unexempted, which is the new violation it should be. On a genuine lift the patch
rides up with the payload (its geometry is in the object frame) and the counter
cells are out of the payload's reach anyway, so they are neither cleared nor
withheld: the kernel sees the separation by geometry, not by anything this
bridge does.

**No attestation, no withholding.** `support_contact_valid == false` — the
honest default for any producer that cannot measure support contact, including
the vision attachment producer — appends no patch, and the payload's cells clear
exactly as they did before the witness existed. A *malformed* attestation is not
downgraded to that: it fails the whole object closed and clears nothing at all,
the same rule `ingest_attached_objects` applies in the kernel.

**Why this is conservative.** The payload does not stop being checked — it
remains collision-active attached geometry, and `check_attached_voxel_collision`
keeps testing it against every *remaining* occupied cell, plus
`check_attached_world_collision` and `check_attached_self_collision`. What the
clearing removes is the robot being stopped by *itself*.
`AttachedVoxelCollision.ClearingThePayloadsOwnCellsKeepsThePayloadVsWorldCheck`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) pins exactly that: with the
payload's own cells cleared and a real obstacle cell 90 mm off its surface left
in place, the kernel still stops, still names that cell, and still reports its
true 40 mm clearance.

**Failure is refusal to clear.** No attachment message, one older than
`attached_state_timeout_s`, a missing `base_frame ← attach_link` TF, or a
primitive the kernel itself would reject (unknown shape tag, too few dimensions,
degenerate quaternion) clears **nothing at all** and logs. The map then keeps
occupancy it should not have, which can only stop the robot early.

**Detach.** The clearing carries no state: it is derived from the attachment set
on every published grid, so the first frame after the object leaves that set is
already the frame that publishes it as an obstacle again. Re-*marking* a cell the
transparency rays cleared costs OctoMap's two-hit confirmation (~0.2 s at 10 Hz),
which is the map's own latency and is pinned by `test_occupancy_persistence`.

**What it does not do.** It clears the grid this node publishes, not
`octomap_server`'s own octree: ROS 2's `octomap_server` exposes `~/reset` but no
`clear_bbx` service, and resetting the whole map at every grasp would discard
every real obstacle in it. So a consumer reading `/octomap_binary` directly still
sees the payload's silhouette until the transparency rays retire it (issue #108).

## The frame contract (ADR-0095)

`OccupancyVoxels` carries **no TF of its own**. `header.frame_id` names the
frame `origin` is measured in, and the kernel rasterizes its FK'd link capsules
against the grid *directly* — no transform is applied on the way in. So the
grid's **content** and the kernel's **collision FK root** must be expressed in
one and the same frame, and that frame must be the one `base_frame` denotes on
`/tf`. Three parties have to agree:

| Party | What it must use |
|---|---|
| whatever feeds `octomap_server` (the sim HAL's depth cloud, a real RGB-D driver) | an extrinsic measured against the body `base_frame` denotes |
| this bridge | `base_frame` as `header.frame_id`, with `origin` in that frame |
| the kernel | per-link `origin_xyz` measured from that same `base_frame` |

On robosuite/RoboCasa mobile manipulators that is not the MJCF chassis root.
The OmronMobileBase stacks a ground-level root (`mobilebase0_base`, world z 0)
under a 0.70 m pedestal whose top plate (`mobilebase0_support`) carries the arm
and the robot-mounted cameras — and `base_link` on `/tf` is the **pedestal
top**, because `MobileBaseBridge` publishes `odom -> base_link` from the HAL's
`base_pose_6dof()` (robosuite's `robot0_base_pos`). Measuring the sim camera
extrinsic against the chassis root instead made the grid's *content*
world-referenced while its *label* said `base_link`, so every TF consumer (Nav2,
SLAM, the dashboard) saw obstacles a pedestal too high. The kernel agreed with
the grid only because PR #103 had pushed the same 0.70 m into its FK root, which
cancelled the error for the kernel alone.

ADR-0095 fixes it at the source: `openral_hal.depth_cloud.resolve_base_frame_body_name`
resolves the arm-mount body for every `base_frame -> …` extrinsic, and the
`panda_mobile` manifests' `panda_joint1` origin is the plain Franka 0.333 m
again. **Both halves must land together** — a half-applied change leaves the
kernel a full pedestal away from the obstacles it is checked against (hazard
HZ-0095-1). `VoxelCollision.BaseFrameAlignmentPreservesTheProtectiveEnvelope`
and `VoxelCollision.HalfAppliedFrameAlignmentMovesTheKernelByThePedestal`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) pin both facts, and
`tests/unit/test_voxel_grid_frame_alignment.py` pins the producer side against a
real `MjModel`.

## Run

### Integrated (recommended) — via `openral deploy sim`

`openral deploy sim --enable-octomap` brings up the whole world-collision leg in
one graph: octomap_server (from the HAL's depth `PointCloud2`), this bridge,
and the kernel's capsule-vs-voxel check (`world_voxel_enabled:=true`). It
**auto-enables** when the robot manifest declares a depth `SensorSpec`:

```bash
openral deploy sim --config scenes/sim/robocasa_panda_mobile_kitchen.yaml   # panda_mobile → auto-on
# or force it / point at a different depth topic:
openral deploy sim --config <cfg> --enable-octomap
```

Requires `ros-${ROS_DISTRO}-octomap-server` apt-installed and this package
colcon-built (both are in the deploy Docker images).

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
| `attached_clear_enabled` | `true` | Clear an attached payload's own cells out of the published grid. Off = the pre-#110 behaviour (the payload stays in the map and stops the robot against itself). |
| `world_state_topic` | `/openral/world_state_fast` | Where the attachment set is read from — the kernel's own source. |
| `attached_clear_padding_m` | `0.0` | Extra reach beyond the cell circumradius, for pose uncertainty. Every millimetre here removes cells the payload cannot explain. |
| `attached_state_timeout_s` | `0.5` | Attachment state older than this clears nothing. |

`size_{x,y,z} = ceil(box_size / resolution)`. Keep
`size_x*size_y*size_z ≤ world_voxel_max_cells` (kernel default 262144), or the
kernel fails closed.

## Producing the upstream OctoMap

The bridge consumes an `octomap_msgs/Octomap`; the canonical producer is
`octomap_server` (`ros-${ROS_DISTRO}-octomap-server`, bundled in both the dev and
inference images alongside `octomap` / `octomap-msgs` / `tf2-geometry-msgs`). It
builds the octree from a **3-D depth point cloud** (`sensor_msgs/PointCloud2`):

```bash
ros2 run octomap_server octomap_server_node \
    --ros-args -p resolution:=0.05 -p frame_id:=map -r cloud_in:=/camera/points
```

## Testing

`test_octree_to_grid` unit-tests the rasterization core (`rasterize_octree_to_grid`)
against a real `octomap::OcTree` — no ROS graph needed. A full
octomap → bridge → kernel chain needs a live OctoMap producer and is a
HIL / on-robot test (the `octomap` Python bindings aren't in CI, so a synthetic
OctoMap publisher would itself be C++).

`test_occupancy_persistence` pins the occupancy semantics the deploy-sim launch
tunes for, on a real `OcTree` seeded with those exact parameters
(`occupancy_thres 0.8`, `sensor_model.max 0.85`, `hit 0.7`, `miss 0.4`):

* one hit is **not** a safety voxel (the transient rejection the raised
  threshold exists for) and two hits are;
* a confirmed voxel survives indefinitely when **no ray ever crosses it** — the
  failure mode a lost self-filter clearing ray creates, since the robot's own
  silhouette then contributes no ray at all and the phantom reaches the kernel's
  grid;
* one clearing ray retires it, and a clearing ray never *marks*;
* a cleared cell still comes back when something real is there — four 10 Hz
  frames (~0.4 s) from the clamping floor, which is the measured cost of the
  clearing behaviour, not an assumption.

`test_payload_clearing` pins the attached-payload half on the same real
`OcTree`, seeded through real rays: the 20 cells a pre-grasp camera marked on
the object reach the kernel's grid unchanged before the fix; the attach clears
exactly those 20 and leaves the real obstacle beside them; a cell 1.8 mm off the
payload surface (the acceptance run's tripping cell) goes, one a millimetre past
the circumradius stays; clearing never marks; it follows the payload's live pose
rather than snapshotting the attach instant; detach restores the object to the
grid in the same frame; and a payload the bridge cannot place — unknown shape,
short dimension list, no primitives, degenerate quaternion — clears nothing at
all.

Its second half pins the **partition** on a payload resting on a counter, on the
lattice phase of the 2026-08-13 baguette run (the support cell's centre 3.2 mm
above the attested face, so a ~1 mm contact reads as 15.7 mm of cube
penetration): the 12 payload-silhouette cells clear and the 16 counter cells
under the footprint survive; the same payload with no attestation clears all 28,
which is the mechanism of the 2026-08-14 defect; the partitioned cleared-set is
a strict subset of the un-partitioned one; the depth bound holds to the
millimetre either side of `resolution/2 + attested depth` and the lateral bound
either side of `patch_radius + circumradius`; a lift clears and withholds
nothing at all; a payload rolled 90° carries its attested plane with it (the
normal is rotated, the point translated); and a malformed attestation clears
nothing at all. The kernel's half of the same partition is
`SupportContactWitness.ThePartitionedClearingLeavesTheWitnessItsEvidence` and
`…ClearingTheAttestedPatchKillsTheWitnessAndTheReturningCellStops`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) — change either predicate
and one of the two suites goes red.

**Sim status.** The sim target is `scenes/sim/robocasa_panda_mobile_kitchen.yaml`
— a mobile manipulator in a cluttered RoboCasa kitchen, the only scene with real
3-D obstacles and an obstacle-avoidance task. The deploy-sim HAL now publishes a
**depth `PointCloud2`** for it: the panda_mobile node ray-casts each depth
`SensorSpec` (`robots/panda_mobile/robot.yaml` → `front_depth`) from MuJoCo via
`openral_sim.backends.depth_camera.synthesize_depth_frame` (one cast per frame,
back-projected to a cloud — raster **and** self-filter clearing mask — by
`openral_hal.depth_cloud.points_from_depth_grid`) and publishes
`/openral/cameras/front_depth/points` (camera optical frame) + a live
`base_link → front_depth_optical_frame` TF. This is robot-agnostic — declare a
depth `SensorSpec` (with `metadata.mjcf_camera`) on any robot and its HAL node
publishes the same — so the full chain is:

```
panda_mobile HAL  ──/openral/cameras/front_depth/points──▶  octomap_server
   (depth ray-cast)                                              │ /octomap_binary
                                                                 ▼
   openral_octomap_bridge  ──/openral/world_voxels──▶  C++ safety kernel
```

Run `octomap_server` against the published cloud (see *Producing the upstream
OctoMap* above) and this bridge against its output. Booting the full RoboCasa
kitchen + octomap_server is a HIL / on-host integration step (RoboCasa assets +
GPU render); the synth, packing, and TF are unit-tested
(`tests/unit/test_depth_camera_synth.py`, `tests/unit/test_depth_cloud_helpers.py`)
and the kernel-side voxel check by
`tests/sim/safety/test_kernel_voxel_collision_synthetic.py` (feeds
`OccupancyVoxels` directly). The frame this chain is expressed in is pinned by
`tests/unit/test_voxel_grid_frame_alignment.py` and, against the live kitchen,
by `tests/sim/safety/test_panda_mobile_robocasa_collision_fk.py` — see *The
frame contract* above.
