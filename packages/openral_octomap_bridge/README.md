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

## How a cell becomes occupied: overlap, not sampling

A base cell is marked occupied when **its cube shares volume with an occupied
octree leaf's cube**. Not when its centre point-queries occupied — that was the
rule until 2026-08-16, and it is a sampling rule applied across two independent
lattices:

* the octree's lattice is fixed in the OctoMap's frame (`map` / `odom`);
* the grid's is fixed in `base_frame` and rides the robot.

Their relative phase is therefore whatever the robot's pose makes it, and a
single sample per cell snaps every surface onto whichever lattice the centre
happened to land on. On the `robocasa_drawer_utensil` field run — 25 mm cells,
a measured phase of ~12.1 mm, almost exactly half a cell — a cabinet door
panel whose true front face is at base x = +0.0614 came out with its outermost
occupied column at x ∈ [0.025, 0.050), **a full voxel closer to the robot**, and
**zero cells where the panel actually is**. The best rigid shift between the
recorded map and a ground-truth rasterization of the same camera at the same
pose was exactly (−1, 0, 0) cells. The z axis was benign on that run only by
luck of its own phase.

Overlap has no phase. Every cell the occupied volume enters is marked, for every
relative phase and (on a mobile base) every relative yaw — the cell-vs-leaf test
is an exact separating-axis test between the two cubes, so a yawed lattice is
not bounded by its axis-aligned box either.

**It only ever adds cells.** A cell centre inside a leaf is a cube overlapping
that leaf, by at least half a cell on every axis, so nothing centre sampling
marked can fall out. `OctreeToGrid.EveryPhaseAndYawKeepsEveryCellTheCentreSample
WouldHaveMarked` sweeps phases and yaws over a real kitchen octree and asserts
exactly that containment, against the old predicate transcribed into the test.

**And it does not dilate.** Overlap is *strict*: cubes that merely touch share
no volume and do not mark each other, so a grid whose lattice is in phase with
the octree's rasterizes bit-for-bit as it did before
(`APhaseAlignedGridRasterizesExactlyAsCentreSamplingDid`). What it costs is half
a grid cell of extra forward reach: the frontmost marked cell's centre can now
sit up to `tree resolution + half a grid cell` in front of a surface, where the
centre rule's bound was one tree resolution.

Cost went **down**, because the work is now proportional to the occupied leaves
in the box rather than to its cell count: on the reference host, a 2 m box over
a kitchen octree fell from 2.3 ms to 0.26 ms at 50 mm cells, and from 15.2 ms to
0.30 ms at 25 mm (64000 / 512000 cells, 6036 occupied leaves).
`RasterizingTheKitchenStaysInsideThePublishBudget` holds it to a quarter of the
10 Hz publish period.

### The ≤1-resolution inflation toward the sensor is octomap's, and stays

The grid inherits one more source of forward error that this node does **not**
correct: `insertPointCloud` marks the cell **containing the ray endpoint**, and
that cell already reaches up to one tree resolution in front of the true
surface. It is inherent to storing a surface in a voxel lattice at all, it is
upstream of this bridge (`octomap_server`'s octree, which Nav2 / SLAM / the
dashboard read directly too), and correcting it here would mean second-guessing
the map with sensor geometry this node does not have.

So: a published grid can report a surface up to one tree resolution + half a
grid cell nearer than it is, and that is the map's discretisation, not a bug in
the lowering. Chasing it belongs upstream, in how the octree is built. The
direction is the safe one — the robot stops early, never late — and the payload
clearing's reaches (below) are measured against the same lattice slop.

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

### …and the attach transition reaches one voxel further than steady state

The reach above is exactly right for the cells the payload's volume explains
*now*. It is not enough for the cells that describe the payload from *before*
the grasp. Those were marked by a sensor that saw the **real object**, while the
clearing measures against a **fitted convex primitive**: fit error plus lattice
quantization leaves a thin residue of pre-attach silhouette just outside one
circumradius, and nothing else removes it either — an occluded cell gets no
clearing ray, and `OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt`
pins that it then stays for the rest of the run.

The 2026-08-14 **round-5** forensics measured it: baguette r1's E-stop cell
`voxel_91633` sat **22.13 mm** from the payload's own primitive surface at
resolution 0.025 — **0.48 mm outside** the 21.65 mm reach, so it was never a
clearing candidate at all. It was not the counter, not a real obstacle, and not
support contact. It was the payload's own stale silhouette, one lattice step out
of the clearing's hands.

So the sweep is **wider while the payload is still on that silhouette, and
unchanged once it has left it**:

| Grid | Reach | Why |
|---|---|---|
| while the payload is within `attach_sweep_padding_m` of where its **object revision** first appeared | `resolution·√3/2 + attached_clear_padding_m + attach_sweep_padding_m` (one voxel by default → 46.65 mm at 25 mm cells) | the stale pre-attach silhouette is still under the payload, and the payload is collision-active attached geometry the kernel keeps checking — the same conservativeness argument as the clearing itself (ced8e06) |
| once it has moved further than that | `resolution·√3/2 + attached_clear_padding_m` — unchanged | the payload has left the silhouette. Occupancy near it now is evidence about the world, and the ongoing appetite near real furniture must not grow |

**Why a position and not a frame count.** This is the difference between fixing
the field case and not. The bridge re-rasterizes the grid from the octree on
*every* tick, so a sweep that widened its reach only on the first grid after the
attach removes the residue from that one grid and the octree hands it straight
back on the next — nothing retires an occluded cell. The round-5 timeline is the
proof: the attach sweep ran at 1786710428.38 and the E-stop fired at
1786710431.16, **2.78 s — ~28 published grids at 10 Hz — later**, with the
payload still where it was grasped.
`PayloadClearing.TheAttachWindowOutlivesTheFieldRunsTwentyEightGrids` replays
exactly that: 29 consecutive grids, the cell re-marked on each, the payload
unmoved. Against a one-shot sweep it fails on 28 of them.

**Why that particular distance.** Once the payload has translated further than
the sweep padding, a cell that sat between the steady and padded reach at the
attach pose is either *inside* the steady reach (the payload moved toward it —
cleared anyway) or *outside* the padded one (it moved away — no longer its
silhouette to clear). The widened reach has stopped doing work the steady reach
cannot, so it shuts. `TheAttachWindowClosesOnceThePayloadHasMovedAVoxel` pins
that to the millimetre either side of the threshold, and closing **latches** —
`AClosedAttachWindowDoesNotReopenWhenThePayloadComesBack`, because by the time a
payload wanders home the map around it is evidence gathered while it was
elsewhere.

Displacement is measured in the **OctoMap's frame, not the base frame**: the
residue is world-fixed occupancy, so a payload motionless in `base_link` on a
driving mobile base has left it behind and the window must close.

`PayloadClearing.TheRoundSixFieldCellSurvivesTheSteadyReachAndGoesOnTheAttachSweep`
pins the field number itself on a real grid cell (survives at the steady reach,
clears at the attach reach), `…StopsOneVoxelPastTheSteadyReach` bounds the
widened reach to the millimetre either side, and
`TheAttachPaddingNeverShrinksTheSteadyReach` pins that no value of the new
parameter can make the widened sweep clear *less* than an ordinary frame
(`attach_transition_padding` floors a negative or non-finite value at 0).

**What "revision" means, and the one piece of state this node keeps.**
`WorldStateStamped.attachment_revision` is the producer's own counter, bumped
once per atomic `openral_msgs/AttachmentState` change and never per frame, so
`(object_id, attachment_revision)` is a payload's *attachment record identity*.
`AttachSweepLedger` remembers only, per live object, that identity, the position
its window is anchored at, and whether the window has shut — bounded by the
current attachment count. It is **not a latch on cells**: nothing about which
cells were cleared is remembered and no cell is ever held out of a later grid,
so the "derive everything from the wire, every frame" property below is intact.
Answering and recording are the same call (`sweep`), which is what keeps the
window's state and the reach it licenses from disagreeing. Consequences, each
with a test:

* A re-grasp, a second payload, and a place-phase arm/disarm each open a window;
  carrying one payload does not re-open one
  (`ANewObjectRevisionOpensANewAttachWindow`).
* A release sweeps the empty set, so the next grasp opens a window even under a
  revision already seen (same test).
* A frame that clears **nothing** — stale attachment state, missing attach-link
  TF, a payload the bridge refuses to place — never calls `sweep`, so it cannot
  advance or close a window
  (`AFrameThatClearsNothingLeavesTheWindowExactlyAsItWas`). A bridge that has
  swept nothing yet also opens a window for an already-attached payload, since
  it has no reason to believe the silhouette is gone.

**The partition still holds inside the padded reach.** The wider sweep reaches a
voxel further into the counter, and every cell it newly reaches inside the
attested patch is withheld exactly as the footprint is —
`PayloadClearing.ThePaddedAttachSweepStillWithholdsTheAttestedSupportPatch`
asserts it against a counter ring one voxel outside the payload's footprint,
with the no-attestation counterfactual (which eats the ring) beside it. Widening
the reach must not re-open the 2026-08-14 defect for the phase the fix was
written against — and the widened reach is live for the whole window, not one
frame, so the partition has to hold throughout it.

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
| Inside the attested patch laterally **and** in the support half-space — no higher above the attested plane than `resolution/2·(\|n.x\|+\|n.y\|+\|n.z\|) + attested depth + resolution`, unbounded below it | the **kernel's witness** | withheld from the clearing, exempted by the kernel, and still there for the latch to measure |
| Anywhere else within a circumradius of a payload primitive — the payload's own silhouette, the residue above the widened slab | this **clearing** | zeroed |

The third term in that bound is the kernel's **one voxel of co-planar headroom**
(hazard log Entry 012, "Calibration 2026-08-15"), and it is mirrored here **in
lockstep with the kernel by obligation of that same entry**: `withheld ⊆ exempt`
is true because the two sides are the same inequality with a strictly tighter
(zero-slack) bound on this one, and a height term present in one and absent from
the other would break the containment by construction.

The withheld region is that **half-space slab**, never the patch cylinder. At
25 mm cells the slab reaches at most `21.65 mm + attested depth + 25 mm` above
the plane (the pad is largest for a cube-diagonal normal), so with the kernel's
own `support_witness_max_penetration_m` of 10 mm no attestation it would accept
can withhold a cell more than 56.65 mm up.
`PayloadClearing.NoAttestationTheKernelAcceptsWithholdsACellAboveTheWidenedSlab`
pins that bound predicate-level at +60 mm, and
`PayloadClearing.TheRoundFiveResidueCellIsWithheldInTheCoplanarBand` pins the
other side of it on a real grid: the field's +35.8 mm cell, which cleared before
the calibration, is now inside the band and is **withheld, not cleared**, with
the kernel exempting it for the object that attested the patch. That is the
recorded, deliberate consequence of the calibration on this side of the mirror —
and it is the conservative direction, because withholding only ever puts
occupancy back into the published map.

`place_attached_object` lifts the wire `SupportContactWitness` (stated in the
attached object's own frame) through the payload's live pose into a grid-frame
`SupportPatch`, and `support_patch_withholds` is the kernel's own
`support_contact_exempts` geometry with **zero slack**. That asymmetry is
deliberate: the kernel adds `attached_contact_tolerance` (1 mm of physical
FK/pose slack) to the same bound, so what this bridge withholds is a strict
subset of what the kernel exempts for the object that attested it, and no
withheld cell can be the one that stops the robot **against that object**.
`PayloadClearing.WithholdingIsTheKernelsExemptionPredicateAtZeroSlack` mechanises
that containment cell for cell against a transcription of the kernel predicate,
and `PayloadClearing.WithholdingOnlyEverPutsOccupancyBack` re-checks it on every
cell the partition actually withholds. Two scope conditions are **not** covered
by it and are load-bearing when reading a field trace:

* **Per-object.** The kernel exempts per object (`check_attached_voxel_collision`
  tests object *i*'s cells against object *i*'s own witness); this bridge
  withholds per message, so one payload's attestation guards a second payload's
  clearing too — deliberately, since a second payload's volume must not erase the
  first one's support evidence. With two payloads attached, a cell inside the
  first's slab but reached only by the second's volume stays in the map and is
  not exempt for the object that reaches it
  (`PayloadClearing.OneObjectsAttestationGuardsEveryObjectsClearing`).
* **Acceptance bounds.** The kernel's lifecycle node caps what it will accept —
  `support_witness_max_patch_radius_m` (0.5 m) and
  `support_witness_max_penetration_m` (10 mm) — and fails the whole attachment
  message closed past either. This bridge applies the kernel's *geometry* but not
  those caps, so an over-claiming producer would make it withhold a slab the
  kernel exempts nothing in. The sim producer holds the same two numbers
  (`openral_hal._sim_attachment_evidence`) and refuses to construct such a claim.

**The partition is phase-blind, which is what makes ADR-0097 free here.**
`place_attached_object` and `support_patch_withholds` read only the geometry on
`AttachedCollisionObject.support_contact` — never `support_id`, never
`evidence_kind` — so a place-phase witness (the second, declaration-gated
attestation ADR-0097 adds, for the surface a payload is being placed *onto*)
rides this partition unchanged and its support surface is withheld from the
clearing exactly as a pick witness's is. That is not an assumption:
`PayloadClearing.APlacePhaseWitnessIsWithheldExactlyAsAPickPhaseOneIs` asserts
the two produce a bit-identical grid. It matters because the failure mode would
be the 2026-08-14 defect re-opened for the phase the fix was not written
against — the place witness arriving at the kernel with the shelf it attests to
already cleared out from under it, and dying on its own liveness test while the
payload had not moved.

The **approach allowance** of ADR-0097's 2026-08-14 amendment — and its Second
Amendment's raised cap (2026-08-15, `min(1.5 × voxel, 4 cm)`) — leaves this node
untouched, for the same reason, one level up: it is a margin the *kernel* applies
to cells inside the declared target's `PlaceRegion`, not a change to which cells
exist. This bridge decides occupancy, never margins, and it never reads
`AttachmentState.place_declaration` — so the declared region cannot shrink or
grow the grid, and the clearing partition above is the only thing standing
between a payload and its own occupancy either way. The bridge's tests are
unchanged by *that* half of the 2026-08-15 decision, which is the assertion, not
an assumption. Its other half — the witness height envelope — is a predicate the
bridge mirrors, so it lands here too, as the `+ resolution` term above.

The two predicates are a deliberate cross-package mirror —
consolidating them would make this Layer-2 node link the Layer-6 kernel's
collision core — and must be changed in lockstep
(`docs/methods/14-duplication-watch.md`, item 8).

**Withholding only ever puts occupancy back.** It can only *skip* a clear, so
the cells removed with the partition are a strict subset of the cells removed
without it (`PayloadClearing.WithholdingOnlyEverPutsOccupancyBack`). Relative to
the un-partitioned clearing this change moves cases from no-stop to stop, never
the reverse.

**No latch here.** The withholding is derived from the attestation on the wire
every time the grid is published — the bridge's only memory is the
`AttachSweepLedger`'s per-object attach window (above), which decides how far a
sweep reaches and never which cells it takes. Hysteresis lives in the kernel, where it belongs — a
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

**Detach.** Which cells clear is derived from the attachment set on every
published grid, so the first frame after the object leaves that set is already
the frame that publishes it as an obstacle again (the ledger's only role at
detach is to forget the revision, so the next grasp sweeps padded again). Re-*marking* a cell the
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
| `attached_clear_padding_m` | `0.0` | Extra reach beyond the cell circumradius **on every frame**, for pose uncertainty. Every millimetre here removes cells the payload cannot explain. |
| `attach_sweep_padding_m` | `resolution` (one voxel) | Extra reach **while a payload is still within this same distance of where its object revision first appeared** — the attach-transition window — for the stale pre-attach silhouette a fitted primitive leaves just outside the circumradius (the round-5 `voxel_91633`, 22.13 mm out). One parameter, two roles: it is both the extra reach and the distance the payload must travel to close the window. Adds to `attached_clear_padding_m`; `0` restores the pre-#111 single reach. Negative or non-finite contributes nothing. |
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

Its second half pins the **lattice phase** (above), on trees seeded through real
`insertPointCloud` rays with the deploy-sim parameters:
`TheFieldPanelKeepsTheColumnItActuallyOccupies` replays the
`robocasa_drawer_utensil` geometry — the panel face at base x = +0.0614 at 25 mm
with the run's 12.1 mm phase — and asserts both halves of it: the old rule
marked exactly one column, a voxel in front of the panel, and the new one marks
the panel's own column as its outermost;
`ASlabIsNeverMissedAndNeverFurtherForwardThanTheLatticesExplain` sweeps all 25
phases of a 25 mm lattice for the same two properties (the face's column always
marked, nothing further forward than one tree resolution + half a grid cell);
`EveryPhaseAndYawKeepsEveryCellTheCentreSampleWouldHaveMarked` is the
conservativeness proof over phases *and* yaws;
`APhaseAlignedGridRasterizesExactlyAsCentreSamplingDid` is the no-dilation half;
`ACoarseLeafMarksEveryCellUnderIt` pins the pruned multi-resolution leaf;
`AnObliqueLeafMarksTheCellsItEntersAndNotTheirDiagonals` pins that a yawed leaf
is tested exactly rather than by its axis-aligned box;
`AGridOutsideTheOctreesKeyRangeStillSeesEveryLeaf` pins that an unusable bbx
falls back to walking the tree instead of iterating nothing; and
`RasterizingTheKitchenStaysInsideThePublishBudget` holds the cost.

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
nothing at all. Its third part pins the **attach-transition window**: the round-5 field cell at
22.13 mm survives the steady reach and clears at the attach reach; the 29 grids
that replay the field run's ~28-grid gap all clear it (the test a one-shot sweep
fails 28 times over); the window closes to the millimetre once the payload has moved a
voxel, and does not re-open when it comes back; a new revision, a second
payload, and a release-then-grasp each open one while carrying does not; a frame
that clears nothing cannot advance one; the padded sweep still withholds
the attested patch (with the no-attestation counterfactual that eats it beside
it); the widened reach stops one voxel past the steady one to the millimetre;
and no parameter value can make the transition clear less than an ordinary
frame. The kernel's half of the same partition is
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
