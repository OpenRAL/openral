# Object primitives before contact — grasp regions without a hovering policy

Status: **design + prototype (default off), pending Safety-WG review**, 2026-10-05. Issue #349.
Branch `feat/object-primitives-349` off the #289 bench integration. The prototype is the map-first
half (§4): `openral_hal._object_primitives` + `GraspTargetLeg(primitives=True)`, behind
`vision_attachment.grasp_target_primitives`. Everything in §3 that is not in §4 is design.

## 1. Problem

The grasp-target exemption (ADR-0115, [real pick-and-place design §2.1](real-pick-place-design.md))
exempts the declared finger links inside a producer-**measured** region. Today the region is
measured by `openral_hal._grasp_target_leg` only once a hand is within `grasp_target_approach_m`
(0.20 m) of occupied cells (*arming*), and only from one clean head-camera `SegmentInView` fit of
the target gated against the voxel map (*measuring*). A scripted driver can hover 5 cm over the
target until a region arrives; a VLA moves continuously, so the leg has at most the time the hand
takes to cross the approach band to get one unoccluded fit — and the hand is what occludes the
target. No region when the fingers arrive is a kernel stop: safe, and the end of the run.

Goal: the target's region exists **before the policy's first chunk reaches the object**, for any
object, scene and robot, with no pause — and the kernel's contract unchanged (no region, no
exemption; the region age bound, `freeze_s` and the caps stay where they are).

## 2. What the map already knows

Every gate the camera fit passes today is a map gate: `region_covers_occupied`,
`occupied_touching_outside`, `map_completed_region` (the fit grown to the target's whole
26-connected component above the measured support, because the map remembers views from before
the hand arrived). The camera's own contributions are two: it **names** the target (the mask is of
one object, so a target standing on a same-footprint body fails `not_on_support`), and it measures
sub-voxel extents. Everything else — support plane, component, whole-object extent, caps, cell
closure, staleness — the map supplies, continuously, from the moment the scene is first seen and
long before any hand approaches. The design turns that around: **the map component is the
primitive; the camera refines and confirms it when it can.**

## 3. Design

### 3.1 Tracked primitive store

One record per object the map holds (`ObjectPrimitive`):

| field | meaning |
|---|---|
| `primitive_id` | stable identity across grids (minted by the tracker, never reused in a session) |
| `region` | gravity-aligned (yaw-only) `PlaceRegion`: the tight box over the component's cell centres, lower face `support_z + resolution` (never into the support, HZ-0115-6), yaw = footprint PCA; `cell_closed_region` is what the kernel gets |
| `cell_count`, `first_seen_ns`, `last_seen_ns` | provenance; `last_seen_ns` is the `source_stamp` of the grid the box came from (the region's `stamp_ns`) |
| *(later)* `hull` | the component's convex hull, for a tighter kernel exemption once `PlaceRegion.geometry` is allowed on a grasp region (v1 forbids it; kernel change, out of scope here) |
| *(later)* `confirmed_ns`, `label` | the last clean camera fit that held exactly this component, and the detector label grounded to it |

**Source.** `primitives_from_voxels` (§3.2) on every grid; a camera fit **refines** a primitive
only when the fit's cell closure holds the whole component (`occupied_touching_outside` empty, or
`map_completed_region` succeeds — the existing completion logic, run the other way round: the
component is known first, the mask must agree with it). A refinement replaces the horizontal
half-extents and yaw with the cloud's (sub-voxel), keeps the map's lower face, and marks the
primitive `confirmed`; a mask that covers only part of the component (a stack, a packed pair)
*splits* it (§3.2). The refinement is not in the prototype: the tracking gate's one-voxel tolerance
would see the ~1.5 cm pad difference between a camera box and a map box as `target_moved`, so the
two sources must share one record rather than alternate in `GraspTargetTracker`.

**Association across frames** (`PrimitiveTracker.update`, label-free): a fit that barely moved
(`track_region` within one voxel) refreshes its primitive; one whose centre lies in a primitive's
box, or that holds the primitive's centre, is the same object nudged or re-fitted (same identity,
box replaced); anything else is a new identity. Largest fits match first.

**Staleness, moved, removed.**
- *Removed / cleared*: a primitive inside the scanned column that no fit matched accrues a miss;
  dropped after `max_misses` (3) grids. Outside the scanned column it is kept — not looked at is
  not absent (the same rule as `ObjectMemory`).
- *Moved*: a nudge keeps the identity with the new box; a jump makes a new identity and the old one
  is dropped by misses. The held region, if it was that primitive, follows the leg's existing
  `_gate_refit` rule: a candidate that no longer tracks the held region is `target_moved`
  (retracted) unless it shrank *inside* it with the declared hand at it (`occluded_refit`, held
  under the freeze TTL).
- *Stale*: every region is stamped with its grid's `source_stamp`; the kernel's
  `grasp_region_max_age_s` (2 × the voxel deadline) and the leg's freeze TTL bound it exactly as a
  camera region. A stalled octomap ages every primitive out (`grid_source_stale`).
- *The hand in the map*: the self-filter removes the robot's points before insertion; if it leaks,
  the component grows and the fit no longer tracks the held region → `occluded_refit` while the
  hand is at it (held under the TTL, never replaced by the grown box), `target_moved` otherwise.

### 3.2 Connected components above a measured support

Per grid, inside the search column (the approach box extended `support_search_below_m` down, or
the reasoner's box):

1. **Anchor** (`raised_anchor_xy`): the top-surface cell (no occupied cell directly above) nearest
   the column centre among those more than one cell above the column's lowest top surface. The
   existing anchor — "the occupied column nearest the box centre" — lands on the bare support
   when the hand approaches from the side and then measures no support under it; anchoring on
   what *stands* on the surface fixes the lateral approach.
2. **Support** (`support_top_from_voxels`, unchanged): the first layer under the anchor whose
   top-surface cells ring the footprint of the component standing on it; its top face is
   `support_z`. No ringed layer → `no_support`, no primitives.
3. **Components**: occupied cells of the column more than one voxel above `support_z`, 26-connected
   (one view's shell of a slanted face is only diagonally connected). Each is a primitive when it
   has `min_cells` cells, its lowest cell is within two voxels of the support (else it stands on
   something else — a shelf above, a riser), none of its cells touches the column's edge (the leg
   cannot vouch for what continues outside; `map_completed_region`'s rule), and its tight box fits
   `GraspDeclaration`'s caps.

**Packed shelves (merged neighbours).** Two bodies within one voxel are one component (HZ-0115-30).
Options, in order of generality:

- *Caps* (in the prototype): a merged pair past 0.20 m / 0.03 m³ yields no primitive — fail safe,
  the kernel stops the fingers at the neighbour exactly as today.
- *Height step*: split a component where its top surface drops by more than two cells across one
  cell (a short can beside a tall box). Cheap, lattice-only; misses equal-height pairs.
- *Footprint neck*: erode the footprint by one cell, relabel, dilate back (watershed-lite); splits
  bodies touching along a corner or an edge of ≤ 2 cells, not face-to-face boxes.
- *Segmentation* (the general one; §3.1 refinement): a SAM mask prompted at the component's
  top-centre that covers only part of the component splits the component at the mask boundary
  projected onto the lattice; the part under the mask is the confirmed primitive, the rest a
  sibling. This is the only split that separates two equal boxes touching face-to-face, and it
  needs the unoccluded view the policy will not wait for — so it runs **before** any approach, on
  every grid where the primitive is unoccluded (the detector/segmenter already runs continuously
  with `enable_object_detector`), and its result is kept on the record until the component changes.

Until a split lands, a merged pair under the caps is one primitive and its region exempts both
bodies for the declared finger links: the residual Safety-WG accepted for `map_completed_region`,
now reachable without a camera confirmation (§6, HZ-0115-36).

### 3.3 A reasoner-declared target grounds to a primitive at dispatch

`ExecuteRskillTool.grasp_target` → `ground_grasp_target` → `GraspDeclaration.search_box` (the
lifted detection's padded box) is unchanged. In primitives mode the leg resolves a named search
box on its **first tick**: the primitives inside the column under it; exactly one → its region is
accepted and published at once, hands wherever they are; none → `no_primitive` (lost view, retried
every tick); several → `ambiguous` (the detection covered two objects). So a named target carries
a region **before the goal's first chunk is executed** — the runner stamps the declaration at goal
start and the leg ticks at 3 Hz — with no camera involved.

Later, with the store published from world state (§5): the detector label is grounded to a
primitive id by the lift (the primitive whose box holds most of the detection's frustum cells),
`GraspTargetRef.object_id` names a primitive id directly, and the declaration carries
`target_id = prim:<id>`; the leg then holds that identity rather than "whatever is in the box", and
a primitive that is dropped or replaced retracts the region.

### 3.4 A VLA that picks its own target

The approach-armed path is unchanged up to the search box (the hand's TCP points grown by
`approach_m`, arming on `min_cells` occupied cells above the box's lowest layer). The candidate is
then `nearest_primitive`: the primitive with the smallest gap from any of the hand's TCP points to
its box; a tie (the runner-up within one voxel of the nearest) is `ambiguous` — refused, as
`target_seed_from_voxels` refuses two comparable clusters (HZ-0115-2), and resolved on a later tick
as the hand comes nearer one of them. The region is available on the arming tick: at 0.20 m
approach and a 3 Hz tick the hand has 7–20 cm left when the kernel has the region, against 0 cm
today when no clean fit arrives. Option 3 of the issue (chunk-lookahead arming) plugs in at one
point — the TCP points handed to `approach_box` and `nearest_primitive` become the predicted
ones — and is out of scope here.

### 3.5 Robot and scene agnosticism

Nothing here names a robot or a scene: hands come from `openral_core.gripper_hands` and their TCPs
from tf2 through the bridge (`jaw_point`); the support is measured, never read off a box; the
lattice resolution, frame and orientation come from `OccupancyVoxels`; the caps are the
declaration's; every threshold is a calibration point with a stated default. Any deploy graph that
publishes `/openral/world_voxels` and runs the vision bridge gets primitives; a robot with one hand
or several, a table, a pallet or a shelf board, a 15 mm sim lattice or a 50 mm real one.

### 3.6 Failure modes

| failure | outcome | why it fails safe |
|---|---|---|
| no object in the column / nothing raised above the surface | `no_primitive` lost view, no region | nothing exempt; the kernel stops at whatever is there |
| no ringed support layer (hollow target, table edge) | `no_support` contradiction | no region; same as today |
| two primitives tie for the hand / two in a named box | `ambiguous` contradiction, retracted | neither exempt until the hand commits to one |
| component touches the column edge | skipped; `no_primitive` | a half-seen body is never exempt |
| merged pair past the caps | skipped | no region; kernel stops at the neighbour |
| merged pair under the caps | one primitive for both | **residual** (HZ-0115-36); as `map_completed_region` today |
| target stacked on a same-footprint body | one primitive down to the support | **residual** (HZ-0115-37); the camera's `not_on_support` catches it — production posture keeps a camera confirmation |
| robot points leak into the map near the hand | grown box fails tracking → `occluded_refit` hold, then `freeze_ttl` | the grown box never replaces the held one |
| object moved or removed | `target_moved` retract; misses drop the primitive | the region dies with the evidence |
| stalled octomap | `grid_source_stale`; every region ages out at `grasp_region_max_age_s` | unchanged kernel bound |
| self-filter cloud missing | not a factor: no camera fit is made | the map's own self-filter is upstream of the grid |

### 3.7 Layers

| piece | layer | status |
|---|---|---|
| `_object_primitives` extractor + tracker, leg wiring | 0 HAL (`openral_hal`, beside `_grasp_target`) | prototype |
| `vision_attachment.grasp_target_primitives` | core schema + HAL param + deploy-sim mapping | prototype |
| publishing the store (`/openral/object_primitives`, an `ObjectPrimitiveSet` IDL) | 2 World State (`packages/world_state`) | design |
| label → primitive id grounding, `GraspTargetRef.object_id = prim:<id>` | 4 Reasoner (`openral_reasoner.grounding`) | design |
| camera refinement / segmentation split | 0 HAL (the bridge owns the segmenter client) | design |
| hull on the grasp region | 5 Safety (kernel; `PlaceRegion.geometry` on a grasp region) | design, separate ADR |

The prototype crosses no layer boundary (the HAL already consumes `/openral/world_voxels`). It
does change a recorded safety judgement: ADR-0115's measurement section says *"the map alone never
arms: no accepted camera fit, no completion"*. Arming from the map alone is a posture change for
the exemption, so **an ADR amendment and hazard-log rows are required before the parameter is
turned on in any committed scene**, Safety-WG reviewed; the kernel is untouched. The world-state
publication and the reasoner grounding to primitive ids cross layers 2 → 0 and 2 → 4 and need the
same ADR (or a second one). Draft text for the private management log (`OpenRAL/management`,
`adr/`):

> **ADR-0116 — Map object primitives as the grasp-target region source (amends ADR-0115).**
> *Context.* ADR-0115 measures the grasp region from one unoccluded `SegmentInView` fit gated
> against the voxel map, armed when a hand is within the approach distance. A continuously moving
> policy gives the producer no unoccluded fit while armed (Isaac trials i25–i76 required a hover);
> without a region the kernel stops the fingers at the target. *Decision.* (1) The voxel map's
> 26-connected component standing on a measured support is a valid region source: the tight
> gravity-aligned box over its cell centres, lower face one voxel above the support, cell-closed,
> within the declaration's caps, stamped with the grid's `source_stamp`, run through the producer's
> existing gates (map cover, whole component, tracking, own-hand occlusion hold). (2) The candidate
> is the primitive nearest the armed hand's TCP points, refused on a tie within one voxel; a named
> search box resolves only to a lone primitive inside it. (3) A component is never split or
> extended by inference: a merged pair is one primitive or none (the caps); a body touching the
> column's edge is none. (4) The kernel contract is unchanged: no region, no exemption; the region
> age bound and `freeze_s` are unchanged; `PlaceRegion.geometry` stays empty on grasp regions.
> (5) Production posture: the mode is off by default; turning it on in a committed scene requires
> either a camera confirmation per primitive (§3.1 refinement, not yet built) or Safety-WG
> acceptance of HZ-0115-36/37 for that cell. *Consequences.* The region exists on the arming tick
> (named targets: on the first tick after dispatch, before motion). The camera's `not_on_support`
> gate is lost while the refinement is unbuilt (HZ-0115-37). Hazard rows: HZ-0115-35 (map-only
> region on the wrong component — tie rule, one hand, caps, age bound), HZ-0115-36 (merged
> neighbours exempt together — caps, residual accepted per cell), HZ-0115-37 (stacked body exempt
> under the target — residual until the refinement lands), HZ-0115-38 (robot points leaked into the
> map grow the component — the grown fit never replaces the held region; freeze TTL).

## 4. Prototype (this branch)

- `python/hal/src/openral_hal/_object_primitives.py`: `primitives_from_voxels`,
  `raised_anchor_xy`, `nearest_primitive`, `point_box_gap_m`, `PrimitiveTracker`, `PrimitiveFit`,
  `ObjectPrimitive`. Pure numpy; reuses `_grasp_target`'s lattice, components and tracking gate.
- `GraspTargetLeg(primitives=True)`: the measurement tick measures the support under the raised
  anchor, extracts and tracks the column's primitives, picks the hand's one, and commits it through
  `_gate_refit` + `tracker.accept(map_cells=<component size>)`; no `SegmentInView` request is made.
  `kernel_region`, `fill`, the handover, the payload and the support witness are untouched.
- Parameter chain: `VisionAttachmentRuntime.grasp_target_primitives` (scene) →
  `vision_attachment_grasp_target_primitives` (HAL param, `deploy_sim._vision_attachment_hal_params`)
  → `VisionAttachmentConfig.grasp_target_primitives` → the leg. Default off everywhere; needs
  `grasp_target_enabled`.
- Tests: `tests/unit/test_object_primitives.py` (the committed Isaac warehouse scene's three props
  on the pallet deck at the sim's 15 mm cells, from `scenes/deploy/isaac_openarm_warehouse.yaml`
  and `robots/openarm/robot.yaml`: extraction, kernel contract, nearest/tie, merged pair, caps,
  column edge, stacked body, lateral anchor, tracker identities and retraction);
  `tests/unit/test_grasp_target_leg.py::test_map_primitives_measure_the_approached_target_with_no_segmenter`
  and `::test_a_named_search_box_grounds_to_the_lone_primitive_in_it_before_any_motion` (a real
  bridge, tf2 and lattice, no segmenter in the graph).

## 5. Evaluation plan — restock shelves in Isaac Sim

**Environment.** A private restock-shelf USD stage (four cartons on a shelf board in front of the
bimanual OpenArm, `deploy sim` with the Isaac backend, octomap at 15 mm, the kernel's world-voxel
check and grasp allowance on, the vision bridge with the head camera). The stage and its scene YAML
live outside this repository and are referenced only as such.

**Driver.** The same perception-driven scripted driver the i25–i76 trials used, with its hover wait
set to zero so it descends continuously from the pre-grasp pose (`descend_without_region` when no
region is held at the start of the descent) — the VLA's behaviour, with a known-good grasp pose.
Two arms: A (primitives off, camera path) and B (primitives on). Same stage, same target, same
driver settings.

**Metrics** (all from the run's logs: the driver's `driver.jsonl`, the HAL's `grasp target region
accepted` lines, the kernel's `safety.collision` / `grasp_region_latched` lines, the runner result):

| metric | definition |
|---|---|
| region before contact | the kernel held a region for the armed hand before the first `safety.collision` involving a finger link or the ATTACH, whichever first |
| region before the first chunk | named-target variant: a region on the envelope before the runner's first `candidate_action` |
| latency from arming | first `accepted` − first `approach` arming line, s |
| latency from scene start | first `accepted` − first `/openral/world_voxels` grid, s |
| false region rate | accepted regions whose box does not overlap the driver's chosen cluster (the only ground truth the driver has, from the same map) / accepted regions |
| stop rate | runs ending in a kernel stop on the target's cells / runs |
| outcome | ATTACH reached; handover with a region; lift without a stop |

**Procedure.** `run_trial.sh`-style: boot the graph, wait for the segmenter (A) / voxels, send the
goal, record; one trial per arm, then repeat the pair if the host allows. Scratch scene YAML and
driver copy stay under the host's scratch directory; nothing is committed.

**Result.** See the PR report for the run or what blocked it; this section is updated when a run
completes.

## 6. Decisions needed

1. Safety-WG: accept the map-only region source (ADR-0116 draft, HZ-0115-35..38) for sim
   evaluation, and decide the production posture (camera confirmation required, or per-cell
   acceptance of HZ-0115-36/37).
2. Where the store lives long-term: keep it in the HAL leg (one consumer, no new topic) or publish
   it from world state for the reasoner (option 4 by primitive id) — the latter needs the ADR's
   layer-crossing clause and an `ObjectPrimitiveSet` IDL.
3. Whether the lateral-approach anchor (`raised_anchor_xy`) should also serve the camera path
   (today a hand beside the target seeds the support and refuses `no_support`); it is used by the
   primitives path only in this branch.
4. Which split to build first for packed shelves (§3.2): the segmentation split is the general one
   and reuses the continuous detector; the height-step split is a day's work and lattice-only.
