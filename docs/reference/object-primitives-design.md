# Object instances before contact — grasp regions without a hovering policy

Status: **design + prototype (default off), pending Safety-WG review**, 2026-10-05. Issue #349.
Branch `feat/object-primitives-349` off the #289 bench integration. The prototype is §4:
`openral_hal._object_primitives` + `GraspTargetLeg(premeasure=True)`, behind
`vision_attachment.grasp_target_premeasure`. Everything in §3 that is not in §4 is design.

Revision 2 (2026-10-05): **camera first, map confirms.** Revision 1 of this note made the voxel
map's connected components the primitives and the camera an optional refinement. The map cannot
give identity and cannot separate objects packed together (cartons in a bin are one
26-connected blob, HZ-0115-30), and it integrates slowly; segmentation is per frame and names one
object per mask. So the order is reversed: every box is **originated** by a head-camera mask and
**confirmed** by the map; the map's one remaining job is to hold the confirmed box while the
hand occludes the camera in the final approach. The map never originates a box.

## 1. Problem

The grasp-target exemption (ADR-0115, [real pick-and-place design §2.1](real-pick-place-design.md))
exempts the declared finger links inside a producer-**measured** region. Without pre-measurement
the region is measured by `openral_hal._grasp_target_leg` only once a hand is within
`grasp_target_approach_m` (0.20 m) of occupied cells (*arming*), and only from one clean
head-camera `SegmentInView` fit of the target gated against the voxel map (*measuring*). A
scripted driver can hover 5 cm over the target until a region arrives; a VLA moves continuously,
so the leg has at most the time the hand takes to cross the approach band to get one unoccluded
fit — and the hand is what occludes the target. No region when the fingers arrive is a kernel
stop: safe, and the end of the run.

Goal: the target's region exists **when the hand arms**, for any object, scene and robot, with no
pause — and the kernel's contract unchanged (no region, no exemption; the region age bound,
`freeze_s` and the caps stay where they are).

## 2. Who knows what

| source | gives | cannot give |
|---|---|---|
| head-camera mask (`SegmentInView`) | identity (one mask, one object); the boundary between touching objects; sub-voxel extents; `not_on_support` (a mask of the target alone that does not reach the support is a stacked body) | a view while the hand occludes; a prompt-free mode (one positive 3-D point per request) |
| voxel map (`/openral/world_voxels`) | a measured support; whether something is *there* (cover); whether a box holds one body or two; memory of views from before the hand arrived | identity; a split between bodies within one voxel of each other (HZ-0115-30); a split between a target and the same-footprint body it stands on |

Every gate the camera path already applies is one of these: the fit (`target_region_from_masks`:
support, caps, nested-mask rules) is the camera's; `region_covers_occupied` and the whole-body
checks are the map's. The design keeps every one of them and changes only **when** the camera
fit is made — before the approach, not during it — and **who holds** the box while the camera
cannot see.

## 3. Design

### 3.1 Pre-measurement: every instance near a hand, before any hand arms

`GraspTargetLeg._premeasure` runs on every leg tick when `premeasure` is on:

1. **Drop what is stale.** An instance last fitted more than `freeze_s` ago (it could not stand a
   live region: `accept` would expire it) or whose box the map no longer covers
   (`region_covers_occupied` below `grasp_target_min_cover` — its cells cleared, the object taken
   away) is dropped.
2. **Pick one prompt** (at most `_PREMEASURE_MAX_HZ` = 2 Hz, one request in flight; the segmenter
   shares the GPU with the policy). Columns, first first: the named search box or the armed
   hand's box; then, for each located hand holding nothing, its TCP box grown by
   `_PREMEASURE_REACH_APPROACHES` = 2 approach distances (the band the hand crosses *before* it
   arms). In each: a *refresh* — the centre of a tracked instance last fitted more than half a
   freeze ago (inside a convex body, so it projects inside its silhouette; in the armed box every
   instance is due at every prompt, as the camera path re-fits every tick) — or a *discovery*
   (`instance_prompt`: the raised top-surface cell no instance covers, nearest the hand,
   preferring interior cells); whichever is nearer the hand. A cell prompted within half a freeze
   is skipped. `SegmentInView` has no prompt-free mode and the continuous detector is evicted
   before every VLA dispatch (`vram_lifecycle_peers`) and lives in layer 2, so the map says
   **where** to look and the camera says **what** is there.
3. **Fit and confirm the reply** (`_on_instance_reply`): the camera path's own fit (`_fit`:
   self-filtered masks, nested-mask rules, support, caps), then `map_confirms`: the map holds the
   box (`map_disagrees` otherwise) and the occupied above-support cells in its cell closure are
   **one** 26-connected body (`map_split` otherwise). The box is never completed or grown from
   the map (`map_completed_region` is not used here): that would merge exactly what the mask
   separated.
4. **Track it** (`PrimitiveTracker.update`, label-free): a fit within one voxel refreshes its
   instance; one whose centre lies in an instance's box (or holds its centre) is that instance
   nudged; anything else is new. Bounded at `max_tracks` = 16 (the one seen longest ago goes).

**Packed objects.** Two cartons face to face are one body in the map. The first discovery prompt
lands one cell inside the near carton's top (interior preference; the merged top's centroid
would be the seam), its mask names that carton, `map_confirms` passes (its padded box holds that
carton and a sliver of the touching one — one body), and once its instance covers that top the
next prompt lands on the other carton. Two masks, two instances; the map alone would have
offered the blob or nothing.

### 3.2 Arming: the region on the arming tick, no segmenter call

`GraspTargetLeg._tick_instances`, on every tick a target is armed (unchanged arming: the approach
box holds `min_cells` occupied cells above its lowest layer, or a named search box):

0. A held region the map no longer covers is retracted at once (`map_disagrees`), hand or no hand.
1. The candidate is the tracked instance nearest the armed hand's TCP points inside the search
   column, refused on a tie within one voxel (`nearest_primitive`, HZ-0115-2); for a named search
   box, the lone instance inside it (two: `ambiguous`; none: the lost view `no_instance`). A named
   target also requires the **whole** measured instance to fit within its search box plus one
   voxel. A camera fit that widens across the named box's edge retracts the region, even when
   its centre remains inside and the map confirms occupied cells there.
2. A candidate fitted after the held region is confirmed on the current grid (`map_confirms`) and
   gated against the held region (`_gate_refit(complete=False)`: tracking within one voxel, the
   own-hand `occluded_refit` hold, `target_moved`, `unoccluded_refit`), then accepted. On the
   arming tick nothing is held, so the pre-measured instance is the region before the next chunk.
3. Otherwise, while a declared contact link's TCP is over the held region (`_hand_over_target`,
   the existing occlusion test), **the map holds it** (`_map_hold`): confirmed on a newer grid,
   the same box is re-accepted stamped with that grid's `source_stamp` — never moved, never grown.
   A second body in it then (the hand's own points leaked into the map) is the lost view
   `occluded_refit`: the region is held under its freeze TTL, not replaced.
4. With neither, nothing new is accepted: the held region runs out its freeze TTL from its own
   stamp.

### 3.3 Staleness — a region never outlives its evidence

| evidence | lifetime |
|---|---|
| instance (camera fit) | dropped `freeze_s` after its fit, or when the map stops covering it |
| region from an instance | its stamp is `min(depth frame, grid source_stamp)` of the fit; the freeze TTL and the kernel's `grasp_region_max_age_s` run from it, as for any camera fit |
| region held by the map (hand over it) | re-stamped to each confirming grid's `source_stamp`; dies `freeze_s` after the last one, at once when a grid no longer covers it; a stalled octomap ages it out (`grid_source_stale`, the kernel's bound) |
| any region | retracted with the declaration, at `approach_ended`, on `target_moved`, as today |

The map hold is the one place a region lives longer than its last camera fit: while the robot's
own declared hand is over it, on map evidence the kernel already bounds by `source_stamp`. That
is the posture change §6 asks the Safety-WG to accept.

### 3.4 A reasoner-declared target

`ExecuteRskillTool.grasp_target` → `ground_grasp_target` → `GraspDeclaration.search_box` is
unchanged. The named box is the first pre-measurement column, so its instance is fitted within a
prompt or two of dispatch, and the region is the lone instance in it on the next tick — before
the goal's first chunk reaches the object, hands wherever they are. Later (§3.7): the store
published from world state, the detector label grounded to an instance id, `target_id =
prim:<id>`.

### 3.5 A VLA that picks its own target

The approach-armed path is unchanged up to the search box. Instances near each free hand are
measured while it is still two approach distances away; the candidate on the arming tick is the
nearest one. At 0.20 m approach and a 3 Hz tick the kernel has the region with 7–20 cm left,
against 0 cm when no clean fit arrives in the band. Chunk-lookahead arming (option 3 of the issue)
plugs in at one point — the TCP points handed to `approach_box`, the pre-measurement box and
`nearest_primitive` become the predicted ones — and is out of scope.

### 3.6 Robot and scene agnosticism

Nothing names a robot or a scene: hands from `openral_core.gripper_hands`, TCPs from tf2 through
the bridge (`jaw_point`), the camera and its depth from the vision leg's config, the support
measured under each prompt, the lattice's resolution/frame/orientation from `OccupancyVoxels`, the
caps the declaration's; every threshold a calibration point with a stated default.

### 3.7 Failure modes

| failure | outcome | why it fails safe |
|---|---|---|
| no segmenter / no depth / deadline | no instance; prompt noted once per change | nothing tracked, no region, kernel stops at the target |
| mask spans two separated bodies | `map_split`, not tracked | a box over two bodies is never an instance |
| mask on something the map does not hold | `map_disagrees`, not tracked | the map must confirm every box |
| packed pair (touching) | one instance per mask; each box holds a sliver of its neighbour | **residual** (HZ-0115-30, now a sliver within the camera padding, not the whole neighbour) |
| target on a same-footprint body | the whole-stack mask is refused `STACKED`; the target's own mask, refused `not_on_support`, is re-fitted on its own lowest point | the body it stands on never enters the region (HZ-0115-37 closed, HZ-0115-42) |
| support unseen (a bin floor out of the camera's view; a neighbour's top or a bin rim above the target's bottom) | the region stands on the target's own lowest kept point, logged `support unseen` | the box never reaches into space nothing measured; a finger below it meets the kernel's voxels |
| two instances tie for the hand / two in a named box | `ambiguous`, retracted | neither exempt until the hand commits |
| hand points leak into the map | the hold's `map_split` is a lost view; the held box is never replaced | freeze TTL bounds it |
| object removed | held region retracted (`map_disagrees`), instance dropped | the region dies with the map evidence |
| object moved | a new fit outside the held box is `target_moved` | as today |
| instance not re-seen | dropped after `freeze_s` | an instance that could not stand a live region is not kept |
| stalled octomap | `grid_source_stale`; nothing pre-measured; every region ages out | unchanged kernel bound |

### 3.8 Layers

| piece | layer | status |
|---|---|---|
| `_object_primitives` (prompt, confirmation, tracker), leg wiring | 0 HAL (`openral_hal`, beside `_grasp_target`; the bridge owns the segmenter client) | prototype |
| `vision_attachment.grasp_target_premeasure` | core schema + HAL param + deploy-sim mapping | prototype |
| publishing the store (`/openral/object_primitives`, an `ObjectPrimitiveSet` IDL) | 2 World State | design |
| label → instance id grounding, `GraspTargetRef.object_id = prim:<id>` | 4 Reasoner (`openral_reasoner.grounding`) | design |
| hull on the grasp region | 5 Safety (kernel; `PlaceRegion.geometry` on a grasp region) | design, separate ADR |

The prototype crosses no layer boundary (the HAL already consumes `/openral/world_voxels` and owns
the `SegmentInView` client). It changes two recorded safety judgements, both in the producer, not
the kernel: the region may be accepted from a camera fit made **before** the arming (bounded by
the freeze TTL from that fit), and the map may **hold** a camera box while the declared hand
occludes it. **An ADR amendment and hazard-log rows are required before the parameter is turned
on in any committed scene**, Safety-WG reviewed; the kernel is untouched. Draft text for the
private management log (`OpenRAL/management`, `adr/`):

> **ADR-0116 — Pre-measured camera instances as the grasp-target region source (amends ADR-0115).**
> *Context.* ADR-0115 measures the grasp region from one unoccluded `SegmentInView` fit gated
> against the voxel map, made after a hand arms. A continuously moving policy gives the producer
> no unoccluded fit while armed (Isaac trials i25–i76 required a hover); without a region the
> kernel stops the fingers at the target. The voxel map alone cannot separate objects packed
> together, and cannot name one. *Decision.* (1) The producer may measure object instances
> before any hand arms: one `SegmentInView` mask per instance, prompted on a raised top surface
> of the map near a free hand or in a named search box (≤ 2 Hz, one in flight), fitted by the
> existing fit (self-filtered masks, nested-mask rules, support, caps) and **confirmed** by the
> map — it covers the box and the box's cell closure holds one 26-connected body above the
> support. The map never originates, completes or grows an instance. (2) On arming, the region
> is the tracked instance nearest the armed hand's TCP points (refused on a tie within one
> voxel), or the lone instance in a named search box, gated as any re-fit (tracking, own-hand
> occlusion); its stamp is the fit's (`min(depth, grid source_stamp)`) and the freeze TTL runs
> from it. (3) While a declared contact link is over the held region, a newer grid that confirms
> it re-stamps the same box (`source_stamp`); a grid that no longer covers it retracts it, hand
> or no hand. (4) An instance is dropped `freeze_s` after its last fit or when the map stops
> covering it. (5) The kernel contract is unchanged: no region, no exemption; the region age
> bound and `freeze_s` are unchanged; `PlaceRegion.geometry` stays empty on grasp regions. (6)
> Production posture: off by default; on in a committed scene only with this ADR accepted.
> *Consequences.* The region exists on the arming tick (named targets: within a prompt or two of
> dispatch). Packed objects are separated by their masks. The camera's `not_on_support` gate is
> kept. Hazard rows: HZ-0115-35 (a pre-measured instance on the wrong object — tie rule, one
> hand, map confirmation, freeze TTL from the fit), HZ-0115-36 (a packed neighbour's sliver
> inside an instance's padded box — residual, bounded by the camera padding and the cell
> closure), HZ-0115-38 (the map hold keeps a region alive past its last camera fit while the
> hand is over it — same box only, retracted when the map stops covering it, bounded by
> `source_stamp` and `freeze_s`; leaked hand points never replace it).

## 4. Prototype (this branch)

- `python/hal/src/openral_hal/_object_primitives.py`: `instance_prompt`, `map_confirms`,
  `nearest_primitive`, `point_box_gap_m`, `PrimitiveTracker` (`update`, `prune`), `PrimitiveFit`,
  `ObjectPrimitive`. Pure numpy; reuses `_grasp_target`'s lattice, components, cover and tracking.
- `GraspTargetLeg(premeasure=True)`: `_premeasure` (prune, one prompt, `_request` with no
  declaration), `_on_instance_reply` (`_fit` → `map_confirms` → tracker), `_tick_instances` (the
  armed region from the instances, `_gate_refit(complete=False)`, `_map_hold`). Every request is a
  pre-measurement in this mode; `kernel_region`, `fill`, the handover, the payload and the support
  witness are untouched.
- Parameter chain: `VisionAttachmentRuntime.grasp_target_premeasure` (scene) →
  `vision_attachment_grasp_target_premeasure` (HAL param, `deploy_sim._vision_attachment_hal_params`)
  → `VisionAttachmentConfig.grasp_target_premeasure` → the leg. Default off everywhere; needs
  `grasp_target_enabled`; the approach-armed path also needs `grasp_target_approach_m`.
- Tests: `tests/unit/test_object_primitives.py` (the committed Isaac warehouse scene's three props
  on the pallet deck at 15 mm cells, ray-cast into the manifest's `head_zed` —
  `tests/unit/_head_camera_render.py`: each prop one confirmed instance in the kernel's contract,
  nearest/tie, the packed pair the map merges separated by two masks with the prompts landing one
  per carton, map refusal of a box it does not hold or that holds two bodies, tracker identities,
  staleness and map-contradiction drops); `tests/unit/test_grasp_target_leg.py::
  test_premeasured_instances_arm_the_region_with_no_segmenter_call` (real bridge, tf2, lattice
  and `SegmentInView.Response` replies through `_on_instance_reply`; no segmenter in the graph:
  the arming tick holds the instance's box, the map hold re-stamps it, removal retracts it,
  staleness drops the rest) and `::test_a_named_search_box_grounds_to_the_lone_instance_in_it`.

Enable in a deploy scene (with the vision leg already enabled):

```yaml
runtime:
  vision_attachment:
    enabled: true                   # camera + driver topics as the scene already sets them
    grasp_target_enabled: true
    grasp_target_approach_m: 0.20   # the approach-armed path (a VLA picking its own target)
    grasp_target_premeasure: true
```

## 5. Evaluation plan — restock shelves in Isaac Sim

**Environment.** A private restock-shelf USD stage (four cartons on a shelf board in front of the
bimanual OpenArm, `deploy sim` with the Isaac backend, octomap at 15 mm, the kernel's world-voxel
check and grasp allowance on, the vision bridge with the head camera and the segmenter). The stage
and its scene YAML live outside this repository and are referenced only as such.

**Driver.** The same perception-driven scripted driver the i25–i76 trials used, with its hover wait
set to zero so it descends continuously from the pre-grasp pose (`descend_without_region` when no
region is held at the start of the descent) — the VLA's behaviour, with a known-good grasp pose.
Two arms: A (`grasp_target_premeasure: false`, the per-arming camera path) and B (`true`). Same
stage, same target, same driver settings. A third arm with cartons packed face to face exercises
the split.

**Metrics** (all from the run's logs: the driver's `driver.jsonl`, the HAL's `grasp target region
accepted` and `instance pre-measurement` lines, the kernel's `safety.collision` /
`grasp_region_latched` lines, the runner result):

| metric | definition |
|---|---|
| region before contact | the kernel held a region for the armed hand before the first `safety.collision` involving a finger link or the ATTACH, whichever first |
| region on the arming tick | first `accepted` − first `approach` arming line ≤ one tick |
| instances before arming | instances tracked around the hand at its arming line |
| split | packed arm: distinct instances on the packed cartons / cartons |
| false region rate | accepted regions whose box does not overlap the driver's chosen cluster / accepted regions |
| map-hold span | longest run of re-stamped (`source_stamp`) acceptances of one box |
| stop rate | runs ending in a kernel stop on the target's cells / runs |
| outcome | ATTACH reached; handover with a region; lift without a stop |

**Procedure.** `run_trial.sh`-style: boot the graph, wait for the segmenter and voxels, send the
goal, record; one trial per arm, then repeat if the host allows. Scratch scene YAML and driver copy
stay under the host's scratch directory; nothing is committed.

**Result.** See the PR report for the run or what blocked it; this section is updated when a run
completes.

## 6. Decisions needed

1. Safety-WG: accept the pre-measured instance source and the map hold (ADR-0116 draft,
   HZ-0115-35/36/38) for sim evaluation, and the production posture.
2. Where the store lives long-term: in the HAL leg (one consumer, no new topic) or published from
   world state for the reasoner (grounding by instance id) — the latter needs the ADR's
   layer-crossing clause and an `ObjectPrimitiveSet` IDL.
3. The pre-measurement budget on the shared GPU (`_PREMEASURE_MAX_HZ`, the reach): measured in the
   Isaac evaluation against the policy's chunk latency.
4. Whether a prompt-free instance mode (SAM automatic masks on the head frame) should replace the
   map-seeded prompts once the segmenter node offers one.
