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

Revision 3 (2026-10-11): **the map adjudicates an occluded re-fit while the hand approaches.**
Isaac s_mix07 (one of 46 deploy trials on the Spark host; the only `target_moved` retraction in
them): the carton's instance was held 7 s, then, as the right hand descended onto it, a
pre-measurement re-fit came back shifted and shrunk on the hand's side — held centre (0.150,
−0.091, −0.508) half (0.059, 0.044, 0.028) → re-fit (0.168, −0.088, −0.517) half (0.047, 0.041,
0.038). It was retracted as `target_moved`; 0.9 s later the finger reached the carton with no
exemption live and the kernel stopped (PhysX ground truth: the finger touched nothing, the
carton had not moved). Two gaps: the re-fit's only excursion past the held box grown by one
voxel was its bottom, 19 mm lower, standing on the measured board where the held fit (support
unseen) stood on its own lowest point; and the hand was approaching, not yet over the held
footprint, so the own-hand occlusion test could not apply. *(Superseded by Revision 4: the
move test's downward growth and the approaching-hand map overrule are removed.)*

Revision 4 (2026-10-11): **frozen after map confirmation.** Revision 3 patched one gate; the next
Isaac deploy trials showed the camera re-fit loop redefining or retracting the held region
exactly while the approaching hand occludes the target, each time through a different gate,
after which the kernel stopped the finger at the target with no exemption live (PhysX: no real
contact):

- **s_mix07** — the occluded re-fit 19 mm lower at the bottom and shifted 18 mm on the hand's
  side → `target_moved` (the case Revision 3 patched);
- **t_mix04** (after Revision 3) — the hand above the carton, the re-fit saw only a thin top
  layer: re-fit centre (0.156, −0.081, −0.495) half_z 0.015, yawed 14° off the held centre
  (0.157, −0.081, −0.511) half_z 0.032 → `target_moved` upward → retracted → the finger stopped
  at the target's edge 0.8 s later, exemption not live;
- **t_mix02** — a re-fit near the tote wall, centre (0.283, 0.106, −0.492) half (0.091, 0.055,
  0.063) against the held (0.262, 0.078, −0.517) half (0.063, 0.042, 0.038), reached outside
  the named search box → `outside_named_search_box` → retracted (re-accepted 2 s later).

Decision (maintainer-approved; the per-gate patching is replaced by one posture): **once a
region is accepted and map-confirmed while a declaration is live, it is frozen for the life of
that declaration.** Camera re-fits never shrink, move, grow or retract it. **Only the voxel map
retracts it** — on every tick with a fresh grid, `map_confirms` of the held box on its measured
support: `map_disagrees` (the grid no longer holds it: the object removed or moved off it) or
`map_split` (a second body inside it, unless a declared hand is over the box, whose own points
can leak into the map: then a lost view and the freeze TTL runs) — or the declaration ends, is
retracted or expires, or the grid goes stale beyond `freeze_s` (a confirming grid re-stamps the
box with its `source_stamp`; with no confirming grid it ages out `freeze_s` after the last one).
Before the first acceptance nothing changes: every gate decides what is accepted. After it,
re-fits keep running (diagnostics, the instance tracker) and their verdict is logged once per
change of outcome, `grasp target re-fit ignored, region frozen (map-confirmed) — <the gate's
reason>`. The camera path (no pre-measurement) is frozen the same way (§3.2). The attach,
handover, payload and release paths are unchanged. Safety argument and what is given up: §3.7.

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

1. **Nothing held — acceptance.** The candidate is the tracked instance nearest the armed hand's
   TCP points inside the search column, refused on a tie within one voxel (`nearest_primitive`,
   HZ-0115-2); for a named search box, the lone instance inside it (two: `ambiguous`; none: the
   lost view `no_instance`). A named target also requires the **whole** measured instance to fit
   within its search box plus one voxel (`outside_named_search_box`). The candidate must be
   confirmed on the current grid (`map_confirms`: covered, one body) and pass
   `_gate_refit(complete=False)`; then it is accepted. On the arming tick the pre-measured
   instance is the region before the next chunk.
2. **Held — frozen (Revision 4).** Every tick, `_map_hold` runs `map_confirms` on the held box and
   its measured support: `map_disagrees` retracts; `map_split` retracts unless a declared
   contact link's TCP is over the box (`_hand_over_target`), where it is a lost view (the freeze
   TTL runs from the box's stamp, nothing re-stamps it); confirmed, the same box is re-accepted
   stamped with the grid's `source_stamp` — never moved, never grown. A newer re-fit of the
   instance still runs every gate of step 1 against the held box (`_gate_instance`, tracking,
   `occluded_refit`, `target_moved`, `unoccluded_refit`), but only its outcome is logged
   (`_refit_ignored`, once per change).

The camera path (`premeasure` off, `_tick` / `_on_reply`) is frozen the same way: `_hold_frozen`
runs `_map_hold` on each tick's grid, and once a region is held every re-fit refusal
(`_refuse_refit`: seed, request, deadline, reply gates) and every re-fit that would have replaced
it is logged as ignored instead of applied. Its first acceptance is unchanged (including the
completion from the map); the held region then meets the same per-tick map rule — a completed
box holds one body by construction (completion is refused when its closure holds another body's
cell). A tick that finds a request in flight takes no grid (as before), so on that path the map
check runs on every tick with no request in flight: the gap is at most `deadline_s` (0.25 s).

### 3.3 Staleness — a region never outlives its evidence

| evidence | lifetime |
|---|---|
| instance (camera fit) | dropped `freeze_s` after its fit, or when the map stops covering it |
| region from an instance | its stamp is `min(depth frame, grid source_stamp)` of the fit; the freeze TTL and the kernel's `grasp_region_max_age_s` run from it, as for any camera fit |
| held (frozen) region (Revision 4) | re-stamped to each confirming grid's `source_stamp`, hand or no hand; dies `freeze_s` after the last one; at once when a grid no longer covers it (`map_disagrees`) or holds a second body in it with no declared hand over it (`map_split`); a stalled octomap ages it out (`grid_source_stale`, the kernel's bound) |
| any region | retracted with the declaration (end, retraction, expiry), at `approach_ended`, as today; camera re-fits never retract it once held |

The frozen region is the one place a region lives longer than its last camera fit: on map
evidence the kernel already bounds by `source_stamp`, for the life of the declaration. That is
the posture change §6 asks the Safety-WG to accept.

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
| hand points leak into the map | with a declared hand over the frozen box its `map_split` is a lost view; the box is never replaced | freeze TTL bounds it |
| second body slides into the frozen box | `map_split`, retracted at once (hand not over the box) | only the target is ever exempt; the map shows the second body within its freshness deadline |
| object removed | held region retracted (`map_disagrees`), instance dropped | the region dies with the map evidence |
| object moved off the frozen box | `map_disagrees` once the box no longer holds `min_cover` of its footprint's cells | the region dies with the map evidence |
| object nudged within the frozen box (e.g. two voxels inside its padding) | kept: the box still holds one body, the target | the exemption still covers only the target's cells; a part pushed out of the box is not exempt, so the kernel stops the finger there (stricter, never looser) |
| camera re-fit disagrees while the hand occludes (Revision 4, Isaac s_mix07, t_mix04, t_mix02) | ignored, logged once per outcome (`re-fit ignored, region frozen (map-confirmed) — target_moved / outside_named_search_box / …`) | **posture change**, see the safety argument below |
| re-fit with no hand near (a person's hand, the target tipped) | ignored, logged (`unoccluded_refit` / `target_moved`) | **given up** (below); the map catches the outcomes that change what the box holds |
| instance not re-seen | dropped after `freeze_s` | an instance that could not stand a live region is not kept |
| stalled octomap | `grid_source_stale`; nothing pre-measured; every region ages out | unchanged kernel bound |

**Safety argument (Revision 4).** The exemption box is a measured, map-confirmed box: a camera
mask originated it, every acceptance gate passed it, and the map confirmed it held one body. While
frozen, the map re-checks exactly that on every fresh grid: a removed or moved-away object is
caught within the map's freshness deadline (≤ `grid_max_age_s` + one octomap integration), the
same bound as before Revision 4; a second body inside the box retracts it (`map_split`) unless a
declared hand is over the box; `source_stamp` + `freeze_s` bound a stalled map; the kernel
contract (no region, no exemption; `grasp_region_max_age_s`; the caps) is unchanged and the box
never grows, so nothing outside the confirmed box is ever exempt.

**What is given up** is camera evidence after arming. A re-fit that contradicts the frozen box no
longer retracts it: a person's hand at the target, the target tipped or nudged inside its box,
or a second object slid into the box while the declared hand is over it (where leaked hand
points and a real second body look the same) stay exempt until the map shows the change — a
second body with the hand away, or the box no longer covered. A second object entering the box
during the same octomap interval in which the target leaves it would be a one-body box again;
this needs two world changes inside one map update and was equally open under the Revision 2/3
hold. **Safety-WG review and a hazard-log row in the management repo are owed** for this
posture (not yet assigned an id); until then it inherits the parameter's default-off posture.

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
the freeze TTL from that fit), and once accepted the map alone **holds** it for the
declaration's life (Revision 4: camera re-fits are logged, never applied). **An ADR amendment and hazard-log rows are required before the parameter is turned
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
> from it. (3) Once accepted, the region is frozen for the declaration's life (Revision 4):
> camera re-fits are logged, never applied; a newer grid that confirms it re-stamps the same box
> (`source_stamp`); a grid that no longer covers it, or holds a second body in it with no
> declared hand over it, retracts it. (4) An instance is dropped `freeze_s` after its last fit or when the map stops
> covering it. (5) The kernel contract is unchanged: no region, no exemption; the region age
> bound and `freeze_s` are unchanged; `PlaceRegion.geometry` stays empty on grasp regions. (6)
> Production posture: off by default; on in a committed scene only with this ADR accepted.
> *Consequences.* The region exists on the arming tick (named targets: within a prompt or two of
> dispatch). Packed objects are separated by their masks. The camera's `not_on_support` gate is
> kept. Hazard rows: HZ-0115-35 (a pre-measured instance on the wrong object — tie rule, one
> hand, map confirmation, freeze TTL from the fit), HZ-0115-36 (a packed neighbour's sliver
> inside an instance's padded box — residual, bounded by the camera padding and the cell
> closure), HZ-0115-38 (the map hold keeps a region alive past its last camera fit — same box
> only, retracted when the map stops covering it or splits it, bounded by `source_stamp` and
> `freeze_s`; leaked hand points never replace it; Revision 4 extends it to the declaration's
> life against disagreeing camera re-fits — its own row owed).

## 4. Prototype (this branch)

- `python/hal/src/openral_hal/_object_primitives.py`: `instance_prompt`, `map_confirms`,
  `nearest_primitive`, `point_box_gap_m`, `PrimitiveTracker` (`update`, `prune`), `PrimitiveFit`,
  `ObjectPrimitive`. Pure numpy; reuses `_grasp_target`'s lattice, components, cover and tracking.
- `GraspTargetLeg(premeasure=True)`: `_premeasure` (prune, one prompt, `_request` with no
  declaration), `_on_instance_reply` (`_fit` → `map_confirms` → tracker), `_tick_instances` (the
  armed region from the instances, `_gate_instance` → `_gate_refit(complete=False)`; once held,
  `_map_hold` and `_refit_ignored`). The camera path's freeze: `_hold_frozen`, `_refuse_refit`. Every request is a
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
  staleness drops the rest), `::test_a_named_search_box_grounds_to_the_lone_instance_in_it`,
  `::test_a_map_confirmed_grasp_target_is_frozen_and_only_the_map_retracts_it` (Revision 4: the
  s_mix07, t_mix04 and t_mix02 re-fits on the s_mix07 cell kept and logged as ignored; map
  cleared, carton moved off the box, second body with the hand away → retracted; nudged inside
  the box, second body with the hand over it → kept; stale grids → `freeze_ttl`; declaration
  retracted → cleared) and `::test_the_camera_path_freezes_its_held_region_the_same_way`.

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
   HZ-0115-35/36/38) for sim evaluation, and the production posture. Revision 4 freezes a
   map-confirmed region for its declaration's life and lets only the map retract it (§3.7): it
   gives up camera evidence after arming, so it needs its own Safety-WG review and a hazard-log
   row in the management repo (not yet assigned; it supersedes Revision 3's, which was owed for
   the narrower approaching-hand overrule).
2. Where the store lives long-term: in the HAL leg (one consumer, no new topic) or published from
   world state for the reasoner (grounding by instance id) — the latter needs the ADR's
   layer-crossing clause and an `ObjectPrimitiveSet` IDL.
3. The pre-measurement budget on the shared GPU (`_PREMEASURE_MAX_HZ`, the reach): measured in the
   Isaac evaluation against the policy's chunk latency.
4. Whether a prompt-free instance mode (SAM automatic masks on the head frame) should replace the
   map-seeded prompts once the segmenter node offers one.
