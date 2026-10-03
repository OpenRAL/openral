# Drafts for the Safety-WG: real pick-and-place on the OpenArm cell

Companion to [`real-pick-place-design.md`](real-pick-place-design.md) §4. These are drafts for the
private `OpenRAL/management` decision log and hazard log; numbers in brackets are placeholders the
WG sets. Nothing here is in force.

## ADR-01xx — Declaration-scoped grasp-target contact for gripper finger links

**Context.** The real OpenArm cell runs the world-voxel check at a 20 mm margin on 20 mm cells.
`check_voxel_collision` (`cpp/openral_safety_kernel/src/collision.cpp` L1240-1380) has no
exemption, so the finger link stops on the grasp target's own cells before any attachment can
exist; the kernel can therefore never reach the post-attachment mechanisms of ADR-0092/0097/0098.
The OpenArm finger link (`openarm_<side>_finger_pair`) is one convex hull swept over the stroke,
so the target lies inside it during a grasp; a margin reduction capped like ADR-0097
(min(1.5 voxel, 40 mm) = 30 mm at 20 mm cells) cannot admit that, and 20 mm cube quantisation plus
the 20 mm margin already exceeds it. Lowering the global margin (HZ-0095-2 class; it also derives
the extrinsic gate in `openral_core.depth_extrinsic`) and masking the target out of the map
upstream (object invisible to every consumer; a safety decision outside the kernel; breaks the
bridge's "an occupied cell is an obstacle" invariant) were considered and rejected.

**Decision.**
- Add `GraspDeclaration` (msgs + core), named by dispatch and region-measured by the attachment
  evidence producer, carried on the `AttachmentState -> WorldStateStamped` envelope beside
  `PlaceDeclaration`. Dispatch never supplies a region (stripped in the runner, refused by the
  producer).
- While it is live, cells whose centre lies in the producer-measured oriented region box are
  exempt **for the declared gripper's contact links only** — the intersection of a launch-derived
  manifest allowlist (the `role: gripper` joints' `child_link`s) and the declaration's
  `contact_links`. An exempt (link, cell) pair never trips and never supplies the reported
  identity; it reaches `sweep_min` at its own (negative) depth, and the kernel's velocity band
  clamps an untripped check's slack to 0, so a finger inside its target reads as the band's
  slowest rate and can never hide the graded slowdown every non-exempt pair still earns (the same
  clamp covers the support-witness and embedded-residue exemptions on the attached path). An exempt pair never spends
  the check's shared stage-2 refinement budget (it cannot trip, so stage 1 suffices), so a finger
  buried in its target cannot starve a non-exempt link of the exact hull distance and false-stop
  it. Net invariant: the region never adds a stop, never removes one for a non-exempt pair that
  exact geometry upholds, and never raises the band slack of a sweep that would have passed
  without it — it may only slow. It may lower `sweep_min` (an exempt hull link reports stage 1's
  bound rather than the refined distance), which errs slower.
- All other links, all cells outside the region, self-collision, attached checks and the force
  gate are unchanged. The support surface under the target is outside the region (producer
  obligation: the region's lower face sits above the support plane).
- The exemption dies on: retraction, `timeout_s`, future stamp, a region measurement
  (`region.stamp_ns`) older than `grasp_region_max_age_s` or stamped in the future
  (`region_stale`, until the handover latches the box — see below), stale world state, grid-frame
  change, rejected attachment set, non-allowlisted link, frame mismatch, oversize/degenerate
  region, non-empty `geometry`, detach, or the payload origin (FK of the measured configuration)
  leaving the region after attach (handover to ADR-0092 attached geometry + bridge payload
  clearing), or an attach on the declaring gripper of an object the declaration does not name
  (`handover_object_mismatch` — a rejection, WARNed once per declaration as
  `grasp_region_rejected` with the declared object, attached label, target, rskill and trace).
- The handover binds only to a payload attached on the declaring gripper's own chain — a declared
  contact link or a non-root ancestor of one. The other hand's payload and a released payload
  frozen on the collision root (the base) can neither be the handover nor retire it.
- Region measurement age (implemented): `grasp_region_max_age_s` / `place_region_max_age_s`,
  default `0` = 2 × `world_voxel_deadline_ms` (2 s at the 1 s schema default), capped at
  2 × the kernel's voxel-deadline cap (4 s); configure refuses a resolved bound outside that
  range while the matching allowance (grasp: `grasp_allowance_enabled`; place:
  `attached_collision_enabled`) is on. `deploy_e2e` passes both as 2 × `world_voxel_deadline_s`.
  Checked at ingest (the region is dropped, `reason=region_stale` logged once) and again per
  candidate, against the kernel clock. **Not applied after the handover latch:** the box is
  frozen there and producer updates are ignored, the fingers occlude the target so it cannot be
  re-measured, and the exemption is then bounded by the payload origin (measured FK) staying in
  the latched box, the stream deadline and `timeout_s`; requiring a fresh measurement would
  retire a valid handover with the fingers closed on the target. The latched box itself must
  have been fresh at the handover edge.
- **No named target required (approach-armed).** The policy decides what to grasp. With the
  HAL's `grasp_target_approach_m` set (default off), the runner arms a region-less goal-scope
  declaration naming every hand and no target; the producer narrows it to the ONE hand whose
  TCP comes within the approach distance of occupied voxels (`target_id="approach:<link>"`,
  that hand's links only), searches a box around its jaws, and measures exactly as for a named
  target (measured support, `not_on_support`, ambiguity refusal, SAM point prompt, map cover,
  tracking, freeze TTL); the hand leaving retracts it at once. The reasoner's named
  `grasp_target` stays an optional hint that wins (search box) or narrows (hand). Arming is
  proximity-only — the exemption is needed while the open jaws straddle the target, before
  any close command — and the kernel now enforces one hand per declaration
  (`reason=links_span_hands`). It still never trusts a region it did not get from the
  producer's measured envelope; it never required the dispatch relay.
- Feature parameter `grasp_allowance_enabled` defaults off.
- **This amends ADR-0097's "arm-vs-world unchanged" invariant for the declared contact links
  only.**

**Bounds (WG).** Region half-extent ≤ [0.20] m; volume ≤ [0.03] m³; `timeout_s` ≤ [120] s;
region measurement age ≤ [2 × `world_voxel_deadline`, ≤ 4] s pre-handover (implemented,
`grasp_region_max_age_s`); `geometry` empty in v1.

**Consequences.** A real grasp becomes possible once the real producers exist. New hazard
HZ-01xx. Follow-up (not a precondition): per-finger collision geometry
(`collision-geometry-review.md` §8.3) would let the exemption be replaced by zero-clearance
adjudication against the target body, as ADR-0098 does for places.

**Judgement calls for the WG.** (1) Full per-cell exemption now vs waiting for per-finger
geometry. (2) The caps. (3) Measure-once vs continuously re-measured region, given the gripper
occludes the target at close-in. (4) Handover retirement rule and whether a failed grasp may
re-arm within one goal. (5) Whether `link7` (the hand body) is a contact link — check the fitter
output first. (6) The producer's prompt source (scene-named + search box recommended).
~~(7) Whether ADR-0100's force gate should arm during close as additive conservatism.~~ *Struck
2026-10-03:* there is no force signal to gate on — the real OpenArm gripper's effort is hard-coded
0.0 by the vendor driver (design note §1.6) and no joint declares a torque sensor. The grasp
event is a position stall (HZ-01xx-8/9 below), not a sensed contact force.

## HZ-01xx — Grasp-target exemption misapplied

| ID | Hazard | Cause | Mitigations |
|---|---|---|---|
| HZ-01xx-1 | Finger link contacts a non-target body (a hand, a neighbouring object) inside the declared region without a stop | Exemption is per cell, not per body | Region small (caps) and measured tight to the target; only the declared gripper's contact links exempt, arm links keep the full margin on the same cells; attended operation + hardware E-stop (mandatory on the cell); disclosure in logs/diagnostics/spans. Residual risk accepted by the WG (no force signal exists on this gripper to gate a close on, design note §1.6) or reduced later by per-finger geometry. |
| HZ-01xx-2 | Wrong object / wrong region declared | Dispatch error or mis-segmentation | Dispatch can never supply a region; `evidence_ref`, `rskill_id`, `trace_id` logged at arm time; the producer checks the measured region against the declared target hint and the occupied cells it covers; frame mismatch refused. |
| HZ-01xx-3 | Stale declaration outlives its goal or grasp | Dispatcher crash, producer stall, missed retraction | Goal-scoped retraction on every runner exit incl. E-stop; `timeout_s` backstop per candidate; region-age bound (`grasp_region_max_age_s`, at ingest and per candidate, pre-handover); world-state freshness; position-based handover retirement; an attach of an undeclared object on the declaring gripper retires the declaration at once (`handover_object_mismatch`) instead of leaving it alive to `timeout_s`; future stamps dead. |
| HZ-01xx-4 | Target moved after measurement; exemption covers vacated space or a new arrival | Measure-once region | Region-age bound (implemented: `grasp_region_max_age_s` / `place_region_max_age_s`, default 2 × `world_voxel_deadline`, ≤ 4 s; a stale or future-stamped region exempts nothing, `reason=region_stale`) and re-measurement. Residual: after the handover latch the age bound no longer applies (the target is occluded and the box frozen); the payload-in-box rule, stream deadline and `timeout_s` bound that window (WG). |
| HZ-01xx-5 | Exemption leaks to other links, arms or robots | Configuration error | Static allowlist resolved at configure (unknown link fails configure); declaration links must be a subset; intersection mask; bimanual test. The handover binds only to a payload on the declaring gripper's chain (contact link or non-root ancestor), never the other hand's payload or a release record frozen on the base, so another arm's attachment can neither extend nor end this gripper's exemption. |
| HZ-01xx-6 | Fingers driven into the support surface under the target | Region extends into the support plane | Producer obligation that the region's lower face sits above the support plane; kernel test pins that support cells outside the region still stop. |
| HZ-01xx-8 | Stall **false positive**: an ATTACH with nothing (or the wrong thing) between the jaws | The position-stall trigger reads any obstruction that stops a closing jaw short of its command — the table, a shelf lip, the other hand, the target's edge before it is seated — as a grasp | Vision AND at ATTACH: the grasp-target region payload only when the jaw is at the region, else a gated `SegmentInView` mask; an unconfirmed stall still attaches only the low-confidence `GRIPPER_CLOSURE` jaw box (collision-conservative; WG to judge the map-clearing cost); settle + debounce (5 + 3 ticks); the kernel's `handover_object_mismatch` retires a declaration when the wrong object attaches. |
| HZ-01xx-9 | Stall **false negative**: a held object never attaches | An object thinner than `stall_gap` in jaw angle (card, cable) stops the jaw too close to its command; a command the controller does not actually track (interpolation, clamping) shifts the gap | Per-side calibration (`JointSpec.closure_calibration`) measured attended (design note §5); the carried object stays in the voxel map, so the world check still sees it — the failure is a stop, not a silent collision; thin objects are out of scope until measured. |
| HZ-01xx-10 | The commanded target the trigger compares against is stale | No action applied since bringup / e-stop; a slot that never names the gripper | The trigger cannot attach without a command (`uncommanded_ticks` counted); only JOINT_POSITION rows naming the gripper joint update it; detach is trigger-driven and needs no command beyond the last one. |
| HZ-01xx-7 | The exemption silences the graded velocity band for the whole chunk | An exempt finger inside its target reads a negative distance; the sweep keeps one minimum, and the band discards a negative slack as "tripped", so every non-exempt pair's slowdown is lost with it | The band clamps an untripped check's slack to `max(slack, 0)`, so any exempt pair (grasp target, support witness, embedded residue) reads as slack 0 — the band's slowest rate, never full speed; lifecycle tests pin the scaled chunk with the exempt finger inside its target and with a payload resting on its witnessed support. |
| HZ-01xx-8 | The exemption arms on an unintended object near the hand (no target is named) | Approach-armed target: the hand passes within the approach distance of a neighbour, or lingers by the wrong object, and the producer measures that one | Arms only around the approaching hand's TCP (box ≤ `MAX_HALF_EXTENT_M`, anchored on the column nearest the TCP), only for that hand's links, only with ONE hand approaching; the object must be measured standing on a measured support (`not_on_support`) and unambiguous (`ambiguous` refuses a comparable second cluster); region caps; the hand leaving retracts at once (`approach_ended`); kernel region-age bound (2 × voxel deadline) and goal-scoped retraction; the exemption covers only that object's cells for the finger links — arm links, the support and every cell beside the region still stop (live kernel row: a neighbour one column beside the target stops the exempt finger); attended operation + hardware E-stop. Residual: while the hand stays by the neighbour, its fingers may touch it without a stop (as HZ-01xx-1). Proximity-only arming (no closing-intent gate) is a WG call: the exemption is needed before the close. |
| HZ-01xx-9 | One exemption covers two hands | A producer bug declares both hands' links on one region (the goal-scope declaration names every hand) | Producer arms one hand at a time and refuses two approaching at once; the kernel refuses any region-carrying declaration whose contact links do not share one mount (`grasp_links_one_hand`, `reason=links_span_hands`, gtests + live row). |

Cite alongside: the existing self-filter shell hazard (2 cm padding around the swept finger hull
already blinds the map near the jaws).

## Attached-payload check on real hardware (prerequisite; §3 row 4 of the design note)

Turning `attached_collision_enabled` on for real (coupled to the vision leg, 1000 ms deadline)
changes drop semantics and trusts vision-only, low-confidence geometry for map clearing.
Hazard-log entry to cover: (a) an undersized vision box clearing a real obstacle from the voxel
map; (b) the window from a dead jaw-position channel to a kernel drop (heartbeat gate ×
`evidence_timeout_s` vs the deadline); (c) a phantom jaw-box fallback when the gripper closes on
nothing; (d) two segmentations in flight on one segmenter (bounded by per-leg deadlines).

## Place-side amendments (design note §2.3)

1. **ADR-0097 amendment:** a unit-surveyed, provenance-carrying fixture that is **verified live
   against the voxel map** before arming counts as a producer-measured region on a fixed-base
   robot. Touches HZ-0097-2/4 and the `_reject_scene_supplied_place_region` rationale for that
   case only.
2. **ADR-0092 D6 amendment:** a proximity-based `DECLARED_FIXTURE` witness (payload's lowest
   primitive within max(1 voxel, survey uncertainty) of a verified plane, gripper still loaded) is
   not sensed contact and is labelled so; review Entry 012's co-planar headroom at 20 mm cells
   (41 mm exempt height).
3. **ADR-0098 offset for joint play** (ship the fixture's box lowered by survey uncertainty + FK
   play bound), if adopted: a deliberate loss of conservatism versus truth, bounded by the blanket
   allowance.
4. **Frozen-release window:** the released object stays an attached, checked record (FK pose at
   the DETACH stamp) until every finger hull is > margin + 1 voxel from it, a timeout, a new
   ATTACH or goal end; the bridge keeps clearing it at zero padding meanwhile. Bounds need
   hazard-log calibration on real data.
5. **Finger allowance inside the place region** (fingers vs the shelf at the 20 mm link margin):
   the same exemption class as ADR-01xx; decide after measuring finger-shelf clearance in the
   attended runs.
6. **Bimanual attachment** (two attach links): shared with the pick half.
