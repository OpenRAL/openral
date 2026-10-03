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
- The exemption dies on: retraction, `timeout_s`, future stamp, stale world state, grid-frame
  change, rejected attachment set, non-allowlisted link, frame mismatch, oversize/degenerate
  region, non-empty `geometry`, detach, or the payload origin (FK of the measured configuration)
  leaving the region after attach (handover to ADR-0092 attached geometry + bridge payload
  clearing), or an attach on the declaring gripper of an object the declaration does not name
  (`handover_object_mismatch` — a rejection, WARNed once per declaration as
  `grasp_region_rejected` with the declared object, attached label, target, rskill and trace).
- The handover binds only to a payload attached on the declaring gripper's own chain — a declared
  contact link or a non-root ancestor of one. The other hand's payload and a released payload
  frozen on the collision root (the base) can neither be the handover nor retire it.
- Feature parameter `grasp_allowance_enabled` defaults off.
- **This amends ADR-0097's "arm-vs-world unchanged" invariant for the declared contact links
  only.**

**Bounds (WG).** Region half-extent ≤ [0.20] m; volume ≤ [0.03] m³; `timeout_s` ≤ [120] s;
region measurement age ≤ [X] s; `geometry` empty in v1.

**Consequences.** A real grasp becomes possible once the real producers exist. New hazard
HZ-01xx. Follow-up (not a precondition): per-finger collision geometry
(`collision-geometry-review.md` §8.3) would let the exemption be replaced by zero-clearance
adjudication against the target body, as ADR-0098 does for places.

**Judgement calls for the WG.** (1) Full per-cell exemption now vs waiting for per-finger
geometry. (2) The caps. (3) Measure-once vs continuously re-measured region, given the gripper
occludes the target at close-in. (4) Handover retirement rule and whether a failed grasp may
re-arm within one goal. (5) Whether `link7` (the hand body) is a contact link — check the fitter
output first. (6) The producer's prompt source (scene-named + search box recommended). (7) Whether
ADR-0100's force gate should arm during close as additive conservatism.

## HZ-01xx — Grasp-target exemption misapplied

| ID | Hazard | Cause | Mitigations |
|---|---|---|---|
| HZ-01xx-1 | Finger link contacts a non-target body (a hand, a neighbouring object) inside the declared region without a stop | Exemption is per cell, not per body | Region small (caps) and measured tight to the target; only the declared gripper's contact links exempt, arm links keep the full margin on the same cells; attended operation + hardware E-stop (mandatory on the cell); disclosure in logs/diagnostics/spans. Residual risk accepted by the WG or reduced later by ADR-0100 force gating during close. |
| HZ-01xx-2 | Wrong object / wrong region declared | Dispatch error or mis-segmentation | Dispatch can never supply a region; `evidence_ref`, `rskill_id`, `trace_id` logged at arm time; the producer checks the measured region against the declared target hint and the occupied cells it covers; frame mismatch refused. |
| HZ-01xx-3 | Stale declaration outlives its goal or grasp | Dispatcher crash, producer stall, missed retraction | Goal-scoped retraction on every runner exit incl. E-stop; `timeout_s` backstop per candidate; region-age bound; world-state freshness; position-based handover retirement; an attach of an undeclared object on the declaring gripper retires the declaration at once (`handover_object_mismatch`) instead of leaving it alive to `timeout_s`; future stamps dead. |
| HZ-01xx-4 | Target moved after measurement; exemption covers vacated space or a new arrival | Measure-once region | Region-age bound and re-measurement, or a short TTL (WG). |
| HZ-01xx-5 | Exemption leaks to other links, arms or robots | Configuration error | Static allowlist resolved at configure (unknown link fails configure); declaration links must be a subset; intersection mask; bimanual test. The handover binds only to a payload on the declaring gripper's chain (contact link or non-root ancestor), never the other hand's payload or a release record frozen on the base, so another arm's attachment can neither extend nor end this gripper's exemption. |
| HZ-01xx-6 | Fingers driven into the support surface under the target | Region extends into the support plane | Producer obligation that the region's lower face sits above the support plane; kernel test pins that support cells outside the region still stop. |
| HZ-01xx-7 | The exemption silences the graded velocity band for the whole chunk | An exempt finger inside its target reads a negative distance; the sweep keeps one minimum, and the band discards a negative slack as "tripped", so every non-exempt pair's slowdown is lost with it | The band clamps an untripped check's slack to `max(slack, 0)`, so any exempt pair (grasp target, support witness, embedded residue) reads as slack 0 — the band's slowest rate, never full speed; lifecycle tests pin the scaled chunk with the exempt finger inside its target and with a payload resting on its witnessed support. |

Cite alongside: the existing self-filter shell hazard (2 cm padding around the swept finger hull
already blinds the map near the jaws).

## Attached-payload check on real hardware (prerequisite; §3 row 4 of the design note)

Turning `attached_collision_enabled` on for real (coupled to the vision leg, 1000 ms deadline)
changes drop semantics and trusts vision-only, low-confidence geometry for map clearing.
Hazard-log entry to cover: (a) an undersized vision box clearing a real obstacle from the voxel
map; (b) the window from a dead effort channel to a kernel drop (heartbeat gate ×
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
