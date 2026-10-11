# Manipulation-workspace contact — completing contact-rich tasks without giving up collision safety

Status: **design draft, pending Safety-WG review**, 2026-10-11. Issue #349, PR #370. Builds on the
grasp-target exemption (ADR-0115), the place allowance (ADR-0097) and the measured-region rule
(`DeployScene._reject_scene_supplied_place_region`). Nothing here is implemented yet; §6 is the
plan. Hazard-log rows and the ADR are owed in `OpenRAL/management` (no ids assigned here).

## 1. Problem

A learned manipulation policy (π0.5 restock, bimanual OpenArm, joint-position chunks at 30 Hz)
completes its task only by touching the world. The safety kernel stops on any overlap between
conservative link geometry and the occupied voxel map (15 mm cells, margin 0 in sim, 20 mm real),
with one scoped relaxation per declared, measured region: the grasp target (fingers) and the place
target (carried payload). Every Isaac deploy-sim run of the restock task ended in a kernel stop
before the grasp, while the same prompts place the box without the kernel.

## 2. Evidence (Isaac Sim 6.1, restock tote scene, 2026-10-11)

**Kernel stops, adjudicated against PhysX ground truth** (`sim.estop_ground_truth_snapshot`,
`docs/reference/sim-environments.md`): 44 stops over 40 deploy-sim trials, 8 prompts. In all but
one, PhysX reported **0 mm penetration** at the stop; the nearest real surface was 0.6–10 mm away
or nothing within the contact band. Causes, in the order they were removed:

| cause | fix | effect |
|---|---|---|
| no grasp region before contact (VLA never pauses) | prompt-grounded `grasp_declaration.search_box` | 8/8 regions before contact |
| finger model = both fingers swept over the stroke (84 mm max excess) | rigid leader + mimic-joint follower (`docs/reference/finger-mimic-geometry.md`) | 16.9 mm max excess |
| occluded camera re-fits retracting the held region at contact | freeze a map-confirmed region for the declaration (`object-primitives-design.md` Rev. 4) | no camera-driven retractions |
| 15 mm cell quantisation | 10 mm cells measured: no fewer stops, 3.4× grid bandwidth, 2× octomap CPU | kept 15 mm |
| a 1 cm finger penetration cap | measured: delays stops 0.3–1.3 s, moves two to the wrist | dropped |

**What a task actually needs** (`openral sim run`, no kernel, `EpisodeResult.contact_stats` —
PhysX robot-vs-world contacts every step; 8 episodes, 3 placed completely):

- **Only the hand touches the world:** the fingers (`ee_link1/2`) and the hand body
  (`ee_base_link`, the kernel's `link7`). **No arm link (link1–link6) touched anything in any
  episode.**
- The hand touches: the target carton (19–41 mm PhysX penetration: soft cartons, the grasp
  squeeze, all 8); **neighbouring cartons** (7–27 mm, 6/8 — in a packed tote the policy squeezes
  the neighbour to get the target out); the staging board under the cartons (0 mm, 8/8); tote walls
  and rims (0–0.5 mm, 4/8); shelf rails while placing (0 mm, 7/8).

So exempting the target alone can never let the task complete, and a depth tolerance cannot either
(commanded poses run 20–40 mm into soft cartons that physics blocks). What the data does show is a
clean partition: **contact is needed only by the hand, only inside the volumes the task works in.**

**Contact speeds** (`EpisodeResult.contact_stats.onset_speed_mps` / `max_speed_mps`, link body
speed over the control period; 16 sim-run episodes, 4 placed completely in the second batch):
only cartons and tote walls/rims exchange force with the hand — every shelf and board "contact"
carries zero impulse (the fingers skim inside PhysX's contact-offset band). For the pairs that
exchanged force:

| hand-class contact | fastest onset | peak while in contact |
|---|---|---|
| finger vs carton | 0.135 m/s | 0.66 m/s (carrying the target — grasp/attach path) |
| finger vs tote | 0.135 m/s | 0.32 m/s |
| hand vs carton | 0.160 m/s | 0.51 m/s (holding) |
| hand vs tote | 0.090 m/s | 0.22 m/s |

The only non-hand body ever in contact was `openarm_body_link0` (the fixed base), once, at zero
speed and zero impulse, touched by the carried carton.

## 3. Rule

Contact classes, per (robot link, world cell, task state):

| robot link | target region (ADR-0115) | **declared, map-confirmed manipulation workspace** | everything else |
|---|---|---|---|
| arm links, torso, base | stop | **stop** | stop |
| hand links (fingers, follower, `link7`) | exempt | **exempt while a task declaration is live and the hand is slow** | stop |

- **Hand links** = the manifest's gripper leader/follower links plus the link that carries them —
  derived from the manifest (`RobotDescription.grasp_contact_links()` + the gripper's parent),
  never configured by name.
- **Manipulation workspace** = a box the scene declares per task phase (the source tote interior
  for a pick, the shelf slot for a place) — like `search_box`, the scene may *declare* it but never
  supply the exemption. A producer **confirms** it against the voxel map before it reaches the
  kernel (the declared fixture walls are occupied where declared, within one voxel, on a fresh
  grid) and retracts it when the map disagrees — the same "declared, then measured" rule as the
  grasp and place targets.
- **Slow** = the hand's commanded Cartesian speed (FK of the checked row vs the measured state,
  over the control period) at most `v_contact`. Calibration (§2): sim evaluation **0.35 m/s**,
  just above the fastest sustained non-target contact (finger sliding on the tote, 0.32 m/s);
  candidate for real hardware **0.25 m/s**, ISO 10218-1's reduced speed — a Safety-WG decision.
  Set explicitly by the scene; there is no default. Faster motion inside the workspace is checked
  as today. On a PD position-controlled arm the commanded-minus-measured gap is proportional to
  the force the joint drives apply, so a hand blocked by a wall while the policy keeps pushing
  grows the gap and trips the bound: it is also a crude push-force limit.
- **Self-collision, the arm links, any hand contact outside the workspace, and any contact after
  the declaration ends are unchanged.** This is what keeps catastrophic collisions — arm into
  shelf/table/person, fast hand strikes, hand outside the task volume — stopped.

## 4. Why this is sound enough to propose

- **Scope is measured, not asserted:** the exemption volume is map-confirmed, bounded by the
  declaration's lifetime and the grid's freshness deadline (as ADR-0115).
- **The relaxed class is the one the task needs and nothing else:** the sim-run data shows zero
  arm-link contacts; arm links lose nothing.
- **Energy, not geometry, bounds the relaxed contacts:** inside the workspace the limit is hand
  speed (the ISO/TS 15066 power-and-force-limiting idea, applied to objects — the standard itself
  covers people only, so this needs its own hazard analysis).
- **Fail-closed:** no confirmed workspace, a stale grid, a missing measured state (speed unknown),
  or no live declaration → today's behaviour.

## 5. What is given up (hazard-log input)

- A person's hand inside the declared tote or shelf slot, touched by the robot's hand at or below
  `v_contact`, is not stopped by the geometric check. Mitigations to evaluate: the speed bound; a
  force/current contact layer (`~/workspace/openral-349-notes/contact-layer-and-unknown-space.md`);
  operator presence rules on the real cell.
- Unknown space inside the workspace is treated as free, as everywhere today (record as an
  explicit assumption).

## 6. Plan

1. ADR draft + hazard rows (management repo).
2. Schema: a declared `workspace_box` per task phase on `GraspDeclaration` / `PlaceDeclaration`
   (scene-declarable, like `search_box`; region rejected from scenes as today).
3. Producer: confirm/retract the workspace against the voxel map (HAL leg, next to the grasp and
   place producers); publish it with the declaration.
4. Kernel: hand-link mask from the manifest; per-cell workspace containment; hand speed from the
   checked row; disclosure on every exempted contact (log + span), never silent.
5. Measure `v_contact` from the sim-run contact data (hand speed at the contact steps).
6. Validate: the same 8 restock prompts in deploy sim with ground truth — target: completion rate
   approaching `sim run`'s, zero arm-link stops missed, and a deliberate arm/shelf collision still
   stopped.
