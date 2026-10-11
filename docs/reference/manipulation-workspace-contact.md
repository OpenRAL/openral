# Manipulation-workspace contact — completing contact-rich tasks without giving up collision safety

Status: **phase 1 (the grasp phase / source tote) implemented, default off, sim-only, pending
Safety-WG review**, 2026-10-11. Issue #349, PR #370. Builds on the grasp-target exemption
(ADR-0115), the place allowance (ADR-0097) and the measured-region rule
(`DeployScene._reject_scene_supplied_place_region`). §6 is what is implemented, §7 the ADR draft,
§8 the owed hazard rows and calibration points. The ADR and the hazard-log rows are owed in
`OpenRAL/management` (no ids assigned here).

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
- **Slow** = two bounds, both required:
  - **Measured speed** (impact energy): the hand's Cartesian speed as **measured** — the
    `/joint_states` velocity carried one control period forward from the measured pose (§6) —
    at most `v_contact`. Impact energy is set by how fast the hand actually moves, not by the
    command: the rSkill runner's chunk executor jumps 0.5–1 rad at chunk boundaries, which the
    physical arm smooths. A commanded jump *into* an obstacle is the lead bound's job (such a jump
    makes a lead far over the bound). Without a fresh measured velocity the speed falls back to
    the **commanded** speed between consecutive commanded rows, which reads the hand faster. Calibration (§2): sim evaluation **0.35 m/s**, just above the
    fastest sustained non-target contact (finger sliding on the tote, 0.32 m/s); candidate for
    real hardware **0.25 m/s**, ISO 10218-1's reduced speed — a Safety-WG decision.
  - **Lead** (push force): the hand's displacement from its **measured** pose to the checked row,
    at most `workspace_contact_max_lead_m`. On a joint-PD position-controlled arm the
    commanded-minus-measured gap is proportional to the force the joint drives apply, so this
    bounds how hard the hand may push into what it touches. The speed bound alone would let a
    hand whose stationary target is buried deep in a carton keep pushing. Calibration: sim
    evaluation **0.05 m**, from the 3.3–4 cm free-motion tracking lag the Isaac stops measured
    plus margin; the real-hardware value is a Safety-WG decision (capped at 0.10 m).

  Both are set explicitly by the scene; there is no default. Faster or further-leading motion
  inside the workspace is checked as today.
- **Self-collision, the arm links, any hand contact outside the workspace, and any contact after
  the declaration ends are unchanged.** This is what keeps catastrophic collisions — arm into
  shelf/table/person, fast hand strikes, hand outside the task volume — stopped.

### 3.1 Static geometry: what the camera has not seen is not free

**Evidence (deliberate-collision probes, 2026-10-11).** Joint sweeps from the restock start
pose, published on `/openral/candidate_action` with no goal active, kernel and Isaac live. The
probe E-stops the graph itself when the kernel has not, so the sim HAL logs PhysX ground truth
at the end pose either way. Single-joint sweeps either hit joint-envelope limits
(`safety.envelope_violation`) or were stopped by the world check before contact. **Correction:**
two of them (`right_joint1` → +1.5, `left_joint2` → −1.5) were first read as "blocked by the side
wall, unchecked" because the joint ended far short of its command; with the arm's TF link poses
recorded, both arms were ≥ 18 cm from the walls — the joint was still converging (drive lag
against gravity), not blocked. Commit `66e7e766`'s message repeats that misreading.

The real case is a two-joint move that abducts the left arm and extends the elbow
(`left_joint2` → −1.5 then `left_joint4` → 0), carrying the hand out of the head camera's view
toward the cell's left wall (y = +0.55 m):

| | kernel | PhysX at the end pose |
|---|---|---|
| no static geometry | **no stop** | left `ee_base_link` and **`link6`** pressed against `Cell/LeftWall`: 0 mm separation, impulse 1.33 and 0.14 N·s |
| with static geometry | stopped at 2.7 s, before any contact | none — the finger reached the declared shelf-rack post first |

Unobserved space is free end to end (octomap → bridge → kernel); for the restock stage the
declared fixtures add about **59 400 cells the camera map lacks**.

**Rule:** a scene declares its static fixtures (walls, shelf frame, table, tote shell) as
geometry in the robot's base frame, and the world grid the kernel checks is the union of the
camera's occupancy and that declared geometry, always — regardless of what the camera has seen
or cleared. A scene may only **add** occupancy (the fail-closed direction), so declared static
geometry needs no measurement before it counts, unlike an exemption. The workspace rule then
applies to it like any other cell: hand links may graze the declared tote/shelf inside a live,
confirmed workspace, slowly; arm links stop on it everywhere. For Isaac scenes the fixtures come
from the stage's static colliders; for a real cell, from its measured cell description. The
safety envelope's `no_go_zones` are declared today but enforced nowhere — this layer is where
declared geometry becomes enforced.

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

## 6. What is implemented (phase 1: the grasp phase, source tote)

| piece | where | what |
|---|---|---|
| schema | `GraspDeclaration.workspace_box` / `.workspace_region` (`openral_core`), `GraspDeclaration.msg` (`workspace_box_valid`, `workspace_box`, `workspace_region_valid`, `workspace_region`) | Backward-compatible addition, no `schema_version` bump. A scene may declare `workspace_box` (like `search_box`); `workspace_region` is producer-only: `_reject_scene_supplied_place_region` refuses it from a scene and the rSkill runner strips it from every goal. Caps `MAX_WORKSPACE_HALF_EXTENT_M` 0.50 m / `MAX_WORKSPACE_VOLUME_M3` 0.25 m³ (WG placeholders, mirrored by the kernel); `geometry` must be empty. |
| producer | `GraspTargetLeg._confirm_workspace` / `published_workspace` (`openral_hal/_grasp_target_leg.py`) | Every leg tick: the live declaration's `workspace_box` is **confirmed** on a fresh grid (the region's freshness rules, `_fresh_grid`) when occupied cells centred in it cover ≥ `WORKSPACE_MIN_COVER` (10 %) of its footprint (`region_covers_occupied`), and published as `workspace_region` — the same box, never grown, stamped with the grid's `source_stamp` — on the attachment envelope the kernel already ingests (`fill`). Retracted at once when the map no longer holds it or is in another frame; on a stale/missing grid the last confirmation ages out `freeze_s` after its stamp. Independent of the target region. Each state change is logged once. |
| hand links | `RobotDescription.hand_links()` | `grasp_contact_links()` (leaders + mimic followers) plus the link each gripper hangs off. OpenArm: `openarm_{left,right}_finger_pair`, `_finger_follower`, `_link7`. |
| runtime | `DeployRuntime.workspace_contact_enabled` / `workspace_contact_max_speed_mps` / `workspace_contact_max_lead_m` | Enabled requires `grasp_allowance_enabled`, a speed bound in (0, 1] m/s and a lead bound in (0, 0.10] m (no defaults). `openral deploy sim` forwards all three; **`openral deploy run` and `deploy_e2e` on a non-sim HAL refuse it** (`ROSConfigError`, sim-only until Safety-WG review). The launch passes the kernel `workspace_contact_links` = `hand_links()` and `workspace_contact_control_period_s` = 1 / `action_spec.control_freq_hz`. |
| kernel | `check_voxel_collision`, `workspace_contact_exempts`, `ingest_workspace_region`, `link_displacement_bound`, `link_geometry_reach`; `SafetyKernelLifecycleNode::ingest_workspace_contact` / `workspace_live` / `workspace_hand_speed` / `workspace_hand_lead` / `workspace_command_fresh` / `measured_velocity_fresh` | See the rule below. Params validated at configure (refuse otherwise); WARN when on. |

**The trip rule as coded.** For a (link, occupied cell) pair at a checked row, after the ADR-0115
exemption (which keeps precedence), the pair does not trip iff the link is in
`workspace_contact_links`, the cell centre lies inside the live `workspace_region` (exact
point-in-OBB), the **hand speed** of that row is ≤ `workspace_contact_max_speed_mps` **and** the
**hand lead** of that row is ≤ `workspace_contact_max_lead_m`:

  D_l(A, B) = |Δp_l| + θ_l · reach_l   (link_displacement_bound)
  hand_speed        = max over hand links l of D_l(q_meas, q_meas + dq_meas · control_period)
                      / control_period                      (measured; the same for every row)
  hand_speed(row s) = max over hand links l of D_l(row s−1, row s) / control_period
                                                            (commanded fallback)
  hand_lead(row s)  = max over hand links l of D_l(measured, row s with the jaws at measured)

with Δ, θ the translation and rotation of link l's frame between the two configurations and
reach_l the radius of its collision geometry about its origin — an upper bound on how far any
point of the hand moved. The lead holds the jaw joints (`workspace_contact_jaw_joints`, the
manifest's `role: gripper` joints) at their measured value: it bounds the arm's push. A jaw still
closing on a held object has the rest of its stroke as commanded-minus-measured, which is a grasp
squeeze bounded by the gripper's effort limit, not by the arm (p_mix02: 72–80 mm follower "lead"
while the leader read 32 mm, stopped at 80.2 mm against the 80 mm bound, fingers on the target).
The measured hand speed likewise leaves out the jaw joints' velocity (p4_mix00: a jaw squeezed past
its closed limit jittered the fingertip speed 0.11 → 0.38 m/s within a second while the arm was
slow); the commanded fallback keeps it, reading faster, never slower.
Hazard-log assumption owed: the gripper's effort limit bounds the squeeze on real hardware.

**Measured velocity.** `/joint_states` `velocity[i]`, stored per dof with the message's arrival
time. A message naming a joint without a finite velocity entry (no `velocity` field, a shorter
one, NaN) forgets that joint's velocity. The measured speed is used iff every FK-relevant dof has a
velocity at most `collision_state_deadline_ms` old (the positions' deadline); else the commanded
fallback below. It is computed once per candidate: FK at q_meas, FK at q_meas + dq_meas · period.

**Row −1** of the commanded fallback (row 0's previous row), per joint the row commands: that joint's **previously accepted
command** — the last row of the last JOINT_POSITION chunk the kernel *accepted* that commanded it,
from any earlier tick or slot — while it is **fresh**, else the joint's **measured** value. Fresh
means: held, not cleared since, its last row due (acceptance + (horizon − 1) control periods) at or
before now, and due at most `kWorkspaceCommandFreshPeriods` = **2** control periods ago (node
clock). Held commands are **cleared** on any refused chunk (every drop path: an
`unavailable`, an advisory or a latching refusal — detected once per candidate from the drop
counter), an external E-stop latch, an E-stop reset, configure, activate, cleanup, and whenever the
live workspace declaration is armed or dropped (a new or ended declaration starts the chain
afresh). Joints the row does not command keep the value the check places them at (the measured
state; for the tick-partner check, the partner's committed target), so they add no phantom motion.
Why: the runner sends `horizon = 1` slot rows, so with the measured state as row −1 every row 0
compared the command with an arm that trails it by several control periods — the measured Isaac
stops read 0.99–1.20 m/s while PhysX had the hands at ≤ 0.66 m/s; the fallback can only read the
hand faster. Rows s ≥ 1 chain from row s − 1 as before. The tick-partner check uses the same rule
(partner joints at their committed targets in both configurations, the moving joints' row −1 as
above).

One max over all hand links (both hands of a bimanual robot), so a fast or far-leading other hand
revokes the exemption for both: conservative.

**Fail-closed (today's stop):** feature off; no live declaration (retracted, `timeout_s` lapsed,
future stamp, a **retired** identity — `handover_exit`, `released`, …); no `workspace_region`; a
workspace in another frame, with geometry, a non-unit quaternion, over the caps, or confirmed longer
ago than `grasp_region_max_age_s` (also between messages); a world-state stream past its deadline;
a stale or incomplete measured state; any control mode but JOINT_POSITION; a speed or lead above
its bound or non-finite. A missing or stale measured velocity is not a refusal but the commanded
fallback, which reads the hand faster. Arm links, cells outside the box, self-collision and the attached checks unchanged.

**Disclosure.** Tolerated pairs within the margin are counted per check
(`CollisionHit::workspace_pairs`); `safety.workspace_contact link= cell= depth_mm= speed_mps=
speed_source= max_mps= lead_m= max_lead_m= pairs= step= target=` WARN ≤ 1 Hz per link;
`safety.workspace_region_armed` (both bounds) / `_dropped` / `_rejected`;
`safety.workspace_contact_summary` (pairs, max depth, max speed tolerated and its
`max_speed_source`, max lead, both bounds) when the workspace drops; span attributes
`safety.workspace_contact_active` / `_pairs` / `_max_depth_m` / `safety.workspace_hand_speed_mps` /
`safety.workspace_hand_speed_source` / `safety.workspace_hand_lead_m`; and the `safety.collision`
stop line carries `workspace_contact_active=<0|1> hand_speed_mps=<v> hand_lead_m=<d>
hand_speed_source=<measured|commanded>`. The configure WARN states both bounds
and the freshness window.

**Deviations from the plan above, and why.**
- *Measured hand speed, not checked-row-vs-measured over one period, nor commanded speed.* The
  measured-relative form divides tracking lag (several control periods on a position-controlled
  arm) by one period: in Isaac every stop inside the workspace read 0.99–1.20 m/s against the 0.35
  bound while PhysX had the hands at ≤ 0.66 m/s, even carrying a carton (force-bearing contact
  onsets ≤ 0.16 m/s). The next step, the speed between consecutive commanded rows (row 0 from the
  previously accepted command), let the exemption fire (2–5 tolerated pairs per run, 4/4 runs) but
  every stop inside the workspace still read 0.91–1.10 m/s with a 3.0–3.7 cm lead: the chunk
  executor's non-RTC prefetch jumps 0.5–1 rad at chunk boundaries, spikes the arm smooths. The
  bound is about impact energy, so it uses the measured velocity; a commanded jump into an
  obstacle is caught by the lead bound. The commanded speed stays as the fallback.
- *A dedicated period param* (`workspace_contact_control_period_s`): the kernel's only other dt
  (`collision_seed_dt_s`) is the velocity look-ahead step and the launch deliberately keeps it 0.
- *Confirmation is a cover test* (≥ 10 % of the footprint occupied), not the "declared fixture
  walls occupied within one voxel" test sketched in §3 — simpler, and still "declared, then
  measured"; the wall test remains an option for the WG.

**Phase 2 (carried payload, sim):** a payload attached to a hand link is part of that hand. In
`check_attached_voxel_collision`, after the support-witness and attach-residue exemptions, a payload
cell is exempt by the same predicate on the payload's attach link (`workspace_contact_exempts`),
disclosed as `safety.workspace_contact link=payload@<attach link>`. The SPEED bound uses the
payload's reach for its attach link (`attached_object_reach`: the farthest primitive point from the
link origin), so a carried box's far corners count, not just the hand's. The lead keeps the hand's
own reach: it stands for the drives' push, set by the joint error, which a longer lever only
magnifies on paper (p3_mix02 read 95.6 mm with the payload's reach and stopped on it).
Why: p2_mix02 carried
the box to the shelf and stopped on the middle deck (payload vs a deck cell, −36 mm of a crude
box), while the place leg refused a region (62/323 footprint cells occupied — a box sliding onto a
shelf edge never has its whole footprint supported). The scene's `workspace_box` covers the source
tote and the prompt's target slot (one box — the hull of both; neighbouring slots inside the hull
are in it too, two boxes are the follow-up). Arm links still always stop; a payload against the
robot itself (`check_attached_self_collision`) is unchanged.

**Two hands meeting (sim):** `check_self_collision` takes the live workspace; a pair of two hand
links whose link origins both lie inside the box does not trip while the hands are slow (same speed
and lead predicate), counted and disclosed as `link=self@<a> cell=<b index>`. Why: across the place
batches at least 6 stops were the two OpenArm hands touching inside the tote/shelf workspace, real
on the MuJoCo meshes (0.07–1.9 mm overlap) at 0.10–0.20 m/s — Isaac runs with self-collision off,
so PhysX cannot see them. Arm links, and hands outside the box, stop as before.

**Not implemented:** a separate place-side box (`PlaceDeclaration.workspace_box`), velocity /
Cartesian control modes, the force/current contact layer.

## 7. ADR draft — "Declaration-scoped manipulation-workspace contact for hand links"

*Draft for the `OpenRAL/management` ADR log; no number assigned here.*

- **Context.** §1–§2: the ADR-0115 target exemption cannot complete a packed-tote pick; every stop
  in 40 Isaac deploy-sim trials was a hand link in the tote with no PhysX contact; `sim run`
  contact statistics show only hand links ever touch the world.
- **Decision.** Add a second, declaration-scoped relaxation of the arm-vs-world voxel check: hand
  links (manifest-derived) vs cells centred in a scene-declared, producer-confirmed workspace box,
  only while the hand's measured speed (joint-state velocity; the commanded speed between
  consecutive commanded rows when no fresh velocity is available) and its lead over the measured
  pose (the push bound, which also catches a commanded jump) are each at most a calibrated bound. Default off, sim-only until this ADR is accepted, ADR-0115 keeps precedence,
  every tolerated contact disclosed.
- **Layer impact.** Schema (Layer 0/2 boundary IDL: `GraspDeclaration`), HAL producer (Layer 0),
  safety kernel (Layer 5). No new layer dependency: the workspace rides the existing grasp
  declaration on the attachment envelope.
- **Consequences.** Contact by a slow hand inside the declared box is no longer stopped by the
  geometric check — including a person's hand there (§5). Arm links, fast hands and anything
  outside the box are unchanged.
- **Alternatives rejected.** A depth tolerance (commanded poses run 20–40 mm into soft cartons);
  exempting neighbours by identity (no per-object map); finer voxels (§2 table).

## 8. Owed before this leaves simulation

**Hazard-log rows (owed in `OpenRAL/management`, no ids assigned here):**
1. A person's hand (or any body not in the scene) inside the declared workspace, touched by the
   robot's hand at ≤ `v_contact`, is not stopped by the geometric check.
2. A wrong `workspace_box` in a scene (wrong pose or size) that the map still confirms (it holds
   ≥ 10 % occupied cells) relaxes contact in the wrong volume.
3. The hand-speed bound trusts the joint-state velocity: a biased or lagging velocity estimate
   (filtered encoder differencing, a simulator's reported velocity) reads the hand slower than it
   is. A commanded jump is bounded by the lead bound, not the speed bound; the lead
   bound limits push force only as far as the arm is joint-PD controlled (a torque/impedance
   controller with other gains, or a gravity-loaded joint, changes the force per metre of lead).
4. Unknown space inside the workspace reads as free, as everywhere today (explicit assumption).
5. Both hands share one max-speed test and one mask: the idle hand entering the declared box while
   slow is exempt too.

**Calibration points:** `workspace_contact_max_speed_mps` (`v_contact`; sim **0.35 m/s** from the
`sim run` contact steps); `workspace_contact_max_lead_m` (sim **0.05 m** from the 3.3–4 cm
free-motion lag of the Isaac stops plus margin; cap 0.10 m); the freshness window
`kWorkspaceCommandFreshPeriods` (2 periods); `WORKSPACE_MIN_COVER` (10 %); the workspace caps
(0.50 m half-extent, 0.25 m³). The real-hardware values of both bounds are a Safety-WG decision.

**Validation owed:** the 8 restock prompts in Isaac deploy sim with ground truth (completion rate
vs `sim run`, zero arm-link stops missed, a deliberate arm/shelf collision still stopped); the
Safety-WG review; then the real-cell posture (the `deploy run` refusal lifts only with the ADR).

## 9. Original plan (for reference)

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
