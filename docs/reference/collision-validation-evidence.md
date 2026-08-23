# Collision-stack validation evidence

Where the numbers in the collision stack's code comments, READMEs and PR bodies
came from, what each validation round actually concluded, and what is still
open.

## Why this page exists

The attached-payload collision work (issue #102, PRs
[#128](https://github.com/OpenRAL/openral/pull/128)–[#135](https://github.com/OpenRAL/openral/pull/135))
is calibrated against measurements taken on a real host, not against
first-principles constants. Those measurements are cited all over the tree —
`spark:~/openral-runs/2026-08-15-final-battery/`, "round-8 r2", "the 2/5
battery" — but until now the ledger they refer to lived only in PR
descriptions and in a run store on one machine. That is a reproducibility gap
(CLAUDE.md §1.8): a reader could see a constant justified by a run they had no
way to look up, and a future maintainer could not tell which conclusions were
still current.

This page is the in-tree ledger. It records **what each round measured and
concluded**, with the artifact path, and it is deliberate about the difference
between a number a file states and an inference drawn from several.

Every round **from 2026-08-22 onward** is produced by the in-tree harness —
[`just validation-matrix`](../contributing/validation-matrix.md) — which writes
a machine-readable `verdicts.json` (`openral_core.ValidationRoundVerdicts`)
alongside its notes. Entries below that point should cite that file rather than
prose, and `just validation-matrix-diff` answers "what changed since the last
round" without anyone re-reading a log. The rounds already listed here predate
the harness and are recorded as they were found — `just
validation-matrix-import` makes one of them queryable in place, without moving
or renaming anything in the run store. The 2026-08-22 and 2026-08-16 `master-1`
rounds' artifacts are also checked in, trimmed, as the harness's own test
fixtures (`tests/unit/fixtures/validation_matrix/`), alongside the harness's
first (failed) live round.

## Reading the citations

- **Artifact paths** are of the form `spark:/home/allopart/openral-runs/<round>/`.
  `spark` is the project's DGX Spark (GB10) validation host; the run store is
  not public and not in this repo. In-tree citations abbreviate it as
  `spark:~/openral-runs/…` — same location.
- **Stack tips** are commit SHAs in this repo. Some are *branch* tips that were
  replayed on merge, so they are reachable by SHA but are not ancestors of
  `master`; where that is true it is said explicitly, with the merged
  counterpart.
- Where a round produced no written conclusion, this page says **outcome
  unrecorded** rather than reconstructing one.

## The rounds

### 2026-08-13 — `matrix-baseline`, `postfix-matrix`

`spark:/home/allopart/openral-runs/2026-08-13-matrix-baseline/`,
`…/2026-08-13-postfix-matrix/`

**No written summary exists for either.** What survives in-tree is the single
finding they are cited for: the baguette run E-stopped at a reported
`-15.70 mm` while MuJoCo's own contact list carried six real contacts of
`-0.87..-1.37 mm`. That gap is the discretisation inflation the support-contact
witness's geometry accounts for exactly — a ~1 mm physical contact reads as up
to ~15 mm of *cube* penetration at 25 mm resolution — and it is why the
witness measures depth against the attested plane rather than the cell cube
(`cpp/openral_safety_kernel/README.md`; hazard log Entry 012).

### 2026-08-14 — `adjudication`, `acceptance`, `final`, `round5`, `round6`, `round7`

**No written summaries.** Two of these rounds are load-bearing and are cited by
their per-run artifact paths:

**Round 5** (`…/2026-08-14-round5/baguette/seed1_run2`) stopped the flagship
counter→cabinet insertion at `-1.78 mm` with the payload 66–67 mm *inside* the
target cabinet. Nothing distinguished "arrived at its declared destination"
from "grazed a wall" — the failure ADR-0097's place-phase witness answers.
The same round supplies the attach-window timeline: the attach sweep ran at
`1786710428.38` and the E-stop at `1786710431.16`, **2.78 s — ~28 published
grids at the bridge's 10 Hz — later**, with the payload still on its stale
silhouette, and the field cell at `22.13 mm`. That timeline is replayed as
`PayloadClearing.TheAttachWindowOutlivesTheFieldRunsTwentyEightGrids`.

**Round 6** (`…/2026-08-14-round6/baguette/seed1_carry_*`) showed the place
witness cannot arm for an enclosed target even with a correct declaration: the
E-stop fired at `horizon_step 0` on `attached:sim:obj_main` vs `voxel_178099`
at `min_distance = -4.9 mm`, with the payload still 22–30 mm from real shelf
contact. A witness earned by touching can never arm if the payload is stopped
before it can touch — the reason ADR-0097 gained its declaration-scoped
approach allowance.

The same day produced the witness/clearing defect, 2/2:
`support_witness_separated live=0x0 was=0x1` 2.7 s after arming with ground
truth `+0.000 mm` still touching, then the same contact E-stopping unexempted
(`sweep_min == min_distance`) as soon as a support cell returned to the map.
That is what the clearing/exemption partition exists to prevent.

### 2026-08-15 — `round8`

`spark:/home/allopart/openral-runs/2026-08-15-round8/`

**No written summary.** Cited for two things: run r2's co-planar cell at
`+42.9 mm` above the attested plane against a then-`~15–19 mm` envelope (the
excess of `+24.4 mm` ≈ one 25 mm voxel that drove the height calibration), and
the refusal-taxonomy defect — **672–811 identical `reason=bounds` warnings per
run**, none of which described a bound.

### 2026-08-15 — `baguette-battery` (5 runs)

`spark:/home/allopart/openral-runs/2026-08-15-baguette-battery/` —
`BATTERY_SUMMARY.txt` (a table plus per-run JSON; no prose verdict).

**0/5 completions.** The battery's value was diagnostic, not a pass rate:

- It **refuted the suspected "sliding" class outright.** Run 4 moved the
  payload `10.15 mm` during the stop and the lateral patch gate (`138.82 mm`
  recorded, quoted in-tree as 135.7 mm for the round-8 lattice) never came near
  the `84.23 mm` actual offset.
- It **named the real class**: adjacent co-planar structure — a raised edge or
  a neighbouring stack on the same support surface — sitting about one voxel
  above the attested plane while the payload is in genuine, continuing support
  contact.
- **Run 1 was the approach allowance's first in-vivo firing**: `-26.48 mm` of
  predictive-check penetration (`min == sweep`, so nothing was exempted) at a
  ground-truth `-2.43 mm` contact — `1.48 mm` past the then-cap of
  `min(one voxel, 25 mm)`. Since the map-vs-truth error at a placement pose is
  itself about one voxel, a one-voxel allowance is *structurally* marginal, not
  occasionally short.

Both findings became maintainer-approved calibrations on 2026-08-15: the
witness height envelope gains `+ resolution` inside the lateral patch, and the
allowance cap moves from `min(one voxel, 2.5 cm)` to `min(1.5 × voxel, 4 cm)`
(hazard log Entry 012 "Calibration 2026-08-15", Entry 013 HZ-0097-4, ADR-0097's
Second Amendment).

Runs 2 and 3 both tripped on `panda_link1` at `-0.42 mm` / `-2.46 mm` with
`min == sweep` — arm-vs-world, nothing to do with the payload. Run 5 tripped
`panda_link5` at `-23.5 mm` after re-attestation.

### 2026-08-15 — `final-battery` (9 runs) — the #102 acceptance round

`spark:/home/allopart/openral-runs/2026-08-15-final-battery/` —
`BATTERY_SUMMARY.txt`, 282 lines, the most complete record in the store.

Stack tip **`e35ed68`** ("one voxel more headroom, on the cap and on the
envelope", branch `feat/quantization-headroom`), clean 27-package colcon
rebuild, DGX Spark.

| suite | n | result |
|---|---|---|
| baguette (`PickPlaceCounterToCabinet`, place declaration) | 5 | **2/5 completions** — run 2 (step 575) and run 3 (step 595), each `transitions=1`, neither with a collision verdict |
| sink_cup | 3 | 0/3 |
| fridge_drawer (no declaration — the control) | 1 | stop persists **byte-identically** to the pre-calibration baseline: `a=panda_link7 b=voxel_165738 min_distance_m=-0.00479038 sweep_min_distance_m=-0.00479038 place_allowance_active=0` |

The summary records the acceptance criterion as **MET, on the named scene,
twice independently, at stack tip `e35ed68`**, and the fridge_drawer control as
proof the two headrooms did not absorb a real interpenetration.

**Three caveats the battery states about itself, kept here verbatim in
substance:**

1. `sweep_min == min_distance` in **every** one of the 5 baguette runs, and
   `place_allowance_active=0` on every trip — 0 occurrences of
   `place_allowance_active=1` anywhere in any run. So neither the witness nor
   the allowance was the mechanism of the two successes; nothing was exempted.
   The summary says so plainly: *"I cannot quote a `place_allowance_active=1`
   line or a live co-planar witness exemption from THIS battery, because
   neither event occurred in any of the 9 runs."*
2. sink_cup's 0/3 is inside its own historical base rate — across all 8
   recorded sink_cup attempts on this host at every tip, sink_cup completed
   exactly **once**. 0/3 is not a signal.
3. **The acceptance ran before PR #135.** `e35ed68` is the branch tip of what
   merged to `master` as `a031f3b` (PR
   [#133](https://github.com/OpenRAL/openral/pull/133), merge
   `b760a5f`); PR [#134](https://github.com/OpenRAL/openral/pull/134) (merge
   `4934080`) and PR [#135](https://github.com/OpenRAL/openral/pull/135) (merge
   `2edcf67`, the octree→grid rasterization fix) both landed after it. **This
   ordering is derived here from the recorded tips; no file in the run store
   draws the conclusion in those words.** It matters because #135 makes the
   occupancy map strictly denser (below), so the 2/5 was measured against a
   sparser map than `master` publishes today.

### 2026-08-16 — `defect-ab`

`spark:/home/allopart/openral-runs/2026-08-16-defect-ab/` — probe JSONs only,
**no summary. Outcome unrecorded.**

### 2026-08-16 — `postfix-matrix` — the #135 before/after

`spark:/home/allopart/openral-runs/2026-08-16-postfix-matrix/` —
`BATTERY_SUMMARY.txt`, 304 lines.

The matrix re-run at `2edcf67` (#135 merged) against the `e35ed68` baseline.
The recorded verdict on the flagship scene:

> baguette (place decl, N=5) — BEFORE: **2/5** completions (run2, run3
> `sim.task_success=true`). AFTER (`2edcf67`): **0/5** completions, no run
> reached `sim.task_success`.

and the summary's own caveat, verbatim:

> HONEST CAVEAT ON THE RATE. This scene is stated to be behaviourally
> non-deterministic under XR-1; 2/5 vs 0/5 at N=5 is not statistically
> separable on its own. But the direction is mechanistically expected: the
> collision map on this scene is now 4094 → 9038 median occupied cells
> (+121%), strictly additive, so the stack can only stop MORE often, never
> less. bag3 shows the full manipulation still runs end-to-end under the
> denser map without tripping. Recommend a larger-N baguette rerun before
> treating 0/5 as a settled rate.

This is the **"2/5 non-determinism" caveat** other documents refer to. #135's
conservativeness proof is the reason the direction is expected rather than
alarming: the fix may only ever *add* cells to the kernel's grid.

### 2026-08-16 — the paired n=15 A/B, and `master-1`

The recommended larger-N rerun, run as a paired control with **one** variable —
the octree→grid rasterization rule:

| arm | path | tip | n |
|---|---|---|---|
| post-#135 | `spark:/home/allopart/openral-runs/2026-08-16-baguette-n15/` | `2edcf67` | 15 (`n1`…`n15`) |
| pre-#135 control | `spark:/home/allopart/openral-runs/2026-08-16-baguette-pre135-n15/` | `4934080` | 15 (`p1`…`p15`) |
| single re-run | `spark:/home/allopart/openral-runs/2026-08-16-master-1/` | `2edcf67` | 1 |

Identical scene YAML, prompt, seed (1) and deadline (420 s) across both arms;
the recorded `git diff --stat 4934080 2edcf67` touches five files — one doc,
one README, and three in `packages/openral_octomap_bridge` — and **zero Python
files**.

**The A/B's conclusion is unrecorded.** No file in the run store states an
outcome, verdict or comparison for the experiment. The pre-#135 arm carries no
`BATTERY_SUMMARY`, no notes, no `VERDICTS.json` and no per-run adjudication;
the post-#135 arm's `VERDICTS.json` is timestamped seven minutes *before* the
control arm started, so it cannot be — and is not — a comparison document. No
later file mentions both arms. The comparison scripts were copied into both
arms and produced output for neither.

What the raw run logs do record:

- **Both arms: 0/15 completions.** All 30 runs `"success": false`; all 30
  `seed1_task_success.txt` files `success=False transitions=0
  ever_succeeded=False`.
- `place_allowance_active=1 occurrences: 0` in all 30 runs.
- Latch classes, post-#135: 13 × `kind_collision (world)`, 1 × `kind_collision
  (self)` (n4), 1 bare `/openral/estop` (n6). Pre-#135: 11 × world, 3 × self
  (p10–p12), 1 bare `/openral/estop` (p6).
- The post-#135 arm has per-run adjudication in `VERDICTS.json`, bucketed as
  `collision-real` (7 runs), `collision-map-conservatism` (7), `self-collision`
  (1). Two of the map-conservatism runs (n11, n13) record
  `gt_beyond_distmax: true` at `distmax 0.1 m` over 883 / 821 probed pairs —
  no solid world geometry within 100 mm of the link the kernel stopped on.

**So the honest statement is: at n=15 neither arm completed the task, the
per-arm latch classes differ but were never compared in writing, and the
experiment's intended verdict on #135's behavioural effect was never
recorded.** Anyone re-opening this should re-run the comparison scripts against
both arms rather than trusting a remembered result.

`2026-08-16-master-1` is a single baguette run at `2edcf67`: E-stop during
**carry** on the attached payload, `a=attached:sim:obj_main b=voxel_92208
step=0 min_distance_m=-0.00418456 sweep_min_distance_m=-0.0355338` —
divergent, i.e. an exemption *was* active — at tick 264, `success=false`.

### 2026-08-22 — `master-1`, four scenes at `2edcf67`

`spark:/home/allopart/openral-runs/2026-08-22-master-1/` — `NOTES.md`. The most
recent round, and the current state of the stack.

Tip `2edcf67`, verified equal to `origin/master` at run time; full clean
rebuild (27/27 packages); reasoner off, SLAM + Nav2 + octomap + kernel check
on; seed 1 pinned. **No safety knob touched.** Exactly one `safety.collision`
and one `estop_ground_truth_snapshot` per run.

| scene | grasped | E-stop pair | kernel min / sweep (m) | ground-truth verdict |
|---|---|---|---|---|
| baguette | yes | `panda_link5` vs `voxel_170781` (step 0) | −0.0209178 / −0.0209178 | **false positive** |
| sink_cup | yes | `attached:sim:obj_main` vs `voxel_87084` (step −1) | −0.0133754 / −0.0133754 | ~~legitimate~~ → **unadjudicated** (see below) |
| fridge | no | `panda_link7` vs `voxel_169769` (step −1) | −0.0247489 / −0.0247489 | ~~legitimate (scene-init)~~ → **unadjudicated** (see below) |
| utensil | no | `panda_link1` vs `voxel_76001` (step −1) | −0.0172764 / −0.0172764 | **false positive** |

**Correction (2026-08-23): the two "legitimate" verdicts are withdrawn.** Both
rested on a probed pair that should never have been measured. The near-miss
probe excluded non-solid geometry on its *world* side only, so a geom with
neither `contype` nor `conaffinity` — a visual shell, a region marker — could
still be ranked first from the robot or payload side, and any distance to one is
not a penetration:

- `sink_cup` — the −1.759 mm pair is `obj_reg_bbox ↔ island_island_group_top_right_0`.
  `obj_reg_bbox` is the payload's own **region bounding box**, the same class of
  marker rounds 5/6 already excluded on the world side.
- `fridge` — the 0.000 m pair is `robot0_g25_vis ↔ fridge_main_group_freezer_door_main`.
  `robot0_g25_vis` is a **visual** geom on `robot0_link6`; that link's collision
  geom is 16.1 mm clear, and the nearest solid pair anywhere on the arm is
  `robot0_link7_collision` at 2.5 mm.

The producer now filters every probe side and discloses the counts
(`openral_hal.sim_sensor_bridge._nearest_pair_records`), and the harness
promotes a `<= 0 m` pair to `real-contact` only on a snapshot that attests it.
These artifacts carry no such attestation, so both stops re-derive as
`estop-collision-unadjudicated` — the recorded evidence does not support
`real-contact` and never did. Neither becomes a *false positive*: nothing here
shows the kernel was wrong, only that this probe cannot say it was right. The
round's artifacts are checked in at
`tests/unit/fixtures/validation_matrix/2026-08-22-master-1/` and the correction
is pinned by `tests/unit/test_validation_matrix.py`.

`min_distance_m == sweep_min_distance_m` in all four → **no exemption active
anywhere this round**; `place_allowance_active=1` occurrences: 0.

**`sim.task_success_final = False` on all four scenes** —
`PickPlaceCounterToCabinet` (1285 steps), `PickPlaceCounterToSink` (350),
`PickPlaceFridgeShelfToDrawer` (102), `PickPlaceCounterToDrawer` (130), all
`transitions=0`.

**The carry phase on baguette was clean — and that is the round's one clear
improvement.** Attach at `t=176.32` → `support_witness_armed`
(`max_penetration_m=0.005597`, `patch_radius_m=0.113964`) →
`place_region_armed` (`allowance_m=0.0375`) → `support_witness_separated` →
detach at `t=226.54` (`place_region_dropped reason=detached`). Across 48 carry
snapshots the payload travelled **870.76 mm** with both clearing shells and the
interior at zero throughout and the nearest cell receding to +281 mm — **no
E-stop during transport**. The prior round at the same tip tripped *during*
carry on the payload itself. Witness, clearing and the partition behaved.

**~~Two link-class false positives are open~~ → one is open, and the
hypothesis is not proven.** Both were robot-link-vs-voxel, both adjudicated
against the ground-truth probe — and both against the **21.7 mm voxel
half-diagonal alone**, which is where the correction bites:

- baguette: 431/431 candidate pairs probed, untruncated, **zero pairs within
  100 mm**, against a kernel reading of −20.9 mm. A >120 mm discrepancy.
  **Still open.**
- utensil: nearest true world geometry `robot0_link1` ↔
  `stack_2_left_group_3_door_g2` at **+43.3 mm** against a kernel −17.3 mm — a
  60.5 mm discrepancy. 489 pairs probed, untruncated. **Withdrawn** — see
  below.

**Correction (2026-08-23): the quantization term alone was the wrong basis.**
Kernel distances are OBB↔voxel and probe distances are mesh↔mesh, so the
admissible gap is `corner_slop(link) + voxel_half_diagonal` — and the slop is
the *larger* term on every panda link (45–88 mm against 21.7 mm). Judging these
two stops against the voxel term alone compares a discrepancy to a strict lower
bound on the gap it should be compared to, which can only over-convict.

The utensil stop settles it, because the **same stop ran again**. On
2026-08-23 the kernel reported `panda_link1` vs `voxel_76001`, reactive step
−1, `min_distance_m` −0.0172764 and a 60.5 mm discrepancy — identical to seven
significant figures. That round's snapshot publishes
`adjudication_budget.admissible_gap_m` = **88.2 mm**, and against it the stop is
`within-quantization`: **the kernel behaving correctly.** Per-link it is tighter
still and reaches the same answer — `panda_link1`'s measured `corner_slop_m` is
**53.4 mm**, so its own gap is 53.4 + 21.7 = **75.1 mm**, and 60.5 mm is inside
it. The two rounds ran the same `collision_geometry` (the `robot.yaml` diff
between `2edcf67` and `d826643` touches only the `sensors:` block), so this is
the same stop and the same collision model, not two similar ones.

**Why it cannot simply be re-derived here.** `adjudication_budget` landed in
**#144 (`ea1b7e8`, 2026-08-22)**, *after* this round; the checked-in fixture
carries no such key, and no budget can be recovered from artifacts that never
recorded one. So the 08-22 utensil verdict is **withdrawn to `unadjudicated`,
not reversed to `within-quantization`**: the later rerun is what shows the
kernel was probably right, and this round's own evidence shows nothing either
way.

**baguette survives the same reconciliation, and is the stronger finding for
it.** Its 120.9 mm discrepancy clears 88.2 mm by 32.7 mm, and `panda_link5`'s
per-link gap (45.3 + 21.7 = 67.0 mm) by more. Note what carries it: an
*untruncated* probe that returned **no pair at all** proves the nearest solid
geometry is beyond `distmax_m`, so 120.9 mm is itself a lower bound. It is the
one link-class stop in the corpus that survives every basis available.

That comparison is a **cross-round inference** — the 88.2 mm comes from a later
round of the same robot, not from these artifacts — so the harness deliberately
does not make it: `tools/validation_matrix.py` re-derives this scene as
`estop-collision-unadjudicated`, because a budget belongs to the round that
measured it. The open question is recorded here, by hand, with its provenance
attached.

The leading hypothesis, recorded as a hypothesis: the utensil evidence voxel
sits at `base_link [0.0375, 0.0875, 0.1625]`, 195 mm from the link-1 origin at
mobile-base height, so the **robot's own base / manipulator mount** may be
entering the octomap as world occupancy. The ground-truth probe excludes
exactly those bodies (`mobilebase0_*`, `robot0_base`, `robot0_link0`,
`manipulator_mount`), so such self-occupancy would be invisible to it. The arm
links themselves *are* correctly carved out — every `panda_link*` origin sits
in a free cell in the baguette snapshot. **Not proven; under investigation.**

**The 2026-08-23 correction weakens this hypothesis at its source.** It was
raised to explain the *utensil* stop, and that stop is no longer evidence of
anything wrong — against a real budget it is the kernel being correctly
conservative. The observation about where the evidence voxel sits still stands
as an observation, but it now has no anomaly to explain on that scene. It
survives, if at all, on baguette alone, whose evidence voxel was never
characterised this way. Treat it as an open question about baguette, not as a
two-scene pattern.

#### 2026-08-23 — self-occupancy settled: **no**, on all four stops

The hypothesis is now **closed, refuted**, by measurement rather than by
argument. Each of the four stops above was reconstructed on a live RoboCasa
model and re-run through the shipped stop record
(`openral_hal.sim_sensor_bridge.estop_ground_truth_snapshot`, whose
`evidence_voxel_backing` is `voxel_backing_record`). Reconstruction method: the
scene YAML the round itself ran, seed 1, with the robot driven to the
`robot_joint_state` that round's `sim.estop_ground_truth_snapshot` published,
then **verified** against the `base_frame_tf` in the same record.

| stop | kernel | evidence-voxel verdict | what backs the cell |
|---|---|---|---|
| utensil | `panda_link1` vs `voxel_76001`, −17.3 mm | **`unbacked`** (0 of 243 rays) | nothing — nearest solid is `stack_2_left_group_3_door_g2` at 43.3 mm, reproducing the round's own probe to 6 dp |
| fridge | `panda_link7` vs `voxel_169769`, −24.7 mm | `self_occupancy_suspect` → **`robot0_link7`, the stopping link itself** | `robot0_link7_collision` is −1.9 mm *inside* `fridge_main_group_freezer_door_main`, with 2 realized MuJoCo contacts |
| baguette | `panda_link5` vs `voxel_170781`, −20.9 mm | **`unbacked`** (0 of 243 rays) | nothing — nearest solid to `link5` is `cab_1_left_group_left_door_g2` at 106.5 mm |
| sink_cup | `attached:sim:obj_main` vs `voxel_87084`, −13.4 mm | **`noncollidable_world`** | `island_island_group_top_right_visual`, a **visual-only** counter top |

**No chassis or mount body backs any of the four cells, nor any of their 26
neighbours.** The only robot bodies that appear are arm links at the stop
instant — which is what a *true positive* looks like, not what self-occupancy
looks like.

**The mechanism cannot produce it either**, and that is now measured on the live
kitchen rather than on a synthetic MJCF:

* `/openral/world_voxels` has exactly one producer — `octomap_server`'s
  `cloud_in`, remapped to the HAL's depth cloud (`sim_e2e.launch.py`). No other
  path writes occupancy.
* `depth_cloud.robot_self_body_ids` resolves **21** bodies on the live
  `PickPlaceCounterToDrawer` model, and that set contains every body the
  hypothesis named: `mobilebase0_base`, `mobilebase0_support`,
  `mobilebase0_fixed_support`, `mobilebase0_wheeled_base`, `manipulator_mount`,
  `robot0_base`, `robot0_link0`. It equals the 14 bodies the stop records report
  as `probe_excluded_robot_bodies` plus the 7 kernel-checked links.
* Casting the real `front_depth` frame at the utensil stop: **13 288 of 65 536**
  rays land on the robot's own bodies unfiltered, and **0** survive
  `depth_camera._transparent_body_geoms`. The chassis does not even appear in
  the *unfiltered* returns — the base-mounted camera does not see it (and
  `mobilebase0_base` is additionally the `mj_ray` `bodyexclude`).

So the earlier refutation (pinned by
`tests/unit/test_sim_estop_voxel_backing.py::test_self_filter_covers_base_and_mount_including_unprefixed`)
**still holds**, on the live model and on the actual stops.

**Read `self_occupancy_suspect` carefully.** The fridge result is the
classification doing what it says and still not meaning what the name suggests:
`voxel_backing_record` reports *a robot body is in this cell now*, which is
equally the signature of a correct stop on a link that has reached real
geometry. It cannot distinguish "the robot wrote this cell" from "the robot has
since driven into it". Treat a `self_occupancy_suspect` verdict as a prompt to
check the near-miss pairs, never as a finding on its own.

**What the baguette 120.9 mm actually is: an unbacked cell, and still
unexplained.** It is not self-occupancy and it is not a payload. Measured
against a probe window wide enough to terminate, the nearest solid world
geometry to `panda_link5` is **106.5 mm** (`cab_1_left_group_left_door_g2`,
confirmed analytically vertex-by-vertex against the door's OBB), so the
discrepancy is **127.4 mm** against an `admissible_gap_m` of 109.9 mm
(88.2 mm max corner slop + 21.7 mm voxel half-diagonal) — 17.5 mm beyond the
budget, and far beyond `panda_link5`'s own per-link gap of 65.4 mm. The cell has
no geometry in it at all, and its 26 neighbours contain solid
`cab_1_left_group_left_door_main` one cell away, so the shape of it is a
**stale or one-cell-displaced map cell**, not a body in the map. Naming the
mechanism that leaves it there is the open question; "the robot's own base" is
not the answer.

**Two caveats on the reconstruction, stated rather than buried.**

1. `panda_mobile`'s manifest declares `base_x` / `base_y` / `base_yaw` and the
   seven arm joints, but RoboCasa's OmronMobileBase also carries
   `mobilebase0_joint_torso_height`, a prismatic arm-mount lift that **no
   manifest joint covers**. At the baguette stop it stood at **+334 mm**. A
   published `robot_joint_state` therefore does not determine the robot's pose;
   only the `base_frame_tf` beside it does. Reconstructions here set the torso
   from that TF's z and then verify the whole pose against it — exact to
   <1 µm on utensil / fridge / sink_cup, and 0.86 mm on baguette, whose stop
   was taken mid-motion.
2. The *world* is at its reset state (t = 0), while the stops are at sim
   t = 4.85 / 6.2 / 64.1 s. Robot-side conclusions are unaffected — the robot's
   pose is set explicitly. World-side distances are t = 0 values; on baguette
   they agree with the round's own finding (nothing within 100 mm), which is the
   cross-check that makes them usable.

**A latent hazard found while measuring, not a live defect.**
`mj_geomDistance` returned exactly `0.000000` for two mesh↔box pairs that are
analytically **361 mm** and **239 mm** apart — but only at a `distmax` of
0.6 m and 0.3 m respectively. At every window ≤ 0.2 m both saturate correctly,
and genuinely-close pairs are exact at every window. `estop_ground_truth_snapshot`
widens its probe window to `admissible_gap_m` (109.9 mm on `panda_mobile`), which
is well clear — but the widening is unbounded by construction, and the harness
promotes a `<= 0 m` pair to `real-contact`. A budget that ever grows past ~0.3 m
would let the probe manufacture the verdict.

**The place machinery was unarmed in the shipped scenes.** `place_declaration`
appears in **zero** files under `scenes/` in this repo. The 08-22 round armed
it only through ad-hoc scene copies in the round's own `scripts/` directory
(`robocasa_baguette_place_seed1.yaml`, `robocasa_sink_cup_place_seed1.yaml`);
the two `_direct_` scenes carry no declaration at all, and their kernel
evidence contains no `place_region` or `support_witness` events. So a user
running a shipped `scenes/deploy/robocasa_*.yaml` today gets **no** place
declaration, **no** place witness and **no** approach allowance. A fix is in
flight; until it lands, the ADR-0097 place path is exercised only by
hand-authored scenes.

**Update (#142 + the follow-up correction).** The three RoboCasa place scenes
now carry a declaration, and the live seed-1 introspection that validated them
is worth recording because two of the three shipped names were wrong:

- `robocasa_baguette` → `sim:cab_1_left_group_main` **confirmed**. The name
  resolves, the evidence producer measures the declared subtree at half-extents
  0.5001 / 0.439962 / 0.3001 m, and the region is published with the 0.0375 m
  approach allowance. This is the first shipped scene on which the ADR-0097
  place path arms end to end.
- `robocasa_sink_cup` → shipped `sim:sink_main_group_1_main`, **refused**;
  the live body table names the sink `sink_island_group_main` (body 319,
  parent `world`).
- `robocasa_fridge_drawer` → shipped `sim:fridge_right_group_main`,
  **refused**; the live fridge is `fridge_main_group`, and the correct target
  is the drawer itself (`fridge_main_group_fridge_drawer0`, body 98) rather
  than the fixture root (body 91), whose subtree would also exempt contact
  with the fridge and freezer doors.

Both wrong names failed exactly as designed — `Place declaration target ...
names no MuJoCo body; refused`, nothing armed, run unchanged — which is what
made the defect a bookkeeping cost rather than a collision. The names were
guessed from the layout corpus because #142's stated procedure, "read it off
`env.fixture_refs[...].name`", is unexecutable: in a live env those values are
tuples with no `.name`, carrying `fixture_name: None`. The scenes now carry the
MuJoCo body-table recipe instead.

The same run also showed the 120 s backstop was mis-sized: on the baguette
scene the declaration armed at goal start and the payload was grasped 92 s of
simulator time later, leaving ~28 s for the whole place phase. The declaring
scenes now use 300 s.

Note also that XR-1 is stochastic across runs even at a pinned scene seed
(`first_chunk_s` 90.96 vs 34.23, 1285 vs 632 steps for the same scene and tip),
so per-run trajectories are not comparable between rounds — **only failure
classes are.**

### 2026-08-22 — `harness-1` / `harness-2`, the harness's first live use

`spark:~/workspace/openral-matrix-baseline/outputs/validation-matrix/2026-08-22-harness-1/`
and `…/2026-08-22-harness-2/` — the first two rounds driven by
[`just validation-matrix`](../contributing/validation-matrix.md) rather than by
the run store's shell scripts. Both carry `metadata.json` + `verdicts.json`.

**`harness-1` is a harness failure, not a stack result.** All four scenes died
in under a second: the harness pinned `--no-enable-reasoner`, an option
`openral deploy sim` does not have, so click exited 2 before the ROS graph
started. It was *reported* as `deadline-no-grasp` with exit 0 — the defect the
`harness-error` bucket exists to prevent, since click's usage error is itself
log lines. Both defects are fixed in PR #145, and this round's artifacts are
checked in as the fixture that pins the fix
(`tests/unit/fixtures/validation_matrix/2026-08-22-harness-1/`). **Cite it as
evidence about the harness, never about the collision stack.**

**`harness-2` ran the matrix and verdicted all four scenes.** Read its
`PROVENANCE.txt` before citing it: its `executed_sha`
(`0a76463e524aaff5a167f14f7727f76ea92ffa47`) is a **local synthetic tree** — the
harness branch merged with `master` (for #142's scene place declarations) plus a
local fix commit, built in a detached worktree on the Spark and never pushed. It
is a reconstruction recipe, not a checkout.

| scene | outcome | E-stop pair | kernel min / sweep (m) | ground truth |
|---|---|---|---|---|
| baguette | `estop-collision-real` | `attached:sim:obj_main` vs `panda_link2` (step 0) | −0.00463106 / −0.00463106 | real contact |
| sink_cup | `estop-collision-real` | `attached:sim:obj_main` vs `voxel_86956` (step −1) | −0.0028367 / −0.0028367 | real contact |
| fridge | `estop-initial-configuration` | `panda_link7` vs `voxel_169769` (step −1) | −0.0247489 / −0.0247489 | real contact |
| utensil | `estop-initial-configuration` | `panda_link1` vs `voxel_76001` (step −1) | −0.0172765 / −0.0172765 | **unadjudicated** |

`sim.task_success_final = False` on all four; no exemption active anywhere
(`min == sweep` throughout); `place_allowance_active=1` occurrences: 0. The
fridge and utensil stops reproduce the 08-22 `master-1` pairs and depths exactly
and are now *classified* as initial-configuration stops, which #139's
`sim.estop_initial_configuration` line made visible.

The utensil verdict is `unadjudicated` for a mechanical reason worth recording:
the scene stopped at sim t≈4.7 s, before the monitor — which attached five
seconds ahead of dispatch, minutes later — had seen a single `world_voxels`
record, so the round had no grid resolution and therefore no quantization budget
to judge the stop by. The monitor now attaches when the graph launches, and a
round's notes name any scene that stopped before it saw a grid.

The `real contact` calls in the table above are subject to the same 2026-08-23
correction as `master-1`: this build's probe filtered its world side only.

### 2026-08-23 — `master-s1/2/3`, `nav143fix-s1/2/3`, `nav143-s1`

`spark:~/workspace/openral-matrix-baseline/outputs/validation-matrix/2026-08-23-*/`
— a 24-run round (four scenes × six rounds) at `d826643` and `87dcda1`. **Cite
it as evidence about the harness, not about the collision stack**, because its
per-run evidence stream is missing.

**Every one of the 24 `run_monitor.jsonl` files contains exactly two lines**,
`monitor_started` and `monitor_stopped`. The cause is mechanical and confirmed:
PR #145 moved the monitor's spawn ahead of the wait for the action server (to
catch stops at sim t≈4.7 s, which it does), so the monitor created its DDS
participant ~6 ms after the deploy started — and `openral deploy sim` then
unlinked every `/dev/shm/fastrtps_*` this user owns
(`_clean_stale_fastrtps_shm`, reached because `RMW_IMPLEMENTATION` is unset on
that host and Fast-DDS is therefore in force). The monitor lost its shared
memory, did not error, and received nothing for the rest of the scene. A monitor
left running across a scene boundary recorded normally until the instant the
next scene's deploy started, then went permanently silent.

Consequences, all now fixed and pinned by
`tests/unit/fixtures/validation_matrix/2026-08-23-master-s1/`:

- `grid_resolution_m` is `null` in every scene of every round, so the round's
  own `NOTES.md` listed all four scenes under "stopped before the monitor saw a
  voxel grid" — which is **not** what happened. The harness now counts what the
  monitor received (`monitor_records`) and reports a deaf monitor separately
  from an early stop.
- The `real-contact` verdicts in these rounds have the same defect as
  `master-1` above: the fridge stop was adjudicated off
  `robot0_g42_vis ↔ fridge_main_group_g43` at 0.000 m, a visual shell, while
  `robot0_link7_collision` was 2.5 mm clear.
- The `utensil` stop (`robot0_link1` 43.3 mm clear against a kernel −17.3 mm,
  a 60.5 mm discrepancy) was adjudicated against the 21.7 mm quantization term
  while the **same snapshot** reports `adjudication_budget.admissible_gap_m =
  88.2 mm`. Against the HAL's own budget it is `within-quantization` — the
  kernel behaving correctly — not a false positive. The harness now consumes the
  published budget. **This is byte-for-byte the same stop the 2026-08-22
  `master-1` round called a false positive** (same link, same voxel, same
  reactive step, `min_distance_m` −0.0172764 in both), which is what withdraws
  that verdict — see the correction in that round's section above.

**`nav143-s1` never ran at all.** At `87dcda1` `ros2 launch` threw —
`executable 'payload_footprint_node.py' not found on the libexec directory
.../openral_nav2_bringup` — and unwound, but the nodes it had already spawned
kept running and kept logging, so there was no marker file, no usage banner and
a 476-line deploy log. The dispatcher then raised
`RuntimeError: /openral/execute_rskill unavailable`, and the scene was bucketed
`deadline-no-grasp` with `harness_error_reason` and `dispatch_failure_reason`
both empty. Both are now `harness-error`; its artifacts are the fixture at
`tests/unit/fixtures/validation_matrix/2026-08-23-nav143-s1/`.

**The round produced no citable collision-stack finding.** Rerun it on a build
carrying the fixes before drawing any conclusion from these six rounds.

## Standing caveats

Seven things a reader should carry away, all of them stated by the artifacts
themselves rather than inferred:

1. **The #102 acceptance is real but narrow, and it predates `master`.** Two
   independent completions on one named scene, at branch tip `e35ed68`, with
   **nothing exempted in either** — before #134 and #135 landed. It has not
   been reproduced at `2edcf67`.
2. **2/5 vs 0/5 at n=5 is not statistically separable** on a scene the run
   store itself calls behaviourally non-deterministic under XR-1.
3. **The n=15 A/B that was supposed to settle that never had its outcome
   written down.** Both arms are 0/15; the comparison was not made.
4. **As of 2026-08-22 the stack completes no scene.** Carry-phase machinery
   behaves; **one** link-class false positive is open (baguette; the utensil one
   is withdrawn — see the correction in that round's section). The place path was unarmed
   in every shipped scene at the time of this round; #142 and its follow-up
   correction arm it on the three RoboCasa place scenes, with `robocasa_baguette`
   the only one observed arming end to end so far (see the update above).
5. **No `real-contact` verdict recorded before 2026-08-23 is safe to cite.**
   Every ground-truth probe up to that date filtered non-solid geometry on its
   world side only, so a visual shell or a region marker on the robot or payload
   side could be — and was — ranked first at 0.000 m. Those stops re-derive as
   `unadjudicated`. This withdraws a conclusion; it does not reverse one: no
   such stop is thereby shown to be a false positive.
6. **No `false-positive` verdict recorded before #144 is safe to cite either,
   and it cannot be re-derived from its own round.** `adjudication_budget`
   landed in #144 (`ea1b7e8`, 2026-08-22); every round before it was judged
   against the voxel half-diagonal alone (21.7 mm at a 25 mm grid). That term is
   a strict *lower bound* on the admissible kernel-vs-probe gap — the real gap
   adds `corner_slop(link)`, which is the larger term on every panda link — so
   exceeding it establishes nothing, and those rounds recorded no slop from
   which the real gap could be reconstructed. This is a **second, independent**
   reason older link-class verdicts are not citable, and it bites in the
   opposite direction from caveat 5: that one withdraws stops called *real*,
   this one withdraws stops called *false*. Asymmetry worth keeping straight:
   a discrepancy **within** the voxel term is still sound (within a lower bound
   implies within the true gap), so `within-quantization` from an old round
   stands.
7. **Several rounds have no written summary at all** (all of 08-13 and 08-14,
   `round7`, `round8`, `defect-ab`). Their findings survive only as constants
   and comments in this repo. Treat a citation to one of those rounds as a
   citation to the in-tree comment that carries it, not to a retrievable
   verdict.

## Related

- [The validation matrix](../contributing/validation-matrix.md) — how to run a
  round, what each verdict means, where the artifacts land, and the Spark
  etiquette. The harness that feeds this page from 2026-08-22 onward.
- `cpp/openral_safety_kernel/README.md` — the witness, the partition, the
  allowance, and the geometry each calibration changed.
- `packages/openral_octomap_bridge/README.md` — payload clearing, the attach
  sweep window, and the rasterization change #135 made.
- [Design decisions](../decisions.md) — how ADR numbers are cited here and
  where the records live.
