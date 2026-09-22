# Collision-safety alternatives survey — is the hand-rolled kernel the right shape?

> **Status: research survey, analysis only (2026-08-30).** Nothing here landed as written.
> **Read with the correction:** all three candidate paths shipped within a week and measurement
> overturned four of this page's claims — `collision-validation-evidence.md` §6, *The survey,
> reviewed against what happened*. Companions: the
> [collision-primitive study](collision-primitive-study.md), the
> [validation evidence ledger](collision-validation-evidence.md),
> [world-map fidelity](world-map-fidelity.md).
>
> **Conventions.** "docs claim" = documentation or abstract; "source shows" = the file was read;
> anything unverifiable against a primary source is **unverified** and supports no verdict.
> §§13–23 are a second pass over methods outside §§3–7's shortlist.

## 1. The problem being shopped for

A C++ kernel vets every VLA action chunk: self-collision via per-link OBBs with an SRDF-derived
16-pair allowed-collision list, world collision via an octomap-derived 25 mm voxel grid, staged
26-DOP → convex-hull-GJK narrow phase for links declaring `tight_geometry`
(`robots/panda_mobile/robot.yaml`; `cpp/openral_safety_kernel/README.md`).

- Run 1 stopped on `panda_link2` vs `panda_link5` at −5.34 mm while adjudication at the
  *recorded* joints shows them **+53 mm apart** — the verdict was for a predicted horizon
  configuration the artifact does not store
  ([#172](https://github.com/OpenRAL/openral/issues/172)). Run 2 stopped on `panda_link7` vs a
  voxel at −2.96 mm resolving 2.2 m away, outside the grid's r = 1.05 m coverage.
- **Of 15 stops in the five-round battery, 11 classified `within-quantization`** — ~1 mm real
  contacts reading as up to 15 mm of penetration; nine are a link against world occupancy at
  −0.29…−11.34 mm.
- **OBB corner slop is 27–76 mm of protrusion per link** (45–88 mm as corner slop) on top of the
  21.7 mm voxel half-diagonal ([primitive study §4.2](collision-primitive-study.md)).

The kernel lacks three things established stacks have: **distance-based speed scaling** (the only
verdict is accept / drop / latch at a fixed margin); **mesh-level checking** (the world side is
voxels vs primitives); and **touch-links / attached-object semantics as a first-class contract** —
the gripper is absent from `collision_geometry` and intended-contact scoping was built bespoke
(ADR-0092 witness, ADR-0097 declaration).

## 2. What the last two weeks built — PR ledger and verdicts

54 merged PRs (2026-08-11 → 08-29), cross-checked against the
[validation evidence ledger](collision-validation-evidence.md)'s 8 standing caveats (3 of them
withdrawals).

### 2.1 PR ledger

| PR | theme | what it established, or what it cost |
|---|---|---|
| #101 / #103 / #117 | frame/FK | Three PRs and one ADR for one constant: #101 put every kernel link 0.7 m too low (joint 1 1.033 → 0.333 m); #103 half-fixed it and replaced 7 hand capsules with mesh-derived OBBs; #117 fixed it properly, bit-identical outcomes |
| #111 | world map | Per-pixel `mj_ray` instead of `mj_multiRay` (whose BVH cull skips visual-only geoms): 947 mm median error fixed, at ~1.9× cost — 6.0 / 18.6 / 55.6 ms per frame at 4 / 301 / 1201 geoms |
| #119 | observability | ADR-0096 latched `SafetyStatus`, implemented twice (C++ kernel and `SafetyPassthroughNode`) — a permanent double-maintenance tax |
| #130 | evidence | Ground-truth E-stop snapshot + `mj_geomDistance` probes — built the adjudication corpus on an instrument #170 later proved unusable (caveat 8) |
| #131 | ADR-0092 | `SupportContactWitness`, payload occupancy clearing, `withheld ⊆ exempt` partition; 53,872-probe invariant sweep; 12.5 mm discrimination floor accepted |
| #132 / #133 | ADR-0097 | Place-phase witness + approach allowance; the corpus's only two full task completions — caveat 1 says that acceptance was never reproduced |
| #135 | world map | Octree→grid by cube overlap instead of centre sampling: 7.9–37× cheaper, but bought +½ cell of forward reach — 48 % of live stops (#171), undone by #178 |
| #138 | evidence | Created the validation-evidence ledger after four documented claims proved wrong — the single most useful artifact of the effort |
| #139 / #154 | validation | `Kitchen._load_model` redraws layout on every reset, so `seed: 1` does not identify a kitchen; ~85 % of this task's kitchens have the defect. #154 falsified #139; #171 falsified #154 |
| #142 / #146 | ADR-0097 | The ADR-0097 machinery had been inert since it merged (`place_allowance_active` logged zero times); 2 of 3 declared targets were guesses and both wrong |
| #143 / #186 | Nav2 | The payload footprint publisher was built, never CI-verified, never run against a translating base, deleted six days later — the payload rides 0.28 m above the costmap slice, so payload avoidance becomes a 3-D E-stop |
| #144 | evidence | `evidence_voxel_backing` + `adjudication_budget` (corner slop 28.3–88.2 mm). Caveat 6: **every `false-positive` verdict before this PR is withdrawn** |
| #145 / #149 | harness | The versioned validation matrix; its first live round died in <1 s on a nonexistent flag and reported exit 0, its second produced 24 runs with empty monitor logs. Caveat 5: **every `real-contact` verdict before 08-23 is withdrawn** |
| #153 | ACM/self | Refuse a disconnected collision graph; kernel FK now matches each robot's normative source to 1e-16 m; 4 manifests corrected |
| #157 / #158 | link geometry | `base_link` has no collision geometry, so a chassis primitive recovers **no** characterised stop; #157's swept-box row falls below a proven Lipschitz lower bound on 6 of 7 links |
| #159 / #160 | evidence | 240-start-state census: `link1`+`link2` = 83 % of stops, `UNEXPLAINED` = 0/72. Self-occupancy settled by measurement (**no**): 13,288/65,536 rays hit the robot unfiltered, **0** survive the self-filter |
| #161 | link geometry | **Perfect link geometry recovers 27/72 worst case**, crossover ~10 mm. Corrects #159, #157 and its own prior 15-axis SAT bound (0.16 mm, not 3–6 mm) |
| #165 | schemas | `CollisionShape` as a real discriminated union — an untagged mapping had silently become a sphere |
| #166 | link geometry | Staged 26-DOP → exact hull: OBB over-reported link1/link2 reach by 53.3 / 46.8 mm, hull → 0.00 mm excess, 1.04–1.11× *faster*. Recovers 26 % of stops |
| #169 | ACM/self | Branch-and-bound proof: `link5`↔`link7` genuinely interpenetrates in 914/14641 poses — the committed ACM was **exempting a real check**. No shipped ACM byte changed |
| #170 | instruments | Certified `convex_geom_distance` replaces `mj_geomDistance` on the evidence path. Caveat 8: **no verdict from any probe before 08-25 is citable**; four `0.000 m` readings re-measure at +14.8 … +107.9 mm |
| #171 | world map | Re-pinned against the *kernel* criterion, 61 live captures. **Split: 26 % link-side, 74 % world-side.** The census's "live map ⊆ ideal grid" assumption is false |
| #177 | evidence | Payload pose (`xquat`) in the E-stop record — it could not replay the geometry it adjudicates (14 of 15 battery stops) |
| #178 | world map | `/openral/world_voxels` on the OctoMap's own lattice; removes #135's 29–35 mm median dilation, closes a 124 mm coverage hole and a fail-*open* empty-grid path. Safety-WG reviewer, hazard entry and sign-off all unchecked |
| #179 | graded outcome | `CollisionHit::advisory`: a declared place's own contact refuses the chunk without latching, under five independent bounds. **Fired 0 times in 20 scene runs** |
| #180 | world map | Filter geoms with neither `contype` nor `conaffinity` out of the depth cast; per-ray monotonicity verified over 16,384 rays × 4 scenes. Inverts two of #111's assertions; sim-only term |
| #185 | Nav2 | **`consider_footprint` measured +0.53 ms, not #143's +8.1 ms** — a 15× discrepancy. The whole MPPI loop fits at ~20 % of a 50 ms cycle |

### 2.2 Theme verdicts

**Frame and FK (#101, #103, #110, #117/#129).** Net effect correct. **Done, stop.**

**Exact geometry and instruments (#153, #157–#161, #165, #166, #169, #170).** The instrument half
found two trusted mechanisms wrong in the *unsafe* direction (an ACM exempting a pair that
interpenetrates 48 mm; a probe reporting 0.000 m where truth is +107.9 mm); the link-envelope half is
a dead end. **Instruments: keep. Link envelopes: done, stop.** *(Overtaken: true of start-state
stops, wrong for carry-phase stops — `tight_geometry` on `link3`/`link4`/`link6` has since landed and
been measured, `collision-validation-evidence.md` §6, lever 1.)*

**Voxel grid / octomap fidelity (#111, #135, #150, #151, #178, #180).** #135 fixed a half-voxel phase
error by adding a half-voxel of dilation, #171 measured that dilation as 48 % of live stops, #178
removed it. **Keep going.**

**Place-phase allowance, ADR-0097 (#132, #133, #142, #146, #179).** Armed on three of four scenes,
**binding on 2 of 15 stops**, advisory band fired zero times in 20 runs; §9 point 1 names the real
fix (Path B). **Stop extending the margin-reduction mechanism.**

**Evidence and adjudication pipeline (#114, #130, #138, #144, #147, #159, #160, #171, #175, #177).**
Three of the eight caveats withdraw verdicts this pipeline produced (5, 6, 8); it is now sound.
**Keep, but freeze the contract**, and record the tripping *horizon configuration*.

**Validation harness (#139, #145, #148, #149, #154, #164, #183).** Every failure it fixed was a
*verification* failure, not a collision one. **Done, stop building.**

**ACM / self-collision (#153, #169).** The only theme with no retraction, with one caveat: #169
changed no manifest, so `panda_link5`↔`panda_link7` is **still exempted** in
`robots/panda_mobile/robot.yaml` *(verified: line 751)* despite being proven a real, checkable pair.
**That exemption is the open item this survey leaves on the ACM.** **Done, stop** otherwise.

**Nav2 boundary (#143, #148, #183, #185, #186).** What survives is the lidar self-filter, the
ADR-0040 boundary docs, a `findCircumscribedCost` correction and `consider_footprint: true`. #186 is
strictly less protective: a payload clipping a tall thin obstacle is now an E-stop rather than an
avoidance. **Stop.**

## 3. MoveIt 2

`github.com/moveit/moveit2` `main`, source fetched raw. BSD-3-Clause; Jazzy LTS.

### 3.1 PlanningScene collision checking is mesh-level, FCL by default

Default detector is FCL, Bullet switchable at runtime (`planning_scene.cpp` L200); URDF collision
**meshes load as real triangle meshes** into `fcl::BVHModel<OBBRSS>`, octomaps as `fcl::OcTree`
(`collision_common.cpp` L900–922). The FCL path is **discrete-only** (`collision_env_fcl.cpp`
L340–346), Bullet has CCD but **no distance queries**, and a distance query is a separate, much
slower pass.

### 3.2 Attached objects and touch_links

`moveit_msgs/AttachedCollisionObject` carries `link_name`, the object, and `touch_links` — the
links allowed to keep touching the attached body. Source shows the semantics live in the FCL
narrowphase callback (`collision_common.cpp` L130–183): a ROBOT_LINK vs ROBOT_ATTACHED contact is
**always allowed if the link is in the body's `touch_links`**; two bodies attached to the same
link never collide; the ACM is checked separately and is **not** modified on attach. Attached
bodies are appended to the robot's FCL object set and checked against both world and robot in
every check (`collision_env_fcl.cpp` L212–240). **This is the upstream precedent for scoping an
*intended-contact* pair by naming it, not by exempting a region.**

### 3.3 Octomap self-filtering also masks attached objects

`excludeRobotLinksFromOctree()` (L886), **`excludeAttachedBodiesFromOctree()`** (L958) and
`excludeWorldObjectsFromOctree()` (L988) register every link, attached-body and world-object shape
with the occupancy updater's `ShapeMask` — upstream equivalent of `openral_octomap_bridge`'s
payload clearing, with the same consequence: masking the payload's cells also masks the
support-surface cells it rests on.

### 3.4 MoveIt Servo: proximity-based velocity scaling, not a binary stop

A thread polls padded robot-vs-world and unpadded self-collision with `request.distance = true` at
`collision_check_rate` (default **10 Hz**). From `moveit_servo/src/collision_monitor.cpp`:

```cpp
// velocity_scale = e ^ k * (collision_distance - threshold)
// k = - ln(0.001) / collision_proximity_threshold
scene_collision_scale = std::exp(scene_velocity_scale_coefficient *
    (scene_collision_result_.distance - servo_params_.scene_collision_proximity_threshold));
collision_velocity_scale_ = std::min(scene_collision_scale, self_collision_scale);
```

Scale is 1.0 at the threshold (defaults: self 0.01 m, scene 0.02 m), decays to 0.001 at distance
0, hard 0.0 only on actual collision. Octomap checking is **off by default**, and nothing in the
docs claims Servo is a certified safety function.

### 3.5 Rate and the standalone-filter pattern

The in-tree pattern for "checkCollision on streamed joint states as a filter" is Servo's monitor,
default 10 Hz, not 100. Published throughput (GSoC Bullet benchmark, Panda,
[moveit#1427](https://github.com/ros-planning/moveit/issues/1427#issuecomment-514541239)): boolean
self-check ~9 µs (FCL) / ~3.7 µs (Bullet); 100 world meshes ~29 µs clear but **~1.25 ms (FCL) with
4 meshes in collision**, plus a full extra pass for `distance=true`.

### 3.6 The contact-rich pattern: phase-scoped ACM edits

The canonical MoveIt answer to "the gripper must touch the object" is `touch_links` on attach
(§3.2) plus MTC's `ModifyPlanningScene` stage, which mutates the ACM **per planning-scene snapshot
for one stage** (`modify_planning_scene.cpp` L79–145). **This is ADR-0097's place-declaration
pattern, established upstream** — and ADR-0097, scoping at run time with a timeout and a
measured-contact gate, is the more conservative of the two.

## 4. FCL and coal (ex hpp-fcl)

Both BSD-3-Clause; coal (the 2024 rename of hpp-fcl) is Pinocchio 3's engine.

- **FCL's signed distance is broken in exactly the mm regime**: with BVH mesh models
  `min_distance` returns 0 in penetration even with `enable_signed_distance` set (FCL
  [#221](https://github.com/flexible-collision-library/fcl/issues/221),
  [#574](https://github.com/flexible-collision-library/fcl/issues/574),
  [#575](https://github.com/flexible-collision-library/fcl/issues/575)). MoveIt inherits it.
- **coal has its own GJK+EPA**, true signed distance, a `security_margin` and a free
  `distance_lower_bound`, at 0.8–2.5 µs per pair near contact (Montaut et al., RSS 2022,
  [arXiv:2205.09663](https://arxiv.org/abs/2205.09663)).
- **Octrees work directly**, but the answer is distance to occupied *cells*, so world-side accuracy
  is still bounded by the voxel size, exactly as in-tree.

### 4.1 Round-2 verification (2026-08-30): coal is not a free swap

- **MoveIt 2 has not migrated** (`moveit_core` still depends on `libfcl-dev`; searching moveit2
  issues+PRs for "coal" returns zero), though Pinocchio 4.1.0 depends on it.
- **Open accuracy issues in the near-contact regime**, all unfixed:
  [#823](https://github.com/coal-library/coal/issues/823) witness points go wrong once meshes
  penetrate; [#755](https://github.com/coal-library/coal/issues/755) a `security_margin` check
  silently misses; [#636](https://github.com/coal-library/coal/issues/636) NaN contacts. The fixes
  sit under `[Unreleased]` in `CHANGELOG.md@devel`; latest tag is 3.0.4.
- [#857](https://github.com/coal-library/coal/issues/857) measures coal's broadphase ~1.8× *slower*
  than FCL 0.6.1, and [#649](https://github.com/coal-library/coal/issues/649) reports **EPA
  allocations on the hot path**, in direct tension with the no-alloc rule.

**Verdict:** keeping the in-house certified GJK — no EPA, no hot-path allocation — is lower-risk
today. coal stays a watch item until those fixes ship in a tagged release.

## 5. cuRobo + nvblox + Isaac ROS cuMotion

The repo already emits cuRobo collision spheres from the kernel's own lowered geometry
(`packages/openral_safety/openral_safety/cumotion_config.py`).

- **Usable as a batched checker** — `RobotWorld.get_world_self_collision_distance_from_joints` over
  `(batch, horizon, dof)`, docs claim 0.05–0.11 ms/query — **but the distance is clamped** to zero
  beyond `activation_distance`: a margin gate, not clearance for graded scaling.
- **Both geometry sides are cm-scale**: `franka.yml` carries 61 collision spheres, nvblox's ESDF
  defaults to 0.05 m — coarser than the in-tree 25 mm grid. Not a route to mm fidelity.
- **The robot segmenter is directly relevant**: it masks depth pixels within sphere radius +
  `distance_threshold` and clears a grasped object's voxels from the SDF — the GPU equivalent of
  the in-tree depth self-filter + ADR-0092 payload clearing.
- **Licences, one trap**: cuRobo relicensed to **Apache-2.0** with V2 and nvblox is Apache-2.0, but
  **`nvblox_torch`, the cuRobo↔nvblox bridge, is still NVIDIA non-commercial**.

## 6. Safety filters for learned policies — the field

- **No flagship VLA stack ships any collision layer**, verified by cloning and grepping at HEAD
  (2026-08-30): `openpi` zero hits (the DROID example's only action processing is
  `np.clip(action, -1, 1)`), `octo` none, `Isaac-GR00T` advice not code, `lerobot` kinematic bounds
  only, `IsaacLab` zero for "safety filter". **The in-tree kernel is ahead of the field here, not
  behind it.**
- **The academic framing is projection and speed modulation, not binary stops**: ATACOM
  ([arXiv:2404.09080](https://arxiv.org/abs/2404.09080)) projects actions onto a constraint
  manifold's tangent space; PACS ([arXiv:2511.06385](https://arxiv.org/abs/2511.06385)) verifies a
  whole predicted chunk by set-based reachability and brakes along its own path.
- **The intended-contact problem has two published answers.** Geometric: exempt the intended target
  and check everything else — the attention-guided filter
  ([arXiv:2606.09749](https://arxiv.org/abs/2606.09749)) derives the target from the model's
  attention, ADR-0097 from an explicit declaration. Force-based: **ISO/TS 15066** bounds contact
  force rather than forbidding contact, and SARA-shield
  ([arXiv:2412.10180](https://arxiv.org/abs/2412.10180)) applies that to learned manipulation.

### 6.1 Round-2 correction: the layer now exists in the literature (Dec 2025 – Jun 2026)

The first pass said no established collision layer for VLAs exists. True of vendor stacks, **now
false of the literature**: four groups published external safety layers over *frozen* VLAs within
seven months, three with code, converging on the in-tree kernel's shape — **VLSA / AEGIS**
([arXiv:2512.11891](https://arxiv.org/abs/2512.11891), code `THU-RCSCT/vlsa-aegis`), a CBF-QP
reporting **+59.16 % obstacle avoidance *with* +17.25 % task success** on SafeLIBERO; the
**attention-guided safety filter** ([arXiv:2606.09749](https://arxiv.org/abs/2606.09749)), whose
stated motivation (VLM-inferred obstacle identification is *too slow for the control loop*)
independently endorses ADR-0097's explicit declaration; **SafeVLA**
([arXiv:2503.03480](https://arxiv.org/abs/2503.03480)) at +83.58 % safety; and **PACT**
([arXiv:2606.08414](https://arxiv.org/abs/2606.08414)) at −31.0 % violations. Benchmarks now
measure VLA safety violations directly (SafeLIBERO, LIBERO-Safety
[arXiv:2606.23686](https://arxiv.org/abs/2606.23686), ForesightSafety-VLA
[arXiv:2606.27079](https://arxiv.org/abs/2606.27079)), and runtime failure monitors reading VLA
internals (SAFE, [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)) are the natural producer of
the replanning ladder's hand-off signal. **Corrected claim: no vendor ships one and no standard
exists, but the layer is actively populated and the kernel matches the published shape.**

## 7. What the ROS community reports

Searched 2026-08-30; the load-bearing discussion is in GitHub issues and the legacy ROS Answers /
moveit-users archives, not Discourse.

### 7.1 What practitioners hit

- **Octomap doesn't clear cleanly around an object you're about to grasp — a 12-year-old,
  still-open MoveIt defect**: [moveit_ros#315](https://github.com/moveit/moveit_ros/issues/315)
  (2013) → [moveit#3221](https://github.com/moveit/moveit/issues/3221) (2022) →
  [discussion #3613](https://github.com/orgs/moveit/discussions/3613) (Nov 2025), where
  **maintainer rhaschke labels it "a bug with MoveIt's octomap"**. Worse at finer resolution.
- **Depth/octomap self-filtering sees the robot's own arm as an obstacle**:
  [moveit#3210](https://github.com/ros-planning/moveit/issues/3210) (2022, open).
- **Servo's checking is too blunt for contact-rich commands**:
  [moveit2#409](https://github.com/ros-planning/moveit2/issues/409) — both modes slow the robot
  even while it is *retreating*. Assigned, never implemented.
- **Padding tuning is a known pain point, not a precision fix**:
  [moveit_ros#342](https://github.com/moveit/moveit_ros/issues/342) (2013) —
  `padding_offset`/`padding_scale` "are very difficult/impossible to tune".
- **nvblox/cuRobo has the same self-vs-world confusion**:
  [curobo#148](https://github.com/NVlabs/curobo/discussions/148) — the depth masking NVIDIA
  recommends also deletes real nearby obstacle voxels (Sept 2025 comment).

### 7.2 What practitioners do

**Standard:** `touch_links` / dynamic ACM entries for the grasp target, *not* tighter geometry;
attach the object to the end-effector link as the grasp closes. **Reported working, with caveats:**
Franka's `setCollisionBehavior`, which distinguishes "contact" (F/T inside a band, robot keeps
moving) from "collision" (reflex trip) but is vendor-specific; and admittance control (PickNik's
`ros2_control` controller; CRISP extends it to learned policies, safety posture unverified). **One
person's hack:** raising padding to compensate for quantization.

### 7.3 Directly actionable for this repo

1. **Use `touch_links`/dynamic ACM entries for the specific link-object pair expected to contact
   during a grasp/place phase, instead of shrinking quantization error** — gated by skill manifest
   / reasoner state so it stays an explicit, logged exception.
2. **Don't chase octomap-clearing fidelity as the fix** — maintainer-confirmed unresolved as of
   Nov 2025.
3. **Adopt a force/torque threshold as the arbiter for "is this a real hazardous collision" during
   contact phases**, separate from the geometric check.
4. **If a compliant-controller layer is in scope**, PickNik's admittance pattern is the structural
   fix for contact-rich E-stops on quantization artifacts.
5. **Treat padding tuning as a low-value lever, not a fix path.**
6. **If moving to nvblox/cuRobo, budget for robot self-segmentation as a required, imperfect
   component.**

### 7.4 Dead ends the community already tried

Padding/offset tuning against false-positive contact detection; relying on octomap/self-filter to
exclude the robot or grasp target (unresolved across 12 years — the community routes around it with
`touch_links`/ACM); Servo's built-in checking as a contact-aware safety layer; and geometric depth
masking to segment the robot out of nvblox.

### 7.5 Coverage notes

Discourse yielded little beyond the CRISP announcement; Robotics Stack Exchange overlapped the
archived ROS Answers content. **No thread was found about a custom safety kernel vetting
learned-policy actions pre-execution** — this approach is ahead of what community venues discuss
rather than following an established pattern.

## 8. Comparison table

"mm near contact?" asks whether the approach can tell a ~1 mm intended touch from a ~15 mm
penetration against a *depth-sensed* world.

| approach | self-collision fidelity | world-collision source | near-contact behavior | runtime rate | ROS 2 integration effort | GPU | maturity / license | mm near contact? |
|---|---|---|---|---|---|---|---|---|
| **in-tree kernel** (today) | OBB + 26-DOP/hull GJK vs voxels; 16-pair ACM | octomap → 25 mm grid | binary margin → drop/latch; ADR-0097 margin reduction + advisory band | 30–200 Hz, allocation-free | — (it is the ROS 2 integration) | no | 2 weeks in-repo; Apache-2.0 | **no** — voxel + OBB slop 67–110 mm combined (45.3–88.2 mm corner slop + 21.7 mm half-diagonal) |
| **MoveIt 2 PlanningScene** | true URDF mesh vs mesh (FCL BVH) | meshes + primitives + octomap (self- and attached-filtered) | signed distance available; discrete only; distance pass is 10–100× a boolean check | Servo monitor default 10 Hz; boolean checks µs-scale, octomap+distance is the slow path | large — replaces the kernel's world model | no | BSD-3; Jazzy LTS; the reference implementation | **vs known meshes yes; vs octomap no** (voxel-bounded) |
| **MoveIt Servo collision monitor** | via PlanningScene | via PlanningScene | **exponential velocity scale** from distance; 0 only on collision | 10 Hz default, tunable | moderate if only the *formula* is adopted | no | BSD-3; "slowdown assist", not a rated safety function | n/a — modulates, doesn't discriminate |
| **coal (hpp-fcl)** ≥ 3.0 | convex-hull signed distance, µs/pair, SRDF pair exclusion via Pinocchio | primitives, convex meshes, BVH, **octree (distance supported)** | true signed distance, EPA penetration depth, `security_margin` | 0.8–2.5 µs/convex pair → 100 Hz trivial for O(100) pairs | small as a library; one dependency into the kernel | no | BSD-3; LAAS-CNRS/INRIA, active | **convex-pair yes; vs octree no** (voxel-bounded) |
| **cuRobo RobotWorld** | 61 spheres (Franka) + buffers — cm-conservative | primitives, meshes, nvblox ESDF, voxel grid | distance **clamped at activation_distance**; batched full-horizon checks | >10⁴ configs/ms batched | moderate (Python/torch sidecar) | **yes** | Apache-2.0 since V2 (verified) | **no** — spheres + ESDF are cm-scale |
| **nvblox + robot segmenter** | n/a (perception side) | ESDF from depth, 0.05 m default voxels; robot + attached object masked from depth | n/a | real-time (GPU) | moderate (Isaac ROS, Jazzy-supported) | yes | nvblox Apache-2.0; **nvblox_torch bridge non-commercial** | **no** — coarser than the in-tree grid |
| **CBF / ATACOM / PACS filters** | whatever distance function you give them | ditto | **QP projection / chunk-consistent braking** — graded by construction | policy-rate | research code; no ROS 2 turnkey | varies | academic; no rated deployments found | inherits its geometry's resolution |
| **force limiting (ISO/TS 15066, Franka reflexes, SARA-shield)** | n/a | n/a — measures contact, not proximity | **discriminates touch from crush by force/energy**, permits intended contact | control-rate (kHz on Franka) | small on hardware that reports external torque; needs a sim analogue | no | ISO TS + shipping arm firmware; SARA-shield is research | **yes — the only row that is** |

## 9. Assessment

**1. Nothing off-the-shelf clears the voxel wall.** Every stack that checks against a
*depth-derived* world model is bounded by its cell size (MoveIt's octomap path, coal's octree
distance, nvblox's 0.05 m ESDF); the 11-of-15 `within-quantization` stops are what a 25 mm grid *is*
near thin geometry. Established stacks reach mm scale only against **known, modeled geometry**, so
the route is a better *world model*, not a better checker — and ADR-0097's declaration half-does it:
it names the target; it does not yet give the kernel its geometry. *Qualifier:* Tesseract's
implicit-SDF path (§17.3) consumes a distance field lazily instead of resampling it, so the claim
stays true of every *mapper* but not every *checker*.

**2. The three missing capabilities are all established, and two are cheap.** (a) Servo's
exponential scale-from-distance is three lines of math over a `sweep_min_distance` the kernel already
computes. (b) `touch_links` ≈ the finger-pair exclusion, `AttachedCollisionObject` ≈ ADR-0092,
`excludeAttachedBodiesFromOctree` ≈ the octomap bridge's payload clearing, MTC's phase-scoped ACM ≈
ADR-0097 — **the bespoke mechanisms are independently converged re-derivations, mostly stricter**.
(c) Mesh-level checking is available, but the
[primitive study §6](collision-primitive-study.md) showed better *primitives* recover none of the
characterised stops and the ceiling for *perfect* link geometry is 27 of 72 states
([tight geometry §7.3](collision-tight-geometry.md)) — the binding term is the map, so mesh-level
geometry matters mainly for self-collision (the `panda_link5`↔`panda_link7` pair is exempted
"under protest" because boxes false-positive on 79.6 % of the interpenetration band).

**3. The verdict-for-a-predicted-config evidence gap is in-house and no library touches it.** Run 1's
verdict is the artifact not storing the horizon configuration the sweep tripped on; Run 2's
out-of-coverage voxel is an evidence-decode defect. Both persist under any stack.

**4. On intended contact, geometry runs out and the field's answer is force.** At the moment of a
grasp the true clearance *is* ~0 mm; no geometric margin at any resolution can pass "touch the drawer
handle" while stopping "crush the drawer handle". ISO/TS 15066 power-and-force limiting is the
industry answer for permitted contact, Franka's reflex thresholds the shipping implementation,
SARA-shield the application to learned policies. ADR-0097 scopes *where* contact is allowed; a force
bound is what would say *how much* — the axis the stack does not have.

**5. What real VLA deployments do: nothing — but the literature has caught up.** openpi, Octo, GR00T
and LeRobot ship clipping, joint bounds and an e-stop key; CRISP, the one ROS 2 learned-policy
controller stack, ships **zero** collision/contact safety (§12). §6.1 stands.

**Bottom line.** The kernel is the right *shape*; what it got wrong is two policy choices — a **binary
stop where the field uses graded slowdown**, and a **voxel-only world model where mm-scale work needs
the intended-contact target as modeled geometry (or a force signal)**.

## 10. Candidate paths

All three keep the E-stop latch and manual `estop_reset`, the topic contract, the OTel evidence
trail, the allocation-free hot path, ADR-0092/0097 and the ACM, and all three start with the
zero-cost fix: **record the tripping horizon configuration (and payload pose, #172) in the stop
evidence**. *All three shipped; what each did and did not deliver is in
`collision-validation-evidence.md` §6.*

### Path A — graded response inside the existing kernel (recommended first)

Adopt Servo's shape, not Servo: compute `exp(k·(min_distance − threshold))` from the sweep the
kernel already runs and scale the outgoing chunk (or shrink the accepted horizon) in the band
between `margin + proximity_threshold` and `margin`; the latch at true penetration is untouched.
*Effort*: kernel-internal, no new dependencies, one parameter family + hazard-log entry.

> **Correction (2026-09-04). The "converts the 9-of-15 stops" claim is withdrawn.** Those stops are
> recorded at **−0.29…−11.34 mm**, `hit.min_distance` is the reported pair's *true surface
> distance*, and both `world_collision_margin_m` and `world_voxel_margin_m` default to **0.0**, so
> the band is `[0, proximity_threshold]` in positive surface distance and all nine sit **below**
> it. Path A can honestly claim only that it slows the *approach*; converting them would mean
> grading into **negative** surface distance, which needs Safety-WG sign-off and a hazard entry.

**Binding precondition: gate the scaling on the action's declared semantics.** `ActionChunk`'s
`control_mode` mirrors `openral_core.ControlMode`, which includes **absolute** spaces
(`JOINT_POSITION`, `CARTESIAN_POSE`, `JOINT_TRAJECTORY`, `GRIPPER_POSITION`, `FOOT_PLACEMENT`,
`DEX_HAND_JOINT`); multiplying an absolute joint target by 0.5 commands the arm halfway to its **zero
configuration**, toward the obstacle as readily as away. Scale only `delta`/velocity/twist modes
(`ControlModeSemantics.mode`); for absolute modes **truncate the horizon or refuse the chunk**.

PACS ([arXiv:2511.06385](https://arxiv.org/abs/2511.06385), ICRA 2026) ran the direct experiment on
robomimic LIFT/CAN/SQUARE with a dynamic obstacle, 100 rollouts each:

| method | safe? | avg success |
|---|---|---:|
| unfiltered policy (same pipeline) | ✗ | 0.70 |
| reactive CBF projection | ✓ | **0.04** |
| single-action braking (SSM) | ✓ | 0.41 |
| **chunk-level graded braking (PACS-PFL)** | ✓ | **0.72** |

Two riders transfer: **scale the whole chunk, not per action** (+28 % in their ablation), and the
published mitigation for distribution shift is **observation-side**, so **audit every deployed
adapter's observation space for velocity / rate / step-index signals before shipping Path A**. No
manipulation-side chattering/deadlock report was found in six query formulations — **unestablished
in either direction**. Ecosystem check (§12): **no ROS 2 component ships arm-side distance-graded
scaling to reuse.**

*Outcome: shipped, and the graded slowdown did not reproduce — the robot travels 45 % further with
completion unchanged (#209 and both #204 arms).*

### Path B — promote the declared target to modeled geometry

Extend the ADR-0097 declaration's producer to ship the target's geometry (sim: the declared body's
meshes/primitives; real: a fitted primitive), and check link hulls / payload primitives against
*that* at mesh resolution while the voxel grid keeps covering everything undeclared. The
narrow-phase engine stays the in-house certified GJK, per §4.1. *Effort*: moderate — a geometry
field on the declaration path (dispatch → HAL → World State → kernel) plus kernel ingest mirroring
`ingest_place_region`.

*Outcome: shipped as ADR-0098, and the premise was wrong — the mm problem is the carried object
against everything it passes, not the declared target; 56 stops were the payload against the rest
of the kitchen. See `collision-validation-evidence.md` §6 and lever 2 of its lever table.*

### Path C — force-based contact gating for the declared phase

During a live ADR-0097 declaration, gate the intended-contact region by measured force/energy
rather than geometry — sim: MuJoCo contact forces the HAL already reads; real Franka: reflex
thresholds plus external-torque estimates. Geometry stays the authority everywhere undeclared.
*Effort*: largest — a new sensor contract through Layer 1/2, a sim/real seam, and force-threshold
calibration with its own hazard entries.

**The numbers exist, the maturity does not.** ISO/TS 15066:2016 Table A.2 (read from the standard):
hands/fingers **140 N quasi-static, 300 N/cm² peak pressure, ×2 transient**; Table A.3 body model
K = 75 N/mm, m_H = 0.6 kg. §A.3.4 states the standard's own actuation knob for a force bound *is
robot velocity*, so **A and C are one lever seen from two ends** and C composes after A. Cite **ISO
10218-2:2025** in hazard entries alongside TS 15066 for the Annex A tables. Realised budgets vary
by an order of magnitude for the same body region (SARA-shield 0.49 J, PACS's HANDOVER 0.014 J), so
**the threshold must be a per-declaration field, not a config default**. Nearest instances:
CompliantVLA-adaptor ([arXiv:2601.15541](https://arxiv.org/abs/2601.15541)) and FORGE
([arXiv:2408.04587](https://arxiv.org/abs/2408.04587), >1000 real trials, 15 N snap-fit). Two
caveats: SARA-shield is one lab, one arm, no repo; and **no paper validates MuJoCo contact-force
*magnitude* against real F/T sensors** (§21.7), so the sim side ships a logged calibration
parameter, not an equivalence claim.

*Outcome: shipped as ADR-0100's declaration-scoped contact-force gate, blocked on hardware force
calibration and dead in sim by design (#218).*

**Not recommended**: MoveIt PlanningScene as the runtime checker (10 Hz-class with distance +
octomap, discrete-only FCL), or cuRobo/nvblox as the safety path (cm-scale by construction; its
right role is plan-time checking and the depth-stream robot segmenter).

### Code map — where each addition lands (verified against `master`, 2026-08-30)

**Zero-cost evidence fix (all paths).** The `report` lambda (`lifecycle_kernel.cpp:780`) names every
hit; capture `q_check_` and `q_predict_`. `CollisionEvidence` (`schemas.py:10056`) grows the joint
row — no IDL mirror, it crosses the wire as JSON inside `FailureTrigger.evidence_json`. Recorder:
`sim.estop_ground_truth_snapshot` (`openral_hal/sim_sensor_bridge.py:1377`).

**Path A.** Seam: `safe_pub_->publish(*msg)` (`lifecycle_kernel.cpp:1057`); the distance to grade on
is folded per pair by `fold_pair` (`collision.cpp:949`) into `sweep_min_distance`. Generalise rather
than rebuild the advisory band's non-latching path — `CollisionHit::advisory`
(`collision.hpp:431`), `place_advisory_depth` (`collision.hpp:290–293`), the consecutive-refusal cap
(`lifecycle_kernel.cpp:243`, `:793`, `:1399`). **Precondition audit, preliminary:** grepping
`python/sim/src/openral_sim/policies/` for velocity terms finds hits in exactly one adapter —
`xvla.py:203` (gripper `qvel`) and `:207` (arm `joints["vel"]`); XR-1 / π0.5 / the state assemblers
are position-only, step-index and elapsed-time signals are unaudited, and XVLA must be excluded.

**Path B.** `PlaceDeclaration` (`schemas.py:2592`) + its IDL grow the geometry field; the wire for
the measured region exists (`PlaceRegion.msg`). Kernel ingest mirrors `ingest_place_region`
(`collision.cpp:1734`, `collision.hpp:877`, called from `lifecycle_kernel.cpp:2005`) — same
bounds-checked shape (finite, positive, ≤ 1.5 m per side, ≤ 8 m³).

**Path C.** Two seams exist and are *dormant*. `SafetyEnvelope.contact_force_threshold_n`
(`schemas.py:897`, default 30.0 N) is loaded, min-folded by `envelope_loader.py:394` and plumbed
into the C++ envelope (`lifecycle_kernel.cpp:143`, `envelope.hpp:70`, `envelope.cpp:80`) — but **no
check ever reads it**. `WorldState.contact_forces` (`schemas.py:3323`) has **zero producers and
zero consumers**. The sim force source belongs in `openral_hal/sim_sensor_bridge.py`.

## 11. What to remove, replace, or stop

Component-level list; whole-theme "stop" verdicts are in §2.2. Locations re-verified against
`master` 2026-08-30.

**Stop investing in:** octomap-clearing fidelity as the false-positive fix (12-year-unsolved
upstream, maintainer-confirmed Nov 2025); extending `tight_geometry` past link1/link2 (27/72
ceiling); extending the ADR-0097 margin-reduction mechanism (binding on 2/15 stops); further harness
building; further Nav2-side payload modelling; and **padding / offset-style tuning knobs anywhere**
— the community's verdict is "very difficult/impossible to tune", unchanged since
[moveit_ros#342](https://github.com/moveit/moveit_ros/issues/342) in 2013, and nobody has since
shown the knob resolving quantization-vs-contact ambiguity.

1. **Per-pixel `mj_ray` depth cast (#111) — replace with batched `mj_multiRay`, now that #180 has
   landed** (`python/sim/src/openral_sim/backends/depth_camera.py:118`). Its justification was the
   body-BVH cull skipping visual-only geoms, which #180 now filters out before the cast; the premium
   is 6.0 / 18.6 / 55.6 ms per frame at 4 / 301 / 1201 geoms against a 5 Hz budget.
   `synthesize_laser_scan_2d` casts per-beam for a different reason and must not be touched. *Risk:
   medium* — the failure direction is unsafe, so it needs #180's per-ray comparison first.

2. **`range_min_m: 0.55` on `panda_mobile`'s `base_scan` (`robots/panda_mobile/robot.yaml:471`) —
   remove or reduce to the sensor's real minimum.** It deletes every real obstacle inside 0.55 m to
   hide a chassis whose circumscribed radius is 0.43 m, and #143's fail-closed
   `payload_scan_filter_node` polygon filter supersedes it. The file contradicts itself:
   `front_depth` (`:487–504`) says `range_min_m` is **not** the self-filter, `base_scan`
   (`:446–448`) says it is. *Risk: low-to-medium, in the safe direction.*

3. **`CollisionHit::advisory` / the place advisory band (#176, shipped by #179) — replace with Path
   A.** Measured inert: zero firings across 20 scene runs, while 9 of 15 stops are a link against
   world occupancy at −0.29 to −11.34 mm, outside its scope. *Risk: **do not delete outright*** —
   the non-latching-drop path and severity ranking are the two hard parts of Path A and #179 already
   built them; widen the five bounds instead (`place_advisory_max_consecutive: 0` is the rollback).

4. **`mj_geomDistance` on the support-contact attestation path
   (`openral_hal/_sim_attachment_evidence.py:649`, inside `_probe_support_hits`) — replace with
   `openral_hal.convex_distance.convex_geom_distance`.** #170 established that it returns
   confidently wrong values for RoboCasa-fixture-vs-`panda_mobile`-mesh pairs in two silent modes
   (caveat 8), failing *toward closer* — here, attesting a contact that is not there on a **safety
   input**. The exposure is likely small (#170 measured 1101 of 1102 pairs agreeing at the
   shipped windows), but the probe's 1 mm window is no defence: #170 is explicit that **no
   `distmax`, no "only trust it below N mm" rule and no `ncon` cross-check separates the good
   answers from the bad**.

5. **`SafetyPassthroughNode` (`packages/openral_safety/.../supervisor_node.py`) — reduce to a test
   double, or retire.** Every ADR-0096-class change is implemented twice. *Risk: high, and the
   weakest of the five* — 20 files depend on it, including four live-ROS tests that use it *because*
   it is not the kernel (CLAUDE.md §1.11). A standing maintenance tax, not a removal proposal.

**Not recommended for removal, against first appearances:** `tight_geometry` / the 26-DOP+hull path
(#166), measurably faster than what it replaced; the SAM 2.1 vision-attachment leg (#134), opt-in
and default-off; and `_footprint_geometry.py`, still imported by `payload_scan_filter_node.py`.

## 12. Round-2 verification (2026-08-30) — eight questions, eight verdicts

Detailed evidence is folded into §4.1, §6.1 and §10; this is the scorecard.

| # | question | verdict | strongest evidence |
|---|---|---|---|
| Q1 | Does graded slowdown rescue task completion? | **Strengthens Path A** — with a binding precondition | PACS: 0.04 (reactive CBF) → 0.72 (chunk-graded) vs 0.70 unfiltered ([2511.06385](https://arxiv.org/abs/2511.06385)) |
| Q2 | Force-gating thresholds and maturity | **Strengthens the axis, weakens maturity** | ISO/TS 15066 Table A.2: 140 N / 300 N/cm² / ×2 for hands; budgets vary 0.014–0.49 J per task |
| Q3 | Any VLA-specific safety layers? | **Corrects §6** — field is no longer empty | VLSA/AEGIS: +59.16 % avoidance *and* +17.25 % success, code public ([2512.11891](https://arxiv.org/abs/2512.11891)) |
| G1 | Is coal a lower-risk narrow phase? | **No — downgraded to watch item** | Near-penetration issues open (#823/#755), fixes unreleased at tag 3.0.4 (§4.1) |
| G2 | Reusable ROS 2 arm-side SSM? | **None exists** — hand-rolling confirmed | Nav2 monitor base-only + threshold-triggered; PILZ SSM never left ROS 1; ros2_controllers has nothing |
| G3 | The 15× `consider_footprint` discrepancy (#143 vs #185) | **Mechanism found, number not** | `CostCritic::score()` `continue`s on free-space points (cost < 1) before `inCollision` is reached; #143's microbenchmark assumed all 56 000 points reach the call |
| G4 | Does CRISP (or any ROS 2 learned-policy stack) ship safety? | **No** — verified from source | `learnsyslab/crisp_controllers`: soft joint-limit repulsion + torque-rate slew only; `setCollisionBehavior` is never called |
| G5 | Any REP/standard to align with? | **None** | Full REP index read; only REP-2006 (cyber disclosure) is safety-adjacent |

**Coverage honesty:** the ROS Discord has no public archive; Reddit r/ROS and r/robotics had nothing
relevant; the `pwc` catalog 404'd six real arXiv papers found via web search — **an absence claim
must never rest on `pwc` alone**.

---

*Sections 13–23 are the second research pass (2026-08-30) over methods outside §§3–7's shortlist.
Every "adopt" is a candidate with a stated cost, not a decision.*

## 13. What the kernel actually needs, restated as test criteria

The two slop sources are additive and both **structural**, not tuning errors: **world-side
quantisation** (25 mm grid, half-diagonal 21.7 mm; a ~1 mm contact reads as up to ~15 mm of cube
penetration) and **robot-side over-approximation** (per-link OBB corner slop 27–76 mm,
[primitive study §4.2](collision-primitive-study.md)). Every method below is scored on **(a) mm
near-contact** — a ~1 mm intended touch vs a real penetration, *against a sensed world*; **(b)
dynamic worlds**; **(c) self-collision** better than OBB. Fixing one fixes half the problem.

Two constraints bound every "adopt": the hot path is **allocation-free C++** at 30–200 Hz (CLAUDE.md
§1.5, §2) and new dependencies must be Apache-2.0 / MIT / BSD (CLAUDE.md §4.4). The kernel already
exposes `WorldModel`, `VoxelGrid`, `check_voxel_collision`, `hull_cell_distance` and an
`AttachedModel`, so a different world representation is a **sibling of `VoxelGrid`**.

## 14. Learned and analytic distance fields for the robot body

This direction attacks the OBB slop term.

### 14.1 RDF — robot geometry as a Bernstein-polynomial distance field

Closed-form Bernstein basis products composed along the chain by FK, so the model is 24 KB. ICRA
2024, <https://arxiv.org/abs/2307.00533>; <https://github.com/yimingli1998/RDF>, **MIT**. *Docs
claim* whole-body MAE **1.41 mm** (spheres 5.91 mm in the same table), query 0.21–0.54 ms on an RTX
3060, no C++, no ROS 2. **Verdict: adopt as a body model, but you write the port** — and a learned
field is not conservative (§14.6).

### 14.2 CDF — configuration-space distance fields

Distance in *joint* space, so projection/IK is one gradient step. RSS 2024,
<https://arxiv.org/abs/2406.01137>; <https://github.com/idiap/cdf>, **MIT**. *Docs claim* MAE
**1.39–1.64 cm** on Franka — an order of magnitude worse than RDF, because CDF optimises gradient
consistency, not surface precision. **Verdict: watch, do not adopt.**

### 14.3 Neural-JSDF

<https://github.com/epfl-lasa/Neural-JSDF> (RA-L 2022). **No licence file at all** (the GitHub
licence endpoint 404s), so all-rights-reserved — a hard blocker under CLAUDE.md §1.9; accuracy is
23.0 mm MAE per RDF's table. **Verdict: irrelevant.**

### 14.4 Learned body SDF + SVM self-collision (Zhu et al. 2024)

<https://arxiv.org/abs/2409.14955> — tiny parallel MLPs for the robot SDF **plus SVMs specifically
for self-collision**, architecturally the closest match to what the kernel needs, but the abstract
gives **no error figure** (the "0.19 cm RMSE" in search snippets is not in the paper —
**unverified**) and there is no code. **Verdict: watch.**

### 14.5 ReDSDF — regularised deep SDFs

<https://arxiv.org/abs/2203.04739> (IROS 2022). The contribution is far-field regularisation: a
vanilla neural SDF degrades outside its training shell, exactly where a safety margin lives. No
official code. **Verdict: watch (concept only).**

### 14.6 The honest constraint on all of the above — Neural Implicit Swept Volumes

<https://arxiv.org/abs/2402.15281> (ICRA 2024, KUKA). *Docs claim* MAE **3.08–4.38 mm** and **0.6 %
false negatives** at a 5 mm margin — but the load-bearing part is that the authors state the network
**cannot be guaranteed conservative** and pair it with a geometric checker, net speedup dropping to
25–49 %. **This is the template any learned field must follow here**: the learned distance is a fast
*filter*, the geometric check keeps veto authority (CLAUDE.md §1.1).

### 14.7 Scene neural SDFs (iSDF and successors) — the negative result

iSDF (<https://arxiv.org/abs/2204.02296>, RSS 2022; **MIT**, archived 2023-02) *docs claim* **< 6 cm**
SDF error under a static-scene assumption — **worse than the 25 mm grid the kernel already has**;
the 2025–26 successors are navigation-flavoured with no mm claim. **Verdict: irrelevant.**

## 15. Point clouds instead of voxels — CAPT, and the SIMD planning family

This direction attacks the 25 mm grid.

### 15.1 CAPT — collision-affording point trees

An **exact** representation of a point cloud (no voxelisation, no occupancy inference) with
SIMD-parallel `collides(center, radius) -> bool` queries. RSS 2024,
<https://arxiv.org/abs/2406.02807>; <https://github.com/KavrakiLab/captree-rs> and the C++ copy in
VAMP, both **Apache-2.0**. *Source shows* (RSS PDF): **9.89 ns** mean per query vs **0.01 ms for an
OctoMap backed by FCL**; clouds at **mean dispersion 7 mm – 2.2 cm**; a RealSense D455 / UR5 run
filters 166,587 points to 2,732 at `r_filter = 2 cm`, **median 7.166 ms** end-to-end. Construction,
not query, is the superlinear bottleneck, and Lemmas V.1/V.2 make it provably conservative only by
inflating. **(a)** partially — the *voxel* quantisation goes, the residual becomes `sensor
dispersion + r_filter + sphere slop`; **(b)** yes; **(c)** no.

**Verdict: adopt — the strongest candidate on this page for the world side.** A `PointCloudWorld`
sibling to `VoxelGrid` behind a `check_point_cloud_collision` entry point shaped like
`check_voxel_collision` (`collision.hpp:766`) — not `check_world_collision`, which consumes a
`WorldModel` of base-frame capsules (`collision.hpp:715`). Costs any ADR must state: boolean, not
signed distance (the advisory band and witness machinery need distance); per-frame superlinear
construction; and a point cloud is a *surface sample* with no free-space/unknown distinction.

### 15.2 VAMP — vectorised sampling-based motion planning

<https://github.com/KavrakiLab/vamp> (**Apache-2.0**), *docs claim* median **35 µs** per Franka
MotionBenchMaker problem. Three facts checked, all *source shows*: **SIMD, not GPU**; **spheres**
(`robots/panda.hh`, `n_spheres = 59`, radii 0.012–0.08 m — the **same accuracy order as the kernel's
OBB slop**); **discrete, not continuous**. **Verdict: irrelevant as a planner, adopt its CAPT.**

### 15.3 pRRTC and Foam

pRRTC (<https://arxiv.org/abs/2503.06757>, **Apache-2.0**) is GPU RRT-Connect with VAMP's sphere
floor and no CCD. **Irrelevant.** **Foam** (<https://github.com/CoMMALab/foam>, MIT) converts a URDF
into a spherical approximation — **watch / cheap fallback**, a fine sphere decomposition being
strictly tighter than an OBB and trivially allocation-free. `batch-cc` and `SPaSM` are
**unlicensed**.

### 15.4 RTCollisionDetection — hardware ray-traced discrete *and* continuous CD

<https://arxiv.org/abs/2409.09918> (ICRA 2025);
<https://github.com/Ssz990220/RTCollisionDetection>, **MIT**, on NVIDIA OptiX. *Docs claim*
mesh-to-swept-volume CCD along B-spline paths, up to 3× / 9× over GPU sphere baselines. **Verdict:
watch, with a hard caveat** — B-spline swept CCD is the right shape for vetting a chunk, but it
binds the safety path to OptiX, a closed NVIDIA SDK (CLAUDE.md §1.9) ruling out a portable aarch64
path, plus GPU latency jitter.

### 15.5 MuJoCo Warp / Newton — explicitly disqualified by its own docs

<https://github.com/google-deepmind/mujoco_warp> (**Apache-2.0**). The collision pipeline **is**
usable standalone (`nxn_broadphase`, `sap_broadphase`, `primitive_narrowphase`, `sdf_narrowphase`
are public exports), but the official page disqualifies it twice: "optimized for throughput" not
latency, and **non-deterministic** — "There may be ordering or small numerical differences between
results computed by different executions of the same code." **Verdict: irrelevant on the vetting
path.**

### 15.6 Other GPU/CCD entries checked

- **Scalable-CCD** (**Apache-2.0**) — GPU tight-inclusion CCD for FEM, no articulated FK. **Watch**:
  its interval arithmetic is the only *provably conservative* CCD primitive found.
- **NeuralSVCD** (<https://arxiv.org/abs/2509.00499>) — no numbers in the abstract, repo 404'd.
  **Watch.**
- **RoboGPU** (<https://arxiv.org/abs/2603.01517>) — proposed GPU *hardware* testing link OBBs
  against octree AABBs with a 15-axis SAT, 3.1×/14.8× claimed, no code. **Irrelevant for adoption**,
  but an outside signal that OBB-vs-octree SAT is defensible.
- **OMPL GPU forks** — none found.

## 16. Dynamic-world mapping — what exists, and why none of it clears the wall

Licences read from each repository's own file, not from a paper.

### 16.1 Dynablox

<https://github.com/ethz-asl/dynablox> (RA-L 2023), **BSD-3-Clause**, *docs claim* 86 % IoU at
17 FPS from a high-confidence free-space layer. **ROS 2: no** — the only port found has no licence
file, another fork is GPL-3.0. **Verdict: watch the algorithm, ignore the package** — its free-space
conservatism is actively wrong for near-contact and the idea already ships in nvblox.

### 16.2 wavemap — the right representation, the wrong plumbing

<https://github.com/ethz-asl/wavemap> (RSS 2023), **BSD-3-Clause**. **It does go below 25 mm**: the
shipped `wavemap_livox_mid360_pico_flexx.yaml` sets `min_cell_width: {meters: 0.01}` for the depth
camera. **It has a true Euclidean SDF, unadvertised**
(`core/utils/sdf/full_euclidean_sdf_generator.h`) — **but it is a batch, whole-map, `const` one-shot
generator**, no update-in-place, a hard blocker at 30–200 Hz; and **ROS 2: no**, `main` has
`interfaces/ros1/` only. **Verdict: watch** — adopting it means porting it *and* writing an
incremental SDF.

### 16.3 Bonxai — a faster wrong answer

<https://github.com/facontidavide/Bonxai>. **Licence correction: MPL-2.0, not Apache** — file-level
weak copyleft, **not on the allowlist**, and the repo-root `package.xml` declares
`<license>TODO: License declaration</license>`. Default resolution 0.02 m with a log-odds sensor
model (an octomap clone), no SDF in the tree, **not thread-safe for writes**. **Verdict:
irrelevant.**

### 16.4 vdb_mapping — best ROS 2 hygiene here, wrong output type

<https://github.com/fzi-forschungszentrum-informatik/vdb_mapping> + `vdb_mapping_ros2`,
**Apache-2.0**, **the only mapper in this section with verified Jazzy CI**. **A stale-fact
correction that matters beyond this page: OpenVDB relicensed from MPL-2.0 to Apache-2.0 in v12.0.0
(2024-10-31)** — but Ubuntu 24.04's `libopenvdb-dev` is 10.x/11.x and therefore **still MPL**, so a
rosdep install can silently pull the copyleft version; pin ≥ 12. Output is log-odds occupancy and
the ROS 2 wrapper exposes **no SDF**. **Verdict: watch.**

### 16.5 nvblox dynamics — production-grade, and pointed the wrong way

`LICENSE.md` is **Apache-2.0** plus a **BSD-3-Clause** block for voxblox-derived files (GitHub's
`NOASSERTION` is an artefact), but the cuRobo↔nvblox bridge `NVlabs/nvblox_torch` restricts use to
"research or evaluation purposes only" — §5's non-commercial flag stands. `dynamics_detection.h` is
Dynablox ported to CUDA. **From the shipped configs:** `voxel_size: 0.05`,
`update_esdf_rate_hz: 10.0`, `esdf_mode: "2d"`, `occupied_region_half_width_m: 0.15` — **dynamic
obstacles are inflated by 150 mm**. **Verdict: watch — solves (b) properly and makes (a) strictly
worse.** Fine for the mobile base's world; keep it out of the manipulation volume.

### 16.6 The rest, briefly

- **Voxfield** (BSD-3) — non-projective ESDF, ROS 1, **dead since 2023-05**; fixes *bias*, not
  *resolution*. **Irrelevant.** **GPU-Voxels** (FZI) — dormant since 2023-08 and **CDDL**, ironic
  since it is the only one designed for *manipulator* voxel collision. **ROG-Map** — **GPL-3.0,
  rejected on licence**, as is the EGO-Planner family that shares its lineage.
- **DSP-Map** (MIT repo but `<license>TODO</license>`, ROS 1) — particle occupancy with **predicted
  future occupancy**; MAV-scale, no SDF. **Irrelevant to adopt**, idea worth reading.
- **Sub-cm mapping for manipulation**: nothing released — ParaMaP
  (<https://arxiv.org/abs/2512.22575>) and DB-TSDF (<https://arxiv.org/abs/2509.20081>) publish no
  resolution, rate, repo or licence, so both are **unverified**.
- CADGrasp (<https://arxiv.org/abs/2601.15039>) independently corroborates the diagnosis: an ablation
  choosing 5 mm voxels, 2.5 mm giving "only marginal gains", 10 mm "a clear performance drop".
  **10 mm is already too coarse for grasp-level geometry** — the kernel is at 25 mm.

**Section verdict.** No 2023–2026 mapper clears the mm wall at 30–200 Hz; the two with the right
ingredients pull opposite ways (wavemap has the resolution and the SDF but no ROS 2 and no
incremental update; nvblox has the engineering but 50 mm cells and 150 mm inflation). **§9's point 1
survives this pass** — with the caveat that §17 finds a way to stop resampling onto a grid at all.

## 17. Tesseract — the one library that beats MoveIt's discrete FCL path

<https://github.com/tesseract-robotics/tesseract>, v0.35.0, active. Multi-licensed **Apache-2.0 +
BSD-2 + BSD-3** per file; every collision and geometry header read carried the Apache-2.0 block.
**Caveat:** `tesseract_gui` and `tesseract_qt` are **LGPL-3.0** — never link.

### 17.1 Continuous collision checking — confirmed in source

`continuous_contact_manager.h` declares `ContinuousContactManager` with
`setCollisionObjectsTransform(id, pose1, pose2)`, and `bullet_cast_bvh_manager.h` /
`bullet_cast_simple_manager.h` implement swept/cast BVH; **the FCL backend contributes no continuous
manager**. At 30–200 Hz a chunk moves the tool tens of millimetres between samples and a discrete
check tunnels through thin geometry — a failure mode the kernel's staged 26-DOP → GJK pipeline
inherits. Results carry `cc_time[2]`: a truncation point, not an E-stop.

### 17.2 Per-pair margins — the correct shape for the intended-contact problem

*Source shows*, from the manager interface: `setCollisionMarginData`,
`setCollisionMarginPairData`, `setDefaultCollisionMargin`,
`setCollisionMarginPair(id1, id2, margin)`, `incrementCollisionMargin`, plus
`setContactAllowedValidator(...)` — a generalisation of the SRDF ACL. Contacts closer than the
margin are "in collision", order-independent. `ContactResult` carries `distance` (negative =
penetration), `nearest_points`, `transform[2]` and a separation normal; `ContactRequest` has
`ContactTestType {FIRST, CLOSEST, ALL, LIMITED}` and `calculate_penetration`.

This is strictly richer than MoveIt's binary ACM plus a single `contact_distance`, and it is exactly
what the kernel lacks: **a negative margin on *one* declared pair** (gripper finger vs the target
object) while every other pair stays conservative. §1 item 3 — "touch-links / attached-object
semantics as a first-class contract" — is this feature.

### 17.3 New in 2026: implicit-SDF collision and an SDF geometry type

Not in the docs; found by reading the tree. `implicit_sdf_collision_solver.h` (©2026, Apache-2.0)
adapts MuJoCo's multi-start SDF strategy (deterministic Halton seeds over the margin-expanded AABB
overlap, nearest points recovered by projecting onto both zero level sets).
`geometry/impl/signed_distance_field.h` is a first-class `SignedDistanceField : public Geometry` with
a **lazy, function-backed** mode (`BatchedSignedDistanceFunction`) whose docstring names "e.g. a
batched nvblox ESDF GPU query" and states "No grid is sampled up front … (exact, no resampling)".
**That is a door in the voxel wall.** Two hard constraints, verbatim: "It is concave, so it is not
supported by continuous/cast managers" → **SDF geometry and CCD are mutually exclusive**; and
serialization and equality force `discretize()`.

### 17.4 Octomap worlds, ROS 2 status, and real-time honesty

Octomap is supported (`geometry/impl/octree.h`), but each cell becomes a Bullet box or sphere at the
*source* resolution, so **feeding the kernel's 25 mm octomap to Tesseract reproduces the 15 mm
phantom penetration exactly**; the win only materialises via `SignedDistanceField`. **No ROS 2 binary
exists** (index.ros.org lists v0.35.0 for Noetic only) though `tesseract_ros2`'s CI builds jazzy.
**Real-time:** structurally against CLAUDE.md §2 — clone-per-thread managers, `contactTest` filling
an allocating `ContactResultMap`, and margin setters that **throw**, i.e. exceptions across what
would be the safety-kernel boundary. **No published latency benchmark.**

### 17.5 Verdict

**Adopt selectively: link `tesseract_collision` (Bullet cast BVH + per-pair margins) and
`tesseract_geometry::SignedDistanceField`; do not adopt `tesseract_ros2`.** (a) **partially, and
better than anything else in either survey**, though its octomap path is bounded as today's is; (b)
**no** — it consumes a world; (c) **yes, cleanly**. The prototype gate is a single number nobody has
published: `contactTest` latency against the kernel's chunk budget.

## 18. Reactive control layers — five independent stacks, one shared inequality

Five independently developed reactive layers express collision avoidance as the *same* constraint —
a signed distance mapped to an **allowed velocity along the contact normal**, not a boolean.

| stack | the inequality | licence |
|---|---|---|
| NEO / Robotics Toolbox | `nᵀJ q̇ ≤ ξ(d−dₛ)/(dᵢ−dₛ) + nᵀv_obs` | MIT |
| mink | `nᵀ(J₂−J₁) q̇ ≤ gain·(d−d_min)/dt + relax` | Apache-2.0 |
| pink | CBF on `h = d − d_min` | Apache-2.0 |
| OCS2 | `d − d_min` as an MPC constraint | BSD-3 |
| fabrics | Finsler geometry leaves | **GPL-3.0 — rejected** |

All five need a signed distance **and the contact normal `n`**, out to an influence band `dᵢ`. A
grasp at 3 mm then gets a *small allowed approach speed* instead of an E-stop — the same conclusion
§3.4 reached from Servo's formula, arrived at independently by five codebases.

> **Correction (2026-09-04): the "~15 lines of C++ over the GJK output `hull_cell_distance` already
> produces" estimate is withdrawn.** That routine returns a single `double` (`collision.hpp:691`) — a
> supporting-hyperplane *lower bound*. It emits **no contact normal**, **early-exits** once the bound
> clears `margin`, and on overlap returns the caller's `fallback` instead of a depth. Supplying both
> changes the narrow phase's **output contract** and must preserve that routine's conservatism
> argument. The certified instrument added by #170/#204 reports exactly this pair
> (`ConvexDistance.distance_m` + `ConvexDistance.direction`, exact even at a flush contact) but is
> offline Python: the right **reference** for a C++ mirror, not the hot path.

### 18.1 mink `CollisionAvoidanceLimit` — the best-verified near-contact geometry

<https://github.com/kevinzakka/mink>, **Apache-2.0**. *Source shows*,
`src/mink/limits/collision_avoidance_limit.py`:

```python
dist = mujoco.mj_geomDistance(model, data, geom1_id, geom2_id, distmax, fromto)
row  = compute_contact_normal_jacobian(...)
if dist > min_dist: upper_bound[idx] = (gain*(dist - min_dist)/dt) + relaxation
else:               upper_bound[idx] = relaxation
sign = -1.0 if dist >= 0 else 1.0     # penetration flips the row
```

Defaults `gain=0.85`, `minimum_distance_from_collisions=0.005`. **`min_dist` may be negative** — "A
negative distance allows the geoms to penetrate by the specified amount" — a directly usable
per-pair knob for "this grasp is allowed to touch"; and `_construct_geom_id_pairs` *derives* an
allowed-collision list with exactly the three filters the SRDF ACL encodes.

**Verdict: adopt as design reference. NOT as an oracle.** ~330 lines of Apache-2.0 specifying what
the kernel should return instead of a boolean. The oracle half is **withdrawn** — mink's distance
*is* `mujoco.mj_geomDistance`, which §22.1 and §11 item 4 record as returning confidently wrong
values on exactly the pair class such an oracle would target;
`openral_hal.convex_distance.convex_geom_distance` certifies every answer and reports the contact
normal mink's damper row consumes.

### 18.2 NEO and the holistic controller (Haviland & Corke)

<https://arxiv.org/abs/2010.08686> (RA-L 2021); `robotics-toolbox-python` and friends, **MIT**.
*Source shows*, `Robot.py::link_collision_damper`: the QP row is bounded by
`xi*(d - ds)/(di - ds) + dp`, where **`dp = norm_h @ shape.v` is the obstacle's own velocity** — the
only mechanism found in this pass that makes a *moving* obstacle part of the constraint rather than
a re-plan trigger. Two caveats, both *source shows*: the published holistic base+arm examples
contain only `joint_velocity_damper`, no obstacles, so the two are demonstrated *separately*; and
there is **no self-collision helper**. **Verdict: adopt the formulation, not the package.**

### 18.3 pink `SelfCollisionBarrier`

<https://github.com/stephane-caron/pink>, **Apache-2.0**. The collision work lives in `barriers/`,
not `limits/` (`ConfigurationLimit` is joint-space only — a common misattribution).
`self_collision_barrier.py` builds a CBF on `h(q) = d(p¹,p²) − d_min` (default 20 mm) over the N
closest pairs from coal via Pinocchio, with the docstring caveat "for non-smooth collision
geometries behaviour is undefined" — and boxes and hulls *are* non-smooth. **Verdict: watch.**

### 18.4 Fabrics, RMPflow — rejected on licence and on maintenance

TU Delft `fabrics` is **GPL-3.0**, **disqualified** under CLAUDE.md §4.4 without TSC review, despite
*docs claim* of 500 Hz replanning. Two ideas worth stealing: `ESDFGeometryLeaf`, whose contract is
"give me φ, J, J̇ as external symbols"; and that `set_self_collision_avoidance()` is **commented
out** of the default problem path. NVIDIA's geometric fabrics have no open implementation (Isaac
Sim's Lula, **no published licence**), `rmp2` (MIT) is dead since 2021 — **no maintained,
permissively licensed RMPflow exists in 2026.**

### 18.5 Pinocchio 3 + coal derivatives, and the zero-distance degeneracy

`pinocchio` is **BSD-2** and the **only** thing in this pass with released ROS 2 Jazzy binaries
(4.1.0, with `coal`). **An honest correction to a common assumption:** Pinocchio advertises analytic
derivatives of RNEA/ABA, **not** of collision distance. Everyone — OCS2, pink — builds the distance
gradient from `nearest_points` plus joint Jacobians, an approximation that **degenerates exactly at
`min_distance == 0`**; OCS2's source carries the admission `// TODO(perry): is there a way to
calculate a correct jacobian for the case of distanceVector = 0?`. That is precisely the regime the
kernel keeps landing in. The one paper attacking it, **iDCOL**
(<https://arxiv.org/abs/2602.03250>, Feb 2026), has an **unverifiable repo and licence**. **Verdict:
watch closely.**

## 19. Whole-body mobile manipulation, base + arm coupled

**There is no shipped ROS 2 stack for this, and the published art is coarser than what the kernel
already has.**

### 19.1 OCS2 — mine the self-collision algebra, do not adopt the stack

<https://github.com/leggedrobotics/ocs2>, **BSD-3**, active. `SelfCollision.cpp` computes
`violations[i] = distanceArray[i].min_distance - minimumDistance_` with the gradient from
`getJointJacobian(..., LOCAL_WORLD_ALIGNED)` translated to the nearest points via
`skewSymmetricMatrix`, sign-flipped on penetration, with a real unit test — the reference
implementation of the Jacobian one would otherwise write from scratch. **ESDF obstacle cost is
weaker than assumed**: the only constraint consuming `DistanceTransformInterface` is
`EndEffectorDistanceConstraint`, so there is **no whole-body ESDF constraint in-tree**. **Verdict:
watch — steal `ocs2_self_collision`, inherit its zero-distance TODO as a known defect.**

### 19.2 RMMI — the closest published answer, and an order of magnitude too slow

<https://arxiv.org/abs/2408.16206> (IROS 2025): a **neural SDF** map consumed by NEO's
velocity-damper QP over a coupled base+arm Jacobian. *Source shows* "a control loop of 20 Hz" and a
**2358-point** body model; the benchmark repo has **no LICENSE**. **Verdict: watch strongly, adopt
nothing** — take the architecture (one shared QP over `[base_dof, arm_dof]`); 20 Hz is an order of
magnitude under the kernel's floor and 2358 points is *coarser* near contact than the 25 mm grid.

### 19.3 The rest of the base+arm field, and a calibrating data point

- **Reactive Base Control for On-The-Move Mobile Manipulation**
  (<https://arxiv.org/abs/2309.09393>) — shared QP over base+arm, 20 Hz, 48 % task-time reduction,
  but the obstacle model is a **2D lidar occupancy grid** and the gripper is a **single point**.
  **Irrelevant to adopt — valuable as calibration:** the field's *deployed* answer to base+arm
  coupling is coarser than this repo's current 3D voxel grid.
- **Zheng et al. 2025** (<https://arxiv.org/abs/2501.02815>, polytopic free regions + AL-DDP, **no
  licence**) and **Chen et al.** (<https://arxiv.org/abs/2409.14775>, base+arm CBF-QP, no rate, no
  code) are both **irrelevant**; the one takeaway is convex-decomposing *free space*. **AutoMoMa**
  (<https://arxiv.org/abs/2604.12565>) is GPU *dataset generation*, CC BY-NC-SA — named so its 80×
  is not mistaken for a control-rate claim. **Perceptive MPC** (RA-L 2020), ancestor of
  `ocs2_perceptive`: cite, don't build on.

The only released ROS 2 Jazzy packages here are the **geometry libraries** (`pinocchio` + `coal`),
not a controller. **Verdict for the kernel: build, don't adopt.**

## 20. Predictive safety filters for action chunks — and why none of them fix mm

**The headline is a negative result:** every predictive, reachability, CBF or flow-matching filter
found **inherits the resolution of whatever geometry it is handed**. An HJ value function or a
barrier function does not change that the 25 mm voxel says "penetration" when a fingertip touches.
Mostly *watch*, with two exceptions that are architectural, not geometric.

### 20.1 The exception worth adopting: mode-switched constraint sets

<https://arxiv.org/abs/2608.00600> (Enwerem et al., Aug 2026): a four-mode hybrid controller
(**reach → close → hold → lift**) with **hysteresis** on the number of fingers in contact, where the
constraint set changes per mode — in *reach* the target is a full obstacle; in *close* **coarse
finger-link constraints drop out of the active set** and **one fingertip clearance constraint per
finger enters at ZERO margin**; in *hold* the arm freezes under a wrench-quality CBF. *Docs claim*
convex hulls at load (**0.1 ms** per query vs **330 ms** exact mesh distance), CBF-CLF QP in
**0.09 ms inside a 20 ms interval**, minimum clearance **6.7 mm**. Licence and code **not stated**.

**Verdict: adopt the idea, not the code.** The binary accept/drop/latch at a *fixed* margin is the
root cause of the 15 mm false positive; the smallest-diff fix is a per-link, per-phase margin table
with hysteresis. **(a)** structurally yes — it does not achieve mm *sensing*, it **stops asking the
question** during grasp phases. ADR-0092/0097 built this bespoke; the paper adds the hysteresis the
in-tree version lacks.

### 20.2 Flow-matching and diffusion filters — watch

- **SafeFlow** (<https://arxiv.org/abs/2504.08661>) — barriers over the **whole planning horizon**,
  training-free at deployment, licence and ROS 2 **unverified**. The closest analogue to vetting a
  chunk as a whole, but it does nothing the kernel cannot get by evaluating its existing check
  across the whole chunk.
- **Neuro-Symbolic Safety Guidance via Constrained Flow Matching**
  (<https://arxiv.org/abs/2607.01378>) — corrects violations *during denoising*; *docs claim* 82.8 %
  avoidance / 81.6 % success on SafeLIBERO, collision representation **not stated**, no code.
  **Watch.**
- **Individual-CBF-guided diffusion** (<https://arxiv.org/abs/2606.12640>) — multi-agent offline RL,
  CC BY-NC-ND. **Irrelevant.**

### 20.3 Reachability and MPSF — irrelevant at manipulator scale

- **refineCBF** — **no LICENSE file**, ROS Noetic, 2D platforms (the usable piece underneath is
  `StanfordASL/hj_reachability`, **MIT**). **Irrelevant** — HJ value functions in a 7-DoF
  configuration space are not tractable and no paper claims otherwise.
- **Language-conditioned latent safety filters** (<https://arxiv.org/abs/2608.00315>) — **latent
  filters are less metrically precise than a voxel grid, not more. Irrelevant to the mm problem;
  watch as an S2-layer complement**: it can express "don't touch the hot pan", which no metric
  kernel can.
- **UPSi** (<https://arxiv.org/abs/2604.26836>) — safe-RL benchmarks only. **Irrelevant.**
  **Towards Safe Robot Foundation Models** (<https://arxiv.org/abs/2503.07404>), a generalist policy
  in an ATACOM safe action space: **watch**.
- **SQ-CBF** (<https://arxiv.org/abs/2602.11049>) — superquadric SDFs via GJK, runtime/code/licence
  **not verifiable**. **Watch**: sub-voxel gradients, but the superquadric *fit* error becomes the
  new bottleneck.

**Standing correction:** PACS is now **accepted to ICRA 2026**; code/licence still unverified.

### 20.4 Real-Time Chunking — a liveness guarantee, not a safety one

<https://arxiv.org/abs/2506.07339> (Physical Intelligence, NeurIPS 2025) poses asynchronous chunking
as **inpainting** with inference delay `d := ⌊δ/Δt⌋`. **Read the guarantee precisely** — *source
shows*: "so long as `d ≤ H − s`, this strategy will satisfy the real-time constraint and guarantee
that an action is always available when it is needed", i.e. an **availability guarantee**. Latency
bounds any in-kernel design: π0 (3B) spends **46 ms on KV-cache prefill alone** on an RTX 4090
against a 20 ms tick; the π0.5 real-robot setup measures **76 ms baseline / 97 ms RTC**, giving
**d ≈ 6**.

**The kernel-relevant consequence.** The frozen prefix of length `d` — ~6 steps, ~120 ms — is a
window in which the policy **cannot** re-plan but the world **can** move, and nothing in RTC
re-checks it. **Verdict: adopt RTC as an execution strategy, and treat its frozen prefix as an
explicit kernel contract** re-vetted against fresh perception every cycle, with braking authority
inside it. For tracking only: PACE (<https://arxiv.org/abs/2606.00537>), action-prior denoising
(<https://arxiv.org/abs/2605.25537>), DREAM-Chunk (<https://arxiv.org/abs/2606.18589>), adaptive
inference-time chunking (<https://arxiv.org/abs/2604.04161>).

### 20.5 Runtime monitoring — one cheap signal worth taking

**VLA-FAIL** (<https://arxiv.org/abs/2606.21386>, KIT, June 2026) offers two failure detectors
needing **no failure data**: last-layer Mahalanobis distance, and the useful one, **Action Chunk
Consistency (ACC)** — flag a failure when *consecutive chunks become inconsistent*; code
**unverified**. **FAIL-Detect** (<https://arxiv.org/abs/2503.08558>) adds conformal thresholds
(**watch**); **KnowNo** (<https://arxiv.org/abs/2307.01928>) is an **S2 / Reasoner-layer** control.

**Verdict: adopt ACC as a third verdict.** Nearly free — consecutive chunks are already in the RTC
buffer — and *"the policy is confused, escalate to S2"* is a different event from *"geometry says
collision, latch"*, mapping onto the replanning ladder, not the E-stop path.

**Also checked and not relevant:** <https://arxiv.org/abs/2604.23775> is an **adversarial-security**
survey, not physical safety, and does **not** supersede this page.
<https://arxiv.org/abs/2512.11908> ("Safe Learning for Contact-Rich Robot Tasks: A Survey", IIT) is
the nearest 2026 superset; only its abstract was verified, and it does **not** mention momentum
observers or intent discrimination.

## 21. Proprioceptive contact discrimination — where the mm problem is actually solvable

The kernel is trying to infer *intent* from *geometry* at a resolution geometry cannot deliver.
Force does deliver it — with the caveat that force measures **contact, not penetration depth**.

### 21.1 What a Franka-class arm can actually measure

*Source shows* — the official **Franka Emika Panda** datasheet
(<https://download.franka.de/Datasheet-EN.pdf>):

| quantity | value |
|---|---|
| force resolution | **< 0.05 N** |
| force repeatability | 0.15 N |
| **force noise (RMS)** | **0.035 N** |
| torque resolution | 0.02 Nm |
| **relative torque accuracy** | **0.15 Nm** |
| torque repeatability | 0.05 Nm |
| torque noise (RMS) | 0.005 Nm |
| pose repeatability | < ±0.1 mm (ISO 9283) |

**The FR3 datasheet publishes none of these numbers** — its full text has only "Force/Torque sensing:
link-side torque sensor in all 7 axes" and "Guiding force ~2.5 N", so **anyone citing "FR3 has
0.05 N force resolution" is citing the Panda sheet**; the FR3 sheet does give a worst-case safe
Cartesian position accuracy for stopping functions of **50 mm**, PL d / Cat. 3. **How
`tau_ext_hat_filtered` is computed is undisclosed** —
[libfranka#91](https://github.com/frankaemika/libfranka/issues/91), the only substantive maintainer
reply: "this data is streamed from the backend and computed within."

For *detection* the margin is comfortable: 0.035 N RMS noise against a 1–5 N deliberate fingertip
touch is 30–140×. On *intent*, no — a 3 N accidental brush and a 3 N deliberate touch are the same
number. The binding constraint is *relative torque accuracy 0.15 Nm*, roughly **0.3 N of unmodelled
force error** at a ~0.5 m lever, 6× the headline "< 0.05 N" *(derived arithmetic, not a Franka
claim)*. Proprioception cannot tell 1 mm from 15 mm, but it *can* tell "contact is occurring and it
is gentle".

### 21.2 The momentum-observer lineage, and the number that bounds it

**Haddadin, De Luca, Albu-Schäffer, T-RO 33(6), 2017** —
<http://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf> — is still canonical and
names the missing stage: detection → isolation → identification → **classification** → reaction →
post-collision, where classification means "accidental or intentional … light or severe", a decision
that "cannot be done purely at the control level". **The kernel today has detection wired straight
to latch, with no classification stage at all.**

**Birjandi & Haddadin, RA-L 2020**, verbatim: "the collision threshold is typically set to **1 Nm**
in the momentum observer … the error of the proposed solution **does not exceed 0.1 Nm** … the
collision is **detected within 1.2 ms**" — **but the cost is one IMU per link**, which an FR3 does
not have. **Verdict: watch, do not build** — start from stock `tau_ext_hat_filtered`; successors are
incremental threshold-reduction papers on the same 2017 observer, **none shipping ROS 2, none
claiming intent discrimination.**

### 21.3 The highest-value line: intent from the torque *spectrum*

**"Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots"** —
<https://arxiv.org/abs/2508.12435> (Aug 2025), Franka Emika Research robot, **built-in joint sensors
only**. *Docs claim* **over 95 % accuracy** in contact detection and gesture classification via
STFT-2D-CNN / STT-3D-CNN, the key finding being that **time-frequency representations significantly
outperform raw torque signals**; class count, force levels, code and licence **could not verify**.

**The most on-point result in either survey:** the *temporal-spectral signature* of a torque
transient, not its amplitude, carries the intent — a grasp closure and an accidental impact at the
same peak force differ as ramp vs broadband impulse — and a ~10 ms STFT over 7 channels at 1 kHz is a
small, allocation-free C++ addition. **Complementary: Aim-Aware Collision Monitoring** (RA-L
8(8):4609–4616, 2023, DOI 10.1109/LRA.2023.3284371) discriminates expected from unexpected
post-impact behaviour against an idealised rigid response. **Verdict: adopt the framing** — compare
against a predicted contact response, not a fixed margin.

### 21.4 Contact localisation — set expectations correctly

The **Contact Particle Filter** (Manuelli & Tedrake, IROS 2016) is the classic worth understanding —
a convex QP inside a particle filter finding the contact point(s) that best explain the measured
external joint torque. **UniTac** (<https://arxiv.org/abs/2507.07980>) does whole-robot touch sensing
from proprioception at ~2000 Hz but localises only **within 8.0 cm on a Franka arm**, and its repo
has **no LICENSE file** — **do not vendor**. **Verdict: budget for mm-accurate contact *detection*,
not mm-accurate *localisation*** — the published open number is 7–8 cm.

### 21.5 The plumbing already exists — this is the cheap rung

*Docs claim*, franka_ros2 (Jazzy): `franka_hardware` exposes a **`ForceTorqueSensor` named
`<arm_prefix><robot_type>_tcp`** carrying **`K_F_ext_hat_K`** as six state interfaces, and the
**Gazebo plugin mirrors the same tcp wrench interfaces**, so sim and hardware controllers activate
identically. (The docs do **not** list `tau_ext_hat_filtered` or `O_F_ext_hat_K` as separate
interfaces — "docs do not show it", not "it does not exist".) `ros2_control`'s
`force_torque_sensor_broadcaster` publishes `WrenchStamped`, and in-controller access goes through
`semantic_components::ForceTorqueSensor` — **no topic hop on the hot path**.

**One landmine:** the Universal Robots driver's broadcast wrench is relative to `base_link` and is
"almost always incorrect as the robot's pose changes"
([UR driver#235](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/235)); an
external wrench in the wrong frame is worse than none, and CLAUDE.md §2's "TF2 is the only source of
coordinate frames" applies to wrenches too. **Verdict: adopt — the plumbing costs nothing**, with
**zero new dependencies** on an FR3.

### 21.6 Gripper-level grasp confirmation — a phase signal, not a safety signal

*Docs claim*, libfranka `Gripper::grasp()` returns true iff
`(width − epsilon_inner) < d < (width + epsilon_outer)`, **defaults
`epsilon_inner = epsilon_outer = 0.005` m — a ±5 mm width band**, and force is a *commanded*
parameter, not a measured confirmation. It is a width predicate: it cannot discriminate a 1 mm touch
and says nothing about arm-link contact. **Verdict: adopt as a *phase* signal only** — a good input
to the §20.1 mode switch, worthless as a contact-force measurement. Adjacent 2026 work with
**unverifiable code/licence**: "Current as Touch" (<https://arxiv.org/abs/2607.03529>), **FACTR 2**
(<https://arxiv.org/abs/2606.12406>).

### 21.7 MuJoCo contact-force fidelity — an honest negative, and a blocker

**No 2023–2026 paper was found validating MuJoCo or MJX contact-force *magnitudes* against real
force-torque measurements.** Five search phrasings were tried. What *was* established:

- **MuJoCo's own documentation frames its contact model as a deliberate approximation, not a
  calibrated force model** — *docs claim*
  (<https://mujoco.readthedocs.io/en/stable/computation/index.html>): it "drops the strict
  complementarity constraint at the heart of the LCP formulation", so "force and velocity in the
  contact normal direction can be simultaneously positive", justified because "all physical
  materials allow some deformation". The docs nowhere claim forces are in calibrated real Newtons,
  and put the burden of physical validity on the user's `solref`/`solimp` choices.
- The strongest indirect evidence is **"Direction Matters: Learning Force Direction Enables
  Sim-to-Real Contact-Rich Manipulation"** (<https://arxiv.org/abs/2602.14174>, 2026), whose premise
  is that force *magnitudes* are "highly sensitive to simulation inaccuracies" while force
  *directions* "remain robust across the sim-to-real gap" — an argument *from* the magnitude gap,
  not a measurement of it.

**Verdict: record this as a blocker.** The sim path cannot produce a trustworthy real-Newton analogue
today, so it should emit **contact-occurring boolean plus force direction** and any magnitude
threshold must be calibrated on hardware only. Anything else fabricates a number, which CLAUDE.md
§1.2 forbids. This is the citation behind §10 Path C's standing caveat that its sim seam ships "a
calibration knob, not a claim".

## 22. Opportunistic finds outside the eight directions

### 22.1 `mj_geomDistance` is already installed — and must NOT be the oracle

**This subsection's original verdict is withdrawn.** It read "adopt as the offline oracle first",
because MuJoCo ships a mesh-exact signed distance and is already a dependency. The premise is true
and the conclusion is wrong: **that call is the defective instrument PR #170 removed from the
evidence path and PR #204 (issue #190) removed from the last safety path** (§11 item 4; standing
caveat 8 of the [validation-evidence ledger](collision-validation-evidence.md)).

**The trap was inferring accuracy from the `nativeccd` default** (GJK/EPA replacing MPR since MuJoCo
3.3.0), whose docs only say "distances are inaccurate when using the legacy CCD pipeline". #170
falsified that by measurement: under `mujoco 3.8.0` on the **default native-CCD path** the call
returns `+0.000000` for `robot0_link7_collision` against `fridge_right_group_freezer_door_main` on
`robocasa_fridge_drawer` layout 9, where the certified truth is `+0.148512 mm`, writing a 126.264 mm
witness segment whose endpoints lie 526.6 mm and 432.9 mm outside the two geoms it claims to touch.
Displacing the link by one picometre returns the right answer, so it is a degenerate *configuration*
— and a scene's reset pose is where such configurations live. #170 states it flatly: **no `distmax`,
no "only trust it below N mm" rule and no `ncon` cross-check separates the good answers from the
bad.** Four checked-in `0.000 m` readings re-measure at +14.8, +82.2, +98.8 and +107.9 mm.

**Corrected verdict: do NOT adopt `mj_geomDistance` as the oracle, or as anything else on an
adjudication or permission path.** The requirement is real — the 26-DOP → hull staging does need an
mm-scale regression oracle — but the instrument exists in-tree:
**`openral_hal.convex_distance.convex_geom_distance`**, which *proves* each answer (separating-axis
certificate when separated, exact SAT when overlapping, bracketing for round types, explicit refusal
otherwise) at ~1 ms per mesh↔box pair against ~0.8 µs, and adds **no** dependency.
`mj_geomDistance` keeps one legitimate use: as the *subject* of the two regression tests asserting
the defect still reproduces (`tests/sim/safety/test_geom_distance_instrument_robocasa.py`,
`tests/sim/safety/test_support_probe_instrument_robocasa.py`).

### 22.2 Contact-implicit MPC — contact as a decision variable, not a violation

**C3 / C3+ (consensus complementarity control)**, Posa's DAIR lab; Push Anything
(<https://arxiv.org/abs/2510.19974>) reports *docs claim* **99.9 % (700/701)** single-object success
at **~14 Hz**, code in `DAIRLab/dairlib`, **MIT**. **Verdict: irrelevant to the kernel, important as
a framing** — 8–15 Hz is far below the 30–200 Hz floor, but it is the clearest existence proof that a
whole research line treats contact as a *decision variable* rather than a *safety violation*, the
same conclusion §20.1 reaches from the grasp side.

### 22.3 A cautionary "sub-millimetre" claim, checked

**"Learning Fast, Tool-aware Collision Avoidance for Collaborative Robots"**
(<https://arxiv.org/abs/2508.20457>, RA-L) is often summarised as achieving "sub-millimetre
accuracy". *Source shows*: that figure is **task-space position tracking error** in nominal
operation, **not** collision discrimination; the actual geometry is "a voxel grid with 0.05 m
resolution". **Verdict: irrelevant, and recorded deliberately** — exactly the class of claim that
must be followed to the primary source before it influences a safety decision (CLAUDE.md §1.2).

### 22.4 Self-filtering the robot and its payload out of depth — a ROS 2 option

§3.3 covers MoveIt's octomap self-filter; the standalone ROS 2 alternative is
**`leggedrobotics/robot_self_filter`** (*source shows* `<license>BSD-3-Clause</license>`,
`ament_cmake`, `rclcpp`, `tf2`, `urdf`), while the widely cited `ctu-vras/robot_body_filter` remains
**ROS 1**. **Verdict: watch** — marginal on its own, but its *shadow test* (removing points seen
*through* a robot link) is the mechanism most directly relevant to the carried-payload false
positives, and no equivalent exists in-tree.

### 22.5 The force-limiting standard §8 cites has been superseded

§8's comparison table rests its only "yes" on **ISO/TS 15066**. The 2025 revision of the ISO 10218
series folded the power-and-force-limiting requirements in (**ISO 10218-1:2025** and **ISO
10218-2:2025**), retiring the TS as a standalone document. Cite ISO 10218-2:2025 in hazard-log
entries, keeping TS 15066 for the Annex A body-model tables (Table A.2/A.3 re-verified against the
standard's own PDF, §12).

## 23. Comparison table — scored against the kernel's three needs

"mm near contact?" asks whether the method can distinguish a ~1 mm intended touch from a real
penetration, **against a sensed world**.

| method | what it changes | (a) mm near-contact | (b) dynamic world | (c) self-collision | rate / resolution | ROS 2 | licence | verdict |
|---|---|---|---|---|---|---|---|---|
| **CAPT** (§15.1) | world: points instead of voxels | **partially** — no voxel quantisation; sensor dispersion 7 mm–2.2 cm + sphere/filter slop remains | **yes** — rebuild per frame, 60 FPS demoed | no | 9.89 ns/query; 7.2 ms end-to-end at 2.7k points | none (3★ prototype only) | Apache-2.0 | **adopt** (world side) |
| **Tesseract collision** (§17) | per-pair margins + swept CCD + SDF geometry | **partially** — best available; its octomap path is not | no (consumes a world) | **yes** — `ContactAllowedValidator` + per-pair margins + CCD | no published latency | source-only, Jazzy in CI | Apache-2.0 (+BSD) | **adopt selectively** |
| **velocity damper** (§18, NEO/mink/pink/OCS2) | verdict semantics: distance → allowed normal velocity | **yes** — graded, negative margins expressible | **yes** with NEO's `+nᵀv_obs` term | yes (mink derives the ACL) | new narrow-phase output contract (distance + normal) | none | MIT / Apache-2.0 / BSD-3 | **adopt the formulation** |
| **MuJoCo `mj_geomDistance`** (§22.1) | mesh-exact signed distance + witness | **no — measured wrong on this pair class** (#170) | modelled objects only | n/a | CPU per-pair, ~0.8 µs | n/a (already a dep) | Apache-2.0 | **REJECTED — use `openral_hal.convex_distance` instead** |
| **external wrench** (§21.5) | measures contact, not proximity | **yes for detection** — 0.035 N RMS vs 1–5 N touch; cannot measure depth | n/a | n/a | control rate, zero new deps | **shipped** (`franka_ros2` Jazzy, `ros2_control`) | Apache-2.0 | **adopt** |
| **torque-spectrum intent classification** (§21.3) | separates intended from accidental at equal force | **yes, structurally** | n/a | n/a | ~10 ms STFT, 7 ch @ 1 kHz | none | unverified | **adopt the idea** |
| **phase-switched constraint sets** (§20.1) | stops asking the geometric question during grasp | **yes, structurally** | no | yes | QP 0.09 ms in 20 ms | none | unverified | **adopt the idea** |
| **RDF Bernstein body field** (§14.1) | robot: 1.41 mm field replaces OBBs | **yes** (robot side only) | no | partially | 0.21–0.54 ms/query on GPU | none | MIT | **adopt, port yourself** |
| **wavemap** (§16.2) | 1 cm adaptive cells + true Euclidean SDF | at representation level, yes | no | no | batch SDF generation only | **no** (dead ROS 2 branch) | BSD-3 | watch |
| **nvblox dynamics** (§16.5) | freespace-intrusion dynamic detection | **no** — 50 mm cells, 150 mm inflation | **yes** | no | 10 Hz ESDF, 2D sliced | **yes**, Jazzy | Apache-2.0 (+BSD-3) | watch (base only) |
| **RTCollisionDetection** (§15.4) | ray-traced mesh CCD along B-splines | yes (mesh-exact) | by re-query | yes | 3×/9× over GPU sphere baselines | none | MIT, **OptiX-bound** | watch |
| **iDCOL** (§18.5) | gradient that survives zero distance | targets exactly this | — | yes | unpublished | none | **unverified** | watch closely |
| **RTC frozen prefix** (§20.4) | names a ~120 ms unre-plannable window | n/a | **it is the blind spot** | n/a | d ≈ 6 steps at π0.5 rates | via LeRobot | **unverified** | adopt as a contract |
| **VLA-FAIL ACC** (§20.5) | third verdict: "policy confused" | no | indirectly | no | free — chunks already buffered | none | unverified | adopt the signal |
| **VAMP / pRRTC** (§15.2–15.3) | fast planners | no — 12–80 mm spheres | via replanning | via spheres | 35 µs/plan | none | Apache-2.0 | irrelevant (planner) |
| **Bonxai** (§16.3) | faster octomap | no | no | no | 20 mm default | undeclared | **MPL-2.0** | irrelevant |
| **fabrics** (§18.4) | Finsler reactive layer | no (spheres/capsules) | yes | leaf exists, disabled | 500 Hz claimed | ROS 1 only | **GPL-3.0** | rejected |
| **MJWarp** (§15.5) | GPU collision pipeline | n/a | n/a | n/a | throughput, **non-deterministic** | no | Apache-2.0 | irrelevant on the safety path |
| **GPU-Voxels / ROG-Map / Neural-JSDF / UniTac / RMMI code** | — | — | — | — | — | — | **CDDL / GPL-3.0 / unlicensed** | rejected on licence |

## 24. What this survey does NOT establish

- **No latency was measured in this tree.** Every rate quoted — coal's and MoveIt's µs numbers,
  CAPT's 9.89 ns, RDF's 0.54 ms, Tesseract's `contactTest` (no published number at all) — is an
  upstream benchmark on other hardware; the §15.1 and §17 decisions turn on a measurement that does
  not yet exist. Servo's formula is likewise read from source, with no claim that it is *rated*.
- Whether graded slowdown (Path A) converts *this repo's* within-quantization stop class into
  completed grasps. *(Since measured: it does not reproduce —
  `collision-validation-evidence.md` §6.)* The two motivating drawer runs still have no in-tree
  fixture; both defects need recorder-side pinning.
- The VLSA-vs-PACS disagreement: VLSA reports a CBF layer *raising* success (+17.25 %) where PACS's
  reactive-CBF baseline collapses it (0.04).
- **MuJoCo contact-force *magnitude* is unvalidated as a real-Newton analogue** (§21.7); Path C's sim
  seam ships a calibration knob, not a claim.
- G3's mechanism (the free-space `continue`) is read from source but the numeric split of the 15× is
  uninstrumented.
- coal's near-contact fixes are on `devel`, unreleased at tag 3.0.4 — re-check §4.1's downgrade when
  Path B's narrow-phase question reopens. cuRobo's Apache-2.0 relicense was verified at `main`;
  anything pinned ≤ 0.7.x and `nvblox_torch` remain non-commercial (CLAUDE.md §1.9).
- **CAPT would not make the kernel mm-accurate on its own** — the combined residual (sphere/OBB slop
  + filter radius + 7 mm–2.2 cm dispersion) is unestablished.
- **No learned distance field found is conservative**, so §14.1 can only be a *filter in front of*
  the geometric check; and **Tesseract's SDF and CCD paths are mutually exclusive**, with nothing
  here establishing which the kernel should choose.
- **The Franka force numbers are Panda numbers**; the FR3 sheet publishes none of them,
  `tau_ext_hat_filtered` is undisclosed, and the 0.3 N lever-arm figure in §21.1 is derived. **The
  torque-spectrum intent claim rests on one paper's abstract**, with nothing establishing that the
  same signal separates a *grasp* from a *collision*.
- **Several load-bearing repos have no licence at all** — Neural-JSDF, UniTac, RMMI's
  `frankie_planner`, Zheng et al., refineCBF, `dynablox_ros2`.
- **Nothing here has been run.** No code built, no benchmark reproduced, no fixture added. This is a
  reading list with verdicts attached.
