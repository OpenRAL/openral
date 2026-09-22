# Collision-safety alternatives survey — is the hand-rolled kernel the right shape?

> **Status: research survey, analysis only (2026-08-30).** Nothing here landed as
> written. It is the evidence a reviewer would need to decide whether the
> mm-scale-discrimination problem the
> [validation evidence](collision-validation-evidence.md) documents should be solved
> inside the kernel or by adopting an external stack.
>
> **Read with the correction.** All three candidate paths shipped within a week, and
> measurement since overturned four of this page's claims — see
> `collision-validation-evidence.md` §6, *The survey, reviewed against what
> happened*. Affected sections point at it.
>
> Companions: the [collision-primitive study](collision-primitive-study.md), the
> [validation evidence ledger](collision-validation-evidence.md), and
> [world-map fidelity](world-map-fidelity.md).
>
> **Conventions.** "docs claim" = documentation or a paper abstract; "source shows"
> = the file was fetched and read; anything unverifiable against a primary source is
> marked **unverified** and supports no verdict. §§13–23 are a second research pass
> over methods outside §§3–7's shortlist.

---

## 1. The problem being shopped for

OpenRAL runs VLA policies that emit action chunks with no collision awareness, and a
C++ safety kernel vets every chunk: self-collision via per-link OBBs with an
SRDF-derived 16-pair allowed-collision list, world collision via an octomap-derived
25 mm voxel grid, with a staged 26-DOP → convex-hull-GJK narrow phase for links
declaring `tight_geometry` (`robots/panda_mobile/robot.yaml`;
`cpp/openral_safety_kernel/README.md`). The measured state:

- **Contact-rich tasks E-stop before the grasp completes.** Run 1 stopped on
  self-collision `panda_link2` vs `panda_link5` at −5.34 mm while offline
  adjudication at the *recorded* joints shows the links **+53 mm apart** — the
  verdict was for a predicted horizon configuration the artifact does not store
  ([#172](https://github.com/OpenRAL/openral/issues/172)). Run 2 stopped on
  `panda_link7` vs a voxel at −2.96 mm whose position resolves 2.2 m from link7,
  outside the grid's own r = 1.05 m coverage.
- **Of 15 stops in the five-round battery, 11 classified `within-quantization`** —
  ~1 mm real contacts reading as up to 15 mm of penetration because 25 mm voxels
  inflate thin geometry (2026-08-13 baguette run: −15.70 mm reported against six
  MuJoCo contacts of −0.87…−1.37 mm). Nine of the fifteen are a link against world
  occupancy at −0.29…−11.34 mm.
- **OBB corner slop is 27–76 mm of protrusion per link** (45–88 mm as corner slop),
  on top of the 21.7 mm voxel half-diagonal
  ([primitive study §4.2](collision-primitive-study.md)).

The kernel stays, E-stop latching on real contact is wanted, the evidence trail
stays. What is in question is **mm-scale discrimination near contact**. The kernel
lacks three things established stacks have:

1. **distance-based speed scaling** — the only verdict is accept / drop / latch at a
   fixed margin;
2. **mesh-level checking** — the world side is voxels vs primitives;
3. **touch-links / attached-object semantics as a first-class contract** — the
   gripper is absent from `collision_geometry` ("checking it against the world would
   veto every grasp"), and intended-contact scoping was built bespoke (ADR-0092
   witness, ADR-0097 declaration).

---

## 2. What the last two weeks built — PR ledger and verdicts

54 merged PRs (2026-08-11 → 08-29), ~52 distinct changes, cross-checked against the
[validation evidence ledger](collision-validation-evidence.md)'s 8 standing caveats
(3 of them withdrawals). The instruments and the voxel-grid fidelity line earned
their cost — each found a trusted mechanism wrong in the *unsafe* direction. The
link-envelope line spent four PRs proving perfect link geometry recovers only 27/72
stops. The place-allowance line is armed but binding on 2 of 15 stops. Roughly ten
days of adjudication output was voided by the pipeline's own later corrections.

### 2.1 PR ledger

The load-bearing findings; the rest are schema, mirror, CI and packaging repairs
whose verdicts are in §2.2.

| PR | theme | what it established, or what it cost |
|---|---|---|
| #101 / #103 / #117 | frame/FK | Three PRs and one ADR for one constant: #101's FK change put every kernel link 0.7 m too low (joint 1 1.033 → 0.333 m); #103 half-fixed it and replaced 7 hand capsules with mesh-derived OBBs — the geometry every later study measures; #117 fixed it properly, bit-identical collision outcomes |
| #111 | world map | Per-pixel `mj_ray` instead of `mj_multiRay` (whose BVH cull skips visual-only geoms): 947 mm median error fixed, at ~1.9× cost — 6.0 / 18.6 / 55.6 ms per frame at 4 / 301 / 1201 geoms |
| #119 | observability | ADR-0096 latched `SafetyStatus`, implemented twice (C++ kernel and `SafetyPassthroughNode`) — a permanent double-maintenance tax |
| #130 | evidence | Ground-truth E-stop snapshot + `mj_geomDistance` probes — built the whole adjudication corpus on an instrument #170 later proved unusable (caveat 8) |
| #131 | ADR-0092 | `SupportContactWitness`, payload occupancy clearing, `withheld ⊆ exempt` partition; 53,872-probe invariant sweep; 12.5 mm discrimination floor accepted |
| #132 / #133 | ADR-0097 | Place-phase witness + approach allowance; produced the corpus's only two full task completions — caveat 1 says that acceptance was never reproduced |
| #135 | world map | Octree→grid by cube overlap instead of centre sampling: 7.9–37× cheaper, but bought +½ cell of forward reach — 48 % of live stops (#171), undone by #178 |
| #138 | evidence | Created the validation-evidence ledger after four documented claims proved wrong — the single most useful artifact of the effort |
| #139 / #154 | validation | `Kitchen._load_model` redraws layout on every reset, so `seed: 1` does not identify a kitchen; ~85 % of this task's kitchens have the defect. #154 falsified #139's diagnosis; #171 then falsified #154's pin |
| #142 / #146 | ADR-0097 | The ADR-0097 machinery had been inert since it merged (`place_allowance_active` logged zero times, ever); 2 of 3 declared targets were guesses and both wrong, and #142's documented lookup procedure does not exist |
| #143 / #186 | Nav2 | The payload footprint publisher was built, never CI-verified (#183), never run against a translating base (#185), deleted six days later — the payload rides 0.28 m above the costmap slice. Payload avoidance becomes a 3-D E-stop instead of an avoidance |
| #144 | evidence | `evidence_voxel_backing` + `adjudication_budget` (corner slop 28.3–88.2 mm). Caveat 6: **every `false-positive` verdict before this PR is withdrawn** |
| #145 / #149 | harness | The versioned validation matrix; its first live round died in <1 s on a nonexistent flag and reported exit 0, its second produced 24 runs with empty monitor logs. Caveat 5: **every `real-contact` verdict before 08-23 is withdrawn** |
| #153 | ACM/self | Refuse a disconnected collision graph; kernel FK now matches each robot's normative source to 1e-16 m; 4 manifests corrected |
| #157 / #158 | link geometry | `base_link` has no collision geometry, so a chassis primitive buys zero and recovers **no** characterised stop; #157's swept-box row falls below a proven Lipschitz lower bound on 6 of 7 links |
| #159 / #160 | evidence | 240-start-state census: `link1`+`link2` = 83 % of stops, `UNEXPLAINED` = 0/72. Self-occupancy settled by measurement (**no**): 13,288/65,536 rays hit the robot unfiltered, **0** survive the self-filter |
| #161 | link geometry | **Perfect link geometry recovers 27/72 worst case**, crossover ~10 mm. Corrects #159, #157 and its own prior 15-axis SAT bound (0.16 mm, not 3–6 mm) |
| #165 | schemas | `CollisionShape` as a real discriminated union — an untagged mapping had silently become a sphere |
| #166 | link geometry | Staged 26-DOP → exact hull: OBB over-reported link1/link2 reach by 53.3 / 46.8 mm, hull → 0.00 mm excess, 1.04–1.11× *faster*. Recovers 26 % of stops |
| #169 | ACM/self | Branch-and-bound proof: `link5`↔`link7` genuinely interpenetrates in 914/14641 poses — the committed ACM was **exempting a real check**. No shipped ACM byte changed |
| #170 | instruments | Certified `convex_geom_distance` replaces `mj_geomDistance` on the evidence path. Caveat 8: **no verdict from any probe before 08-25 is citable**; four `0.000 m` readings re-measure at +14.8 … +107.9 mm |
| #171 | world map | Re-pinned against the *kernel* criterion, 61 live captures. **Split: 26 % link-side, 74 % world-side.** The census's "live map ⊆ ideal grid" assumption is false |
| #177 | evidence | Payload pose (`xquat`) in the E-stop record — it could not replay the geometry it adjudicates (14 of 15 battery stops). Closes half the reconstruction gap |
| #178 | world map | `/openral/world_voxels` on the OctoMap's own lattice; removes #135's 29–35 mm median dilation, closes a 124 mm coverage hole and a fail-*open* empty-grid path. Safety-WG reviewer, hazard entry and sign-off all unchecked |
| #179 | graded outcome | `CollisionHit::advisory`: a declared place's own contact refuses the chunk without latching, under five independent bounds. **Fired 0 times in 20 scene runs** |
| #180 | world map | Filter geoms with neither `contype` nor `conaffinity` out of the depth cast; per-ray monotonicity verified over 16,384 rays × 4 scenes. Inverts two of #111's assertions; sim-only term |
| #185 | Nav2 | **`consider_footprint` measured +0.53 ms, not #143's +8.1 ms** — a 15× discrepancy. The whole MPPI loop fits at ~20 % of a 50 ms cycle |

### 2.2 Theme verdicts

**Frame and FK (#101, #103, #110, #117/#129).** Net effect correct; #129's
differential evidence (distance identical to six decimals while the cell moved
exactly 28 voxels = 0.700 m) is the cleanest proof in the corpus. **Done, stop.**

**Exact geometry and instruments (#153, #157–#161, #165, #166, #169, #170).** The
*instrument* half is the highest-value work of the two weeks: #170 and #169 each
found a trusted mechanism wrong in the *unsafe* direction (an ACM exempting a pair
that interpenetrates 48 mm; a probe reporting 0.000 m where truth is +107.9 mm). The
*link-envelope* half is a dead end reached honestly. **Instruments: keep. Link
envelopes: done, stop** — do not extend `tight_geometry` past link1/link2 without a
measurement showing the map term has been removed first. *(Overtaken: true of
start-state stops, wrong for carry-phase stops; `tight_geometry` on
`link3`/`link4`/`link6` has since landed and been measured —
`collision-validation-evidence.md` §6, lever 1.)*

**Voxel grid / octomap fidelity (#111, #135, #150, #151, #178, #180).** Where the
measured yield is: #135 fixed a half-voxel phase error by adding a half-voxel of
dilation, #171 measured that dilation as 48 % of live stops, #178 removed it by
changing the message format. The oriented-grid A/B is the only paired round showing
across-the-board improvement (3.5–6.3× less reported penetration on three link-side
stops), though one round per arm proves direction, not magnitude. **Keep going.**
#173 and #174 are closed; a continuation needs a new issue.

**Place-phase allowance, ADR-0097 (#132, #133, #142, #146, #179).** A scoped,
fail-closed, evidence-gated allowance **armed on three of four scenes, binding on 2
of 15 stops**, whose advisory band fired zero times in 20 runs. §9 point 1 names the
real fix: the declaration "names the target; it does not yet give the kernel its
geometry" — Path B. **Stop extending the margin-reduction mechanism.**

**Evidence and adjudication pipeline (#114, #130, #138, #144, #147, #159, #160,
#171, #175, #177).** Three of the eight standing caveats are withdrawals of verdicts
this same pipeline produced (caveats 5, 6, 8), voiding roughly ten days of output.
It is now sound. **Keep, but freeze the contract.** The remaining zero-cost gap:
record the tripping *horizon configuration*, not just the measured one.

**Validation harness (#139, #145, #148, #149, #154, #164, #183).** Every failure it
fixed was a *verification* failure, not a collision one, and each cost a round:
exit 0 on zero runs, 24 runs with empty monitor logs, master unable to launch at
all, a decorative CI lane gate, a deploy image missing the package under test.
**Done, stop building.**

**ACM / self-collision (#153, #169).** The only theme with no retraction, with one
caveat: #169 changed no manifest, so `panda_link5`↔`panda_link7` is **still
exempted** in `robots/panda_mobile/robot.yaml` *(verified: line 751)* despite being
proven a real, checkable pair. That exemption is the open item this survey leaves on
the ACM. **Done, stop** otherwise.

**Nav2 boundary (#143, #148, #183, #185, #186).** What survives is the lidar
self-filter, the ADR-0040 boundary docs, a `findCircumscribedCost` correction and
`consider_footprint: true`. #186's direction is defensible but strictly less
protective: a payload clipping a tall thin obstacle is now an E-stop rather than an
avoidance. **Stop.**

---

## 3. MoveIt 2

`github.com/moveit/moveit2` `main`, source fetched raw. BSD-3-Clause; Jazzy LTS.

### 3.1 PlanningScene collision checking is mesh-level, FCL by default

Source shows the default detector is FCL, Bullet switchable at runtime
(`planning_scene.cpp` L200); URDF collision **meshes load as real triangle meshes**
into `fcl::BVHModel<OBBRSS>`, octomaps as `fcl::OcTree` (`collision_common.cpp`
L900–922); `checkCollision` = padded robot-vs-world + unpadded self-collision, with
signed distance available via `DistanceRequest::enable_signed_distance`. Caveats
from source: the FCL path is **discrete-only** (`collision_env_fcl.cpp` L340–346);
Bullet has CCD but **no distance queries** and is not thread-safe; a distance query
is a separate, much slower pass.

### 3.2 Attached objects and touch_links

`moveit_msgs/AttachedCollisionObject` carries `link_name`, the object, and
`touch_links` — the links allowed to keep touching the attached body. Source shows
the semantics live in the FCL narrowphase callback (`collision_common.cpp`
L130–183): a ROBOT_LINK vs ROBOT_ATTACHED contact is always allowed if the link is
in the body's `touch_links`; two bodies attached to the same link never collide; the
ACM is checked separately and is **not** modified on attach. Attached bodies are
appended to the robot's FCL object set and checked against both world and robot in
every check (`collision_env_fcl.cpp` L212–240). This is the upstream precedent for
scoping an *intended-contact* pair by naming it, not by exempting a region.

### 3.3 Octomap self-filtering also masks attached objects

Source shows `excludeRobotLinksFromOctree()` (L886),
**`excludeAttachedBodiesFromOctree()`** (L958) and `excludeWorldObjectsFromOctree()`
(L988) register every link, attached-body and world-object shape with the occupancy
updater's `ShapeMask`, re-run on attach/detach; only `OUTSIDE` points become
occupied cells. Upstream equivalent of `openral_octomap_bridge`'s payload clearing —
with the same structural consequence this repo found the hard way: masking the
payload's cells also masks the support-surface cells it rests on.

### 3.4 MoveIt Servo: proximity-based velocity scaling, not a binary stop

A dedicated thread polls padded robot-vs-world and unpadded self-collision with
`request.distance = true` at `collision_check_rate` (default **10 Hz**). From
`moveit_servo/src/collision_monitor.cpp`:

```cpp
// If collision detected scale velocity to 0, else start decelerating exponentially.
// velocity_scale = e ^ k * (collision_distance - threshold)
// k = - ln(0.001) / collision_proximity_threshold
scene_collision_scale = std::exp(scene_velocity_scale_coefficient *
    (scene_collision_result_.distance - servo_params_.scene_collision_proximity_threshold));
collision_velocity_scale_ = std::min(scene_collision_scale, self_collision_scale);
```

Scale is 1.0 at the threshold (defaults: self 0.01 m, scene 0.02 m), decays to 0.001
at distance 0, hard 0.0 only on actual collision. Octomap checking is **off by
default**, and nothing in the docs claims Servo is a certified safety function.

### 3.5 Rate and the standalone-filter pattern

The in-tree pattern for "checkCollision on streamed joint states as a filter" is
Servo's monitor, default 10 Hz, not 100. Published throughput (GSoC Bullet
benchmark, Panda,
[moveit#1427](https://github.com/ros-planning/moveit/issues/1427#issuecomment-514541239)):
boolean self-check ~9 µs (FCL) / ~3.7 µs (Bullet); 100 world meshes ~29 µs clear,
but **~1.25 ms (FCL) with 4 meshes in collision** — plus a full extra pass for
`distance=true`. No official MoveIt octomap-latency benchmark exists.

### 3.6 The contact-rich pattern: phase-scoped ACM edits

The canonical MoveIt answer to "the gripper must touch the object" is two scoped
mechanisms: `touch_links` on attach (§3.2), and MTC's `ModifyPlanningScene` stage,
which mutates the ACM **per planning-scene snapshot for one stage** — allow
(hand,object) → close → attach → lift; on place, allow object↔support-surface, open,
forbid, detach (`modify_planning_scene.cpp` L79–145). **This is ADR-0097's
place-declaration pattern, established upstream.** MTC scopes at plan time where
ADR-0097 scopes at run time with a timeout and a measured-contact gate, and ADR-0097
fails toward *less* permission — the in-repo mechanism is the more conservative.

---

## 4. FCL and coal (ex hpp-fcl)

Both BSD-3-Clause; coal (the 2024 rename of hpp-fcl) is Pinocchio 3's engine.

- **FCL's signed distance is broken in exactly the mm regime**: with BVH mesh models
  `min_distance` returns 0 in penetration even with `enable_signed_distance` set
  (FCL [#221](https://github.com/flexible-collision-library/fcl/issues/221),
  [#574](https://github.com/flexible-collision-library/fcl/issues/574),
  [#575](https://github.com/flexible-collision-library/fcl/issues/575)); box-box can
  hang on a libccd EPA assertion (#578, closed-unfixed). MoveIt inherits this.
- **coal has its own GJK+EPA**, true signed distance, a `security_margin` and a free
  `distance_lower_bound`; tolerance ≥1e-6, comfortably inside 1 mm vs 15 mm
  discrimination *for convex pairs*, at 0.8–2.5 µs per pair near contact (Montaut et
  al., RSS 2022, [arXiv:2205.09663](https://arxiv.org/abs/2205.09663)) — ~140 pairs
  ≈ 0.15–0.5 ms/cycle. No published octree-distance latency.
- **Octrees work directly** — but the answer is distance to occupied *cells*,
  axis-aligned boxes at octree resolution, so **world-side accuracy is still bounded
  by the voxel size**, exactly as the in-tree grid is.
- **Self-collision exists upstream**: Pinocchio 3's `addAllCollisionPairs` →
  `srdf::removeCollisionPairs` (the same SRDF format the in-tree ACM is generated
  from) → `computeDistances` per active pair.

The kernel already has an in-house convex-hull GJK with certified conservative
fallbacks and the HAL a certified separating-axis instrument
(`openral_hal.convex_distance`); coal is what those would have been as a dependency.

### 4.1 Round-2 verification (2026-08-30): coal is not a free swap

- **Adoption is partial** — Pinocchio 4.1.0 depends on `coal` and coal 3.0.3 is
  buildfarm-released in Jazzy, but **MoveIt 2 has not migrated** (`moveit_core`
  still depends on `libfcl-dev`; a search of moveit2 issues+PRs for "coal" returns
  zero results — a genuine negative).
- **Open accuracy issues in exactly the near-contact regime**:
  [#823](https://github.com/coal-library/coal/issues/823) witness points go wrong
  once meshes penetrate; [#755](https://github.com/coal-library/coal/issues/755) a
  `security_margin` check silently misses;
  [#636](https://github.com/coal-library/coal/issues/636) NaN contacts from
  mesh-shape traversal. All open.
- **The fixes this survey leaned on are unreleased** — the GJK/EPA-NaN and
  octree-octree fixes sit under `[Unreleased]` in `CHANGELOG.md@devel`; latest tag
  is 3.0.4, so pinning `coal>=3.0` does not get them.
- **Performance is not uniformly better**: an independent migration benchmark
  ([#857](https://github.com/coal-library/coal/issues/857)) measures coal's
  broadphase ~1.8× *slower* than FCL 0.6.1, and
  [#649](https://github.com/coal-library/coal/issues/649) reports **EPA allocations
  on the hot path** — in direct tension with the kernel's no-alloc rule.

**Verdict:** keeping the in-house certified GJK is *lower-risk today* than adopting
coal for the near-contact narrow phase. coal stays a watch item; re-evaluate when
the `[Unreleased]` fixes ship in a tagged release.

---

## 5. cuRobo + nvblox + Isaac ROS cuMotion

The repo already emits cuRobo collision spheres from the kernel's own lowered
geometry (`packages/openral_safety/openral_safety/cumotion_config.py`).

- **Usable as a pure checker, batched**: `RobotWorld` exposes
  `get_world_self_collision_distance_from_joints(q)` on `(batch, dof)` and a
  trajectory variant on `(batch, horizon, dof)`; docs claim collision-checked IK at
  0.05–0.11 ms/query (RTX 6000 Ada, batch 2000), so a whole chunk horizon fits.
- **But the distance is clamped**: positive only inside `activation_distance`,
  **zero beyond it** — fine for a margin gate, no true clearance for graded scaling.
- **Self-collision is spheres, cm-conservative**: `franka.yml` carries **61
  collision spheres** with per-link buffers. Not a route to mm fidelity.
- **nvblox ESDF is voxel-bounded**: default `voxel_size` = 0.05 m — **coarser than
  the in-tree 25 mm grid**.
- **The robot segmenter is real and directly relevant**: it FKs the sphere model and
  masks depth pixels within sphere radius + `distance_threshold`, and the
  object-attachment node adds a grasped object's spheres to the robot model and
  clears its voxels from the nvblox SDF — the GPU equivalent of the depth
  self-filter + ADR-0092 payload clearing already in-tree.
- **Licenses, one trap**: cuRobo relicensed to **Apache-2.0** with V2 (source
  shows), so "cuRobo is non-commercial" is stale, and nvblox is Apache-2.0. **But
  `nvblox_torch` — the cuRobo↔nvblox bridge — is still NVIDIA non-commercial**
  ("research or evaluation purposes only"), gating the depth-camera BLOX path.

---

## 6. Safety filters for learned policies — the field

- **No flagship VLA stack ships any collision layer.** Verified by cloning and
  grepping at HEAD (2026-08-30): `openpi` has zero hits for
  collision/e-stop/deadman/safety — the DROID example's only action processing is
  `np.clip(action, -1, 1)`; `octo` has none; `Isaac-GR00T` ships *advice*, not code;
  `lerobot` is the most careful and is still purely kinematic bounds; a code search
  of `IsaacLab` for "safety filter" returns zero results. The de-facto contact
  safety under research Franka deployments is the arm's own reflex layer
  (`Robot::setCollisionBehavior`). **The in-tree kernel is ahead of the field here,
  not behind it.**
- **The academic framing is projection and speed modulation, not binary stops.**
  ATACOM ([arXiv:2404.09080](https://arxiv.org/abs/2404.09080)) projects any action
  onto a constraint manifold's tangent space, and its 2025 follow-up
  ([arXiv:2505.10219](https://arxiv.org/abs/2505.10219)) applies it as a safety
  layer *after a foundation policy*. For chunked policies, PACS
  ([arXiv:2511.06385](https://arxiv.org/abs/2511.06385)) verifies the whole
  predicted chunk by set-based reachability and brakes along the policy's own path,
  reporting up to 68 % higher task success than reactive CBF filtering.
- **The intended-contact problem has two published answers.** Geometric: exempt the
  intended target and check everything else — the attention-guided VLA filter
  ([arXiv:2606.09749](https://arxiv.org/abs/2606.09749)) derives the target from the
  model's own attention, and ADR-0097 is the same shape with an explicit
  declaration. Force/energy-based: **ISO/TS 15066** power-and-force limiting bounds
  contact force/pressure rather than forbidding contact, and SARA-shield
  ([arXiv:2412.10180](https://arxiv.org/abs/2412.10180)) applies exactly this to
  *learned* manipulation. Near contact, force is the observable that discriminates a
  1 mm touch from a crush; geometry at any voxel resolution is not.

### 6.1 Round-2 correction: the layer now exists in the literature (Dec 2025 – Jun 2026)

The first pass said no established collision layer for VLAs exists. True of vendor
stacks, **now false of the literature** — four groups published external safety
layers over *frozen* VLAs within seven months, three with code, converging on the
in-tree kernel's shape: **VLSA / AEGIS**
([arXiv:2512.11891](https://arxiv.org/abs/2512.11891), code `THU-RCSCT/vlsa-aegis`),
a plug-and-play CBF-QP reporting **+59.16 % obstacle avoidance *with* +17.25 % task
success** on SafeLIBERO; the **attention-guided safety filter**
([arXiv:2606.09749](https://arxiv.org/abs/2606.09749)), training-free, whose stated
motivation — VLM-inferred obstacle identification is *too slow for the control loop*
— independently endorses ADR-0097's explicit declaration over inferred targets;
**SafeVLA** ([arXiv:2503.03480](https://arxiv.org/abs/2503.03480), NeurIPS 2025,
code), the training-time lane at +83.58 % safety / +3.85 % performance; and **PACT**
([arXiv:2606.08414](https://arxiv.org/abs/2606.08414), ICML 2026), post-training
constraint projection at −31.0 % violations / +30.7 % success. Three benchmarks now
measure VLA safety violations directly — SafeLIBERO, LIBERO-Safety
([arXiv:2606.23686](https://arxiv.org/abs/2606.23686)) and ForesightSafety-VLA
([arXiv:2606.27079](https://arxiv.org/abs/2606.27079)) — and runtime *failure*
monitors reading VLA internals (SAFE,
[arXiv:2506.09937](https://arxiv.org/abs/2506.09937)) are the natural producer of
the replanning ladder's hand-off signal. The field's own survey
([arXiv:2604.23775](https://arxiv.org/abs/2604.23775)) still names "unified runtime
safety architectures" an open problem. Corrected claim: **no vendor ships one and no
standard exists, but the layer is actively populated and the in-tree kernel matches
the published shape.**

---

## 7. What the ROS community reports

Threads searched 2026-08-30; the load-bearing discussion lives in GitHub issues and
the legacy ROS Answers / moveit-users archives, not Discourse.

### 7.1 What practitioners hit

- **Octomap doesn't clear cleanly around an object you're about to grasp — a
  12-year-old, still-open MoveIt defect**:
  [moveit_ros#315](https://github.com/moveit/moveit_ros/issues/315) (2013) →
  [moveit#3221](https://github.com/moveit/moveit/issues/3221) (2022, over-clearing,
  open) → [discussion #3613](https://github.com/orgs/moveit/discussions/3613)
  (Nov 2025), where **maintainer rhaschke labels it "a bug with MoveIt's octomap"**.
  Worse at finer resolution.
- **Depth/octomap self-filtering sees the robot's own arm as an obstacle**:
  [moveit#3210](https://github.com/ros-planning/moveit/issues/3210) (2022, open) —
  both updaters tried with padding tuning, no fix.
- **MoveIt Servo's checking is too blunt for contact-rich commands**:
  [moveit2#409](https://github.com/ros-planning/moveit2/issues/409) (2021) — both
  modes slow the robot even while it is *retreating*, or halt outright. Assigned,
  never implemented.
- **Padding tuning is a known pain point, not a precision fix**:
  [moveit_ros#342](https://github.com/moveit/moveit_ros/issues/342) (2013) —
  `padding_offset`/`padding_scale` "are very difficult/impossible to tune". The
  scale parameter was separately inverted for collision objects, and attached
  objects didn't inherit padding at all until
  [PR #2721](https://github.com/moveit/moveit/pull/2721). MoveIt's sampling-based
  ACM generator is "susceptible to omissions"
  ([arXiv:2512.23140](https://arxiv.org/html/2512.23140)).
- **nvblox/cuRobo has the same self-vs-world confusion**:
  [curobo#148](https://github.com/NVlabs/curobo/discussions/148) — an NVIDIA
  maintainer first replies "We currently do not have a way to segment out the
  robot", then points at depth masking; a Sept 2025 comment reports the masking also
  deletes real nearby obstacle voxels. cuRobo's Known Issues page calls
  camera-perception collision-free planning "an open research problem" and flags
  voxel sizes <1 cm hitting GPU memory limits fast.

### 7.2 What practitioners do

**Standard and well-established:** `touch_links` / dynamic ACM entries for the grasp
target, *not* tighter geometry; attach the object to the end-effector link the
instant the grasp closes. **Reported working, with caveats:** Franka's
`setCollisionBehavior`, which distinguishes "contact" (F/T inside a band, robot
keeps moving) from "collision" (reflex trip above it) but is vendor-specific; and
admittance control — PickNik's `ros2_control` admittance controller, built for
"realtime contact tasks such as tool insertion", with CRISP (TU Munich, 2025)
extending it to learned policies, safety posture unverified from its announcement
thread alone. **One person's hack:** raising padding to compensate for quantization.

### 7.3 Directly actionable for this repo

1. **Use `touch_links`/dynamic ACM entries for the specific link-object pair
   expected to contact during a grasp/place phase, instead of shrinking quantization
   error** — gated by skill manifest / reasoner state so it stays an explicit,
   logged exception rather than a blanket disable.
2. **Don't chase octomap-clearing fidelity as the fix** — maintainer-confirmed
   unresolved as of Nov 2025; it is fighting upstream's own unsolved problem.
3. **Adopt a force/torque threshold as the arbiter for "is this a real hazardous
   collision" during contact phases**, separate from the geometric check.
4. **If a compliant-controller layer is in scope**, PickNik's admittance pattern is
   the structural fix for contact-rich E-stops on quantization artifacts.
5. **Treat padding tuning as a low-value lever, not a fix path.**
6. **If moving to nvblox/cuRobo, budget for robot self-segmentation as a required,
   imperfect component** — NVIDIA's own guidance is a workaround with a documented
   2025 regression report.

### 7.4 Dead ends the community already tried

Padding/offset tuning to eliminate false-positive contact detection ("very
difficult/impossible to tune" since 2013, plus an inverted-scale bug and attached
objects not inheriting padding); relying on octomap/self-filter to exclude the robot
or grasp target (chronic and unresolved across 12 years — the community routes
around it with `touch_links`/ACM); MoveIt Servo's built-in checking as a
contact-aware safety layer (binary, directional awareness never implemented); and
geometric depth masking to segment the robot out of nvblox (deletes real nearby
obstacles).

### 7.5 Coverage notes

Discourse yielded little beyond the CRISP announcement; Robotics Stack Exchange
overlapped the archived ROS Answers content already cited. **No thread was found
about a custom safety kernel vetting learned-policy actions pre-execution** — the
exact shape of this repo's problem — so this approach is ahead of what community
venues discuss rather than following an established pattern.

---

## 8. Comparison table

"mm near contact?" asks whether the approach can tell a ~1 mm intended touch from a
~15 mm penetration against a *depth-sensed* world.

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

---

## 9. Assessment

**1. Nothing off-the-shelf clears the voxel wall.** Every surveyed stack that checks
against a *depth-derived* world model is bounded by its cell size: MoveIt's octomap
path (FCL `OcTree` = occupied boxes), coal's octree distance, nvblox's ESDF (0.05 m
default). The 11-of-15 `within-quantization` stops are not a defect a library fixes;
they are what a 25 mm occupancy grid *is* near thin geometry. Where established
stacks reach mm scale is against **known, modeled geometry**. The route to mm
world-side discrimination is therefore not a better checker but a better *world
model* — objects the robot intends to touch promoted from anonymous voxels to posed
meshes/primitives, which ADR-0097's declaration already half-does: it names the
target; it does not yet give the kernel its geometry. *Qualifier:* Tesseract's 2026
implicit-SDF path (§17.3) is the one checker found that consumes a distance field
lazily instead of resampling it onto a grid — the claim stays true of every
*mapper*, but no longer of every *checker*.

**2. The three missing capabilities are all established, and two are cheap.**
(a) *Distance-based speed scaling*: Servo's exponential scale-from-distance is
shipped, source-readable and three lines of math, and PACS independently finds
graded braking beats binary vetoes; the kernel already computes
`min_distance`/`sweep_min_distance` per chunk, so only the graded response is
missing. (b) *Touch-links / attached-object semantics*: MoveIt formalizes what the
repo built ad hoc — `touch_links` ≈ the finger-pair exclusion,
`AttachedCollisionObject` ≈ ADR-0092, `excludeAttachedBodiesFromOctree` ≈ the
octomap bridge's payload clearing, MTC's phase-scoped ACM ≈ ADR-0097. **The bespoke
mechanisms are independently converged re-derivations of the established pattern,
mostly stricter.** (c) *Mesh-level checking*: available, but the
[primitive study §6](collision-primitive-study.md) showed better *primitives*
recover none of the characterised stops and the tight-geometry census puts the
ceiling for *perfect* link geometry at 27 of 72 states
([tight geometry §7.3](collision-tight-geometry.md)) — the binding term is the map,
so mesh-level matters mainly for self-collision (the `panda_link5`↔`panda_link7`
pair exempted "under protest" because boxes false-positive on 79.6 % of the
interpenetration band).

**3. The verdict-for-a-predicted-config evidence gap is in-house and no library
touches it.** Run 1's −5.34 mm verdict against links +53 mm apart at the recorded
joints is the artifact not storing the horizon configuration the sweep tripped on;
Run 2's voxel resolving outside the grid's own coverage is an evidence-decode or
indexing defect. Both persist unchanged under MoveIt, coal or cuRobo. Fix the
recorder regardless of any architecture decision.

**4. On intended contact, geometry runs out and the field's answer is force.** At
the moment of a grasp the true clearance *is* ~0 mm; no geometric margin at any
resolution can pass "touch the drawer handle" while stopping "crush the drawer
handle". ISO/TS 15066 power-and-force limiting is the industry answer for permitted
contact, Franka's reflex thresholds the shipping implementation, and SARA-shield
applies it to learned policies. ADR-0097 scopes *where* contact is allowed; a force
bound is what would say *how much* — the axis the stack does not have at all.

**5. What real VLA deployments do: nothing — but the literature has caught up.**
openpi, Octo, GR00T and LeRobot ship clipping, joint bounds and an e-stop key;
CRISP, the one ROS 2 learned-policy controller stack, ships **zero**
collision/contact safety (verified from source, §12). But §6.1 stands.

**Bottom line.** The hand-rolled kernel is the right *shape*. What it got wrong is
not buildable-vs-buyable but two policy choices established stacks made differently:
a **binary stop where the field uses graded slowdown**, and a **voxel-only world
model where mm-scale work needs the intended-contact target as modeled geometry (or
a force signal)**. Both are adoptable as bounded changes.

---

## 10. Candidate paths

All three keep the E-stop latch and manual `estop_reset`, the topic contract, the
OTel evidence trail, the allocation-free hot path, ADR-0092/0097 machinery and the
ACM, and all three start with the zero-cost fix: **record the tripping horizon
configuration (and payload pose, #172) in the stop evidence**. *All three shipped;
what each did and did not deliver is in `collision-validation-evidence.md` §6.*

### Path A — graded response inside the existing kernel (recommended first)

Adopt Servo's shape, not Servo: compute a velocity scale
`exp(k·(min_distance − threshold))` from the sweep the kernel already runs and scale
the outgoing chunk (or shrink the accepted horizon) in the band between
`margin + proximity_threshold` and `margin`; the latch at true penetration is
untouched. This generalises the #176/#179 advisory band (payload-only, place-only,
depth-capped) into the graded outcome the field uses, with the same severity ladder.
*Effort*: kernel-internal, no new dependencies, one scaling parameter family +
hazard-log entry. *Risk*: scaling a chunk changes what the policy's next observation
sees.

> **Correction (2026-09-04). The "converts the 9-of-15 stops" claim is withdrawn.**
> Those stops are recorded at **−0.29…−11.34 mm**, `hit.min_distance` is the
> reported pair's *true surface distance*, and both `world_collision_margin_m` and
> `world_voxel_margin_m` default to **0.0**. The proposed band is therefore
> `[0, proximity_threshold]` in positive surface distance, and every one of those
> nine stops sits **below** it; they latch exactly as before. What Path A can
> honestly claim is that it slows the *approach* to such a stop. Converting them
> would mean running the graded band into **negative** surface distance — commanding
> motion while two surfaces already interpenetrate by up to ~11 mm — which needs
> Safety-WG sign-off and a hazard-log entry, not a parameter choice.

**Binding precondition: gate the scaling on the action's declared semantics.**
Servo's formula scales a *velocity*, but `ActionChunk`'s `control_mode` mirrors
`openral_core.ControlMode`, which includes **absolute** spaces (`JOINT_POSITION`,
`CARTESIAN_POSE`, `JOINT_TRAJECTORY`, `GRIPPER_POSITION`, `FOOT_PLACEMENT`,
`DEX_HAND_JOINT`). Multiplying an absolute joint-position target by 0.5 commands the
arm halfway to its **zero configuration** — an arbitrarily large motion, toward the
obstacle as readily as away. `ControlModeSemantics.mode` is
`Literal["absolute", "delta"]`; Path A must scale only `delta`/velocity/twist modes
and, for absolute modes, **truncate the horizon or refuse the chunk**. A first
implementation that scales unconditionally is unsafe on most shipped adapters.

**Round-2 evidence (2026-08-30) — the direct experiment exists.** PACS
([arXiv:2511.06385](https://arxiv.org/abs/2511.06385), ICRA 2026) ran this
comparison on robomimic LIFT/CAN/SQUARE with a dynamic obstacle, 100 rollouts each:

| method | safe? | avg success |
|---|---|---:|
| unfiltered policy (same pipeline) | ✗ | 0.70 |
| reactive CBF projection | ✓ | **0.04** |
| single-action braking (SSM) | ✓ | 0.41 |
| **chunk-level graded braking (PACS-PFL)** | ✓ | **0.72** |

Graded slowdown along the policy's own path is *free*; reactive binary filtering is
what destroys the task. Reproduced on real FR3 and on SmolVLA. Two riders transfer:
**scale the whole chunk, not per action** (+28 % in their ablation —
`sweep_min_distance` is already chunk-level); and the published mitigation for the
distribution-shift risk is **observation-side** — PACS excludes velocity from the
policy's observations so slowing is not itself OOD, so **audit every deployed
adapter's observation space for velocity / rate / step-index signals before shipping
Path A**. Cost of the *un*-graded alternative: SARA-shield measures plain SSM at
59.2 %/55.9 % of unrestricted throughput vs 92.6 %/93.7 % for a contact-classified
graded bound. **No manipulation-side chattering/deadlock report was found in six
query formulations — unestablished in either direction**, so a live battery remains
the arbiter. Ecosystem check (§12): **no ROS 2 component ships arm-side
distance-graded scaling to reuse** — Nav2's collision monitor is base-only and
threshold-triggered, PILZ's SSM never left ROS 1, ros2_controllers has nothing
SSM-shaped.

*Outcome: shipped, and the graded slowdown did not reproduce — the robot travels
45 % further with completion unchanged (#209 and both #204 arms).*

### Path B — promote the declared target to modeled geometry

The mm problem is confined to the object the robot intends to touch; ADR-0097
already names it. Extend the declaration's producer to ship the target's geometry
(sim: the declared body's meshes/primitives, the same subtree it already measures
the `PlaceRegion` box from; real: a fitted primitive from perception), and have the
kernel check link hulls / payload primitives against *that* at mesh resolution — its
staged GJK narrow phase already does this query class against cells — while the
voxel grid keeps covering everything undeclared. The narrow-phase engine stays the
in-house certified GJK, per §4.1's coal downgrade. This is the touch-links /
`AttachedCollisionObject` semantics, arrived at via the declaration the repo already
has. *Effort*: moderate — a geometry field on the declaration path (dispatch → HAL →
World State → kernel), kernel ingest + bounds checks mirroring `ingest_place_region`,
safety-WG review.

*Outcome: shipped as ADR-0098, and the premise was wrong — the mm problem is the
carried object against everything it passes, not the declared target. Path B
modelled the target; 56 stops were the payload against the rest of the kitchen. See
`collision-validation-evidence.md` §6 and lever 2 of its lever table.*

### Path C — force-based contact gating for the declared phase

Add the missing axis: during a live ADR-0097 declaration, gate the intended-contact
region by measured force/energy rather than geometry — sim: MuJoCo contact forces the
HAL already reads; real Franka: the reflex thresholds (`setCollisionBehavior`) plus
external-torque estimates. Geometry remains the authority everywhere undeclared.
*Effort*: largest — a new sensor contract through Layer 1/2 into the kernel, a
sim/real seam, and force-threshold calibration with its own hazard entries. Do it
after A, and only if A + B leave contact-phase stops on the table.

**Round-2 evidence (2026-08-30) — the numbers exist, the maturity does not.** ISO/TS
15066:2016 Table A.2 (read from the standard): hands/fingers **140 N quasi-static,
300 N/cm² peak pressure, ×2 transient**; Table A.3 body model K = 75 N/mm,
m_H = 0.6 kg. §A.3.4 states the standard's own actuation knob for a force bound *is
robot velocity* — **A and C are one lever seen from two ends**, which is why C
composes after A. Hazard entries should cite **ISO 10218-2:2025**, which absorbed
the PFL content, alongside TS 15066 for the Annex A tables. Realised budgets vary by
an order of magnitude for the same body region — SARA-shield uses 0.49 J where
PACS's HANDOVER task uses 0.014 J — so **the threshold must be a per-declaration
field, not a config default**. Nearest published instances: CompliantVLA-adaptor
([arXiv:2601.15541](https://arxiv.org/abs/2601.15541), F/T-bounded impedance around
frozen RDT/π0.5/OpenVLA-OFT, code) and FORGE
([arXiv:2408.04587](https://arxiv.org/abs/2408.04587), force-threshold-conditioned
assembly, >1000 real trials, 15 N snap-fit), though FORGE conditions the policy
rather than gating it externally. Two caveats: SARA-shield is one lab, one arm, no
repo, no ROS 2; and **no paper validates MuJoCo contact-force *magnitude* against
real F/T sensors** (§21.7) — FORGE re-tunes on hardware, which is itself the
finding. The sim side must ship a logged calibration parameter mapping MuJoCo force
to nominal Newtons, defaulted conservatively — a knob, not an equivalence claim.

*Outcome: shipped as ADR-0100's declaration-scoped contact-force gate, blocked on
hardware force calibration and dead in sim by design (#218).*

**Not recommended**: adopting MoveIt PlanningScene wholesale as the runtime checker
(10 Hz-class with distance + octomap, discrete-only FCL, replacing
better-instrumented machinery to gain semantics Paths A/B add piecemeal), or the
cuRobo/nvblox stack as the safety path (cm-scale by construction; its right role is
plan-time checking and the depth-stream robot segmenter, noting the `nvblox_torch`
non-commercial gate).

### Code map — where each addition lands (verified against `master`, 2026-08-30)

**Zero-cost evidence fix (all paths).** The `report` lambda
(`lifecycle_kernel.cpp:780`) names every hit; capture `q_check_` and `q_predict_`.
`CollisionEvidence` (`schemas.py:10056`) grows the joint row — there is no IDL
mirror, it crosses the wire as serialized JSON inside `FailureTrigger.evidence_json`,
so the schema change is the whole wire change. Recorder:
`sim.estop_ground_truth_snapshot` (`openral_hal/sim_sensor_bridge.py:1377`).

**Path A.** The scale/shrink seam is `safe_pub_->publish(*msg)`
(`lifecycle_kernel.cpp:1057`); the distance to grade on is folded per pair by
`fold_pair` (`collision.cpp:949`) into `sweep_min_distance`. Generalise rather than
rebuild the advisory band's non-latching path — `CollisionHit::advisory`
(`collision.hpp:431`), `place_advisory_depth` (`collision.hpp:290–293`), the
consecutive-refusal cap (`lifecycle_kernel.cpp:243`, `:793`, `:1399`).
**Precondition audit, preliminary:** grepping
`python/sim/src/openral_sim/policies/` for velocity terms finds hits in exactly one
adapter — `xvla.py:203` feeds gripper `qvel` and `:207` arm `joints["vel"]`; XR-1 /
π0.5 / the state assemblers are position-only. Step-index and elapsed-time signals
are unaudited, and XVLA must be excluded from Path A or re-analyzed first.

**Path B.** `PlaceDeclaration` (`schemas.py:2592`) + its IDL grow the geometry
field; the wire for the measured region already exists (`PlaceRegion.msg`). Kernel
ingest mirrors `ingest_place_region` (`collision.cpp:1734`, `collision.hpp:877`,
called from `lifecycle_kernel.cpp:2005`) — same bounds-checked shape (finite,
positive, ≤ 1.5 m per side, ≤ 8 m³). Narrow phase to reuse: the staged 26-DOP →
exact-hull check in `collision.cpp`, pointed at the declared body's primitives.

**Path C.** Two seams exist and are *dormant*.
`SafetyEnvelope.contact_force_threshold_n` (`schemas.py:897`, default 30.0 N) is
loaded, min-folded by `envelope_loader.py:394` and plumbed into the C++ envelope
(`lifecycle_kernel.cpp:143`, `envelope.hpp:70`, `envelope.cpp:80`) — but **no check
ever reads it**; wiring that last hop is the core of Path C. (The 30.0 N default
predates the ISO reading above and should be re-derived per-declaration.)
`WorldState.contact_forces` (`schemas.py:3323`) is declared with **zero producers
and zero consumers**. The sim force source (`mj_contactForce` plus the calibration
knob) belongs in `openral_hal/sim_sensor_bridge.py`.

---

## 11. What to remove, replace, or stop

Component-level list; whole-theme "stop" verdicts are in §2.2. Locations re-verified
against `master` 2026-08-30.

**Stop investing in:** octomap-clearing fidelity as the false-positive fix
(12-year-unsolved upstream, maintainer-confirmed Nov 2025 — route around it with
declared/attached geometry); extending `tight_geometry` past link1/link2 (27/72
ceiling, the map term dominates); extending the ADR-0097 margin-reduction mechanism
(binding on 2/15 stops); further harness building; further Nav2-side payload
modelling; and **padding / offset-style tuning knobs anywhere** — the community's
verdict is "very difficult/impossible to tune", unchanged since
[moveit_ros#342](https://github.com/moveit/moveit_ros/issues/342) in 2013, and
nobody has since shown the knob resolving quantization-vs-contact ambiguity.

1. **Per-pixel `mj_ray` depth cast (#111) — replace with batched `mj_multiRay`, now
   that #180 has landed.** `_cast_depth_rays`
   (`python/sim/src/openral_sim/backends/depth_camera.py:118`); its entire
   justification was that `mj_multiRay`'s body-BVH cull skips visual-only geoms,
   which #180 now filters out before the cast. The premium is measured: 6.0 / 18.6 /
   55.6 ms per frame at 4 / 301 / 1201 geoms, against a 5 Hz budget.
   `synthesize_laser_scan_2d` casts per-beam for a different reason and must not be
   touched. *Risk: medium* — "the cull only ever mis-skipped non-collidable geoms" is
   stated in #180 but not proved there, and the failure direction is unsafe (missing
   geometry), so it needs the same per-ray exhaustive comparison #180 ran (16,384
   rays × 4 scenes) first.

2. **`range_min_m: 0.55` on `panda_mobile`'s `base_scan`
   (`robots/panda_mobile/robot.yaml:471`) — remove or reduce to the sensor's real
   minimum.** #143 documented it as "a blunt radial cutoff that deletes every real
   obstacle inside 0.55 m in every direction, to hide a chassis whose circumscribed
   radius is 0.43 m", then shipped `payload_scan_filter_node`, which removes
   self-returns by proving they lie inside the manifest's own chassis polygon,
   fail-closed. The polygon filter supersedes the cutoff, which was never lowered.
   The file contradicts itself — the `front_depth` comment (`:487–504`) says
   `range_min_m` is **not** the self-filter while the `base_scan` comment
   (`:446–448`) says it is — so resolve both. *Risk: low-to-medium, in the safe
   direction*, but it surfaces chassis returns wherever the polygon filter is not
   running; land it with a live round on the #185 carry scene.

3. **`CollisionHit::advisory` / the place advisory band (#176, shipped by #179) —
   replace with Path A.** Measured inert: zero firings across 20 scene runs, while 9
   of 15 stops are a link against world occupancy at −0.29 to −11.34 mm, entirely
   inside map discretisation and entirely outside this mechanism's scope. *Risk: **do
   not delete outright.*** The non-latching-drop path and the severity ranking are
   the two hard parts of Path A and #179 already built and tested them; widen the
   five bounds instead. The band has never had its safety-WG reviewer, hazard entry
   or sign-off, so any change re-enters that queue regardless, and
   `place_advisory_max_consecutive: 0` is the documented exact rollback.

4. **`mj_geomDistance` on the support-contact attestation path
   (`openral_hal/_sim_attachment_evidence.py:649`, inside `_probe_support_hits`) —
   replace with `openral_hal.convex_distance.convex_geom_distance`.** #170
   established that `mj_geomDistance` returns confidently wrong values for
   RoboCasa-fixture-vs-`panda_mobile`-mesh pairs in two silent modes and withdrew
   every verdict resting on it (caveat 8); the witness path was not converted. Unlike
   the evidence path this one is a **safety input** — an attestation earns an
   exemption in the kernel — and #170's measured failure direction is *toward
   closer*, which here means attesting a contact that is not there. *Risk: low to
   convert; the finding matters more than the fix.* Two honest caveats: the probe
   runs at a 1 mm window rather than the 0.1 m windows #170 characterised, and #170
   measured 1101 of 1102 pairs agreeing at the shipped windows. But #170 is explicit
   that "no `distmax`, no 'only trust it below N mm' rule and no `ncon` cross-check
   separates the good answers from the bad" — precisely the reasoning a narrow window
   would rely on.

5. **`SafetyPassthroughNode` (`packages/openral_safety/.../supervisor_node.py`) —
   reduce to a test double, or retire.** Every ADR-0096-class change is implemented
   twice, and #138 recorded three further hand-synchronised mirrors. *Risk: high, and
   this is the weakest of the five.* 20 files depend on it, including four live-ROS
   tests that use it *because* it is not the kernel — real-component tests by design
   (CLAUDE.md §1.11) — and the kernel is not always buildable on a dev host. Flagged
   as a standing maintenance tax rather than proposed for removal.

**Not recommended for removal, against first appearances:** `tight_geometry` / the
26-DOP+hull path (#166), measurably faster than what it replaced with definitional
containment; the SAM 2.1 vision-attachment leg (#134), opt-in, default-off, the only
real-hardware attachment path; and `_footprint_geometry.py`, since
`payload_scan_filter_node.py` still imports `base_footprint_polygon` after #186
removed the publisher, and that function calls `convex_hull_2d` internally.

---

## 12. Round-2 verification (2026-08-30) — eight questions, eight verdicts

Detailed evidence is folded into §4.1, §6.1 and §10; this is the scorecard.

| # | question | verdict | strongest evidence |
|---|---|---|---|
| Q1 | Does graded slowdown rescue task completion? | **Strengthens Path A** — with a binding precondition | PACS: 0.04 (reactive CBF) → 0.72 (chunk-graded) vs 0.70 unfiltered ([2511.06385](https://arxiv.org/abs/2511.06385)) |
| Q2 | Force-gating thresholds and maturity | **Strengthens the axis, weakens maturity** | ISO/TS 15066 Table A.2: 140 N / 300 N/cm² / ×2 for hands; budgets vary 0.014–0.49 J per task |
| Q3 | Any VLA-specific safety layers? | **Corrects §6** — field is no longer empty | VLSA/AEGIS: +59.16 % avoidance *and* +17.25 % success, code public ([2512.11891](https://arxiv.org/abs/2512.11891)) |
| G1 | Is coal a lower-risk narrow phase? | **No — downgraded to watch item** | Near-penetration issues open (#823/#755), fixes unreleased at tag 3.0.4 (§4.1) |
| G2 | Reusable ROS 2 arm-side SSM? | **None exists** — hand-rolling confirmed | Nav2 monitor base-only + threshold-triggered; PILZ SSM never left ROS 1; ros2_controllers has nothing |
| G3 | The 15× `consider_footprint` discrepancy (#143 vs #185) | **Mechanism found, number not** | `CostCritic::score()` `continue`s on free-space points (cost < 1) before `inCollision` is reached; #143's microbenchmark assumed all 56 000 points reach the call. Quantifying needs live call-count instrumentation |
| G4 | Does CRISP (or any ROS 2 learned-policy stack) ship safety? | **No** — verified from source | `learnsyslab/crisp_controllers`: soft joint-limit repulsion + torque-rate slew only; `setCollisionBehavior` is never called |
| G5 | Any REP/standard to align with? | **None** | Full REP index read; only REP-2006 (cyber disclosure) is safety-adjacent |

**Coverage honesty:** the ROS Discord has no public archive — genuinely unreachable.
Reddit r/ROS and r/robotics: nothing relevant. The `pwc` catalog 404'd six real
arXiv papers found via web search — **an absence claim must never rest on `pwc`
alone**. The queries behind the absence claims, so they can be re-run: "speed and
separation monitoring human robot collaboration productivity"; "ISO/TS 15066 power
and force limiting collaborative robot" (pwc does not index standards work); "safety
filter vision-language-action policy"; "safety filter causes distribution shift
degrades learned policy performance"; "control barrier function deadlock oscillation
near obstacle boundary"; "freezing robot problem conservative safety overly cautious
task failure"; "force torque contact detection distinguish intended contact from
collision insertion"; "learned manipulation policy force threshold monitor
contact-rich peg-in-hole safety"; "MuJoCo simulated contact force fidelity
sim-to-real contact-rich manipulation". All run 2026-08-30.

---

*Sections 13–23 are the second research pass (2026-08-30) over methods outside
§§3–7's shortlist. Every "adopt" is a candidate with a stated cost, not a decision.*

## 13. What the kernel actually needs, restated as test criteria

The kernel's two slop sources are additive and both **structural**, not tuning
errors: **world-side quantisation** (a 25 mm grid, half-diagonal 21.7 mm; a ~1 mm
real contact reads as up to ~15 mm of cube penetration) and **robot-side
over-approximation** (per-link OBB corner slop of 27–76 mm protrusion,
[primitive study §4.2](collision-primitive-study.md)). Every method below is scored
on three questions, and one that fixes only one fixes only half the problem:
**(a) mm near-contact** — can it tell a ~1 mm intended touch from a real penetration,
*against a sensed world*; **(b) dynamic worlds** — moving obstacles, carried
payloads, base + arm; **(c) self-collision** — link-vs-link, better than OBB.

Two constraints bound every "adopt": the hot path is **allocation-free C++** at
30–200 Hz (CLAUDE.md §1.5, §2), and new dependencies must be Apache-2.0 / MIT / BSD
(CLAUDE.md §4.4) — a GPU-only or Python-only method cannot decide whether motors
stay energised. The kernel already exposes `WorldModel`, `VoxelGrid`,
`check_voxel_collision`, `hull_cell_distance` and an `AttachedModel`, fed from
`/openral/world_voxels`, so a different world representation is a **sibling of
`VoxelGrid`**, not a rewrite.

---

## 14. Learned and analytic distance fields for the robot body

This direction attacks the OBB slop term.

### 14.1 RDF — robot geometry as a Bernstein-polynomial distance field

Each link's SDF is a product of Bernstein basis functions composed along the chain by
FK — a closed-form basis expansion, not an MLP, which is why the model is 24 KB.
ICRA 2024, <https://arxiv.org/abs/2307.00533>;
<https://github.com/yimingli1998/RDF>, **MIT** (*source shows*). *Docs claim*
whole-body MAE **1.41 mm** at N=24 and, unusually, **better near the surface**
(1.71 mm near / 1.18 mm far); the same table puts spheres at 5.91 mm and Neural-JSDF
at 23.0 mm. Query 0.21–0.54 ms on an RTX 3060; no C++, no ROS 2. **(a)** yes at
1.4 mm on the robot side; **(b)** no; **(c)** partially. **Verdict: adopt as a body
model, but you write the port** — 1.4 mm against 27–76 mm of OBB corner slop is the
right order of magnitude and a fixed-order polynomial sum survives an
allocation-free C++ rewrite, but you inherit nothing runnable and a learned field is
not conservative (§14.6).

### 14.2 CDF — configuration-space distance fields

Distance in *joint* space, so projection/IK is one gradient step. RSS 2024,
<https://arxiv.org/abs/2406.01137>; <https://github.com/idiap/cdf>, **MIT**. *Docs
claim* MAE **1.39–1.64 cm** on Franka — an order of magnitude worse than RDF, because
CDF optimises gradient consistency, not surface precision. **Verdict: watch, do not
adopt.** Its win is planning/IK throughput; the kernel's problem is millimetres.

### 14.3 Neural-JSDF

<https://github.com/epfl-lasa/Neural-JSDF> (RA-L 2022). **No licence file at all** —
*source shows*, the GitHub licence endpoint returns 404, so all-rights-reserved, a
hard blocker under CLAUDE.md §1.9. Accuracy is 23.0 mm MAE per RDF's table.
**Verdict: irrelevant** — unlicensed, superseded by RDF.

### 14.4 Learned body SDF + SVM self-collision (Zhu et al. 2024)

<https://arxiv.org/abs/2409.14955> — parallel tiny MLPs for the robot SDF **plus SVMs
specifically for self-collision**, architecturally the closest match to what the
kernel needs. The abstract gives **no error figure** (a "0.19 cm RMSE" figure in
search snippets is not in the paper — **unverified**) and there is no code.
**Verdict: watch.**

### 14.5 ReDSDF — regularised deep SDFs

<https://arxiv.org/abs/2203.04739> (IROS 2022). The contribution is far-field
regularisation: a vanilla neural SDF degrades outside its training shell, exactly
where a safety margin lives. No official code; two same-named repos with no licence.
**Verdict: watch (concept only).**

### 14.6 The honest constraint on all of the above — Neural Implicit Swept Volumes

<https://arxiv.org/abs/2402.15281> (ICRA 2024, KUKA), a neural implicit SDF of a
motion's *swept volume*. *Docs claim* MAE **3.08–4.38 mm**, 93.1 % classification at
a 5 mm margin, **false negatives 0.6 %**. The load-bearing part is what the authors
do next: they state the network **cannot be guaranteed conservative** and pair it
with a geometric checker to restore the guarantee — net speedup dropping to 25–49 %.
**This is the template any learned field must follow inside this kernel**: the
learned distance is a fast *filter*, the geometric check keeps veto authority. A
0.6 % false-negative rate cannot decide whether motors stay energised
(CLAUDE.md §1.1).

### 14.7 Scene neural SDFs (iSDF and successors) — the negative result

iSDF (<https://arxiv.org/abs/2204.02296>, RSS 2022; **MIT**, archived since 2023-02)
*docs claim* **< 6 cm** SDF error under a static-scene assumption — **worse than the
25 mm grid the kernel already has**. The 2025–26 successors found
(<https://arxiv.org/abs/2511.21312>, <https://arxiv.org/abs/2509.11185>) are
navigation/MPC-flavoured with no mm claim and no verified permissive repo.
**Verdict: irrelevant** — neural *scene* SDF fusion is a navigation accuracy class,
not a near-contact one.

---

## 15. Point clouds instead of voxels — CAPT, and the SIMD planning family

This direction attacks the 25 mm grid.

### 15.1 CAPT — collision-affording point trees

An **exact** representation of a point cloud (no voxelisation, no occupancy
inference) supporting SIMD-parallel sphere-vs-cloud queries: built once per cloud
with a legal radius range, queried as `collides(center, radius) -> bool`. Ramsey,
Kingston, Thomason, Kavraki, RSS 2024 — <https://arxiv.org/abs/2406.02807>; Rust
<https://github.com/KavrakiLab/captree-rs>, C++ inside VAMP as
`src/impl/vamp/collision/capt.hh`. **Apache-2.0** for both (*source shows*). The
query is **boolean**, not a signed distance: the robot is a set of spheres with known
`r_min`/`r_max`, tested against a leaf's *affordance set*, a conservative superset of
points.

**Numbers — *source shows* (RSS PDF).** **9.89 ns** mean per query versus 309 ns for
NanoFLANN and **0.01 ms for an OctoMap backed by FCL**; benchmark clouds up to 50,000
points at **mean dispersion 7 mm – 2.2 cm**; a real sensor run (RealSense D455, UR5)
filters 166,587 raw points to 2,732 at `r_filter = 2 cm`, `r_min = 1.5 cm` and takes
**median 7.166 ms** end-to-end (~140 Hz). **Construction is the bottleneck, not
query**: superlinear, "the lion's share of planning time", though it beats OctoMap
construction on the same data. **Conservativeness — *source shows*:** Lemma V.1, the
tree does not alter the collision status of any query sphere; Lemma V.2, there exists
`r_filter ≤ r_min − dispersion` inserting no gap larger than the minimum sphere
diameter — provably conservative, but only by inflating, which reintroduces cm-scale
margin exactly where the kernel cannot afford it. **ROS 2 maturity: none official**;
the one third-party prototype translates neither MoveIt collision objects, attached
objects nor octomap geometry, and its robot model is a compile-time C++ type.
**(a)** partially, and this is the key nuance — CAPT removes the *voxel* quantisation
entirely, but the robot is still spheres and the filter still pads, so the residual
becomes `sensor dispersion (7 mm–2.2 cm) + r_filter + sphere slop`. **(b)** yes.
**(c)** no.

**Verdict: adopt — the strongest candidate on this page for the world side.** A
`PointCloudWorld` sibling to `VoxelGrid` behind a `check_point_cloud_collision` entry
point shaped like the existing `check_voxel_collision` (`collision.hpp:766`) — not
`check_world_collision`, which consumes a `WorldModel` of base-frame capsules
(`collision.hpp:715`) — fed from the depth `PointCloud2` the HAL already publishes.
Costs any ADR must state: it is boolean not signed-distance (the advisory band and
witness machinery need distance — recoverable by bisecting on radius at ~10 ns per
probe, but that is new code); construction is per-frame and superlinear; and a point
cloud is a *surface sample* with no free-space/unknown distinction, a different
failure mode from a grid, not a strictly smaller one.

### 15.2 VAMP — vectorised sampling-based motion planning

<https://github.com/KavrakiLab/vamp> (**Apache-2.0**, active), *docs claim* median
**35 µs** per Franka MotionBenchMaker problem on one core. Three facts checked, all
*source shows*: **SIMD, not GPU** (`vector/isa/` holds `avx.hh`, `neon.hh`,
`wasm.hh`, no CUDA); **spheres** (`robots/panda.hh` declares `n_spheres = 59`, radii
0.012–0.08 m, so its accuracy floor is the **same order as the kernel's OBB slop**);
**discrete, not continuous** (a SIMD rake of interpolated configurations, not a swept
volume). **Verdict: irrelevant as a planner, adopt its CAPT** — the kernel vets
chunks, it does not plan, and VAMP's robot model is generated C++ per robot.

### 15.3 pRRTC and Foam

pRRTC (<https://arxiv.org/abs/2503.06757>, **Apache-2.0**) is a GPU-parallel
RRT-Connect with the same sphere accuracy floor as VAMP and no CCD. **Irrelevant.**
**Foam** (<https://github.com/CoMMALab/foam>, MIT) converts a URDF into a spherical
approximation — **watch / cheap fallback**, since a fine sphere decomposition is
strictly tighter than an OBB and trivially allocation-free. Sibling repos `batch-cc`
and `SPaSM` are **unlicensed**.

### 15.4 RTCollisionDetection — hardware ray-traced discrete *and* continuous CD

<https://arxiv.org/abs/2409.09918> (ICRA 2025);
<https://github.com/Ssz990220/RTCollisionDetection>, **MIT**, on NVIDIA OptiX. *Docs
claim* mesh-to-mesh discrete CD **and** mesh-to-swept-volume CCD along B-spline
paths; up to 3× (discrete) and 9× (CCD) over GPU sphere baselines at 24k-triangle
robot meshes. No ROS 2. **Verdict: watch, with a hard caveat.** B-spline swept CCD is
precisely the right shape for vetting an action chunk rather than a pose, but it
binds the safety path to OptiX — a closed NVIDIA SDK, which under CLAUDE.md §1.9 sits
behind a licence guard and an env var and rules out a portable aarch64 path — plus
GPU latency jitter on a 200 Hz deadline.

### 15.5 MuJoCo Warp / Newton — explicitly disqualified by its own docs

<https://github.com/google-deepmind/mujoco_warp> (**Apache-2.0**). The collision
pipeline **is** usable standalone (*source shows*: `collision`, `nxn_broadphase`,
`sap_broadphase`, `primitive_narrowphase`, `sdf_narrowphase` are public exports).
Two disqualifiers from the official page: it is "optimized for throughput" not
latency, aimed at RL sampling; and it is **non-deterministic** — "There may be
ordering or small numerical differences between results computed by different
executions of the same code." **Verdict: irrelevant on the vetting path.** A check
the vendor documents as non-deterministic cannot decide whether motors stay
energised.

### 15.6 Other GPU/CCD entries checked

- **Scalable-CCD** (**Apache-2.0**, active) — GPU tight-inclusion CCD for
  deformable/FEM simulation, no articulated FK. **Watch**: its tight-inclusion
  interval arithmetic is the only *provably conservative* CCD primitive in this pass.
- **NeuralSVCD** (<https://arxiv.org/abs/2509.00499>) — **no numbers in the
  abstract** and the linked repo 404'd. **Watch.**
- **RoboGPU** (<https://arxiv.org/abs/2603.01517>) — a proposed GPU *hardware* unit
  testing link OBBs against octree AABBs with a 15-axis SAT, i.e. the kernel's exact
  representation, claiming 3.1×/14.8× over RT/CUDA baselines; no code. **Irrelevant
  for adoption**, but a useful outside signal that OBB-vs-octree SAT is defensible.
- **OMPL GPU forks** — none found.

---

## 16. Dynamic-world mapping — what exists, and why none of it clears the wall

Licences read from each repository's own file, not from a paper.

### 16.1 Dynablox

<https://github.com/ethz-asl/dynablox> (RA-L 2023), **BSD-3-Clause**, *docs claim*
86 % IoU at 17 FPS; output is occupancy plus a *high-confidence free-space* layer, so
anything observed inside known free space is declared dynamic. **ROS 2: no** —
Noetic/catkin throughout; the only ROS 2 port found has **no licence file**, another
fork is GPL-3.0. **Verdict: watch the algorithm, ignore the package.** Addresses (b)
only, at LiDAR scale; its free-space conservatism is actively wrong for near-contact
— it widens the "unknown means obstacle" band — and the idea already ships in nvblox.

### 16.2 wavemap — the right representation, the wrong plumbing

<https://github.com/ethz-asl/wavemap> (RSS 2023), **BSD-3-Clause**. **It does go
below 25 mm**: the shipped `wavemap_livox_mid360_pico_flexx.yaml` sets
`min_cell_width: {meters: 0.01}` for the depth camera, multi-resolution — 1 cm cells
only near surfaces, the memory argument a uniform grid cannot make. **It has a true
Euclidean SDF, unadvertised** (`core/utils/sdf/full_euclidean_sdf_generator.h`) —
**but it is a batch, whole-map, `const` one-shot generator**, no update-in-place API,
a hard blocker at 30–200 Hz. **ROS 2: no** — `main` has `interfaces/ros1/` only, and
the unmerged `feature/ros2` branch's `package.xml` still declares `catkin`, `roscpp`
and `rosbag`. **(a)** at the representation level yes — the only mapper here that
does; **(b)**, **(c)** no. **Verdict: watch.** Adopting it means porting it *and*
writing an incremental SDF: a project, not an integration.

### 16.3 Bonxai — a faster wrong answer

<https://github.com/facontidavide/Bonxai>, active. **Licence correction: MPL-2.0, not
Apache** — *source shows*: file-level weak copyleft, **not on the allowlist**, needs
TSC review; worse, the repo-root `package.xml` declares
`<license>TODO: License declaration</license>`, a provenance defect. Default
resolution `0.02` m with a log-odds sensor model — an octomap clone; *docs claim* 22×
faster creation, but no SDF anywhere in the tree, the README calls it "primarily for
educational purposes" and it is **not thread-safe for writes**. **Verdict:
irrelevant.** It makes the existing representation faster, and speed is not the
reported bottleneck.

### 16.4 vdb_mapping — best ROS 2 hygiene here, wrong output type

<https://github.com/fzi-forschungszentrum-informatik/vdb_mapping> +
`vdb_mapping_ros2`, **Apache-2.0**, and **the only mapper in this section with
verified Jazzy CI** (*source shows*: humble / iron / jazzy / rolling). **A stale-fact
correction that matters beyond this page: OpenVDB relicensed from MPL-2.0 to
Apache-2.0 in v12.0.0 (2024-10-31)** — *source shows*; caveat, Ubuntu 24.04's
`libopenvdb-dev` is 10.x/11.x and therefore **still MPL**, so a rosdep install can
silently pull the copyleft version. Pin ≥ 12. Output is log-odds occupancy and the
ROS 2 wrapper exposes **no SDF**; voxel floor and update rate could **not** be
verified from source. **Verdict: watch.** (b) partially, (a) and (c) no.

### 16.5 nvblox dynamics — production-grade, and pointed the wrong way

**Licence:** GitHub reports `NOASSERTION`, but `LICENSE.md` is **Apache-2.0** plus a
**BSD-3-Clause** block covering voxblox-derived files, and `isaac_ros_nvblox` is
cleanly Apache-2.0. **But the cuRobo↔nvblox bridge is a separate repo,
`NVlabs/nvblox_torch`, whose LICENSE restricts use to "research or evaluation
purposes only"** — *source shows*; §5's non-commercial flag stands.
`dynamics_detection.h`, class docstring verbatim: "A class for detecting dynamic
objects. It takes a depth frame and compares it to the freespace layer. If any
surface seen on the depth image falls into freespace it is assumed to be dynamic."
That is Dynablox ported to CUDA. **From the shipped configs:** `voxel_size: 0.05`
(50 mm), `update_esdf_rate_hz: 10.0`, `esdf_mode: "2d"`,
`occupied_region_half_width_m: 0.15` — **dynamic obstacles are inflated by 150 mm**.
**Verdict: watch — solves (b) properly and makes (a) strictly worse.** The only
production-grade, Apache-2.0, Jazzy, GPU dynamic mapper found, but every default is
navigation-shaped: near a gripper the 15 mm phantom penetration would become
~150 mm. Reasonable for the mobile base's world; keep it out of the manipulation
volume.

### 16.6 The rest, briefly

- **Voxfield** (BSD-3) — non-projective ESDF, ROS 1, **dead since 2023-05**; the
  correction fixes *bias*, not *resolution*. **Irrelevant.**
- **GPU-Voxels** (FZI) — dormant since 2023-08; `LICENSE.txt` states GPU-Voxels and
  `icl_core` are **CDDL**. **Irrelevant — dead and CDDL**, ironic since it is the
  only one originally designed for *manipulator* voxel collision.
- **ROG-Map** — **GPL-3.0**, **rejected on licence**; the EGO-Planner family shares
  that lineage.
- **DSP-Map** (MIT repo but `<license>TODO</license>`, ROS 1 only) — particle-based
  continuous occupancy with **predicted future occupancy**; MAV-scale, no SDF.
  **Irrelevant to adopt**; the future-occupancy idea is worth reading.
- **Sub-cm mapping for manipulation specifically**: nothing released. ParaMaP
  (<https://arxiv.org/abs/2512.22575>) and DB-TSDF
  (<https://arxiv.org/abs/2509.20081>) publish no resolution, rate, repo or licence —
  both **unverified**.
- Independent corroboration of the diagnosis: CADGrasp
  (<https://arxiv.org/abs/2601.15039>) *docs claim* an ablation choosing 5 mm voxels,
  2.5 mm giving "only marginal gains" and 10 mm "a clear performance drop". **10 mm
  is already too coarse for grasp-level geometry** — the kernel is at 25 mm.

**Section verdict.** No 2023–2026 mapper clears the mm wall at 30–200 Hz. The two
with the right ingredients pull in opposite directions: wavemap has the resolution
and a true Euclidean SDF but no ROS 2 and no incremental update; nvblox has the
engineering and the dynamics but 50 mm cells, a 10 Hz 2D ESDF and 150 mm inflation.
**§9's point 1 survives this pass** — with the caveat that §17 finds a way to stop
resampling onto a grid at all.

---

## 17. Tesseract — the one library that beats MoveIt's discrete FCL path

<https://github.com/tesseract-robotics/tesseract>, v0.35.0, active. Multi-licensed
**Apache-2.0 + BSD-2 + BSD-3** per file (*source shows*); every collision and
geometry header read carried the Apache-2.0 block, so GitHub's `NOASSERTION` is an
artefact of the multi-licence root. **Caveat:** `tesseract_gui` and `tesseract_qt`
are **LGPL-3.0** — GUI only, never link.

### 17.1 Continuous collision checking — confirmed in source

`continuous_contact_manager.h` declares `ContinuousContactManager` with
`setCollisionObjectsTransform(id, pose1, pose2)` — a swept hull needs both endpoints.
`bullet_cast_bvh_manager.h` and `bullet_cast_simple_manager.h` implement swept/cast
BVH; **the FCL backend contributes no continuous manager**. Results carry
`cc_time[2]`, `cc_type[2]` and `cc_transform[2]`, with the docs telling you to
interpolate on `cc_time` to locate the contact. **Why this matters here:** at
30–200 Hz a VLA chunk moves the tool tens of millimetres between samples, and a
discrete check tunnels through thin geometry — a failure mode the kernel's staged
26-DOP → GJK pipeline inherits. And `cc_time` says *when* along the chunk the contact
occurs, which is a truncation point rather than an E-stop.

### 17.2 Per-pair margins — the correct shape for the intended-contact problem

*Source shows*, from the manager interface: `setCollisionMarginData`,
`setCollisionMarginPairData`, `setDefaultCollisionMargin`,
`setCollisionMarginPair(id1, id2, margin)`, `incrementCollisionMargin`, plus
`setContactAllowedValidator(...)` — a generalisation of the SRDF ACL. Contacts closer
than the margin are "in collision", order-independent. `ContactResult` carries
`distance` (negative = penetration), `nearest_points` and `nearest_points_local`,
`transform[2]` and a separation normal; `ContactRequest` has
`ContactTestType {FIRST, CLOSEST, ALL, LIMITED}` and `calculate_penetration`.

This is strictly richer than MoveIt's binary ACM plus a single `contact_distance`,
and it is exactly what the kernel lacks: **a negative margin on *one* declared pair**
(gripper finger vs the target object) while every other pair stays conservative.
§1 item 3 — "touch-links / attached-object semantics as a first-class contract" — is
this feature.

### 17.3 New in 2026: implicit-SDF collision and an SDF geometry type

Not in the docs; found by reading the tree. `implicit_sdf_collision_solver.h` (©2026,
Apache-2.0), docstring verbatim: "This adapts MuJoCo's multi-start SDF collision
strategy: deterministic Halton seeds are optimized over the overlap of the shapes'
margin-expanded AABBs using a composite collision objective and backtracking gradient
descent. Contact distance and nearest points are then recovered by projecting the
converged point onto both zero level sets." `geometry/impl/signed_distance_field.h`
is a first-class `SignedDistanceField : public Geometry` with a **lazy,
function-backed** mode (`BatchedSignedDistanceFunction`), whose docstring names the
intended sampler — "e.g. a batched nvblox ESDF GPU query" — and states "No grid is
sampled up front: sampler is the field's source of truth, so queries and each
collision backend evaluate it directly (exact, no resampling)." **That is a door in
the voxel wall.** Two hard constraints, verbatim from that header: "It is concave, so
it is not supported by continuous/cast managers" → **SDF geometry and CCD are
mutually exclusive**; and serialization and equality force `discretize()`. VDB
interop is backed by a **vendored `third_party/tinyvdb`, Apache-2.0**, so there is no
dependency on AcademySoftwareFoundation/openvdb.

### 17.4 Octomap worlds, ROS 2 status, and real-time honesty

**Octomap** is supported (`geometry/impl/octree.h`, `liboctomap-dev` a hard dep) —
but each cell becomes a Bullet box or sphere at the *source* resolution, so **feeding
the kernel's 25 mm octomap to Tesseract reproduces the 15 mm phantom penetration
exactly**; the win only materialises via `SignedDistanceField`. **ROS 2:** core
`tesseract` is ROS-agnostic plain CMake and **no ROS 2 binary exists**
(index.ros.org lists v0.35.0 for **Noetic only**), but `tesseract_ros2`'s CI matrix
builds humble, jazzy and kilted, so Jazzy is genuinely built, source-only — fine for
a kernel that wants to link `libtesseract_collision_bullet`. **Real-time:**
`contact_monitor.h` is a *sensor-rate monitor and visualiser*, not a hard gate — a
discrete manager, a 100 mm default margin, a `condition_variable` driven by incoming
`JointState`. Structurally against CLAUDE.md §2: managers are clone-per-thread,
`contactTest` fills an allocating `ContactResultMap`, and margin setters **throw** —
exceptions across what would be the safety-kernel boundary, which this repo's C++
standard forbids. **No published latency benchmark exists.**

### 17.5 Verdict

**Adopt selectively: link `tesseract_collision` (Bullet cast BVH + per-pair margins)
and `tesseract_geometry::SignedDistanceField`; do not adopt `tesseract_ros2`.**
(a) **partially, and better than anything else in either survey** — per-pair margins
plus the implicit-SDF exact path, though its octomap path is bounded exactly as
today's is. (b) **no** — Tesseract consumes a world; the
`BatchedSignedDistanceFunction` hook is how a mapper would wire in. (c) **yes,
cleanly** — `ContactAllowedValidator` generalises the SRDF ACL, per-pair margins let
tight link pairs be tuned individually, and CCD catches self-collisions discrete
sampling skips between waypoints. The prototype gate is a single number nobody has
published: `contactTest` latency against the kernel's chunk budget, on this repo's
hardware, with an SDF-backed world.

---

## 18. Reactive control layers — five independent stacks, one shared inequality

The strongest convergence in this pass: five independently developed reactive layers
all express collision avoidance as the *same* constraint — a signed distance mapped
to an **allowed velocity along the contact normal**, not a boolean.

| stack | the inequality | licence |
|---|---|---|
| NEO / Robotics Toolbox | `nᵀJ q̇ ≤ ξ(d−dₛ)/(dᵢ−dₛ) + nᵀv_obs` | MIT |
| mink | `nᵀ(J₂−J₁) q̇ ≤ gain·(d−d_min)/dt + relax` | Apache-2.0 |
| pink | CBF on `h = d − d_min` | Apache-2.0 |
| OCS2 | `d − d_min` as an MPC constraint | BSD-3 |
| fabrics | Finsler geometry leaves | **GPL-3.0 — rejected** |

All five need the same two inputs: a signed distance **and the contact normal `n`**,
out to an influence band `dᵢ`.

> **Correction (2026-09-04): the "~15 lines of C++ over the GJK output
> `hull_cell_distance` already produces" estimate is withdrawn.** That routine
> returns a single `double` (`collision.hpp:691`) — a supporting-hyperplane *lower
> bound*. It emits **no contact normal** (`seed_dir` is an *input*), it
> **early-exits** as soon as the bound clears `margin`, and on overlap it returns the
> caller's `fallback` instead of a depth. Supplying both is a change to the narrow
> phase's **output contract**, and it must preserve the conservatism argument that
> routine is built around: every value it can return today is a lower bound, so the
> early exit and the overlap case can only make the answer *more* conservative. A
> normal-carrying variant must inherit that property or it is a regression. The
> certified instrument added by #170/#204 already reports exactly this pair —
> `ConvexDistance.distance_m` plus `ConvexDistance.direction`, the latter exact even
> at a flush contact — but it is offline Python: the right **reference** for a C++
> mirror and the right oracle to test one against, not the hot path.

The formulation dissolves the reported failure mode directly: a grasp at 3 mm gets a
*small allowed approach speed* instead of an E-stop — the same conclusion §3.4
reached from Servo's formula, arrived at independently by five codebases.

### 18.1 mink `CollisionAvoidanceLimit` — the best-verified near-contact geometry

<https://github.com/kevinzakka/mink>, **Apache-2.0** (*source shows*). *Source
shows*, `src/mink/limits/collision_avoidance_limit.py`:

```python
dist = mujoco.mj_geomDistance(model, data, geom1_id, geom2_id, distmax, fromto)
row  = compute_contact_normal_jacobian(...)
if dist > min_dist: upper_bound[idx] = (gain*(dist - min_dist)/dt) + relaxation
else:               upper_bound[idx] = relaxation
sign = -1.0 if dist >= 0 else 1.0     # penetration flips the row
```

Defaults `gain=0.85`, `minimum_distance_from_collisions=0.005`,
`collision_detection_distance=0.01`. **`min_dist` may be negative** — "A negative
distance allows the geoms to penetrate by the specified amount" — a directly usable
per-pair knob for "this grasp is allowed to touch". **Self-collision is
first-class**: `_construct_geom_id_pairs` applies exactly the three filters the SRDF
ACL encodes, i.e. it *derives* an allowed-collision list from the model, and the
broadphase is documented as a **strict** pre-filter. Scratch buffers are
pre-allocated; no published rate; no ROS 2. **(a)** yes — the strongest here;
**(b)** partially, no obstacle-velocity feed-forward; **(c)** yes, including the ACL
equivalent.

**Verdict: adopt as design reference. NOT as an oracle.** ~330 lines of Apache-2.0
specifying what the kernel should return instead of a boolean: mirror the algebra in
C++ over the existing GJK, and keep Python out of the 200 Hz path. The oracle half is
**withdrawn** — mink's distance *is* `mujoco.mj_geomDistance`, which §22.1 and §11
item 4 record as returning confidently wrong values on exactly the fixture-vs-link
pair class such an oracle would target. The oracle that belongs here is
`openral_hal.convex_distance.convex_geom_distance`, which certifies every answer and
refuses rather than guess — and which reports the contact normal mink's damper row
consumes (`ConvexDistance.direction`, exact even at a flush contact, where
differencing two coincident witness points yields nothing).

### 18.2 NEO and the holistic controller (Haviland & Corke)

<https://arxiv.org/abs/2010.08686> (RA-L 2021); `robotics-toolbox-python`, `swift`,
`spatialgeometry` — all **MIT**. *Source shows*, `Robot.py::link_collision_damper`:
the QP row is `norm_h @ Je` bounded by `xi*(d - ds)/(di - ds) + dp`, where
**`dp = norm_h @ shape.v` is the obstacle's own velocity** — the only mechanism found
anywhere in this pass that makes a *moving* obstacle a first-class part of the
constraint rather than a re-plan trigger. Defaults: influence 0.3 m, stop 0.05 m.
**2026 update**: `spatialgeometry` now dispatches to **coal**, so NEO's damper
already rides the same GJK/EPA family the kernel uses. **Honest caveats, both *source
shows*:** the published holistic base+arm examples contain **only**
`joint_velocity_damper` — no obstacles — so NEO's avoidance and the holistic base+arm
QP are demonstrated *separately*, never together; and there is **no self-collision
helper**. Rate "a few ms" per QP; no ROS 2. **Verdict: adopt the formulation, not the
package** — MIT means the code may even be lifted, and the `+ nᵀv_obs` term is the
piece to take for dynamic worlds.

### 18.3 pink `SelfCollisionBarrier`

<https://github.com/stephane-caron/pink>, **Apache-2.0**. The collision work lives in
`barriers/`, not `limits/` — `ConfigurationLimit` is joint-space only, which corrects
a common misattribution. `self_collision_barrier.py` builds a CBF on
`h(q) = d(p¹,p²) − d_min` (default 20 mm) over the N closest pairs from hpp-fcl/coal
via Pinocchio. Docstring caveat, quoted: "Note that for non-smooth collision
geometries behaviour is undefined." Boxes and hulls *are* non-smooth, and the mm
regime is exactly where that bites. **Verdict: watch.** (c) yes, and it is the only
surveyed package whose *primary* product is self-collision; (a) partially; (b) no
velocity term.

### 18.4 Fabrics, RMPflow — rejected on licence and on maintenance

TU Delft `fabrics` is **GPL-3.0** (*source shows*), **disqualified** under
CLAUDE.md §4.4 without TSC review, despite *docs claim* of up to 500 Hz replanning.
Two ideas worth stealing: `ESDFGeometryLeaf`, whose contract is "give me φ, J, J̇ as
external symbols" — a collision term that consumes distance+gradient from someone
else's field rather than re-deriving geometry; and that
`set_self_collision_avoidance()` is **commented out** of the default problem path.
NVIDIA's geometric fabrics have no open implementation — the only shipped one is
inside Isaac Sim's **Lula**, API docs only, **no published licence** ("no licence
found", not "closed"); <https://arxiv.org/abs/2405.02250> is the closest published
framing of *wrapping a learned policy in a medium rather than gating it*. `rmp2`
(MIT) has been dead since 2021. **No maintained, permissively licensed RMPflow exists
in 2026.**

### 18.5 Pinocchio 3 + coal derivatives, and the zero-distance degeneracy

`pinocchio` is **BSD-2** and the **only** thing in this entire pass with released
ROS 2 Jazzy binaries (4.1.0, with `coal` as a dependency). **An honest correction to
a common assumption:** Pinocchio advertises analytic derivatives of RNEA/ABA, **not**
of collision distance. Everyone — OCS2, pink — builds the distance gradient from
`nearest_points` plus joint Jacobians, an approximation that **degenerates exactly at
`min_distance == 0`**. OCS2's own source carries the admission:
`// TODO(perry): is there a way to calculate a correct jacobian for the case of
distanceVector = 0?` That degeneracy is precisely the regime the kernel keeps landing
in. The one paper attacking it is **iDCOL** (<https://arxiv.org/abs/2602.03250>, Feb
2026) — analytic derivatives of contact distance, location and normal, explicitly
regularising degenerate geometries, with a claimed open-source C++ implementation
whose **repo URL and licence could not be verified**. **Verdict: watch closely** —
the only thing surveyed that attacks the specific numerical failure at the heart of
the mm problem.

---

## 19. Whole-body mobile manipulation, base + arm coupled

**There is no shipped ROS 2 stack for this, and the published art is coarser than
what the kernel already has.**

### 19.1 OCS2 — mine the self-collision algebra, do not adopt the stack

<https://github.com/leggedrobotics/ocs2>, **BSD-3**, active. **Self-collision via
pinocchio/hpp-fcl: confirmed** — `SelfCollision.cpp` computes
`violations[i] = distanceArray[i].min_distance - minimumDistance_` with the gradient
from `getJointJacobian(..., LOCAL_WORLD_ALIGNED)` translated to the nearest points
via `skewSymmetricMatrix`, sign-flipped on penetration, with a real unit test — the
reference implementation of the Jacobian one would otherwise write from scratch.
**ESDF obstacle cost: weaker than assumed** — the **only** constraint consuming
`DistanceTransformInterface` is `EndEffectorDistanceConstraint`, one row per
end-effector frame, so there is **no whole-body ESDF constraint in-tree**. **ROS 2:
community forks only**, no Jazzy fork verified, no published MPC rate. **Verdict:
watch — steal `ocs2_self_collision`, inherit its zero-distance TODO as a known open
defect.**

### 19.2 RMMI — the closest published answer, and an order of magnitude too slow

<https://arxiv.org/abs/2408.16206> (IROS 2025): a **neural SDF** map consumed by NEO's
velocity-damper QP extended to a coupled base+arm Jacobian. *Source shows* "fixed
step time of 0.05 s (i.e, a control loop of 20 Hz)", with real-world deployment on a
**2358-point** body model; +25 % success on cluttered reaching, no QP-solve or
SDF-query timing published. The benchmark repo has **no LICENSE** on either branch.
**Verdict: watch strongly, adopt nothing.** Take the architecture (one shared QP over
`[base_dof, arm_dof]`); 20 Hz is an order of magnitude under the kernel's floor and a
2358-point body model is *coarser* near contact than the 25 mm grid.

### 19.3 The rest of the base+arm field, and a calibrating data point

- **Reactive Base Control for On-The-Move Mobile Manipulation**
  (<https://arxiv.org/abs/2309.09393>) — shared QP over base+arm, 20 Hz, 48 %
  real-world task-time reduction, but the obstacle model is a **2D lidar occupancy
  grid** and the gripper is queried as a **single point**. No code. **Verdict:
  irrelevant to adopt — valuable as calibration.** The field's *deployed* answer to
  base+arm coupling is coarser than this repo's current 3D voxel grid.
- **Zheng et al. 2025** (<https://arxiv.org/abs/2501.02815>) — polytopic free regions
  + AL-DDP; no rate, no self-collision, **no licence**. **Irrelevant**; the one
  takeaway is convex-decomposing *free space* rather than enumerating obstacles.
- **Chen et al.** (<https://arxiv.org/abs/2409.14775>) — base+arm CBF-QP with
  self-collision; no rate, no code. **Irrelevant.**
- **AutoMoMa** (<https://arxiv.org/abs/2604.12565>) — GPU trajectory *dataset
  generation*, CC BY-NC-SA. **Irrelevant** — named explicitly so its 80× number is
  not mistaken for a control-rate claim.
- **Perceptive MPC** (RA-L 2020) — ancestor of `ocs2_perceptive`; ROS 1 research
  code. **Watch, cite, do not build on.**

The only released ROS 2 Jazzy packages in this space are the **geometry libraries**
(`pinocchio` + `coal`), not a controller. **Verdict for the kernel: build, don't
adopt.**

---

## 20. Predictive safety filters for action chunks — and why none of them fix mm

**The headline is a negative result:** every predictive, reachability, CBF or
flow-matching filter found **inherits the resolution of whatever geometry it is
handed**. Replacing a 25 mm voxel grid with an HJ value function or a barrier
function does not change that the 25 mm voxel says "penetration" when a fingertip
touches. Mostly *watch*, with two exceptions that are architectural, not geometric.

### 20.1 The exception worth adopting: mode-switched constraint sets

<https://arxiv.org/abs/2608.00600> (Enwerem et al., Aug 2026): a four-mode hybrid
controller — **reach → close → hold → lift** — with **hysteresis** on the number of
fingers in contact (band `N⁻ < N⁺` plus dwell limits), where the constraint set
changes per mode: in *reach* the target object is a full obstacle; in *close*
**coarse finger-link constraints drop out of the active set**, palm constraints
shrink, and **one fingertip clearance constraint per finger enters at ZERO margin**;
in *hold* the arm is frozen with a wrench-quality CBF holding the risk-adjusted
force-closure margin within `k_wq = 0.02` of its value at onset. *Docs claim* convex
hulls substituted for collision meshes at load (**0.1 ms** per query vs **330 ms**
for exact mesh distance), CBF-CLF QP solved in **0.09 ms inside a 20 ms interval**,
minimum observed obstacle clearance **6.7 mm**. Licence and code **not stated**.
**(a)** structurally yes — it does not achieve mm *sensing*, it **stops asking the
question** during grasp phases; **(b)** no; **(c)** yes, via hulls.

**Verdict: adopt the idea, not the code.** The kernel's binary accept/drop/latch at a
*fixed* margin is the root cause of the 15 mm false positive; the smallest-diff fix
is a per-link, per-phase margin table with hysteresis, and the target object's ACL
entry toggled by the declared task phase. ADR-0092/0097 built this bespoke; the paper
is external evidence the shape is right, plus the hysteresis detail the in-tree
version does not have.

### 20.2 Flow-matching and diffusion filters — watch

- **SafeFlow** (<https://arxiv.org/abs/2504.08661>) — barriers constraining the
  generated trajectory over the **whole planning horizon**, training-free at
  deployment; licence and ROS 2 **could not verify**. Structurally the closest
  analogue to vetting a π0.5 / GR00T chunk as a whole — but it does nothing the
  kernel cannot get by evaluating its existing check across the whole chunk.
- **Neuro-Symbolic Safety Guidance via Constrained Flow Matching**
  (<https://arxiv.org/abs/2607.01378>) — corrects violations *during denoising*.
  *Docs claim* 82.8 % collision avoidance / 81.6 % task success on SafeLIBERO;
  collision representation **not stated**, no code. **Watch.**
- **Individual-CBF-guided diffusion** (<https://arxiv.org/abs/2606.12640>) —
  multi-agent offline RL, CC BY-NC-ND. **Irrelevant.**

### 20.3 Reachability and MPSF — irrelevant at manipulator scale

- **refineCBF** — **no LICENSE file**, ROS wrapper "tested solely with ROS Noetic",
  2D platforms; the usable piece underneath is `StanfordASL/hj_reachability`
  (**MIT**, JAX). **Irrelevant** — HJ value functions in a 7-DoF configuration space
  are not tractable and none of the papers claim otherwise.
- **Language-conditioned latent safety filters**
  (<https://arxiv.org/abs/2608.00315>) — an HJ actor/critic conditioned on
  natural-language constraints; the abstract claims *reduced* violations and *partial*
  transfer. **Latent filters are less metrically precise than a voxel grid, not
  more.** **Irrelevant to the mm problem; watch as an S2-layer complement** — it can
  express "don't touch the hot pan", which no metric kernel can.
- **UPSi** (<https://arxiv.org/abs/2604.26836>) — safe-RL benchmarks only, no
  physical robot. **Irrelevant.**
- **Towards Safe Robot Foundation Models** (<https://arxiv.org/abs/2503.07404>) — a
  generalist policy in an ATACOM safe action space, air hockey. **Watch.**
- **SQ-CBF** (<https://arxiv.org/abs/2602.11049>) — superquadric SDFs via GJK;
  runtime, code and licence **not verifiable**. **Watch**: a continuous SDF gives
  sub-voxel gradients, but the superquadric *fit* error to a cluttered scene becomes
  the new bottleneck and nobody quantified it.

**Standing correction:** PACS is now **accepted to ICRA 2026**; code/licence still
could not be verified.

### 20.4 Real-Time Chunking — a liveness guarantee, not a safety one

<https://arxiv.org/abs/2506.07339> (Physical Intelligence, NeurIPS 2025) poses
asynchronous chunking as **inpainting**, with inference delay `d := ⌊δ/Δt⌋` and
execution horizon `s`. **Read the guarantee precisely** — *source shows*: "so long as
`d ≤ H − s`, this strategy will satisfy the real-time constraint and guarantee that
an action is always available when it is needed." That is an **availability
guarantee**; the paper is equally explicit that without inpainting the transition
between chunks "may be arbitrarily discontinuous and out-of-distribution". Latency
numbers bound any in-kernel design: π0 (3B) spends **46 ms on KV-cache prefill
alone** on an RTX 4090 against a 20 ms tick; the real-robot setup (π0.5, H=50,
Δt=20 ms, 5 denoising steps) measures **76 ms baseline / 97 ms RTC** model latency,
LAN adding 10–20 ms, giving **d ≈ 6**. Code licence **could not be verified**.

**The kernel-relevant consequence.** The frozen prefix of length `d` — ~6 steps,
~120 ms at π0.5 rates — is a window in which the policy **cannot** re-plan but the
world **can** move, and nothing in RTC re-checks it. So the frozen prefix is a
*safety obligation*: it must be re-vetted against fresh perception every cycle, with
braking authority inside it. RTC and PACS are a coherent pair; RTC alone is not.
**Verdict: adopt RTC as an execution strategy, and treat its frozen prefix as an
explicit kernel contract.** Newer chunk-execution work, for tracking only: PACE
(<https://arxiv.org/abs/2606.00537>), action-prior denoising
(<https://arxiv.org/abs/2605.25537>), DREAM-Chunk
(<https://arxiv.org/abs/2606.18589>), adaptive inference-time chunking
(<https://arxiv.org/abs/2604.04161>).

### 20.5 Runtime monitoring — one cheap signal worth taking

**VLA-FAIL** (<https://arxiv.org/abs/2606.21386>, KIT, June 2026) offers two failure
detectors needing **no failure data**: last-layer Mahalanobis distance, and the
useful one, **Action Chunk Consistency (ACC)** — flag a failure when *consecutive
chunks become inconsistent*; code **could not be verified**. **FAIL-Detect**
(<https://arxiv.org/abs/2503.08558>) distils policy inputs/outputs into scalar
signals then applies conformal prediction for guaranteed thresholds (**watch**), and
**KnowNo** (<https://arxiv.org/abs/2307.01928>) is an **S2 / Reasoner-layer**
control, not an actuation-path one.

**Verdict: adopt ACC as a third verdict.** Nearly free — consecutive chunks are
already held in the RTC buffer — and it expresses something the current binary
verdict cannot: *"the policy is confused, escalate to S2"* is a different event from
*"geometry says collision, latch"*, and maps onto the replanning ladder, not the
E-stop path.

**Also checked and not relevant:** <https://arxiv.org/abs/2604.23775>
("Vision-Language-Action Safety") is an **adversarial-security** survey — poisoning,
backdoors, jailbreaks — not physical safety, and does **not** supersede this page. By
contrast, <https://arxiv.org/abs/2512.11908> ("Safe Learning for Contact-Rich Robot
Tasks: A Survey", IIT) is the nearest thing to a 2026 superset and is worth reading
in full; only its abstract was verified here, and it does **not** mention momentum
observers or intent discrimination.

---

## 21. Proprioceptive contact discrimination — where the mm problem is actually solvable

The kernel is trying to infer *intent* from *geometry* at a resolution geometry
cannot deliver. Force does deliver it — with the caveat that force measures
**contact, not penetration depth**.

### 21.1 What a Franka-class arm can actually measure

*Source shows* — text extracted from the official **Franka Emika Panda** datasheet
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

**Correction that matters for this repo's hardware claims: the FR3 datasheet
publishes none of these numbers.** *Source shows* — its full text contains only
"Force/Torque sensing: link-side torque sensor in all 7 axes" and "Guiding force
~2.5 N", so **anyone citing "FR3 has 0.05 N force resolution" is citing the Panda
sheet.** The FR3 sheet does give a worst-case safe Cartesian position accuracy for
stopping functions of **50 mm**, PL d / Cat. 3 on E-stop. **How
`tau_ext_hat_filtered` is computed is undisclosed** — *source shows*,
[libfranka#91](https://github.com/frankaemika/libfranka/issues/91), the only
substantive maintainer reply: "this data is streamed from the backend and computed
within." Treat it as an uncharacterised black box; the one independent measurement
(Petrea & Bertoni, IEEE 2021) is paywalled and is the single most valuable missing
number for this decision.

**Can this discriminate a ~1 mm intended touch from a collision?** On magnitude,
comfortably yes for *detection*: a 0.035 N RMS noise floor against a 1–5 N deliberate
fingertip touch is 30–140× margin. On *intent*, no — a 3 N accidental brush and a
3 N deliberate touch are the same number. A caveat that must be carried: the binding
constraint is *relative torque accuracy 0.15 Nm*, not the 0.005 Nm noise RMS, which
at a ~0.5 m effective lever is roughly **0.3 N of unmodelled force error**, 6× the
headline "< 0.05 N" *(derived arithmetic, not a Franka claim)*. The honest framing:
proprioception measures **force, not penetration**. It cannot tell 1 mm from 15 mm.
It *can* tell "contact is occurring and it is gentle" — the discriminator actually
needed, orthogonal to and vastly more precise than a 25 mm voxel grid.

### 21.2 The momentum-observer lineage, and the number that bounds it

**Haddadin, De Luca, Albu-Schäffer, "Robot Collisions: A Survey on Detection,
Isolation, and Identification", T-RO 33(6), 2017** —
<http://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf>. Still
canonical, and it names the missing stage: detection → isolation → identification →
**classification** → reaction → post-collision, where classification means
"accidental or intentional … light or severe … permanent, transient, or repetitive".
It says plainly that this decision "cannot be done purely at the control level, as
global environmental information and reasoning is certainly needed", and establishes
(§IV-B) that the contact point and force vector can be estimated **from
proprioception alone**, the momentum observer usable "in place of a wrist
force-torque sensor". **The kernel today has detection wired straight to latch, with
no classification stage at all.** That is the gap, and the reference architecture for
closing it is nine years old and uncontroversial.

**Birjandi & Haddadin, RA-L 2020**, quoted verbatim: "it is due to this error that
the collision threshold is typically set to **1 Nm** in the momentum observer … the
error of the proposed solution **does not exceed 0.1 Nm** … With such 0.1 Nm
threshold, the collision is **detected within 1.2 ms**." **But the cost is one IMU
per link**, which an FR3 does not have, and no code or licence was found. **Verdict:
watch, do not build** — start from stock `tau_ext_hat_filtered`. Modern successors
(Kalman external-torque estimation, momentum observer + LSTM, finite-time and
super-twisting observers, SafePR) are incremental threshold-reduction papers on the
same 2017 observer; **none ships ROS 2, none claims intent discrimination.**

### 21.3 The highest-value line: intent from the torque *spectrum*

**"Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots"** —
<https://arxiv.org/abs/2508.12435> (Aug 2025), Franka Emika Research robot,
**built-in joint sensors only**, no skin, no vision. *Docs claim* **over 95 %
accuracy** in both contact detection and gesture classification via STFT-2D-CNN /
STT-3D-CNN, with the key finding that **time-frequency (spectrogram) representations
significantly outperform raw torque signals**. Class count, force levels, code and
licence **could not verify**.

**Why this is the most on-point result in either survey.** It shows the
*temporal-spectral signature* of a torque transient — not its amplitude — carries the
intent information. A grasp closure and an accidental impact at the same peak force
have different spectra: the impact is a broadband impulse, the grasp a ramp. That is
precisely the discriminator the kernel lacks, and a ~10 ms STFT over 7 channels at
1 kHz is a small, allocation-free C++ addition.

**Complementary, and directly about "intended vs unintended": Aim-Aware Collision
Monitoring** (Proper, Kurdas, Abdolshah, Haddadin, Saccon; RA-L 8(8):4609–4616, 2023,
DOI 10.1109/LRA.2023.3284371) discriminates **expected from unexpected post-impact
behaviour** by comparing an idealised rigid robot-object-environment response against
the measured one, with a **causal envelope filter** generating classification error
bounds, driven by a **bandpass momentum observer**. This is impact-*aware*
manipulation — contact intended by design — exactly the regime the kernel keeps
mis-classifying. **Verdict: adopt the framing** (compare against a predicted contact
response, not a fixed margin).

### 21.4 Contact localisation — set expectations correctly

The **Contact Particle Filter** (Manuelli & Tedrake, IROS 2016) is the classic worth
understanding — a convex QP inside a particle filter finding the contact point(s)
that best explain the measured external joint torque. **UniTac**
(<https://arxiv.org/abs/2507.07980>) does whole-robot touch sensing from
proprioception only at ~2000 Hz, localising **within 8.0 cm on a Franka arm**, but
its repo has **no LICENSE file** — all-rights-reserved, **do not vendor**. DLR's
Science Robotics 9(93):eadn4008 (2024) localises touch *trajectories* well enough to
read handwritten letters, but its resolution and sensitivity figures are paywalled,
**could not verify**. **Verdict: budget for mm-accurate contact *detection*, not
mm-accurate contact *localisation*** — the published open number for proprioceptive
localisation is 7–8 cm.

### 21.5 The plumbing already exists — this is the cheap rung

*Docs claim*, franka_ros2 (Jazzy): `franka_hardware` exposes a **`ForceTorqueSensor`
named `<arm_prefix><robot_type>_tcp`** carrying **`K_F_ext_hat_K`** (estimated
external wrench in the stiffness frame) as six state interfaces, and the **Gazebo
plugin mirrors the same tcp wrench interfaces**, so a sim and a hardware controller
activate identically — which matters directly for this repo's sim-first test tiers.
The fetched docs do **not** list `tau_ext_hat_filtered` or `O_F_ext_hat_K` as
separate state interfaces; that is "docs do not show it", not "it does not exist".
`ros2_control`'s `force_torque_sensor_broadcaster` (Apache-2.0) publishes
`WrenchStamped` with filter-chain support, and in-controller access goes through
`semantic_components::ForceTorqueSensor` — **no topic hop on the hot path**.

**One landmine:** the Universal Robots driver's broadcast wrench is expressed
relative to `base_link` and is "almost always incorrect as the robot's pose changes"
([UR driver#235](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/235)),
with a `wrench_transformer_node` as the workaround. An external wrench in the wrong
frame is worse than none — CLAUDE.md §2's "TF2 is the only source of coordinate
frames" applies to wrenches too.

**Verdict: adopt — the plumbing costs nothing.** For an FR3 the external wrench
arrives as a standard `ForceTorqueSensor` semantic component inside `ros2_control`,
consumable in a C++ controller at kernel rate with **zero new dependencies**.

### 21.6 Gripper-level grasp confirmation — a phase signal, not a safety signal

*Docs claim*, libfranka `Gripper::grasp()`: returns true iff
`(width − epsilon_inner) < d < (width + epsilon_outer)`, with **defaults
`epsilon_inner = epsilon_outer = 0.005` m — a ±5 mm width band**; force is a
*commanded* parameter, not a measured confirmation. So it is a **width predicate**:
"something roughly the expected size is between my fingers". It cannot discriminate a
1 mm touch (its own tolerance is 5 mm) and says nothing about arm-link contact.
**Verdict: adopt as a *phase* signal only** — a good input to the §20.1 mode switch,
worthless as a contact-force measurement. Adjacent 2026 work, both with
**unverifiable code/licence**: "Current as Touch"
(<https://arxiv.org/abs/2607.03529>) and **FACTR 2**
(<https://arxiv.org/abs/2606.12406>), whose NEXT component learns external joint
torque from **motor current on commodity arms with no torque sensors**. **Watch** —
relevant only if OpenRAL ever targets a current-only arm.

### 21.7 MuJoCo contact-force fidelity — an honest negative, and a blocker

**No 2023–2026 paper was found validating MuJoCo or MJX contact-force *magnitudes*
against real force-torque measurements.** Five search phrasings were tried. What *was*
established:

- **MuJoCo's own documentation frames its contact model as a deliberate
  approximation, not a calibrated force model** — *docs claim*
  (<https://mujoco.readthedocs.io/en/stable/computation/index.html>): it "drops the
  strict complementarity constraint at the heart of the LCP formulation", so "force
  and velocity in the contact normal direction can be simultaneously positive",
  justified because "all physical materials allow some deformation". The docs nowhere
  claim forces are in calibrated real Newtons, and put the burden of physical
  validity on the user's `solref`/`solimp` choices.
- The strongest indirect evidence is **"Direction Matters: Learning Force Direction
  Enables Sim-to-Real Contact-Rich Manipulation"**
  (<https://arxiv.org/abs/2602.14174>, 2026), whose entire premise is that force
  *magnitudes* are "highly sensitive to simulation inaccuracies" while force
  *directions* "remain robust across the sim-to-real gap". It is an argument *from*
  the magnitude gap, not a measurement of it, and gives no quantitative error.

**Verdict: record this as a blocker.** The sim path cannot produce a trustworthy
real-Newton analogue today, and no published work says otherwise. The correct move is
to have the sim path emit **contact-occurring boolean plus force direction**, and to
calibrate any magnitude threshold on hardware only. Anything else fabricates a
number, which CLAUDE.md §1.2 forbids. This is the citation behind §10 Path C's
standing caveat that its sim seam ships "a calibration knob, not a claim".

---

## 22. Opportunistic finds outside the eight directions

### 22.1 `mj_geomDistance` is already installed — and must NOT be the oracle

**This subsection's original verdict is withdrawn.** It read "adopt as the offline
oracle first", on the reasoning that MuJoCo ships a mesh-exact signed distance and is
already a dependency. The premise is true and the conclusion is wrong: **that call is
the defective instrument PR #170 removed from the evidence path and PR #204 (issue
#190) removed from the last safety path** (§11 item 4; standing caveat 8 of the
[validation-evidence ledger](collision-validation-evidence.md)). The description is
kept because the facts are right and the trap is instructive.

*Docs claim*, MuJoCo API reference:

```c
mjtNum mj_geomDistance(const mjModel* m, mjData* d, int geom1, int geom2,
                       mjtNum distmax, mjtNum fromto[6]);
```

"Returns the smallest signed distance between two geoms and optionally the segment
from geom1 to geom2". *Source shows*, upstream `doc/changelog.rst`: added in **3.1.6
(2024-06-03)** together with `distance`/`normal`/`fromto` MJCF sensors; the
`nativeccd` flag (GJK/EPA replacing MPR) arrived in **3.2.3** and **became the
default in 3.3.0 (2025-02-26)**, with the docs warning that "distances are inaccurate
when using the legacy CCD pipeline".

**And that last inference is exactly the trap.** "The legacy path is inaccurate,
therefore the pinned default is accurate" is the reasoning this survey originally
made, and #170 falsified it by measurement: under `mujoco 3.8.0` on the **default
native-CCD path** the call returns `+0.000000` for `robot0_link7_collision` against
`fridge_right_group_freezer_door_main` on `robocasa_fridge_drawer` layout 9, where
the certified truth is `+0.148512 mm`, writing a 126.264 mm witness segment whose
endpoints lie 526.6 mm and 432.9 mm outside the two geoms it claims to touch.
Displacing the link by one picometre returns the right answer, so it is a degenerate
*configuration*, not a distance regime — and a scene's reset pose is where such
configurations live, because fixtures are placed on exact axis-aligned numbers. #170
states it flatly: **no `distmax`, no "only trust it below N mm" rule and no `ncon`
cross-check separates the good answers from the bad.** The libccd path is worse and
unbounded in `distmax` (−57 mm through a 48 mm panel). Re-measuring the checked-in
rounds found four recorded `0.000 m` readings whose certified values are +14.8,
+82.2, +98.8 and +107.9 mm. The original text also called its `fromto` witness "the
same shape as the existing `GjkWitness`", which is **wrong**: `GjkWitness`
(`collision.hpp:661`) stores hull vertex *indices* and cube-corner sign codes as a
warm-start hint — "never Minkowski difference points" — so it is a cache, not a
nearest-point pair.

**Corrected verdict: do NOT adopt `mj_geomDistance` as the oracle, or as anything
else on an adjudication or permission path.** The requirement it was proposed for is
real — the 26-DOP → hull staging does need an mm-scale regression oracle — but the
instrument already exists in-tree:
**`openral_hal.convex_distance.convex_geom_distance`**, which *proves* each answer
(separating-axis certificate when separated, exact SAT when overlapping,
inscribed/circumscribed bracketing for round types with no ball form, explicit
refusal for anything it cannot certify). It costs ~1 ms per mesh↔box pair against
~0.8 µs, affordable precisely because an oracle runs offline, and adds **no**
dependency.

`mj_geomDistance` keeps exactly one legitimate use in this repo, already in the tree:
as the *subject* of the two regression tests that assert the defect still reproduces
(`tests/sim/safety/test_geom_distance_instrument_robocasa.py`,
`tests/sim/safety/test_support_probe_instrument_robocasa.py`), so an upstream fix is
noticed rather than silently assumed. Two limits recorded for any future
MuJoCo-backed proposal: it needs `mjModel`/`mjData`, so the *world* side would have
to exist as MuJoCo geoms — fine for the robot and declared objects, not for a
depth-derived world, where runtime geom insertion is not supported; and it is a CPU
per-pair call with no published rate on this repo's hardware.

### 22.2 Contact-implicit MPC — contact as a decision variable, not a violation

**C3 / C3+ (consensus complementarity control)**, Posa's DAIR lab. Push Anything
(<https://arxiv.org/abs/2510.19974>) reports *docs claim* **99.9 % (700/701)**
single-object success at **~14 Hz**, and 8–15 Hz multi-object. Code in
`DAIRLab/dairlib`, **MIT** (*source shows*), built on Drake. **Verdict: irrelevant to
the kernel, important as a framing.** 8–15 Hz is far below the 30–200 Hz floor and
this is a controller, not a filter — but it is the clearest existence proof that a
whole research line treats contact as a *decision variable* rather than a *safety
violation*. A kernel whose only verdict on contact is "latch" is structurally at odds
with contact-rich manipulation, the same conclusion §20.1 reaches from the grasp
side.

### 22.3 A cautionary "sub-millimetre" claim, checked

**"Learning Fast, Tool-aware Collision Avoidance for Collaborative Robots"**
(<https://arxiv.org/abs/2508.20457>, RA-L) is frequently summarised as achieving
"sub-millimetre accuracy". *Source shows*: the sub-millimetre figure is **task-space
position tracking error** in nominal operation — **not** collision discrimination.
The actual geometry is "a voxel grid with 0.05 m resolution", 50 Hz control, on an
Indy7 with a RealSense D435; no code or ROS package. **Verdict: irrelevant, and
recorded deliberately.** 50 mm voxels — exactly the class of claim that must be
followed to the primary source before it influences a safety decision
(CLAUDE.md §1.2).

### 22.4 Self-filtering the robot and its payload out of depth — a ROS 2 option

§3.3 covers MoveIt's octomap self-filter. A standalone ROS 2 alternative exists:
**`leggedrobotics/robot_self_filter`** — *source shows*, `package.xml` declares
`<license>BSD-3-Clause</license>`, `ament_cmake`, `rclcpp`, `tf2`, `urdf`. The widely
cited `ctu-vras/robot_body_filter` (clip / contains / **shadow** tests) remains
**ROS 1**. No binary Jazzy release was verified for the ROS 2 port. **Verdict:
watch.** Marginal on its own, but the *shadow test* — removing points seen *through* a
robot link — is the mechanism most directly relevant to the carried-payload false
positives, and no equivalent exists in-tree.

### 22.5 The force-limiting standard §8 cites has been superseded

§8's comparison table rests its only "yes" on **ISO/TS 15066**. The 2025 revision of
the ISO 10218 series is published and folded ISO/TS 15066's power-and-force-limiting
requirements in (**ISO 10218-1:2025** and **ISO 10218-2:2025**), retiring the TS as a
standalone document. Cite ISO 10218-2:2025 in hazard-log entries, with TS 15066 kept
for the Annex A body-model tables (Table A.2/A.3 values re-verified against the
standard's own PDF, §12).

---

## 23. Comparison table — scored against the kernel's three needs

"mm near contact?" asks whether the method can distinguish a ~1 mm intended touch
from a real penetration, **against a sensed world**.

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

---

## 24. What this survey does NOT establish

- **No latency was measured in this tree.** Every rate quoted — the µs-scale coal and
  MoveIt numbers, CAPT's 9.89 ns, RDF's 0.54 ms, mink's QP, Tesseract's `contactTest`
  (for which there is no published number at all) — is an upstream benchmark on other
  hardware. The adoption decisions for §15.1 and §17 turn on a measurement that does
  not yet exist.
- Servo's scaling formula is read from source, but no claim is made that it is a
  *rated* safety function — MoveIt's docs make no such claim either.
- The two motivating drawer runs have no in-tree fixture; the predicted-config and
  out-of-coverage-voxel defects should get the same recorder-side pinning the earlier
  corpus got.
- Whether graded slowdown (Path A) converts *this repo's* within-quantization stop
  class into completed grasps: PACS establishes the effect exists on robomimic + real
  FR3, but the magnitude on the RoboCasa scenes was open at the time of writing, and
  the Path A precondition (velocity-free policy observations) had not been audited
  per adapter. *(Since measured: it does not reproduce —
  `collision-validation-evidence.md` §6.)*
- The VLSA-vs-PACS disagreement is unresolved: VLSA reports a CBF layer *raising*
  success (+17.25 %) where PACS's reactive-CBF baseline collapses it (0.04). Setups
  differ; neither invalidates the other yet.
- **MuJoCo contact-force *magnitude* is unvalidated as a real-Newton analogue**
  (§21.7; FORGE re-tunes on hardware). Path C's sim seam ships a calibration knob,
  not a claim.
- G3's mechanism (the free-space `continue`) is read from source but the numeric
  split of the 15× is uninstrumented.
- coal's near-contact fixes are on `devel`, unreleased at tag 3.0.4 — §4.1's
  downgrade should be re-checked against whatever tag exists when Path B's
  narrow-phase question is next opened.
- cuRobo's Apache-2.0 relicense was verified at `main`; anything pinned to a pre-V2
  release (≤ 0.7.x) is still non-commercial, and `nvblox_torch` remains
  non-commercial — re-verify before any dependency lands (CLAUDE.md §1.9).
- **CAPT would not make the kernel mm-accurate on its own.** It removes voxel
  quantisation; it leaves sphere/OBB slop, a conservative filter radius, and the
  sensor's own 7 mm–2.2 cm dispersion. The *combined* residual is unestablished.
- **No learned distance field found is conservative.** Joho et al. state this
  explicitly and pair their network with a geometric checker to restore the
  guarantee, so §14.1 can only be a *filter in front of* the geometric check.
- **Tesseract's SDF path and its CCD path are mutually exclusive** — nothing here
  establishes which the kernel should choose, or whether a split (cast managers for
  self-collision, implicit-SDF for the manipulation volume) is coherent.
- **The Franka force numbers are Panda numbers.** The FR3 datasheet publishes none of
  them, `tau_ext_hat_filtered`'s computation is undisclosed, and the one independent
  measurement is paywalled. The 0.3 N lever-arm figure in §21.1 is derived
  arithmetic.
- **The torque-spectrum intent claim rests on one paper's abstract**, and nothing
  establishes that the same signal separates a *grasp* from a *collision* as opposed
  to separating deliberate human gestures.
- **Several load-bearing repos have no licence at all** — Neural-JSDF, UniTac, RMMI's
  `frankie_planner`, Zheng et al., refineCBF, `dynablox_ros2`. Those are
  all-rights-reserved by default.
- **Nothing here has been run.** No code was built, no benchmark reproduced, no
  fixture added. This is a reading list with verdicts attached.
