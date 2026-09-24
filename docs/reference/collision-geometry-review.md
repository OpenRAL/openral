# Collision geometry review: capsules, hulls, and the seven URDF-lowered robots

> Status: analysis, plus the conservative refit on PR #324 (`perf/generalization-capsules`).
> Everything in `cpp/openral_safety_kernel/` and `packages/openral_safety/` is Safety-WG gated
> (hazard-log entries listed in §9). The kernel was not changed.
> The one-shot analysis scripts are not checked in (§1.11 keeps fixtures for tests, not studies).
> The two properties that matter are: every vertex is enclosed, and the kernel never misses a
> sampled contact. Both are pinned in CI by `tests/unit/test_collision_geometry_enclosure_urdf.py`
> and `tests/unit/test_collision_contact_sweep.py`, which use the same method at 120 poses per robot.

## 0. Decision

1. **Ship capsules now, fitted to the whole link.** Every capsule now holds its link's
   `<collision>` and `<visual>` geometry, at minimum volume, with a declared 1 mm headroom.
   With this, the 3000-pose contact sweep finds **0 false negatives on all 7 robots**. The PR as
   it stood missed 7,354 contacts on H1 and 518 on G1. Capsules are the cheapest exact
   predicate the kernel has for self-collision (11 ns per pair). They are the right
   representation for the four serial arms: Franka, UR5e, UR10e and Rizon 4 clear their rest
   poses and false positives drop 24–53 % against the PR (21–54 % over all six non-H1 robots).
2. **A capsule cannot be both conservative and usable on three robots.** On H1, G1 and SO-100,
   the real links rest 1–9 mm apart. The capsule excess is 20–40 mm. Single capsules and chains
   of 2–4 capsules per link all still overlap at rest, while the exact convex hulls are clear.
   The representation that fixes this is the kernel's existing **box + `tight_geometry`
   (26-DOP → exact hull, GJK)** narrow phase, extended to every mesh-backed link. It is
   **recommended, not implemented** (§8, draft ADR). The three rest-pose tests stay strict
   xfails with measured reasons. No ACM exemption was added to make them pass.

## 1. What the kernel does today (read from source)

| job | where | primitive path | per-check routine |
|---|---|---|---|
| robot vs itself | `collision.cpp` `check_self_collision` | capsule↔capsule, box↔capsule, box↔box; box↔box re-asked on exact hulls when both links declare `tight_geometry` (#191) | `capsule_distance` (exact segment–segment − radii), `box_capsule_distance` (48-step ternary search), `box_box_distance` (15-axis SAT lower bound), `hull_hull_distance` (GJK, lower bound) |
| robot vs world (octomap → voxel grid) | `check_voxel_collision` | capsule pass (each cell as a cube vs capsule), box pass (staged 26-DOP → hull when declared) | `box_capsule_distance` per cell; `box_box_distance` / `dop_cell_lower_bound` / `hull_cell_distance` |
| robot vs grasped / nearby objects | `check_attached_voxel_collision`, `check_attached_self_collision` | payload primitives (staged hull for the voxel pass, #266) vs world; payload vs robot-link boxes | as above |

Facts that constrain the choice:
- The hot path is allocation-free and deterministic. `NoAlloc.*` covers the staged hull path.
- `CollisionModel` already supports several capsules per link. The manifest lowering
  (`envelope_loader._capsules_by_link`) rejects more than one.
- `tight_geometry` refines a `BoxShape` only. `collision-hull-narrow-phase.md` §5.2 says a
  capsule-lowered robot cannot use it.
- The voxel broad-phase window is computed from the primitive's own extents. Any refinement
  must stay inside it (`collision-hull-narrow-phase.md` §4.2).

## 2. What the reference stacks do (sources read for this review)

- **Drake.** "the Mesh's .obj file will generally be replaced by its convex hull". It uses the
  convex hull for point contact and signed-distance queries
  ([Drake geometry file formats](https://drake.mit.edu/doxygen_cxx/group__geometry__file__formats.html)).
  This is the same exact-hull choice as the kernel's stage 2.
- **MoveIt.** The octomap "can directly be passed into FCL, the collision checking library that
  MoveIt uses". Robot self-filtering from the octree uses the link meshes with
  `padding_offset`/`padding_scale`
  ([planning scene monitor](https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html),
  [perception pipeline](https://moveit.picknik.ai/main/doc/examples/perception_pipeline/perception_pipeline_tutorial.html)).
  The FCL signed-distance issues (#221/#574/#575) are recorded in
  `collision-safety-alternatives-survey.md` §4. I did not re-read them for this review.
- **cuRobo.** Robot geometry is spheres per link, made in Isaac Sim's Lula Robot Description
  Editor. The page does not claim the spheres cover the mesh. It adds
  `collision_sphere_buffer: 0.005` and a `self_collision_buffer`
  ([cuRobo robot configuration](https://curobo.org/tutorials/1_robot_configuration.html)).
  A third-party repo reports 61 spheres for the Franka
  ([Robot-SphereShield](https://github.com/JayadityaVetsa/Robot-SphereShield)). I did not open
  `franka.yml` itself (404 at the path tried).
- **Pinocchio / coal.** GJK on convex shapes. Nesterov-accelerated GJK is "up to two times
  faster" than GJK ([Montaut et al., arXiv:2205.09663](https://arxiv.org/abs/2205.09663)). The
  repo's survey (§4.1) records coal's EPA hot-path allocations (#649) as the reason not to
  adopt it on the kernel path.
- **In-repo prior art.** Measured on the Panda: `collision-tight-geometry.md` §3 has sphere
  sets at 27–41 mm excess even with 128 spheres, a 26-DOP at 25.7 mm, and the exact hull at 0.
  `collision-hull-narrow-phase.md` §3 has the costs.

## 3. Coverage, per robot and link (measured)

Truth is the link's **visual mesh**: the CAD of the part (robot_descriptions / vendored URDF).
Metrics per primitive:
- `out`: vertices outside the primitive.
- `max out`: worst depth outside, in mm.
- `excess`: max support excess `max_u h_P(u) − h_mesh(u)` over 4000 directions, in mm. This is
  the worst-case clearance the kernel under-reports.
- `vol`: primitive volume ÷ convex-hull volume.

The per-link table is in the PR #324 review record.

Per-robot aggregates. For `vol` and `excess`, lower is tighter; `out` must be 0.

| robot | base (pre-PR) out / max | PR #324 out / max | PR vol | PCA fit on visual vol / excess | **min-volume capsule** vol / excess | 2-capsule chain | 3-capsule chain | PCA OBB vol / excess |
|---|---|---|---|---|---|---|---|---|
| franka_panda | 10 / 0.0 mm | 25 / 0.1 mm | 2.69 | 2.87 / 53 | **1.68 / 38** | 1.57 / 43 | 1.53 / 39 | 1.61 / 55 |
| g1 | 15,048 / 60.8 mm | 17,087 / **61.1 mm** | 2.75 | 2.71 / 40 | **1.78 / 27** | 1.70 / 29 | 1.61 / 27 | 1.53 / 35 |
| h1 | 820,104 / 549.3 mm | 822,705 / **549.3 mm** | 0.26 | 2.60 / 51 | **2.15 / 42** | 1.80 / 39 | 1.44 / 37 | 1.72 / 63 |
| so100_follower | 41 / 3.4 mm | 1,246 / **15.4 mm** | 2.78 | 3.02 / 28 | **2.22 / 26** | 2.19 / 27 | 2.01 / 24 | 1.64 / 29 |
| rizon4 | 63 / 3.3 mm | 272 / **3.3 mm** | 1.97 | 3.05 / 69 | **1.51 / 35** | 1.34 / 46 | 1.47 / 39 | 1.41 / 39 |
| ur5e | 30 / 0.1 mm | 77 / 0.0 mm | 3.06 | 3.56 / 63 | **1.62 / 29** | 1.47 / 33 | 1.19 / 35 | 1.28 / 35 |
| ur10e | 11 / 0.5 mm | 41 / 0.5 mm | 3.64 | 4.06 / 79 | **1.70 / 34** | 1.48 / 37 | 1.22 / 38 | 1.29 / 43 |

What it shows:
- **The dominant defect was the truth source, not the primitive.** The vendor `<collision>`
  blocks under-cover the parts:
  - H1 uses placeholder primitives: a 5 cm pelvis sphere, 1 cm hip cylinders, and a torso box.
    The torso mesh reaches **549 mm** outside the kernel's torso capsule.
  - G1's shoulder pitch/roll are cylinders the shoulder mesh leaves by 31/61 mm, and its ankle
    rolls are four contact spheres the foot leaves by 19 mm.
  - SO-100's collision meshes omit the servo housings (26 mm).
  - Rizon 4's 27–33-vertex collision meshes cut 3–8 mm into the CAD.
  - The PR's "zero vertices outside" held against those collision blocks, not against the robot.
    Its sub-0.1 mm residuals on Franka and UR come from the manifest's 4-dp rounding.
- **The PCA axis was the second defect.** On the same truth, the minimum-volume capsule is
  1.2–2.4× smaller than the PCA fit, and its support excess is 1.1–2.3× lower.
- **Chains of capsules buy 5–30 % volume and no support excess.** The 3-capsule chain is best
  on UR (1.19–1.22 vs 1.62–1.70). Excess is flat or worse, because each sub-capsule still has
  round caps on a blocky part.
- **A PCA OBB is tighter by volume on four robots, but no better on excess.** It is also a box
  without a hull, and `box_capsule_distance` costs 60× a capsule pair (§4).

### After the refit (committed manifests, truth = collision ∪ visual)

| robot | links | vertices outside | min margin | vol ÷ hull | mean max-excess | worst link excess |
|---|---:|---:|---:|---:|---:|---|
| franka_panda | 9 | 0 | 0.93 mm | 1.74 | 38.9 mm | panda_link0 83.0 mm |
| g1 | 27 | 0 | 0.90 mm | 1.85 | 28.1 mm | left_hip_roll_link 51.0 mm |
| h1 | 20 | 0 | 0.91 mm | 2.08 | 40.2 mm | torso_link 109.4 mm |
| so100_follower | 5 | 0 | 0.95 mm | 2.36 | 26.9 mm | base 43.7 mm |
| rizon4 | 8 | 0 | 0.94 mm | 1.59 | 36.9 mm | base_link 82.3 mm |
| ur5e | 6 | 0 | 0.95 mm | 1.68 | 28.7 mm | forearm_link 33.8 mm |
| ur10e | 6 | 0 | 0.92 mm | 1.74 | 34.6 mm | upper_arm_link 51.1 mm |

Every vertex sits at least 0.90 mm inside its committed capsule (1 mm headroom minus the 4-dp
rendering). `tests/unit/test_collision_geometry_enclosure_urdf.py` pins ≥ 0.5 mm.

### OpenArm (out of scope for edits; MJCF collision meshes, q = 0)

- The hand-authored primitives under-cover by up to 83.7 mm (finger_pair), 64.6 (link1) and
  37.5 (link7). See the OpenArm link-3 evaluation in the PR #324 review record.
- Fitting the same meshes gives `vol ÷ hull / max-excess`:

| link | PCA trimmed capsule (the parallel session's `fit_trimmed_capsule_to_vertices`) | PCA box | min-volume capsule (this PR) |
|---|---|---|---|
| link1 | 2.00 / 30.9 | 2.37 / 53.3 | **1.51 / 24.2** |
| link2 | 2.47 / 30.3 | 2.49 / 50.3 | **2.10 / 31.4** |
| link3 | 1.86 / 25.1 | **1.62** / 33.7 | 1.64 / **20.0** |
| link4 | 2.16 / 32.9 | **1.34 / 26.1** | 1.79 / 23.8 |
| link5 | 2.03 / 29.9 | 2.34 / 53.1 | **1.88 / 28.1** |
| link6 | 3.08 / 22.4 | 2.84 / 36.3 | **2.04 / 14.2** |
| ee_base_link | 2.45 / 25.4 | 2.18 / 40.3 | **1.92 / 21.9** |

The chosen approach works for OpenArm. The MJCF fitter should call `fit_capsule_to_vertices`
(min-volume + headroom) and still keep a box where it is smaller (link4), and it should add the
MJCF visual meshes to the cloud.

## 4. Per-check cost in the kernel (measured)

A standalone benchmark links the unmodified `collision.cpp` and runs it single-core on an Intel Core
Ultra 7 255HX (`g++ -O2 -DNDEBUG`, median of 31 repetitions). The laptop was shared, so treat
the numbers as ±20 %.

| routine | ns / call |
|---|---:|
| `capsule_distance` (capsule↔capsule, self pair) | **11.1** |
| `box_box_distance` (15-axis SAT) | 138.8 |
| `box_capsule_distance` (box↔capsule, ternary search) | 669.3 |
| voxel cell (25 mm cube) vs **capsule** link (`box_capsule_distance`) | **763.7** |
| voxel cell vs box link (`box_box_distance`) | 142.7 |
| `hull_hull_distance`, 32 / 64 / 152 / 320 vertices (incl. box fallback, poses near contact) | 855 / 1,488 / 3,175 / 6,756 |

- The 26-DOP stage-1 per cell is 28.7–35.3 ns on the i5-8600K
  (`collision-hull-narrow-phase.md` §3.1). I did not re-measure it here.
- Reading:
  - For **self-collision**, capsules are 13–600× cheaper than any alternative. A G1 at 27 links
    and about 350 pairs costs about 4 µs per configuration on capsules. Hulls cost 0.3–2 ms if
    most box pairs overlap. The box broad phase prunes far pairs, so the real figure is lower.
    It is not measured.
  - For **robot vs world**, capsules are the *slow* path: 764 ns per cell against 143 (box) and
    about 30 (DOP).
  - Capsules fit self-collision. Boxes with tight hulls fit the voxel check and the
    close-clearance self pairs.

## 5. Rest-pose and ACM problems

Rest pose is q = 0 when the joint limits admit it, else the SRDF `ready`. The kernel trips at
`d ≤ self_collision_margin_m` (0 on all seven).

| robot | rest | before (PR) | after | tightest real gap at rest (exact hull / nearest vertices) | best capsule set (k = 1…4 per link) |
|---|---|---|---|---|---|
| franka_panda | SRDF ready | clear | clear | — | — |
| ur5e / ur10e | q = 0 | not checked (the test skipped UR: `base_frame` outside the chain) | **clear** (now checked) | — | — |
| rizon4 | SRDF ready | clear | clear | — | — |
| h1 | q = 0 | "clear", only because the capsules did not cover the torso | **trips** (8 pairs) | shoulder_roll↔torso **+1.2 / 2.8 mm**; shoulder_yaw +27.4; elbow +55.0; hip_pitch↔pelvis +38.3 | k = 1: −9.8; k = 2: −6.8; k = 3: −15.8; k = 4: −45.3 mm |
| g1 | q = 0 | trips | trips (2 pairs) | shoulder_yaw↔torso **+7.8 / 8.8 mm** (L), +9.2 (R) | k = 1: −12.6; k = 2…4: −35…−64 mm |
| so100_follower | q = 0 (folded) | trips (1 pair) | trips (5 pairs) | base↔upper_arm hulls overlap (vertices 1.0 mm apart: the resting arm); lower_arm↔shoulder +0.9 | all k < 0 |

- Longer chains can come out worse because the slice planes are generic. The rows show only
  that no capsule set of that size reached a positive gap.
- H1, G1 and SO-100 stay **strict xfails**, with these numbers as the reason.
- **H1 forearm miss (from the PR body).** At `left_shoulder_pitch 0.99, roll −0.15, yaw −1.03,
  left_elbow −0.62`, 13,604 `left_elbow_link` visual vertices lie inside `torso_link`. The PR's
  geometry did not trip there; the refit trips at **−199.1 mm**. It is now a negative test case.
- **Negative cases, all 7 robots** (`test_real_self_collision_still_trips`). Each is an in-limit
  pose with mesh interpenetration confirmed by vertex containment in a watertight mesh:
  - Franka: hand in link1.
  - H1: two forearm-into-chest poses.
  - UR5e / UR10e: 32,601 / 33,530 wrist_2 vertices inside upper_arm.
  - Rizon 4: 93 base vertices inside link6.
  - SO-100: 207 wrist vertices inside base.
  - G1: 163 torso vertices inside right_elbow.

**The ACM exempted pairs that really collide.** The sweep checked every ACM-exempt,
non-adjacent pair whose hulls met, using vertex containment:
- **Three vendor SRDF "Never" rows are false.** The kernel exempted:
  - Franka `panda_link2`/`panda_link6`: 5 of 5 samples, up to 3,705 vertices inside.
  - UR10e `shoulder_link`/`wrist_2_link`: 10 of 10, up to 429.
  - Rizon 4 `base_link`/`link4`: 2 of 2, up to 112.

  **Removed** from the vendored SRDFs, each with an evidence comment, and the ACM regenerated.
  The kernel now checks these pairs. This is more conservative, and the rest poses still clear.
- **G1 `shoulder_roll`↔`torso` and `ankle_roll`↔`knee` are certified always-colliding for the
  capsules**, and their meshes overlap in 32/36 and 63/71 sampled poses. Both pairs are 1-DoF
  apart (through `shoulder_pitch` / `ankle_pitch`), and the meshes nest at the joint. The kernel
  cannot tell CAD nesting from a collision with any capsule, so the exemption stays. It is
  **listed for the WG** (hazard entry below) and is not called harmless.
- The refit changed the ACM. Every row is still generated by the certificate or the SRDF, with
  no hand rows:
  - **g1:** −`left_shoulder_yaw↔torso` (now checked), +`right_shoulder_roll↔torso` and
    +`left_wrist_roll↔left_wrist_yaw`. These two mirror rows the other side already had, so the
    matrix is now left/right symmetric.
  - **h1:** +`hip_pitch↔hip_yaw` (L/R) and +`pelvis↔hip_roll` (L/R). The two hip pairs' hulls
    overlap in 100 % of samples, with 0 of 100 mesh penetrations. The pelvis pairs had no hull
    contact in the sweep.

## 6. Contact sweep: kernel verdict vs mesh ground truth

3000 uniform in-limit configurations per robot, seed 20260924.
- **Kernel side:** manifest FK and `kernel_predicates`.
- **Truth side:** URDF FK (yourdfpy) and the visual meshes. A pair is clear only when a
  separating plane between the convex hulls is proven (GJK witness + separating-axis bound, else
  an exact LP). Hull contact ⊇ mesh contact, so an FN count here is an upper bound on the
  mesh-level count.
- FK agreement between the two kinematic sources was ≤ 1.5e-7 m everywhere.
- FN = truth contact on a checked pair that the kernel does not trip on. FP = kernel trips and a
  separating plane is proven.

| robot | pre-PR base: contacts / **FN** / FP | PR #324: contacts / **FN** / FP | **after**: contacts / **FN** / FP | FP vs PR | FP vs base |
|---|---|---|---|---:|---:|
| franka_panda | 651 / **0** / 5,701 | 651 / **0** / 2,550 | 656 / **0** / 1,196 | −53 % | −79 % |
| g1 | 10,257 / **249** / 33,492 | 10,392 / **518** / 13,595 | 10,041 / **0** / 6,235 | −54 % | −81 % |
| h1 | 7,444 / **7,338** / 22 | 7,444 / **7,354** / 4 | 1,444 / **0** / 9,222 | (was under-covering) | — |
| so100_follower | 5,483 / **0** / 6,604 | 5,483 / **0** / 4,275 | 5,483 / **0** / 3,356 | −21 % | −49 % |
| rizon4 | 146 / **0** / 1,411 | 146 / **0** / 905 | 148 / **0** / 685 | −24 % | −51 % |
| ur5e | 1,494 / **0** / 4,694 | 1,494 / **0** / 1,955 | 1,494 / **0** / 1,339 | −32 % | −71 % |
| ur10e | 1,339 / **0** / 4,238 | 1,339 / **0** / 1,869 | 1,349 / **0** / 1,092 | −42 % | −74 % |

- Contacts are counted on pairs the kernel checks (non-ACM), over 3000 poses.
- **Robot-level misses**, meaning poses with a real contact and no trip anywhere: H1 had 2,893
  (base) and 2,925 (PR) of 3000. It has 0 after, and every robot is at 0 after.
- H1's contact count drops from 7,444 to 1,444 because its two hip pairs are now certified
  always-colliding (§5). Their hulls overlap in all 6,000 pair-samples, with 0 of 100 mesh
  penetrations.
- Franka, UR10e and Rizon 4 gain 5, 10 and 2 contacts: the three SRDF rows removed in §5 are
  now checked, and all of those contacts trip.

- The in-tree test `tests/unit/test_collision_contact_sweep.py` runs 120 poses per robot through
  the exact kernel parameters (`collision_params_from_description` + the kernel's FK
  convention). It fails on any FN. Against the PR's manifests it fails on H1 (285 of 287
  contacts missed) and G1 (12 of 295).
- The enclosure test fails on H1 (549 mm), G1 (61 mm) and SO-100 (15 mm).

## 7. What changed on the branch

- `urdf_lowering.fit_capsule_to_vertices`:
  - Minimum-volume capsule. The axis is searched over the 3 principal axes, a 256-direction
    Fibonacci hemisphere and a local refinement.
  - The axis line goes through the minimal enclosing circle of the projection, and the segment
    is the shortest that holds every vertex.
  - `+ CAPSULE_HEADROOM_M = 1 mm`. Containment holds by construction for every candidate axis.
  - Clouds are reduced to their hull vertices first (exact for a convex primitive).
- `lower_link_geometry`:
  - Fits `<collision>` ∪ `<visual>`.
  - Cylinders and spheres become circumscribing polytopes (the old rim sampling was inscribed).
  - It fails closed when a link's geometry resolves only partially.
- `robots/h1/h1.urdf`: the visual mesh refs `package://h1_description/…` became
  `rd:h1_description:robots/h1_description/meshes/…`, the portable form `vendor-urdf` emits.
- Re-lowered: franka_panda, g1, h1, so100_follower, rizon4, ur5e, ur10e. The loosening guard
  flagged the links that grew; those entries were deleted first, as the CLI instructs.
- SRDF: three false "Never" rows removed (§5).
- Tests:
  - New `test_collision_geometry_enclosure_urdf.py` and `test_collision_contact_sweep.py`.
  - `test_collision_geometry_zero_pose.py`: negative cases for all 7 robots including the H1
    forearm miss; FK roots at the chain root (UR now checked); H1/G1/SO-100 strict xfails with
    measured reasons; `tight_geometry` box pairs deferred to the kernel-level hull test.
  - Fitter unit tests: headroom, and never larger than the PCA-axis capsule.
  - g1 certificate tests retargeted to pairs still certified. `torso↔shoulder_yaw` is now a
    live check.
  - `panda_mobile` loosening expectation: now 5 of 7 links, up to 1.31×.
- Pre-existing, separate commits:
  - `test_robot_manifest_migration` compared the lowered blocks against a 0.1 fixture.
  - An unused `type: ignore`.

## 8. Recommendation: box + exact hull for mesh-backed links (not implemented)

**Why.**
- §5: on H1, G1 and SO-100 the real rest clearance (0.9–9 mm) is below any capsule's excess
  (20–110 mm). The exact hull clears every case except SO-100's resting contact.
- §4: for the world check, the DOP/hull path is 5–25× cheaper per cell than the capsule path.
- The kernel already has this narrow phase, proven on `panda_mobile` (#166/#191/#266). No C++
  change is needed for box↔box pairs.

**Migration cost (estimate, not built).**
- Lowering (Python, ~150 lines):
  - A mesh link lowers to an OBB (containing) + `tight_geometry`: the 26-DOP from tangent slabs
    of the collision ∪ visual hull, and the hull vertices when there are ≤ 320
    (`kMaxTightHullVertices`), else DOP only.
  - The visual hulls here have 100–2,187 vertices. Most exceed 320, so either stage-1 only
    (≈ 13–26 mm excess on the Panda) or a *containing* vertex reduction is needed. A containing
    reduction is not available in-tree. It needs a proof obligation, for example the hull of a
    tangent polytope over a direction set, which contains by construction.
- Mixed pairs: box↔capsule has no hull path, so a robot migrates all links at once.
  `check_attached_self_collision` stays box↔box.
- Cost:
  - Self-collision on a 27-link humanoid could rise from µs to sub-ms per configuration. Measure
    with the DOP-vs-DOP stage-1 that `collision-hull-narrow-phase.md` §9.3 names as the next
    lever before enabling it.
  - The real-HAL margin (0.02 m) caveat of §3.2 applies per robot.
- Manifests grow by 13×2 slab values + up to 320×3 vertex coordinates per link. The schema
  already exists.
- The ACM must be re-derived against hulls: `_certified_always_colliding` already withholds for
  tight pairs.
- Tests: extend the enclosure and sweep tests to boxes + hulls; the kernel-level
  `test_kernel_panda_link5_link7` pattern per robot.

**Draft ADR text (for `OpenRAL/management/adr/`).**

> **ADR-XXXX — Mesh-backed robot links lower to box + exact convex hull, not capsules**
>
> *Status:* proposed. *Deciders:* Safety WG. *Date:* 2026-09-24.
>
> *Context.* The safety kernel checks self-collision and robot-vs-world against primitives
> lowered from each robot's URDF. PR #324's review measured the URDF-lowered capsules on seven
> robots:
> - The vendor `<collision>` blocks under-cover the parts by up to 549 mm (H1), 61 mm (G1) and
>   15 mm (SO-100). The lowering now fits collision ∪ visual geometry, minimum-volume, with 1 mm
>   headroom, and a 3000-pose sweep finds zero false negatives.
> - Conservative capsules have 20–110 mm of support excess. On H1, G1 and SO-100 the links
>   rest 0.9–9 mm apart, so no capsule set (1–4 per link) can clear the rest pose. The exact
>   convex hulls do.
> - Per 25 mm voxel, a capsule costs 764 ns against 143 ns for a box and ~30 ns for the 26-DOP
>   stage.
>
> *Decision.* A link whose URDF geometry includes a mesh lowers to a containing OBB plus
> `tight_geometry` (26-DOP; exact hull when ≤ `kMaxTightHullVertices`). Primitive-only links
> keep capsules. The lowering proves containment (mesh ⊆ hull ⊆ DOP ⊆ OBB) with no optimiser
> tolerance. The ACM is re-derived against the hulls.
>
> *Consequences.*
> - Tighter self- and world-collision without kernel changes.
> - Self-collision cost rises (measure per robot, gate on the 10 ms budget with the horizon).
> - Larger manifests.
> - A containing vertex reduction is needed for hulls over 320 vertices, or those links run
>   stage 1 only.
> - H1/G1/SO-100 rest poses become checkable without exemptions.
> - The capsule lowering remains the onboarding default until each robot's migration passes the
>   enclosure test, the contact sweep (FN = 0) and the rest-pose test.
>
> *Alternatives rejected.*
> - Capsule chains: they do not clear the rest poses (§5).
> - Sphere trees: 27–41 mm excess even at 128 spheres on the Panda; the in-repo study.
> - SDF: a proven bound needs 64³ grids per link, 7 MiB for seven links, and gives a filter,
>   not a containment proof.
> - coal/FCL on the hot path: allocation and near-contact accuracy issues; the survey §4.

## 9. Hazard-log and management text needed (not written to the management repo)

- **Entry 046 (update, H1/G1/SO-100/Rizon 4 under-coverage), cause:** vendor `<collision>`
  blocks under-cover the parts (numbers in §3). The kernel accepted real contacts: in 3000
  poses, 7,354 on H1 and 518 on G1 with the PR's manifests. The H1 forearm miss is one instance.
- **Entry 046, mitigation:**
  - Fit collision ∪ visual, minimum-volume, with 1 mm headroom.
  - Enclosure test (≥ 0.5 mm margin) and contact-sweep test (FN = 0), both in CI.
  - H1 URDF visual refs made resolvable.
- **Entry 046, residual:**
  - Truth is the CAD visual mesh. Cables, covers and grippers not in the URDF are not modelled.
  - The sweep samples 3000 poses and does not prove FN = 0. Containment does, for non-exempt
    pairs.
- **Entry 043 (update, loss of conservatism):** now measured. Across 3000 poses per robot, FN = 0
  on all seven. FP falls 21–54 % against the PR and 49–81 % against the pre-PR base on the six
  robots the PR already covered. H1's FP rises from 4 to 9,222, because its capsules now cover
  the body; see §6.
- **New entry: false "Never" SRDF rows.** Franka link2/link6, UR10e shoulder/wrist_2 and Rizon 4
  base/link4 really collide within limits; the kernel exempted them. The rows are removed.
  - Residual: the other SRDF "Never" rows are only as good as the MoveIt sampling that produced
    them. The sweep found no other hidden contact in 3000 poses.
- **New entry: capsule-certified exemptions with real mesh overlap.** G1 `shoulder_roll↔torso`
  (L/R) and `ankle_roll↔knee` (L/R) are 1-DoF-coupled and CAD-nested. The kernel cannot check
  them with capsules; resolving that needs the §8 hull lowering. WG to accept or require it.
- **New entry / WG decision: rest-pose refusals.** With conservative capsules, H1, G1 and SO-100
  trip at their rest pose, so the kernel refuses motion from rest. All three are sim-only
  (`hal.real: null`) except the SO-100 bench. The WG chooses between the §8 migration, a
  measured rest pose, or accepting the refusal. An ACM exemption is not an option: it would
  hide contacts that are real at other poses.
- **ADR:** the §8 draft.

## 10. Overlap with the parallel OpenArm refit session (read-only review)

Worktree `.claude/worktrees/agent-a798ebe88185f702f`, branch `fix/openarm-collision-refit`,
uncommitted.
- **Scope.** It fixes **OpenArm only**. `fit_collision_geometry_from_mjcf` runs on the MJCF
  lowering path, behind `--fit-mjcf-geometry`. The 7 URDF-lowered robots are untouched by it.
- **Overlap in `urdf_lowering.py`:**
  - Its `fit_trimmed_capsule_to_vertices` is the PR's pre-refit `fit_capsule_to_vertices`
    (PCA axis, trimmed ends), duplicated. On merge it should become a call to the new
    `fit_capsule_to_vertices` (min-volume, headroom). Per §3, that gives a 10–34 % smaller
    volume on 6 of 7 OpenArm links.
  - Its `fit_tightest_primitive` (capsule vs PCA box by volume) and `fit_obb_to_vertices` do
    not conflict. Keep them, but note a PCA box of the same cloud has no declared headroom.
  - The textual conflict risk is low. The branches touch different functions: this branch
    rewrites `fit_capsule_to_vertices`, the helpers above it and `lower_link_geometry`; that one
    adds functions after `fit_capsule_to_vertices`, plus the MJCF path.
- **Tests.** Its `tests/unit/test_collision_geometry_enclosure.py` places vertices by MuJoCo FK
  and primitives by the kernel's FK, a stronger cross-check. This branch's enclosure test is
  named `..._enclosure_urdf.py` to avoid a clash. The two could share the vertex-in-primitive
  helper after merge.
- **Gap in its approach.** It fits the MJCF **collision** meshes only. On OpenArm those are
  close to the parts, but §3 shows why the visual meshes should join the cloud.
