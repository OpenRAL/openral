# Collision geometry review: one fitter, capsules and hulls, eight robots

> Status: analysis plus the refit on PR #324 (`perf/generalization-capsules`), which now also carries
> PR #325's MJCF fitter (the OpenArm), folded into the same code path.
> Everything in `cpp/openral_safety_kernel/` and `packages/openral_safety/` is Safety-WG gated
> (hazard-log entries listed in §9). **The C++ kernel was not changed.**
> The one-shot analysis scripts are not checked in (§1.11 keeps fixtures for tests, not studies).
> What CI pins: every vertex **and every surface triangle** of every link lies inside its
> primitives (`test_collision_geometry_enclosure_urdf.py`; `test_collision_geometry_enclosure.py`
> for every robot with an MJCF twin), and every refined link's mesh lies inside its hull; the kernel
> misses no sampled contact (`test_collision_contact_sweep.py`); no robot trips at rest except
> SO-100 (`test_collision_geometry_zero_pose.py`).

## 0. Decision

1. **One fitter, whatever the geometry source.** The URDF and MJCF readers only collect a link's
   collision **and** visual geometry (plus the MJCF twin's CAD where the twin's frames coincide
   with the URDF's, §12) and hand it to `fit_link_primitives`. It fits one capsule, one box and
   capsule chains of 2-3, each holding every vertex with 1 mm of headroom, and keeps the one with
   the **least mean protrusion** (support excess over the link's convex hull, averaged over 1024
   directions); ties within 3 % go to the cheaper kernel kind. Least volume, the first rule
   tried, refused *more* poses than before on four of seven robots (§3). A link may carry several
   primitives; the manifest lowering no longer refuses them (the kernel always accepted them).
2. **Box + exact hull, chosen per robot by measurement.** A link lowered with `--tight-link`
   becomes a box plus `tight_geometry` (26-DOP + exact convex hull, ≤ 320 vertices), and the
   kernel already re-asks a box pair of the two hulls. The kernel's own verdict on 1000 seeded
   poses decided (§8.2):
   - **the six arms (Franka, UR5e, UR10e, Rizon 4, SO-100, OpenArm): a hull on every link.**
     False stops fall to 0-14 in 3000 poses (from 684-3356), and self-collision stays cheaper
     than mixing boxes with capsules;
   - **the humanoids (H1, G1): hulls on the links of the pairs that trip at rest or carry ≥ 5 %
     false stops, capsules and chains elsewhere.** A hull on every link refuses every pose:
     their CAD nests at the hips and shoulders, and only the capsules' always-colliding
     certificate exempts those pairs. H1 and G1 clear their rest pose for the first time.
   Capsules and hulls work together **across links**, not within one link (§8.1).
3. **What stays with the WG:** SO-100's resting contact (its upper arm lies on the base; the
   exact hulls touch); the remaining OpenArm exemptions; the humanoids' self-collision cost
   (up to 0.4 ms per configuration, §4) and the kernel broad phase that would remove it (§8.3);
   the G1 and UR twins whose frames differ from the URDF (§12); Franka's fingers and SO-101's
   gripper, which carry no primitive (pre-existing, §12); the octomap self-filter padding
   (§8.4); a follower-joint finger in the kernel (§8.3).

## 1. What the kernel does today (read from source)

| job | where | primitive path | per-check routine |
|---|---|---|---|
| robot vs itself | `collision.cpp` `check_self_collision` | capsule↔capsule, box↔capsule, box↔box; box↔box re-asked on exact hulls when both links declare `tight_geometry` (#191) | `capsule_distance` (exact segment–segment − radii), `box_capsule_distance` (48-step ternary search), `box_box_distance` (15-axis SAT lower bound), `hull_hull_distance` (GJK, lower bound) |
| robot vs world (octomap → voxel grid) | `check_voxel_collision` | capsule pass (each cell as a cube vs capsule), box pass (staged 26-DOP → hull when declared) | `box_capsule_distance` per cell; `box_box_distance` / `dop_cell_lower_bound` / `hull_cell_distance` |
| robot vs grasped / nearby objects | `check_attached_voxel_collision`, `check_attached_self_collision` | payload primitives (staged hull for the voxel pass, #266) vs world; payload vs robot-link boxes | as above |

Facts that constrain the choice:
- The hot path is allocation-free and deterministic. `NoAlloc.*` covers the staged hull path.
- `CollisionModel` supports several capsules and boxes per link. The manifest lowering
  (`envelope_loader`) refused more than one until this change; it now lowers them all.
- `tight_geometry` refines a `BoxShape` only. `collision-hull-narrow-phase.md` §5.2 says a
  capsule-lowered robot cannot use it; §8 below is what that means per link.
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

## 3. The fitter and how it chooses (measured)

**Truth** for every link is the union of its `<collision>` and `<visual>` geometry (URDF), or every
geom of its body (MJCF: collision and visual, meshes and primitives, with the OpenArm's
equality-coupled second finger swept over the gripper stroke into `finger_pair`). For H1, Franka
and Rizon 4 the MJCF twin's meshes are added too (§12). The vendor `<collision>` blocks
under-cover the parts (H1's torso by 549 mm, G1's shoulders by 61 mm, SO-100's servo housings by
26 mm), which is why the visual meshes count.

**Candidates**, all built so every vertex lies inside with `CAPSULE_HEADROOM_M` = 1 mm:
- one minimum-volume capsule (`fit_capsule_to_vertices`);
- one minimum-volume oriented box (`fit_obb_to_vertices`, rotating calipers over the hull);
- capsule chains of 2 and 3: the link's mesh cut into equal slabs along the capsule axis, one
  capsule per slab. The slab keeps its cut points, so each capsule holds its slab's piece of the
  surface and the chain holds the whole surface, seams included.

**Metrics**, reported per candidate in `LinkFit.candidates`:
- `vol ÷ hull`: primitive volume (union, Monte-Carlo for a chain) over the link's convex-hull
  volume;
- max and mean **protrusion**: the support-function excess `h_P(u) − h_hull(u)` over 1024
  directions. The mean is how far, on average, the kernel's surface stands off the real part;
  the max is the worst direction.

**The rule** is least mean protrusion, ties within 3 % going to capsule < fewer capsules < box
(kernel cost, §4). Four rules were compared on the same fits, 1000 seeded poses each, false stops
and refused poses (a pose is refused when any checked pair trips):

| robot | A: least volume | **M: least mean protrusion** | B: least max protrusion | M1: M without chains |
|---|---:|---:|---:|---:|
| franka_panda | 596 / 679 | **454 / 459** | 456 / 461 | — |
| ur5e | 396 / 415 | **318 / 393** | 388 / 478 | 404 / 481 |
| ur10e | 498 / 470 | **340 / 318** | 419 / 422 | 385 / 353 |
| rizon4 | 255 / 256 | **199 / 198** | 256 / 261 | — |
| so100_follower | 725 / 560 | **761 / 631** | 761 / 633 | — |

Cells: poses refused / false stops per 1000 poses (hull truth). Same fits, same ACM rule, only
the pick changes. (H1, G1 and the OpenArm were not run under every rule: 45 min per G1 pass.)

Least volume (A) picks boxes on blocky links; their corners reach further in the directions
that matter, and UR10e, Franka and Rizon 4 refuse more poses than with single capsules. Least
mean protrusion (M) is best on four of five, and chains matter: without them (M1, convex only)
UR5e refuses 27 % more poses. SO-100 is the exception (A is 5 % better); it is resolved by hulls
anyway (§8).

## 4. Per-check cost in the kernel (measured)

**Isolated routines** (measured for #324). A standalone benchmark links the unmodified `collision.cpp` and runs it single-core on an Intel Core
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
- Reading: per pair, capsules are the cheapest self-collision check; for robot vs world they are
  the *slow* path (764 ns per cell against 143 for a box and about 30 for the DOP). A box pair
  runs the GJK hull test only when the boxes already meet, so an all-box robot pays SAT on far
  pairs and GJK on near ones; a box ↔ capsule pair always pays the 669 ns ternary search.

**Real-kernel check of the shipped manifests.** `collision.cpp`, unmodified, linked into a
harness that builds the `CollisionModel` exactly as `lifecycle_kernel.cpp` does from
`collision_params_from_description` and runs `forward_kinematics` + `check_self_collision` on the
same seeded poses (one configuration per check; the kernel runs one per horizon step):

| robot | before: median / p99 / max µs | after: median / p99 / max µs | after, 16 steps at max |
|---|---:|---:|---:|
| franka_panda | 0.9 / 1.1 / 1.3 | 8.9 / 35.6 / 50.1 | 0.80 ms |
| ur5e | 0.5 / 0.6 / 0.9 | 2.1 / 20.6 / 26.6 | 0.42 ms |
| ur10e | 0.4 / 0.5 / 1.0 | 2.3 / 18.4 / 26.1 | 0.42 ms |
| rizon4 | 0.6 / 0.7 / 1.1 | 2.5 / 15.0 / 25.0 | 0.40 ms |
| so100_follower | 0.3 / 0.4 / 0.6 | 6.5 / 29.0 / 40.7 | 0.65 ms |
| h1 | 7.6 / 8.7 / 44.4 | 175.9 / 219.2 / 234.4 | 3.75 ms |
| g1 | 16.7 / 19.1 / 57.9 | 267.2 / 376.4 / 436.6 | 6.99 ms |
| openarm | 4.6 / 5.3 / 6.1 | 41.1 / 71.9 / 115.1 | 1.84 ms |

The refused counts agree with the Python verdict used in §6 (`kernel_link_gap`), so the GJK
iteration cap (24) did not fall back to a box bound on these poses. The laptop was shared; treat
the times as ±30 %.

What the cost is made of: not the hulls. Re-deriving the OpenArm's and G1's hulls at a 16- to
320-vertex budget moved their median cost by under 10 % (G1: 235 → 258 µs), while fewer vertices
raised G1's refusals (936 at 16 vertices against 759 at 320). The cost is the box ↔ capsule pairs:
`box_capsule_distance` is a 48-step ternary search (669 ns) and `check_self_collision` has no
broad phase, so a humanoid with 11 boxes and 30 capsules pays for ~330 of them on every
configuration, near or far. The arms, all boxes, pay box ↔ box SAT (139 ns) and GJK only on
the few pairs whose boxes meet. The kernel checks every horizon step, so a 16-step chunk costs
H1 ~3-4 ms and G1 ~4-7 ms of self-collision (both simulation-only, `hal.real: null`); the arms
stay under ~0.8 ms (Franka, max) and the OpenArm under 2 ms. `chunk_validation_deadline_us`
(1 ms, "p99 target") is declared but not enforced by the kernel today; the humanoids exceed it.
§8.3 recommends the broad phase that removes this.

## 5. Rest poses

Rest pose is q = 0 when the joint limits admit it, else the SRDF `ready` (Franka). The
kernel trips at `d ≤ self_collision_margin_m` (0 on all eight).

| robot | rest pose | before | after |
|---|---|---|---|
| franka_panda | SRDF group_state 'ready' | clear | clear |
| ur5e | q = 0 | clear | clear |
| ur10e | q = 0 | clear | clear |
| rizon4 | q = 0 | clear | clear |
| so100_follower | q = 0 | trips (5 pairs; base↔lower_arm -1.0 mm, base↔upper_arm -26.7 mm, …) | trips (1 pairs; base↔upper_arm -2.5 mm) |
| h1 | q = 0 | trips (8 pairs; left_elbow_link↔torso_link -7.9 mm, left_hip_pitch_link↔pelvis -0.4 mm, …) | clear |
| g1 | q = 0 | trips (2 pairs; left_shoulder_yaw_link↔torso_link -12.6 mm, right_shoulder_yaw_link↔torso_link -17.1 mm) | clear |
| openarm | q = 0 | clear | clear |

- H1 and G1 were strict xfails (no capsule set, 1-4 per link, cleared them). With the resting
  links as box + hull, the kernel re-asks those pairs of the exact hulls, which clear by
  +0.9 mm (H1 shoulder roll ↔ torso) to +55 mm. They are no longer xfails.
- **SO-100 stays a strict xfail.** Its folded q = 0 puts the upper arm on the base: the exact
  hulls touch (nearest mesh vertices 1.0 mm apart), so every conservative representation trips,
  the exact hull included. It needs a measured rest pose or a WG call, never an exemption.
- A per-link volume rule (A) made Franka trip at `ready` (link5 ↔ link7, −2.8 mm, a box corner);
  the protrusion rule does not.

## 6. Contact sweep: kernel verdict vs ground truth, before and after

3000 uniform in-limit configurations per robot, seed 20260924.
- **Kernel side:** manifest FK and `kernel_predicates`, each link pair folded as the minimum over
  its primitives and re-asked of the exact hulls where both boxes carry one — exactly
  `check_self_collision` (`tests/unit/_collision_kernel_verdict.kernel_link_gap`; matched
  pose-for-pose by the C++ kernel in §4).
- **Truth side:** the link's convex hull, or — for a link with several primitives, whose union is
  not convex — the hulls of its surface split among its primitives (`surface_pieces`). A link's
  single hull would fill the concavities a chain leaves out and report contacts no mesh makes
  (the first run of this sweep counted 86 such "misses" on H1). Hull contact ⊇ mesh contact, so
  `missed` is an upper bound on real misses.
- `false stops` = the kernel trips and a separating plane is proven. `slop-only` = refused poses
  with no truth contact on any checked pair.

| robot | set | primitives | vol ÷ hull | max / mean protrusion (mm) | vertices outside | contacts | missed | false stops | poses refused | slop-only | rest |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| franka_panda | before | 9 (9 capsule) | 1.84 | 38.3 / 11.0 | 0 | 657 | 0 | 1195 | 1166 / 3000 | 800 | clear |
| franka_panda | after | 9 (9 hull) | 1.69 | 56.3 / 27.3 | 0 | 657 | 0 | 2 | 366 / 3000 | 0 | clear |
| ur5e | before | 6 (6 capsule) | 1.78 | 28.2 / 8.9 | 0 | 1494 | 0 | 1339 | 1178 / 3000 | 613 | clear |
| ur5e | after | 6 (6 hull) | 1.38 | 36.7 / 17.9 | 0 | 1494 | 0 | 3 | 565 / 3000 | 0 | clear |
| ur10e | before | 6 (6 capsule) | 1.76 | 33.7 / 10.3 | 0 | 1349 | 0 | 1092 | 1171 / 3000 | 611 | clear |
| ur10e | after | 6 (6 hull) | 1.37 | 44.9 / 20.9 | 0 | 1349 | 0 | 14 | 563 / 3000 | 3 | clear |
| rizon4 | before | 8 (8 capsule) | 1.59 | 35.4 / 7.6 | 0 | 149 | 0 | 684 | 675 / 3000 | 607 | clear |
| rizon4 | after | 8 (8 hull) | 1.49 | 40.6 / 21.4 | 0 | 149 | 0 | 1 | 68 / 3000 | 0 | clear |
| so100_follower | before | 5 (5 capsule) | 2.41 | 26.5 / 8.8 | 0 | 5483 | 0 | 3356 | 2557 / 3000 | 578 | base/lower_arm -1.0; base/upper_arm -26.7; lower_arm/shoulder -20.5 … |
| so100_follower | after | 5 (5 hull) | 1.71 | 31.1 / 12.0 | 0 | 5483 | 0 | 0 | 1979 / 3000 | 0 | base/upper_arm -2.5 |
| h1 | before | 20 (20 capsule) | 2.05 | 39.6 / 13.0 | 0 | 1507 | 0 | 9159 | 2998 / 3000 | 2170 | left_elbow_link/torso_link -7.9; left_hip_pitch_link/pelvis -0.4; left_shoulder_roll_link/torso_link -9.7 … |
| h1 | after | 32 (22 capsule, 10 hull) | 1.81 | 53.6 / 20.2 | 0 | 1503 | 0 | 241 | 907 / 3000 | 79 | clear |
| g1 | before | 27 (27 capsule) | 1.95 | 27.7 / 7.8 | 0 | 10044 | 0 | 6232 | 2355 / 3000 | 264 | left_shoulder_yaw_link/torso_link -12.6; right_shoulder_yaw_link/torso_link -17.1 |
| g1 | after | 41 (30 capsule, 11 hull) | 1.75 | 30.6 / 10.6 | 0 | 3522 | 0 | 999 | 759 / 1000 | 36 | clear |
| openarm | before | 16 (14 capsule, 2 sphere) | 1.66 | 69.6 / 8.0 | 149302 | 146 | 79 | 1867 | 749 / 3000 | 708 | clear |
| openarm | #325 | 18 (14 box, 4 capsule) | 1.81 | 35.1 / 15.4 | 4564 | 322 | 3 | 923 | 514 / 3000 | 395 | clear |
| openarm | after | 18 (18 hull) | 1.60 | 36.0 / 15.4 | 0 | 322 | 0 | 1 | 119 / 3000 | 0 | clear |

- `vol ÷ hull` and `protrusion` are per link, averaged over the robot's links; for a refined link
  they are the **box's** (the broad phase). What a refined pair is judged on is the hull: its
  support excess over the link's hull is ≤ 1.1 mm on every arm link (mean ≤ 0.19 mm; the
  320-vertex refinement of the larger meshes), and 2.1 mm mean / 45 mm max on H1, where the hull
  also holds the twin's longer forearm (§12).
- G1 `after` is 1000 poses (a 3000-pose pass of the hull-heavy G1 takes over two hours in
  Python); every other row is 3000.

`before` is the branch at 86789ab2 (one capsule per link, #324's refit; the OpenArm's
hand-written primitives). `#325` is the OpenArm as #325 fitted it. `after` is this commit.

- **Missed contacts: 0 on all eight**, and 0 vertices and 0 surface triangles outside. The
  OpenArm's hand-written primitives missed 79 contacts in 3000 poses and left 149,302 mesh
  vertices outside; #325's fit (collision meshes only) left 4,564 visual-mesh vertices outside
  and missed 3.
- **False stops: 0-14 on the six arms** (from 684-3356). SO-100's 1979 refused poses are all
  real hull contacts: the folded arm touches itself over much of its range.
- Contact counts move with the ACM: a pair that is now checked adds its contacts; see §7.

## 7. What changed on the branch

- `urdf_lowering`:
  - `fit_link_primitives` (new): the one fitter, candidates and rule as in §3; `LinkFit` carries
    every candidate's primitives and metrics. `fit_obb_to_vertices` is now minimum-volume with
    headroom. `fit_trimmed_capsule_to_vertices` and `fit_tightest_primitive` (#325) are gone.
  - `fit_link_with_tight_geometry` (new): box + `tight_geometry` derived in the box's rendered
    (4-dp) frame, written at 1 nm, slabs rounded outward.
  - Both readers (`lower_link_geometry`, `fit_collision_geometry_from_mjcf`) hand whole meshes to
    the fitter, which returns primitives already at the manifest's 4 dp (`_rendered`), so the ACM
    certificate proves what the kernel loads; `tight_links` selects box + hull; `_tight_links_of` makes a refined link sticky on
    re-lowering. `_twin_link_meshes` adds the MJCF twin's CAD when its frames coincide.
  - The MJCF path fits by default (`--fit-mjcf-geometry` is gone): `openral collision lower`
    treats URDF and MJCF robots the same, and `acm_only` keeps the manifest geometry on both.
  - The ACM (certificate and MJCF sweep) folds a link pair over its primitives, and the MJCF sweep
    never exempts a pair whose two links carry hulls (the kernel re-asks those of the hulls), as
    the certificate already withheld.
- `envelope_loader.collision_params_from_description` lowers several primitives per link.
- `tight_geometry.py` (new): the DOP/hull derivation moved from `tools/generate_tight_geometry.py`
  (re-exported there) plus a KD-tree upper bound on the hull overhang for large CAD meshes.
- CLI: `--tight-link`, and `render_blocks` writes `tight_geometry`. The loosening guard measures a
  link's primitives as a set.
- `cumotion_config`, `sim_sensor_bridge.collision_model_mesh_slop`: several primitives per link.
- Manifests re-lowered: franka_panda, g1, h1, openarm, rizon4, so100_follower, ur5e, ur10e.
  `openarm.srdf`: the link1 ↔ link3 exemption (both arms) retired (§11).
- Tests: see the PR body; the new ones are `test_collision_tight_lowering.py`, the surface
  containment check, the per-primitive-union truth, and the multi-primitive params tests.

## 8. Hulls: where they go, where they cannot, and what would need the kernel

### 8.1 What the kernel supports today (read from source)

- `tight_geometry` refines a **box** only (`LinkCollisionGeometry` refuses it on a capsule).
- A self pair is re-asked of the hulls only when **both** of its boxes carry one
  (`refine_self_pair`). A box ↔ capsule pair uses `box_capsule_distance` on the box: the hull does
  nothing there. So a hull pays for a pair only when both links are refined.
- The voxel (world) check runs the DOP then the hull on every refined box, at most 32 stage-2
  cells per check (`kMaxStage2PerCheck`), independently of the other links.
- Several primitives per link are fine in the kernel, so a link could carry a refined box **and**
  capsules; but the pair gap is the minimum over primitives, so the capsules would bound the gap
  from below and the hull would buy nothing. A link with two capsules cannot carry a hull.

So "capsules and hull at the same time" works per link (some links capsules, others box + hull),
not within one link, and it pays on pairs of refined links.

### 8.2 The per-robot choice (measured)

Five configurations per robot, judged by the **C++ kernel itself** (§4 harness) on the same 1000
seeded in-limit poses: poses refused, and the median / max cost of one `check_self_collision`.
- `before`: 86789ab2 (one capsule per link; the OpenArm's hand-written primitives).
- `M`: the fitter alone (capsules and chains by least mean protrusion).
- `H`: `M` plus box + hull on both links of every pair that trips at rest or carries ≥ 5 %
  false stops (from a 1000-pose sweep of `M`), iterated once on the rest pose.
- `ALL`: box + hull on every link.
- **bold** = shipped.

| robot | before | #325 | M | H | H2 | ALL |
|---|---:|---:|---:|---:|---:|---:|
| franka_panda | 393 · 0.9 / 1 µs | — | 454 · 3.6 / 5 µs | 143 · 24.0 / 61 µs | — | **129 · 8.5 / 48 µs** |
| ur5e | 404 · 0.5 / 1 µs | — | 318 · 0.6 / 1 µs | 319 · 3.6 / 5 µs | — | **194 · 2.1 / 25 µs** |
| ur10e | 385 · 0.4 / 1 µs | — | 340 · 0.7 / 2 µs | 252 · 4.9 / 40 µs | — | **199 · 3.6 / 24 µs** |
| rizon4 | 198 · 0.6 / 1 µs | — | 199 · 1.2 / 2 µs | 32 · 7.4 / 58 µs | — | **21 · 2.4 / 26 µs** |
| so100_follower | 849 · 0.3 / 1 µs | — | 761 · 1.4 / 2 µs | = ALL | — | **658 · 6.5 / 47 µs** |
| h1 | 999 · 7.6 / 44 µs | — | 950 · 42.3 / 82 µs | **277 · 175.9 / 234 µs** | — | 1000 · 65.8 / 119 µs |
| g1 | 781 · 16.7 / 58 µs | — | 933 · 54.1 / 97 µs | **759 · 273.1 / 373 µs** | — | 1000 · 118.0 / 324 µs |
| openarm | 264 · 4.6 / 6 µs | 167 · 49.0 / 74 µs | 100 · 11.8 / 49 µs | 127 · 67.2 / 101 µs | 59 · 80.8 / 116 µs | **40 · 41.2 / 103 µs** |

Cells: poses refused of 1000 · median / max µs per `check_self_collision` call. SO-100's `H` refined
all five links (every pair carried ≥ 5 % false stops), so it is `ALL`. `H2` is the OpenArm's
intermediate step (link0/1/3/5 refined, link1 ↔ link3 exemption retired).

Reading:
- On the arms `ALL` beats `H` on refusals **and** cost: a hull only pays against another hull
  (§8.1), so a partly refined arm leaves box ↔ capsule pairs that are both slow and loose.
- On H1 and G1 `ALL` refuses every pose. With every link refined, the always-colliding
  certificate withholds for every pair (the kernel would re-ask it of the hulls), and the hulls
  of their CAD-nested neighbours (hip pitch ↔ hip yaw, shoulder roll ↔ torso) overlap at every
  pose. With capsules there, the certificate proves those pairs always collide and exempts them.
  So the humanoids ship `H`.
- `H1`'s left elbow is refined too (the rest-pose iteration only needed the right one) so both
  arms share one model.

### 8.3 Kernel changes this review recommends (not done; WG + ADR)

- **Follower-joint finger.** The OpenArm's second finger is equality-coupled to the first; the
  lowering folds it into `finger_pair`, swept over 9 stroke samples. The enclosure test allows a
  swept link 1 mm outside between samples. A kernel joint kind "follower of dof i with
  q = c0 + c1·q_i" would let the finger be its own link with exact geometry. Needs a new
  `JointKind` in `collision.hpp` and the envelope loader.
- **Hull against capsule.** A refined box against a capsule link is checked on the box
  (`box_capsule_distance`, 669 ns). A GJK hull ↔ capsule (the capsule is `segment ⊕ ball`,
  exactly what `convex_distance` already solves in Python) would let one refined link pay off
  against every neighbour, not only refined ones. Measure: the OpenArm link0 ↔ link3 false stops
  rose when only link3 was refined (27 → 40 per 1000) and vanished when link0 was refined too.
- **A self-collision broad phase.** `check_self_collision` runs the full narrow phase on every
  non-exempt primitive pair. A bounding-sphere test per pair (a few ns) before
  `box_capsule_distance` / `box_box_distance` would make far pairs nearly free; the humanoids'
  0.2-0.4 ms per configuration (§4) is almost all far box ↔ capsule pairs. Conservative by
  construction (a pair is skipped only when its bounding spheres are further apart than the
  margin), allocation-free, and the only change here that the humanoids' latency needs.
- None of the three changes today's verdicts; each needs a hazard-log entry and an ADR.

### 8.4 Octomap self-filter padding (noted, not changed)

#325's author reports that the octomap self-filter deletes cloud points inside each collision
primitive plus 5 cm of padding, so a primitive that protrudes past the part blinds the map by
its protrusion **plus** the padding: over-sized primitives are not conservative for the world
check. I did not find that filter in this repository (it is not in `openral_octomap_bridge`,
whose padding is for carried payloads), so I could not check which geometry it reads. If it
reads the primitives, note the direction this change moves it:
- on the humanoids the fitter lowered the mean protrusion (capsules and chains, §3);
- on the six arms the kernel now carries **boxes**, whose mean protrusion is larger than the
  capsules' (12-27 mm against 8-11 mm, §6), because the hull inside does the exact work. A filter
  that used the box would blind more than before. It should use the refined geometry — the
  hull, whose protrusion is 0 by construction plus its overhang (§6) — where one exists.

### 8.5 Draft ADR text (for `OpenRAL/management/adr/`)

> **ADR-XXXX — One primitive fitter for URDF and MJCF robots; box + exact hull where measured**
>
> *Status:* proposed. *Deciders:* Safety WG. *Date:* 2026-09-25.
>
> *Context.* PR #324 fitted URDF robots' capsules to the whole link; PR #325 fitted the OpenArm's
> MJCF meshes with a second fitter. Measured on eight robots: one capsule per link cannot clear
> the H1 and G1 rest poses, a least-volume rule refuses more poses than single capsules on four
> robots, and the OpenArm refused 16.7 % of in-limit poses.
>
> *Decision.* One fitter for both sources: per link, the least-mean-protrusion of {capsule, box,
> capsule chain of 2-3}, all containing the link's collision + visual geometry with 1 mm headroom;
> several primitives per link are lowered. Which links become a box plus `tight_geometry` is
> decided per robot on the kernel's measured refusals and cost: every link on the six arms; on
> the humanoids, the links of pairs that trip at rest or carry ≥ 5 % false stops. A refined link
> stays refined on re-lowering. The C++ kernel is unchanged.
>
> *Consequences.* 0 missed contacts and 0 uncovered surface triangles on eight robots; false
> stops fall on every robot, to 0-14 per 3000 poses on the arms (§6); H1/G1 rest poses clear;
> self-collision costs up to ~0.1 ms per configuration on the arms and 0.2-0.4 ms on the
> humanoids (§4) until the kernel gets a broad phase (§8.3); manifests grow by up to 320 hull
> vertices per refined link. SO-100's resting contact and the remaining OpenArm exemptions stay
> WG items.
>
> *Alternatives rejected.* Least volume (more refusals on four of seven robots); convex-only (no
> chains: UR5e +27 % refusals); hulls only on the worst pairs of an arm (loses to hulls everywhere
> on refusals and cost); hulls everywhere on a humanoid (refuses every pose: CAD nesting loses
> the always-colliding certificate); a smaller hull vertex budget (no cost gain, more refusals).

## 9. Hazard-log and management text needed (not written to the management repo)

- **Entry 045 (update, OpenArm under-coverage), mitigation:** the OpenArm is fitted by the same
  fitter as the URDF robots, from every MJCF geom (collision and visual) with the second finger
  swept over the stroke; 0 vertices and 0 surface triangles outside; 0 missed contacts in 3000
  poses (the hand-written primitives missed 79 and left 149,302 vertices outside). Refused poses:
  4.0 % in 1000 poses by the C++ kernel (#325's fit: 16.7 %, the hand capsules: 26.4 %), 0.8 % of them with no collision-mesh contact (#325: 15.8 %).
  - Residual: the swept finger may poke up to 1 mm out between the 9 stroke samples (enclosure
    test tolerance for swept links); cameras and cables absent from the MJCF are not modelled.
- **Entry 046 (update, URDF under-coverage), mitigation:** unchanged in kind (collision ∪ visual,
  1 mm headroom), now also the **surface** proven inside (not just vertices), and the MJCF twin's
  CAD folded in where its frames coincide (H1, Franka, Rizon 4). H1's twin forearm reaches 46 mm
  and its ankle 25 mm past the URDF CAD; the kernel now covers both.
- **Entry 043 (update, loss of conservatism):** 0 missed contacts on all eight robots over 3000
  poses; false stops and refused poses fall on every robot against the pre-refit geometry (§6).
- **New entry: box + hull links (per-robot list in §8.2).** The kernel's hull narrow phase
  (Entry 018) now runs on self pairs of eight robots, not only panda_mobile. Hazard: a hull is
  exact (no headroom), so its correctness rests on the containment proof
  (`mesh ⊆ hull ⊆ DOP ⊆ box`, derived in the rendered frame, written at 1 nm, re-validated by the
  kernel at configure and by the enclosure tests on every manifest). Cost: up to tens of µs per
  configuration (§4), inside the 10 ms budget.
- **Exemption change (OpenArm):** link1 ↔ link3 retired on both arms (now checked on the hulls,
  ≥ 12 mm apart). Remaining exemptions (finger_pair ↔ link5/link6, link5 ↔ link7) re-measured in
  §11; the WG keeps or retires them.
- **SO-100 rest (unchanged decision needed):** the folded q = 0 is a resting contact; the kernel
  refuses motion from rest with every conservative geometry.
- **G1 twin frames:** the menagerie G1 twin's body frames sit 10 mm off the URDF link frames at
  q = 0 (UR5e/UR10e: different frame convention, 100+ mm). The kernel's FK is the URDF's, so in
  `deploy sim` the kernel and the simulated G1 disagree by up to 10 mm. Not fixed here (§12).
- **New entry: uncovered grippers (pre-existing, found by the generalised twin test).** Franka's
  fingers and SO-101's gripper/jaw have no collision primitive, so the kernel cannot see them hit
  the arm (§12). Mitigation owed: fit them like the OpenArm's finger pair.
- **ADR:** the §8.5 draft. The three kernel changes in §8.3 each need their own entry and ADR.

## 10. The two fitters, unified

PR #325 (`fix/openarm-collision-refit`) fitted the OpenArm with `fit_trimmed_capsule_to_vertices`
(the PCA capsule this PR had replaced) against a PCA box, on the MJCF collision meshes only,
behind `--fit-mjcf-geometry`. Its two commits were cherry-picked here (62fa002f, 58356ab1) and
then folded into the one fitter: the trimmed capsule and `fit_tightest_primitive` are deleted,
the MJCF reader hands its meshes (collision + visual, finger swept) to `fit_link_primitives`,
and the flag is gone. #325's enclosure test (MuJoCo FK for the meshes, kernel FK for the
primitives) now runs on every robot with an MJCF twin.

## 11. OpenArm

Three OpenArm models, same kernel harness and truths (§4, §6):

| | hand-written (86789ab2) | #325 fit (58356ab1) | this PR |
|---|---:|---:|---:|
| primitives | 16 (14 capsule, 2 sphere) | 18 (14 box, 4 capsule) | 18 box + hull |
| vertices outside (collision + visual) | 149,302 | 4,564 | 0 |
| missed contacts, 3000 poses (hull truth) | 79 | 3 | 0 |
| poses refused, 1000 (C++ kernel) | 264 | 167 | **40** |
| false stops, 3000 poses | 1,867 | 923 | **1** |
| `check_self_collision` median / max | 4.6 / 6.1 µs | 49 / 74 µs | 41 / 115 µs |

#325 fitted the collision meshes only; the visual meshes leave its primitives by 4,564 vertices,
and 3 sampled contacts fall in that gap. Folding the visual geometry in (§3) closes it.

**False-self-stop split** (1500 seeded poses; truth = the MJCF collision meshes placed by MuJoCo,
which MuJoCo compiles convex, measured with the certified GJK/SAT of
`openral_hal.convex_distance` — #325's own instrument):

| | refused | with a real collision-mesh contact | primitive slop only |
|---|---:|---:|---:|
| #325 fit | 280 (18.7 %) | 43 | 237 (15.8 %) |
| this PR | 55 (3.7 %) | 43 | 12 (0.8 %) |

The real-contact count is the same because it is the robot's; everything else was slop, and
95 % of it is gone. The remaining slop is the finger pair against link0/link1 of either arm
(hull ↔ hull at the swept finger's hull, which covers the whole stroke).

**Exemption margins** (`openarm.srdf`, same instrument, minimum and 1st percentile over 1500
poses):

| pair (both arms) | min | p1 | status |
|---|---:|---:|---|
| link1 ↔ link3 | 14.8 mm (#325) | — | **retired**: both links now carry hulls, whose exact (collision + visual) hulls stay ≥ 12.1 / 12.3 mm apart over 500 poses (+16.6 mm at rest), so the kernel checks the pair |
| finger_pair ↔ link5 | 16.2 / 16.2 mm | 16.9 / 17.0 mm | kept |
| finger_pair ↔ link6 | 23.8 / 23.8 mm | 23.9 / 23.9 mm | kept |
| link5 ↔ link7 | 2.95 / 2.93 mm | 3.02 / 3.01 mm | kept; the thin one |

The kept rows are pairs whose collision meshes never touch but whose convex covers always do:
finger_pair ↔ link6 and link5 ↔ link7 sit either side of the wrist, where every convex set
around each link (hull included) overlaps the other at every pose. Only a non-convex
representation (a decomposition of each link, or the follower-joint finger of §8.3 with the
finger as two links) could check them. The WG keeps or retires them.

Per link (support excess over the link's hull, box = the kernel's broad phase):

| link | pr325: kind / vol ÷ hull / max·mean mm | after: kind / vol ÷ hull / max·mean mm |
|---|---|---|
| openarm_left_link0 | box / 1.51 / 30·14 | hull / 1.43 / 29·14 |
| openarm_left_link1 | box / 1.55 / 34·19 | hull / 1.56 / 35·19 |
| openarm_left_link2 | capsule / 2.28 / 38·10 | hull / 2.12 / 51·21 |
| openarm_left_link3 | box / 1.18 / 22·9 | hull / 1.42 / 26·13 |
| openarm_left_link4 | box / 2.24 / 46·24 | hull / 1.33 / 24·11 |
| openarm_left_link5 | box / 1.55 / 32·13 | hull / 1.66 / 30·14 |
| openarm_left_link6 | capsule / 1.93 / 15·5 | hull / 1.55 / 21·10 |
| openarm_left_link7 | box / 1.54 / 40·15 | hull / 1.71 / 52·17 |
| openarm_left_finger_pair | box / 2.54 / 58·29 | hull / 1.64 / 55·20 |
| openarm_right_link0 | box / 1.51 / 30·14 | hull / 1.43 / 30·14 |
| openarm_right_link1 | box / 1.55 / 35·19 | hull / 1.56 / 35·19 |
| openarm_right_link2 | capsule / 2.28 / 38·10 | hull / 2.12 / 51·21 |
| openarm_right_link3 | box / 1.18 / 22·9 | hull / 1.42 / 26·13 |
| openarm_right_link4 | box / 2.24 / 46·24 | hull / 1.33 / 24·11 |
| openarm_right_link5 | box / 1.55 / 32·13 | hull / 1.66 / 30·14 |
| openarm_right_link6 | capsule / 1.93 / 15·5 | hull / 1.55 / 21·10 |
| openarm_right_link7 | box / 1.54 / 40·15 | hull / 1.71 / 53·17 |
| openarm_right_finger_pair | box / 2.54 / 58·29 | hull / 1.64 / 55·20 |

The hulls themselves sit on the links: support excess 0.00-0.28 mm (the 320-vertex refinement of
the larger meshes) and a sampled overhang past the mesh surface of 10-48 mm (concavities the hull
bridges, e.g. link2's cut-out).

## 12. MJCF twins

The enclosure test now places every twinned robot's meshes by MuJoCo and its primitives by the
kernel's FK. What it found:
- **H1:** the menagerie twin's CAD is not h1_description's: the forearm (elbow link) reaches
  0.316 m against 0.270 m, the ankle is ±40 mm wide against ±15 mm, and the torso carries a logo
  mesh. H1 runs only in sim, so the twin is the robot the kernel guards; its meshes are now part
  of the fit (`_twin_link_meshes`). Its twin's primitive geoms (arm capsules) are the
  simulator's contact proxies and reach up to 145 mm past the CAD; they are not fitted.
- **Franka, Rizon 4:** frames coincide; twin meshes folded in (they change little).
- **G1 (10 mm), UR5e / UR10e (102 / 116 mm, a different frame convention), SO-101:** frames differ,
  so the twin's meshes are not folded in and the twin test skips with the measured offset. The
  URDF enclosure test covers the geometry they were lowered from. G1's 10 mm is a kinematic
  disagreement between the sim twin and the URDF, which the kernel uses (§9).
- **SO-100:** the twin names no body for the manifest's links; skipped, URDF test covers it.
- **Franka's fingers are in no primitive (pre-existing).** The manifest's `panda_finger_pair`
  is a synthetic link (a prismatic joint off the hand) with no URDF counterpart, so the URDF
  lowering fits nothing to it; the twin maps it to the left finger, whose mesh (and the coupled
  right finger's) reaches ~50 mm past the hand. The twin test is a strict xfail with that
  reason. Covering it means the OpenArm's follower-finger sweep on a URDF-lowered robot, a
  follow-up for the WG list.
- **SO-101** (hand-authored manifest, not re-lowered here): `gripper_base` and `moving_jaw`
  carry no primitive; strict xfail, same follow-up.
