# The live world map, measured

What is actually in `/openral/world_voxels`, how it differs from the
geometrically ideal grid the collision studies score against, and how the three
world-side terms apportion once the real map is in the loop.

Companion to the [start-state census](robocasa-start-state-census.md) (which
limb stops a kitchen) and to [tight link geometry](collision-tight-geometry.md)
(what better link primitives could recover). Both score against a grid built
from true surfaces and both flag that as unmeasured against the live octomap.
**This page measures the live octomap.**

> **Superseded in part (2026-08-25).** The largest world-side term this page
> isolated — the octree→grid cube-overlap dilation, [issue #173](https://github.com/OpenRAL/openral/issues/173),
> 48 % of the live start-state stops — **no longer exists**. It was not narrowed:
> the overlap rule was already the *minimum sound* cover for a base-aligned
> grid, so the grid stopped being base-aligned. `/openral/world_voxels` is now
> published on the OctoMap's own lattice, one cell per octree cell, with the
> rotation carried in `OccupancyVoxels.orientation`, and the published obstacle
> set is the octree's occupied volume exactly.
>
> Two directions this page's issue floated were measured and rejected first:
> snapping the grid origin when the yaw is ~0 (works only at exact multiples of
> 90° — half a degree restores the full 25 mm, so it is a sim-only knife-edge),
> and shrinking the effective leaf half-extent (at 0.75× it buys 12 % of the
> dilation and leaves 9.9 % of the leaf uncovered in the worst case — protection
> given up, not a phase fixed).
>
> Everything else on this page stands as measured, including the quantisation
> and sensor-phantom terms and the non-collidable-geometry finding
> ([#174](https://github.com/OpenRAL/openral/issues/174)). The apportionment
> below is the state of the map **before** that change.

## The question

[PR #161](https://github.com/OpenRAL/openral/pull/161) closed on three
world-side terms and asked for the third to be characterised:

- **quantisation** — a 25 mm cell reaches up to 21.65 mm past the surface it was
  built from;
- **lattice phase** — a measured 20.3 mm swing in the kernel's reported distance
  depending purely on where the grid's cells fall;
- **map fidelity** — [PR #160](https://github.com/OpenRAL/openral/pull/160)
  found two of four characterised stops sitting on cells containing nothing at
  all, one with solid cabinet geometry a single cell away.

The first two mean the grid is *right and coarse*. The third means the grid is
*wrong*, and no link geometry helps. This page establishes how often each
direction of wrongness occurs on the real map, and what that does to #161's
finding that 45 of 72 stops lie beyond the reach of any link geometry.

## The instrument, and what validates it

Real components throughout (§1.11). One process holds the whole chain, so the
map and the geometry that produced it are snapshotted at the same instant:

- the production `ManifestHALLifecycleNode`, driven UNCONFIGURED → INACTIVE →
  ACTIVE in-process, owning the real `SimAttachedHAL` + `SimSensorBridge` and
  rendering depth from the real MuJoCo model;
- the real `octomap_server_node` (`ros-jazzy-octomap-server`), parameterised
  **verbatim** from `sim_e2e.launch.py`'s `hal_mode="sim"` branch — resolution
  0.025 m, `occupancy_thres` 0.8, `sensor_model.miss` 0.4, `sensor_model.max`
  0.85, `sensor_model.max_range` 4.0, `filter_speckles` true, frame `odom`;
- the real `openral_octomap_bridge/octomap_voxel_bridge` — `base_link`, a 1.6 m
  box, 0.025 m, publishing `/openral/world_voxels`.

Nav2, SLAM, the detector and the policy are absent: none of them writes
`/openral/world_voxels` (the world-state object-lift *subscribes* it), and a
start-state measurement applies zero actions.

The C++ safety kernel could **not** be built on this host at the time — the
vendored `opentelemetry_cpp_vendor` failed at its `find_package(Protobuf)`.

> **Corrected 2026-08-26.** That was never a conflict "against the system
> protobuf": protobuf's *dev* files were simply not installed. With
> `libprotobuf-dev` + `protobuf-compiler` present the vendor package configures
> and the whole 27-package graph builds, kernel included. (A second, self-
> inflicted variant of the same symptom: putting `.venv/bin` on `PATH` shadows
> system cmake 3.28 with the venv's pip-installed cmake 4.1, whose
> `FindProtobuf` then misses it too.) Nothing on this page depended on the
> kernel being unbuildable — the numpy port was validated against the shipped
> C++ either way — but the diagnosis should not be carried forward as a known
> breakage, because it is not one.

Kernel verdicts are therefore a numpy port of
`box_box_distance` (`collision.cpp:327`) treating each occupied cell as a
12.5 mm-half-extent cube exactly as `check_voxel_collision`
(`collision.cpp:625`) does. **The port was validated against the shipped C++,
not assumed**: `collision.cpp` was compiled into a bare oracle and fed 4 000
random OBB↔cell pairs — maximum disagreement **4.4 × 10⁻¹⁶ m**.

Mesh distances are **densely sampled** (5 mm on both surfaces; boxes,
cylinders, spheres and subdivided mesh triangles), never `mj_geomDistance`
(standing caveat 8). Sampled point-to-point is an upper bound on the true
surface gap with error bounded by the step.

> **An exact alternative now exists** (2026-08-25, after this page was
> measured): `openral_hal.convex_distance` returns a *certified* signed distance
> — GJK with a separating-axis optimality proof, exact SAT for penetration —
> rather than a sampled upper bound. Nothing on this page needs redoing: a
> sampled upper bound is conservative in the safe direction, and the two agree
> wherever they have been compared. It is the better instrument for any future
> pass, and it removes the step-size caveat above. The two studies converge:
> this page names the *mechanism* behind the displaced cells, and
> [the validation-evidence ledger](collision-validation-evidence.md#2026-08-25--the-ruler-was-wrong-and-here-is-what-it-moves)
> eliminates the competing explanation for them by certifying the distances —
> including reaching this page's own conclusion about the fridge `0.000 m`
> reading independently.

### Two instrument bugs, and one retraction

Both silently invert results, and both are worth naming because any future
reconstruction will hit them:

1. **`robot_self_body_ids` needs the manifest's `sim_joint_name`s.** The
   rollout exposes no such attribute; passing an empty sequence returns an
   empty self-filter, after which every robot geom counts as "world" and each
   link measures **itself** as its nearest obstacle at 0.000 mm.
2. **The grid frame body must be resolved *with* the description.**
   `resolve_base_frame_body_name(model, description=...)` gives
   `mobilebase0_support` (world z = 0.700) — the body the live bridge uses.
   Without the description it falls back to `mobilebase0_base` (world z = 0.000):
   a silent **0.7 m** vertical offset applied to every cell.

Under bug 2 this investigation briefly concluded that the live map was missing
the freezer door the arm was touching by 464 mm, and that three of the seven
kernel-checked links sat outside the checked volume. **Both were artefacts of
the offset and are retracted.** With the frame corrected, every link's nearest
solid surface *is* in the map, 10.2–20.2 mm from the nearest cell centre —
inside the 21.65 mm half-diagonal, which is exactly what quantisation predicts.

### Cross-validation against independently reported numbers

Nothing below rests on the instrument alone:

| quantity | measured here | independently reported | source |
| --- | ---: | ---: | --- |
| fridge, unpinned seed 1, live kernel min | **−24.75 mm** | −24.7 mm (`voxel_169769`) | #154 field round |
| fridge at `layout_ids: [30]`, kernel min | **−23.47 mm** (live) | −23.47 mm (ideal grid) | census |
| unpinned seed-1 draw | layout 29 / style 20 | layout 29 / style 20 | #154 |
| `panda_link7` ↔ freezer door, mesh gap | **2.819 mm** | 2.5 mm | #154 ground truth |
| `panda_link6` ↔ freezer door, mesh gap | **16.262 mm** | 16.1 mm | #154 ground truth |

The two sampled mesh gaps also settle what the later `0.000 m` reading on that
pair was: `mj_geomDistance`'s documented failure mode, not a touch.

### Re-verified after rebasing

The headline numbers were **re-derived, not carried forward** — first at
`a0f3d58`, then again on the master that carries this page. The numpy port
re-validated against each tree's `collision.cpp` at **4.441 × 10⁻¹⁶ m** over the
same 4 000 pairs, and every verdict re-scored from the stored captures unchanged
at both revisions:

| state | captures (occupied cells) | kernel min |
| --- | --- | ---: |
| layout 30 | 5 427 and 8 239 | **−23.47 mm** on both |
| layout 47 | 5 435 and 9 112 | **+19.34 mm** on both |
| unpinned (29 / 20) | 7 411 | **−24.75 mm** |

None of [#164](https://github.com/OpenRAL/openral/pull/164) (CI only),
[#165](https://github.com/OpenRAL/openral/pull/165) (`CollisionShape` as a
discriminated union) or [#169](https://github.com/OpenRAL/openral/pull/169)
(ACM exemptions; zero manifest pairs added or removed) moves them — which is
what one would expect, and is why it was checked rather than assumed.

### A natural replicate: map size moves, the verdict does not

Two independent live captures of the *identical* start state (fridge layout 30),
in separate processes on separate ROS domains, integrating different numbers of
depth frames:

| capture | occupied cells | kernel verdict |
| --- | ---: | --- |
| A | 5 427 | `panda_link2` −23.47 mm |
| B | 8 239 | `panda_link2` −23.47 mm |

The map's **size** varies by 52 % with observation time; the **verdict** does not
move at all. The surface that trips the kernel is seen early and stays. This is
the check that a live measurement of this kind is worth making: it is not a
sample of a noisy process.

## The live map is *more* conservative than the ideal grid, not less

61 live captures of `robocasa_fridge_drawer` — the scene that owns 56 of the
census's 72 stops — each a real octomap built from real depth over ~22 s of
observation, then scored with the validated port. 59 layouts have both a census
verdict and a live verdict:

| | census (ideal grid) stops | census clears |
| --- | ---: | ---: |
| **live map stops** | **55** | **2** |
| **live map clears** | **0** | 2 |

The live map reproduces **every** ideal-grid stop, adds two more, and on the 55
shared stops reports a kernel minimum that is on median **13.04 mm more
negative** than the ideal grid's (range −63.47 mm to +0.09 mm). 49 of the 55
agree on the dominant link.

This inverts the assumption both prior studies carry. The census states that the
live map "can only ever be a *subset*" of the ideal grid, so the ideal grid
bounds the live kernel from the conservative side. **It does not.** The live map
contains cells the ideal grid cannot: the ideal grid is built from
`contype`/`conaffinity`-bearing surfaces only, while the depth camera renders —
and octomap therefore records — every *visible* geom, including the 713 of 1 714
geoms in this scene that are non-collidable decoration. The bridge then dilates
whatever the octree holds (below). Both terms add occupancy that no true-surface
grid contains.

So the ideal-grid census is not a conservative bound on live behaviour. On this
population it is an *optimistic* one.

## The fridge pin, re-decided against the live map

The census recommended re-pinning `robocasa_fridge_drawer` away from
`layout_ids: [30]`, which clears the *mesh* criterion (+37.0 mm) but not the
kernel one (−23.47 mm), and named layouts **18, 21, 43, 47** as clearing the
kernel on the ideal grid, with 18 and 21 also clearing the mesh criterion. It
asked for a live round before acting. This is that round:

| layout | live kernel min | verdict |
| ---: | ---: | --- |
| unpinned (29 / 20) | `panda_link7` −24.75 mm | STOPS — the original defect, reproduced |
| **30** (shipped) | `panda_link2` **−23.47 mm** | **STOPS** — census reproduced exactly |
| 18 | `panda_link2` −2.26 mm | **STOPS** — census called it clear |
| 21 | `panda_link7` −10.96 mm | **STOPS** — census called it clear |
| 43 | `panda_link7` +0.69 mm | clears by 3 % of one voxel |
| **47** | `panda_link1` **+19.34 mm** | **clears** |

**Layouts 18 and 21 are the trap.** They are exactly the two the census
recommended, and the live map refutes both. Pinning either would have repeated
#154's mistake in a new form — a pin verified against a criterion that does not
hold where the robot actually runs. That is the concrete reason a live round was
worth the cost.

**Layout 47 is pinned.** It is the only candidate that clears on all three
criteria and does not move when the map changes:

- live kernel **+19.34 mm**, *identical* across two independent captures whose
  maps held 5 435 and 9 112 occupied cells — 68 % more map, 0.00 mm of verdict
  change;
- mesh clearance **48.40 mm** (`panda_link1` vs
  `fridgesidebyside_main_group_1_fridge_drawer3`, densely sampled), larger than
  layout 30's 37.0 mm, so it satisfies the older criterion too and by more;
- every kernel-checked link clears; the next worst is `panda_link7` at
  +77.46 mm.

Layout 43 was rejected deliberately: +0.69 mm is 3 % of one voxel, and it
reproduced at +0.69 mm on a second, denser capture — stable, but with no
headroom to spend. A pin is meant to buy margin, not to sit on the boundary.

Two consequences that travel with the pin, both handled in the same commit:

- `place_declaration.target_id` is **a function of the pin**. At layout 30 the
  fridge was `fridge_main_group`; at layout 47 the scene composes to layout 47 /
  style 32 and the fridge is `fridgesidebyside_main_group_1`, so the old target
  named no body. The HAL fails closed on that, which means the symptom of
  forgetting is a silently unarmed place phase, not an error.
- Pinning changes the RNG **draw order**, so layout 47 pinned is not the kitchen
  an unpinned run that drew 47 would produce. Both scene files already say so;
  it is repeated here because it governs how the table above may be read.

### The utensil scene

`robocasa_drawer_utensil` keeps `layout_ids: [3]`, but its header's reasoning is
corrected. It asserted that "43.3 mm clears the 25 mm occupancy grid, so the arm
does NOT start inside the kernel's world model" — the same invalid inference
#154 made, since `panda_link1` needs `53.40 + 21.65 = 75.05 mm`, not 25 mm. And
at its shipped pin the live map puts it at **−1.94 mm**: it *does* take a
marginal initial-configuration stop, which the ideal-grid census did not see.

It is left pinned because that stop is **entirely link-side**: re-scored against
the same live map with exact link geometry the state reads **+20.01 mm**. The
staged tighter-primitive work ([#166](https://github.com/OpenRAL/openral/pull/166))
removes it with no scene change, whereas a re-pin would churn the whole kitchen
to work around a link-model artefact.

## Why the live map holds cells no true-surface grid contains

Two mechanisms, both in the shipped path, both measurable, neither previously
costed.

### 1. Non-collidable geometry is real to the camera

Tracked as [#174](https://github.com/OpenRAL/openral/issues/174).

The ideal grid is built from geoms with `contype` or `conaffinity` set. The
depth camera has no such filter — `mj_multiRay` strikes whatever is visible, and
the synthesised cloud back-projects it. In the fridge scene **713 of 1 714**
world geoms are non-collidable decoration, 42 %. Every one the camera can see
becomes occupancy the kernel treats as an obstacle and against which no
collision can ever occur; in the whole-map census below, **18.8 %** of occupied
cells are backed only by such a geom.

**This is the unfinished half of
[#149](https://github.com/OpenRAL/openral/pull/149).** That PR established that
a geom with neither `contype` nor `conaffinity` cannot collide with anything,
and taught the **ground-truth probe** to exclude those geoms — a `0.000 m` pair
against a visual mesh can no longer be promoted to a `real_contact`. Nothing
does the same for the **perception path**: the depth synth, octomap and the
bridge carry visual-only geometry straight into `/openral/world_voxels`, and the
kernel stops on it. The two halves of the stack disagree about what counts as
world geometry — the adjudicator is *required* to ignore a surface the
perception leg is *required* to voxelise.

It is worth naming as simulation-specific: on real hardware everything a depth
camera sees *is* physical, so this term has no hardware counterpart. It cuts in
the conservative direction, so nothing is unsafe today. What it does distort is
every sim-derived number the collision programme rests on — a sim round's stop
rate is not a prediction of a hardware round's stop rate, which is exactly what
the four-scene validation matrix is used for.

#### Costed, and closed

The filter #174 asked for is now in the depth synth: `noncollidable_geom_ids`
is hidden from every `mj_ray` pass, so a return is always a surface something
can collide with. What that costs was measured first, per ray, on all four
validation-matrix scenes at the deploy stride — the real camera, the real
built scene, one cast with every visible geom and one with intangible
geometry filtered:

| scene | geoms | intangible | returns changed | lost | median push-back | max | **nearer** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baguette | 1 763 | 714 (40 %) | 30.2 % | 0 | 7.9 mm | 59.6 mm | **0** |
| sink_cup | 1 539 | 617 (40 %) | 30.8 % | 0 | 0.9 mm | 79.7 mm | **0** |
| fridge | 1 798 | 689 (38 %) | 45.1 % | 0 | 0.0 mm | 814.9 mm | **0** |
| utensil | 1 472 | 586 (40 %) | 19.8 % | 0 | 0.6 mm | 367.6 mm | **0** |

Three things this settles.

**The direction is safe by construction, and measured to be.** Filtering
removes candidates from `mj_ray`'s nearest-hit search, so a return can only
move farther along its ray or vanish; a collidable surface is never among the
removed candidates and stays hittable. Over 16 384 rays, no return anywhere
came back nearer, and none was lost — every ray still finds collidable
geometry. Along a ray, no collidable surface can be cleared either: if one lay
between the removed decoration and the new endpoint, the ray would have
stopped on *it*.

**Countertops do not drop.** The concern recorded in `depth_camera.py` — a
visual-only RoboCasa countertop at world z 0.920 over a collision carcass at
0.890, so filtering would lower the perceived surface by 30 mm — does not
occur. All **ten** `*_top_visual` geoms sampled across the four scenes move by
a maximum of **0.0 mm** and lose no ray: each has a collidable top geom at the
same height. That concern was about `mj_multiRay`'s body-level BVH cull, which
is a different mechanism from per-geom collidability.

**The change that has teeth is the fridge interior.** 234 of 4 087 fridge rays
travel more than one 25 mm voxel further, most of them 200–800 mm, once the
intangible door panel `fridgesidebyside_main_group_1_g17` stops returning.
That is the intended effect: the interior really is open, and it is the volume
the arm has to enter. Elsewhere the median push-back is under 2 mm — well
inside the 21.65 mm voxel half-diagonal, so most cells do not move at all.

Two consequences worth carrying forward. The two `test_depth_camera_synth.py`
tests that asserted a visual-only countertop *must* be the reported surface
were inverted deliberately, and say so; their scene is kept because it is the
adversarial one. And since the `mj_multiRay` body cull only ever mis-skipped
non-collidable geoms, the ~1.9x premium the synth pays for `mj_ray` may now be
recoverable — unverified, and deliberately out of scope here.

#### The premium is not recoverable, and the reason is worse than #174 thought

Tracked as [#195](https://github.com/OpenRAL/openral/issues/195), which
measured the paragraph above rather than acting on it. **The conjecture is
false.** `mj_multiRay` does not only mis-skip non-collidable geoms; with the
intangible filter applied to *both* casters, so they see the identical world,
every geom it skips is **collidable**.

Same protocol as the filter measurement — the four validation-matrix scenes,
built through `build_sim_env_from_yaml`, the `panda_mobile` `front_depth`
intrinsics rescaled to each scene's 512² render, `stride=4` → 16 384 rays per
scene, and the shipped filter set (intangible geometry + the robot's own
bodies) hidden from both:

| scene | geoms | rays disagreeing | geoms skipped | of those, intangible | max **farther** | **nearer** | lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baguette | 2 383 | 3 815 (23.3 %) | 3 | **0** | 434.2 mm | **0** | 0 |
| sink_cup | 2 380 | 39 (0.2 %) | 2 | **0** | 480.2 mm | **0** | 0 |
| fridge | 1 541 | **0** | 0 | — | — | **0** | 0 |
| utensil | 1 502 | 259 (1.6 %) | 6 | **0** | 205.9 mm | **0** | 0 |

Four things this settles.

**The skipped geoms are the ones a robot hits.** On the baguette scene the
batched cast walks through `counter_1_left_group_top_left_0` and `_1` —
`contype=1 conaffinity=1` countertop slabs, the surface the whole task reaches
onto — and answers with `stack_1_left_group_3_top`, the cabinet doors and
`wall_left_4_room_g0` behind them. This is the same *shape* of error #111
described, but not the same cause: #111 blamed visual-only geometry, and after
#174 there is none left in the cast.

**The direction is uniformly unsafe.** Across all 65 536 rays not one return
came back nearer and not one was lost; 4 113 came back *farther*, by up to
480 mm. A depth cloud that reports free space where a countertop is, feeding
`octomap_server` and then the kernel's world grid, is the one error mode this
whole page exists to bound. #174's filter was safe *because* it could only push
returns farther from a surface that was not there; this pushes them past a
surface that is.

**`mj_ray` is the correct one, adjudicated independently.** An analytic
ray/box slab intersection sharing no code with MuJoCo's caster was run over
200 sampled disputed rays: it agrees with `mj_ray` on **200** and with
`mj_multiRay` on **0**. The batched call is wrong, not merely different.

**The fridge's zero is a property of that camera, not a reprieve.** Its
`front_depth` looks at the fridge front at close range — median return 1.28 m,
`..._freezer_door_main` alone taking 4 711 of 16 384 rays — with no counter run
in view for the cull to fire on. One scene over the same caster loses 3 815
rays.

**The mechanism is open, and #111's account of it is not it.** #111 recorded
that MuJoCo builds a body's BVH from collision geoms only, so a visual geom
lying outside its body's collision extent is skipped. That cannot explain this:
every geom skipped here is collidable and so is inside its own body's collision
extent, and the batched cast still returns 184 rays on
`counter_1_left_group_main` — the very body whose countertop it walks through —
so it is not culling the body wholesale either. #195 also could not reproduce
the skip in a hand-written MJCF: far-outlier collision geoms in the same body,
body yaw and geom yaw all agree ray for ray. Whatever scene property triggers
it is not isolated, which is why the gate has to run on the real scenes and
there is no unit-tier version of it. **This is the open thread to pull if the
premium is ever worth another attempt** — the swap cannot be justified on a
mechanism nobody has pinned down.

The cost that stays paid is larger than #111's ~1.9×: **4.2–6.2×** on these
scenes (83–129 ms per pass per-pixel against 13.5–22.3 ms batched, q-laptop,
CPU cast), because a real kitchen fires the cull far more often than the
1200-geom clutter scene #111 measured on. `tests/sim/safety/test_depth_multiray_equivalence_robocasa.py`
holds both halves as a gate: the shipped caster is adjudicated against the slab
test (so a swap fails), and the per-scene disagreement counts are asserted (so a
change in upstream MuJoCo's cull re-opens #195 rather than silently licensing
the swap).

### 2. The octree→grid bridge dilates by up to one cell, by design

Tracked as [#173](https://github.com/OpenRAL/openral/issues/173) — it turns out
to hold 48 % of the live stops measured below, which is why it gets its own
issue rather than a line in this page.

`rasterize_octree_to_grid` (`octree_to_grid.cpp`) marks a base-frame cell when
its cube **shares volume** with an occupied octree leaf's cube. The two lattices
have the same resolution but an arbitrary relative phase — and, on a mobile
base, an arbitrary relative yaw — so a single leaf generically overlaps up to
eight base cells. The rule is deliberate and it is the safe direction: the
comment above it records that the previous centre-sampling rule displaced
surfaces by a whole voxel toward the robot. But it is not free, and it has never
been costed alongside the 21.65 mm quantisation term and the 20.3 mm lattice
swing.

A cell whose **centre** lies inside no occupied octree leaf exists solely
because of this rule. That is directly measurable by decoding
`/octomap_binary` alongside the published grid, and the census below reports it.

### What the whole map is made of

Both mechanisms show up in a census of the map itself. A uniform sample of 1 200
of the 9 667 occupied cells at the `robocasa_baguette` layout-7 state, each put
to the 243-ray `voxel_backing_record` probe and independently checked against
the octree:

| what the cell contains | n | share | centre in an octree leaf |
| --- | ---: | ---: | --- |
| `solid_world` — real collidable geometry | 243 | 20.3 % | 180 / 243 |
| `noncollidable_world` — visual-only geometry | 225 | 18.8 % | 198 / 225 |
| `unbacked` — nothing at all | 731 | 60.9 % | 359 / 731 |
| `self_occupancy_suspect` | 1 | 0.1 % | 0 / 1 |

**463 of the 1 200 cells (38.6 %) have centres in no occupied octree leaf** —
they are the bridge's dilation, present in the grid the kernel reads and absent
from the octree it was built from.

Read the 60.9 % carefully: it is **not** an error rate. A correct map of a solid
surface necessarily contains cells that lie just behind that surface and so hold
no surface themselves — an octree leaf is a cube, and the surface it was built
from touches only part of it. What the whole-map census establishes is the
*scale* of the two mechanisms; the tripping-cell analysis below is what
establishes their *consequence*, because only cells a link can reach change a
verdict.

### What octomap's own thresholds can and cannot explain

Analytic, from the shipped sim parameters — `prob_hit` 0.7, `prob_miss` 0.4,
`occupancy_thres` 0.8, `sensor_model.max` 0.85:

| event | log-odds | vs threshold 1.386 |
| --- | ---: | --- |
| one hit | 0.847 | below — not yet a safety voxel |
| two hits | 1.694 | above — becomes one |
| confirmed cell, then one clearing ray | 1.289 | below — free again |

The thresholds behave exactly as their comment in `sim_e2e.launch.py` claims:
two frames to create a safety voxel, one clearing ray to remove it. **So a
persistent phantom cannot be a threshold artefact.** It can only survive where
no ray ever passes — behind the first surface a ray strikes, or outside the
frustum. That rules out the "spurious sensor noise" hypothesis for anything
that lasts, and it rules out octomap's clearing behaviour as a cause.

The bridge's payload-clearing path is likewise not implicated at a start state:
nothing is attached, so `clear_attached_payload_cells` operates on an empty
attachment set.

## Apportioning a live stop

#161 asked how the 45 world-held stops divide across quantisation, lattice
phase and map fidelity. The apportionment below is done differently, and the
difference matters: rather than reasoning about what *would* remain if link
geometry were perfect, it **measures both verdicts against the same live map**.

For each start state, two numbers on the same published occupancy array:

- **OBB** — what the kernel computes today: `box_box_distance` between the
  manifest's one box per link and each occupied cell cube.
- **EXACT** — the minimum distance from the link's *own collision surface* to
  each occupied cell cube. MuJoCo convexifies mesh collision geoms, so the
  sampled surface is the convex hull's surface up to the sampling step: this is
  the quantity #166's staged 26-DOP → exact-hull pipeline reaches, and for
  `panda_link2` that PR measures the remaining support excess at 0.00 mm.

The scan is a branch-and-bound — `box_box_distance` on a box containing the hull
lower-bounds the exact hull distance, so cells are visited in increasing OBB
distance and the scan stops when that bound exceeds the best exact distance
found. It is exact, and it visits 2–46 cells instead of thousands.

**A stop that EXACT clears is link-side. A stop that EXACT keeps is world-side
by construction.**

### The split

Measured on **23 live stops** across all four scenes (27 states scored; a
sample of the 62 stopping captures, not the full population):

| | n | share |
| --- | ---: | ---: |
| **link-side** — exact geometry clears it | **6** | 26 % |
| **world-side** — exact geometry still stops | **17** | 74 % |

The link-side excess removed by exact geometry is large and matches the census's
corner-slop analysis: median **28.72 mm**, and on `panda_link2` up to 42.56 mm
against that link's 48.22 mm corner slop.

### Splitting the world-side residue, and the mechanism behind it

For each world-side stop, the cell that stops exact geometry was put to two
independent tests: the repo's own `voxel_backing_record` at 243 rays, and
whether its **centre** lies inside an occupied leaf of the octree the bridge
rasterised (decoded from the same capture's `/octomap_binary`).

The two agree on **17 of 17**, and that agreement is the finding:

| tripping cell | n | centre in an octree leaf? | nearest real solid surface |
| --- | ---: | --- | --- |
| `solid_world` — real collidable geometry inside it | **5** | **always** (5/5) | ≤ 13.59 mm |
| no world geometry inside it | **12** | **never** (0/12) | 16.64–299.28 mm, median 21.03 |

`voxel_backing_record` ranks `solid_world` above `self_occupancy_suspect`, so
the second row means *no world geom passes through that cell at all* — the ray
fan finds only the robot's own link, which is there because the link has reached
into the cell. Yet every one of those 12 cells sits **16.17–30.52 mm from an
occupied octree leaf**, and 11 of the 12 sit within two half-diagonals of a real
surface.

**Those are not sensor phantoms. They are the octree→grid bridge's own
dilation.** `rasterize_octree_to_grid` marks every base cell whose cube shares
volume with an occupied leaf's cube; the two lattices share a resolution but not
a phase, so a leaf holding a real surface also marks cells up to one full cell
away from it. A cell whose centre is inside no leaf can only have come from that
rule — and 12 of 12 are exactly that.

The single exception is worth its own line. On `robocasa_fridge_drawer` layout
7 the tripping cell contains nothing but `robot0_link3`/`link4`, has an occupied
octree leaf 17.25 mm away, and has **no collidable surface within 299 mm** — and
the 243-ray fan finds no non-collidable geom in it either. That one is a genuine
map defect in the sense #160 raised: octomap believes something is there and
nothing is.

So the world-side residue divides:

| world-side term | n | of world-side | of all stops |
| --- | ---: | ---: | ---: |
| **quantisation** — grid right and coarse; surface genuinely in the cell | 5 | 29 % | 22 % |
| **bridge rasterisation dilation** — cell holds nothing, is in no leaf, real surface ≈ one cell away | 11 | 65 % | 48 % |
| **sensor-side phantom** — nothing real within 0.29 m | 1 | 6 % | 4 % |

### What this says about the three terms #161 named

- **Quantisation** is real and is the smaller half of the world-side residue
  here: 2 of 7.
- **Lattice phase** is not separately measured. This page scores the one phase
  the bridge actually publishes rather than sweeping it, so #161's 20.3 mm swing
  is carried forward unchanged and is folded into the "grid right, coarse"
  reading. Marked unmeasured, not dismissed.
- **Map fidelity** — in the sense #160 raised it, a cell that is *wrong* — is
  confirmed to exist and to dominate the world-side residue, but its cause is
  **not** what the framing supposed. Taking the three hypotheses in turn:
  - **stale?** No. Nothing has moved: every capture is a start state with zero
    actions applied, and the arm cannot have written the cells itself (the depth
    self-filter makes robot bodies transparent).
  - **spurious sensor noise?** No, and it cannot be: octomap needs two hits to
    lift a cell over `occupancy_thres` and one clearing ray to drop it back, so
    noise cannot persist anywhere a ray passes.
  - **displaced?** Yes — by exactly one cell, and *deterministically*. It is not
    a residue of a frame-alignment bug but a **dilation introduced on purpose**
    by the bridge, whose own comment records that the previous centre-sampling
    rule was worse. #160's "empty cell with solid cabinet exactly one cell away"
    is precisely this signature, and this is its mechanism.

That reframes the engineering question #161 posed. The choice is **not** "map
quality versus grid resolution". Nearly half the live stops in this sample
(48 %) are held by a **rasterisation rule** — which is neither of those: it is a
few lines of the bridge, adjustable without touching the kernel or the
resolution. Genuine sensor-side map error accounts for 1 stop in 23.

**Nothing here proposes changing that rule.** Narrowing it is a safety-relevant
loosening — it would shrink the obstacle set the kernel sees — and it belongs on
the safety-WG path with a hazard-log entry, not in a measurement PR. What the
measurement does establish is that the rule is where the leverage is, and that
it currently costs about one voxel of reach in every direction, on top of the
21.65 mm quantisation term and #161's 20.3 mm lattice swing. The two terms this
page isolates are tracked as work rather than as prose:
[#173](https://github.com/OpenRAL/openral/issues/173) (dilation) and
[#174](https://github.com/OpenRAL/openral/issues/174) (non-collidable geometry
in the map).

### Caveats on this split

- **n = 23 stops** over 27 states, sampled across all four scenes, not the full
  62. The correlation between backing verdict and octree membership is 17/17,
  but the population shares are sample estimates.
- `exact_mm` bottoms out at 0.00 for a link whose surface reaches inside a cell;
  it says the stop survives, not how deep it is.
- The nearest-solid distances behind the 16.64–27.35 mm column are sampled at
  10 mm, so they carry that error. They are used to say "about one cell", not to
  resolve the 21.65 mm half-diagonal boundary.
- `panda_link5`↔`panda_link7` are now known to genuinely interpenetrate at some
  configurations while the shipped ACM exempts the pair
  ([#169](https://github.com/OpenRAL/openral/pull/169)). Every stop here is a
  **world**-voxel stop, so none of them is that self-collision — but a future
  round that adjudicates self-collision stops must not attribute them here.


## What is not measured here

- **Any state after `t = 0`.** Every capture is a start state with zero actions
  applied. The four characterised live stops of the 2026-08-22 round happened
  during motion and are not re-adjudicated here.
- **Real hardware.** Mechanism 1 above is a simulation artefact and does not
  transfer; how much of the apportionment survives on a real depth sensor is
  unmeasured.
- **The lattice-phase term in isolation.** The 20.3 mm figure is #161's, carried
  forward unchanged. This page measures the map at the one phase the bridge
  actually publishes (`base_link`, origin −0.8/−0.8/−0.3) rather than sweeping
  it, so quantisation and lattice phase are reported together as one
  "grid is right but coarse" share.
- **Whether a cleared start state is a solvable task.** Layout 47 starts legal.
  No rollout has been run on it.

## Reproducing

Measurement code is not checked in — §1.11 keeps fixtures for tests, not for
studies — so the method is recorded here in enough detail to rebuild it.

**Capture.** One Python process: construct `ManifestHALLifecycleNode`, set
`robot_yaml`, `hal_mode=sim`, `sim_env_yaml=<scene>`, `viewer_enabled=false`,
then `trigger_configure()` + `trigger_activate()`. Separately spawn
`ros2 run octomap_server octomap_server_node` with `cloud_in` remapped to
`/openral/cameras/front_depth/points` and the sim parameters listed above, and
`ros2 run openral_octomap_bridge octomap_voxel_bridge`. Spin ~22 s, then record
in one atomic snapshot: the `OccupancyVoxels` message, `/octomap_binary`, the
`odom`→`base_link` transform, and `data.qpos` / `data.qvel` from the live
`MjModel`. Holding the HAL in the same process is what makes the map and the
geometry simultaneous.

**Rebuild.** `openral_hal.sim_bringup.build_sim_env_from_yaml` on the same
scene, then restore the recorded `qpos`/`qvel` and `mj_forward` — the idle
stepper advances the sim, so the pose must be restored rather than re-derived.

**Two things to get right or the results invert** (both cost this
investigation a retracted conclusion):
`scene_build.self_body_ids` must come from the manifest's `sim_joint_name`s, and
the grid frame body must be `resolve_base_frame_body_name(model,
description=...)` — `mobilebase0_support`, not `mobilebase0_base`.

**Octree decoding.** `octomap_msgs/Octomap.data` is the raw
`writeBinaryData()` stream, with no file header. Feed it straight to
`octomap::OcTree::readBinaryData` (a ~20-line C++ tool linking `-loctomap
-loctomath`); synthesising a `.bt` header round-trips to an empty tree.

**Kernel port.** Mirror `box_box_distance` (`collision.cpp:327`) and the
cell-cube treatment in `check_voxel_collision` (`collision.cpp:625`). Validate
it by compiling `collision.cpp` into a standalone oracle and comparing on random
OBB↔cell pairs; anything worse than ~1e-15 m means the rotation convention or
the axis set is wrong.

Run everything with per-state checkpointing. On a 16 GB host the surface-sampling
passes are memory-bound and two of them in parallel will be OOM-killed.

## Related

- [RoboCasa start-state collision census](robocasa-start-state-census.md)
- [Tight link geometry in the safety kernel](collision-tight-geometry.md)
- [Collision-stack validation evidence](collision-validation-evidence.md)
- `packages/openral_octomap_bridge/README.md` — the octree→grid bridge
- `cpp/openral_safety_kernel/README.md` — the kernel geometry
