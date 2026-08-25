# RoboCasa start-state collision census

Which limb makes the safety kernel refuse a RoboCasa kitchen before the robot
has been commanded, what it is touching, and whether the start pose or the base
placement is responsible.

Companion to [collision-stack validation evidence](collision-validation-evidence.md),
which is the chronological ledger of validation *rounds*. This page is a single
exhaustive measurement instead: every layout of every scene in the four-scene
matrix, at the scene's pinned seed, in one pass.

## The question

A kernel stop that fires before the robot has been commanded is a scene defect,
not a policy failure — that is why
`openral_hal.sim_sensor_bridge.initial_configuration_stop_record` exists. What
had never been established is **which link** causes it. `panda_mobile`'s
`collision_geometry` covers `panda_link1`..`panda_link7` and nothing else, so
the mobile base and the gripper fingers are outside the kernel's collision model
entirely (base-vs-world is Nav2's 2-D costmap job). The limb has to be one of
those seven.

## What was measured

240 start states: **4 scenes × 60 layouts**, each at the scene's own seed, built
through the registered RoboCasa adapter and read at reset with zero actions
applied.

| scene | task | seed | shipped pin |
| --- | --- | --- | --- |
| `robocasa_baguette` | `PickPlaceCounterToCabinet` | 1 | none (layout drawn) |
| `robocasa_sink_cup` | `PickPlaceCounterToSink` | 1 | none (layout drawn) |
| `robocasa_fridge_drawer` | `PickPlaceFridgeShelfToDrawer` | 1 | `layout_ids: [30]` |
| `robocasa_drawer_utensil` | `PickPlaceCounterToDrawer` | 1 | `layout_ids: [3]` |

Every row of the census pins `layout_ids: [N]`. **A pinned layout N is not the
same state an unpinned run that happened to draw layout N produces** — pinning
changes the RNG draw order, so style and object placement differ (established in
[#154](https://github.com/OpenRAL/openral/pull/154) and repeated here because it
governs how this table may be read). The census therefore describes the
population "what you get if you pin layout N", which is exactly the population
the two pinned scenes now live in, and is *not* a claim about any unpinned draw.
The effective layout and style are recorded per row and were checked to match
the pin on every row.

Per state and per link the census records:

- the nearest **solid** world geom and the mesh↔mesh gap to it;
- the **OBB↔voxel minimum the kernel would compute**, at 25 mm resolution;
- the link's corner slop, hence its admissible gap and its classification.

It also records the nearest world geom for the mobile base and the gripper
fingers — the parts the kernel does not model — so the "is it the base?"
question is answerable from the same run.

## How each side was measured, and why not the obvious way

### Kernel side

The kernel's own arithmetic, ported to numpy: `box_box_distance` from
`cpp/openral_safety_kernel/src/collision.cpp:327` — the separating-axis maximum
over 6 face normals and 9 edge-edge cross products, which is a *lower* bound on
true surface distance and is why the kernel never under-reports a collision.
Each occupied cell is treated as an axis-aligned cube of 12.5 mm half-extent, as
`check_voxel_collision` does. `world_voxel_margin_m` is `0.0` in sim
(`packages/openral_rskill_ros/launch/sim_e2e.launch.py`), so *stops* means
`SAT ≤ 0`.

Two things were self-tested rather than assumed:

- **OBB placement.** Every link's MuJoCo collision mesh lies inside its manifest
  OBB with 0.08–0.18 mm to spare on every axis, matching the manifest's claim to
  be face-tight. A misplaced OBB would have shown as a mesh vertex outside.
- **The rotation convention** matches `sim_sensor_bridge._rpy_to_matrix` to
  0.000e+00 on the manifest's own `origin_xyz_rpy` triples.

### World occupancy

The 25 mm grid is built geometrically: a cell is occupied if any **solid**
(`contype` or `conaffinity` non-zero) world surface falls in it, sampled at 5 mm
over boxes, spheres, cylinders/capsules and subdivided mesh triangles, with
robot bodies excluded because they are in the depth self-filter set.

**This is an idealisation and is marked as one.** The live map is
depth-camera-derived through octomap, so it can only ever be a *subset* of this
grid — surfaces the camera has not seen are absent from it. The reconstructed
minimum is therefore the distance the kernel would report with perfect
perception, and it bounds the live kernel from the conservative side. It is
consistent with the field observation that the 2026-08-22 fridge round E-stopped
at `t = 4.85 s` rather than at `t = 0`: the map fills in as the camera sees.
**How much of this grid the live octomap actually contains at reset is
unmeasured here.**

### Mesh side — `mj_geomDistance` is not usable for these pairs

The ground-truth probes in `estop_ground_truth_snapshot` are built on
`mujoco.mj_geomDistance`. Under **mujoco 3.8.0** that function is unreliable for
the pairs this census needs — a RoboCasa fixture geom against a `panda_mobile`
collision mesh. Measured on `robocasa_fridge_drawer` layout 9,
`robot0_link7_collision` vs `fridge_right_group_freezer_door_main`:

| path | `distmax` | returned | witness segment |
| --- | --- | --- | --- |
| native CCD (3.8 default) | 0.1 m | `+0.000 mm` | 126.264 mm |
| native CCD (3.8 default) | 0.3 m | `+0.000 mm` | 126.264 mm |
| `mjDSBL_NATIVECCD` (libccd) | 0.1 m | `−57.032 mm` | 57.032 mm |
| `mjDSBL_NATIVECCD` (libccd) | 0.3 m | `−351.570 mm` | 351.570 mm |

Densely sampling both surfaces puts the true gap at **+7.61 mm**. Every returned
value above is wrong, and the `fromto` witness the native path writes has
endpoints (x ≈ 5.72) lying outside both geoms — the link OBB is centred at
x = 5.18 and the door geom spans x = 5.06..5.49. A −351.570 mm penetration is
independently absurd: the door panel is 48 mm thick.

> **Superseded on 2026-08-25, and sharpened.** This pair now has a *certified*
> answer — **+0.148512 mm**, with a separating-axis optimality proof closing to
> `1.8e-14 m` — from `openral_hal.convex_distance`, the instrument this
> section's recommendation 6 asked for. The `+7.61 mm` above is not withdrawn so
> much as re-read: sampled point-to-point is an **upper bound** on the surface
> gap (as this section says two paragraphs down), and 7.61 > 0.149 is that bound
> behaving as one where the sampling grid missed the closest approach. The
> conclusions this census draws are unaffected — they were all drawn *against*
> the sampled measurement, which is conservative in the safe direction — but a
> reader citing the +7.61 mm as a truth rather than a bound should cite the
> certified value instead. The full characterisation, including that the native
> failure is a knife-edge degenerate configuration (a **1 picometre**
> displacement returns the right answer) rather than a distance regime, is the
> [2026-08-25 correction](collision-validation-evidence.md#2026-08-25--the-ruler-was-wrong-and-here-is-what-it-moves).

A second, distinct failure of the same function was found on
`robocasa_sink_cup` layout 1: `robot0_link1_collision` vs `sink_main_group_g2`
returns exactly `0.000000` at `distmax = 0.3` while the witness segment is
correct at 0.193 m (brute force: 0.1907 m). At the repo's **default**
`distmax = 0.1 m` this second mode does not occur — an audit of all
7 × 645 solid link↔world pairs on that state found **0 false zeros at 0.1 m and
1 of 75 computed pairs at 0.3 m**, all at true gaps beyond 0.19 m.

So the mesh side here is measured by **nearest-point between densely sampled
surfaces** instead: 5 mm on world geometry, 4 mm on link geometry, refined to
**1 mm** locally around the closest approach whenever the coarse gap is under
25 mm. Sampled point-to-point is an upper bound on the true surface gap with
error bounded by the sampling step. Interpenetration is taken from MuJoCo's own
contact solver, whose documented caveat carries over unchanged: `contype` /
`conaffinity` exclusions can suppress a contact, so "no contact reported" is not
proof of no interpenetration. A gap at or below the refined 1 mm resolution is
reported as **`contact_unresolved`**, not as contact — the measurement cannot
separate a touch from a sub-millimetre gap, and says so.

> **This bears on the existing ledger.** The fridge scene's documented
> "`robot0_link7_collision` measures 0.000 m mesh-to-mesh against the closed
> freezer door — a real touch" is exactly the reading `mj_geomDistance` produces
> in its failure mode. This census does not show that particular claim to be
> false — it was measured on a different state — but it does show that a bare
> `0.000 m` from this probe on a fixture↔link pair is not by itself evidence of
> contact. That is a third, independent reason to distrust old `real-contact`
> verdicts, alongside caveats 5 and 6 of the validation-evidence page.

## The collision model's own conservatism, per link

Measured against the live RoboCasa `panda_mobile` model by the repo's own
`collision_model_mesh_slop`. Pose-independent, so these seven numbers hold for
every row of the census.

| link | OBB half-extents (m) | worst face slop | **corner slop** | admissible gap = slop + 21.65 mm |
| --- | --- | --- | --- | --- |
| `panda_link1` | 0.0552, 0.0724, 0.1410 | 0.18 mm | **53.40 mm** | 75.05 mm |
| `panda_link2` | 0.0552, 0.0719, 0.1407 | 0.13 mm | **48.22 mm** | 69.87 mm |
| `panda_link3` | 0.0655, 0.0673, 0.1264 | 0.12 mm | **86.44 mm** | 108.09 mm |
| `panda_link4` | 0.0658, 0.0684, 0.1272 | 0.14 mm | **88.22 mm** | 109.87 mm |
| `panda_link5` | 0.0552, 0.0672, 0.1705 | 0.18 mm | **45.33 mm** | 66.98 mm |
| `panda_link6` | 0.0467, 0.0706, 0.0921 | 0.17 mm | **53.35 mm** | 75.00 mm |
| `panda_link7` | 0.0295, 0.0440, 0.0711 | 0.17 mm | **28.27 mm** | 49.93 mm |

The voxel term is `25 mm × √3 / 2 = 21.651 mm`, the cube's circumradius — the
worst case over relative orientation. Against an axis-aligned cell the effective
inflation on a face-aligned approach is only 12.5 mm, so the budget above is the
loosest admissible gap, not the typical one.

Two corrections to figures in circulation:

- The corner-slop **range is 28.3–88.2 mm**, not "12.6 mm to 88.2 mm". No link
  measures 12.6 mm; the minimum is `panda_link7` at 28.27 mm.
- Worst **face** slop against this model is **0.18 mm**, not 2.73 mm. (The 2.73 mm
  figure is cited against `panda_mj_description`; this census measures the
  RoboCasa model the scenes actually build, which is the model the kernel is
  checked against here.)

A third point matters for adjudication. `adjudication_budget.admissible_gap_m`
as published is a **single scalar built from `max_corner_slop_m`** — 88.22 mm +
21.65 mm = 109.87 mm — while the rule text beside it correctly says
`corner_slop(link) + voxel_half_diagonal`. Applying the published scalar per
link over-forgives by up to 60 mm: on `panda_link7` the real budget is 49.93 mm,
not 109.87 mm. The per-link values are already in the same record under
`collision_model_slop.links[<link>].corner_slop_m`; this census uses those.

## The answer

**`panda_link2` and `panda_link1`, and almost nothing else.** They dominate
**60 of the 72** start states in which the kernel refuses the scene — 83.3% —
and which of the two it is depends entirely on the scene.

### Ranked limb table

Dominant link = the kernel-checked link with the smallest OBB↔voxel distance,
counted only over the 72 states where the kernel stops. This table is the one a
primitive-fitting study should cite.

| link | stops dominated | share | corner slop | baguette | sink_cup | fridge_drawer | drawer_utensil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `panda_link2` | **46** | **63.9%** | 48.22 mm | 0 | 0 | 46 | 0 |
| `panda_link1` | **14** | **19.4%** | 53.40 mm | 0 | 0 | 0 | 14 |
| `panda_link7` | 8 | 11.1% | 28.27 mm | 1 | 0 | 7 | 0 |
| `panda_link5` | 3 | 4.2% | 45.33 mm | 1 | 0 | 2 | 0 |
| `panda_link6` | 1 | 1.4% | 53.35 mm | 0 | 0 | 1 | 0 |
| `panda_link3` | 0 | 0.0% | 86.44 mm | 0 | 0 | 0 | 0 |
| `panda_link4` | 0 | 0.0% | 88.22 mm | 0 | 0 | 0 | 0 |
| **total** | **72** | | | **2** | **0** | **56** | **14** |

`panda_link3` and `panda_link4` — the two links with the *worst* corner slop,
86.4 mm and 88.2 mm — dominate **nothing**. Slop alone does not predict which
link stops a scene; where the link sits relative to the fixtures does. A
primitive study that optimises the links with the largest slop would be
optimising the two links that never cause this failure.

### Outcome per scene

| scene | layouts | kernel stops | real contact | contact unresolved | within envelope | unexplained | no stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `robocasa_baguette` | 60 | 2 | 0 | 0 | 2 | 0 | 58 |
| `robocasa_sink_cup` | 60 | 0 | 0 | 0 | 0 | 0 | 60 |
| `robocasa_fridge_drawer` | 60 | **56** | 2 | 4 | 50 | 0 | 4 |
| `robocasa_drawer_utensil` | 60 | 14 | 0 | 0 | 14 | 0 | 46 |
| **total** | **240** | **72** | **2** | **4** | **66** | **0** | **168** |

`robocasa_fridge_drawer` is not merely the worst scene, it is nearly the whole
problem: **56 of the 72** stopping states are its. `robocasa_sink_cup` never
stops at any layout.

### Real contact versus envelope conservatism

**66 of 72 stops — 91.7% — are envelope conservatism**: the link is measurably
clear of the fixture, and the kernel stops because the OBB's corner slop plus
the voxel's 21.65 mm inflation together exceed that clearance. The kernel is
right to stop; it is not seeing something that is not there.

At most **6 of 72 — 8.3%** involve contact: 2 states where MuJoCo's contact
solver reports a genuine contact, and 4 more where the gap is at or below the
1 mm measurement resolution and is reported as unresolved rather than claimed
either way. All six are `robocasa_fridge_drawer`, and five of the six are
`panda_link7` against a freezer door:

| layout | link | classification | mesh gap | world body |
| ---: | --- | --- | ---: | --- |
| 2 | `panda_link7` | real_contact | 0.11 mm | `fridge_main_group_freezer_door` |
| 24 | `panda_link7` | real_contact | 0.14 mm | `fridge_right_group_freezer_door` |
| 9 | `panda_link7` | contact_unresolved | 0.69 mm | `fridge_right_group_freezer_door` |
| 44 | `panda_link7` | contact_unresolved | 0.14 mm | `fridgesidebyside_main_group_1_freezer_door` |
| 57 | `panda_link7` | contact_unresolved | 0.28 mm | `fridgesidebyside_main_group_1_freezer_door` |
| 41 | `panda_link2` | contact_unresolved | 0.56 mm | `fridgesidebyside_main_group_1_fridge_drawer4` |

So the freezer-door contact that the original fridge defect was named for is
real, and it is `panda_link7` — but it accounts for 5 of 72 stopping states, not
for the scene's 56.

**`UNEXPLAINED`: zero.** No start state in the census has the kernel stopping on
a mesh gap wider than `corner_slop(link) + 21.65 mm`. Nothing here indicates a
kernel defect.

> An earlier pass of this census produced three `UNEXPLAINED` rows. All three
> were artifacts of measuring the mesh side with `mj_geomDistance`, and all three
> disappear under the sampled measurement. They are recorded here because the
> conclusion "the kernel is wrong" was, briefly, what the numbers said.

### What each limb is hitting, and at what height

| dominant link | states | median witness height | what it hits |
| --- | ---: | ---: | --- |
| `panda_link2` | 46 | **1.03 m** | the fridge's own lower housing — `*_fridge_drawer0` on every fridge variant (`fridge_main`, `fridge_left`, `fridge_right`, `fridgefrenchdoor_*`, `fridgesidebyside_*`) |
| `panda_link1` | 14 | **0.89 m** | stacked-cabinet doors — `stack_*_door_main`, the cabinet run the base parks against |
| `panda_link7` | 8 | 1.42 m | freezer doors, and one cabinet door |
| `panda_link5` | 3 | 1.49 m | fridge doors, and one cabinet door |
| `panda_link6` | 1 | 1.47 m | `fridge_main_group_freezer_door` |

This **confirms** the standing description — "usually `link1`/`link2` at
shoulder height against the fixture housing" — and sharpens it: the two links
hit *different fixture families in different scenes*. `link2` is a fridge-scene
failure against the fridge's lower drawer housing; `link1` is a utensil-scene
failure against a stacked cabinet run. Neither is the freezer door that the
original fridge defect was attributed to — the freezer door accounts for the 8
`link7` states and 1 `link6` state, 9 of 72.

### It is not the base

`base_link`, `base_x_link`, `base_y_link` and `panda_finger_pair` have no entry
in `panda_mobile`'s `collision_geometry`, so the kernel's collision model does
not contain them and **they cannot cause a stop**. The census confirms this is
not a technicality that hides a real base problem — it is also not where the
geometry is worst in a way that would matter:

- the mobile base is nearer to the world than any kernel-checked arm link in
  **79 of 240** states — it is routinely the closest part of the robot;
- it is at or within 1 mm of the world in **5 of 240** states (usually the floor
  or a fridge drawer front);
- the gripper fingers are never within 1 mm of the world at reset (**0 of 240**).

So the base does park very close to furniture, exactly as the manifest comment
says it does, and that is precisely why it is excluded — base-vs-world is Nav2's
2-D costmap job (ADR-0040). **The base is not the limb causing these stops, and
no base-primitive change would alter a single row of this census.**

### Per-layout evidence

**`robocasa_fridge_drawer`** — 56 of 60 layouts stop. Dominant link per layout:

| layout | dominant link | kernel min | mesh gap | budget | classification | nearest world body |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | `panda_link2` | -36.2 mm | 14.8 mm | 69.9 mm | within_envelope | `fridge_main_group_fridge_drawer0` |
| 2 | `panda_link6` | -25.6 mm | 6.9 mm | 75.0 mm | within_envelope | `fridge_main_group_freezer_door` |
| 3 | `panda_link2` | -23.5 mm | 24.7 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 4 | `panda_link2` | -35.9 mm | 8.9 mm | 69.9 mm | within_envelope | `fridge_main_group_fridge_drawer0` |
| 5 | `panda_link2` | -22.5 mm | 21.8 mm | 69.9 mm | within_envelope | `fridge_left_group_fridge_drawer0` |
| 6 | `panda_link2` | -35.8 mm | 1.8 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 7 | `panda_link2` | -19.8 mm | 22.8 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_door_Clear` |
| 8 | `panda_link2` | -36.9 mm | 9.6 mm | 69.9 mm | within_envelope | `fridge_left_group_fridge_drawer0` |
| 9 | `panda_link7` | -25.0 mm | 0.7 mm | 49.9 mm | contact_unresolved | `fridge_right_group_freezer_door` |
| 10 | `panda_link2` | -35.9 mm | 2.0 mm | 69.9 mm | within_envelope | `fridge_left_group_fridge_drawer0` |
| 11 | `panda_link2` | -37.7 mm | 16.5 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 12 | `panda_link2` | -37.7 mm | 6.5 mm | 69.9 mm | within_envelope | `fridge_1_left_group_fridge_drawer0` |
| 13 | `panda_link2` | -23.5 mm | 37.0 mm | 69.9 mm | within_envelope | `fridge_1_right_group_1_fridge_drawer0` |
| 14 | `panda_link2` | -36.4 mm | 11.5 mm | 69.9 mm | within_envelope | `fridge_left_group_1_fridge_drawer0` |
| 15 | `panda_link2` | -22.8 mm | 20.4 mm | 69.9 mm | within_envelope | `fridge_1_main_group_1_fridge_drawer0` |
| 16 | `panda_link2` | -23.3 mm | 19.4 mm | 69.9 mm | within_envelope | `fridge_1_main_group_1_fridge_drawer0` |
| 17 | `panda_link2` | -36.1 mm | 10.0 mm | 69.9 mm | within_envelope | `fridge_1_left_group_1_fridge_drawer0` |
| 19 | `panda_link2` | -22.9 mm | 32.2 mm | 69.9 mm | within_envelope | `fridge_1_main_group_3_fridge_drawer0` |
| 20 | `panda_link2` | -23.6 mm | 36.2 mm | 69.9 mm | within_envelope | `fridge_1_front_group_1_fridge_drawer0` |
| 22 | `panda_link2` | -20.5 mm | 42.8 mm | 69.9 mm | within_envelope | `fridge_main_group_fridge_drawer0` |
| 23 | `panda_link2` | -36.6 mm | 5.1 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 24 | `panda_link7` | -23.3 mm | 0.1 mm | 49.9 mm | real_contact | `fridge_right_group_freezer_door` |
| 25 | `panda_link2` | -35.8 mm | 1.8 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 26 | `panda_link2` | -22.0 mm | 20.4 mm | 69.9 mm | within_envelope | `fridge_main_group_fridge_drawer0` |
| 27 | `panda_link2` | -36.7 mm | 5.1 mm | 69.9 mm | within_envelope | `fridge_left_group_fridge_drawer0` |
| 28 | `panda_link2` | -38.0 mm | 5.9 mm | 69.9 mm | within_envelope | `fridge_right_group_fridge_drawer0` |
| 29 | `panda_link7` | -18.7 mm | 10.4 mm | 49.9 mm | within_envelope | `fridge_main_group_freezer_door` |
| 30 | `panda_link2` | -23.5 mm | 37.0 mm | 69.9 mm | within_envelope | `fridge_left_group_fridge_drawer0` |
| 31 | `panda_link2` | -27.9 mm | 18.2 mm | 69.9 mm | within_envelope | `fridgebottomfreezer_main_group_1_fridge_drawer0` |
| 32 | `panda_link7` | -5.5 mm | 23.4 mm | 49.9 mm | within_envelope | `fridgesidebyside_right_group_1_freezer_door` |
| 33 | `panda_link2` | -35.8 mm | 1.8 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_right_group_1_fridge_drawer0` |
| 34 | `panda_link2` | -23.6 mm | 27.6 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_left_group_1_fridge_drawer0` |
| 35 | `panda_link2` | -35.9 mm | 8.9 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 36 | `panda_link2` | -23.2 mm | 34.2 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_front_group_1_fridge_drawer0` |
| 37 | `panda_link5` | -25.9 mm | 12.4 mm | 67.0 mm | within_envelope | `fridgefrenchdoor_left_group_1_fridge_left_door` |
| 38 | `panda_link5` | -18.4 mm | 12.2 mm | 67.0 mm | within_envelope | `fridgesidebyside_left_group_1_fridge_door_Clear` |
| 39 | `panda_link2` | -35.1 mm | 15.4 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 40 | `panda_link2` | -23.5 mm | 23.2 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_right_group_1_fridge_drawer0` |
| 41 | `panda_link2` | -18.7 mm | 0.6 mm | 69.9 mm | contact_unresolved | `fridgesidebyside_main_group_1_fridge_drawer4` |
| 42 | `panda_link2` | -22.7 mm | 24.6 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 44 | `panda_link7` | -22.0 mm | 0.1 mm | 49.9 mm | contact_unresolved | `fridgesidebyside_main_group_1_freezer_door` |
| 45 | `panda_link7` | -18.7 mm | 10.5 mm | 49.9 mm | within_envelope | `fridgesidebyside_right_group_1_freezer_door` |
| 46 | `panda_link2` | -22.7 mm | 20.7 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 48 | `panda_link2` | -37.2 mm | 3.6 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_right_group_1_fridge_drawer0` |
| 49 | `panda_link2` | -35.0 mm | 1.0 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_left_group_2_fridge_drawer0` |
| 50 | `panda_link2` | -23.5 mm | 37.0 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_left_group_1_fridge_drawer0` |
| 51 | `panda_link2` | -36.6 mm | 2.5 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 52 | `panda_link2` | -33.6 mm | 6.1 mm | 69.9 mm | within_envelope | `fridgebottomfreezer_left_group_1_fridge_drawer0` |
| 53 | `panda_link2` | -37.4 mm | 4.0 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_front_group_1_fridge_drawer0` |
| 54 | `panda_link2` | -23.1 mm | 34.2 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_main_group_1_fridge_drawer0` |
| 55 | `panda_link2` | -23.5 mm | 38.0 mm | 69.9 mm | within_envelope | `fridgebottomfreezer_main_group_1_fridge_drawer0` |
| 56 | `panda_link2` | -37.7 mm | 18.6 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_right_group_1_fridge_drawer0` |
| 57 | `panda_link7` | -22.2 mm | 0.3 mm | 49.9 mm | contact_unresolved | `fridgesidebyside_main_group_1_freezer_door` |
| 58 | `panda_link2` | -38.3 mm | 1.8 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_left_group_1_fridge_drawer0` |
| 59 | `panda_link2` | -37.1 mm | 16.3 mm | 69.9 mm | within_envelope | `fridgefrenchdoor_left_group_1_fridge_drawer0` |
| 60 | `panda_link2` | -36.6 mm | 5.1 mm | 69.9 mm | within_envelope | `fridgebottomfreezer_right_group_1_fridge_drawer0` |

**`robocasa_drawer_utensil`** — 14 of 60 layouts stop. Every one is `panda_link1` against a stacked-cabinet door:

| layout | dominant link | kernel min | mesh gap | budget | classification | nearest world body |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 2 | `panda_link1` | -5.5 mm | 33.4 mm | 75.1 mm | within_envelope | `stack_5_main_group_2_door_main` |
| 5 | `panda_link1` | -5.0 mm | 49.3 mm | 75.1 mm | within_envelope | `stack_1_main_group_3_door_main` |
| 6 | `panda_link1` | -7.7 mm | 40.5 mm | 75.1 mm | within_envelope | `stack_2_right_group_2_door_main` |
| 10 | `panda_link1` | -6.0 mm | 59.7 mm | 75.1 mm | within_envelope | `stack_3_main_group_2_door_main` |
| 24 | `panda_link1` | -6.4 mm | 39.6 mm | 75.1 mm | within_envelope | `stack_4_main_group_3_door_main` |
| 31 | `panda_link1` | -5.2 mm | 17.7 mm | 75.1 mm | within_envelope | `stack_5_right_group_1_2_door_main` |
| 41 | `panda_link1` | -6.4 mm | 17.7 mm | 75.1 mm | within_envelope | `stack_2_island_group_1_2_door_main` |
| 48 | `panda_link1` | -6.5 mm | 28.4 mm | 75.1 mm | within_envelope | `stack_03_right_group_1_2_door_main` |
| 49 | `panda_link1` | -5.4 mm | 46.0 mm | 75.1 mm | within_envelope | `stack_14_front_group_1_2_door_main` |
| 51 | `panda_link1` | -2.9 mm | 45.3 mm | 75.1 mm | within_envelope | `stack_09_island_group_1_3_door_main` |
| 52 | `panda_link1` | -6.3 mm | 19.2 mm | 75.1 mm | within_envelope | `stack_05_left_group_1_3_door_main` |
| 54 | `panda_link1` | -4.7 mm | 27.2 mm | 75.1 mm | within_envelope | `stack_07_island_group_1_2_door_main` |
| 55 | `panda_link1` | -4.7 mm | 28.8 mm | 75.1 mm | within_envelope | `stack_01_main_group_1_2_door_main` |
| 59 | `panda_link1` | -8.8 mm | 18.2 mm | 75.1 mm | within_envelope | `stack_02_main_group_1_2_door_main` |

**`robocasa_baguette`** — 2 of 60 layouts stop:

| layout | dominant link | kernel min | mesh gap | budget | classification | nearest world body |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 7 | `panda_link7` | -1.8 mm | 17.4 mm | 49.9 mm | within_envelope | `cab_main_main_group_door_main` |
| 27 | `panda_link5` | -6.4 mm | 33.6 mm | 67.0 mm | within_envelope | `cab_2_main_group_door_main` |

**`robocasa_sink_cup`** — no layout stops at any of the 60.

### The shipped pins

`robocasa_drawer_utensil` at its shipped `layout_ids: [3]` pin — **no stop**, every
link `genuinely_clear`. The pin does what it claims.

`robocasa_fridge_drawer` at its shipped `layout_ids: [30]` pin — **still stops**,
on `panda_link2`:

| link | mesh gap | kernel min | budget | classification | nearest world body |
| --- | ---: | ---: | ---: | --- | --- |
| `panda_link1` | 37.91 mm | +19.46 mm | 75.05 mm | genuinely_clear | `fridge_left_group_fridge_drawer0` |
| `panda_link2` | 37.04 mm | **−23.47 mm** | 69.87 mm | **within_envelope** | `fridge_left_group_fridge_drawer0` |
| `panda_link3` | 224.17 mm | +175.72 mm | 108.09 mm | genuinely_clear | `fridge_left_group_fridge_left_door` |
| `panda_link4` | 226.31 mm | +158.19 mm | 109.87 mm | genuinely_clear | `fridge_left_group_fridge_left_door` |
| `panda_link5` | 157.25 mm | +88.28 mm | 66.98 mm | genuinely_clear | `fridge_left_group_fridge_left_door` |
| `panda_link6` | 226.12 mm | +186.03 mm | 75.00 mm | genuinely_clear | `fridge_left_group_fridge_left_door` |
| `panda_link7` | 203.36 mm | +166.78 mm | 49.92 mm | genuinely_clear | `fridge_left_group_fridge_left_door` |

This does not contradict the measurement [#154](https://github.com/OpenRAL/openral/pull/154)
made — it reproduces it and then goes one step further. That PR reported layout
30 as "37.0 mm clear of the 25 mm occupancy grid, 0 contacts"; this census
measures the same 37 mm (37.04 mm on `link2`, 37.91 mm on `link1`) and finds no
contact. But 37.04 mm of *mesh* clearance is not kernel clearance for `link2`:
its 48.22 mm corner slop alone exceeds it, and adding the 21.65 mm voxel term
puts the reconstructed OBB↔voxel distance at **−23.47 mm**.

**The pin fixed the mesh clearance without fixing the kernel verdict.** Layout 30
is the best of the 60 by mesh gap and one of only four that avoid a
contact-class reading, which is why it was chosen; it is not one of the four that
clear the kernel. Under a geometrically ideal grid it still stops. Whether it
stops on the *live* depth-derived grid is **unmeasured** — see the idealisation
caveat above — and is the one thing worth a live round before acting on this.

## What can move each link

Holding the base where the scene parks it, the grid was rebuilt once over the
arm's whole reach and the kernel minimum re-evaluated over many joint
configurations. Single-joint sweeps across each joint's full range, from the
shipped pose:

| swept joint | `panda_link1` spread | `panda_link2` spread |
| --- | --- | --- |
| `joint1` | 81.61 mm | 110.33 mm |
| `joint2` | **0.00 mm** | 87.77 mm |
| `joint3` | **0.00 mm** | **0.00 mm** |
| `joint4` | **0.00 mm** | **0.00 mm** |

(`robocasa_fridge_drawer` layout 30; the utensil scene gives the same structure,
with `joint1` moving `panda_link1` by 45.55 mm and `joint2`–`joint4` by
0.00 mm.)

This is kinematics, not a measurement artifact: `panda_link1` is the first
moving link, so only `joint1` can move it at all, and `panda_link2` can only be
moved by `joint1` and `joint2`. **Joints 3–7 cannot change a start-state verdict
whenever `panda_link1` or `panda_link2` is the dominant link** — which is 83.3%
of the stopping states.

### Is the rest pose implicated?

Yes, and it is recoverable — but only through `joint1` and `joint2`. Two
representative stopping states, each evaluated at the shipped reset pose, at the
canonical Panda ready pose, at all-zeros, and over 1500 uniform samples inside
the live model's joint limits (self-colliding samples discarded via MuJoCo's own
contact set):

| state | shipped | `ready` pose | all-zeros | best of 1500 sampled | samples that clear |
| --- | ---: | ---: | ---: | ---: | ---: |
| fridge L30 (`link2`) | −23.47 mm | −7.75 mm | +19.77 mm (self-collides) | **+16.12 mm** | 9 / 1500 |
| utensil L2 (`link1`) | −5.54 mm | −5.22 mm | −5.22 mm (self-collides) | **+23.85 mm** | 481 / 1500 |

Both are recoverable by a pose change, and **neither is recovered by the
canonical `ready` pose** — the obvious first thing to try moves fridge L30 from
−23.5 mm to −7.8 mm and utensil L2 from −5.5 mm to −5.2 mm, in both cases still
stopping. The reason is visible in the sweep table above: `ready` differs from
the shipped pose almost entirely in `joint2` (0.25 rad; every other joint moves
under 0.10 rad), which does nothing at all for `panda_link1`.

The two states differ in how much room there is:

- **utensil L2 is comfortably recoverable.** `joint1` alone moves `panda_link1`
  across −21.75 mm to **+23.80 mm**, and about a third of sampled poses clear.
- **fridge L30 is marginal.** The best sampled pose reaches only +16.12 mm, and
  9 of 1500 samples clear. `panda_link1` is pinned at +19.3 to +19.8 mm in every
  configuration tried — sweeping `joint1` never raises it above **+19.77 mm** —
  so ~20 mm is a hard ceiling on that layout no matter what the arm does. The
  shipped `joint1` is already within 0.3 mm of the best value available to it.

That ceiling is the important part. `panda_link1` sits close to `joint1`'s own
axis, so rotating `joint1` translates it very little; whatever clearance
`panda_link1` has is set almost entirely by **where the base parked**. On a
layout where `panda_link1`'s best-case clearance is already negative, no arm
configuration can rescue the start state.

## What follows — proposals, not changes

This page is an investigation. Nothing in the kernel, the manifest, or any
threshold was changed. In rough order of cost:

1. **Do not pursue a base collision primitive for this failure.** The base is
   outside the kernel's collision model by design and no row of this census
   depends on it. Confirmed, not assumed.
2. **A primitive study should target `panda_link2` and `panda_link1`, not
   `link3`/`link4`.** The two links with the worst corner slop dominate zero
   stops. `link2` at 48.22 mm and `link1` at 53.40 mm are where the 60 of 72
   stops are, and both are dominated by their corner slop rather than by the
   21.65 mm voxel term — for `link2` the slop is 69% of its 69.87 mm budget. A
   tighter primitive on those two would move most of this census. How much is
   needed is measurable directly from the deficits, and the two links are in
   very different regimes:

   | if a tighter primitive recovered… | `panda_link1` states cleared | `panda_link2` states cleared |
   | --- | ---: | ---: |
   | 10 mm | **14 / 14** | 0 / 46 |
   | the link's full corner slop | 14 / 14 | **46 / 46** |

   The `link1`-dominated states run only −2.9 to −8.8 mm, so ~10 mm of recovered
   clearance clears every one of them. The `link2`-dominated states run −18.7 to
   −38.3 mm and none of them clear at 10 mm; they need most of `link2`'s 48.22 mm
   of corner slop back. So a modest primitive improvement fixes the utensil scene
   outright and does nothing for the fridge scene.
3. **Re-tune the fridge start pose in `joint1`/`joint2` only** — or re-park the
   base. This is the cheapest fix for the fridge scene and needs no kernel
   change. It is bounded: ~+16 mm is the best available at layout 30.
4. **Re-examine the `layout_ids: [30]` pin.** It clears the mesh but not the
   reconstructed kernel. The four fridge layouts that avoid a kernel stop here
   are **18, 21, 43 and 47** — 18 and 21 are also in the set #154 identified on
   the mesh criterion, so either is a candidate that satisfies both. A pin chosen
   against the kernel criterion would be better founded than one chosen against
   the mesh criterion. **This should be confirmed on a live round first**,
   because the idealised grid is the conservative bound and the live octomap may
   be sparser at reset.
5. **Fix `adjudication_budget.admissible_gap_m`** to be per-link rather than a
   single `max_corner_slop_m`-derived scalar, or rename it so it cannot be read
   as a per-link budget. As published it over-forgives `panda_link7` by 60 mm.
6. ~~**Stop relying on `mj_geomDistance` for fixture↔link adjudication**, or
   gate it behind a self-check.~~ **Done, 2026-08-25.**
   `estop_ground_truth_snapshot` now measures with
   `openral_hal.convex_distance`, which carries a separating-axis optimality
   certificate and refuses rather than emit a number it cannot defend; the
   regression test this item asked for is
   `tests/sim/safety/test_geom_distance_instrument_robocasa.py`, pinned on this
   census's own layout-9 state. An audit of 1 102 probed pairs across five
   RoboCasa states found `mj_geomDistance` wrong on exactly one — the pair
   above — and re-measuring the checked-in validation rounds found four
   recorded `0.000 m` readings whose certified values are +14.8 / +82.2 / +98.8
   / +107.9 mm. See the
   [2026-08-25 correction](collision-validation-evidence.md#2026-08-25--the-ruler-was-wrong-and-here-is-what-it-moves).

## What was not measured

- **The live octomap.** Every kernel number here is against a geometrically
  ideal grid. How much of it the depth camera has actually voxelised at reset is
  unmeasured, and it is the difference between "would stop" and "does stop".
- **Any state after `t = 0`.** This is a start-state census only.
- **Unpinned draws.** Pinning changes the RNG draw order, so these 240 rows say
  nothing about what an unpinned seed-1 run produces for `robocasa_baguette` or
  `robocasa_sink_cup`.
- **Whether a recovered start pose is reachable or useful for the task.** The
  sampled poses were screened for self-collision only, not for whether the arm
  can still see or reach the object.

## Related

- [Collision-stack validation evidence](collision-validation-evidence.md) — the
  round-by-round ledger, and the standing caveats this page adds to.
- [The validation matrix](../contributing/validation-matrix.md) — how a round is
  run and what each verdict means.
- `cpp/openral_safety_kernel/README.md` — the kernel geometry the reconstruction
  mirrors.
