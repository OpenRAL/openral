# Collision-primitive study — is the OBB the reason the arm reads as in-collision?

> **Status: analysis only.** Nothing here has landed. It touches no code in
> `cpp/openral_safety_kernel/` or `packages/openral_safety/`, both safety-WG
> gated and needing a hazard-log entry before a change. Companion: the
> [validation evidence](collision-validation-evidence.md) holds the *failure*
> census; this is the *geometry* half, reproducible by
> [§9](#9-reproducing-the-numbers).

---

## 1. The premise this study was opened on is wrong, and that is the first result

The hypothesis was that the boxes are badly proud at their corners, the **mobile
base** worst of all. The geometry is right; the applicability to `panda_mobile`
is not: **its collision model contains no base.** `robots/panda_mobile/robot.yaml`
declares geometry for `panda_link1` … `panda_link7` and nothing else —
`base_link` excluded because base-vs-world is Nav2's 2-D costmap job,
`panda_finger_pair` because checking the intended-contact part against the world
would veto every grasp. The kernel never places a primitive on the chassis, so
**a better chassis primitive buys exactly zero.** Whatever makes kitchen layouts
read as in-collision is one of the seven arm links, or is not a link at all.

---

## 2. What the kernel supports today

### 2.1 The lowered model

`CollisionModel` (`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`)
is two flat, parallel primitive arrays, each tagged with the link it rides on:

| array | element | fields |
|---|---|---|
| `capsule_link[c]` / `capsules[c]` | `Capsule` | `radius`, `half_length`, `origin` |
| `box_link[b]` / `boxes[b]` | `Obb` | `half_extents`, `origin` |

A sphere is a capsule with `half_length == 0`, which is how
`envelope_loader.collision_params_from_description` lowers `SphereShape`. No
mesh path — the kernel is analytic-convex-only so the hot path stays
allocation-free.

### 2.2 What the collision routines actually compute

| routine | box↔? | what it computes | exact? |
|---|---|---|---|
| `capsule_distance` | — | segment–segment distance − both radii | exact |
| `box_capsule_distance` | box↔capsule | 48-step ternary search of point→AABB distance along the capsule segment, in box-local coordinates, − capsule radius | exact when disjoint |
| `box_box_distance` | box↔box | separating-axis theorem over 15 axes; returns the **largest** per-axis gap | conservative **lower bound** |
| `check_voxel_collision` | box↔voxel | each occupied cell is an axis-aligned cube, then `box_box_distance` | conservative |

`fold_pair` / `finish_sweep` (`collision.cpp:480-523`) are pure bookkeeping and
read no geometry, so **neither is affected by a primitive change.**

### 2.3 Would a radius on the box path be nearly free? Yes — verified, with one trap

**Both distance routines offset exactly, one subtraction each.** For convex sets
the Minkowski sum with a ball is an exact offset of the distance function, and
the support radius of `Box(a) ⊕ B(r)` along **any** unit `n` is
`Σ_k a_k |â_k·n| + r` — the `+ r` is axis-independent, so **the 15-axis SAT set
stays exactly as conservative as it is today**.

**The trap: the voxel broad phase.** `check_voxel_collision` (`collision.cpp:684-703`)
computes the box's world AABB from `half_extents` alone —

```cpp
const double ex = |R[0]|*he.x + |R[1]|*he.y + |R[2]|*he.z;   // and ey, ez
const double reach = margin + half_side;
```

— and only visits cells inside that window. With `a = h − r` the extents
under-report by exactly `r`, so cells that genuinely intersect the swept box are
**never visited**: not a conservatism loss but a *missed collision*, silently
returning clear. The fix is one term
(`reach = margin + half_side + boxes[b].radius`) and it is the single line in
the change that can make the kernel unsafe rather than merely tighter.

### 2.4 Every site that would move

`struct Obb` (`collision.hpp`) gains `double radius{0.0}`; six sites in
`collision.cpp` subtract it — `check_self_collision` box↔capsule (570) and
box↔box (581), `check_world_collision` (617), `check_voxel_collision` box↔voxel
(716), `check_attached_self_collision` payload↔link-box (1369-1374), and the
voxel **broad phase** (689-697), which *adds* it to `reach` (§2.3);
`lifecycle_kernel.cpp:169,1364,1426` declare and load a `collision_box_radius`
array. Everything else — `fold_pair`, `finish_sweep`, `forward_kinematics`,
`jacobian_dls_step`, `support_contact_exempts`,
`update_support_contact_witnesses`, `place_approach_allowance`,
`check_attached_voxel_collision` — reads no `model.boxes`.

---

## 3. Multi-primitive links: **the kernel already supports them; the manifest lowering rejects them**

### 3.1 The rejection is real

```
$ PYTHONPATH=packages/openral_safety python -c "<duplicate panda_link1 in the manifest>"
pydantic entries: 8
LOWERING REJECTED: ROSConfigError link 'panda_link1' has >1 collision primitive;
                   split it into separate links (unsupported in this lowering phase)
```

**Pydantic accepts it** — `RobotDescription.collision_geometry` is a plain
`list[LinkCollisionGeometry]` with no uniqueness constraint on `link_name`, and
the eight-entry manifest validated, so **no schema change is needed to express a
multi-primitive link.** The refusal is `envelope_loader._capsules_by_link`,
which raises on the second entry for a link.

### 3.2 The kernel is fine with it — and a shipped test proves it

`CollisionModel`'s docstring says a link may carry *"zero, one, or **several**
capsules"*; `load_collision_model` (`lifecycle_kernel.cpp:1402-1431`) only
range-checks the link index; `check_self_collision` skips same-link pairs
explicitly (`li == lj`, `lb == lb2`, `lb == lc`);
`mjcf_lowering.lower_collision_params` already emits several capsules per link;
and `tests/sim/safety/test_kernel_h1_self_collision.py:128` asserts
`len(cap_links) > len(set(cap_links))`, then starts a **live kernel** on it.
`panda_mobile` cannot use the path only because it comes through the *manifest*
path (URDF-lowered, hand-committed `collision_geometry`), not the MJCF path.

### 3.3 What lifting it costs

`_capsules_by_link` becomes `_primitives_by_link` returning
`dict[str, list[LinkCollisionGeometry]]` and the emit loop iterates it; the flat
arrays are already per-primitive with a link tag, so nothing downstream changes
shape. `urdf_lowering.lower_link_geometry` unions **all** of a link's
`<collision>` elements into **one** PCA capsule (`urdf_lowering.py:252-264`) and
must stop unioning — the real work, but a generator change with no safety
surface of its own, since the safety property is whatever the emitted primitives
cover, checked at validation time. The ACM sweep keys on a single shape per link
(`geoms[ln].shape`) and would need to fold several; `_collision_z_extent_m`
already handles repeats.

**No schema change. No kernel change. No new C++** — a materially smaller safety
surface than adding a radius to `Obb`, which is why §7 recommends it first.

---

## 4. The fit table

### 4.1 Method

Ground truth is robosuite's panda collision meshes — one `link<N>_collision`
mesh geom per link, the geoms the kernel's OBBs are documented as enclosing and
the set `sim_sensor_bridge._body_collision_points` samples. Each candidate is
fitted in the link frame and measured on **max protrusion**, the one-sided
Hausdorff distance from the primitive's *boundary* to the mesh *surface* (not a
proxy for the slop: it **is** the worst-case distance under-report, since the
kernel reports `d(q, prim)` where the truth is `d(q, mesh)`), and **min
containment margin**, the minimum inward depth of the mesh inside the primitive,
`>= 0` being containment.

The margin is a **proof, not a sample**: a convex primitive containing every
mesh vertex contains `conv(vertices) ⊇ mesh`. The multi-primitive unions are not
convex, so whole *triangles* are assigned to one part each and that part's three
vertices checked. Nothing here was accepted on sampled evidence.

Cross-check: `panda_link1`'s `corner_slop_m` is **53.4 mm** in
[collision-validation-evidence.md](collision-validation-evidence.md) against
52.6 mm here; they should differ in exactly this direction, since `corner_slop`
measures to the nearest sampled mesh *vertex*, from 8 corners only.

### 4.2 The table

Protrusion in **bold**; the containment margin underneath it.

| candidate | link1 | link2 | link3 | link4 | link5 | link6 | link7 |
|---|---|---|---|---|---|---|---|
| OBB (shipped baseline) — protrusion (mm) | **52.6** | **45.2** | **74.0** | **76.3** | **43.7** | **49.6** | **27.3** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.055 | 0.088 | 0.083 | 0.083 | 0.132 | 0.098 |
| Sphere-swept box, **faces fixed** (`a = h − r`) — protrusion (mm) | **45.5** | **31.6** | **33.2** | **32.2** | **30.0** | **43.0** | **25.0** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.021 | 0.040 | 0.012 | 0.072 | 0.067 | 0.132 | 0.037 |
| Sphere-swept box, free fit — protrusion (mm) | **44.8** | **25.4** | **21.0** | **19.4** | **27.3** | **36.0** | **17.0** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.120 | 0.151 | 0.083 | 0.131 | 0.132 | 0.098 |
| Capsule (PCA — what `urdf_lowering` emits) — protrusion (mm) | **111.7** | **107.5** | **99.7** | **99.6** | **104.1** | **96.2** | **54.5** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.055 | 0.088 | 0.083 | 0.083 | 0.132 | 0.098 |
| 2 × OBB, slab split along the long axis — protrusion (mm) | **45.9** | **46.5** | **74.0** | **74.9** | **41.3** | **51.3** | **27.3** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 |
| 3 × OBB, slab split along the long axis — protrusion (mm) | **52.0** | **54.6** | **74.4** | **76.5** | **49.7** | **51.7** | **27.8** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 |
| 2 × OBB, k-means split — protrusion (mm) | **52.6** | **44.1** | **65.9** | **63.3** | **42.9** | **58.8** | **39.4** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.083 | 0.055 | 0.088 | 0.083 | 0.083 | 0.132 | 0.098 |
| 2 × SSB, k-means split — protrusion (mm) | **43.5** | **37.3** | **36.2** | **37.6** | **32.6** | **53.3** | **27.3** |
| &nbsp;&nbsp;↳ containment margin (mm) | 0.137 | 0.107 | 0.088 | 0.148 | 0.133 | 0.203 | 0.098 |

Per-link `r` for the swept candidates (mm): faces-fixed **23 / 19 / 39 / 41 / 15
/ 7 / 3**; free fit **32.7 / 54.5 / 64.8 / 65.1 / 21.8 / 27.7 / 23.3**.

### 4.3 Reading it

**The sphere-swept box works.** Shrinking to `a = h − r` and sweeping `r` is not
automatically containing — a point near the box corner is up to `√3·r` from
`Box(h − r)` — but it holds up to a per-link maximum `r` of 3–41 mm, with
containment proven over the full vertex set. Protrusion falls **8 % (link7) to
58 % (link4)**, faces are untouched by construction, volume falls slightly.
**A strict improvement: never more conservative anywhere than today, less
conservative only at the corners.** The free fit reaches 17–21 mm on the mid-arm
links but grows faces by up to **+11.7 mm on link2**, so it is *more*
conservative than today on a face-on approach — the better number, the worse
guarantee, and §7 does not recommend it.

**Corner reach is the quantity that moves**, and by how much is what §6 needs:

| link | OBB corner reach (mm) | faces-fixed SSB | Δ | free-fit SSB | Δ | max face growth (free fit) |
|---|---|---|---|---|---|---|
| link1 | 167.8 | 154.9 | **−12.9** | 150.8 | **−17.0** | +3.0 |
| link2 | 167.4 | 156.6 | **−10.8** | 149.0 | **−18.4** | +11.7 |
| link3 | 157.5 | 134.6 | **−22.9** | 130.1 | **−27.4** | +7.2 |
| link4 | 158.7 | 134.8 | **−23.9** | 131.1 | **−27.6** | +5.8 |
| link5 | 191.4 | 183.9 | **−7.5** | 182.5 | **−8.9** | +1.6 |
| link6 | 125.1 | 120.4 | **−4.7** | 112.0 | **−13.1** | +5.9 |
| link7 | 88.7 | 86.8 | **−1.9** | 79.4 | **−9.3** | +6.7 |

**The capsule is catastrophic here — 96–112 mm on every link but the wrist**,
because the Panda's links are flanged, tapered blocks, not rods. This vindicates
the #103 conversion from capsules to boxes, and
`urdf_lowering.lower_link_geometry` **still emits a PCA capsule for any mesh
collision**, so re-lowering `panda_mobile` from its URDF today would regress the
envelope by 1.3–2.1×. A live trap, not a historical note.

**Multi-primitive decomposition does not help these links.** Equal-mass slabs
along the long axis and k-means on triangle centroids, at k = 2 and k = 3:
neither beats a single sphere-swept box on any link, several are *worse* than
the single OBB, and volume rises 1.0–1.8×. One column says why:

| link | link1 | link2 | link3 | link4 | link5 | link6 | link7 |
|---|---|---|---|---|---|---|---|
| convex-hull volume ÷ mesh volume | **1.303** | 1.000 | 1.0003 | 1.0001 | 1.000 | 1.0003 | 1.0002 |

**Six of the seven collision meshes already *are* convex hulls** — convex to
1e-4 in volume ratio, so on those links `conv(mesh)` **is** the mesh and there
is nothing to decompose. Decomposition removes concavity; the slop here is the
*corner* of a bounding box around a convex, rounded, tapered solid, and every
sub-box has corners of its own. Only `panda_link1` is meaningfully non-convex
(30 % hull excess), and even there the 2-slab split (45.9 mm) ties the single
swept box (45.5 mm) at 1.1× the volume.

**The shipped OBBs carry no padding to give back.** Every containment margin
above is 0.012–0.203 mm, so the slop is entirely the box's own corner geometry.
A refit should *declare* an explicit headroom rather than inherit the current
accidental one (§7.3).

---

## 5. Named finding: **a box has no radius, and every scalar surrogate for one has now bitten us**

Three places in this repo have substituted a single scalar radius for a box's
three half-extents. Each picked a different surrogate, each was wrong in a
different direction, **none was conservative at both edges.**

| site | surrogate | direction | consequence |
|---|---|---|---|
| the safety kernel's OBB, as an envelope for a rounded, tapered link | *circumscribed*, implicitly — a sharp box reaches `\|h\|` at its corners | **over**-approximates | corners proud of the real link (§4); the kernel reads in-collision while the surface is clear |
| ~~`urdf_lowering._capsule_segment_radius` (offline ACM sweep)~~ — **deleted, issue [#155](https://github.com/OpenRAL/openral/issues/155) fixed** | was *inscribed*, `min(half_extents_m)` | **under**-approximated | The sweep now uses `kernel_predicates.shape_distance` — the kernel's own predicate on the true primitive. The remaining capsule lowering, `bounding_capsule_segment`, **circumscribes** and feeds only the cuMotion planner, where over-covering is the safe direction. See the note below: #155's conclusion was inverted, and the replacement numbers are grid-exhaustive rather than sampled. |
| `openral_slam_bringup.depth_height_filter_node._collision_z_span` | neither — it was forced to take the **exact** OBB support projection `Σ_k \|R[2][k]\|·h_k` | — | the node's own docstring records why: the inscribed radius *"shrinks the band and hides obstacles at body height"*, the circumscribed radius *"grows it … and the node re-marks the floor it exists to remove"*, so **"no single scalar is conservative at both edges"** |

The SLAM node got it right by refusing to pick a scalar at all:

> **A box's extent is direction-dependent. Any code that reduces it to one
> number is choosing a direction to be wrong in, and the two available choices
> are wrong in opposite directions. Use the support projection
> `Σ_k |û·ê_k|·h_k`, or use the true box.**

The middle row's original version inferred `panda_link5`↔`panda_link7` was a
capsule-junction artifact the sweep ought to restore. Wrong: **the pair
genuinely collides.** Over `(panda_joint6, panda_joint7)`, the only two joints
that move it, **914 of 14641 poses interpenetrate, up to 48.3 mm deep**, across
a **39.6° band of `joint6`** spanning the full `joint7` range; MoveIt agrees and
emits no `Never` row. The committed ACM has been *exempting* a real
self-collision, and the inscribed-sphere bug produced the correct ACM for the
wrong reason.

It stays exempt, because the box envelopes cannot express it either way: the box
check fires on **86.5%** of that space while only 6.9% is real (**79.6% false**),
including the arm's `ready` home pose at -9.14 mm, and **no margin separates the
populations** — real collisions reach -9.97 mm, collision-free poses -40.07 mm.
The exemption is a hand-owned `reason="User"` row in each panda SRDF carrying
this evidence. **This is the strongest single argument in this document for
tighter link geometry**, and it argues for a change that **stops substituting**.

---

## 6. What it would actually recover — **on the present evidence, nothing**

### 6.1 The decision rule

A stop is recoverable only if the reach removed **on the reporting link, in the
direction of the reporting cell** exceeds the penetration reported. Reach
removal is maximal at the corner and zero on the faces, so §4.3's *corner-reach
reduction* is a strict upper bound:

> A stop reporting `min_distance = −d` recovers only if
> `d < Δcorner_reach(reporting link)`.

Ceilings, faces-fixed variant: **link1 12.9 · link2 10.8 · link3 22.9 ·
link4 23.9 · link5 7.5 · link6 4.7 · link7 1.9 mm.** Doubly an upper bound: the
cell is almost never at the exact corner, and the voxel half-diagonal (21.7 mm
at the 25 mm sim grid) is untouched by any primitive change.

### 6.2 Applying it to the characterised stops

Applied to the link-class stops
[collision-validation-evidence.md](collision-validation-evidence.md) has
characterised (the layout census is produced separately; the rule is stated so
it can be applied to that table when it lands):

| stop | link | kernel `min_distance` | ceiling for that link | recovers? |
|---|---|---|---|---|
| baguette, 2026-08-22 | `panda_link5` | **−20.9 mm** | 7.5 mm | **no** — short by 13.4 mm |
| utensil, 2026-08-22 / -23 | `panda_link1` | **−17.3 mm** | 12.9 mm | **no** — short by 4.4 mm |
| harness-2 fridge | `panda_link7` | **−24.7 mm** | 1.9 mm | **no** — short by 22.8 mm (and already adjudicated *real contact*) |

**None of the three recovers.** Even the free-fit variant lifts the link5
ceiling only to 8.9 mm and the link1 ceiling to 17.0 mm. **On every stop this
repo has characterised, a tighter primitive does not clear the stop.** §4 proves
the envelope can be improved by 13–58 %; §6 says that improvement recovers no
layout we know about.

### 6.3 What that leaves

The baguette stop is the one the evidence corpus calls unexplained: 120.9 mm of
discrepancy against an 88.2 mm admissible gap, with an *untruncated* ground-truth
probe returning **no pair at all** within 100 mm. Removing 7.5 mm of link5
corner reach leaves ~113 mm unexplained; the envelope is not the mechanism.

The corpus's leading hypothesis is the one §1 sharpens: **the robot's own base
entering the octomap as world occupancy.** Excluding `base_link` from
`collision_geometry` is why nothing downstream knows the chassis is the robot,
and `sim_sensor_bridge.voxel_backing_record` already marks a cell
`self_occupancy_suspect` when *"the robot's own body, base and mount included"*
backs it. The utensil evidence voxel sits 195 mm from the link-1 origin at
mobile-base height. That is a *map* defect, not a *primitive* defect; one
`voxel_backing_record` call per characterised stop confirms or kills it, and
should be run before any geometry change is contemplated.

---

## 7. Cost, and what a safety-WG review would have to assert

### 7.1 Schema (§1.6)

**Multi-primitive links: no schema change at all** (§3.1).

**Sphere-swept box: additive, but not silently.** `BoxShape` carries
`extra="forbid"`, so adding `corner_radius_m: float = Field(default=0.0, ge=0.0)`
is **backward-compatible** (every existing manifest loads unchanged; `0.0`
reproduces today's box bit for bit through every distance routine, §2.3 — no
migrator) but **not forward-compatible**: a manifest that *uses* the field fails
to load on an older reader, which is fail-closed but still a real break for
anyone pinning an older `openral-core` against a newer `robots/` tree. Under
§1.6 it may evolve in place *provided* the release notes state that coupling; a
`RoundedBoxShape` union member has the same property and buys nothing. Either
way `docs/methods/00-core-schemas.md` and `docs/methods/06-…` need updating.

### 7.2 Kernel and lowering

| change | files | size | safety surface |
|---|---|---|---|
| multi-primitive lowering | `envelope_loader._capsules_by_link` + emit loop; `urdf_lowering.lower_link_geometry` | ~60 lines Python | **none new** — the kernel path is already exercised by `test_kernel_h1_self_collision` |
| `Obb::radius` | `collision.hpp`, 6 call sites in `collision.cpp`, 3 in `lifecycle_kernel.cpp` (§2.4) | ~25 lines C++ | one line that can make the kernel *unsafe* — the voxel broad-phase `reach` (§2.3) |
| downstream box readers | `depth_height_filter_node._collision_z_span`; `tools/viz_collision.py` | ~10 lines | the SLAM height band must add the radius or the band under-covers (`_capsule_segment_radius` is gone — #155) |
| per-robot refits | the three box-bearing manifests: `panda_mobile` (7 boxes), `panda_mobile_vslam` (7), `so101_follower` (5) | data | each needs its own containment proof re-run |

### 7.3 What the hazard entry and the safety-WG review would have to assert

1. **Containment, per link, as a proof and not a sample** — every vertex of the
   link's collision mesh inside the refitted primitive, achieved margin reported
   (method in §4).
2. **An explicitly declared headroom.** The shipped margins are 0.012–0.203 mm
   (§4.3) — accidental, not designed. A refit must state the headroom it chose
   (1 mm is a defensible default) and the achieved margin against it.
3. **`radius = 0` is bit-identical to today** — a regression test, not a claim.
4. **The voxel broad phase was widened**, by name, with a test that fails
   without it: a cell intersecting the swept box but outside the shrunken box's
   AABB must still be visited. It is the only way the change can *miss* a
   collision rather than merely tighten one.
5. **The SAT bound is unchanged in character** — `box_box_distance` stays a
   conservative lower bound after the offset (§2.3).
6. ~~**`_capsule_segment_radius` is fixed first, or the ACM is not
   regenerated.**~~ **Discharged** — #155 is fixed and a regeneration reproduces
   every shipped ACM byte-identically (all 10 manifests).
7. **The recovery claim is the one in §6, not a larger one.** Justifying a
   *less* conservative envelope on a benefit the evidence does not show is the
   exact failure mode §1.2 exists to prevent.

### 7.4 Hazard-log Entry 012 lockstep — **not touched**

Entry 012 obliges `support_contact_exempts` and the bridge's
`payload_clearing.support_patch_withholds` to move together so `withheld ⊆
exempt` stays true by construction. `support_contact_exempts` reads no
`model.boxes`, and `payload_clearing.cpp`'s `surface_distance` /
`bounding_radius` switch on the **attached payload's**
`PayloadPrimitive::shape_type` — so a robot-link `Obb` radius engages neither.
**Conditional on scope:** extended to `AttachedPrimitive` (a rounded *payload*
box) it would touch both, since `bounding_radius` would under-report the AABB
span exactly as the voxel broad phase does (§2.3). **Do not extend it to
payloads.** One in-scope interaction still belongs in the hazard entry:
`check_attached_self_collision` compares payload primitives against robot link
boxes (`collision.cpp:1369-1374`), so a link-box radius makes that check less
conservative too, bounded by the same per-link Δ.

---

## 8. Recommendation

**8.1 Do not do the primitive change yet — and possibly not at all.** §6
governs: the tightening is provably containing and **no characterised stop
recovers**, and a less conservative safety envelope for a benefit the evidence
does not demonstrate is not a trade this repo should make (§1.1, §1.2). First
run `voxel_backing_record` and settle the self-occupancy hypothesis (§6.3), then
apply §6.1's rule to the layout census when it lands. If the census shows stops
reporting shallower than their link's ceiling — say under 10 mm on link1/2/3/4 —
the calculus changes and §8.2 becomes worth doing.

**8.2 If it is done, do the faces-fixed sphere-swept box, not the free fit.**
`a = h − r` with a per-link `r` is a strict improvement whose safety argument
fits in a paragraph. The free fit is 4–13 mm tighter at the corner but grows
faces by up to 11.7 mm, making it a *different* envelope rather than a smaller
one.

**8.3 Do not pursue multi-primitive links for this problem.** It is the cheaper
change with the smaller safety surface, and it does not work: six of seven
collision meshes are already convex hulls (§4.3), and every split tried was
equal to or worse than a single swept box at 1.0–1.8× the volume. Lifting the
`_capsules_by_link` restriction is still worthwhile as *removing an asymmetry
between the two lowering paths*, but not as a fix for envelope slop.

**8.4 ~~Fix issue #155 regardless.~~ Done**, and it strengthened this document's
recommendation: `panda_link5`↔`panda_link7` is a real self-collision the box
envelopes cannot distinguish from an artifact at any margin.

**Retired by [issue #191](https://github.com/OpenRAL/openral/issues/191), and
this section called it correctly.** Tighter geometry was the only thing that
could — but not the tighter *box* §8.2 proposes: the corner-reach reduction
available there is 7.5 mm on link5 and 1.9 mm on link7 (§4.3's table) against
the ~28 mm the separation needs. What retired it is the exact hull, via the
[staged narrow phase](collision-hull-narrow-phase.md) extended from world voxels
to self-pairs, and §4.3's convexity column is why it works: link5 and link7 are
convex to 1e-4 in volume ratio, so `conv(mesh)` **is** the mesh and the hull's
verdict is the mesh's. Over the pair's full `(joint6, joint7)` grid the
false-positive population goes from 11 680 of 14 641 poses to **zero**.

**8.5 Fix `urdf_lowering.lower_link_geometry`'s mesh path regardless.** It still
emits a PCA capsule for a mesh collision, which §4.2 measures at 96–112 mm of
protrusion against the shipped boxes' 27–76 mm. Re-lowering `panda_mobile` from
its URDF today would inflate the envelope slop by up to 2.1× and silently undo
the #103 conversion. Whether or not §8.2 lands, a mesh should lower to a box.

> **This is not hypothetical, and #191 hit it.** `render_cumotion_config` was
> sourcing its collision spheres from `LoweredCollisionModel.collision_geometry`
> — i.e. from exactly this PCA-capsule path — rather than from the manifest the
> kernel checks, so "plan-time and kernel-time share one source of truth" was
> untrue for every hand-authored box manifest. It now prefers the manifest. The
> lowering path itself is still unfixed.

---

## 9. Reproducing the numbers

The scripts are not checked in (one-shot analysis; §1.11 keeps fixtures for
tests rather than studies), but are reconstructible from this section.

**Inputs.** robosuite `1.5.2`'s `models/assets/robots/panda/robot.xml` under
`mujoco 3.8.0`; the collision geoms are the `link<N>_collision` mesh geoms
(`contype`/`conaffinity` non-zero — exactly one per link body, `link1`…`link7`).
Manifest values from `robots/panda_mobile/robot.yaml`. Vertices come from
`model.mesh_vert` placed by each geom's `geom_pos`/`geom_quat`, matching
`sim_sensor_bridge._body_collision_points` exactly.

**Signed distances.** Box: `‖max(|p| − a, 0)‖ + min(max_k(|p_k| − a_k), 0)`.
Sphere-swept box: the same, minus `r`. Capsule: `‖p − clamp_z(p)‖ − r`.

**Protrusion.** The primitive's boundary is sampled (12 000 points; flat faces
directly, the rounded shell as box-surface points pushed along random outward
directions and filtered to `|dist(·, Box(a)) − r| < 1e−9`), then
`trimesh.proximity.closest_point` against the collision mesh; maximum reported.

**SSB fitting.** Faces-fixed: `a = h − r`, `r` swept at 1 mm (0.1 mm for §4.3's
headroom check), largest containing `r` taken. Free fit: SLSQP minimising
`Σ a_k` subject to `max_p dist(p, Box(a)) ≤ r − headroom`, then a monotone
inflation loop until containment holds exactly — an optimiser's convergence
tolerance is not a safety argument. **Multi-primitive:** triangles (not
vertices) are assigned to one part, by equal-mass slabs along the link's longest
shipped half-extent and by `kmeans2` on triangle centroids; each part's three
vertices are checked against that part alone.

Environment: workspace `.venv` (numpy 2.2.6, scipy 1.17.1, trimesh 4.12.2,
mujoco 3.8.0, robosuite 1.5.2). CPU only.
