# Collision-primitive study — is the OBB the reason the arm reads as in-collision?

> **Status: analysis only.** Nothing in this document has landed. It touches no
> code in `cpp/openral_safety_kernel/` or `packages/openral_safety/`, both of
> which are safety-WG gated and need a hazard-log entry before a change. What
> follows is the evidence a reviewer would need in order to decide.
>
> Companion document: [Collision-stack validation
> evidence](collision-validation-evidence.md), which is where the *failure*
> census lives. This document is the *geometry* half.

Measured on `e1b9915`. Every number below is reproducible by the method in
[§9](#9-reproducing-the-numbers).

---

## 1. The premise this study was opened on is wrong, and that is the first result

The study was opened on the hypothesis that the kernel's oriented boxes are
badly proud at their corners, that the **mobile base** in particular is really a
rounded rectangle and is being modelled as a sharp one, and that a sphere-swept
box would fix it.

The geometry of that hypothesis is correct. Its applicability to `panda_mobile`
is not, for a reason that has nothing to do with corners:

**`panda_mobile`'s collision model contains no base.** `robots/panda_mobile/robot.yaml`
declares collision geometry for `panda_link1` … `panda_link7` and for nothing
else, and says so in a comment that is explicit about why:

> Scope: the ARM LINKS (1–7) — the parts that must never strike the
> environment. Deliberately EXCLUDED, so the world-voxel check is usable for
> real picks in a cluttered kitchen:
> * `base_link` — the mobile base parks ~1 cm from cabinets; base-vs-world is
>   Nav2's 2-D costmap job, so a base capsule here only false-positives on
>   furniture beside the parked base.
> * `panda_finger_pair` — the gripper is the intended-contact part (it grasps
>   the target); checking it against the world would veto every grasp.

So the chassis question is closed before it is asked. The chassis is owned by
Nav2 in 2-D; the safety kernel never places a primitive there; **a better
chassis primitive buys exactly zero.** Whatever is making kitchen layouts read
as in-collision at their start state is one of the seven arm links or is not a
link at all.

That redirects the rest of the study, and it also sharpens the question worth
answering: *which arm link, and can a better primitive on that link fix it?*

---

## 2. What the kernel supports today

### 2.1 The lowered model

`CollisionModel` (`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`)
is two flat, parallel primitive arrays, each tagged with the link it rides on:

| array | element | fields |
|---|---|---|
| `capsule_link[c]` / `capsules[c]` | `Capsule` | `radius`, `half_length`, `origin` |
| `box_link[b]` / `boxes[b]` | `Obb` | `half_extents`, `origin` |

A sphere is a capsule with `half_length == 0`; that is how
`envelope_loader.collision_params_from_description` lowers `SphereShape`. There
is no third array and no mesh path — the kernel is deliberately
analytic-convex-only so the hot path stays allocation-free.

### 2.2 What the collision routines actually compute

| routine | box↔? | what it computes | exact? |
|---|---|---|---|
| `capsule_distance` | — | segment–segment distance − both radii | exact |
| `box_capsule_distance` | box↔capsule | 48-step ternary search of point→AABB distance along the capsule segment, in box-local coordinates, − capsule radius | exact when disjoint |
| `box_box_distance` | box↔box | separating-axis theorem over 15 axes; returns the **largest** per-axis gap | conservative **lower bound** |
| `check_voxel_collision` | box↔voxel | each occupied cell is an axis-aligned cube, then `box_box_distance` | conservative |

`fold_pair` / `finish_sweep` are pure bookkeeping: `fold_pair` accumulates
`sweep_min` over **every** pair the check touches, and separately promotes a
pair to the reported evidence only when it both `tripped` and is deeper than the
pair already recorded, so `link_a` / `link_b` / `min_distance` always describe
one and the same pair. `finish_sweep` publishes the sweep-wide minimum and, when
nothing tripped, lets `min_distance` keep its clearance meaning. Neither touches
geometry, so **neither is affected by a primitive change.** Confirmed by reading
`collision.cpp:480-523`, not assumed.

### 2.3 Would a radius on the box path be nearly free? Yes — verified, with one trap

The hypothesis in the brief was that a sphere-swept box's distance to anything
is `box_distance(shrunken) − r`, so adding a radius is a subtraction. Checked
against the real code, that holds for both distance routines, for a reason worth
writing down because it is what makes the change reviewable:

**Box↔capsule.** A sphere-swept box is the Minkowski sum `Box(a) ⊕ B(r)`. For
convex sets the Minkowski sum with a ball is exactly an offset of the distance
function, so `d(Box(a) ⊕ B(r), C) = d(Box(a), C) − r` whenever the two are
disjoint. `box_capsule_distance` already returns `d(Box(a), segment) − cap_r`,
so the swept version is that value minus `r`. **Exact, one subtraction.**

**Box↔box.** The SAT loop projects both boxes onto a candidate axis `n` and
takes `|dc·n| − ra − rb`, where `ra = Σ_k a_k |â_k·n|` is the box's support
radius along `n`. The support radius of `Box(a) ⊕ B(r)` along **any** unit `n`
is exactly `Σ_k a_k |â_k·n| + r` — the `+ r` is axis-independent. So the per-axis
gap for two swept boxes is `g_n − r_a − r_b`, and since the routine returns
`max_n g_n` and both radii are constants, the swept answer is
`box_box_distance(shrunken_a, shrunken_b) − r_a − r_b`. **The 15-axis set stays
exactly as conservative as it is today** — it was already a subset heuristic, and
the offset does not change which axis wins. One subtraction.

**The trap: the voxel broad phase.** `check_voxel_collision` (`collision.cpp:684-703`)
computes the box's world AABB from `half_extents` alone —

```cpp
const double ex = |R[0]|*he.x + |R[1]|*he.y + |R[2]|*he.z;   // and ey, ez
const double reach = margin + half_side;
```

— and only visits cells inside that inflated window. With shrunken half-extents
`a = h − r`, `ex/ey/ez` under-report the true extent by exactly `r`, and cells
that genuinely intersect the swept box are **never visited at all**. That is not
a conservatism loss, it is a *missed collision*: the check silently returns
clear. The fix is one term (`reach = margin + half_side + boxes[b].radius`) and
it is trivially correct, but it is the single line in the whole change that can
make the kernel unsafe rather than merely tighter, and a reviewer should be
pointed straight at it.

### 2.4 Every site that would move

Robot-link box geometry is read in four collision routines plus the loader:

| file | site | change |
|---|---|---|
| `collision.hpp` | `struct Obb` | `+ double radius{0.0}` |
| `collision.cpp:570` | `check_self_collision`, box↔capsule | `− boxes[bi].radius` |
| `collision.cpp:581` | `check_self_collision`, box↔box | `− boxes[bi].radius − boxes[bj].radius` |
| `collision.cpp:617` | `check_world_collision`, box↔world-capsule | `− boxes[b].radius` |
| `collision.cpp:689-697` | `check_voxel_collision` broad phase | **`reach += boxes[b].radius`** (§2.3) |
| `collision.cpp:716` | `check_voxel_collision`, box↔voxel | `− boxes[b].radius` |
| `collision.cpp:1369-1374` | `check_attached_self_collision`, payload↔link-box | `− model.boxes[b].radius` |
| `lifecycle_kernel.cpp:169,1364,1426` | parameter declare + load | `+ collision_box_radius` array |

`fold_pair`, `finish_sweep`, `forward_kinematics`, `jacobian_dls_step`,
`support_contact_exempts`, `update_support_contact_witnesses`,
`place_approach_allowance` and `check_attached_voxel_collision` are all
untouched — none of them reads `model.boxes`.

---

## 3. Multi-primitive links: **the kernel already supports them; the manifest lowering rejects them**

This was flagged in the brief as a suspicion. It is confirmed, and the answer is
better than expected: the limitation is one function in the Python lowering, not
the kernel.

### 3.1 The rejection is real

Executable, on `e1b9915`:

```
$ PYTHONPATH=packages/openral_safety python -c "<duplicate panda_link1 in the manifest>"
pydantic entries: 8
LOWERING REJECTED: ROSConfigError link 'panda_link1' has >1 collision primitive;
                   split it into separate links (unsupported in this lowering phase)
```

Two things follow immediately:

* **Pydantic accepts it.** `RobotDescription.collision_geometry` is a plain
  `list[LinkCollisionGeometry]` with no uniqueness constraint on `link_name`.
  The eight-entry manifest validated. **No schema change is needed to express a
  multi-primitive link.**
* The refusal is `envelope_loader._capsules_by_link`, which builds a
  `dict[link_name → geometry]` and raises on the second entry.

### 3.2 The kernel is fine with it — and a shipped test proves it

* `CollisionModel`'s own docstring: *"A link may carry zero, one, or **several**
  capsules (real MJCF bodies often have several collision geoms); `capsule_link[c]`
  names the link capsule `c` is rigidly attached to."*
* `load_collision_model` (`lifecycle_kernel.cpp:1402-1431`) only range-checks the
  link index. Repeated indices are legal.
* `check_self_collision` skips same-link pairs explicitly (`li == lj` for
  capsules, `lb == lb2` for boxes, and the box↔capsule loop skips `lb == lc`), so
  two primitives on one link never self-collide with each other.
* `mjcf_lowering.lower_collision_params` **already emits several capsules per
  link** — *"Every collidable primitive on the body becomes one capsule tagged
  with this link's index (multi-capsule per link)."*
* `tests/sim/safety/test_kernel_h1_self_collision.py:128` asserts
  `len(cap_links) > len(set(cap_links))` — "expected ≥1 link with multiple
  capsules" — and then starts a **live kernel** against that model.

So the multi-primitive path is not theoretical. It is exercised on every H1 run.
`panda_mobile` cannot use it only because `panda_mobile` comes through the
*manifest* path (URDF-lowered, hand-committed `collision_geometry`) rather than
the MJCF path.

### 3.3 What lifting it costs

Small, and none of it is in the kernel:

1. `_capsules_by_link` → `_primitives_by_link`, returning
   `dict[str, list[LinkCollisionGeometry]]`; drop the raise.
2. `collision_params_from_description`'s emit loop iterates that list instead of
   a single `geom`. The flat arrays it builds are already per-primitive with a
   link tag, so nothing downstream changes shape.
3. `urdf_lowering.lower_link_geometry` currently unions **all** of a link's
   `<collision>` elements into **one** PCA capsule
   (`urdf_lowering.py:252-264`). To emit several it must stop unioning. This is
   the real work, and it is a generator change with no safety surface of its own
   — the safety property is whatever the emitted primitives cover, checked at
   validation time.
4. The ACM sweep still keys on a single shape per link (`geoms[ln].shape`); it
   would need to fold several primitives per link. (§5's separate defect there —
   issue #155 — is now **fixed**; the sweep asks
   `kernel_predicates.shape_distance` for the true primitive.)
5. `openral_slam_bringup._collision_z_extent_m` iterates `collision_geometry`
   entries, so it already handles repeats without change.

**No schema change. No kernel change. No new C++.** That is a materially smaller
safety surface than adding a radius to `Obb`, and it is why §7 recommends it
first.

---

## 4. The fit table

### 4.1 Method

Ground truth is robosuite's panda collision meshes — one `link<N>_collision`
mesh geom per link, exactly the geoms the kernel's OBBs are documented as
enclosing and the same set `sim_sensor_bridge._body_collision_points` samples.
Every candidate is fitted in the link frame and measured against that mesh.

Two numbers per link per candidate:

* **max protrusion** — the one-sided Hausdorff distance from the primitive's
  *boundary* to the mesh *surface*. This is not a proxy for the slop; it **is**
  the worst-case distance under-report. For an external query `q` approaching
  the primitive boundary at `x`, the kernel reports `d(q, prim)` while the truth
  is `d(q, mesh)`, and the error tends to `dist(x, mesh)`. Lower is tighter.
* **min containment margin** — the minimum inward depth of the mesh inside the
  primitive. `>= 0` is containment.

The margin is a **proof, not a sample**, for every convex candidate: a convex
primitive containing every mesh vertex contains `conv(vertices) ⊇ mesh`. For the
multi-primitive candidates the union is not convex, so whole *triangles* are
assigned to one part each and that part's three vertices are checked — convexity
then proves the triangle, hence the mesh. No candidate below was accepted on
sampled evidence.

Cross-check against the published figure: `panda_link1`'s `corner_slop_m` is
recorded as **53.4 mm** in
[collision-validation-evidence.md](collision-validation-evidence.md); this study
measures 52.6 mm of max protrusion on the same link. They agree, and they should
differ in exactly this direction — `corner_slop` measures to the nearest sampled
mesh *vertex* (never closer than the surface), from 8 corners only.

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

**The sphere-swept box works, and the brief's construction is admissible.** The
brief proposed keeping the face reach at `h` by shrinking to `a = h − r` and
sweeping `r`. That is not automatically containing — a point near the box corner
is up to `√3·r` from `Box(h − r)`, not `r` — so it had to be tested rather than
assumed. It holds, up to a per-link maximum `r` of 3–41 mm, with containment
proven over the full vertex set. Protrusion falls **8 % (link7) to 58 %
(link4)**, faces are untouched by construction, and volume falls slightly on
every link. **It is a strict improvement: never more conservative anywhere than
today, less conservative only at the corners.**

**The free fit is tighter but not strictly better.** Letting the optimiser also
move the outer extents reaches 17–21 mm of protrusion on the mid-arm links, but
it pays for that by growing some faces — up to **+11.7 mm on link2**, +7.2 on
link3, +6.7 on link7. On a face-on approach it is *more* conservative than
today. It is the better *number* and the worse *guarantee*; §7 does not
recommend it.

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

**The capsule is catastrophic here — 96–112 mm on every link but the wrist.**
None of the seven links is rod-like enough for one; the Panda's links are
flanged, tapered blocks, not rods. This retroactively vindicates #103's
conversion from capsules to boxes, and it is worth recording because
`urdf_lowering.lower_link_geometry` **still emits a PCA capsule for any mesh
collision** — so re-lowering `panda_mobile` from its URDF today would regress
the envelope by 1.3–2.1×. That is a live trap, not a historical note.

**Multi-primitive decomposition does not help these links, and the measurement
says why.** Both splits were tried (equal-mass slabs along the link's long axis,
and k-means on triangle centroids), at k = 2 and k = 3, and neither beats a
single sphere-swept box on any link. Several are *worse* than the single OBB
(link2 46.5 vs 45.2; link6 51.3 vs 49.6; link7 39.4 vs 27.3 under k-means), and
volume rises 1.0–1.8×. The reason is measurable in one column:

| link | link1 | link2 | link3 | link4 | link5 | link6 | link7 |
|---|---|---|---|---|---|---|---|
| convex-hull volume ÷ mesh volume | **1.303** | 1.000 | 1.0003 | 1.0001 | 1.000 | 1.0003 | 1.0002 |

**Six of the seven collision meshes already *are* convex hulls.** There is
nothing to decompose. Convex decomposition pays when a link is non-convex — an
L-bend, a fork, a yoke — and the slop it removes is the concavity. The slop here
is not concavity; it is the *corner* of a bounding box around a convex, rounded,
tapered solid, and every sub-box you cut it into has corners of its own. Only
`panda_link1` is meaningfully non-convex (30 % hull excess), and even there the
2-slab split (45.9 mm) merely ties the single swept box (45.5 mm) at 1.1× the
volume.

This refutes the prior that a tapered arm link "decomposes naturally into two or
three". It would, if the kernel were fitting the *link*. It is fitting the
link's *convex hull*, and that changes the answer.

**The shipped OBBs carry no padding to give back.** Every containment margin in
the table is 0.012–0.203 mm — tens of microns. The shipped boxes are tight fits,
not padded ones. There is no safety pad to trim: the slop is entirely the box's
own corner geometry. Any refit should therefore *declare* an explicit headroom
rather than inherit the current accidental one (§7.3).

---

## 5. Named finding: **a box has no radius, and every scalar surrogate for one has now bitten us**

This is the most transferable thing the study produced, and it is stated here as
a finding rather than as a footnote to a table.

Three separate places in this repo have substituted a single scalar radius for a
box's three half-extents. Each picked a different surrogate; each was wrong in a
different direction; **none of the three was conservative at both edges.**

| site | surrogate | direction | consequence |
|---|---|---|---|
| the safety kernel's OBB, as an envelope for a rounded, tapered link | *circumscribed*, implicitly — a sharp box reaches `\|h\|` at its corners | **over**-approximates | corners proud of the real link (§4); the kernel reads in-collision while the surface is clear |
| ~~`urdf_lowering._capsule_segment_radius` (offline ACM sweep)~~ — **deleted, issue [#155](https://github.com/OpenRAL/openral/issues/155) fixed** | was *inscribed*, `min(half_extents_m)` | **under**-approximated | The sweep now uses `kernel_predicates.shape_distance` — the kernel's own predicate on the true primitive. The remaining capsule lowering, `bounding_capsule_segment`, **circumscribes** and feeds only the cuMotion planner, where over-covering is the safe direction. See the note below: #155's conclusion was inverted, and the replacement numbers are grid-exhaustive rather than sampled. |
| `openral_slam_bringup.depth_height_filter_node._collision_z_span` | neither — it was forced to take the **exact** OBB support projection `Σ_k \|R[2][k]\|·h_k` | — | the node's own docstring records why: the inscribed radius *"shrinks the band and hides obstacles at body height"*, the circumscribed radius *"grows it … and the node re-marks the floor it exists to remove"*, so **"no single scalar is conservative at both edges"** |

The SLAM node is the one that got it right, and it got it right by refusing to
pick a scalar at all. That is the generalisation:

> **A box's extent is direction-dependent. Any code that reduces it to one
> number is choosing a direction to be wrong in, and the two available choices
> are wrong in opposite directions. Use the support projection
> `Σ_k |û·ê_k|·h_k`, or use the true box.**

The middle row's lesson survives its own correction, and is worth restating
because the original version of this section got the consequence backwards.

"Conservative" is not a property of a single predicate; it is a property of the
pipeline. That much was right. What was wrong was the claim that
under-approximating in the ACM sweep is safe "for the ACM in isolation", and the
inference that `panda_link5`↔`panda_link7` was a capsule-junction pair the sweep
ought to be restoring.

**It is not a junction artifact — the pair genuinely collides.** Measured on the
URDF's own collision meshes over `(panda_joint6, panda_joint7)`, the only two
joints that move it: **914 of 14641 poses interpenetrate, up to 48.3 mm deep**,
across a **39.6° band of `joint6`** spanning the full `joint7` range. MoveIt
agrees and emits no `Never` row. So the committed ACM has been *exempting* a real
self-collision, and the inscribed-sphere bug was producing the correct ACM for
the wrong reason.

It stays exempt, because the box envelopes cannot express it either way: the box
check fires on **86.5%** of that space while only 6.9% is real (**79.6% false**),
including the arm's `ready` home pose at -9.14 mm, and **no margin separates the
populations** — real collisions reach -9.97 mm, collision-free poses -40.07 mm.
The exemption is now a hand-owned `reason="User"` row in each panda SRDF carrying
this evidence, not something a tool manufactures. **This is the strongest single
argument in this document for tighter link geometry**: it is a live unchecked
self-collision that only envelope precision can retire.

This matters for the recommendation. If the answer to "how do we stop the arm
reading as in-collision" is a shape change, it should be a change that **stops
substituting**, not one that substitutes a better single scalar.

---

## 6. What it would actually recover — **on the present evidence, nothing**

### 6.1 The decision rule

A stop is recoverable by an envelope tightening if and only if the reach removed
**on the reporting link, in the direction of the reporting cell** exceeds the
penetration the kernel reported. The reach removed is maximal at the corner and
zero on the faces, so the *corner-reach reduction* from §4.3 is a strict upper
bound on it:

> A stop reporting `min_distance = −d` recovers only if
> `d < Δcorner_reach(reporting link)`.

Ceilings, faces-fixed variant: **link1 12.9 · link2 10.8 · link3 22.9 ·
link4 23.9 · link5 7.5 · link6 4.7 · link7 1.9 mm.**

That is an upper bound in two further ways: the cell is almost never at the
exact corner, and the voxel half-diagonal (21.7 mm at the 25 mm sim grid) is
untouched by any primitive change. The realised recovery is smaller than these
numbers, never larger.

### 6.2 Applying it to the characterised stops

The layout census is being produced separately and is not in hand; the rule
above is stated so it can be applied to that table when it lands. Applied to the
link-class stops that
[collision-validation-evidence.md](collision-validation-evidence.md) has already
characterised:

| stop | link | kernel `min_distance` | ceiling for that link | recovers? |
|---|---|---|---|---|
| baguette, 2026-08-22 | `panda_link5` | **−20.9 mm** | 7.5 mm | **no** — short by 13.4 mm |
| utensil, 2026-08-22 / -23 | `panda_link1` | **−17.3 mm** | 12.9 mm | **no** — short by 4.4 mm |
| harness-2 fridge | `panda_link7` | **−24.7 mm** | 1.9 mm | **no** — short by 22.8 mm (and already adjudicated *real contact*) |

**None of the three recovers.** Even the free-fit variant, which is not
recommended because it grows faces, lifts the link5 ceiling only to 8.9 mm and
the link1 ceiling to 17.0 mm — still under 20.9 and just under 17.3.

This is the study's decisive result and it should be stated plainly: **on every
stop this repo has characterised, a tighter primitive does not clear the stop.**
The corner slop is real, it is large, and it is *not* what is holding those
layouts. A reviewer should read §4 as "the envelope can be improved by 13–58 %
and here is the proof", and §6 as "and that improvement recovers no layout we
currently know about".

### 6.3 What that leaves

The baguette stop is the one the evidence corpus calls unexplained: 120.9 mm of
discrepancy against an 88.2 mm admissible gap, an *untruncated* ground-truth
probe that returned **no pair at all** within 100 mm. Removing 7.5 mm of link5
corner reach leaves ~113 mm unexplained. The envelope is not the mechanism.

The corpus's own leading hypothesis is the one this study's §1 finding makes
sharper rather than weaker: **the robot's own base entering the octomap as world
occupancy.** `panda_mobile` excludes `base_link` from `collision_geometry` — and
that exclusion is precisely why nothing downstream knows the chassis is the
robot. The instrumentation for this already exists and already names the class:
`sim_sensor_bridge.voxel_backing_record` classifies a cell as
`self_occupancy_suspect` when *"the robot's own body, base and mount included"*
backs it, and its docstring says *"the depth self-filter is supposed to make
these transparent, so a hit here means the robot is in its own world map."* The
utensil evidence voxel sits 195 mm from the link-1 origin at mobile-base height.

That is a hypothesis this study did not test and is not equipped to settle — it
belongs to the failure census, and it is a *map* defect, not a *primitive*
defect. But it is where the evidence points, and it costs one `voxel_backing_record`
call per characterised stop to confirm or kill. It should be run before any
geometry change is contemplated.

---

## 7. Cost, and what a safety-WG review would have to assert

### 7.1 Schema (§1.6)

**Multi-primitive links: no schema change at all.** `collision_geometry` is
already `list[LinkCollisionGeometry]` with no uniqueness constraint on
`link_name`, and an eight-entry `panda_mobile` with a duplicate link validated
(§3.1).

**Sphere-swept box: additive, but not silently.** `BoxShape` carries
`model_config = ConfigDict(extra="forbid")`. Adding
`corner_radius_m: float = Field(default=0.0, ge=0.0)` is:

* **backward-compatible** — every existing manifest loads unchanged, and
  `0.0` reproduces today's box exactly, bit for bit, through every distance
  routine (§2.3). No migrator is needed for existing data.
* **not forward-compatible** — `extra="forbid"` means a manifest that *uses* the
  field fails to load on an older reader with a validation error rather than a
  degraded model. That is the fail-closed direction, but it is a real
  compatibility break for anyone pinning an older `openral-core` against a newer
  `robots/` tree.

The honest reading of §1.6 is that this is a backward-compatible addition that
may evolve in place (no `schema_version` bump, no migrator), *provided* the
release notes state that manifests using `corner_radius_m` require the matching
core version. The alternative — a new `RoundedBoxShape` union member — has the
same forward-compatibility property and adds a discriminator value, so it buys
nothing.

Also required either way: `docs/methods/00-core-schemas.md` (`BoxShape`,
`CollisionShape`) and `docs/methods/06-…` (`collision_params_from_description`),
per §1.13.

### 7.2 Kernel and lowering

| change | files | size | safety surface |
|---|---|---|---|
| multi-primitive lowering | `envelope_loader._capsules_by_link` + emit loop; `urdf_lowering.lower_link_geometry` | ~60 lines Python | **none new** — the kernel path is already exercised by `test_kernel_h1_self_collision` |
| `Obb::radius` | `collision.hpp`, 6 call sites in `collision.cpp`, 3 in `lifecycle_kernel.cpp` (§2.4) | ~25 lines C++ | one line that can make the kernel *unsafe* — the voxel broad-phase `reach` (§2.3) |
| downstream box readers | `depth_height_filter_node._collision_z_span`; `tools/viz_collision.py` | ~10 lines | the SLAM height band must add the radius or the band under-covers (`_capsule_segment_radius` is gone — #155) |
| per-robot refits | the three box-bearing manifests: `panda_mobile` (7 boxes), `panda_mobile_vslam` (7), `so101_follower` (5) | data | each needs its own containment proof re-run |

### 7.3 What the hazard entry and the safety-WG review would have to assert

1. **Containment, per link, as a proof and not a sample.** For every refitted
   primitive: every vertex of the link's collision mesh lies inside it, with the
   achieved margin reported. §4 supplies the method; the entry must supply the
   numbers for the exact shipped values.
2. **An explicitly declared headroom.** The shipped OBBs' margins are 0.012–0.203
   mm (§4.3) — accidental, not designed. A refit must state the headroom it
   chose (1 mm is a defensible default) and show the achieved margin against it,
   so the next person does not inherit another accidental number.
3. **`radius = 0` is bit-identical to today.** Every distance routine reduces to
   the current expression at `r = 0`; this is a regression test, not a claim.
4. **The voxel broad phase was widened.** Explicitly, by name, with a test that
   fails without it: a cell that intersects the swept box but lies outside the
   shrunken box's AABB must still be visited. This is the only way the change
   can *miss* a collision rather than merely tighten one.
5. **The SAT bound is unchanged in character.** `box_box_distance` stays a
   conservative lower bound after the offset (§2.3), so §3's "at least as
   conservative" obligation is met axis-for-axis.
6. ~~**`_capsule_segment_radius` is fixed first, or the ACM is not
   regenerated.**~~ **Discharged** — issue #155 is fixed and the helper is gone.
   A regeneration now reproduces every shipped ACM byte-identically (all 10
   manifests), so a refit no longer risks silently dropping a pair. The
   sequencing constraint it imposed on this work is lifted.
7. **The recovery claim is the one in §6, not a larger one.** No layout in the
   present corpus recovers. A hazard entry that justifies a *less* conservative
   envelope on a benefit the evidence does not show is the exact failure mode
   §1.2 exists to prevent.

### 7.4 Hazard-log Entry 012 lockstep — **not touched**

Entry 012 obliges the kernel's `support_contact_exempts` and the bridge's
`payload_clearing.support_patch_withholds` to move together, so that
`withheld ⊆ exempt` stays true by construction. Checked against both sources:

* `support_contact_exempts` (`collision.cpp`) takes an `AttachedObject`, the
  attested support plane, and a cell centre. It reads no `model.boxes`.
* `payload_clearing.cpp`'s `surface_distance` and `bounding_radius` switch on
  `PayloadPrimitive::shape_type` — the **attached payload's** wire primitives
  (`AttachedCollisionObject.SHAPE_*`), not the robot's link geometry.

A radius on the robot-link `Obb` therefore touches **neither**, and the lockstep
obligation is not engaged. **This is conditional on scope:** if the change were
extended to `AttachedPrimitive` (a rounded *payload* box), it would touch both —
`bounding_radius` would under-report the AABB span exactly as the kernel's voxel
broad phase does (§2.3), and the two predicates would have to move in the same
commit. **Recommendation: do not extend it to payloads.** There is no measured
motivation to, and it converts a one-package change into an Entry-012 change.

One real, in-scope interaction to declare: `check_attached_self_collision`
compares payload primitives against robot link boxes
(`collision.cpp:1369-1374`). A link-box radius makes *that* check less
conservative too. It is bounded by the same per-link Δ as everything else, but
it belongs in the hazard entry because it is payload-adjacent and a reviewer
scanning for Entry-012 exposure will look there.

---

## 8. Recommendation

**8.1 Do not do the primitive change yet — and possibly not at all.** §6 is
the governing result: the envelope can be tightened by 8–58 %, the tightening
is provably containing, and **no characterised stop recovers.** Landing a less
conservative safety envelope for a benefit that the evidence does not
demonstrate is not a trade this repo should make (§1.1, §1.2). The right
sequence is:

1. Run `voxel_backing_record` on the characterised stops and settle the
   self-occupancy hypothesis (§6.3). It is cheap, it is already built, and it
   is where the evidence points.
2. Wait for the layout census, then apply §6.1's rule to it. If the census shows
   a population of stops reporting shallower than their link's ceiling — say
   under 10 mm on link1/2/3/4 — the calculus changes and §8.2 becomes worth
   doing.
3. Only then propose the geometry.

**8.2 If it is done, do the faces-fixed sphere-swept box, not the free fit.**
`a = h − r` with a per-link `r` is a strict improvement — never more conservative
anywhere than today, less only at the corners — and it is the variant whose
safety argument fits in a paragraph. The free fit is 4–13 mm tighter at the
corner but grows faces by up to 11.7 mm, which makes it a *different* envelope
rather than a smaller one, and no reviewer should have to reason about both
directions at once.

**8.3 Do not pursue multi-primitive links for this problem.** It is the cheaper
change with the smaller safety surface, and it does not work: six of the seven
collision meshes are already convex hulls (§4.3), so there is no concavity to
decompose, and every split tried was equal to or worse than a single swept box at
1.0–1.8× the volume. Lifting the `_capsules_by_link` restriction is still
worthwhile on its own merits — it is a real limitation, the kernel has supported
it all along, and MJCF-lowered robots already rely on it — but it should be
motivated as *removing an asymmetry between the two lowering paths*, not as a
fix for envelope slop.

**8.4 ~~Fix issue #155 regardless.~~ Done.** The sweep now asks the kernel's own
predicate for the true primitive and proves "always-colliding" instead of
sampling it. What it surfaced is a *stronger* case for this document's
recommendation, not a weaker one: `panda_link5`↔`panda_link7` turns out to be a
real self-collision that the box envelopes cannot distinguish from an artifact at
any margin, so it stays exempt under protest. Tighter geometry is the only thing
that retires it.

**Retired 2026-09-02 by [issue #191](https://github.com/OpenRAL/openral/issues/191),
and this section called it correctly.** Tighter geometry was indeed the only
thing that could — but not the tighter *box* §8.2 proposes: the corner-reach
reduction available there is 7.5 mm on link5 and 1.9 mm on link7 (§4.3's table)
against the ~28 mm the separation needs. What retired it is the exact hull, via
the [staged narrow phase](collision-hull-narrow-phase.md) extended from world
voxels to self-pairs. The convexity column in §4.3 is why it works: link5 and
link7 are convex to 1e-4 in volume ratio, so `conv(mesh)` **is** the mesh and the
hull's verdict is the mesh's. Over the pair's full `(joint6, joint7)` grid the
false-positive population goes from 11 680 of 14 641 poses to **zero**.

**8.5 Fix `urdf_lowering.lower_link_geometry`'s mesh path regardless.** It still
emits a PCA capsule for a mesh collision, which the table measures at 96–112 mm
of protrusion against the shipped boxes' 27–76 mm. Re-lowering `panda_mobile`
from its URDF today would inflate the envelope slop by up to 2.1× and silently undo
#103. Whether or not §8.2 ever lands, a mesh should lower to a box.

> **This is not hypothetical, and #191 hit it.** `render_cumotion_config` was
> sourcing its collision spheres from `LoweredCollisionModel.collision_geometry`
> — i.e. from exactly this PCA-capsule path — rather than from the manifest the
> kernel checks, so "plan-time and kernel-time share one source of truth" was
> untrue for every hand-authored box manifest. It now prefers the manifest. The
> lowering path itself is still unfixed.

---

## 9. Reproducing the numbers

The measurement scripts are not checked in (they are one-shot analysis, and
§1.11 keeps fixtures for tests rather than studies). They are reconstructible in
full from this section.

**Inputs.** robosuite `1.5.2`'s `models/assets/robots/panda/robot.xml`, loaded
with `mujoco 3.8.0`; the collision geoms are the `link<N>_collision` mesh geoms
(`contype`/`conaffinity` non-zero — exactly one per link body, `link1`…`link7`
being the seven the manifest declares). Manifest
values from `robots/panda_mobile/robot.yaml` at `e1b9915`. Vertices are pulled
from `model.mesh_vert` and placed by each geom's `geom_pos`/`geom_quat`, matching
`sim_sensor_bridge._body_collision_points` exactly.

**Signed distances.** Box: `‖max(|p| − a, 0)‖ + min(max_k(|p_k| − a_k), 0)`.
Sphere-swept box: the same, minus `r`. Capsule: `‖p − clamp_z(p)‖ − r`.

**Protrusion.** The primitive's boundary is sampled (12 000 points; flat faces
sampled directly, the rounded shell as box-surface points pushed along random
outward directions and filtered to `|dist(·, Box(a)) − r| < 1e−9`), then
`trimesh.proximity.closest_point` against the collision mesh; the maximum is
reported.

**SSB fitting.** Faces-fixed: `a = h − r`, `r` swept at 1 mm (0.1 mm for §4.3's
headroom check) and the largest containing `r` taken. Free fit: SLSQP minimising
`Σ a_k` subject to `max_p dist(p, Box(a)) ≤ r − headroom`, followed by a
monotone inflation loop until containment holds exactly — an optimiser's
convergence tolerance is not a safety argument.

**Multi-primitive.** Triangles (not vertices) are assigned to one part, by equal-
mass slabs along the link's longest shipped half-extent, and by `kmeans2` on
triangle centroids. Each part's three vertices are checked against that part
alone, so convexity proves the triangle and hence the mesh.

Environment: workspace `.venv` (numpy 2.2.6, scipy 1.17.1, trimesh 4.12.2,
mujoco 3.8.0, robosuite 1.5.2). CPU only — no GPU work is involved.
