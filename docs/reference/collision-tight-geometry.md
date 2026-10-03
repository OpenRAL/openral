# Tight link geometry in the safety kernel — can it stop being the binding constraint?

> **Status: design investigation, since implemented.** PR #166 shipped §12.2's
> staged design for `check_voxel_collision`, scoped to `link1` + `link2`, and
> [issue #191](https://github.com/OpenRAL/openral/issues/191) extended it to
> `check_self_collision`'s box↔box pass, adding `link5` + `link7` — for a reason
> this study did not anticipate, because it was scoring the **world** side. The
> change record for both is
> [the staged hull narrow phase](collision-hull-narrow-phase.md); its §9 is the
> self-collision half. This document is the evidence and the design argument,
> not a statement of what is in the tree.
>
> Companion documents: the [collision-primitive study](collision-primitive-study.md)
> (PR #157, the geometry argument), [its picture book and
> correction](collision-primitive-images.md) (PR #158), the [RoboCasa start-state
> census](robocasa-start-state-census.md) (PR #159, which limb actually stops the
> robot), and the [validation-evidence
> ledger](collision-validation-evidence.md) — whose 2026-08-23 entry (PR #160)
> independently corroborates this study's world-side conclusion (§7.7).

Measured on `fcf7f01`, against kernel sources, `packages/openral_safety/` and
`robots/panda_mobile/robot.yaml` byte-identical to the #157/#158/#159 baseline.

---

## 1. The question, and the answer in three sentences

Can collision checking stay fast enough while representing the robot's real
geometry, so the single-OBB-per-link envelopes (28.3–76.7 mm of corner slop) stop
being the binding constraint in cluttered scenes?

**Yes, and it is cheaper than what ships today — but it does not solve the
problem it was opened to solve.** A 26-DOP is 4.8× faster than the current
15-axis SAT on an OBB and cuts worst-case slop from 75.9 mm to 25.7 mm; a
warm-started GJK on the exact convex hull costs roughly today's routine and cuts
the slop to **zero**. But against PR #159's 72 stopping states, *perfect* link
geometry recovers **27 of 72 in the worst case and 42 of 72 nominally**. The rest
are held by the world side: the 25 mm voxel grid, half-diagonal 21.65 mm, whose
*lattice phase alone* moves the kernel's reported distance by a measured 20.3 mm.
**Below about 10 mm of link-geometry excess, further tightening buys one extra
state per 2 mm** (§7).

**Cost points the same way.** Over 99 % of the cells the kernel visits are empty,
the empty-cell scan is 65 % of total cost, and compressing occupancy buys nothing
while *skipping* cells buys everything (§9.2). **Both the recovery ceiling and
the cost bottleneck are on the world side, not the link side.** PR #160 reached
the same place by a third method: two of the four characterised stops sit on
cells that contain nothing at all, a map-fidelity term no link geometry touches
(§7.7).

---

## 2. Where the conservatism actually lives

At a stop the kernel reports `kernel_min` while the true link-mesh-to-fixture gap
is `mesh_gap`; PR #159 publishes both for every state. The difference decomposes
exactly:

```
mesh_gap − kernel_min  =  E(u*)  +  W(u*)  +  S
```

| term | what it is | owner |
|---|---|---|
| `E(u*)` | the link primitive's support excess over the true mesh, **along the approach direction** `u*` | link geometry — what this study can change |
| `W(u*)` | world-side voxel inflation: the nearest occupied cell reaches past the surface point it was built from by `12.5·(\|u_x\|+\|u_y\|+\|u_z\|)` mm → **12.50 mm face-on, 21.65 mm corner-on** | the 25 mm grid — out of scope |
| `S` | the SAT lower-bound deficit of `box_box_distance` versus the exact distance | the algorithm, not the shape |

**`E(u*)` is a directional quantity, not the corner slop.** The corner slop is
`max_u E(u)`; crediting a state with it assumes the world cell sits at the box's
worst corner, which it never does (§8.2).

**The decomposition is falsifiable, and was tested.** For all 72 states the
implied `E(u*) = mesh_gap − kernel_min − W − S` must lie in `[0, max_u E(u)]`:

| `W` (mm) | `S` (mm) | implied `E(u*)` range (mm) | admissible? |
|---:|---:|---|---|
| 0.00 | 0.0 | 19.2 … 65.7 | **no** — 23 states exceed their link's maximum |
| 12.50 | 0.0 | 6.7 … 53.2 | no — 6 states exceed |
| 12.50 | 6.0 | 0.7 … 47.2 | **yes** |
| 21.65 | 0.0 | −2.4 … 44.1 | 2 states marginally negative (−2.3, −2.4 mm) |
| 21.65 | 6.0 | −5.4 … 41.1 | no — 7 states negative |

The admissible band is `W` between 12.5 and 21.65 mm with `S` small — the
face-on/corner-on range the geometry predicts. **The world-side term is forced by
the data**; a `W` of zero is arithmetically impossible. Recovering `u*` directly
at the two representative states over 64 lattice phases gives `E(u*)` =
**38.6 mm on link2** (fridge layout 30) and **36.7 mm on link1** (utensil layout
2), against maxima of 46.8 and 53.2 mm.

---

## 3. Candidate fit — how tight is each representation?

Ground truth is robosuite 1.5.2's `link<N>_collision` meshes, placed in the link
frame by the geom transform **only** (MuJoCo folds mesh recentring into the geom
frame; #158 hit this), verified against the raw STL assets to **8 × 10⁻⁹ m**. The
metric is **max support excess** `max_u [h_C(u) − h_mesh(u)]` — the kernel's
worst-case distance under-report against a separating cell. For the shipped OBB
it reproduces #158's protrusion figures to 0.2 mm on all seven links.

| representation | L1 | L2 | L3 | L4 | L5 | L6 | L7 | worst | primitives/link |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **OBB (shipped)** | 53.2 | 46.8 | 75.4 | 75.9 | 45.2 | 52.8 | 28.3 | **75.9** | 8 vtx |
| 14-DOP | 34.6 | 32.7 | 36.8 | 39.6 | 27.5 | 27.0 | 21.5 | **39.6** | 24 vtx |
| 18-DOP | 32.6 | 30.3 | 36.6 | 37.3 | 19.9 | 27.6 | 17.4 | **37.3** | 32 vtx |
| **26-DOP** | 25.7 | 23.1 | 23.8 | 23.1 | 19.0 | 21.6 | 13.1 | **25.7** | 48 vtx |
| 32 spheres | 35.6 | 37.6 | 36.0 | 32.6 | 41.4 | 34.9 | 23.7 | **41.4** | 32 |
| 64 spheres | 32.5 | 30.7 | 30.3 | 31.2 | 35.4 | 25.9 | 22.4 | **35.4** | 64 |
| 128 spheres | 23.7 | 26.5 | 27.5 | 25.6 | 27.6 | 23.5 | 18.0 | **27.6** | 128 |
| link voxels @ 25 mm | 39.8 | 40.0 | 39.6 | 40.8 | 37.8 | 40.3 | 41.5 | **41.5** | 267–366 cells |
| link voxels @ 12.5 mm | 19.2 | 20.8 | 20.3 | 20.6 | 19.6 | 20.7 | 19.8 | **20.8** | 382–2328 cells |
| 32-vtx hull + offset | 12.4 | 14.9 | 14.8 | 15.1 | 13.6 | 9.7 | 8.0 | **15.1** | 31–32 vtx |
| 64-vtx hull + offset | 9.2 | 10.5 | 7.7 | 5.1 | 6.6 | 6.2 | 4.1 | **10.5** | 51–63 vtx |
| link SDF, 32³ (proven bound) | — | — | — | — | — | — | — | **9.18** | 0.12 MiB |
| link SDF, 64³ (proven bound) | — | — | — | — | — | — | — | **4.53** | 1.00 MiB |
| link SDF, 128³ (proven bound) | — | — | — | — | — | — | — | **2.25** | 8.00 MiB |
| **exact convex hull** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 102–1588 vtx |

Multi-primitive splits are absent because they do nothing (§8.1).

### 3.0 The SDF rows are a bound, not a fit

A trilinear-interpolated signed-distance field is an approximation, not a
containing solid. For a 1-Lipschitz field on pitch `p` the interpolant's
deviation is bounded by `p·√3/2`; the routine must subtract that bound to keep
the never-under-report property, so the SDF's *effective* excess **is** the bound
— 9.18 mm at 32³, 4.53 mm at 64³, 2.25 mm at 128³. Measured deviation is ~3.5×
smaller: the **maximum over-report** (the unsafe direction) is 0.67–2.33 mm at
32³ and 0.90–1.27 mm at 64³. Subtracting only that would give ~1.3 mm and 25/72
recovery, but on sampled evidence, which §4's proof obligation refuses. Memory
for seven links: 0.9 MiB at 32³, **7.0 MiB at 64³**, 56 MiB at 128³.

### 3.1 Reading it

**The exact hull is exactly exact.** `h_hull(u) = h_mesh(u)` by definition,
including on the non-convex `link1`: a concavity costs something only when a
world cell sits *inside* it. Six of the seven collision meshes already *are*
their own convex hulls (#157).

**Spherization is the worst option measured, and it saturates**, because the
Panda's links are flanged, flat-faced blocks and a union of balls cannot
represent a flat face without an unbounded number of them. It must also be fitted
to the **volume** (§11); fitted to *surface* triangle clusters it is
anti-monotone, worsening with more spheres (link2: 95.6 mm at N=8, 111.7 mm at
N=64).

**Link-frame voxelisation at the world grid's pitch is not "exact at grid
resolution"** — at 25 mm it is barely better than the shipped OBB and *worse* on
link7, because every cell the surface enters is kept, inflating the solid by up
to a full cell in every direction, and that inflation composes with the world
grid's rather than cancelling it.

---

## 4. Containment — the proof obligation, per candidate

A tighter envelope is a **smaller** envelope, so every candidate owes a
containment proof, not a sample. Measured minimum inward margin of the mesh
inside each candidate:

| candidate | proof | measured margin |
|---|---|---|
| OBB (shipped) | convex body ⊇ all mesh vertices ⟹ ⊇ `conv(vertices)` ⊇ mesh | 0.055–0.132 mm |
| convex hull | `conv(vertices)` **is** the hull; containment is definitional | 0.000 mm (exact) |
| k-DOP (14/18/26) | intersection of tangent halfspaces `u·x ≤ h_mesh(u)`; every mesh point satisfies every constraint by construction | 0.000 mm (exact, tangent) |
| tangent polytope (any direction set) | same construction as the k-DOP | 0.000 mm (exact) |
| hull + outward offset | `hull_k ⊕ B(δ)` with `δ = max_v dist(v, hull_k)` over all mesh vertices | 0.000 mm (δ chosen to close it) |
| N spheres | **per-triangle**: each triangle assigned to one sphere, that sphere contains its 3 vertices; a ball is convex, so the 3 vertices prove the triangle, hence the mesh | 0.000 mm |
| link voxelisation | every cell the surface enters is kept, plus interior fill | exact at pitch |

For the k-DOP and tangent polytope containment is true **by construction**, for
any direction set, with no optimiser tolerance in the argument — materially
easier to put in front of a safety-WG reviewer than a fitted radius. **The
shipped OBBs carry no headroom to inherit**: their 0.055–0.132 mm margins are
accidental, so any refit must *declare* an explicit headroom (1 mm is a
defensible default) and report the achieved margin against it (#157 §7.3).

---

## 5. The property that decides the safety case: is the candidate inside the shipped OBB?

`check_voxel_collision` sizes its broad-phase window from `half_extents` alone
(`collision.cpp:691-703`) — the one place a representation change can make the
kernel **unsafe** rather than merely tighter (#157): an under-sized window never
visits cells that genuinely intersect the solid, and the check silently returns
clear, a missed collision rather than a lost conservatism.
The hazard is **three windows** wide: `collision.cpp:656-662` (capsule pass),
`:697-703` (box pass), and `:803-837` (`primitive_cell_box`, attached payloads);
`check_self_collision` has no broad phase at all (nor had `check_world_collision`, retired
2026-09-23 with the capsule world phase, ADR-0109).
So one measurement is decisive: **is each candidate a subset of the shipped
OBB?**

| candidate | max protrusion beyond the shipped OBB | verdict |
|---|---:|---|
| exact convex hull | −0.05 … −0.13 mm | **subset** |
| 14-DOP / 18-DOP / 26-DOP | −0.05 … −0.13 mm | **subset** |
| 64-vtx hull + offset | +4.0 … +10.5 mm | escapes |
| 32-vtx hull + offset | +7.9 … +15.1 mm | escapes |
| 128 spheres | +17.0 … +25.8 mm | escapes |
| link voxels @ 12.5 mm | +12.5 … +19.9 mm | escapes |
| link voxels @ 25 mm | +24.6 … +35.5 mm | escapes |
| link SDF (any resolution) | escapes — its *domain* is a padded box, and the conservative bound inflates the zero level set outward | escapes |

The hull and every k-DOP sit inside the shipped box by exactly its containment
margin. **They are strict improvements: never more conservative anywhere than
today, less conservative only where the box was proud**, and the broad phase
needs no change at all. **Everything else grows the envelope somewhere.** The
offset that restores containment for a decimated hull costs **up to 10.5 mm
(64-vtx) or 15.1 mm (32-vtx) of new face slop**, on faces the shipped OBB holds
to 0.13 mm — and per §2 the face-on direction is where `W` is smallest and link
geometry matters most. Those are different envelopes, not smaller ones, and they
force the reach formula open; the same objection retires #157's swept box.

### 5.1 The broad-phase reach formula, stated explicitly

For a link transform `(R, t)`, per world axis `k`:

| representation | correct `ex/ey/ez` |
|---|---|
| **hull or k-DOP kept inside the shipped OBB** | **unchanged** — `Σ_j \|R_kj\|·he_j` about `t`, with `hull ⊆ Box(he)` asserted at configure time |
| convex hull, general | `max_i (R_k · v_i)` and `min_i (R_k · v_i)` over hull vertices |
| k-DOP, general | if the axis set contains the three link axes, those slabs give the link-frame AABB and the existing formula applies about its centre; otherwise enumerate the polytope's vertices at configure time. **Never** take `max_j M_j` over diagonal slabs — a diagonal slab bound does not bound a coordinate axis |
| union of N spheres | `max_i (R_k·c_i + r_i)` / `min_i (R_k·c_i − r_i)`; better, emit N independent windows, exactly as the capsule pass already does |
| link-frame SDF | `Σ_j \|R_kj\|·h_sdf` about `R·c_sdf + t`, **plus** two obligations with no analogue elsewhere: queries outside the SDF domain must return `+∞` and the solid must be proven inside the domain at configure time; and a truncated SDF needs `τ ≥ margin + half_side·√3` or the kernel cannot distinguish "τ away" from "far" |

The recommendation is the first row: keep the audited OBB as the broad-phase
bound, so `collision.cpp:691-703` never changes. Note that `clamp_index` cannot
restore a missed cell but can **mask** one in testing — the shipped regression
`VoxelCollisionBox.BoxLinkIsCheckedAgainstOccupiedVoxel`
(`test_collision.cpp:959-980`) puts the occupied cell at the box's centre and
**passes with the reach term deleted entirely**.

---

## 6. Measured cost

**Intel Core i5-8600K, 6C/6T, 4.2 GHz sustained, x86_64**, Linux 6.8.0-137,
`g++ 13.3.0`, `-std=c++17 -O2 -g -DNDEBUG` — the kernel's real flags, a generic
baseline x86-64 build (SSE2, no AVX2, no FMA). `taskset`-pinned, 151 timed
repetitions per candidate. **Allocation-freedom was verified, not assumed**
(counting `operator new` wrappers either side of every timed region): **0
allocations** for every candidate. Each query is one link primitive against one
occupied 25 mm cell, in the kernel's own streaming pattern.

| candidate | ns/query | vs baseline | returns |
|---|---:|---:|---|
| **`box_box_distance` (ships today)** | **304.0** | 1.00× | lower bound |
| 15-axis SAT, cube-specialised | 111.2 | **2.73× faster** | lower bound (bit-identical) |
| 14-DOP SAT (7 axes) | 34.7 | **8.8× faster** | lower bound |
| 18-DOP SAT (9 axes) | 44.1 | **6.9× faster** | lower bound |
| **26-DOP SAT (13 axes)** | **62.8** | **4.8× faster** | lower bound |
| sphere↔cube, N=1 | 3.2 | 95× faster | exact |
| sphere↔cube, N=32 | 97.7 | 3.1× faster | exact |
| sphere↔cube, N=64 | 192.7 | 1.6× faster | exact |
| SDF trilinear lookup, 32³ (128 KiB) | 12.9 | 23× faster | exact (interpolated) |
| SDF trilinear lookup, 64³ (1 MiB) | 14.2 | **21× faster** | exact (interpolated) |
| GJK on hull, cold, N=8 | 193.8 | 1.6× faster | exact |
| GJK on hull, cold, N=102 | 841.0 | 2.8× slower | exact |
| GJK on hull, cold, N=152 | 1195.2 | 3.9× slower | exact |
| GJK on hull, cold, N=1588 | 11004.5 | 36× slower | exact |

### 6.1 Warm-starting across the predictive horizon

`lifecycle_kernel.cpp:916` integrates the predicted configuration one step at a
time, so consecutive queries are coherent. Simulating that coherence (σ = 2 mm
translation, 3 mrad rotation per step) and rebuilding the previous witness
simplex:

| hull vertices | cold ns | warm ns | speedup | cold iters | warm iters |
|---:|---:|---:|---:|---:|---:|
| 8 | 214.9 | 71.4 | 3.01× | 4.26 | 1.34 |
| 64 | 587.9 | 169.0 | 3.48× | 5.22 | 1.51 |
| **152** | 1783.4 | **350.8** | **5.08×** | 5.78 | 1.57 |
| **1588** | 18319.2 | **2763.4** | **6.63×** | 7.30 | 1.83 |

**Warm-started GJK on a 152-vertex hull costs 350.8 ns against the 304.0 ns the
kernel already spends** — the exact answer for roughly the price of today's lower
bound; spatial coherence between adjacent cells gives a further 2.1–2.5×.

### 6.2 Two results that came out against expectation

**The SAT deficit is not a lever.** Over 129 847 disjoint poses on the shipped
half-extents: mean 0.910 mm, median 0.000, p95 5.68 mm — but **restricted to the
near-contact band (0–30 mm) that decides a stop it collapses to mean 0.159 mm,
exact 80.2 % of the time, only 5.2 % of poses above 1 mm** (an independent Python
measurement agrees: mean 0.61 mm, median 0.00, p95 3.92 mm). So an exact distance
on the *unchanged* OBB recovers zero states, and §2's table is explained by `W`
varying with the approach direction, with `S ≈ 0`. The tail still belongs in a
hazard entry — the near-contact **maximum** is 22.5 mm, nearly a whole voxel.

**Every k-DOP is faster than the OBB SAT it would replace.** A 26-DOP is 4.8×
cheaper *and* cuts worst-case slop from 75.9 mm to 25.7 mm *and* is a strict
subset of the shipped box; the shipped representation is dominated in both
directions. The cube-specialised SAT is 2.73× for a **bit-identical** result (max
difference 2.9 × 10⁻¹⁶ m over 4096 poses): six of the fifteen axes are unit by
construction, removing six `sqrt` and six divides from a serial dependency chain,
expensive on a no-FMA build. No geometry, distance or safety-posture change.

---

## 7. What it recovers — scored against the census

### 7.1 The scoring rule

For a candidate with measured worst-case excess `E_c`, with an exact narrow
phase:

```
kernel_min′  ≥  mesh_gap − E_c − 21.65 mm        (guaranteed: corner-on world cell)
kernel_min′  ≈  mesh_gap − E_c − 12.50 mm        (nominal: face-on world cell)
```

`E_c`'s **maximum** over directions makes the guaranteed verdict conservative.
`mesh_gap` is PR #159's own published per-state measurement.

### 7.2 Uniform representations

| representation | worst excess | guaranteed | face-on |
|---|---:|---:|---:|
| OBB (shipped) | 75.9 mm | 0/72 | 0/72 |
| 14-DOP | 39.6 mm | 1/72 | 2/72 |
| 18-DOP | 37.3 mm | 1/72 | 6/72 |
| 26-DOP | 25.7 mm | 2/72 | 13/72 |
| 32 spheres | 41.4 mm | 1/72 | 2/72 |
| 128 spheres | 27.6 mm | 3/72 | 7/72 |
| link voxels @ 25 mm | 41.5 mm | 0/72 | 1/72 |
| link voxels @ 12.5 mm | 20.8 mm | 5/72 | 16/72 |
| 32-vtx hull + offset | 15.1 mm | 11/72 | 22/72 |
| link SDF, 32³ | 9.18 mm | 17/72 | 27/72 |
| 64-vtx hull + offset | 10.5 mm | 17/72 | 26/72 |
| **link SDF, 64³** | 4.53 mm | **21/72** | **38/72** |
| link SDF, 128³ | 2.25 mm | 23/72 | 42/72 |
| **exact convex hull** | 0.0 mm | **27/72** | **42/72** |
| *perfect geometry (E = 0)* | 0.0 mm | 27/72 | 42/72 |

### 7.3 The ceiling, and the crossover

The last two rows are the same row. **The exact convex hull is the ceiling**,
because `h_hull(u) = h_mesh(u)`: **perfect link geometry recovers 27 of the 72
census stopping states guaranteed, 42 of 72 face-on.**

| link-geometry excess `E_c` | states recovered (guaranteed) |
|---:|---:|
| 76 mm (today) | 0/72 |
| 40 mm | 0/72 |
| 35 mm | 1/72 |
| 25 mm | 2/72 |
| 20 mm | 5/72 |
| 15 mm | 11/72 |
| **10 mm** | **17/72** |
| 8 mm | 17/72 |
| 6 mm | 19/72 |
| 4 mm | 21/72 |
| 2 mm | 23/72 |
| 0 mm | 27/72 |

**The crossover is at roughly 10 mm.** 76 mm → 10 mm buys 17 states; 10 mm →
perfect buys 10 more, about one state per 2 mm, against a term (`W`) no
link geometry can touch. **45 of the 72 states have `mesh_gap ≤ 21.65 mm` and
cannot be cleared by any link-geometry change at a 25 mm grid**; thirty are under
even the face-on 12.5 mm term. Their mesh gaps are 0.1, 0.1, 0.3, 0.6, 0.7, 1.0,
1.8, 1.8, 1.8, 1.8, 2.0, 2.5, 3.6, 4.0, 5.1, 5.1, 5.1, 5.9, 6.1, 6.5, 6.9, 8.9,
8.9, 9.6, 10.0, 10.4, 10.5, 11.5, 12.2, 12.4, 14.8, 15.4, 16.3, 16.5, 17.4, 17.7,
17.7, 18.2, 18.2, 18.6, 19.2, 19.4, 20.4, 20.4, 20.7 mm; six are the census's own
contact or `contact_unresolved` states. **So the link envelope can be made to
stop being the binding constraint, and the 25 mm voxel grid becomes the binding
constraint instead.**

### 7.4 Per-link configurations — where the recovery actually comes from

PR #159's central finding: `link2` (46 stops) and `link1` (14 stops) dominate 60
of 72, while `link3`/`link4` — the worst slop — dominate zero.

| configuration | guaranteed | face-on |
|---|---:|---:|
| A — shipped OBB (today) | 0/72 | 0/72 |
| B — 26-DOP on every link | 2/72 | 13/72 |
| C — exact hull L2–L7, 26-DOP on L1 | 19/72 | 34/72 |
| D — exact hull L2–L7, tangent-200 on L1 | 24/72 | 38/72 |
| **E — exact hull on every link** | **27/72** | **42/72** |
| F — 26-DOP on L2 only, rest shipped | 0/72 | 6/72 |
| **G — exact hull on L1 and L2 only, rest shipped** | **25/72** | **39/72** |

**Configuration G captures 25 of the 27 achievable states by changing two links**
— L1 10/14 guaranteed and 14/14 face-on, L2 15/46 and 25/46, L5–L7 zero either
way because their stops have millimetre mesh gaps. Configuration B, the cheapest
and fastest option in §6, recovers 2/72: speed is not the constraint, tightness
is, and 25.7 mm is not tight enough.

### 7.5 `link1`'s vertex-count tradeoff

`link1` is the only link whose convex hull is large (1588 vertices; the other six
are 102–152). Containing tangent polytopes at various direction counts:

| `link1` representation | vertices | excess | guaranteed | face-on |
|---|---:|---:|---:|---:|
| 26-DOP | 48 | 25.70 mm | 2/14 | 6/14 |
| tangent-50 | 108 | 16.45 mm | 6/14 | 7/14 |
| tangent-100 | 208 | 11.79 mm | 6/14 | 10/14 |
| tangent-200 | 408 | 8.29 mm | 7/14 | 10/14 |
| tangent-400 | 787 | 5.41 mm | **10/14** | 12/14 |
| exact hull | 1588 | 0.00 mm | **10/14** | **14/14** |

A 787-vertex tangent polytope matches the exact hull's guaranteed recovery at
half the vertices; the hull's extra value is in the face-on column. Warm-starting
gives 6.63× on the 1588-vertex hull — see §9.

### 7.6 Grid lattice phase — a world-side term nobody has costed

Sweeping the 25 mm grid origin over a full cell in each axis (125 phases) at the
shipped fridge pin and re-evaluating the kernel's own arithmetic on `link2`:

```
kernel min:  min −29.2 mm   median −18.5 mm   max −8.9 mm   SPREAD 20.3 mm
occupied cells in the window:  20 … 44
tripping cells:                 2 … 15
```

**Where the lattice happens to fall moves the reported distance by 20.3 mm** —
comparable to the whole voxel half-diagonal, and larger than the gain from any
candidate in §3 except the exact hull. PR #159's −23.47 mm for this state sits
inside this range, which resolves an apparent disagreement with an independent
reconstruction (−11.6 mm at grid origin zero) as lattice phase rather than
geometry; #159's numbers are authoritative throughout §7. The term is real,
unbudgeted, purely world-side, and not reduced by better link geometry.

### 7.7 A second, independent world-side term: map fidelity (PR #160)

Lattice phase is quantisation — the grid is *right*, merely coarse and
arbitrarily aligned. PR #160 reconstructed the four stops on the live model:

| stop | kernel | evidence-voxel verdict |
|---|---|---|
| utensil | `panda_link1`, −17.3 mm | **`unbacked`** — 0 of 243 rays; nearest solid 43.3 mm away |
| baguette | `panda_link5`, −20.9 mm | **`unbacked`** — 0 of 243 rays; nearest solid 106.5 mm away |
| fridge | `panda_link7`, −24.7 mm | `self_occupancy_suspect`, backed by **the stopping link itself** — the kernel being right |
| sink_cup | attached payload, −13.4 mm | `noncollidable_world` — a **visual-only** counter top |

It also closed the self-occupancy hypothesis #157 §6.3 pointed at: **13 288 of
65 536 depth rays hit the robot's own bodies unfiltered and 0 survive the
self-filter**. **Two of the four stops sit on cells that contain nothing at
all**, the baguette's with solid cabinet geometry **one cell away** — a
**map-fidelity** term distinct from both the 21.65 mm quantisation term `W` and
the 20.3 mm lattice-phase swing, which **no link-geometry change touches at
all**. Two independent methods therefore converge: this study scores 72 states
against an idealised grid, #160 reconstructs four live stops ray by ray. It also
bounds what §7's scoring can mean, since the live map departs from the ideal grid
in **both** directions — cells missing where geometry is (so the live kernel may
stop *less* than scoring predicts) and cells occupied where geometry is not.

---

## 8. Corrections to prior work — including my own

### 8.1 Multi-primitive links: the conclusion holds, now against the right criterion

#157's rejection does not change when the split is chosen to minimise corner
reach **along the stopping direction** rather than by volume:

| split | `link2` excess along `u*` | `link1` excess along `u*` |
|---|---:|---:|
| single OBB (shipped) | 38.6 mm | 36.7 mm |
| 2 boxes, k-means | 38.4 mm | 35.3 mm |
| 2 boxes, **slabs ⟂ `u*`** | 38.4 mm | 36.4 mm |
| 3 boxes, slabs ⟂ `u*` | 38.4 mm | 36.4 mm |
| 4 boxes, k-means | 32.8 mm | 35.3 mm |
| 4 boxes, slabs ⟂ `u*` | 38.4 mm | 36.4 mm |
| exact hull | **0.0 mm** | **0.0 mm** |

A split chosen explicitly for the stopping direction gains **0.2 mm on link2 and
0.3 mm on link1**. The excess along `u*` comes from the box's extent
*perpendicular* to `u*`, and slicing along `u*` leaves every slab a face in the
same plane.

### 8.2 Both prior recovery estimates over-state what is achievable

PR #159's proposal 2 scores recovery as `kernel_min + corner_slop(link) > 0`,
crediting every state with the link's **worst-case** corner slop:

| link | states | #159's rule | exact-geometry truth (guaranteed / face-on) |
|---|---:|---:|---|
| `link1` | 14 | 14 | 10 / 14 |
| `link2` | 46 | **46** | **15 / 25** |
| `link5` | 3 | 3 | 1 / 1 |
| `link6` | 1 | 1 | 0 / 0 |
| `link7` | 8 | 8 | 1 / 2 |

The `link2` row is the consequential one: #159 concludes **46/46**, the truth is
**15–25 of 46**, because the realised `E(u*)` is 38.6 mm rather than 46.8 mm and
`W` takes 12.5–21.65 mm no primitive change reaches (§2); its *ranking*
conclusion is unaffected (§7.4). PR #157 §6 errs in the safe direction, too
pessimistic for the exact hull, so both rules — and this study's own SAT-deficit
hypothesis (§6.2) — are directional approximations of §2.

### 8.3 Two census figures worth reconciling (neither changes its conclusion)

PR #159's per-link corner slop uses `collision_model_mesh_slop`, the distance
from each box corner to the nearest sampled mesh **vertex**, where #158 and this
study measure distance to the mesh **surface** — never larger, so #159's slop
over-estimates by **+10.4 mm on `link3` and +11.5 mm on `link4`** (86.44 vs 76.0;
88.22 vs 76.7). Re-running the classification with the corrected surface-based
slop leaves **`UNEXPLAINED` at 0/72**, the largest mesh gap staying well inside
its tightened budget (`link1` 59.7 vs 74.9; `link2` 42.8 vs 68.5).

---

## 9. The staged pipeline, and whether cost is a constraint at all

The real workload at the shipped fridge pin, reconstructing the kernel's own
window arithmetic:

| link | window cells | occupied cells | tripping |
|---|---:|---:|---:|
| link1 | 1 260 | 0 | 0 |
| link2 | 2 496 | 20 | 2 |
| link3 | 1 980 | 0 | 0 |
| link4 | 2 352 | 0 | 0 |
| link5 | 2 178 | 0 | 0 |
| link6 | 924 | 0 | 0 |
| link7 | 378 | 0 | 0 |
| **total** | **11 568** | **20** | **2** |

(Over the 125 lattice phases of §7.6 the occupied count ranges 20–44.)
**More than 99 % of the cells the kernel's triple loop visits are empty** and take
the `if (grid.occupancy[idx] == 0) continue;` fast path. That scan measures
**0.87–0.97 ns/cell**, so the 11 568-cell scan costs **11.17 µs** per check. End
to end over seven links, against a 16-step horizon and a 10 ms budget at 100 Hz:

| narrow phase | at 20 occupied (0.17 %) | | at 400 occupied (3.46 %) | |
|---|---:|---:|---:|---:|
| | total µs | % budget | total µs | % budget |
| **`box_box_distance` (ships today)** | 17.25 | 2.8 % | 137.15 | **21.9 %** |
| 15-axis SAT, cube-specialised | 13.39 | 2.1 % | 60.02 | 9.6 % |
| **26-DOP SAT** | 12.42 | 2.0 % | 40.67 | **6.5 %** |
| sphere↔cube, N=32 | 13.12 | 2.1 % | 54.62 | 8.7 % |
| **SDF lookup, 64³** | 11.45 | **1.8 %** | 21.24 | **3.4 %** |
| GJK warm, N=152 | 20.87 | 3.3 % | 209.56 | 33.5 % |
| GJK cold, N=152 | 35.07 | 5.6 % | 493.62 | **79.0 %** |
| mixed (`link1` N=1588 + rest N=152, warm) | — | — | 473.86 | 75.8 % |

**At start-state occupancy the scan is 65 % of the cost and every candidate
fits.** At cluttered occupancy the ranking inverts and becomes decisive.

| candidate | crossover (narrow = scan) | breaks the 10 ms budget at |
|---|---:|---:|
| GJK cold, N=152 | 9 cells | 510 cells |
| GJK warm, N=152 | 23 cells | 1 245 cells |
| **`box_box_distance` (ships today)** | **37 cells** | 1 985 cells |
| sphere↔cube, N=64 | 60 cells | 3 130 cells |
| 15-axis SAT, cube-specialised | 108 cells | 5 420 cells |
| 26-DOP SAT | 210 cells | 9 590 cells |
| SDF lookup, 64³ | ~1 600 cells *(extrapolated)* | **never** (100 % occupancy = 3.00 ms) |
| GJK warm, N=1588 (`link1` alone) | — | **120 cells** |

**The shipped baseline is already past its own crossover**: `box_box_distance`
equals the scan at 37 cells, while the lattice-phase range at a single stopping
state is already 20–44. `link1`'s 1588-vertex hull is the one disqualifying
number — warm-started it breaks budget at **120 occupied cells, about 1 % of a
link's window**. Cost is a function of **how occupancy distributes across
links**, not of hull sizes alone: at the measured state the mixed configuration
costs the same as pure N=152 because `link1`'s window held no cells, and a
utensil-scene state, where `link1` *is* the stopping link, is the adverse case.

### 9.1 The staged pipeline — and why the cost data rescues it

The scan dominating does not make staging unnecessary: an always-on GJK path
crosses over at 9–23 cells and is at 79 % of budget in moderate clutter.
**Staging is what makes exact geometry affordable**, from the cheapest
conservative stage available:

* **Stage 1 — 26-DOP SAT (62.8 ns).** A lower bound on the DOP distance, and the
  DOP contains the hull, so `DOP_SAT > margin` ⟹ the cell is certainly clear. It
  is a strict subset of the shipped OBB (§5), so the broad phase is untouched.
* **Stage 2 — GJK on the exact hull**, run only on cells stage 1 fails to clear.

At the measured state 20 cells are occupied and **2 trip** (2–15 over lattice
phases), so the staged narrow phase costs `20 × 62.8 ns + 2 × 351 ns ≈ 2.0 µs`
against today's 6.08 µs; at 400 occupied with 40 tripping,
`400 × 62.8 + 40 × 351 ≈ 39 µs` against 121.6 µs. **Staged 26-DOP + exact hull is
roughly 3× cheaper than what ships today and returns the exact distance.** For
`sweep_min_distance`, stage 1's lower bound is folded for cells that clear (an
under-report of clearance, the safe direction, and `sweep_min` is explicitly a
diagnostic) and stage 2's exact distance for cells that do not — consistent with
how `fold_pair` already consumes both.

### 9.2 A finding the brief did not anticipate: the scan is the real target

Memory is **not** the bottleneck: an L1-resident 32 KiB grid scans at
0.903 ns/cell against 0.966 ns/cell for a 4 MB room-sized grid — 7 % for 128× the
size. **Compressing occupancy buys nothing; skipping cells does.** At 20 occupied
cells in an 11 568-cell window **99.83 % of the scan is wasted work**, and it is
65 % of the total cost. A hierarchical occupancy summary — a coarse per-block
"any occupied?" bitmap, an octree, or run-length skipping — would cut the
dominant term directly, and it is a change to the *world* representation with no
envelope-shrinking safety argument at all. **Both the recovery ceiling and the
cost bottleneck are on the world side, not the link side**: the strongest single
result in this document.

### 9.3 Measurement caveats

1. With a trivial loop body GCC contracts the scan to a branchless
   `hits += (b != 0)`, giving a spurious density-independent 0.54 ns/cell. The
   real kernel calls `box_box_distance` and `fold_pair` there, so the branch
   survives; the corrected figure is 0.87–0.97 ns/cell and *is* density-sensitive.
2. Running several densities back-to-back in one process is ~2× optimistic — the
   branch predictor memorises the pattern. Each density is a separate process.

**Treat the scan figures as ±6 %, not the ±0.5 % the interquartile ranges
suggest.** The SDF crossover at ~1 600 cells is extrapolated beyond the 10 %
measured range.

---

## 10. Integration cost, and what a safety-WG review would have to assert

### 10.1 Hazard-log Entry 012 — not engaged, re-verified for this representation

Entry 012 obliges `support_contact_exempts` (`collision.cpp:1206-1241`) and the
bridge's `payload_clearing.support_patch_withholds` (`payload_clearing.cpp:165-195`)
to move together so `withheld ⊆ exempt` holds by construction. Neither reads
robot link geometry at any arity, so **the lockstep is not engaged by a
robot-link representation change**, for a hull, k-DOP, sphere union or SDF alike.
`check_attached_self_collision` (`collision.cpp:1341-1372`) **does** read the
link boxes and has no exemption path, so it is not an Entry-012 engagement, but a
tighter link representation makes it less conservative too and it belongs in the
hazard entry by name. **Attached payloads keep their current primitive path**,
because extending to `AttachedPrimitive` engages the lockstep on both sides at
once. Re-verified against the shipped code in
[the staged hull narrow phase](collision-hull-narrow-phase.md) §6.

### 10.2 Schema

`collision_geometry` is `list[LinkCollisionGeometry]` with **no uniqueness
constraint on `link_name`**, so a multi-primitive link already validates, and
adding a union member is a widening change under §1.6 — no `schema_version` bump
and no migrator. Two findings that would have gated a new shape variant, an
untagged `CollisionShape` union and two fail-open shape branches, are **now
closed** (§12.8).

### 10.3 Wire format and the allocation constraint

`n_boxes` derives from `box_link.size()` independently of `n_links`, so **a
variable number of fixed-arity primitives per link needs no wire or parameter
change at all**; a hull's variable-length vertex list does, and the fix is a CSR
offset layout — new parameter arrays, not a schema change. The binding real-time
constraint is `NoAlloc.ForwardKinematicsAndSelfCollisionAreAllocationFree`
(`test_collision.cpp:619-678`), so any variant must be a bounded, pre-sized array.
An SDF is the exception: a per-link voxel buffer as a ROS `double[]` parameter
would need a new out-of-band configure-time asset surface, which is why §12 does
not recommend it despite its cost.

### 10.4 Lowering, and a live trap

The one-primitive-per-link limit is `envelope_loader._capsules_by_link`
(`envelope_loader.py:498-514`), although `mjcf_lowering` already emits N per link
(`mjcf_lowering.py:337-346`) — exercised by
`tests/sim/safety/test_kernel_h1_self_collision.py:125-128`, which drives the
**MJCF** path and must not be cited as evidence that the manifest path does.
`urdf_lowering` does worse than reject: `lower_link_geometry` **silently
collapses** a link's `<collision>` elements into one PCA capsule, measured by
PR #160 at **1.87–3.67× volume on `panda_mobile`** (and **`so101_follower`, up
to 4.95×**) before it made `openral collision lower --write` refuse outright
(exit 3, no override flag). The lowering is still wrong and still blocks any
re-lowering workflow, but it can no longer loosen a shipped envelope silently.

### 10.5 What the hazard entry and the safety-WG review would have to assert

1. **Containment, per link, as a proof and not a sample** — every mesh vertex
   inside the primitive, with the achieved margin reported for the exact shipped
   values. For the hull and k-DOP this is definitional (§4), which is the point of
   choosing them.
2. **An explicitly declared headroom**, replacing the accidental 0.055–0.132 mm.
3. **The candidate is a subset of the shipped OBB**, asserted at configure time by
   vertex containment (§5) — enforced at load, not assumed. This is what keeps the
   broad phase unchanged.
4. **The broad phase was verified, not merely unchanged** — a regression test with
   the occupied cell at the outer face of an unclamped window, because the
   existing test passes with the reach term deleted (§5).
5. **The narrow phase's conservatism direction is declared.** A k-DOP SAT stays a
   lower bound; GJK returns the **exact** distance, strictly *less* conservative
   than what ships today — mean 0.16 mm, near-contact max 22.5 mm (§6.2). That
   loss is invisible in a "tighter geometry" framing and must be stated separately
   from the shape change.
6. **`check_attached_self_collision` becomes less conservative too**, bounded by
   the same per-link Δ (§10.1).
7. **The recovery claim is the one in §7, not a larger one** — 25/72 guaranteed for
   the recommended configuration, and 45/72 states unreachable by any link-geometry
   change. A hazard entry justifying a smaller envelope on a benefit the evidence
   does not show is the exact failure mode §1.2 exists to prevent.
8. ~~**Issue #155 is fixed first, or the ACM is not regenerated**~~ (#157 §7.3
   item 6) — **discharged**: a regeneration reproduces all 10 shipped ACMs
   byte-identically. What it found raises this document's stakes:
   `panda_link5`↔`panda_link7` is a **real** self-collision (up to 48.3 mm of mesh
   interpenetration across a 39.6° band of `joint6`) that the box envelopes cannot
   separate from an artifact at *any* margin — real collisions reach a box gap of
   −9.97 mm, collision-free poses −40.07 mm. It ships exempt under a residual-risk
   SRDF row; tightening `link5`/`link6`/`link7` is the only thing that retires it.

---

## 11. Reproducing

Measurement code is not checked in (§1.11 keeps fixtures for tests, not studies);
it is reconstructible from this section.

**Inputs.** robosuite 1.5.2 `models/assets/robots/panda/robot.xml` under
mujoco 3.8.0; the seven `link<N>_collision` mesh geoms. Vertices from
`model.mesh_vert` placed by `geom_pos`/`geom_quat` **only** — MuJoCo folds mesh
recentring into the geom frame, and `mesh_pos == geom_pos` on all seven links, so
applying both double-counts. Cross-checked against the raw STL assets to
8 × 10⁻⁹ m. Manifest values from `robots/panda_mobile/robot.yaml`; rotation
convention `Rz·Ry·Rx`, matching `transform_from_xyz_rpy`.

**Support excess.** `h_C(u) − h_mesh(u)` over a **20 000-direction Fibonacci
set**. For polytopes `h_C(u) = max_i u·v_i`; for sphere unions
`max_i (u·c_i + r_i)`; for cell unions
`max_i (u·c_i) + p/2·(|u_x|+|u_y|+|u_z|)` — the union-of-cubes support, not the
ball bound.

**Candidates.** k-DOPs and tangent polytopes by `scipy.spatial.HalfspaceIntersection`
over tangent halfspaces `u·x ≤ h_mesh(u)`, axes in the manifest box's frame;
decimated hulls by support vertices in `k` Fibonacci directions plus the outward
offset closing containment; spheres by volumetric k-means over a 6 mm filled
voxelisation, radius = max distance to an assigned cell + that cell's
half-diagonal; multi-OBB by triangle partition. Exact distances for the
SAT-deficit cross-check as a 6-variable SLSQP QP, not a hand-rolled GJK.

**Census states** built through `robocasa.utils.env_utils.create_env` with
`layout_ids` pinned and seed 1, `MUJOCO_GL=egl`; world occupancy from solid
(`contype`/`conaffinity` non-zero) non-robot geoms subdivided to 5 mm; the
kernel's `box_box_distance` and window arithmetic ported to numpy from
`collision.cpp:327-366` and `:691-703`. **SDF error** from
`trimesh.proximity.signed_distance` grids over a 40 mm padded box, sampled along
**outward face normals** (random directions put samples inside the mesh);
conservative bound analytic, `p·√3/2`. **Benchmarks** as §6, scan medians over 5
separate process invocations × 151 repetitions (§9.3). Environment: workspace
`.venv` — numpy 2.2.6, scipy 1.17.1, trimesh 4.12.2, mujoco 3.8.0, robosuite
1.5.2, robocasa. CPU only.

---

## 12. Recommendation

**12.1 The binding constraint is no longer the link envelope, and the study
should be closed on that finding** (§7.3): perfect link geometry recovers 27 of
72 census stops guaranteed, and the remaining 45 are held by the 25 mm voxel
grid. **Below about 10 mm of link excess, tightening buys one state per 2 mm.**

**12.2 If a link-geometry change is made, make it a staged 26-DOP → exact convex
hull, scoped to `link1` and `link2`.** (Shipped by #166. The scope grew to
`link5` + `link7` in #191 — not because the world-side ranking moved, but
because the **self**-collision check needs those two at hull fidelity to retire
the `panda_link5`↔`panda_link7` ACM exemption, a use this section was not
scoring. The scoping rule survives intact: an addition needs its own
measurement, and #191 brought one.) Stage 1 is a 26-DOP SAT (62.8 ns, a
conservative lower bound); stage 2 is a warm-started GJK on the exact hull, run
only on cells stage 1 cannot clear (§9.1). This combination is the only one that
is simultaneously:

* **exact** — 0 mm support excess, the ceiling, 27/72 guaranteed (§7.3);
* **a strict subset of the shipped OBB at both stages**, so the broad phase and
  the unsafe-window hazard §5 identifies never move at all;
* **containing by definition** rather than by a fitted parameter — the hull is
  `conv(vertices)` and the DOP is an intersection of tangent halfspaces, so
  neither containment argument contains an optimiser tolerance (§4);
* **cheaper than what ships today** — roughly 2.0 µs against 6.08 µs of narrow
  phase at the measured state, and ~39 µs against 121.6 µs in moderate clutter.

Scope it to **`link1` and `link2` only** (configuration G): **25 of the 27
achievable states for two links' worth of change and two containment proofs.**
`link3` and `link4`, with the worst slop in the fleet, stop nothing and must not
be touched.

**`link1`'s 1588-vertex hull needs an explicit decision.** Warm-started it breaks
the 100 Hz budget at 120 occupied cells — about 1 % of a link's window — and the
utensil scene, where `link1` is the stopping link, is the adverse case (§9).
Either the 787-vertex tangent polytope (which matches the exact hull's
*guaranteed* recovery, 10/14, and loses only in the face-on column) or **a hard
cap on stage-2 invocations per step**. Do not ship the raw 1588-vertex hull on an
always-on path.

**12.2b The SDF is the strongest cost candidate and the weakest safety case; it
is the right answer only if clutter, not recovery, becomes the binding
requirement.** At 64³ it is 21× cheaper than today's routine, is the only
candidate **bounded at any occupancy** (100 % occupancy costs 3.0 ms of a 10 ms
budget), and recovers 21/72. Against that: containment rests on a Lipschitz
*bound* (§3.0); it **escapes the shipped OBB**, so the reach formula must be
restated and re-reviewed, plus domain-closure and truncation obligations with no
analogue elsewhere (§5.1); and a per-link voxel buffer is a poor fit for the ROS
parameter path (§10.3), at 7 MiB for the fleet. A materially larger safety and
integration surface for 6 fewer recovered states.

**12.2c The 26-DOP alone is the no-precompute fallback, and it is not enough.**
It needs no refit pipeline beyond a slab fit, is 4.8× faster than today and is
the best non-precomputed candidate under clutter (6.5 % of budget at 400 cells),
but at 25.7 mm of excess it recovers **2/72 guaranteed and 13/72 face-on**. Ship
it as stage 1 of §12.2, not as the answer.

**12.3 Take the cube-specialised SAT regardless — it is free.** 2.73× faster for a
bit-identical result, with no geometry, containment or safety-posture change.

**12.4 Do not pursue spherization, link voxelisation, or decimated-hull-plus-offset.**
Spheres saturate at 18–27 mm even at 128 per link (§3.1) and link voxelisation at
the world pitch is barely better than the shipped OBB; all three **escape the
shipped OBB** by 10–35 mm, growing faces currently tight to 0.13 mm and forcing
the broad-phase reach formula open.

**12.5 Do not pursue multi-primitive splits for this problem.** A split chosen for
the stopping direction gains 0.2–0.3 mm (§8.1). Lifting `_capsules_by_link`
remains worthwhile on its own merits — ~15 lines of Python, touching no schema, no
wire format and no kernel (§10.3) — but not as a fix for envelope slop.

**12.6 Do not extend anything here to attached payloads.** Entry 012 is not
engaged as scoped (§10.1) and extending to `AttachedPrimitive` would engage it on
both sides simultaneously, with no measured motivation. (Superseded for
`check_attached_voxel_collision` by #266, which brought that measurement — see
[the staged hull narrow phase](collision-hull-narrow-phase.md) §10.)

**12.7 The highest-value next piece of work is on the world side, and it is not a
geometry change.** The recovery ceiling is world-side: 45 of 72 states cannot be
cleared by *any* link geometry at a 25 mm grid, and lattice phase alone moves the
reported distance by 20.3 mm (§7.6). The cost bottleneck is world-side: 99.83 %
of the empty-cell scan is wasted work and it is 65 % of the total (§9.2). Neither
shrinks a safety envelope, so neither carries §12.2's containment obligation.

**12.8 Independent of all the above**, three items this study surfaced are **now
closed**: `CollisionShape` carries `Field(discriminator=...)` on the alias
(§10.2); both fail-open shape branches — `envelope_loader.py`'s bare `else` and
`AttachedCollisionPrimitive.fill_idl`'s `else`-less `isinstance` chain — raise
`ROSConfigError` naming the shape, the lowering one having been **unsafe** (an
unknown primitive became a zero-length capsule of its `radius_m`, strictly
smaller than declared) and the encoder one a diagnosability defect only; and
§10.4's re-lowering trap is guarded by PR #160's hard refusal.

---

## See also

* [Collision-primitive study (PR #157)](collision-primitive-study.md) — the geometry argument and the sphere-swept box.
* [What the kernel's link boxes enclose (PR #158)](collision-primitive-images.md) — the corrected protrusion figures this study reproduces.
* [RoboCasa start-state census (PR #159)](robocasa-start-state-census.md) — the 72 stopping states scored in §7.
* [Self-occupancy settled, and the re-lower guard (PR #160)](collision-validation-evidence.md) — the independent live-model corroboration of the world-side conclusion (§7.7), and the guard that closed §10.4's re-lowering trap.
* [Collision-stack validation evidence](collision-validation-evidence.md) — the round-by-round ledger and standing caveats.
* `cpp/openral_safety_kernel/README.md` — the kernel geometry this study measures against.
