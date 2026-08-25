# Tight link geometry in the safety kernel — can it stop being the binding constraint?

> **Status: design investigation with measured prototypes. Nothing has landed.**
> This document touches no code in `cpp/openral_safety_kernel/` or
> `packages/openral_safety/`, both safety-WG gated and requiring a hazard-log
> entry before a change. Prototype measurement code lives outside the repo
> (§11). What follows is the evidence a reviewer would need in order to decide,
> and a recommendation with the safety case each option would require.
>
> Companion documents: the [collision-primitive study](collision-primitive-study.md)
> (PR #157, the geometry argument), [its picture book and
> correction](collision-primitive-images.md) (PR #158), the [RoboCasa start-state
> census](robocasa-start-state-census.md) (PR #159, which limb actually stops the
> robot), and the [validation-evidence
> ledger](collision-validation-evidence.md) — whose 2026-08-23 entry (PR #160)
> independently corroborates this study's world-side conclusion (§7.7).

Measured on `fcf7f01`. The kernel sources, `packages/openral_safety/`, and
`robots/panda_mobile/robot.yaml` are **byte-identical** between `e1b9915` (the
baseline #157/#158/#159 measured on) and this commit, verified with
`git diff e1b9915 master -- cpp/openral_safety_kernel packages/openral_safety robots/panda_mobile`
returning empty, so every geometry number in those documents carries over
unchanged.

---

## 1. The question, and the answer in three sentences

The question put to this study was: the single-OBB-per-link representation is so
coarse (28.3–76.7 mm of corner slop) that the robot cannot move in cluttered
scenes — is there a way to keep collision checking fast enough while
representing the robot's real geometry, so the envelopes stop being the binding
constraint?

**Yes, and it is cheaper than what ships today — but it does not solve the
problem it was opened to solve.** Tighter geometry is available at *lower* cost
than the current 15-axis SAT on an OBB: a 26-DOP is 4.8× faster and cuts the
worst-case slop from 75.9 mm to 25.7 mm; a warm-started GJK on the exact convex
hull is roughly the cost of today's routine and cuts the slop to **zero**.

**But the envelope stops being the binding constraint before the scenes unblock.**
Scored against PR #159's 72 stopping states, *perfect* link geometry — the exact
mesh, zero error — recovers **27 of 72 states in the worst case and 42 of 72
nominally**. The other 30–45 are held by the world side: the 25 mm voxel grid,
whose half-diagonal is 21.65 mm and whose *lattice phase alone* moves the
kernel's reported distance by a measured 20.3 mm. **Below about 10 mm of
link-geometry excess, further tightening buys roughly one extra state per 2 mm.**
That is the crossover this study was asked to find, and it is stated plainly in
§7.

**The cost measurement points the same way from the other end.** More than 99 % of
the cells the kernel visits are empty, the empty-cell scan is 65 % of the total
cost, and compressing occupancy buys nothing while *skipping* cells buys
everything (§9.2). **Both the recovery ceiling and the cost bottleneck are on the
world side, not the link side** — which is the single most useful result here, and
it says the next study is about the occupancy grid.

**And a third method, run independently, lands in the same place.**
[PR #160](https://github.com/OpenRAL/openral/pull/160) reconstructed the four
characterised stops ray by ray on the live model, closed the self-occupancy
hypothesis by measurement (13 288 of 65 536 depth rays hit the robot unfiltered,
**0** survive the self-filter — it is not the base), and found **two of the four
stops sitting on cells that contain nothing at all**, one of them with solid
cabinet geometry a single cell away. That is a *map-fidelity* term on top of
quantisation and lattice phase, and no link-geometry change touches it: a phantom
cell stops an exact convex hull exactly as readily as a coarse box (§7.7).

---

## 2. Where the conservatism actually lives

At a stop the kernel reports `kernel_min` while the true link-mesh-to-fixture
gap is `mesh_gap`. PR #159 publishes both for every state. The difference
decomposes exactly into three terms:

```
mesh_gap − kernel_min  =  E(u*)  +  W(u*)  +  S
```

| term | what it is | owner |
|---|---|---|
| `E(u*)` | the link primitive's support excess over the true mesh, **along the approach direction** `u*` | link geometry — what this study can change |
| `W(u*)` | world-side voxel inflation: the nearest occupied cell reaches past the surface point it was built from by `12.5·(\|u_x\|+\|u_y\|+\|u_z\|)` mm → **12.50 mm face-on, 21.65 mm corner-on** | the 25 mm grid — out of scope |
| `S` | the SAT lower-bound deficit of `box_box_distance` versus the exact distance | the algorithm, not the shape |

Three things follow, and the rest of the document is their consequences.

**`E(u*)` is a directional quantity, not the corner slop.** The corner slop is
`max_u E(u)`. Crediting a state with the full corner slop assumes the world cell
sits exactly at the box's worst corner. It never does. §8.2 shows this is where
both prior recovery estimates went wrong.

**The decomposition is falsifiable, and it was tested.** For every one of the 72
states the implied `E(u*) = mesh_gap − kernel_min − W − S` must lie in
`[0, max_u E(u)]` — the measured per-link maximum. Sweeping `W` and `S`:

| `W` (mm) | `S` (mm) | implied `E(u*)` range (mm) | admissible? |
|---:|---:|---|---|
| 0.00 | 0.0 | 19.2 … 65.7 | **no** — 23 states exceed their link's maximum |
| 12.50 | 0.0 | 6.7 … 53.2 | no — 6 states exceed |
| 12.50 | 6.0 | 0.7 … 47.2 | **yes** |
| 21.65 | 0.0 | −2.4 … 44.1 | 2 states marginally negative (−2.3, −2.4 mm) |
| 21.65 | 6.0 | −5.4 … 41.1 | no — 7 states negative |

The admissible band is `W` between 12.5 and 21.65 mm with `S` small — exactly the
face-on/corner-on range the geometry predicts, with `W` varying per state with
the approach direction. **The world-side term is not a modelling convenience; it
is forced by the data.** A `W` of zero is arithmetically impossible.

**Independent confirmation from the live scene.** Recovering `u*` directly at
the two representative states — the direction from the link box centre to the
nearest tripping cell, over 64 grid lattice phases — gives a measured
`E(u*)` of **38.6 mm on link2** (fridge layout 30) and **36.7 mm on link1**
(utensil layout 2), against maxima of 46.8 and 53.2 mm. The implied values from
the table above peak at 41.6 and 44.1 mm. The two methods agree that the realised
excess is well below the corner slop.

---

## 3. Candidate fit — how tight is each representation?

Ground truth is robosuite 1.5.2's `link<N>_collision` meshes, placed in the link
frame by the geom transform **only** (MuJoCo folds mesh recentring into the geom
frame; #158 hit this). Verified against the raw STL assets to **8 × 10⁻⁹ m**, and
`mesh_pos` confirmed equal to `geom_pos` on all seven links.

The metric is **max support excess** `max_u [h_C(u) − h_mesh(u)]` — the kernel's
worst-case distance under-report against a separating cell. For the shipped OBB
it reproduces #158's protrusion figures to 0.2 mm on all seven links, which is
the cross-check that the harness is measuring the same thing.

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

Multi-primitive splits are absent from the table because they do nothing; §8.1
gives the measurement and the reason.

### 3.0 The SDF rows are a bound, not a fit

A trilinear-interpolated signed-distance field is an **approximation**, not a
containing solid, so it cannot be scored like the others until its error is made
conservative. For a 1-Lipschitz distance field on a grid of pitch `p`, the
interpolant's deviation is bounded by `p·√3/2`; to preserve the kernel's
never-under-report property the routine must subtract that bound, so the SDF's
*effective* support excess **is** the bound — 9.18 mm at 32³, 4.53 mm at 64³,
2.25 mm at 128³.

Measured deviation is much smaller than the bound. Sampling exterior query points
2–60 mm off the surface and comparing the interpolant against the true mesh
distance, the **maximum over-report** — the unsafe direction, where the SDF calls
a cell clearer than it is — is 0.67–2.33 mm at 32³ and 0.90–1.27 mm at 64³, i.e.
roughly 3.5× tighter than the provable bound.

**The gap between those two numbers is the whole safety question for this
candidate.** Subtracting the proven bound gives a real containment argument and
4.53 mm of excess at 64³. Subtracting only the measured maximum would give
~1.3 mm and 25/72 recovery, but it would rest on sampled evidence, which is
exactly what §4's proof obligation exists to refuse. The rows above use the
proven bound.

Memory for all seven links: 0.9 MiB at 32³, **7.0 MiB at 64³**, 56 MiB at 128³.

### 3.1 Reading it

**The exact hull is exactly exact.** `h_hull(u) = h_mesh(u)` holds by definition
of the convex hull, on every link including the non-convex `link1`. The hull
still fills `link1`'s concavity, but a concavity only costs anything when a world
cell sits *inside* it; for a cell approaching from outside, the hull's support
error is zero. Six of the seven collision meshes already *are* their own convex
hulls (#157), so for those the hull is not an approximation at all — it is the
mesh.

**Spherization is the worst option measured, and it saturates.** Even 128 spheres
per link leaves 18.0–27.6 mm — worse than a 26-DOP built from 13 axis pairs. The
reason is structural: the Panda's links are flanged, flat-faced blocks, and a
union of balls cannot represent a flat face without an unbounded number of them.
This is worth recording because cuRobo-style spherization is the reflexive answer
to "many cheap primitives", and on this geometry it is the wrong basis.

The first attempt at this row was also wrong in an instructive way. Fitting
spheres to *surface* triangle clusters produced numbers that got **worse** with
more spheres (link2: 95.6 mm at N=8, 111.7 mm at N=64), because the enclosing
ball of a surface patch bulges outward by roughly the patch radius. The table
above is fitted **volumetrically** — the solid is voxelised at 6 mm and filled,
cells are clustered, and each sphere's radius is the max distance to an assigned
cell plus that cell's half-diagonal. Spherization must be fitted to the volume;
fitted to the surface it is not merely suboptimal, it is anti-monotone.

**Link-frame voxelisation at the world grid's own pitch is not "exact at grid
resolution".** At 25 mm it measures 37.8–41.5 mm of excess — barely better than
the shipped OBB on the two links that matter, and *worse* on link7. A cell is
kept whenever any part of the surface enters it, so the represented solid is
inflated by up to a full cell in every direction, and that inflation composes
with the world grid's own rather than cancelling it. Halving the pitch to
12.5 mm gets to ~20 mm at 4–8× the memory.

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

The k-DOP and tangent-polytope constructions deserve emphasis: because they are
built as intersections of **tangent** halfspaces, containment is not something
achieved by a search and then verified — it is true by construction, for any
direction set, with no optimiser tolerance anywhere in the argument. That is a
materially easier thing to put in front of a safety-WG reviewer than a fitted
radius.

**The shipped OBBs carry no headroom to inherit.** Their margins are
0.055–0.132 mm — tens of microns, accidental rather than designed. Any refit must
*declare* an explicit headroom (1 mm is a defensible default) and report the
achieved margin against it, rather than inheriting another accidental number.
This repeats #157 §7.3 and it still stands.

---

## 5. The property that decides the safety case: is the candidate inside the shipped OBB?

`check_voxel_collision` sizes its broad-phase window from `half_extents` alone
(`collision.cpp:691-703`). PR #157 correctly identified this as the one place a
representation change can make the kernel **unsafe** rather than merely tighter:
if the window under-reports the solid's true reach, cells that genuinely
intersect it are never visited, `fold_pair` is never called for them, and the
check silently returns clear. That is a missed collision, not a lost
conservatism.

An independent audit of the kernel confirms and extends this: the hazard is
**three windows wide**, not one line — `collision.cpp:656-662` (capsule pass),
`collision.cpp:697-703` (box pass), and `collision.cpp:803-837`
(`primitive_cell_box`, the structurally identical window for attached payloads).
It also found that `check_self_collision` and `check_world_collision` have **no
broad phase at all** — they are exhaustive sweeps — so the unsafe-window class is
confined to those three sites.

This makes one measurement decisive. **Is each candidate a subset of the shipped
OBB?** If it is, the existing window is already a conservative superset for it,
and the trap cannot fire at all.

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
today, less conservative only where the box was proud.** The broad phase needs no
change whatsoever, and the single line #157 flagged as the only way to make the
kernel unsafe is never touched.

**Everything else grows the envelope somewhere.** This is the answer to "quantify
the offset cost in mm" for the decimated-hull candidate: the outward offset that
restores containment costs **up to 10.5 mm (64-vtx) or 15.1 mm (32-vtx) of new
face slop**, on faces where the shipped OBB is tight to 0.13 mm. Those candidates
are tighter at the corners and *looser on the faces* — and per §2 the face-on
direction is where `W` is smallest and therefore where link geometry matters
most. They are a different envelope, not a smaller one, and they require the
broad-phase reach formula to be restated and re-reviewed. They are not
recommended, and the same objection retires #157's "free fit" swept box for the
same reason.

### 5.1 The broad-phase reach formula, stated explicitly

Required by the brief. For a link transform `(R, t)`, per world axis `k`:

| representation | correct `ex/ey/ez` |
|---|---|
| **hull or k-DOP kept inside the shipped OBB** | **unchanged** — `Σ_j \|R_kj\|·he_j` about `t`, with `hull ⊆ Box(he)` asserted at configure time |
| convex hull, general | `max_i (R_k · v_i)` and `min_i (R_k · v_i)` over hull vertices |
| k-DOP, general | if the axis set contains the three link axes, those slabs give the link-frame AABB and the existing formula applies about its centre; otherwise enumerate the polytope's vertices at configure time. **Never** take `max_j M_j` over diagonal slabs — a diagonal slab bound does not bound a coordinate axis |
| union of N spheres | `max_i (R_k·c_i + r_i)` / `min_i (R_k·c_i − r_i)`; better, emit N independent windows, exactly as the capsule pass already does |
| link-frame SDF | `Σ_j \|R_kj\|·h_sdf` about `R·c_sdf + t`, **plus** two obligations with no analogue elsewhere: queries outside the SDF domain must return `+∞` and the solid must be proven inside the domain at configure time; and a truncated SDF needs `τ ≥ margin + half_side·√3` or the kernel cannot distinguish "τ away" from "far" |

The recommended option is the first row: keep the audited OBB as the explicit
broad-phase bound and let the tighter representation live only in the narrow
phase. Under that design `collision.cpp:691-703` never changes.

The audit also found that `clamp_index` cannot restore a missed cell — it only
moves indices toward the grid, so an under-sized window stays under-sized — but
that it can **mask** one in testing, and that the shipped regression
`VoxelCollisionBox.BoxLinkIsCheckedAgainstOccupiedVoxel`
(`test_collision.cpp:959-980`) places the occupied cell at the box's centre and
**passes with the reach term deleted entirely**. Any change here needs a new test
with the cell at the outer face of an unclamped window.

---

## 6. Measured cost

Benchmarked on this machine: **Intel Core i5-8600K, 6C/6T, 4.2 GHz sustained,
x86_64**, Linux 6.8.0-137, `g++ 13.3.0`, `-std=c++17 -O2 -g -DNDEBUG` — the
kernel's real flags (`cpp/openral_safety_kernel/CMakeLists.txt` defaults
`CMAKE_BUILD_TYPE` to `RelWithDebInfo` and sets no `-march`, so the shipped
build is generic baseline x86-64: SSE2, no AVX2, no FMA). Pinned with `taskset`,
151 timed repetitions per candidate, all poses pre-generated into reserved
storage.

**Allocation-freedom was verified, not assumed**: global `operator new`/`delete`
were replaced with counting wrappers and the count sampled either side of every
timed region. Every candidate reported **0 allocations**; all 218 process-lifetime
allocations occur in setup, before any timed region. GJK uses fixed-size stack
arrays only.

Each query is one link primitive against one occupied 25 mm cell, in the kernel's
own streaming pattern (link pose fixed, cell centres streamed — which is what
`check_voxel_collision` does).

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
time and calls the full check per step, so consecutive queries are highly
coherent. Simulating that coherence (σ = 2 mm translation, 3 mrad rotation per
step) and rebuilding the previous witness simplex:

| hull vertices | cold ns | warm ns | speedup | cold iters | warm iters |
|---:|---:|---:|---:|---:|---:|
| 8 | 214.9 | 71.4 | 3.01× | 4.26 | 1.34 |
| 64 | 587.9 | 169.0 | 3.48× | 5.22 | 1.51 |
| **152** | 1783.4 | **350.8** | **5.08×** | 5.78 | 1.57 |
| **1588** | 18319.2 | **2763.4** | **6.63×** | 7.30 | 1.83 |

The mechanism is visible in the iteration columns: warm start converges in 1.3–1.8
iterations instead of 4.3–7.3. **Warm-started GJK on a 152-vertex hull costs
350.8 ns against the 304.0 ns the kernel already spends** — the exact answer for
roughly the price of today's lower bound. Spatial coherence between adjacent
cells in the same window gives a smaller but real 2.1–2.5×.

### 6.2 Two results that came out against expectation

**The SAT deficit is not a lever.** §2 left open how large `S` is. Measured
directly over 129 847 disjoint poses on the shipped link half-extents: mean
0.910 mm, median 0.000, p95 5.68 mm — but **restricted to the near-contact band
(0–30 mm) that actually decides a stop, it collapses to mean 0.159 mm, exact
80.2 % of the time, only 5.2 % of poses above 1 mm**. An independent Python
measurement on realistic near-contact poses agrees (mean 0.61 mm, median 0.00,
p95 3.92 mm).

I had hypothesised from §2's consistency table that `S` might be 3–6 mm and that
switching to an exact distance on the *unchanged* OBB would recover states for
free, with no containment obligation at all. **That hypothesis is dead**: at
`S ≈ 0.16 mm` it recovers zero states. The §2 table is fully explained by `W`
varying between 12.5 and 21.65 mm with the approach direction, with `S ≈ 0`. The
tail is still worth naming in a hazard entry — the near-contact **maximum** is
22.5 mm, so there exist stop-relevant poses where the 15-axis bound gives away
nearly a whole voxel — but it is a tail, not a systematic gain.

**Every k-DOP is faster than the OBB SAT it would replace.** A 26-DOP is 4.8×
cheaper *and* cuts worst-case slop from 75.9 mm to 25.7 mm *and* is a strict
subset of the shipped box. There is no cost/tightness trade here in either
direction; the shipped representation is dominated.

The cube-specialised SAT deserves separate note: 2.73× for a **bit-identical**
result (max difference 2.9 × 10⁻¹⁶ m over 4096 poses). Six of the fifteen axes
are unit by construction, which removes six `sqrt` and six divides from a serial
dependency chain — expensive on a no-FMA baseline build. It changes no geometry,
no distance, and no safety posture.

---

## 7. What it recovers — scored against the census

### 7.1 The scoring rule

For a candidate with measured worst-case excess `E_c`, evaluated with an exact
narrow phase:

```
kernel_min′  ≥  mesh_gap − E_c − 21.65 mm        (guaranteed: corner-on world cell)
kernel_min′  ≈  mesh_gap − E_c − 12.50 mm        (nominal: face-on world cell)
```

Using `E_c`'s **maximum** over directions makes the guaranteed verdict
conservative — the realised excess along `u*` is at most that. `mesh_gap` is
PR #159's own published per-state measurement.

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

The last two rows are the same row. **The exact convex hull is the ceiling** —
there is nothing beyond it, because `h_hull(u) = h_mesh(u)`.

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

**The crossover is at roughly 10 mm.** Going from the shipped 76 mm to 10 mm buys
17 states. Going from 10 mm to *perfect* buys 10 more — about one state per 2 mm,
against a term (`W`) that no link-geometry change can touch.

**45 of the 72 states have `mesh_gap ≤ 21.65 mm` and cannot be cleared by any
link-geometry change at a 25 mm grid.** Thirty of them are under even the face-on
12.5 mm term. Their mesh gaps are 0.1, 0.1, 0.3, 0.6, 0.7, 1.0, 1.8, 1.8, 1.8,
1.8, 2.0, 2.5, 3.6, 4.0, 5.1, 5.1, 5.1, 5.9, 6.1, 6.5, 6.9, 8.9, 8.9, 9.6, 10.0,
10.4, 10.5, 11.5, 12.2, 12.4, 14.8, 15.4, 16.3, 16.5, 17.4, 17.7, 17.7, 18.2,
18.2, 18.6, 19.2, 19.4, 20.4, 20.4, 20.7 mm. Six of them are the census's own
contact or `contact_unresolved` states — the robot really is touching the fixture
and the kernel is right to stop.

**So the honest answer to the framing question is: the link envelope can be made
to stop being the binding constraint, and the 25 mm voxel grid becomes the
binding constraint instead.** Beyond ~10 mm of link excess the grid holds
45 of 72 states on its own. If the goal is to unblock these scenes, the grid — its
resolution, and the lattice-phase effect measured in §7.5 — is where the
remaining work is, and it is out of this study's scope by construction.

### 7.4 Per-link configurations — where the recovery actually comes from

PR #159's central finding is that `link2` (46 stops) and `link1` (14 stops)
dominate 60 of 72, while `link3`/`link4` — the two links with the worst slop —
dominate zero. Scoring configurations rather than uniform representations:

| configuration | guaranteed | face-on |
|---|---:|---:|
| A — shipped OBB (today) | 0/72 | 0/72 |
| B — 26-DOP on every link | 2/72 | 13/72 |
| C — exact hull L2–L7, 26-DOP on L1 | 19/72 | 34/72 |
| D — exact hull L2–L7, tangent-200 on L1 | 24/72 | 38/72 |
| **E — exact hull on every link** | **27/72** | **42/72** |
| F — 26-DOP on L2 only, rest shipped | 0/72 | 6/72 |
| **G — exact hull on L1 and L2 only, rest shipped** | **25/72** | **39/72** |

**Configuration G captures 25 of the 27 achievable states by changing two links.**
Per link: L1 10/14 guaranteed and 14/14 face-on; L2 15/46 and 25/46; L5, L6, L7
zero either way, because their stops are the ones with millimetre mesh gaps. This
is the census's "target link1 and link2" recommendation, confirmed
quantitatively: 93 % of the total available recovery lives in two links, and
`link3`/`link4` contribute nothing at any representation because they stop
nothing.

Note that configuration B — a 26-DOP everywhere, the *cheapest and fastest*
option in §6 — recovers only 2/72 guaranteed. Speed is not the constraint;
tightness is, and 25.7 mm is not tight enough.

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
half the vertices; the exact hull's extra value is in the face-on column.
Warm-started GJK is 6.63× on the 1588-vertex hull (2.76 µs vs 18.3 µs), and
§6's cost table shows even the cold case is affordable at census occupancy —
see §9.

### 7.6 Grid lattice phase — a world-side term nobody has costed

Sweeping the 25 mm grid origin over a full cell in each axis (125 phases) at the
shipped fridge pin and re-evaluating the kernel's own arithmetic on `link2`:

```
kernel min:  min −29.2 mm   median −18.5 mm   max −8.9 mm   SPREAD 20.3 mm
occupied cells in the window:  20 … 44
tripping cells:                 2 … 15
```

**Where the lattice happens to fall moves the reported distance by 20.3 mm** —
comparable to the entire voxel half-diagonal, and larger than the total gain from
any candidate in §3 except the exact hull. PR #159 reports −23.47 mm for this
state; that value sits inside this range, which resolves the apparent
disagreement with my own reconstruction (−11.6 mm at grid origin zero) as lattice
phase rather than a dispute about geometry. #159's published per-state numbers are
used as authoritative throughout §7; nothing in the scoring depends on my
reconstruction.

This is worth its own line in any future grid work: it is a real, unbudgeted,
purely world-side term, and it is not reduced by better link geometry.

### 7.7 A second, independent world-side term: map fidelity (PR #160)

Lattice phase is quantisation — the grid is *right*, merely coarse and
arbitrarily aligned. [PR #160](https://github.com/OpenRAL/openral/pull/160)
reconstructed the four characterised stops on the live model and found something
stronger, arriving from a completely different direction:

| stop | kernel | evidence-voxel verdict |
|---|---|---|
| utensil | `panda_link1`, −17.3 mm | **`unbacked`** — 0 of 243 rays; nearest solid 43.3 mm away |
| baguette | `panda_link5`, −20.9 mm | **`unbacked`** — 0 of 243 rays; nearest solid 106.5 mm away |
| fridge | `panda_link7`, −24.7 mm | `self_occupancy_suspect`, backed by **the stopping link itself** — the kernel being right |
| sink_cup | attached payload, −13.4 mm | `noncollidable_world` — a **visual-only** counter top |

It also closed, by measurement, the self-occupancy hypothesis that #157 §6.3
pointed at as where the evidence led: **13 288 of 65 536 depth rays hit the
robot's own bodies unfiltered and 0 survive the self-filter**, and the chassis
does not appear even in the *unfiltered* returns. It is not the base.

**Two of the four stops sit on cells that contain nothing at all**, and the
baguette's cell has solid cabinet geometry **one cell away** — the signature #160
names as a *stale or one-cell-displaced map cell*. That is a **map-fidelity**
term, distinct from both the 21.65 mm quantisation term `W` and the 20.3 mm
lattice-phase swing, and **no link-geometry change touches it at all**: a phantom
cell stops an exact convex hull exactly as readily as it stops a coarse box.

**This is the most defensible claim in the chain, because two independent methods
converge on it.** This study reaches "the world model, not the link model" by
scoring 72 states against an idealised grid; #160 reaches it by reconstructing
four live stops ray by ray. Neither rests on the other's method, and they were
produced without reference to each other.

It also bounds what §7's scoring can mean. #159's census and this study both
build a **geometrically ideal** grid from true surfaces, which #159 explicitly
flags as unmeasured against the live octomap. #160 shows the live map departs
from that ideal in **both** directions: cells missing where geometry is (so the
live kernel may stop *less* than ideal-grid scoring predicts), and cells occupied
where geometry is not (which link geometry can never recover). The §7 recovery
figures are therefore a model of the ideal grid, not a prediction of live
behaviour — and the live map's departures push in the same direction as
everything else here: **away from link geometry.**

---

## 8. Corrections to prior work — including my own

### 8.1 Multi-primitive links: the conclusion holds, now against the right criterion

The brief asked whether #157's rejection of multi-primitive splits changes for
`link1` and `link2` when the split is chosen to minimise corner reach **along the
stopping direction** rather than by volume. It does not. Measured at the live
stopping states, along the recovered `u*`:

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
0.3 mm on link1**. The reason is geometric and worth stating because it
generalises: the excess along `u*` is produced by the box's extent
*perpendicular* to `u*` — the leading face's corner reach — and slicing the box
along `u*` leaves every slab with a face in the same plane. You cannot cut away a
corner by cutting perpendicular to the direction you are approaching from.
#157's conclusion was right; this is the directional criterion it was not tested
against.

### 8.2 Both prior recovery estimates over-state what is achievable

PR #159's proposal 2 scores recovery as `kernel_min + corner_slop(link) > 0`,
which credits every state with the link's **worst-case** corner slop:

| link | states | #159's rule | exact-geometry truth (guaranteed / face-on) |
|---|---:|---:|---|
| `link1` | 14 | 14 | 10 / 14 |
| `link2` | 46 | **46** | **15 / 25** |
| `link5` | 3 | 3 | 1 / 1 |
| `link6` | 1 | 1 | 0 / 0 |
| `link7` | 8 | 8 | 1 / 2 |

The `link2` row is the consequential one: #159 concludes that recovering the full
corner slop clears **46/46** fridge states, and the truth is **15–25 of 46**. The
error is the one named in §2 — the corner slop is `max_u E(u)`, and the realised
`E(u*)` is 38.6 mm rather than 46.8 mm, while `W` takes 12.5–21.65 mm that no
primitive change reaches. PR #159's *ranking* conclusion (target link1 and link2,
not link3/link4) is unaffected and is confirmed in §7.4; only the magnitude of the
promised recovery changes.

PR #157 §6 makes the mirror-image error in the safe direction, scoring recovery
against Δcorner-reach and concluding "none of the three recovers". Applied to the
census that rule is too pessimistic for the exact hull, which recovers 27/72.
Both prior rules are directional approximations of the decomposition in §2.

**And my own hypothesis was wrong.** §6.2 records it: I inferred from the §2
consistency table that the SAT bound was giving away 3–6 mm and that an exact
distance on the unchanged OBB would recover states for free. Direct measurement
puts it at 0.16 mm in the regime that matters. The consistency table is explained
by a direction-dependent `W`, not by `S`.

### 8.3 Two census figures worth reconciling (neither changes its conclusion)

PR #159's per-link corner slop is measured by `collision_model_mesh_slop`, which
takes the distance from each box corner to the nearest sampled mesh **vertex**;
#158 and this study measure the distance to the mesh **surface**. Vertex distance
is never smaller, so #159's slop is an over-estimate — negligible on five links
but **+10.4 mm on `link3` and +11.5 mm on `link4`** (86.44 vs 76.0; 88.22 vs
76.7). That makes #159's admissible-gap budget more forgiving than the true
geometry warrants.

Re-running the census's classification with the corrected surface-based slop:
**`UNEXPLAINED` remains 0/72**. The largest mesh gap seen on any link stays well
inside its tightened budget (`link1` 59.7 vs 74.9; `link2` 42.8 vs 68.5). The
census's headline conclusion — the kernel has no defect, only coarseness — is
robust to the correction.

---

## 9. The staged pipeline, and whether cost is a constraint at all

The brief asked for a staged model — OBB as broad phase, tight representation
only when the OBB trips — priced with census frequencies. Measuring the real
workload at the shipped fridge pin, reconstructing the kernel's own window
arithmetic:

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

(Over the 125 lattice phases of §7.5 the occupied count ranges 20–44.)

**More than 99 % of the cells the kernel's triple loop visits are empty** and take
the `if (grid.occupancy[idx] == 0) continue;` fast path.

The empty-cell scan was measured at **0.87–0.97 ns/cell** (density-dependent —
the growth is occupied-branch misprediction, not work), so the 11 568-cell scan
costs **11.17 µs** per configuration check. End-to-end, over all seven links,
against a 16-step predictive horizon and a 10 ms budget at 100 Hz:

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

**The shipped baseline is already past its own crossover.** The occupancy at
which the narrow phase equals the scan is 37 cells for `box_box_distance` — and
the measured lattice-phase range at a single stopping state is already **20–44
cells**. The kernel is operating at the point where its distance routine starts
to dominate, today, at a *start* state.

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

`link1`'s 1588-vertex hull is the one disqualifying number: warm-started, it
breaks budget at **120 occupied cells — about 1 % of a link's window.** Cost is a
function of **how occupancy distributes across links**, not of hull sizes alone;
at the measured state the mixed configuration costs the same as pure N=152
because `link1`'s window happened to hold no cells at all. A utensil-scene state,
where `link1` *is* the stopping link, is the adverse case.

### 9.1 The staged pipeline — and why the cost data rescues it

My first reading of this table was that staging is unnecessary because the scan
dominates. That is wrong, and the crossover row above is why: an always-on GJK
path crosses over at 9–23 cells and is at 79 % of budget in moderate clutter.
**Staging is what makes exact geometry affordable**, and it should be built from
the cheapest conservative stage available:

* **Stage 1 — 26-DOP SAT (62.8 ns).** A lower bound on the DOP distance, and the
  DOP contains the hull, so `DOP_SAT > margin` ⟹ the cell is certainly clear and
  the exact test can be skipped. It is a strict subset of the shipped OBB (§5), so
  the broad phase is untouched.
* **Stage 2 — GJK on the exact hull**, run only on cells stage 1 fails to clear.

At the measured state 20 cells are occupied and **2 trip** (2–15 over lattice
phases), so the staged narrow phase costs `20 × 62.8 ns + 2 × 351 ns ≈ 2.0 µs`
against today's 6.08 µs. At 400 occupied with 40 tripping it is
`400 × 62.8 + 40 × 351 ≈ 39 µs` against today's 121.6 µs.

**Staged 26-DOP + exact hull is roughly 3× cheaper than what ships today and
returns the exact distance.** For `sweep_min_distance`, stage 1's lower bound is
folded for cells that clear (an under-report of clearance, the safe direction, and
`sweep_min` is explicitly a diagnostic) and stage 2's exact distance for cells
that do not — which is consistent with how `fold_pair` already consumes both.

### 9.2 A finding the brief did not anticipate: the scan is the real target

Memory is **not** the bottleneck: an L1-resident 32 KiB grid scans at
0.903 ns/cell against 0.966 ns/cell for a 4 MB room-sized grid — a 7 % penalty
for 128× the size. **Compressing occupancy buys nothing; skipping cells does.**

At 20 occupied cells in an 11 568-cell window, **99.83 % of the scan is wasted
work**, and it is 65 % of the total cost. A hierarchical occupancy summary — a
coarse per-block "any occupied?" bitmap, an octree, or run-length skipping — would
cut the dominant term directly, and it is a change to the *world* representation
with no envelope-shrinking safety argument at all.

This converges with §7's conclusion from the opposite direction. **Both the
recovery ceiling and the cost bottleneck are on the world side, not the link
side.** That is the strongest single result in this document.

### 9.3 Measurement caveats

Two corrections the benchmark surfaced and which must not be silently absorbed:

1. An initial scan figure of 0.54 ns/cell, apparently density-independent, was an
   artifact: with a trivial loop body GCC contracts the scan to a branchless
   `hits += (b != 0)` (confirmed in `objdump`). The real kernel calls
   `box_box_distance` and `fold_pair` there, so the branch survives. The corrected
   figure is 0.87–0.97 ns/cell and *is* density-sensitive; the branchless number
   was ~2× optimistic.
2. Running several densities back-to-back in one process is ~2× optimistic — the
   branch predictor memorises the occupancy pattern. Each density is now a separate
   process invocation.

**Treat the scan figures as ±6 %, not the ±0.5 % the interquartile ranges
suggest.** The SDF crossover at ~1 600 cells is extrapolated beyond the 10 %
measured range and is marked as such.

---

## 10. Integration cost, and what a safety-WG review would have to assert

### 10.1 Hazard-log Entry 012 — not engaged, re-verified for this representation

Entry 012 obliges the kernel's `support_contact_exempts` and the octomap bridge's
`payload_clearing.support_patch_withholds` to move together so `withheld ⊆ exempt`
holds by construction. Both were re-read for this study rather than inherited from
#157:

* `support_contact_exempts` (`collision.cpp:1206-1241`) takes an `AttachedObject`,
  its attested support plane, a cell centre and the resolution. It does not
  receive a `CollisionModel` at all.
* `support_patch_withholds` (`payload_clearing.cpp:165-195`) keys on `SupportPatch`
  only. The bridge's `surface_distance` and `bounding_radius` switch on the
  **attached payload's** wire primitives.

Neither reads robot link geometry at any arity. **The lockstep is not engaged by a
robot-link representation change** — #157's conclusion, confirmed independently
and shown to generalise from a swept box to a hull, k-DOP, sphere union or SDF,
because the coupling is entirely payload-side.

`check_attached_voxel_collision` was checked specifically: it does **not** read
robot link boxes — its `CollisionModel` parameter is literally unnamed
(`collision.cpp:1040`), and its window comes from the attached primitive's own
extents.

`check_attached_self_collision` **does** read them (`collision.cpp:1341-1372`,
comparing payload primitives against robot link boxes). It has no exemption path
of any kind, so it is not an Entry-012 engagement, but a tighter link
representation makes that check less conservative too and it belongs in the hazard
entry by name — a reviewer scanning for Entry-012 exposure will look at
payload-vs-robot comparisons first.

**Attached payloads keep their current primitive path.** Extending the change to
`AttachedPrimitive` would engage the lockstep on both sides at once —
`primitive_cell_box` would under-report the payload AABB exactly as §5 describes,
and `bounding_radius` would under-report the clearing reach — so `withheld ⊆ exempt`
would need re-proving against new geometry. There is no measured motivation to,
and doing so converts a one-package change into an Entry-012 change.

### 10.2 Schema

`collision_geometry` is `list[LinkCollisionGeometry]` with **no uniqueness
constraint on `link_name`** and no `RobotDescription` validator touching it, so a
multi-primitive link already validates. `schema_version` is
`Literal["0.1"]` with no migration machinery; adding a union member is a widening
change under §1.6 — every existing manifest still validates — so no bump and no
migrator.

**One finding that should gate any new shape variant — now closed.** This study
found `CollisionShape` documented as a discriminated union while **not being
one**: there was no `Field(discriminator="shape")` anywhere in `schemas.py`, and
resolution worked only through pydantic v2 smart-union plus `extra="forbid"`
plus the `Literal` defaults. A new variant whose field set is a superset of an
existing member's would have resolved to the wrong member *silently*. The alias
now carries `Field(discriminator="shape")`, which covers all three `shape:`
sites at once (`LinkCollisionGeometry`, `WorldCollisionPrimitive`,
`AttachedCollisionPrimitive`) because the annotation lives on the alias rather
than on each field. Measured effect: an unknown tag went from six per-variant
errors naming none of the valid tags to one error enumerating
`'capsule', 'sphere', 'box'`, and an untagged mapping went from being silently
accepted as a `SphereShape` to `union_tag_not_found`.

The narrowing is real but lands on the published contract rather than past it:
the tag has been the documented discriminator since the alias was introduced,
and all 116 `collision_geometry` entries across the 17 committed manifests
already write it explicitly. `schema_version` stays `"0.1"` with no migrator
(§10.2's reading of §1.6). The generated JSON Schema now emits `oneOf` plus a
`discriminator` mapping instead of a bare `anyOf`, so an external consumer can
finally see which field is the tag — note it stays *looser* than runtime, since
the per-variant `Literal` defaults keep `shape` out of `required`. Dropping
those defaults would close that gap but break direct construction at ~40 sites,
so they stay.

**Two** downstream consumers still **fail open** on an unknown shape and would
mis-lower a new variant rather than reject it: `envelope_loader.py:605` (an `else`
that treats anything not a `BoxShape` as a capsule) and
`AttachedCollisionPrimitive.fill_idl` (`schemas.py:2603-2618`, an `isinstance`
chain with no `else`, which would emit a message with `shape_type` and
`shape_dimensions` never assigned).

A third — the CLI's primitive measurement — **was** the same pattern and is now
fail-closed: [PR #160](https://github.com/OpenRAL/openral/pull/160) rewrote it to
`raise ROSConfigError` on an unrecognised shape
(`python/cli/src/openral_cli/collision.py:188-190`). That is the pattern the
other two should follow, and it is worth noting that it took a regression when
`BoxShape` landed to motivate it.

### 10.3 Wire format and the allocation constraint

The kernel's collision parameters are flat typed arrays with an explicit per-primitive
link tag, and `n_boxes` derives from `box_link.size()` independently of `n_links`
— repeated link indices are already legal. **A variable number of fixed-arity
primitives per link therefore needs no wire or parameter change at all**;
multi-OBB and sphere unions fit today.

A variable-length *per-primitive* payload — a hull's vertex list — does need a new
array shape: the arity checks hard-code the 3-and-6 strides. The fix is a CSR
offset layout (`hull_link`, `vertex_first`, `vertex_count`, `vertices`), and that
idiom already exists in-tree — `AttachedModel` uses `prim_first`/`prim_count` and
`touch_first`/`touch_count`. New parameter arrays and a new arity-validation
block; not a schema change.

**The binding real-time constraint** is `NoAlloc.ForwardKinematicsAndSelfCollisionAreAllocationFree`
(`test_collision.cpp:619-678`), which counts allocations over 10 000 iterations.
The fixed-arity `Obb`/`Capsule` PODs exist precisely so `CollisionModel` is
pre-sizable at configure time. Any variant must be a bounded, pre-sized CSR array
or this guarantee is the first thing that breaks. §6 confirms a GJK narrow phase
can be written allocation-free (fixed-size stack simplex), so this is satisfiable
— but it must be designed in, not discovered.

An SDF is the exception: a per-link voxel buffer as a ROS `double[]` parameter is a
poor fit and would want an out-of-band asset reference plus a load step — a
genuinely new configure-time surface, and the reason §12 does not recommend it
despite its excellent cost number.

### 10.4 Lowering, and a live trap

The one-primitive-per-link limit is `envelope_loader._capsules_by_link`
(`envelope_loader.py:498-514`) — a hard `ROSConfigError`, plus a
`dict[str, LinkCollisionGeometry]` return type that could not express two
primitives even with the raise removed. The kernel has always supported several
primitives per link, and `mjcf_lowering` already emits them
(`mjcf_lowering.py:337-346`), exercised on every H1 run by
`tests/sim/safety/test_kernel_h1_self_collision.py:125-128`.

One citation correction: that test drives the **MJCF** lowering path, not a
`RobotDescription.collision_geometry` manifest. It proves the *kernel* and the
*MJCF lowering* support N-per-link; it must not be cited as evidence that the
manifest path does.

`urdf_lowering` does something worse than reject: it **silently collapses**.
`lower_link_geometry` unions all of a link's `<collision>` elements into one PCA
capsule, and two dict sites are last-wins. Re-lowering `panda_mobile` from its
URDF today would replace boxes measuring 28–76 mm of slop with capsules measuring
96–112 mm (#157 §4.3).

That trap **was** live and is now guarded rather than fixed.
[PR #160](https://github.com/OpenRAL/openral/pull/160) measured the regression at
**1.87–3.67× volume on `panda_mobile`**, found a second unrecorded case at
**`so101_follower`, up to 4.95×**, and made `openral collision lower --write`
refuse outright (exit 3, no override flag). The underlying lowering is still
wrong — a mesh still lowers to a PCA capsule — so it remains a blocker for any
re-lowering workflow, but it can no longer loosen a shipped envelope silently.

### 10.5 What the hazard entry and the safety-WG review would have to assert

1. **Containment, per link, as a proof and not a sample** — every mesh vertex
   inside the primitive, with the achieved margin reported for the exact shipped
   values. For the hull and k-DOP this is definitional (§4), which is the point of
   choosing them.
2. **An explicitly declared headroom**, replacing the accidental 0.055–0.132 mm.
3. **The candidate is a subset of the shipped OBB**, asserted at configure time by
   vertex containment (§5). This is the assertion that keeps the broad phase
   unchanged and it must be enforced at load, not assumed.
4. **The broad phase was verified, not merely unchanged** — with a regression test
   that places the occupied cell at the outer face of an unclamped window, because
   the existing test passes with the reach term deleted (§5).
5. **The narrow phase's conservatism direction is declared.** A k-DOP SAT stays a
   lower bound. GJK returns the **exact** distance, which is strictly *less*
   conservative than what ships today — mean 0.16 mm, near-contact max 22.5 mm
   (§6.2). This loss is invisible in a "tighter geometry" framing and must be
   stated separately from the shape change.
6. **`check_attached_self_collision` becomes less conservative too**, bounded by
   the same per-link Δ (§10.1).
7. **The recovery claim is the one in §7, not a larger one** — 25/72 guaranteed for
   the recommended configuration, and 45/72 states unreachable by any link-geometry
   change. A hazard entry justifying a smaller envelope on a benefit the evidence
   does not show is the exact failure mode §1.2 exists to prevent.
8. ~~**Issue #155 is fixed first, or the ACM is not regenerated**~~ (#157 §7.3
   item 6) — **discharged**: #155 is fixed, the ACM sweep uses the kernel's own
   predicates, and a regeneration now reproduces all 10 shipped ACMs
   byte-identically, so this no longer constrains sequencing. Note what it found,
   because it raises this document's stakes rather than settling them:
   `panda_link5`↔`panda_link7` is a **real** self-collision (up to 48.3 mm of
   mesh interpenetration across a 39.6° band of `joint6`) that the current box
   envelopes cannot separate from an artifact at *any* margin — real collisions
   reach a box gap of -9.97 mm, collision-free poses -40.07 mm. It therefore
   ships exempt and unchecked, under an explicit residual-risk SRDF row.
   Tightening `link5`/`link6`/`link7` is the only thing that retires it.

---

## 11. Reproducing

Measurement code is not checked in (§1.11 keeps fixtures for tests, not studies);
it is reconstructible from this section and lived in a scratch directory outside
the repo.

**Inputs.** robosuite 1.5.2 `models/assets/robots/panda/robot.xml` under
mujoco 3.8.0; the seven `link<N>_collision` mesh geoms. Vertices from
`model.mesh_vert` placed by `geom_pos`/`geom_quat` **only** — MuJoCo folds mesh
recentring into the geom frame, and `mesh_pos == geom_pos` on all seven links, so
applying both double-counts. Cross-checked against the raw STL assets
(`~/.cache/openral/repos/robosuite/.../panda/meshes/link{1..7}.stl`) to 8 × 10⁻⁹ m.
Manifest values from `robots/panda_mobile/robot.yaml`; rotation convention
`Rz·Ry·Rx`, matching `transform_from_xyz_rpy`.

**Support excess.** `h_C(u) − h_mesh(u)` over a 20 000-direction Fibonacci set.
For polytopes `h_C(u) = max_i u·v_i`; for sphere unions `max_i (u·c_i + r_i)`; for
cell unions `max_i (u·c_i) + p/2·(|u_x|+|u_y|+|u_z|)` — the union-of-cubes support,
not the ball bound.

**Candidates.** k-DOPs and tangent polytopes by `scipy.spatial.HalfspaceIntersection`
over tangent halfspaces `u·x ≤ h_mesh(u)`, axes expressed in the manifest box's
frame. Decimated hulls by taking support vertices in `k` Fibonacci directions,
then the outward offset closing containment. Spheres by volumetric k-means over a
6 mm filled voxelisation, radius = max distance to an assigned cell + that cell's
half-diagonal. Multi-OBB by triangle partition (k-means on centroids, and slabs
perpendicular to `u*`), each part's box in the shipped frame.

**Exact distances** for the SAT-deficit cross-check as a 6-variable SLSQP QP
(`x` in the body's halfspaces, `y` in the cube), not a hand-rolled GJK — the
first attempt used one and returned implausible 55 mm deficits.

**Census states** built through `robocasa.utils.env_utils.create_env` with
`layout_ids` pinned and seed 1, `MUJOCO_GL=egl`; world occupancy from solid
(`contype`/`conaffinity` non-zero) non-robot geoms with adaptive barycentric
triangle subdivision to 5 mm; the kernel's `box_box_distance` and window
arithmetic ported to numpy from `collision.cpp:327-366` and `:691-703`.

**SDF error.** Signed-distance grids built with `trimesh.proximity.signed_distance`
over a 40 mm padded box, queried by trilinear interpolation at exterior points
2–60 mm off the surface along **outward face normals** (a first pass used random
directions, which put samples inside the mesh and corrupted the mean; the maximum
over-report was unaffected, since an interior point cannot over-report). Truth
from `trimesh.proximity.closest_point`; exterior membership from the sign of the
field. The conservative bound is analytic, `p·√3/2` for a 1-Lipschitz field.

**Benchmarks.** `g++ 13.3.0`, `-std=c++17 -O2 -g -DNDEBUG` (the kernel's real
flags — `CMakeLists.txt` defaults to `RelWithDebInfo` and sets no `-march`),
`taskset`-pinned on an i5-8600K, 151 repetitions per candidate,
poses pre-generated, global `operator new` replaced with counting wrappers to
verify zero allocations inside every timed region. Scan medians are 5 separate
process invocations × 151 repetitions, because running densities back-to-back in
one process lets the branch predictor memorise the occupancy pattern (~2×
optimistic); scan figures carry **±6 %**, not the ±0.5 % the interquartile ranges
suggest (§9.3).

Environment: workspace `.venv` — numpy 2.2.6, scipy 1.17.1, trimesh 4.12.2,
mujoco 3.8.0, robosuite 1.5.2, robocasa. CPU only; no GPU work is involved.

---

## 12. Recommendation

**12.1 The binding constraint is no longer the link envelope, and the study
should be closed on that finding.** Perfect link geometry recovers 27 of 72
census stops in the worst case and 42 of 72 nominally. The remaining 45 (or 30)
are held by the 25 mm voxel grid, whose half-diagonal is 21.65 mm and whose
lattice phase alone swings the reported distance by a measured 20.3 mm. **Below
about 10 mm of link excess, tightening buys one state per 2 mm.** If the goal is
to unblock cluttered kitchens, the next study is about the world grid, not the
robot.

**12.2 If a link-geometry change is made, make it a staged 26-DOP → exact convex
hull, scoped to `link1` and `link2`.** Stage 1 is a 26-DOP SAT (62.8 ns, a
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
be touched. Scoping also bounds the cost problem, since only two links pay GJK at
all.

**`link1`'s 1588-vertex hull needs an explicit decision.** Warm-started it breaks
the 100 Hz budget at 120 occupied cells — about 1 % of a link's window — and the
utensil scene, where `link1` is the stopping link, is exactly the adverse case
(§9). Either the 787-vertex tangent polytope (which matches the exact hull's
*guaranteed* recovery, 10/14, and loses only in the face-on column) or a hard cap
on stage-2 invocations per step. Do not ship the raw 1588-vertex hull on an
always-on path.

**12.2b The SDF is the strongest cost candidate and the weakest safety case; it
is the right answer only if clutter, not recovery, becomes the binding
requirement.** At 64³ it is 21× cheaper than today's routine, is the only
candidate whose worst case is **bounded at any occupancy** (100 % occupancy costs
3.0 ms of a 10 ms budget, so it can never break the deadline — a real property for
a hard-real-time kernel), and recovers 21/72. Against that: its containment rests
on a Lipschitz *bound* rather than vertex containment, its 4.53 mm effective
excess is the bound rather than the 1.27 mm it actually achieves (§3.0), it
**escapes the shipped OBB** so the broad-phase reach formula must be restated and
re-reviewed (§5.1), it adds two obligations with no analogue elsewhere (domain
closure and truncation, §5.1), and a per-link voxel buffer is a poor fit for the
ROS parameter path and would need a new out-of-band configure-time asset surface
(§10.3), at 7 MiB for the fleet. That is a materially larger safety and
integration surface for 6 fewer recovered states. It should be revisited if
late-manipulation clutter turns out to drive occupancy far above the start-state
range measured here.

**12.2c The 26-DOP alone is the no-precompute fallback, and it is not enough.**
It needs no per-link refit pipeline beyond a slab fit, is 4.8× faster than today,
and is the best non-precomputed candidate under clutter (6.5 % of budget at 400
cells). But at 25.7 mm of excess it recovers **2/72 guaranteed and 13/72
face-on** — it is a genuine improvement to the kernel's cost and coarseness that
does **not** unblock the scenes. Ship it as stage 1 of §12.2, not as the answer.

**12.3 Take the cube-specialised SAT regardless — it is free.** 2.73× faster for a
bit-identical result (max difference 2.9 × 10⁻¹⁶ m), no geometry change, no
containment obligation, no safety posture change. It is the only item in this
document with no downside, and it is independent of every other recommendation.

**12.4 Do not pursue spherization, link voxelisation, or decimated-hull-plus-offset.**
Spheres saturate at 18–27 mm even at 128 per link and are the wrong basis for
flat-faced links (§3.1). Link voxelisation at the world pitch is barely better than
the shipped OBB. Both, plus the offset hulls, **escape the shipped OBB** by
10–35 mm, which grows faces that are currently tight to 0.13 mm and forces the
broad-phase reach formula open — trading the one hazard §5 shows is otherwise
avoidable for tightness that §7 shows does not pay.

**12.5 Do not pursue multi-primitive splits for this problem.** Confirmed against
the directional criterion #157 was not tested on: a split chosen explicitly for
the stopping direction gains 0.2–0.3 mm (§8.1). Lifting the
`_capsules_by_link` restriction remains worthwhile on its own merits — it is a
real asymmetry between the two lowering paths, it costs ~15 lines of Python, and
it touches no schema, no wire format and no kernel (§10.3) — but it must not be
motivated as a fix for envelope slop.

**12.6 Do not extend anything here to attached payloads.** Entry 012 is not
engaged as scoped (§10.1) and extending to `AttachedPrimitive` would engage it on
both sides simultaneously, with no measured motivation.

**12.7 The highest-value next piece of work is on the world side, and it is not a
geometry change.** Two independent measurements point at it. The recovery ceiling
is world-side: 45 of 72 states cannot be cleared by *any* link geometry at a
25 mm grid, and grid lattice phase alone moves the reported distance by 20.3 mm
(§7.5). The cost bottleneck is world-side: 99.83 % of the empty-cell scan is
wasted work and it is 65 % of the total, while compressing occupancy buys nothing
and **skipping** cells buys everything (§9.2). A hierarchical occupancy summary
would cut the dominant cost term, and grid resolution or a phase-invariant
occupancy test would move the recovery ceiling. Neither shrinks a safety envelope,
so neither carries the containment obligation everything in §12.2 does.

**12.8 Independent of all the above**, two items this study surfaced and did not
cause. The first is **now closed**: `CollisionShape` was documented as a
discriminated union but was a bare `TypeAlias` union with no
`Field(discriminator=...)` anywhere; it now carries the discriminator on the
alias, so all three `shape:` sites are tagged at once (§10.2). **The second is
now closed too**: both remaining fail-open shape branches —
`envelope_loader.py`'s bare `else` and `AttachedCollisionPrimitive.fill_idl`'s
`else`-less `isinstance` chain — now raise `ROSConfigError` naming the shape,
following the fail-closed pattern #160 established in the CLI. They were not
the same severity and the fixes say so: the lowering one was **unsafe** (an
unknown primitive became a zero-length capsule of its `radius_m`, a strictly
smaller volume than the manifest declared), while the encoder one was a
diagnosability defect only, since every consumer already refuses the IDL's
default tag `0`.

**The third item is already done.** This study flagged
`urdf_lowering.lower_link_geometry`'s mesh path as a live trap that would
silently regress `panda_mobile`'s envelope if the manifest were ever re-lowered.
[PR #160](https://github.com/OpenRAL/openral/pull/160) measured it — **1.87–3.67×
volume on `panda_mobile`** — found a second, unrecorded case at
**`so101_follower`, up to 4.95×**, and guarded both with a hard refusal in
`openral collision lower --write` (exit 3, no override flag). Nothing further is
needed here; it is recorded because §10.4's description of the trap is now
history rather than a standing hazard.

---

## See also

* [Collision-primitive study (PR #157)](collision-primitive-study.md) — the geometry argument and the sphere-swept box.
* [What the kernel's link boxes enclose (PR #158)](collision-primitive-images.md) — the corrected protrusion figures this study reproduces.
* [RoboCasa start-state census (PR #159)](robocasa-start-state-census.md) — the 72 stopping states scored in §7.
* [Self-occupancy settled, and the re-lower guard (PR #160)](collision-validation-evidence.md) — the independent live-model corroboration of the world-side conclusion (§7.7), and the guard that closed §10.4's re-lowering trap.
* [Collision-stack validation evidence](collision-validation-evidence.md) — the round-by-round ledger and standing caveats.
* `cpp/openral_safety_kernel/README.md` — the kernel geometry this study measures against.
