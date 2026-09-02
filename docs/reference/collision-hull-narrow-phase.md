# The staged hull narrow phase — what shipped, what it cost, and what it did not do

> **Status: implemented, safety-WG review pending — hazard-log Entry 018.**
> (Drafted as Entry 014; renumbered because 014 was already taken by the
> 2026-08-16 Panda box→capsule entry in the private log.) This is the change record for
> the staged 26-DOP → exact-convex-hull narrow phase in
> `check_voxel_collision`'s box pass. It is scoped to **arm-link vs
> world-voxel** checks only.
>
> **Extended 2026-09-02 by [issue #191](https://github.com/OpenRAL/openral/issues/191)**
> to `check_self_collision`'s box↔box pass, which is why §9 exists. That
> extension is what retired the `panda_link5`↔`panda_link7` ACM exemption — the
> one open item the collision-safety alternatives survey §2.2 left on the ACM
> theme ("#169 changed no manifest, so the unsafe exemption is still in the
> manifests"). `panda_link5` and `panda_link7` now declare
> `tight_geometry` too, so §5.1's "today that is link1 and link2" reads
> link1, link2, link5 and link7.
>
> The evidence this rests on was produced by four earlier studies and is cited,
> not re-argued: the [collision-primitive study](collision-primitive-study.md)
> (#157), its [picture book](collision-primitive-images.md) (#158), the
> [RoboCasa start-state census](robocasa-start-state-census.md) (#159), the
> [validation-evidence ledger](collision-validation-evidence.md) (#160), and
> above all [tight link geometry in the safety
> kernel](collision-tight-geometry.md) (#161), whose §12.2 is the design this
> implements. **Every number in §2 and §3 below was re-measured on this
> machine against this implementation**, not carried over.

---

## 1. The question that was asked, and the answer

The request was: *do the arm-to-world-voxel safety check via convex hull rather
than primitives, and deprecate the primitive code — if it is faster and more
precise.*

**More precise: yes, unconditionally.** The shipped OBB over-reports the link's
reach by 53.27 mm on `panda_link1` and 46.83 mm on `panda_link2`. The 26-DOP
cuts that to 25.69 mm and 23.01 mm; the exact convex hull cuts it to **0.00 mm**,
by definition (`h_hull(u) = h_mesh(u)`).

**Faster: yes, but only when the exact stage is bounded** — and the bounds are
the substance of this change, not an afterthought:

* the **26-DOP** stage is 8.5–10× cheaper per query than `box_box_distance`
  (30–36 ns against 307 ns) and is the reason the whole thing pays;
* the **exact hull** stage is 2–2.5× *more* expensive per query than the routine
  it replaces, so it is affordable only because it runs on the small minority of
  cells the DOP cannot clear;
* an **unbounded** exact hull is a straightforward loss. `panda_link1`'s hull is
  1588 vertices and measured **0.77× the shipped routine's speed** at 400
  occupied cells. The study predicted this (§12.2: "Do not ship the raw
  1588-vertex hull on an always-on path") and the implementation refuses it.

**Deprecate the primitive code: no, and a blanket removal would be wrong.** §5
sets out exactly what was replaced and what was not, and why the "not" list is
longer than the "yes" list.

**One result came out against the framing and against my own expectation**, and
it changed the implementation: a 26-DOP is a strictly smaller solid than the OBB
that contains it, but it does **not** follow that its separating-axis bound is
always the larger number. See §4.2. The kernel now folds the shipped bound back
in, and the change is provably unable to introduce a stop that does not already
happen today.

---

## 2. Precision, measured

Max support excess `max_u [h_C(u) - h_mesh(u)]` over a 20 000-direction
Fibonacci set, against robosuite 1.5.2's `link<N>_collision` meshes placed by
the geom transform only ([#161 §11](collision-tight-geometry.md)). Reproduced on
this machine; it matches #161 §3 to 0.1 mm on all seven links, which is the
cross-check that the pipeline measures the same thing.

| link | shipped OBB | 26-DOP (stage 1) | exact hull (stage 2) |
|---|---:|---:|---:|
| `panda_link1` | 53.27 mm | **25.69 mm** | 0.00 mm |
| `panda_link2` | 46.83 mm | **23.01 mm** | **0.00 mm** |
| link3 | 75.57 | 23.84 | 0.00 |
| link4 | 76.12 | 23.15 | 0.00 |
| link5 | 45.20 | 18.99 | 0.00 |
| link6 | 52.70 | 21.53 | 0.00 |
| link7 | 28.25 | 12.97 | 0.00 |

Links 3–7 are measured but **not shipped** — see §5.2.

`tests/unit/test_collision_tight_geometry.py` pins the `link1`/`link2` rows, so
a change that quietly stopped tightening fails rather than passing every
containment test and going unnoticed.

---

## 3. Cost, measured

**Host**: Intel Core i5-8600K, 6C/6T, 4.2 GHz sustained, x86_64, Linux 6.8.0-137,
`g++ 13.3.0`, `-std=c++17 -O2 -g -DNDEBUG`, no `-march` — the kernel's real
flags (`CMakeLists.txt` defaults `CMAKE_BUILD_TYPE` to `RelWithDebInfo`). This
is the same machine #161 measured on, and `box_box_distance` reproduces there at
307.0 ns against its published 304.0 ns.

`taskset`-pinned, median of 151 repetitions, all fixtures pre-built. **Allocation
freedom was verified, not assumed**: global `operator new`/`new[]` were replaced
with counting wrappers and the count sampled either side of every timed region —
**0 allocations inside every timed region** in the benchmark, and pinned in CI by
`NoAlloc.ForwardKinematicsAndSelfCollisionAreAllocationFree`, which now drives the
staged path (GJK simplex included) 10 000 times under the same counter.

### 3.1 Per query

One link primitive against one occupied 25 mm cell, in the kernel's own
streaming pattern.

| narrow phase | ns/query | vs shipped |
|---|---:|---:|
| `box_box_distance` (ships today) | 307.0 | 1.00× |
| stage 1 — 26-DOP SAT, `link1` | **28.7** | **10.7× faster** |
| stage 1 — 26-DOP SAT, `link2` | **30.3–35.3** | **8.7–10.1× faster** |
| stage 2 — GJK on `link2`'s 152-vertex hull, warm | ~590–740 | 1.9–2.4× slower |
| stage 2 — GJK on `link1`'s 1588-vertex hull, warm | ~3 700 | 12× slower |

Stage 1 is cheaper than #161's published 62.8 ns because every term that depends
only on the link pose — the 13 axes in the base frame, their dot with the box
origin, the cell's support radius on each — is hoisted out of the cell loop by
`tight_pose_init`. The kernel streams cell centres against a fixed pose, so that
hoist is free.

### 3.2 End to end, through the real `check_voxel_collision`

Seven-link Panda at a mid-reach configuration, 96×96×60 grid at 25 mm (a
room-sized 0.55 MB map), 10 475 cells across the seven link windows. Occupancy
is the *N cells closest to the arm*, ranked by the shipped routine so every
variant sees an identical grid — deliberately adverse, since real clutter is
spread out rather than pressed against every link.

`world_voxel_margin_m` is **0.0 in sim** and **0.02 on real hardware**
(`packages/openral_rskill_ros/launch/sim_e2e.launch.py`), so both are measured.

**At the sim margin (0.0 m) — the configuration `panda_mobile` actually runs:**

| occupied cells | shipped | staged (as shipped) | raw 1588-vtx hull on link1 |
|---:|---:|---:|---:|
| 0 | 8.71 µs | 8.81 µs (0.99×) | 8.81 µs |
| 20 *(the census's measured occupancy)* | 22.34 | **21.49 (1.04×)** | 22.28 |
| 100 | 70.75 | **64.01 (1.11×)** | 89.32 (0.79×) |
| 400 | 255.06 | **235.83 (1.08×)** | 345.56 (0.74×) |
| 1200 | 728.48 | **665.73 (1.09×)** | 995.57 (0.73×) |

**At the real-HAL margin (0.02 m):**

| occupied cells | shipped | staged (as shipped) |
|---:|---:|---:|
| 20 | 26.30 µs | 27.22 (0.97×) |
| 100 | 79.86 | 97.13 (**0.82×**) |
| 400 | 287.10 | 309.11 (**0.93×**) |
| 1200 | 841.99 | 823.08 (1.02×) |

**Read the second table as the standing caveat it is.** A wider margin pushes
more cells past stage 1's clear test and into stage 2, and stage 2 is the
expensive half. `panda_mobile` is sim-only (`hal.real: null`), so the shipped
configuration is the first table — but **any real-HAL rollout of tight geometry
must re-measure at that robot's own margin before enabling it**, and that
condition belongs in the hazard entry rather than in a footnote here.

### 3.3 Why the totals move so little

Because the empty-cell scan dominates, exactly as #161 §9.2 found: at 20
occupied cells in a 10 475-cell window, more than 99.8 % of the loop is the
`if (occupancy[idx] == 0) continue;` fast path. Making the narrow phase 10×
cheaper moves a term that is a third of the cost. **This confirms #161 §12.7
from the implementation side: the remaining work is on the world grid, not the
robot**, and nothing here should be read as contradicting that.

---

## 4. The safety case

### 4.1 The containment chain

```
link mesh  ⊆  exact convex hull  ⊆  26-DOP  ⊆  shipped OBB
```

Every link is **definitional, not fitted** — there is no optimiser tolerance
anywhere in the argument:

| link | why it holds | where it is proved |
|---|---|---|
| `mesh ⊆ hull` | the hull *is* `conv(mesh vertices)` | `tests/unit/test_collision_tight_geometry.py`, facet by facet against the real robosuite mesh |
| `hull ⊆ 26-DOP` | the slabs are **tangent** halfspaces `u·x ≤ max_mesh(u·x)`, so every mesh point satisfies every constraint by construction | `TightCollisionGeometry` validator, and again in `validate_tight_geometry` at `on_configure` |
| `26-DOP ⊆ OBB` | a 26-DOP lies inside its own first three slabs, and those three axes **are** the box's axes | `LinkCollisionGeometry` validator, `validate_tight_geometry`, and the Python test — with **no tolerance**, because the margin here is real |

Measured inward margin of the DOP inside the shipped box: **0.0831 mm on
`link1`, 0.0546 mm on `link2`** — reproducing #157 §7.3's observation that the
shipped boxes carry 0.055–0.132 mm of accidental headroom and nothing may
consume it.

The one place a floating-point tolerance appears is `hull ⊆ DOP`, at
`kTightContainmentEpsilonM = 1e-9` m. That relation is an *equality* on at least
one vertex per axis by construction, so the only open question is evaluation
order; one nanometre buys numerical agreement, not geometric room.

### 4.2 The result that changed the design

`check_voxel_collision` sizes its broad-phase window from `half_extents` alone.
#157 identified this as the one place a representation change can make the
kernel **unsafe** rather than merely tighter: an under-sized window never visits
cells that genuinely intersect the solid, and the check silently returns clear.
Because both stages are proved subsets of the shipped OBB **at configure time**,
that window is not touched — not widened, not narrowed, not restated. The
formula at `collision.cpp` remains `Σ_j |R_kj|·he_j` about the box origin, and
the trap cannot fire.

What did not survive contact with a test is a weaker, more tempting claim:
*"the DOP is a smaller solid, so its bound must be at least as large as the
box's."* **That is false.** The DOP's 16 separating axes (13 body + 3 world) are
not a superset of `box_box_distance`'s 15 (6 face + 9 edge-cross); the box SAT's
edge-cross axes beat every DOP axis at some cells, so the *tighter solid* can
still yield the *looser bound*. A test written to assert the intuition failed on
real `panda_link1` geometry at the first rotated pose.

The consequence is behavioural, not unsafe — a smaller reported distance is more
conservative — but it would mean **new false stops**, which is a regression even
though it is a safe one, and it would undercut the entire justification for
touching this code. So `check_voxel_collision` folds `box_box_distance` back in
for any cell the staged path is about to stop on, and takes the maximum. The
maximum of two lower bounds is a lower bound, so this costs no soundness and
buys a strong property:

> **The staged path can never stop where the shipped path would not have.**
> It removes stops the box was causing; it cannot add one.

Pinned by `VoxelCollisionTightGeometry.NeverStopsWhereTheShippedPathWouldNotHave`
over ~22 000 single-cell grids on the real seven-link Panda, and by
`TightNarrowPhase.TheDopAloneIsSometimesTheMoreConservativeOfTheTwo`, which
fails if the fold ever *becomes* redundant so nobody deletes it as dead weight.

### 4.3 Conservatism, stated in the right direction

This is easy to state backwards, so plainly:

* **Against the true geometry**, the staged path never over-reports clearance.
  Every value it can return — stage 1's SAT bound, stage 2's converged distance,
  stage 2's truncated bound, the overlap fallback, the budget-exhausted fallback
  — is a lower bound on the true link-mesh-to-cell distance. That is the safety
  property, and it is what the tests check.
* **Against the shipped OBB**, it deliberately reports *more* clearance, by up
  to 53 mm on `link1`. Removing excess conservatism is the change. Keeping the
  kernel sound is the constraint. A hazard entry that blurred these two would be
  the failure mode CLAUDE.md §1.2 exists to prevent.

Every early exit is conservative by construction, which is what makes the
real-time bounds safe rather than merely convenient:

| bound | what happens when it binds | direction |
|---|---|---|
| `kGjkMaxIterations = 24` | returns the best supporting-hyperplane bound so far | more conservative |
| early exit once the bound clears `margin` | stops sharpening a diagnostic | more conservative |
| `kMaxStage2PerCheck = 32` | falls back to stage 1 + `box_box_distance` | today's answer or better |
| `kMaxTightHullVertices = 320` | link never gets stage 2 at all | stage 1 only, still a tightening |
| GJK detects overlap | returns stage 1's bound, which is ≤ 0 there because the DOP contains the hull | more conservative |

### 4.4 Why the support scan is exhaustive

`hull_cell_distance`'s support function is a linear scan of the vertex list, and
that is a deliberate refusal of the standard optimisation. Hill-climbing over
the hull's edge graph is the usual way to make GJK cheap on large hulls, and
under exact arithmetic it is correct for a convex polytope. Under floating point
it can stop one vertex short of the true support — and a support that is not the
true maximum turns the supporting-hyperplane bound from a lower bound into an
**over-report of clearance**. The kernel buys soundness with a linear scan and
bounds the cost with `kMaxTightHullVertices` instead. That trade is the direct
cause of `panda_link1` shipping stage 1 only (§5.2).

---

## 5. Scope — what was replaced, and what was not

### 5.1 Replaced

Two call sites, both gated on a manifest declaring `tight_geometry`:

* the `box_box_distance` in `check_voxel_collision`'s **box pass** (this
  document's original scope) — `panda_link1` (stage 1) and `panda_link2`
  (stages 1 and 2);
* the `box_box_distance` in `check_self_collision`'s **box↔box pass**, added by
  #191 and described in §9 — `panda_link5` and `panda_link7` (stages 1 and 2)
  joined the first two for this.

### 5.2 Not replaced, and why

| surface | keeps the primitive path because |
|---|---|
| **attached payloads** (`check_attached_voxel_collision`, `check_attached_world_collision`, `check_attached_self_collision`) | §6. Extending here would engage hazard-log Entry 012's lockstep on both sides at once, and there is no measured motivation |
| **world-capsule obstacles** (`check_world_collision`) | out of scope; not voxel geometry |
| **capsule-lowered robots** (`h1`, `rizon4`, every MJCF-lowered model) | tight geometry refines a `BoxShape`; a capsule has no box to state the containment proof against, and the schema refuses it |
| **the broad-phase window** | §4.2 — this is the one thing that must not move |
| **`panda_link3`, `link4`, `link6`** | #159: they hold **zero** of the 72 census stops, and they are not half of a self-pair the boxes cannot separate. Adding them would be three more containment proofs and three more hazard-entry lines for no measured recovery |
| **`panda_link1`'s exact hull** | 1588 vertices, over `kMaxTightHullVertices`, measured 0.74–0.79× the shipped routine's speed. It gets stage 1 only |

> `check_self_collision` was in this table as "out of the asked scope" until
> #191, on the reasoning that leaving it untouched kept
> `check_attached_self_collision`'s conservatism unchanged. That is still true —
> the attached path is untouched — but the self path is no longer, because the
> ACM exemption it was propping up turned out to be the more expensive of the
> two positions. §9.

`box_box_distance` itself is **not deprecated**. It remains the narrow phase for
every link without tight geometry, the fold in §4.2, and the routine every other
check uses. What is deprecated is *relying on the OBB as the arm-link↔world-voxel
narrow phase for a link that could declare tight geometry* — a manifest-authoring
posture, not dead code.

### 5.3 What #171 measured against the live map, and what it means for this change

[PR #171](world-map-fidelity.md) landed after this work and re-scored the
question against the **live** octomap rather than an idealised grid, apportioning
each stop by whether exact link geometry clears it. Its numbers are the best
available statement of what this change is worth, and they cut both ways:

| | n | share |
|---|---:|---:|
| **link-side** — exact geometry clears it | 6 | **26 %** |
| **world-side** — exact geometry still stops | 17 | **74 %** |

*(23 live stops across four scenes.)* And within the world-side residue, **11
stops — 48 % of all stops — are held by the octree→grid bridge's rasterisation
dilation**: a cell that contains nothing, sits in no leaf, with real surface
about one cell away. `rasterize_octree_to_grid` marks a base-frame cell whenever
its cube *shares volume* with an occupied leaf's cube, and the two lattices have
an arbitrary relative phase (and yaw, on a mobile base), so one leaf generically
lights up to eight cells. That is [issue #173][i173] — a separate safety-WG
decision, and the single largest term in the whole apportionment. The other
world-side mechanism, non-collidable geometry being real to the depth camera, is
[issue #174][i174].

[i173]: https://github.com/OpenRAL/openral/issues/173
[i174]: https://github.com/OpenRAL/openral/issues/174

**So this change is necessary and correct, and it is not sufficient.** It fixes
the link-side term exactly; the majority term is elsewhere and now has an owner.
Neither fact argues against the other: the link-side excess #171 removes is
large (median **28.72 mm**, up to 42.56 mm on `panda_link2`), and it is real
clearance the robot has today and is not being credited with.

#### The utensil scene: a measured recovery attributable to this change

`robocasa_drawer_utensil` at its shipped pin takes a marginal initial-configuration
stop the ideal-grid census never saw: the live map reads **−1.94 mm**, and the
same state under exact geometry reads **+20.01 mm**. #171 calls it *entirely
link-side* and names this PR as removing it with no scene change.

That is worth stating precisely rather than banking, because **the stop is on
`panda_link1`, the link that ships stage 1 only.** #171's EXACT column is the
convex-hull surface — what stage 2 reaches — and `link1` does not get stage 2
here. So the question is whether the **26-DOP alone** clears it.

Decomposing as `d_C = d_true − E_C(u*)`: the exact reading gives
`d_true = +20.01 mm`, so the shipped box's realised excess along the approach
direction is `E_OBB(u*) = 21.95 mm`, and the DOP clears the stop iff
`E_DOP(u*) < 20.01 mm`. `u*` is not published, so it was bracketed over a
200 000-direction Fibonacci set on the real `link1` mesh:

| direction set | n | max `E_DOP` | median | clears? |
|---|---:|---:|---:|---|
| `E_OBB(u)` within ±0.5 mm of the measured 21.95 mm | 5 507 | **18.04 mm** | 4.27 mm | **yes**, 1.97 mm spare |
| `E_OBB(u) ≥ 20.01 mm` (a strictly larger, more adverse set) | 155 529 | 25.70 mm | 4.95 mm | not guaranteed |
| all directions | 200 000 | 25.70 mm | 4.40 mm | not guaranteed |

The first row is the one conditioned on what was actually measured, and every
direction consistent with `E_OBB(u*) = 21.95 mm` leaves the DOP under the
threshold. The rows below it are reported because they are the honest
sensitivity: they include directions where the box is far prouder than it was
at this state, and the DOP's worst case does reach its full 25.70 mm there.

**Three caveats keep this a strong expectation rather than a proof.** The 1.97 mm
of spare is a *geometry* term; the DOP's own separating-axis deficit eats into it,
and #161 §6.2 measures that deficit at mean 0.159 mm near contact (exact 80.2 %
of the time) but with a 22.5 mm tail. The decomposition assumes the approach
direction is the same for both representations. And #171's EXACT column is a
**densely sampled** surface distance (5 mm step) — an upper bound on the true
gap, conservative in the safe direction, but it means the `+20.01 mm` input is
itself bracketed rather than exact. (It is *not* affected by the
`mj_geomDistance` defect #170 fixed: #171 never used that probe, and #175
records that nothing on that page needs redoing.) The floor is firm
regardless: the kernel folds `box_box_distance` back in, so the reported
distance is never worse than today's −1.94 mm. For a state stopping on
`panda_link2` the same argument is a guarantee rather than an expectation,
because there the shipped representation *is* exact.

### 5.4 What this does **not** buy

#161 §7.3 is the honest ceiling and it has not moved: **perfect link geometry
recovers 27 of the census's 72 stopping states; 45 are held by the world grid.**
The recommended configuration (exact hull on link1 *and* link2) scores 25/72;
what shipped here is less than that, because `link1` gets the DOP rather than its
hull. This change is justified by **cost and correctness**, not by unblocking
scenes, and it should not be described as unblocking anything. The 25 mm voxel
grid — its resolution, its 20.3 mm lattice-phase swing (#161 §7.5), and the
map-fidelity term #160 measured — is still where the remaining recovery lives.

---

## 6. Hazard-log Entry 012 — not engaged, re-verified against this implementation

Entry 012 obliges the kernel's `support_contact_exempts` and the octomap
bridge's `payload_clearing.support_patch_withholds` to move in lockstep so
`withheld ⊆ exempt` holds by construction. #157 and #161 both concluded it is not
engaged by a robot-link representation change. Re-verified here **against what
was actually written**, not inherited:

* `support_contact_exempts` takes an `AttachedObject`, its attested support
  plane, a cell centre and a resolution. It receives no `CollisionModel`, and
  this change added no argument to it.
* `support_patch_withholds` keys on `SupportPatch` only; the bridge's
  `surface_distance` and `bounding_radius` switch on the **attached payload's**
  wire primitives, which this change does not touch.
* `check_attached_voxel_collision`'s `CollisionModel` parameter is still
  literally unnamed.
* `check_attached_self_collision` **does** read robot link boxes — and still
  reads exactly the boxes it read before. Tight geometry lives in `box_hull` /
  `hulls` / `hull_vertices` and is consulted at **one** call site, the box pass
  of `check_voxel_collision`. So unlike the change #161 costed, this one does not
  make `check_attached_self_collision` less conservative at all; its conservatism
  is bit-identical.

**Verdict: Entry 012 is not engaged, and the payload-vs-robot check named in
#161 §10.5 item 6 is not affected either.**

### 6.1 The `kernel_predicates.py` lockstep — checked, and it does not bite

[PR #169](https://github.com/OpenRAL/openral/pull/169) added
`packages/openral_safety/openral_safety/kernel_predicates.py`, a line-by-line
Python mirror of the kernel's narrow phase, and a duplication-watch obligation
that it move in lockstep with `collision.cpp` — naming this PR explicitly.

Checked rather than complied with: the mirror exists so the **offline ACM
sweep** asks the kernel's own question, and the ACM sweep is about
**link-vs-link self-collision**. `shape_distance` reproduces
`check_self_collision`'s type routing, nothing else. This change is confined to
`check_voxel_collision`'s box pass — `check_self_collision`, `box_box_distance`,
`box_capsule_distance` and `capsule_distance` are byte-identical on this branch,
so an ACM regenerated with or without it is the same matrix.

`kernel_predicates.py` is therefore **unchanged, deliberately**, and
`docs/methods/14-duplication-watch.md` item 12 was corrected to scope the
obligation to the narrow phase it actually mirrors — in both directions, so a
future change to the *self* path still owes it an update however small. The broad-phase reach formula,
stated explicitly as the brief requires: for a link transform `(R, t)`, per world
axis `k`, the window half-reach is

```
e_k = Σ_j |R_kj| · half_extents_j          (about t, plus margin + half_side)
```

— **unchanged**, with `hull ⊆ DOP ⊆ Box(half_extents)` asserted at configure time
by `validate_tight_geometry`, which is what licenses leaving it alone.

---

## 7. Reproducing

```bash
# Containment against the real meshes, and the manifest-vs-mesh check.
just sync
python -m pytest tests/unit/test_collision_tight_geometry.py tests/unit/test_collision_params.py

# Re-derive the manifest blocks (emit) or verify them (check, exit 3 on drift).
python tools/generate_tight_geometry.py check --robot robots/panda_mobile/robot.yaml

# The kernel's own conservatism suite.
just safety-kernel-build && just safety-kernel-test
```

The benchmark in §3 is not checked in (CLAUDE.md §1.11 keeps fixtures for tests,
not studies). It builds `cpp/openral_safety_kernel/src/collision.cpp` with the
flags above against a generated fixture holding the shipped OBBs, the hulls, and
MuJoCo-computed link world poses at robosuite's `init_qpos`; occupancy is the N
cells nearest the arm ranked by `box_box_distance`; global `operator new` is
replaced with a counting wrapper and sampled either side of each timed region.

---

## 8. What the safety-WG must assert

Restating #161 §10.5 against what actually shipped, so a reviewer checks claims
rather than intentions:

1. **Containment is proved per link, not sampled** — `mesh ⊆ hull ⊆ DOP ⊆ OBB`,
   with the achieved margin reported (0.0831 mm / 0.0546 mm) and re-derived from
   the real mesh in CI. ✔ §4.1
2. **The broad phase was verified, not merely left alone** — the reach formula is
   unchanged and the subset property is asserted at load, fail-closed. ✔ §6
3. **The narrow phase's conservatism direction is declared, in both directions**
   — never over-reports against the truth; deliberately less conservative than
   the shipped box. ✔ §4.3
4. **The change cannot add a stop** — proved by construction and pinned by test.
   ✔ §4.2
5. **Every real-time bound fails conservative** — iteration cap, refinement
   budget, vertex ceiling, overlap fallback. ✔ §4.3
6. **`check_attached_self_collision` is unaffected**, unlike the change #161
   costed. ✔ §6
7. **The recovery claim is the one the evidence supports** — this is a cost and
   correctness change. Against the live map #171 apportions **26 % link-side /
   74 % world-side**, with 48 % of all stops held by the octree→grid
   rasterisation rule (issue #173). ✔ §5.3, §5.4
8. **The real-HAL margin is an open condition.** §3.2 measures 0.82× at
   `world_voxel_margin_m = 0.02` with clutter pressed against the arm.
   `panda_mobile` is sim-only so nothing ships into that regime today, and the
   entry should make re-measurement a precondition for any robot that would.
9. ~~**Issue #155 sequencing is unchanged** — this change regenerates no ACM and
   does not re-lower any manifest.~~ **Superseded by #191**, which regenerates
   the `panda_mobile` / `panda_mobile_vslam` ACM precisely so that it stops
   carrying the exemption. §9.

---

## 9. The self-collision extension (issue #191)

`check_self_collision`'s box↔box pass now re-asks the exact hulls when both
links declare `tight_geometry` (`hull_hull_distance`). The box stays the broad
phase and the fallback: a pair the OBBs already clear never reaches the GJK, and
a manifest declaring no tight geometry is bit-for-bit unchanged.

### 9.1 Why — the pair the boxes cannot answer

`panda_link5`↔`panda_link7` shipped **ACM-exempted** from
[#155](https://github.com/OpenRAL/openral/issues/155) and stayed that way through
[#169](https://github.com/OpenRAL/openral/pull/169), which proved by
branch-and-bound (not sampling) that the pair genuinely interpenetrates and then
changed no manifest byte. Measured over the pair's **entire** relative-DoF
subspace — `panda_joint6` × `panda_joint7`, the only two joints that move it, on
the same 121 × 121 = 14 641 grid #169 used, at `self_collision_margin_m = 0`:

| representation | fires | of which real | false |
|---|---:|---:|---:|
| shipped OBB | 12 647 (86.38 %) | 967 | **11 680 (79.78 %)** |
| exact hull | 967 (6.60 %) | 967 | **0** |

and no margin separates the OBB's populations: the deepest **real** collision
sits at a box gap of −8.37 mm while the shallowest **false** one is at
−36.64 mm. The hull is not merely tighter here, it is exact: the Panda's link5
and link7 collision meshes are convex to 1e-4 in volume ratio
([primitive study §4.3](collision-primitive-study.md)), so `conv(mesh)` **is**
the mesh for this pair and the hull verdict is the mesh verdict.

Tightening the boxes was the retirement path the SRDF comment named, and it does
not reach: #157 §4.3 measures the achievable corner-reach reduction at 7.5 mm on
link5 and 1.9 mm on link7, against the ~28 mm the separation needs.

### 9.2 What the un-exempted pair costs at box fidelity

Simply deleting the ACM row — the issue's Option A — E-stops the robot at its
own reset pose. Measured through the kernel at robosuite's `PandaOmron.init_qpos`:

| model | link5↔link7 gap | verdict |
|---|---:|---|
| shipped OBB | −11.68 mm | **STOP** |
| exact hull | +21.81 mm | clear |

The SRDF's own `ready` and `extended` group states are the same story (−9.14 mm
box, +22.13 mm hull). Its `transport` state is the other half of the case: box
−11.68 mm, hull −5.64 mm, and both agree it is a stop — a real self-collision
the exemption was hiding.

### 9.3 Cost

`check_self_collision` on the shipped `panda_mobile` at its reset pose, `-O3`,
200 000 calls:

| variant | µs/call | vs before |
|---|---:|---:|
| A — before #191 (pair exempt, no hulls) | 0.758 | 1.00× |
| B — after #191 (pair checked, hulls) | 2.303 | **3.04×** |
| C — Option A (pair checked, no hulls) | 0.820 | 1.08× |

The multiplier is large because the routine it sits in is small: +1.545 µs, or
24.7 µs over a 16-step horizon, 0.25 % of a 10 ms budget. It is close to the
sustained cost rather than a spike — this pair's boxes fail to clear on 86.38 %
of its configuration space, so the GJK runs on nearly every call. The next lever,
if a future robot makes that bite, is a DOP-vs-DOP stage 1 in front of the GJK,
not a cheaper support function (§4.4).

### 9.4 The interaction with #188's graded velocity band

`hull_hull_distance` deliberately omits `hull_cell_distance`'s "stop at the first
bound clearing `margin`" exit. There the exit is right — hundreds of cells are
visited and only their minimum is reported. Here the answer **is** the reported
`sweep_min_distance`, and the early exit was measured returning 4.2 mm for a pair
genuinely 60.0 mm apart: sound, but it would crawl the arm past a clear pose.

More consequentially, the self-collision check's `sweep_min_distance` is **no
longer folded into the graded band's slack at all**. A robot's tightest self-pair
is a property of how it is built, not of where it is going: over all 14 641
poses, `panda_link5`↔`panda_link7` never opens past **22.72 mm**, and **zero**
of them clear a 50 mm band, let alone a 100 mm one. Folding it in pinned the
scale at a constant — measured at 0.21 on the live kernel at a 100 mm band with
k = 20 — after which the world term, the one a chunk can actually act on, only
mattered below 22 mm. That is a permanent speed limit, not a slowdown. The
self-collision **latch** is untouched; only its contribution to the velocity band
is dropped, which restores the pre-#188 rate on the self path and leaves the
world path doing what #188 built it for. Pinned by
`LifecycleKernelTest.GradedScalingIgnoresTheRobotsOwnSelfClearance` and
`…TheSelfCollisionLatchSurvivesTheBandExclusion`.

### 9.5 Two consequences worth naming

**The all-zero Panda arm self-collides for real** — 5.65 mm at its own collision
meshes. Two existing kernel tests flew that configuration and only passed because
the pair was exempt; both now seed `panda_joint6` at the SRDF's own `ready`
value. This is the "new stop class" #191 asked to be quantified, and it is a true
positive, not a regression.

**The cuMotion emitter needed its own matrix.** `render_cumotion_config` lowers
each link to a *containing* capsule, and for that geometry the pair really is
always-colliding: it overlaps at **100.00 %** of the same grid, the shallowest by
1.03 mm. Copying the kernel's matrix would have handed cuRobo a constraint that
rejects `ready`. Given the URDF the emitter now re-derives the ACM against the
spheres it actually writes (`sphere_model_geometry` + `acm_for_geometry`), so the
planner's model stays **looser** than the kernel's — never tighter, which is the
only safe direction for a planner (#169). Fixing that surfaced a second defect
the same measurement explains: the emitter was sourcing its spheres from
`LoweredCollisionModel.collision_geometry` — what the lowering tool would
*write*, a PCA capsule for a mesh collision (#157 §8.5) — rather than from the
manifest the kernel checks. It now prefers the manifest.

Symmetrically, `_certified_always_colliding` **withholds** for any pair whose
links both declare `tight_geometry`: it reasons with `shape_distance`, the box,
and `hull_gap >= box_gap` everywhere, so "the boxes always overlap" no longer
implies "the kernel always trips". Certifying on it would grant an ACM entry
that hides a live check. No shipped robot loses an entry to this today — every
`panda_mobile` ACM row is adjacent or SRDF-sourced — so it is a fail-closed
guard, not a behaviour change.

### 9.6 Reproducing §9

```bash
# The grid, the verdicts, and the pair through the real kernel node.
python -m pytest tests/sim/safety/test_kernel_panda_link5_link7.py
python -m pytest packages/openral_safety/test/test_urdf_lowering_always_colliding.py

# The kernel mechanics.
just safety-kernel-build && ./build/openral_safety_kernel/test_collision \
    --gtest_filter='SelfCollisionHull.*'
```

The 14 641-pose sweep and the §9.3 benchmark are one-shot analysis and are not
checked in (§7). The sweep drives `openral_hal.convex_distance` — GJK with a
separating-axis certificate, plus exact SAT on overlap — over MuJoCo FK of
robosuite's Panda at every node of the `(joint6, joint7)` grid, comparing
`kernel_predicates.box_box_distance` on the shipped OBBs against `conv(mesh)` for
the same poses. The benchmark inlines the manifest's boxes, hulls and ACM into a
standalone `main` linked against `collision.cpp` at `-O3`.

---

## See also

* [Tight link geometry in the safety kernel (#161)](collision-tight-geometry.md) — the study this implements, and the recovery ceiling it must not be described as exceeding.
* [Collision-primitive study (#157)](collision-primitive-study.md) — the broad-phase hazard this change is built to avoid touching.
* [RoboCasa start-state census (#159)](robocasa-start-state-census.md) — why the scope is `link1` and `link2`.
* [Collision-stack validation evidence (#160)](collision-validation-evidence.md) — the world-side terms no link geometry reaches.
* `cpp/openral_safety_kernel/README.md` — the kernel's own geometry documentation.
