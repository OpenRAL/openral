# Collision programme — where it stands, and the one measurement that decides it

> Written 2026-09-06 after the #204 A/B (120 runs, q-laptop). Companion to
> [`docs/reference/collision-validation-evidence.md`](docs/reference/collision-validation-evidence.md)
> (the round ledger), [`docs/reference/collision-safety-alternatives-survey.md`](docs/reference/collision-safety-alternatives-survey.md)
> (the 2026-08-30 survey) and [`docs/reference/robocasa-start-state-census.md`](docs/reference/robocasa-start-state-census.md).
>
> **Status, 2026-09-07: landing in three slices, not one.** The work grew to
> 4 822 lines across 33 commits — over the 800-line ceiling in CLAUDE.md §4.2.5,
> and no longer one logical change. It is split by *what gates it*:
>
> | slice | branch | contents | gate |
> | --- | --- | --- | --- |
> | **A** | `fix/collision-instrument-repairs` | four instrument repairs, two evidence producers, the generator's `refine_dop_to_budget`, the ceiling probe, the narrow-phase latency surface, this page and the evidence ledger | none — diagnostics and tools (§1.4); **changes no shipped envelope** |
> | **B** | `feat/216-tight-geometry-link3-4-6` | `tight_geometry` on `panda_link3`/`link4`/`link6`, both manifests | safety-WG + hazard-log Entry 026 (§3) |
> | **C** | — | `panda_link1`'s refined envelope | **withdrawn.** The tool lands in A; the manifest does not, because the battery measured it moving link1's stops by 0.0003 mm |
>
> A is the urgent one: `master` has been carrying a validation harness that
> cannot see the graph it launches and an adjudicator that stamps every stop
> `real-contact`, since 2026-09-05. Every round taken on it is unusable, so
> nothing downstream can be measured until A lands.

---

## 1. What actually cost the failures

120 runs, 60 per arm, on `q-laptop`. 9 completed. 12 hit the deadline (11 of
them never grasped — policy, not kernel). **99 were stopped by the safety
kernel.** For 91 of those the certified mesh gap at the moment of the stop is
recorded in the run's own ground-truth snapshot:

| what was physically there when the kernel stopped | stops |
| --- | ---: |
| real contact (gap ≤ 0) | **6** |
| clear by 0–5 mm | 8 |
| clear by 5–20 mm | 31 |
| clear by 20–50 mm | 30 |
| clear by **more than 50 mm** | 16 |

**Median true clearance at the stop: 20.1 mm.** The kernel's own reported
median depth was −5.3 mm. Six of ninety-one stops were the kernel being right;
**eighty-five stopped a robot that was physically clear, half of them by more
than two centimetres.**

Who trips, and in which phase:

| tripping party | phase | stops |
| --- | --- | ---: |
| **carried payload** | carrying | **56** |
| carried payload | placing | 14 |
| arm link | pre-grasp | 18 |
| arm link | carrying / placing | 11 |

**71 % of all stops are the grasped object hitting the world while being
carried.** The grasp works. The robot then walks the object through the kitchen
and the kernel stops it at 20 mm of air.

Per scene:

| scene | n | success | stops | payload | link | median true gap | place declaration seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baguette` | 30 | 0 | 26 | 21 | 5 | 38.9 mm | 26 |
| `sink_cup` | 30 | 5 | 22 | 12 | 10 | 45.7 mm | 22 |
| `fridge` | 30 | 4 | 23 | 9 | 14 | 36.3 mm | 23 |
| `utensil` | 30 | 0 | 28 | **28** | 0 | 16.2 mm | **0** |

`utensil` is the purest case: 0/30, every stop the payload, and the place
declaration was seen **zero** times — Path B never armed there at all.

### Two things nobody had named

- **One fridge cell, `voxel_352030`, stopped `panda_link6` seven times**, at
  37–63 mm of certified clearance from the nearest real body. `panda_link6`
  accounts for 18 of the 29 link stops. It has **no `tight_geometry`** — only
  links 1, 2, 5 and 7 do — and carries 53.35 mm of corner slop.
- Of the **17 stops against the declared place target itself** (Path B's own
  geometry, #200), the allowance was active for only 9, and 8 came back
  `unadjudicated`. Path B is shipped but half-armed.

---

## 2. The root cause, in one sentence

**The kernel is doing exactly what its geometry says, and its geometry is
20–50 mm too fat for a task that means carrying an object through a kitchen at
10–20 mm clearances.**

A month of PRs made the *measurement* of that fatness certified, exact and
reproducible — that work is sound and worth keeping. None of it changed the
fatness. The survey said as much on 08-30 ("the binding term is the map"),
recommended Paths A, B and C, all three shipped inside a week, and completion
went from 25 % (2026-08-26) to 5–10 % (today).

---

## 3. The number nobody has ever measured

**Nobody has run the policy with the world-voxel gate off.** Not once, in any
ledger entry, census, survey or round. The validation harness *refuses* to —
`no_enable_octomap_kernel_check` is on its `_SAFETY_KNOB_PATTERNS` list
(`tools/validation_matrix.py:163`), which is correct for a validation round and
is exactly why the measurement was never taken.

So after a month we still do not know whether XR-1's ceiling on these four
scenes is 12 % or 60 %.

The survey itself quotes the experiment that shows why this matters — PACS
(arXiv:2511.06385) Table I: **unfiltered policy 0.70, reactive binary filter
0.04.** That is the shape of what this battery is showing, and the survey
quoted it without ever asking for the in-tree equivalent.

---

## 4. The ceiling experiment — the gate on everything else

Run the same four scenes, same seed, same XR-1 checkpoint, with the world-voxel
kernel gate **off**, and measure task success. It is a **ceiling measurement,
not a configuration**: it never lands in a scene file, a launch default or a
manifest, and the harness's refusal stays exactly as it is.

In a simulator this carries no safety cost — nothing can be hurt by a simulated
robot passing through a simulated cabinet.

### The fork

| ceiling | reading | action |
| --- | --- | --- |
| **≈ 10–15 %** | the kernel costs almost nothing; the policy cannot do these tasks | **Close #102, #108, #217** with "kernel correct and conservative by a measured 20–50 mm; completion is policy-bound". Keep the instruments. Stop spending on collision. |
| **≈ 40 %+** | the kernel is throwing away 30+ points of completion | pull the levers in §5, each one round each |

### ANSWERED — 2026-09-07, on `spark`

**The gate costs AT LEAST 29 points of completion.** 88 valid runs, 4 scenes
x 2 arms, 10-12 per cell, same commit (`80027b18`) and host, arms run
simultaneously:

| | valid runs | completed | rate |
| --- | ---: | ---: | ---: |
| world-voxel gate **OFF** | 45 | **14** | **31.1 %** (floor) |
| world-voxel gate **ON** (shipped) | 43 | **1** | **2.3 %** (floor) |

> **Both rates are lower bounds** (issue #256, 2026-09-09). 31 of these 89
> valid runs were killed mid-run by a Nav2 bond teardown — the default 4 s
> heartbeat timeout tearing down the whole navigation stack — and were scored
> `deadline-no-grasp`, a policy failure they were not. Excluding runs the
> policy was never given a fair share of compute puts the arms at **66.7 % vs
> 4.0 %**. **Re-measured on the fixed harness 2026-09-10: 62.5 % vs 2.7 %, a
> 59.8-point gap, `p = 3.1e-08`** — the two agree within 3 points by independent
> routes, so the gap is about **twice** the 29 recorded here. The direction and the decision below are
> unaffected; only the size is, and it moves in the programme's favour. Fixed
> and re-running — see `docs/reference/collision-validation-evidence.md`.

**Fisher p = 3.5e-04**, power 0.97 against this effect. Leave-one-scene-out
confirms no single scene carries it (p = 1.3e-02 … 5.4e-02 worst case).

Per scene, and this is where the actionable structure is:

| scene | gate OFF | gate ON | p | reading |
| --- | ---: | ---: | ---: | --- |
| `utensil` | **7/12 (58 %)** | 0/10 (0 %) | 0.005 | the gate is the *entire* failure |
| `fridge` | **5/11 (45 %)** | 0/10 (0 %) | 0.035 | same |
| `sink_cup` | 2/11 (18 %) | 1/11 (9 %) | 1.0 | mostly policy-bound |
| `baguette` | 0/11 (0 %) | 0/12 (0 %) | 1.0 | **policy-bound; the kernel is irrelevant here** |

**So the answer is the second branch, but not uniformly.** Two scenes are
almost entirely kernel-bound and two are not. `utensil` is the sharpest case
and it lines up exactly with §1: every one of its 28 stops was the carried
payload, and its place declaration was seen **zero** times — so the payload
bounding box is the whole story there, and it is worth 58 points.
`baguette` is 0 % with the gate off, so no amount of collision work will ever
recover it — it should be dropped from the collision programme's scorecard.

**Do not read this as "turn the gate off".** 6 of 91 stops in the #204 battery
were real contact, and the gate-off arm here is a *ceiling*, not a
configuration. The number says how much headroom the §5 levers are competing
for: **at least 29 points** — see the #256 correction above, which re-adjudicates
it to 62.7 and whose re-measurement puts it at 59.8 — concentrated in the
payload class.

---

## 5. If continuing — the levers never pulled

> **Reordered 2026-09-07 after measuring, and the original order was wrong.**
> This section first said "payload as a tight hull" was the top lever because
> the payload is 71 % of stops. Measured against the 120-run battery, that
> mechanism is **not** where the payload error comes from.

### The decomposition that reorders everything

For every stop the battery records both the kernel's reported depth and the
certified mesh gap that was really there. The difference is the kernel's
over-approximation, and it splits cleanly by class:

| stop class | n | median excess over truth | voxel half-diagonal | **excess beyond the voxel term** |
| --- | ---: | ---: | ---: | ---: |
| **payload** | 62 | 20.1 mm | 21.65 mm | **−1.5 mm** |
| **link** | 29 | 54.8 mm | 21.65 mm | **+33.1 mm** |

**The payload primitives are already tight.** `extract_body_primitives` lowers
each collision geom separately — spheres, capsules and boxes exactly, meshes to
their own `geom_aabb` — and only falls back to clustered boxes past 16 geoms.
A payload stop's entire error is the **25 mm voxel grid**; there is essentially
no payload-geometry headroom to recover (−1.5 mm).

**The link envelopes are not tight.** 33 mm of excess survives after the voxel
term, which is the OBB corner slop (`panda_link6`: 53.35 mm).

### The levers, in measured order

| # | lever | targets | headroom | cost |
| --- | --- | --- | --- | --- |
| 1 | **`tight_geometry` on `panda_link6`** (and `link3`/`link4`, which have none) | 18 of 29 link stops, incl. the whole `voxel_352030` class | ~33 mm of the link excess | **one manifest edit**; `tools/generate_tight_geometry.py` exists |
| 2 | **Promote the fixtures the payload passes to modeled geometry** | 51 of 70 payload stops (the `voxel_` ones) | removes the voxel term entirely for those | moderate — generalises ADR-0098/#200 from the *declared place target* to the fixture the payload is near |
| 3 | **Voxel resolution 25 → 15 mm** — **with the WG as #253**; every cost term measured (kernel 0.825 ms, rasterize 1.60 ms, wire +55 ms p99, see the item below) | **class B only** — the 8 of 20 gate-ON stops whose excess a 25 mm cell can account for (≤ 21.65 mm); it does **not** reach class C (9 of 20, excess 23–85 mm, → lever 2 / #254) nor the 3 real contacts | **8.66 mm** of the 21.65 mm half-diagonal; **5 of 20 stops** plausibly clear at 15 mm (3 of class B already sit under 12.99 mm) | one parameter, reversible; staleness cost is probabilistic and priced against carry speed (n=5, now widenable to n=13) |
| 4 | ~~Payload as a tight hull~~ | — | **−1.5 mm: none** | struck; measured out |

### 2026-09-10 — the stop census that makes lever 3 first, not third

Every gate-**ON** stop in the 2026-09-10 re-run, traced from the kernel's own
verdict to the certified mesh distance in the same run's ground-truth snapshot.
This is the measurement that reorders §5.

**What stopped them.** Of 37 valid gate-ON runs, **22 were stopped by the
kernel** and 15 merely ran out of deadline. All 22 are `kind=world`, and

| party the kernel named | stops |
| --- | ---: |
| **the carried payload** (`attached:sim:obj_main`) | **19** |
| `panda_link1` / `link6` / `link7` | 1 each |

Every one reported penetration — median **−6.4 mm**, worst −23.4 mm.

**What was actually there.** Each snapshot carries certified GJK distances from
the stopped body to real collidable geometry. Sorted by how close the contact
really was:

| scene | party the kernel named | kernel depth | **certified real gap** | nearest real surface |
| --- | --- | ---: | ---: | --- |
| `fridge` | `*payload*` | -11.7 mm | **-0.7 mm** | `fridgesidebyside_main_group_1_g25` |
| `sink_cup` | `*payload*` | -11.2 mm | **-0.2 mm** | `island_island_group_top_right_0` |
| `baguette` | `*payload*` | -1.1 mm | **-0.1 mm** | `counter_1_left_group_top_left_1` |
| `fridge` | `*payload*` | -14.8 mm | **+0.0 mm** | `fridgesidebyside_main_group_1_g25` |
| `fridge` | `*payload*` | -12.1 mm | **+0.2 mm** | `fridgesidebyside_main_group_1_g25` |
| `fridge` | `*payload*` | -15.8 mm | **+3.7 mm** | `fridgesidebyside_main_group_1_g25` |
| `utensil` | `*payload*` | -5.1 mm | **+4.4 mm** | `stack_1_right_group_2_door_g1` |
| `utensil` | `*payload*` | -9.6 mm | **+8.5 mm** | `counter_1_right_group_top_0` |
| `sink_cup` | `*payload*` | -0.4 mm | **+9.1 mm** | `island_island_group_top_left_2` |
| `baguette` | `panda_link1` | -15.7 mm | **+9.3 mm** | `obj_g15` |
| `utensil` | `*payload*` | -1.0 mm | **+12.3 mm** | `counter_1_right_group_top_0` |
| `utensil` | `*payload*` | -8.6 mm | **+14.8 mm** | `counter_1_right_group_top_0` |
| `utensil` | `*payload*` | -2.8 mm | **+15.4 mm** | `stack_1_right_group_2_door_g1` |
| `utensil` | `*payload*` | -6.8 mm | **+18.0 mm** | `counter_1_right_group_top_0` |
| `utensil` | `*payload*` | -6.1 mm | **+23.5 mm** | `counter_1_right_group_top_0` |
| `utensil` | `*payload*` | -5.0 mm | **+27.3 mm** | `counter_1_right_group_top_0` |
| `fridge` | `panda_link6` | -6.0 mm | **+43.4 mm** | `fridgesidebyside_main_group_1_g28` |
| `sink_cup` | `*payload*` | -5.9 mm | **+45.0 mm** | `island_island_group_top_front_0` |
| `baguette` | `*payload*` | -8.1 mm | **+48.5 mm** | `counter_1_left_group_top_left_1` |
| `sink_cup` | `panda_link7` | -23.4 mm | **+61.2 mm** | `island_island_group_top_front_1` |
| `baguette` | `*payload*` | -1.9 mm | — | no snapshot |
| `baguette` | `*payload*` | -4.2 mm | — | no snapshot |

**So it is not a contact problem.** Only **3 of 20** were genuinely touching,
and those three graze at −0.7, −0.2 and −0.1 mm. The median stop fires with
**+10.8 mm of real clearance**; six fire with more than 20 mm, the worst being
`panda_link7` stopped at −23.4 mm while **61.2 mm clear** of anything.

**The over-approximation is the voxel cell, not the geometry.** Median excess
of certified gap over reported depth is **18.9 mm** — essentially the 25 mm
cell's half-diagonal, `25·√3/2 = 21.65 mm`. An occupied cell asserts only that
*something is somewhere inside that cube*, so every surface is inflated by up to
that much regardless of how tight the primitives are.

**Which is why the lever order changes.** The stop population is now **86 %
payload** (19 of 22), and lever 4 already measured payload primitives as tight
(**−1.5 mm** beyond the voxel term). Tight link geometry — lever 1, shipped —
can only reach the 3 link stops. **No amount of geometry work can recover the
18.9 mm, because it is not geometry.** Lever 3 is the only one that touches it.

25 → 15 mm takes the half-diagonal from 21.65 mm to **12.99 mm**, recovering
**8.7 mm**. Against the certified-clearance column above that clears the whole
8.5–18 mm band outright — roughly 6 of the 20 stops — and moves others from
penetrating to advisory. It does **not** clear the three real contacts, which
is correct: those are the stops the kernel exists for.

**Correction (same day): the 18.9 mm median is bimodal, and only half of it is
the cell.** Splitting the same 20 adjudicable stops by how much of the excess a
25 mm cell can actually account for:

| class | n | excess over certified truth | which lever reaches it |
| --- | ---: | --- | --- |
| **A — genuinely touching** (real gap ≤ 0) | 3 | 1.0, 10.9, 11.0 mm | **none, correctly** — these must keep stopping |
| **B — quantisation** (clear, excess ≤ 21.65 mm) | 8 | 9.5 … 19.5 mm | **#253**, voxel 25 → 15 mm |
| **C — not explained by the cell** (excess > 21.65 mm) | 9 | **23.4 … 84.6 mm** | **#254 / ADR-0101** — occupancy no nearby real body explains |

Class C is the one that changes the lever order. An excess of 84.6 mm cannot be
a 25 mm cell's half-diagonal under any argument, so **a finer grid does not
reach those stops at all** — that is arithmetic, not judgement. Four of the nine
are `utensil` against `counter_1_right_group_top_0`, the counter the payload was
lifted *from*, which is the stale pre-attach silhouette the attach sweep window
exists to clear; the rest name a surface tens of millimetres from anything the
certified probe could find.

And within class B, three stops already have an excess under **12.99 mm** — a
15 mm grid's own half-diagonal — so they would not clear either. **25 → 15 mm
plausibly converts 5 of 20 stops; class C is 9 of 20.** The two levers are still
complementary, but the claim above that "the over-approximation *is* the cell"
holds for 8 stops and is wrong for 9. `#253` and `#254` carry this split.

*Method note:* a finer cell moves the kernel's reported depth **toward**
certified truth rather than subtracting a fixed amount from it, so a genuinely
touching pair keeps stopping at any resolution. Reasoning by subtraction on the
reported depth gives the wrong answer for class A and was corrected here.

**Which phase the stops happen in — and it is never the pick.** Traced each
stop against the run's own `automatic sim attachment revision` (grasp),
`support_witness_separated` (payload leaves its support) and
`place_declaration_armed` markers:

| phase | stops |
| --- | ---: |
| **PLACING** — a place declaration was armed before the stop | **11** |
| **CARRYING** — lifted clear of support, no declaration yet | **7** |
| grasped but not yet lifted | 1 |
| an arm link, not the payload | 3 |
| approach / before the grasp | **0** |

It splits by scene: `baguette`, `sink_cup` and `fridge` stop **at the place**;
`utensil` (`PickPlaceCounterToDrawer`) stops **in transit** — all 7 of its
payload stops are mid-carry.

**What the placing stops actually hit** is the part that picks the lever:

| what the payload hit | stops | `place_allowance_active` |
| --- | ---: | --- |
| **a world voxel that is *not* the declared target** | **9** | `False` |
| the declared place target | 2 | `False` |

So in 9 of 11 the payload was approaching its target and clipped the
*surrounding* occupancy — the counter top beside the drop point, the shelf next
to it — not the thing it was declared to place onto. The allowance is scoped to
`target_id`, so it correctly does not cover them, and `place_allowance_active`
is `False` on **all 22** stop lines.

**This makes levers 2 and 3 complementary, not competing**, which §5 asserted
and this measures:

- **Lever 2** reaches the **9 placing stops** — the exemption machinery already
  exists there and is simply scoped too narrowly.
- **Lever 3** reaches the **8 carrying stops** as well, because mid-transit
  there is no declaration to widen: nothing to exempt against, only the
  21.65 mm cell inflation to shrink. `utensil` — the scene with the highest
  gate-off ceiling at **9/10** — is *entirely* carrying stops, so lever 3 is
  what unlocks it.

**Open before acting on lever 2:** the two stops that hit the declared target
with the allowance inactive are unexplained — the support witness is the other
half of what arms it, and whether it had dropped there has not been checked.

**What is still owed on lever 3 — corrected 2026-09-10 on unification with the
resolution branch.** The octree→grid conversion is **not** unmeasured: the lever-3
item below records it at **1.60 ms at 15 mm** (`RasterizationCostAcrossTreeResolutions`),
together with the kernel (0.825 ms) and the wire (+55 ms p99, the term that bites).
What remains is exactly what #253 lists: (1) a **live 15 mm A/B** — the ceiling
battery is now a working instrument to run it in, `WORKERS=2`; (2) widening the
carry-speed sample from n=5 — this battery adds 8 carrying-phase stops; (3) nothing
on hardware. The `experiment/voxel-resolution-15mm-v2` branch is PR #252, rebased
onto this plan on 2026-09-10 with both records unified here.

**Caveat.** 20 of the 22 stops carry a snapshot; two `baguette` stops do not.
And these are gate-ON runs from the battery whose opening lanes lost 11 runs to
a concurrent GPU job — the stop census is unaffected by that (a stopped run is a
stop regardless), but the per-scene counts inherit the same thin `baguette`.

### 2026-09-10 — the ceiling itself is depressed by self-collision false positives

The world-voxel check is **off** in the gate-OFF arm, so any kernel stop there is
the envelope or self-collision path — no voxel, no quantisation term, nothing
lever 2 or lever 3 can touch. Seven of the 32 valid gate-OFF runs carry one;
five of those runs failed. Traced to the certified distance **for the exact pair
the kernel named**, out of the same run's ground-truth snapshot:

| scene | kernel pair | kernel depth | **certified, that pair** | run |
| --- | --- | ---: | ---: | --- |
| `sink_cup` | `link5` ↔ `link7` | −26.6 mm | **+3.5 mm** | failed |
| `sink_cup` | `link5` ↔ `link7` | −32.2 mm | **+1.4 mm** | failed |
| `sink_cup` | `link5` ↔ `link7` | −29.6 mm | **+4.9 mm** | failed |
| `utensil` | `link1` ↔ `link7` | −1.7 mm | **+87.9 mm** | completed |
| `baguette` | *payload* ↔ `link1` | −1.6 mm | **+69.2 mm** | failed |
| `baguette` | *payload* ↔ `link1` | −0.2 mm | **+25.2 mm** | failed |
| `fridge` | `link1` ↔ `link6` | −2.8 mm | not in snapshot | completed |

**Every adjudicable one is a false positive.** The `link5`↔`link7` trio reports
~30 mm of penetration on a pair that is 1–5 mm *clear* — a ~30 mm
over-approximation with **no voxel in the path at all**. That is the pair whose
ACM exemption #191 retired as "proven real", and all seven links ship stage-2
`tight_geometry` on `master`; so either the hull for `link5`/`link7` is not
tight where these configurations put it, or the retirement was premature. This
plan does not decide which. The payload↔`link1` pair is the
`check_attached_self_collision` path and is 25–70 mm out.

**Why this is a lever and where it sits.** These stops are *inside the gate-OFF
ceiling* — they cost completions with the world check off — so fixing them
raises the **62.5 %** the whole programme is measured against, and they recur
in gate-ON (the 3 link stops there). It is the cheapest thing on this page:
hull fidelity on two named links and one named pair, adjudicable from artifacts
already on disk. **It has no issue.** Filing one is the first item under "What
to test next".

*Probe caveat, so nobody reads a number wrongly:* the snapshot's nearest
link↔link pair is `link5`↔`link6` at −23 to −28 mm in every one of these runs.
That is an **adjacent** pair overlapping at its joint by construction; the
kernel never fired on it in 80 runs, so it is handled and is not a missed stop.
The pair-specific rows above are the ones that mean something.

### Resolution was struck on an estimate, and the estimate was wrong (lever 3)

**This was the one lever struck on paper rather than by test, and it does not
survive being tested.** The original argument: `OccupancyVoxels.occupancy` is a
dense `uint8[]` and the kernel's per-link window holds `O(1/res³)` cells, so
halving the cell is an 8× check cost, giving 26.7 ms at 15 mm against a 33 ms
ceiling. The table below is what it predicted, and what the kernel actually does.

Measured 2026-09-07 on `q-laptop`: the real kernel binary, the real
`panda_mobile` manifest (all seven links lowering their tight geometry), the
`layout_ids: [47]` RoboCasa kitchen rasterised cell by cell at each resolution
over the **same volume**, `world_voxel_margin_m = 0.0`, 200 chunks per point,
end-to-end `/openral/candidate_action` → `/openral/safe_action`:

| resolution | grid cells | occupied | median | **measured p99** | *estimated* | error term |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **25 mm (today)** | 82 368 | 9 891 | 0.104 ms | **0.517 ms** | *5.8 ms* | 21.65 mm |
| 20 mm | 158 400 | 17 321 | 0.166 ms | **0.597 ms** | *11.3 ms* | 17.32 mm |
| **15 mm** | 376 680 | 35 828 | 0.184 ms | **0.825 ms** | *26.7 ms* | 12.99 mm |
| 12.5 mm | 630 054 | 59 948 | 0.162 ms | **0.838 ms** | *46.1 ms* | 10.83 mm |

The estimate is wrong by 11× at the baseline and **55× at 12.5 mm**. Two
independent errors compounded:

1. **The 5.8 ms baseline was never the kernel.** It came from the shipped hull
   benchmark (`collision-hull-narrow-phase.md` §4), a microbenchmark of the
   narrow phase, not a round trip through the kernel under a real grid. There
   was no latency surface to take that number from until one was built on
   2026-09-07 — the same day the strike was written. Measured, the baseline is
   **0.517 ms**.
2. **The cubic factor was applied to the wrong term.** `collision.cpp:1541`'s
   window loop is `if (grid.occupancy[idx] == 0) { continue; }` as its first
   statement. The cubic growth therefore lands on an array load and a
   branch-not-taken; the expensive work — the support-exemption test and the
   staged 26-DOP → hull distance — runs only on **occupied** cells, and
   occupancy is a *surface*. Measured: 25 → 12.5 mm multiplies total cells by
   7.65× and occupied cells by 6.06×, but p99 by **1.62×**.

**The cap objection also fails on its own numbers.** 15 mm is 376 680 cells,
*under* the shipped `world_voxel_max_cells = 614 125` — no cap change at all.
Only 12.5 mm exceeds it (630 054), and that cap is a `voxel_occupancy_.assign()`
at `on_configure` (`lifecycle_kernel.cpp:1655`), so raising it costs 0.63 MB of
pre-allocated memory, not hot-loop time. The plan's original "2 803 221 cells at
15 mm" was a **whole-kitchen** grid; the kernel scans an arm-neighbourhood
window, which is a different volume.

**What it buys, and what it does not.** The error term is the cell half-diagonal,
so 25 → 15 mm recovers **8.7 mm** and 25 → 12.5 mm recovers **10.8 mm**. Against
a 20.1 mm median true clearance at the stop that is roughly half the payload
class, and it applies to the link class and the start-state class equally —
it is the only lever that touches all three. It is *not* on its own sufficient:
the two clear start-state stops sit at +23.13 mm and +22.01 mm, beyond what even
12.5 mm recovers. It composes with modeled fixtures (lever 2) rather than
competing: ADR-0101 removes the voxel term entirely for the fixtures it models,
and this shrinks it for everything else.

**The limits this measurement had, and what closed them (2026-09-08).** It
measured the kernel *consuming* a grid, and named two terms it could not see.
Both are now measured:

* **the bridge producing one** —
  `test_octree_to_grid.cpp::RasterizationCostAcrossTreeResolutions`: **1.60 ms**
  at 15 mm against a 100 ms publish period, from 0.88 ms at 25 mm. Nearly flat,
  because the marking loop iterates occupied leaves, a *surface*. 12.5 mm is
  refused outright by the bridge's own `kMaxCells = 4 000 000` guard;
* **the message on the wire** — `tools/voxel_transport_probe.py`: nothing
  dropped and the rate held at every size, but publish→receive latency triples,
  p99 **19-23 ms → 68-83 ms**. That is map staleness, and it is the term that
  bites.

The staleness is settled against measured arm speed rather than assumed:
carry-phase stops run 0.051 m/s median and 0.265 m/s max, start-state stops
exactly 0, so the trade is **net positive in three of four corners** and free for
the whole start-state class (`tools/stop_ee_speed.py`).

It remains one pose in one layout — the window is sized by where the links are —
and the carry-phase speeds come from n=5 stops of *this* policy. Those are the
two limits that stand.

Reproduce with the shipped test:

```
OPENRAL_FRIDGE_GRID_RES_M=0.015 uv run pytest -m sim \
  tests/sim/safety/test_kernel_fridge_layout_pin_start_state.py \
  -k narrow_phase_meets_the_chunk_budget
```

### Why modeled fixtures is the lever (lever 2)

51 of 70 payload stops are against anonymous `voxel_` cells, and the certified
nearest real body at those stops is almost always a **static kitchen fixture
whose exact geometry MuJoCo already has**:

| nearest body at a payload stop | stops |
| --- | ---: |
| `counter_1_right` | 25 |
| `fridgesidebyside_main` | 9 |
| `counter_1_left` | 8 |
| `cab_1_left` | 5 |
| `island` | 4 |

The robot carries an object over a counter and the counter's 25 mm cubes stop
it at 20 mm of air. This is exactly the survey's §9 point 1 — *"the route to mm
world-side discrimination is not a better checker but a better world model —
objects the robot intends to touch promoted from anonymous voxels to posed
meshes/primitives"* — and #200 already built the machinery for the **declared
place target**. Generalising it from "the target" to "the fixture the payload is
near" is the measured next step.

**Deliberately not leading with a negative world margin.** It is zero
engineering and there is now a paper trail for it, but it also lets the 6 real
contacts through, and it is precisely the "padding knob" the survey's §11 says
the community has failed to tune since 2013.

---

## 6. The survey, reviewed against what happened

The [2026-08-30 survey](docs/reference/collision-safety-alternatives-survey.md)
was **right on the diagnosis** — voxel wall, map term dominates, binary stop is
the wrong policy shape — and it was genuinely actionable: all three paths
shipped within a week of it landing. Four things it got wrong, all of them now
measured rather than argued:

1. **It never asked for the unfiltered baseline**, while quoting PACS's. That
   single omission is what let a month pass without the ceiling number.
2. **Path A's prediction did not reproduce.** PACS says graded slowdown
   converts stops into completions (0.04 → 0.72). Measured here: the robot
   travels 45 % further and completion does not move. The band was live in
   #209 and is live in both arms of the #204 A/B.
3. **"The mm problem is confined to the object the robot intends to touch"** —
   wrong. It is the object the robot *carries*, against everything it passes.
   Path B modelled the *target*; 56 stops are the payload against the rest of
   the kitchen. The survey treated the payload as solved by ADR-0092's
   semantics, and the payload is a single bounding box.
4. **"Link envelopes: done, stop"** was measured on start-state stops, which
   are links 1–2. Carry-phase stops are `panda_link6`, which has no tight
   geometry. The verdict was right for the data it had and wrong for the data
   that turned out to matter.

Path C, sequenced last, is now blocked on hardware force calibration — dead in
sim by its own design (#218).

---

## 7. Status

### The ceiling run, as launched (2026-09-06, `spark`)

Running now. Worktree `~/workspace/openral-217-with204` @ `80027b18`
(`1ebe71a` = #204, plus the state-adapter race fix), **both arms on the same
commit and host**, 8 workers in parallel — 4 scenes x 2 arms, 10 rounds each,
**40 runs per arm**. Tools: `tools/_ceiling_probe.py` + `tools/ceiling_battery.sh`.

Both arms run *simultaneously* by design: CPU contention and drift then load
onto both equally and cannot masquerade as an effect. The gate is the only
difference — verified in the launch argv as
`enable_octomap_kernel_check:=false`.

Four things had to be discovered to make it run at all, each worth keeping:

1. **`deploy sim` cannot run concurrently with itself.** Every launch calls
   `_kill_orphan_openral_graph_processes()`, which matches by argv signature
   and **cannot tell a concurrent sibling from a crashed orphan** — so the
   second parallel worker kills the first (measured: worker 1 never got its
   action server, 626 s to timeout). Gated behind `OPENRAL_SKIP_ORPHAN_REAP=1`,
   uncommitted and spark-local. Each worker has its own `ROS_DOMAIN_ID`, hence
   its own fastrtps SHM port, so the reap is not what was protecting them.

   **Corrected 2026-09-10:** that env var was read by nothing, in any commit —
   the battery believed it had opted out and had not, so every worker's
   startup sweep was killing its siblings for the whole life of the parallel
   design. It is honoured now. The same investigation found the sweep's needle
   set missing the `octomap_voxel_bridge` / `octomap_server_node` pair, which
   `ros2 launch` also starts in its own session: the only graph members that
   neither the sweep nor a `killpg` could reach. They accumulated one pair per
   round — 46 alive on q-laptop, oldest 23.7 h — and each holds the
   `fastrtps_port<N>_el` lock file that makes the *next* run on that domain
   fail `open_and_lock_file` and receive 0 action chunks. So the premise above
   is half wrong: a per-worker `ROS_DOMAIN_ID` does **not** protect a worker,
   because the poison is left behind on its own domain by its own prior round.
2. **The XR-1 sidecar is stateless and can be shared.** `_xr1_server.py`'s
   `reset()` is literally `return`; all episode state (history deques, action
   queue) is client-side in `openral_sim/policies/xr1.py`. One sidecar serves
   all 8 workers; ZMQ REP serialises and replies to the sender, so episodes
   cannot cross-contaminate.
3. **Racing to spawn that sidecar is probably a known failure in disguise.** A
   second client that pings before the model has loaded spawns its own on the
   same port, loses the bind, and dies with *"xr1 sidecar process exited with
   code 1 during boot"* — the exact error that cost runs on q-laptop. The first
   worker now gets a 210 s head start.
4. **Per-worker rSkill copies do not work** for pinning distinct ports: the
   runner node resolves through `rSkill.from_pretrained`, whose repo root
   differs from the launching shell's, so both path- and hub-shaped copy ids
   404'd against the real Hub and produced policy-free runs in ~35 s.

- [x] **Ceiling experiment** — done, 2026-09-07. 31.1 % vs 2.3 %, p = 3.5e-04.
      Result in §4. **Both absolutes are floors** (#256): a Nav2 bond teardown
      voided 31 of the 89 runs and they were scored as policy failures.
      Corrected reading 66.7 % vs 4.0 %; **re-measured 2026-09-10 at 62.5 % vs
      2.7 %, a 59.8-point gap** (`p = 3.1e-08`).
- [x] **Close-vs-continue** — **continue.** The gate is worth at least 29
      points of completion — **measured at 59.8 on 2026-09-10** — so the
      §5 levers are competing for real headroom rather than for noise.
- [x] ~~**Lever 1: the payload bounding box**~~ — **struck by measurement,
      2026-09-07.** This entry predates §5's reordering and kept the old
      numbering. The payload *is* 71 % of stops, but the decomposition shows its
      primitives are already tight to **−1.5 mm** beyond the voxel term: there is
      no payload-geometry headroom to recover. `extract_body_primitives` lowers
      each collision geom separately, so the payload's entire error is the 25 mm
      grid. The class is still the right target — but the mechanism that reaches
      it is **modeled fixtures (ADR-0101)**, not a tighter payload box. Tracked
      there, not here.
- [x] **Lever 1: `tight_geometry` on `panda_link3`/`link4`/`link6`** — built and
      measured 2026-09-07; **on `feat/216-tight-geometry-link3-4-6`, not on
      `master`**, pending safety-WG sign-off on hazard-log Entry 026 (§3). Support excess 75.6→23.8, 76.1→23.2 and **52.7→21.5 mm**;
      `link6`'s 31.2 mm recovery is almost exactly the 33.1 mm of measured
      link-class excess. Applied to `panda_mobile` **and** `panda_mobile_vslam`
      (their arm geometry is contract-tested identical). `generate_tight_geometry
      check` reports mesh-outside-DOP **0.000000000 mm** on all seven links, so
      containment stays definitional. Both manifests are `hal.real: null`, so the
      sim-margin benchmark applies, where the staged path is 1.04-1.09x *faster*
      than the box path. Reverses the exclusion recorded in
      `docs/reference/collision-hull-narrow-phase.md` §5.2, whose stated reason
      ("zero of the 72 census stops") was true of **start states** and wrong for
      the carry phase.

      **Gap found while validating it:** `tests/sim/safety/test_kernel_latency_soak.py`
      never publishes `OccupancyVoxels` and runs a synthetic `soak_test` envelope
      with no collision geometry, so it **cannot reach the narrow phase** — its
      pass is vacuous for any hull change. The kernel's dominant cost path has no
      latency surface. Same class as the #183 vacuous Nav2 test.
- [x] **Give the narrow phase a latency surface** — done 2026-09-07.
      `test_the_narrow_phase_meets_the_chunk_budget_on_a_real_grid` lives in the
      fridge pin file, the only place in the tree with a real grid (a real
      RoboCasa kitchen rasterised cell by cell, the real manifest so all seven
      links lower, the real kernel binary, at the sim margin of 0.0 m).
      **Measured: p99 2.0 ms, median 0.1 ms over 5 638 occupied cells** — 15x
      under the 33 ms 30 Hz ceiling, with the three new hulls live. Mutation-
      checked by forcing the budget to 0.001 ms to read the real numbers out.
      That also answers the latency question the `link3`/`link4`/`link6` change
      raised, on the shipped configuration rather than by extrapolation.
- [ ] **Lever 3: voxel resolution 25 -> 15 mm** — **now with the WG,
      2026-09-09: hazard-log Entry 027** (`OpenRAL/management#36`), tracked as
      **#253**. Every cost
      term measured, the asymmetry named (deterministic 8.66 mm gain against a
      probabilistic staleness cost), and the one thing that does not exist stated
      in the record: no live A/B of completion rate, which the WG may reasonably
      require first. Nothing further is mine until it is ruled on. — **un-struck 2026-09-07, and
      the strike was mine.** It was the one lever struck on an *estimate* rather
      than a measurement, and measuring it moved the number by **32×**: p99
      **0.825 ms** at 15 mm, not the estimated 26.7 ms, against a 33 ms ceiling
      (§5 for the full curve and both errors — a baseline that was never the
      kernel, and a cubic factor applied to a branch-not-taken). 15 mm needs no
      change to `world_voxel_max_cells` either; it is 376 680 cells against the
      shipped 614 125.

      Recovers **8.7 mm** of the 21.65 mm quantisation term, in *every* stop
      class — the only lever that touches payload, link and start-state alike.
      Not sufficient alone (the two clear start-state stops are +22 to +23 mm)
      and it composes with lever 2 rather than competing.

      **Producer measured 2026-09-08 — it is not the obstacle either.**
      `test_octree_to_grid.cpp::RasterizationCostAcrossTreeResolutions`, real
      octree, real rasterizer: **1.60 ms at 15 mm** against a 100 ms publish
      period, from 0.88 ms at 25 mm. Nearly flat, for the same reason the
      consumer was — the marking loop iterates occupied leaves, a *surface*,
      while only the dense buffer scales with volume. **12.5 mm is refused
      outright** by `octree_to_grid.cpp`'s `kMaxCells = 4 000 000` guard, so it
      needs that raised; 15 mm does not.

      **And a correction to this item, which was also mine.** The un-strike
      claimed 15 mm "needs no change to `world_voxel_max_cells`; it is 376 680
      cells", dismissing §5's 2.8 M as a whole-kitchen grid. Both halves wrong:
      the ball is sized by the arm's reach (`_octomap_coverage_radius` measures
      1016 mm, ships 1.05 m), so 15 mm needs **141³ = 2 803 221** cells and §5's
      original figure was right. The cap consequence is real — a kernel still
      reserving 614 125 rejects every grid, which reads as "no world" and is a
      **fail-open on the world check**. `_world_voxel_max_cells` now derives the
      cap from the resolution, which is what makes the lever safe to pull.

      **Wire measured 2026-09-08 — and it is the term that bites.**
      `tools/voxel_transport_probe.py`, two processes over real DDS at the
      deployed 10 Hz under the kernel's own QoS: nothing is dropped and the rate
      holds at every size, but publish→receive latency triples, from p99
      **19-23 ms** at 25 mm to **68-83 ms** at 15 mm. That latency is map
      **staleness**, and staleness is millimetres too.

      **Settled 2026-09-08, and in favour of the lever.**
      `tools/stop_ee_speed.py` measures how fast the arm actually is at each
      stop, from the recorded joint state through the real Panda Jacobian:
      carry-phase **0.051 m/s median, 0.265 m/s max** (n=5); start-state
      **exactly 0.000** (n=7, the arm has not moved yet). Base contributes
      ≤0.0013 m/s.

      | | median stop | fastest stop |
      | --- | ---: | ---: |
      | median staleness | **+7.89 mm** | **+4.68 mm** |
      | p99 staleness | **+5.84 mm** | **−5.93 mm** |

      Net positive in three of four corners, and the whole start-state class —
      43 % of stops — takes the full 8.66 mm with **no** staleness cost at all.
      All three cost terms are now measured: kernel 0.825 ms, rasterize 1.60 ms,
      wire +15/+55 ms.

      **Actionable, with two caveats to carry into the change.** n=5 carry-phase
      stops is thin, and the speeds are *this* policy's — a faster one moves the
      p99 corner from marginal to routine. Widening n is the thing to do before
      the manifest edit, not instead of it.

      **Reach, bounded 2026-09-10 (#258).** "Recovers 8.7 mm in every stop
      class" is true of the *term* and not of the *stops*: tracing the 20
      adjudicable gate-ON stops to certified truth, only **8** have an excess a
      25 mm cell can account for, and 3 of those already sit under a 15 mm
      cell's 12.99 mm. **25 → 15 mm plausibly converts 5 of 20.** The other 9
      have an excess of 23–85 mm that no cell explains — that population is
      #254's, and this lever cannot touch it. The three real contacts survive
      at any resolution (a finer cell moves reported depth *toward* truth; it
      does not subtract from it).
- [ ] ~~**Drop `baguette` from the collision scorecard**~~ — **withdrawn
      2026-09-10.** It completed **1/3** with the gate off on the fixed harness,
      so the 0/11 behind this was partly starved runs, not a policy ceiling.
      Weak, not dead; keep it until it has ten valid rounds. Original reasoning,
      recorded 2026-09-07 in
      the ceiling entry of `docs/reference/collision-validation-evidence.md`:
      0/11 with the gate **off**, so it is policy-bound and cannot report on
      collision work in either direction. Four of the five task completions in
      the ledger's whole history were baguette runs, which is what made it look
      like the bellwether scene; at a 0 % ceiling it is not one. It **stays in
      the matrix** — it still exercises the launch, the attach sweep and the stop
      path, and still yields usable stop records — it just leaves the
      *completion* scorecard. `utensil` (58 % ceiling, 0 % shipped) carries that
      signal instead.
- [x] Landed in `docs/reference/collision-validation-evidence.md`; commented on
      #102, #108 and #217.
- [x] **Hazard-log Entry 026** — the record #235 owes, written 2026-09-07
      (`OpenRAL/management#33`). Containment argued and verified, Entry 018's
      exclusion falsified, latency measured, three open items named. **WG
      sign-off still PENDING** — that is a human ruling and is not faked.
- [x] **ADR-0101 — modeled static fixtures for the carried payload**, proposed
      and deliberately *not* implemented (`OpenRAL/management#33`). Recorded
      before code because it crosses Layer 2 → Layer 6 (CLAUDE.md §3), and
      because the last three levers were each cheaper to argue than to build —
      two were struck by measurement after this plan had committed to them.
      It states plainly that its suppression step is the **first fail-open
      mechanism** in the hazard log, proposes four bounds, and offers a
      conservative first landing: ship the model as observability with
      suppression **off** and measure how often it *would* have explained a stop
      before giving up any protection.
- [x] **Decide #217** — **done: closed COMPLETED 2026-09-06**, before this item
      was written, which is why it lingered here. The #204 A/B ran at the size
      the power analysis specified (60 runs per arm, alternating rounds) and came
      back **null with teeth**: success 5.0 % vs 10.0 %, Fisher p = 0.491, at
      0.85 power against a 25 %→5 % effect. #204 converted an *instrument* off
      `mj_geomDistance` and never had a mechanism by which it would move
      completions; it is off the suspect list. The closing comment carries the
      full table.

- [x] **Quantified ADR-0101's recovery offline** — 48 of 51 payload-vs-`voxel_`
      stops (94 %) would be recovered, median true clearance 16.2 mm; the 3 that
      correctly still stop are real penetration (−0.25, −2.02, −2.76 mm).
      Minimum recovered clearance is **0.1 mm**, which is the number that argues
      for the suppression-off first landing. Folded into the ADR, so the WG
      rules on a measured proposal rather than an unknown.
- [x] **Fixed the instrument that adjudicates every stop.** Chasing the ADR's
      premise against the *live* map (not certified ground truth) found that
      `voxel_backing_record` stopped at the first `mj_ray` strike — so a
      non-collidable shell in front of the slab it wraps was the only thing it
      saw, and the cell was blamed on decoration. **6 of the 8 stops in the
      2026-09-06 battery that carried a backing record at all** were
      misattributed this way, naming `counter_1_right_group_top_visual` while
      the collision surface sat ~16 mm inside the same cell. The ray now walks
      past decoration within the cell. Diagnostics only; mutation-checked.
- [x] **Fixed the harness itself — it could not see the graph it launched, and
      had not since #231.** Every scene of every `validation_matrix` round on
      post-#231 `master` reported `harness-error` ("action server never
      appeared") beside a healthy graph. Two independent defects, each
      sufficient alone: (a) `_launch_env` never applied the sim DDS scope, so the
      graph came up on domain 77 and the harness polled domain 0; (b) `ros2
      action list --no-daemon` cannot discover an advertised action at all — its
      one-shot node's discovery window is too short — so the poll could never
      succeed on *any* scope. Both measured against a live round, both fixed;
      the round that had failed twice then completed with a real outcome
      (`utensil`, `deadline-no-grasp`) on the first attempt.

      **This is the reason the post-#231 rounds looked like launch failures.**
      It also sets the precondition for everything still open below: no further
      number on this page can be taken until this fix is on the branch the round
      runs from. The ceiling battery's `80027b18` arm predates #231 and is
      unaffected — checked, not assumed.
- [x] **Fixed the launch parser's interpreter** (separate branch,
      `fix/deploy-run-jetson-bringup`). `/opt/ros/<distro>/bin/ros2` carries a
      `#!/usr/bin/python3` shebang, so `ros2 launch` parsed `deploy_e2e.launch.py`
      under the *system* interpreter; `PYTHONPATH` only prepends, so anything the
      venv lacks still resolved out of `dist-packages`. On a Jetson AGX Thor with
      `python3-pandas` that aborted the whole launch with `ValueError:
      numpy.dtype size changed` via `lerobot` → `deepdiff` → `import pandas`
      (deepdiff guards that import with `except ImportError`, which a
      `ValueError` sails through). Running the parser under `sys.executable`
      makes the venv's `include-system-site-packages = false` apply and closes
      the whole apt-shadowing class. **This is the `spark`-side launch failure,
      distinct from the harness one above.**
- [x] **Re-derive ADR-0101's 94 % from post-fix live-map rounds — done
      2026-09-09, on `spark`.** 36 `utensil` rounds at `448818c4` (both
      backing-probe fixes), which is a condition no `q-laptop` round had ever
      met. **24 of 28 payload stops recovered (86 %)** against the offline 48/51
      (94 %): Fisher two-sided **p = 0.237**, so the rate survives its own
      re-derivation.

      **What did not survive is the clearance distribution.** Median recovered
      clearance is **8.56 mm, against the offline 16.2 mm** — half. The mechanism
      would be suppressing cells whose true surface is twice as close as the
      offline figure implied, which shrinks the margin its own modelling error
      has to fit inside. Two of the four that correctly still stop are within
      0.1 mm of the surface. **The WG should rule on 8.56 mm, not 16.2 mm** — and
      that argues harder for the ADR's own suppression-off first landing.

      Side effect worth keeping: the stop decomposition now rests on **n = 28**
      rather than n = 4 — payload **−9.93 mm** beyond the voxel term, link
      **+4.28 mm**. The exhaustion conclusion holds harder than when drawn.

- [x] **Re-derived the false-positive rate on a repaired instrument — it is
      71 %, unchanged.** Thirteen rounds, `adr0101-live-*`, 2026-09-07. Seven
      stops: five of a physically clear robot (+0.67 … +24.86 mm), two real
      contact (−2.32, −0.11 mm). Reproduces the #204 battery's 85-of-91 on an
      independent battery and a different commit. **The programme's headline
      number has not moved**, and it is now measured with an instrument that was
      itself broken for two days (below).
- [x] **Found and fixed the adjudicator inversion (#220 regression).** The
      link-vs-link probe #220 added was folded into `nearest_any`, whose first
      rule is "any probed pair ≤ 0 m → `real-contact`". Adjacent links overlap
      permanently and are ACM-permitted, so **every** adjudicable stop was
      stamped `real-contact` from 2026-09-05. On `master`. Three of this
      battery's rounds flipped `real-contact → within-quantization` on
      re-derivation. The error runs one way — it manufactures real contacts,
      never clears one — so nothing was wrongly passed as safe. Standing caveat
      11; two mutation-checked tests.
- [x] **Resolved the probe contradiction — it was the instrument, a third time.**
      The tripping cell spans `z ∈ [0.90005, 0.92505]` and the collidable chunk
      `counter_1_right_group_top_0` has its surface at `z = 0.920`: the solid
      geometry was inside the cell all along. `c7bd2c7`'s walk-past fix handles
      decoration *in front of* a slab but not decoration **coincident** with it,
      which is how RoboCasa builds every counter top. Fixed with an AABB overlap
      sweep, consulted only when the rays find nothing solid. Diagnostics only —
      the certified probe never used rays, so the 71 % and the decomposition
      stand; what moves is the backing *class* the "32 % too sparse" entry and
      ADR-0101's "cells no real body explains" premise rest on.
- [x] **`panda_link1`'s budget-fitting envelope does nothing, and that IS the
      result — so it is not shipped.** The generator routine landed on
      `fix/collision-instrument-repairs`; the manifest change was withdrawn.
      What follows is the measurement that withdrew it. `refine_dop_to_budget` intersects the DOP with the
      exact hull's tangent face planes inside the existing 320-vertex budget, no
      kernel change; support gap 4.52/25.68 mm → **0.18/0.65 mm**, containment
      definitional, and it measured *faster* (p99 0.5 ms on 9891 cells vs 2.0 ms
      on 5638).

      **Then it was tested, and the prediction failed.** Three rounds on `spark`
      — same seeds, same scene, one commit apart:

      | seed | 26-DOP | refined | Δ |
      | --- | ---: | ---: | ---: |
      | s2 | −2.37794 mm | −2.37825 mm | **0.0003 mm** |
      | s4 | −8.31 mm | −8.31495 mm | ~0 |

      Neither `panda_link1` start-state stop clears. **My reasoning was wrong
      twice over**: first striking the lever for the wrong reason, then
      un-striking it for another wrong one. The census's "10 mm clears 14/14" is
      computed against the **manifest OBB**, and the 26-DOP shipped later had
      already collected that recovery (53.27 → 25.69 mm) — I read an
      OBB-relative deficit as headroom still available after the DOP.

      Hazard-log Entry 026's amendment now says the justification is withdrawn
      and asks the WG to rule on a change that is safe, free and unproven, with
      the recommendation that reverting the manifest while keeping the generator
      tool is the cheaper option.

- [x] **Start-state investigated end to end — it is quantisation, same as the
      payload class.** Three stops, read map-side for the first time. Two
      candidate root causes were raised and both were refuted by measurement:

      * **map inflation** — refuted. `utensil-s2`'s tripping cell *contains* the
        true nearest surface point (0.00 mm from the cell box), so the cell is
        where the world is.
      * **self-occupancy** — refuted. `fridge-s2` looked like the robot in its
        own map (15/27 rays on `robot0_link2_collision`, no world geom). Widening
        the backing sweep to run whenever the rays found no collidable *world*
        geometry, then re-running on `spark`, found the **fridge drawer** in that
        cell — the same body its near-miss pair already named at +0.673 mm. The
        rays had missed it.
      * **link envelope** — refuted separately: the `panda_link1` envelope moved
        these same stops by 0.0003 mm.

      | stop | link | true clearance | reported |
      | --- | --- | ---: | ---: |
      | `utensil-s2` | `panda_link1` | +23.13 mm | −2.38 mm |
      | `utensil-s4` | `panda_link1` | +22.01 mm | −8.31 mm |
      | `fridge-s2` | `panda_link2` | **+0.67 mm** | −21.98 mm |

      Two are stops of a demonstrably clear robot; `fridge-s2` at 0.67 mm is a
      genuine near-contact and is arguably a correct stop. **The class has the
      same single lever as the payload class**, which is the real conclusion:
      ADR-0101 is scoped to the carried payload, and extending it to bare links
      would cover both with one mechanism. A scope note for the WG, not a
      decision — the fail-open suppression step still needs ruling on either way.
- [x] ~~**NEW: the residual may be map inflation, not geometry.**~~ **Struck the
      same day it was raised** — the tripping cells contain the true surface
      point, so they are not displaced toward the sensor.
- [x] ~~**Half of these scenes never reach the kernel**~~ — **measured, then
      refuted 2026-09-09, and the first measurement was mine.** Across 26
      `q-laptop` rounds the carry-phase yield was 18-27 % with 40-64 %
      `deadline-no-grasp`, and that was recorded as a property of the *policy*.
      Thirty-six rounds on `spark` produced **zero** `deadline-no-grasp` and
      **78 %** payload stops. Same policy, different host: the figure was a 420 s
      deadline meeting a machine at load 19 with a shared GPU. **The mechanism
      was found in #257:** Nav2's 4 s bond timeout tearing the whole nav stack
      down, silently — not load per se, but a threshold load trips.

      So there is no scene-selection problem to fix here, and the battery-sizing
      advice derived from it was wrong for an idle host. What stands is narrower
      and more useful: **`deadline-no-grasp` is a load symptom first**, which
      also means the ceiling battery's own policy-free exclusions deserve
      re-reading in that light.

- [ ] **Implement ADR-0101** once ruled on — **the ruling now has the live-map
      numbers it was missing, 2026-09-09** (`OpenRAL/management#36`, tracked as
      **#254**): 86 % on
      n=28, Fisher p=0.237 against the offline 94 %, but median recovered
      clearance **8.56 mm against 16.2 mm**. The WG should rule on 8.56 mm; it
      strengthens the ADR's own suppression-off first landing. Original text: — the one lever with headroom left.
      Note the fix above changes what the *live-map* evidence will say, so the
      ADR's 94 % should be re-derived from post-fix rounds before implementation
      leans on it: the offline figure rests on certified mesh truth, which was
      never affected by the probe defect, but the two should now agree and that
      agreement is worth checking rather than assuming.

### What to test next — recommendation, 2026-09-10

In order, each chosen because it is unblocked and its answer changes the next one:

1. **File and adjudicate the self-collision false positives above.** No WG ruling
   needed — it is a fidelity bug, not a conservatism trade. Adjudicate every
   `kind=self` stop in both arms with the pair-specific certified distance, and
   regenerate the `link5`/`link7` hulls against the configurations that trip.
   **Success criterion:** the three `sink_cup` stops read within the
   hull-overhang budget of their +1.4…+4.9 mm. This raises the ceiling every
   other lever is measured against, so it goes first.
2. **Answer #259's open question before designing anything:** why is
   `place_allowance_active` false on all 20 stops with 9 declarations armed?
   Check `support_contact_witness` at those 9 stops from the archived logs. If the
   allowance never engages, that is a bug fix worth up to 9 of 20 stops at zero
   conservatism — and it changes what the WG is being asked in #253/#254.
3. **Run the live 15 mm A/B #253 says is missing.** `OPENRAL_OCTOMAP_RESOLUTION_M=0.015`,
   `tools/ceiling_battery.sh`, `WORKERS=2`, 10 rounds, gate-ON only, on `q-laptop`
   with nothing else on the GPU. **Prediction to falsify:** ≈5 of 20 stops
   convert (class B minus the 3 under 12.99 mm); if fewer than 3 do, the class-B
   bound is wrong. Read `latest_chunk` and the bond-teardown flag on every
   record before believing any rate.
4. **Widen the carry-speed sample** from n=5 with this battery's 8 carrying-phase
   stops (`tools/stop_ee_speed.py` over `openral-256/outputs/ceiling/2026-09-09-fixed`).
   It is the staleness half of #253's trade and is a two-hour job.
5. **The top-up** (`CORRUPTED.md`, 11 runs, one lane at a time) — so `baguette-off`
   stops resting on 3 valid rounds.

Not on the list: anything against the **43 % of gate-ON runs that never get
stopped at all**. That is a policy limit, and no collision lever reaches it.

### Filed as issues, 2026-09-09

The two rulings above are **#253** (15 mm resolution) and **#254** (ADR-0101).
Two findings from this week that are nobody's open item otherwise:

- **#256 — the ceiling battery's absolute rates are lower bounds.** It ran both
  arms simultaneously under contention and excluded only sidecar-crash runs, not
  `deadline-no-grasp`. The *contrast* survives (paired design, shared load), but
  31.1 % / 2.3 % — and therefore **the 29-point figure this whole programme is
  justified by** — are floors measured on a loaded host. `spark` produces zero
  `deadline-no-grasp` on these scenes; re-running the A/B there would settle it.
- **#258 — the 18.9 mm over-approximation is bimodal**, correcting #257's "it is
  the cell": true for 8 stops, wrong for 9 (excess 23–85 mm). Merged.
- **#259 — the place allowance is scoped to the declared target.** 9 of 20 gate-ON
  stops fire with a declaration armed and `place_allowance_active` **false**,
  the payload clipping occupancy *beside* the target. Six were genuinely clear.
  Open question inside it: the allowance never armed on **any** of 20 stops.
- **#255 — a full-tier test-ordering flake** unrelated to collision work, which
  under `-x` stops the run and silently skips everything after it. **Closed the
  same day: already fixed on `master` by `855fe1b`**, which landed in #244 hours
  after the issue was filed. The cause was not a missing teardown — ROS 2's
  `launch.logging` calls `logging.setLoggerClass(LaunchLogger)` at import, and
  that class sets `propagate = False`, so any test that imports `launch` first
  severs propagation for every logger created afterwards in the process.

### Ceiling-run mechanics worth keeping

Three defects had to be fixed before the number was trustworthy; each would
have produced a confidently wrong answer:

1. **Uncaught `subprocess.TimeoutExpired` killed whole workers**, not single
   rounds, leaving the arms *scene-confounded* — gate-off had run mostly
   `fridge` (which completes) and gate-on mostly `utensil` (which then never
   did). The interim 4/17 vs 1/20 was an artifact of scene composition.
2. **The shared sidecar was reaped mid-run.** `SidecarClient` reaps the sidecar
   *it* spawned on exit, so the first crashed worker took it down and every
   later run was policy-free (30-85 s instead of 600+). Fixed structurally with
   a keeper process that owns the sidecar and nothing else.
3. **Policy-free runs must be excluded**, by reading each run's own goal log
   for `ROSConfigError` / sidecar-exit. 14 of 102 runs were dropped this way.
