# Collision programme — where it stands, and the one measurement that decides it

> Written 2026-09-06 after the #204 A/B (120 runs, q-laptop). Companion to
> [`docs/reference/collision-validation-evidence.md`](docs/reference/collision-validation-evidence.md)
> (the round ledger), [`docs/reference/collision-safety-alternatives-survey.md`](docs/reference/collision-safety-alternatives-survey.md)
> (the 2026-08-30 survey) and [`docs/reference/robocasa-start-state-census.md`](docs/reference/robocasa-start-state-census.md).
>
> **Status: analysis + proposal.** Nothing here has landed. The ceiling
> experiment in §4 is the gate on everything below it.

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

**The gate costs 29 points of completion.** 88 valid runs, 4 scenes x 2 arms,
10-12 per cell, same commit (`80027b18`) and host, arms run simultaneously:

| | valid runs | completed | rate |
| --- | ---: | ---: | ---: |
| world-voxel gate **OFF** | 45 | **14** | **31.1 %** |
| world-voxel gate **ON** (shipped) | 43 | **1** | **2.3 %** |

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
for: **up to 29 points**, concentrated in the payload class.

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
| 3 | ~~Voxel resolution 25 → 12.5–15 mm~~ | — | **struck: cost is cubic** | see below |
| 4 | ~~Payload as a tight hull~~ | — | **−1.5 mm: none** | struck; measured out |

### Why resolution is struck (lever 3)

`OccupancyVoxels.occupancy` is a **dense** `uint8[]`, and the kernel's per-link
window holds `O(1/res³)` cells, so halving the cell size is an **8× check cost**,
not a 2× one:

| resolution | grid cells | message | window cost | est. check | error term |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **25 mm (today)** | 614 125 | 0.61 MB | 1.00× | 5.8 ms | 21.7 mm |
| 20 mm | 1 191 016 | 1.19 MB | 1.95× | 11.3 ms | 17.3 mm |
| 15 mm | 2 803 221 | 2.80 MB | 4.63× | **26.7 ms** | 13.0 mm |
| 12.5 mm | 4 826 809 | 4.83 MB | 8.00× | **46.1 ms** | 10.8 mm |

Against a 33 ms hard budget at 30 Hz (25 ms soak target), 15 mm is marginal and
12.5 mm is over. 20 mm fits but buys only 4.4 mm of the 20.1 mm payload excess.
The cap would also have to rise 2-8×. **Poor return; not the lever.**

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
      Result in §4.
- [x] **Close-vs-continue** — **continue.** The gate is worth 29 points of
      completion, so the §5 levers are competing for real headroom rather than
      for noise.
- [ ] **Lever 1: the payload bounding box** — first, and by a distance. It is
      71 % of all stops, and `utensil` (58 % ceiling, 0 % shipped) is a pure
      instance of it.
- [x] **Lever 1: `tight_geometry` on `panda_link3`/`link4`/`link6`** — done
      2026-09-07. Support excess 75.6→23.8, 76.1→23.2 and **52.7→21.5 mm**;
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
- [ ] **Lever 3: voxel resolution 25 -> 15 mm.**
- [ ] Drop `baguette` from the collision scorecard: 0 % with the gate off means
      it is policy-bound and cannot report on collision work either way.
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
- [ ] **Decide #217** — recommended: close it. #204 is excluded at 0.85 power,
      the suspect window is narrowed to pre-`34e7b5f`, and the standing 29-point
      cost dwarfs the drop it was chasing. The alternative is re-scoping it to
      the single remaining suspect (#202's ACM retirement) rather than a full
      bisect. Needs a human call.
- [ ] **Implement ADR-0101** once ruled on — the one lever with headroom left.

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
