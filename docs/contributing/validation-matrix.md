# The validation matrix

"Run the four sims and see how they perform" is **one versioned command with a
machine-readable verdict**.

| Question | Entry point |
| --- | --- |
| Run a round on the validation host | `just validation-matrix --round-id <id> --expect-sha $(git rev-parse HEAD)` |
| Re-derive a round's verdicts offline | `just validation-matrix-verdicts <round-dir>` |
| What changed since the last round | `just validation-matrix-diff <round-dir> <baseline-dir>` |
| Make a pre-harness round queryable | `just validation-matrix-import <round-dir> --round-id <id> --executed-sha <sha> --stem seed1` |

All four are [`tools/validation_matrix.py`](https://github.com/OpenRAL/openral/blob/master/tools/validation_matrix.py).
Every round it produces feeds the evidence ledger,
[`collision-validation-evidence.md`](../reference/collision-validation-evidence.md).

Exit codes: **0** clean · **2** usage · **3** a guardrail refused, and nothing
was written — not even the round directory · **4** the round ran but at least
one scene bucketed `harness-error`.

---

## 1. Why this exists

Before the harness, the four-scene matrix was driven by shell scripts that lived
only in `~/openral-runs/<date>-<name>/scripts/` on a single machine. Rounds were
not reproducible by anyone but their operator, results were not queryable,
several rounds were never written up at all, and each round re-derived its
tooling with drift.

The harness versions the tooling, makes the round write its own notes, types the
verdict, and refuses the conditions that lost rounds. Which rule closed which
round is recorded in the evidence ledger — see the 2026-09-22 entry, `harness
rules and the rounds that produced them`.

## 2. What a round is

Four RoboCasa scenes, one seed, one policy, one stack:

| Scene key | Config | Task | Kitchen |
| --- | --- | --- | --- |
| `baguette` | `scenes/deploy/robocasa_baguette.yaml` | counter → cabinet | drawn |
| `sink_cup` | `scenes/deploy/robocasa_sink_cup.yaml` | counter → sink | drawn |
| `fridge` | `scenes/deploy/robocasa_fridge_drawer.yaml` | fridge shelf → fridge drawer | `layout_ids: [47]` |
| `utensil` | `scenes/deploy/robocasa_drawer_utensil.yaml` | counter → drawer | `layout_ids: [3]` |

### Why two scenes pin their kitchen

`seed: 1` does not identify a kitchen. RoboCasa draws layout, style, fixtures,
which drawer the task resolves, the door-open amounts and the robot spawn
deviation from one shared `env.rng` inside `Kitchen._load_model`, **on every
reset** — so any upstream change to the draw order silently reshuffles the scene
at the same seed.

For `fridge` that is not merely a reproducibility problem: unpinned, most of
this task's kitchens spawn a robot link inside the kernel's world model, and the
run E-stops before applying a single action chunk. The pin is `[47]`. For
`utensil` the pin is reproducibility only — that scene has no
initial-configuration defect at any of the layouts measured.

Each scene file carries the full measurement record for its own pin in its
header comments and is the source of truth for it; the per-layout census is
[`robocasa-start-state-census.md`](../reference/robocasa-start-state-census.md).
Do not restore a layout that a scene file records as refuted — some are clear on
the ideal grid and stop live.

Every reset records the kitchen it actually composed as
`robocasa_scene_composition` in the run artifacts
([telemetry.md](../reference/telemetry.md)), and a pin the env cannot honour is
a `ROSConfigError` rather than a silent substitution — so a round can no longer
report a layout it did not run.

### The two control surfaces the stack is pinned on

The tracked scenes are **never modified**. Each round materialises a *resolved
copy* beside its artifacts, and launches that.

Seven of the eight pinned knobs have a CLI flag, and are pinned there, because
the deploy CLI's precedence is *explicit flag > scene `runtime:` > default*:

```
--enable-slam --enable-nav2 --enable-octomap --enable-octomap-kernel-check
--no-object-detector --no-enable-scene-vlm --no-dashboard
--hal viewer_enabled=false
```

The eighth, **`enable_reasoner`, has no flag at all.** `openral deploy sim`
resolves it from the scene's `runtime:` block and defaults it to `true`
(`resolve_launch_invocation`), and the tracked scenes carry no `runtime:` block,
so the direct-dispatch stack cannot be expressed in argv. It is spliced into the
resolved copy instead, along with the seed when it differs from the scene's own:

```yaml
runtime:
  enable_reasoner: false
```

`verdicts` records both halves — `stack_argv` and `scene_pins` — so a round's
metadata states the whole stack rather than half of it.

The splice is verified, not assumed: the copy is re-parsed and refused unless it
carries the pins and the seed, and it is diffed against the tracked scene so a
safety key cannot ride along (§3).

The reasoner is **off**, so nothing would issue a goal;
[`tools/_validation_matrix_dispatch.py`](https://github.com/OpenRAL/openral/blob/master/tools/_validation_matrix_dispatch.py)
sends one `ExecuteRskill` goal directly.
[`tools/_validation_matrix_monitor.py`](https://github.com/OpenRAL/openral/blob/master/tools/_validation_matrix_monitor.py)
records the attachment / voxel / witness stream alongside it.

## 3. Guardrails, and the round each one closes

`run` **refuses** — exit code 3, no partial round — rather than warning. Every
one of these has already cost a round; which round, and when, is the
2026-09-22 entry of the evidence ledger, `harness rules and the rounds that
produced them`.

| Guardrail | Refuses when | Why |
| --- | --- | --- |
| `assert_worktree_clean` | `git status --porcelain` is non-empty | A recorded SHA must describe code someone can check out. |
| `assert_sha(--expect-sha)` | `HEAD` is not the requested checkout | The round must validate the commit it claims. |
| `assert_overlay_fresh` | `install/` is older than any tracked `.cpp/.hpp/.h/.msg/.idl` under `cpp/` or `packages/` | An un-rebuilt overlay silently validates the previous commit's C++. Clean rebuild when those change: `rm -rf build install log && just ros2-build`. |
| `resolve_launcher` | this checkout's `.venv/bin/openral` is missing | The `~/.local/bin/openral` wrapper execs the parent checkout's venv, overlay **and `robots/` manifests**. The harness invokes the venv binary by absolute path and exports `OPENRAL_REPO_ROOT`. |
| `assert_sidecar_wire` | `pyzmq` is not importable | Without it the XR-1 adapter is dead. Correct sync: **`just sync --group robocasa --group sidecar-wire`**. |
| `assert_no_safety_overrides` | any argv token matches a safety-knob pattern | A standing prohibition (CLAUDE.md §1.1, §3): the matrix observes the kernel, it never moves it. Tokens are matched lowercase with `-` folded to `_`, so the CLI and parameter spellings of a knob are one pattern. |
| `assert_scene_safety_unmoved` | the resolved scene copy moves a safety-relevant key of the tracked scene | The *other* control surface: argv inspection would never see a margin moved in the scene copy. |
| `gpu_status` | other compute processes hold the GPU (override: `--force-shared-gpu`) | The validation host is shared. A round is announced against what is already resident rather than started blind. |

### Which scene keys are safety-relevant

The scene guard is deliberately precise, because pinning *stack composition* is
the entire point of the harness and must stay possible:

| Refused (safety) | Pinnable (composition) |
| --- | --- |
| `safety:` (the kernel envelope), `hal:` (margins, tolerances), `extra_allowed_collision_pairs:`, `place_declaration:` (it grants the ADR-0097 exemption), `runtime.enable_octomap_kernel_check` (the collision gate) | `runtime.enable_reasoner`, `runtime.enable_slam`, `runtime.enable_nav2`, `runtime.enable_octomap`, the detector, the scene VLM, `seed:` |

Anywhere in the document, a key whose name reads as a margin, tolerance,
allowance, limit, watchdog, deadman or E-stop is safety-relevant by name too.
The comparison is *tracked vs resolved*: composition may differ between the two
files, a safety key may not.

Every round's `metadata.json` records the executed SHA, worktree cleanliness,
the overlay's build time, the resolved launcher path and `OPENRAL_REPO_ROOT`,
the robot manifest, the sync group set, and the exact stack argv — so a reader
can answer "what ran" without asking the operator.

## 4. The verdict contract

Each round writes `verdicts.json`, a
[`ValidationRoundVerdicts`](../reference/schemas/ValidationRoundVerdicts.json)
(defined in `openral_core.schemas`), plus a human-readable `NOTES.md`. A round
cannot end without a written summary, because the summary is a by-product of
running rather than something someone remembers to write.

Per scene, one `outcome`:

| Outcome | Meaning |
| --- | --- |
| `completed` | `sim.task_success_final` reported success. The only success. |
| `estop-collision-real` | The kernel stopped the run and the simulator's distance probe found **solid** geometry at or below 0 m. The stop was correct. |
| `estop-collision-false-positive` | The kernel stopped the run and the nearest true geometry is further from the tripping party than the admissible kernel-vs-probe gap can explain. |
| `estop-collision-within-quantization` | The kernel was conservative by an amount that gap accounts for. Correct behaviour, not a defect. |
| `estop-collision-unadjudicated` | The ground-truth probe was truncated or absent, no budget was known, or the probe does not attest that both of its sides were collidability-filtered. **Not** a synonym for "fine"; `ground_truth.unadjudicated_reason` says which. |
| `estop-initial-configuration` | The stop landed before any action reached the HAL, so the refused configuration is the one the scene reset produced. A scene-config defect; no margin change can clear it. Outranks the ground-truth adjudication. |
| `deadline-after-grasp` / `deadline-no-grasp` | No stop and no success — the run ran out of deadline, with or without a grasp. **Read `deadline-no-grasp` as an instrument symptom first**: it is defined by absence (no success, no stop), which is exactly what a silently dead graph produces. Check the run's delivered chunk count before reading it as a policy property (#256). |
| `harness-error` | The run produced no usable artifact set. Never read as a clean deadline. |

### What counts as "no usable artifact set"

`bool(deploy_lines)` is not the test, because click's usage error *is* lines. A
scene is a `harness-error` when

1. the runner wrote `<stem>_launch_failed.txt` — `/openral/execute_rskill` never
   appeared, so nothing was ever dispatched;
2. the deploy log carries `[ERROR] [launch]: Caught exception in launch` —
   `ros2 launch` threw and unwound. This one has **no marker file and no usage
   banner**: the nodes launch had already spawned keep running and keep logging,
   so the log is long and looks like a run;
3. the deploy log's first line is a `Usage: openral …` banner — the CLI rejected
   its own argv before the graph started;
4. `<stem>_goal.log` has output but no JSON status line — the dispatcher raised
   before any goal reached a terminal state. Every raising path in
   `_validation_matrix_dispatch.py` is a harness failure; a genuine deadline
   overrun still prints `{"status": -1, …}`, and stays a deadline;
5. there is no deploy log at all; or
6. Nav2's `lifecycle_manager_navigation` lost a managed server's bond heartbeat
   early in the run and tore the **whole** navigation stack down. This is the
   one that leaves a *healthy-looking* log: no traceback, nothing exits
   non-zero, the graph simply goes inert and idles out its deadline. The same
   message also appears late in runs that did their work, so the test is *when*:
   the threshold sits in a measured 99 s gap, and is declined outright on a
   `_deploy_excerpt.log`, which begins mid-run. `BOND_TIMEOUT_S` in
   `openral_nav2_bringup/launch/nav2.launch.py` raises Nav2's 4 s default to
   30 s, which makes the trigger itself much rarer; or
7. a lifecycle node never completed a transition — `RuntimeError: transition
   'configure' on '/openral_hal_panda_mobile' did not advance the FSM within
   300.0s` — so the graph never came up at all. Loud, but it still produces
   absence, which is why it needs its own test.

The reason is recorded in the verdict's `harness_error_reason`, named in
`NOTES.md`, and the round exits **4**.

### How adjudication works

The kernel measures **OBB-to-voxel**; the ground-truth probe measures
**mesh-to-mesh**. Subtracting the two directly is only meaningful against the
gap that difference of representation can already produce, and the sim HAL
computes that gap per run and publishes it as
`adjudication_budget.admissible_gap_m` — the collision model's corner slop plus
the voxel half-diagonal, **88.2 mm** on the panda at the 25 mm grid. The harness
uses it whenever the snapshot carries one. The voxel term alone is 21.7 mm at
that grid; applying it on its own is roughly a factor of four too narrow and
turns conservative, correct stops into false positives. It survives as
`quantization_budget_m`, the fallback for a snapshot recorded before the HAL
published a budget; `budget_source` says which was applied.

**That fallback is asymmetric, and deliberately so.** The voxel term is a strict
*lower bound* on the admissible gap — the real gap adds `corner_slop(link)`,
which is the larger of the two on every panda link (45–88 mm against 21.7 mm).
So on a snapshot with no published budget:

| | sound? | verdict |
|---|---|---|
| `discrepancy <= voxel term` | yes — within the smaller bound implies within the true one | `within-quantization` |
| `discrepancy > voxel term` | **no** — proves nothing about the true gap | `unadjudicated` |

`adjudication_budget` landed in #144, so no `false-positive` call from a round
before it can be re-derived from its own artifacts (ledger, "Standing caveats"
§6). Then, in order:

1. Any probed pair at or below 0 m → `real-contact`. Note this is deliberately
   *not* keyed to the body the kernel named: if the kernel says `panda_link7`
   and the probe finds `robot0_link6` at 0.000 m, the configuration really is
   unsafe and the stop stands. It **is** keyed to the pair being solid on both
   sides — see below.
2. Otherwise the tripping party's clearance is compared against the kernel's
   reported depth. Beyond the gap → `false-positive`; within it →
   `within-quantization`. **Beyond a `grid-quantization` gap → `unadjudicated`,
   not `false-positive`** — see the asymmetry above.
3. A truncated probe, a missing snapshot, an unknown budget or an unattested
   probe → `unadjudicated`, with `ground_truth.unadjudicated_reason` naming
   which.

**A 0 m pair is only evidence when both geoms are solid.** A geom with neither
`contype` nor `conaffinity` cannot collide with anything and the safety kernel
never checks it, so a distance to one is not a penetration. The HAL's probe
filters every side to solid geoms and discloses the counts as
`noncollidable_{world,side,other}_geoms_excluded`; the harness promotes a
`<= 0 m` pair to `real-contact` only when that attestation is present. Snapshots
recorded before it are `unadjudicated` rather than trusted, because they really
did rank visual geometry first.

**A very early stop can be unadjudicable, and the notes say so — but so can a
deaf monitor, and the notes say that separately.** The grid resolution is read
from the monitor's first `world_voxels` record, so a scene that trips before the
monitor has seen a grid has no voxel term to fall back on. The monitor attaches
as early as it can, but **not** before the deploy's DDS purge — `openral deploy
sim` unlinks every `/dev/shm/fastrtps_*` this user owns immediately before
spawning `ros2 launch`, and a participant created earlier loses its segments
silently and then receives nothing for the whole scene. It is therefore gated on
the CLI's own `dds_transport_ready:` line, printed after the purge and before
`ros2 launch`; the gate's outcome is recorded per scene in
`<stem>_monitor_gate.txt`.

Both causes read identically in `verdicts.json` — `grid_resolution_m: null` —
so `monitor_records` counts what the monitor actually received (its own
start/stop pair excluded) and `NOTES.md` lists them under **separate**
headings: "Monitor received nothing" (a harness fault; the scene's evidence is
missing) and "Stopped before the monitor saw a voxel grid" (a fact about the
run). `unadjudicated` never reads as "nothing to see", and never as the wrong
reason.

Two things the snapshot itself insists on, and the harness honours:

- **A zero MuJoCo contact count is not an emptiness test.** `contype`/
  `conaffinity` exclusions suppress contacts at real interpenetration — a scene
  can report `payload_contacts == 0` alongside a link at 0.000 m. Adjudicate
  from the distance probes, never the contact list.
- **An untruncated probe that returns no pair is not missing data.** It proves
  the nearest solid geometry is beyond `distmax_m`, which is used as a strict
  lower bound.

### Exemptions and the allowance

`ValidationStopEvidence.exemption_active` is `sweep_min_distance_m` strictly
deeper than `min_distance_m` — the sweep found a deeper cell than the one
reported, which means a support-contact witness exempted it. That inequality is
the authoritative "was an exemption live at the trip" evidence; the kernel's
arm/separate transitions are recorded alongside it, and a disagreement between
the two is itself a finding. `place_allowance_active` is transcribed from the
trip line, and the count of `place_allowance_active=1` disclosures anywhere in
the run is recorded separately.

### The DDS scope a round ran on

`verdicts.json` metadata records `ros_domain_id` and `ros_automatic_discovery_range`
from #227 onward. Before that they were captured nowhere, so for any earlier
round it is **not knowable after the fact** whether it shared a ROS graph with
another machine.

`openral deploy sim` now confines itself (`LOCALHOST` + a private domain) and
both `deploy sim` and `deploy run` refuse to start onto a graph that already has
a `/joint_states` publisher. An explicitly-exported scope still wins, which is
why the values are recorded rather than assumed:

```
ros_domain_id: "77"
ros_automatic_discovery_range: "LOCALHOST"
```

An empty string means the variable was unset — i.e. domain 0 with subnet-wide
discovery, the combination that let a simulation read a live OpenArm's joints.
`validation_matrix.py diff` surfaces a change in either field between rounds.

### Adjudicating a payload-vs-link self stop

A stop whose payload side is the *carried object* and whose other side is a
**robot link** — `a=attached:sim:obj_main b=panda_link1`, `kind=self` — is not a
world stop and must not be scored against world geometry. The rule branches on
**who the other side is**, not on `involves_payload`:

| other side | pair set | coverage block | budget |
| --- | --- | --- | --- |
| `voxel_*` or `place:*` | `nearest_payload_world_pairs` | `nearest_payload_world_coverage` | `admissible_gap_m` |
| a robot link | `nearest_payload_robot_pairs` | `nearest_payload_robot_coverage` | `self_collision.admissible_gap_m` |

The attached-payload self budget is the link's corner slop plus the payload's
(**124.6 mm** = 88.2 + 36.3 on the panda carrying `obj_main`).

The coverage block moves with the pair set for the same reason the pairs do: an
untruncated *robot-world* probe must not be allowed to assert "nothing within
`distmax`" about a *payload-robot* probe that was never consulted. Absence of
the right pair set is a missing instrument, not evidence — such a stop is
`unadjudicated`, never scored off whichever pairs happen to be present.

### Adjudicating a link-vs-link self stop

A stop naming two bare robot links is a different comparison from every other
one the matrix scores. It is never scored against world geometry, and which
budget applies is stated by the kernel, never guessed:

| `depth_is_box_bound` | what the kernel measured | budget | outcome |
| --- | --- | --- | --- |
| `1` | the OBB's bound | `2 × corner_slop` (`adjudication_budget.link_link.admissible_gap_box_m`) | adjudicable |
| `0` | the exact hulls | `hull_overhang_m(link_a) + hull_overhang_m(link_b)`, summed by `hal_admissible_gap_m` from `collision_model_slop.links` | adjudicable once **both** links have a measured overhang, else `unadjudicated` |

`hull_overhang_m` is each declared hull's own overhang past its real source
mesh, sampled by `tools/generate_tight_geometry.py` and published per link in
`adjudication_budget.collision_model_slop.links[<link>].hull_overhang_m`.

The hull term is **per link, never maxed or doubled** the way the box term is
(a consumer has both link names in hand): a link with no stage-2 hull, or a
hull with no source mesh on disk to sample, ships no `hull_overhang_m` at all
— never a silent `0` — and the pair it is part of stays `unadjudicated`.
Charging the box budget instead would forgive a real overlap by up to twice
the OBB corner slop, which on `panda_mobile` is 172 mm; the hull budget is
two orders of magnitude tighter (tenths of a millimetre on every panda link
with a hull), which is the point of measuring it at all.

The grid-quantization fallback is a **voxel** term and is deliberately not
reachable here: a link-vs-link stop has no voxel on either side, so charging it
would budget a comparison the stop never made.

## 5. Diffing rounds

```console
$ just validation-matrix-diff outputs/validation-matrix/<new> outputs/validation-matrix/<old>
2026-08-16-master-1 -> 2026-08-22-master-1  (reproducibility)
  CHANGED baguette   estop-collision-real -> estop-collision-false-positive
           stop.party_a: attached:sim:obj_main -> panda_link5
           stop.min_distance_m: -0.00418456 -> -0.0209178
           stop.sweep_min_distance_m: -0.0355338 -> -0.0209178
```

Equal `executed_sha` **and** equal `seed` on both sides makes it a
**reproducibility** comparison; anything else is a **before/after**. The seed is
part of the test because it decides the scene's initial configuration — a
seed-1-vs-seed-2 pair at one SHA compares two different scenes.

The distinction matters: the policy is stochastic across runs even at a pinned
scene seed (`first_chunk_s` 90.96 vs 34.23, 1285 vs 632 steps, same scene and
tip), so per-run trajectories are not comparable between rounds — **only failure
classes are.** Diff outcomes and tripping pairs; do not read a step count as a
regression.

## 6. Artifacts

Per round, under `outputs/validation-matrix/<round-id>/`:

```
metadata.json              what ran, on what, from what
verdicts.json              ValidationRoundVerdicts
NOTES.md                   the human-readable summary
<scene>/
  <scene>_seed<N>.yaml     the resolved scene copy that was launched
  run_deploy.log           the full deploy log
  run_monitor.jsonl        the monitor stream
  run_goal.log             the dispatcher's one JSON line
  run_monitor_gate.txt     whether the monitor started after the DDS purge
  run_launch_failed.txt    written only when the graph never came up
  run_snapshots/           .npz at every E-stop and while carrying
  run_cinecam/             frames
  run_kernel_evidence.txt  \
  run_task_success.txt      | derived excerpts, for reading
  run_allowance_active.txt  | (no verdict field depends on them)
  run_gt_snapshot.json     /
```

The verdict is derived from `run_deploy.log`, `run_monitor.jsonl`,
`run_goal.log` and `run_launch_failed.txt` **only**. The excerpts exist so a
reviewer can read the evidence without opening a 1.7 MB log.

## 7. Importing the rounds that predate the harness

The pre-harness rounds in `spark:~/openral-runs/<date>-<name>/` use the layout
their shell scripts had: scene directories `bag1` / `sink1` / `fridge1` /
`utensil1`, artifact stem `seed1`, and no metadata block. `import-round` makes
one queryable in place.

```console
$ just validation-matrix-import ~/openral-runs/2026-08-22-master-1 \
    --round-id 2026-08-22-master-1 \
    --executed-sha 2edcf67c3b087958d475813fe19234c12e90698c --stem seed1 \
    --host nvidia --gpu-name "NVIDIA GB10" \
    --sync-group robocasa --sync-group sidecar-wire
imported …/2026-08-22-master-1 as 2026-08-22-master-1:
  {'baguette': 'bag1', 'sink_cup': 'sink1', 'fridge': 'fridge1', 'utensil': 'utensil1'}
```

It writes `metadata.json`, then derives `verdicts.json` + `NOTES.md` in place,
after which `verdicts` and `diff` treat the round like any other — the mapping
and the stem are recorded in `scene_dirs` / `artifact_stem`, so nobody repeats
them. Unrecognised layouts take `--scene-alias <scene>=<dir>`.

What it **derives from the artifacts**, never asks for:

- `stack_argv` — the stack tokens of the deploy log's own resolved
  `argv: … launch …` echo, which is the only record of what the CLI resolved.
  The head varies (older rounds echo a bare `ros2 launch`, newer ones the
  venv-wrapped `<venv>/bin/python …/ros2 launch`), so the line is found by its
  `argv: ` prefix and a `launch` token; only `key:=value` tokens are read.
  Scenes that disagree are refused: two stacks are two rounds.
- `started_at` — the first ROS timestamp in the log.
- `repo_root`, `robot_id`, `robot_manifest_path` — from the argv's `robot_yaml:=`.
- `scene_configs` — the per-round scene YAML kept beside the artifacts.

What it stores **as given**, because no artifact states it: the executed SHA
(from the round's own NOTES/build log), the hostname, the GPU, the sync groups.
What it leaves **empty rather than guess**: `launcher_path`, and
`worktree_clean`, which is `null` for every imported round.

## 8. Running on the Spark

RoboCasa is not installable on every dev host, so rounds run on the project's
DGX Spark. It is a **shared machine**.

```bash
ssh spark
cd <checkout>                       # a clean, committed checkout
just sync --group robocasa --group sidecar-wire
rm -rf build install log && just ros2-build     # when kernel / msgs / bridge changed
source /opt/ros/jazzy/setup.bash && source install/setup.bash
source .venv/bin/activate
export PATH="$PATH:$HOME/.local/bin"   # APPEND. See below — never prepend.
nvidia-smi                          # who else is on the GPU?
just validation-matrix --round-id $(date +%F)-<name> --expect-sha $(git rev-parse HEAD)
```

Etiquette:

- **Check `nvidia-smi` before you start.** The harness refuses when other
  compute processes are resident; `--force-shared-gpu` only if you know they are
  yours. Do not evict someone else's run.
- **`~/.local/bin` goes on the END of `PATH`, never the front.** Both facts
  matter: `just` itself lives *only* there, so without it on `PATH` no recipe
  runs at all — and `~/.local/bin/openral` is a wrapper that hardcodes
  `_OPENRAL_DIR=~/workspace/openral` and execs the **parent** checkout's venv,
  overlay and `robots/` manifests. Append it after activating the venv, so the
  venv's `openral` still wins. The harness also invokes its launcher by absolute
  path, so a mis-ordered `PATH` cannot silently decide which code runs — but the
  ordering still decides it for everything else you type.
- **Sync deliberately, and only when you mean to.** All the `just
  validation-matrix*` recipes use `uv run --no-sync`: a bare `uv run` re-resolves
  the environment on every invocation, and a mid-round re-resolve has already
  cost a round.
- **One round at a time.** Four scenes at ~7 min each plus teardown is roughly
  half an hour of exclusive GPU.
- **Copy the round off the host** and re-derive verdicts anywhere with
  `just validation-matrix-verdicts` — it is offline and side-effect free.

## 9. Testing the harness

`tests/unit/test_validation_matrix.py` runs the verdict derivation, the diff,
the import and every guardrail against **recorded artifacts** from three real
rounds (`tests/unit/fixtures/validation_matrix/`, provenance in `SOURCE.txt`).
The assertions are pinned to what the evidence ledger concluded about those
rounds, so if the extractor stops reproducing the published table the suite goes
red.

No run is faked, and no fixture is synthesized:

- the two `master-1` rounds are the same code, the same scene and the same seed
  with a different outcome, which makes the diff test a real round-over-round
  comparison rather than an invented one. They keep their original pre-harness
  layout, so reading them at all exercises the importer;
- `2026-08-22-harness-1` is the harness's own first live round — four scenes of
  click usage error — and is the fixture for "a launch failure is not a
  deadline";
- the flags in `STACK_ARGV` are checked against the **live** `openral deploy
  sim` parser, and the absence of a reasoner flag is asserted rather than
  assumed.

The runner half needs a GPU host with RoboCasa, so it is exercised on the Spark
rather than in CI: rounds `2026-08-22-harness-1` (the failure) and
`2026-08-22-harness-2` (four scenes, all four verdicted) live under
`spark:~/workspace/openral-matrix-baseline/outputs/validation-matrix/`.
