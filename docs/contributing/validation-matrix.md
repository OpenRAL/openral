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

For roughly seventeen rounds over ten days, the four-scene collision matrix was
driven by tooling that lived **only** in `~/openral-runs/<date>-<name>/scripts/`
on one machine: `run_matrix.sh`, `drive_round.sh`, `attach_monitor4.py`,
`postprocess.sh`, `adjudicate.py`, `verdict_table.py`, and a per-scene copy of
each scene YAML. Nothing was in the repo. The costs were concrete:

- **Rounds were not reproducible** by anyone but the operator who ran them.
- **Results were not queryable.** Comparing two rounds meant an agent re-reading
  two multi-megabyte logs and writing prose.
- **Several rounds have no written summary at all** — their findings survive
  only as constants in code comments (ledger, "Standing caveats" §5).
- **One round silently executed the wrong checkout**, because `~/.local/bin/openral`
  is a wrapper that hardcodes a repo root and execs the *parent* checkout's venv
  and overlay.
- **Each round re-derived the tooling with drift** — one round's sync recipe
  (`just sync --group robocasa`, without `--group sidecar-wire`) uninstalled
  `pyzmq` and broke the XR-1 adapter mid-round.

The harness is the fix for all five: the tooling is versioned, the round writes
its own notes, the verdict is a typed contract, and the guardrails refuse the
conditions that lost the rounds above.

## 2. What a round is

Four RoboCasa scenes, one seed, one policy, one stack:

| Scene key | Config | Task |
| --- | --- | --- |
| `baguette` | `scenes/deploy/robocasa_baguette.yaml` | counter → cabinet |
| `sink_cup` | `scenes/deploy/robocasa_sink_cup.yaml` | counter → sink |
| `fridge` | `scenes/deploy/robocasa_fridge_drawer.yaml` | fridge shelf → fridge drawer |
| `utensil` | `scenes/deploy/robocasa_drawer_utensil.yaml` | counter → drawer |

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

This is a correction to how the harness shipped: it originally pinned
`--no-enable-reasoner`, a flag that does not exist, and its first live round
(`2026-08-22-harness-1`) died in all four scenes in under a second. `verdicts`
records both halves — `stack_argv` and `scene_pins` — so a round's metadata
states the whole stack rather than half of it.

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
one of these has already cost a round.

| Guardrail | Refuses when | The round it closes |
| --- | --- | --- |
| `assert_worktree_clean` | `git status --porcelain` is non-empty | A round run from uncommitted changes: its recorded SHA describes code nobody can check out. |
| `assert_sha(--expect-sha)` | `HEAD` is not the requested checkout | **The wrong-checkout round.** |
| `assert_overlay_fresh` | `install/` is older than any tracked `.cpp/.hpp/.h/.msg/.idl` under `cpp/` or `packages/` | A round that silently validated the *previous* commit's C++ because the kernel/msgs/bridge were not rebuilt. Clean rebuild when those change: `rm -rf build install log && just ros2-build`. |
| `resolve_launcher` | this checkout's `.venv/bin/openral` is missing | The `~/.local/bin/openral` wrapper execs the parent checkout's venv, overlay **and `robots/` manifests**. The harness invokes the venv binary by absolute path and exports `OPENRAL_REPO_ROOT`. |
| `assert_sidecar_wire` | `pyzmq` is not importable | The round where `just sync --group robocasa` alone stripped `pyzmq` and broke the XR-1 adapter. Correct: **`just sync --group robocasa --group sidecar-wire`**. |
| `assert_no_safety_overrides` | any argv token matches a safety-knob pattern | Not a past incident — a standing prohibition (CLAUDE.md §1.1, §3). The matrix observes the kernel; it never moves it. Tokens are matched lowercase with `-` folded to `_`, so the CLI and parameter spellings of a knob are one pattern. |
| `assert_scene_safety_unmoved` | the resolved scene copy moves a safety-relevant key of the tracked scene | The *other* control surface. Once a round materialises a scene copy, that copy is where a margin could move invisibly — argv inspection would never see it. |
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
can no longer end without a written summary, because the summary is a
by-product of running rather than something someone remembers to write.

Per scene, one `outcome`:

| Outcome | Meaning |
| --- | --- |
| `completed` | `sim.task_success_final` reported success. The only success. |
| `estop-collision-real` | The kernel stopped the run and the simulator's distance probe found **solid** geometry at or below 0 m. The stop was correct. |
| `estop-collision-false-positive` | The kernel stopped the run and the nearest true geometry is further from the tripping party than the admissible kernel-vs-probe gap can explain. |
| `estop-collision-within-quantization` | The kernel was conservative by an amount that gap accounts for. Correct behaviour, not a defect. |
| `estop-collision-unadjudicated` | The ground-truth probe was truncated or absent, no budget was known, or the probe does not attest that both of its sides were collidability-filtered. **Not** a synonym for "fine"; `ground_truth.unadjudicated_reason` says which. |
| `estop-initial-configuration` | The stop landed before any action reached the HAL, so the refused configuration is the one the scene reset produced. A scene-config defect; no margin change can clear it. Outranks the ground-truth adjudication. |
| `deadline-after-grasp` / `deadline-no-grasp` | No stop and no success — the run ran out of deadline, with or without a grasp. |
| `harness-error` | The run produced no usable artifact set. Never read as a clean deadline. |

### What counts as "no usable artifact set"

`bool(deploy_lines)` is not the test, because click's usage error *is* lines:
the first live round passed a flag that does not exist and all four scenes were
reported as `deadline-no-grasp` with exit 0. A scene is a `harness-error` when

1. the runner wrote `<stem>_launch_failed.txt` — `/openral/execute_rskill` never
   appeared, so nothing was ever dispatched;
2. the deploy log carries `[ERROR] [launch]: Caught exception in launch` —
   `ros2 launch` threw and unwound. This is the one with **no marker file and no
   usage banner**: the nodes launch had already spawned keep running and keep
   logging, so the log is long and looks like a run. At `87dcda1` a missing
   `payload_footprint_node.py` produced exactly that and the scene was bucketed
   `deadline-no-grasp` with `harness_error_reason` and
   `dispatch_failure_reason` both empty;
3. the deploy log's first line is a `Usage: openral …` banner — the CLI rejected
   its own argv before the graph started;
4. `<stem>_goal.log` has output but no JSON status line — the dispatcher raised
   before any goal reached a terminal state. Every raising path in
   `_validation_matrix_dispatch.py` is a harness failure; a genuine deadline
   overrun still prints `{"status": -1, …}`, and stays a deadline; or
5. there is no deploy log at all.

The reason is recorded in the verdict's `harness_error_reason`, named in
`NOTES.md`, and the round exits **4**.

### How adjudication works

The kernel measures **OBB-to-voxel**; the ground-truth probe measures
**mesh-to-mesh**. Subtracting the two directly is only meaningful against the
gap that difference of representation can already produce, and the sim HAL
computes that gap per run and publishes it as
`adjudication_budget.admissible_gap_m` — the collision model's corner slop plus
the voxel half-diagonal, **88.2 mm** on the 2026-08-23 rounds. The harness uses
it whenever the snapshot carries one. The voxel term alone is 21.7 mm at the
25 mm grid these rounds use; applying that on its own is roughly a factor of
four too narrow and turned conservative, correct stops into false positives
(the 2026-08-23 `utensil` stop: `robot0_link1` 43.3 mm clear against a −17.3 mm
read). It survives as `quantization_budget_m`, and as the fallback for a
snapshot recorded before the HAL published a budget; `budget_source` says which
was applied. Then, in order:

1. Any probed pair at or below 0 m → `real-contact`. Note this is deliberately
   *not* keyed to the body the kernel named: if the kernel says `panda_link7`
   and the probe finds `robot0_link6` at 0.000 m, the configuration really is
   unsafe and the stop stands. It **is** keyed to the pair being solid on both
   sides — see below.
2. Otherwise the tripping party's clearance is compared against the kernel's
   reported depth. Beyond the gap → `false-positive`; within it →
   `within-quantization`.
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
did rank visual geometry first: the 2026-08-23 fridge stop was adjudicated
`real-contact` off `robot0_g42_vis` at 0.000 m while the same link's collision
geom was 2.5 mm clear, and the 2026-08-22 `sink_cup` stop off `obj_reg_bbox`,
the payload's own region marker.

**A very early stop can be unadjudicable, and the notes say so — but so can a
deaf monitor, and the notes say that separately.** The grid resolution is read
from the monitor's first `world_voxels` record, so a scene whose initial
configuration trips the kernel before the monitor has seen a grid has no voxel
term to fall back on. The monitor therefore attaches as soon as the graph is
launched — not, as it did originally, five seconds before dispatch, which is
minutes later and after the sim clock has already run. The 2026-08-22 utensil
scene tripped at sim t≈4.7 s and recorded zero snapshots, a null resolution and
`unadjudicated`.

But **not** before the deploy's DDS purge. `openral deploy sim` unlinks every
`/dev/shm/fastrtps_*` this user owns immediately before spawning `ros2 launch`;
a participant created earlier loses its segments silently and then receives
nothing for the whole scene. The 2026-08-23 round attached ~6 ms in and all 24
of its `run_monitor.jsonl` files contain exactly `monitor_started` and
`monitor_stopped`. The monitor is now gated on the deploy CLI's own
`dds_transport_ready:` line, which is printed after the purge and before
`ros2 launch` is spawned — so it is up tens of seconds before the sim clock
starts, and the early-stop coverage is kept in full. The gate's outcome is
recorded per scene in `<stem>_monitor_gate.txt`.

The two causes read identically in `verdicts.json` — `grid_resolution_m: null`
— so `monitor_records` counts what the monitor actually received (its own
start/stop pair excluded) and `NOTES.md` lists them under **separate**
headings: "Monitor received nothing" (a harness fault; the scene's evidence is
missing) and "Stopped before the monitor saw a voxel grid" (a fact about the
run). `unadjudicated` never reads as "nothing to see", and never as the wrong
reason.

Two things the snapshot itself insists on, and the harness honours:

- **A zero MuJoCo contact count is not an emptiness test.** `contype`/
  `conaffinity` exclusions suppress contacts at real interpenetration — the
  fridge scene has `payload_contacts == 0` alongside a link at 0.000 m.
  Adjudicate from the distance probes, never the contact list.
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
seed-1-vs-seed-2 pair at one SHA compares two different scenes, and was once
labelled `reproducibility` on the SHA alone. The distinction matters: the policy is
stochastic across runs even at a pinned scene seed (`first_chunk_s` 90.96 vs
34.23, 1285 vs 632 steps, same scene and tip), so per-run trajectories are not
comparable between rounds — **only failure classes are.** Diff outcomes and
tripping pairs; do not read a step count as a regression.

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

Roughly seventeen rounds live in `spark:~/openral-runs/<date>-<name>/`, in the
layout their shell scripts used: scene directories `bag1` / `sink1` / `fridge1`
/ `utensil1`, artifact stem `seed1`, and no metadata block at all. Diffing one
against a harness round used to mean mapping those by hand.

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
  `argv: ros2 launch …` echo, which is the only record of what the CLI resolved.
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
  overlay and `robots/` manifests. Append it (`export
  PATH="$PATH:$HOME/.local/bin"`) after activating the venv, so the venv's
  `openral` still wins. The harness also invokes its launcher by absolute path,
  so a mis-ordered `PATH` cannot silently decide which code runs — but the
  ordering still decides it for everything else you type.
- **Sync deliberately, and only when you mean to.** All the `just
  validation-matrix*` recipes use `uv run --no-sync`: a bare `uv run` re-resolves
  the environment on every invocation, and the incident this harness exists to
  prevent is exactly a sync that uninstalled `pyzmq` in the middle of a round.
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
