# The validation matrix

"Run the four sims and see how they perform" is **one versioned command with a
machine-readable verdict**.

| Question | Entry point |
| --- | --- |
| Run a round on the validation host | `just validation-matrix --round-id <id> --expect-sha $(git rev-parse HEAD)` |
| Re-derive a round's verdicts offline | `just validation-matrix-verdicts <round-dir>` |
| What changed since the last round | `just validation-matrix-diff <round-dir> <baseline-dir>` |

All three are [`tools/validation_matrix.py`](https://github.com/OpenRAL/openral/blob/master/tools/validation_matrix.py).
Every round it produces feeds the evidence ledger,
[`collision-validation-evidence.md`](../reference/collision-validation-evidence.md).

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

The scenes are used **verbatim from the tree**. The stack is pinned by CLI flags,
not by a per-round YAML copy — the deploy CLI's precedence is *explicit flag >
scene `runtime:` > default*, and per-round copies are precisely how the
historical configs drifted away from the tracked ones. Only a seed different
from the scene's own `seed:` materialises a resolved copy, and then only that
one line is rewritten.

The pinned stack, recorded verbatim into every round's metadata:

```
--no-enable-reasoner --enable-slam --enable-nav2 --enable-octomap
--enable-octomap-kernel-check --no-object-detector --no-enable-scene-vlm
--no-dashboard --hal viewer_enabled=false
```

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
| `gpu_status` | other compute processes hold the GPU (override: `--force-shared-gpu`) | The validation host is shared. A round is announced against what is already resident rather than started blind. |

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
| `estop-collision-real` | The kernel stopped the run and the simulator's distance probe found geometry at or below 0 m. The stop was correct. |
| `estop-collision-false-positive` | The kernel stopped the run and the nearest true geometry is further from the tripping party than the occupancy grid's quantization budget can explain. |
| `estop-collision-within-quantization` | The kernel was conservative by an amount the voxel size accounts for. Correct behaviour, not a defect. |
| `estop-collision-unadjudicated` | The ground-truth probe was truncated, absent, or the grid resolution unknown. **Not** a synonym for "fine". |
| `estop-initial-configuration` | The stop landed before any action reached the HAL, so the refused configuration is the one the scene reset produced. A scene-config defect; no margin change can clear it. Outranks the ground-truth adjudication. |
| `deadline-after-grasp` / `deadline-no-grasp` | No stop and no success — the run ran out of deadline, with or without a grasp. |
| `harness-error` | The run produced no usable artifact set. Never read as a clean deadline. |

### How adjudication works

The quantization budget is **half the voxel's body diagonal**, read from the
run's own monitor records rather than assumed — at the 25 mm grid the rounds
use, 21.7 mm. Then, in order:

1. Any probed pair at or below 0 m → `real-contact`. Note this is deliberately
   *not* keyed to the body the kernel named: if the kernel says `panda_link7`
   and the probe finds `robot0_link6` at 0.000 m, the configuration really is
   unsafe and the stop stands.
2. Otherwise the tripping party's clearance is compared against the kernel's
   reported depth. Beyond the budget → `false-positive`; within it →
   `within-quantization`.
3. A truncated probe, a missing snapshot or an unknown resolution →
   `unadjudicated`.

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

Equal `executed_sha` on both sides makes it a **reproducibility** comparison;
differing SHAs a **before/after**. The distinction matters: the policy is
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
  run_deploy.log           the full deploy log
  run_monitor.jsonl        the monitor stream
  run_goal.log             the dispatcher's one JSON line
  run_snapshots/           .npz at every E-stop and while carrying
  run_cinecam/             frames
  run_kernel_evidence.txt  \
  run_task_success.txt      | derived excerpts, for reading
  run_allowance_active.txt  | (no verdict field depends on them)
  run_gt_snapshot.json     /
```

The verdict is derived from `run_deploy.log`, `run_monitor.jsonl` and
`run_goal.log` **only**. The excerpts exist so a reviewer can read the evidence
without opening a 1.7 MB log.

## 7. Running on the Spark

RoboCasa is not installable on every dev host, so rounds run on the project's
DGX Spark. It is a **shared machine**.

```bash
ssh spark
cd <checkout>                       # a clean, committed checkout
just sync --group robocasa --group sidecar-wire
rm -rf build install log && just ros2-build     # when kernel / msgs / bridge changed
source /opt/ros/jazzy/setup.bash && source install/setup.bash
nvidia-smi                          # who else is on the GPU?
just validation-matrix --round-id $(date +%F)-<name> --expect-sha $(git rev-parse HEAD)
```

Etiquette:

- **Check `nvidia-smi` before you start.** The harness refuses when other
  compute processes are resident; `--force-shared-gpu` only if you know they are
  yours. Do not evict someone else's run.
- **Never append `~/.local/bin` before the venv on `PATH`.** The wrapper there
  execs a different checkout. The harness invokes its launcher by absolute path
  precisely so a mis-ordered `PATH` cannot decide which code runs.
- **One round at a time.** Four scenes at ~7 min each plus teardown is roughly
  half an hour of exclusive GPU.
- **Copy the round off the host** and re-derive verdicts anywhere with
  `just validation-matrix-verdicts` — it is offline and side-effect free.

## 8. Testing the harness

`tests/unit/test_validation_matrix.py` runs the verdict derivation, the diff and
every guardrail against **recorded artifacts** from two real rounds
(`tests/unit/fixtures/validation_matrix/`, provenance in `SOURCE.txt`). The
assertions are pinned to what the evidence ledger concluded about those rounds,
so if the extractor stops reproducing the published table the suite goes red.

No run is faked, and no fixture is synthesized: the two rounds included are the
same code, the same scene and the same seed with a different outcome, which
makes the diff test a real round-over-round comparison rather than an invented
one.
