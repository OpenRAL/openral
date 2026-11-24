# Run a benchmark and publish the number

**Goal:** run a benchmark suite against an rSkill and write the result into the rSkill package.

`openral sim run` answers "does this pairing work?". The benchmark tier answers "how well, under a protocol someone else can repeat?", and it is the only tier that writes a result into an rSkill package.

## Prerequisites

- [Your first sim rollout](../sim/first-rollout.md), and `just sync --group sim`.
- A GPU for anything beyond a smoke run. Diffusion Policy on CPU takes seconds per step, so the 50-episode PushT suite is impractical there. Use `--n-episodes 2` to try the flow.

---

## 1. Which tier writes what

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
 deploy sim      sim run      benchmark scene  →  benchmark run --suite
```

A `BenchmarkScene` is a `SimScene` plus a non-`None` `seed` and `n_episodes`, and a `metadata` block with the `paper` and an `honest_scope` statement. Per-tier loaders reject a wrong-tier YAML at parse time, so an ad-hoc config cannot produce a published number.

| Command | Input | Writes |
| --- | --- | --- |
| `openral sim run` | `scenes/sim/*.yaml` | a summary to `--save-dir`; never the rSkill package |
| `openral benchmark scene` | `scenes/benchmark/*.yaml` | `rskills/<dir>/eval/scene_<scene_id>.json` |
| `openral benchmark run --suite` | `benchmarks/*.yaml` | `rskills/<dir>/eval/<suite_id>.json`, plus episodes and videos (step 4) |
| `openral benchmark report` | `rskills/*/eval/*.json` | nothing; it reads and validates |

A suite aggregates N scenes under uniform protocol invariants. The suite id is the YAML filename stem. The 17 shipped suites are tabulated in [Sim environments: benchmark suites](../../reference/sim-environments.md#benchmark-suites).

## 2. Find a suite

```bash
uv run openral benchmark list       # suite ids from benchmarks/*.yaml
uv run openral rskill list          # paste-able --rskill names
```

The rSkill is a suite's only free axis. Robot, scene, task, seed and episode count are pinned by the YAML, so two rSkills on one suite are comparable.

## 3. Dry-run the matrix

```bash
uv run openral benchmark run --suite pusht --rskill rskills/diffusion-pusht --dry-run
```

It resolves the suite and the rSkill, applies the task filter, prints the plan and exits without a rollout:

```
suite pusht — robot=pusht_2d scene=pusht tasks=1 n_episodes=50
vla   id=diffusion weights=rskills/diffusion-pusht
plan  50 episodes (1 tasks x 50 reps)
```

`benchmark scene` has the same flag. `--suite` also accepts a path to a suite YAML, so you can run a new suite before moving it into `benchmarks/`.

## 4. Run the suite

```bash
uv run openral benchmark run --suite pusht --rskill rskills/diffusion-pusht
```

The runner iterates `scenes × range(seed, seed + n_episodes)` through the same `SimRunner` as `sim run`, so the compatibility gate, spans and latency-budget reporting are identical. On success it prints `wrote .../eval/pusht.json — avg success = 0.xxx over N episodes` and writes:

| File | Tracked | Content |
| --- | --- | --- |
| `rskills/diffusion-pusht/eval/pusht.json` | yes | the full `RSkillEvalResult` |
| `rskills/diffusion-pusht/rskill.yaml` | yes | the aggregate, written back to `benchmarks.pusht` as a comment-preserving edit |
| `rskills/diffusion-pusht/eval/episodes/pusht.json` | no | one record per episode |
| `rskills/diffusion-pusht/eval/videos/pusht/` | no (git-ignored) | one MP4 per episode plus `videos.json` |

Review the tracked diffs before you commit.

Videos are on by default and slow the run. There is no flag that writes nothing; these narrow it:

| Flag | Effect |
| --- | --- |
| `--no-update-manifest` | Leave `rskill.yaml` alone (the eval JSON, episodes and videos are still written) |
| `--no-video` | Skip the MP4s, for allocation-light runs |
| `--out PATH` | Write the eval JSON elsewhere (episodes go in an `episodes/` folder beside it) |
| `--video-dir PATH` | Write the MP4s elsewhere |
| `--n-episodes N` | Override every scene's count |
| `--task ID` | Run one task of the suite |
| `--device`, `--save-dir`, `--benchmarks-dir`, `--dashboard`, `--dashboard-port` | As in `sim run` |

`--n-episodes` is for iteration. A number produced under it is not paper-comparable, because the published protocol is the count in the YAML.

```bash
# Fast iteration: 2 episodes, leave the manifest alone.
uv run openral benchmark run --suite pusht --rskill rskills/diffusion-pusht \
  --n-episodes 2 --no-update-manifest --no-video
```

`benchmark scene` adds `--no-write-eval`: it prints the score and writes nothing, and implies `--no-update-manifest`.

### One task, as a plain sim run

`openral sim run` accepts a suite file too. `--task` picks the entry, which runs as that task's own sim scene. Relative asset paths resolve beside the suite file, the suite's `metadata` is dropped and nothing is written to the rSkill package:

```bash
uv run openral sim run --config benchmarks/libero_10.yaml --task libero_10/3 \
  --rskill smolvla-libero --n-episodes 1
```

---

## 5. The two guardrails

**The suite must be consistent.** `raise_on_invalid_suite` rejects a suite unless every scene shares one non-`None` `robot_id`, the same `n_episodes` and `seed`, and a byte-identical `metadata` block, and every `task.id` is unique. Otherwise an "aggregate" would average across protocols. `task.success_key` and `max_steps` may differ per scene.

**The rSkill must be trained for the task.** A suite is auto-filtered to the tasks the rSkill lists in `evaluated_tasks`; skipped scenes appear in a `benchmark_suite_task_filter` log line. If you pick a task with `--task` that the manifest does not cover, nothing is left to run and you get a typed `ROSCapabilityMismatch`. A policy on the wrong task emits smooth actions but cannot satisfy the success condition, so the 0% would read like a bad model. If the checkpoint does cover the task, add it to `evaluated_tasks`. A manifest with an empty `evaluated_tasks` skips the gate and logs `rskill_task_compat_undeclared`; declare the field.

---

## 6. Read the result

```bash
uv run openral benchmark report            # table across every rSkill
uv run openral benchmark report --json     # same rows, machine-readable
```

It validates every `rskills/<id>/eval/*.json` against `RSkillEvalResult`, the schema the loader uses at install time, so a rotted file fails the command instead of being skipped.

`reproduced_locally` is the honesty switch. A result you produced carries `reproduced_locally: true`. A paper number you have not reproduced is allowed, but it must carry `reproduced_locally: false` and a `reproduction_cli` so a reader can run it. `rskills/diffusion-pusht/eval/pusht.json` shows the reproduced shape:

```json
"source": {
  "paper": "https://arxiv.org/abs/2303.04137",
  "model_variant": "diffusion",
  "evaluated_by": "OpenRAL:openral benchmark run",
  "reproduced_locally": true,
  "reproduction_cli": "openral benchmark run --suite pusht --rskill diffusion-pusht",
  "status": "reproduced"
}
```

The result also carries a `trace_id` that links the number to the run's `cli.command` root span (dashboard `/api/spans/<id>`, Jaeger or Tempo). It is `null` when the run had nowhere to send spans; pass `--dashboard` or set `OTEL_EXPORTER_OTLP_ENDPOINT`.

Never hand-edit a success rate into a manifest or an eval JSON. The commands write both, and the number must be reproducible from the trace alone.

---

## 7. Add a new suite

A suite YAML is a bare list of `BenchmarkScene`s at the document root. A `{id, tasks, metadata}` wrapper is rejected with a redirect. YAML anchors keep per-scene fields short:

```yaml
- scene: &scene
    id: pusht
    backend: mujoco
    observation_height: 96
    observation_width: 96
  task:
    id: pusht/0
    scene_id: pusht
    max_steps: 300
    success_key: is_success
  robot_id: pusht_2d
  n_episodes: 50
  seed: 0
  metadata:
    paper: "https://arxiv.org/abs/2303.04137"
    display_name: "PushT (gym-pusht)"
    simulator: "gym-pusht (pymunk 2-D)"
    honest_scope: >
      50 evaluation rollouts on the single PushT-v0 task, matching the
      Diffusion Policy paper's protocol.
```

`honest_scope` is required: it states what the number does not cover. A suite that ships its own assets names them relative to the suite file. A file at `benchmarks/<id>.yaml` is suite `<id>`. The full invariant list is in [`benchmarks/README.md`](https://github.com/OpenRAL/openral/blob/master/benchmarks/README.md).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSCapabilityMismatch` before any rollout | The rSkill's `evaluated_tasks` cover no task in the suite; see step 5 |
| `--task ... is not in suite` | The message lists the suite's task ids |
| Unexpected `eval/episodes/` and `eval/videos/` files | Expected; pass `--no-video` to skip the videos |
| `rskill.yaml` changed | Pass `--no-update-manifest` next time |
| `benchmark report` fails | An eval JSON no longer validates; fix or regenerate it |

## See also

- [Your first sim rollout](../sim/first-rollout.md): the ad-hoc tier.
- [Write & publish an rSkill](../rskill/write-and-publish-an-rskill.md): section 5 covers eval results as part of publishing.
- [Sim environments reference](../../reference/sim-environments.md): the scene catalogue and the suite table.
