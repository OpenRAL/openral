# Run a benchmark and publish the number

`openral sim run` answers "does this pairing work?". The **benchmark** tier
answers "how well, under a protocol someone else can repeat?" — and it is the
only tier that writes a result back into an rSkill package.

This page covers the three benchmark commands, what each writes, and the
guardrails that stop a number from being quietly wrong.

Prerequisite: you can already complete
[your first sim rollout](../sim/first-rollout.md).

---

## 1. Which tier writes what

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
 deploy sim      sim run      benchmark scene  →  benchmark run --suite
```

A `BenchmarkScene` is a `SimScene` plus the two things a claim needs: a
non-`None` `seed` and `n_episodes`, and a `metadata` block naming the `paper`
and an `honest_scope` statement. Per-tier loaders reject a wrong-tier YAML at
parse time, so you cannot accidentally publish a number from an ad-hoc config.

| Command | Input | Writes |
| --- | --- | --- |
| `openral sim run` | `scenes/sim/*.yaml` | a summary to `--save-dir`; **never** the rSkill package |
| `openral benchmark scene` | `scenes/benchmark/*.yaml` | `rskills/<dir>/eval/scene_<scene_id>.json` |
| `openral benchmark run --suite` | `benchmarks/*.yaml` | `rskills/<dir>/eval/<suite_id>.json` |
| `openral benchmark report` | `rskills/*/eval/*.json` | nothing — reads and validates |

A **suite** aggregates N scenes under uniform protocol invariants; a **scene**
is one of them. The suite id is the YAML filename stem.

---

## 2. Find a suite

```bash
openral benchmark list        # suite ids from benchmarks/*.yaml
openral rskill list           # paste-able --rskill names
```

The rSkill is the only free axis of a suite — robot, scene, task, seed and
episode count are all pinned by the YAML. That is the point: two rSkills
evaluated on one suite are directly comparable.

---

## 3. Dry-run the matrix

```bash
openral benchmark run --suite pusht --rskill rskills/diffusion-pusht --dry-run
```

Prints the planned (task × seed) matrix and exits without a single rollout.
Use it to confirm wiring in CI, and to see how many episodes you are about to
commit to — `benchmarks/pusht.yaml` pins 50.

`benchmark scene` has the same flag.

---

## 4. Run the suite

```bash
openral benchmark run --suite pusht --rskill rskills/diffusion-pusht
```

The runner iterates `scenes × range(seed, seed + n_episodes)`, delegating each
rollout to the same `SimRunner` that backs `openral sim run` — so the
compatibility gate, OpenTelemetry spans and latency-budget reporting are
identical. Two files change on success:

- **`rskills/diffusion-pusht/eval/pusht.json`** — the full
  `RSkillEvalResult`.
- **`rskills/diffusion-pusht/rskill.yaml`** — the aggregate written back to
  `benchmarks.pusht`, as a surgical edit that preserves comments.

Both are tracked paths. Review the diff before committing it.

### When you do not want either write

```bash
# Read-only: still write the eval JSON, leave the manifest alone.
openral benchmark run --suite pusht --rskill rskills/diffusion-pusht \
  --no-update-manifest

# Fast iteration: 2 episodes instead of the suite's 50.
openral benchmark run --suite pusht --rskill rskills/diffusion-pusht \
  --n-episodes 2 --no-update-manifest
```

`--n-episodes` overrides every scene in the suite. It exists for iteration —
a number produced under it is **not** paper-comparable, because the published
protocol is the count in the YAML.

`benchmark scene` adds a third opt-out, `--no-write-eval`: the rollout runs and
prints its score, and nothing at all is written to the package. It implies
`--no-update-manifest`.

---

## 5. The two guardrails

### The suite must be internally consistent

`openral_core.raise_on_invalid_suite` rejects a suite unless every scene shares
one non-`None` `robot_id`, the same `n_episodes` and `seed`, and a
byte-identical `metadata` block — and every `task.id` is unique. Without that,
an "aggregate" would be averaging across different protocols.

`task.success_key` and `max_steps` **may** differ per scene, because suites like
ManiSkill3 legitimately span tasks with different step budgets.

### The rSkill must have been trained for the task

A suite is auto-filtered to the tasks the rSkill declares in
`evaluated_tasks`; scenes it does not cover are skipped with a logged
`benchmark_suite_task_filter` summary. Pick a task explicitly with `--task`
that the manifest does not cover and the filter leaves nothing to run, so you
get a typed `ROSCapabilityMismatch` instead of a rollout. `benchmark scene`
applies the same gate to its single scene, and its message says why the gate
exists:

> the checkpoint was trained/validated for a different task; running it here
> yields a plausible-looking rollout that cannot succeed

This is the failure mode the gate exists for. A policy pointed at the wrong
task still emits smooth, confident actions — it simply can never satisfy the
success condition, and the resulting 0% reads like a bad model rather than a
mispairing. If the checkpoint genuinely covers the task, add it to
`evaluated_tasks` in the manifest.

A manifest with an **empty** `evaluated_tasks` skips the gate and logs
`rskill_task_compat_undeclared`. Declare the field.

---

## 6. Read the result

```bash
openral benchmark report              # rich table across every rSkill
openral benchmark report --json       # same rows, machine-readable
```

It walks every `rskills/<id>/eval/*.json` and validates each against
`RSkillEvalResult` — the same schema the loader uses at install time — so a
rotted file **fails the command** rather than being skipped. Rows carry the
benchmark, robot, simulator, `model_variant`, `status`, the results block, and
`reproduced_locally`.

### `reproduced_locally` is the honesty switch

A result you produced carries `reproduced_locally: true`. A number you are
citing from a paper without having reproduced it is allowed, but must carry
`reproduced_locally: false` **plus** a `reproduction_cli` so a reader can run it
themselves.

The in-tree `rskills/diffusion-pusht/eval/pusht.json` shows the reproduced
shape:

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

Locally-produced results also carry a `trace_id`, so a reviewer can deep-link
from the number straight to its trace tree in Jaeger or Tempo.

Never hand-edit a success rate into a manifest or an eval JSON. The commands
write both, and the number is meant to be reproducible from the trace alone.

---

## 7. Add a new suite

Suite YAMLs are a **bare list** of `BenchmarkScene`s at the document root — not
a `{id, tasks, metadata}` wrapper, which is rejected with a redirect. Use YAML
anchors to keep the per-scene fields DRY:

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

`honest_scope` is required and is not decoration: it is where you state what
the number does **not** cover. Drop the file in `benchmarks/<id>.yaml` and it
is a suite id; see [`benchmarks/README.md`](https://github.com/OpenRAL/openral/blob/master/benchmarks/README.md)
for the full invariant list.

---

## See also

- [Your first sim rollout](../sim/first-rollout.md) — the ad-hoc tier.
- [Write & publish an rSkill](../rskill/write-and-publish-an-rskill.md) — §5
  covers producing eval results as part of publishing.
- [Sim environments reference](../../reference/sim-environments.md) — the
  BenchmarkScene catalogue.
- [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md)
  — the three-tier cookbook.
