# benchmarks/

Reproducible benchmark **suites** for `OpenRAL` rSkills (ADR-0009 + ADR-0041 +
ADR-0042).

Each YAML in this directory is a **bare list of `openral_core.BenchmarkScene`s**
at the YAML root (ADR-0042). A suite **aggregates** N scene rollouts under
uniform protocol invariants; the suite id is the filename stem (e.g.
`benchmarks/libero_spatial.yaml` → suite id `"libero_spatial"`). The rSkill
(VLA) is the **only** free axis; supplied at the CLI by
`openral benchmark run --suite <id> --rskill <name>`.

This is intentionally different from the sibling [`scenes/`](../scenes/), which
holds the three scene tiers (`DeployScene` / `SimScene` / `BenchmarkScene`,
ADR-0041) for single-rollout commands (`openral deploy sim`, `openral sim run`,
`openral benchmark scene`).

## Schema

Suite YAMLs are loaded by `openral_core.load_benchmark_suite(path)` — returns
`list[BenchmarkScene]`. Per-scene Pydantic validation runs at load time;
suite-level invariants are enforced separately by
`openral_core.raise_on_invalid_suite(scenes, suite_id=...)` so callers can also
validate in-memory suites. Both live in `python/core/src/openral_core/loaders.py`.

Suite invariants (`raise_on_invalid_suite`):

- The scene list is non-empty.
- Every `scenes[i].robot_id` is the same non-`None` value.
- Every `scenes[i].n_episodes` and `scenes[i].seed` are uniform across the suite.
- Every `scenes[i].metadata` (the full `BenchmarkMetadata` block — `paper`,
  `honest_scope`, optional `display_name` + `simulator`) is byte-identical.
- Every `scenes[i].task.id` is unique.

Per-scene `task.success_key` and `task.max_steps` MAY differ (e.g. ManiSkill3
shares one suite across `PickCube-v1` and `StackCube-v1` with different step
budgets).

Pre-ADR-0042 the YAML root was a `{id, tasks, metadata}` wrapper around
`BenchmarkSpec`; that shape is now rejected with an explicit redirect message
naming ADR-0042. YAML authors should use anchors (`&scene` / `<<: *scene`) to
keep the per-scene fields DRY — see [`libero_spatial.yaml`](libero_spatial.yaml)
for the pattern.

## Catalogue

| Suite YAML | Robot | Scene | Tasks | Per-scene `n_episodes` | `success_key` | `max_steps` | Total rollouts |
|---|---|---|---|---|---|---|---|
| `libero_spatial.yaml`     | franka_panda   | libero_spatial (MuJoCo)         | 10 | 10 | `is_success` | 280 | 100 |
| `libero_object.yaml`      | franka_panda   | libero_object (MuJoCo)          | 10 | 10 | `is_success` | 280 | 100 |
| `libero_goal.yaml`        | franka_panda   | libero_goal (MuJoCo)            | 10 | 10 | `is_success` | 300 | 100 |
| `libero_10.yaml`          | franka_panda   | libero_10 / LIBERO-Long (MuJoCo) | 10 | 10 | `is_success` | 520 | 100 |
| `metaworld_mt50.yaml`     | sawyer         | metaworld (MuJoCo via lerobot)  | 50 |  5 | `success`    | 200 | 250 |
| `aloha_transfer_cube.yaml` | aloha_bimanual | aloha_transfer_cube (gym-aloha) |  1 | 50 | `is_success` | 400 |  50 |
| `aloha_insertion.yaml`    | aloha_bimanual | aloha_insertion (gym-aloha)     |  1 | 50 | `is_success` | 400 |  50 |
| `pusht.yaml`              | pusht_2d       | pusht (gym-pusht pymunk)        |  1 | 50 | `is_success` | 300 |  50 |
| `maniskill3_pick_place.yaml`       | google_robot   | maniskill3 (SAPIEN, GPU)                                                |  2 |  5 | `success`    | 100–200 |  10 |
| `maniskill3_franka_pick_cube.yaml` | franka_panda   | maniskill3 PickCube-v1 (SAPIEN, GPU)                                    |  1 | 10 | `success`    | 100 |  10 |
| `simpler_env_widowx.yaml`          | widowx         | simpler_env / Bridge V2 (SAPIEN via ManiSkill)                          |  4 |  5 | `success`    |  60 |  20 |
| `robocasa_pnp.yaml`                | panda_mobile   | robocasa/PickPlaceCounterToCabinet (MuJoCo via robosuite + kitchen fork) |  1 | 10 | `is_success` | 500 |  10 |
| `gr1_tabletop.yaml`                | gr1            | robocasa/gr1/PnPCupToDrawerClose (MuJoCo via robosuite + GR1 fork)       |  1 | 10 | `is_success` | 720 |  10 |

Per-suite `max_steps` mirrors the upstream `lerobot.envs.libero.TASK_SUITE_MAX_STEPS`
table for the LIBERO suites and the ACT / Diffusion Policy paper protocols for
the others. `metaworld_mt50` runs with 5 episodes per scene by default to keep
the wall-clock under ~4 h on a single GPU; bump per-scene `n_episodes` together
for a paper-equivalent reproduction. `maniskill3_pick_place` has two scenes with
different `max_steps` per task (PickCube-v1=100, StackCube-v1=200) — the suite
aggregator uses `max(scene.task.max_steps)` for the suite-level bound (Task 10).

The four SAPIEN rows (`maniskill3_*` and `simpler_env_*`) require opt-in extras
(ADR-0010). Without `uv sync --group maniskill3` (or `simpler-env`)
`openral benchmark run` will raise a typed `ROSConfigError` at lazy import time
with the install hint. The simpler-env package has no PyPI release; after
`uv sync --group simpler-env` users must also run:

```
uv run pip install "simpler-env @ git+https://github.com/simpler-env/SimplerEnv.git@maniskill3"
```

## Adding a new benchmark

1. Pick the published protocol you want to reproduce. Don't invent protocols —
   the value of `openral benchmark report` is apples-to-apples across papers.
2. Drop a `<id>.yaml` here whose YAML root is a bare list of `BenchmarkScene`
   mappings (ADR-0042) — use the YAML-anchor pattern from
   `libero_spatial.yaml` to keep per-scene fields DRY.
3. Add a test under `tests/unit/test_benchmark_schemas.py` that loads the
   fixture via `load_benchmark_suite` + `raise_on_invalid_suite` — CLAUDE.md
   §1.11 (real fixtures, no placeholders).
4. Single-scene paper claim instead of a suite? Drop a `BenchmarkScene` YAML
   under [`scenes/benchmark/`](../scenes/benchmark/) and run it with
   `openral benchmark scene --config scenes/benchmark/<id>.yaml --rskill <name>`
   (ADR-0041).
