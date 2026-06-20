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
| `libero_spatial.yaml`     | franka_panda   | libero_spatial (MuJoCo)         | 10 | 50 | `is_success` | 280 | 500 |
| `libero_object.yaml`      | franka_panda   | libero_object (MuJoCo)          | 10 | 50 | `is_success` | 280 | 500 |
| `libero_goal.yaml`        | franka_panda   | libero_goal (MuJoCo)            | 10 | 50 | `is_success` | 300 | 500 |
| `libero_10.yaml`          | franka_panda   | libero_10 / LIBERO-Long (MuJoCo) | 10 | 50 | `is_success` | 520 | 500 |
| `metaworld_mt10.yaml`     | sawyer         | metaworld (MuJoCo via lerobot)  | 10 | 10 | `success`    | 500 | 100 |
| `metaworld_mt50.yaml`     | sawyer         | metaworld (MuJoCo via lerobot)  | 50 |  5 | `success`    | 500 | 250 |
| `aloha.yaml`              | aloha_bimanual | aloha_transfer_cube + aloha_insertion (gym-aloha) |  2 | 50 | `is_success` | 400 | 100 |
| `pusht.yaml`              | pusht_2d       | pusht (gym-pusht pymunk)        |  1 | 50 | `is_success` | 300 |  50 |
| `maniskill3_panda.yaml`            | franka_panda   | maniskill3 — Panda tabletop, 7 tasks (SAPIEN, GPU)                      |  7 | 100 | `is_success` | 50–200 | 700 |
| `simpler_env_widowx.yaml`          | widowx         | simpler_env / Bridge V2 (SAPIEN via ManiSkill)                          |  4 |  5 | `success`    |  60 |  20 |
| `robocasa_pnp.yaml`                | panda_mobile   | robocasa/PickPlaceCounterToCabinet (MuJoCo via robosuite + kitchen fork) |  1 | 10 | `is_success` | 500 |  10 |
| `gr1_tabletop.yaml`                | gr1            | robocasa/gr1/PnPCupToDrawerClose (MuJoCo via robosuite + GR1 fork)       |  1 | 10 | `is_success` | 720 |  10 |
| `robotwin.yaml`                    | aloha_agilex   | robotwin (SAPIEN via lerobot, py3.10 sidecar)                           |  5 | 100 | `is_success` | 300 | 500 |
| `rlbench.yaml`                     | franka_panda   | RLBench PerAct subset: open_drawer / meat_off_grill / close_jar (CoppeliaSim/PyRep, py3.10 sidecar) |  3 | 25 | `is_success` | 25 |  75 |

Per-suite `max_steps` mirrors the upstream `lerobot.envs.libero.TASK_SUITE_MAX_STEPS`
table for the LIBERO suites and the ACT / Diffusion Policy paper protocols for
the others. `metaworld_mt50` runs with 5 episodes per scene by default to keep
the wall-clock under ~4 h on a single GPU; bump per-scene `n_episodes` together
for a paper-equivalent reproduction. `maniskill3_panda` has per-task `max_steps`
that differ across its 7 scenes (PickCube-v1=50 … PlugCharger-v1=200) — the suite
aggregator uses `max(scene.task.max_steps)` for the suite-level bound (Task 10).
It is OpenRAL-curated (ManiSkill3 has no single canonical suite); `--rskill`
auto-filters to the tasks a policy declares, so it stays runnable as task-matched
MS3 Panda policies land.

The ManiSkill/SimplerEnv SAPIEN rows (`maniskill3_*` and `simpler_env_*`) require
opt-in extras (ADR-0010). Without `uv sync --group maniskill3` (or `simpler-env`)
`openral benchmark run` will raise a typed `ROSConfigError` at lazy import time
with the install hint. The simpler-env package has no PyPI release; after
`uv sync --group simpler-env` users must also run:

```
uv run pip install "simpler-env @ git+https://github.com/simpler-env/SimplerEnv.git@maniskill3"
```

`robotwin.yaml` is the first **dual-arm** suite (RoboTwin 2.0, SAPIEN; ADR-0061). It
runs out-of-process through a py3.10 sidecar (`tools/robotwin_sidecar.py`) because
RoboTwin's SAPIEN/CuRobo/pytorch3d stack is incompatible with the py3.12 workspace;
`uv sync --group robotwin --inexact` installs only the openral-side wire (pyzmq +
msgpack). The heavy lerobot-main + RoboTwin + asset venv is externally provisioned
(`OPENRAL_ROBOTWIN_AUTO_PROVISION=1` or the manual recipe in ADR-0061). Task-matched
rSkill: [`rskills/smolvla-robotwin`](../rskills/smolvla-robotwin) (the official
`lerobot/smolvla_robotwin` checkpoint). The shown 5-task slice is a representative
subset of RoboTwin's 50 tasks; the `smolvla-robotwin` checkpoint is multi-task so it
covers all of them.

The `rlbench.yaml` row (ADR-0062) runs RLBench on **CoppeliaSim/PyRep** — a
proprietary (free-EDU) simulator that is **never vendored** (CLAUDE.md §1.9) and
the released 3D keyframe policies pin the `MohitShridhar/RLBench@peract` fork.
Both the scene and the **3D Diffuser Actor** policy (`rskills/3d-diffuser-actor-rlbench`,
MIT) run in an externally-provisioned py3.10 sidecar venv; the openral side only
needs `uv sync --group rlbench` (pyzmq + msgpack). The scene factory raises a
typed `ROSConfigError` with the full provisioning recipe when the sidecar venv /
`COPPELIASIM_ROOT` are absent. Verified live on an 8 GB Ada GPU host
(open_drawer 4/4, meat_off_grill 3/3, close_jar solved).

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
