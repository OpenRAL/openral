# ADR-0041: Scene three-tier hierarchy (DeployScene / SimScene / BenchmarkScene)

- Status: **Accepted**
- Date: 2026-06-08
- Deciders: TSC, sim-WG
- Related: [ADR-0002](0002-eval-and-sim-environments.md) (eval and sim
  environments — the single-file `SceneEnvironment` shape this ADR
  retires); [ADR-0009](0009-separate-sim-and-benchmarking.md)
  (rSkill eval contract — `BenchmarkSpec` / `RSkillEvalResult` /
  `openral benchmark run`); [ADR-0010](0010-inference-runner.md)
  (inference runner amendment 1 — the per-task seed-loop semantics
  now formalised on `BenchmarkScene`);
  [ADR-0019](0019-rosbag2-lerobot-dataset-bridge.md) (dataset bridge —
  state/action dim contracts that travel via the rSkill manifest, not
  the scene); [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md)
  (deploy-sim scene attach for arms — the no-task, env-only scene shape
  this ADR formalises as `DeployScene`); CLAUDE.md §1.3 (types are the
  contract), §1.4 (explicit beats implicit), §1.6 (schemas evolve, but
  never silently), §1.11 (real components, not mocks).

## Context

`SceneEnvironment` (ADR-0002) was a single Pydantic model serving every
config-driven entrypoint in the codebase — `openral sim run`,
`openral deploy sim`, `openral benchmark run`, the audit tool, the
tutorials. The model accreted optional fields one entrypoint at a time:
`n_episodes` / `seed` / `record_video` for sim, `task: TaskSpec | None`
for deploy, and (via `BenchmarkSpec`) a `protocol` / per-task `seeds`
list for benchmark eval. The result was a permissive schema that
silently widened across tiers: a YAML missing `task` was a legal deploy
config and a silently broken sim config; a YAML with `n_episodes: 500`
+ rich metadata was a legal sim config and a silently sub-canonical
benchmark config; the audit tool, tutorials, and Justfile each had to
pick the right one by convention, not by type.

Two specific failures motivated the refactor:

1. **rSkill names leaked into scene filenames.** `scenes/benchmarks/`
   contained 21 YAMLs named after `<rskill>_<scene>.yaml`
   (`smolvla_libero_spatial.yaml`, `pi05_robocasa_pnp_nf4.yaml`, …). Pairing
   any other rSkill with the same scene meant duplicating the YAML or
   editing in place. The scene/policy axis cross-product was muddied at
   the filesystem layer.

2. **The CLI couldn't tell tiers apart.** `openral sim run` happily
   loaded a benchmark YAML and silently dropped its `n_episodes` /
   `seed` / `metadata.paper`. `openral deploy sim` happily loaded a sim
   YAML and silently ignored its `task`. The user got the *wrong* run
   semantics with no error.

3. **`BenchmarkSpec` carried duplicate scene metadata.** A suite YAML
   embedded `robot_id` + `scene` + `protocol` at the top level *and*
   per-task `TaskSpec` entries. When the protocol's `max_steps`
   contradicted a per-task value, the loader silently preferred one over
   the other. The per-task list was syntactically `TaskSpec` but
   semantically a `(scene, task, n_episodes, seed, metadata)` tuple —
   the same information `SceneEnvironment` carried in a flat YAML.

### Non-goals

- This ADR does **not** change the `openral deploy run` real-HAL
  entrypoint, which never took a scene config (ADR-0032).
- It does **not** alter the on-disk `schema_version` (per CLAUDE.md
  §1.6 the surface evolves in place pre-publish; the file stays at
  `"0.1"`).
- It does **not** add new actuation-path code; `Skill` / `Reasoner` /
  `Safety` layers are untouched.
- It does **not** introduce a `rskill_ref` field on any scene tier;
  every entrypoint takes `--rskill` on the CLI so scenes stay
  rSkill-agnostic.

## Decision

**Replace the single `SceneEnvironment` model with three Pydantic
models forming a strict inheritance chain:**

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
```

each backing exactly one CLI entrypoint, each with its own subdirectory
under `scenes/`, and each strictly rejected by the *other* CLIs (no
silent widening across tiers).

```
scenes/
├── deploy/                      # DeployScene YAMLs — env-only, no task
│   ├── libero_pnp.yaml
│   ├── openarm_tabletop.yaml
│   ├── robocasa_pnp.yaml
│   └── so101_box.yaml
├── sim/                         # SimScene YAMLs — env + task, no metadata
│   ├── franka_libero_pnp.yaml
│   ├── libero_spatial.yaml
│   ├── openarm_tabletop.yaml
│   ├── robocasa_gr1_pnp_cup_to_drawer.yaml
│   ├── robocasa_panda_mobile_kitchen.yaml
│   ├── robocasa_pnp.yaml
│   ├── so101_tube_insertion.yaml
│   └── tabletop_cube_push.yaml
└── benchmark/                   # BenchmarkScene YAMLs — env + task + n_episodes + seed + metadata
    ├── aloha_insertion.yaml
    ├── aloha_transfer_cube.yaml
    ├── libero_spatial.yaml
    ├── maniskill_pick_cube.yaml
    ├── metaworld_push.yaml
    ├── pusht.yaml
    └── widowx_carrot_on_plate.yaml
```

### Schema contracts

| Field            | DeployScene | SimScene             | BenchmarkScene             |
| ---------------- | ----------- | -------------------- | -------------------------- |
| `scene: SceneSpec`     | required    | required             | required                   |
| `robot_id: str | None`  | optional    | optional             | **required (non-None)**    |
| `task: TaskSpec | None` | **forbidden** | optional        | **required**               |
| `task.max_steps`       | n/a         | optional             | **required**               |
| `task.success_key`     | n/a         | optional             | **required**               |
| `n_episodes: int`      | not used    | optional (defaults 1)| **required**               |
| `seed: int`            | not used    | optional             | **required**               |
| `metadata: BenchmarkMetadata` | **forbidden** | **forbidden** | **required** (`paper` URL + `honest_scope`) |

`TaskSpec` loses the dead `seed` field (seeding lives on the scene
runtime, not per-task — fixed in a separate prior `fix(core)` commit)
and gains optional `max_steps` / `success_key` so SimScene tasks can
omit eval-only fields. `BenchmarkScene.model_post_init` enforces that
`task.max_steps` and `task.success_key` are non-None (no defaults
inherited).

### CLI strictness

A new `openral_core.load_scene_strict(path, expected)` helper is the
single ingest path for every scene-driven CLI. It loads `path` as a
YAML mapping, refuses to widen across tiers, and raises
`ROSConfigError` carrying a redirect message that names the right CLI
command:

| Entrypoint                                    | Accepts          | Rejects (with redirect)            |
| --------------------------------------------- | ---------------- | ---------------------------------- |
| `openral deploy run`                          | (no config)      | n/a                                |
| `openral deploy sim --config <DeployScene>`   | `DeployScene`    | `SimScene` / `BenchmarkScene`      |
| `openral sim run --config <SimScene>`         | `SimScene`       | `DeployScene` / `BenchmarkScene`   |
| `openral benchmark scene --config <BenchmarkScene>` | `BenchmarkScene` | `DeployScene` / `SimScene`   |
| `openral benchmark run --suite <id>`          | `BenchmarkSpec`  | n/a (suite-file loader)            |

`mypy --strict` overloads on `load_scene_strict` narrow the return type
to the exact expected tier so call sites cannot accidentally use a
field that is `None` at runtime. Tier detection is by structural
matching of the YAML's top-level keys (presence of `metadata` +
`n_episodes` + `seed` ⇒ BenchmarkScene; absence of `task` ⇒
DeployScene; otherwise SimScene). The matching is conservative — a
DeployScene YAML that grows a `task:` key becomes a SimScene by virtue
of the new key, and the loader's existing extra-key strictness catches
misnamed keys early.

### `BenchmarkSpec` convergence (C2)

A multi-scene suite is exactly *an ordered collection of reproducible
single-scene evals*. To reflect that, `BenchmarkSpec` is flattened to
three fields:

```python
class BenchmarkSpec(BaseModel):
    id: str
    tasks: list[BenchmarkScene]   # field name kept for backward semantic
    metadata: dict[str, Any]      # free-form suite-level provenance
```

`model_post_init` enforces suite-level invariants:

- `tasks` non-empty
- per-task `task.id` unique within the suite
- every scene shares `robot_id` (non-None), `n_episodes`, `seed`, and
  `metadata: BenchmarkMetadata`

Per-task `success_key` and `task.max_steps` *may* differ across scenes
(maniskill3_pick_place ships `PickCube-v1=100` + `StackCube-v1=200`,
which the aggregator now reports as a worst-case `max_steps=200` in the
`RSkillEvalResult.protocol`). The standalone `ProtocolSpec` schema is
retained for ADR-0009 report tooling but no longer embedded in
`BenchmarkSpec`.

### `run_benchmark_scene`: the single-scene sibling

`openral benchmark scene --config <BenchmarkScene>` (added by Task 9)
fills the gap between `openral sim run` (sim-only, drops eval
metadata) and `openral benchmark run --suite <id>` (multi-scene). It
iterates `range(scene.seed, scene.seed + scene.n_episodes)` against
the one `(scene, task)` pair, writes the same `RSkillEvalResult` JSON
shape as the suite runner (so `openral benchmark report` does not need
to distinguish entrypoints), and surgically updates the rSkill
manifest's `benchmarks.<scene_id>` field with the average success rate
(opt-out via `--no-update-manifest`).

## Consequences

### Positive

- **No silent widening.** Each CLI accepts exactly its tier; the
  loader's redirect message names the right command so users hit the
  right error in <5 s of staring at a stack trace.
- **rSkills are a CLI flag, not a filename.** Pairing a new rSkill
  with an existing scene is `--rskill <id>`, not a YAML duplicate. The
  cross-product of (scene × rSkill) is the catalogue
  (`tools/audit_sim_configs.py`), not the filesystem layout.
- **`BenchmarkSpec` is a list of reproducible units.** No more
  top-level `robot_id` / `scene` / `protocol` contradicting per-task
  values. The `_aggregate_results` rollup is byte-identical to the
  pre-refactor JSON on all 13 in-tree suites
  (`tests/unit/test_benchmark_aggregator_byte_identical.py`, gated by
  fixtures in `tests/unit/fixtures/benchmark_eval_baseline/`).
- **Tier-aware audit.** `tools/audit_sim_configs.py` now carries a
  per-row `run_mode: Literal["sim", "benchmark", "deploy"]` and
  dispatches to the matching CLI (`openral sim run`, `openral
  benchmark scene --no-update-manifest --n-episodes 1`, `openral
  deploy sim --no-dashboard` + SIGINT probe), mirroring the
  Justfile's `sim-*` recipes. Catalogue is pure scene×rSkill pairs
  that exist in tree — scenes without a matching in-tree rSkill are
  not represented (schema-load coverage stays in
  `tests/unit/test_examples_sim_configs_load.py`).
- **`mypy --strict` clean across the boundary.** The
  `load_scene_strict` overloads remove every existing `# type: ignore`
  on the scene-loader path.

### Negative

- **Breaking change for any consumer importing `SceneEnvironment`.**
  Pre-refactor code that did `from openral_core import SceneEnvironment`
  must switch to `SimScene` / `DeployScene` / `BenchmarkScene`
  (whichever role the YAML plays). The symbol is removed, not
  deprecated, so the build fails loudly.
- **Three Pydantic models instead of one.** Marginal duplication on
  the common `scene: SceneSpec` block; a small amount of overlap on
  `robot_id` / `task` field declarations. Accepted as the cost of
  per-tier strictness.
- **One additional CLI verb** (`openral benchmark scene`). Mirrors
  `openral sim run` semantically; documented as the single-scene
  sibling of `openral benchmark run --suite`.
- **YAML migration tax.** All 13 `benchmarks/*.yaml` suite files were
  rewritten to inline `BenchmarkScene` entries via YAML anchors
  (`&scene` / `&task_proto` / `<<: *libero_scene`). Hand-edited
  once; no migration script ships (per CLAUDE.md §1.6, the on-disk
  `schema_version` stays at `"0.1"`).

### Neutral

- The scene-id binds to the `@SCENES.register("…")` Python factory
  key; the YAML's `scene.id` is the source of truth, not the filename.
  Renaming a YAML on disk is a free operation; renaming `scene.id`
  requires a matching factory rename.
- Historical ADRs (ADR-0007, 0010, 0017, 0019, 0025, 0026, 0038) keep
  their stale `scenes/benchmarks/<rskill>_<scene>.yaml` references —
  ADRs are frozen historical text. Forward-looking ADRs (this one
  included) use the new paths.

## Implementation status (this branch)

Phased delivery on the `refactor/scenes` branch, one PR (#274):

| Task | Scope | Status |
| --- | --- | --- |
| 1 | Remove dead `TaskSpec.seed` field | done |
| 2 | `TaskSpec.max_steps` / `success_key` optional | done |
| 3 | Add `DeployScene` / `SimScene` / `BenchmarkScene` / `BenchmarkMetadata` Pydantic models | done |
| 4 | Add `load_scene_strict()` typed-overload helper | done |
| 5 | Create `scenes/benchmark/*.yaml` (7 files) | done |
| 6 | Create `scenes/sim/*.yaml` (9 files) and `scenes/deploy/*.yaml` (4 files) | done |
| 7 | Delete `scenes/benchmarks/` (21 YAMLs) and `scenes/native/` (4 YAMLs + 1 BDDL) | done |
| 8 | Migrate every `SceneEnvironment` callsite (3 production + 8 tests) to the right tier | done |
| 9 | Add `openral benchmark scene --config <BenchmarkScene>` CLI + `run_benchmark_scene` runner | done |
| 10 | Flatten `BenchmarkSpec` to `list[BenchmarkScene]` + migrate 13 `benchmarks/*.yaml` files | done (superseded by ADR-0042) |
| 11 | Justfile `sim-*` recipes repointed at the new layout | done |
| 12 | `tools/audit_sim_configs.py` catalogue rewritten with `run_mode` dispatch | done |
| 13 | This ADR + tier-aware `scenes/README.md` + tutorial rewrite | in progress |

Regression coverage:

- `tests/unit/test_load_scene_strict.py` — 15 tests asserting redirect
  behaviour for every (got, expected) mismatch.
- `tests/unit/test_scene_tier_schemas.py` — per-tier construction +
  invariant tests for `DeployScene` / `SimScene` / `BenchmarkScene` /
  `BenchmarkMetadata`.
- `tests/unit/test_run_benchmark_scene.py` (4) + `test_cli_benchmark_scene.py` (5) —
  single-scene runner end-to-end + CLI wiring.
- `tests/unit/test_benchmark_schemas.py` — 30 tests including the
  10-row parametrised catalogue load of `benchmarks/*.yaml`.
- `tests/unit/test_benchmark_aggregator_byte_identical.py` — 13
  parametrised cases asserting `RSkillEvalResult` JSON output is
  byte-identical across the refactor for every in-tree suite.
- `tests/unit/test_examples_sim_configs_load.py` — per-tier guard that
  every YAML under `scenes/<tier>/` validates as the matching schema.

## Amendments

### 2026-06-08 — `BenchmarkSpec` wrapper deleted by ADR-0042

Task 10 above flattened `BenchmarkSpec` to a near-empty wrapper around
`list[BenchmarkScene]`. [ADR-0042](0042-drop-benchmarkspec.md) then
deleted the wrapper class entirely: `benchmarks/<id>.yaml` is now a
bare YAML list of `BenchmarkScene` mappings at the root, the suite id
is the filename stem, and the five suite invariants previously enforced
in `BenchmarkSpec.model_post_init` moved to a free function
`openral_core.raise_on_invalid_suite(scenes, *, suite_id)` that the new
loader `openral_core.load_benchmark_suite(path)` calls separately. The
`BenchmarkSpec.{from_yaml, model_post_init, byte-identical baseline
fixtures}` and the matching public-symbol export are gone.

Schema-rejection note: the pre-Task-10 `{robot_id, scene, protocol,
tasks, metadata}` wrapper had already been removed when this task
landed; the loader's only legacy redirect is for the post-Task-10
shape (`{id, tasks, metadata}` ⇒ ADR-0042 redirect message).

`run_benchmark` now takes `(scenes, vla, *, suite_id, ...)` and its
aggregator pulls `display_name` / `simulator` from per-scene
`BenchmarkMetadata` instead of a free-form suite-level dict
(`BenchmarkMetadata.display_name` / `.simulator` added in ADR-0042).
`arxiv` is auto-derived from `metadata.paper` when the URL contains
`arxiv.org/`. Output JSON is unchanged for every shipped suite.

## References

- ADR-0002 — eval and sim environments (the `SceneEnvironment` shape
  this ADR retires).
- ADR-0009 — separate sim and benchmarking (`BenchmarkSpec` /
  `RSkillEvalResult` / `openral benchmark run` / `openral benchmark report`).
- ADR-0010 — inference runner, amendment 1 (per-task seed-loop, now
  formalised on `BenchmarkScene`).
- ADR-0034 — deploy-sim scene attach for arms (the no-task, env-only
  shape now formalised as `DeployScene`).
- ADR-0042 — drop `BenchmarkSpec` for a bare `list[BenchmarkScene]`
  (the post-Task-10 wrapper deletion).
- CLAUDE.md §1.3 / §1.4 / §1.6 / §1.11.
- Implementation plan: `docs/superpowers/plans/2026-06-07-scene-hierarchy-refactor.md`.
