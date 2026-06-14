# ADR-0042: Drop `BenchmarkSpec` for a bare `list[BenchmarkScene]`

- Status: **Accepted**
- Date: 2026-06-09
- Deciders: TSC, sim-WG
- Related: [ADR-0009](0009-separate-sim-and-benchmarking.md) (the
  original `BenchmarkSpec` / `ProtocolSpec` proposal — this ADR
  rescinds the suite Pydantic model while keeping the eval contract
  unchanged); [ADR-0041](0041-scene-three-tier-hierarchy.md) (the
  three-tier scene hierarchy whose Task 10 flattened `BenchmarkSpec`
  to `{id, tasks: list[BenchmarkScene], metadata}` — the immediate
  precursor that exposed `BenchmarkSpec` as a near-empty wrapper);
  CLAUDE.md §1.3 (types are the contract), §1.4 (explicit beats
  implicit), §1.6 (schemas evolve, but never silently), §1.11 (real
  components, not mocks), §1.13 (no duplicate helpers).

## Context

ADR-0041 / Task 10 flattened `BenchmarkSpec` from
`{robot_id, scene, protocol, tasks: list[TaskSpec]}` to
`{id, tasks: list[BenchmarkScene], metadata}`. Each `BenchmarkScene`
became self-contained — carrying its own `robot_id`, `task`,
`n_episodes`, `seed`, and `metadata: BenchmarkMetadata` (paper +
honest_scope provenance) — and the suite wrapper retained only:

1. `id: str` — a stable suite identifier (e.g. `"libero_spatial"`)
   that doubled as the JSON filename under `rskills/<vla>/eval/<id>.json`.
2. `metadata: dict[str, object]` — a free-form bag holding the suite
   display name (`metadata["suite"]` → `"LIBERO-Spatial"`),
   a human-readable simulator description
   (`metadata["simulator"]` → `"gym-pusht (pymunk 2-D)"`), and an
   optional `metadata["arxiv"]` URL.
3. `tasks: list[BenchmarkScene]` — the actual eval payload.
4. `model_post_init` enforcing five suite-level invariants (non-empty,
   unique `task.id`s, uniform `robot_id` / `n_episodes` / `seed` /
   `metadata` across the list).

The result is a class whose only structural job is *to be a list with
two adjacent labels and a validator*. Three specific failures motivated
removing it entirely:

1. **The `id` field duplicates the filename.** Every
   `benchmarks/<id>.yaml` carries a top-level `id: <id>` matching its
   own filename stem. The CLI already addresses suites by filename
   (`openral benchmark run --suite libero_spatial` →
   `benchmarks/libero_spatial.yaml`). The YAML field is a redundancy
   that goes wrong when authors rename a file without updating the
   field, or vice versa. The validator never catches it because the
   only ground truth is the path the user typed.

2. **The free-form `metadata` dict is a structurally unbounded surface
   on a typed contract.** `BenchmarkSpec.metadata` is declared as
   `dict[str, object]` so the aggregator does `spec.metadata.get(...)`
   with string keys and runtime `isinstance` narrowing — exactly the
   shape Pydantic v2 was adopted to retire (CLAUDE.md §1.3). The two
   strings the aggregator actually emits (`benchmark.name`,
   `benchmark.simulator`) are paper-comparison labels that belong with
   the per-paper provenance block (`BenchmarkMetadata`) — they describe
   the published protocol the scene reproduces, not the wrapper that
   collects them.

3. **The invariants are list-shape invariants, not class invariants.**
   "Every entry shares `robot_id`" and "all `task.id`s are unique" are
   properties of a `list[BenchmarkScene]`. Encoding them as
   `model_post_init` on a wrapper class hides them from callers that
   build the list programmatically (the unit tests, `_make_tiny_libero_spec`
   in `tests/sim/`, future scripted suite generators) — those callers
   either round-trip through the Pydantic wrapper (cost: re-validation
   + a constructor call) or skip the invariants entirely.

### Non-goals

- This ADR does **not** change the on-disk `schema_version` (CLAUDE.md
  §1.6 — pre-publish, the surface evolves in place; the file stays
  `"0.1"`).
- It does **not** change the `RSkillEvalResult` schema or its
  filename convention (`rskills/<vla>/eval/<suite_id>.json`).
- It does **not** alter `BenchmarkScene` or `BenchmarkMetadata` field
  semantics — only adds two optional display fields to the latter.
- It does **not** alter `ProtocolSpec`, which remains an independent
  schema for ADR / report tooling that quotes published protocols
  verbatim.
- It does **not** introduce a versioned migrator. The 13 in-tree
  `benchmarks/*.yaml` files are rewritten in the same commit; there is
  no released artefact to migrate.

## Decision

**Delete `BenchmarkSpec`**. A benchmark suite is a bare
`list[BenchmarkScene]` on disk and in memory. The suite identifier is
the filename basename. Two paper-comparison labels move from the
free-form suite dict onto the per-scene `BenchmarkMetadata` block.

### Shape change

```python
# Before (post-Task-10, pre-ADR-0042):
class BenchmarkSpec(BaseModel):
    id: str
    tasks: list[BenchmarkScene]
    metadata: dict[str, object]
    # model_post_init enforces suite invariants

# After (ADR-0042):
# (no class — a benchmark suite is just list[BenchmarkScene])

class BenchmarkMetadata(BaseModel):
    paper: str
    honest_scope: str
    display_name: str | None = None     # was BenchmarkSpec.metadata["suite"]
    simulator: str | None = None        # was BenchmarkSpec.metadata["simulator"]
```

### On-disk YAML shape

```yaml
# benchmarks/libero_spatial.yaml  — bare YAML list, no top-level dict
- &libero_scene
  scene: &scene_block
    id: libero_spatial
    backend: mujoco
    observation_height: 256
    observation_width: 256
  task: &task_proto
    id: libero_spatial/0
    scene_id: libero_spatial
    max_steps: 280
    success_key: is_success
  robot_id: franka_panda
  n_episodes: 10
  seed: 0
  metadata: &meta_block
    paper: "https://arxiv.org/abs/2306.03310"
    honest_scope: "10 episodes per task across all 10 LIBERO-Spatial tasks (100 rollouts total)."
    display_name: "LIBERO-Spatial"
    simulator: "LIBERO (MuJoCo)"
- <<: *libero_scene
  task:
    <<: *task_proto
    id: libero_spatial/1
# … etc
```

The top-level `id:` and `metadata:` blocks are gone. The YAML anchor
pattern (`&libero_scene` / `<<: *libero_scene`) carries through
unchanged — DRY-ness was never tied to `BenchmarkSpec`.

### Loader API

```python
# python/core/src/openral_core/loaders.py

def load_benchmark_suite(path: str | Path) -> list[BenchmarkScene]:
    """Load benchmarks/<id>.yaml — a bare YAML list of BenchmarkScene entries.

    Suite-id is derived from the filename stem at the call site. Calls
    `raise_on_invalid_suite(scenes, suite_id=Path(path).stem)` so the
    same five invariants the deleted `BenchmarkSpec.model_post_init`
    enforced still hold.
    """

def raise_on_invalid_suite(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
) -> None:
    """Suite-level invariants: non-empty, unique task ids, uniform
    robot_id (non-None) / n_episodes / seed / metadata across the list.
    """
```

`raise_on_invalid_suite` is public — callers building suites
programmatically (sim tests, future scripted generators) validate
without round-tripping through a Pydantic model. The invariants are
exactly the five from the deleted `BenchmarkSpec.model_post_init`;
their error messages name the offending `suite_id` so the rejection
points at the right `benchmarks/*.yaml`.

### Runner API

```python
# python/sim/src/openral_sim/benchmark.py

def run_benchmark(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
    vla: VLASpec,
    device: str | None = None,
    save_dir: str | None = None,
) -> tuple[RSkillEvalResult, list[EpisodeResult]]:
    """Run a benchmark suite end-to-end against one VLA."""
```

Callers pass `(scenes, suite_id)` rather than a `BenchmarkSpec`.
`suite_id` is the only thing the deleted class added that the list
cannot represent itself; making it a keyword-only argument keeps the
runner signature self-documenting.

### Aggregator changes

The `_aggregate_results` rollup that emits `RSkillEvalResult` was
reading three fields off `spec.metadata`. Their replacements:

| Old source                                     | New source                                          |
| ---------------------------------------------- | --------------------------------------------------- |
| `spec.metadata.get("suite", spec.id)`          | `first.metadata.display_name or suite_id`           |
| `spec.metadata.get("simulator", first.scene.id)` | `first.metadata.simulator or first.scene.id`      |
| `spec.metadata.get("arxiv")`                   | `first.metadata.paper if "arxiv.org/" in paper else None` |

The arxiv auto-derivation mirrors the existing behaviour of
`_aggregate_scene_results` (the single-scene sibling added by
ADR-0041 Task 9), so paper-comparison reports built on top of both
runner entrypoints stay uniform.

### CLI changes

`openral benchmark run --suite <id-or-path>` is unchanged at the user
surface. Internally, `_resolve_benchmark_spec(suite, benchmarks_dir)`
becomes `_resolve_benchmark_suite(suite, benchmarks_dir)` and returns
`tuple[list[BenchmarkScene], str]` (the scenes plus the derived
suite-id). `openral benchmark list` continues to walk
`benchmarks/*.yaml` and emit basenames; that path was always
filename-driven.

`openral benchmark scene --config <BenchmarkScene>` is untouched — it
already accepted a single `BenchmarkScene` YAML, never a suite.

### Tests

- `tests/unit/test_benchmark_schemas.py` rewritten: the
  `BenchmarkSpec` happy-path / invariants tests become
  `raise_on_invalid_suite` + `load_benchmark_suite` tests.
  The 13-row catalogue parametric stays — now via `load_benchmark_suite`.
- `tests/unit/test_benchmark_runner.py` switches from
  `_mini_spec()` returning `BenchmarkSpec` to returning
  `tuple[list[BenchmarkScene], str]`.
- `tests/unit/test_benchmark_aggregator_byte_identical.py`
  **deleted** along with its 13 baseline JSONs under
  `tests/unit/fixtures/benchmark_eval_baseline/`. The test was a
  one-shot Task-10 regression guard; with the `BenchmarkSpec` shape
  gone there is no pre-refactor surface to compare against. The new
  catalogue parametric (`test_benchmarks_catalogue_fixture_is_a_valid_benchmark_spec`,
  ironically retained name) plus the runner tests cover the same
  ground.
- `tests/sim/test_*_cli_benchmark*.py` rewrite their `_make_tiny_*_spec`
  helpers to emit a bare YAML list — the sim CLI tests exercise the
  full new ingest path.

## Consequences

### Positive

- **One fewer class on the public surface.** `BenchmarkSpec` was a
  validator-on-a-list with two free-form labels. Deleting it removes
  ~170 lines of schema, one `from_yaml` classmethod, one model_post_init
  validator, one JSON Schema export, and one entry from
  `openral_core.__all__`.
- **Suite identity is unforgeable.** `suite_id` is the filename stem;
  there is no `id:` field that can desync. Renaming `benchmarks/foo.yaml`
  to `benchmarks/bar.yaml` is a one-step rename — no editing.
- **Display labels travel with their paper provenance.**
  `BenchmarkMetadata.{display_name, simulator}` live alongside `paper`
  + `honest_scope` on every `BenchmarkScene`. A scene reused in a
  different suite carries its labels with it.
- **Invariants are reusable.** `raise_on_invalid_suite(scenes,
  suite_id=...)` validates any `list[BenchmarkScene]` regardless of
  origin — sim tests that build suites programmatically no longer have
  to construct a Pydantic wrapper just to get the validator.
- **Aggregator output is structured all the way down.** The auto-derived
  `arxiv` URL mirrors `_aggregate_scene_results`, so the two runner
  entrypoints emit identically-shaped `RSkillEvalResult` JSONs.

### Negative

- **Breaking change for any consumer importing `BenchmarkSpec`.**
  `from openral_core import BenchmarkSpec` no longer works. Consumers
  switch to `from openral_core import load_benchmark_suite,
  raise_on_invalid_suite` (or accept the bare `list[BenchmarkScene]`
  shape). No deprecation shim — the symbol is removed in the same
  commit so the build fails loudly.
- **Byte-identicality baselines deleted.** The 13 JSONs under
  `tests/unit/fixtures/benchmark_eval_baseline/` were captured against
  the pre-Task-10 `BenchmarkSpec`. Their content is now stale by
  construction (paths through `_aggregate_results` differ). Re-capturing
  them against the new aggregator would only re-pin the post-ADR-0042
  shape; the dedicated catalogue + runner tests already cover that.
- **YAML migration tax (again).** All 13 `benchmarks/*.yaml` files are
  rewritten in the same commit to drop the top-level dict and inline
  the two display fields onto each scene's `metadata`. Hand-edited
  once; no migration script ships.
- **Suite-level `notes:` and `arxiv:` fields disappear.** A handful of
  YAMLs carried free-form `metadata.notes` strings that were never read
  by any consumer (only the suite name, simulator, and arxiv URL ever
  surfaced in `RSkillEvalResult`). The notes are preserved as YAML
  comments at the top of each rewritten file — visible to authors,
  invisible to the loader.

### Neutral

- **`ProtocolSpec` survives unchanged.** Independent schema, no
  embedding in benchmark suites since Task 10. Kept for ADR drafts and
  benchmark-report tooling that wants to quote a published protocol
  outside a suite context.
- **`tasks` field disappears with the wrapper.** The on-disk YAML is
  a bare list now, so there is no `tasks:` key to bikeshed. Code-side
  the list is just `scenes` (variable name) — keeping the old name
  would only confuse new readers about what kind of object it is.

## Implementation status (this branch)

Phased delivery on the `refactor/benchmark-spec-removal` branch
(forked off `refactor/scenes` HEAD after ADR-0041 Task 16):

| Task | Scope | Status |
| --- | --- | --- |
| 1 | This ADR | done |
| 2 | Schema: add `BenchmarkMetadata.{display_name, simulator}`; delete `BenchmarkSpec`; export `load_benchmark_suite` + `raise_on_invalid_suite` | done |
| 3 | Runner: `run_benchmark(scenes, *, suite_id, vla, …)`; aggregator switches to per-scene metadata | done |
| 4 | CLI: `_resolve_benchmark_spec` → `_resolve_benchmark_suite`; `--dry-run` / `--out` paths updated | done |
| 5 | Rewrite all 13 `benchmarks/*.yaml` as bare lists with display fields on per-scene `metadata` | done |
| 6 | Tests: rewrite schema + runner tests; rewrite tiny-suite helpers in sim tests; delete byte-identicality fixtures + test | done |
| 7 | Docs: `scenes/README.md`, `benchmarks/README.md`, `docs/reference/sim-environments.md`, `docs/METHODS.md`, repo state map, regenerated JSON Schema export | done |

Regression coverage:

- `tests/unit/test_benchmark_schemas.py` — `load_benchmark_suite` happy
  path + `raise_on_invalid_suite` invariants (non-empty, unique ids,
  uniform robot_id/n_episodes/seed/metadata, first-scene robot_id
  non-None) + 13-row parametrised catalogue load.
- `tests/unit/test_benchmark_runner.py` — `_aggregate_results` rollup
  + `run_benchmark` end-to-end against the mock scene + zero policy
  (2 tasks × 3 episodes = 6 episodes without GPU).
- `tests/sim/test_franka_panda_smolvla_cli_benchmark.py` +
  `tests/sim/test_panda_mobile_pi05_cli_benchmark_robocasa.py` — real
  CLI invocation against LIBERO + SmolVLA and RoboCasa + pi05, with
  the tiny-suite helpers rewriting to a bare YAML list and the
  manifest writeback assertion preserved.

## References

- ADR-0009 — the original `BenchmarkSpec` / `ProtocolSpec` proposal.
  This ADR rescinds the suite class while leaving the eval contract
  intact.
- ADR-0041 — three-tier scene hierarchy. Task 10 flattened
  `BenchmarkSpec` to the near-empty wrapper that ADR-0042 removes.
- CLAUDE.md §1.3 / §1.4 / §1.6 / §1.11 / §1.13.
