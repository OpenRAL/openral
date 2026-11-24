# Selective testing and the test audit

Running the whole suite on every push is unaffordable, so CI maps a git diff to the minimal set of pytest targets that can see the change. Two read-only tools keep the suite fast and meaningful:

| Tool | Question it answers | Entry point |
| --- | --- | --- |
| [`tools/select_tests.py`](https://github.com/OpenRAL/openral/blob/master/tools/select_tests.py) | Given this diff, which tests can observe the change? | `just test-changed` |
| [`tools/audit_tests.py`](https://github.com/OpenRAL/openral/blob/master/tools/audit_tests.py) | Which tests are dead, duplicated or low-signal? | `just test-audit` ([report](test-audit.md)) |

Both pass `mypy --strict tools/` and have unit tests (`tests/unit/test_select_tests.py`, `tests/unit/test_audit_tests.py`). How the workflows schedule the results: [CI overview](ci.md).

## 1. The selector

### Usage

```bash
just test-changed                       # print the plan for origin/master..HEAD
just test-changed-run                   # select, then run only those tests
uv run python tools/select_tests.py --files python/state_adapter/src/openral_state_adapter/_registry.py
uv run python tools/select_tests.py --base origin/master --head HEAD
```

`test-changed-run` mirrors CI's `core_selected` path: it falls back to the full unit suite on a blast-radius change and runs fork-unsafe targets (rule 7) in their own processes with single-thread env vars.

To verify the optional environments locally, the lane verifier syncs each group, runs its targets and fails on any skip:

```bash
uv run python tools/verify_test_envs.py --groups sim opencv lowering
uv run python tools/verify_test_envs.py --include-provisioned --groups qwen-vlm locateanything simpler-env robocasa-gr1
```

Sidecar lanes block until their env vars are present; the script never fakes a proprietary sidecar.

### How it decides

Every decision is explicit, and every selected target carries a human-readable reason.

1. **Blast radius first.** A changed path matching `full_run_globs` in [`tools/test_selection.toml`](https://github.com/OpenRAL/openral/blob/master/tools/test_selection.toml) emits `full_run = true` and CI runs everything: root `pyproject.toml`, `uv.lock`, the `Justfile`, a shared `conftest.py`, the selector's own inputs, the selection-related workflows, and `openral_core` source (the root of the dependency graph). A wrong negative would silently skip a regression, so wide diffs are not second-guessed. A full run also expands every opt-in dependency lane. A source file that maps to no known package is also a full run. The release PR is the one exception, handled in the workflow ([Releasing](releasing.md)).
2. **Dependency graph, derived.** The package graph is read from each `python/<pkg>/pyproject.toml` (`openral-*` deps), so it cannot drift from the workspace.
3. **Transitive dependents.** A change to a package selects the tests of every package that imports it, directly or through a chain.
4. **Two selection paths per affected package.** Its own `tests/` directory, plus every top-level `tests/**` file whose `import openral_*` set intersects the affected packages (most of `core`, `cli`, `rskill`, `runner` and `sim` keep tests in the shared `tests/` tree).
5. **Fixture triggers.** `robots/**`, `rskills/**` and selected `scenes/**` map to tests that load them for real. The `robots/**` mapping reaches into `packages/**` test files that load a real `robots/<id>/robot.yaml`; a new `packages/**` test that reads a real manifest must be added to that mapping.
6. **Dependency lanes.** Selected targets matching `requirement_globs` are also emitted per opt-in group: `sim`, `libero`, `robocasa`, `robocasa-gr1`, `maniskill3`, `simpler-env`, `isaacsim`, `robotwin`, `sidecar-wire`, `dataset`, `rlbench`, `gr00t`, `locateanything`, `qwen-vlm`, `omdet`, `onnx-export`, `clip`, `opencv`, `lowering` (19 lanes). The cheap partition may skip those tests; CI reruns them in the lane with the group installed and fails the lane on no passing tests or any skip that is not a declared capability gap ([lane policy](#lane-policy-what-a-skip-is-allowed-to-mean)).
    - A directory target (a package's `tests/` or an `extra_triggers` value) counts as selecting the lane files inside it; the selector expands it to concrete files.
    - A lane is assigned per file, so **one file must not need two mutually exclusive groups** (LIBERO's `robosuite==1.4` versus RoboCasa/OpenArm's `>=1.5`). If a new test needs a different optional stack from its file's neighbours, give it its own file.
    - A lane declared with no globs can never run; `tests/unit/test_lane_report.py` fails it.
    - The `sim` lane also installs `dataset`. Wire-only sidecar tests use `sidecar-wire`; sidecar-backed lanes keep separate names because their preflights differ.
7. **Fork-isolated tests run in their own process.** Targets matching `isolate_globs` cannot share an interpreter with a sibling: dataset tests that fork lerobot's `compute_stats` pool crash at interpreter exit once numpy/torch threadpools exist, and EGL/robosuite env creators lose their GL context when a sibling env is torn down. The selector moves them to `isolated_targets`; full runs `--ignore` them and run each alone, and each lane does the same under its own `--group`.
8. **Ignored domains.** `cpp/**` is covered by `test-ros2` (colcon/ctest), so a pure-C++ change selects nothing here.

### Worked examples

| Change | Result |
| --- | --- |
| `python/state_adapter/src/openral_state_adapter/_registry.py` | `python/state_adapter/tests` plus the `tests/unit` files importing it (leaf package) |
| `python/hal/src/openral_hal/sim_attached.py` | broad: `hal` fans out to the runtime, CLI and sim tests |
| `python/core/src/openral_core/schemas.py` | full run (every package depends on core) |
| `packages/openral_hal_node/**` | `packages/openral_hal_node/test` |
| `rskills/act-aloha/**` | unit fixture checks plus ALOHA sim tests, with the `sim` lane |
| `pyproject.toml`, `uv.lock`, shared `conftest.py` | full run |
| `cpp/openral_safety_kernel/**` | nothing here (covered by `test-ros2`) |
| `docs/**`, `scenes/**` | nothing, or the fixture-loader test only |

## Lane policy — what a skip is allowed to mean

A GitHub-hosted runner has no NVIDIA GPU, no Vulkan ICD and no proprietary sidecars, so tests gated on those skip there forever. A blanket "no skips" rule would redden lanes that did real work; a blanket tolerance would hide provisioning bugs. The rule is a **declared capability allowlist**:

- `[capability_gaps]` in `tools/test_selection.toml` names each capability the runner provably lacks (`cuda`, `vulkan`, `sidecar`, `ros`, `hf-cache`, `pro-quant` for the OpenRAL Pro quantization package), what would satisfy it, and the skip-reason patterns that indicate it.
- A skip matching a declared pattern is **declared-not-run**: allowed, counted and printed in the job summary.
- **Every other skip fails the lane.** `panda.srdf not installed` is a ROS-underlay gap (declared); `mujoco not installed` is a provisioning bug and stays red. Matching is fail-closed: reword a skip reason out of the patterns and the lane goes red.
- A lane whose every selected test is explained by a declared gap is `declared-not-run`: a loud pass naming the capability it needs. A lane that collected no tests at all fails.
- Batched and isolated targets are judged by the same rule.

### Vacuous green is a failure

[`tools/lane_report.py`](https://github.com/OpenRAL/openral/blob/master/tools/lane_report.py) parses each lane's junit XML into a `LaneRecord` and uploads it as the `lane-ledger-<lane>` artifact. The `heavy-lanes` job merges the ledgers and cross-checks them against the selector's output. It fails when a `full_run` diff expanded to zero lanes, a selected lane produced no record, or any record is a failure. The summary names every declared-not-run lane.

Two junit details matter: a module-level `importorskip` writes `message="collection skipped"` with the real reason in the element text, so the parser reads the text; and `simpler-env` / `robocasa-gr1` run pytest inside `verify_test_envs.py`, which is passed `--junit-xml` so they are accounted like every other lane.

## 2. The test audit

`tools/audit_tests.py` reads every test with `ast` and classifies:

- **trivial**: body is only `pass`, `...` or a docstring. Dead.
- **shadowed**: the same name defined twice in one scope. Python keeps only the last, so the earlier one never runs; always safe to delete.
- **duplicate-body**: byte-identical normalized ASTs. Usually a `parametrize` opportunity, not a deletion (per-robot HAL-contract tests share a body but exercise different robots).
- **no-assertion**: no `assert` and no recognised validation call (`from_yaml`, `model_validate`, `pytest.raises`, ...). A candidate for review, not an auto-delete.

`just test-audit` writes the [snapshot report](test-audit.md). `tests/unit/test_audit_tests.py::test_repo_has_no_dead_tests` fails if anyone lands a trivial or shadowed test. The tool never deletes anything: per CLAUDE.md §1.15 a deletion is its own reviewed commit.
