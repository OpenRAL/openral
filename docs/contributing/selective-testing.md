# Selective testing & the test audit

OpenRAL carries **~3.3k test functions across ~360 files**. Two tools keep that
suite fast *and* meaningful:

| Tool | Question it answers | Entry point |
| --- | --- | --- |
| [`tools/select_tests.py`](https://github.com/OpenRAL/openral/blob/master/tools/select_tests.py) | "Given this diff, which tests can actually observe the change?" | `just test-changed` |
| [`tools/audit_tests.py`](https://github.com/OpenRAL/openral/blob/master/tools/audit_tests.py) | "Which tests are dead, duplicated, or low-signal?" | `just test-audit` → [`test-audit.md`](test-audit.md) |

Both are read-only with respect to the suite (the auditor never deletes; the
selector never edits). They pass `mypy --strict tools/` and are themselves
covered by `tests/unit/test_select_tests.py` and `tests/unit/test_audit_tests.py`.

---

## 1. Selective execution — `select_tests.py`

### Why

The cheap, high-signal workflows run on every PR (`quality` — ruff + mypy +
schema drift + `mkdocs --strict`; `test-selective`; `dco`), while the expensive
full-matrix suites (`test-python`, `hal`) stay `workflow_dispatch`-only ("out of
GitHub Actions credits" — see the headers in `.github/workflows/`). Running 2.9k
tests on every push is the difference between affordable and not. Selective
execution maps a
git diff to the **minimal** set of pytest targets that can see the change, so a
one-line edit to a leaf package runs a handful of tests instead of the whole
suite.

### How it decides (explicit, never magic — CLAUDE.md §1.4)

1. **Blast radius first.** If a changed path matches `full_run_globs` in
   [`tools/test_selection.toml`](https://github.com/OpenRAL/openral/blob/master/tools/test_selection.toml) — root
   `pyproject.toml`, `uv.lock`, a shared `conftest.py`, the selector's own
   inputs — it emits `full_run = true` and the caller runs everything. We never
   try to be clever about a wide-blast change; a wrong *negative* would silently
   skip a regression.

   The one exception is release-please's release PR, and it is handled in the
   workflow rather than here — precisely so the selector keeps no special
   cases. `test-selective` short-circuits the `release-please--*` branch before
   invoking the selector at all; see
   [Releasing](releasing.md). That PR rewrites all 15 pyprojects, so it would
   otherwise trip both this rule and the package graph below, for a diff whose
   every commit already ran on its own PR.
2. **Dependency graph, derived not hand-written.** The package graph is read
   straight from each `python/<pkg>/pyproject.toml` (`openral-*` deps). It can
   never drift from the real workspace.
3. **Transitive dependents.** A change to `openral_core` pulls in every package
   that imports it — directly or through a chain — and therefore their tests.
4. **Two selection paths.** For each affected package: its own `tests/` dir
   (when present), plus every top-level `tests/**` file whose `import openral_*`
   set intersects the affected packages. (Most of `python/core`, `cli`, `rskill`,
   `runner`, `sim` keep their tests under the shared `tests/` tree, so the
   import scan is what attributes them.)
5. **Fixture triggers.** Non-code fixtures (`robots/**`, `rskills/**`,
   selected `scenes/**`) explicitly map to tests that load them for real. A
   change to `rskills/act-aloha/**`, for example, selects the ALOHA sim tests
   instead of only the broad unit fixture checks.
6. **Dependency lanes.** Selected targets matching `requirement_globs` in
   `tools/test_selection.toml` are also emitted per opt-in dependency group
   (`sim`, `libero`, `robocasa`, `robocasa-gr1`, `maniskill3`, `simpler-env`,
   `isaacsim`, `robotwin`, `sidecar-wire`, `dataset`, `rlbench`, `gr00t`,
   `locateanything`, `qwen-vlm`, `omdet`, `onnx-export`, `clip`, `opencv`,
   `lowering`).
   The default cheap lane may still skip those tests; CI then reruns the matching
   targets with the named group installed and fails if the lane produces no
   passing tests **or any skip**. This is what prevents “selected but skipped
   because `gym_aloha` is absent” from going green. Mixed files with intentional
   fixture-absence skips stay outside strict lanes. The `sim` lane also installs
   the `dataset` extra because its dataset-emission target writes and reloads a
   real LeRobot dataset. Wire-only sidecar tests use the `sidecar-wire` lane;
   sidecar-backed runtime lanes keep separate logical names because their
   preflight requirements differ.
7. **Ignored domains.** `cpp/**` is covered by `test-ros2` (colcon) / the
   safety-kernel ctest, not the Python suite, so a pure-C++ change selects
   nothing here rather than forcing a wasteful full Python run.
8. **Unattributed source ⇒ full run.** A changed `.py`/`.cpp`/… that maps to no
   known package is treated conservatively as a full run.
9. **Fork-isolated tests run in their own process.** A handful of tests
   (`isolate_globs` in the toml) cannot share an interpreter with a sibling test.
   Two failure modes live here:
   - **Dataset forkers** drive lerobot's `compute_stats`, which forks a
     multiprocessing pool. Folded into the broad CLI partition — which has
     already spun up numpy/pyarrow/torch threadpools — the fork happens in a
     multi-threaded interpreter and a forked child / C-extension `atexit` handler
     crashes during Python finalization: the process exits non-zero **after** an
     all-pass summary, turning green tests into red CI
     ([issue #24](https://github.com/OpenRAL/openral/issues/24)).
   - **EGL/robosuite env creators** (`test_sim_attached_action_dim.py`,
     `test_sim_attached_idle_step.py`) each spin up a real LIBERO /
     robosuite-MJCF `OffScreenRenderEnv`. A robosuite/MuJoCo EGL context does not
     survive a sibling env's teardown in the same process: once one file's env is
     GC'd, the next file's LIBERO env dies at `reset` with
     `<...RethinkMount> is not a MujocoXML instance` — even on the correct
     robosuite 1.4 (so it reads like a 1.4-vs-1.5 conflict but is not). Each file
     passes cleanly run alone.

   `select_tests.py` peels any in-scope match out of `targets` into
   `isolated_targets`; the full-run path `--ignore`s them from every partition
   and runs each in its own `pytest` invocation, and each opt-in dependency lane
   (`run_lane`) likewise splits its isolated targets out of the batched run and
   runs them one-per-process (still under that lane's `--group`). An isolated
   file may legitimately `importorskip` an optional dep it does not need in that
   lane (e.g. `rclpy` on the no-ROS libero lane host), so its run is judged by
   exit code — a real failure fails the lane; an all-/partial-skip does not.

   The `robocasa` and `robocasa-gr1` lanes need robosuite 1.5.2 from the git pin
   (`232ce7d4`). The `libero` lane runs first and leaves robosuite 1.4.0
   installed, and a bare `uv run --group robocasa` group switch does not reliably
   reinstall the git package (uv evicts master for a wheel — see
   `python/sim/src/openral_sim/_deps.py`), which surfaces as SO100 missing from
   `REGISTERED_ROBOTS` or `NullMount` not being a `MountModel`. `run_lane` guards
   this with an explicit `uv sync --frozen --all-packages --group robocasa
   --reinstall-package robosuite` before the lane's `uv run`s, so the env lands
   on the pinned tree deterministically.

Every selected target carries a human-readable reason.

### Usage

```bash
# What would run for the current branch vs origin/master?
just test-changed                       # prints the plan
uv run python tools/select_tests.py --files python/wam/src/openral_wam/core.py
uv run python tools/select_tests.py --base origin/master --head HEAD

# Actually run only the affected tests:
just test-changed-run                    # selects, then invokes pytest
```

To verify every optional environment locally before CI, run the strict lane
verifier. It syncs each dependency group, runs the lane targets, and fails on
any skip:

```bash
uv run python tools/verify_test_envs.py --groups sim opencv lowering
uv run python tools/verify_test_envs.py --include-provisioned --groups qwen-vlm locateanything simpler-env robocasa-gr1
```

Sidecar lanes intentionally block until the required env vars are present; the
script does not fake proprietary sidecars.

In CI, the [`test-selective`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/test-selective.yml)
workflow runs `select_tests.py --github-output`, then either runs the whole
suite (`full_run=true`) or just the emitted targets — `--ignore`ing the
`isolated_targets` from those partitions and re-running each in its own process
(see rule 7 above). `just test-changed-run` mirrors this locally.

### CI speed-up design

The `test-selective` workflow is optimised so that slow setup steps are never
paid for runs that select zero tests:

1. **Selection runs first, before heavy installs.** `select_tests.py` only needs
   `pydantic` + stdlib; the workflow runs it via `uv run --isolated --with
   pydantic` — a disposable ephemeral env that resolves in a few seconds with no
   workspace sync required.
2. **FFmpeg install and `uv sync` are conditional.** Both are skipped entirely
   when `steps.select.outputs.any != 'true'` (docs-only diffs, pure markdown
   changes, etc.), saving 1–3 min of pointless setup per such PR.
3. **Test-root partitions run in parallel.** The bash loop in "Run selected
   targets" launches each group as a background job (`&`), collects exit codes
   after all finish, and streams the logs in collapsible GitHub groups — cutting
   wall-clock time by roughly the number of partitions.
4. **Opt-in lanes run only when selected and are zero-skip.** If no selected target needs
   `gym_aloha`, the `sim` group is never installed. If one does, CI reruns just
   those targets under `uv run --all-packages --group sim ...` and requires
   passing tests with no skips in that lane.
5. **Stale runs are cancelled.** A `concurrency` group with `cancel-in-progress:
   true` stops any in-progress run on the same branch the moment a new push
   arrives.
6. **Documentation-only PRs skip heavy setup.** The required workflow still
   reports success, but the selector emits no targets, so FFmpeg install,
   workspace sync, and pytest execution are skipped.

### Worked examples

| Change | Result |
| --- | --- |
| `python/wam/src/openral_wam/core.py` | `python/wam/tests` only (leaf package) |
| `python/core/src/openral_core/schemas.py` | broad — core fans out to ~every package's tests |
| `packages/openral_hal_so100/**` | `packages/openral_hal_so100/test` |
| `rskills/act-aloha/**` | unit fixture checks + ALOHA sim tests, with the `sim` dependency lane |
| `pyproject.toml` / `uv.lock` / shared `conftest.py` | **full run** |
| `cpp/openral_safety_kernel/**` | nothing (covered by `test-ros2`) |
| `docs/**`, `scenes/**` | nothing / fixture-loader test only |

---

## 2. Test audit — `audit_tests.py`

Generates [`test-audit.md`](test-audit.md). It reads every test with `ast` and
classifies:

- **trivial** — body is only `pass` / `...` / a docstring. Genuinely dead.
- **shadowed** — the same name defined twice in one scope (file + class). Python
  keeps only the last; the earlier definition is never collected. **This is the
  one duplicate that is always safe to delete.**
- **duplicate-body** — two+ tests with byte-identical normalized ASTs. Usually a
  *parametrize* opportunity, not a deletion: the per-robot HAL-contract tests
  (`test_satisfies_hal_protocol`, `test_estop_*`, …) share a body but exercise
  *different robots*.
- **no-assertion** — neither `assert` nor a recognised validation call
  (`from_yaml`, `model_validate`, `pytest.raises`, …). A *candidate* for review,
  not an auto-delete: a constructor that raises on bad input is a real check.

### Current state (regenerate with `just test-audit`)

As of the last run the suite is **disciplined**: **0 trivial** and **0 shadowed**
tests — there is nothing obviously dead to prune. The real redundancy signal is
the **29 duplicate-body groups**, dominated by per-robot HAL-contract tests that
are prime candidates for consolidation into a single parametrized contract
module (a reviewed refactor, since each currently asserts on a distinct robot).
The **111 no-assertion** entries are flagged for human review.

> Pruning is never bundled into this tooling. Per CLAUDE.md §1.7/§1.11 tests are
> part of the contract; per §1.15 any deletion is its own reviewed commit.

A regression guard lives in `tests/unit/test_audit_tests.py::test_repo_has_no_dead_tests`
— if anyone lands a trivial or shadowed test, that test goes red.
