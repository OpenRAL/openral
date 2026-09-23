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

   A full run expands **every opt-in dependency lane** as well (rule 6): a
   root-`pyproject.toml`/`uv.lock` change is a *dependency* change, which is
   precisely what the lanes exist to check.

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
   instead of only the broad unit fixture checks. The `robots/**` mapping
   deliberately reaches **outside** `tests/unit` into the `packages/**` test
   files that load a real `robots/<id>/robot.yaml`: those dirs are otherwise
   selected only by a `packages/<pkg>/**` change, so a manifest edit could —
   and did — break a ROS package's test with nothing on any lane to see it.
   Adding a `packages/**` test that reads a real manifest means adding it
   there too.
6. **Dependency lanes.** Selected targets matching `requirement_globs` in
   `tools/test_selection.toml` are also emitted per opt-in dependency group
   (`sim`, `libero`, `robocasa`, `robocasa-gr1`, `maniskill3`, `simpler-env`,
   `isaacsim`, `robotwin`, `sidecar-wire`, `dataset`, `rlbench`, `gr00t`,
   `locateanything`, `qwen-vlm`, `omdet`, `onnx-export`, `clip`, `opencv`,
   `lowering`).
   The default cheap lane may still skip those tests; CI then reruns the matching
   targets with the named group installed and fails if the lane produces no
   passing tests **or any skip that is not a declared capability gap** (see
   [Lane policy](#lane-policy-what-a-skip-is-allowed-to-mean) below). This is
   what prevents “selected but skipped because `gym_aloha` is absent” from going
   green. The `sim` lane also installs the `dataset` extra because its
   dataset-emission target writes and reloads a real LeRobot dataset. Wire-only
   sidecar tests use the `sidecar-wire` lane; sidecar-backed runtime lanes keep
   separate logical names because their preflight requirements differ.

   A lane declared with an **empty** glob list can never run — `run_lane`
   returns at its `[ -z "$targets" ]` guard, logging nothing, and looks exactly
   like a lane that simply was not selected. `tests/unit/test_lane_report.py`
   now fails any lane declared with no globs.

   **A selected directory counts as selecting the lane files inside it.**
   `targets` mixes files (from the import scan) with directories (a package's
   whole `tests/` dir, or an `extra_triggers` value like `tests/unit`), and a
   directory string never `fnmatch`es a file glob. So a lane-owned file only
   reachable *inside* a directory target used to drop out of its lane while
   still being "selected": it ran in the cheap partition, skipped for want of
   the optional stack, and reported green.

   The selector now expands directory targets to the concrete lane files
   beneath them, so containment is enough. Practical
   effect: a `robots/**` diff (which selects `tests/unit`) now also reruns the
   `clip`, `opencv`, `onnx-export` and `sidecar-wire` lanes it always implicitly
   selected but never actually exercised.

   **A lane is assigned per FILE, so one file must not straddle two mutually
   exclusive groups.** LIBERO pins `robosuite==1.4`, RoboCasa/OpenArm need
   `>=1.5`, and the two cannot coexist in one venv (CLAUDE.md — "LIBERO↔RoboCasa
   groups are mutually exclusive"). `python/hal/tests/test_sim_attached_action_dim.py`
   used to hold both LIBERO tests and the OpenArm tabletop test; run in the
   `libero` lane the LIBERO half imported robosuite 1.4 first, and the OpenArm
   half then hit `openral_sim._deps._assert_no_live_dependency_swap`, which
   correctly refuses to swap robosuite underneath live objects — so it *failed*
   rather than ran, on every PR whose diff reached `python/hal/tests`.

   The OpenArm test now lives in its own file
   (`python/hal/tests/test_sim_attached_openarm_action_dim.py`) on the `robocasa`
   lane. When you add a test needing a different optional stack from its file's
   neighbours, give it its own file.
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
     all-pass summary, turning green tests into red CI.
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

   The `robocasa` and `robocasa-gr1` lanes need robosuite 1.5.2 from a git pin
   (no PyPI release). Each opt-in lane is its own CI job with its own runner and
   venv (`lane` in `test-selective.yml`), so the cross-lane contamination this
   guard originally targeted — the `libero` lane leaving robosuite 1.4.0
   behind for a *later* lane sharing the same job/venv — cannot happen any
   more. It is kept anyway: a bare `uv run --group robocasa` group switch does
   not reliably land the git-pinned package on its own (uv evicts master for a
   wheel — see `python/sim/src/openral_sim/_deps.py`), which surfaces as SO100
   missing from `REGISTERED_ROBOTS` or `NullMount` not being a `MountModel`.
   `run_lane` guards this with an explicit `uv sync --frozen --all-packages
   --group robocasa --reinstall-package robosuite` before the lane's `uv
   run`s, so the env lands on the pinned tree deterministically regardless.

Every selected target carries a human-readable reason.

### Lane policy — what a skip is allowed to mean

Once a full run expands every lane, the lanes have to be *judged* correctly, and
the old rule ("a lane must have passing tests and must not skip **at all**")
could not survive that. A GitHub-hosted `ubuntu-24.04` runner has no NVIDIA GPU,
no Vulkan ICD, and no proprietary simulator sidecars. Tests gated on those skip
there **forever**, so the blunt rule reddened lanes that had in fact done real
work — the `sim` lane ran 162 passing tests and was failed anyway by
13 `requires CUDA` skips.

Three options were on the table, and only one is honest:

| Option | Why not / why |
| --- | --- |
| Run nothing on a full run, but say so loudly | Rejected. It annotates the coverage hole instead of closing it, and a blast-radius diff (`uv.lock`!) is exactly when lanes matter most. |
| Gate the unsatisfiable lanes behind a self-hosted runner label | Right long-term, unavailable today: no such runner is registered, and a required check waiting on a label that never appears is left `Expected` forever, blocking the merge. The declarations below name what *would* satisfy each gap, so this stays the upgrade path. |
| **Run every satisfiable lane; declare the rest** | **Chosen.** Preserves all reachable coverage and makes the unreachable part legible instead of invisible. |

The mechanism is a **declared capability allowlist**, not a blanket tolerance:

- `[capability_gaps]` in `tools/test_selection.toml` names each capability the
  runner provably lacks (`cuda`, `vulkan`, `sidecar`, `ros`), what it is, what
  *would* satisfy it, and the skip-reason patterns that indicate it.
- A skip matching a declared pattern is **declared-not-run** — allowed,
  attributed to the gap, counted, and printed in the job summary. It is never
  called "skipped" and never silently absent.
- **Every other skip still fails the lane.** `panda.srdf not installed` is a
  ROS-underlay asset (declared); `mujoco not installed` is a *provisioning bug*
  and stays red. That distinction is the entire value of an allowlist over
  "tolerate skips", and it is why the patterns are narrow.
- Matching is **fail-closed**: reword a skip reason out of the declared patterns
  and the lane goes red, not quietly green.
- A lane whose **every selected test** is explained by a declared gap is
  `declared-not-run`: a pass, but a loudly reported one, named in the summary
  with the capability it needs. A lane that collected **no tests at all** still
  fails — that is a broken lane, not absent coverage.

  The verdict is deliberately about what the diff *selected*, not a lane's full
  potential. `sim` yields 162 passing tests when all seven of its files are
  selected, but a diff touching only `rskills/act-aloha/**` selects just
  `test_aloha_bimanual_act_aloha.py`, whose six tests are all CUDA-gated.
  Failing that would punish a PR for
  touching a GPU-only file — precisely the breakage this policy exists to
  remove. There is deliberately **no** hand-maintained list of "lanes that
  cannot run here": that would be a second source of truth that rots, while the
  per-skip classification derives the verdict from evidence every run.

Batched and isolated targets are now judged by the **same** rule — previously
isolated files were graded on exit code alone, so a skip inside one was
invisible. Every skip they emit today is explained by a declared gap, so
closing the hole costs nothing.

### Vacuous green is a failure, not a result

`tools/lane_report.py` parses each lane's **junit XML** (not pytest's terse
summary text) into a `LaneRecord`, appended to that lane job's own ledger and
uploaded as an artifact (`lane-ledger-<lane>`). The `heavy-lanes` job's "Merge
ledgers and attest lane coverage" step downloads every lane's artifact,
concatenates them into one ledger, and cross-checks it against the selector's
own output. It fails when:

- a `full_run` diff expanded to **zero** lanes;
- a lane the selector **selected** produced no ledger record ("selected but
  never executed");
- any lane record is a failure.

The ledger is written to the job summary, naming every declared-not-run lane and
the capability it needs. So "tested nothing" and "tested everything and passed"
can no longer report the same thing.

> Two junit details are load-bearing and were verified against real pytest
> output rather than assumed. A module-level `importorskip` emits
> `message="collection skipped"` and puts the real reason in the element
> **text**; reading only `message` would misclassify every such skip as
> undeclared. And `simpler-env` / `robocasa-gr1` run their pytest *inside*
> `verify_test_envs.py`, so they have no junit report at all and are recorded
> explicitly as exit-code-only rather than being credited with per-test
> accounting they never had.

### Usage

```bash
# What would run for the current branch vs origin/master?
just test-changed                       # prints the plan
uv run python tools/select_tests.py --files python/state_adapter/src/openral_state_adapter/core.py
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
workflow's `select` job runs `select_tests.py --github-output` once, then fans
its outputs out to two downstream jobs: `core_full` (the whole suite,
`full_run=true`) and `core_selected` (just the emitted targets — `--ignore`ing
the `isolated_targets` from those partitions and re-running each in its own
process, see rule 7 above). The opt-in `lane` jobs (one per dependency group,
built from the `lanes` output) run in the separate
[`heavy-lanes`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/heavy-lanes.yml)
workflow, on request only. `just test-changed-run` mirrors the
`core_selected` path locally.

### CI job graph and speed-up design

```
test-selective.yml (every push):
select ──┬── core_full (matrix: 4 file-shards + isolated) ──┐
         └── core_selected ────────────────────────────────┴── select-and-test
                                                               (required check)

heavy-lanes.yml (only for the `heavy-lanes` PR label or a manual dispatch):
select ── lane (matrix: one job per opt-in dependency group) ── heavy-lanes
                                                                (gate + ledger
                                                                 attest)
```

Splitting one job into this graph is what makes `test-selective` fast on
every push instead of ~15 min every time — the 19 opt-in lanes used to run
one after another in the same job as everything else; now they live in their
own workflow, run in parallel in their own jobs, and only when asked for (see
[Review policy](development.md#review-policy)). Both workflows' `select` jobs
run the same [`select-tests`](https://github.com/OpenRAL/openral/blob/master/.github/actions/select-tests/action.yml)
composite action.

1. **Selection runs first, in its own cheap job.** `select_tests.py` only
   needs `pydantic` + stdlib; `select` runs it via `uv run --isolated --with
   pydantic` — a disposable ephemeral env that resolves in a few seconds. The
   job doesn't even enable the `uv` cache, since it never touches `uv.lock`.
2. **Everything downstream is conditional.** `core_full`, `core_selected`, and
   `lane` are skipped entirely when nothing is selected (docs-only diffs,
   pure markdown changes, etc.) — no FFmpeg install, no workspace sync, no
   pytest.
3. **The full `tests/unit/` suite runs as a 4-way file-based matrix** (plus one
   dedicated shard for `isolated_targets`) instead of serially in one job.
4. **Test-root partitions within `core_selected` run in parallel.** The bash
   loop in "Run selected targets" launches each group as a background job
   (`&`), collects exit codes after all finish, and streams the logs in
   collapsible GitHub groups.
5. **Every opt-in dependency lane is its own matrix job.** Each pays its own
   ~2 min setup, but 19 lanes running in parallel finish close to the
   *slowest* lane instead of the *sum* of all of them. A lane only runs when
   selected, and every skip inside it must be accounted for — if no selected
   target needs `gym_aloha`, the `sim` group is never installed; if one does,
   the lane's own job reruns just those targets under `uv run --all-packages
   --group sim ...` and requires passing tests, with no skip beyond the
   declared capability gaps (see
   [Lane policy](#lane-policy-what-a-skip-is-allowed-to-mean)).
6. **Lanes run only on request.** `lane` and its `heavy-lanes` gate live in
   `heavy-lanes.yml`, whose `select` job runs only for a `workflow_dispatch`
   or when the PR carries the `heavy-lanes` label. Otherwise every job there
   is skipped — a skipped job is not red and reports success to
   required-check evaluation. `select-and-test` never waits on the lanes.
7. **`robot_descriptions` / openarm asset clones are cached** across runs
   (`.github/actions/setup-test-env`), instead of re-cloned by every job that
   needs them.
8. **Stale runs are cancelled.** A `concurrency` group with `cancel-in-progress:
   true` stops any in-progress run on the same branch the moment a new push
   arrives.
9. **Documentation-only PRs skip heavy setup entirely.** Both required checks
   still report success, but the selector emits no targets, so no downstream
   job does any real work.

### Worked examples

| Change | Result |
| --- | --- |
| `python/state_adapter/src/openral_state_adapter/core.py` | `python/state_adapter/tests` only (leaf package) |
| `python/core/src/openral_core/schemas.py` | broad — core fans out to ~every package's tests |
| `packages/openral_hal_node/**` | `packages/openral_hal_node/test` |
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
