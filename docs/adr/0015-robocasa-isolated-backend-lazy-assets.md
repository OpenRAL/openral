# ADR-0015: RoboCasa as an isolated `openral sim` backend with lazy asset download

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: ADR-0002 (eval/sim environments), ADR-0007 (robot/sim split),
  ADR-0009 (separate sim from benchmarking),
  ADR-0014 (ManiSkill3 + SimplerEnv backends — original RoboCasa deferral)
- Tracking issue: [#88](https://github.com/OpenRAL/openral/issues/88)

## Context

ADR-0014 explicitly **deferred** RoboCasa from the second wave of sim
backends because of two blockers documented in its own context section:

1. **Robosuite version conflict.** RoboCasa pins `robosuite>=1.5`;
   LIBERO (via `lerobot[libero]`) pins `robosuite==1.4`. The two cannot
   coexist in a single Python environment — uv's solver fails the join.
2. **Asset weight.** RoboCasa kitchens ship as a ~10 GB CC-BY-4.0 asset
   bundle fetched post-install by
   `robocasa/scripts/download_kitchen_assets.py`. A default
   `uv sync --group robocasa` cannot block on a 10 GB download
   (especially in CI), and the assets are CC-BY which differs from the
   Apache-2.0 / MIT defaults the rest of the open core assumes
   (CLAUDE.md §1.9).

RoboCasa is, however, the canonical MuJoCo kitchen suite in the
robosuite ecosystem: ~100 prebuilt atomic-PnP tasks plus a procedural
kitchen authoring API (style × layout × fixtures × spawnable objects ×
task verb) that closes the "test on other sim environments other than
LIBERO" ergonomics gap raised against `openral sim run` (issue #88). It is
also the closest existing analogue to **custom MuJoCo scenarios with
custom robots and tasks** — exactly the surface we want a sim-agnostic
harness to expose.

The user-facing pain `openral sim` had until now is that LIBERO hard-wires
the Franka Panda regardless of what `--robot` says (a separate UX issue
landed in the same branch as this ADR — see [`docs/METHODS.md`](../METHODS.md)
"sim ergonomics" entries — but the deeper limitation is that the
harness has no MuJoCo backend in which `(robot, scene, task)` are
genuinely free axes). RoboCasa's procedural API plus its
`PandaMobile` / `GR1` / multi-arm robot support is that backend.

This ADR captures the **design decisions** for the RoboCasa
integration. Issue #88 splits the actual delivery into five
independently-mergeable PRs (A–E) of ≤ ~800 lines each
(CLAUDE.md §7.2). This ADR is the foundation for
**PR A**; the typed `RoboCasaBackendOptions` Pydantic model + the
`robocasa` uv extras group land alongside this document. The adapter
itself (PR B), examples + skill manifest (PR C), tutorial (PR D), and
benchmark catalogue (PR E) follow on separate branches.

## Decision

1. **Isolated `robocasa` extras group, mutually exclusive with `libero`.**
   `pyproject.toml` gains a new `[dependency-groups].robocasa` group
   pinning `robosuite>=1.5`. The `robocasa` package itself is **not**
   PyPI-published today (same situation as `simpler-env` in ADR-0014);
   the group documents the manual install command
   (`uv run pip install "robocasa @ git+https://github.com/robocasa/robocasa.git"`)
   and the scene adapter (PR B) raises `ROSConfigError` with that
   exact hint when `robocasa` is absent. When upstream cuts a PyPI
   release we promote the entry to a real pin without amending this
   decision. The mutual exclusion with the existing `libero` group
   is enforced **by uv's solver** (`robosuite==1.4` vs
   `robosuite>=1.5` cannot resolve into the same environment) and
   documented but **not** programmatically enforced inside Python —
   any solver-failure message uv produces is already actionable, and
   trying to mirror that check in our own code would add a brittle
   second-source-of-truth. The README + ADR call out the workaround:
   create a dedicated venv per backend (`uv sync --group robocasa`
   for kitchens, `uv sync --group libero` for LIBERO).

2. **Lazy first-use asset download under
   `$OPENRAL_CACHE_HOME/robocasa/`.** No `uv sync` step blocks on
   the 10 GB fetch. The scene factory probes a readiness sentinel
   (`<cache_home>/robocasa/.openral-ready`); if absent, it displays
   a Rich license banner (CC-BY-4.0, upstream license URL, target
   path, estimated size) and gates the download behind either:

   - the env var `OPENRAL_ALLOW_ROBOCASA_ASSETS=1` (CI bypass,
     mirrors the `OPENRAL_ALLOW_NONCOMMERCIAL` precedent in
     `python/rskill/src/openral_rskill/loader.py:71`), **or**
   - an interactive `typer.confirm()` prompt.

   On confirm, the factory invokes
   `robocasa.scripts.download_kitchen_assets` (Python entry-point
   preferred, `subprocess.run([sys.executable, "-m", ...])` fallback)
   and touches the sentinel on success. On refusal it raises
   `ROSConfigError` with the exact manual-fetch command. Subsequent
   runs are silent.

3. **Dual scenario surface.** The adapter registers both:

   - ~100 **prebuilt** scene IDs of the form `robocasa/<task>` — one
     for each RoboCasa atomic task (e.g. `robocasa/PnPCounterToCab`,
     `robocasa/OpenSingleDoor`). `openral sim list` discovers them
     automatically; existing `--config` and explicit-flag invocations
     pick them up with no further wiring.
   - **One procedural** scene ID `robocasa` consumed via a typed
     `SceneSpec.backend_options` block. The user passes
     `kitchen_style`, `layout_id`, `fixtures`, `spawn_objects`,
     `task_verb`, `robots`, `controller`, `horizon` directly in the
     YAML; the adapter validates them through the new
     `RoboCasaBackendOptions` Pydantic model
     (see [`docs/reference/schemas/RoboCasaBackendOptions.json`](../reference/schemas/RoboCasaBackendOptions.json)
     once `just schema-export` regenerates it). A
     `prebuilt_task` XOR procedural-keys validator forbids mixing the
     two modes.

   `SceneSpec.backend_options: dict[str, object]` already exists in
   `openral_core.schemas`; the new model is an **additive
   validator helper**, not a schema migration. The on-disk
   `schema_version` stays at `"0.1"` while we are pre-publish, so no
   migrator is needed (CLAUDE.md §1.6).

4. **No new `openral sim` subcommands.** `openral sim list` already enumerates
   registered scenes, so `robocasa/<task>` IDs surface automatically
   once PR B's adapter is imported. `openral sim run` triggers the asset
   prompt on first launch and is unchanged thereafter.

5. **Layer placement.** The adapter lives in
   `python/sim/src/openral_sim/backends/robocasa.py` next to the
   other scene adapters (`maniskill3.py`, `aloha.py`, etc.). The
   asset-fetch helper lives in a fresh
   `python/sim/src/openral_sim/_assets.py` so it can be reused if
   any later backend needs the same banner/sentinel/confirm pattern;
   it does **not** live under `python/rskill/` (would cross the Sim →
   Skill layer boundary, violating
   CLAUDE.md §6.1).

## Consequences

**Positive**

- Closes issue #88. `openral sim run` gains a MuJoCo backend where
  `(robot, scene, task)` are genuinely free axes (the procedural path)
  plus ~100 prebuilt tasks for benchmark coverage.
- Mirrors the multi-backend pattern from ADR-0014 (extras-group + lazy
  scene-factory imports), so the cognitive load on contributors is the
  same as for ManiSkill3 / SimplerEnv.
- The lazy asset fetch keeps default `uv sync` fast and the open core
  install footprint unchanged. CI runs that do not exercise RoboCasa
  pay zero cost.

**Negative**

- Users running LIBERO **and** RoboCasa in the same workflow must
  swap virtual environments. Documented; not programmatically guarded.
- A 10 GB asset fetch is a meaningful first-use latency. The Rich
  banner and `typer.confirm()` make the cost explicit, but it remains
  the slowest first-use UX of any harness backend.
- CC-BY-4.0 asset attribution must be carried in any derivative
  artefact (videos, traces) that ships those scenes. We surface the
  license in the banner and in the rSkill manifest README; we do not
  yet ship a `LICENSES/robocasa-assets.md` aggregator (deferred to a
  later docs PR).

**Exit criterion**

- If `lerobot[libero]` moves to `robosuite>=1.5`, collapse the
  `libero` and `robocasa` extras groups into a single shared backend
  group. Re-anchor this ADR's Decision §1 against that change and
  amend the Consequences section in-place
  (CLAUDE.md §7.9 — additive amendments preserve the
  original Decision text).

## Alternatives considered

- **Vendor robosuite-1.5 ourselves.** Rejected: re-vendoring a
  fast-moving upstream library is a maintenance liability that does
  not buy parity with LIBERO (which would still want robosuite-1.4).
- **Pre-bundle a slimmed asset pack.** Rejected: CC-BY-4.0 licensing
  is upstream's, and slimming the pack would either lose coverage (so
  reproducibility of paper-cited RoboCasa numbers slips) or
  re-introduce the 10 GB problem behind a different door.
- **Make the asset prompt a top-level `ral skill install` flow.**
  Rejected: the assets are scene-level, not skill-level — they belong
  to the sim backend's cache, not the rSkill cache. Mixing them would
  contradict ADR-0009 (separate sim from benchmarking) and
  CLAUDE.md §6.4 (rSkills are skill weights, not
  arbitrary blobs).
- **Programmatically forbid `libero + robocasa` extras in the same
  venv.** Rejected: a Python-side check duplicates uv's solver
  failure with strictly less information (no resolver trace), and
  silently passing the check inside `openral_sim` would let users
  end up in a partially-broken state. We rely on the solver.

## References

- Upstream: <https://github.com/robocasa/robocasa>
- Upstream asset script:
  `robocasa/scripts/download_kitchen_assets.py`
- ADR-0014 — original RoboCasa deferral and the precedent extras-group
  pattern reused here.
- ADR-0002 — `SimEnvironment` / `SceneSpec.backend_options` contract.
- Issue #88 — the five-PR delivery plan that imports this ADR.
- CLAUDE.md §1.6, §1.11, §1.13, §1.14, §6.1, §7.2, §7.9, §10
  — discipline this ADR adheres to.

## Amendments

### 2026-05-18 — Status flipped Proposed → Accepted

The RoboCasa backend is live as an isolated extras-group:

- `python/sim/src/openral_sim/backends/robocasa.py` — RoboCasa adapter
  with the lazy asset-download check declared in the Decision.
- `benchmarks/robocasa_pnp.yaml` — atomic-PnP suite.
- The LIBERO ↔ RoboCasa robosuite-version conflict declared in the
  Context section is encoded directly in the workspace solver — see
  root `pyproject.toml` `[tool.uv] conflicts = [...]` block listing
  `{ group = "libero" }` and `{ group = "robocasa" }` as mutually
  exclusive, so a `uv sync --group libero --group robocasa` fails with
  the resolver's full trace as designed.

No behavioural change against the Decision text — only the status field
flips.
