# CLAUDE.md — OpenRAL Engineering Playbook

> Single source of truth for AI coding agents and human contributors on **OpenRAL**. Read in full before touching code. See [README.md](README.md) for product identity.
>
> Pointers (kept out of this file to stay tight):
> - Repo layout → [`docs/architecture/repo-map.md`](docs/architecture/repo-map.md) + live [`repo-state-map.html`](docs/architecture/repo-state-map.html).
> - Toolchain & `openral` CLI → [`docs/contributing/toolchain.md`](docs/contributing/toolchain.md).
> - Glossary → [`docs/reference/glossary.md`](docs/reference/glossary.md).
> - Public-symbol inventory → [`docs/METHODS.md`](docs/METHODS.md) index + per-layer files in [`docs/methods/`](docs/methods/). **`grep -rn <symbol> docs/methods/` before adding a helper.**
> - Agent-tool entry points → [`AGENTS.md`](AGENTS.md) is the tool-neutral root pointer (Cursor / Codex / Copilot / Aider read it) and **redirects here**; keep it a 3-line pointer, never a copy or symlink of this file. Vendor-neutral skills live in [`.agents/skills/`](.agents/skills/) (`SKILL.md` + `references/`). `AGENTS.md` itself stays at repo root — it does **not** belong under `.agents/`.

---

## 1. Operating Principles (read every session)

In priority order. When two conflict, the earlier wins.

1. **Safety beats helpfulness.** Refuse any request to bypass a safety check, silently catch `ROSSafetyViolation`, lower a velocity limit without a paper trail, or remove a deadman/E-stop subscription. Surface the concern, propose a safe alternative.
2. **Truth over plausibility.** Don't know a constant (DDS topic, FCI port, RealSense extrinsic)? Say so and look it up. Never invent. Never paraphrase a citation.
3. **Types are the contract.** Pydantic schemas in `python/openral_core/` and IDL in `packages/openral_msgs/` are normative API. Everything else is implementation detail.
4. **Explicit beats implicit.** No hidden retries, fallbacks, or magic globals. Replanning, dispatcher fallback, quantization, and license posture must show up in logs/traces.
5. **The hot path is C++ and bounded.** Python touches motors only through a typed bridge to `ros2_control` with a watchdog. Anything >100 Hz is C++ unless proven otherwise.
6. **Schemas evolve, but never silently.** Now the repo is published, on-disk `schema_version` is versioned for real: a backward-incompatible change bumps it and ships a migrator; backward-compatible additions may evolve in place. Every change still needs (a) an ADR if it crosses a layer boundary, (b) a test loading a real fixture from `robots/`, `rskills/`, or `scenes/`.
7. **Tests are part of the change.** Every PR ships the tests that would have caught the bug or covered the feature. Untested actuation-path code is rejected.
8. **Reproducibility over speed.** A skill execution must be replayable from the trace alone (weights revision pinned, prompts logged, sensor frames captured).
9. **License lineage is enforced.** OpenRAL's own code is uniformly **Apache-2.0** — every package, no commercial / source-available / non-open tier ([ADR-0012](docs/adr/0012-open-core-licensing.md)); copy-left incoming is rejected without TSC review. Third-party model **weights** keep their upstream license — version-specific, not family-wide: GR00T N1/N1.5/N1.6 are non-commercial (loader refuses commercial deployment without `OPENRAL_ALLOW_NONCOMMERCIAL=1`), while GR00T N1.7+ ships under the commercially-permissive NVIDIA Open Model License ([ADR-0046](docs/adr/0046-nvidia-gr00t-backend.md)). This weight lineage is compliance for models OpenRAL does not own; it does not gate OpenRAL's Apache-2.0 code. Closed third-party SDK code is never bundled — it stays behind the license guard and an env var.
10. **Be helpful and honest.** Propose simpler approaches; surface tradeoffs; never apologize at the start of a response.
11. **Real components, not mocks.** Tests, examples, demos exercise real schemas, real `RobotDescription` manifests, real `rSkill` packages, real simulators. No mocks, stubs, smoke tests, `--dry-run` / `--collect-only` substitutes. Unavailable dependency → `pytest.skip(reason=...)`, never faked. The only acceptable doubles are at process/network boundaries (fake OTLP collector, recorded HF Hub response) under `tests/<tier>/fakes/`. Pydantic models validate against fixtures in `robots/` / `rskills/` / `scenes/` — never `"foo"` / `"test"` placeholders.
12. **Prefer the local GPU when available.** Before CPU fallback or `pytest.skip`, check `nvidia-smi`. GPU present → install the right `uv sync --group …` and run for real. CI runners without GPUs are the legitimate skip path; a dev host with a GPU is not.
13. **Don't duplicate; consult [`docs/METHODS.md`](docs/METHODS.md) first.** Hand-curated public-symbol inventory — a slim index at `docs/METHODS.md` over per-layer files in `docs/methods/`. `grep -rn <symbol> docs/methods/` before writing a helper. Add/rename/move/remove a public symbol → update the matching `docs/methods/` file in the same PR (and recheck `docs/methods/14-duplication-watch.md`). Run `python tools/refresh_methods_linenos.py` to refresh `(LNN)` line markers. Missing entry = incomplete PR.
14. **Docs travel with the code.** Every PR updates all affected docs in the same commit range: relevant `README.md`, `docs/` pages, ADRs, the repo state map (§4.3), and `docs/METHODS.md`. "Docs follow-up" PRs are not allowed.
15. **Pre-existing errors get their own commit.** Drive-by lint/type/doc/test fixes go in a separate prior `fix(...)` commit on the same branch — never bundled into a feature commit, never left in place.

---

## 2. Coding Standards

**Python 3.12 only** (`pyproject.toml` pins `>=3.12,<3.13`). `mypy --strict` must pass; no `Any` without `# type: ignore[<rule>] # reason: <why>`. **Pydantic v2** for every schema/config/manifest/external interface; plain `dataclass` only inside a single non-boundary module. **Async-first** at the application boundary; sync for hot loops. No `time.sleep` in coroutines, no blocking I/O on the event loop. Absolute imports inside the workspace. Raise typed `ROSError` subclasses (§3); never `except Exception: pass`; never silence `ROSSafetyViolation`. `structlog` only, OTel handler; no INFO logs at import; no tensors at INFO. f-strings (no `%`-format outside the logger). Google-style docstrings with runnable `Example` for user-facing behavior. Line length 100. `_private` modules are not API; `__init__` re-exports are.

**C++17 minimum** (C++20 where ROS 2 distro allows). `clang-format` LLVM 100 col; `clang-tidy` with `cppcoreguidelines-*`, `bugprone-*`, `performance-*`, `readability-*`. No raw `new`/`delete`. No exceptions across the safety-kernel boundary — use `Result<T,E>` / `std::expected`. Realtime code: no allocations in hot loops, no `std::cout`, no `std::string` ops, pre-reserved containers, lock-free queues. Include order: own → C system → C++ std → other libs → ROS → project.

**ROS 2:** Lifecycle nodes for anything stateful. QoS by data class — sensors `BEST_EFFORT, VOLATILE, KEEP_LAST=5–10`; control `RELIABLE, VOLATILE, KEEP_LAST=1`; description/static `RELIABLE, TRANSIENT_LOCAL, KEEP_LAST=1`; safety/e-stop `RELIABLE, VOLATILE, KEEP_LAST=10`; action feedback `BEST_EFFORT`. Actions for >100 ms or cancellable work; services for instantaneous queries; topics for streams. Composable nodes for intra-process zero-copy. TF2 is the only source of coordinate frames — never inline a 4×4 matrix, never hardcode a `frame_id`. Middleware: Cyclone default; Zenoh for fleet/cloud/wireless, via `ROS_AGENT_MIDDLEWARE`.

**Tests:** No mocks/stubs/smoke tests (§1.11). Unit (`tests/unit/`) <30 s. Integration (`tests/integration/`) uses `launch_testing` + real lifecycle nodes + real IDL; <5 min. Sim (`tests/sim/`) headless MuJoCo MJX vs real LIBERO / MetaWorld / gym-aloha / gym-pusht; <10 min; naming `test_<robot>_<vla>_<sim>.py` (HAL-only sim → `<vla>=hal`). HIL (`tests/hil/`) gated by `[self-hosted, lab-<robot>]`, idempotent, <10 min, e-stop hook on teardown. `hypothesis` fuzz on every Pydantic model (round-trip + JSON Schema). Every public docstring `Example` runs. Per-skill latency budgets enforced on the reference host.

---

## 3. Architecture Discipline

**The eight layers** (do not cross without an ADR in `docs/adr/`):

```
0 HAL  ← packages/openral_hal_*/                3 rSkill (S1)  ← python/rskill/, packages/openral_skill/
1 Sensors  ← packages/openral_sensors/          4 Reasoning (S2)  ← python/openral_reasoner/
2 World State  ← packages/openral_world_state/  5 World Action Model  ← python/openral_wam/
                                                6 Safety  ← packages/openral_safety/, cpp/openral_safety_kernel/
                                                7 Observability  ← python/openral_observability/
```

Adding, removing, renaming, or moving a responsibility between layers → ADR required. A non-adjacent-layer dependency (Skill calling HAL directly, etc.) is rejected.

**Dual-system pattern.** **S1** fast policy (30–200 Hz, action chunks) as a `Skill`. **S2** slow reasoning (event-driven; ~0.2 Hz heartbeat) as the `Reasoner` — emits typed tool calls (`ExecuteSkill`, `ReloadGstPipeline`, `LifecycleTransition`, `EmitPrompt`) via the `ReasonerToolCall` discriminated union; BT v4 XML is a future option behind `bt_executor_node` (ADR-0018 §4 / F4 + 2026-05-25 amendment). **S0** cerebellar layer (500–1000 Hz, C++ only) in `ros2_control` controllers, for humanoids. Skill manifest declares `role: s1 | s2 | s0`; loader enforces.

**rSkill packaging.** One HF Hub repo per skill: `rskill.yaml` (name, version, license, embodiment_tags, capabilities_required, runtime, quantization, latency budgets, fallback_skill_id; plus `state_contract.dim` + `action_contract.dim` for ADR-0019 dataset bridge users), weights (`model.safetensors`), optional `engine.plan`/`Dockerfile`, `README.md` per [`rskills/template/README.md`](rskills/template/README.md) (publish gate enforced by `rskill_publisher` validator), `eval/<benchmark>.json` validating against `openral_core.SkillEvalResult`. Canonical eval producer: `openral benchmark run --suite <id> --vla <vla_id>:rskills/<this_skill>` (ADR-0009 PR D). Paper-cited numbers allowed with `reproduced_locally: false` + `reproduction_cli`. **Provenance:** sigstore signing/verification is the planned control but is **not yet implemented** (ADR-0006 — no manifest `signature` field, no verification in the loader). Until it lands, `rSkill.from_pretrained`/`from_yaml` emit an `rskill.unverified_provenance` warning and honor `OPENRAL_REQUIRE_SIGNED_SKILLS=1` to fail closed; `*.pt` weights are treated as untrusted code and require `OPENRAL_ALLOW_UNSAFE_PICKLE=1` to load (prefer `model.safetensors`); `trust_remote_code` models (e.g. MolmoAct2) execute repo-shipped code and require `OPENRAL_ALLOW_REMOTE_CODE=1`. Do not describe skills as "signed/verified" until the control exists (§1.2).

**VLA license matrix.** SmolVLA, OpenVLA/OFT, Octo, ACT, DP/DP3, UnifoLM-VLA-0/WMA-0 — Apache-2.0 / MIT (free). π0 / π0.5 / π0.6 / π0.7 — code Apache-2.0; weights permissive research (flag in manifest). GR00T N1 / N1.5 / N1.6 — NVIDIA OneWay **Noncommercial** (install-time guard); GR00T **N1.7+** — NVIDIA Open Model License, **commercial OK** (`nvidia_open_model` posture); N2 announced, unreleased. GR00T runs out-of-process via a ZMQ sidecar (Py3.10), reusing the `rldx` adapter ([ADR-0046](docs/adr/0046-nvidia-gr00t-backend.md)). Helix / Gemini Robotics / Skild Brain — closed, API-only. Quantize for target; action chunks not single actions; embodiment tags must match `RobotCapabilities.embodiment_tags`; latency budget is contractual.

**Safety.** Touching `packages/openral_safety/` or `cpp/openral_safety_kernel/` requires (a) safety-WG reviewer, (b) hazard-log update, (c) tests proving the new behavior is at least as conservative. **Never** add a flag that disables safety. **Never** add a debug mode that bypasses E-stop. **Never** add a path where a Python crash leaves motors energized. Python proposes; C++ disposes.

**Reasoner & dispatch.** LLM tool calls are Pydantic structured output via the provider's tool-use API — no free-form JSON. Tool palette generated from the local skill registry, rebuilt on `/openral/skill_registry_changed`. Bounded replanning ladder (per-kind cap in `ReasonerCore`): retry → param-tweak → substitute-skill → goal-replan → human-handoff. LLM selected at activate-time via `OPENRAL_REASONER_LLM_*` env (`PROVIDER` ∈ {`anthropic`, `openai-compatible`, `openrouter`}); see [`packages/openral_reasoner_ros/README.md`](packages/openral_reasoner_ros/README.md). No hidden default. The dispatcher (edge/cloud/split), like every OpenRAL package, is Apache-2.0. Deadline fallback mandatory. No PII in cloud logs without consent.

**WAMs** are optional planning-layer components — mental-simulation gating, failure anticipation, replanning subgoals. Thor-class compute or cloud dispatch; deadline fallback applies. See [docs/roadmap/index.md](docs/roadmap/index.md).

---

## 4. Workflow & PR Checklist

### 4.1 Before you change anything

Re-read this file if >1 day or >1 PR since last; read relevant RFC/ADR sections; `just bootstrap && just test` once for a clean baseline; skim the last 20 commits on `master`.

### 4.2 Implementing a feature

1. **Plan** in 3–10 bullets in the PR description before coding. Layer boundary → write the ADR first.
2. **`grep -rn <symbol> docs/methods/`** for the helper you're about to write.
3. **Schemas first** if a typed contract is touched; validate against a real fixture.
4. **Tests first** for actuation-path code; TDD required for safety-touching code.
5. **Smallest viable PR.** >800 lines needs maintainer pre-approval.
6. **Pre-existing errors → separate prior `fix(...)` commit** on the same branch (§1.15).
7. **Update docs in the same PR** (§1.14).
8. **Conventional Commits** (`feat(skill): …`, `fix(safety): …`). Squash on merge only when the branch is a single logical change.
9. **Run full local CI** before pushing: `just lint && just test && just sim` (where applicable).

### 4.3 Repo state map

[`docs/architecture/repo-state-map.html`](docs/architecture/repo-state-map.html) is the visual source of truth — hand-edited, data-driven JS arrays (`LAYERS`, `CROSS`, `CANCELLED`, `SCHEMAS`), no build step, self-contained. **PR that changes the repo surface without updating the map is incomplete.** Update when you add/remove/rename a Python/ROS package (fix block & `pkg`), cross a status boundary (`green`/`yellow`/`blue`/`red` — `green` requires source **and** tests), add/remove a Pydantic model in `openral_core.schemas` / exception in `exceptions.py` / IDL in `packages/msgs/` (update `SCHEMAS` + per-block `schemas: […]`), or change a layer boundary / data-flow direction (revise layer `desc`, runtime data-flow strip, and affected `inputs`/`outputs`). Schema names must match `openral_core` / `openral_msgs` exactly.

### 4.4 PR checklist

- [ ] Conventional commit title; description has "What changed", "Why", "How tested".
- [ ] Schemas: real fixture validates; on-disk `schema_version` bumped + migrator shipped for any backward-incompatible change (post-publish — no longer frozen at `"0.1"`).
- [ ] Layer boundary crossed → ADR added.
- [ ] Tests: unit + integration + sim where applicable; HIL if a HAL changed. No new mocks/stubs/smoke tests (§1.11).
- [ ] The matching `docs/methods/` file updated for every added/renamed/removed/moved public symbol (signature + line number + layer section); `tools/refresh_methods_linenos.py --check` clean. Searched first.
- [ ] Docs updated in the same PR (READMEs, `docs/`, ADRs) — no follow-up deferrals.
- [ ] Pre-existing errors fixed in a separate prior `fix(...)` commit.
- [ ] Repo state map updated when a module is added/renamed/removed/status-flipped.
- [ ] `just lint` passes; `mypy --strict` clean; no new `# type: ignore` without `# reason: ...`; no new `try/except: pass`; no new global mutable state.
- [ ] Performance budgets met if relevant; no PII in fixtures/logs; new deps Apache-2.0 / MIT / BSD (no GPL without TSC review).

---

## 5. Exception Hierarchy (use these, do not invent)

```python
ROSError                            # base
├─ ROSConfigError                   # bad manifest, missing weights, invalid YAML
├─ ROSCapabilityMismatch            # skill needs lidar; robot lacks it
├─ ROSRuntimeError                  # ROSInferenceTimeout, ROSQuantizationError, ROSGPUMemoryError
├─ ROSSafetyViolation               # ROSWorkspaceViolation, ROSForceLimitExceeded, ROSEStopRequested
├─ ROSPerceptionStale               # sensor older than deadline
├─ ROSPlanningError                 # ROSReasonerInvalidPlan, ROSBTValidationError
└─ ROSFleetError                    # ROSDispatchUnavailable, ROSDeadlineMissed
```

`ROSSafetyViolation` is **never** caught except at the safety supervisor boundary, where it triggers E-stop + structured incident log.

---

## 6. When in doubt

Default to **safer → more typed → more observable → smaller → closer to convention**. Propose deviations from this file as an ADR or discussion, not a commit. STOP if you are about to disable a safety check, fake a benchmark number, bundle closed-source weights, add a new top-level package without an ADR, refactor across all 8 layers in one PR, or ship code you haven't run.

---

*End of CLAUDE.md. Last reviewed: update on every meaningful change. Stale = file an issue.*
