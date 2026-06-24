# Tools

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `tools/profile_policy_load.py`
_One-shot wall-time breakdown of a single policy load. Drives `openral_sim.factory.make_policy` against an in-tree rSkill manifest and prints a phase-by-phase summary built from every `<prefix>_<name>_{start,done}` event emitted by `openral_rskill._diagnostics.phase_timer`. Use when `ros2 launch openral_rskill_ros …_e2e.launch.py` or `openral sim run` is slow to first action — answers "where do the seconds go" before changing any code._

- `class _PhaseCapture` — `structlog` processor that buffers `_start` / `_done` events and pairs them by name. Insertion-ordered so the rendered table mirrors the actual load order. (L49)
- `_parse_args(argv) -> argparse.Namespace` — `--rskill <dir>` (required), `--device` (default `auto`). (L88)
- `_build_env_cfg(rskill_dir, *, device) -> _SimpleEnvCfg` — Builds a minimal env_cfg from `<rskill_dir>/rskill.yaml`; mirrors `rskill_runner_node._SimpleEnvCfg`. (L122)
- `_render(pairs, total_s) -> str` — Formats the captured pairs as `phase / elapsed_s / share` columns plus an `(unaccounted)` row when phase coverage misses >1 s. (L141)
- `main(argv=None) -> int` — Late-imports `openral_sim.factory.make_policy` so the import cost lands inside the profiled total; reports `HF_HUB_OFFLINE` status alongside the result.

### `tools/schema_export.py`
_Generates JSON Schema files for every public `openral_core` model._

- `_enum_schema(cls) -> dict[str, Any]` — Minimal JSON Schema for a `str` Enum. (L134)
- `export_schemas(out_dir=_OUT_DIR) -> dict[str, Any]` — Export JSON Schema for every public model. (L146)
- `check_drift(out_dir=_OUT_DIR) -> bool` — On-disk schemas == regenerated. (L199)

### `tools/audit_sim_configs.py`
_Real GPU rollout audit for every YAML under `scenes/`. Operator-driven (not a pytest test); 1 episode per config; writes `outputs/audit_sim_configs.json` and prints a Markdown table. See `just sim-audit`. Two modes: default (full rollout for sim/benchmark, Tier-2 launch + SIGINT for deploy) and `--check-compatibility` (cheap in-process scene+rSkill+HAL gate, no subprocess / no GPU)._

- `DEFAULT_TIMEOUT_S = 600` / `DEFAULT_DEPLOY_ALIVE_GRACE_S = 90` / `DEFAULT_DEPLOY_SHUTDOWN_GRACE_S = 30` (L57) — Module-level grace constants overridable via `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace`.
- `RunMode = Literal["sim", "benchmark", "deploy"]` (L61) — Tier selector on each `ConfigSpec` row; drives `_run_one` vs `_run_one_deploy` dispatch.
- `@dataclass(frozen=True) class ConfigSpec(config, rskill, uv_group, run_mode)` (L65) — One row in the audit catalogue. `uv_group` is one of `libero / metaworld / robocasa / maniskill3 / simpler-env / sim`; `run_mode` is `"sim"` / `"benchmark"` / `"deploy"`; `rskill` is `""` for deploy rows (env-only — reasoner picks at runtime). Catalogue holds only pairs that actually exist in the tree — scenes without a matching in-tree rSkill are tracked in the scene YAML itself (and in `tests/unit/test_examples_sim_configs_load.py` for schema-load coverage), not as audit rows.
- `CATALOGUE: tuple[ConfigSpec, ...]` (L100) — Explicit (YAML → rSkill → uv group → run_mode) mapping for every config currently in the tree: 13 sim + 7 benchmark + 4 deploy = 24 rows.
- `@dataclass class AuditRow(config, rskill, status, exit_code, wall_s, peak_vram_mib, tail)` (L195) — One result. `status` ∈ {`pass`, `pass-compat`, `fail-oom`, `fail-asset`, `fail-sidecar`, `fail-timeout`, `fail-other`, `fail-compat`, `skipped-opt-dep`, `skipped-host-setup`}.
- `_classify(returncode: int, tail: str) -> str` (L250) — Map subprocess result to a status by substring-matching stderr against `_OOM_PATTERNS` / `_ASSET_PATTERNS` / `_SIDECAR_PATTERNS` / `_OPT_DEP_PATTERNS` / `_HOST_SETUP_PATTERNS`. Exit 139 (MuJoCo/GL atexit SIGSEGV in gym-aloha) is treated as `pass` when no error patterns appear.
- `class _VramSampler` (L289) — Background `nvidia-smi --query-gpu=memory.used` poller, 200 ms cadence; `peak_mib` reported on `.stop()`. No-op without `nvidia-smi` on `$PATH`.
- `_check_compat(spec: ConfigSpec) -> AuditRow` (L334) — `--check-compatibility` gate: load scene via `openral_core.load_scene_strict`, validate rSkill manifest (sim/benchmark) or assert robot resolves in `openral_cli.deploy_sim._ROBOT_HAL_REGISTRY` (deploy). No subprocess, no GPU. Returns `pass-compat` / `fail-compat`.
- `_build_run_cmd(spec: ConfigSpec) -> list[str]` (L431) — Build the `uv run … openral <sim|benchmark> …` argv for sim/benchmark rows. Refactored out of `_run_one` so the deploy path can stay focused on lifecycle teardown.
- `_run_one_deploy(spec, *, alive_grace_s, shutdown_grace_s, timeout_s) -> AuditRow` (L480) — Tier-2 deploy launch via `openral deploy sim --config <yaml> --no-dashboard`: `Popen` in its own process group, wait `alive_grace_s`, send SIGINT to the group, wait `shutdown_grace_s`, escalate to SIGKILL on timeout. Pass criteria: banner seen in stdout AND returncode in `{0, -SIGINT, 130, -SIGTERM}`.
- `_classify_or_fallback(returncode, tail, spec, wall_s, peak_vram) -> AuditRow` (L665) — Deploy-mode wrapper around `_classify` that defaults to `fail-other` when no pattern matches (sim path defaults to `pass`).
- `_run_one(spec: ConfigSpec, timeout_s: int) -> AuditRow` (L710) — Tier-3 sim/benchmark rollout via `_build_run_cmd(spec)` with `MUJOCO_GL=egl` and `OPENRAL_SIM_SEQUENTIAL_INIT=1`.
- `main(argv) -> int` (L857) — CLI entry; flags `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace` / `--check-compatibility` / `--report`. Returns 0 on all-pass, 1 if any config failed, 2 on filter mismatch.

### `tools/select_tests.py`
_Selective test execution — maps a git diff to the minimal pytest targets that can observe it. Backs `just test-changed` / the `test-selective` workflow. See [`docs/contributing/selective-testing.md`](../contributing/selective-testing.md)._

- `class SelectionConfig(BaseModel)` (L65) — Typed view of `tools/test_selection.toml`: `full_run_globs`, `ignore_globs`, `isolate_globs`, `extra_triggers`.
- `class SelectionResult(BaseModel)` (L74) — `full_run` / `full_run_reason` / `affected_packages` / `targets` / `isolated_targets` (own-process, issue #24) / `reasons` (per-target rationale).
- `load_config(path) -> SelectionConfig` (L97) — Load + validate the TOML config.
- `package_dir_import_names(repo_root) -> dict[str, str]` (L109) — `python/<dir>` → its `src/openral_*` import name.
- `build_dependency_graph(repo_root) -> dict[str, set[str]]` (L128) — Import-name → direct `openral` deps, derived from each `pyproject.toml` (never hand-written).
- `transitive_dependents(graph, changed) -> set[str]` (L152) — Closure of packages that depend on any changed package (includes `changed`).
- `map_test_imports(repo_root) -> dict[str, set[str]]` (L179) — Each top-level `tests/` file → the `openral_*` packages it imports.
- `select(repo_root, changed_files, config) -> SelectionResult` (L260) — Resolve changed paths to pytest targets (blast-radius → full run; else per-package dirs + import-intersecting tests), peeling `isolate_globs` matches into `isolated_targets`.
- `changed_files_from_git(base, head, repo_root) -> list[str]` (L364) — Merge-base `git diff --name-only base...head`.
- `main(argv=None) -> int` (L399) — CLI; `--files` / `--base/--head`, `--github-output` for CI step outputs.

### `tools/audit_tests.py`
_Test-suite auditor — flags dead / shadowed / duplicate / no-assertion tests; writes `docs/contributing/test-audit.md`. Read-only; never deletes. Backs `just test-audit`._

- `class TestFuncInfo(BaseModel)` (L73) — One `test_*` function: `path`, `qualname` (class-scoped), `markers`, `has_assertion`, `is_trivial`, `body_hash`, …
- `class DuplicateGroup(BaseModel)` (L88) — A `body_hash` shared by ≥2 functions and their `members`.
- `class AuditReport(BaseModel)` (L93) — Inventory (`by_tier`/`by_marker`/`by_directory`) plus `trivial` / `shadowed` / `no_assertion` / `duplicate_groups`.
- `collect(repo_root) -> list[TestFuncInfo]` (L212) — Scope-aware AST walk over every test root (same name in two classes is not conflated).
- `build_report(records) -> AuditReport` (L245) — Group into the inventory + finding buckets; `shadowed` = same `(path, qualname)` redefined (earlier def is dead).
- `render_markdown(report) -> str` (L297) — Render the committed report.
- `main(argv=None) -> int` (L384) — CLI; `--json` / `--write-report`.

### `python/observability/src/openral_observability/replay/`
_ADR-0018 F7 — query-time joiner for rosbag2 (mcap) ↔ OTel spans. Backs `openral replay` + `openral record`._

- `bag_reader.py`:
  - `@dataclass(frozen=True) class BagMessage(topic, log_time_ns, publish_time_ns, trace_id, traceparent, schema_name, payload_summary)` (L43) — One mcap record surfaced to the correlator.
  - `read_bag(bag_path: str | Path) -> Iterator[BagMessage]` (L109) — Iterate an `.mcap` file or rosbag2 directory; extracts `trace_id` from `jsonschema`-encoded payloads (a packed W3C `traceparent` in the `trace_id` field, OR — for the `openral_msgs/Tick` schema — a raw 32-hex `trace_id` + raw 16-hex `span_id` pair, ISSUE-109) or `ros2msg`-encoded CDR payloads (regex match on the W3C `traceparent` substring). No `rosbag2_py` dep.
- `trace_query.py`:
  - `class TraceQueryError(RuntimeError)` (L17) — Raised on a non-JSON or unreachable dashboard response.
  - `@dataclass(frozen=True) class DashboardTraceClient(base_url="http://127.0.0.1:8000", timeout_s=5.0)` (L24) — `list_traces() -> list[dict]` over `/api/traces`; `get_spans(trace_id) -> list[dict]` over `/api/spans/<id>`.
- `correlator.py`:
  - `@dataclass(frozen=True) class TimelineEntry(kind, ts_ns, trace_id, topic, span_name, attrs, duration_ms)` (L26) — One row of the joined timeline; `.to_json()` returns a plain dict.
  - `list_bag_trace_ids(bag_messages) -> list[dict]` (L67) — Distinct trace_ids in the bag with counts, busiest first.
  - `build_timeline(bag_messages, spans, *, trace_id=None) -> list[TimelineEntry]` (L94) — Pure join. Filters both inputs to `trace_id`, merges, sorts ascending by `ts_ns`.
- `cli.py`:
  - `RECORD_PROFILES: dict[str, dict[str, list[str]]]` (L45) — Slim and full topic + regex presets matching ADR-0018 §F7.
  - `build_record_command(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=()) -> list[str]` (L85) — Compose `ros2 bag record` argv.
  - `@dataclass(frozen=True) class ReplayResult(trace_id, bag_trace_ids, timeline, bag_path)` (L132) — `.to_json()` returns a plain dict.
  - `run_replay(*, bag_path, trace_id, dashboard_url) -> ReplayResult` (L162) — Read a bag, fetch matching spans from the dashboard, return the joined timeline.
  - `run_record(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=(), dry_run=False) -> tuple[list[str], CompletedProcess | None]` (L210) — Spawn `ros2 bag record` in a new process group; forwards SIGINT/SIGTERM received by the parent as **SIGINT** to the child group so rosbag2 flushes `metadata.yaml` cleanly. Waits up to 5 s after the child exits for that file to appear.
  - `write_timeline(result: ReplayResult, out_path: Path) -> None` (L246) — Persist the timeline JSON.

### `tools/rskill_publisher.py`
_Package and publish a local rSkill directory to the HF Hub._

- `public_visibility_error(manifest, public) -> str | None` (L76) — §9 license gate (pure, no network): returns an error string if `--public` is requested for a non-commercial-licensed skill (`not manifest.is_commercial_use_allowed`), else `None`. Lets `main` fail fast before any HF call.
- `_resolve_token(token_arg) -> str` — Prefer CLI arg, fall back to env. (L121)
- `_validate_manifest(skill_dir) -> RSkillManifest` (L140)
- `_validate_docs(skill_dir, manifest) -> DocValidationReport` — Print + return the README / manifest documentation report via `_rskill_doc_validator.validate_rskill_docs`. Runs in both dry-run and `--publish` paths; the caller decides whether to exit on errors.
- `_bump_revision(manifest_path, weights_uri_base, token) -> str` — Resolve latest weights commit, patch `rskill.yaml`. (L229)
- `_ensure_private(api, repo_id) -> None` — Abort if the repo is public. (L280)
- `_ensure_public(api, repo_id) -> None` — The `--public` counterpart: abort if the (reused) repo is private, so a `--public` publish never lands in a private repo.
- `_publish(skill_dir, manifest, token, *, public=False) -> str` — Create the HF repo (private unless `public`) and upload; runs the matching visibility gate (`_ensure_public` / `_ensure_private`) after `create_repo`.
- `main() -> None` — Entry point. Sequence: parse args (`--publish` / `--public` / `--bump-revision` / `--token`) → validate manifest → validate docs → `public_visibility_error` gate (exit 1 if `--public` on a non-commercial skill) → exit 1 on doc errors → optional `--bump-revision` → `--publish` (private unless `--public`).

### `tools/rskill_scaffolder.py`
_Standalone argparse wrapper around `openral_cli._rskill_scaffolder.scaffold_rskill`._
Mirrors `openral rskill new`; exists so power users can scaffold without installing the CLI distribution.

- `_parse_args(argv) -> argparse.Namespace` — argparse setup. (L35)
- `main(argv=None) -> int` — Entry point; returns a process exit code. (L77)

### `tools/generate_rskill_skillmd.py`
_Generate the standard agent-skill `SKILL.md` discovery view for every in-tree rSkill from its `rskill.yaml`._
The single canonical producer of the `SKILL.md` mirror (CLAUDE.md §1.3): `rskill.yaml` is authoritative; the generated `SKILL.md` is discovery-only and never hand-edited. `--check` fails on any stale/missing `SKILL.md`, so the same process applies to every kind — including `playbook` (ADR-0072), whose `_KIND_NOUN` entry renders identically to `vla`/`detector`/`vlm`/`reward`.

- `render_skill_md(manifest_path: Path) -> str` (L162) — Render the `SKILL.md` text (YAML frontmatter + capability/verb summary + license/provenance) from one manifest; `_KIND_NOUN` maps each `kind` to its discovery noun.
- `main(argv=None) -> int` (L261) — Entry point. No args = regenerate every `rskills/<id>/SKILL.md`; positional ids regenerate a subset; `--check` reports stale/missing without writing (exit 1 on drift).

### `tools/rldx_sidecar.py`
_Boot helper for the RLDX-1 inference sidecar (companion to `openral_sim.policies.rldx`)._
Materialises a Python 3.10 venv under `_DEFAULT_HOME` (`~/.cache/openral/rldx-sidecar`, override via `--home`), clones the upstream `RLWRLD/RLDX-1` repo, runs `uv sync` (rldx + transformers + flash-attn + …), optionally adds `bitsandbytes` for NF4, then writes a wrapper that monkey-patches `transformers.AutoModel.from_pretrained` to apply NF4 / int8 to the Qwen3-VL-8B backbone (the MSAT diffusion head is left at bf16) and `os.execvpe`s into `rldx.eval.run_rldx_server`. Required because the `rldx` package pins `requires-python = "~=3.10"` and ships a custom `architectures=["RLDX"]` class not in HF Transformers.

- `_install_deps(*, source, uv, quantization) -> Path` — `uv sync` in the cloned source tree (creates `source/.venv` with rldx + deps), then `uv pip install --python <venv>/bin/python bitsandbytes>=0.43.0` when `quantization in {nf4, int8}`. Returns the venv path. (L48)
- `_make_wrapper(*, work, source, args) -> Path` — Generate `<work>/boot_server.py`: monkey-patches `AutoModel.from_pretrained` for the Qwen3-VL backbone with NF4 / int8 / no-op, sets `sys.argv`, and calls into `rldx.eval.run_rldx_server`. (L82)
- `main() -> int` — argparse entry point; flags `--model`, `--port`, `--quantization {none,nf4,int8}`, `--home`. Calls `run_sidecar(..., family="rldx", ...)`, which stamps the sidecar identity record (so the adapter can verify reuse) and then `os.execvpe`s into the sidecar venv so SIGINT reaches the server. (L237)

### `tools/qwen_vlm_sidecar.py` + `tools/_qwen_vlm_server.py`
_Boot helper + server for the Qwen3.5-4B scene-VLM sidecar (ADR-0047), companion to `openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`._ The launcher provisions an isolated venv (`OPENRAL_QWEN_VLM_SIDECAR_VENV` to reuse one) with transformers + bitsandbytes + `qwen-vl-utils` + pyzmq/msgpack, then `os.execvpe`s into the server. The server answers a ZMQ REQ/REP + msgpack protocol (`{"op":"query","image","question"}` → `{"ok","answer"}`); out-of-process for dependency/VRAM isolation (same pattern as `rldx_sidecar`). Apache-2.0 model.

- `ensure_venv(home, *, override=None) -> Path` (sidecar) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded); honours `$OPENRAL_QWEN_VLM_SIDECAR_VENV`.
- `main() -> int` (sidecar) — argparse (`--model`, `--host`, `--port`, `--max-side`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME` and `os.execvpe`s into `_qwen_vlm_server.py`.
- `_load(model_id) -> (processor, model)` / `_query(...) -> str` / `main() -> int` (server) — dual-path NF4 load (auto-detect a pre-quantized checkpoint via the embedded `quantization_config` → load 4-bit directly; else quantize-at-load, serial materialization for 8 GB); one scene-question→answer generate via the canonical Qwen-VL recipe (strips the `<think>` trace); ZMQ REP loop (`ping`/`query`/`shutdown`). Validated live (CLAUDE.md §1.2).

### `tools/robometer_sidecar.py` + `tools/_robometer_server.py`
_Boot helper + stateless scoring server for the Robometer-4B reward-monitor sidecar (ADR-0057), companion to `openral_runner.backends.reward.robometer_reward.RobometerReward`._ The launcher provisions an isolated venv (`OPENRAL_ROBOMETER_SIDECAR_VENV` to reuse one) by `uv pip install`ing the pinned `robometer` package (commit `a669dffc`) then **forcing `transformers==4.57.1`** (the resolver pulls 5.x, which breaks the processor), plus pyzmq/msgpack; sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; strips `PYTHONPATH`/`PYTHONHOME`; `os.execvpe`s into the server. The server loads the NF4 RBM once (vanilla `AutoModel` can't load it — no `auto_map`; loads via `robometer.utils.save.load_model_from_hf`) and answers a ZMQ REQ/REP + msgpack protocol (`{"op":"score","frames","n","width","height","task","num_bins"}` → `{"ok","progress","success"}`). Out-of-process for dependency/VRAM isolation. `robometer` is not an OpenRAL-trusted org (pinned, isolated). Validated live: 3.33 GB NF4, progress ramps to 0.88 + success 0.90 at task completion.

- `ensure_venv(home, *, override=None) -> Path` (sidecar) — provision/reuse the sidecar venv (pinned robometer + transformers 4.57.1 + pyzmq/msgpack; sentinel-guarded); honours `$OPENRAL_ROBOMETER_SIDECAR_VENV`.
- `main() -> int` (sidecar) — argparse (`--weights`, `--host`, `--port`, `--home`, `--venv`); `os.execvpe`s into `_robometer_server.py`.
- `_quantize_nf4_in_place(root, compute_dtype) -> int` / `class _Scorer` (`score(frames_rgb, task, num_bins) -> (progress, success)`, discrete mode) / `main() -> int` (server) — inlined NF4 rewrite (the `openral_sim._quantization` rule), load-once + per-clip scoring, BGR→RGB conversion, ZMQ REP loop (`ping`/`score`/`shutdown`).

### `tools/build_qwen_vlm_nf4_checkpoint.py`
_Reproducible recipe for the published `OpenRAL/rskill-qwen35-4b-nf4` pre-quantized NF4 checkpoint (ADR-0047). Runs in the sidecar venv._ `main() -> int` — argparse (`--source`, `--out`); loads the upstream model once (NF4 + serial materialization so the bf16 pass fits 8 GB), `save_pretrained`s the 4-bit weights + processor, then verifies the checkpoint reloads directly as 4-bit (no bf16 spike) and answers a smoke query. Pre-quantizing lets deployment load the 4-bit weights directly (~3.3 GB) with no loader workaround. Distinct from `quantize_rskill.py`, which writes an `install_prequantized_linears`-loaded pack for the in-process lerobot runtime; this writes a transformers-native `save_pretrained` checkpoint for the isolated VLM sidecar.

### `tools/fix_libero_config.py`
_Auto-fix for the stale `~/.libero/config.yaml` pitfall._

Detects and repairs `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero/config.yaml`) when its absolute paths no longer match the currently-active `libero` package — the file is written once at first LIBERO import and never refreshed, so switching venv / clone / workspace path leaves it pointing at a directory that no longer exists. Wired into the `Justfile` as `_ensure-libero-config` (chained off `sim-libero` / `sim-xvla-libero` / `sim-pi05-libero`). Idempotent.

- `_expected_config(libero_pkg_dir) -> dict[str, str]` — Compute the canonical `assets/bddl_files/benchmark_root/datasets/init_states` payload that LIBERO writes on first import. (L47)
- `_parse_yaml_map(text) -> dict[str, str]` — Parse the flat `key: value` map LIBERO writes — no PyYAML dependency. (L65)
- `_render_yaml(payload) -> str` — Render the same flat layout. (L79)
- `_locate_active_libero() -> Path` — `import libero` and return its package directory; raises `RuntimeError` with a clear message when LIBERO is absent (caller treats as no-op). (L84)
- `main() -> int` — argparse entry point; flags `--dry-run`, `--verbose`. Returns 0 when the config matches or after rewriting. (L110)
