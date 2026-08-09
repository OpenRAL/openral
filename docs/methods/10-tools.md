# Tools

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `tools/lifecycle_autostart.py`
_Drives a lifecycle node through `configure` → `activate` after `ros2 launch` brings it up. Spawned as an `ExecuteProcess` per node by `packages/openral_rskill_ros/launch/sim_e2e.launch.py` (HAL, safety kernel, reasoner)._

- `--transition-timeout-s` — per-transition spin budget, **default `300.0`** (`lifecycle_autostart.py:161`). The kernel gets a 120 s literal and the reasoner 300 s; the **HAL's is derived per scene** by `openral_hal.sim_bringup.hal_transition_timeout_s` (`sim_e2e.launch.py`), not hardcoded. There is still no `openral deploy sim` flag — raise the scene's `backend_options.boot_timeout_s` instead and the lifecycle budget follows.
- **Why the HAL's is derived.** The bound covers everything the scene factory does inside `on_configure`, including a sidecar's spawn + `_wait_for_boot`. Those backends carry their own, larger budgets — `isaac_sim` 900 s, `behavior` 1200 s, `robotwin` 600 s (`rlbench` 300 s, and it passes the constant directly so `backend_options.boot_timeout_s` is silently ignored) — and all four `scenes/deploy/isaac_*.yaml` plus `scenes/deploy/behavior_r1pro.yaml` set `boot_timeout_s: 1200`. Against the old fixed 300 s those five scenes declared a budget the launcher would not honour. **Measured** (`scenes/sim/isaac_franka_bowl_plate.yaml`, headless, RTX 4070 Laptop 8 GB): Isaac Sim 5.1 reaches `app ready` in **12.6 s**, so nominal boot is *not* the problem. The reachable case is a sidecar that wedges after startup — observed live on this host, where `connect()` waited the client's full **1200 s** before raising `did not answer ping within 1200s`. The transition's worst case is therefore the declared `boot_timeout_s`, not the nominal boot time. On expiry `_drive_transition` raises and the autostart exits non-zero **while `on_configure` keeps running**, which can leave the HAL parked in INACTIVE with nothing left to drive ACTIVATE and no message naming the cause. Cost of the fix: on those scenes a genuinely wedged HAL now goes unreported for up to ~21 min, which is why the 300 s floor still applies to every scene that does not ask for more.

### `tools/profile_policy_load.py`
_One-shot wall-time breakdown of a single policy load. Drives `openral_sim.factory.make_policy` against an in-tree rSkill manifest and prints a phase-by-phase summary built from every `<prefix>_<name>_{start,done}` event emitted by `openral_rskill._diagnostics.phase_timer`. Use when `ros2 launch openral_rskill_ros …_e2e.launch.py` or `openral sim run` is slow to first action — answers "where do the seconds go" before changing any code._
_Wired as `just profile-load <rskill> [args]` (Justfile). Measured on an RTX 4070 with `rskills/act-so101-pen` (warm cache): imports 4.6 s (54%), snapshot 0.3 s, from_pretrained 0.7 s, to_device 0.0 s, 8.5 s end-to-end — the import tax dominates even a small ResNet+transformer policy. Families whose adapter lacks a `_<family>_phase` helper (`xvla`, `diffusion`) report "no phase_timer events captured"; the end-to-end total is still valid._


- `class _PhaseCapture` — `structlog` processor that buffers `_start` / `_done` events and pairs them by name. Insertion-ordered so the rendered table mirrors the actual load order. (L49)
- `_parse_args(argv) -> argparse.Namespace` — `--rskill <dir>` (required), `--device` (default `auto`). (L88)
- `_build_env_cfg(rskill_dir, *, device) -> _SimpleEnvCfg` — Builds a minimal env_cfg from `<rskill_dir>/rskill.yaml`; mirrors `rskill_runner_node._SimpleEnvCfg`. (L122)
- `_render(pairs, total_s) -> str` — Formats the captured pairs as `phase / elapsed_s / share` columns plus an `(unaccounted)` row when phase coverage misses >1 s. (L141)
- `main(argv=None) -> int` — Late-imports `openral_sim.factory.make_policy` so the import cost lands inside the profiled total; reports `HF_HUB_OFFLINE` status alongside the result.

### `tools/viz_collision.py`
_Overlays a robot's **kernel** collision primitives (the box/capsule geometry the C++ safety kernel checks, lowered by `collision_params_from_description`) on its real MJCF meshes at any joint pose — the offline way to eyeball whether the SO-101 `base` OBB (issue #84) hugs the housing and clears the folded distal links without a `deploy run`. Standalone inspection tool, not a pytest test. Run the venv python with `PYTHONPATH=packages/openral_safety`._

- `--viewer` — interactive MuJoCo window (`MUJOCO_GL=glfw`). `--screenshot PATH` — offscreen PNG (`MUJOCO_GL=egl`). `--rviz` — real RViz (spawns `robot_state_publisher` for RobotModel + TF and publishes the primitives as a latched `/collision_markers` MarkerArray; needs ROS sourced). `--robot <id>` (default `so101_follower`), `--deg <j...>` sets the pose in degrees (manifest joint order). Box = translucent red, capsules = translucent blue (cylinder + end spheres in RViz).

### `tools/schema_export.py`
_Generates JSON Schema files for every public `openral_core` model._

- `_enum_schema(cls) -> dict[str, Any]` — Minimal JSON Schema for a `str` Enum. (L145)
- `export_schemas(out_dir=_OUT_DIR) -> dict[str, Any]` — Export JSON Schema for every public model. (L157)
- `check_drift(out_dir=_OUT_DIR) -> bool` — On-disk schemas == regenerated. (L209)

### `tools/audit_sim_configs.py`
_Real GPU rollout audit for every YAML under `scenes/`. Operator-driven (not a pytest test); 1 episode per config; writes `outputs/audit_sim_configs.json` and prints a Markdown table. See `just sim-audit`. Two modes: default (full rollout for sim/benchmark, Tier-2 launch + SIGINT for deploy) and `--check-compatibility` (cheap in-process scene+rSkill+HAL gate, no subprocess / no GPU)._

- `DEFAULT_TIMEOUT_S = 600` / `DEFAULT_DEPLOY_ALIVE_GRACE_S = 90` / `DEFAULT_DEPLOY_SHUTDOWN_GRACE_S = 30` (L57) — Module-level grace constants overridable via `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace`.
- `RunMode = Literal["sim", "benchmark", "deploy"]` (L61) — Tier selector on each `ConfigSpec` row; drives `_run_one` vs `_run_one_deploy` dispatch.
- `@dataclass(frozen=True) class ConfigSpec(config, rskill, uv_group, run_mode)` (L65) — One row in the audit catalogue. `uv_group` is one of `libero / metaworld / robocasa / maniskill3 / simpler-env / sim`; `run_mode` is `"sim"` / `"benchmark"` / `"deploy"`; `rskill` is `""` for deploy rows (env-only — reasoner picks at runtime). Catalogue holds only pairs that actually exist in the tree — scenes without a matching in-tree rSkill are tracked in the scene YAML itself (and in `tests/unit/test_examples_sim_configs_load.py` for schema-load coverage), not as audit rows.
- `CATALOGUE: tuple[ConfigSpec, ...]` (L99) — Explicit (YAML → rSkill → uv group → run_mode) mapping for every config currently in the tree: 13 sim + 7 benchmark + 4 deploy = 24 rows.
- `@dataclass class AuditRow(config, rskill, status, exit_code, wall_s, peak_vram_mib, tail)` (L195) — One result. `status` ∈ {`pass`, `pass-compat`, `fail-oom`, `fail-asset`, `fail-sidecar`, `fail-timeout`, `fail-other`, `fail-compat`, `skipped-opt-dep`, `skipped-host-setup`}.
- `_classify(returncode: int, tail: str) -> str` (L250) — Map subprocess result to a status by substring-matching stderr against `_OOM_PATTERNS` / `_ASSET_PATTERNS` / `_SIDECAR_PATTERNS` / `_OPT_DEP_PATTERNS` / `_HOST_SETUP_PATTERNS`. Exit 139 (MuJoCo/GL atexit SIGSEGV in gym-aloha) is treated as `pass` when no error patterns appear.
- `class _VramSampler` (L289) — Background `nvidia-smi --query-gpu=memory.used` poller, 200 ms cadence; `peak_mib` reported on `.stop()`. No-op without `nvidia-smi` on `$PATH`.
- `_check_compat(spec: ConfigSpec) -> AuditRow` (L334) — `--check-compatibility` gate: load scene via `openral_core.load_scene_strict`, validate rSkill manifest (sim/benchmark) or assert robot resolves in `openral_cli.deploy_sim._ROBOT_HAL_REGISTRY` (deploy). No subprocess, no GPU. Returns `pass-compat` / `fail-compat`.
- `_build_run_cmd(spec: ConfigSpec) -> list[str]` (L431) — Build the `uv run … openral <sim|benchmark> …` argv for sim/benchmark rows. Refactored out of `_run_one` so the deploy path can stay focused on lifecycle teardown.
- `_run_one_deploy(spec, *, alive_grace_s, shutdown_grace_s, timeout_s) -> AuditRow` (L480) — Tier-2 deploy launch via `openral deploy sim --config <yaml> --no-dashboard`: `Popen` in its own process group, wait `alive_grace_s`, send SIGINT to the group, wait `shutdown_grace_s`, escalate to SIGKILL on timeout. Pass criteria: banner seen in stdout AND returncode in `{0, -SIGINT, 130, -SIGTERM}`.
- `_classify_or_fallback(returncode, tail, spec, wall_s, peak_vram) -> AuditRow` (L665) — Deploy-mode wrapper around `_classify` that defaults to `fail-other` when no pattern matches (sim path defaults to `pass`).
- `_run_one(spec: ConfigSpec, timeout_s: int) -> AuditRow` (L710) — Tier-3 sim/benchmark rollout via `_build_run_cmd(spec)` with `MUJOCO_GL=egl` and `OPENRAL_SIM_SEQUENTIAL_INIT=1`.
- `main(argv) -> int` (L848) — CLI entry; flags `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace` / `--check-compatibility` / `--report`. Returns 0 on all-pass, 1 if any config failed, 2 on filter mismatch.

### `tools/select_tests.py`
_Selective test execution — maps a git diff to the minimal pytest targets that can observe it. Backs `just test-changed` / the `test-selective` workflow. See [`docs/contributing/selective-testing.md`](../contributing/selective-testing.md)._

- `class SelectionConfig(BaseModel)` (L65) — Typed view of `tools/test_selection.toml`: `full_run_globs`, `ignore_globs`, `isolate_globs`, `extra_triggers`.
- `class SelectionResult(BaseModel)` (L75) — `full_run` / `full_run_reason` / `affected_packages` / `targets` / `isolated_targets` (own-process, issue #24) / `reasons` (per-target rationale).
- `load_config(path) -> SelectionConfig` (L99) — Load + validate the TOML config.
- `package_dir_import_names(repo_root) -> dict[str, str]` (L111) — `python/<dir>` → its `src/openral_*` import name.
- `build_dependency_graph(repo_root) -> dict[str, set[str]]` (L130) — Import-name → direct `openral` deps, derived from each `pyproject.toml` (never hand-written).
- `transitive_dependents(graph, changed) -> set[str]` (L154) — Closure of packages that depend on any changed package (includes `changed`).
- `map_test_imports(repo_root) -> dict[str, set[str]]` (L181) — Each top-level `tests/` file → the `openral_*` packages it imports.
- `select(repo_root, changed_files, config) -> SelectionResult` (L297) — Resolve changed paths to pytest targets (blast-radius → full run; else per-package dirs + import-intersecting tests), peeling `isolate_globs` matches into `isolated_targets`.
- `changed_files_from_git(base, head, repo_root) -> list[str]` (L404) — Merge-base `git diff --name-only base...head`.
- `main(argv=None) -> int` (L444) — CLI; `--files` / `--base/--head`, `--github-output` for CI step outputs.

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
_Query-time joiner for rosbag2 (mcap) ↔ OTel spans. Backs `openral replay` + `openral record`._

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
  - `RECORD_PROFILES: dict[str, dict[str, list[str]]]` (L45) — Slim and full topic + regex presets.
  - `build_record_command(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=()) -> list[str]` (L85) — Compose `ros2 bag record` argv.
  - `@dataclass(frozen=True) class ReplayResult(trace_id, bag_trace_ids, timeline, bag_path)` (L132) — `.to_json()` returns a plain dict.
  - `run_replay(*, bag_path, trace_id, dashboard_url) -> ReplayResult` (L162) — Read a bag, fetch matching spans from the dashboard, return the joined timeline.
  - `run_record(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=(), dry_run=False) -> tuple[list[str], CompletedProcess | None]` (L210) — Spawn `ros2 bag record` in a new process group; forwards SIGINT/SIGTERM received by the parent as **SIGINT** to the child group so rosbag2 flushes `metadata.yaml` cleanly. Waits up to 5 s after the child exits for that file to appear.
  - `write_timeline(result: ReplayResult, out_path: Path) -> None` (L246) — Persist the timeline JSON.

### `tools/rskill_publisher.py`
_Package and publish a local rSkill directory to the HF Hub._

- `public_visibility_error(manifest, public) -> str | None` (L106) — §9 license gate (pure, no network): returns an error string if `--public` is requested for a non-commercial-licensed skill (`not manifest.is_commercial_use_allowed`), else `None`. Lets `main` fail fast before any HF call.
- `_resolve_token(token_arg) -> str` — Prefer CLI arg, fall back to env. (L151)
- `_validate_manifest(skill_dir) -> RSkillManifest` (L170)
- `_validate_docs(skill_dir, manifest) -> DocValidationReport` — Print + return the README / manifest documentation report via `_rskill_doc_validator.validate_rskill_docs`. Runs in both dry-run and `--publish` paths; the caller decides whether to exit on errors.
- `_rewrite_manifest_name(manifest_path, old_name, new_name) -> None` — Rewrite the top-level `name:` value of `rskill.yaml` in place (column-0 line only, so nested `name:` keys are untouched; preserves quotes + trailing comment). Exits 1 if the line isn't found exactly once. Backs `--fix-name`.
- `_enforce_repo_name(skill_dir, manifest, *, fix_name) -> RSkillManifest` — Enforce the ratified naming grammar via `openral_core.schemas.repo_name_is_canonical` (kind-aware: `rskill-playbook-<name>` for playbooks, `rskill-<model>-<robot>-<task>-<quant>` otherwise). No kind is exempt. On a non-canonical name: `fix_name=True` rewrites to `expected_repo_name(manifest)` + reloads; `fix_name=False` hard-fails (exit 1) printing the suggested name (exit 1 too if no canonical name can be suggested). Runs in both dry-run and `--publish` paths.
- `_bump_revision(manifest_path, weights_uri_base, token) -> str` — Resolve latest weights commit, patch `rskill.yaml`. (L376)
- `_ensure_private(api, repo_id) -> None` — Abort if the repo is public. (L427)
- `_ensure_public(api, repo_id) -> None` — The `--public` counterpart: abort if the (reused) repo is private, so a `--public` publish never lands in a private repo.
- `_publish(skill_dir, manifest, token, *, public=False) -> str` — Create the HF repo (private unless `public`) and upload; runs the matching visibility gate (`_ensure_public` / `_ensure_private`) after `create_repo`.
- `main() -> None` — Entry point. Sequence: parse args (`--publish` / `--public` / `--bump-revision` / `--fix-name` / `--token`) → validate manifest → `_enforce_repo_name` (exit 1 on a non-compliant VLA name unless `--fix-name`) → validate task space → validate docs → `public_visibility_error` gate (exit 1 if `--public` on a non-commercial skill) → exit 1 on doc errors → optional `--bump-revision` → `--publish` (private unless `--public`).

### `tools/rskill_scaffolder.py`
_Standalone argparse wrapper around `openral_cli._rskill_scaffolder.scaffold_rskill`._
Mirrors `openral rskill new`; exists so power users can scaffold without installing the CLI distribution.

- `_parse_args(argv) -> argparse.Namespace` — argparse setup. (L35)
- `main(argv=None) -> int` — Entry point; returns a process exit code. (L77)

### `tools/generate_rskill_skillmd.py`
_Generate the standard agent-skill `SKILL.md` discovery view for every in-tree rSkill from its `rskill.yaml`._
The single canonical producer of the `SKILL.md` mirror (CLAUDE.md §1.3): `rskill.yaml` is authoritative; the generated `SKILL.md` is discovery-only and never hand-edited. `--check` fails on any stale/missing `SKILL.md`, so the same process applies to every kind — including `playbook`, whose `_KIND_NOUN` entry renders identically to `vla`/`detector`/`vlm`/`reward`.

- `render_skill_md(manifest_path: Path) -> str` (L162) — Render the `SKILL.md` text (YAML frontmatter + capability/verb summary + license/provenance) from one manifest; `_KIND_NOUN` maps each `kind` to its discovery noun.
- `main(argv=None) -> int` (L261) — Entry point. No args = regenerate every `rskills/<id>/SKILL.md`; positional ids regenerate a subset; `--check` reports stale/missing without writing (exit 1 on drift).

### `tools/rldx_sidecar.py`
_Boot helper for the RLDX-1 inference sidecar (companion to `openral_sim.policies.rldx`)._
Materialises a Python 3.10 venv under `_DEFAULT_HOME` (`~/.cache/openral/rldx-sidecar`, override via `--home`), clones the upstream `RLWRLD/RLDX-1` repo, runs `uv sync` (rldx + transformers + flash-attn + …), optionally adds `bitsandbytes` for NF4, then writes a wrapper that monkey-patches `transformers.AutoModel.from_pretrained` to apply NF4 / int8 to the Qwen3-VL-8B backbone (the MSAT diffusion head is left at bf16) and `os.execvpe`s into `rldx.eval.run_rldx_server`. Required because the `rldx` package pins `requires-python = "~=3.10"` and ships a custom `architectures=["RLDX"]` class not in HF Transformers.

- `_install_deps(*, source, uv, quantization) -> Path` — `uv sync` in the cloned source tree (creates `source/.venv` with rldx + deps), then `uv pip install --python <venv>/bin/python bitsandbytes>=0.43.0` when `quantization in {nf4, int8}`. Returns the venv path. (L48)
- `_make_wrapper(*, work, source, args) -> Path` — Generate `<work>/boot_server.py`: monkey-patches `AutoModel.from_pretrained` for the Qwen3-VL backbone with NF4 / int8 / no-op, sets `sys.argv`, and calls into `rldx.eval.run_rldx_server`. (L82)
- `main() -> int` — argparse entry point; flags `--model`, `--port`, `--quantization {none,nf4,int8}`, `--home`. Calls `run_sidecar(..., family="rldx", ...)`, which stamps the sidecar identity record (so the adapter can verify reuse) and then `os.execvpe`s into the sidecar venv so SIGINT reaches the server. (L237)

### `tools/xr1_sidecar.py` + `tools/_xr1_server.py`
_Boot helper + inference server for Xiaomi Robotics XR-1. The launcher provisions torch 2.9.1 / transformers 4.57.1 / FlashAttention 2.8.3 + bitsandbytes and execs the server (torch is one minor ahead of upstream's 2.8.0, which has no linux-aarch64 `cu128` wheel — `docs/reference/aarch64-support.md`). The server loads the pinned MiBoT custom-code checkpoint in manifest-selected NF4 (or BF16), recreates Xiaomi's benchmark-specific chat template and padded state tensor, decodes actions with the checkpoint processor, and serves `ping/reset/get_action/close` over the shared ZMQ/msgpack ndarray wire._
- `main() -> int` (launcher) — parse model/profile/quantization/host/port/home; `--export-dir` persists an NF4 checkpoint and exits, otherwise stamp sidecar identity and `os.execvpe` into `_xr1_server.py`.
- `_messages(profile, images, instruction) -> list[dict[str, Any]]` — exact RoboCasa, RoboCasa365-video, or VLABench message layout from upstream revision `7c20088`.
- `_pad_state(state) -> NDArray[np.float32]` — pad one frame or a four-frame history to XR-1's 60-D internal state.
- `class _XR1Policy` — pinned `AutoModel`/`AutoProcessor` custom-code loader; bitsandbytes NF4 uses a BF16 compute dtype + CUDA device map, `prequantized_nf4` reloads a saved packed checkpoint without re-quantizing, and checkpoint-owned action de-normalization remains unchanged. `export_pretrained(output_dir)` writes sharded packed safetensors, custom MiBoT code/processor assets, and `quantization_metadata.json` while explicitly excluding cached BF16 shards.

### `tools/qwen_vlm_sidecar.py` + `tools/_qwen_vlm_server.py`
_Boot helper + server for the Qwen3.5-4B scene-VLM sidecar, companion to `openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`._ The launcher provisions an isolated venv (`OPENRAL_QWEN_VLM_SIDECAR_VENV` to reuse one) with transformers + bitsandbytes + `qwen-vl-utils` + pyzmq/msgpack, then `os.execvpe`s into the server. The server answers a ZMQ REQ/REP + msgpack protocol (`{"op":"query","image","question"}` → `{"ok","answer"}`); out-of-process for dependency/VRAM isolation (same pattern as `rldx_sidecar`). Apache-2.0 model.

- `ensure_venv(home, *, override=None) -> Path` (sidecar) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded); honours `$OPENRAL_QWEN_VLM_SIDECAR_VENV`.
- `main() -> int` (sidecar) — argparse (`--model`, `--host`, `--port`, `--max-side`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME` and `os.execvpe`s into `_qwen_vlm_server.py`.
- `_load(model_id) -> (processor, model)` / `_query(...) -> str` / `main() -> int` (server) — dual-path NF4 load (auto-detect a pre-quantized checkpoint via the embedded `quantization_config` → load 4-bit directly; else quantize-at-load, serial materialization for 8 GB); one scene-question→answer generate via the canonical Qwen-VL recipe (strips the `<think>` trace); ZMQ REP loop (`ping`/`query`/`shutdown`). Validated live (CLAUDE.md §1.2).

### `tools/cosmos3_reasoner_sidecar.py`
_Boot helper for the curated NVIDIA Cosmos 3 Edge reasoner model (`OPENRAL_REASONER_MODEL=cosmos3-edge`), companion to `openral_reasoner.cosmos3.Cosmos3ToolUseClient`._ Provisions an isolated Python 3.12 venv (`OPENRAL_COSMOS3_SIDECAR_VENV` to reuse one) from the pure-PyPI hash-pinned `tools/sidecar_requirements/cosmos3_reasoner.lock` (vllm≥0.23) **plus a SHA-pinned `transformers` overlay** (`_TRANSFORMERS_EDGE_SHA` — no *released* transformers recognises `cosmos3_edge` yet), then — for the Edge diffusers layout — downloads the checkpoint and builds a **flattened reasoner view** (vLLM's loader can't follow the diffusers subfolder `weight_map`) before `os.execvpe`ing into `vllm serve <view> --served-model-name nvidia/Cosmos3-Edge --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager`. No ZMQ server script: vLLM's OpenAI-compatible HTTP API *is* the wire contract the reasoner already speaks. Out-of-process for the same dependency/VRAM isolation reasons as `qwen_vlm_sidecar.py`. Cosmos 3 weights are OpenMDW-1.1 (commercial OK) — no license guard. **Boots live on an 8 GB RTX 4070; forward-pass inference is blocked by an upstream `get_rope_index` bug** (see `docs/reference/cosmos3-edge-reasoner.md`).

- `ensure_venv(home, *, override=None) -> Path` — return the sidecar venv python, provisioning + installing the pinned lock **and the SHA-pinned transformers overlay** if absent (sentinel-guarded); honours `$OPENRAL_COSMOS3_SIDECAR_VENV`.
- `is_diffusers_reasoner_layout(model_dir) -> bool` — True when the top-level `model.safetensors.index.json` maps tensors into subfolders (the Edge layout needing a view); Nano/Super (bare top-level shards) return False.
- `materialize_reasoner_view(model_dir, dest) -> Path` — build a vLLM-loadable flat view: bare-named shard symlinks + a rewritten weight index + tokenizer/config symlinks. Idempotent.
- `resolve_served_model(model, home) -> tuple[str, str | None]` — resolve a repo id / local dir to a `vllm serve` target + optional `--served-model-name` (snapshot-downloads and builds the view for the Edge layout; serves Nano/Super by id).
- `build_serve_argv(*, vllm_bin, model, host, port, tool_call_parser, max_model_len, gpu_memory_utilization, enforce_eager, served_model_name=None, kv_cache_dtype="auto") -> list[str]` — the `vllm serve` argv. `--max-model-len` (default 8192) holds the reasoner prompt + tools; `--enforce-eager` + `--gpu-memory-utilization` (default 0.90, `$OPENRAL_COSMOS3_GPU_MEM_UTIL`) + `--kv-cache-dtype fp8` are the 8 GB-fit knobs (fp8 KV verified required for the 8192 window with the native Edge impl on an 8 GB 4070).
- `main() -> int` — argparse (`--model` default `nvidia/Cosmos3-Edge`, `--host`, `--port` default 8901, `--tool-call-parser`, `--max-model-len`, `--gpu-memory-utilization`, `--no-enforce-eager`, `--kv-cache-dtype`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME`, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation OOM fix verified live), resolves the served view, and `os.execvpe`s into `vllm serve`.

### `tools/behavior_groot_sidecar.py`
_Python 3.10 sidecar for the official 2026 BEHAVIOR-1K GR00T N1.7 checkpoint. Imports the pinned `wensi-ai/Isaac-GR00T` behavior runtime, registers the official R1Pro modality slices, wraps `Gr00tPolicy` with `B1KPolicyWrapper`, and serves ZMQ `ping/reset/get_action/close`. Two 8 GB-host memory measures: whole-model NF4 (`--quantization nf4`, default) and dropping the unused Qwen3-VL `lm_head` (the wrapper reads only hidden states; ~840 MiB saved — 2.77 GiB inference peak, live-validated alongside OmniGibson)._

- `main(argv) -> int` — Parse checkpoint/task/instruction/control-mode/device/endpoint plus `--quantization {none,nf4,int8}` / `--nf4-min-params` (the min-params threshold applies to both quantizers), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, load the real checkpoint (CPU-first under NF4/int8), and serve the 23-D action API. `OPENRAL_BEHAVIOR_GROOT_DUMP_OBS=<dir>` pickles the first 32 observations for offline quantization A/Bs.
- `_drop_backbone_lm_head(model) -> None` — Replace the Qwen3-VL `lm_head` with `Identity`; the B1K wrapper consumes only `hidden_states[-1]`, so the full-vocab logits projection is dead weight and was the largest single inference allocation.
- `_quantize_nf4(model, *, device, min_params) -> None` — Whole-model `bitsandbytes` `Linear4bit` rewrite of large `nn.Linear`s (manifest pins `nf4_min_params: 1000000`), keeping small layers floating; patches the model `dtype` property to the first floating parameter dtype, then moves to CUDA. Offline A/B/C on captured obs: 2531 MiB peak, action MAE 0.0029 vs bf16 — behaviourally lossless for this checkpoint.
- `_quantize_int8(model, *, device, min_params) -> None` — Same rewrite with `bitsandbytes` `Linear8bitLt` (LLM.int8, threshold 6.0). 3678 MiB peak / MAE 0.0017 vs bf16 on the same captured obs — too big next to OmniGibson on 8 GB, so its role is offline quality comparison, not co-resident serving.

### `tools/behavior_scene_sidecar.py`
_Official OmniGibson evaluator environment sidecar for `scene.id=behavior`. Resolves the configured task instance, exposes the challenge wrapper observation, applies external 23-D actions, and returns success/metrics/sim time over ZMQ._

- `main(argv) -> int` — Parse task/instance/mode/wrapper/max-steps/endpoint, load the real BEHAVIOR environment, and serve `ping/reset/step/close`.

### `tools/_robometer_scorer.py`
_In-process stateless scorer for the Robometer-4B reward monitor, companion to `openral_runner.backends.reward.robometer_reward.RobometerInProcessReward`._ `reward_monitor_node` imports `_robometer_scorer.py::_Scorer` directly; there is no separate Robometer ZMQ process or dedicated venv. As of lerobot 0.6.0 the reward model is lerobot's in-tree `lerobot.rewards.robometer.RobometerRewardModel` — a vanilla `AutoModelForImageTextToText` (Qwen3-VL-4B) loaded with plain `transformers`. There is **no** pinned `robometer` git package and **no** `transformers==4.57.1` force-pin. The scorer keeps OpenRAL's NF4 pre-quantized checkpoint (`OpenRAL/rskill-robometer_4b-any-general-nf4`, ~3.3 GB resident), meta-builds the native `RobometerRewardModel` skeleton and drops the packed 4-bit weights (remapped into the native module) in directly — no bf16 spike, no Qwen weight download. Validated live: 3.33 GB NF4, progress ramps to 0.88 + success 0.90 at task completion.

- `class _Scorer` (scorer) — meta-builds the native `RobometerRewardModel`, remaps + loads the NF4 prequant pack, then `score(frames_rgb, task, num_bins) -> (progress, success)` computes per-frame progress via the module-level `decode_progress_outputs` on `_compute_rbm_logits` (not `compute_reward`, which returns only a scalar). `_load_prequantized`, `_native_config`, `_remap_backbone_key`, `_resolve_local_dir` support the meta-load.

### `tools/build_qwen_vlm_nf4_checkpoint.py`
_Reproducible recipe for the published `OpenRAL/rskill-qwen35_4b-any-general-nf4` pre-quantized NF4 checkpoint. Runs in the sidecar venv._ `main() -> int` — argparse (`--source`, `--out`); loads the upstream model once (NF4 + serial materialization so the bf16 pass fits 8 GB), `save_pretrained`s the 4-bit weights + processor, then verifies the checkpoint reloads directly as 4-bit (no bf16 spike) and answers a smoke query. Pre-quantizing lets deployment load the 4-bit weights directly (~3.3 GB) with no loader workaround. Distinct from `quantize_rskill.py`, which writes an `install_prequantized_linears`-loaded pack for the in-process lerobot runtime; this writes a transformers-native `save_pretrained` checkpoint for the isolated VLM sidecar.

### `tools/fix_libero_config.py`
_Auto-fix for the stale `~/.libero/config.yaml` pitfall._

Detects and repairs `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero/config.yaml`) when its absolute paths no longer match the currently-active `libero` package — the file is written once at first LIBERO import and never refreshed, so switching venv / clone / workspace path leaves it pointing at a directory that no longer exists. Wired into the `Justfile` as `_ensure-libero-config` (chained off `sim-libero` / `sim-xvla-libero` / `sim-pi05-libero`). Idempotent.

- `_expected_config(libero_pkg_dir) -> dict[str, str]` — Compute the canonical `assets/bddl_files/benchmark_root/datasets/init_states` payload that LIBERO writes on first import. (L47)
- `_parse_yaml_map(text) -> dict[str, str]` — Parse the flat `key: value` map LIBERO writes — no PyYAML dependency. (L65)
- `_render_yaml(payload) -> str` — Render the same flat layout. (L79)
- `_locate_active_libero() -> Path` — `import libero` and return its package directory; raises `RuntimeError` with a clear message when LIBERO is absent (caller treats as no-op). (L84)
- `main() -> int` — argparse entry point; flags `--dry-run`, `--verbose`. Returns 0 when the config matches or after rewriting. (L110)
