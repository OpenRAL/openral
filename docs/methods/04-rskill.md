# Layer 4 — rSkill (S1)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/rskill/src/openral_rskill/base.py`
_rSkillBase — abstract base class with lifecycle state machine._

- const `_TRANSITIONS: dict[RSkillState, frozenset[RSkillState]] = {...}` — Legal lifecycle edges keyed by current state; `_require_transition` checks membership. (L59)
- `class rSkillBase(abc.ABC)` — Abstract base class for all OpenRAL skills (rSkill is the official package-format name). (L68)
  - `__init__(name, *, version='0.1.0', role='s1', embodiment_tags=None, latency_budget_ms=None)` — Init only; does not configure or load weights. (L98)
  - `info -> RSkillInfo` [@property] (L122)
  - `name -> str` [@property] (L130)
  - `state -> RSkillState` [@property] (L135)
  - `configure() -> None` — `unconfigured → inactive`. (L142)
  - `activate() -> None` — `inactive → active`. (L167)
  - `deactivate() -> None` — `active → inactive`. (L189)
  - `shutdown() -> None` — Any state → `finalized`. (L209)
  - `step(world_state) -> Action` — One inference step (hot path). (L233)
  - `on_load_weights() -> None` — Hook: load weights. (L279)
  - `on_unload_weights() -> None` — Hook: release weights, called by `shutdown()` (VRAM eviction). (L286)
  - `on_quantize() -> None` — Hook: apply quantization. (L296)
  - `on_warmup() -> None` — Hook: dummy forward pass, called by `activate()` before `_activate_impl`; default no-op. The deploy path implements it in the runner's policy shim (`rskill_runner_node.py` `on_warmup`), which runs `adapter.step` on a synthetic observation and swallows+logs failures as `rskill_runner.warmup_failed`. (L303)
  - `_configure_impl/_activate_impl/_deactivate_impl/_shutdown_impl/_step_impl()` [@abstractmethod] (L313)
  - private: `_transition`, `_update`, `_require_transition`, `_enter_error`

### `python/rskill/src/openral_rskill/runtime.py`
_Runtime Protocol and NullRuntime — inference backend contract._

- `class Runtime(Protocol)` — Structural protocol for skill inference backends. (L24)
  - `is_loaded -> bool` [@property] (L40)
  - `device -> str` [@property] (L45)
  - `load(path) -> None` (L49)
  - `infer(inputs) -> dict[str, Any]` (L61)
  - `quantize(config: QuantizationConfig) -> None` (L76)
  - `warmup(inputs) -> None` (L88)
  - `unload() -> None` (L96)
- `class NullRuntime` — No-op backend for testing; same surface as `Runtime`, every method a no-op. (L101)
  - `__init__(device='cpu')` (L123)
  - `is_loaded -> bool` [@property] (L133)
  - `device -> str` [@property] (L138)
  - `load(path) -> None` (L142)
  - `infer(inputs) -> dict[str, Any]` (L150)
  - `quantize(config) -> None` (L161)
  - `warmup(inputs) -> None` (L168)
  - `unload() -> None` (L175)

### `python/rskill/src/openral_rskill/runtime_pytorch.py`
- const `_ALLOW_UNSAFE_PICKLE_ENV = "OPENRAL_ALLOW_UNSAFE_PICKLE"` — Env var gating `load()`'s pickle deserialization (C2). (L38)
- `class PyTorchRuntime` — `torch`-backed `Runtime`. (L41)
  - `__init__(device='cpu')` (L70)
  - `is_loaded -> bool` [@property] (L81)
  - `device -> str` [@property] (L86)
  - `load(path) -> None` — Unpickles a full module — gated behind `_ALLOW_UNSAFE_PICKLE_ENV`, C2. (L90)
  - `load_safetensors(path, *, model, strict=True) -> None` — Safe: loads a `state_dict` into a caller-supplied module, no code execution — preferred for new skills. (L146)
  - `infer(inputs) -> dict[str, Any]` (L202)
  - `quantize(config) -> None` — Dynamic INT8 on Linear. (L229)
  - `warmup(inputs) -> None` (L261)
  - `unload() -> None` — Frees CUDA cache. (L273)

### `python/rskill/src/openral_rskill/runtime_onnx.py`
- const `_CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]` (L30)
- const `_CPU_PROVIDERS = ["CPUExecutionProvider"]` (L31)
- `class ONNXRuntime` — `onnxruntime`-backed `Runtime`. (L52)
  - `__init__(device='cpu')` (L80)
  - `is_loaded -> bool` [@property] (L100)
  - `device -> str` [@property] (L105)
  - `load(path) -> None` (L109)
  - `infer(inputs) -> dict[str, Any]` (L132)
  - `quantize(config) -> None` — Always raises — ONNX quantization is pre-applied. (L157)
  - `warmup(inputs) -> None` (L173)
  - `unload() -> None` (L186)

### `python/rskill/src/openral_rskill/backend_registry.py`
_Extraction seam: entry-point-based runtime-backend + policy-attach-hook registry._

- const `_RUNTIME_BACKENDS_GROUP = "openral.runtime_backends"` (L41)
- const `_POLICY_ATTACH_HOOKS_GROUP = "openral.policy_attach_hooks"` (L42)
- const `_BUILTIN_RUNTIME_BACKENDS: dict[str, str] = {...}` — Built-in `kind` → runtime-class-name map (`pytorch`/`onnx`/`null`). (L48)
- `resolve_runtime_backend(kind: str) -> type[Runtime]` — built-in dict (`pytorch`→`PyTorchRuntime`, `onnx`→`ONNXRuntime`, `null`→`NullRuntime`) first, else looked up via the `openral.runtime_backends` entry-point group (e.g. `openral-pro-trt` registering `tensorrt`). Miss → `ROSConfigError` naming `openral-pro-trt`. (L55)
- `maybe_attach_pro_hooks(policy_name: str, skill, **kwargs) -> bool` — Generic policy-attach-hook lookup via the `openral.policy_attach_hooks` entry-point group (name = `policy_name`). No hook installed → debug log + `False`, never a silent skip; hook found → invoked as `hook(skill, **kwargs)`. (L106)

> **Moved to OpenRAL Pro:** `runtime_tensorrt.py` (`TensorRTRuntime`), `smolvla_export.py`, `smolvla_trt.py`, and `act_trt.py` now live in the private `openral-pro-trt` package as `openral_pro_trt.*`, plugging in via the `openral.runtime_backends` / `openral.policy_attach_hooks` entry-point groups above.

### `python/rskill/src/openral_rskill/loader.py`
_rSkill loader — HF Hub download, manifest validation, license guard, local registry._

- const `_DATA_HOME = Path(os.environ.get("OPENRAL_DATA_HOME", ...))` — Base dir for the local rSkill registry. (L64)
- const `_CACHE_HOME = Path(...)` — Base dir HF Hub downloads land under. (L65)
- const `DEFAULT_REGISTRY_PATH: Path = _DATA_HOME / "rskills.json"` — Default JSON registry file written by `rSkill.from_pretrained`. (L69)
- const `_RSKILL_MANIFEST_CACHE: dict[str, RSkillManifest] = {}` — In-process memoisation for `load_rskill_manifest`. (L74)
- const `_ALLOW_NONCOMMERCIAL_ENV = "OPENRAL_ALLOW_NONCOMMERCIAL"` — Env var gating noncommercial-weight loads (CLAUDE.md §1.9). (L78)
- const `_REQUIRE_SIGNED_ENV = "OPENRAL_REQUIRE_SIGNED_SKILLS"` — Env var that fails closed when provenance signing is unverified. (L83)
- const `_EMBODIMENT_ANY_TAG: str = "any"` — Sentinel embodiment tag matching every robot. (L91)
- `class InstalledRSkillEntry(BaseModel)` — One row in the local registry. (L97)
  fields: `repo_id, version, revision, local_dir, manifest_path, license, role, embodiment_tags, installed_at`
- `class rSkill` — Packaged, capability-tagged robot skill; provenance signing is not yet enforced. (L141)
  - `__init__(manifest, local_dir)` (L165)
  - `from_pretrained(cls, repo_id, *, revision=None, cache_dir=None, force_download=False, commercial_use=True, registry_path=None) -> rSkill` [@classmethod] — Download from HF Hub, validate, register. (L178)
  - `from_yaml(cls, path, *, local_dir=None) -> rSkill` [@classmethod] — Load locally without network. (L297)
  - `list_installed(registry_path=None) -> list[InstalledRSkillEntry]` [@staticmethod] (L332)
  - `uninstall(repo_id, registry_path=None) -> bool` [@staticmethod] — Remove from registry only. (L363)
  - `check_embodiment_tags(manifest, robot_capabilities) -> None` [@staticmethod] — Verify embodiment tag intersection (raises on disjoint sets). Exempt for perception kinds (`detector`/`segmenter`/`vlm`, `_EMBODIMENT_AGNOSTIC_KINDS`): they are camera-in → detections/text-out with no action contract, so they match any robot regardless of tags. (L393)
  - `check_capability_flags(manifest, robot_capabilities) -> None` [@staticmethod] — Verify every `manifest.capabilities_required` flag against `RobotCapabilities`. (L421)
  - `check_runtime(manifest, robot_capabilities) -> None` [@staticmethod] — Verify `manifest.runtime` ∈ `gpu_supported_runtimes`; skipped when the legacy capability field is missing or empty (the GPU support fields moved to `ComputeSpec`, so missing means "unknown" for now). (L453)
  - `check_quantization_dtype(manifest, robot_capabilities) -> None` [@staticmethod] — Verify `manifest.quantization.dtype` ∈ `gpu_supported_dtypes`; skipped when the legacy capability field is missing or empty. (L482)
  - `check_capabilities(manifest, robot_capabilities) -> None` [@staticmethod] — Composition of the four narrower checks; raises on first failure. (L510)
  - `check_sensors(manifest, robot_sensors) -> None` [@staticmethod] — Verify the robot exposes every `SensorRequirement` the manifest declares: a `vla_feature_key` requirement resolves to exactly one matching robot sensor, otherwise the robot must expose `count` sensors of the requested modality meeting any resolution minimum. `ROSCapabilityMismatch` on any unmet requirement. (L548)
  - `check_compatibility(manifest, robot) -> None` [@staticmethod] — Umbrella entry point combining `check_capabilities` (embodiment tags + capability flags, with `compute=robot.compute_edge or robot.compute_local`) and `check_sensors` against `robot.sensors`. (L589)
  - `_check_license(manifest, *, commercial_use) -> None` [@staticmethod] — Enforce license guards for the manifest's declared posture. (L708)
  - `_validate_eval_jsons(skill_dir) -> None` [@staticmethod] — Validate every `<skill_dir>/eval/*.json` against `RSkillEvalResult`. (L808)
  - `_register(entry, registry_path) -> None` [@staticmethod] (L834)
  - `__repr__() -> str` (L858)
- `resolve_rskill_local_dir(uri) -> Path | None` — Return the absolute on-disk directory of an in-tree rSkill from a bare skill ref (name, `rskills/<name>`, or Hub repo id), or `None` for Hub-only refs. Used by `openral benchmark run` to write eval results and update `rskill.yaml` regardless of cwd. (L870)
- `_candidate_local_paths(uri) -> list[Path]` — Enumerate on-disk candidates (cwd-relative + repo-root anchored) for a skill reference. Also unwraps HF Hub form `<org>/rskill-<name>` to in-tree `rskills/<name>`. (L891)
- `discover_intree_rskills() -> list[tuple[str, RSkillManifest]]` — Walk `<repo>/rskills/*/rskill.yaml` and return `(name, manifest)` pairs. Malformed entries are skipped with a stderr warning. (L926)
- `find_repo_root_from(start) -> Path | None` — Walk up from `start` for the first ancestor containing both `pyproject.toml` and `rskills/`. Public, re-exported from `openral_rskill.__init__`; used across the package boundary by `openral_sim.cli`, `openral_cli.main`, and the deploy_e2e launch file. (L957)
- `SKILL_ONLY_EMBODIMENT_TAGS: frozenset[str]` — `{"any", "custom", "multi"}`: embodiment tags valid on an rSkill manifest that no robot declares. (L970)
- `intree_embodiment_tags() -> frozenset[str]` — `SKILL_ONLY_EMBODIMENT_TAGS` plus every `capabilities.embodiment_tags` of the checkout's `robots/*/robot.yaml`; the typo guard for the open `EmbodimentTag` id (scaffolders + CI). Public, re-exported. (L975)
- `known_benchmark_ids() -> frozenset[str] | None` — `benchmarks/*.yaml` ∪ `scenes/benchmark/*.yaml` stems of the checkout, or `None` without one; the membership set for the open `BenchmarkName` id (writeback, `benchmark scene` guard, CI). Public, re-exported. (L992)
- `_warn_unknown_benchmarks(manifest, *, source) -> None` — `rskill.unknown_benchmarks` warning (never a failure) when a Hub manifest cites a benchmark the checkout lacks; called by `rSkill.from_pretrained`. (L1008)
- `validate_skill_ref(raw) -> str` — Validate and return a bare rSkill reference unchanged (bare name, `rskills/<name>` path, or HF repo id); rejects a known URI scheme (`hf://`, `local://`, `file://`, `http(s)://`). Public, re-exported from `openral_rskill.__init__`. (L1020)
- `load_rskill_manifest(uri) -> RSkillManifest` — Resolve a bare skill reference to a parsed manifest. Tries local path → in-tree mapping → HF Hub download. In-process memoised. (L1060)
- `resolve_rskill_to_hf(uri) -> str` — Resolve a skill reference to either the underlying HF Hub repo id (`hf://...`) or an absolute local path (`local://...`); both forms are accepted by `from_pretrained` helpers. (L1132)
- `resolve_rskill_to_hf_with_revision(uri) -> tuple[str, str | None]` — Like `resolve_rskill_to_hf` but splits the optional `@<branch-or-sha>` pin off an `hf://` `weights_uri` into a separate `revision`, since HF drops a pin glued onto the repo id. A directory that already holds `model.safetensors` next to its `rskill.yaml` — the snapshot `rSkill.from_pretrained` leaves in `~/.cache/openral/rskills` — resolves to **itself** rather than following its `hf://` pointer, which otherwise re-resolved a just-downloaded repo through the *default* HF cache (second fetch online, hard failure offline). In-tree `rskills/<name>/` dirs ship no weights and are unaffected. (L1166)

### `python/rskill/src/openral_rskill/hub_search.py`
_Free-text + facet search over an HF Hub org's rSkills — backs `openral rskill search`._

- `class HubRSkillHit(BaseModel)` [frozen] — One matching Hub repo. (L43)
  fields: `repo_id, manifest: RSkillManifest, local: Literal["in-tree","installed"] | None, hub_tags: tuple[str, ...]`. `local == "in-tree"` compares the Hub `repo_id` against the `name` declared by a real `rskills/<dir>/rskill.yaml` — never against the fetched (possibly stale) remote manifest's own `name` field.
- `class HubRSkillSearchResult(BaseModel)` — Search result: `hits` (sorted by `repo_id`), `skipped_repo_ids` — the explicit record of dropped repos, since this module never logs a skip itself — and `inspected`. (L82)
- `search_hub_rskills(query='', *, kind='', role='', embodiment='', license='', family='', org='OpenRAL', limit=None, max_workers=8) -> HubRSkillSearchResult` — Lists every repo the org tagged `rskill` in one Hub call, fetches each candidate's `rskill.yaml` concurrently, then matches `query` locally (case-insensitive, every token substring-matched against id/name/description/tags) and applies the facet filters. `limit` caps hits after sorting by `repo_id`. (L207)
- `_manifest_haystack(repo_id, manifest, hub_tags) -> str` — Build the lower-cased blob `_matches_query` searches. (L112)
- `_matches_query(query, haystack) -> bool` — Every whitespace-split token must substring-match. (L127)
- `_matches_facets(m, *, kind, role, embodiment, license_, family) -> bool` — Whether a manifest passes every non-empty facet filter. (L133)
- `_fetch_one(model) -> tuple[str, RSkillManifest | None, tuple[str, ...], str]` — Worker-thread fetch + validate of one repo's `rskill.yaml`; narrow-catches expected errors so one bad repo never fails the whole search. The skip reason isn't logged (would corrupt `--json | jq` output) — only the repo id surfaces, via `HubRSkillSearchResult.skipped_repo_ids`. (L148)
- `_local_markers() -> tuple[frozenset[str], frozenset[str]]` — `(in_tree_names, installed_repo_ids)` for the `local` marker. A corrupt registry or an invalid row is treated as no installed entries, with `log.warning("rskill.hub_search.registry_unreadable", ...)` — never silently swallowed. (L182)

### `python/rskill/src/openral_rskill/gpu_passthrough.py`
_GpuPassthroughSkill — minimal rSkill whose per-step image processing provably runs on GPU (M8 PR I/10)._

- `_REDUCTION_SIZE: int = 64` — module constant; reduction-target size used to bound GPU latency. (L48)
- `_RGB_CHANNELS: int = 3` — module constant; channel count for the GPU mean-reduction read-back. (L52)
- `class GpuPassthroughSkill(rSkillBase)` — Uploads each `SensorFrame` to torch.cuda, runs per-channel mean reduction (with explicit `torch.cuda.synchronize`), emits result as `Action.confidence`. Refuses silent CPU fallback. (L55)
  - `__init__(sensor_id='wrist_rgb', n_joints=6, horizon=1, device='cuda', latency_budget_ms=None)` (L76)
  - `step_count -> int` [@property] (L104)
  - `on_load_weights() -> None` — no-op (skill is weight-less). (L110)
  - `on_quantize() -> None` — no-op (skill is weight-less). (L114)
  - `on_warmup() -> None` — Allocate the GPU input buffer + launch a kernel so the first step doesn't pay cudaMalloc latency. (L118)
  - `_configure_impl()` — Lazy-import torch, resolve device, raise if `cuda` requested and `torch.cuda.is_available()` is False. (L144)
  - `_activate_impl/_deactivate_impl/_shutdown_impl` (L169)
  - `_step_impl(world_state) -> Action` — Pull frame → CPU→GPU upload → GPU reduction → action with confidence. (L185)
  - private: `_extract_latest_image(world_state) -> NDArray[np.uint8]`, `_gpu_reduce(frame, *, torch) -> (float, float, float)`. (L227 / L260)
- `_channels(encoding: FrameEncoding) -> int` — Per-pixel channel count for a `FrameEncoding`; MONO8 → 1, BGR8/RGB8 → 3, everything else falls through to a defensive BGR default of 3. (L297)
- `_zero_frame() -> NDArray[np.uint8]` — Resilient placeholder when no sensor frame is available yet. (L310)

### `python/rskill/src/openral_rskill/_diagnostics.py`
_Shared load-phase instrumentation seam — generalises the inline `_heartbeat` originally inside `openral_sim.policies.pi05` so every VLA adapter's `_build_*` factory uses the same `<prefix>_<name>_{start,heartbeat,done}` event shape (CLAUDE.md §1.13 — single seam, no duplicates)._

- const `_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096` — Page size in bytes, for converting `/proc/self/statm`'s page count into `rss_mb`. (L46)
- `phase_timer(name, *, prefix="phase", interval_s=15.0, log=None, gpu_mb=False, **fields) -> Iterator[None]` [@contextmanager] — Emits `<prefix>_<name>_start/heartbeat/done` events with `elapsed_s`, `rss_mb`, and `major_faults` so a slow-but-idle load is attributable to page reclaim. `gpu_mb=True` also attaches CUDA memory to the heartbeat; torch import is lazy so CPU-only hosts still work. (L136)
- `gpu_allocated_mb(*, no_import=False) -> float | None` — Current CUDA allocator usage in MB (`memory_allocated`, not `memory_reserved`), or `None` if unavailable — the single GPU-memory probe eviction accounting and phase-timer heartbeats both read. `no_import=True` consults only an already-imported torch. Public, re-exported from `openral_rskill.__init__`. (L49)
- `_rss_majflt() -> tuple[float, int] | None` — Process RSS (MB) + lifetime major-fault count from `/proc/self/{statm,stat}`; `None` off Linux. (L82)
- const `_PAGE_SIZE: int = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096` — Page size in bytes used to convert `/proc/self/statm`'s page count into `rss_mb`. (L46)

### `python/rskill/src/openral_rskill/executor.py`
_Action-chunk executor shared by every chunked VLA family. Also the home of Real-Time Chunking (RTC): with an enabled lerobot `RTCConfig` the buffer becomes an `ActionQueue` whose tail a landing prefetch replaces._

- const `_CHUNK_TENSOR_RANK = 3` — Expected `(batch, chunk, action_dim)` tensor rank for a chunk producer's output. (L80)
- `class ChunkedExecutor` — Owns the per-step action buffer for every chunked VLA family, optionally overlapping chunk N+1 inference with execution via a background thread; calls `predict_action_chunk` directly rather than resetting lerobot's shared queue. Default producer is `policy.predict_action_chunk`, or a custom `chunk_fn`; `prefetch_at=0` is synchronous. `postprocess_action` finishes each chunk where it is produced: one host copy of the whole chunk, then the per-action postprocessor on every row, in the producing (prefetch) thread, so pops never touch the GPU and lerobot's `AbsoluteActionsProcessorStep` adds the state cached for that chunk's own inference; under RTC it runs at merge time into the `ActionQueue`'s processed copy, which pops serve as NumPy, while the raw rows stay for the guidance blend. **RTC mode** (an enabled lerobot `RTCConfig`) swaps the buffer for an `ActionQueue` and replaces its tail on each prefetch landing instead of appending, requiring `prefetch_at >= 1` for real overlap. (L83)
  - `__init__(policy=None, *, chunk_fn=None, chunk_size=None, prefetch_at=20, rtc_config=None)` — Takes policy OR chunk_fn+chunk_size (`ValueError` otherwise). Negative `prefetch_at` raises `ROSConfigError`; it's clamped to `chunk_size - 1` at construction. An enabled `rtc_config` requires a clamped `prefetch_at >= 1` (`ROSConfigError` otherwise); a lead shorter than `execution_horizon` warns rather than errors. (L86)
  - `start() -> None` — Mark as running (call after policy is on-device). (L217)
  - `stop() -> None` — Signal background thread, join. (L221)
  - `reset() -> None` — Clear buffer/bg state and the RTC queue + last delay; `policy.reset()` when a policy was given. (L228)
  - `select_action(batch_or_fn) -> Any` — Next action; cold-start foreground inference / buffer pop / wait-on-prefetch. Delegates to `_select_action_rtc` in RTC mode. (L245)
  - private: `_pop_and_maybe_prefetch(batch)` (every pop routes through it so the trigger is branch-independent), `_produce(payload, chunk_index, kind, rtc_kwargs=None)` (passes `synchronize=True` on the prefetch path only), `_materialize`, `_extend_buffer`, `_launch_prefetch(batch)`
  - private (RTC): `_select_action_rtc(batch)` — serve from the `ActionQueue`, re-checking after the prefetch wait so a landed chunk isn't discarded. `_rtc_merge(chunk, *, idx_before)` — replace the tail; `real_delay` is the index delta the queue advanced during inference, and the producer's batch dim must be 1. `_raise_bg_error_if_any()` — re-raise a latched background error on the foreground thread. The prefetch path truncates the leftover tail to `execution_horizon` rows so tail length doesn't depend on how early `prefetch_at` fires.

### `python/rskill/src/openral_rskill/ros_action_rskill.py`
_ROS-wrapping rSkill adapter — bridges arbitrary ROS 2 action / service servers (MoveIt, Nav2, …) into the `rSkillBase` lifecycle. Selected by `make_default_skill_resolver` when `manifest.kind in {"ros_action", "ros_service"}`._

- const `_RESULT_DEADLINE_MULTIPLIER = 5.0` (L77)
- const `_MIN_RESULT_DEADLINE_S = 2.0` (L82)
- const `_FUTURE_POLL_INTERVAL_S = 0.02` (L86)
- const `_WAIT_FOR_SERVER_TIMEOUT_S = 15.0` (L94)
- const `_GOAL_STATUS_LABELS: dict[int, str] = {...}` — Human-readable labels for `action_msgs/GoalStatus` codes, used in error/log messages. (L101)
- `build_joint_permutation_from_names(*, source_names, target_names) -> list[int]` — Build the permutation that reorders a wrapped server's `JointTrajectory.positions` into the host `RobotDescription.joints` order. Raises `ROSConfigError` on set-inequality so a joint mismatch surfaces loudly instead of silently swapping bytes. (L172)
- const `CUMOTION_PIPELINE_ID = "isaac_ros_cumotion"` — the cuMotion MoveIt planning-pipeline id. (L141)
- `maybe_inject_cumotion_pipeline(goal_dict, *, interface_type, capabilities) -> dict` — On a host clearing the cuMotion GPU floor, sets `request.pipeline_id = CUMOTION_PIPELINE_ID` on a `MoveGroup` goal so MoveIt plans with cuMotion; no-op for non-MoveGroup actions, low-VRAM hosts, an already-set `pipeline_id`, or a goal with no `request` block. Pure — never mutates the input. (L144)
- `class ROSActionRskill(rSkillBase)` — `rSkillBase` shim wrapping a ROS 2 ActionClient (or service client), in two modes selected by `manifest.ros_integration.result_trajectory_field`: trajectory mode replays one waypoint per `step()`, result-only mode awaits the wrapped result — both raise `ROSRskillGoalSatisfied` on completion. ROS imports are deferred to `_configure_impl` so the module imports cleanly without ROS sourced. (L299)
  - `__init__(*, manifest, ros_node, robot_description, prompt, prompt_metadata_json)` (L332)
  - `_configure_impl()` — Lazy-import IDL, build ActionClient/service client, parse `default_goal_json`. (L403)
  - `_activate_impl()` — no-op; the wrapped action dispatches on first `step()`. (L493)
  - `_deactivate_impl()` / `_shutdown_impl()` — Release the wrapped client. (L496)
  - `_step_impl(world_state) -> Action` — First call sends goal and caches result; subsequent calls dequeue waypoints. (L513)

### `python/rskill/src/openral_rskill/look_at_rskill.py`
_Camera-aiming MoveGroup skill. Selected by `make_default_skill_resolver` when `manifest.ros_integration.goal_builder == "look_at"` (new `RosIntegration.goal_builder` field; `RSkillAction` gains `LOOK = "look"`)._

- const `_DEFAULT_CAMERA = "wrist"` (L46)
- const `_DEFAULT_POSITION_TOLERANCE_M = 0.02` (L47)
- const `_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.15` (L48)
- const `_TF_LOOKUP_TIMEOUT_S = 5.0` (L49)
- const `_XYZ_LEN = 3` (L50)
- const `_MIN_DIRECTION_NORM = 1e-9` (L51)
- `resolve_camera_sensor(description, camera) -> SensorSpec` — Find the named camera in `RobotDescription.sensors`; `ROSConfigError` listing the available sensor names on a miss (explicit beats implicit — default camera is `"wrist"`). (L54)
- `build_look_at_constraints(*, camera_goal: Pose6D, link_name, link_t_cam=None, position_tolerance_m=0.02, orientation_tolerance_rad=0.15) -> dict` — Lower a camera gaze pose into one MoveGroup `goal_constraints` entry by delegating to `pose_goal_rskill.build_pose_constraints` with the optical (z) axis tolerance set to π (roll free). With `link_t_cam` the goal is re-expressed for the mount link; without it the camera frame is the constrained link. (L100)
- `class LookAtRskill(ROSActionRskill)` — Consumes the merged goal's `look_at` block (`target_xyz` required, plus `frame_id`/`camera`/`standoff_m`/tolerances) instead of raw constraints, lowering it lazily on the first `step()` via `compute_gaze_pose` → `build_pose_constraints`. The manifest ships `plan_only: true` so MoveIt-side execution never bypasses the safety kernel. (L132)

### `python/rskill/src/openral_rskill/pose_goal_rskill.py`
_Generic Cartesian end-effector pose MoveGroup skill. Selected by `make_default_skill_resolver` when `ros_integration.goal_builder == "pose"`. Home of the shared pose→constraints lowering `LookAtRskill` reuses._

- const `_SOLID_PRIMITIVE_SPHERE = 2` — `shape_msgs/SolidPrimitive.SPHERE`. (L40)
- const `_DEFAULT_POSITION_TOLERANCE_M = 0.01` (L41)
- const `_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.05` (L42)
- const `_XYZ_LEN = 3` (L43)
- const `_QUAT_LEN = 4` (L44)
- const `_QUATERNION_ORDERS = ("xyzw", "wxyz")` (L45)
- const `_TF_LOOKUP_TIMEOUT_S = 5.0` (L46)
- `build_pose_constraints(*, pose: Pose6D, link_name, link_t_target=None, position_tolerance_m=0.01, orientation_axis_tolerances_rad=(0.05, 0.05, 0.05)) -> dict` — Lower a target pose into one MoveGroup `goal_constraints` entry (sphere position region + per-axis orientation constraint); `link_t_target` re-expresses the goal for the constrained link. The one place pose→MoveGroup-constraint math lives — do not re-implement. (L49)
- `pose_from_block(block) -> tuple[Pose6D, str, float, float]` — Parse a `pose` goal block → `(pose, link_name, pos_tol, orient_tol)`. Orientation is a 4-float quaternion array; component order from `block["quaternion_order"]` (`"xyzw"` default / `"wxyz"`). `ROSConfigError` on missing/ill-typed fields or an unknown order. (L141)
- `class PoseGoalRskill(ROSActionRskill)` — Consumes the merged goal's `pose` block; lowers it via `build_pose_constraints` (full orientation) on the first `step()`, then dispatches/replays like the parent. `link_t_target` is identity in v1 (the RobotDescription tool-frame offset is a later phase). (L198)

### `python/rskill/src/openral_rskill/joint_goal_rskill.py`
_Joint-space MoveGroup skill. Selected when `ros_integration.goal_builder == "joint"`. The LLM-facing replacement for hand-written `joint_constraints` JSON._

- const `_DEFAULT_JOINT_TOLERANCE_RAD = 0.001` (L23)
- const `_JOINT_WEIGHT = 1.0` (L24)
- `joint_constraints_from_block(block) -> dict` — Lower a `joint` block (`joint_names`, `positions`, optional `position_tolerance_rad`) into one `goal_constraints` entry (`{"joint_constraints": [{joint_name, position, tolerance_above, tolerance_below, weight}, …]}`). `ROSConfigError` on missing/ill-typed fields or a name/position length mismatch. (L27)
- `class JointGoalRskill(ROSActionRskill)` — Consumes the merged goal's `joint` block; lowers it into a `joint_constraints` goal at `_configure_impl`, then dispatches/replays like the parent. (L69)
  - `_configure_impl() -> None` (L77)

### `python/rskill/src/openral_rskill/smolvla.py`
_SmolVLA adapter — rSkillBase implementation for the SmolVLA family of VLAs._

- `from openral_rskill.executor import ChunkedExecutor` — re-exported via `__all__` for back-compat (`from openral_rskill.smolvla import ChunkedExecutor` still works after the move). (L53)
- const `_SO100_JOINT_NAMES: tuple[str, ...] = (...)` — SO-100 6-DoF joint order used by `_so100_obs_fn`. (L63)
- `class SmolVLAAdapter(rSkillBase)` — Drives any SmolVLA-family policy. (L76)
  - `__init__(repo_id, obs_fn, prompt, *, device='cuda:0', n_dof=6, n_cameras=None, prefetch_at=20, name='smolvla', version='0.1.0', embodiment_tags=None, latency_budget_ms=None)` — `n_cameras` (default `len(config.image_features)`) truncates warmup to the cameras the deploy feeds and threads to the TRT export (phantom-camera fix). (L121)
  - `on_load_weights() -> None` — Fetch checkpoint from HF Hub. (L159)
  - `on_warmup() -> None` — Dummy inference. (L218)
  - `_configure_impl()` — Validate IO shapes match `n_dof`. (L251)
  - `_activate_impl()` — Reset policy, start `ChunkedExecutor`. (L267)
  - `_deactivate_impl()` — Stop pre-fetch, keep weights. (L275)
  - `_shutdown_impl()` — Stop threads, free GPU memory. (L281)
  - `_step_impl(world_state) -> Action` — One S1 step. (L298)
  - `_preprocess(raw) -> dict[str, Any]` — Lerobot preprocessor + tensor → device. (L334)
- `class SO100SmolVLASkill(SmolVLAAdapter)` — Pre-configured for the SO-100 6-DoF arm. (L390)
  - `__init__(prompt, *, repo_id='lerobot/smolvla_base', device='cuda:0', extra_images=None, **kwargs)` (L412)
- `_so100_obs_fn(world_state, *, device, extra_images=None, prompt) -> dict[str, Any]` — SO-100 WorldState → SmolVLA raw input. (L348)

### `python/rskill/src/openral_rskill/_vla_core.py`
_Shared helpers for VLA adapters (Layer 3); internal — no public re-export._

- `InferenceKind` — Re-export of `openral_observability.InferenceKind` (`Literal["foreground", "prefetch", "single"]`), the closed value set for the `inference.kind` label — a timing axis, deliberately without a `"chunk"` value since chunk shape rides `inference.chunk_size`/`chunk_index` instead. (L45)
- `resolve_device(spec: VLASpec) -> str` — `"auto"` → `"cuda:0"` / `"mps"` / `"cpu"` against real torch. (L48)
- `resolve_rskill_repo_id(weights_uri: str, *, adapter_name: str) -> str` — Validate skill reference and resolve to bare HF repo id; `adapter_name` is used in the `ROSConfigError` message. (L73)
- `resolve_rskill_repo_revision(weights_uri: str, *, adapter_name: str) -> tuple[str, str | None]` — Like `resolve_rskill_repo_id` but also returns the optional `@<sha>` revision pin, threaded into `from_pretrained`/`snapshot_download` by the sim adapters, and warns `rskill.unpinned_weights` when an `hf://` skill is unpinned. (L104)
- `resolve_image_preprocessing(manifest, spec_extra) -> ImagePreprocessing` — Build the `ImagePreprocessing` block an adapter applies, by strict precedence: `spec_extra` YAML override → `manifest.image_preprocessing` per-checkpoint contract → schema defaults. No auto-derivation from `policy.config.input_features` — missing hints surface as the schema default rather than a guessed heuristic. Raises `ROSConfigError` when a manifest alias key is not a slot declared by `sensors_required[].vla_feature_key`. (L147)
- `resolve_state_dim(manifest, spec_extra) -> int | None` — Per-checkpoint proprio state dimension: `spec_extra["state_dim"]` override → `manifest.state_contract.dim` → `None` (adapter falls back to the policy's own preprocessor width). (L252)
- `resolve_camera_keys(manifest, spec_extra, *, scene_cameras=None, default=("camera1", "camera2")) -> tuple[str, ...]` — Resolve which scene camera keys an adapter pulls from the observation: `spec_extra["camera_keys"]` override → `scene_cameras` (the SimEnvironment YAML's `scene.cameras`) → `default`. (L273)
- `manifest_camera_slots(manifest) -> tuple[str, ...]` — The RGB VLA slots an rSkill declares, in `sensors_required` order (`vla_feature_key` suffix). The `resolve_camera_keys` default for adapters whose cameras are the manifest's slots (openvla, rldx, lingbot, diffusion). (L305)
- `checkpoint_image_keys(image_preprocessing, camera_keys) -> tuple[str, ...]` — The one slot→checkpoint rename: `input_template.format(cam=aliases.get(slot, slot))` per slot. Used by the xvla / gr00t / diffusion / lingbot / molmoact2 adapters so manifest `aliases` is the only rename mechanism. (L326)
- `resolve_n_action_steps(manifest, extra, *, default, chunk_size=None) -> int` — The one chunk-slicing rule every policy adapter calls (xr1, xvla, gr00t, lingbot_vla/vla2, rldx, molmoact2, openvla, and via `apply_chunk_replay` smolvla/pi05/act): `extra["n_action_steps"]` (`--n-action-steps`) > `manifest.n_action_steps` > deprecated `extra["replan_steps"]` (warns `vla.deprecated_replan_steps`) > `default`; bounded by `chunk_size` (else `manifest.chunk_size`, logs `vla.n_action_steps_clamped`); `ValueError` below 1. (L354)
- `apply_chunk_replay(policy, spec_extra, *, manifest=None, default_n_action_steps=None) -> int` — Set `policy.config.n_action_steps` via `resolve_n_action_steps`, defaulting to and bounded by the policy's own `config.chunk_size`; used by the lerobot-style adapters (`smolvla`, `act`, `pi05`, `xvla`). (L405)
- `_CUDAGRAPH_COMPILE_MODES` — `frozenset({"reduce-overhead", "max-autotune"})`; the `torch.compile` modes that may capture CUDA graphs and therefore require output cloning (static replay buffers would otherwise be overwritten under lerobot's queued action views / `ChunkedExecutor` pre-fetch). (L460)
- `_has_bnb_quantized_modules(policy) -> bool` — True when any submodule's class comes from `bitsandbytes` (`Linear4bit` / `Linear8bitLt` rewrites from `openral_sim._quantization`). Class-module-path check; never imports bnb. (L472)
- `_clone_chunk_output(out, torch) -> Any` — Recursively `.clone()` every tensor in a chunk forward's output (tensor / tuple / list / dict; non-tensor leaves pass through) so downstream holders own their storage, detached from CUDA-graph static buffers. (L486)
- `maybe_compile_chunk_forward(policy, spec_extra, device, torch, *, method_name="_get_action_chunk") -> bool` — Best-effort `torch.compile` of the chunk forward with a runtime fallback that latches into eager mode on backend errors; skipped on CPU or when `vla.extra.compile` is falsy. Never compiles bitsandbytes-quantized policies (dtype-mismatch under mixed nf4/bf16), and routes cudagraph-mode output through `_clone_chunk_output` on both branches. (L504)
- `run_inference(policy, batch, *, chunk_index=None, kind="single", chunk_size=None, engine=None, call=None, call_kwargs=None, synchronize=False) -> Tensor` — Single seam wrapping a policy inference call in `inference_span` + `torch.no_grad()` — the only place `inference.kind`/`chunk_index`/`chunk_size`/`inference.engine` attributes are emitted. `call` replaces the default `policy.select_action(batch)` for custom chunk producers; `ChunkedExecutor` passes `synchronize=True` on its prefetch so the span times compute, not just kernel launch. **Must stay `torch.no_grad()`, never `torch.inference_mode()`** — RTC's guidance calls `autograd.grad`, which raises under inference mode. (L607)
- `resolve_inference_engine(owner, declared=None) -> str` — Resolve the backend actually executing after optional runtime attachment: explicit `_openral_inference_engine` marker wins, then the `openral-pro-trt` module, then the manifest name (normalized `pytorch→torch`/`tensorrt→trt`). Keeps the dashboard from reporting a stale pre-attachment runtime. (L696)
- `_RTC_ADAPTERS` — `frozenset({"smolvla", "pi05"})`; the flow-matching adapters whose lerobot policies carry an `rtc_config`. molmoact2 / pi0_fast support RTC upstream but are out of scope — extend this set *and* the adapter's chunk_fn kwargs pass-through together. (L736)
- const `_RTC_KEYS` — `frozenset({"enabled", "execution_horizon", "max_guidance_weight", "prefix_attention_schedule", "debug"})`; the closed key set `_parse_rtc_config` accepts in a `policy_extras.rtc` block. (L743)
- `_parse_rtc_config(spec_extra, *, adapter_name) -> RTCConfig | None` — Parse the manifest's `policy_extras.rtc` block into a lerobot `RTCConfig`, or `None` if absent. `ROSConfigError` on a non-mapping block, an unknown key or schedule, a non-positive horizon, `RTCConfig` rejection, or an adapter outside `_RTC_ADAPTERS`; lerobot imports are deferred. (L748)
- `rtc_enabled_in_extra(spec_extra, *, adapter_name) -> bool` — True only for a present, well-formed, enabled `rtc` block — keyed on the parsed `enabled` flag, not just presence, so `rtc: {enabled: false}` still compiles. Lets a factory (e.g. smolvla skipping `maybe_compile_chunk_forward`) decide before the executor exists. (L813)
- `build_chunk_executor(spec_extra, *, policy=None, chunk_fn=None, chunk_size=None, adapter_name="policy", postprocess_action=None) -> ChunkedExecutor | None` — Shared executor construction: `chunk_prefetch` enables overlap, `chunk_prefetch_at` calibrates the lead (default 20, clamped to the chunk). An enabled `policy_extras.rtc` block sets `policy.config.rtc_config` and calls `init_rtc_processor()` before building the executor; RTC is refused with `ROSConfigError` (never a silent downgrade) on chunk size 1, `chunk_prefetch: false`, a policy without `init_rtc_processor`, or bitsandbytes-quantized weights. With no `rtc` block, construction and served actions are byte-identical to before. (L838)
- `to_numpy_action(action_tensor) -> NDArray[np.float32]` — `(1, A)` torch tensor → 1-D float32 NumPy. (L964)
- `release_torch_modules(owner, *attrs, device="") -> None` — Drop an adapter's references to its loaded torch modules, then `gc.collect()` and (on CUDA) `torch.cuda.empty_cache()` — order matters: `empty_cache()` only returns already-free blocks, so it must run after the reference is dropped, and `gc.collect()` is needed since a policy is typically part of a reference cycle. Best-effort; called from `close()` in every sim VLA adapter. (L1381)
- `parse_hf_file_uri(uri: str) -> tuple[str, str | None, str]` — Splits `hf://owner/repo[@rev]/path/to/file.ext` into `(repo_id, revision, filename)` for per-file `hf_hub_download` calls. Rejects bare-repo URIs with a typed `ROSConfigError`. (L1092)
- `materialize_processor_dir(manifest: RSkillManifest) -> str` — Downloads the manifest's per-file `processors` artefacts via `hf_hub_download` and symlinks them under lerobot-canonical filenames in a fresh temp directory, also fetching each processor JSON's sibling `state_file` so `PolicyProcessorPipeline.from_pretrained` resolves every step locally. Used by the SmolVLA and modern-ACT adapters; raises `ROSConfigError` if `manifest.processors is None`. (L1142)
- `hf_download_cached_first(hf_hub_download, local_not_found_exc, *, repo_id, filename, revision=None, **extra) -> str` — Cache-first wrapper around `huggingface_hub.hf_hub_download`: tries `local_files_only=True` first, falling back to a normal call on `LocalEntryNotFoundError`. Avoids the per-file HEAD validation that turns a cached load into seconds of latency on a cold connection; `HF_HUB_OFFLINE=1` extends the skip to inner lerobot/transformers calls this helper doesn't wrap. Public, re-exported from `openral_rskill.__init__`. (L980)
- `local_snapshot_dir(repo_id, *, revision=None, ignore_patterns=("*.md",), **extra) -> str` (L1043) — The resolved checkpoint as a local directory: a directory (an installed rSkill snapshot, which `resolve_rskill_to_hf_with_revision` returns for a snapshot holding `model.safetensors`) is returned as is, a Hub repo id is `snapshot_download`ed with `ignore_patterns` / `extra` forwarded. `snapshot_download` rejects a filesystem path as a repo id, so ACT, Diffusion, GR00T and the processor-sidecar fallback all broke on an installed skill until they routed through this (2026-09-23). Public, re-exported from `openral_rskill.__init__`.
- `suppress_hf_weight_init() -> Iterator[None]` [@contextmanager] — Context manager that patches `transformers.PreTrainedModel._init_weights` to a no-op, skipping SmolVLA's full random backbone init when the checkpoint sets `load_vlm_weights=False`. Only sound when the checkpoint covers every parameter — pair with `assert_all_parameters_finite`. Process-global while active; the skill runner serialises loads behind its resident-skill lock. (L1436)
- `assert_all_parameters_finite(policy, *, repo_id) -> None` — Raises `ROSConfigError` if any floating-point parameter is NaN/Inf — the guard that makes `suppress_hf_weight_init` safe, since uninitialised memory read as float is overwhelmingly non-finite. (L1510)
- `call_make_processors_cached_first(make_pre_post_processors, policy_config, *, pretrained_path, **kwargs) -> tuple[Any, Any]` — Wraps lerobot's `make_pre_post_processors` so the tokenizer's `AutoTokenizer.from_pretrained` call skips its HF HEAD revalidation when that tokenizer is already cached, by flipping `HF_HUB_OFFLINE` for the inner call. Passthrough on a cold cache or for adapters with no tokenizer step (ACT, Diffusion Policy). (L1292)
- private: `_read_tokenizer_repo_from_preprocessor(pretrained_path) -> str | None` — Parses `<pretrained_path>/policy_preprocessor.json` for the `tokenizer_processor` step's `config.tokenizer_name`. Returns `None` on missing/malformed JSON or absent step (ACT / Diffusion Policy). Used by `call_make_processors_cached_first`.
- private: `_hf_tokenizer_is_cached(repo_id) -> bool` — Probes `huggingface_hub.try_to_load_from_cache(repo_id, "tokenizer_config.json")` and returns `True` only when the result is a real cached path (`str`), not `None` or the `_CACHED_NO_EXIST` sentinel. Returns `False` on any import error so callers fall back to a normal online load.

### `python/rskill/src/openral_rskill/_policy_io.py`
_The one policy <-> robot I/O codec (joint order, deg<->rad, gripper scale, clamp), applied on every deploy dispatch path; internal — no public re-export._

- `class PolicyIOCodec(BaseModel)` — Frozen codec built once per skill from the manifest's `action_contract` (`joint_units`, `joint_names`, `gripper_scale`, legacy `policy_extras.gripper_scale` with a deprecation warning) + the `RobotDescription`. State: permute robot→policy order, then `gripper * gripper_scale` / `degrees(joint)`. Action: `gripper / gripper_scale` / `radians(joint)` back in robot order; on the slot path the same per-channel conversion in policy order per slot (JOINT_POSITION/JOINT_VELOCITY + GRIPPER_POSITION), no permutation. (L98)
  - `from_manifest(manifest, description, *, adapter=None) -> PolicyIOCodec` — Joint order: `action_contract.joint_names` (robot names in policy order; unknown name → `ROSConfigError`) > the adapter's `policy.config.action_feature_names` (`.pos`/`_joint_` normalised; mismatch → identity + `policy_io.feature_names_unmatched`) > identity. `ROSConfigError` for a joint-position contract without `joint_units` or a legacy extras scale contradicting the contract. (L134)
  - `is_identity -> bool` [@property] — True when the codec changes nothing (radians, `gripper_scale == 1`, robot joint order); `openral_sim.sim_runner` refuses a non-identity codec on a scene without `converts_policy_units`. (L265)
  - `to_policy_state(robot_state) -> NDArray` — Robot-order radians → policy order + units. (L284)
  - `to_robot_action(policy_action, *, slots=None) -> NDArray` — Policy action → robot units (+ robot order without slots). (L297)
  - `clamp(robot_action) -> NDArray` — Pull each joint strictly (1e-3) inside `RobotDescription` position limits; no-op on a width mismatch. (L345)
- private: `_effective_perm(robot_to_policy, n) -> list[int]` — Identity unless a valid length-`n` permutation, so unit conversion runs on the no-reorder path too. (L57)

### `python/rskill/src/openral_rskill/testing.py`
_Test helpers for skill latency-budget enforcement (CLAUDE.md §5.4)._

- `class LatencyBudgetExceededError(AssertionError)` — Raised by `assert_within_budget` on overrun; a plain `AssertionError` subclass (not `ROSRuntimeError`) so pytest reports a crisp failure with the budget delta rather than the operational exception the safety supervisor would catch. (L42)
  - `__init__(*, stage, measured_ms, budget_ms, rskill_id=None)` — Builds the failure message and stores `stage`/`measured_ms`/`budget_ms`/`rskill_id` on the instance. (L56)
- `assert_within_budget(*, measured_ms, budget: RSkillLatencyBudget, stage="per_chunk", rskill_id=None, tolerance_pct=0.0) -> None` — Assert `measured_ms` does not exceed `budget`'s field for `stage` (`per_chunk_ms`/`warmup_ms`/`load_ms`), with an optional percent tolerance. Unset optional budget stages (`warmup_ms`/`load_ms` are `None`) pass silently — enforcement is opt-in per manifest. Raises `LatencyBudgetExceededError` on overrun, `ValueError` on a negative input. (L78)

### `python/rskill/src/openral_rskill/_lerobot_compat.py`
_Compatibility shim for `lerobot.policies` import side-effects._

- const `_STUB_NAME = "lerobot.policies.groot.modeling_groot"` (L32)
- private: `_install_stub() -> None` (L35)
- `sanitize_smolvla_config(repo_id, *, revision=None) -> None` — Strip config keys lerobot's `SmolVLAConfig` rejects (e.g. `pretrained_revision`) from a checkpoint's `config.json`, in place, before `SmolVLAPolicy.from_pretrained`. Idempotent; no-op when the config is already clean or lerobot/the config are unavailable. (L54)
