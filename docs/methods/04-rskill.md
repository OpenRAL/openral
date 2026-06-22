# Layer 4 — rSkill (S1)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/rskill/src/openral_rskill/base.py`
_rSkillBase — abstract base class with lifecycle state machine._

- `class rSkillBase(abc.ABC)` — Abstract base class for all OpenRAL skills (rSkill is the official package-format name, CLAUDE.md §6.4). (L72)
  - `__init__(name, *, version='0.1.0', role='s1', embodiment_tags=None, latency_budget_ms=None)` — Init only; does not configure or load weights. (L102)
  - `info -> RSkillInfo` [@property] (L126)
  - `name -> str` [@property] (L134)
  - `state -> RSkillState` [@property] (L139)
  - `configure() -> None` — `unconfigured → inactive`. (L146)
  - `activate() -> None` — `inactive → active`. (L171)
  - `deactivate() -> None` — `active → inactive`. (L193)
  - `shutdown() -> None` — Any state → `finalized`. (L213)
  - `step(world_state) -> Action` — One inference step (hot path). (L237)
  - `on_load_weights() -> None` — Hook: load weights. (L283)
  - `on_unload_weights() -> None` — Hook: release weights, called by `shutdown()` (ADR-0050 VRAM eviction). (L290)
  - `on_quantize() -> None` — Hook: apply quantization. (L300)
  - `on_warmup() -> None` — Hook: dummy forward pass. (L307)
  - `_configure_impl/_activate_impl/_deactivate_impl/_shutdown_impl/_step_impl()` [@abstractmethod] (L317)
  - private: `_transition`, `_update`, `_require_transition`, `_enter_error`

### `python/rskill/src/openral_rskill/runtime.py`
_Runtime Protocol and NullRuntime — inference backend contract._

- `class Runtime(Protocol)` — Structural protocol for skill inference backends. (L24)
  - `is_loaded -> bool` [@property] / `device -> str` [@property]
  - `load(path) -> None` (L49)
  - `infer(inputs) -> dict[str, Any]` (L61)
  - `quantize(config: QuantizationConfig) -> None` (L77)
  - `warmup(inputs) -> None` (L89)
  - `unload() -> None` (L97)
- `class NullRuntime` — No-op backend for testing. (L102) — same surface as `Runtime`.

### `python/rskill/src/openral_rskill/runtime_pytorch.py`
- `class PyTorchRuntime` — `torch`-backed `Runtime`. (L41)
  - `__init__(device='cpu')`, `is_loaded`, `device`, `load(path)` (unpickles a full module — gated behind `OPENRAL_ALLOW_UNSAFE_PICKLE`, C2), `load_safetensors(path, *, model, strict=True)` (safe: loads a `state_dict` into a caller-supplied module, no code execution — preferred for new skills), `infer(inputs)`, `quantize(config)` (dynamic INT8 on Linear), `warmup(inputs)`, `unload()` (frees CUDA cache).

### `python/rskill/src/openral_rskill/runtime_onnx.py`
- `class ONNXRuntime` — `onnxruntime`-backed `Runtime`. (L52)
  - same surface as `Runtime`. `quantize(config)` always raises — ONNX quantization is pre-applied.

### `python/rskill/src/openral_rskill/runtime_tensorrt.py`
- `class TensorRTRuntime` — `tensorrt`-backed `Runtime`. (L155)
  - `__init__(device, *, rskill_id, quantization, cache)`, `is_loaded`, `device`, `load(path)`, `serialized_engine(path) -> bytes` (L188), `infer(inputs)`, `quantize(config)`, `warmup(inputs)`, `unload()`. Builds a TensorRT engine from the rSkill's ONNX on first load, cached per host (arch + TRT-version keyed via `EngineCache`). Parses the ONNX via `OnnxParser.parse_from_file` so an external-data sidecar (e.g. `model.onnx.data`, as the RT-DETR detector rSkills ship) is resolved relative to the ONNX directory. Dynamic input dims get a build-time optimization profile; host numpy `infer` is a two-pass (set inputs → resolve output shapes) flow. `serialized_engine(path)` returns the built/cached engine bytes without creating an execution context — the portable artifact `TrtNvmmExecutor` deserializes for the zero-copy NVMM device-pointer path (ADR-0037 PR5b).
- `_engine_cache_tag(compute_capability, trt_version) -> str` (L42) — arch/version cache discriminator; builds the `EngineCache` backend tag (e.g. ``"tensorrt-sm89-trt10.5.0"``). Pure — no GPU or TRT import required.
- `_detect_compute_capability(device_index) -> tuple[int, int]` (L135) — live GPU compute-capability probe via `cuda.bindings`; returns ``(major, minor)`` for the given CUDA device index.

### `python/rskill/src/openral_rskill/engine_cache.py`
_Filesystem-based per-host engine cache for compiled skill runtimes._

- `class EngineCache` — Filesystem-backed cache for compiled skill engine files. (L30)
  - `__init__(cache_dir=DEFAULT_CACHE_DIR)` (L49)
  - `cache_key(rskill_id, backend, config: QuantizationConfig) -> str` — Stable key for skill+runtime+quant. (L56)
  - `get(key) -> Path | None` — Cached engine path or `None`. (L96)
  - `put(key, engine_path) -> Path` — Copy into cache. (L114)
  - `invalidate(key) -> None` — No-op on miss. (L133)
  - `clear() -> None` — Remove all engine files. (L143)
  - `size_bytes -> int` [@property] (L151)
  - `entry_count -> int` [@property] (L163)
  - private: `_key_path`

### `python/rskill/src/openral_rskill/quantization.py`
- `auto_select_quant(device_info: DeviceInfo) -> QuantizationConfig` — Heuristic to pick dtype/backend. (L72)

### `python/rskill/src/openral_rskill/loader.py`
_rSkill loader — HF Hub download, manifest validation, license guard, local registry._

- `class InstalledRSkillEntry(BaseModel)` — One row in the local registry. (L96)
  fields: `repo_id, version, revision, local_dir, manifest_path, license, role, embodiment_tags, installed_at`
- `class rSkill` — Packaged, signed, capability-tagged robot skill. (L140)
  - `__init__(manifest, local_dir)` (L164)
  - `from_pretrained(cls, repo_id, *, revision=None, cache_dir=None, force_download=False, commercial_use=True, registry_path=None) -> rSkill` [@classmethod] — Download from HF Hub, validate, register. (L177)
  - `from_yaml(cls, path, *, local_dir=None) -> rSkill` [@classmethod] — Load locally without network. (L295)
  - `list_installed(registry_path=None) -> list[InstalledRSkillEntry]` [@staticmethod] (L330)
  - `uninstall(repo_id, registry_path=None) -> bool` [@staticmethod] — Remove from registry only. (L361)
  - `check_embodiment_tags(manifest, robot_capabilities) -> None` [@staticmethod] — Verify embodiment tag intersection (raises on disjoint sets). Exempt for perception kinds (`detector`/`vlm`, `_EMBODIMENT_AGNOSTIC_KINDS`): they are camera-in → detections/text-out with no action contract, so they match any robot regardless of tags. (L391)
  - `check_capability_flags(manifest, robot_capabilities) -> None` [@staticmethod] — Verify every `manifest.capabilities_required` flag against `RobotCapabilities`. (L421)
  - `check_runtime(manifest, robot_capabilities) -> None` [@staticmethod] — Verify `manifest.runtime` ∈ `gpu_supported_runtimes` (skipped when host list empty). (L453)
  - `check_quantization_dtype(manifest, robot_capabilities) -> None` [@staticmethod] — Verify `manifest.quantization.dtype` ∈ `gpu_supported_dtypes` (skipped when host list empty). (L476)
  - `check_capabilities(manifest, robot_capabilities) -> None` [@staticmethod] — Composition of the four narrower checks; raises on first failure. (L498)
  - `_check_license(manifest, *, commercial_use) -> None` [@staticmethod] — Enforce license guards (CLAUDE §7.4, §12). (L687)
  - `_validate_eval_jsons(skill_dir) -> None` [@staticmethod] — Validate every `<skill_dir>/eval/*.json` against `RSkillEvalResult` (CLAUDE §6.4). (L788)
  - `_register(entry, registry_path) -> None` [@staticmethod] (L814)
  - `__repr__() -> str` (L838)
- `resolve_rskill_local_dir(uri) -> Path | None` — Return the absolute on-disk directory of an in-tree rSkill referenced by a bare skill ref (bare name, `rskills/<name>`, or Hub repo id), or `None` for Hub-only refs with no in-tree shim. Used by `openral benchmark run` to write `<skill_dir>/eval/<id>.json` and update `<skill_dir>/rskill.yaml` regardless of cwd or which ref form the user typed. (L850)
- `_candidate_local_paths(uri) -> list[Path]` — Enumerate on-disk candidates (cwd-relative + repo-root anchored) for a skill reference. Also unwraps HF Hub form `<org>/rskill-<name>` to in-tree `rskills/<name>`. (L871)
- `discover_intree_rskills() -> list[tuple[str, RSkillManifest]]` — Walk `<repo>/rskills/*/rskill.yaml` and return `(name, manifest)` pairs. Malformed entries are skipped with a stderr warning. (L906)
- `_find_repo_root_from(start) -> Path | None` — Walk up from `start` for the first ancestor containing both `pyproject.toml` and `rskills/`. (L937)
- `_validate_skill_ref(raw) -> str` — Validate and return a bare rSkill reference unchanged. Accepts bare names, `rskills/<name>` paths, or HF repo ids; rejects inputs carrying a known URI scheme (`hf://`, `local://`, `file://`, `http(s)://`). Private — used internally by the CLI and loader. (L950)
- `load_rskill_manifest(uri) -> RSkillManifest` — Resolve a bare skill reference to a parsed manifest. Tries local path → in-tree mapping → HF Hub download. In-process memoised. (L990)
- `resolve_rskill_to_hf(uri) -> str` — Resolve a skill reference to either the underlying HF Hub repo id (`hf://...`) or an absolute local path (`local://...`); both forms are accepted by `from_pretrained` helpers. (L1062)
- `resolve_rskill_to_hf_with_revision(uri) -> tuple[str, str | None]` — Like `resolve_rskill_to_hf` but splits the optional `@<branch-or-sha>` pin off an `hf://` `weights_uri` into a separate `revision` so loaders can pass it to `from_pretrained`/`snapshot_download` instead of gluing it onto the repo id where HF drops it (security audit 2026-06, H4). (L1096)

### `python/rskill/src/openral_rskill/gpu_passthrough.py`
_GpuPassthroughSkill — minimal rSkill whose per-step image processing provably runs on GPU (M8 PR I/10)._

- `_REDUCTION_SIZE: int = 64` — module constant; reduction-target size used to bound GPU latency. (L48)
- `_RGB_CHANNELS: int = 3` — module constant; channel count for the GPU mean-reduction read-back. (L52)
- `class GpuPassthroughSkill(rSkillBase)` — Uploads each `SensorFrame` to torch.cuda, runs per-channel mean reduction (with explicit `torch.cuda.synchronize`), emits result as `Action.confidence`. Refuses silent CPU fallback. (L55)
  - `__init__(sensor_id='wrist_rgb', n_joints=6, horizon=1, device='cuda', latency_budget_ms=None)` (L76)
  - `step_count -> int` [@property] (L104)
  - `on_load_weights/on_quantize() -> None` — no-ops (skill is weight-less). (L110)
  - `on_warmup() -> None` — Allocate the GPU input buffer + launch a kernel so the first step doesn't pay cudaMalloc latency. (L118)
  - `_configure_impl()` — Lazy-import torch, resolve device, raise if `cuda` requested and `torch.cuda.is_available()` is False. (L144)
  - `_activate_impl/_deactivate_impl/_shutdown_impl` (L169)
  - `_step_impl(world_state) -> Action` — Pull frame → CPU→GPU upload → GPU reduction → action with confidence. (L185)
  - private: `_extract_latest_image(world_state) -> NDArray[np.uint8]`, `_gpu_reduce(frame, *, torch) -> (float, float, float)`. (L227 / L260)
- `_channels(encoding: FrameEncoding) -> int` — Per-pixel channel count for a `FrameEncoding`; MONO8 → 1, BGR8/RGB8 → 3, everything else falls through to a defensive BGR default of 3. (L297)
- `_zero_frame() -> NDArray[np.uint8]` — Resilient placeholder when no sensor frame is available yet. (L310)

### `python/rskill/src/openral_rskill/_diagnostics.py`
_Shared load-phase instrumentation seam — generalises the inline `_heartbeat` originally inside `openral_sim.policies.pi05` so every VLA adapter's `_build_*` factory uses the same `<prefix>_<name>_{start,heartbeat,done}` event shape (CLAUDE.md §1.13 — single seam, no duplicates)._

- `phase_timer(name, *, prefix="phase", interval_s=15.0, log=None, gpu_mb=False, **fields) -> Iterator[None]` [@contextmanager] — Emits `<prefix>_<name>_start` / `..._heartbeat` every `interval_s` / `..._done` with `elapsed_s`. `gpu_mb=True` attaches `torch.cuda.memory_allocated()` to the heartbeat for phases that move tensors to/from the GPU. Lazy torch import so CPU-only hosts still work. Consumed by `_pi05_phase` + `_smolvla_phase` in the sim adapters and by `tools/profile_policy_load.py`. (L66)
- `_gpu_mb() -> float | None` — Cheap helper. (L46)

### `python/rskill/src/openral_rskill/executor.py`
_Action-chunk executor — promoted from `smolvla` so every chunked VLA family reuses one implementation (ADR-0010, PR B)._

- `class ChunkedExecutor` — Overlaps GPU chunk inference with robot execution via a background daemon thread. Policy-agnostic; works with any lerobot-style policy exposing `select_action(batch)` + `config.n_action_steps`. (L61)
  - `__init__(policy, *, prefetch_at=5)` — Stash refs, no threads. (L86)
  - `start() -> None` — Mark as running (call after policy is on-device). (L114)
  - `stop() -> None` — Signal background thread, join. (L118)
  - `reset() -> None` — Reset state between episodes. (L126)
  - `select_action(batch) -> Any` — Next action, pre-fetching following chunk if needed. Foreground inference for step 1; pop-from-queue for step 2..N; wait-on-prefetch when queue drains. (L139)
  - private: `_launch_prefetch(batch)` (L207)

### `python/rskill/src/openral_rskill/ros_action_rskill.py`
_ROS-wrapping rSkill adapter — bridges arbitrary ROS 2 action / service servers (MoveIt, Nav2, …) into the `rSkillBase` lifecycle (ADR-0024). Selected by `make_default_skill_resolver` when `manifest.kind in {"ros_action", "ros_service"}`._

- `build_joint_permutation_from_names(*, source_names, target_names) -> list[int]` — Build the permutation that reorders a wrapped server's `JointTrajectory.positions` into the host `RobotDescription.joints` order. Raises `ROSConfigError` on set-inequality so a joint mismatch surfaces loudly instead of silently swapping bytes. (L172)
- `CUMOTION_PIPELINE_ID = "isaac_ros_cumotion"` — the cuMotion MoveIt planning-pipeline id (ADR-0065 D1).
- `maybe_inject_cumotion_pipeline(goal_dict, *, interface_type, capabilities) -> dict` (ADR-0065 D1) — On a host that clears the cuMotion GPU floor (`RobotCapabilities.supports_cumotion()`), set `request.pipeline_id = CUMOTION_PIPELINE_ID` on a `MoveGroup` goal so MoveIt plans with cuMotion; no-op for non-MoveGroup actions, CPU/low-VRAM hosts (→ OMPL default), an already-set `pipeline_id`, or a goal with no `request` block. Pure; never mutates the input. Called by `_configure_impl` after the ADR-0026 goal-merge.
- `class ROSActionRskill(rSkillBase)` — `rSkillBase` shim wrapping a ROS 2 ActionClient (or service client). Two modes selected by `manifest.ros_integration.result_trajectory_field`: trajectory mode replays one waypoint per `step()` and raises `ROSRskillGoalSatisfied` after the last; result-only mode awaits the wrapped result and raises `ROSRskillGoalSatisfied` on success without emitting any `Action`. ROS imports are deferred to `_configure_impl` so the module imports cleanly without ROS sourced. (L301)
  - `__init__(*, manifest, ros_node, robot_description, prompt, prompt_metadata_json)` (L334)
  - `_configure_impl()` — Lazy-import IDL, build ActionClient/service client, parse `default_goal_json`. (L405)
  - `_activate_impl()` — no-op; the wrapped action dispatches on first `step()`. (L493)
  - `_deactivate_impl()` / `_shutdown_impl()` — Release the wrapped client. (L496)
  - `_step_impl(world_state) -> Action` — First call sends goal and caches result; subsequent calls dequeue waypoints. (L513)

### `python/rskill/src/openral_rskill/look_at_rskill.py`
_ADR-0044 Phase 3 — camera-aiming MoveGroup skill. Selected by `make_default_skill_resolver` when `manifest.ros_integration.goal_builder == "look_at"` (new `RosIntegration.goal_builder` field; `RSkillAction` gains `LOOK = "look"`)._

- `resolve_camera_sensor(description, camera) -> SensorSpec` — Find the named camera in `RobotDescription.sensors`; `ROSConfigError` listing the available sensor names on a miss (explicit beats implicit — default camera is `"wrist"`). (ADR-0044)
- `build_look_at_constraints(*, camera_goal: Pose6D, link_name, link_t_cam=None, position_tolerance_m=0.02, orientation_tolerance_rad=0.15) -> dict` — Lower a camera gaze pose into one MoveGroup `goal_constraints` entry. ADR-0054: **delegates to `pose_goal_rskill.build_pose_constraints`** with the optical (z) axis tolerance set to π (roll free); the position/offset math lives there now. With `link_t_cam` the goal is re-expressed for the mount link; without it the camera frame is the constrained link.
- `class LookAtRskill(ROSActionRskill)` — Consumes the merged goal's `look_at` block (`target_xyz` required; `frame_id`, `camera`, `standoff_m`, tolerances) instead of raw constraints. `_configure_impl` pops/validates the block, resolves the camera, builds a TF2 listener; the lowering runs lazily on the first `step()` (needs the camera's *current* TF pose): re-aim in place, or place the camera at `standoff_m` from the target along its current line of approach, then `compute_gaze_pose` (+z optical) → `build_pose_constraints` → constraints injected into `request.goal_constraints` before the parent dispatches. Trajectory replays waypoint-per-chunk through the safety supervisor; the manifest ships `plan_only: true` so MoveIt-side execution never bypasses the kernel.

### `python/rskill/src/openral_rskill/pose_goal_rskill.py`
_ADR-0054 — generic Cartesian end-effector pose MoveGroup skill. Selected by `make_default_skill_resolver` when `ros_integration.goal_builder == "pose"`. Home of the shared pose→constraints lowering `LookAtRskill` reuses._

- `build_pose_constraints(*, pose: Pose6D, link_name, link_t_target=None, position_tolerance_m=0.01, orientation_axis_tolerances_rad=(0.05, 0.05, 0.05)) -> dict` — Lower a target pose into one MoveGroup `goal_constraints` entry (sphere position region + per-axis orientation constraint). `link_t_target` re-expresses the goal for the constrained link (`goal_link = goal_target @ inv(link_t_target)`); the per-axis tolerance tuple lets a generic pose constrain all three axes while look-at frees the optical (z) axis at π. **Reuse watch:** the one place pose→MoveGroup-constraint math lives — do not re-implement.
- `pose_from_block(block) -> tuple[Pose6D, str, float, float]` — Parse a `pose` goal block → `(pose, link_name, pos_tol, orient_tol)`. Orientation is a 4-float quaternion array; component order from `block["quaternion_order"]` (`"xyzw"` default / `"wxyz"`, ADR-0054 Q2). `ROSConfigError` on missing/ill-typed fields or an unknown order.
- `class PoseGoalRskill(ROSActionRskill)` — Consumes the merged goal's `pose` block; lowers it via `build_pose_constraints` (full orientation) on the first `step()`, then dispatches/replays like the parent. `link_t_target` is identity in v1 (the RobotDescription tool-frame offset is ADR-0054 phase 6).

### `python/rskill/src/openral_rskill/joint_goal_rskill.py`
_ADR-0054 — joint-space MoveGroup skill. Selected when `ros_integration.goal_builder == "joint"`. The LLM-facing replacement for hand-written `joint_constraints` JSON._

- `joint_constraints_from_block(block) -> dict` — Lower a `joint` block (`joint_names`, `positions`, optional `position_tolerance_rad`) into one `goal_constraints` entry (`{"joint_constraints": [{joint_name, position, tolerance_above, tolerance_below, weight}, …]}`). `ROSConfigError` on missing/ill-typed fields or a name/position length mismatch.
- `class JointGoalRskill(ROSActionRskill)` — Consumes the merged goal's `joint` block; lowers it into a `joint_constraints` goal at `_configure_impl`, then dispatches/replays like the parent.

### `python/rskill/src/openral_rskill/smolvla.py`
_SmolVLA adapter — rSkillBase implementation for the SmolVLA family of VLAs._

- `from openral_rskill.executor import ChunkedExecutor` — re-exported via `__all__` for back-compat (`from openral_rskill.smolvla import ChunkedExecutor` still works post-ADR-0010 PR B). (L90)
- `class SmolVLAAdapter(rSkillBase)` — Drives any SmolVLA-family policy. (L113)
  - `__init__(repo_id, obs_fn, prompt, *, device='cuda:0', n_dof=6, prefetch_at=5, name='smolvla', version='0.1.0', embodiment_tags=None, latency_budget_ms=None)` (L151)
  - `on_load_weights() -> None` — Fetch checkpoint from HF Hub. (L187)
  - `on_warmup() -> None` — Dummy inference. (L226)
  - `_configure_impl()` — Validate IO shapes match `n_dof`. (L247)
  - `_activate_impl()` — Reset policy, start `ChunkedExecutor`. (L263)
  - `_deactivate_impl()` — Stop pre-fetch, keep weights. (L271)
  - `_shutdown_impl()` — Stop threads, free GPU memory. (L277)
  - `_step_impl(world_state) -> Action` — One S1 step. (L294)
  - `_preprocess(raw) -> dict[str, Any]` — Lerobot preprocessor + tensor → device. (L330)
- `class SO100SmolVLASkill(SmolVLAAdapter)` — Pre-configured for the SO-100 6-DoF arm. (L386)
  - `__init__(prompt, *, repo_id='lerobot/smolvla_base', device='cuda:0', extra_images=None, **kwargs)` (L408)
- `_so100_obs_fn(world_state, *, device, extra_images=None, prompt) -> dict[str, Any]` — SO-100 WorldState → SmolVLA raw input. (L344)

### `python/rskill/src/openral_rskill/_vla_core.py`
_Shared helpers for VLA adapters (Layer 3); internal — no public re-export._

- `InferenceKind` — `Literal["foreground", "prefetch", "single"]` for the `inference.kind` span attribute. (L38)
- `resolve_device(spec: VLASpec) -> str` — `"auto"` → `"cuda:0"` / `"mps"` / `"cpu"` against real torch. (L41)
- `resolve_rskill_repo_id(weights_uri: str, *, adapter_name: str) -> str` — Validate skill reference and resolve to bare HF repo id; `adapter_name` is used in the `ROSConfigError` message. (L66)
- `resolve_rskill_repo_revision(weights_uri: str, *, adapter_name: str) -> tuple[str, str | None]` — Like `resolve_rskill_repo_id` but also returns the optional `@<sha>` revision pin (threaded into `from_pretrained`/`snapshot_download` by the sim adapters) and warns `rskill.unpinned_weights` when an `hf://` skill is unpinned (security audit 2026-06, H4). (L97)
- `apply_chunk_replay(policy, spec_extra) -> int` — Override `policy.config.n_action_steps` from `vla.extra` (default `chunk_size // 2`); used by all lerobot-style adapters (`smolvla`, `act`, `pi05`) to amortise the heavy chunk forward over multiple env steps. (L273)
- `_CUDAGRAPH_COMPILE_MODES` — `frozenset({"reduce-overhead", "max-autotune"})`; the `torch.compile` modes that may capture CUDA graphs and therefore require output cloning (static replay buffers would otherwise be overwritten under lerobot's queued action views / `ChunkedExecutor` pre-fetch). (L332)
- `_has_bnb_quantized_modules(policy) -> bool` — True when any submodule's class comes from `bitsandbytes` (`Linear4bit` / `Linear8bitLt` rewrites from `openral_sim._quantization`). Class-module-path check; never imports bnb. (L344)
- `_clone_chunk_output(out, torch) -> Any` — Recursively `.clone()` every tensor in a chunk forward's output (tensor / tuple / list / dict; non-tensor leaves pass through) so downstream holders own their storage, detached from CUDA-graph static buffers. (L358)
- `maybe_compile_chunk_forward(policy, spec_extra, device, torch, *, method_name="_get_action_chunk") -> bool` — Best-effort `torch.compile` of the chunk forward with a runtime fallback wrapper (latches into eager mode on backend errors). Skipped on CPU and when `vla.extra.compile` is falsy. Two safety gates: bitsandbytes-quantized policies are never compiled (mixed nf4/bf16 graphs trip dtype-mismatch errors — the same reason the pi05 adapter forces `compile_model = False`; logs `vla_compile_skipped_bnb_quantized`), and under cudagraph modes (`_CUDAGRAPH_COMPILE_MODES`) every output is routed through `_clone_chunk_output` on both the compiled and eager-fallback branches. Logs `vla_compile_setup_failed` / `vla_compile_runtime_fallback` on failure. (L376)
- `run_inference(policy, batch, *, chunk_index=None, kind="single", chunk_size=None, engine=None) -> Tensor` — Single seam wrapping `policy.select_action` in `inference_span` + `torch.no_grad()`; the only place `inference.kind` / `chunk_index` / `chunk_size` / `inference.engine` / `inference.device` attributes are emitted across both eval and skill paths. `engine` defaults to `"torch"`; `device` is auto-lifted from `policy.device` when present. (L479)
- `to_numpy_action(action_tensor) -> NDArray[np.float32]` — `(1, A)` torch tensor → 1-D float32 NumPy. (L530)
- `parse_hf_file_uri(uri: str) -> tuple[str, str | None, str]` — Splits `hf://owner/repo[@rev]/path/to/file.ext` into `(repo_id, revision, filename)` for per-file `hf_hub_download` calls. Rejects bare-repo URIs with a typed `ROSConfigError`. (L609)
- `materialize_processor_dir(manifest: RSkillManifest) -> str` — Downloads the manifest's per-file `processors` artefacts (Gap 1+3 of the rSkill self-containment audit) via `hf_hub_download` calls and symlinks them under the lerobot-canonical filenames (`policy_preprocessor.json` / `policy_postprocessor.json`) in a fresh temp directory. Also walks each downloaded JSON's `steps[*]` and downloads any sibling `state_file` (normalizer / unnormalizer `.safetensors`) into the same staging dir so lerobot's `PolicyProcessorPipeline.from_pretrained(<dir>)` resolves every step locally without falling back to `hf_hub_download(repo_id=<dir>)`. Single seam used by the SmolVLA and modern-ACT adapters; raises `ROSConfigError` if `manifest.processors is None`. Every download is routed through `_hf_download_cached_first` so a cache-hit avoids the per-file HF HEAD validation that otherwise stacks 3–5 seconds onto every load. (L659)
- `_hf_download_cached_first(hf_hub_download, local_not_found_exc, *, repo_id, filename, revision=None, **extra) -> str` — Cache-first wrapper around `huggingface_hub.hf_hub_download`. Tries `local_files_only=True` first; on `LocalEntryNotFoundError` falls back to the normal call. Eliminates the per-file HEAD validation that turns a "cached" load into 500 ms – 3 s × N files on a cold TLS connection. Set `HF_HUB_OFFLINE=1` to extend the same skip to the inner lerobot / transformers calls this helper does not wrap.
- `call_make_processors_cached_first(make_pre_post_processors, policy_config, *, pretrained_path, **kwargs) -> tuple[Any, Any]` — Wraps lerobot's `make_pre_post_processors` so the `TokenizerProcessorStep.__post_init__ → AutoTokenizer.from_pretrained` call skips its 5 HF HEAD / `tree/main` revalidations against the backbone tokenizer (`google/paligemma-3b-pt-224` for π0.5; SmolVLM for SmolVLA) when that tokenizer is already in the local HF cache. Reads `<pretrained_path>/policy_preprocessor.json`, probes for `tokenizer_config.json` via `huggingface_hub.try_to_load_from_cache`, and flips `huggingface_hub.constants.HF_HUB_OFFLINE` to `True` for the duration of the inner call (`transformers.utils.hub.is_offline_mode` delegates to the same constant). Passthrough on a cold cache or for adapters whose preprocessor has no tokenizer step (ACT, Diffusion Policy). Call sites: `openral_sim.policies.{pi05,smolvla,act,diffusion}._build_*`.
- private: `_read_tokenizer_repo_from_preprocessor(pretrained_path) -> str | None` — Parses `<pretrained_path>/policy_preprocessor.json` for the `tokenizer_processor` step's `config.tokenizer_name`. Returns `None` on missing/malformed JSON or absent step (ACT / Diffusion Policy). Used by `call_make_processors_cached_first`.
- private: `_hf_tokenizer_is_cached(repo_id) -> bool` — Probes `huggingface_hub.try_to_load_from_cache(repo_id, "tokenizer_config.json")` and returns `True` only when the result is a real cached path (`str`), not `None` or the `_CACHED_NO_EXIST` sentinel. Returns `False` on any import error so callers fall back to a normal online load.

### `python/rskill/src/openral_rskill/_lerobot_compat.py`
_Compatibility shim for `lerobot.policies` import side-effects._

- private: `_install_stub() -> None` (L26)
