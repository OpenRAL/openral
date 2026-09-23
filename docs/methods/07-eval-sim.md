# Eval (sim)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/sim/src/openral_sim/policy.py`
_Policy adapter protocol — the contract every VLA backend must satisfy._

- `class PolicyAdapter(Protocol)` — Uniform VLA / policy interface. (L25)
  - attr `spec: VLASpec`, `device: str`
  - `reset() -> None` — Reset action queue / RNG at episode start. (L36)
  - `step(observation, instruction) -> NDArray[np.float32]` — Next action. (L39)
  - `close() -> None` — Release GPU / file handles. (L57)

### `python/sim/src/openral_sim/rollout.py`
_Sim rollout protocol — the typed contract every scene adapter must satisfy._

- `class StepResult` — One environment transition. (L130)
  fields: `observation, reward, terminated, truncated, info`
- `class SimRollout(Protocol)` — Minimal gym-style env contract. (L149)
  - attr `scene: SceneSpec`, `task: TaskSpec`
  - `reset(seed=None) -> Observation` (L237)
  - `step(action) -> StepResult` (L240)
  - `render() -> NDArray[np.uint8] | None` — HWC uint8 RGB or `None`. (L243)
  - `close() -> None` (L246)
  - duck-typed extension: `mujoco_handles() -> tuple[mujoco.MjModel, mujoco.MjData] | None` — Optional, not part of the Protocol; MuJoCo-backed adapters implement it so `sim run --view` can open a passive viewer. Callers must `getattr(...)` and tolerate `None`.
  - duck-typed extension: `task_success() -> bool | None` — Optional, not part of the Protocol: the backend's own live task-success predicate, read on demand and side-effect free (not latched — an undone task reads `False` again). `None` means no predicate and must never be read as failure; used by `deploy sim`'s `SimAttachedHAL`, not by `sim run`/benchmark.
  - duck-typed extension: `sim_time_ns() -> int | None` — Optional, not part of the Protocol: the backend's elapsed sim time in ns, feeding a `/clock` publisher. Monotonic only within an episode; callers must treat missing and `None` alike as “no clock” (fall back to wall time).
  - duck-typed extension: `enable_intrinsic_viewer() -> None` — Optional, not part of the Protocol; adapters with their own render window (e.g. gym_pusht) implement it so `SimRunner` uses live-view mode instead of the MuJoCo viewer.
- `sim_time_ns_from_mujoco_handles(handles: tuple[Any, Any] | None) -> int | None` — Shared helper: `round(MjData.time * 1e9)` from a `mujoco_handles()` pair, `None` when `handles` is `None`. (L21)
- `env_action_dim(env: object) -> int | None` — The one action-width probe: `env.action_dim`, then `env._env.action_dim`, then a wrapped gym `Box` shape; `None` when not introspectable. Used by `SimAttachedHAL._probe_env_action_dim` and by `SimRunner` to size mock policies. (L90)
- `render_named_rgb_mujoco(renderer, model, data, camera_name, *, height, width) -> tuple[Any, NDArray[np.uint8]]` — Render one named MuJoCo camera, lazily building `mujoco.Renderer` when absent; returns `(renderer, rgb)` so the caller can cache the expensive-to-build renderer. Shared by the `tabletop_push` and `so101_box` scene backends. (L52)
- `class EpisodeResult` — Outcome of one episode. (L251)
  fields: `success, steps, total_reward, mean_step_latency_ms, max_step_latency_ms, latency_budget_ms, budget_violations, frames, metadata`
  - `summary() -> str` — Human-readable single line. (L303)

### `python/sim/src/openral_sim/registry.py`
_Registries that map ID strings to backend factories._

- `class _Registry(Generic[T])` — Tiny ID → factory map. (L43)
  - `__init__(kind)` (L54)
  - `kind -> str` [@property] (L62)
  - `register(name, *, fixed_robot=None, provision=None, **meta) -> Callable[[Callable[..., T]], Callable[..., T]]` — Decorator. Ids are `/`-namespaced: a family registers its prefix once (`robocasa`, `robocasa/gr1`) and every `<prefix>/<task>` id resolves to it (longest prefix wins), so a task is YAML, not a registration. `fixed_robot: str | frozenset[str] | None` (SCENES-only) declares the robot id(s) the backend can instantiate, enforced by `resolve_robot`. SCENES `meta` flags: `sequential_init=True` (env/policy build must not run in parallel — `SimRunner`), `sim_clock=True` (rollout reports `sim_time_ns`, so `deploy sim` uses the simulation clock origin), `base_pose=True` (the scene honours `SimEnvironment.base_pose`; `sim run` rejects a `base_pose:` elsewhere). `provision` (SCENES-only) is the scene's slow first-run setup, run ahead of the HAL's 300s `on_configure`; must be idempotent. `**meta` stores static per-entry facts read back via `meta`; on `POLICIES` every family declares `install_groups` + `required_imports` (optional `install_note`), the single source `policy_deps` reads. (L65)
  - `meta(name) -> dict[str, object]` — Static facts registered with `name` (prefix-resolved; `{}` for an unknown id, like `fixed_robot`). (L210)
  - `get(name) -> Callable[..., T]` — Look up by ID (prefix-resolved); `ROSConfigError` listing the registered ids when nothing matches. (L219)
  - `_key(name) -> str | None` — Exact id, else the longest registered `/`-prefix, else `None`. Every lookup routes through it. (L130)
  - `allowed_robots(name) -> frozenset[str] | None` — Robot ids the scene can instantiate (`None` for free-axis scenes or unregistered names). (L142)
  - `fixed_robot(name) -> str | None` — The scene's default robot (`sorted(allowed_robots)[0]`), `None` if free-axis. (L151)
  - `resolve_robot(name, requested) -> str` — THE robot-binding rule used by `sim run`, `benchmark`, `deploy sim` and the sim HAL: fixed scene → `requested` must be allowed (else `ROSConfigError`), `None` → default; free-axis → `requested` required. (L160)
  - `provision(name) -> Callable[[], None] | None` — Scene's pre-launch provisioner (`None` for backends with nothing to fetch **and** for unregistered names — a preflight is advisory, so unlike `get` this does not raise on an unknown id). (L198)
  - `names() -> list[str]` — Sorted IDs. (L234)
  - `__contains__(name) -> bool` — prefix-resolved, like `get`. (L238)
- module-level globals: `SCENES`, `POLICIES`, `ROBOTS` — three `_Registry[T]` singletons.
- `const T = TypeVar('T')` (L39)
- `const F = TypeVar('F', bound=Callable[..., object])` (L40)
- `const SCENES: _Registry[SimRollout] = _Registry('scene')` (L246)
- `const POLICIES: _Registry[PolicyAdapter] = _Registry('policy')` (L247)
- `const ROBOTS: _Registry[RobotDescription] = _Registry('robot')` (L248)

### `python/sim/src/openral_sim/factory.py`
- `make_env(env_cfg) -> SimRollout` — Build the simulated environment. (L25)
- `make_policy(env_cfg) -> PolicyAdapter` — Build the policy. (L43)
- `make_robot(env_cfg) -> RobotDescription | None` — Resolve robot description if registered. (L61)

### `python/sim/src/openral_sim/sim_runner.py`
_Per-step `InferenceRunner` for the simulation runtime._

- `class SimRunner(InferenceRunnerBase)` — One-tick = one-env-step inference runner that drives a `SimEnvironment` for `n_episodes` episodes. Subclasses `InferenceRunnerBase` so sim and hardware (`DeployRunner`) share the `InferenceRunner` Protocol. (L295)
- `SimRunner.__init__(env_cfg, *, view=False, strict_view=False, instruction_override=None, deadline_overrun_policy=WARN, recorder=None)` — Defers env/policy construction to `activate()`; `rate_hz` is fixed at 1000 Hz (sim is not real-time). `instruction_override` (the `--instruction` CLI value) wins over a scene's per-episode language instruction. `recorder` is an optional `RolloutRecorder` fanned out alongside the episode buffer, additive not a substitute. (L339)
- `SimRunner._record_to_recorder(action, reward, terminated, truncated) -> None` — Extracts per-step state/rendered frame/action and forwards to the attached recorder; broadcasts the single rendered viewpoint to every camera key declared on the robot. Errors are logged, not raised. (L380-ish)
- `SimRunner.activate() -> None` — Validates the manifest, builds env + policy concurrently, arms the first reset-tick, opens the outer `sim.run` OTel span. (L416)
- `SimRunner.deactivate() -> None` — Flush a trailing episode if any, close the viewer / policy / env, close the outer span. Idempotent. (L513)
- `SimRunner._should_terminate() -> bool` — Returns True once `n_episodes` `EpisodeResult`s have been emitted. (L550)
- `SimRunner._tick_impl(tick_idx) -> TickResult` — Dispatch reset-tick vs step-tick. (L598)
- `SimRunner._reset_tick(tick_idx) -> TickResult` — env.reset + policy.reset; `action_applied=False`, `inference_ms=0.0`, `step_idx=None`. (L608)
- `SimRunner._step_tick(tick_idx) -> TickResult` — policy.step + env.step; populates `step_idx`, `reward`, `terminated`, `truncated`, `action_applied=True`. (L662)
- `SimRunner._finalize_episode() -> None` — Build an `EpisodeResult` from the per-step `_EpisodeBuffer`, append to `episode_results`, reset the buffer. (L864)
- `class _EpisodeBuffer` — Private dataclass accumulating per-step latencies / frames / rewards inside one episode; reset on each boundary. (L263)
- `_check_rskill_compatibility(env_cfg) -> RSkillManifest | None` — Load the rSkill manifest, run `rSkill.check_compatibility` against the registered `RobotDescription`, return the manifest or `None` for built-in mock policies. Strict-by-construction. (L975)
- `_SEQUENTIAL_INIT_ENV: str = "OPENRAL_SIM_SEQUENTIAL_INIT"` — Module-level constant: the env var that forces `_build_env_and_policy` onto the legacy sequential path (set to `"1"`). (L1074)
- `_policy_scene_cameras(scene_cameras, description, manifest) -> (list[str], dict[str, str])` — Camera keys the policy is built on plus the obs rekey (scene name → VLA slot): scene cameras that are all robot RGB sensor names become their slots; an empty `scene.cameras` becomes `required_vla_camera_slots`; anything else is unchanged. Makes slot-keyed `image_preprocessing.aliases` mean the same under `sim run` as on deploy. (L207)
- `_rekey_obs_images(obs, rekey) -> Observation` — Returns `obs` with `images` rekeyed scene-name → slot (renamed keys win over an env's own slot alias); applied after every `env.reset`/`env.step`. (L241)
- `_build_env_and_policy(env_cfg, policy_cfg=None) -> (SimRollout, PolicyAdapter)` — `make_env` sees `env_cfg`, `make_policy` sees `policy_cfg` (slot-keyed scene cameras) when given. Builds env + policy concurrently on a 2-worker `ThreadPoolExecutor` by default; sequential when `OPENRAL_SIM_SEQUENTIAL_INIT=1`. Logs `sim_init_parallel`/`sim_init_sequential` timing. Exceptions from either side propagate verbatim. (L1124)
- `_seed_global_rngs(seed) -> None` — Seed Python / NumPy / Torch RNGs so stochastic policies reproduce per `(seed + episode_idx)`. (L1217)
- `_open_viewer_and_pacing(env, env_cfg, *, strict_view) -> (Any, float | None)` — Opens a passive `mujoco.viewer` against the adapter's `mujoco_handles()`, sets the camera + geom visibility via `_aim_viewer_camera`, and computes the per-step sleep budget so the viewer renders at the env's natural sim-time. (L1266)
- `_aim_viewer_camera(viewer, env, mj_model, mj_data) -> None` — Sets the viewer's opening free-camera pose + geom visibility (hides collision shells so textures render); the user can still orbit/zoom afterward. Best effort — failure logs and leaves MuJoCo's default camera. (L1328)
- `SimRunner.episode_start(task_string) -> int` — No-op on `SimRunner`: sim derives episode boundaries from env signals, not this `InferenceRunner` hook. (L564)
- `SimRunner.episode_end(*, success) -> None` — No-op on `SimRunner`: sim closes episodes inside `_finalize_episode`. (L573)
- `const _VIDEO_FRAMES_INFO_KEY = '_openral_video_frames'` (L58)
- `const _IMAGE_NDIM = 3` (L64)
- `const _VIDEO_FRAME_CAP = 8192` (L63)
- `const _GRAYSCALE_CHANNELS = 1` (L65)
- `const _RGB_CHANNELS = 3` (L66)
- `const _RGBA_CHANNELS = 4` (L67)
- `const _MOCK_POLICY_IDS = frozenset({'zero', 'random'})` (L937)
- `const _MOCK_PLACEHOLDER_URI = 'placeholder'` (L938)
- `const _VIEW_ENV = 'OPENRAL_SIM_VIEW'` (L1082)
- `_scene_requires_sequential_init(env_cfg) -> bool` — `SCENES.meta(scene_id)["sequential_init"]`: scenes whose backend construction races the policy build on parallel threads (`openarm_tabletop_pnp`, `tabletop_push`, `maniskill3`, `simpler_env`, `robocasa*`). (L1117)

### `python/sim/src/openral_sim/benchmark.py`
_Benchmark runner — loops a bare `list[BenchmarkScene]` (loaded via `load_benchmark_suite` + `raise_on_invalid_suite`) and emits a `RSkillEvalResult`._

- `check_benchmark_task_compatibility(manifest, *, task_id, scene_id) -> None` — Task-data gate: raises `ROSCapabilityMismatch` when the manifest declares `evaluated_tasks` and none cover the scene's task/scene id (e.g. blocks a LiftCube policy from running on PickCube). Permissive when `evaluated_tasks` is empty. Skipped for mock policies and `hf://` URIs. (L77)
- `_task_matches(task_id, scene_id, declared) -> bool` — Whether a `declared` entry covers the scene: exact `task.id`, a `"<scene>/<…>"` family prefix (`"libero_spatial"` covers `libero_spatial/0..9`), or the bare `scene.id`. (L63)
- `filter_scenes_for_skill(scenes, manifest) -> tuple[list[BenchmarkScene], list[BenchmarkScene]]` — Suite analogue of `check_benchmark_task_compatibility`: partitions a suite into `(kept, skipped)` against `manifest.evaluated_tasks`. Lets one `benchmark run` execute every task an rSkill supports and skip the rest instead of silently scoring 0 on a mismatch. (L140)
- `_manifest_for_filter(vla) -> RSkillManifest | None` — Loads the rSkill manifest for `filter_scenes_for_skill`, or `None` for unfilterable skills (built-in mock policies / raw `hf://` URIs — same guard as the single-scene gate). (L123)
- `run_benchmark(scenes, *, suite_id, vla, device=None, save_dir=None, video_dir=None) -> tuple[RSkillEvalResult, list[EpisodeResult]]` — Auto-filters `scenes` to the rSkill's `evaluated_tasks`, then runs each `(BenchmarkScene, seed)` through a fresh `SimRunner` and aggregates into a validated `RSkillEvalResult`. `video_dir` enables per-episode MP4 capture via `write_world_videos`, freeing frames after each write to stay memory-flat. All args after `scenes` are keyword-only. (L178)
- `_aggregate_results(scenes, *, suite_id, vla, per_task, episodes) -> RSkillEvalResult` — Rolls per-task booleans into per-task/avg success rates. Suite metadata (`name`/`simulator`/`arxiv`) is pulled from `scenes[0].metadata` when present, else falls back to `suite_id`. `max_steps` is the suite worst-case, not just `scenes[0]`. (L345)
- `run_benchmark_scene(scene, vla, *, device=None, save_dir=None, config_path=None, view=None, record_video=False) -> tuple[RSkillEvalResult, list[EpisodeResult]]` — Single-scene sibling of `run_benchmark`; backs `openral benchmark scene`. `record_video` captures per-step frames for clean website MP4s. Raises `ROSConfigError` when `scene.robot_id is None`. `view` mirrors `sim run`'s tri-state viewer flag. (L461)
- `_aggregate_scene_results(scene, vla, successes, episodes, config_path) -> RSkillEvalResult` — Single-scene counterpart of `_aggregate_results`; shares the output schema. Embeds `config_path` into `reproduction_cli` for byte-identical reruns from disk. (L604)
- `default_output_path(weights_uri, benchmark_id) -> str` — Canonical mapping `rskills/<dir>` (or bare name) → `rskills/<dir>/eval/<id>.json`. (L697)
- `update_rskill_benchmarks(skill_dir, benchmark_id, score) -> Path` — Surgical rewrite of the `benchmarks:` block in `<skill_dir>/rskill.yaml` that preserves every other line; re-validates the merged manifest and checks `benchmark_id` against `openral_rskill.loader.known_benchmark_ids()` before writing. Keeps manifest headlines in sync with eval JSONs. Raises `FileNotFoundError`/`ROSConfigError` on a missing manifest or invalid input. (L740)
- `update_rskill_benchmarks_from_uri(weights_uri, benchmark_id, score) -> Path` — Resolve the skill reference to a local dir and delegate to the manifest updater; mirrors `default_output_path` so the CLI passes the same ref it already holds. (L848)
- `const _BENCHMARKS_BLOCK_RE` — Compiled regex matching a `benchmarks:` YAML block (and its indented body) for `update_rskill_benchmarks`'s surgical in-place rewrite. (L734)

### `python/sim/src/openral_sim/cli.py`
- `sim_app: typer.Typer` — Public `openral sim` Typer group. Mounted into the top-level `openral` Typer tree by `openral_cli.main`. Hosts the `run` leaf (`--config / --rskill / --robot / --task / …`) and the `list` leaf (registry printer).
- `sim_run_app: typer.Typer` — The leaf Typer (`invoke_without_command=True`) exposing every rollout CLI flag; users invoke it as `openral sim run`.
- `_sim_run_callback(...)` — Typer callback carrying every rollout CLI flag; builds a `SimpleNamespace` and dispatches to `_run`. The optional `--dashboard` / `--dashboard-port` flags wrap `_run` in `attached_dashboard(...)`. Same flag is mirrored on `openral deploy run` and `openral benchmark run`.
- `_discover_sim_configs() -> list[Path]` — Recursive read-only walk of `scenes/**/*.yaml` (benchmark / sim / deploy) under the repo root, sorted by relative path. Safe to call without any sim dependencies. (L485)
- `sim_list() -> None` — `@sim_app.command("list")` callback that prints every sim config under `scenes/**/*.yaml`, each a paste-able `--config` path for `openral sim run`. No rollout, no OTel span, no GPU. (L506)
- `_resolve_save_video(raw) -> Path | None` — Map the `--save-video` Typer string to the legacy `Path | None` semantics (empty string ⇒ `example_videos/`). (L302)
- `_load_or_build_env(args) -> SimEnvironment` — `--config` and `--rskill` are both required. Per-axis flags overlay the loaded config rather than conflicting with it, and the scene-fixed-robot guard raises `ROSConfigError` when `--robot` disagrees with the scene's fixed robot. `--dry-run` still runs the embodiment/sensor compatibility check. (L318)
- `_resolve_view(flag) -> tuple[bool, bool]` — tri-state resolver returning `(view, strict_view)` from the `--view/--no-view/auto` flag plus `MUJOCO_GL` / `DISPLAY` env. (L524)
- `main(argv=None) -> int` — Thin wrapper invoking `sim_run_app` with `standalone_mode=False` so tests get a return code without `sys.exit`. (L453)
- `_run(args) -> int` — Body of the callback after argv parsing + OTel setup. Configures observability with service name `ral-sim`. (L713)
- `_write_videos(args, results, env_cfg) -> None` — Dispatch `--save-video` to the debug or world writer per `--video-style`; raises `ROSConfigError` for any other style. (L818)
- `_write_debug_videos(args, results, env_cfg) -> None` — Render the two-band debug MP4(s) via `openral_sim._video.save_episode_mp4`. (L833)
- `_write_website_videos(args, results, env_cfg) -> None` — Thin adapter that pulls scene/rskill/section from the run and delegates to `openral_sim._website_video.write_world_videos`. `--video-style world`. (L877)

### `python/sim/src/openral_sim/_video.py`
_Shared two-band rollout-debug MP4 helper (was `examples/_video.py`)._

- `save_episode_mp4(result: EpisodeResult, path: Path, *, title: str = "") -> Path` — Render `(policy input grid | observation-state plot)` for one episode, falling back to the rollout/world stream only when the adapter recorded no input frames. Re-exported from `openral_sim`. (L62)
- `_stack_padded_states(states) -> NDArray[np.float32]` — Pad ragged observation-state arrays for plotting. (L167)
- `_resize_sequence(frames, target) -> list[NDArray[np.uint8]]` (L182)
- `_resize_frame(frame, target) -> NDArray[np.uint8]` (L220)
- `class _JointPlotRenderer` — Reusable matplotlib canvas rasteriser; labels generic state channels `s0...` rather than claiming every backend's `observation.state` is a joint vector. (L240)
  - `__init__`, `render_at_step`, `_snapshot`, `__del__`
- `const _TILE_SIZE = 256` — Per-camera input-grid tile side, px. (L47)
- `const _TOP_W = _TILE_SIZE * 2` (L48)
- `const _PLOT_H = 192` (L49)
- `const _CANVAS_W = _TOP_W` (L50)
- `const _CANVAS_H = _TILE_SIZE + _PLOT_H` (L51)
- `const _RGB_CHANNELS = 3` (L53)
- `const _GRAYSCALE_NDIM = 2` (L54)
- `const _MAX_LEGEND_JOINTS = 8` (L57)
- `const _LABEL_PAD = 6` (L58)
- `const _LABEL_BAR_H = 24` (L59)

### `python/sim/src/openral_sim/_website_video.py`
_Clean single-view world MP4 helper for website hero clips (overlays rendered by the page, not burned into pixels)._

- `save_world_mp4(result: EpisodeResult, path: Path, *, fps: int = 20, size: int = 1024, min_duration_s: float = 2.0) -> Path` — Write only `result.frames` (the world/viewer render), center-cropped to a square and resized to `size × size`, libx264/yuv420p, holding the final frame when needed so short successes remain watchable. Raises `ValueError` on empty frames, non-`.mp4` suffix, non-positive `size`, or negative `min_duration_s`. (L43)
- `write_world_videos(episodes, out_dir, *, scene, rskill, section, size=1024, fps=20) -> list[dict]` — Write one clean world MP4 per episode named `<scene>_<rskill>[_ep<i>]_<success|fail>.mp4` + merge a `videos.json` manifest. Shared by `openral sim run --video-style world` and `openral benchmark scene --save-video`. (L117)
- `append_video_manifest(manifest, records) -> None` — Merge video records into `videos.json`, replacing same-`file` entries; no-op on empty. (L189)
- `_square(frame, size) -> NDArray[np.uint8]` — Center-crop one HWC frame to a square and resize to `(size, size)` RGB. (L225)
- `const _RGB_CHANNELS = 3` (L38)
- `const _GRAYSCALE_NDIM = 2` (L39)
- `const _DEFAULT_SIZE = 1024` (L40)

### `python/sim/src/openral_sim/_assets.py`
_Lazy asset-fetch helpers for sim backends with large CC-BY downloads (today only the RoboCasa adapter, named generically for a future MuJoCo backend with its own multi-GB bundle). Asset caches live under `$OPENRAL_CACHE_HOME` (default `~/.cache/openral/`), one subdirectory per backend, gated by a readiness sentinel + a Rich license banner + an env-var CI bypass. Sibling of `openral_sim._deps`, which handles the install-chain half of first-run provisioning._

- `const _DEFAULT_CACHE_HOME = Path.home() / '.cache' / 'openral'` (L40)
- `const _READY_SENTINEL = '.openral-ready'` — Marks "assets fully unpacked" so a re-run short-circuits. (L41)
- `const _ROBOCASA_ALLOW_ENV = 'OPENRAL_ALLOW_ROBOCASA_ASSETS'` — CI bypass for the download-confirmation prompt. (L43)
- `const _ROBOCASA_ASSETS_SIZE_GB = 11` (L44)
- `ensure_robocasa_assets() -> Path` — Make sure the RoboCasa assets are on disk; download if needed. (L107)

### `python/sim/src/openral_sim/_deps.py`
_Lazy auto-install helpers for sim backends with bespoke install chains that go beyond `uv sync --group <name>` (editable git clones, `--no-deps` robosuite pins, compiler env overrides). `BackendInstallPlan` declares a sequence of subprocess steps + probe imports; `ensure_backend_deps` short-circuits when the probes already pass, else prints a Rich banner and auto-installs (`OPENRAL_AUTO_INSTALL_DEPS=0` prompts instead). Sibling of `openral_sim._assets`, which handles the asset-download half._

- `class InstallStep` — One subprocess step in a backend install plan: `description`, `argv`, `env`, `cwd`. (L74)
- `class BackendInstallPlan` — A backend's "what to install + how to check it's there" recipe: `backend_id`, `display_name`, `license_note`, `probe`, plus its install `steps`. (L98)
- `const _STEP_TAIL_BYTES = 8192` — Cap on a failed step's captured stdout/stderr tail. (L442)
- `const _STEP_TAIL_LINES = 25` (L445)
- `const _AUTO_INSTALL_ENV = 'OPENRAL_AUTO_INSTALL_DEPS'` (L47)
- `const _DEFAULT_CACHE_HOME = Path.home() / '.cache' / 'openral'` (L46)
- `const _INSTALL_LOCK = threading.Lock()` — Serializes concurrent `ensure_backend_deps` calls (e.g. the parallel `_build_env_and_policy` env/policy construction) so two threads never race the same install plan. (L49)
- `const _ROBOCASA_RUNTIME_DEPS` — Import probes for the RoboCasa kitchen plan (`lxml`, `h5py`, `llvmlite`, `numba`, `robosuite`, `robosuite.examples`, `robocasa`, …). (L173)
- `const _ROBOCASA_GR1_RUNTIME_DEPS` — `_ROBOCASA_RUNTIME_DEPS` minus `robosuite.examples` (the GR1 fork ships no examples package). (L185)
- `const _ROBOSUITE_PIN = '5ce6643f3092639d08f7b0f90ed1c6a84f50552c'` — Pinned robosuite fork commit for the RoboCasa plans. (L616)
- `const _SIMPLER_ENV_GIT_URL` — `pip`-installable git spec for the SimplerEnv maniskill3 branch. (L1144)
- `const _VLABENCH_RRT_STUB_SYMBOLS` — `(module, symbol)` pairs VLABench's motion-planning stub must provide when the real RRT deps are absent. (L1665)
- `const _RLDX_SIDECAR_REPO_URL = 'https://github.com/RLWRLD/RLDX-1.git'` (L1392)
- `const _RLDX_SIDECAR_HOME = _DEFAULT_CACHE_HOME / 'rldx-sidecar'` (L1393)
- `const _PLANS: dict[str, Callable[[], BackendInstallPlan]]` — Backend id → lazy plan-builder registry (`robocasa_kitchen`, `robocasa_gr1`, …). (L1843)
- `get_plan(backend_id) -> BackendInstallPlan` — Resolve a backend id to its install plan, building lazily. (L1862)
- `remediation(manual_hint) -> str` — Return `manual_hint`, plus a runnable equivalent when `just` is absent. (L1927)
- `ensure_backend_deps(backend_id) -> None` — Install `backend_id` deps if missing, after the user confirms. (L1988)

### Eval adapters

#### `python/sim/src/openral_sim/backends/robocasa.py`
_RoboCasa kitchen + GR1 tabletop adapter._
- `provision_robocasa(backend_id) -> None` — The slow half of a first run: installs deps, probes `import robocasa` (with an actionable libero/robocasa conflict hint), then downloads the ~11 GB asset bundle. Registered as the scene's `provision=` so `deploy sim` can run it before `ros2 launch` instead of inside the HAL's 300s `on_configure`. Idempotent. (L1729)
- `_robocasa_backend_id(scene_id) -> str` — `"robocasa_gr1"` for the `robocasa/gr1/<Task>` family, else `"robocasa_kitchen"`. The two packages both import as `robocasa` but ship different asset trees, so provisioning the wrong one leaves the scene unrunnable.
- `_build_robocasa_kitchen(env_cfg) -> _RoboCasaSim` / `_build_robocasa_gr1(env_cfg) -> _RoboCasaSim` — The two scene registrations: `robocasa` (procedural + every `robocasa/<Task>` kitchen id, `fixed_robot={panda_mobile, panda_mobile_vslam}`) and `robocasa/gr1` (every GR1 tabletop id, `fixed_robot=gr1`). The task is read off `env_cfg.scene.id`; `_CURATED_PREBUILT_TASKS` / `_GR1_TABLETOP_TASKS` are documentation only.
- `_require_registered_env(env_name, *, gym_env_id=None) -> None` — Typed `ROSConfigError` for a task name robosuite (or the GR1 gym registry) does not know — where a typo'd `robocasa/<Task>` now surfaces, since the registry accepts any task by prefix.
- `_resolve_state_layout(env_cfg, opts) -> str` — Obs `state_layout`: an explicit `backend_options.state_layout` wins; else the rSkill manifest's `state_contract.layout` mapped via `_MANIFEST_TO_ROBOCASA_LAYOUT` (`rc365`→`human300_16d`, `human300_16d`, `smolvla_9d`, `gr1`); else the schema default. One scene YAML serves every rSkill.
- `_RoboCasaSim.enable_continuous() -> None` — Deploy-sim continuous mode: disables task evaluation and sets robosuite `ignore_done` on the live env (replaces the old scene-id-keyed injection in `sim_bringup`).
- `_RoboCasaSim.action_dim -> int` [@property] — Flat width `step` accepts (29 for the GR1 BASIC composite, else the robosuite env's).
- `_single_scene_pin(ids) -> int | None` — The one concrete scene id a `layout_ids`/`style_ids` value pins, or `None` when it denotes a pool to sample from. A scalar or single-id list is a pin; a multi-element list or negative group shorthand (e.g. `-2` = all train layouts) is a draw.
- `_assert_and_log_scene_composition() -> None` (method on `_RoboCasaSim`) — Emitted once per `reset`; raises `ROSConfigError` when the pinned layout/style is not what RoboCasa actually composed (task filtering or a replayed `_ep_meta` can silently override a pin). Runs at reset because composition is redrawn every episode. No-op on the GR1 tabletop fork.
- `_fit_panda_mobile_action(action, *, env_dim, state_layout) -> NDArray[np.float32]` — Reconciles the known 12↔11 RoboCasa dataset/BASIC skew; only `state_layout="xr1_8d"` may zero-fill a 7-D arm+gripper action into PandaMobile base+torso slots. Unknown widths raise `ROSConfigError`.
- `_xr1_robocasa_state(raw) -> NDArray[np.float32]` — Builds XR-1 RoboCasa's 8-D `[arm_joint_pos(7), gripper(1)]` state from the real robosuite observation.
- `read_panda_mobile_base_velocity(model, data) -> NDArray[np.float32]` — Returns body-frame `(vx, vy, wz)` for the robosuite OmronMobileBase, de-rotated by the live yaw. Returns `zeros(3)` when the base joints aren't in this model (no-op for non-PandaMobile envs). (L1288)
- `synthesize_laser_scan_2d(*, model, data, base_body_id=None, n_beams=360, max_range_m=12.0, laser_height_m=0.30) -> NDArray[np.float32]` — Single-origin 2D laser fan from the panda_mobile base via one `mj_ray` per beam (not the batched `mj_multiRay`, which can't re-cast past a self-hit). Ranges are clamped to `max_range_m`, never NaN/inf, so a Nav2 costmap can't be poisoned. Self-excludes the chassis body. (L1367)
- `head_cam_enabled() -> bool` — True iff `OPENRAL_ROBOCASA_HEAD_CAM` is set, gating the synthetic forward nav camera so a manipulation run pays for no extra render. `deploy sim`/`run` derive it automatically from the palette; an explicit env value still wins. (L1175)
- `render_head_view(renderer, model, data, base_joint_names, *, height_m=1.30, forward_offset_m=0.42) -> NDArray[np.uint8] | None` — Renders the forward egocentric `head` frame from the robot.yaml `head` sensor's geometry. `None` for non-mobile-base models. Consumed by the InternVLA-N1 VLN rSkill via `observation.images.head`. (L1195)
- `_emit_panda_mobile_extras(obs)` (method on `_RoboCasaSim`) — Attaches `obs["robot0_base_vel"]`/`obs["robot0_scan"]` when the robot has a mobile base (name contains `"Mobile"`/`"Omron"`); no-op otherwise. (L569 in `_wrap_obs`)
- `sim_time_ns() -> int | None` (method on `_RoboCasaSim`) — `round(MjData.time * 1e9)` off `mujoco_handles()`. RoboCasa rewinds the clock on `reset`, so it is monotonic only within an episode — `SimAttachedHAL.sim_time_ns` adds the cross-reset offset. (L705)
- `task_success() -> bool | None` (method on `_RoboCasaSim`) — The env's own task-success predicate (robosuite/RoboCasa `_check_success()`), read on demand and side-effect free. Live, not latched — an undone task reads `False` again. `None` means no predicate and must never be read as failure. Not consulted during `deploy sim`'s continuous stepping; `SimAttachedHAL` polls it instead. (L490)
- `refresh_obs() -> Observation | None` (method on `_RoboCasaSim`) — Re-reads observations via robosuite's non-stepping `_get_observations(force_update=True)`, so a direct base-qpos write can refresh the dashboard/WorldState without advancing physics. Must never step the env — stepping lets the mobile-base controller regulate the qpos write away and double-advances sim time. `None` for backends with no such refresh.
- Constants `_OMRON_BASE_JOINT_NAMES`, `_OMRON_BASE_JOINT_NAMES_FALLBACK`, `_LASER_DEFAULT_N_BEAMS=360`, `_LASER_DEFAULT_MAX_RANGE_M=12.0`. (L1132)

#### `python/sim/src/openral_sim/backends/depth_camera.py`
_Simulated depth camera via MuJoCo CPU ray-casting (the 3-D analogue of `synthesize_laser_scan_2d`); robot-agnostic, no GL/EGL context. Feeds the deploy-sim HAL → octomap_server → the kernel's world-collision voxel check. The cast is the whole cost, so callers cast once per camera per frame and derive every output from that raster._
- `noncollidable_geom_ids(model) -> NDArray[np.int64]` — The geom ids MuJoCo can never form a contact pair for (neither `contype` nor `conaffinity` set). Filtered out of every depth cast, because a geom no body can touch is not world the map is allowed to contain. The predicate is MuJoCo's own collision test, not a scene rendering convention, so it holds for any MJCF and any robot. (L47)
- `const _GEOMGROUP_ALL = np.ones(6, dtype=np.uint8)` — All-groups-visible mask used by `noncollidable_geom_ids`'s temporary group-hide trick. (L44)
- `synthesize_depth_pointcloud(*, model, data, camera_name, width, height, fx, fy, cx, cy, max_range_m, min_range_m=0.0, stride=1, exclude_body_id=None, exclude_body_ids=None) -> NDArray[np.float32]` — One `mj_ray` per (strided) pixel through a pinhole model on the camera's live pose — never the batched `mj_multiRay`, whose broad-phase culling can report free space where a surface exists. Intangible geometry is filtered from both passes so a return can only move farther or vanish, never nearer, keeping touchable geometry in the map. Returns `(N,3)` camera-optical-frame hits; `exclude_body_ids` filters the robot's own bodies without leaving a hole in the map. (L276)
- `synthesize_depth_frame(*, model, data, camera_name, width, height, fx, fy, cx, cy, max_range_m, min_range_m=0.0, stride=1, exclude_body_id=None, exclude_body_ids=None) -> tuple[NDArray[np.float32], NDArray[np.bool_]]` — One cast, both products: the depth raster and the disjoint mask of pixels whose only return was a self-filtered body with no farther surface. OctoMap needs the mask to distinguish “no return” from “clear because only the robot's own arm is there” — without it the robot's silhouette becomes a write-only region of the map that a stale voxel can never clear. This is what the deploy-sim depth timer calls. Raises `ROSConfigError` if `camera_name` is absent. (L365)
- `synthesize_depth_image(*, model, data, camera_name, width, height, fx, fy, cx, cy, max_range_m, min_range_m=0.0, stride=1, exclude_body_id=None, exclude_body_ids=None) -> NDArray[np.float32]` — The raster half of `synthesize_depth_frame`, for callers that only integrate depth: a dense grid of perpendicular optical-Z depth in metres, `0.0` where the cast misses or hit only a self-filtered body. Feeds nvblox's projective depth integrator; do not build an OctoMap cloud from this alone (use `synthesize_depth_frame`, which also carries the clearing mask). Raises `ROSConfigError` if `camera_name` is absent. (L451)

#### `python/sim/src/openral_sim/backends/libero.py`
- `class _LiberoSim` — `SimRollout` wrapping `LiberoEnv`. (L105) — `reset/step/render/close/action_dim/mujoco_handles/sim_time_ns/_wrap_obs/enable_continuous/_apply_ignore_done/_robosuite_env`. `enable_continuous()` (used by `deploy sim`, no-op for `sim run`) suppresses robosuite's `ignore_done` and swallows LiberoEnv's inline re-randomizing reset, so a task success or horizon no longer resets the scene mid-mission (which would orphan the passive viewer). `mujoco_handles()`/`sim_time_ns()` expose the MuJoCo viewer/clock seams; `action_dim` sums robosuite `robots[*].action_dim` (LIBERO OSC_POSE = 7) for `SimAttachedHAL`.
- `_parse_task_id(task_id, scene_id) -> int` — Validate `<suite>/<int>` format. (L70)
- `_quat_to_axisangle(quat) -> NDArray[np.float32]` — `[x,y,z,w]` → axis-angle. (L350)
- `_build_libero_scene(env_cfg) -> _LiberoSim` (L446)
- `const _LIBERO_SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')` (L44)

#### `python/sim/src/openral_sim/backends/vlabench.py`
_VLABench (ICCV 2025) Franka adapter. Pins the upstream-tested MuJoCo 3.2.2 + dm_control 1.0.22 pair (newer resolution crashes during dm_control model indexing). Policy cameras use VLABench's real raw camera order `[front=2, base=0, wrist=3]`, not LeRobot's stale `[0,1,2]` mapping._
- `class _VLABenchSim` — `SimRollout` over lerobot's `VLABenchEnv`; emits three RGB cameras, 7-D EE/gripper state, and `info["is_success"]`.
- `_parse_task_id(task_id) -> str` — Validate `vlabench/<task-name>`.
- `_select_policy_cameras(raw_rgb) -> dict[str, NDArray[np.uint8]]` — Select raw camera indices `(2,0,3)` into `camera1/2/3`; normalize HWC channels without vertical/horizontal flips.
- `provision_vlabench() -> None` — Pre-launch provisioner: `ensure_backend_deps("vlabench")` + resolves the root + checks assets. The ~12 GB CC-BY bundle is a Google-Drive pull deliberately never automated, so this verifies it and raises the recipe when absent. Registered as `provision=` on the `vlabench` scene. (L236)
- `_build_vlabench_scene(env_cfg) -> _VLABenchSim` — Verify assets, build the single-env vector wrapper, and register `scene.id="vlabench"` with fixed robot `franka_panda` (and `provision=provision_vlabench`).

#### `python/sim/src/openral_sim/policy_deps.py`

_Import-probe + install-hint helpers for policy runtimes, keyed on `RSkillManifest.model_family`. The per-family facts are the `install_groups` / `required_imports` / `install_note` each family declares on its `@POLICIES.register(...)` (read via `POLICIES.meta`), so an unregistered family is unknown (kept, generic hint) and a registered one can never lack facts (`test_every_registered_family_declares_its_deps`). Sidecar families (XR-1, RLDX, LingBot, InternVLA-N1, 3DDA, `gr00t_b1k`) probe only their ZMQ/msgpack wire._

_**Probe tiers.** The default probe resolves only each import's top-level package (`importlib.util.find_spec`), not the deep module — the deep import costs seconds because `lerobot.policies.__init__` eagerly imports every family's config class. This catches a never-installed group but not an installed-but-broken one, which still surfaces at dispatch as a `ROSRuntimeError`. Set `OPENRAL_STRICT_POLICY_PROBE=1` to restore the deep-import probe._

- `can_import_policy_manifest(manifest) -> tuple[bool, str | None]` — `can_import_policy_family(manifest.model_family)`. (L198)
- `manifest_install_groups(manifest) -> tuple[str, ...]` — `model_family_install_groups(manifest.model_family)`. (L203)
- `manifest_install_hint(manifest) -> str` — `model_family_install_hint(manifest.model_family)`. (L208)
- `model_family_required_imports(family) -> tuple[str, ...]` — Leaf modules whose presence proves `family`'s policy factory will load; empty tuple for an unknown family (no false-negative on an out-of-tree policy). Read from the family's registered `required_imports`. (L126)
- `filter_importable_manifests(manifests, *, log_fn=None) -> list` — Return the subset of manifests whose `model_family` can be imported (via `can_import_policy_manifest`); an unknown family is kept unchanged, and each dropped manifest is reported through `log_fn` with an actionable install hint. (L213)
- `can_import_policy_family(family) -> tuple[bool, str | None]` — Probe whether `family`'s policy factory can resolve its imports (fast top-level probe, or the deep `_deep_import_probe` under `OPENRAL_STRICT_POLICY_PROBE=1`). (L136)
- `model_family_install_groups(family) -> tuple[str, ...]` — The `uv sync --group …` group names that install `family` (its registered `install_groups`). (L116)
- `model_family_install_hint(family) -> str` — Actionable install command for a given `model_family`, formatted from its `install_groups` (+ `install_note`); generic hint for an unregistered family. (L90)
- `purge_partial_imports(prefixes) -> None` — Drop modules under `prefixes` from `sys.modules` after a failed import, so a retry doesn't see a half-imported package. (L245)
- `const _STRICT_PROBE_ENV = 'OPENRAL_STRICT_POLICY_PROBE'` (L82)

### `python/sim/src/openral_sim/backends/metaworld.py`
_MetaWorld MT-50 scene adapter. Opt-in via the `metaworld` dependency group + a `metaworld==3.0.0 --no-deps` pip install (its transitive deps conflict with the workspace lock). Scene id `metaworld`; task id `metaworld/<task-name>` (e.g. `metaworld/reach-v3`)._
- `class _MetaworldSim` — `SimRollout` wrapping `MetaworldEnv`. (L46) — `reset/step/render/close/mujoco_handles/sim_time_ns/_wrap_obs`. `mujoco_handles()` reaches through `unwrapped.{model,data}` for `sim run --view`; `sim_time_ns()` returns `round(MjData.time * 1e9)`.
- `_parse_task_id(task_id) -> str` (L36)
- `_build_metaworld_scene(env_cfg) -> _MetaworldSim` (L133)
- `const _METAWORLD_SCENE_ID = 'metaworld'` (L32)
- `const _METAWORLD_RENDER_SIZE = 480` (L33)

#### `python/sim/src/openral_sim/backends/maniskill3.py`
_ManiSkill3 (SAPIEN-backed) free-axis scene adapter. Opt-in via the `maniskill3` dependency group (auto-install banner on first use, bypass with `OPENRAL_AUTO_INSTALL_DEPS=1`). Scene id `maniskill3`; task id `maniskill3/<env_id>` (e.g. `maniskill3/PickCube-v1`)._
- `_MANISKILL3_SCENE_ID = "maniskill3"` — module constant; scene-registry key. (L40)
- `_sapien_sim_time_ns(env) -> int | None` — Derive elapsed SAPIEN/ManiSkill sim time from a live env's elapsed step counter (`elapsed_steps` / `_elapsed_steps`) and control period (`control_timestep` / `control_dt` / `control_freq`). Returns `None` when the env does not expose a usable clock seam.
- `class _ManiSkill3Sim` — `SimRollout` wrapping a MS3 gym env with `num_envs=1`, unwrapping the leading batch dim on every obs/step. (L215) — `reset/step/action_dim/sim_time_ns/render/close/_wrap_obs`. `action_dim` returns the single-env width for `SimAttachedHAL`; `sim_time_ns()` returns SAPIEN elapsed control time.
- `_parse_task_id(task_id) -> str` — Validates `maniskill3/<env_id>` and returns `<env_id>`. (L50)
- `_task_id_for_env(env_cfg) -> str` — Resolves the concrete ManiSkill env id. Normal sim tasks parse `maniskill3/<env_id>`; deploy-sim's synthetic `_hal_deploy_noop` task maps to `scene.backend_options.deploy_task_id` or `PickCube-v1` so taskless DeployScenes still build a real backend env.
- `_reconcile_robot_uids(env_id, robot_uids) -> None` — Validates a scene's `robot_uids` against the task's `SUPPORTED_ROBOTS`, accepting a registered camera-variant subclass of a supported base; raises `ROSCapabilityMismatch` for a genuinely unsupported robot instead of MS3's vague warning + downstream crash. (L66)
- `class _DropUnsupportedRobotWarning(logging.Filter)` / `_suppress_unsupported_robot_warning()` — Context manager that drops MS3's false “not in the task's list of supported robots” log record around `gym.make`, after `_reconcile_robot_uids` has already validated the variant. (L113 / L123)
- `_unbatch(value)`, `_unbatch_info(info)`, `_unbatch_obs(obs)` — recursive numpy / torch unbatch helpers shared with the SimplerEnv adapter. (L106 / L114 / L130)
- `_extract_rgb(flat)` — Returns the first MS3 `sensor_data.<camera>.rgb` stream as `NDArray[uint8]`. (L360)
- `_extract_state(flat)` — Concatenates `agent.qpos` + `agent.qvel` into a 1-D float32 vector (returns 0-D when the obs mode doesn't expose the nested `agent` block). (L403)
- `_build_maniskill3_scene(env_cfg) -> _ManiSkill3Sim` — `gym.make` with `obs_mode` / `control_mode` overridable via `scene.backend_options`; default `state_dict+rgb` + `pd_ee_delta_pose`. (L426)
- Module side effect: `SCENES.register("maniskill3", fixed_robot="franka_panda", sequential_init=True, sim_clock=True)(_build_maniskill3_scene)` at import (L426).
- `const _VIEW_ENV = 'OPENRAL_SIM_VIEW'` (L47)
- `const _DEPLOY_NOOP_SUFFIX = '/_hal_deploy_noop'` (L41)
- `const _UNSUPPORTED_ROBOT_WARNING = "not in the task's list of supported robots"` — The false MS3 log-record text `_suppress_unsupported_robot_warning` filters. (L124)

#### `python/sim/src/openral_sim/backends/simpler_env.py`
_SimplerEnv real-to-sim correlator adapter. Opt-in via the `simpler-env` dependency group (no PyPI release; the typed `ROSConfigError` carries the install hint). Reuses `backends/maniskill3`'s obs-extraction helpers since SimplerEnv sits on MS3 v3.0.x. Scene id `simpler_env`; task id `simpler_env/<friendly_name>`, translated via `simpler_env.ENVIRONMENT_MAP`. Only the four WidowX bridge tasks are wired end-to-end; `google_robot_*` names are not yet registered upstream._
- `_SIMPLER_ENV_SCENE_ID = "simpler_env"` — module constant; scene-registry key. (L71)
- `_DEFAULT_OBS_MODE = "rgb+segmentation"` — Only obs mode the MS3 v3.0.x Bridge envs advertise; overridable via `scene.backend_options.obs_mode`. (L79)
- `class _SimplerEnvSim` — `SimRollout` wrapping a SimplerEnv-via-MS3 gym env. (L270) — `reset/step/action_dim/sim_time_ns/render/close/_wrap_obs`. Reshapes the single-env action to `(1, action_dim)` for MS3's batched API; `action_dim`/`sim_time_ns()` mirror the ManiSkill adapter.
- `_parse_task_id(task_id) -> str` — Validates `simpler_env/<friendly_name>` and returns `<friendly_name>`. (L95)
- `_task_name_for_env(env_cfg) -> str` — Resolves the concrete SimplerEnv friendly/raw task name. Normal sim tasks parse `simpler_env/<friendly_name>`; deploy-sim's synthetic `_hal_deploy_noop` task maps to `scene.backend_options.deploy_task_id` or `widowx_carrot_on_plate`.
- `_bump_version_if_deprecated(env_id) -> str` — Rounds an upstream `-v0` env id up to the highest registered `-v*` suffix; upstream `simpler_env.ENVIRONMENT_MAP` still ships `-v0` but MS3 v3.0.x registers `-v1`. (L111)
- `_resolve_friendly_name(task_name) -> tuple[str, dict[str, Any]]` — Translates a SimplerEnv friendly task name into `(ms3_env_id, kwargs)`. Falls back to passing the input through unchanged so users can author configs against raw MS3 env ids. (L134)
- `_build_simpler_env_scene(env_cfg) -> _SimplerEnvSim` — Calls `gym.make` directly (bypassing the broken upstream `simpler_env.make()` which still passes `prepackaged_config=True` / `obs_mode='rgbd'` that MS3 v3.0.x rejects). (L371)
- Module side effect: `SCENES.register("simpler_env", fixed_robot="widowx", sequential_init=True, sim_clock=True)(_build_simpler_env_scene)` at import (L371).
- `const _VIEW_ENV = 'OPENRAL_SIM_VIEW'` (L68)
- `const _DEPLOY_NOOP_SUFFIX = '/_hal_deploy_noop'` (L72)
- `const _TCP_LINK_CANDIDATES: tuple[str, ...] = ('ee_gripper_link', 'link_ee', 'tcp')` (L155)

#### `python/sim/src/openral_sim/sidecar.py`
_Canonical openral-side out-of-process sidecar transport — ZMQ REQ/REP + a numpy-aware msgpack codec. Shared by new sidecar integrations (e.g. Isaac Sim); the RLDX-1 adapter predates it and keeps its own wire-locked codec, though it now shares this module's `open_req_socket` for REQ-socket setup._
- `encode_ndarray(obj) -> Any` — msgpack `default` hook: serialize ndarrays via `np.save` into a `{"__ndarray__": True, "npy": bytes}` sentinel. (L117)
- `decode_ndarray(obj) -> Any` — msgpack `object_hook`: reverse `encode_ndarray`; a sentinel missing `npy` passes through unchanged rather than raising `KeyError`. (L130)
- `const _NDARRAY_SENTINEL = '__ndarray__'` (L55)
- `require_key(reply, key, *, name) -> Any` — typed-`ROSRuntimeError` guard for a reply missing a required key. (L141)
- `coerce_sim_time_ns(value) -> int | None` — coerce an optional wire `sim_time_ns` (int/float/None) to `int | None`; used by both `backends.isaac_sim` and `backends.robotwin` (formerly two identical private copies) so a HAL backed by either sidecar can publish `/clock`. (L152)
- `open_req_socket(ctx, timeout_ms, host, port, *, old=None) -> Any` — Create (or, given `old`, recreate) a ZMQ REQ socket connected to `host:port`; recreating clears the EFSM lock left by a timed-out `recv()`. Shared by `SidecarClient` and the GR00T-family sidecar adapter's `_init_socket`. (L168)
- `class SidecarClient` — Owns the ZMQ REQ socket + optional child `Popen`. `connect()` pings an existing sidecar or spawns + boot-polls one; `call()` does one REQ/REP round trip, raising `ROSRuntimeError` on a sidecar-side fault; recreates the REQ socket on timeout to clear the EFSM lock. `_spawn` strips `PYTHONPATH`/`VIRTUAL_ENV` from the child env so the parent's site-packages can't shadow the sidecar venv's own. `_boot_failure_error` distinguishes a crashed child, a port bound but mute, and a slow boot so the error points at the right fix. (L192)
  - `SidecarClient.connect() -> None` — Ping an existing sidecar, else spawn one and wait for it to bind. (L228)
  - `SidecarClient.call(endpoint, data=None) -> dict[str, Any]` — One REQ/REP round trip; raise on a sidecar-side or transport fault. (L321)
  - `SidecarClient.require(reply, key) -> Any` — `require_key` bound to this client's `name` for error text. (L341)
  - `SidecarClient.close() -> None` — Idempotent teardown: socket first (so child exit can't race a recv). (L345)
- `class SidecarSimRollout` — Shared `SimRollout` base for sidecar-backed scene adapters (Isaac Sim, RoboTwin): owns the client/state fields and the five reset/step/render/close/sim_time_ns methods. `_wrap_obs` raises `NotImplementedError` — each backend subclass implements it and its own `action_dim`. (L515)
  - `SidecarSimRollout.reset(seed=None) -> Observation` — Reset the sidecar env and return its first wrapped observation. (L536)
  - `SidecarSimRollout.step(action) -> StepResult` — Apply one action over the wire and return the wrapped transition. (L542)
  - `SidecarSimRollout.sim_time_ns() -> int | None` — Elapsed simulation time in ns from the last sidecar reply, or `None`. (L558)
  - `SidecarSimRollout.render() -> NDArray[np.uint8] | None` — Return the last cached RGB frame, or `None` before the first step. (L568)
  - `SidecarSimRollout.close() -> None` — Idempotent teardown: best-effort sidecar-side close, then the client. (L572)

#### `python/sim/src/openral_sim/backends/isaac_sim.py`
_NVIDIA Isaac Sim (Omniverse + PhysX + RTX) free-axis scene adapter. Drives an Isaac Sim env in a separate py3.11 sidecar venv (Isaac ships per-interpreter wheels; the openral workspace is py3.12) over the shared `SidecarClient`. The heavy `isaacsim`/`isaaclab` install is an externally-provisioned venv (Omniverse Kit is proprietary, never vendored). Scene id `isaac_sim`; task id `isaac_sim/<name>`._
- `_ISAAC_SCENE_ID = "isaac_sim"` — module constant; scene-registry key.
- `class _IsaacSimSidecar(SidecarSimRollout)` — `reset`/`step`/`render`/`close`/`sim_time_ns` inherited from `SidecarSimRollout`. Implements `_wrap_obs` for the eval-shaped Observation and caches `action_dim` from the sidecar `ping`. Minimal bring-up only — `/joint_states` reads zeros for this non-MuJoCo backend.
- `_sidecar_python() -> Path` / `_provision_isaac_venv() -> Path` / `_locate_sidecar_script() -> Path` — Resolve the py3.11 interpreter (env override → opt-in auto-provision → cache default → typed `ROSConfigError`) and `tools/isaac_sidecar.py` (env override → walk-up). Auto-provision runs before the existing-venv shortcut so a venv built from superseded pins gets repaired, not silently reused.
- `_ISAAC_CUDA_FLOORS = {"nvidia-nvjitlink-cu12": "12.8", "nvidia-cusparse-cu12": "12.5"}` → `_ISAAC_CUDA_DEPS` — CUDA runtime floors forced on top of the Isaac install, and the single source of both the pip spec and the sidecar's `--require-min` boot probe. Needed because Isaac's bundled CUDA-12.8 `libcusparse` has no matching `libnvJitLink`, and a too-old bundled one hangs Kit past `app ready` instead of raising.
- `_sensor_dict(sensor) -> dict` / `_build_robot_spec(desc, robot_id) -> dict` / `_write_robot_spec(env_cfg) -> str` — Robot-agnostic `--layout manifest` marshalling: since the py3.11 sidecar can't import `openral_core`, `_build_robot_spec` serialises the `RobotDescription` to plain JSON (joints, action contract, sensors); `_write_robot_spec` writes it to a temp file passed via `--robot-spec`.
- `_build_isaac_sim_scene(env_cfg) -> _IsaacSimSidecar` — Factory: builds the launch argv from `backend_options.layout`; for `layout == "manifest"` writes the robot spec, appends `--robot-spec`, and unlinks it after `connect()`. Connects a `SidecarClient(name="isaac", …)`.
- `provision_isaac_sim() -> None` — Pre-launch provisioner: builds the sidecar venv (auto under `OPENRAL_ISAAC_AUTO_PROVISION=1`, else raises the manual recipe). Covers provisioning only — the Omniverse Kit boot still runs inside the HAL's 300s-bounded `on_configure`, and the scene's boot timeout is raised well past that bound. (L513)
- Module side effect: `SCENES.register("isaac_sim", fixed_robot=None, provision=provision_isaac_sim, sim_clock=True)(_build_isaac_sim_scene)` at import.
- `const _ISAAC_SCENE_ID = 'isaac_sim'` (L76)
- `const _AUTO_SPAWN_ENV = 'OPENRAL_ISAAC_AUTO_SPAWN'` (L77)
- `const _SIDECAR_PYTHON_ENV = 'OPENRAL_ISAAC_SIDECAR_PYTHON'` (L78)
- `const _SIDECAR_SCRIPT_ENV = 'OPENRAL_ISAAC_SIDECAR_SCRIPT'` (L79)
- `const _AUTO_PROVISION_ENV = 'OPENRAL_ISAAC_AUTO_PROVISION'` (L80)
- `const _ISAAC_PYTHON = '3.11'` (L90)
- `const _ISAAC_SIDECAR_HOME = Path.home() / '.cache' / 'openral' / 'isaac-sidecar'` (L89)
- `const _ISAAC_DEPS` — Pinned `isaacsim==5.1.0.0` install spec (core/robot/replicator/… subpackages). (L91)
- `const _ISAAC_CUDA_FLOORS = {'nvidia-nvjitlink-cu12': '12.8', 'nvidia-cusparse-cu12': '12.5'}` (L110)
- `const _ISAAC_CUDA_DEPS` — `_ISAAC_CUDA_FLOORS` rendered as `>=floor,<13` pip specs. (L114)
- `const _DEFAULT_HOST = '127.0.0.1'` (L117)
- `const _SIDECAR_PORT_MIN = 20000` (L125)
- `const _SIDECAR_PORT_MAX = 40000` (L126)
- `const _DEFAULT_TIMEOUT_MS = 120000` (L146)
- `const _DEFAULT_BOOT_TIMEOUT_S = 900.0` (L147)
- `const _DEFAULT_MAX_STEPS = 1000000` (L149)
- `const _MAX_PHYSICAL_GRIPPER_TRAVEL_M = 0.1` — Panda finger-travel fallback used when a manifest's normalised `[0,1]` gripper limit is degenerate. (L153)

#### `tools/isaac_sidecar.py` + `tools/_isaac_scene_base.py` + `tools/isaac_scene.py` + `tools/isaac_bowl_plate_scene.py` + `tools/isaac_manifest_scene.py`
_Isaac-side sidecar (runs under the py3.11 Isaac Sim venv only). Verifies its dependency floors before launching the headless Omniverse Kit `SimulationApp`, then serves a ZMQ REP loop (`ping`/`reset`/`step`/`render`/`close`) speaking the same msgpack+ndarray framing as the openral side. `IsaacSceneBase` owns the shared lifecycle; subclasses override `build`/`_apply_action`/`_images`/`_state`/`_reward_terminated`. The `--layout` arg picks the scene class:_
- _`lift_cube` (`isaac_scene.IsaacLiftScene`) — `World` + `Franka` + `DynamicCuboid` + `Camera`; 8-D joint-delta action, cube-height reward. Kept as a deploy/wire PoC only; no SimScene YAML is shipped because the repo has no task-capable 8-D Franka joint-delta rSkill._
- _`bowl_plate` (`isaac_bowl_plate_scene.IsaacBowlPlateScene`) — table + YCB `024_bowl` USD + thin-cylinder plate + `Franka` + agent-view & eye-in-hand cameras, mirroring the **LIBERO** contract (camera1/camera2 + 8-D state, 7-D OSC-pose-delta action). End-effector control uses the core `isaacsim.robot_motion.motion_generation` Lula kinematics solver for position-delta IK — no Isaac Lab required. Drives `act-libero`/`smolvla-libero` through `openral sim run`. Scene `scenes/sim/isaac_franka_bowl_plate.yaml`._
- _`manifest` (`isaac_manifest_scene.IsaacManifestScene`) — robot-agnostic, URDF-driven scene: imports the manifest robot's URDF via Isaac's URDF importer and drives a JOINT_POSITION-delta articulation controller, action layout `[arm deltas, gripper, base twist]`. `map_dof_to_manifest(...)` maps the articulation DOF vector to the full manifest joint order. A robot with `base_joints` gets a kinematic holonomic base: imported `fix_base=True`, with `_integrate_base` teleporting the articulation root from an integrated base-frame twist each step (no PhysX base joints). Manifest-declared sensors get base-relative Isaac cameras (RGB/depth) and a 2-D lidar cast via `resolve_beam_range`, which re-casts a beam that hits the robot's own body so it reports the obstacle behind it instead of failing open. Verified live on `scenes/deploy/isaac_franka_urdf.yaml` and `scenes/deploy/isaac_panda_mobile_urdf.yaml`._

_All three layouts use Isaac Sim core (not Isaac Lab's env machinery, which the PyPI `isaaclab` wheel doesn't ship). Not imported by the openral venv — invoked as a subprocess; `backends/isaac_sim.py` forwards `scene.backend_options.layout` (and, for `manifest`, the `--robot-spec` JSON)._

#### `python/sim/src/openral_sim/backends/robotwin.py`
_RoboTwin 2.0 dual-arm SAPIEN scene adapter, fixed to the `aloha_agilex` embodiment (14-DoF), run in a separate py3.10 sidecar venv (RoboTwin pins SAPIEN/CuRobo/mplib/pytorch3d incompatibly with the openral py3.12 venv) over the shared `SidecarClient`. The heavy SAPIEN+RoboTwin stack is externally-provisioned, never vendored. Scene id `robotwin`; task id `robotwin/<snake_case_task>`._
- `_ROBOTWIN_SCENE_ID = "robotwin"` / `_ROBOTWIN_ROBOT_ID = "aloha_agilex"` — module constants; scene-registry key + fixed robot.
- `class _RoboTwinSimSidecar(SidecarSimRollout)` — `reset`/`step`/`render`/`close`/`sim_time_ns` inherited verbatim from `sidecar.SidecarSimRollout` (shared with `_IsaacSimSidecar`). Implements `_wrap_obs` for the eval-shaped Observation (`images` keyed `camera1/camera2/camera3` / `state` / `task`) via `client.require(...)`, preferring the head frame for the cached render image, and exposes `action_dim` (14, read from the sidecar `ping`, cached).
- `_scene_default_port(task_id, robot_id) -> int` — deterministic per-scene ZMQ port in `[_SIDECAR_PORT_MIN, _SIDECAR_PORT_MAX)` (SHA-256 digest, not the salted builtin `hash`), so distinct tasks never share a sidecar endpoint; an explicit `backend_options.port` still wins.
- `_robotwin_task_name(task_id) -> str` — strips the `robotwin/` namespace to the bare upstream task name the LeRobot env wants.
- `_provision_robotwin_venv() -> Path` / `_sidecar_python() -> Path` / `_locate_sidecar_script() -> Path` — opt-in (`OPENRAL_ROBOTWIN_AUTO_PROVISION=1`) provisioning of the py3.10 venv (lerobot from git `main` + SAPIEN + wire — the RoboTwin task package + multi-GB assets remain a manual step), interpreter resolution (env override → cache default → opt-in provision → typed `ROSConfigError` carrying the full conda recipe), and `tools/robotwin_sidecar.py` location (env override → walk-up).
- `_download_assets_script(root: Path) -> str` (L317) — Path of RoboTwin's asset download script relative to the checkout; accepts the pre-2026 `script/` and current `scripts/` layouts.
- `_build_robotwin_scene(env_cfg) -> _RoboTwinSimSidecar` — factory: builds the launch argv (`--task`, `--cameras`, `--episode-length`, obs h/w, host/port), connects a `SidecarClient(name="robotwin", expected_identity={"env": "robotwin", "task": <name>})`.
- `provision_robotwin() -> None` — Pre-launch provisioner: `ensure_backend_deps("robotwin_client")` + `_sidecar_python()` (opt-in multi-GB LeRobot + SAPIEN venv under `OPENRAL_ROBOTWIN_AUTO_PROVISION=1`, else the manual recipe). Registered as `provision=`; keeps the venv build out of the HAL's 300 s `on_configure`. Note this covers **provisioning only** — the sidecar *boot* still happens inside `on_configure` via `_build_robotwin_scene`'s `connect()`, and this backend's `_DEFAULT_BOOT_TIMEOUT_S = 600.0` exceeds that bound (latent: no in-tree deploy scene selects `robotwin`). (L414)
- Module side effect: `SCENES.register("robotwin", fixed_robot="aloha_agilex", provision=provision_robotwin, sim_clock=True)(_build_robotwin_scene)` at import.
- `const _ROBOTWIN_SCENE_ID = 'robotwin'` (L72)
- `const _ROBOTWIN_ROBOT_ID = 'aloha_agilex'` (L73)
- `const _DEPLOY_NOOP_SUFFIX = '/_hal_deploy_noop'` (L74)
- `const _DEFAULT_DEPLOY_TASK = 'lift_pot'` (L75)
- `const _AUTO_SPAWN_ENV = 'OPENRAL_ROBOTWIN_AUTO_SPAWN'` (L77)
- `const _SIDECAR_PYTHON_ENV = 'OPENRAL_ROBOTWIN_SIDECAR_PYTHON'` (L78)
- `const _SIDECAR_SCRIPT_ENV = 'OPENRAL_ROBOTWIN_SIDECAR_SCRIPT'` (L79)
- `const _ROBOTWIN_ROOT_ENV = 'OPENRAL_ROBOTWIN_ROOT'` (L80)
- `const _AUTO_PROVISION_ENV = 'OPENRAL_ROBOTWIN_AUTO_PROVISION'` (L81)
- `const _ROBOTWIN_SIDECAR_HOME = Path.home() / '.cache' / 'openral' / 'robotwin-sidecar'` (L89)
- `const _ROBOTWIN_PYTHON = '3.12'` (L90)
- `const _ROBOTWIN_REPO = 'https://github.com/RoboTwin-Platform/RoboTwin.git'` (L91)
- `const _ROBOTWIN_BASE_DEPS` — Pinned `lerobot==0.6.0` + `sapien==3.0.3` + CUDA/Open3D/h5py sidecar install spec. (L94)
- `const _ROBOTWIN_MPLIB_DEPS = ('mplib==0.2.1',)` (L108)
- `const _DEFAULT_HOST = '127.0.0.1'` (L110)
- `const _SIDECAR_PORT_MIN = 20000` (L116)
- `const _SIDECAR_PORT_MAX = 40000` (L117)
- `const _DEFAULT_TIMEOUT_MS = 120000` (L121)
- `const _DEFAULT_BOOT_TIMEOUT_S = 600.0` (L123)
- `const _DEFAULT_MAX_STEPS = 1000000` (L125)
- `const _FALLBACK_SIM_DT_S = 1.0 / 30.0` (L129)

#### `tools/robotwin_sidecar.py`
_RoboTwin-side sidecar (runs under the py3.10 lerobot-main + RoboTwin + SAPIEN venv only). Constructs LeRobot's native `robotwin` gym env and serves a ZMQ REP loop matching the openral wire framing, re-keying the env's native cameras to `camera1`/`camera2`/`camera3` and including `sim_time_ns`. Not imported by the openral venv._
- `main(argv) -> int` — argparse (`--task/--cameras/--episode-length/--obs-height/--obs-width/--host/--port`) then serves the ZMQ REP loop. (L339)

#### `python/sim/src/openral_sim/backends/rlbench.py`
_RLBench (CoppeliaSim/PyRep) single-robot (fixed `franka_panda`) scene adapter, run in an externally-provisioned py3.10 sidecar venv (CoppeliaSim is proprietary, free-EDU, never vendored) over the shared `SidecarClient`. Opt-in via the `rlbench` dependency group. Scene id `rlbench`; `step` takes an 8-D keyframe `[x y z qx qy qz qw gripper_open]`._
- `_RLBENCH_SCENE_ID = "rlbench"`; `_scene_default_port(rlbench_task, variation)` — deterministic per-task ZMQ port (SHA-256, range 21000–21999).
- `_RLBenchSidecar(scene, task, _client)` — `SimRollout` proxy; `_wrap_obs` carries `images`/`point_clouds` (dict per camera) + `gripper_pose`/`gripper_open` for the 3D keyframe policy. Caches `sim_time_ns` from reset/step replies and exposes `sim_time_ns()` for clock projection.
- `_build_rlbench_scene(env_cfg) -> _RLBenchSidecar` — factory; reads `backend_options.{rlbench_task,variation,port,max_tries}`. Connects a `SidecarClient(name="rlbench", …)`.
- `provision_rlbench() -> None` — Pre-launch provisioner: `ensure_backend_deps("rlbench_client")` + `_sidecar_python()`. CoppeliaSim is proprietary with no auto-install plan, so this only locates the externally-provisioned venv or raises the manual recipe — surfacing that refusal during preflight instead of as an opaque 300 s `on_configure` timeout is the point. (L262)
- Module side effect: `SCENES.register("rlbench", fixed_robot="franka_panda", provision=provision_rlbench, sim_clock=True)(_build_rlbench_scene)` at import.
- `const _RLBENCH_SCENE_ID = 'rlbench'` (L57)
- `const _SIDECAR_PYTHON_ENV = 'OPENRAL_RLBENCH_SIDECAR_PYTHON'` (L58)
- `const _SIDECAR_SCRIPT_ENV = 'OPENRAL_RLBENCH_SIDECAR_SCRIPT'` (L59)
- `const _COPPELIASIM_ROOT_ENV = 'COPPELIASIM_ROOT'` (L60)
- `const _AUTO_SPAWN_ENV = 'OPENRAL_RLBENCH_AUTO_SPAWN'` (L61)
- `const _RLBENCH_VENV = Path.home() / '.cache' / 'openral' / 'rlbench-policy' / '.venv'` (L64)
- `const _COPPELIASIM_DEFAULT` — Default CoppeliaSim Edu install path under the openral cache. (L65)
- `const _PORT_MIN = 21000` (L70)
- `const _PORT_MAX = 21999` (L71)
- `const _DEFAULT_TIMEOUT_MS = 120000` (L74)
- `const _DEFAULT_BOOT_TIMEOUT_S = 300.0` (L75)
- `const _VIDEO_FRAMES_INFO_KEY = '_openral_video_frames'` (L76)
- `const _RENDER_CAMERA = 'front'` (L77)

#### `tools/rlbench_sidecar.py`
_RLBench-side scene sidecar (externally-provisioned py3.10 venv only; no openral import). Launches CoppeliaSim/PyRep headless via RLBench's peract-fork `Environment`, serves the same ZMQ REP loop as other sidecars, and executes each keyframe via a plan-and-retry mover. Cameras: left_shoulder/right_shoulder/wrist/front (RGB + point cloud)._
- `main(argv) -> int` — argparse + boot the CoppeliaSim/PyRep `Environment` and serve the ZMQ REP loop. (L397)

#### `python/sim/src/openral_sim/backends/aloha.py`
_gym-aloha bimanual MuJoCo scene adapter. Opt-in via the `sim` dependency group (gym-aloha lives there alongside mujoco/gymnasium/lerobot); auto-installs on first use via `ensure_backend_deps("aloha")`._
- `class _AlohaSim` — `SimRollout` wrapping a `gym_aloha` env. — `reset/step/render/close/mujoco_handles/sim_time_ns/_wrap_obs`. `mujoco_handles()` reaches through `env.unwrapped._env.physics.{model,data}.ptr` (dm_control wrapper) for `openral sim run --view`. `sim_time_ns()` returns `round(MjData.time * 1e9)`.
- `const _ALOHA_SCENES: dict[str, str]` — Scene id → `gym_aloha` env id (`aloha_transfer_cube` / `aloha_insertion`). (L37)

#### `python/sim/src/openral_sim/backends/pusht.py`
- `class _PushTSim` — `SimRollout` wrapping `gym_pusht/PushT-v0` (pymunk 2-D rigid body). — `reset/step/render/close/enable_intrinsic_viewer/_wrap_obs/_paint_view`. `enable_intrinsic_viewer()` opens a pygame window from inside the adapter and `_paint_view()` blits the last pixel frame on each `reset` / `step`, leaving the env in `render_mode="rgb_array"` so the Diffusion Policy still gets `observation.image`.
- `const _PUSHT_SCENE_ID = 'pusht'` (L30)
- `const _PUSHT_ENV_ID = 'gym_pusht/PushT-v0'` (L31)

#### `python/sim/src/openral_sim/backends/so100_robosuite/`
_robosuite integration for the Hugging Face SO-100 follower. Not a `SCENES.register(...)` adapter (no benchmarked VLA-driven suite yet) — a standalone subpackage that registers the SO-100 with robosuite's robot/gripper factories and provides a runnable scripted-pick demo. Used by `tests/sim/test_so100_robosuite_lift.py`._
- `__init__.py` — re-exports `SO100`, `SO100Gripper`, `make_so100_lift_env`, `so100_osc_controller_config`; importing the package side-effect-registers the robot + gripper.
- `_assets.py` — `ensure_so100_assets() -> SO100Assets` lazily rewrites the DeepMind `mujoco_menagerie` `trs_so_arm100` MJCF into two robosuite-compatible XMLs (arm with 5 motor actuators + base/right_hand body, gripper with the Jaw joint + `eef` body + finger pads), caches under `$OPENRAL_CACHE_DIR/so100_robosuite/<menagerie-fingerprint>/`. `class SO100Assets(frozen dataclass)` carries `robot_xml`, `gripper_xml`, `menagerie_dir`. The rewrite handles three robosuite quirks: nested `<default>` flattening (so `_replace_defaults_inline` resolves classes), absolute mesh paths (robosuite's `resolve_asset_dependency` ignores `meshdir`), and `childclass` stripping (robosuite drops the defaults block before MuJoCo compiles).
  - `ensure_so100_assets() -> SO100Assets` (`_assets.py` L583)
  - `class SO100Assets` (`_assets.py` L48)
  - `const _GRIPPER_JOINT: str = 'Jaw'` (`_assets.py` L43)
- `model.py`:
  - `class SO100(ManipulatorModel)` — 5-DOF arm (Rotation / Pitch / Elbow / Wrist_Pitch / Wrist_Roll); `default_base = "NullMount"`, `default_gripper = {"right": "SO100Gripper"}`, init_qpos matches the menagerie `home` keyframe. Registered in `REGISTERED_ROBOTS` and `ROBOT_CLASS_MAPPING` (as a `FixedBaseRobot`) at import. (`model.py` L106)
    - `SO100.default_base -> str` [@property] — No physical mount — the SO-100 bolts directly to the desk. (`model.py` L125)
    - `SO100.default_gripper -> dict[str, str]` [@property] (`model.py` L130)
    - `SO100.default_controller_config -> dict[str, str]` [@property] (`model.py` L134)
    - `SO100.init_qpos -> NDArray[np.float64]` [@property] — Home pose — arm folded compactly, tip near the table. (`model.py` L141)
    - `SO100.base_xpos_offset -> dict[str, object]` [@property] — Robot base offsets per arena (world coords). (`model.py` L150)
    - `SO100.top_offset -> NDArray[np.float64]` [@property] (`model.py` L171)
    - `SO100.arm_type -> str` [@property] (`model.py` L179)
  - `class SO100Gripper(GripperModel)` — 1-DOF Jaw with `_important_geoms` for `left_fingerpad` / `right_fingerpad` so robosuite's `_check_grasp` resolves cleanly. Registered in `GRIPPER_MAPPING` at import. (`model.py` L30)
    - `SO100Gripper.format_action(action) -> NDArray[np.float32]` — Pass through the single scalar gripper command. (`model.py` L43)
    - `SO100Gripper.init_qpos -> NDArray[np.float64]` [@property] — Jaw starts wide open (~0.05 rad past the lower limit). (`model.py` L56)
    - `SO100Gripper.speed -> float` [@property] (`model.py` L61)
    - `SO100Gripper.dof -> int` [@property] (`model.py` L65)
- `env.py`:
  - `class _So100Lift(Lift)` — `Lift` with the SO-100 bolted onto the standard `TableArena` top, a small upright redwood block (1.2 cm half-edge × 4 cm tall) sized for the SO-100 jaw aperture, and a `_check_success` that scales with the block height (success when the bottom face clears the table by `lift_height_m`).
  - `so100_osc_controller_config() -> dict[str, Any]` — Loads robosuite's shipped OSC-position config, narrows `output_max` and raises `kp` for the SO-100's much smaller mass matrix, and pins `input_ref_frame="world"` so world-frame Cartesian targets aren't re-rotated by the SO-100's 90°-z base orientation. (`env.py` L163)
  - `make_so100_lift_env(*, has_renderer, has_offscreen_renderer, use_camera_obs, camera_names, camera_heights, camera_widths, horizon, control_freq, table_full_size, cube_half_extent_m, cube_block_height_m, x_range, y_range, seed, lift_height_m, reward_shaping) -> _So100Lift` — composes the registered robot + gripper + `OSC_POSITION` config; cube placement reference matches the Panda-default `table_offset = (0, 0, 0.8)` so the stock `agentview` / `frontview` cameras frame the scene correctly. (`env.py` L208)
  - `const _DEFAULT_LIFT_HEIGHT_M = 0.02` (`env.py` L51)
- `policy.py`:
  - `class PolicyTelemetry` (dataclass) — Per-step diagnostics: `phase / eef_to_cube_distance_m / gripper_command / cartesian_delta / cube_height_m`. (`policy.py` L41)
  - `class ScriptedPickPolicy` (dataclass) — Four-phase Cartesian state machine (approach → descend → close → lift); `step()` returns a 4-vec `[dx, dy, dz, gripper]` normalised to [-1, 1]. OSC owns the IK — the policy just clips a Cartesian delta toward the latched initial cube pose. Gripper convention: positive opens, negative closes. (`policy.py` L66)
    - `ScriptedPickPolicy.reset() -> None` — Re-arm the state machine for a fresh episode. (`policy.py` L99)
    - `ScriptedPickPolicy.step(env, obs) -> tuple[NDArray[np.float64], PolicyTelemetry]` — Produce one `env.action_dim` command + telemetry. (`policy.py` L106)

#### `python/sim/src/openral_sim/backends/openarm_robosuite/`
_Custom MJCF composer + `SCENES.register("openarm_tabletop_pnp")` adapter for the bimanual OpenArm v2 pick-and-place scene. Rewrites the vendor MJCF in-place to add scene bodies, motor actuators, and a camera; state/action dims and the actuator inventory derive from `RobotDescription.sim`. Opt-in via the `robocasa` dependency group (the workspace's only `robosuite>=1.5` declaration; used here purely as an MJCF wrapper, not the kitchen/GR1 forks)._
- `_assets.py`:
  - `load_openarm_description() -> RobotDescription` — Resolves the canonical `robots/openarm/robot.yaml` manifest as the single source of truth for actuator metadata. (_assets.py L65)
  - `actuator_specs_from_description(desc) -> list[ActuatorSpec]` — Build the ordered list of MJCF actuator specs (name, joint, ctrlrange, gear, side) from `desc.joints` + `desc.sim.grippers`. Mirrors the OpenArm v2 actuator block but stays robot-data-driven so adding a new joint in `robot.yaml` is enough — no edit here. (_assets.py L104)
  - `motor_actuator_names_from_description(desc) -> list[str]` — Convenience wrapper returning just the actuator names in MJCF order; used by the env's action wiring. Replaces the removed `MOTOR_ACTUATOR_NAMES` module-level tuple. (_assets.py L151)
  - `_render_actuator_block(specs) -> str` — Render an `<actuator>` XML block from the spec list (motor actuators with per-actuator ctrlrange + gear). (_assets.py L242)
  - `compose_openarm_tabletop_mjcf(env_cfg) -> str` — Compose the scene MJCF: pulls the v2 bimanual MJCF, substitutes position actuators with the motor block from `_render_actuator_block`, injects scene bodies (table / target / base sites) + the top camera, and lifts the robot bases. (_assets.py L402)
  - Module constants `_FALLBACK_TOP_CAMERA_POS / _FALLBACK_TOP_CAMERA_TARGET / _FALLBACK_TOP_CAMERA_FOVY` (L251–L253) — Fallbacks used only when neither `RobotDescription.scene_defaults.top_camera` nor `scene.backend_options.top_camera_*` is set; the canonical defaults live on the robot manifest (`SceneDefaults`/`TopCameraDefaults`).
  - Private helpers: `_look_at_quat`, `_inject_base_center_sites`, `_lift_robot_bases`, `_rename_upstream_wrist_cameras`, `_inject_white_skybox`, `_strip_position_actuators`.
  - `const _SCENE_BODIES` (`_assets.py` L156)
  - `const _FALLBACK_TOP_CAMERA_TARGET: tuple[float, float, float] = (0.65, 0.0, 0.05)` (`_assets.py` L232)
  - `const _FALLBACK_TOP_CAMERA_FOVY: float = 65.0` (`_assets.py` L233)
  - `const _BASE_CENTER_SITES` (`_assets.py` L258)
  - `const _WHITE_SKYBOX_ASSET` (`_assets.py` L330)
- `env.py`:
  - `_resolve_state_dim(env_cfg, rskill_manifest=None) -> int` — Derive the observation-state dim from the rSkill manifest's `state_contract.dim` when present, else from `OPENARM_DESCRIPTION.observation_spec.state_shape`. Replaces the removed `_OBS_STATE_DIM` module-level constant so the scene is self-consistent when the rSkill ships a different state contract. (env.py L185)
  - `_resolve_initial_pose_from_rskill(rskill_manifest)` — Read the per-rSkill initial joint pose if the manifest pins one. (env.py L254)
  - `_resolve_base_translation(env_cfg) -> tuple[float, float]` — Parse the scene's `base_translation` override; defaults to the OpenArm tabletop layout. (env.py L137)
  - `class _ArmHandles` / `_build_arm_handles(model, side) -> _ArmHandles` — Per-side MuJoCo qpos/qvel/actuator handles. (env.py L312, L368)
  - `class _OpenArmTabletopRollout` — `SimRollout` for the openarm_tabletop_pnp scene: composes the MJCF via `_build_openarm_tabletop_scene`, drives both arms through the manifest-derived actuator block, exposes `mujoco_handles()` + `sim_time_ns()` (`round(MjData.time * 1e9)`) for the viewer / sim-clock, and `action_dim` (== bimanual `state_dim`) so `SimAttachedHAL._probe_env_action_dim` resolves the deploy-sim action width (probe-gap fix). (env.py L433)
  - `_build_openarm_tabletop_scene(env_cfg) -> _OpenArmTabletopRollout` — Scene factory registered as `SCENES.register("openarm_tabletop_pnp", fixed_robot="openarm", sequential_init=True, sim_clock=True, base_pose=True)(_build_openarm_tabletop_scene)`. (env.py L762)
  - `const _ARM_JOINT_COUNT = 7` (`env.py` L50)
  - `const _GRIPPER_CTRL_COUNT = 1` (`env.py` L51)
  - `const _ACTION_PER_ARM = 7` (`env.py` L52)
  - `const _DOF_PER_ARM = _ARM_JOINT_COUNT + _GRIPPER_CTRL_COUNT` (`env.py` L60)
  - `const _DEFAULT_RENDER_WIDTH = 256` (`env.py` L62)
  - `const _DEFAULT_RENDER_HEIGHT = 256` (`env.py` L63)
  - `const _GRIPPER_ENCODER_DEADBAND = 0.05` (`env.py` L64)
  - `const _WRIST_CAM_LOCAL_POS = np.asarray([0.12, 0.0, -0.06])` (`env.py` L79)
  - `const _WRIST_CAM_LOCAL_QUAT_WXYZ = np.asarray([0.632177, -0.316784, 0.316784, -0.632177])` (`env.py` L80)
  - `const _WRIST_CAMERA_FOVY = 80.0` (`env.py` L83)
  - `const _IDENTITY_QUAT_TOL: float = 1e-06` (`env.py` L117)
  - `const _SCENE_ID = 'openarm_tabletop_pnp'` (`env.py` L889)

#### `python/sim/src/openral_sim/backends/so101_box/`
_Parameterised raw-MuJoCo scene: SO-101 in a configurable box arena, registered as `@SCENES.register("so101_box", fixed_robot="so101_follower")`. Every dimension (arena, cameras, block/tube geometry, spawn ranges, insertion thresholds) is driven by `scene.backend_options` via a typed `BoxSceneOptions` — no scene geometry is hard-coded in Python. Each `reset()` randomises block + tube placement; success fires when the tube is inserted into the block hole within tolerance._
- `_assets.py`:
  - `class BoxSceneOptions` — Typed dataclass holding every scene-geometry knob (arena, robot mount, two cameras, block + tube dimensions, spawn ranges, insertion thresholds, `control_hz`, LeRobot joint-units affine). Fed from `scene.backend_options` by `_options_from_backend_options`. (_assets.py L43)
  - `compose_so101_box_mjcf(options=None, robot_description=None) -> tuple[str, Path]` — Reads the robot's MJCF, re-anchors its `base` body to `options.robot_base_xyz`, splices a wrist camera into the `gripper` body, and appends the arena/walls/light/overhead camera/block/tube to the worldbody. The `base`/`gripper` anchors are fixed MJCF body names, so `deploy sim` composes the twin off the robot's own MJCF. (_assets.py L530)
  - Private helpers: `_resolve_so101_mjcf`, `_look_at_quat`, `_reanchor_robot_base`, `_splice_wrist_camera`, `_render_arena_geoms`, `_render_overhead_camera`, `_render_slot_block` (5-box decomposition with a square hole + slot), `_render_tube` (cylinder + two end-tip sites).
  - `const _UNSET: tuple[float, ...] = ()` — Sentinel default distinguishing "not overridden" from an explicit empty range in `BoxSceneOptions`. (`_assets.py` L39)
- `env.py`:
  - `_options_from_backend_options(raw) -> BoxSceneOptions` — Validate + parse `scene.backend_options` into a `BoxSceneOptions`; rejects unknown keys loudly so YAML typos surface immediately. (env.py L72)
  - `class _So101BoxRollout` — `SimRollout` driving the SO-101's 6 position actuators, two MuJoCo renderers (RGB + depth on the same overhead camera), random spawn each reset, and the geometric insertion success check. Each `step()` advances a full control period of physics (from `options.control_hz`); `joint_units: degrees` mode uses the shared `_so_arm_units` conversions in both directions — a 30 FPS-trained checkpoint needs a full control period per action, not one physics tick. Exposes `mujoco_handles()`/`sim_time_ns()`/`action_dim` (== 6) for the viewer, sim clock, and `SimAttachedHAL`. (env.py L160)
  - `build_so101_box_scene(env_cfg) -> _So101BoxRollout` — Scene factory registered as `SCENES.register("so101_box", fixed_robot="so101_follower", sim_clock=True)(build_so101_box_scene)`. Composes the MJCF, resolves the 6 arm actuators by their upstream numeric names (`"1"`…`"6"`), and caches the block / tube / hole / tip site indices for the success check. (env.py L563)
  - `const _SO101_ARM_DOF = 6` (`env.py` L63)
  - `const _DEFAULT_MAX_STEPS = 500` (`env.py` L64)
  - `const _DEFAULT_RENDER_HEIGHT = 480` (`env.py` L65)
  - `const _DEFAULT_RENDER_WIDTH = 640` (`env.py` L66)
  - `const _XYZ_LEN = 3` (`env.py` L68)
  - `const _RANGE_PAIR_LEN = 2` (`env.py` L69)

#### `python/sim/src/openral_sim/backends/_so_arm_units.py`
_Shared unit + cadence conversions for the raw-MuJoCo SO-ARM bench scenes (`so101_eraser`, `so101_box`) — one home for the two conventions LeRobot-trained SO-101 checkpoints impose, so a fix in one scene can never miss the other._
- `steps_per_control_period(timestep_s, control_hz, *, scene) -> int` — Physics steps in one control period: `round(1 / (control_hz * timestep_s))`, at least 1; raises `ROSConfigError` on non-positive `control_hz`. One policy action must cover a full control period, not a single physics tick, or a 30 FPS-trained checkpoint's targets never have time to be reached. (L29)
- `lerobot_action_to_radians(action, *, joint_signs, joint_offsets_deg, gripper_range) -> NDArray[float64]` — LeRobot degrees-mode action → MuJoCo radian ctrl targets: arm channels invert the calibration affine then convert to radians. The last (gripper) channel is NOT degrees — LeRobot SO-ARM datasets store it normalised `[0, 100]` over jaw travel, so it maps that fraction onto the jaw's radian range. (L48)
- `radians_to_lerobot_state(qpos, *, joint_signs, joint_offsets_deg, gripper_range) -> NDArray[float64]` — Inverse map for proprio: arm qpos → LeRobot servo degrees via the affine, gripper qpos → normalised `[0, 100]` (a plain `degrees(qpos)` would report the closed jaw as ≈ -8, below the checkpoint normalizer's observed minimum). (L67)

#### `python/sim/src/openral_sim/backends/tabletop_push/`
_Robot-agnostic native scene: push-cube-to-goal on a configurable tabletop, registered free-axis (no `fixed_robot`). `env_cfg.robot_id` resolves a `RobotDescription` whose MJCF gets the table/cube/goal/cameras appended and its root body re-anchored — no robot-specific scene code needed. Success is geometric (cube within `goal_radius` of the goal, still on the table), so it makes no gripper/end-effector assumption. Action/state dim = the compiled model's actuator count `nu`._
- `_assets.py`:
  - `class TabletopOptions` — Typed dataclass holding every scene-geometry knob (table slab, robot-mount fallback, cube, goal disc + radius, two world cameras, opt-in wrist camera, settle steps, optional reset joint pose, scaled joint-unit affine, lighting). Fed from `scene.backend_options` by `_options_from_backend_options`. (_assets.py L52)
  - `compose_tabletop_mjcf(description, options=None, *, base_pose=None) -> mujoco.MjModel` — Resolves the robot MJCF from the manifest, re-anchors the robot root body to `base_pose` (or the yaw-only `robot_base_xyz` fallback), appends the table/cube/goal/cameras/light, and compiles. Robot-agnostic — no body-name regex. (_assets.py L227)
  - Private helpers: `_resolve_robot_mjcf`, `_base_pos_quat`, `_append_table`, `_append_cube`, `_append_goal_marker`, `_append_world_cameras`, `_append_overhead_light`, `_append_wrist_camera`, `_look_at_quat`.
  - `infer_wrist_camera_mount_body(description) -> str | None` — Return the wrist camera's MJCF parent body from the robot manifest. (`_assets.py` L190)
- `env.py`:
  - `_options_from_backend_options(raw) -> TabletopOptions` — Validate + parse `scene.backend_options` into a `TabletopOptions`; rejects unknown keys loudly. (env.py L63)
  - `class _TabletopPushRollout` — `SimRollout` driving the robot's `nu` actuators by index (clipped to each transmission joint's range), rendering world cameras, randomising cube + goal each reset, and running the robot-agnostic on-goal success check. Exposes `mujoco_handles()`/`sim_time_ns()`/`action_dim` (== `nu`) for the viewer, sim clock, and `SimAttachedHAL`. (env.py L175)
  - `build_tabletop_push_scene(env_cfg) -> _TabletopPushRollout` — Scene factory registered as `SCENES.register("tabletop_push", sequential_init=True, sim_clock=True, base_pose=True)(build_tabletop_push_scene)` (free-axis). Composes the model, resolves the robot's actuator→joint transmissions for state read + action clipping, and caches the cube body/freejoint + goal site indices. Raises `ROSConfigError` when `robot_id` has no registered manifest. (env.py L433)
  - `const _DEFAULT_MAX_STEPS = 300` (`env.py` L54)
  - `const _DEFAULT_RENDER_HEIGHT = 480` (`env.py` L55)
  - `const _DEFAULT_RENDER_WIDTH = 640` (`env.py` L56)
  - `const _XY_LEN = 2` (`env.py` L57)
  - `const _XYZ_LEN = 3` (`env.py` L58)
  - `const _SPAWN_ATTEMPTS = 32` (`env.py` L60)
  - `const _RANGE_PAIR_LEN = 2` (`env.py` L59)

#### `python/sim/src/openral_sim/policies/robots.py`
_Registers every `robots/<id>/robot.yaml` as an `openral_sim.ROBOTS` factory at import._
- `resolve_robot_manifest(robot_id: str, *, repo_root: Path | None = None) -> Path` — The one robot-manifest lookup: `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml` when it exists, else `<repo_root>/robots/<id>/robot.yaml` (`repo_root=None` walks up from the module). Shared by the `ROBOTS` factories (`sim run`), `openral deploy sim|run` and `tools/audit_sim_configs.py`. Raises `ROSConfigError` when neither exists. (L87)

#### `python/sim/src/openral_sim/policies/_video_capture.py`
_Tiny shared utility for VLA adapters to record what they fed the policy — the *post-processing* image (channel reorder / normalize / resize / crop / optional 180° flip), not the raw env frame, so a debug video can show what the policy actually saw (a flipped wrist camera, a bad crop). One-liner shared by every visuomotor adapter (SmolVLA, ACT, π0.5, xVLA, Diffusion Policy, …) instead of each reimplementing uint8 conversion / flip._

- `to_input_frame(image, *, flip_180=False) -> NDArray[np.uint8] | None` — Convert an adapter input image to an HWC uint8 RGB frame for video; `None` in, `None` out. (L24)
- `tile_input_frames(images) -> NDArray[np.uint8] | None` — Compose multiple adapter input images into one debug-preview frame: unchanged for one camera, a padded near-square grid for several. (L53)

#### `python/sim/src/openral_sim/policies/mock.py`
- `class _MockSim` — Tiny gym-like env for tests. (L32)
- `class _ZeroPolicy` — Always emits zero-vector actions. (L128)
- `class _RandomPolicy` — Fixed-seed Gaussian samples. (L147)
- `_coerce_int(value, default) -> int` (L89)
- `_build_mock_scene(env_cfg) -> _MockSim` (L101)
- `_resolve_action_dim(env_cfg) -> int | None` — Explicit mock width from `vla.extra` / `backend_options.action_dim`, else `None` (sized to the built env by `SimRunner` via `env_action_dim`; no per-scene table). (L170)
- `_require_dim(action_dim) -> int` — `ROSConfigError` when a mock policy steps without a width. (L117)
- `_build_zero_policy(env_cfg) -> _ZeroPolicy` (L188)
- `_build_random_policy(env_cfg) -> _RandomPolicy` (L201)

#### `python/sim/src/openral_sim/policies/smolvla.py`
- **Chunk-executor wiring (all chunked adapters)** — policy factories assign `build_chunk_executor(...)` to their adapter (SmolVLA/xVLA via `predict_action_chunk`; pi05/GR00T/MolmoAct2/OpenVLA via family-specific producers). Diffusion Policy is excluded (needs full observation history each tick); ZMQ sidecars stay synchronous (REQ sockets can't share a prefetch thread). **Real-Time Chunking**: an enabled `policy_extras.rtc` block (smolvla + pi05 only, requires `chunk_prefetch`) installs lerobot's `RTCProcessor` to blend prefetched chunks; the producer must accept the `inference_delay`/`prev_chunk_left_over` kwargs.
- `class _SmolVLAAdapter` (L182) — Lerobot-style policy adapter driving the chunk executor (per-step `run_inference` when `n_action_steps <= 1`). NVMM `image_handles` are encoded on-device via `NvmmVisionEncoder` and raise `ROSRuntimeError` without a TRT runtime, never falling back to CPU pixels; only image inputs are cast to bf16, `observation.state` stays float32.
- `_build_smolvla(env_cfg) -> _SmolVLAAdapter` — Resolves device + rSkill manifest, loads the lerobot `PolicyProcessorPipeline` per-file from the manifest (no `snapshot_download`). `torch.compile` is skipped when RTC is enabled in `vla.extra` (RTC rewrites the same forward compile would capture). If processor files 404 (common on community finetunes), falls back to rebuilding processors from the dataset's normalization stats; raises `ROSConfigError` if neither is available. (L562)
- `_smolvla_phase(name, **fields) -> ContextManager[None]` — Adapter-local shortcut for `phase_timer(name, prefix="smolvla", log=_log)`. (L163)
- `_is_processor_missing(exc: BaseException) -> bool` — Walks `__cause__` / `__context__` looking for a HF Hub `RemoteEntryNotFoundError` / `EntryNotFoundError`. Detects 404s that `materialize_processor_dir` re-raised as `ROSConfigError`; stays decoupled from HF Hub's exception module path. (L66)
- `_load_lerobot_dataset_stats(dataset_uri: str) -> dict[str, dict[str, Any]]` — Aggregates per-feature normalization stats from a LeRobotDataset on HF Hub: tries v3's single `stats.json` first, falling back to aggregating v2.1's per-episode stats. Returns a dict suitable for `make_pre_post_processors(..., dataset_stats=...)`. (L81)

#### `python/sim/src/openral_sim/policies/openvla.py`
- `class _OpenVLAAdapter` — Transformers custom-code OpenVLA/OpenVLA-OFT policy adapter. Calls `predict_action` or RLinf's `generate_action_verl`, normalizes the result to `(chunk, action_dim)`, applies optional postprocess (`action_scale`, gripper threshold), and replays from an internal queue. `reset()` reapplies `openvla_torch_seed` after SimRunner's per-episode RNG seeding so stochastic generation stays reproducible.
- `_build_openvla(env_cfg) -> _OpenVLAAdapter` (`@POLICIES.register("openvla")`) — Requires `OPENRAL_ALLOW_REMOTE_CODE=1`; loads the HF repo via `AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True)` with NF4/device-map placement on CUDA, patches `_unnormalize_actions` to move CUDA tensors to CPU before NumPy, and resolves cameras/generation-method/`policy_extras` from the manifest.
- Pure helper surface: `_decode_prompt(instruction) -> str`, `_unnormalize_action(norm, stats) -> np.ndarray`, `_as_action_chunk(arr, action_dim) -> np.ndarray`, `_postprocess_action_chunk(arr, *, action_scale, binarize_gripper, gripper_threshold) -> np.ndarray`. Unit-tested without GPU; the opt-in sim test covers the real RLinf checkpoint on SimplerEnv WidowX.

#### `python/sim/src/openral_sim/policies/xr1.py`
_Xiaomi Robotics XR-1 / MiBoT adapter. The openral process owns history, state-layout conversion, action replay, and the shared sidecar wire; manifest-selected bitsandbytes NF4 runs inside the torch 2.9.1 / transformers 4.57.1 / FlashAttention custom-code sidecar._
- `class _XR1Adapter` — `PolicyAdapter` for `robocasa_mg`, `robocasa365`, and `vlabench_choice`; requires `OPENRAL_ALLOW_REMOTE_CODE=1`, validates three cameras and per-profile state/action dimensions, keeps the RC365 interval-two history, and replays the upstream cadence.
- `_rc365_state(state) -> NDArray[np.float32]` — 16-D OpenRAL quaternion layout → 14-D XR-1 axis-angle layout.
- `_history_sample(values) -> NDArray[np.float32]` — left-pad a seven-frame history and select offsets `[-6,-4,-2,0]`.
- `_vlabench_targets(deltas, state) -> NDArray[np.float32]` — integrate XR-1's VLABench deltas into absolute env targets with wrapped Euler angles.
- `_quantization_mode(spec, manifest) -> str` — shared `resolve_quant_plan` → `sidecar_quant_token` (`nf4`, or `prequantized_nf4` when `policy_extras.prequantized_nf4`; `bf16` → the unquantized `none` path); any other token raises `ROSConfigError`.
- The module docstring carries the local persistent-NF4 export command. A manifest with `policy_extras.prequantized_nf4: true` selects `prequantized_nf4`, which reloads Transformers-native packed shards directly.

#### `python/sim/src/openral_sim/policies/act.py`
- `class _ACTAdapter` — ACT policy adapter. Manifest-first `image_preprocessing` resolution (`_cam_alias` + `_image_input_template` + `_flip_images_180`) lets LIBERO `camera1` / `camera2` feed an ACT checkpoint whose input features are `observation.images.image` / `observation.images.image2`; legacy `_state_mean` / `_action_std` path stays for `act-aloha`-style checkpoints with norm stats in `model.safetensors`.
- `_build_act(env_cfg) -> _ACTAdapter` — Loads `ACTPolicy.from_pretrained` after sanitizing `config.json`. Dispatches on `manifest.processors`: modern checkpoints (e.g. `act-libero`) materialize and compose lerobot processor pipelines, overriding a baked-in `device: mps` so CUDA hosts don't crash; legacy checkpoints (`act-aloha`) read norm stats straight from `model.safetensors`. Resolves camera keys/state dim/image aliases the same way smolvla does.
- `_load_manifest_for_spec(spec) -> RSkillManifest | None` — Loads the rSkill manifest from a skill reference in `spec.weights_uri`; returns `None` when the URI is not resolvable to a manifest. Also imported by `backends.libero` (formerly a duplicate copy there).
- `_sanitize_act_config_json(snapshot_dir) -> None` — Drops `ACTConfig` fields the installed lerobot version doesn't accept (e.g. `n_state_dim` on training-fork checkpoints). Mutates `config.json` in-place, no-op when there's nothing to strip.
- `_apply_temporal_ensemble(policy, spec_extra) -> float | None` — Sets `policy.config.temporal_ensemble_coeff` and builds the missing `ACTTemporalEnsembler` (lerobot only attaches one when the coeff is non-None at `__init__` time).
- `_try_load_act_norm_stats(repo_id, device, torch, cam_keys) -> dict` — Pulls `normalize_*` / `unnormalize_*` tensors from `model.safetensors` for the legacy `act-aloha`-shaped checkpoints.

#### `python/sim/src/openral_sim/_quantization.py`
_Shared bitsandbytes NF4 quantization helpers + prequantized-state-dict fast path + manifest-driven dtype resolution. Family-agnostic — the same primitives serve pi05 today and any future bnb-quantized backbone (pi0.6, smolvla-large). All helpers defer torch / bitsandbytes imports so installing `openral-sim` does not pull them transitively._

- `DEFAULT_MIN_PARAMS_TO_QUANTIZE: int = 4_000_000` — Per-Linear weight-element threshold for the nf4 rewrite. PaliGemma / SmolVLA paper default. (L61)
- `const _BNB_META_SUFFIXES` — bitsandbytes per-tensor meta-state suffixes (`.absmax`, `.quant_map`, `.nested_absmax`, …) stripped when reconciling a prequantized state dict against the live module tree. (L285)
- `const _PACKING_TOKENS: frozenset[str] = frozenset({'nf4', 'int8'})` — The two tokens for which `QuantPlan.quantize` is `True`. (L662)
- `const _QUANT_TOKEN_ALIASES: dict[str, str]` — Dtype-spelling normalization table consumed by `canonical_quant_token` (`int4`/`4bit` → `nf4`, float aliases fold, …). (L669)
- `quantize_nf4_in_place(policy, *, torch, compute_dtype, min_params=DEFAULT_MIN_PARAMS_TO_QUANTIZE, new_modules_on_meta=False) -> None` — Walks the policy and replaces every `torch.nn.Linear` with ≥ `min_params` elements with a `bnb.nn.Linear4bit`; the nf4 pack runs on the next `.to(<cuda>)`, bias terms stay in `compute_dtype`. `new_modules_on_meta=True` builds the replacements on the meta device for callers about to `to_empty(...)` the tree. (L73)
- `quantize_int8_in_place(policy, *, torch, compute_dtype, min_params=DEFAULT_MIN_PARAMS_TO_QUANTIZE, threshold=6.0, new_modules_on_meta=False) -> None` — Sibling of `quantize_nf4_in_place` that swaps the same large Linears for `bnb.nn.Linear8bitLt` (LLM.int8 mixed decomposition). `int8` here means LLM.int8, not torchao dynamic int8 — bitsandbytes has no `nf8`. No prequant fast-path: `Int8Params`'s SCB sub-state makes a separate Hub artefact brittle. (L175)
- `install_prequantized_linears(policy, state, *, device, torch) -> tuple[int, set[str]]` — Replaces every `Linear4bit.weight` with `Params4bit.from_prequantized(...)` data read from `state`. Returns `(n_modules_rebuilt, consumed_state_keys)` so the caller can subtract the consumed keys before calling `policy.load_state_dict` for the residual. (L295)
- `detect_prequantized_nf4(spec) -> str | None` — Probes the rSkill's HF repo for a `quantization_metadata.json` sentinel; returns the repo id when the pack is present, `None` otherwise. Routed through `hf_download_cached_first` so a cache hit avoids the HEAD request. (L377)
- `load_prequantized_state_for_rskill(policy, spec, *, torch, log_event_prefix="rskill") -> None` — Combined entry point: validates the metadata sentinel, downloads `model.safetensors`, calls `install_prequantized_linears`, then applies the residual via `policy.load_state_dict(leftover, strict=False)`. Silent no-op when the rSkill ships bf16 weights — adapters can call it unconditionally after their own `quantize_nf4_in_place`. (L442)
- `peek_safetensors_keys(repo_id, *, filename="model.safetensors") -> set[str] | None` — Reads only the safetensors header and returns its key set, for both prequantized (nf4) and bare (int8) checkpoints. Used by `targeted_reset_parameters` to skip the init walk for modules whose params the upcoming state load will overwrite anyway. (L571)
- `targeted_reset_parameters(policy, *, covered_keys) -> None` — Calls `module.reset_parameters()` only on modules whose params are NOT covered by the about-to-load state dict (`covered_keys=None` reproduces the unconditional reset). Model-agnostic, shared by π0.5, MolmoAct2, and future meta-init families. (L933)
- `tie_transformers_weights(policy) -> None` — Calls `module.tie_weights()` on each outermost transformers backbone (pre-order, skipping already-tied descendants); a raising `tie_weights` is non-fatal. Shared by π0.5 and MolmoAct2. (L993)
- `normalise_manifest_dtype(manifest) -> str | None` — Pulls `manifest.quantization.dtype.value` as a string; returns `None` for manifests without a quantization block. Lifted out of `pi05.py` into the shared module so smolvla / xvla / future quantized adapters share one implementation. (L633)
- `QUANTIZATION_DTYPE_ENV: str` — `"OPENRAL_QUANTIZATION_DTYPE"` — the one all-family per-run dtype override, replacing the old per-family `OPENRAL_GR00T_QUANTIZATION`/`OPENRAL_RLDX_QUANTIZATION` env vars. (L657)
- `canonical_quant_token(raw) -> str | None` — Normalises a dtype spelling to the one token every consumer understands (`int4`/`4bit` → `nf4`, float aliases fold, unknown tokens pass through lowercased). `None` stays `None` so “unset” is distinguishable from an explicit `"none"` (= do not quantize). (L686)
- `QuantPlan(dtype, quantize, source, extra)` — `NamedTuple` returned by `resolve_quant_plan`. `quantize` is `True` only for a packing token (`nf4` / `int8`), which is the single definition of "does this load rewrite Linears"; `source` is `env` / `spec_quantization` / `spec_extra` / `manifest` / `default` / `unset` so the adapter can log what won; `extra` carries the manifest's `quantization.extra` knobs. (L710)
- `quantization_extra(manifest) -> Mapping[str, Any]` — Returns `manifest.quantization.extra` (the `quantize_scope` / `nf4_min_params` knobs), or `{}`. One home for the packing knobs, which used to sit in `policy_extras` next to unrelated adapter settings — which is why each family grew its own reader. (L732)
- `resolve_quant_plan(spec, manifest=None, *, default=None) -> QuantPlan` — The single quantization resolver for every policy family, precedence `$OPENRAL_QUANTIZATION_DTYPE` > `spec.quantization.dtype` (typed) > `spec.extra["dtype"]` > `manifest.quantization.dtype` > `default`. `manifest.quantization.dtype` is the RUNTIME dtype (a checkpoint stored at another precision records it in `quantization.extra.stored_dtype`, e.g. the GR00T skills: `int4` runs, `bf16` stored), so `active_min_vram_gb` / `check_quantization_dtype` key what actually loads. Logs at WARNING when the resolved dtype differs from the manifest's declared one. Every adapter then calls `require_supported_dtype` / `sidecar_quant_token`. (L745)
- `require_supported_dtype(plan, supported, family) -> None` — Raises `ROSConfigError` naming the family, the requested token + its source, and the supported set when `plan.dtype` is outside what the family's load path honours (an unset plan = adapter default, always accepted). Called by every family: pi05 {nf4,int8,bf16,fp16,fp32,none}, molmoact2 {nf4,bf16,fp16,fp32,none}, openvla {nf4,bf16,none} (+ nf4 needs CUDA), gr00t {nf4,fp32,none}, smolvla {bf16}, act/diffusion/xvla/diffuser_actor {fp32}, lingbot_va_a1 {bf16}. (L826)
- `torch_dtype_for(torch, dtype_str, device) -> Any` — Map a dtype token (`bf16`/`bfloat16`, `fp16`/`float16`/`half`, `fp32`/`float32`) to a torch dtype; `None`/`"none"` pick the CUDA-aware default (bf16 on CUDA, fp32 elsewhere). A packing (`nf4`/`int8`) or unknown token raises `ROSConfigError` — falling through is how MolmoAct2 loaded `int8` as bf16. Packing adapters pass `None` for their compute dtype. Lifted out of `pi05.py`. (L862)
- `sidecar_quant_token(plan, accepted, family) -> str` — Map a plan onto a sidecar's `--quantization` choices: `bf16` → `none` (every sidecar loads `none` at bf16), anything outside `accepted` raises via `require_supported_dtype`. Used by rldx / gr00t_b1k / xr1 / lingbot_vla(2) / internvla_n1. (L891)
- `default_dtype_for_device(device) -> str` — Picks a default load dtype when the manifest doesn't specify one: `nf4` on CUDA (so 3.4 B-param backbones fit in ~4 GiB), `fp32` elsewhere. Lifted out of `pi05.py`. (L921)

#### `python/sim/src/openral_sim/policies/_policy_loading.py`
_Shared loader helpers for openral_sim policy adapters (extracted in this branch to remove the parallel `_load_manifest_for_spec` copies from `smolvla.py` / `rldx.py` / `pi05.py`). Module docstring explains why the manifest-resolution branch is generic but the quantization branch deliberately stayed family-specific._

- `load_manifest_for_spec(spec) -> RSkillManifest | None` — Returns the parsed `openral_core.RSkillManifest` when `spec.weights_uri` is a resolvable skill reference; returns `None` for bare `hf://` URIs and local paths so the caller can decide whether the missing manifest is fatal (SmolVLA raises; pi05 / RLDX fall back to the URI directly). Tolerant of `spec=None` / `spec.weights_uri=None`. (L42)
- `lazy_import_lerobot(adapter_name, *, install_hint="just sync --all-packages --group libero") -> tuple[Any, Any]` — Imports `torch` + lerobot's `make_pre_post_processors` behind a typed `ROSConfigError` install hint. Returns `(torch, make_pre_post_processors)`; the caller imports its own `Policy` class separately. (L73)

#### `python/sim/src/openral_sim/policies/pi05.py`
- `_build_pi05(env_cfg) -> _PI05Adapter` — Calls `_vla_core.apply_chunk_replay`. `compile` is intentionally NOT plumbed: the adapter sets `pi05_cfg.compile_model = False` to keep the quantization path stable. Supports `quantization.dtype` ∈ {`nf4`/`int4`, `int8`, `bf16`, `fp16`, `fp32`}: `nf4`/`int4` runs `quantize_nf4_in_place` + optional prequant fast-path; `int8` runs `quantize_int8_in_place` against `bnb.nn.Linear8bitLt` (LLM.int8, CUDA-only); everything else casts and moves to device. The dtype comes from `openral_sim._quantization.resolve_quant_plan` (guarded by `require_supported_dtype`) (the four `_manifest_dtype` / `_normalise_manifest_dtype` / `_torch_dtype_for` / `_default_dtype` helpers used to live here — moved to `_quantization.py` so smolvla / xvla can reuse them). Manifest resolution routes through `openral_sim.policies._policy_loading.load_manifest_for_spec`. Processor sidecars resolved via `_resolve_pretrained_path(spec, repo_id)` → delegates to `_processors.resolve_processor_dir` (manifest-first, per the rSkill self-containment audit Gap 1+3; snapshot fallback for non-rSkill refs). Every load phase is wrapped in `_pi05_phase(...)` so the operator sees a per-phase wall-time + GPU footprint in the logs and in `openral dashboard`. **Both nf4 and int8 take a fast meta-init path on CUDA** that skips lerobot's slow `PI05Policy.from_pretrained` (~152 s for the 3.4 B-param backbone): nf4 loads from the prequant safetensors via `load_prequantized_state_for_rskill`; int8 loads the source bf16 safetensors via `_load_bf16_state_for_int8` + `_rebuild_int8_params_for_linear8bitlt`. Combined effect on warm RTX 4070 cache: 95 s → 10 s (9×) for nf4 / 165 s → 11 s (14.6×) for int8. (L425)
- `_PI05Adapter._chunk_forward(batch, **kwargs) -> Any` — Chunk producer for the executor; predicts under the adapter's autocast. `**kwargs` carries the executor's RTC arguments (`inference_delay` / `prev_chunk_left_over`) straight through to `predict_action_chunk`, and is empty on the non-RTC path.
- `_pi05_phase(name, **fields) -> ContextManager[None]` — Adapter-local shortcut for `phase_timer(name, prefix="pi05", gpu_mb=True, log=_log)`. (L383)
- `_targeted_reset_parameters`, `_tie_transformers_weights` — Module-level aliases re-importing `_quantization.targeted_reset_parameters`/`tie_transformers_weights`, kept so the int8 fast path here still resolves them.
- `_expand_covered_keys_via_tied_storage(policy, covered_keys) -> set[str]` — Detects tied parameters via `Tensor.untyped_storage().data_ptr()` and extends `covered_keys` to include every key in a tied group whenever any member is already covered. Uses `named_parameters(remove_duplicate=False)` because the default dedups tied params away. (L251)
- `_load_bf16_state_for_int8(policy, repo_id, *, torch) -> None` — Downloads `<repo>/model.safetensors` via `hf_download_cached_first` and applies it via `policy.load_state_dict(strict=False)`. The int8 fast meta-init path's substitute for lerobot's ~152 s `PI05Policy.from_pretrained`. (L329)
- `_rebuild_int8_params_for_linear8bitlt(policy) -> int` — Re-wraps each `Linear8bitLt.weight` as a fresh `bnb.nn.Int8Params(has_fp16_weights=False)`. `to_empty(device=...)` strips `Parameter` subclasses; without this rewrap the downstream `policy.to(<cuda>)` would never trigger bnb's int8 pack. (L284)
- `_resolve_pretrained_path(spec, repo_id) -> str` — Returns a local directory containing the lerobot processor sidecars. Local path → verbatim; otherwise routes through `_processors.resolve_processor_dir`. (L397)

#### `python/sim/src/openral_sim/policies/molmoact2.py`

MolmoAct2 loads lerobot's in-tree `MolmoAct2ForConditionalGeneration` directly — no `trust_remote_code`, not a lerobot policy. Drives `predict_action(...)` and replays the chunk from its own queue. Model, processor, and `norm_stats.json` load from the manifest's `source_repo`; NF4 weights overlay from `weights_uri`.

- `_build_molmoact2(env_cfg) -> _MolmoAct2Adapter` (L406, `@POLICIES.register("molmoact2")`) — Loads Ai2's MolmoAct2 (~5.49B params, Molmo2-ER VLM + flow-matching action expert). Resolves manifest + dtype, delegates to `_load_molmoact2_model`, then wires replay cadence (clamped to `max_action_horizon`), norm tag, image flips, state/action dims, and autocast. (L686)
- `_load_molmoact2_model(*, torch, model_cls, config_cls, processor_cls, source_repo, spec, device, dtype_str, max_crops) -> tuple[model, processor, use_nf4, torch_dtype]` (L382) — Always loads the processor. With a prequant NF4 pack on CUDA, builds the model on the meta device, quantizes, then loads the prequant state directly — skipping lerobot's slow bf16 `from_pretrained` materialisation. Falls back to the slow bf16-load-then-quantize path for bf16/non-CUDA/no-pack. nf4 is CUDA-only and the default (bf16 needs ~11 GiB and OOMs an 8 GiB GPU; nf4 needs ~4 GiB). (L519)
- `_resolve_max_crops(spec, manifest) -> int | None` (L305) — Resolves the image-processor `max_crops` override: `vla.extra` → env var → manifest → `None` (checkpoint default 8). A secondary lever only — the actual 8 GiB fit is enabled by `_enable_expandable_segments`, not this. (L492)
- `_enable_expandable_segments() -> None` (L131) — Sets the CUDA allocator's `expandable_segments:True` before the first allocation, called when `device` is CUDA. Without it, MolmoAct2 NF4's first-forward embedding `cat` OOMs an 8 GiB card; with it, it fits. No-op if the operator already set the var. (L134)
- `_molmoact2_phase(name, **fields) -> ContextManager[None]` (L105) — Adapter-local shortcut for `phase_timer(name, prefix="molmoact2", gpu_mb=True, log=_log)`. (L160)
- `_import_molmoact2() -> tuple[Any, Any, Any]` (L118) — Imports lerobot's in-tree `MolmoAct2ForConditionalGeneration` + `MolmoAct2Config` + transformers `AutoProcessor` behind a typed `ROSConfigError` install hint. Returns `Any` (lerobot / transformers are optional, unstubbed deps). (L173)
- `_strip_hf_uri(uri, *, field_name) -> str` (L145) — Strip the `hf://` prefix off a manifest URI, validating it is present. (L282)

#### `python/sim/src/openral_sim/policies/_processors.py`
_Shared `resolve_processor_dir(spec, repo_id) -> str` helper used by the diffusion/xvla/pi05 adapters to fetch `policy_preprocessor.json`/`policy_postprocessor.json`, mirroring the smolvla/modern-ACT pattern._

- `resolve_processor_dir(spec, repo_id) -> str` — Manifest-first: when `spec.weights_uri` resolves to a manifest that declares a `processors` block, delegates to `materialize_processor_dir(manifest)` (per-file `hf_hub_download`). Otherwise falls back to `snapshot_download(repo_id, ignore_patterns=["*.md"])` — the path legacy `hf://lerobot/diffusion_pusht` URIs still rely on. (L23)

#### `python/sim/src/openral_sim/policies/rldx.py`
_Auto-managed sidecar adapter for RLWRLD/RLDX-1 (Qwen3-VL-8B + Multi-Stream Action Transformer, ~6.9B params). Runs in an out-of-process Python 3.10 venv over the sidecar's native ZMQ+msgpack wire, because `rldx` pins `python==3.10.*` and ships a custom `RLDX` architecture class not in HF Transformers (no `trust_remote_code` escape). Auto-spawns on first observation (`OPENRAL_RLDX_AUTO_SPAWN=1`). Sidecar boot helper: [`tools/rldx_sidecar.py`](10-tools.md#toolsrldx_sidecarpy). Used as `policy_id: "rldx"` in `rskills/rldx1-*/rskill.yaml`._
- `class _RLDXSidecarAdapter` — ZMQ-backed RLDX policy adapter. Pings the server on init; if absent and `auto_spawn=True`, forks the sidecar boot script and polls `ping` until it answers or `boot_timeout_s` elapses. Replays the upstream 16-action chunk (replan step count from `resolve_n_action_steps`: `--n-action-steps` > manifest `n_action_steps` > deprecated `vla.extra.replan_steps` > half-chunk default). Dispatches obs/action layout by the manifest's `state_layout` (LIBERO/GR1/RC365/SimplerEnv WidowX/SimplerEnv Google). `close()` tears down a spawned child, no-op for a pre-existing server. Before adopting a pre-existing sidecar, verifies its on-disk identity record and raises `ROSConfigError` on a checkpoint mismatch, so two checkpoints can never silently share a port.
- `_encode_ndarray(obj) -> Any` — msgpack `default` hook; serialises ndarrays via `np.save → BytesIO` wrapped in `{"__ndarray_class__": True, "as_npy": <bytes>}` (mirrors `MsgSerializer.encode_ndarray` in `rldx/policy/server_client.py`).
- `_decode_ndarray(obj) -> Any` — msgpack `object_hook`; reverse of `_encode_ndarray`.
- Manifest resolution: the adapter resolves skill references in `weights_uri` through the shared `openral_sim.policies._policy_loading.load_manifest_for_spec` helper so it can read `state_contract.layout` for LIBERO / GR1 / RC365 dispatch, `image_preprocessing.flip_180`, and the canonical `hf://` model id. (The private `_load_manifest_for_spec` copy that used to live here was removed in the 2026-05 cleanup.)
- `_RLDXSidecarAdapter._init_socket / _try_ping / _verify_existing_identity / _wait_for_boot / _spawn_sidecar / _terminate_child / _is_port_busy / _locate_sidecar_script / _resolve_model_id` — auto-spawn lifecycle helpers. `_init_socket` recreates the ZMQ REQ socket on failure (a timed-out `recv()` leaves it stuck in the strict REQ state machine); `_try_ping` does one bounded round trip and resets the socket on failure so `_wait_for_boot` makes real progress; `_verify_existing_identity` fails closed on a checkpoint mismatch via the sidecar identity record.
- `_build_libero_obs / _build_gr1_obs / _build_rc365_obs / _build_simpler_widowx_obs / _build_simpler_google_obs / _pick_single_camera / _pick_images / _pick_state / _normalize_action_column / _assemble_libero_chunk / _assemble_gr1_chunk / _assemble_rc365_chunk / _assemble_simpler_chunk` — per-embodiment wire builders/parsers. The LIBERO assembler rescales the gripper column from the RLDS `[0,1]` convention to LIBERO/robosuite's `[-1,+1]` via `_rldx_gripper_to_libero` (without this the Franka gripper never actuates). GR1 concatenates the Fourier-native action groups into the 29-D BASIC composite; RC365 concatenates the PandaMobile groups into 12-D, trimmed to the 11-D BASIC env action by `openral_sim.backends.robocasa`.
- `_rldx_gripper_to_libero(gripper) -> NDArray[float32]` — Maps an RLDS-convention gripper column (`[0,1]`, 0=close/1=open) to LIBERO/robosuite's (`[-1,+1]`, -1=open/+1=close) via `out = -sign(2*g - 1)`. Called from `_assemble_libero_chunk`; without it the Franka gripper never actuates.
- `_resolve_state_layout(manifest) -> str` — module-level helper shared by the `rldx` and `gr00t` factories; maps `manifest.state_contract.layout` to one of `gr1/rc365/simpler_widowx/simpler_google`, else `"libero"`. Single source of truth for obs/action dispatch so neither factory hardcodes an embodiment.
- `_derive_sidecar_port(*, family, model, embodiment_tag, quantization, layout) -> int` / `_resolve_sidecar_port(*, port_env, extra_port, …) -> int` — per-identity default port (SHA-1-bucketed into 20000–39999, non-crypto) so two different checkpoints never collide on the old hard `5555`; `_resolve_sidecar_port` applies precedence env-pin > `vla.extra.port` > derived default.
- `_build_rldx(env_cfg) -> _RLDXSidecarAdapter` — `@POLICIES.register("rldx")` factory. Honours `OPENRAL_RLDX_*` env overrides and `vla.extra` (`image_size`/`timeout_ms`/`camera_keys`/`auto_spawn`/`boot_timeout_s`/`embodiment_tag`/`model_id`). Quantization comes from `resolve_quant_plan` (default `nf4`) → `sidecar_quant_token` over the sidecar's {none,nf4,int8}; dispatches the obs/action contract via `_resolve_state_layout` and the port via `_resolve_sidecar_port`.

#### `python/sim/src/openral_sim/policies/gr00t.py`
_NVIDIA Isaac GR00T N1.7 policy adapter — standard checkpoints run in-process under the workspace's Python 3.12 via lerobot 0.6.0's native `GrootPolicy`, mirroring the smolvla adapter. NF4 (backbone-only) keeps the ~3B model on an 8 GiB card. The official BEHAVIOR-1K organizer checkpoint is its own registered family `gr00t_b1k` (`_build_gr00t_b1k` → `behavior_groot.build_behavior_groot_policy`), since it is pinned to the separate `wensi-ai/Isaac-GR00T` Python 3.10 runtime. RLDX-1 (a GR00T-N1.5 finetune) still runs on its own ZMQ sidecar via the `rldx` adapter._
- `_build_gr00t_b1k(env_cfg) -> PolicyAdapter` — `@POLICIES.register("gr00t_b1k")` factory: merges the manifest's `policy_extras` under `spec.extra` and delegates to `behavior_groot.build_behavior_groot_policy` (which resolves quantization via `sidecar_quant_token` over {none,nf4,int8}). (L424)
- `_build_gr00t(env_cfg) -> _GrootAdapter` — `@POLICIES.register("gr00t")` factory. Resolves the rSkill manifest, `snapshot_download`s the raw N1.7 checkpoint locally (the GR00T processor factory only recognises a local raw-checkpoint dir), builds the config, loads `GrootPolicy.from_pretrained` on CPU under a float32 default dtype, NF4-rewrites the backbone, then a pure `policy.to(device)` packs the `Linear4bit` shells onto the GPU. Reads `OPENRAL_GR00T_EMBODIMENT_TAG` env + `vla.extra` (`embodiment_tag` default `libero_sim`, `camera_keys`). Quantization comes from the shared `resolve_quant_plan` (default `nf4`) guarded by `require_supported_dtype` over {nf4, fp32, none} (an unpacked load runs fp32 params — `model_params_fp32=not quantize` — so bf16/fp16/int8 raise); the manifests declare `int4` (what runs) with `quantization.extra.stored_dtype: bf16`; `quantize_scope` reads from `quantization.extra`. Execution horizon (`_replan_steps`) comes from `resolve_n_action_steps` (manifest / `--n-action-steps`), defaulting to the policy's `_action_queue_steps` — GR00T's `libero_sim` 8-of-16. (L454)

#### `python/sim/src/openral_sim/_behavior_wire.py`
_Shared BEHAVIOR-1K R1Pro wire constants + option helpers — the single source of truth for the OmniGibson evaluator's raw observation keys and the official R1Pro state/action widths, imported by `backends.behavior`, `policies.behavior_groot`, and the `openral behavior serve` WebSocket bridge (`openral_cli.behavior`) so the three do not hand-copy the constants and drift independently. The out-of-process `tools/` sidecars cannot import `openral_sim`; their copies of these constants are the wire-contract mirror and must be updated in lockstep._

- `const STATE_KEY = 'robot_r1::proprio'` — The official evaluator's proprioception observation key. (L21)
- `const STATE_DIM = 61` (L22)
- `const ACTION_DIM = 23` (L23)
- `const CAMERA_SENSORS: dict[str, str]` — Camera role → the evaluator's sensor name (`head`/`left_wrist`/`right_wrist`). (L27)
- `const CAMERA_RGB_KEYS: dict[str, str]` — Camera role → the raw observation's `<sensor>::rgb` key. (L32)
- `explicit_port(env_value, opt_value, default, *, env_var) -> int` — Resolve a sidecar port: explicit values must parse or fail loud, so a typo in an env var or scene YAML pointing at an already-running sidecar is never silently discarded in favour of the hash-derived default. (L37)

#### `python/sim/src/openral_sim/policies/behavior_groot.py`
_Official BEHAVIOR-1K GR00T policy adapter. The organizer checkpoint depends on the pinned `wensi-ai/Isaac-GR00T` `behavior` branch rather than lerobot's native N1.7 loader, so it runs in an externally-provisioned Python 3.10 sidecar. The OpenRAL process carries only pyzmq/msgpack (`behavior-groot` dependency group)._

- `build_behavior_groot_policy(env_cfg, manifest, extra) -> _BehaviorGrootAdapter` — Resolves the local organizer checkpoint, sidecar endpoint, task/instruction, control mode, and quantization; auto-spawns `tools/behavior_groot_sidecar.py` or connects to an operator/remote sidecar. (L202)
- `_BehaviorGrootAdapter` — `PolicyAdapter` implementation that preserves the official flattened evaluator observation when present, otherwise maps canonical `images.{head,left_wrist,right_wrist}` + 61-D `state` back to the R1Pro wire keys; validates a finite 23-D action.
- `_behavior_wire_observation(observation, *, instruction) -> dict[str, object]` — Pure official-wire assembler used by the adapter and unit tests.

#### `python/sim/src/openral_sim/backends/behavior.py`
_BEHAVIOR-1K / OmniGibson scene adapter. Runs the official evaluator environment in its own sidecar and registers `scene.id=behavior`, fixed to `r1pro`._

- `class _BehaviorSidecar` — ZMQ-backed `SimRollout`; surfaces 61-D `state`/`policy_state`, manifest-order 22-joint positions/velocities, three RGB views, and sim time.
- `step_action_group(actions) -> StepResult` — Validate six equal-tick safety-approved slots and commit them as one official 23-D action.
- `_compose_action_group(actions) -> NDArray[float32]` — Compose base twist + torso/arm joint targets + dual grippers into the official action order.
- `provision_behavior() -> None` — Pre-launch provisioner: locates the externally-installed OmniGibson + BEHAVIOR dataset environment (set up via upstream's `./setup.sh`) or raises the setup command. Covers provisioning only — the OmniGibson boot still runs inside the HAL's 300s-bounded `on_configure`, and the scene's boot timeout is set well past that bound. (L362)
- `const _SCENE_ID = 'behavior'` (L27)
- `const _ROBOT_ID = 'r1pro'` (L28)
- `const _SIDECAR_PYTHON_ENV = 'OPENRAL_BEHAVIOR_SIDECAR_PYTHON'` (L29)
- `const _SIDECAR_SCRIPT_ENV = 'OPENRAL_BEHAVIOR_SIDECAR_SCRIPT'` (L30)
- `const _AUTO_SPAWN_ENV = 'OPENRAL_BEHAVIOR_AUTO_SPAWN'` (L31)
- `const _HOST_ENV = 'OPENRAL_BEHAVIOR_HOST'` (L32)
- `const _PORT_ENV = 'OPENRAL_BEHAVIOR_PORT'` (L33)
- `const _DEFAULT_HOST = '127.0.0.1'` (L35)
- `const _PORT_MIN = 23000` (L36)
- `const _PORT_MAX = 23999` (L37)
- `const _DEFAULT_TIMEOUT_MS = 120000` (L38)
- `const _DEFAULT_BOOT_TIMEOUT_S = 1200.0` (L39)
- `const _ACTION_DIM = 23` (L40)
- `const _ACTION_GROUP_SIZE = 6` (L41)
- `const _JOINT_DIM = 22` (L42)
- `const _STATE_KEY = _behavior_wire.STATE_KEY` (L44)
- `const _CAMERA_KEYS = _behavior_wire.CAMERA_RGB_KEYS` (L45)
- `_build_behavior_scene(env_cfg) -> _BehaviorSidecar` — Resolve the official BEHAVIOR Python and auto-spawn `tools/behavior_scene_sidecar.py`.
- `_GrootAdapter(spec, device, _policy, _preprocessor, _postprocessor, _torch, ...)` — In-process `GrootPolicy` adapter. `libero_sim`'s relative-action decode step refuses per-step decoding, so the adapter predicts the full chunk while pack-step state is fresh, then queues and replays it one step at a time (chunk-replay, like the rldx adapter). `_build_batch` feeds float CHW `[0,1]` images to the embodiment's modality keys plus a proprio state vector, honouring `image_preprocessing.flip_180`/`flip_vertical`.
- `_build_groot_config(*, local_path, embodiment_tag, quantize, image_keys, state_dim, action_dim) -> GrootConfig` — Constructs a `GrootConfig` with `embodiment_tag` set at construction time (the `libero_sim` transform resolves in `__post_init__`; setting it later is too late) and explicit `input_features`/`output_features` pinning the head to the real 7-D LIBERO action instead of the 132-D padded default.
- `_quantize_groot_nf4(policy, torch, *, scope) -> None` — NF4-rewrites the GR00T model via `quantize_nf4_in_place`. `scope="backbone"` (default) packs only the ~2B Qwen3-VL backbone, leaving the diffusion head bf16 (enough for a 16-layer LIBERO head to fit 8 GB); `scope="model"` also packs the DiT head's large Linears for heavier heads. The `>=4M`-param threshold always spares the small `TimestepEncoder` MLP so its params stay bf16.
- `_patch_groot_dtype_property(torch) -> None` — Rebinds `GR00TN17.dtype` to report the first floating-point param dtype — after NF4 the first param is a uint8 `Params4bit`, which would otherwise make `prepare_input` cast float inputs to uint8 and produce garbage. No-op for a non-quantized load.
- `_import_real_groot_policy() -> Any` — Returns the real lerobot `GrootPolicy`, evicting the empty compat stub some older-lerobot installs leave in `sys.modules` so it can't shadow the real N1.7 class.
- `_to_groot_state(state, expected_dim) -> NDArray[float32]` — Returns the GR00T proprio vector as flat float32, width-checked against `expected_dim` (the rSkill's `state_contract.dim`). GR00T pads to `max_state_dim` internally, so only a width check is needed here.
- `_groot_phase(name, **fields)` — `phase_timer` shortcut (prefix `gr00t`).
- `_env_bool(name, default) -> bool` — permissive boolean env-var parser (`1` / `true` / `yes` / `on`). Also imported by `policies.rldx` (formerly a duplicate copy there).

#### `python/sim/src/openral_sim/policies/rlbench_3dda.py`
_3D Diffuser Actor RLBench keyframe policy adapter (MIT). Proxies `tools/rlbench_3dda_sidecar.py` over the shared `SidecarClient` — the model + obs history live in the externally-provisioned py3.10 venv shared with the rlbench scene sidecar. `step()` returns an 8-D keyframe `[x y z qx qy qz qw gripper_open]` for `backends/rlbench.py` to plan + execute._
- `_Diffuser3DActorAdapter(spec, device, _client)` — `PolicyAdapter`; `last_input_frame()` for episode-video capture.
- `_build_diffuser_actor(env_cfg) -> _Diffuser3DActorAdapter` — `@POLICIES.register("diffuser_actor")` factory; reads `backend_options.{rlbench_task,variation,policy_port}` (the policy needs the task to pick the matching CLIP instruction embedding). Connects a `SidecarClient(name="rlbench-3dda", …)`.

#### `tools/rlbench_3dda_sidecar.py`
_3D Diffuser Actor policy sidecar (py3.10 venv only; no openral import). Loads the published PerAct checkpoint, keeps per-episode obs history, and serves `ping`/`reset`/`get_action`/`close`. `dgl.geometry.farthest_point_sampler` is replaced by a pure-torch FPS shim so neither flash-attn nor pytorch3d is needed._
- `main(argv) -> int` — argparse + load the `DiffuserActor` checkpoint and serve the ZMQ REP loop. (L213)

#### `python/sim/src/openral_sim/policies/lingbot_vla2.py`
_Robbyant LingBot-VLA 2.0 policy adapter (Apache-2.0 code + weights). Proxies the auto-provisioning boot helper `tools/lingbot_vla2_sidecar.py` over the shared `SidecarClient`; the 6.38B model runs in its own Python 3.12 + torch-2.9.1 venv, since upstream pins torch==2.8.0 + custom Triton MoE kernels incompatible with the workspace, and ships no `config.json` architecture (no `trust_remote_code` escape). Default embodiment `robotwin` (dual-arm AgileX Cobot Magic)._
- `_LingBotVla2Adapter(spec, device, _client, _camera_keys, _replan_steps, ...)` — `PolicyAdapter` proxying the sidecar. Predicts a `(50, 14)` chunk and replays one 14-D step per `step`, refilling when the queue drains (`_DEFAULT_REPLAN_STEPS=25`, mirroring the upstream deploy default). `_refill` marshals the 3 scene cameras (re-keyed onto `cam_high`/`cam_left_wrist`/`cam_right_wrist`) + proprio state + instruction to `get_action`. (L215)
- `_build_lingbot(env_cfg, *, variant) -> _LingBotVla2Adapter` — Shared factory behind an auto-managed `SidecarClient`. `variant` selects the model family (`v2` 6B Qwen3-VL MoE / `v1` 4B Qwen2.5-VL dense expert); reads `vla.extra` for model/camera/port overrides; quantization via `resolve_quant_plan` → `sidecar_quant_token` over {none,nf4} (the per-family `OPENRAL_LINGBOT_VLA*_QUANTIZATION` / `vla.extra.quantization` knobs were removed). `expected_identity={model,robo_name}` guards against adopting a stale sidecar on the default port. (L293)
- `_build_lingbot_vla2(env_cfg)` / `_build_lingbot_vla(env_cfg)` — `@POLICIES.register("lingbot_vla2")` (6B) and `@POLICIES.register("lingbot_vla")` (4B RoboTwin post-train, `model_family: "lingbot_vla"` in `rskills/lingbot-vla-4b-robotwin/rskill.yaml`) thin wrappers over `_build_lingbot`. (L406)
- `_policy_default_port(model_id, robo_name, variant="v2") -> int` / `_resolve_camera_keys(env_cfg, extra) -> tuple[str, ...]` / `_resolve_model_id(spec, extra, default_model_id) -> str` / `_locate_sidecar_script() -> Path` / `_opt_int(value, default) -> int` — per-(variant, model, embodiment) SHA-256-bucketed default port (20000–39999); scene-camera resolution (order load-bearing: top/left-wrist/right-wrist); model-id resolution (`vla.extra.model_id` > manifest `weights_uri` > `spec.weights_uri` > per-variant default); boot-helper locator; `vla.extra` int coercion. (L123)

#### `python/sim/src/openral_sim/policies/lingbot_va_a1.py`
_Thin LingBot-VA real-deployment adapter for the Galaxea A1 embodiment. Connects to the Runtime-owned versioned policy gateway and returns absolute six-joint plus normalized-gripper proposals through OpenRAL's normal candidate-action and safety path._

- `_LingBotVaA1Adapter(spec, *, robot_description)` — Validates the A1 embodiment and rSkill identity, negotiates the joint envelope and step bound, sends joints + paired RGB observations, and validates the returned 7-D action. Runtime owns deployment config, EEF transforms, replay, and IK.
- `_joint_contract(spec, robot_description)` — Extracts ordered finite command limits and rejects a policy substep above either active HAL phase limit.
- `_model_identity(spec) -> tuple[str, str]` — Resolves the rSkill's immutable model repo/revision for the gateway identity handshake.
- `_build_lingbot_va_a1(env_cfg) -> _LingBotVaA1Adapter` — `@POLICIES.register("lingbot_va_a1")` factory. The adapter explicitly accepts only the `galaxea_a1` embodiment.

#### `tools/lingbot_vla2_sidecar.py` + `tools/_lingbot_vla2_server.py`
_Boot helper + server for the LingBot-VLA 2.0 sidecar, companion to `openral_sim.policies.lingbot_vla2`._ The boot helper (openral interpreter) clones the pinned upstream repo, builds a Python 3.12 venv with the torch/triton overrides needed for aarch64, stamps the sidecar identity record, then execs into the server. The server (sidecar venv, no openral import) loads the NF4-backbone/bf16-expert model and answers `ping`/`reset`/`get_action`/`close` over the shared msgpack wire. flash-attn is not installed; the upstream `flash_attention_2` hardcode is patched to `sdpa`/`eager` before the model is built.
A `--variant {v2,v1}` switch selects the model family end-to-end. **v1** (LingBot-VLA 1.0, 4B) clones a separate pinned repo and provisions its own `transformers==4.51.3`/`lerobot==0.4.2` venv — it uses real lerobot (no stub) and needs three server-side patches for its flash-free path (inject `rotate_half`, force `attn=eager`, skip `o_proj` in NF4).
- `_ensure_source(home, *, url, sha, repo_env) -> Path` (boot) — pinned-SHA shallow clone into `<home>/source`; per-variant URL/SHA/override env.
- `_ensure_venv(home, source, *, install, venv_env) -> Path` / `_install_v1(uv, py)` (boot) — Python 3.12 venv; v2 installs the upstream `requirements.txt` + `pyzmq`/`bitsandbytes`, v1 installs the explicit V1 stack (torch cu128, lerobot==0.4.2, transformers==4.51.3, numpy==1.26.4, + wire/quant extras); per-variant `*_SIDECAR_PYTHON` override.
- `main() -> int` (boot) — argparse (`--model/--robo-name/--quantization/--device/--attn/--variant/--host/--port/--home`); per-variant provisioning, writes the identity record (`family="lingbot_vla_<variant>"`), then `os.execvpe`s `tools/_lingbot_vla2_server.py --variant <v>` with the per-variant repo + QWEN path env set.
- `_install_attn_fallback(*, target)` / `_install_attn_fallback_v1(*, target)` / `_patch_eager_vision_rotary_v1()` / `_coerce_attn_config(config, target)` (server) — coerce `_attn_implementation` off flash before construction: v2 patches the Qwen3-VL / Qwen2 `_from_config`; v1 patches the shared `PreTrainedModel._from_config` (many sites) and injects `rotate_half`.
- `_LingBotPolicy` / `_LingBotV1Policy` / `_serve` / `_nf4_backbone_in_place(*, skip_names=…)` / `_write_cli_yaml` (server) — load the NF4-backbone/bf16-expert model (v1 skips `o_proj`), flatten the upstream `action.arm.position`(12)+`action.effector.position`(2) to a `(chunk, 14)` array, and serve the ZMQ REP loop.

#### `python/sim/src/openral_sim/_sidecar_common.py`
_Shared boot scaffolding for the out-of-process `rldx` VLA sidecar (a GR00T-N1.5 finetune, hence the `_Gr00tFamilySidecarAdapter` class name — GR00T N1.7 itself now runs in-process): clone, venv, install, env isolation, exec — plus the sidecar identity registry._
- `const _SIDECAR_REGISTRY_DIR = Path.home() / '.cache' / 'openral' / 'sidecars'` — Root of the per-port sidecar identity registry. (L87)
- `run_sidecar(*, label, family, repo_url, args, install_deps, make_wrapper) -> int` — orchestrates a boot (uv → clone → install → wrapper → exec). Writes the identity record (`write_sidecar_identity`) just before `exec_server` so every sidecar this repo starts — auto-spawned or operator-launched — is identifiable. (L418)
- `sidecar_identity_path(port) -> Path` — the per-port identity record path under `~/.cache/openral/sidecars/port-<port>.json`. (L90)
- `write_sidecar_identity(*, port, family, model, embodiment_tag, quantization) -> None` — Record which checkpoint a sidecar is about to serve on `port`. (L95)
- `read_sidecar_identity(port) -> dict[str,str] | None` — Read the identity record back; the adapter uses it in `_verify_existing_identity` to refuse reusing a sidecar serving a different checkpoint. `None` = no record (unverifiable, not a mismatch). (L118)
- `ensure_pip_venv(*, label, home, python, install, override=None, override_env=None, sentinel_name=".deps-installed", spec=None) -> Path` — Creates/reuses/repairs the `<home>/.venv` of a pip-installable sidecar. `spec` (the dependency spec `install` applies) is hashed into the completion sentinel via `spec_marker`, so correcting a pin invalidates the sentinel and re-runs `install` instead of the venv staying frozen at whatever it first resolved. `spec=None` keeps the historical opaque marker. (L185)
- `spec_marker(spec) -> str` — the sentinel content for a dependency spec: a NUL-separated sha256 digest, or `"ok\n"` for `None`. (L169)
- `opt_num(opts, key, default, cast) -> int|float` — Coerces a `scene.backend_options` value via `cast`, ignoring `bool` and swallowing `ValueError`/`TypeError` (returns `default`, never raises). Shared by `backends.isaac_sim` and `backends.robotwin`. (L41)
- `ensure_uv() -> str` — Return the path to `uv` or exit with an install hint. (L147)
- `ensure_source(label, work, repo_url) -> Path` — Shallow-clone `repo_url` into `<work>/source` if absent; return it. (L158)
- `make_isolated_env(venv) -> dict[str, str]` — Builds the scrubbed environment for the sidecar interpreter; also `setdefault`s `TRITON_PTXAS_PATH` to a venv-local CUDA 12.9 `ptxas` when installed, since triton 3.5.1's bundled `ptxas` can't target `sm_121` (GB10/Jetson Thor) until redirected. (L258)
- `exec_server(venv, wrapper, env) -> None` — Replace this process with the sidecar venv's python running `wrapper` (`os.execvpe`). (L376)
- `build_parser(*, description, default_home, default_embodiment_tag, model_help, quant_help) -> argparse.ArgumentParser` — Construct the shared sidecar CLI (`--model/--port/--quantization/--embodiment-tag/--home`). (L386)
- `run_cmd(label, cmd, *, cwd=None, env=None) -> None` — Run `cmd` (echoed with a `[label]` prefix) and raise on non-zero exit. (L139)
- `venv_ptxas(venv) -> Path | None` — the `nvidia-cuda-nvcc-cu12` `ptxas` inside `venv`, or `None` when that wheel isn't installed (the normal case on x86_64). Split out of `make_isolated_env` so a sidecar execing by another route, or a test, can ask the same question. (L362)
- `venv_torch_version(venv) -> str | None` — Return the torch version installed in `venv`, or `None` if absent. (L346)
- `sidecar_port_for_key(key, *, port_min=20_000, port_max=40_000, algorithm="sha256") -> int` — Deterministic port in `[port_min, port_max)` from a `hashlib` digest of `key` (not the builtin `hash`, which is `PYTHONHASHSEED`-salted), so a sidecar's spawn port matches what a later client probes for the same identity. `algorithm="sha1"` reproduces the `rldx` policy's original derivation exactly. Shared by `backends.robotwin`/`rlbench`/`isaac_sim` and `policies.rldx`/`lingbot_vla2`. (L59)
- `alloc_conf_var(torch_version) -> str` — Return the allocator-config env var name (`PYTORCH_ALLOC_CONF` on torch ≥2.9, else `PYTORCH_CUDA_ALLOC_CONF`) that `torch_version` reads. (L306)
- `installed_alloc_conf_var() -> str` — `alloc_conf_var` for the torch installed in *this* interpreter. (L332)

#### `python/sim/src/openral_sim/policies/internvla_n1.py`
_InternVLA-N1 / DualVLN vision-language navigation policy adapter (InternRobotics; weights CC-BY-NC-SA-4.0 noncommercial, code MIT). Proxies `tools/internvla_n1_sidecar.py` over the shared `SidecarClient`; the 8.3B dual-system model runs in an auto-provisioned py3.11 venv. `step()` sends RGB + DA3 monocular depth to the sidecar and maps the returned `twist=[v_forward, w_yaw]` into a 6-D `BODY_TWIST` row, latching to zero twist on the model's STOP until `reset()`. `OPENRAL_INTERNVLA_N1_DEPTH=none` sends a unit-depth placeholder for depth-agnostic checkpoints (e.g. DualVLN's `nextdit_async`)._
- `_InternVLAN1Adapter(spec, device, _client, _camera_key, _da3, _stopped, _last_input)` — `PolicyAdapter`; `last_input_frame()` for episode-video capture.
- `_build_internvla_n1(env_cfg) -> _InternVLAN1Adapter` — `@POLICIES.register("internvla_n1")` factory. Resolves the manifest + quantization (`resolve_quant_plan` → `sidecar_quant_token` over {none,nf4,int8}; it used to read `vla.quantization or manifest` and ignore the env override) + camera key, derives a per-identity port, connects (auto-spawning) a `SidecarClient`, then builds a `Da3DepthClient` unless `OPENRAL_INTERNVLA_N1_DEPTH=none`.

#### `python/sim/src/openral_sim/da3_depth.py`
_Thin ZMQ client for the DA3 monocular metric-depth sidecar — the same model the SLAM/nvblox depth provider uses. RGB in → float32-metres depth out, resized to the RGB frame; monocular, so it feeds a policy identically in sim and on real hardware. Speaks the perception bus's `{"op": ...}` protocol directly, not the `SidecarClient` framing._
- `class Da3DepthClient` — `connect()` pings the default port and **reuses an existing sidecar** (e.g. SLAM's) when present, else auto-spawns `tools/da3_depth_sidecar.py`; `infer(rgb) -> depth(H,W) float32`; `close()` reaps an auto-spawned child. (L56)
  - `Da3DepthClient.connect() -> None` (L67)
  - `Da3DepthClient.infer(rgb) -> NDArray[np.float32]` (L99)
  - `Da3DepthClient.close() -> None` (L133)
- `const DEFAULT_DA3_PORT = 5771` — the shared bind port (matches `depth_provider_node`). (L39)

#### `tools/internvla_n1_sidecar.py` + `tools/_internvla_n1_server.py`
_Boot helper + server for the InternVLA-N1 nav sidecar, companion to `openral_sim.policies.internvla_n1`._ The launcher provisions a py3.11 venv (clones `InternRobotics/InternNav`, installs the inference-only pin set, no flash-attn), then execs the server. The server loads the checkpoint with quantization-aware `from_pretrained` (NF4 on the Qwen backbone, NavDP head left bf16) and answers the same ZMQ+msgpack protocol as other sidecars; refuses an all-zero depth frame.
- `discrete_action_to_twist(actions, *, forward_mps, turn_radps) -> (v, w, stop)` (server) — pure VLN-CE discrete-action → base-twist mapping; unit-tested in `tests/unit/test_internvla_n1_action_mapping.py`.
- `main() -> int` (both) — sidecar: `run_sidecar(..., family="internvla_n1")`; server: argparse (`--model/--host/--port/--quantization/--resize/--num-history/--plan-step-gap/--forward-mps/--turn-radps/--work-dir`) + the ZMQ serve loop.

#### `python/sim/src/openral_sim/policies/__init__.py`
- `_register_policies() -> None` — Side-effect imports of the policy-adapter modules so each registers its factory in `openral_sim.POLICIES` at import time. (L15)

#### `python/sim/src/openral_sim/backends/__init__.py`
- `_register_backends() -> None` — Side-effect imports of the scene-backend modules so each registers its factory in `openral_sim.SCENES` at import time. (L67)
