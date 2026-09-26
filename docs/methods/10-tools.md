# Tools

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `tools/lifecycle_autostart.py`
_Drives a lifecycle node through `configure` → `activate` after `ros2 launch` brings it up. Spawned per node by `deploy_e2e.launch.py` for the HAL, safety kernel, and reasoner._

- `--transition-timeout-s` — Per-transition spin budget, default `300.0`; the HAL's budget is derived per scene, not hardcoded — raise the scene's `backend_options.boot_timeout_s` instead of this flag.
- **Why the HAL's is derived.** The bound must cover the whole `on_configure`, including a sidecar's spawn — some backends need far more than 300 s to boot, so the timeout is set per scene rather than fixed.
- `--required` (flag) — Treat an absent `change_state` service as a failure (exit 1) instead of informational, so a missing node cannot come up looking healthy; used for the deadman watchdog so a deploy without it refuses to run.
- `_STATE_TO_TRANSITION: dict[str, list[int]]` (L36) — `--target` → the `Transition` constants to drive in order (`inactive` = CONFIGURE; `active` = CONFIGURE then ACTIVATE).
- `main() -> int` (L118) — CLI entry; parses `--node` / `--target` / `--required` / `--service-timeout-s` / `--transition-timeout-s` (documented above) and drives the transitions via `_drive_transition`.

### `tools/profile_policy_load.py`
_One-shot wall-time breakdown of a single policy load. Drives `openral_sim.factory.make_policy` against an in-tree rSkill manifest and prints a phase-by-phase summary. Use when a launch or `openral sim run` is slow to first action, to see where the seconds go._
_Wired as `just profile-load <rskill> [args]`. Families whose adapter lacks a phase-timer helper report "no phase_timer events captured", but the end-to-end total is still valid._


- `class _PhaseCapture` — `structlog` processor that buffers `_start` / `_done` events and pairs them by name. Insertion-ordered so the rendered table mirrors the actual load order. (L35)
- `_parse_args(argv) -> argparse.Namespace` — `--rskill <dir>` (required), `--device` (default `auto`). (L69)
- `_build_env_cfg(rskill_dir, *, device) -> _SimpleEnvCfg` — Builds a minimal env_cfg from `<rskill_dir>/rskill.yaml`; mirrors `rskill_runner_node._SimpleEnvCfg`. (L103)
- `_render(pairs, total_s) -> str` — Formats the captured pairs as `phase / elapsed_s / share` columns plus an `(unaccounted)` row when phase coverage misses >1 s. (L122)
- `main(argv=None) -> int` (L149) — Late-imports `openral_sim.factory.make_policy` so the import cost lands inside the profiled total; reports `HF_HUB_OFFLINE` status alongside the result.

### `tools/viz_collision.py`
_Overlays a robot's kernel collision primitives — the shapes the C++ safety kernel checks — on its real meshes at any joint pose, for eyeballing whether they clear the robot without running a deploy. Standalone inspection tool, not a pytest test; run with `PYTHONPATH=packages/openral_safety`._

- `--viewer` — interactive MuJoCo window; `--screenshot PATH` writes an offscreen PNG; `--rviz` publishes the primitives as markers in RViz. `--robot <id>` (default `so101_follower`) and `--deg <j...>` set the pose in degrees.
- `main(argv=None) -> int` (L334) — CLI entry; builds the robot's kernel collision primitives and dispatches to the viewer/screenshot/RViz renderer named above.

### `tools/generate_tight_geometry.py`

_Derives a robot manifest's `tight_geometry` blocks from its real collision meshes, for the safety kernel's staged 26-DOP → convex-hull narrow phase. Both stages are proven to contain the mesh by construction, and mesh placement follows the GEOM transform only. Standalone operator/CI tool needing `mujoco` + `scipy` + `robosuite`; see [`collision-hull-narrow-phase.md`](../reference/collision-hull-narrow-phase.md)._

_Two modes: `emit --robot <path>` prints the YAML fragment to paste into the manifest, annotated per link with vertex counts and margins; `check --robot <path>` re-derives from the mesh and verifies every declared block still contains it, exiting 3 on any failure._

- `REPO_ROOT: Path` (L46) — Repo root, derived from this file's location.
- `PANDA_GEOM_OF_LINK: dict[str, str]` (L78) — Panda manifest link name → MJCF collision geom name (`panda_link{i}` → `link{i}_collision`).
- `link_mesh_in_box_frame(xml_path, geom_name, origin_xyz_rpy) -> np.ndarray` (L112) — Collision-mesh vertices of `geom_name`, expressed in the manifest box's frame. Raises when the geom is missing, isn't a mesh, or when `mesh_pos != geom_pos`.
- `link_mesh_faces(xml_path, geom_name) -> Points` (L148) — Triangle face indices of `geom_name`'s collision mesh, paired with `link_mesh_in_box_frame`'s vertices for the overhang check.
- `ROBOT_MESH_SOURCES: dict[str, tuple[str, dict[str, str]]]` (L82) — Robot name → its MJCF asset path and manifest-link → MJCF-geom name map. A new robot must be registered here before either mode will run for it.
- `main(argv=None) -> int` (L295) — CLI entry; `emit --robot <path>` / `check --robot <path>` subcommands.
- Re-exported from `openral_safety.tight_geometry` (moved there so the collision lowering can refine links; see `06-reasoning-wam-safety-observability.md`): `derive_tight_geometry`, `hull_overhang_m`, `refine_dop_to_budget`, `_dop_axes`, and `_round_up_m` / `_HULL_OVERHANG_SAFETY_MARGIN` under their old names. The tool inserts `packages/openral_safety` on `sys.path` for a standalone run.

### `tools/schema_export.py`
_Generates JSON Schema files for every public `openral_core` model._

- `_enum_schema(cls) -> dict[str, Any]` — Minimal JSON Schema for a `str` Enum. (L181)
- `export_schemas(out_dir=_OUT_DIR) -> dict[str, Any]` — Export JSON Schema for every public model. (L193)
- `check_drift(out_dir=_OUT_DIR) -> bool` — On-disk schemas == regenerated. (L245)

### `tools/check_repo_state_map.py`
_Pre-commit drift guard checking the mechanically verifiable half of `docs/architecture/repo-state-map.html`: that its `pkg:` pointers name something real and its asserted counts haven't rotted. Prose on the map stays a human judgement call._

- `REPO_ROOT` (L37) — Repo root, derived from this file's location.
- `MAP_PATH = REPO_ROOT / "docs" / "architecture" / "repo-state-map.html"` (L38) — The map file this script checks.
- module const `COUNT_TOLERANCE: float` (L45) — Counts are held to within 10%, not to the digit — an exact check would go red on every added test file and get disabled.
- `iter_cards(html: str) -> list[tuple[str, str]]` — Pair each `pkg:` value with the `desc:` that follows it in the same card. (L71)
- `_resolves(head: str, token: str) -> bool` — Whether a `pkg:` token names a real path under any of the three spellings the map uses: repo-root-relative, relative to the card's leading package under the workspace src layout, or a bare module name. (L80)
- `check_paths(cards) -> list[str]` — Report `pkg:` tokens that name nothing on disk. (L101)
- `check_counts(cards) -> list[str]` — Report `desc` counts that have drifted past `COUNT_TOLERANCE`. (L120)
- `main(argv: list[str] | None = None) -> int` (L139) — CLI entry; `--quiet`; runs `check_paths` + `check_counts`, prints drift and returns 1, else 0.

### `tools/audit_sim_configs.py`
_Real GPU rollout audit for every YAML under `scenes/`. Operator-driven, one episode per config; writes `outputs/audit_sim_configs.json` and prints a Markdown table (`just sim-audit`). Two modes: a full rollout, or `--check-compatibility` for a cheap in-process gate with no subprocess or GPU._

- `REPO_ROOT: Final[Path]` (L39) — Repo root, derived from this file's location.
- `SCENES_DIR: Final[Path] = REPO_ROOT / "scenes"` (L40) — Root of the scanned scene tree.
- `OUTPUT_DIR: Final[Path] = REPO_ROOT / "outputs"` (L41) — Where `audit_sim_configs.json` is written.
- `DEFAULT_TIMEOUT_S = 600` (L42) — Default `--timeout`, 10 min/config — covers cold weights download.
- `DEFAULT_DEPLOY_ALIVE_GRACE_S = 90` (L43) — Default `--deploy-alive-grace`: how long a deploy-mode graph is left running before SIGINT.
- `DEFAULT_DEPLOY_SHUTDOWN_GRACE_S = 30` (L44) — Default `--deploy-shutdown-grace`: how long to wait after SIGINT before escalating to SIGKILL.
- `RunMode = Literal["sim", "benchmark", "deploy"]` (L46) — Tier selector on each `ConfigSpec` row; drives `_run_one` vs `_run_one_deploy` dispatch.
- `@dataclass(frozen=True) class ConfigSpec(config, rskill, uv_group, run_mode)` (L50) — One row in the audit catalogue; `rskill` is empty for deploy rows since the reasoner picks at runtime. Only holds pairs that actually exist in the tree.
- `CATALOGUE: tuple[ConfigSpec, ...]` (L78) — Explicit (YAML → rSkill → uv group → run_mode) mapping for every config currently in the tree: 13 sim + 7 benchmark + 4 deploy = 24 rows.
- `@dataclass class AuditRow(config, rskill, status, exit_code, wall_s, peak_vram_mib, tail)` (L168) — One result. `status` ∈ {`pass`, `pass-compat`, `fail-oom`, `fail-asset`, `fail-sidecar`, `fail-timeout`, `fail-other`, `fail-compat`, `skipped-opt-dep`, `skipped-host-setup`}.
- `_classify(returncode: int, tail: str) -> str` (L219) — Maps a subprocess result to a status by matching stderr against known OOM/asset/sidecar/opt-dep/host-setup patterns; a known MuJoCo/GL exit code is treated as pass when no error pattern appears.
- `class _VramSampler` (L258) — Background `nvidia-smi --query-gpu=memory.used` poller, 200 ms cadence; `peak_mib` reported on `.stop()`. No-op without `nvidia-smi` on `$PATH`.
- `_check_compat(spec: ConfigSpec) -> AuditRow` (L303) — `--check-compatibility` gate: load scene via `openral_core.load_scene_strict`, validate rSkill manifest (sim/benchmark) or assert the robot manifest resolves via `openral_sim.policies.robots.resolve_robot_manifest` and loads (deploy). No subprocess, no GPU. Returns `pass-compat` / `fail-compat`.
- `_build_run_cmd(spec: ConfigSpec) -> list[str]` (L386) — Build the `uv run … openral <sim|benchmark> …` argv for sim/benchmark rows. Refactored out of `_run_one` so the deploy path can stay focused on lifecycle teardown.
- `_run_one_deploy(spec, *, alive_grace_s, shutdown_grace_s, timeout_s) -> AuditRow` (L435) — Tier-2 deploy launch via `openral deploy sim`: runs in its own process group, waits `alive_grace_s`, sends SIGINT to the group, waits `shutdown_grace_s`, then escalates to SIGKILL. Passes when the startup banner appears and the exit reflects a clean or SIGINT/SIGTERM shutdown.
- `_classify_or_fallback(returncode, tail, spec, wall_s, peak_vram) -> AuditRow` (L606) — Deploy-mode wrapper around `_classify` that defaults to `fail-other` when no pattern matches (sim path defaults to `pass`).
- `_run_one(spec: ConfigSpec, timeout_s: int) -> AuditRow` (L645) — Tier-3 sim/benchmark rollout via `_build_run_cmd(spec)` with `MUJOCO_GL=egl` and `OPENRAL_SIM_SEQUENTIAL_INIT=1`.
- `main(argv) -> int` (L783) — CLI entry; flags `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace` / `--check-compatibility` / `--report`. Returns 0 on all-pass, 1 if any config failed, 2 on filter mismatch.

### `tools/validation_matrix.py`
_The four-scene collision-stack validation matrix as one versioned command, emitting both `NOTES.md` and a machine-readable `verdicts.json` per round. Backs `just validation-matrix` / `-verdicts` / `-diff`. See [`docs/contributing/validation-matrix.md`](../contributing/validation-matrix.md) and the ledger it feeds, [`docs/reference/collision-validation-evidence.md`](../reference/collision-validation-evidence.md)._

- `REPO_ROOT: Final[Path]` (L60) — `Path(__file__).resolve().parents[1]`.
- `OUTPUT_ROOT: Final[Path]` (L61) — `REPO_ROOT/"outputs"/"validation-matrix"`.
- `DEFAULT_RSKILL_ID: Final[str]` (L119) — `"OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"`.
- `LAUNCH_FAILED_MARKER: Final[str] = "launch_failed.txt"` (L1089) — Suffix of the runner's own launch-failure marker file, the first thing `detect_launch_failure` checks for.
- `DISPATCH_READY_TIMEOUT_S: Final[float] = 180.0` (L2183) — Timeout bound for the goal re-dispatch loop when the graph answers but is not assembled yet (paired with `DISPATCH_RETRY_INTERVAL_S`).
- `@dataclass(frozen=True) class SceneSpec(key, config, prompt, deadline_s)` — One matrix row; `config` is the tracked DeployScene YAML, and the round launches a resolved copy carrying the seed and CLI-less pins. (L79)
- `MATRIX: tuple[SceneSpec, ...]` — The four scenes: `baguette`, `sink_cup`, `fridge`, `utensil`. (L96)
- `SYNC_GROUPS = ("robocasa", "sidecar-wire")` — Both, always: `--group robocasa` alone strips `pyzmq` and breaks the XR-1 adapter. (L123)
- `STACK_ARGV: tuple[str, ...]` — The flag-pinnable stack: SLAM/Nav2/octomap/kernel-check on, detector + scene VLM off, headless. (L128)
- `SCENE_RUNTIME_PIN: tuple[tuple[str, bool], ...]` — `enable_reasoner=False`: the one pinned knob with no CLI flag, spliced into the resolved scene copy since `deploy sim` defaults it to `True`. (L169)
- `LEGACY_SCENE_DIRS: tuple[tuple[str, tuple[str, ...]], ...]` — Scene key → the directory names pre-harness rounds used, so `import-round` needs no hand-mapping. (L196)
- `quantization_budget_m(grid_resolution_m: float) -> float` — Half the voxel's body diagonal; the largest kernel-vs-ground-truth discrepancy a correct grid can produce. (L207)
- `collision_scale_env() -> dict[str, float]` — The graded-velocity band the round runs with, read from the `OPENRAL_COLLISION_SCALE_*` env vars `deploy_e2e.launch.py` consumes; recorded because an argv-based override check cannot see it. (L1761)
- `parse_kernel_collision(lines) -> ValidationStopEvidence | None` — Transcribe the first `safety.collision` line verbatim. (L227)
- `parse_json_log_line(lines, event) -> dict[str, Any] | None` — Payload of the first `<event> {...}` line (`sim.task_success_final`, `sim.estop_ground_truth_snapshot`, `sim.estop_initial_configuration`). (L285)
- `read_monitor(path) -> list[dict[str, Any]]` — Load a monitor JSONL, skipping non-object lines. (L319)
- `grid_resolution_from_monitor(records) -> float | None` — Cell size from the first `world_voxels` record; the budget is read from the run, never assumed. (L343)
- `monitor_subscription_records(records) -> int` — Record count excluding the monitor's own start/stop markers, i.e. what it actually received. `0` means the monitor's DDS participant missed the run — a harness fault, not evidence the run stopped early. (L366)
- `build_witness_timeline(records, deploy_lines) -> ValidationWitnessTimeline` — Producer side from the monitor, consumer side from the kernel's own `safety.support_witness_*` / `safety.place_region_*` lines. (L388)
- `probe_is_collidability_filtered(snapshot) -> bool` — Whether every non-empty coverage block in the snapshot excludes non-collidable side geoms, i.e. the HAL filtered both sides of each probe to solid geoms. `False` for any snapshot recorded before that filter existed. (L490)
- `hal_admissible_gap_m(snapshot, stop) -> float | None` — The HAL's own kernel-vs-probe budget for a stop, dispatched by stop class. `None` for a snapshot predating the budget; an over-large budget silently excuses a stop rather than failing loudly, which is the direction that hides a real defect. (L558)
- `_payload_world_gap_m(budget, grid_resolution_m) -> float | None` — The payload-vs-world-voxel half of `hal_admissible_gap_m`: model overhang plus the voxel half-diagonal, re-derived when the snapshot's voxel term is zero. Returns `None` (unadjudicated) rather than a number that convicts when it can't be computed.
- `_arm_world_gap_m(budget, grid_resolution_m) -> float | None` — The arm-link-vs-world-voxel half: corner slop of the worst link plus the voxel half-diagonal, re-derived when the snapshot's is zero so an omission can't understate the budget. Falls back to the published composition when a term is absent.
- `_link_link_gap_m(budget, stop) -> float | None` — The link-vs-link half, extracted alongside it so the dispatcher above reads as one line per stop class.
- `probe_is_distance_certified(snapshot) -> bool` (L519) — Whether every non-empty coverage block reports zero uncertified pairs, i.e. distances were measured with the kernel's own convex-distance code rather than MuJoCo's, which can be wrong by tens of millimetres on some pairs. `False` for any round predating this certification check.
- `adjudicate_ground_truth(snapshot, stop, grid_resolution_m, *, monitor_records=None) -> ValidationGroundTruthAdjudication | None` — Distance-probe adjudication ladder: a pair at ≤0 m is `real-contact` only when collidability-filtered; a gap beyond budget is `false-positive` only when the budget is trusted. Every verdict is withdrawn to `unadjudicated` unless the probe is distance-certified. (L768)
- `detect_launch_failure(run_dir, stem, deploy_lines) -> str` — Why a scene is not a run at all: a launch-failure marker, a `ros2 launch` exception, a CLI usage-error banner, no log, a Nav2 bond teardown, or a graph that never came up. Empty when the scene ran; this is what `artifacts_complete` checks. (L1236)
- `_nav2_bond_teardown(deploy_lines) -> str` — A Nav2 bond-heartbeat timeout tears the whole stack down silently, leaving the graph inert until its deadline — which the harness would otherwise score as a policy failure to grasp. Discriminated from a late teardown by how early the loss occurs.
- `_lacks_stage2_hull(link) -> bool` — Whether a link is known to carry no stage-2 hull. Only an explicit `False` counts — an older snapshot without the field must read as unknown, never as "no hull". Used by `_link_link_hull_gap_m`.
- `dispatch_not_ready_reason(goal_log) -> str` (L2187) — Why a dispatch reports the graph as not assembled yet (e.g. a disconnected TF tree or a camera that published nothing), so the goal can be retried instead of scored as a non-completion. Never matches a real E-stop, deadline or capability mismatch. `tests/unit/test_dispatch_readiness.py`.
- `DISPATCH_RETRY_INTERVAL_S: float` (L2184) — `12.0`; how often `dispatch_not_ready_reason` triggers a re-dispatch.
- `_NAV2_BOND_LOSS_EARLY_S: float` (L1129) — `120.0`; a bond-loss timestamp inside this window of the log's start is scored as `_nav2_bond_teardown`, not an ordinary non-completion.
- `_lifecycle_never_came_up(deploy_lines) -> str` — A lifecycle node never completed a transition, so the graph never came up and nothing after it is a policy outcome; bucketed alongside `_nav2_bond_teardown` since both represent absence rather than a real attempt.
- `parse_goal_log(lines) -> tuple[dict[str, Any] | None, str]` — The dispatcher's single JSON status line, or why it never wrote one (e.g. a traceback with no `status` field). (L1306)
- `classify_outcome(*, task_success_ever, stop, ground_truth, initial_configuration, grasped, artifacts_complete) -> str` — Buckets one scene's outcome; ordering is load-bearing — success wins outright, and an initial-configuration E-stop outranks the ground-truth adjudication. (L1347)
- `scene_verdict_from_artifacts(run_dir, *, scene, config_path, seed, prompt, rskill_id, stem) -> ValidationSceneVerdict` — Derive one scene's verdict from recorded artifacts only. (L1401)
- `diff_rounds(current, baseline) -> ValidationRoundDiff` — Field-by-field round comparison; carries both rounds' seed as well as SHA, since a comparison counts as reproducibility only when both match — the seed decides the scene's initial configuration. (L1572)
- `class GuardrailError(RuntimeError)` — A precondition is not met; raised, never warned. (L1631)
- `assert_worktree_clean() -> None` — Refuse a dirty worktree: its recorded SHA would be a lie. (L1644)
- `assert_sha(expected: str | None) -> str` — Return `HEAD`, refusing when it is not the requested checkout. (L1657)
- `assert_overlay_fresh(install_dir: Path) -> int` — Refuse an `install/` older than any tracked `.cpp/.hpp/.h/.msg/.idl` under `cpp/` or `packages/`. (L1678)
- `resolve_launcher() -> Path` — This checkout's `.venv/bin/openral`, by absolute path; the `~/.local/bin` wrapper execs the **parent** checkout. (L1720)
- `assert_sidecar_wire() -> None` — Refuse when `pyzmq` is absent. (L1744)
- `assert_no_safety_overrides(argv) -> None` — Refuse any argv token matching `_SAFETY_KNOB_PATTERNS` (normalised lowercase, `-`→`_`). (L1824)
- `scene_safety_surface(document) -> dict[str, object]` — The safety-relevant keys of a parsed DeployScene: `safety` / `hal` / ACM / place-declaration wholesale, plus any leaf that reads as a margin, tolerance, limit, watchdog or E-stop. Stack composition is deliberately excluded. (L1880)
- `assert_scene_safety_unmoved(tracked, resolved) -> None` — The second control surface: refuse a materialised scene copy that adds, removes or changes any key of `scene_safety_surface` relative to the tracked scene. (L1907)
- `gpu_status() -> tuple[str | None, list[str]]` — GPU name + resident compute processes; the host is shared. (L1945)
- `pin_runtime_block(text, pins) -> str` — Splice `runtime:` pins into a scene YAML, changing nothing else (comments and safety commentary survive verbatim). (L1979)
- `materialise_scene(spec, seed, run_dir) -> tuple[str, Path]` — Write the round's resolved scene copy (seed + `SCENE_RUNTIME_PIN`), re-parse it to prove the pins landed, and check it against the tracked scene. The tracked file is never touched. (L2030)
- `wait_for_dds_transport_ready(deploy_log, proc, *, timeout_s, poll_s=0.05) -> str` — Block until the deploy log shows `dds_transport_ready:` (stale shared-memory clean done, `ros2 launch` not yet spawned), so the monitor joins the graph being launched. Returns `""` on timeout or a dead deploy. (L2115)
- `render_notes(verdicts) -> str` — The round's Markdown summary: names scenes whose monitor received nothing (a harness fault) separately from those that stopped before seeing a voxel grid (a fact about the run), plus any with an uncertified probe or a lower-bound-only budget. (L2431)
- `round_exit_code(verdicts) -> int` — `4` when any scene bucketed `harness-error`, else `0`: a round in which a scene never launched must not exit successfully. (L2560)
- `parse_launch_argv(lines) -> list[str]` — The resolved `argv: … launch …` the deploy CLI echoed: the only artifact stating the stack a run actually got; head-agnostic, so a venv-wrapped `ros2` still parses. (L2652)
- `stack_tokens(argv) -> list[str]` — The stack-defining `key:=value` tokens of that argv, per-scene tokens dropped. (L2695)
- `robot_facts_from_launch_argv(argv) -> dict[str, str]` — `repo_root` / `robot_id` / `robot_manifest_path`, from the argv's `robot_yaml:=` token. (L2714)
- `parse_log_start_time(lines) -> str | None` — UTC timestamp of the log's first ROS stamp; pre-harness rounds recorded no start time, their logs did. (L2743)
- `resolve_scene_dirs(round_dir, aliases) -> dict[str, str]` — Map each matrix scene onto the directory a round kept it in; `--scene-alias` wins over `LEGACY_SCENE_DIRS`. (L2767)
- `cmd_verdicts(round_dir, *, stem=None) -> int` (L2576) — `stem=None` reads the round's recorded `artifact_stem`.
- `cmd_diff(round_dir, baseline_dir, out_path) -> int` (L2627) — `verdicts` subcommand body: field-by-field round comparison via `diff_rounds`.
- `cmd_import(args) -> int` (L2797) — `import-round` subcommand body.
- `cmd_run(args) -> int` (L2886) — `run` subcommand body.
- `main(argv=None) -> int` — CLI entry; `run` / `verdicts` / `diff` / `import-round`. `3` on a guardrail refusal (nothing written), `4` when a scene bucketed `harness-error`. (L2966)
- `octomap_resolution_env() -> dict[str, float]` — The world-voxel resolution the round actually runs with, read from `OPENRAL_OCTOMAP_RESOLUTION_M`; a finer grid is less conservative than the shipped default. Returns `{}` (not a value) when the override is absent, so metadata never misdescribes the run. (L1792)

### `tools/_validation_matrix_monitor.py`
_Private helper of `validation_matrix.py`, spawned alongside each scene's ROS graph. Records the attachment stream, kernel failure evidence, the occupied-cell set hash, the place declaration plus producer-measured region, and periodic `.npz` snapshots. Needs ROS 2 sourced._

- `class Monitor` (L164) — Owns the ROS subscriptions + state for the attachment/failure/place/voxel recording described above; `__init__(node, sink, snap_dir)`.
- `main() -> int` (L434) — CLI: `sys.argv[1]`=output path, `[2]`=snapshot dir; creates the node, runs `Monitor`, spins until SIGINT/SIGTERM, writes `monitor_started`/`monitor_stopped` markers.

### `tools/_validation_matrix_dispatch.py`
_Private helper of `validation_matrix.py`. Sends one `ExecuteRskill` goal at the live graph (the matrix runs with the reasoner off) and prints one JSON line — the run's `goal.log`. Needs ROS 2 sourced._

- `main(argv=None) -> int` — `--deadline-s` / `--rskill-id` / `--prompt` / `--server-timeout-s`. (L24)

### `tools/_ceiling_probe.py`
_One worker of the ceiling battery: one scene, one gate arm, N rounds. Bypasses `validation_matrix.py`'s refusal to disable the octomap check, while reusing its scene materialisation, readiness gate and dispatch tool. Answers: of the runs the kernel stops, how many would have succeeded anyway?_

- `DEADLINE_S: float = 420.0` (L56) — Deadline per scene run; the harness's own default, kept identical so a gate-off run is not given more time to succeed than a gate-on one.
- `_reap_domain(domain, sig) -> int` — Teardown sweep for orphan graph nodes `killpg` can't reach, since `ros2 launch` starts the octomap pair in its own session and each orphan holds a Fast-DDS shm lock that starves the next round on that domain. Scoped by `ROS_DOMAIN_ID`; SIGTERM first so segments are unlinked cleanly.
- `run_one(spec, gate, seed, run_dir, rskill) -> dict[str, object]` (L199) — Launch one scene end to end and return its record, including host load, delivered action chunks, and whether a Nav2 bond teardown occurred, so a starved run is legible from the record alone.
- `_chunks_from_goal_log(goal_log) -> int | None` — Action chunks the policy delivered inside the deadline — the single number showing whether a run got a fair trial; a starved or torn-down run delivers a small fraction of a healthy one's rate.
- `_bond_teardown(deploy_log) -> str` — Delegates to `validation_matrix._nav2_bond_teardown` so the probe and the harness agree on what a voided run looks like.
- `main(argv: list[str] | None = None) -> int` (L343) — CLI entry: `--scene/--gate/--rounds/--seed/--rskill/--out/--start-round`; drives `run_one` per round, appends to `records.json`, prints a running success tally.

### `tools/ceiling_battery.sh`
_Drives the eight `_ceiling_probe.py` workers (4 scenes × 2 gate arms), interleaved by scene so the two arms run under identical host conditions. `WORKERS` caps concurrency; needs `OPENRAL_SKIP_ORPHAN_REAP=1` so a worker's startup sweep can't kill a concurrent sibling's live graph — the domain-scoped `_reap_domain` in `_ceiling_probe.py` keeps the host clean between rounds instead._

### `tools/_nav2_mppi_loop_probe.py`
_Times the whole MPPI control loop against its 50 ms budget on a live `openral deploy sim` graph, dividing `controller_server`'s CPU time by the cycles it published, in sim time since the stack runs `use_sim_time`. Validates which footprint polygon the costmap adopted per run. Private evidence tool, not a shipped entry point; recorded in [`docs/reference/robocasa-carry-survey.md`](../reference/robocasa-carry-survey.md)._

- `BARE` (L53) — the manifest chassis footprint polygon.
- `GROWN` (L54) — the chassis footprint grown over the payload at the live 0.860 m forward reach.
- `CLK = os.sysconf("SC_CLK_TCK")` (L55) — clock ticks per second, for converting `/proc/<pid>/stat` fields to seconds.
- `controller_pid() -> int` (L58) — `pgrep` the live `nav2_controller/controller_server` PID; `SystemExit` if not running.
- `cpu_seconds(pid) -> float` (L71) — utime+stime from `/proc/<pid>/stat`, the clock-source-independent cost.
- `class Probe(Node)` (L77) — publishes the arm's polygon at 20 Hz, counts `/cmd_vel_nav`, and reads back `/local_costmap/published_footprint` for the validity check.
- `main(argv=None) -> int` (L129) — `--grown`, `--seconds=N`; emits one JSON line carrying `cpu_ms_per_cycle`, `cycles`, `footprint_len_m` and `arm_valid`.

### `tools/_nav2_costmap_silhouette_probe.py`
_Sweeps a live graph's Nav2 costmaps for cells marked inside the robot's own silhouette or a carried payload's, which neither costmap should ever do now that Nav2 is base-only — the scan filter is the only thing preventing it. Private, like `_nav2_mppi_loop_probe.py`: attaches to an already-launched graph, evidence tooling rather than a shipped entry point._

- `LETHAL_OBSTACLE = 254` (L68) — `nav2_costmap_2d`'s LETHAL_OBSTACLE; deliberately not 253 (INSCRIBED_INFLATED_OBSTACLE) since the claim is that no cell inside the robot was **marked**, and 253 is what an inflation layer writes near a legitimate obstacle elsewhere.
- `payload_mask(objects, base_from, points_xy) -> tuple[Any, int, tuple[float, float] | None]` (L103) — Projects every attached object onto the costmap plane; returns `(union mask, objects fully placed, z span in base frame)`. Counts per object (not primitive) — an object counts as placed only when every one of its primitives projected.
- `class SilhouetteProbe(Node)` (L158) — Samples both costmaps and counts marked cells inside the robot silhouette.
- `main() -> int` (L351) — argparse (`--robot-yaml`, `--scene`, `--seconds`, `--settle`, `--drive`); spins the probe, optionally drives a `NavigateToPose`, emits a verdict dict with non-vacuity guards (`lethal_cells_anywhere`, `base_travel_m`, `payload_samples`, etc).

### `tools/isaac_sidecar.py`
_Isaac Sim scene sidecar, running Isaac Lab/Kit in its own py3.11 venv, auto-spawned by `openral_sim.backends.isaac_sim`. Launches Omniverse Kit headless, builds the scene named by `--layout`, and serves ZMQ REP + msgpack framing (`ping`/`reset`/`step`/`render`/`close`)._

- `main(argv: list[str]) -> int` (L133) — Checks required dep versions (before the ~50 s Kit boot), launches `SimulationApp`, then imports and builds the scene named by `--layout` (`lift`/`bowl_plate`/`manifest`) and serves the ZMQ loop.

### `tools/_isaac_scene_base.py`
_Shared base for the Isaac Sim sidecar scenes (py3.11 venv only), owning the obs/step lifecycle, RGBA→HWC frame grabbing, the warmup + physics-substep loop, and eval-layer observation assembly, so a new layout only overrides a few template methods._

- `franka_joint_positions(franka: Any) -> NDArray[np.float32]` (L51) — Franka joint angles in manifest order (8 = 7 arm + gripper).
- `franka_joint_velocities(franka: Any) -> NDArray[np.float32]` (L56) — Franka joint velocities in manifest order (8 = 7 arm + gripper).
- `class IsaacSceneBase` (L61) — Lifecycle + obs skeleton common to the Isaac Sim sidecar scenes; class attrs `warmup_steps` / `physics_substeps`. Template methods `build`/`_apply_action`/`_images`/`_state`/`_reward_terminated` are overridden per scene.
  - `IsaacSceneBase.build() -> None` (L154) — Template method (raises `NotImplementedError`): construct the stage (robot, props, cameras, controllers).
  - `IsaacSceneBase.reset(seed: int | None = None) -> dict[str, Any]` (L91) — Per-episode reset: randomize, reset physics, warm up, observe.
  - `IsaacSceneBase.render() -> NDArray[np.uint8] | None` (L149) — Last grabbed RGB frame, or `None`.
  - `IsaacSceneBase.sim_time_ns() -> int | None` (L127) — Elapsed sim time in ns for `/clock`; prefers Isaac's `SimulationContext.current_time`, else integrates step count × physics dt.
  - `IsaacSceneBase.step(action: NDArray[np.float32]) -> dict[str, Any]` (L101) — Apply one action, advance physics (renders only the final substep), return a StepResult dict (`observation`/`reward`/`terminated`/`truncated`/`info`/`sim_time_ns`).

### `tools/isaac_scene.py`
_Minimal Isaac Sim lift-cube scene for the sidecar, built on Isaac Sim core (not Isaac Lab). Franka on a ground plane, a cube in front, two RTX cameras matching the LIBERO contract. Lifecycle/obs skeleton in `_isaac_scene_base.IsaacSceneBase`._

- `_ARM_DOF = 7` (L36) — arm joint count.
- `_ACTION_DIM = 8` (L37) — 7 arm joint deltas + 1 gripper command.
- `_ARM_DELTA_SCALE = 0.05` (L38) — rad per unit action, keeps a unit action sane.
- `_GRIPPER_OPEN = 0.04` (L39) — Franka finger joint upper bound (m).
- `_GRIPPER_CLOSED = 0.0` (L40) — Franka finger joint lower bound (m).
- `_LIFT_SUCCESS_Z = 0.10` (L41) — cube CoM height (m) counted as "lifted".
- `_CUBE_HALF = 0.025` (L42) — 5 cm cube → 2.5 cm half-extent.
- `_AGENT_CAMERA_POS: NDArray[np.float64]` (L43) — front agent-view camera position.
- `class IsaacLiftScene(IsaacSceneBase)` (L46) — A real Isaac Sim PhysX + RTX lift-cube scene, driven step-by-step.
  - `IsaacLiftScene.build() -> None` (L62) — Construct the stage: ground + Franka + cube + camera.

### `tools/isaac_bowl_plate_scene.py`
_Isaac Sim table + bowl + plate Franka scene with the LIBERO obs/action contract, built on Isaac Sim core: arm motion via `ArticulationController`, end-effector control via the core Lula IK solver. Mirrors the LIBERO contract so act-libero/smolvla-libero drive it unchanged._

- `_ARM_DOF = 7` (L38) — arm joint count.
- `_ACTION_DIM = 7` (L39) — LIBERO OSC-pose delta: dpos(3) + drot(3) + gripper(1).
- `_POS_SCALE = 0.03` (L40) — metres per unit position-delta action.
- `_GRIPPER_OPEN = 0.04` (L41) — Franka finger joint upper bound.
- `_GRIPPER_CLOSED = 0.0` (L42) — Franka finger joint lower bound.
- `_TABLE_TOP_Z = 0.0` (L43) — table surface at the robot base height.
- `_OBJ_Z = 0.03` (L44) — object resting height.
- `_TASK_CENTER: NDArray[np.float64]` (L45) — nominal bowl/plate task centre.
- `_WORKSPACE_LOW: NDArray[np.float64]` (L46) — randomization box lower bound for object placement.
- `_WORKSPACE_HIGH: NDArray[np.float64]` (L47) — randomization box upper bound for object placement.
- `_AGENT_CAMERA_POS: NDArray[np.float64]` (L48) — front agent-view camera position.
- `_SECONDARY_CAMERA_POS: NDArray[np.float64]` (L49) — secondary prop-view camera position.
- `class IsaacBowlPlateScene(IsaacSceneBase)` (L84) — Table + bowl + plate scene, LIBERO two-camera contract.
  - `IsaacBowlPlateScene.build() -> None` (L101) — Construct the stage: table, Franka, bowl, plate, Lula IK solver, two cameras.

### `tools/isaac_manifest_scene.py`
_Robot-agnostic, URDF-driven Isaac Sim scene, unlike the PoC scenes that hardcode Isaac's Franka asset: imports the manifest robot's URDF and wires joints/sensors/control from a JSON robot spec marshaled across the venv boundary. This cut (M1): fixed-base arm, one RGB camera, JOINT_POSITION-delta control._

- `_GRIPPER_DEADBAND = 1e-3` (L43) — Below this magnitude a gripper action channel means HOLD, letting a pure `BODY_TWIST` step (zero arm/gripper slots) leave the gripper alone.
- `_SCAN_MAX_SELF_SKIPS = 8` (L50) — Self-occlusion handling for the lidar fan (mirrors `openral_sim.backends.robocasa._LASER_MAX_SELF_SKIPS`): caps how many self-layers one beam steps through before giving up and reading `range_max_m`.
- `_SCAN_SELF_SKIP_EPS_M = 1e-3` (L53) — Nudge (m) added past a self-hit before re-casting.
- `map_dof_to_manifest(values, *, dof_index, manifest_joints, finger_dof_idx, base_values=None, base_joints=None) -> NDArray[np.float32]` (L56) — Maps an Isaac articulation DOF vector to the full manifest joint order; a base joint reads `base_values`, a gripper joint the mean of its finger DOFs, everything else its matching URDF DOF or `0.0`.
- `resolve_beam_range(raycast_closest, *, origin_xy, angle_rad, z, range_min_m, range_max_m) -> float` (L96) — One lidar beam's range, with the robot's own body skipped by rigid-body-path identity rather than assumed clear; re-casts past a self-hit up to a bounded number of times. Pure, so unit-testable without a Kit app.
- `class IsaacManifestScene(IsaacSceneBase)` (L162) — URDF-driven, robot-agnostic Isaac scene built from the marshaled robot spec.
  - `IsaacManifestScene.build() -> None` (L280) — Import the manifest robot's URDF, wire joints/sensors/control, place cameras per `_plan_cameras`.

### `tools/robocasa_carry_survey.py`
_Answers whether any RoboCasa task makes the base drive while holding something, by building the env and measuring the distance from the base's start pose to every manipulated object — not by reading the source. See [`docs/reference/robocasa-carry-survey.md`](../reference/robocasa-carry-survey.md)._

- `REQUIRED_BASE_TRANSLATION_M = 1.0` — Criterion 1 of the scene that closes #108 (`openral_nav2_bringup` README); below it the arm bridges the gap and `NavigateToPose` never runs. (L37)
- `DISTRACTOR_PREFIXES = ("distr", "dstr")` — RoboCasa's two clutter-object spellings; both are needed, or some tasks report a distractor as their furthest object. (L44)
- `class SurveyError(RuntimeError)` — RoboCasa unavailable, or its task registry unreadable. (L47)
- `@dataclass(frozen=True) class CarryMeasurement(task, seed, layout, style, furthest_object_m, furthest_object, nearest_object_m, nearest_object, lang)` — One `(task, seed)` reset, measured; `.requires_base_translation` applies the threshold. (L52)
  - `CarryMeasurement.requires_base_translation` (L70) — `@property`; `furthest_object_m > REQUIRED_BASE_TRANSLATION_M`.
- `robocasa_root(explicit=None) -> Path` — `--robocasa-root`, else an importable `robocasa`, else the HAL's provisioning cache. (L74)
- `read_target50(root) -> list[str]` — The 50 task names XR-1's model card pins as its reference configuration, parsed from `dataset_registry.py`. (L95)
- `measure(task, seed) -> CarryMeasurement` — Build, reset, measure each object's distance from `mobilebase0_base`. Needs a provisioned RoboCasa. (L120)
- `main(argv=None) -> int` — CLI; `--tasks` (default: target50), `--seeds`, `--robocasa-root`, `--json`. Per-`(task, seed)` build failures are reported, not fatal. (L175)

### `tools/select_tests.py`
_Selective test execution — maps a git diff to the minimal pytest targets that can observe it. Backs `just test-changed` / the `test-selective` workflow. See [`docs/contributing/selective-testing.md`](../contributing/selective-testing.md)._

- `REPO_ROOT` (L39) — Repo root, derived from this file's location.
- `class CapabilityGap(BaseModel)` (L56) — A capability the CI runner provably cannot provide: `summary`, `satisfied_by`, `skip_patterns` (fnmatch globs matched against a skip's reason).
- `class SelectionConfig(BaseModel)` (L70) — Typed view of `tools/test_selection.toml`: `full_run_globs`, `ignore_globs`, `isolate_globs`, `extra_triggers`, `requirement_globs`, `capability_gaps`.
- `class SelectionResult(BaseModel)` (L81) — `full_run` / `full_run_reason` / `affected_packages` / `targets` / `isolated_targets` (own-process, issue #24) / `requirement_targets` (per opt-in lane) / `reasons` (per-target rationale).
- `load_config(path) -> SelectionConfig` (L105) — Load + validate the TOML config.
- `package_dir_import_names(repo_root) -> dict[str, str]` (L117) — `python/<dir>` → its `src/openral_*` import name.
- `build_dependency_graph(repo_root) -> dict[str, set[str]]` (L136) — Import-name → direct `openral` deps, derived from each `pyproject.toml` (never hand-written).
- `transitive_dependents(graph, changed) -> set[str]` (L160) — Closure of packages that depend on any changed package (includes `changed`).
- `map_test_imports(repo_root) -> dict[str, set[str]]` (L187) — Each top-level `tests/` file → the `openral_*` packages it imports.
- `select(repo_root, changed_files, config) -> SelectionResult` (L346) — Resolve changed paths to pytest targets (blast-radius → full run; else per-package dirs + import-intersecting tests), peeling `isolate_globs` matches into `isolated_targets`.
- `changed_files_from_git(base, head, repo_root) -> list[str]` (L462) — Merge-base `git diff --name-only base...head`.
- `main(argv=None) -> int` (L502) — CLI; `--files` / `--base/--head`, `--github-output` for CI step outputs.

### `tools/lane_report.py`
_Opt-in lane accounting — decides, records and attests what each dependency lane actually ran. Makes a vacuous green impossible (issue #163). See [`docs/contributing/selective-testing.md`](../contributing/selective-testing.md)._

- `REPO_ROOT: Path` (L43) — Repo root, derived from `__file__`.
- `DEFAULT_CONFIG: Path` (L53) — `REPO_ROOT / "tools" / "test_selection.toml"`.
- `DEFAULT_LEDGER: Path` (L54) — `REPO_ROOT / ".lane-ledger.jsonl"`.
- `STATUS_RAN: str` (L58) — Status vocabulary. Deliberately not "skipped": a lane is either coverage got, coverage declared unreachable here, or a failure.
- `STATUS_DECLARED_NOT_RUN: str` (L59) — A lane whose every test is gap-explained.
- `STATUS_FAILED: str` (L60) — Any lane with an undeclared skip, or that collected no tests at all.
- `class LaneRecord(BaseModel)` (L68) — One lane's accounted outcome: `lane` / `status` (`ran` / `declared-not-run` / `failed`) / `passed` / `failed` / `declared_skips` (gap → count) / `undeclared_skips` / `note`.
  - `declared_total` (L80) — How many tests were not run because of a declared capability gap; sum of `declared_skips.values()`.
- `read_junit(paths) -> tuple[int, int, list[str]]` (L111) — Aggregate junit XML reports into `(passed, failed, skip_reasons)`; a lane may emit a batched report plus one per isolated file.
- `build_record(lane, *, passed, failed, skip_reasons, config, exit_code=None, from_exit_code=False) -> LaneRecord` (L136) — Apply the lane policy: declared capability gaps are allowed and attributed, every other skip fails, an all-gated lane is `declared-not-run` and an empty one fails.
- `main(argv=None) -> int` (L351) — CLI; `lane` (account one lane's reports) / `attest` (cross-check the ledger against the selection, write the job summary).

### `tools/heavy_lanes_trigger.py`
_Starts the required `heavy-lanes` check on a PR automatically once the PR is ready — no approval, no label. Run by `.github/workflows/heavy-lanes-trigger.yml`. See [`docs/contributing/development.md`](../contributing/development.md#review-policy)._

- `API` (L65) — `"https://api.github.com"`.
- `LANES_WORKFLOW` (L66) — `"heavy-lanes.yml"`, the workflow it dispatches.
- `OWN_CHECK_NAMES` (L70) — Check runs `heavy-lanes.yml` posts itself (`lanes-select`, `lane`, `heavy-lanes`) plus the trigger workflow's own job (`heavy-lanes-trigger`, posted on the PR head by a `workflow_run` sweep); never a precondition. Kept equal to both workflows' job names by `tests/unit/test_heavy_lanes_trigger.py`.
- `OWN_CHECK_PREFIXES` (L71) — `("lane (",)`: matrix legs of the `lane` job.
- `GREEN_CONCLUSIONS` (L72) — `success`, `skipped`, `neutral`.
- `DEFAULT_REQUIRED_CHECKS` (L73) — Checks that must have reported green before dispatch: `select-and-test`, `quality`, `Verify Signed-off-by`.
- `class GitHubAPIError(RuntimeError)` (L76) — A 200 response that is unusable: GraphQL `errors` payload or null `data` (rate limit, missing scope, vanished PR).
- `class CheckRunState(BaseModel)` (L80) — One check run on the PR head: `name` / `status` / `conclusion`.
- `class LaneRunState(BaseModel)` (L88) — One `heavy-lanes.yml` run for the PR head: `status` / `conclusion`.
- `class PullRequestState(BaseModel)` (L95) — Everything the verdict reads: draft, same-repo, `behind_by`, checks, commit statuses, unresolved threads, existing lane runs.
- `class Verdict(BaseModel)` (L111) — `ready` + the `reasons` it is not.
- `readiness(pr, required_checks) -> Verdict` (L122) — Pure verdict: not draft, not a fork, 0 behind base, every other check green and the required ones present, statuses `success`, threads resolved, no non-cancelled lane run for this head yet.
- `class GitHub` (L157) — Minimal REST + GraphQL client over `urllib` (the network boundary).
  - `GitHub.get(path)` (L174) — GET `/repos/<repo>/<path>`, decoded JSON.
  - `GitHub.paginate(path, key=None) -> list` (L178) — Every page of a list endpoint.
  - `GitHub.unresolved_threads(number) -> int` (L191) — Unresolved review threads, via GraphQL.
  - `GitHub.fetch(pr) -> PullRequestState` (L220) — Gather one PR's state.
  - `GitHub.dispatch(pr) -> None` (L243) — `workflow_dispatch` `heavy-lanes.yml` on the PR branch.
- `main(argv=None) -> int` (L258) — CLI; `--repo`, `--pr` (repeatable), `--require-check`, `--dry-run`. Re-reads the head SHA right before dispatch and defers if it moved. One PR's API trouble is a `::warning::`, never a failed sweep.

### `tools/audit_tests.py`
_Test-suite auditor — flags dead / shadowed / duplicate / no-assertion tests; writes `docs/contributing/test-audit.md`. Read-only; never deletes. Backs `just test-audit`._

- `REPO_ROOT` (L39) — Repo root, derived from this file's location.
- `REPORT_PATH = REPO_ROOT / "docs" / "contributing" / "test-audit.md"` (L40) — Committed report path written by `--write-report`.
- `class TestFuncInfo(BaseModel)` (L67) — One `test_*` function: `path`, `qualname` (class-scoped), `markers`, `has_assertion`, `is_trivial`, `body_hash`, …
- `class DuplicateGroup(BaseModel)` (L82) — A `body_hash` shared by ≥2 functions and their `members`.
- `class AuditReport(BaseModel)` (L87) — Inventory (`by_tier`/`by_marker`/`by_directory`) plus `trivial` / `shadowed` / `no_assertion` / `duplicate_groups`.
- `collect(repo_root) -> list[TestFuncInfo]` (L206) — Scope-aware AST walk over every test root (same name in two classes is not conflated).
- `build_report(records) -> AuditReport` (L239) — Group into the inventory + finding buckets; `shadowed` = same `(path, qualname)` redefined (earlier def is dead).
- `render_markdown(report) -> str` (L291) — Render the committed report.
- `main(argv=None) -> int` (L378) — CLI; `--json` / `--write-report`.

### `python/observability/src/openral_observability/replay/`
_Query-time joiner for rosbag2 (mcap) ↔ OTel spans. Backs `openral replay` + `openral record`._

- `@dataclass(frozen=True) class BagMessage(topic, log_time_ns, publish_time_ns, trace_id, traceparent, schema_name, payload_summary)` (bag_reader.py L84) — One mcap record surfaced to the correlator.
- `read_bag(bag_path: str | Path) -> Iterator[BagMessage]` (bag_reader.py L144) — Iterate an `.mcap` file or rosbag2 directory; extracts `trace_id` from `jsonschema`-encoded payloads (a packed W3C `traceparent`, or a raw trace/span id pair for the `Tick` schema) or `ros2msg` CDR payloads via regex match. No `rosbag2_py` dep.
- `class TraceQueryError(RuntimeError)` (trace_query.py L26) — Raised on a non-JSON or unreachable dashboard response.
- `@dataclass(frozen=True) class DashboardTraceClient(base_url="http://127.0.0.1:8000", timeout_s=5.0)` (trace_query.py L31) — `list_traces() -> list[dict]` over `/api/traces`; `get_spans(trace_id) -> list[dict]` over `/api/spans/<id>`.
- `@dataclass(frozen=True) class TimelineEntry(kind, ts_ns, trace_id, topic, span_name, attrs, duration_ms)` (correlator.py L27) — One row of the joined timeline; `.to_json()` returns a plain dict.
- `list_bag_trace_ids(bag_messages) -> list[dict]` (correlator.py L68) — Distinct trace_ids in the bag with counts, busiest first.
- `build_timeline(bag_messages, spans, *, trace_id=None) -> list[TimelineEntry]` (correlator.py L95) — Pure join. Filters both inputs to `trace_id`, merges, sorts ascending by `ts_ns`.
- `RECORD_PROFILES: dict[str, dict[str, list[str]]]` (cli.py L52) — Slim and full topic + regex + topic-type presets (`full` records `LaserScan` / `PointCloud2` / `Imu` by type).
- `build_record_command(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=()) -> list[str]` (cli.py L102) — Compose `ros2 bag record` argv.
- `@dataclass(frozen=True) class ReplayResult(trace_id, bag_trace_ids, timeline, bag_path)` (cli.py L156) — `.to_json()` returns a plain dict.
- `run_replay(*, bag_path, trace_id, dashboard_url) -> ReplayResult` (cli.py L185) — Read a bag, fetch matching spans from the dashboard, return the joined timeline.
- `run_record(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=(), dry_run=False) -> tuple[list[str], CompletedProcess | None]` (cli.py L233) — Spawn `ros2 bag record` in a new process group; forwards SIGINT/SIGTERM received by the parent as **SIGINT** to the child group so rosbag2 flushes `metadata.yaml` cleanly. Waits up to 5 s after the child exits for that file to appear.
- `write_timeline(result: ReplayResult, out_path: Path) -> None` (cli.py L306) — Persist the timeline JSON.

### `tools/rskill_publisher.py`
_Package and publish a local rSkill directory to the HF Hub._

- `_REPO_ROOT: Path` (L59) — Repo root, derived from `__file__`.
- `_REQUIRED_FILES: list[str]` (L82) — `["rskill.yaml"]`, the minimum a directory must contain to be a candidate rSkill.
- `public_visibility_error(manifest, public) -> str | None` (L85) — License gate (pure, no network): returns an error string when `--public` is requested for a non-commercial-licensed skill, so `main` can fail fast before any HF call.
- `_resolve_token(token_arg) -> str` — Prefer CLI arg, fall back to env. (L130)
- `_validate_manifest(skill_dir) -> RSkillManifest` (L149)
- `_validate_docs(skill_dir, manifest) -> DocValidationReport` — Print + return the README / manifest documentation report via `_rskill_doc_validator.validate_rskill_docs`. Runs in both dry-run and `--publish` paths; the caller decides whether to exit on errors.
- `_rewrite_manifest_name(manifest_path, old_name, new_name) -> None` — Rewrite the top-level `name:` value of `rskill.yaml` in place (column-0 line only, so nested `name:` keys are untouched; preserves quotes + trailing comment). Exits 1 if the line isn't found exactly once. Backs `--fix-name`.
- `_enforce_repo_name(skill_dir, manifest, *, fix_name) -> RSkillManifest` — Enforces the ratified rSkill naming grammar; no kind is exempt. `fix_name=True` rewrites to the expected name and reloads, `fix_name=False` hard-fails printing the suggested name.
- `_bump_revision(manifest_path, weights_uri_base, token) -> str` — Resolve latest weights commit, patch `rskill.yaml`. (L337)
- `_ensure_private(api, repo_id) -> None` — Abort if the repo is public. (L388)
- `_ensure_public(api, repo_id) -> None` — The `--public` counterpart: abort if the (reused) repo is private, so a `--public` publish never lands in a private repo.
- `_publish(skill_dir, manifest, token, *, public=False) -> str` — Create the HF repo (private unless `public`) and upload; runs the matching visibility gate (`_ensure_public` / `_ensure_private`) after `create_repo`.
- `main() -> None` (L529) — Entry point: validate manifest → enforce repo name → validate task space → validate docs → license-visibility gate → optional `--bump-revision` → `--publish` (private unless `--public`).

### `tools/voxel_transport_probe.py`

- `RADIUS_M: float` (L57) — the shipped coverage radius (1.05 m), so grid sizes are the deployed ones.
- `TARGET_HZ: float` (L58) — the publish rate the sweep drives each role at.
- `WARMUP: int` (L59) — messages discarded before timing starts, so cold-start latency does not skew the measured rate.
- `qos() -> QoSProfile` (L62) — the kernel's own `/openral/world_voxels` profile: `RELIABLE`, `KEEP_LAST(1)`, `VOLATILE`.
- `per_axis(res: float) -> int` (L70) — cells per axis at that resolution.
- `make_msg(res: float) -> tuple[Any, int]` (L74) — a full-size `OccupancyVoxels` at that resolution.
- `run_pub(res, count)` (L83) — publisher role, emitting JSON; run as a separate **process** so intra-process short-circuiting cannot hide the transport.
- `run_sub(res, count)` (L108) — subscriber role, emitting JSON; likewise a separate process.
- `run_sweep(resolutions, count) -> int` (L142) — drives both roles per resolution and prints the table, reporting the RMW measured.
- CLI: `uv run python tools/voxel_transport_probe.py sweep [--resolutions ...] [--count N]`. Needs a sourced ROS 2 overlay.

Measures the wire cost of the dense `uint8[]` payload as publish→receive latency, i.e. map staleness. Result is transport- and host-specific.

### `tools/depth_extrinsic_check.py`

- `check(args) -> int` (L357) — Reads the depth cloud, the camera-internal TF (`frame_id -> cloud frame`) and, when the sensor's `parent_frame` is not the manifest's `base_frame` (G1 head on `torso_link`, SO-100/101 wrist on `gripper`, Galaxea A1 wrist on `arm_seg6`), the recorded `base_frame -> parent_frame` TF chain at each cloud's stamp — refusing if that chain moved during the recording or is missing. Places the cloud through the `--sensor` pose a deploy of `--unit` publishes (the manifest entry with that `RobotUnit`'s `SensorOverlay` applied; a robot that ships `units/` requires `--unit`, via `_unit_description`), fits the table plane and marker centroids in the base frame, and writes a JSON report (residuals, pass/fail, `base_frame`, `parent_in_base_xyz_rpy`, and `suggested_static_transform_xyz_rpy` in `parent_frame`, to be copied into the unit overlay — or the manifest, for a robot without units; the report records the unit). Returns 0 iff it passes against the `openral_core.depth_extrinsic` limits.
- `verify(args) -> int` (L430) — `openral_core.depth_extrinsic.verify_extrinsic_report` for one sensor: 0 iff the report passed, at criteria no looser than the shipped limits, for the same unit and that unit's *current* pose. `openral deploy run` applies the same check itself.
- `main(argv=None) -> int` (L453) — CLI: `check --robot --sensor --bag --cloud-topic --table-z --table-roi --marker X Y ...` / `verify --robot --sensor [--unit] [--report]` (`--sensor` required; `--unit` selects `robots/<id>/units/<unit>.yaml`; report defaults to `robots/<id>/calibration/<unit>/<sensor>_extrinsic.json`, or `calibration/<sensor>_extrinsic.json` without units). RGB-only sensors are refused (exit 2): no cloud to fit. Needs a sourced ROS 2 overlay (rosbag2_py, tf2_ros).

Measures the one input the kernel's world-voxel check trusts absolutely on a real depth camera — the extrinsic — which `openral calibrate camera` (intrinsics only) does not. Runbook: `docs/tutorials/deploy/openarm-real-world-voxel-check.md`. Tested in `tests/unit/test_depth_extrinsic_check.py` on real rosbag2 bags (OpenArm `head_zed`, G1 `head`, SO-101 `wrist`; `--unit` on a unit overlay).

### `tools/openarm_world_voxel_run.sh`

_The only sanctioned launcher for `scenes/deploy/openarm_real_world_voxels.yaml`. Refuses unless `OPENRAL_OPENARM_ALLOW_MOTION=1` and `OPENRAL_OPENARM_ATTENDED=1`, `OPENRAL_ROBOT_UNIT` naming the cell, sourced ROS 2, `openral` on PATH, `depth_extrinsic_check.py verify --sensor head_zed --unit $OPENRAL_ROBOT_UNIT` passing against `robots/openarm/robot.yaml` + `robots/openarm/calibration/<unit>/head_zed_extrinsic.json` (early refusal; `openral deploy run` re-applies the gate), and an interactive terminal; then asks for a typed confirmation and execs `openral deploy run`. Extra args pass through._

### `tools/stop_ee_speed.py`

- `REPO_ROOT: Path` (L49) — Repo root, derived from `__file__`.
- `EE_BODY: str` (L52) — `link7`, the body the payload attaches to, so its linear velocity is the carried object's.
- `QUANTISATION_GAIN_M: float` (L56) — what 25 → 15 mm recovers (the two cells' half-diagonal difference), the figure staleness cost is weighed against.
- `class StopSpeed(NamedTuple)` (L63) — `round_id`, `stop_class`, `ee_speed_mps`, `base_speed_mps`.
  - `StopSpeed.world_speed_mps` (L72) — adds the base contribution worst-case-aligned.
  - `StopSpeed.is_carry` (L77) — selects `attached_payload` stops.
- `collect(round_dirs) -> list[StopSpeed]` (L100) — End-effector speed at each stop, from the round's recorded joint state via `mj_jacBody` on the real Panda model. Raises on an arm-joint name mismatch, since a silent mismatch would read as a stationary arm.
- `summarise(stops) -> dict[str, Any]` (L176) — per-class medians and maxima, and the net millimetres at the two staleness figures the wire probe measured.
- `render(stops, summary) -> str` (L195) — Render the summary as the human-readable table.
- `main(argv=None) -> int` (L226) — CLI entry. `uv run python tools/stop_ee_speed.py <round dirs...> [--json]`. Needs MuJoCo and the robosuite Panda assets; reads recorded artifacts only.

Settles the 25 → 15 mm trade by measuring real stop speeds rather than assuming worst case.

### `tools/resolution_ab.sh`

_The 25 mm vs 15 mm world-voxel A/B. Both arms are gate-ON; the only difference is `OPENRAL_OCTOMAP_RESOLUTION_M`, recorded so a round can never be mistaken for the other. The primary endpoint is the paired per-stop over-approximation, not completion rate, which is under-powered at battery scale (`tools/round_power.py`)._

_Runs one graph at a time, alternating arms round by round, since scene-interleaved concurrent lanes broke pairing and exhausted host memory. `ROUNDS` and `SIDECAR_BOOT_S` are the knobs; `RUN_ROUND_CMD` is a test seam. `tests/unit/test_resolution_ab_pairing.py`._

### `tools/resolution_ab_report.py`

- `HALF_DIAGONAL_MM: dict[str, float]` (L38) — resolution string → cell half-diagonal in mm (`0.025` → 21.65, `0.015` → 12.99); the PRIMARY section's per-stop budget.
- `_stops(arm_dir) -> list[dict[str, Any]]` — every stop under one arm, with its certified truth when the run recorded one, plus the run's `started_at` / `wall_s` window.
- `main(argv=None) -> int` (L99) — CLI over the A/B's output root; writes `report.json` beside it.

_Pure and offline, so a battery run on one commit can be re-reported by another. Four sections: PRIMARY (paired per-stop over-approximation), INTEGRITY (flags an arm that silently ran at the wrong resolution), PAIRING (flags rounds that ran too far apart to pair), and SECONDARY (completion counts, explicitly under-powered). `tests/unit/test_resolution_ab_report_pairing.py`._

### `tools/stop_excess.py`

- `PAYLOAD_PREFIX: str` (L41) — `"attached:"`, the party-name prefix identifying a payload side in a stop record.
- `class Excess(NamedTuple)` (L44) — one stop's decomposition (`round_id`, `scene`, `party`, `is_payload`, `reported_depth_m`, `certified_gap_m`, `voxel_half_diagonal_m`, `verdict`).
  - `Excess.excess_m` (L57) — `certified_gap − reported_depth`.
  - `Excess.beyond_voxel_m` (L62) — subtracts the half-diagonal.
- `half_diagonal(resolution_m: float) -> float` (L67) — half a cubic cell's body diagonal, the grid's worst-case error.
- `collect(round_dirs: list[Path]) -> tuple[list[Excess], list[str]]` (L72) — Certified stops plus the reasons for every skip. Reads each round's own grid resolution; a stop without one is skipped, never defaulted, since the half-diagonal is the quantity being subtracted.
- `summarise(stops, skipped) -> dict[str, Any]` (L119) — per-class `n`, median excess, median beyond-voxel, and `has_geometry_headroom` (strict `> 0`: a class exactly at the voxel term has none).
- `render(stops, summary) -> str` (L141) — the per-stop table plus the per-class verdict.
- `main(argv=None) -> int` (L174) — CLI. `uv run python tools/stop_excess.py <round dirs...> [--json]`.

Produces the decomposition table for the programme note §5 in `docs/reference/collision-validation-evidence.md`. Pure, offline, stdlib-only; tested in `tests/unit/test_stop_excess.py`.

### `tools/adr0101_recovery.py`

- `PAYLOAD_PREFIX: Final[str] = "attached:"` (L50) — `party_a` prefix identifying an attached-payload stop.
- `CELL_PREFIX: Final[str] = "voxel_"` (L51) — `party_b` prefix identifying a world-voxel-cell stop.
- `CLEAR_THRESHOLD_M: float` (L48) — The clearance/contact boundary, `0.0`; zero belongs to contact, not clearance, so a real contact is never counted as a recovery.
- `class Stop(NamedTuple)` (L54) — one payload-vs-cell stop with the certified truth behind it (`round_id`, `scene`, `payload`, `cell`, `reported_depth_m`, `certified_gap_m`, `nearest_body`).
  - `Stop.recovered` (L66) — `certified_gap_m > CLEAR_THRESHOLD_M`.
- `class Excluded(NamedTuple)` (L71) — A payload-vs-cell stop that could not be adjudicated, and why; reported, never silently dropped, since shrinking the denominator would inflate the recovery rate.
- `UNATTRIBUTED: str` (L84) — `"<unattributed>"`, the by-fixture label when the body cannot be established.
- `fixture_at_stop(scene_dir: Path, certified_gap_m: float) -> str` (L87) — The world body the payload was nearest, read from the raw ground-truth snapshot and cross-checked against the adjudicated distance; the recovery count never depends on this, only the breakdown does.
- `collect(round_dirs: list[Path]) -> tuple[list[Stop], list[Excluded]]` (L138) — read each round's `verdicts.json` into adjudicable stops plus exclusions. Selects only world stops whose `party_a` is `attached:<id>` and `party_b` is `voxel_<n>`, and only when the probe certified its distances.
- `summarise(stops, excluded) -> dict[str, Any]` (L194) — ADR-0101's Consequences table as data: counts, recovery rate, clearance stats, and the by-fixture breakdown. Returns `recovery_rate=None` over an empty set rather than a number.
- `render(summary) -> str` (L216) — the human-readable report; refuses to print a percentage over an empty denominator.
- `main(argv=None) -> int` (L254) — CLI. `uv run python tools/adr0101_recovery.py <round dirs...> [--json]`.

Produces the recovery-rate figure ADR-0101 rests on. Pure, offline, stdlib-only; reads recorded artifacts, needs no GPU or simulator. Tested in `tests/unit/test_adr0101_recovery.py`.

### `tools/round_power.py`
_Answers how many validation-matrix runs a comparison needs, before a battery is run. Exact (every outcome enumerated, not sampled) and stdlib-only, validated against `scipy.stats.fisher_exact`. Feeds [`docs/reference/collision-validation-evidence.md`](../reference/collision-validation-evidence.md)._

- `RUNS_PER_ROUND = 4` — Scene runs in one matrix round, so `required()` can report rounds as well as runs. (L38)
- `fisher_exact_two_sided(a, n1, b, n2) -> float` — Two-sided Fisher p-value for `a`/`n1` against `b`/`n2`; the "no more probable" tail convention `scipy.stats.fisher_exact` uses. (L45)
- `power(baseline, alternative, n, *, alpha=0.05) -> float` — Exact power at `n` per arm: enumerates every `(a, b)` under the two binomials and sums the probability of the pairs Fisher rejects. `O(n²)` Fisher evaluations, so sub-second to n=100 and a few seconds by n=300. (L83)
- `required(baseline, alternative, *, alpha=0.05, target=0.80) -> tuple[int | None, float]` — Smallest ladder `n` per arm reaching `target` power, or `(None, best)` when the ladder tops out at 800 — a comparison needing more than that is not one this programme can afford, and saying so is the useful answer. (L118)
- `_LADDER: tuple[int, ...]` (L42) — The candidate per-arm sample sizes `required` searches, `20` to `800`.
- `main(argv=None) -> int` (L145) — CLI entry; `--baseline`/`--alternative` rates, `--alpha`, `--target`; prints the required `n` (and rounds) or that the ladder tops out.

### `tools/rskill_scaffolder.py`
_Standalone argparse wrapper around `openral_cli._rskill_scaffolder.scaffold_rskill`._
Mirrors `openral rskill new`; exists so power users can scaffold without installing the CLI distribution.

- `_REPO_ROOT: Path` (L26) — Repo root, derived from `__file__`; used to add `python/<pkg>/src` onto `sys.path` before importing the CLI package.
- `_parse_args(argv) -> argparse.Namespace` — argparse setup. (L35)
- `main(argv=None) -> int` — Entry point; returns a process exit code. (L78)

### `tools/generate_rskill_skillmd.py`
_Generate the standard agent-skill `SKILL.md` discovery view for every in-tree rSkill from its `rskill.yaml`._
The single canonical producer of the `SKILL.md` mirror: `rskill.yaml` is authoritative and the generated file is discovery-only, never hand-edited. `--check` fails on any stale or missing `SKILL.md`, across every skill kind including `playbook`.

- `render_skill_md(manifest_path: Path) -> str` (L162) — Render the `SKILL.md` text (YAML frontmatter + capability/verb summary + license/provenance) from one manifest; `_KIND_NOUN` maps each `kind` to its discovery noun.
- `main(argv=None) -> int` (L261) — Entry point. No args = regenerate every `rskills/<id>/SKILL.md`; positional ids regenerate a subset; `--check` reports stale/missing without writing (exit 1 on drift).

### `tools/rldx_sidecar.py`
_Boot helper for the RLDX-1 inference sidecar, companion to `openral_sim.policies.rldx`._
Materialises a Python 3.10 venv, clones upstream RLDX-1, installs its dependencies (an override path on aarch64), patches in NF4/int8 quantization for the Qwen3-VL backbone, and execs into the inference server. Needed because upstream pins Python ~3.10 and ships a custom model class Transformers doesn't recognise.

- `_LABEL: str` (L34) — `"rldx-sidecar"`, the sentinel/log label for this sidecar's provisioning.
- `_REPO_URL: str` (L35) — Upstream `RLWRLD/RLDX-1` git URL cloned into the venv's source tree.
- `_DEFAULT_HOME: Path` (L36) — `~/.cache/openral/rldx-sidecar`, overridable via `--home`.
- `_AARCH64_OVERRIDE: Path` (L46) — `sidecar_requirements/rldx-aarch64-override.txt`, the torch/torchvision pin override `_install_deps_aarch64` installs under.
- `_NVRTC_OVERRIDE: Path` (L54) — Shared `aarch64-nvrtc-override.txt`, raising nvrtc past the sm_121 cliff.

- `_install_deps(*, source, uv, quantization) -> Path` — `uv sync` in the cloned source tree, then installs `bitsandbytes` when quantizing to NF4/int8. Delegates to `_install_deps_aarch64` on aarch64, where the plain sync can't succeed. (L121)
- `_install_deps_aarch64(*, source, quantization) -> Path` — aarch64 replacement for `uv sync`: provisions the venv with pinned torch/cuda overrides plus the quantization extras, keyed so a corrected pin repairs an existing venv. (L59)
- `_aarch64_extras(quantization) -> list[str]` — the packages installed on top of upstream's own dependency set on aarch64; shared between the install pass and the sentinel spec so they cannot drift. (L105)
- `_make_wrapper(*, work, source, args) -> Path` — Generates the server-launching wrapper script: monkey-patches model loading for NF4/int8/no-op quantization, falls back to `sdpa` attention when flash-attn isn't importable, and hands off to the upstream server entry point. (L157)
- `main() -> int` — argparse entry point (`--model`, `--port`, `--quantization`, `--home`); stamps the sidecar identity record and execs into the sidecar venv so SIGINT reaches the server. (L314)

### `tools/xr1_sidecar.py` + `tools/_xr1_server.py`
_Boot helper + inference server for Xiaomi Robotics XR-1. The launcher provisions its pinned dependency stack and execs the server, which loads the pinned MiBoT checkpoint (NF4 or BF16), recreates the benchmark-specific chat template, and serves `ping/reset/get_action/close` over the shared ZMQ/msgpack wire._
- `main() -> int` (xr1_sidecar.py L94) — parse model/profile/quantization/host/port/home; `--export-dir` persists an NF4 checkpoint and exits, otherwise stamp sidecar identity and `os.execvpe` into `_xr1_server.py`.
- `_NDARRAY_SENTINEL = "__ndarray__"` (_xr1_server.py L17) — Marker key for the msgpack ndarray codec (mirrors `openral_sim.sidecar.encode_ndarray`'s `_NDARRAY_SENTINEL`, re-implemented here because the sidecar venv cannot import `openral_sim`).
- `_STATE_DIM = 60` (_xr1_server.py L18) — XR-1's internal padded state width; `_pad_state` pads any frame/history up to this.
- `_ACTION_DIMS = {"robocasa_mg": 7, "robocasa365": 12, "vlabench_choice": 7}` (_xr1_server.py L19) — Per-benchmark-profile action width the checkpoint processor decodes.
- `main() -> int` (_xr1_server.py L341) — ZMQ REP server loop entry point; loads the pinned checkpoint then answers `ping/reset/get_action/close`.
- `_messages(profile, images, instruction) -> list[dict[str, Any]]` — exact RoboCasa, RoboCasa365-video, or VLABench message layout from upstream revision `7c20088`.
- `_pad_state(state) -> NDArray[np.float32]` — pad one frame or a four-frame history to XR-1's 60-D internal state.
- `class _XR1Policy` — Pinned `AutoModel`/`AutoProcessor` custom-code loader; NF4 uses a BF16 compute dtype, and `prequantized_nf4` reloads a saved packed checkpoint without re-quantizing. `export_pretrained` writes sharded packed safetensors plus quantization metadata.

### `tools/qwen_vlm_sidecar.py` + `tools/_qwen_vlm_server.py`
_Boot helper + server for the Qwen3.5-4B scene-VLM sidecar, companion to `QwenSceneVlm`. Provisions an isolated venv and execs into a ZMQ REQ/REP + msgpack server answering scene questions, run out-of-process for dependency/VRAM isolation. Apache-2.0 model._

- `_DEFAULT_HOME: Path` (L34) — `~/.cache/openral/qwen-vlm-sidecar`, overridable via `--home` / `$OPENRAL_QWEN_VLM_SIDECAR_HOME`.
- `_VENV_ENV: str` (L35) — Env var name for the `--venv` override.
- `_HOME_ENV: str` (L36) — Env var name for the `--home` override.
- `_LOCK: Path` (L46) — Hash-locked pinned deps file the venv is provisioned from.
- `_NVRTC_OVERRIDE: Path` (L51) — aarch64 nvrtc override passed at install time alongside `_LOCK`.
- `ensure_venv(home, *, override=None) -> Path` (L56) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded); honours `$OPENRAL_QWEN_VLM_SIDECAR_VENV`.
- `main() -> int` (L97) — argparse (`--model`, `--host`, `--port`, `--max-side`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME` and `os.execvpe`s into `_qwen_vlm_server.py`.
- `_load(model_id) -> (processor, model)` (_qwen_vlm_server.py L59) — dual-path NF4 load: auto-detects a pre-quantized checkpoint via its embedded config and loads 4-bit directly, else quantizes at load with serial materialization for 8 GB.
- `_query(processor, model, *, image, question, max_side, max_new_tokens) -> str` (_qwen_vlm_server.py L101) — one scene-question→answer generate via the canonical Qwen-VL recipe (strips the `<think>` trace).
- `main() -> int` (_qwen_vlm_server.py L170) — ZMQ REP loop (`ping`/`query`/`shutdown`) that replies with an error object rather than dying on exception. Validated live against real hardware.

### `tools/locateanything_sidecar.py` + `tools/_locateanything_server.py`
_Boot helper + server for the `nvidia/LocateAnything-3B` open-vocabulary detector sidecar, companion to `LocateAnythingDetector`. Runs out-of-process over the same ZMQ pattern as `rldx_sidecar` because its pinned transformers version conflicts with the workspace's. No upstream repo to clone — the model is custom-code on the Hub._

- `_DEFAULT_HOME: Path` (L33) — `~/.cache/openral/locateanything-sidecar`.
- `_VENV_ENV: str` (L34) — Env var name for the `--venv` override.
- `_HOME_ENV: str` (L35) — Env var name for the `--home` override.
- `_LOCK: Path` (L44) — Hash-locked pinned deps (`transformers==4.57.1` load-bearing); regenerated via `uv pip compile tools/sidecar_requirements/locateanything.in`.
- `_NVRTC_OVERRIDE: Path` (L49) — aarch64 nvrtc override passed at install time alongside `_LOCK`.
- `ensure_venv(home, *, override=None) -> Path` (L54) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded, keyed on `_LOCK` + `_NVRTC_OVERRIDE`); honours `$OPENRAL_LOCATEANYTHING_SIDECAR_VENV`.
- `main() -> int` (L96) — argparse (`--model` required, `--host`, `--port`, `--max-side`, `--home`, `--venv`); sets `OPENRAL_ALLOW_REMOTE_CODE=1` (the operator opted in by launching this sidecar) and execs into `_locateanything_server.py`.
- `_split_model_ref(model_ref) -> tuple[str, str]` (_locateanything_server.py L44) — Split a `repo@revision` reference into `(repo, revision)`.
- `_load(model_ref) -> (processor, model)` (_locateanything_server.py L55) — NF4 bitsandbytes load of the pinned custom-code checkpoint.
- `_detect(processor, model, image, query, *, max_side, mode, max_new_tokens) -> str` (_locateanything_server.py L99) — One detection request → the model's raw generated text; `<ref>`/`<box>` parsing stays in the main-env backend.
- `main() -> int` (_locateanything_server.py L146) — ZMQ REP loop (`ping`/`detect`/`shutdown`); `{"ok": False, "error": ...}` on exception rather than dying.

### `tools/da3_depth_sidecar.py` + `tools/_da3_depth_server.py`
_Boot helper + server for the Depth Anything 3 monocular metric-depth sidecar, companion to `openral_perception_ros`'s depth-provider node. Runs in its own Python 3.12 venv behind a ZMQ REQ/REP + msgpack wire, since the model ships outside transformers; the provider republishes the reply as a depth image for nvblox. Loaded by file path so the deploy launch can autostart it under a plain interpreter._

- `_DEFAULT_HOME: Path` (L51) — default cache dir.
- `_VENV_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_VENV"` (L52) — `--venv` override env var name.
- `_HOME_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_HOME"` (L53) — `--home` override env var name.
- `_REQUIREMENTS = ("depth-anything-3", "pyzmq", "msgpack")` (L58) — x86_64 install set.
- `_NVRTC_OVERRIDE: Path` (L72) — aarch64 nvrtc override file passed at install time.
- `_AARCH64_REQUIREMENTS: tuple[str, ...]` (L75) — upstream's dependency list minus `open3d` / `pycolmap` / `xformers` / `pre-commit`, plus the undeclared `addict` and the repo's `torch==2.9.1` aarch64 pins.
- `_AARCH64_DA3 = "depth-anything-3==0.1.1"` (L102) — installed `--no-deps` after `_AARCH64_REQUIREMENTS` on aarch64.
- `_PYCOLMAP_EAGER_IMPORT = "\nimport pycolmap\n"` (L108) — the anchor statement `_defer_pycolmap_import` rewrites.
- `_PYCOLMAP_LAZY_IMPORT: str` (L109) — the lazy `__getattr__` proxy it rewrites that statement into.
- `ensure_venv(home, *, override=None) -> Path` (L163) — Returns the sidecar venv python, provisioning it if absent. x86_64 installs the plain requirement set; aarch64 substitutes packages with no aarch64 wheel and applies an nvrtc override, load-bearing because the model's JIT-fused op otherwise fails on later requests.
- `_site_packages(py) -> Path` (L128) — the `site-packages` of the venv owning `py`; `SystemExit` if absent.
- `_defer_pycolmap_import(py) -> None` (L136) — aarch64 only: rewrites the installed package's eager `import pycolmap` into a lazy proxy, since that dependency has no aarch64 wheel but is only needed by a code path this sidecar never calls. A deferred import, not a stub — the real error still raises if that path is used.
- `main() -> int` (L200) — argparse (`--model` default `depth-anything/DA3-SMALL`, `--host`, `--port` default 5771, `--process-res` default 504, `--home`, `--venv`); execs into `_da3_depth_server.py`. No JIT/fuser env knobs needed, since the nvrtc override fixes the issue at install time.
- `_load(model_id) -> (torch, model)` / `_infer(torch, model, image, process_res) -> (depth, K, latency_ms)` — load DA3 to CUDA in eval mode; one image → metric depth `(H, W)` float32 + 3×3 intrinsics.
- `main() -> int` (_da3_depth_server.py L60) — ZMQ REP loop (`ping`/`depth`/`shutdown`) that replies with an error object rather than dying on a bad frame. Validated live on both an 8 GB Ada GPU and a GB10 aarch64 host.

### `tools/internvla_n1_sidecar.py` + `tools/_internvla_n1_server.py`
_Boot helper + server for InternVLA-N1/DualVLN, companion to `openral_sim.policies.internvla_n1`. Runs out-of-process over ZMQ because upstream's pinned transformers version conflicts with the workspace's; provisions a py3.11 venv from an inference-only pin set and serves the Qwen2.5-VL-7B S2 planner + NavDP DiT S1 policy, mapping discrete VLN-CE actions to a `BODY_TWIST`._

- `_LABEL = "internvla-n1-sidecar"` (L39) — sidecar identity.
- `_REPO_URL = "https://github.com/InternRobotics/InternNav.git"` (L40) — upstream clone URL.
- `_DEFAULT_HOME: Path` (L41) — cache dir.
- `_PYTHON = "3.11"` (L42) — venv Python version.
- `_PINNED_DEPS: list[str]` (L52) — Inference-only pin set mirroring upstream, minus flash-attn (sdpa instead); torch 2.9.1 on cu128 for aarch64 support, and `diffusers==0.32.2` specifically since a later release changed the architecture the checkpoint was trained against.
- `_DIFFUSION_POLICY_PIN: str` (L78) — Pins `diffusion_policy` to one commit installed with `--no-deps`, since System-1's NavDP imports only `SinusoidalPosEmb` from it.
- `_NVRTC_OVERRIDE: Path` (L87) — aarch64 nvrtc override; without it transformers 4.51's Qwen2.5-VL vision stack's `.prod()` dies past the sm_121 ceiling on every System-2 replan.
- `main() -> int` (L165) — argparse via the shared `build_parser`; calls `run_sidecar` with `family="internvla_n1"`, `install_deps=_install_deps`, `make_wrapper=_make_wrapper`.
- `_NDARRAY_SENTINEL = "__ndarray__"` (_internvla_n1_server.py L55) — Marker key for the msgpack ndarray codec (mirrors `openral_sim.sidecar.encode_ndarray`, re-implemented here because the sidecar venv cannot import `openral_sim`).
- `_ACTION_STOP = 0` (_internvla_n1_server.py L60) — Discrete VLN-CE action ids emitted by System-2 (upstream `InternVLAN1AsyncAgent.actions2idx`): STOP.
- `_ACTION_FORWARD = 1` (_internvla_n1_server.py L61) — forward 0.25 m.
- `_ACTION_LEFT = 2` (_internvla_n1_server.py L62) — turn left 15°.
- `_ACTION_RIGHT = 3` (_internvla_n1_server.py L63) — turn right 15°.
- `_ACTION_LOOK_DOWN = 5` (_internvla_n1_server.py L64) — look down (no base motion).
- `_DEFAULT_FORWARD_MPS = 0.25` (_internvla_n1_server.py L69) — forward speed magnitude for `_ACTION_FORWARD`, held for ~1 s of wall time between S2 replans.
- `_DEFAULT_TURN_RADPS = float(np.deg2rad(15.0))` (_internvla_n1_server.py L70) — turn speed magnitude for `_ACTION_LEFT`/`_ACTION_RIGHT`.
- `discrete_action_to_twist(actions, *, forward_mps, turn_radps) -> tuple[float, float, bool]` (_internvla_n1_server.py L73) — Maps a VLN-CE discrete action sequence to `(v_forward, w_yaw, stop)`; only the first action drives motion this step, STOP anywhere latches `stop=True`. Pure/side-effect-free (`tests/unit/test_internvla_n1_action_mapping.py`).
- `main() -> int` (_internvla_n1_server.py L321) — argparse (`--model`, `--host`, `--port`, `--device`, `--quantization`, `--resize`, `--num-history`, `--plan-step-gap`, `--forward-mps`, `--turn-radps`, `--work-dir`); boots the ZMQ server.

### `tools/lingbot_vla2_sidecar.py` + `tools/_lingbot_vla2_server.py`
_Boot helper + server for LingBot-VLA (v1 and v2), companion to `openral_sim.policies.lingbot_vla2`. v2 (6B Qwen3-VL MoE) is the default; `--variant v1` selects the 4B Qwen2.5-VL dense RoboTwin post-train, which is x86_64-only. Runs out-of-process because upstream pins a torch/kernel stack incompatible with the workspace._

- `_LABEL = "lingbot-sidecar"` (L48) — sidecar identity.
- `_REPO_URL: str` (L49) — upstream `robbyant/lingbot-vla-v2` clone URL.
- `_PINNED_SHA: str` (L52) — pinned v2 commit.
- `_DEFAULT_HOME: Path` (L53) — v2 cache dir.
- `_HOME_ENV: str` (L54) — `--home` override env var name.
- `_VENV_ENV: str` (L55) — `--venv` override env var name (v2).
- `_REPO_ENV: str` (L56) — repo-dir override env var name (v2).
- `_REPO_URL_V1: str` (L62) — the v1 (`robbyant/lingbot-vla`) clone URL, driven by `--variant v1`.
- `_PINNED_SHA_V1: str` (L63) — pinned v1 commit.
- `_DEFAULT_HOME_V1: Path` (L64) — v1 cache dir.
- `_VENV_ENV_V1: str` (L65) — `--venv` override env var name (v1).
- `_REPO_ENV_V1: str` (L66) — repo-dir override env var name (v1).
- `_SERVER: Path` (L68) — path to `_lingbot_vla2_server.py`, the `os.execvpe` target for both variants.
- `_TORCH_PIN` (L83) — v2's torch 2.9.1 pin, overridden from upstream's 2.8.0 so the sidecar provisions on aarch64.
- `_TORCHVISION_PIN` (L84) — matching v2 torch-stack pin.
- `_TORCHAUDIO_PIN` (L85) — matching v2 torch-stack pin.
- `_TORCHCODEC_PIN` (L90) — matching v2 torch-stack pin (excluded on aarch64).
- `_TRITON_PIN` (L91) — matching v2 torch-stack pin.
- `_V2_OVERRIDES: tuple[str, ...]` (L92) — the override requirement files applied on top of upstream's v2 lock.
- `main() -> int` (L307) — argparse (`--model`, `--host`, `--port`, `--home`, `--variant {v1,v2}`, quantization); provisions the matching variant's venv and `os.execvpe`s into the server.
- `_NDARRAY_SENTINEL = "__ndarray__"` (_lingbot_vla2_server.py L47) — Marker key for the msgpack ndarray codec.
- `_BNB_META_SUFFIXES` (_lingbot_vla2_server.py L319) — bitsandbytes `Params4bit` packed-stat suffixes written alongside each `.weight`; mirrors `openral_sim._quantization._BNB_META_SUFFIXES` so the on-disk pack format matches every other OpenRAL nf4 rSkill.
- `main(argv: list[str]) -> int` (_lingbot_vla2_server.py L944) — Resolves the torch-version-dependent CUDA allocator env var from installed-package metadata before CUDA initializes, then serves either the v2 or v1 (`--variant v1`) policy.

### `tools/cosmos3_reasoner_sidecar.py`
_Boot helper for the curated NVIDIA Cosmos 3 Edge reasoner model (`OPENRAL_REASONER_MODEL=cosmos3-edge`), companion to `Cosmos3ToolUseClient`. Provisions an isolated venv plus a SHA-pinned transformers overlay, builds a flattened reasoner view for vLLM's loader, and execs into `vllm serve` — its OpenAI-compatible HTTP API is the wire contract the reasoner already speaks. Cosmos 3 weights are commercial-OK, so there is no license guard; forward-pass inference is currently blocked by an upstream bug (see [`docs/reference/cosmos3-edge-reasoner.md`](../reference/cosmos3-edge-reasoner.md))._

- `ensure_venv(home, *, override=None) -> Path` (L94) — Returns the sidecar venv python, provisioning the pinned lock and the SHA-pinned transformers overlay if absent.
- `is_diffusers_reasoner_layout(model_dir) -> bool` (L149) — True when the top-level `model.safetensors.index.json` maps tensors into subfolders (the Edge layout needing a view); Nano/Super (bare top-level shards) return False.
- `materialize_reasoner_view(model_dir, dest) -> Path` (L167) — build a vLLM-loadable flat view: bare-named shard symlinks + a rewritten weight index + tokenizer/config symlinks. Idempotent.
- `resolve_served_model(model, home, *, native_edge=False) -> tuple[str, str | None]` (L250) — Resolves a repo id or local dir to a `vllm serve` target. For the Edge diffusers layout the target depends on the serving vLLM: a native-Edge-aware vLLM serves the snapshot as-is, otherwise the flattened view is required.
- `vllm_has_native_edge_model(py) -> bool` (L215) — Asks the serving venv's own model registry whether it knows the native Edge model class, since the answer differs by platform. Fails closed to the flattened-view path on any error.
- `build_serve_argv(*, vllm_bin, model, host, port, tool_call_parser, max_model_len, gpu_memory_utilization, enforce_eager, served_model_name=None, kv_cache_dtype="auto") -> list[str]` (L285) — Builds the `vllm serve` argv; `--max-model-len`, `--enforce-eager`, `--gpu-memory-utilization` and fp8 KV cache are the knobs that make an 8 GB card fit.
- `main() -> int` (L362) — argparse entry point; sets the CUDA allocator config, resolves the served view, and execs into `vllm serve`.

### `tools/behavior_groot_sidecar.py`
_Python 3.10 sidecar for the official 2026 BEHAVIOR-1K GR00T N1.7 checkpoint. Imports the pinned behavior runtime, wraps `Gr00tPolicy` for the R1Pro modality slices, and serves ZMQ `ping/reset/get_action/close`. Supports whole-model NF4 quantization (default) plus dropping the unused Qwen3-VL `lm_head` to fit an 8 GB host._

- `main(argv) -> int` (L366) — Parses checkpoint/task/instruction/control-mode/device/endpoint plus a quantization mode, loads the real checkpoint (CPU-first under NF4/int8), and serves the 23-D action API.
- `_drop_backbone_lm_head(model) -> None` — Replace the Qwen3-VL `lm_head` with `Identity`; the B1K wrapper consumes only `hidden_states[-1]`, so the full-vocab logits projection is dead weight and was the largest single inference allocation.
- `_quantize_nf4(model, *, device, min_params) -> None` — Whole-model bitsandbytes NF4 rewrite of large linear layers, keeping small layers floating; behaviourally lossless for this checkpoint against a bf16 reference.
- `_quantize_int8(model, *, device, min_params) -> None` — Same rewrite with 8-bit linear layers; higher fidelity but too large to co-reside with OmniGibson on 8 GB, so used for offline comparison only.

### `tools/behavior_scene_sidecar.py`
_Official OmniGibson evaluator environment sidecar for `scene.id=behavior`. Resolves the configured task instance, exposes the challenge wrapper observation, applies external 23-D actions, and returns success/metrics/sim time over ZMQ._

- `main(argv) -> int` (L222) — Parse task/instance/mode/wrapper/max-steps/endpoint, load the real BEHAVIOR environment, and serve `ping/reset/step/close`.

### `tools/_robometer_quant.py`
_Shared NF4 quantize + pre-quantized meta-load helpers for the Robometer reward scorer, re-implementing the repo's NF4 rule rather than importing `openral_sim._quantization`. Determinism is forced so a meta pre-quantized load matches a bf16-then-quantize reference bit-for-bit._

- `MIN_PARAMS = 4_000_000` (L25) — Mirrors `openral_sim._quantization.DEFAULT_MIN_PARAMS_TO_QUANTIZE`.
- `BNB_META_SUFFIXES: tuple[str, ...]` (L28) — bitsandbytes `Params4bit` packed-stat suffixes written alongside each `.weight`.
- `set_cublas_workspace_env() -> None` (L38) — Pins the cuBLAS workspace BEFORE CUDA initialises (required for determinism).
- `apply_determinism() -> None` (L43) — Forces byte-stable numerics: disables cudnn TF32, forces math SDP (flash/mem-efficient SDP are process-state dependent, math SDP is not).
- `quantize_nf4_in_place(root, compute_dtype) -> int` (L55) — Rewrites large `nn.Linear` → `bnb.nn.Linear4bit` in place (the repo's NF4 rule); returns count replaced. Quantization happens lazily on first `.to("cuda")`.
- `install_linear4bit_shells(root, compute_dtype) -> int` (L93) — Replaces large linear layers with empty `Linear4bit` shells on a meta-device skeleton, so a prequantized load can drop in packed weights without the RSS spike of an off-meta load.
- `install_prequantized(policy, state, device) -> set` (L133) — Drops saved packed NF4 weights into `Linear4bit` shells via `from_prequantized`; returns consumed checkpoint keys. Bit-identical to the in-place quantization that produced the checkpoint.
- `assign_meta_buffers(model, state, device) -> int` (L167) — Assigns any still-meta buffer by hand, since `load_state_dict` skips non-persistent buffers; raises if a meta buffer is missing from the checkpoint.
- `is_prequantized_checkpoint(path) -> bool` (L192) — True if `path` holds a `model.safetensors` with bnb NF4 packed keys.

### `tools/_robometer_scorer.py`
_In-process stateless scorer for the Robometer-4B reward monitor, companion to `RobometerInProcessReward`. Loaded directly by `reward_monitor_node` — no separate process or dedicated venv. Meta-builds lerobot's native reward-model skeleton and drops in OpenRAL's NF4 pre-quantized weights directly, with no bf16 spike and no extra download._

- `_NATIVE_CONFIG_REPO: str` (L51) — `"lerobot/Robometer-4B"`, the HF repo `_native_config` downloads `config.json` from to meta-build the native module skeleton.
- `class Scorer` (L104) — Meta-builds the native `RobometerRewardModel` and remaps + loads the NF4 prequant pack.
  - `__init__(weights, device="cuda", *, meta_buffers=True)` (L107) — `meta_buffers=False` builds buffers for real from the modules' own `__init__`: the reference `tests/sim/test_reward_nf4_buffer_equivalence.py` compares the meta load against (issue #304). Seeding `original_inv_freq` from `inv_freq` raises unless the rotary module's `rope_type` is `"default"`.
  - `score(frames_rgb, task, num_bins) -> tuple[list[float], list[float]]` (L211) — Computes per-frame progress/success via the module's logit-decoding path (not the scalar-only `compute_reward`). `num_bins` is accepted for interface parity but unused.

### `tools/build_qwen_vlm_nf4_checkpoint.py`
_Reproducible recipe for the published `OpenRAL/rskill-qwen35_4b-any-general-nf4` pre-quantized NF4 checkpoint. Runs in the sidecar venv. Distinct from `quantize_rskill.py`, which writes an `install_prequantized_linears`-loaded pack for the in-process lerobot runtime; this writes a transformers-native `save_pretrained` checkpoint for the isolated VLM sidecar._

- `main() -> int` (L49) — argparse (`--source`, `--out`); loads the upstream model once, quantizes to NF4, saves the weights + processor, then verifies the checkpoint reloads directly as 4-bit and answers a smoke query.

### `tools/fix_libero_config.py`
_Auto-fix for the stale `~/.libero/config.yaml` pitfall._

Detects and repairs `$LIBERO_CONFIG_PATH/config.yaml` when its absolute paths no longer match the active `libero` package, since the file is written once at first import and never refreshed. Wired into the Justfile ahead of the LIBERO sim recipes; idempotent.

- `_expected_config(libero_pkg_dir) -> dict[str, str]` — Compute the canonical `assets/bddl_files/benchmark_root/datasets/init_states` payload that LIBERO writes on first import. (L31)
- `_parse_yaml_map(text) -> dict[str, str]` — Parse the flat `key: value` map LIBERO writes — no PyYAML dependency. (L47)
- `_render_yaml(payload) -> str` — Render the same flat layout. (L61)
- `_locate_active_libero() -> Path` — `import libero` and return its package directory; raises `RuntimeError` with a clear message when LIBERO is absent (caller treats as no-op). (L66)
- `main() -> int` — argparse entry point; flags `--dry-run`, `--verbose`. Returns 0 when the config matches or after rewriting. (L91)

### `tools/refresh_methods_linenos.py`
_Refreshes the `(LNN)` line citations in the `docs/methods/` inventory; `--check` reports drift or unresolved entries and exits 1; `--coverage` lists undocumented public symbols (wired into `just lint`)._

- `REPO_ROOT` (L40)
- `METHODS_DIR: Path` (L41) — `REPO_ROOT / "docs" / "methods"`.
- `refresh_file(md_path: Path, *, check: bool) -> tuple[int, list[str], dict[Path, set[str]]]` — Rewrite one inventory file's markers; returns the changed-marker count, the unresolved-entry descriptions, and the symbols each source file has an entry for. (L194)
- `coverage_report(documented: dict[Path, set[str]]) -> list[str]` — Public symbols under `python/`, `packages/`, `tools/` (tests, `setup.py`, `conftest.py` excluded) with no inventory entry, one line per file. (L365)
- `main(argv: list[str] | None = None) -> int` (L378) — CLI entry; `--check` / `--coverage` flags.

### `tools/build_robometer_nf4_checkpoint.py`
_Builds the publishable pre-quantized Robometer-4B NF4 checkpoint: loads the upstream bf16 model, quantizes in place, and saves a self-contained directory the scorer can meta-load directly as 4-bit. Uploads to `OpenRAL/rskill-robometer_4b-any-general-nf4`._

- `main() -> int` — CLI entry; `--out <dir>` writes the quantized checkpoint directory. (L45)

### `tools/export_act_onnx.py`
_Exports a LeRobot ACT policy to a single whole-model ONNX graph (`ACTPolicy.model`, called as `predict_action_chunk` does). Normalization stays external (checkpoint's `policy_preprocessor.json`/`policy_postprocessor.json` MEAN_STD sidecars); inputs ordered by `config.image_features` plus the checkpoint's declared state dim. Parity checked in `tests/integration/test_act_onnx.py`._

- `export(out_path, repo_id, *, device="cpu", preprocess="host") -> str` — CLI entry (`--repo-id`, `--out`). (L147)

### `tools/export_rtdetr_onnx.py`
_Exports `PekingU/rtdetr_r18vd_coco_o365` to ONNX matching `ObjectsDetector`'s contract: single image input, /255 preprocessing, two 3-D outputs (pre-sigmoid logits + cxcywh-normalised boxes). Run in an isolated `uv run --isolated --no-project` overlay — the project venv's torchvision would shadow it and break the `RTDetrForObjectDetection` import; do not `uv sync` the onnx-export group._

- `export(out_path, model_id="PekingU/rtdetr_r18vd_coco_o365") -> None` — CLI entry (`--out`). (L30)

### `tools/gen_nav2_visual.py`
_Generates `packages/openral_nav2_bringup/config/nav2_visual.yaml` (the Nav2 costmap profile for the visual-SLAM backend — cuVSLAM + nvblox, `static_layer` off the backend-agnostic `/map` OccupancyGrid instead of ray-casting `/scan`) from the base lidar profile `nav2_panda_mobile.yaml`, so the two stay in sync. One-shot; re-run after editing the base config._

- `render(base_text: str) -> str` — The derived visual-SLAM profile text for a base-profile text; pure, so the sync test can diff it against the checked-in file. (L51)
- `main(argv=None) -> int` — Writes the derived profile; `--check` exits 1 when the checked-in copy is stale (run by `just lint`). (L80)

### `tools/gen_ros_topic_graph.py`
_Generates `docs/topics/README.md`, the ROS 2 topic / service / action graph, by statically joining every publisher, subscriber, server and client in `python/`, `packages/`, `tools/` and `cpp/` (tests excluded) on its resolved name, plus launch-file remappings. Pre-commit rewrites the page; `just lint` and the quality workflow run `--check`._

- `prop REPO_ROOT, OUT_PATH, SCAN_ROOTS` (L48–50) — Repo root, the generated page (`docs/topics/README.md`), and the scanned top-level trees (`python`, `packages`, `cpp`, `tools`).
- `class Endpoint` — One side of a connection: `kind` (topic/service/action), `role`, resolved `name` (or the source expression), `resolved`, canonical `pkg/Type`, and `where` (`path (Class.method)` plus a `[param `x`]` note). (L90)
- `extract_python(files) -> list[Endpoint]` — Endpoints from Python sources; names resolve through literals, f-strings, enclosing-scope and module constants (across `from x import`), class and `self._x` attributes, parameter defaults (a parameter without one stays unresolved), `declare_parameter` defaults, argparse `add_argument(default=...)` and `openral_core.camera_topic(name, kind)` (a non-literal name becomes `{sensor}`); name and type are read positionally or by rclpy's keywords (`topic`/`msg_type`, `srv_name`/`srv_type`, `action_name`/`action_type`). (L428)
- `extract_cpp(files) -> list[Endpoint]` — Endpoints from C++ sources; arguments are split with balanced brackets, both `rclcpp_action::create_server`/`create_client` overloads (node, or the four node interfaces) are handled, and a name resolves from a string literal or a `declare_parameter<T>("x", "default")` variable. (L471)
- `extract_remappings(files) -> list[tuple[str, str, str]]` — `(from, to, launch file)` for every `remappings=` pair. (L537)
- `render(endpoints, remaps) -> str` — The Markdown page: concrete and per-instance (`{placeholder}`) topics, services and actions, unresolved endpoints, remappings. (L569)
- `build() -> str` — Scans the tree and renders the page. (L639)
- `main(argv=None) -> int` — Writes the page; `--check` exits 1 when the checked-in copy is stale. (L647)

### `tools/quantize_lingbot_vla2.py`
_Pre-quantizes LingBot-VLA 2.0's Qwen3-VL backbone to an NF4 pack ahead of time (the sidecar normally does this at load) so deploys download ~7 GB instead of 25.5 GB and skip the per-boot conversion. Runs in the sidecar venv (torch 2.9 / transformers 4.57.3 / bitsandbytes) importing `tools/_lingbot_vla2_server.py`'s own helpers so the pack matches the runtime shells byte-for-byte. Frugal streaming keeps GPU peak at a few hundred MB and host peak at ~30 GB._

- `main(argv) -> int` — CLI entry (`--ckpt`, `--out`, `--qwen`). (L313)

### `tools/quantize_rskill.py`
_Quantize a lerobot policy and upload the result to the HuggingFace Hub. Policy-agnostic counterpart to the in-process fast-path in `openral_sim._quantization`. Loads any lerobot policy via `from_pretrained`, quantizes in place, saves `model.safetensors` + `quantization_metadata.json`, and uploads to a target rSkill repo — the adapter's `load_prequantized_state_for_rskill` detects the metadata sentinel and skips the on-line nf4 conversion on every subsequent launch. Not part of CI — a one-shot Hub-mutating tool gated on `HF_TOKEN`._

- `main() -> int` (L537) — CLI entry; `--repo-id`/`--out-repo-id`/`--quantization`/`--trust-remote-code`; loads, quantizes in place, uploads.

### `tools/verify_test_envs.py`
_Verifies selected optional test environments locally: syncs the matching dependency group declared in `tools/test_selection.toml`, runs its selected targets, and fails if any test skips. Hardware and externally-provisioned sidecars are opt-in since this script cannot create physical devices or proprietary simulator installs._

- `REPO_ROOT: Path` (L22) — Repo root, derived from this file's location.
- `CONFIG: Path` (L23) — `tools/test_selection.toml`, the lane config this script drives.
- `PROVISIONED_GROUPS: dict[str, str]` (L25) — Lane name → the sidecar/fork env var (or manual-install note) that provisions it.
- `LANE_EXTRA_GROUPS: dict[str, list[str]]` (L35) — Extra `uv sync --group` names to pull in alongside a lane's own group (`sim` also needs `dataset`).
- `LANE_DEPENDENCY_GROUP: dict[str, str]` (L39) — Lane name → the actual `uv` dependency group when it differs from the lane name (`robocasa-gr1` syncs `robocasa`).
- `MANUAL_INSTALL_GROUPS: frozenset[str] = {"robocasa-gr1", "simpler-env"}` (L43) — Lanes this script cannot provision itself; skipped with a note instead of failing.
- `ONNX_EXPORT: Path` (L44) — Expected `rtdetr-coco-r18/model.onnx` export artifact path, checked before the export-dependent lane runs.
- `SIMPLER_ENV_SPEC: str` (L45) — pip spec for the `simpler-env` git package (maniskill3 branch).
- `ROBOCASA_GR1_REPO: str` (L46) — Upstream RoboCasa-GR1 tabletop-tasks fork clone URL.
- `ROBOCASA_GR1_DIR: Path = Path("/tmp/rg1")` (L47) — Fixed checkout directory for the RoboCasa-GR1 clone.
- `LOCATEANYTHING_VENV: Path` (L48) — Expected LocateAnything sidecar venv under `~/.cache/openral`.
- `QWEN_VLM_VENV: Path` (L49) — Expected Qwen-VLM sidecar venv under `~/.cache/openral`.
- `RLBENCH_HOME: Path` (L50) — RLBench sidecar cache root.
- `RLBENCH_VENV: Path` (L51) — RLBench sidecar venv, under `RLBENCH_HOME`.
- `RLBENCH_PYTHON: Path` (L52) — That venv's interpreter.
- `COPPELIASIM_ROOT: Path` (L53) — Expected CoppeliaSim 4.1.0 EDU install directory.
- `COPPELIASIM_ARCHIVE: Path` (L54) — Cached CoppeliaSim download archive.
- `COPPELIASIM_URL: str` (L57) — Upstream CoppeliaSim download URL.
- `POLICY_SRC: Path` (L60) — Shared clone root for policy-adjacent source checkouts (PyRep/RLBench/3D-Diffuser-Actor).
- `PYREP_DIR: Path` (L61) — PyRep checkout dir under `POLICY_SRC`.
- `RLBENCH_DIR: Path` (L62) — RLBench checkout dir under `POLICY_SRC` (distinct from `RLBENCH_HOME`, the sidecar's own cache).
- `THREEDDA_DIR: Path` (L63) — 3D-Diffuser-Actor checkout dir under `POLICY_SRC`.
- `PYREP_REPO: str` (L64) — PyRep upstream repo URL.
- `RLBENCH_REPO: str` (L65) — RLBench upstream repo URL.
- `THREEDDA_REPO: str` (L66) — 3D-Diffuser-Actor upstream repo URL.
- `THREEDDA_INSTRUCTIONS: Path` (L67) — Expected path to the PerAct instructions pickle 3D-Diffuser-Actor needs.
- `main(argv=None) -> int` — CLI entry; drives group sync, provisioning, and `_strict_pytest` per lane. (L468)

### `tools/wait_for_action_and_signal_palette.py`
_Polls for a ROS 2 action server to appear, then publishes on `/openral/skill_registry_changed` so the reasoner re-seeds its rSkill palette. Needed because a wrapped-ROS rSkill can drop from the palette if its action isn't advertised yet at launch (e.g. Nav2 takes 15-30 s). Exits 0 once published, 1 on timeout._

- `main() -> int` — CLI entry (`--action`, `--timeout-s`). (L101)

### `tools/galaxea_a1_ros1_sidecar.py`
_Galaxea A1 ROS 1 sidecar for `openral_hal.galaxea_a1.GalaxeaA1HAL`; owns roscore, the official serial driver, and the official joint tracker, stopping all three on every exit path. No vendor code is bundled in OpenRAL._

- `JOINT_NAMES = tuple(f"arm_joint{i}" for i in range(1, 7))` (L28) — The six A1 arm joint names, in driver order.
- `JOINT_LIMITS` (L29) — Per-joint `(lower, upper)` radian limits, same order as `JOINT_NAMES`.
- `PROTOCOL_VERSION = 1` (L37) — Wire protocol version for the staged/host command relay.
- `GRIPPER_STATUS_INDEX = len(JOINT_NAMES)` (L38) — Index of the gripper's status code within a 7-element status tuple.
- `MAX_STATUS_MASK = 0xFFFFFFFF` (L39) — Valid uint32 range for a motor status/error code.
- `TRACKER_NODE = "openral_a1_joint_tracker"` (L40) — Name of the official joint tracker process this sidecar starts.
- `STAGED_COMMAND_TOPIC = "/openral/arm_joint_command_staged"` (L41) — Topic the HAL publishes staged joint targets on.
- `HOST_COMMAND_TOPIC = "/arm_joint_command_host"` (L42) — Topic the sidecar forwards accepted commands to on the host driver side.
- `TRACKER_ALLOWED_MODES = (0,)` (L43) — Tracker control-mode values the relay accepts as safe to arm on.
- `class Stack` (L92) — Owns and tears down the ROS 1 process group in reverse start order.
  - `Stack.start(argv)` (L100) — Launch a child in its own session, tracked for teardown.
  - `Stack.stop()` (L105) — SIGINT every child (reverse order), wait, escalate to SIGKILL on timeout.
- `class Bridge` (L125) — ROS 1 topic relay: buffers joint/gripper/status feedback, gates a staged command through the tracker's activation handshake, and republishes to the host driver.
  - `Bridge.configure(config)` (L170) — Install the validated client contract while keeping motion locked.
  - `Bridge.ready() -> bool` (L373) — Whether joint, gripper and status feedback have all arrived at least once.
  - `Bridge.command_paths_ready() -> bool` (L381) — Whether staged and host command topic consumers/publishers are all connected.
  - `Bridge.relay_state() -> str` (L390) — Current relay state (`LOCKED`/`ARMING`/…).
  - `Bridge.state(config)` (L394) — Assemble the current feedback+relay JSON state, raising `RuntimeError` on stale feedback, a latched fault, or an out-of-limits reading.
  - `Bridge.apply(packet, config, first_joint_command)` (L457) — Validate and forward one command packet (`joint_targets` / gripper) to the host driver after re-checking state and motor status.
- `main() -> int` (L709) — argparse (`--bind`, `--port`, `--serial`, `--startup-timeout-s`); starts roscore + the vendor serial driver + tracker via `Stack`, wires `Bridge`, serves the sidecar's TCP/JSON protocol.

### `tools/joint_state_staleness_probe.py`

- `pct(xs, p) -> float` (L68) — nearest-rank percentile; NaN when empty.
- `build_probe_hal(robot_yaml) -> RosControlHAL` (L76) — builds the robot's real HAL from its manifest via `build_hal(mode="real")` (so it reads at the manifest's `safety.joint_state_staleness_limit_s` on the manifest's joint-state topic); refuses a non-`RosControlHAL` robot (SO-100, ALOHA, Galaxea) with `SystemExit`.
- `run(robot_yaml, duration_s, rate_hz, read_hz, load) -> int` (L103) — publishes synthetic `JointState` at `rate_hz` over real DDS into `RosControlTransport` (production QoS + callback) on one executor, and prints callback inter-arrival gaps, stamp→callback latency, the read-side age `now - last_arrival()` at `read_hz` (default: the manifest's control rate), and how often `hal.read_state()` raised `ROSPerceptionStale` at the manifest's limit. `load=True` adds a GIL-contending thread in the same process.
- `main(argv=None) -> int` (L257) — CLI: `uv run python tools/joint_state_staleness_probe.py --robot robots/<id>/robot.yaml --rate <joint_states Hz> [--duration 60] [--read-hz N] [--load]`. Needs a sourced ROS 2 overlay; uses `ROS_DOMAIN_ID` 77 unless set, so it never touches a live graph.

Backs `safety.joint_state_staleness_limit_s` in a real manifest — the age past which `read_state` (and the runner's waits) refuse to answer. The number must come from a measurement on the deploy host, not a schema default. Thor 2026-09-23 (Fast-DDS): idle gap p99.9 2.6 ms / max 4.6 ms, read-side age max 32 ms; under GIL starvation latency max 43 ms, no gap above 33 ms → `robots/openarm/robot.yaml` declares 0.1 s (three control periods). The other real manifests are unmeasured and marked provisional (Franka / Sawyer 0.2 s, the real HALs' former constructor default).
