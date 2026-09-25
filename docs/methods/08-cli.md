# CLI

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/cli/src/openral_cli/main.py`
_openral CLI entry point — `openral` command (typer-based)._

- `_root(ctx)` — Top-level callback: configures observability with the mode-derived trace sample ratio and opens a `cli.command` root span that closes when the subcommand returns. (L390)
- `_RUN_MODE_BY_SUBCOMMAND: dict[str, str]` — Subcommand → `openral.run.mode` mapping consumed by `_root`. (L371)
- `_SAMPLE_RATIO_BY_MODE: dict[str, float]` — `openral.run.mode` → trace sample ratio; hardware samples at 0.1. (L384)
- `class CheckResult(NamedTuple)` — One row in `openral doctor` output. (L421)
  fields: `check, status, details`
- `_check_python() -> CheckResult` (L438)
- `_check_platform() -> CheckResult` (L443)
- `_check_openral_core() -> CheckResult` (L447)
- `_check_ros2() -> list[CheckResult]` — ROS 2 binary, distro, RMW. A missing `ros2` binary is `absent` (non-fatal), not `missing`: Tier-0 ships no ROS 2 by design, so `openral doctor` must still exit 0 there. (L455)
- `_check_colcon() -> CheckResult` — `absent` (non-fatal) when unavailable, for the same Tier-0 reason as `_check_ros2`. (L503)
- `_check_gpu(result, warnings) -> list[CheckResult]` — One row per GPU. (L512)
- `_check_compute_spec(result) -> list[CheckResult]` — Builds `ComputeSpec` rows from `_check_gpu`'s `GpuProbeResult` (no second probe), mirroring exactly what `openral detect` would write into the manifest so doctor and detect stay in sync. (L549)
- `_check_usb() -> list[CheckResult]` — Candidate robot USB serial devices. (L620)
- `_check_just() -> CheckResult` (L638)
- `_check_reasoner_llm() -> list[CheckResult]` — Model-first doctor dispatcher: reads `OPENRAL_REASONER_MODEL`, resolves it against `REASONER_MODELS`, and reports dialect/hosting/endpoint/key status. An uncurated model fails until `OPENRAL_REASONER_ENDPOINT` names an endpoint or dialect; never prints the API key value. (L851)
- `_check_reasoner_model(model_key) -> list[CheckResult]` — Resolve one curated/uncurated model into doctor rows; delegates loopback checks to `_reasoner_endpoint_probe_row`. Resolves a named `OPENRAL_REASONER_ENDPOINT` through the same `openral_core.REASONER_ENDPOINT_PRESETS` table the factory uses, without importing the optional reasoner package. (L734)
- `_reasoner_endpoint_probe_row(label, base_url, *, managed, autostart) -> CheckResult` — Generic loopback probe. Down managed-local + autostart is `info`; managed with autostart disabled and every BYO-local endpoint are `warn`. (L697)
- `_cosmos_autostart_enabled() -> bool` — mirrors the reasoner client's `OPENRAL_COSMOS3_AUTOSTART` parsing (falsy spellings `0/false/no/off`); kept local so doctor never imports the optionally-installed reasoner package. (L653)
- `_is_local_base_url(url) -> bool` — True when host resolves to a loopback name. (L664)
- `_probe_tcp(host, port, *, timeout_s=0.2) -> bool` — Fast non-blocking TCP probe used to diagnose the Ollama daemon. (L670)
- `_gather_checks() -> list[CheckResult]` (L873)
- `_YELLOW_STATUSES: frozenset[str]` — `{"absent", "info", "warn"}`; `CheckResult.status` values rendered yellow (advisory, non-fatal) rather than red/green in `doctor`'s table. (L418)
- `doctor(--json)` — Diagnose host: Python, OS, ROS 2, GPU, USB. Delegates GPU enumeration to `probe_gpus`. (L906)
- `detect(--output, --robot/--as, --report, --dds-timeout, --include, --no-write, --deployment, --yes)` (L941) — Always-interactive `robot.yaml` builder; `--no-write` short-circuits to non-interactive probe-only inspection (CI-safe), `--report <path>` dumps the raw `DetectionReport` JSON alongside either mode. `--robot/--as` forces a canonical base manifest as the template over USB/DDS inference. Prompts a rig name, walks detected cameras (V4L2/RealSense/Orbbec) into robot or workcell sensors, optionally overrides joint/safety limits, and relocates `file:` asset refs; `--deployment <path>` additionally scaffolds a self-contained `DeployScene`.
  - `_prompt_robot_name(default) -> str` — `typer.prompt` for the custom rig name; blank input keeps `default`.
  - `_run_camera_binding_wizard(canonical, detection) -> tuple[list[SensorSpec], list[SensorSpec]]` — `(robot_sensors, workcell_sensors)`; the routing prompt loop above over `_iter_wizard_cameras` (V4L2 + RealSense + Orbbec). Enter skips a device, dropping any canonical sensor never bound.
  - `_iter_wizard_cameras(detection) -> list[tuple[str, str, str]]` — `(device_path, label, serial)` per bindable camera; V4L2 keys on `device_path`, RealSense/Orbbec key on `serial` (empty `device_path`).
  - `_grab_camera_thumbnail(device_path, out_dir) -> Path | None` — Best-effort one-frame JPEG grab (opencv) so the operator can see which physical camera a `/dev/video*` node is.
- `_CV2_WARNED: set[str]` (L1359) — Print-once guard for the "opencv missing" hint; a mutated set avoids a `global` statement while staying a single-process latch. (Used by `_grab_camera_thumbnail`.)
  - `_maybe_customize_limits(description) -> RobotDescription` — Opt-in gate `"Customize joint limits & safety envelope?"` (default No); declining inherits the canonical `JointSpec`/`SafetySpec` fields verbatim, accepting walks each present (non-`None`) joint field and the four safety scalars with Enter-to-keep-default prompts.
  - `_relocate_file_assets(description, *, canonical_dir, output_path) -> RobotDescription` — Rewrites `assets.urdf.ref`/`mjcf`/`srdf` `file:<rel>` refs to repo-root-relative when `output_path` is outside `canonical_dir`, so the custom manifest's URDF/SRDF still resolve after relocation.
  - `_write_deploy_scene_scaffold(path, description, sensor_specs, *, detection, assume_yes) -> None` — Validates through `DeployScene` then writes the scaffold + review banner. `sensor_specs` here are always workcell (non-robot) cameras from the wizard. Overrides the seeded HAL `port` with the one probed in `detection.usb.matches` (when present) instead of the manifest's stale default.
- `_ROBOT_NAME_RE: re.Pattern[str]` (L1142) — `[a-z0-9_]+`; validates a custom rig name typed at the `_prompt_robot_name` prompt.
- `_render_detection_summary(detection)` — Print per-probe Rich table for `openral detect`.
- `connect(--robot, --port)` (L1553) — Open HAL, read state, disconnect. `--robot` accepts `so100` / `so101` (both drive the shared `SO100FollowerHAL`).
- `_connect_so_follower(label: str, port: str) -> None` — Connect an SO-100/SO-101 follower arm via `SO100FollowerHAL`, read one state, disconnect.
- `calibrate_camera(--sensor, --topic, --chessboard-size, --square-size, --dry-run)` (L1619) — Run ROS 2 `camera_calibration`.
- `prompt_command(TEXT, --topic, --wait-s, --discovery-wait-s, --new-goal)` (`python/cli/src/openral_cli/prompt.py` L31) — `openral prompt`: publishes a one-shot `PromptStamped` onto `/openral/prompt_in/cli`, which `prompt_router_node` fans out to the reasoner; spins up to `--discovery-wait-s` for a subscriber before publishing anyway. `--new-goal` tells the reasoner to rebuild its mission queue instead of treating the prompt as conversational context.
- `robot_vendor_urdf(robot_id, --upstream, --out, --rename, --raw-text)` — `openral robot vendor-urdf` typer command; defers the `robot_descriptions`/`xacrodoc`/`yourdfpy` import and delegates to `robot.py`'s `vendor_urdf`. (L3694)
- `rskill_install(HUB_ID, --revision, --force, --non-commercial, --yes)` (L1737) — `openral rskill install`. Download rSkill, validate, register. An org-less `HUB_ID` (no `/`) fails fast with an `OpenRAL/<name>` suggestion + `rskill search` hint instead of a raw Hub 404; a 404 on a qualified id appends the same search hint.
- `_DEFAULT_RSKILL_ORG: Final[str]` (L1733) — `"OpenRAL"`; suggested org prefix for an org-less `HUB_ID` passed to `rskill_install`.
- `rskill_search(QUERY?..., --kind, --role, --embodiment, --license, --family, --limit, --json)` (L1901) — Thin renderer over `openral_rskill.hub_search.search_hub_rskills` (see `docs/methods/04-rskill.md`). `QUERY` is variadic and matched locally (case-insensitive substring match against id/name/description/tags), not sent to the Hub as `search=`. Table gains a `local` column (`in-tree`/`installed`/`—`).
- `_render_rskill_search_results(result: HubRSkillSearchResult, query) -> None` — Print the `rskill search` table or the no-results notice, including the `local` column and a skipped-count notice that names every skipped id (`N OpenRAL repo(s) skipped — no valid rskill.yaml: <id>, <id>`).
- `_RSKILL_SEARCH_DESC_MAX: int` (L1853) — `60`; character cap `rskill search`'s table truncates a manifest `description` to.
- `rskill_list(--json)` (L1989) — `openral rskill list`. List installed rSkills.
- `rskill_check(rskill_id?, --robot, --rskills-dir, --json)` (L2173) — `openral rskill check`. Two modes. With a positional id, resolves it via `load_rskill_manifest` and renders a per-section breakdown via `check_single_rskill`. Without an id, falls back to the legacy walk-all path (`check_installed_rskills`). `--rskills-dir` defaults to `rskills/` and is silently skipped when the directory does not exist. Exits 1 on any blocking failure.
- `_SECTION_DISPLAY: dict[str, str]` (L2094) — Manifest section key → display label for `rskill_check`'s per-section breakdown table.
- `rskill_new(ID, --out-dir, --owner, --license, --embodiment-tag, --family, --from-hf, --yes, --overwrite)` — Scaffolds a new local rSkill from `rskills/template/`. Three modes: `--from-hf <repo>` introspects the Hub config to auto-fill manifest fields; `--family <...>` overlays family-aware defaults; otherwise interactive prompts fill any missing flag (skipped under `--yes`). `--embodiment-tag` is checked against `openral_rskill.loader.intree_embodiment_tags()` (robot-declared tags + any/custom/multi), not a closed schema Literal. (L2265)
- `_resolve_or_prompt(value, *, prompt, default, skip_prompt) -> str` — Drives the owner / license / embodiment prompts only when the flag was not provided and `--yes` is off.
- `_DEFAULT_OWNER: str` (L2259) — `"your-org"`; default for `rskill new --owner` when unset.
- `_DEFAULT_LICENSE: str` (L2260) — `"apache-2.0"`; default for `rskill new --license` when unset.
- `_DEFAULT_EMBODIMENT: str` (L2261) — `"franka_panda"`; default for `rskill new --embodiment-tag` when unset.
- `_resolve_family_and_patch(*, family, from_hf, yes) -> tuple[RSkillFamily | None, RSkillPatch | None]` — Resolves `--family` / `--from-hf` into a family + manifest patch for `scaffold_rskill`. Prompts for family in interactive mode; bails non-zero with a clear message when `--from-hf` introspection fails or `--family` is unrecognized.
- `_display_license_banner(name, license_value, version, con) -> None`
- `sensor_list(--vendor, --modality, --kind, --json)` — List entries in sensor catalog. (L2560)
- `sensor_show(SENSOR_ID, --name, --parent-frame, --json)` — Resolve catalog entry to a `SensorSpec`/`SensorBundle`. (L2651)
- `benchmark_list(--benchmarks-dir)` — List every benchmark suite id under `benchmarks/*.yaml`, one paste-able `--suite` value per line. (L2720)
- `benchmark_report(--rskills-dir, --json)` — Aggregate `rskills/*/eval/*.json` benchmark blocks into a rich-table or JSON dump. Validates every JSON against `RSkillEvalResult`. `--rskills-dir` defaults to `rskills/`. (L3460)
- `benchmark_run(--suite, --rskill, --out, --device, --save-dir, --benchmarks-dir, --task, --n-episodes, --dry-run, --update-manifest/--no-update-manifest, --video/--no-video, --video-dir, --dashboard, --dashboard-port)` — Resolves `--suite` to a list of `BenchmarkScene`s, parses the rSkill reference, dispatches to `openral_sim.run_benchmark`, and writes a validated `RSkillEvalResult` JSON. `--video` (default on) records one world MP4 per episode; `--task` runs a single suite task, otherwise the run auto-filters to the rSkill's `evaluated_tasks`. `--update-manifest` (default on) writes `avg_success_rate` back into the manifest. (L2744)
- `benchmark_scene(--config, --rskill, --out, --device, --save-dir, --save-video, --video-size, --n-episodes, --view/--no-view, --dry-run, --update-manifest/--no-update-manifest, --dashboard, --dashboard-port)` — Single-scene sibling of `benchmark_run`. `--view` opens a live `mujoco.viewer` per episode; `--save-video DIR` writes a clean single-view world MP4 per episode plus a `videos.json` manifest, for website hero clips. Strictly accepts a `BenchmarkScene` YAML (rejects DeployScene/SimScene) and writes results to `rskills/<dir>/eval/scene_<scene_id>.json`. (L3126)
- `_default_benchmark_scene_out_path(vla_spec, scene) -> Path` — Mirrors `_default_benchmark_out_path` but for single-scene JSONs; the `scene_` prefix distinguishes per-scene outputs from multi-task suite outputs under the same rSkill directory. (L3443)
- `deploy sim(--config, --robot, --dashboard-port, --foxglove/--no-foxglove, --foxglove-port, --reset-to-pose-service, --hal, --memory-dir, --initial-task, --dry-run)` — Boots the full ROS graph (dashboard, safety kernel, reasoner, prompt router, runtime, HAL) against a digital-twin HAL via one generic `deploy_e2e.launch.py` launch. No envelope YAML is written: the launch computes the safety envelope intersection and forwards it as kernel ROS parameters. No `--rskill` flag — the reasoner picks the active rSkill dynamically from the in-tree palette. `--memory-dir` wires in the deploy's self-maintained memory, 3D scene graph, and occupancy grid when present. Defined in `openral_cli.deploy_sim`.
- `deploy_list()` — `openral deploy list`; list every deploy scene under `scenes/deploy/*.yaml`, one paste-able `--config` path per line. (L3844)
- `dashboard(--host, --port, --log-level, --inprocess)` (L3774) — `openral dashboard`. Boots a live debug pane that doubles as an OTLP/HTTP receiver on the same port (default `127.0.0.1:4318`), so it works without Jaeger/Tempo. `--inprocess` spawns a shell-quoted child workload with the OTLP env vars pre-set, so a demo needs no second shell. Inverse path: `openral sim run --dashboard`.
- `replay(BAG, --trace, --frame, --dataset-root, --dashboard, --out)` (L4245) — Reads a `.mcap` file or rosbag2 directory, joins with OTel spans from `--dashboard`, and emits a chronological JSON timeline keyed by `trace_id`. `--frame <repo_id>/<episode>/<frame>` pivots from a written LeRobotDataset frame instead of an explicit `--trace`.
- `_resolve_frame_trace_id(frame_spec, dataset_root) -> str` — Parse a `<repo_id>/<episode>/<frame>` spec (`rsplit('/', 2)`; repo_id keeps its own slash) and return that frame's stored `trace_id` via `read_frame_trace`. `typer.Exit(2)` on a malformed spec, missing frame, or a frame with no trace.
- `record(--out, --profile, --storage, --extra-topic, --extra-regex, --dry-run)` (L4321) — Spawn `ros2 bag record` with `slim` (default) or `full` profile presets; `--dry-run` prints the composed argv without forking.
- `profile_session(ACTION, --output, --name)` (L4398) — `openral profile session (start | stop | view)`. Drive an LTTng session for the realtime hot path; surfaces `LttngSessionError` cleanly when `lttng` is missing on PATH. Set `OPENRAL_ROS2_TRACING=1` on the agent process to emit tracepoints; without the gate every tracepoint is a no-op.
- `_print_benchmark_run_plan(scenes, *, suite_id, vla_spec) -> None` — `benchmark run --dry-run`'s plan printer. Applies the same `filter_scenes_for_skill` / `_manifest_for_filter` pass `run_benchmark` applies, so the printed plan is the plan that would execute: exits 1 when the rSkill's `evaluated_tasks` cover no suite task (previously dry-ran clean, then raised `ROSCapabilityMismatch` on the real run), and prints a skip note for partial coverage.
- `_resolve_benchmark_suite(suite: str, benchmarks_dir: Path) -> tuple[list[BenchmarkScene], str]` — Accept either a built-in id (looked up at `benchmarks/<id>.yaml`) or a direct YAML path; raise `typer.BadParameter` listing catalogue entries on a typo.
- `_parse_rskill_cli_arg(raw)` — Parse `--rskill <ref>` into a `VLASpec`. Accepts bare names (`smolvla-libero`), paths (`rskills/smolvla-libero`), or HF repo ids; validates via `openral_rskill.validate_skill_ref` so `VLASpec.weights_uri` rejects explicit URI schemes. The adapter id is read from the manifest's `model_family`. Raises `typer.BadParameter` on an invalid scheme or empty input.
- `behavior_serve(--rskill, --task, --instruction, --host, --port, --device, --state-dim, --action-dim)` (L3597) — Serve one rSkill through the official BEHAVIOR Challenge WebSocket policy protocol. Defaults to the R1Pro observation/action contract; OmniGibson remains out-of-process and owns task loading, metrics, and videos.
- `_summarize_results(results: dict[str, object]) -> str` — Headline-line picker for free-form `results` blocks (`*_avg` → numeric → status fallback). (L3550)
- `_path_completer(text: str, state: int) -> str | None` — Stdlib `readline`-shaped Tab completer for `_run_repl`: globs `text*` (with `~` expansion), adds trailing `/` to directories, and rewrites `$HOME` back to `~`. Lets the REPL complete filesystem paths after flags like `--config`/`--rskill`.
- `render_banner(version_str: str, *, width: int | None = None) -> RenderableType` (L178) — Builds the interactive-REPL welcome box as a content-sized rich `Panel`, choosing a two-column or stacked layout to fit `width`. Returns a renderable (not a print), so it exports to plain text in tests independent of TTY/colour state.
- `_logo_wordmark(*, stacked: bool) -> RenderableType` — Logo mark + OPENRAL wordmark side by side (vertical-middle grid) or stacked.
- `_identity(*, stacked: bool) -> RenderableType` — `_logo_wordmark` above the tagline + capability strip.
- `_kv_grid(rows: tuple[tuple[str, str], ...], key_style: str) -> Table` — Borderless two-column `key  value` grid (styled key, dim value) used for the links and commands cells.
- `_LOGO_ART: str` (L99) — White (single-weight, no gradient) OpenRAL logo mark — a 6-row block icon (horns flaring out and down into a rounded head, eyes below) matching the wordmark height.
- `_WORDMARK_ART: str` (L110) — The OPENRAL block-letter wordmark.
- `_TAGLINE_TAIL: Final[str]` (L120) — `" — Open Robot Agentic Layer (harness) for embodied AI"`, appended to the banner identity line.
- `_CAPABILITIES: Final[str]` (L121) — `"fast policies · slow reasoning · rewards · perception · control"` capability strip shown under the tagline.
- `_LINKS` (L123) — Community links (Discord / GitHub / Hugging Face / Website) shown in the right cell.
- `_COMMANDS: tuple[tuple[str, str], ...]` (L130) — Quick-start commands (`doctor`, `rskill search`, `help`, `exit`) shown in the right cell.
- `_WIDE_MIN` (L141) — Minimum terminal columns (127) for the two-column layout, measured from the rendered box so the richest layout that fits is chosen and the box never overflows.
- `_SIDE_BY_SIDE_MIN: int` (L142) — Minimum terminal columns (82) below which the logo stacks above the wordmark instead of sharing a line.
- `_cli_version() -> str` — Best-effort `openral-cli` package version for the banner title; suppresses `PackageNotFoundError` and falls back to `"0.0.0"`.
- `_print_banner() -> None` — Print `render_banner(_cli_version(), width=console.width)` to the REPL `console` at `_run_repl` startup, sizing to the live terminal.

### `python/cli/src/openral_cli/_dds_scope.py`
_Keeps a simulation and a real robot off each other's ROS graph._

Two controls, covering different directions: confinement stops a sim reaching a robot, and the occupancy guard stops a launch in *either* direction joining a graph a robot is already on — something confinement alone cannot do, since a real robot's graph legitimately spans machines and is never confined. The guard's signature is a foreign `/joint_states` publisher, because every robot has exactly one whether real or simulated.

- `confine_sim_scope(env: dict[str, str]) -> None` — Pin a sim to `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` + `ROS_DOMAIN_ID=77`, in place. `LOCALHOST` not `OFF`: measured against the live robot, `LOCALHOST` hid its nodes while same-host discovery was unaffected, whereas `OFF` hides the sim's own nodes from each other. `setdefault`, so an explicitly-exported scope wins and attaching a dashboard from another host stays possible. (L101)
- `assert_graph_unoccupied(env, *, hal_mode: str) -> None` — Refuse to launch onto a graph that already carries a robot. Symmetric: same rule whichever side starts first, `hal_mode` only shapes the wording. Raises `ROSConfigError` naming the occupying node, the scope scanned, and the ros2_control signature when the other side looks like real hardware. **An unreadable graph also raises** — a guard that passes when its instrument is broken is worse than no guard. (L187)
- `ALLOW_UNVERIFIED_GRAPH_ENV: Final[str]` — `OPENRAL_ALLOW_UNVERIFIED_GRAPH=1`. Waives only the unreadable-graph refusal (`rclpy` won't import, the probe crashed) — a broken instrument, not a detected hazard. The occupied-graph refusal is unwaivable: no env var starts a robot onto a graph that already carries one. (L65)
- `SIM_DOMAIN_ID: Final[str]` — `"77"`. Any non-zero value would do; 0 is where an unconfigured robot host lands. (L69)
- `_PROBE_SPIN_S: Final[float]` — Seconds the graph-scope probe spins before deciding the graph is empty (3.0); too short passes the guard on an occupied graph. (L75)
- `_PROBE_NODE_NAME: Final[str]` — `"openral_graph_scope_probe"`, the probe's own node name (excluded from the occupancy count). (L77)
- `_NO_ROS: Final[str]` — Sentinel `_scan_graph` returns when `rclpy` is not importable; distinct from a probe failure — a non-ROS host has no graph to refuse. (L85)
- `_MAX_NODES_LISTED: Final[int]` — `12`; how many occupying node names the refusal message lists before eliding. (L89)
- `_HARDWARE_SIGNATURES: Final[tuple[str, ...]]` — `ros2_control` node-name substrings (`controller_manager`, `joint_state_broadcaster`, `hardware_interface`) that flag an occupying graph as "looks like real hardware" in the refusal wording only. (L94)

> The `ros2` CLI **daemon** is unusable for this check: it answers from the environment *it* was started with, so `ros2 node list` under `LOCALHOST` still reported the remote robot's nodes. The probe runs in a subprocess under the exact launch env and talks to `rclpy` directly.

### `python/cli/src/openral_cli/_hf_publish.py`
_Shared HF Hub publishing helpers, de-duped from `tools/rskill_publisher.py`._

- `resolve_token(token_arg: str | None = None) -> str` — Resolve the HF token from arg → `HF_TOKEN` → `HUGGINGFACE_HUB_TOKEN`. Raises `ROSConfigError` with actionable hint when missing. (L47)
- `ensure_private(api: HfApi, repo_id: str, *, repo_type: str = "model") -> None` — Re-fetch repo metadata and raise `ROSConfigError` if the repo is public. Critical safety gate; `repo_type` supports `"model"` / `"dataset"` / `"space"`. (L80)
- `IGNORE_PATTERNS: Final[list[str]]` — Glob patterns excluded from every upload (`.env`, `__pycache__`, `*.key`, etc.). (L35)

### `python/cli/src/openral_cli/_rskill_doc_validator.py`
_rSkill README + manifest publish-readiness validator (the publish gate)._

Hard gate consumed by `tools/rskill_publisher.py`: refuses publish when the README is missing/too short/missing sections/has template sentinels, or the manifest still has template defaults. Supports single-hop delegation via an HTML marker so sibling stubs can share one canonical README.

- `class DocValidationIssue(BaseModel)` — one problem record: `severity` (`"error"` / `"warning"`), `field` (`readme.section.License` / `manifest.description` / …), `message`. (L147)
- `class DocValidationReport(BaseModel)` — `skill_dir` + `manifest_name` + `issues`; `.is_valid` / `.errors` / `.warnings` derived properties. (L164)
  - `is_valid() -> bool` — True iff no `error`-severity issue was found. (L184)
  - `errors() -> list[DocValidationIssue]` — The `issues` subset with `severity == "error"`. (L189)
  - `warnings() -> list[DocValidationIssue]` — The `issues` subset with `severity == "warning"`. (L194)
- `validate_rskill_docs(skill_dir: Path, manifest: RSkillManifest) -> DocValidationReport` — Public entry point; composes README and manifest checks into one report. (L202)
- `_validate_readme(skill_dir) -> list[DocValidationIssue]` — Presence, min-length, required-sections-via-heading, placeholder-sentinel scan. Honors single-hop delegation.
- `_resolve_delegation(skill_dir, body) -> (list[str] | None, list[DocValidationIssue])` — Resolve a `openral:rskill-readme-delegates-to` marker; rejects double-hop and missing-target.
- `_validate_manifest_content(manifest) -> list[DocValidationIssue]` — Description / provenance (`paper_url` ∨ `source_repo`) / `name` / `weights_uri` / `source_repo` checks beyond what the Pydantic schema catches.
- `format_report(report) -> str` — Human-readable summary used by the publisher dry-run. (L508)
- `README_REQUIRED_SECTIONS: Final[tuple[tuple[str, tuple[str, ...]], ...]]` — Canonical README section coverage (display label → acceptable case-insensitive heading substrings). (L69)
- `PLACEHOLDER_SENTINELS: Final[tuple[str, ...]]` — Strings (plus `RSKILL_TEMPLATE_SENTINELS`) that must not survive into a published README or manifest. (L102)
- `PLACEHOLDER_MANIFEST_DESCRIPTION_MARKERS: Final[tuple[str, ...]]` — Substrings that flag a manifest `description` left at the template default. (L118)
- `DELEGATION_MARKER_NAME: Final[str]` — `"openral:rskill-readme-delegates-to"`, the HTML-comment marker name letting a stub README delegate section checks to a sibling. (L124)
- `_README_FILENAME: Final[str]` — `"README.md"`. (L54)
- `_MIN_README_BODY_LENGTH: Final[int]` — `200`; minimum stripped-body length before the README is flagged too short. (L55)
- `_MIN_DESCRIPTION_LENGTH: Final[int]` — `30`; minimum manifest `description` length. (L56)
- `_MEDIA_PATTERN: Final[re.Pattern[str]]` — Matches a markdown image, `<img>`, or `<video>` tag; missing preview media is a warning, not a publish-blocking error. (L58)
- `_DELEGATION_MARKER_PATTERN: Final[re.Pattern[str]]` — Compiled regex parsing the `DELEGATION_MARKER_NAME` HTML comment into its target path. (L138)

### `python/cli/src/openral_cli/_rskill_readme.py`
_Derive an HF model-card README from an rSkill manifest, the single source of truth._

The manifest-derived front-matter is emitted uniformly so every published repo, private or public, carries a consistent card; `tools/rskill_publisher.py` rebuilds it at publish time while preserving the human-written prose body verbatim. Hand-curated extras already in the in-tree README are unioned in; derived fields win on conflict.

- `build_rskill_frontmatter(manifest: RSkillManifest) -> dict[str, Any]` — Pure, deterministic derivation of the model-card front-matter dict (no network). `base_model` comes from `source_repo` (NF4 repos self-host weights via `weights_uri`); `base_model_relation` is `quantized` when a quantization block is present, else `finetune`. (L113)
- `render_frontmatter(fm: dict[str, Any]) -> str` — Render the dict as a `---`-fenced YAML block, stable field order. (L172)
- `build_rskill_readme(manifest: RSkillManifest, body: str) -> str` — Full README = merged front-matter + prose body; strips any existing front-matter from `body` and merges its curated extras. Idempotent. (L178)
- `_LEROBOT_FAMILIES: frozenset[str]` — `model_family` values that load through lerobot's `PolicyProcessorPipeline` (so `library_name` should point there); gr00t/rldx (out-of-process) and molmoact2 (transformers custom-code) are excluded. (L38)
- `_LICENSE_MAP: dict[RSkillLicensePosture, tuple[str, str | None]]` — License posture → `(HF license id, optional license_name)`; non-standard postures map to `"other"` + a descriptive `license_name`. (L43)
- `_NON_QUANTIZED_DTYPES: frozenset[str]` — dtype strings (`fp32`/`bf16`/`fp16`/`float32`/`float16`) that do NOT imply a quantized `base_model_relation`. (L55)
- `_KIND_PIPELINE_TAG: dict[str, str]` — rSkill `kind` → HF `pipeline_tag` (`detector` → `object-detection`, `segmenter` → `mask-generation`); other kinds fall back to `robotics`. (L61)
- `_KIND_EXTRA_TAGS: dict[str, list[str]]` — rSkill `kind` → extra model-card tags (`vla`, `detector`, `segmenter`). (L65)
- `_FRONTMATTER_RE: re.Pattern[str]` — Matches a leading `---`-fenced YAML front-matter block to strip from an existing README body. (L82)

### `python/cli/src/openral_cli/deploy_sim.py`
_`openral deploy sim` — boot the full ROS graph against a digital-twin HAL via `ros2 launch openral_rskill_ros deploy_e2e.launch.py` (one generic launch, no `--rskill`)._

- `deploy_sim_command(--config, --robot, --dashboard-port, --foxglove/--no-foxglove, --foxglove-port, --reset-to-pose-service, --hal, --memory-dir, --initial-task, --dry-run)` (L2824) — Typer callback for `deploy sim`. Resolves the launch invocation, runs preflight checks (package discoverability, orphan-process reap, palette dependency + VRAM fit), writes the ephemeral envelope + HAL params, and shells `ros2 launch` via `_run_launch`, which tears the launch's process group down in escalating stages on exit (`OPENRAL_SKIP_ORPHAN_REAP=1` opts a parallel harness out of the startup sweep so concurrent workers don't kill each other's graphs).
- `assert_ros2_packages_discoverable(packages, *, prefix_lookup=_ros2_pkg_prefix) -> None` (L2436) — Raises `ROSConfigError` listing every `pkg` `ros2 pkg prefix` cannot resolve — catches an un-sourced workspace overlay or a stale `just ros2-build` before `ros2 launch`'s terse failure. `prefix_lookup` is injectable for a deterministic test fake.
- `_repo_root_from(start) -> Path` (L293) — Resolves the repo root holding `robots/`+`rskills/`: `$OPENRAL_REPO_ROOT` override, then walking up from `start` (editable install), then from the cwd (wheel install run from inside a checkout) — the cwd step is what makes `deploy sim` work off a published wheel at all, since the manifest trees are repo data with no site-packages ancestor.
- `_preflight_scene_assets(config) -> None` (L2469) — Provisions the scene's sim backend before `ros2 launch`, backend-agnostic via each scene's own `SCENES.provision` hook, so a multi-GB asset pull or license prompt happens on a reachable TTY instead of timing out inside the HAL's bounded `on_configure`. Advisory — a failure warns and continues, since the backend retries and raises there. Called after `assert_ros2_packages_discoverable` so a missing overlay fails fast, not after a multi-GB download.
- `_preflight_palette_deps(*, repo_root, robot_yaml, commercial_deployment=False) -> None` — Advisory (not a gate) policy-extras preflight mirroring `ReasonerNode._maybe_seed_palette_from_search_paths`: probes each capability-matched rSkill's importability. When any are blocked it either auto-installs (`OPENRAL_AUTO_INSTALL_DEPS=1`), prompts on a TTY, or drops them from the palette and proceeds — failing hard only if that would leave the palette empty. Also calls `_apply_palette_head_cam` once the palette is known non-empty.
- `_apply_palette_head_cam(matched) -> bool` — Sets `OPENRAL_ROBOCASA_HEAD_CAM=1` when any capability-matched rSkill declares `observation.images.head` in `sensors_required`, so RoboCasa's forward nav camera renders without per-scene config. An operator-set value always wins. Called from `_preflight_palette_deps`; defined in `openral_cli.deploy_sim`.
- `run_launch_invocation(invocation, *, run_preflight=True) -> int` (L1731) — Shared shelling path for `deploy sim` + `deploy run`: under `run_preflight` (the default) runs the same preflight sequence as `deploy sim` (package discoverability, orphan reap, palette deps, reward VRAM fit), then writes the ephemeral HAL params and shells the resolved argv via `_run_launch`, returning its exit code. `run_preflight=False` is a full bypass. Defined in `openral_cli.deploy_sim`.
- `DDS_TRANSPORT_READY_MARKER: Final[str] = "dds_transport_ready:"` (L2153) — Printed once the orphan reap and stale Fast-DDS shm purge (`/dev/shm/fastrtps_*` for Fast-DDS 2.x, `fastdds_*` for 3.x) have run and before `ros2 launch` spawns, as `dds_transport_ready: rmw=<rmw> shm_purged=<N> shm_kept_live=<M>`; waiters gate on this line. The purge only unlinks segment/port groups no live process has open, mapped or `flock`ed (per `/proc/<pid>/{fd,maps}` and `/proc/locks`), so a participant already running keeps its shared memory. Outside the initial PID namespace (inside a container) liveness cannot be proven — a driver in another namespace sharing `/dev/shm` is invisible — so nothing is unlinked. Defined in `openral_cli.deploy_sim`.
- `FASTDDS_SHM_CLEAN_ENV: Final[str] = "OPENRAL_FASTDDS_SHM_CLEAN"` (L2208) — Set to `0` to skip the Fast-DDS shm purge entirely (counts print as `n/a`). Defined in `openral_cli.deploy_sim`.
- `_preflight_depth_extrinsics(description, robot_yaml, unit) -> None` (L2373) — Real-deploy gate called from `resolve_launch_invocation` when `hal_mode == "real"`, octomap is on and the world-voxel kernel check is on: every depth camera, at the pose the selected robot unit publishes (`description` already carries its `SensorOverlay`), must have a report at `calibration/<unit>/<sensor>_extrinsic.json` (unit-less only for a robot without `units/`, which is the only case with no unit selected) that `openral_core.depth_extrinsic.verify_extrinsic_report(..., unit=unit)` clears, else `ROSConfigError` listing each. No override flag; `--no-enable-octomap-kernel-check` is the only other way past. Defined in `openral_cli.deploy_sim`.
- `_detect_gpu_free_vram_gb() -> float` — `openral_core.gpu.detect_gpu_vram_gb("memory.free")`; budgets against *free*, not total, VRAM, since a desktop compositor or sibling worktree can already hold GBs the VLA+reward pair never sees. Returns `0.0` on any failure, which makes the preflight skip and defer to the reasoner's runtime check. Defined in `openral_cli.deploy_sim`.
- `_capability_matched_manifests(repo_root, description, *, commercial_deployment=False) -> list[RSkillManifest]` — Loads every `rskills/*/rskill.yaml` and returns those the reasoner palette would admit, since `deploy sim` doesn't preselect a VLA and reward resolution + the VRAM preflight reason over this set instead. Defined in `openral_cli.deploy_sim`.
- `_resolve_reward_monitor_manifest(*, repo_root, description, explicit_manifest) -> str` — Reward-model resolution: an explicit `--reward-monitor-manifest` wins, else derives it from the capability-matched VLA palette's `reward_rskill_name` consensus, defaulting to `robometer-4b` when none names one or they disagree. Defined in `openral_cli.deploy_sim`.
- `_preflight_reward_vram_fit(*, repo_root, description, reward_manifest_path, gpu_budget_gb, commercial_deployment=False) -> None` — Pre-launch VLA↔reward VRAM gate: checks each capability-matched VLA against the reward model via `assert_vla_reward_fits`, advisory per-VLA but a hard `typer.Exit(1)` only when none can co-reside with it. Defined in `openral_cli.deploy_sim`.
- `_ros2_argv_head() -> list[str]` — Argv prefix for the `ros2 launch` shell-out: `[sys.executable, <abs path to ros2>]` when the `ros2` script's shebang names a Python, else bare `["ros2"]`. Running the launch parser under `sys.executable` keeps the venv's `include-system-site-packages = false` in effect, so apt-installed system packages (e.g. `python3-pandas`) can't shadow the venv's own compiled extensions and crash the launch. Defined in `openral_cli.deploy_sim`.
- Tegra allocator gate (in `_prepare_launch_env`) — `openral_core.is_tegra_host()` **skips** the `expandable_segments:True` allocator default on an L4T host: the iGPU shares system RAM (no discrete-card fragmentation to recover) and torch's expandable path queries NVML GPU-fabric info the iGPU cannot answer — on qorin1 (torch 2.13+cu130, 2026-09-22) the runtime_node's first CUDA allocation raised `Expected NVML_SUCCESS == …nvmlDeviceGetGpuFabricInfoV_…` 363 s into a policy load. DGX Spark (GB10) keeps the default: expandable segments are reported to work and help there ([vllm#55569](https://github.com/vllm-project/vllm/issues/55569); [unsloth-zoo#1235](https://github.com/unslothai/unsloth-zoo/pull/1235) carves GB10 out of the same Tegra exclusion); not verified on our hosts. An operator's explicit setting still passes through.
- `resolve_launch_invocation(*, config=None, robot_override, dashboard_port, reset_to_pose_service, approach_skill_id=None, hal_param_overrides=None, hal_mode="sim", enable_slam=None, enable_nav2=None, enable_octomap=None, enable_dashboard=True, initial_task_prompt=None) -> LaunchInvocation` (L836) — Pure resolver shared by `deploy sim`/`deploy run`: loads the `DeployScene`, merges its committed `runtime:` block under the CLI flags (precedence CLI > scene > auto), and merges any `hal:` binding defaults into `hal_params` below a `--hal` override. Forwards `workcell_json:=...` when the scene declares safety/ACM overrides, so the launch applies tighten-only safety before configuring the kernel. Always forwards the scene YAML as `deploy_config:=` (sim and real) so the launch derives the HAL autostart budget from `backend_options.boot_timeout_s` and merges scene `sensors:`; the real-only camera leg is gated on `hal_mode` inside the launch. `LaunchInvocation` also carries `preload_rskill_id` / `preload_rskill_revision` / `preload_prompt` (scene-only, from `DeployRuntime`), forwarded as `preload_rskill_id:=` / `preload_rskill_revision:=` / `preload_prompt:=` only when the scene sets them.

- `deploy_run(--config, --robot, --hal, --dashboard/--no-dashboard, --dashboard-port, --foxglove/--no-foxglove, --foxglove-port, --initial-task, --enable-reward-monitor/--no-enable-reward-monitor, --reward-monitor-manifest, --dataset-out, --dataset-repo-id, --dataset-license, --dry-run)` (`python/cli/src/openral_cli/main.py` L3869) — Typer callback for `deploy run`: resolves the `DeployScene` and shells the same launch graph as `deploy sim` with `hal_mode:=real`. `--enable-reward-monitor` brings up the reward monitor parallel to the VLA; `--dataset-out`/`--dataset-repo-id`/`--dataset-license` mirror `deploy sim`'s recording args, since a real-hardware session is exactly the kind of run worth recording. Defined in `openral_cli.main`.
- `deploy_validate(--config, --robot, --hal)` (`python/cli/src/openral_cli/main.py` L4042) — Pre-run readiness check for `openral deploy run`, touching no hardware. Loads and resolves the `DeployScene`, then checks real-run-only requirements that otherwise fail late: a declared serial `port`, calibration files present when `calibrate_on_connect=false`, and every deploy sensor's `deploy_binding` (the manifest's robot cameras with the host's `RobotUnit` overlay, plus the scene's workcell cameras, via `merge_deploy_sensors`); a scene naming a robot camera is refused. Missing committed data errors (exit non-zero); a device merely not attached now only warns.
- `_parse_hal_overrides(raw: list[str] | None) -> dict[str, object]` — Parse repeated `--hal key=value` flags. Values are JSON-decoded where possible (so `--hal viewer_enabled=false` parses as bool); fall back to raw string for `--hal port=/dev/ttyUSB0`.
- `class LaunchInvocation` (L126) — Frozen dataclass carrying the resolved launch config for `ros2 launch deploy_e2e.launch.py`: robot/HAL identity and mode, the optional MoveIt `approach_skill_id`, and the object-detector, reward-monitor, scene-VLM, and critic-producer legs. Each optional field forwards as `field:=…` only when set, since `ros2 launch` rejects an empty `name:=`.
- `_omdet_runtime_available() -> bool` — Probe (`importlib.util.find_spec` for `transformers` + `timm`) deciding whether `resolve_launch_invocation`'s default object detector is the open-vocab `omdet-turbo-indoor` continuous backend or the in-tree RT-DETR COCO ONNX fallback. Patched in unit tests to exercise both branches deterministically. Defined in `openral_cli.deploy_sim`.
- `class _HalSpec` (L75) — Frozen dataclass `package` / `executable` / `node_name` / `bare_twin_sim`, **derived** per deploy (no per-robot table). `package` is always `_GENERIC_HAL_PACKAGE` (`"openral_hal_node"`, the one manifest-driven HAL lifecycle node package); `node_name` is the ROS node NAME `openral_hal_<robot_id>` (the launch's `__node:=` remap — not a package name). A robot's vendor real bringup comes from the manifest's `hal.real_bringup`, not from the package.
- `_resolve_clock_origin(*, hal_mode, config, pinned=None, cloud_topic="") -> str` (L453) — The graph's clock authority: a scene's `runtime.clock_origin` wins; else `host_wall` for `deploy run`, and for `deploy sim` `simulation` when the scene's backend exposes a clock (every scene-configured bare twin, and scene-attached backends registered with `sim_clock=True`), `host_wall` otherwise — including a scene-less `--robot`-only twin, and a twin whose octomap `cloud_topic` lies outside `/openral/cameras/` (a real driver stamping on wall-clock). Raises `ROSConfigError` for a `simulation` pin with no clock or with such a real-driver cloud. `resolve_launch_invocation` also refuses (`ROSConfigError`) octomap enabled — explicitly, or auto-enabled by a cloud source (`openral_core.deploy_cloud_topic` non-empty, or on `deploy run` any `SensorSpec.is_cloud_source` in manifest ∪ scene) — with no cloud to map, and forwards `DeployRuntime.voxel_freshness_s` as `world_voxel_deadline_s:=` / `max_octree_age_s:=`.
- `_scene_builds_bare_twin(deploy_scene: DeployScene | None) -> bool` (L97) — The SCENE decides sim HAL shape: bare MuJoCo twin iff there is no scene, the DeployScene declares its own `composition`, or its `scene.id` is not in `openral_sim.SCENES`; otherwise scene-attach (`SimAttachedHAL` via `sim_env_yaml`).
- `_derive_hal_spec(robot_id: str, deploy_scene: DeployScene | None) -> _HalSpec` (L115) — Builds the `_HalSpec` above. Adding a robot to `deploy sim|run` needs only `robots/<id>/robot.yaml` (or `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml`); `resolve_launch_invocation` resolves it via `openral_sim.policies.robots.resolve_robot_manifest`, rejects sim mode early when the robot has no `hal.sim`, no `sim:` block and a bare-twin scene, and forwards the merged DeployScene `hal.defaults` + `--hal` kwargs as the `hal_transport_json` node param.

### `python/cli/src/openral_cli/dataset.py`
_`openral dataset` Typer app + `push` subcommand._

- `dataset_app: typer.Typer` — Public Typer group mounted under `openral` at `name="dataset"`. (L41)
- `push_command(root, *, repo_id, yes, dry_run, token, commit_message) -> None` — `openral dataset push <root>`. Reads `meta/info.json`, resolves the repo_id, runs the PII consent prompt (skippable via `--yes` or `OPENRAL_DATASET_CONSENT=1`), then `HfApi.create_repo(private=True) → ensure_private → upload_folder`. (L294)
- `from_bag_command(bag_path, *, robot, output, repo_id, license, fps) -> None` — `openral dataset from-bag <bag.mcap> --robot robots/<x>/robot.yaml --output <ds-root>`. Calls `Rosbag2ToLeRobotConverter.from_bag`; produces a v3 dataset ready for `openral dataset push`. (L59)
- `_read_info_json(root: Path) -> dict[str, object]` — Parse `meta/info.json`; raises `ROSConfigError` on missing / malformed file. (L185)
- `_camera_keys_from_info(info) -> list[str]` — Extract `observation.images.*` feature keys for the consent prompt's camera disclosure. (L212)
- `_confirm_consent(repo_id, root, info, yes) -> None` — Render the PII / regulatory disclosure Panel, accept `--yes` / env-var overrides, refuse non-TTY without override. (L220)
- `_resolve_repo_id(root, info, cli_repo_id) -> str` — CLI override → info.json → error. Validates `<org>/<name>` format. (L266)
- `_CONSENT_ENV_VAR: Final[str]` — `"OPENRAL_DATASET_CONSENT"`; unattended-consent override for the PII disclosure prompt (mirrors `--yes`). (L38)
- `_CONSENT_PROMPT_BODY: str` — Rendered body text of the PII / regulatory disclosure Panel shown by `_confirm_consent`. (L158)

### `python/cli/src/openral_cli/collision.py`
_`openral collision lower|check` Typer app: offline URDF/SRDF → manifest self-collision model. Defers `openral_safety.urdf_lowering.lower_robot` (yourdfpy/trimesh) so `openral --help` stays fast._

- `collision_app: typer.Typer` — Public Typer group mounted under `openral` at `name="collision"`.
- `lower(robot, *, write, acm_only, geometry_only, emit_cumotion=None, tight_link=[]) -> None` — `openral collision lower --robot <yaml>`. Prints a unified diff of the regenerated collision blocks and mutates the manifest only with `--write`; MJCF-lowered robots fit their geometry from the MJCF like URDF robots do from the URDF. `--tight-link <link>` (repeatable) lowers that link as a box plus its exact hull (`tight_geometry`); links the manifest already refines stay refined. Refuses (exit 3, untouched) when the lowered geometry is looser than what's already committed, with no override flag — deleting the tighter entries is the only way past it. `--emit-cumotion <path>` additionally renders a cuRobo robot-config from the manifest's own collision geometry. (L557)
- `class GeometryLoosening` — Frozen dataclass report record: `link_name`, `shipped`, `lowered`, `volume_ratio`, `circumradius_ratio`. `lowered == ""` with infinite ratios means the lowering emits no primitive for a link the manifest covers. (L118)
- `collision_primitive_envelope(shape) -> tuple[float, float]` — `(volume_m³, circumradius_m)` for one `CollisionShape`, closed-form for all three kinds — the two numbers a "did this get looser?" comparison needs, independent of `origin_xyz_rpy`. Raises `ROSConfigError` on an unknown discriminator rather than silently scoring it zero. (L145)
- `_BLOCK_FLOAT_DP: int` — `4`; decimal places both `render_blocks` and `geometry_loosening`'s quantization round to, so a re-rendered manifest never reads as looser than itself. (L198)
- `_render_tight(tight) -> list[str]` — A box's `tight_geometry` block for `render_blocks`, at full float precision (the refinement is written at 1 nm by the lowering and has no headroom, so it is never rounded to 4 dp). (L407)
- `geometry_loosening(shipped, lowered) -> list[GeometryLoosening]` — Links where re-lowering would claim more space than the committed manifest, worst volume ratio first, reported when volume or circumradius grows (they can fail independently). Both sides are quantized to `render_blocks`'s 4 dp first, so a re-rendered manifest never reads as looser than itself; a dropped link gets infinite ratios, an added link is not reported. (L237)
- `_JOINT_NAME_RE: re.Pattern[str]` — Matches a manifest joint-list `- name: "<joint>"` line, used by `inject_joint_fk` to locate the block to patch. (L344)
- `_FK_ZERO_SNAP_M: float` — `1e-9`; magnitudes below this snap to `0.0` when `inject_joint_fk` renders FK floats, avoiding `-0.0`/near-zero noise in the diff. (L349)
- `_lower(robot_path, *, acm_only, geometry_only) -> tuple[RobotDescription, LoweredCollisionModel]` — Shared loader: parse the manifest and lower via `lower_robot_auto` (the provenance dispatcher). Used by `_lowered_text` and the `--emit-cumotion` path. (L483)
- `_lowered_text(robot_path, *, acm_only, geometry_only) -> tuple[str, str, list[GeometryLoosening]]` — `(current_text, spliced_text, loosening)` — the one place holding the committed manifest and its replacement at once, which is why the loosening comparison lives here. The third element is empty whenever the geometry block isn't rewritten at all, so a caller can't mistake "not compared" for "compared and clean". (L512)
- `check(robot, *, all_robots, acm_only, geometry_only) -> None` — `openral collision check (--robot <yaml> | --all)`. Exits 1 if any manifest drifts from its lowered model (the fleet-wide ACM drift guard). A drifting manifest whose re-lower would *loosen* it also says so, rather than sending the reader into a `lower --write` that will refuse. (L654)
- `splice_collision_blocks(text, *, geometry_block=None, acm_block=None) -> str` — Replace only the two collision blocks in a manifest's text, preserving every other line + comment (absorbs the block's own header comment so repeated lowers stay idempotent). (L99)
- `render_blocks(model) -> tuple[str, str]` — Render a `LoweredCollisionModel` to `(geometry_block, acm_block)` YAML text with a generated-provenance header; floats rounded to 4 dp for a stable diff. (L432)
- `inject_joint_fk(text, joint_fk) -> str` — Inject `origin_xyz`/`origin_rpy`/`axis_xyz` into the named manifest joint blocks (matched by name), dropping any pre-existing FK lines. Used when onboarding a robot onto self-collision (the kernel needs joint FK to place capsules). Idempotent; preserves all other lines/comments. (L362)

### `python/cli/src/openral_cli/behavior.py`
_BEHAVIOR Challenge WebSocket policy server backing `openral behavior serve` (`behavior_serve` in `main.py`)._

- `_R1PRO_CAMERA_SENSORS` — Re-export of `openral_sim._behavior_wire.CAMERA_SENSORS`, the R1Pro observation contract's camera sensor keys. (L25)
- `_R1PRO_STATE_KEY` — Re-export of `openral_sim._behavior_wire.STATE_KEY`. (L26)
- `_R1PRO_STATE_DIM` — Re-export of `openral_sim._behavior_wire.STATE_DIM`. (L27)
- `_R1PRO_ACTION_DIM` — Re-export of `openral_sim._behavior_wire.ACTION_DIM`. (L28)
- `_RGB_RANK: int` — `3`; expected `ndarray.ndim` for an RGB observation frame. (L29)
- `_RGB_CHANNELS: int` — `3`; expected channel count for an RGB observation frame. (L30)

### `python/cli/src/openral_cli/check.py`
_`openral check`: static, host-independent validation of the declarative robot/skill/scene set. Imports only `openral_core` (no hardware probe); complements the host-specific `openral rskill check`. Manifest JSON-Schema emission lives in `tools/schema_export.py` (CI-gated), not here._

- `class CheckFinding(BaseModel)` — One problem: `rule` (`robot_parse`/`rskill_parse`/`scene_parse`/`asset_ref`/`scene_robot_id`/`scene_sensor_geometry`/`robot_unit`/`scene_robot_unit`/`embodiment_reach`/`frames`), `severity` (`error`/`warning`), `target`, `message`. (L84)
- `class GraphCheckReport(BaseModel)` — Typed `openral check --json` payload: `generated_at`, `n_robots`/`n_rskills`/`n_scenes`, `findings`; `.errors` / `.warnings` / `.ok` properties. (L95)
  - `errors() -> list[CheckFinding]` — Findings that fail the check. (L108)
  - `warnings() -> list[CheckFinding]` — Advisory findings (fail only under `--strict`). (L113)
  - `ok() -> bool` — True when there are no error-severity findings. (L118)
- `check_description_graph(repo_root, *, resolve_remote_assets=False) -> GraphCheckReport` — Parses every robot/rskill/scene manifest, resolves `file:`/`ros2://` asset refs, and checks that scene `robot_id`s resolve, no scene sensor reuses a robot-manifest sensor's name (`check_scene_sensor_overrides`), every `robots/<id>/units/*.yaml` and scene `robot_unit` resolves and applies (`load_robot_unit`, `apply_sensor_overlays`), rSkill embodiment tags reach an in-repo robot, and sensor `parent_frame`s are declared — reusing `RobotDescription.from_yaml`/`resolve_asset` rather than parallel validation logic. (L330)
- `check_command(--repo-root, --strict, --resolve-remote-assets, --json)` — The `openral check` leaf command; exit 1 on any error (and on warnings under `--strict`). Registered in `main.py` via `app.command("check")`. (L399)
- `_REMOTE_ASSET_PREFIXES: tuple[str, ...]` — Asset-ref prefixes (`rd:`/`gym_aloha:`/`openarm:`/`menagerie:`) that download a package or need a sim-only dep; skipped unless `resolve_remote_assets`. (L72)
- `_SCENE_TIERS: dict[str, type[DeployScene]]` — Scene subdirectory name (`deploy`/`sim`/`benchmark`) → its schema class. (L74)
- `_LOAD_ERRORS: tuple[type[Exception], ...]` — Exceptions a manifest `from_yaml`/`model_validate` may raise for a bad file, caught to produce a `CheckFinding` instead of crashing the whole check. (L81)
- `_URDF_LINK_RE: re.Pattern[str]` — Matches a URDF `<link name="...">` tag, used to enumerate link names for the `frames` rule. (L152)
- `_SEVERITY_STYLE: dict[str, str]` — `CheckFinding.severity` → Rich console style (`error` → red, `warning` → yellow). (L367)

### `python/cli/src/openral_cli/install.py`
_`openral install <group>` — post-install escape hatch for the Tier-0 curl-bash installer; each command wraps `_install_group` for one `uv sync --group` extras group (sim physics, LIBERO, MetaWorld, ManiSkill3, SimplerEnv, RoboCasa, RLDX sidecar) or the sudo+apt ROS 2 bootstrap._

- `install_sim(--force)` — `openral install sim`; installs the `sim` group (gym-aloha, gym-pusht, MuJoCo, bitsandbytes). (L296)
- `install_libero(--force)` — `openral install libero`; installs LIBERO (mutually exclusive with `robocasa`). (L304)
- `install_metaworld(--force)` — `openral install metaworld`; installs the MetaWorld MT50 task suite (Sawyer scenes). (L312)
- `install_maniskill3(--force)` — `openral install maniskill3`; installs ManiSkill3 (SAPIEN GPU physics). (L320)
- `install_simpler_env(--force)` — `openral install simpler-env`; installs the SimplerEnv real-to-sim correlator backend. (L328)
- `install_robocasa(--force)` — `openral install robocasa`; installs RoboCasa (mutually exclusive with `libero`). (L336)
- `install_rldx(--force)` — `openral install rldx`; installs the RLDX-1 sidecar client (pyzmq + msgpack). (L344)
- `install_ros()` — `openral install ros`; re-runs the packaged `bootstrap_<os>.sh` (sudo + apt, Linux/macOS only); raises `ROSConfigError` on an unsupported platform or non-zero bootstrap exit. (L352)
- `install_list()` — `openral install list`; table of every known group with package counts + conflicts. (L390)
- `_GROUPS: Final[dict[str, list[str]]]` — `uv sync --group` extras group name → its package list; source of truth for `install_list` and the per-group commands. (L50)
- `_CONFLICTS: Final[tuple[frozenset[str], ...]]` — Mutually-exclusive group pairs (`{"libero", "robocasa"}`); checked before `_install_group` runs. (L116)

### `python/cli/src/openral_cli/robot.py`
_`openral robot vendor-urdf <id>`: expand an upstream xacro to a flat, committed URDF so end users need no xacro tooling at runtime. Defers `robot_descriptions`/`xacrodoc`/`yourdfpy` inside the command so `openral --help` stays fast._

- `vendor_urdf(robot_id, *, upstream, out_dir, rename=None, raw_text=False) -> Path` (L137) — Loads `upstream` (`rd:` xacro expanded via xacrodoc, or `file:` already-flat URDF), applies the per-robot `rename` pattern(s), and writes `<robot_id>.urdf` with a provenance header. For `rd:` upstreams, mesh paths are rewritten to portable `rd:<module>:<path>` refs rather than machine-specific absolute ones. `raw_text=True` copies an already-flat URDF's text verbatim (no yourdfpy round-trip), preserving `package://` paths and CRLF byte-for-byte, for joint-name-only patches.
- `_PROVENANCE: str` — HTML-comment provenance header (`openral robot vendor-urdf {id}` + source) written after the XML declaration of every vendored URDF. (L35)
- `_SO_ARM_JOINT_NAMES: tuple[str, ...]` — SO-ARM semantic joint names (`shoulder_pan` … `gripper`) keyed by the upstream so100/so101 URDFs' numeric joint order; shared by both follower arms' `_RAW_RENAMES`. (L42)
- `_RAW_RENAMES: dict[str, list[tuple[str, str]]]` — Per-robot raw-text `(regex, replacement)` rename pairs applied in order under `raw_text=True` (so100/so101 numeric→semantic; gr1/h1 `_joint`-suffix strip; gr1 elbow-pitch collapse). (L60)
- `_RENAME: dict[str, tuple[str, str]]` — Per-robot default rename applied to a `rd:`/`file:` upstream via `yourdfpy`. Empty today — openarm's `openarm_`-prefix strip was removed once the wire names were standardised to match the arm's own ros2_control graph. (L88)
- `_PAIR_LEN: int` — `2`; length of a `(pattern, repl)` rename tuple, named to satisfy the magic-value lint. (L91)

### `python/cli/src/openral_cli/_rskill_scaffolder.py`
_Scaffolder helper backing `openral rskill new` and `tools/rskill_scaffolder.py`._
Copies `rskills/template/` into a target directory, rewrites manifest sentinels (`name` / `license` / `embodiment_tags` / `weights_uri` / `source_repo`) plus README sentinels, then re-validates the result through `RSkillManifest.from_yaml` + `rSkill.from_yaml` so a malformed scaffold fails at scaffold-time. Partial scaffolds are cleaned up on validation failure.

- `_TEMPLATE_DIR_NAME: str` — `"template"`; the directory name under `rskills/` copied by `_default_template_dir`. (L36)
- `_MANIFEST_FILENAME: str` — `"rskill.yaml"`. (L37)
- `_README_FILENAME: str` — `"README.md"`. (L38)
- `_default_template_dir() -> Path` — Walks parents of this file until a `rskills/template/` directory is found. (L41)
- `scaffold_rskill(rskill_id, *, out_dir, owner, license_, embodiment_tag, family=None, patch=None, template_dir=None, overwrite=False) -> Path` — Public entry point; copies the template, applies family defaults + introspection patch, rewrites placeholders, re-validates the manifest. (L60)
- `_rewrite_manifest(manifest_path, *, rskill_id, owner, license_, embodiment_tag, family, patch) -> None` — Layered rewrite: (1) template baseline → (2) family defaults from `_rskill_intel.family_defaults` → (3) explicit `patch` → (4) CLI rename/license/embodiment_tags. (L173)
- `_apply_patch(raw, patch) -> None` — Overlay patch keys onto the raw manifest dict; a `None` value removes the key (so e.g. ACT family clears `min_vram_gb`).
- `_rewrite_readme(readme_path, *, rskill_id, owner) -> None` — Replace `TEMPLATE_ORG` / `TEMPLATE_ID` sentinels in `README.md`.
- `_validate_scaffold(scaffold_dir) -> None` — Round-trip through `RSkillManifest.from_yaml` + `rSkill.from_yaml`.

### `python/cli/src/openral_cli/_rskill_intel.py`
_Per-family scaffold defaults + HF Hub config introspection for `openral rskill new`._

- `RSkillFamily` — `Literal["act", "smolvla", "pi05", "xvla", "diffusion"]`; mirrors the keys of `openral_sim.registry.POLICIES` minus the mock entries. (L26)
- `_CHW_DIMS: int` — `2`; minimum `shape` length for HxW extraction from an `input_features` entry, else the template default is used. (L30)
- `RSKILL_FAMILIES: tuple[RSkillFamily, ...]` — Tuple form of the above for menu rendering / validation. (L36)
- `class RSkillPatch(TypedDict, total=False)` — Subset of manifest fields the scaffolder overlays: `model_family`, `chunk_size`, `quantization`, `latency_budget`, `min_vram_gb`, `n_action_steps`, `image_preprocessing`, `state_contract`, `action_contract`, `sensors_required`, `weights_uri`, `source_repo`, `description`. (L39)
- `family_defaults(family: RSkillFamily) -> RSkillPatch` — Per-family manifest baseline mirroring the in-tree reference manifests (`act-aloha`, `smolvla-libero`, `pi05-libero-int8`, `xvla-libero`, `diffusion-pusht`). (L67)
- `_CONFIG_TYPE_TO_FAMILY: dict[str, RSkillFamily]` — Maps a HF `config.json` policy `type` string to an `RSkillFamily`, including the `diffusion_policy` alias for `diffusion`. (L158)
- `introspect_hf(repo_id, *, default_family=None) -> tuple[RSkillFamily, RSkillPatch]` — Fetches `config.json` from a HF Hub repo, infers the policy family from `type`, and derives chunk_size / sensors / state_contract / image_preprocessing.aliases / weights_uri from `input_features`. (L168)
- `_fetch_hf_json(repo_id, filename) -> Any` — `huggingface_hub.hf_hub_download` + `json.load`; raises `ValueError` on network / parse error.
- `_sensors_from_input_features(input_features) -> list[dict]` — One `SensorRequirement`-shaped dict per `observation.images.*` feature, with min_width / min_height pulled off the CHW shape.
- `_state_dim_from_input_features(input_features) -> int | None` — Reads `observation.state.shape[0]`.
- `_aliases_from_input_features(input_features) -> dict[str, str]` — Pairs `camera<N>` source keys with the checkpoint's image-feature names; empty when names already match.

### `python/cli/src/openral_cli/autodetect.py`
_USB VID/PID enumeration and DDS topic discovery for `openral detect`._

- `class UsbDevice(NamedTuple)` — A USB serial device on the host. (L66)
  fields: `port, vid, pid, description`
- `class KnownDevice(NamedTuple)` — A known USB adapter/controller from the VID/PID table. (L82)
  fields: `chip, driver_hint, embodiment_tag, bh_robot_type`
- `class UsbMatch(NamedTuple)` — A detected device matched against the table. (L100)
  fields: `device, known`
- `class DdsTopic(NamedTuple)` — A ROS 2 topic observed during DDS scan. (L112)
  fields: `name, type_name`
- `class CanInterface(NamedTuple)` — One SocketCAN interface on the host. Defined in `openral_core.can` and re-exported here; the enumeration is transport mechanism shared with the HAL layer's bus preflight, while the robot *matching* below is this module's own concern.
  fields: `name, is_up, fd_enabled, bitrate, data_bitrate, state, driver, mtu, vid, pid, description`
- `class CanMatch(NamedTuple)` — CAN interfaces that together identify one robot; a bimanual arm contributes one per side. (L496)
  fields: `interfaces, known`
- `_enumerate_linux_pyudev() -> list[UsbDevice]` (L249)
- `_enumerate_macos_system_profiler() -> list[UsbDevice]` (L285)
- `_enumerate_glob_fallback() -> list[UsbDevice]` — `/dev/tty*` glob. (L330)
- `enumerate_usb_devices() -> list[UsbDevice]` — OS-routed enumerator. (L353)
- `match_known_devices(devices) -> list[UsbMatch]` (L385)
- `_VID_PID_TABLE: dict[tuple[int, int], KnownDevice]` — USB `(VID, PID)` → `KnownDevice` lookup driving `match_known_devices`. (L128)
- `scan_dds_topics(timeout_s=5.0) -> list[DdsTopic]` — `ros2 topic list -t`. (L412)
- `_TOPIC_ROBOT_MAP: dict[str, str]` — DDS topic name substring → robot slug, driving `infer_robot_from_topics`. (L233)
- `infer_robot_from_topics(topics) -> str | None` (L460)
- `enumerate_can_interfaces(*, sysfs_net=None) -> list[CanInterface]` — Re-exported from `openral_core.can` (see [Layer 0](00-core-schemas.md)); Linux-only, dependency-free SocketCAN enumeration via `/sys/class/net` and `ip -details -json link show`. Needs no root and opens no socket, so it never perturbs a running robot.
- `match_can_interfaces(interfaces) -> list[CanMatch]` — Groups *up* interfaces by the robot their names declare, since a CAN bus carries no vendor descriptor — only a udev-pinned interface name is a durable signal. A bare `can0` stays unmatched by design. (L586)
- `_CAN_NAME_ROBOT_TABLE: dict[str, KnownDevice]` — Interface-name substring → `KnownDevice`, driving `match_can_interfaces`. (L522)
- `infer_robot_from_can(interfaces) -> str | None` — First matched robot slug, else `None`. (L629)
- `can_adapter_name(iface) -> str` — Catalogued adapter name (`_CAN_ADAPTER_TABLE`), else the USB product string, else a vendor-only label, else `"SocketCAN"` for an on-SoC controller. Descriptive only — never used to infer a robot, since the same dongle drives any CAN machine. (L546)
- `_CAN_ADAPTER_TABLE: dict[tuple[int, int], str]` — USB `(VID, PID)` → catalogued CAN adapter name, used only for the descriptive label in `can_adapter_name`. (L536)
- `_CAN_VENDOR_TABLE: dict[int, str]` — USB VID → vendor-only fallback label for `can_adapter_name` when the PID is not catalogued. (L541)
