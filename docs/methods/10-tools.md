# Tools

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `tools/lifecycle_autostart.py`
_Drives a lifecycle node through `configure` → `activate` after `ros2 launch` brings it up. Spawned as an `ExecuteProcess` per node by `packages/openral_rskill_ros/launch/deploy_e2e.launch.py` (HAL, safety kernel, reasoner)._

- `--transition-timeout-s` — per-transition spin budget, **default `300.0`** (`lifecycle_autostart.py:161`). The kernel gets a 120 s literal and the reasoner 300 s; the **HAL's is derived per scene** by `openral_hal.sim_bringup.hal_transition_timeout_s` (`deploy_e2e.launch.py`), not hardcoded. There is still no `openral deploy sim` flag — raise the scene's `backend_options.boot_timeout_s` instead and the lifecycle budget follows.
- **Why the HAL's is derived.** The bound covers everything the scene factory does inside `on_configure`, including a sidecar's spawn + `_wait_for_boot`. Those backends carry their own, larger budgets — `isaac_sim` 900 s, `behavior` 1200 s, `robotwin` 600 s (`rlbench` 300 s, and it passes the constant directly so `backend_options.boot_timeout_s` is silently ignored) — and all four `scenes/deploy/isaac_*.yaml` plus `scenes/deploy/behavior_r1pro.yaml` set `boot_timeout_s: 1200`. Against the old fixed 300 s those five scenes declared a budget the launcher would not honour. **Measured** (`scenes/sim/isaac_franka_bowl_plate.yaml`, headless, RTX 4070 Laptop 8 GB): Isaac Sim 5.1 reaches `app ready` in **12.6 s**, so nominal boot is *not* the problem. The reachable case is a sidecar that wedges after startup — observed live on this host, where `connect()` waited the client's full **1200 s** before raising `did not answer ping within 1200s`. The transition's worst case is therefore the declared `boot_timeout_s`, not the nominal boot time. On expiry `_drive_transition` raises and the autostart exits non-zero **while `on_configure` keeps running**, which can leave the HAL parked in INACTIVE with nothing left to drive ACTIVATE and no message naming the cause. Cost of the fix: on those scenes a genuinely wedged HAL now goes unreported for up to ~21 min, which is why the 300 s floor still applies to every scene that does not ask for more.
- `--required` (flag) — treat an absent `change_state` service as a **failure** (exit 1) rather than informational. Without it an autostart whose target node was never built exits 0, so a graph missing that node comes up looking healthy. `deploy_e2e.launch.py` passes it for `/openral_deadman_watchdog` and pairs it with an `OnProcessExit` → `Shutdown`, so a deploy that cannot arm its only independent E-stop source refuses to run instead of running unprotected.
- `_STATE_TO_TRANSITION: dict[str, list[int]]` (L36) — `--target` → the `Transition` constants to drive in order (`inactive` = CONFIGURE; `active` = CONFIGURE then ACTIVATE).
- `main() -> int` (L118) — CLI entry; parses `--node` / `--target` / `--required` / `--service-timeout-s` / `--transition-timeout-s` (documented above) and drives the transitions via `_drive_transition`.

### `tools/profile_policy_load.py`
_One-shot wall-time breakdown of a single policy load. Drives `openral_sim.factory.make_policy` against an in-tree rSkill manifest and prints a phase-by-phase summary built from every `<prefix>_<name>_{start,done}` event emitted by `openral_rskill._diagnostics.phase_timer`. Use when `ros2 launch openral_rskill_ros …_e2e.launch.py` or `openral sim run` is slow to first action — answers "where do the seconds go" before changing any code._
_Wired as `just profile-load <rskill> [args]` (Justfile). Measured on an RTX 4070 with a small ACT policy (warm cache): imports 4.6 s (54%), snapshot 0.3 s, from_pretrained 0.7 s, to_device 0.0 s, 8.5 s end-to-end — the import tax dominates even a small ResNet+transformer policy. Families whose adapter lacks a `_<family>_phase` helper (`xvla`, `diffusion`) report "no phase_timer events captured"; the end-to-end total is still valid._


- `class _PhaseCapture` — `structlog` processor that buffers `_start` / `_done` events and pairs them by name. Insertion-ordered so the rendered table mirrors the actual load order. (L35)
- `_parse_args(argv) -> argparse.Namespace` — `--rskill <dir>` (required), `--device` (default `auto`). (L69)
- `_build_env_cfg(rskill_dir, *, device) -> _SimpleEnvCfg` — Builds a minimal env_cfg from `<rskill_dir>/rskill.yaml`; mirrors `rskill_runner_node._SimpleEnvCfg`. (L103)
- `_render(pairs, total_s) -> str` — Formats the captured pairs as `phase / elapsed_s / share` columns plus an `(unaccounted)` row when phase coverage misses >1 s. (L122)
- `main(argv=None) -> int` (L149) — Late-imports `openral_sim.factory.make_policy` so the import cost lands inside the profiled total; reports `HF_HUB_OFFLINE` status alongside the result.

### `tools/viz_collision.py`
_Overlays a robot's **kernel** collision primitives (the box/capsule geometry the C++ safety kernel checks, lowered by `collision_params_from_description`) on its real MJCF meshes at any joint pose — the offline way to eyeball whether the SO-101 `base` OBB (issue #84) hugs the housing and clears the folded distal links without a `deploy run`. Standalone inspection tool, not a pytest test. Run the venv python with `PYTHONPATH=packages/openral_safety`._

- `--viewer` — interactive MuJoCo window (`MUJOCO_GL=glfw`). `--screenshot PATH` — offscreen PNG (`MUJOCO_GL=egl`). `--rviz` — real RViz (spawns `robot_state_publisher` for RobotModel + TF and publishes the primitives as a latched `/collision_markers` MarkerArray; needs ROS sourced). `--robot <id>` (default `so101_follower`), `--deg <j...>` sets the pose in degrees (manifest joint order). Box = translucent red, capsules = translucent blue (cylinder + end spheres in RViz).
- `main(argv=None) -> int` (L334) — CLI entry; builds the robot's kernel collision primitives and dispatches to the viewer/screenshot/RViz renderer named above.

### `tools/generate_tight_geometry.py`

_Derives a robot manifest's `tight_geometry` blocks (`openral_core.TightCollisionGeometry`) from its **real** collision meshes — the offline producer for the safety kernel's staged 26-DOP → exact-convex-hull world-voxel narrow phase. Deliberately **not** a fit: the 26-DOP is the intersection of 26 *tangent* halfspaces `u·x <= max over the mesh of u·x`, the hull is `conv(mesh vertices)`, so containment is definitional and no optimiser tolerance enters the argument. Both are emitted in the manifest box's own local frame, and the tool refuses anything that does not sit inside that box. Mesh placement follows the GEOM transform only — MuJoCo folds mesh recentring into the geom frame, so applying `mesh_pos` as well double-counts it (PR #158 hit exactly this), and the tool asserts that premise rather than trusting it. Standalone operator/CI tool, not a pytest test; needs `mujoco` + `scipy` + `robosuite`. See [`collision-hull-narrow-phase.md`](../reference/collision-hull-narrow-phase.md)._

_Two modes. `emit --robot robots/<name>/robot.yaml` prints the YAML fragment to paste into the manifest, annotated per link with mesh/hull vertex counts, whether stage 2 is on, the DOP's achieved inward margin inside the shipped box, and — for a link that ships a hull — its `hull_overhang_m` (issue #221). `check --robot …` re-derives from the mesh and verifies every declared block still contains it AND that a hull's declared `hull_overhang_m` is not exceeded by an independent, finer re-sample, exiting **3** on any failure — the same fail-closed exit `openral collision lower --write` uses when it refuses to loosen a shipped envelope._

- `REPO_ROOT: Path` (L42) — Repo root, derived from this file's location.
- `PANDA_GEOM_OF_LINK: dict[str, str]` (L46) — Panda manifest link name → MJCF collision geom name (`panda_link{i}` → `link{i}_collision`).
- `link_mesh_in_box_frame(xml_path, geom_name, origin_xyz_rpy) -> np.ndarray` (L88) — Collision-mesh vertices of `geom_name` in the compiled MJCF, expressed in the manifest box's frame (geom transform, then the inverse of the manifest's `origin_xyz_rpy` under the kernel's `transform_from_xyz_rpy` rpy convention). Raises when the geom is missing, is not a mesh geom, or when `mesh_pos != geom_pos` (the geom-only placement premise fails for that asset).
- `link_mesh_faces(xml_path, geom_name) -> Points` (L124) — Triangle face indices of `geom_name`'s collision mesh, paired with `link_mesh_in_box_frame`'s vertices for `hull_overhang_m`'s sampled-surface check.
- `hull_overhang_m(...)` (L152) — Sampled lower bound on how far a declared hull envelope sits outside the real mesh surface, batched (`_OVERHANG_BATCH`) and capped (`_OVERHANG_MAX_SAMPLES`) to bound peak memory.
- `derive_tight_geometry(points, half_extents) -> dict[str, Any]` (L307) — Build the DOP slabs (min/max of the points projected on `DOP_AXES`) and, when the exact hull fits `MAX_TIGHT_HULL_VERTICES`, its vertex list; over the ceiling the hull is dropped and the link ships stage 1 only. Returns a mapping ready for `TightCollisionGeometry` plus the diagnostics a reviewer needs — `_hull_vertex_count`, `_stage2`, and `_dop_inward_margin_m` (the achieved clearance of the DOP inside the shipped box).
- `ROBOT_MESH_SOURCES: dict[str, tuple[str, dict[str, str]]]` (L50) — Robot name → (MJCF path under the robosuite asset root, manifest-link-name → MJCF-geom-name map). The registry a new robot must be added to before either mode will run for it; manifest link names are URDF/TF names while robosuite's MJCF uses bare `linkN`, so the mapping cannot be inferred.
- `main(argv=None) -> int` (L498) — CLI entry; `emit --robot <path>` / `check --robot <path>` subcommands.

### `tools/schema_export.py`
_Generates JSON Schema files for every public `openral_core` model._

- `_enum_schema(cls) -> dict[str, Any]` — Minimal JSON Schema for a `str` Enum. (L179)
- `export_schemas(out_dir=_OUT_DIR) -> dict[str, Any]` — Export JSON Schema for every public model. (L191)
- `check_drift(out_dir=_OUT_DIR) -> bool` — On-disk schemas == regenerated. (L243)

### `tools/check_repo_state_map.py`
_The other pre-commit drift guard (`always_run`, ~0.2 s): pins the mechanically checkable half of `docs/architecture/repo-state-map.html` — that its `pkg:` pointers name something real, and that its asserted counts have not rotted. Both classes have shipped to master (a green card against an `examples/` directory that never existed; all five counts wrong at once, unit 190 vs 379). Prose on the map stays a human judgement call._

- `REPO_ROOT` (L37) — Repo root, derived from this file's location.
- `MAP_PATH = REPO_ROOT / "docs" / "architecture" / "repo-state-map.html"` (L38) — The map file this script checks.
- module const `COUNT_TOLERANCE: float` (L45) — counts are held to within 10%, not to the digit; an exact check would go red on every added test file and be disabled long before it caught the next 190-vs-379.
- `iter_cards(html: str) -> list[tuple[str, str]]` — Pair each `pkg:` value with the `desc:` that follows it in the same card. (L71)
- `_resolves(head: str, token: str) -> bool` — Whether a `pkg:` token names a real path under any of the three spellings the map uses: repo-root-relative, relative to the card's leading package under the workspace src layout, or a bare module name. (L80)
- `check_paths(cards) -> list[str]` — Report `pkg:` tokens that name nothing on disk. (L101)
- `check_counts(cards) -> list[str]` — Report `desc` counts that have drifted past `COUNT_TOLERANCE`. (L120)
- `main(argv: list[str] | None = None) -> int` (L139) — CLI entry; `--quiet`; runs `check_paths` + `check_counts`, prints drift and returns 1, else 0.

### `tools/audit_sim_configs.py`
_Real GPU rollout audit for every YAML under `scenes/`. Operator-driven (not a pytest test); 1 episode per config; writes `outputs/audit_sim_configs.json` and prints a Markdown table. See `just sim-audit`. Two modes: default (full rollout for sim/benchmark, Tier-2 launch + SIGINT for deploy) and `--check-compatibility` (cheap in-process scene+rSkill+HAL gate, no subprocess / no GPU)._

- `REPO_ROOT: Final[Path]` (L39) — Repo root, derived from this file's location.
- `SCENES_DIR: Final[Path] = REPO_ROOT / "scenes"` (L40) — Root of the scanned scene tree.
- `OUTPUT_DIR: Final[Path] = REPO_ROOT / "outputs"` (L41) — Where `audit_sim_configs.json` is written.
- `DEFAULT_TIMEOUT_S = 600` (L42) — Default `--timeout`, 10 min/config — covers cold weights download.
- `DEFAULT_DEPLOY_ALIVE_GRACE_S = 90` (L43) — Default `--deploy-alive-grace`: how long a deploy-mode graph is left running before SIGINT.
- `DEFAULT_DEPLOY_SHUTDOWN_GRACE_S = 30` (L44) — Default `--deploy-shutdown-grace`: how long to wait after SIGINT before escalating to SIGKILL.
- `RunMode = Literal["sim", "benchmark", "deploy"]` (L46) — Tier selector on each `ConfigSpec` row; drives `_run_one` vs `_run_one_deploy` dispatch.
- `@dataclass(frozen=True) class ConfigSpec(config, rskill, uv_group, run_mode)` (L50) — One row in the audit catalogue. `uv_group` is one of `libero / metaworld / robocasa / maniskill3 / simpler-env / sim`; `run_mode` is `"sim"` / `"benchmark"` / `"deploy"`; `rskill` is `""` for deploy rows (env-only — reasoner picks at runtime). Catalogue holds only pairs that actually exist in the tree — scenes without a matching in-tree rSkill are tracked in the scene YAML itself (and in `tests/unit/test_examples_sim_configs_load.py` for schema-load coverage), not as audit rows.
- `CATALOGUE: tuple[ConfigSpec, ...]` (L78) — Explicit (YAML → rSkill → uv group → run_mode) mapping for every config currently in the tree: 13 sim + 7 benchmark + 4 deploy = 24 rows.
- `@dataclass class AuditRow(config, rskill, status, exit_code, wall_s, peak_vram_mib, tail)` (L174) — One result. `status` ∈ {`pass`, `pass-compat`, `fail-oom`, `fail-asset`, `fail-sidecar`, `fail-timeout`, `fail-other`, `fail-compat`, `skipped-opt-dep`, `skipped-host-setup`}.
- `_classify(returncode: int, tail: str) -> str` (L225) — Map subprocess result to a status by substring-matching stderr against `_OOM_PATTERNS` / `_ASSET_PATTERNS` / `_SIDECAR_PATTERNS` / `_OPT_DEP_PATTERNS` / `_HOST_SETUP_PATTERNS`. Exit 139 (MuJoCo/GL atexit SIGSEGV in gym-aloha) is treated as `pass` when no error patterns appear.
- `class _VramSampler` (L264) — Background `nvidia-smi --query-gpu=memory.used` poller, 200 ms cadence; `peak_mib` reported on `.stop()`. No-op without `nvidia-smi` on `$PATH`.
- `_check_compat(spec: ConfigSpec) -> AuditRow` (L309) — `--check-compatibility` gate: load scene via `openral_core.load_scene_strict`, validate rSkill manifest (sim/benchmark) or assert robot resolves in `openral_cli.deploy_sim._ROBOT_HAL_REGISTRY` (deploy). No subprocess, no GPU. Returns `pass-compat` / `fail-compat`.
- `_build_run_cmd(spec: ConfigSpec) -> list[str]` (L397) — Build the `uv run … openral <sim|benchmark> …` argv for sim/benchmark rows. Refactored out of `_run_one` so the deploy path can stay focused on lifecycle teardown.
- `_run_one_deploy(spec, *, alive_grace_s, shutdown_grace_s, timeout_s) -> AuditRow` (L446) — Tier-2 deploy launch via `openral deploy sim --config <yaml> --no-dashboard`: `Popen` in its own process group, wait `alive_grace_s`, send SIGINT to the group, wait `shutdown_grace_s`, escalate to SIGKILL on timeout. Pass criteria: banner seen in stdout AND returncode in `{0, -SIGINT, 130, -SIGTERM}`.
- `_classify_or_fallback(returncode, tail, spec, wall_s, peak_vram) -> AuditRow` (L617) — Deploy-mode wrapper around `_classify` that defaults to `fail-other` when no pattern matches (sim path defaults to `pass`).
- `_run_one(spec: ConfigSpec, timeout_s: int) -> AuditRow` (L656) — Tier-3 sim/benchmark rollout via `_build_run_cmd(spec)` with `MUJOCO_GL=egl` and `OPENRAL_SIM_SEQUENTIAL_INIT=1`.
- `main(argv) -> int` (L794) — CLI entry; flags `--timeout` / `--deploy-alive-grace` / `--deploy-shutdown-grace` / `--check-compatibility` / `--report`. Returns 0 on all-pass, 1 if any config failed, 2 on filter mismatch.

### `tools/validation_matrix.py`
_The four-scene collision-stack validation matrix as one versioned command. Recovers the tooling that lived only in `spark:~/openral-runs/<round>/scripts/` (`run_matrix.sh`, `drive_round.sh`, `attach_monitor4.py`, `postprocess.sh`, `adjudicate.py`, `verdict_table.py`) and emits both `NOTES.md` and a machine-readable `verdicts.json` (`openral_core.ValidationRoundVerdicts`) per round. Backs `just validation-matrix` / `-verdicts` / `-diff`. See [`docs/contributing/validation-matrix.md`](../contributing/validation-matrix.md) and the ledger it feeds, [`docs/reference/collision-validation-evidence.md`](../reference/collision-validation-evidence.md)._

- `REPO_ROOT: Final[Path]` (L60) — `Path(__file__).resolve().parents[1]`.
- `OUTPUT_ROOT: Final[Path]` (L61) — `REPO_ROOT/"outputs"/"validation-matrix"`.
- `DEFAULT_RSKILL_ID: Final[str]` (L119) — `"OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"`.
- `LAUNCH_FAILED_MARKER: Final[str] = "launch_failed.txt"` (L1089) — Suffix of the runner's own launch-failure marker file, the first thing `detect_launch_failure` checks for.
- `DISPATCH_READY_TIMEOUT_S: Final[float] = 180.0` (L2182) — Timeout bound for the goal re-dispatch loop when the graph answers but is not assembled yet (paired with `DISPATCH_RETRY_INTERVAL_S`).
- `@dataclass(frozen=True) class SceneSpec(key, config, prompt, deadline_s)` — One matrix row. `config` is the **tracked** DeployScene YAML; the round launches a resolved copy of it carrying the seed and the pins that have no CLI flag. (L79)
- `MATRIX: tuple[SceneSpec, ...]` — The four scenes: `baguette`, `sink_cup`, `fridge`, `utensil`. (L96)
- `SYNC_GROUPS = ("robocasa", "sidecar-wire")` — Both, always: `--group robocasa` alone strips `pyzmq` and breaks the XR-1 adapter. (L123)
- `STACK_ARGV: tuple[str, ...]` — The flag-pinnable stack: SLAM/Nav2/octomap/kernel-check on, detector + scene VLM off, headless. (L128)
- `SCENE_RUNTIME_PIN: tuple[tuple[str, bool], ...]` — `enable_reasoner=False`: the one pinned knob with **no CLI flag**, spliced into the resolved scene copy because `deploy sim` reads it from the scene and defaults it to `True`. (L169)
- `LEGACY_SCENE_DIRS: tuple[tuple[str, tuple[str, ...]], ...]` — Scene key → the directory names the pre-harness rounds used (`bag1`, `sink1`, `fridge1`, `utensil1`), so `import-round` needs no hand-mapping. (L196)
- `quantization_budget_m(grid_resolution_m: float) -> float` — Half the voxel's body diagonal; the largest kernel-vs-ground-truth discrepancy a correct grid can produce. (L207)
- `collision_scale_env() -> dict[str, float]` — The #188 graded-velocity band the round will run with, read from the `OPENRAL_COLLISION_SCALE_*` env vars that `deploy_e2e.launch.py` consumes. Recorded rather than refused, because `assert_no_safety_overrides` inspects argv and cannot see them, and arming the band is the point of the A/B battery. (L1761)
- `parse_kernel_collision(lines) -> ValidationStopEvidence | None` — Transcribe the first `safety.collision` line verbatim. (L227)
- `parse_json_log_line(lines, event) -> dict[str, Any] | None` — Payload of the first `<event> {...}` line (`sim.task_success_final`, `sim.estop_ground_truth_snapshot`, `sim.estop_initial_configuration`). (L285)
- `read_monitor(path) -> list[dict[str, Any]]` — Load a monitor JSONL, skipping non-object lines. (L319)
- `grid_resolution_from_monitor(records) -> float | None` — Cell size from the first `world_voxels` record; the budget is read from the run, never assumed. (L343)
- `monitor_subscription_records(records) -> int` — Record count excluding the monitor's own `monitor_started` / `monitor_stopped` pair, i.e. how much it actually **received**. `0` on a monitor that ran the whole scene is the 2026-08-23 defect signature — its DDS participant was created before `openral deploy sim` purged `/dev/shm/fastrtps_*`, so all 24 runs of that round wrote exactly two lines. A harness fault, and it must not read as "the run stopped before there was anything to see". (L366)
- `build_witness_timeline(records, deploy_lines) -> ValidationWitnessTimeline` — Producer side from the monitor, consumer side from the kernel's own `safety.support_witness_*` / `safety.place_region_*` lines. (L388)
- `probe_is_collidability_filtered(snapshot) -> bool` — Whether every non-empty coverage block in the snapshot carries `noncollidable_side_geoms_excluded`, i.e. the HAL filtered **both** sides of each probe to solid geoms. `False` on anything recorded before that filter existed, where a 0 m pair may be a purely visual mesh. (L490)
- `hal_admissible_gap_m(snapshot, stop) -> float | None` — The HAL's own kernel-vs-probe budget for this stop, dispatched by stop class: `payload_world_voxel` for a payload-vs-world stop (#266), `self_collision` for an attached-payload **self** stop (OBB on both sides, no voxel; 124.6 mm on `panda_mobile`), `link_link` for a bare link pair, and the top-level `admissible_gap_m` for an arm-link-vs-voxel stop (88.2 mm). `None` for a snapshot predating the budget. **The payload-world branch is a fix, not an addition:** that class routed to the top-level block until #266, charging `corner_slop(worst LINK) + voxel_half_diagonal` to a stop no robot link is a party to — 88.2 mm where the payload's own model needs ~31 mm — and it is **97 %** of the 2026-09-10 A/B's 15 mm stops and 79 % of its 25 mm ones. An over-large budget does not fail loudly, it silently excuses, which is the direction that hides a real defect. A snapshot recorded before the block existed still falls through to the old number rather than losing its budget. (L558)
- `_payload_world_gap_m(budget, grid_resolution_m) -> float | None` — The payload-vs-world-voxel half of `hal_admissible_gap_m` (#266): `payload_model_overhang + voxel_half_diagonal`, read off the snapshot's `payload_world_voxel` block. Deliberately **not** `max_payload_corner_slop_m` — that is the box's term and belongs to the self-collision block, and charging it here over-budgets a refined primitive by exactly what the refinement recovered (24.7 / 20.5 / 24.1 mm on the three RoboCasa payloads measured). **The voxel term is re-derived when the snapshot's is zero**, which is not belt-and-braces: `estop_ground_truth_snapshot` fills it only from an `evidence_voxel` it was handed, and on the top-level block that omission hides behind a 45-88 mm link term while here the overhang is 8.9-19.9 mm — the *same order* as the 21.65 mm dropped, so composing with zero halves the budget and convicts correct stops (the 2026-09-11 A/B flagged 6 of 16 hull-arm stops that way before the term was restored; 6 → 1 after). With no resolution to re-derive from the budget is `None` — `unadjudicated` rather than a number that convicts, because an under-stated budget cries wolf and that is the one direction an adjudicator must not fail in. A **present but unusable** block returns `None` rather than falling through to the top-level one: falling through would hand the stop the worst link's corner slop, the wrong-pair budget this exists to stop charging. Only a snapshot with **no** block falls through.
- `_arm_world_gap_m(budget, grid_resolution_m) -> float | None` — The arm-link-vs-world-voxel half: `corner_slop(worst link) + voxel_half_diagonal`, **re-deriving the second term** when the snapshot's is zero, for the same reason `_payload_world_gap_m` does. Every snapshot in the 2026-09-11 A/B reported `voxel_half_diagonal_m: 0.0` (no `evidence_voxel` was handed to the producer), so every arm-stop budget was the link term alone — 88.22 mm where it should have been 109.87 mm, understated by 21.65 mm, which is 25-48 % of a Panda link's 45-88 mm slop. It hid here because that term is large; on the payload block the 8.9-19.9 mm overhang made the same omission impossible to miss. An under-stated budget convicts a conservative, correct stop. Falls back to the published composition when the slop term or the resolution is absent, so a snapshot shaped differently keeps whatever budget it carries.
- `_arm_world_gap_m(budget, grid_resolution_m) -> float | None` — The arm-link-vs-world-voxel half: `corner_slop(worst link) + voxel_half_diagonal`, **re-deriving the second term** when the snapshot's is zero, for the same reason `_payload_world_gap_m` does. Every snapshot in the 2026-09-11 A/B reported `voxel_half_diagonal_m: 0.0` (the producer was handed no `evidence_voxel`), so every arm-stop budget was the link term alone — 88.22 mm where it should have been 109.87 mm, understated by 21.65 mm, which is 25-48 % of a Panda link's 45-88 mm slop. It hid here because that term is large; on the payload block the 8.9-19.9 mm overhang made the same omission impossible to miss. An under-stated budget convicts a conservative, correct stop. Falls back to the published composition when the slop term or the resolution is absent, so a snapshot shaped differently keeps whatever budget it carries.
- `_link_link_gap_m(budget, stop) -> float | None` — The link-vs-link half (#216), extracted alongside it so the dispatcher above reads as one line per stop class.
- `probe_is_distance_certified(snapshot) -> bool` (L519) — Whether every non-empty coverage block reports `uncertified_pairs` and that count is zero, i.e. the HAL measured with `openral_hal.convex_distance` rather than `mujoco.mj_geomDistance`. Same shape and same fail-closed consequence as `probe_is_collidability_filtered`, for a different defect: that call is wrong by 15-108 mm on RoboCasa-fixture-vs-panda-mesh pairs, and every one of the observed cases is a `0.000 m` reading — exactly what rule 1 promotes to `real-contact`. `False` on every round recorded before 2026-08-25.
- `adjudicate_ground_truth(snapshot, stop, grid_resolution_m, *, monitor_records=None) -> ValidationGroundTruthAdjudication | None` — Distance-probe adjudication (the contact list is not an emptiness test): any pair ≤ 0 m → `real-contact`, **but only when `probe_is_collidability_filtered`** — otherwise `unadjudicated`, because the pair may be a visual shell (`robot0_g42_vis` at 0.000 m from a freezer door whose collision geom was 2.5 mm clear); **Which pair set answers is chosen from the stop's other side, not from `involves_payload`** — a payload against a `voxel_*`/`place:*` reads `nearest_payload_world_pairs`, a payload against a robot link reads `nearest_payload_robot_pairs`, and the matching `*_coverage` block travels with it so an untruncated probe cannot assert `distmax` about a probe that was never consulted (#228, #208 one class over); absence of the right pair set is `unadjudicated`, never a fallback to the wrong one; discrepancy beyond the admissible gap → `false-positive`, **but only when `budget_source == "hal-adjudication-budget"`** — the voxel term alone is a lower bound on that gap, so exceeding it proves nothing and yields `unadjudicated`, while falling within it still soundly proves `within-quantization`; truncated/absent probe or unknown budget → `unadjudicated`. **Whatever the ladder concludes is then withdrawn to `unadjudicated` unless `probe_is_distance_certified`** — every rule above reads a number off the probe, and the instrument every pre-2026-08-25 round used is unreliable for this exact pair class. The check runs **last** rather than first so the record still names what it takes (`withdrawn from 'within-quantization': …`) instead of erasing it; it withdraws verdicts and reverses none. The gap is `hal_admissible_gap_m` when the snapshot carries one and `quantization_budget_m` only as a fallback — the voxel term alone is 21.7 mm against the HAL's 88.2 mm and turned conservative, correct stops into false positives. `monitor_records` lets a missing budget name its cause. (L768)
- `detect_launch_failure(run_dir, stem, deploy_lines) -> str` — Why a scene is not a run at all: the runner's `<stem>_launch_failed.txt` marker, an `[ERROR] [launch]: Caught exception in launch` line (`ros2 launch` threw and unwound, leaving a partial graph and a long log — no marker, no usage banner, so it used to read as a clean run), a `Usage: openral …` banner (the CLI rejected its own argv), no log, or — since #256 — a Nav2 bond teardown (`_nav2_bond_teardown`) or a graph that never came up (`_lifecycle_never_came_up`). Empty when the scene ran. This is what `artifacts_complete` reads — `bool(deploy_lines)` was not the test, because click's usage error is lines. (L1236)
- `_nav2_bond_teardown(deploy_lines) -> str` — The fifth way, and the only one that leaves a *healthy-looking* log. `lifecycle_manager_navigation` tears the whole Nav2 stack down when a managed server misses its bond heartbeat (default 4 s, raised to `BOND_TIMEOUT_S = 30.0` in `openral_nav2_bringup/launch/nav2.launch.py`): no traceback, no non-zero exit, the graph simply goes inert and idles out its deadline, and the harness scored the corpse as `deadline-no-grasp` — the policy failing to grasp. 31 of the 89 valid runs in the 2026-09-06 ceiling battery died this way, 25 naming `controller_server`. The same message also appears late in runs that did their work, so the discriminator is *when*: `_NAV2_BOND_LOSS_EARLY_S = 120.0` sits in a measured 99 s empty gap (worst dead run lost its bond at `t0 + 100.8 s`, earliest loss in a run that had completed its task at `t0 + 199.6 s`). Anchored on the log's first stamp and gated on `_DEPLOY_BANNER_PREFIX`, so a `<stem>_deploy_excerpt.log` — which begins mid-run — is declined rather than mis-anchored: a missed teardown leaves the old behaviour, a false one would silently drop a real result out of the denominator. `tests/unit/test_validation_matrix_nav2_bond.py`.
- `_lacks_stage2_hull(link) -> bool` — Whether a link is *known* to carry no stage-2 hull (#260), from `has_stage2_hull` in the snapshot's `collision_model_slop`. Only an explicit `False` counts: a snapshot recorded before the HAL published the field omits it, and absence must read as "unknown", never "no hull" — the other way round would hand the box budget to genuine hull measurements on every historical round and turn correct stops into `within-quantization`. Used by `_link_link_hull_gap_m`.
- `dispatch_not_ready_reason(goal_log) -> str` (L2186) — Why a dispatch says the graph was not assembled yet, or `""` (#263). The action server answering does not mean the graph is up: the TF tree can still be two disjoint trees (`ConnectivityException … not part of the same tree`) and a declared camera can still have published nothing (`ROSConfigError: XR-1 expected camera …`). Such a goal returns in ~0.4 s with zero chunks and, because the graph survives and prints `sim.task_success_final` at teardown, reads as an ordinary non-completion — 6 + 17 of the 78 goal logs in the 2026-09-10 ceiling battery, every one scored against the policy. Matches only that class, so a real E-stop, deadline or capability mismatch is never retried, and a goal that delivered any action chunk is never in it whatever it says afterwards. `DISPATCH_READY_TIMEOUT_S` / `DISPATCH_RETRY_INTERVAL_S` bound the re-dispatch; `tests/unit/test_dispatch_readiness.py`.
- `DISPATCH_RETRY_INTERVAL_S: float` (L2183) — `12.0`; how often `dispatch_not_ready_reason` triggers a re-dispatch.
- `_NAV2_BOND_LOSS_EARLY_S: float` (L1129) — `120.0`; a bond-loss timestamp inside this window of the log's start is scored as `_nav2_bond_teardown`, not an ordinary non-completion.
- `_lifecycle_never_came_up(deploy_lines) -> str` — The sixth way: a lifecycle node never completed a transition (`RuntimeError: transition 'configure' on '/openral_hal_panda_mobile' did not advance the FSM within 300.0s`), so the graph never came up and nothing after it is a policy outcome. Loud, unlike the bond teardown — it raises — but it landed in the same `deadline-no-grasp` bucket, because that bucket is defined by absence and a graph that never started produces absence too. Needs no threshold and no clock: a completed transition is a precondition for a run existing at all. In the 2026-09-06 ceiling battery it appears in 12 of 89 valid runs, every one `deadline-no-grasp`, and in no run that completed its task or was stopped by the kernel. With `_nav2_bond_teardown` it voids 43 of the 47 starved runs and no healthy one.
- `parse_goal_log(lines) -> tuple[dict[str, Any] | None, str]` — The dispatcher's single JSON status line, or why it never wrote one. A raising `_validation_matrix_dispatch.py` leaves a Python traceback with no `status` in it, which left `dispatch_failure_reason` empty on a run that never dispatched anything. (L1306)
- `classify_outcome(*, task_success_ever, stop, ground_truth, initial_configuration, grasped, artifacts_complete) -> str` — Buckets one scene. Ordering is load-bearing: success wins outright, and `estop-initial-configuration` outranks the ground-truth adjudication. (L1347)
- `scene_verdict_from_artifacts(run_dir, *, scene, config_path, seed, prompt, rskill_id, stem) -> ValidationSceneVerdict` — Derive one scene's verdict from recorded artifacts only. (L1401)
- `diff_rounds(current, baseline) -> ValidationRoundDiff` — Field-by-field round comparison over `_DIFF_FIELDS`. Carries both rounds' **seed** as well as their SHA: a comparison is `reproducibility` only when both match, since the seed decides the scene's initial configuration and a seed-1-vs-seed-2 pair at one SHA is a before/after of two different scenes. (L1572)
- `class GuardrailError(RuntimeError)` — A precondition is not met; raised, never warned. (L1631)
- `assert_worktree_clean() -> None` — Refuse a dirty worktree: its recorded SHA would be a lie. (L1644)
- `assert_sha(expected: str | None) -> str` — Return `HEAD`, refusing when it is not the requested checkout. (L1657)
- `assert_overlay_fresh(install_dir: Path) -> int` — Refuse an `install/` older than any tracked `.cpp/.hpp/.h/.msg/.idl` under `cpp/` or `packages/`. (L1678)
- `resolve_launcher() -> Path` — This checkout's `.venv/bin/openral`, by absolute path; the `~/.local/bin` wrapper execs the **parent** checkout. (L1720)
- `assert_sidecar_wire() -> None` — Refuse when `pyzmq` is absent. (L1744)
- `assert_no_safety_overrides(argv) -> None` — Refuse any argv token matching `_SAFETY_KNOB_PATTERNS` (normalised lowercase, `-`→`_`). (L1824)
- `scene_safety_surface(document) -> dict[str, object]` — The safety-relevant keys of a parsed DeployScene: `safety` / `hal` / `extra_allowed_collision_pairs` / `place_declaration` wholesale, `runtime.enable_octomap_kernel_check`, plus anything whose leaf name reads as a margin/tolerance/allowance/limit/watchdog/E-stop. Stack composition (reasoner, SLAM, Nav2, octomap, detector, VLM) is deliberately **not** in it. (L1880)
- `assert_scene_safety_unmoved(tracked, resolved) -> None` — The second control surface: refuse a materialised scene copy that adds, removes or changes any key of `scene_safety_surface` relative to the tracked scene. (L1907)
- `gpu_status() -> tuple[str | None, list[str]]` — GPU name + resident compute processes; the host is shared. (L1945)
- `pin_runtime_block(text, pins) -> str` — Splice `runtime:` pins into a scene YAML, changing nothing else (comments and safety commentary survive verbatim). (L1979)
- `materialise_scene(spec, seed, run_dir) -> tuple[str, Path]` — Write the round's resolved scene copy (seed + `SCENE_RUNTIME_PIN`), re-parse it to prove the pins landed, and check it against the tracked scene. The tracked file is never touched. (L2030)
- `wait_for_dds_transport_ready(deploy_log, proc, *, timeout_s, poll_s=0.05) -> str` — Block until the deploy log carries `openral_cli.deploy_sim.DDS_TRANSPORT_READY_MARKER`, i.e. the `/dev/shm/fastrtps_*` purge is done and `ros2 launch` has not spawned yet. The monitor is started on the far side of it: a participant created earlier loses its shared-memory segments silently and receives nothing for the rest of the scene. Early-stop coverage is unaffected — the sim clock does not start until the HAL node comes up, tens of seconds later. Returns `""` on timeout or a dead deploy, and the caller records that it started anyway. (L2115)
- `render_notes(verdicts) -> str` — The round's Markdown summary, written as a by-product of running; names the scenes whose **monitor received nothing** (a harness fault — the evidence is missing) *separately* from those that stopped before the monitor saw a voxel grid (a fact about the run), any whose probe was not collidability-filtered, any judged against a lower-bound budget only, and any that did not run at all. (L2430)
- `round_exit_code(verdicts) -> int` — `4` when any scene bucketed `harness-error`, else `0`: a round in which a scene never launched must not exit successfully. (L2559)
- `parse_launch_argv(lines) -> list[str]` — The resolved `argv: … launch …` the deploy CLI echoed: the only artifact stating the stack a run actually got; head-agnostic, so a venv-wrapped `ros2` still parses. (L2651)
- `stack_tokens(argv) -> list[str]` — The stack-defining `key:=value` tokens of that argv, per-scene tokens dropped. (L2694)
- `robot_facts_from_launch_argv(argv) -> dict[str, str]` — `repo_root` / `robot_id` / `robot_manifest_path`, from the argv's `robot_yaml:=` token. (L2713)
- `parse_log_start_time(lines) -> str | None` — UTC timestamp of the log's first ROS stamp; pre-harness rounds recorded no start time, their logs did. (L2742)
- `resolve_scene_dirs(round_dir, aliases) -> dict[str, str]` — Map each matrix scene onto the directory a round kept it in; `--scene-alias` wins over `LEGACY_SCENE_DIRS`. (L2766)
- `cmd_verdicts(round_dir, *, stem=None) -> int` (L2575) — `stem=None` reads the round's recorded `artifact_stem`.
- `cmd_diff(round_dir, baseline_dir, out_path) -> int` (L2626) — `verdicts` subcommand body: field-by-field round comparison via `diff_rounds`.
- `cmd_import(args) -> int` (L2796) — `import-round` subcommand body.
- `cmd_run(args) -> int` (L2885) — `run` subcommand body.
- `main(argv=None) -> int` — CLI entry; `run` / `verdicts` / `diff` / `import-round`. `3` on a guardrail refusal (nothing written), `4` when a scene bucketed `harness-error`. (L2965)
- `octomap_resolution_env() -> dict[str, float]` — The world-voxel resolution the round will actually run with, read from `OPENRAL_OCTOMAP_RESOLUTION_M` (which `deploy_e2e.launch.py` honours in `[0.001, 0.5]`, and from which the kernel's `world_voxel_max_cells` is derived). Same mechanism as `collision_scale_env`, opposite direction: a **finer** grid shrinks the kernel's quantisation term, so the round is *less* conservative than the shipped default. Returns `{}` — not a value — when the override is absent or out of range, because the round then ran at the default and metadata claiming otherwise would misdescribe it. (L1792)

### `tools/_validation_matrix_monitor.py`
_Private helper of `validation_matrix.py`, spawned alongside each scene's ROS graph. Recovered verbatim from `attach_monitor4.py`. Records the attachment stream, the kernel's `FailureTrigger.evidence_json`, the occupied-cell **set** hash (a frozen map keeps an identical set, which a count cannot detect), the `PlaceDeclaration` + producer-measured region, and an `.npz` snapshot at every E-stop **and** periodically while a payload is carried (an E-stop-only monitor records nothing on a run that passes). Needs ROS 2 sourced._

- `class Monitor` (L164) — Owns the ROS subscriptions + state for the attachment/failure/place/voxel recording described above; `__init__(node, sink, snap_dir)`.
- `main() -> int` (L434) — CLI: `sys.argv[1]`=output path, `[2]`=snapshot dir; creates the node, runs `Monitor`, spins until SIGINT/SIGTERM, writes `monitor_started`/`monitor_stopped` markers.

### `tools/_validation_matrix_dispatch.py`
_Private helper of `validation_matrix.py`. Recovered from `run_xr1_full.py` and parameterised over the rSkill and prompt the original hardcoded per round. Sends one `ExecuteRskill` goal at the live graph (the matrix runs with the reasoner off, so nothing else would) and prints one JSON line — the run's `goal.log`. Needs ROS 2 sourced._

- `main(argv=None) -> int` — `--deadline-s` / `--rskill-id` / `--prompt` / `--server-timeout-s`. (L24)

### `tools/_ceiling_probe.py`
_One worker of the ceiling battery: one scene, one gate arm, N rounds. Deliberately bypasses `validation_matrix.py` (which refuses `--no-enable-octomap-kernel-check` — a validation round must never silently disable the collision check) while reusing its `materialise_scene`, readiness gate and dispatch tool, so the gate flag is the only difference between arms. Answers "of the runs the kernel stops, how many would have succeeded anyway?"._

- `DEADLINE_S: float = 420.0` (L57) — Deadline per scene run; the harness's own default, kept identical so a gate-off run is not given more time to succeed than a gate-on one.
- `_reap_domain(domain, sig) -> int` — Teardown sweep for the graph nodes `killpg` cannot reach. `ros2 launch` starts the octomap pair in its own session, so `run_one`'s `killpg` left one `octomap_server_node` + `octomap_voxel_bridge` behind every round; each keeps its Fast-DDS `/dev/shm` segments and `fastrtps_port<N>_el` lock file open, so a later round on the same domain fails `open_and_lock_file`, is handed 0 chunks, and buckets `harness-error` with nothing in any log naming the cause. Measured 2026-09-10: 46 orphans surviving up to 23.7 h — 30 % of a core, 826 MB, 253 stale shm segments — and a round launched **alone** at load 1.2 still died with `latest_chunk: 0`; with the sweep it returned 248 chunks and a scored result. `ROS_DOMAIN_ID` is the ownership key (this is the reaper that *can* tell concurrent workers apart, which the CLI's argv-signature sweep cannot) and `--ros-args` separates a node from the probe and worker shell, which share the domain and must survive. SIGTERM first — Fast-DDS unlinks its own segments on a graceful exit and leaks them on SIGKILL. `tests/unit/test_ceiling_probe_reap.py`.
- `run_one(spec, gate, seed, run_dir, rskill) -> dict[str, object]` (L200) — Launch one scene end to end and return its record. Since #256 the record also carries `load_start` / `load_end` (host `getloadavg`), `chunks` (`_chunks_from_goal_log`) and `nav2_bond_teardown` (`_bond_teardown`), so a run starved by host load is legible in the record instead of having to be reconstructed from deploy logs afterwards — the gap that made the 2026-09-07 ceiling absolutes unreadable.
- `_chunks_from_goal_log(goal_log) -> int | None` — Action chunks the policy delivered inside the deadline: the single number saying whether a run was given a fair trial. A healthy run delivers 1.2–1.8 chunks/s; a starved or torn-down one delivers ~0.04/s and cannot reach a grasp whatever the policy does.
- `_bond_teardown(deploy_log) -> str` — Delegates to `validation_matrix._nav2_bond_teardown` so the probe and the harness agree on what a voided run looks like.
- `main(argv: list[str] | None = None) -> int` (L344) — CLI entry: `--scene/--gate/--rounds/--seed/--rskill/--out/--start-round`; drives `run_one` per round, appends to `records.json`, prints a running success tally.

### `tools/ceiling_battery.sh`
_Drives the eight `_ceiling_probe.py` workers (4 scenes x 2 gate arms). `ROUNDS` rounds each; `WORKERS` (default **2**, was a hard 8) caps how many run at once, and the worklist is interleaved **by scene** so the two live lanes are one scene's off and on arm — the arms stay paired under identical host conditions, which is the property the whole design rests on. Worker 0 runs alone for 210 s because it is the one that spawns the shared XR-1 sidecar. Needs `OPENRAL_SKIP_ORPHAN_REAP=1`: `openral deploy sim` reaps orphan graphs by argv signature and cannot tell a concurrent sibling from a crash leftover. That flag was exported here from the day the battery went parallel but **read by nothing until 2026-09-10**, so worker B's startup sweep was killing worker A's live graph the whole time — the measured "worker 1 never got its action server, 626 s to timeout". Now that it is honoured, the domain-scoped `_reap_domain` in `_ceiling_probe.py` is what keeps the host clean between rounds._

### `tools/_nav2_mppi_loop_probe.py`
_Times the WHOLE MPPI control loop against its 50 ms budget on a live `openral deploy sim` graph — the measurement `CostCritic.consider_footprint` was deferred pending. Attaches to a running graph, drives a real `NavigateToPose`, and divides `controller_server`'s `/proc` CPU time by the cycles it published on `/cmd_vel_nav`. CPU, not wall clock, because the stack runs `use_sim_time` and the 20 Hz is sim-time. Validates which polygon the costmap adopted per run (`published_footprint` longest edge: 0.72 m bare / 1.23 m grown), which caught the shipped 2 Hz footprint publisher (since removed — Nav2 is base-only) silently alternating with it. Private: attaches to someone else's graph, evidence rather than a shipped entry point. Round recorded in [`docs/reference/robocasa-carry-survey.md`](../reference/robocasa-carry-survey.md)._

- `BARE` (L53) — the manifest chassis footprint polygon.
- `GROWN` (L54) — the chassis footprint grown over the payload at the live 0.860 m forward reach.
- `CLK = os.sysconf("SC_CLK_TCK")` (L55) — clock ticks per second, for converting `/proc/<pid>/stat` fields to seconds.
- `controller_pid() -> int` (L58) — `pgrep` the live `nav2_controller/controller_server` PID; `SystemExit` if not running.
- `cpu_seconds(pid) -> float` (L71) — utime+stime from `/proc/<pid>/stat`, the clock-source-independent cost.
- `class Probe(Node)` (L77) — publishes the arm's polygon at 20 Hz, counts `/cmd_vel_nav`, and reads back `/local_costmap/published_footprint` for the validity check.
- `main(argv=None) -> int` (L129) — `--grown`, `--seconds=N`; emits one JSON line carrying `cpu_ms_per_cycle`, `cycles`, `footprint_len_m` and `arm_valid`.

### `tools/_nav2_costmap_silhouette_probe.py`
_Sweep a live graph's Nav2 costmaps for cells marked inside the robot itself. Issue #108: neither costmap should mark LETHAL inside the robot's own silhouette or a carried payload's — since PR #186 made Nav2 base-only (ADR-0099) the scan filter is the only thing preventing that. Private (like `_nav2_mppi_loop_probe.py`): attaches to an already-launched graph, evidence tooling not a shipped entry point._

- `LETHAL_OBSTACLE = 254` (L68) — `nav2_costmap_2d`'s LETHAL_OBSTACLE; deliberately not 253 (INSCRIBED_INFLATED_OBSTACLE) since the claim is that no cell inside the robot was **marked**, and 253 is what an inflation layer writes near a legitimate obstacle elsewhere.
- `payload_mask(objects, base_from, points_xy) -> tuple[Any, int, tuple[float, float] | None]` (L103) — Projects every attached object onto the costmap plane; returns `(union mask, objects fully placed, z span in base frame)`. Counts per object (not primitive) — an object counts as placed only when every one of its primitives projected.
- `class SilhouetteProbe(Node)` (L158) — Samples both costmaps and counts marked cells inside the robot silhouette.
- `main() -> int` (L351) — argparse (`--robot-yaml`, `--scene`, `--seconds`, `--settle`, `--drive`); spins the probe, optionally drives a `NavigateToPose`, emits a verdict dict with non-vacuity guards (`lethal_cells_anywhere`, `base_travel_m`, `payload_samples`, etc).

### `tools/isaac_sidecar.py`
_Isaac Sim scene sidecar — runs Isaac Lab/Kit in its own py3.11 venv, auto-spawned by `openral_sim.backends.isaac_sim` (py3.12). Launches Omniverse Kit headless, builds the scene selected by `--layout`, and serves ZMQ REP + msgpack/ndarray framing (`ping`/`reset`/`step`/`render`/`close`). Constructs `SimulationApp` before any `omni.*`/`isaaclab` import — heavy imports live inside `main`, not module scope._

- `main(argv: list[str]) -> int` (L133) — Checks required dep versions (before the ~50 s Kit boot), launches `SimulationApp`, then imports and builds the scene named by `--layout` (`lift`/`bowl_plate`/`manifest`) and serves the ZMQ loop.

### `tools/_isaac_scene_base.py`
_Shared base for the Isaac Sim sidecar scenes. Runs under the Isaac Sim py3.11 venv only (imported by the scene modules, which `isaac_sidecar.py` imports after `SimulationApp` is live). Owns the bits both the `lift_cube` and `bowl_plate` scenes share — the obs/step lifecycle, RGBA→HWC frame grabbing, the warmup + physics-substep loop, and the eval-layer observation assembly — so a new layout is a few template-method overrides rather than a third copy of the skeleton._

- `franka_joint_positions(franka: Any) -> NDArray[np.float32]` (L51) — Franka joint angles in manifest order (8 = 7 arm + gripper).
- `franka_joint_velocities(franka: Any) -> NDArray[np.float32]` (L56) — Franka joint velocities in manifest order (8 = 7 arm + gripper).
- `class IsaacSceneBase` (L61) — Lifecycle + obs skeleton common to the Isaac Sim sidecar scenes; class attrs `warmup_steps` / `physics_substeps`. Template methods `build`/`_apply_action`/`_images`/`_state`/`_reward_terminated` are overridden per scene.
  - `IsaacSceneBase.build() -> None` (L154) — Template method (raises `NotImplementedError`): construct the stage (robot, props, cameras, controllers).
  - `IsaacSceneBase.reset(seed: int | None = None) -> dict[str, Any]` (L91) — Per-episode reset: randomize, reset physics, warm up, observe.
  - `IsaacSceneBase.render() -> NDArray[np.uint8] | None` (L149) — Last grabbed RGB frame, or `None`.
  - `IsaacSceneBase.sim_time_ns() -> int | None` (L127) — Elapsed sim time in ns for `/clock`; prefers Isaac's `SimulationContext.current_time`, else integrates step count × physics dt.
  - `IsaacSceneBase.step(action: NDArray[np.float32]) -> dict[str, Any]` (L101) — Apply one action, advance physics (renders only the final substep), return a StepResult dict (`observation`/`reward`/`terminated`/`truncated`/`info`/`sim_time_ns`).

### `tools/isaac_scene.py`
_Minimal Isaac Sim lift-cube scene for the sidecar. Isaac Sim **core**, not Isaac Lab — `isaacsim.core.api.World` + `Franka` + `Camera` are on PyPI and enough for a real PhysX+RTX scene. Franka on a ground plane, red cube in front, two RTX cameras (LIBERO two-camera contract). Lifecycle/obs skeleton: `_isaac_scene_base.IsaacSceneBase`._

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
_Isaac Sim table + bowl + plate Franka scene with the LIBERO obs/action contract. Built on Isaac Sim **core**: arm motion via `ArticulationController.apply_action`, end-effector control via the core `isaacsim.robot_motion.motion_generation` Lula solver (position-delta IK on `right_gripper`). Mirrors the LIBERO contract so act-libero/smolvla-libero drive it through `openral sim run` unchanged._

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
_Robot-agnostic, URDF-driven Isaac Sim scene for the sidecar. Unlike the PoC scenes (`isaac_scene.py`, `isaac_bowl_plate_scene.py`), which hardcode Isaac's `Franka` USD asset, this scene honours the forwarded `--robot`: it imports the manifest robot's URDF and wires joints/sensors/control from a plain-JSON "isaac robot spec" the openral-side backend marshals across the venv boundary (the sidecar cannot import `openral_core`). M1 (this cut): fixed-base arm, one RGB camera, JOINT_POSITION-delta control; M2 adds depth/lidar sensors, M3 the mobile base._

- `_GRIPPER_DEADBAND = 1e-3` (L43) — Below this magnitude a gripper action channel means HOLD, letting a pure `BODY_TWIST` step (zero arm/gripper slots) leave the gripper alone.
- `_SCAN_MAX_SELF_SKIPS = 8` (L50) — Self-occlusion handling for the lidar fan (mirrors `openral_sim.backends.robocasa._LASER_MAX_SELF_SKIPS`): caps how many self-layers one beam steps through before giving up and reading `range_max_m`.
- `_SCAN_SELF_SKIP_EPS_M = 1e-3` (L53) — Nudge (m) added past a self-hit before re-casting.
- `map_dof_to_manifest(values, *, dof_index, manifest_joints, finger_dof_idx, base_values=None, base_joints=None) -> NDArray[np.float32]` (L56) — Map an Isaac articulation DOF vector to the full manifest joint order; generic replacement for the Franka-specific `_franka_dof_to_manifest`. A base joint reads `base_values`, a gripper joint reads the mean of `finger_dof_idx`, everything else its matching URDF DOF or `0.0`.
- `resolve_beam_range(raycast_closest, *, origin_xy, angle_rad, z, range_min_m, range_max_m) -> float` (L96) — One lidar beam's range, with the robot's own body skipped by rigid-body-path identity (`/panda` prefix) rather than assumed clear (#194); re-casts past a self-hit up to `_SCAN_MAX_SELF_SKIPS` times. Pure — `raycast_closest` is an injected PhysX scene-query callable, so unit-testable without a Kit app.
- `class IsaacManifestScene(IsaacSceneBase)` (L162) — URDF-driven, robot-agnostic Isaac scene built from the marshaled robot spec.
  - `IsaacManifestScene.build() -> None` (L280) — Import the manifest robot's URDF, wire joints/sensors/control, place cameras per `_plan_cameras`.

### `tools/robocasa_carry_survey.py`
_Answers the fact [issue #108](https://github.com/OpenRAL/openral/issues/108) was blocked on — does any RoboCasa task make the base drive while holding something — by **building the env and measuring it**, not by reading the source. Measures the distance from the base's start pose to every non-distractor object the task manipulates. Found `DeliverStraw` (3.16-3.80 m) and `GetToastedBread` (2.17-3.48 m), both in `target50`; `scenes/deploy/robocasa_deliver_straw.yaml` pins the first. Replaces an `ast` classifier that got this wrong twice — `Kitchen.get_fixture`'s `ref=` is a nearest-of-type tie-break, not the 0.10 m bound its docstring claims. See [`docs/reference/robocasa-carry-survey.md`](../reference/robocasa-carry-survey.md)._

- `REQUIRED_BASE_TRANSLATION_M = 1.0` — Criterion 1 of the scene that closes #108 (`openral_nav2_bringup` README); below it the arm bridges the gap and `NavigateToPose` never runs. (L37)
- `DISTRACTOR_PREFIXES = ("distr", "dstr")` — RoboCasa's two clutter-object spellings (117 and 6 uses). Both are needed: `distr` alone leaves `ArrangeBreadBasket` / `PanTransfer` reporting `dstr_dining*` as their furthest object, and dropping the filter entirely makes `StoreLeftoversInBowl`'s ~5 m distractors read as a cross-kitchen carry. (L44)
- `class SurveyError(RuntimeError)` — RoboCasa unavailable, or its task registry unreadable. (L47)
- `@dataclass(frozen=True) class CarryMeasurement(task, seed, layout, style, furthest_object_m, furthest_object, nearest_object_m, nearest_object, lang)` — One `(task, seed)` reset, measured; `.requires_base_translation` applies the threshold. (L52)
  - `CarryMeasurement.requires_base_translation` (L70) — `@property`; `furthest_object_m > REQUIRED_BASE_TRANSLATION_M`.
- `robocasa_root(explicit=None) -> Path` — `--robocasa-root`, else an importable `robocasa`, else the HAL's provisioning cache. (L74)
- `read_target50(root) -> list[str]` — The 50 task names XR-1's model card pins as its reference configuration, parsed from `dataset_registry.py` with `ast`. (L95)
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
- `read_bag(bag_path: str | Path) -> Iterator[BagMessage]` (bag_reader.py L144) — Iterate an `.mcap` file or rosbag2 directory; extracts `trace_id` from `jsonschema`-encoded payloads (a packed W3C `traceparent` in the `trace_id` field, OR — for the `openral_msgs/Tick` schema — a raw 32-hex `trace_id` + raw 16-hex `span_id` pair, ISSUE-109) or `ros2msg`-encoded CDR payloads (regex match on the W3C `traceparent` substring). No `rosbag2_py` dep.
- `class TraceQueryError(RuntimeError)` (trace_query.py L26) — Raised on a non-JSON or unreachable dashboard response.
- `@dataclass(frozen=True) class DashboardTraceClient(base_url="http://127.0.0.1:8000", timeout_s=5.0)` (trace_query.py L31) — `list_traces() -> list[dict]` over `/api/traces`; `get_spans(trace_id) -> list[dict]` over `/api/spans/<id>`.
- `@dataclass(frozen=True) class TimelineEntry(kind, ts_ns, trace_id, topic, span_name, attrs, duration_ms)` (correlator.py L27) — One row of the joined timeline; `.to_json()` returns a plain dict.
- `list_bag_trace_ids(bag_messages) -> list[dict]` (correlator.py L68) — Distinct trace_ids in the bag with counts, busiest first.
- `build_timeline(bag_messages, spans, *, trace_id=None) -> list[TimelineEntry]` (correlator.py L95) — Pure join. Filters both inputs to `trace_id`, merges, sorts ascending by `ts_ns`.
- `RECORD_PROFILES: dict[str, dict[str, list[str]]]` (cli.py L45) — Slim and full topic + regex presets.
- `build_record_command(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=()) -> list[str]` (cli.py L85) — Compose `ros2 bag record` argv.
- `@dataclass(frozen=True) class ReplayResult(trace_id, bag_trace_ids, timeline, bag_path)` (cli.py L133) — `.to_json()` returns a plain dict.
- `run_replay(*, bag_path, trace_id, dashboard_url) -> ReplayResult` (cli.py L162) — Read a bag, fetch matching spans from the dashboard, return the joined timeline.
- `run_record(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=(), dry_run=False) -> tuple[list[str], CompletedProcess | None]` (cli.py L210) — Spawn `ros2 bag record` in a new process group; forwards SIGINT/SIGTERM received by the parent as **SIGINT** to the child group so rosbag2 flushes `metadata.yaml` cleanly. Waits up to 5 s after the child exits for that file to appear.
- `write_timeline(result: ReplayResult, out_path: Path) -> None` (cli.py L283) — Persist the timeline JSON.

### `tools/rskill_publisher.py`
_Package and publish a local rSkill directory to the HF Hub._

- `_REPO_ROOT: Path` (L58) — Repo root, derived from `__file__`.
- `_REQUIRED_FILES: list[str]` (L81) — `["rskill.yaml"]`, the minimum a directory must contain to be a candidate rSkill.
- `public_visibility_error(manifest, public) -> str | None` (L84) — §9 license gate (pure, no network): returns an error string if `--public` is requested for a non-commercial-licensed skill (`not manifest.is_commercial_use_allowed`), else `None`. Lets `main` fail fast before any HF call.
- `_resolve_token(token_arg) -> str` — Prefer CLI arg, fall back to env. (L129)
- `_validate_manifest(skill_dir) -> RSkillManifest` (L148)
- `_validate_docs(skill_dir, manifest) -> DocValidationReport` — Print + return the README / manifest documentation report via `_rskill_doc_validator.validate_rskill_docs`. Runs in both dry-run and `--publish` paths; the caller decides whether to exit on errors.
- `_rewrite_manifest_name(manifest_path, old_name, new_name) -> None` — Rewrite the top-level `name:` value of `rskill.yaml` in place (column-0 line only, so nested `name:` keys are untouched; preserves quotes + trailing comment). Exits 1 if the line isn't found exactly once. Backs `--fix-name`.
- `_enforce_repo_name(skill_dir, manifest, *, fix_name) -> RSkillManifest` — Enforce the ratified naming grammar via `openral_core.schemas.repo_name_is_canonical` (kind-aware: `rskill-playbook-<name>` for playbooks, `rskill-<model>-<robot>-<task>-<quant>` otherwise). No kind is exempt. On a non-canonical name: `fix_name=True` rewrites to `expected_repo_name(manifest)` + reloads; `fix_name=False` hard-fails (exit 1) printing the suggested name (exit 1 too if no canonical name can be suggested). Runs in both dry-run and `--publish` paths.
- `_bump_revision(manifest_path, weights_uri_base, token) -> str` — Resolve latest weights commit, patch `rskill.yaml`. (L336)
- `_ensure_private(api, repo_id) -> None` — Abort if the repo is public. (L387)
- `_ensure_public(api, repo_id) -> None` — The `--public` counterpart: abort if the (reused) repo is private, so a `--public` publish never lands in a private repo.
- `_publish(skill_dir, manifest, token, *, public=False) -> str` — Create the HF repo (private unless `public`) and upload; runs the matching visibility gate (`_ensure_public` / `_ensure_private`) after `create_repo`.
- `main() -> None` (L528) — Entry point. Sequence: parse args (`--publish` / `--public` / `--bump-revision` / `--fix-name` / `--token`) → validate manifest → `_enforce_repo_name` (exit 1 on a non-compliant VLA name unless `--fix-name`) → validate task space → validate docs → `public_visibility_error` gate (exit 1 if `--public` on a non-commercial skill) → exit 1 on doc errors → optional `--bump-revision` → `--publish` (private unless `--public`).

### `tools/generate_tight_geometry.py` (additions)

- `refine_dop_to_budget(points, dop_lo, dop_hi, budget) -> Points` (L223) — a ≤`budget`-vertex convex envelope strictly tighter than the 26-DOP, for a link whose exact hull is over `MAX_TIGHT_HULL_VERTICES`. Starts from the DOP and intersects it with the exact hull's own face planes, worst-violation first, skipping any plane that would overrun the budget. Every candidate plane is tangent to `conv(mesh)`, so containment stays definitional and the result is `⊆ DOP ⊆ box` by construction — which a subset-then-expand approach cannot guarantee (expansion escapes the DOP slabs; `panda_link1`'s DOP has 0.083 mm of room inside its box). Refuses rather than emit an envelope that cuts its mesh. On `panda_link1`: 0.18 mm median / 0.65 mm max support gap against the DOP's 4.52 / 25.68 mm. Measured, but **not shipped** — under a live battery that tightening moved link1's stops by 0.0003 mm, so no manifest declares a refined envelope; the routine is here for a link where the measurement comes out differently.
- `_OVERHANG_BATCH: int`, `_OVERHANG_MAX_SAMPLES: int` — bound `hull_overhang_m`'s peak memory and total sample count. The single-call form asked for 57.8 GiB on a 320-vertex envelope over a 12k-triangle mesh. Coarsening lowers a sampled lower bound, and `_check` fails only when a declared overhang is *below* a fresh resample, so it can only make that gate more permissive, never wrongly fail a correct manifest.

### `tools/voxel_transport_probe.py`

- `RADIUS_M: float` (L58) — the shipped coverage radius (1.05 m), so grid sizes are the deployed ones.
- `TARGET_HZ: float` (L59) — the publish rate the sweep drives each role at.
- `WARMUP: int` (L60) — messages discarded before timing starts, so cold-start latency does not skew the measured rate.
- `qos() -> QoSProfile` (L63) — the kernel's own `/openral/world_voxels` profile: `RELIABLE`, `KEEP_LAST(1)`, `VOLATILE`.
- `per_axis(res: float) -> int` (L71) — cells per axis at that resolution.
- `make_msg(res: float) -> tuple[Any, int]` (L75) — a full-size `OccupancyVoxels` at that resolution.
- `run_pub(res, count)` (L84) — publisher role, emitting JSON; run as a separate **process** so intra-process short-circuiting cannot hide the transport.
- `run_sub(res, count)` (L109) — subscriber role, emitting JSON; likewise a separate process.
- `run_sweep(resolutions, count) -> int` (L143) — drives both roles per resolution and prints the table, reporting the RMW measured.
- CLI: `uv run python tools/voxel_transport_probe.py sweep [--resolutions ...] [--count N]`. Needs a sourced ROS 2 overlay.

Measures the third cost term on the 25 → 15 mm lever — the dense `uint8[]` on the wire, 0.61 MB → 2.80 MB per publish at 10 Hz — as publish→receive latency, i.e. map staleness. Result is transport- and host-specific.

### `tools/stop_ee_speed.py`

- `REPO_ROOT: Path` (L50) — Repo root, derived from `__file__`.
- `EE_BODY: str` (L53) — `link7`, the body the payload attaches to, so its linear velocity is the carried object's.
- `QUANTISATION_GAIN_M: float` (L57) — what 25 → 15 mm recovers (the two cells' half-diagonal difference), the figure staleness cost is weighed against.
- `class StopSpeed(NamedTuple)` (L64) — `round_id`, `stop_class`, `ee_speed_mps`, `base_speed_mps`.
  - `StopSpeed.world_speed_mps` (L73) — adds the base contribution worst-case-aligned.
  - `StopSpeed.is_carry` (L78) — selects `attached_payload` stops.
- `collect(round_dirs) -> list[StopSpeed]` (L101) — end-effector speed at each stop, from the round's recorded `robot_joint_state` through `mj_jacBody` on the real Panda model. Matches arm joints on their trailing index (`panda_jointN` → `jointN`) and **raises** if none matched, because a silent mismatch reads as a perfectly stationary arm.
- `summarise(stops) -> dict[str, Any]` (L177) — per-class medians and maxima, and the net millimetres at the two staleness figures the wire probe measured.
- `render(stops, summary) -> str` (L196) — Render the summary as the human-readable table.
- `main(argv=None) -> int` (L227) — CLI entry. `uv run python tools/stop_ee_speed.py <round dirs...> [--json]`. Needs MuJoCo and the robosuite Panda assets; reads recorded artifacts only.

Settles the 25 → 15 mm trade: carry-phase stops are 0.051 m/s median / 0.265 m/s max, start-state stops exactly 0.

### `tools/resolution_ab.sh`

_The 25 mm vs 15 mm world-voxel A/B (#253). Both arms are gate-ON; the only difference is `OPENRAL_OCTOMAP_RESOLUTION_M`, which `deploy_e2e.launch.py` validates and `validation_matrix.py` records, so a round that changed it can never afterwards be mistaken for one that did not. The primary endpoint is **not** completion rate — against the predicted 2.7 % → 10.8 % that needs 200 runs/arm for 80 % power (`tools/round_power.py`), ~33 h on one host, and at the 40/arm a battery gives, power is 0.11. It is the paired, continuous quantity the mechanism actually predicts: per stop, the over-approximation (certified mesh gap minus reported depth), whose half-diagonal goes 21.65 mm → 12.99 mm, so the prediction is a **−8.66 mm shift**._

_**One graph at a time, arms alternating round by round.** Two ways to get the pairing wrong, both paid for. The first version fed a scene-interleaved worklist to `xargs -P`, which interleaves nothing — a worker holds its slot for all `ROUNDS` rounds, so only the first pair overlapped, and on 2026-09-10 `sink_cup` ran its arms 29 minutes apart while `baguette`, the one lane that stayed paired, came out 3/10 vs 3/10 (`p = 1.0`). Running the lanes genuinely concurrently then fixed the pairing and broke the host: q-laptop's baseline occupancy is ~9.1 GB of 15.4 (browser, editor sessions, the shared XR-1 sidecar) and one deploy graph adds ~4.9 GB — HAL/MuJoCo 3.2, `runtime_node` 0.9, Nav2 ~0.8 — so two exhausted all 4 GB of swap, drove load averages to 225, and returned 9 of 20 rounds unreadable. The runner therefore keeps one graph live and alternates `r01@25mm, r01@15mm, r02@25mm, …`, which is *tighter* pairing than concurrent lanes gave: adjacent rounds share an instant, not merely a window. ~4.5 h for 4 scenes x 10 rounds. `ROUNDS` and `SIDECAR_BOOT_S` are the knobs; `RUN_ROUND_CMD` is a test seam substituting the round body. `tests/unit/test_resolution_ab_pairing.py`._

### `tools/resolution_ab_report.py`

- `HALF_DIAGONAL_MM: dict[str, float]` (L38) — resolution string → cell half-diagonal in mm (`0.025` → 21.65, `0.015` → 12.99); the PRIMARY section's per-stop budget.
- `_stops(arm_dir) -> list[dict[str, Any]]` — every stop under one arm, with its certified truth when the run recorded one, plus the run's `started_at` / `wall_s` window.
- `main(argv=None) -> int` (L99) — CLI over the A/B's output root; writes `report.json` beside it.

_Pure and offline, so a battery run on one commit can be re-reported by another. Four sections. **PRIMARY** is the paired over-approximation per stop against `HALF_DIAGONAL_MM`. **INTEGRITY** prints each arm's *observed* `resolution_m`, read back from the HAL's voxel backing record, and flags `MISMATCH` when an arm did not run at the resolution it claims — an arm that silently fell back to the 0.025 default cannot pass unnoticed. **PAIRING** asks the question per round rather than per lane — for each scene it reports the median and worst gap between round *n* of one arm and round *n* of the other, against a budget of three typical rounds, and flags `NOT PAIRED` when the worst exceeds it. Lanes that ran back to back fail this however many rounds they carry. Data predating `started_at` is reported as unverifiable rather than passed. **SECONDARY** counts completions and stops and states in the output that it is under-powered, excluding runs with `chunks in (0, None)` — the #263 correction that moved the ceiling battery from 62.5/2.7 to 80.0/4.3. `tests/unit/test_resolution_ab_report_pairing.py`._

### `tools/stop_excess.py`

- `PAYLOAD_PREFIX: str` (L42) — `"attached:"`, the party-name prefix identifying a payload side in a stop record.
- `class Excess(NamedTuple)` (L45) — one stop's decomposition (`round_id`, `scene`, `party`, `is_payload`, `reported_depth_m`, `certified_gap_m`, `voxel_half_diagonal_m`, `verdict`).
  - `Excess.excess_m` (L58) — `certified_gap − reported_depth`.
  - `Excess.beyond_voxel_m` (L63) — subtracts the half-diagonal.
- `half_diagonal(resolution_m: float) -> float` (L68) — half a cubic cell's body diagonal, the grid's worst-case error.
- `collect(round_dirs: list[Path]) -> tuple[list[Excess], list[str]]` (L73) — certified stops plus the reasons for every skip. Reads each round's own `grid_resolution_m`; a stop without one is skipped, **never** defaulted to 25 mm, since the half-diagonal is the whole quantity being subtracted.
- `summarise(stops, skipped) -> dict[str, Any]` (L120) — per-class `n`, median excess, median beyond-voxel, and `has_geometry_headroom` (strict `> 0`: a class exactly at the voxel term has none).
- `render(stops, summary) -> str` (L142) — the per-stop table plus the per-class verdict.
- `main(argv=None) -> int` (L175) — CLI. `uv run python tools/stop_excess.py <round dirs...> [--json]`.

Produces `PLAN.md` §5's decomposition table — the measurement that struck the payload-hull and voxel-resolution levers and promoted modeled fixtures. Pure, offline, stdlib-only. Tested in `tests/unit/test_stop_excess.py`.

### `tools/adr0101_recovery.py`

- `PAYLOAD_PREFIX: Final[str] = "attached:"` (L50) — `party_a` prefix identifying an attached-payload stop.
- `CELL_PREFIX: Final[str] = "voxel_"` (L51) — `party_b` prefix identifying a world-voxel-cell stop.
- `CLEAR_THRESHOLD_M: float` (L48) — the clearance/contact boundary, `0.0`. Zero belongs to **contact**, not clearance: a payload touching a surface is stopped by a modeled body exactly as it was by the cube, so putting `0.0` on the clearance side would count real contacts as recoveries — the one class ADR-0101 must never suppress.
- `class Stop(NamedTuple)` (L54) — one payload-vs-cell stop with the certified truth behind it (`round_id`, `scene`, `payload`, `cell`, `reported_depth_m`, `certified_gap_m`, `nearest_body`).
  - `Stop.recovered` (L66) — `certified_gap_m > CLEAR_THRESHOLD_M`.
- `class Excluded(NamedTuple)` (L71) — a payload-vs-cell stop that could not be adjudicated, and why. Reported, never silently dropped: shrinking the denominator inflates the rate, which is the direction that would overstate the case for a fail-open mechanism.
- `UNATTRIBUTED: str` (L84) — `"<unattributed>"`, the by-fixture label when the body cannot be established.
- `fixture_at_stop(scene_dir: Path, certified_gap_m: float) -> str` (L87) — the world body the payload was nearest, read from the raw `sim.estop_ground_truth_snapshot` line's `nearest_payload_world_pairs`. **Not** `ground_truth.nearest_pair`, which is the closest probed pair of *any* kind and for a carried payload is routinely two robot links. The match is verified: the list's minimum certified distance must equal the `nearest_tripping_party_m` the stop was adjudicated on, else no attribution is made. The recovery count never depends on this — only the breakdown does.
- `collect(round_dirs: list[Path]) -> tuple[list[Stop], list[Excluded]]` (L138) — read each round's `verdicts.json` into adjudicable stops plus exclusions. Selects only world stops whose `party_a` is `attached:<id>` and `party_b` is `voxel_<n>`, and only when the probe certified its distances.
- `summarise(stops, excluded) -> dict[str, Any]` (L194) — ADR-0101's Consequences table as data: counts, recovery rate, median **and minimum** recovered clearance, the still-stopping depths, and the by-fixture breakdown. Returns `recovery_rate=None` over an empty set rather than a number.
- `render(summary) -> str` (L216) — the human-readable report; refuses to print a percentage over an empty denominator.
- `main(argv=None) -> int` (L254) — CLI. `uv run python tools/adr0101_recovery.py <round dirs...> [--json]`.

Produces the "48 of 51 (94 %)" figure ADR-0101 rests on, which until now had no producer in the repo. Pure, offline, stdlib-only; reads recorded artifacts, needs no GPU or simulator. Tested in `tests/unit/test_adr0101_recovery.py` against the real `2026-08-23-master-s1` round.

### `tools/round_power.py`
_Answers "how many validation-matrix runs does this comparison need?" **before** a battery is run — [#217](https://github.com/OpenRAL/openral/issues/217) step 1. Exists because the programme has twice drawn a conclusion a battery could not support: the n=1 reading on #176, and the 20-run success comparison #217 was opened to settle, which had under 30 % power against the very effect it observed. Exact (every outcome pair enumerated, not sampled) so the answer is reproducible on any host, and stdlib-only so scipy stays out of the workspace for a planning script; validated against `scipy.stats.fisher_exact` on 300 random 2×2 tables. Feeds the 2026-09-05 entry in [`docs/reference/collision-validation-evidence.md`](../reference/collision-validation-evidence.md)._

- `RUNS_PER_ROUND = 4` — Scene runs in one matrix round, so `required()` can report rounds as well as runs. (L38)
- `fisher_exact_two_sided(a, n1, b, n2) -> float` — Two-sided Fisher p-value for `a`/`n1` against `b`/`n2`; the "no more probable" tail convention `scipy.stats.fisher_exact` uses. (L45)
- `power(baseline, alternative, n, *, alpha=0.05) -> float` — Exact power at `n` per arm: enumerates every `(a, b)` under the two binomials and sums the probability of the pairs Fisher rejects. `O(n²)` Fisher evaluations, so sub-second to n=100 and a few seconds by n=300. (L83)
- `required(baseline, alternative, *, alpha=0.05, target=0.80) -> tuple[int | None, float]` — Smallest ladder `n` per arm reaching `target` power, or `(None, best)` when the ladder tops out at 800 — a comparison needing more than that is not one this programme can afford, and saying so is the useful answer. (L118)
- `_LADDER: tuple[int, ...]` (L42) — The candidate per-arm sample sizes `required` searches, `20` to `800`.
- `main(argv=None) -> int` (L145) — CLI entry; `--baseline`/`--alternative` rates, `--alpha`, `--target`; prints the required `n` (and rounds) or that the ladder tops out.

### `tools/rskill_scaffolder.py`
_Standalone argparse wrapper around `openral_cli._rskill_scaffolder.scaffold_rskill`._
Mirrors `openral rskill new`; exists so power users can scaffold without installing the CLI distribution.

- `_REPO_ROOT: Path` (L27) — Repo root, derived from `__file__`; used to add `python/<pkg>/src` onto `sys.path` before importing the CLI package.
- `_parse_args(argv) -> argparse.Namespace` — argparse setup. (L35)
- `main(argv=None) -> int` — Entry point; returns a process exit code. (L77)

### `tools/generate_rskill_skillmd.py`
_Generate the standard agent-skill `SKILL.md` discovery view for every in-tree rSkill from its `rskill.yaml`._
The single canonical producer of the `SKILL.md` mirror (CLAUDE.md §1.3): `rskill.yaml` is authoritative; the generated `SKILL.md` is discovery-only and never hand-edited. `--check` fails on any stale/missing `SKILL.md`, so the same process applies to every kind — including `playbook`, whose `_KIND_NOUN` entry renders identically to `vla`/`detector`/`vlm`/`reward`.

- `render_skill_md(manifest_path: Path) -> str` (L162) — Render the `SKILL.md` text (YAML frontmatter + capability/verb summary + license/provenance) from one manifest; `_KIND_NOUN` maps each `kind` to its discovery noun.
- `main(argv=None) -> int` (L261) — Entry point. No args = regenerate every `rskills/<id>/SKILL.md`; positional ids regenerate a subset; `--check` reports stale/missing without writing (exit 1 on drift).

### `tools/rldx_sidecar.py`
_Boot helper for the RLDX-1 inference sidecar (companion to `openral_sim.policies.rldx`)._
Materialises a Python 3.10 venv under `_DEFAULT_HOME` (`~/.cache/openral/rldx-sidecar`, override via `--home`), clones the upstream `RLWRLD/RLDX-1` repo, runs `uv sync` (rldx + transformers + flash-attn + …) — on aarch64, an override-driven `uv pip install` instead, see `_install_deps_aarch64` — optionally adds `bitsandbytes` for NF4, then writes a wrapper that monkey-patches `transformers.AutoModel.from_pretrained` to apply NF4 / int8 to the Qwen3-VL-8B backbone (the MSAT diffusion head is left at bf16) and `os.execvpe`s into `rldx.eval.run_rldx_server`. Required because the `rldx` package pins `requires-python = "~=3.10"` and ships a custom `architectures=["RLDX"]` class not in HF Transformers.

- `_LABEL: str` (L34) — `"rldx-sidecar"`, the sentinel/log label for this sidecar's provisioning.
- `_REPO_URL: str` (L35) — Upstream `RLWRLD/RLDX-1` git URL cloned into the venv's source tree.
- `_DEFAULT_HOME: Path` (L36) — `~/.cache/openral/rldx-sidecar`, overridable via `--home`.
- `_AARCH64_OVERRIDE: Path` (L46) — `sidecar_requirements/rldx-aarch64-override.txt`, the torch/torchvision pin override `_install_deps_aarch64` installs under.
- `_NVRTC_OVERRIDE: Path` (L54) — Shared `aarch64-nvrtc-override.txt`, raising nvrtc past the sm_121 cliff.

- `_install_deps(*, source, uv, quantization) -> Path` — `uv sync` in the cloned source tree (creates `source/.venv` with rldx + deps), then `uv pip install --python <venv>/bin/python bitsandbytes>=0.43.0` when `quantization in {nf4, int8}`. Returns the venv path. Delegates to `_install_deps_aarch64` on aarch64, where that `uv sync` cannot succeed. (L121)
- `_install_deps_aarch64(*, source, quantization) -> Path` — aarch64 replacement for the `uv sync` pass: same `<source>/.venv` and same Python 3.10, provisioned by `ensure_pip_venv` + `uv pip install -e <source> --torch-backend=cu128` under `sidecar_requirements/rldx-aarch64-override.txt` (torch 2.9.1 / torchvision 0.24.1; `torchcodec` and `flash-attn` marker-dropped) and the shared `aarch64-nvrtc-override.txt`. Second pass adds `nvidia-cuda-nvcc-cu12==12.9.86` (sm_121 `ptxas`) plus `bitsandbytes` for NF4/int8. Sentinel is keyed on both override texts so a corrected pin repairs an existing venv. (L59)
- `_aarch64_extras(quantization) -> list[str]` — the packages installed on top of upstream's own dependency set on aarch64; shared between the install pass and the sentinel spec so they cannot drift. (L105)
- `_make_wrapper(*, work, source, args) -> Path` — Generate `<work>/boot_server.py`: monkey-patches `AutoModel.from_pretrained` for the Qwen3-VL backbone with NF4 / int8 / no-op, sets `RLDX_ATTN_IMPL=sdpa` when `flash_attn` is not importable (upstream's own documented opt-out; a no-op on x86_64, where `uv sync` installs it), sets `sys.argv`, and calls into `rldx.eval.run_rldx_server`. (L157)
- `main() -> int` — argparse entry point; flags `--model`, `--port`, `--quantization {none,nf4,int8}`, `--home`. Calls `run_sidecar(..., family="rldx", ...)`, which stamps the sidecar identity record (so the adapter can verify reuse) and then `os.execvpe`s into the sidecar venv so SIGINT reaches the server. (L314)

### `tools/xr1_sidecar.py` + `tools/_xr1_server.py`
_Boot helper + inference server for Xiaomi Robotics XR-1. The launcher provisions torch 2.9.1 / transformers 4.57.1 / FlashAttention 2.8.3 + bitsandbytes and execs the server (torch is one minor ahead of upstream's 2.8.0, which has no linux-aarch64 `cu128` wheel — `docs/reference/aarch64-support.md`). The server loads the pinned MiBoT custom-code checkpoint in manifest-selected NF4 (or BF16), recreates Xiaomi's benchmark-specific chat template and padded state tensor, decodes actions with the checkpoint processor, and serves `ping/reset/get_action/close` over the shared ZMQ/msgpack ndarray wire._
- `main() -> int` (xr1_sidecar.py L94) — parse model/profile/quantization/host/port/home; `--export-dir` persists an NF4 checkpoint and exits, otherwise stamp sidecar identity and `os.execvpe` into `_xr1_server.py`.
- `_NDARRAY_SENTINEL = "__ndarray__"` (_xr1_server.py L17) — Marker key for the msgpack ndarray codec (mirrors `openral_sim.sidecar.encode_ndarray`'s `_NDARRAY_SENTINEL`, re-implemented here because the sidecar venv cannot import `openral_sim`).
- `_STATE_DIM = 60` (_xr1_server.py L18) — XR-1's internal padded state width; `_pad_state` pads any frame/history up to this.
- `_ACTION_DIMS = {"robocasa_mg": 7, "robocasa365": 12, "vlabench_choice": 7}` (_xr1_server.py L19) — Per-benchmark-profile action width the checkpoint processor decodes.
- `main() -> int` (_xr1_server.py L341) — ZMQ REP server loop entry point; loads the pinned checkpoint then answers `ping/reset/get_action/close`.
- `_messages(profile, images, instruction) -> list[dict[str, Any]]` — exact RoboCasa, RoboCasa365-video, or VLABench message layout from upstream revision `7c20088`.
- `_pad_state(state) -> NDArray[np.float32]` — pad one frame or a four-frame history to XR-1's 60-D internal state.
- `class _XR1Policy` — pinned `AutoModel`/`AutoProcessor` custom-code loader; bitsandbytes NF4 uses a BF16 compute dtype + CUDA device map, `prequantized_nf4` reloads a saved packed checkpoint without re-quantizing, and checkpoint-owned action de-normalization remains unchanged. `export_pretrained(output_dir)` writes sharded packed safetensors, custom MiBoT code/processor assets, and `quantization_metadata.json` while explicitly excluding cached BF16 shards.

### `tools/qwen_vlm_sidecar.py` + `tools/_qwen_vlm_server.py`
_Boot helper + server for the Qwen3.5-4B scene-VLM sidecar, companion to `openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`._ The launcher provisions an isolated venv (`OPENRAL_QWEN_VLM_SIDECAR_VENV` to reuse one) with transformers + bitsandbytes + `qwen-vl-utils` + pyzmq/msgpack, then `os.execvpe`s into the server. The server answers a ZMQ REQ/REP + msgpack protocol (`{"op":"query","image","question"}` → `{"ok","answer"}`); out-of-process for dependency/VRAM isolation (same pattern as `rldx_sidecar`). Apache-2.0 model.

- `_DEFAULT_HOME: Path` (L34) — `~/.cache/openral/qwen-vlm-sidecar`, overridable via `--home` / `$OPENRAL_QWEN_VLM_SIDECAR_HOME`.
- `_VENV_ENV: str` (L35) — Env var name for the `--venv` override.
- `_HOME_ENV: str` (L36) — Env var name for the `--home` override.
- `_LOCK: Path` (L46) — Hash-locked pinned deps file the venv is provisioned from.
- `_NVRTC_OVERRIDE: Path` (L51) — aarch64 nvrtc override passed at install time alongside `_LOCK`.
- `ensure_venv(home, *, override=None) -> Path` (L56) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded); honours `$OPENRAL_QWEN_VLM_SIDECAR_VENV`.
- `main() -> int` (L97) — argparse (`--model`, `--host`, `--port`, `--max-side`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME` and `os.execvpe`s into `_qwen_vlm_server.py`.
- `_load(model_id) -> (processor, model)` (_qwen_vlm_server.py L59) — dual-path NF4 load: auto-detect a pre-quantized checkpoint via the embedded `quantization_config` → load 4-bit directly; else quantize-at-load, serial materialization for 8 GB.
- `_query(processor, model, *, image, question, max_side, max_new_tokens) -> str` (_qwen_vlm_server.py L101) — one scene-question→answer generate via the canonical Qwen-VL recipe (strips the `<think>` trace).
- `main() -> int` (_qwen_vlm_server.py L170) — ZMQ REP loop (`ping`/`query`/`shutdown`); `{"ok": False, "error": ...}` on exception rather than dying. Validated live (CLAUDE.md §1.2).

### `tools/locateanything_sidecar.py` + `tools/_locateanything_server.py`
_Boot helper + server for the `nvidia/LocateAnything-3B` open-vocabulary detector sidecar, companion to `openral_runner.backends.gstreamer.locateanything_detector.LocateAnythingDetector`._ `LocateAnything-3B` ships `trust_remote_code` modeling files pinned to `transformers==4.57.1`, incompatible with the workspace's `transformers>=5`, so it runs out-of-process over the same ZMQ REQ/REP + msgpack pattern as `rldx_sidecar`. No upstream repo to clone — the model is custom-code on the Hub — so the sidecar is just the pinned venv plus the thin server.

- `_DEFAULT_HOME: Path` (L33) — `~/.cache/openral/locateanything-sidecar`.
- `_VENV_ENV: str` (L34) — Env var name for the `--venv` override.
- `_HOME_ENV: str` (L35) — Env var name for the `--home` override.
- `_LOCK: Path` (L44) — Hash-locked pinned deps (`transformers==4.57.1` load-bearing); regenerated via `uv pip compile tools/sidecar_requirements/locateanything.in`.
- `_NVRTC_OVERRIDE: Path` (L49) — aarch64 nvrtc override passed at install time alongside `_LOCK`.
- `ensure_venv(home, *, override=None) -> Path` (L54) — return the sidecar venv python, provisioning + installing pinned deps if absent (sentinel-guarded, keyed on `_LOCK` + `_NVRTC_OVERRIDE`); honours `$OPENRAL_LOCATEANYTHING_SIDECAR_VENV`.
- `main() -> int` (L96) — argparse (`--model` required, `--host`, `--port`, `--max-side`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME`, sets `OPENRAL_ALLOW_REMOTE_CODE=1` (the operator opted in by launching this sidecar), `os.execvpe`s into `_locateanything_server.py`.
- `_split_model_ref(model_ref) -> tuple[str, str]` (_locateanything_server.py L44) — Split a `repo@revision` reference into `(repo, revision)`.
- `_load(model_ref) -> (processor, model)` (_locateanything_server.py L55) — NF4 bitsandbytes load of the pinned custom-code checkpoint.
- `_detect(processor, model, image, query, *, max_side, mode, max_new_tokens) -> str` (_locateanything_server.py L99) — One detection request → the model's raw generated text; `<ref>`/`<box>` parsing stays in the main-env backend.
- `main() -> int` (_locateanything_server.py L146) — ZMQ REP loop (`ping`/`detect`/`shutdown`); `{"ok": False, "error": ...}` on exception rather than dying.

### `tools/da3_depth_sidecar.py` + `tools/_da3_depth_server.py`
_Boot helper + server for the Depth Anything 3 monocular metric-depth sidecar, companion to `openral_perception_ros`'s depth-provider node (and `Da3DepthClient`, `docs/methods/07-eval-sim.md`)._ `depth-anything/DA3-SMALL` ships as the `depth-anything-3` package rather than transformers-native, so it runs in its own Python 3.12 venv behind a ZMQ REQ/REP + msgpack wire (`ping` / `depth` / `shutdown`); the provider republishes the reply as a `32FC1` depth Image + CameraInfo for nvblox. Measured 0.27 GB / ~27 Hz on an 8 GB Ada. The launcher loads `_sidecar_common` **by file path** (not `from openral_sim…`) so it stays runnable by any `python3` — the deploy launch autostarts it under a plain interpreter (`slam_depth_sidecar_autostart`).

- `_DEFAULT_HOME: Path` (L51) — default cache dir.
- `_VENV_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_VENV"` (L52) — `--venv` override env var name.
- `_HOME_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_HOME"` (L53) — `--home` override env var name.
- `_REQUIREMENTS = ("depth-anything-3", "pyzmq", "msgpack")` (L58) — x86_64 install set.
- `_NVRTC_OVERRIDE: Path` (L72) — aarch64 nvrtc override file passed at install time.
- `_AARCH64_REQUIREMENTS: tuple[str, ...]` (L75) — upstream's dependency list minus `open3d` / `pycolmap` / `xformers` / `pre-commit`, plus the undeclared `addict` and the repo's `torch==2.9.1` aarch64 pins.
- `_AARCH64_DA3 = "depth-anything-3==0.1.1"` (L102) — installed `--no-deps` after `_AARCH64_REQUIREMENTS` on aarch64.
- `_PYCOLMAP_EAGER_IMPORT = "\nimport pycolmap\n"` (L108) — the anchor statement `_defer_pycolmap_import` rewrites.
- `_PYCOLMAP_LAZY_IMPORT: str` (L109) — the lazy `__getattr__` proxy it rewrites that statement into.
- `ensure_venv(home, *, override=None) -> Path` (L163) — return the sidecar venv python, provisioning it if absent (sentinel-guarded, keyed on the platform's pin set — including the override text, so raising nvrtc repairs existing venvs); honours `$OPENRAL_DA3_DEPTH_SIDECAR_VENV`. **x86_64** installs `_REQUIREMENTS` (`depth-anything-3`, `pyzmq`, `msgpack`) in one pass. **aarch64** (GB10 / DGX Spark, Jetson Thor) cannot: `open3d` and `pycolmap` publish no linux-aarch64 wheel, so it installs `_AARCH64_REQUIREMENTS` (upstream's dependency list minus `open3d` / `pycolmap` / `xformers` / `pre-commit`, plus the undeclared `addict` and the repo's `torch==2.9.1` aarch64 pins) and then `_AARCH64_DA3` (`depth-anything-3==0.1.1`) with `--no-deps` — both passes under `--overrides _NVRTC_OVERRIDE` (`sidecar_requirements/aarch64-nvrtc-override.txt`), which is load-bearing here: DA3's `@torch.jit.script affine_inverse` is NVRTC-fused once the profiling executor warms up, so on stock nvrtc 12.8 request #1 succeeds and every later one dies on sm_121.
- `_site_packages(py) -> Path` (L128) — the `site-packages` of the venv owning `py`; `SystemExit` if absent.
- `_defer_pycolmap_import(py) -> None` (L136) — aarch64 only. Rewrites the module-level `import pycolmap` in the installed `depth_anything_3/utils/export/colmap.py` into a lazy `__getattr__` proxy, because `api.py` imports the export subsystem eagerly and `pycolmap` has no aarch64 wheel. A **deferred import, not a stub** (§1.11): `export_to_colmap` still raises the real `ModuleNotFoundError`. Idempotent; unlinks before writing (uv hardlinks wheel contents out of its shared cache); raises `SystemExit` if the anchor statement is not found exactly once.
- `main() -> int` (L200) — argparse (`--model` default `depth-anything/DA3-SMALL`, `--host`, `--port` default 5771, `--process-res` default 504, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME` and `os.execvpe`s into `_da3_depth_server.py`. No JIT/fuser env knobs — the sm_121 problem is fixed at install time by the nvrtc override, so TorchScript stays on.
- `_load(model_id) -> (torch, model)` / `_infer(torch, model, image, process_res) -> (depth, K, latency_ms)` — load DA3 to CUDA in eval mode; one image → metric depth `(H, W)` float32 + 3×3 intrinsics.
- `main() -> int` (_da3_depth_server.py L60) — ZMQ REP loop (`ping`/`depth`/`shutdown`) that replies with `{"ok": False, "error": …}` rather than dying on a bad frame. Validated live on both an 8 GB Ada and a GB10 (CLAUDE.md §1.2, `docs/reference/aarch64-support.md`).

### `tools/internvla_n1_sidecar.py` + `tools/_internvla_n1_server.py`
_Boot helper + server for InternVLA-N1/DualVLN, companion to `openral_sim.policies.internvla_n1`. InternNav pins `transformers==4.51.0` (+ matching diffusers/accelerate), incompatible with the workspace's transformers 5.x, so it runs out-of-process (like `rldx`) over ZMQ + msgpack. The launcher provisions a py3.11 venv from upstream's explicit inference-only pin set (skipping the full habitat/isaac eval stack `setup.py` would otherwise drag in) and installs `internnav` + `diffusion_policy` with `--no-deps` (System-1's NavDP needs only `SinusoidalPosEmb` from the latter); the server runs the Qwen2.5-VL-7B S2 planner + NavDP DiT S1 policy and maps discrete VLN-CE actions to a `BODY_TWIST`._

- `_LABEL = "internvla-n1-sidecar"` (L39) — sidecar identity.
- `_REPO_URL = "https://github.com/InternRobotics/InternNav.git"` (L40) — upstream clone URL.
- `_DEFAULT_HOME: Path` (L41) — cache dir.
- `_PYTHON = "3.11"` (L42) — venv Python version.
- `_PINNED_DEPS: list[str]` (L52) — Inference-only pin set mirroring upstream `requirements/internvla_n1.txt` minus flash-attn (sdpa instead); torch 2.9.1 on the cu128 index (PyPI's aarch64 wheel is CPU-only otherwise); `diffusers==0.32.2` specifically (not upstream's 0.33.1, which dropped the SwiGLU inner-dim reduction the DualVLN checkpoint's NextDiT was trained against).
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
_Boot helper + server for LingBot-VLA (v1 and v2), companion to `openral_sim.policies.lingbot_vla2`. v2 (6B Qwen3-VL MoE, `transformers==4.57.3`) is the default; `--variant v1` selects the 4B Qwen2.5-VL dense RoboTwin post-train on `transformers==4.51.3`/`lerobot==0.4.2`, which caps `torch<2.8.0` and is x86_64-only. Runs out-of-process because `lingbotvla` pins torch + custom kernels incompatible with the workspace stack._

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
- `main(argv: list[str]) -> int` (_lingbot_vla2_server.py L944) — Resolves the torch-version-dependent `PYTORCH_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF` env var from installed-package metadata (avoids initializing CUDA before it's set), then serves either `_LingBotPolicy` (v2) or `_LingBotV1Policy` (`--variant v1`).

### `tools/cosmos3_reasoner_sidecar.py`
_Boot helper for the curated NVIDIA Cosmos 3 Edge reasoner model (`OPENRAL_REASONER_MODEL=cosmos3-edge`), companion to `openral_reasoner.cosmos3.Cosmos3ToolUseClient`._ Provisions an isolated Python 3.12 venv (`OPENRAL_COSMOS3_SIDECAR_VENV` to reuse one) from the pure-PyPI hash-pinned `tools/sidecar_requirements/cosmos3_reasoner.lock` (vllm≥0.23) **plus a SHA-pinned `transformers` overlay** (`_TRANSFORMERS_EDGE_SHA` — no *released* transformers recognises `cosmos3_edge` yet), then — for the Edge diffusers layout — downloads the checkpoint and builds a **flattened reasoner view** (vLLM's loader can't follow the diffusers subfolder `weight_map`) before `os.execvpe`ing into `vllm serve <view> --served-model-name nvidia/Cosmos3-Edge --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager`. No ZMQ server script: vLLM's OpenAI-compatible HTTP API *is* the wire contract the reasoner already speaks. Out-of-process for the same dependency/VRAM isolation reasons as `qwen_vlm_sidecar.py`. Cosmos 3 weights are OpenMDW-1.1 (commercial OK) — no license guard. **Boots live on an 8 GB RTX 4070; forward-pass inference is blocked by an upstream `get_rope_index` bug** (see `docs/reference/cosmos3-edge-reasoner.md`).

- `ensure_venv(home, *, override=None) -> Path` (L94) — return the sidecar venv python, provisioning + installing the pinned lock **and the SHA-pinned transformers overlay** if absent (sentinel-guarded); honours `$OPENRAL_COSMOS3_SIDECAR_VENV`.
- `is_diffusers_reasoner_layout(model_dir) -> bool` (L149) — True when the top-level `model.safetensors.index.json` maps tensors into subfolders (the Edge layout needing a view); Nano/Super (bare top-level shards) return False.
- `materialize_reasoner_view(model_dir, dest) -> Path` (L167) — build a vLLM-loadable flat view: bare-named shard symlinks + a rewritten weight index + tokenizer/config symlinks. Idempotent.
- `resolve_served_model(model, home, *, native_edge=False) -> tuple[str, str | None]` (L250) — resolve a repo id / local dir to a `vllm serve` target + optional `--served-model-name`. For the Edge diffusers layout the target depends on the serving vLLM: `native_edge=True` serves the **snapshot dir as-is** (the native `Cosmos3EdgeForConditionalGeneration` follows the subfolder `weight_map` itself, and the flattened view *breaks* it — `RuntimeError: Cannot find any model weights`, seen on a Jetson with vLLM 0.28.0), `False` builds the flattened view the Transformers-fallback loader needs. Nano/Super (standard layout) serve by id.
- `vllm_has_native_edge_model(py) -> bool` (L215) — asks the serving venv's own `vllm…registry.ModelRegistry` whether it knows `Cosmos3EdgeForConditionalGeneration`, rather than comparing version strings — the x86 and aarch64 branches of one lock resolve different vLLM releases (0.24.0 / 0.28.0) and only the latter has it. Fails closed: any import error/timeout returns `False`, selecting the pre-existing flattened-view path.
- `build_serve_argv(*, vllm_bin, model, host, port, tool_call_parser, max_model_len, gpu_memory_utilization, enforce_eager, served_model_name=None, kv_cache_dtype="auto") -> list[str]` (L285) — the `vllm serve` argv. `--max-model-len` (default 8192) holds the reasoner prompt + tools; `--enforce-eager` + `--gpu-memory-utilization` (default 0.90, `$OPENRAL_COSMOS3_GPU_MEM_UTIL`) + `--kv-cache-dtype fp8` are the 8 GB-fit knobs (fp8 KV verified required for the 8192 window with the native Edge impl on an 8 GB 4070).
- `main() -> int` (L362) — argparse (`--model` default `nvidia/Cosmos3-Edge`, `--host`, `--port` default 8901, `--tool-call-parser`, `--max-model-len`, `--gpu-memory-utilization`, `--no-enforce-eager`, `--kv-cache-dtype`, `--home`, `--venv`); strips `PYTHONPATH`/`PYTHONHOME`, sets the expandable-segments allocator config (fragmentation OOM fix verified live; `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` depending on the sidecar venv's torch — see `alloc_conf_var`/`venv_torch_version`), resolves the served view, and `os.execvpe`s into `vllm serve`.

### `tools/behavior_groot_sidecar.py`
_Python 3.10 sidecar for the official 2026 BEHAVIOR-1K GR00T N1.7 checkpoint. Imports the pinned `wensi-ai/Isaac-GR00T` behavior runtime, registers the official R1Pro modality slices, wraps `Gr00tPolicy` with `B1KPolicyWrapper`, and serves ZMQ `ping/reset/get_action/close`. Two 8 GB-host memory measures: whole-model NF4 (`--quantization nf4`, default) and dropping the unused Qwen3-VL `lm_head` (the wrapper reads only hidden states; ~840 MiB saved — 2.77 GiB inference peak, live-validated alongside OmniGibson)._

- `main(argv) -> int` (L366) — Parse checkpoint/task/instruction/control-mode/device/endpoint plus `--quantization {none,nf4,int8}` / `--nf4-min-params` (the min-params threshold applies to both quantizers), set the expandable-segments allocator config (`PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` depending on the installed torch), load the real checkpoint (CPU-first under NF4/int8), and serve the 23-D action API. `OPENRAL_BEHAVIOR_GROOT_DUMP_OBS=<dir>` pickles the first 32 observations for offline quantization A/Bs.
- `_drop_backbone_lm_head(model) -> None` — Replace the Qwen3-VL `lm_head` with `Identity`; the B1K wrapper consumes only `hidden_states[-1]`, so the full-vocab logits projection is dead weight and was the largest single inference allocation.
- `_quantize_nf4(model, *, device, min_params) -> None` — Whole-model `bitsandbytes` `Linear4bit` rewrite of large `nn.Linear`s (manifest pins `nf4_min_params: 1000000`), keeping small layers floating; patches the model `dtype` property to the first floating parameter dtype, then moves to CUDA. Offline A/B/C on captured obs: 2531 MiB peak, action MAE 0.0029 vs bf16 — behaviourally lossless for this checkpoint.
- `_quantize_int8(model, *, device, min_params) -> None` — Same rewrite with `bitsandbytes` `Linear8bitLt` (LLM.int8, threshold 6.0). 3678 MiB peak / MAE 0.0017 vs bf16 on the same captured obs — too big next to OmniGibson on 8 GB, so its role is offline quality comparison, not co-resident serving.

### `tools/behavior_scene_sidecar.py`
_Official OmniGibson evaluator environment sidecar for `scene.id=behavior`. Resolves the configured task instance, exposes the challenge wrapper observation, applies external 23-D actions, and returns success/metrics/sim time over ZMQ._

- `main(argv) -> int` (L222) — Parse task/instance/mode/wrapper/max-steps/endpoint, load the real BEHAVIOR environment, and serve `ping/reset/step/close`.

### `tools/_robometer_quant.py`
_Shared NF4 quantize + pre-quantized meta-load helpers for the Robometer reward scorer. Does not import `openral_sim._quantization` — re-implements the same NF4 rule (`nn.Linear` with `numel >= MIN_PARAMS` → bnb `Linear4bit` nf4/bf16) plus a pre-quantized direct-load path. Used by both `tools/build_robometer_nf4_checkpoint.py` (produces the checkpoint) and `tools/_robometer_scorer.py` (loads it directly as 4-bit via meta device). Determinism is load-bearing (CLAUDE.md §8): the math SDP kernel + deterministic algorithms are forced so a meta pre-quantized load equals the bf16+quantize reference bit-for-bit._

- `MIN_PARAMS = 4_000_000` (L25) — Mirrors `openral_sim._quantization.DEFAULT_MIN_PARAMS_TO_QUANTIZE`.
- `BNB_META_SUFFIXES: tuple[str, ...]` (L28) — bitsandbytes `Params4bit` packed-stat suffixes written alongside each `.weight`.
- `set_cublas_workspace_env() -> None` (L38) — Pins the cuBLAS workspace BEFORE CUDA initialises (required for determinism).
- `apply_determinism() -> None` (L43) — Forces byte-stable numerics: disables cudnn TF32, forces math SDP (flash/mem-efficient SDP are process-state dependent, math SDP is not).
- `quantize_nf4_in_place(root, compute_dtype) -> int` (L55) — Rewrites large `nn.Linear` → `bnb.nn.Linear4bit` in place (the repo's NF4 rule); returns count replaced. Quantization happens lazily on first `.to("cuda")`.
- `install_linear4bit_shells(root, compute_dtype) -> int` (L93) — Replaces large `nn.Linear` with EMPTY `Linear4bit` shells (no packing) on a meta-device skeleton, so `install_prequantized` can drop in saved packed weights without the ~14 GB RSS spike of an off-meta load.
- `install_prequantized(policy, state, device) -> set` (L133) — Drops saved packed NF4 weights into `Linear4bit` shells via `from_prequantized`; returns consumed checkpoint keys. Bit-identical to the in-place quantization that produced the checkpoint.
- `assign_meta_buffers(model, state, device) -> int` (L167) — Assigns any still-meta buffer (non-persistent rotary `inv_freq`) from `state` by hand, since `load_state_dict` skips non-persistent buffers; raises if a meta buffer is absent from the checkpoint.
- `is_prequantized_checkpoint(path) -> bool` (L192) — True if `path` holds a `model.safetensors` with bnb NF4 packed keys.

### `tools/_robometer_scorer.py`
_In-process stateless scorer for the Robometer-4B reward monitor, companion to `openral_runner.backends.reward.robometer_reward.RobometerInProcessReward`._ `reward_monitor_node` imports `_robometer_scorer.py::_Scorer` directly; there is no separate Robometer ZMQ process or dedicated venv. As of lerobot 0.6.0 the reward model is lerobot's in-tree `lerobot.rewards.robometer.RobometerRewardModel` — a vanilla `AutoModelForImageTextToText` (Qwen3-VL-4B) loaded with plain `transformers`. There is **no** pinned `robometer` git package and **no** `transformers==4.57.1` force-pin. The scorer keeps OpenRAL's NF4 pre-quantized checkpoint (`OpenRAL/rskill-robometer_4b-any-general-nf4`, ~3.3 GB resident), meta-builds the native `RobometerRewardModel` skeleton and drops the packed 4-bit weights (remapped into the native module) in directly — no bf16 spike, no Qwen weight download. Validated live: 3.33 GB NF4, progress ramps to 0.88 + success 0.90 at task completion.

- `_NATIVE_CONFIG_REPO: str` (L50) — `"lerobot/Robometer-4B"`, the HF repo `_native_config` downloads `config.json` from to meta-build the native module skeleton.
- `class _Scorer` (L103) — meta-builds the native `RobometerRewardModel`, remaps + loads the NF4 prequant pack, then `score(frames_rgb, task, num_bins) -> (progress, success)` computes per-frame progress via the module-level `decode_progress_outputs` on `_compute_rbm_logits` (not `compute_reward`, which returns only a scalar). `_load_prequantized`, `_native_config`, `_remap_backbone_key`, `_resolve_local_dir` support the meta-load.

### `tools/build_qwen_vlm_nf4_checkpoint.py`
_Reproducible recipe for the published `OpenRAL/rskill-qwen35_4b-any-general-nf4` pre-quantized NF4 checkpoint. Runs in the sidecar venv. Distinct from `quantize_rskill.py`, which writes an `install_prequantized_linears`-loaded pack for the in-process lerobot runtime; this writes a transformers-native `save_pretrained` checkpoint for the isolated VLM sidecar._

- `main() -> int` (L49) — argparse (`--source`, `--out`); loads the upstream model once (NF4 + serial materialization so the bf16 pass fits 8 GB), `save_pretrained`s the 4-bit weights + processor, then verifies the checkpoint reloads directly as 4-bit (no bf16 spike) and answers a smoke query. Pre-quantizing lets deployment load the 4-bit weights directly (~3.3 GB) with no loader workaround.

### `tools/fix_libero_config.py`
_Auto-fix for the stale `~/.libero/config.yaml` pitfall._

Detects and repairs `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero/config.yaml`) when its absolute paths no longer match the currently-active `libero` package — the file is written once at first LIBERO import and never refreshed, so switching venv / clone / workspace path leaves it pointing at a directory that no longer exists. Wired into the `Justfile` as `_ensure-libero-config` (chained off `sim-libero` / `sim-xvla-libero` / `sim-pi05-libero`). Idempotent.

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

### `tools/topreward_per_frame_demo.py`
_Per-frame TOPReward progress over one recorded episode, rendered as an overlay video. NF4 on an 8 GB GPU._

- `class NF4TOPRewardModel(TOPRewardModel)` — TOPReward whose Qwen3-VL backbone loads in NF4 to fit 8 GB; overrides lerobot's `__init__`, which hard-codes `model_kwargs` with no quantization knob. (L56)
- `per_frame_progress(*, dataset_repo_id, episode, vlm_name, image_key, ...) -> NDArray` — Score every frame of one episode. (L85)
- `render_overlay(frame, value, task) -> NDArray[np.uint8]` — Draw a progress bar, the value and the task caption under an RGB frame. (L157)
- `write_media(frames, progress, task, media_dir) -> None` — Write `progress.mp4` plus start/mid/end stills carrying the overlay. (L191)
- `main() -> int` (L217) — CLI; `--dataset`/`--episode`/`--vlm` (default `Qwen/Qwen3-VL-4B-Instruct`)/`--image-key`/`--num-samples`/`--max-frames`/`--out`/`--media-dir`; skips with `SKIP: no CUDA GPU` when CUDA is unavailable.

### `tools/build_robometer_nf4_checkpoint.py`
_Builds the publishable pre-quantized Robometer-4B NF4 checkpoint: loads the upstream Apache-2.0 `robometer/Robometer-4B` bf16 via the pinned robometer loader, NF4-quantizes in place, and saves a self-contained directory (`model.safetensors` ~3.32 GB, `config.json`/`config.yaml`, tokenizer, preprocessor config) the scorer can meta-load directly as 4-bit. Run with Robometer build dependencies installed; the output uploads to `OpenRAL/rskill-robometer_4b-any-general-nf4`._

- `main() -> int` — CLI entry; `--out <dir>` writes the quantized checkpoint directory. (L45)

### `tools/export_act_onnx.py`
_Exports a LeRobot ACT policy to a single whole-model ONNX graph (`ACTPolicy.model`, called as `predict_action_chunk` does). Normalization stays external (checkpoint's `policy_preprocessor.json`/`policy_postprocessor.json` MEAN_STD sidecars); inputs ordered by `config.image_features` plus the checkpoint's declared state dim. Parity checked in `tests/integration/test_act_onnx.py`._

- `export(out_path, repo_id, *, device="cpu", preprocess="host") -> str` — CLI entry (`--repo-id`, `--out`). (L147)

### `tools/export_rtdetr_onnx.py`
_Exports `PekingU/rtdetr_r18vd_coco_o365` to ONNX matching `ObjectsDetector`'s contract: single image input, /255 preprocessing, two 3-D outputs (pre-sigmoid logits + cxcywh-normalised boxes). Run in an isolated `uv run --isolated --no-project` overlay — the project venv's torchvision would shadow it and break the `RTDetrForObjectDetection` import; do not `uv sync` the onnx-export group._

- `export(out_path, model_id="PekingU/rtdetr_r18vd_coco_o365") -> None` — CLI entry (`--out`). (L30)

### `tools/gen_nav2_visual.py`
_Generates `packages/openral_nav2_bringup/config/nav2_visual.yaml` (the Nav2 costmap profile for the visual-SLAM backend — cuVSLAM + nvblox, `static_layer` off the backend-agnostic `/map` OccupancyGrid instead of ray-casting `/scan`) from the base lidar profile `nav2_panda_mobile.yaml`, so the two stay in sync. One-shot; re-run after editing the base config._

- `main() -> int` — CLI entry, no arguments. (L50)

### `tools/openpi_to_lerobot_pi05.py`
_Converts an OpenPI π0.5 Orbax checkpoint (stacked JAX params, as the RoboCasa365 release ships) into a LeRobot PI05 checkpoint (unstacked PyTorch `model.safetensors` + policy processor sidecars). Deterministic format bridge only — quantization stays in `tools/quantize_rskill.py`._

- `main() -> int` — CLI entry; downloads/restores Orbax params, maps state dict, copies sidecars, patches config, validates shapes. (L312)

### `tools/quantize_lingbot_vla2.py`
_Pre-quantizes LingBot-VLA 2.0's Qwen3-VL backbone to an NF4 pack ahead of time (the sidecar normally does this at load) so deploys download ~7 GB instead of 25.5 GB and skip the per-boot conversion. Runs in the sidecar venv (torch 2.9 / transformers 4.57.3 / bitsandbytes) importing `tools/_lingbot_vla2_server.py`'s own helpers so the pack matches the runtime shells byte-for-byte. Frugal streaming keeps GPU peak at a few hundred MB and host peak at ~30 GB._

- `main(argv) -> int` — CLI entry (`--ckpt`, `--out`, `--qwen`). (L313)

### `tools/quantize_rskill.py`
_Quantize a lerobot policy and upload the result to the HuggingFace Hub. Policy-agnostic counterpart to the in-process fast-path in `openral_sim._quantization`. Loads any lerobot policy via `from_pretrained`, quantizes in place, saves `model.safetensors` + `quantization_metadata.json`, and uploads to a target rSkill repo — the adapter's `load_prequantized_state_for_rskill` detects the metadata sentinel and skips the on-line nf4 conversion on every subsequent launch. Not part of CI — a one-shot Hub-mutating tool gated on `HF_TOKEN`._

- `main() -> int` (L537) — CLI entry; `--repo-id`/`--out-repo-id`/`--quantization`/`--trust-remote-code`; loads, quantizes in place, uploads.

### `tools/_verify_lingbot_nf4.py`
_Verifies the pre-quantized LingBot-VLA 2.0 nf4 pack (sidecar venv, one process). Loads the nf4 pack through the SERVER's prequant fast path (`_LingBotPolicy` with `--model <nf4 dir>`), runs one seeded RoboTwin-shaped dummy observation through the real 6.38 B model, and proves the overlaid backbone weights are byte-for-byte what the on-line `.to(cuda)` pack would have produced — without loading a second 7 GB model. Also reports the inherent nf4 round-trip error against the bf16 source (informational)._

- `main(argv: list[str]) -> int` (L43) — argparse (`--nf4`, `--source`, `--qwen`, `--samples`); for sampled backbone `Linear4bit` modules, dequantizes the overlaid (fast-path) weight and compares it against the same weight packed fresh from the source fp32 checkpoint, asserting `max|Δ| == 0`.

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
_Polls for a ROS 2 action server to appear, then publishes an `Empty` message on `/openral/skill_registry_changed` so the reasoner re-seeds its rSkill palette. Needed because wrapped-ROS rSkills (`kind: ros_action`/`ros_service`) drop from the palette when their `interface_name` isn't yet advertised at the reasoner's early-launch autostart, but e.g. Nav2's lifecycle dance takes 15-30 s to advertise `/navigate_to_pose`. Exits 0 once the trigger is published; exits 1 if the action never appears within `--timeout-s`._

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
