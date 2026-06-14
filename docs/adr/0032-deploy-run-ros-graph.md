# ADR-0032: `deploy run` runs the ROS launch graph against real hardware (Effort 2)

- Status: **Proposed**
- Date: 2026-06-01
- Related: [ADR-0031](0031-sim-real-hal-separation.md) (`build_hal(mode)`; this builds on it);
  [ADR-0029](0029-unified-hal-lifecycle-node.md) (one robot.yaml-driven lifecycle node — this
  ADR subsumes its motivation); [ADR-0025](0025-reasoner-managed-background-services.md)
  (`sim_e2e.launch.py`); CLAUDE.md §3 (HAL layer 0), §1.1 (safety), §1.5 (hot path is C++).

## Context

After ADR-0031, HAL *type* is chosen by `openral_hal.build_hal(description, *, mode)`. But the
two deployment commands still use **different stacks**:

- `openral deploy sim` shells `ros2 launch openral_rskill_ros sim_e2e.launch.py` — the full
  production ROS graph (HAL lifecycle node, runtime/skill node, **C++ safety kernel**, world
  state, reasoner, SLAM/Nav2/octomap, dashboard) with a **simulation** HAL at layer 0.
- `openral deploy run` spawns an **in-process** `HardwareRunner` (so100-only) — a single Python
  tick loop with sensor readers, `NullSafetyClient`, and a Rich summary. It has **no C++ safety
  kernel, no reasoner, no world-collision check** — strictly less than the sim graph.

This is backwards: real hardware gets the *weaker* stack. The goal: `deploy run` should run the
**same ROS graph** as `deploy sim`, swapping the sim HAL for the real HAL and dropping the sim
scene — and only work if real hardware is connected.

Two enabling facts from the Effort-2 investigation:
1. The ROS graph is **sim-only today**: the so100 node switches sim↔real on a param, but the
   other 7 HAL nodes (`franka/ur5e/ur10e/aloha/g1/h1/rizon4`) hardcode their sim class via
   `make_lifecycle_main(node_name, <SimClass>)` — no real path.
2. `HardwareRunner` is **not** the thing being retired — `openral_rskill_ros`'s `runtime_node`
   already composes it internally. Only the `deploy run` **CLI entry** changes (from spawning
   an in-process runner to shelling the launch).

## Decision

1. **One parameterized launch.** Add `hal_mode:=sim|real` to `sim_e2e.launch.py` (rename later
   if desired). `deploy sim` passes `sim`; `deploy run` passes `real`. No separate
   `real_e2e.launch.py` — one graph, two modes (DRY; the only delta is the HAL the node builds
   and the absence of the sim scene-attach).
2. **Manifest-driven HAL nodes.** Convert all HAL lifecycle nodes to a single
   `make_lifecycle_main_from_manifest(node_name)` helper whose `_create_hal()` reads the
   forwarded `robot_yaml` + `hal_mode` params and returns `build_hal(description, mode=hal_mode,
   transport=<ros params>)`. This finishes the ADR-0029 unification and adds real support to
   every robot at once.
3. **`deploy run` shells the launch.** The `deploy run` CLI mirrors `deploy_sim.py`'s
   `resolve_launch_invocation`, shelling `ros2 launch … hal_mode:=real`. Real `connect()` fails
   loudly with no hardware (ADR-0031). The in-process-runner spawn is removed from the CLI;
   `HardwareRunner` lives on inside `runtime_node`.
4. **Real safety is the C++ kernel.** In real mode the HAL publishes `/joint_states` and obeys
   `/openral/safe_action` from the kernel — the `NullSafetyClient` stub is gone from the
   real path. This is a safety *improvement* (§1.1).
5. **ros2_control arms assume a pre-launched driver.** For UR/Franka (`RosControlHAL`), the
   `ur_robot_driver` / `franka_ros2` stack is started by the operator (or a HIL fixture);
   the HAL node waits for `/joint_states` + the controller topics. Auto-launching the vendor
   driver is a follow-up.

## Alternatives considered

- **Separate `real_e2e.launch.py`** — rejected; duplicates ~800 lines that drift. One launch +
  `hal_mode` is the symmetric vision.
- **Keep the in-process `deploy run`** — rejected as the primary path; it can't host the C++
  kernel/reasoner and is so100-only. (May survive as a thin library entry for offline dataset
  replay, out of scope here.)
- **Build the GStreamer GPU-passthrough ROS node now** — deferred; the CPU sensor path works as
  a baseline. GPU-passthrough on Jetson/H100 is a follow-up (no ROS-node equivalent exists yet).

## Consequences

- Real hardware gets the full production stack (kernel, reasoner, world state, nav) — strictly
  better than today.
- Every robot (not just so100) can run on real hardware via the ROS graph; robots with
  `hal.real = null` raise `ROSCapabilityMismatch` at launch.
- Layer touch: layer 0 (HAL nodes) + the skill-ros launch + the CLI. No schema change.
- Known follow-ups (own PRs): GStreamer GPU-passthrough ROS node; dataset recording
  (`RolloutRecorder`) in the ROS graph; per-tick deadline/budget reporting on the skill node.

## Phased plan

- **Phase 1 — HAL node mode-switching — *done.*** `make_lifecycle_main_from_manifest` +
  `_ManifestHALLifecycleNode` (`robot_yaml` + `hal_mode` params → `build_hal(mode)`); all 7
  hardcoded-sim nodes converted; `deploy_sim` forwards `robot_yaml` + `hal_mode="sim"` via the
  `manifest_driven` `_HalSpec` flag. Verified on an RTX 4070 + ROS 2 Jazzy host (rclpy node
  mode-dispatch) + `tests/unit/test_hal_lifecycle_manifest.py`.
- **Phase 2 — launch resolution + CLI — *done.*** `resolve_launch_invocation` gained
  `hal_mode` + optional `config`; real mode skips the sim twin/scene injection and fast-fails a
  sim-only robot; `run_launch_invocation` is the shared shelling path. `deploy run` now resolves
  the robot from a `RobotEnvironment` and shells `sim_e2e.launch.py` with `hal_mode:=real`, no
  in-process runner. No `sim_e2e.launch.py` change was needed (the launch argv has no scene arg;
  `hal_mode` flows via the HAL params file). Resolution + error/ happy paths unit-tested
  (`test_deploy_run_real_resolution.py`, `test_cli_deploy.py`); the live `ros2 launch` + real
  `connect()` are HIL-verified.
- **Phase 3 — HIL tests + docs:** `tests/hil/test_real_e2e_<robot>.py` (live launch on a robot
  host); per-robot real transport plumbing for the manifest nodes (so100's `port` works today;
  UR/Franka `robot_ip`/`fci_ip` need the node to declare+forward transport — a follow-up);
  update CLAUDE.md §1.13, deploy CLI docs. METHODS + repo-state-map are updated in this change.
