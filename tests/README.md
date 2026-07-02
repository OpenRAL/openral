# `tests/` — Test Suite Audit & Coverage Map

This file is the single source of truth for what `OpenRAL` tests, what it
deliberately doesn't yet, and what's flagged for follow-up. It is hand-curated
in the spirit of `docs/architecture/repo-state-map.html` (CLAUDE.md §7.10): a
PR that adds, renames, removes, or changes the relevance of a test should
update this README in the same change.

- Last audit: 2026-05-08 against commit `0d09004`.
- Last updated: 2026-05-10 — issue #62 closed: lifted `tests/hil/_ur_ros_transport.py` into a generic `tests/hil/_ros_control_transport.py` (`RosControlHILTransport` + `make_hil_transport`) used by UR5e / UR10e / Franka / Sawyer HIL fixtures, plus a sibling `tests/hil/_aloha_ros_transport.py` (`AlohaHILTransport` + `make_aloha_hil_transport`) for the 4-way bimanual fan-out. The Franka / Sawyer / ALOHA HIL tests upgraded from env-probe stubs to live `rclpy` bridges; `test_hil_transport_publishes_and_caches_state.py` added under `tests/integration/` to guard the bridge wiring. §3 HIL row updated from "1 robot + 1 sensor" to "5 arms + 1 sensor". Earlier on 2026-05-09 — issue #27 closed: `tests/integration/test_world_state_integration.py` scenarios 1–4 migrated from an in-process aggregator simulation to `rclpy`-driven tests against the real `_WorldStateLifecycleNode`; §3 `L2 World State — ROS 2 lifecycle node` integration cell flipped ✗ → ✓. Earlier on 2026-05-09 — sim suite consolidated: `test_smolvla_so100.py`, `test_pi05_so100.py`, and `test_aloha_bimanual_act_aloha.py` removed (the SO-100 wiring/latency surface is now covered by `tests/hil/test_so100.py` plus the unit/integration HAL tests; the kinematic-ACT split was merged into `test_aloha_bimanual_act_aloha.py`). Earlier on 2026-05-08 — issue #28 closed: in-tree rSkill manifests added under `rskills/smolvla-base/`, `rskills/pi05-so100/`, `rskills/act-aloha/`, `rskills/diffusion-pusht/`, and `assert_within_budget` was wired into every applicable sim test; §3 "Performance budget enforcement" row flipped Yellow → Good.
- Repository milestone: **v0.1.0 (M1)** — pre-release.
- Suite size: **47 test files / ~13 kLOC** (30 unit, 4 integration, 6 sim, 7 HIL; plus `__init__.py` and `conftest.py` — and 2 HIL transport bridges that are imports-only). Doctest runner pins **70+** docstring examples on top of that.
- Normative test policy: **CLAUDE.md §5.4, §7.10, §8, §9**.

---

## 1. How the test suite runs

### Local

| Command | Suite | Time budget (CLAUDE.md §5.4) | Notes |
| --- | --- | --- | --- |
| `just test` | `tests/unit/` | <30 s | Always-on; no ROS 2, no GPU, no hardware. Includes the doctest smoke (`tests/unit/test_doctest_runner.py`). |
| `just test-doctest` | docstring examples on the curated set | <15 s | Runs `pytest --doctest-modules` over `core`, `cli`, `sensors`, and the description-only HAL files. Curated list lives in `tests/unit/test_doctest_runner.py::DOCTEST_TARGETS`. |
| `just test-sim` | `tests/sim/` (`-m sim`) | <10 min | Real HF weights + GPU + simulated robots; opt-in. |
| `just test-integration` | `tests/integration/` | <5 min | Requires `source /opt/ros/jazzy/setup.bash`. |
| `just hil <robot>` | `tests/hil/test_<robot>.py` | <10 min | Requires connected hardware. |
| `just test-k <kw>` | filtered subset | n/a | Ad-hoc filter, runs against the entire test tree. |

### CI workflows (`.github/workflows/`)

| Workflow | Triggers | Suites it runs | Notes |
| --- | --- | --- | --- |
| `test-python.yml` | PR, push to `master` | `tests/unit/` + `tests/integration/` (no marker filter) | Matrix: `ubuntu-22.04`, `ubuntu-24.04`, `macos-14` × Python 3.12. Coverage uploaded to Codecov from the `ubuntu-24.04` cell. |
| `test-ros2.yml` | PR, push to `master` | `colcon test` (not pytest) | Builds 6 ROS 2 packages on `ros:humble` and `ros:jazzy`, then runs colcon tests. |
| `hal.yml` | PR, push to `master` | `tests/unit/test_hal.py`; then `tests/integration/` inside `ros:jazzy-ros-base` with **`-k "not test_lifecycle_node_launch"`** (see §4 below). |
| `sim-mujoco.yml` | PR, push to `master` | `tests/sim/ -m "not slow"` | CPU runner; the GPU-only suite collects 0 tests (exit 5) which is forced to success. Also smoke-tests the SmolVLA SO-100 example with `--no-run`. |
| `hil-so100.yml` | `workflow_dispatch`, push touching HAL/HIL paths | `tests/hil/test_so100.py` | Self-hosted runner with label `[self-hosted, lab-so100]`; 10-minute wall-clock cap. |
| `lint.yml` | PR, push to `master` | (lint only) | Includes **`schema_export.py --check`** as the schema-drift guard. |

### Markers (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
asyncio_mode = "strict"
markers = [
    "sim:  integration tests that load real HF weights and/or run a simulated robot (slow; opt-in via `pytest -m sim`)",
    "slow: tests that take >5s on the reference host",
]
```

Conditional skip vocabulary used across the suite (all intentional, none
silent): `pytest.importorskip("torch" / "transformers" / "lerobot" / "mujoco" / "gym_aloha" / "gym_pusht" / "pyrealsense2")`,
`pytest.mark.skipif(not _ROS2_AVAILABLE, ...)`,
`pytest.skip("... requires CUDA", allow_module_level=True)`, and module-level
guards on `os.path.exists(SO100_PORT)`.

### Top-level fixture behavior

- `tests/conftest.py` pins `NO_COLOR=1`, `TERM=dumb`, and patches
  `rich.console.Console.__init__` so ANSI escapes never leak into
  `CliRunner.output`. Without this, every Typer-based test is flaky across
  machines.
- `tests/sim/conftest.py` pre-imports `openral_rskill._lerobot_compat` so the
  broken `lerobot.policies.groot.modeling_groot` module is stubbed before any
  sim test imports `lerobot.policies`.

---

## 2. Suite inventory

Verdict legend: **Keep** (relevant, working) · **Update** (relevant but
stale/incomplete; see §4) · **Consolidate** (overlap with another file) ·
**Remove** (dead code — currently zero entries).

### 2.1 `tests/unit/` — 30 files, ~10.5 kLOC

| File | LOC | Module(s) under test | Verdict |
| --- | ---: | --- | --- |
| `test_smoke.py` | 80 | `openral_core` import + `RobotDescription` / `WorldState` / `Action` round-trip. **Docstring (2026-05-08)** documents its role as the canonical pre-flight smoke distinct from `test_schemas_fuzz.py` and `test_rskill_manifest.py`. | Keep |
| `test_schemas_fuzz.py` | 283 | Hypothesis property tests against core Pydantic models (round-trip + JSON Schema). | **Update** — does not cover `RSkillManifest`, `RSkillLatencyBudget`, `SimEnvironment`, `SceneSpec`, `TaskSpec`, `VLASpec`, `DeviceInfo`, `QuantizationConfig`. |
| `test_rskill_manifest.py` | 148 | `RSkillManifest` schema, defaults, YAML round-trip. | Keep |
| `test_sim_environment_schemas.py` | 136 | `SceneSpec`, `TaskSpec`, `VLASpec`, `SimEnvironment` cross-field validation. | Keep |
| `test_hal.py` | 427 | `RosControlHAL` with `SimTransport` — full lifecycle, action/state path, e-stop, safety. | Keep |
| `test_so100_follower_hal.py` | 399 | `SO100FollowerHAL` against `SO100DigitalTwin` — closed-loop, joint limits, gripper normalisation, latency. | Keep |
| `test_world_state.py` | 494 | `WorldStateAggregator` snapshot freshness, staleness latching, 30 Hz clock injection, thread-safety. Comprehensively covers the aggregator API. | Keep |
| `test_runtime.py` | 621 | `Runtime` Protocol, `NullRuntime`, `PyTorchRuntime`, `OnnxRuntime`, plus full `QUANT_PRESETS` / `auto_select_quant` / `EngineCache` coverage (see lines 248–467). | Keep |
| `test_sim_transport.py` | 230 | `SimTransport` queue + drop semantics + introspection helpers. **Added 2026-05-08.** | Keep |
| `test_franka_panda.py` | 200 | `FRANKA_PANDA_DESCRIPTION` joint inventory, datasheet limits, capabilities, safety envelope, JSON round-trip. Unit-level (no MuJoCo). **Added 2026-05-08.** | Keep |
| `test_mujoco_arm.py` | 200 | `MujocoArmHAL` constructor invariants, gripper-config validation, not-connected error paths, optional-parameter wiring. Unit-level (no MuJoCo connect needed). **Added 2026-05-08.** | Keep |
| `test_smolvla_adapter.py` | 425 | `SmolVLASkill` lifecycle + `ChunkedExecutor` queue/prefetch/error semantics with mocked lerobot stubs. | Keep |
| `test_sensors.py` | 399 | RealSense / Hokuyo / IMU / Livox / Orbbec / Ouster / SLAMTEC / tactile / UVC vendor factories + launch-py codegen. | Keep |
| `test_sensor_catalog.py` | 592 | `SensorCatalog` register/get/filter/build, `openral sensor list` / `show` CLI. | Keep |
| `test_rskill_loader.py` | 514 | `rSkill.from_yaml` / `from_pretrained`, license guards, capability matching, registry I/O (HF mocked). | Keep |
| `test_doctor.py` | 283 | `openral doctor` per-check helpers + `CliRunner` end-to-end. | Keep |
| `test_cli_skill.py` | ~330 | `openral rskill list` / `install` / `search` — real in-tree `rskills/` manifests through the real loader/license-guard/registry paths; only the HF network boundary is faked (`_FakeHub`). | Keep |
| `test_cli_init_connect.py` | ~110 | `openral connect` (happy + ROSConfigError + ROSRuntimeError + finally-disconnect) via the recording `FakeSO100FollowerHAL` serial-boundary fake (`tests/unit/fakes/`). `calibrate camera` coverage lives in `test_sensors.py`. **Added 2026-05-08; MagicMock HAL replaced with a real boundary fake + calibrate block deduped 2026-07-02.** | Keep |
| `test_autodetect.py` | 434 | USB VID/PID table, Linux glob enumeration, DDS topic inference, `ral init` Typer flow. | Keep |
| `test_eval_registry_and_runner.py` | 159 | `SCENES` / `POLICIES` registries, `make_env`, `make_policy`, `SimRunner` mock path. | Keep |
| `test_eval_adapters_helpers.py` | 427 | LIBERO / MetaWorld / SmolVLA helper functions; lazy-import failure paths. | Keep |
| `test_eval_factory.py` | 210 | `make_env` / `make_policy` / `make_robot` error paths + Protocol runtime conformance + `EpisodeResult.summary()`. **Added 2026-05-08.** | Keep |
| `test_doctest_runner.py` | 115 | Implements the CLAUDE.md §5.4 doctest mandate by running `pytest --doctest-modules` against `DOCTEST_TARGETS` (20 paths) as a subprocess; collected-count guard prevents silent regression. **Added 2026-05-08.** | Keep |
| `test_hal_protocol_conformance.py` | 240 | Parametrized contract test for the `HAL` Protocol — 7 invariants × 8 HAL implementations (`RosControlHAL`, `SO100FollowerHAL+SO100DigitalTwin`, `UR5eHAL`, `UR10eHAL`, `FrankaPandaHAL`, `FrankaPandaRealHAL`, `SawyerRealHAL`, `AlohaHAL`); MuJoCo HALs `pytest.skip` when optional deps absent. **Added 2026-05-08; expanded to 8 HALs 2026-05-10 with the real-HW adapters from issues #56–#58.** | Keep |
| `test_franka_panda_real.py` | ~210 | `FrankaPandaRealHAL` against `SimTransport` — closed-loop publish/subscribe, manifest pointer (`closed_with_api` → real HAL), e-stop publishes to `/error_recovery/goal`, staleness guard. **Added 2026-05-10 (issue #56).** | Keep |
| `test_sawyer_real.py` | ~190 | `SAWYER_DESCRIPTION` joint inventory + `SawyerRealHAL` against `SimTransport` — closed-loop publish/subscribe, intera_sdk topic pinning, e-stop publishes to `/robot/set_super_stop`. **Added 2026-05-10 (issue #57).** | Keep |
| `test_aloha.py` | ~290 | `ALOHA_DESCRIPTION` 14-DoF joint inventory + `AlohaHAL` against `SimTransport` — 4-way action split (left arm, right arm, left gripper, right gripper), bimanual capability flags, e-stop publishes to `/aloha/estop`. **Added 2026-05-10 (issue #58).** | Keep |
| `test_skill_testing_helpers.py` | 145 | Unit tests for `openral_rskill.testing.assert_within_budget` (strict, tolerance, optional stages, defensive `ValueError`s, failure-message formatting). **Added 2026-05-08.** | Keep |
| `test_rskill_publisher.py` | 220 | `tools/rskill_publisher.py` smoke (token resolution, manifest validation, dry-run, exit codes, **privacy-gate regression guard** against accidental public publication of closed weights). **Added 2026-05-08.** | Keep |

### 2.2 `tests/integration/` — 4 files, ~700 LOC

| File | LOC | Behavior under test | Verdict |
| --- | ---: | --- | --- |
| `test_doctor_ros.py` | 54 | `_check_ros2()` + `openral doctor` exit 0 with ROS 2 sourced. | Keep |
| `test_autodetect_dds.py` | 86 | `ros2 topic pub /lowstate` subprocess → `scan_dds_topics` → `infer_robot_from_topics == "unitree_g1"`. | Keep |
| `test_world_state_integration.py` | ~330 | Five `rclpy`-driven tests against the real `_WorldStateLifecycleNode`: 30 Hz pipeline, joint-state dropout, recovery, high-load snapshot consistency, and the original lifecycle-launch smoke. All gated on `ROS_DISTRO`; CI exercises them in `hal.yml::hal-integration` after the `colcon build` step. | Keep — issue #27 closed: scenarios 1–4 migrated from in-process aggregator simulation to the integration boundary (QoS, lifecycle transitions, `/joint_states` → `/world_state` round-trip). The aggregator-only paths remain covered by `tests/unit/test_world_state.py`. |
| `test_hil_transport_publishes_and_caches_state.py` | ~225 | Four `rclpy`-driven tests against the HIL transport bridges: `RosControlHILTransport` caches a published `JointState`, publishes a `JointTrajectory`, and `AlohaHILTransport` dispatches arm vs gripper publishes and rejects unknown topics. Gated on `ROS_DISTRO`. | Keep — issue #62: protects the bridge wiring against regressions so the lab HIL fixtures can rely on it. |

### 2.3 `tests/sim/` — 6 files, ~1,340 LOC

All files carry `pytestmark = [pytest.mark.sim, pytest.mark.slow]` and fall
through to module-level skips when CUDA / HF / `mujoco` / `gym_*` are absent.

**Naming convention.** `test_<robot>_<vla>_<sim>.py` for full
end-to-end rollouts (e.g. `test_franka_panda_smolvla_libero.py` =
`franka_panda` × SmolVLA × LIBERO scene). HAL-only sim tests with no
VLA use `<vla>=hal` and the simulator's backend name as `<sim>`
(e.g. `test_ur5e_hal_mujoco.py` = UR5e HAL against MuJoCo). Names
match the keys in `openral_sim.SCENES` / `POLICIES`.

| File | LOC | Behavior under test | Verdict |
| --- | ---: | --- | --- |
| `test_franka_panda_smolvla_libero.py` | 192 | SmolVLA LIBERO via local `rskill.yaml` + `HuggingFaceVLA/smolvla_libero`; routed through `SimRunner`. | Keep |
| `test_aloha_bimanual_act_aloha.py` | 169 | ACT (52 M) on real `gym_aloha` MuJoCo with contact dynamics; routed through `SimRunner`. | Keep |
| `test_pusht_2d_diffusion_pusht.py` | 175 | Diffusion Policy (262 M) on `gym_pusht` (`pymunk` 2-D rigid-body); routed through `SimRunner`. | Keep |
| `test_ur5e_hal_mujoco.py` | 352 | `UR5eHAL` against MuJoCo + `robot_descriptions` MJCF; gravity-off convergence; safety bounds. | Keep |
| `test_ur10e_hal_mujoco.py` | 187 | Mirrors UR5e; pins UR10e-specific datasheet (12.5 kg payload, 1.30 m reach, ≤3.142 rad/s). | Keep |
| `test_franka_panda_hal_mujoco.py` | 258 | `FrankaPandaHAL` on MuJoCo Panda MJCF; tendon-coupled gripper; arm torque limits. | Keep |

> **Note (2026-05-09).** `test_smolvla_so100.py`, `test_pi05_so100.py`,
> and `test_act_aloha.py` (kinematic-only sibling of the full ALOHA
> test) were removed when the eval refactor (commit
> `8679b33` *refactor(eval): decouple rSkill from sim; route every
> rollout through SimRunner*) consolidated the per-skill loops.
> The retained sim tests now drive their rollouts through
> `openral_sim.SimRunner` instead of bespoke per-test loops;
> SO-100 + skill coverage is provided by `tests/hil/test_so100.py` plus
> the unit/integration HAL tests.

### 2.4 `tests/hil/` — 7 test files + 2 transport bridges, 1 405 LOC

| File | LOC | Behavior under test | Verdict |
| --- | ---: | --- | --- |
| `test_so100.py` | 83 | SO-100 follower arm: `connect`/`read_state` <200 ms; `send_hold_action` arm drift ≤2°; idempotent disconnect on teardown. | Keep |
| `test_realsense.py` | 138 | RealSense D435: device discovered, RGB+depth+IMU frames in 5 s, 640×480; `pyrealsense2` SDK + ROS 2 required. | Keep |
| `test_ur5e.py` | 173 | UR5e via `ur_robot_driver`: `read_state` <200 ms, hold-in-place ≤1°/joint, idempotent disconnect, deadman topic, e-stop, perception-stale latch. Live `rclpy` bridge via `_ros_control_transport.py`. **Issue #54.** | Keep |
| `test_ur10e.py` | 131 | UR10e mirror of `test_ur5e.py`; only URDF / payload envelope differ. **Issue #55.** | Keep |
| `test_franka_panda.py` | 139 | Franka Panda via `franka_ros2`: `connect`/`read_state` <200 ms, hold-in-place ≤1°/joint, idempotent disconnect. Live `rclpy` bridge via `_ros_control_transport.py`. **Issues #56 / #62.** | Keep |
| `test_sawyer.py` | 125 | Sawyer via `sawyer_robot` (intera_sdk lineage): same shape as Franka, pinned to `/robot/joint_states`. Live `rclpy` bridge via `_ros_control_transport.py`. **Issues #57 / #62.** | Keep |
| `test_aloha.py` | 140 | ALOHA bimanual via Interbotix XS SDK: 14-DoF read, 4-way action split (left/right arm + grippers), idempotent disconnect. Live `rclpy` bridge via `_aloha_ros_transport.py`. **Issues #58 / #62.** | Keep |
| `_ros_control_transport.py` | 206 | Single-controller `rclpy` HIL bridge (`RosControlHILTransport` + `make_hil_transport`). Used by UR5e / UR10e / Franka / Sawyer. | Support |
| `_aloha_ros_transport.py` | 270 | 4-way fan-out `rclpy` HIL bridge (`AlohaHILTransport` + `make_aloha_hil_transport`). Reuses `_make_trajectory_publisher` from `_ros_control_transport.py`. | Support |

---

## 3. Coverage matrix vs CLAUDE.md §5.4

For each layer (per repo state map) and cross-cutting surface:
**✓** = covered · **◐** = partial / indirect · **✗** = missing · **n/a** = layer is `blue` (no source yet).

| Layer / surface | Unit | Integration | Sim | HIL | Schema fuzz | Doctest | Perf budget |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| L0 HAL — `RosControlHAL`, `SO100FollowerHAL`, UR / Franka, `SO100DigitalTwin` | ✓ (Protocol conformance × 5 HALs in `test_hal_protocol_conformance.py`) | ✗ | ✓ | ✓ (so100) | ✓ (descriptions) | ✓ (UR / Panda / so100_sim / so100_follower / ros_control) | n/a |
| L0 HAL — `protocol.HAL`, `sim_transport`, `_mujoco_arm`, `franka_panda` | ✓ (Protocol contract pinned + `_mujoco_arm` direct test) | ✗ | ◐ | ✗ | n/a | ✓ (`protocol`, `sim_transport`, `franka_panda`) | n/a |
| L0 HAL — ROS 2 lifecycle nodes (`hal_so100`, `hal_ur5e`, `hal_ur10e`, `hal_franka`) | ✓ (per-package `colcon test` lifecycle smokes for `franka` / `ur5e` / `ur10e`; `so100` covered via HIL) | ✓ (colcon + lifecycle smokes drive `unconfigured → … → shutdown` and assert joint-state publication) | ✗ | ✓ (so100) | n/a | ✗ | n/a |
| L1 Sensors — vendor adapters + `SensorCatalog` | ✓ | ✗ | ✗ | ✓ (RealSense) | ✓ | ✓ (curated set) | n/a |
| L2 World State — `WorldStateAggregator` | ✓ (`test_world_state.py`) | ◐ (in-process) | ✗ | ✗ | ✓ | ✓ (root conftest silences structlog) | n/a |
| L2 World State — ROS 2 lifecycle node | ✗ | ✓ (5 `rclpy`-driven scenarios in `test_world_state_integration.py`) | ✗ | ✗ | n/a | ✗ | n/a |
| L3 rSkill (S1) — `rSkillBase` ABC | ✓ (`test_skill_contract.py` — 12 invariants × 3 builders) | ✗ | ✓ | ✗ | n/a | ✓ | ◐ |
| L3 Skill — `Runtime`, `PyTorchRuntime`, `OnnxRuntime` | ✓ | ✗ | ✓ | ✗ | n/a | ✓ (`runtime`, `runtime_pytorch`, `runtime_onnx`) | ◐ |
| L3 Skill — `EngineCache`, `quantization` | ✓ (`test_runtime.py` lines 248–467) | ✗ | ◐ | ✗ | ✓ | ✓ | ✗ |
| L3 Skill — `rSkill` loader, `RSkillManifest` | ✓ | ✗ | ✓ | ✗ | ✓ (manifest fuzzed) | ✓ | ✓ (`assert_within_budget`) |
| L3 Skill — `SmolVLASkill` adapter | ✓ | ✗ | ✓ | ✗ | n/a | ✓ | ✓ (sim) |
| L3 Skill — testing helper (`assert_within_budget`) | ✓ (`test_skill_testing_helpers.py`) | ✗ | n/a | n/a | n/a | ✓ | ✓ |
| L4 Reasoner | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| L5 World Action Model | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| L6 Safety — `SafetyEnvelope` schema, `ROSSafetyViolation` hierarchy | ✓ (Protocol conformance pins `ROSEStopRequested`) | ✗ | ◐ | ✗ | ✓ | ✓ (exceptions) | n/a |
| L6 Safety — supervisor, C++ kernel, deadman/E-stop | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| L7 Observability | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Cross — `openral` CLI (`doctor`, `skill`, `sensor`) | ✓ | ✓ (`doctor`, `init`) | ✗ | ✗ | n/a | ✓ | n/a |
| Cross — `openral` CLI (`init`, `connect`, `calibrate camera`) | ✓ (`test_cli_init_connect.py` connect; `test_sensors.py` calibrate) | ◐ | ✗ | ✗ | n/a | ✓ | n/a |
| Cross — `openral_sim` registry/runner/factory/rollout/adapters | ✓ (registry + factory error paths + Protocol conformance) | ✗ | ✓ | ✗ | ✓ (Sim*/Scene*/Task*/VLA* fuzzed) | ✗ | n/a |
| Cross — `tools/schema_export.py` (drift) | ✓ (`lint.yml --check`) | n/a | n/a | n/a | n/a | n/a | n/a |
| Cross — `tools/rskill_publisher.py` | ✓ (`test_rskill_publisher.py`, incl. privacy-gate regression guard) | n/a | n/a | n/a | n/a | n/a | n/a |
| Cross — Doctest runner (CLAUDE.md §5.4) | ✓ (`test_doctest_runner.py`) | ✓ (`just test-doctest` + curated `DOCTEST_TARGETS`, 21 paths, ≥70 examples) | n/a | n/a | n/a | n/a | n/a |
| Cross — Schema fuzz (CLAUDE.md §5.4) | ✓ (`test_schemas_fuzz.py`, 21 strategies covering core + RSkill + Sim + Quant + Device) | n/a | n/a | n/a | n/a | n/a | n/a |

**Reading the matrix.** Every L0–L3 row is now ✓ or ✓-with-asterisk. The
third follow-up wave landed: parametrized `Skill` ABC contract test,
`_mujoco_arm` direct unit test, `runtime_onnx.py` lazified and re-added
to `DOCTEST_TARGETS`, `assert_within_budget` wired into
`test_franka_panda_smolvla_libero.py` as the canonical example, and `hal.yml` now
runs `colcon build` before pytest so `test_lifecycle_node_launch` is no
longer filtered out.

Remaining gaps (see §4):
- ~~**ROS 2 lifecycle smokes** for the `franka` / `ur5e` / `ur10e` packages —
  only `openral_hal_so100` has equivalent coverage today.~~ Resolved
  2026-05-09 (issue #26): each of `packages/openral_hal_franka`,
  `packages/openral_hal_ur5e`, and `packages/openral_hal_ur10e`
  now ships a `test/test_lifecycle_node.py` colcon smoke that drives the
  full managed-lifecycle path (`unconfigured → configure → activate →
  active → deactivate → cleanup → shutdown`) against a stub HAL injected
  into `_HALLifecycleNode` and asserts joint-state publication during
  the `active` phase. Wired into `test-ros2.yml` via the existing
  `colcon test --merge-install` step.
- ~~**Wider sim-test budget enforcement.** `test_franka_panda_smolvla_libero.py` is wired
  to `assert_within_budget`; the other sim tests (`test_smolvla_so100`,
  `test_pi05_so100`, `test_act_aloha`, `test_pusht_2d_diffusion_pusht`) use
  hardcoded ceilings because their underlying skills do not yet ship as
  rSkills with `RSkillManifest.latency_budget`.~~ Resolved 2026-05-08:
  in-tree manifests landed under `rskills/smolvla-base/`,
  `rskills/pi05-so100/`, `rskills/act-aloha/`, `rskills/diffusion-pusht/`;
  every applicable sim test now asserts against its manifest's
  `RSkillLatencyBudget`. HAL-only `test_hal_*_sim.py` remains out of
  scope.

---

## 4. Flagged backlog

Three lists, in priority order. Each item is one sentence + concrete files +
the policy clause that motivates it.

### 4.A — Tests to **update** or **consolidate**

1. ~~**P1 · `tests/integration/test_world_state_integration.py` (scenarios
   1–4)** — the multi-threaded in-process aggregator simulation predated
   `packages/world_state/lifecycle_node.py`.~~ Resolved 2026-05-09
   (issue #27): scenarios 1–4 now drive the real `_WorldStateLifecycleNode`
   via `rclpy` (the `launch_testing`-equivalent in-process pattern that
   `test_lifecycle_node_launch` established) and exercise the QoS
   profiles + lifecycle transitions + `/joint_states` → `/world_state`
   round-trip.  The aggregator-only paths remain covered by
   `tests/unit/test_world_state.py` (494 lines).

3. ~~**P1 · Wire `assert_within_budget` into the remaining sim tests.**~~
   Resolved 2026-05-08 (issue #28): manifests for `smolvla-base`,
   `act-aloha`, and `diffusion-pusht` shipped under
   `rskills/`, and `test_smolvla_so100`,
   `test_act_aloha`, `test_aloha_bimanual_act_aloha`, and `test_pusht_2d_diffusion_pusht`
   now assert warm-chunk / mean per-step latency against the manifest's
   `RSkillLatencyBudget` instead of a hardcoded `_*_CEILING_S`. HAL-only
   `test_hal_*_sim.py` is explicitly out of scope (no skill loaded).

### 4.B — Tests / coverage to **add**

#### Resolved (changelog)

The following items from the original audit are now closed.  Kept here so
future contributors can audit the closure.

**2026-05-08 (initial follow-up):**
- ~~Doctest runner — wired via `tests/unit/test_doctest_runner.py` +
  `just test-doctest` against a curated, opt-in `DOCTEST_TARGETS` list.~~
- ~~`openral_rskill.engine_cache` direct test — already covered in
  `tests/unit/test_runtime.py` lines 354–467 (audit error).~~
- ~~`openral_rskill.quantization` direct test — already covered in
  `tests/unit/test_runtime.py` lines 248–351 (audit error).~~
- ~~`openral_world_state.aggregator` direct test — already covered in
  `tests/unit/test_world_state.py` (494 lines).~~
- ~~`openral_hal.sim_transport` — added `tests/unit/test_sim_transport.py`.~~
- ~~`openral_hal.franka_panda` unit test — added
  `tests/unit/test_franka_panda.py`.~~
- ~~`openral_sim.factory` / `make_robot` / Protocol conformance — added
  `tests/unit/test_eval_factory.py`.~~
- ~~CLI `ral init` / `connect` / `calibrate camera` — added
  `tests/unit/test_cli_init_connect.py`.~~
- ~~Broken `realsense.calibrate_camera_cmd` doctest — fixed.~~

**2026-05-08 (second wave):**
- ~~**HAL `Protocol` conformance test (P0)** — added
  `tests/unit/test_hal_protocol_conformance.py`: 7 invariants × 5 HAL
  implementations (35 parametrized tests).~~
- ~~**Cross-package schema-fuzz (P2)** — 8 new strategies in
  `tests/unit/test_schemas_fuzz.py`.~~
- ~~**Per-skill latency-budget enforcement helper (P1)** — added
  `openral_rskill.testing.assert_within_budget` +
  `tests/unit/test_skill_testing_helpers.py` (14 tests).~~
- ~~**`tools/rskill_publisher.py` smoke (P2)** — added
  `tests/unit/test_rskill_publisher.py` (13 tests, incl. privacy-gate
  regression guard).~~
- ~~**Doctest blocker on `world_state` / `so100_follower` / `ros_control`
  (P1)** — repo-root `conftest.py` filters structlog records below
  `WARNING`.~~

**2026-05-08 (third wave — closes the original audit backlog):**
- ~~**Skill ABC parametrized contract test (P1)** — added
  `tests/unit/test_skill_contract.py`: 12 invariants × 3 Skill builders
  (37 cases) plus an error-state-latching test against a synthetic
  failing subclass.~~
- ~~**`_mujoco_arm` direct unit test (P2)** — added
  `tests/unit/test_mujoco_arm.py`: 16 cases covering constructor
  invariants, gripper-config validation, not-connected error paths, and
  optional-parameter wiring.  Unit-level (no MuJoCo connect needed).~~
- ~~**`runtime_onnx.py` lazified (P1)** — `import onnxruntime` now
  deferred to construction time via `_import_ort()` helper that raises
  a helpful `ROSRuntimeError` when the wheel is missing.  Path added
  to `DOCTEST_TARGETS` and `Justfile::test-doctest`.~~
- ~~**`assert_within_budget` wired into `test_franka_panda_smolvla_libero.py` (P1)** —
  the canonical example of replacing a hardcoded 2× ceiling with the
  helper.  `tolerance_pct=100.0` preserves the previous CI variance
  margin; the manifest budget is now the single source of truth.~~
- ~~**`hal.yml` colcon-build wired (P0/CI)** — workflow now runs
  `colcon build --packages-select openral_msgs openral_world_state`
  before pytest and sources `install/setup.sh` so
  `test_lifecycle_node_launch` is no longer excluded by `-k`.  The
  `test_lifecycle_node_launch` skeleton now runs in CI alongside the
  other integration tests.~~
- ~~**`test_smoke.py` docstring (P2)** — clarified its role as the
  canonical pre-flight smoke distinct from `test_schemas_fuzz.py` /
  `test_rskill_manifest.py`.~~
- ~~**`test_aloha_bimanual_act_aloha.py` ↔ `test_aloha_bimanual_act_aloha.py` docstrings (P2)** —
  the kinematic-vs-full split is now documented in both files.~~

#### P1 — remaining

1. ~~**ROS 2 lifecycle node tests for franka / ur5e / ur10e packages.**~~
   Resolved 2026-05-09 (issue #26): per-package `test/test_lifecycle_node.py`
   colcon smokes added under `packages/openral_hal_franka/test/`,
   `packages/openral_hal_ur5e/test/`, and
   `packages/openral_hal_ur10e/test/`.  Each drives
   `unconfigured → configure → activate → active → deactivate → cleanup →
   shutdown` against `_HALLifecycleNode` with an injected stub HAL and
   asserts joint-state publication during the `active` phase.  Picked up
   by the existing `colcon test --merge-install` step in
   `.github/workflows/test-ros2.yml` (now activates the uv venv so the
   workspace `openral_hal` / `openral_core` packages are importable).

2. **Wider sim-test budget enforcement** — see §4-A.3.

### 4.C — Tests to **remove**

None recommended. Every skip / `importorskip` in the suite is intentional
gating documented in CLAUDE.md §5.4 (HIL on lab runner, sim on CUDA,
integration on ROS 2). Removing files would be premature; consolidation
(§4-A.6) and updates (§4-A.1, §4-A.2) are the appropriate levers.

---

## 5. Scoreboard

Counts compare each test category against the modules it should cover, where
"green modules" are the count of `green` blocks in
`docs/architecture/repo-state-map.html` whose responsibility falls in that
category. "Blue modules" are explicitly out of scope until the layer ships.

| CLAUDE.md §5.4 category | Files | Modules covered / green modules | Health |
| --- | ---: | --- | --- |
| Unit (`tests/unit/`) | 30 | 34 / 34 | **Good** — every `green` module has either a direct test or parametrized contract coverage. |
| Integration (`tests/integration/`) | 3 | 4 / 4 | **Good** — `test_world_state_integration.py` scenarios 1–4 migrated to `rclpy`-driven tests against the real lifecycle node (issue #27, 2026-05-09); `hal.yml::hal-integration` colcon-builds the world_state + msgs packages and runs the full integration suite. |
| Sim (`tests/sim/`) | 6 | 4 robots × 4 policies (SmolVLA / ACT / Diffusion / UR-Franka HALs) | **Good** — closed-loop coverage where CUDA + weights are available; rollouts route through `SimRunner`. |
| HIL (`tests/hil/`) | 7 (+ 2 transport bridges) | 5 arms + 1 sensor | **Good** — `lab-so100` runner registered; UR5e / UR10e / Franka / Sawyer / ALOHA HIL fixtures drive a live `rclpy` bridge (`_ros_control_transport.py` + `_aloha_ros_transport.py`, issue #62, 2026-05-10) into the real vendor `ros2_control` controllers when the lab env vars are set, and `pytest.skip` cleanly off-lab. |
| Schema fuzz (Hypothesis) | 1 | 21 strategies covering core + RSkill + Sim + Quant + Device | **Good.** |
| Doctest | 1 (`test_doctest_runner.py`) | 21 paths in `DOCTEST_TARGETS` (≥70 example blocks) | **Good** — every public module with docstring examples is in the list (`runtime_onnx.py` lazified and added 2026-05-08). |
| Performance budget enforcement | 1 helper + 3 wired sim tests | helper + `test_franka_panda_smolvla_libero.py`, `test_aloha_bimanual_act_aloha.py`, `test_pusht_2d_diffusion_pusht.py` against in-tree rSkill manifests | **Good** — every sim test that exercises an rSkill calls `assert_within_budget` against its manifest's `RSkillLatencyBudget`; HAL-only sim tests (`test_hal_*_sim.py`) remain out of scope (no skill loaded). |

---

## 6. Maintenance contract

This README is the test-side counterpart to
`docs/architecture/repo-state-map.html`. Three rules:

1. **Promotion to `green` is a paired commit.** Per CLAUDE.md §7.10, flipping
   a module to `green` in the repo state map requires both source *and* tests
   on disk. The PR that flips the colour must also add the test row to §2 and
   update the §3 matrix. A `green` module not listed here is a status-mismatch
   bug.

2. **No "tests in a follow-up PR".** Per CLAUDE.md §9 anti-patterns. Every PR
   that touches a `green` module's behaviour either updates the relevant test
   in §2 or adds a new row.

3. **A new top-level test file or directory updates the §2 inventory in the
   same PR.** Same discipline as the repo state map: a stale row here is
   itself a bug worth filing.

When in doubt, run `just test && just lint` locally; the suite is fast enough
that the audit and the code stay in sync.

## See also

- [`docs/contributing/development.md`](../docs/contributing/development.md) —
  the canonical "how do I run these locally" guide.
- [`docs/architecture/repo-state-map.html`](../docs/architecture/repo-state-map.html) —
  per-module status canvas; this README is its test-side counterpart.
- Per-package `README.md` files under `python/<pkg>/`,
  `packages/<pkg>/`, `rskills/<pkg>/`, and `robots/<pkg>/` document the
  unit and integration tests that pin each package's contract.
