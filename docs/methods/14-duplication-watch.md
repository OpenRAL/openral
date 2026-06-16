# Duplication & Reuse Watch

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

This is the user-facing deliverable for the goal of "ensure there are no
duplication or redundancy of methods". Each item is something a future
contributor should look at before adding similar code.

### Confirmed redundancy candidates

1. **Sensor `_spec()` private factory helpers** — seven structurally
   identical kwargs-only `_spec()` functions across:
   - `python/sensors/src/openral_sensors/force_torque.py:28`
   - `python/sensors/src/openral_sensors/imu.py:27`
   - `python/sensors/src/openral_sensors/livox.py:31`
   - `python/sensors/src/openral_sensors/ouster.py:23`
   - `python/sensors/src/openral_sensors/hokuyo.py:25`
   - `python/sensors/src/openral_sensors/slamtec.py:27`
   - `python/sensors/src/openral_sensors/usb_uvc.py:51` (`_uvc_spec`)

   Each just constructs a `SensorSpec(...)` from kwargs that map to the
   same `openral_core.SensorSpec` field set. Reasonable
   consolidation: a single `_make_sensor_spec(modality, **fields)`
   helper in `openral_sensors.catalog` (or a new `_factories.py`
   sibling). Low risk because the public `*_spec()` API is unchanged.

2. **Three parallel registries** with the same lookup-by-string pattern:
   - `python/rskill/src/openral_rskill/loader.py:114` — `rSkill` +
     `InstalledRSkillEntry` JSON file registry.
   - `python/sensors/src/openral_sensors/catalog.py:85` —
     `SensorCatalog` in-memory dict.
   - `python/sim/src/openral_sim/registry.py:43` — `_Registry[T]`
     decorator-driven dict.

   These are different in lifecycle (file-backed vs. in-memory) and
   value type (skill vs. sensor entry vs. factory), so deep
   consolidation is not warranted. **Worth aligning method names
   though** — `SensorCatalog.list_ids()`, `_Registry.names()`, and
   `rSkill.list_installed()` all answer the same question with
   different verbs. A future ADR could standardise on one verb.

3. **VLA adapter boundary helpers** — *resolved.* `resolve_device`,
   `resolve_rskill_repo_id`, `run_inference`, `to_numpy_action`,
   `parse_hf_file_uri`, and `materialize_processor_dir` now live in
   `python/rskill/src/openral_rskill/_vla_core.py`. All five eval
   adapters (`smolvla`, `pi05`, `xvla`, `act`, `diffusion`) and the
   skill-side `ChunkedExecutor` route through it. The
   `diffusion` / `xvla` / `pi05` adapters now go through the small
   `python/sim/src/openral_sim/policies/_processors.py::resolve_processor_dir`
   helper, which delegates to `materialize_processor_dir` when the
   weights URI resolves to a manifest that declares a `processors`
   block, falling back to `snapshot_download` for legacy `hf://`
   shapes — the three sister TODOs on the audit closed 2026-05-18.
   **When adding a new VLA family, do NOT re-implement device or
   rSkill resolution; do NOT wrap `policy.select_action` in your own
   `inference_span` block — call `run_inference` so the OTel span
   fires uniformly. For loading the lerobot
   `PolicyProcessorPipeline`, call `_processors.resolve_processor_dir`
   (sim-layer) or `materialize_processor_dir(manifest)` (skill-layer)
   — do NOT call `snapshot_download` directly.**

4. **SmolVLA skill-side `SmolVLAAdapter` vs eval-side `_SmolVLAAdapter` —
   *not a duplication target.*** The two have incompatible input
   contracts on purpose: the skill takes `WorldState` and emits an
   `Action` inside the ROS2 lifecycle (Layer 3, S1 runtime); the eval
   adapter takes a dict `Observation` and emits a flat numpy array
   (Layer 8, sim driver). Collapsing them would force either
   ceremonial Pydantic wrapping in the sim hot loop or widening
   `Skill.step()` to accept dicts (breaks §6.1). With `_vla_core`
   absorbing the cross-cutting seams, residual overlap (checkpoint
   load + processor factory, ~30 LOC each side) is below the
   abstraction-cost threshold. Keep them separate.

5. **`_build_libero_scene` / `_build_metaworld_scene` / `_build_mock_scene`**
   in `python/sim/src/openral_sim/{policies,backends}/{libero,metaworld,mock}.py`
   share the same structure: lazy-import a backend module, instantiate a
   `_*Sim` wrapper, return it. Already correctly DRY through the
   `SCENES.register(...)` decorator pattern; do **not** consolidate
   further.

6. **Policy load-phase heartbeat — *resolved.*** The original threaded
   heartbeat (`pi05._heartbeat`) lived inline in the pi05 adapter and
   hard-coded the `pi05_*` event prefix, the daemon thread plumbing,
   and the GPU memory probe. It now lives once in
   `python/rskill/src/openral_rskill/_diagnostics.py::phase_timer(name,
   *, prefix, gpu_mb, **fields)` and the pi05 / smolvla adapters apply
   it through one-line per-adapter shortcuts (`_pi05_phase` /
   `_smolvla_phase`). **When adding a new VLA family, do NOT roll your
   own heartbeat thread** — wrap every load phase
   (`imports` / `from_pretrained` / `to_device` / `processor_dir` /
   `make_processors` / family-specific quant or compile phases) with a
   thin `_<family>_phase` shortcut so `tools/profile_policy_load.py`
   and `openral dashboard` see the same event shape across all adapters.

### Already correctly DRY (do not flag)

- **SimSensorBridge (ADR-0034)** — the single source for RGB camera publishing + MuJoCo viewer
  under `deploy sim`. All manifest-driven arms route through `openral_hal.sim_sensor_bridge.SimSensorBridge`
  via `_ManifestHALLifecycleNode`. The `panda_mobile` package retains its own wiring until
  Phase 2 of ADR-0034 (the planned dedup refactor). **Do NOT add per-arm camera or viewer
  timers in lifecycle subclasses; extend `SimSensorBridge` instead.**

- **HAL adapters (sim)** — `FrankaPandaHAL`, `UR5eHAL`, `UR10eHAL`,
  `SO100MujocoHAL`, `Rizon4MujocoHAL`, `G1MujocoHAL`, `H1MujocoHAL`,
  `AlohaMujocoHAL`, `OpenArmMujocoHAL` all extend `MujocoArmHAL`.
  Post-ADR-0023 (including the bimanual amendment + the 2026-05
  cleanup that collapsed each subclass `__init__` into a single
  forward to `MujocoArmHAL._init_from_description(<DESCRIPTION>, …)`),
  each subclass is now **one line of meaningful code** — the typed
  `__init__(*, mjcf_path, settle_steps, gravity_enabled,
  staleness_limit_s)` signature is kept so IDEs surface the four
  user-tunable knobs, but every per-robot constant (MJCF URI,
  joint→qpos/actuator maps, gripper config, keyframe/seed-ctrl flags)
  lives entirely in `<ROBOT>_DESCRIPTION.sim` (`SimDescription` /
  `SimGripperDescription`). The seam is
  `MujocoArmHAL._init_from_description` (instance method) → which
  delegates to `MujocoArmHAL._sim_kwargs_for` (static method,
  returning a `_MujocoArmInitKwargs` TypedDict so the `**kwargs`
  unpack into `__init__` is typed-clean under `mypy --strict` with
  no per-subclass `# type: ignore`). Per-robot `_<robot>_mjcf_path`
  helpers were also retired in the same cleanup — every MJCF ref resolves
  through the central `openral_core.assets.resolve_asset` grammar (`rd:`
  / `gym_aloha:` / `openarm:` / `menagerie:` / `file:` schemes; ADR-0058). New
  MuJoCo HALs — single-arm, floating-base humanoid, **or** bimanual —
  should declare an `assets.mjcf` ref (plus an optional `sim:` joint-wiring
  block) in `robots/<id>/robot.yaml` and call
  `MujocoArmHAL.from_description(desc)`. No per-robot Python file is
  required at all; the existing classes only exist so the explicit
  `hal.sim` strings (`"openral_hal.<robot>:<Class>"`) some manifests pin keep resolving.
  `H1MujocoHAL` retains a real subclass body only for its
  `_per_step_update` torque hook (default no-op in `MujocoArmHAL`,
  overridden by H1 to recompute `tau = kp*(target-q) - kv*dq` every
  step) — that PD behavior is H1-specific cerebellar substitute, not
  arm-data, and stays in code.
- **Policy adapter loader seams — *resolved.*** The 2026-05 cleanup
  pulled three parallel copies of `_load_manifest_for_spec` (one each
  in `policies/smolvla.py`, `policies/rldx.py`, `policies/pi05.py`)
  and one copy of the lerobot lazy-import + `ROSConfigError` install
  hint into a new
  `python/sim/src/openral_sim/policies/_policy_loading.py` —
  `load_manifest_for_spec(spec)` and
  `lazy_import_lerobot(adapter_name, *, install_hint=...)`.
  Similarly, the four dtype helpers that used to live in
  `policies/pi05.py` (`_manifest_dtype`, `_normalise_manifest_dtype`,
  `_torch_dtype_for`, `_default_dtype`) were lifted into
  `python/sim/src/openral_sim/_quantization.py` as public
  `manifest_dtype`, `normalise_manifest_dtype`, `torch_dtype_for`,
  `default_dtype_for_device`. The `act.py` adapter still carries
  its own `_load_manifest_for_spec` because the rest of its load
  path is structured around a snapshot of the policy weights; if a
  fifth adapter ever needs the same shape, route it through
  `_policy_loading.load_manifest_for_spec`.
- **Humanoid contract validators vs useful humanoid sims** —
  `G1MujocoHAL` and `H1MujocoHAL` are contract validators only.
  Both robots' floating bases fall without an S0 cerebellar balance
  controller (CLAUDE.md §6.2), so closed-loop sim tests run with
  `gravity_enabled=False`.  This is the same situation a future GR1
  HAL twin (currently still deferred — see below) will be in until
  the C++ S0 cerebellum lands.  Do NOT promote these HALs to "useful
  humanoid sim" by bolting Python balance heuristics onto them —
  that path crosses the S0 layer boundary §6.1 reserves for C++.
  Note that `H1MujocoHAL`'s software PD position loop is **not** a
  balance controller — it's a per-joint Kp/Kd that converts the
  H1 menagerie's torque actuators into the position-target contract
  every other `MujocoArmHAL` subclass implements, and mirrors what
  `unitree_sdk2` does on real hardware.
- **Deliberate digital-twin gaps** — `Sawyer` and `GR1` intentionally
  ship without a MuJoCo HAL twin:
  - **Sawyer**: Rethink Robotics is defunct; no real Sawyer hardware
    will ever be plugged in. Sawyer remains only as a MetaWorld
    VLA-eval robot (no `SawyerHAL`, only `SawyerRealHAL` skeleton).
    Twin would be busywork.
  - **GR1**: still no Python HAL twin — Fourier GR1 is one humanoid
    family along with Unitree G1, and once the C++ S0 cerebellum
    lands (M2) it's the natural second consumer of the humanoid
    HAL pattern that `G1MujocoHAL` set up. Currently only exists as
    an `openral_sim` rollout robot.
  These are documented absences, **not** missing work; do not add HAL
  twins for them speculatively.
- **Real-HW manifest derivation** — every real-HW adapter publishes a
  `*_REAL_DESCRIPTION` constant derived from a sim-side baseline via
  `openral_hal._real_description.make_real_description(base, sdk_kind=...)`.
  The helper centralises the `model_copy` + `sdk_kind` override pattern
  (the `hal` entrypoints are shared, ADR-0031), so kinematics + safety
  envelope + capabilities + HAL entrypoints never
  drift between the sim and real-HW siblings of the same robot. New
  real-HW adapters MUST go through this helper rather than re-typing the
  whole `RobotDescription` constructor. The UR real-HW module (`ur_real.py`)
  uses this helper to derive `UR5e_REAL_DESCRIPTION` /
  `UR10e_REAL_DESCRIPTION` from `UR{5,10}e_DESCRIPTION`.
- **HAL adapters (real-HW)** — three shapes coexist on purpose:
  - `FrankaPandaRealHAL` and `SawyerRealHAL` **compose** `RosControlHAL`
    (delegating wrapper) and add robot-specific structlog metadata + a
    vendor-specific recovery / halt topic publish in `estop()`. This is
    the intended pattern for any real-HW arm whose vendor stack exposes
    a single `ros2_control` joint trajectory controller plus a separate
    recovery topic.
  - `UR5eRealHAL` / `UR10eRealHAL` **subclass** a private
    `_URRealHAL(RosControlHAL)` base in `ur_real.py` to share the
    `ur_robot_driver` controller / topic / deadman defaults. Pick
    subclassing when two adapters share enough defaults to warrant a
    base; pick composition when each adapter has distinct recovery /
    metadata semantics. Any future UR variant (UR3e, UR16e, …) is a
    one-line subclass that swaps the `RobotDescription`.
  - `AlohaHAL` **inlines** the publish/state machinery rather than
    wrapping `RosControlHAL` because it splits a single 14-D action
    across four controllers (two arms + two grippers) — a contract that
    doesn't match `RosControlHAL`'s single-controller assumption.
    Adding a sixth composed-real-HW adapter is the trigger to hoist
    `RosControlHAL`-wrapping logic into a `_RealHALMixin`; adding a
    second multi-controller adapter is the trigger to hoist AlohaHAL's
    fan-out into a `MultiRosControlHAL`.
- **HIL transport bridges (real-HW HALs)** — the single-controller
  `RosControlHILTransport` (`tests/hil/_ros_control_transport.py`) is the
  source of truth for the trajectory wiring; `AlohaHILTransport`
  (`tests/hil/_aloha_ros_transport.py`) reuses the module-private
  `_make_trajectory_publisher` helper rather than duplicating the
  `JointTrajectory` + QoS setup four times.  Both bridges share the
  joint-state caching shape (`_latest` dict, `state()` projection over
  `joint_names`, `wait_for_first_state` helper).  Adding a third HIL
  bridge variant is the trigger to extract the shared subscriber half
  into a `_JointStateCache` mixin.
- **Kernel-twin sim tests** — the four `tests/sim/safety/test_kernel_with_<robot>_*.py`
  files (`so100_digital_twin`, `openarm_twin`, `rizon4_twin`,
  `h1_humanoid_twin`) used to each open-code the subprocess + lifecycle
  + ROS-graph envelope around the C++ safety kernel. After the 2026-05
  cleanup, all four route through
  `tests/sim/safety/_kernel_subprocess.py::{start_kernel, activate_kernel_node, build_kernel_envelope, terminate_kernel}`
  and only declare their embodiment-specific joint-name lists +
  per-test action / state vectors. Adding a fifth robot's kernel-twin
  test means one new short test file that calls the same four
  helpers — do NOT re-roll the lifecycle ceremony.
- **rSkillBase subclasses** — `GpuPassthroughSkill`,
  `SmolVLAAdapter`, `SO100SmolVLASkill` all override the same five
  `_*_impl` hooks. The duplicated method *names* are the contract from
  `Skill` ABC; this is inheritance, not redundancy. `GpuPassthroughSkill`
  (M8 PR I/10) is the canonical "this skill provably runs on GPU"
  reference — its `_step_impl` is the right starting point when
  prototyping a torch.cuda-based Skill that consumes a CPU
  `SensorFrame.data: bytes` and needs to be explicit about device
  placement (raises on missing CUDA rather than silently falling back).
- **Runtime backends** — `NullRuntime`, `PyTorchRuntime`, `ONNXRuntime`, `TensorRTRuntime`
  all implement the `Runtime` Protocol surface
  (`load/infer/quantize/warmup/unload`). Same situation as Skill.
- **`backends/so100_robosuite/`** — `_So100Lift` extends
  `robosuite.environments.manipulation.lift.Lift` rather than
  reimplementing the arena / reward / observable / placement
  scaffolding, and the controller config is the shipped
  `parts/osc_position.json` with three knobs overridden
  (`output_max`, `kp`, `input_ref_frame`) — NOT a custom
  controller class. The scripted policy is correspondingly tiny
  (~150 lines, just Cartesian deltas) because OSC owns the IK.
  The next new robosuite-integrated robot should follow the same
  pattern: register the robot model + gripper in robosuite's
  factories, build the env via robosuite's stock manipulation
  subclasses, pick a stock part controller (`osc_position` /
  `osc_pose` / `joint_position`) and tune only the gain / output
  ranges — do not write a JOINT_POSITION + custom-IK stack like
  the early `so100_robosuite` drafts did.

### Watch list (not yet a problem, but worth tracking)

- **`_validate_action()`** appears in both `MujocoArmHAL` (L296) and
  `RosControlHAL` (L250). They validate different invariants today
  (MuJoCo: `joint_targets` rank; ros2_control: control mode). If a
  third HAL grows a third `_validate_action`, lift the common parts
  into a free function in `openral_hal.protocol`.
- **`_require_connected()`** appears in `MujocoArmHAL` (L289),
  `SO100FollowerHAL` (L386), `RosControlHAL` (L243), and `AlohaHAL`
  (L426). Four is over the threshold — the next HAL adapter that adds
  a fifth `_require_connected` is the trigger to hoist this into a
  base mixin (`openral_hal._lifecycle.RequireConnectedMixin`).
  `FrankaPandaRealHAL` / `SawyerRealHAL` deliberately delegate the
  check to their inner `RosControlHAL` rather than duplicating it.
- **`from_yaml(cls, path)`** classmethods appear in `RSkillManifest`
  (L883) and `SimEnvironment` (L1127). Both are "open file → parse YAML
  → `model_validate(dict)`". Acceptable as a copy because they live in
  the normative schemas module, but a `openral_core._yaml.py`
  helper would remove the boilerplate.

---

*Generated and curated 2026-05-08 from a single AST pass over
`python/`, `packages/`, and `tools/`. Re-run `python3 -c "import ast"`-based
extraction whenever a module is added or renamed; this file is hand-edited
afterwards. If a future contributor automates regeneration, mirror the
pattern in `tools/schema_export.py`.*

