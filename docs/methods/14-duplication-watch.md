# Duplication & Reuse Watch

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

This is the user-facing deliverable for the goal of "ensure there are no
duplication or redundancy of methods". Each item is something a future
contributor should look at before adding similar code.

### Confirmed redundancy candidates

1. **Sensor `_spec()` private factory helpers — *retired.*** The seven-way
   `_spec()` duplication this item originally tracked is gone: the
   `imu` / `livox` / `ouster` / `hokuyo` / `slamtec` modules no longer
   exist under `python/sensors/src/openral_sensors/`. The surviving spec
   factories (e.g. `force_torque.robotiq_ft300s_spec`) are one-per-file
   public API, not duplication.

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
   — do NOT call `snapshot_download` directly.** (The two remaining
   direct `snapshot_download` calls — `policies/diffusion.py`
   norm-stat loading and the exempted `act.py` adapter — are weight/
   norm-stat fetches, not processor-dir resolution; they are not
   regressions of this item.)

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

7. **BEHAVIOR R1Pro wire constants — *resolved in-process, mirrored
   cross-venv.*** The evaluator's raw observation keys and the 61-D/23-D
   contract widths were hand-copied in three importable modules; they now
   live once in `python/sim/src/openral_sim/_behavior_wire.py`
   (`STATE_KEY` / `STATE_DIM` / `ACTION_DIM` / `CAMERA_SENSORS` /
   `CAMERA_RGB_KEYS` / `explicit_port`), imported by the scene backend,
   the `behavior_groot` policy adapter, and `openral_cli.behavior`. The
   sidecar scripts under `tools/behavior_*_sidecar.py` run in isolated
   venvs that cannot import `openral_sim` — their copies are a deliberate
   wire-contract MIRROR: update them in lockstep with `_behavior_wire`.

8. **Support-contact patch predicate — *deliberate cross-package mirror,
   update in lockstep.*** `support_contact_exempts`
   (`cpp/openral_safety_kernel/src/collision.cpp`) and
   `support_patch_withholds`
   (`packages/openral_octomap_bridge/src/payload_clearing.cpp`) evaluate
   the same attested support plane with the same two exact
   discretisation pads (the voxel cube's half-width projected on the
   support normal, and its circumradius laterally) and the same one voxel
   of co-planar headroom on the along-normal bound (added to both sides
   on 2026-08-15, hazard log Entry 012's "Calibration 2026-08-15"). They are not
   consolidated because consolidating them would make a Layer-2
   perception bridge link the Layer-6 safety kernel's collision core —
   the wrong dependency direction, and one that would put `octomap` and
   `tf2` a link away from the real-time kernel. The mirror is safe in
   one direction only and must stay that way: the bridge evaluates the
   bound with **zero slack** while the kernel adds
   `attached_contact_tolerance`, so what the bridge withholds is a
   subset of what the kernel exempts **for the object that attested it**
   (the two scope conditions are in
   `packages/openral_octomap_bridge/README.md`).
   Both predicates bound the along-normal coordinate: the withheld set is
   the support HALF-SPACE below the attested plane plus one projected
   cube half-width of slab above it, never the whole patch cylinder.
   Changing either predicate without the other breaks the partition — the
   failure mode is the 2026-08-14 witness/clearing defect.
   `SupportContactWitness.ThePartitionedClearingLeavesTheWitnessItsEvidence`
   (kernel) and `PayloadClearing.TheAttestedSupportSurfaceSurvivesThe
   Clearing` (bridge) pin the two halves;
   `PayloadClearing.WithholdingIsTheKernelsExemptionPredicateAtZeroSlack`
   pins the mirror itself, evaluating the bridge predicate against a
   term-for-term transcription of the kernel's on the same cells, so a
   drift on either side fails a test instead of a run.
   Both predicates are also **phase-blind** (ADR-0097): neither reads
   `support_id` or `evidence_kind`, so the place-phase witness reuses
   both unchanged and neither side gained a place-specific branch.
   `PayloadClearing.APlacePhaseWitnessIsWithheldExactlyAsAPickPhaseOneIs`
   pins that identity — keep it that way, because a phase-aware branch on
   one side only is exactly how this mirror would drift.

9. **Support-witness acceptance caps — *deliberate C++/Python mirror, update
   in lockstep.*** The kernel's two ingest caps on what a
   `SupportContactWitness` may claim are ROS parameters declared with their
   defaults in
   `cpp/openral_safety_kernel/src/lifecycle_kernel.cpp` —
   `support_witness_max_patch_radius_m` (`0.5`) and
   `support_witness_max_penetration_m` (`0.01`) — and the *producer* refuses
   at the same two numbers, written independently as
   `_SUPPORT_MAX_PATCH_RADIUS_M = 0.5` and
   `_SUPPORT_MAX_PENETRATION_M = 0.01` in
   `python/hal/src/openral_hal/_sim_attachment_evidence.py:51-52`. They are
   not consolidated because the kernel must not trust a producer-supplied
   bound — the cap is exactly the thing the kernel applies to a message it
   did not author, and a shared constant would make the check circular.
   **The drift consequence is asymmetric and both directions are bad.**
   Loosen the producer past the kernel and the kernel fails the whole
   attachment message closed (`ROSSafetyViolation`-class drop, not a
   silent one) — noisy, but conservative. Tighten the producer below the
   kernel and the shortfall is invisible: legitimate contact is never
   attested, the witness never arms, and the robot stops on ordinary
   support contact with nothing in the logs naming a cap as the reason.
   **When either number moves, move both in the same PR** and re-run the
   kernel's ingest gtests together with the producer's refusal tests.

10. **Place-region bounds — *deliberate C++/Pydantic mirror, update in
    lockstep.*** `kMaxPlaceRegionHalfExtentM = 1.5` and
    `kMaxPlaceRegionVolumeM3 = 8.0`
    (`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`)
    are the kernel's bounds on the ADR-0097 approach region, and
    `PlaceRegion.MAX_HALF_EXTENT_M = 1.5` / `PlaceRegion.MAX_VOLUME_M3 = 8.0`
    (`python/core/src/openral_core/schemas.py`, enforced in
    `_validate_region`) are the schema's. The double check is deliberate and
    is named in `PlaceRegion`'s own docstring: an over-large region is
    "rejected here and again in the kernel, both times toward *no
    allowance*". Since this bound gates a **margin reduction**, both sides
    fail closed and neither may be deleted in favour of the other — the
    kernel cannot assume the Pydantic validator ran (the message may reach
    it from a producer that never constructed the model), and the schema
    must still refuse locally so a bad region is a producer-side error
    rather than a kernel-side refusal log. **The drift consequence:** raise
    the Pydantic ceiling without the kernel's and every oversize region is
    accepted by the producer and then silently discarded by the kernel, so
    the place phase runs with **no** allowance while every log line says the
    declaration is live — the exact failure the ADR-0097 amendment's
    clock-domain bug already produced once. Raise the kernel's without the
    schema's and the extra room is unreachable. `PlaceRegionStatus::kOversize`
    is the kernel-side refusal to grep for.

11. **`FailureTrigger` / `SafetyStatus` `KIND_*` numbers — *three-way
    mirror, and one leg is currently drifted.*** The numbers are declared in
    `packages/msgs/msg/FailureTrigger.msg`, **redeclared** in
    `packages/msgs/msg/SafetyStatus.msg` (ROS IDL has no cross-message
    constant reuse), and mirrored a third time as plain Python ints in
    `python/observability/src/openral_observability/failure_bus.py` so
    callers can publish typed failure events without a sourced ROS install.
    The C++ `ViolationKind` enum
    (`cpp/openral_safety_kernel/include/openral_safety_kernel/validator.hpp`)
    is a fourth partial copy of the same numbering.
    `tests/unit/test_safety_status_msg.py` pins the two IDL blocks against
    each other and pins `DROP_*` disjoint from `KIND_*`; a kernel gtest
    (`ViolationKindMapping.EnumValuesMatchFailureTriggerConstants`) pins the
    enum. **Nothing pins the `failure_bus.py` leg, and it has drifted:** it
    stops at `KIND_REASONER_TIMEOUT = 9` (plus
    `KIND_SUPPRESSED_SUMMARY = 254`) and is missing `KIND_COLLISION = 10`,
    which the IDL, the kernel enum and the kernel's
    `publish_collision_failure` all carry. The consequence is scoped but
    real: any non-ROS caller that reaches for a symbolic collision kind
    finds none and must hard-code `10`. Fixing it is a code change and is
    deliberately **not** made in this docs pass — it is recorded here so the
    next PR touching `failure_bus.py` closes it and adds the missing
    contract test rather than re-deriving the discovery. **When a `KIND_*`
    number is added, all four sites move together.**

### Already correctly DRY (do not flag)

- **SimSensorBridge** — the single source for RGB camera publishing + MuJoCo viewer
  under `deploy sim`. All manifest-driven arms route through `openral_hal.sim_sensor_bridge.SimSensorBridge`
  via `_ManifestHALLifecycleNode`. The `panda_mobile` package retains its own wiring until
  the planned dedup refactor lands. **Do NOT add per-arm camera or viewer
  timers in lifecycle subclasses; extend `SimSensorBridge` instead.**
  Its two MJCF body-set resolvers answer different questions and are ***not
  a duplication target***: `depth_cloud.robot_self_body_ids` is "what is the
  robot" (prefix-derived, includes descendants — the depth self-filter),
  while `sim_sensor_bridge.kernel_checked_body_ids` is "what does the safety
  kernel check" (the manifest's `collision_geometry` links, resolved through
  each joint's `sim_joint_name`). The E-stop near-miss probe needs the
  second precisely because it is *narrower* than the first.

- **Bounded `mj_geomDistance` probing** — `sim_sensor_bridge._pair_distance_lower_bound`
  (the vectorised bounding-sphere/plane prefilter) and
  `sim_sensor_bridge._round_robin_candidates` (the fair exact-call budget) are the
  single source for "measure the closest geom pairs across a boundary without
  paying O(n·m)". Two callers share them and **must keep sharing them**:
  `sim_sensor_bridge._nearest_pair_records` (the E-stop ground-truth record) and
  `_sim_attachment_evidence._probe_support_hits` (the support-contact witness).
  They are *not* a duplication target for each other, because they need different
  outputs from the same measurement: the diagnostics path wants named,
  rounded records for a log line, while the witness needs the closest-point
  segment (`fromto`) to reconstruct a contact point and a support plane. **Do NOT
  reimplement the prefilter or the budget; if a third caller needs a third
  output shape, extract the exact-call loop, not the ranking.**
  The reason both exist at all is the same field lesson, recorded twice: MuJoCo's
  contact list is not a proximity oracle — `contype`/`conaffinity` suppression
  empties whole geom pairs (an arm 30 mm inside a freezer door with `ncon == 0`;
  a cup flush on a RoboCasa island with no contact record), so signed distance is
  the adjudicator in both the diagnostics and the evidence path.

- **HAL adapters (sim)** — `FrankaPandaHAL`, `UR5eHAL`, `UR10eHAL`,
  `SO100MujocoHAL`, `Rizon4MujocoHAL`, `G1MujocoHAL`, `H1MujocoHAL`,
  `AlohaMujocoHAL`, `OpenArmMujocoHAL` all extend `MujocoArmHAL`.
  Following the bimanual amendment and the 2026-05
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
  / `gym_aloha:` / `openarm:` / `menagerie:` / `file:` schemes). New
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
  `H1MujocoHAL` and G1's default joint-position path are contract validators.
  Both robots' floating bases fall without an S0 cerebellar balance controller
  (CLAUDE.md §6.2); their joint-convergence tests run with
  `gravity_enabled=False`.
  This is the same situation a future GR1 HAL twin (currently still
  deferred — see below) will be in until the C++ S0 cerebellum
  lands.  Do NOT promote these HALs to "useful humanoid sim" by
  bolting Python balance heuristics onto them — that path crosses
  the S0 layer boundary §6.1 reserves for C++.  The one sanctioned
  sanctioned G1 sim exceptions are ADR-0087's **kinematic-glide base**:
  the free joint is *pinned* upright each step and BODY_TWIST
  Euler-integrates the planar pose — a kinematic navigation
  stand-in with zero dynamics control, NOT a balance controller,
  and ADR-0089's exact upstream MuJoCo Playground ONNX policy + matching MJCF.
  The latter is selected explicitly by `walking_enabled=True`, runs only in the
  sim HAL, and is not a Python balance heuristic or a real-hardware S0.
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
  (the `hal` entrypoints are shared), so kinematics + safety
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
- **Runtime backends** — `NullRuntime`, `PyTorchRuntime`, `ONNXRuntime` (plus
  `TensorRTRuntime` in the private `openral-pro-trt` package)
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

- **Pinhole back-projection of a `32FC1` depth raster** now exists twice:
  `openral_hal.depth_cloud.points_from_depth_grid` (raster → `(N, 3)`
  optical-frame cloud, the deploy-sim bridge's single-cast path) and
  `openral_slam_bringup.depth_height_filter_node.filter_depth_by_global_height`
  (raster → *filtered raster*, projecting only the global-z component
  through one rotation row). Same `(u-cx)/fx` core, different outputs and
  different packages — a third copy is the trigger to hoist a typed
  `deproject_depth(...)` into `openral_core.geometry` and route all of
  them through it.
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
- **`from_yaml(cls, path)` classmethods — *resolved.*** The pattern had
  grown to six copies (`RobotDescription`, `RSkillManifest`,
  `DeployScene`, `SimScene`, `BenchmarkScene`);
  all now share `openral_core.schemas._load_yaml_model(cls, path)`,
  and the byte-identical `SimScene` / `BenchmarkScene` overrides were
  deleted (they inherit `DeployScene.from_yaml`, which returns `Self`).
  A new `from_yaml` on a schemas-module model should be a one-line
  delegation to `_load_yaml_model`.
- **LeRobot SO-ARM unit + cadence conversions in the native MuJoCo
  scenes — consolidated** into
  `openral_sim/backends/_so_arm_units.py`
  (`steps_per_control_period`, `lerobot_action_to_radians`,
  `radians_to_lerobot_state`): `so101_eraser` and `so101_box` each
  carried their own copy of the degrees-mode affine and physics-stepping
  loop, and the copies drifted — `so101_box` shipped a
  single-`mj_step`-per-action cadence and a gripper-as-degrees mapping
  that `so101_eraser` had already fixed. **A new raw-MuJoCo scene that
  accepts LeRobot-convention actions must route through this module**,
  not re-derive the conversions. (`tabletop_push` keeps its own
  `_joint_scales` affine + `settle_steps` knob on purpose: it is
  robot-agnostic, so it cannot assume the SO-ARM "last channel is a
  [0, 100] gripper" convention.)
- **Rotation/quaternion math scattered across packages** — the **yaw
  family is now consolidated** into `openral_core.geometry`
  (`yaw_to_quat_xyzw`, `yaw_to_quat_wxyz`, `quat_xyzw_to_yaw`): the five
  former copies in `openral_hal/…/mobile_base_bridge.py`,
  `openral_world_state/…/{spatial_memory,grid}.py`,
  `openral_sim/…/backends/so101_box/_assets.py`, and
  `openral_runner/…/slam_bridge.py` all route through them. **Before
  adding another yaw↔quat helper, use these.** The remaining full-3-DOF
  conversions are **deliberately left in place**, each for a concrete
  reason, not oversight:
  - quat→matrix in `openral_sim/…/policies/rldx.py` (`_quat_wxyz_to_mat`)
    is SAPIEN wxyz with its own norm-epsilon, pinned "bit-identical to
    upstream WidowXBridgeEnv" — a calibration surface, not a duplicate.
  - rpy→matrix/euler in `openral_safety/…/{mjcf,urdf}_lowering.py` is
    safety-kernel lowering; touching it needs safety-WG review + a
    recorded safety-impact update (CLAUDE.md §3), so it does not move on a cleanup PR.
  - the remaining `_quat_to_matrix` (`world_cloud_bridge.py`, float32) and
    `_rpy_to_*` (`bucket2_markers.py`, `depth_height_filter_node.py`) are
    single-caller and return package-specific types; consolidating them
    would need a typed `quat_xyzw_to_matrix` / `rpy_to_*` set in
    `openral_core.geometry` and is only worth it once a *second* caller
    appears. Add that set (and route new code through it) at that point.

## Resolved by the SAM 2.1 vision-attachment work

- **`_resolve_cameras` (perception nodes)** — `ros_image_detector_node`,
  `scene_vlm_node` and `reward_monitor_node` each carried a byte-identical
  eight-line copy of the `"id=topic"` → map resolution. All three (and the new
  `segmenter_node`) now delegate to
  `openral_perception_ros.camera_topics.resolve_camera_topics`, which is pure,
  ROS-free and unit-tested (`tests/unit/test_perception_camera_topics.py`).
- **`homogeneous_from_quat_xyz`** — was the TF→4x4 step private to
  `openral_world_state.object_lift` (layer 2). The layer-0 HAL's vision
  attachment bridge needs the same step and must not depend on layer 2, so the
  math moved to `openral_core.geometry` and the world-state name became a thin
  wrapper preserving its `ObjectsLiftError` contract. This is the "second
  caller appears" trigger the quat→matrix note above anticipated, for the
  `xyzw`-quaternion-plus-translation shape specifically; the remaining
  `_quat_to_matrix` / `_rpy_to_*` cases listed there are unchanged.
- **Still duplicated, deliberately**: the `SensorSpec`-by-name search exists as
  a private `_sensor_spec` in `packages/world_state/…/lifecycle_node.py` and as
  `sensor_spec_by_name` in `segmenter_node.py`. Two call sites in two ROS
  packages, one of them already private; promoting it means adding public
  surface to `openral_core.schemas`, which is worth doing when a third caller
  appears and not before.

---

*Generated and curated 2026-05-08 from a single AST pass over
`python/`, `packages/`, and `tools/`. Re-run `python3 -c "import ast"`-based
extraction whenever a module is added or renamed; this file is hand-edited
afterwards. If a future contributor automates regeneration, mirror the
pattern in `tools/schema_export.py`.*
