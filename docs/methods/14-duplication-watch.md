# Duplication & Reuse Watch

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated. **Last cleaned 2026-09-22** (openral PR #296): every resolved item collapsed to one line, stale references removed, this pass's findings added. Cite code here as `module.py::symbol`, never with a bare `(LNN)` marker — this file has no source-file sections for the refresher to resolve them against.

This is the user-facing deliverable for the goal of "ensure there is no duplication or redundancy of methods". The mechanical half of that goal now lives in `tools/refresh_methods_linenos.py --check --coverage` (run by `just lint`): every public symbol must have an inventory entry, so a new helper is visible next to its siblings before it is merged. This file carries what a checker cannot: the mirrors that must move in lockstep, the look-alikes that must stay apart, and the copies waiting for a third caller.

### Deliberate mirrors — update in lockstep

Each of these is the same logic on two sides of a boundary that must not be crossed by an import (real-time C++ vs Python, an isolated sidecar venv vs the workspace, a producer vs the kernel that checks it). The drift consequence is named per item; read it before touching either side.

- **BEHAVIOR R1Pro wire constants — resolved in-process, mirrored cross-venv.** Canonical in `python/sim/src/openral_sim/_behavior_wire.py` (`STATE_KEY`, `STATE_DIM`, `ACTION_DIM`, `CAMERA_SENSORS`, `CAMERA_RGB_KEYS`, `explicit_port`), imported by the scene backend, `behavior_groot`, and `openral_cli.behavior`. `tools/behavior_*_sidecar.py` runs in isolated venvs that cannot import `openral_sim`, so its copies are a deliberate wire-contract mirror — update both in lockstep.

- **Support-contact patch predicate — deliberate cross-package mirror, update in lockstep.** `support_contact_exempts` (`cpp/openral_safety_kernel/src/collision.cpp`) and `support_patch_withholds` (`packages/openral_octomap_bridge/src/payload_clearing.cpp`) evaluate the same attested support plane. Kept apart because consolidating would make the Layer-2 perception bridge depend on the Layer-6 safety kernel; the bridge withholds a zero-slack subset of what the kernel exempts. `PayloadClearing.WithholdingIsTheKernelsExemptionPredicateAtZeroSlack` pins the mirror.

- **Support-witness acceptance caps — deliberate C++/Python mirror, update in lockstep.** Kernel ROS params in `cpp/openral_safety_kernel/src/lifecycle_kernel.cpp` (`support_witness_max_patch_radius_m=0.5`, `support_witness_max_penetration_m=0.01`) vs. the same two numbers written independently in `python/hal/src/openral_hal/_sim_attachment_evidence.py`. Kept apart because the kernel must not trust a producer-supplied bound. Drift is asymmetric: loosening the producer fails safe (noisy); tightening it silently starves the witness.

- **Place-region bounds — deliberate C++/Pydantic mirror, update in lockstep.** `kMaxPlaceRegionHalfExtentM`/`kMaxPlaceRegionVolumeM3` (`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`) vs. `PlaceRegion.MAX_HALF_EXTENT_M`/`MAX_VOLUME_M3` (`python/core/src/openral_core/schemas.py`). Both fail closed independently — the kernel can't assume the Pydantic validator ran, and the schema must refuse locally too. Raising the Python ceiling alone silently zeroes the kernel's allowance; grep `PlaceRegionStatus::kOversize` for the refusal.

- **`FailureTrigger` / `SafetyStatus` `KIND_*` numbers — mirrored on every leg, all pinned.** Declared in `packages/msgs/msg/FailureTrigger.msg`, redeclared in `SafetyStatus.msg` (IDL has no cross-message constant reuse), and mirrored as plain ints in `openral_observability/failure_bus.py` for ROS-free callers (partial copies also live in the kernel's `ViolationKind` enum and `reasoner_node.py`). `failure_bus.py` once drifted silently (missing `KIND_COLLISION`); `tests/unit/test_failure_bus_idl_mirror.py` now pins it bidirectionally against the generated IDL.

- **Kernel narrow-phase predicates — deliberate C++/Python mirror, update in lockstep.** `kernel_predicates.py` (`box_box_distance`, `box_capsule_distance`, `capsule_distance`, `shape_distance`) is a line-by-line Python port of the self-collision narrow phase in `cpp/openral_safety_kernel/src/collision.cpp`. Kept in sync because the offline ACM sweep must reason about the same predicate the kernel runs — they previously diverged on box modelling (inscribed sphere vs. true OBB). Scope is `check_self_collision`'s link-vs-link routing only, not the voxel narrow phase; verified equal to 2.2e-16 against `mjcf_lowering._seg_seg_distance`.

- **`_seg_seg_distance` — two Python copies, one batched.** `kernel_predicates._seg_seg_distance` (vectorised over a batch) and `mjcf_lowering._seg_seg_distance` (scalar) are the same clamped-parametric segment solve (Ericson §5.1.9), split because the MJCF path evaluates one pose while the ACM certificate evaluates hundreds of thousands at once. Pinned equal to 2.2e-16 in `test_urdf_lowering_always_colliding`. Collapse all three if a third copy appears.

- **Convex distance — three implementations, three different questions. Collapsing them produces a green that confirms itself.** `collision.cpp`/`kernel_predicates` answer "what will the kernel do" (manifest OBBs/capsules, a lower bound by construction); `openral_hal.convex_distance` answers "where is the geometry actually" (MuJoCo's exact hulls, the evidence-path ground truth); `test_so101_base_box_collision._box_box` is a third box-SAT copy flagged for collapse onto `kernel_predicates`, not onto `convex_distance`. Routing the kernel check through the ground-truth module would compare the kernel to its own arithmetic.

- **Collision-model FK — deliberate C++/C++ mirror across a layer boundary, pinned by test.** `openral_octomap_bridge::self_forward_kinematics` + `build_self_model` + `self_transform_from_xyz_rpy` (`packages/openral_octomap_bridge/src/self_filter.cpp`, Layer 2) mirror the safety kernel's `forward_kinematics` / `load_collision_model` / `transform_from_xyz_rpy` (Layer 5) over the same `collision_*` parameters, so the real-camera self-filter removes returns exactly where the kernel places the robot. Linking the kernel at runtime would be a non-adjacent layer dependency (ADR-0110). `test_self_filter.cpp` links the kernel library (test-only) and requires every link frame to agree to 1e-12; a change to either side's joint convention, RPY order or primitive routing must land in both. Same for the tight-geometry narrow phase: `self_hull_distance` + the hull checks in `build_self_model` mirror the kernel's `dop_cell_lower_bound` / `hull_cell_distance` / `validate_tight_hull` for a point, pinned to 1e-9 on panda_mobile's hulls and to the kernel's constants by the same test.

- **26-DOP axis table + tight-geometry bounds — deliberate C++/Python mirror, update in lockstep.** `kDopAxis`/`kMaxTightHullVertices`/`kTightContainmentEpsilonM` (`cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`) vs. `DOP_AXES`/`MAX_TIGHT_HULL_VERTICES`/`TIGHT_CONTAINMENT_EPSILON_M` (`python/core/src/openral_core/schemas.py`). The axis table is positional — reordering it or flipping a sign on one side still validates but silently breaks the containment proof, over-reporting clearance. `generate_tight_geometry check` and `tests/unit/test_collision_tight_geometry.py` re-derive and re-prove it against the mesh.

- **The world-voxel grid derivation, in three places — left duplicated, pinned by test.** `deploy_e2e.launch.py::_world_voxel_max_cells` and `tools/voxel_transport_probe.py::per_axis` both derive `(2R/res + 1)^3` from the coverage radius. Neither can import the other — the launch file isn't an importable package, and the probe must run standalone. `test_deploy_e2e_voxel_resolution.py::test_the_transport_probe_sizes_the_grid_the_kernel_reserves` pins them together.

- **The quantisation budget, twice — left duplicated, pinned by test.** `tools/validation_matrix.py::quantization_budget_m` (canonical half body-diagonal) and `tools/stop_ee_speed.py::QUANTISATION_GAIN_M` (their 25/15 mm difference) are two standalone scripts with no shared module. The 8.66 mm figure is what the programme note §5 in `docs/reference/collision-validation-evidence.md` weighs the resolution trade against. `test_the_quantisation_gain_matches_the_matrix_budget_it_is_derived_from` pins it.

### Deliberately not consolidated

Repeated bodies that consolidation would make worse: different contracts, illegal imports, or a green that would confirm itself.

- **Three parallel registries** with the same lookup-by-string pattern: `openral_rskill.loader.rSkill` (file-backed JSON registry), `openral_sensors.catalog.SensorCatalog` (in-memory dict), `openral_sim.registry._Registry` (decorator-driven dict). Different lifecycle and value type, so deep consolidation isn't warranted — but the method names (`list_ids()`/`names()`/`list_installed()`) should align; a future ADR could standardise the verb.

- **SmolVLA skill-side `SmolVLAAdapter` vs eval-side `_SmolVLAAdapter` — not a duplication target.** Incompatible contracts: the skill takes `WorldState`/emits `Action` inside the ROS2 S1 runtime; the eval adapter takes a dict `Observation`/emits a flat numpy array in the sim driver. Collapsing would force Pydantic wrapping into the sim hot loop or widen `Skill.step()` to accept dicts. Residual overlap (~30 LOC/side) is below the abstraction-cost threshold.

- **`_build_libero_scene` / `_build_metaworld_scene` / `_build_mock_scene`** in `python/sim/src/openral_sim/{policies,backends}/{libero,metaworld,mock}.py` share the same lazy-import-instantiate-return shape. Already correctly DRY through the `SCENES.register(...)` decorator pattern; do not consolidate further.

- **`UsbDevice` / `UsbDeviceRecord` — not consolidated, deliberately different types.** `openral_cli.autodetect.UsbDevice` is a `NamedTuple` (hot in OS-probing loops); `openral_detect.report.UsbDeviceRecord` is a Pydantic `BaseModel` (the JSON/YAML report boundary, CLAUDE.md §2). Same fields, same reason to stay two types.

- **`camera_info_from_intrinsics` — not consolidated, illegal import.** `openral_hal.depth_cloud` and `openral_perception_ros.depth_convert` carry near-identical builders, but `openral_perception_ros/package.xml` doesn't depend on `openral_hal`, so the ROS package can't legally import the HAL's copy without a new dependency.

- **MJCF compile trio in three sim tests — four lines each.** `sim`/`_compiled`/`_model_data` in `test_sim_attachment_evidence.py`, `test_sim_estop_payload_slop.py`, `test_sim_estop_voxel_backing.py` — four lines; sharing needs a parameter at every call site.
- **`_wait_until` in two live tests — not hoisted.** `test_hal_attachment_barrier_live.py` vs. `test_estop_voxel_backing_live.py`; a seven-line spin-wait, hoisting costs a 21-call-site refactor.
- **`isolated_ros` fixtures — the domain is the difference.** `test_ros2_image_sensor_reader.py` (domain 91) vs. `tests/hil/test_openarm_ros_transport.py` (domain 92); the differing domain is the point.
- **Per-package ROS test clones — colcon isolation.** (`captured_spans`, `_spin_until`, `*_sigint_shape.py`, `slam_bringup` launch tests) — colcon builds each package standalone, so sharing needs a new shared package. The per-robot HAL lifecycle-test clones are gone: the twelve `openral_hal_<robot>` packages collapsed into one `openral_hal_node`, whose tests are parametrised by robot manifest.

- **`load_manifest_for_spec` — one copy left, on purpose.** Ten adapters call `policies/_policy_loading.load_manifest_for_spec`; `policies/act.py` keeps a private `_load_manifest_for_spec` (imported by `backends/libero.py`) that differs on empty `weights_uri` — the shared version returns `None`, act's raises `ROSConfigError`. Removing the duplicate is a behaviour change: decide the empty-URI contract first (CLAUDE.md §1.4 favours the loud version), then unify all eleven call sites.

- **`_connect` / `_rpc` in `locateanything_detector.py` / `qwen_scene_vlm.py` — not consolidated, no shared home.** AST-identical ZMQ REQ-socket bodies; the same shape also appears in `omdet_turbo_detector.py`, `sam2_segmenter.py`, and the reward backends. A two-file extraction would miss the real six-way duplication — leave as-is until a `ZmqSidecarClient` base takes all six at once.

- **`_find_metric` (`python/observability/tests/conftest.py` fixture vs. `tests/unit/test_runner_observability.py` module function) — left alone, no shared home.** Different installable-package test tiers, each with its own `conftest.py`; a shared helper would need a new top-level module, which the no-new-top-level-modules rule forbids.

- **`close` (`omdet_turbo_detector.py::OmDetTurboDetector` vs. `sam2_segmenter.py::Sam2Segmenter`) — left alone, deliberate keep.** Byte-identical six-line CUDA teardown, but the two classes share no other structure — a shared base for six lines would cost more to read than the duplication it removes.

- **`SensorSpec`-by-name search — two ROS packages, deliberately.** Private `_sensor_spec` in `packages/world_state/…/lifecycle_node.py` vs. public `sensor_spec_by_name` in `segmenter_node.py`. Two call sites in two packages; promoting it to `openral_core.schemas` is worth doing when a third caller appears, not before.

---

### Already correctly DRY (do not flag)

- **SimSensorBridge** — the single source for RGB camera publishing + MuJoCo viewer under `deploy sim`, via `openral_hal.sim_sensor_bridge.SimSensorBridge` and `_ManifestHALLifecycleNode`. `panda_mobile` included (its lifecycle node is manifest-driven, like every arm). Its two body-set resolvers answer different questions and are not a duplication target: `depth_cloud.robot_self_body_ids` ("what is the robot") vs. `sim_sensor_bridge.kernel_checked_body_ids` ("what does the safety kernel check", narrower by design). Do not add per-arm camera/viewer timers — extend `SimSensorBridge` instead.

- **Bounded certified-distance probing** — `sim_sensor_bridge._pair_distance_lower_bound` (bounding-sphere/plane prefilter) and `_round_robin_candidates` (fair exact-call budget) are the single source for measuring closest geom pairs without paying O(n·m). Shared by `_nearest_pair_records` (E-stop ground truth) and `_sim_attachment_evidence._probe_support_hits` (support-contact witness) — they need different output shapes from the same measurement, not different math. Both route through `openral_hal.convex_distance.convex_geom_distance`; neither may fall back to `mujoco.mj_geomDistance`, which misses geom pairs that `contype`/`conaffinity` suppress. Do not reimplement the prefilter or the budget; extract only the exact-call loop for a third output shape.

- **Bimanual real-HW fan-out** — `AlohaHAL.send_action` (14-DoF) and `OpenArmRealHAL.send_action` (16-DoF) both split one action across per-side arm+gripper controllers and publish four `joint_trajectory` messages, on different bases (`HALBase` vs. `RosControlHAL`). A third bimanual adapter is the trigger to lift the `(topic, slice, joint_names)` table into a shared mixin. Any consolidation must keep OpenArm's two properties: it builds all four messages before publishing any, and names `joint_names` per-message rather than relying on positional order.

- **HAL adapters (sim)** — `robots/<id>/robot.yaml` is the runtime source of truth. The pure-data arms (franka_panda, ur5e, ur10e, rizon4, openarm, anvil_openarm_v2, aloha_bimanual, so100/so101) set `hal.sim: null`, so `build_hal` derives `MujocoArmHAL.from_description(manifest)` from the manifest's `sim:` block — no per-robot Python class is involved, and a new MuJoCo robot needs no Python file (an `assets.mjcf` ref + `sim:` block in its `robot.yaml` suffices). `FrankaPandaHAL` / `UR5eHAL` / `UR10eHAL` / `Rizon4MujocoHAL` / `OpenArmMujocoHAL` / `AnvilOpenArmV2MujocoHAL` / `AlohaMujocoHAL` survive only as thin wrappers over their in-code `<ROBOT>_DESCRIPTION` mirror for direct construction in tests; no manifest names them. Only HALs with real behaviour keep a manifest entrypoint, and each takes `description=` so `build_hal` binds the loaded manifest: `G1MujocoHAL` (walking controller / kinematic glide), `H1MujocoHAL` (`_per_step_update` PD torque hook, an H1-specific cerebellar substitute), `PandaMobileHAL` (planar-base twin; joint layout read from the manifest). The `*_DESCRIPTION` constants are mirrors pinned to the manifests by `tests/unit/test_robot_manifests_match_hal_constants.py` (manifest wins).

- **Humanoid contract validators vs useful humanoid sims** — `H1MujocoHAL` and G1's default joint-position path are contract validators, not balance controllers: both fall without an S0 cerebellar controller (CLAUDE.md §6.2) and run joint-convergence tests with `gravity_enabled=False`. Do not bolt Python balance heuristics onto them — that crosses the S0 layer boundary. The sanctioned G1 exceptions are ADR-0087's kinematic-glide base (pinned-upright, zero-dynamics) and ADR-0089's upstream MuJoCo Playground ONNX policy (`walking_enabled=True`, sim-only). `H1MujocoHAL`'s software PD loop is a position-contract adapter for its torque actuators, not a balance controller.

- **Deliberate digital-twin gaps** — `Sawyer` and `GR1` intentionally ship without a MuJoCo HAL twin. Sawyer: Rethink Robotics is defunct and no real hardware will ever be plugged in, so it stays MetaWorld-only. GR1: no consumer yet — the natural second humanoid HAL once the C++ S0 cerebellum lands; today it exists only as an `openral_sim` rollout robot. These are documented absences, not missing work.

- **Real-HW manifest derivation** — every real-HW adapter derives its `*_REAL_DESCRIPTION` from a sim-side baseline via `openral_hal._real_description.make_real_description(base, sdk_kind=...)`, so kinematics/safety envelope/capabilities never drift between sim and real-HW siblings. `ur_real.py` uses it to derive `UR5e_REAL_DESCRIPTION`/`UR10e_REAL_DESCRIPTION`. New real-HW adapters must go through this helper rather than re-typing the `RobotDescription` constructor.

- **HAL adapters (real-HW)** — two shapes coexist on purpose. `UR5eRealHAL`/`UR10eRealHAL` (via a private `_URRealHAL` base), `FrankaPandaRealHAL` and `SawyerRealHAL` all **subclass** `RosControlHAL`, pinning their vendor's controller / topic / metadata defaults and overriding only the `_vendor_stop` / `_vendor_reset` hooks — a future UR variant is a one-line subclass. Franka and Sawyer used to *compose* a `RosControlHAL` behind a delegating wrapper that exposed none of the `RosControlDrivable` surface, so the lifecycle node never attached the production transport to either; composition is therefore **not** an acceptable shape for a ros2_control adapter. `AlohaHAL` inlines the publish machinery on `HALBase` because it fans one action across four controllers, which `RosControlHAL` doesn't support (and, per issue #250, a real ALOHA exposes no ros2_control surface at all); a second multi-controller adapter triggers a `MultiRosControlHAL`.

- **Downstream e-stop seams (issue #295)** — one contract, two implementations each, deliberately. `ControllerStopSeam` is implemented by the production `RosControlTransport` (real `controller_manager` services) and by `SimTransport` (an in-memory controller table with the same drop-when-inactive semantics), so the unit lane and the live graph run the identical `RosControlHAL.estop` / `reset_estop` code; `InterbotixStopSeam` is implemented by `InterbotixXSTransport` and `SimTorqueSeam` the same way. Do **not** add a per-robot stop path: a vendor's extra step goes in that adapter's `_vendor_stop` / `_vendor_reset` hook and is declared via `vendor_stop_services()` / `vendor_stop_topics()`, never as a new transport. The ros2_control stack helpers (`tests/integration/_ros2_control_stack.py`) were hoisted out of `tests/sim/test_openarm_hal_ros2_control.py` when the e-stop live test needed the same bring-up; both files import them.

- **HIL transport bridges (real-HW HALs)** — `RosControlHILTransport` (`tests/hil/_ros_control_transport.py`) is the source of truth for trajectory wiring; `AlohaHILTransport` (`tests/hil/_aloha_ros_transport.py`) reuses its `_make_trajectory_publisher` helper rather than duplicating `JointTrajectory`+QoS setup. Both share the joint-state caching shape (`_latest` dict, `state()` projection, `wait_for_first_state`). A third HIL bridge is the trigger to extract the subscriber half into a `_JointStateCache` mixin.

- **Kernel-twin sim tests** — the four `tests/sim/safety/test_kernel_with_<robot>_*.py` files (`so100_digital_twin`, `openarm_twin`, `rizon4_twin`, `h1_humanoid_twin`) all route through `tests/sim/safety/_kernel_subprocess.py::{start_kernel, activate_kernel_node, build_kernel_envelope, terminate_kernel}` and only declare their own joint names + action/state vectors. A fifth robot's kernel-twin test should call the same four helpers, not re-roll the lifecycle ceremony.

- **rSkillBase subclasses** — every `rSkillBase` subclass (`GpuPassthroughSkill`, the `ROSActionRskill` family) overrides the same five `_*_impl` hooks; the duplicated names are the `Skill` ABC contract, not redundancy. `GpuPassthroughSkill`'s `_step_impl` is the reference for a torch.cuda-based skill that must be explicit about device placement.

- **Runtime backends** — `NullRuntime`, `PyTorchRuntime`, `ONNXRuntime` (plus `TensorRTRuntime` in the private `openral-pro-trt` package) all implement the `Runtime` Protocol surface (`load`/`infer`/`quantize`/`warmup`/`unload`). Same situation as `Skill`.

- **`backends/so100_robosuite/`** — `_So100Lift` extends `robosuite`'s `Lift` env rather than reimplementing arena/reward/observable scaffolding, and the controller is the shipped `parts/osc_position.json` with three knobs overridden (`output_max`, `kp`, `input_ref_frame`), not a custom controller class. New robosuite-integrated robots should follow the same pattern: register the model in robosuite's factories, use a stock manipulation env subclass, and tune a stock part controller rather than writing a custom IK stack.

### Watch list (not yet a problem, but worth tracking)

- **Pinhole back-projection of a `32FC1` depth raster** now exists twice: `openral_hal.depth_cloud.points_from_depth_grid` (raster → `(N, 3)` cloud) and `openral_slam_bringup.depth_height_filter_node.filter_depth_by_global_height` (raster → filtered raster). Same `(u-cx)/fx` core, different outputs and packages. A third copy is the trigger to hoist a typed `deproject_depth(...)` into `openral_core.geometry`.

- **`_validate_action()`** appears in both `_mujoco_arm.py::MujocoArmHAL` and `ros_control.py::RosControlHAL`. They validate different invariants today (MuJoCo: `joint_targets` rank; ros2_control: control mode). A third HAL growing its own `_validate_action` is the trigger to lift the common part into a free function in `openral_hal.protocol`.

- **`_require_connected()`** appears in `_mujoco_arm.py::MujocoArmHAL`, `so100_follower.py::SO100FollowerHAL`, `ros_control.py::RosControlHAL`, and `aloha.py::AlohaHAL`. Four is over the threshold — a fifth copy is the trigger to hoist this into a base mixin (`openral_hal._lifecycle.RequireConnectedMixin`). `FrankaPandaRealHAL`/`SawyerRealHAL`/`_URRealHAL`/`OpenArmRealHAL` inherit `RosControlHAL`'s rather than duplicating it.

- **LeRobot SO-ARM unit + cadence conversions — consolidated** into `openral_sim/backends/_so_arm_units.py` (`steps_per_control_period`, `lerobot_action_to_radians`, `radians_to_lerobot_state`). Its one importer is `backends/so101_box/env.py` (the `so101_box` deploy scene and `so101_tube_insertion` sim scene). A new raw-MuJoCo scene accepting LeRobot-convention actions must route through this module. (`tabletop_push` keeps its own `_joint_scales` affine on purpose — it's robot-agnostic and can't assume the SO-ARM gripper convention.)

- **Rotation/quaternion math scattered across packages** — the yaw family is now consolidated into `openral_core.geometry` (`yaw_to_quat_xyzw`, `yaw_to_quat_wxyz`, `quat_xyzw_to_yaw`); five former copies route through them. Use these before adding another yaw↔quat helper. The remaining full-3-DOF conversions stay separate deliberately:
  - `rldx.py::_quat_wxyz_to_mat` — SAPIEN wxyz with its own norm-epsilon, pinned bit-identical to upstream; a calibration surface, not a duplicate.
  - `{mjcf,urdf}_lowering.py` rpy→matrix/euler — safety-kernel lowering; any change needs safety-WG review (CLAUDE.md §3).
  - `world_cloud_bridge.py::_quat_to_matrix` and `{bucket2_markers,depth_height_filter_node}.py::_rpy_to_*` — single-caller, package-specific types; worth a typed `openral_core.geometry` helper once a second caller appears.

- **`_TokenBucket` twice** — `openral_runner.backends.gstreamer.perception_tee.PerceptionEventPublisher._TokenBucket` and `openral_observability.failure_bus._TokenBucket` are the same rate limiter, implemented independently on purpose (the runner must not import the observability bus's ROS-facing module for a 20-line primitive). A third copy is the trigger to hoist it into `openral_core`.

- **`_canonicalize_quat_xyzw` (`openral_state_adapter.layouts.human300_16d`) vs `_canonicalize_quat_xyzw_np` (`openral_sim.backends.robocasa`)** — the sim copy is a float32 numpy mirror kept for byte parity between the deploy and benchmark paths (its docstring says so). Update in lockstep; do not merge unless the parity test moves with it.

- **`tools/rldx_sidecar.py` constants mirrored in `openral_sim._deps`** — same cross-venv wire-contract mirror as the BEHAVIOR sidecars above: the sidecar runs where `openral_sim` cannot be imported. Update both in one PR.

- **msgpack `_encode_ndarray` / `_decode_ndarray` in nine `tools/_*_server.py` sidecars** — structurally forced: each server runs in its own isolated venv and must not import the workspace. A shared file copied into each sidecar's venv at provisioning time is the only consolidation that would not break isolation; not done.

- **`_sim_attachment_evidence._tight_geometry_from_points` vs `openral_safety.tight_geometry.derive_tight_geometry`** — intentional online/offline twins of the DOP + hull refinement (same two stages, same `MAX_TIGHT_HULL_VERTICES` ceiling); the offline one is the certificate, the online one the evidence. Keep both, keep them equal.

- **`openral_human_estop.forwarder_node` QoS copy** — byte-identical to the watchdog package's `_qos` helpers, left in place because sharing needs a package edge (`openral_human_estop → openral_safety_watchdog` or `→ openral_observability`) that neither `package.xml` declares. Adding that edge is a decision, not a cleanup.

- **Image-subscription QoS in `openral_perception_ros`** (`ros_image_detector_node`, `scene_vlm_node`, `reward_monitor_node`, `segmenter_node`) — built inline four times with the same parameters. Same package, so a `_qos.py` there is the obvious next step when any of them is next touched.

- **Per-unit sensor values** — `openral_core.resolve_sensor_overlays` + `apply_sensor_overlays` are the one way a host's camera binding or a unit's calibrated mount/intrinsics reach a robot sensor (`robots/<id>/units/<unit>.yaml`, `$OPENRAL_ROBOT_UNIT` / `DeployScene.robot_unit`). Do not add per-host fields to `robot.yaml`, a scene-level copy of a robot sensor, or a second env var.

- **Robot manifest lookup** — `openral_sim.policies.robots.resolve_robot_manifest` is the one `$OPENRAL_ROBOTS_DIR` → `robots/<id>/robot.yaml` resolver, shared by `sim run` (the `ROBOTS` factories), `deploy sim|run` (`resolve_launch_invocation`) and `tools/audit_sim_configs.py`. The former per-robot `openral_cli.deploy_sim._ROBOT_HAL_REGISTRY` table is gone (its fields were derivable); do not reintroduce a robot-id → HAL table.

- **`openral_nav2_bringup._footprint_geometry`** is a private module reached from `tools/_nav2_costmap_silhouette_probe.py` and an integration test. Its public names (`convex_hull_2d`, `base_footprint_polygon`, `SHAPE_*`) are de-facto API; drop the module's underscore when it is next touched.

- **`openral_reasoner.spatial_query.ApproachRefiner`** — live (it types `refine_approach` on the two exported query functions), but `reasoner_node.py` passes its callback as `Any` with a comment instead of importing the alias. Import it.

- **`openral_rskill_ros.make_local_skill_resolver`** — exported, documented for `openral sim run`, and called only from inside its own module today. Public with no external consumer stays public; revisit if the docstring's use case never materialises.

- **`openral_safety.supervisor_node.SafetySupervisorNode`** — back-compat alias of `SafetyPassthroughNode` with zero references outside its own package's `__init__` docstring. Removing it touches `packages/openral_safety/` (safety-WG review + hazard log), so it is recorded here rather than deleted by a cleanup PR.

### Resolved in the 2026-09-22 cleanup (openral PR #296)

- **`_h1_group` / `_g1_group`** — one `_mujoco_arm._kinematic_group(joint_name, groups, *, robot)`; each robot keeps its group tuple.
- **Sidecar port derivation ×5** (`isaac_sim`/`robotwin`/`rlbench`/`lingbot_vla2`/`rldx`) — `_sidecar_common.sidecar_port_for_key`; pinned in `tests/unit/test_sidecar_common.py`.
- **`_CARTESIAN_KINDS` / `_GRIPPER_KINDS`** — byte-identical to `_CARTESIAN_MODES` / `_GRIPPER_MODES` in `schemas.py`; deleted, second validator uses the surviving pair.
- **Tegra host probe** (`openral_cli.deploy_sim._is_tegra_host`, `openral_runner.backends.gstreamer.pipeline._TEGRA_RELEASE_PATH`, `openral_detect.probes.gpu._DEFAULT_RELEASE_PATH`) — `openral_core.is_tegra_host()` / `TEGRA_RELEASE_PATH`. Do not re-probe `/etc/nv_tegra_release` elsewhere.
- **CameraInfo topic rule** — `openral_sensors.ros_publisher.camera_info_topic_for` resolves OpenRAL's own `/openral/cameras/<name>/(depth/)image` through `openral_core.camera_topic(name, CAMERA_INFO | DEPTH_CAMERA_INFO)`; its suffix rules only cover driver topics (a RealSense `ros2_topic`).
- **Torch-free GPU VRAM probe** (`openral_cli.deploy_sim._detect_gpu_vram_gb` / `openral_reasoner_ros.reasoner_node._query_gpu_gb`) — `openral_core.detect_gpu_vram_gb(field)` (ADR-0103).
- **Unified-memory-SoC VRAM fallback** in `openral_detect.probes.gpu` (`_probe_nvidia_pynvml` / `_probe_nvidia_smi`) — one `_unified_memory_vram_fallback` helper.
- **`_MIN_POLYGON_VERTICES`** — `payload_scan_filter_node` imports `_footprint_geometry`'s.
- **E-stop / failure `QoSProfile`s** in `deadman_watchdog_node` / `hardware_estop_node` — `openral_safety_watchdog._qos.estop_qos()` / `failure_qos()` (`openral_human_estop`'s copy stays; see Watch list).
- **Cross-package private imports promoted** — `openral_rskill.{hf_download_cached_first,gpu_allocated_mb,find_repo_root_from,validate_skill_ref}`; `openral_core.sensor_name_to_slot`; `tools/_robometer_scorer.Scorer`.
- **Dead code removed** — `packages/openral_foxglove_bringup/tools/demo_publisher.py` and `tools/_verify_lingbot_nf4.py` (no invocation site anywhere outside this inventory).
- **`openral_wam` removed** (ADR-0104) — `NullWorldModel`, `Rollout` and the `WorldModel` Protocol had no consumer in the open repo; the Protocol lives with its implementations in OpenRAL Pro.
- **`_CARTESIAN_MODES`-style bundled inventory bullets** — every `docs/methods` bullet now names one symbol with one marker; `refresh_methods_linenos.py --check --coverage` is now the primary duplication detector.

### Retired (resolved earlier; one line each)

- **Sensor `_spec()` private factory helpers — the seven vendor modules were deleted; one-per-file public spec factories remain** (was item 1).
- **VLA adapter boundary helpers — `resolve_device`/`resolve_rskill_repo_id`/`run_inference`/`to_numpy_action`/`parse_hf_file_uri`/`materialize_processor_dir`, once in `openral_rskill._vla_core`** (was item 3).
- **Policy load-phase heartbeat — `openral_rskill._diagnostics.phase_timer`, applied per adapter through a one-line `_<family>_phase` shortcut** (was item 6).
- **`_load_manifest_for_spec` — `backends/libero.py` imports `policies/act.py`'s copy** (was item 16).
- **`_coerce_sim_time_ns` / `_opt_num` — `sidecar.coerce_sim_time_ns` and `_sidecar_common.opt_num`** (was item 17).
- **`_env_bool` — `policies/rldx.py` imports `policies/gr00t.py`'s** (was item 18).
- **`_sensor_name_to_slot` — the ROS node imported the runner's private copy under an alias; now public as `openral_core.sensor_name_to_slot`, also replacing `SimSensorBridge`'s `_obs_key_for_sensor` and the runner node's `_vla_camera_slots` / `_required_vla_camera_slots` (now `openral_core.required_vla_camera_slots`)** (was item 19).
- **Test scaffolding (`_av`, `_find_metric`, `_zero_frame`, `_build_so101_hal`, 2 of 7 `_import_launch_module`) — tier `conftest.py` fixtures; 5 `_import_launch_module` copies remain** (was item 22).
- **Reward-monitor `assess()` — both monitors call `frame_source.assess_from_score`** (was item 23).
- **Test-tier fixture duplication — tier-wide fixtures live in `tests/unit/conftest.py` / `tests/sim/conftest.py`** (was item 24).
- **`disconnect()` / `_floats` in `AlohaHAL` / `RosControlHAL` — `HALBase.disconnect` default and `_base._raw_floats`** (was item 27).
- **Sidecar scene/socket duplication — `sidecar.SidecarSimRollout`, `sidecar.open_req_socket`, `rollout.render_named_rgb_mujoco`** (was item 29).
- **`connected_hal` leftover shadows in two sim HAL tests — deleted, conftest fixture used** (was item 30).
- **HIL transport `state` / `_on_joint_state` / `wait_for_first_state` — `_JointStateCache` + `_PolledJointStateMixin`** (was item 31).
- **Safety-kernel place-\* live test harness closures — three factory fixtures in `tests/integration/conftest.py`** (was item 32).
- **`test_disconnect_idempotent` per real HAL — already covered by `test_hal_protocol_conformance`** (was item 33).
- **`test_after_estop_send_action_fails` — parametrized in `test_hal_protocol_conformance`** (was item 34).
- **`test_manifest_has_latency_budget` ×3 — `tests/sim/conftest.py::assert_manifest_has_latency_budget`** (was item 35).
- **`test_send_action_holds_zero_pose` / `test_hold_zero_pose` — `tests/sim/conftest.py::assert_send_action_holds_zero_pose`** (was item 36).
- **`_expand` PEP 735 walker ×2 — `tests/unit/conftest.py::expand_dependency_group`** (was item 37).
- **`_CaptureProcessor` shadow in `test_reasoner_core.py` — deleted, conftest one imported** (was item 38).
- **`_connect` / `_rpc` / `_try_ping` (`LocateAnythingDetector` / `QwenSceneVlm`) — `backends/gstreamer/_zmq_sidecar.ZmqSidecarMixin`** (was item 40).
- **Policy adapter loader seams** — `policies/_policy_loading.load_manifest_for_spec` / `lazy_import_lerobot` and the `_quantization` dtype helpers; see the DRY section.
- **`NDArrayOrNone` alias** — `look_at_rskill` imports `pose_goal_rskill`'s.
- **`from_yaml(cls, path)` ×6** — `openral_core.schemas._load_yaml_model`; `SimScene` / `BenchmarkScene` inherit `DeployScene.from_yaml`.
- **`_resolve_cameras` in three perception nodes** — `openral_perception_ros.camera_topics.resolve_camera_topics`.
- **Hand-built `/openral/cameras/<name>/<kind>` strings at ~40 sites** (sim bridge, sensor leg, world-state `camera_topic_prefix`, deploy launch, Foxglove layout + allowlist, vision-attachment bridge, record profiles, perception/SLAM node defaults) — `openral_core.camera_topic` / `CAMERA_TOPIC_PREFIX` (ADR-0108); `tests/unit/test_camera_topic_layout.py` fails on a new one.
- **`homogeneous_from_quat_xyz`** — moved from `openral_world_state.object_lift` to `openral_core.geometry`; world-state keeps a thin wrapper.

---

*Curated by hand since 2026-05-08. Regenerate the inventory files with `tools/refresh_methods_linenos.py`; this file is prose and is edited directly.*
