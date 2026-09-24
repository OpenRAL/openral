# Layer 0 — Core Schemas & Exceptions

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Authoritative Pydantic v2 contracts. Anything imported from `openral_core.__init__` is API. On-disk schemas (`RobotDescription`, `RSkillManifest`, `SceneGraph`, …) are versioned via `schema_version` and evolve in place unless a change breaks compatibility.

### `python/core/src/openral_core/schemas.py`
_openral schema v0 — normative Pydantic v2 contracts for all layers._

**Enums**

- `class EmbodimentKind(str, Enum)` — Top-level kinematic class. (L64)
  `HUMANOID, MANIPULATOR, BIMANUAL, QUADRUPED, MOBILE_BASE, MOBILE_MANIPULATOR, DRONE`
- `class JointType(str, Enum)` — URDF joint type. (L76)
  `REVOLUTE, PRISMATIC, CONTINUOUS, FIXED, FLOATING, PLANAR`
- `class ClockOrigin(str, Enum)` — Authoritative source for `stamp_ns` values; `/clock` is a projection, not an origin. (L87)
  `HOST_WALL, SIMULATION, HARDWARE_SYNCED`
- `class ClockEpoch(str, Enum)` — Epoch a `stamp_ns` value is measured from. (L99)
  `UNIX, SIMULATION_ELAPSED, HARDWARE`
- `JointRole: TypeAlias = Literal[…]` — Structural role of a `JointSpec` (arm/base/gripper/…), read instead of name-substring matching; defaults to `"unknown"` so legacy manifests stay loadable. (L107)
  `"arm", "base", "gripper", "torso", "leg", "head", "neck", "wheel", "unknown"`. Used by runner/safety/dataset-bridge to identify a channel without name-substring heuristics. Default `"unknown"` keeps legacy manifests loadable.
- `class ControlMode(str, Enum)` — Action space / control interface. (L211)
  `JOINT_POSITION, JOINT_VELOCITY, JOINT_TORQUE, JOINT_TRAJECTORY, CARTESIAN_POSE, CARTESIAN_DELTA, CARTESIAN_TWIST, BODY_TWIST, FOOT_PLACEMENT, GRIPPER_BINARY, GRIPPER_POSITION, DEX_HAND_JOINT, COMPOSITE_MODE`
- `CONTROL_MODE_TO_UINT8: dict[ControlMode, int]` — Wire encoding for `ActionChunk.control_mode`; mirrors the C++ kernel's `validator.hpp::ControlMode` enum. (L241)
- `UINT8_TO_CONTROL_MODE: dict[int, ControlMode]` — Inverse of `CONTROL_MODE_TO_UINT8`. (L259)
- `const BODY_TWIST_DIM: int = 6` — Width of a BODY_TWIST / CARTESIAN_* twist row `(vx, vy, vz, wx, wy, wz)`; shared by the HAL packers and safety supervisor that validate 6-vec twist payloads. (L264)
- `class SensorModality(str, Enum)` — Physical sensing modality. (L277)
  `RGB, DEPTH, STEREO, IR, POINT_CLOUD, LIDAR_2D, IMU, FORCE_TORQUE, JOINT_STATE, TACTILE_VISION, TACTILE_ARRAY, AUDIO, GPS, BATTERY`
- `class Hand(str, Enum)` — End-effector laterality. (L296) `LEFT, RIGHT, NA`
- `class StateRepresentation(str, Enum)` — State vector format. (L1191)
  `JOINT_POSITIONS, EEF_POS_AXISANGLE, EEF_POS_EULER, EEF_POS_QUAT, EEF_POS_AXISANGLE_GRIPPER`
- `class ActionRepresentation(str, Enum)` — Action vector format: which coordinates and gripper encoding a policy's output represents. (L1201)
- `class JointUnits(str, Enum)` — Angular convention a joint-position checkpoint was trained in. The runner converts deg↔rad at the policy boundary; a joint-position skill reaching the runner without a declaration raises `ROSConfigError` rather than guessing. (L1216)
  `JOINT_POSITIONS, JOINT_VELOCITIES, DELTA_EE_6D_PLUS_GRIPPER, DELTA_EE_6D, CARTESIAN_POSE`
- `class RSkillAction(str, Enum)` — Closed vocabulary of action verbs an rSkill can perform; declared on `RSkillManifest.actions` and surfaced to the reasoner's tool palette so it can pick a skill by what it does. (L1234)
  Manipulation primitives: `PICK, PLACE, PICK_AND_PLACE, TRANSFER, GRASP, RELEASE`; articulated / contact-rich: `OPEN, CLOSE, PUSH, PULL, SLIDE, INSERT, POUR, WIPE, ROTATE`; motion: `REACH`; mobile: `NAVIGATE`; social/expressive: `WAVE, SHAKE`; generalist marker (foundation / multi-task checkpoints): `GENERALIST`; perception producer: `DETECT` (for `kind: "detector"` rSkills); scene VLM: `QUERY` (for `kind: "vlm"` rSkills); reward monitor: `MONITOR` (for `kind: "reward"` rSkills); playbook decision procedure: `PLAN` (for `kind: "playbook"` rSkills). New entries are additive.
- `class QuantizationDtype(str, Enum)` — Weight numeric format. (L4381)
  `FP32, FP16, BF16, INT8, INT4, FP4_NVFP4`
- `class QuantizationBackend(str, Enum)` — Inference backend. (L4415)
  `PYTORCH, ONNX, TENSORRT, GGUF, MLX`
- `class RSkillState(str, Enum)` — Skill lifecycle. (L4487)
  `UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED, ERROR`
- `class RSkillLicensePosture(str, Enum)` — License posture. (L4567)
  `APACHE_2_0, MIT, BSD, PERMISSIVE_RESEARCH, NVIDIA_NON_COMMERCIAL, NVIDIA_OPEN_MODEL, RLWRLD_NON_COMMERCIAL, PROPRIETARY, UNKNOWN` (NVIDIA_OPEN_MODEL = GR00T N1.7+, commercial OK)
- `class RSkillRuntime(str, Enum)` — Manifest runtime hint. (L4582)
  `PYTORCH, ONNX, TENSORRT, TRT_LLM, VLLM, GGUF, MLX, JAX`
- `class PhysicsBackend(str, Enum)` — Sim backend. (L9114)
  `MUJOCO, MUJOCO_MJX, PYBULLET, SAPIEN, ISAACSIM, COPPELIASIM, GENESIS, MOCK` (SAPIEN = ManiSkill3 / RoboTwin engine; RoboTwin uses it via a py3.10 sidecar; `COPPELIASIM` = CoppeliaSim/PyRep RLBench backend, out-of-process py3.10 sidecar)

**Pydantic models — robot manifest hierarchy**

- `class IntrinsicsPinhole(BaseModel)` — Pinhole camera intrinsics. (L307)
  fields: `width, height, fx, fy, cx, cy, distortion_model, distortion_coeffs`
- `scale_intrinsics_to(base, width, height) -> IntrinsicsPinhole` — Rescales pinhole intrinsics to a new render resolution, preserving FOV and distortion; used when deploy-sim renders at a different resolution than the manifest declares. (L335)
- `class ClockAuthority(BaseModel)` — Named origin for OpenRAL timestamps across sim, rSkills, ROS, and hardware; construct via `host_wall()`, `simulation()`, or `hardware_synced()`. A validator rejects impossible origin/epoch pairs and competing hardware `/clock` publishers. (L131)
  - `host_wall(cls) -> ClockAuthority` [@classmethod] — Real-deployment wall/ROS system time origin. (L163)
  - `simulation(cls, clock_id, timestep_s=None, publishes_ros_clock=True) -> ClockAuthority` [@classmethod] — Simulator elapsed time projected to `/clock`. (L168)
  - `hardware_synced(cls, clock_id, epoch=UNIX) -> ClockAuthority` [@classmethod] — Future synchronized controller/PTP clock origin. (L185)
- `class CameraSimPlacement(BaseModel)` — Where an RGB sensor's camera sits in the sim MJCF, so the generic HAL camera rig can splice it into a bare-arm MJCF for `deploy sim`. Replaces per-robot `scene_defaults.composition` for camera-only deploy twins. (L389)
- `class SensorSpec(BaseModel)` — Generalizable sensor descriptor covering all modalities; `sim_camera_name` names the MJCF camera when it differs from the sensor `name`, `sim_placement` carries the camera's sim pose, and `deploy_binding` is the real-hardware counterpart (host-specific, filled by `openral detect`; a robot camera's lives in robot.yaml only, a workcell camera's on its own `DeployScene.sensors` entry). `ros2_topic` names only an *external* driver's image topic (ADR-0108); `None` (every in-tree manifest) means OpenRAL publishes the sensor under `camera_topic(name, kind)`. (L433)
  - `is_depth_camera` [@property] (L538) — A depth / point-cloud camera with intrinsics: the one test for "can this be back-projected into a cloud", shared by the sim bridge's depth synth (`openral_hal.depth_cloud.is_depth_sensor` delegates to it), the sim cloud topic (`deploy_cloud_topic`), the Nav2-over-visual-SLAM guard and the launch's depth-camera pick.
  - `is_cloud_source` [@property] (L553) — Any `depth` or `point_cloud` sensor (a 3D lidar needs no intrinsics): a real driver can feed octomap its cloud, so on `deploy run` it auto-enables the octomap leg, which then needs the driver topic pinned.
  fields: `name, modality, frame_id, parent_frame, static_transform_xyz_rpy, rate_hz, intrinsics, encoding, fov_h_deg, fov_v_deg, sim_camera_name, sim_placement, n_channels, range_min_m, range_max_m, accel_noise_density, gyro_noise_density, n_axes, tactile_grid, vla_feature_key, ros2_topic, ros2_msg_type, qos_profile, catalog_id, vendor, model, driver_pkg, metadata`
- `CAMERA_TOPIC_PREFIX: Final[str]` — `"/openral/cameras"`, the root of the canonical camera topic layout (ADR-0108). Consumers that need a pattern use `re.escape(CAMERA_TOPIC_PREFIX)`. (L572)
- `class CameraTopicKind(StrEnum)` — Per-camera stream suffix: `IMAGE` (`image`), `CAMERA_INFO`, `DEPTH_IMAGE` (`depth/image`), `DEPTH_CAMERA_INFO` (`depth/camera_info`), `POINTS`. (L575)
- `camera_topic(name, kind=CameraTopicKind.IMAGE, *, prefix=CAMERA_TOPIC_PREFIX) -> str` — `<prefix>/<name>/<kind>`; the only place the camera layout is spelled. Every producer (sim bridge, sensor leg) and consumer (world state, perception, reasoner, Foxglove, SLAM, deploy launch, record profiles) builds through it; `tests/unit/test_camera_topic_layout.py` rejects a hand-built string. Raises `ROSConfigError` on an empty or `/`-containing name. (L585)
- `class SensorBundle(BaseModel)` — Multi-modal sensor group. (L620)
- `sensor_name_to_slot(description) -> dict[str, str]` — Maps each RGB sensor name to its VLA slot (`vla_feature_key` suffix, else the name). The one camera namespace shared by `openral sim run`, `deploy sim` and `deploy run`; used by `rskill_runner_node`, `DatasetRecorderBridge`, `SimSensorBridge` and `SimRunner`. (L636)
- `check_scene_sensor_overrides(manifest_sensors, scene_sensors) -> None` (L681) — Raises `ROSConfigError` when a `DeployScene.sensors` entry reuses the name of a sensor the robot manifest defines. A deploy scene never touches a robot camera: its geometry, frame, intrinsics and real-hardware `deploy_binding` live in `robots/<robot_id>/robot.yaml`; the scene only adds workcell cameras under new names. Called by `merge_deploy_sensors`, `resolve_launch_invocation` (so `deploy sim/run/validate`) and `openral check`.
- `merge_deploy_sensors(manifest_sensors, scene_sensors) -> list[SensorSpec]` (L714) — The deploy's sensor set: the manifest's sensors, then the scene's workcell sensors; a checked concatenation (`check_scene_sensor_overrides` refuses a name clash). Shared by the CLI, the launch and `compose_runtime`.
- `publishing_sensors(manifest_sensors, scene_sensors, hal_mode) -> list[SensorSpec]` (L737) — The sensors that actually publish a camera topic on a deploy: sim → the manifest's (the sim bridge renders them); real → `merge_deploy_sensors` filtered to sensors with a `deploy_binding`. The one rule `openral deploy` (detector auto-downgrade) and `deploy_e2e.launch.py` (completion, detector, reward-monitor and scene-VLM cameras) share.
- `deploy_cloud_topic(manifest_sensors, *, pinned, hal_mode) -> str` (L763) — The `PointCloud2` topic octomap's `cloud_in` and the world-state object-lift fallback read: a pinned `runtime.octomap_cloud_topic` wins; sim → `camera_topic(<the one is_depth_camera>, POINTS)` (the sim bridge's back-projection); real → `""` (nothing in-tree publishes a cloud). Raises `ROSConfigError` in sim for several depth cameras and no pin (declared choice, not first-wins). Shared by `openral deploy`'s pre-launch refusal and `deploy_e2e.launch.py`.
- `required_vla_camera_slots(manifest, description) -> tuple[str, ...]` — The robot's RGB slots trimmed to those the rSkill's `sensors_required[].vla_feature_key` names (all slots when it names none); the policy camera keys on deploy and on an empty-`scene.cameras` sim run. (L660)
  fields: `bundle_name, sensors, sync, sync_tolerance_ms`
- `class JointSpec(BaseModel)` — URDF-derived joint spec; `origin_xyz`/`origin_rpy`/`axis_xyz` let the kernel compute FK for self-collision, `role` identifies gripper/base/arm DoFs structurally instead of by name-substring, and `sim_joint_name` names the MJCF joint when it differs from the logical `name` (needed only when a sim adapter looks the joint up by name and the loaded MJCF renames it). (L810)
  fields: `name, joint_type, parent_link, child_link, axis_xyz, origin_xyz, origin_rpy, position_limits, velocity_limit, effort_limit, has_position_sensor, has_velocity_sensor, has_torque_sensor, backlash_estimate, actuator_kind, sim_joint_name, role`
- `class EndEffectorSpec(BaseModel)` — End-effector spec; `actuated=False` marks a passive tool (inert flange, kinematic-only mount) so the safety kernel rejects chunks addressed at it. (L892)
  fields: `name, kind, hand, n_dof, max_grip_force_n, max_payload_kg, workspace_radius_m, tactile_sensors, actuated`
- `class ComputeSpec(BaseModel)` — Compute profile for one deployment tier (edge/local/cloud), populated by `openral_detect._enrich_compute` from GPU probe results and attached to `RobotDescription.compute_edge/local/cloud`. (L952)
  fields: `compute_tops, system_memory_gb, num_gpus, gpu_vram_gb, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, gpu_supported_runtimes, gpu_supported_dtypes, nvmm_available, endpoint, network_latency_ms`
  - `supports_cumotion() -> bool` — True when the host meets the cuMotion (Isaac ROS) GPU floor on compute capability, CUDA toolkit version, and VRAM; used by the MoveIt planner gate to pick cuMotion vs OMPL. (L1019)
- `prop _CUMOTION_MIN_COMPUTE_CAPABILITY, _CUMOTION_MIN_CUDA_MAJOR, _CUMOTION_MIN_VRAM_GIB` — The cuMotion GPU floor `supports_cumotion()` checks against. (L934–936)
- `ReasonerDialect = Literal["anthropic", "openai"]` — Wire dialect a `ReasonerModel` / named endpoint speaks; only needed on `OPENRAL_REASONER_DIALECT` for a bare, unclassified URL. (L10723)
- `ReasonerHosting = Literal["cloud", "managed_local", "byo_local"]` — Where a `ReasonerModel` runs; drives `ReasonerModel.is_local`. (L10726)
- `class ReasonerModel(BaseModel)` — Frozen curated S2 model registry entry. `REASONER_MODELS` is the curated map; membership means the model passed the robotics tool-calling contract. (L10736)
  - `is_local(self) -> bool` [@property] — True for `managed_local` / `byo_local` hosting (needs local compute); False for `cloud`. (L10805)
- `REASONER_MANAGED_ENDPOINT: str = "managed"` — Sentinel for `ReasonerModel.default_endpoint` meaning OpenRAL spawns and manages the local server, resolved to the model's managed loopback endpoint at client-build time. (L10733)
- `_OPENROUTER_ENDPOINT: str` — OpenRouter's OpenAI-compatible endpoint URL; default gateway for the curated GPT-5.x entries. (L10812)
- `REASONER_MODELS: dict[str, ReasonerModel]` — The curated model registry keyed by `id`; adding a model means adding one entry here after it clears the tool-calling bar. (L10816)
- `class ReasonerEndpointPreset(NamedTuple)` — Everything a named `OPENRAL_REASONER_ENDPOINT` implies beyond its URL (dialect, auth, cold-start timeout, tool_choice). (L10885)
- `REASONER_ENDPOINT_PRESETS: dict[str, ReasonerEndpointPreset]` — Presets for the accepted `OPENRAL_REASONER_ENDPOINT` names. (L10919)
- `prop ANTHROPIC_BASE_URL, OPENROUTER_BASE_URL, OLLAMA_BASE_URL, VLLM_BASE_URL, GEMINI_BASE_URL, XAI_BASE_URL, DEEPSEEK_BASE_URL, HUGGINGFACE_BASE_URL` — Base URLs backing the named `OPENRAL_REASONER_ENDPOINT` presets. (L10865–10882)
- `class RobotCapabilities(BaseModel)` — Physical capability flags for skill compatibility; `has_vision_slam` gates the camera-based SLAM backend for lidar-less robots, independent of `has_lidar` (lidar backend wins when both set). (L1045)
  fields: `locomotion, can_lift_kg, has_dexterous_hands, has_tactile, has_force_control, has_vision, has_lidar, has_vision_slam, has_audio, bimanual, supported_control_modes, supported_vla_embodiments, embodiment_tags`
- `class SafetyEnvelope(BaseModel)` — Constraints enforced by the C++ safety kernel; `self_collision_margin_m` can go negative to tolerate a compact arm's known in-distribution grazing contact while gross folds still trip. (L1093)
  fields: `workspace_box_min_xyz, workspace_box_max_xyz, no_go_zones, max_ee_speed_m_s, max_ee_accel_m_s2, max_joint_speed_factor, max_force_n, max_torque_nm, deadman_required, e_stop_topic, e_stop_qos, contact_force_threshold_n, cycle_time_violation_threshold_ms, human_in_loop_required` + per-mode bounds (`max_cartesian_step_m/_rad`, `max_ee_angular_speed_rad_s`, `max_base_linear/angular_speed_rad_s`) + `self_collision_margin_m: float = 0.0`
- `class ObservationSpec(BaseModel)` — VLA observation config. (L1292)
  fields: `state_key, state_shape, state_representation, image_flip_180` (`image_flip_180` deprecated, no effect on image processing — setting it emits a `FutureWarning`; use `RSkillManifest.image_preprocessing.flip_180`)
- `class ActionSpec(BaseModel)` — VLA action config. (L1334) `dim` is optional (issue #303): a robot with no committed policy action contract declares only `control_freq_hz`, and the dataset recorder takes the width from the action itself.
  fields: `dim, representation, control_freq_hz, chunk_size`
- `class ActionSlot(BaseModel)` — One contiguous slice of an rSkill's action vector, with per-mode field requirements enforced by a validator; `discard=True` drops a slice silently. (L4953)
  fields: `range, control_mode, discard, ee, frame, joint_names, input_bounds`
- `class ActionContract(BaseModel)` — Per-rSkill action-vector contract; when `slots` is set, every index is covered by exactly one `ActionSlot`, else the legacy single-Action JOINT_POSITION path applies. (L5099)
  fields: `dim, representation, slots, cartesian_delta_scale, joint_names, gripper_scale, joint_units` — `joint_names` (policy joint order in robot joint names; forbidden with `slots`, must match `dim`, unique) and `gripper_scale` (policy gripper units per HAL `[0, 1]` unit, `> 0`) are consumed by `openral_rskill._policy_io.PolicyIOCodec` on every dispatch path.
- `prop _JOINT_MODES, _CARTESIAN_MODES, _GRIPPER_MODES` — The `ControlMode` partitions used by `ActionSlot`'s and `ControlModeSemantics`'s per-mode validators. (L5088–5094)
- `prop _EE_6D_WIDTH, _EE_3D_WIDTH` — Cartesian slice widths `canonical_slots_for_representation` uses to size a representation-only `ActionContract`'s slots. (L5256–5257)
- `class TaskSpaceFamily(str, Enum)` — Coarse classification of a `ControlMode` for task-space views (DRAFT). (L5455)
- `_FAMILY_FOR_MODE: dict[ControlMode, TaskSpaceFamily]` — Exhaustive mapping from every `ControlMode` to its `TaskSpaceFamily`, lockstep-tested. (L5475)
- `class TaskSpaceSegment(BaseModel)` — One typed slice of an action vector, layer-neutral; the gripper is an explicit 1-D segment. (L5492)
  fields: `family, control_mode, width (>0), target`
- `class TaskSpaceMatch(BaseModel)` — Result of `task_space_compatible`. (L5534)
  fields: `ok, reasons`
- `class TaskSpace(BaseModel)` — Layer-neutral view of an action interface as ordered `TaskSpaceSegment`s (DRAFT); derived via `from_action_contract`, never hand-authored, so it cannot drift from the primitives. (L5549)
  - `total_dim(self) -> int` [@property] — Sum of segment widths — the flat action-vector dimensionality. (L5600)
  - `control_modes(self) -> set[ControlMode]` [@property] — The distinct `ControlMode`s this space drives. (L5605)
  - `from_action_contract(cls, action: ActionContract, robot: RobotDescription) -> TaskSpace` [@classmethod] — Builds the task space an rSkill emits, expanding `slots` / `representation`. (L5610)
- `class SceneTaskSpace(BaseModel)` — The control interface a scene-adapter family executes, declared once per family in `SCENE_FAMILY_TASK_SPACE`; `runs_via_default_packers=False` marks dedicated-controller adapters. (L5826)
  fields: `modes (frozenset[ControlMode]), action_dim (int|None), runs_via_default_packers (bool)`
- `class SphereShape(BaseModel)` — Sphere collision primitive; discriminator `shape="sphere"`, field `radius_m (>0)`. (L1687)
- `class CapsuleShape(BaseModel)` — Capsule collision primitive (segment along local +Z swept by a radius); discriminator `shape="capsule"`, fields `radius_m (>0), length_m (>=0)`. (L1705)
- `class BoxShape(BaseModel)` — Oriented box (OBB) collision primitive for blocky links; fits a near-cubic link far tighter than a capsule, whose round section over-reports clearance past flat faces. (L1734)
- `CollisionShape: TypeAlias = Annotated[CapsuleShape | SphereShape | BoxShape, Field(discriminator="shape")]` — Discriminated union of convex collision primitives; mesh shapes excluded so the allocation-free kernel checks only analytic convex volumes. The `shape` discriminator is enforced, not just documented, so a bad tag fails clearly rather than silently resolving to the wrong member. Never dump with `exclude_defaults=True` — it drops the tag field. (L1765)
- `DOP_AXES: tuple[tuple[float, float, float], ...]` — The 26-DOP's 13 unit axis directions, in the owning box's own local frame; mirrors `kDopAxis` in the C++ safety kernel. Its first three entries are the box's own axes, which is load-bearing — checking those three proves containment for the whole polytope without enumerating a vertex. (L1787)
- `MAX_TIGHT_HULL_VERTICES: int` — `320`; ceiling on a stage-2 hull's vertex count, mirroring `kMaxTightHullVertices` — a cost bound, not a safety one, since a link over the ceiling still runs the DOP-only stage 1. (L1816)
- `TIGHT_CONTAINMENT_EPSILON_M: float` — `1e-9` m of floating-point slack when checking a hull vertex against its own DOP slab; mirrors `kTightContainmentEpsilonM`. The box-containment check on `LinkCollisionGeometry` takes no slack at all. (L1826)
- `class TightCollisionGeometry(BaseModel)` — Tight convex geometry refining one link's `BoxShape`, used by the safety kernel for the arm-link-vs-world-voxel check only; drives a staged narrow phase (26-DOP bound, then GJK on the exact hull). Generated by `tools/generate_tight_geometry.py`, not hand-authored. (L1836)
- `check_tight_geometry_fits_box(shape, tight, owner) -> None` — Raises unless `tight` refines a `BoxShape` and stays inside its `half_extents_m` — the containment check that must not be deferred to the kernel, since a representation reaching outside the box would make the kernel skip cells it should visit. Shared by `LinkCollisionGeometry` and `AttachedCollisionPrimitive` so neither is held to a weaker standard. (L1940)
- `class LinkCollisionGeometry(BaseModel)` — One convex collision volume attached to a robot link; hand-authored or emitted by the offline lowering tool from MJCF/URDF. (L1995)
  fields: `link_name, shape: CollisionShape, origin_xyz_rpy, tight_geometry: TightCollisionGeometry | None`
  - `_tight_geometry_fits_inside_the_box(self) -> Self` [@model_validator(mode="after")] — Rejects `tight_geometry` on any non-`BoxShape`, or a DOP whose slabs reach outside `half_extents_m`; no slack is allowed here. (L2042)
- `class FixedAttachment(BaseModel)` — One rigid, zero-DoF parent→child link of the kinematic tree, completing the connectivity `joints` alone cannot express (a bolted-on hand or bimanual pedestal would otherwise present as disconnected trees). Origins must come from the real URDF/MJCF, never estimated. (L2051)
  fields: `name, parent_link, child_link, origin_xyz, origin_rpy`
- `class RobotDescription(BaseModel)` — Top-level robot manifest, one per robot; `assets` is the single URDF/MJCF/SRDF reference block, `compute_edge/local/cloud` hold per-tier compute profiles, `collision_geometry`/`allowed_collision_pairs` carry the safety-kernel-facing self-collision data, and `fixed_attachments` completes the kinematic tree with rigid zero-DoF mounts that `joints` alone can't express. (L2212)
  fields: `name, embodiment_kind, assets, base_frame, odom_frame, map_frame, joints, end_effectors, sensors, sensor_bundles, capabilities, safety, ros2_namespace, middleware, onboard_compute, sdk_kind, hal, observation_spec, action_spec, sim, scene_defaults, base_joints, footprint_radius, base_kinematics, collision_geometry, allowed_collision_pairs, fixed_attachments, footprint_polygon, compute_edge, compute_local, compute_cloud, schema_version`
  - `scene_defaults: SceneDefaults | None = None` — Optional scene-level defaults (top-camera POV, etc.) consumed by the MJCF composers as the fallback when an environment does not pin its own values.
  - `validate_for_e2e_pipeline(self) -> None` — Asserts every actuated joint has `position_limits`/`velocity_limit`/`effort_limit` set, so a misshapen manifest fails at launch-parse time rather than mid-actuation. (L2599)
  - `lidar_sensor(self) -> SensorSpec | None` [@property] — First declared `lidar_2d` sensor, or None; single source of truth for the synthetic `/scan` envelope. (L2481)
  - `nav2_footprint_param(self) -> str` — This base's Nav2 `footprint` parameter string from `footprint_polygon`; `"[]"` when undeclared so a radius-only robot keeps its own circular footprint instead of inheriting the shared param file's shape. (L2497)
  - `nav2_param_overrides(self) -> dict[str, str]` — Nav2 param substitutions derived from `footprint_radius` and `base_kinematics`, so one shared base param file serves any mobile base; `{}` for fixed-base arms. (L2523)
  - `control_rate_hz(self) -> float | None` [@property] (L2356) — `action_spec.control_freq_hz` when positive, else `None`. The one place a robot's rate lives: the skill runner ticks at it, the HAL node publishes proprio at it, the real ros2_control HAL derives every trajectory deadline from it, and the recorder stamps it as fps (issue #303).
  - `REAL_HARDWARE_SAFETY_FIELDS` [ClassVar] — the `SafetyEnvelope` fields the kernel's envelope loader reads and that carry a schema default (`max_ee_speed_m_s`, `max_ee_accel_m_s2`, `max_joint_speed_factor`, `max_force_n`, `max_torque_nm`, `contact_force_threshold_n`, `deadman_required`, `self_collision_margin_m`).
  - `_validate_real_hardware_contract(self) -> RobotDescription` [model_validator] (L2373) — a manifest with `hal.real` set must declare `control_rate_hz`, every `REAL_HARDWARE_SAFETY_FIELDS` entry explicitly in `safety` (an explicit value equal to the default is fine; inheriting it silently is not), and `safety.starting_pose_max_joint_speed_rad_s` / `safety.starting_pose_tolerance_rad` declared (the runner's approach to a starting pose has no default speed; deriving one from rated `velocity_limit` put the OpenArm at 2 rad/s). Every gap is reported in one error, at load time — before `deploy validate`, `build_hal` or any node. Sim-only manifests keep the schema defaults.
  - `from_yaml(cls, path: str) -> RobotDescription` [@classmethod] — Load and validate a `RobotDescription` YAML manifest from disk. (L2581)
- `def extract_base_sim_joint_names(description: RobotDescription) -> tuple[str, str, str] | None` — Returns `(forward, side, yaw)` MJCF joint names for any mobile-base description declaring both `base_joints` and `sim_joint_name`. (L2644)
- `const NAV2_INFLATION_CLEARANCE_M: float = 0.05` — Costmap inflation clearance added to `footprint_radius` for `nav2_param_overrides`. (L274)
- `class GripperReadMode(str, Enum)` — How `MujocoArmHAL` reports the gripper qpos. Values: `SUM_OVER_SCALE` (Franka parallel — normalised to `[0,1]`), `AFFINE_LOW_HIGH` (SO-100 revolute Jaw — normalised to `[0,1]`), `PASSTHROUGH` (Aloha prismatic / OpenArm revolute — raw qpos in MJCF units). (L1358)
- `class GripperWriteMode(str, Enum)` — How `MujocoArmHAL` maps an Action's gripper value to `ctrl`. Values: `NORMALISED` (`[0,1]` → `ctrl_range`), `PASSTHROUGH` (raw → `ctrl`; MuJoCo clips). (L1380)
- `class SimGripperDescription(BaseModel)` — Gripper wiring inside a MuJoCo MJCF. (L1396)
  fields: `joint, ctrl_range, qpos_addrs, qpos_scale, read_mode, write_mode, actuator_index, mirror_actuator_index`
- `class UrdfAsset(BaseModel)` — A URDF asset reference plus its `robot_state_publisher` wiring. (L1460)
  fields: `ref: str` (validated against the `resolve_asset` scheme grammar), `root_frame: str | None` (URDF root link when it differs from `base_frame`), `base_to_root_xyz_rpy: tuple[float×6] | None` (static `base_frame`→`root_frame` transform [x,y,z,roll,pitch,yaw], metres+radians)
- `class AssetRefs(BaseModel)` — Unified `RobotDescription.assets` block: one URDF/MJCF/SRDF reference set replacing the former scattered asset fields. (L1497)
- `prop _ASSET_SCHEMES, _ROS2_DYNAMIC` — Manifest-side mirror of `openral_core.assets`'s scheme grammar, kept in lock-step with the resolver. (L1447–1448)
  fields: `urdf: UrdfAsset | None`, `mjcf: str | None`, `srdf: str | None` (the last two are bare refs, validated against the same scheme grammar; all default `None`)
- `class SimDescription(BaseModel)` — Optional `RobotDescription.sim` block holding MuJoCo joint↔qpos/qvel/actuator wiring for `MujocoArmHAL.from_description`. (L1528)
  fields: `floating_base, joint_qpos_addr, joint_qvel_addr, actuator_index, grippers, settle_steps_default, keyframe_index, seed_ctrl_from_qpos`
- `class HalEntrypoints(BaseModel)` — `RobotDescription.hal` block: the robot's sim + real-hardware HAL import strings, resolved by `openral_hal.build_hal`, plus optional `real_bringup: str | None` (`"<ros_pkg>:<file>.launch.py"`) — the vendor ros2_control bringup `deploy run` includes; `None` falls back to the HAL package's `launch/real_bringup.launch.py` convention. (L2166)
  fields: `sim: str | None` (null → derive `MujocoArmHAL.from_description` when a `sim:` block exists), `real: str | None` (null → simulation-only robot), `parameters: HalParameters` (per-robot HAL construction defaults)
- `class HalParameters(BaseModel)` — `RobotDescription.hal.parameters` block: per-robot HAL construction defaults merged into the constructor by `openral_hal.build_hal`, so a parameterised robot needs no bespoke lifecycle subclass. (L2108)
  fields: `defaults: dict[str, object]`
- `class TopCameraDefaults(BaseModel)` — Default placement for the scene-level "top"/"base" camera consumed by sim backends that render an overview camera. (L1587)
  fields: `pos: tuple[float, float, float], target: tuple[float, float, float], fovy: float (gt=0, lt=180)`
  - A backend YAML override still wins — this submodel is only the default fed to the composer.
- `class SceneDefaults(BaseModel)` — Per-robot scene rendering defaults consulted when the scene YAML does not override them. (L1652)
  fields: `top_camera: TopCameraDefaults | None`, `composition: SceneComposition | None`
- `class SceneComposition(BaseModel)` — Declarative MJCF scene composition; `composer: "module:fn"` returns `(xml, meshdir)`. (L1617)
  fields: `composer: str`, `params: dict[str, object]`
  - First consumer is the `openarm_tabletop_pnp` MJCF composer; future scenes can extend this as more backend hardcodes are pulled out.

**Pydantic models — runtime snapshots**

- `class JointState(BaseModel)` — Real-time joint state snapshot. (L2737)
  fields: `name, position, velocity, effort, stamp_ns`
- `class Pose6D(BaseModel)` — 6D pose (position + xyzw quaternion). (L2755)
  fields: `xyz, quat_xyzw, frame_id`
- `class DetectedObject(BaseModel)` — Object detection. (L2769)
  fields: `label, confidence, pose, bbox_3d, track_id`
- `class AttachmentEvidenceKind(str, Enum)` — Evidence source confirming an attachment or its support contact. The two sim kinds are not interchangeable: a simulator's contact list is not a proximity oracle, so `sim_geom_distance` (a distance probe) can see contact the contact list hides. (L2787)
- `class AttachedCollisionPrimitive(BaseModel)` — One sphere/capsule/OBB plus `pose_in_object`, with an optional `tight_geometry` refining it for the kernel's payload-vs-world-voxel check; `None` is not a defect — sphere/box/capsule sim geoms lower exactly, only mesh geoms leave a gap. Produced by `openral_hal._sim_attachment_evidence._tight_geometry_from_points`. (L3460)
  - `_tight_geometry_fits_inside_the_box(self) -> Self` [@model_validator(mode="after")] — Delegates to `check_tight_geometry_fits_box` at the producer boundary; the kernel's ingest re-runs the same proof and drops what it can't verify.
  - `from_idl(cls, msg, *, object_id) -> AttachedCollisionPrimitive` [@classmethod] — Decode one duck-typed IDL message without importing ROS; raises `ValueError` on an unknown `shape_type`. (L3511)
  - `fill_idl(self, msg) -> None` — Encode to a duck-typed IDL message without importing ROS; raises `ROSConfigError` for a shape it cannot represent, rather than publishing a default-tagged message with no dimensions. (L3549)
- `class PlaceRegion(BaseModel)` — The producer-measured bounded region of a declared place target; inside it, the payload's world-collision margin is reduced so it can reach its earned support contact. Oriented, base-frame only, and producer-specific — no allowance without a measurement (real hardware has none yet), and a refusal means no allowance, never a dropped message. (L2819)
  - `volume_m3(self) -> float` — Volume of the region box in cubic metres. (L2941)
  - `from_idl(cls, msg) -> Self` [@classmethod] — Decode the duck-typed OpenRAL ROS IDL message without importing ROS. (L2947)
  - `fill_idl(self, msg, *, primitive_factory=None) -> None` — Populate a duck-typed OpenRAL ROS IDL message without importing ROS. (L2972)
- `class PlaceDeclaration(BaseModel)` — Dispatch's typed statement that a place phase is active for a payload; `region`, when set, is the only thing that lifts the pick witness's mid-carry anti-scope, and is never inferred from motion or sim introspection. `is_live` fails toward dead (retracted, past `timeout_s`, or future-stamped), and its `now_ns` must read the same clock domain that produced `stamp_ns` — never wall-clock time. (L3018)
  - `is_live(self, *, now_ns) -> bool` — Whether this declaration is still in force at `now_ns`; fails toward dead. See the class docstring for the clock-domain contract on `now_ns`. (L3118)
  - `from_idl(cls, msg) -> Self` [@classmethod] — Decode the duck-typed OpenRAL ROS IDL message without importing ROS. (L3152)
  - `fill_idl(self, msg, *, primitive_factory=None) -> None` — Populate a duck-typed OpenRAL ROS IDL message without importing ROS. (L3170)
- `class ContactForceWitness(BaseModel)` — Layer 2's bounded attestation of a measured contact force between a payload and its declared place target; the gate it feeds only ever adds a refusal, never licenses a contact or widens an allowance. `magnitude_n` is newtons only when `magnitude_calibrated`. Absence of a witness is not evidence of absent contact. (L3313)
  - `from_idl(cls, msg) -> Self` [@classmethod] — Decode the duck-typed OpenRAL ROS IDL message without importing ROS. (L3391)
  - `fill_idl(self, msg) -> None` — Populate a duck-typed OpenRAL ROS IDL message without importing ROS. (L3410)
- `class SupportContactWitness(BaseModel)` — World State's bounded attestation that a payload rests on a named support surface; geometry is in the attached object's own frame, and `max_penetration_m` is the physical depth (occupancy quantisation is accounted for separately). (L3200)
  - `from_idl(cls, msg) -> Self` [@classmethod] — Decode the duck-typed OpenRAL ROS IDL message without importing ROS. (L3282)
  - `fill_idl(self, msg) -> None` — Populate a duck-typed OpenRAL ROS IDL message without importing ROS. (L3298)
- `class AttachedCollisionObject(BaseModel)` — Collision-active payload rigidly attached to one robot link; `support_contact=None` (the default) means the safety kernel exempts no payload-vs-world contact. (L3606)
  - `from_idl(cls, msg) -> Self` [@classmethod] — Decode the duck-typed OpenRAL ROS IDL message without importing ROS. (L3680)
  - `fill_idl(self, msg, *, primitive_factory) -> None` — Populate a duck-typed OpenRAL ROS IDL message without importing ROS. (L3738)
- `class OccupancyGridRef(BaseModel)` — Reference to a 2D occupancy grid for mobile-base world-collision, mirroring `nav_msgs/OccupancyGrid` metadata. (L3780)
  fields: `frame_id, resolution_m (>0), width (>=0), height (>=0), origin: Pose6D, data_topic`
- `class WorldState(BaseModel)` — Snapshot consumed by Reasoner and Skills; `policy_state` carries a checkpoint-specific proprio vector when joint state alone can't represent it, and `place_declaration` relays the resolved place phase (region included) to the safety kernel — `None` means no place phase, no witness, no allowance. (L3928)
  fields: `stamp_ns, joint_state, base_pose, base_twist, ee_poses, contact_forces, images, image_frames, point_clouds, tactile, detected_objects, battery_pct, diagnostics, attached_objects, attachment_revision, attachment_stamp_ns, occupancy_grid`
  - **occupancy_grid** — Optional 2D occupancy-grid reference for mobile-base footprint checks; an absent or stale grid is treated as unavailable (fail-closed).
  - **attached_objects / attachment_revision / attachment_stamp_ns** — Collision-active payloads plus the producer-owned revision and freshness timestamp; republishing `WorldState` does not refresh stale attachment evidence.
  - **image_frames** — Optional in-process frame carrier for no-ROS deployments; `None` keeps the existing `images` topic-ref path unchanged.

#### Persistent spatial memory — scene graph

_Advisory, queryable Layer-2 world model the S2 Reasoner consults to recall objects/places/agents. Never a safety input — the kernel gates only on the geometric world. Poses anchored in the tf2 `map` frame._

- `class SpatialNodeKind(str, Enum)` — `OBJECT | PLACE | ROOM | AGENT`. (L3992)
- `class SpatialRelationKind(str, Enum)` — `CONTAINS | AT_PLACE | TRAVERSABLE_TO | ON | NEAR`. (L4008)
- `class SpatialNode(BaseModel)` — A typed scene-graph node; superset of `DetectedObject` for `kind=OBJECT`. (L4025)
  fields: `node_id, kind, pose: Pose6D, label, confidence, bbox_3d, embedding_ref, is_container, occludes_contents, first_seen_ns, last_seen_ns, observation_count`
- `class SpatialEdge(BaseModel)` — Directed relation between two nodes. (L4100)
  fields: `src, dst, kind: SpatialRelationKind`
- `class SceneGraph(BaseModel)` — Persistent scene-graph memory; validates unique node ids and that every edge references an existing node. (L4116)
  fields: `schema_version="0.1", nodes: list[SpatialNode], edges: list[SpatialEdge]`
- `class RecallObjectQuery(BaseModel)` — Read-only object recall query. (L4157)
  fields: `text, label, near: Pose6D | None, max_age_ns, limit`
- `class ApproachViewpoint(BaseModel)` — Camera-facing standoff goal. (L4189)
  fields: `pose: Pose6D, standoff_m (>0), camera_frame_id`
- `class RecallObjectMatch(BaseModel)` — One ranked recall result. (L4206)
  fields: `node_id, label, pose: Pose6D, score, last_seen_ns, approach: ApproachViewpoint | None, inside_container_id: str | None`
- `class RecallObjectResult(BaseModel)` — `matches` list; empty means unknown, so the caller raises `ROSObjectNotInMemory`. (L4231)
- `class ResolvePlaceQuery(BaseModel)` — Resolve a place/room/agent reference. (L4246)
  fields: `reference, kind: SpatialNodeKind | None`
- `class ResolvePlaceResult(BaseModel)` — Resolved node plus a `traversable_to` path. (L4261)
  fields: `node_id, goal: Pose6D, path_node_ids: list[str]`
- `class Action(BaseModel)` — Action step or chunk produced by a Skill; `tick_index` preserves multi-slot atomicity across the safety wire, and optional `joint_names` names which joints a sub-slot targets (needed since a zero-padded slot would otherwise read `0.0` as a legal target). (L4282)
  fields: `control_mode, horizon, joint_targets, joint_velocities, joint_torques, cartesian_pose, cartesian_delta, cartesian_delta_scale, cartesian_twist, body_twist, foot_placements, gripper, dex_hand_joints, confidence, stamp_ns, ee_name, frame_id, safety_overrides`
- `class QuantizationConfig(BaseModel)` — Quantization recipe. (L4437)
  fields: `dtype, backend, per_channel, calibration_dataset, extra`
- `class DeviceInfo(BaseModel)` — Host compute snapshot. (L4461)
  fields: `device_str, gpu_memory_bytes, cuda_compute_capability, cpu_count, arch`
- `class RSkillInfo(BaseModel)` — Skill runtime state snapshot. (L4513)
  fields: `name, version, state, weights_loaded, quantized, warmed_up, embodiment_tags, role, latency_budget_ms, last_inference_ms, error_msg, stamp_ns`

**Pydantic models — skill packaging (rSkill)**

- `class RSkillLatencyBudget(BaseModel)` — Per-stage latency budget; `max_execution_s` is the total wall-clock budget for one `execute_rskill` goal, bounding a VLA that never self-terminates. (L4595)
  fields: `per_chunk_ms, warmup_ms, load_ms, max_execution_s`
- `class SensorRequirement(BaseModel)` — One sensor an rSkill needs the robot to provide. (L4620)
  fields: `modality, vla_feature_key, min_width, min_height, count`
- `class ImagePreprocessing(BaseModel)` — Per-rSkill checkpoint image-preprocessing contract, so the sim adapter doesn't need to learn a checkpoint's frame conventions from a YAML override. `aliases` keys are VLA slots (`camera1`), never sensor/scene camera names. (L4677)
  - `unknown_alias_slots(sensors_required) -> list[str]` — Alias keys no `sensors_required[].vla_feature_key` declares as a slot; `resolve_image_preprocessing` raises on any. (L4754)
- `class ControlModeSemantics(BaseModel)` — Action-space semantics on each `ActuatorRequirement`. (L6050)
  fields: `mode: Literal["absolute","delta"], gripper_convention, joint_order, reference_frame`
  - Cross-validator on `ActuatorRequirement`: gripper kinds require `gripper_convention`; cartesian kinds require `reference_frame`; other kinds forbid both.
- `GripperConvention` (TypeAlias = Literal[...]) — Closed gripper-action encoding set. Members: `normalized_open_unit, normalized_open_symmetric, binary_close_one, raw_joint_rad, width_meters`. (L6023)
- `class ActuatorRequirement(BaseModel)` — One actuator slot an rSkill emits actions for; `n_dof`/`vla_action_key` auto-fill from the robot YAML for canonical embodiments, required for the `"custom"` hatch. (L6099)
  fields: `kind, n_dof, vla_action_key, control_mode_semantics`
  - `kind` reuses `ControlMode`; `n_dof` / `vla_action_key` auto-fill from the robot YAML for canonical embodiments, required on the manifest for the `"custom"` hatch.
  - `control_mode_semantics` is required: declares absolute-vs-delta and, when applicable, gripper convention / reference frame.
- `class EmbodimentExtra(BaseModel)` — Sensor + actuator surface for the `"custom"` embodiment hatch. (L6197)
  fields: `sensors: list[SensorRequirement] (≥1), actuators: list[ActuatorRequirement] (≥1)`
- `class RSkillProcessors(BaseModel)` — Explicit lerobot `PolicyProcessorPipeline` artefact pointers. (L6434)
  fields: `preprocessor_uri, postprocessor_uri`
  - Per-file URI shape `hf://owner/repo[@rev]/path/to/file.ext` — the file tail is required; a bare repo URI is the implicit-snapshot shape this deliberately replaced.
  - Cross-validator rejects identical pre/post URIs.
- `_PROCESSOR_URI_PATTERN: str` — Validates `RSkillProcessors.preprocessor_uri`/`postprocessor_uri` requires a file tail. (L6409)
- `class RosIntegration(BaseModel)` — Wiring for a wrapped ROS 2 action/service; required when `RSkillManifest.kind` is `ros_action`/`ros_service`, forbidden otherwise. (L6565)
  fields: `package, interface_type, interface_name, result_trajectory_field, default_goal_json, ros_dependencies`
  - `result_trajectory_field is None` → result-only mode (Nav2 shape); set → trajectory mode (MoveIt shape, replaying one waypoint per `step()`).
  - `default_goal_json` validator round-trips the literal through `json.loads` and rejects non-dict payloads.
- `_ROS_WRAPPER_KINDS: frozenset[str]` — `{"ros_action", "ros_service"}`, the kinds requiring `RosIntegration`. (L6555)
- `class SegmenterEngine(str, Enum)` — Backend selector for `kind: "segmenter"` rSkills; required on `SegmenterContract.engine` (no legacy fallback, unlike `DetectorEngine`). (L6819)
- `class SegmenterContract(BaseModel)` — Manifest contract for `kind: "segmenter"` rSkills, the sibling of `DetectorContract` for models answering a geometric prompt rather than a semantic one. Deliberately carries no `labels`/`score_threshold` — a segmenter never classifies and its consumer never gates on model confidence. (L6840)
- `class DetectorEngine(str, Enum)` — Backend selector for `kind: "detector"` rSkills, disambiguating backends that otherwise share a `runtime`; `None` keeps the legacy `runtime`-keyed dispatch. (L6677)
- `class DetectorMode(str, Enum)` — Invocation mode of a detector, orthogonal to `DetectorEngine`: `CONTINUOUS` (always-on, feeds `WorldState.detected_objects`, not dispatchable) vs `ON_DEMAND` (prompted via the `locate_in_view` tool). (L6713)
- `class DetectorContract(BaseModel)` — Manifest contract for `kind: "detector"` rSkills; `max_side` caps the VLM-sidecar's resize before grounding, trading detection resolution for lower activation VRAM. (L6749)
  fields: `labels: list[str]` (min_length=1; class-label list indexed by model class-id), `input_size: tuple[int, int]` (width × height, both > 0; default (640, 640)), `score_threshold: float` (ge=0.0 le=1.0; default 0.5), `engine: DetectorEngine | None`, `mode: DetectorMode` (default `continuous`)
- `class RewardContract(BaseModel)` — Manifest contract for `kind: "reward"` rSkills; a reward monitor is a pure perception consumer with no actuators or action/state contract, so its progress/success signal is advisory only. (L6895)
- `class PlaybookContract(BaseModel)` — Manifest contract for `kind: "playbook"` rSkills (human-authored S2 decision procedure); content the reasoner reads into its system prompt, never code it executes — no weights, actuators, or action/state contract. (L6998)
- `class RSkillManifest(BaseModel)` — `rskill.yaml` manifest; a pre-publish surface extended in place several times without a schema-version bump. (L7055)
  fields: `schema_version, name, version, license, role, kind, model_family, embodiment_tags, embodiment_extra, capabilities_required, sensors_required, actuators_required, runtime, quantization, weights_uri, chunk_size, latency_budget, min_vram_gb, fallback_skill_id, benchmarks, evaluated_tasks, sim_env_control_mode, policy_extras, paper_url, dataset_uri, source_repo, description, default_prompt, actions, objects, scenes, processors, image_preprocessing, state_contract, action_contract, n_action_steps, ros_integration, detector, reward, reward_rskill_name, playbook`. `kind` gates which other fields are required or forbidden (see `_check_kind_consistency` below). `evaluated_tasks` gates a scene's task against what the checkpoint was trained on, raising `ROSCapabilityMismatch` when it doesn't cover the scene. `reward_rskill_name` names the reward monitor a VLA pairs with, since a VLA emits no success signal of its own; `None` defers to the deployment default reward model rather than meaning "run without reward". `default_prompt` is the literal conditioning string a single-task checkpoint was trained on — distinct from `description`, which is prose for the LLM to pick the skill by.
  - `from_yaml(cls, path: str) -> RSkillManifest` [@classmethod] — Load and validate an `rskill.yaml`. (L7877)
  - `active_min_vram_gb(self) -> float | None` — Declared VRAM for this skill at its active quantization dtype, or `None` when undeclared; consumed by `assert_vla_reward_fits`. (L7231)
  - `is_commercial_use_allowed: bool` [@property] — Derived from `license`: True for apache-2.0/mit/bsd, False otherwise. (L7319)
  - `is_scaffold_placeholder: bool` [@property] — True when `name`/`weights_uri`/`source_repo` still carry an unresolved template sentinel, i.e. this is the `rskills/template/` scaffold; the reasoner palette gate and the publish gate both use it to keep the scaffold from being dispatched or published. (L7331)
  - Cross-validators: `"custom" ∈ embodiment_tags ↔ embodiment_extra is not None`; a `"custom"` entry requires `n_dof` and `vla_action_key` on every actuator.
  - rSkill self-containment audit cross-validator: `processors` required for most model families; legacy `act` may omit it.
  - Cross-validator (`_check_kind_consistency`): each `kind` requires and forbids a specific field set (e.g. `vla` needs `model_family`+`weights_uri`, `detector` needs `detector`+`weights_uri` and no actuators); `wam` validates schema-side but the loader rejects it at resolve time.
  - Cross-validator (`_check_n_action_steps_within_chunk`): `n_action_steps <= chunk_size` (`None` = the adapter's default, resolved by `openral_rskill._vla_core.resolve_n_action_steps`).
  - The historical `policy_id` field was removed in favour of dispatching on `model_family` directly.
- `prop _HF_HUB_ID_PATTERN, _SEMVER_PATTERN, _WEIGHTS_URI_PATTERN, _HF_DATASET_URI_PATTERN, _HTTPS_URL_PATTERN` — Regexes validating `RSkillManifest`'s URI/version/URL fields, pinned at module scope so the patterns stay greppable. (L6367–6376)
- `_LICENSES_ALLOWING_COMMERCIAL: frozenset[RSkillLicensePosture]` — Licenses `RSkillManifest.is_commercial_use_allowed` treats as commercial-OK: `apache_2_0`/`mit`/`bsd`. (L6414)
- `_MODERN_PROCESSOR_FAMILIES: frozenset[str]` — Model families requiring `RSkillManifest.processors`; `act` may omit it via the legacy norm-stats-in-safetensors path. (L6429)
- `RSKILL_TEMPLATE_SENTINELS: tuple[str, ...] = ("TEMPLATE_ORG", "TEMPLATE_ID")` — Canonical unresolved-scaffold sentinels the `openral rskill new` scaffolder rewrites; shared by the reasoner palette gate and the publish gate so the "is this a published skill?" rule can't drift. (L6386)
- `def contains_rskill_template_sentinel(text: str | None) -> bool` — True when `text` carries an `RSKILL_TEMPLATE_SENTINELS` substring; the text-level primitive behind `is_scaffold_placeholder`. (L6389)
- `def assert_vla_reward_fits(vla: RSkillManifest, reward: RSkillManifest, gpu_total_gb: float, *, margin_gb: float = 0.5) -> float` — Pre-load gate verifying a VLA and its paired reward model co-reside in GPU VRAM; raises `ROSConfigError` if either size is undeclared, `ROSGPUMemoryError` if they don't fit, else returns the combined GB. (L8318)
- rSkill HF-repo naming, enforced by `tools/rskill_publisher.py`. Hyphens are only the segment separator; a name parses by `split("-")` into one of three shapes — `rskill-<model>-<robot>-<task>-<quant>` (weight-bearing kinds), `rskill-<model>-<robot>-<task>` (`ros_action`/`ros_service`, no weights), or `rskill-playbook-<name>`.
  - `_DEFAULT_RSKILL_OWNER: str = "OpenRAL"` — Owner used when a manifest `name` carries no `<owner>/` prefix. (L7924)
  - `_RSKILL_NAME_SEGMENT_RE: re.Pattern[str]` — Every name segment must match `^[a-z0-9][a-z0-9_]*$`. (L7927)
  - `prop _RSKILL_NAME_PARTS, _RSKILL_ROS_NAME_PARTS, _RSKILL_PLAYBOOK_NAME_PARTS` — Expected hyphen-segment counts for the three name shapes. (L7934–7936)
  - `CANONICAL_MODEL_TOKENS: frozenset[str]` — Versioned `<model>` checkpoint vocabulary; distinct from `ModelFamily` since several tokens can share one family, and tool models have no family. (L7965)
  - `CANONICAL_ROBOT_NAME_TOKENS: frozenset[str]` — The reserved robot-agnostic `<robot>` tokens `{"any", "multi"}` (`multi` = a skill declaring more than one concrete robot). Concrete robot tokens are an open registry: `repo_name_is_canonical` checks them by shape, and CI (`tests/unit/test_manifest_registry_ids.py`) checks in-tree names against `robots/*/robot.yaml` tags. (L8068)
  - `CANONICAL_QUANT_TOKENS: frozenset[str]` — `{fp32, fp16, bf16, int8, nf4}`; the weightless ROS-wrapper kinds omit the `<quant>` segment entirely. (L7938)
  - `_WEIGHTLESS_KINDS: frozenset[str]` — `{"ros_action", "ros_service"}` — kinds carrying no weights, so their name omits `<quant>`. (L7943)
  - `_NAME_TAIL_QUANT_LIKE: frozenset[str]` — Trailing tokens the author-slug fallback strips from a name tail. (L7947)
  - `_QUANT_DTYPE_TO_TOKEN: dict[QuantizationDtype, str]` — Maps `quantization.dtype` to its `<quant>` name token. (L7951)
  - `_MODEL_FAMILY_ALLOWED_TOKENS: dict[str, frozenset[str]]` — Per-family allowlist of `<model>` tokens; only the special cases (versioned / multi-token families) need an entry. (L8027)
  - `def _allowed_model_tokens(model_family: str) -> frozenset[str]` — `_MODEL_FAMILY_ALLOWED_TOKENS` lookup defaulting to `{model_family}`, so a new family needs no schema edit. (L8055)
  - `_MODEL_FAMILY_TO_TOKEN: dict[str, str]` — `model_family` → canonical `<model>` suggestion token; a family absent from the map suggests its own id. (L8005)
  - `_EMBODIMENT_TO_ROBOT_TOKEN: dict[str, str]` — Robot tags whose bare form would collide, mapped to a distinct `<robot>` token; every other tag uses its bare id. (L8060)
  - `def repo_name_is_canonical(name: str, *, kind: RSkillKind, model_family: str \| None = None) -> bool` — The enforced validator: kind-selected shape plus `<model>` vocab (`CANONICAL_MODEL_TOKENS`, or the family's allowed tokens when `model_family` is set); `<robot>` and `<task>` are checked by shape only. (L8177)
  - `def expected_repo_name(manifest: RSkillManifest) -> str` — The canonical name suggestion printed on a mismatch and written by `--fix-name`; always satisfies `repo_name_is_canonical` for the manifest's kind. (L8253)
- `_REGISTRY_ID_PATTERN: str` — `^[a-z][a-z0-9_]*$`, the shape of the four open registry ids below (`StateLayout`, `EmbodimentTag`, `BenchmarkName`, `ModelFamily`). Membership is checked by the registries and CI, not the core schema. (L4770)
- `EmbodimentTag` (TypeAlias = Annotated[str, StringConstraints(pattern=_REGISTRY_ID_PATTERN)]) — Open robot-embodiment id: the tags `robots/*/robot.yaml` declare, plus `"custom"` escape hatch, `"mobile_base"` class tag, `"any"` (embodiment-agnostic wildcard for perception/playbook kinds), and `"multi"` (repo-name aggregate). Typo guard: `openral_rskill.loader.intree_embodiment_tags` + CI. (L6244)
- `StateLayout` (TypeAlias = Annotated[str, StringConstraints(pattern=_REGISTRY_ID_PATTERN)]) — Open per-checkpoint proprioception layout id, naming the trained shape (field order, frame convention, gripper encoding, quaternion handedness); per-robot source bindings live on `StateContractBindings`. Membership: `openral_state_adapter.registered_layouts()` (palette drop / `assemble_state` refusal) + CI. (L4782)
- `WRAPPED_TASK_SPACE_LAYOUTS: frozenset[StateLayout]` — Subset of `StateLayout` covering Cartesian/FK-derived composites, which require `StateContract.bindings`; joint-space layouts are excluded and served verbatim from raw `JointState.position`. (L4817)
- `class StateContract(BaseModel)` — Per-rSkill state-vector contract, surfacing the proprioception layout a checkpoint was trained against so the runtime adapter doesn't learn it from a YAML override; requires `bindings` when `layout` is in `WRAPPED_TASK_SPACE_LAYOUTS`, forbids it otherwise. (L4879)
  fields: `layout: StateLayout | None, dim: int | None, bindings: StateContractBindings | None`
- `StateContractBindings` (Pydantic model) — Per-robot source bindings for an rSkill's `state_contract.layout`; the state-side symmetric counterpart of `ControlModeSemantics`. (L4833)
  fields: `eef_frame: str | None`, `base_frame: str | None`, `world_frame: str | None = "map"`, `gripper_qpos_joints: list[str]`, `quaternion_convention: Literal["xyzw","wxyz"] = "xyzw"`
- `BenchmarkName` (TypeAlias = Annotated[str, StringConstraints(pattern=_REGISTRY_ID_PATTERN)]) — Open benchmark id (`RSkillManifest.benchmarks` keys). Membership: `openral_rskill.loader.known_benchmark_ids` (`benchmarks/*.yaml` ∪ `scenes/benchmark/*.yaml` stems) at writeback / CI; `from_pretrained` only warns. (L6281)
- `ModelFamily` (TypeAlias = Annotated[str, StringConstraints(pattern=_REGISTRY_ID_PATTERN)]) — Open VLA/policy family id used by the eval/runner adapter dispatch; required only when `RSkillManifest.kind == "vla"` (forbidden on every other kind). Membership: `openral_sim.POLICIES` + CI. Several families run out-of-process via a dedicated ZMQ sidecar rather than in-process. (L6302)
- `RSkillKind` (TypeAlias = Literal["vla","wam","ros_action","ros_service","detector","vlm","reward","playbook"]) — Discriminator selecting the loader/runner branch; `wam` is reserved and the loader rejects it at resolve time. (L6481)

**Pydantic models — skill benchmark results (`rskills/<id>/eval/<benchmark>.json`)**

- `class RSkillEvalSource(BaseModel)` — Provenance of a benchmark block. (L8386)
  fields: `paper, arxiv, model_variant, evaluated_by, reproduced_locally, reproduction_planned, reproduction_cli, table, status`
- `class RSkillEvalBenchmark(BaseModel)` — Suite identity for a benchmark block. (L8424)
  fields: `name, dataset, protocol, robot, simulator`
- `class RSkillEvalResult(BaseModel)` — On-disk shape of `rskills/<id>/eval/*.json`; optional `trace_id` cross-references the OTel trace tree when the run had an OTLP endpoint to export to. (L8446)
  fields: `schema_version, source, benchmark, eval_config, results, baselines`
  - `from_json(cls, path: str) -> RSkillEvalResult` [@classmethod] — Load and validate a single benchmark JSON. (L8496)

**Pydantic models — validation-matrix round verdicts (`outputs/validation-matrix/<round>/verdicts.json`)**

Written by `tools/validation_matrix.py`; the machine-readable half of [`docs/reference/collision-validation-evidence.md`](../reference/collision-validation-evidence.md). The kernel's verdict is transcribed verbatim and adjudicated only against the simulator's own ground truth.

- `ValidationOutcome` (TypeAlias = Literal[...]) — How one scene ended. Members: `completed, estop-collision-real, estop-collision-false-positive, estop-collision-within-quantization, estop-collision-unadjudicated, estop-initial-configuration, deadline-after-grasp, deadline-no-grasp, harness-error`.
- `GroundTruthAdjudication` (TypeAlias = Literal["real-contact","false-positive","within-quantization","unadjudicated"]) — Verdict of the simulator's distance probe on a kernel stop; `real-contact` requires a solid-geom pair at or below 0 m, an unfiltered snapshot reads as `unadjudicated`.
- `class ValidationStopEvidence(BaseModel)` — The kernel's `safety.collision` line, transcribed field for field. (L8579)
  fields: `kind, party_a, party_b, horizon_step, min_distance_m, sweep_min_distance_m, place_allowance_active, place_target, depth_is_box_bound`
  - `exemption_active` [@property] — Whether a support witness exempted a deeper cell at the trip. (L8643)
  - `involves_payload` [@property] — Either party is `attached:<object_id>`. (L8650)
- `class ValidationGroundTruthAdjudication(BaseModel)` — Adjudication of a stop against `sim.estop_ground_truth_snapshot`; withdraws to `unadjudicated` when the probe's distances carry no certified proof, since the simulator's own distance function is known wrong on this pair class. (L8655)
  fields: `verdict, stop_class, sim_time_s, grid_resolution_m, quantization_budget_m, admissible_gap_m, budget_source, probe_collidability_filtered, probe_distance_certified, unadjudicated_reason, nearest_any_m, nearest_tripping_party_m, nearest_pair, discrepancy_m, probed_pairs, probe_truncated, distmax_m, payload_contacts`
- `class ValidationWitnessTimeline(BaseModel)` — Attach/witness/place-declaration lifecycle, recorded from both the monitor and the kernel's own log lines since a disagreement is itself a finding. (L8733)
  fields: `attach_t_s, detach_t_s, support_id, kernel_witness_armed, kernel_witness_separated, place_declaration_seen, place_region_armed, place_allowance_active_lines`
- `class ValidationSceneVerdict(BaseModel)` — One scene of one round, queryable; `config_sha256` pins the exact YAML bytes that ran. (L8772)
  fields: `scene, config_path, config_sha256, seed, prompt, rskill_id, outcome, task_success_final, task_success_ever, task_success_steps, task_success_transitions, stop, ground_truth, witness, monitor_records, dispatch_failure_reason, wall_s, artifacts, harness_error_reason`
- `class ValidationRoundMetadata(BaseModel)` — The reproducibility half of a round; every field exists because a past round could not be reproduced without it. (L8845)
  fields: `round_id, started_at, host, executed_sha, worktree_clean, overlay_built_at_ns, launcher_path, repo_root, robot_manifest_path, robot_id, sync_groups, stack_argv, safety_overrides_absent, gpu_name, notes_path, scene_dirs, seed, artifact_stem, imported_from, scene_pins`
- `class ValidationRoundVerdicts(BaseModel)` — On-disk `verdicts.json`. (L8934)
  fields: `schema_version, metadata, scenes`
  - `from_json(cls, path: str) -> ValidationRoundVerdicts` [@classmethod] — Load and validate a round's verdicts. (L8966)
  - `scene(self, name: str) -> ValidationSceneVerdict | None` — The verdict for one scene key. (L8985)
- `class ValidationSceneDelta(BaseModel)` — How one scene moved between rounds. (L8997)
  fields: `scene, baseline_outcome, outcome, changed, changed_fields`
- `class ValidationRoundDiff(BaseModel)` — Round-over-round comparison. (L9031)
  fields: `schema_version, round_id, baseline_round_id, executed_sha, baseline_executed_sha, seed, baseline_seed, scenes`
  - `same_sha` [@property] — Both rounds ran the same code. (L9073)
  - `same_seed` [@property] — Both rounds pinned the same scene seed. (L9078)
  - `is_reproducibility` [@property] — `same_sha and same_seed`; anything else is a before/after, since a seed change moves the scene's initial configuration however equal the SHAs are. (L9083)
  - `changed_scenes` [@property] — Scene keys whose outcome moved. (L9093)

**Pydantic models — sim eval**

- `class SceneSpec(BaseModel)` — Physics scene declaration. (L9144)
  fields: `id, backend, assets_uri, observation_height, observation_width, cameras, backend_options`
- `class TaskSpec(BaseModel)` — What the robot must achieve. (L9187)
  fields: `id, scene_id, instruction, max_steps: int | None, success_key: str | None, metadata`
- `class VLASpec(BaseModel)` — Policy / brain declaration. (L9241)
  fields: `id, weights_uri, device, runtime, quantization, deterministic, extra`
- `class SimEnvironment(BaseModel)` — Runtime (robot × scene × task × VLA) tuple, composed at the CLI from a `SimScene`/`BenchmarkScene` YAML plus an `RSkillManifest`; never loaded from YAML directly. (L9282)
  fields: `robot_id, scene, task, vla, base_pose, seed, n_episodes, record_video, save_dir, metadata`
  - `base_pose: Pose6D | None = None` — Per-rollout robot mounting pose in the scene's world frame; honoured by free-axis scenes only.
  - `model_post_init(_context: object) -> None` — Cross-field validation `task.scene_id == scene.id`. (L9341)
- `class BenchmarkMetadata(BaseModel)` — Provenance block required on every `BenchmarkScene`; suite invariants treat it as byte-identical across scenes. (L9351)
- `class LaunchInclude` — A vendor ROS 2 launch a deploy must bring up alongside the graph, since a `SensorDeployBinding` names a topic to subscribe to but not who publishes it. Carried by `DeployScene.drivers`, real path only. (L9375)
- `class DeployRuntime(BaseModel)` — Committed deploy-posture toggles for a workcell scene (SLAM, Nav2, octomap, object detector, reward monitor, critic, spatial-memory ingest), all tri-state with explicit CLI flag > scene `runtime:` > auto/built-in default precedence. **`preload_rskill_id` / `preload_prompt`** (optional, scene-only): the rSkill the skill_runner resolves and loads right after it activates, in a worker thread, so the first `execute_rskill` goal finds it GPU-resident — the deadman watchdog opens its 120 s first-chunk window on goal ACCEPT, and a 3.6 B π0.5 needs ~350 s to load on a Jetson AGX Orin, so a cold load inside a goal is E-stopped, correctly; the prompt must be the exact string later goals send (resident key = id, revision, prompt). Forwarded as `preload_rskill_id:=` / `preload_prompt:=` only when set. (L9413) **`joint_states_topic: str | None`** (scene-only): the `JointState` topic the in-process world state ingests (`None` = `/joint_states`); read by `runtime_node` from the scene and set on the composed world_state node before its `on_configure`. On a ros2_control arm `/joint_states` is the broadcaster's 750 Hz stream, and every message wakes the runtime's Python executor — on an AGX Orin that loop plus the callback held ~half of the process's GIL, the in-process inference thread got <2 %, and a 1.6 s π0.5 forward took 322 s. Point it at the HAL's 30 Hz `~/joint_states` republish (`/openral_hal_openarm/joint_states`); the C++ kernel keeps the full-rate topic. `clock_origin` (`host_wall` | `simulation` | `None`) pins the graph's clock authority; `None` on a twin whose `octomap_cloud_topic` lies outside `/openral/cameras/` already selects `host_wall` (`simulation` with it is refused). **`world_voxel_deadline_s: float = 1.0`** / **`max_octree_age_s: float | None`**: the kernel's voxel deadline and the octomap bridge's octree-age bound, per rig (a source slower than ~1 Hz raises both); a validator refuses an age above the deadline so the kernel, not the bridge, fails closed (hazard log Entry 033).
  - `voxel_freshness_s -> tuple[float, float]` [@property] (L9574) — `(world_voxel_deadline_s, max_octree_age_s)` with the age defaulted to the deadline; what `openral deploy` forwards to the launch.
- `class DeployScene(BaseModel)` — Unified deploy/workcell scene for `openral deploy sim`/`run`; `place_declaration` is the committed place-phase declaration for a direct dispatch (`None` means no place witness can arm), and a scene may not supply its own `place_declaration.region` since that must be producer-measured. No `tasks` field — deploy goals come from the operator via `--initial-task` / `/openral/prompt`. (L9618)
  - `from_yaml(cls, path: str) -> Self` [@classmethod] — Load and validate a scene YAML from disk; inherited by `SimScene`/`BenchmarkScene`, which validate against their own stricter schemas. (L9759)
- `class SimScene(DeployScene)` — Extends `DeployScene` with `task`, `seed`, `n_episodes`, `record_video`, `save_dir`, `metadata`; cross-validates `task.scene_id == scene.id`; accepted by `openral sim run`. (L9768)
- `class BenchmarkScene(SimScene)` — Extends `SimScene` with required eval fields (`n_episodes`, `seed`, `metadata`, `task.success_key`, `task.max_steps`); consumed by `openral benchmark`. (L9795)
- `class ProtocolSpec(BaseModel)` — Standalone eval-protocol schema, retained for decision-record drafts and report tooling that quote a published protocol verbatim; never embedded in a benchmark suite. (L9840)
  fields: `n_episodes, seeds, success_key, max_steps, min_reps`
  - `model_post_init(_context: object) -> None` — Cross-field validation: `len(seeds) >= n_episodes` and `min_reps <= n_episodes`. (L9883)

**Pydantic models — inference runner**

On-disk + runtime contracts for the hardware inference runner (`openral deploy --config R.yaml`), sibling of `SimEnvironment` / `openral sim run`. Schemas are additive — `SimEnvironment` / `RSkillEvalResult` / `BenchmarkScene` are untouched.

- `class FrameEncoding(str, Enum)` — How `SensorFrame` bytes are interpreted. (L3808)
  `BGR8, RGB8, MONO8, DEPTH16, JPEG, PNG, CUDA_NV12, CUDA_RGBA, RAW` — `CUDA_NV12` is the Tegra NVMM handle layout, `CUDA_RGBA` the x86 DeepStream one
- `class SensorFrame(BaseModel)` — Single sensor frame: metadata plus optional inline/topic/handle payload; JSON-serializes the binary payload as base64. (L3830)
  fields: `sensor_id, stamp_monotonic_ns, stamp_wall_ns, encoding, width, height, channels, data, topic, handle, metadata`
  - `_decode_data(cls, value: Any) -> bytes | None` [@field_validator("data", mode="before")] — Accepts raw `bytes` or a base64-encoded `str` on JSON parse. (L3897)
  - `_encode_data(self, value: bytes | None) -> str | None` [@field_serializer("data", when_used="json")] — JSON-serializes the binary payload as base64. (L3913)
  - `model_post_init(_context: object) -> None` — Cross-field validation: exactly one of `(data, topic, handle)` must be set. (L3917)
- `class SensorReaderBackend(str, Enum)` — Which `SensorReader` implementation a sensor uses. (L9913)
  `OPENCV_THREAD, ROS2_IMAGE, GSTREAMER`
- `class DeadlineOverrunPolicy(str, Enum)` — Behaviour when a tick exceeds `1 / rate_hz`. (L9941)
  `WARN, DROP, RAISE`
- `class SensorReaderConfig(BaseModel)` — Per-sensor reader backend plus optional ROS-tee. (L9955)
  fields: `sensor_id, backend, backend_params, max_age_ms, publish_to_ros, publish_topic, publish_rate_hz, publish_frame_id, publish_camera_info` — `publish_frame_id` stamps the tee's `Image`/`CameraInfo` headers (default `sensor_id`); `publish_camera_info: IntrinsicsPinhole | None` makes the tee also publish `CameraInfo` on the image topic's sibling.
  - `model_post_init(self, _context: object) -> None` — Cross-field validation for the ROS tee: `publish_to_ros ↔ publish_topic`; `publish_frame_id` / `publish_camera_info` require `publish_to_ros`. (L10017)
- `class SensorDeployBinding(BaseModel)` — Optional `SensorSpec.deploy_binding` payload letting `openral deploy run` open the physical camera; the runtime counterpart of `sim_placement`. Robot cameras carry it in `robot.yaml`; a deploy scene never touches them. (L10047)
- `class HalConfig(BaseModel)` — Which HAL adapter to instantiate plus transport params (serial port / FCI URI / ROS namespace). (L10101)
  fields: `adapter, transport, params`
- `class TickResult(BaseModel)` — One tick's record returned by `InferenceRunner.tick`; optional sim-only fields and trace context default to `None` so hardware ticks serialize unchanged from v1. (L10134)
  fields: `stamp_ns, tick_idx, sensors_ms, world_state_ms, inference_ms, safety_ms, hal_ms, tick_ms, chunk_index, safety_violations, action_applied, step_idx, episode_idx, reward, terminated, truncated`
- `class RunResult(BaseModel)` — Aggregated summary returned by `InferenceRunner.run`. (L10220)
  fields: `n_ticks, success, budget_violations, avg_inference_ms, p99_inference_ms, avg_tick_ms, p99_tick_ms, trace_id, save_dir, metadata`

**Pydantic models — failure evidence**

Discriminated union backing the `evidence_json` field of `openral_msgs/msg/FailureTrigger`. Discriminator is `kind`; decode via `pydantic.TypeAdapter(FailureEvidence).validate_json(...)`. All variants are frozen and reject extra fields.

- `class _FailureEvidenceBase(BaseModel)` — Private base. (L10256)
- `class TimeoutEvidence` (L10267) — `kind="timeout"`; fields `operation, deadline_s, elapsed_s`.
- `class ForceEvidence` (L10284) — `kind="force"`; fields `joint_or_ee, measured_n, limit_n`.
- `class WorkspaceEvidence` (L10300) — `kind="workspace"`; fields `ee_name, measured_xyz, box_min, box_max`.
- `class PerceptionStaleEvidence` (L10318) — `kind="perception"`; fields `sensor_id, staleness_ms, threshold_ms`.
- `class CriticEvidence` (L10334) — `kind="critic"`; fields `critic_id, score, threshold`.
- `class ControllerEvidence` (L10350) — `kind="controller"`; fields `controller_name, state, detail`.
- `class SelfVerifyEvidence` (L10366) — `kind="selfverify"`; fields `check, expected, observed`.
- `class HumanEvidence` (L10382) — `kind="human"`; fields `actor, reason`.
- `class WamEvidence` (L10396) — `kind="wam"`; fields `horizon, discrepancy, wam_id`.
- `class ReasonerTimeoutEvidence` (L10412) — `kind="reasoner_timeout"`; fields `model, deadline_s, elapsed_s`.
- `class CollisionEvidence` (L10428) — `kind="collision"`; `horizon_step=-1` (`REACTIVE_HORIZON_STEP`) marks the kernel's reactive measured-state check rather than a predicted chunk step; `joint_positions_rad` is the FK'd configuration for that step, making a predicted stop adjudicatable. Maps to `FailureTrigger.KIND_COLLISION`.
  fields: `collision_kind: Literal["self"|"world"], link_a, link_b_or_object, horizon_step, min_distance_m, joint_positions_rad: list[float]`
  - `REACTIVE_HORIZON_STEP: ClassVar[int]` — the `-1` sentinel; use it instead of a literal.
  - `is_reactive` (property) — `True` when `horizon_step == REACTIVE_HORIZON_STEP`. (L10485)
- `class SuppressedSummaryEvidence` (L10490) — `kind="suppressed_summary"`; fields `window_s, kinds: list[int], severities: list[int], counts: list[int]`. A model-validator enforces the arrays stay parallel.
- `FailureEvidence: TypeAlias` (L10524) — Discriminated union over the twelve variants above.

**Pydantic models — perception event metadata**

Discriminated union backing the `metadata_json` field of `openral_msgs/msg/PromptStamped` when published on `/openral/perception/<kind>`. Discriminator is `kind`; decode via `pydantic.TypeAdapter(PerceptionEventMetadata).validate_json(...)`. New kinds mean new topics, not a schema bump.

- `class _PerceptionEventBase(BaseModel)` — Private base; carries `sensor_id`. (L10559)
- `class ObjectDetection2D(BaseModel)` (L10578) — Single 2D detection inside `ObjectsMetadata`; `det_id` is a stable per-detector/per-camera identity assigned at detection time, letting an object be de-duplicated without the 3D lift, then propagated into `DetectedObject.track_id`.
  fields: `label, confidence, bbox_xyxy, det_id: int = -1`
- `class MotionMetadata` (L10607) — `kind="motion"`; fields `magnitude, threshold, region_bbox`.
- `class ObjectsMetadata` (L10630) — `kind="objects"`; `frame_width`/`frame_height` make the pixel space of `bbox_xyxy` explicit so the voxel lifter can scale it to the camera's intrinsics resolution.
  fields: `detections: list[ObjectDetection2D], model_id, frame_width: int (>0), frame_height: int (>0)`
- `class OcrMetadata` (L10655) — `kind="ocr"`; fields `text, confidence, region_bbox`.
- `class SceneChangeMetadata` (L10673) — `kind="scene_change"`; fields `distance, threshold, metric`.
- `PerceptionEventMetadata: TypeAlias` (L10696) — Discriminated union over the four variants above.

**Pydantic models — reasoner tool calls**

Discriminated union over the closed palette of typed tool calls the reasoner can emit each tick. Discriminator is `tool`; decode via `pydantic.TypeAdapter(ReasonerToolCall).validate_json(...)`. All variants are frozen and reject extra fields so an LLM cannot smuggle ad-hoc ones onto the wire. The reasoner holds no direct actuation authority — it never publishes `ActionChunk` itself; `ExecuteRskillTool` dispatches indirectly via the action server, which gates through safety.

- `class _ReasonerToolBase(BaseModel)` — Private base; carries optional `rationale`. (L10937)
- `class ExecuteRskillTool` (L10972) — `tool="execute_rskill"`; fields `rskill_id: str` (min_length=1), `prompt: str` (default ""), `goal_params_json: str` (default ""), `deadline_s: float` (ge=0.0; default 0.0; 0 = use manifest latency budget), `patience_s: float | None` (default None; gt=0.0; task-adaptive execution ceiling override — None uses the reward model's `default_patience_s`), `progress_tolerance: float | None` (default None; ge=0.0; overrides the reward model's `plateau_tolerance` for a noisy critic — None uses the model default).
- `class ReloadGstPipelineTool` (L11022) — `tool="reload_gst_pipeline"`; fields `sensor_id, pipeline_yaml`.
- `class LifecycleTransitionTool` (L11046) — `tool="lifecycle_transition"`; `shutdown` is deliberately absent — that authority belongs to the safety supervisor. Canonical primitive for managing long-lived background services (slam_toolbox, RTAB-Map, perception trees), which are LifecycleNode peers, not rSkills.
  fields: `node, transition: Literal["configure"|"activate"|"deactivate"|"cleanup"]`
- `class EmitPromptTool` (L11070) — `tool="emit_prompt"`; the reasoner node publishes on `target_topic` itself via a per-topic publisher cache.
  fields: `target_topic` (must start with `/`), `text`, `metadata_json`
- `class WaitTool` — deliberate no-op; `tool="wait"`, no fields beyond `rationale`. Since the reasoner's tool choice is forced, this lets the LLM choose "observe and wait" instead of acting every tick. No actuation authority. (L11456)
- `class RecallObjectTool` — read-only query; `tool="recall_object"`; recalls an object from the scene-graph memory. No actuation authority. Dispatch is planned Phase 2, not yet in the live provider palette. (L11098)
  fields: `query` (free-text/label), `limit`
- `class ResolvePlaceTool` — read-only query; `tool="resolve_place"`; resolves a place/room/agent to a goal pose plus path. No actuation authority. Dispatch is planned Phase 2. (L11120)
  fields: `reference` ("the kitchen", "where I was standing")
- `class LocateInViewTool` — read-only query; `tool="locate_in_view"`; asks a live VLM detector whether an object is in the current frame (vs `recall_object`'s remembered objects). No actuation authority. (L11137)
  fields: `query` (concrete object noun(s)), `camera` (optional viewpoint id, default primary), `detector` (optional locator selector, default the deployment default)
- `class QuerySceneTool` — read-only query; `tool="query_scene"`; asks a scene VLM an open-ended question about the current frame, answer fed back as a re-prompt. Distinct from `locate_in_view`: returns free text, not boxes. (L11183)
  fields: `question` (open-ended scene-state question, min_length=1), `camera` (optional viewpoint id)
- `class QueryTaskProgressTool` — read-only query; `tool="query_task_progress"`; asks the reward monitor for a windowed progress/success assessment, fed back to drive the replanning ladder. Distinct from `query_scene`: returns normalized scalars, not free text. (L11216)
  fields: `window_s` (seconds of recent frames to assess, > 0, default 8.0), `task` (optional instruction override)
- `MemorySection: TypeAlias = Literal[...]` — the five fixed sections of the self-maintained `MEMORY.md` core: `home_map`, `preferences`, `lessons`, `object_locations`, `open_tasks`.
- `class MemoryWriteTool` — write; the reasoner's first write-capable variant; `tool="memory_write"`; edits the advisory `MEMORY.md` via an explicit add/update/supersede/delete op. Writes the memory file only — no actuation authority. (L11261)
  fields: `op` (`add`/`update`/`supersede`/`delete`), `section: MemorySection`, `content` (required unless `delete`), `importance` (0–1, default 0.5), `target` (required for `update`/`supersede`/`delete`)
- `class MemorySearchTool` — read-only; `tool="memory_search"`; pages archived entries evicted from the bounded core back in. No actuation. (L11302)
  fields: `query` (min_length=1), `section: MemorySection | None`, `limit` (1–100, default 5)
- `is_collective_target(text) -> bool` — True when `text` targets a set rather than one specific object (a quantifier or bare generic plural); shared by `GroundedSubtask`'s validator and the reasoner node's runtime execute gate. (L11338)
- `_COLLECTIVE_TARGET_RE: re.Pattern[str]` — Backing regex for `is_collective_target`. (L11332)
- `class GroundedSubtask` — One subtask bound to exactly one specific object; a validator forbids a collective `object_ref`/`text` and requires `text` to name `object_ref`, so "the first batch of objects" isn't representable. (L11354)
  fields: `object_ref: str` (min_length=1), `text: str` (min_length=1)
  - `render(self) -> str` — The instruction string handed to `MissionState` / the skill. (L11409)
- `class DecomposeMissionTool` — task-ledger write; `tool="decompose_mission"`; the typed path for a playbook to write the deterministic `MissionState` — empty `target_task_id` replaces the whole queue, a set one flat-splices into that blocked task. Edits the S2 task ledger only, no actuation authority. (L11414)
  fields: `subtasks: list[GroundedSubtask]` (min_length=1), `target_task_id: str` (default `""`)
  - `rendered_subtasks(self) -> list[str]` — The ordered subtask instruction strings for `MissionState`. (L11451)
- `ReasonerToolCall: TypeAlias` — Discriminated union over the thirteen variants above.

**Module-level functions (Layer 0)**

- `def control_modes_for_representation(rep: ActionRepresentation) -> set[ControlMode]` (L5260) — Maps a VLA's declared `ActionRepresentation` to the `ControlMode`s it drives; single source of truth for the reasoner's deploy-path palette gate (a skill is offered only when the target robot advertises every returned mode).
- `SIM_EXECUTABLE_CONTROL_MODES: frozenset[ControlMode]` (L5322) — Canonical set of `ControlMode`s the default sim HAL action-packers can execute; single source of truth for the reasoner's sim-mode palette gate, pinned to the packers by a lockstep test. Excludes modes that are decoded but never pack-executed (would E-stop mid-run) or have no sim controller.
- `def canonical_slots_for_representation(rep: ActionRepresentation, *, dim: int, description: RobotDescription) -> list[ActionSlot] | None` (L5334) — Builds the canonical `ActionSlot` layout the skill_runner dispatches a representation-only `ActionContract` through. Joint representations return `None` (legacy whole-vector path). Raises `ROSConfigError` when the representation needs an EE the robot lacks, or `dim` is too small.
- `def task_space_compatible(skill_space: TaskSpace, robot: RobotDescription, *, hal_mode: Literal["sim", "real"] = "real") -> TaskSpaceMatch` (L5712) — DRAFT cross-layer gate subsuming today's implicit embodiment-tag/dim/adapter wiring; checks control-mode executability plus EE existence and joint-width bounds, returning `ok` + a reason per incompatibility. Wired warn-only so far.
- `def scene_family(task_id: str) -> str` — Reduces an `evaluated_tasks` entry to its scene-family key (leading token before any `/`). (L5863)
- `SCENE_FAMILY_TASK_SPACE: dict[str, SceneTaskSpace]` — Single source of truth for the control interface each scene-adapter family executes, keyed by `scene_family(evaluated_task)`. (L5884)
- `def scene_task_space_compatible(family: str, skill_space: TaskSpace) -> TaskSpaceMatch` — Third leg of the cross-layer gate: every `ControlMode` the rSkill emits must be in the scene family's executed set. Pairs with `task_space_compatible` to close the rSkill × robot × scene triangle. (L5968)

### `python/core/src/openral_core/loaders.py`
_Strict YAML loaders for the three scene tiers._

- `def load_scene_strict(path: str, expected: type[DeployScene | SimScene | BenchmarkScene]) -> DeployScene | SimScene | BenchmarkScene` (L36) — Loads `path` as exactly `expected`, rejecting other tiers with a redirect message naming the right CLI command, so a YAML one tier too rich isn't silently widened.
- `def load_benchmark_suite(path: str) -> list[BenchmarkScene]` (L148) — Loads a bare `list[BenchmarkScene]` from `benchmarks/<id>.yaml`; per-scene validation runs here, suite-level invariants do not — call `raise_on_invalid_suite` separately. Raises `FileNotFoundError`/`ROSConfigError`, never a bare `ValidationError`.
- `def raise_on_invalid_suite(scenes: list[BenchmarkScene], *, suite_id: str) -> None` (L222) — Enforces the suite invariants: non-empty, every `task.id` unique, and `robot_id`/`n_episodes`/`seed`/`metadata` identical across scenes (per-scene `success_key`/`max_steps` may differ). First violation wins; raises `ROSConfigError`.

### `python/core/src/openral_core/assets.py`
_The single resolver for robot description assets — URDF / MJCF / SRDF._

- `class AssetRefError(ValueError)` (L39) — A description-asset reference is malformed or cannot be resolved.
- `def resolve_asset(ref: str, kind: AssetKind, *, manifest_dir: Path | None = None) -> Path | None` (L43) — Resolves one asset `ref` to a concrete file path for the requested kind; one grammar replacing several prior scheme-specific loaders. Raises `AssetRefError` for an unresolvable or malformed ref.
- `_REPO_ROOT: Path` — Resolved repo root; the base `file:<relpath>` refs resolve against after the manifest dir. (L34)
- `_ROS2_DYNAMIC: str = "ros2://robot_description"` — The dynamic-detection marker scheme; `resolve_asset` returns `None` for it so the caller subscribes at runtime instead. (L35)
- `_RD_ATTR: dict[AssetKind, str]` — Maps `kind` to the `robot_descriptions` module attribute it reads for the `rd:<module>` scheme. (L36)

`HalParameters` also carries `can_bus_bindings: dict[str, str]`, mapping a `defaults` key to the role token naming which physical CAN bus fills it — matched by token, never position, since a CAN interface name is a host property, not a robot one.

### `python/core/src/openral_core/gpu.py`
_Torch-free GPU VRAM probe shared across layers, so the CLI and the reasoner ROS node share one `nvidia-smi` query instead of each carrying its own copy._

- `detect_gpu_vram_gb(field) -> float` (L17) — One `nvidia-smi --query-gpu=<field>` value for GPU 0 in GB; `0.0` on any failure so callers skip their check rather than block.

### `python/core/src/openral_core/can.py`
_Robot-agnostic SocketCAN transport discovery — a CAN-bus robot is invisible to USB/serial discovery, so this is the one place that reads the kernel's view of CAN links. Linux-only, dependency-free, needs no root, and never perturbs a running robot._

- `_ARPHRD_CAN: int = 280` — The `/sys/class/net/<if>/type` value marking a CAN link. (L40)
- `_CLASSIC_CAN_MTU: int = 16` — Classic CAN's MTU; CAN FD needs 72, so MTU is the FD indicator on hosts without `ip`. (L44)
- `SYSFS_NET: str = "/sys/class/net"` — Sysfs root for network devices, as a module constant so tests can point enumeration at a fixture tree. (L48)
- `class CanInterface(NamedTuple)` — One SocketCAN interface on the host; `state` is the field to check when a detected arm won't move. (L51)
- `enumerate_can_interfaces(*, sysfs_net=None) -> list[CanInterface]` — Lists CAN interfaces from sysfs, enriched with bitrate/FD/state via `ip -details -json link show` when available. (L208)
- `can_link_state(interface, *, sysfs_net=None) -> tuple[bool, str]` — Single-interface counterpart: `(is_up, reason)` naming the specific problem. (L274)
- `preflight_can_links(interfaces, *, hal_label, remedy="", sysfs_net=None) -> dict[str, str]` — The connect-time gate every CAN robot needs; reports every failing bus in one message and raises `ROSConfigError` otherwise. (L303)

### `python/core/src/openral_core/geometry.py`
_Shared rotation geometry — look-at/camera gaze poses plus planar yaw↔quaternion helpers, so every layer uses one implementation instead of duplicating them. Import-on-demand, not re-exported by `openral_core.__init__`, so schemas stay numpy-free on the fast CLI path._

- `ViewAxis` (TypeAlias = `Literal["-z", "+z", "+x"]`) — Camera forward-axis conventions: `"-z"` MuJoCo, `"+z"` ROS optical frames, `"+x"` body-frame forward.
- `_ZERO_NORM: float = 1e-9` — Below this vector norm, `look_at_quat_wxyz` treats the gaze direction as degenerate. (L48)
- `_PARALLEL: float = 0.999` — Above this `|cos|` between gaze and `up`, `look_at_quat_wxyz` swaps to the alternate up vector. (L49)
- `_QUAT_NORM_EPS: float = 1e-12` — Below this squared norm, a quaternion is treated as degenerate. (L50)
- `look_at_quat_wxyz(eye, target, *, up=(0,0,1), view_axis="-z") -> tuple[float, float, float, float]` — Unit quaternion orienting a camera at `eye` so `view_axis` points at `target`; never raises, falling back to a fixed orientation on degenerate input. (L187)
- `compute_gaze_pose(camera_xyz, target_xyz, *, frame_id="map", up=(0,0,1), view_axis="+z") -> Pose6D` — Full 6-DOF camera pose whose view axis hits `target_xyz`. (L250)
- `rotation_to_quat_wxyz(rot: 3x3) -> tuple[w, x, y, z]` — Matrix→quaternion conversion (Shepperd's method). (L173)
- `homogeneous_from_quat_xyz(translation, quat_xyzw) -> NDArray[np.float64]` — The canonical TF2 transform→4x4-matrix step; raises `ROSConfigError` on a degenerate rotation rather than guessing. (L53)
- `yaw_to_quat_xyzw(yaw: float) -> tuple[x, y, z, w]` — Unit quaternion (ROS order) for `Rz(yaw)`. (L109)
- `yaw_to_quat_wxyz(yaw: float) -> tuple[w, x, y, z]` — Same rotation in MuJoCo order. (L120)
- `quat_xyzw_to_yaw(x, y, z, w) -> float` — Planar yaw in `[-pi, pi]`, roll/pitch ignored. (L131)

### `python/core/src/openral_core/detection_tracker.py`
_Camera-space 2D detection tracker — pure, ROS-free, stateful; the 2D analog of `ObjectMemory`._

- `def aabb_iou_2d(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float` — 2D axis-aligned bbox IoU in `[0, 1]`; `0.0` for disjoint/degenerate boxes. (L21)
- `class DetectionTracker2D(*, iou_threshold=0.3, max_misses=3)` — Stateful per-camera 2D-IoU tracker assigning a stable `det_id` to each detection, so an object can be de-duplicated without a 3D lift. (L60)
  - `assign(self, detections: list[ObjectDetection2D]) -> list[ObjectDetection2D]` — Stamps each detection with a stable `det_id` via greedy same-label IoU association; retires tracks unseen for `max_misses` frames. (L90)

### `python/core/src/openral_core/exceptions.py`
_openral exception hierarchy — use these, do not invent new base classes._

- `class ROSError(Exception)` — Base class for all OpenRAL errors. (L41)
- `class ROSConfigError(ROSError)` — Bad manifest, missing weights, invalid YAML/URDF. (L48)
- `class ROSCapabilityMismatch(ROSError)` — Skill requires a capability the robot lacks. (L52)
- `class ROSRuntimeError(ROSError)` — General runtime failure. (L59)
- `class ROSQuantizationError(ROSRuntimeError)` — Quantization failed. (L63)
- `class ROSGPUMemoryError(ROSRuntimeError)` — Out of GPU memory. (L67)
- `class ROSSafetyViolation(ROSError)` — Safety constraint violated; never silently caught. (L74)
- `class ROSWorkspaceViolation(ROSSafetyViolation)` — Action outside allowed workspace. (L82)
- `class ROSForceLimitExceeded(ROSSafetyViolation)` — Contact force exceeds limit. (L86)
- `class ROSCollisionImminent(ROSSafetyViolation)` — Proposed motion would self-collide or strike a world obstacle. (L90)
- `class ROSEStopRequested(ROSSafetyViolation)` — Emergency stop requested. (L99)
- `class ROSPerceptionStale(ROSError)` — Sensor reading exceeds staleness deadline. (L106)
- `class ROSObjectNotInMemory(ROSPerceptionStale)` — A scene-graph query matched no node or only stale nodes; caller degrades to "unknown", never fabricates a pose. (L110)
- `class ROSPlanningError(ROSError)` — Reasoner failed to produce valid plan. (L123)
- `class ROSReasonerInvalidPlan(ROSPlanningError)` — LLM returned invalid plan. (L127)
- `class ROSFleetError(ROSError)` — Fleet-level / dispatch error. (L134)
- `class ROSDispatchUnavailable(ROSFleetError)` — No dispatcher available. (L138)
- `class ROSRskillGoalSatisfied(ROSError)` — Typed control-flow completion signal raised once a wrapped-ROS rSkill finishes; caught only at the runner's execute-callback boundary, which closes the goal with `success=True`. Not an error despite the base class. (L149)
- `class ROSDeadlineMissed(ROSFleetError)` — Cloud RTT exceeded skill deadline. (L142)
