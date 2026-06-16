# Layer 0 — Core Schemas & Exceptions

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Authoritative Pydantic v2 contracts (CLAUDE.md §1.3). Anything imported
from `openral_core.__init__` is API; while we are pre-publish the
on-disk schemas sit at `schema_version: "0.1"` and the surface evolves
in place (CLAUDE.md §1.6).

### `python/core/src/openral_core/schemas.py`
_openral schema v0 — normative Pydantic v2 contracts for all layers._

**Enums**

- `class EmbodimentKind(str, Enum)` — Top-level kinematic class. (L34)
  `HUMANOID, MANIPULATOR, BIMANUAL, QUADRUPED, MOBILE_BASE, MOBILE_MANIPULATOR, DRONE`
- `class JointType(str, Enum)` — URDF joint type. (L46)
  `REVOLUTE, PRISMATIC, CONTINUOUS, FIXED, FLOATING, PLANAR`
- `JointRole: TypeAlias = Literal[…]` — Structural classification of a `JointSpec` (ADR-0028a). (L57)
  `"arm", "base", "gripper", "torso", "leg", "head", "neck", "wheel", "unknown"`. Used by runner/safety/dataset-bridge to identify a channel without name-substring heuristics. Default `"unknown"` keeps legacy manifests loadable.
- `class ControlMode(str, Enum)` — Action space / control interface. (L81)
  `JOINT_POSITION, JOINT_VELOCITY, JOINT_TORQUE, JOINT_TRAJECTORY, CARTESIAN_POSE, CARTESIAN_DELTA, CARTESIAN_TWIST, BODY_TWIST, FOOT_PLACEMENT, GRIPPER_BINARY, GRIPPER_POSITION, DEX_HAND_JOINT, COMPOSITE_MODE`
- `const BODY_TWIST_DIM: int = 6` — Width of a BODY_TWIST / CARTESIAN_* twist row `(vx, vy, vz, wx, wy, wz)`. Single source for the HAL packers (`openral_hal.panda_mobile`, `openral_hal.sim_attached`) + the safety supervisor that validate 6-vec twist payloads; matches `Action.body_twist` (a 6-tuple). (L134)
- `class SensorModality(str, Enum)` — Physical sensing modality. (L147)
  `RGB, DEPTH, STEREO, IR, POINT_CLOUD, LIDAR_2D, IMU, FORCE_TORQUE, JOINT_STATE, TACTILE_VISION, TACTILE_ARRAY, AUDIO, GPS, BATTERY`
- `class Hand(str, Enum)` — End-effector laterality. (L166) `LEFT, RIGHT, NA`
- `class StateRepresentation(str, Enum)` — State vector format. (L590)
  `JOINT_POSITIONS, EEF_POS_AXISANGLE, EEF_POS_EULER, EEF_POS_QUAT, EEF_POS_AXISANGLE_GRIPPER`
- `class ActionRepresentation(str, Enum)` — Action vector format. (L600)
  `JOINT_POSITIONS, JOINT_VELOCITIES, DELTA_EE_6D_PLUS_GRIPPER, DELTA_EE_6D, CARTESIAN_POSE`
- `class RSkillAction(str, Enum)` — Closed vocabulary of high-level action verbs an rSkill can perform (ADR-0022); declared on `RSkillManifest.actions` and surfaced to the reasoner LLM tool palette so it can pick a skill by what it does. (L610)
  Manipulation primitives: `PICK, PLACE, PICK_AND_PLACE, TRANSFER, GRASP, RELEASE`; articulated / contact-rich: `OPEN, CLOSE, PUSH, PULL, SLIDE, INSERT, POUR, WIPE, ROTATE`; motion: `REACH`; mobile: `NAVIGATE`; social/expressive: `WAVE, SHAKE`; generalist marker (foundation / multi-task checkpoints): `GENERALIST`; perception producer: `DETECT` (ADR-0037, for `kind: "detector"` rSkills); scene VLM: `QUERY` (ADR-0047, for `kind: "vlm"` rSkills).
- `class QuantizationDtype(str, Enum)` — Weight numeric format. (L2135)
  `FP32, FP16, BF16, INT8, INT4, FP4_NVFP4`
- `class QuantizationBackend(str, Enum)` — Inference backend. (L2159)
  `PYTORCH, ONNX, TENSORRT, GGUF, MLX`
- `class RSkillState(str, Enum)` — Skill lifecycle. (L2231)
  `UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED, ERROR`
- `class RSkillLicensePosture(str, Enum)` — License posture (CLAUDE §7.4). (L2311)
  `APACHE_2_0, MIT, BSD, PERMISSIVE_RESEARCH, NVIDIA_NON_COMMERCIAL, NVIDIA_OPEN_MODEL, RLWRLD_NON_COMMERCIAL, PROPRIETARY, UNKNOWN` (NVIDIA_OPEN_MODEL = GR00T N1.7+, commercial OK — ADR-0046)
- `class RSkillRuntime(str, Enum)` — Manifest runtime hint. (L2325)
  `PYTORCH, ONNX, TENSORRT, TRT_LLM, VLLM, GGUF, MLX, JAX`
- `class PhysicsBackend(str, Enum)` — Sim backend. (L4598)
  `MUJOCO, MUJOCO_MJX, PYBULLET, ISAACSIM, GENESIS, MOCK`

**Pydantic models — robot manifest hierarchy**

- `class IntrinsicsPinhole(BaseModel)` — Pinhole camera intrinsics. (L177)
  fields: `width, height, fx, fy, cx, cy, distortion_model, distortion_coeffs`
- `scale_intrinsics_to(base, width, height) -> IntrinsicsPinhole` — Linearly rescale pinhole intrinsics to a new render resolution (fx/fy/cx/cy scale by width/height ratios; FOV and distortion preserved). ADR-0035: deploy-sim renders the same MuJoCo camera at `scene.observation_width/height`, so the HAL scales the manifest's nominal intrinsics to the render resolution before the depth back-projection — keeping the published camera model matched to what was rendered. Returns `base` unchanged when the target equals its resolution; raises `ValueError` on non-positive dims. (L205)
- `class SensorSpec(BaseModel)` — Generalizable sensor descriptor (all modalities). `sim_camera_name: str | None` (issue #191 Phase 3b, mirrors `JointSpec.sim_joint_name`) carries the MJCF camera name when it differs from the sensor `name` — `MujocoArmHAL.read_images` renders `sim_camera_name or name` (e.g. openarm's `base` sensor renders the MJCF `top` camera). (L259)
  fields: `name, modality, frame_id, parent_frame, static_transform_xyz_rpy, rate_hz, intrinsics, encoding, fov_h_deg, fov_v_deg, n_channels, range_min_m, range_max_m, accel_noise_density, gyro_noise_density, n_axes, tactile_grid, vla_feature_key, ros2_topic, ros2_msg_type, qos_profile, vendor, model, driver_pkg, metadata`
- `class SensorBundle(BaseModel)` — Multi-modal sensor group. (L328)
  fields: `bundle_name, sensors, sync, sync_tolerance_ms`
- `class JointSpec(BaseModel)` — URDF-derived joint spec. (L347)
  fields: `name, joint_type, parent_link, child_link, axis_xyz, origin_xyz, origin_rpy, position_limits, velocity_limit, effort_limit, has_position_sensor, has_velocity_sensor, has_torque_sensor, backlash_estimate, actuator_kind, sim_joint_name, role`. **`origin_xyz` / `origin_rpy`** (ADR-0030) are the fixed parent-link→joint transform (URDF `<joint><origin>`); with `axis_xyz` they let the kernel compute forward kinematics for self-collision. Default zeros; populated by the offline lowering tool only for robots that enable collision checking. **`role`** (ADR-0028a) is a `JointRole` literal that downstream code reads to identify gripper / base / arm DoFs structurally instead of substring-matching the joint name (default `"unknown"`). `sim_joint_name` (ADR-0025) carries the MJCF/MuJoCo joint name when it differs from the logical `name` — used by `openral_sim.backends.robocasa.{synthesize_laser_scan_2d,read_panda_mobile_base_velocity}`, `SimSensorBridge._compute_scan_ranges`, and `openral_hal.sim_attached.SimAttachedHAL.read_state` to look up `mj_name2id` without hardcoding robosuite/robocasa naming. `None` = "MJCF name matches `name`" (the common case for fixed-base manipulators). **Population contract:** a robot needs `sim_joint_name` populated only when (a) its sim adapter does `mj_name2id` on a joint name, AND (b) the loaded MJCF differs from `name`. Today: `panda_mobile` (robocasa auto-prefixes with `mobilebase0_*` + `robot0_*`). LIBERO / ManiSkill3 / aloha / so100_robosuite / ur5e / widowx preserve URDF names — populating `sim_joint_name` for those is a no-op. `openarm_robosuite` does its own hardcoded `mj_name2id` lookups in `env.py:309-318` (`openarm_{side}_joint{i}`) and is a candidate to refactor through this field.
- `class EndEffectorSpec(BaseModel)` — End-effector spec. (L418)
  fields: `name, kind, hand, n_dof, max_grip_force_n, max_payload_kg, workspace_radius_m, tactile_sensors, actuated`. **`actuated`** (ADR-0028a) defaults to `True`; set `False` for passive tools (inert flanges, kinematic-only mounts) so the safety kernel can reject chunks addressed at them.
- `class RobotCapabilities(BaseModel)` — Capability flags for skill compatibility. (L456)
  fields: `locomotion, can_lift_kg, has_dexterous_hands, has_tactile, has_force_control, has_vision, has_lidar, has_audio, bimanual, onboard_compute_tops, onboard_memory_gb, gpu_vram_gb, cuda_compute_capability, cuda_toolkit_version, tensorrt_version, gpu_supported_runtimes, gpu_supported_dtypes, supported_control_modes, supported_vla_embodiments, embodiment_tags`
- `class SafetyEnvelope(BaseModel)` — Constraints enforced by C++ safety kernel. (L520)
  fields: `workspace_box_min_xyz, workspace_box_max_xyz, no_go_zones, max_ee_speed_m_s, max_ee_accel_m_s2, max_joint_speed_factor, max_force_n, max_torque_nm, deadman_required, e_stop_topic, e_stop_qos, contact_force_threshold_n, cycle_time_violation_threshold_ms, human_in_loop_required`
- `class ObservationSpec(BaseModel)` — VLA observation config. (L661)
  fields: `state_key, state_shape, state_representation, image_flip_180`
- `class ActionSpec(BaseModel)` — VLA action config. (L678)
  fields: `dim, representation, control_freq_hz, chunk_size`
- `class ActionSlot(BaseModel)` — One contiguous slice of an rSkill's action vector (ADR-0028b). fields: `range, control_mode, discard, ee, frame, joint_names`. Per-mode field requirements enforced by `@model_validator`: cartesian needs ee+frame, body_twist needs frame only, gripper needs ee only, joint needs neither (joint_names optional, length must equal slot width when supplied). `discard=True` slots drop their slice silently — used for dataset artefacts (RoboCasa365 torso placeholder, paired gripper channels).
- `class ActionContract(BaseModel)` — Per-rSkill action-vector contract (ADR-0019 + ADR-0028b). fields: `dim, representation, slots`. When `slots` is set, every index in `[0, dim)` is covered by exactly one `ActionSlot` (`@model_validator` rejects gaps + overlaps + over-range slots). When `slots is None`, the legacy single-Action JOINT_POSITION path applies (back-compat). Manifests carrying `slots` are exempt from the ADR-0028a `dim <= len(robot.joints)` invariant — the slot decoder gives a per-slice typed contract.
- `class SphereShape(BaseModel)` — Sphere collision primitive; discriminator `shape="sphere"`, field `radius_m (>0)`. (ADR-0030, L835)
- `class CapsuleShape(BaseModel)` — Capsule collision primitive (segment along local +Z swept by a radius); discriminator `shape="capsule"`, fields `radius_m (>0), length_m (>=0)`. (ADR-0030, L853)
- `CollisionShape: TypeAlias = CapsuleShape | SphereShape` — Discriminated union of convex collision primitives (discriminator `shape`); mesh shapes excluded so the allocation-free kernel checks only analytic convex volumes. (ADR-0030, L882)
- `class LinkCollisionGeometry(BaseModel)` — One convex collision volume attached to a robot link; fields `link_name, shape: CollisionShape, origin_xyz_rpy`. Lowered, kernel-facing form (hand-authored or emitted by the offline lowering tool from MJCF/URDF). (ADR-0030, L894)
- `class RobotDescription(BaseModel)` — Top-level robot manifest, one per robot. (L1121)
  fields: `name, embodiment_kind, urdf_path, base_frame, odom_frame, map_frame, joints, end_effectors, sensors, sensor_bundles, capabilities, safety, ros2_namespace, middleware, onboard_compute, sdk_kind, hal, observation_spec, action_spec, sim, scene_defaults, urdf_root_frame, static_base_to_urdf_root_xyz_rpy, footprint_radius, base_kinematics, collision_geometry, allowed_collision_pairs, srdf_path, footprint_polygon`. **`collision_geometry: list[LinkCollisionGeometry]`** + **`allowed_collision_pairs: list[tuple[str, str]]`** + **`srdf_path: str | None`** (ADR-0030) carry the per-link collision primitives and the self-collision allowed-collision matrix the safety kernel consumes; all default empty/`None` and `joints` stays normative for the kinematic chain (URDF/SRDF add geometry + ACM only). `urdf_root_frame` + `static_base_to_urdf_root_xyz_rpy` (ADR-0027) configure `sim_e2e.launch.py`'s generic robot_state_publisher block — set `urdf_path` (filesystem path OR `python:<module>:<attribute>` reference resolved against the upstream `robot_descriptions` package), and when the URDF root differs from `base_frame` declare `urdf_root_frame` + `static_base_to_urdf_root_xyz_rpy: [x,y,z,roll,pitch,yaw]` so the launch publishes the bridging static transform. `footprint_radius: float | None` (>0) + `base_kinematics: Literal["differential","holonomic","omni","ackermann"] | None` (ADR-0025) drive the generic Nav2 bringup (see `nav2_param_overrides`). `footprint_polygon: list[tuple[float, float]] | None` — optional base-frame XY polygon vertices (metres, CCW); when set, draws the true base outline on the SLAM occupancy grid instead of the `footprint_radius` circle (ADR-0025).
  - `scene_defaults: SceneDefaults | None = None` — Optional scene-level defaults (top-camera POV, etc.) consumed by the MJCF composers as the fallback when an environment does not pin its own values.
  - `validate_for_e2e_pipeline(self) -> None` — Assert this manifest carries every field the e2e ROS graph (`openral deploy sim` → C++ safety kernel) needs: every actuated joint must have `position_limits`, `velocity_limit`, and `effort_limit` set. Raises `ROSConfigError` listing every missing field at once — used by `sim_e2e.launch.py` so a misshapen manifest fails at launch-parse time, not later in the HAL's first actuation tick. Pure validation; for synthesis of the kernel `EnvelopeIntersection` use `openral_safety.envelope_loader.compute_intersection(robot, skill=None)`.
  - `lidar_sensor(self) -> SensorSpec | None` [@property] — First declared `lidar_2d` `SensorSpec` (beam count `n_channels`, `range_min_m`/`range_max_m`, `rate_hz`), or None. Single source of truth for the synthetic `/scan` envelope: `openral deploy sim` (`deploy_sim._scan_params_from_description`) forwards it as HAL `scan_*` ROS params and `SimSensorBridge` (which owns `/scan` for the manifest-driven node) reads the envelope from them, so neither hardcodes a scan envelope. ADR-0025.
  - `nav2_param_overrides(self) -> dict[str, str]` — Nav2 param substitutions derived from `footprint_radius` (→ `robot_radius` **and** costmap `inflation_radius` = `footprint_radius` + `NAV2_INFLATION_CLEARANCE_M`, kept ≥ the inscribed/circumscribed radius Nav2 derives from the footprint) + `base_kinematics` (→ MPPI `motion_model`). `{}` for fixed-base arms. `nav2.launch.py` `RewrittenYaml`-rewrites the shared base param file with these so one base file serves any mobile base — no hand-vendored per-robot Nav2 yaml. ADR-0025.
- `class GripperReadMode(str, Enum)` — How `MujocoArmHAL` reports the gripper qpos. (ADR-0023) Values: `SUM_OVER_SCALE` (Franka parallel — normalised to `[0,1]`), `AFFINE_LOW_HIGH` (SO-100 revolute Jaw — normalised to `[0,1]`), `PASSTHROUGH` (Aloha prismatic / OpenArm revolute — raw qpos in MJCF units).
- `class GripperWriteMode(str, Enum)` — How `MujocoArmHAL` maps an Action's gripper value to `ctrl`. (ADR-0023 bimanual amendment) Values: `NORMALISED` (`[0,1]` → `ctrl_range`), `PASSTHROUGH` (raw → `ctrl`; MuJoCo clips).
- `class SimGripperDescription(BaseModel)` — Gripper wiring inside a MuJoCo MJCF. (ADR-0023)
  fields: `joint, ctrl_range, qpos_addrs, qpos_scale, read_mode, write_mode, actuator_index, mirror_actuator_index`
- `class SimDescription(BaseModel)` — Optional `RobotDescription.sim` block holding MuJoCo wiring for `MujocoArmHAL.from_description`. (ADR-0023)
  fields: `mjcf_uri, floating_base, joint_qpos_addr, joint_qvel_addr, actuator_index, grippers, settle_steps_default, keyframe_index, seed_ctrl_from_qpos`
- `class HalEntrypoints(BaseModel)` — `RobotDescription.hal` block: the robot's simulation + real-hardware HAL import strings, resolved by `openral_hal.build_hal`. (ADR-0031)
  fields: `sim: str | None` (null → derive `MujocoArmHAL.from_description` when a `sim:` block exists), `real: str | None` (null → simulation-only robot), `parameters: HalParameters` (per-robot HAL construction defaults; ADR-0029)
- `class HalParameters(BaseModel)` — `RobotDescription.hal.parameters` block: per-robot HAL construction defaults (serial `port`, `robot_ip`, …) merged into the constructor by `openral_hal.build_hal` (explicit `transport` wins; unaccepted keys dropped), so a parameterised robot needs no bespoke lifecycle subclass. Empty by default. (ADR-0029, issue #191)
  fields: `defaults: dict[str, object]`
- `class TopCameraDefaults(BaseModel)` — Default placement for the scene-level "top" / "base" camera consumed by sim backends that render an overview camera. (L853)
  fields: `pos: tuple[float, float, float], target: tuple[float, float, float], fovy: float (gt=0, lt=180)`
  - Replaces the dataset-specific `_DEFAULT_TOP_CAMERA_*` module-level constants previously hard-coded in `openral_sim.backends.openarm_robosuite._assets`. Backend YAML overrides (`scene.backend_options.top_camera_*`) still win — this submodel is the default fed to the composer.
- `class SceneDefaults(BaseModel)` — Per-robot scene rendering defaults consulted when the scene YAML does not override them. Fields: `top_camera: TopCameraDefaults | None`, `composition: SceneComposition | None`. (L918)
- `class SceneComposition(BaseModel)` — Declarative MJCF scene composition (issue #191 Phase 3b). `composer: "module:fn"` returning `(xml, meshdir)` + `params: dict`. The manifest-driven `ManifestHALLifecycleNode._create_hal` calls the composer and threads the composed MJCF in as the HAL's `mjcf_path` — replaced openarm's bespoke `_create_hal` tabletop splicing.
  fields: `composer: str`, `params: dict[str, object]`
  fields: `top_camera: TopCameraDefaults | None = None`
  - First consumer is the `openarm_tabletop_pnp` MJCF composer (`openral_sim.backends.openarm_robosuite._assets.compose_openarm_tabletop_mjcf`). Future scenes can extend this submodel as new defaults are pulled out of backend hardcodes.

**Pydantic models — runtime snapshots**

- `class JointState(BaseModel)` — Real-time joint state snapshot. (L1531)
  fields: `name, position, velocity, effort, stamp_ns`
- `class Pose6D(BaseModel)` — 6D pose (position + xyzw quaternion). (L1549)
  fields: `xyz, quat_xyzw, frame_id`
- `class DetectedObject(BaseModel)` — Object detection. (L1563)
  fields: `label, confidence, pose, bbox_3d, track_id`
- `class WorldCollisionPrimitive(BaseModel)` — A placed convex obstacle in the world (world-frame analogue of `LinkCollisionGeometry`); fields `shape: CollisionShape, pose: Pose6D, object_id: str | None`. (ADR-0030, L1371)
- `class OccupancyGridRef(BaseModel)` — Reference to a 2D occupancy grid for mobile-base world-collision (mirrors `nav_msgs/OccupancyGrid` metadata); fields `frame_id, resolution_m (>0), width (>=0), height (>=0), origin: Pose6D, data_topic`. (ADR-0030, L1395)
- `class WorldState(BaseModel)` — Snapshot consumed by Reasoner and Skills. (L1747)
  fields: `stamp_ns, joint_state, base_pose, base_twist, ee_poses, contact_forces, images, image_frames, point_clouds, tactile, detected_objects, battery_pct, diagnostics, collision_primitives, occupancy_grid`
  - **collision_primitives / occupancy_grid (added ADR-0030)** — `list[WorldCollisionPrimitive]` (default empty) + `OccupancyGridRef | None` (default `None`): the bounded world surface the kernel's world-collision phase checks robot links against; an absent/stale world is treated as unavailable (fail-closed).
  - **image_frames (added ADR-0010)** — `dict[str, SensorFrame] | None`. Optional in-process frame carrier for no-ROS deployments; default `None` keeps the existing `images: dict[str, str]` topic-ref path unchanged.

#### Persistent spatial memory — scene graph (ADR-0038)

_Advisory, queryable Layer-2 world model the S2 Reasoner consults to recall where objects/places/agents are. Never a safety input (the kernel gates only on the ADR-0030 geometric world). Poses anchored in the tf2 `map` frame._

- `class SpatialNodeKind(str, Enum)` — `OBJECT | PLACE | ROOM | AGENT`. (ADR-0038)
- `class SpatialRelationKind(str, Enum)` — `CONTAINS | AT_PLACE | TRAVERSABLE_TO | ON | NEAR`. (ADR-0038)
- `class SpatialNode(BaseModel)` — A typed scene-graph node; superset of `DetectedObject` for `kind=OBJECT`. fields `node_id, kind, pose: Pose6D, label, confidence, bbox_3d, embedding_ref, is_container, occludes_contents, first_seen_ns, last_seen_ns, observation_count`. Validators: `last_seen_ns >= first_seen_ns`; `occludes_contents` requires `is_container`. (ADR-0038)
- `class SpatialEdge(BaseModel)` — Directed relation; fields `src, dst, kind: SpatialRelationKind`. (ADR-0038)
- `class SceneGraph(BaseModel)` — Persistent scene-graph memory; fields `schema_version="0.1", nodes: list[SpatialNode], edges: list[SpatialEdge]`. Validators: unique `node_id`; every edge references an existing node. (ADR-0038)
- `class RecallObjectQuery(BaseModel)` — Read-only object recall; fields `text, label, near: Pose6D | None, max_age_ns, limit`. Validator: at least one of `text` / `label` non-empty. (ADR-0038)
- `class ApproachViewpoint(BaseModel)` — Camera-facing standoff goal; fields `pose: Pose6D, standoff_m (>0), camera_frame_id`. (ADR-0038)
- `class RecallObjectMatch(BaseModel)` — One ranked recall; fields `node_id, label, pose: Pose6D, score, last_seen_ns, approach: ApproachViewpoint | None, inside_container_id: str | None`. (ADR-0038)
- `class RecallObjectResult(BaseModel)` — `matches: list[RecallObjectMatch]` (empty = unknown → caller raises `ROSObjectNotInMemory`). (ADR-0038)
- `class ResolvePlaceQuery(BaseModel)` — Resolve a place/room/agent reference; fields `reference, kind: SpatialNodeKind | None`. (ADR-0038)
- `class ResolvePlaceResult(BaseModel)` — fields `node_id, goal: Pose6D, path_node_ids: list[str]` (a `traversable_to` path). (ADR-0038)
- `class Action(BaseModel)` — Action step or chunk produced by a Skill. (L551)
  fields: `control_mode, horizon, joint_targets, joint_velocities, joint_torques, cartesian_pose, cartesian_delta, cartesian_twist, body_twist, foot_placements, gripper, dex_hand_joints, confidence, stamp_ns, ee_name, frame_id, safety_overrides`
- `class QuantizationConfig(BaseModel)` — Quantization recipe. (L644)
  fields: `dtype, backend, per_channel, calibration_dataset, extra`
- `class DeviceInfo(BaseModel)` — Host compute snapshot. (L668)
  fields: `device_str, gpu_memory_bytes, cuda_compute_capability, cpu_count, arch`
- `class RSkillInfo(BaseModel)` — Skill runtime state snapshot. (L720)
  fields: `name, version, state, weights_loaded, quantized, warmed_up, embodiment_tags, role, latency_budget_ms, last_inference_ms, error_msg, stamp_ns`

**Pydantic models — skill packaging (rSkill)**

- `class RSkillLatencyBudget(BaseModel)` — Per-stage latency budget. (L799)
  fields: `per_chunk_ms, warmup_ms, load_ms`
- `class SensorRequirement(BaseModel)` — One sensor an rSkill needs the robot to provide. (L980)
  fields: `modality, vla_feature_key, min_width, min_height, count`
- `class ControlModeSemantics(BaseModel)` — Action-space semantics on each `ActuatorRequirement` (rSkill self-containment audit, Gap 2). (L1180)
  fields: `mode: Literal["absolute","delta"], gripper_convention, joint_order, reference_frame`
  - Cross-validator on `ActuatorRequirement`: gripper kinds REQUIRE `gripper_convention`; cartesian kinds REQUIRE `reference_frame`; other kinds forbid both.
- `GripperConvention` (TypeAlias = Literal[...]) — Closed gripper-action encoding set. Members: `normalized_open_unit, normalized_open_symmetric, binary_close_one, raw_joint_rad, width_meters`. (L1147)
- `class ActuatorRequirement(BaseModel)` — One actuator slot an rSkill emits actions for (ADR-0013). (L1198)
  fields: `kind, n_dof, vla_action_key, control_mode_semantics`
  - `kind` reuses `ControlMode`; `n_dof` / `vla_action_key` auto-fill from the robot YAML for canonical embodiments, REQUIRED on the manifest for the `"custom"` hatch.
  - `control_mode_semantics` is REQUIRED per the rSkill self-containment audit (Gap 2): declares absolute-vs-delta and (when applicable) gripper convention / reference frame.
- `class EmbodimentExtra(BaseModel)` — Sensor + actuator surface for the `"custom"` embodiment hatch (ADR-0013). (L1304)
  fields: `sensors: list[SensorRequirement] (≥1), actuators: list[ActuatorRequirement] (≥1)`
- `class RSkillProcessors(BaseModel)` — Explicit lerobot `PolicyProcessorPipeline` artefact pointers (rSkill self-containment audit, Gap 1 + Gap 3). (L1402)
  fields: `preprocessor_uri, postprocessor_uri`
  - Per-file URI shape `hf://owner/repo[@rev]/path/to/file.ext` (file tail REQUIRED — bare repo URIs are the implicit-snapshot shape we deliberately replaced).
  - Cross-validator rejects identical pre/post URIs.
- `class RosIntegration(BaseModel)` — Wiring for a wrapped ROS 2 action / service (ADR-0024). Required when `RSkillManifest.kind in {"ros_action", "ros_service"}`; forbidden otherwise. (L2783)
  fields: `package, interface_type, interface_name, result_trajectory_field, default_goal_json, ros_dependencies`
  - `result_trajectory_field is None` → result-only mode (Nav2 shape); set → trajectory mode (MoveIt shape, adapter replays one waypoint per `step()`).
  - `default_goal_json` validator round-trips the literal through `json.loads` and rejects non-dict payloads.
- `class DetectorEngine(str, Enum)` — Backend selector for `kind: "detector"` rSkills (ADR-0037 2026-06-12 amendment): `RTDETR_ONNX = "rtdetr_onnx"`, `VLM_SIDECAR = "vlm_sidecar"`, `ZEROSHOT_HF = "zeroshot_hf"`. Set on `DetectorContract.engine` to disambiguate backends that share a `runtime` (the VLM sidecar and the in-process Transformers zero-shot detector are both `runtime: pytorch`); `None` keeps the legacy `runtime`-keyed dispatch.
- `class DetectorMode(str, Enum)` — Invocation mode of a `kind: "detector"` rSkill (ADR-0051), orthogonal to `DetectorEngine`: `CONTINUOUS = "continuous"` (always-on background producer → `WorldState.detected_objects`; reasoner reads it passively, never prompts it; not ExecuteSkill-dispatchable) and `ON_DEMAND = "on_demand"` (prompted open-vocab locator surfaced via the `locate_in_view` tool). Cleanly separates open-vocabulary from prompting: continuous detectors cover a fixed bank the reasoner reads for free; the on-demand locator handles the long tail.
- `class DetectorContract(BaseModel)` — Manifest contract for `kind: "detector"` rSkills (ADR-0037). Required when `RSkillManifest.kind == "detector"`; forbidden otherwise. Frozen, `extra="forbid"`. (L2878)
  fields: `labels: list[str]` (min_length=1; class-label list indexed by model class-id), `input_size: tuple[int, int]` (width × height, both > 0; default (640, 640)), `score_threshold: float` (ge=0.0 le=1.0; default 0.5), `engine: DetectorEngine | None` (default None; explicit backend selector — ADR-0037 2026-06-12 amendment), `mode: DetectorMode` (default `continuous`; invocation mode — ADR-0051).
- `class RewardContract(BaseModel)` — Manifest contract for `kind: "reward"` rSkills (ADR-0057; Robometer-4B reward monitor). Required when `RSkillManifest.kind == "reward"`; forbidden otherwise. Frozen, `extra="forbid"`. fields: `progress_range: tuple[float, float]` (default (0.0, 1.0); validated max > min), `success_threshold: float` (ge=0.0 le=1.0; default 0.5), `preference: bool` (default False), `frame_window_s: float` (> 0; rolling-buffer horizon), `target_fps: float` (> 0; sampling rate), `num_bins: int` (> 0; default 100; discrete-mode progress bins → normalized [0,1]), `instruction_required: bool` (default True). A reward monitor is a pure perception consumer: no actuators, no action/state contract; its progress/success signal is advisory-only.
- `class RSkillManifest(BaseModel)` — `rskill.yaml` manifest (`schema_version="0.1"`; pre-publish surface — ADR-0013 / ADR-0022 / ADR-0024 / ADR-0037 each extended it in place without bumping). (L2931)
  fields: `schema_version, name, version, license, role, kind, model_family, embodiment_tags, embodiment_extra, capabilities_required, sensors_required, actuators_required, runtime, quantization, weights_uri, chunk_size, latency_budget, min_vram_gb, fallback_skill_id, benchmarks, paper_url, dataset_uri, source_repo, description, actions, objects, scenes, processors, image_preprocessing, state_contract, action_contract, n_action_steps, ros_integration, detector`. ADR-0019 amendment: `action_contract` (mirrors `state_contract`) declares the per-checkpoint action dim consumed by the dataset bridge. ADR-0022 amendment: `description` is now REQUIRED (was optional) and three new fields surface skill semantics to the reasoner LLM tool palette — `actions: list[RSkillAction]` (closed-vocabulary, REQUIRED; min_length enforced per-kind in `_check_kind_consistency`), `objects: list[str]` (free-form discriminative keywords), `scenes: list[str]` (free-form). ADR-0024 amendment: `kind: RSkillKind` is REQUIRED (no default); `model_family` and `weights_uri` became optional and are gated on `kind == "vla"`; new optional `ros_integration: RosIntegration | None` block. ADR-0037 amendment: new `kind: "detector"` value + optional `detector: DetectorContract | None` field (required iff `kind == "detector"`); `actuators_required` constraint relaxed from global min_length=1 to per-kind enforcement in `_check_kind_consistency` (detectors have no actuators). ADR-0047 amendment: new `kind: "vlm"` value for video-language scene-understanding models (role: s2; no actuators, no action/state contract, no detector block); `RSkillAction` gains `QUERY = "query"`. Perception amendment: `embodiment_tags` constraint relaxed from global `min_length=1` to per-kind enforcement in `_check_embodiment_tags_present` — perception kinds (`detector`/`vlm`/`reward`, `_PERCEPTION_KINDS`) are embodiment-agnostic and ship empty `embodiment_tags` (match-any); every other kind still requires ≥1 tag. ADR-0057 amendment: new `kind: "reward"` value for robotic reward/progress monitors (role: s2; no actuators, no action/state contract) + optional `reward: RewardContract | None` field (required iff `kind == "reward"`); `RSkillAction` gains `MONITOR = "monitor"`.
  - `from_yaml(cls, path: str) -> RSkillManifest` [@classmethod] — Load and validate an `rskill.yaml`.
  - `is_commercial_use_allowed: bool` [@property] — Derived from `license`: True for apache-2.0/mit/bsd, False otherwise (incl. unknown). Replaces V0's free-field `commercial_use_allowed`.
  - ADR-0013 cross-validators: `"custom" ∈ embodiment_tags ↔ embodiment_extra is not None`; when `"custom"` is present every `actuators_required` entry must carry both `n_dof` and `vla_action_key`.
  - rSkill self-containment audit cross-validator: `processors` REQUIRED when `model_family in {smolvla, pi05, xvla, diffusion, rldx}`; only `act` may omit it (legacy norm-stats-in-safetensors path).
  - ADR-0024 / ADR-0037 cross-validator (`_check_kind_consistency`): `kind == "vla"` requires `model_family` + `weights_uri` + ≥1 `actuators_required`, forbids `ros_integration` + `detector`. `kind in {"ros_action","ros_service"}` requires `ros_integration` + ≥1 `actuators_required`, forbids `model_family`/`weights_uri`/`processors`/`state_contract`/`action_contract`/`n_action_steps`/`image_preprocessing`/`starting_pose`/`detector`, pins `chunk_size == 1`. `kind == "detector"` (ADR-0037) requires `detector` + `weights_uri`, forbids `model_family`/`ros_integration`/`action_contract`/`state_contract`/`processors`/`n_action_steps`/`starting_pose`, requires empty `actuators_required`. `kind == "wam"` validates schema-side; the loader rejects it at resolve time.
  - The historical `policy_id` field was removed in favour of dispatching on `model_family` directly.
- `EmbodimentTag` (TypeAlias = Literal[...]) — Closed canonical robot embodiments matching `robots/*/robot.yaml`, plus `"custom"` escape hatch and `"mobile_base"` class tag for any planar-base robot (so base-only rSkills like Nav2 can target the class without naming each specific mobile platform). (L2596)
- `StateLayout` (TypeAlias = Literal[...]) — Closed set of per-checkpoint proprioception layouts: `smolvla_9d, human300_16d, gr1, rc365, simpler_widowx, simpler_google`. Names the SHAPE the checkpoint was trained on (field order, frame convention, gripper encoding, quaternion handedness). Per-robot SOURCE bindings live on `StateContractBindings`. ADR-0027. (The `pi0_16d` / `eef_pose_7d` / `base_pose_7d` robocasa sim-observation layouts were removed — no state-adapter assembler existed; recreate alongside an assembler when next needed.)
- `WRAPPED_TASK_SPACE_LAYOUTS: frozenset[StateLayout]` — Subset of `StateLayout` covering Cartesian/FK-derived composites: `{rc365, human300_16d}`. These layouts REQUIRE `StateContract.bindings`; the cross-validator on `StateContract` enforces this at manifest load. Joint-space layouts (`smolvla_9d`, `gr1`, `simpler_*`) are excluded — they're served verbatim from raw `JointState.position`. ADR-0027.
- `StateContractBindings` (Pydantic model) — Per-robot source bindings for an rSkill's `state_contract.layout`. Fields: `eef_frame: str | None`, `base_frame: str | None`, `world_frame: str | None = "map"`, `gripper_qpos_joints: list[str]`, `quaternion_convention: Literal["xyzw","wxyz"] = "xyzw"`. Symmetric to `ControlModeSemantics` on the action side. Required when `StateContract.layout` is in `WRAPPED_TASK_SPACE_LAYOUTS`, forbidden otherwise. ADR-0027.
- `BenchmarkName` (TypeAlias = Literal[...]) — Closed canonical benchmark suite ids matching `benchmarks/*.yaml`. Members: `aloha_insertion, aloha_transfer_cube, gr1_tabletop, libero_10, libero_goal, libero_object, libero_spatial, maniskill3_franka_pick_cube, maniskill3_pick_place, metaworld_mt50, pusht, robocasa_pnp, simpler_env_widowx`. (L2633)
- `ModelFamily` (TypeAlias = Literal["smolvla","pi05","xvla","act","diffusion","rldx","molmoact2","gr00t"]) — Closed VLA/policy family used by the eval/runner adapter dispatch. Required only when `RSkillManifest.kind == "vla"`. `gr00t` (NVIDIA Isaac GR00T) runs out-of-process via a ZMQ sidecar, reusing the `rldx` adapter — ADR-0046. (L2655)
- `RSkillKind` (TypeAlias = Literal["vla","wam","ros_action","ros_service","detector","vlm"]) — Discriminator selecting the loader / runner branch (ADR-0024 + ADR-0037 + ADR-0047). `"vla"` is today's learnable policy path; `"ros_action"` / `"ros_service"` route through `ROSActionRskill`; `"wam"` is reserved (loader rejects at resolve time); `"detector"` (ADR-0037) is a perception producer that runs an exported ONNX/TRT detection model and publishes `ObjectsMetadata` — emits no `Action`; `"vlm"` (ADR-0047) is a video-language model answering natural-language scene queries from camera frames — emits text, no actions/boxes, `role: s2` (reached via the read-only `query_scene` tool, not `ExecuteSkill`). (L2752)

**Pydantic models — skill benchmark results (`rskills/<id>/eval/<benchmark>.json`)**

- `class RSkillEvalSource(BaseModel)` — Provenance of a benchmark block. (L985)
  fields: `paper, arxiv, model_variant, evaluated_by, reproduced_locally, reproduction_planned, reproduction_cli, table, status`
- `class RSkillEvalBenchmark(BaseModel)` — Suite identity for a benchmark block. (L1023)
  fields: `name, dataset, protocol, robot, simulator`
- `class RSkillEvalResult(BaseModel)` — On-disk shape of `rskills/<id>/eval/*.json`. Carries an optional `trace_id: str | None` (32-hex) populated by `openral benchmark run` for offline cross-reference into the OTel trace tree. (L1042)
  fields: `schema_version, source, benchmark, eval_config, results, baselines`
  - `from_json(cls, path: str) -> RSkillEvalResult` [@classmethod] — Load and validate a single benchmark JSON. (L1083)

**Pydantic models — sim eval**

- `class SceneSpec(BaseModel)` — Physics scene declaration. (L1283)
  fields: `id, backend, assets_uri, observation_height, observation_width, cameras, backend_options`
- `class RoboCasaBackendOptions(BaseModel)` — Typed validator for `SceneSpec.backend_options` under the RoboCasa backend (ADR-0015). Prebuilt-vs-procedural XOR enforced by a `model_validator`. (L1324)
  fields: `mode, prebuilt_task, kitchen_style, layout_id, fixtures, spawn_objects, task_verb, robots, controller, horizon`
- `class TaskSpec(BaseModel)` — What the robot must achieve. (L4500)
  fields: `id, scene_id, instruction, max_steps: int | None, success_key: str | None, metadata`
- `class VLASpec(BaseModel)` — Policy / brain declaration. (L1029)
  fields: `id, weights_uri, device, runtime, quantization, deterministic, extra`
- `class SimEnvironment(BaseModel)` — **Runtime** (robot × scene × task × VLA) tuple. Composed at the CLI from a `SimScene` or `BenchmarkScene` YAML plus an `RSkillManifest` (`--rskill`); never loaded from YAML directly. (L2343)
  fields: `robot_id, scene, task, vla, base_pose, seed, n_episodes, record_video, save_dir, metadata`
  - `base_pose: Pose6D | None = None` — Per-rollout robot mounting pose in the scene's world frame; honoured by free-axis scenes only. See ADR-0002 (Amendment 3).
  - `model_post_init(_context: object) -> None` — Cross-field validation `task.scene_id == scene.id`. (L2389)
  - `from_yaml(cls, path: str) -> SimEnvironment` [@classmethod] — Deprecated shim; always raises `ROSConfigError` directing the caller to `SimScene.from_yaml(path) + --rskill rskills/<id>`. (L2405)
- `class BenchmarkMetadata(BaseModel)` — Provenance block required on every `BenchmarkScene`; fields: `paper: str`, `honest_scope: str`, optional `display_name: str | None`, optional `simulator: str | None`. The two optional fields (ADR-0042) become `RSkillEvalResult.benchmark.name` / `.simulator` when present; suite invariants treat the whole block as byte-identical across scenes. (L4687)
- `class DeployScene(BaseModel)` — Env-only scene for `openral deploy run`; carries `scene`, `robot_id`, and `base_pose`; no task or eval config. Rejects legacy `vla:` blocks. (L4711)
  - `from_yaml(cls, path: str) -> DeployScene` [@classmethod] — Load and validate a `DeployScene` YAML from disk. (L4734)
- `class SimScene(DeployScene)` — Extends `DeployScene` with `task`, `seed`, `n_episodes`, `record_video`, `save_dir`, `metadata`; cross-validates `task.scene_id == scene.id`; accepted by `openral sim run`. (L4743)
  - `from_yaml(cls, path: str) -> SimScene` [@classmethod] — Load and validate a `SimScene` (or `BenchmarkScene`) YAML from disk. (L4770)
- `class BenchmarkScene(SimScene)` — Extends `SimScene` with required `n_episodes`, `seed`, and `metadata: BenchmarkMetadata`; also requires `task.success_key` and `task.max_steps`; consumed by `openral benchmark`. (L4779)
  - `from_yaml(cls, path: str) -> BenchmarkScene` [@classmethod] — Load and validate a `BenchmarkScene` YAML; raises `ValidationError` if eval fields are missing. (L4808)
- `class ProtocolSpec(BaseModel)` — Standalone eval-protocol schema (ADR-0009). Retained as a public surface for ADR drafts and report tooling that quote a published protocol verbatim; never embedded in a benchmark suite (Task 10 of ADR-0041 flattened the per-scene fields onto `BenchmarkScene`; ADR-0042 then deleted the `BenchmarkSpec` wrapper entirely so a suite is now a bare `list[BenchmarkScene]`). (L1386)
  fields: `n_episodes, seeds, success_key, max_steps, min_reps`
  - `model_post_init(_context: object) -> None` — Cross-field validation: `len(seeds) >= n_episodes` and `min_reps <= n_episodes`. (L1427)

**Pydantic models — inference runner (ADR-0010)**

On-disk + runtime contracts for the hardware inference runner (`openral deploy --config R.yaml`), sibling of `SimEnvironment` / `openral sim run`. Schemas are additive — `SimEnvironment` / `RSkillEvalResult` / `BenchmarkScene` are untouched.

- `class FrameEncoding(str, Enum)` — How `SensorFrame` bytes are interpreted. (L557)
  `BGR8, RGB8, MONO8, DEPTH16, JPEG, PNG, CUDA_NV12, RAW`
- `class SensorFrame(BaseModel)` — Single sensor frame: metadata + optional inline / topic / handle payload. JSON-serializes the binary payload as base64. (L576)
  fields: `sensor_id, stamp_monotonic_ns, stamp_wall_ns, encoding, width, height, channels, data, topic, handle, metadata`
  - `_decode_data(cls, value: Any) -> bytes | None` [@field_validator("data", mode="before")] — Accept raw `bytes` or a base64-encoded `str` on JSON parse. (L614)
  - `_encode_data(self, value: bytes | None) -> str | None` [@field_serializer("data", when_used="json")] — JSON-serialize the binary payload as base64. (L632)
  - `model_post_init(_context: object) -> None` — Cross-field validation: exactly one of `(data, topic, handle)` must be set. (L637)
- `class SensorReaderBackend(str, Enum)` — Which `SensorReader` implementation a sensor uses. (L1691)
  `OPENCV_THREAD, ROS2_IMAGE, GSTREAMER`
- `class DeadlineOverrunPolicy(str, Enum)` — Behaviour when a tick exceeds `1 / rate_hz`. (L1706)
  `WARN, DROP, RAISE`
- `class SensorReaderConfig(BaseModel)` — Per-sensor reader backend + optional ROS-tee. (L1720)
  fields: `sensor_id, backend, backend_params, max_age_ms, publish_to_ros, publish_topic, publish_rate_hz`
  - `model_post_init(_context: object) -> None` — Cross-field validation: `publish_to_ros ↔ publish_topic`. (L1774)
- `class HalConfig(BaseModel)` — Which HAL adapter to instantiate + transport params (serial port / FCI URI / ROS namespace). (L1786)
  fields: `adapter, transport, params`
- `class RobotEnvironment(BaseModel)` — Full hardware deployment configuration; `openral deploy` artefact. Sibling of `SimEnvironment`. (L1819)
  fields: `robot_id, hal, sensors, task, vla, safety, rate_hz, thumbnail_hz, deadline_overrun_policy, max_ticks, save_dir, metadata` (`thumbnail_hz: float = 25.0`, ge 0 — per-camera dashboard thumbnail rate, 0 disables)
  - `model_post_init(_context: object) -> None` — Cross-field validation: unique sensor ids + `vla.weights_uri` must be a valid skill reference (bare name, `rskills/<name>`, or HF repo id — no `rskill://` scheme). (L1900)
  - `from_yaml(cls, path: str) -> RobotEnvironment` [@classmethod] — Load and validate a `RobotEnvironment` YAML from disk. (L1918)
- `class TickResult(BaseModel)` — One tick's record returned by `InferenceRunner.tick`. v2 (ADR-0010 amendment 1) adds five optional sim-only fields (`step_idx`, `episode_idx`, `reward`, `terminated`, `truncated`) and an optional `trace_context: str | None` (full W3C `traceparent` for the tick's `rskill.tick` span). All optional fields default to `None`; hardware ticks that don't carry sim metadata or a live trace context serialise byte-identically to v1 under `exclude_none=True`. (L1937)
  fields: `stamp_ns, tick_idx, sensors_ms, world_state_ms, inference_ms, safety_ms, hal_ms, tick_ms, chunk_index, safety_violations, action_applied, step_idx, episode_idx, reward, terminated, truncated`
- `class RunResult(BaseModel)` — Aggregated summary returned by `InferenceRunner.run`. (L1980)
  fields: `n_ticks, success, budget_violations, avg_inference_ms, p99_inference_ms, avg_tick_ms, p99_tick_ms, trace_id, save_dir, metadata`

**Pydantic models — failure evidence (ADR-0018 F3)**

Discriminated union backing the `evidence_json` field of `openral_msgs/msg/FailureTrigger`. Discriminator is the `kind` field (a `Literal[...]` on each variant); decode via `pydantic.TypeAdapter(FailureEvidence).validate_json(...)`. All variants are `frozen=True` and `extra="forbid"`.

- `class _FailureEvidenceBase(BaseModel)` — Private base, `frozen=True`, `extra="forbid"`. (L3064)
- `class TimeoutEvidence` (L3075) — `kind="timeout"`; fields `operation, deadline_s, elapsed_s`.
- `class ForceEvidence` (L3092) — `kind="force"`; fields `joint_or_ee, measured_n, limit_n`.
- `class WorkspaceEvidence` (L3108) — `kind="workspace"`; fields `ee_name, measured_xyz, box_min, box_max`.
- `class PerceptionStaleEvidence` (L3126) — `kind="perception"`; fields `sensor_id, staleness_ms, threshold_ms`.
- `class CriticEvidence` (L3142) — `kind="critic"`; fields `critic_id, score, threshold`.
- `class ControllerEvidence` (L3158) — `kind="controller"`; fields `controller_name, state, detail`.
- `class SelfVerifyEvidence` (L3174) — `kind="selfverify"`; fields `check, expected, observed`.
- `class HumanEvidence` (L3190) — `kind="human"`; fields `actor, reason`.
- `class WamEvidence` (L3204) — `kind="wam"`; fields `horizon, discrepancy, wam_id`.
- `class ReasonerTimeoutEvidence` (L3220) — `kind="reasoner_timeout"`; fields `model, deadline_s, elapsed_s`.
- `class CollisionEvidence` (L4694) — `kind="collision"`; fields `collision_kind: Literal["self"|"world"], link_a, link_b_or_object, horizon_step, min_distance_m`. Maps to `FailureTrigger.KIND_COLLISION = 10` (ADR-0030).
- `class SuppressedSummaryEvidence` (L3236) — `kind="suppressed_summary"`; fields `window_s, kinds: list[int], severities: list[int], counts: list[int]`. Model-validator enforces parallel arrays (raises `ROSConfigError`).
- `FailureEvidence: TypeAlias` (L3270) — Discriminated union over the twelve variants above. Module docstring shows the encode/decode pattern.

**Pydantic models — perception event metadata (ADR-0018 F6)**

Discriminated union backing the `metadata_json` field of `openral_msgs/msg/PromptStamped` when published on `/openral/perception/<kind>`. Discriminator is the `kind` field (a `Literal[...]` on each variant); decode via `pydantic.TypeAdapter(PerceptionEventMetadata).validate_json(...)`. All variants are `frozen=True` and `extra="forbid"`. New kinds = new topics, not a schema bump.

- `class _PerceptionEventBase(BaseModel)` — Private base; carries `sensor_id: str`. (L3304)
- `class ObjectDetection2D(BaseModel)` (L3323) — single 2D detection inside `ObjectsMetadata`; fields `label, confidence, bbox_xyxy`.
- `class MotionMetadata` (L3344) — `kind="motion"`; fields `magnitude, threshold, region_bbox`.
- `class ObjectsMetadata` (L5070) — `kind="objects"`; fields `detections: list[ObjectDetection2D], model_id, frame_width: int (>0), frame_height: int (>0)`. ADR-0035: `frame_width`/`frame_height` added to make the pixel space of `bbox_xyxy` explicit so the `VoxelFrustumLifter` can scale to the intrinsics resolution (CLAUDE.md §1.4). Producers (`ObjectsDetector`, `NvmmObjectsDetector`) populate both at detect-time.
- `class OcrMetadata` (L3386) — `kind="ocr"`; fields `text, confidence, region_bbox`.
- `class SceneChangeMetadata` (L3404) — `kind="scene_change"`; fields `distance, threshold, metric`.
- `PerceptionEventMetadata: TypeAlias` (L3427) — Discriminated union over the four variants above. Module docstring shows the encode/decode pattern.

**Pydantic models — reasoner tool calls (ADR-0018 F4)**

Discriminated union over the closed palette of typed tool calls the F4 reasoner can emit each tick. Discriminator is the `tool` field (a `Literal[...]` on each variant); decode via `pydantic.TypeAdapter(ReasonerToolCall).validate_json(...)`. All variants are `frozen=True` and `extra="forbid"` so an LLM cannot smuggle ad-hoc fields onto the wire. The reasoner holds **no** authority over actuation — it never publishes `ActionChunk`; `ExecuteRskillTool` is indirect (action goal on the F1 server which gates through F5 safety).

- `class _ReasonerToolBase(BaseModel)` — Private base; carries optional `rationale: str`. (L3455)
- `class ExecuteRskillTool` (L3480) — `tool="execute_rskill"`; fields `rskill_id, prompt, deadline_s`.
- `class ReloadGstPipelineTool` (L3508) — `tool="reload_gst_pipeline"`; fields `sensor_id, pipeline_yaml`.
- `class LifecycleTransitionTool` (L3532) — `tool="lifecycle_transition"`; fields `node, transition: Literal["configure"|"activate"|"deactivate"|"cleanup"]`. `shutdown` is intentionally absent — shutdown is the safety supervisor's authority per CLAUDE.md §6 Layer 6. **ADR-0025**: this is the canonical primitive for managing long-lived background services (slam_toolbox, RTAB-Map, perception trees) — they are LifecycleNode peers, not rSkills.
- `class EmitPromptTool` (L3556) — `tool="emit_prompt"`; fields `target_topic` (must start with `/`), `text`, `metadata_json`.
- `class RecallObjectTool` — **read-only query** (ADR-0039); `tool="recall_object"`; fields `query` (free-text/label), `limit`. Recalls an object from the ADR-0038 scene-graph memory; no actuation authority. Dispatch + result-return is ADR-0039 Phase 2 (not yet in the live provider palette).
- `class ResolvePlaceTool` — **read-only query** (ADR-0039); `tool="resolve_place"`; field `reference` ("the kitchen", "where I was standing"). Resolves a place/room/agent to a goal pose + path; no actuation authority. Dispatch is ADR-0039 Phase 2.
- `class LocateInViewTool` — **read-only query** (ADR-0043); `tool="locate_in_view"`; fields `query` (object to look for), `camera` (optional viewpoint id, default `""` = primary — camera-agnostic, not a hardcoded name), `detector` (ADR-0056 — optional on-demand locator selector / alias, default `""` = the deployment default; the reasoner routes to `/openral/perception/<detector>/locate_in_view`). Asks a live VLM detector if the object is in the CURRENT frame (vs `recall_object`'s *remembered* objects). No actuation authority — choosing a model does not grant it.
- `class QuerySceneTool` — **read-only query** (ADR-0047); `tool="query_scene"`; fields `question` (open-ended scene-state question, min_length=1), `camera` (optional viewpoint id, default `""`). Asks a scene VLM (Qwen3.5-4B) an open-ended question about the CURRENT frame for task-progress / success verification; dispatched via `/openral/perception/query_scene`, answer fed back as a re-prompt. Distinct from `locate_in_view` (localization → boxes): returns free text. No actuation authority.
- `class QueryTaskProgressTool` — **read-only query** (ADR-0057); `tool="query_task_progress"`; fields `window_s` (seconds of recent frames to assess, > 0, default 8.0), `task` (optional instruction override, default `""` → reuse the active goal). Asks the Robometer reward monitor for a quantitative windowed assessment of the CURRENT task; dispatched via `/openral/perception/query_task_progress`, the verdict (progress/success now + trends + `stalled`/`succeeded`) fed back as a re-prompt driving the replanning ladder. Distinct from `query_scene` (free text): returns normalized scalars. No actuation authority.
- `ReasonerToolCall: TypeAlias` — Discriminated union over the eight variants above (four actuation/effect + four ADR-0039/0043/0047 read-only query). Module docstring shows the encode/decode pattern.

**Module-level functions (Layer 0)**

- `def control_modes_for_representation(rep: ActionRepresentation) -> set[ControlMode]` (L2376) — ADR-0036. Maps a VLA's declared `ActionRepresentation` to the set of `ControlMode`s it drives (`JOINT_POSITIONS→{JOINT_POSITION}`, `JOINT_VELOCITIES→{JOINT_VELOCITY}`, `DELTA_EE_6D→{CARTESIAN_DELTA}`, `DELTA_EE_6D_PLUS_GRIPPER→{CARTESIAN_DELTA, GRIPPER_POSITION}`, `CARTESIAN_POSE→{CARTESIAN_POSE}`). Single source of truth for the reasoner's deploy-path palette gate (a skill is offered only when the target robot advertises every returned mode).
- `SIM_EXECUTABLE_CONTROL_MODES: frozenset[ControlMode]` (L2492) — ADR-0036 (amended 2026-06-04). Canonical set of `ControlMode`s the **default sim HAL action-packers** can execute, and the single source of truth for the reasoner's `hal_mode == "sim"` deploy-path palette gate: `{JOINT_POSITION, JOINT_VELOCITY, CARTESIAN_DELTA, GRIPPER_POSITION, BODY_TWIST, COMPOSITE_MODE}`. Mirrors `CONTROL_MODE_TO_UINT8` as a shared core constant (core is a dep of both reasoner and HAL). Pinned in both directions to the packers in `python/hal/src/openral_hal/sim_attached.py` (`pack_action_for_env`, `SimAttachedHAL._pack_with_composite_split`, and the `BODY_TWIST` direct-qpos path) by `tests/unit/test_sim_executable_modes_match_packers.py`. Excludes `JOINT_TORQUE` / `JOINT_TRAJECTORY` / `CARTESIAN_POSE` / `GRIPPER_BINARY` (decoded but never pack-executed → would E-stop mid-run) and `CARTESIAN_TWIST` / `FOOT_PLACEMENT` / `DEX_HAND_JOINT` (no sim controller).
- `def canonical_slots_for_representation(rep: ActionRepresentation, *, dim: int, description: RobotDescription) -> list[ActionSlot] | None` (L2408) — ADR-0036. Builds the canonical `ActionSlot` layout the skill_runner dispatches a representation-only `ActionContract` through. Joint representations → `None` (caller keeps the legacy whole-vector `JOINT_POSITION` path). `DELTA_EE_6D`/`CARTESIAN_POSE` → one cartesian slot `range=(0,5)` addressed at the primary EE (`description.end_effectors[0]`); `DELTA_EE_6D_PLUS_GRIPPER` adds a `GRIPPER_POSITION` slot `range=(6, dim-1)`. `EndEffectorSpec` has no explicit tf-frame field, so the EE `name` is used as the slot `frame`. Raises `ROSConfigError` when the representation needs an EE but `end_effectors` is empty, or when `dim` is too small (`DELTA_EE_6D`/`CARTESIAN_POSE` need `dim>=6`; `DELTA_EE_6D_PLUS_GRIPPER` needs `dim>=7`).

### `python/core/src/openral_core/loaders.py`
_Strict YAML loaders for the three scene tiers (ADR-0041)._

- `def load_scene_strict(path: str, expected: type[DeployScene | SimScene | BenchmarkScene]) -> DeployScene | SimScene | BenchmarkScene` (L36) — Load `path` as exactly `expected`; reject other tiers with `ROSConfigError` carrying a redirect message that names the right CLI command. `mypy --strict` overloads narrow the return to the requested concrete type. Centralises the rejection logic used by every scene-driven CLI loader (`openral deploy sim` → `DeployScene`, `openral sim run` → `SimScene`, `openral benchmark scene` → `BenchmarkScene`) so a YAML one tier too rich is not silently widened (e.g. a BenchmarkScene YAML passed to `openral sim run` would otherwise drop `n_episodes`/`metadata` on the floor). Raises `FileNotFoundError` for a missing path and `ROSConfigError` for a non-mapping YAML root, an extra-key mismatch, or a tier mismatch.
- `def load_benchmark_suite(path: str) -> list[BenchmarkScene]` (L148) — ADR-0042. Load a bare `list[BenchmarkScene]` from `benchmarks/<id>.yaml`. The suite id is the filename stem; the YAML root MUST be a list. Pre-ADR-0042 `{id, tasks, metadata}` dict shape is rejected with an explicit ADR-0042 redirect message naming the migration. Per-scene Pydantic validation runs here; suite-level invariants (uniformity, uniqueness, non-empty) are NOT enforced — call `raise_on_invalid_suite` separately so tests can construct invalid in-memory suites without touching disk. Raises `FileNotFoundError` for missing paths and `ROSConfigError` on every shape / validation failure (never a bare `ValidationError`).
- `def raise_on_invalid_suite(scenes: list[BenchmarkScene], *, suite_id: str) -> None` (L223) — ADR-0042. Free-function replacement for the deleted `BenchmarkSpec.model_post_init`. Enforces the five suite invariants: non-empty list, every `scenes[i].robot_id` non-`None`, every `task.id` unique within the suite, every `scenes[i].robot_id` / `n_episodes` / `seed` / `metadata` byte-identical across the list. Per-scene `task.success_key` and `task.max_steps` MAY differ (ManiSkill3 mixed-budget suite). `suite_id` is embedded in every error message so failures point back at the right `benchmarks/<id>.yaml`. First violation wins — no batched reporting. Raises `ROSConfigError` on any invariant violation.

### `python/core/src/openral_core/urdf_resolve.py`
_Shared URDF-path resolver — lifted from `sim_e2e.launch.py` (CLAUDE.md §1.13)._

- `resolve_urdf_path(value, *, repo_root=None) -> str | None` — Resolve a `RobotDescription.urdf_path` (`python:<module>:<attribute>` | absolute | repo-relative) to an on-disk file, else `None`. Shared by `sim_e2e.launch.py` and the offline collision-lowering tool (ADR-0030). _Being superseded by `assets.resolve_asset` (ADR-0057)._

### `python/core/src/openral_core/assets.py`
_The single resolver for robot description assets — URDF / MJCF / SRDF (ADR-0057)._

- `class AssetRefError(ValueError)` (L33) — A description-asset reference is malformed or cannot be resolved.
- `def resolve_asset(ref: str, kind: AssetKind, *, manifest_dir: Path | None = None) -> Path | None` (L41) — Resolve one asset `ref` to a concrete file path for the requested `kind` (`urdf`/`mjcf`/`srdf`). One grammar replacing `resolve_urdf_path`, `resolve_mjcf_uri`, plain-path SRDF, and `urdf_lowering._load_urdf_model`. Schemes: `rd:<module>` (upstream `robot_descriptions`, downloads on first use; xacro-only URDF → `AssetRefError` directing to `openral robot vendor-urdf`), `file:<relpath>` (manifest dir then repo root), `gym_aloha:<scene>` / `openarm:<variant>` / `menagerie:<model>` (sim-only MJCF loaders, lazy-imported; menagerie not yet wired), `ros2://robot_description` (URDF-only dynamic marker → returns `None`). Raises `AssetRefError` for every other unresolvable/malformed ref.

### `python/core/src/openral_core/exceptions.py`
_openral exception hierarchy — use these, do not invent new base classes._

- `class ROSError(Exception)` — Base class for all OpenRAL errors. (L43)
- `class ROSConfigError(ROSError)` — Bad manifest, missing weights, invalid YAML/URDF. (L50)
- `class ROSCapabilityMismatch(ROSError)` — Skill requires a capability the robot lacks. (L54)
- `class ROSRuntimeError(ROSError)` — General runtime failure. (L61)
- `class ROSInferenceTimeout(ROSRuntimeError)` — VLA inference exceeded latency budget. (L65)
- `class ROSQuantizationError(ROSRuntimeError)` — Quantization failed. (L69)
- `class ROSGPUMemoryError(ROSRuntimeError)` — Out of GPU memory. (L73)
- `class ROSSafetyViolation(ROSError)` — Safety constraint violated. **Never silently caught.** (L80)
- `class ROSWorkspaceViolation(ROSSafetyViolation)` — Action outside allowed workspace. (L88)
- `class ROSForceLimitExceeded(ROSSafetyViolation)` — Contact force exceeds limit. (L92)
- `class ROSCollisionImminent(ROSSafetyViolation)` — Proposed motion would self-collide or strike a world obstacle. (ADR-0030, L95)
- `class ROSEStopRequested(ROSSafetyViolation)` — Emergency stop requested. (L107)
- `class ROSPerceptionStale(ROSError)` — Sensor reading exceeds staleness deadline. (L114)
- `class ROSObjectNotInMemory(ROSPerceptionStale)` — A scene-graph query (`RecallObjectQuery` / `ResolvePlaceQuery`) matched no node or only stale nodes; caller degrades to "unknown" (may trigger active search), never fabricates a pose. (ADR-0038)
- `class ROSPlanningError(ROSError)` — Reasoner failed to produce valid plan. (L131)
- `class ROSReasonerInvalidPlan(ROSPlanningError)` — LLM returned invalid plan. (L135)
- `class ROSBTValidationError(ROSPlanningError)` — BehaviorTree XML failed BT.CPP v4 validation. (L139)
- `class ROSFleetError(ROSError)` — Fleet-level / dispatch error. (L146)
- `class ROSDispatchUnavailable(ROSFleetError)` — No dispatcher available. (L150)
- `class ROSRskillGoalSatisfied(ROSError)` — Typed control-flow completion signal raised by `ROSActionRskill._step_impl` once a wrapped-ROS rSkill (kind: ros_action / ros_service) has emitted its last waypoint (trajectory mode) or finished awaiting the wrapped action's result (result-only mode). Caught only at the `rskill_runner_node` execute-callback boundary; the runner closes the goal with `success=True`. NOT an error — inherits `ROSError` only to stay inside the OpenRAL exception surface. (L161)
- `class ROSDeadlineMissed(ROSFleetError)` — Cloud RTT exceeded skill deadline. (L154)

