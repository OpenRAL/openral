# Layer 1 — Hardware Abstraction (HAL)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/hal/src/openral_hal/protocol.py`
_Normative HAL protocol plus explicit optional lifecycle extensions._

- `class HAL(Protocol)` (L89) — Structural protocol every HAL adapter must satisfy.
  - attr `description: RobotDescription`
  - `connect() -> None` (L106) — Open connection to robot/sim.
  - `disconnect() -> None` (L116) — Close connection (idempotent).
  - `read_state() -> JointState` (L124) — Latest joint state snapshot (hot path).
  - `send_action(action: Action) -> None` (L138) — Forward action chunk to controller (hot path).
  - `estop() -> None` (L154) — Trigger emergency stop, always raises `ROSEStopRequested`.
- `class LifecycleEStopHAL(Protocol)` (L64) — Opt-in propagation of the generic
  lifecycle e-stop to downstream hardware or owned processes.
  - `estop() -> None` (L74) — Propagate the generic lifecycle e-stop to downstream hardware / owned processes.
- `class ResettableLifecycleEStopHAL(LifecycleEStopHAL, Protocol)` (L80) — Opt-in
  in-process recovery contract.
  - `reset_estop() -> None` (L83) — In-process recovery from a resettable e-stop.
- `class EStopRecovery(StrEnum)` (L35) — Declares whether recovery is resettable or
  requires a full lifecycle restart.
- `class HALHealthReport` (L43) — Cached, I/O-free health payload (message + fields) for the lifecycle diagnostics heartbeat.
- `class HALHealthProvider(Protocol)` (L55) — Opt-in health-reporting protocol.
  - `health() -> HALHealthReport` (L58) — Cached, I/O-free diagnostics consumed by the generic lifecycle heartbeat.

### `python/hal/src/openral_hal/_mujoco_arm.py`
_Internal MuJoCo-backed HAL implementation shared by UR / Franka / SO-100 / G1 / H1 / Rizon-4 / OpenArm / ALOHA adapters. Reads its wiring from `RobotDescription.sim`._

- `class _MujocoArmInitKwargs(TypedDict)` — Typed shape of the kwargs `MujocoArmHAL.__init__` accepts. (L61) Lets `_sim_kwargs_for` build a value that unpacks under `mypy --strict` with no `# type: ignore`.
- `_resolve_mjcf_path(desc: RobotDescription) -> str` [private] — Resolves `desc.assets.mjcf` to an absolute MJCF path via `openral_core.assets.resolve_asset`; raises `ROSConfigError` if unset/unresolvable. (L81)
- `_kinematic_group(joint_name, groups, *, robot) -> str` [private] (L111) — Maps a joint name to its kinematic group (substring match against `groups`) for the G1/H1 humanoid velocity/effort/PD-gain lookup tables; raises `ROSConfigError` naming `robot` if none match.
- `build_hal(description, *, mode: Literal["sim","real"], transport=None, sim_env_yaml=None) -> HAL` (`resolver.py` L39) — The single seam for building a robot's sim or real HAL from its manifest, used by `deploy sim`/`deploy run`. `mode` picks `description.hal.sim`/`.real` (or a scene YAML via `sim_env_yaml`), merged under `hal.parameters.defaults`; raises `ROSConfigError`/`ROSCapabilityMismatch` for a bad or missing entry.
  - `_import_object(path: str) -> object` [private] — Resolves a `"module.path:Attribute"` import string; raises `ROSConfigError` if malformed/missing. Reuse watch: the canonical entrypoint importer — don't hand-roll `importlib` in HAL callers.
- `class MujocoArmHAL` — Generic MuJoCo-backed HAL adapter for position-controlled arms (and, via the `_per_step_update` hook, torque-controlled humanoids like the H1). (L141)
  - `read_images() -> dict[str, NDArray]` (L399) — Renders the manifest's RGB `SensorSpec`s off the live MJCF, keyed by sensor name, each at that sensor's own `intrinsics` resolution (one cached `mujoco.Renderer` per resolution). Skips a missing camera / render error with a one-shot warning; returns `{}` when disconnected or no RGB sensors.
  - `__init__(description, *, mjcf_path, joint_qpos_addr, actuator_index, joint_qvel_addr=None, grippers=(), keyframe_index=None, seed_ctrl_from_qpos=False, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Init only; MJCF loads at `connect()`. `joint_qvel_addr` defaults to `joint_qpos_addr`, overridden by humanoid HALs (G1/H1) whose floating base shifts qvel by 1. `grippers` is a `SimGripperDescription` sequence — one for single-arm robots, two for bimanual (Aloha, OpenArm). (L191)
  - `_per_step_update(targets) -> None` — Hook run before every `mj_step`; default no-op, overridden by torque-mode subclasses (`H1MujocoHAL`) to recompute actuator torque from `qpos`/`qvel` each step.
  - `connect() -> None` (L267) — Loads the MJCF and prepares `MjData`; first splices in manifest RGB cameras missing from the MJCF via `_camera_rig.rig_cameras_into_mjcf` (bare-arm twins like so100/so101). Idempotent.
  - `disconnect() -> None` (L343) — Release the MuJoCo model (idempotent).
  - `mujoco_handles() -> tuple[Any, Any] | None` (L356) — Exposes the live MuJoCo `(model, data)` handles for the bare-twin arm, mirroring `SimAttachedHAL.mujoco_handles` so `SimSensorBridge`'s offscreen cinecam can render a 3rd-person view; `None` until connected.
  - `read_state() -> JointState` (L487) — Joint state in description-joint order, read live from `MjData` — never raises `ROSPerceptionStale` (the data is always current); a stale gap logs a one-shot warning instead of raising. Staleness is instead policed by the subscription HALs (`ros_control`/`aloha`), not here.
  - `send_action(action: Action) -> None` (L572) — Forward last waypoint to MuJoCo and step. Stamps `_last_action_ns` so the idle stepper yields to a recent command.
  - `sim_time_ns() -> int | None` (L368) — Bare-twin MuJoCo elapsed time in ns from live `MjData.time`; `None` before connect/after disconnect/e-stop. The `/clock` seam for OpenArm/SO-100/SO-101 deploy-sim graphs, matching `SimAttachedHAL.sim_time_ns()`.
  - `clock_authority() -> ClockAuthority` (L385) — Return `ClockAuthority.simulation("mujoco", timestep_s=model.opt.timestep)` while connected, otherwise `ClockAuthority.host_wall()`.
  - `idle_step(wall_dt_s=None) -> bool` (L711) — Sim-only HOLD stepper that keeps cameras/joint-state live while idle; leaves `ctrl` at its last commanded pose and, with `wall_dt_s`, advances that wall-time slice (capped by `_IDLE_STEP_CAP`). Returns `False` after disconnect/e-stop, so it can never drive an e-stopped robot.
  - **(property)** `last_action_ns -> int` (L702) — `time.monotonic_ns()` of the last `send_action` (`0` if never actuated). Read by `SimSensorBridge.should_idle_step` to yield the idle stepper to a recently-commanded skill; mirrors `SimAttachedHAL.last_action_ns`.
  - `reset_to_pose(pose: list[float]) -> None` — Maintenance/test snap of `qpos` with `ctrl` re-seeded; gripper values are mapped through `SimGripperDescription.ctrl_range` so they stay in the HAL's normalized units. Skill startup instead uses the runner's checked ramp or MoveIt. (L614)
  - `estop() -> None` (L687) — Zero `ctrl` and raise `ROSEStopRequested`.
  - **(classmethod)** `from_description(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> MujocoArmHAL` — Manifest-driven constructor: builds the HAL's MJCF path, qpos/qvel/actuator maps and gripper config from `description.sim`, removing the need for per-robot subclasses. (L966)
  - **(staticmethod)** `_sim_kwargs_for(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> _MujocoArmInitKwargs` — Translates `description.sim` into the `__init__` kwarg dict; default 1:1 joint→qpos/actuator mapping from `description.joints`, offset 7/6 when `sim.floating_base=True`. Used by `from_description` and `_init_from_description`. (L890)
  - **(instance method)** `_init_from_description(description, *, mjcf_path=None, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5) -> None` — Shared seam every thin per-robot subclass (UR5e/UR10e, Franka, ALOHA, OpenArm, Rizon4, G1, H1, SO-100) forwards to instead of hand-rolling `super().__init__(DESC, **_sim_kwargs_for(DESC, …))`. (L1019)
  - private: `_require_connected`, `_validate_action`, `_last_arm_targets`, `_apply_arm_targets`, `_apply_gripper_targets`, `_read_gripper_value`, `_gripper_command_to_raw`, `_reset_gripper_qpos`, `_effective_actuator_index_for`
- module const `_IDLE_STEP_CAP = 200` (L136) — Upper bound (physics steps) on how much sim time one `idle_step` wall-time tick may advance, so an executor stall cannot fast-forward the world by seconds.

### `python/hal/src/openral_hal/_base.py`
_Shared HAL mixin — `_connected` flag, validation helpers, and a default `disconnect`._

- `class HALBase` — Non-ABC mixin every adapter subclasses. (L22)
  - `disconnect() -> None` (L40) — Flag-and-log default (guard on `_connected`, `log.info("hal.disconnect", ...)`, clear the flag). Idempotent. Adapters holding a real resource (SDK handle, MuJoCo buffers, USB port) override it; `AlohaHAL` and `RosControlHAL` use the default as-is.
  - private: `_require_connected`, `_require_control_mode`, `_validate_action_dims`
- `_raw_floats(raw: dict[str, object], key: str, width: int) -> list[float]` [private] — Shared `read_state` decode: `raw[key]` as floats if it's a list, else `width` zeros. Used by `AlohaHAL.read_state` and `RosControlHAL.read_state`.

### `python/hal/src/openral_hal/_camera_rig.py`
_Generic sim camera rig — splice manifest cameras into a bare-arm MJCF for deploy sim._

- `rig_cameras_into_mjcf(xml: str, sensors: list[SensorSpec]) -> tuple[str, bool]` (L108) — For each RGB `SensorSpec` with a `sim_placement` missing from `xml`, splices a `<camera>` into its `parent_body` (wrist) or `<worldbody>` (overhead), plus minimal staging (visual-only floor + fill light) when the MJCF has none. Returns `(xml, changed)` — `changed=False` when no rigging is needed, so an already-composed MJCF passes through unchanged. Idempotent; raises `ROSConfigError` when a sensor's `parent_body` is missing. Called by `MujocoArmHAL.connect`.
- module const `_STAGING_FLOOR: str` (L70) — Visual-only (`contype=0 conaffinity=0`) ground-plane `<geom>` XML fragment spliced in by `_ensure_staging` so a bare-arm deploy twin's cameras have a surface to see.

### `python/hal/src/openral_hal/_real_description.py`
_Internal helper to derive a real-hardware ``RobotDescription`` from a sim baseline._

- `make_real_description(base, *, sdk_kind) -> RobotDescription` — `model_copy(update={"sdk_kind": sdk_kind})`; the `hal` entrypoints (`hal.sim` / `hal.real`) are inherited from *base*. (L48)

### `python/hal/src/openral_hal/_sensor_wiring.py`
_Resolve sensor catalog ids into `SensorSpec` / `SensorBundle` and attach them to a `RobotDescription` copy — the shared helper behind every per-robot `*_with_sensors(...)` factory (`franka_panda_with_sensors`, `ur5e_with_sensors`, `ur10e_with_sensors`, `so100_with_sensors`)._

- `SensorRequest = str | tuple[str, Mapping[str, Any]]` (L23) — Either a bare catalog id (`"intel/realsense_d435"`) or an `(id, kwargs)` pair forwarding kwargs to the catalog factory.
- `with_sensors(description, requests, *, default_parent_frame=None) -> RobotDescription` (L28) — Deep-copies `description` and appends catalog-resolved sensors/bundles (via `openral_sensors.CATALOG.build`); each request's `parent_frame` defaults to `default_parent_frame or description.base_frame`. Raises `KeyError` for an unknown catalog id.

### `python/hal/src/openral_hal/franka_panda.py`
_HAL adapter for the Franka Emika Panda 7-DoF arm (sim, MuJoCo)._

- `class FrankaPandaHAL(MujocoArmHAL)` — Franka Panda HAL (MuJoCo-backed). Thin manifest-driven wrapper around `MujocoArmHAL`; `__init__` forwards to `self._init_from_description(FRANKA_PANDA_DESCRIPTION, …)`. (L266)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L295)
- `_panda_joint_specs() -> list[JointSpec]` (L122)
- module const `_PANDA_ARM_JOINT_NAMES: list[str]` (L52) — 7 arm joint names, upstream `panda_jointN` naming.
- module const `_PANDA_GRIPPER_JOINT_NAME = "panda_gripper"` (L62) — synthetic 1-DoF gripper channel, normalized like SO-100's.
- module const `_PANDA_JOINT_NAMES: list[str]` (L64) — arm + gripper, full manifest joint order.
- module const `_PANDA_SIM_JOINT_NAMES: dict[str, str]` (L74) — manifest joint name → native MJCF joint name, used by `SimAttachedHAL.read_state`'s name resolution.
- module const `_PANDA_POSITION_LIMITS: dict[str, tuple[float, float]]` (L87) — per-joint position limits (Franka FR3/Panda data sheet, rad).
- module const `_PANDA_VELOCITY_LIMITS: dict[str, float]` (L98) — per-joint velocity limits (rad/s).
- module const `_PANDA_EFFORT_LIMITS: dict[str, float]` (L109) — per-joint torque limits (Nm).
- const `FRANKA_PANDA_DESCRIPTION = RobotDescription(...)` (L176) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.franka_panda:FrankaPandaHAL"` + `hal.real="openral_hal.franka_panda_real:FrankaPandaRealHAL"`. All MuJoCo wiring (MJCF URI, joint→qpos/actuator maps, gripper config) lives in `FRANKA_PANDA_DESCRIPTION.sim`. The real-HW companion `FRANKA_PANDA_REAL_DESCRIPTION` lives in `franka_panda_real.py`.
- `franka_panda_with_sensors(catalog_ids=None) -> RobotDescription` — Copy of `FRANKA_PANDA_DESCRIPTION` with catalog sensors attached; `None` defaults to the wrist-mounted RealSense D435i reference loadout. (L237)

### `python/hal/src/openral_hal/franka_panda_real.py`
_Real-hardware HAL adapter for the Franka Emika Panda over the FCI._

- module const `_DEFAULT_FRANKA_CONTROLLER: str = "franka_arm_controller"` (L57)
- module const `_DEFAULT_FRANKA_JOINT_STATE_TOPIC: str = "/joint_states"` (L62)
- module const `_DEFAULT_FRANKA_ERROR_RECOVERY: str = "/error_recovery"` (L69) — the `franka_msgs/action/ErrorRecovery` name recorded for the operator; never called by `estop()`.
- `class FrankaPandaRealHAL(RosControlHAL)` — Production adapter for a physical Panda over `franka_ros2` / FCI. **Subclasses** `RosControlHAL` (the UR shape) so it is structurally `RosControlDrivable` and the lifecycle node attaches the production `RosControlTransport` under `hal_mode:=real`; the earlier composed wrapper forwarded only the five HAL Protocol methods, exposed none of the drivable surface, and was therefore never wired — a real Panda deploy published every command into `_default_publish`. (L88)
  - `__init__(*, fci_ip='172.16.0.2', controller_name='franka_arm_controller', joint_state_topic='/joint_states', command_topic=None, error_recovery_action='/error_recovery', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L162)
  - `estop_recovery = RESTART_REQUIRED` — `stopRobot()` ends the FCI session; the reflex is cleared and the FCI re-armed by the operator.
  - `fci_ip -> str` [@property] (L195)
  - `error_recovery_action -> str` [@property] — `franka_ros2`'s `franka_msgs/action/ErrorRecovery` the operator runs before re-arming; recorded, **never** called by `estop()` (it clears a reflex — a recovery, not a stop). (L200)
  - `connect() -> None` — Logs the FCI target, then the base `connect`. (L206)
  - `estop()` — **Inherited, not overridden**: deactivating `franka_arm_controller` through `controller_manager` is the stop — `franka_hardware`'s `on_deactivate` calls `libfranka`'s `stopRobot()`. The earlier override published a dict to `/error_recovery/goal`, an undeclared topic the transport refused, so it was a silent no-op; it is gone.
  - Inherits `description` (= `FRANKA_PANDA_REAL_DESCRIPTION`), `controller_name`, `disconnect`, `read_state`, `send_action`, `command_bindings`, `attach_transport`, `attach_controller_stop`, `reset_estop` (refuses: `RESTART_REQUIRED`) from `RosControlHAL`.
- const `FRANKA_PANDA_REAL_DESCRIPTION = make_real_description(FRANKA_PANDA_DESCRIPTION, sdk_kind="closed_with_api")` (L82) — inherits the shared `hal`; what `robots/franka_panda/robot.yaml` mirrors.

### `python/hal/src/openral_hal/sawyer_real.py`
_Real-hardware HAL adapter for the Rethink Sawyer 7-DoF arm._

- module const `_SAWYER_JOINT_NAMES: tuple[str, ...]` (L65) — 7 arm joint names.
- module const `_SAWYER_POSITION_LIMITS: dict[str, tuple[float, float]]` (L78)
- module const `_SAWYER_VELOCITY_LIMITS: dict[str, float]` (L88)
- module const `_SAWYER_EFFORT_LIMITS: dict[str, float]` (L98)
- module const `_DEFAULT_SAWYER_CONTROLLER: str = "sawyer_arm_controller"` (L202)
- module const `_DEFAULT_SAWYER_JOINT_STATE_TOPIC: str = "/robot/joint_states"` (L207)
- module const `_DEFAULT_SAWYER_ESTOP_TOPIC: str = "/robot/set_super_stop"` (L216)
- `class SawyerRealHAL(RosControlHAL)` — Production adapter for a physical Sawyer over `intera_sdk` / `sawyer_robot`. **Subclasses** `RosControlHAL` for the same reason as `FrankaPandaRealHAL`: the composed wrapper it replaced was not `RosControlDrivable`, so the lifecycle node never attached the production transport to it. (L222)
  - `__init__(*, hostname='sawyer.local', controller_name='sawyer_arm_controller', joint_state_topic='/robot/joint_states', command_topic=None, estop_topic='/robot/set_super_stop', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L293)
  - `hostname -> str` [@property] (L326)
  - `connect() -> None` (L330)
  - `estop_recovery = RESTART_REQUIRED` — the super stop is cleared by `/robot/set_super_reset` + re-enable, an operator step.
  - **(property)** `estop_topic -> str` — `/robot/set_super_stop`, intera's super-stop topic. (L345)
  - `vendor_stop_topics() -> list[str]` — `[estop_topic]`, declared so the transport creates its `std_msgs/Empty` publisher at wire-up. (L351)
  - `_vendor_stop(seam) -> str` — After the base deactivated `sawyer_arm_controller`, publishes intera's super stop (`RobotEnable.stop()`'s topic: "Simulate an e-stop button being pressed. Robot must be reset to clear the stopped state"). The topic carries no service ack; the acknowledged half is the controller deactivation, and the HIL gate reads `/robot/state.stopped`.
  - Inherits `description` (= `SAWYER_REAL_DESCRIPTION`), `controller_name`, `disconnect`, `read_state`, `send_action`, `command_bindings`, `attach_transport`, `attach_controller_stop`, `estop`, `reset_estop` (refuses: `RESTART_REQUIRED`) from `RosControlHAL`.
- `_sawyer_joint_specs() -> list[JointSpec]` (L109)
- const `SAWYER_DESCRIPTION = RobotDescription(...)` (L152) — sim baseline; `sdk_kind="open"`, `hal.sim=None` (no MuJoCo HAL adapter today) + `hal.real="openral_hal.sawyer_real:SawyerRealHAL"`.
- const `SAWYER_REAL_DESCRIPTION = make_real_description(SAWYER_DESCRIPTION, sdk_kind="closed_with_api")` (L192) — inherits the shared `hal`; what `robots/sawyer/robot.yaml` mirrors.

### `python/hal/src/openral_hal/panda_mobile.py`
_In-process digital-twin HAL for the `panda_mobile` embodiment (Franka 7-DoF arm on a holonomic 3-DoF base). Built by `build_hal` for the manifest-driven `ManifestHALLifecycleNode` and by tests; ROS node entrypoint in `packages/openral_hal_panda_mobile/`._

- const `PANDA_MOBILE_BASE_JOINT_NAMES: list[str]` — Base joints `[base_x, base_y, base_yaw]`, derived from `PANDA_MOBILE_DESCRIPTION.base_joints` (not hardcoded). (L83)
- const `PANDA_MOBILE_JOINT_NAMES: list[str]` — Full 11-DoF order: base (3) + arm (7, role-derived) + gripper (1, role-derived) — all from the description. (L99)
- const `PANDA_MOBILE_DESCRIPTION: RobotDescription` — Canonical RobotDescription, loaded from `robots/panda_mobile/robot.yaml` at module import. Single source of truth for joint metadata + `sim_joint_name` overrides; the arm/base/gripper name constants above derive from it via `JointSpec.role`. (L76)
- `class PandaMobileHAL` — In-process digital-twin HAL. Routes `BODY_TWIST` → planar Euler integration of (vx, vy, wz); routes `JOINT_POSITION` → 7-vec arm targets or 11-vec base+arm+gripper targets. (L115)
  - `connect() -> None` (L180) — Idempotent; flips the connected flag.
  - `disconnect() -> None` (L184) — Idempotent; clears the connected flag.
  - `read_state() -> JointState` (L188) — Fresh 10-DoF snapshot; raises `ROSConfigError` if called before `connect()`.
  - `send_action(action: Action) -> None` (L202) — Applies the action to in-memory state; accepts `JOINT_POSITION` / `BODY_TWIST` / `CARTESIAN_DELTA` / `GRIPPER_POSITION`, each reading its matching `Action` field. No-ops while e-stop is latched. Raises `ROSConfigError` for an unsupported mode or a missing/mis-shaped payload.
  - `estop() -> None` (L259) — Latch the e-stop flag; subsequent `send_action` calls no-op until `reset_estop`.
  - `reset_estop() -> None` (L270) — Clear the e-stop latch; caller asserts the cause has been resolved.
  - **(property)** `estop_latched -> bool` (L275)
  - **(property)** `base_pose -> tuple[float, float, float]` (L280) — Current base `(x, y, yaw)` for tests / odom publishers.
  - **(property)** `base_twist -> tuple[float, ...]` (L285) — Last commanded base body twist `(vx, vy, vz, wx, wy, wz)`; zeroed once a non-BODY_TWIST action is sent.
- module const `_PANDA_MOBILE_ARM_JOINT_NAMES` (L87) — the 7 Franka arm joint names within the 11-DoF panda_mobile order.
- module const `_PANDA_MOBILE_GRIPPER_JOINT_NAME` (L95) — the 1-DoF gripper joint name.
- module const `_PLANAR_TWIST_EPS` (L112) — tolerance for detecting a non-planar `BODY_TWIST` component.
- _(removed: the `base_sim_joint_names` re-export wrapper — callers now import `openral_core.extract_base_sim_joint_names` directly.)_

### `python/hal/src/openral_hal/depth_cloud.py`
_Reusable, robot-agnostic depth-camera → `sensor_msgs/PointCloud2` plumbing for deploy-sim HAL nodes (octomap_server source → kernel world-collision check). Pure SensorSpec adapters + the ROS msg builder; the ray-cast synth lives in `openral_sim.backends.depth_camera`._
- module const `_OPTICAL_IN_MJCAM` (L31) — REP-103 optical frame (x right, y down, z forward) expressed in the MuJoCo camera frame, as a diagonal flip-y/z matrix.
- module const `_MIN_EYE_DISTANCE_M = 1e-6` (L34) — degenerate eye→lookat distance below which azimuth/elevation are undefined.
- module const `_VIEWER_LOOKAT_LIFT_M = 0.7` (L39) — deploy-viewer orbit-pivot lift off the floor onto the robot body.
- module const `_VIEWER_PULLBACK_M = 2.0` (L40) — deploy-viewer eye pullback so cluttered mobile-manip scenes show the whole robot.
- module const `_VIEWER_CAMERA_PREFS` (L214) — default `prefer` substrings for `preferred_viewer_camera_id` (`"agentview"`, `"top"`, `"frontview"`, `"front"`).
- `is_depth_sensor(spec) -> bool` — True when `spec.modality in ("depth", "point_cloud")` **and** it carries pinhole `intrinsics` (required to back-project). (L43)
- `mjcf_camera_name(spec) -> str` — Resolves the backing MJCF `<camera>` name: `spec.metadata["mjcf_camera"]` if set (the sim camera name can differ from the ROS sensor name), else `spec.name`. (L52)
- `robot_self_body_ids(model, sim_joint_names) -> frozenset[int]` — Every MJCF body sharing a first-`_`-token prefix with one of the robot's `sim_joint_name`s (`mobilebase0`/`robot0`/`gripper0`). Passed as `synthesize_depth_image(exclude_body_ids=…)` so the depth raster self-filters the robot out of its own world map. (L103)
- `depth_synth_kwargs(spec, *, max_range_default, render_size=None) -> dict` — Maps a depth `SensorSpec` to depth-synth kwargs (intrinsics + `min/max_range_m`, falling back to `max_range_default`); with `render_size`, intrinsics are rescaled via `openral_core.scale_intrinsics_to` to match the render resolution. (L65)
- `resolve_base_body_name(model, *, description=None) -> str | None` — Resolves the MJCF body backing the robot's `base_frame` (first base joint's prefix + `_base`, else `mobilebase0_base`/`base`/`robot0_base`/`base_link`); `None` if none exist. Backs the depth self-filter anchor and viewer fallback — not the TF/extrinsic parent, which is `resolve_base_frame_body_name`. (L135)
- `resolve_base_frame_body_name(model, *, description=None) -> str | None` — Resolves the MJCF body whose pose `base_frame` actually carries on `/tf` (ADR-0095): tries the base joint's `_support` body (robosuite's arm-mount plate) before delegating to `resolve_base_body_name`. They differ only on robosuite/RoboCasa mobile manipulators, whose chassis root sits well below the arm mount `MobileBaseBridge` publishes as `base_link`; fixed-base arms resolve identically either way. (L171)
- `preferred_viewer_camera_id(model, *, prefer=("agentview","top","frontview","front")) -> int` — Picks the MJCF camera the viewer should open from: first camera name matching a `prefer` substring, else the first declared camera, else `-1` if none. Consumed by `initial_viewer_camera`. (L217)
- `initial_viewer_camera(*, model, data, description=None) -> tuple[tuple[float,float,float], float, float, float]` — Opening free-camera pose `(lookat, distance, azimuth_deg, elevation_deg)` for the viewer (always `mjCAMERA_FREE`, so mouse orbit/zoom stay live). Eye placed at the `preferred_viewer_camera_id` camera with orbit pivot on the robot base when one exists, else delegates to `base_aligned_free_camera`. (L370)
- `apply_robosuite_visual_geomgroups(opt, model) -> bool` — For a robosuite/RoboCasa model, hides collision shells (geomgroup 0) and shows textured visual geoms (group 1) so `mujoco.viewer` renders textures instead of collision boxes. Gated on a robosuite signature, not geom counts (dm_control/gym-aloha put visuals in group 0). Used by `sim run --view`. (L257)
- `base_aligned_free_camera(*, model, data, base_body_name=None, azimuth_offset_deg=135.0, elevation_deg=-25.0, distance_scale=2.0, max_distance_m=3.5) -> tuple[tuple[float,float,float], float, float, float]` — Fallback free-camera framing for camera-less models: centres on the robot base, offset by its world yaw, at `distance_scale × model.stat.extent` capped at `max_distance_m`. Falls back to `model.stat.center` with no yaw when `base_body_name` is absent. (L288)
- `camera_optical_tf_to_base(*, model, data, camera_name, base_body_name) -> tuple[tuple[float,float,float], tuple[float,float,float,float]]` — Live `(translation_xyz, quat_xyzw)` of the camera optical frame (REP-103) expressed in the base body, for broadcasting `base_frame → <camera>_optical_frame`. Raises `ROSConfigError` if camera/body absent. (L424)
- `pointcloud2_from_points_xyz(points, *, frame_id, stamp=None) -> PointCloud2` — Packs an `(N, 3)` float32 array into an unordered XYZ-float32 `sensor_msgs/PointCloud2`, the layout octomap_server's `cloud_in` expects. (L476)
- `depth_image_from_grid(depth, *, frame_id, stamp=None) -> Image` — Packs an `(H, W)` float32 metric-depth raster into a `32FC1 sensor_msgs/Image` (`0.0` = no measurement) for nvblox's projective depth integrator. (L517)
- `points_from_depth_grid(depth, *, fx, fy, cx, cy, clearing=None, max_range_m=None) -> NDArray[np.float32]` — Back-projects an `(H, W)` metric-depth raster into an `(N, 3)` optical-frame cloud, dropping `0.0` pixels. `clearing` re-adds the self-filter's clearing rays at `max_range_m` so OctoMap can clear cells behind the robot's silhouette; without it those cells can never clear. Raises `ROSConfigError` on a shape mismatch or a marked mask with no `max_range_m`. (L554)
- `depth_grid_from_image(msg) -> NDArray[np.float64]` — Inverse of `depth_image_from_grid`: decodes a driver's depth `Image` into an `(H, W)` metre raster, accepting `32FC1` (metres) and `16UC1` (millimetres, rescaled). Raises `ROSConfigError` on an unsupported encoding or a length mismatch. Used by `VisionAttachmentBridge` to read real wrist-camera depth. (L662)
- `camera_info_from_intrinsics(*, width, height, fx, fy, cx, cy, frame_id, stamp=None) -> CameraInfo` — Builds a pinhole `sensor_msgs/CameraInfo` for a synthesised depth image — `K=[fx,0,cx;0,fy,cy;0,0,1]`, identity `R`, `P` mirroring `K` (no baseline), zero `plumb_bob` distortion (MuJoCo ray-cast has none). Callers pass the **stride-scaled** intrinsics so the model matches the rasterised image. (L709)

### `python/hal/src/openral_hal/aloha.py`
_HAL adapter for the Trossen ALOHA bimanual setup, plus the MuJoCo digital twin._

- `class AlohaHAL(HALBase)` — Real-hardware adapter for the 14-DoF ALOHA over the Interbotix XS SDK. Stays on `HALBase`, not `RosControlHAL`: the driver owns the bus, like `SO100FollowerHAL`. Its command topics are ros2_control names a real ALOHA does not expose, so `send_action` reports success and moves nothing — issue #250 holds the `xs_sdk` wire contract and the on-rig checks needed before changing the defaults. Its **e-stop**, by contrast, targets what a real ALOHA does expose: `estop_recovery = RESTART_REQUIRED`, and `estop()` cuts torque on every arm namespace through an attached `InterbotixStopSeam`. (L384)
  - `__init__(*, left_arm_controller=..., right_arm_controller=..., left_gripper_controller=..., right_gripper_controller=..., joint_state_topic='/joint_states', arm_namespaces=('follower_left', 'follower_right'), publish_fn=None, state_fn=None, staleness_limit_s=0.2, stop_timeout_s=5.0)` — The former `estop_topic` (`/aloha/estop`, a broadcast for a watchdog node that does not exist in this repo) is gone; `arm_namespaces` names the `xs_sdk` robots the stop torques off. (L453)
  - `arm_namespaces() -> list[str]` — The `xs_sdk` robot namespaces the stop torques off; the `InterbotixStoppable` surface the lifecycle node reflects on. (L490)
  - `attach_torque_stop(seam) -> None` — Bind the `InterbotixStopSeam` after construction (`build_hal` runs before any ROS node exists); rejects a non-seam with `ROSConfigError`. (L494)
  - **(property)** `last_stop_report -> DownstreamStopReport | None` — The `DownstreamStopReporting` surface the lifecycle heartbeat and FATAL log read. (L508)
  - `connect() -> None` (L514)
  - `read_state() -> JointState` (L531)
  - `send_action(action) -> None` — Splits the 14-D action 4-ways across per-arm + per-gripper controllers. (L557)
  - `estop() -> None` — Drop the connection flag first, then `torque_enable(cmd_type='group', name='all', enable=false)` on every arm namespace (each attempted even if an earlier one refused), record a `DownstreamStopReport` (`stopped` only when every arm acknowledged; `controller_states` per arm `torque_off` / `torque_unknown`), raise `ROSEStopRequested`. Torque off leaves the ViperX arms limp (no brakes) — the same outcome as the rig's hardware e-stop and the only stop `xs_sdk` offers. (L621)
  - private: `_require_connected`, `_cut_torque`
- `class InterbotixStopSeam(Protocol)` — `torque_enable(robot_name, *, group, enable, timeout_s) -> TriggerReport`; production `InterbotixXSTransport`, unit lane `SimTorqueSeam`. (L356)
  - `torque_enable(robot_name, *, group, enable, timeout_s) -> TriggerReport` — `/<robot_name>/torque_enable` with `cmd_type='group'`; `success` is the call completing. (L364)
- `class InterbotixStoppable(Protocol)` — `arm_namespaces()` + `attach_torque_stop(seam)`; what the lifecycle node's `_attach_interbotix_transport` reflects on. (L372)
  - `arm_namespaces() -> list[str]` (L375)
  - `attach_torque_stop(seam) -> None` (L379)
- `class AlohaMujocoHAL(MujocoArmHAL)` — MuJoCo digital twin for the 14-DoF bimanual ALOHA; thin manifest-driven wrapper around `MujocoArmHAL` (bimanual amendment). All wiring lives in `ALOHA_DESCRIPTION.sim`: `gym_aloha:bimanual_viperx_transfer_cube` URI, explicit `joint_qpos_addr` / `actuator_index` (left arm 0-5, left gripper 6, right arm 8-13, right gripper 14 — skipping the negative-finger slots), two `PASSTHROUGH` grippers with `mirror_actuator_index` (positive finger + negative finger), `keyframe_index: 0` (seeds the fingers inside `ctrlrange=[0.021, 0.057]`). (L699)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(ALOHA_DESCRIPTION, …)`. (L734)
- `_aloha_joint_specs() -> list[JointSpec]` (L165)
- `_default_publish(topic, msg) -> None` (L687)
- module const `_ALOHA_LEFT_ARM_JOINTS: tuple[str, ...]` (L114) — left-arm joint names, ViperX 300 order.
- module const `_ALOHA_LEFT_GRIPPER_JOINT: str` (L122) — left gripper joint name.
- module const `_ALOHA_RIGHT_ARM_JOINTS: tuple[str, ...]` (L123) — right-arm joint names.
- module const `_ALOHA_RIGHT_GRIPPER_JOINT: str` (L131) — right gripper joint name.
- module const `_ALOHA_JOINT_NAMES: tuple[str, ...]` (L133) — full 14-DoF joint order: left arm + left gripper + right arm + right gripper.
- module const `_PI: float = 3.14159` (L145) — truncated π matching the YAML manifest's string form, used by `_ALOHA_ARM_POSITION_LIMITS` so the manifest-vs-HAL drift guard stays byte-equal.
- module const `_ALOHA_ARM_POSITION_LIMITS: dict[str, tuple[float, float]]` (L146) — per-joint-group position limits from the ViperX 300 data sheet.
- module const `_ALOHA_GRIPPER_POSITION_LIMITS: tuple[float, float]` (L154)
- module const `_ALOHA_ARM_AXIS: dict[str, tuple[float, float, float]]` (L155) — per-joint-group rotation axis.
- module const `_DEFAULT_LEFT_ARM_CONTROLLER: str` (L333) — `AlohaHAL`'s default left-arm `xs_sdk` controller name.
- module const `_DEFAULT_RIGHT_ARM_CONTROLLER: str` (L334) — default right-arm controller name.
- module const `_DEFAULT_LEFT_GRIPPER_CONTROLLER: str` (L335) — default left-gripper controller name.
- module const `_DEFAULT_RIGHT_GRIPPER_CONTROLLER: str` (L336) — default right-gripper controller name, matching the udev-pinned CAN dev-id convention.
- module const `_DEFAULT_ALOHA_JOINT_STATE_TOPIC: str` (L340) — default `/joint_states` topic.
- module const `_DEFAULT_ALOHA_ARM_NAMESPACES: tuple[str, str]` (L348) — the `xs_sdk` robot namespaces `estop` torques off (`follower_left`, `follower_right`).
- module const `_TORQUE_GROUP_ALL = "all"` (L349) — the `TorqueEnable` group name covering every joint of one arm.
- const `ALOHA_DESCRIPTION = RobotDescription(...)` (L209) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.aloha:AlohaMujocoHAL"` + `hal.real="openral_hal.aloha:AlohaHAL"`.
- const `ALOHA_REAL_DESCRIPTION = make_real_description(ALOHA_DESCRIPTION, sdk_kind="closed_with_api")` (L321) — inherits the shared `hal`; what `robots/aloha_bimanual/robot.yaml` mirrors.

### `python/hal/src/openral_hal/ur.py`
_HAL adapters for the Universal Robots UR5e and UR10e arms (sim, MuJoCo)._

- `class UR5eHAL(MujocoArmHAL)` — UR5e HAL (MuJoCo-backed). Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(UR5e_DESCRIPTION, …)`. (L302)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L326)
- `class UR10eHAL(MujocoArmHAL)` — UR10e HAL (MuJoCo-backed). Same shape as `UR5eHAL`. (L344)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L356)
- `ur5e_with_sensors(catalog_ids=None) -> RobotDescription` (L246)
- `ur10e_with_sensors(catalog_ids=None) -> RobotDescription` (L272)
- `_ur_joint_specs(velocity_limits, effort_limits) -> list[JointSpec]` (L119)
- module const `_UR_JOINT_NAMES: list[str]` (L58) — 6 arm joint names, shared by UR5e/UR10e.
- module const `_UR5E_POSITION_LIMITS: dict[str, tuple[float, float]]` (L69) — shared by UR5e and UR10e (same joint range family).
- module const `_UR5E_VELOCITY_LIMITS: dict[str, float]` (L78)
- module const `_UR5E_EFFORT_LIMITS: dict[str, float]` (L88)
- module const `_UR10E_VELOCITY_LIMITS: dict[str, float]` (L99)
- module const `_UR10E_EFFORT_LIMITS: dict[str, float]` (L109)
- const `UR5e_DESCRIPTION = RobotDescription(...)` (L157) — sim manifest; all MuJoCo wiring lives in `UR5e_DESCRIPTION.sim`.
- const `UR10e_DESCRIPTION = RobotDescription(...)` (L201) — sim manifest; all MuJoCo wiring lives in `UR10e_DESCRIPTION.sim`.

### `python/hal/src/openral_hal/ur_real.py`
_Real-hardware HAL adapters for UR5e / UR10e via `ros2_control` + `ur_robot_driver` (URCap / RTDE)._

- `class UR5eRealHAL(_URRealHAL)` — Real UR5e via `ur_robot_driver`. (L194)
- `class UR10eRealHAL(_URRealHAL)` — Real UR10e via `ur_robot_driver`. (L246)
- `class _URRealHAL(RosControlHAL)` — Shared real-HW base (controller / topic defaults + `deadman_topic` + `dashboard_stop_service`). `estop_recovery = RESTART_REQUIRED`. (L87)
  - `vendor_stop_services() -> list[str]` — `[dashboard_stop_service]` (default `/dashboard_client/stop`). (L168)
  - `_vendor_stop(seam) -> str` — After the base deactivated `scaled_joint_trajectory_controller`, calls the dashboard `stop` (`std_srvs/Trigger`), which stops the `external_control` program on the pendant so the robot performs a controlled stop and the driver's control connection ends; a refused Trigger raises `ROSRuntimeError` so the report reads unacknowledged. Recovery: operator restarts the program (`/dashboard_client/play` + `resend_robot_program`), relaunches the HAL node, re-aligns. (L172)
- module const `_UR_CONTROLLER_NAME: str` (L60) — default `ur_robot_driver` `JointTrajectoryController` name.
- module const `_UR_JOINT_STATE_TOPIC: str` (L61)
- module const `_UR_DEADMAN_TOPIC: str` (L62)
- module const `_UR_DASHBOARD_STOP_SERVICE: str` (L69) — default `ur_robot_driver` dashboard `stop` service (`std_srvs/Trigger`).
- const `UR5e_REAL_DESCRIPTION = make_real_description(UR5e_DESCRIPTION, sdk_kind="closed")` (L76) — inherits the shared `hal`; what `robots/ur5e/robot.yaml` mirrors.
- const `UR10e_REAL_DESCRIPTION = make_real_description(UR10e_DESCRIPTION, sdk_kind="closed")` (L81) — inherits the shared `hal`; what `robots/ur10e/robot.yaml` mirrors.

### `python/hal/src/openral_hal/so100_follower.py`
_SO100FollowerHAL — wraps lerobot's SO-100 follower arm USB driver._

- module const `_SO100_JOINT_NAMES: list[str]` (L79) — canonical joint order, matching lerobot's bus motor dict order.
- module const `_RESET_MAX_RAD_S = 0.5` (L94) — `reset_to_pose` ramp speed cap.
- module const `_RESET_MIN_S = 1.0` (L95) — `reset_to_pose` ramp minimum duration.
- module const `_RESET_MAX_S = 6.0` (L96) — `reset_to_pose` ramp maximum duration.
- module const `_RESET_STEP_HZ = 30.0` (L97) — `reset_to_pose` ramp waypoint rate.
- module const `_RESET_DEADBAND_RAD` (L100) — below this max joint delta the arm counts as already at the target and the ramp is skipped.
- `class SO100FollowerHAL` — HAL adapter wrapping lerobot's SO-100/SO-101 follower. Declares `estop_recovery = RESTART_REQUIRED`: its vendor E-stop disconnects the motor bus, so command execution requires a fresh lifecycle start and alignment. (L288)
  - `__init__(port='/dev/ttyUSB0', *, calibrate_on_connect=False, id=None, calibration_dir=None, max_relative_target=None, staleness_limit_s=0.5, robot=None)` (L334)
  - `connect() -> None` — Open USB serial connection. (L376)
  - `disconnect() -> None` — Close USB, disable motor torque (idempotent). (L492)
  - `read_state() -> JointState` — Joint state in radians. (L505)
  - `send_action(action: Action) -> None` — Forward one step to the SO-100 motor bus. (L533)
  - `reset_to_pose(pose: list[float]) -> None` — Explicit maintenance/test ramp from current → target (speed-capped `_RESET_MAX_RAD_S`, duration clamped `[_RESET_MIN_S, _RESET_MAX_S]`, `_RESET_STEP_HZ` waypoints). The lifecycle node exposes it at `/openral/<robot>/reset_to_pose`; skill startup uses the runner's safety-kernel path. (L556)
  - `estop() -> None` — Disconnect motors then raise. (L619)
  - `_require_connected(operation: str)`, `_obs_to_positions(obs)` [@staticmethod], `_action_to_lerobot(action)`
  - `_joint_values_to_lerobot(step) -> dict[str, float]` (module-level) — THE single manifest-order → lerobot `{"<joint>.pos": …}` unit conversion (rad→deg arm joints, `[0,1]`→`[0,100]` gripper); both `_action_to_lerobot` and the `reset_to_pose` ramp route through it so a calibration/range change can never apply to one actuation path and not the other. (L260)
- `_deg_to_rad(deg) -> float` (L255)
- `_rad_to_deg(rad) -> float` (L280)
- const `SO100_DESCRIPTION = RobotDescription(...)` (L104)
- `so100_with_sensors(catalog_ids=None) -> RobotDescription` — Copy of `SO100_DESCRIPTION` with catalog sensors attached; `None` defaults to the LeRobot reference loadout (`["logitech/c920"]`). (L224)

### `python/hal/src/openral_hal/galaxea_a1.py`
_Real-only Galaxea A1 HAL. OpenRAL stays ROS 2 / Python 3.12; the operator's
official ROS 1 Noetic SDK runs out of process behind a literal IPv4-loopback
JSON-lines sidecar. No vendor source, binary, or message package is distributed._

- `class GalaxeaA1HAL(HALBase)` (L561) — Six-axis joint-position + normalized-gripper adapter. `read_state`/`send_action` use a cached snapshot/latest target so network I/O stays off the hot path. Commands fail closed on stale state/status, unaccepted motor bits, non-finite values, target misalignment, or an excessive feedback-relative step. `estop` asks the sidecar to stop its owned ROS 1 stack and always raises `ROSEStopRequested`.
  - `connect() -> None` (L657) — Connect to the sidecar (sends the `hello` handshake with joint names/limits/timeouts/masks); requires one complete fresh snapshot. Raises `ROSRuntimeError` if already connected.
  - `disconnect() -> None` (L691) — Close the sidecar transport idempotently.
  - `read_state() -> JointState` (L696) — Return the latest non-stale six-joint feedback snapshot.
  - `send_action(action: Action) -> None` (L708) — Queue one validated joint or normalized-gripper target; raises `ROSConfigError` for an unsupported control mode.
  - `health() -> HALHealthReport` (L763) — Cached hardware health for the lifecycle diagnostics heartbeat (sidecar address, feedback/status ages, motor status, relay state, staged/forwarded targets).
  - `estop() -> None` (L785) — Stop the sidecar-owned ROS 1 stack (bounded, SIGINT then SIGKILL) and always raise `ROSEStopRequested`.
- module const `_JOINT_NAMES` (L62) — six `arm_jointN` names.
- module const `_JOINT_LIMITS` (L63) — per-joint `(min, max)` position limits, official A1 URDF.
- module const `_JOINT_ORIGINS_XYZ` (L71) — per-joint URDF origin translation, transcribed from the official A1 URDF.
- module const `_JOINT_ORIGINS_RPY` (L79) — per-joint URDF origin rotation, transcribed from the official A1 URDF.
- module const `_PROTOCOL_VERSION = 1` (L87) — sidecar wire-protocol version sent in the `hello` handshake.
- module const `_GRIPPER_STATUS_INDEX` (L88) — index of the gripper's status code within the sidecar's status array.
- module const `_MAX_TCP_PORT = 65536` (L89) — upper bound accepted for the sidecar's loopback TCP port.
- module const `_MAX_STATUS_MASK = 0xFFFFFFFF` (L90) — bit-mask upper bound for a motor status/error mask.
- const `GALAXEA_A1_DESCRIPTION` (L93) — Real-only `RobotDescription`, mirrored by
  `robots/galaxea_a1/robot.yaml`: official A1 URDF joint names/limits, sidecar
  deadlines, motor masks, 0..104 mm normalized gripper mapping, and calibrated
  D455 front / D405 wrist RGB observation contracts. Collision primitives are
  absent until their redistribution/lowering provenance is cleared.
- `tools/galaxea_a1_ros1_sidecar.py` — Python-3.8 ROS 1 process owning `roscore`,
  `signal_arm/single_arm_node.launch`, and the official `mobiman/jointTracker_demo_node`
  binary. A sidecar-owned relay is the sole publisher to `/arm_joint_command_host`,
  gated `LOCKED → ARMING → ACTIVE` (aligned to target + feedback before forwarding);
  a gripper command while not `ACTIVE` is refused fail-closed. A lease timeout,
  malformed command, stale feedback, motor fault, disconnect, or e-stop stops the
  whole owned process group.
- `tools/run_galaxea_a1_sidecar.sh` — Docker launcher requiring an explicit
  operator-provided image and SDK path; mounts both read-only, claims only the
  selected serial device. `--check-only` verifies image, SDK files, cache, serial
  ownership, loopback port, container name, and process lock without opening the
  serial device or starting a container.

Galaxea A1 hardware bring-up and the LingBot-VA rSkill deploy runbook moved to [`robots/galaxea_a1/README.md`](https://github.com/OpenRAL/openral/blob/master/robots/galaxea_a1/README.md).

### `python/hal/src/openral_hal/h1.py`
_MuJoCo digital twin for the Unitree H1 humanoid (Menagerie MJCF). Contract validator only — falls without an S0 cerebellum, so gravity must stay disabled in closed-loop tests. Unlike G1/UR/Franka/SO-100, the H1 menagerie ships torque actuators, so this HAL runs a software PD position loop every physics step._

- `class H1MujocoHAL(MujocoArmHAL)` — 19-DoF humanoid HAL driving `mujoco_menagerie/unitree_h1/h1.xml` (legs + torso + arms, no wrists); thin manifest-driven wrapper around `MujocoArmHAL`. Overrides `_apply_arm_targets` to a no-op and `_per_step_update` to compute `tau = kp*(target - q) - kv*dq` clamped to `ctrlrange`, so the public action contract stays "position targets in radians" — mirroring how `unitree_sdk2` wraps torque control in a position loop on real hardware. (L329)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L369)
  - `_per_step_update(targets) -> None` — Recomputes PD torque every `mj_step`.
  - `_apply_arm_targets(targets) -> None` — No-op (PD loop runs per-step instead).
- module const `_H1_LEFT_LEG_JOINTS: tuple[str, ...]` (L80) — left-leg joint names.
- module const `_H1_RIGHT_LEG_JOINTS: tuple[str, ...]` (L87) — right-leg joint names.
- module const `_H1_TORSO_JOINTS: tuple[str, ...]` (L94) — torso joint name(s).
- module const `_H1_LEFT_ARM_JOINTS: tuple[str, ...]` (L95) — left-arm joint names.
- module const `_H1_RIGHT_ARM_JOINTS: tuple[str, ...]` (L101) — right-arm joint names.
- module const `_H1_JOINT_NAMES: tuple[str, ...]` (L107) — full 19-DoF joint order (legs + torso + arms).
- module const `_H1_POSITION_LIMITS: dict[str, tuple[float, float]]` (L122) — per-joint position limits.
- module const `_H1_EFFORT_LIMITS: dict[str, float]` (L149) — per-joint actuator effort ceiling used to clamp the software PD loop's torque.
- module const `_H1_VELOCITY_LIMITS_BY_GROUP: dict[str, float]` (L176) — per-kinematic-group velocity limits.
- module const `_H1_KP_BY_GROUP: dict[str, float]` (L309) — per-kinematic-group PD proportional gain.
- module const `_H1_KV_BY_GROUP: dict[str, float]` (L317) — per-kinematic-group PD derivative gain.
- `_h1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `torso` / `shoulder` / `elbow`) for `joint_name`. (L189)
- `_h1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for an H1 joint. (L194)
- `_h1_joint_specs() -> list[JointSpec]` — Build the 19 `JointSpec`s from the joint-name tuples + the per-joint limit tables. (L230)
- `_h1_pd_gains() -> dict[str, tuple[float, float]]` — Per-joint `(kp, kv)` for the software PD loop (kv = 0.05*kp; kp sized so a 1-rad error roughly saturates each actuator's ctrlrange). (L320)
- const `H1_DESCRIPTION = RobotDescription(...)` (L253) — Sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.h1:H1MujocoHAL"`, `hal.real=None` (sim-only). All MuJoCo wiring lives in `H1_DESCRIPTION.sim`; drift-guarded against `robots/h1/robot.yaml`.

### `python/hal/src/openral_hal/flexiv_rizon4.py`
_MuJoCo digital twin for the Flexiv Rizon 4 — 7-DoF cobot with whole-body force sensitivity (0.1 N).  Structurally identical to the UR / Franka sim HALs: position actuators, no gripper, no floating base, no PD-loop overrides — a clean `MujocoArmHAL` subclass._

- `class Rizon4MujocoHAL(MujocoArmHAL)` — 7-DoF HAL driving `mujoco_menagerie/flexiv_rizon4/flexiv_rizon4.xml` via `MujocoArmHAL`. Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(RIZON4_DESCRIPTION, …)`. (L180)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L210)
- `_rizon4_joint_specs() -> list[JointSpec]` — Build the 7 `JointSpec`s from the joint-name tuple + per-joint limit tables. (L111)
- module const `_RIZON4_JOINT_NAMES: tuple[str, ...]` (L65) — 7 arm joint names.
- module const `_RIZON4_POSITION_LIMITS: dict[str, tuple[float, float]]` (L80)
- module const `_RIZON4_VELOCITY_LIMIT: float` (L94)
- module const `_RIZON4_EFFORT_LIMITS: dict[str, float]` (L100)
- const `RIZON4_DESCRIPTION = RobotDescription(...)` (L132) — Sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.flexiv_rizon4:Rizon4MujocoHAL"`, `hal.real=None` (sim-only). All MuJoCo wiring lives in `RIZON4_DESCRIPTION.sim`; drift-guarded against `robots/rizon4/robot.yaml`.

### `python/hal/src/openral_hal/openarm.py`
_MuJoCo digital twin for the Enactic OpenArm **v2** bimanual humanoid arm.  Fresh `HALBase` subclass — v2's native `<position>` actuators with per-class PD baked into the MJCF mean the HAL just writes target → ctrl and steps, no software PD loop needed._

- `class OpenArmMujocoHAL(MujocoArmHAL)` — 16-DoF (7 arm + 1 gripper per side) bimanual HAL driving `enactic/openarm_mujoco/v2/openarm_v20_bimanual.xml`; thin manifest-driven wrapper around `MujocoArmHAL`. All wiring (MJCF URI fetched via `ensure_openarm_v2_mjcf`, joint/actuator maps, gripper config) lives in `OPENARM_DESCRIPTION.sim`. (L431)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(OPENARM_DESCRIPTION, …)`. (L466)
- `_openarm_arm_joint_specs(names, position_limits, side) -> list[JointSpec]`, `_openarm_gripper_joint_spec(name, side, position_limits) -> JointSpec`, `_openarm_joint_specs() -> list[JointSpec]` (L167, L190, L206)
- module const `_OPENARM_LEFT_ARM_JOINTS: tuple[str, ...]` (L102) — left-arm joint names.
- module const `_OPENARM_RIGHT_ARM_JOINTS: tuple[str, ...]` (L103) — right-arm joint names.
- module const `_OPENARM_LEFT_GRIPPER_JOINT: str` (L104) — left gripper joint name.
- module const `_OPENARM_RIGHT_GRIPPER_JOINT: str` (L105) — right gripper joint name.
- module const `_OPENARM_JOINT_NAMES: tuple[str, ...]` (L107) — full 16-DoF joint order.
- module const `_OPENARM_LEFT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]]` (L118)
- module const `_OPENARM_RIGHT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]]` (L127)
- module const `_OPENARM_LEFT_GRIPPER_POSITION_LIMITS: tuple[float, float]` (L139)
- module const `_OPENARM_RIGHT_GRIPPER_POSITION_LIMITS: tuple[float, float]` (L140)
- module const `_OPENARM_ARM_EFFORT_LIMITS: dict[str, float]` (L146)
- module const `_OPENARM_GRIPPER_EFFORT_LIMIT: float` (L162)
- module const `_OPENARM_ARM_VELOCITY_LIMIT: float` (L165)
- module const `_OPENARM_GRIPPER_VELOCITY_LIMIT: float` (L166)
- const `OPENARM_DESCRIPTION = RobotDescription(...)` (L235) — Shared baseline for sim **and** real (`name="openarm_v2"`, 16 revolute joints). `sdk_kind="open"` (the whole real path — `openarm_can` + `openarm_ros2` — is Apache-2.0, unlike UR/Franka's closed vendor runtime). One flat `hal.parameters.defaults` block serves both `hal.sim`/`hal.real` entrypoints; `build_hal` drops the keys each constructor doesn't accept. Drift-guarded against `robots/openarm/robot.yaml`.

### `python/hal/src/openral_hal/openarm_real.py`
_Real-hardware adapter for the Enactic OpenArm v2. Commands `openarm_bringup`'s ros2_control stack (400 Hz) rather than SocketCAN directly: Skill → Action → this adapter → four command topics → controller_manager → `openarm_hardware` SystemInterface → `openarm_can` → SocketCAN → Damiao motors._

- module const `_LEFT_ARM_CONTROLLER: str` (L93) — default left-arm `JointTrajectoryController` name.
- module const `_LEFT_GRIPPER_CONTROLLER: str` (L94) — default left-gripper controller name.
- module const `_RIGHT_ARM_CONTROLLER: str` (L95) — default right-arm controller name.
- module const `_RIGHT_GRIPPER_CONTROLLER: str` (L96) — default right-gripper controller name, matching the udev-pinned CAN dev-id convention.
- module const `_JOINT_STATE_TOPIC: str` (L97)
- module const `_LEFT_CAN_INTERFACE: str` (L98) — default left CAN interface name.
- module const `_RIGHT_CAN_INTERFACE: str` (L99) — default right CAN interface name.
- module const `_LEFT_ARM_SLICE` (L102) — left-arm slice of the 16-DoF action vector.
- module const `_LEFT_GRIPPER_SLICE` (L103) — left-gripper slice.
- module const `_RIGHT_ARM_SLICE` (L104) — right-arm slice.
- module const `_RIGHT_GRIPPER_SLICE` (L105) — right-gripper slice.
- `class OpenArmRealHAL(RosControlHAL)` (L120) — 16-DoF real-HW adapter, same action layout as `OpenArmMujocoHAL`. Guards three silent-failure modes: fan-out (one action sliced atomically across the bimanual bringup's four controller topics, so a rejection can't leave one arm on a new chunk and the other stale), read-by-name (`read_state` matches `/joint_states` by joint name, raising rather than zero-filling a missing joint), and bus preflight (`connect()` refuses a missing/wrong-type/down CAN link rather than reporting connected over a dead bus). `estop_recovery = RESETTABLE`.
  - `__init__(description=None, *, left_can_interface='openarm_left', right_can_interface='openarm_right', left_arm_controller=..., left_gripper_controller=..., right_arm_controller=..., right_gripper_controller=..., joint_state_topic='/joint_states', publish_fn=None, state_fn=None, staleness_limit_s=0.5, require_can_links=True)` — Rejects a manifest that is not 16-joint or whose joints lack `sim_joint_name` (the URDF joint name, i.e. the logical→ros2_control mapping).
  - `connect() -> None` (L300) — Bus preflight (`openral_core.can.preflight_can_links`) then delegates to `RosControlHAL.connect`; refuses a CAN link that is missing, is not a CAN device, or is down.
  - `ros2_control_joint_names() -> list[str]` (L256) — 16 URDF names in action-vector order.
  - `controller_names() -> list[str]` — The four controllers `estop` deactivates, in fan-out order (left arm, left gripper, right arm, right gripper); the fleet conformance test pins that every `command_bindings` topic belongs to one of them. (L288)
  - `estop_recovery = RESETTABLE` — the deactivated controllers hold and the CAN bus stays configured, so the inherited `reset_estop` re-activates all four through the same `controller_manager` seam and reconnects only once the manager confirms every one `active`.
  - `read_state() -> JointState` (L333) — Matches `/joint_states` by ros2_control joint name and returns manifest order; raises rather than zero-filling when a joint is absent.
  - `command_bindings() -> dict[str, ControllerKind]` (L273) — The four `/…/joint_trajectory` topics, all `JOINT_TRAJECTORY`: `openarm_bringup` configures each gripper as a 1-DoF `JointTrajectoryController`, so grippers and arms take the same message.
  - `health() -> HALHealthReport` (L524) — Cached per-bus state from `connect()`; performs no I/O.
  - `send_action(action)` (L388) — ADR-0102 slot groups: a `tick_group_size > 1` action is staged (`SlotGroupStager`) and composed into one 16-DoF action by `_compose_group` before the same all-or-nothing publish. A standalone gripper action raises `ROSConfigError` — the four bimanual controllers share one joint vector, so there's nowhere to route it.
  - `_compose_group(group) -> Action` — Delegates to `compose_slot_group` over `description.joints`; logs `hal.send_action.slot_group`.
  - `estop() -> None` (L473) — Resets the stager before delegating, so a slot staged when the stop landed cannot outlive it and corrupt the first post-e-stop tick. The downstream stop itself (all four controllers deactivated and confirmed) is the base implementation's.
  - `disconnect() -> None` (L463) — Also resets the stager, so a slot staged before disconnect cannot raise an incomplete-group error on the first tick after reconnect.
- **Real hardware needs an upstream patch.** `openarm_bringup` generates the real `robot_description` via xacro, and on OpenArm **v2.0** the `left_can_interface`/`right_can_interface` arguments are silently dropped, falling back to `can1`/`can0` (v1.0 is unaffected). Unpatched, the arms don't move and the CAN preflight passes because those buses really are up — it's the description pointing elsewhere. `robots/openarm/patches/` carries the fix and an idempotent `apply.sh`.
- The bus preflight is not implemented here — `connect()` delegates to `openral_core.can.preflight_can_links`, passing the two udev-named interfaces and OpenArm-specific remedy text; the mechanism lives in layer 0 for other CAN robots to reach. Narrower than `openral_detect.probes.can`, the richer read behind `openral detect`.
- const `OPENARM_REAL_DESCRIPTION = make_real_description(OPENARM_DESCRIPTION, sdk_kind="open")` (L117) — Shares kinematics, safety envelope and HAL entrypoints with the sim baseline.

### `python/hal/src/openral_hal/_slot_group.py`
_Reassembles a slot-dispatched inference tick into one whole-robot command (ADR-0102): `_dispatch_slots` emits one `Action` per non-discard slot (four for OpenArm v2 bimanual), each published separately through the safety kernel, so a HAL that commands one vector must recombine them._

- module const `GRIPPER_MODES` (L35) — `(GRIPPER_POSITION, GRIPPER_BINARY)`, the modes composed from `Action.gripper` + `ee_name` rather than from a joint row.
- `compose_slot_group(actions, joint_names) -> list[float]` (L38) — Merges one tick's slots into a full-dof target row, addressing by name (never position): `JOINT_*` slots via `Action.joint_names`, gripper slots via `ee_name`. Raises `ROSConfigError` for an unnamed joint slot, an unknown joint/end-effector, a joint claimed twice, or any joint left uncommanded.
- `class SlotGroupStager` (L147) — Buffers at most one tick. `stage(action) -> list[Action] | None` returns the group when the last slot lands, else `None`; `pending` is the count in flight; `reset()` drops it. A tick change mid-group raises `ROSRuntimeError` rather than committing survivors and leaving one arm stale; the slot that exposed it opens the new tick instead of being dropped with it. Mirrors `SimAttachedHAL._stage_action_group`.
  - `stage(action) -> list[Action] | None` (L177) — see above.
  - `pending -> int` [@property] (L168) — count of slots in flight.
  - `reset() -> None` (L172) — drop the buffered slots.

### `python/hal/src/openral_hal/anvil_openarm_v2.py`
_MuJoCo digital twin for the Anvil OpenARM 2.0 — Anvil Robotics' manufactured variant of the standard OpenArm v2. Differs from the Enactic v2 twin only in two joint ranges (J1, J6) and a wrist support bracket, baked into the fetched MJCF, so the HAL stays a thin manifest-driven subclass._

- `class AnvilOpenArmV2MujocoHAL(MujocoArmHAL)` — 16-DoF (7 arm + 1 gripper per side) bimanual HAL driving `models/anvil_openarm_bimanual.xml` from the pinned `bensonlee5/anvil-openarm-mujoco` clone. All wiring lives in `ANVIL_OPENARM_V2_DESCRIPTION.sim`. Structurally identical to `OpenArmMujocoHAL` — the Anvil-ness is entirely in the asset. (L365)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(ANVIL_OPENARM_V2_DESCRIPTION, …)`. (L403)
- `_anvil_arm_joint_specs(names, position_limits, side) -> list[JointSpec]`, `_anvil_gripper_joint_spec(name, side, position_limits) -> JointSpec`, `_anvil_joint_specs() -> list[JointSpec]` (L160, L184, L204)
- module const `_ANVIL_LEFT_ARM_JOINTS: tuple[str, ...]` (L89) — left-arm joint names.
- module const `_ANVIL_RIGHT_ARM_JOINTS: tuple[str, ...]` (L90) — right-arm joint names.
- module const `_ANVIL_LEFT_GRIPPER_JOINT: str` (L91) — left gripper joint name.
- module const `_ANVIL_RIGHT_GRIPPER_JOINT: str` (L92) — right gripper joint name.
- module const `_ANVIL_LEFT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]]` (L103) — left-arm per-joint position limits, incl. the Anvil-specific J1/J6 ranges.
- module const `_ANVIL_RIGHT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]]` (L112) — right-arm per-joint position limits.
- module const `_ANVIL_LEFT_GRIPPER_POSITION_LIMITS: tuple[float, float]` (L122)
- module const `_ANVIL_RIGHT_GRIPPER_POSITION_LIMITS: tuple[float, float]` (L123)
- module const `_ANVIL_ARM_EFFORT_LIMITS: dict[str, float]` (L128) — per-joint effort limit table.
- module const `_ANVIL_GRIPPER_EFFORT_LIMIT: float` (L144)
- module const `_ANVIL_ARM_VELOCITY_LIMIT: float` (L148)
- module const `_ANVIL_GRIPPER_VELOCITY_LIMIT: float` (L149)
- const `ANVIL_OPENARM_V2_DESCRIPTION = RobotDescription(...)` (L213) — Sim baseline (`name="anvil_openarm_v2"`, 16 revolute joints, Anvil's J1/J6 ranges, two wrist RGB `SensorSpec`s). `sdk_kind="open"`, `hal.sim="openral_hal.anvil_openarm_v2:AnvilOpenArmV2MujocoHAL"`, `hal.real=None` (sim-only; a real driver wrapper is a tracked follow-up). Drift-guarded against `robots/anvil_openarm_v2/robot.yaml`.

### `python/hal/src/openral_hal/_pinned_clone.py`

- `fetch_pinned_clone(repo_url, sha, repo_dir, *, submodule=None, what=…) -> None` — The shared staged-clone + atomic-rename dance for pinned-SHA asset fetchers: shallow clone into a staging dir, checkout the pinned SHA, optional submodule init, `os.rename` into place (loser of a concurrent race discarded). Raises `ROSConfigError` when `git` is missing or any step fails. Reuse watch: the canonical pinned-clone fetcher. (L30)

### `python/hal/src/openral_hal/_anvil_openarm_v2_assets.py`
_Vendor the Anvil OpenARM 2.0 MJCF from `bensonlee5/anvil-openarm-mujoco` — no upstream package ships the Anvil variant._

- `ensure_anvil_openarm_v2_mjcf() -> str` — Idempotently clones `bensonlee5/anvil-openarm-mujoco` at a pinned SHA, initialises its mesh submodule, and returns the bimanual MJCF path. Raises `ROSConfigError` if `git` is missing or the clone/submodule init fails. Mirrors `_openarm_v2_assets.ensure_openarm_v2_mjcf` plus the submodule step. (L60)
- module const `_ANVIL_PINNED_SHA: str` (L40) — bump when the generator or the local Anvil spec changes.
- module const `_ANVIL_REPO_URL: str` (L41) — the pinned clone's URL.
- module const `_ANVIL_MJCF_REL: str` (L42) — the bimanual MJCF's path within the clone.
- module const `_ANVIL_MESH_SUBMODULE: str = "upstream/openarm_mujoco"` (L46) — mesh submodule the generated MJCF's meshdir points into; initialised by `ensure_anvil_openarm_v2_mjcf`.

### `python/hal/src/openral_hal/_openarm_v2_assets.py`
_Vendor the upstream `enactic/openarm_mujoco` v2 MJCF until `robot_descriptions` bumps its own pin to match._

- `ensure_openarm_v2_mjcf() -> str` — Idempotently clones `enactic/openarm_mujoco` at a pinned v2 SHA into `$OPENRAL_CACHE_DIR/openarm_v2/<sha>/`, returns the bimanual MJCF path. Raises `ROSConfigError` when `git` is missing or the clone fails. Mirrors the pattern used by `python/sim/src/openral_sim/backends/so100_robosuite/_assets.py`. (L57)
- module const `_OPENARM_V2_PINNED_SHA: str` (L40) — bump to track upstream v2 updates.
- module const `_OPENARM_REPO_URL: str` (L41) — the pinned clone's URL.
- module const `_OPENARM_V2_MJCF_REL: str` (L42) — the bimanual MJCF's path within the clone.

### `python/hal/src/openral_hal/g1.py`
_MuJoCo digital twin for the Unitree G1 humanoid. The default stock-Menagerie path provides joint-contract validation + ADR-0087 kinematic glide; explicit `walking_enabled=True` selects ADR-0089's pinned MuJoCo Playground ONNX policy and matching gravity-on dynamics._

- `class G1MujocoHAL(MujocoArmHAL)` — 29-DoF humanoid HAL. Default mode drives `mujoco_menagerie/unitree_g1/g1.xml`; walking mode swaps in the pinned policy-tuned MJCF + ONNX controller. Floating-base joint remains implicit world state. (L376)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5, body_twist_dt_s=0.05, walking_enabled=False)` (L414)
  - `connect() -> None` (L484) — Connect and re-arm the base pin from the fresh qpos; in walking mode, resets to the `knees_bent` keyframe and constructs/initializes the `G1WalkingController`. Raises `ROSConfigError` (and disconnects) if the walking MJCF lacks that keyframe.
  - `disconnect() -> None` (L504) — Release the walking controller, then the MuJoCo model.
  - `base_pose -> tuple[float, float, float]` (property) — Current glide pin or live walking base pose. (L472)
  - `base_twist -> tuple[float, ...]` (property) — Last commanded 6-vec base twist (`base_link` frame), zeroed by any non-BODY_TWIST action. (L480)
  - `send_action(action)` — Routes `BODY_TWIST` to the glide or walking controller; every other mode defers to the joint path. Non-planar twist components raise `ROSConfigError`. (L509)
  - `idle_step(wall_dt_s=None) -> bool` — HOLD-step with the base pinned; walking state resets so a stale velocity command is never replayed. Opts out of the bare-arm `_step_while_active` path since an active BODY_TWIST must not be overridden by HOLD. (L556)
  - `reset_to_pose(pose: list[float]) -> None` (L568) — Reset the joint pose plus the base pin and (when walking) the gait state.
  - `_per_step_update(targets)` / `_pin_base()` / `_pin()` — The upright pin: `qpos[0:7]` fixed to the glide pose (roll/pitch clamped to 0), `qvel[0:6]` zeroed. Replacing the pin with a balance controller is the designed S0 upgrade seam. (L596)
- module const `_PLANAR_TWIST_EPS` (L80) — tolerance for detecting a non-planar `BODY_TWIST` component (`send_action`'s validation).
- module const `_G1_LEFT_LEG_JOINTS: tuple[str, ...]` (L92) — left-leg joint names.
- module const `_G1_RIGHT_LEG_JOINTS: tuple[str, ...]` (L100) — right-leg joint names.
- module const `_G1_WAIST_JOINTS: tuple[str, ...]` (L108) — waist joint names.
- module const `_G1_LEFT_ARM_JOINTS: tuple[str, ...]` (L113) — left-arm joint names.
- module const `_G1_RIGHT_ARM_JOINTS: tuple[str, ...]` (L122) — right-arm joint names.
- module const `_G1_JOINT_NAMES: tuple[str, ...]` (L131) — full 29-DoF joint order (legs + waist + arms).
- module const `_G1_POSITION_LIMITS: dict[str, tuple[float, float]]` (L150) — per-joint position limits.
- module const `_G1_VELOCITY_LIMITS_BY_GROUP: dict[str, float]` (L186) — per-kinematic-group velocity limits.
- module const `_G1_EFFORT_LIMITS_BY_GROUP: dict[str, float]` (L195) — per-kinematic-group effort limits.
- `_g1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `waist` / `shoulder` / `elbow` / `wrist`) for `joint_name`. (L209)
- `_g1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for a G1 joint, following the menagerie URDF convention. (L214)
- `_g1_joint_specs() -> list[JointSpec]` — Build the 29 `JointSpec`s from the joint-name tuples and the per-joint limit tables. (L264)
- const `G1_DESCRIPTION = RobotDescription(...)` (L292) — Sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.g1:G1MujocoHAL"`, `hal.real=None` (sim-only). Advertises `supported_control_modes=[joint_position, body_twist]`, `mobile_base` in `embodiment_tags`, and a forward `head` RGB camera so BODY_TWIST nav skills (InternVLA-N1) match. All MuJoCo wiring lives in `G1_DESCRIPTION.sim`; drift-guarded against `robots/g1/robot.yaml`.

### `python/hal/src/openral_hal/_g1_walking.py`
_ADR-0089's private, sim-only walking implementation._

- `ensure_g1_walking_assets() -> tuple[str, str]` (L84) — downloads four files from pinned MuJoCo Playground commit `43d180a`, verifies SHA-256, links the cached Menagerie meshes, flattens the included MJCF, and returns `(mjcf_path, policy_path)`.
- module const `_PINNED_SHA: str` (L19) — pinned MuJoCo Playground commit.
- module const `_RAW_ROOT: str` (L20) — raw-GitHub root URL derived from `_PINNED_SHA`.
- module const `_ASSETS: dict[str, str]` (L21) — the four pinned-asset relative paths mapped to their expected SHA-256 hex digests, verified by `ensure_g1_walking_assets`.
- `class G1WalkingController` (L128) — 50 Hz CPU ONNX inference over 500 Hz MuJoCo physics; builds the upstream 103-D observation, validates the 29-D finite output, scales/clips joint targets, and persists gait phase/action history.
  - `initialize(data: mujoco.MjData) -> None` (L157) — Capture `data.qpos[7:]` as the standing default-angle reference, then `reset_state()`.
  - `reset_state() -> None` (L161) — Zero the last-action history and gait phase, reset the decimation counter.
  - `apply(model: mujoco.MjModel, data: mujoco.MjData, command: tuple[float, float, float]) -> None` (L166) — Every `_CONTROL_DECIMATION`-th physics step, build the 103-D observation (base linear velocity, gyro, projected gravity, command, joint-angle delta, joint velocity, last action, phase sin/cos), run the ONNX policy, and scale/clip/apply the resulting joint targets.

### `python/hal/src/openral_hal/so100_mujoco.py`
_MuJoCo digital twin for the SO-100 follower arm (Menagerie MJCF)._

- `class SO100MujocoHAL(MujocoArmHAL)` — SO-100 follower MuJoCo HAL, driving `mujoco_menagerie`'s `trs_so_arm100/so_arm100.xml` with the same 6-DoF action layout as `SO100FollowerHAL`. Maps lerobot-style joint names to the Menagerie joints and normalises the revolute Jaw range `[-0.174, 1.75]` to `[0, 1]`. (L65)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L105)

### `python/hal/src/openral_hal/so100_sim.py`
_SO100DigitalTwin — in-process simulator for the SO-100 follower arm._

- module const `_JOINT_NAMES: list[str]` (L37) — canonical joint order, mirrors lerobot's SO100Follower motor dict.
- module const `_DEFAULT_POSITIONS: dict[str, float]` (L48) — default joint positions in lerobot's native units (degrees; gripper `[0, 100]`, mid-range).
- `class SO100DigitalTwinConfig(RobotConfig)` — Config for the digital twin. (L59)
  field: `initial_positions`
- `class SO100DigitalTwin(Robot)` — In-process digital twin. (L76)
  - `__init__(config)` (L101)
  - `observation_features() -> dict[str, type]` — One float per joint pos. (L116)
  - `action_features() -> dict[str, type]` — One float per target. (L125)
  - `is_connected -> bool` [@property] (L134)
  - `is_calibrated -> bool` [@property] — Always True. (L139)
  - `connect(calibrate=True) -> None` — Activate (no serial port opened). (L146)
  - `calibrate() -> None` — No-op. (L154)
  - `configure() -> None` — No-op. (L158)
  - `get_observation() -> RobotObservation` — Lerobot-native units. (L162)
  - `send_action(action) -> RobotAction` — Apply position cmd, update state. (L175)
  - `disconnect() -> None` — Deactivate (idempotent). (L194)

### `python/hal/src/openral_hal/ros_control.py`
_RosControlHAL — `ros2_control`-backed HAL adapter, plus the typed downstream e-stop seam every ros2_control robot stops through (issue #295)._

- `class ControllerKind(StrEnum)` — Which ros2_control controller sits behind one command topic, and therefore which message type commands it: `JOINT_TRAJECTORY` (`trajectory_msgs/JointTrajectory` — every ros2_control robot in this repo, OpenArm's 1-DoF grippers included) and `FORWARD_COMMAND` (`std_msgs/Float64MultiArray`, the `forward_command_controller` family; no in-repo robot yet). Declared rather than assumed because publishing the wrong type is **silent** — DDS never delivers it, so the arm does not move and nothing logs an error. Re-exported from `openral_hal`. (L70)
- `class ControllerSwitchReport` — Frozen dataclass: acknowledged outcome of one `controller_manager` switch. `ok` is the service's verdict **and** the post-switch confirmation (every controller listed in the requested state); `states` is what `list_controllers` reported per name; `detail` names what did not move. (L106)
- `class TriggerReport` — Frozen dataclass: `success` + `message` of one `std_srvs/Trigger`-shaped vendor call. (L123)
- `class DownstreamStopReport` — Frozen dataclass: what a HAL knows about its last downstream stop — `stopped` (only when controller deactivation **and** the vendor stop were acknowledged; a local latch never counts), `controllers`, `controller_states`, `vendor_stop` (service/topic label or `""`), `detail`. `fields()` flattens it for `/diagnostics` (`downstream_stop=acknowledged|unacknowledged`, …). Read by the lifecycle node, which logs FATAL on an unacknowledged stop. (L131)
  - `fields() -> dict[str, str]` — Flattens the report for `/diagnostics` and the log line: `downstream_stop=acknowledged|unacknowledged`, `downstream_controllers`, `downstream_controller_states`, `downstream_vendor_stop`, `downstream_stop_detail`. (L149)
- `class ControllerStopSeam(Protocol)` — What a transport must expose for a ros2_control HAL to stop its controllers: `deactivate_controllers(names, *, timeout_s)`, `activate_controllers(names, *, timeout_s)` (both → `ControllerSwitchReport`), `call_trigger(service, *, timeout_s) -> TriggerReport`, `publish_empty(topic)`. Production: `RosControlTransport`; unit lane: `SimTransport`. (L163)
  - `deactivate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — The stop: switch every named controller to `inactive` and confirm. (L176)
  - `activate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — The `RESETTABLE` re-arm: switch to `active` and confirm. (L182)
  - `call_trigger(service, *, timeout_s) -> TriggerReport` — One declared `std_srvs/Trigger` vendor service. (L188)
  - `publish_empty(topic) -> None` — One `std_msgs/Empty` on a declared vendor-stop topic. (L192)
- `class ControllerStoppable(Protocol)` — The stop-side counterpart of `RosControlDrivable`: `controller_names()`, `vendor_stop_services()`, `vendor_stop_topics()`, `attach_controller_stop(seam)`. The lifecycle node **refuses to configure** a `RosControlDrivable` real HAL that is not also this; `RosControlHAL` answers all four. Pinned per manifest by `tests/unit/test_real_hal_estop_fleet_conformance.py`. (L198)
  - `controller_names() -> list[str]` — Every `controller_manager` controller `estop` must deactivate. (L210)
  - `vendor_stop_services() -> list[str]` — `std_srvs/Trigger` services the vendor stop calls. (L214)
  - `vendor_stop_topics() -> list[str]` — `std_msgs/Empty` topics the vendor stop publishes on. (L218)
  - `attach_controller_stop(seam) -> None` — Bind the `ControllerStopSeam` at wire-up. (L222)
- `class DownstreamStopReporting(Protocol)` — `last_stop_report -> DownstreamStopReport | None` property; what the lifecycle heartbeat and FATAL log read. (L228)
  - **(property)** `last_stop_report -> DownstreamStopReport | None` (L232)
- `class RosControlHAL` — `ros2_control`-backed HAL adapter. Implements `LifecycleEStopHAL` with `estop_recovery = RESTART_REQUIRED` by default (the conservative policy; OpenArm opts into `RESETTABLE`). (L259)
  - `__init__(description, controller_name, *, joint_state_topic='/joint_states', command_topic=None, publish_fn=None, state_fn=None, staleness_limit_s=0.5, stop_timeout_s=5.0)` (L308)
  - `attach_transport(publish_fn, state_fn, stamp_fn=None) -> None` — Bind a live transport after construction; `build_hal` runs before any ROS node exists, so a real deployment cannot pass one to `__init__`. `stamp_fn` is the transport's per-message arrival clock and is what makes the staleness check a freshness check rather than a time-since-connect check. Rejects a non-callable `stamp_fn` at wire-up. (L347)
  - `attach_controller_stop(seam) -> None` — Bind the `ControllerStopSeam` (same rationale as `attach_transport`); rejects a non-seam at wire-up with `ROSConfigError`. (L386)
  - `controller_names() -> list[str]` — Every `controller_manager` controller `estop` deactivates; `[controller_name]` by default, overridden by the bimanual OpenArm alongside `command_bindings`. (L409)
  - `vendor_stop_services() -> list[str]` — `std_srvs/Trigger` services and `std_msgs/Empty` topics the vendor stop uses; declared up front so the transport creates the clients/publishers at wire-up and refuses an undeclared name loudly. Empty in the base. (L418)
  - `vendor_stop_topics() -> list[str]` — `std_msgs/Empty` topics the vendor stop publishes on, declared for the same reason. Empty in the base; Sawyer overrides. (L427)
  - **(property)** `last_stop_report -> DownstreamStopReport | None` (L432)
  - `command_bindings() -> dict[str, ControllerKind]` — Every controller topic this HAL publishes to, each with the wire format its controller speaks; one entry becomes one typed transport publisher. Insertion order is the `send_action` fan-out order. **This is the override point for a new robot** — overridden by the bimanual OpenArm (4 topics, all `JOINT_TRAJECTORY`). (L436)
  - `command_topics() -> list[str]` — Derived from `command_bindings()` so the topic list and the declared formats cannot drift; override `command_bindings`, not this. (L450)
  - `ros2_control_joint_names() -> list[str]` — Joint names in ros2_control's namespace, action order; what `/joint_states` is keyed by. Overridden where URDF names differ from manifest names (OpenArm). (L458)
  - **(property)** `joint_state_topic -> str` — The aggregated `sensor_msgs/JointState` topic this HAL reads. (L468)
  - **(property)** `controller_name -> str` — Name of the primary `ros2_control` controller this HAL commands (the one `command_topic` defaults from). Hoisted here from the Franka / Sawyer wrappers so every subclass answers it. (L473)
  - `connect() -> None` (L479)
  - `disconnect() -> None` — inherited from `HALBase` (flag-and-log default; no extra teardown needed).
  - `read_state() -> JointState` — Age is measured from `stamp_fn()` when a transport supplied one, else from `connect()`. (L499)
  - `send_action(action) -> None` — Publish JointTrajectory; carries `joint_names` so no transport keeps a second copy of the mapping. (L541)
  - `estop() -> None` — Drop the connection flag **first** (nothing more leaves this HAL), then `_stop_downstream()`, then raise `ROSEStopRequested` whose message says whether the downstream stop was acknowledged. The outcome lives in `last_stop_report`, not the exception type, so the HAL Protocol's "always `ROSEStopRequested`" holds. With no seam attached the report is `stopped=False` / "no controller stop seam attached" — never a claimed stop. (L576)
  - `_stop_downstream() -> DownstreamStopReport` [private] — Deactivates `controller_names()` through the seam (STRICT switch + listing confirmation), then runs `_vendor_stop`; both are attempted even if the first fails, so the report names everything that did and did not acknowledge. (L609)
  - `_vendor_stop(seam) -> str` [hook] — Vendor-specific stop beyond controller deactivation; returns its label for the report or raises a `ROSError`. Base: none (for a plain ros2_control robot deactivation *is* the stop; for `franka_hardware` it is what calls `libfranka`'s `stopRobot()`). Overridden by UR (dashboard `stop` Trigger) and Sawyer (super-stop Empty). (L667)
  - `_vendor_reset(seam) -> None` [hook] — Vendor re-arm before controllers are re-activated; no-op by default. (L679)
  - `reset_estop() -> None` — Only for `RESETTABLE` adapters: `_vendor_reset`, then `activate_controllers` through the seam, and only once the manager confirms every controller `active` is the HAL marked connected again. Raises `ROSRuntimeError` for a `RESTART_REQUIRED` policy ("in-process reset is forbidden"), a missing seam, or an unacknowledged re-activation — so the lifecycle latch can never clear ahead of the vendor reset. (L683)
  - private: `_require_connected`, `_validate_action`

### `python/hal/src/openral_hal/ros_control_transport.py`
_Production `rclpy` transport shared by every real `RosControlHAL` robot._

- `class RosControlDrivable(Protocol)` — **runtime-checkable**, structural: what a HAL must expose to be driven by `RosControlTransport` (`command_bindings`, `ros2_control_joint_names`, `joint_state_topic`, `attach_transport`). The lifecycle node's wiring gates on *this*, not on `isinstance(hal, RosControlHAL)`, so a ros2_control robot that reimplements the fan-out on `HALBase` rather than inheriting can opt in by shape instead of being silently skipped. Non-method member (`joint_state_topic` is a property) so `isinstance` works but `issubclass` raises — gate with `isinstance`. (L80)
  - `command_bindings() -> dict[str, ControllerKind]` (L93) — see `RosControlHAL.command_bindings`.
  - `ros2_control_joint_names() -> list[str]` (L97) — see `RosControlHAL.ros2_control_joint_names`.
  - **(property)** `joint_state_topic -> str` (L102)
  - `attach_transport(publish_fn, state_fn, stamp_fn=None) -> None` (L106) — see `RosControlHAL.attach_transport`.
- `class RosControlTransport` — Bridges one `RosControlHAL` to live ros2_control topics **and** implements `ControllerStopSeam` over `controller_manager`. Built only from what the HAL reports about itself (`command_bindings()`, `ros2_control_joint_names()`, `joint_state_topic`), so a one-controller UR and the four-controller bimanual OpenArm are one code path and a new robot needs no transport code. Wired automatically by the lifecycle node under `hal_mode:=real`. **Reuse watch:** the canonical real-HW ros2_control bridge — do not hand-roll publishers in a per-robot HAL. (L128)
  - `__init__(node, *, command_topics, joint_names, joint_state_topic='/joint_states', command_kinds=None)` — Refuses an empty topic or joint list, and a `command_kinds` entry naming a topic not in `command_topics` (a typo there would silently leave the real topic on the `JOINT_TRAJECTORY` default). Topics absent from `command_kinds` default to `JOINT_TRAJECTORY`. (L157)
  - `publish(topic, msg) -> None` — One command dict → the message type this topic's declared `ControllerKind` calls for. Publishes the chunk's final step only (a point is an absolute target the controller interpolates toward). Raises on a topic the HAL never declared, **and on a payload it cannot express** (no `joint_targets` key — e.g. a `{"position": …}` gripper command): dropping that quietly would be a silent no-op on the actuation path. An empty `joint_targets` list is the distinct, benign "nothing to send this tick". (L291)
  - `state() -> dict[str, object]` — Newest joint state projected onto the HAL's joint order; merged **by name, never index**, since split controllers publish independently. (L388)
  - `last_arrival() -> float` — `time.monotonic()` of the newest message; 0.0 before the first. Feeds `RosControlHAL.attach_transport(stamp_fn=...)`. (L409)
  - `deactivate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — **The stop.** `controller_manager/switch_controller` with `deactivate_controllers=names`, `strictness=STRICT`, then `list_controllers` to confirm every name reads `inactive`; `ok` is both. A deactivated `JointTrajectoryController` holds its last position command (zeroes velocity / effort), drops late trajectories (its subscriber goes inactive) and writes nothing until re-activated — verified against `ros2_controllers` jazzy, which is also why an *empty* trajectory is **not** used as a stop: ROS 2 rejects it ("Empty trajectory received"). (L415)
  - `activate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — Mirror for a `RESETTABLE` re-arm; confirms `active`. (L428)
  - `call_trigger(service, *, timeout_s) -> TriggerReport` — One declared `std_srvs/Trigger` (UR `/dashboard_client/stop`); `ROSConfigError` for an undeclared service, `success=False` when absent or unanswered. (L434)
  - `publish_empty(topic) -> None` — One `std_msgs/Empty` on a declared vendor-stop topic (Sawyer `/robot/set_super_stop`); `ROSConfigError` for an undeclared one. (L463)
  - `controller_states(*, timeout_s) -> dict[str, str]` — `{name: state}` from `list_controllers`; `{}` if the manager did not answer. The lifecycle node logs a warning at wire-up when the HAL's controllers are not yet listed active. (L480)
  - `close() -> None` — Destroy the helper node the service clients live on (lifecycle cleanup). (L492)
  - `__init__` also takes `controller_names`, `trigger_services`, `empty_topics` (from `hal.controller_names()` / `vendor_stop_services()` / `vendor_stop_topics()`) and `controller_manager='/controller_manager'`. Service calls run on a private helper node + executor because the e-stop callback runs on the lifecycle node's single-threaded executor and a future issued from inside a callback can only complete if something else spins the client.
  - `seen_joints() -> set[str]` (L551)
  - `missing_joints() -> list[str]` (L555)
  - private: `_time_from_start_s`, `_on_joint_state`
- module const `_COMMAND_DEPTH = 1` (L120) — command QoS depth (RELIABLE, VOLATILE, shallow — CLAUDE.md §2).
- module const `_STATE_DEPTH = 10` (L125) — state QoS depth.
- module const `_DEFAULT_TIME_FROM_START_S` (L591) — default `time_from_start` for a published `JointTrajectoryPoint` when the caller supplies none.
- module const `_STRICT = 2` (L76) — `controller_manager_msgs/SwitchController.STRICT`: a switch that cannot fully apply is refused rather than partially applied.
- `_message_type(kind) -> type` — The one place a `ControllerKind` becomes a ROS message class. Adding an enum member without extending this raises at wire-up rather than publishing a plausible-but-wrong type onto the actuation path; `tests/unit/test_ros_control_transport.py::test_every_controller_kind_maps_to_a_message_type` pins that. (L594)

### `python/hal/src/openral_hal/interbotix_transport.py`
_Production `rclpy` torque-stop seam for the Interbotix XS arms (ALOHA) — the stop only, not the command path (#250)._

- `class InterbotixXSTransport` — Implements `openral_hal.aloha.InterbotixStopSeam` over real `interbotix_xs_msgs/srv/TorqueEnable` clients, one per `arm_namespaces` entry, created at wire-up on a private helper node (same callback-deadlock rationale as `RosControlTransport`). Raises `ROSConfigError` at construction when `interbotix_xs_msgs` is not installed, so the ALOHA e-stop fails at configure, not on the first stop. Attached by the lifecycle node under `hal_mode:=real` to any `InterbotixStoppable` HAL. (L38)
  - `torque_enable(robot_name, *, group, enable, timeout_s) -> TriggerReport` — `/<robot_name>/torque_enable` with `cmd_type='group'`; `TorqueEnable` has an empty response, so the acknowledgement is the call completing — absent or unanswered → `success=False`. `ROSConfigError` for an undeclared arm. (L88)
  - `close() -> None` — Destroy the helper node. (L131)

### `python/hal/src/openral_hal/sim_transport.py`
_SimTransport — in-memory simulated `ros2_control` transport **and** `ControllerStopSeam`; SimTorqueSeam — in-memory `InterbotixStopSeam`. The in-repo controller simulators the unit lane proves the e-stop path on._

- `class SimTransport` — In-memory transport simulating JointTrajectory controllers behind a `controller_manager` table. Mirrors the two verified facts about a deactivated `JointTrajectoryController`: a command on an `inactive` controller's topic is **dropped** (recorded in `dropped_calls`, not applied) and nothing moves until re-activated. (L59)
  - `__init__(n_joints, *, controllers=None, trigger_responses=None, switch_faults=None)` — `controllers` given → STRICT table (an unknown name refuses a switch like an unloaded controller); omitted → names register `active` on first use. `trigger_responses` maps a service to the `(success, message)` its `call_trigger` returns (simulate a refusing UR dashboard); `switch_faults` maps `"deactivate"` / `"activate"` to the exception that switch raises instead of answering (the rclpy-shaped fault the HAL must contain in its report). (L103)
  - `publish(topic, msg) -> None` — Record msg, apply `joint_targets` — unless the topic's controller is inactive. (L128)
  - `state() -> dict[str, object]` — Current simulated joint state. (L152)
  - `deactivate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — Flip the table to `inactive`; refuse unknown names under STRICT; raise the `switch_faults["deactivate"]` exception when configured. (L167)
  - `activate_controllers(names, *, timeout_s) -> ControllerSwitchReport` — Mirror for the re-arm; `switch_faults["activate"]` likewise. (L173)
  - `call_trigger(service, *, timeout_s) -> TriggerReport` — Recorded; answered from `trigger_responses` (default: success). (L179)
  - `publish_empty(topic) -> None` — Recorded in `empty_publishes`. (L185)
  - `controller_state(name) -> str` — `active` / `inactive` / `unloaded`. (L189)
  - `call_count -> int` [@property] — Number of commands delivered to an active controller. (L196)
  - `last_call -> tuple | None` [@property] — The most recent delivered `(topic, msg)`. (L201)
  - `calls -> list[tuple]` [@property] — Every delivered `(topic, msg)`, in order. (L206)
  - `dropped_calls -> list[tuple]` [@property] — Every command that reached an `inactive` controller and was dropped. (L211)
  - `switch_calls -> list[tuple]` [@property] — Every `("deactivate" | "activate", names)` switch, in order. (L216)
  - `trigger_calls -> list[str]` [@property] — Every `std_srvs/Trigger` service called, in order. (L221)
  - `empty_publishes -> list[str]` [@property] — Every `std_msgs/Empty` topic published on, in order. (L226)
- `class SimTorqueSeam` — In-memory `InterbotixStopSeam`: a simulated `xs_sdk` torque table. `torque_enable(robot_name, *, group, enable, timeout_s) -> TriggerReport` flips the arm's torque, refuses an arm not in `arms` (absent service) or listed in `failing`, and raises the exception `faults` maps an arm to (the rclpy-shaped fault the HAL must contain while still stopping the other arms); `torque(name)` / `calls` for assertions. (L264)
  - `torque_enable(robot_name, *, group, enable, timeout_s) -> TriggerReport` — Record the call and flip the arm's torque unless it is absent, listed in `failing`, or mapped in `faults` (then raise that exception). (L304)
  - `torque(robot_name) -> bool` — Whether the arm is currently torqued on. (L320)
  - `calls -> list[tuple[str, str, bool]]` [@property] — Every `(robot_name, group, enable)` call, in order. (L325)

### `python/hal/src/openral_hal/lifecycle.py`
_Generic ROS 2 managed lifecycle node wrapper for every HAL adapter — UR5e / UR10e / Franka / SO-100 / OpenArm / H1 / future HALs all share the same publish / subscribe / heartbeat / OTel-span wiring._

- `class HALLifecycleNodeBase(LifecycleNode)` — Public base class. Owns the standard `/joint_states` publishers, the `/openral/safe_action` + `/openral/estop` subscribers, the 1 Hz `DiagnosticsHeartbeat`, the per-tick `hal.read_state`/`hal.send_action` OTel spans, the estop latch, and the full configure → activate → deactivate → cleanup → shutdown transition wiring. (L348)
  - `_create_hal(self) -> HAL` — Subclass hook (required): construct and return a HAL instance, reading ROS-parameter-driven constructor args via `self.get_parameter(...)`. (L425)
  - `_heartbeat_extra_fields(self) -> dict[str, str]` — Subclass hook (optional): extra key/values for the `/diagnostics` payload. Default `{}`. (L437)
  - `on_configure_post_hal(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific setup after the HAL connects. Default `SUCCESS`. (L494)
  - `on_activate_post_subs(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific timers/publishers after the base wires its subs. Default `SUCCESS`. (L503)
  - `on_deactivate_pre_teardown(self) -> None` — Subclass hook (optional): stop robot-specific timers before base teardown. Default no-op. (L511)
  - `on_cleanup_pre_disconnect(self) -> None` — Subclass hook (optional): tear down robot-specific resources before `HAL.disconnect()`. Default no-op. (L518)
  - `shutdown_hal(self) -> None` — Disconnects the HAL on a signal-driven process teardown, since `rclpy`'s SIGINT handler raises out of `spin` without requesting the lifecycle `shutdown` transition, so `on_shutdown`/`on_cleanup` (and `HAL.disconnect`) would otherwise never run. Called from both `main()` factories' `finally`, before `destroy_node`; exactly-once and idempotent. Deliberately HAL-only — no publisher/timer teardown, since the rclpy context is already down. SIGKILL stays uncatchable. (L769)
  - `_publish_joint_state(self) -> None` — Timer callback: reads `self._hal.read_state()` under a `hal.read_state` span and publishes `/joint_states`; when the HAL exposes `read_policy_state`, also publishes `/openral/policy_state` once per new `ProprioFrame` (never a latched republish). Subclasses may override + call `super()` to extend (OpenArm does, for viewer-sync). (L903)
  - `_on_safe_action(self, msg) -> None` — `/openral/safe_action` callback. Decodes the `openral_msgs/ActionChunk` into an `openral_core.Action` and forwards via `_send_action_traced(action, source="safe_action")`. (L1004)
  - `_send_action_traced(self, action, *, source) -> None` — Forwards `action` to `self._hal.send_action` inside a `hal.send_action` span; `source` disambiguates the origin on the dashboard's Commands card. (L1118)
  - `_on_estop(self, msg) -> None` — `/openral/estop` callback. Ordered latch → stop → report: sets `_estopped` (so `_on_safe_action` drops commands), calls `_invoke_hal_estop`, then reports via `_emit_estop_telemetry` in a `finally` so an e-stop is counted even if the vendor stop path raises. Nothing is added ahead of the physical stop.
  - `_invoke_hal_estop(self) -> None` — Calls `self._hal.estop()` for HALs implementing `LifecycleEStopHAL`; on the expected `ROSEStopRequested` it reads `_downstream_stop_fields()` and logs ERROR "hardware estop completed" only when the HAL's `DownstreamStopReport` says `acknowledged`, else **FATAL** "hardware estop NOT acknowledged downstream — only the local latch holds"; any other exception is logged at fatal — the latch must survive a vendor stop-path failure.
  - `_downstream_stop_fields(self) -> dict[str, str]` — Flattens the HAL's `last_stop_report` (`DownstreamStopReporting`) for the log line and the `/diagnostics` heartbeat: `downstream_stop=acknowledged|unacknowledged|unproven` plus controllers / states / vendor stop / detail. `_heartbeat_status` merges these into the `estop latched` payload so an operator sees whether the physical stop was proven.
  - `_emit_estop_telemetry(self) -> None` — Emits the `openral.event.estop_requested` span event and increments `openral.hal.estop.count`. The only producer of either signal — the dashboard ingests OTLP, not `/openral/estop` topics, so this is the sole chokepoint every robot HAL shares on the actuation side. Never raises; telemetry must not disturb the stop path.
- `make_lifecycle_main(node_name, hal_factory) -> Callable[[], None]` — Builds a `main()` entry point for a zero-parameter HAL adapter, via a `_FactoryHALLifecycleNode` whose `_create_hal()` returns `hal_factory()`. Superseded for standard arms by `make_lifecycle_main_from_manifest`; retained for bespoke nodes. (L223)
- module const `_HALLifecycleNode = _FactoryHALLifecycleNode` (L1907) — Back-compat alias for the pre-rename internal factory node class; imported directly by several `packages/openral_hal_*` lifecycle tests.
- `class ManifestHALLifecycleNode(HALLifecycleNodeBase)` — Public generic manifest-driven lifecycle node. Reads `robot_yaml` + `hal_mode` + sensor knobs as ROS params and builds its HAL via `openral_hal.build_hal`, so a robot's construction kwargs come from `hal.parameters.defaults` — no bespoke `_create_hal` subclass needed. Attaches `SimSensorBridge` in `on_activate_post_subs`. Under `hal_mode:=real` it attaches a `RosControlTransport` to any `RosControlHAL` and drops this node's global `/joint_states` publisher so the vendor's `joint_state_broadcaster` stays the sole writer; since issue #295 it also attaches that transport as the HAL's `ControllerStopSeam` (`attach_controller_stop`, built from `controller_names()` / `vendor_stop_services()` / `vendor_stop_topics()`), refuses to configure (`ROSConfigError`) a drivable HAL that is not `ControllerStoppable`, warns when `controller_manager` does not yet list its controllers active, and attaches an `InterbotixXSTransport` to an `InterbotixStoppable` HAL (ALOHA) via `_attach_interbotix_transport`, closing both seams' helper nodes in `on_cleanup`; it opens `/openral/<robot>/reset_to_pose` iff the HAL exposes `reset_to_pose`. When a scene composition is declared, the named composer's MJCF is threaded in as the HAL's `mjcf_path`, read from the `scene_composition_json` ROS param — which takes precedence over the manifest's `scene_defaults.composition` fallback, so the scene owns its arena and the manifest describes the robot. When the manifest declares a planar base (`base_joints`), also attaches a `MobileBaseBridge`. A back-compat alias `_ManifestHALLifecycleNode` is retained. (L1368)
- `make_lifecycle_main_from_manifest(node_name) -> Callable[[], None]` — Builds a `main()` that spins up `ManifestHALLifecycleNode`, reading `robot_yaml` + `hal_mode` ("sim"|"real") ROS params and constructing the HAL via `build_hal(description, mode=hal_mode)` — one node class serves both modes for every robot. `deploy sim`/`deploy run` inject the respective mode; a robot lacking it raises `ROSCapabilityMismatch`. (L281)
- `decode_action_chunk(msg) -> Action | None` — Inverse of `ros_publishing_hal._flatten_action_payload`: decodes the `ActionChunk` wire shape back into a typed `openral_core.Action` with the per-mode payload populated. Returns `None` for degenerate chunks or modes the F1/F5 publisher doesn't encode (`CARTESIAN_POSE`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT`). Preserves `joint_names` (ADR-0102; absent decodes to `None`, a whole-vector action) and `cartesian_delta_scale` (predictive-safety metadata only — the raw delta passes through unchanged). (L81)

### `python/hal/src/openral_hal/sim_bringup.py`
_Resolve a `SimScene` or `BenchmarkScene` YAML path to a live `SimRollout`. Used by `build_hal`, which every manifest-driven node (incl. panda_mobile) routes through._

- `build_sim_env_from_yaml(sim_env_yaml: str, *, robot_id_fallback: str | None = None) -> tuple[SimRollout, int | None]` — Loads a `SimScene`/`BenchmarkScene` YAML, resolves its scene id in `openral_sim.SCENES`, and instantiates the env; relative paths resolve against the source file's parents (ROS param values are cwd-naïve). Returns `(env, seed)` for `SimAttachedHAL(env_reset_seed=seed)`. Raises `ROSConfigError` if the YAML is missing, the scene id is unregistered, or validation fails. Robocasa scenes get `ignore_done=True` injected so deploy-sim continuous stepping doesn't trip the episode-done guard. (L165)
- `hal_transition_timeout_s(deploy_config: str | None) -> str` — Per-transition budget for the HAL's `tools/lifecycle_autostart.py`, derived from the scene: `max(300.0, scene.backend_options.boot_timeout_s + 60.0)`. Falls back to the 300 s floor for no config, an unreadable path, or a malformed value. A sidecar backend boots the simulator inside `on_configure`, so this bounds the same call `build_sim_env_from_yaml` makes. (L268)
- const `HAL_TRANSITION_TIMEOUT_FLOOR_S = 300.0` / `HAL_TRANSITION_TIMEOUT_MARGIN_S = 60.0` — The floor covers a robocasa-kitchen first boot; the margin covers the import + first reset following a sidecar handshake. Also the ceiling on how long a wedged HAL stays unreported. (L262)
- const `HAL_TRANSITION_TIMEOUT_MARGIN_S = 60.0` (L265) — see above.
- module const `_INDEX_PARSING_SCENE_PREFIXES` (L60) — LIBERO scene-id prefixes whose `task.id` parses as `"<suite>/<int>"`, requiring a valid integer index rather than tolerating the inert `_hal_deploy_noop` suffix other backends ignore.

### `python/hal/src/openral_hal/sim_attached.py`
_`SimAttachedHAL` — generic HAL Protocol adapter that wraps any in-process `SimRollout`. Shared by `panda_mobile`, manifest-driven arms, and tests; not import-safe without `openral_sim` + `mujoco`._

- module const `_TASK_SUCCESS_LOGGER` (L64) — dedicated `openral.sim.task_success` structlog logger (dotted under `openral.` so the OTel bridge attaches).
- module const `_EVENT_TASK_SUCCESS` (L69) — the `sim.task_success` event name `_emit_task_success` logs.
- module const `_EVENT_TASK_SUCCESS_FINAL` (L70) — the `sim.task_success_final` event name, emitted from `disconnect`.
- module const `_EVENT_TASK_SUCCESS_PROBE_FAILED` (L71) — the `sim.task_success_probe_failed` event name, emitted once when the backend's success predicate raises.
- module const `_ROBOSUITE_GROUP_PREFIX` (L108) — robosuite body-name prefix (`robot0_`/`robot1_`) stripped by `normalized_joint_index`'s prefix-fallback.
- module const `_PLANAR_TWIST_EPS` (L153) — tolerance for detecting a non-planar `BODY_TWIST` component.
- module const `_PENDING_GROUP_STALE_NS` (L160) — how long `idle_step` waits before loudly discarding a pending mid-flight atomic action group.
- `ActionPacker` — Type alias `Callable[..., np.ndarray]`. Per-composition translator between an OpenRAL `Action` and the env's flat action vector; default factory is `pack_action_for_env`. The optional trailing `prev` arg lets a packer carry an untouched slot (e.g. the gripper while the arm steps) across a split policy step. Pass a custom instance to `SimAttachedHAL.__init__` for whole-body/dexterous-hand layouts. (L170)
- `normalized_joint_index(model_joint_names: list[str]) -> dict[str, int]` — Maps MJCF joint names (exact + robosuite-prefix-stripped) to model index; exact names always win, `robot0_joint1` → `joint1` is added only when it doesn't shadow or collide. Used by `SimAttachedHAL.read_state` so one manifest serves both native MjSpec and robosuite scenes. (L111)
- `pack_action_for_env(action: Action, description: RobotDescription, env_action_dim: int, prev: np.ndarray | None = None) -> np.ndarray` — Default `ActionPacker`: translates `JOINT_POSITION`, `BODY_TWIST` (slots 0-2), `CARTESIAN_DELTA` (arm slots), and `GRIPPER_POSITION` (last slot) into the env's flat action vector. Raises `ROSConfigError` for unsupported modes or mismatched row widths. `prev` carries an untouched slot (e.g. gripper) across a policy step that splits into two Actions on a non-composite env — without it each Action zeroed the other's slots. (L188)
- `class SimAttachedHAL` — HAL Protocol adapter wrapping an in-process `SimRollout`. Reads live joint state via `normalized_joint_index` + `mj_name2id`; sends actions via `pack_action_for_env` (or a caller-supplied `ActionPacker`) into `env.step()`. Exposes `read_images()`, `mujoco_handles()`, `sim_time_ns()`, `base_pose`, `base_twist`, `base_pose_6dof()` for the lifecycle node's camera, viewer, clock, and odom wiring. (L349)
  - `__init__(env: SimRollout, description: RobotDescription, *, action_packer: ActionPacker | None = None, env_reset_seed: int | None = None, env_action_dim: int | None = None, body_twist_dt_s: float = 0.05) -> None` (L378)
  - `connect() -> None` — Resets the env at `env_reset_seed`; probes `env_action_dim` via `_probe_env_action_dim`, raising `ROSConfigError` naming the backend if not introspectable and no override given. Idempotent. (L512)
  - `disconnect() -> None` — Emits the terminal `sim.task_success_final` line, then releases the env handle (idempotent). (L583)
  - `read_state() -> JointState` — Walks `description.joints`, resolves each via `normalized_joint_index`, reads live `qpos`/`qvel` from MJCF. (L604)
  - `send_action(action: Action) -> None` (L746) — Packs the action via composite-split or `ActionPacker`, calls `env.step`. `BODY_TWIST` Euler-integrates qpos directly on a MuJoCo backend, else routes through `_apply_body_twist_via_env_step`. Stamps `last_action_ns` so the idle stepper yields.
  - `_stage_action_group(action, group_step) -> None` — For backends exposing `action_group_size`/`step_action_group`, stages each safety-approved slot by `Action.tick_index` and commits one sim step only when the complete tick is present (atomicity); a newer tick arriving with an incomplete group raises `ROSRuntimeError`. Latches the group's BODY_TWIST slot into `_last_body_twist` on commit.
  - `read_policy_state() -> list[float] | None` (L1953) — Cached simulator-native checkpoint proprioception (e.g. BEHAVIOR's 61-D R1Pro vector) for `/openral/policy_state`; only from an explicit `obs["policy_state"]`, never inferred.
  - `reset_estop() -> None` (L1949) — Clear the e-stop latch; caller asserts the cause has been resolved.
  - `idle_step() -> bool` — Sim-only free-running stepper: advances the `SimRollout` one HOLD tick (`np.zeros(env_action_dim)`) so cameras keep rendering when idle. Returns `False` when not connected, estop-latched, or no live MuJoCo handles. Defined **only** on `SimAttachedHAL` — real HALs never define it, since a zero vector is HOLD in sim but "drive to 0 rad" on a real arm. (L1189) Refuses to interleave a HOLD inside a mid-flight atomic action group, discarding a stalled pending group after 5 s.
  - `read_images() -> dict[str, Any]` — Return latest rendered camera frames keyed by camera name from the cached `_last_obs`. (L1769)
  - `read_depth_clouds() -> dict[str, NDArray]` (L1798) — Per-depth-sensor `(N,3)` `base_link` point clouds from `_last_obs["depth_points"]` (a non-MuJoCo backend like Isaac deprojects them, so the HAL never re-derives geometry); `{}` when the backend renders no depth. `SimSensorBridge` publishes them as `PointCloud2` for octomap.
  - `update_attached_objects(objects: list[AttachedCollisionObject]) -> None` (L678) — Atomically resolves a complete sim attachment snapshot from `evidence_ref="mujoco_body:<name>"`, including descendant bodies; preserves the previous snapshot on malformed/unknown identities.
  - `read_attached_objects() -> list[AttachedCollisionObject]` (L733) — Stable payload state `SimSensorBridge` unions with the robot self-filter so carried objects are absent from world occupancy.
  - `read_attached_body_ids() -> frozenset[int]` (L737) — Exact MuJoCo body ids of everything currently attached.
  - `add_post_step_observer(observer: Callable[[], None]) -> None` (L741) — Registers an idempotent synchronous hook after each real simulator transition; automatic attachment evidence uses this so grasp/release changes publish before the next policy tick is acknowledged.
  - `read_scan() -> NDArray | None` (L1816) — 2-D LaserScan range fan (`base_link`) from `_last_obs["scan"]` on a non-MuJoCo backend that ray-casts a lidar (Isaac); `None` otherwise. Read by `SimSensorBridge._compute_scan_ranges` for `/scan`.
  - `mujoco_handles() -> tuple[Any, Any] | None` — Forward the env's `(model, data)` MJCF handles. (L1644)
  - `task_success() -> bool | None` (L1033) — Forwards the wrapped rollout's optional `task_success()` predicate (RoboCasa: the task's own `_check_success()`); `None` means "no predicate, or one that raised", never "failed". Observability only — feeds no termination/reset/reward/action path. Logged as `sim.task_success` (both edges, since RoboCasa success isn't latched) and `sim.task_success_final` at disconnect. Full signal table in [`docs/reference/telemetry.md`](../reference/telemetry.md).
  - `last_committed_tick -> int` [@property] (L1656) — Most recent complete atomic action group committed to the simulator.
  - `sim_time_ns() -> int | None` (L1691) — Cross-reconnect-monotonic elapsed sim time in ns; `connect` folds elapsed time into an offset so a lifecycle reset doesn't rewind it.
  - `clock_authority() -> ClockAuthority` (L1716) — Returns `ClockAuthority.simulation(<backend>, timestep_s=body_twist_dt_s)` when `sim_time_ns()` is live, else `ClockAuthority.host_wall()`.
  - `estop() -> None` — Latch e-stop; subsequent `send_action` calls are dropped. (L1636)
  - **(property)** `env -> SimRollout` (L1749) — Direct access to the wrapped sim env (for tests / advanced wiring).
  - **(property)** `estop_latched -> bool` (L1754)
  - `base_pose -> tuple[float, float, float]` [@property] — Current `(x, y, yaw)`, from MJCF qpos on a MuJoCo backend or `obs["base_pose"]` on Isaac; `(0,0,0)` if the backend reports neither. Feeds the `/odom` publisher. (L1833)
  - `base_twist -> tuple[float, float, float, float, float, float]` [@property] — Last commanded body twist `(vx, vy, vz, wx, wy, wz)`. (L1886)
  - `_apply_body_twist_via_env_step(row: list[float]) -> None` — Non-MuJoCo `BODY_TWIST`: validates the planar twist, latches it for `/odom`, packs `(vx, vy, wz)` into the scene's final base-twist slots, zeros arm/gripper, and steps.
  - `base_pose_6dof() -> tuple[...] | None` — Full 6-DoF `(xyz, quat_xyzw)` from robocasa `raw_proprio`; `None` for non-robocasa backends. (L1896)
  - `last_action_ns -> int` [@property] — Monotonic ns of the last real action through `send_action`; `0` until the first one. Read via `should_idle_step` to yield the idle stepper to an active skill. (L1759)

### `python/hal/src/openral_hal/convex_distance.py`
_Certified signed distance between two MuJoCo convex geoms — replaces `mujoco.mj_geomDistance`, which under mujoco 3.8.0 returns confidently wrong values (silently) on some geom pairs the collision-evidence probes adjudicate. Represents each geom as `conv(core) ⊕ ball(radius)`, solving via GJK (separated cores) or exact SAT (overlapping), with curved geoms bracketed by inscribed/circumscribed polytopes; runs only in evidence collection (~2-4 ms/pair), never on the 100 Hz path._

- `ConvexDistance` [@dataclass(frozen=True)] — One certified signed distance with its proof: `distance_m` (the closest end of the bracket — never overstates separation; negative is penetration depth), `lower_m`/`upper_m`, `certified`, `uncertified_reason`, `witness_a`/`witness_b`, `direction`, `duality_gap_m`, `method`. `direction` (unit `b`→`a`, or the MTD axis when overlapping) is the only reliable contact direction at a flush contact, where the two witnesses coincide. `as_record()` serialises non-finite bounds as `None`, never `Infinity`, for strict-JSON round-tripping. (L93)
- `ConvexDistance.as_record() -> dict[str, object]` (L140) — see above.
- module const `_ARC_SEGMENTS: Final[int] = 256` (L78) — default polygonisation segment count for a round-type (cylinder/ellipsoid) bracket.
- module const `_BRACKET_TOL_M: Final[float] = 1e-4` (L80) — certify a bracket this tight or tighter.
- module const `_DUALITY_TOL_M: Final[float] = 1e-9` (L82) — certify a separating-axis duality gap this tight or tighter.
- module const `_OVERLAP_TOL_M: Final[float] = 1e-12` (L86) — below this, GJK's terminal simplex has collapsed onto the origin (overlap is SAT's branch, not GJK's).
- module const `_MAX_SAT_AXES: Final[int] = 8_000_000` (L89) — refuse an exact penetration depth beyond this many SAT axes rather than spend unbounded time or silently subsample.
- `ConvexBody` [@dataclass(frozen=True)] — A geom as `conv(core) ⊕ ball(radius)` plus the `faces` SAT needs; `faces` is `None` for a visual mesh (no compiled hull graph), costing nothing until the cores actually overlap. (L167)
- `convex_geom_distance(model, data, geom_a: int, geom_b: int, *, distmax_m: float | None = None, arc_segments: int = 256) -> ConvexDistance` — The instrument. `distmax_m` enables a certified window rejection (`method="beyond-window"`) via a separating-axis bound valid in every direction. `certified` is `False`, with a reason, for a geom with no bounded convex hull, a round-type bracket wider than 0.1 mm, or an unclosed/oversized-axis-set overlap; no plausible number is ever substituted for a defensible one. (L586)
- `witness_clearance_m(model, data, geom_id: int, point, *, arc_segments: int = 256) -> float` — Certified lower bound on how far `point` lies outside `geom_id`; positive proves it's outside. Used to contradict a claimed nearest-point witness (both ends of a real nearest-pair segment lie on their own geoms). Falls back to the bounding sphere for a geom with no hull, `-inf` for one with neither. (L724)
- `separating_axis_bound(points_a, points_b, direction) -> float` — Weak-duality lower bound on the distance along one direction: `min_B u·b − max_A u·a`. Evaluated at the direction a witness claims, turning a GJK answer into a proof. Returns `-inf` for a degenerate direction. (L449)
- `mesh_hull(model, mesh_id: int) -> tuple[Any, Any | None]` — The hull MuJoCo itself collides, `(vertices, faces)` in mesh-local frame, decoded from the compiled hull graph rather than recomputed. `faces` is `None` for a mesh with no graph — distance stays exact, only penetration depth loses its axis set. (L188)
- `geom_convex_bracket(model, data, geom_id, *, arc_segments=256) -> tuple[ConvexBody, ConvexBody] | None` — World-frame `(inscribed, circumscribed)` `ConvexBody` pair bracketing one geom (exact for box/mesh/sphere/capsule, polygonised for cylinder/ellipsoid); `None` for a geom with no bounded convex hull. (L280)

### `python/hal/src/openral_hal/sim_sensor_bridge.py`
_Shared sim-sensor + viewer bridge for scene-attached HAL lifecycle nodes: republishes RGB camera frames, `/scan`, depth `PointCloud2` (MuJoCo ray-cast, or a non-MuJoCo backend's own clouds), a live MuJoCo viewer, and the sim-only idle stepper for any manifest-driven node. Also owns the attachment-state, ADR-0097 place-declaration, and E-stop ground-truth snapshot wiring; a place declaration naming a target absent from the scene is refused and logged, and its region/geometry is always this producer's own measurement, never a relayed value. rclpy imported lazily._

- `should_idle_step(now_ns: int, last_action_ns: int, idle_hold_ns: int, *, step_while_active=False) -> bool` — Pure predicate for the sim-only wall-time stepper: scene-attached envs return `True` only after the idle-hold window; bare `MujocoArmHAL` passes `step_while_active=True` so its target keeps integrating during active skills. Used by `SimSensorBridge._idle_step_tick`. (L146)
- module const `_THUMB_INTERVAL_NS = 1_000_000_000` (L35) — throttle dashboard thumbnail emission to ~1 Hz per camera; the live ROS topic itself stays at `camera_rate_hz`.
- module const `_IMAGE_DIM = 3` (L36) — HWC ndarray rank.
- module const `_RGB_CHANNELS = 3` (L37)
- module const `_MAX_PREATTACH_CELLS = 2_000_000` (L41) — cap on the pre-attach occupancy snapshot; an over-large grid is refused/truncated rather than partially sampled.
- module const `_DEGENERATE_QUAT_NORM = 1e-12` (L45) — below this a quaternion carries no usable direction; keep identity rather than divide by ~0.
- module const `_NEAREST_PROBE_DISTMAX_M = 0.10` (L59) — near-miss probe bounding-sphere prefilter gap for the e-stop ground-truth snapshot.
- module const `_NEAREST_PROBE_MAX_CALLS = 4096` (L60) — exact-distance call budget for one snapshot's near-miss probes.
- module const `_NEAREST_PROBE_MAX_PAIRS = 32` (L61) — closest-pairs cap reported per probe.
- module const `_VOXEL_BACKING_RAYS_PER_AXIS = 3` (L68) — rays per cube axis for `voxel_backing_record`'s ray-fan sampling.
- module const `_QUAT_NORM_TOL = 1e-6` (L72) — tolerance on `|q|^2` for a published grid orientation; wide enough for a wire round-trip, too tight for an unset all-zero field.
- module const `_VOXEL_BACKING_EPS_FRACTION = 0.01` (L76) — ray start stand-off outside a voxel cube face, as a fraction of cell size.
- module const `_VOXEL_BACKING_MAX_GEOMS = 8` (L79) — cap on backing geoms reported per cell.
- module const `_XYZ = 3` (L81) — Cartesian dimension count for a grid size / origin / half-extent triple.
- module const `_CONTACTS_CAVEAT` (L86) — the `robot_world_contacts==0 does NOT mean no interpenetration` string emitted verbatim on every `estop_ground_truth_snapshot` stop line.
- module const `_CANDIDATE_CHUNK_HISTORY = 3` (L102) — candidate chunks retained (ring buffer) for predicted-horizon reconstruction.
- module const `_ESTOP_EVIDENCE_WINDOW_NS = 500_000_000` (L105) — a `CollisionEvidence` older than this is not attributed to the stop being snapshotted.
- module const `_DOP_VERTEX_EPS = 1e-9` (L910) — slack (metres/determinant) for the 3x3 solves behind `_dop_vertices`.
- module const `_VOXEL_BACKING_MAX_LAYERS = 4` (L1201) — how many non-collidable shells one ray looks past inside a single cell before giving up.
- `constant_scan_no_hit_ranges(*, n_beams: int, max_range_m: float) -> list[float]` — Pure synthetic `/scan` fan: every beam clamped to `max_range_m` ("no hit everywhere"), the honest reading for an in-process digital twin with no scene to ray-cast. Used by `SimSensorBridge._compute_scan_ranges`'s no-handles branch. (L130)
- `estop_ground_truth_snapshot(model, data, *, robot_body_ids: frozenset[int], attached_body_ids: frozenset[int] = frozenset(), probe_body_ids: frozenset[int] | None = None, base_frame_body: str | None = None, joint_state: Any = None, description: Any = None, evidence_voxel: Any = None, distmax_m: float = 0.10, max_pairs: int = 32, max_calls: int = 4096) -> dict[str, object]` — Pure MuJoCo ground truth for one safety stop (attached-payload or robot-world): contacts, poses, and near-miss probe pairs across robot↔world, payload↔world, payload↔robot-link, and link↔link. Contact lists are not a penetration oracle — contype/conaffinity exclusions can suppress a real contact entirely — so the near-miss probes are the adjudicator, measured via `convex_geom_distance` (never `mujoco.mj_geomDistance`, which returns confidently wrong values on some pair classes under mujoco 3.8.0). Every probe side is filtered to solid geometry only (a geom with neither `contype` nor `conaffinity` is never a real penetration) and reports its own coverage (candidate/probed pairs, certified/uncertified, noncollidable exclusions) so an absent pair reads as "looked and found nothing," not silence. `description` adds an `adjudication_budget` (mesh-vs-model slop) so a kernel OBB↔voxel distance and a probe mesh↔mesh distance can be compared fairly, widening (never narrowing) the probe window to the admissible gap; `evidence_voxel` adds `evidence_voxel_backing` (the only part of this record that reads the published map rather than MuJoCo alone) and must only be passed for evidence judged fresh. Called by `SimSensorBridge._on_estop_ground_truth`. (L1997)
- `kernel_checked_body_ids(model, description) -> frozenset[int]` — Pure MJCF bodies for exactly the links the safety kernel collision-checks (the manifest's `collision_geometry` is the kernel's model, so a link with no entry is deliberately invisible). Resolved exactly via each joint's `sim_joint_name`, never name-mangled; empty when the manifest declares no collision geometry. Scopes the near-miss probes so a mobile base's floor contacts can't bury the arm's. Sibling of `depth_cloud.robot_self_body_ids` (that answers "what is the robot"; this answers "what does the kernel check"). (L650)
- `kernel_checked_link_bodies(model, description) -> dict[str, int]` — Pure name-keyed form of `kernel_checked_body_ids`, resolving each `collision_geometry` link name to its MJCF body id the same exact way. Needed wherever a diagnostic must line a manifest OBB up with its MuJoCo geometry (`collision_model_mesh_slop` is the caller); unresolved links are reported, not guessed. (L675)
- `collision_model_mesh_slop(model, description) -> dict[str, object]` — Pure measurement of how far the kernel's manifest-OBB collision model reaches beyond the real meshes per link, so a kernel distance (OBB↔voxel) and a probe distance (mesh↔mesh) can be compared fairly — the admissible gap is `corner_slop(link) + voxel_half_diagonal`. Returns per-link `obb_half_extents_m`/`face_slop_m`/`corner_slop_m`/`hull_overhang_m` (`has_stage2_hull` distinguishes "no hull" from "unmeasured") plus `max_corner_slop_m`; `{}` when the manifest declares no collision geometry, so the caller reports "no budget" rather than assuming zero. Robot side only — an attached-payload self stop also needs `attached_payload_mesh_slop`. (L752)
- `_dop_vertices(dop_lo, dop_hi) -> NDArray[np.float64]` — Vertices of a 26-DOP from its 26 halfspaces, by direct plane-triple enumeration (no hull-intersection library). Needed because distance-to-a-convex-set is a convex function, so its maximum over a polytope occurs at a vertex — the same reason a box's slop term is its corner, not something face-sampled.
- `_payload_model_overhang_m(prim, points) -> float` — How far one payload primitive's kernel model (its DOP where one ships, else the box) reaches past the real mesh — the payload-side analogue of a link's `hull_overhang_m`, used by a world-voxel payload stop's budget. Deliberately the DOP and not a tighter stage-2 hull even when one ships, since the kernel falls back to the DOP whenever its hull budget caps out. `inf` for a degenerate refinement, so the consumer reads "no budget" rather than scoring against zero.
- `attached_payload_mesh_slop(model, data, *, attached_body_ids: frozenset[int], max_primitives: int = 16) -> dict[str, object]` — Pure measurement of how far a carried payload's kernel primitives reach beyond its own meshes — the payload side of the adjudication budget an attached-payload self stop needs (that check is OBB-vs-OBB on both sides, unlike a world-voxel stop's single OBB). Calls `extract_body_primitives` directly so the budget can't drift from the geometry the kernel was actually handed; only box primitives can be loose (sphere/capsule primitives have no corner to overhang). Returns `{"objects": {<body>: {n_primitives, n_box_primitives, corner_slop_m, ...}}, "max_corner_slop_m", "unresolved_objects", "method"}`; `{}` when nothing is carried, so the caller reports "no budget" rather than assuming zero. (L1015)
- `voxel_backing_for_cell(model, data, cell: Mapping[str, Any], *, robot_body_ids: frozenset[int], attached_body_ids: frozenset[int], base_frame_body: str | None) -> dict[str, object] | None` — The single place a cached `/openral/world_voxels` cell dict is unpacked into `voxel_backing_record`'s arguments, so the in-snapshot and late-evidence call sites can't disagree about which fields (notably `grid_orientation_xyzw`) they pass — omitting it silently places the probe cube by the wrong axes on a rotated base. Returns `None` when the cell names no index. (L1950)
- `voxel_backing_record(model, data, *, voxel_index: int, grid_origin: Sequence[float], grid_resolution: float, grid_size: Sequence[int], robot_body_ids: frozenset[int], attached_body_ids: frozenset[int] = frozenset(), base_frame_body: str | None = None, grid_orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0), rays_per_axis: int = 3) -> dict[str, object]` — Pure answer to what MuJoCo geometry, if any, backs one occupancy voxel — the near-miss probes never look at `/openral/world_voxels`, so this is the only check of whether the map and the world agree. Decodes the kernel's `b=voxel_<n>` index against the live grid, lifts it to world coordinates, then classifies via ray-fan sweep as `solid_world`, `attached_payload`, `self_occupancy_suspect`, `noncollidable_world`, or `unbacked`, with `verdict` picking by adjudication order (real world geometry always explains a stop first). `rays_cast`/`rays_hit` make `unbacked` readable as "looked and found nothing"; an index outside the grid returns `verdict: "out_of_range"`. (L1411)
- `attached_model_slip(model, data, *, description: Any, attached_objects: Sequence[Any]) -> list[dict[str, object]]` — Pure answer to where the kernel *thinks* each carried payload is (forward-kinematics from `attach_link` + `pose_in_link`, fixed at the attach) versus where the simulator actually has it — a payload that slips, settles, or pivots in the gripper is then charged as collision depth against cells the real object isn't near. One record per object: `object_id`, `attach_link`, `resolved`, `kernel_world_xyz`, `sim_world_xyz`, `slip_m`, `rotation_deg`, `body_radius_m`, `max_point_slip_m` (translation plus the rotation's sweep at the body's radius, since `slip_m` alone understates a pivoting payload). An object whose link/body/pose can't be resolved reports `resolved: False`, never a silent zero. Emitted on every ground-truth snapshot as `payload_model_slip`. (L1707)
- `grid_with_origin(history: Sequence[Mapping[str, object]], origin: Sequence[float], *, tol_m: float = 1e-9) -> Mapping[str, object] | None` — The cached `/openral/world_voxels` geometry whose published origin matches the one the kernel discloses (`CollisionEvidence.world_grid_origin_m`); exact where `grid_current_at`'s stamp match is ambiguous, since two grids published within one `/clock` tick can be indistinguishable by stamp while naming different cells. Newest match wins. `None` when the kernel disclosed nothing or no cached grid matches; the caller then falls back to the stamp proxy and records which mechanism resolved it (`grid_decode.decode_source`). (L1645)
- `grid_current_at(history: Sequence[Mapping[str, object]], stamp_ns: int) -> Mapping[str, object] | None` — The newest cached `/openral/world_voxels` geometry whose header stamp is at or before `stamp_ns`, chosen by stamp, never by arrival order: decoding a stop's cell index against whichever grid arrived last is wrong the moment the base drifts across one lattice boundary, misplacing `world_xyz` by a full grid resolution. `None` when nothing in `history` is old enough; the caller then falls back to the latest grid and records `fallback_to_latest: true` rather than papering over the gap. (L1678)
- `occupied_cell_keys(*, occupancy: Sequence[int] | Any, grid_origin: Sequence[float], grid_resolution: float, grid_size: Sequence[int], grid_orientation_xyzw: Sequence[float], base_rot: Any, base_offset: Any, max_cells: int = 2_000_000) -> tuple[frozenset[tuple[int, int, int]], bool]` — Pure reduction of one published `OccupancyVoxels` grid to the set of world cells (keyed by `floor(centre_world / resolution)`, not rounded or by voxel index — both are wrong once the base has moved) it marks occupied, so a later stop can ask whether its cell predates a given moment (e.g. a grasp). An occupied count above `max_cells` returns `(frozenset(), True)` rather than a partial set, since a partial set would wrongly answer "not pre-existing" for cells it never looked at. (L1804)
- `preattach_verdict(*, world_xyz: Sequence[float], resolution: float, preattach_keys: frozenset[tuple[int, int, int]] | None) -> dict[str, object]` — Did the cell a stop tripped on already exist before the payload was masked? Returns `{available, preexisting, within_one_cell, cell_key, preattach_cells}`, or `{available: False, reason}` when no snapshot was taken — never a bare `False`, which would misread as "the cell is new." `preexisting` alone is not a payload-authored finding (world geometry is always pre-existing too); that requires the conjunction with `voxel_backing_record`'s verdict being `attached_payload`. Consumed by `SimSensorBridge._annotate_preattach`. (L1892)
- `candidate_chunk_digest(*, stamp_ns: int, control_mode: int, horizon: int, n_dof: int, flat: Sequence[float], cartesian_delta_scale: Sequence[float] = (), ee_name: str = "", frame_id: str = "", rskill_id: str = "", trace_id: str = "", tick_index: int = 0) -> dict[str, object]` — Pure JSON-safe digest of one `openral_msgs/ActionChunk`, reshaped into `horizon` rows of `n_dof` under `ticks` (a mismatched length is kept verbatim with `shape_mismatch: true`, never truncated). The command-side record of a predicted-horizon stop — the configuration the kernel actually adjudicated is instead `CollisionEvidence.joint_positions_rad`, the authority. Cached per chunk by `SimSensorBridge._on_candidate_action`. (L2361)
- `initial_configuration_stop_record(snapshot: Mapping[str, object], *, stop_seq: int, last_action_ns: int, candidate_chunks_seen: int) -> dict[str, object] | None` — Pure classifier separating a scene-initialization stop from a mid-task one: `None` once anything has been applied (`last_action_ns != 0`), else a record naming the stop as one the scene reset produced rather than one the robot was driven into — no policy, chunk, or margin change can clear it. `candidate_chunks_seen` is reported but doesn't gate, since a chunk the kernel rejected never reached `send_action`. Diagnostics only — never suppresses, delays, or alters the stop. Called by `SimSensorBridge._log_initial_configuration_stop`. (L2425)
- `class SimSensorBridge` — Wires and tears down sim-sensor publishers and the MuJoCo viewer on a HAL lifecycle node, gated on the robot manifest + HAL capability. Owns `/scan` for both a live MJCF ray-cast (`SimAttachedHAL`) and a `constant_scan_no_hit_ranges` fan for the bare digital twin. (L2510)
  - `__init__(node: Any, hal: Any, description: RobotDescription, *, viewer_enabled: bool = True, camera_rate_hz: float = 10.0, viewer_sync_rate_hz: float = 30.0, scan_rate_hz: float = 10.0, scan_n_beams: int = 360, scan_max_range_m: float = 12.0, scan_min_range_m: float = 0.05, depth_rate_hz: float = 10.0, depth_max_range_m: float = 5.0, depth_pixel_stride: int = 4, idle_hold_ms: float = 2000.0, on_step: Any = None, on_attachment_perception_ready: Any = None) -> None` — `on_step` refreshes the proprio snapshot after idle sim steps. `on_attachment_perception_ready` releases a deferred grouped-action acknowledgement only once a newly attached payload is transparent in both a depth frame and a newer `/openral/world_voxels` raster.
  - `setup() -> None` — Activates every stream the manifest + HAL support: RGB camera + `CameraInfo` publishers (advertised lazily per camera on its first real frame — see `_advertise_camera`), the sim-only idle stepper, and the passive MuJoCo viewer (always `mjCAMERA_FREE`, opening pose from `initial_viewer_camera`; a GL/DISPLAY failure warns and continues). Idempotent per activate. (L2718)
  - `_advertise_camera(name: str) -> Any` — Creates (once) and returns the `Image` + `camera_info` publisher for one camera, on its first real rendered frame — never eagerly for the manifest's full camera list, since a manifest may declare a camera the current scene can't render and an eagerly-advertised topic would be indistinguishably silent forever. A declared camera that never yields a frame gets one warning and no topic. (L2847)
  - `teardown() -> None` — Cancel timers (incl. the idle-step timer), destroy publishers, close viewer. Called from `on_deactivate` / `on_cleanup`. (L2729)
  - `_setup_estop_ground_truth() / _on_estop_ground_truth(msg)` — Diagnostics only — nothing here gates, delays, or alters actuation. Gated on live MuJoCo handles, not attachment support, so a pre-grasp arm↔world stop still gets ground truth. Subscribes `/openral/estop`, `/openral/candidate_action` (a ring of cached `candidate_chunk_digest`s), `/openral/failure/safety` (the kernel's `CollisionEvidence`), and the `/openral/world_voxels` grid geometry, then emits one `sim.estop_ground_truth_snapshot` line per stop plus a separate `sim.estop_ground_truth_evidence` line when the kernel's evidence lands late (never delayed for it, since sim state must be captured at the stop instant). Also runs `_log_initial_configuration_stop`, emitting `sim.estop_initial_configuration` when nothing has been applied yet. Full signal table in [`docs/reference/telemetry.md`](../reference/telemetry.md).
  - `attachment_action_ack_ready() -> bool` (L4072) — Attachment transaction gate for the lifecycle node: an addition stays unacknowledged until kernel acceptance, one transparent depth frame, and the following voxel raster all land. An empty detach snapshot is immediately ready, since detach unmasks before the kernel drops the old payload.
  - `_on_attachment_state_applied(msg)` — Applies the staged revision once the kernel accepts it, then arms the perception barrier only when the revision masks geometry not already masked (compared exactly by `(object_id, evidence_ref)`); a revision masking nothing new (e.g. a re-published place witness) releases the barrier immediately instead of deadlocking a successful place.
- `describes_mobile_base(description: RobotDescription) -> bool` (`mobile_base_bridge.py` L40) — Pure predicate: does the manifest declare a planar base (`base_joints`)? The single answer to "does something already own `odom → base_frame`", so `MobileBaseBridge`'s attach and `SimSensorBridge`'s TF skip-guard can't drift apart; reading any other field risks a second `base_link` parent and a split `/tf` tree.
- `class MobileBaseBridge` (`mobile_base_bridge.py` L66) — Generic planar-mobile-base ROS wiring (sibling of `SimSensorBridge`): owns `/odom`, the `odom->base_link` TF, and the `/cmd_vel`→BODY_TWIST bridge (bypasses the safety supervisor — out of scope here). Frame ids come from `RobotDescription`; `ManifestHALLifecycleNode` attaches it in `on_activate_post_subs` iff the manifest declares `base_joints`.
  - `__init__(node, hal, description, *, odom_rate_hz: float = 20.0, cmd_vel_topic: str = "/cmd_vel", proprio: Any = None) -> None` — `proprio`: when set (sim-attached HALs), odom is published from the node's dedicated thread via `publish_from_snapshot`, reading the snapshot not the simulator; `None` (real HALs) keeps the legacy odom timer.
  - `setup() -> None` (`mobile_base_bridge.py` L109) — Create the `/odom` publisher + TF broadcaster + `/cmd_vel` subscription; the odom timer is created **only when `proprio is None`** (sim HALs publish odom off the node's thread).
  - `publish_from_snapshot() -> None` (`mobile_base_bridge.py` L163) — Dedicated-thread entry: publish one `/odom` + TF sample from the proprio snapshot (never the simulator). Thin alias over `_publish_odom` (which branches on `proprio`).
  - `teardown() -> None` (`mobile_base_bridge.py` L150) — Cancel the timer (if any) + destroy the publisher/subscription. Idempotent.

### `python/hal/src/openral_hal/proprio_snapshot.py`

Decouples the control-critical publishers (odom / joint_state / TF) from the single executor thread that runs `env.step` + render + raycast. The sim-attached HAL node captures a frame after each step (on the executor thread, where reading the sim is safe) and a dedicated publisher thread re-emits it at ~30 Hz, so odom stays fresh (~28 Hz live, vs ~1.8 Hz starved) without ever touching `MjData`/GL off the executor thread (a `MultiThreadedExecutor` was rejected — MuJoCo's GL context is thread-affine).

- `class ProprioFrame` (L38) — Frozen dataclass: one coherent proprio sample (`state`, `base_pose`, `base_pose_6dof`, `base_twist`, `sim_time_ns`, `policy_state`). Plain immutable data, no live simulator handles, so it's safe to publish from a different thread than the one that stepped the sim.
- `class ProprioSnapshot` (L66) — Lock-guarded holder for the latest `ProprioFrame`: the executor thread calls `set` after each step, the publisher thread calls `latest`; the frame is swapped under the lock so a reader never sees a torn frame or reaches the HAL directly.
  - `set(frame: ProprioFrame) -> None` (L94) — Atomically publish `frame` as the latest sample (executor thread only).
  - `latest() -> ProprioFrame | None` (L99) — Return the most recent frame, or `None` before the first capture.

### `python/hal/src/openral_hal/_sim_attachment_evidence.py`

_MuJoCo attachment evidence — the counterpart of `_vision_attachment_evidence.py`, emitting the identical `AttachedCollisionObject` contract. Only the contact-force and payload-lowering halves are inventoried here; support-contact and place-region predate this section._

- module const `_LOGGER` (L38) — module structlog logger.
- module const `_DEGENERATE_NORM = 1e-9` (L42) — below this a summed support normal carries no direction; used by `support_contact_witness`.
- module const `_SUPPORT_PROBE_GAP_M = 0.001` (L57) — beyond this gap nothing is resting on anything (the safety kernel's own `attached_contact_tolerance_m`).
- module const `_SUPPORT_MAX_PENETRATION_M = 0.01` (L63) — the safety kernel's `support_witness_max_penetration_m`; a claim past this fails the whole attachment message, so the producer never constructs one.
- module const `_SUPPORT_MAX_PATCH_RADIUS_M = 0.5` (L64) — the safety kernel's `support_witness_max_patch_radius_m` mirrored here.
- module const `_SUPPORT_PROBE_MAX_CALLS = 1024` (L74) — exact-distance call budget for one support attestation.
- module const `_CONTACT_FORCE_SCALE_ENV = "OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT"` (L87) — ADR-0100 contact-force scale env var read by `contact_force_calibration`.
- module const `_CONTACT_FORCE_CALIBRATION_REF_ENV = "OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF"` (L88) — companion calibration-reference env var; both must be set for `calibrated=True`.
- module const `_CONTACT_FORCE_DEFAULT_SCALE = 1.0` (L89) — identity scale used when calibration env vars are unset.
- module const `_NORMAL_AGREEMENT_MIN = 0.5` (L96) — minimum cosine agreement between a surface normal and the probe's own contact direction before a support hit is attested.
- `class SimObjectMobility(str, Enum)` (L99) — Kinematic mobility of a contacted non-robot MuJoCo body: `FREE`, `HINGE`, `SLIDE`, `ARTICULATED`, `FIXED`. Returned by `classify_body_mobility`.
- `subtree_region_box(model, data, *, root_body_id) -> tuple[NDArray, NDArray] | None` (L197) — Sim producer for a declared place target's `PlaceRegion`: bounds one body's whole collision subtree by a box axis-aligned in the declared body's own frame, computed once at declaration time from the model, never inferred from the payload's position. Returns `(centre_in_body_frame, half_extents)`, or `None` when the subtree has no (or degenerate) collision geometry.
- `classify_body_mobility(model, body_id) -> SimObjectMobility` (L257) — Classifies the contacted body's root as `FIXED` (no joint), `FREE`, `HINGE`/`SLIDE` (all one joint type), or `ARTICULATED` (mixed).
- `geom_is_polytope(model, geom) -> bool` (L361) — Is `conv(geom_surface_points(geom))` the geom's own solid? True for mesh/box, false otherwise: for a curved surface (e.g. a sphere sampled at its poles) that containment runs the wrong way and can under-measure a payload's real extent past a voxel's half-diagonal. Gates both lowering paths; in `_clustered_box_primitives` one ungroundable geom disqualifies the whole cluster.
- `geom_surface_points(model, geom) -> NDArray[np.float64]` (L407) — Points lying on one geom's own surface in its local frame (mesh vertices; box corners; sphere/capsule poles); empty for geometry with no enumerable surface (plane, heightfield, SDF). Deliberately not a bounding cube, which would overstate the geometry and understate the slop both callers measure.
- `_tight_geometry_from_points(points, half_extents) -> TightCollisionGeometry | None` — Refines one payload box into the kernel's staged 26-DOP + exact hull online at attach (the payload twin of `tools/generate_tight_geometry.derive_tight_geometry`), so that `mesh ⊆ hull ⊆ DOP ⊆ box` holds exactly. No convex-hull solver runs — stage 2 is a support scan over the deduplicated points. Over `MAX_TIGHT_HULL_VERTICES` it returns the DOP alone, disclosed by an empty vertex tuple, never a truncated hull. `None` only when there are fewer than four surface points to refine.
- module const `_HULL_DEDUP_CEILING = 2 * MAX_TIGHT_HULL_VERTICES` (L470) — bounds the sort `_tight_geometry_from_points` runs when deduplicating hull vertices.
- `extract_body_primitives(model, data, *, root_body_id, object_id, max_primitives=16) -> list[AttachedCollisionPrimitive]` (L819) — Extracts conservative collision primitives for one unknown sim object: one per collision geom in the body's subtree, or `_clustered_box_primitives` when there are more geoms than `max_primitives`. Called by `collision_model_mesh_slop`/`attached_payload_mesh_slop` to keep their slop budget from drifting off the kernel's actual geometry. Raises `ROSConfigError` when the subtree has no collision geometry.
- `support_contact_witness(model, data, *, root_body_id, robot_body_ids, stamp_ns, support_roots=None) -> SupportContactWitness | None` (L1206) — Attests one payload's bounded support contact from certified signed distances (`openral_hal.convex_distance`, not the solver's contact list or `mj_geomDistance`), so a grasped object resting on the counter it was picked from isn't indistinguishable from driving it through a wall. Only a non-free environment body counts as a support; the place phase restricts `support_roots` to the declared target's own subtree. `None` when no eligible support contact exists.
- `contact_force_calibration() -> tuple[float, bool, str | None]` (L1336) — Resolves the sim contact-force calibration from `OPENRAL_SIM_CONTACT_FORCE_N_PER_UNIT` + `OPENRAL_SIM_CONTACT_FORCE_CALIBRATION_REF`; `calibrated=True` only when both are set and the scale parses finite and positive — a sim magnitude is a calibration knob, not a newtons claim, and a scale nobody can name must not silently arm the kernel's force gate.
- `probe_contact_force(model, data, *, payload_geoms, target_root_body, target_name, stamp_ns) -> ContactForceWitness | None` (L1380) — Walks MuJoCo's solver contact list for payload↔declared-target pairs and attests the total normal load via `mj_contactForce` (never just the largest single contact, which understates a multi-corner rest). `None` is not evidence of absent contact — contype/conaffinity exclusions can suppress a pair entirely — it only means the gate does not arm; that blind spot is survivable because this check can only add a refusal, never remove one.
- `class SimAttachmentEvidenceTracker(model, description, *, stable_ticks=3, release_ticks=3, translation_tolerance_m=0.01, rotation_tolerance_rad=0.15)` (L1502) — Confirms free-object grasp/release from exact MuJoCo contacts and motion; fed every physics step by `SimSensorBridge`. Resolves the manifest's gripper-role joints to one attach link + touch links at construction; raises `ROSConfigError` for none, several parent links, or too few contact branches for the end effector's `kind`.
  - `update(data, *, stamp_ns) -> list[AttachedCollisionObject] | None` (L1615) — Returns a new complete attachment set only when the attachment changes (attach, detach, or a place-witness arm/disarm); `None` otherwise, so a heartbeat never re-arms an exemption the kernel had killed.
  - `set_place_declaration(declaration: PlaceDeclaration | None) -> None` (L1790) — Installs, replaces, or retracts the active place-phase declaration; resolves `target_id` to a MuJoCo body at install time, so an unresolvable target is refused (`ROSConfigError`) rather than silently no-op'd per tick.
  - `place_declaration(data, *, stamp_ns) -> PlaceDeclaration | None` (L1853) — The live declaration, with its region and geometry posed into the robot base frame; overwrites an incoming region/geometry it can measure and drops what it cannot, so a value a dispatcher relayed can never reach the kernel wearing this producer's authority.

### `python/hal/src/openral_hal/_vision_attachment_evidence.py`

_Real-hardware attachment evidence from a segmenter mask plus wrist depth — the counterpart of `_sim_attachment_evidence.py`, emitting the identical `AttachedCollisionObject` contract so the safety kernel cannot tell the two producers apart. Every fail-closed gate is geometric; the segmenter's own confidence is recorded but never thresholded. Scoped to free objects only — vision gives geometry, not kinematic class._

- module const `_MIN_CLOUD_POINTS = 24` (L97) — minimum points a cloud needs before a PCA orientation means anything; below this `clustered_obb_primitives` stays axis-aligned.
- module const `_BOX_EPSILON_M = 1e-4` (L101) — padding added to every fitted box half-extent, matching the sim producer's own epsilon so neither producer emits a zero-thickness slab.
- module const `_VISION_CONFIDENCE = 0.7` (L107) — confidence stamped on an accepted vision attachment (below the sim producer's 1.0 since this one infers geometry from one partially-occluded viewpoint). Calibration point, not measured.
- module const `_FALLBACK_CONFIDENCE = 0.2` (L112) — confidence stamped on the conservative `jaw_span_primitive` fallback box. Calibration point, not measured.
- `class VisionGateConfig` — Frozen dataclass of gate thresholds: `containment_radius_m=0.12`, `max_payload_extents_m=(0.10, 0.25, 0.25)`, `max_payload_volume_m3=0.004`, `min_depth_valid_fraction=0.30`, `min_depth_m=0.02`, `max_depth_m=1.0`, `view_ray_inflation_m=0.02`, `jaw_span_m=0.05`, `max_primitives=16`. Every value is a chosen calibration point, not a measured constant; the extent/jaw-span caps aren't derivable from the manifest today (`EndEffectorSpec` has no jaw-aperture field). (L116)
- `class VisionAttachmentReport` — Frozen dataclass carrying what the gates measured, for the trace: `accepted`, `rejections` (`"no_candidates"` / `"empty_mask"` / `"depth_validity"` / `"payload_extent"` / `"payload_volume"` / `"jaw_containment"` / `"no_valid_depth"`), `depth_valid_fraction`, `point_count`, `extents_m`, `volume_m3`, `centroid_distance_m`, `candidate_index`/`candidate_count`, `mask_score_advisory` (recorded only, never thresholded or used to select). (L185)
- `backproject_masked_depth(mask, depth_m, intrinsics, *, min_depth_m, max_depth_m) -> tuple[NDArray, float]` — Back-projects a mask's pixels through a metric depth frame into the camera optical frame (REP-103), returning `(N, 3)` points and the fraction of masked pixels with usable depth. An empty mask returns `((0, 3), 0.0)` — not an error. Raises `ROSConfigError` on shape disagreement or mismatched-resolution intrinsics. (L227)
- `clustered_obb_primitives(points_in_link, *, object_id, max_primitives=16, inflation_m=0.0, inflation_axis_in_link=None) -> tuple[list[AttachedCollisionPrimitive], NDArray, NDArray]` — Reduces a payload cloud to bounded oriented boxes, the same reduction as the sim producer's `_clustered_box_primitives` but split in the cloud's PCA frame so boxes orient to the object. `inflation_axis_in_link` pads each axis by its share of the view ray, compensating for the unobserved far side. Clouds under 24 points stay axis-aligned rather than orienting off noise. (L317)
- `jaw_span_primitive(*, object_id, jaw_span_m) -> AttachedCollisionPrimitive` — The conservative fallback box: a cube of half-extent `jaw_span_m` centred on the TCP, deliberately crude so a rejected mask still leaves geometry in front of the collision checker. (L407)
- `class VisionAttachmentEvidenceProducer(description, *, config=None)` — Event-driven producer (no per-frame `update`, unlike the ticked sim tracker). Resolves `attach_link`/`touch_links` from the manifest's gripper-role joints; raises `ROSConfigError` when there are none or they don't share one parent link. (L444)
  - `on_grasp(*, masks, depth_m, intrinsics, t_link_from_cam, tcp_in_link, object_id, stamp_ns, mask_scores_advisory=()) -> tuple[AttachedCollisionObject, VisionAttachmentReport]` — Fits and gates one payload from the `SegmentInView` reply's candidate masks; among candidates clearing every gate, the largest fitted volume wins (over-approximation is the conservative error), never the model's own score. Always returns an attachment: any rejection degrades to the `jaw_span_primitive` fallback at low confidence, never `None`. An accepted mask is stamped `VISION_SEGMENTATION`; `mass_kg`/`center_of_mass_m`/`inertia_kg_m2` stay `None` (vision gives geometry only). (L589)
  - `on_release() -> list[AttachedCollisionObject]` — Detach: an empty list, a *complete* attachment set meaning "nothing is held", mirroring the sim producer's release path. (L498)
  - **(property)** `attach_link -> str` (L489)
  - **(property)** `config -> VisionGateConfig` (L494)

**Not covered** (see the module docstring): articulated objects, support contact, transparent/thin objects, self-occlusion, mass/CoM/inertia, deformables, multi-object grasps, and the attach trigger itself.

### `python/hal/src/openral_hal/_grasp_trigger.py`

_When to ask perception "what is in the jaws?" — a debounced state machine over the gripper joint's effort channel in the typed `JointState` the HAL already reads every tick. Pure: no ROS, no numpy, no clock. `_vision_attachment_evidence.py` is *told* a grasp happened; this is what tells it._

- `class GraspEvent(str, Enum)` (L73) — `ATTACH` (empty → loaded: segment, gate, attach), `REGRASP` (still loaded but the jaw moved materially, so the fitted geometry is stale: segment again), `DETACH` (loaded → empty: publish an empty attachment set; no segmentation).
- `class GraspTriggerConfig` (L94) — Frozen dataclass: `attach_effort_fraction=0.30`, `release_effort_fraction=0.10`, `consecutive_ticks=3`, `regrasp_position_delta=0.15`. Every value is a chosen calibration point, fractions of the gripper joint's manifest `effort_limit`. The release/attach gap is a hysteresis band against chatter; the debounce stops a jaw-acceleration spike from attaching a payload never grasped.
- module const `_MIN_HEALTHY_SPAN_FRACTION` (L161) — the 5% of `effort_limit` floor `assess_effort_readback` requires a trace's span to clear before scoring it `usable`.
- `gripper_joint(description) -> JointSpec` (L164) — The manifest's single `role: "gripper"` joint. Raises `ROSConfigError` for zero or several — guessing which of several is "the" gripper is not something to bury in a safety-adjacent trigger.
- `class EffortReadbackHealth` (L130) — Frozen dataclass verdict: `samples`, `present_samples`, `nonzero_samples`, `distinct_values`, `span`, `usable`, `reason`.
- `assess_effort_readback(states, *, description) -> EffortReadbackHealth` (L188) — Judges a recorded open → close-on-object → hold → open effort trace as a usable grasp signal: an absent channel, an identically-zero channel, or a span under 5% of `effort_limit` all score `usable=False` with a named reason. A `False` verdict means `GripperEffortTrigger` cannot be the attach signal on that robot and needs a different one (tactile, current sense, explicit skill-level attach). Raises `ROSConfigError` when the gripper joint declares no positive `effort_limit`.
- `class GripperEffortTrigger(description, *, config=None)` (L273) — Fed one `JointState` per HAL read tick. Raises `ROSConfigError` on a missing/multiple gripper joint, a non-positive `effort_limit`, an inverted hysteresis band, or `consecutive_ticks < 1`.
  - `update(state) -> GraspEvent | None` (L358) — Folds one snapshot in; returns the event on the tick a transition is confirmed (N agreeing samples), else `None`. Effort is compared by magnitude, so a driver signing the closing load negative still attaches. A tick with no gripper effort value increments `missing_effort_ticks` and resets the streak, rather than being silently ignored.
  - **(property)** `joint_name -> str` (L344) — the resolved gripper joint's manifest name.
  - **(property)** `attached -> bool` (L349) — current latched grasp state.
  - **(property)** `thresholds_n -> tuple[float, float]` (L354) — the absolute `(attach, release)` efforts actually in force.
  - **(property)** `missing_effort_ticks -> int` — count of ticks whose gripper carried no effort value.

### `python/hal/src/openral_hal/vision_attachment_bridge.py`

_The ROS wiring that turns `_grasp_trigger` events into `openral_msgs/srv/SegmentInView` calls, feeds the replies to `VisionAttachmentEvidenceProducer`, and publishes the resulting `AttachmentState` the safety kernel already consumes. The real-hardware, opt-in sibling of `SimSensorBridge`'s attachment leg (the two must not both drive `/openral/attachment_state`) — torch-free by construction, since everything model-shaped lives behind the service in `openral_perception_ros.segmenter_node`._

**The deferred-ack barrier.** The HAL holds a grouped tick's `action_applied` until attached-payload perception settles; segmentation runs inside that same wait. The wait is bounded (`deadline_s`), never skipped (every failure path ends in the conservative jaw-span box at `GRIPPER_FORCE`), and visible (every fallback logs its typed reason).

- module const `DEFAULT_SEGMENT_SERVICE` (L75) (`"/openral/perception/segment_in_view"`) — shared with `segmenter_node.DEFAULT_SEGMENT_SERVICE`; a unit test asserts the two halves agree.
- `class VisionAttachmentConfig` (L87) — Frozen dataclass: `camera=""` (empty picks the manifest's first camera with intrinsics), `depth_topic=""` (empty → `/openral/cameras/<camera>/depth`), `service_name=DEFAULT_SEGMENT_SERVICE`, `deadline_s=0.25`, `tcp_frame=""` (empty → the gripper joint's `child_link`, an approximation — no schema field carries a calibrated TCP today), `jaw_tip_frames=()`, `object_id="grasped_payload"`. `deadline_s` is sized off a warmed GPU segmenter call plus round trip; a CPU-only host must raise it deliberately or every grasp falls back.
- `class SegmentOutcome` (L128) — Frozen `(use_masks: bool, reason: str)`.
- `resolve_segment_outcome(*, timed_out, ok, failure_reason, mask_count) -> SegmentOutcome` (L142) — Pure decision for one round trip: a deadline miss wins as `ROSDeadlineMissed:`; `ok=False` uses the server's typed `failure_reason` (or a `ROSPerceptionStale:` default); `ok=True` with zero masks is also `ROSPerceptionStale:`. No input combination yields an unexplained fallback.
- `decode_mono8_mask(data, *, height, width) -> NDArray[np.bool_]` (L191) — Reader half of `segmenter_node.mono8_bytes_from_mask`; any non-zero pixel is set. Raises `ROSConfigError` on a length mismatch.
- `class VisionAttachmentBridge(node, description, *, on_perception_ready=None, config=None, gate_config=None, trigger_config=None)` (L221) — Owns the depth subscription, tf2 listener, `SegmentInView` client and the `/openral/attachment_state` publisher (RELIABLE/TRANSIENT_LOCAL, matching the sim bridge). Raises `ROSConfigError` at construction when the manifest can't support the producer or the trigger.
  - `setup() -> None` (L273) — Create the ROS entities.
  - `teardown() -> None` (L316) — Destroy the ROS entities. Idempotent and always reopens the barrier, so a mid-flight deactivate can never leave the node deferring an acknowledgement forever.
  - `attachment_action_ack_ready() -> bool` (L339) — The same shape `SimSensorBridge` exposes, so `ManifestHALLifecycleNode`'s deferred-ack path treats both identically.
  - `observe_joint_state(state) -> None` (L349) — Called from `_publish_joint_state` with the same typed snapshot the HAL just read (no second source of truth). `DETACH` publishes an empty attachment set; `ATTACH`/`REGRASP` closes the barrier and dispatches one bounded request.
  - **(property)** `missing_effort_ticks -> int` (L368) — Passthrough of the trigger's driver-health counter.

**Node wiring** (`ManifestHALLifecycleNode`): `vision_attachment_*` params (`enabled` defaults to False); built in `_setup_vision_attachment` on activate, torn down on deactivate/cleanup. `_attachment_barrier_holders()` / `_attachment_perception_ready()` require every present holder (sim bridge + vision bridge) to clear before a tick is acknowledged.
