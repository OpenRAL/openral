# Layer 1 — Hardware Abstraction (HAL)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/hal/src/openral_hal/protocol.py`
_Normative HAL protocol plus explicit optional lifecycle extensions._

- `class HAL(Protocol)` — Structural protocol every HAL adapter must satisfy.
  - attr `description: RobotDescription`
  - `connect() -> None` — Open connection to robot/sim.
  - `disconnect() -> None` — Close connection (idempotent).
  - `read_state() -> JointState` — Latest joint state snapshot (hot path).
  - `send_action(action: Action) -> None` — Forward action chunk to controller (hot path).
  - `estop() -> None` — Trigger emergency stop, always raises `ROSEStopRequested`.
- `class LifecycleEStopHAL(Protocol)` — Opt-in propagation of the generic
  lifecycle e-stop to downstream hardware or owned processes.
- `class ResettableLifecycleEStopHAL(LifecycleEStopHAL, Protocol)` — Opt-in
  in-process recovery contract.
- `class EStopRecovery(StrEnum)` — Declares whether recovery is resettable or
  requires a full lifecycle restart.
- `class HALHealthProvider(Protocol)` / `class HALHealthReport` — Cached,
  I/O-free diagnostics consumed by the generic lifecycle heartbeat.

### `python/hal/src/openral_hal/_mujoco_arm.py`
_Internal MuJoCo-backed HAL implementation shared by UR / Franka / SO-100 / G1 / H1 / Rizon-4 / OpenArm / ALOHA adapters. Reads its wiring from `RobotDescription.sim`._

- `class _MujocoArmInitKwargs(TypedDict)` — Typed shape of the kwargs accepted by `MujocoArmHAL.__init__`. (L61) Lets `MujocoArmHAL._sim_kwargs_for` return a value that unpacks cleanly into the constructor under `mypy --strict` without the `# type: ignore[arg-type]` hatch every thin subclass used to need. Fields: `mjcf_path, joint_qpos_addr, joint_qvel_addr, actuator_index, grippers, keyframe_index, seed_ctrl_from_qpos, settle_steps, gravity_enabled, staleness_limit_s`.
- `_resolve_mjcf_path(desc: RobotDescription) -> str` [private] — Resolve `desc.assets.mjcf` to an absolute MJCF path via `openral_core.assets.resolve_asset`; raises `ROSConfigError` when the ref is unset or unresolvable. Replaced the former public `resolve_mjcf_uri` / `SimDescription.mjcf_uri`. (L81)
- `build_hal(description, *, mode: Literal["sim","real"], transport=None, sim_env_yaml=None) -> HAL` — Single seam for constructing a robot's simulation or real-hardware HAL from its manifest (`resolver.py`, L50). `mode="sim"` + `sim_env_yaml` set → calls `build_sim_env_from_yaml` and returns a `SimAttachedHAL` wrapping the scene's `SimRollout`; bypasses the bare-twin / `hal.sim` class entirely. `mode="sim"` without `sim_env_yaml` builds `description.hal.sim` or derives `MujocoArmHAL.from_description` when it is null + a `sim:` block exists. `mode="real"` imports `description.hal.real` and threads `transport` kwargs (real HALs take `port` / `robot_ip` / `fci_ip` and embed their own description). Both modes merge `description.hal.parameters.defaults` **underneath** the explicit `transport` so the manifest carries a robot's construction kwargs; unaccepted keys are dropped. `sim_env_yaml` + `mode="real"` → `ROSConfigError`. Missing HAL for the mode → `ROSCapabilityMismatch`; malformed/unresolvable entry → `ROSConfigError`. Routed by `deploy sim` (sim) and `deploy run` (real).
  - `_import_object(path: str) -> object` [private] — Resolve a `"module.path:Attribute"` import string; raises `ROSConfigError` on malformed/unimportable/missing. **Reuse watch:** the canonical entrypoint-string importer for HAL classes — do not hand-roll `importlib` in HAL callers.
- `class MujocoArmHAL` — Generic MuJoCo-backed HAL adapter for position-controlled arms (and, via the `_per_step_update` hook, torque-controlled humanoids like the H1). (L119)
  - `read_images() -> dict[str, NDArray]` — Render the manifest's RGB `SensorSpec`s off the live MJCF, keyed by sensor `name` (issue #191 Phase 3b). Same contract `SimAttachedHAL.read_images` exposes, so `SimSensorBridge` publishes a composed-scene arm's cameras (openarm) through the shared path. Renders the MJCF camera `sim_camera_name or name` **at that sensor's own `intrinsics` resolution** — one `mujoco.Renderer` is cached per distinct `(height, width)`, so e.g. a 256×256 wrist camera alongside a 640×480 overhead publishes 256×256 (not a shared max); the published frame size always matches the sensor's camera model. A missing camera / render error is skipped with a one-shot warning (never raises). Renderers are created lazily per resolution so each EGL context binds on the caller (executor) thread. Returns `{}` when disconnected / no RGB sensors / after a renderer failure.
  - `__init__(description, *, mjcf_path, joint_qpos_addr, actuator_index, joint_qvel_addr=None, grippers=(), keyframe_index=None, seed_ctrl_from_qpos=False, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Init only; MJCF is not loaded until `connect()`. `joint_qvel_addr` defaults to `joint_qpos_addr` (correct for arms without a floating base) and is passed explicitly by humanoid HALs like `G1MujocoHAL` / `H1MujocoHAL` where the free joint shifts the qvel indices by 1. `grippers` is a sequence of `SimGripperDescription` entries; single-arm robots ship one (or none), bimanual robots (Aloha, OpenArm) ship two. (L174)
  - `_per_step_update(targets) -> None` — Hook invoked before every `mj_step` inside the settle loop. Default no-op; subclasses driving torque-mode actuators (`H1MujocoHAL`) override to recompute the actuator torque each step from the current `qpos` / `qvel`.
  - `connect() -> None` — Load MJCF, prepare `MjData` buffer. Before compiling, runs the generic camera rig (`_camera_rig.rig_cameras_into_mjcf`): if the MJCF lacks a manifest RGB camera that declares a `sim_placement`, it splices the camera (+ visual-only floor + fill light) into a sibling `<name>_camrig.xml` and loads that — so a bare-arm deploy twin (so100/so101) renders its declared cameras without a scene composer. Idempotent: a scene-attached / composed MJCF that already has the cameras loads unchanged.
  - `disconnect() -> None` — Release the MuJoCo model (idempotent).
  - `read_state() -> JointState` — Joint state in description-joint order. Reads live in-process `MjData` (always current), so it **never latches `ROSPerceptionStale`**: a gap > `staleness_limit_s` since the last service means the single-threaded executor was starved (e.g. a slow camera render), not bad data — it emits a one-shot `hal.read_state.starved` WARNING and returns the live state (re-armed on the next healthy read). The prior behaviour raised *before* refreshing the clock, so one transient stall bricked the HAL permanently (the deploy-sim "Joint state is X s old" loop). Async live-feedback staleness is policed by the subscription HALs (`ros_control`/`aloha`), not here.
  - `send_action(action: Action) -> None` — Forward last waypoint to MuJoCo and step. Stamps `_last_action_ns` so the idle stepper yields to a recent command.
  - `sim_time_ns() -> int | None` — Bare-twin MuJoCo elapsed time in ns, read from live `MjData.time`; `None` before connect / after disconnect or e-stop. This is the `/clock` seam for OpenArm / SO-100 / SO-101 deploy-sim graphs, matching `SimAttachedHAL.sim_time_ns()` for scene-attached rollouts.
  - `clock_authority() -> ClockAuthority` — Return `ClockAuthority.simulation("mujoco", timestep_s=model.opt.timestep)` while connected, otherwise `ClockAuthority.host_wall()`.
  - `idle_step(wall_dt_s=None) -> bool` — **Sim-only** HOLD stepper that gives a bare `MujocoArmHAL` the cameras-stay-live treatment, plus joint_state published off the executor via `ProprioSnapshot` + dedicated thread, that the lifecycle node gates on a *callable* `idle_step`. Leaves `ctrl` untouched (it already holds the last commanded / seeded pose). With `wall_dt_s`, advances that wall-time slice (bounded to 200 physics steps); without it, advances one legacy `mj_step`. Bare MuJoCo arms set the internal `_step_while_active` capability so this wall-time integrator continues during active skills: `send_action()` advances only one physics tick, and yielding the stepper previously collapsed `/clock` and top/wrist camera publication to ~0.25 Hz during rollout. Returns `False` after disconnect/e-stop, so it can never autonomously drive an e-stopped robot.
  - **(property)** `last_action_ns -> int` — `time.monotonic_ns()` of the last `send_action` (`0` if never actuated → idle-stepping starts immediately). The `SimSensorBridge` reads it (`should_idle_step`) to yield the idle stepper to a recently-commanded skill. Mirrors `SimAttachedHAL.last_action_ns`.
  - `reset_to_pose(pose: list[float]) -> None` — Snap `qpos` to a manifest `starting_pose` and re-seed `ctrl` (instantaneous teleport; best-effort). Gripper entries use the HAL's public units: normalized values are mapped through `SimGripperDescription.ctrl_range`, so SO-101 `0.0195` reads back as `0.0195` rather than being mistaken for raw jaw radians. The collision-aware alternative is **not** a HAL method — the runner dispatches the `rskill-moveit-multi-joints` rSkill to plan a collision-free MoveGroup motion to `starting_pose` (see `05-inference-runner` / `08-cli`). (L597)
  - `estop() -> None` — Zero `ctrl` and raise `ROSEStopRequested`.
  - **(classmethod)** `from_description(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> MujocoArmHAL` — Manifest-driven constructor. Reads `description.sim` and builds the HAL with the right MJCF path, qpos/qvel/actuator maps and gripper config. Removes the need for per-robot Python subclasses. (L952)
  - **(staticmethod)** `_sim_kwargs_for(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> _MujocoArmInitKwargs` — Translate `description.sim` into the `__init__` kwarg dict.  Default 1:1 joint→qpos/actuator mapping is derived from `description.joints`, offset by 7 (qpos) / 6 (qvel) when `sim.floating_base=True`.  Used by both `from_description`, `_init_from_description`, and any caller that wants to post-process the kwargs. (L876)
  - **(instance method)** `_init_from_description(description, *, mjcf_path=None, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5) -> None` — Seam every thin per-robot subclass (UR5e/UR10e, Franka, ALOHA, OpenArm, Rizon4, G1, H1, SO-100) uses to drop the boilerplate `super().__init__(DESC, **MujocoArmHAL._sim_kwargs_for(DESC, …))` dance. Subclasses keep their typed `__init__(*, mjcf_path, settle_steps, gravity_enabled, staleness_limit_s)` signature (so IDEs still surface the four user-tunable knobs) and forward straight to here. (L1008)
  - private: `_require_connected`, `_validate_action`, `_last_arm_targets`, `_apply_arm_targets`, `_apply_gripper_target`, `_read_gripper_normalised`, `_effective_actuator_index_for`

### `python/hal/src/openral_hal/_camera_rig.py`
_Generic sim camera rig — splice manifest cameras into a bare-arm MJCF for deploy sim._

- `rig_cameras_into_mjcf(xml: str, sensors: list[SensorSpec]) -> tuple[str, bool]` — For each RGB `SensorSpec` with a `sim_placement` whose camera (`sim_camera_name or name`) is absent from `xml`, splice a `<camera>` (look-at orientation via `openral_core.geometry.look_at_quat_wxyz`, `-z` MuJoCo view axis; FoV from `sim_placement.fovy_deg` or derived from `intrinsics`) into the named `parent_body` (a wrist camera) or `<worldbody>` (a world-fixed overhead), plus minimal staging — a visual-only (`contype=0 conaffinity=0`, no collisions) ground plane and an ambient fill light — when the MJCF declares none. Returns `(xml, changed)`; `changed=False` (input untouched) when no rigging is needed, so a scene-attached / already-composed MJCF passes through and the caller loads the original. Idempotent. Raises `ROSConfigError` when a sensor's `parent_body` is missing or there is no `</worldbody>` for a world camera. Called by `MujocoArmHAL.connect`.

### `python/hal/src/openral_hal/_real_description.py`
_Internal helper to derive a real-hardware ``RobotDescription`` from a sim baseline._

- `make_real_description(base, *, sdk_kind) -> RobotDescription` — `model_copy(update={"sdk_kind": sdk_kind})`; the `hal` entrypoints (`hal.sim` / `hal.real`) are inherited from *base*. (L48)

### `python/hal/src/openral_hal/franka_panda.py`
_HAL adapter for the Franka Emika Panda 7-DoF arm (sim, MuJoCo)._

- `class FrankaPandaHAL(MujocoArmHAL)` — Franka Panda HAL (MuJoCo-backed). Thin manifest-driven wrapper around `MujocoArmHAL`; `__init__` forwards to `self._init_from_description(FRANKA_PANDA_DESCRIPTION, …)`. (L266)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L295)
- `_panda_joint_specs() -> list[JointSpec]` (L122)
- const `FRANKA_PANDA_DESCRIPTION = RobotDescription(...)` (L176) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.franka_panda:FrankaPandaHAL"` + `hal.real="openral_hal.franka_panda_real:FrankaPandaRealHAL"`. All MuJoCo wiring (MJCF URI, joint→qpos/actuator maps, gripper config) lives in `FRANKA_PANDA_DESCRIPTION.sim`. The real-HW companion `FRANKA_PANDA_REAL_DESCRIPTION` lives in `franka_panda_real.py`.

### `python/hal/src/openral_hal/franka_panda_real.py`
_Real-hardware HAL adapter for the Franka Emika Panda over the FCI (issue #56)._

- `class FrankaPandaRealHAL` — Production adapter for a physical Panda over `franka_ros2` / FCI. Wraps `RosControlHAL` via composition. (L90)
  - `__init__(*, fci_ip='172.16.0.2', controller_name='franka_arm_controller', joint_state_topic='/joint_states', command_topic=None, error_recovery_topic='/error_recovery/goal', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L144)
  - `description -> RobotDescription` [@property] — Returns `FRANKA_PANDA_REAL_DESCRIPTION`. (L180)
  - `controller_name -> str` [@property] (L185)
  - `fci_ip -> str` [@property] (L190)
  - `connect() -> None` (L196)
  - `disconnect() -> None` (L214)
  - `read_state() -> JointState` (L220)
  - `send_action(action) -> None` (L230)
  - `estop() -> None` — Publishes to `/error_recovery/goal` then raises `ROSEStopRequested`. (L246)
- const `FRANKA_PANDA_REAL_DESCRIPTION = make_real_description(FRANKA_PANDA_DESCRIPTION, sdk_kind="closed_with_api")` (L84) — inherits the shared `hal`; what `robots/franka_panda/robot.yaml` mirrors.

### `python/hal/src/openral_hal/sawyer_real.py`
_Real-hardware HAL adapter for the Rethink Sawyer 7-DoF arm (issue #57)._

- `class SawyerRealHAL` — Production adapter for a physical Sawyer over `intera_sdk` / `sawyer_robot`. (L219)
  - `__init__(*, hostname='sawyer.local', controller_name='sawyer_arm_controller', joint_state_topic='/robot/joint_states', command_topic=None, estop_topic='/robot/set_super_stop', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L266)
  - `description -> RobotDescription` [@property] — Mirrors `SAWYER_DESCRIPTION`. (L302)
  - `hostname -> str` [@property] (L307)
  - `controller_name -> str` [@property] (L312)
  - `connect() -> None` (L316)
  - `disconnect() -> None` (L330)
  - `read_state() -> JointState` (L334)
  - `send_action(action) -> None` (L344)
  - `estop() -> None` (L354)
- `_sawyer_joint_specs() -> list[JointSpec]` (L111)
- const `SAWYER_DESCRIPTION = RobotDescription(...)` (L154) — sim baseline; `sdk_kind="open"`, `hal.sim=None` (no MuJoCo HAL adapter today) + `hal.real="openral_hal.sawyer_real:SawyerRealHAL"`.
- const `SAWYER_REAL_DESCRIPTION = make_real_description(SAWYER_DESCRIPTION, sdk_kind="closed_with_api")` (L194) — inherits the shared `hal`; what `robots/sawyer/robot.yaml` mirrors.

### `python/hal/src/openral_hal/panda_mobile.py`
_In-process digital-twin HAL for the `panda_mobile` embodiment (Franka 7-DoF arm on a holonomic 3-DoF base). Built by `build_hal` for the manifest-driven `ManifestHALLifecycleNode` (issue #191 Phase 3) and by tests; ROS node entrypoint in `packages/openral_hal_panda_mobile/`._

- const `PANDA_MOBILE_BASE_JOINT_NAMES: list[str]` — Base joints `[base_x, base_y, base_yaw]`, derived from `PANDA_MOBILE_DESCRIPTION.base_joints` (not hardcoded). (L100)
- const `PANDA_MOBILE_JOINT_NAMES: list[str]` — Full 11-DoF order: base (3) + arm (7, role-derived) + gripper (1, role-derived) — all from the description. (L116)
- const `PANDA_MOBILE_DESCRIPTION: RobotDescription` — Canonical RobotDescription, loaded from `robots/panda_mobile/robot.yaml` at module import. Single source of truth for joint metadata + `sim_joint_name` overrides; the arm/base/gripper name constants above derive from it via `JointSpec.role`. (L93)
- `class PandaMobileHAL` — In-process digital-twin HAL. Routes `BODY_TWIST` → planar Euler integration of (vx, vy, wz); routes `JOINT_POSITION` → 7-vec arm targets or 11-vec base+arm+gripper targets. (L132)
- _(removed: the `base_sim_joint_names` re-export wrapper — callers now import `openral_core.extract_base_sim_joint_names` directly.)_

### `python/hal/src/openral_hal/depth_cloud.py`
_Reusable, robot-agnostic depth-camera → `sensor_msgs/PointCloud2` plumbing for deploy-sim HAL nodes (octomap_server source → kernel world-collision check). Pure SensorSpec adapters + the ROS msg builder; the ray-cast synth lives in `openral_sim.backends.depth_camera`._
- `is_depth_sensor(spec) -> bool` — True when `spec.modality in ("depth", "point_cloud")` **and** it carries pinhole `intrinsics` (required to back-project). (L41)
- `mjcf_camera_name(spec) -> str` — Resolves the backing MJCF `<camera>` name: `spec.metadata["mjcf_camera"]` if set (the sim camera name can differ from the ROS sensor name), else `spec.name`. (L50)
- `robot_self_body_ids(model, sim_joint_names) -> frozenset[int]` — Every MJCF body whose name shares a first-`_`-token prefix with one of the robot's `sim_joint_name`s (e.g. `mobilebase0` / `robot0` / `gripper0`). Passed as `synthesize_depth_pointcloud(exclude_body_ids=…)` so the depth cloud is self-filtered (the robot is not voxelised into its own world map). (L105)
- `depth_synth_kwargs(spec, *, max_range_default, render_size=None) -> dict` — Maps a depth `SensorSpec` to `synthesize_depth_pointcloud` kwargs (width/height/fx/fy/cx/cy + `min_range_m`/`max_range_m` from `range_min_m`/`range_max_m`, falling back to `max_range_default`). When `render_size=(width, height)` is given (the scene's `observation_width/height`), the intrinsics are first rescaled via `openral_core.scale_intrinsics_to` so the ray-cast grid matches the render resolution. (L63)
- `resolve_base_body_name(model, *, description=None) -> str | None` — Resolve the MJCF body backing the robot's `base_frame`: when a `RobotDescription` is given, the first base joint's prefix + `_base` (`mobilebase0_base`); then the bare candidates `mobilebase0_base` / `base` / `robot0_base` / `base_link` — `mobilebase0_base` tried before `robot0_base` because in composed robosuite/RoboCasa scenes `robot0_base` is a placeholder mount at a fixed offset; `None` if none exist. Backs both the depth/TF base resolution (`SimSensorBridge._resolve_depth_base_body`) and the viewer free-camera fallback. (L128)
- `preferred_viewer_camera_id(model, *, prefer=("agentview","top","frontview","front")) -> int` — Pick the named MJCF camera whose vantage the viewer should open from: the first camera whose name contains a `prefer` substring (a 3rd-person workspace view — `robot0_agentview_left`, `top`, `agentview`), else the first declared camera (e.g. a wrist/eye-in-hand cam), else `-1` when the model has no cameras. Scene cameras are authored to frame the action, sidestepping the free orbit's occlusion in cluttered scenes (a base-centred orbit in a RoboCasa kitchen stares at a wall). Consumed by `initial_viewer_camera`. (L174)
- `initial_viewer_camera(*, model, data, description=None) -> tuple[tuple[float,float,float], float, float, float]` — Opening **free-camera** pose `(lookat, distance, azimuth_deg, elevation_deg)` for the viewer. The viewer always uses `mjCAMERA_FREE` so the user keeps full mouse control (drag-orbit, scroll-zoom) — this only sets the *initial* view; a `mjCAMERA_FIXED` lock would freeze those controls. When `preferred_viewer_camera_id` finds an authored camera, the eye is placed at that camera's `data.cam_xpos` with the orbit pivot on the robot base (`resolve_base_body_name`, else `model.stat.center`), so the opening view matches the authored vantage yet orbits around the robot; else delegates to `base_aligned_free_camera`. Reproduces the eye exactly via MuJoCo's `eye = lookat − distance·f`, `f = (cos el cos az, cos el sin az, sin el)`. (L341)
- `apply_robosuite_visual_geomgroups(opt, model) -> bool` — For a robosuite/RoboCasa model, set `opt.geomgroup` to hide collision shells (group 0 — RoboCasa's red kitchen / green robot capsules) and show the textured visual geoms (group 1), so `mujoco.viewer` renders textures instead of a red collision box. Gated on a robosuite signature (a `robot0_`/`gripper0_`/`mobilebase0_` body **or** an `agentview`/`frontview` camera) — **not** geom counts, since dm_control/gym scenes (gym-aloha) put visuals in group 0; returns `True` when it acted, `False` (no-op) otherwise. Used by the eval `sim run --view` viewer. (L217)
- `base_aligned_free_camera(*, model, data, base_body_name=None, azimuth_offset_deg=135.0, elevation_deg=-25.0, distance_scale=2.0, max_distance_m=3.5) -> tuple[tuple[float,float,float], float, float, float]` — **Fallback** free-camera framing `(lookat_xyz, distance, azimuth_deg, elevation_deg)` for camera-less models (single-robot twins): centres on the robot base and offsets the azimuth by the base frame's world yaw so the view aligns to the base's own axes (MuJoCo's world frame is immutable, so the viewer cannot be re-rooted onto `base_link`). `distance` is `distance_scale × model.stat.extent` capped at `max_distance_m` (a composed scene's whole-model extent would otherwise push the camera tens of metres out). Falls back to `model.stat.center` with no yaw when `base_body_name` is `None`/absent. Shared with the `openral sim run --view` eval path. (L252)
- `camera_optical_tf_to_base(*, model, data, camera_name, base_body_name) -> tuple[tuple[float,float,float], tuple[float,float,float,float]]` — Live `(translation_xyz, quat_xyzw)` of the camera optical frame (REP-103) expressed in the base body, from `data.cam_xpos`/`cam_xmat` vs the base body pose, so a node broadcasts `base_frame → <camera>_optical_frame`. Raises `ROSConfigError` if camera/body absent. (L399)
- `pointcloud2_from_points_xyz(points, *, frame_id, stamp=None) -> PointCloud2` — Packs an `(N, 3)` float32 array into an unordered (`height=1`) XYZ-float32 `sensor_msgs/PointCloud2` — the layout octomap_server's `cloud_in` expects (`sensor_msgs` imported lazily). (L451)
- `depth_image_from_grid(depth, *, frame_id, stamp=None) -> Image` — Packs an `(H, W)` float32 metric-depth raster (from `synthesize_depth_image`) into a `32FC1 sensor_msgs/Image` (`step=4·W`, row-major; `0.0` = no measurement) for nvblox's projective depth integrator (`sensor_msgs` imported lazily). (L492)
- `camera_info_from_intrinsics(*, width, height, fx, fy, cx, cy, frame_id, stamp=None) -> CameraInfo` — Builds a pinhole `sensor_msgs/CameraInfo` for a synthesised depth image — `K=[fx,0,cx;0,fy,cy;0,0,1]`, identity `R`, `P` mirroring `K` (no baseline), zero `plumb_bob` distortion (MuJoCo ray-cast has none). Callers pass the **stride-scaled** intrinsics so the model matches the rasterised image. (L529)

### `python/hal/src/openral_hal/aloha.py`
_HAL adapter for the Trossen ALOHA bimanual setup (issue #58) + the MuJoCo digital twin._

- `class AlohaHAL(HALBase)` — Real-hardware adapter for the 14-DoF ALOHA over the Interbotix XS SDK. (L332)
  - `__init__(*, left_arm_controller='left_arm/arm_controller', right_arm_controller='right_arm/arm_controller', left_gripper_controller='left_arm/gripper_controller', right_gripper_controller='right_arm/gripper_controller', joint_state_topic='/joint_states', estop_topic='/aloha/estop', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L380)
  - `connect() -> None` (L411)
  - `disconnect() -> None` (L428)
  - `read_state() -> JointState` (L435)
  - `send_action(action) -> None` — Splits the 14-D action 4-ways across per-arm + per-gripper controllers. (L467)
  - `estop() -> None` (L531)
  - private: `_require_connected`
- `class AlohaMujocoHAL(MujocoArmHAL)` — MuJoCo digital twin for the 14-DoF bimanual ALOHA; thin manifest-driven wrapper around `MujocoArmHAL` (bimanual amendment). All wiring lives in `ALOHA_DESCRIPTION.sim`: `gym_aloha:bimanual_viperx_transfer_cube` URI, explicit `joint_qpos_addr` / `actuator_index` (left arm 0-5, left gripper 6, right arm 8-13, right gripper 14 — skipping the negative-finger slots), two `PASSTHROUGH` grippers with `mirror_actuator_index` (positive finger + negative finger), `keyframe_index: 0` (seeds the fingers inside `ctrlrange=[0.021, 0.057]`). (L568)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(ALOHA_DESCRIPTION, …)`. (L603)
- `_aloha_joint_specs() -> list[JointSpec]` (L147)
- `_default_publish(topic, msg) -> None` (L556)
- const `ALOHA_DESCRIPTION = RobotDescription(...)` (L191) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.aloha:AlohaMujocoHAL"` + `hal.real="openral_hal.aloha:AlohaHAL"`.
- const `ALOHA_REAL_DESCRIPTION = make_real_description(ALOHA_DESCRIPTION, sdk_kind="closed_with_api")` (L303) — inherits the shared `hal`; what `robots/aloha_bimanual/robot.yaml` mirrors.

### `python/hal/src/openral_hal/ur.py`
_HAL adapters for the Universal Robots UR5e and UR10e arms (sim, MuJoCo)._

- `class UR5eHAL(MujocoArmHAL)` — UR5e HAL (MuJoCo-backed). Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(UR5e_DESCRIPTION, …)`. (L302)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L326)
- `class UR10eHAL(MujocoArmHAL)` — UR10e HAL (MuJoCo-backed). Same shape as `UR5eHAL`. (L344)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L356)
- `ur5e_with_sensors(catalog_ids=None) -> RobotDescription` (L246)
- `ur10e_with_sensors(catalog_ids=None) -> RobotDescription` (L272)
- `_ur_joint_specs(velocity_limits, effort_limits) -> list[JointSpec]` (L119)
- const `UR5e_DESCRIPTION = RobotDescription(...)` (L157) — sim manifest; all MuJoCo wiring lives in `UR5e_DESCRIPTION.sim`.
- const `UR10e_DESCRIPTION = RobotDescription(...)` (L201) — sim manifest; all MuJoCo wiring lives in `UR10e_DESCRIPTION.sim`.

### `python/hal/src/openral_hal/ur_real.py`
_Real-hardware HAL adapters for UR5e / UR10e via `ros2_control` + `ur_robot_driver` (URCap / RTDE)._

- `class UR5eRealHAL(_URRealHAL)` — Real UR5e via `ur_robot_driver`. (L147)
- `class UR10eRealHAL(_URRealHAL)` — Real UR10e via `ur_robot_driver`. (L199)
- `class _URRealHAL(RosControlHAL)` — Shared real-HW base (controller / topic defaults + `deadman_topic`). (L88)
- const `UR5e_REAL_DESCRIPTION = make_real_description(UR5e_DESCRIPTION, sdk_kind="closed")` (L77) — inherits the shared `hal`; what `robots/ur5e/robot.yaml` mirrors.
- const `UR10e_REAL_DESCRIPTION = make_real_description(UR10e_DESCRIPTION, sdk_kind="closed")` (L82) — inherits the shared `hal`; what `robots/ur10e/robot.yaml` mirrors.

### `python/hal/src/openral_hal/so100_follower.py`
_SO100FollowerHAL — wraps lerobot's SO-100 follower arm USB driver._

- `class SO100FollowerHAL` — HAL adapter wrapping lerobot's SO-100 follower. (L281)
  - `__init__(port='/dev/ttyUSB0', *, calibrate_on_connect=False, max_relative_target=None, staleness_limit_s=0.5, robot=None)` (L323)
  - `connect() -> None` — Open USB serial connection. (L365)
  - `disconnect() -> None` — Close USB, disable motor torque (idempotent). (L475)
  - `read_state() -> JointState` — Joint state in radians. (L488)
  - `send_action(action: Action) -> None` — Forward one step to the SO-100 motor bus. (L516)
  - `reset_to_pose(pose: list[float]) -> None` — Slow linear ramp current → target (speed-capped `_RESET_MAX_RAD_S`, duration clamped `[_RESET_MIN_S, _RESET_MAX_S]`, `_RESET_STEP_HZ` waypoints) — the real-arm counterpart of the sim arms' qpos snap; makes the HAL lifecycle node auto-open `/openral/<robot>/reset_to_pose`, so real `deploy run` starts VLAs from their manifest `starting_pose`. (L539)
  - `estop() -> None` — Disconnect motors then raise. (L602)
  - `_require_connected(operation: str)`, `_obs_to_positions(obs)` [@staticmethod], `_action_to_lerobot(action)`
  - `_joint_values_to_lerobot(step) -> dict[str, float]` (module-level) — THE single manifest-order → lerobot `{"<joint>.pos": …}` unit conversion (rad→deg arm joints, `[0,1]`→`[0,100]` gripper); both `_action_to_lerobot` and the `reset_to_pose` ramp route through it so a calibration/range change can never apply to one actuation path and not the other. (L253)
- `_deg_to_rad(deg) -> float` (L248)
- `_rad_to_deg(rad) -> float` (L273)
- const `SO100_DESCRIPTION = RobotDescription(...)` (L103)

### `python/hal/src/openral_hal/galaxea_a1.py`
_Real-only Galaxea A1 HAL. OpenRAL stays ROS 2 / Python 3.12; the operator's
official ROS 1 Noetic SDK runs out of process behind a literal IPv4-loopback
JSON-lines sidecar. No vendor source, binary, or message package is distributed._

- `class GalaxeaA1HAL(HALBase)` — six-axis joint-position + normalized
  gripper adapter. `read_state` and `send_action` use a cached snapshot/latest
  target so network I/O stays off the HAL hot path. Commands fail closed on
  stale state/status, unaccepted motor bits, non-finite values, initial target
  misalignment, or an excessive feedback-relative target step. Command limits
  remain exact; a separately tracked 0.01 rad feedback-only endpoint tolerance
  absorbs encoder zero/quantization at a nominal URDF boundary. `estop` asks
  the sidecar to stop its owned ROS 1 stack and always raises
  `ROSEStopRequested`.
- const `GALAXEA_A1_DESCRIPTION` — real-only `RobotDescription`, mirrored by
  `robots/galaxea_a1/robot.yaml`; official A1 URDF joint names/limits, explicit
  sidecar deadlines, motor masks, 0..104 mm normalized gripper mapping, and
  the calibrated D455 front / D405 wrist RGB observation contracts. The six
  joint origins, orientations, and axes are transcribed from the official A1
  URDF; collision primitives remain absent until their redistribution and
  lowering provenance is cleared.
- `tools/galaxea_a1_ros1_sidecar.py` — Python-3.8-compatible ROS 1 process that
  owns `roscore`, `signal_arm/single_arm_node.launch`, and the official
  `mobiman/jointTracker_demo_node` binary. The tracker publishes to
  `/openral/arm_joint_command_staged`; a sidecar-owned relay is the sole
  publisher to `/arm_joint_command_host`. The relay stays `LOCKED` until the
  first target, then `ARMING` until a fresh, valid tracker command is aligned
  with both that target and measured joint feedback. Only `ACTIVE` forwards
  unchanged tracker commands, and gripper setpoints are gated on the same
  machine — a gripper command while the relay is `LOCKED`/`ARMING` is refused
  fail-closed (the gripper bypasses the tracker's staged-hold interpolation,
  so it must never actuate before alignment). Repeated identical joint and gripper setpoints
  refresh the command lease without restarting the official tracker. A command
  lease, alignment timeout, malformed command, stale feedback, motor fault,
  client disconnect, or e-stop stops the complete owned process group.
- `tools/run_galaxea_a1_sidecar.sh` — Docker launcher. Requires an explicit
  operator-provided image and SDK path; mounts both read-only and claims only the
  selected serial device. The SDK remains read-only; the official tracker's
  generated CppAD files go to
  `$XDG_CACHE_HOME/openral/galaxea-a1/x1_robot` (or the equivalent path below
  `~/.cache`). `--check-only` verifies the local image, required SDK files,
  cache parent, serial ownership, loopback port, container name, and process
  lock without opening the serial device or starting a container.

#### Galaxea A1 hardware bring-up

The first session is observation-only until the HAL graph is healthy. Ensure no
other process/container owns the serial device, the arm workspace is clear, and
the physical e-stop is reachable.

```bash
# One-time: build OpenRAL's vendor-free Noetic runtime image. The official SDK
# is mounted at run time and is never copied into the image.
docker build \
  -t openral/galaxea-a1-sidecar:noetic \
  docker/galaxea_a1_sidecar

# One-time: build OpenRAL's standard public x86 deploy image (Jazzy/Python 3.12).
just docker-build-x86

# Read-only gate — checks the image, SDK, serial ownership, port, and lock.
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1 \
  --check-only

# Terminal 1 — isolated ROS 1 bridge network; only TCP 46011 reaches loopback.
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1

# Terminal 2 — OpenRAL's standard real-hardware path. The OpenRAL container uses
# host networking only for ROS 2 DDS and the sidecar's loopback TCP port; it owns
# no Galaxea serial device and cannot see the ROS 1 master inside the sidecar.
docker run --rm --name openral-galaxea-a1 --network host \
  --volume "$(pwd)/robots:/workspace/robots:ro" \
  --volume "$(pwd)/scenes:/workspace/scenes:ro" \
  --volume "$(pwd)/tests:/workspace/tests:ro" \
  openral:x86 \
  --config scenes/deploy/galaxea_a1_bench.yaml

# Terminal 3 — observation gate: six named joints update; diagnostics are clean.
docker exec openral-galaxea-a1 bash -lc \
  'source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && \
   ros2 topic echo /joint_states --once && ros2 topic echo /diagnostics --once'
```

An optional HAL-level HIL gate can run between Terminal 1 and the full deploy.
It opens one sidecar session, validates three fresh finite named-joint samples
plus cached motor health, and ends by verifying that downstream e-stop stops the
owned ROS 1 stack. Restart Terminal 1 afterwards:

```bash
GALAXEA_A1_HIL=1 just hil galaxea_a1
```

Only after that observation-only run passes, opt into a measured-current-pose
hold. Feedback within the tracked 0.01 rad endpoint tolerance is projected to
the exact command limit; any larger projection fails before publication. The
test also waits for the sidecar relay to report `ACTIVE`, proving the official
tracker has converged from its compiled `task.info` initial pose before any
host motor command is forwarded:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_HOLD=1 just hil galaxea_a1
```

After the hold passes, a separate lab opt-in moves `arm_joint1` by +0.01 rad,
requires it to settle within 0.008 rad (covering the measured 0.007 rad
small-command residual), continuously bounds all six joint excursions, returns
to the measured start, and then performs the same downstream e-stop:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_NUDGE=1 just hil galaxea_a1
```

The G2 gripper has its own opt-in. It uses the vendor example's 10 mm step,
mapped through the normalized `0..1` contract over the configured 104 mm
stroke, chooses the direction away from the nearest endpoint, verifies feedback
within the measured 2.5 mm steady-state tolerance, and returns to the measured
opening even when the outbound-leg assertion fails:

```bash
GALAXEA_A1_HIL=1 GALAXEA_A1_ALLOW_GRIPPER=1 just hil galaxea_a1
```

After the HAL-level gates pass, the full-graph HIL runs inside the deploy
container. It captures the current named-joint feedback itself, requires the
C++ kernel and real HAL to be active while the relay is still `LOCKED`, then
publishes only that measured hold through `candidate_action`. It verifies the
matching `safe_action`, exact staged/forwarded targets, zero kernel drops, and
less than one degree of drift. Its `finally` path publishes `/openral/estop`
three times and requires the HAL diagnostics to confirm the latch:

```bash
docker exec \
  --env GALAXEA_A1_DEPLOY_HIL=1 \
  --env GALAXEA_A1_ALLOW_HOLD=1 \
  openral-galaxea-a1 \
  bash -lc 'source /opt/ros/jazzy/setup.bash && \
    source /workspace/install/setup.bash && \
    pytest -q /workspace/tests/hil/test_galaxea_a1_deploy.py'
```

After the current-pose full-graph gate passes, the same fixture has a separate
motion opt-in. It moves `arm_joint1` by +0.01 rad through
`candidate_action -> C++ safety kernel -> safe_action`, bounds all six joint
excursions, and returns to the measured start before the downstream e-stop:

```bash
docker exec \
  --env GALAXEA_A1_DEPLOY_HIL=1 \
  --env GALAXEA_A1_ALLOW_HOLD=1 \
  --env GALAXEA_A1_ALLOW_NUDGE=1 \
  openral-galaxea-a1 \
  bash -lc 'source /opt/ros/jazzy/setup.bash && \
    source /workspace/install/setup.bash && \
    pytest -q /workspace/tests/hil/test_galaxea_a1_deploy.py'
```

This test intentionally ends the hardware session. Restart both the sidecar
and deploy container before any later motion test.

Do not start a policy on the first pass. Stop both commands and investigate if
feedback/status becomes stale, a motor code other than the manifest's explicit
idle/gripper masks appears, joint order differs, the sidecar exits, or the arm
moves before an approved safe action. Motion validation then proceeds with a
current-pose hold and a single <=0.01 rad joint increment through OpenRAL's
standard candidate-action → C++ kernel → safe-action path, then return-to-start;
only afterwards run an A1-specific rSkill.

#### LingBot-VA rSkill through the complete OpenRAL path

`rskills/lingbot-va-galaxea-a1-fruit-placement/rskill.yaml` is the first
checkpoint-specific A1 rSkill. The dependency direction is deliberate:

```text
A1 Camera Bridge -> OpenRAL WorldState -> LingBot-VA rSkill
  -> A1 Runtime policy gateway (model contract + EEF/cache + IK)
  -> OpenRAL candidate_action -> C++ safety kernel -> safe_action
  -> GalaxeaA1HAL -> isolated ROS 1 sidecar -> official A1 driver
```

The A1 Runtime is a public capability provider, not a second controller:
start only its persistent camera owner, LingBot policy server, and OpenRAL
policy gateway. The gateway has no ROS imports or command publisher. Do not
start its LingBot ROS execution bridge or A1 joint runtime while OpenRAL owns
the deployment. The rSkill owns its `policy_extras.max_joint_substep_rad` replay
setting and reads the independent `max_target_step_rad` ceiling from the same
`RobotDescription` used to construct the HAL. Startup rejects a policy bound
that exceeds either the HAL's live target-step ceiling or locked-relay
alignment tolerance. The policy bound is 0.045 rad, below the 0.05 rad
locked-relay alignment threshold; the independent HAL/sidecar live limit is
0.08 rad. The gateway constructs Runtime's IK implementation with the active
OpenRAL `RobotDescription`'s ordered command limits after verifying they are no
wider than Runtime's envelope. Runtime calibration margins therefore cannot
widen the typed OpenRAL, safety-kernel, or official sidecar command envelope.

The gateway emits one bounded target per 30 Hz control tick. When its IK
solution is farther than 0.045 rad from fresh feedback, it keeps advancing
toward that same solved target on subsequent ticks and only consumes the next
model action after the solved target has been dispatched. The FK of the actual
dispatched target is written into the KV cache, so the policy state reflects
what OpenRAL commanded rather than an unreachable ideal. The official tracker's
steady-state error cannot widen the command envelope or bypass the bounded
step. The A1 Runtime's 1.70 rad IK-solution validation remains an upstream
reachability check, not a motor-command step limit.

```bash
# Terminal A — A1 Runtime capability providers only (no ROS command publisher).
cd /absolute/path/to/A1-Research
just cameras start
scripts/apps/lingbot/a1_lingbot_runtime.sh server
uv run galaxea-a1-openral-policy \
  --config configs/deployments/lingbot/fruit_placement_eef.toml \
  --repo-root .

# Terminal B — official ROS 1 sidecar, as in the bring-up section above.
cd /absolute/path/to/OpenRAL
tools/run_galaxea_a1_sidecar.sh \
  --image openral/galaxea-a1-sidecar:noetic \
  --sdk-root /absolute/path/to/A1_SDK \
  --serial /dev/a1

# Terminal C — the complete OpenRAL real deployment.
cd /absolute/path/to/OpenRAL
uv run --group lingbot openral deploy run \
  --config scenes/deploy/galaxea_a1_bench.yaml
```

Submit the exact trained prompt (for example, `put the red mango into the blue
plate`) through the dashboard. Before allowing a task motion, first repeat the
observation, hold, joint-nudge, gripper, and full-graph gates above. Stop the
LingBot server afterwards with
`scripts/apps/lingbot/a1_lingbot_runtime.sh server-stop`.

The A1 opts into hardware-downstream e-stop: `/openral/estop` stops the
sidecar-owned tracker and driver immediately. The generic
`/openral/estop_cleared` broadcast cannot re-arm this HAL; restart the lifecycle
and sidecar, re-read motor health, and repeat initial alignment instead.

### `python/hal/src/openral_hal/h1.py`
_MuJoCo digital twin for the Unitree H1 humanoid (Menagerie MJCF). Contract validator only — falls without an S0 cerebellum; gravity must be disabled in closed-loop tests (CLAUDE.md §6.2). Unlike the G1 / UR / Franka / SO-100 MJCFs, the H1 menagerie ships ``motor`` (torque) actuators, so this HAL runs a software PD position loop every physics step._

- `class H1MujocoHAL(MujocoArmHAL)` — 19-DoF humanoid HAL driving `mujoco_menagerie/unitree_h1/h1.xml`. Joint inventory: 5 leg + 5 leg + 1 torso + 4 arm + 4 arm (no wrists). Thin manifest-driven wrapper around `MujocoArmHAL`; `__init__` forwards to `self._init_from_description(H1_DESCRIPTION, …)`. Inherits `connect/disconnect/read_state/estop`; overrides `_apply_arm_targets` to a no-op and `_per_step_update` to compute `tau = kp*(target - q) - kv*dq` clamped to `ctrlrange` so the public action contract stays "position targets in radians". Mirrors how `unitree_sdk2` wraps motor-level torque control in a position loop on real hardware. (L358)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L398)
  - `_per_step_update(targets) -> None` — Recomputes PD torque every `mj_step`.
  - `_apply_arm_targets(targets) -> None` — No-op (PD loop runs per-step instead).
- `_h1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `torso` / `shoulder` / `elbow`) for `joint_name`. (L209)
- `_h1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for an H1 joint. (L217)
- `_h1_joint_specs() -> list[JointSpec]` — Build the 19 `JointSpec`s from the joint-name tuples + the per-joint limit tables. (L253)
- `_h1_pd_gains() -> dict[str, tuple[float, float]]` — Per-joint `(kp, kv)` for the software PD loop (kv = 0.05*kp; kp sized so a 1-rad error roughly saturates each actuator's ctrlrange). (L349)
- const `H1_DESCRIPTION = RobotDescription(...)` (L276) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.h1:H1MujocoHAL"` + `hal.real=None` (sim-only until M2). All MuJoCo wiring (MJCF URI, floating-base joint offsets +7/+6, PD gains) lives in `H1_DESCRIPTION.sim`. Drift-guarded against `robots/h1/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/flexiv_rizon4.py`
_MuJoCo digital twin for the Flexiv Rizon 4 — 7-DoF cobot with whole-body force sensitivity (0.1 N).  Structurally identical to the UR / Franka sim HALs: position actuators, no gripper, no floating base, no PD-loop overrides — a clean `MujocoArmHAL` subclass._

- `class Rizon4MujocoHAL(MujocoArmHAL)` — 7-DoF HAL driving `mujoco_menagerie/flexiv_rizon4/flexiv_rizon4.xml` via `MujocoArmHAL`. Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(RIZON4_DESCRIPTION, …)`. (L180)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L210)
- `_rizon4_joint_specs() -> list[JointSpec]` — Build the 7 `JointSpec`s from the joint-name tuple + per-joint limit tables. (L111)
- const `RIZON4_DESCRIPTION = RobotDescription(...)` (L132) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.flexiv_rizon4:Rizon4MujocoHAL"` + `hal.real=None` (sim-only). All MuJoCo wiring lives in `RIZON4_DESCRIPTION.sim`. Drift-guarded against `robots/rizon4/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/openarm.py`
_MuJoCo digital twin for the Enactic OpenArm **v2** bimanual humanoid arm.  Fresh `HALBase` subclass — v2's native `<position>` actuators with per-class PD baked into the MJCF mean the HAL just writes target → ctrl and steps, no software PD loop needed._

- `class OpenArmMujocoHAL(MujocoArmHAL)` — 16-DoF (7 arm + 1 gripper per side) bimanual HAL driving `enactic/openarm_mujoco/v2/openarm_v20_bimanual.xml`; thin manifest-driven wrapper around `MujocoArmHAL` (bimanual amendment). All wiring lives in `OPENARM_DESCRIPTION.sim`: `openarm_v2:bimanual` URI (fetched lazily via `ensure_openarm_v2_mjcf`), explicit `joint_qpos_addr` that skips the passive follower-finger qpos slots (8, 17), two `PASSTHROUGH` grippers (left jaw `[0, 0.7854]`, right jaw `[-0.7854, 0]`), `seed_ctrl_from_qpos: true` so the v2 `<position>` actuators hold the initial pose on the first `mj_step`. (L403)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(OPENARM_DESCRIPTION, …)`. (L438)
- `_openarm_arm_joint_specs(names, position_limits, side) -> list[JointSpec]`, `_openarm_gripper_joint_spec(name, side, position_limits) -> JointSpec`, `_openarm_joint_specs() -> list[JointSpec]` (L167, L190, L206)
- const `OPENARM_DESCRIPTION = RobotDescription(...)` (L238) — sim baseline (`name="openarm_v2"`, all 16 joints revolute matching v2's hinge gripper).  `sdk_kind="open"`, `hal.sim="openral_hal.openarm:OpenArmMujocoHAL"` + `hal.real=None` (sim-only).  Drift-guarded against `robots/openarm/robot.yaml`.

### `python/hal/src/openral_hal/anvil_openarm_v2.py`
_MuJoCo digital twin for the Anvil OpenARM 2.0 — Anvil Robotics' manufactured variant of the standard OpenArm v2 (docs.anvil.bot/introduction/openarm-2.0).  Differs from the Enactic v2 twin in exactly two documented joint ranges (J1 ±135 deg; J6 -45..+70 deg radial deviation) plus the wrist support bracket that enables it — all baked into the fetched MJCF, so the HAL stays a thin manifest-driven subclass (ADR-0023)._

- `class AnvilOpenArmV2MujocoHAL(MujocoArmHAL)` — 16-DoF (7 arm + 1 gripper per side) bimanual HAL driving `models/anvil_openarm_bimanual.xml` from the pinned `bensonlee5/anvil-openarm-mujoco` clone via the `openarm:anvil_v2_bimanual` ref.  All wiring lives in `ANVIL_OPENARM_V2_DESCRIPTION.sim`: joint→qpos map skipping the equality-coupled follower fingers (qpos 8, 17), two `PASSTHROUGH` hinge grippers (left jaw `[0, 0.7854]`, right jaw `[-0.7854, 0]`), `seed_ctrl_from_qpos: true` for the native `<position>` actuators.  Structurally identical to `OpenArmMujocoHAL` — the Anvil-ness is entirely in the asset. (L367)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(ANVIL_OPENARM_V2_DESCRIPTION, …)`. (L405)
- `_anvil_arm_joint_specs(names, position_limits, side) -> list[JointSpec]`, `_anvil_gripper_joint_spec(name, side, position_limits) -> JointSpec`, `_anvil_joint_specs() -> list[JointSpec]` (L160, L184, L204)
- const `ANVIL_OPENARM_V2_DESCRIPTION = RobotDescription(...)` (L215) — sim baseline (`name="anvil_openarm_v2"`, all 16 joints revolute, hinge grippers; J1/J6 carry the Anvil ranges on both arms, J2 keeps the v2 mirrored asymmetry; two wrist RGB `SensorSpec`s rendering the MJCF's `camera_wrist_{left,right}` at 640×400).  `sdk_kind="open"`, `hal.sim="openral_hal.anvil_openarm_v2:AnvilOpenArmV2MujocoHAL"` + `hal.real=None` (sim-only; a wrapper around Anvil's driver stack is a tracked follow-up).  Drift-guarded against `robots/anvil_openarm_v2/robot.yaml`.

### `python/hal/src/openral_hal/_pinned_clone.py`

- `fetch_pinned_clone(repo_url, sha, repo_dir, *, submodule=None, what=…) -> None` — THE shared staged-clone + atomic-rename dance for the pinned-SHA asset fetchers (shallow `--filter=blob:none` clone into a per-call staging dir, checkout of the pinned SHA, optional single-submodule init, `os.rename` into place with the loser of a concurrent race discarded). Extracted from the two OpenArm fetchers, which carried ~30 identical concurrency-critical lines each; a future fix (e.g. `EXDEV` on cross-filesystem rename) now lands once. Raises `ROSConfigError` when `git` is missing or any git step fails. (L30)

### `python/hal/src/openral_hal/_anvil_openarm_v2_assets.py`
_Vendor the Anvil OpenARM 2.0 MJCF from `bensonlee5/anvil-openarm-mujoco` — no upstream package ships the Anvil variant._

- `ensure_anvil_openarm_v2_mjcf() -> str` — Idempotently clones `bensonlee5/anvil-openarm-mujoco` at a pinned SHA into `$OPENRAL_CACHE_DIR/anvil_openarm_v2/<sha>/`, initialises its `upstream/openarm_mujoco` mesh submodule (the generated MJCF's meshdir points into it), and returns the bimanual MJCF path.  Raises `ROSConfigError` when `git` is missing or the clone / submodule init fails.  Mirrors `_openarm_v2_assets.ensure_openarm_v2_mjcf` plus the submodule step. (L60)
- module const `_ANVIL_PINNED_SHA: str` (L40) — bump when the generator or the local Anvil spec changes.

### `python/hal/src/openral_hal/_openarm_v2_assets.py`
_Vendor the upstream `enactic/openarm_mujoco` v2 MJCF until `robot_descriptions` bumps its pin past PR #19._

- `ensure_openarm_v2_mjcf() -> str` — Idempotently clones `enactic/openarm_mujoco` at a pinned v2 SHA into `$OPENRAL_CACHE_DIR/openarm_v2/<sha>/`, returns the bimanual MJCF path. Raises `ROSConfigError` when `git` is missing or the clone fails. Mirrors the pattern used by `python/sim/src/openral_sim/backends/so100_robosuite/_assets.py`. (L64)
- module const `_OPENARM_V2_PINNED_SHA: str` (L47) — bump to track upstream v2 updates.

### `python/hal/src/openral_hal/g1.py`
_MuJoCo digital twin for the Unitree G1 humanoid. The default stock-Menagerie path provides joint-contract validation + ADR-0087 kinematic glide; explicit `walking_enabled=True` selects ADR-0089's pinned MuJoCo Playground ONNX policy and matching gravity-on dynamics._

- `class G1MujocoHAL(MujocoArmHAL)` — 29-DoF humanoid HAL. Default mode drives `mujoco_menagerie/unitree_g1/g1.xml`; walking mode swaps in the pinned policy-tuned MJCF + ONNX controller. Floating-base joint remains implicit world state. (L388)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5, body_twist_dt_s=0.05, walking_enabled=False)` (L426)
  - `base_pose -> tuple[float, float, float]` (property) — current glide pin or live walking base pose. (L484)
  - `base_twist -> tuple[float, ...]` (property) — last commanded 6-vec base twist (`base_link` frame), zeroed by any non-BODY_TWIST action (matches `PandaMobileHAL`). (L492)
  - `send_action(action)` — routes `BODY_TWIST` to either the ADR-0087 glide or ADR-0089 walking controller; every other mode defers to the joint path. Non-planar twist components raise `ROSConfigError`. (L521)
  - `idle_step(wall_dt_s=None) -> bool` — HOLD-step with the base pinned; walking state resets so a stale velocity command is never replayed. G1 opts out of the bare-arm `_step_while_active` path because an active BODY_TWIST must not be replaced by this zero/HOLD behavior. (L568)
  - `_per_step_update(targets)` / `_pin_base()` / `_pin()` — the upright pin: `qpos[0:7]` = glide pose (roll/pitch clamped to 0), `qvel[0:6]` = 0, captured lazily from the fresh qpos after connect. Replacing the pin with a balance controller is the designed S0 upgrade seam. (L608)
- `_g1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `waist` / `shoulder` / `elbow` / `wrist`) for `joint_name`. (L218)
- `_g1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for a G1 joint, following the menagerie URDF convention. (L226)
- `_g1_joint_specs() -> list[JointSpec]` — Build the 29 `JointSpec`s from the joint-name tuples and the per-joint limit tables. (L276)
- const `G1_DESCRIPTION = RobotDescription(...)` (L304) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.g1:G1MujocoHAL"` + `hal.real=None` (sim-only until M2). Advertises `supported_control_modes=[joint_position, body_twist]`, `embodiment_tags` incl. `mobile_base`, and a forward `head` RGB camera (`vla_feature_key=observation.images.head`, rigged onto `torso_link` via the ADR-0086 camera rig) so BODY_TWIST nav skills (InternVLA-N1 VLN) match. All MuJoCo wiring (MJCF URI, floating-base joint offsets) lives in `G1_DESCRIPTION.sim`. Drift-guarded against `robots/g1/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/_g1_walking.py`
_ADR-0089's private, sim-only walking implementation._

- `ensure_g1_walking_assets() -> tuple[str, str]` — downloads four files from pinned MuJoCo Playground commit `43d180a`, verifies SHA-256, links the cached Menagerie meshes, flattens the included MJCF, and returns `(mjcf_path, policy_path)`.
- `class G1WalkingController` — 50 Hz CPU ONNX inference over 500 Hz MuJoCo physics; builds the upstream 103-D observation, validates the 29-D finite output, scales/clips joint targets, and persists gait phase/action history.

### `python/hal/src/openral_hal/so100_mujoco.py`
_MuJoCo digital twin for the SO-100 follower arm (Menagerie MJCF)._

- `class SO100MujocoHAL(MujocoArmHAL)` — SO-100 follower MuJoCo HAL, driving the `mujoco_menagerie` `trs_so_arm100/so_arm100.xml` with the same 6-DoF action layout as `SO100FollowerHAL`. Maps the lerobot-style description joint names to the Menagerie joints (`shoulder_pan→Rotation`, …, `gripper→Jaw`) and normalises the revolute Jaw range `[-0.174, 1.75]` to `[0, 1]`. Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(SO100_DESCRIPTION, …)`. (L68)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L108)
  - `_read_gripper_normalised() -> float` — Override that offsets the closed position from `-0.174` rad (the base helper assumes closed at qpos == 0).

### `python/hal/src/openral_hal/so100_sim.py`
_SO100DigitalTwin — in-process simulator for the SO-100 follower arm._

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
_RosControlHAL — `ros2_control`-backed HAL adapter._

- `class RosControlHAL` — `ros2_control`-backed HAL adapter. (L72)
  - `__init__(description, controller_name, *, joint_state_topic='/joint_states', command_topic=None, publish_fn=None, state_fn=None, staleness_limit_s=0.5)` (L101)
  - `connect() -> None` (L132)
  - `disconnect() -> None` (L150)
  - `read_state() -> JointState` (L162)
  - `send_action(action) -> None` — Publish JointTrajectory. (L199)
  - `estop() -> None` (L230)
  - private: `_require_connected`, `_validate_action`
- `_default_publish(topic, msg) -> None` — No-op publish when no real ROS 2 node. (L62)

### `python/hal/src/openral_hal/sim_transport.py`
_SimTransport — in-memory simulated `ros2_control` transport._

- `class SimTransport` — In-memory transport simulating a JointTrajectory controller. (L32)
  - `__init__(n_joints)` (L63)
  - `publish(topic, msg) -> None` — Record msg, apply `joint_targets`. (L73)
  - `state() -> dict[str, object]` — Current simulated joint state. (L91)
  - `call_count -> int` [@property] (L107)
  - `last_call -> tuple | None` [@property] (L112)
  - `calls -> list[tuple]` [@property] (L117)

### `python/hal/src/openral_hal/lifecycle.py`
_Generic ROS 2 managed lifecycle node wrapper for every HAL adapter — UR5e / UR10e / Franka / SO-100 / OpenArm / H1 / future HALs all share the same publish / subscribe / heartbeat / OTel-span wiring._

- `class HALLifecycleNodeBase(LifecycleNode)` — Public base class. Owns the standard `/joint_states` + `~/joint_states` publishers, the `/openral/safe_action` + `/openral/estop` subscribers, the 1 Hz `DiagnosticsHeartbeat`, the per-tick `hal.read_state` + `hal.send_action` OTel spans, the estop latch, and the full configure → activate → deactivate → cleanup → shutdown transition wiring. The formerly used `~/command` (`trajectory_msgs/JointTrajectory`) subscriber + its `_on_command` callback + the `_subscriber` field were removed; `_send_action_traced` is now driven only by `_on_safe_action`. (L369)
  - `_create_hal(self) -> HAL` — **Subclass hook (required)**: construct and return a HAL instance. Reads ROS-parameter-driven constructor args via `self.get_parameter(...)`. (L437)
  - `_heartbeat_extra_fields(self) -> dict[str, str]` — Subclass hook (optional): extra key/values for the `/diagnostics` payload (e.g. `{"port": "/dev/ttyUSB0"}` for SO-100, `{"mjcf": "..."}` for OpenArm). Default: `{}`. (L449)
  - `on_configure_post_hal(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific setup after the HAL connects (e.g. opening a camera renderer on OpenArm). Default: `SUCCESS`. (L501)
  - `on_activate_post_subs(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific timers/publishers after the base wires its subs (e.g. the OpenArm camera-render timer). Default: `SUCCESS`. (L510)
  - `on_deactivate_pre_teardown(self) -> None` — Subclass hook (optional): stop robot-specific timers before base teardown. Default: no-op. (L518)
  - `on_cleanup_pre_disconnect(self) -> None` — Subclass hook (optional): tear down robot-specific resources (viewers, renderers) before HAL.disconnect(). Default: no-op. (L525)
  - `_publish_joint_state(self) -> None` — Timer callback. Wraps `self._hal.read_state()` in a `hal.read_state` span (identity attrs + `producer.record_joint_state`) and publishes the standard `/joint_states` + `~/joint_states` messages; when the HAL exposes `read_policy_state`, also publishes `/openral/policy_state` (`std_msgs/Float32MultiArray`) — only when the underlying `ProprioFrame` is NEW (one publish per env.step capture, never a latched republish, so the world-state aggregator's dedicated `policy_state` staleness window genuinely trips on a wedged simulator). Subclasses may override + call `super()._publish_joint_state()` to extend (OpenArm does this for viewer-sync). (L848)
  - `_on_safe_action(self, msg) -> None` — `/openral/safe_action` callback. Decodes the `openral_msgs/ActionChunk` into an `openral_core.Action` and forwards through `_send_action_traced(action, source="safe_action")`. (L943)
  - `_send_action_traced(self, action, *, source) -> None` — Forward `action` to `self._hal.send_action` inside a `hal.send_action` span. The `source` attribute disambiguates the origin on the dashboard's Commands card (kept on the span so future subscriber additions can fan in without changing the span shape). (L963)
  - `_on_estop(self, msg) -> None` — `/openral/estop` callback. Ordered latch → stop → report: sets `_estopped` (which makes `_on_safe_action` drop commands), calls `_invoke_hal_estop`, then reports via `_emit_estop_telemetry` in a `finally` so an e-stop is counted even when the vendor stop path raises and even for HALs that opt out of it. Nothing is added ahead of the physical stop.
  - `_invoke_hal_estop(self) -> None` — Calls `self._hal.estop()` for HALs implementing `LifecycleEStopHAL`; swallows `ROSEStopRequested` (expected completion signal) and logs any other failure at fatal — the latch must survive a vendor stop-path failure.
  - `_emit_estop_telemetry(self) -> None` — Emits the `openral.event.estop_requested` span event (on the active span, else a transient `hal.estop` span) and increments `openral.hal.estop.count` with a `hal.adapter` label. **This is the only producer of either signal.** The dashboard ingests OTLP, not ROS topics, so `/openral/estop` is invisible to it and its Command-band `e-stops` counter read 0 no matter how many e-stops fired — a safety indicator that cannot leave zero reads as an affirmative "no e-stops have occurred". This node is the right chokepoint: it is the shared base every robot HAL runs on and it sits on the actuation side, whereas counting at the six `/openral/estop` publishers would need six call sites (one of them the C++ kernel) and counting at every subscriber would multiply one e-stop into several. Never raises — telemetry must not disturb the stop path (CLAUDE.md §1.1).
- `make_lifecycle_main(node_name, hal_factory) -> Callable[[], None]` — Build a `main()` entry point for a zero-parameter HAL adapter. Internally constructs a `_FactoryHALLifecycleNode(HALLifecycleNodeBase)` whose `_create_hal()` returns `hal_factory()`. Superseded for the standard arms by `make_lifecycle_main_from_manifest`; retained for bespoke nodes. (L241)
- `class ManifestHALLifecycleNode(HALLifecycleNodeBase)` — Public generic manifest-driven lifecycle node (promoted from the private `_ManifestHALLifecycleNode` under issue #191). Reads `robot_yaml` + `hal_mode` + sensor knobs as ROS params and builds its HAL via `openral_hal.build_hal`, so a robot's construction kwargs come from the manifest's `hal.parameters.defaults` — no bespoke `_create_hal` subclass. Attaches `SimSensorBridge` (cameras / depth / scan / viewer) in `on_activate_post_subs`. In `on_configure_post_hal`, **reflects** on the built HAL and opens `/openral/<robot>/reset_to_pose` (`openral_msgs/srv/ResetToPose`) iff it exposes `reset_to_pose` — generalising the openarm-only service to every `MujocoArmHAL` sim arm (issue #191 Phase 2); HALs without the method (panda_mobile, scene-attached twins) get no service. In `_create_hal`, when a scene composition is declared (and not scene-attaching), calls the named composer and threads the composed MJCF in as the HAL's `mjcf_path` (issue #191 Phase 3b — openarm tabletop); the composition is read from the `scene_composition_json` ROS param (the DeployScene's own `composition`) which **takes precedence** over the robot manifest's `scene_defaults.composition` (back-compat fallback) — so the scene owns its arena, the robot manifest describes the robot. Bare-twin camera robots (so100/so101) need no composition: their cameras are spliced by the generic camera rig at HAL connect from `sensors[].sim_placement`. In `on_activate_post_subs`, when the manifest declares a planar base (`base_joints`), also attaches a `MobileBaseBridge` (`/odom` + `odom->base_link` TF + `/cmd_vel`→BODY_TWIST) — so panda_mobile runs on this node with no subclass (issue #191 Phase 3a). The per-robot lifecycle packages collapse into this node (issue #191 Phases 2-3). A back-compat alias `_ManifestHALLifecycleNode` is retained. (L1184)
- `make_lifecycle_main_from_manifest(node_name) -> Callable[[], None]` — Build a `main()` that spins up `ManifestHALLifecycleNode`. The node reads `robot_yaml` + `hal_mode` ("sim"|"real") ROS params and constructs its HAL via `openral_hal.build_hal(description, mode=hal_mode)` — one node class serves both modes for every robot. Used by franka / ur5e / ur10e / aloha / g1 / h1 / rizon4 / so100 / so101 (issue #191 Phase 2 migrated so100/so101 off their bespoke node); `openral deploy sim` injects `hal_mode="sim"`, `openral deploy run` injects `hal_mode="real"`. A robot lacking the requested mode raises `ROSCapabilityMismatch`. (L299)
- `decode_action_chunk(msg) -> Action | None` — Inverse of `ros_publishing_hal._flatten_action_payload`. Decodes the `ActionChunk` wire shape (`flat` + `n_dof` + `horizon` + `control_mode`) back into a typed `openral_core.Action` with the per-mode payload field populated (`cartesian_delta` / `gripper` / `body_twist` / `joint_*`). Returns `None` for degenerate chunks (`flat=[]`, `n_dof≤0`) and for modes the F1/F5 publisher doesn't encode (`CARTESIAN_POSE`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT`). Preserves `ee_name`, `frame_id`, `confidence` (decoded verbatim — an explicit 0.0 is NOT coerced to 1.0), and the shared inference `tick_index` across the safety wire. Used by `HALLifecycleNodeBase._on_safe_action`; lives at module scope so unit tests in `tests/unit/test_lifecycle_action_chunk_decoder.py` exercise it without a ROS 2 install. (L106)

### `python/hal/src/openral_hal/sim_bringup.py`
_Resolve a `SimScene` or `BenchmarkScene` YAML path to a live `SimRollout`. Used by `build_hal`, which every manifest-driven node (incl. panda_mobile, issue #191 Phase 3) routes through._

- `build_sim_env_from_yaml(sim_env_yaml: str, *, robot_id_fallback: str | None = None) -> tuple[SimRollout, int | None]` — Load a `SimScene` or `BenchmarkScene` YAML, resolve its scene id in `openral_sim.SCENES`, and instantiate the env. Relative paths are resolved by walking parents of the source file (ROS param values are cwd-naïve). Returns `(env, seed)` — the caller plumbs the seed into `SimAttachedHAL(env_reset_seed=seed)`. Raises `ROSConfigError` when the YAML is not found, the scene id is unregistered, or schema validation fails. Robocasa scenes have `ignore_done=True` injected so deploy-sim continuous stepping does not trip the episode-done guard; strict-validation native backends (e.g. `tabletop_push`) are unchanged. (L173)

### `python/hal/src/openral_hal/sim_attached.py`
_`SimAttachedHAL` — generic HAL Protocol adapter that wraps any in-process `SimRollout`. Shared by `panda_mobile`, manifest-driven arms, and tests; not import-safe without `openral_sim` + `mujoco`._

- `ActionPacker` — Type alias `Callable[..., np.ndarray]`. Per-composition translator between an OpenRAL `Action` and the env's flat action vector; the default factory is `pack_action_for_env`. The optional trailing `prev` arg (the previous env-frame command, or `None`) lets a packer carry an untouched slot — e.g. the gripper while the arm steps — across the two typed Actions one policy step splits into on a non-composite env. Pass a custom instance to `SimAttachedHAL.__init__` for whole-body humanoid or dexterous-hand action layouts. (L181)
- `normalized_joint_index(model_joint_names: list[str]) -> dict[str, int]` — Map MJCF joint names (exact + robosuite-prefix-stripped) to model index. Exact names always win; `robot0_joint1` → `joint1` (strip `^[a-z]+[0-9]+_`) is added only when it neither shadows an exact name nor collides (bimanual `robot0_`/`robot1_` ambiguity → keep explicit). Used by `SimAttachedHAL.read_state` so one manifest serves both native MjSpec and robosuite scenes; `robot0_` never appears in a manifest. (L121)
- `is_terminated_episode_error(exc: BaseException) -> bool` — True iff `exc` is robosuite's post-terminal step guard (`ValueError("executing action in terminated episode")`, `environments/base.py`), matched by message substring (case-insensitive, stable across robosuite releases). Raw robosuite-backed adapters with `ignore_done=False` can HARD-RAISE this instead of returning a terminal `StepResult`; `SimAttachedHAL._step_and_cache` keys its raised-terminal recovery (reset + re-step) off this predicate so deploy-sim's continuous twin keeps driving. Any other `step` fault returns `False` and propagates (never silently swallowed). (L103)
- `pack_action_for_env(action: Action, description: RobotDescription, env_action_dim: int, prev: np.ndarray | None = None) -> np.ndarray` — Default `ActionPacker`. Translates `JOINT_POSITION` (arm-only or full base+arm), `BODY_TWIST` (vx/vy/wz → slots 0-2), `CARTESIAN_DELTA` (6-vec → arm slots `[base_dim:]`), and `GRIPPER_POSITION` (→ last slot) into the env's flat action vector. Raises `ROSConfigError` for unsupported modes or mismatched row widths. A single VLA policy step on a non-composite env (LIBERO OSC_POSE, SimplerEnv widowx — every `delta_ee_6d_plus_gripper` rSkill) splits into a CARTESIAN_DELTA then a GRIPPER_POSITION Action, each `env.step`-ed; `prev` (threaded from `SimAttachedHAL._last_env_action`) carries the last commanded gripper through the arm step and holds the arm on the gripper step, so the arm advances once per policy step with the gripper always commanded — mirroring `_pack_with_composite_split`. Without it each Action zeroed the other's slots and the arm barely moved. (L199)
- `class SimAttachedHAL` — HAL Protocol adapter wrapping an in-process `SimRollout`. Reads live joint state via `normalized_joint_index` + `mj_name2id`; sends actions via `pack_action_for_env` (or a caller-supplied `ActionPacker`) into `env.step()`. Exposes `read_images()`, `mujoco_handles()`, `sim_time_ns()`, `base_pose`, `base_twist`, `base_pose_6dof()` for the ROS lifecycle node's camera publisher, viewer, sim-clock, and odom wiring. (L383)
  - `__init__(env: SimRollout, description: RobotDescription, *, action_packer: ActionPacker | None = None, env_reset_seed: int | None = None, env_action_dim: int | None = None, body_twist_dt_s: float = 0.05) -> None` (L412)
  - `connect() -> None` — Reset the env at `env_reset_seed`; probe `env_action_dim` (via `_probe_env_action_dim`, which raises `ROSConfigError` naming the backend when no `action_dim` is introspectable and no override was given — never a silent fallback); invalidate joint-index cache. Idempotent. (L533)
  - `disconnect() -> None` — Release env handle (idempotent). (L608)
  - `read_state() -> JointState` — Walk `description.joints`, resolve each joint via `normalized_joint_index`, read live `qpos`/`qvel` from MJCF. (L612)
  - `send_action(action: Action) -> None` — Pack action via composite-split or `ActionPacker`; call `env.step`. `BODY_TWIST` takes a direct-qpos Euler-integration path on a MuJoCo backend (`_apply_body_twist_to_qpos`, skips `env.step` so the arm doesn't churn); on a non-MuJoCo backend (Isaac kinematic base) it routes through `_apply_body_twist_via_env_step` instead — the scene integrates the base inside `env.step`. Stamps `last_action_ns` at the top (the single choke point both `_on_safe_action` and `_on_cmd_vel` reach) so the idle stepper yields to it. Routes the step through `_step_and_cache`, which auto-resets on episode termination — both the *returned* terminal (`StepResult.terminated/truncated` latched as `_episode_done`) and a *raised* terminal (raw-robosuite `ignore_done=False` backends throwing `is_terminated_episode_error`); the raised path resets once and re-steps so deploy-sim never freezes with the "env.step failed: executing action in terminated episode" spam. (L686)
  - `_stage_action_group(action, group_step) -> None` — For backends exposing `action_group_size` + `step_action_group` (BEHAVIOR-1K's six typed slots committed atomically as one 23-D evaluator step), stage each safety-approved slot by `Action.tick_index` and commit exactly one simulator step only when the complete tick is present. Rejected/missing slots never actuate (atomicity), and drops are LOUD: each incomplete-group drop prints an ERROR line, and after 3 consecutive drops it raises `ROSRuntimeError` naming the slot-count mismatch / persistently-rejected slot — the sim never silently freezes. On commit, latches the group's BODY_TWIST slot into `_last_body_twist` so `base_twist` reflects the commanded base velocity.
  - `read_policy_state() -> list[float] | None` — Cached simulator-native checkpoint proprioception (e.g. BEHAVIOR's official 61-D R1Pro vector) for `/openral/policy_state`; only an explicit `obs["policy_state"]`, never inferred from joint state.
  - `idle_step() -> bool` — **Sim-only** free-running stepper (2026-06-04 amendment). Advances the wrapped `SimRollout` one tick with `np.zeros(env_action_dim)` (HOLD) so cameras keep rendering when no skill is executing — without it an idle deploy-sim scene freezes and the perception bus sees a dead scene. Returns `False` (suppressed) when not connected, estop-latched, `env_action_dim is None`, or no live MuJoCo handles; else steps and returns `True`. Mirrors `send_action`'s deferred-reset branch and `_last_obs` re-cache; does NOT touch `_last_env_action` / `_last_body_twist`. **Defined ONLY on `SimAttachedHAL`** — real HALs never define it; this method-only exclusion (not "zero is harmless") is the primary real-hardware guard, since a zero vector is a HOLD in sim but "drive to 0 rad" on a real position arm. (L1023) Refuses to interleave a HOLD inside a mid-flight atomic action group, but discards a pending group loudly after 5 s without a new slot so a skill that dies mid-tick cannot freeze scene physics/cameras forever.
  - `read_images() -> dict[str, Any]` — Return latest rendered camera frames keyed by camera name from the cached `_last_obs`. (L1628)
  - `read_depth_clouds() -> dict[str, NDArray]` — Per-depth-sensor `(N,3)` `base_link` point clouds from `_last_obs["depth_points"]` (a non-MuJoCo backend, e.g. the Isaac scene, deprojects via `Camera.get_pointcloud` so the HAL never re-derives geometry); `{}` when the backend renders no depth. `SimSensorBridge` publishes them as `PointCloud2` for octomap.
  - `read_scan() -> NDArray | None` — The 2-D LaserScan range fan (`base_link`, `angle_min=-π`→`+π`) from `_last_obs["scan"]` when a non-MuJoCo backend ray-casts a lidar (the Isaac scene); `None` when it renders no lidar. `SimSensorBridge._compute_scan_ranges` reads it for `/scan`.
  - `mujoco_handles() -> tuple[Any, Any] | None` — Forward the env's `(model, data)` MJCF handles. (L1508)
  - `sim_time_ns() -> int | None` — Cross-reset-monotonic elapsed sim time in ns — the seam a sim `/clock` publisher reads. Returns the wrapped `SimRollout.sim_time_ns()` (per-episode) plus an accumulated offset: `connect` and the auto-resets fold each finished episode's elapsed sim-time into the offset (`_accumulate_sim_time_before_reset`) BEFORE the backend rewinds its clock, so the value is monotonic non-decreasing across `env.reset` (robocasa rewinds `MjData.time` to 0). `None` when the wrapped rollout has no sim clock (clock-less backend / sidecar) — the consumer then falls back to wall time.
  - `clock_authority() -> ClockAuthority` — Return the timestamp authority this HAL contributes to the graph: `ClockAuthority.simulation(<backend>, timestep_s=body_twist_dt_s)` when `sim_time_ns()` is live, otherwise `ClockAuthority.host_wall()` so launch keeps the graph on the host-wall authority.
  - `estop() -> None` — Latch e-stop; subsequent `send_action` calls are dropped. (L1500)
  - `base_pose -> tuple[float, float, float]` [@property] — Current `(x, y, yaw)`: from MJCF qpos on a MuJoCo backend, else from `obs["base_pose"]` the SimRollout surfaces (Isaac kinematic base); `(0,0,0)` when the backend reports neither. Feeds the `/odom` publisher. (L1692)
  - `base_twist -> tuple[float, float, float, float, float, float]` [@property] — Last commanded body twist `(vx, vy, vz, wx, wy, wz)`. (L1745)
  - `_apply_body_twist_via_env_step(row: list[float]) -> None` — Non-MuJoCo `BODY_TWIST`: validate the planar twist, latch it for `/odom`, pack `(vx, vy, wz)` into the FINAL three env-action slots (the manifest scene's `[arm…, gripper, base-twist]` layout), zero the arm/gripper so a pure base move holds the arm, and `_step_and_cache`.
  - `base_pose_6dof() -> tuple[...] | None` — Full 6-DoF `(xyz, quat_xyzw)` from robocasa `raw_proprio`; falls back to `None` for non-robocasa backends. (L1755)
  - `last_action_ns -> int` [@property] — Monotonic ns of the last real action through `send_action`; `0` until the first one. The idle stepper reads it (via `should_idle_step`) to yield to an active skill. (L1618)

### `python/hal/src/openral_hal/sim_sensor_bridge.py`
_Shared sim-sensor + viewer bridge for scene-attached HAL lifecycle nodes. Republishes RGB camera frames and a live MuJoCo viewer for any manifest-driven node, and runs the sim-only idle stepper. Phase 2 adds `/scan` + depth `PointCloud2`. Depth comes from the MuJoCo ray-cast (`_publish_depth_clouds`) OR, for a non-MuJoCo backend that surfaces ready `base_link` clouds in obs (the Isaac scene), `_publish_depth_clouds_from_obs` — which wraps `hal.read_depth_clouds()` into a `base_link` `PointCloud2` (no ray-cast, no per-camera optical TF); `_setup_depth` creates the publishers when either source is present. rclpy imported lazily._

- `should_idle_step(now_ns: int, last_action_ns: int, idle_hold_ns: int, *, step_while_active=False) -> bool` — Pure predicate (no rclpy) for the sim-only wall-time stepper. Scene-attached environments return `True` only after the idle-hold window; bare `MujocoArmHAL` passes `step_while_active=True` so its current control target keeps integrating during active skills and `/clock` + camera timers stay live. Used by `SimSensorBridge._idle_step_tick`; unit-testable in isolation. (L48)
- `constant_scan_no_hit_ranges(*, n_beams: int, max_range_m: float) -> list[float]` — Pure (no rclpy) synthetic `/scan` fan: every beam clamped to `max_range_m` ("no hit everywhere"), the honest reading for an in-process digital twin with no scene to ray-cast. Used by `SimSensorBridge._compute_scan_ranges`'s no-handles branch; moved out of the panda_mobile node in issue #191 Phase 3. (`sim_sensor_bridge.py`)
- `class SimSensorBridge` — Wire and tear down sim-sensor publishers and the MuJoCo viewer on a HAL lifecycle node. Streams are gated on the robot manifest + HAL capability. Owns `/scan` for **both** paths since issue #191 Phase 3: live MJCF ray-cast when `hal.mujoco_handles()` is bound (`SimAttachedHAL`), a `constant_scan_no_hit_ranges` fan for the bare digital twin (the node no longer publishes its own scan). (L147)
  - `__init__(node: Any, hal: Any, description: RobotDescription, *, viewer_enabled: bool = True, camera_rate_hz: float = 10.0, viewer_sync_rate_hz: float = 30.0, scan_rate_hz: float = 10.0, scan_n_beams: int = 360, scan_max_range_m: float = 12.0, scan_min_range_m: float = 0.05, depth_rate_hz: float = 10.0, depth_max_range_m: float = 5.0, depth_pixel_stride: int = 4, idle_hold_ms: float = 200.0, on_step: Any = None) -> None` — `on_step`: optional zero-arg callback invoked after each successful `idle_step` (the node refreshes the proprio snapshot through it, so odom/joint_state stay fresh while idle). (L164)
  - `setup() -> None` — Activate all streams the manifest + HAL support: RGB camera publishers on `/openral/cameras/<n>/image` + a per-frame `CameraInfo` companion on `/openral/cameras/<n>/camera_info` (pinhole intrinsics derived from the MJCF `cam_fovy`: `fy=(h/2)/tan(fovy/2)`, `fx=fy`, centre principal point, stamped with the manifest `frame_id` — what cuVSLAM builds its rig from and nvblox frames mono depth against; gated on `hasattr(hal, "read_images")` + RGB `SensorSpec`), the sim-only idle stepper (gated on `callable(getattr(hal, "idle_step", None))` + live MuJoCo handles), viewer (`mujoco.viewer.launch_passive` with `show_left_ui=False, show_right_ui=False` so only the sim renders; `_aim_viewer_camera` then sets the opening **free-camera** pose via `initial_viewer_camera` — eye at a 3rd-person scene camera, orbit pivot on the base, base-aligned default for camera-less twins — leaving the camera `mjCAMERA_FREE` so the user can orbit/zoom; GL/DISPLAY failure → warn + continue). Idempotent per activate. (L296)
  - `teardown() -> None` — Cancel timers (incl. the idle-step timer), destroy publishers, close viewer. Called from `on_deactivate` / `on_cleanup`. (L305)
- `class MobileBaseBridge` — Generic planar-mobile-base ROS wiring (sibling of `SimSensorBridge`): owns `/odom`, the `odom->base_link` TF, and the `/cmd_vel`→BODY_TWIST bridge (out-of-scope: bypasses the safety supervisor; Nav2's `velocity_smoother` caps velocity). Frame ids come from `RobotDescription.{odom_frame,base_frame}`; the HAL must expose `base_pose` (`base_pose_6dof()` / `base_twist` used when present). `ManifestHALLifecycleNode` attaches it in `on_activate_post_subs` iff the manifest declares `base_joints` — so a mobile robot needs no node subclass (issue #191 Phase 3, replaced the bespoke panda_mobile node). (`mobile_base_bridge.py`)
  - `__init__(node, hal, description, *, odom_rate_hz: float = 20.0, cmd_vel_topic: str = "/cmd_vel", proprio: Any = None) -> None` — `proprio`: when set (sim-attached HALs), odom is published from the node's dedicated thread via `publish_from_snapshot`, reading the snapshot not the simulator; `None` (real HALs) keeps the legacy odom timer.
  - `setup() -> None` — Create the `/odom` publisher + TF broadcaster + `/cmd_vel` subscription; the odom timer is created **only when `proprio is None`** (sim HALs publish odom off the node's thread).
  - `publish_from_snapshot() -> None` — Dedicated-thread entry: publish one `/odom` + TF sample from the proprio snapshot (never the simulator). Thin alias over `_publish_odom` (which branches on `proprio`).
  - `teardown() -> None` — Cancel the timer (if any) + destroy the publisher/subscription. Idempotent.

### `python/hal/src/openral_hal/proprio_snapshot.py`

Decouples the control-critical publishers (odom / joint_state / TF) from the single executor thread that runs `env.step` + render + raycast. The sim-attached HAL node captures a frame after each step (on the executor thread, where reading the sim is safe) and a dedicated publisher thread re-emits it at ~30 Hz, so odom stays fresh (~28 Hz live, vs ~1.8 Hz starved) without ever touching `MjData`/GL off the executor thread (a `MultiThreadedExecutor` was rejected — MuJoCo's GL context is thread-affine).

- `class ProprioFrame` — Frozen dataclass: one coherent proprio sample (`state: JointState`, `base_pose: (x,y,yaw)`, `base_pose_6dof: ((x,y,z),(qx,qy,qz,qw)) | None`, `base_twist: tuple[float, ...]`, `sim_time_ns: int | None` — sim time carried for the /clock publisher, `policy_state: tuple[float, ...] | None` — simulator-native checkpoint state vector). Plain immutable data only — no live simulator handles — so it is safe to publish from a different thread than the one that stepped the sim. (`proprio_snapshot.py`)
- `class ProprioSnapshot` — Lock-guarded holder for the latest `ProprioFrame`. One writer (the executor thread, after each step) calls `set`; readers (the publisher thread) call `latest`; the immutable frame is swapped under the lock so a reader never sees a torn frame and never reaches the HAL. HAL-agnostic — the node does the capture. (`proprio_snapshot.py`)
  - `set(frame: ProprioFrame) -> None` — Atomically publish `frame` as the latest sample (executor thread only). (`proprio_snapshot.py`)
  - `latest() -> ProprioFrame | None` — Return the most recent frame, or `None` before the first capture. (`proprio_snapshot.py`)
