# Layer 1 — Hardware Abstraction (HAL)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/hal/src/openral_hal/protocol.py`
_HAL Protocol — the normative interface every HAL adapter must satisfy._

- `class HAL(Protocol)` — Structural protocol every HAL adapter must satisfy. (L26)
  - attr `description: RobotDescription`
  - `connect() -> None` — Open connection to robot/sim. (L43)
  - `disconnect() -> None` — Close connection (idempotent). (L53)
  - `read_state() -> JointState` — Latest joint state snapshot (hot path). (L61)
  - `send_action(action: Action) -> None` — Forward action chunk to controller (hot path). (L75)
  - `estop() -> None` — Trigger emergency stop, always raises `ROSEStopRequested`. (L91)

### `python/hal/src/openral_hal/_mujoco_arm.py`
_Internal MuJoCo-backed HAL implementation shared by UR / Franka / SO-100 / G1 / H1 / Rizon-4 / OpenArm / ALOHA adapters. Reads its wiring from `RobotDescription.sim` (ADR-0023)._

- `class _MujocoArmInitKwargs(TypedDict)` — Typed shape of the kwargs accepted by `MujocoArmHAL.__init__`. (L61) Lets `MujocoArmHAL._sim_kwargs_for` return a value that unpacks cleanly into the constructor under `mypy --strict` without the `# type: ignore[arg-type]` hatch every thin subclass used to need. Fields: `mjcf_path, joint_qpos_addr, joint_qvel_addr, actuator_index, grippers, keyframe_index, seed_ctrl_from_qpos, settle_steps, gravity_enabled, staleness_limit_s`.
- `_resolve_mjcf_path(desc: RobotDescription) -> str` [private] — Resolve `desc.assets.mjcf` (ADR-0058) to an absolute MJCF path via `openral_core.assets.resolve_asset`; raises `ROSConfigError` when the ref is unset or unresolvable. Replaced the former public `resolve_mjcf_uri` / `SimDescription.mjcf_uri` (ADR-0058). (L81)
- `build_hal(description, *, mode: Literal["sim","real"], transport=None, sim_env_yaml=None) -> HAL` — Single seam for constructing a robot's simulation or real-hardware HAL from its manifest (`resolver.py`, L50). `mode="sim"` + `sim_env_yaml` set → calls `build_sim_env_from_yaml` and returns a `SimAttachedHAL` wrapping the scene's `SimRollout` (ADR-0034); bypasses the bare-twin / `hal.sim` class entirely. `mode="sim"` without `sim_env_yaml` builds `description.hal.sim` or derives `MujocoArmHAL.from_description` when it is null + a `sim:` block exists. `mode="real"` imports `description.hal.real` and threads `transport` kwargs (real HALs take `port` / `robot_ip` / `fci_ip` and embed their own description). Both modes merge `description.hal.parameters.defaults` (ADR-0029) **underneath** the explicit `transport` so the manifest carries a robot's construction kwargs; unaccepted keys are dropped. `sim_env_yaml` + `mode="real"` → `ROSConfigError`. Missing HAL for the mode → `ROSCapabilityMismatch`; malformed/unresolvable entry → `ROSConfigError`. Routed by `deploy sim` (sim) and `deploy run` (real). (ADR-0031, ADR-0034)
  - `_import_object(path: str) -> object` [private] — Resolve a `"module.path:Attribute"` import string; raises `ROSConfigError` on malformed/unimportable/missing. **Reuse watch:** the canonical entrypoint-string importer for HAL classes — do not hand-roll `importlib` in HAL callers.
- `class MujocoArmHAL` — Generic MuJoCo-backed HAL adapter for position-controlled arms (and, via the `_per_step_update` hook, torque-controlled humanoids like the H1). (L114)
  - `read_images() -> dict[str, NDArray]` — Render the manifest's RGB `SensorSpec`s off the live MJCF, keyed by sensor `name` (issue #191 Phase 3b). Same contract `SimAttachedHAL.read_images` exposes, so `SimSensorBridge` publishes a composed-scene arm's cameras (openarm) through the shared path. Renders the MJCF camera `sim_camera_name or name` **at that sensor's own `intrinsics` resolution** — one `mujoco.Renderer` is cached per distinct `(height, width)`, so e.g. a 256×256 wrist camera alongside a 640×480 overhead publishes 256×256 (not a shared max); the published frame size always matches the sensor's camera model. A missing camera / render error is skipped with a one-shot warning (never raises). Renderers are created lazily per resolution so each EGL context binds on the caller (executor) thread. Returns `{}` when disconnected / no RGB sensors / after a renderer failure.
  - `__init__(description, *, mjcf_path, joint_qpos_addr, actuator_index, joint_qvel_addr=None, grippers=(), keyframe_index=None, seed_ctrl_from_qpos=False, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Init only; MJCF is not loaded until `connect()`. `joint_qvel_addr` defaults to `joint_qpos_addr` (correct for arms without a floating base) and is passed explicitly by humanoid HALs like `G1MujocoHAL` / `H1MujocoHAL` where the free joint shifts the qvel indices by 1. `grippers` is a sequence of `SimGripperDescription` entries; single-arm robots ship one (or none), bimanual robots (Aloha, OpenArm) ship two. (L164)
  - `_per_step_update(targets) -> None` — Hook invoked before every `mj_step` inside the settle loop. Default no-op; subclasses driving torque-mode actuators (`H1MujocoHAL`) override to recompute the actuator torque each step from the current `qpos` / `qvel`.
  - `connect() -> None` — Load MJCF, prepare `MjData` buffer. Before compiling, runs the generic camera rig (`_camera_rig.rig_cameras_into_mjcf`, ADR-0065): if the MJCF lacks a manifest RGB camera that declares a `sim_placement`, it splices the camera (+ visual-only floor + fill light) into a sibling `<name>_camrig.xml` and loads that — so a bare-arm deploy twin (so100/so101) renders its declared cameras without a scene composer. Idempotent: a scene-attached / composed MJCF that already has the cameras loads unchanged.
  - `disconnect() -> None` — Release the MuJoCo model (idempotent).
  - `read_state() -> JointState` — Joint state in description-joint order. Reads live in-process `MjData` (always current), so it **never latches `ROSPerceptionStale`**: a gap > `staleness_limit_s` since the last service means the single-threaded executor was starved (e.g. a slow camera render), not bad data — it emits a one-shot `hal.read_state.starved` WARNING and returns the live state (re-armed on the next healthy read). The prior behaviour raised *before* refreshing the clock, so one transient stall bricked the HAL permanently (the deploy-sim "Joint state is X s old" loop). Async live-feedback staleness is policed by the subscription HALs (`ros_control`/`aloha`), not here.
  - `send_action(action: Action) -> None` — Forward last waypoint to MuJoCo and step. Stamps `_last_action_ns` so the idle stepper yields to a recent command.
  - `sim_time_ns() -> int | None` — Bare-twin MuJoCo elapsed time in ns, read from live `MjData.time`; `None` before connect / after disconnect or e-stop. This is the `/clock` seam for OpenArm / SO-100 / SO-101 deploy-sim graphs, matching `SimAttachedHAL.sim_time_ns()` for scene-attached rollouts.
  - `clock_authority() -> ClockAuthority` — Return `ClockAuthority.simulation("mujoco", timestep_s=model.opt.timestep)` while connected, otherwise `ClockAuthority.host_wall()`.
  - `idle_step() -> bool` — **Sim-only** HOLD stepper that gives a bare `MujocoArmHAL` the ADR-0034 (idle cameras stay live) + ADR-0049 (joint_state published off the executor via `ProprioSnapshot` + dedicated thread) treatment the lifecycle node gates on a *callable* `idle_step`. Leaves `ctrl` untouched (it already holds the last commanded / seeded pose) and advances one `mj_step` so physics + the staleness clock refresh without moving the arm. Returns `False` (no-op) when not connected — i.e. after `estop()` disconnects — so it can never autonomously drive an e-stopped robot; `True` otherwise. **Defined on the sim-only `MujocoArmHAL` hierarchy only** (no real HAL inherits it), mirroring the same method-only real-hardware guard as `SimAttachedHAL.idle_step`.
  - **(property)** `last_action_ns -> int` — `time.monotonic_ns()` of the last `send_action` (`0` if never actuated → idle-stepping starts immediately). The `SimSensorBridge` reads it (`should_idle_step`) to yield the idle stepper to a recently-commanded skill. Mirrors `SimAttachedHAL.last_action_ns`.
  - `reset_to_pose(pose: list[float]) -> None` — Snap `qpos` to a manifest `starting_pose` and re-seed `ctrl` (instantaneous teleport; best-effort). The ADR-0053 collision-aware alternative is **not** a HAL method — the runner dispatches the `rskill-moveit-joints` rSkill to plan a collision-free MoveGroup motion to `starting_pose` (see `05-inference-runner` / `08-cli`). (L587)
  - `estop() -> None` — Zero `ctrl` and raise `ROSEStopRequested`.
  - **(classmethod)** `from_description(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> MujocoArmHAL` — Manifest-driven constructor. Reads `description.sim` and builds the HAL with the right MJCF path, qpos/qvel/actuator maps and gripper config. Removes the need for per-robot Python subclasses (ADR-0023). (L890)
  - **(staticmethod)** `_sim_kwargs_for(description, *, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5, mjcf_path_override=None) -> _MujocoArmInitKwargs` — Translate `description.sim` into the `__init__` kwarg dict.  Default 1:1 joint→qpos/actuator mapping is derived from `description.joints`, offset by 7 (qpos) / 6 (qvel) when `sim.floating_base=True`.  Used by both `from_description`, `_init_from_description`, and any caller that wants to post-process the kwargs. (L814)
  - **(instance method)** `_init_from_description(description, *, mjcf_path=None, settle_steps=None, gravity_enabled=True, staleness_limit_s=0.5) -> None` — Seam every thin per-robot subclass (UR5e/UR10e, Franka, ALOHA, OpenArm, Rizon4, G1, H1, SO-100) uses to drop the boilerplate `super().__init__(DESC, **MujocoArmHAL._sim_kwargs_for(DESC, …))` dance. Subclasses keep their typed `__init__(*, mjcf_path, settle_steps, gravity_enabled, staleness_limit_s)` signature (so IDEs still surface the four user-tunable knobs) and forward straight to here. (L946)
  - private: `_require_connected`, `_validate_action`, `_last_arm_targets`, `_apply_arm_targets`, `_apply_gripper_target`, `_read_gripper_normalised`, `_effective_actuator_index_for`

### `python/hal/src/openral_hal/_camera_rig.py`
_Generic sim camera rig (ADR-0065) — splice manifest cameras into a bare-arm MJCF for deploy sim._

- `rig_cameras_into_mjcf(xml: str, sensors: list[SensorSpec]) -> tuple[str, bool]` — For each RGB `SensorSpec` with a `sim_placement` whose camera (`sim_camera_name or name`) is absent from `xml`, splice a `<camera>` (look-at orientation via `openral_core.geometry.look_at_quat_wxyz`, `-z` MuJoCo view axis; FoV from `sim_placement.fovy_deg` or derived from `intrinsics`) into the named `parent_body` (a wrist camera) or `<worldbody>` (a world-fixed overhead), plus minimal staging — a visual-only (`contype=0 conaffinity=0`, no collisions) ground plane and an ambient fill light — when the MJCF declares none. Returns `(xml, changed)`; `changed=False` (input untouched) when no rigging is needed, so a scene-attached / already-composed MJCF passes through and the caller loads the original. Idempotent. Raises `ROSConfigError` when a sensor's `parent_body` is missing or there is no `</worldbody>` for a world camera. Called by `MujocoArmHAL.connect`.

### `python/hal/src/openral_hal/_real_description.py`
_Internal helper to derive a real-hardware ``RobotDescription`` from a sim baseline._

- `make_real_description(base, *, sdk_kind) -> RobotDescription` — `model_copy(update={"sdk_kind": sdk_kind})`; the `hal` entrypoints (`hal.sim` / `hal.real`) are inherited from *base* (ADR-0031). (L48)

### `python/hal/src/openral_hal/franka_panda.py`
_HAL adapter for the Franka Emika Panda 7-DoF arm (sim, MuJoCo)._

- `class FrankaPandaHAL(MujocoArmHAL)` — Franka Panda HAL (MuJoCo-backed). Thin manifest-driven wrapper around `MujocoArmHAL`; `__init__` forwards to `self._init_from_description(FRANKA_PANDA_DESCRIPTION, …)` (ADR-0023). (L266)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L295)
- `_panda_joint_specs() -> list[JointSpec]` (L122)
- const `FRANKA_PANDA_DESCRIPTION = RobotDescription(...)` (L176) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.franka_panda:FrankaPandaHAL"` + `hal.real="openral_hal.franka_panda_real:FrankaPandaRealHAL"` (ADR-0031). All MuJoCo wiring (MJCF URI, joint→qpos/actuator maps, gripper config) lives in `FRANKA_PANDA_DESCRIPTION.sim`. The real-HW companion `FRANKA_PANDA_REAL_DESCRIPTION` lives in `franka_panda_real.py`.

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

- `class SawyerRealHAL` — Production adapter for a physical Sawyer over `intera_sdk` / `sawyer_robot`. (L220)
  - `__init__(*, hostname='sawyer.local', controller_name='sawyer_arm_controller', joint_state_topic='/robot/joint_states', command_topic=None, estop_topic='/robot/set_super_stop', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L267)
  - `description -> RobotDescription` [@property] — Mirrors `SAWYER_DESCRIPTION`. (L303)
  - `hostname -> str` [@property] (L308)
  - `controller_name -> str` [@property] (L313)
  - `connect() -> None` (L317)
  - `disconnect() -> None` (L331)
  - `read_state() -> JointState` (L335)
  - `send_action(action) -> None` (L345)
  - `estop() -> None` (L355)
- `_sawyer_joint_specs() -> list[JointSpec]` (L112)
- const `SAWYER_DESCRIPTION = RobotDescription(...)` (L155) — sim baseline; `sdk_kind="open"`, `hal.sim=None` (no MuJoCo HAL adapter today) + `hal.real="openral_hal.sawyer_real:SawyerRealHAL"`.
- const `SAWYER_REAL_DESCRIPTION = make_real_description(SAWYER_DESCRIPTION, sdk_kind="closed_with_api")` (L195) — inherits the shared `hal`; what `robots/sawyer/robot.yaml` mirrors.

### `python/hal/src/openral_hal/panda_mobile.py`
_ADR-0025 — in-process digital-twin HAL for the `panda_mobile` embodiment (Franka 7-DoF arm on a holonomic 3-DoF base). Built by `build_hal` for the manifest-driven `ManifestHALLifecycleNode` (issue #191 Phase 3) and by tests; ROS node entrypoint in `packages/openral_hal_panda_mobile/`._

- const `PANDA_MOBILE_BASE_JOINT_NAMES: list[str]` — Base joints `[base_x, base_y, base_yaw]`, derived from `PANDA_MOBILE_DESCRIPTION.base_joints` (not hardcoded). (L100)
- const `PANDA_MOBILE_JOINT_NAMES: list[str]` — Full 11-DoF order: base (3) + arm (7, role-derived) + gripper (1, role-derived) — all from the description. (L116)
- const `PANDA_MOBILE_DESCRIPTION: RobotDescription` — Canonical RobotDescription, loaded from `robots/panda_mobile/robot.yaml` at module import. Single source of truth for joint metadata + `sim_joint_name` overrides; the arm/base/gripper name constants above derive from it via `JointSpec.role`. (L93)
- `class PandaMobileHAL` — In-process digital-twin HAL. Routes `BODY_TWIST` → planar Euler integration of (vx, vy, wz); routes `JOINT_POSITION` → 7-vec arm targets or 11-vec base+arm+gripper targets. (L132)
- _(removed: the `base_sim_joint_names` re-export wrapper — callers now import `openral_core.extract_base_sim_joint_names` directly.)_

### `python/hal/src/openral_hal/depth_cloud.py`
_ADR-0030 — reusable, robot-agnostic depth-camera → `sensor_msgs/PointCloud2` plumbing for deploy-sim HAL nodes (octomap_server source → kernel world-collision check). Pure SensorSpec adapters + the ROS msg builder; the ray-cast synth lives in `openral_sim.backends.depth_camera`._
- `is_depth_sensor(spec) -> bool` — True when `spec.modality in ("depth", "point_cloud")` **and** it carries pinhole `intrinsics` (required to back-project). (L41)
- `mjcf_camera_name(spec) -> str` — Resolves the backing MJCF `<camera>` name: `spec.metadata["mjcf_camera"]` if set (the sim camera name can differ from the ROS sensor name), else `spec.name`. (L50)
- `robot_self_body_ids(model, sim_joint_names) -> frozenset[int]` — Every MJCF body whose name shares a first-`_`-token prefix with one of the robot's `sim_joint_name`s (e.g. `mobilebase0` / `robot0` / `gripper0`). Passed as `synthesize_depth_pointcloud(exclude_body_ids=…)` so the depth cloud is self-filtered (the robot is not voxelised into its own world map). (L105)
- `depth_synth_kwargs(spec, *, max_range_default, render_size=None) -> dict` — Maps a depth `SensorSpec` to `synthesize_depth_pointcloud` kwargs (width/height/fx/fy/cx/cy + `min_range_m`/`max_range_m` from `range_min_m`/`range_max_m`, falling back to `max_range_default`). ADR-0035: when `render_size=(width, height)` is given (the scene's `observation_width/height`), the intrinsics are first rescaled via `openral_core.scale_intrinsics_to` so the ray-cast grid matches the render resolution. (L63)
- `resolve_base_body_name(model, *, description=None) -> str | None` — Resolve the MJCF body backing the robot's `base_frame`: when a `RobotDescription` is given, the first base joint's prefix + `_base` (`mobilebase0_base`); then the bare candidates `mobilebase0_base` / `base` / `robot0_base` / `base_link` — `mobilebase0_base` tried before `robot0_base` because in composed robosuite/RoboCasa scenes `robot0_base` is a placeholder mount at a fixed offset; `None` if none exist. Backs both the depth/TF base resolution (`SimSensorBridge._resolve_depth_base_body`) and the viewer free-camera fallback. (L128)
- `preferred_viewer_camera_id(model, *, prefer=("agentview","top","frontview","front")) -> int` — Pick the named MJCF camera whose vantage the viewer should open from: the first camera whose name contains a `prefer` substring (a 3rd-person workspace view — `robot0_agentview_left`, `top`, `agentview`), else the first declared camera (e.g. a wrist/eye-in-hand cam), else `-1` when the model has no cameras. Scene cameras are authored to frame the action, sidestepping the free orbit's occlusion in cluttered scenes (a base-centred orbit in a RoboCasa kitchen stares at a wall). Consumed by `initial_viewer_camera`. (L174)
- `initial_viewer_camera(*, model, data, description=None) -> tuple[tuple[float,float,float], float, float, float]` — Opening **free-camera** pose `(lookat, distance, azimuth_deg, elevation_deg)` for the viewer. The viewer always uses `mjCAMERA_FREE` so the user keeps full mouse control (drag-orbit, scroll-zoom) — this only sets the *initial* view; a `mjCAMERA_FIXED` lock would freeze those controls. When `preferred_viewer_camera_id` finds an authored camera, the eye is placed at that camera's `data.cam_xpos` with the orbit pivot on the robot base (`resolve_base_body_name`, else `model.stat.center`), so the opening view matches the authored vantage yet orbits around the robot; else delegates to `base_aligned_free_camera`. Reproduces the eye exactly via MuJoCo's `eye = lookat − distance·f`, `f = (cos el cos az, cos el sin az, sin el)`. (L341)
- `apply_robosuite_visual_geomgroups(opt, model) -> bool` — For a robosuite/RoboCasa model, set `opt.geomgroup` to hide collision shells (group 0 — RoboCasa's red kitchen / green robot capsules) and show the textured visual geoms (group 1), so `mujoco.viewer` renders textures instead of a red collision box. Gated on a robosuite signature (a `robot0_`/`gripper0_`/`mobilebase0_` body **or** an `agentview`/`frontview` camera) — **not** geom counts, since dm_control/gym scenes (gym-aloha) put visuals in group 0; returns `True` when it acted, `False` (no-op) otherwise. Used by the eval `sim run --view` viewer. (L217)
- `base_aligned_free_camera(*, model, data, base_body_name=None, azimuth_offset_deg=135.0, elevation_deg=-25.0, distance_scale=2.0, max_distance_m=3.5) -> tuple[tuple[float,float,float], float, float, float]` — **Fallback** free-camera framing `(lookat_xyz, distance, azimuth_deg, elevation_deg)` for camera-less models (single-robot twins): centres on the robot base and offsets the azimuth by the base frame's world yaw so the view aligns to the base's own axes (MuJoCo's world frame is immutable, so the viewer cannot be re-rooted onto `base_link`). `distance` is `distance_scale × model.stat.extent` capped at `max_distance_m` (a composed scene's whole-model extent would otherwise push the camera tens of metres out). Falls back to `model.stat.center` with no yaw when `base_body_name` is `None`/absent. Shared with the `openral sim run --view` eval path. (L252)
- `camera_optical_tf_to_base(*, model, data, camera_name, base_body_name) -> tuple[tuple[float,float,float], tuple[float,float,float,float]]` — Live `(translation_xyz, quat_xyzw)` of the camera optical frame (REP-103) expressed in the base body, from `data.cam_xpos`/`cam_xmat` vs the base body pose, so a node broadcasts `base_frame → <camera>_optical_frame`. Raises `ROSConfigError` if camera/body absent. (L399)
- `pointcloud2_from_points_xyz(points, *, frame_id, stamp=None) -> PointCloud2` — Packs an `(N, 3)` float32 array into an unordered (`height=1`) XYZ-float32 `sensor_msgs/PointCloud2` — the layout octomap_server's `cloud_in` expects (`sensor_msgs` imported lazily). (L451)
- `depth_image_from_grid(depth, *, frame_id, stamp=None) -> Image` — ADR-0064. Packs an `(H, W)` float32 metric-depth raster (from `synthesize_depth_image`) into a `32FC1 sensor_msgs/Image` (`step=4·W`, row-major; `0.0` = no measurement) for nvblox's projective depth integrator (`sensor_msgs` imported lazily). (L492)
- `camera_info_from_intrinsics(*, width, height, fx, fy, cx, cy, frame_id, stamp=None) -> CameraInfo` — ADR-0064. Builds a pinhole `sensor_msgs/CameraInfo` for a synthesised depth image — `K=[fx,0,cx;0,fy,cy;0,0,1]`, identity `R`, `P` mirroring `K` (no baseline), zero `plumb_bob` distortion (MuJoCo ray-cast has none). Callers pass the **stride-scaled** intrinsics so the model matches the rasterised image. (L529)

### `python/hal/src/openral_hal/aloha.py`
_HAL adapter for the Trossen ALOHA bimanual setup (issue #58) + the MuJoCo digital twin._

- `class AlohaHAL(HALBase)` — Real-hardware adapter for the 14-DoF ALOHA over the Interbotix XS SDK. (L336)
  - `__init__(*, left_arm_controller='left_arm/arm_controller', right_arm_controller='right_arm/arm_controller', left_gripper_controller='left_arm/gripper_controller', right_gripper_controller='right_arm/gripper_controller', joint_state_topic='/joint_states', estop_topic='/aloha/estop', publish_fn=None, state_fn=None, staleness_limit_s=0.2)` (L384)
  - `connect() -> None` (L415)
  - `disconnect() -> None` (L432)
  - `read_state() -> JointState` (L439)
  - `send_action(action) -> None` — Splits the 14-D action 4-ways across per-arm + per-gripper controllers. (L471)
  - `estop() -> None` (L535)
  - private: `_require_connected`
- `class AlohaMujocoHAL(MujocoArmHAL)` — MuJoCo digital twin for the 14-DoF bimanual ALOHA; thin manifest-driven wrapper around `MujocoArmHAL` (ADR-0023 bimanual amendment). All wiring lives in `ALOHA_DESCRIPTION.sim`: `gym_aloha:bimanual_viperx_transfer_cube` URI, explicit `joint_qpos_addr` / `actuator_index` (left arm 0-5, left gripper 6, right arm 8-13, right gripper 14 — skipping the negative-finger slots), two `PASSTHROUGH` grippers with `mirror_actuator_index` (positive finger + negative finger), `keyframe_index: 0` (seeds the fingers inside `ctrlrange=[0.021, 0.057]`). (L572)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(ALOHA_DESCRIPTION, …)`. (L607)
- `_aloha_joint_specs() -> list[JointSpec]` (L151)
- `_default_publish(topic, msg) -> None` (L560)
- const `ALOHA_DESCRIPTION = RobotDescription(...)` (L195) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.aloha:AlohaMujocoHAL"` + `hal.real="openral_hal.aloha:AlohaHAL"`.
- const `ALOHA_REAL_DESCRIPTION = make_real_description(ALOHA_DESCRIPTION, sdk_kind="closed_with_api")` (L307) — inherits the shared `hal`; what `robots/aloha_bimanual/robot.yaml` mirrors.

### `python/hal/src/openral_hal/ur.py`
_HAL adapters for the Universal Robots UR5e and UR10e arms (sim, MuJoCo)._

- `class UR5eHAL(MujocoArmHAL)` — UR5e HAL (MuJoCo-backed). Thin manifest-driven wrapper; `__init__` forwards to `self._init_from_description(UR5e_DESCRIPTION, …)` (ADR-0023). (L302)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L326)
- `class UR10eHAL(MujocoArmHAL)` — UR10e HAL (MuJoCo-backed). Same shape as `UR5eHAL` (ADR-0023). (L344)
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

- `class SO100FollowerHAL` — HAL adapter wrapping lerobot's SO-100 follower. (L246)
  - `__init__(port='/dev/ttyUSB0', *, calibrate_on_connect=False, max_relative_target=None, staleness_limit_s=0.5, robot=None)` (L288)
  - `connect() -> None` — Open USB serial connection. (L312)
  - `disconnect() -> None` — Close USB, disable motor torque (idempotent). (L378)
  - `read_state() -> JointState` — Joint state in radians. (L391)
  - `send_action(action: Action) -> None` — Forward one step to the SO-100 motor bus. (L419)
  - `estop() -> None` — Disconnect motors then raise. (L442)
  - `_require_connected(operation: str)`, `_obs_to_positions(obs)` [@staticmethod], `_action_to_lerobot(action)`
- `_deg_to_rad(deg) -> float` (L233)
- `_rad_to_deg(rad) -> float` (L238)
- const `SO100_DESCRIPTION = RobotDescription(...)` (L88)

### `python/hal/src/openral_hal/h1.py`
_MuJoCo digital twin for the Unitree H1 humanoid (Menagerie MJCF). Contract validator only — falls without an S0 cerebellum; gravity must be disabled in closed-loop tests (CLAUDE.md §6.2). Unlike the G1 / UR / Franka / SO-100 MJCFs, the H1 menagerie ships ``motor`` (torque) actuators, so this HAL runs a software PD position loop every physics step._

- `class H1MujocoHAL(MujocoArmHAL)` — 19-DoF humanoid HAL driving `mujoco_menagerie/unitree_h1/h1.xml`. Joint inventory: 5 leg + 5 leg + 1 torso + 4 arm + 4 arm (no wrists). Thin manifest-driven wrapper around `MujocoArmHAL` (ADR-0023); `__init__` forwards to `self._init_from_description(H1_DESCRIPTION, …)`. Inherits `connect/disconnect/read_state/estop`; overrides `_apply_arm_targets` to a no-op and `_per_step_update` to compute `tau = kp*(target - q) - kv*dq` clamped to `ctrlrange` so the public action contract stays "position targets in radians". Mirrors how `unitree_sdk2` wraps motor-level torque control in a position loop on real hardware. (L359)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L399)
  - `_per_step_update(targets) -> None` — Recomputes PD torque every `mj_step`.
  - `_apply_arm_targets(targets) -> None` — No-op (PD loop runs per-step instead).
- `_h1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `torso` / `shoulder` / `elbow`) for `joint_name`. (L209)
- `_h1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for an H1 joint. (L217)
- `_h1_joint_specs() -> list[JointSpec]` — Build the 19 `JointSpec`s from the joint-name tuples + the per-joint limit tables. (L253)
- `_h1_pd_gains() -> dict[str, tuple[float, float]]` — Per-joint `(kp, kv)` for the software PD loop (kv = 0.05*kp; kp sized so a 1-rad error roughly saturates each actuator's ctrlrange). (L350)
- const `H1_DESCRIPTION = RobotDescription(...)` (L276) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.h1:H1MujocoHAL"` + `hal.real=None` (sim-only until M2). All MuJoCo wiring (MJCF URI, floating-base joint offsets +7/+6, PD gains) lives in `H1_DESCRIPTION.sim`. Drift-guarded against `robots/h1/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/flexiv_rizon4.py`
_MuJoCo digital twin for the Flexiv Rizon 4 — 7-DoF cobot with whole-body force sensitivity (0.1 N).  Structurally identical to the UR / Franka sim HALs: position actuators, no gripper, no floating base, no PD-loop overrides — a clean `MujocoArmHAL` subclass._

- `class Rizon4MujocoHAL(MujocoArmHAL)` — 7-DoF HAL driving `mujoco_menagerie/flexiv_rizon4/flexiv_rizon4.xml` via `MujocoArmHAL`. Thin manifest-driven wrapper (ADR-0023); `__init__` forwards to `self._init_from_description(RIZON4_DESCRIPTION, …)`. (L180)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` (L210)
- `_rizon4_joint_specs() -> list[JointSpec]` — Build the 7 `JointSpec`s from the joint-name tuple + per-joint limit tables. (L111)
- const `RIZON4_DESCRIPTION = RobotDescription(...)` (L132) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.flexiv_rizon4:Rizon4MujocoHAL"` + `hal.real=None` (sim-only). All MuJoCo wiring lives in `RIZON4_DESCRIPTION.sim`. Drift-guarded against `robots/rizon4/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/openarm.py`
_MuJoCo digital twin for the Enactic OpenArm **v2** bimanual humanoid arm.  Fresh `HALBase` subclass — v2's native `<position>` actuators with per-class PD baked into the MJCF mean the HAL just writes target → ctrl and steps, no software PD loop needed._

- `class OpenArmMujocoHAL(MujocoArmHAL)` — 16-DoF (7 arm + 1 gripper per side) bimanual HAL driving `enactic/openarm_mujoco/v2/openarm_v20_bimanual.xml`; thin manifest-driven wrapper around `MujocoArmHAL` (ADR-0023 bimanual amendment). All wiring lives in `OPENARM_DESCRIPTION.sim`: `openarm_v2:bimanual` URI (fetched lazily via `ensure_openarm_v2_mjcf`), explicit `joint_qpos_addr` that skips the passive follower-finger qpos slots (8, 17), two `PASSTHROUGH` grippers (left jaw `[0, 0.7854]`, right jaw `[-0.7854, 0]`), `seed_ctrl_from_qpos: true` so the v2 `<position>` actuators hold the initial pose on the first `mj_step`. (L399)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)` — Forwards to `self._init_from_description(OPENARM_DESCRIPTION, …)`. (L434)
- `_openarm_arm_joint_specs(names, position_limits, side) -> list[JointSpec]`, `_openarm_gripper_joint_spec(name, side, position_limits) -> JointSpec`, `_openarm_joint_specs() -> list[JointSpec]` (L167, L190, L206)
- const `OPENARM_DESCRIPTION = RobotDescription(...)` (L234) — sim baseline (`name="openarm_v2"`, all 16 joints revolute matching v2's hinge gripper).  `sdk_kind="open"`, `hal.sim="openral_hal.openarm:OpenArmMujocoHAL"` + `hal.real=None` (sim-only).  Drift-guarded against `robots/openarm/robot.yaml`.

### `python/hal/src/openral_hal/_openarm_v2_assets.py`
_Vendor the upstream `enactic/openarm_mujoco` v2 MJCF until `robot_descriptions` bumps its pin past PR #19._

- `ensure_openarm_v2_mjcf() -> str` — Idempotently clones `enactic/openarm_mujoco` at a pinned v2 SHA into `$OPENRAL_CACHE_DIR/openarm_v2/<sha>/`, returns the bimanual MJCF path. Raises `ROSConfigError` when `git` is missing or the clone fails. Mirrors the pattern used by `python/sim/src/openral_sim/backends/so100_robosuite/_assets.py`. (L64)
- module const `_OPENARM_V2_PINNED_SHA: str` (L47) — bump to track upstream v2 updates.

### `python/hal/src/openral_hal/g1.py`
_MuJoCo digital twin for the Unitree G1 humanoid (Menagerie MJCF). Contract validator only — falls without an S0 cerebellum, gravity must be disabled in closed-loop tests (CLAUDE.md §6.2)._

- `class G1MujocoHAL(MujocoArmHAL)` — 29-DoF humanoid HAL driving `mujoco_menagerie/unitree_g1/g1.xml` via `MujocoArmHAL` with an explicit `joint_qvel_addr` mapping (the free joint occupies 7 qpos slots but only 6 qvel slots). Floating-base joint is implicit world state, not in `description.joints`. Thin manifest-driven wrapper (ADR-0023); `__init__` forwards to `self._init_from_description(G1_DESCRIPTION, …)`. (L347)
  - `__init__(*, mjcf_path=None, settle_steps=1, gravity_enabled=True, staleness_limit_s=0.5)`
- `_g1_group(joint_name) -> str` — Return the kinematic group token (`hip` / `knee` / `ankle` / `waist` / `shoulder` / `elbow` / `wrist`) for `joint_name`. (L205)
- `_g1_parent_child(joint_name) -> tuple[str, str]` — Return `(parent_link, child_link)` for a G1 joint, following the menagerie URDF convention. (L213)
- `_g1_joint_specs() -> list[JointSpec]` — Build the 29 `JointSpec`s from the joint-name tuples and the per-joint limit tables. (L263)
- const `G1_DESCRIPTION = RobotDescription(...)` (L291) — sim baseline; `sdk_kind="open"`, `hal.sim="openral_hal.g1:G1MujocoHAL"` + `hal.real=None` (sim-only until M2). All MuJoCo wiring (MJCF URI, floating-base joint offsets) lives in `G1_DESCRIPTION.sim`. Drift-guarded against `robots/g1/robot.yaml` by `tests/unit/test_robot_manifests_match_hal_constants.py`.

### `python/hal/src/openral_hal/so100_mujoco.py`
_MuJoCo digital twin for the SO-100 follower arm (Menagerie MJCF)._

- `class SO100MujocoHAL(MujocoArmHAL)` — SO-100 follower MuJoCo HAL, driving the `mujoco_menagerie` `trs_so_arm100/so_arm100.xml` with the same 6-DoF action layout as `SO100FollowerHAL`. Maps the lerobot-style description joint names to the Menagerie joints (`shoulder_pan→Rotation`, …, `gripper→Jaw`) and normalises the revolute Jaw range `[-0.174, 1.75]` to `[0, 1]`. Thin manifest-driven wrapper (ADR-0023); `__init__` forwards to `self._init_from_description(SO100_DESCRIPTION, …)`. (L68)
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

- `class HALLifecycleNodeBase(LifecycleNode)` — Public base class. Owns the standard `/joint_states` + `~/joint_states` publishers, the `/openral/safe_action` + `/openral/estop` subscribers (ADR-0018 F1/F5), the 1 Hz `DiagnosticsHeartbeat`, the per-tick `hal.read_state` + `hal.send_action` OTel spans, the estop latch, and the full configure → activate → deactivate → cleanup → shutdown transition wiring. The pre-ADR-0018 `~/command` (`trajectory_msgs/JointTrajectory`) subscriber + its `_on_command` callback + the `_subscriber` field were removed; `_send_action_traced` is now driven only by `_on_safe_action`. (L322)
  - `_create_hal(self) -> HAL` — **Subclass hook (required)**: construct and return a HAL instance. Reads ROS-parameter-driven constructor args via `self.get_parameter(...)`. (L380)
  - `_heartbeat_extra_fields(self) -> dict[str, str]` — Subclass hook (optional): extra key/values for the `/diagnostics` payload (e.g. `{"port": "/dev/ttyUSB0"}` for SO-100, `{"mjcf": "..."}` for OpenArm). Default: `{}`. (L392)
  - `on_configure_post_hal(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific setup after the HAL connects (e.g. opening a camera renderer on OpenArm). Default: `SUCCESS`. (L404)
  - `on_activate_post_subs(self) -> TransitionCallbackReturn` — Subclass hook (optional): robot-specific timers/publishers after the base wires its subs (e.g. the OpenArm camera-render timer). Default: `SUCCESS`. (L413)
  - `on_deactivate_pre_teardown(self) -> None` — Subclass hook (optional): stop robot-specific timers before base teardown. Default: no-op. (L421)
  - `on_cleanup_pre_disconnect(self) -> None` — Subclass hook (optional): tear down robot-specific resources (viewers, renderers) before HAL.disconnect(). Default: no-op. (L428)
  - `_publish_joint_state(self) -> None` — Timer callback. Wraps `self._hal.read_state()` in a `hal.read_state` span (identity attrs + `producer.record_joint_state`) and publishes the standard `/joint_states` + `~/joint_states` messages. Subclasses may override + call `super()._publish_joint_state()` to extend (OpenArm does this for viewer-sync). (L731)
  - `_on_safe_action(self, msg) -> None` — `/openral/safe_action` callback (ADR-0018 F1/F5). Decodes the `openral_msgs/ActionChunk` into an `openral_core.Action` and forwards through `_send_action_traced(action, source="safe_action")`. (L795)
  - `_send_action_traced(self, action, *, source) -> None` — Forward `action` to `self._hal.send_action` inside a `hal.send_action` span. The `source` attribute disambiguates the origin on the dashboard's Commands card (kept on the span so future subscriber additions can fan in without changing the span shape). (L815)
- `make_lifecycle_main(node_name, hal_factory) -> Callable[[], None]` — Build a `main()` entry point for a zero-parameter HAL adapter. Internally constructs a `_FactoryHALLifecycleNode(HALLifecycleNodeBase)` whose `_create_hal()` returns `hal_factory()`. Superseded for the standard arms by `make_lifecycle_main_from_manifest` (ADR-0032); retained for bespoke nodes. (L194)
- `class ManifestHALLifecycleNode(HALLifecycleNodeBase)` — Public generic manifest-driven lifecycle node (ADR-0032; promoted from the private `_ManifestHALLifecycleNode` under issue #191). Reads `robot_yaml` + `hal_mode` + sensor knobs as ROS params and builds its HAL via `openral_hal.build_hal`, so a robot's construction kwargs come from the manifest's `hal.parameters.defaults` (ADR-0029) — no bespoke `_create_hal` subclass. Attaches `SimSensorBridge` (cameras / depth / scan / viewer) in `on_activate_post_subs`. In `on_configure_post_hal`, **reflects** on the built HAL and opens `/openral/<robot>/reset_to_pose` (`openral_msgs/srv/ResetToPose`) iff it exposes `reset_to_pose` — generalising the openarm-only service to every `MujocoArmHAL` sim arm (ADR-0029 blocker #4, issue #191 Phase 2); HALs without the method (panda_mobile, scene-attached twins) get no service. In `_create_hal`, when a scene composition is declared (and not scene-attaching), calls the named composer and threads the composed MJCF in as the HAL's `mjcf_path` (issue #191 Phase 3b — openarm tabletop); the composition is read from the `scene_composition_json` ROS param (the DeployScene's own `composition`, ADR-0066) which **takes precedence** over the robot manifest's `scene_defaults.composition` (back-compat fallback) — so the scene owns its arena, the robot manifest describes the robot. Bare-twin camera robots (so100/so101) need no composition: their cameras are spliced by the generic camera rig at HAL connect from `sensors[].sim_placement` (ADR-0065). In `on_activate_post_subs`, when the manifest declares a planar base (`base_joints`), also attaches a `MobileBaseBridge` (`/odom` + `odom->base_link` TF + `/cmd_vel`→BODY_TWIST) — so panda_mobile runs on this node with no subclass (issue #191 Phase 3a). The per-robot lifecycle packages collapse into this node (issue #191 Phases 2-3). A back-compat alias `_ManifestHALLifecycleNode` is retained. (L913)
- `make_lifecycle_main_from_manifest(node_name) -> Callable[[], None]` — Build a `main()` that spins up `ManifestHALLifecycleNode` (ADR-0032). The node reads `robot_yaml` + `hal_mode` ("sim"|"real") ROS params and constructs its HAL via `openral_hal.build_hal(description, mode=hal_mode)` — one node class serves both modes for every robot. Used by franka / ur5e / ur10e / aloha / g1 / h1 / rizon4 / so100 / so101 (issue #191 Phase 2 migrated so100/so101 off their bespoke node); `openral deploy sim` injects `hal_mode="sim"`, `openral deploy run` (ADR-0032) injects `hal_mode="real"`. A robot lacking the requested mode raises `ROSCapabilityMismatch`. (L252)
- `decode_action_chunk(msg) -> Action | None` — Inverse of `ros_publishing_hal._flatten_action_payload`. Decodes the ADR-0028b `ActionChunk` wire shape (`flat` + `n_dof` + `horizon` + `control_mode`) back into a typed `openral_core.Action` with the per-mode payload field populated (`cartesian_delta` / `gripper` / `body_twist` / `joint_*`). Returns `None` for degenerate chunks (`flat=[]`, `n_dof≤0`) and for modes the F1/F5 publisher doesn't encode (`CARTESIAN_POSE`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT`). Used by `HALLifecycleNodeBase._on_safe_action`; lives at module scope so unit tests in `tests/unit/test_lifecycle_action_chunk_decoder.py` exercise it without a ROS 2 install. (L103)

### `python/hal/src/openral_hal/sim_bringup.py`
_Resolve a `SimScene` or `BenchmarkScene` YAML path to a live `SimRollout`. Used by `build_hal` (ADR-0034), which every manifest-driven node (incl. panda_mobile, issue #191 Phase 3) routes through._

- `build_sim_env_from_yaml(sim_env_yaml: str, *, robot_id_fallback: str | None = None) -> tuple[SimRollout, int | None]` — Load a `SimScene` or `BenchmarkScene` YAML, resolve its scene id in `openral_sim.SCENES`, and instantiate the env. Relative paths are resolved by walking parents of the source file (ROS param values are cwd-naïve). Returns `(env, seed)` — the caller plumbs the seed into `SimAttachedHAL(env_reset_seed=seed)`. Raises `ROSConfigError` when the YAML is not found, the scene id is unregistered, or schema validation fails. Robocasa scenes have `ignore_done=True` injected so deploy-sim continuous stepping does not trip the episode-done guard; strict-validation native backends (e.g. `tabletop_push`) are unchanged. (L173)

### `python/hal/src/openral_hal/sim_attached.py`
_`SimAttachedHAL` — generic HAL Protocol adapter that wraps any in-process `SimRollout` (ADR-0025 Stage 3, ADR-0034). Shared by `panda_mobile`, manifest-driven arms, and tests; not import-safe without `openral_sim` + `mujoco`._

- `ActionPacker` — Type alias `Callable[[Action, RobotDescription, int], np.ndarray]`. Per-composition translator between an OpenRAL `Action` and the env's flat action vector; the default factory is `pack_action_for_env`. Pass a custom instance to `SimAttachedHAL.__init__` for whole-body humanoid or dexterous-hand action layouts. (L170)
- `normalized_joint_index(model_joint_names: list[str]) -> dict[str, int]` — Map MJCF joint names (exact + robosuite-prefix-stripped) to model index (ADR-0034 §3.6). Exact names always win; `robot0_joint1` → `joint1` (strip `^[a-z]+[0-9]+_`) is added only when it neither shadows an exact name nor collides (bimanual `robot0_`/`robot1_` ambiguity → keep explicit). Used by `SimAttachedHAL.read_state` so one manifest serves both native MjSpec and robosuite scenes; `robot0_` never appears in a manifest. (L121)
- `is_terminated_episode_error(exc: BaseException) -> bool` — True iff `exc` is robosuite's post-terminal step guard (`ValueError("executing action in terminated episode")`, `environments/base.py`), matched by message substring (case-insensitive, stable across robosuite releases). Raw robosuite-backed adapters with `ignore_done=False` can HARD-RAISE this instead of returning a terminal `StepResult`; `SimAttachedHAL._step_and_cache` keys its raised-terminal recovery (reset + re-step) off this predicate so deploy-sim's continuous twin keeps driving. Any other `step` fault returns `False` and propagates (never silently swallowed). ADR-0036 (amended). (L103)
- `pack_action_for_env(action: Action, description: RobotDescription, env_action_dim: int) -> np.ndarray` — Default `ActionPacker`. Translates `JOINT_POSITION` (arm-only or full base+arm), `BODY_TWIST` (vx/vy/wz → slots 0-2), `CARTESIAN_DELTA` (6-vec → arm slots `[base_dim:]`), and `GRIPPER_POSITION` (→ last slot) into the env's flat action vector. Raises `ROSConfigError` for unsupported modes or mismatched row widths. (L173)
- `class SimAttachedHAL` — HAL Protocol adapter wrapping an in-process `SimRollout`. Reads live joint state via `normalized_joint_index` + `mj_name2id`; sends actions via `pack_action_for_env` (or a caller-supplied `ActionPacker`) into `env.step()`. Exposes `read_images()`, `mujoco_handles()`, `sim_time_ns()`, `base_pose`, `base_twist`, `base_pose_6dof()` for the ROS lifecycle node's camera publisher, viewer, sim-clock, and odom wiring. (L332)
  - `__init__(env: SimRollout, description: RobotDescription, *, action_packer: ActionPacker | None = None, env_reset_seed: int | None = None, env_action_dim: int | None = None, body_twist_dt_s: float = 0.05) -> None` (L361)
  - `connect() -> None` — Reset the env at `env_reset_seed`; probe `env_action_dim` (via `_probe_env_action_dim`, which raises `ROSConfigError` naming the backend when no `action_dim` is introspectable and no override was given — never a silent fallback); invalidate joint-index cache. Idempotent. (L468)
  - `disconnect() -> None` — Release env handle (idempotent). (L540)
  - `read_state() -> JointState` — Walk `description.joints`, resolve each joint via `normalized_joint_index`, read live `qpos`/`qvel` from MJCF. (L544)
  - `send_action(action: Action) -> None` — Pack action via composite-split or `ActionPacker`; call `env.step`. `BODY_TWIST` takes a direct-qpos Euler-integration path on a MuJoCo backend (`_apply_body_twist_to_qpos`, skips `env.step` so the arm doesn't churn); on a non-MuJoCo backend (Isaac kinematic base, ADR-0045) it routes through `_apply_body_twist_via_env_step` instead — the scene integrates the base inside `env.step`. Stamps `last_action_ns` at the top (the single choke point both `_on_safe_action` and `_on_cmd_vel` reach) so the idle stepper yields to it. Routes the step through `_step_and_cache`, which auto-resets on episode termination — both the *returned* terminal (`StepResult.terminated/truncated` latched as `_episode_done`) and a *raised* terminal (raw-robosuite `ignore_done=False` backends throwing `is_terminated_episode_error`); the raised path resets once and re-steps so deploy-sim never freezes with the "env.step failed: executing action in terminated episode" spam. ADR-0036 (amended). (L618)
  - `idle_step() -> bool` — **Sim-only** free-running stepper (ADR-0034 2026-06-04 amendment). Advances the wrapped `SimRollout` one tick with `np.zeros(env_action_dim)` (HOLD) so cameras keep rendering when no skill is executing — without it an idle deploy-sim scene freezes and the ADR-0035 perception bus sees a dead scene. Returns `False` (suppressed) when not connected, estop-latched, `env_action_dim is None`, or no live MuJoCo handles; else steps and returns `True`. Mirrors `send_action`'s ADR-0036 deferred-reset branch and `_last_obs` re-cache; does NOT touch `_last_env_action` / `_last_body_twist`. **Defined ONLY on `SimAttachedHAL`** — real HALs never define it; this method-only exclusion (not "zero is harmless") is the primary real-hardware guard, since a zero vector is a HOLD in sim but "drive to 0 rad" on a real position arm. (L840)
  - `read_images() -> dict[str, Any]` — Return latest rendered camera frames keyed by camera name from the cached `_last_obs`. (L1400)
  - `read_depth_clouds() -> dict[str, NDArray]` — Per-depth-sensor `(N,3)` `base_link` point clouds from `_last_obs["depth_points"]` (a non-MuJoCo backend, e.g. the Isaac scene, deprojects via `Camera.get_pointcloud` so the HAL never re-derives geometry); `{}` when the backend renders no depth. `SimSensorBridge` publishes them as `PointCloud2` for octomap. (ADR-0045)
  - `read_scan() -> NDArray | None` — The 2-D LaserScan range fan (`base_link`, `angle_min=-π`→`+π`) from `_last_obs["scan"]` when a non-MuJoCo backend ray-casts a lidar (the Isaac scene, ADR-0045); `None` when it renders no lidar. `SimSensorBridge._compute_scan_ranges` reads it for `/scan`. (ADR-0045)
  - `mujoco_handles() -> tuple[Any, Any] | None` — Forward the env's `(model, data)` MJCF handles. (L1280)
  - `sim_time_ns() -> int | None` — Cross-reset-monotonic elapsed sim time in ns (ADR-0048 Phase 1) — the seam a sim `/clock` publisher reads. Returns the wrapped `SimRollout.sim_time_ns()` (per-episode) plus an accumulated offset: `connect` and the ADR-0036 auto-resets fold each finished episode's elapsed sim-time into the offset (`_accumulate_sim_time_before_reset`) BEFORE the backend rewinds its clock, so the value is monotonic non-decreasing across `env.reset` (robocasa rewinds `MjData.time` to 0). `None` when the wrapped rollout has no sim clock (clock-less backend / sidecar) — the consumer then falls back to wall time.
  - `clock_authority() -> ClockAuthority` — Return the timestamp authority this HAL contributes to the graph: `ClockAuthority.simulation(<backend>, timestep_s=body_twist_dt_s)` when `sim_time_ns()` is live, otherwise `ClockAuthority.host_wall()` so launch keeps the graph on the host-wall authority.
  - `estop() -> None` — Latch e-stop; subsequent `send_action` calls are dropped. (L1274)
  - `base_pose -> tuple[float, float, float]` [@property] — Current `(x, y, yaw)`: from MJCF qpos on a MuJoCo backend, else from `obs["base_pose"]` the SimRollout surfaces (Isaac kinematic base, ADR-0045); `(0,0,0)` when the backend reports neither. Feeds the `/odom` publisher. (L1464)
  - `base_twist -> tuple[float, float, float, float, float, float]` [@property] — Last commanded body twist `(vx, vy, vz, wx, wy, wz)`. (L1517)
  - `_apply_body_twist_via_env_step(row: list[float]) -> None` — Non-MuJoCo `BODY_TWIST`: validate the planar twist, latch it for `/odom`, pack `(vx, vy, wz)` into the FINAL three env-action slots (the manifest scene's `[arm…, gripper, base-twist]` layout), zero the arm/gripper so a pure base move holds the arm, and `_step_and_cache`. (ADR-0045)
  - `base_pose_6dof() -> tuple[...] | None` — Full 6-DoF `(xyz, quat_xyzw)` from robocasa `raw_proprio`; falls back to `None` for non-robocasa backends. (L1527)
  - `last_action_ns -> int` [@property] — Monotonic ns of the last real action through `send_action`; `0` until the first one. The idle stepper reads it (via `should_idle_step`) to yield to an active skill. (L1390)

### `python/hal/src/openral_hal/sim_sensor_bridge.py`
_Shared sim-sensor + viewer bridge for scene-attached HAL lifecycle nodes (ADR-0034). Republishes RGB camera frames and a live MuJoCo viewer for any manifest-driven node, and runs the sim-only idle stepper. Phase 2 adds `/scan` + depth `PointCloud2`. Depth comes from the MuJoCo ray-cast (`_publish_depth_clouds`) OR, for a non-MuJoCo backend that surfaces ready `base_link` clouds in obs (the Isaac scene, ADR-0045), `_publish_depth_clouds_from_obs` — which wraps `hal.read_depth_clouds()` into a `base_link` `PointCloud2` (no ray-cast, no per-camera optical TF); `_setup_depth` creates the publishers when either source is present. rclpy imported lazily._

- `should_idle_step(now_ns: int, last_action_ns: int, idle_hold_ns: int) -> bool` — Pure predicate (no rclpy) for the sim-only idle stepper: `True` iff `now_ns - last_action_ns >= idle_hold_ns` — i.e. no real action arrived within the idle-hold window, so the stepper may HOLD-step the env. Used by `SimSensorBridge._idle_step_tick`; unit-testable in isolation. (ADR-0034 2026-06-04 amendment) (L48)
- `constant_scan_no_hit_ranges(*, n_beams: int, max_range_m: float) -> list[float]` — Pure (no rclpy) synthetic `/scan` fan: every beam clamped to `max_range_m` ("no hit everywhere"), the honest reading for an in-process digital twin with no scene to ray-cast. Used by `SimSensorBridge._compute_scan_ranges`'s no-handles branch; moved out of the panda_mobile node in issue #191 Phase 3. (`sim_sensor_bridge.py`)
- `class SimSensorBridge` — Wire and tear down sim-sensor publishers and the MuJoCo viewer on a HAL lifecycle node. Streams are gated on the robot manifest + HAL capability. Owns `/scan` for **both** paths since issue #191 Phase 3: live MJCF ray-cast when `hal.mujoco_handles()` is bound (`SimAttachedHAL`), a `constant_scan_no_hit_ranges` fan for the bare digital twin (the node no longer publishes its own scan). (L138)
  - `__init__(node: Any, hal: Any, description: RobotDescription, *, viewer_enabled: bool = True, camera_rate_hz: float = 10.0, viewer_sync_rate_hz: float = 30.0, scan_rate_hz: float = 10.0, scan_n_beams: int = 360, scan_max_range_m: float = 12.0, scan_min_range_m: float = 0.05, depth_rate_hz: float = 10.0, depth_max_range_m: float = 5.0, depth_pixel_stride: int = 4, idle_hold_ms: float = 200.0, on_step: Any = None) -> None` — `on_step` (ADR-0049): optional zero-arg callback invoked after each successful `idle_step` (the node refreshes the proprio snapshot through it, so odom/joint_state stay fresh while idle). (L155)
  - `setup() -> None` — Activate all streams the manifest + HAL support: RGB camera publishers on `/openral/cameras/<n>/image` (gated on `hasattr(hal, "read_images")` + RGB `SensorSpec`), the sim-only idle stepper (gated on `callable(getattr(hal, "idle_step", None))` + live MuJoCo handles), viewer (`mujoco.viewer.launch_passive` with `show_left_ui=False, show_right_ui=False` so only the sim renders; `_aim_viewer_camera` then sets the opening **free-camera** pose via `initial_viewer_camera` — eye at a 3rd-person scene camera, orbit pivot on the base, base-aligned default for camera-less twins — leaving the camera `mjCAMERA_FREE` so the user can orbit/zoom; GL/DISPLAY failure → warn + continue). Idempotent per activate. (L273)
  - `teardown() -> None` — Cancel timers (incl. the idle-step timer), destroy publishers, close viewer. Called from `on_deactivate` / `on_cleanup`. (L282)
- `quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]` — Pure `(x, y, z, w)` quaternion for a yaw-only rotation. Used by `MobileBaseBridge._publish_odom` for the Odometry orientation + TF; moved out of the panda_mobile node in issue #191 Phase 3. (`mobile_base_bridge.py`)
- `class MobileBaseBridge` — Generic planar-mobile-base ROS wiring (sibling of `SimSensorBridge`): owns `/odom`, the `odom->base_link` TF, and the `/cmd_vel`→BODY_TWIST bridge (ADR-0024 out-of-scope: bypasses the safety supervisor; Nav2's `velocity_smoother` caps velocity). Frame ids come from `RobotDescription.{odom_frame,base_frame}`; the HAL must expose `base_pose` (`base_pose_6dof()` / `base_twist` used when present). `ManifestHALLifecycleNode` attaches it in `on_activate_post_subs` iff the manifest declares `base_joints` — so a mobile robot needs no node subclass (issue #191 Phase 3, replaced the bespoke panda_mobile node). (`mobile_base_bridge.py`)
  - `__init__(node, hal, description, *, odom_rate_hz: float = 20.0, cmd_vel_topic: str = "/cmd_vel", proprio: Any = None) -> None` — `proprio` (ADR-0049): when set (sim-attached HALs), odom is published from the node's dedicated thread via `publish_from_snapshot`, reading the snapshot not the simulator; `None` (real HALs) keeps the legacy odom timer.
  - `setup() -> None` — Create the `/odom` publisher + TF broadcaster + `/cmd_vel` subscription; the odom timer is created **only when `proprio is None`** (ADR-0049 — sim HALs publish odom off the node's thread).
  - `publish_from_snapshot() -> None` — ADR-0049 dedicated-thread entry: publish one `/odom` + TF sample from the proprio snapshot (never the simulator). Thin alias over `_publish_odom` (which branches on `proprio`).
  - `teardown() -> None` — Cancel the timer (if any) + destroy the publisher/subscription. Idempotent.

### `python/hal/src/openral_hal/proprio_snapshot.py`

ADR-0049 — decouples the control-critical publishers (odom / joint_state / TF) from the single executor thread that runs `env.step` + render + raycast. The sim-attached HAL node captures a frame after each step (on the executor thread, where reading the sim is safe) and a dedicated publisher thread re-emits it at ~30 Hz, so odom stays fresh (~28 Hz live, vs ~1.8 Hz starved) without ever touching `MjData`/GL off the executor thread (a `MultiThreadedExecutor` was rejected — MuJoCo's GL context is thread-affine).

- `class ProprioFrame` — Frozen dataclass: one coherent proprio sample (`state: JointState`, `base_pose: (x,y,yaw)`, `base_pose_6dof: ((x,y,z),(qx,qy,qz,qw)) | None`, `base_twist: tuple[float, ...]`, `sim_time_ns: int | None` — ADR-0048 Phase 2, sim time carried for the /clock publisher). Plain immutable data only — no live simulator handles — so it is safe to publish from a different thread than the one that stepped the sim. (`proprio_snapshot.py`)
- `class ProprioSnapshot` — Lock-guarded holder for the latest `ProprioFrame`. One writer (the executor thread, after each step) calls `set`; readers (the publisher thread) call `latest`; the immutable frame is swapped under the lock so a reader never sees a torn frame and never reaches the HAL. HAL-agnostic — the node does the capture. (`proprio_snapshot.py`)
  - `set(frame: ProprioFrame) -> None` — Atomically publish `frame` as the latest sample (executor thread only). (`proprio_snapshot.py`)
  - `latest() -> ProprioFrame | None` — Return the most recent frame, or `None` before the first capture. (`proprio_snapshot.py`)
