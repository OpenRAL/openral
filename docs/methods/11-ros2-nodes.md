# ROS 2 Lifecycle Nodes (`packages/`)

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Thin wrappers around the Python-layer adapters; each exposes a single
`main()` entry point for `ros2 run`.

### `packages/openral_hal_node/openral_hal_node/lifecycle_node.py`

- `main() -> None` (L28) — The one manifest-driven HAL lifecycle node every robot runs: `openral_hal.lifecycle.make_lifecycle_main_from_manifest(node_name="openral_hal_node")`, i.e. a `ManifestHALLifecycleNode` that builds its HAL from the `robot_yaml` + `hal_mode` parameters via `openral_hal.build_hal`. Heartbeat, `/openral/safe_action` consumer and `/openral/estop` latch come from `HALLifecycleNodeBase`. `openral deploy sim|run` launches it under the ROS node NAME `openral_hal_<robot_id>` (a `__node:=` remap — not a package name). Covered by `packages/openral_hal_node/test/` (parametrised by robot manifest) and `tests/integration/test_openarm_hal_lifecycle.py`.

### `packages/world_state/openral_world_state_ros/lifecycle_node.py`

- `main() -> None` (L53) — `WorldStateAggregator` lifecycle node. Subscribes joint/policy/camera/attachment streams and publishes typed fast/slow world-state summaries; attachment updates atomically replace the prior set so multiple changes never expose a partial model. Shared with `rskill_runner_node`.
  - `_direct_image_frame_sensors() -> set[str]` — The `direct_image_frame_sensors` parameter as a set; `_on_image` skips these cameras entirely, since the co-located sensor leg's pump already writes them to the aggregator at full cadence.
- `_IDL_DIAG_TO_STR: dict[int, str]` (L904) — `DiagnosticState.diag_status` uint8 → the lowercase label (`"ok"`/`"warn"`/`"stale"`/`"error"`) `world_state_from_idl` writes onto `openral_core.WorldState`.
- `world_state_from_idl(msg) -> WorldState` (L912) — Symmetric inverse of `build_world_state_stamped_msg`: reconstructs an `openral_core.WorldState` from a `WorldStateStamped` msg. Used by the reasoner node to feed real state into its `ContextRenderer`; `image_frames` is always `None` (the IDL carries image topic refs, not inline pixels).
- `build_world_state_stamped_msg(node, world_state) -> WorldStateStamped` (L1067) — Inverse translation: a Pydantic `WorldState` → `WorldStateStamped` IDL msg. Parallel arrays are deterministically ordered (sorted by key) so two consumers comparing snapshots at the same timestamp see the same byte layout.

### `packages/openral_nav2_bringup/openral_nav2_bringup/`

- `_footprint_geometry.py` — Footprint geometry shared by this package's nodes: `convex_hull_2d` (monotone-chain convex hull) and `base_footprint_polygon` (manifest polygon, else an N-gon around the radius). The costmaps' footprint is now static, substituted at launch — no node publishes it.
- `prop SHAPE_SPHERE, SHAPE_CAPSULE, SHAPE_BOX` (_footprint_geometry.py L23–25) — Mirror `openral_msgs/AttachedCollisionPrimitive`'s shape enum as plain ints, so this pure module imports without the colcon message overlay.
- `_MIN_POLYGON_VERTICES` (_footprint_geometry.py L27) — `3`; minimum distinct-vertex count `convex_hull_2d` accepts.
- `convex_hull_2d(points) -> list[tuple[float, float]]` (_footprint_geometry.py L30) — Counter-clockwise convex hull via Andrew's monotone chain; drops collinear points, raises on fewer than `_MIN_POLYGON_VERTICES` distinct points.
- `base_footprint_polygon(description, *, circle_samples=12) -> list[tuple[float, float]]` (_footprint_geometry.py L94) — The manifest's `footprint_polygon` if declared, else a circumscribed N-gon around `footprint_radius`.
- `payload_scan_filter_node.py` — Removes a carried payload's and the robot's own silhouette from the `/scan` Nav2's costmaps read, republishing on `/openral/nav2/scan`. Every failure mode leaves more obstacles in the scan, never fewer, except a startup gate that withholds output until the self-TF first resolves.
- `DEFAULT_OUTPUT_TOPIC` (payload_scan_filter_node.py L66) — `"/openral/nav2/scan"`; where the filtered scan is published — every costmap observation source and `config/nav2_panda_mobile.yaml`'s collision monitor point here.
- `points_in_primitive(points_xyz, shape_type, shape_dimensions, transform, *, margin_m=0.0)` (payload_scan_filter_node.py L69) — The kernel's `surface_distance` containment at zero slack, vectorised over points.
- `_MIN_POLYGON_VERTICES` (payload_scan_filter_node.py L45) — Now imported from `_footprint_geometry` (formerly a local copy of the constant).
- `points_in_convex_polygon(points_xy, polygon, *, margin_m=0.0)` (payload_scan_filter_node.py L140) — Half-plane containment test that *verifies* CCW convexity rather than assuming it (a concave or CW outline raises, so the node then removes nothing).
- `filter_scan_ranges(ranges, *, angle_min, angle_increment, range_min, range_max, placements, margin_m=0.0, self_polygon=None, base_from_scan=None, self_margin_m=0.0)` (payload_scan_filter_node.py L208) — Sets removed beams to `inf` (Nav2 discards those: `inf_is_valid` defaults to `False`) and leaves readings already outside `[range_min, range_max]` alone.
- `main(args=None) -> None` (payload_scan_filter_node.py L309) — ROS 2 entry point; defines `PayloadScanFilterNode` lazily inside so the module stays importable without the colcon overlay.

### `packages/openral_nav2_bringup/launch/nav2.launch.py`

- `generate_launch_description() -> LaunchDescription` (L81) — Stand-alone bring-up of the upstream Nav2 `navigation_launch.py` include; `robot_yaml` rewrites `robot_radius`/`inflation_radius`/`motion_model` over the shared base params file (lidar or, for `slam_backend=visual`, the visual variant).
- `BOND_TIMEOUT_S` (L57) — `30.0`; how long a Nav2 lifecycle manager waits on a bond heartbeat before treating a descheduled server as dead. Not disabled (`0.0`), so a server that really dies is still caught.
- `DEFAULT_PARAMS_PATH` (L279) — Default `config/nav2_panda_mobile.yaml` path; used by `test/test_nav2_launch.py` for hermetic argument validation without spawning a real ROS 2 graph.

### `packages/openral_perception_ros/openral_perception_ros/`

- `main(args=None) -> None` (ros_image_detector_node.py L167) — Standalone ROS-image object detector: subscribes a camera topic, runs an `openral_runner` backend (RT-DETR ONNX or an open-vocab VLM), and publishes detections as a `PromptStamped`. `continuous` mode publishes every frame; `on_demand` instead serves a one-shot `locate_in_view` query.
- `DETECTOR_LOG_LEVEL_ENV` (ros_image_detector_node.py L72) — `"OPENRAL_DETECTOR_LOG_LEVEL"`; operators set this to `debug` to surface the continuous leg's per-publish DEBUG line, which the default INFO console hides.
- `_LOG_LEVEL_ALIASES` (ros_image_detector_node.py L76) — Accepted spellings → the rclpy `LoggingSeverity` member name (`"warning"` → `"WARN"`).
- `normalize_log_level(value) -> str | None` (ros_image_detector_node.py L86) — Normalises an operator-supplied log level to an rclpy severity name; case-insensitive, whitespace-trimmed. `None` for an empty/unrecognised value so a typo never silences the node.
- `classify_continuous_tick(*, error, detection_count) -> tuple[str, str]` (ros_image_detector_node.py L112) — Maps one continuous-leg detect tick to `(log_level, message)`: a `detect()` exception → `warning`, an empty result → `info` liveness heartbeat, a non-empty result → `debug`.
- `main(args=None) -> None` (scene_vlm_node.py L45) — Scene-VLM query service node: caches each camera's latest frame and serves `/openral/perception/query_scene`, a free-text answer to a question about the current frame, from a `QwenSceneVlm` backend. The reasoning counterpart of `locate_in_view`.
- `main(args=None) -> None` (reward_monitor_node.py L79) — Reward-monitor query service node: buffers the active VLA's camera frames and serves `/openral/perception/query_task_progress` from a reward backend (Robometer or TOPReward). A scoring heartbeat also drives the dashboard's progress bar and an optional Tier-C `CriticScore` publish. Advisory-only.
- `main(args=None) -> None` / `make_segmenter_node(node_name="openral_segmenter") -> Any` — Promptable-segmenter service node (`kind: segmenter` rSkill): serves `/openral/perception/segment_in_view`, turning a 3-D point into the pixel mask of the object there via an in-process SAM 2.1 model. Every failure path returns an empty `failure_reason` rather than raising.
- `main(args=None) -> None` (segmenter_node.py L591) — Entry point: init ROS, spin the segmenter node, shut down cleanly.
- `make_segmenter_node(node_name="openral_segmenter") -> Any` (segmenter_node.py L576) — Constructs the real segmenter lifecycle node behind lazy ROS imports; the seam `test_segment_in_view_service.py` uses to stand up the node in-process.
- `DEFAULT_SEGMENT_SERVICE` (segmenter_node.py L99) — `"/openral/perception/segment_in_view"`; the service name the HAL's vision attachment bridge calls by default.
- `DEFAULT_DEBUG_MASKS_TOPIC` (segmenter_node.py L105) — `"/openral/perception/masks"`; diagnostic topic the latest mask set is re-published on when `publish_debug_masks` is enabled.
- `sensor_spec_by_name(description, name) -> SensorSpec | None` (segmenter_node.py L108) — Finds a `SensorSpec` by name across `sensors` + `sensor_bundles`.
- `mono8_bytes_from_mask(mask) -> bytes` (segmenter_node.py L128) — 255 set / 0 clear; raw `mono8`, deliberately not RLE (a one-shot event call does not justify a bespoke codec).
- `resolve_camera_topics(entries, *, primary_camera, image_topic) -> dict[str, str]` (camera_topics.py L21) — The one implementation of the camera-agnostic `id=topic` → ordered map every node in this package resolves its cameras with. Insertion order matters: the first key is the primary camera. Pure and ROS-free.
- `image_to_bgr_bytes(msg) -> tuple[bytes, int, int]` (image_convert.py L14) — Converts a `sensor_msgs/Image` (`rgb8`/`bgr8`, tightly-packed rows) to contiguous H·W·3 BGR `uint8` bytes plus `(width, height)` for `ObjectsDetector.detect`; `rgb8` is reversed to BGR, `bgr8` passes through. No `cv_bridge` dependency.
- `ImageConvertError` (image_convert.py L10) — Raised by `image_to_bgr_bytes` on an unsupported encoding or a padded row stride (`step != width*3`).
- `depth_array_to_image_msg(depth_m, *, frame_id, stamp=None) -> Image` (depth_convert.py L30) — **Metric-depth message boundary** (no torch). Builds a `32FC1` `sensor_msgs/Image` in metres (NaN = no return) with tightly-packed rows.
- `image_msg_to_depth_array(msg) -> ndarray` (depth_convert.py L67) — Inverse of `depth_array_to_image_msg`.
- `camera_info_from_intrinsics(*, fx, fy, cx, cy, width, height, frame_id, stamp=None) -> CameraInfo` (depth_convert.py L89) — Builds a plain pinhole `CameraInfo` (`plumb_bob`, zero distortion, K/P matrices). Used by `depth_provider_node` to feed nvblox.
- `DepthConvertError` (depth_convert.py L26) — Raised on non-2D depth, a non-`32FC1` encoding, or a padded stride (`step != width*4`).
- `main(args=None) -> None` (depth_provider_node.py L26) — Monocular metric-depth provider: forwards each RGB frame to the DA3 depth sidecar over ZMQ and republishes the metric depth as a `32FC1` image + `CameraInfo` for `nvblox.launch.py`. Best-effort; rebuilds its socket on a timed-out reply so a slow sidecar can't wedge depth forever.

### `packages/openral_slam_bringup/openral_slam_bringup/`

- `main(args=None) -> None` (depth_height_filter_node.py L406) — Nvblox floor-exclusion prefilter: derives a robot-relative navigation-height band from the manifest's collision geometry and zeroes depth pixels outside it before nvblox integrates them. Needed because nvblox's occupancy integrator still projects raw floor returns into `/map`.
- `RobotRelativeHeightBand` (depth_height_filter_node.py L35) — Frozen dataclass: `min_z_m`/`max_z_m` navigation-height band plus `source` (`"collision_geometry"` or the `min_body_height_m` floor default).
- `quaternion_to_matrix_z_row(x, y, z, w) -> tuple[float, float, float]` (depth_height_filter_node.py L43) — The rotation matrix's third row only — enough to project a shape's local z-extent onto the world z axis.
- `derive_robot_relative_height_band(description, *, floor_clearance_m=0.10, min_body_height_m=0.30) -> RobotRelativeHeightBand` (depth_height_filter_node.py L266) — Measures the manifest's collision geometry (or falls back to `min_body_height_m`) to produce the navigation-height band.
- `filter_depth_by_global_height(depth_m, *, fx, fy, cx, cy, rotation_z_row, translation_z_m, min_height_m, max_height_m)` (depth_height_filter_node.py L316) — Zeroes out-of-band/invalid pixels; raises `ValueError` on invalid intrinsics, shape, or band.
- `main(args=None) -> None` (pycuvslam_node.py L215) — In-process cuVSLAM visual SLAM (PyCuVSLAM wheel): tracks a rectified stereo pair, or one RGB camera plus a metric-depth stream, publishes `nav_msgs/Odometry`, and broadcasts the `map → odom` TF edge. The pip-wheel alternative to the composable `isaac_ros_visual_slam` path.
- `stereo_baseline_m(right_p) -> float` (pycuvslam_node.py L43) — Raises `ValueError` unless the rectified right `P` encodes a positive baseline.
- `transform_to_pose(transform)` (pycuvslam_node.py L130) — A `geometry_msgs/Transform` → `rig_from_camera` pose.
- `compose_pose(a, b)` (pycuvslam_node.py L102) — Compose two `((qx,qy,qz,qw),(tx,ty,tz))` poses.
- `invert_pose(p)` (pycuvslam_node.py L110) — Inverse of a `((qx,qy,qz,qw),(tx,ty,tz))` pose.
- `map_from_odom(map_from_rig, odom_from_rig)` (pycuvslam_node.py L118) — Composes the tracked pose with the live `odom ← rig` TF to yield the `map → odom` edge.
- `depth_to_uint16_mm(msg, width, height, scale)` (pycuvslam_node.py L156) — A `32FC1` metres Image → the `uint16` grid cuVSLAM RGBD eats (PIL float-bilinear resize onto the RGB resolution, `metres × scale` clipped to `uint16`); raises `ValueError` on a non-`32FC1` encoding.

### `packages/openral_octomap_bridge/launch/octomap_voxel_bridge.launch.py`

- `generate_launch_description() -> LaunchDescription` (L21) — Declares params (`base_frame`, `octomap_topic`, `output_topic`, `resolution`, `coverage_radius_m`, `coverage_center_z`, `publish_rate_hz`) and spawns the `octomap_voxel_bridge` node that turns an OctoMap into the `openral_msgs/OccupancyVoxels` grid the safety kernel consumes.

### `packages/openral_hal_openarm/launch/real_bringup.launch.py`

_Bringup-only package: no HAL node of its own (OpenArm runs `openral_hal_node`); `robots/openarm/robot.yaml` reaches this file through `hal.real_bringup`._

- `generate_launch_description() -> LaunchDescription` (L54) — Real-hardware `ros2_control` bringup for the OpenArm v2: includes upstream `openarm_bringup`'s bimanual launch with this HAL's own CAN interface names. Never run alongside a deploy — two copies would double-publish `/joint_states`; activation energises the motors.
- `prop _LEFT_CAN_INTERFACE, _RIGHT_CAN_INTERFACE` (L45–46) — `"openarm_left"` / `"openarm_right"`; must match the udev names `openral_hal.openarm_real._LEFT_CAN_INTERFACE` / `_RIGHT_CAN_INTERFACE` assign to the USB CAN-FD adapter's two channels.
- `_ROBOT_CONTROLLER` (L51) — `"joint_trajectory_controller"`; must match the controller type `OpenArmRealHAL` publishes `JointTrajectory` messages to, not `forward_position_controller` (which takes `Float64MultiArray`).

### `packages/openral_slam_bringup/launch/`

- `generate_launch_description() -> LaunchDescription` (slam_toolbox.launch.py L40) — Stand-alone bring-up of upstream `slam_toolbox/async_slam_toolbox_node` as a `LifecycleNode` (auto-transitions only to INACTIVE; the Reasoner promotes to ACTIVE via `LifecycleTransitionTool`). Composed into `deploy_e2e.launch.py` when `enable_slam:=true`.
- `_SLAM_NODE_NAME` (slam_toolbox.launch.py L26) — The `async_slam_toolbox_node` node name.
- `prop DEFAULT_PARAMS_PATH, NODE_NAME` (slam_toolbox.launch.py L93–94) — Test-hermeticity re-exports used by `tests/test_slam_toolbox_launch.py` without spawning a real ROS 2 graph.
- `generate_launch_description() -> LaunchDescription` (pycuvslam.launch.py L36) — Stand-alone bring-up of the in-process PyCuVSLAM stereo/mono-RGBD node (a plain `Node`, not a `ComposableNodeContainer`); the pip-wheel alternative to `cuvslam.launch.py`'s NITROS path.
- `prop _NODE_NAME, _PACKAGE, _EXECUTABLE` (pycuvslam.launch.py L26–28) — The `pycuvslam_node` node/package/executable names.
- `prop DEFAULT_PARAMS_PATH, NODE_NAME, PACKAGE, EXECUTABLE` (pycuvslam.launch.py L134–137) — Test-hermeticity re-exports used by `test/test_pycuvslam_launch.py` without spawning a real ROS 2 graph or the cuVSLAM wheel.
- `generate_launch_description() -> LaunchDescription` (nvblox.launch.py L46) — Stand-alone bring-up of the upstream `nvblox::NvbloxNode` composable node inside a `ComposableNodeContainer`, remapping its `~/static_occupancy_grid` onto the backend-agnostic `/map` for lidar-less Nav2.
- `prop _NVBLOX_NODE_NAME, _NVBLOX_CONTAINER_NAME, _NVBLOX_PACKAGE, _NVBLOX_PLUGIN` (nvblox.launch.py L33–38) — Node/container name and the Isaac ROS composable-node package+plugin id for `NvbloxNode`.
- `prop DEFAULT_PARAMS_PATH, NODE_NAME, PACKAGE, PLUGIN` (nvblox.launch.py L181–184) — Test-hermeticity re-exports used by `test/test_nvblox_launch.py` without spawning a real ROS 2 graph or the nvblox engine.
- `generate_launch_description() -> LaunchDescription` (cuvslam.launch.py L49) — Stand-alone bring-up of the upstream Isaac ROS `VisualSlamNode` (cuVSLAM) composable node inside a `ComposableNodeContainer` for NITROS-zero-copy image topics.
- `prop _VSLAM_NODE_NAME, _VSLAM_CONTAINER_NAME, _VSLAM_PACKAGE, _VSLAM_PLUGIN` (cuvslam.launch.py L36–41) — Node/container name and the Isaac ROS composable-node package+plugin id for `VisualSlamNode`.
- `prop DEFAULT_PARAMS_PATH, NODE_NAME, PACKAGE, PLUGIN` (cuvslam.launch.py L139–142) — Test-hermeticity re-exports used by `test/test_cuvslam_launch.py` without spawning a real ROS 2 graph or the cuVSLAM engine.

### `packages/openral_human_estop/openral_human_estop/forwarder_node.py`

- `class HumanEstopForwarderNode(LifecycleNode)` (L24) — Subscribes `/openral/human_estop` and republishes onto `/openral/estop` plus a `FailureTrigger` on `/openral/failure/safety`. No in-repo node currently publishes `/openral/human_estop` — the dashboard stop button publishes `/openral/estop` directly.
  - `on_configure(state) -> TransitionCallbackReturn` (L42) — Opens the estop publisher, the `FailureTrigger` publisher, and the `/openral/human_estop` subscription.
  - `on_activate(state) -> TransitionCallbackReturn` (L68) — No additional resources to start.
  - `on_deactivate(state) -> TransitionCallbackReturn` (L73) — No additional resources to stop.
  - `on_cleanup(state) -> TransitionCallbackReturn` (L78) — Releases the subscription + publishers.
  - `on_shutdown(state) -> TransitionCallbackReturn` (L92) — Forces cleanup (delegates to `on_cleanup`).
- `main(args=None) -> int` (L122) — Entry point for `ros2 run openral_human_estop forwarder_node`.

### `packages/openral_safety_watchdog/openral_safety_watchdog/`

- `estop_qos() -> QoSProfile` (_qos.py L19) — QoS for `/openral/estop`: `RELIABLE` / `VOLATILE` / `KEEP_LAST=10`.
- `failure_qos() -> QoSProfile` (_qos.py L30) — QoS for `/openral/failure/safety`: `RELIABLE` / `VOLATILE` / `KEEP_LAST=50`. Both factories are the single shared definition both nodes' `on_configure` calls, keeping their QoS identical by construction.
- `class DeadmanWatchdogNode(LifecycleNode)` (deadman_watchdog_node.py L96) — Fires `/openral/estop` plus a timeout `FailureTrigger` when `/openral/safe_action` dies, in its own process so a kernel crash still brakes. `arm_status_topic` gates the deadline to an active goal; the post-estop latch releases only on a genuine `SafetyStatus` transition to clear.
- `prop DEFAULT_SAFE_ACTION_DEADLINE_S, DEFAULT_CHECK_PERIOD_S, DEFAULT_FIRST_CHUNK_DEADLINE_S, DEFAULT_SAFETY_STATUS_TOPIC, SAFETY_STATUS_LIVENESS_S` (deadman_watchdog_node.py L49–64) — Defaults for the safe-action deadline (0.2 s), the deadline-check timer period (50 ms), the goal-accepted→first-chunk grace (60 s), the latched `SafetyStatus` topic, and the HZ-0096-1 liveness window (3 s) past which a `SafetyStatus` sample is treated as unknown-not-safe.
- `_LIVE_GOAL_STATUS_NAMES` (deadman_watchdog_node.py L73) — `("STATUS_ACCEPTED", "STATUS_EXECUTING", "STATUS_CANCELING")`; resolved to values off the generated message class rather than hardcoded, so an upstream renumbering cannot silently widen or narrow the gate.
  - `DeadmanWatchdogNode.on_configure(state) -> TransitionCallbackReturn` (deadman_watchdog_node.py L162) — Opens the `/openral/safe_action` subscription, the estop/failure publishers (QoS from `_qos.estop_qos`/`_qos.failure_qos`), and (if `arm_status_topic` is set) the goal-status subscription.
  - `DeadmanWatchdogNode.on_activate(state) -> TransitionCallbackReturn` (deadman_watchdog_node.py L228) — Arms the deadline check; opens the execution window immediately when free-running, else waits for the first live goal status.
  - `DeadmanWatchdogNode.on_deactivate(state) -> TransitionCallbackReturn` (deadman_watchdog_node.py L242) — Disarms — deadline checks become no-ops until the next activate.
  - `DeadmanWatchdogNode.on_cleanup(state) -> TransitionCallbackReturn` (deadman_watchdog_node.py L248) — Releases the timer, subscriptions and publishers.
  - `DeadmanWatchdogNode.on_shutdown(state) -> TransitionCallbackReturn` (deadman_watchdog_node.py L266) — Forces cleanup (delegates to `on_cleanup`).
- `main(args=None) -> int` (deadman_watchdog_node.py L396) — Entry point for `ros2 run openral_safety_watchdog deadman_watchdog_node`.
- `class HardwareEstopNode(LifecycleNode)` (hardware_estop_node.py L48) — Polls a hardware E-stop source and publishes `/openral/estop` plus a `FailureTrigger` on the rising edge. With no device declared, `on_configure` refuses to activate rather than coming up ACTIVE over a stub; a read that raises latches as unknown-therefore-unsafe.
- `DEFAULT_POLL_RATE_HZ` (hardware_estop_node.py L44) — `100.0`; how often to query the hardware estop state.
  - `HardwareEstopNode.on_configure(state) -> TransitionCallbackReturn` (hardware_estop_node.py L97) — Opens publishers and starts the polling timer only if a device is really there; returns `FAILURE` (leaving the node UNCONFIGURED) when this host has no hardware E-stop this node could read.
  - `HardwareEstopNode.on_activate(state) -> TransitionCallbackReturn` (hardware_estop_node.py L130) — Starts braking; a button already held at activation brakes on the first poll.
  - `HardwareEstopNode.on_deactivate(state) -> TransitionCallbackReturn` (hardware_estop_node.py L138) — Stops braking; the timer keeps running but `_poll` is a no-op.
  - `HardwareEstopNode.on_cleanup(state) -> TransitionCallbackReturn` (hardware_estop_node.py L144) — Releases the timer + publishers.
  - `HardwareEstopNode.on_shutdown(state) -> TransitionCallbackReturn` (hardware_estop_node.py L158) — Forces cleanup (delegates to `on_cleanup`).
- `main(args=None) -> int` (hardware_estop_node.py L261) — Entry point for `ros2 run openral_safety_watchdog hardware_estop_node`.

### `packages/openral_rskill_ros/openral_rskill_ros/`

- `RskillRunnerNode(*, node_name="openral_skill_runner", robot_description, aggregator, skill_resolver=None)` (rskill_runner_node.py L222) — The `ExecuteRskill` action server: a lifecycle node dispatching to the resolved rSkill at 30 Hz, aborting any blocking wait the instant `/openral/estop` or a latched `/openral/safety_status` reports unsafe. Every terminal outcome stamps both a prose `failure_reason` and a typed `failure_kind`.
  - `_failure_kind_for_exception(exc) -> int` (module-level) — Maps the OpenRAL exception hierarchy onto the `ExecuteRskill.Result.FAILURE_*` uint8 constants, most-specific-first. A `ROSSafetyViolation` other than `ROSEStopRequested` has no kind by design — it is re-raised to the safety supervisor, never folded into a Result.
  - `_classify_runtime_failure(exc) -> tuple[str, int]` (staticmethod) — One message probe yielding both the typed `failure_reason` and the matching `failure_kind` for a raw non-`ROSError` escape, so the string and the uint8 can never disagree. `_label_runtime_failure(exc) -> str` is the thin projection onto the first element.
  - `_apply_starting_pose(skill, goal_handle) -> tuple[int, str] | None` / `_interpolate_starting_pose(pose, goal_handle) -> tuple[int, str] | None` / `_dispatch_moveit_approach(pose, goal_handle) -> tuple[int, str] | None` — Reach a manifest `starting_pose` before inference via either a configured MoveIt plan or a bounded joint ramp, checking cancellation between waypoints and failing the goal on any error.
  - `make_default_skill_resolver(ros_node, *, search_paths=(), scene_cameras=()) -> SkillResolver` (rskill_runner_node.py L2078) — Production resolver factory: routes `kind: vla` to the local/HF-Hub path and `ros_action`/`ros_service` skills to `ROSActionRskill`. The Hub fallback `_default_skill_resolver(..., scene_cameras=(), tf_lookup=None)` now **binds** `rSkill.from_pretrained`'s packaging handle to a runtime skill via `_build_runtime_skill_from_manifest(handle.local_dir / "rskill.yaml", …)` — until 2026-09-22 it returned the handle raw, so every Hub-hosted VLA aborted at the embodiment gate (which now raises a typed `ROSConfigError` naming a non-`rSkillBase` return instead of a bare AttributeError with an empty `failure_reason`). **Preload:** parameters `preload_rskill_id` / `preload_rskill_revision` / `preload_prompt` make `on_activate` start a daemon worker (`_preload_resident_skill`) that runs `_acquire_skill` under `_execute_serial` so the skill is GPU-resident before any goal exists; `_goal_cb` REJECTS goals while `_preload_in_flight` is set (acceptance is what arms the deadman's first-chunk window), and `_acquire_skill` logs `rskill_runner.resident_key_mismatch` when a goal's (id, revision, prompt) differs from the resident one, since that mismatch costs the full cold load inside the watchdog window.
  - `make_local_skill_resolver(search_paths, *, scene_cameras=()) -> SkillResolver` (rskill_runner_node.py L2228) — In-tree VLA resolver. Walks each search path once and indexes every `*/rskill.yaml` by manifest name; on resolve calls `_build_runtime_skill_from_manifest`. Accepts `ros_node=None` kwarg for signature uniformity with the default resolver.
- `_warmup_observation()` / shim `on_warmup` (rskill_runner_node.py) — Warm-up through the real tick path: the policy shim's `on_warmup` runs `adapter.step` on `_warmup_observation()` — zeros shaped like the cell's RGB intrinsics for every required camera slot, `state_contract.dim` zeros, the goal prompt — then `adapter.reset()`, instead of a bare `warm_up_lerobot_policy` forward. The bare forward left the first real tick paying 306–357 s on a Jetson AGX Orin under a live graph (chunk executor thread, autocast, preprocessor on real-sized frames each have first-call costs) against the deadman's 120 s first-chunk window; a warm second dispatch in the same process took 426 ms, which is what the preload now pays.
- `_clamp_joint_position_slice(values, joint_names, description) -> list[float]` (rskill_runner_node.py L2766) — Pulls each JOINT_POSITION slot target strictly inside `RobotDescription.joints[].position_limits` (the single-surface path's 1e-3 epsilon; the kernel checks open intervals) before `_dispatch_slots` pads and emits it. Slot dispatch used to clip only to `input_bounds` and propose the rest raw — the OpenArm restock π0.5's first tick put left_joint5 0.019 rad past its limit and the kernel E-stopped the cell (qorin1, 2026-09-22). The kernel stays the authority; this only stops proposing what it will certainly refuse.
- `prop _CANCEL_DRAIN_S, _MAX_APPROACH_WAYPOINTS, _DEFAULT_EXECUTION_DEADLINE_S` (rskill_runner_node.py L80–91) — The 100 ms cancel drain; the runaway guard on the MoveIt approach replay waypoint count; and the global fallback (45 s) for a goal's wall-clock budget when neither the dispatch nor the skill manifest declares one.
- `prop _SAFETY_STATUS_TOPIC, _SAFETY_STATUS_LIVENESS_S` (rskill_runner_node.py L95–105) — ADR-0096's latched safety-state topic (`/openral/safety_status`) both the C++ kernel and `SafetyPassthroughNode` publish on, and the HZ-0096-1 liveness window (3 s) past which a stale `SafetyStatus` is reported as an abort reason rather than downgraded to clear.
- `main(args=None) -> int` (rskill_runner_node.py L3210) — Entry point for `ros2 run openral_rskill_ros rskill_runner_node`; bootstraps a standalone node with a fresh `RobotDescription` and `WorldStateAggregator`. Production launches use `compose_so100_runtime` instead, to share the aggregator with the colocated `world_state_node`.
- `compose_runtime(robot_yaml, *, skill_resolver=None, enable_world_cloud_bridge=False, world_cloud_topic="", slam_source_node="", dataset_out=None, image_staleness_limit_s=None, deploy_sensors=(), …) -> ComposedRuntime` (compose.py L81) — Loads a robot manifest and builds one `WorldStateAggregator` shared by a colocated world-state node and `RskillRunnerNode`, always attaching a `SlamMapBridge` so any `/map` publisher populates the dashboard. Cameras get their own freshness window, independent of joint/EE staleness. `deploy_sensors` (a `DeployScene.sensors` block) is merged into the manifest's sensors via `merge_deploy_sensors` before the aggregator and runner are built, so a scene's per-cell `vla_feature_key` reaches the runner's camera slots, the dataset recorder and world state alike — until 2026-09-23 only the sensor readers saw it and the OpenArm restock π0.5 ran with its `context` (ZED head) view replaced by lerobot's masked blank.
- `start_world_state_executor(world_state_node) -> Callable[[], None] | None` (compose.py) — spins the world-state node on rclpy's C++ `rclpy.experimental.EventsExecutor` in a daemon thread and returns a stop callable (`shutdown` + join), or `None` on an rclpy without it (pre-Jazzy 7.1) so the caller adds the node to its own executor. Why: the Python `MultiThreadedExecutor` rebuilds its wait set in Python on every wake-up over every entity of every node it holds; with world_state's ~40 topics on it that loop held 64 % of the deploy runtime's GIL on an AGX Orin and the in-process π0.5 inference thread got <2 % (1.6 s idle forward → ~300 s live). The skill runner stays on the multi-threaded executor, whose reentrant group keeps `execute_rskill` accept/cancel/result live while a goal blocks a worker.
- `compose_so100_runtime(*, skill_resolver=None) -> ComposedRuntime` (compose.py L277) — SO-100 convenience wrapper over `compose_runtime`.
- `ComposedRuntime` (compose.py L39) — Dataclass returned by `compose_runtime`: `description`, `aggregator`, `world_state_node`, `skill_runner_node`, `slam_bridge`, `world_cloud_bridge`, `dataset_recorder_bridge` fields.
- `joint_names_from_goal_json(default_goal_json) -> list[str]` (_starting_pose.py L10) — Extracts the planning-group joint names from a MoveGroup `default_goal_json`'s `joint.joint_names` block; used to length-check a robot's flat `starting_pose` before building the retarget override. Raises `ValueError` on malformed JSON or a missing/empty `joint.joint_names`.
- `moveit_joint_goal_override(joint_names, positions) -> str` (_starting_pose.py L47) — Builds the `goal_params_json` that retargets the approach manifest's `joint.positions` to `positions` (deep-merge over the manifest, so `joint_names`'s order is preserved); `joint_names` is used only to length-check `positions`.
- `SensorLeg` (sensor_leg.py L197) — Real-mode camera leg for `openral deploy run`, the real-hardware counterpart of the sim HAL's `SimSensorBridge`. Dataclass holding readers + publishers + which sensors write directly to the aggregator without a duplicate ROS-topic path.
  - `SensorLeg.start() -> None` (sensor_leg.py L217) — Starts every prepared publisher after ROS lifecycle configuration; entity creation and lifecycle transitions remain single-threaded until this point.
  - `SensorLeg.close() -> None` (sensor_leg.py L230) — Stops publishers first, then closes readers, idempotently and exception-safe (teardown must always reach the HAL shutdown behind it).
- `DEFAULT_TOPIC_PREFIX` (sensor_leg.py L57) — `"/openral/cameras"`; WorldState's camera subscription prefix (`<prefix>/<name>/image`).
- `_DEFAULT_PUBLISH_RATE_HZ` (sensor_leg.py L61) — `10.0`; publish cadence when the binding's `backend_params` carry no fps. Matches the `WorldStateAggregator` staleness-gate expectation.
- `_RGB_CHANNELS` (sensor_leg.py L65) — `3`; channel count of an interleaved colour frame — the only layout the 180° dashboard flip knows how to reshape.
- `_MAX_FALLBACK_TOPIC_RATE_HZ` (sensor_leg.py L85) — Cap on the ROS topic cadence for the Python fallback publisher only (the native GStreamer tee is uncapped).
- `_DEFAULT_TOPIC_MAX_SIZE` (sensor_leg.py L94) — `(320, 240)`; resolution ceiling `topic_frame_size` falls back to.
- `_DEFAULT_SLAM_STEREO_CAMERAS` (sensor_leg.py L295) — `("left", "right")`; the implicit stereo pair a `None` `slam_stereo_cameras` field resolves to.
- `slam_camera_names(runtime) -> frozenset[str]` (sensor_leg.py L298) — Camera names feeding visual SLAM, which must never be rate-capped (cuVSLAM / PyCuVSLAM lose the track on a starved stream). Empty when SLAM is off, in which case the fallback-rate cap applies to every camera.
- `open_deploy_sensor_readers(sensors, *, topic_prefix="/openral/cameras", aggregator=None, ros_node=None, uncapped_sensors=(), topic_max_size=None) -> SensorLeg` (sensor_leg.py L500) — Builds the complete deploy-bound sensor batch; every camera publishes on `<topic_prefix>/<name>/image` — `gstreamer` backends via an in-pipeline ROS tee, others via a polling publisher. A shared `aggregator` also gets a pump writing frames directly into `WorldStateAggregator`.
- `_AggregatorPump` (sensor_leg.py L121) — Daemon-thread pump polling `read_latest` at the spec rate into the aggregator, deduplicating on the frame's monotonic stamp. Emits the per-frame observability span after each write, guarded so a failing display path can never starve the policy.
- `_emit_frame_observability(sensor_name, frame, flip_180) -> None` (sensor_leg.py L97) — Emits the dashboard's per-frame observability span for a pump-fed camera; exists because the ROS tee these cameras skip is what the dashboard would otherwise read from.
- `_publish_rate_hz(spec) -> float` (sensor_leg.py L280) — Binding fps, else spec `rate_hz`, else 10 Hz. This is the **capture/native-tee** cadence and is not capped.
- `_fallback_topic_rate_hz(spec) -> float` (sensor_leg.py L420) — Caps the ROS topic cadence for the Python fallback publisher only (3 Hz), since every tick is a GIL-held full-resolution conversion; the native GStreamer tee and the policy's in-process read stay at full rate.
- `topic_frame_size(runtime) -> tuple[int, int] | None` (sensor_leg.py L388) — Resolution ceiling for the fallback topic; `None` (native pixels) when the object detector or SLAM is on, since both need full resolution. The policy is unaffected either way, reading the aggregator in-process at capture resolution.
- `apply_launch_overrides(runtime, *, enable_object_detector=None, enable_slam=None, slam_stereo_cameras=None, slam_mono_camera=None) -> object | None` (sensor_leg.py L332) — Folds the launch's resolved auto-detected flags over the scene's raw runtime block before camera rate-capping decisions are made, since only the CLI resolves the scene's tri-state `auto` settings.
- `merge_deploy_sensors(manifest_sensors, scene_sensors) -> list[SensorSpec]` (sensor_leg.py L251) — Robot-manifest ∪ scene sensors, merged field-wise on a name collision (the scene's explicitly-set fields win). Wholesale replacement would silently discard a manifest mount whenever a scene supplied only a device path.

### `packages/openral_rskill_ros/scripts/runtime_node`

_Composed-runtime entry point installed as `lib/openral_rskill_ros/runtime_node`._ Reads the `robot_yaml` parameter, calls `compose_runtime`, and spins both lifecycle nodes on a `MultiThreadedExecutor`; for a real deploy it also opens every deploy-bound sensor reader before spinning. The script eagerly initializes GStreamer before any numpy/pydantic/rclpy import, since a later `rclpy.Node()` can segfault otherwise. Spins the world-state node on the C++ `EventsExecutor` (`start_world_state_executor`; falls back to sharing the multi-threaded executor with a stderr notice) and the skill runner on a 4-thread executor. Param `joint_states_topic` (from `DeployRuntime.joint_states_topic`) is set on **both** composed nodes before configure: world_state's ingest and the runner's `ROSPublishingHAL` joint-state cache (`RskillRunnerNode` param `joint_states_topic`, `""` = `/joint_states`) share the one topic, so a ros2_control arm's 750 Hz broadcaster stream never wakes the Python executors. With a `deploy_config`, it loads the scene BEFORE `compose_runtime` and passes its `sensors:` as `deploy_sensors`; the sensor leg then reads readers off that merged description.

- `_prewarm_vla_imports() -> None` — Imports `torch` + `lerobot.policies.factory` early so the shared bulk of any VLA adapter's import cost is paid before any camera-reader thread exists. Must run ahead of `open_deploy_sensor_readers`, since a `transformers` import can otherwise starve a reader thread of the GIL for minutes.

### `packages/openral_rskill_ros/launch/deploy_e2e.launch.py`

- `compose_runtime_graph(context, *_args, **_kwargs) -> list` (L814) — Resolves every launch arg, loads the robot manifest, and assembles the full deploy-sim ROS graph — HAL, safety kernel, reasoner, SLAM/Nav2, sensor drivers, optional Foxglove viz. On `hal_mode:=real` it also starts the robot's vendor `ros2_control` bringup itself.
- `generate_launch_description() -> LaunchDescription` (L2564) — Robot-agnostic deploy-sim launch graph entry point; wraps `compose_runtime_graph` in an `OpaqueFunction`.
- `_build_real_bringup_include(real_bringup) -> object | None` (L516) — `IncludeLaunchDescription` of the manifest's `hal.real_bringup` (`"<pkg>:<file>.launch.py"`) on `hal_mode:=real`; `None` when the manifest declares none; raises `RuntimeError` when the declared package or file is not installed. There is no package-name convention fallback.
- `_VENV_SITE` (L42) — Optional workspace-editable-install site-dir from `OPENRAL_VENV_SITE`, registered via `site.addsitedir` (plain `PYTHONPATH` is not enough: `.pth` files are only processed by the `site` module on registered site-dirs).
- `_REPO_ROOT` (L112) — Resolved repo root (`_resolve_repo_root()`).
- `_RSKILLS_DIR` (L113) — `str(_REPO_ROOT / "rskills")`.
- `_VENV_RAL` (L115) — `_REPO_ROOT / ".venv" / "bin" / "openral"`.
- `_RAL_EXECUTABLE` (L116) — The workspace venv's `openral` binary when it exists, else the bare `"openral"` on PATH.

### `python/runner/src/openral_runner/ros_publishing_hal.py`
_HAL Protocol adapter that publishes `ActionChunk` on `/openral/candidate_action`._

- `ROSPublishingHAL(*, node, description, skill_id_getter=..., skill_revision_getter=..., tick_index_getter=lambda: 0, joint_state_topic="/joint_states", candidate_action_topic="/openral/candidate_action", action_applied_topic="/openral/action_applied", safety_abort_getter=lambda: None, action_applied_timeout_s=5.0, joint_state_staleness_limit_s=0.5)` (L117) — Publishes typed candidate action chunks and blocks with bounded backpressure until the HAL acknowledges application, polling the injected safety latch every 50 ms and raising `ROSEStopRequested` instead of hiding a stop behind a plain timeout.
- `_row_major_flatten(rows) -> list[float]` (L57) — Private helper used by `_action_to_chunk`; preserves row-major ordering for the chunk `flat` array.
- `_CONTROL_MODE_TO_UINT8: dict[ControlMode, int]` (L114) — Stable mapping from `openral_core.ControlMode` to the uint8 slot in `ActionChunk.control_mode`.
- `ROSPublishingHAL.connect() -> None` (L212) — Opens the publisher on `candidate_action` + subscriber on `joint_states`.
- `ROSPublishingHAL.disconnect() -> None` (L265) — Closes the publisher + subscription. Idempotent.
- `ROSPublishingHAL.read_state() -> JointState` (L282) — Returns the latest cached `JointState` from `/joint_states`; raises `ROSRuntimeError` if called before `connect()` or before any message has arrived.
- `ROSPublishingHAL.send_action(action) -> None` (L298) — Publishes an `ActionChunk` on `/openral/candidate_action`. Does NOT touch motors — the safety node republishes the chunk on `/openral/safe_action`, which the per-robot HAL lifecycle node consumes and forwards to the underlying controller.
- `ROSPublishingHAL.begin_goal() -> None` (L313) — Resets goal-local partial-group state.
- `ROSPublishingHAL.estop() -> None` (L388) — Triggers an emergency stop by raising `ROSEStopRequested` so the safety supervisor boundary can record and brake; the owning lifecycle node also publishes on `/openral/estop` as defense-in-depth.
- `_SAFETY_ABORT_POLL_S` (L54) — `0.05`; how often a blocked apply-wait polls the injected safety latch — a latched safety stop produces silence, nothing to wake the wait on, so this bounds how long the abort stays invisible.

### `python/runner/src/openral_runner/slam_bridge.py`
_rclpy → OTLP bridge for slam_toolbox `/map`. Throttles to 1 Hz, rasterises the `nav_msgs/OccupancyGrid` to a base64 PNG, looks up the robot's map-frame pose via tf2, and emits one `slam.occupancy_grid` OTel span the dashboard's SLAM Map card renders (store handler in `openral_observability.dashboard.store`)._

- `SLAM_MAP_TOPIC_DEFAULT = "/map"` (L43) — Default `nav_msgs/OccupancyGrid` topic slam_toolbox publishes on.
- `encode_occupancy_grid_png(*, width: int, height: int, data: list[int]) -> str` (L70) — Pure function rendering an `OccupancyGrid.data` array as a base64 PNG (unknown→mid-grey, free→white, occupied→black, in-between linear ramp; flipped so map-north points up). Raises `ValueError` if `len(data) != width * height`. Exercised directly by tests against synthetic grids.
- `robot_pose_from_transform(*, translation_xyz: tuple[float, float, float], rotation_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float]` (L128) — Planar `(x, y, yaw)` from a tf2 transform's translation + rotation (`z` ignored); delegates yaw to `openral_core.geometry.quat_xyzw_to_yaw`. Used by `SlamMapBridge` to project the `map→base_frame` lookup into the span attributes.
- `class SlamMapBridge` (L159) — `rclpy.node.Node`-hosted subscription on `/map`; on each accepted callback rasterises the grid, looks up the robot pose, and emits a `slam.occupancy_grid` span the dashboard renders. Degrades gracefully (no robot marker) when TF is unavailable.
  - `__init__(node, *, topic=SLAM_MAP_TOPIC_DEFAULT, base_frame="base_link", footprint_radius_m=None, footprint_polygon=None, source_node_name="openral_slam_toolbox", publish_interval_s=1.0, max_cells=4_000_000)` (L205) — Subscribes with slam_toolbox's QoS; `footprint_polygon` (base-frame XY) lets the dashboard draw the true oriented outline instead of falling back to the `footprint_radius_m` circle.
  - `destroy() -> None` (L277) — Release the ROS subscription. Safe to call multiple times.
- `_DEFAULT_PUBLISH_INTERVAL_S` (L49) — `1.0`; throttle so a busy SLAM run doesn't flood the OTLP pipeline (the dashboard re-renders on every span).
- `_UNKNOWN_GREY` (L58) — `128`; the 8-bit greyscale value an unknown (`-1`) occupancy cell renders as.
- `prop _OCC_FREE_THRESHOLD, _OCC_OCCUPIED_THRESHOLD` (L61–62) — `0` / `100`; the `nav_msgs/OccupancyGrid` probability bounds free/occupied map to (white/black), with values in between scaled linearly.
- `prop _OVERSIZE_WARN_BURST, _OVERSIZE_WARN_RATE` (L66–67) — `3` / `100`; how many oversize-grid warnings are emitted verbosely before falling back to a 1-in-100 rate to keep log noise bounded.

### `packages/openral_foxglove_bringup/`
_Read-only Foxglove live-scene surface, hybrid with the OTel dashboard which keeps traces, metrics, and every write path. Provides the bridge + republishers, a converter node, and an MCAP recorder; layout generated by `python -m openral_foxglove_bringup.layout`. Spawned into deploy-sim by `openral deploy sim --foxglove`._

- `SCENE_TOPICS: list[str]` (topics.py L19) — Allowlist group 1: the geometry needed to draw the robot in its world — camera images + `camera_info`, `/map`, octomap cloud, `/scan`, `/odom`, `/joint_states`, `/robot_description`, `/openral/robot_description`, `/tf`(`_static`).
- `DEPTH_TOPICS: list[str]` (topics.py L49) — Allowlist group 2: the RGBD / nvblox / visual-SLAM leg — per-camera depth + points, the DA3 sidecar's depth topics, nvblox's filtered depth + ESDF cloud, `/openral/nav2/scan`, `/openral/imu`, `/openral/visual_slam/odometry`. Published only under the matching deploy posture.
- `BUCKET2_TOPICS: list[str]` (topics.py L67) — Allowlist group 3: the `bucket2_markers` converter's outputs (`/openral/world_collisions_markers`, `/openral/world_voxels_cloud`).
- `TELEMETRY_TOPICS: list[str]` (topics.py L89) — Allowlist group 4: the mission/state plane mirrored from the OTel dashboard's cards — world/policy state, episode, critic score, active task, objects, attachments, skill registry, diagnostics. Observation-only; `/openral/safety_status` is excluded pending safety-WG sign-off.
- `BUCKET1_TOPIC_WHITELIST: list[str]` (topics.py L108) — The `foxglove_bridge` `topic_whitelist`: the four groups above concatenated, nothing else — notably never the safety/e-stop/action/command topics. Imported by both launch files so the allowlist has one source of truth.
- `ASSET_URI_ALLOWLIST: list[str]` (topics.py L136) — The bridge's `asset_uri_allowlist`, controlling which mesh/asset paths a connected viewer may fetch to draw the URDF; extended from upstream's default to admit dotted version segments while still refusing path traversal. Applied by both launch files.
- `READ_ONLY_CAPABILITIES: list[str]` (topics.py L141) — `["connectionGraph", "assets"]`; omits `clientPublish`/`services`/`parameters` so a connected viewer cannot publish, call services, or write params (cannot actuate).
- `DEFAULT_CAMERAS: tuple[str, ...]` (layout.py L55) — `("top", "wrist_left", "wrist_right")`; the camera slots the shipped layout is generated for. These are sensor names from the robot manifest, not labels — a slot that doesn't exist on a given robot renders as a dead panel.
- `DEFAULT_LAYOUT_PATH: str` (layout.py L58) — `"config/openral_layout.json"`; where the generated layout is shipped, relative to the package root.
- `camera_image_topic(camera, *, compressed=False) -> str` (layout.py L61) — `/openral/cameras/<camera>/image`, or the `image_transport` `/compressed` sibling.
- `build_layout(cameras=DEFAULT_CAMERAS, *, compressed=False, follow_frame="base_link") -> dict[str, Any]` (layout.py L227) — Builds the Foxglove layout for a scene's camera slots: a hero 3D panel plus one Image panel per camera and tabbed panels for nav, joints, policy/world state, and diagnostics. Every referenced topic is on `BUCKET1_TOPIC_WHITELIST`, and no generated layout contains a write-capable panel.
- `main(argv=None) -> int` (layout.py L410) — `python -m openral_foxglove_bringup.layout` entry point: `--cameras`, `--compressed`, `--follow-frame`, `-o/--output`, `--write-default` (overwrite the shipped `config/openral_layout.json`).
- `class MarkerSpec` (bucket2_markers.py L44) — Frozen dataclass: one `visualization_msgs/Marker`'s pose/scale/type as plain data (ROS-free, so the conversion is unit-testable).
- `capsule_markers(radius, half_length, origin_xyzrpy, object_id) -> list[MarkerSpec]` (bucket2_markers.py L86) — Pure: convert `openral_msgs/WorldCollision` parallel capsule arrays to cylinder marker specs (length `2·half_length`, scale = diameter `2·radius`; sphere → zero-length cylinder). Raises `ValueError` on array-length mismatch.
- `occupied_voxel_centers(origin, resolution, size, occupancy, orientation_xyzw=(0,0,0,1)) -> list[tuple[float,float,float]]` (bucket2_markers.py L153) — Pure: centre coordinates of occupied voxels, placed by the grid's own rotation — without it every voxel is drawn somewhere the obstacle is not. Raises on a mismatched occupancy length or a non-unit orientation quaternion.
- `class Bucket2MarkersNode` (bucket2_markers.py L227) — `rclpy.node.Node` subscribing `/openral/world_collisions` + `/openral/world_voxels`; re-publishes `/openral/world_collisions_markers` (`MarkerArray`) + `/openral/world_voxels_cloud` (`PointCloud2`) via the pure functions. Read-only viz; defers rclpy/openral_msgs imports.
  - `spin() -> None` (bucket2_markers.py L370) — Spins the node until shutdown.
  - `destroy() -> None` (bucket2_markers.py L376) — Releases the subscriptions and publishers. Safe to call multiple times.
- `main() -> None` (bucket2_markers.py L381) — Console entry point (installed as `lib/openral_foxglove_bringup/bucket2_markers`); `rclpy.init` → spin → shutdown.
- `generate_launch_description() -> LaunchDescription` (bucket2.launch.py L29) — Bucket-2 converter node launch; declares `use_sim_time` and spawns the `bucket2_markers` node.
- `generate_launch_description() -> LaunchDescription` (foxglove.launch.py L42) — Read-only `foxglove_bridge` bring-up for the Bucket-1 topics; `topic_whitelist` is an explicit allowlist, so anything unmatched — including safety/e-stop/action/command topics — stays invisible.
- `generate_launch_description() -> LaunchDescription` (record.launch.py L35) — Opt-in MCAP recorder for the Bucket-1 allowlist (`ros2 bag record -e`, one regex per allowlisted topic).

### `packages/openral_reasoner_ros/openral_reasoner_ros/critic_producer_node.py`
_Tier-C critic producer, the default publisher for `/openral/failure/critic`. Subscribes the generic `/openral/critic/score` topic that any reward model publishes, routes each sample through a `CriticWatchdogGroup`, and on a stall emits a `FailureTrigger` that `reasoner_node` maps to a forced Tier-C tick. Advisory only._

- `class CriticProducerNode` (L58) — `rclpy.node.Node`. Builds a `CriticWatchdogGroup` and a `FailureBusPublisher`, subscribes `CriticScore`, and routes each sample; on a stall publishes one `KIND_CRITIC`/`SEVERITY_FAIL` `FailureTrigger` per stall.
  - `destroy_node() -> None` (L118) — Tears down the `FailureBusPublisher` before base teardown.
- `main(args=None) -> None` (L124) — Console entry point (installed as `lib/openral_reasoner_ros/critic_producer_node.py`); `rclpy.init` → spin → shutdown.
