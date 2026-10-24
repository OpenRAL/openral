#!/usr/bin/env bash
# The live-ROS test suite (`OPENRAL_TEST_ROS_LIVE=1`) — one list, two callers.
#
# `just test-ros-live` and the `docker-build` workflow's "Live ROS tests" step
# both exec this file, so the list can't drift between dev and CI. Add a
# live-ROS test here and nowhere else.
#
# Needs a real `rclpy`, a real DDS graph, and the colcon-built `openral_msgs` /
# `openral_reasoner_ros` / `openral_prompt_router` overlay. No GPU — free-VRAM
# reads are pinned at the `nvidia-smi` process boundary.
#
#   dev host:  source /opt/ros/jazzy/setup.bash && just ros2-build \
#              && source install/setup.bash && just test-ros-live
#   CI:        inside `openral:x86`, which bakes all of the above.
#
# NOT in TARGETS: tests/unit/test_gstreamer_perception_tee.py — needs
# PyGObject, which the open deploy image doesn't ship since GStreamer is
# OpenRAL Pro; `just test` picks it up on a dev host with the `gstreamer`
# extra.
#
# `python` resolves to the workspace venv: `just` runs this under `uv run`,
# and the image puts /workspace/.venv/bin first on PATH.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGETS=(
    tests/integration/test_reasoner_node_end_to_end.py
    tests/integration/test_reasoner_dispatch_robustness.py
    tests/integration/test_reasoner_async_llm.py
    tests/integration/test_reasoner_vram_pair_refusal.py
    tests/integration/test_reasoner_vram_eviction.py
    tests/integration/test_reasoner_palette_primes_vram_gate.py
    tests/integration/test_reasoner_palette_preserves_detector_coverage.py
    tests/integration/test_critic_producer_node.py
    # The scene-VLM node ends the sidecar its warm-up spawned (fake sidecar, no GPU).
    tests/integration/test_scene_vlm_node_teardown_live.py
    # A completion check that raises does not end the check worker thread.
    tests/integration/test_completion_check_worker_live.py
    # Declaration-scoped place-approach allowance against the real
    # safety_kernel_node; needs the colcon kernel binary + openral_msgs
    # overlay, which only this image has.
    tests/integration/test_safety_kernel_place_allowance_band.py
    # The real robot_state_publisher fed the HAL's URDF-unit joint states (`urdf_joint_map`).
    tests/integration/test_rsp_urdf_units_live.py
    # ADR-0098: same rig with the declared target's geometry shipped, run at
    # the DEPLOYED margin -- the collision gtests run at margin 0, where
    # gating at `margin` vs at the surface is algebraically identical.
    tests/integration/test_safety_kernel_place_target_geometry.py
    # Grasp-target exemption on the REAL OpenArm collision model at the real
    # cell's 20 mm margin -- every fail-closed row, plus the default-off kernel.
    tests/integration/test_safety_kernel_grasp_target_band.py
    # ADR-0102 slot rows on the real OpenArm model: a row's uncommanded joints
    # are checked at their measured pose, never the zero padding.
    tests/integration/test_safety_kernel_slot_row_measured_fill.py
    # Issue #356: the torso is in the OpenArm model; a hand folded into the
    # shoulder block is refused as a self hit naming openarm_body_link0, and
    # the rest pose (pedestals inside the torso, SRDF-exempt) is accepted.
    tests/integration/test_safety_kernel_openarm_torso.py
    # Issue #373: the real kernel's external-torque observer on the MuJoCo OpenArm
    # (lowered force_observer model): no stop in free motion, a stop on a push
    # and on a deliberate arm/box collision. Needs mujoco in the venv.
    tests/integration/test_safety_kernel_force_observer_openarm.py
    tests/integration/test_sim_sensor_bridge_tf_guard.py
    tests/integration/test_sim_depth_registered_colour_live.py
    # DeployRuntime.estimated_depth_camera: the real depth provider in metric
    # mode against a fake DA3 sidecar at its ZMQ boundary (no GPU).
    tests/integration/test_estimated_depth_provider_live.py
    tests/integration/test_hal_sigint_teardown_live.py
    # Every MuJoCo twin heartbeats a fresh, empty attachment set, not only the
    # HALs with attach mechanics (kernel fails closed on an unstamped set).
    tests/integration/test_sim_sensor_bridge_attachment_heartbeat.py
    # Defense-in-depth E-stop, on a real three-process graph: the in-band
    # safety node is SIGKILLed mid chunk-stream and /openral/estop still fires,
    # from a different process, with the structured KIND_TIMEOUT trigger. Needs
    # the colcon overlay for openral_safety / openral_safety_watchdog /
    # openral_human_estop executables, so only this image can run it.
    tests/integration/test_estop_watchdog_graph_live.py
    tests/integration/test_safety_status_latched_topic.py
    tests/integration/test_perception_overlay_live_topic.py
    # ADR-0097 place-witness attachment barrier. A bumped revision on an
    # unchanged payload must release the deferred action_applied tick; when it
    # did not, a SUCCESSFUL place aborted its own goal 8 s later.
    tests/integration/test_hal_attachment_barrier_live.py
    # The vision attachment leg heartbeats its set only while every gripper's
    # jaw-position channel is live, and its revision survives a re-activate.
    tests/integration/test_vision_attachment_heartbeat_live.py
    # Grasp masks back-project in the depth header's optical frame through the
    # driver's CameraInfo, never the manifest's body frame or nominal K.
    tests/integration/test_vision_attachment_optical_frame_live.py
    # A released payload stays a checked attached record, frozen in the base frame, until
    # the jaws are clear; the real kernel accepts the retreat only while it is published.
    tests/integration/test_vision_attachment_release_window_live.py
    # The pre-grasp target leg measures the declared target's region from the
    # search box, the voxel map and a SegmentInView mask; freezes it under a TTL
    # on a lost view, refuses two candidates, dies with the declaration.
    tests/integration/test_grasp_target_leg_live.py
    # Issue #353: a close that ends without an ATTACH publishes one KIND_GRASP_OUTCOME
    # miss on /openral/failure/hal (FAIL when declared, WARN when not), the recovery
    # ATTACH one INFO attach, a first-try ATTACH nothing.
    tests/integration/test_grasp_outcome_live.py
    # Megabyte best-effort clouds cross the real robot_self_filter (both lossy hops)
    # under the deploy's Fast DDS large-data profile: >= 95 % delivered, publish()
    # non-blocking. Forces rmw_fastrtps_cpp (the image defaults to Cyclone).
    tests/integration/test_large_cloud_transport_live.py
    # The real place producer leg measures the surface under the carried payload
    # in the voxel map (no fixture, no named target), declares it with a slab
    # region, attests the map_support_proximity witness, and retracts within the
    # freeze TTL on a silent grid; through a real aggregator into a real safety
    # kernel (needs the colcon kernel + openral_msgs overlay).
    tests/integration/test_place_target_leg_live.py
    # Set-down then release on a measured surface against the real kernel: the
    # retreat is accepted only with the map_support_proximity witness.
    tests/integration/test_place_target_release_live.py
    # Every scene vision_attachment_* param the deploy forwards is declared by the
    # real HAL node and reaches the grasp trigger (rclpy drops undeclared names).
    tests/integration/test_vision_attachment_hal_params_live.py
    # A reasoner-named grasp target grounds from a live lifted detection into a
    # search box on the goal's declaration (never a region); refusals send no goal.
    tests/integration/test_reasoner_grounded_targets_live.py
    # The sim producer measures a dispatched grasp declaration's target box and
    # rides it on every attachment envelope, revision unchanged, until retracted.
    tests/integration/test_sim_bridge_grasp_declaration_live.py
    # Grasp-target control pair on a MuJoCo twin: real kernel (20 mm margin,
    # grasp_allowance_enabled), World State node, SimSensorBridge producer.
    tests/sim/test_gripper_twin_hal_mujoco_grasp_pair.py
    # A world-voxel stop names its cell only as an index; the grid that index
    # addresses arrives on a different topic, and until joined the record
    # can't look at the map -- how the 2026-08-22 round adjudicated two stops
    # on ground truth that never examined the cell.
    tests/integration/test_estop_voxel_backing_live.py
    tests/integration/test_segment_in_view_service.py
    # `openral estop reset` clears the kernel, then the runner; a latched
    # goal rejection logs a WARNING naming the recovery.
    tests/integration/test_estop_reset_cli_live.py
    # Nav2's side of issue #108: a filtered scan keeps a carried payload out
    # of the cost grid, and a self-return inside the robot's own footprint is
    # dropped while a real obstacle at the same bearing survives -- only a
    # real nav2_costmap_2d binary settles this. Load-bearing since Nav2 is
    # base-only, with no footprint growth over the payload. Needs
    # ros-jazzy-nav2-bringup, which this image bakes.
    tests/integration/test_nav2_scan_filter_live.py
    # Issue #211: Nav2 filters observation z in each costmap's OWN
    # global_frame; `map`'s z origin isn't the floor -- slam_toolbox flattens
    # `base_link`, the arm mount 0.700 m up the pedestal, to z = 0, so
    # `min_obstacle_height` 0.0 kept local-costmap returns but discarded every
    # global one, and the planner ran against a blank grid silently. Paired
    # control restores the pre-fix gate and re-empties the grid -- claims the
    # upstream binary applies the cut TWICE, from two differently-scoped
    # parameters.
    tests/integration/test_nav2_global_costmap_height_live.py
    # Opt-in /cmd_vel relay for body_twist robots without base_joints: real
    # relay + real kernel booted from r1pro -- chunk content, one stale zero,
    # over-speed veto, and refusal of a base_joints manifest.
    tests/integration/test_twist_to_action_relay_live.py
    # Under tests/unit/ -- pure constant contract, no graph -- but reads
    # numbers off the colcon-generated openral_msgs, so it importorskips
    # everywhere but this image. Pins the SafetyStatus/FailureTrigger KIND_*
    # redeclaration that decides which fault a dashboard renders for a real
    # safety stop.
    tests/unit/test_safety_status_msg.py
    # Same reasoning: no graph, but importing
    # tools/_nav2_costmap_silhouette_probe.py needs the colcon overlay. Pins
    # the counting rule for issue #108's payload half -- it compared
    # primitives placed against objects declared, so more than one primitive
    # per payload could never satisfy it. A live baguette carry measures 16
    # primitives on one object, which is why every scene run to date reported
    # it unmeasured.
    tests/unit/test_nav2_costmap_probe_payload_count.py
    # Pins the generated FailureTrigger constants against the plain-int copy
    # in openral_observability.failure_bus, both directions -- the leg a
    # ROS-free caller reads, which silently missed KIND_COLLISION for the
    # whole life of the collision stack.
    tests/unit/test_failure_bus_idl_mirror.py
    # ROS_DISTRO-gated rather than OPENRAL_TEST_ROS_LIVE-gated: needs the
    # colcon openral_msgs overlay only this image has -- without this entry a
    # deadline-abort semantics change shipped with a stale success assertion
    # nobody could see; caught on a dev host with the overlay sourced. Baked
    # in via `COPY packages/`, so PR builds run the PR's version. NOTE: keep
    # comments in this array free of close-parens -- the guard test parses
    # the block with a non-greedy regex.
    packages/openral_rskill_ros/test/test_rskill_runner_node.py
    # Same reasoning: ROS_DISTRO-gated, and its ExecuteRskill.Result
    # failure_kind assertions read the colcon-generated constants, so only
    # this image can run them.
    packages/openral_rskill_ros/test/test_rskill_runner_failure_reason.py
    # ADR-0097 place-declaration goal lifecycle; same overlay requirement.
    # Pins safety-relevant transitions -- retract on end / cancel / E-stop /
    # executor-escape -- so they must run on CI, not only a dev host.
    packages/openral_rskill_ros/test/test_place_declaration_lifecycle.py
    # Builds a real LaunchDescription, needs ROS `launch`, skips the
    # plain-pytest lane -- ran on NO CI surface until a diff first selected
    # this package, sitting latently broken. Pins the MCAP recorder's scope:
    # asserts /openral/estop, /openral/safe_action, and other actuation
    # topics are NOT reachable through the Bucket-1 record patterns.
    packages/openral_foxglove_bringup/test/test_record_launch.py
)

# Files that need rclpy, the colcon overlay, or the live gate and ran on NO CI
# surface until the 2026-10-15 sweep -- the safety-kernel twin and
# synthetic-collision suites, the ROS-gated integration tests, the
# packages/*/test node suites that test-ros2 runs only on dispatch, and the
# tests/unit files that skip without ROS. Each runs in its OWN pytest process:
# the packages/*/test suites own rclpy init/shutdown per module, and sharing
# one interpreter with the block above cascades into failures. MUJOCO_GL=egl:
# the camera-rendering twins need a GL context, and Mesa's EGL gives one with
# no display or GPU. Left out until fixed: test_manifest_hal_reset_to_pose and
# test_openarm_hal_lifecycle -- both abort at teardown, the exit 139 / 134
# family #384 tracks -- and test_franka_scene_attach.launch, whose process
# dies mid-test in this image.
ISOLATED_TARGETS=(
    packages/openral_foxglove_bringup/test/test_bucket2_markers.py
    packages/openral_foxglove_bringup/test/test_foxglove_launch.py
    packages/openral_foxglove_bringup/test/test_layout.py
    packages/openral_hal_node/test/test_lifecycle_estop_real_adapters.py
    packages/openral_hal_node/test/test_lifecycle_estop_safety.py
    packages/openral_hal_node/test/test_lifecycle_node.py
    packages/openral_hal_node/test/test_lifecycle_slot_group_stop_boundary.py
    packages/openral_hal_node/test/test_mobile_base_helpers.py
    packages/openral_hal_node/test/test_real_hal_spans.py
    packages/openral_human_estop/test/test_forwarder_node.py
    packages/openral_human_estop/test/test_forwarder_node_sigint_shape.py
    packages/openral_nav2_bringup/test/test_nav2_launch.py
    packages/openral_nav2_bringup/test/test_payload_scan_filter.py
    packages/openral_perception_ros/test/test_ros_image_detector_node_sigint_shape.py
    packages/openral_prompt_router/test/test_prompt_router_node.py
    packages/openral_prompt_router/test/test_prompt_router_node_sigint_shape.py
    packages/openral_reasoner_ros/test/test_reasoner_node.py
    packages/openral_reasoner_ros/test/test_reasoner_node_sigint_shape.py
    packages/openral_rskill_ros/test/test_deploy_e2e_autostart.py
    packages/openral_rskill_ros/test/test_deploy_e2e_clock_domain.py
    packages/openral_rskill_ros/test/test_deploy_e2e_detector_available.py
    packages/openral_rskill_ros/test/test_deploy_e2e_estop_sources.py
    packages/openral_rskill_ros/test/test_deploy_e2e_joint_states_topic.py
    packages/openral_rskill_ros/test/test_deploy_e2e_octomap_frames.py
    packages/openral_rskill_ros/test/test_deploy_e2e_prompt_router_autostart.py
    packages/openral_rskill_ros/test/test_deploy_e2e_scene_sensor_mounts.py
    packages/openral_rskill_ros/test/test_deploy_e2e_visual_slam.py
    packages/openral_rskill_ros/test/test_deploy_rig_values_reach_nodes.py
    packages/openral_rskill_ros/test/test_grasp_declaration_lifecycle.py
    packages/openral_rskill_ros/test/test_kernel_params_no_empty_lists.py
    packages/openral_rskill_ros/test/test_no_dashboard_otlp_env.py
    packages/openral_rskill_ros/test/test_openarm_real_world_voxels_scene.py
    packages/openral_rskill_ros/test/test_rskill_runner_node_sigint_shape.py
    packages/openral_rskill_ros/test/test_rskill_runner_shutdown_publish.py
    packages/openral_rskill_ros/test/test_runtime_node_prewarm_shape.py
    packages/openral_rskill_ros/test/test_runtime_node_sigint_shape.py
    packages/openral_rskill_ros/test/test_starting_pose_dispatch.py
    packages/openral_rskill_ros/test/test_world_state_executor.py
    packages/openral_safety/test/test_lowering_regression.py
    packages/openral_safety/test/test_supervisor_node.py
    packages/openral_safety/test/test_supervisor_node_sigint_shape.py
    packages/openral_safety/test/test_urdf_lowering_always_colliding.py
    packages/openral_safety/test/test_urdf_lowering_fk.py
    packages/openral_safety/test/test_urdf_lowering_geometry.py
    packages/openral_safety/test/test_urdf_lowering_lower_robot.py
    packages/openral_safety/test/test_urdf_lowering_sampling.py
    packages/openral_safety_watchdog/test/test_deadman_watchdog.py
    packages/openral_safety_watchdog/test/test_deadman_watchdog_node_sigint_shape.py
    packages/openral_safety_watchdog/test/test_hardware_estop_node.py
    packages/openral_safety_watchdog/test/test_hardware_estop_node_sigint_shape.py
    packages/openral_slam_bringup/test/test_cuvslam_launch.py
    packages/openral_slam_bringup/test/test_depth_height_filter.py
    packages/openral_slam_bringup/test/test_nvblox_launch.py
    packages/openral_slam_bringup/test/test_pycuvslam_launch.py
    packages/openral_slam_bringup/test/test_pycuvslam_node.py
    packages/openral_slam_bringup/test/test_slam_toolbox_launch.py
    packages/world_state/test/test_lifecycle_node_image_qos.py
    packages/world_state/test/test_lifecycle_node_sigint_shape.py
    tests/integration/test_act_onnx.py
    tests/integration/test_autodetect_dds.py
    tests/integration/test_compose_deploy_sensors.py
    tests/integration/test_dashboard_attach_root_span.py
    tests/integration/test_dashboard_end_to_end.py
    tests/integration/test_dataset_recorder_bridge.py
    tests/integration/test_detect_ros2_topology.py
    tests/integration/test_doctor_ros.py
    tests/integration/test_failure_bus.py
    tests/integration/test_fastrtps_shm_purge_live.py
    tests/integration/test_hil_transport_publishes_and_caches_state.py
    tests/integration/test_manifest_hal_lifecycle_live.py
    tests/integration/test_nav2_global_costmap_height_live.py
    tests/integration/test_nav2_scan_filter_live.py
    tests/integration/test_object_lift_real_camera_live.py
    tests/integration/test_object_lift_world_state.py
    # Issue #418 object tracker: node lifecycle, query escalation, kernel K2/K5 wiring.
    tests/integration/test_object_query_escalation_live.py
    tests/integration/test_object_tracker_node_live.py
    tests/integration/test_object_tracking_kernel_live.py
    tests/integration/test_panda_mobile_hal_lifecycle.py
    tests/integration/test_real_hal_estop_ros2_control_live.py
    tests/integration/test_record_profile_full_live.py
    tests/integration/test_sim_idle_stepper_wiring.py
    tests/integration/test_smolvla_eraser_place_rskill.py
    tests/integration/test_so100_digital_twin_e2e.py
    tests/integration/test_world_state_events_executor.py
    tests/integration/test_world_state_integration.py
    tests/integration/test_world_state_stamped_objects.py
    tests/sim/safety/test_kernel_graded_scaling_approach.py
    tests/sim/safety/test_kernel_h1_self_collision.py
    tests/sim/safety/test_kernel_latency_soak.py
    tests/sim/safety/test_kernel_mjcf_lowered_self_collision.py
    tests/sim/safety/test_kernel_mobile_base_collision.py
    tests/sim/safety/test_kernel_per_mode_envelopes.py
    tests/sim/safety/test_kernel_self_collision_synthetic.py
    tests/sim/safety/test_kernel_voxel_bridge_staleness.py
    tests/sim/safety/test_kernel_voxel_collision_synthetic.py
    tests/sim/safety/test_kernel_with_gpu_policy_inference.py
    tests/sim/safety/test_kernel_with_h1_humanoid_twin.py
    tests/sim/safety/test_kernel_with_openarm_twin.py
    tests/sim/safety/test_kernel_with_rizon4_twin.py
    tests/sim/safety/test_kernel_with_so100_digital_twin.py
    tests/unit/test_deploy_e2e_camera_wiring.py
    tests/unit/test_deploy_e2e_coverage_ball.py
    tests/unit/test_deploy_e2e_voxel_resolution.py
    tests/unit/test_depth_convert.py
    tests/unit/test_execution_budget.py
    tests/unit/test_hal_lifecycle_error_logging.py
    tests/unit/test_hal_lifecycle_manifest.py
    tests/unit/test_hal_lifecycle_shutdown_verdict.py
    tests/unit/test_lifecycle_autostart.py
    tests/unit/test_lifecycle_error_logging.py
    tests/unit/test_real_bringup_include.py
    tests/unit/test_reasoner_execute_prompt.py
    tests/unit/test_reasoner_gpu_probe.py
    tests/unit/test_reasoner_palette_action_gate.py
    tests/unit/test_reasoner_scene_objects_gate.py
    tests/unit/test_reasoner_search_cascade.py
    tests/unit/test_reasoner_subdivision.py
    tests/unit/test_resolution_ab_pairing.py
    tests/unit/test_ros2_image_sensor_reader.py
    tests/unit/test_safety_supervisor_per_mode.py
    tests/unit/test_skill_runner_deadline.py
    tests/unit/test_skill_runner_import_errors.py
    tests/unit/test_skill_runner_representation_dispatch.py
)
# tests/unit/test_ros_live_targets.py asserts every OPENRAL_TEST_ROS_LIVE-gated
# file under tests/integration/ appears above — a gated test missing from this
# list runs on NO CI surface.

export OPENRAL_TEST_ROS_LIVE=1

# Exit 5 ("nothing collected") is expected under an explicit -k / -m selection,
# which empties some runs, and for a file skipped whole at import -- pytest
# reports a module-level importorskip as "1 skipped" yet exits 5. Otherwise it
# means a target ran nothing, which fails the step.
filtered=0
for a in "$@"; do
    case "$a" in -k | -k* | -m | -m*) filtered=1 ;; esac
done
rc=0
out=$(mktemp)
trap 'rm -f "$out"' EXIT
run() {
    local r=0
    # pipefail (set above) makes the pipeline's status pytest's, not tee's.
    "$@" 2>&1 | tee "$out" || r=$?
    if [ "$r" -eq 5 ] && { [ "$filtered" -eq 1 ] || grep -qE '[0-9]+ skipped' "$out"; }; then
        r=0
    fi
    [ "$r" -eq 0 ] || rc=$r
}
run python -m pytest "${TARGETS[@]}" "$@"
for t in "${ISOLATED_TARGETS[@]}"; do
    run env MUJOCO_GL="${MUJOCO_GL:-egl}" python -m pytest "$t" "$@"
done
exit "$rc"
