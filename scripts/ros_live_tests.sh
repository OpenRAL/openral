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
    tests/integration/test_critic_producer_node.py
    # Declaration-scoped place-approach allowance against the real
    # safety_kernel_node; needs the colcon kernel binary + openral_msgs
    # overlay, which only this image has.
    tests/integration/test_safety_kernel_place_allowance_band.py
    # ADR-0098: same rig with the declared target's geometry shipped, run at
    # the DEPLOYED margin -- the collision gtests run at margin 0, where
    # gating at `margin` vs at the surface is algebraically identical.
    tests/integration/test_safety_kernel_place_target_geometry.py
    tests/integration/test_sim_sensor_bridge_tf_guard.py
    tests/integration/test_safety_status_latched_topic.py
    tests/integration/test_perception_overlay_live_topic.py
    # ADR-0097 place-witness attachment barrier. A bumped revision on an
    # unchanged payload must release the deferred action_applied tick; when it
    # did not, a SUCCESSFUL place aborted its own goal 8 s later.
    tests/integration/test_hal_attachment_barrier_live.py
    # A world-voxel stop names its cell only as an index; the grid that index
    # addresses arrives on a different topic, and until joined the record
    # can't look at the map -- how the 2026-08-22 round adjudicated two stops
    # on ground truth that never examined the cell.
    tests/integration/test_estop_voxel_backing_live.py
    tests/integration/test_segment_in_view_service.py
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
# tests/unit/test_ros_live_targets.py asserts every OPENRAL_TEST_ROS_LIVE-gated
# file under tests/integration/ appears above — a gated test missing from this
# list runs on NO CI surface.

export OPENRAL_TEST_ROS_LIVE=1

exec python -m pytest "${TARGETS[@]}" "$@"
