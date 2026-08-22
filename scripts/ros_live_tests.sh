#!/usr/bin/env bash
# The live-ROS test suite (`OPENRAL_TEST_ROS_LIVE=1`) — one list, two callers.
#
# `just test-ros-live` and the `docker-build` workflow's "Live ROS tests" step
# both exec this file, so the target list cannot drift between the dev loop and
# CI. Adding a live-ROS test means adding it to TARGETS below and nowhere else.
#
# What these tests need that a plain `pytest` runner does not have: a real
# `rclpy`, a real DDS graph, and the colcon-built `openral_msgs` /
# `openral_reasoner_ros` / `openral_prompt_router` overlay. No GPU — the
# free-VRAM readings are pinned at the `nvidia-smi` process boundary.
#
#   dev host:  source /opt/ros/jazzy/setup.bash && just ros2-build \
#              && source install/setup.bash && just test-ros-live
#   CI:        inside `openral:x86`, which bakes all of the above.
#
# NOT in TARGETS: the one live test in tests/unit/test_gstreamer_perception_tee.py.
# It additionally needs PyGObject, which the open deploy image deliberately does
# not ship (the GStreamer media stack is OpenRAL Pro), so no CI surface can run
# it. `just test` picks it up on a dev host that has the `gstreamer` extra.
#
# `python` resolves to the workspace venv: `just` invokes this under `uv run`,
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
    # The declaration-scoped place-approach allowance, fired deterministically
    # against the real safety_kernel_node. Needs the colcon-built kernel binary
    # + openral_msgs overlay, which only this image has.
    tests/integration/test_safety_kernel_place_allowance_band.py
    tests/integration/test_sim_sensor_bridge_tf_guard.py
    tests/integration/test_safety_status_latched_topic.py
    tests/integration/test_perception_overlay_live_topic.py
    # ADR-0097 place-witness attachment barrier. A bumped revision on an
    # unchanged payload must release the deferred action_applied tick; when it
    # did not, a SUCCESSFUL place aborted its own goal 8 s later.
    tests/integration/test_hal_attachment_barrier_live.py
    tests/integration/test_segment_in_view_service.py
    # Lives under tests/unit/ because it is a pure constant contract with no
    # graph, but it reads the numbers off the colcon-generated openral_msgs,
    # so it importorskips everywhere except this image. Without this entry the
    # SafetyStatus/FailureTrigger KIND_* redeclaration - the thing that decides
    # which fault a dashboard renders for a real safety stop - is pinned by a
    # test that runs on NO CI surface.
    tests/unit/test_safety_status_msg.py
    # ROS_DISTRO-gated rather than OPENRAL_TEST_ROS_LIVE-gated: the composed
    # runner/world-state/safety action-protocol suite needs the colcon
    # openral_msgs overlay, which only this image has — without this entry it
    # ran on NO CI surface, and the deadline-abort semantics change shipped
    # with a stale success assertion nobody could see; caught on a dev host
    # with the overlay sourced. Baked into the image via `COPY packages/`,
    # so PR builds run the PR's version. NOTE: keep comments in this array
    # free of close-parens — the guard test parses the block with a
    # non-greedy regex.
    packages/openral_rskill_ros/test/test_rskill_runner_node.py
    # Same reasoning: ROS_DISTRO-gated, and its ExecuteRskill.Result
    # failure_kind assertions read the colcon-generated constants, so only
    # this image can run them.
    packages/openral_rskill_ros/test/test_rskill_runner_failure_reason.py
    # ADR-0097 place-declaration goal lifecycle. Same overlay requirement,
    # and the transitions it pins - retract on end / cancel / E-stop /
    # executor-escape - are safety-relevant, so they must run on a CI
    # surface rather than only on a dev host.
    packages/openral_rskill_ros/test/test_place_declaration_lifecycle.py
)
# tests/unit/test_ros_live_targets.py asserts every OPENRAL_TEST_ROS_LIVE-gated
# file under tests/integration/ appears above — a gated test missing from this
# list runs on NO CI surface.

export OPENRAL_TEST_ROS_LIVE=1

exec python -m pytest "${TARGETS[@]}" "$@"
