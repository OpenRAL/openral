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
    tests/integration/test_sim_sensor_bridge_tf_guard.py
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
)
# tests/unit/test_ros_live_targets.py asserts every OPENRAL_TEST_ROS_LIVE-gated
# file under tests/integration/ appears above — a gated test missing from this
# list runs on NO CI surface.

export OPENRAL_TEST_ROS_LIVE=1

exec python -m pytest "${TARGETS[@]}" "$@"
