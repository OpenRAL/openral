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
)
# tests/unit/test_ros_live_targets.py asserts every OPENRAL_TEST_ROS_LIVE-gated
# file under tests/integration/ appears above — a gated test missing from this
# list runs on NO CI surface.

export OPENRAL_TEST_ROS_LIVE=1

exec python -m pytest "${TARGETS[@]}" "$@"
