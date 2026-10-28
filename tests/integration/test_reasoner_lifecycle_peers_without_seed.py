"""Configured lifecycle peers must reach the palette even when the skill seed is skipped.

`_dispatch_lifecycle_transition` and the tool-use decoder both refuse a node outside
`ToolPalette.node_ids`. The peers from `lifecycle_peer_node_ids` used to be merged only
after the seed's `robot_yaml` / `rskill_search_paths` early return, so a deploy without
search-path skills refused every transition of its own configured peers.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_lifecycle_peers_without_seed.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_lifecycle_peers_enter_palette_without_search_paths() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    # No robot_yaml and no rskill_search_paths: the skill seed returns early.
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "lifecycle_peer_node_ids:=[/openral_slam_toolbox, '']",
        ],
    )
    try:
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()
        assert reasoner._palette.node_ids == frozenset({"/openral_slam_toolbox"})
        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
