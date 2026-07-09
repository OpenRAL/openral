"""reward_monitor_node's camera-label derivation for the dashboard rSkill card.

Surfaces which camera the Robometer reward monitor attends to (``reward.camera``
span attribute on ``reward.score``) — see
``openral_perception_ros.reward_monitor_node._camera_label``.
"""

from __future__ import annotations

import pytest

# openral_perception_ros is a colcon-built ROS package (ament_cmake); like every
# other ROS-package unit test (test_image_convert, test_skill_runner_*), skip
# cleanly when the workspace overlay isn't sourced. The ros2-test CI job sources
# install/setup.bash and runs this for real. _camera_label itself is pure-Python
# (plain string parsing, no rclpy) — the guard is about the package being on the
# path, and the module keeps rclpy imports inside main() specifically so this
# helper stays importable without ROS.
pytest.importorskip("openral_perception_ros")

from openral_perception_ros.reward_monitor_node import _camera_label


def test_camera_label_extracts_name_from_convention_topic() -> None:
    """``/openral/cameras/<name>/image`` -> ``<name>`` (the common case)."""
    assert _camera_label("/openral/cameras/top/image") == "top"


def test_camera_label_extracts_name_with_underscores() -> None:
    """Multi-word camera names (e.g. the LIBERO default) round-trip untouched."""
    assert _camera_label("/openral/cameras/agentview_left/image") == "agentview_left"


def test_camera_label_falls_back_to_raw_topic_when_off_convention() -> None:
    """A topic that doesn't match the convention is returned verbatim, not guessed."""
    assert _camera_label("/some/other/topic") == "/some/other/topic"


def test_camera_label_falls_back_on_short_topic() -> None:
    """Fewer than 3 path segments after the leading slash can't hold a camera name."""
    assert _camera_label("/openral/cameras") == "/openral/cameras"
