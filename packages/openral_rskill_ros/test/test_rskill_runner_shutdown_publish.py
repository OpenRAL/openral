"""The runner's goal-end publishes after SIGINT shut the rclpy context down.

SIGINT invalidates the context first; a goal's execute callback that finishes afterwards
published its ``Episode(PHASE_END)`` marker on a dead publisher and printed an
``InvalidHandle`` traceback from the action server (Isaac ``deploy sim`` teardown,
2026-10-04). Real node, real context, real topic (CLAUDE.md §1.11).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


@contextmanager
def _runner_with_episode_listener() -> Iterator[tuple[Any, Any, list[Any]]]:
    import rclpy
    from openral_msgs.msg import Episode
    from openral_rskill_ros.compose import compose_so100_runtime
    from rclpy.lifecycle import TransitionCallbackReturn

    rclpy.init()
    runtime = compose_so100_runtime(skill_resolver=lambda **_: None)
    node = runtime.skill_runner_node
    node.set_parameters([rclpy.parameter.Parameter("joint_state_staleness_limit_s", value=0.5)])
    helper = rclpy.create_node("openral_episode_listener")
    seen: list[Any] = []
    helper.create_subscription(Episode, "/openral/episode", seen.append, 10)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield executor, node, seen
    finally:
        executor.shutdown()
        helper.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        rclpy.try_shutdown()


def _spin_until(executor: Any, seen: list[Any], n: int) -> None:
    import time

    deadline = time.monotonic() + 5.0
    while len(seen) < n and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


def test_episode_end_still_publishes_while_the_context_is_up() -> None:
    with _runner_with_episode_listener() as (executor, node, seen):
        node._publish_episode_end(task_string="pick the cube", success=True)
        _spin_until(executor, seen, 1)
        assert [(m.phase, m.task_string, m.success) for m in seen] == [(1, "pick the cube", True)]
        assert node._episode_counter == 1


def test_episode_end_after_context_shutdown_is_quiet_and_publishes_nothing() -> None:
    import rclpy

    with _runner_with_episode_listener() as (_executor, node, seen):
        rclpy.shutdown()  # what the SIGINT handler does, from outside the goal thread
        node._publish_episode_end(task_string="pick the cube", success=False)  # must not raise
        node._publish_episode_end(task_string="pick the cube", success=False)
        assert seen == []
        assert node._episode_counter == 0
        assert node._shutdown_publish_logged


def test_a_publish_failure_with_the_context_up_still_propagates() -> None:
    from rclpy.exceptions import InvalidHandle

    with _runner_with_episode_listener() as (_executor, node, _seen):
        node.destroy_publisher(node._episode_pub)  # dead publisher, live context
        with pytest.raises(InvalidHandle):
            node._publish_episode_end(task_string="pick the cube", success=True)
        assert not node._shutdown_publish_logged
        node._episode_pub = None


def test_declaration_retraction_after_shutdown_is_quiet_and_keeps_the_in_force_record() -> None:
    import rclpy
    from openral_core import GraspDeclaration

    declaration = GraspDeclaration(
        target_id="cell:restock_box",
        contact_links=("openarm_left_finger_pair",),
        timeout_s=70.0,
        stamp_ns=0,
    )
    with _runner_with_episode_listener() as (_executor, node, _seen):
        node._publish_grasp_declaration(declaration)
        assert node._active_grasp_declaration is not None  # published while the context was up
        rclpy.shutdown()
        node._retract_grasp_declaration()  # must not raise nor log an error
        assert node._active_grasp_declaration is not None  # nothing retracted on the wire
