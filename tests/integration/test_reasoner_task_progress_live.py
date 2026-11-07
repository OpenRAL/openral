"""Live ROS test for the LLM-dispatched ``query_task_progress`` path.

Two behaviours of ``ReasonerNode._dispatch_query_task_progress``:

1. A blank ``QueryTaskProgressTool.task`` reaches the reward monitor as the active
   mission task's text. Sent blank, ``reward_monitor_node`` would score its launch-time
   ``task`` parameter, not the goal the LLM is pursuing.
2. A reply that lands after the active task changed (a new operator goal replaced the
   mission while the call was pending) is dropped instead of being re-prompted into the
   new goal's context, as ``_on_mission_verify_response`` already does.

Real reasoner node + a real ``QueryTaskProgress`` service + a real ``/openral/prompt``
subscriber on a real DDS graph (CLAUDE.md §1.11); the only double is the scripted LLM
client ``on_configure`` needs. Gated on ``OPENRAL_TEST_ROS_LIVE=1``
(``scripts/ros_live_tests.sh``); locally, after ``just ros2-build && source
install/setup.bash``, run ``just test-ros-live -k task_progress``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))

pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy service/topic test — set OPENRAL_TEST_ROS_LIVE=1 and source "
    "install/setup.bash first.",
)


def _spin_until(executor: Any, predicate: Any, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_blank_task_is_the_active_mission_task_and_stale_replies_are_dropped() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.srv")
    from openral_core import QueryTaskProgressTool, WaitTool
    from openral_msgs.msg import PromptStamped
    from openral_msgs.srv import QueryTaskProgress
    from openral_reasoner import ToolPalette
    from openral_reasoner.mission import MissionState
    from openral_reasoner_ros import ReasonerNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    asked: list[str] = []
    replace_goal_before_reply = False
    progress_prompts: list[str] = []

    rclpy.init()
    try:
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            tick_hz=0.2,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        reasoner._renderer.set_mission(MissionState.from_prompt("pick up the milk"))

        def _serve(request: Any, response: Any) -> Any:
            asked.append(request.task)
            if replace_goal_before_reply:
                # The operator's new goal lands while the reward model is still scoring.
                reasoner._renderer.set_mission(MissionState.from_prompt("wipe the table"))
            response.ok = True
            response.progress_now = 0.4
            response.success_now = 0.2
            response.frames_seen = 12
            return response

        server = rclpy.create_node("openral_test_reward_monitor")
        server.create_service(QueryTaskProgress, "/openral/perception/query_task_progress", _serve)
        listener = rclpy.create_node("openral_test_prompt_listener")
        listener.create_subscription(
            PromptStamped,
            "/openral/prompt",
            lambda m: (
                progress_prompts.append(m.text)
                if m.text.startswith("query_task_progress[")
                else None
            ),
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
        executor = rclpy.executors.SingleThreadedExecutor()
        for node in (reasoner, server, listener):
            executor.add_node(node)
        _spin_until(executor, lambda: False, 1.5)  # DDS discovery

        reasoner._dispatch_query_task_progress(QueryTaskProgressTool(), traceparent=None)
        assert _spin_until(executor, lambda: bool(progress_prompts), 10.0), "no re-prompt"
        assert asked == ["pick up the milk"]

        replace_goal_before_reply = True
        reasoner._dispatch_query_task_progress(QueryTaskProgressTool(), traceparent=None)
        assert _spin_until(executor, lambda: len(asked) == 2, 10.0), "second query never served"
        _spin_until(executor, lambda: len(progress_prompts) > 1, 2.0)
        assert len(progress_prompts) == 1, f"stale reply re-prompted: {progress_prompts!r}"
    finally:
        rclpy.try_shutdown()
