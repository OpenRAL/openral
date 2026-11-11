"""Live ROS test: a wrapped action's own success settles the reasoner's mission task.

Reproduces the 2026-11-11 OpenAMRobot run (Isaac, claude-sonnet-5-5): "drive to x=4,y=0
in map" dispatched the Nav2 ``navigate_to_pose`` rSkill, Nav2 SUCCEEDED at x=3.79 (inside
its 0.25 m goal tolerance), and the reasoner, seeing only ``trace=…`` and with no reward
monitor to complete the task, re-issued the goal twice and never marked the task done; a
second prompt ran as an aside after which the first task resumed. Covers:

1. The success's ``ExecuteRskill.Result.outcome_summary`` reaches the ``## EXECUTION`` line,
   and the Nav2 skill's ``completes_task_on_success`` completes the task.
2. With no reward monitor, the per-task attempt cap still abandons a task (after its one
   subdivision offer).
3. A success nobody can verify lets the next operator prompt replace the mission.

Real reasoner node, real ``openral_msgs`` result types, the real in-tree Nav2 / VLA
manifests and ``panda_mobile`` robot manifest (CLAUDE.md §1.11); the only double is the
scripted LLM client ``on_configure`` needs. Gated on ``OPENRAL_TEST_ROS_LIVE=1``
(``scripts/ros_live_tests.sh``); locally, after ``just ros2-build && source
install/setup.bash``, run ``just test-ros-live -k action_completes``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))

pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy test — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_REPO = Path(__file__).resolve().parents[2]
_NAV = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"


def _reasoner() -> Any:  # reason: ReasonerNode imports only with ROS sourced
    from openral_core import RobotDescription, RSkillManifest, WaitTool
    from openral_reasoner import build_tool_palette
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    robot = RobotDescription.from_yaml(str(_REPO / "robots" / "panda_mobile" / "robot.yaml"))
    manifests = [
        RSkillManifest.from_yaml(str(p)) for p in (_REPO / "rskills").glob("*/rskill.yaml")
    ]
    palette = build_tool_palette(installed_skills=manifests, robot_capabilities=robot.capabilities)
    node = ReasonerNode(
        client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
        palette=palette,
        tick_hz=0.2,
    )
    node.trigger_configure()
    node.trigger_activate()
    assert not node._task_progress_available, "fixture drift: no reward monitor here"
    return node


def test_nav2_success_reports_its_outcome_and_completes_the_task() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner.mission import MissionState
    from rclpy.task import Future

    summary = (
        "wrapped result-only action completed: status=STATUS_SUCCEEDED result{error_code=0} "
        "goal{pose.header.frame_id=map pose.pose.position.x=4 pose.pose.position.y=0}"
    )
    rclpy.init()
    try:
        reasoner = _reasoner()
        mission = MissionState.from_prompt("drive to x=4,y=0 in map")
        reasoner._renderer.set_mission(mission)
        mission.record_attempt(rskill_id=_NAV)
        reasoner._active_dispatch_generation = 7
        wrapped = ExecuteRskill.Impl.GetResultService.Response()
        wrapped.status = 4  # STATUS_SUCCEEDED
        wrapped.result.success = True
        wrapped.result.trace_id = "1a2b3c4d5e6f"
        wrapped.result.outcome_summary = summary
        future = Future()
        future.set_result(wrapped)
        reasoner._on_execute_rskill_result(
            ExecuteRskillTool(rskill_id=_NAV), 7, b"\x00" * 16, future, None
        )
        assert reasoner._renderer.executions[-1].summary == f"{summary} trace=1a2b3c4d"
        assert mission.is_complete(), "Nav2 success must complete the task"
        assert mission.tasks[0].status == "done"
    finally:
        rclpy.try_shutdown()


def test_attempt_cap_abandons_a_task_no_reward_monitor_can_verify() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_ATTEMPTS, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        call = ExecuteRskillTool(rskill_id=vla)
        mission = MissionState.from_prompt("pick up the cube")
        reasoner._renderer.set_mission(mission)
        for _ in range(DEFAULT_MAX_ATTEMPTS - 1):
            mission.record_attempt(rskill_id=vla)
            reasoner._maybe_verify_active_mission_task(call, traceparent=None, succeeded=True)
            assert mission.tasks[0].status == "active", "below the cap the LLM decides"
        mission.record_attempt(rskill_id=vla)
        reasoner._maybe_verify_active_mission_task(call, traceparent=None, succeeded=True)
        # At the cap: one subdivision offer first (the existing abandon path) ...
        assert mission.tasks[0].task_id in reasoner._subdivide_offered
        assert mission.tasks[0].status == "active"
        # ... then the next unverified attempt abandons it.
        mission.record_attempt(rskill_id=vla)
        reasoner._maybe_verify_active_mission_task(call, traceparent=None, succeeded=True)
        # Read through the summary: mypy keeps the attribute narrowed to "active".
        summary = mission.to_summary()["tasks"]
        assert isinstance(summary, list)
        assert summary[0]["status"] == "abandoned"
        assert mission.is_complete()
    finally:
        rclpy.try_shutdown()


def test_prompt_after_an_unverified_success_replaces_the_mission() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner.mission import MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        mission = MissionState.from_prompt("pick up the cube")
        reasoner._renderer.set_mission(mission)
        mission.record_attempt(rskill_id=vla)
        prompt = PromptStamped()
        prompt.header.frame_id = "dashboard"
        prompt.text = "drive back to start"
        # Control: after a FAILED attempt a prompt is guidance, the queue is kept.
        reasoner._maybe_verify_active_mission_task(
            ExecuteRskillTool(rskill_id=vla), traceparent=None, succeeded=False
        )
        reasoner._on_prompt(prompt)
        assert reasoner._renderer.mission is mission
        reasoner._maybe_verify_active_mission_task(
            ExecuteRskillTool(rskill_id=vla), traceparent=None, succeeded=True
        )
        assert mission.tasks[0].status == "active", "no verifier: the task is not completed"
        reasoner._on_prompt(prompt)
        replaced = reasoner._renderer.mission
        assert replaced is not mission
        assert replaced is not None and replaced.tasks[0].text == "drive back to start"
    finally:
        rclpy.try_shutdown()


def test_a_stale_success_does_not_complete_the_goal_that_replaced_its_mission() -> None:
    """A Nav2 success that lands after a new operator goal rebuilt the mission settles
    nothing: the new goal's first task (no attempt yet) is not completed by it."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import PromptStamped
    from openral_reasoner.mission import MissionState
    from rclpy.task import Future

    rclpy.init()
    try:
        reasoner = _reasoner()
        first = MissionState.from_prompt("drive to x=4,y=0 in map")
        reasoner._renderer.set_mission(first)
        first.record_attempt(rskill_id=_NAV)
        active = first.active()
        assert active is not None
        # The goal is in flight for the first mission's task ...
        reasoner._active_dispatch_generation = 3
        reasoner._active_dispatch_task_id = active.task_id
        reasoner._active_dispatch_mission_generation = reasoner._mission_generation
        # ... when the operator starts over.
        prompt = PromptStamped()
        prompt.header.frame_id = "dashboard"
        prompt.text = "drive to the kitchen"
        prompt.metadata_json = '{"new_goal": true}'
        reasoner._on_prompt(prompt)
        second = reasoner._renderer.mission
        assert second is not None and second is not first
        wrapped = ExecuteRskill.Impl.GetResultService.Response()
        wrapped.status = 4
        wrapped.result.success = True
        future = Future()
        future.set_result(wrapped)
        reasoner._on_execute_rskill_result(
            ExecuteRskillTool(rskill_id=_NAV), 3, b"\x00" * 16, future, None
        )
        assert not second.is_complete(), "a stale success completed the new goal's task"
        assert reasoner._unverified_success is None
    finally:
        rclpy.try_shutdown()
