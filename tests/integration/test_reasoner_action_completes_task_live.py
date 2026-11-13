"""Live ROS test: a wrapped action's own success settles the reasoner's mission task.

Reproduces the 2026-11-11 OpenAMRobot run (Isaac, claude-sonnet-5-5): "drive to x=4,y=0
in map" dispatched the Nav2 ``navigate_to_pose`` rSkill, Nav2 SUCCEEDED at x=3.79 (inside
its 0.25 m goal tolerance), and the reasoner, seeing only ``trace=…`` and with no reward
monitor to complete the task, re-issued the goal twice and never marked the task done; a
second prompt ran as an aside after which the first task resumed. Covers:

1. The success's ``ExecuteRskill.Result.outcome_summary`` reaches the ``## EXECUTION`` line,
   and the Nav2 skill's ``completes_task_on_success`` completes the task.
2. With no reward monitor, the per-task attempt cap counts *failed* attempts (after its one
   subdivision offer it abandons); successes do not trip it, and the hard dispatch bound
   still terminates a task that keeps succeeding without ever completing (#562).
3. A success nobody can verify lets the next operator prompt replace the mission.
4. A Nav2 success that is one step of a longer task (``final_step=False``) completes only
   its step (#562).
5. With a reward monitor, the reward gate's 3-attempt ladder counts only the dispatches it
   judges not done: approach steps it never judges do not spend it, and a task fed
   successes forever still terminates. The mission summary the dashboard reads carries
   that failure count next to the dispatch count.

Real reasoner node, real ``openral_msgs`` result types, the real in-tree Nav2 / VLA
manifests and ``panda_mobile`` robot manifest (CLAUDE.md §1.11); the only double is the
scripted LLM client ``on_configure`` needs. A reward verdict is the real
``QueryTaskProgress.Response`` handed to the node's real response callback, as an action
result is handed to ``_on_execute_rskill_result``. Gated on ``OPENRAL_TEST_ROS_LIVE=1``
(``scripts/ros_live_tests.sh``); locally, after ``just ros2-build && source
install/setup.bash``, run ``just test-ros-live -k action_completes``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import MissionState

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


def _settle(
    reasoner: Any,  # reason: ReasonerNode imports only with ROS sourced
    call: ExecuteRskillTool,
    *,
    succeeded: bool,
) -> None:
    mission = reasoner._renderer.mission
    mission.record_attempt(rskill_id=call.rskill_id)
    reasoner._maybe_verify_active_mission_task(call, traceparent=None, succeeded=succeeded)


def _status(mission: MissionState) -> str:
    # Read through the summary: mypy keeps the attribute narrowed to "active".
    summary = mission.to_summary()["tasks"]
    assert isinstance(summary, list)
    return str(summary[0]["status"])


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
            _settle(reasoner, call, succeeded=False)
            assert mission.tasks[0].status == "active", "below the cap the LLM decides"
        _settle(reasoner, call, succeeded=False)
        # At the cap: one subdivision offer first (the existing abandon path) ...
        assert mission.tasks[0].task_id in reasoner._subdivide_offered
        assert mission.tasks[0].status == "active"
        # ... then the next failed attempt abandons it.
        _settle(reasoner, call, succeeded=False)
        assert _status(mission) == "abandoned"
        assert mission.is_complete()
    finally:
        rclpy.try_shutdown()


def test_successful_steps_do_not_abandon_a_multi_skill_task_regression_562() -> None:
    """#562: with no reward monitor, three successful dispatches (pick, carry, place)
    used to hit the attempt cap: subdivision offer, then abandonment of a task whose
    every step had worked. The task now stays active for the LLM."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_ATTEMPTS, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        call = ExecuteRskillTool(rskill_id=vla)
        mission = MissionState.from_prompt("pick up the cube and put it in the bin")
        reasoner._renderer.set_mission(mission)
        for _ in range(DEFAULT_MAX_ATTEMPTS + 1):
            _settle(reasoner, call, succeeded=True)
        assert reasoner._subdivide_offered == set(), "a success is not a failed attempt"
        assert _status(mission) == "active"
        assert reasoner._unverified_success is not None, "the operator's next prompt is a goal"
    finally:
        rclpy.try_shutdown()


def test_a_task_that_keeps_succeeding_still_terminates_regression_562() -> None:
    """The ladder stays bounded: successes nobody can judge, sent forever, end in the
    abandon path at ``DEFAULT_MAX_TASK_DISPATCHES`` (one subdivision offer first)."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_TASK_DISPATCHES, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        call = ExecuteRskillTool(rskill_id=vla)
        mission = MissionState.from_prompt("pick up the cube and put it in the bin")
        reasoner._renderer.set_mission(mission)
        dispatches = 0
        while not mission.is_complete():
            assert dispatches < 4 * DEFAULT_MAX_TASK_DISPATCHES, "unbounded success loop"
            _settle(reasoner, call, succeeded=True)
            dispatches += 1
            if dispatches < DEFAULT_MAX_TASK_DISPATCHES:
                assert reasoner._subdivide_offered == set()
        assert dispatches == DEFAULT_MAX_TASK_DISPATCHES + 1  # bound → offer, then abandon
        assert _status(mission) == "abandoned"
    finally:
        rclpy.try_shutdown()


def test_the_dispatch_bound_holds_when_the_reward_monitor_is_off_the_graph() -> None:
    """A configured reward monitor whose service never comes up gives no verdict; the
    dispatch bound still ends the task instead of leaving it active forever."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_TASK_DISPATCHES, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        reasoner._task_progress_available = True  # configured, but nothing serves it
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        call = ExecuteRskillTool(rskill_id=vla)
        mission = MissionState.from_prompt("pick up the cube and put it in the bin")
        reasoner._renderer.set_mission(mission)
        dispatches = 0
        while not mission.is_complete():
            assert dispatches < 2 * DEFAULT_MAX_TASK_DISPATCHES, "unbounded without a verdict"
            _settle(reasoner, call, succeeded=True)
            dispatches += 1
        assert dispatches == DEFAULT_MAX_TASK_DISPATCHES + 1  # bound → offer, then abandon
        assert _status(mission) == "abandoned"
        assert mission.tasks[0].failures == 0, "a success with no verdict is not a failure"
    finally:
        rclpy.try_shutdown()


def test_nav2_success_as_one_step_completes_only_that_step_regression_562() -> None:
    """#562: "drive to the kitchen and pick the cup" — Nav2's success used to complete the
    whole task, so the pick never ran. Sent as a step (``final_step=False``) it completes
    only its step; the task stays active for the pick, the operator's next prompt is not
    taken for a new goal, and the final step's success still completes the task."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner.mission import MissionState
    from rclpy.task import Future

    rclpy.init()
    try:
        reasoner = _reasoner()
        mission = MissionState.from_prompt("drive to the kitchen and pick up the cup")
        reasoner._renderer.set_mission(mission)
        mission.record_attempt(rskill_id=_NAV)
        reasoner._active_dispatch_generation = 9
        reasoner._active_dispatch_completes_task = True  # captured from the manifest
        wrapped = ExecuteRskill.Impl.GetResultService.Response()
        wrapped.status = 4  # STATUS_SUCCEEDED
        wrapped.result.success = True
        future = Future()
        future.set_result(wrapped)
        reasoner._on_execute_rskill_result(
            ExecuteRskillTool(rskill_id=_NAV, final_step=False), 9, b"\x00" * 16, future, None
        )
        assert _status(mission) == "active", "a navigation step completed the whole task"
        assert reasoner._unverified_success is None
        assert reasoner._renderer.executions[-1].outcome == "ok"
        # The task's last step (here another drive, the only skill this robot has with an
        # own-success opt-in) still completes it.
        _settle(reasoner, ExecuteRskillTool(rskill_id=_NAV), succeeded=True)
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


def _judge(
    reasoner: Any,  # reason: ReasonerNode imports only with ROS sourced
    call: ExecuteRskillTool,
    *,
    progress: float,
    succeeded: bool = True,
    scored: bool = True,
) -> None:
    """One dispatch the reward gate judges: it returns, then the monitor's
    ``QueryTaskProgress`` reply reaches the node's response callback.

    A failed dispatch returns through ``_maybe_verify_active_mission_task``, which counts it.
    No monitor serves the query here, so the node then takes its off-graph branch, which
    settles nothing below the dispatch bound. A successful one only marks the task verifying
    on return. ``scored=False`` is the monitor's no-score reply (stale frames, no task, a
    backend error)."""
    from openral_msgs.srv import QueryTaskProgress
    from rclpy.task import Future

    mission = reasoner._renderer.mission
    mission.record_attempt(rskill_id=call.rskill_id)
    if succeeded:
        mission.mark_verifying()
    else:
        reasoner._maybe_verify_active_mission_task(call, traceparent=None, succeeded=False)
    active = mission.active()
    response = QueryTaskProgress.Response()
    response.ok = scored
    response.stale = not scored
    response.progress_now = progress if scored else 0.0
    response.success_now = progress if scored else 0.0
    response.frames_seen = 30 if scored else 0
    future = Future()
    future.set_result(response)
    reasoner._on_mission_verify_response(
        active.task_id,
        active.text,
        future,
        mission=mission,
        traceparent=None,
        succeeded=succeeded,
    )


_NOT_DONE = 0.2  # below the reward gate's check floor: the ladder, no VLM adjudication


def test_approach_steps_do_not_spend_the_reward_gates_failure_budget() -> None:
    """#562 class, reward gate: "drive to the kitchen and pick up the cup" with a reward
    monitor. Two approach drives (``final_step=False``) succeed without the gate judging
    them; they must not use up the gate's three tries of the pick."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_ATTEMPTS, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        reasoner._task_progress_available = True  # a reward monitor judges the task
        pick = ExecuteRskillTool(
            rskill_id=next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        )
        mission = MissionState.from_prompt("drive to the kitchen and pick up the cup")
        reasoner._renderer.set_mission(mission)
        for _ in range(2):
            _settle(reasoner, ExecuteRskillTool(rskill_id=_NAV, final_step=False), succeeded=True)
        assert _status(mission) == "active"
        picks = 0
        while not reasoner._subdivide_offered:
            assert picks < 2 * DEFAULT_MAX_ATTEMPTS, "the gate never ended the pick"
            _judge(reasoner, pick, progress=_NOT_DONE)
            picks += 1
        assert picks == DEFAULT_MAX_ATTEMPTS, f"the pick got {picks} of its 3 tries"
        active = mission.active()
        assert active is not None and active.failures == DEFAULT_MAX_ATTEMPTS
    finally:
        rclpy.try_shutdown()


@pytest.mark.parametrize(
    ("pattern", "dispatches"),
    [
        # Steps alone: the dispatch bound (10), its one subdivision offer, then abandon.
        (("step",), 11),
        # Every result judged not done: three failures, the offer, then abandon.
        (("judged",), 4),
        # Interleaved: the steps spend no failure budget, the judged results still end it.
        (("step", "judged"), 8),
    ],
)
def test_successes_fed_forever_terminate_under_the_reward_gate(
    pattern: tuple[str, ...], dispatches: int
) -> None:
    """The ladder stays bounded with a reward monitor: dispatches that keep succeeding
    (approach steps, or VLA runs the gate judges not done) end in the abandon path."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_TASK_DISPATCHES, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        reasoner._task_progress_available = True
        pick = ExecuteRskillTool(
            rskill_id=next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        )
        drive = ExecuteRskillTool(rskill_id=_NAV, final_step=False)
        mission = MissionState.from_prompt("drive to the kitchen and pick up the cup")
        reasoner._renderer.set_mission(mission)
        sent = 0
        while not mission.is_complete():
            assert sent < 4 * DEFAULT_MAX_TASK_DISPATCHES, "unbounded success loop"
            if pattern[sent % len(pattern)] == "step":
                _settle(reasoner, drive, succeeded=True)
            else:
                _judge(reasoner, pick, progress=_NOT_DONE)
            sent += 1
        assert _status(mission) == "abandoned"
        assert sent == dispatches
    finally:
        rclpy.try_shutdown()


def test_mission_summary_counts_failures_not_dispatches() -> None:
    """The dashboard's per-task "n/3" reads the summary: a healthy task with five
    successful dispatches and one failure is 1 failure of 3, not "6/3"."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import (
        DEFAULT_MAX_ATTEMPTS,
        DEFAULT_MAX_TASK_DISPATCHES,
        MissionState,
    )

    rclpy.init()
    try:
        reasoner = _reasoner()
        vla = next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        call = ExecuteRskillTool(rskill_id=vla)
        mission = MissionState.from_prompt("pick up the cube and put it in the bin")
        reasoner._renderer.set_mission(mission)
        for succeeded in (True, True, False, True, True, True):
            _settle(reasoner, call, succeeded=succeeded)
        summary = mission.to_summary()
        assert summary["max_attempts"] == DEFAULT_MAX_ATTEMPTS
        assert summary["max_dispatches"] == DEFAULT_MAX_TASK_DISPATCHES
        tasks = summary["tasks"]
        assert isinstance(tasks, list)
        assert (tasks[0]["attempts"], tasks[0]["failures"]) == (6, 1)
        assert tasks[0]["status"] == "active"
    finally:
        rclpy.try_shutdown()


def test_a_failed_dispatch_spends_one_try_of_the_reward_gate() -> None:
    """#586 review: a failed pick is counted once, when it returns, not again when the gate
    judges it: three failed picks reach the subdivision offer, not two."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_ATTEMPTS, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        reasoner._task_progress_available = True
        pick = ExecuteRskillTool(
            rskill_id=next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        )
        mission = MissionState.from_prompt("pick up the cup")
        reasoner._renderer.set_mission(mission)
        picks = 0
        while not reasoner._subdivide_offered:
            assert picks < 2 * DEFAULT_MAX_ATTEMPTS, "the gate never ended the pick"
            _judge(reasoner, pick, progress=_NOT_DONE, succeeded=False)
            picks += 1
            active = mission.active()
            assert active is not None and active.failures == picks
        assert picks == DEFAULT_MAX_ATTEMPTS
    finally:
        rclpy.try_shutdown()


def test_a_success_the_reward_monitor_could_not_score_is_not_a_failure() -> None:
    """#586 review: ``ok=False`` (stale frames, no task, a backend error) is no verdict. A
    successful dispatch it could not score spends no try, as with no reward monitor (#562),
    and the dispatch bound still ends a task fed such successes forever."""
    rclpy = pytest.importorskip("rclpy")
    from openral_core import ExecuteRskillTool
    from openral_reasoner.mission import DEFAULT_MAX_TASK_DISPATCHES, MissionState

    rclpy.init()
    try:
        reasoner = _reasoner()
        reasoner._task_progress_available = True
        look = ExecuteRskillTool(
            rskill_id=next(s.rskill_id for s in reasoner._palette.skills if s.rskill_id != _NAV)
        )
        mission = MissionState.from_prompt("look at the shelf and pick up the cup")
        reasoner._renderer.set_mission(mission)
        sent = 0
        while not mission.is_complete():
            assert sent < 4 * DEFAULT_MAX_TASK_DISPATCHES, "unbounded unscored successes"
            _judge(reasoner, look, progress=_NOT_DONE, scored=False)
            sent += 1
            if sent < DEFAULT_MAX_TASK_DISPATCHES:
                assert reasoner._subdivide_offered == set(), f"offered after {sent} successes"
        assert sent == DEFAULT_MAX_TASK_DISPATCHES + 1  # bound → offer, then abandon
        assert mission.tasks[0].failures == 0
        assert _status(mission) == "abandoned"
    finally:
        rclpy.try_shutdown()
