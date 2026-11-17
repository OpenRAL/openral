"""Live ROS tests for event-driven reasoner ticks (issue #620).

A real ``ReasonerNode`` on a slow timer beat (``tick_hz=0.1``: one beat per 10 s,
so within each test window only events can reach the LLM), real
``openral_msgs`` topics and a real ``ExecuteRskill`` ``ActionServer``; the LLM is
the ``FakeToolUseClient`` (the only double allowed, at the LLM process
boundary, CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live tier::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_reasoner_event_ticks_live.py
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)
_SKILL = "openral/skill-test-events"


def _spin_until(executor: Any, done: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not done():
        executor.spin_once(timeout_sec=0.05)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_a_safety_latch_with_no_failure_trigger_reaches_the_llm() -> None:
    """A dashboard-style e-stop (SafetyStatus only, no FailureTrigger) forces a tick.

    Before #620 the reasoner did not subscribe ``/openral/safety_status``: a latch
    with no skill in flight was invisible to it.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import WaitTool
    from openral_msgs.msg import SafetyStatus
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    try:
        client = FakeToolUseClient(responses=[WaitTool(until="operator", timeout_s=120.0)])
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.1,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        pub_node = rclpy.create_node("openral_test_safety_status_pub")
        pub = pub_node.create_publisher(
            SafetyStatus,
            "/openral/safety_status",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        status = SafetyStatus()
        status.header.stamp = pub_node.get_clock().now().to_msg()
        status.latched = True
        status.drop_reason = SafetyStatus.DROP_EXTERNAL_ESTOP
        status.detail = "dashboard E-STOP"
        pub.publish(status)  # latched: delivered on match, no re-publish loop needed

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(pub_node)
        _spin_until(executor, lambda: client.calls >= 1, 5.0)
        # A 1 Hz-style refresh of the same latch is not a new event.
        status.header.stamp = pub_node.get_clock().now().to_msg()
        pub.publish(status)
        _spin_until(executor, lambda: False, 1.0)
        executor.remove_node(reasoner)
        executor.remove_node(pub_node)
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert client.calls == 1, f"expected exactly one tick for one latch, got {client.calls}"
    context = client.traces[0].context_text
    assert "## SAFETY\nLATCHED reason=drop_external_estop (dashboard E-STOP)" in context, context


def _run_one_goal(*, succeed: bool, responses: list[Any]) -> tuple[Any, list[Any]]:
    """Prompt → execute_rskill → the server succeeds or aborts; spin until it settles."""
    import rclpy
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger, PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=50,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    failures: list[FailureTrigger] = []
    executed = threading.Event()
    rclpy.init()
    try:
        client = FakeToolUseClient(responses=responses)
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.1,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        server_node = rclpy.create_node("openral_test_event_ticks_server")

        def _execute(goal_handle: Any) -> Any:
            result = ExecuteRskill.Result()
            result.success = succeed
            result.failure_reason = "" if succeed else "gripper slipped"
            result.trace_id = "00-trace-events"
            if succeed:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            executed.set()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )
        server_node.create_subscription(
            FailureTrigger, "/openral/failure/rskill", failures.append, qos
        )
        prompt_pub = server_node.create_publisher(PromptStamped, "/openral/prompt", qos)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(server_node)
        deadline = time.monotonic() + 5.0
        while client.calls == 0 and time.monotonic() < deadline:
            prompt = PromptStamped()
            prompt.header.stamp = server_node.get_clock().now().to_msg()
            prompt.header.frame_id = "cli"
            prompt.text = "run the test skill"
            prompt.metadata_json = "{}"
            prompt_pub.publish(prompt)
            _spin_until(executor, lambda: client.calls > 0, 0.3)
        _spin_until(executor, lambda: client.calls >= 2, 5.0)
        _spin_until(executor, lambda: False, 1.5)  # any echo or extra tick lands here
        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
    assert executed.is_set(), "the reasoner never sent the goal"
    return client, failures


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_a_skill_success_reaches_the_llm_without_waiting_for_a_beat() -> None:
    from openral_core import ExecuteRskillTool, WaitTool

    client, _ = _run_one_goal(
        succeed=True,
        responses=[ExecuteRskillTool(rskill_id=_SKILL, prompt="go"), WaitTool(), WaitTool()],
    )
    assert client.calls == 2, f"expected dispatch + one result tick, got {client.calls}"
    assert f"[ok] skill={_SKILL}" in client.traces[1].context_text


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_an_abort_is_one_tick_and_its_echo_is_dropped() -> None:
    """The node's own /openral/failure/rskill publish must not come back as a second tick."""
    from openral_core import ExecuteRskillTool, WaitTool

    client, failures = _run_one_goal(
        succeed=False,
        responses=[ExecuteRskillTool(rskill_id=_SKILL, prompt="go"), WaitTool(), WaitTool()],
    )
    assert failures, "the abort was not published on /openral/failure/rskill"
    assert failures[0].header.frame_id == "openral_reasoner"
    assert client.calls == 2, f"expected dispatch + one failure tick, got {client.calls}"
    context = client.traces[1].context_text
    assert f"[failed] skill={_SKILL}" in context
    failures_section = context.split("## FAILURES\n", 1)[1].split("\n## ", 1)[0]
    assert failures_section.strip() == "(none)", failures_section


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_a_plan_for_a_task_that_ended_mid_call_is_not_sent() -> None:
    """A verdict ends the task while the LLM call is in flight: its execute_rskill is dropped."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_reasoner import MissionState, ToolPalette
    from openral_reasoner.core import PreparedTick
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    try:
        call = ExecuteRskillTool(rskill_id=_SKILL, prompt="pick the cube")
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[call]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.1,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        reasoner._renderer.set_mission(MissionState.from_prompt("pick the cube. then wave"))
        core = reasoner._core
        assert core is not None
        prep = core.prepare_tick(
            world_state=None, renderer=reasoner._renderer, palette=reasoner._palette, force=True
        )
        assert isinstance(prep, PreparedTick)  # force=True never suppresses
        task_key = reasoner._task_key()
        # The reward verdict lands while the call is in flight: task t1 is done.
        reasoner._renderer.advance_mission(done=True, verdict="reward")
        reasoner._finish_llm_tick(
            prep,
            call=core.run_prepared_llm(prep),
            error=None,
            generation=reasoner._llm_generation,
            task_key=task_key,
        )
        rendered = reasoner._renderer.render(world_state=None)
        dispatched = reasoner.dispatched_calls
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert dispatched == ()
    assert "not sent: planned for task t1" in rendered
