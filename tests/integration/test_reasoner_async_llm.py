"""Acceptance tests for the async LLM phase (issue #21).

A blocking ``select_tool`` used to run on the single-threaded rclpy
executor, so every other callback — action goal results, deadline
("patience") timers, Tier-A safety preemptions — queued behind the LLM
round-trip (live evidence: a 10 s patience timer fired at 76 s). The LLM
phase now runs on a worker thread; these tests pin the acceptance
criteria from the issue with an artificially slow
:class:`FakeToolUseClient`:

(a) a goal result callback lands < 1 s after the server terminates, while
    an LLM call is still in flight;
(b) a skill deadline timer fires within 1 s of its deadline, while an LLM
    call is still in flight;
(c) a Tier-A safety failure is ingested during the in-flight call and its
    forced tick runs immediately after that call returns;
plus a bounded dispatch/abort soak over the multithreaded executor.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like
``test_reasoner_node_end_to_end.py`` (rclpy + DDS init clash with glib
pulled in by torch/pyarrow in the regular pytest invocation)::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest tests/integration/test_reasoner_async_llm.py \\
        -v -p no:launch_testing -p no:launch_ros
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

#: Artificial LLM round-trip. Long enough that a starved callback is
#: unmistakable (the pre-#21 sync design delays it by exactly this much),
#: short enough to keep the suite tight. The < 1 s acceptance bounds
#: below are only meaningful because this is ≥ 3×.
_SLOW_LLM_S = 3.0


def _prompt_qos() -> Any:
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _publish_prompt(pub_node: Any, pub: Any, text: str) -> None:
    from openral_msgs.msg import PromptStamped

    msg = PromptStamped()
    msg.header.stamp = pub_node.get_clock().now().to_msg()
    msg.header.frame_id = "openral_test_async_llm"
    msg.text = text
    msg.metadata_json = "{}"
    pub.publish(msg)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_goal_result_lands_while_llm_call_is_in_flight() -> None:
    """(a) A goal result is processed < 1 s after the server succeeds, mid-LLM-call."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool, WaitTool
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    slow_started = threading.Event()
    slow_finished = threading.Event()
    server_succeeded_at: list[float] = []
    dispatched = threading.Event()

    def _selector(_context: str, _palette: Any) -> Any:
        if not dispatched.is_set():
            dispatched.set()
            return ExecuteRskillTool(
                rskill_id="openral/skill-async-a",
                prompt="succeed while I think",
                deadline_s=0.0,
            )
        slow_started.set()
        time.sleep(_SLOW_LLM_S)
        slow_finished.set()
        return WaitTool(rationale="observing the in-flight goal")

    rclpy.init()
    try:
        client = FakeToolUseClient(selector=_selector)
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({"openral/skill-async-a"})),
            tick_hz=5.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_async_server_a")

        def _execute(goal_handle: Any) -> Any:
            # Blocks its own (reentrant-group) thread until the slow LLM
            # call is definitely in flight, THEN succeeds — the reasoner's
            # result callback must not have to wait out the LLM call.
            slow_started.wait(timeout=10.0)
            result = ExecuteRskill.Result()
            result.success = True
            result.failure_reason = ""
            result.trace_id = ""
            goal_handle.succeed()
            server_succeeded_at.append(time.monotonic())
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        from openral_msgs.msg import PromptStamped

        pub_node = rclpy.create_node("openral_test_async_prompt_a")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", _prompt_qos())

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(reasoner)
        executor.add_node(server_node)

        _publish_prompt(pub_node, prompt_pub, "run the async success skill")

        result_processed_at: float | None = None
        llm_still_in_flight_at_result = False
        deadline = time.monotonic() + _SLOW_LLM_S + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            # The in-flight flag flip IS the result-callback observable.
            if server_succeeded_at and not reasoner._rskill_inflight:
                result_processed_at = time.monotonic()
                llm_still_in_flight_at_result = not slow_finished.is_set()
                break

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        pub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert server_succeeded_at, "action server never reached succeed() — dispatch broke"
    assert result_processed_at is not None, (
        "goal result was never processed — result callback starved or lost"
    )
    latency = result_processed_at - server_succeeded_at[0]
    assert latency < 1.0, (
        f"goal result took {latency:.2f}s to process after server success — "
        f"executor starved by the in-flight LLM call (pre-#21 behavior)"
    )
    assert llm_still_in_flight_at_result, (
        "the slow LLM call had already finished when the result was processed — "
        "the test did not actually measure mid-flight latency; raise _SLOW_LLM_S"
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_deadline_timer_fires_on_time_while_llm_call_is_in_flight() -> None:
    """(b) The skill deadline ("patience") timer fires within 1 s of its deadline, mid-LLM-call."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool, WaitTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer, CancelResponse
    from rclpy.action.server import GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    deadline_s = 1.0
    slow_finished = threading.Event()
    goal_started_at: list[float] = []
    timeout_seen = threading.Event()
    timeout_seen_at: list[float] = []
    dispatched = threading.Event()

    def _selector(_context: str, _palette: Any) -> Any:
        if not dispatched.is_set():
            dispatched.set()
            return ExecuteRskillTool(
                rskill_id="openral/skill-async-b",
                prompt="stall past my deadline",
                deadline_s=deadline_s,
            )
        time.sleep(_SLOW_LLM_S)
        slow_finished.set()
        return WaitTool(rationale="waiting out the stall")

    rclpy.init()
    try:
        client = FakeToolUseClient(selector=_selector)
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({"openral/skill-async-b"})),
            tick_hz=5.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_async_server_b")

        def _execute(goal_handle: Any) -> Any:
            goal_started_at.append(time.monotonic())
            hard_stop = time.monotonic() + _SLOW_LLM_S + 8.0
            while time.monotonic() < hard_stop and not goal_handle.is_cancel_requested:
                time.sleep(0.05)
            result = ExecuteRskill.Result()
            result.success = False
            result.trace_id = ""
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.failure_reason = "canceled by reasoner deadline"
            else:
                goal_handle.abort()
                result.failure_reason = "test hard stop"
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        sub_node = rclpy.create_node("openral_test_async_failsub_b")

        def _on_failure(msg: FailureTrigger) -> None:
            if msg.kind == 0 and not timeout_seen.is_set():  # KIND_TIMEOUT
                timeout_seen_at.append(time.monotonic())
                timeout_seen.set()

        sub_node.create_subscription(
            FailureTrigger, "/openral/failure/rskill", _on_failure, _prompt_qos()
        )

        from openral_msgs.msg import PromptStamped

        pub_node = rclpy.create_node("openral_test_async_prompt_b")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", _prompt_qos())

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)

        _publish_prompt(pub_node, prompt_pub, "run the stalling skill")

        deadline = time.monotonic() + _SLOW_LLM_S + 8.0
        while time.monotonic() < deadline and not timeout_seen.is_set():
            executor.spin_once(timeout_sec=0.05)
        llm_still_in_flight_at_timeout = not slow_finished.is_set()

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.remove_node(sub_node)
        pub_node.destroy_node()
        sub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert goal_started_at, "action server never started the goal — dispatch broke"
    assert timeout_seen.is_set(), "KIND_TIMEOUT never published — deadline timer starved or lost"
    fired_after = timeout_seen_at[0] - goal_started_at[0]
    assert fired_after < deadline_s + 1.0, (
        f"deadline timer fired {fired_after:.2f}s after goal start (deadline {deadline_s}s) — "
        f"timer starved by the in-flight LLM call (live bug: 10 s patience fired at 76 s)"
    )
    assert llm_still_in_flight_at_timeout, (
        "the slow LLM call had already finished when the timeout fired — "
        "the test did not actually measure mid-flight latency; raise _SLOW_LLM_S"
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_tier_a_failure_is_ingested_mid_call_and_ticks_right_after() -> None:
    """(c) A Tier-A safety WARN lands during the in-flight call; its forced tick runs
    immediately after that call returns (coalesced, not starved for its duration)."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import WaitTool
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.executors import MultiThreadedExecutor

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    call_started_at: list[float] = []
    call_finished_at: list[float] = []
    first_call_started = threading.Event()

    def _selector(_context: str, _palette: Any) -> Any:
        call_started_at.append(time.monotonic())
        first_call_started.set()
        if len(call_started_at) == 1:
            time.sleep(_SLOW_LLM_S)
        call_finished_at.append(time.monotonic())
        return WaitTool(rationale=f"tick {len(call_started_at)}")

    rclpy.init()
    try:
        client = FakeToolUseClient(selector=_selector)
        reasoner = ReasonerNode(
            client=client,
            # Non-empty palette so heartbeat ticks are not palette_empty-suppressed.
            palette=ToolPalette(execute_rskill_ids=frozenset({"openral/skill-async-c"})),
            tick_hz=5.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        from openral_msgs.msg import PromptStamped

        pub_node = rclpy.create_node("openral_test_async_pub_c")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", _prompt_qos())
        failure_pub = pub_node.create_publisher(
            FailureTrigger, "/openral/failure/safety", _prompt_qos()
        )

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(reasoner)

        _publish_prompt(pub_node, prompt_pub, "start thinking")

        # Spin until the slow call is in flight, then publish the Tier-A WARN.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not first_call_started.is_set():
            executor.spin_once(timeout_sec=0.05)
        assert first_call_started.is_set(), "first (slow) LLM call never started"

        fail = FailureTrigger()
        fail.header.stamp = pub_node.get_clock().now().to_msg()
        fail.kind = 1
        fail.severity = 1  # SEVERITY_WARN — Tier-A preempts on WARN for source=safety
        fail.evidence_json = ""
        fail.rskill_id = ""
        fail.trace_id = ""

        ingested_mid_call_at: list[float] = []
        deadline = time.monotonic() + _SLOW_LLM_S + 8.0
        while time.monotonic() < deadline and len(call_finished_at) < 2:
            failure_pub.publish(fail)  # republish: VOLATILE + discovery is best-effort
            executor.spin_once(timeout_sec=0.05)
            # Renderer ingest is the executor-liveness observable.
            if not ingested_mid_call_at and reasoner._renderer.failures:
                ingested_mid_call_at.append(time.monotonic())

        executor.remove_node(reasoner)
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert ingested_mid_call_at, "safety failure never reached the renderer"
    assert len(call_started_at) >= 2, (
        "no follow-up tick after the slow call — Tier-A forced tick lost"
    )
    # The failure callback must have run WHILE the slow call was in flight
    # (executor not starved)…
    assert ingested_mid_call_at[0] < call_finished_at[0], (
        "safety failure was only ingested after the LLM call returned — executor starved"
    )
    # …and its coalesced forced tick must run immediately after that call.
    gap = call_started_at[1] - call_finished_at[0]
    assert gap < 1.0, (
        f"forced Tier-A tick started {gap:.2f}s after the in-flight LLM call returned — "
        "queued-replay trampoline broke"
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_async_llm_soak_dispatch_abort_cycles() -> None:
    """Bounded soak: repeated dispatch/abort cycles with a delayed client, no stuck state.

    Every goal aborts server-side; the failure feedback forces the next
    tick, which redispatches (unique prompt per attempt so the retry cap
    never suppresses). The pass criterion is throughput: several full
    cycles complete and the single-flight window is released at the end —
    a starved/stuck trampoline stalls the cycle count instead.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    aborted = threading.Semaphore(0)
    attempts: list[int] = [0]

    def _selector(_context: str, _palette: Any) -> Any:
        attempts[0] += 1
        return ExecuteRskillTool(
            rskill_id="openral/skill-async-soak",
            prompt=f"attempt {attempts[0]}",  # unique identity per cycle — retry cap stays quiet
            deadline_s=0.0,
        )

    rclpy.init()
    try:
        client = FakeToolUseClient(selector=_selector, delay_s=0.15)
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({"openral/skill-async-soak"})),
            tick_hz=10.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_async_server_soak")

        def _execute(goal_handle: Any) -> Any:
            result = ExecuteRskill.Result()
            result.success = False
            result.failure_reason = "soak abort"
            result.trace_id = ""
            goal_handle.abort()
            aborted.release()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        from openral_msgs.msg import PromptStamped

        pub_node = rclpy.create_node("openral_test_async_prompt_soak")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", _prompt_qos())

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(reasoner)
        executor.add_node(server_node)

        _publish_prompt(pub_node, prompt_pub, "soak the dispatch loop")

        cycles = 0
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and cycles < 5:
            executor.spin_once(timeout_sec=0.05)
            while aborted.acquire(blocking=False):
                cycles += 1

        # Let any in-flight tick settle so the release path is exercised.
        settle = time.monotonic() + 2.0
        while time.monotonic() < settle:
            executor.spin_once(timeout_sec=0.05)
        # Single-flight release is the stuck-state observable.
        tick_released = not reasoner._tick_in_flight or not reasoner._rskill_inflight

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        pub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert cycles >= 5, (
        f"only {cycles} dispatch/abort cycles in 15 s with a 0.15 s LLM delay — "
        "the async tick loop is stalling"
    )
    assert tick_released, "single-flight tick window never released after the soak"
