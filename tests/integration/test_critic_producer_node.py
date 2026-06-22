"""Live ROS integration test for the Tier-C critic producer (ADR-0064 / audit R3).

Stands up the **real** ``CriticProducerNode`` and ``ReasonerNode`` on one
executor, publishes a stalled ``openral_msgs/CriticScore`` series on
``/openral/critic/score``, and asserts the full path:

    CriticScore (stalled) → CriticWatchdogGroup stall → FailureTrigger
    (KIND_CRITIC / SEVERITY_FAIL) on /openral/failure/critic → reasoner
    forced Tier-C tick → EmitPromptTool dispatched on /openral/prompt.

This is the deterministic counterpart of the live ``deploy sim`` run: it
"enforces a trigger" by feeding below-threshold scores, then checks the new
elements actually fire end-to-end. The only test double is the
``FakeToolUseClient`` at the LLM process boundary (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the sibling reasoner/failure-bus
integration tests. Run with::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest tests/integration/test_critic_producer_node.py \\
        -v -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)

# A real W3C traceparent to assert the critic score's trace_id is propagated
# onto the emitted FailureTrigger (ADR-0064 / observability §6).
_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_stalled_critic_score_drives_tier_c_reasoner_preemption() -> None:
    """A stalled CriticScore stream fires a Tier-C FailureTrigger + reasoner tick."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import EmitPromptTool
    from openral_msgs.msg import CriticScore, FailureTrigger, PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import CriticProducerNode, ReasonerNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    critic_id = "OpenRAL/rskill-robometer-4b"
    threshold = 0.8

    rclpy.init()
    received_prompts: list[PromptStamped] = []
    received_failures: list[FailureTrigger] = []
    prompt_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                EmitPromptTool(
                    target_topic="/openral/prompt",
                    text="critic flagged a task stall; replanning",
                    rationale="Tier-C critic preemption",
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            tick_hz=0.5,  # slow heartbeat so the critic preemption is the only path
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        # The real producer node — default stall_patience=5; the republish loop
        # below feeds well over that many stalled samples.
        producer = CriticProducerNode()

        sub_node = rclpy.create_node("openral_test_subscriber_critic")
        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        score_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def _on_prompt(msg: PromptStamped) -> None:
            if msg.header.frame_id == "openral_reasoner":
                received_prompts.append(msg)
                prompt_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", _on_prompt, prompt_qos)
        sub_node.create_subscription(
            FailureTrigger, "/openral/failure/critic", received_failures.append, failure_qos
        )

        pub_node = rclpy.create_node("openral_test_publisher_critic_score")
        score_pub = pub_node.create_publisher(CriticScore, "/openral/critic/score", score_qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(producer)
        executor.add_node(sub_node)

        def _publish_stalled_score() -> None:
            score = CriticScore()
            score.header.stamp = pub_node.get_clock().now().to_msg()
            score.header.frame_id = critic_id
            score.critic_id = critic_id
            score.score = 0.05  # far below threshold, never improving → a stall
            score.threshold = threshold
            score.trace_id = _TRACEPARENT
            score_pub.publish(score)

        # Republish each spin: VOLATILE durability drops samples published before
        # the producer's subscriber matched; the watchdog's latch bounds the
        # producer to one FailureTrigger per stall regardless of resends.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not prompt_event.is_set():
            _publish_stalled_score()
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(producer)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        producer.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    # 1) The producer turned the stall into a Tier-C FailureTrigger on the bus.
    assert received_failures, (
        "No FailureTrigger landed on /openral/failure/critic — the critic "
        "producer did not fire on the stalled score stream."
    )
    failure = received_failures[-1]
    assert failure.kind == 4, f"expected KIND_CRITIC (4), got {failure.kind}"
    assert failure.severity == 2, f"expected SEVERITY_FAIL (2), got {failure.severity}"
    # 2) The score sample's trace_id propagated onto the failure event.
    assert failure.trace_id == _TRACEPARENT, (
        "critic score trace_id not propagated to FailureTrigger"
    )
    evidence = json.loads(failure.evidence_json)
    assert evidence["kind"] == "critic"
    assert evidence["critic_id"] == critic_id
    assert evidence["threshold"] == pytest.approx(threshold)

    # 3) The reasoner preempted on the critic FAIL (source critic → Tier C in
    #    reasoner_node._FAILURE_TIER_FOR_SOURCE) and dispatched the canned tool.
    assert received_prompts, (
        "Reasoner did not preempt a tick after the critic FailureTrigger — the "
        "Tier-C critic→reasoner path is broken."
    )
    assert received_prompts[-1].text == "critic flagged a task stall; replanning"
