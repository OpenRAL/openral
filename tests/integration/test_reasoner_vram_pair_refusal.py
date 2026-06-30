"""Live ROS integration test for the ADR-0077 VLA+reward VRAM pair refusal.

A VLA emits no success signal of its own, so it must run with its reward model
resident alongside it (ADR-0074). When the pair does not fit GPU VRAM, the
reasoner must refuse the ``execute_rskill`` dispatch *before* the goal is sent —
publishing a ``FailureTrigger`` (so the reasoner sees it and bounds retries →
handoff) instead of OOMing mid-run or running the policy blind.

This drives a real reasoner node + a real ``ExecuteRskill`` ``ActionServer`` and
asserts that, with a deliberately-too-small GPU budget, the action server is
NEVER called and a ``vram_insufficient`` ``FailureTrigger`` is published. The only
doubles are ``FakeToolUseClient`` at the LLM boundary (CLAUDE.md §1.11) and the
three guard inputs set directly on the node (``__init__`` reads the reward /
gpu-total params at construction, before a test can set them — the param→attr
plumbing is covered live by the ADR-0077 ARMED log).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_vram_pair_refusal.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_SMOLVLA = _REPO_ROOT / "rskills" / "smolvla-libero" / "rskill.yaml"
_ROBOMETER = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"
_VLA_ID = "OpenRAL/rskill-smolvla-libero"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_rskill_refused_when_vla_reward_pair_exceeds_vram() -> None:
    """A VLA whose pair (VLA + reward) exceeds GPU VRAM is refused before dispatch.

    smolvla (1.2 GB bf16) + robometer (3.6 GB int4) = 4.8 GB; with a 4.0 GB budget
    the pair does not fit, so ``_refuse_unfittable_vla`` must:

    1. publish a ``KIND_CONTROLLER`` / ``vram_insufficient`` ``FailureTrigger`` for
       the VLA id, and
    2. NEVER send the goal — the ``ExecuteRskill`` action server's execute
       callback must not run (no OOM, no blind run).
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool, RSkillManifest
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger, PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from opentelemetry import trace as ot_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    vla_manifest = RSkillManifest.from_yaml(_SMOLVLA)
    reward_manifest = RSkillManifest.from_yaml(_ROBOMETER)

    executed: list[float] = []  # execute-callback timestamps — must stay empty
    failures: list[Any] = []  # captured FailureTrigger messages

    # Capture the OTLP span path the live dashboard consumes: the refusal must
    # also emit an ``openral.event.skill_failure`` span event (ADR-0074/0077) so
    # the dashboard's "skill failures" counter tallies it and shows the state.
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    ot_trace.set_tracer_provider(provider)

    rclpy.init()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id=_VLA_ID,
                    prompt="pick up the teapot and put it in the basket",
                    deadline_s=0.0,
                ),
                # Absorb post-refusal tick(s) without erroring.
                *[
                    __import__("openral_core", fromlist=["EmitPromptTool"]).EmitPromptTool(
                        target_topic="/openral/prompt", text="standing by"
                    )
                    for _ in range(4)
                ],
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({_VLA_ID})),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        # ADR-0077 guard inputs (see module docstring — __init__ already read the
        # params, so set the attributes the guard reads directly):
        reasoner._reward_manifest = reward_manifest
        reasoner._gpu_total_vram_gb = 4.0  # < 4.8 GB pair → must refuse
        reasoner._manifest_for_rskill = lambda _rskill_id: vla_manifest  # type: ignore[method-assign]

        # Real ExecuteRskill server — records if a goal ever reaches execute.
        server_node = rclpy.create_node("openral_test_vla_server")

        def _execute(goal_handle: Any) -> Any:
            executed.append(time.monotonic())
            result = ExecuteRskill.Result()
            result.success = True
            result.failure_reason = ""
            result.trace_id = "00-trace-vla"
            goal_handle.succeed()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        # Capture FailureTriggers the reasoner publishes on /openral/failure/rskill.
        sub_node = rclpy.create_node("openral_test_failure_sub")
        fail_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sub_node.create_subscription(
            FailureTrigger, "/openral/failure/rskill", failures.append, fail_qos
        )

        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub_node = rclpy.create_node("openral_test_pair_prompt_pub")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)

        # Let the action server be discovered so the reasoner's server-ready probe
        # passes (the guard runs *after* that probe), then drive a dispatch.
        discover = time.monotonic() + 2.0
        while time.monotonic() < discover:
            executor.spin_once(timeout_sec=0.05)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_pair_prompt_pub"
        prompt.text = "pick up the teapot and put it in the basket"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            if failures:
                break

        # Drain briefly so a just-sent goal (if the guard wrongly let it through)
        # would reach the server's execute callback and be caught.
        drain = time.monotonic() + 1.5
        while time.monotonic() < drain:
            executor.spin_once(timeout_sec=0.05)

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.remove_node(sub_node)
        server_node.destroy_node()
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert failures, (
        "no FailureTrigger published; the ADR-0077 pair check did not refuse the "
        "over-budget VLA dispatch."
    )
    vram_failures = [m for m in failures if "vram_insufficient" in m.evidence_json]
    assert vram_failures, (
        f"a FailureTrigger fired but none was vram_insufficient; "
        f"evidence={[m.evidence_json for m in failures]}"
    )
    assert vram_failures[0].rskill_id == _VLA_ID, (
        f"vram_insufficient failure named the wrong skill: {vram_failures[0].rskill_id!r}"
    )
    # The crux: the goal was NEVER dispatched to the runner.
    assert not executed, (
        f"the VLA goal reached the action server {len(executed)} time(s) despite the "
        "VRAM refusal — the guard must skip the dispatch entirely."
    )
    # The refusal is also mirrored onto the OTLP span path for the dashboard.
    skill_failure_events = [
        ev
        for span in span_exporter.get_finished_spans()
        for ev in span.events
        if ev.name == "openral.event.skill_failure"
    ]
    assert skill_failure_events, (
        "no openral.event.skill_failure span event emitted; the dashboard's skill-"
        "failures counter would never see the ADR-0077 refusal."
    )
    assert any(
        ev.attributes is not None
        and ev.attributes.get("openral.event.skill_failure.state") == "vram_insufficient"
        for ev in skill_failure_events
    ), "skill_failure event fired but carried no vram_insufficient state for the dashboard."
