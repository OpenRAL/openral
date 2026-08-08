"""Live ROS integration test for the VLA+reward VRAM pair refusal.

A VLA emits no success signal of its own, so it must run with its reward model
resident alongside it. When the pair does not fit GPU VRAM, the
reasoner must refuse the ``execute_rskill`` dispatch *before* the goal is sent —
publishing a ``FailureTrigger`` (so the reasoner sees it and bounds retries →
handoff) instead of OOMing mid-run or running the policy blind.

This drives a real reasoner node + a real ``ExecuteRskill`` ``ActionServer`` and
asserts that, with a deliberately-too-small GPU budget, the action server is
NEVER called and a ``vram_insufficient`` ``FailureTrigger`` is published. The only
doubles are ``FakeToolUseClient`` at the LLM boundary (CLAUDE.md §1.11) and the
three guard inputs set directly on the node (``__init__`` reads the reward /
gpu-total params at construction, before a test can set them — the param→attr
plumbing is covered live by the VRAM-pair-refusal ARMED log).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite
(``scripts/ros_live_tests.sh``). CI runs it inside ``openral:x86`` (the
``docker-build`` workflow) — this file is the falsification test for issue #46:
a structurally dead ``_refuse_unfittable_vla`` turns it red. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live            # whole suite; `-k <expr>` narrows it
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
_VLA_ID = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"


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
    from openral_core import EmitPromptTool, ExecuteRskillTool, RSkillManifest
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
    # also emit an ``openral.event.skill_failure`` span event so
    # the dashboard's "skill failures" counter tallies it and shows the state.
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    # OTel's global provider is set-once per process; when another live test in
    # the same pytest invocation installed one first, this call would silently
    # no-op and the exporter would never see a span. Same test-only reset as
    # tests/unit/test_reasoner_observability.py / the e2e file.
    ot_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    ot_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
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
                    EmitPromptTool(target_topic="/openral/prompt", text="standing by")
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

        # VRAM-pair-refusal guard inputs (see module docstring — __init__ already read the
        # params, so set the attributes the guard reads directly):
        reasoner._reward_manifest = reward_manifest
        reasoner._gpu_total_vram_gb = 4.0  # < 4.8 GB pair → must refuse
        # Prime the manifest cache the way `_seed_palette` does, rather than
        # replacing `_manifest_for_rskill` itself.
        #
        # This line used to read:
        #     reasoner._manifest_for_rskill = lambda _rskill_id: vla_manifest
        # which stubbed out the very method that was broken in production, so
        # the test could only ever exercise the refusal *arithmetic*. It could
        # not see that the real lookup returned `None` — it consulted the
        # install registry (`~/.local/share/openral/rskills.json`), which a
        # search-path-seeded palette never populates — and that `None` makes
        # `_refuse_unfittable_vla` return `False` and let the dispatch through.
        # A structurally dead gate passed this test for as long as it existed;
        # a real deploy caught it on 2026-08-04 when molmoact2 (4.0 + 5.5 GB on
        # an 8 GB card) was dispatched anyway and reported as a 20 s timeout.
        # Injecting at the cache keeps the production lookup in the assertion.
        reasoner._manifests_by_id[_VLA_ID] = vla_manifest
        assert reasoner._manifest_for_rskill(_VLA_ID) is vla_manifest

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
        "no FailureTrigger published; the VRAM pair check did not refuse the "
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
        "failures counter would never see the VRAM pair refusal."
    )
    assert any(
        ev.attributes is not None
        and ev.attributes.get("openral.event.skill_failure.state") == "vram_insufficient"
        for ev in skill_failure_events
    ), "skill_failure event fired but carried no vram_insufficient state for the dashboard."


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_rskill_refused_when_live_free_vram_is_below_vla_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier-1 gate: a VLA whose declared footprint exceeds *currently free* VRAM
    is refused before dispatch, even with NO reward model wired.

    The static pair check budgets against the card's TOTAL and is blind to
    other processes — observed live (2026-07-20): an external vLLM server held
    4.7 GB of an 8 GB card, molmoact2 (declared 4.0 GB) passed every static
    gate and burned ~30 s in a CUDA OOM abort. Here the probe is pinned to
    0.5 GB free vs smolvla's 1.2 GB declaration: the goal must never reach the
    action server and a ``vram_insufficient`` FailureTrigger must fire.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import EmitPromptTool, ExecuteRskillTool, RSkillManifest
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger, PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_reasoner_ros import reasoner_node as rn_module
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
    assert (vla_manifest.active_min_vram_gb() or 0.0) > 0.5, "fixture must declare > 0.5 GB"

    # Pin the live probe: only 0.5 GB free right now.
    monkeypatch.setattr(rn_module, "_detect_gpu_free_vram_gb", lambda: 0.5)

    executed: list[float] = []
    failures: list[Any] = []

    rclpy.init()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id=_VLA_ID,
                    prompt="pick up the teapot and put it in the basket",
                    deadline_s=0.0,
                ),
                *[
                    EmitPromptTool(target_topic="/openral/prompt", text="standing by")
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
        # NO reward manifest — the free-VRAM tier must fire on its own.
        assert reasoner._reward_manifest is None
        reasoner._manifest_for_rskill = lambda _rskill_id: vla_manifest  # type: ignore[method-assign]  # reason: inject fixture manifest at the guard's seam

        server_node = rclpy.create_node("openral_test_vla_server_free_vram")

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

        sub_node = rclpy.create_node("openral_test_failure_sub_free_vram")
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
        pub_node = rclpy.create_node("openral_test_free_vram_prompt_pub")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)

        discover = time.monotonic() + 2.0
        while time.monotonic() < discover:
            executor.spin_once(timeout_sec=0.05)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_free_vram_prompt_pub"
        prompt.text = "pick up the teapot and put it in the basket"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            if failures:
                break
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

    vram_failures = [m for m in failures if "vram_insufficient" in m.evidence_json]
    assert vram_failures, (
        f"no vram_insufficient FailureTrigger; the live free-VRAM gate did not refuse "
        f"(failures={[m.evidence_json for m in failures]})"
    )
    assert "free right now" in vram_failures[0].evidence_json
    assert not executed, "the goal reached the action server despite the free-VRAM refusal"
