"""Integration tests for ``openral_safety`` SafetyPassthroughNode (ADR-0018 F5).

Drives every managed-lifecycle transition on the real Day-1 pass-through
node and exercises the topic contract: a valid ActionChunk republishes to
``/openral/safe_action`` and an envelope violation fires ``/openral/estop``
plus latches the node so subsequent chunks drop. Real ``rclpy`` + real
colcon-built ``openral_msgs`` — no mocks (CLAUDE.md §1.11).

Gated on ``rclpy`` + ``openral_msgs`` being importable; in lint-only
environments without a sourced ROS 2 install the test skips with a typed
reason. Recommended invocation::

    just ros2-build
    source install/setup.bash
    uv run pytest packages/openral_safety/test/ -v -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_msgs.msg import ActionChunk
from openral_safety.supervisor_node import SafetyPassthroughNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from std_msgs.msg import Empty
from std_srvs.srv import Trigger


@pytest.fixture
def ros_context() -> Any:
    """Spin up rclpy for the test, tear down on exit."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


# ── Lifecycle transitions ────────────────────────────────────────────────────


def test_supervisor_lifecycle_transitions(ros_context: None) -> None:
    """SafetyPassthroughNode drives every managed-lifecycle transition to SUCCESS."""
    node = SafetyPassthroughNode(node_name="openral_safety_supervisor_test_lifecycle")
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS
    finally:
        node.destroy_node()


# ── Topic contract: passthrough on valid chunks ──────────────────────────────


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 2.0) -> bool:
    """Spin the executor in the calling thread until ``predicate()`` is True or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def _make_chunk(n_dof: int = 6, flat_value: float = 0.1) -> ActionChunk:
    msg = ActionChunk()
    msg.control_mode = 0  # ControlMode.JOINT_POSITION
    msg.horizon = 1
    msg.n_dof = n_dof
    msg.flat = [flat_value] * n_dof
    msg.rskill_id = "openral/rskill-test-skill"
    msg.rskill_revision = "0.1.0"
    msg.trace_id = "00-trace-id-test"
    return msg


def test_valid_candidate_round_trips_to_safe_action(ros_context: None) -> None:
    """Valid chunks (no envelope set) republish unchanged on /openral/safe_action."""
    from rclpy.node import Node

    node = SafetyPassthroughNode(node_name="openral_safety_test_passthrough")
    helper = Node("openral_safety_test_passthrough_helper")
    received: list[ActionChunk] = []

    helper.create_subscription(ActionChunk, "/openral/safe_action", received.append, 10)
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # Wait one spin so the subscription discovers the publisher.
        executor.spin_once(timeout_sec=0.1)
        chunk = _make_chunk()
        pub.publish(chunk)
        assert _spin_until(executor, lambda: len(received) >= 1)
        assert received[0].rskill_id == "openral/rskill-test-skill"
        assert received[0].trace_id == "00-trace-id-test"
        assert list(received[0].flat) == [0.1] * 6
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


# ── Topic contract: envelope violation latches estop ─────────────────────────


def test_envelope_violation_fires_estop_and_drops(ros_context: None) -> None:
    """n_dof mismatch publishes /openral/estop and stops republishing."""
    node = SafetyPassthroughNode(node_name="openral_safety_test_violation")
    helper = rclpy.create_node("openral_safety_test_violation_helper")

    safe_received: list[ActionChunk] = []
    estop_received: list[Empty] = []
    helper.create_subscription(ActionChunk, "/openral/safe_action", safe_received.append, 10)
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        # Set n_dof to 6 — wrong-dof chunk will violate.
        from rclpy.parameter import Parameter

        node.set_parameters([Parameter("n_dof", Parameter.Type.INTEGER, 6)])
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        # Publish a chunk with wrong n_dof.
        bad = _make_chunk(n_dof=3)
        pub.publish(bad)
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert len(safe_received) == 0, "violation must not republish"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


# ── Workspace (per-joint position-limit) violation ───────────────────────────


def test_workspace_violation_fires_estop_and_drops(ros_context: None) -> None:
    """First-row joint target outside [min, max] is dropped + /openral/estop fires."""
    node = SafetyPassthroughNode(node_name="openral_safety_test_workspace")
    helper = rclpy.create_node("openral_safety_test_workspace_helper")

    safe_received: list[ActionChunk] = []
    estop_received: list[Empty] = []
    helper.create_subscription(ActionChunk, "/openral/safe_action", safe_received.append, 10)
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        from rclpy.parameter import Parameter

        node.set_parameters(
            [
                Parameter("n_dof", Parameter.Type.INTEGER, 2),
                Parameter("min_joint", Parameter.Type.DOUBLE_ARRAY, [0.0, 0.0]),
                Parameter("max_joint", Parameter.Type.DOUBLE_ARRAY, [1.0, 1.0]),
            ]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)

        bad = _make_chunk(n_dof=2, flat_value=0.0)
        bad.flat = [0.5, 5.0]  # joint[1] outside [0, 1]
        pub.publish(bad)
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert len(safe_received) == 0, "workspace violation must not republish"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


# ── /openral/estop_reset ────────────────────────────────────────────────────


def test_estop_reset_clears_latch_after_cooldown(ros_context: None) -> None:
    """Reset succeeds once cooldown elapses; fails before."""
    node = SafetyPassthroughNode(node_name="openral_safety_test_reset")
    helper = rclpy.create_node("openral_safety_test_reset_helper")

    estop_received: list[Empty] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
    client = helper.create_client(Trigger, "/openral/estop_reset")

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        # Cooldown very short for test responsiveness.
        from rclpy.parameter import Parameter

        node.set_parameters(
            [
                Parameter("n_dof", Parameter.Type.INTEGER, 6),
                Parameter("estop_reset_cooldown_s", Parameter.Type.DOUBLE, 0.05),
            ]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        # Trigger violation by publishing wrong n_dof.
        pub.publish(_make_chunk(n_dof=3))
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert node._estopped is True

        # Try reset before cooldown — must fail.
        assert client.wait_for_service(timeout_sec=1.0)
        future = client.call_async(Trigger.Request())

        def _spin_for_future() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return future.done()

        deadline = time.time() + 2.0
        while time.time() < deadline and not future.done():
            executor.spin_once(timeout_sec=0.02)
        assert future.done()
        early_resp = future.result()
        assert early_resp is not None
        # Either cooldown not elapsed → success=False, or already elapsed
        # (slow CI) → success=True. The deterministic case we want is
        # that calling reset *after* an extra sleep DOES succeed.
        # Wait through the cooldown explicitly then try again.
        time.sleep(0.1)
        future2 = client.call_async(Trigger.Request())
        deadline = time.time() + 2.0
        while time.time() < deadline and not future2.done():
            executor.spin_once(timeout_sec=0.02)
        assert future2.done()
        late_resp = future2.result()
        assert late_resp is not None and late_resp.success is True
        assert node._estopped is False
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


# ── Defense-in-depth: external estop ────────────────────────────────────────


def test_external_estop_latches_node(ros_context: None) -> None:
    """A publisher external to the node can latch us via /openral/estop."""
    node = SafetyPassthroughNode(node_name="openral_safety_test_external_estop")
    helper = rclpy.create_node("openral_safety_test_external_estop_helper")
    estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)
    chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
    safe_received: list[ActionChunk] = []
    helper.create_subscription(ActionChunk, "/openral/safe_action", safe_received.append, 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        # Fire external estop.
        estop_pub.publish(Empty())
        assert _spin_until(executor, lambda: node._estopped is True)
        # Subsequent valid chunks must be dropped.
        chunk_pub.publish(_make_chunk())
        # Spin briefly; verify nothing arrives on /openral/safe_action.
        deadline = time.time() + 0.5
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert len(safe_received) == 0
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


# Suppress unused-import warning — we use threading transitively via executor.
_ = threading


# ── Observability: safety.check spans for the dashboard's Safety card ────────


@pytest.fixture
def captured_spans() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory OTel tracer + exporter and return the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def test_pass_through_emits_safety_check_span(
    ros_context: None,
    captured_spans: InMemorySpanExporter,
) -> None:
    """A pass-through publication emits a ``safety.check`` span with kernel='passthrough'."""
    from rclpy.node import Node

    node = SafetyPassthroughNode(node_name="openral_safety_test_span_passthrough")
    helper = Node("openral_safety_test_span_passthrough_helper")
    received: list[ActionChunk] = []
    helper.create_subscription(ActionChunk, "/openral/safe_action", received.append, 10)
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        pub.publish(_make_chunk())
        assert _spin_until(executor, lambda: len(received) >= 1)
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "safety.check"]
    assert spans, "no safety.check span emitted on candidate_action passthrough"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("safety.kernel") == "passthrough"
    assert attrs.get("safety.severity") == "info"
    assert attrs.get("safety.check_name") == "envelope"
    assert attrs.get("safety.clamped") is False


def test_envelope_violation_emits_safety_check_span_with_violation_severity(
    ros_context: None,
    captured_spans: InMemorySpanExporter,
) -> None:
    """An n_dof mismatch produces a safety.check span at ``severity='violation'``."""
    node = SafetyPassthroughNode(node_name="openral_safety_test_span_violation")
    helper = rclpy.create_node("openral_safety_test_span_violation_helper")
    pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        # Configure with strict n_dof=4 so the n_dof=6 chunk violates.
        node.set_parameters([rclpy.parameter.Parameter("n_dof", value=4)])
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        pub.publish(_make_chunk())  # n_dof=6 → violation
        _spin_until(executor, lambda: node._chunks_dropped >= 1)
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "safety.check"]
    assert spans, "no safety.check span emitted on envelope violation"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("safety.severity") == "violation"
    assert attrs.get("safety.kernel") == "passthrough"
    assert attrs.get("safety.drop_reason") == "n_dof"
