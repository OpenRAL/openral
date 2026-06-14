"""Real-rclpy integration test for the deadman_watchdog_node.

Spins up the actual lifecycle node + a tiny test publisher of
``/openral/safe_action``; asserts that the watchdog fires
``/openral/estop`` + ``/openral/failure/safety`` when the publisher
stops. No mocks (CLAUDE.md §1.11).

Gated on ``rclpy`` + ``openral_msgs`` being importable; without them the
test skips with a typed reason.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_msgs.msg import ActionChunk, FailureTrigger
from openral_safety_watchdog.deadman_watchdog_node import DeadmanWatchdogNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from std_msgs.msg import Empty


@pytest.fixture
def ros_context() -> Any:
    """Spin up rclpy for the test; tear down after."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return False


def test_deadman_fires_when_safe_action_stops(ros_context: None) -> None:
    """No /openral/safe_action within deadline → estop + FailureTrigger."""
    node = DeadmanWatchdogNode(node_name="deadman_watchdog_test")
    helper = rclpy.create_node("deadman_watchdog_test_helper")
    estop_received: list[Empty] = []
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/safety",
        failures_received.append,
        50,
    )

    # /openral/failure/safety is a shared safety bus in production —
    # deadman_watchdog (KIND_TIMEOUT), the C++ safety kernel, the
    # human-estop forwarder (KIND_HUMAN), and any future safety
    # publisher all share the topic. Under parallel `colcon test` the
    # sibling openral_human_estop test process lands on the same DDS
    # domain and can publish KIND_HUMAN at the same instant; whichever
    # message DDS delivers first would race into [0]. Filter on
    # kind == KIND_TIMEOUT so the assertion reflects the real
    # production contract ("a KIND_TIMEOUT reached the bus") rather
    # than the over-specific "no other safety publisher exists at this
    # instant". Mirrors the fix in
    # packages/openral_human_estop/test/test_forwarder_node.py.
    def _have_timeout_failure() -> bool:
        return any(ft.kind == FailureTrigger.KIND_TIMEOUT for ft in failures_received)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        from rclpy.parameter import Parameter

        # Tight deadline + fast check period for test responsiveness.
        node.set_parameters(
            [
                Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1),
                Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
            ]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        # Don't publish any safe_action — let the deadline expire. Wait
        # for *both* topics; estop and FailureTrigger are published in
        # the same callback but ROS delivery order can interleave so we
        # spin until each has at least one message rather than checking
        # one and asserting on the other.
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert _spin_until(executor, _have_timeout_failure)
        ft = next(ft for ft in failures_received if ft.kind == FailureTrigger.KIND_TIMEOUT)
        assert ft.severity == FailureTrigger.SEVERITY_ABORT
        evidence = json.loads(ft.evidence_json)
        assert evidence["kind"] == "timeout"
        assert evidence["operation"] == "safe_action"
        assert evidence["deadline_s"] == 0.1
        assert evidence["elapsed_s"] > 0.1
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_deadman_does_not_fire_when_safe_action_arrives(ros_context: None) -> None:
    """Continuous /openral/safe_action → no KIND_TIMEOUT on the failure bus."""
    node = DeadmanWatchdogNode(node_name="deadman_watchdog_test_alive")
    helper = rclpy.create_node("deadman_watchdog_test_alive_helper")
    chunk_pub = helper.create_publisher(ActionChunk, "/openral/safe_action", 10)
    # /openral/estop is std_msgs/Empty so there is no `kind` field to
    # filter on; any sibling test process publishing an estop on the
    # same DDS domain (e.g. the human-estop forwarder) would inflate
    # `len(estop_received)` and turn this "no estop fired" check into
    # a false positive. Pivot the assertion to the watchdog's own
    # FailureTrigger publish on /openral/failure/safety: the watchdog
    # publishes both /openral/estop and FailureTrigger(KIND_TIMEOUT)
    # in the same callback as a unit, so the absence of one implies
    # the absence of the other for *this* node, and KIND_TIMEOUT is
    # specific enough to filter past cross-talk from human-estop's
    # KIND_HUMAN / a future kernel's KIND_ENVELOPE_VIOLATION / etc.
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/safety",
        failures_received.append,
        50,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        from rclpy.parameter import Parameter

        node.set_parameters(
            [
                Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.2),
                Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
            ]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        # Publish a chunk every ~50 ms for 500 ms — deadline never expires.
        deadline = time.time() + 0.5
        while time.time() < deadline:
            chunk = ActionChunk()
            chunk.control_mode = 0
            chunk.horizon = 1
            chunk.n_dof = 3
            chunk.flat = [0.0, 0.0, 0.0]
            chunk_pub.publish(chunk)
            executor.spin_once(timeout_sec=0.05)
        # The watchdog must not have published KIND_TIMEOUT. Any other
        # kinds on the bus are someone else's traffic (production has
        # multiple safety publishers) and are explicitly out-of-scope.
        timeout_failures = [
            ft for ft in failures_received if ft.kind == FailureTrigger.KIND_TIMEOUT
        ]
        assert timeout_failures == [], (
            f"expected 0 KIND_TIMEOUT FailureTriggers, got {len(timeout_failures)}: "
            f"{[ft.evidence_json for ft in timeout_failures]}"
        )
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_external_estop_suppresses_double_publish(ros_context: None) -> None:
    """Receiving /openral/estop externally must not race the watchdog."""
    node = DeadmanWatchdogNode(node_name="deadman_watchdog_test_suppress")
    helper = rclpy.create_node("deadman_watchdog_test_suppress_helper")
    estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)
    # Count only KIND_TIMEOUT (the watchdog's own publish) so the
    # assertion isn't poisoned by cross-talk from sibling safety
    # publishers on the shared /openral/failure/safety bus — see the
    # matching comment in test_deadman_fires_when_safe_action_stops.
    own_timeout_count = 0

    def _on_failure(msg: FailureTrigger) -> None:
        nonlocal own_timeout_count
        if msg.kind == FailureTrigger.KIND_TIMEOUT:
            own_timeout_count += 1

    helper.create_subscription(FailureTrigger, "/openral/failure/safety", _on_failure, 50)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        from rclpy.parameter import Parameter

        node.set_parameters(
            [
                Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1),
                Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
            ]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # External estop before deadline → watchdog should suppress its own fire.
        time.sleep(0.02)
        estop_pub.publish(Empty())
        # Let some spinning happen — deadline would expire too if watchdog
        # didn't suppress itself.
        deadline = time.time() + 0.4
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.02)
        # No FailureTrigger from our watchdog because the external estop
        # already triggered the latch.
        assert own_timeout_count == 0
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
