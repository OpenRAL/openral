"""Real-rclpy test for the human estop forwarder.

Spins up the lifecycle node + a tiny publisher of
``/openral/human_estop``; asserts that the forwarder republishes onto
``/openral/estop`` and emits ``FailureTrigger(KIND_HUMAN)`` with the
configured ``channel_label``. No mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_human_estop.forwarder_node import HumanEstopForwarderNode
from openral_msgs.msg import FailureTrigger
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from std_msgs.msg import Empty


@pytest.fixture
def ros_context() -> Any:
    """Spin rclpy up / down per test."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return False


def test_forwarder_republishes_estop_with_failure_trigger(
    ros_context: None,
) -> None:
    """/openral/human_estop → /openral/estop + FailureTrigger(KIND_HUMAN)."""
    node = HumanEstopForwarderNode(node_name="human_estop_forwarder_test")
    helper = rclpy.create_node("human_estop_forwarder_test_helper")
    human_pub = helper.create_publisher(Empty, "/openral/human_estop", 10)
    estop_received: list[Empty] = []
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/safety",
        failures_received.append,
        50,
    )

    # /openral/failure/safety is a shared safety bus in production — the
    # watchdog, the kernel, and this forwarder all publish on it, and a
    # parallel `colcon test` puts the sibling openral_safety_watchdog
    # test on the same DDS domain. Filter to OUR kind instead of
    # trusting message order; the assertion is "at least one
    # FailureTrigger(KIND_HUMAN) reached the bus", not "no other
    # publisher exists" — the latter would be a false production
    # invariant. Same realism applies if you ever wire openral_safety
    # into the runner during this test's lifetime.
    def _have_human_failure() -> bool:
        return any(ft.kind == FailureTrigger.KIND_HUMAN for ft in failures_received)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        from rclpy.parameter import Parameter

        node.set_parameters([Parameter("channel_label", Parameter.Type.STRING, "dashboard")])
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # Spin once so subscriptions discover publishers.
        executor.spin_once(timeout_sec=0.1)
        human_pub.publish(Empty())
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert _spin_until(executor, _have_human_failure)
        ft = next(ft for ft in failures_received if ft.kind == FailureTrigger.KIND_HUMAN)
        assert ft.severity == FailureTrigger.SEVERITY_ABORT
        evidence = json.loads(ft.evidence_json)
        assert evidence["kind"] == "human"
        assert evidence["channel"] == "dashboard"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_forwarder_uses_default_channel_label_when_unset(ros_context: None) -> None:
    """When channel_label is empty, default ``unknown_human_channel`` is used."""
    node = HumanEstopForwarderNode(node_name="human_estop_default_channel")
    helper = rclpy.create_node("human_estop_default_channel_helper")
    human_pub = helper.create_publisher(Empty, "/openral/human_estop", 10)
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/safety",
        failures_received.append,
        50,
    )

    # Same kind-filter rationale as test_forwarder_republishes_estop above —
    # /openral/failure/safety is a shared bus, so wait for OUR KIND_HUMAN
    # message instead of trusting [0].
    def _have_human_failure() -> bool:
        return any(ft.kind == FailureTrigger.KIND_HUMAN for ft in failures_received)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.1)
        human_pub.publish(Empty())
        assert _spin_until(executor, _have_human_failure)
        ft = next(ft for ft in failures_received if ft.kind == FailureTrigger.KIND_HUMAN)
        evidence = json.loads(ft.evidence_json)
        assert evidence["channel"] == "unknown_human_channel"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
