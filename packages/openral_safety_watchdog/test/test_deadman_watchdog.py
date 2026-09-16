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
from std_msgs.msg import Empty, String

# The execution-window topic the deploy graph wires (the runner's own
# /openral/reward/active_task). Named here so the gated tests below exercise
# the same contract deploy_e2e.launch.py sets, not an invented one.
_ARM_TOPIC = "/openral/reward/active_task"


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

    # /openral/failure/safety is a shared safety bus in production — deadman_watchdog
    # (KIND_TIMEOUT), the C++ safety kernel, and the human-estop forwarder (KIND_HUMAN) all
    # share it. Under parallel `colcon test` the sibling openral_human_estop process can land
    # on the same DDS domain and race a KIND_HUMAN into [0], so filter on KIND_TIMEOUT instead
    # (mirrors packages/openral_human_estop/test/test_forwarder_node.py).
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
    # /openral/estop is std_msgs/Empty (no `kind` field), so a sibling test process publishing
    # an estop on the same DDS domain would inflate `len(estop_received)` into a false
    # positive. Pivot to the watchdog's own FailureTrigger on /openral/failure/safety instead:
    # it publishes /openral/estop + FailureTrigger(KIND_TIMEOUT) as a unit, so absence of one
    # implies absence of the other for *this* node, and KIND_TIMEOUT filters past cross-talk.
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


# ── Execution-window gate (``arm_topic``) ────────────────────────────────────
#
# Why the gate exists: /openral/safe_action is bursty. The runner publishes
# chunks only while an ExecuteRskill goal runs, so a free-running watchdog in
# the deploy graph would fire ``safe_action_deadline_s`` after boot and latch an
# E-stop on an idle robot. These three tests pin the narrowing: quiet before a
# window, quiet after one closes, and firing inside an open one.


def _chunk() -> ActionChunk:
    """A 3-DoF single-step chunk — the shape the supervisor republishes."""
    chunk = ActionChunk()
    chunk.control_mode = 0
    chunk.horizon = 1
    chunk.n_dof = 3
    chunk.flat = [0.0, 0.0, 0.0]
    return chunk


def _gated_node(node_name: str) -> Any:
    """A watchdog gated on ``_ARM_TOPIC`` with a test-tight deadline."""
    from rclpy.parameter import Parameter

    node = DeadmanWatchdogNode(node_name=node_name)
    node.set_parameters(
        [
            Parameter("arm_topic", Parameter.Type.STRING, _ARM_TOPIC),
            Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1),
            Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
        ]
    )
    return node


def _timeout_failures(received: list[FailureTrigger]) -> list[FailureTrigger]:
    """Only this watchdog's own kind — the failure bus is shared."""
    return [ft for ft in received if ft.kind == FailureTrigger.KIND_TIMEOUT]


def test_gated_watchdog_is_quiet_before_any_window_opens(ros_context: None) -> None:
    """An idle deploy publishes no safe_action and must NOT be braked.

    This is the regression that kept the watchdog out of the deploy graph: with
    the gate removed, this test fires an E-stop within 100 ms of activation.
    """
    node = _gated_node("deadman_watchdog_test_gate_idle")
    helper = rclpy.create_node("deadman_watchdog_test_gate_idle_helper")
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(
        FailureTrigger, "/openral/failure/safety", failures_received.append, 50
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # Five deadlines' worth of silence with nothing executing.
        deadline = time.time() + 0.5
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert _timeout_failures(failures_received) == []
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_gated_watchdog_fires_when_the_chunk_stream_dies_mid_window(ros_context: None) -> None:
    """Window open + chunks started + chunks stop → estop + TimeoutEvidence.

    The producer dying mid-goal is the hazard: nothing closes the window, so
    the silence is real evidence that the actuation path is gone.
    """
    node = _gated_node("deadman_watchdog_test_gate_fires")
    helper = rclpy.create_node("deadman_watchdog_test_gate_fires_helper")
    arm_pub = helper.create_publisher(String, _ARM_TOPIC, 1)
    chunk_pub = helper.create_publisher(ActionChunk, "/openral/safe_action", 10)
    estop_received: list[Empty] = []
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    helper.create_subscription(
        FailureTrigger, "/openral/failure/safety", failures_received.append, 50
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        window = String()
        window.data = "put the cube in the box"
        arm_pub.publish(window)

        # A live stream inside the open window: still no fire.
        deadline = time.time() + 0.3
        while time.time() < deadline:
            chunk_pub.publish(_chunk())
            executor.spin_once(timeout_sec=0.02)
        assert _timeout_failures(failures_received) == [], "fired while chunks were flowing"

        # Producer dies: no more chunks, and no closing "" either.
        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert _spin_until(executor, lambda: bool(_timeout_failures(failures_received)))
        ft = _timeout_failures(failures_received)[0]
        assert ft.severity == FailureTrigger.SEVERITY_ABORT
        evidence = json.loads(ft.evidence_json)
        assert evidence["kind"] == "timeout"
        assert evidence["operation"] == "safe_action"
        assert evidence["elapsed_s"] > evidence["deadline_s"]
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_gated_watchdog_is_quiet_after_the_window_closes(ros_context: None) -> None:
    """A goal that ends normally stops the chunks AND closes the window."""
    node = _gated_node("deadman_watchdog_test_gate_closed")
    helper = rclpy.create_node("deadman_watchdog_test_gate_closed_helper")
    arm_pub = helper.create_publisher(String, _ARM_TOPIC, 1)
    chunk_pub = helper.create_publisher(ActionChunk, "/openral/safe_action", 10)
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(
        FailureTrigger, "/openral/failure/safety", failures_received.append, 50
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        opened = String()
        opened.data = "put the cube in the box"
        arm_pub.publish(opened)
        deadline = time.time() + 0.2
        while time.time() < deadline:
            chunk_pub.publish(_chunk())
            executor.spin_once(timeout_sec=0.02)

        closed = String()
        closed.data = ""
        arm_pub.publish(closed)

        deadline = time.time() + 0.5
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert _timeout_failures(failures_received) == []
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
