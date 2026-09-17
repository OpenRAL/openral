"""Real-rclpy tests for the deadman_watchdog_node.

Spins up the actual lifecycle node plus real publishers of
``/openral/safe_action``, the runner's action-status topic and
``/openral/safety_status``; asserts when the watchdog fires and — just as
importantly — that every guard which can *suppress* a fire is bounded and
recoverable. No mocks (CLAUDE.md §1.11).

Gated on ``rclpy`` + ``openral_msgs`` being importable; without them the test
skips with a typed reason.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from action_msgs.msg import GoalStatus, GoalStatusArray
from openral_msgs.msg import ActionChunk, FailureTrigger, SafetyStatus
from openral_safety_watchdog.deadman_watchdog_node import DeadmanWatchdogNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_action_status_default,
)
from std_msgs.msg import Empty

# The execution-window topic deploy_e2e.launch.py gates the deadman on — the
# runner's own ExecuteRskill action status, not an advisory task string.
_ARM_STATUS_TOPIC = "/openral/execute_rskill/_action/status"
_SAFETY_STATUS_TOPIC = "/openral/safety_status"


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
    return bool(predicate())


def _chunk() -> ActionChunk:
    """A 3-DoF single-step chunk — the shape the supervisor republishes."""
    chunk = ActionChunk()
    chunk.control_mode = 0
    chunk.horizon = 1
    chunk.n_dof = 3
    chunk.flat = [0.0, 0.0, 0.0]
    return chunk


def _status(*statuses: int) -> GoalStatusArray:
    """A GoalStatusArray carrying one entry per given status value."""
    array = GoalStatusArray()
    entries = []
    for value in statuses:
        entry = GoalStatus()
        entry.status = value
        entries.append(entry)
    array.status_list = entries
    return array


def _safety_status(*, latched: bool) -> SafetyStatus:
    """A SafetyStatus reporting the kernel's current latch state."""
    msg = SafetyStatus()
    msg.latched = latched
    msg.drop_reason = SafetyStatus.DROP_NONE
    msg.detail = "test"
    return msg


def _gated_node(node_name: str, *, first_chunk_deadline_s: float = 5.0) -> Any:
    """A watchdog gated on the action-status topic, with test-tight budgets."""
    from rclpy.parameter import Parameter

    node = DeadmanWatchdogNode(node_name=node_name)
    node.set_parameters(
        [
            Parameter("arm_status_topic", Parameter.Type.STRING, _ARM_STATUS_TOPIC),
            Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1),
            Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
            Parameter("first_chunk_deadline_s", Parameter.Type.DOUBLE, first_chunk_deadline_s),
        ]
    )
    return node


def _timeout_failures(received: list[FailureTrigger]) -> list[FailureTrigger]:
    """Only this watchdog's own kind — the failure bus is shared."""
    return [ft for ft in received if ft.kind == FailureTrigger.KIND_TIMEOUT]


class _Harness:
    """Node + helper publishers/subscribers on one executor."""

    def __init__(self, node: Any, name: str) -> None:
        """Wire the helper's publishers and collectors around ``node``."""
        self.node = node
        self.helper = rclpy.create_node(f"{name}_helper")
        self.estop: list[Empty] = []
        self.failures: list[FailureTrigger] = []
        self.helper.create_subscription(Empty, "/openral/estop", self.estop.append, 10)
        self.helper.create_subscription(
            FailureTrigger, "/openral/failure/safety", self.failures.append, 50
        )
        self.arm_pub = self.helper.create_publisher(
            GoalStatusArray, _ARM_STATUS_TOPIC, qos_profile_action_status_default
        )
        self.chunk_pub = self.helper.create_publisher(ActionChunk, "/openral/safe_action", 10)
        self.status_pub = self.helper.create_publisher(
            SafetyStatus,
            _SAFETY_STATUS_TOPIC,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self.executor.add_node(self.helper)

    def activate(self) -> None:
        """Drive the node to ACTIVE."""
        assert self.node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert self.node.trigger_activate() == TransitionCallbackReturn.SUCCESS

    def stream(self, seconds: float) -> None:
        """Publish chunks at ~50 Hz for ``seconds``."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.chunk_pub.publish(_chunk())
            self.executor.spin_once(timeout_sec=0.02)

    def idle(self, seconds: float) -> None:
        """Spin without publishing anything."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def close(self) -> None:
        """Tear down both nodes."""
        self.executor.remove_node(self.node)
        self.executor.remove_node(self.helper)
        self.node.destroy_node()
        self.helper.destroy_node()


def test_deadman_fires_when_safe_action_stops(ros_context: None) -> None:
    """Free-running: no /openral/safe_action within deadline → estop."""
    from rclpy.parameter import Parameter

    node = DeadmanWatchdogNode(node_name="deadman_free_running")
    node.set_parameters(
        [
            Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1),
            Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
        ]
    )
    harness = _Harness(node, "deadman_free_running")
    try:
        harness.activate()
        assert _spin_until(harness.executor, lambda: len(harness.estop) >= 1)
        assert _spin_until(harness.executor, lambda: bool(_timeout_failures(harness.failures)))
        trigger = _timeout_failures(harness.failures)[0]
        assert trigger.severity == FailureTrigger.SEVERITY_ABORT
        evidence = json.loads(trigger.evidence_json)
        assert evidence["kind"] == "timeout"
        assert evidence["operation"] == "safe_action"
    finally:
        harness.close()


def test_gated_watchdog_is_quiet_before_any_goal_is_live(ros_context: None) -> None:
    """An idle deploy publishes no safe_action and must NOT be braked.

    This is the regression that kept the watchdog out of the deploy graph: with
    the gate removed, this fires an E-stop within 100 ms of activation.
    """
    harness = _Harness(_gated_node("deadman_gate_idle"), "deadman_gate_idle")
    try:
        harness.activate()
        harness.idle(0.5)  # five deadlines' worth of silence, nothing executing
        assert _timeout_failures(harness.failures) == []
        assert harness.estop == []
    finally:
        harness.close()


def test_gated_watchdog_fires_when_the_chunk_stream_dies_mid_goal(ros_context: None) -> None:
    """Goal live + chunks started + chunks stop → estop + TimeoutEvidence."""
    harness = _Harness(_gated_node("deadman_gate_fires"), "deadman_gate_fires")
    try:
        harness.activate()
        harness.arm_pub.publish(_status(GoalStatus.STATUS_EXECUTING))
        harness.stream(0.3)
        assert _timeout_failures(harness.failures) == [], "fired while chunks were flowing"

        # Producer dies: no more chunks, and no terminal status either.
        assert _spin_until(harness.executor, lambda: len(harness.estop) >= 1)
        assert _spin_until(harness.executor, lambda: bool(_timeout_failures(harness.failures)))
        trigger = _timeout_failures(harness.failures)[0]
        evidence = json.loads(trigger.evidence_json)
        assert evidence["operation"] == "safe_action"
        assert evidence["elapsed_s"] > evidence["deadline_s"]
    finally:
        harness.close()


@pytest.mark.parametrize(
    "terminal",
    [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED],
)
def test_gated_watchdog_is_quiet_after_the_goal_reaches_a_terminal_status(
    ros_context: None, terminal: int
) -> None:
    """A goal that ends stops the chunks AND closes the window."""
    name = f"deadman_gate_done_{terminal}"
    harness = _Harness(_gated_node(name), name)
    try:
        harness.activate()
        harness.arm_pub.publish(_status(GoalStatus.STATUS_EXECUTING))
        harness.stream(0.2)
        harness.arm_pub.publish(_status(terminal))
        harness.idle(0.5)
        assert _timeout_failures(harness.failures) == []
        assert harness.estop == []
    finally:
        harness.close()


def test_a_goal_that_never_produces_a_first_chunk_is_bounded(ros_context: None) -> None:
    """The arm→first-chunk wait must be bounded, not infinite.

    A runner that accepts a goal then dies before emitting a chunk leaves the
    window open with no chunk ever seen and no terminal status. Unbounded, the
    watchdog would wait forever — the actuation path is dead and nothing brakes.
    """
    node = _gated_node("deadman_first_chunk", first_chunk_deadline_s=0.3)
    harness = _Harness(node, "deadman_first_chunk")
    try:
        harness.activate()
        harness.arm_pub.publish(_status(GoalStatus.STATUS_ACCEPTED))
        # Never publish a chunk. Well inside the budget: still quiet.
        harness.idle(0.15)
        assert harness.estop == [], "fired before the first-chunk budget elapsed"
        # Past the budget: must brake.
        assert _spin_until(harness.executor, lambda: len(harness.estop) >= 1, timeout_s=3.0), (
            "a goal that never produced a chunk was never braked"
        )
        assert _spin_until(harness.executor, lambda: bool(_timeout_failures(harness.failures)))
        trigger = _timeout_failures(harness.failures)[0]
        evidence = json.loads(trigger.evidence_json)
        assert evidence["operation"] == "first_chunk"
        assert evidence["elapsed_s"] > evidence["deadline_s"]
    finally:
        harness.close()


def test_a_slow_policy_load_inside_the_budget_is_not_braked(ros_context: None) -> None:
    """A cold VLA load between goal accept and first chunk must not fire."""
    node = _gated_node("deadman_slow_load", first_chunk_deadline_s=5.0)
    harness = _Harness(node, "deadman_slow_load")
    try:
        harness.activate()
        harness.arm_pub.publish(_status(GoalStatus.STATUS_ACCEPTED))
        harness.idle(0.6)  # six safe_action deadlines of loading
        assert harness.estop == [], "braked a policy that was still loading"
        harness.stream(0.2)  # chunks finally start
        assert _timeout_failures(harness.failures) == []
    finally:
        harness.close()


def test_external_estop_suppresses_double_publish(ros_context: None) -> None:
    """An estop from another source latches this node rather than storming."""
    harness = _Harness(_gated_node("deadman_no_storm"), "deadman_no_storm")
    try:
        harness.activate()
        foreign = harness.helper.create_publisher(Empty, "/openral/estop", 10)
        harness.arm_pub.publish(_status(GoalStatus.STATUS_EXECUTING))
        harness.stream(0.15)
        foreign.publish(Empty())
        harness.idle(0.5)  # chunks have stopped, but someone else already braked
        assert _timeout_failures(harness.failures) == [], (
            "the watchdog published behind an estop that had already fired"
        )
    finally:
        harness.close()


def test_the_latch_is_released_when_safety_status_reports_recovery(ros_context: None) -> None:
    """After an estop + reset, the watchdog must arm again.

    Without this the node is a one-shot: the first estop of a deploy — from any
    source, including the dashboard stop button — silences it for the life of
    the process, and the graph then has exactly the hole this node exists to
    close, undetectably.
    """
    harness = _Harness(_gated_node("deadman_rearm"), "deadman_rearm")
    try:
        harness.activate()
        foreign = harness.helper.create_publisher(Empty, "/openral/estop", 10)

        # Somebody else brakes; the watchdog latches behind them.
        foreign.publish(Empty())
        harness.idle(0.2)
        assert _timeout_failures(harness.failures) == []

        # Operator resets: the kernel republishes SafetyStatus(latched=False).
        harness.status_pub.publish(_safety_status(latched=False))
        harness.idle(0.2)

        # A fresh goal whose chunk stream then dies must fire again.
        harness.arm_pub.publish(_status(GoalStatus.STATUS_EXECUTING))
        harness.stream(0.15)
        assert _spin_until(harness.executor, lambda: bool(_timeout_failures(harness.failures))), (
            "the watchdog never re-armed after recovery — it is one-shot per activation"
        )
    finally:
        harness.close()


def test_a_still_latched_safety_status_does_not_release_the_latch(ros_context: None) -> None:
    """Only a cleared SafetyStatus re-arms; a latched one must not."""
    harness = _Harness(_gated_node("deadman_rearm_denied"), "deadman_rearm_denied")
    try:
        harness.activate()
        foreign = harness.helper.create_publisher(Empty, "/openral/estop", 10)
        foreign.publish(Empty())
        harness.idle(0.2)

        harness.status_pub.publish(_safety_status(latched=True))
        harness.idle(0.2)

        harness.arm_pub.publish(_status(GoalStatus.STATUS_EXECUTING))
        harness.stream(0.15)
        harness.idle(0.4)
        assert _timeout_failures(harness.failures) == [], (
            "the watchdog re-armed while the safety layer was still latched"
        )
    finally:
        harness.close()
