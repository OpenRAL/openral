"""Real-rclpy test for the compliance recorder (issue #467).

Runs the lifecycle node against real ``FailureTrigger`` / ``SafetyStatus``
publishers and reads its records back from the OTel SDK's in-memory log
exporter (the process boundary; CLAUDE.md §1.11).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_compliance.recorder_node import STOP_EXPLAIN_WINDOW_S, ComplianceRecorderNode
from openral_core import (
    CollisionEvidence,
    ComplianceRecord,
    FailureRecord,
    SafetyStopRecord,
    SessionRecord,
    SoftwareInventoryRecord,
)
from openral_msgs.msg import FailureTrigger, SafetyStatus
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from pydantic import TypeAdapter
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

_EXPORTER = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: OTel SDK leaves __init__ unannotated
_provider = LoggerProvider()
_provider.add_log_record_processor(SimpleLogRecordProcessor(_EXPORTER))
set_logger_provider(_provider)
_ADAPTER: TypeAdapter[ComplianceRecord] = TypeAdapter(ComplianceRecord)


@pytest.fixture
def ros_context() -> Any:
    """Spin rclpy up / down per test."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _records() -> list[Any]:
    return [
        _ADAPTER.validate_json(str(log.log_record.body)) for log in _EXPORTER.get_finished_logs()
    ]


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return False


def test_collision_stop_is_recorded_and_linked_to_its_failure(ros_context: None) -> None:
    """Kernel collision trigger + latch → FailureRecord, then SafetyStopRecord pointing at it."""
    _EXPORTER.clear()
    node = ComplianceRecorderNode(node_name="compliance_recorder_test")
    helper = rclpy.create_node("compliance_recorder_test_helper")
    bus = helper.create_publisher(FailureTrigger, "/openral/failure/safety", 50)
    status = helper.create_publisher(
        SafetyStatus,
        "/openral/safety_status",
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert isinstance(_records()[0], SessionRecord)
        executor.spin_once(timeout_sec=0.2)

        evidence = CollisionEvidence(
            collision_kind="world",
            link_a="panda_hand",
            link_b_or_object="shelf",
            horizon_step=2,
            min_distance_m=-0.003,
            joint_positions_rad=[0.0] * 7,
        )
        trigger = FailureTrigger()
        trigger.kind = FailureTrigger.KIND_COLLISION
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        trigger.evidence_json = evidence.model_dump_json()
        trigger.rskill_id = "smolvla-libero"
        trigger.trace_id = f"00-{'a' * 32}-{'b' * 16}-01"
        bus.publish(trigger)
        assert _spin_until(executor, lambda: any(isinstance(r, FailureRecord) for r in _records()))

        clear = SafetyStatus()  # the kernel's activation sample: unlatched
        status.publish(clear)
        executor.spin_once(timeout_sec=0.2)
        latch = SafetyStatus()
        latch.latched = True
        latch.drop_reason = SafetyStatus.KIND_COLLISION
        latch.detail = "world collision panda_hand/shelf"
        latch.trace_id = trigger.trace_id
        status.publish(latch)
        assert _spin_until(
            executor, lambda: any(isinstance(r, SafetyStopRecord) for r in _records())
        )
        status.publish(latch)  # a 1 Hz refresh of the same latch is not a second stop
        executor.spin_once(timeout_sec=0.2)

        failure = next(r for r in _records() if isinstance(r, FailureRecord))
        stops = [r for r in _records() if isinstance(r, SafetyStopRecord)]
        assert len(stops) == 1
        (stop,) = stops
        assert failure.source == "safety" and failure.evidence == evidence
        assert failure.trace_id == "a" * 32 and failure.span_id == "b" * 16
        assert stop.trigger_source == "kernel"
        assert stop.parent_record_id == failure.record_id
        assert stop.evidence == evidence
        assert stop.seq > failure.seq
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_undecodable_evidence_and_bad_trace_still_record_the_failure(
    ros_context: None,
) -> None:
    """A producer bug costs the evidence and the trace link, never the record."""
    _EXPORTER.clear()
    node = ComplianceRecorderNode(node_name="compliance_recorder_bad_payload")
    helper = rclpy.create_node("compliance_recorder_bad_payload_helper")
    bus = helper.create_publisher(FailureTrigger, "/openral/failure/hal", 50)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.2)
        trigger = FailureTrigger()
        trigger.kind = FailureTrigger.KIND_CONTROLLER
        trigger.severity = FailureTrigger.SEVERITY_FAIL
        trigger.evidence_json = '{"kind": "controller"}'  # required fields missing
        trigger.trace_id = "not-a-trace"
        bus.publish(trigger)
        assert _spin_until(executor, lambda: any(isinstance(r, FailureRecord) for r in _records()))
        failure = next(r for r in _records() if isinstance(r, FailureRecord))
        assert (failure.source, failure.evidence, failure.trace_id) == ("hal", None, None)
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


class _Graph:
    """A recorder plus a helper publishing on the failure bus and the latched status topic."""

    def __init__(self, name: str) -> None:
        self.node = ComplianceRecorderNode(node_name=name)
        self.helper = rclpy.create_node(f"{name}_helper")
        self.bus = self.helper.create_publisher(FailureTrigger, "/openral/failure/safety", 50)
        self.status = self.helper.create_publisher(
            SafetyStatus,
            "/openral/safety_status",
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.helper)

    def bring_up(self) -> None:
        assert self.node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert self.node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        self.executor.spin_once(timeout_sec=0.2)

    def publish_status(self, *, latched: bool, drop_reason: int = SafetyStatus.DROP_NONE) -> None:
        msg = SafetyStatus()
        msg.latched = latched
        msg.drop_reason = drop_reason
        self.status.publish(msg)
        self.executor.spin_once(timeout_sec=0.2)

    def publish_failure(self, kind: int, evidence_json: str = "") -> None:
        msg = FailureTrigger()
        msg.kind = kind
        msg.severity = FailureTrigger.SEVERITY_ABORT
        msg.evidence_json = evidence_json
        self.bus.publish(msg)

    def close(self) -> None:
        self.executor.remove_node(self.node)
        self.executor.remove_node(self.helper)
        self.node.destroy_node()
        self.helper.destroy_node()


def _stops() -> list[SafetyStopRecord]:
    return [r for r in _records() if isinstance(r, SafetyStopRecord)]


def test_estop_latch_before_its_trigger_still_names_the_human(ros_context: None) -> None:
    """The forwarder publishes /openral/estop before its FailureTrigger: wait for the cause."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_estop_order")
    try:
        g.bring_up()
        g.publish_status(latched=False)
        g.publish_status(latched=True, drop_reason=SafetyStatus.DROP_EXTERNAL_ESTOP)
        assert _stops() == []  # waiting for the cause, not guessing "kernel"
        g.publish_failure(
            FailureTrigger.KIND_HUMAN, '{"kind": "human", "actor": "pendant", "reason": "x"}'
        )
        assert _spin_until(g.executor, lambda: len(_stops()) == 1)
        (stop,) = _stops()
        human = next(r for r in _records() if isinstance(r, FailureRecord))
        assert stop.trigger_source == "human"
        assert stop.parent_record_id == human.record_id
        assert stop.ts_ns < human.ts_ns  # stamped when the latch arrived
    finally:
        g.close()


def test_an_unrelated_recent_failure_is_not_the_cause(ros_context: None) -> None:
    """A workspace warning just before a collision latch is not its cause."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_unrelated")
    try:
        g.bring_up()
        g.publish_status(latched=False)
        g.publish_failure(FailureTrigger.KIND_WORKSPACE)
        assert _spin_until(
            g.executor, lambda: any(isinstance(r, FailureRecord) for r in _records())
        )
        g.publish_status(latched=True, drop_reason=SafetyStatus.KIND_COLLISION)
        assert _spin_until(
            g.executor, lambda: len(_stops()) == 1, timeout_s=STOP_EXPLAIN_WINDOW_S + 2.0
        )
        (stop,) = _stops()
        assert (stop.parent_record_id, stop.evidence, stop.trigger_source) == (None, None, "kernel")
    finally:
        g.close()


def test_replayed_latch_and_inactive_node_record_nothing(ros_context: None) -> None:
    """A latch replayed at configure is not a new stop; nothing records before activate."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_replay")
    try:
        g.publish_status(latched=True, drop_reason=SafetyStatus.KIND_COLLISION)  # durable
        assert g.node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        g.executor.spin_once(timeout_sec=0.3)
        g.publish_failure(FailureTrigger.KIND_COLLISION)
        g.executor.spin_once(timeout_sec=0.3)
        assert _records() == []  # INACTIVE: no session, no records
        assert g.node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        g.executor.spin_once(timeout_sec=0.3)
        assert [type(r) for r in _records()] == [SessionRecord, SoftwareInventoryRecord]
        g.node.close_session()  # what main() does on SIGINT
        session_end = _records()[-1]
        assert isinstance(session_end, SessionRecord) and session_end.end_ts_ns is not None
        g.node.close_session()  # idempotent
        assert len(_records()) == 3
    finally:
        g.close()


def test_an_out_of_range_severity_is_logged_not_fatal(ros_context: None) -> None:
    """A producer bug on one trigger must not end the recorder; the next one records."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_bad_severity")
    try:
        g.bring_up()
        bad = FailureTrigger()
        bad.kind = FailureTrigger.KIND_COLLISION
        bad.severity = 9  # SEVERITY_* stops at 3 (ABORT)
        g.bus.publish(bad)
        g.executor.spin_once(timeout_sec=0.3)
        g.publish_failure(FailureTrigger.KIND_HUMAN)
        assert _spin_until(
            g.executor, lambda: any(isinstance(r, FailureRecord) for r in _records())
        )
        (failure,) = [r for r in _records() if isinstance(r, FailureRecord)]
        assert failure.failure_kind == FailureTrigger.KIND_HUMAN
    finally:
        g.close()


def test_the_recorder_writes_the_on_robot_log(
    ros_context: None, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With $OPENRAL_COMPLIANCE_DIR set, every record also lands in a verifiable log (#469)."""
    from openral_core import RetentionClass
    from openral_observability.compliance_log import records, verify

    monkeypatch.setenv("OPENRAL_COMPLIANCE_DIR", str(tmp_path))
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_log")
    try:
        g.bring_up()
        g.publish_failure(FailureTrigger.KIND_HUMAN)
        assert _spin_until(
            g.executor, lambda: any(isinstance(r, FailureRecord) for r in _records())
        )
        g.node.close_session()
    finally:
        g.close()
    assert verify(tmp_path) == []
    (inventory,) = records(tmp_path, RetentionClass.MR_TRACE_5Y)
    assert isinstance(inventory, SoftwareInventoryRecord)
    sessions = list(records(tmp_path, RetentionClass.AI_LOG_6M))
    assert sum(isinstance(r, SessionRecord) for r in sessions) == 2  # start and end
    assert sum(isinstance(r, FailureRecord) for r in sessions) == 1


def _interventions() -> list[Any]:
    from openral_core import InterventionRecord

    return [r for r in _records() if isinstance(r, InterventionRecord)]


def _announce(
    g: _Graph, action: str, *, channel: str, actor_id: str = "", detail: str = ""
) -> None:
    from openral_compliance.recorder_node import OPERATOR_ACTION_TOPIC
    from openral_msgs.msg import OperatorAction

    pub = g.helper.create_publisher(OperatorAction, OPERATOR_ACTION_TOPIC, 10)
    g.executor.spin_once(timeout_sec=0.2)
    msg = OperatorAction()
    msg.action = action
    msg.channel_label = channel
    msg.actor_id = actor_id
    msg.detail = detail
    pub.publish(msg)
    g.executor.spin_once(timeout_sec=0.2)


def test_an_announced_reset_is_attributed(ros_context: None) -> None:
    """`openral estop reset --operator` announces, then the latch clears: one attributed record."""
    from openral_core import pseudonymous_actor_id

    _EXPORTER.clear()
    g = _Graph("compliance_recorder_reset_announced")
    try:
        g.bring_up()
        g.publish_status(latched=False)
        g.publish_status(latched=True, drop_reason=SafetyStatus.DROP_EXTERNAL_ESTOP)
        actor = pseudonymous_actor_id("alice", "deployer-key")
        _announce(g, "estop_reset", channel="cli", actor_id=actor)
        g.publish_status(latched=False)
        assert _spin_until(g.executor, lambda: len(_interventions()) == 1)
        (reset,) = _interventions()
        assert reset.intervention_kind == "estop_reset"
        assert (reset.channel, reset.legitimate, reset.actor.actor_id) == ("cli", True, actor)
    finally:
        g.close()


def test_an_unannounced_reset_is_recorded_unattributed(ros_context: None) -> None:
    """A `ros2 service call` reset nobody announced: recorded, legitimacy unknown."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_reset_unannounced")
    try:
        g.bring_up()
        g.publish_status(latched=False)
        g.publish_status(latched=True, drop_reason=SafetyStatus.KIND_COLLISION)
        g.publish_status(latched=False)
        assert _interventions() == []  # waiting for someone to say who
        assert _spin_until(
            g.executor, lambda: len(_interventions()) == 1, timeout_s=STOP_EXPLAIN_WINDOW_S + 2.0
        )
        (reset,) = _interventions()
        assert (reset.channel, reset.legitimate) == ("unknown", None)
        assert reset.actor.channel_label == "unattributed"
    finally:
        g.close()


def test_a_late_announcement_still_attributes_and_param_set_is_recorded(
    ros_context: None,
) -> None:
    """The announcement may arrive after the latch clears; a param change is an intervention."""
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_reset_late")
    try:
        g.bring_up()
        g.publish_status(latched=False)
        g.publish_status(latched=True, drop_reason=SafetyStatus.DROP_EXTERNAL_ESTOP)
        g.publish_status(latched=False)
        _announce(g, "estop_reset", channel="dashboard", actor_id="not-a-hash")
        _announce(g, "param_set", channel="dashboard", detail="/openral_reasoner:temperature")
        assert _spin_until(g.executor, lambda: len(_interventions()) == 2)
        reset, param = sorted(_interventions(), key=lambda r: r.intervention_kind)
        assert (reset.channel, reset.legitimate, reset.actor.actor_id) == ("dashboard", True, None)
        assert (param.intervention_kind, param.diff_ref) == (
            "param_set",
            "/openral_reasoner:temperature",
        )
    finally:
        g.close()


def test_a_log_it_cannot_write_stops_the_machine_once(
    ros_context: None, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hazard-log entry (#468): no evidence, no motion. One stop per failure episode."""
    import os

    from openral_core import ControllerEvidence
    from std_msgs.msg import Empty

    monkeypatch.setenv("OPENRAL_COMPLIANCE_DIR", str(tmp_path))
    _EXPORTER.clear()
    g = _Graph("compliance_recorder_log_fails")
    estops: list[Any] = []
    triggers: list[Any] = []
    g.helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
    g.helper.create_subscription(FailureTrigger, "/openral/failure/safety", triggers.append, 50)
    try:
        g.bring_up()
        if os.geteuid() == 0:
            pytest.skip("running as root: a read-only directory is still writable")
        for path in tmp_path.rglob("*"):  # a full disk, as far as appends are concerned
            os.chmod(path, 0o500 if path.is_dir() else 0o400)
        g.publish_failure(FailureTrigger.KIND_WORKSPACE)
        g.publish_failure(FailureTrigger.KIND_WORKSPACE)
        assert _spin_until(g.executor, lambda: len(estops) >= 1)
        for _ in range(10):
            g.executor.spin_once(timeout_sec=0.05)
        assert len(estops) == 1  # its own trigger also fails to log, and does not re-stop
        ours = [t for t in triggers if t.kind == FailureTrigger.KIND_CONTROLLER]
        assert len(ours) == 1
        evidence = ControllerEvidence.model_validate_json(ours[0].evidence_json)
        assert evidence.state == "log_write_failed"
    finally:
        for path in [tmp_path, *tmp_path.rglob("*")]:
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        g.close()
