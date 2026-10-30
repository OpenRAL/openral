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

from openral_compliance.recorder_node import ComplianceRecorderNode
from openral_core import (
    CollisionEvidence,
    ComplianceRecord,
    FailureRecord,
    SafetyStopRecord,
    SessionRecord,
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
