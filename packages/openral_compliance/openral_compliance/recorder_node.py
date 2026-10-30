#!/usr/bin/env python3
"""Compliance recorder lifecycle node (issue #467).

Turns what the graph already publishes into ``openral_core.ComplianceRecord``s:

* every ``FailureTrigger`` on ``/openral/failure/<source>`` → ``FailureRecord``
  (evidence decoded; an undecodable ``evidence_json`` keeps the record, without evidence);
* each ``/openral/safety_status`` transition into ``latched`` → ``SafetyStopRecord``,
  linked (``parent_record_id``) to the safety failure that explains it when one arrived
  within ``_STOP_EXPLAIN_WINDOW_S``;
* activate / deactivate → ``SessionRecord`` start / end.

Records go out as OTLP log records (``ComplianceEmitter``). The recorder only listens: it
publishes nothing and never sits on the actuation path. Writing the on-robot log is #469.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

import rclpy
from openral_core import (
    SITE_ID_ENV,
    FailureEvidence,
    FailureRecord,
    SafetyStopRecord,
    SessionRecord,
)
from openral_observability import configure_observability
from openral_observability.compliance import ComplianceEmitter
from openral_observability.failure_bus import FailureSource, topic_for
from pydantic import TypeAdapter, ValidationError
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["ComplianceRecorderNode", "main"]

_EVIDENCE: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)
_STOP_EXPLAIN_WINDOW_S = 2.0
"""A safety failure this recent explains a latch (the kernel publishes it first)."""

_TRACE_ID = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2})$")
"""Raw 32-hex or a W3C traceparent: the two forms the envelope normalises (ISSUE-109)."""

_KIND_TIMEOUT = 0
_KIND_HUMAN = 7
_DROP_EXTERNAL_ESTOP = 106


class ComplianceRecorderNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Records failures, safety stops and sessions as compliance records.

    Parameters:
        ``run_mode``: ``"sim"`` or ``"hardware"`` for the ``SessionRecord`` (default ``"sim"``).
    """

    def __init__(self, node_name: str = "openral_compliance_recorder") -> None:
        """Declare parameters; opens no resources until on_configure."""
        super().__init__(node_name)
        self.declare_parameter("run_mode", "sim")
        self._emitter: ComplianceEmitter | None = None
        self._subs: list[Any] = []
        self._latched = False
        self._last_safety: tuple[float, FailureRecord] | None = None
        self._session_start_ns: int | None = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open the failure-bus and safety-status subscriptions."""
        del state
        from openral_msgs.msg import FailureTrigger, SafetyStatus
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

        self._emitter = ComplianceEmitter()
        bus_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        for source in FailureSource:
            self._subs.append(
                self.create_subscription(
                    FailureTrigger,
                    topic_for(source),
                    lambda msg, s=source.value: self._on_failure(s, msg),
                    bus_qos,
                )
            )
        self._subs.append(
            self.create_subscription(
                SafetyStatus, "/openral/safety_status", self._on_safety_status, status_qos
            )
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open the session."""
        del state
        self._session_start_ns = time.time_ns()
        self._emit_session(end_ts_ns=None)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Close the session."""
        del state
        if self._session_start_ns is not None:
            self._emit_session(end_ts_ns=time.time_ns())
            self._session_start_ns = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release the subscriptions."""
        del state
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []
        self._emitter = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Close an open session, then clean up."""
        if self._session_start_ns is not None and self._emitter is not None:
            self._emit_session(end_ts_ns=time.time_ns())
            self._session_start_ns = None
        return self.on_cleanup(state)

    def _emit_session(self, *, end_ts_ns: int | None) -> None:
        assert self._emitter is not None and self._session_start_ns is not None
        run_mode = self.get_parameter("run_mode").get_parameter_value().string_value
        self._emitter.emit(
            SessionRecord,
            start_ts_ns=self._session_start_ns,
            end_ts_ns=end_ts_ns,
            run_mode="hardware" if run_mode == "hardware" else "sim",
            site_id=os.environ.get(SITE_ID_ENV) or None,
        )

    def _trace_id(self, wire: str) -> str:
        """The wire trace id if the envelope can take it, else ``""``.

        A malformed id must cost the trace link, never the record.
        """
        if wire == "" or _TRACE_ID.match(wire):
            return wire
        self.get_logger().warning(f"compliance.trace_id_malformed {wire[:64]!r}")
        return ""

    def _on_failure(self, source: str, msg: Any) -> None:  # reason: rclpy message
        """One ``FailureTrigger`` → one ``FailureRecord``."""
        if self._emitter is None:
            return
        evidence = None
        if msg.evidence_json:
            try:
                evidence = _EVIDENCE.validate_json(msg.evidence_json)
            except ValidationError as exc:
                # Keep the failure; an unreadable evidence payload must not lose the event.
                self.get_logger().warning(
                    f"compliance.evidence_undecodable source={source} kind={msg.kind}: "
                    f"{exc.error_count()} error(s)"
                )
        record = self._emitter.emit(
            FailureRecord,
            source=source,
            failure_kind=msg.kind,
            severity=msg.severity,
            evidence=evidence,
            rskill_id=msg.rskill_id or None,
            trace_id=self._trace_id(msg.trace_id),
        )
        if source == FailureSource.SAFETY.value:
            self._last_safety = (time.monotonic(), record)

    def _on_safety_status(self, msg: Any) -> None:  # reason: rclpy message
        """A transition into ``latched`` is a safety stop; the latch clearing is not."""
        was_latched, self._latched = self._latched, bool(msg.latched)
        if self._emitter is None or was_latched or not msg.latched:
            return
        cause = None
        if self._last_safety is not None:
            seen_at, failure = self._last_safety
            if time.monotonic() - seen_at <= _STOP_EXPLAIN_WINDOW_S:
                cause = failure
        self._emitter.emit(
            SafetyStopRecord,
            trigger_source=_trigger_source(msg.drop_reason, cause),
            drop_reason=msg.drop_reason,
            detail=msg.detail,
            evidence=cause.evidence if cause is not None else None,
            latched=True,
            rskill_id=msg.rskill_id or None,
            trace_id=self._trace_id(msg.trace_id),
            parent_record_id=cause.record_id if cause is not None else None,
        )


def _trigger_source(drop_reason: int, cause: FailureRecord | None) -> str:
    """Who stopped the machine: the kernel, unless an external e-stop names someone else."""
    if drop_reason == _KIND_HUMAN:
        return "human"
    if drop_reason == _DROP_EXTERNAL_ESTOP and cause is not None:
        if cause.failure_kind == _KIND_HUMAN:
            return "human"
        if cause.failure_kind == _KIND_TIMEOUT:
            return "deadman"
    return "kernel"


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_compliance recorder_node.py``."""
    configure_observability(service_name="openral.compliance")
    rclpy.init(args=args)
    try:
        node = ComplianceRecorderNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if already shut down
    return 0


if __name__ == "__main__":
    sys.exit(main())
