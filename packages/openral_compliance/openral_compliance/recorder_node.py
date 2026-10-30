#!/usr/bin/env python3
"""Compliance recorder lifecycle node (issue #467).

Turns what the graph already publishes into ``openral_core.ComplianceRecord``s:

* every ``FailureTrigger`` on ``/openral/failure/<source>`` → ``FailureRecord``
  (evidence decoded; an undecodable ``evidence_json`` keeps the record, without evidence);
* each ``/openral/safety_status`` transition into ``latched`` → ``SafetyStopRecord``,
  linked (``parent_record_id``) to the safety failure that explains it. The cause must
  match the latch's ``drop_reason`` and may arrive up to ``STOP_EXPLAIN_WINDOW_S`` before
  or after it: the kernel publishes its trigger first, but the e-stop forwarder, pendant
  and deadman publish ``/openral/estop`` before theirs, and two topics have no order;
* activate / deactivate (and process shutdown) → ``SessionRecord`` start / end.

Only an ACTIVE node records. Records go out as OTLP log records (``ComplianceEmitter``).
The recorder only listens: it publishes nothing and never sits on the actuation path.
Writing the on-robot log is #469.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any

import rclpy
from openral_core import (
    SITE_ID_ENV,
    FailureEvidence,
    FailureRecord,
    SafetyStopRecord,
    SessionRecord,
)
from openral_msgs.msg import FailureTrigger, SafetyStatus
from openral_observability import configure_observability
from openral_observability.compliance import ComplianceEmitter
from openral_observability.failure_bus import FailureSource, topic_for
from pydantic import TypeAdapter, ValidationError
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["STOP_EXPLAIN_WINDOW_S", "ComplianceRecorderNode", "explains", "main"]

_EVIDENCE: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)

STOP_EXPLAIN_WINDOW_S = 2.0
"""How far apart a latch and the safety failure that explains it may arrive, either way."""

# The kernel's fail-closed input drops publish KIND_PERCEPTION (#454).
_PERCEPTION_DROPS = frozenset(
    {
        SafetyStatus.DROP_VOXEL_UNAVAILABLE,
        SafetyStatus.DROP_STATE_UNAVAILABLE,
        SafetyStatus.DROP_ATTACHED_UNAVAILABLE,
    }
)


def explains(drop_reason: int, failure_kind: int) -> bool:
    """Whether a safety failure of ``failure_kind`` can be the cause of this latch.

    Example:
        >>> explains(SafetyStatus.KIND_COLLISION, FailureTrigger.KIND_COLLISION)
        True
        >>> explains(SafetyStatus.KIND_COLLISION, FailureTrigger.KIND_HUMAN)
        False
    """
    if drop_reason == SafetyStatus.DROP_EXTERNAL_ESTOP:
        return bool(failure_kind in (FailureTrigger.KIND_HUMAN, FailureTrigger.KIND_TIMEOUT))
    if drop_reason in _PERCEPTION_DROPS:
        return bool(failure_kind == FailureTrigger.KIND_PERCEPTION)
    return bool(
        drop_reason < SafetyStatus.DROP_ENVELOPE_UNCONFIGURED and failure_kind == drop_reason
    )


def _trigger_source(drop_reason: int, cause: FailureRecord | None) -> str:
    """Who stopped the machine: the kernel, unless the cause names a person or the deadman."""
    kind = cause.failure_kind if cause is not None else drop_reason
    if kind == FailureTrigger.KIND_HUMAN:
        return "human"
    if kind == FailureTrigger.KIND_TIMEOUT and drop_reason == SafetyStatus.DROP_EXTERNAL_ESTOP:
        return "deadman"
    return "kernel"


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
        self._active = False
        self._reset_state()

    def _reset_state(self) -> None:
        # ``None`` until the first status sample: the TRANSIENT_LOCAL topic replays the
        # current latch on every (re)configure, and that latch may already be recorded.
        self._latched: bool | None = None
        self._recent: deque[tuple[float, FailureRecord]] = deque(maxlen=32)
        self._pending: tuple[float, int, Any] | None = None  # (seen_at, ts_ns, status msg)
        self._pending_timer: Any = None
        self._session_start_ns: int | None = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open the failure-bus and safety-status subscriptions."""
        del state
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
        """Open the session and start recording."""
        del state
        self._active = True
        self._session_start_ns = time.time_ns()
        self._emit_session(end_ts_ns=None)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop recording and close the session."""
        del state
        self.close_session()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release the subscriptions and forget the latch state."""
        del state
        self.close_session()
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []
        self._emitter = None
        self._reset_state()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Close an open session, then clean up."""
        return self.on_cleanup(state)

    def close_session(self) -> None:
        """Record a stop still waiting for its cause, then the session's end; stop recording.

        Idempotent. ``main`` calls it on SIGINT, where rclpy destroys the node without
        running the shutdown transition.
        """
        if self._pending is not None:
            self._emit_pending_stop(cause=None)
        if self._active and self._session_start_ns is not None and self._emitter is not None:
            self._emit_session(end_ts_ns=time.time_ns())
        self._active = False
        self._session_start_ns = None

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

    def _on_failure(self, source: str, msg: Any) -> None:  # reason: rclpy message
        """One ``FailureTrigger`` → one ``FailureRecord``; it may explain a waiting latch."""
        if not self._active or self._emitter is None:
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
            trace_id=msg.trace_id,
        )
        if source != FailureSource.SAFETY.value:
            return
        self._recent.append((time.monotonic(), record))
        if self._pending is not None and explains(
            self._pending[2].drop_reason, record.failure_kind
        ):
            self._emit_pending_stop(cause=record)

    def _on_safety_status(self, msg: Any) -> None:  # reason: rclpy message
        """A transition into ``latched`` is a safety stop; the latch clearing is not."""
        was_latched, self._latched = self._latched, bool(msg.latched)
        if was_latched is None:
            if msg.latched:
                self.get_logger().info(
                    "compliance.latch_inherited: already latched at configure; not re-recorded"
                )
            return
        if was_latched or not msg.latched or not self._active:
            return
        now = time.monotonic()
        cause = next(
            (
                failure
                for seen_at, failure in reversed(self._recent)
                if now - seen_at <= STOP_EXPLAIN_WINDOW_S
                and explains(msg.drop_reason, failure.failure_kind)
            ),
            None,
        )
        if self._pending is not None:  # a second latch before the first's cause arrived
            self._emit_pending_stop(cause=None)
        self._pending = (now, time.time_ns(), msg)
        if cause is not None:
            self._emit_pending_stop(cause=cause)
            return
        self._pending_timer = self.create_timer(
            STOP_EXPLAIN_WINDOW_S, lambda: self._emit_pending_stop(cause=None)
        )

    def _emit_pending_stop(self, *, cause: FailureRecord | None) -> None:
        """Record the waiting latch, with its cause if one arrived in the window."""
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self.destroy_timer(self._pending_timer)
            self._pending_timer = None
        if self._pending is None or self._emitter is None:
            return
        _, ts_ns, msg = self._pending
        self._pending = None
        self._emitter.emit(
            SafetyStopRecord,
            ts_ns=ts_ns,
            trigger_source=_trigger_source(msg.drop_reason, cause),
            drop_reason=msg.drop_reason,
            detail=msg.detail,
            evidence=cause.evidence if cause is not None else None,
            latched=True,
            rskill_id=msg.rskill_id or None,
            trace_id=msg.trace_id,
            parent_record_id=cause.record_id if cause is not None else None,
        )


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
            # rclpy destroys a lifecycle node without its shutdown transition, so the
            # session's end (and a stop still waiting for its cause) is recorded here.
            node.close_session()
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if already shut down
    return 0


if __name__ == "__main__":
    sys.exit(main())
