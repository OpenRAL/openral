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
* each latch clearing → ``InterventionRecord(estop_reset)`` (#468). The recorder sees every
  reset, whatever path made it; an ``OperatorAction`` on ``/openral/operator_action`` up to
  ``STOP_EXPLAIN_WINDOW_S`` either side says who (channel + pseudonymous actor). A reset
  nobody announced is recorded as ``channel="unknown"``, ``legitimate=None``: MR Annex III
  §1.1.9 asks for evidence of illegitimate interventions too;
* an announced ``param_set`` → ``InterventionRecord(param_set)``;
* activate / deactivate (and process shutdown) → ``SessionRecord`` start / end, plus a
  ``SoftwareInventoryRecord`` at every activate.

Only an ACTIVE node records. Records go to the on-robot log (#469) and out as OTLP log
records (``ComplianceEmitter``). The recorder is not on the actuation path; the one thing it
publishes is a stop: when a record cannot be written to the log it publishes
``/openral/estop`` and a ``FailureTrigger(KIND_CONTROLLER, SEVERITY_ABORT)`` — a machine that
cannot keep its safety evidence does not move (#468, hazard-log entry). It warns while the
log's filesystem is under ``DISK_WARN_FREE_FRACTION`` free.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from collections import deque
from typing import Any, TypeVar

import rclpy
from openral_core import (
    SITE_ID_ENV,
    Actor,
    ComplianceRecord,
    ControllerEvidence,
    FailureEvidence,
    FailureRecord,
    InterventionRecord,
    SafetyStopRecord,
    SessionRecord,
    SoftwareInventoryRecord,
)
from openral_msgs.msg import FailureTrigger, OperatorAction, SafetyStatus
from openral_observability import configure_observability
from openral_observability.compliance import ComplianceEmitter, software_inventory
from openral_observability.compliance_log import ComplianceLog
from openral_observability.failure_bus import FailureSource, topic_for
from pydantic import TypeAdapter, ValidationError
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from std_msgs.msg import Empty

__all__ = [
    "DISK_WARN_FREE_FRACTION",
    "OPERATOR_ACTION_TOPIC",
    "STOP_EXPLAIN_WINDOW_S",
    "ComplianceRecorderNode",
    "explains",
    "main",
]

_EVIDENCE: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)
_R = TypeVar("_R", bound=ComplianceRecord)

OPERATOR_ACTION_TOPIC = "/openral/operator_action"
"""Where channels announce who is acting (``openral_msgs/OperatorAction``)."""

DISK_CHECK_PERIOD_S = 30.0
DISK_WARN_FREE_FRACTION = 0.2
"""Below this much free space under the log, the recorder warns (it stops when writes fail)."""

_HEX = re.compile(r"[0-9a-f]{64}")

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


_CHANNELS = frozenset({"local", "remote", "ota", "dashboard", "pendant", "chatops", "cli"})


def _channel(label: str) -> str:
    """``InterventionRecord.channel`` for an announced channel label; other labels are remote."""
    return label if label in _CHANNELS else "remote"


def _actor(msg: Any) -> Actor:  # reason: rclpy message
    """The ``Actor`` an ``OperatorAction`` names; a malformed id is dropped, never stored."""
    actor_id = msg.actor_id if len(msg.actor_id) == 64 and _HEX.fullmatch(msg.actor_id) else None
    return Actor(channel_label=msg.channel_label or "unlabelled", actor_id=actor_id)


def _legitimacy(actor: Actor) -> bool | None:
    """``True`` only for an identified operator; an anonymous claim proves nothing (``None``).

    Identified means a pseudonymous id under the deployer's key, which only a channel holding
    that key can produce. Authentication proper (logins) is #36.
    """
    return True if actor.actor_id is not None else None


class ComplianceRecorderNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Records failures, safety stops and sessions as compliance records.

    Parameters:
        ``run_mode``: ``"sim"`` or ``"hardware"`` for the ``SessionRecord`` (default ``"sim"``).
    """

    def __init__(self, node_name: str = "openral_compliance_recorder") -> None:
        """Declare parameters; opens no resources until on_configure."""
        super().__init__(node_name)
        self.declare_parameter("run_mode", "sim")
        # Real hardware: a recorder that stops recording (deactivated while the graph runs)
        # stops the machine too, the same way a failed log write does.
        self.declare_parameter("stop_when_inactive", False)
        self._emitter: ComplianceEmitter | None = None
        self._log: ComplianceLog | None = None
        self._subs: list[Any] = []
        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._disk_timer: Any = None
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
        self._actions: deque[tuple[float, Any]] = deque(maxlen=32)
        self._pending_reset: tuple[int, Any] | None = None  # (ts_ns, timer)
        self._log_failure_stopped = False
        self._write_failed = False
        self._disk_low = False

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open the failure-bus and safety-status subscriptions."""
        del state
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

        try:
            self._log = ComplianceLog.from_env(writer="recorder")
        except OSError as exc:
            # A recorder that cannot open its log must not come up looking healthy: refuse
            # to configure (on real hardware the launch then ends the graph).
            self.get_logger().error(f"compliance.log_unavailable: {exc}")
            return TransitionCallbackReturn.FAILURE
        self._emitter = ComplianceEmitter(log=self._log, on_log_error=self._on_log_error)
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
        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._subs.append(
            self.create_subscription(
                OperatorAction, OPERATOR_ACTION_TOPIC, self._on_operator_action, estop_qos
            )
        )
        # The one thing the recorder publishes: a stop when it cannot keep the evidence.
        self._estop_pub = self.create_publisher(Empty, "/openral/estop", estop_qos)
        self._failure_pub = self.create_publisher(
            FailureTrigger, topic_for(FailureSource.SAFETY), bus_qos
        )
        if self._log is not None:
            self._disk_timer = self.create_timer(DISK_CHECK_PERIOD_S, self._check_disk)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open the session and start recording."""
        del state
        self._active = True
        self._session_start_ns = time.time_ns()
        self._emit_session(end_ts_ns=None)
        # What is installed, at every start (MR Annex III §1.1.9): a changed digest between two
        # sessions is the evidence of a software intervention.
        self._record(SoftwareInventoryRecord, components=software_inventory())
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop recording and close the session; with ``stop_when_inactive``, stop the machine."""
        del state
        self.close_session()
        if self.get_parameter("stop_when_inactive").get_parameter_value().bool_value:
            self._log_failure_stopped = False
            self._stop_for_lost_evidence("the compliance recorder was deactivated", "inactive")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release the subscriptions and forget the latch state."""
        del state
        self.close_session()
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []
        for pub in (self._estop_pub, self._failure_pub):
            if pub is not None:
                self.destroy_publisher(pub)
        self._estop_pub = self._failure_pub = None
        if self._disk_timer is not None:
            self.destroy_timer(self._disk_timer)
            self._disk_timer = None
        self._emitter = None
        self._log = None
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
        if self._pending_reset is not None:
            self._emit_reset(action=None)
        if self._active and self._session_start_ns is not None and self._emitter is not None:
            self._emit_session(end_ts_ns=time.time_ns())
        self._active = False
        self._session_start_ns = None

    def _record(self, kind: type[_R], **fields: object) -> _R | None:
        """Emit one record; a record that cannot be built or logged never ends the node.

        An out-of-range ``kind`` / ``severity`` is a producer bug: recording it as some other
        value would be false evidence, and raising out of a callback would end ``rclpy.spin``
        and every later record. A record that went out over OTLP but not into the on-robot log
        is still returned, so it can explain a stop; ``_on_log_error`` stops the machine.
        """
        assert self._emitter is not None
        self._write_failed = False
        try:
            record = self._emitter.emit(kind, **fields)
        except ValidationError as exc:
            self.get_logger().error(
                f"compliance.record_invalid {kind.__name__}: {exc.error_count()} error(s) "
                f"in {sorted(fields)}"
            )
            return None
        if not self._write_failed:
            self._log_failure_stopped = False  # the log writes again
        return record

    def _on_log_error(self, record: ComplianceRecord, error: OSError) -> None:
        """A record went out over OTLP but not into the on-robot log: stop the machine."""
        self._write_failed = True
        self.get_logger().error(
            f"compliance.log_write_failed {record.kind} {record.record_id}: {error}"
        )
        self._stop_for_lost_evidence(str(error))

    def _stop_for_lost_evidence(self, detail: str, state: str = "log_write_failed") -> None:
        """Latch the machine: one that cannot keep its safety evidence does not move.

        Publishes ``/openral/estop`` (every latch holder — kernel, HALs, runner — subscribes)
        and a ``FailureTrigger(KIND_CONTROLLER, SEVERITY_ABORT)`` saying why. Once per
        failure episode: the recorder hears its own trigger, fails to log that too, and must
        not loop. A reset while the log still fails is itself unloggable, so it latches again.
        """
        if self._log_failure_stopped or self._estop_pub is None or self._failure_pub is None:
            return
        self._log_failure_stopped = True
        self._estop_pub.publish(Empty())
        trigger = FailureTrigger()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_CONTROLLER
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        trigger.evidence_json = ControllerEvidence(
            controller_name=self.get_name(), state=state, detail=detail[:200]
        ).model_dump_json()
        self._failure_pub.publish(trigger)
        self.get_logger().error(f"compliance.estop_published ({state}): {detail[:200]}")

    def _check_disk(self) -> None:
        """Warn while the log's filesystem is below ``DISK_WARN_FREE_FRACTION`` free."""
        if self._log is None:
            return
        usage = shutil.disk_usage(self._log.root)
        low = usage.free / usage.total < DISK_WARN_FREE_FRACTION
        if low and not self._disk_low:
            self.get_logger().warning(
                f"compliance.disk_low: {usage.free / 2**30:.1f} GiB free under "
                f"{self._log.root}; the machine stops when the log can no longer be written"
            )
        self._disk_low = low

    def _on_operator_action(self, msg: Any) -> None:  # reason: rclpy message
        """An announced operator action: who did the next reset, or a parameter change."""
        if not self._active:
            return
        if msg.action == OperatorAction.ACTION_PARAM_SET:
            self._record(
                InterventionRecord,
                intervention_kind="param_set",
                actor=(actor := _actor(msg)),
                channel=_channel(msg.channel_label),
                legitimate=_legitimacy(actor),
                diff_ref=msg.detail or None,
            )
            return
        if msg.action != OperatorAction.ACTION_ESTOP_RESET:
            return
        if self._pending_reset is not None:  # the latch cleared first; this says who
            self._emit_reset(action=msg)
        else:
            self._actions.append((time.monotonic(), msg))

    def _on_latch_cleared(self) -> None:
        """A latch cleared: a reset happened. Attribute it, or wait briefly for who did it.

        Re-arms the lost-evidence stop: a reset whose record cannot be written either (the log
        still fails) stops the machine again.
        """
        self._log_failure_stopped = False
        now = time.monotonic()
        action = next(
            (a for seen, a in reversed(self._actions) if now - seen <= STOP_EXPLAIN_WINDOW_S),
            None,
        )
        if self._pending_reset is not None:
            self._emit_reset(action=None)
        timer = None
        if action is None:
            timer = self.create_timer(STOP_EXPLAIN_WINDOW_S, lambda: self._emit_reset(action=None))
        self._pending_reset = (time.time_ns(), timer)
        if action is not None:
            self._actions.clear()
            self._emit_reset(action=action)

    def _emit_reset(self, *, action: Any) -> None:  # reason: rclpy message or None
        """Record the waiting reset; unannounced means unattributed, legitimacy unknown."""
        if self._pending_reset is None:
            return
        ts_ns, timer = self._pending_reset
        self._pending_reset = None
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
        actor = _actor(action) if action is not None else Actor(channel_label="unattributed")
        self._record(
            InterventionRecord,
            ts_ns=ts_ns,
            intervention_kind="estop_reset",
            actor=actor,
            channel=_channel(action.channel_label) if action is not None else "unknown",
            legitimate=_legitimacy(actor),
        )

    def _emit_session(self, *, end_ts_ns: int | None) -> None:
        assert self._emitter is not None and self._session_start_ns is not None
        run_mode = self.get_parameter("run_mode").get_parameter_value().string_value
        self._record(
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
        record = self._record(
            FailureRecord,
            source=source,
            failure_kind=msg.kind,
            severity=msg.severity,
            evidence=evidence,
            rskill_id=msg.rskill_id or None,
            trace_id=msg.trace_id,
        )
        if record is None or source != FailureSource.SAFETY.value:
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
        if was_latched and not msg.latched and self._active:
            self._on_latch_cleared()
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
        self._record(
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
