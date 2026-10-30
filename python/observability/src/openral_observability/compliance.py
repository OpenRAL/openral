"""Compliance records out over OTLP (issue #467).

``ComplianceEmitter`` stamps the ``openral_core.ComplianceRecord`` envelope (record id,
per-process sequence, clock source) and sends each record as one OTLP **log record**:
the body is the record's JSON, the attributes carry ``openral.record.kind`` /
``openral.schema.version`` / ``openral.record.id`` / ``openral.retention.class``, and the
record's trace id links it to the span it happened under. Unit, org and site ride on
the OTel resource (``semconv.UNIT_SERIAL`` …), not on the record.

With no OTLP endpoint configured the global logger provider is the no-op one, so
emitting costs a model build and nothing else.

Example:
    >>> from openral_core import FailureRecord
    >>> emitter = ComplianceEmitter(clock_source="ntp")
    >>> record = emitter.emit(FailureRecord, source="safety", failure_kind=7, severity=3)
    >>> (record.seq, record.retention_class.value)
    (0, 'ai_log_6m')
"""

from __future__ import annotations

import itertools
import os
import threading
import time
import uuid
from typing import Literal, TypeVar, cast

from openral_core import ComplianceRecord, FailureRecord
from opentelemetry import trace
from opentelemetry._logs import LoggerProvider, LogRecord, SeverityNumber, get_logger
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from openral_observability import metrics, semconv

__all__ = ["CLOCK_SOURCE_ENV", "ComplianceEmitter", "images_may_leave_host"]

CLOCK_SOURCE_ENV = "OPENRAL_CLOCK_SOURCE"
"""How the host clock is disciplined (``ptp`` / ``ntp`` / ``monotonic_unsynced``)."""

_IMAGE_CONSENT_ENV = "OPENRAL_TELEMETRY_IMAGE_CONSENT"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

_ClockSource = Literal["ptp", "ntp", "monotonic_unsynced"]
_R = TypeVar("_R", bound=ComplianceRecord)

# A record's severity on the OTLP log: stops and aborts are what an auditor filters for.
_SEVERITY: dict[str, SeverityNumber] = {
    "safety_stop": SeverityNumber.WARN,
    "security_incident": SeverityNumber.ERROR,
}
_FAILURE_SEVERITY = (
    SeverityNumber.INFO,
    SeverityNumber.WARN,
    SeverityNumber.ERROR,
    SeverityNumber.FATAL,
)


def _clock_source_from_env() -> _ClockSource:
    value = os.environ.get(CLOCK_SOURCE_ENV, "monotonic_unsynced")
    if value not in ("ptp", "ntp", "monotonic_unsynced"):
        return "monotonic_unsynced"
    return value  # type: ignore[return-value]  # reason: narrowed by the membership test


class ComplianceEmitter:
    """Builds compliance records with a fresh envelope and sends them over OTLP.

    One emitter per process that writes records, so ``seq`` is gap-free within that
    process. Thread-safe.

    Args:
        clock_source: How this host's clock is disciplined. Default: ``$OPENRAL_CLOCK_SOURCE``,
            else ``monotonic_unsynced`` (the honest answer when nobody said).
        logger_provider: Where records go. Default: the global provider that
            ``configure_observability`` installs.
    """

    def __init__(
        self,
        clock_source: _ClockSource | None = None,
        *,
        logger_provider: LoggerProvider | None = None,
    ) -> None:
        """Start the sequence at 0."""
        self._clock_source: _ClockSource = clock_source or _clock_source_from_env()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._logger = get_logger("openral.compliance", logger_provider=logger_provider)

    def emit(self, kind: type[_R], **fields: object) -> _R:
        """Build one record of ``kind`` from ``fields`` and send it.

        ``record_id``, ``seq``, ``clock_source`` and (unless given) ``ts_ns`` and the
        active span's ``trace_id`` / ``span_id`` are filled in here.

        Raises:
            pydantic.ValidationError: ``fields`` do not make a valid record. Callers on a
                safety path build from typed sources, so this is a programming error.
        """
        if "trace_id" not in fields:
            ctx = trace.get_current_span().get_span_context()
            if ctx.is_valid:
                fields["trace_id"] = format(ctx.trace_id, "032x")
                fields["span_id"] = format(ctx.span_id, "016x")
        with self._lock:
            # mypy widens model_validate on type[_R] to the whole union; _R is what was built.
            record = cast(
                _R,
                kind.model_validate(
                    {
                        "record_id": uuid.uuid4().hex,
                        "seq": next(self._seq),
                        "ts_ns": time.time_ns(),
                        "clock_source": self._clock_source,
                        **fields,
                    }
                ),
            )
            self._send(record)
        if isinstance(record, FailureRecord):
            metrics.get_failure_count().add(
                1,
                {
                    semconv.LABEL_COMPONENT: record.source,
                    semconv.LABEL_KIND: record.failure_kind,
                    semconv.LABEL_SEVERITY: record.severity,
                },
            )
        return record

    def _send(self, record: ComplianceRecord) -> None:
        context = None
        if record.trace_id is not None:
            span_context = SpanContext(
                trace_id=int(record.trace_id, 16),
                span_id=int(record.span_id or "1", 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            context = trace.set_span_in_context(NonRecordingSpan(span_context))
        if isinstance(record, FailureRecord):
            severity = _FAILURE_SEVERITY[record.severity]
        else:
            severity = _SEVERITY.get(record.kind, SeverityNumber.INFO)
        self._logger.emit(
            LogRecord(
                timestamp=record.ts_ns,
                context=context,
                severity_number=severity,
                body=record.model_dump_json(),
                attributes={
                    semconv.RECORD_KIND: record.kind,
                    semconv.RECORD_ID: record.record_id,
                    semconv.RECORD_SCHEMA_VERSION: record.schema_version,
                    semconv.RECORD_RETENTION_CLASS: record.retention_class.value,
                },
            )
        )


def images_may_leave_host(endpoint: str | None = None) -> bool:
    """Whether camera images may ride in exported telemetry.

    Only to a loopback collector (the local ``openral dashboard``), unless the deployer
    consented with ``OPENRAL_TELEMETRY_IMAGE_CONSENT=1`` (CLAUDE.md §3: no PII in cloud
    logs without consent; a camera frame of a cell is personal data when a worker is in it).

    Args:
        endpoint: The OTLP endpoint; default ``$OTEL_EXPORTER_OTLP_ENDPOINT``.

    Example:
        >>> images_may_leave_host("http://127.0.0.1:4318")
        True
        >>> images_may_leave_host("https://otlp.example.com")
        False
    """
    if os.environ.get(_IMAGE_CONSENT_ENV) == "1":
        return True
    from urllib.parse import urlsplit

    url = endpoint if endpoint is not None else os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not url:
        return True  # nothing is exported
    host = urlsplit(url if "//" in url else f"//{url}").hostname or ""
    return host in _LOOPBACK_HOSTS
