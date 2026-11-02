"""Compliance records out over OTLP (issue #467).

``ComplianceEmitter`` stamps the ``openral_core.ComplianceRecord`` envelope (record id,
a per-emitter ``stream_id`` + gap-free ``seq``, clock source) and sends each record as
one OTLP **log record**:
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

import functools
import ipaddress
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar, cast
from urllib.parse import urlsplit

import structlog
from openral_core import ComplianceRecord, FailureRecord, SoftwareComponent, trace_ids_from_wire
from openral_core.exceptions import ROSRuntimeError
from opentelemetry import trace
from opentelemetry._logs import Logger, LoggerProvider, LogRecord, SeverityNumber, get_logger
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from openral_observability import metrics, semconv
from openral_observability.compliance_log import ComplianceLog

__all__ = [
    "CLOCK_SOURCE_ENV",
    "ComplianceEmitter",
    "images_may_leave_host",
    "send_record",
    "software_inventory",
]

CLOCK_SOURCE_ENV = "OPENRAL_CLOCK_SOURCE"
"""How the host clock is disciplined (``ptp`` / ``ntp`` / ``monotonic_unsynced``)."""

_IMAGE_CONSENT_ENV = "OPENRAL_TELEMETRY_IMAGE_CONSENT"

log = structlog.get_logger(__name__)

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
        log: The on-robot compliance log each record is also written to, before it goes out
            over OTLP (#469). ``None``: OTLP only.
        on_log_error: Called with ``(record, error)`` when a record went out over OTLP but
            could not be written to ``log``; ``emit`` then still returns the record. Without
            it, ``emit`` raises ``ROSRuntimeError``.
    """

    def __init__(
        self,
        clock_source: _ClockSource | None = None,
        *,
        logger_provider: LoggerProvider | None = None,
        log: ComplianceLog | None = None,
        on_log_error: Callable[[ComplianceRecord, OSError], None] | None = None,
    ) -> None:
        """Open a fresh stream: a new ``stream_id``, ``seq`` from 0."""
        self._clock_source: _ClockSource = clock_source or _clock_source_from_env()
        self._stream_id = uuid.uuid4().hex
        self._next_seq = 0
        self._lock = threading.Lock()
        self._logger = get_logger("openral.compliance", logger_provider=logger_provider)
        self._log = log
        self._on_log_error = on_log_error

    def emit(self, kind: type[_R], **fields: object) -> _R:
        """Build one record of ``kind`` from ``fields`` and send it.

        ``record_id``, ``stream_id``, ``seq``, ``clock_source`` and (unless given) ``ts_ns``
        and the active span's ``trace_id`` / ``span_id`` are filled in here. A ``trace_id``
        given in a wire form (traceparent, ``""``) is parsed; a malformed one is dropped with
        a warning, because a producer's bad trace field must cost the link, not the record.

        Raises:
            pydantic.ValidationError: ``fields`` do not make a valid record. Callers on a
                safety path build from typed sources, so this is a programming error.
            ROSRuntimeError: The record could not be written to the on-robot log and no
                ``on_log_error`` is set (it was still sent over OTLP, and its ``seq`` is spent,
                so ``verify`` shows the gap).
        """
        wire = fields.get("trace_id")
        if isinstance(wire, str):
            try:
                fields["trace_id"], span_id = trace_ids_from_wire(wire)
            except ValueError:
                log.warning("compliance.trace_id_malformed", trace_id=wire[:64], kind=kind.__name__)
                fields["trace_id"], span_id = None, None
            fields.setdefault("span_id", span_id)
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
                        "stream_id": self._stream_id,
                        "seq": self._next_seq,
                        "ts_ns": time.time_ns(),
                        "clock_source": self._clock_source,
                        **fields,
                    }
                ),
            )
            self._next_seq += 1  # only a record that validated takes a seq: no false gap
            write_error: OSError | None = None
            if self._log is not None:
                try:
                    self._log.append(record)
                except OSError as exc:
                    write_error = exc
            send_record(self._logger, record)
        if write_error is not None and self._on_log_error is not None:
            self._on_log_error(record, write_error)
        elif write_error is not None:
            raise ROSRuntimeError(
                f"compliance log write failed for {record.kind} record {record.record_id}: "
                f"{write_error}"
            ) from write_error
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


def send_record(logger: Logger, record: ComplianceRecord) -> None:
    """Send one record as an OTLP log record (``openral compliance ship`` reuses this)."""
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
    logger.emit(
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


def _digest(path: Path) -> str | None:
    """SHA-256 of a file, or of a package directory's ``*.py`` sources (names and bytes).

    ``None`` (and a warning) when it cannot be read: the component is still listed, so an
    unreadable binary never hides the rest of the inventory.
    """
    import hashlib

    h = hashlib.sha256()
    try:
        if path.is_file():
            h.update(path.read_bytes())
        else:
            for f in sorted(path.rglob("*.py")):
                h.update(str(f.relative_to(path)).encode() + b"\0" + f.read_bytes())
    except OSError as exc:
        log.warning("compliance.inventory_unreadable", path=str(path), error=str(exc))
        return None
    return h.hexdigest()


def software_inventory() -> list[SoftwareComponent]:
    """What is installed that the machine needs, for MR Annex III §1.1.9 identification.

    Every ``openral*`` Python distribution (name + version), plus — when a ROS overlay is
    sourced — the safety software, marked ``safety_relevant`` with a content digest: the C++
    kernel's ``safety_kernel_node`` binary and the ``openral_safety`` supervisor's sources.
    Software that is not installed is left out, never invented.

    Example:
        >>> any(c.name == "openral-observability" for c in software_inventory())
        True
    """
    import importlib.util
    import re
    from importlib import metadata

    components = [
        SoftwareComponent(name=dist.metadata["Name"], version=dist.version)
        for dist in metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("openral")
    ]
    try:
        from ament_index_python.packages import (  # type: ignore[import-not-found,unused-ignore]  # reason: ROS overlay, optional
            PackageNotFoundError,
            get_package_prefix,
        )
    except ImportError:
        return sorted(components, key=lambda c: c.name)

    def version_of(prefix: Path, package: str) -> str:
        xml = prefix / "share" / package / "package.xml"
        try:
            match = re.search(r"<version>([^<]+)</version>", xml.read_text())
        except OSError:
            match = None
        return match[1] if match else "unknown"

    try:
        prefix = Path(get_package_prefix("openral_safety_kernel"))
        binary = prefix / "lib" / "openral_safety_kernel" / "safety_kernel_node"
        components.append(
            SoftwareComponent(
                name="openral_safety_kernel",
                version=version_of(prefix, "openral_safety_kernel"),
                sha256=_digest(binary) if binary.exists() else None,
                safety_relevant=True,
            )
        )
    except PackageNotFoundError:
        pass
    spec = importlib.util.find_spec("openral_safety")
    if spec is not None and spec.origin is not None:
        try:
            version = version_of(Path(get_package_prefix("openral_safety")), "openral_safety")
        except PackageNotFoundError:
            version = "unknown"
        components.append(
            SoftwareComponent(
                name="openral_safety",
                version=version,
                sha256=_digest(Path(spec.origin).parent),
                safety_relevant=True,
            )
        )
    return sorted(components, key=lambda c: c.name)


def images_may_leave_host(endpoint: str | None = None) -> bool:
    """Whether camera images may ride in exported telemetry.

    Yes when nothing is exported, or the collector is on the deployer's own network: a
    loopback, private or link-local address, or a single-label host name (a compose service
    such as ``otelcol``, a LAN machine). A public endpoint needs the deployer's consent,
    ``OPENRAL_TELEMETRY_IMAGE_CONSENT=1`` (CLAUDE.md §3: no PII in cloud logs without
    consent; a camera frame of a cell is personal data when a worker is in it). Withholding
    is logged once per endpoint, so blank camera tiles are never a silent mystery.

    Args:
        endpoint: The OTLP endpoint; default the one ``configure_observability`` installed.

    Example:
        >>> images_may_leave_host("http://127.0.0.1:4318")
        True
        >>> images_may_leave_host("http://otelcol:4317")
        True
    """
    if endpoint is None:
        from openral_observability import _sdk

        endpoint = _sdk._endpoint
    return _images_allowed(endpoint or "", os.environ.get(_IMAGE_CONSENT_ENV) == "1")


@functools.lru_cache(maxsize=8)
def _images_allowed(endpoint: str, consent: bool) -> bool:
    """The decision for one ``(endpoint, consent)``; cached so it is logged once."""
    if consent or not endpoint:
        return True
    host = urlsplit(endpoint if "//" in endpoint else f"//{endpoint}").hostname or ""
    try:
        address = ipaddress.ip_address(host)
        local = address.is_loopback or address.is_private or address.is_link_local
    except ValueError:
        local = host == "localhost" or "." not in host
    if not local:
        log.warning(
            "telemetry.images_withheld",
            endpoint=endpoint,
            hint=f"set {_IMAGE_CONSENT_ENV}=1 if the deployer consents to camera frames leaving",
        )
    return local
