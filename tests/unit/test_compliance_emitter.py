"""Compliance records over OTLP and unit identity (issue #467).

The OTLP side reads records back from the SDK's in-memory log exporter: the
process boundary is the only double (CLAUDE.md §1.11).

Run with:
    uv run pytest tests/unit/test_compliance_emitter.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_core import (
    ComplianceRecord,
    FailureRecord,
    HumanEvidence,
    ROSConfigError,
    SafetyStopRecord,
    resolve_unit_identity,
)
from openral_observability import semconv
from openral_observability.compliance import ComplianceEmitter, images_may_leave_host
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from pydantic import TypeAdapter

_REPO = Path(__file__).resolve().parents[2]
_ADAPTER: TypeAdapter[ComplianceRecord] = TypeAdapter(ComplianceRecord)


@pytest.fixture
def exported() -> tuple[ComplianceEmitter, InMemoryLogRecordExporter]:
    """An emitter whose records land in an in-memory OTLP log exporter."""
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: OTel SDK leaves __init__ unannotated
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return ComplianceEmitter(clock_source="ntp", logger_provider=provider), exporter


def test_record_goes_out_as_one_log_record_with_its_json_body(
    exported: tuple[ComplianceEmitter, InMemoryLogRecordExporter],
) -> None:
    """Body decodes back to the record; attributes name kind, version, id, retention."""
    emitter, exporter = exported
    record = emitter.emit(
        FailureRecord,
        source="safety",
        failure_kind=7,
        severity=3,
        evidence=HumanEvidence(actor="pendant", reason="human_estop"),
    )
    (log,) = exporter.get_finished_logs()
    body = log.log_record.body
    assert isinstance(body, str)
    assert _ADAPTER.validate_json(body) == record
    attrs = dict(log.log_record.attributes or {})
    assert attrs[semconv.RECORD_KIND] == "failure"
    assert attrs[semconv.RECORD_ID] == record.record_id
    assert attrs[semconv.RECORD_SCHEMA_VERSION] == "0.1"
    assert attrs[semconv.RECORD_RETENTION_CLASS] == "ai_log_6m"
    assert log.log_record.timestamp == record.ts_ns


def test_seq_is_gap_free_and_the_active_span_links_the_record(
    exported: tuple[ComplianceEmitter, InMemoryLogRecordExporter],
) -> None:
    """Consecutive records count up; a record emitted under a span carries its ids."""
    emitter, exporter = exported
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("safety.check") as span:
        stop = emitter.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    after = emitter.emit(SafetyStopRecord, trigger_source="human", latched=True)
    assert (stop.seq, after.seq) == (0, 1)
    ctx = span.get_span_context()
    assert stop.trace_id == format(ctx.trace_id, "032x")
    assert stop.span_id == format(ctx.span_id, "016x")
    assert after.trace_id is None
    first = exporter.get_finished_logs()[0].log_record
    assert first.trace_id == ctx.trace_id
    assert json.loads(str(first.body))["kind"] == "safety_stop"
    assert trace.get_current_span().get_span_context().is_valid is False


def test_images_stay_home_without_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loopback dashboard: allowed. Remote collector: only with explicit consent."""
    monkeypatch.delenv("OPENRAL_TELEMETRY_IMAGE_CONSENT", raising=False)
    assert images_may_leave_host("http://127.0.0.1:4318")
    assert images_may_leave_host("localhost:4317")
    assert images_may_leave_host("http://otelcol:4317")  # docker-compose.dev.yml
    assert images_may_leave_host("http://192.168.1.20:4318")  # a laptop dashboard on the LAN
    assert not images_may_leave_host("https://ingest.example.eu")
    assert not images_may_leave_host("http://8.8.8.8:4317")
    monkeypatch.setenv("OPENRAL_TELEMETRY_IMAGE_CONSENT", "1")
    assert images_may_leave_host("https://ingest.example.eu")


def test_unit_identity_from_the_unit_file_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env wins over the unit file; a real deploy without a serial is refused."""
    for var in ("OPENRAL_ROBOT_UNIT", "OPENRAL_UNIT_SERIAL", "OPENRAL_ORG_ID", "OPENRAL_SITE_ID"):
        monkeypatch.delenv(var, raising=False)
    so101 = _REPO / "robots" / "so101_follower" / "robot.yaml"
    franka = _REPO / "robots" / "franka_panda" / "robot.yaml"
    with pytest.raises(ROSConfigError, match="OPENRAL_UNIT_SERIAL"):
        resolve_unit_identity(franka, None, required=True)
    monkeypatch.setenv("OPENRAL_UNIT_SERIAL", "SO101-0042")
    monkeypatch.setenv("OPENRAL_SITE_ID", "lab-bcn")
    identity = resolve_unit_identity(so101, "bench_laptop", required=True)
    assert identity.serial == "SO101-0042"
    assert identity.site_id == "lab-bcn"
    monkeypatch.setenv("OPENRAL_UNIT_SERIAL", "has,comma")
    with pytest.raises(ROSConfigError, match="OTEL_RESOURCE_ATTRIBUTES"):
        resolve_unit_identity(so101, "bench_laptop", required=False)


def test_a_malformed_wire_trace_costs_the_link_not_the_record(
    exported: tuple[ComplianceEmitter, InMemoryLogRecordExporter],
) -> None:
    """Every producer gets the guard: bad trace field → record without a trace id."""
    emitter, _ = exported
    record = emitter.emit(
        SafetyStopRecord, trigger_source="supervisor", latched=False, trace_id="not-a-trace"
    )
    assert (record.trace_id, record.span_id) == (None, None)
    wire = emitter.emit(
        SafetyStopRecord,
        trigger_source="kernel",
        latched=True,
        trace_id=f"00-{'c' * 32}-{'d' * 16}-01",
    )
    assert (wire.trace_id, wire.span_id) == ("c" * 32, "d" * 16)


def test_each_emitter_is_its_own_stream() -> None:
    """Two processes' emitters never share a (stream_id, seq), so gaps stay meaningful."""
    a, b = ComplianceEmitter(), ComplianceEmitter()
    ra = a.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    rb = b.emit(SafetyStopRecord, trigger_source="kernel", latched=True)
    assert ra.seq == rb.seq == 0
    assert ra.stream_id != rb.stream_id
