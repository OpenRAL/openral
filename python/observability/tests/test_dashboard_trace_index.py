"""ADR-0018 F7 — dashboard per-trace span index + /api/traces / /api/spans.

Exercises the real :class:`TelemetryStore` against real
:class:`opentelemetry.proto.trace.v1.trace_pb2.Span` protobuf payloads
posted through a real ``ASGITransport`` to the FastAPI app — no mocks
per CLAUDE.md §1.11.
"""

from __future__ import annotations

import httpx
import pytest
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.store import _TRACE_INDEX_MAX_TRACES
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span


def _av(value: object) -> AnyValue:
    if isinstance(value, str):
        return AnyValue(string_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    return AnyValue(string_value=str(value))


def _make_span(*, trace_id: bytes, span_id: bytes, name: str, start_ns: int) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        name=name,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=start_ns + 5_000_000,  # 5 ms
        attributes=[KeyValue(key="rskill.id", value=_av("smolvla-libero"))],
    )


def _otlp_payload(spans: list[Span]) -> bytes:
    req = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(attributes=[KeyValue(key="service.name", value=_av("ral"))]),
                scope_spans=[ScopeSpans(spans=spans)],
            )
        ]
    )
    return req.SerializeToString()


def test_store_indexes_spans_by_trace_id() -> None:
    """Two spans with the same trace_id appear under one lookup; distinct traces don't merge."""
    store = TelemetryStore()
    trace_a = b"\x11" * 16
    trace_b = b"\x22" * 16
    spans = [
        _make_span(trace_id=trace_a, span_id=b"\x01" * 8, name="rskill.execute", start_ns=1_000),
        _make_span(trace_id=trace_a, span_id=b"\x02" * 8, name="hal.send_action", start_ns=2_000),
        _make_span(trace_id=trace_b, span_id=b"\x03" * 8, name="safety.check", start_ns=3_000),
    ]
    payload = _otlp_payload(spans)
    req = ExportTraceServiceRequest.FromString(payload)
    store.ingest_spans(list(req.resource_spans))

    traces = store.list_traces()
    assert {t["trace_id"] for t in traces} == {trace_a.hex(), trace_b.hex()}
    bucket_a = {t["span_count"] for t in traces if t["trace_id"] == trace_a.hex()}
    assert bucket_a == {2}

    looked_up = store.lookup_trace(trace_a.hex())
    assert looked_up is not None
    assert [s["name"] for s in looked_up] == ["rskill.execute", "hal.send_action"]
    # Spans come back sorted by start_unix_ns.
    assert looked_up[0]["start_unix_ns"] < looked_up[1]["start_unix_ns"]

    assert store.lookup_trace("00" * 16) is None


def test_store_evicts_oldest_trace_when_index_full() -> None:
    """Once _TRACE_INDEX_MAX_TRACES distinct trace_ids land, FIFO eviction kicks in."""
    store = TelemetryStore()
    for i in range(_TRACE_INDEX_MAX_TRACES + 5):
        tid = i.to_bytes(16, "big")
        store.ingest_spans(
            list(
                ExportTraceServiceRequest.FromString(
                    _otlp_payload(
                        [
                            _make_span(
                                trace_id=tid,
                                span_id=b"\x99" * 8,
                                name="rskill.tick",
                                start_ns=1_000 + i,
                            )
                        ]
                    )
                ).resource_spans
            )
        )
    traces = store.list_traces()
    assert len(traces) == _TRACE_INDEX_MAX_TRACES
    # The first five injected trace_ids should have been evicted.
    evicted = {(i.to_bytes(16, "big").hex()) for i in range(5)}
    indexed = {t["trace_id"] for t in traces}
    assert evicted.isdisjoint(indexed)


@pytest.mark.asyncio
async def test_get_traces_and_spans_endpoints() -> None:
    """The two F7 HTTP routes round-trip JSON for a real ingested trace."""
    store = TelemetryStore()
    app = create_app(store)
    trace_a = b"\xab" * 16
    payload = _otlp_payload(
        [_make_span(trace_id=trace_a, span_id=b"\x01" * 8, name="rskill.execute", start_ns=1_000)]
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/traces",
            content=payload,
            headers={"Content-Type": "application/x-protobuf"},
        )
        traces = (await client.get("/api/traces")).json()
        assert traces["traces"][0]["trace_id"] == trace_a.hex()

        spans = (await client.get(f"/api/spans/{trace_a.hex()}")).json()
        assert spans["trace_id"] == trace_a.hex()
        assert spans["spans"][0]["name"] == "rskill.execute"

        miss = await client.get("/api/spans/" + "00" * 16)
        assert miss.status_code == 404
