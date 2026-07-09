"""W3C traceparent inject/extract round-trips and links child to parent."""

from __future__ import annotations

from openral_observability import rskill_span
from openral_observability.propagation import (
    current_traceparent,
    extract_traceparent,
    inject_traceparent,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_current_traceparent_inside_span(memory_exporter: InMemorySpanExporter) -> None:
    with rskill_span("rskill.execute", rskill_id="demo"):
        traceparent = current_traceparent()
    assert traceparent is not None
    # ``00-<32hex>-<16hex>-<2hex>``
    parts = traceparent.split("-")
    assert len(parts) == 4
    assert parts[0] == "00"
    assert len(parts[1]) == 32
    assert len(parts[2]) == 16
    assert len(parts[3]) == 2


def test_current_traceparent_without_span() -> None:
    assert current_traceparent() is None


def test_inject_then_extract_links_child_to_parent(memory_exporter: InMemorySpanExporter) -> None:
    # Producer side: open a tick span, inject the traceparent.
    with rskill_span("rskill.tick", rskill_id="demo") as parent_span:
        producer_trace_id = parent_span.get_span_context().trace_id
        producer_span_id = parent_span.get_span_context().span_id
        headers = inject_traceparent()

    assert "traceparent" in headers

    # Consumer side: extract and open a child span.
    ctx = extract_traceparent(headers["traceparent"])
    token = otel_context.attach(ctx)
    try:
        tracer = trace.get_tracer("openral")
        with tracer.start_as_current_span("safety.check") as child_span:
            child_ctx = child_span.get_span_context()
            # Child inherits the parent's trace id.
            assert child_ctx.trace_id == producer_trace_id
            # Child's parent span id matches the producer's span id.
            assert child_span.parent is not None
            assert child_span.parent.span_id == producer_span_id
    finally:
        otel_context.detach(token)


def test_extract_traceparent_malformed_returns_empty_context() -> None:
    """Garbage in → no usable context out; downstream span starts a fresh trace."""
    ctx = extract_traceparent("not-a-valid-traceparent")
    # The propagator returns an empty Context for malformed input.
    span = trace.get_current_span(ctx)
    assert not span.get_span_context().is_valid
