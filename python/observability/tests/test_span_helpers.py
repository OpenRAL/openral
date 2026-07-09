"""rskill_span / inference_span / safety_span emit spans with expected attrs."""

from __future__ import annotations

from openral_observability import (
    inference_span,
    rskill_span,
    safety_span,
    traced,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def test_skill_span_records_skill_id(memory_exporter: InMemorySpanExporter) -> None:
    with rskill_span("rskill.configure", rskill_id="hello", role="s1"):
        pass
    spans = memory_exporter.get_finished_spans()
    assert _names(memory_exporter) == ["rskill.configure"]
    assert spans[0].attributes is not None
    assert spans[0].attributes["rskill.id"] == "hello"
    assert spans[0].attributes["rskill.role"] == "s1"


def test_inference_span_records_chunk_index(memory_exporter: InMemorySpanExporter) -> None:
    with inference_span(chunk_index=7, kind="prefetch", chunk_size=50):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "rskill.chunk_inference"
    assert span.attributes is not None
    assert span.attributes["inference.chunk_index"] == 7
    assert span.attributes["inference.kind"] == "prefetch"
    assert span.attributes["inference.chunk_size"] == 50


def test_safety_span_records_severity(memory_exporter: InMemorySpanExporter) -> None:
    with safety_span(check_name="workspace.aabb", severity="violation"):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.name == "safety.check"
    assert span.attributes is not None
    assert span.attributes["safety.check_name"] == "workspace.aabb"
    assert span.attributes["safety.severity"] == "violation"


def test_nested_spans_form_parent_child(memory_exporter: InMemorySpanExporter) -> None:
    with rskill_span("rskill.execute", rskill_id="hello"), inference_span(chunk_index=1):
        pass
    spans = memory_exporter.get_finished_spans()
    # The child finishes first.
    child, parent = spans[0], spans[1]
    assert child.name == "rskill.chunk_inference"
    assert parent.name == "rskill.execute"
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def test_traced_decorator(memory_exporter: InMemorySpanExporter) -> None:
    @traced("compute")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert _names(memory_exporter) == ["compute"]
