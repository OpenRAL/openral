"""semconv constants are consistent with what the span helpers actually emit.

This is the seam between the semantic-convention contract and the
existing span helpers. If a future refactor changes the on-wire attribute
key for e.g. ``rskill.id`` but forgets to update the constant, this test
fails and the regression is caught at PR time.
"""

from __future__ import annotations

from openral_observability import (
    inference_span,
    rskill_span,
    safety_span,
    semconv,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_rskill_span_emits_semconv_keys(memory_exporter: InMemorySpanExporter) -> None:
    with rskill_span("rskill.execute", rskill_id="demo", role="s1"):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert semconv.RSKILL_ID in span.attributes
    assert semconv.RSKILL_ROLE in span.attributes
    assert span.attributes[semconv.RSKILL_ID] == "demo"


def test_inference_span_emits_semconv_keys(memory_exporter: InMemorySpanExporter) -> None:
    with inference_span(chunk_index=3, kind="prefetch", chunk_size=50):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert semconv.INFERENCE_KIND in span.attributes
    assert semconv.INFERENCE_CHUNK_INDEX in span.attributes
    assert span.attributes[semconv.INFERENCE_CHUNK_INDEX] == 3


def test_safety_span_emits_semconv_keys(memory_exporter: InMemorySpanExporter) -> None:
    with safety_span(check_name="workspace.aabb", severity="violation"):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert semconv.SAFETY_CHECK_NAME in span.attributes
    assert semconv.SAFETY_SEVERITY in span.attributes
    assert span.attributes[semconv.SAFETY_SEVERITY] == "violation"


def test_namespace_invariants() -> None:
    """Every ``openral.*`` constant lives under its declared sub-namespace."""
    sub_namespaces = (
        "openral.run.",
        "openral.tick.",
        "openral.skill.",
        "openral.rskill.",
        "openral.hal.",
        "openral.sensors.",
        "openral.world_state.",
        "openral.dataset.",
        "openral.event.",
        "openral.inference.",
        "openral.metric.",
        "openral.safety.",
        "openral.observability.",
        "openral.sim.",
        "openral.system.",
    )
    openral_keys = {
        getattr(semconv, name)
        for name in dir(semconv)
        if isinstance(getattr(semconv, name), str) and getattr(semconv, name).startswith("openral.")
    }
    # Every openral.* key matches one of the declared sub-namespaces.
    for key in openral_keys:
        assert any(key.startswith(prefix) for prefix in sub_namespaces), (
            f"semconv constant {key!r} does not match any declared sub-namespace"
        )
