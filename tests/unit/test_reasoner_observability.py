"""Unit tests for the ADR-0018 F4 OTel reasoner.tick instrumentation.

Real :class:`ReasonerCore` + real OTel SDK + real
:class:`InMemorySpanExporter` (the only test double is
:class:`FakeToolUseClient` at the LLM process boundary per CLAUDE.md
§1.11). The InMemorySpanExporter ships with the OTel SDK as a
zero-overhead, real exporter — it isn't a mock.
"""

from __future__ import annotations

import pytest
from openral_core import (
    EmitPromptTool,
    ExecuteRskillTool,
)
from openral_core.exceptions import ROSPlanningError
from openral_observability import reasoner_span, semconv
from openral_reasoner import ContextRenderer, PromptRecord, ReasonerCore, ToolPalette
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.fakes.fake_llm import FakeToolUseClient


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Replace the global TracerProvider with one that records to memory.

    Mirrors the canonical pattern from
    ``python/observability/tests/conftest.py``: bypass the OTel API's
    set-once guard via the private holder so each test gets a fresh
    provider + fresh in-memory exporter (the OTel SDK only allows
    :func:`trace.set_tracer_provider` to take effect once per process
    otherwise). Per CLAUDE.md §1.11 the exporter and provider are
    real SDK components — only the on-the-wire destination is swapped
    for in-memory storage.
    """
    from opentelemetry import trace

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exp
    finally:
        exp.clear()


def _renderer_with_prompt() -> ContextRenderer:
    """One-prompt renderer (so the empty-palette short-circuit doesn't fire)."""
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="x", metadata_json="", stamp_ns=0))
    return r


def _palette(*skills: str) -> ToolPalette:
    return ToolPalette(execute_rskill_ids=frozenset(skills))


def test_tick_emits_reasoner_tick_span(exporter: InMemorySpanExporter) -> None:
    """A successful tick emits exactly one reasoner.tick span."""
    client = FakeToolUseClient(
        model_id="fake-model",
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=_palette())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == semconv.SPAN_REASONER_TICK
    assert span.attributes is not None
    assert span.attributes[semconv.REASONER_MODEL] == "fake-model"
    assert span.attributes[semconv.REASONER_TICK_IDX] == 1
    assert span.attributes[semconv.REASONER_TOOL] == "emit_prompt"


def test_tick_records_skill_id_on_execute_skill(exporter: InMemorySpanExporter) -> None:
    """ExecuteRskillTool dispatch lands the rskill_id attribute on the span."""
    palette = _palette("openral/rskill-x")
    client = FakeToolUseClient(
        responses=[ExecuteRskillTool(rskill_id="openral/rskill-x")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_TOOL] == "execute_rskill"
    assert spans[0].attributes[semconv.REASONER_RSKILL_ID] == "openral/rskill-x"


def test_tick_records_suppressed_reason_palette_empty(
    exporter: InMemorySpanExporter,
) -> None:
    """An empty palette + no prompts produces a span with suppressed_reason=palette_empty."""
    client = FakeToolUseClient(responses=[])
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(world_state=None, renderer=ContextRenderer(), palette=_palette())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_SUPPRESSED_REASON] == "palette_empty"


def test_tick_records_error_kind_on_provider_failure(
    exporter: InMemorySpanExporter,
) -> None:
    """ROSPlanningError from the client lands on the span as REASONER_ERROR_KIND."""
    client = FakeToolUseClient(raise_on_call=ROSPlanningError("provider timeout"))
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=_palette())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_ERROR_KIND] == "ROSPlanningError"
    # record_exception() adds an exception event to the span.
    assert any(event.name == "exception" for event in spans[0].events)


def test_min_interval_suppression_does_not_emit_span(
    exporter: InMemorySpanExporter,
) -> None:
    """A min-interval-gated tick is suppressed before the span opens (no trace noise)."""
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text="a"),
            EmitPromptTool(target_topic="/openral/prompt", text="b"),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.1, clock=clock)
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=_palette())
    clock_value = 0.05  # within min_interval_s
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=_palette())

    spans = exporter.get_finished_spans()
    # Exactly one span — the first tick. The min-interval-suppressed
    # second tick deliberately does not open a span.
    assert len(spans) == 1


def test_force_force_attribute_recorded(exporter: InMemorySpanExporter) -> None:
    """Event-preempted ticks (force=True) record reasoner.force=True on the span."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(),
        palette=_palette(),
        force=True,
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_FORCE] is True


def test_reasoner_span_helper_no_op_without_provider() -> None:
    """The helper is safe to call before configure_observability — no-op span."""
    with reasoner_span(tick_idx=0, model="fake") as span:
        assert span is not None


def test_dashboard_store_picks_up_reasoner_tick_span() -> None:
    """The dashboard store's ``reasoner.tick`` handler populates ``_topics["reasoner"]``.

    ADR-0018 F4 — the Reasoner emits one ``reasoner.tick`` span per
    orchestrator pass via ``openral_observability.reasoner_span``. The
    dashboard's headline-family map (added alongside ADR-0025) routes
    that span name into the per-tick ``_topics["reasoner"]`` slot the
    operator-facing card reads.

    This test mirrors ``test_slam_bridge.test_dashboard_store_picks_up
    _slam_occupancy_grid_span`` — builds a single OTLP span by hand,
    feeds it through ``TelemetryStore.ingest_spans``, asserts the
    ``snapshot()["topics"]["reasoner"]`` slot carries every attribute
    the card renderer expects.
    """
    pytest.importorskip("opentelemetry.proto")
    from openral_observability.dashboard.store import TelemetryStore
    from opentelemetry.proto.common.v1.common_pb2 import (
        AnyValue,
        KeyValue,
    )
    from opentelemetry.proto.trace.v1.trace_pb2 import (
        ResourceSpans,
        ScopeSpans,
        Span,
    )

    span = Span(
        trace_id=b"0" * 16,
        span_id=b"0" * 8,
        name="reasoner.tick",
        start_time_unix_nano=2_000_000_000,
        end_time_unix_nano=2_000_500_000,
        attributes=[
            KeyValue(key="reasoner.tick.idx", value=AnyValue(int_value=7)),
            KeyValue(key="reasoner.tool", value=AnyValue(string_value="ExecuteSkill")),
            KeyValue(
                key="reasoner.rskill_id",
                value=AnyValue(string_value="OpenRAL/rskill-nav2-navigate-to-pose"),
            ),
            KeyValue(key="reasoner.model", value=AnyValue(string_value="claude-haiku-4-5")),
            KeyValue(key="reasoner.force", value=AnyValue(bool_value=False)),
        ],
    )
    scope_spans = ScopeSpans(spans=[span])
    resource_spans = ResourceSpans(scope_spans=[scope_spans])

    store = TelemetryStore()
    store.ingest_spans([resource_spans])
    snapshot = store.snapshot()

    reasoner = snapshot["topics"]["reasoner"]
    assert reasoner["tick_idx"] == 7
    assert reasoner["tool"] == "ExecuteSkill"
    assert reasoner["rskill_id"] == "OpenRAL/rskill-nav2-navigate-to-pose"
    assert reasoner["model"] == "claude-haiku-4-5"
    assert reasoner["force"] is False
    assert "ts_unix" in reasoner
