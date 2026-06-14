"""Unit tests for ADR-0018 2026-05-25 amendment: trigger-tier observability.

The full Tier-A/B/C/D preemption flow (callbacks driving the LLM)
lives in ``tests/integration/test_reasoner_node_end_to_end.py``
because it needs a real rclpy executor. This module covers the
transport-agnostic surface — the ``tier`` argument on
:meth:`ReasonerCore.tick` and its landing on the OTel span.
"""

from __future__ import annotations

import pytest
from openral_core import EmitPromptTool
from openral_observability import semconv
from openral_reasoner import ContextRenderer, PromptRecord, ReasonerCore, ToolPalette
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.fakes.fake_llm import FakeToolUseClient


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Real OTel SDK + in-memory exporter; per-test reset of the global provider."""
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
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="x", metadata_json="", stamp_ns=0))
    return r


@pytest.mark.parametrize("tier", ["A", "B", "C", "D", "heartbeat"])
def test_tier_attribute_lands_on_span(tier: str, exporter: InMemorySpanExporter) -> None:
    """The ``tier`` argument is recorded verbatim on ``reasoner.tier``."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(),
        palette=ToolPalette(execute_rskill_ids=frozenset()),
        tier=tier,
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_TIER] == tier


def test_tier_defaults_to_heartbeat(exporter: InMemorySpanExporter) -> None:
    """Omitting the ``tier`` arg records the heartbeat fallback on the span."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(),
        palette=ToolPalette(execute_rskill_ids=frozenset()),
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[semconv.REASONER_TIER] == "heartbeat"
