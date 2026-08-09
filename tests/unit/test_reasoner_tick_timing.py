"""Unit tests for the tick timing split (issue #92).

``ReasonerTickResult.elapsed_s`` is end-to-end tick wall-clock, so a tick that
drifts from 6 s to 99 s over a mission cannot be blamed on the provider or on
the reasoner. These tests pin the two attributions the trace now carries:
``reasoner.llm_s`` (provider round-trip alone) and ``reasoner.prompt_tokens``.

Real :class:`ReasonerCore` + real OTel SDK + real
:class:`InMemorySpanExporter`; the only double is
:class:`FakeToolUseClient` at the LLM process boundary (CLAUDE.md §1.11).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_core import EmitPromptTool
from openral_core.exceptions import ROSPlanningError
from openral_observability import semconv
from openral_reasoner import (
    ContextRenderer,
    PromptRecord,
    ReasonerCore,
    ReasonerTickResult,
    ToolPalette,
)
from openral_reasoner.tool_use import _prompt_tokens
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.fakes.fake_llm import FakeToolUseClient

LLM_DELAY_S = 0.05
"""Modelled provider round-trip. Long enough to dominate the sub-millisecond
render + bookkeeping either side of it, short enough to keep the tier <30 s."""


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    """Fresh in-memory-exporting TracerProvider (see test_reasoner_observability)."""
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


def _renderer() -> ContextRenderer:
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="pick up the baguette", metadata_json="", stamp_ns=0))
    return r


def _tick(core: ReasonerCore) -> ReasonerTickResult:
    return core.tick(
        world_state=None, renderer=_renderer(), palette=ToolPalette(execute_rskill_ids=frozenset())
    )


def test_llm_duration_is_recorded_and_bounded_by_tick_elapsed(
    exporter: InMemorySpanExporter,
) -> None:
    """``reasoner.llm_s`` measures the call alone, and fits inside ``elapsed_s``."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
        delay_s=LLM_DELAY_S,
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)

    result = _tick(core)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    llm_s = span.attributes[semconv.REASONER_LLM_S]
    assert isinstance(llm_s, float)
    # The fake sleeps for the whole call, so llm_s must cover the sleep …
    assert llm_s >= LLM_DELAY_S
    # … and the tick as a whole must cover the call.
    assert result.elapsed_s >= llm_s


def test_llm_duration_recorded_when_the_call_fails(exporter: InMemorySpanExporter) -> None:
    """A slow *failing* call is the one worth attributing — time it too."""
    client = FakeToolUseClient(
        raise_on_call=ROSPlanningError("Request timed out"),
        delay_s=LLM_DELAY_S,
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)

    result = _tick(core)

    assert result.error is not None
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[semconv.REASONER_LLM_S] >= LLM_DELAY_S


def test_prompt_tokens_are_recorded_when_the_client_reports_usage(
    exporter: InMemorySpanExporter,
) -> None:
    """A client exposing ``last_prompt_tokens`` lands on the span."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    client.last_prompt_tokens = 4321  # type: ignore[attr-defined]  # reason: optional client attr
    core = ReasonerCore(client=client, min_interval_s=0.0)

    _tick(core)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[semconv.REASONER_PROMPT_TOKENS] == 4321


def test_prompt_tokens_absent_when_the_client_reports_nothing(
    exporter: InMemorySpanExporter,
) -> None:
    """No usage reported → no attribute, rather than a misleading zero."""
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)

    _tick(core)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert semconv.REASONER_PROMPT_TOKENS not in span.attributes


class _Usage:
    """Provider usage payload — the shape the SDKs decode off the wire."""

    def __init__(self, **fields: int) -> None:
        self.__dict__.update(fields)


class _Response:
    def __init__(self, usage: _Usage | None) -> None:
        self.usage = usage


def test_prompt_tokens_reads_openai_and_anthropic_usage_shapes() -> None:
    """OpenAI reports one field; Anthropic splits the same total across three."""
    assert _prompt_tokens(_Response(_Usage(prompt_tokens=1200, completion_tokens=40))) == 1200
    # Anthropic: cached prefix + fresh context are all prompt the model ingested.
    anthropic = _Usage(input_tokens=300, cache_read_input_tokens=900, cache_creation_input_tokens=0)
    assert _prompt_tokens(_Response(anthropic)) == 1200
    assert _prompt_tokens(_Response(None)) is None
    assert _prompt_tokens(_Response(_Usage())) is None
