"""A ``wait`` tick says why it waits, end to end: ``ReasonerCore`` span → dashboard store.

The console's reasoner trace rendered every wait tick as "WAIT / wait". The tick span
now carries the model's rationale and the in-flight skill, and the store folds them,
with the tier and LLM latency, into ``topics.reasoner.detail``.

Real ``ReasonerCore`` + real OTel SDK span, encoded to OTLP and ingested by the real
``TelemetryStore``; the only double is ``FakeToolUseClient`` at the LLM boundary
(CLAUDE.md §1.11).
"""

from __future__ import annotations

import re

from openral_core import WaitTool
from openral_observability import semconv
from openral_observability.dashboard import TelemetryStore
from openral_reasoner import ContextRenderer, PromptRecord, ReasonerCore, ToolPalette
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.fakes.fake_llm import FakeToolUseClient

# Seen live (station restock run): the heartbeat wait rationale while a skill runs.
RATIONALE = (
    "The restock skill is running and was dispatched about 4 seconds ago. "
    "There are no failures, so I'm letting it continue."
)
SKILL = "OpenRAL/rskill-act-libero"


def test_wait_tick_carries_rationale_and_inflight_skill_to_the_store(
    exporter: InMemorySpanExporter,
) -> None:
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="restock aisle 3", metadata_json="", stamp_ns=0))
    renderer.set_inflight_skill(SKILL, stamp_ns=1, state="running")
    core = ReasonerCore(
        client=FakeToolUseClient(responses=[WaitTool(rationale=RATIONALE)], delay_s=0.06),
        min_interval_s=0.0,
    )

    core.tick(
        world_state=None,
        renderer=renderer,
        palette=ToolPalette(execute_rskill_ids=frozenset({SKILL})),
        tier="heartbeat",
    )

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[semconv.REASONER_RATIONALE] == RATIONALE
    assert span.attributes[semconv.REASONER_INFLIGHT_SKILL] == SKILL

    store = TelemetryStore()
    store.ingest_spans(list(encode_spans([span]).resource_spans))
    reasoner = store.snapshot()["topics"]["reasoner"]
    assert reasoner["tool"] == "wait"
    assert reasoner["rationale"] == RATIONALE
    assert reasoner["tier"] == "heartbeat"
    # e.g. "rskill-act-libero running · heartbeat · 0.1s — The restock skill is running …"
    head, rationale = reasoner["detail"].split(" — ", 1)
    assert re.fullmatch(r"rskill-act-libero running · heartbeat · \d+\.\ds", head)
    assert rationale == RATIONALE


def test_rationale_is_capped_on_the_span_and_in_the_detail(
    exporter: InMemorySpanExporter,
) -> None:
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="restock aisle 3", metadata_json="", stamp_ns=0))
    core = ReasonerCore(
        client=FakeToolUseClient(responses=[WaitTool(rationale="x" * 2000)]), min_interval_s=0.0
    )

    core.tick(world_state=None, renderer=renderer, palette=ToolPalette())

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert len(str(span.attributes[semconv.REASONER_RATIONALE])) == semconv.REASONER_RATIONALE_MAX
    assert semconv.REASONER_INFLIGHT_SKILL not in span.attributes
    store = TelemetryStore()
    store.ingest_spans(list(encode_spans([span]).resource_spans))
    # The trace row keeps 120 rationale chars; the full (capped) text stays on `rationale`.
    detail = store.snapshot()["topics"]["reasoner"]["detail"]
    assert detail.endswith(" — " + "x" * 119 + "…")
