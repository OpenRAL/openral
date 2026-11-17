"""A ``wait`` tick says why it waits, end to end: ``ReasonerCore`` span → dashboard store.

The console's reasoner trace rendered every wait tick as "WAIT / wait". The tick span
now carries the model's rationale and the in-flight skill, and the store folds them,
with the tier and LLM latency, into ``topics.reasoner.detail``.

Real ``ReasonerCore`` + real OTel SDK span, encoded to OTLP and ingested by the real
``TelemetryStore``; the only double is ``FakeToolUseClient`` at the LLM boundary
(CLAUDE.md §1.11).
"""

from __future__ import annotations

import itertools
import re

import pytest
from openral_core import EmitPromptTool, WaitTool
from openral_observability import semconv
from openral_observability.dashboard import TelemetryStore
from openral_reasoner import ContextRenderer, PromptRecord, ReasonerCore, ToolPalette
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.fakes.fake_llm import FakeToolUseClient

# Seen live (station stocking run): the heartbeat wait rationale while a skill runs.
RATIONALE = (
    "The stocking skill is running and was dispatched about 4 seconds ago. "
    "There are no failures, so I'm letting it continue."
)
SKILL = "OpenRAL/rskill-act-libero"


def test_wait_tick_carries_rationale_and_inflight_skill_to_the_store(
    exporter: InMemorySpanExporter,
) -> None:
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="stock aisle 3", metadata_json="", stamp_ns=0))
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
    assert reasoner["tier"] == "heartbeat"
    assert "rationale" not in reasoner  # once, inside `detail`
    # e.g. "rskill-act-libero running · heartbeat · 0.1s — The stocking skill is running …"
    head, rationale = reasoner["detail"].split(" — ", 1)
    assert re.fullmatch(r"rskill-act-libero running · heartbeat · (\d+\.\ds|\d+ms)", head)
    assert rationale == RATIONALE
    # The event log row keeps the why too (the topic slot is latest-wins).
    (event,) = [e for e in store.snapshot()["events"] if e["kind"] == "reasoner.tick"]
    assert event["title"].endswith(reasoner["detail"])


def test_rationale_is_capped_on_the_span_and_in_the_detail(
    exporter: InMemorySpanExporter,
) -> None:
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="stock aisle 3", metadata_json="", stamp_ns=0))
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
    detail = store.snapshot()["topics"]["reasoner"]["detail"]
    assert detail.endswith(" — " + "x" * semconv.REASONER_RATIONALE_MAX)


def _ingest(span: object) -> dict[str, object]:
    store = TelemetryStore()
    store.ingest_spans(list(encode_spans([span]).resource_spans))  # type: ignore[list-item]  # reason: a finished SDK span
    reasoner: dict[str, object] = store.snapshot()["topics"]["reasoner"]
    return reasoner


def test_detail_says_dispatching_while_the_policy_cold_loads(
    exporter: InMemorySpanExporter,
) -> None:
    """A goal still loading its weights must not read "<skill> running"."""
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="stock aisle 3", metadata_json="", stamp_ns=0))
    renderer.set_inflight_skill(SKILL + "/", stamp_ns=1, state="dispatching")
    # A stepping clock (+20 ms per read), not a sleep: a loaded host cannot push the
    # LLM call past the ms/s boundary the assertion below pins.
    ticks = itertools.count()
    core = ReasonerCore(
        client=FakeToolUseClient(responses=[WaitTool(rationale=RATIONALE)]),
        min_interval_s=0.0,
        clock=lambda: next(ticks) * 0.02,
    )

    core.tick(
        world_state=None,
        renderer=renderer,
        palette=ToolPalette(execute_rskill_ids=frozenset({SKILL + "/"})),
        tier="D",
    )

    (span,) = exporter.get_finished_spans()
    detail = str(_ingest(span)["detail"])
    # The trailing "/" still names the skill; the tier is a word, and a sub-0.1 s
    # LLM call reads in ms, never "0.0s".
    assert re.match(r"rskill-act-libero dispatching · operator event · \d+ms — ", detail)


def test_detail_marks_a_retry_capped_tick_as_suppressed(exporter: InMemorySpanExporter) -> None:
    """A retry-capped call never ran: its rationale must not read as the call that did."""
    renderer = ContextRenderer()
    core = ReasonerCore(
        client=FakeToolUseClient(
            responses=[
                EmitPromptTool(target_topic="/openral/prompt", text="same", rationale=RATIONALE)
                for _ in range(2)
            ]
        ),
        min_interval_s=0.0,
        retry_cap_per_kind=1,
    )
    for i in range(2):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        core.tick(world_state=None, renderer=renderer, palette=ToolPalette())

    capped = exporter.get_finished_spans()[-1]
    assert "suppressed: retry_cap — " in str(_ingest(capped)["detail"])


def test_rationale_stays_on_host_without_consent(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free model text goes to a public collector only with the deployer's consent (§3)."""
    from openral_observability import _sdk

    monkeypatch.setattr(_sdk, "_endpoint", "https://otlp.example.com:4318")
    monkeypatch.delenv("OPENRAL_TELEMETRY_TEXT_CONSENT", raising=False)
    renderer = ContextRenderer()
    renderer.append_prompt(PromptRecord(text="stock aisle 3", metadata_json="", stamp_ns=0))
    core = ReasonerCore(
        client=FakeToolUseClient(responses=[WaitTool(rationale=RATIONALE)] * 2),
        min_interval_s=0.0,
    )
    core.tick(world_state=None, renderer=renderer, palette=ToolPalette())
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert semconv.REASONER_RATIONALE not in span.attributes

    monkeypatch.setenv("OPENRAL_TELEMETRY_TEXT_CONSENT", "1")
    renderer.append_prompt(PromptRecord(text="stock aisle 4", metadata_json="", stamp_ns=1))
    core.tick(world_state=None, renderer=renderer, palette=ToolPalette())
    consented = exporter.get_finished_spans()[-1]
    assert consented.attributes is not None
    assert consented.attributes[semconv.REASONER_RATIONALE] == RATIONALE
