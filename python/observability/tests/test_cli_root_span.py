"""``cli_command_span`` opens a root span with openral.run.* attributes."""

from __future__ import annotations

from openral_observability import cli_command_span, semconv
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_cli_command_span_is_root(memory_exporter: InMemorySpanExporter) -> None:
    with cli_command_span("sim run", mode=semconv.RUN_MODE_SIM):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.name == semconv.SPAN_CLI_COMMAND
    assert span.parent is None
    assert span.attributes is not None
    assert span.attributes["cli.subcommand"] == "sim run"
    assert span.attributes[semconv.RUN_MODE] == "sim"
    # A run id is always populated, even when the caller doesn't pass one.
    assert semconv.RUN_ID in span.attributes
    assert isinstance(span.attributes[semconv.RUN_ID], str)


def test_cli_command_span_caller_run_id(memory_exporter: InMemorySpanExporter) -> None:
    with cli_command_span("benchmark run", mode=semconv.RUN_MODE_BENCHMARK, run_id="custom-id"):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert span.attributes[semconv.RUN_ID] == "custom-id"
