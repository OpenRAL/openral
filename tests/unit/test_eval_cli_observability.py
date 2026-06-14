"""``openral`` CLI top-level owns OTel; every invocation runs under ``cli.command``.

Ownership moved from the sim leaf (``openral_sim.cli``) to the top-level
``openral_cli.main:_root`` callback so the entire invocation — including
the sim / benchmark / connect subcommands — runs inside a single
``cli.command`` root span. Shutting down the providers inside the sim
leaf used to drain the BatchSpanProcessor before the root span's
``__exit__`` ran, dropping the export silently; that bug is now
prevented by structure.

Tests use the real OTel SDK with an in-memory exporter (CLAUDE.md
§1.11 / §5.4). No mocks of ``configure_observability`` —
the production code path is exercised end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_cli.main import app
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typer.testing import CliRunner


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Swap the global TracerProvider for one backed by an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


def test_ral_sim_list_runs_under_cli_command_span(memory_exporter: InMemorySpanExporter) -> None:
    """``openral sim list`` emits a single root ``cli.command`` span covering the call."""
    cli = CliRunner()
    result = cli.invoke(app, ["sim", "list"])
    assert result.exit_code == 0, result.output

    spans = memory_exporter.get_finished_spans()
    root_spans = [s for s in spans if s.name == "cli.command" and s.parent is None]
    assert len(root_spans) == 1, (
        f"expected one cli.command root span, got: {[s.name for s in spans]}"
    )
    root = root_spans[0]
    assert root.attributes is not None
    assert root.attributes["cli.subcommand"] == "sim"
    assert root.attributes["openral.run.mode"] == "sim"
    assert "openral.run.id" in root.attributes


def test_ral_doctor_runs_under_cli_command_span(memory_exporter: InMemorySpanExporter) -> None:
    """``openral doctor`` (a top-level command, no mode mapping) still gets wrapped.

    The mode mapping in ``main._RUN_MODE_BY_SUBCOMMAND`` covers
    ``sim`` / ``benchmark`` / ``deploy`` / ``connect`` — every other
    subcommand emits a ``cli.command`` span with no ``openral.run.mode``
    attribute. Asserting that explicitly catches regressions where a
    future contributor accidentally adds an unconditional mode.
    """
    cli = CliRunner()
    result = cli.invoke(app, ["doctor"])
    # `openral doctor` may print a non-ok row on the test runner (no ROS, no
    # GPU); we only care that it returned and emitted the root span.
    assert result.exit_code in (0, 1)

    spans = memory_exporter.get_finished_spans()
    root_spans = [s for s in spans if s.name == "cli.command" and s.parent is None]
    assert len(root_spans) == 1
    root = root_spans[0]
    assert root.attributes is not None
    assert root.attributes["cli.subcommand"] == "doctor"
    # No mode set for `doctor` — host introspection is neither sim nor hardware.
    assert "openral.run.mode" not in root.attributes
