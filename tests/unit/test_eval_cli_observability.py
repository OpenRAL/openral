"""``openral`` CLI top-level owns OTel; every invocation runs under ``cli.command``.

Ownership lives in ``openral_cli.main:_root`` so sim/benchmark/deploy/connect
subcommands all run inside one root ``cli.command`` span. The sim leaf
(``openral_sim.cli``) used to shut providers down itself, draining the
BatchSpanProcessor before the root span's ``__exit__`` ran and silently
dropping the export.

Tests use the real OTel SDK with an in-memory exporter (CLAUDE.md §1.11);
no mocks of ``configure_observability``.
"""

from __future__ import annotations

from openral_cli.main import app
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typer.testing import CliRunner


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
    """``openral doctor`` (no mode mapping) still gets a ``cli.command`` span.

    ``main._RUN_MODE_BY_SUBCOMMAND`` covers sim/benchmark/deploy/connect only;
    every other subcommand's span carries no ``openral.run.mode`` attribute.
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
