"""structlog → OTel log bridge: trace_id/span_id appear on log events."""

from __future__ import annotations

import logging

import pytest
import structlog
from openral_observability.logging import resolve_log_level, trace_context_processor
from openral_observability.tracing import rskill_span
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _capturing_log() -> tuple[structlog.BoundLogger, list[dict[str, object]]]:
    captured: list[dict[str, object]] = []

    def capture(_logger: object, _method: str, event: dict[str, object]) -> dict[str, object]:
        captured.append(event)
        # Stop the chain — do not pass to a real logger backend.
        raise structlog.DropEvent

    structlog.configure(
        processors=[trace_context_processor, capture],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return structlog.get_logger("test"), captured


def test_processor_stamps_trace_ids_inside_span(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Inside an active span, the processor adds trace_id and span_id."""
    log, captured = _capturing_log()

    with rskill_span("rskill.configure", rskill_id="hello"):
        log.info("hello-event")

    assert len(captured) == 1
    assert "trace_id" in captured[0]
    assert "span_id" in captured[0]
    assert isinstance(captured[0]["trace_id"], str)
    assert len(captured[0]["trace_id"]) == 32


def test_processor_no_op_outside_span() -> None:
    """No active span → no trace_id / span_id keys."""
    log, captured = _capturing_log()
    log.info("no-span")

    assert "trace_id" not in captured[0]
    assert "span_id" not in captured[0]


# ── OPENRAL_LOG_LEVEL floor ───────────────────────────────────────────────────


def test_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset → INFO.

    The bridge used to hard-pin DEBUG, so every per-tick debug record in the
    deploy graph was JSON-rendered and shipped as an OTLP log record.
    """
    monkeypatch.delenv("OPENRAL_LOG_LEVEL", raising=False)
    assert resolve_log_level() == logging.INFO


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("  WARNING  ", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("10", logging.DEBUG),
    ],
)
def test_log_level_parses_names_and_integers(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    """Level names are case-insensitive and whitespace-tolerant; integers pass through."""
    monkeypatch.setenv("OPENRAL_LOG_LEVEL", raw)
    assert resolve_log_level() == expected


@pytest.mark.parametrize("raw", ["", "LOUD", "not-a-level"])
def test_unparseable_log_level_falls_back_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A typo must not take down a deploy at bring-up."""
    monkeypatch.setenv("OPENRAL_LOG_LEVEL", raw)
    assert resolve_log_level() == logging.INFO
