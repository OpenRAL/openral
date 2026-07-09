"""structlog → OTel log bridge: trace_id/span_id appear on log events."""

from __future__ import annotations

import structlog
from openral_observability.logging import trace_context_processor
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
