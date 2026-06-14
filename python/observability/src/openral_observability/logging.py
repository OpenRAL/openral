"""structlog ↔ OpenTelemetry log bridge.

Wires structlog so that:

1. Every log record carries the active span's ``trace_id`` / ``span_id`` so
   logs and traces correlate in Jaeger.
2. The final record is forwarded to a stdlib logger that has been attached
   to an OTel ``LoggerProvider`` via ``LoggingHandler`` — i.e. logs ship as
   OTLP log records to the same collector as the spans.

Single-process only.  Multiprocess workers (dispatcher, future fleet
supervisor) need additional setup — out of scope for Day 22.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

__all__ = ["install_structlog_bridge", "trace_context_processor"]

_BRIDGE_LOGGER_NAME = "openral.otel_bridge"
_INSTALLED = False


def trace_context_processor(
    _logger: object, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Stamp the active OTel ``trace_id`` and ``span_id`` on a log event.

    structlog processor.  Used both for terminal output and for the OTel
    bridge so even the rendered text shows the trace correlation.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def install_structlog_bridge(logger_provider: LoggerProvider) -> None:
    """Attach an OTel ``LoggingHandler`` to a stdlib logger.

    Routes structlog through that logger so events ship as OTLP log
    records.  Idempotent.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    bridge_logger = logging.getLogger(_BRIDGE_LOGGER_NAME)
    bridge_logger.setLevel(logging.DEBUG)
    # Avoid double-emission via the root logger.
    bridge_logger.propagate = False
    bridge_logger.addHandler(LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            trace_context_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # Route structlog records through our bridge logger, which has the
    # OTel handler attached.  We use a ProcessorFormatter so the final
    # message is a JSON line, which the OTel handler captures verbatim.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    for h in bridge_logger.handlers:
        h.setFormatter(formatter)

    # Make `structlog.get_logger(name)` resolve to the bridge logger by
    # routing the root stdlib logger's `openral*` namespace there too.
    root_bot = logging.getLogger("openral")
    root_bot.setLevel(logging.DEBUG)
    if not any(isinstance(h, LoggingHandler) for h in root_bot.handlers):
        root_bot.addHandler(LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider))

    _INSTALLED = True
