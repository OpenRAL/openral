"""structlog ↔ OpenTelemetry log bridge.

Wires structlog so that:

1. Every log record carries the active span's ``trace_id`` / ``span_id`` so
   logs and traces correlate in Jaeger.
2. The final record is forwarded to a stdlib logger that has been attached
   to an OTel ``LoggerProvider`` via ``LoggingHandler`` — i.e. logs ship as
   OTLP log records to the same collector as the spans.
3. Records below the ``OPENRAL_LOG_LEVEL`` floor (default ``INFO``, see
   :func:`resolve_log_level`) are dropped by the stdlib level check before
   they are rendered or exported.

The bridge itself is global and idempotent (see
:func:`install_structlog_bridge`), so it works unchanged inside a spawned
worker once that worker has run :func:`configure_observability` (or the
convenience :func:`configure_worker_observability`). Multiprocess workers
(the dispatcher, the future fleet supervisor) correlate their logs and
spans to the parent trace by having the parent pass
:func:`openral_observability.propagation.traceparent_env` into the child's
environment and the worker attach it via
:func:`configure_worker_observability` /
:func:`openral_observability.propagation.attach_traceparent_from_env`; the
``trace_context_processor`` then stamps the parent's ``trace_id`` /
``span_id`` on every worker log line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

__all__ = ["install_structlog_bridge", "resolve_log_level", "trace_context_processor"]

_BRIDGE_LOGGER_NAME = "openral.otel_bridge"
_INSTALLED = False

_LOG_LEVEL_ENV = "OPENRAL_LOG_LEVEL"
# INFO, not DEBUG. Every record that clears this floor is JSON-rendered and
# shipped to the collector as an OTLP log record; below it, the stdlib level
# check short-circuits before either happens. The deploy graph has ~73 DEBUG
# call sites and several fire per control tick — `world_state.*.updated` (7
# sites in the aggregator), `skill.step`, `safety.null_check` — so at 30 Hz a
# DEBUG floor pays serialisation plus export for a few hundred records a
# second, on the same GIL the camera readers and the VLA weight load are
# fighting over. That contention is not hypothetical here: it is the same
# class of stall that stretched a 23 s import to 8+ minutes on the SO-101
# bench.
#
# Set OPENRAL_LOG_LEVEL=DEBUG to get them back. Note this floor governs LOG
# RECORDS only — dashboard span rows are banded separately in
# `dashboard.store._is_headline_span`, so the Event Log's DEBUG chip still
# shows the per-tick span stream at the default floor.
_DEFAULT_LEVEL = logging.INFO


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


def resolve_log_level() -> int:
    """Resolve the OpenRAL log floor from ``OPENRAL_LOG_LEVEL``.

    Accepts a level name (``DEBUG`` / ``info`` / ``WARNING`` / …) or an
    integer. Anything unparseable falls back to :data:`_DEFAULT_LEVEL`
    rather than raising — a typo in an env var must not take down a
    deploy, and a too-quiet logger is easier to notice than a crash at
    bring-up.

    Example:
        >>> import logging, os
        >>> os.environ["OPENRAL_LOG_LEVEL"] = "DEBUG"
        >>> resolve_log_level() == logging.DEBUG
        True
        >>> del os.environ["OPENRAL_LOG_LEVEL"]
        >>> resolve_log_level() == logging.INFO
        True
    """
    raw = os.environ.get(_LOG_LEVEL_ENV)
    if raw is None:
        return _DEFAULT_LEVEL
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    resolved = logging.getLevelNamesMapping().get(raw.upper())
    return resolved if resolved is not None else _DEFAULT_LEVEL


def install_structlog_bridge(logger_provider: LoggerProvider) -> None:
    """Attach an OTel ``LoggingHandler`` to a stdlib logger.

    Routes structlog through that logger so events ship as OTLP log
    records.  Idempotent.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    level = resolve_log_level()

    bridge_logger = logging.getLogger(_BRIDGE_LOGGER_NAME)
    bridge_logger.setLevel(level)
    # Avoid double-emission via the root logger.
    bridge_logger.propagate = False
    bridge_logger.addHandler(LoggingHandler(level=level, logger_provider=logger_provider))

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
    root_bot.setLevel(level)
    if not any(isinstance(h, LoggingHandler) for h in root_bot.handlers):
        root_bot.addHandler(LoggingHandler(level=level, logger_provider=logger_provider))

    _INSTALLED = True
