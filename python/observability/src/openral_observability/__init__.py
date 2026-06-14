"""openral observability — OpenTelemetry tracing + metrics + structlog log bridge.

Public API:
    configure_observability — idempotent SDK setup (no-op when no endpoint).
    shutdown_observability  — drain and shut down providers.
    rskill_span             — span for Skill lifecycle (configure / activate / execute).
    inference_span          — span for one VLA chunk inference (foreground or prefetch).
    reasoner_span           — span for one ReasonerCore.tick (ADR-0018 F4).
    safety_span             — span for one safety check.
    cli_command_span        — root span for one ``openral`` CLI invocation.
    traced                  — decorator equivalent of the above context managers.
    semconv                 — single source of truth for attribute / span / metric names.
    metrics                 — pre-registered OTel meter instruments.
    propagation             — W3C ``traceparent`` inject / extract helpers.
    dashboard               — live debugging dashboard (FastAPI + SSE + OTLP/HTTP receiver).

Example:
    >>> from openral_observability import configure_observability, rskill_span
    >>> configure_observability(service_name="ral")
    >>> with rskill_span("rskill.configure", rskill_id="hello"):
    ...     pass
"""

from __future__ import annotations

from openral_observability import (
    failure_bus,
    metrics,
    producer,
    propagation,
    semconv,
    system_metrics,
)
from openral_observability._sdk import (
    configure_observability,
    shutdown_observability,
)
from openral_observability.cli import cli_command_span
from openral_observability.diagnostics import DiagnosticsHeartbeat, Level
from openral_observability.failure_bus import (
    FailureBusPublisher,
    FailureSource,
)
from openral_observability.lifecycle import log_lifecycle_errors
from openral_observability.tracing import (
    inference_span,
    reasoner_span,
    rskill_span,
    safety_span,
    traced,
)

__all__ = [
    "DiagnosticsHeartbeat",
    "FailureBusPublisher",
    "FailureSource",
    "Level",
    "cli_command_span",
    "configure_observability",
    "failure_bus",
    "inference_span",
    "log_lifecycle_errors",
    "metrics",
    "producer",
    "propagation",
    "reasoner_span",
    "rskill_span",
    "safety_span",
    "semconv",
    "shutdown_observability",
    "system_metrics",
    "traced",
]
