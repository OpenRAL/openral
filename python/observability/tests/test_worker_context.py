"""Cross-process trace correlation: env-var carrier round-trips parent → worker.

Exercises the R2 multiprocess bootstrap (``traceparent_env`` /
``attach_traceparent_from_env`` / ``remote_parent_from_env``). No real
subprocess is spawned: the *carrier* is the contract, so a fresh OTel
context inside the same process faithfully simulates the worker — the SDK
behaves identically whether the env dict crossed a ``fork`` or not. Per
CLAUDE.md §1.11 the OTel SDK components here are real (in-memory exporter
fixtures from ``conftest.py``), not mocks.
"""

from __future__ import annotations

from contextvars import Token
from typing import cast

from openral_observability import rskill_span
from openral_observability.propagation import (
    attach_traceparent_from_env,
    remote_parent_from_env,
    traceparent_env,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_worker_span_joins_parent_trace(memory_exporter: InMemorySpanExporter) -> None:
    """A worker span attached from the env carrier inherits the parent trace."""
    # Parent process: open a span and capture the env-var carrier.
    with rskill_span("rskill.tick", rskill_id="demo") as parent_span:
        parent_ctx = parent_span.get_span_context()
        parent_trace_id = parent_ctx.trace_id
        parent_span_id = parent_ctx.span_id
        env = traceparent_env()

    assert "OTEL_TRACEPARENT" in env

    # Worker process (simulated): no active span, attach from the env, then
    # open a span. It must land in the parent's trace and link to it.
    assert not trace.get_current_span().get_span_context().is_valid
    token = attach_traceparent_from_env(env)
    assert token is not None
    try:
        tracer = trace.get_tracer("openral")
        with tracer.start_as_current_span("dispatcher.handle") as worker_span:
            worker_ctx = worker_span.get_span_context()
            assert worker_ctx.trace_id == parent_trace_id
            # ``.parent`` lives on the SDK ``ReadableSpan``, not the abstract API.
            readable = cast("ReadableSpan", worker_span)
            assert readable.parent is not None
            assert readable.parent.span_id == parent_span_id
    finally:
        otel_context.detach(cast("Token[Context]", token))


def test_traceparent_env_empty_without_active_span() -> None:
    """No span in scope → empty carrier, so a worker starts a fresh trace."""
    assert traceparent_env() == {}


def test_attach_from_empty_env_returns_none() -> None:
    """An absent carrier is a clean no-op, not an error."""
    assert attach_traceparent_from_env({}) is None


def test_remote_parent_from_env_restores_context(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """The context manager attaches on enter and detaches on exit."""
    with rskill_span("rskill.tick", rskill_id="demo"):
        env = traceparent_env()

    # Before the block: no valid span in scope.
    assert not trace.get_current_span().get_span_context().is_valid

    with remote_parent_from_env(env) as token:
        assert token is not None
        # Inside the block the parent is the current context.
        assert trace.get_current_span().get_span_context().is_valid

    # After the block the previous (empty) context is restored.
    assert not trace.get_current_span().get_span_context().is_valid


def test_remote_parent_from_env_noop_without_carrier() -> None:
    """An empty carrier yields ``None`` and detaches nothing on exit."""
    with remote_parent_from_env({}) as token:
        assert token is None
