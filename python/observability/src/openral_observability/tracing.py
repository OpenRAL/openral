"""Span helpers for the openral layers.

All helpers are safe to call before :func:`configure_observability` runs —
in that case they create spans on the default no-op ``TracerProvider`` and
emit nothing.

The helpers are deliberately thin (``contextmanager`` over the standard
``Tracer.start_as_current_span``) so they cost <1 µs in the no-op path.

Attribute keys come from :mod:`openral_observability.semconv` — never
hardcode a string at a call site. See CLAUDE.md §1.13 (no duplication)
and design §3 (semantic-convention namespace).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span

from openral_observability import semconv

__all__ = ["inference_span", "reasoner_span", "rskill_span", "safety_span", "traced"]

_TRACER_NAME = "openral"

P = ParamSpec("P")
R = TypeVar("R")


def _tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def rskill_span(
    name: str,
    *,
    rskill_id: str | None = None,
    role: str | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Span around a Skill lifecycle method (configure/activate/execute).

    Args:
        name: Span name (e.g. ``"rskill.configure"``).
        rskill_id: rSkill identifier; recorded as ``rskill.id``.
        role: rSkill role (``s0``/``s1``/``s2``); recorded as ``rskill.role``.
        **attrs: Extra attributes recorded with a ``skill.`` prefix.

    Yields:
        The active :class:`opentelemetry.trace.Span`.
    """
    tagged: dict[str, Any] = {}
    if rskill_id is not None:
        tagged[semconv.RSKILL_ID] = rskill_id
    if role is not None:
        tagged[semconv.RSKILL_ROLE] = role
    for k, v in attrs.items():
        tagged[f"skill.{k}"] = v
    with _tracer().start_as_current_span(name, attributes=tagged) as span:
        yield span


@contextmanager
def inference_span(
    name: str = semconv.SPAN_RSKILL_CHUNK_INFERENCE,
    *,
    chunk_index: int | None = None,
    kind: str = "foreground",
    **attrs: Any,
) -> Iterator[Span]:
    """Span around one VLA chunk inference.

    Args:
        name: Span name.
        chunk_index: Sequence number of the chunk being computed.
        kind: ``"foreground"`` or ``"prefetch"``.
        **attrs: Extra attributes recorded with an ``inference.`` prefix.

    Yields:
        The active span.
    """
    tagged: dict[str, Any] = {semconv.INFERENCE_KIND: kind}
    if chunk_index is not None:
        tagged[semconv.INFERENCE_CHUNK_INDEX] = chunk_index
    for k, v in attrs.items():
        tagged[f"inference.{k}"] = v
    with _tracer().start_as_current_span(name, attributes=tagged) as span:
        yield span


@contextmanager
def safety_span(
    name: str = semconv.SPAN_SAFETY_CHECK,
    *,
    check_name: str | None = None,
    severity: str = "info",
    **attrs: Any,
) -> Iterator[Span]:
    """Span around one safety check.

    The Python-side :class:`~openral_runner.safety.NullSafetyClient` uses
    this helper; the C++ safety kernel (planned at ``packages/safety/``)
    will emit a sibling ``safety.check`` span via ``opentelemetry-cpp``
    parented to the same ``rskill.tick`` via the W3C ``traceparent``
    carried on ``ActionChunk.msg`` (see :mod:`openral_observability.propagation`).

    Args:
        name: Span name.
        check_name: Specific check identifier (e.g. ``"workspace.aabb"``).
        severity: ``"info"``, ``"warn"``, or ``"violation"``.
        **attrs: Extra attributes recorded with a ``safety.`` prefix.

    Yields:
        The active span.
    """
    tagged: dict[str, Any] = {semconv.SAFETY_SEVERITY: severity}
    if check_name is not None:
        tagged[semconv.SAFETY_CHECK_NAME] = check_name
    for k, v in attrs.items():
        tagged[f"safety.{k}"] = v
    with _tracer().start_as_current_span(name, attributes=tagged) as span:
        yield span


@contextmanager
def reasoner_span(
    name: str = semconv.SPAN_REASONER_TICK,
    *,
    tick_idx: int | None = None,
    model: str | None = None,
    force: bool | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Span around one :meth:`openral_reasoner.ReasonerCore.tick` (ADR-0018 F4).

    Wraps the entire orchestrator pass — context render, LLM tool-use
    selection, retry-cap / min-interval gates, and dispatch routing on
    the ROS side. The Python-side :class:`ReasonerCore` opens the span;
    the surrounding ``reasoner_node`` reads
    :func:`openral_observability.propagation.current_traceparent` from
    inside this scope to stamp the outbound ``EmitPromptTool``
    PromptStamped's ``metadata_json`` (per ADR-0018 §6 "OTel context is
    the truth; ROS fields are set from it").

    Args:
        name: Span name. Default :data:`semconv.SPAN_REASONER_TICK`.
        tick_idx: Monotonic tick counter; recorded as
            ``reasoner.tick.idx`` (sortable in trace search).
        model: LLM model identifier from the active
            :attr:`ToolUseClient.model_id`; recorded as
            ``reasoner.model``.
        force: ``True`` when the tick was preempted by a high-severity
            ``FailureTrigger`` or a new ``PromptStamped``; recorded as
            ``reasoner.force``.
        **attrs: Extra attributes recorded with a ``reasoner.`` prefix.
            Callers should prefer the typed constants
            (:data:`semconv.REASONER_TOOL`,
            :data:`semconv.REASONER_RSKILL_ID`,
            :data:`semconv.REASONER_SUPPRESSED_REASON`,
            :data:`semconv.REASONER_ERROR_KIND`) and pass them via
            ``span.set_attribute`` from inside the context.

    Yields:
        The active :class:`opentelemetry.trace.Span`.

    Example:
        >>> from openral_observability import reasoner_span
        >>> with reasoner_span(tick_idx=0, model="fake") as span:
        ...     pass
    """
    tagged: dict[str, Any] = {}
    if tick_idx is not None:
        tagged[semconv.REASONER_TICK_IDX] = tick_idx
    if model is not None:
        tagged[semconv.REASONER_MODEL] = model
    if force is not None:
        tagged[semconv.REASONER_FORCE] = force
    for k, v in attrs.items():
        tagged[f"reasoner.{k}"] = v
    with _tracer().start_as_current_span(name, attributes=tagged) as span:
        yield span


def traced(
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a sync function in a span named after it.

    The span name defaults to ``module.qualname`` of the wrapped function.
    No attributes are auto-recorded; use :func:`rskill_span` /
    :func:`inference_span` directly for richer instrumentation.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with _tracer().start_as_current_span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
