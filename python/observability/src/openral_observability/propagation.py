"""W3C TraceContext inject / extract helpers for cross-process correlation.

OpenRAL's `ActionChunk.msg`, `ExecuteRskill.action`, and `FailureTrigger.msg`
ROS 2 IDL all carry a ``string trace_id`` field. Per the OTel design
doc §5, that field is reinterpreted as the full W3C ``traceparent``
value (``00-<trace>-<span>-<flags>``) so the C++ safety kernel (and any
other ROS-side consumer) can resume the trace from the Python producer.

The implementation is a thin wrapper over OTel's stock
:class:`~opentelemetry.propagators.textmap.TraceContextTextMapPropagator`
— W3C-compliant out of the box, with no bespoke parsing.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagators.textmap import (
    DefaultGetter,
    DefaultSetter,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

__all__ = [
    "current_traceparent",
    "extract_traceparent",
    "inject_traceparent",
]

_PROPAGATOR = TraceContextTextMapPropagator()
_TRACEPARENT_HEADER = "traceparent"
_TRACESTATE_HEADER = "tracestate"


def current_traceparent() -> str | None:
    """Return the W3C ``traceparent`` value for the active span, or ``None``.

    Returns ``None`` when no valid span is in scope (e.g. before the runner
    has opened ``rskill.tick``, or in fully no-op observability mode). The
    bool-ish nature makes it easy to use as ``msg.trace_id = current_traceparent() or ""``.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    # W3C v0 traceparent: ``<version>-<trace_id>-<parent_id>-<flags>``.
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"


def inject_traceparent(carrier: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Inject the active span's ``traceparent`` (and optional ``tracestate``) into a dict.

    Args:
        carrier: Dict to inject into. A fresh dict is created when ``None``.

    Returns:
        A new dict containing ``traceparent`` (and ``tracestate`` when
        non-empty) for the active span. Empty when no valid span is in
        scope. When ``carrier`` is provided, the same keys are also
        written into it as a side effect.

    Example:
        >>> from openral_observability import rskill_span
        >>> from openral_observability.propagation import inject_traceparent
        >>> with rskill_span("rskill.execute", rskill_id="demo"):
        ...     headers = inject_traceparent()
        >>> "traceparent" in headers
        True
    """
    out: dict[str, str] = dict(carrier) if carrier is not None else {}
    # ``inject`` is generic over the carrier type but mypy cannot infer it
    # through ``MutableMapping``; the concrete dict satisfies the
    # ``set: (carrier, key, value) -> None`` shape used by ``DefaultSetter``.
    _PROPAGATOR.inject(out, setter=DefaultSetter())  # type: ignore[misc]  # reason: propagator CarrierT is inferred from setter
    if carrier is not None:
        carrier.update(out)
    return out


def extract_traceparent(traceparent: str, tracestate: str | None = None) -> otel_context.Context:
    """Parse a W3C ``traceparent`` string into an OTel :class:`Context`.

    The returned context is suitable for use with
    :func:`opentelemetry.context.attach` /
    :func:`opentelemetry.trace.use_span` to make subsequent spans children
    of the extracted parent.

    Args:
        traceparent: The full W3C value (``00-<trace>-<span>-<flags>``).
            Pass the ``ActionChunk.msg.trace_id`` field directly per the
            design doc's Option B.
        tracestate: Optional vendor extension (W3C). Empty / ``None`` is
            normal in OpenRAL.

    Returns:
        OTel :class:`Context`. When ``traceparent`` is empty or malformed,
        the propagator returns the current (empty) context — callers that
        want a span-context check should consult
        :func:`opentelemetry.trace.get_current_span` after attaching.

    Example:
        >>> from openral_observability.propagation import extract_traceparent, inject_traceparent
        >>> from openral_observability import rskill_span
        >>> with rskill_span("rskill.execute", rskill_id="demo"):
        ...     headers = inject_traceparent()
        >>> ctx = extract_traceparent(headers["traceparent"])
        >>> ctx is not None
        True
    """
    carrier: dict[str, str] = {_TRACEPARENT_HEADER: traceparent}
    if tracestate:
        carrier[_TRACESTATE_HEADER] = tracestate
    return _PROPAGATOR.extract(carrier, getter=DefaultGetter())


def _extract_from_mapping(carrier: Mapping[str, str]) -> otel_context.Context:
    """Test helper: extract context from a generic header-style mapping."""
    return _PROPAGATOR.extract(carrier, getter=DefaultGetter())
