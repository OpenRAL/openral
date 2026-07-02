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

import os
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import Token
from typing import cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context.context import Context
from opentelemetry.propagators.textmap import (
    DefaultGetter,
    DefaultSetter,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

__all__ = [
    "attach_traceparent_from_env",
    "current_traceparent",
    "extract_traceparent",
    "inject_traceparent",
    "remote_parent_from_env",
    "traceparent_env",
]

_PROPAGATOR = TraceContextTextMapPropagator()
_TRACEPARENT_HEADER = "traceparent"
_TRACESTATE_HEADER = "tracestate"

# Environment-variable carrier names. These mirror the OTel SDK's own
# ``OTEL_TRACEPARENT`` / ``OTEL_TRACESTATE`` convention so a worker spawned
# with this dict in its environment is correlated the same way a manually
# attached one is. The W3C *header* names (``traceparent`` / ``tracestate``)
# are lowercase by spec; the *env-var* names are uppercase by convention.
_ENV_TRACEPARENT = "OTEL_TRACEPARENT"
_ENV_TRACESTATE = "OTEL_TRACESTATE"


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


def traceparent_env(carrier: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment-variable dict carrying the active span's trace context.

    This is the cross-process bootstrap for spawning a worker (the
    dispatcher, the future fleet supervisor) so its logs and spans land in
    the *parent* trace. Pass the returned dict as ``env=`` to
    :mod:`subprocess` / :mod:`multiprocessing`; the worker recovers the
    parent context with :func:`attach_traceparent_from_env` (or, more
    conveniently, :func:`openral_observability.configure_worker_observability`).

    The keys are ``OTEL_TRACEPARENT`` and — only when the active span
    carries a non-empty ``tracestate`` — ``OTEL_TRACESTATE``. The values
    are the standard W3C strings produced by :func:`inject_traceparent`,
    re-keyed from the W3C header names to the env-var names.

    Args:
        carrier: Optional dict the env vars are *also* written into as a
            side effect (handy for merging into an existing ``os.environ``
            copy). A fresh dict is returned regardless.

    Returns:
        A dict suitable for ``env=`` injection, or ``{}`` when no valid span
        is in scope (no-op observability mode, or called outside any span).

    Example:
        >>> from opentelemetry import trace
        >>> from opentelemetry.sdk.trace import TracerProvider
        >>> from openral_observability.propagation import traceparent_env
        >>> tracer = TracerProvider().get_tracer("doctest")
        >>> with trace.use_span(tracer.start_span("demo"), end_on_exit=True):
        ...     env = traceparent_env()
        >>> "OTEL_TRACEPARENT" in env
        True
    """
    headers = inject_traceparent()
    out: dict[str, str] = {}
    if _TRACEPARENT_HEADER in headers:
        out[_ENV_TRACEPARENT] = headers[_TRACEPARENT_HEADER]
    tracestate = headers.get(_TRACESTATE_HEADER)
    if tracestate:
        out[_ENV_TRACESTATE] = tracestate
    if carrier is not None:
        carrier.update(out)
    return out


def attach_traceparent_from_env(env: Mapping[str, str] | None = None) -> object | None:
    """Attach the trace context carried in ``OTEL_TRACEPARENT`` env vars, if any.

    The worker-side counterpart of :func:`traceparent_env`. Reads
    ``OTEL_TRACEPARENT`` / ``OTEL_TRACESTATE`` from ``env`` (default
    :data:`os.environ`) and, when a ``traceparent`` is present, calls
    :func:`opentelemetry.context.attach` so subsequent spans become
    children of the parent process's span.

    The caller owns the returned detach token: pass it to
    :func:`opentelemetry.context.detach` to restore the previous context.
    Use :func:`remote_parent_from_env` instead when a ``with`` block scopes
    the attach/detach for you.

    Args:
        env: Mapping to read the env-var carrier from. Defaults to
            :data:`os.environ`.

    Returns:
        The detach token from :func:`opentelemetry.context.attach`, or
        ``None`` when no (non-empty) ``OTEL_TRACEPARENT`` is present.

    Example:
        >>> from opentelemetry import context as otel_context, trace
        >>> from opentelemetry.sdk.trace import TracerProvider
        >>> from openral_observability.propagation import (
        ...     attach_traceparent_from_env,
        ...     traceparent_env,
        ... )
        >>> tracer = TracerProvider().get_tracer("doctest")
        >>> with trace.use_span(tracer.start_span("demo"), end_on_exit=True):
        ...     env = traceparent_env()
        >>> token = attach_traceparent_from_env(env)
        >>> token is not None
        True
        >>> otel_context.detach(token)
    """
    source = env if env is not None else os.environ
    traceparent = source.get(_ENV_TRACEPARENT)
    if not traceparent:
        return None
    tracestate = source.get(_ENV_TRACESTATE)
    ctx = extract_traceparent(traceparent, tracestate)
    return otel_context.attach(ctx)


@contextmanager
def remote_parent_from_env(env: Mapping[str, str] | None = None) -> Iterator[object | None]:
    """Scope the parent trace context carried in env vars for a worker ``main()``.

    Attaches the context from :func:`attach_traceparent_from_env` on enter
    and detaches it on exit, so a worker entrypoint can wrap its body and be
    sure the global OTel context is restored afterwards (relevant when the
    same process later runs unrelated work). A no-op — yielding ``None`` and
    detaching nothing — when no ``OTEL_TRACEPARENT`` is present.

    Args:
        env: Mapping to read the env-var carrier from. Defaults to
            :data:`os.environ`.

    Yields:
        The detach token (or ``None`` when nothing was attached).

    Example:
        >>> from opentelemetry import trace
        >>> from opentelemetry.sdk.trace import TracerProvider
        >>> from openral_observability.propagation import (
        ...     remote_parent_from_env,
        ...     traceparent_env,
        ... )
        >>> tracer = TracerProvider().get_tracer("doctest")
        >>> with trace.use_span(tracer.start_span("demo"), end_on_exit=True):
        ...     env = traceparent_env()
        >>> with remote_parent_from_env(env) as token:
        ...     token is not None
        True
    """
    token = attach_traceparent_from_env(env)
    try:
        yield token
    finally:
        if token is not None:
            # ``attach_traceparent_from_env`` widens the token to ``object``
            # to keep the OTel ``Token`` type off the public surface; narrow
            # it back for ``detach``, which is the value ``attach`` returned.
            otel_context.detach(cast("Token[Context]", token))
