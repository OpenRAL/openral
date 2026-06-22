"""SDK initialisation for tracing, metrics, and logging.

``configure_observability`` is idempotent: calling it twice with the same
arguments is a no-op.  Calling it with no endpoint (the default when the
``OTEL_EXPORTER_OTLP_ENDPOINT`` env var is unset) leaves the no-op default
:class:`~opentelemetry.trace.TracerProvider` /
:class:`~opentelemetry.metrics.MeterProvider` in place — span and metric
helpers still work, they just emit nothing.

This is required behaviour: CI runs without an OTLP collector and must
not fail. CLAUDE.md §9 calls out "observability as a hard dependency of
the actuation path" as an anti-pattern; the no-op fallback enforces that.

``shutdown_observability`` flushes all three providers and shuts them
down. It is registered via :mod:`atexit` on first successful
configuration so even short-lived scripts that forget to call it
explicitly will still drain the BatchSpanProcessor /
PeriodicExportingMetricReader / BatchLogRecordProcessor before the
process exits. Callers are encouraged to invoke it explicitly in a
``finally`` block to get a deterministic shutdown ordering and a
non-zero exit code on flush failures.
"""

from __future__ import annotations

import atexit
import os
import threading

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)

__all__ = [
    "configure_observability",
    "configure_worker_observability",
    "shutdown_observability",
]

_ENV_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
_ENV_PROTOCOL = "OTEL_EXPORTER_OTLP_PROTOCOL"
_ENV_METRIC_INTERVAL_MS = "OPENRAL_OTEL_METRIC_INTERVAL_MS"
_DEFAULT_METRIC_INTERVAL_MS = 5_000
_ENV_SAMPLE_RATIO = "OPENRAL_OTEL_SAMPLE_RATIO"
_ENV_SPAN_SCHEDULE_DELAY_MS = "OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS"
_DEFAULT_SPAN_SCHEDULE_DELAY_MS = 30


def _load_exporters() -> tuple[type, type, type]:
    """Return the (span, metric, log) OTLP exporter classes.

    Picks gRPC by default and switches to HTTP when
    ``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`` is set. The HTTP
    transport is what the in-tree `openral dashboard` speaks; without
    honouring the protocol env var, a user following the documented
    `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` recipe sees a stream
    of `StatusCode.UNAVAILABLE` exporter errors and no spans landing
    in the dashboard.

    Imports are deferred so the gRPC dep doesn't need to be importable
    when HTTP is selected and vice versa.
    """
    protocol = os.environ.get(_ENV_PROTOCOL, "").lower().strip()
    if protocol in {"http/protobuf", "http-protobuf", "http"}:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as _HttpLog,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as _HttpMetric,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as _HttpSpan,
        )

        return _HttpSpan, _HttpMetric, _HttpLog

    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter as _GrpcLog,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as _GrpcMetric,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as _GrpcSpan,
    )

    return _GrpcSpan, _GrpcMetric, _GrpcLog


_lock = threading.Lock()
_service_name: str | None = None
_endpoint: str | None = None
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_logger_provider: LoggerProvider | None = None
_atexit_registered = False


def configure_observability(
    *,
    service_name: str = "openral",
    endpoint: str | None = None,
    sample_ratio: float | None = None,
) -> bool:
    """Initialise OTel TracerProvider, MeterProvider, and LoggerProvider.

    Args:
        service_name: ``service.name`` resource attribute.
        endpoint: OTLP/gRPC endpoint (e.g. ``http://localhost:4317``).
            If ``None``, falls back to the ``OTEL_EXPORTER_OTLP_ENDPOINT``
            env var.  If still unset, the function returns ``False`` and
            leaves the default no-op providers in place.
        sample_ratio: Optional head-based sampling ratio for child spans
            without an explicit parent decision. ``None`` (the default)
            and ``1.0`` both mean ``ALWAYS_ON`` — every span exported.
            Values in ``(0, 1)`` install
            ``ParentBased(TraceIdRatioBased(ratio))``: spans inherit a
            parent's decision when one is in scope (so a cli.command
            sampled IN keeps its whole tick subtree), and untraced root
            spans are sampled at the given rate. Recommended for
            ``openral.run.mode == "hardware"`` per ADR-0010 to keep
            100 Hz over 24 h runs under the OTLP/gRPC headroom. Also
            reads ``OPENRAL_OTEL_SAMPLE_RATIO`` as a fallback.

    Returns:
        ``True`` if exporters were installed; ``False`` if no endpoint was
        resolved (no-op mode).
    """
    global \
        _service_name, \
        _endpoint, \
        _tracer_provider, \
        _meter_provider, \
        _logger_provider, \
        _atexit_registered

    resolved = endpoint if endpoint is not None else os.environ.get(_ENV_ENDPOINT)
    with _lock:
        if _service_name == service_name and _endpoint == resolved:
            return resolved is not None
        if resolved is None:
            _service_name = service_name
            _endpoint = None
            return False

        resource = Resource.create({"service.name": service_name})

        span_exporter_cls, metric_exporter_cls, log_exporter_cls = _load_exporters()
        is_http = os.environ.get(_ENV_PROTOCOL, "").lower().strip() in {
            "http/protobuf",
            "http-protobuf",
            "http",
        }
        if is_http:
            base = resolved.rstrip("/")
            span_exporter = span_exporter_cls(endpoint=f"{base}/v1/traces")
            metric_exporter = metric_exporter_cls(endpoint=f"{base}/v1/metrics")
            log_exporter = log_exporter_cls(endpoint=f"{base}/v1/logs")
        else:
            span_exporter = span_exporter_cls(endpoint=resolved, insecure=True)
            metric_exporter = metric_exporter_cls(endpoint=resolved, insecure=True)
            log_exporter = log_exporter_cls(endpoint=resolved, insecure=True)

        sampler = _resolve_sampler(sample_ratio)
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                span_exporter,
                schedule_delay_millis=_resolve_span_schedule_delay_ms(),
            )
        )
        trace.set_tracer_provider(tracer_provider)

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=_resolve_metric_interval_ms(),
                )
            ],
        )
        metrics.set_meter_provider(meter_provider)

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logger_provider)

        _service_name = service_name
        _endpoint = resolved
        _tracer_provider = tracer_provider
        _meter_provider = meter_provider
        _logger_provider = logger_provider

        # Install the structlog→OTel log bridge once both providers exist.
        # Imported lazily to avoid a circular dep at module import time.
        from openral_observability.logging import install_structlog_bridge

        install_structlog_bridge(logger_provider)

        # Start the host sampler so the dashboard's System health card sees
        # CPU / RAM / GPU. No-op when neither psutil nor pynvml is importable.
        from openral_observability.system_metrics import (
            start_system_metrics_collector,
        )

        start_system_metrics_collector()

        if not _atexit_registered:
            atexit.register(shutdown_observability)
            _atexit_registered = True
        return True


def configure_worker_observability(
    service_name: str,
    *,
    endpoint: str | None = None,
    sample_ratio: float | None = None,
) -> bool:
    """Bootstrap observability in a spawned worker so it joins the parent trace.

    The cross-process counterpart of :func:`configure_observability` for a
    subprocess (the dispatcher, the future fleet supervisor). It does two
    things in order:

    1. Calls :func:`configure_observability` so the worker gets its own OTLP
       pipeline **and** the structlog→OTel log bridge (logs and spans both
       ship to the collector with the worker's ``service.name``).
    2. Calls
       :func:`openral_observability.propagation.attach_traceparent_from_env`
       so the worker's root OTel context is the parent process's span —
       every span the worker opens, and every log line it stamps, carries
       the parent's ``trace_id``.

    The parent **must** propagate its active context into the child's
    environment. Spawn the worker with
    ``env={**os.environ, **traceparent_env()}`` (see
    :func:`openral_observability.propagation.traceparent_env`); otherwise
    step 2 is a no-op and the worker starts a fresh, uncorrelated trace.

    Args:
        service_name: ``service.name`` resource attribute for the worker —
            give it a distinct value (e.g. ``"openral-dispatcher"``) so its
            spans are filterable from the parent's.
        endpoint: OTLP endpoint, forwarded to
            :func:`configure_observability`. ``None`` falls back to the
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var (inherited from the
            parent), then to no-op mode.
        sample_ratio: Optional head-based sampling ratio, forwarded to
            :func:`configure_observability`. ``ParentBased`` sampling means
            the worker inherits the parent's sampling decision when the
            attached context carries one.

    Returns:
        Whatever :func:`configure_observability` returns — ``True`` if
        exporters were installed, ``False`` for the no-op path. (The
        context attach in step 2 happens regardless, so trace correlation
        works even before an endpoint is configured.)

    Example:
        >>> from openral_observability import configure_worker_observability
        >>> # In a worker entrypoint, after the parent spawned us with
        >>> # env={**os.environ, **traceparent_env()}:
        >>> _ = configure_worker_observability("openral-dispatcher")
    """
    installed = configure_observability(
        service_name=service_name,
        endpoint=endpoint,
        sample_ratio=sample_ratio,
    )
    # Imported lazily to match the module's no-import-at-top-for-cycles style.
    from openral_observability.propagation import attach_traceparent_from_env

    attach_traceparent_from_env()
    return installed


def _resolve_sampler(sample_ratio: float | None) -> Sampler:
    """Resolve the sampler from explicit arg + env, defaulting to ALWAYS_ON.

    Resolution order: explicit ``sample_ratio`` arg → ``OPENRAL_OTEL_SAMPLE_RATIO``
    env var → ``None`` (always-on). ``1.0`` is treated as always-on so
    common configurations don't pay the ``ParentBased`` indirection. Values
    outside ``(0, 1]`` fall back to always-on with a quiet WARN-equivalent
    so a typo doesn't accidentally drop every span.
    """
    if sample_ratio is None:
        raw = os.environ.get(_ENV_SAMPLE_RATIO)
        if raw is None:
            return ALWAYS_ON
        try:
            sample_ratio = float(raw)
        except ValueError:
            return ALWAYS_ON
    if sample_ratio >= 1.0 or sample_ratio <= 0.0:
        return ALWAYS_ON
    return ParentBased(root=TraceIdRatioBased(sample_ratio))


def _resolve_metric_interval_ms() -> int:
    raw = os.environ.get(_ENV_METRIC_INTERVAL_MS)
    if raw is None:
        return _DEFAULT_METRIC_INTERVAL_MS
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_METRIC_INTERVAL_MS
    return parsed if parsed > 0 else _DEFAULT_METRIC_INTERVAL_MS


def _resolve_span_schedule_delay_ms() -> int:
    """Resolve the BatchSpanProcessor flush interval in milliseconds.

    Defaults to :data:`_DEFAULT_SPAN_SCHEDULE_DELAY_MS` (30 ms, ~33 Hz) so a
    local dashboard refreshes at ~25 Hz instead of the OTel default 5 s
    batching. The flush rate is set ~1.3x the thumbnail rate (not equal to it):
    the dashboard keeps only the latest frame per batch, so a flush period equal
    to the emit period aliases and drops ~15% of frames; the headroom captures
    every frame.
    Production/cloud deployments that prefer coarser batching (less export
    traffic) raise ``OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS``. Invalid /
    non-positive values fall back to the default (mirrors
    :func:`_resolve_metric_interval_ms`).
    """
    raw = os.environ.get(_ENV_SPAN_SCHEDULE_DELAY_MS)
    if raw is None:
        return _DEFAULT_SPAN_SCHEDULE_DELAY_MS
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_SPAN_SCHEDULE_DELAY_MS
    return parsed if parsed > 0 else _DEFAULT_SPAN_SCHEDULE_DELAY_MS


def shutdown_observability() -> None:
    """Flush and shut down the OTel providers installed by :func:`configure_observability`.

    Idempotent and safe to call when no exporter was installed (e.g. when
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` was unset). Drains the
    BatchSpanProcessor, PeriodicExportingMetricReader, and
    BatchLogRecordProcessor queues so spans / metrics / logs reach the
    collector before the process exits.
    """
    global _tracer_provider, _meter_provider, _logger_provider

    with _lock:
        tracer_provider = _tracer_provider
        meter_provider = _meter_provider
        logger_provider = _logger_provider
        _tracer_provider = None
        _meter_provider = None
        _logger_provider = None

    # Stop the host sampler before draining the meter so the final tick
    # lands in the export batch instead of being dropped on shutdown.
    from openral_observability.system_metrics import stop_system_metrics_collector

    stop_system_metrics_collector()

    if tracer_provider is not None:
        tracer_provider.shutdown()
    if meter_provider is not None:
        meter_provider.shutdown()
    if logger_provider is not None:
        logger_provider.shutdown()
