"""configure_observability honours the env var and is a no-op without it."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_observability import (
    _sdk,
    configure_observability,
    shutdown_observability,
)


@pytest.fixture(autouse=True)
def _reset_sdk_state() -> Iterator[None]:
    """Wipe module-level provider state so each test starts from a no-op SDK."""
    _sdk._service_name = None
    _sdk._endpoint = None
    _sdk._tracer_provider = None
    _sdk._meter_provider = None
    _sdk._logger_provider = None
    yield
    shutdown_observability()
    _sdk._service_name = None
    _sdk._endpoint = None
    _sdk._tracer_provider = None
    _sdk._meter_provider = None
    _sdk._logger_provider = None


def test_no_op_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert configure_observability(service_name="test-no-op") is False


def test_idempotent_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert configure_observability(service_name="test-idem") is False
    assert configure_observability(service_name="test-idem") is False


def test_endpoint_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://from-env:4317")
    # Explicit None falls back to env; explicit "" does not override.
    # We do not actually export here; we just assert the return value
    # signals "exporter installed" when an endpoint is resolved.
    assert configure_observability(service_name="test-env", endpoint=None) is True


def test_shutdown_is_safe_when_unconfigured() -> None:
    """shutdown_observability() must be a no-op if no exporter was installed."""
    shutdown_observability()
    shutdown_observability()  # idempotent


def test_shutdown_flushes_and_clears_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """After shutdown the module-level providers are cleared and shutdown() ran."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    assert configure_observability(service_name="test-shutdown") is True

    tracer = _sdk._tracer_provider
    meter = _sdk._meter_provider
    logger = _sdk._logger_provider
    assert tracer is not None and meter is not None and logger is not None

    shutdown_observability()

    assert _sdk._tracer_provider is None
    assert _sdk._meter_provider is None
    assert _sdk._logger_provider is None
    # Calling shutdown again is harmless.
    shutdown_observability()


def test_configure_starts_system_metrics_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_observability must boot the host sampler so the dashboard's
    System health card receives CPU / RAM / GPU gauges. Without this wiring
    the card shows "no system metrics yet" — the bug behind this test.
    """
    pytest.importorskip("psutil")  # collector no-ops without psutil/pynvml.
    from openral_observability import system_metrics

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    assert system_metrics._thread is None, "collector should start out idle"

    assert configure_observability(service_name="test-system-metrics") is True
    try:
        thread = system_metrics._thread
        assert thread is not None, "collector thread was never started"
        assert thread.is_alive(), "collector thread exited immediately"
    finally:
        shutdown_observability()

    assert system_metrics._thread is None, "shutdown must stop the collector"


def test_no_op_configure_does_not_start_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no endpoint is resolved, configure_observability is a no-op and
    must not spin up the sampler — otherwise a script with no OTel endpoint
    would still pay for a background psutil/pynvml loop.
    """
    from openral_observability import system_metrics

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert system_metrics._thread is None

    assert configure_observability(service_name="test-noop") is False
    assert system_metrics._thread is None


# ── Sampler resolution ──────────────────────────────────────────────────────


def test_resolve_sampler_default_always_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """No arg, no env var → ALWAYS_ON."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    monkeypatch.delenv("OPENRAL_OTEL_SAMPLE_RATIO", raising=False)
    assert _sdk._resolve_sampler(None) is ALWAYS_ON


def test_resolve_sampler_ratio_one_is_always_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sample_ratio=1.0`` skips the ParentBased indirection for the common case."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    monkeypatch.delenv("OPENRAL_OTEL_SAMPLE_RATIO", raising=False)
    assert _sdk._resolve_sampler(1.0) is ALWAYS_ON


def test_resolve_sampler_ratio_zero_falls_back_to_always_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo of ``0.0`` would drop every span — fall back to ALWAYS_ON instead."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    monkeypatch.delenv("OPENRAL_OTEL_SAMPLE_RATIO", raising=False)
    assert _sdk._resolve_sampler(0.0) is ALWAYS_ON


def test_resolve_sampler_fractional_returns_parent_based(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sample_ratio=0.1`` returns ``ParentBased(TraceIdRatioBased(0.1))``."""
    from opentelemetry.sdk.trace.sampling import ParentBased

    monkeypatch.delenv("OPENRAL_OTEL_SAMPLE_RATIO", raising=False)
    sampler = _sdk._resolve_sampler(0.1)
    assert isinstance(sampler, ParentBased)
    # ``ParentBased.get_description()`` carries the ratio in its
    # ``TraceIdRatioBased{0.100000}`` root rendering.
    assert "0.1" in sampler.get_description()


def test_resolve_sampler_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the arg is None, the env var drives the choice."""
    from opentelemetry.sdk.trace.sampling import ParentBased

    monkeypatch.setenv("OPENRAL_OTEL_SAMPLE_RATIO", "0.25")
    sampler = _sdk._resolve_sampler(None)
    assert isinstance(sampler, ParentBased)
    assert "0.25" in sampler.get_description()


def test_resolve_sampler_malformed_env_var_is_always_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A garbage value in the env var must not silently drop spans."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    monkeypatch.setenv("OPENRAL_OTEL_SAMPLE_RATIO", "not-a-float")
    assert _sdk._resolve_sampler(None) is ALWAYS_ON


# ── Exporter selection (gRPC vs HTTP) ──────────────────────────────────────


def test_load_exporters_defaults_to_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``OTEL_EXPORTER_OTLP_PROTOCOL`` is unset, gRPC exporters are used."""
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter as GrpcLog,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as GrpcMetric,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcSpan,
    )

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    span_cls, metric_cls, log_cls = _sdk._load_exporters()
    assert span_cls is GrpcSpan
    assert metric_cls is GrpcMetric
    assert log_cls is GrpcLog


def test_load_exporters_http_protobuf_picks_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`` selects the HTTP exporters.

    Required for `openral dashboard` — the dashboard speaks OTLP/HTTP, so a sim
    run that doesn't honour the protocol env var fails with
    ``StatusCode.UNAVAILABLE`` and no spans land in the UI.
    """
    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
        OTLPLogExporter as HttpLog,
    )
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HttpMetric,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpan,
    )

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    span_cls, metric_cls, log_cls = _sdk._load_exporters()
    assert span_cls is HttpSpan
    assert metric_cls is HttpMetric
    assert log_cls is HttpLog


def test_configure_http_protobuf_installs_http_span_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: configure with http/protobuf, verify HTTP span exporter wired."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpan,
    )

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:18765")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    assert configure_observability(service_name="test-http") is True

    tracer_provider = _sdk._tracer_provider
    assert tracer_provider is not None
    # Walk the span processor chain to find the OTLP BatchSpanProcessor.
    found_http = False
    for proc in tracer_provider._active_span_processor._span_processors:  # type: ignore[attr-defined]
        exporter = getattr(proc, "span_exporter", None)
        if isinstance(exporter, HttpSpan):
            found_http = True
            break
    assert found_http, "expected an HTTP OTLPSpanExporter on the tracer provider"


def test_span_schedule_delay_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS", raising=False)
    assert _sdk._resolve_span_schedule_delay_ms() == 30


def test_span_schedule_delay_valid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS", "1000")
    assert _sdk._resolve_span_schedule_delay_ms() == 1000


def test_span_schedule_delay_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS", "not-a-number")
    assert _sdk._resolve_span_schedule_delay_ms() == 30


def test_span_schedule_delay_non_positive_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS", "0")
    assert _sdk._resolve_span_schedule_delay_ms() == 30
