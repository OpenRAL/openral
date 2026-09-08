"""Shared fixtures for observability tests.

Installs an in-memory span exporter on the global TracerProvider and an
in-memory metric reader on the global MeterProvider so each test can
assert on emitted telemetry without needing a live OTLP collector.

Per CLAUDE.md §1.11 / §5.4 these are *real* OTel SDK components — not
mocks. Tests exercise the same provider classes shipping in production;
only the exporter is swapped for an in-memory one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from opentelemetry import metrics, trace
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def av() -> Callable[[object], AnyValue]:
    """Return the OTLP ``AnyValue`` converter shared by the dashboard tests."""

    def _av(value: object) -> AnyValue:
        if isinstance(value, bool):
            return AnyValue(bool_value=value)
        if isinstance(value, int):
            return AnyValue(int_value=value)
        if isinstance(value, float):
            return AnyValue(double_value=value)
        return AnyValue(string_value=str(value))

    return _av


@pytest.fixture
def _find_metric() -> Callable[[InMemoryMetricReader, str], object | None]:
    """Return a lookup for a named metric in an :class:`InMemoryMetricReader` snapshot."""

    def _find_metric(reader: InMemoryMetricReader, name: str) -> object | None:
        data = reader.get_metrics_data()
        if data is None:
            return None
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == name:
                        return metric
        return None

    return _find_metric


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Replace the global TracerProvider with one that records to memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # opentelemetry-api guards against re-setting the global provider once
    # it has been set; bypass that by writing through the private holder.
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


@pytest.fixture
def memory_metric_reader() -> Iterator[InMemoryMetricReader]:
    """Replace the global MeterProvider with one whose reader keeps data in memory.

    Use :meth:`InMemoryMetricReader.get_metrics_data` to inspect emitted
    instruments; the reader caches the latest data point per attribute set
    so tests can assert on aggregated state.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # Same private-holder dance as TracerProvider — the API enforces
    # set-once semantics that get in the way of per-test isolation.
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    metrics.set_meter_provider(provider)

    # The metrics module caches instruments per ``id(meter)``; swapping the
    # provider invalidates those keys naturally, but we drop the cache
    # explicitly so two tests sharing the same fixture order don't see
    # stale instruments.
    from openral_observability.metrics import _reset_instrument_cache

    _reset_instrument_cache()

    try:
        yield reader
    finally:
        provider.shutdown()
        _reset_instrument_cache()
