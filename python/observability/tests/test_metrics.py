"""Metric instruments record on the in-memory reader.

Real SDK, in-memory exporter — no mocks (CLAUDE.md §1.11 / §5.4).
"""

from __future__ import annotations

from typing import cast

from openral_observability import metrics as ral_metrics
from openral_observability import semconv
from opentelemetry.sdk.metrics.export import (
    HistogramDataPoint,
    InMemoryMetricReader,
    NumberDataPoint,
)


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


def test_tick_duration_records_histogram(memory_metric_reader: InMemoryMetricReader) -> None:
    h = ral_metrics.get_tick_duration()
    h.record(12.5, {semconv.LABEL_RSKILL_ID: "demo", semconv.LABEL_RSKILL_REVISION: "abc"})
    h.record(7.2, {semconv.LABEL_RSKILL_ID: "demo", semconv.LABEL_RSKILL_REVISION: "abc"})

    metric = _find_metric(memory_metric_reader, semconv.METRIC_TICK_DURATION)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]  # reason: in-memory reader shape
    assert len(points) == 1
    point = cast(HistogramDataPoint, points[0])
    assert point.count == 2
    assert point.sum > 0
    assert point.attributes[semconv.LABEL_RSKILL_ID] == "demo"


def test_safety_violations_counter(memory_metric_reader: InMemoryMetricReader) -> None:
    c = ral_metrics.get_safety_violations()
    c.add(1, {semconv.LABEL_CHECK_NAME: "workspace", semconv.LABEL_SEVERITY: "violation"})
    c.add(2, {semconv.LABEL_CHECK_NAME: "workspace", semconv.LABEL_SEVERITY: "violation"})

    metric = _find_metric(memory_metric_reader, semconv.METRIC_SAFETY_VIOLATIONS)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = cast(NumberDataPoint, points[0])
    assert point.value == 3


def test_world_state_updown_counter(memory_metric_reader: InMemoryMetricReader) -> None:
    g = ral_metrics.get_world_state_components_stale()
    g.add(1)
    g.add(2)
    g.add(-1)

    metric = _find_metric(memory_metric_reader, semconv.METRIC_WORLD_STATE_COMPONENTS_STALE)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = cast(NumberDataPoint, points[0])
    assert point.value == 2


def test_record_histogram_ms_skips_negatives(memory_metric_reader: InMemoryMetricReader) -> None:
    h = ral_metrics.get_inference_duration()
    ral_metrics.record_histogram_ms(h, -1.0, {semconv.LABEL_ENGINE: "torch"})
    ral_metrics.record_histogram_ms(h, float("nan"), {semconv.LABEL_ENGINE: "torch"})
    ral_metrics.record_histogram_ms(h, 5.0, {semconv.LABEL_ENGINE: "torch"})

    metric = _find_metric(memory_metric_reader, semconv.METRIC_INFERENCE_DURATION)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = cast(HistogramDataPoint, points[0])
    # Only the valid 5.0 ms sample is recorded; -1.0 and NaN are dropped.
    assert point.count == 1


def test_get_meter_returns_openral_meter() -> None:
    """``get_meter`` resolves against the global MeterProvider unconditionally."""
    meter = ral_metrics.get_meter()
    assert meter is not None
    # Sanity-check: we get a Meter, not None or a string.
    h = meter.create_histogram("openral.test.smoke", unit="ms")
    h.record(1.0)
