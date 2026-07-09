"""End-to-end coverage for the background system-metrics sampler.

Drives :func:`openral_observability.system_metrics.start_system_metrics_collector`
against the real OTel meter + in-memory reader (CLAUDE.md §1.11 / §5.4
— no mocks) and asserts that the ``openral.system.*`` instruments
receive at least one update. Skips cleanly if neither ``psutil`` nor
``pynvml`` is importable; the production deployment relies on at
least ``psutil`` so CI runners install it.
"""

from __future__ import annotations

import time

import pytest
from openral_observability import semconv
from openral_observability.system_metrics import (
    start_system_metrics_collector,
    stop_system_metrics_collector,
)
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

pytest.importorskip("psutil")


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


def test_collector_emits_cpu_and_ram_gauges(
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """The sampler updates the ``openral.system.cpu`` / ``ram`` gauges within ~1.5 s."""
    started = start_system_metrics_collector(interval_s=0.1)
    assert started, "psutil is importable yet the sampler refused to start"
    try:
        # Two sample intervals + scheduling slack.
        time.sleep(0.4)
    finally:
        stop_system_metrics_collector(timeout_s=1.0)

    cpu_metric = _find_metric(memory_metric_reader, semconv.METRIC_SYSTEM_CPU_UTIL_PCT)
    assert cpu_metric is not None, "openral.system.cpu.utilization_pct not exported"
    cpu_points = list(cpu_metric.data.data_points)  # type: ignore[attr-defined]
    assert cpu_points, "no CPU data points exported"
    cpu_value = float(cpu_points[0].value)
    # 0-100 inclusive; psutil clamps and we only emit deltas so cumulative
    # ends up equal to the latest absolute reading.
    assert 0.0 <= cpu_value <= 100.0, f"implausible cpu_util_pct={cpu_value}"

    ram_used = _find_metric(memory_metric_reader, semconv.METRIC_SYSTEM_RAM_USED_MB)
    assert ram_used is not None
    ram_used_points = list(ram_used.data.data_points)  # type: ignore[attr-defined]
    assert ram_used_points
    assert float(ram_used_points[0].value) > 0.0

    ram_total = _find_metric(memory_metric_reader, semconv.METRIC_SYSTEM_RAM_TOTAL_MB)
    assert ram_total is not None
    ram_total_points = list(ram_total.data.data_points)  # type: ignore[attr-defined]
    assert ram_total_points
    assert float(ram_total_points[0].value) > 0.0


def test_collector_is_idempotent() -> None:
    """Double-start does not spawn two threads or raise."""
    started_first = start_system_metrics_collector(interval_s=0.2)
    started_again = start_system_metrics_collector(interval_s=0.2)
    try:
        assert started_first is True
        assert started_again is True
    finally:
        stop_system_metrics_collector(timeout_s=1.0)
