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


# ---------------------------------------------------------------------------
# Partial NVML support (unified-memory SoCs: GB10 / DGX Spark, Thor)
#
# `nvmlDeviceGetMemoryInfo` raises NVMLError_NotSupported on those devices —
# there is no discrete VRAM pool — while `nvmlDeviceGetUtilizationRates` works.
# The stand-in below is a driver-boundary double: a real GPU cannot be asked to
# report "not supported", and the CI runners that do have a GPU have a discrete
# one, so this behaviour is unreachable with real hardware.
# ---------------------------------------------------------------------------


class _NotSupportedError(Exception):
    """Stands in for ``pynvml.NVMLError_NotSupported``."""


class _UtilRates:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _UnifiedMemoryNvml:
    """One device that supports utilisation but not the memory query."""

    def __init__(self) -> None:
        self.memory_calls = 0
        self.util_calls = 0

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802  # reason: mirrors the pynvml API
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:  # noqa: N802  # reason: pynvml API
        return f"handle-{index}"

    def nvmlDeviceGetName(self, handle: str) -> str:  # noqa: N802  # reason: pynvml API
        return "NVIDIA GB10"

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> _UtilRates:  # noqa: N802  # reason: pynvml API
        self.util_calls += 1
        return _UtilRates(gpu=42)

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> object:  # noqa: N802  # reason: pynvml API
        self.memory_calls += 1
        raise _NotSupportedError("Not Supported")


def _instruments() -> dict[str, object]:
    from openral_observability import metrics as _metrics

    return {
        "cpu_util": _metrics.get_system_cpu_util_pct(),
        "ram_used": _metrics.get_system_ram_used_mb(),
        "ram_total": _metrics.get_system_ram_total_mb(),
        "gpu_mem_used": _metrics.get_system_gpu_memory_used_mb(),
        "gpu_mem_total": _metrics.get_system_gpu_memory_total_mb(),
        "gpu_util": _metrics.get_system_gpu_util_pct(),
    }


def test_unsupported_memory_query_does_not_cost_the_gpu_util_metric(
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """An unsupported memory query must not take utilisation down with it.

    Regression: the unguarded `nvmlDeviceGetMemoryInfo` propagated out of
    `_sample_once`, so a GB10 lost every GPU metric each tick — including the
    utilisation the device does report — and logged a traceback per interval.
    """
    from openral_observability import system_metrics

    nvml = _UnifiedMemoryNvml()
    prev: dict[str, float] = {}

    # Must not raise, unlike the pre-fix behaviour.
    system_metrics._sample_once(None, nvml, _instruments(), prev)

    assert nvml.util_calls == 1
    assert nvml.memory_calls == 1
    # Utilisation survived; the unsupported memory figures were skipped.
    assert _find_metric(memory_metric_reader, semconv.METRIC_SYSTEM_GPU_UTIL_PCT) is not None
    assert any(key.startswith("gpu_util.") for key in prev)
    assert not any(key.startswith("gpu_used.") or key.startswith("gpu_total.") for key in prev)


def test_unsupported_query_is_logged_once_not_every_tick(
    memory_metric_reader: InMemoryMetricReader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The degradation is a permanent hardware property, so log it once."""
    from openral_observability import system_metrics

    system_metrics._UNSUPPORTED_LOGGED.clear()
    nvml = _UnifiedMemoryNvml()
    prev: dict[str, float] = {}

    with caplog.at_level("INFO", logger="openral_observability.system_metrics"):
        for _ in range(5):
            system_metrics._sample_once(None, nvml, _instruments(), prev)

    assert nvml.memory_calls == 5, "every tick should still attempt the query"
    matching = [r for r in caplog.records if "does not support the memory query" in r.message]
    assert len(matching) == 1, f"expected exactly one degradation log, got {len(matching)}"
