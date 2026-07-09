"""OpenRAL OpenTelemetry metric instruments.

Module-level lazy accessors for every metric instrument an OpenRAL layer
emits. The functions return ``opentelemetry.metrics.*`` instruments
backed by the currently installed :class:`~opentelemetry.metrics.MeterProvider`
— including the no-op provider in place when
:func:`~openral_observability.configure_observability` has not yet
been called with an endpoint. This keeps the helpers safe to import and
call from anywhere (hot path included).

Instruments are cached per-name on the active meter so the per-call
overhead is one dict lookup. The cache is keyed on the meter object
identity, so swapping the global :class:`MeterProvider` (the test
fixture pattern) invalidates the cache automatically.

Cardinality discipline (design §9): metric labels are restricted to the
closed-set vocabularies declared in :mod:`openral_observability.semconv`.
High-cardinality dimensions (``tick.idx``, ``trace_id``, raw prompts) go
on spans, never on metrics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, Meter, UpDownCounter

from openral_observability import semconv

_T = TypeVar("_T")

__all__ = [
    "get_hal_estop_count",
    "get_hal_read_state_duration",
    "get_hal_send_action_duration",
    "get_inference_duration",
    "get_inference_timeouts",
    "get_meter",
    "get_observability_export_failures",
    "get_safety_clamps",
    "get_safety_violations",
    "get_sensors_age_ms",
    "get_sensors_stale_reads",
    "get_sim_episode_count",
    "get_sim_episode_success",
    "get_system_cpu_util_pct",
    "get_system_gpu_memory_total_mb",
    "get_system_gpu_memory_used_mb",
    "get_system_gpu_util_pct",
    "get_system_ram_total_mb",
    "get_system_ram_used_mb",
    "get_tick_budget_violations",
    "get_tick_deadline_misses",
    "get_tick_duration",
    "get_world_state_components_stale",
    "get_world_state_staleness_ms",
]

_METER_NAME = "openral"
_INSTRUMENT_CACHE: dict[tuple[int, str], Any] = {}


def get_meter() -> Meter:
    """Return the OpenRAL :class:`~opentelemetry.metrics.Meter`.

    Resolves against the currently installed :class:`MeterProvider`. When
    no provider has been installed, the API ships a no-op provider so the
    returned meter still produces working (silent) instruments. This makes
    it safe to call :func:`get_tick_duration().record(...)` even when
    :func:`configure_observability` was never called.
    """
    return metrics.get_meter(_METER_NAME)


def _cached(name: str, factory: Callable[[Meter, str], _T]) -> _T:
    """Return a cached instrument, creating it via ``factory(meter, name)`` if missing.

    The cache key includes ``id(meter)`` so a swapped MeterProvider (e.g.
    in tests) does not return stale instruments bound to the old meter.
    """
    meter = get_meter()
    key = (id(meter), name)
    cached = _INSTRUMENT_CACHE.get(key)
    if cached is None:
        cached = factory(meter, name)
        _INSTRUMENT_CACHE[key] = cached
    return cached


def _reset_instrument_cache() -> None:
    """Drop every cached instrument. Test-only — public sites should never call."""
    _INSTRUMENT_CACHE.clear()


# ── Histograms (ms) ────────────────────────────────────────────────────────


def get_tick_duration() -> Histogram:
    """``openral.tick.duration`` — runner tick latency histogram, unit ``ms``.

    Labels: ``skill.id``, ``skill.revision`` (closed sets per design §9).
    """
    return _cached(
        semconv.METRIC_TICK_DURATION,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="End-to-end runner tick latency in milliseconds.",
        ),
    )


def get_inference_duration() -> Histogram:
    """``openral.inference.duration`` — VLA chunk inference latency, unit ``ms``."""
    return _cached(
        semconv.METRIC_INFERENCE_DURATION,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="Per-chunk VLA inference latency in milliseconds.",
        ),
    )


def get_hal_read_state_duration() -> Histogram:
    """``openral.hal.read_state.duration`` — HAL state-read latency, unit ``ms``."""
    return _cached(
        semconv.METRIC_HAL_READ_STATE_DURATION,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="HAL.read_state() latency in milliseconds.",
        ),
    )


def get_hal_send_action_duration() -> Histogram:
    """``openral.hal.send_action.duration`` — HAL action-write latency, unit ``ms``."""
    return _cached(
        semconv.METRIC_HAL_SEND_ACTION_DURATION,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="HAL.send_action() latency in milliseconds.",
        ),
    )


def get_sensors_age_ms() -> Histogram:
    """``openral.sensors.age_ms`` — sensor freshness histogram, unit ``ms``."""
    return _cached(
        semconv.METRIC_SENSORS_AGE_MS,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="Sensor sample age at read time in milliseconds.",
        ),
    )


def get_world_state_staleness_ms() -> Histogram:
    """``openral.world_state.staleness_ms`` — world-state component freshness, unit ``ms``."""
    return _cached(
        semconv.METRIC_WORLD_STATE_STALENESS_MS,
        lambda meter, name: meter.create_histogram(
            name=name,
            unit="ms",
            description="Per-component world-state staleness at snapshot in milliseconds.",
        ),
    )


# ── Counters ───────────────────────────────────────────────────────────────


def get_tick_budget_violations() -> Counter:
    """``openral.tick.budget_violations`` — ticks that exceeded their latency budget."""
    return _cached(
        semconv.METRIC_TICK_BUDGET_VIOLATIONS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Ticks whose tick_ms exceeded the runner's latency budget.",
        ),
    )


def get_tick_deadline_misses() -> Counter:
    """``openral.tick.deadline_misses`` — ticks that overran the period."""
    return _cached(
        semconv.METRIC_TICK_DEADLINE_MISSES,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Ticks that overran the runner cadence period.",
        ),
    )


def get_inference_timeouts() -> Counter:
    """``openral.inference.timeouts`` — ``ROSInferenceTimeout`` occurrences."""
    return _cached(
        semconv.METRIC_INFERENCE_TIMEOUTS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="ROSInferenceTimeout occurrences on the inference path.",
        ),
    )


def get_safety_violations() -> Counter:
    """``openral.safety.violations`` — :class:`ROSSafetyViolation` family counter.

    Labels: ``check_name``, ``severity`` (both closed sets).
    """
    return _cached(
        semconv.METRIC_SAFETY_VIOLATIONS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="ROSSafetyViolation family occurrences.",
        ),
    )


def get_safety_clamps() -> Counter:
    """``openral.safety.clamps`` — safety-driven action clamps (no violation raised)."""
    return _cached(
        semconv.METRIC_SAFETY_CLAMPS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Safety-driven action clamps that did not raise a violation.",
        ),
    )


def get_hal_estop_count() -> Counter:
    """``openral.hal.estop.count`` — :meth:`HAL.estop` invocations."""
    return _cached(
        semconv.METRIC_HAL_ESTOP_COUNT,
        lambda meter, name: meter.create_counter(
            name=name,
            description="HAL.estop() invocations.",
        ),
    )


def get_sensors_stale_reads() -> Counter:
    """``openral.sensors.stale_reads`` — sensor reads that exceeded their age budget."""
    return _cached(
        semconv.METRIC_SENSORS_STALE_READS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Sensor reads whose age exceeded the configured budget.",
        ),
    )


def get_sim_episode_count() -> Counter:
    """``openral.sim.episode.count`` — episodes completed (any outcome)."""
    return _cached(
        semconv.METRIC_SIM_EPISODE_COUNT,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Sim episodes that ran to completion (terminated or truncated).",
        ),
    )


def get_sim_episode_success() -> Counter:
    """``openral.sim.episode.success`` — episodes that reached the success key."""
    return _cached(
        semconv.METRIC_SIM_EPISODE_SUCCESS,
        lambda meter, name: meter.create_counter(
            name=name,
            description="Sim episodes that hit task.success_key at least once.",
        ),
    )


def get_observability_export_failures() -> Counter:
    """``openral.observability.export_failures`` — dropped OTLP batches.

    Labels: ``signal_kind`` (``trace`` | ``metric`` | ``log``).
    """
    return _cached(
        semconv.METRIC_OBSERVABILITY_EXPORT_FAILURES,
        lambda meter, name: meter.create_counter(
            name=name,
            description="OTLP export attempts that failed and were dropped.",
        ),
    )


# ── UpDownCounters ─────────────────────────────────────────────────────────


def get_world_state_components_stale() -> UpDownCounter:
    """``openral.world_state.components_stale`` — current count of stale components."""
    return _cached(
        semconv.METRIC_WORLD_STATE_COMPONENTS_STALE,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            description="Number of world-state components currently latched stale.",
        ),
    )


# ── System-health gauges ───────────────────────────────────────────────────


def get_system_gpu_memory_used_mb() -> UpDownCounter:
    """``openral.system.gpu.memory_used_mb`` — GPU memory in use (MB)."""
    return _cached(
        semconv.METRIC_SYSTEM_GPU_MEMORY_USED_MB,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="MBy",
            description="GPU memory currently allocated.",
        ),
    )


def get_system_gpu_memory_total_mb() -> UpDownCounter:
    """``openral.system.gpu.memory_total_mb`` — total GPU memory (MB)."""
    return _cached(
        semconv.METRIC_SYSTEM_GPU_MEMORY_TOTAL_MB,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="MBy",
            description="GPU memory installed.",
        ),
    )


def get_system_gpu_util_pct() -> UpDownCounter:
    """``openral.system.gpu.utilization_pct`` — GPU SM utilisation (%)."""
    return _cached(
        semconv.METRIC_SYSTEM_GPU_UTIL_PCT,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="%",
            description="GPU SM utilisation.",
        ),
    )


def get_system_cpu_util_pct() -> UpDownCounter:
    """``openral.system.cpu.utilization_pct`` — CPU utilisation (%)."""
    return _cached(
        semconv.METRIC_SYSTEM_CPU_UTIL_PCT,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="%",
            description="Aggregate CPU utilisation.",
        ),
    )


def get_system_ram_used_mb() -> UpDownCounter:
    """``openral.system.ram.used_mb`` — system RAM in use (MB)."""
    return _cached(
        semconv.METRIC_SYSTEM_RAM_USED_MB,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="MBy",
            description="System RAM in use.",
        ),
    )


def get_system_ram_total_mb() -> UpDownCounter:
    """``openral.system.ram.total_mb`` — system RAM installed (MB)."""
    return _cached(
        semconv.METRIC_SYSTEM_RAM_TOTAL_MB,
        lambda meter, name: meter.create_up_down_counter(
            name=name,
            unit="MBy",
            description="System RAM installed.",
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def record_histogram_ms(
    instrument: Histogram, value_ms: float, attributes: Mapping[str, Any] | None = None
) -> None:
    """Record a millisecond value on a histogram, skipping ``NaN`` and negatives.

    Cheap guard against the common "I forgot to start the timer" bug where
    the runner reports ``inference_ms = -timer`` on an aborted tick. The
    OTel SDK accepts negatives but Prometheus does not, and the resulting
    histogram is useless.
    """
    if value_ms < 0 or math.isnan(value_ms):
        return
    instrument.record(value_ms, attributes=attributes)
