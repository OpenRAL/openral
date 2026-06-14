"""HardwareRunner emits ``hal.read_state`` / ``hal.send_action`` spans + metrics.

End-to-end against a real SO-100 digital twin (no mocks per CLAUDE.md
§1.11 / §5.4). The runner-side wrapping covers every HAL adapter
without touching adapter code; the in-process timing stays on the
``TickResult`` (used by the rate-limiter) while the spans/histograms
give Jaeger / Prom callers structured visibility.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

import pytest
from openral_core import Action, ControlMode
from openral_core.schemas import WorldState
from openral_hal.so100_follower import SO100FollowerHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_observability import semconv
from openral_observability.metrics import _reset_instrument_cache
from openral_rskill.base import rSkillBase
from openral_runner import HardwareRunner
from openral_world_state.aggregator import WorldStateAggregator
from opentelemetry import metrics, trace
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    HistogramDataPoint,
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class _NoOpTestSkill(rSkillBase):
    def __init__(self) -> None:
        super().__init__(name="hal_obs_skill", embodiment_tags=["so100_follower"])

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 6],
            confidence=1.0,
        )


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


@pytest.fixture
def memory_metric_reader() -> Iterator[InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]
    metrics.set_meter_provider(provider)
    _reset_instrument_cache()
    try:
        yield reader
    finally:
        provider.shutdown()
        _reset_instrument_cache()


@pytest.fixture
def active_skill() -> Generator[_NoOpTestSkill, None, None]:
    skill = _NoOpTestSkill()
    skill.configure()
    skill.activate()
    yield skill
    if skill.info.state.value == "active":
        skill.deactivate()
    if skill.info.state.value != "finalized":
        skill.shutdown()


def test_hardware_runner_emits_hal_spans_and_histograms(
    active_skill: _NoOpTestSkill,
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """One tick → one ``hal.read_state`` + one ``hal.send_action`` span + histograms."""
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(hal.description)
    runner = HardwareRunner(
        hal=hal,
        skill=active_skill,
        aggregator=aggregator,
        rate_hz=30.0,
    )

    runner.activate()
    try:
        runner.tick()
    finally:
        runner.deactivate()

    span_names = [s.name for s in memory_exporter.get_finished_spans()]
    assert semconv.SPAN_HAL_READ_STATE in span_names
    assert semconv.SPAN_HAL_SEND_ACTION in span_names

    # Confirm the spans carry the adapter label and tick idx.
    read_span = next(
        s for s in memory_exporter.get_finished_spans() if s.name == semconv.SPAN_HAL_READ_STATE
    )
    assert read_span.attributes is not None
    assert read_span.attributes[semconv.HAL_ADAPTER] == "so100followerhal"
    assert read_span.attributes[semconv.TICK_IDX] == 0

    send_span = next(
        s for s in memory_exporter.get_finished_spans() if s.name == semconv.SPAN_HAL_SEND_ACTION
    )
    assert send_span.attributes is not None
    assert send_span.attributes[semconv.HAL_CONTROL_MODE] == "joint_position"

    # Both histograms populated.
    data = memory_metric_reader.get_metrics_data()
    assert data is not None
    metric_names = {
        m.name
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for m in scope_metric.metrics
    }
    assert semconv.METRIC_HAL_READ_STATE_DURATION in metric_names
    assert semconv.METRIC_HAL_SEND_ACTION_DURATION in metric_names

    # The HAL histograms record a sample (count >= 1) keyed by adapter.
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == semconv.METRIC_HAL_READ_STATE_DURATION:
                    for point in metric.data.data_points:  # type: ignore[attr-defined]
                        assert isinstance(point, HistogramDataPoint)
                        assert point.attributes[semconv.LABEL_HAL_ADAPTER] == "so100followerhal"
                        assert point.count >= 1
