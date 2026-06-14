"""SimRunner emits physics.step child spans + episode counters.

Real ``mock`` SimEnvironment (no mocks per CLAUDE.md §1.11) driven
through 1–N episodes; asserts the new telemetry surface added by
``feat(sim): physics.step child span + episode counters``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_core import (
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_observability import semconv
from openral_observability.metrics import _reset_instrument_cache
from openral_sim import SimRunner
from opentelemetry import metrics, trace
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    NumberDataPoint,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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


def _mock_env(*, n_episodes: int = 1, success_step: int = 2, max_steps: int = 5) -> SimEnvironment:
    return SimEnvironment(
        robot_id="so100_follower",
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": success_step, "action_dim": 7},
        ),
        task=TaskSpec(
            id="mock/0",
            scene_id="mock",
            instruction="noop",
            max_steps=max_steps,
            success_key="is_success",
        ),
        vla=VLASpec(
            id="zero",
            weights_uri="placeholder",
            extra={"action_dim": 7, "seed": 0},
        ),
        n_episodes=n_episodes,
    )


def test_physics_step_span_emitted_per_step_tick(memory_exporter: InMemorySpanExporter) -> None:
    """Each step-tick carries a ``physics.step`` child span with step_ms attr."""
    runner = SimRunner(_mock_env(success_step=2))
    runner.activate()
    try:
        runner.tick()  # reset
        runner.tick()  # step 1 — physics.step emitted
        runner.tick()  # step 2 — physics.step emitted (success)
    finally:
        runner.deactivate()

    physics_spans = [
        s for s in memory_exporter.get_finished_spans() if s.name == semconv.SPAN_PHYSICS_STEP
    ]
    assert len(physics_spans) >= 2
    for span in physics_spans:
        assert span.attributes is not None
        assert "physics.step_ms" in span.attributes
        assert isinstance(span.attributes["physics.step_ms"], float)


def test_episode_counters_increment_on_finalize(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """``run()`` over N episodes increments count and success by N."""
    env_cfg = _mock_env(n_episodes=3, success_step=2)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=1000)
    finally:
        runner.deactivate()

    data = memory_metric_reader.get_metrics_data()
    assert data is not None

    by_name: dict[str, object] = {}
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                by_name[metric.name] = metric

    count_metric = by_name.get(semconv.METRIC_SIM_EPISODE_COUNT)
    success_metric = by_name.get(semconv.METRIC_SIM_EPISODE_SUCCESS)
    assert count_metric is not None
    assert success_metric is not None

    count_points = list(count_metric.data.data_points)  # type: ignore[attr-defined]
    success_points = list(success_metric.data.data_points)  # type: ignore[attr-defined]
    # Single label set (closed-set scene/task/vla ids), so one point.
    assert len(count_points) == 1
    assert isinstance(count_points[0], NumberDataPoint)
    assert count_points[0].value == 3
    assert count_points[0].attributes["scene.id"] == "mock"
    assert count_points[0].attributes["task.id"] == "mock/0"
    assert count_points[0].attributes["vla.id"] == "zero"

    # All three mock episodes succeed because success_step=2 < max_steps=5.
    assert len(success_points) == 1
    assert isinstance(success_points[0], NumberDataPoint)
    assert success_points[0].value == 3
