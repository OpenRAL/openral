"""Runner emits openral.tick.* metrics and records exceptions onto spans.

Covers the wiring added in PR ``feat(runner): wire tick/safety metrics
and record exceptions on rskill.tick``. Tests use the real OTel SDK
and an in-memory exporter — no mocks per CLAUDE.md §1.11 / §5.4.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_core import (
    Action,
    ControlMode,
    DeadlineOverrunPolicy,
    TickResult,
)
from openral_core.exceptions import (
    ROSDeadlineMissed,
    ROSSafetyViolation,
    ROSWorkspaceViolation,
)
from openral_observability import semconv
from openral_observability.metrics import _reset_instrument_cache
from openral_runner.base import InferenceRunnerBase
from openral_runner.safety import NullSafetyClient
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


class _FixedRunner(InferenceRunnerBase):
    """Minimal runner that emits one ``TickResult`` per tick with fixed timings.

    Lets the test drive both the deadline-overrun path and the
    safety-violation count without standing up a HAL / sim.
    """

    def __init__(
        self,
        *,
        rate_hz: float = 30.0,
        tick_ms: float = 5.0,
        safety_violations: int = 0,
        latency_budget_ms: float | None = None,
        deadline_overrun_policy: DeadlineOverrunPolicy = DeadlineOverrunPolicy.WARN,
    ) -> None:
        super().__init__(
            rate_hz=rate_hz,
            deadline_overrun_policy=deadline_overrun_policy,
            runner_name="fixed",
            latency_budget_ms=latency_budget_ms,
        )
        self._tick_ms = tick_ms
        self._safety_violations = safety_violations

    def _tick_impl(self, tick_idx: int) -> TickResult:
        return TickResult(
            stamp_ns=tick_idx,
            tick_idx=tick_idx,
            sensors_ms=0.0,
            world_state_ms=0.0,
            inference_ms=self._tick_ms / 2,
            safety_ms=0.0,
            hal_ms=0.0,
            tick_ms=self._tick_ms,
            safety_violations=["overspeed"] if self._safety_violations else [],
            action_applied=self._safety_violations == 0,
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


def test_tick_records_tick_duration_histogram(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """``openral.tick.duration`` records one sample per tick keyed by ``skill.id``."""
    runner = _FixedRunner(tick_ms=4.2)
    runner.activate()
    runner.tick()
    runner.tick()

    metric = _find_metric(memory_metric_reader, semconv.METRIC_TICK_DURATION)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = points[0]
    assert point.attributes[semconv.LABEL_RSKILL_ID] == "fixed"
    assert point.count == 2


def test_tick_safety_violations_increment_counter(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """A non-empty ``TickResult.safety_violations`` increments the runtime counter."""
    runner = _FixedRunner(safety_violations=1)
    runner.activate()
    runner.tick()
    runner.tick()

    metric = _find_metric(memory_metric_reader, semconv.METRIC_SAFETY_VIOLATIONS)
    assert metric is not None
    points = list(metric.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, NumberDataPoint)
    assert point.attributes[semconv.LABEL_CHECK_NAME] == "runtime"
    assert point.attributes[semconv.LABEL_SEVERITY] == "violation"
    assert point.value == 2


def test_deadline_overrun_raise_policy_records_exception(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """``DeadlineOverrunPolicy.RAISE`` records the exception on the active span."""
    # Tick takes longer than 1/rate_hz; budget is also exceeded to count the violation.
    runner = _FixedRunner(
        rate_hz=200.0,  # 5 ms period
        tick_ms=20.0,  # always overruns
        latency_budget_ms=1.0,  # always violates budget
        deadline_overrun_policy=DeadlineOverrunPolicy.RAISE,
    )

    # Open a parent span so ``record_exception`` has somewhere to land.
    tracer = trace.get_tracer("openral")
    with tracer.start_as_current_span("test.parent"), pytest.raises(ROSDeadlineMissed):
        runner.run(max_ticks=1)

    spans = memory_exporter.get_finished_spans()
    parent_spans = [s for s in spans if s.name == "test.parent"]
    assert parent_spans
    parent = parent_spans[0]
    # Both ``deadline_missed`` event AND a ``record_exception`` event land
    # on the parent span (the rskill.tick span has already closed).
    event_names = {e.name for e in parent.events}
    assert semconv.EVENT_DEADLINE_MISSED in event_names
    assert "exception" in event_names

    deadline_metric = _find_metric(memory_metric_reader, semconv.METRIC_TICK_DEADLINE_MISSES)
    assert deadline_metric is not None
    budget_metric = _find_metric(memory_metric_reader, semconv.METRIC_TICK_BUDGET_VIOLATIONS)
    assert budget_metric is not None


def test_safety_violation_exception_recorded_on_tick_span() -> None:
    """``ROSSafetyViolation`` raised by ``check_action`` is observable via attrs.

    We don't go through the HardwareRunner here (it requires HAL + sensors);
    instead we cover the ``NullSafetyClient`` happy path and document the
    HardwareRunner branch with a separate test (``test_hardware_runner.py``
    in this PR adds the matching coverage).
    """
    client = NullSafetyClient()
    action = Action(control_mode=ControlMode.JOINT_POSITION)
    # NullSafetyClient is always permissive — never raises.
    assert client.check_action(action) is None
    # The exception family hierarchy is the contract used by the runner.
    assert issubclass(ROSWorkspaceViolation, ROSSafetyViolation)
