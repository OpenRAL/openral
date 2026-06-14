"""WorldStateAggregator.snapshot emits an OTel span + staleness telemetry.

Real OTel SDK + in-memory exporter (CLAUDE.md §1.11 / §5.4). The
``world_state.snapshot`` span gates the on-the-wire vocabulary defined
in design §4.3:

* attributes: ``openral.world_state.components_stale``,
  ``openral.world_state.has_latched_error``;
* events: ``openral.event.staleness_latched`` /
  ``openral.event.error_latched`` only on transition (first tick a
  component flips stale / acquires a forced error).
* metrics: ``openral.world_state.staleness_ms`` histogram per
  component, ``openral.world_state.components_stale`` up-down counter.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.schemas import JointState, SensorBundle, SensorModality, SensorSpec
from openral_observability import semconv
from openral_observability.metrics import _reset_instrument_cache
from openral_world_state import WorldStateAggregator
from opentelemetry import metrics, trace
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    HistogramDataPoint,
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
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


@pytest.fixture
def memory_metric_reader() -> Iterator[InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    metrics.set_meter_provider(provider)
    _reset_instrument_cache()
    try:
        yield reader
    finally:
        provider.shutdown()
        _reset_instrument_cache()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_description(sensor_names: list[str]) -> RobotDescription:
    return RobotDescription(
        name="test_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j0",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link_0",
            )
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
        sensor_bundles=[
            SensorBundle(
                bundle_name=f"{n}_bundle",
                sensors=[
                    SensorSpec(
                        name=n,
                        modality=SensorModality.RGB,
                        frame_id=f"{n}_frame",
                        rate_hz=30.0,
                        ros2_topic=f"/{n}/image_raw",
                        ros2_msg_type="sensor_msgs/Image",
                    )
                ],
            )
            for n in sensor_names
        ],
    )


class _FakeClock:
    """Monotonic int-ns clock driven by tests."""

    def __init__(self) -> None:
        self.now = 1_000_000_000  # 1 s in ns; arbitrary baseline

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += int(ms * 1e6)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_snapshot_emits_world_state_span(memory_exporter: InMemorySpanExporter) -> None:
    """A single ``world_state.snapshot`` span is emitted per call with attrs."""
    clock = _FakeClock()
    agg = WorldStateAggregator(_make_description(["cam0"]), clock_fn=clock)
    agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=clock.now))
    agg.update_image("cam0", "/cam0/image_raw", clock.now)

    agg.snapshot()

    spans = memory_exporter.get_finished_spans()
    assert [s.name for s in spans] == [semconv.SPAN_WORLD_STATE_SNAPSHOT]
    span = spans[0]
    assert span.attributes is not None
    # Fresh updates → no stale components, no latched error.
    assert span.attributes[semconv.WORLD_STATE_COMPONENTS_STALE] == 0
    assert span.attributes[semconv.WORLD_STATE_HAS_LATCHED_ERROR] is False


def test_first_stale_tick_emits_staleness_latched_event(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """``staleness_latched`` fires only on the tick a component first goes stale."""
    clock = _FakeClock()
    agg = WorldStateAggregator(
        _make_description(["cam0"]),
        clock_fn=clock,
        staleness_limit_s=0.05,
    )
    agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=clock.now))
    agg.update_image("cam0", "/cam0/image_raw", clock.now)

    # Tick 1: everything fresh.
    agg.snapshot()
    # Advance past staleness threshold; cam0 + joint_state both go stale.
    clock.advance(100.0)

    # Tick 2: transition.
    agg.snapshot()
    # Tick 3: still stale — no new event should fire.
    agg.snapshot()

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 3
    transition_events = [e for e in spans[1].events if e.name == semconv.EVENT_STALENESS_LATCHED]
    persisted_events = [e for e in spans[2].events if e.name == semconv.EVENT_STALENESS_LATCHED]
    transition_components = {e.attributes[semconv.WORLD_STATE_COMPONENT] for e in transition_events}
    assert transition_components == {"joint_state", "cam0"}
    assert persisted_events == []


def test_error_latched_event_fires_once(memory_exporter: InMemorySpanExporter) -> None:
    """``error_latched`` fires when ``set_error`` is called, not on every snapshot."""
    clock = _FakeClock()
    agg = WorldStateAggregator(_make_description([]), clock_fn=clock)
    agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=clock.now))

    agg.snapshot()  # no error
    agg.set_error("joint_state", "error")
    agg.snapshot()  # transition tick
    agg.snapshot()  # persists; no new event

    spans = memory_exporter.get_finished_spans()
    err_events_tick2 = [e for e in spans[1].events if e.name == semconv.EVENT_ERROR_LATCHED]
    err_events_tick3 = [e for e in spans[2].events if e.name == semconv.EVENT_ERROR_LATCHED]
    assert len(err_events_tick2) == 1
    assert err_events_tick2[0].attributes[semconv.WORLD_STATE_COMPONENT] == "joint_state"
    assert err_events_tick3 == []


def test_staleness_histogram_records_per_component(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """Each known component contributes a per-``component`` histogram sample."""
    clock = _FakeClock()
    agg = WorldStateAggregator(_make_description(["cam0"]), clock_fn=clock)
    agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=clock.now))
    agg.update_image("cam0", "/cam0/image_raw", clock.now)

    clock.advance(10.0)
    agg.snapshot()

    data = memory_metric_reader.get_metrics_data()
    assert data is not None
    hist = None
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == semconv.METRIC_WORLD_STATE_STALENESS_MS:
                    hist = metric
    assert hist is not None
    components = {
        point.attributes[semconv.LABEL_COMPONENT]
        for point in hist.data.data_points  # type: ignore[attr-defined]
        if isinstance(point, HistogramDataPoint)
    }
    assert components == {"joint_state", "cam0"}


def test_components_stale_up_down_counter(
    memory_exporter: InMemorySpanExporter,
    memory_metric_reader: InMemoryMetricReader,
) -> None:
    """The UpDownCounter tracks the current stale-set size."""
    clock = _FakeClock()
    agg = WorldStateAggregator(
        _make_description(["cam0"]),
        clock_fn=clock,
        staleness_limit_s=0.05,
    )
    agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=clock.now))
    agg.update_image("cam0", "/cam0/image_raw", clock.now)

    agg.snapshot()  # 0 stale
    clock.advance(100.0)
    agg.snapshot()  # 2 stale (joint_state + cam0): delta +2
    # Refresh one of them.
    agg.update_image("cam0", "/cam0/image_raw", clock.now)
    agg.snapshot()  # 1 stale (only joint_state): delta -1

    data = memory_metric_reader.get_metrics_data()
    assert data is not None
    udc = None
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == semconv.METRIC_WORLD_STATE_COMPONENTS_STALE:
                    udc = metric
    assert udc is not None
    points = list(udc.data.data_points)  # type: ignore[attr-defined]
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, NumberDataPoint)
    assert point.value == 1
