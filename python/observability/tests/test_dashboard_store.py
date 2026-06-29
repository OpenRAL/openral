"""Unit tests for :mod:`openral_observability.dashboard.store`.

Tests feed real ``ResourceSpans`` / ``ResourceMetrics`` protobuf
messages — built via ``opentelemetry-proto`` directly — into the
store and assert against its snapshot. No mocks per CLAUDE.md §1.11
and §5.4; the only test scaffolding is the protobuf builder, which
constructs the same wire format an OTLP exporter would have sent.
"""

from __future__ import annotations

import json
import time

import pytest
from openral_observability.dashboard import TelemetryStore
from openral_observability.dashboard.store import _log_level
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, InstrumentationScope, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import (
    LogRecord,
    ResourceLogs,
    ScopeLogs,
    SeverityNumber,
)
from opentelemetry.proto.metrics.v1.metrics_pb2 import (
    AggregationTemporality,
    HistogramDataPoint,
    Metric,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
)
from opentelemetry.proto.metrics.v1.metrics_pb2 import Histogram as HistogramProto
from opentelemetry.proto.metrics.v1.metrics_pb2 import Sum as SumProto
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
    Status,
)


def _av(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    return AnyValue(string_value=str(value))


def _attrs(d: dict[str, object]) -> list[KeyValue]:
    return [KeyValue(key=k, value=_av(v)) for k, v in d.items()]


def _resource(d: dict[str, object]) -> Resource:
    return Resource(attributes=_attrs(d))


def _make_span(
    name: str,
    *,
    duration_ms: float = 12.5,
    attrs: dict[str, object] | None = None,
    status_code: int = 0,
    events: list[tuple[str, dict[str, object]]] | None = None,
) -> Span:
    start = time.time_ns()
    end = start + int(duration_ms * 1_000_000)
    span_events = []
    if events:
        for ev_name, ev_attrs in events:
            span_events.append(
                Span.Event(
                    name=ev_name,
                    time_unix_nano=end,
                    attributes=_attrs(ev_attrs),
                )
            )
    return Span(
        trace_id=b"\x01" * 16,
        span_id=b"\x01" * 8,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        attributes=_attrs(attrs or {}),
        status=Status(code=status_code),
        events=span_events,
    )


def _wrap_spans(
    spans: list[Span],
    resource_attrs: dict[str, object] | None = None,
) -> list[ResourceSpans]:
    return [
        ResourceSpans(
            resource=_resource(resource_attrs or {"service.name": "ral"}),
            scope_spans=[ScopeSpans(spans=spans)],
        )
    ]


def _make_log(
    body: str,
    *,
    severity_number: int = SeverityNumber.SEVERITY_NUMBER_INFO,
    severity_text: str = "",
    attrs: dict[str, object] | None = None,
) -> LogRecord:
    return LogRecord(
        time_unix_nano=time.time_ns(),
        severity_number=severity_number,
        severity_text=severity_text,
        body=_av(body),
        attributes=_attrs(attrs or {}),
    )


def _wrap_logs(
    records: list[LogRecord],
    *,
    scope_name: str = "openral.otel_bridge",
    resource_attrs: dict[str, object] | None = None,
) -> list[ResourceLogs]:
    return [
        ResourceLogs(
            resource=_resource(resource_attrs or {"service.name": "ral"}),
            scope_logs=[
                ScopeLogs(
                    scope=InstrumentationScope(name=scope_name),
                    log_records=records,
                )
            ],
        )
    ]


def test_ingest_rskill_execute_populates_headline_card() -> None:
    store = TelemetryStore()
    span = _make_span(
        "rskill.execute",
        duration_ms=21.4,
        attrs={
            "rskill.id": "smolvla-libero",
            "rskill.role": "s1",
            "openral.tick.idx": 42,
        },
    )
    recorded = store.ingest_spans(_wrap_spans([span]))
    assert recorded == 1

    snap = store.snapshot()
    assert snap["service_name"] == "ral"
    card = snap["cards"]["rskill_execute"]
    assert card["name"] == "rskill.execute"
    assert card["attrs"]["rskill.id"] == "smolvla-libero"
    assert card["duration_ms"] == 21.4
    assert any(ev["kind"] == "rskill.execute" for ev in snap["events"])


def test_run_mode_and_run_id_propagate_from_resource() -> None:
    store = TelemetryStore()
    span = _make_span("rskill.tick")
    store.ingest_spans(
        _wrap_spans(
            [span],
            resource_attrs={
                "service.name": "ral-sim",
                "openral.run.id": "deadbeef-1234",
                "openral.run.mode": "sim",
            },
        )
    )
    snap = store.snapshot()
    assert snap["service_name"] == "ral-sim"
    assert snap["run_id"] == "deadbeef-1234"
    assert snap["run_mode"] == "sim"


def test_span_event_counters_increment() -> None:
    store = TelemetryStore()
    span = _make_span(
        "safety.check",
        attrs={"safety.check_name": "ee_speed", "safety.severity": "warn"},
        events=[
            ("openral.event.safety_violation", {"reason": "limit"}),
            ("openral.event.deadline_missed", {}),
        ],
    )
    store.ingest_spans(_wrap_spans([span]))
    snap = store.snapshot()
    assert snap["counters"]["openral.event.safety_violation"] == 1
    assert snap["counters"]["openral.event.deadline_missed"] == 1
    severities = {ev["severity"] for ev in snap["events"]}
    assert "error" in severities  # safety_violation


def test_skill_failure_event_counts_and_carries_state() -> None:
    """ADR-0074/0077 — a Reasoner-published skill failure (mirrored onto the
    span path by ``_publish_skill_failure``) tallies on its own counter, lands
    on the event log at ``error`` severity, and carries the failure state so the
    dashboard can show *why* a skill failed (e.g. ``vram_insufficient``)."""
    store = TelemetryStore()
    span = _make_span(
        "reasoner.skill_failure",
        events=[
            (
                "openral.event.skill_failure",
                {
                    "openral.event.skill_failure.state": "vram_insufficient",
                    "reasoner.rskill_id": "pi05-libero-nf4",
                },
            ),
        ],
    )
    store.ingest_spans(_wrap_spans([span]))
    snap = store.snapshot()
    assert snap["counters"]["openral.event.skill_failure"] == 1
    failure = next(
        ev for ev in snap["events"] if ev["kind"] == "openral.event.skill_failure"
    )
    assert failure["severity"] == "error"
    assert failure["attrs"]["openral.event.skill_failure.state"] == "vram_insufficient"


def test_error_status_propagates_to_card_severity() -> None:
    store = TelemetryStore()
    span = _make_span("rskill.execute", status_code=2, attrs={"rskill.id": "x"})
    store.ingest_spans(_wrap_spans([span]))
    snap = store.snapshot()
    assert snap["cards"]["rskill_execute"]["status_code"] == 2
    err_events = [ev for ev in snap["events"] if ev["severity"] == "error"]
    assert any(ev["kind"] == "rskill.execute" for ev in err_events)


def test_histogram_metric_records_samples_and_percentiles() -> None:
    store = TelemetryStore()
    rm = ResourceMetrics(
        resource=_resource({"service.name": "ral"}),
        scope_metrics=[
            ScopeMetrics(
                metrics=[
                    Metric(
                        name="openral.tick.duration",
                        unit="ms",
                        histogram=HistogramProto(
                            aggregation_temporality=AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE,
                            data_points=[
                                HistogramDataPoint(
                                    count=10,
                                    sum=100.0,  # avg = 10.0
                                    attributes=_attrs({"skill.id": "smolvla"}),
                                )
                            ],
                        ),
                    )
                ]
            )
        ],
    )
    recorded = store.ingest_metrics([rm])
    assert recorded == 1
    # Two exports with different averages so percentiles are meaningful.
    rm.scope_metrics[0].metrics[0].histogram.data_points[0].sum = 500.0  # avg=50
    store.ingest_metrics([rm])
    rm.scope_metrics[0].metrics[0].histogram.data_points[0].sum = 50.0  # avg=5
    store.ingest_metrics([rm])

    snap = store.snapshot()
    series = next(m for m in snap["metrics"] if m["name"] == "openral.tick.duration")
    assert series["kind"] == "histogram"
    assert series["unit"] == "ms"
    assert series["labels"] == {"skill.id": "smolvla"}
    assert len(series["samples"]) == 3
    assert series["p50"] == 10.0  # median of [5, 10, 50]
    assert series["p95"] >= 10.0


def test_sum_metric_tracks_cumulative() -> None:
    store = TelemetryStore()
    rm = ResourceMetrics(
        resource=_resource({"service.name": "ral"}),
        scope_metrics=[
            ScopeMetrics(
                metrics=[
                    Metric(
                        name="openral.safety.violations",
                        unit="1",
                        sum=SumProto(
                            aggregation_temporality=AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE,
                            is_monotonic=True,
                            data_points=[
                                NumberDataPoint(
                                    as_int=7, attributes=_attrs({"check_name": "ee_speed"})
                                )
                            ],
                        ),
                    )
                ]
            )
        ],
    )
    store.ingest_metrics([rm])
    snap = store.snapshot()
    series = next(m for m in snap["metrics"] if m["name"] == "openral.safety.violations")
    assert series["kind"] == "sum"
    assert series["cumulative"] == 7.0


def test_event_ring_is_bounded() -> None:
    store = TelemetryStore()
    spans = [_make_span("rskill.tick", attrs={"i": i}) for i in range(500)]
    store.ingest_spans(_wrap_spans(spans))
    snap = store.snapshot()
    assert len(snap["events"]) == 200  # _EVENT_RING_SIZE


def test_world_scene_objects_span_populates_topic() -> None:
    """``world.scene_objects`` (ADR-0038) → decoded objects in the topic bucket."""
    store = TelemetryStore()
    objects = [
        {
            "id": "obj_track_1",
            "label": "wine_bottle",
            "x": 3.0,
            "y": 1.0,
            "z": 0.9,
            "frame_id": "map",
            "confidence": 0.91,
            "last_seen_ns": 1_700_000_000_000_000_000,
            "observation_count": 4,
            "is_container": False,
        },
    ]
    span = _make_span(
        "world.scene_objects",
        attrs={
            "openral.world_state.scene_objects.count": 1,
            "openral.world_state.scene_objects.frame_id": "map",
            "openral.world_state.scene_objects.source_node": "openral_reasoner",
            "openral.world_state.scene_objects.list": json.dumps(objects),
        },
    )
    store.ingest_spans(_wrap_spans([span]))

    topic = store.snapshot()["topics"]["scene_objects"]
    assert topic["count"] == 1
    assert topic["frame_id"] == "map"
    assert topic["source_node"] == "openral_reasoner"
    assert topic["objects"] == objects


def test_world_scene_objects_malformed_list_degrades_to_empty() -> None:
    """A malformed ``list`` attr yields ``[]`` rather than crashing the receiver."""
    store = TelemetryStore()
    span = _make_span(
        "world.scene_objects",
        attrs={
            "openral.world_state.scene_objects.count": 0,
            "openral.world_state.scene_objects.list": "{not json",
        },
    )
    store.ingest_spans(_wrap_spans([span]))
    assert store.snapshot()["topics"]["scene_objects"]["objects"] == []


# ─────────────────────────── log bridge (issue #318) ──────────────────────────


@pytest.mark.parametrize(
    ("severity_number", "expected"),
    [
        (SeverityNumber.SEVERITY_NUMBER_TRACE, "debug"),  # 1-4 floor to debug
        (SeverityNumber.SEVERITY_NUMBER_DEBUG, "debug"),
        (SeverityNumber.SEVERITY_NUMBER_DEBUG4, "debug"),  # top of the DEBUG band
        (SeverityNumber.SEVERITY_NUMBER_INFO, "info"),
        (SeverityNumber.SEVERITY_NUMBER_INFO3, "info"),
        (SeverityNumber.SEVERITY_NUMBER_WARN, "warn"),
        (SeverityNumber.SEVERITY_NUMBER_ERROR, "error"),
        (SeverityNumber.SEVERITY_NUMBER_FATAL, "fatal"),
        (SeverityNumber.SEVERITY_NUMBER_FATAL4, "fatal"),
    ],
)
def test_log_level_maps_severity_number_to_bucket(severity_number: int, expected: str) -> None:
    assert _log_level(severity_number, "") == expected


def test_log_level_unspecified_falls_back_to_text() -> None:
    """An unset severity_number (0) buckets via severity_text, default info."""
    unspecified = SeverityNumber.SEVERITY_NUMBER_UNSPECIFIED
    assert _log_level(unspecified, "warning") == "warn"
    assert _log_level(unspecified, "critical") == "fatal"
    assert _log_level(unspecified, "") == "info"
    assert _log_level(unspecified, "nonsense") == "info"


def test_ingest_logs_appends_debug_event() -> None:
    """A bridged DEBUG log record becomes a debug-severity event (issue #318)."""
    store = TelemetryStore()
    recorded = store.ingest_logs(
        _wrap_logs(
            [
                _make_log(
                    "world_state.detected_objects count=0",
                    severity_number=SeverityNumber.SEVERITY_NUMBER_DEBUG,
                    attrs={"count": 0},
                )
            ],
            scope_name="openral.world_state",
        )
    )
    assert recorded == 1
    snap = store.snapshot()
    debug_events = [ev for ev in snap["events"] if ev["severity"] == "debug"]
    assert len(debug_events) == 1
    ev = debug_events[0]
    assert ev["kind"] == "openral.world_state"
    assert ev["title"] == "world_state.detected_objects count=0"
    assert ev["attrs"]["count"] == 0


def test_ingest_logs_maps_all_levels() -> None:
    """info/warn/error log lines land alongside today's trace events."""
    store = TelemetryStore()
    store.ingest_logs(
        _wrap_logs(
            [
                _make_log("hello", severity_number=SeverityNumber.SEVERITY_NUMBER_INFO),
                _make_log("careful", severity_number=SeverityNumber.SEVERITY_NUMBER_WARN),
                _make_log("boom", severity_number=SeverityNumber.SEVERITY_NUMBER_ERROR),
            ]
        )
    )
    snap = store.snapshot()
    severities = {ev["title"]: ev["severity"] for ev in snap["events"]}
    assert severities["hello"] == "info"
    assert severities["careful"] == "warn"
    assert severities["boom"] == "error"
    # service.name from the ResourceLogs resource propagates like spans/metrics.
    assert "ral" in snap["services"]


def test_ingest_logs_shares_the_bounded_event_ring() -> None:
    """Log-derived events obey the same 200-event cap as spans."""
    store = TelemetryStore()
    store.ingest_logs(
        _wrap_logs(
            [
                _make_log(f"line {i}", severity_number=SeverityNumber.SEVERITY_NUMBER_DEBUG)
                for i in range(500)
            ]
        )
    )
    assert len(store.snapshot()["events"]) == 200  # _EVENT_RING_SIZE
