"""The dashboard's ``e-stops`` counter must be able to leave zero.

``index.html`` renders a Command-band ``e-stops`` tile from the
``openral.event.estop_requested`` counter, but nothing in the tree emitted
that event — so the tile read **0 no matter how many e-stops fired**. A
safety indicator that cannot leave zero is worse than an absent one: it
reads as an affirmative "no e-stops have occurred".

The emitter is now ``openral_hal.lifecycle._emit_estop_telemetry``, on the
shared HAL lifecycle node's ``/openral/estop`` callback. That node needs
rclpy, which CI's cheap env does not have, so these tests cover the half
that broke: the store's counter path, driven by the exact span shape the
HAL emits.

``python/observability/tests/test_declared_not_emitted.py`` covers the
other half — that an emitter exists at all, and keeps existing.

Real protobuf spans, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import time

from openral_observability import semconv
from openral_observability.dashboard import TelemetryStore
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
    Status,
)

_COUNTER = "openral.event.estop_requested"


def _attrs(d: dict[str, object]) -> list[KeyValue]:
    return [KeyValue(key=k, value=AnyValue(string_value=str(v))) for k, v in d.items()]


def _estop_span(*, adapter: str = "SO100FollowerHAL") -> Span:
    """The span shape ``_emit_estop_telemetry`` produces on a bare callback.

    A transient ``hal.estop`` span carrying one ``estop_requested`` event —
    there is no active span inside a ROS subscription callback.
    """
    start = time.time_ns()
    return Span(
        trace_id=b"\x0e" * 16,
        span_id=b"\x0e" * 8,
        name="hal.estop",
        start_time_unix_nano=start,
        end_time_unix_nano=start + 200_000,
        status=Status(code=0),
        events=[
            Span.Event(
                name=semconv.EVENT_ESTOP_REQUESTED,
                time_unix_nano=start,
                attributes=_attrs({semconv.HAL_ADAPTER: adapter}),
            )
        ],
    )


def _wrap(*spans: Span) -> list[ResourceSpans]:
    return [
        ResourceSpans(
            resource=Resource(attributes=_attrs({"service.name": "openral.hal.so101_follower"})),
            scope_spans=[ScopeSpans(spans=list(spans))],
        )
    ]


def test_an_estop_increments_the_counter_the_ui_reads() -> None:
    """The regression: this counter was structurally pinned to 0."""
    store = TelemetryStore()

    store.ingest_spans(_wrap(_estop_span()))

    assert store.snapshot()["counters"].get(_COUNTER) == 1


def test_the_counter_accumulates_across_estops() -> None:
    store = TelemetryStore()

    store.ingest_spans(_wrap(_estop_span(), _estop_span()))
    store.ingest_spans(_wrap(_estop_span()))

    assert store.snapshot()["counters"].get(_COUNTER) == 3


def test_the_estop_reaches_the_event_log_with_its_adapter() -> None:
    """An operator needs to see WHICH robot stopped, not just that one did."""
    store = TelemetryStore()

    store.ingest_spans(_wrap(_estop_span(adapter="SO100FollowerHAL")))

    events = [e for e in store.snapshot()["events"] if e["kind"] == semconv.EVENT_ESTOP_REQUESTED]
    assert events, "the e-stop must leave a row in the Event Log"
    assert events[0]["attrs"][semconv.HAL_ADAPTER] == "SO100FollowerHAL"
    assert events[0]["severity"] == "error"


def test_the_estop_row_survives_a_flood_of_routine_traffic() -> None:
    """An e-stop must still be findable seconds later.

    The main event ring is 200 entries and the per-tick span stream cycles
    it quickly. ``estop_requested`` is an error event, so it is mirrored
    into the protected lane — the whole reason that lane exists. This is
    the property that makes the counter *useful* rather than merely
    non-zero.
    """
    store = TelemetryStore()
    store.ingest_spans(_wrap(_estop_span()))

    flood = [
        Span(
            trace_id=b"\x11" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=time.time_ns(),
            end_time_unix_nano=time.time_ns() + 1000,
            status=Status(code=0),
        )
        for i in range(400)  # twice the main ring
    ]
    store.ingest_spans(_wrap(*flood))

    kinds = [e["kind"] for e in store.snapshot()["events"]]
    assert semconv.EVENT_ESTOP_REQUESTED in kinds, (
        "the e-stop was evicted by routine traffic — the operator would never find it"
    )


def test_no_estop_leaves_the_counter_absent() -> None:
    """Sanity: the counter is not incremented by unrelated traffic."""
    store = TelemetryStore()

    store.ingest_spans(
        _wrap(
            Span(
                trace_id=b"\x0f" * 16,
                span_id=b"\x0f" * 8,
                name="hal.read_state",
                start_time_unix_nano=time.time_ns(),
                end_time_unix_nano=time.time_ns() + 1000,
                status=Status(code=0),
            )
        )
    )

    assert store.snapshot()["counters"].get(_COUNTER, 0) == 0
