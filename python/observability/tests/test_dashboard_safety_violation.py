"""Safety-violation surfacing in the dashboard store.

A ``safety.check`` span whose ``safety.severity == "violation"`` (the C++
kernel dropping an action — self-collision / envelope breach) must reach the
operator on two durable surfaces, because the raw per-span event is severity
``info`` (the kernel span's own status is OK — dropping the action IS the
kernel working) and the 30 Hz ``hal.read_state`` stream evicts it from the
200-slot event ring within seconds:

1. ``topics.safety.last_violation`` — a persistent slot only the next
   violation overwrites (the per-check ledger row resets on the next OK check).
2. a dedicated ``safety.violation`` error-severity event + the
   ``openral.event.safety_violation`` counter the UI's Safety tile reads.

Real protobuf spans, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import time

from openral_observability.dashboard import TelemetryStore
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
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


def _safety_span(attrs: dict[str, object]) -> Span:
    start = time.time_ns()
    return Span(
        trace_id=b"\x09" * 16,
        span_id=b"\x09" * 8,
        name="safety.check",
        start_time_unix_nano=start,
        end_time_unix_nano=start + 300_000,  # 0.3 ms — matches the kernel span
        attributes=_attrs(attrs),
        status=Status(code=0),  # kernel span is OK even on a violation
    )


def _wrap(*spans: Span) -> list[ResourceSpans]:
    return [
        ResourceSpans(
            resource=Resource(attributes=_attrs({"service.name": "openral_safety_kernel"})),
            scope_spans=[ScopeSpans(spans=list(spans))],
        )
    ]


_VIOLATION_ATTRS = {
    "safety.check_name": "envelope",
    "safety.kernel": "cpp",
    "safety.severity": "violation",
    "safety.drop_reason": "collision",
    "safety.collision_mode": 0,
    "safety.violation_value": -0.0841,
    "rskill.id": "OpenRAL/rskill-smolvla-so101-pen-bf16",
}


def test_violation_populates_persistent_slot() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    lv = store.snapshot()["topics"]["safety"]["last_violation"]
    assert lv["drop_reason"] == "collision"
    assert lv["violation_value"] == -0.0841
    assert lv["rskill_id"] == "OpenRAL/rskill-smolvla-so101-pen-bf16"


def test_violation_emits_error_event_and_counter() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    snap = store.snapshot()

    violations = [e for e in snap["events"] if e["kind"] == "safety.violation"]
    assert len(violations) == 1
    assert violations[0]["severity"] == "error"
    assert "collision" in violations[0]["title"]
    # The counter the UI's Safety tile reads (cnt-safety).
    assert snap["counters"]["openral.event.safety_violation"] == 1


def test_persistent_slot_survives_a_high_rate_span_flood() -> None:
    """The last_violation slot outlives the event-ring eviction the flood causes."""
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))

    # Flood the 200-slot event ring with unrelated high-rate spans — exactly
    # what evicts the raw per-span event on a live deploy.
    start = time.time_ns()
    flood = [
        Span(
            trace_id=b"\x01" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=start,
            end_time_unix_nano=start + 1_000_000,
            attributes=_attrs({"openral.hal.adapter": "so100"}),
            status=Status(code=0),
        )
        for i in range(300)
    ]
    store.ingest_spans(_wrap(*flood))

    snap = store.snapshot()
    # The safety.violation event now SURVIVES the flood via the protected error
    # lane (it used to be evicted from the shared 200-slot ring within seconds,
    # leaving no trace) …
    assert [e for e in snap["events"] if e["kind"] == "safety.violation"]
    # … and the persistent slot + counter still carry the violation too.
    assert snap["topics"]["safety"]["last_violation"]["drop_reason"] == "collision"
    assert snap["counters"]["openral.event.safety_violation"] == 1


def test_error_events_survive_high_rate_flood_via_protected_lane() -> None:
    """Any error event (skill_failure, estop, ...) outlives the main-ring flood.

    The shared 200-slot event ring cycles in ~seconds under a 30 Hz stream; the
    protected error lane keeps the last N error/fatal events so the operator can
    still find WHY the robot stopped. Generic — not tied to safety.violation.
    """
    store = TelemetryStore()
    # An error-status span → a synthesised error-severity event (this is how a
    # reasoner skill-failure / a HAL estop surface a red row on the dashboard).
    es = time.time_ns()
    err_span = Span(
        trace_id=b"\x07" * 16,
        span_id=b"\x07" * 8,
        name="reasoner.skill_failure",
        start_time_unix_nano=es,
        end_time_unix_nano=es + 1_000,
        status=Status(code=2),  # ERROR
    )
    store.ingest_spans(_wrap(err_span))

    # Flood the 200-slot main ring well past capacity with info spans.
    fs = time.time_ns()
    flood = [
        Span(
            trace_id=b"\x01" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=fs,
            end_time_unix_nano=fs + 1_000,
            status=Status(code=0),
        )
        for i in range(400)
    ]
    store.ingest_spans(_wrap(*flood))

    snap = store.snapshot()
    error_kinds = [e["kind"] for e in snap["events"] if e["severity"] in ("error", "fatal")]
    assert "reasoner.skill_failure" in error_kinds


def test_ok_check_does_not_overwrite_last_violation() -> None:
    """A subsequent passing check resets its ledger pill but not last_violation."""
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    store.ingest_spans(
        _wrap(
            _safety_span(
                {
                    "safety.check_name": "envelope",
                    "safety.kernel": "cpp",
                    "safety.severity": "ok",
                }
            )
        )
    )
    safety = store.snapshot()["topics"]["safety"]
    assert safety["checks"]["envelope"]["severity"] == "ok"  # pill reset
    assert safety["last_violation"]["drop_reason"] == "collision"  # slot survives


def _clean_pass_span(kernel: str = "cpp") -> Span:
    """A passing safety check.

    The value is ``"info"``. This fixture used to send ``"ok"`` — which no
    emitter has ever produced — so it validated a fiction: the test passed
    while the real latch could only ever be set, never cleared.
    """
    return _safety_span(
        {"safety.check_name": "envelope", "safety.kernel": kernel, "safety.severity": "info"}
    )


def test_estopped_flag_defaults_false() -> None:
    """Before any safety activity the dashboard shows the robot as runnable."""
    assert TelemetryStore().snapshot()["topics"]["safety"]["estopped"] is False


def test_estopped_flag_latches_on_violation_and_clears_on_a_clean_pass() -> None:
    """The e-stop button's mode follows the kernel latch.

    A violation (self-collision, envelope, or an /openral/estop drop) latches
    the kernel → ``estopped`` True → the single button switches to Reset e-stop.
    A subsequent passing check means the kernel is running clean again →
    ``estopped`` False → the button switches back to E-STOP. Self-corrects after
    a reset with no rclpy node.

    That last sentence was aspirational until the clean-pass value was
    corrected from ``"ok"`` to ``"info"``: the kernel cannot publish a
    passing chunk while ``fault_latch_`` is set (it returns early with
    ``estop_latched`` at ``warn``), so a pass is proof of a clear.
    """
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    assert store.snapshot()["topics"]["safety"]["estopped"] is True
    store.ingest_spans(_wrap(_clean_pass_span()))
    assert store.snapshot()["topics"]["safety"]["estopped"] is False


def _skill_failure_span(state: str) -> Span:
    """A reasoner span carrying a skill_failure event (how the red row is born)."""
    start = time.time_ns()
    ev = Span.Event(
        name="openral.event.skill_failure",
        time_unix_nano=start,
        attributes=_attrs(
            {"openral.event.skill_failure.state": state, "reasoner.rskill_id": "OpenRAL/x"}
        ),
    )
    return Span(
        trace_id=b"\x0a" * 16,
        span_id=b"\x0a" * 8,
        name="reasoner.execute_rskill",
        start_time_unix_nano=start,
        end_time_unix_nano=start + 1_000,
        events=[ev],
        status=Status(code=0),
    )


def test_set_estopped_forces_flag() -> None:
    """The dashboard's own e-stop action drives the latch flag authoritatively.

    An operator e-stop aborts the skill, so no more safety.check spans flow and
    the telemetry path can't flip the flag — the API handler calls set_estopped
    so the Reset control still appears.
    """
    store = TelemetryStore()
    assert store.snapshot()["topics"]["safety"]["estopped"] is False
    store.set_estopped(True)
    assert store.snapshot()["topics"]["safety"]["estopped"] is True
    store.set_estopped(False)
    assert store.snapshot()["topics"]["safety"]["estopped"] is False


def test_skill_failure_is_error_when_not_latched() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap(_skill_failure_span("timeout")))
    ev = [e for e in store.snapshot()["events"] if e["kind"] == "openral.event.skill_failure"]
    assert ev and ev[0]["severity"] == "error"


def test_skill_failure_while_estopped_is_warning() -> None:
    """A skill_failure while e-stop-latched is a consequence of the stop, not a fault."""
    store = TelemetryStore()
    store.set_estopped(True)
    store.ingest_spans(_wrap(_skill_failure_span("aborted")))
    ev = [e for e in store.snapshot()["events"] if e["kind"] == "openral.event.skill_failure"]
    assert ev and ev[0]["severity"] == "warn"


def test_skill_failure_survives_flood_even_when_warn() -> None:
    """A latched (warn) skill_failure must still leave a durable trace + reason.

    Regression: skill_failure downgraded to "warn" while e-stopped used to fall
    out of the protected lane and get evicted by the 30 Hz read_state stream in
    seconds — the counter climbed but the event log showed nothing (the operator
    couldn't see WHY the skill stopped). It is now protected by kind regardless
    of severity.
    """
    store = TelemetryStore()
    store.set_estopped(True)
    store.ingest_spans(_wrap(_skill_failure_span("aborted")))

    # Flood the 200-slot main ring well past capacity — what evicts it live.
    start = time.time_ns()
    flood = [
        Span(
            trace_id=b"\x02" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=start,
            end_time_unix_nano=start + 1_000_000,
            attributes=_attrs({"openral.hal.adapter": "so100"}),
            status=Status(code=0),
        )
        for i in range(300)
    ]
    store.ingest_spans(_wrap(*flood))

    snap = store.snapshot()
    sf = [e for e in snap["events"] if e["kind"] == "openral.event.skill_failure"]
    assert sf, "latched skill_failure was evicted — no durable trace"
    assert sf[0]["severity"] == "warn"
    assert "aborted" in sf[0]["title"]  # the reason is carried on the surviving row
    # Counter still tallies it too.
    assert snap["counters"]["openral.event.skill_failure"] == 1


# ── e-stop latch self-clear ──────────────────────────────────────────────────
#
# The latch was set on "violation" and cleared only on "ok" — a value no
# emitter has ever produced. The C++ kernel sends `info` on a pass, `warn`
# while latched and `violation` on a drop; the Python passthrough matches it.
# So `estopped` stuck true until an explicit POST /api/estop_reset, and the
# store's own comment claiming it "self-corrects after a reset" was false.


def test_estop_latched_drops_do_not_clear_the_latch() -> None:
    """While latched the kernel emits `warn`/`estop_latched` — still stopped."""
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))

    store.ingest_spans(
        _wrap(
            _safety_span(
                {
                    "safety.check_name": "envelope",
                    "safety.kernel": "cpp",
                    "safety.severity": "warn",
                    "safety.drop_reason": "estop_latched",
                }
            )
        )
    )

    assert store.snapshot()["topics"]["safety"]["estopped"] is True


def test_the_null_client_cannot_unlatch_the_ui() -> None:
    """The null client emits "info" unconditionally without checking anything.

    Treating that as evidence of a clear would let a no-op safety client
    silently unlatch the operator's E-STOP indicator.
    """
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))

    store.ingest_spans(_wrap(_clean_pass_span(kernel="null")))

    assert store.snapshot()["topics"]["safety"]["estopped"] is True
