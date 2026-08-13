"""The dashboard's Safety · current state card (ADR-0096).

The card reads the LATCHED ``/openral/safety_status`` topic — the graph's
authoritative answer to "what is safety doing right now" — rather than
inferring a latch from ``safety.check`` spans. That inference only moves while
command chunks flow, and a latched kernel is precisely the state in which
chunks stop flowing, so the span path cannot report the thing an operator most
needs to see.

Split the way ``test_dashboard_estop_counter.py`` splits: the rclpy half
(``SafetyStatusSubscriber``) needs a ROS workspace CI's cheap env does not
have, so it is import-gated; the store + snapshot contract the frontend reads
is covered unconditionally. The UI contract is pinned by asserting the exact
snapshot keys ``dashboard.js``'s ``renderSafetyStatus`` reads, which is how the
rest of this suite pins card behaviour (no test parses the JS).

No mocks (CLAUDE.md §1.11): the store is the production class, and the
subscriber test drives the real generated ``openral_msgs/SafetyStatus``.
"""

from __future__ import annotations

import time

import pytest
from openral_observability.dashboard import TelemetryStore
from openral_observability.dashboard.safety_status_subscriber import drop_reason_label

# The keys dashboard.js's renderSafetyStatus() reads off
# ``snapshot()["topics"]["safety_status"]``. Renaming one without touching the
# JS silently blanks the card, which is a safety surface going quiet.
_UI_KEYS = {
    "latched",
    "drop_reason",
    "drop_reason_label",
    "detail",
    "rskill_id",
    "trace_id",
    "stamp_unix",
    "ts_unix",
}


def _set(store: TelemetryStore, **overrides: object) -> None:
    """Write one status into the store with realistic defaults."""
    payload: dict[str, object] = {
        "latched": False,
        "drop_reason": 255,
        "drop_reason_label": "drop_none",
        "detail": "kernel activated",
        "rskill_id": "",
        "trace_id": "",
        "stamp_unix": time.time(),
    }
    payload.update(overrides)
    store.set_safety_status(**payload)  # type: ignore[arg-type]  # reason: kwargs built dynamically


def test_safety_status_starts_empty_not_clear() -> None:
    """With no safety node seen, the topic is EMPTY — never a fabricated clear.

    "We cannot see the safety layer" and "the safety layer says all good" are
    different facts; the card renders "waiting" for the first and must never be
    handed the second by default (CLAUDE.md §1.2).
    """
    store = TelemetryStore()
    assert store.snapshot()["topics"]["safety_status"] == {}


def test_latched_status_reaches_the_snapshot_with_every_ui_key() -> None:
    """A latched status surfaces with the full field set the card renders."""
    store = TelemetryStore()
    _set(
        store,
        latched=True,
        drop_reason=10,
        drop_reason_label="kind_collision",
        detail="world",
        rskill_id="rskills/rskill-smolvla-so100",
        trace_id="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )
    status = store.snapshot()["topics"]["safety_status"]
    assert set(status) == _UI_KEYS
    assert status["latched"] is True
    assert status["drop_reason"] == 10
    assert status["drop_reason_label"] == "kind_collision"
    assert status["detail"] == "world"
    assert status["rskill_id"] == "rskills/rskill-smolvla-so100"
    # The receipt stamp is on the dashboard's own clock so the card's age is
    # meaningful even when the publisher runs on sim time.
    assert status["ts_unix"] > 0


def test_non_latching_drop_is_distinguishable_from_a_latch() -> None:
    """A fail-closed drop reports its DROP_* reason with ``latched=False``.

    The card says "dropping · <reason>" for this, which is a different
    operator situation from a latch: motion resumes on its own once the
    upstream input comes back, no ``/openral/estop_reset`` needed.
    """
    store = TelemetryStore()
    _set(
        store,
        latched=False,
        drop_reason=103,
        drop_reason_label="drop_state_unavailable",
        detail="state_unavailable",
    )
    status = store.snapshot()["topics"]["safety_status"]
    assert status["latched"] is False
    assert status["drop_reason_label"] == "drop_state_unavailable"


def test_recovery_overwrites_the_latched_value() -> None:
    """The card follows recovery — a cleared kernel must not read as latched."""
    store = TelemetryStore()
    _set(store, latched=True, drop_reason=106, drop_reason_label="drop_external_estop")
    assert store.snapshot()["topics"]["safety_status"]["latched"] is True
    _set(store, latched=False, drop_reason=255, drop_reason_label="drop_none")
    status = store.snapshot()["topics"]["safety_status"]
    assert status["latched"] is False
    assert status["drop_reason_label"] == "drop_none"


def test_status_update_reaches_sse_subscribers() -> None:
    """Every status is pushed, not just polled — the card updates on transition."""
    import asyncio

    async def _run() -> dict[str, object]:
        store = TelemetryStore()
        queue = store.subscribe()
        try:
            _set(store, latched=True, drop_reason=2, drop_reason_label="kind_workspace")
            payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        finally:
            store.unsubscribe(queue)
        return dict(payload)

    payload = asyncio.run(_run())
    assert payload["topics"]["safety_status"]["latched"] is True  # type: ignore[index]  # reason: snapshot is a plain nested dict


def test_drop_reason_label_reads_the_idl_constants() -> None:
    """Labels come from the generated message class, with no second table.

    Needs the colcon ``openral_msgs`` overlay; skips without it rather than
    asserting against a stand-in (CLAUDE.md §1.11).
    """
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import SafetyStatus

    assert drop_reason_label(SafetyStatus.KIND_COLLISION, SafetyStatus) == "kind_collision"
    assert (
        drop_reason_label(SafetyStatus.DROP_ENVELOPE_UNCONFIGURED, SafetyStatus)
        == "drop_envelope_unconfigured"
    )
    assert drop_reason_label(SafetyStatus.DROP_NONE, SafetyStatus) == "drop_none"
    # A publisher newer than this dashboard degrades to a readable placeholder
    # instead of silently borrowing some other constant's name.
    assert drop_reason_label(200, SafetyStatus) == "drop_reason_200"


def test_subscriber_writes_a_real_message_into_the_store() -> None:
    """The rclpy half: a real ``SafetyStatus`` lands as the card's payload.

    Drives ``SafetyStatusSubscriber._on_status`` with a real generated message
    (no ROS graph needed for the store-write path, which is the half that can
    break silently). Skips without the ``openral_msgs`` overlay.
    """
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import SafetyStatus
    from openral_observability.dashboard.safety_status_subscriber import SafetyStatusSubscriber

    store = TelemetryStore()
    subscriber = SafetyStatusSubscriber.__new__(SafetyStatusSubscriber)
    subscriber._store = store  # reason: exercising the callback without a ROS graph

    msg = SafetyStatus()
    msg.latched = True
    msg.drop_reason = SafetyStatus.KIND_WORKSPACE
    msg.detail = "joint_position"
    msg.rskill_id = "rskills/rskill-smolvla-so100"
    msg.header.stamp.sec = 1234
    msg.header.stamp.nanosec = 500_000_000
    subscriber._on_status(msg)  # reason: the subscription callback is the unit under test

    status = store.snapshot()["topics"]["safety_status"]
    assert status["latched"] is True
    assert status["drop_reason_label"] == "kind_workspace"
    assert status["detail"] == "joint_position"
    assert status["stamp_unix"] == pytest.approx(1234.5)
