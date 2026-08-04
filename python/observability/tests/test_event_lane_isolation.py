"""Headline `info` rows must not evict safety events from the protected lane.

The dashboard keeps two protected rings beside the main FIFO. They exist for
different reasons and must not share a budget: a flood of routine `info` used
to push `safety.violation` / `estop_requested` out of the 64 slots that exist
precisely to keep them findable.
"""

from __future__ import annotations

from openral_observability.dashboard.store import (
    _ERROR_EVENT_RING_SIZE,
    _HEADLINE_EVENT_RING_SIZE,
    TelemetryEvent,
    TelemetryStore,
)


def _event(kind: str, severity: str, ts: float) -> TelemetryEvent:
    return TelemetryEvent(ts_unix=ts, kind=kind, title=kind, attrs={}, severity=severity)


def test_info_flood_does_not_evict_a_safety_event() -> None:
    """A safety violation stays findable under 10x the lane size in info rows."""
    store = TelemetryStore()
    store._append_event(_event("openral.event.safety_violation", "error", 1.0))

    # `world.scene_objects` ran at ~0.10/s live; this is ~3 hours of it.
    for i in range(_ERROR_EVENT_RING_SIZE * 10):
        store._append_event(_event("world.scene_objects", "info", 2.0 + i))

    kinds = [ev.kind for ev in store._merged_events()]
    assert "openral.event.safety_violation" in kinds


def test_headline_info_still_outlives_the_debug_flood() -> None:
    """The reason the mirroring was added in the first place still holds."""
    store = TelemetryStore()
    store._append_event(_event("deploy.bringup", "info", 1.0))

    # The main ring is 200 and cycles in seconds at 30 Hz; debug must not
    # take the bringup row with it.
    for i in range(1000):
        store._append_event(_event("hal.read_state", "debug", 2.0 + i))

    kinds = [ev.kind for ev in store._merged_events()]
    assert "deploy.bringup" in kinds


def test_each_class_lands_in_its_own_lane() -> None:
    """Errors and skill failures go to the error lane; other non-debug to headline."""
    store = TelemetryStore()
    store._append_event(_event("openral.event.estop_requested", "error", 1.0))
    store._append_event(_event("openral.event.skill_failure", "warn", 2.0))
    store._append_event(_event("deploy.bringup", "info", 3.0))
    store._append_event(_event("hal.read_state", "debug", 4.0))

    assert [ev.kind for ev in store._error_events] == [
        "openral.event.estop_requested",
        "openral.event.skill_failure",  # warn, but protected by kind
    ]
    assert [ev.kind for ev in store._headline_events] == ["deploy.bringup"]
    assert _HEADLINE_EVENT_RING_SIZE == _ERROR_EVENT_RING_SIZE
