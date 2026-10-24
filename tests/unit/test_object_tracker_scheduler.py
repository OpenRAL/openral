"""The query scheduler gives visible confirmed tracks a bounded refresh turn."""

from __future__ import annotations

from openral_world_state.object_tracker.scheduler import (
    PRIORITY_DISAGREE,
    PRIORITY_QUERY,
    PRIORITY_REFRESH,
    Scheduler,
    TrackerRequest,
)


def test_unresolved_destination_query_cannot_starve_confirmed_track_refresh() -> None:
    scheduler = Scheduler(segment_hz=2.0, budget_wall_s=5.0)
    query = TrackerRequest(
        "locate_in_view", "exterior", 1, PRIORITY_QUERY, 0, "plate", query_key=("deploy", "q1")
    )
    refresh = TrackerRequest(
        "segment_in_view", "exterior", 1, PRIORITY_REFRESH, 0, "banana", object_id="trk:1"
    )

    first = scheduler.next([query, refresh], wall_s=0.0)
    assert first is not None and first.query_key == ("deploy", "q1")
    scheduler.complete(first.request_id)

    second = scheduler.next([query, refresh], wall_s=0.5)
    assert second is not None and second.object_id == "trk:1"
    scheduler.complete(second.request_id)

    third = scheduler.next([query, refresh], wall_s=1.0)
    assert third is not None and third.query_key == ("deploy", "q1")


def test_owed_refresh_does_not_preempt_disagree_work_after_query_disappears() -> None:
    scheduler = Scheduler(segment_hz=2.0, budget_wall_s=5.0)
    query = TrackerRequest(
        "locate_in_view", "exterior", 1, PRIORITY_QUERY, 0, "plate", query_key=("deploy", "q1")
    )
    refresh = TrackerRequest(
        "segment_in_view", "exterior", 1, PRIORITY_REFRESH, 0, "banana", object_id="trk:1"
    )
    released = TrackerRequest(
        "segment_in_view", "exterior", 1, PRIORITY_DISAGREE, 0, "orange", object_id="trk:2"
    )

    first = scheduler.next([query, refresh], wall_s=0.0)
    assert first is not None and first.query_key == ("deploy", "q1")
    scheduler.complete(first.request_id)

    # The query resolved or disappeared while its refresh turn was owed. Preserve the
    # ordinary priority ordering among the remaining candidates.
    second = scheduler.next([refresh, released], wall_s=0.5)
    assert second is not None and second.object_id == "trk:2"
