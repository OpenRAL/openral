"""Segmenter scheduling under the GPU budget (issue #418, step 7).

One ``LocateInView`` / ``SegmentInView`` request in flight at most, issued at most every
``1 / segment_hz`` seconds of **wall** time (the GPU's clock, logged apart from the source
clock every lifetime runs on). Among the candidate prompts the tracker builds each frame,
the lowest priority number wins, then the one served longest ago (fair round-robin):

1. unresolved queries (``PRIORITY_QUERY``) — before any confirmed track is refreshed;
2. RELEASED tracks, searched in their fall column;
3. tracks the map disagrees with, re-found at their predicted pose;
4. bound-object protrusion checks;
5. the oldest visible CONFIRMED refresh, and resolved ``count: "all"`` queries, which keep
   detecting.

OCCLUDED tracks are never prompted: they are the map's job. A request whose answer does
not come back within ``budget_wall_s`` is abandoned and logged, freeing the slot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import structlog

__all__ = [
    "PRIORITY_BOUND",
    "PRIORITY_DISAGREE",
    "PRIORITY_QUERY",
    "PRIORITY_REFRESH",
    "PRIORITY_RELEASED",
    "Scheduler",
    "TrackerRequest",
]

log = structlog.get_logger(__name__)

PRIORITY_QUERY = 1
PRIORITY_RELEASED = 2
PRIORITY_DISAGREE = 3
PRIORITY_BOUND = 4
PRIORITY_REFRESH = 5


@dataclass(frozen=True)
class TrackerRequest:
    """One prompt for the node to issue, on the frame ``(camera, stamp_ns)``.

    Attributes:
        service: ``"locate_in_view"`` (open-vocab boxes for ``label``) or
            ``"segment_in_view"`` (one mask for a box and/or point prompt).
        camera: The camera the frame came from.
        stamp_ns: The depth stamp of that frame; the answer is fed back with it.
        priority: Step-7 priority it was chosen at.
        served_ns: Source time its subject was last served (fairness key).
        label: The label it looks for.
        query_key: ``(goal_id, query_id)`` it serves, if any.
        object_id: Track it refreshes, if any.
        box_xyxy: Box prompt, pixels.
        point_uv: Positive point prompt, pixels.
        request_id: Assigned by ``Scheduler.next``; 0 until issued.
    """

    service: Literal["locate_in_view", "segment_in_view"]
    camera: str
    stamp_ns: int
    priority: int
    served_ns: int
    label: str
    query_key: tuple[str, str] | None = None
    object_id: str | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    point_uv: tuple[float, float] | None = None
    request_id: int = 0


class Scheduler:
    """One request in flight, rate-capped in wall time, highest priority first.

    Args:
        segment_hz: Most requests per wall second. *Calibration point.*
        budget_wall_s: Wall seconds before an unanswered request is abandoned.

    Example:
        >>> s = Scheduler(segment_hz=2.0, budget_wall_s=5.0)
        >>> a = TrackerRequest("segment_in_view", "top", 1, 5, 0, "orange", object_id="trk:1")
        >>> b = TrackerRequest("locate_in_view", "top", 1, 1, 0, "orange", query_key=("g", "q"))
        >>> s.next([a, b], wall_s=0.0).service
        'locate_in_view'
        >>> s.next([a], wall_s=0.1) is None  # one in flight
        True
        >>> s.complete(1).request_id, s.next([a], wall_s=0.2) is None  # rate cap
        (1, True)
        >>> s.next([a], wall_s=0.6).object_id
        'trk:1'
    """

    def __init__(self, *, segment_hz: float, budget_wall_s: float) -> None:
        """Start idle."""
        self._period_s = 1.0 / segment_hz
        self._budget_s = budget_wall_s
        self._in_flight: TrackerRequest | None = None
        self._issued_s = -float("inf")
        self._next_id = 1

    @property
    def in_flight(self) -> TrackerRequest | None:
        """The request awaiting its answer, if any."""
        return self._in_flight

    def complete(self, request_id: int) -> TrackerRequest | None:
        """Free the slot for ``request_id``; return the request, or ``None`` if unknown."""
        req = self._in_flight
        if req is None or req.request_id != request_id:
            log.warning("object_tracker.unexpected_result", request_id=request_id)
            return None
        self._in_flight = None
        return req

    def next(self, candidates: Sequence[TrackerRequest], *, wall_s: float) -> TrackerRequest | None:
        """Issue the best candidate if the slot is free and the rate allows, else ``None``."""
        if self._in_flight is not None:
            if wall_s - self._issued_s <= self._budget_s:
                return None
            log.warning(
                "object_tracker.segment_budget_wall_s",
                request_id=self._in_flight.request_id,
                budget_wall_s=self._budget_s,
            )
            self._in_flight = None
        if not candidates or wall_s - self._issued_s < self._period_s:
            return None
        best = min(candidates, key=lambda r: (r.priority, r.served_ns))
        self._in_flight = replace(best, request_id=self._next_id)
        self._next_id += 1
        self._issued_s = wall_s
        return self._in_flight
