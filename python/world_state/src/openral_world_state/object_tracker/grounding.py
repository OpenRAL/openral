"""Standing queries: grounding never takes a first-frame snapshot (issue #418, Addendum G).

Each ``ObjectQuery`` of a goal lives as long as the goal, keyed ``(goal_id, query_id)``,
with its own ``ObjectQueryStatus``:

* ``UNRESOLVED`` — no track matches yet;
* ``RESOLVING`` — ≥ 1 TENTATIVE (or RELEASED) match, none counted yet;
* ``RESOLVED`` — ≥ ``min_instances`` CONFIRMED / OCCLUDED / BOUND matches. When every one
  is lost again the query goes back to ``UNRESOLVED`` and its search timer restarts;
* ``EXHAUSTED`` — the search budget is spent; terminal for the goal.

A track matches a query by label and role. A query that stays unresolved for
``search_start_s`` of **source** time emits one ``QueryEscalation("unresolved")`` — the data
the reasoner turns into ``ObjectQueryUnresolved`` and the ``find-object`` playbook. It is
emitted, never executed here. After ``search_max_s`` the query is ``EXHAUSTED`` and a
``QueryEscalation("exhausted")`` hands the goal to the replanning ladder (#353). The last
LOST match's box rides along as ``last_seen``: a search prior, never kernel geometry.

``due`` is the passive-search order (G.2): every query not yet resolved, least recently
served first (fair: no query is starved by another, nor by the hand), then resolved
``count: "all"`` queries, which keep detecting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import structlog
from openral_core import ObjectQuery, ObjectQueryStatus, PlaceRegion, TrackedObject, TrackStatus

__all__ = ["QueryBook", "QueryEscalation", "StandingQuery"]

log = structlog.get_logger(__name__)

_NS = 1_000_000_000
_COUNTED = frozenset({TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.BOUND})
_PENDING = frozenset({TrackStatus.TENTATIVE, TrackStatus.RELEASED})


@dataclass
class StandingQuery:
    """One goal's query, as the tracker holds it.

    Attributes:
        goal_id: The goal the query lives for.
        query_id: Its key within the goal.
        query: The prompt's query.
        declaration_key: ``"<target_id>@<stamp_ns>"`` minted tracks carry.
        status: Query status.
        since_ns: Source time the current search started (goal start or last loss).
        last_served_ns: Source time a detection was last spent on it.
        escalated: The ``unresolved`` escalation was emitted for this search.
    """

    goal_id: str
    query_id: str
    query: ObjectQuery
    declaration_key: str
    status: ObjectQueryStatus
    since_ns: int
    last_served_ns: int = -1
    escalated: bool = False

    @property
    def key(self) -> tuple[str, str]:
        """``(goal_id, query_id)``."""
        return self.goal_id, self.query_id

    def matches(self, track: TrackedObject) -> bool:
        """Whether ``track`` is an instance of this query (label and role)."""
        return track.label == self.query.label and track.role is self.query.role


@dataclass(frozen=True)
class QueryEscalation:
    """Active-search data for the reasoner (G.3), never acted on by the tracker.

    Attributes:
        kind: ``"unresolved"`` (start ``find-object``) or ``"exhausted"`` (budget spent).
        goal_id: The goal.
        query_id: The query.
        label: Its label.
        last_seen: The last lost match's box, a search prior; ``None`` if never seen.
        stamp_ns: Source time of the escalation.
    """

    kind: Literal["unresolved", "exhausted"]
    goal_id: str
    query_id: str
    label: str
    last_seen: PlaceRegion | None
    stamp_ns: int


class QueryBook:
    """The standing queries of every live goal.

    Args:
        search_start_s: Unresolved source time before active search is requested.
        search_max_s: Unresolved source time before the query is ``EXHAUSTED``.

    Example:
        >>> from openral_core import ObjectRole
        >>> book = QueryBook(search_start_s=5.0, search_max_s=60.0)
        >>> q = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)
        >>> book.open("g1", "q0", q, "obj:orange@0", stamp_ns=0)
        >>> [e.kind for e in book.update([], now_ns=6 * 10**9)]
        ['unresolved']
        >>> [e.kind for e in book.update([], now_ns=61 * 10**9)], book.get("g1", "q0").status.value
        (['exhausted'], 'exhausted')
    """

    def __init__(self, *, search_start_s: float, search_max_s: float) -> None:
        """Start with no queries."""
        self._start_ns = int(search_start_s * _NS)
        self._max_ns = int(search_max_s * _NS)
        self._queries: dict[tuple[str, str], StandingQuery] = {}

    @property
    def queries(self) -> list[StandingQuery]:
        """Every standing query."""
        return list(self._queries.values())

    def get(self, goal_id: str, query_id: str) -> StandingQuery | None:
        """The query ``(goal_id, query_id)``, or ``None``."""
        return self._queries.get((goal_id, query_id))

    def open(
        self,
        goal_id: str,
        query_id: str,
        query: ObjectQuery,
        declaration_key: str,
        *,
        stamp_ns: int,
    ) -> None:
        """Start a standing query at goal dispatch (source time ``stamp_ns``)."""
        self._queries[(goal_id, query_id)] = StandingQuery(
            goal_id, query_id, query, declaration_key, ObjectQueryStatus.UNRESOLVED, stamp_ns
        )
        log.info("object_tracker.query_open", goal_id=goal_id, query_id=query_id, label=query.label)

    def close_goal(self, goal_id: str) -> None:
        """Drop every query of ``goal_id`` (success, cancel or the terminal rung)."""
        for key in [k for k in self._queries if k[0] == goal_id]:
            del self._queries[key]

    def ready(self, goal_id: str) -> bool:
        """``grounding_ready``: every query of the goal is ``RESOLVED``."""
        mine = [q for q in self._queries.values() if q.goal_id == goal_id]
        return bool(mine) and all(q.status is ObjectQueryStatus.RESOLVED for q in mine)

    def update(self, tracks: Sequence[TrackedObject], *, now_ns: int) -> list[QueryEscalation]:
        """Re-derive every query's status from the tracks; return new escalations."""
        out: list[QueryEscalation] = []
        for q in self._queries.values():
            if q.status is ObjectQueryStatus.EXHAUSTED:
                continue
            mine = [t for t in tracks if q.matches(t)]
            counted = sum(t.status in _COUNTED for t in mine)
            if counted >= q.query.min_instances:
                status = ObjectQueryStatus.RESOLVED
            elif any(t.status in _PENDING for t in mine):
                status = ObjectQueryStatus.RESOLVING
            else:
                status = ObjectQueryStatus.UNRESOLVED
            if q.status is ObjectQueryStatus.RESOLVED and status is not q.status:
                q.since_ns, q.escalated = now_ns, False  # lost again: the search restarts
            if status is not q.status:
                log.info("object_tracker.query", key=q.key, status=status.value)
                q.status = status
            if status is ObjectQueryStatus.RESOLVED:
                continue
            lost = [t for t in mine if t.status is TrackStatus.LOST]
            last = max(lost, key=lambda t: t.last_fit_stamp_ns).box if lost else None
            waited = now_ns - q.since_ns
            if waited >= self._max_ns:
                q.status = ObjectQueryStatus.EXHAUSTED
                out.append(QueryEscalation("exhausted", *q.key, q.query.label, last, now_ns))
            elif waited >= self._start_ns and not q.escalated:
                q.escalated = True
                out.append(QueryEscalation("unresolved", *q.key, q.query.label, last, now_ns))
        for e in out:
            log.info("object_tracker.escalation", kind=e.kind, goal_id=e.goal_id, label=e.label)
        return out

    def due(self) -> list[StandingQuery]:
        """Queries owed a detection, in fair order (module docstring)."""
        open_ = [
            q
            for q in self._queries.values()
            if q.status in (ObjectQueryStatus.UNRESOLVED, ObjectQueryStatus.RESOLVING)
        ]
        more = [
            q
            for q in self._queries.values()
            if q.status is ObjectQueryStatus.RESOLVED and q.query.count == "all"
        ]
        return sorted(open_, key=lambda q: q.last_served_ns) + sorted(
            more, key=lambda q: q.last_served_ns
        )
