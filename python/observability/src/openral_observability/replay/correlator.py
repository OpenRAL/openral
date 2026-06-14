"""Join rosbag2 messages with OTel spans at view time.

ADR-0018 F7 — the canonical message log lives in mcap; the canonical
span log lives behind the dashboard receiver (ADR-0017). This module
opens both, joins on ``trace_id``, and emits one chronological list of
:class:`TimelineEntry` records suitable for ``openral replay`` output or a
dashboard scrub UI.

The join is pure: it takes already-loaded iterables of bag messages and
span dicts and returns a sorted timeline. The CLI layer
(:mod:`openral_observability.replay.cli`) handles fetching them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from openral_observability.replay.bag_reader import BagMessage

__all__ = ["TimelineEntry", "build_timeline", "list_bag_trace_ids"]


@dataclass(frozen=True)
class TimelineEntry:
    """One row in the correlated timeline.

    Either a ROS bag message or an OTel span. Both carry ``ts_ns`` and
    ``trace_id`` so callers can render a unified scrub line.

    Attributes:
        kind: ``"bag"`` for a rosbag2 record, ``"span"`` for an OTel span.
        ts_ns: Unix nanoseconds of the event start (``log_time_ns`` for
            bag messages, ``start_unix_ns`` for spans).
        trace_id: 32-hex-char trace_id (empty for unjoined bag messages
            published without an ADR-0018 ``trace_id`` field).
        topic: ROS topic when ``kind=='bag'``; empty for spans.
        span_name: OTel span name when ``kind=='span'``; empty for bag
            messages.
        attrs: Payload summary (bag) or attribute dict (span).
        duration_ms: Span duration in ms when ``kind=='span'``; ``None``
            for bag messages.
    """

    kind: Literal["bag", "span"]
    ts_ns: int
    trace_id: str
    topic: str
    span_name: str
    attrs: dict[str, Any]
    duration_ms: float | None

    def to_json(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for ``json.dumps``."""
        return {
            "kind": self.kind,
            "ts_unix_ns": self.ts_ns,
            "trace_id": self.trace_id,
            "topic": self.topic,
            "span_name": self.span_name,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
        }


def list_bag_trace_ids(bag_messages: Iterable[BagMessage]) -> list[dict[str, Any]]:
    """Return distinct trace_ids in ``bag_messages`` with their per-trace counts.

    Sorted by descending count so the user can pick the busy trace (the
    one with the actual rollout) over noise. Empty trace_ids are
    filtered out — those records simply don't participate in F7's
    correlation contract.
    """
    counter: Counter[str] = Counter()
    earliest: dict[str, int] = {}
    for m in bag_messages:
        if not m.trace_id:
            continue
        counter[m.trace_id] += 1
        prev = earliest.get(m.trace_id)
        if prev is None or m.log_time_ns < prev:
            earliest[m.trace_id] = m.log_time_ns
    return [
        {
            "trace_id": tid,
            "bag_message_count": count,
            "earliest_log_time_ns": earliest[tid],
        }
        for tid, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ]


def build_timeline(
    bag_messages: Iterable[BagMessage],
    spans: Iterable[dict[str, Any]],
    *,
    trace_id: str | None = None,
) -> list[TimelineEntry]:
    """Merge ``bag_messages`` + ``spans`` into one sorted timeline.

    Args:
        bag_messages: Records from :func:`read_bag`.
        spans: Span dicts as returned by ``/api/spans/{trace_id}``
            (i.e. :meth:`TelemetryStore.lookup_trace`).
        trace_id: When set, both inputs are filtered down to this
            trace_id before merging — spans with a different trace_id
            are dropped, and bag messages with an empty trace_id are
            dropped only if any of them have a non-empty one (so a bag
            recorded without ADR-0018 trace_ids still surfaces).

    Returns:
        Timeline entries sorted by ``ts_ns`` ascending.
    """
    bag_list = list(bag_messages)
    span_list = list(spans)

    if trace_id is not None:
        span_list = [s for s in span_list if s.get("trace_id") == trace_id]
        any_tagged = any(m.trace_id for m in bag_list)
        bag_list = [
            m for m in bag_list if (m.trace_id == trace_id) or (not any_tagged and not m.trace_id)
        ]

    entries: list[TimelineEntry] = []
    for m in bag_list:
        entries.append(
            TimelineEntry(
                kind="bag",
                ts_ns=m.log_time_ns,
                trace_id=m.trace_id,
                topic=m.topic,
                span_name="",
                attrs=m.payload_summary,
                duration_ms=None,
            )
        )
    for s in span_list:
        entries.append(
            TimelineEntry(
                kind="span",
                ts_ns=int(s.get("start_unix_ns", 0)),
                trace_id=str(s.get("trace_id", "")),
                topic="",
                span_name=str(s.get("name", "")),
                attrs=dict(s.get("attrs", {})),
                duration_ms=(
                    float(s["duration_ms"])
                    if isinstance(s.get("duration_ms"), int | float)
                    else None
                ),
            )
        )
    entries.sort(key=lambda e: e.ts_ns)
    return entries
