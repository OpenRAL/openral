"""Query-time joiner for rosbag2 messages ↔ OTel spans.

ADR-0018 F7 — `rosbag2` stays the canonical message log; the dashboard
receiver (ADR-0017) stays the canonical span log; this module opens both
at view time and emits a single chronological timeline keyed by
``trace_id``.

Three pieces:

* :func:`read_bag` — iterate an mcap-backed rosbag2 directory or a bare
  ``.mcap`` file and yield :class:`BagMessage` records, surfacing the
  ``trace_id`` field that lives on every ADR-0018-typed ROS message
  (``ActionChunk``, ``FailureTrigger``, ``WorldStateStamped``,
  ``PromptStamped`` and the ``ExecuteSkill`` action goal/feedback/result).
* :class:`DashboardTraceClient` — HTTP client over the receiver's
  ``/api/traces`` + ``/api/spans/{trace_id}`` endpoints.
* :func:`build_timeline` — merge bag messages with spans for the given
  ``trace_id`` and return a sorted list of :class:`TimelineEntry`.

Per CLAUDE.md §1.11 the bag reader uses a real ``mcap`` reader against a
real recorded file; no message-stream mocks. Tests build a one-off mcap
in ``tmp_path`` with ``mcap.writer.Writer`` and replay it through this
module end-to-end.
"""

from __future__ import annotations

from openral_observability.replay.bag_reader import BagMessage, read_bag
from openral_observability.replay.correlator import (
    TimelineEntry,
    build_timeline,
    list_bag_trace_ids,
)
from openral_observability.replay.trace_query import DashboardTraceClient

__all__ = [
    "BagMessage",
    "DashboardTraceClient",
    "TimelineEntry",
    "build_timeline",
    "list_bag_trace_ids",
    "read_bag",
]
