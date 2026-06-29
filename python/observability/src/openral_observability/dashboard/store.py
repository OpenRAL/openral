"""In-memory aggregator for the live dashboard.

:class:`TelemetryStore` is the single shared object the OTLP/HTTP
receiver writes into and the SSE / JSON endpoints read from. It is
thread-safe (an asyncio task feeds it from request handlers; the SSE
generator subscribes from another task) and bounded — every internal
container has a fixed cap so a long-running session cannot grow
unboundedly.

The store keeps a *latest-wins* view of the three signals the dashboard
foregrounds — ``rskill.execute`` / ``rskill.chunk_inference`` /
``safety.check`` — plus a small ring of events (e-stop, safety
violation, deadline missed, sensor stale, ...) and per-instrument
rolling samples for metrics.

Wire format: callers feed in already-decoded
:class:`opentelemetry.proto.trace.v1.trace_pb2.ResourceSpans` /
:class:`opentelemetry.proto.metrics.v1.metrics_pb2.ResourceMetrics`
messages. The receiver in :mod:`openral_observability.dashboard.receivers`
does the protobuf decode; the store never parses protobuf itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs
from opentelemetry.proto.metrics.v1.metrics_pb2 import Metric, ResourceMetrics
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span

__all__ = ["TelemetryEvent", "TelemetryStore"]

_EVENT_RING_SIZE = 200
_METRIC_SAMPLE_RING_SIZE = 600  # ~5 min at one sample per 500 ms
_SUBSCRIBER_QUEUE_SIZE = 256
# OTLP Status.code values per opentelemetry-proto: 0=UNSET, 1=OK, 2=ERROR.
_STATUS_ERROR = 2

# OTLP SeverityNumber bands (opentelemetry-proto logs/v1): four numbers per
# level — TRACE 1-4, DEBUG 5-8, INFO 9-12, WARN 13-16, ERROR 17-20, FATAL
# 21-24. The dashboard event log collapses each band to its level name so a
# structlog→OTel DEBUG record renders as a `debug` row (issue #318). DEBUG
# shares the `>= 1` floor in _log_level (DEBUG 5-8 + the TRACE band both
# surface as `debug`), so it needs no dedicated threshold constant here.
_SEVERITY_FATAL_MIN = 21
_SEVERITY_ERROR_MIN = 17
_SEVERITY_WARN_MIN = 13
_SEVERITY_INFO_MIN = 9

# ADR-0018 F7 — query-time bag↔OTel join. Cap memory: keep at most
# _TRACE_INDEX_MAX_TRACES distinct trace_ids and _TRACE_INDEX_MAX_SPANS
# spans per trace. Old traces evict in arrival order.
_TRACE_INDEX_MAX_TRACES = 64
_TRACE_INDEX_MAX_SPANS = 2048


_ANY_VALUE_DECODERS: dict[str, Any] = {
    "string_value": lambda av: av.string_value,
    "bool_value": lambda av: av.bool_value,
    "int_value": lambda av: av.int_value,
    "double_value": lambda av: av.double_value,
    "bytes_value": lambda av: av.bytes_value,
}


def _attr_value(av: AnyValue) -> Any:
    """Decode an OTLP ``AnyValue`` into a plain Python value.

    The proto uses a oneof; we surface whichever field is set, falling
    back to ``None`` so the renderer never has to special-case missing
    attributes. ``array_value`` and ``kvlist_value`` recurse.
    """
    which = av.WhichOneof("value")
    if which is None:
        return None
    decoder = _ANY_VALUE_DECODERS.get(which)
    if decoder is not None:
        return decoder(av)
    if which == "array_value":
        return [_attr_value(v) for v in av.array_value.values]
    if which == "kvlist_value":
        return {kv.key: _attr_value(kv.value) for kv in av.kvlist_value.values}
    return None


def _attrs_to_dict(attrs: list[KeyValue]) -> dict[str, Any]:
    """Flatten a repeated ``KeyValue`` into a plain dict."""
    return {kv.key: _attr_value(kv.value) for kv in attrs}


_MIN_POLYGON_FLOATS = 6  # 3 points x 2 coordinates -- fewer than this is not a polygon


def _reshape_xy_pairs(flat: object) -> list[list[float]] | None:
    """Reshape a flat ``[x0, y0, x1, y1, ...]`` attr into ``[[x0, y0], ...]``.

    Returns ``None`` when the attribute is absent or not an even-length list
    of >= 6 values (a malformed payload should fall back to the circle, not
    draw garbage).
    """
    if not isinstance(flat, list) or len(flat) < _MIN_POLYGON_FLOATS or len(flat) % 2 != 0:
        return None
    return [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat), 2)]


def _parse_object_list(raw: object) -> list[dict[str, object]]:
    """Decode the ``world.scene_objects.list`` JSON attr into a list of dicts.

    Returns ``[]`` for an absent or malformed payload — a bad map must show as
    "no objects", never crash the receiver.
    """
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [obj for obj in parsed if isinstance(obj, dict)]


def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


@dataclass(frozen=True)
class TelemetryEvent:
    """One event surfaced on the dashboard event log.

    Events come from three places: explicit OTel span events (e.g.
    ``openral.event.safety_violation``, ``openral.event.estop_requested``),
    a synthesised entry per ingested span (``rskill.execute``,
    ``safety.check``, ...), and real log lines bridged from structlog over
    OTLP (``ingest_logs`` — issue #318) so the operator can see the most
    recent activity in chronological order.

    Attributes:
        ts_unix: Wall-clock seconds.
        kind: Short event kind (``rskill.execute``, ``safety.violation``,
            or the logger/scope name for a bridged log line).
        title: One-line human label rendered in the UI.
        attrs: Decoded attribute dict; rendered as key/value pairs.
        severity: ``info`` (default), ``debug``, ``warn``, ``error``, or
            ``fatal``.
    """

    ts_unix: float
    kind: str
    title: str
    attrs: dict[str, Any]
    severity: str = "info"

    def to_json(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for JSON serialization."""
        return {
            "ts_unix": self.ts_unix,
            "kind": self.kind,
            "title": self.title,
            "attrs": self.attrs,
            "severity": self.severity,
        }


@dataclass
class _IndexedSpan:
    """One span retained by trace_id for the F7 query-time correlator.

    The dashboard store keeps a bounded per-trace index so ``openral replay``
    can join a rosbag2 against the canonical span log without writing a
    separate trace store. Slimmed to what the timeline view needs:
    name, timestamps, attributes, status, and the parent/span ids so a
    consumer can rebuild the local tree.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str
    start_ns: int
    end_ns: int
    attrs: dict[str, Any]
    status_code: int
    status_message: str
    events: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_unix_ns": self.start_ns,
            "end_unix_ns": self.end_ns,
            "duration_ms": _ns_to_ms(self.end_ns - self.start_ns),
            "attrs": self.attrs,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "events": self.events,
        }


@dataclass
class _SpanCard:
    """Latest-wins record for one of the headline span families."""

    name: str
    ts_unix: float
    duration_ms: float
    attrs: dict[str, Any]
    status_code: int = 0  # OTLP StatusCode (UNSET=0, OK=1, ERROR=2)
    status_message: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ts_unix": self.ts_unix,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
            "status_code": self.status_code,
            "status_message": self.status_message,
        }


@dataclass
class _MetricSeries:
    """Rolling samples for one metric instrument, keyed by metric name + labels.

    For histograms we keep the per-export bucket sums and counts as
    plain samples (treated as average per export interval); for
    counters we store the cumulative value; for gauges the latest
    value. Percentiles are computed on read from the sample ring.
    """

    name: str
    kind: str  # "histogram" | "sum" | "gauge"
    unit: str
    samples: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=_METRIC_SAMPLE_RING_SIZE)
    )
    cumulative: float = 0.0  # last observed cumulative value for sums
    labels: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        values = [v for _, v in self.samples]
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "unit": self.unit,
            "labels": self.labels,
            "latest": values[-1] if values else None,
            "cumulative": self.cumulative,
            "samples": list(self.samples),
        }
        if self.kind == "histogram" and values:
            sorted_vals = sorted(values)
            out["p50"] = statistics.median(sorted_vals)
            out["p95"] = _percentile(sorted_vals, 0.95)
            out["p99"] = _percentile(sorted_vals, 0.99)
        return out


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class TelemetryStore:
    """Bounded, thread-safe aggregator over OTLP signals.

    The store is the read-side of the dashboard. Two writers feed it
    (``ingest_spans`` and ``ingest_metrics``); two readers consume it
    (``snapshot`` for the JSON endpoint, ``subscribe`` for the SSE
    stream). All public methods are safe to call from any thread.
    """

    def __init__(self) -> None:
        """Initialise an empty store with no subscribers."""
        self._lock = threading.Lock()
        # All distinct service.name values seen this run. A deploy graph has
        # many nodes (openral.runtime, openral.hal.<robot>, openral.reasoner,
        # …), each its own OTLP resource; the Identity card's "service" must
        # not flicker as spans interleave, so we pick a stable primary from
        # this set (see _primary_service) instead of last-write-wins.
        self._services: set[str] = set()
        self._run_id: str = ""
        self._run_mode: str = ""
        self._git_sha: str = ""
        self._last_ingest_ts: float = 0.0
        self._cards: dict[str, _SpanCard] = {}
        self._events: deque[TelemetryEvent] = deque(maxlen=_EVENT_RING_SIZE)
        self._counters: dict[str, int] = defaultdict(int)
        self._metrics: dict[str, _MetricSeries] = {}
        # Topical state buckets — one per "topic" the dashboard renders
        # as a dedicated card. Latched/static keys (run mode, robot
        # model, skill id, kernel) live in :attr:`_identity`; everything
        # high-frequency lives under :attr:`_topics` keyed by topic name.
        # ADR-0018 F7: bounded per-trace span index. Ordered dict so
        # eviction is FIFO on first-seen trace_id; each value is a deque
        # capped by _TRACE_INDEX_MAX_SPANS.
        self._spans_by_trace: dict[str, deque[_IndexedSpan]] = {}
        self._trace_last_seen: dict[str, float] = {}
        self._identity: dict[str, Any] = {}
        self._topics: dict[str, dict[str, Any]] = {
            "robot_state": {},
            "commands": {},
            "world_state": {},
            "perception": {},  # per-camera modality + age + thumbnail
            "inference": {},
            "safety": {"checks": {}},  # check_name -> {last_verdict, severity, ts}
            "system": {},  # populated by metrics ingest (gpu/cpu/ram)
            # ADR-0025 — live 2D occupancy map from slam_toolbox (and any
            # future Reasoner-managed mapping service). Populated by
            # ``slam.occupancy_grid`` spans emitted by
            # ``openral_runner.slam_bridge.SlamMapBridge``.
            "slam": {},
            # ADR-0030 — robot-perspective octomap pointcloud render.
            # Populated by ``world.pointcloud`` spans emitted by
            # ``openral_runner.world_cloud_bridge.WorldCloudBridge``.
            "pointcloud": {},
            # ADR-0038 — durable spatial-memory objects (table card + map
            # overlay). Populated by ``world.scene_objects`` spans emitted by
            # ``openral_world_state.emit_scene_objects_span`` (the Reasoner's
            # preloaded map today; the World-State node post-producer).
            "scene_objects": {},
            # ADR-0018 F4 — last Reasoner tick (one entry per
            # `reasoner.tick` span emitted by `ReasonerCore.tick`).
            # The dashboard "Reasoner" card reads this to show the
            # latest tool decision; the Event Log carries the full
            # history.
            "reasoner": {},
            "trace": {},  # latest_trace_id
        }
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        # We notify subscribers via the loop that created them. The
        # receiver may run in a worker thread (uvicorn worker pool);
        # the subscription endpoint captures its loop on `subscribe`.
        self._sub_loops: dict[int, asyncio.AbstractEventLoop] = {}

    # ── Receiver-facing API ────────────────────────────────────────────

    def ingest_spans(self, payload: list[ResourceSpans]) -> int:
        """Decode + record a batch of ``ResourceSpans``.

        Returns the number of spans recorded — useful for receiver
        observability.
        """
        recorded = 0
        snapshot_payload: dict[str, Any] | None = None
        with self._lock:
            for resource_spans in payload:
                resource_attrs = _attrs_to_dict(list(resource_spans.resource.attributes))
                service = str(resource_attrs.get("service.name", ""))
                if service:
                    self._services.add(service)
                run_id = str(resource_attrs.get("openral.run.id", ""))
                if run_id:
                    self._run_id = run_id
                run_mode = str(resource_attrs.get("openral.run.mode", ""))
                if run_mode:
                    self._run_mode = run_mode
                git_sha = str(resource_attrs.get("openral.run.git_sha", ""))
                if git_sha:
                    self._git_sha = git_sha
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        self._record_span(span)
                        recorded += 1
            if recorded:
                self._last_ingest_ts = time.time()
                snapshot_payload = self._snapshot_locked()
        if snapshot_payload is not None:
            self._publish(snapshot_payload)
        return recorded

    def ingest_metrics(self, payload: list[ResourceMetrics]) -> int:
        """Decode + record a batch of ``ResourceMetrics``."""
        recorded = 0
        snapshot_payload: dict[str, Any] | None = None
        with self._lock:
            for resource_metrics in payload:
                resource_attrs = _attrs_to_dict(list(resource_metrics.resource.attributes))
                service = str(resource_attrs.get("service.name", ""))
                if service:
                    self._services.add(service)
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        self._record_metric(metric)
                        recorded += 1
            if recorded:
                self._last_ingest_ts = time.time()
                snapshot_payload = self._snapshot_locked()
        if snapshot_payload is not None:
            self._publish(snapshot_payload)
        return recorded

    def ingest_logs(self, payload: list[ResourceLogs]) -> int:
        """Decode + record a batch of ``ResourceLogs`` as event-log rows.

        Each OTLP ``LogRecord`` becomes one :class:`TelemetryEvent`: the
        body is the title, the instrumentation scope (logger) name is the
        kind, the record attributes are the attrs, and ``severity_number``
        maps to ``debug``/``info``/``warn``/``error``/``fatal`` via
        :func:`_log_level`. This is the structlog→OTel bridge
        (:mod:`openral_observability.logging`) surfacing on the UI — every
        level incl. DEBUG ships to the dashboard's ``/v1/logs`` endpoint,
        which calls this (issue #318). Records land in the same bounded
        event ring as spans/span-events; the UI defaults the Debug chip
        off so high-rate DEBUG (e.g. world_state ~30 Hz) stays opt-in and
        does not crowd the 60-event view.

        Returns the number of log records recorded.
        """
        recorded = 0
        snapshot_payload: dict[str, Any] | None = None
        with self._lock:
            for resource_logs in payload:
                resource_attrs = _attrs_to_dict(list(resource_logs.resource.attributes))
                service = str(resource_attrs.get("service.name", ""))
                if service:
                    self._services.add(service)
                for scope_logs in resource_logs.scope_logs:
                    scope_name = scope_logs.scope.name or "log"
                    for record in scope_logs.log_records:
                        self._record_log(record, scope_name)
                        recorded += 1
            if recorded:
                self._last_ingest_ts = time.time()
                snapshot_payload = self._snapshot_locked()
        if snapshot_payload is not None:
            self._publish(snapshot_payload)
        return recorded

    # ── Reader-facing API ──────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a plain-dict snapshot of the current state."""
        with self._lock:
            return self._snapshot_locked()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register an asyncio queue that receives every state update.

        The caller (the SSE endpoint) awaits ``queue.get()`` in a loop
        and must call :meth:`unsubscribe` when the client disconnects.
        Bounded at :data:`_SUBSCRIBER_QUEUE_SIZE`; if a slow client
        causes the queue to fill the oldest delta is dropped so the
        producer never blocks.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            self._subscribers.append(queue)
            self._sub_loops[id(queue)] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Drop a subscriber's queue. Safe to call twice or with an unknown queue."""
        with self._lock:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(queue)
            self._sub_loops.pop(id(queue), None)

    # ── Internal helpers ───────────────────────────────────────────────

    def _record_span(self, span: Span) -> None:
        attrs = _attrs_to_dict(list(span.attributes))
        duration_ms = _ns_to_ms(span.end_time_unix_nano - span.start_time_unix_nano)
        ts_unix = span.end_time_unix_nano / 1_000_000_000.0
        card = _SpanCard(
            name=span.name,
            ts_unix=ts_unix,
            duration_ms=duration_ms,
            attrs=attrs,
            status_code=int(span.status.code),
            status_message=span.status.message or "",
        )
        # Headline cards: latest-wins by family.
        family = _classify_span(span.name)
        if family is not None:
            self._cards[family] = card

        # Latch identity-style attrs (slow-changing config/identity values)
        # so the dashboard's Identity card shows them whatever span they
        # rode in on.
        for key in _IDENTITY_KEYS:
            if key in attrs:
                self._identity[key] = attrs[key]

        # Topical routing — by span name, populate per-topic state buckets
        # with the dynamic fields the dashboard's cards will render.
        self._update_topics(span.name, attrs, ts_unix, duration_ms)

        # Trace anchor: always remember the most-recent trace_id so the
        # dashboard can deep-link to Jaeger.
        trace_id_hex = span.trace_id.hex() if span.trace_id else ""
        if trace_id_hex:
            self._topics["trace"]["latest_trace_id"] = trace_id_hex
            self._topics["trace"]["latest_ts_unix"] = ts_unix
            # ADR-0018 F7: index full spans by trace_id for `openral replay`.
            self._index_span(span, trace_id_hex, ts_unix, attrs)

        # Always append a one-line event so the operator sees the
        # most recent activity. Severity escalates on ERROR status.
        severity = "error" if span.status.code == _STATUS_ERROR else "info"
        title = _summarise_span(span.name, attrs, duration_ms)
        self._events.append(
            TelemetryEvent(
                ts_unix=ts_unix,
                kind=span.name,
                title=title,
                attrs=attrs,
                severity=severity,
            )
        )

        # Span events (e.g. estop_requested, safety_violation) get their
        # own event log entries with elevated severity.
        for event in span.events:
            event_attrs = _attrs_to_dict(list(event.attributes))
            self._events.append(
                TelemetryEvent(
                    ts_unix=event.time_unix_nano / 1_000_000_000.0,
                    kind=event.name,
                    title=event.name,
                    attrs=event_attrs,
                    severity=_event_severity(event.name),
                )
            )
            if event.name in _COUNTED_EVENTS:
                self._counters[event.name] += 1

    def _record_log(self, record: LogRecord, scope_name: str) -> None:
        """Append one bridged OTLP ``LogRecord`` to the event ring (issue #318)."""
        attrs = _attrs_to_dict(list(record.attributes))
        ts_ns = record.time_unix_nano or record.observed_time_unix_nano
        ts_unix = ts_ns / 1_000_000_000.0 if ts_ns else time.time()
        body = _attr_value(record.body)
        title = str(body) if body is not None else scope_name
        self._events.append(
            TelemetryEvent(
                ts_unix=ts_unix,
                kind=scope_name,
                title=title,
                attrs=attrs,
                severity=_log_level(int(record.severity_number), record.severity_text),
            )
        )

    def _index_span(
        self, span: Span, trace_id_hex: str, ts_unix: float, attrs: dict[str, Any]
    ) -> None:
        """Append a span to the per-trace index, evicting the oldest trace at cap."""
        events_json: list[dict[str, Any]] = [
            {
                "name": e.name,
                "time_unix_ns": int(e.time_unix_nano),
                "attrs": _attrs_to_dict(list(e.attributes)),
            }
            for e in span.events
        ]
        indexed = _IndexedSpan(
            name=span.name,
            trace_id=trace_id_hex,
            span_id=span.span_id.hex() if span.span_id else "",
            parent_span_id=span.parent_span_id.hex() if span.parent_span_id else "",
            start_ns=int(span.start_time_unix_nano),
            end_ns=int(span.end_time_unix_nano),
            attrs=attrs,
            status_code=int(span.status.code),
            status_message=span.status.message or "",
            events=events_json,
        )
        bucket = self._spans_by_trace.get(trace_id_hex)
        if bucket is None:
            if len(self._spans_by_trace) >= _TRACE_INDEX_MAX_TRACES:
                oldest = next(iter(self._spans_by_trace))
                self._spans_by_trace.pop(oldest, None)
                self._trace_last_seen.pop(oldest, None)
            bucket = deque(maxlen=_TRACE_INDEX_MAX_SPANS)
            self._spans_by_trace[trace_id_hex] = bucket
        bucket.append(indexed)
        self._trace_last_seen[trace_id_hex] = ts_unix

    # ── F7 trace-query API ────────────────────────────────────────────

    def list_traces(self) -> list[dict[str, Any]]:
        """Return one record per indexed trace_id, most-recent first.

        Each entry is ``{trace_id, span_count, last_seen_unix}`` — the
        dashboard exposes this on ``/api/traces`` so ``openral replay`` can
        pick the right trace when the user does not pass ``--trace``.
        """
        with self._lock:
            items: list[tuple[str, deque[_IndexedSpan], float]] = [
                (tid, self._spans_by_trace[tid], self._trace_last_seen.get(tid, 0.0))
                for tid in self._spans_by_trace
            ]
        items.sort(key=lambda x: x[2], reverse=True)
        return [
            {"trace_id": tid, "span_count": len(bucket), "last_seen_unix": ts}
            for tid, bucket, ts in items
        ]

    def lookup_trace(self, trace_id: str) -> list[dict[str, Any]] | None:
        """Return every indexed span for ``trace_id`` in chronological order, or ``None``."""
        with self._lock:
            bucket = self._spans_by_trace.get(trace_id)
            if bucket is None:
                return None
            spans = sorted((s.to_json() for s in bucket), key=lambda s: s["start_unix_ns"])
        return spans

    def _update_topics(  # noqa: PLR0912  # reason: linear span-name dispatch; each arm sets a different topic slot. Splitting into per-span methods (as already done for slam.occupancy_grid / reasoner.tick) hurts the read-this-and-see-every-routed-family ergonomic that the operator-facing dashboard handlers benefit from.
        self, span_name: str, attrs: dict[str, Any], ts_unix: float, duration_ms: float
    ) -> None:
        """Route span attributes into per-topic dynamic-state buckets."""
        if span_name == "hal.read_state":
            names = attrs.get("openral.hal.joint.names")
            positions = attrs.get("openral.hal.joint.positions")
            # Only update when the span carries real joint data. Error-path spans
            # (where read_state() raised and record_joint_state never ran) close
            # without joint attributes; an unconditional update would overwrite
            # previously good names/positions with None, blanking the joint card.
            if names is not None and positions is not None:
                self._topics["robot_state"].update(
                    {
                        "ts_unix": ts_unix,
                        "duration_ms": duration_ms,
                        "names": names,
                        "positions": positions,
                        "velocities": attrs.get("openral.hal.joint.velocities"),
                        "efforts": attrs.get("openral.hal.joint.efforts"),
                        "limits_lo": attrs.get("openral.hal.joint.position_limits_lo"),
                        "limits_hi": attrs.get("openral.hal.joint.position_limits_hi"),
                        "stamp_ns": attrs.get("openral.hal.joint.stamp_ns"),
                    }
                )
        elif span_name == "hal.send_action":
            self._topics["commands"].update(
                {
                    "ts_unix": ts_unix,
                    "duration_ms": duration_ms,
                    "next_row": attrs.get("openral.hal.action.next"),
                    "dim": attrs.get("openral.hal.action.dim"),
                    "horizon": attrs.get("openral.hal.action.horizon"),
                    "applied": attrs.get("openral.hal.action.applied"),
                    "control_mode": attrs.get("openral.hal.control_mode"),
                    "gripper_position": attrs.get("openral.hal.gripper.position"),
                    "gripper_force_n": attrs.get("openral.hal.gripper.force_n"),
                }
            )
        elif span_name == "world_state.snapshot":
            # ee poses come in as `openral.hal.ee.pose.<name>` keys, flatten.
            ee_poses: dict[str, list[float]] = {}
            ee_prefix = "openral.hal.ee.pose."
            for k, v in attrs.items():
                if k.startswith(ee_prefix):
                    ee_poses[k[len(ee_prefix) :]] = list(v) if v is not None else []
            diag_keys = attrs.get("openral.world_state.diagnostics_keys") or []
            diag_vals = attrs.get("openral.world_state.diagnostics_values") or []
            diagnostics = dict(zip(diag_keys, diag_vals, strict=False))
            self._topics["world_state"].update(
                {
                    "ts_unix": ts_unix,
                    "components_stale": attrs.get("openral.world_state.components_stale"),
                    "has_latched_error": attrs.get("openral.world_state.has_latched_error"),
                    "battery_pct": attrs.get("openral.world_state.battery_pct"),
                    "ee_poses": ee_poses,
                    "diagnostics": diagnostics,
                }
            )
            # ee poses are also useful on the robot_state card.
            self._topics["robot_state"]["ee_poses"] = ee_poses
        elif span_name == "sensors.read_latest":
            source = str(attrs.get("openral.sensors.source", "unknown"))
            per_camera = self._topics["perception"].setdefault("cameras", {})
            entry: dict[str, Any] = {
                "ts_unix": ts_unix,
                "modality": attrs.get("openral.sensors.modality"),
                "encoding": attrs.get("openral.sensors.encoding"),
                "width": attrs.get("openral.sensors.width"),
                "height": attrs.get("openral.sensors.height"),
                "channels": attrs.get("openral.sensors.channels"),
                "age_ms": attrs.get("openral.sensors.age_ms"),
            }
            thumb = attrs.get("openral.sensors.thumbnail_jpeg_b64")
            if thumb:
                # Persist the thumb until a newer one arrives so the
                # card doesn't flicker between high-rate frames without
                # one and the low-rate frames that carry one.
                entry["thumbnail_jpeg_b64"] = thumb
            else:
                existing = per_camera.get(source, {})
                if "thumbnail_jpeg_b64" in existing:
                    entry["thumbnail_jpeg_b64"] = existing["thumbnail_jpeg_b64"]
            per_camera[source] = entry
        elif span_name == "slam.occupancy_grid":
            # ADR-0025 — live 2D occupancy map from slam_toolbox.
            # Bridge emits one span per /map message (1 Hz throttled
            # in `openral_runner.slam_bridge.SlamMapBridge`).
            self._topics["slam"].update(
                {
                    "ts_unix": ts_unix,
                    "frame_id": attrs.get("openral.slam.frame_id"),
                    "width": attrs.get("openral.slam.width"),
                    "height": attrs.get("openral.slam.height"),
                    "resolution_m": attrs.get("openral.slam.resolution_m"),
                    "origin_x": attrs.get("openral.slam.origin_x"),
                    "origin_y": attrs.get("openral.slam.origin_y"),
                    "png_b64": attrs.get("openral.slam.png_b64"),
                    "source_node": attrs.get("openral.slam.source_node"),
                    "robot_x": attrs.get("openral.slam.robot_x"),
                    "robot_y": attrs.get("openral.slam.robot_y"),
                    "robot_yaw": attrs.get("openral.slam.robot_yaw"),
                    "base_frame": attrs.get("openral.slam.base_frame"),
                    "footprint_radius_m": attrs.get("openral.slam.footprint_radius_m"),
                    "footprint_polygon": _reshape_xy_pairs(
                        attrs.get("openral.slam.footprint_polygon_xy")
                    ),
                }
            )
        elif span_name == "world.pointcloud":
            # ADR-0030 — robot-frame octomap pointcloud render (one span per
            # accepted cloud, throttled in
            # ``openral_runner.world_cloud_bridge.WorldCloudBridge``).
            self._topics["pointcloud"].update(
                {
                    "ts_unix": ts_unix,
                    "frame_id": attrs.get("openral.world_cloud.frame_id"),
                    "n_points": attrs.get("openral.world_cloud.n_points"),
                    "png_b64": attrs.get("openral.world_cloud.png_b64"),
                    "source_node": attrs.get("openral.world_cloud.source_node"),
                    "range_max_m": attrs.get("openral.world_cloud.range_max_m"),
                }
            )
        elif span_name == "world.scene_objects":
            # ADR-0038 — durable spatial-memory objects. One span per emit
            # (0.2 Hz from the Reasoner's preloaded map today). ``objects`` is a
            # decoded list of {id,label,x,y,z,frame_id,confidence,
            # last_seen_ns,observation_count,is_container} dicts.
            object_list = attrs.get("openral.world_state.scene_objects.list")
            self._topics["scene_objects"].update(
                {
                    "ts_unix": ts_unix,
                    "count": attrs.get("openral.world_state.scene_objects.count"),
                    "frame_id": attrs.get("openral.world_state.scene_objects.frame_id"),
                    "source_node": attrs.get("openral.world_state.scene_objects.source_node"),
                    "objects": _parse_object_list(object_list),
                }
            )
        elif span_name == "rskill.chunk_inference":
            self._topics["inference"].update(
                {
                    "ts_unix": ts_unix,
                    "duration_ms": duration_ms,
                    "kind": attrs.get("inference.kind"),
                    "chunk_index": attrs.get("inference.chunk_index"),
                    "chunk_size": attrs.get("inference.chunk_size"),
                    "engine": attrs.get("inference.engine"),
                    "device": attrs.get("inference.device"),
                }
            )
        elif span_name == "safety.check":
            check_name = attrs.get("safety.check_name") or "(unknown)"
            severity = attrs.get("safety.severity", "info")
            clamped = attrs.get("safety.clamped", False)
            ledger: dict[str, dict[str, Any]] = self._topics["safety"].setdefault("checks", {})
            ledger[str(check_name)] = {
                "ts_unix": ts_unix,
                "severity": severity,
                "clamped": clamped,
                "kernel": attrs.get("safety.kernel"),
                "duration_ms": duration_ms,
            }
            self._topics["safety"]["latest_ts_unix"] = ts_unix
        elif span_name == "reasoner.tick":
            self._record_reasoner_tick(attrs, ts_unix, duration_ms)

    def _record_reasoner_tick(
        self, attrs: dict[str, Any], ts_unix: float, duration_ms: float
    ) -> None:
        """Stash the latest ``reasoner.tick`` attributes for the dashboard card.

        ADR-0018 F4 — :meth:`openral_reasoner.ReasonerCore.tick` emits one
        of these spans per orchestrator pass via
        :func:`openral_observability.reasoner_span`. The dashboard's
        Reasoner card reads the slot this writes (the Event Log carries
        the full history; this is the "headline latest" surface so the
        operator can see what the LLM just picked).
        """
        mission_raw = attrs.get("reasoner.mission_json")
        mission: Any = None
        if isinstance(mission_raw, str):
            try:
                mission = json.loads(mission_raw)
            except json.JSONDecodeError:
                mission = None
        self._topics["reasoner"].update(
            {
                "ts_unix": ts_unix,
                "duration_ms": duration_ms,
                "tick_idx": attrs.get("reasoner.tick.idx"),
                "tool": attrs.get("reasoner.tool"),
                "rskill_id": attrs.get("reasoner.rskill_id"),
                "model": attrs.get("reasoner.model"),
                "force": attrs.get("reasoner.force"),
                "suppressed_reason": attrs.get("reasoner.suppressed_reason"),
                "error_kind": attrs.get("reasoner.error_kind"),
                "mission": mission,
            }
        )

    def _record_metric(self, metric: Metric) -> None:
        which = metric.WhichOneof("data")
        if which is None:
            return
        unit = metric.unit or ""
        name = metric.name
        ts_unix = time.time()
        if which == "histogram":
            for hp in metric.histogram.data_points:
                labels = _attrs_to_dict(list(hp.attributes))
                series = self._series(name, "histogram", unit, labels)
                avg = (hp.sum / hp.count) if hp.count else 0.0
                series.samples.append((ts_unix, avg))
                self._mirror_system_metric(name, labels, avg, ts_unix)
        elif which == "sum":
            for sp in metric.sum.data_points:
                labels = _attrs_to_dict(list(sp.attributes))
                series = self._series(name, "sum", unit, labels)
                value = sp.as_double if sp.HasField("as_double") else float(sp.as_int)
                series.cumulative = value
                series.samples.append((ts_unix, value))
                self._mirror_system_metric(name, labels, value, ts_unix)
        elif which == "gauge":
            for gp in metric.gauge.data_points:
                labels = _attrs_to_dict(list(gp.attributes))
                series = self._series(name, "gauge", unit, labels)
                value = gp.as_double if gp.HasField("as_double") else float(gp.as_int)
                series.samples.append((ts_unix, value))
                self._mirror_system_metric(name, labels, value, ts_unix)
        # exponential_histogram + summary are not emitted by OpenRAL today.

    def _mirror_system_metric(
        self, name: str, labels: dict[str, Any], value: float, ts_unix: float
    ) -> None:
        """Surface ``openral.system.*`` gauges on the System topic card."""
        if not name.startswith("openral.system."):
            return
        sys_bucket: dict[str, Any] = self._topics["system"]
        gpu_idx = labels.get("openral.system.gpu.index")
        if gpu_idx is not None:
            gpus: dict[int, dict[str, Any]] = sys_bucket.setdefault("gpus", {})
            entry = gpus.setdefault(int(gpu_idx), {})
            entry["name"] = labels.get("openral.system.gpu.name", entry.get("name", ""))
            entry["ts_unix"] = ts_unix
            if name.endswith(".memory_used_mb"):
                entry["memory_used_mb"] = value
            elif name.endswith(".memory_total_mb"):
                entry["memory_total_mb"] = value
            elif name.endswith(".utilization_pct"):
                entry["util_pct"] = value
        else:
            sys_bucket["ts_unix"] = ts_unix
            if name.endswith(".cpu.utilization_pct"):
                sys_bucket["cpu_util_pct"] = value
            elif name.endswith(".ram.used_mb"):
                sys_bucket["ram_used_mb"] = value
            elif name.endswith(".ram.total_mb"):
                sys_bucket["ram_total_mb"] = value

    def _series(self, name: str, kind: str, unit: str, labels: dict[str, Any]) -> _MetricSeries:
        key = f"{name}|" + ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        series = self._metrics.get(key)
        if series is None:
            series = _MetricSeries(name=name, kind=kind, unit=unit, labels=labels)
            self._metrics[key] = series
        return series

    def _primary_service(self) -> str:
        """Pick a stable representative ``service.name`` for the Identity card.

        A deploy graph reports several services; last-write-wins made the card
        flicker. Prefer the composite ``openral.runtime`` (owns skill
        execution), then any non-HAL node, then the lexicographically-first —
        all deterministic, so the field stops flipping mid-run.
        """
        if not self._services:
            return ""
        if "openral.runtime" in self._services:
            return "openral.runtime"
        non_hal = sorted(s for s in self._services if not s.startswith("openral.hal."))
        if non_hal:
            return non_hal[0]
        return sorted(self._services)[0]

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "service_name": self._primary_service(),
            "services": sorted(self._services),
            "run_id": self._run_id,
            "run_mode": self._run_mode,
            "git_sha": self._git_sha,
            "last_ingest_ts": self._last_ingest_ts,
            "now_unix": time.time(),
            "identity": dict(self._identity),
            "topics": _deep_copy_topics(self._topics),
            "cards": {k: v.to_json() for k, v in self._cards.items()},
            "events": [e.to_json() for e in reversed(self._events)],
            "counters": dict(self._counters),
            "metrics": [s.to_json() for s in self._metrics.values()],
        }

    def _publish(self, payload: dict[str, Any]) -> None:
        # Copy under lock; the lock is already released in callers.
        with self._lock:
            subs = list(self._subscribers)
            loops = dict(self._sub_loops)
        for q in subs:
            loop = loops.get(id(q))
            if loop is None:
                continue
            loop.call_soon_threadsafe(_offer, q, payload)


def _offer(queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
    """Best-effort enqueue: drop the oldest item if full so the producer never blocks."""
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(payload)


def _deep_copy_topics(topics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Shallow-deep copy: one level per topic so the JSON snapshot is a fresh tree."""
    out: dict[str, dict[str, Any]] = {}
    for topic, bucket in topics.items():
        # Use json round-trip to detach from store internals; bucket values
        # are plain dicts/lists/strings/numbers/floats so this is cheap.
        out[topic] = {k: _copy_nested(v) for k, v in bucket.items()}
    return out


def _copy_nested(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _copy_nested(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy_nested(x) for x in v]
    return v


# ── Classification helpers ────────────────────────────────────────────────

# Attribute keys that describe latched / configuration state ("who is
# this run?"). They get hoisted into the dashboard's Identity card
# regardless of which span carries them.
_IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "openral.run.git_sha",
        "openral.hal.adapter",
        "openral.hal.robot.model",
        "openral.hal.control_mode",
        # The policy's action-chunk horizon is identity-stable for a run;
        # it rides in on every `hal.send_action` span (producer.record_action
        # → semconv.HAL_ACTION_HORIZON) so we latch it for the Identity card.
        "openral.hal.action.horizon",
        # rSkill identity ships under both the short `rskill.*` prefix
        # (semconv.RSKILL_ID / RSKILL_ROLE, emitted by every rskill span)
        # and the namespaced `openral.rskill.*` form (future emitters);
        # latch both so whichever a span carries wins.
        "openral.rskill.id",
        "openral.rskill.revision",
        "openral.rskill.role",
        "openral.rskill.action_horizon",
        "rskill.id",
        "rskill.role",
        # Safety kernel label rides on `safety.check` spans as
        # `safety.kernel` (semconv.SAFETY_KERNEL); the C++ kernel will emit
        # "cpp", the Python NullSafetyClient emits "null".
        "safety.kernel",
        "inference.engine",
        "inference.device",
    }
)

_HEADLINE_FAMILIES: dict[str, str] = {
    "rskill.execute": "rskill_execute",
    "rskill.tick": "rskill_tick",
    "rskill.activate": "rskill_activate",
    "rskill.configure": "rskill_configure",
    "rskill.chunk_inference": "inference",
    "safety.check": "safety",
    "hal.send_action": "hal_send_action",
    "hal.read_state": "hal_read_state",
    "sensors.read_latest": "sensors_read",
    "world_state.snapshot": "world_state",
    "slam.occupancy_grid": "slam_map",
    "reasoner.tick": "reasoner_tick",
    "sim.run": "sim_run",
    "sim.step": "sim_step",
    "cli.command": "cli_command",
}

_COUNTED_EVENTS = frozenset(
    {
        "openral.event.estop_requested",
        "openral.event.safety_violation",
        "openral.event.deadline_missed",
        "openral.event.sensor_stale",
        "openral.event.action_dropped",
        # ADR-0074/0077 — Reasoner-published skill failures (vram_insufficient,
        # reward_plateau, unavailable, timeout, aborted). Tallied so the
        # dashboard's "skill failures" counter makes a failing run obvious.
        "openral.event.skill_failure",
    }
)

_ERROR_EVENTS = frozenset(
    {
        "openral.event.estop_requested",
        "openral.event.safety_violation",
        "openral.event.error_latched",
        "openral.event.skill_failure",
    }
)

_WARN_EVENTS = frozenset(
    {
        "openral.event.deadline_missed",
        "openral.event.sensor_stale",
        "openral.event.staleness_latched",
        "openral.event.action_dropped",
    }
)


def _classify_span(name: str) -> str | None:
    return _HEADLINE_FAMILIES.get(name)


def _event_severity(name: str) -> str:
    if name in _ERROR_EVENTS:
        return "error"
    if name in _WARN_EVENTS:
        return "warn"
    return "info"


# Fallback when an OTLP LogRecord carries no severity_number (0): map the
# free-text level. structlog/stdlib always sets the number, so this is
# defensive — but a malformed exporter must still bucket cleanly.
_TEXT_LEVEL_ALIASES: dict[str, str] = {
    "debug": "debug",
    "info": "info",
    "warn": "warn",
    "warning": "warn",
    "error": "error",
    "critical": "fatal",
    "fatal": "fatal",
}


def _log_level(severity_number: int, severity_text: str) -> str:
    """Map an OTLP ``SeverityNumber`` to the dashboard's event-log level.

    The UI renders five levels -- ``debug`` / ``info`` / ``warn`` /
    ``error`` / ``fatal``. OTLP severity numbers arrive in bands of four
    per level (DEBUG=5-8, INFO=9-12, ...); we collapse each band to its
    level name. TRACE (1-4) floors to ``debug`` (the lowest level the UI
    surfaces). An unset number (0) falls back to ``severity_text`` when it
    names a known level, else ``info``.
    """
    if severity_number >= _SEVERITY_FATAL_MIN:
        return "fatal"
    if severity_number >= _SEVERITY_ERROR_MIN:
        return "error"
    if severity_number >= _SEVERITY_WARN_MIN:
        return "warn"
    if severity_number >= _SEVERITY_INFO_MIN:
        return "info"
    if severity_number >= 1:  # DEBUG band (>=5) + TRACE (1-4) floor to debug.
        return "debug"
    return _TEXT_LEVEL_ALIASES.get(severity_text.strip().lower(), "info")


def _summarise_span(name: str, attrs: dict[str, Any], duration_ms: float) -> str:
    """Build a one-line label for the event log."""
    parts: list[str] = [name]
    for key in (
        "rskill.id",
        "openral.skill.id",
        "openral.hal.adapter",
        "openral.sensors.modality",
        "safety.check_name",
        "openral.tick.idx",
        "inference.chunk_index",
    ):
        if key in attrs:
            parts.append(f"{key.rsplit('.', 1)[-1]}={attrs[key]}")
    parts.append(f"{duration_ms:.1f}ms")
    return " · ".join(parts)
