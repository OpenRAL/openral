"""ADR-0018 F3 — publisher helper for the namespaced FailureTrigger bus.

The OpenRAL graph publishes typed :class:`openral_msgs.msg.FailureTrigger`
events on six layer-namespaced topics::

    /openral/failure/hal
    /openral/failure/sensor
    /openral/failure/rskill
    /openral/failure/safety
    /openral/failure/wam
    /openral/failure/critic

(The ``rskill`` suffix replaced the original ``skill`` on 2026-05-25 —
ADR-0018 amendment §5 — for consistency with the carried ``rskill_id``
field on :class:`openral_msgs.msg.FailureTrigger`.)

One source layer per topic, one topic per source layer; the reasoner
(``openral_reasoner``, planned in F4) subscribes to the relevant subset
and ``rqt_graph`` shows source provenance directly.

This module ships:

- :class:`FailureSource` — enum mapping a source layer to a topic suffix.
- :func:`topic_for` — pure helper, ``FailureSource → /openral/failure/<suffix>``.
- ``KIND_*`` / ``SEVERITY_*`` constants — mirror the IDL constants on
  ``openral_msgs/msg/FailureTrigger`` so callers can construct events
  without importing the generated IDL (handy for unit tests).
- :class:`FailureBusPublisher` — opens a ROS 2 publisher on the source's
  topic, rate-limits ``(kind, severity)`` buckets with a token bucket,
  emits a 1 Hz ``KIND_SUPPRESSED_SUMMARY`` roll-up when buckets dropped.

The publisher class is import-safe on hosts without ``rclpy``: it
deferred-imports rclpy + the generated IDL inside the methods that
actually touch the ROS layer, so unit tests can exercise the token
bucket and constants without a sourced ROS install.

Example:
    >>> from openral_observability.failure_bus import (
    ...     FailureSource,
    ...     KIND_FORCE,
    ...     SEVERITY_ABORT,
    ...     topic_for,
    ... )
    >>> topic_for(FailureSource.SAFETY)
    '/openral/failure/safety'
    >>> KIND_FORCE, SEVERITY_ABORT
    (1, 3)
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from openral_core import FailureEvidence, SuppressedSummaryEvidence

if TYPE_CHECKING:
    from openral_observability import propagation as _propagation  # noqa: F401

__all__ = [
    "DEFAULT_RATE_LIMIT_HZ",
    "DEFAULT_SUMMARY_PERIOD_S",
    "KIND_CONTROLLER",
    "KIND_CRITIC",
    "KIND_FORCE",
    "KIND_HUMAN",
    "KIND_PERCEPTION",
    "KIND_REASONER_TIMEOUT",
    "KIND_SELFVERIFY",
    "KIND_SUPPRESSED_SUMMARY",
    "KIND_TIMEOUT",
    "KIND_WAM",
    "KIND_WORKSPACE",
    "SEVERITY_ABORT",
    "SEVERITY_FAIL",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "TOPIC_PREFIX",
    "FailureBusPublisher",
    "FailureSource",
    "topic_for",
]


# ─── IDL-mirror constants (ADR-0018 §2) ────────────────────────────────────────
#
# These mirror ``openral_msgs/msg/FailureTrigger`` so callers can write
# typed event publications without depending on the generated IDL —
# useful for unit tests and for code paths that may run without a
# sourced ROS install (``openral`` CLI, sim runner, fakes).
#
# When the IDL changes, **bump both**.

KIND_TIMEOUT: int = 0
KIND_FORCE: int = 1
KIND_WORKSPACE: int = 2
KIND_PERCEPTION: int = 3
KIND_CRITIC: int = 4
KIND_CONTROLLER: int = 5
KIND_SELFVERIFY: int = 6
KIND_HUMAN: int = 7
KIND_WAM: int = 8
KIND_REASONER_TIMEOUT: int = 9
KIND_SUPPRESSED_SUMMARY: int = 254

SEVERITY_INFO: int = 0
SEVERITY_WARN: int = 1
SEVERITY_FAIL: int = 2
SEVERITY_ABORT: int = 3

TOPIC_PREFIX: str = "/openral/failure"


class FailureSource(str, Enum):
    """One source layer per topic on the failure bus.

    The string value is the topic suffix appended to
    :data:`TOPIC_PREFIX`. Use :func:`topic_for` to get the full topic.
    """

    HAL = "hal"
    SENSOR = "sensor"
    RSKILL = "rskill"
    SAFETY = "safety"
    WAM = "wam"
    CRITIC = "critic"


def topic_for(source: FailureSource) -> str:
    """Return the namespaced topic for a source layer.

    Args:
        source: The layer publishing the trigger.

    Returns:
        Full topic name (e.g. ``"/openral/failure/safety"``).
    """
    return f"{TOPIC_PREFIX}/{source.value}"


# ─── Rate-limit defaults ───────────────────────────────────────────────────────
#
# Per ADR-0018 / capability review F3: token-bucket rate limit per
# (kind, severity); WARN defaults to 10/s, ABORT is never limited.
# INFO defaults to 10/s as well (log spam guard). FAIL is unlimited
# because FAIL events are rare and must always reach the reasoner.
#
# Override per-publisher via ``FailureBusPublisher(rate_limit_hz=...)``.

DEFAULT_RATE_LIMIT_HZ: dict[int, float | None] = {
    SEVERITY_INFO: 10.0,
    SEVERITY_WARN: 10.0,
    SEVERITY_FAIL: None,
    SEVERITY_ABORT: None,
}

DEFAULT_SUMMARY_PERIOD_S: float = 1.0


class _TokenBucket:
    """Per-bucket token bucket — caps :meth:`try_consume` at ``rate_hz``/s.

    Bucket capacity is exactly one token. On every :meth:`try_consume`
    call, accumulated tokens (``rate_hz * elapsed``) are clamped to the
    capacity ceiling. A bucket constructed with ``rate_hz=None`` is
    unlimited (``try_consume`` always returns ``True``).
    """

    __slots__ = ("_capacity", "_clock", "_last_ns", "_lock", "_rate_hz", "_tokens")

    def __init__(
        self,
        rate_hz: float | None,
        *,
        capacity: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a bucket.

        Args:
            rate_hz: Token regeneration rate (Hz). ``None`` disables
                rate-limiting (every call succeeds).
            capacity: Maximum stored tokens. Default ``1.0`` — burst of
                one event per bucket per regeneration window.
            clock: Monotonic time source in seconds. Override in tests.
        """
        self._rate_hz = rate_hz
        self._capacity = capacity
        self._clock = clock
        self._tokens = capacity
        self._last_ns = clock()
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        """Return ``True`` if a token was consumed, ``False`` if dropped."""
        if self._rate_hz is None:
            return True
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._last_ns)
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_hz)
            self._last_ns = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class FailureBusPublisher:
    """ROS 2 publisher for :class:`openral_msgs.msg.FailureTrigger`.

    One per ``(node, source)`` pair. Owns the publisher on
    :func:`topic_for(source) <topic_for>`, per-``(kind, severity)``
    token buckets, and the periodic ``KIND_SUPPRESSED_SUMMARY`` timer
    that emits a roll-up of dropped events.

    The publisher class is import-safe on hosts without ``rclpy``; the
    generated IDL and rclpy primitives are imported lazily inside
    :meth:`create_publisher` and :meth:`publish`.

    Example:
        >>> # Real usage requires a sourced ROS 2 install; see
        >>> # tests/integration/test_failure_bus.py for an end-to-end run.
        >>> from openral_observability.failure_bus import FailureSource
        >>> FailureSource.HAL.value
        'hal'
    """

    def __init__(
        self,
        node: object,
        source: FailureSource,
        *,
        rate_limit_hz: dict[int, float | None] | None = None,
        summary_period_s: float = DEFAULT_SUMMARY_PERIOD_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure the publisher; does not open the ROS publisher.

        Call :meth:`create_publisher` from ``on_configure`` to open the
        ROS publisher, and :meth:`start` from ``on_activate`` to start
        the suppressed-summary timer.

        Args:
            node: The owning ``rclpy.node.Node`` (or
                ``rclpy.lifecycle.LifecycleNode``).
            source: Which layer this publisher represents.
            rate_limit_hz: Per-severity rate override. Defaults to
                :data:`DEFAULT_RATE_LIMIT_HZ`. ``None`` value disables
                rate-limiting for that severity.
            summary_period_s: Cadence of the
                ``KIND_SUPPRESSED_SUMMARY`` roll-up. Default 1.0 s.
            clock: Monotonic clock for the token buckets and the
                summary window. Defaults to :func:`time.monotonic`.
                Override in tests.
        """
        self._node = node
        self._source = source
        self._topic = topic_for(source)
        self._rate_limit_hz = {**DEFAULT_RATE_LIMIT_HZ, **(rate_limit_hz or {})}
        self._summary_period_s = summary_period_s
        self._clock = clock or time.monotonic

        self._publisher: object | None = None
        self._summary_timer: object | None = None
        self._suppressed_counter: Counter[tuple[int, int]] = Counter()
        self._counter_lock = threading.Lock()
        self._buckets: dict[tuple[int, int], _TokenBucket] = {}
        self._buckets_lock = threading.Lock()

    @property
    def topic(self) -> str:
        """The full topic this publisher writes to."""
        return self._topic

    @property
    def source(self) -> FailureSource:
        """The source layer this publisher represents."""
        return self._source

    def create_publisher(self) -> None:
        """Open the ROS 2 publisher with the bus QoS profile.

        QoS per ADR-0018 §1: ``RELIABLE+VOLATILE+KEEP_LAST=50`` (deep
        history so a slow consumer doesn't drop events).
        """
        from openral_msgs.msg import (  # type: ignore[import-not-found,unused-ignore]  # reason: rclpy-generated module
            FailureTrigger,
        )
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )
        self._publisher = self._node.create_publisher(FailureTrigger, self._topic, qos)  # type: ignore[attr-defined]

    def start(self) -> None:
        """Start the 1 Hz ``KIND_SUPPRESSED_SUMMARY`` roll-up timer.

        Must be called from the owning node's ``on_activate``. No-op
        when :meth:`create_publisher` has not run.
        """
        if self._publisher is None or self._summary_timer is not None:
            return
        self._summary_timer = self._node.create_timer(  # type: ignore[attr-defined]
            self._summary_period_s,
            self._emit_summary_if_any,
        )

    def stop(self) -> None:
        """Cancel the suppressed-summary timer (``on_deactivate``)."""
        if self._summary_timer is not None:
            self._summary_timer.cancel()  # type: ignore[attr-defined]  # reason: object|None typed; rclpy Timer.cancel() at runtime
            self._summary_timer = None

    def destroy(self) -> None:
        """Tear down the publisher + timer (``on_cleanup``)."""
        self.stop()
        if self._publisher is not None:
            self._node.destroy_publisher(self._publisher)  # type: ignore[attr-defined]
            self._publisher = None
        with self._counter_lock:
            self._suppressed_counter.clear()
        with self._buckets_lock:
            self._buckets.clear()

    def publish(
        self,
        *,
        kind: int,
        severity: int,
        evidence: FailureEvidence,
        rskill_id: str = "",
        trace_id: str | None = None,
    ) -> bool:
        """Emit one :class:`FailureTrigger`, subject to rate-limiting.

        Args:
            kind: One of the ``KIND_*`` constants.
            severity: One of the ``SEVERITY_*`` constants.
            evidence: Typed :data:`openral_core.FailureEvidence` payload
                — must match the ``kind`` (e.g. ``KIND_FORCE`` →
                :class:`ForceEvidence`). Not enforced at runtime so the
                publisher stays cheap; consumers validate via the
                Pydantic discriminator.
            rskill_id: Identifier of the running skill, when known.
            trace_id: Override for the W3C ``traceparent`` field. When
                ``None`` (default) the active OTel context is used via
                :func:`openral_observability.propagation.current_traceparent`.

        Returns:
            ``True`` if the event was published, ``False`` if the
            ``(kind, severity)`` bucket was empty and the event was
            suppressed. Suppressed events accumulate into the next
            ``KIND_SUPPRESSED_SUMMARY`` roll-up.
        """
        if self._publisher is None:
            return False
        bucket = self._bucket_for(kind, severity)
        if not bucket.try_consume():
            with self._counter_lock:
                self._suppressed_counter[kind, severity] += 1
            return False
        self._emit(
            kind=kind,
            severity=severity,
            evidence_json=evidence.model_dump_json(),
            rskill_id=rskill_id,
            trace_id=trace_id,
        )
        return True

    # ── Internals ─────────────────────────────────────────────────────────────

    def _bucket_for(self, kind: int, severity: int) -> _TokenBucket:
        """Lazily allocate a token bucket for one ``(kind, severity)`` pair."""
        key = (kind, severity)
        with self._buckets_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(self._rate_limit_hz.get(severity), clock=self._clock)
                self._buckets[key] = bucket
            return bucket

    def _emit_summary_if_any(self) -> None:
        """Timer callback: if any events were dropped, publish a summary.

        The summary is itself a :class:`FailureTrigger` with
        ``kind=KIND_SUPPRESSED_SUMMARY`` and
        ``severity=SEVERITY_WARN``; the payload's
        :class:`SuppressedSummaryEvidence` carries parallel arrays of
        the suppressed ``(kind, severity)`` pairs and their counts.
        """
        with self._counter_lock:
            if not self._suppressed_counter:
                return
            snapshot = dict(self._suppressed_counter)
            self._suppressed_counter.clear()
        kinds = [k for (k, _s) in snapshot]
        severities = [s for (_k, s) in snapshot]
        counts = [snapshot[k] for k in snapshot]
        evidence = SuppressedSummaryEvidence(
            window_s=self._summary_period_s,
            kinds=kinds,
            severities=severities,
            counts=counts,
        )
        # Bypass the token bucket: the summary itself must never be dropped.
        self._emit(
            kind=KIND_SUPPRESSED_SUMMARY,
            severity=SEVERITY_WARN,
            evidence_json=evidence.model_dump_json(),
            rskill_id="",
            trace_id=None,
        )

    def _emit(
        self,
        *,
        kind: int,
        severity: int,
        evidence_json: str,
        rskill_id: str,
        trace_id: str | None,
    ) -> None:
        """Build the IDL message and publish (no rate-limit check)."""
        from openral_msgs.msg import FailureTrigger

        from openral_observability import propagation

        msg = FailureTrigger()
        msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[attr-defined]
        msg.header.frame_id = self._source.value
        msg.kind = int(kind)
        msg.severity = int(severity)
        msg.evidence_json = evidence_json
        msg.rskill_id = rskill_id
        if trace_id is None:
            trace_id = propagation.current_traceparent() or ""
        msg.trace_id = trace_id
        self._publisher.publish(msg)  # type: ignore[union-attr]
