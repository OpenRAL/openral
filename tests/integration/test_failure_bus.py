"""ADR-0018 F3 — integration tests for the namespaced FailureTrigger bus.

Drives the real
:class:`openral_observability.FailureBusPublisher` against the
colcon-built ``openral_msgs/msg/FailureTrigger`` IDL through ``rclpy``
and asserts:

1. Each :class:`FailureSource` publishes on its own
   ``/openral/failure/<suffix>`` topic.
2. The typed uint8 ``kind`` / ``severity`` fields are wire-correct
   against the IDL ``KIND_*`` / ``SEVERITY_*`` constants.
3. ``evidence_json`` round-trips through the Pydantic
   :data:`openral_core.FailureEvidence` discriminated union.
4. The publisher's token bucket drops excess WARN events.
5. After dropped events, a ``KIND_SUPPRESSED_SUMMARY`` roll-up appears
   at the configured cadence carrying the suppressed counts.
6. ``SEVERITY_ABORT`` is never rate-limited.

Per CLAUDE.md §1.11 / §5.4: real schemas, real IDL, real rclpy, no
mocks. The test skips with a typed reason when ROS 2 is not sourced.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


# ── Harness ──────────────────────────────────────────────────────────────────


@contextmanager
def _bus_harness(
    source: Any,
    *,
    rate_limit_hz: dict[int, float | None] | None = None,
    summary_period_s: float = 0.2,
) -> Iterator[tuple[Any, Any, Any, list[Any]]]:
    """Bring up a publisher node + helper subscriber node on one executor.

    Yields ``(executor, publisher_node, bus_publisher, received_msgs)``
    where ``received_msgs`` is appended to by a subscription on the
    topic owned by ``bus_publisher`` (i.e. ``topic_for(source)``).
    """
    import rclpy  # type: ignore[import-untyped]
    from openral_msgs.msg import FailureTrigger  # type: ignore[import-untyped]
    from openral_observability.failure_bus import FailureBusPublisher, topic_for
    from rclpy.qos import (  # type: ignore[import-untyped]
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    rclpy.init()
    pub_node = rclpy.create_node("openral_failure_bus_test_publisher")
    sub_node = rclpy.create_node("openral_failure_bus_test_subscriber")
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(pub_node)
    executor.add_node(sub_node)

    bus = FailureBusPublisher(
        pub_node,
        source,
        rate_limit_hz=rate_limit_hz,
        summary_period_s=summary_period_s,
    )
    bus.create_publisher()
    bus.start()

    received: list[Any] = []
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=50,
    )
    sub_node.create_subscription(FailureTrigger, topic_for(source), received.append, qos)

    import contextlib

    try:
        # Let pub/sub discovery settle before yielding control.
        _spin_for(executor, 0.3)
        yield executor, pub_node, bus, received
    finally:
        with contextlib.suppress(Exception):  # reason: best-effort teardown
            bus.destroy()
        executor.shutdown()
        pub_node.destroy_node()
        sub_node.destroy_node()
        rclpy.shutdown()


def _spin_for(executor: Any, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_every_source_owns_a_namespaced_topic() -> None:
    """Each ``FailureSource`` publishes on its own ``/openral/failure/<suffix>``."""
    from openral_core import ForceEvidence
    from openral_observability.failure_bus import (
        KIND_FORCE,
        SEVERITY_ABORT,
        FailureSource,
        topic_for,
    )

    seen_topics: dict[FailureSource, str] = {}
    for source in FailureSource:
        with _bus_harness(source) as (executor, _pub_node, bus, received):
            ok = bus.publish(
                kind=KIND_FORCE,
                severity=SEVERITY_ABORT,
                evidence=ForceEvidence(joint_or_ee="ee", measured_n=12.0, limit_n=10.0),
                rskill_id="openral/test-skill",
            )
            assert ok, f"publish on {topic_for(source)} returned False"
            _spin_for(executor, 0.5)
            assert received, (
                f"no FailureTrigger landed on {topic_for(source)} for source={source.value}"
            )
            seen_topics[source] = bus.topic
            assert bus.topic == topic_for(source)

    # All six topics must be distinct.
    assert len(set(seen_topics.values())) == len(FailureSource)


def test_publish_payload_is_typed_uint8_and_evidence_round_trips() -> None:
    """uint8 ``kind`` / ``severity`` are wire-correct; ``evidence_json`` decodes."""
    from openral_core import FailureEvidence, ForceEvidence
    from openral_msgs.msg import FailureTrigger  # type: ignore[import-untyped]
    from openral_observability.failure_bus import (
        KIND_FORCE,
        SEVERITY_ABORT,
        FailureSource,
    )
    from pydantic import TypeAdapter

    adapter: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)
    evidence = ForceEvidence(joint_or_ee="ee", measured_n=15.7, limit_n=10.0)

    with _bus_harness(FailureSource.SAFETY) as (executor, _pub_node, bus, received):
        ok = bus.publish(
            kind=KIND_FORCE,
            severity=SEVERITY_ABORT,
            evidence=evidence,
            rskill_id="openral/test-skill",
        )
        assert ok
        _spin_for(executor, 0.5)
        assert received, "no FailureTrigger arrived"
        msg = received[0]

    # IDL-side constants — wire-correct uint8 values.
    assert msg.kind == FailureTrigger.KIND_FORCE == KIND_FORCE
    assert msg.severity == FailureTrigger.SEVERITY_ABORT == SEVERITY_ABORT
    assert msg.rskill_id == "openral/test-skill"
    assert msg.header.frame_id == FailureSource.SAFETY.value

    # Pydantic discriminator must decode back to the same variant.
    back = adapter.validate_json(msg.evidence_json)
    assert isinstance(back, ForceEvidence)
    assert back == evidence


def test_warn_severity_token_bucket_drops_excess_and_summary_publishes() -> None:
    """Burst of WARN events is rate-limited; a SUPPRESSED_SUMMARY rolls them up."""
    from openral_core import FailureEvidence, SuppressedSummaryEvidence, TimeoutEvidence
    from openral_msgs.msg import FailureTrigger  # type: ignore[import-untyped]
    from openral_observability.failure_bus import (
        KIND_SUPPRESSED_SUMMARY,
        KIND_TIMEOUT,
        SEVERITY_WARN,
        FailureSource,
    )
    from pydantic import TypeAdapter

    adapter: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)

    # Tight bucket so the test stays fast: 2/s on WARN, summary at 5 Hz.
    rate_limit = {SEVERITY_WARN: 2.0}
    summary_period = 0.2

    with _bus_harness(
        FailureSource.HAL,
        rate_limit_hz=rate_limit,
        summary_period_s=summary_period,
    ) as (executor, _pub_node, bus, received):
        # Burst 20 WARN events in rapid succession. At 2/s with a
        # capacity-of-1 bucket the publisher accepts at most ~1 of
        # them in this window, so >=15 must drop.
        results = []
        for i in range(20):
            results.append(
                bus.publish(
                    kind=KIND_TIMEOUT,
                    severity=SEVERITY_WARN,
                    evidence=TimeoutEvidence(
                        operation=f"op_{i}",
                        deadline_s=0.05,
                        elapsed_s=0.06,
                    ),
                )
            )
        n_accepted = sum(results)
        n_dropped = len(results) - n_accepted
        assert n_dropped >= 15, (
            f"expected ≥15 drops at rate=2/s, got accepted={n_accepted}, dropped={n_dropped}"
        )

        # Wait for the summary timer to fire ≥1×.
        _spin_for(executor, summary_period * 4)

    real_warn_msgs = [m for m in received if m.kind == KIND_TIMEOUT]
    summary_msgs = [m for m in received if m.kind == KIND_SUPPRESSED_SUMMARY]

    assert real_warn_msgs, "expected at least one WARN to make it through the bucket"
    assert len(real_warn_msgs) == n_accepted, (
        f"accepted {n_accepted} but {len(real_warn_msgs)} arrived on the wire"
    )
    assert summary_msgs, "expected at least one SUPPRESSED_SUMMARY roll-up"

    # The summary must decode to SuppressedSummaryEvidence and account
    # for at least the dropped events from this run.
    total_summarized = 0
    for s in summary_msgs:
        assert s.kind == FailureTrigger.KIND_SUPPRESSED_SUMMARY
        decoded = adapter.validate_json(s.evidence_json)
        assert isinstance(decoded, SuppressedSummaryEvidence)
        # Every entry in the summary must reference our (KIND_TIMEOUT, SEVERITY_WARN) bucket.
        for k, sev in zip(decoded.kinds, decoded.severities, strict=True):
            assert k == KIND_TIMEOUT
            assert sev == SEVERITY_WARN
        total_summarized += sum(decoded.counts)
    assert total_summarized >= n_dropped, (
        f"summary counts ({total_summarized}) < observed drops ({n_dropped})"
    )


def test_abort_severity_is_never_rate_limited() -> None:
    """A burst of SEVERITY_ABORT must all be published — no bucket on ABORT."""
    from openral_core import ForceEvidence
    from openral_observability.failure_bus import (
        KIND_FORCE,
        SEVERITY_ABORT,
        FailureSource,
    )

    n_burst = 50
    with _bus_harness(FailureSource.SAFETY) as (executor, _pub_node, bus, received):
        for i in range(n_burst):
            assert bus.publish(
                kind=KIND_FORCE,
                severity=SEVERITY_ABORT,
                evidence=ForceEvidence(
                    joint_or_ee=f"ee_{i}",
                    measured_n=11.0 + i,
                    limit_n=10.0,
                ),
            ), f"ABORT publish #{i} was suppressed — bucket leaked onto ABORT"
        _spin_for(executor, 0.5)

    abort_msgs = [m for m in received if m.kind == KIND_FORCE]
    assert len(abort_msgs) == n_burst, (
        f"expected all {n_burst} ABORT publications on the wire, got {len(abort_msgs)}"
    )
