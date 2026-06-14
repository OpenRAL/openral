"""Integration tests for the World State ROS 2 lifecycle node.

These tests drive the real
``openral_world_state_ros.lifecycle_node._WorldStateLifecycleNode``
through ``rclpy`` (the ``launch_testing``-equivalent in-process pattern
that ``test_lifecycle_node_launch`` established) and verify the
ADR-0018 F2 contract at the integration boundary: the typed
``WorldStateStamped`` publication on the fast (30 Hz) and slow (5 Hz)
topics, QoS profiles, lifecycle transitions, and the ``/joint_states``
→ aggregator → typed message round-trip.

The five scenarios below mirror the structure of the older JSON-based
integration tests (the JSON ``/world_state`` topic is removed by ADR-0018
F2 — typed is the only path now):

1. **Fast/slow rate ratio** — drive ``/joint_states`` at 30 Hz; assert
   the fast topic publishes ≥6× as often as the slow topic over a 2 s
   window (matches the ``round(30/5) = 6`` divider).
2. **30 Hz pipeline → DIAG_OK** — drive at 30 Hz; the latest fast
   message reports ``DIAG_OK`` for ``joint_state`` in the parallel
   diagnostic arrays.
3. **Joint-state dropout** → ``DIAG_STALE`` within one staleness
   window.
4. **Recovery** → ``DIAG_OK`` again once fresh updates resume.
5. **High-load consistency** — 8 concurrent publishers; every typed
   snapshot's ``joint_state.position`` length is internally consistent.

All tests skip if ROS 2 is not sourced. CI runs them in the
``hal-integration`` job (``.github/workflows/hal.yml``) which colcon-builds
``openral_msgs`` and ``openral_world_state`` first.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


# ── Test harness ─────────────────────────────────────────────────────────────


@contextmanager
def _lifecycle_harness(
    *,
    publish_rate_hz_fast: float = 30.0,
    publish_rate_hz_slow: float = 5.0,
    staleness_limit_s: float = 0.1,
) -> Iterator[tuple[Any, Any, Any, list[Any], list[Any]]]:
    """Bring up a ``_WorldStateLifecycleNode`` + helper publisher node.

    Yields ``(executor, helper_node, joint_pub, fast_msgs, slow_msgs)``.
    The ``fast_msgs`` / ``slow_msgs`` lists are appended to by
    subscriptions on the two ADR-0018 F2 topics. Cleanly tears down on
    exit.
    """
    import rclpy  # type: ignore[import-untyped]
    from openral_msgs.msg import WorldStateStamped  # type: ignore[import-untyped]
    from openral_world_state_ros.lifecycle_node import (
        TOPIC_FAST,
        TOPIC_SLOW,
        _WorldStateLifecycleNode,
    )
    from rclpy.lifecycle import TransitionCallbackReturn  # type: ignore[import-untyped]
    from rclpy.qos import (  # type: ignore[import-untyped]
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import JointState as RosJointState  # type: ignore[import-untyped]

    rclpy.init()
    node = _WorldStateLifecycleNode()
    node.set_parameters(
        [
            rclpy.parameter.Parameter("publish_rate_hz_fast", value=publish_rate_hz_fast),
            rclpy.parameter.Parameter("publish_rate_hz_slow", value=publish_rate_hz_slow),
            rclpy.parameter.Parameter("staleness_limit_s", value=staleness_limit_s),
        ]
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    helper = rclpy.create_node("test_world_state_helper")
    executor.add_node(helper)
    joint_pub = helper.create_publisher(RosJointState, "/joint_states", 1)
    fast_msgs: list[Any] = []
    slow_msgs: list[Any] = []
    ws_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    helper.create_subscription(WorldStateStamped, TOPIC_FAST, fast_msgs.append, ws_qos)
    helper.create_subscription(WorldStateStamped, TOPIC_SLOW, slow_msgs.append, ws_qos)

    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield executor, helper, joint_pub, fast_msgs, slow_msgs
    finally:
        try:
            node.trigger_deactivate()
            node.trigger_cleanup()
        except Exception:
            pass
        executor.remove_node(helper)
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def _spin_for(executor: Any, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


def _make_joint_msg(helper: Any, i: int, n_joints: int = 6) -> Any:
    from sensor_msgs.msg import JointState as RosJointState  # type: ignore[import-untyped]

    msg = RosJointState()
    msg.header.stamp = helper.get_clock().now().to_msg()
    msg.name = [f"j{k}" for k in range(n_joints)]
    msg.position = [float(i) * 0.01] * n_joints
    msg.velocity = [0.0] * n_joints
    msg.effort = [0.0] * n_joints
    return msg


def _joint_diag_status(msg: Any) -> int | None:
    """Return the ``joint_state`` diagnostic status uint8 (or None)."""
    for key, status in zip(msg.diagnostic_keys, msg.diagnostic_statuses, strict=True):
        if key == "joint_state":
            return int(status)
    return None


# ── Scenario 1: Fast/slow rate ratio (ADR-0018 F2) ───────────────────────────


def test_fast_topic_publishes_six_times_per_slow_topic() -> None:
    """Fast 30 Hz / slow 5 Hz → fast must publish ≥5× as often as slow.

    The lifecycle node uses ``round(fast_hz / slow_hz)`` as a tick
    divider, so the exact ratio is 6:1 in steady state — we assert
    ≥5 to leave headroom for the first tick boundary.
    """
    with _lifecycle_harness(
        publish_rate_hz_fast=30.0,
        publish_rate_hz_slow=5.0,
        staleness_limit_s=1.0,
    ) as (
        executor,
        helper,
        joint_pub,
        fast_msgs,
        slow_msgs,
    ):
        stop = threading.Event()

        def _publish() -> None:
            i = 0
            while not stop.is_set():
                joint_pub.publish(_make_joint_msg(helper, i, n_joints=1))
                i += 1
                time.sleep(1 / 30)

        t = threading.Thread(target=_publish, daemon=True)
        t.start()
        try:
            _spin_for(executor, 2.0)
        finally:
            stop.set()
            t.join(timeout=1.0)

        assert len(fast_msgs) >= 28, f"Expected ≥28 fast messages in 2 s, got {len(fast_msgs)}"
        assert len(slow_msgs) >= 5, f"Expected ≥5 slow messages in 2 s, got {len(slow_msgs)}"
        ratio = len(fast_msgs) / max(len(slow_msgs), 1)
        assert ratio >= 5.0, (
            f"Fast/slow ratio {ratio:.2f} below 5.0 — divider broken "
            f"({len(fast_msgs)} fast vs {len(slow_msgs)} slow)"
        )


# ── Scenario 2: 30 Hz pipeline diagnostics ok ────────────────────────────────


def test_30hz_pipeline_diagnostics_ok() -> None:
    """Publish /joint_states at 30 Hz → fast topic carries DIAG_OK for joint_state."""
    from openral_msgs.msg import WorldStateStamped  # type: ignore[import-untyped]

    with _lifecycle_harness(
        publish_rate_hz_fast=30.0,
        publish_rate_hz_slow=5.0,
        staleness_limit_s=1.0,
    ) as (
        executor,
        helper,
        joint_pub,
        fast_msgs,
        _slow_msgs,
    ):
        stop = threading.Event()

        def _publish() -> None:
            i = 0
            while not stop.is_set():
                joint_pub.publish(_make_joint_msg(helper, i, n_joints=1))
                i += 1
                time.sleep(1 / 30)

        t = threading.Thread(target=_publish, daemon=True)
        t.start()
        try:
            _spin_for(executor, 2.0)
        finally:
            stop.set()
            t.join(timeout=1.0)

        assert len(fast_msgs) >= 28, f"Expected ≥28 fast messages in 2 s, got {len(fast_msgs)}"
        last_status = _joint_diag_status(fast_msgs[-1])
        assert last_status == WorldStateStamped.DIAG_OK, (
            f"Expected DIAG_OK joint_state, got status={last_status} "
            f"(keys={list(fast_msgs[-1].diagnostic_keys)})"
        )


# ── Scenario 3: Joint-state dropout latches stale ────────────────────────────


def test_joint_state_dropout_latches_stale() -> None:
    """Stop the publisher → diagnostic flips to DIAG_STALE within ~1 staleness window."""
    from openral_msgs.msg import WorldStateStamped  # type: ignore[import-untyped]

    staleness = 0.1
    with _lifecycle_harness(
        publish_rate_hz_fast=30.0,
        publish_rate_hz_slow=5.0,
        staleness_limit_s=staleness,
    ) as (
        executor,
        helper,
        joint_pub,
        fast_msgs,
        _slow_msgs,
    ):
        # Warm-up: publish briefly so the diagnostic becomes OK.
        for i in range(15):
            joint_pub.publish(_make_joint_msg(helper, i))
            _spin_for(executor, 1 / 30)

        assert any(_joint_diag_status(m) == WorldStateStamped.DIAG_OK for m in fast_msgs), (
            "Expected at least one DIAG_OK snapshot during warm-up"
        )

        # Drop out: stop publishing for >2 staleness windows.
        fast_msgs.clear()
        _spin_for(executor, staleness * 3)

        assert fast_msgs, "No fast snapshots received during dropout window"
        last_status = _joint_diag_status(fast_msgs[-1])
        assert last_status == WorldStateStamped.DIAG_STALE, (
            f"Expected DIAG_STALE after {staleness * 3:.2f} s of silence, got status={last_status}"
        )


# ── Scenario 4: Recovery restores ok ─────────────────────────────────────────


def test_joint_state_recovery_restores_ok() -> None:
    """After dropout, fresh updates clear DIAG_STALE back to DIAG_OK."""
    from openral_msgs.msg import WorldStateStamped  # type: ignore[import-untyped]

    staleness = 0.1
    with _lifecycle_harness(
        publish_rate_hz_fast=30.0,
        publish_rate_hz_slow=5.0,
        staleness_limit_s=staleness,
    ) as (
        executor,
        helper,
        joint_pub,
        fast_msgs,
        _slow_msgs,
    ):
        # Warm-up.
        for i in range(15):
            joint_pub.publish(_make_joint_msg(helper, i))
            _spin_for(executor, 1 / 30)

        # Drop out.
        _spin_for(executor, staleness * 3)
        fast_msgs.clear()

        # Recover: resume publishing for ~0.5 s.
        stop = threading.Event()

        def _publish() -> None:
            i = 0
            while not stop.is_set():
                joint_pub.publish(_make_joint_msg(helper, i))
                i += 1
                time.sleep(1 / 30)

        t = threading.Thread(target=_publish, daemon=True)
        t.start()
        try:
            _spin_for(executor, 0.5)
        finally:
            stop.set()
            t.join(timeout=1.0)

        statuses = [_joint_diag_status(m) for m in fast_msgs]
        assert WorldStateStamped.DIAG_OK in statuses, (
            f"Expected diagnostic to recover to DIAG_OK after publisher resumed; "
            f"saw only {statuses}"
        )


# ── Scenario 5: High-load consistency ────────────────────────────────────────


def test_high_load_snapshot_consistency() -> None:
    """8 concurrent publishers → every typed snapshot is internally consistent."""
    n_joints = 6
    n_writers = 8
    writes_per_writer = 200

    with _lifecycle_harness(
        publish_rate_hz_fast=60.0,
        publish_rate_hz_slow=5.0,
        staleness_limit_s=10.0,
    ) as (
        executor,
        helper,
        joint_pub,
        fast_msgs,
        _slow_msgs,
    ):
        errors: list[Exception] = []

        def _writer(idx: int) -> None:
            try:
                for i in range(writes_per_writer):
                    joint_pub.publish(_make_joint_msg(helper, i + idx, n_joints=n_joints))
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_writer, args=(idx,), daemon=True) for idx in range(n_writers)
        ]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            executor.spin_once(timeout_sec=0.02)
        for t in threads:
            t.join(timeout=2.0)

        # Drain remaining timer ticks.
        _spin_for(executor, 0.2)

        assert errors == [], f"Writer thread errors: {errors}"
        assert fast_msgs, "No fast snapshots received under load"

        for msg in fast_msgs:
            positions = list(msg.joint_state.position)
            # Either the joint state was never populated (empty default) or
            # it carries the expected width — never a torn intermediate.
            assert len(positions) in (0, n_joints), (
                f"Inconsistent joint state length: {len(positions)} (expected 0 or {n_joints})"
            )
            # Diagnostic and staleness arrays are always parallel.
            assert len(msg.diagnostic_keys) == len(msg.diagnostic_statuses), (
                "diagnostic_keys / diagnostic_statuses lengths diverged"
            )
            assert len(msg.staleness_keys) == len(msg.staleness_ms), (
                "staleness_keys / staleness_ms lengths diverged"
            )


# ── Scenario 6: Camera image → sensors.read_latest span (dashboard wiring) ───


@pytest.fixture
def captured_spans() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory OTel tracer + exporter and return the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def test_on_image_emits_sensors_read_latest_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """An RGB Image on /openral/cameras/top/image emits a ``sensors.read_latest`` span.

    The dashboard's Perception card subscribes to this span family;
    without it, the card stays in `waiting for sensors.read_latest`.
    """
    import rclpy
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image as RosImage

    rclpy.init()
    node = _WorldStateLifecycleNode()
    node.set_parameters(
        [
            rclpy.parameter.Parameter("camera_names", value=["top"]),
            rclpy.parameter.Parameter("publish_rate_hz_fast", value=30.0),
            rclpy.parameter.Parameter("publish_rate_hz_slow", value=5.0),
        ]
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    helper = rclpy.create_node("test_world_state_image_helper")
    executor.add_node(helper)
    image_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    pub = helper.create_publisher(RosImage, "/openral/cameras/top/image", image_qos)

    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        # Drive a single 32x24 RGB frame; small enough to keep the test
        # fast yet large enough for `encode_frame_thumbnail` to produce
        # a real JPEG.
        msg = RosImage()
        msg.header.stamp = helper.get_clock().now().to_msg()
        msg.header.frame_id = "openral_camera_top"
        msg.height = 24
        msg.width = 32
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        msg.data = bytes([128] * (msg.height * msg.width * 3))
        pub.publish(msg)
        _spin_for(executor, 0.3)
    finally:
        try:
            node.trigger_deactivate()
            node.trigger_cleanup()
        except Exception:
            pass
        executor.remove_node(helper)
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "sensors.read_latest"]
    assert spans, "world_state never emitted a sensors.read_latest span on incoming Image"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.sensors.source") == "top"
    assert attrs.get("openral.sensors.modality") == "rgb"
    assert attrs.get("openral.sensors.encoding") == "rgb8"
    assert attrs.get("openral.sensors.width") == 32
    assert attrs.get("openral.sensors.height") == 24
    assert attrs.get("openral.sensors.channels") == 3
    assert attrs.get("openral.sensors.age_ms") is not None
