"""Lifecycle smoke test for ``openral_hal_ur5e``.

Drives the standard managed-lifecycle transition path against the generic
``_HALLifecycleNode`` from :mod:`openral_hal.lifecycle` using the real
``UR5eHAL`` factory.  The HAL pulls its MJCF lazily from the
``robot_descriptions`` package and ships its canonical
:class:`openral_core.RobotDescription` (``UR5e_DESCRIPTION``) — so this
smoke exercises the same RobotDescription wiring used at runtime, not a
stub.

Skips cleanly when ``rclpy`` / ``openral_hal`` / ``mujoco`` /
``robot_descriptions`` are unavailable (lint-only environments).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")


from openral_hal import UR5eHAL
from openral_hal.lifecycle import _HALLifecycleNode  # type: ignore[attr-defined]
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState as RosJointState


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


NODE_NAME = "openral_hal_ur5e"
N_JOINTS = 6


def _hal_factory() -> UR5eHAL:
    return UR5eHAL(gravity_enabled=False)


def test_lifecycle_smoke() -> None:
    """Drive the full configure→activate→deactivate→cleanup transition cycle."""
    rclpy.init()
    try:
        node = _HALLifecycleNode(NODE_NAME, _hal_factory)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        def spin_for(seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)

        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

            assert node._hal is not None  # type: ignore[attr-defined]
            desc = node._hal.description  # type: ignore[attr-defined]
            assert len(desc.joints) == N_JOINTS, (
                f"expected {N_JOINTS}-DoF RobotDescription on the live HAL, got {len(desc.joints)}"
            )

            helper = rclpy.create_node("test_lifecycle_subscriber")
            executor.add_node(helper)
            received: list[RosJointState] = []
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            helper.create_subscription(
                RosJointState,
                f"/{NODE_NAME}/joint_states",
                received.append,
                qos,
            )

            spin_for(1.0)

            assert len(received) >= 1, (
                f"expected ≥1 joint_states message during active phase, got {len(received)}"
            )
            assert len(received[-1].position) == N_JOINTS

            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS

            executor.remove_node(helper)
            helper.destroy_node()
        finally:
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_generic_hal_lifecycle_emits_hal_read_state_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """``_HALLifecycleNode`` emits ``hal.read_state`` spans for any HAL adapter.

    Dashboard contract: the per-tick span carries
    ``openral.hal.adapter``, ``openral.hal.robot.model``,
    ``openral.tick.idx``, plus the per-joint reality arrays
    (``names`` / ``positions`` / ``position_limits_*``). The Identity
    row latches ``hal.adapter`` and ``hal.robot.model`` from the span.
    """
    rclpy.init()
    try:
        node = _HALLifecycleNode(NODE_NAME, _hal_factory)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
        finally:
            for transition in ("trigger_deactivate", "trigger_cleanup"):
                try:
                    getattr(node, transition)()
                except Exception:
                    pass
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.read_state"]
    assert spans, "_HALLifecycleNode did not emit any hal.read_state spans"
    attrs = dict(spans[0].attributes or {})
    # The adapter label is class-name lowercased.
    assert attrs.get("openral.hal.adapter") == "ur5ehal"
    assert str(attrs.get("openral.hal.robot.model"))  # any non-empty
    assert attrs.get("openral.tick.idx") == 0
    names = list(attrs.get("openral.hal.joint.names") or [])
    positions = list(attrs.get("openral.hal.joint.positions") or [])
    assert len(names) == N_JOINTS, f"expected {N_JOINTS} joint names, got {len(names)}"
    assert len(positions) == N_JOINTS


def test_generic_hal_lifecycle_emits_hal_send_action_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """A safe_action publication produces a ``hal.send_action`` span."""
    from openral_msgs.msg import ActionChunk
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    try:
        node = _HALLifecycleNode(NODE_NAME, _hal_factory)
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        helper = rclpy.create_node("test_send_action_helper")
        executor.add_node(helper)
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)

            chunk_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            pub = helper.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)
            chunk = ActionChunk()
            chunk.n_dof = N_JOINTS
            chunk.horizon = 1
            chunk.flat = [0.1] * N_JOINTS
            chunk.rskill_id = "openral/test-generic-hal-span"
            pub.publish(chunk)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
        finally:
            for transition in ("trigger_deactivate", "trigger_cleanup"):
                try:
                    getattr(node, transition)()
                except Exception:
                    pass
            executor.remove_node(helper)
            executor.remove_node(node)
            helper.destroy_node()
            node.destroy_node()
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.send_action"]
    assert spans, "_HALLifecycleNode did not emit a hal.send_action span on safe_action"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == "ur5ehal"
    assert attrs.get("openral.hal.control_mode") == "joint_position"
    assert attrs.get("openral.hal.action.source") == "safe_action"
    next_row = list(attrs.get("openral.hal.action.next") or [])
    assert len(next_row) == N_JOINTS
    assert attrs.get("openral.hal.action.dim") == N_JOINTS
    assert attrs.get("openral.hal.action.applied") is True
