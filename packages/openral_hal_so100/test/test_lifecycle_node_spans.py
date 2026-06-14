"""HAL OTel-span coverage for the SO-100 lifecycle node (ADR-0018 + dashboard).

Drives the real ``ManifestHALLifecycleNode`` (the generic node the SO-100
package now uses after the issue #191 Phase 2 migration) against a real
``SO100FollowerHAL`` backed by ``SO100DigitalTwin`` — no mocks
(CLAUDE.md §1.11). The spans are emitted by the shared ``HALLifecycleNodeBase``,
so this exercises the exact path every robot's node takes. Sidesteps
`on_configure` (which would open a serial port) by constructing the node,
injecting the digital-twin-backed HAL, then invoking ``_publish_joint_state`` /
``_send_action_traced`` directly to assert the spans the dashboard expects.

Skips cleanly when ``rclpy`` or the SO-100 simulator dependencies are
not on the path.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")

from openral_core.schemas import Action, ControlMode
from openral_hal.so100_follower import SO100FollowerHAL

try:
    from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
except ImportError:  # reason: lerobot extras may not be installed
    pytest.skip(
        "openral_hal.so100_sim unavailable (install lerobot extras)", allow_module_level=True
    )


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


def _make_node_with_twin_hal() -> object:
    """Construct the manifest node and inject a digital-twin-backed HAL."""
    from openral_hal.lifecycle import ManifestHALLifecycleNode

    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    hal.connect()
    node = ManifestHALLifecycleNode("openral_hal_so100")
    node._hal = hal  # type: ignore[attr-defined]
    # Minimal publisher state so `_publish_joint_state` does not bail out.
    node._publisher = object()  # type: ignore[attr-defined]
    # Re-publishers are touched in the body — patch them out cheaply.
    node._joint_state_pub = None  # type: ignore[attr-defined]
    return node


def _publish_one_state(node: object) -> None:
    """Call the timer callback once, suppressing the ROS publisher path."""
    publisher = node._publisher  # type: ignore[attr-defined]

    class _Capture:
        """Minimal stand-in for the rclpy publisher; just discards."""

        def publish(self, _msg: object) -> None:
            pass

    node._publisher = _Capture()  # type: ignore[attr-defined]
    try:
        node._publish_joint_state()  # type: ignore[attr-defined]
    finally:
        node._publisher = publisher  # type: ignore[attr-defined]


def test_publish_joint_state_emits_hal_read_state_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """The SO-100 lifecycle node emits hal.read_state with the right attrs."""
    rclpy.init()
    try:
        node = _make_node_with_twin_hal()
        try:
            _publish_one_state(node)
        finally:
            try:
                node._hal.disconnect()  # type: ignore[attr-defined]
            except Exception:
                pass
            node.destroy_node()  # type: ignore[attr-defined]
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.read_state"]
    assert spans, "ManifestHALLifecycleNode did not emit a hal.read_state span"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == "so100followerhal"
    assert str(attrs.get("openral.hal.robot.model"))  # non-empty
    assert attrs.get("openral.tick.idx") == 0
    names = list(attrs.get("openral.hal.joint.names") or [])
    positions = list(attrs.get("openral.hal.joint.positions") or [])
    assert len(names) >= 1
    assert len(positions) >= 1


def test_send_action_traced_emits_hal_send_action_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """The SO-100 lifecycle node emits hal.send_action with the right attrs."""
    rclpy.init()
    try:
        node = _make_node_with_twin_hal()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        try:
            node._send_action_traced(action, source="safe_action")  # type: ignore[attr-defined]
        finally:
            try:
                node._hal.disconnect()  # type: ignore[attr-defined]
            except Exception:
                pass
            node.destroy_node()  # type: ignore[attr-defined]
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.send_action"]
    assert spans, "ManifestHALLifecycleNode did not emit a hal.send_action span"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == "so100followerhal"
    assert attrs.get("openral.hal.control_mode") == "joint_position"
    assert attrs.get("openral.hal.action.source") == "safe_action"
    next_row = list(attrs.get("openral.hal.action.next") or [])
    assert len(next_row) == 6
    assert attrs.get("openral.hal.action.applied") is True
