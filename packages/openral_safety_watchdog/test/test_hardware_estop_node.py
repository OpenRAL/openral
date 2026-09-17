"""Real-rclpy tests for the hardware_estop_node's honesty contract.

Two properties:

1. **Not-ready is reported, never faked.** This repo ships no pendant driver,
   and the base ``_read_pressed`` would answer "not pressed" forever — which
   reads exactly like a healthy E-stop. ``on_configure`` must therefore refuse
   to leave UNCONFIGURED when no device is declared, when the declared device
   is absent, or when a present device has no driver behind it.
2. **When a real read source exists, the rising edge brakes.** Estop plus a
   ``FailureTrigger(KIND_HUMAN, SEVERITY_ABORT)`` carrying ``HumanEvidence``.
3. **A read that raises fails closed.** An unplugged pendant or a driver fault
   means the state is unknown, which for a brake source is unsafe: brake once
   and latch, rather than letting the exception kill the timer and take the
   E-stop source off the graph silently.

Real lifecycle node, real ``openral_msgs`` IDL, real DDS (CLAUDE.md §1.11).
``read_pressed_hook`` is the node's own documented per-vendor injection seam,
not a stand-in for another OpenRAL component. Gated on ``rclpy`` +
``openral_msgs`` being importable; without them the test skips with a typed
reason.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import HumanEvidence
from openral_msgs.msg import FailureTrigger
from openral_safety_watchdog.hardware_estop_node import HardwareEstopNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Empty

# A device path that really is on every Linux host, so the "declared device is
# present" leg of the check is satisfied by a fact rather than by a patch.
_PRESENT_DEVICE = "/dev/null"


@pytest.fixture
def ros_context() -> Any:
    """Spin up rclpy for the test; tear down after."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return predicate()


def test_no_device_declared_refuses_to_configure(ros_context: None) -> None:
    """The default (no pendant on this host) must report not-ready."""
    node = HardwareEstopNode(node_name="hardware_estop_test_no_device")
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
        assert node.get_parameter("device").get_parameter_value().string_value == ""
    finally:
        node.destroy_node()


def test_absent_device_refuses_to_configure(ros_context: None) -> None:
    """A declared device that is not on the host is not a hardware E-stop."""
    node = HardwareEstopNode(node_name="hardware_estop_test_absent_device")
    try:
        node.set_parameters(
            [Parameter("device", Parameter.Type.STRING, "/dev/openral-pendant-not-here")]
        )
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
    finally:
        node.destroy_node()


def test_present_device_without_a_driver_refuses_to_configure(ros_context: None) -> None:
    """A real path is not enough: the base class cannot read any pendant.

    The regression this pins is the silent one — a node sitting ACTIVE on a
    ``_read_pressed`` that returns ``False`` forever looks, on the graph and in
    every dashboard, exactly like an armed hardware E-stop.
    """
    node = HardwareEstopNode(node_name="hardware_estop_test_no_driver")
    try:
        node.set_parameters([Parameter("device", Parameter.Type.STRING, _PRESENT_DEVICE)])
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
    finally:
        node.destroy_node()


def test_rising_edge_fires_estop_and_human_evidence(ros_context: None) -> None:
    """With a real read source, a press publishes estop + KIND_HUMAN once."""
    node = HardwareEstopNode(node_name="hardware_estop_test_press")
    helper = rclpy.create_node("hardware_estop_test_press_helper")
    estop_received: list[Empty] = []
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    helper.create_subscription(
        FailureTrigger, "/openral/failure/safety", failures_received.append, 50
    )

    pressed = False

    def _read() -> bool:
        return pressed

    node.read_pressed_hook = _read
    node.set_parameters(
        [
            Parameter("device", Parameter.Type.STRING, _PRESENT_DEVICE),
            Parameter("channel_label", Parameter.Type.STRING, "bench_pendant"),
        ]
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        # Not pressed: nothing on the wire.
        for _ in range(20):
            executor.spin_once(timeout_sec=0.01)
        own = [ft for ft in failures_received if ft.kind == FailureTrigger.KIND_HUMAN]
        assert own == [], "hardware estop fired with the pendant released"

        pressed = True

        def _have_human_failure() -> bool:
            return any(ft.kind == FailureTrigger.KIND_HUMAN for ft in failures_received)

        assert _spin_until(executor, lambda: len(estop_received) >= 1)
        assert _spin_until(executor, _have_human_failure)
        ft = next(ft for ft in failures_received if ft.kind == FailureTrigger.KIND_HUMAN)
        assert ft.severity == FailureTrigger.SEVERITY_ABORT
        evidence = json.loads(ft.evidence_json)
        assert evidence == {"kind": "human", "actor": "bench_pendant", "reason": "pressed"}
        # The whole point of evidence_json is that the reasoner can parse it.
        assert HumanEvidence.model_validate(evidence).actor == "bench_pendant"

        # Held, not re-pressed: rising edge only, no storm on the brake topic.
        before = len([f for f in failures_received if f.kind == FailureTrigger.KIND_HUMAN])
        for _ in range(30):
            executor.spin_once(timeout_sec=0.01)
        after = len([f for f in failures_received if f.kind == FailureTrigger.KIND_HUMAN])
        assert after == before, "a held pendant re-published the estop"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()


def test_a_raising_read_brakes_and_does_not_kill_the_node(ros_context: None) -> None:
    """A driver fault is "state unknown", which for a brake source is unsafe.

    The regression: an exception out of ``_read_pressed`` propagates through the
    timer and out of ``rclpy.spin``, the process exits, and because the launch
    has no ``on_exit`` handler for it the hardware E-stop source just disappears
    — no estop, no further diagnostic.
    """
    node = HardwareEstopNode(node_name="hardware_estop_test_read_raises")
    helper = rclpy.create_node("hardware_estop_test_read_raises_helper")
    estop_received: list[Empty] = []
    failures_received: list[FailureTrigger] = []
    helper.create_subscription(Empty, "/openral/estop", estop_received.append, 10)
    helper.create_subscription(
        FailureTrigger, "/openral/failure/safety", failures_received.append, 50
    )

    def _read() -> bool:
        raise OSError("pendant unplugged")

    node.read_pressed_hook = _read
    node.set_parameters(
        [
            Parameter("device", Parameter.Type.STRING, _PRESENT_DEVICE),
            Parameter("channel_label", Parameter.Type.STRING, "bench_pendant"),
        ]
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        def _have_human_failure() -> bool:
            return any(ft.kind == FailureTrigger.KIND_HUMAN for ft in failures_received)

        assert _spin_until(executor, lambda: len(estop_received) >= 1), (
            "a pendant whose read raised did not brake"
        )
        assert _spin_until(executor, _have_human_failure)
        ft = next(ft for ft in failures_received if ft.kind == FailureTrigger.KIND_HUMAN)
        evidence = json.loads(ft.evidence_json)
        assert evidence["reason"] == "read_failed"
        assert HumanEvidence.model_validate(evidence).actor == "bench_pendant"

        # Latched: a persistently broken driver must not storm the brake topic.
        before = len([f for f in failures_received if f.kind == FailureTrigger.KIND_HUMAN])
        for _ in range(40):
            executor.spin_once(timeout_sec=0.01)
        after = len([f for f in failures_received if f.kind == FailureTrigger.KIND_HUMAN])
        assert after == before, "a persistently failing read re-published the estop"
    finally:
        executor.remove_node(node)
        executor.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
