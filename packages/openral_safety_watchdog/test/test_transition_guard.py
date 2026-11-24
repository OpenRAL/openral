"""Real-rclpy tests for ``openral_safety_watchdog.transition_guard``.

The E-stop sources' dependency-free copy of
``openral_observability.refuse_unavailable_transitions``: an id-only ``change_state``
naming a transition the current state lacks is refused instead of raising ``RCLError``
out of the executor (which killed the node). Calls the real guarded service callback of
real lifecycle nodes; the live executor path, per node, is
``tests/integration/test_safety_lifecycle_unavailable_transition_live.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")

from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from openral_safety_watchdog.deadman_watchdog_node import DeadmanWatchdogNode
from openral_safety_watchdog.transition_guard import refuse_unavailable_transitions
from rclpy.lifecycle import LifecycleNode


@pytest.fixture
def ros_context() -> Iterator[None]:
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _change_state(node: Any, transition_id: int, label: str = "") -> bool:
    req = ChangeState.Request()
    req.transition.id = transition_id
    req.transition.label = label
    return bool(node._service_change_state.callback(req, ChangeState.Response()).success)


def _state(node: Any) -> str:
    return str(node._state_machine.current_state[1])


def test_id_only_unavailable_transition_is_refused(ros_context: None) -> None:
    node = LifecycleNode("transition_guard_test_refused")
    try:
        refuse_unavailable_transitions(node)
        assert _change_state(node, Transition.TRANSITION_ACTIVATE) is False  # unconfigured
        assert _state(node) == "unconfigured"
        assert _change_state(node, Transition.TRANSITION_CONFIGURE) is True
        assert _change_state(node, Transition.TRANSITION_CONFIGURE) is False  # repeated
        assert _state(node) == "inactive"
    finally:
        node.destroy_node()


def test_labelled_request_is_left_to_rclpy(ros_context: None) -> None:
    node = LifecycleNode("transition_guard_test_labelled")
    try:
        refuse_unavailable_transitions(node)
        assert _change_state(node, Transition.TRANSITION_ACTIVATE, "activate") is False
        assert _change_state(node, Transition.TRANSITION_CONFIGURE, "configure") is True
        assert _state(node) == "inactive"
    finally:
        node.destroy_node()


def test_the_deadman_is_guarded(ros_context: None) -> None:
    """The watchdog installs the guard itself: a repeated activate leaves it active."""
    node = DeadmanWatchdogNode(node_name="transition_guard_test_deadman")
    try:
        assert _change_state(node, Transition.TRANSITION_CONFIGURE) is True
        assert _change_state(node, Transition.TRANSITION_ACTIVATE) is True
        assert _change_state(node, Transition.TRANSITION_ACTIVATE) is False
        assert _state(node) == "active"
    finally:
        node.destroy_node()
