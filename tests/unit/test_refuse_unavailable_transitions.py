"""Unit tests for ``openral_observability.refuse_unavailable_transitions``.

rclpy (Jazzy) raises ``RCLError`` out of the ``change_state`` service callback, and so
out of the executor, when an id-only request names a transition the current state lacks
(a second ``deactivate``). The guard answers ``success: false`` instead. These tests call
the real guarded service callback of real lifecycle nodes (a bare ``LifecycleNode`` and a
``HALLifecycleNodeBase`` over a real ``SimAttachedHAL``); the live executor path is in
``tests/integration/test_lifecycle_unavailable_transition_live.py``. No mocks (§1.11);
``FakeSimEnv`` is the sim-rollout recorder ``test_hal_lifecycle_shutdown_verdict`` uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("rclpy.lifecycle")

_ROBOT_YAML = Path(__file__).resolve().parents[2] / "robots" / "so101_follower" / "robot.yaml"


@pytest.fixture
def node() -> Iterator[Any]:
    """A real, guarded LifecycleNode driven to ACTIVE."""
    from openral_observability import refuse_unavailable_transitions
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    n: Any = None
    try:
        n = LifecycleNode("openral_refuse_unavailable_transitions_test")
        refuse_unavailable_transitions(n)
        n.trigger_configure()
        n.trigger_activate()
        yield n
    finally:
        if n is not None:
            n.destroy_node()
        rclpy.shutdown()


def _change_state(node: Any, transition_id: int, label: str = "") -> bool:
    from lifecycle_msgs.srv import ChangeState

    req = ChangeState.Request()
    req.transition.id = transition_id
    req.transition.label = label
    resp = node._service_change_state.callback(req, ChangeState.Response())
    return bool(resp.success)


def _state(node: Any) -> str:
    return str(node._state_machine.current_state[1])


def test_valid_transition_runs_unchanged(node: Any) -> None:
    """A transition the current state has still runs and succeeds."""
    from lifecycle_msgs.msg import Transition

    assert _change_state(node, Transition.TRANSITION_DEACTIVATE) is True
    assert _state(node) == "inactive"


def test_repeated_deactivate_is_refused_not_raised(node: Any) -> None:
    """A second id-only deactivate answers success: false and leaves the node inactive."""
    from lifecycle_msgs.msg import Transition

    assert _change_state(node, Transition.TRANSITION_DEACTIVATE) is True
    assert _change_state(node, Transition.TRANSITION_DEACTIVATE) is False
    assert _state(node) == "inactive"
    # Still usable: the next valid transition goes through.
    assert _change_state(node, Transition.TRANSITION_ACTIVATE) is True
    assert _state(node) == "active"


def test_a_labelled_request_is_left_to_rclpy(node: Any) -> None:
    """rclpy resolves a label itself: an unavailable one is refused, a valid one runs."""
    from lifecycle_msgs.msg import Transition

    assert _change_state(node, Transition.TRANSITION_ACTIVATE, "activate") is False
    assert _state(node) == "active"
    assert _change_state(node, Transition.TRANSITION_DEACTIVATE, "deactivate") is True
    assert _state(node) == "inactive"


def test_unguarded_node_raises() -> None:
    """Pins the upstream behaviour the guard exists for; if rclpy fixes it, drop the guard."""
    from lifecycle_msgs.msg import Transition
    from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    n: Any = None
    try:
        n = LifecycleNode("openral_unguarded_transition_test")
        n.trigger_configure()
        with pytest.raises(_rclpy.RCLError):
            _change_state(n, Transition.TRANSITION_DEACTIVATE)
    finally:
        if n is not None:
            n.destroy_node()
        rclpy.shutdown()


def test_a_hal_node_refuses_a_repeated_transition() -> None:
    """``HALLifecycleNodeBase`` (actuation path) is guarded: a repeat leaves the HAL as is.

    Driven through configure / cleanup: ``FakeSimEnv`` emits no joint positions, so the
    node (rightly) refuses to activate on it.
    """
    from lifecycle_msgs.msg import Transition
    from openral_core import RobotDescription, SceneSpec, TaskSpec
    from openral_hal.lifecycle import HALLifecycleNodeBase
    from openral_hal.protocol import HAL
    from openral_hal.sim_attached import SimAttachedHAL

    from tests.unit.fakes.fake_sim_env import FakeSimEnv

    env = FakeSimEnv(
        action_dim=6,
        has_sim_clock=True,
        sim_dt_ns=20_000_000,
        scene=SceneSpec(id="tabletop_push", backend="mujoco"),
        task=TaskSpec(
            id="tabletop_push/push_to_goal",
            scene_id="tabletop_push",
            instruction="push the red cube onto the green goal marker",
        ),
    )
    description = RobotDescription.from_yaml(str(_ROBOT_YAML))

    class _SimHALNode(HALLifecycleNodeBase):
        def _create_hal(self) -> HAL:
            return SimAttachedHAL(env, description)

    rclpy.init()
    n: Any = None
    try:
        n = _SimHALNode("openral_hal_refuse_unavailable_transitions_test")
        assert _change_state(n, Transition.TRANSITION_ACTIVATE) is False  # before configure
        assert _state(n) == "unconfigured"
        assert _change_state(n, Transition.TRANSITION_CONFIGURE) is True
        hal = n._hal
        assert hal is not None
        assert _change_state(n, Transition.TRANSITION_CONFIGURE) is False  # repeated
        assert _state(n) == "inactive"
        assert n._hal is hal  # not rebuilt, not torn down
        assert _change_state(n, Transition.TRANSITION_CLEANUP) is True
        assert _state(n) == "unconfigured"
    finally:
        if n is not None:
            n.shutdown_hal()
            n.destroy_node()
        rclpy.shutdown()
