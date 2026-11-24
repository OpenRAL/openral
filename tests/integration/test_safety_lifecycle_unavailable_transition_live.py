"""Live: a safety lifecycle node refuses an unavailable transition and keeps braking.

On ROS 2 Jazzy an id-only ``change_state`` naming a transition the node's current state
lacks (a second ``activate``) raises ``RCLError`` out of the executor and kills the node.
For the E-stop sources that meant a launch_ros ``ChangeState`` event or a hand-typed
``ros2 service call`` could take a brake source off the graph. Each test drives a real
safety node on its own executor with id-only requests, checks the repeat is refused and
the executor is alive, then proves the node still does its safety job:

* ``HardwareEstopNode`` — a pendant press still reaches ``/openral/estop``;
* ``HumanEstopForwarderNode`` — ``/openral/human_estop`` still reaches ``/openral/estop``;
* ``DeadmanWatchdogNode`` — a silent ``/openral/safe_action`` still fires ``/openral/estop``;
* ``SafetyPassthroughNode`` — still answers ``get_state`` in the state it was left in.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (``scripts/ros_live_tests.sh``)::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k safety_lifecycle_unavailable
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason="live rclpy — set OPENRAL_TEST_ROS_LIVE=1")


class _Rig:
    """A safety node spun on its own executor thread, plus an id-only lifecycle client."""

    def __init__(self, rclpy: Any, node: Any) -> None:
        from lifecycle_msgs.srv import ChangeState, GetState
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Empty

        self._rclpy = rclpy
        self.node = node
        self.spin_errors: list[BaseException] = []
        self.estop: list[Any] = []
        self.client = rclpy.create_node(f"{node.get_name()}_test_client")
        self.client.create_subscription(Empty, "/openral/estop", self.estop.append, 10)
        name = node.get_name()
        self._change = self.client.create_client(ChangeState, f"/{name}/change_state")
        self._get = self.client.create_client(GetState, f"/{name}/get_state")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except BaseException as exc:  # reason: record what killed the node, asserted below
            self.spin_errors.append(exc)

    def _call(self, client: Any, req: Any) -> Any:
        assert client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(req)
        self._rclpy.spin_until_future_complete(self.client, future, timeout_sec=5.0)
        assert future.done(), f"no reply; executor errors: {self.spin_errors!r}"
        return future.result()

    def transition(self, transition_id: int) -> bool:
        from lifecycle_msgs.srv import ChangeState

        req = ChangeState.Request()
        req.transition.id = transition_id  # id only, as a launch_ros ChangeState event
        return bool(self._call(self._change, req).success)

    def state(self) -> str:
        from lifecycle_msgs.srv import GetState

        return str(self._call(self._get, GetState.Request()).current_state.label)

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not predicate():
            self._rclpy.spin_once(self.client, timeout_sec=0.05)
        return predicate()

    def assert_alive(self) -> None:
        assert self.spin_errors == []
        assert self._thread.is_alive()

    def close(self) -> None:
        self.executor.shutdown()
        self.client.destroy_node()
        self.node.destroy_node()


@pytest.fixture
def rclpy_ctx() -> Iterator[Any]:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    rclpy.init()
    try:
        yield rclpy
    finally:
        rclpy.shutdown()


@contextlib.contextmanager
def _rig(rclpy: Any, make: Callable[[], Any]) -> Iterator[_Rig]:
    rig: _Rig | None = None
    try:
        rig = _Rig(rclpy, make())
        yield rig
    finally:
        if rig is not None:
            rig.close()


def _refuse_repeats(rig: _Rig) -> None:
    """configure, activate, then a repeated activate and an early configure are refused."""
    from lifecycle_msgs.msg import Transition

    assert rig.transition(Transition.TRANSITION_CONFIGURE)
    assert rig.transition(Transition.TRANSITION_ACTIVATE)
    assert not rig.transition(Transition.TRANSITION_ACTIVATE)  # repeated: refused
    assert not rig.transition(Transition.TRANSITION_CONFIGURE)  # not from active: refused
    rig.assert_alive()
    assert rig.state() == "active"


def test_hardware_estop_refuses_a_repeat_and_still_forwards_a_press(rclpy_ctx: Any) -> None:
    from openral_safety_watchdog.hardware_estop_node import HardwareEstopNode
    from rclpy.parameter import Parameter

    pressed = [False]

    def make() -> Any:
        node = HardwareEstopNode(node_name="openral_test_safety_transition_hw_estop")
        # /dev/null is on every Linux host; the hook is the node's documented driver seam.
        node.set_parameters([Parameter("device", Parameter.Type.STRING, "/dev/null")])
        node.read_pressed_hook = lambda: pressed[0]
        return node

    with _rig(rclpy_ctx, make) as rig:
        _refuse_repeats(rig)
        pressed[0] = True
        assert rig.wait_for(lambda: len(rig.estop) >= 1), "press no longer reaches /openral/estop"
        rig.assert_alive()


def test_human_estop_forwarder_refuses_a_repeat_and_still_forwards(rclpy_ctx: Any) -> None:
    from openral_human_estop.forwarder_node import HumanEstopForwarderNode
    from std_msgs.msg import Empty

    def make() -> Any:
        return HumanEstopForwarderNode(node_name="openral_test_safety_transition_forwarder")

    with _rig(rclpy_ctx, make) as rig:
        _refuse_repeats(rig)
        human = rig.client.create_publisher(Empty, "/openral/human_estop", 10)
        assert rig.wait_for(lambda: human.get_subscription_count() >= 1)
        human.publish(Empty())
        assert rig.wait_for(lambda: len(rig.estop) >= 1), "human estop no longer forwarded"
        rig.assert_alive()


def test_deadman_refuses_a_repeat_and_still_fires_on_silence(rclpy_ctx: Any) -> None:
    from openral_safety_watchdog.deadman_watchdog_node import DeadmanWatchdogNode
    from rclpy.parameter import Parameter

    def make() -> Any:
        node = DeadmanWatchdogNode(node_name="openral_test_safety_transition_deadman")
        # Long enough that the refusals land before it fires; nobody publishes safe_action.
        node.set_parameters(
            [
                Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 1.0),
                Parameter("check_period_s", Parameter.Type.DOUBLE, 0.02),
            ]
        )
        return node

    with _rig(rclpy_ctx, make) as rig:
        _refuse_repeats(rig)
        assert rig.wait_for(lambda: len(rig.estop) >= 1), "silent safe_action no longer brakes"
        rig.assert_alive()


def test_supervisor_refuses_a_repeat_and_keeps_serving(rclpy_ctx: Any) -> None:
    from lifecycle_msgs.msg import Transition
    from openral_safety.supervisor_node import SafetyPassthroughNode

    def make() -> Any:
        return SafetyPassthroughNode(node_name="openral_test_safety_transition_supervisor")

    with _rig(rclpy_ctx, make) as rig:
        _refuse_repeats(rig)
        assert rig.transition(Transition.TRANSITION_DEACTIVATE)
        assert not rig.transition(Transition.TRANSITION_DEACTIVATE)  # repeated: refused
        rig.assert_alive()
        assert rig.state() == "inactive"
