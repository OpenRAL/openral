"""Live: a repeated id-only ``deactivate`` is refused and the lifecycle node keeps serving.

On ROS 2 Jazzy a ``change_state`` request carrying only a transition *id* the node's
current state lacks raises ``RCLError`` out of the executor and kills the node (rclpy
refuses an unknown *label* cleanly). The reasoner and ``tools/lifecycle_autostart.py``
send the label too, but launch_ros ``ChangeState`` events and a hand-typed
``ros2 service call`` send an id alone. ``refuse_unavailable_transitions`` (called by every
OpenRAL Python lifecycle node the reasoner, runner or launch transitions) answers
``success: false`` instead.

Exercises a real ``PromptRouterNode`` (one of the guarded repo nodes) spun on a real
executor, driven by a real id-only ``ChangeState`` client.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (``scripts/ros_live_tests.sh``)::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k unavailable_transition
"""

from __future__ import annotations

import os
import threading
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))


@pytest.mark.skipif(not _LIVE_ROS, reason="live rclpy — set OPENRAL_TEST_ROS_LIVE=1")
def test_repeated_deactivate_is_refused_and_node_keeps_serving() -> None:
    """deactivate → deactivate (refused) → get_state → activate, executor alive throughout."""
    rclpy = pytest.importorskip("rclpy")
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState, GetState
    from openral_prompt_router.prompt_router_node import PromptRouterNode
    from rclpy.executors import SingleThreadedExecutor

    spin_errors: list[BaseException] = []
    rclpy.init()
    peer: Any = None
    client_node: Any = None
    executor: Any = None
    try:
        peer = PromptRouterNode(node_name="openral_test_unavailable_transition_peer")
        client_node = rclpy.create_node("openral_test_unavailable_transition_client")
        executor = SingleThreadedExecutor()
        executor.add_node(peer)

        def _spin() -> None:
            try:
                executor.spin()
            except BaseException as exc:  # reason: record what killed the node, assert below
                spin_errors.append(exc)

        spinner = threading.Thread(target=_spin, daemon=True)
        spinner.start()
        name = peer.get_name()
        change = client_node.create_client(ChangeState, f"/{name}/change_state")
        get_state = client_node.create_client(GetState, f"/{name}/get_state")

        def _call(client: Any, req: Any) -> Any:
            assert client.wait_for_service(timeout_sec=5.0)
            future = client.call_async(req)
            rclpy.spin_until_future_complete(client_node, future, timeout_sec=5.0)
            assert future.done(), f"no reply; executor errors: {spin_errors!r}"
            return future.result()

        def _transition(transition_id: int) -> bool:
            req = ChangeState.Request()
            req.transition.id = transition_id  # id only, as a launch_ros ChangeState event
            return bool(_call(change, req).success)

        assert _transition(Transition.TRANSITION_CONFIGURE)
        assert _transition(Transition.TRANSITION_ACTIVATE)
        assert _transition(Transition.TRANSITION_DEACTIVATE)
        assert not _transition(Transition.TRANSITION_DEACTIVATE)  # repeated: refused
        assert spin_errors == []
        assert _call(get_state, GetState.Request()).current_state.label == "inactive"
        assert _transition(Transition.TRANSITION_ACTIVATE)  # still serving
        assert _call(get_state, GetState.Request()).current_state.label == "active"
        assert spinner.is_alive()
    finally:
        if executor is not None:
            executor.shutdown()
        if client_node is not None:
            client_node.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()
