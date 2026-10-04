"""The one-shot ``startup_prompt`` waits for the ExecuteRskill server.

The reasoner dispatches on its first tick after ``/openral/prompt`` lands and
probes ``/openral/execute_rskill`` for only 100 ms; ``runtime_node`` creates that
server in the runner's ``on_configure``, which can finish after the prompt
router activates. Publishing first fails the dispatch with KIND_CONTROLLER.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like ``test_reasoner_node_end_to_end.py``;
run with ``just test-ros-live -k startup_gate`` after ``just ros2-build``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))


@pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)
def test_startup_prompt_waits_for_execute_rskill_server() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import PromptStamped
    from openral_prompt_router import PromptRouterNode
    from rclpy.action import ActionServer
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    received: list[float] = []
    try:
        router = PromptRouterNode()
        router.set_parameters(
            [Parameter("startup_prompt", Parameter.Type.STRING, "pick the red cube")]
        )
        router.trigger_configure()

        reasoner_side = rclpy.create_node("startup_gate_reasoner_side")
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        reasoner_side.create_subscription(
            PromptStamped, "/openral/prompt", lambda _m: received.append(time.monotonic()), qos
        )
        executor = SingleThreadedExecutor()
        executor.add_node(reasoner_side)

        # on_activate blocks in the gate, so run it beside the spinning executor.
        activator = threading.Thread(target=router.trigger_activate, daemon=True)
        activator.start()

        hold_s = 2.0
        end = time.monotonic() + hold_s
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.05)
        assert not received, "startup_prompt published before execute_rskill was advertised"

        def _accept(goal_handle: Any) -> Any:
            goal_handle.succeed()
            return ExecuteRskill.Result()

        server = ActionServer(reasoner_side, ExecuteRskill, "/openral/execute_rskill", _accept)
        advertised_at = time.monotonic()
        end = advertised_at + 10.0
        while time.monotonic() < end and not received:
            executor.spin_once(timeout_sec=0.05)
        activator.join(timeout=5.0)

        assert received, "startup_prompt never published once execute_rskill was advertised"
        assert received[0] >= advertised_at
        server.destroy()
        reasoner_side.destroy_node()
        router.destroy_node()
    finally:
        rclpy.shutdown()
