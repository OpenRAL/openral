"""Live ROS coverage for ``openral estop reset`` and the runner's latched-goal warning.

The 2026-10-04 twin pass reset an e-stop with ``ros2 service call /openral/estop_reset``:
the kernel cleared, the rSkill runner stayed latched (only ``/openral/estop_cleared``
clears it), and every later goal was rejected with nothing in the log. This proves, on a
real DDS graph with the real ``SafetyPassthroughNode`` (which serves the same
``/openral/estop_reset`` contract as the C++ kernel) and the real ``RskillRunnerNode``:

1. a latched runner rejects a goal sent through a real action client AND says why at
   WARNING, naming the recovery command;
2. ``openral estop reset`` run as a real subprocess, inside the kernel's cooldown, is
   refused (exit 1) and broadcasts nothing — the runner stays latched;
3. after the cooldown the same command clears the kernel AND the runner.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

# Long enough that the first CLI call lands inside it, short enough to wait out.
_COOLDOWN_S = 3.0


def _wait_until(predicate: Any, *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _openral_estop_reset() -> subprocess.CompletedProcess[str]:
    """Run the real CLI in its own process (its own rclpy context and DDS participant)."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from openral_cli.main import app; app()",
            "estop",
            "reset",
            "--service-timeout-s",
            "15",
            "--discovery-wait-s",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ.copy(),
        check=False,
    )


def test_estop_reset_cli_clears_kernel_then_runner_and_rejection_is_loud(
    capfd: pytest.CaptureFixture[str],
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("std_srvs.srv")
    pytest.importorskip("openral_msgs.action")
    from openral_msgs.action import ExecuteRskill
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.action import ActionClient
    from rclpy.action.server import GoalResponse
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from std_msgs.msg import Empty

    rclpy.init()
    runtime = compose_so100_runtime()
    runner = runtime.skill_runner_node
    runner.set_parameters([rclpy.parameter.Parameter("joint_state_staleness_limit_s", value=0.5)])
    safety = SafetyPassthroughNode(node_name="openral_estop_reset_cli_kernel")
    safety.set_parameters(
        [
            rclpy.parameter.Parameter("n_dof", value=6),
            rclpy.parameter.Parameter("estop_reset_cooldown_s", value=_COOLDOWN_S),
        ]
    )
    helper = rclpy.create_node("openral_estop_reset_cli_helper")
    estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)
    action_client = ActionClient(helper, ExecuteRskill, "/openral/execute_rskill")
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (safety, runner, helper):
        executor.add_node(node)
    stop = threading.Event()

    def _spin() -> None:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    try:
        for node in (safety, runner):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        assert _wait_until(
            lambda: (
                (estop_pub.publish(Empty()) or True)
                and safety._estopped is True
                and runner._estop_latched is True
            )
        ), "the e-stop never latched both the kernel and the runner"
        latched_at = time.monotonic()

        # 1. A real goal is rejected, and the reason reaches the log at WARNING.
        assert action_client.wait_for_server(timeout_sec=10.0)
        future = action_client.send_goal_async(ExecuteRskill.Goal())
        assert _wait_until(future.done), "the runner never answered the goal"
        assert future.result().accepted is False
        logged: list[str] = []
        assert _wait_until(
            lambda: (
                (logged.append(capfd.readouterr().err) or True)
                and "rskill_runner.goal_rejected: e-stop latched" in "".join(logged)
            ),
            timeout_s=5.0,
        ), "a latched rejection must log a WARNING naming the latch"
        line = next(ln for ln in "".join(logged).splitlines() if "e-stop latched" in ln)
        assert "WARN" in line and "openral estop reset" in line, line

        # 2. Inside the cooldown the kernel refuses; nothing is broadcast.
        assert time.monotonic() - latched_at < _COOLDOWN_S, "test too slow for its cooldown"
        refused = _openral_estop_reset()
        assert refused.returncode == 1, refused.stdout + refused.stderr
        assert "REFUSED" in refused.stderr
        assert safety._estopped is True
        time.sleep(0.5)  # anything the CLI might have broadcast would have landed by now
        assert runner._estop_latched is True, "a refused kernel reset must not clear the runner"

        # 3. After the cooldown: kernel first, then the runner via /openral/estop_cleared.
        time.sleep(max(0.0, _COOLDOWN_S - (time.monotonic() - latched_at)) + 0.2)
        cleared = _openral_estop_reset()
        assert cleared.returncode == 0, cleared.stdout + cleared.stderr
        assert safety._estopped is False
        assert _wait_until(lambda: runner._estop_latched is False), (
            "openral estop reset must un-latch the runner, not only the kernel"
        )
        assert runner._goal_cb(None) == GoalResponse.ACCEPT
    finally:
        stop.set()
        spinner.join(timeout=5.0)
        with __import__("contextlib").suppress(Exception):
            for node in (runner, safety):
                node.trigger_deactivate()
                node.trigger_cleanup()
        executor.shutdown()
        action_client.destroy()
        helper.destroy_node()
        runner.destroy_node()
        runtime.world_state_node.destroy_node()
        safety.destroy_node()
        rclpy.try_shutdown()
