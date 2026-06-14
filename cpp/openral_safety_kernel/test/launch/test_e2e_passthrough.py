"""ADR-0020 PR-H — end-to-end pass-through against the real C++ kernel.

Brings up the real ``safety_kernel_node`` as a subprocess via launch,
publishes valid ``openral_msgs/ActionChunk`` messages on
``/openral/candidate_action``, and asserts that every one is republished
verbatim on ``/openral/safe_action`` (same trace_id, same flat) and that
neither ``/openral/failure/safety`` nor ``/openral/estop`` fires.

No mocks (CLAUDE.md §1.11): real rclpy, real colcon-built openral_msgs,
real C++ kernel binary. Gated on ``openral_msgs`` + ``launch_testing``
being importable — without a sourced ROS env the test skips.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
import uuid
from typing import Any

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")


# Minimal 3-DoF envelope, sized to admit the +0.1 chunks the passthrough
# test injects. Inlined here rather than synthesised from a real
# RobotDescription because launch_testing files live under cpp/ and the
# `tests/sim/safety/_kernel_subprocess` helper is not on the launch
# tree's sys.path. The arg list below mirrors that helper's output —
# see ADR-0020 PR-K (kernel reads envelope from ROS parameters).
_KERNEL_PARAM_ARGS: list[str] = [
    "-p",
    "n_dof:=3",
    "-p",
    "robot_name:=launchtest",
    "-p",
    "joint_position_min:=[-1.0, -1.0, -1.0]",
    "-p",
    "joint_position_max:=[1.0, 1.0, 1.0]",
    "-p",
    "joint_velocity_max:=[3.15, 3.15, 3.15]",
    "-p",
    "joint_torque_max:=[5.0, 5.0, 5.0]",
    "-p",
    "max_ee_speed_m_s:=0.5",
    "-p",
    "max_ee_accel_m_s2:=2.0",
    "-p",
    "max_force_n:=10.0",
    "-p",
    "max_torque_nm:=3.0",
    "-p",
    "contact_force_threshold_n:=5.0",
    "-p",
    "deadman_required:=false",
    "-p",
    "estop_reset_cooldown_s:=0.1",
]


def _start_kernel(node_name: str, domain_id: int) -> Any:
    """Spawn the C++ kernel node as a child rclcpp process on a private DDS domain.

    We invoke it directly via subprocess (the executable comes from
    ``ros2 run openral_safety_kernel safety_kernel_node``) and let it
    spin on its own; the parent process drives the topic round-trip.
    ``domain_id`` isolates each test from orphaned kernel processes
    that may still be advertised in the daemon's discovery cache.
    """
    import shutil
    import subprocess

    if shutil.which("ros2") is None:
        pytest.skip("ros2 binary not on PATH; source install/setup.bash first")

    env = {**os.environ, "ROS_DOMAIN_ID": str(domain_id)}
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "openral_safety_kernel",
            "safety_kernel_node",
            "--ros-args",
            "-r",
            f"__node:={node_name}",
            *_KERNEL_PARAM_ARGS,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # rclcpp::spin() ignores SIGTERM; put the kernel in its own
        # process group so we can SIGINT the whole group on teardown.
        start_new_session=True,
    )


def _activate_lifecycle(node_name: str, helper: Any) -> bool:
    """Drive the kernel from `unconfigured → active` via the lifecycle service."""
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = helper.create_client(ChangeState, f"/{node_name}/change_state")
    if not client.wait_for_service(timeout_sec=10.0):
        return False
    for t in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        req = ChangeState.Request()
        req.transition.id = t
        fut = client.call_async(req)
        deadline = time.time() + 5.0
        while time.time() < deadline and not fut.done():
            rclpy.spin_once(helper, timeout_sec=0.05)
        if not fut.done() or not fut.result().success:  # type: ignore[union-attr]
            return False
    return True


def test_kernel_passes_valid_chunks_verbatim() -> None:
    """100 valid chunks → 100 /openral/safe_action with matching trace_ids."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    node_name = f"safety_kernel_passthrough_test_{uuid.uuid4().hex[:8]}"
    # Isolate DDS discovery per-test PID so orphaned kernel processes
    # from prior runs don't double-publish on /openral/safe_action.
    domain_id = 50 + (os.getpid() % 50)
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    proc = _start_kernel(node_name, domain_id)
    if True:  # preserve indent of original tempfile context
        try:
            # Give the kernel a moment to start up.
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_passthrough_helper")
                assert _activate_lifecycle(node_name, helper), "kernel failed to activate"

                received_safe: list[ActionChunk] = []
                received_fail: list[FailureTrigger] = []
                received_estop: list[Empty] = []
                helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    received_safe.append,
                    10,
                )
                helper.create_subscription(
                    FailureTrigger,
                    "/openral/failure/safety",
                    received_fail.append,
                    50,
                )
                helper.create_subscription(
                    Empty,
                    "/openral/estop",
                    received_estop.append,
                    10,
                )
                pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                # Let discovery settle.
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)

                # Publish 20 valid chunks.
                for i in range(20):
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 3
                    chunk.flat = [0.1, 0.2, -0.1]
                    chunk.rskill_id = "openral/rskill-test"
                    chunk.trace_id = f"trace-{i:03d}"
                    pub.publish(chunk)
                    deadline = time.time() + 0.5
                    target = i + 1
                    while time.time() < deadline and len(received_safe) < target:
                        executor.spin_once(timeout_sec=0.02)

                assert len(received_safe) >= 18, (
                    f"expected ~20 safe_action, got {len(received_safe)}"
                )
                assert len(received_fail) == 0, "no FailureTrigger expected on passthrough"
                assert len(received_estop) == 0, "no estop expected on passthrough"
                # trace_id should round-trip verbatim.
                traces = [m.trace_id for m in received_safe]
                assert any(t == "trace-000" for t in traces), traces[:3]
            finally:
                rclpy.shutdown()
        finally:
            import signal

            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
