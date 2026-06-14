"""ADR-0020 PR-H — envelope-violation flow against the real C++ kernel.

Brings up the real ``safety_kernel_node`` and publishes a chunk that
violates the joint-position envelope. Asserts:

* No ``/openral/safe_action`` is republished.
* Exactly one ``/openral/failure/safety`` with
  ``kind=KIND_WORKSPACE``, ``severity=SEVERITY_ABORT``, and an
  ``evidence_json`` payload that round-trips through
  :class:`openral_core.FailureEvidence`.
* ``/openral/estop`` fires.

No mocks (CLAUDE.md §1.11): real rclpy, real openral_msgs IDL, real
``safety_kernel_node`` binary.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")


# Tight envelope: position_max=0.5 forces the +0.6 chunk to violate
# the joint-position bound. Kernel reads each field as a ROS parameter
# (ADR-0020 PR-K).
_KERNEL_PARAM_ARGS: list[str] = [
    "-p",
    "n_dof:=3",
    "-p",
    "robot_name:=launchtest",
    "-p",
    "joint_position_min:=[-0.5, -0.5, -0.5]",
    "-p",
    "joint_position_max:=[0.5, 0.5, 0.5]",
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
    "estop_reset_cooldown_s:=0.05",
]


def _start_kernel(node_name: str, domain_id: int) -> Any:
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
        start_new_session=True,
    )


def _activate(node_name: str, helper: Any) -> bool:
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


def _spin_until(executor: Any, predicate: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_envelope_violation_fires_failure_and_estop() -> None:
    """Workspace-violating chunk → KIND_WORKSPACE + estop, no safe_action."""
    import rclpy
    from openral_core import FailureEvidence  # type: ignore[attr-defined]
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from pydantic import TypeAdapter
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
        node_name = f"safety_kernel_violation_test_{uuid.uuid4().hex[:8]}"
        domain_id = 50 + (os.getpid() % 50)
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        proc = _start_kernel(node_name, domain_id)
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_violation_helper")
                assert _activate(node_name, helper), "kernel failed to activate"

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
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)

                # Publish a chunk that violates the envelope on joint 1
                # (position 5.0 > max 0.5).
                bad = ActionChunk()
                bad.control_mode = 0
                bad.horizon = 1
                bad.n_dof = 3
                bad.flat = [0.0, 5.0, 0.0]
                bad.rskill_id = "openral/rskill-violator"
                bad.trace_id = "trace-violation-001"
                pub.publish(bad)

                assert _spin_until(executor, lambda: len(received_fail) >= 1)
                assert _spin_until(executor, lambda: len(received_estop) >= 1)
                assert len(received_safe) == 0, (
                    "violating chunk must not be republished as safe_action"
                )

                ft = received_fail[0]
                assert ft.kind == FailureTrigger.KIND_WORKSPACE
                assert ft.severity == FailureTrigger.SEVERITY_ABORT
                assert ft.rskill_id == "openral/rskill-violator"
                assert ft.trace_id == "trace-violation-001"

                # evidence_json must round-trip through the Pydantic
                # FailureEvidence discriminated union — the bridge to
                # the reasoner relies on this contract.
                evidence_obj = json.loads(ft.evidence_json)
                assert evidence_obj["kind"] == "workspace"
                adapter = TypeAdapter(FailureEvidence)
                parsed = adapter.validate_python(evidence_obj)
                assert parsed.kind == "workspace"
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


def test_estop_latch_blocks_subsequent_chunks() -> None:
    """After a violation latches the kernel, valid chunks also drop."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
        node_name = f"safety_kernel_latch_test_{uuid.uuid4().hex[:8]}"
        domain_id = 50 + (os.getpid() % 50)
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        proc = _start_kernel(node_name, domain_id)
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_latch_helper")
                assert _activate(node_name, helper)
                received_safe: list[ActionChunk] = []
                received_estop: list[Empty] = []
                helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    received_safe.append,
                    10,
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
                deadline = time.time() + 1.5
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)

                # Step 1: violate. Latch should set.
                bad = ActionChunk()
                bad.control_mode = 0
                bad.horizon = 1
                bad.n_dof = 3
                bad.flat = [0.0, 5.0, 0.0]
                bad.trace_id = "trace-bad"
                pub.publish(bad)
                assert _spin_until(executor, lambda: len(received_estop) >= 1)

                # Step 2: publish a VALID chunk. Latch must block it.
                ok = ActionChunk()
                ok.control_mode = 0
                ok.horizon = 1
                ok.n_dof = 3
                ok.flat = [0.0, 0.0, 0.0]
                ok.trace_id = "trace-ok"
                pub.publish(ok)
                deadline = time.time() + 0.5
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)
                assert len(received_safe) == 0, "latched kernel must drop subsequent chunks"
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
