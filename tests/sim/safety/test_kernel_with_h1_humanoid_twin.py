"""ADR-0020 PR-H — closed-loop Unitree H1 humanoid twin + C++ safety kernel.

H1 is a 19-DoF humanoid with a floating base; the largest single-robot
envelope this branch tests through the kernel. Exercises the kernel's
allocation-free validator at a meaningful DoF count and confirms it
handles the manifest's per-joint position limits across the full
humanoid body.

20 closed-loop steps; deterministic kinematic policy that stays inside
the H1 envelope per slot. Software PD inside the HAL handles the
motor → position conversion.
"""

from __future__ import annotations

import math
import tempfile
import time
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")

from tests.sim.safety._kernel_subprocess import (
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)


def test_h1_humanoid_twin_closed_loop_through_kernel() -> None:
    """20 steps of 19-DoF humanoid sinusoid through the kernel; no violations."""
    import rclpy
    from openral_core import Action, ControlMode
    from openral_hal import H1_DESCRIPTION, H1MujocoHAL
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_steps = 20
    n_dof = 19  # Unitree H1

    hal = H1MujocoHAL(staleness_limit_s=10.0)
    hal.connect()
    try:
        state0 = hal.read_state()
        assert len(state0.position) == n_dof

        # Per-joint limits from the H1 manifest — feed to the kernel via
        # envelope_loader so the kernel knows the real ceiling.
        per_joint_limits = [
            (j.position_limits[0], j.position_limits[1])
            if j.position_limits is not None
            else (-math.inf, math.inf)
            for j in H1_DESCRIPTION.joints
            if j.joint_type.value in {"revolute", "prismatic", "continuous"}
        ]
        assert len(per_joint_limits) == n_dof

        with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
            node_name = f"safety_kernel_h1_{uuid.uuid4().hex[:8]}"
            domain_id = isolated_domain_id()
            proc = start_kernel(H1_DESCRIPTION, node_name, domain_id)
            try:
                time.sleep(1.5)
                rclpy.init()
                try:
                    helper = rclpy.create_node("h1_kernel_helper")
                    assert activate_kernel_node(node_name, helper)

                    latest_safe: dict[str, ActionChunk] = {}
                    failures: list[FailureTrigger] = []
                    estops: list[Empty] = []
                    safe_sub = helper.create_subscription(
                        ActionChunk,
                        "/openral/safe_action",
                        lambda m: latest_safe.__setitem__(m.trace_id, m),
                        10,
                    )
                    helper.create_subscription(
                        FailureTrigger,
                        "/openral/failure/safety",
                        failures.append,
                        50,
                    )
                    helper.create_subscription(
                        Empty,
                        "/openral/estop",
                        estops.append,
                        10,
                    )
                    pub = helper.create_publisher(
                        ActionChunk,
                        "/openral/candidate_action",
                        10,
                    )

                    executor = SingleThreadedExecutor()
                    executor.add_node(helper)
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        if (
                            pub.get_subscription_count() >= 1
                            and safe_sub.get_publisher_count() >= 1
                        ):
                            break
                        executor.spin_once(timeout_sec=0.05)

                    eps = 0.001
                    completed = 0
                    for step in range(n_steps):
                        state = hal.read_state()
                        targets = []
                        for j in range(n_dof):
                            base = float(state.position[j])
                            lo, hi = per_joint_limits[j]
                            t = base + 0.01 * math.sin(step * 0.1 + j)
                            t = max(lo + eps, min(hi - eps, t))
                            targets.append(t)
                        chunk = ActionChunk()
                        chunk.control_mode = 0
                        chunk.horizon = 1
                        chunk.n_dof = n_dof
                        chunk.flat = targets
                        chunk.rskill_id = "openral/h1-humanoid-sinusoid"
                        chunk.trace_id = f"h1-{step:03d}"
                        pub.publish(chunk)
                        spin_deadline = time.time() + 0.5
                        while time.time() < spin_deadline:
                            if chunk.trace_id in latest_safe:
                                break
                            executor.spin_once(timeout_sec=0.01)
                        if chunk.trace_id not in latest_safe:
                            break
                        safe = latest_safe[chunk.trace_id]
                        action = Action(
                            control_mode=ControlMode.JOINT_POSITION,
                            horizon=1,
                            joint_targets=[list(safe.flat)],
                        )
                        hal.send_action(action)
                        completed += 1

                    assert completed == n_steps, (
                        f"closed-loop only completed {completed}/{n_steps}; "
                        f"failures={[(f.kind, f.evidence_json) for f in failures]}"
                    )
                    assert len(failures) == 0
                    assert len(estops) == 0
                finally:
                    rclpy.shutdown()
            finally:
                terminate_kernel(proc)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            hal.disconnect()
