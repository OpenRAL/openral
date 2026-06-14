"""ADR-0020 PR-H — closed-loop Flexiv Rizon 4 twin + C++ safety kernel.

The Rizon 4 is a 7-DoF cobot with whole-body force sensitivity (0.1 N).
This test wires the real ``Rizon4MujocoHAL`` (PR #124 / #129) through
the C++ safety kernel:

    HAL.read_state()
        ↓
    sinusoidal kinematic policy (joint-position, in radians)
        ↓
    /openral/candidate_action  (rclpy publish)
        ↓
    real safety_kernel_node  (subprocess on isolated ROS_DOMAIN_ID)
        ↓
    /openral/safe_action  (rclpy subscribe)
        ↓
    HAL.send_action()  → MuJoCo step

Verifies that the kernel handles the 7-DoF Rizon envelope cleanly under
the loaded ``robots/rizon4/robot.yaml`` ceiling, with no false-positive
violations on a policy that stays well inside joint limits.
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


def test_rizon4_twin_through_safety_kernel() -> None:
    """30 closed-loop steps: Rizon 4 twin ↔ kernel; envelope from manifest."""
    import rclpy
    from openral_core import Action, ControlMode
    from openral_hal import RIZON4_DESCRIPTION, Rizon4MujocoHAL
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_steps = 30
    n_dof = 7  # Rizon 4 is 7-DoF

    # Bring up the MuJoCo HAL twin (mujoco asset download may be slow on
    # first run; HAL connects() synchronously). Bump the staleness limit
    # so the kernel-startup wait (~1.5 s) doesn't blow up our first
    # read_state() — the HAL's default 0.5 s assumes a steady control
    # loop, which we set up after activating the kernel.
    hal = Rizon4MujocoHAL(staleness_limit_s=10.0)
    hal.connect()
    try:
        # Verify the twin is real.
        state0 = hal.read_state()
        assert len(state0.position) == n_dof
        assert state0.name == [j.name for j in RIZON4_DESCRIPTION.joints]

        with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
            node_name = f"safety_kernel_rizon4_{uuid.uuid4().hex[:8]}"
            domain_id = isolated_domain_id()
            proc = start_kernel(RIZON4_DESCRIPTION, node_name, domain_id)
            try:
                time.sleep(1.5)
                rclpy.init()
                try:
                    helper = rclpy.create_node("rizon4_kernel_helper")
                    assert activate_kernel_node(node_name, helper), "kernel activation failed"

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

                    completed = 0
                    for step in range(n_steps):
                        # Sense.
                        state = hal.read_state()
                        # Think: small-amplitude sinusoid around current pose,
                        # safely inside Rizon 4 joint limits (~±2.88 rad on
                        # joints 1/3/5/7, ±2.5 on 2/4, ±2.96 on 6).
                        targets = [
                            float(state.position[j]) + 0.05 * math.sin(step * 0.1 + j)
                            for j in range(n_dof)
                        ]
                        # Build chunk.
                        chunk = ActionChunk()
                        chunk.control_mode = 0  # JOINT_POSITION
                        chunk.horizon = 1
                        chunk.n_dof = n_dof
                        chunk.flat = targets
                        chunk.rskill_id = "openral/rizon4-sinusoid"
                        chunk.trace_id = f"rizon4-{step:03d}"
                        pub.publish(chunk)
                        # Wait for the safe_action.
                        spin_deadline = time.time() + 0.5
                        while time.time() < spin_deadline:
                            if chunk.trace_id in latest_safe:
                                break
                            executor.spin_once(timeout_sec=0.01)
                        if chunk.trace_id not in latest_safe:
                            break
                        # Act.
                        safe = latest_safe[chunk.trace_id]
                        action = Action(
                            control_mode=ControlMode.JOINT_POSITION,
                            horizon=1,
                            joint_targets=[list(safe.flat)],
                        )
                        hal.send_action(action)
                        completed += 1

                    assert completed == n_steps, (
                        f"closed-loop only completed {completed}/{n_steps} steps; "
                        f"failures={[(f.kind, f.evidence_json) for f in failures]}"
                    )
                    assert len(failures) == 0
                    assert len(estops) == 0

                    # Twin actually moved.
                    state_final = hal.read_state()
                    drift = sum(
                        abs(state_final.position[j] - state0.position[j]) for j in range(n_dof)
                    )
                    assert drift > 0.001, f"twin did not actuate (drift={drift})"
                finally:
                    rclpy.shutdown()
            finally:
                terminate_kernel(proc)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            hal.disconnect()
