"""ADR-0020 PR-H — closed-loop Enactic OpenArm v2 twin + C++ safety kernel.

OpenArm v2 is a 16-DoF bimanual humanoid arm (7 arm + 1 gripper per
side). This exercises the kernel on the largest single-robot envelope
in the tree, with heterogeneous per-joint position limits across two
arms.

Real ``OpenArmMujocoHAL`` (PR #124 / #129) closed loop through the
kernel for 20 steps. The policy is a deterministic sinusoid scaled to
stay well inside the v2 ctrlrange (the manifest's joint_position_min /
max), guaranteeing no false-positive violations.

Also exercises a deliberate-violation path: command joint 0 (left arm
shoulder pan) to its upper limit + 1 rad → kernel must drop the chunk,
fire ``KIND_WORKSPACE``, and latch.
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


def test_openarm_v2_twin_closed_loop_through_kernel() -> None:
    """20 steps of bimanual sinusoid through the kernel; no violations."""
    import rclpy
    from openral_core import Action, ControlMode
    from openral_hal import OPENARM_DESCRIPTION, OpenArmMujocoHAL
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_steps = 20
    n_dof = 16

    hal = OpenArmMujocoHAL(staleness_limit_s=10.0)
    hal.connect()
    try:
        state0 = hal.read_state()
        assert len(state0.position) == n_dof
        with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
            node_name = f"safety_kernel_openarm_{uuid.uuid4().hex[:8]}"
            domain_id = isolated_domain_id()
            proc = start_kernel(OPENARM_DESCRIPTION, node_name, domain_id)
            try:
                time.sleep(1.5)
                rclpy.init()
                try:
                    helper = rclpy.create_node("openarm_kernel_helper")
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

                    # OpenArm v2 joint limits (per _OPENARM_*_POSITION_LIMITS).
                    # We compute each per-joint safe target by clamping
                    # ``state + small sinusoid`` inside the manifest's
                    # ``[min + ε, max - ε]`` for that slot — the kernel's
                    # envelope is loaded straight from OPENARM_DESCRIPTION
                    # so this guarantees no false-positive violation. The
                    # sinusoid amplitudes are tiny (≤ 0.02 rad) so the
                    # MuJoCo PD controllers track them every tick.
                    eps = 0.001
                    per_joint_limits = [
                        (j.position_limits[0], j.position_limits[1])
                        if j.position_limits is not None
                        else (-math.inf, math.inf)
                        for j in OPENARM_DESCRIPTION.joints
                        if j.joint_type.value in {"revolute", "prismatic", "continuous"}
                    ]
                    assert len(per_joint_limits) == n_dof

                    completed = 0
                    for step in range(n_steps):
                        state = hal.read_state()
                        targets = []
                        for j in range(n_dof):
                            base = float(state.position[j])
                            lo, hi = per_joint_limits[j]
                            delta = 0.01 * math.sin(step * 0.1 + j)
                            t = base + delta
                            # Stay safely inside the manifest's ctrlrange.
                            t = max(lo + eps, min(hi - eps, t))
                            targets.append(t)
                        chunk = ActionChunk()
                        chunk.control_mode = 0
                        chunk.horizon = 1
                        chunk.n_dof = n_dof
                        chunk.flat = targets
                        chunk.rskill_id = "openral/openarm-bimanual-sinusoid"
                        chunk.trace_id = f"openarm-{step:03d}"
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

                    state_final = hal.read_state()
                    drift = sum(
                        abs(state_final.position[j] - state0.position[j]) for j in range(n_dof)
                    )
                    assert drift > 0.0, f"openarm twin did not actuate (drift={drift})"
                finally:
                    rclpy.shutdown()
            finally:
                terminate_kernel(proc)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            hal.disconnect()


def test_openarm_envelope_violation_latches_kernel_and_protects_hal() -> None:
    """Inject a command beyond joint-0 limit; kernel must intercept it."""
    import rclpy
    from openral_hal import OPENARM_DESCRIPTION, OpenArmMujocoHAL
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_dof = 16

    hal = OpenArmMujocoHAL(staleness_limit_s=10.0)
    hal.connect()
    try:
        # Joint 0 (left_joint1) ctrlrange is (-3.49066, 1.39626) per the v2
        # MJCF. Commanding 5.0 rad on slot 0 is a clean envelope violation.
        state0 = hal.read_state()
        joint0_initial = state0.position[0]

        with tempfile.TemporaryDirectory():  # ADR-0020 PR-K: no temp envelope file needed
            node_name = f"safety_kernel_openarm_v_{uuid.uuid4().hex[:8]}"
            domain_id = isolated_domain_id()
            proc = start_kernel(OPENARM_DESCRIPTION, node_name, domain_id)
            try:
                time.sleep(1.5)
                rclpy.init()
                try:
                    helper = rclpy.create_node("openarm_violation_helper")
                    assert activate_kernel_node(node_name, helper)

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
                    pub = helper.create_publisher(
                        ActionChunk,
                        "/openral/candidate_action",
                        10,
                    )

                    executor = SingleThreadedExecutor()
                    executor.add_node(helper)
                    deadline = time.time() + 3.0
                    while time.time() < deadline:
                        if pub.get_subscription_count() >= 1:
                            break
                        executor.spin_once(timeout_sec=0.05)

                    bad_targets = [float(state0.position[j]) for j in range(n_dof)]
                    bad_targets[0] = 5.0  # outside left_joint1 ctrlrange
                    chunk = ActionChunk()
                    chunk.control_mode = 0
                    chunk.horizon = 1
                    chunk.n_dof = n_dof
                    chunk.flat = bad_targets
                    chunk.rskill_id = "openral/openarm-violator"
                    chunk.trace_id = "openarm-violation-001"
                    pub.publish(chunk)

                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        if len(received_fail) >= 1 and len(received_estop) >= 1:
                            break
                        executor.spin_once(timeout_sec=0.05)

                    # Kernel must have intercepted; the HAL has NOT been
                    # told to actuate the violating target.
                    assert len(received_fail) == 1
                    assert received_fail[0].kind == FailureTrigger.KIND_WORKSPACE
                    assert received_fail[0].severity == FailureTrigger.SEVERITY_ABORT
                    assert len(received_estop) == 1
                    assert len(received_safe) == 0

                    # Sanity: the HAL's joint 0 is still where it started
                    # (we never called send_action with the bad target).
                    state_after = hal.read_state()
                    assert abs(state_after.position[0] - joint0_initial) < 0.01, (
                        "HAL must not have moved — kernel intercepts before "
                        "the action reaches the actuator"
                    )
                finally:
                    rclpy.shutdown()
            finally:
                terminate_kernel(proc)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            hal.disconnect()
