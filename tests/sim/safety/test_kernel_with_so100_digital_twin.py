"""ADR-0020 PR-H — closed-loop SO-100 digital twin + C++ safety kernel.

Builds an in-process sense-think-act loop where every action passes
through the real C++ safety kernel before reaching the digital twin:

    SO100DigitalTwin.get_observation()
        ↓
    deterministic test-policy (Python; no GPU dep)
        ↓
    /openral/candidate_action  (publish via rclpy)
        ↓
    real C++ safety_kernel_node  (subprocess)
        ↓
    /openral/safe_action  (subscribe via rclpy)
        ↓
    SO100DigitalTwin.send_action()

This is the closest we can get to ``ral run`` against a real arm
without F1's rskill_runner_node — it proves the kernel sits cleanly
between policy and HAL, that the digital twin can be driven through
the topic contract, and that no false-positive violations occur when
the policy stays inside the SO-100 envelope.

No GPU required (the test policy is a deterministic kinematic
function). No mocks (CLAUDE.md §1.11) — real HAL, real kernel binary,
real ROS messages.
"""

from __future__ import annotations

import math
import time
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("openral_hal")

from tests.sim.safety._kernel_subprocess import (
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# Test envelope: chosen to admit the policy sweep below. Matches what
# `kernel_params_from_envelope(compute_intersection(SO100_DESCRIPTION,
# None))` would emit for the SO-100 follower; the cartesian workspace
# box is added here (not declared in the manifest) so we exercise that
# kernel field too.
_ENVELOPE_PARAMS: dict[str, object] = {
    "n_dof": 6,
    "robot_name": "so100_follower",
    "joint_position_min": [-2.0944, -1.7453, -1.7453, -1.7453, -2.7925, 0.0],
    "joint_position_max": [2.0944, 1.7453, 1.7453, 1.7453, 2.7925, 1.0],
    "joint_velocity_max": [3.15, 3.15, 3.15, 3.15, 3.15, 3.15],
    "joint_torque_max": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "workspace_box_min_xyz": [-0.4, -0.4, 0.0],
    "workspace_box_max_xyz": [0.4, 0.4, 0.6],
    "max_ee_speed_m_s": 0.5,
    "max_ee_accel_m_s2": 2.0,
    "max_force_n": 10.0,
    "max_torque_nm": 3.0,
    "contact_force_threshold_n": 5.0,
    "deadman_required": False,
}


# SO-100 joint names match openral_hal.so100_sim._JOINT_NAMES.
_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _test_policy(obs: dict[str, float], step_idx: int, n_dof: int = 6) -> list[float]:
    """Deterministic sinusoidal policy that stays inside the SO-100 envelope.

    Each joint gets a small-amplitude sinusoid centered on its current
    position; the gripper oscillates inside ``[0.3, 0.7]``. No GPU,
    no torch — purely kinematic so the test runs anywhere.
    """
    target = []
    for i, name in enumerate(_JOINT_NAMES[:n_dof]):
        center = float(obs[f"{name}.pos"]) if f"{name}.pos" in obs else 0.0
        # SO-100 native units: degrees for arm joints, [0,100] for gripper.
        # The envelope is in radians + [0,1] (the manifest unit system).
        if i == 5:  # gripper
            cmd_deg_or_pct = 50.0 + 20.0 * math.sin(step_idx * 0.1)  # [30, 70]
        else:
            cmd_deg_or_pct = center + 5.0 * math.sin(step_idx * 0.1 + i)
        # Convert lerobot deg → radians for arm joints; gripper [0,100] → [0,1].
        if i == 5:
            target.append(cmd_deg_or_pct / 100.0)
        else:
            target.append(math.radians(cmd_deg_or_pct))
    return target


def test_so100_digital_twin_through_safety_kernel() -> None:
    """50 closed-loop steps: twin → policy → kernel → twin, all inside envelope."""
    import rclpy
    from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_steps = 50

    node_name = f"safety_kernel_twin_{uuid.uuid4().hex[:8]}"
    domain_id = isolated_domain_id()
    proc = start_kernel(_ENVELOPE_PARAMS, node_name, domain_id)
    if True:  # preserve existing indent for the rest of the body
        # Bring up the digital twin.
        twin = SO100DigitalTwin(SO100DigitalTwinConfig())
        twin.connect(calibrate=False)

        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("so100_twin_kernel_helper")
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
                pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                # Discovery.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                        break
                    executor.spin_once(timeout_sec=0.05)

                # Closed-loop sense-think-act with the kernel inline.
                steps_completed = 0
                for step in range(n_steps):
                    # 1. Sense: read the twin's current joint positions.
                    obs = twin.get_observation()
                    # 2. Think: compute a target action.
                    target = _test_policy(obs, step)
                    # 3. Build the ActionChunk and publish.
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 6
                    chunk.flat = target
                    chunk.rskill_id = "openral/twin-test-policy"
                    chunk.trace_id = f"twin-{step:03d}"
                    pub.publish(chunk)
                    # 4. Wait for the kernel's safe_action response.
                    deadline = time.time() + 0.5
                    while time.time() < deadline:
                        if chunk.trace_id in latest_safe:
                            break
                        executor.spin_once(timeout_sec=0.01)
                    if chunk.trace_id not in latest_safe:
                        # The kernel dropped this step — bail rather than
                        # silently continue (CLAUDE.md §1.4).
                        break
                    # 5. Act: apply the validated action to the twin.
                    # Convert from envelope units back to lerobot native.
                    safe = latest_safe[chunk.trace_id]
                    action_dict = {}
                    for i, name in enumerate(_JOINT_NAMES):
                        if i == 5:  # gripper [0, 1] → [0, 100]
                            action_dict[f"{name}.pos"] = safe.flat[i] * 100.0
                        else:  # rad → deg
                            action_dict[f"{name}.pos"] = math.degrees(safe.flat[i])
                    twin.send_action(action_dict)
                    steps_completed += 1

                assert steps_completed == n_steps, (
                    f"closed-loop only completed {steps_completed}/{n_steps} steps"
                )
                assert len(failures) == 0, (
                    f"unexpected violations on sinusoidal in-envelope policy: {failures}"
                )
                assert len(estops) == 0

                # Verify the twin actually moved (state changed from initial).
                final_obs = twin.get_observation()
                # At step ~50 with 0.1 phase the policy has rotated through
                # all joints; at least one position should differ from zero.
                assert any(abs(final_obs[f"{name}.pos"]) > 0.01 for name in _JOINT_NAMES), (
                    "twin did not actuate"
                )
            finally:
                rclpy.shutdown()
        finally:
            import contextlib

            with contextlib.suppress(Exception):
                twin.disconnect()
            terminate_kernel(proc)
