"""ADR-0020 PR-H — GPU-aware soak test.

When a CUDA-capable GPU is available, run real ``torch`` policy inference
on a small set of synthetic observations and feed the resulting action
chunks through the C++ safety kernel via a child rclpy publisher. Assert:

* GPU is detected (or the test skips).
* All N inferences produce numerically-finite action chunks.
* The kernel republishes every chunk on ``/openral/safe_action`` with
  the source ``trace_id`` intact (no false-positive violations under a
  default-envelope so100 ceiling).

This is the "the GPU code path actually runs and the kernel handles its
output cleanly" smoke. It does NOT load a full VLA + sim environment
because that requires the F1 ``rskill_runner_node`` (still pending — see
ADR-0020 §"Rollout"). Instead we use a deterministic PyTorch tensor
network to emit shape-correct ActionChunks; the GPU code path is
exercised through the tensor ops.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
import uuid
from typing import Any

import pytest

from tests.sim.safety._kernel_subprocess import start_kernel, terminate_kernel

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU not available; skip GPU-aware sim smoke",
)


_ENVELOPE_PARAMS: dict[str, object] = {
    "n_dof": 6,
    "robot_name": "gpu_smoke",
    "joint_position_min": [-2.0944, -1.7453, -1.7453, -1.7453, -2.7925, 0.0],
    "joint_position_max": [2.0944, 1.7453, 1.7453, 1.7453, 2.7925, 1.0],
    "joint_velocity_max": [3.15, 3.15, 3.15, 3.15, 3.15, 3.15],
    "joint_torque_max": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "max_ee_speed_m_s": 0.5,
    "max_ee_accel_m_s2": 2.0,
    "max_force_n": 10.0,
    "max_torque_nm": 3.0,
    "contact_force_threshold_n": 5.0,
    "deadman_required": False,
}


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


def test_gpu_policy_inference_feeds_safety_kernel_cleanly() -> None:
    """Real GPU tensor ops → ActionChunk → C++ kernel → /openral/safe_action."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    # 1. Verify the GPU is real and torch can use it.
    device = torch.device("cuda")
    obs_batch = torch.randn(8, 32, device=device)  # 8 observations × 32 features
    # Tiny policy MLP — real CUDA matmul + nonlinearity. Output: 6-dof action.
    weights = torch.randn(32, 6, device=device) * 0.05
    bias = torch.randn(6, device=device) * 0.01
    raw = torch.tanh(obs_batch @ weights + bias) * 0.4  # bounded in [-0.4, 0.4]
    # The 6th DoF in the so100 envelope is the gripper with limits
    # [0.0, 1.0]; remap that channel into [0.1, 0.9] so it sits inside
    # the gripper-safe range. Real policy outputs come from a checkpoint
    # trained on the embodiment so this remap is artificial; the test's
    # point is to exercise the C++ kernel with GPU-produced numerics,
    # not to validate a real VLA's compatibility with so100.
    gripper_remap = (raw[:, 5] + 1.0) * 0.4 + 0.1  # [-1,1] → [0.1, 0.9]
    raw[:, 5] = gripper_remap
    actions = raw
    actions_cpu = actions.cpu().tolist()
    # Force a CUDA sync so the ops genuinely completed on-device.
    torch.cuda.synchronize()
    assert all(all(abs(v) <= 1.0 for v in row) for row in actions_cpu)
    # Verify the gripper channel is safely inside [0, 1].
    assert all(0.0 < row[5] < 1.0 for row in actions_cpu)

    # 2. Bring up the C++ kernel via the shared helper.
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        node_name = f"safety_kernel_gpu_test_{uuid.uuid4().hex[:8]}"
        domain_id = 50 + (os.getpid() % 50)
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        proc = start_kernel(
            _ENVELOPE_PARAMS,
            node_name,
            domain_id,
            log_path=log_path,
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_gpu_helper")
                assert _activate(node_name, helper), "kernel activation failed"
                received: list[ActionChunk] = []
                failures: list[FailureTrigger] = []
                estops: list[Empty] = []
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    received.append,
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
                # Wait for bi-directional discovery: the kernel must
                # subscribe to /openral/candidate_action AND publish on
                # /openral/safe_action before we publish our first chunk.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    sub_ready = pub.get_subscription_count() >= 1
                    pub_ready = safe_sub.get_publisher_count() >= 1
                    if sub_ready and pub_ready:
                        break
                    executor.spin_once(timeout_sec=0.05)
                assert pub.get_subscription_count() >= 1, (
                    "kernel did not subscribe to /openral/candidate_action"
                )
                assert safe_sub.get_publisher_count() >= 1, (
                    "kernel did not publish /openral/safe_action"
                )

                # 3. Publish each GPU-produced action. Spin the executor
                # extensively between publishes so each chunk's round-trip
                # completes before the next publish — RELIABLE+VOLATILE+
                # KEEP_LAST=1 on the candidate_action subscription means a
                # back-to-back burst can coalesce in the kernel's input
                # queue. Throttling here matches how rskill_runner_node
                # publishes at chunk-rate (≤30 Hz).
                for i, row in enumerate(actions_cpu):
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 6
                    chunk.flat = list(row)
                    chunk.rskill_id = "openral/gpu-smoke-skill"
                    chunk.trace_id = f"gpu-trace-{i:03d}"
                    pub.publish(chunk)
                    # Spin AT LEAST 100ms (5x typical kernel roundtrip
                    # latency on a desktop) to drain the safe_action
                    # response before the next publish.
                    spin_deadline = time.time() + 0.15
                    while time.time() < spin_deadline:
                        executor.spin_once(timeout_sec=0.02)
                # Final drain — anything still in flight should appear.
                final_deadline = time.time() + 1.0
                while time.time() < final_deadline and len(received) < len(actions_cpu):
                    executor.spin_once(timeout_sec=0.05)

                # On failure, surface the kernel's log to help debug.
                if len(received) != len(actions_cpu):
                    try:
                        log_contents = log_path.read_text()
                        print(f"\n=== kernel log ===\n{log_contents}\n==================")
                    except Exception:
                        pass

                # 4. Assertions: every GPU action republished, no
                # violations on the so100-shaped envelope (which permits
                # |joint| ≤ 1.7 rad — our bounded-to-0.5 actions are well
                # inside).
                assert len(received) == len(actions_cpu), (
                    f"expected {len(actions_cpu)} safe_action messages, got {len(received)}"
                )
                assert len(failures) == 0, (
                    f"unexpected FailureTrigger on GPU-produced chunks: {failures}"
                )
                assert len(estops) == 0
                # trace_id round-trips verbatim.
                trace_ids = {m.trace_id for m in received}
                assert trace_ids == {f"gpu-trace-{i:03d}" for i in range(len(actions_cpu))}
            finally:
                rclpy.shutdown()
        finally:
            terminate_kernel(proc)
