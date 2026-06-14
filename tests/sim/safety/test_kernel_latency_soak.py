"""ADR-0020 PR-H — kernel latency soak.

Publishes 500 ``ActionChunk`` messages to the real C++ safety kernel at
chunk rate (~30 Hz) and measures round-trip latency
(``/openral/candidate_action`` → ``/openral/safe_action``). Asserts:

* every chunk is republished (no drops),
* every chunk's ``trace_id`` round-trips verbatim,
* p99 round-trip latency is below the chunk-rate budget (33 ms at
  30 Hz — we set 25 ms as the soak target).

Real C++ kernel binary, real rclpy, real openral_msgs IDL, no mocks
(CLAUDE.md §1.11).
"""

from __future__ import annotations

import statistics
import time
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from tests.sim.safety._kernel_subprocess import (
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_ENVELOPE_PARAMS: dict[str, object] = {
    "n_dof": 6,
    "robot_name": "soak_test",
    "joint_position_min": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
    "joint_position_max": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "joint_velocity_max": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "joint_torque_max": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "max_ee_speed_m_s": 1.0,
    "max_ee_accel_m_s2": 5.0,
    "max_force_n": 50.0,
    "max_torque_nm": 10.0,
    "contact_force_threshold_n": 20.0,
    "deadman_required": False,
}


# A 25 ms p99 target is conservative — the kernel's allocation-free
# validator is sub-millisecond; the rest is rclpy publish + DDS
# transport. On a busy CI runner with virtualized network we tolerate
# more. The hard ceiling is the 33 ms 30 Hz chunk budget; failing
# this assertion means the kernel is missing its real-time contract.
P99_BUDGET_MS = 25.0


def test_kernel_round_trip_latency_p99_under_budget() -> None:
    """500 chunks → measure p99 round-trip; must beat the 30 Hz chunk budget."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    n_chunks = 500
    publish_period_s = 1.0 / 30.0  # chunk-rate cap (30 Hz)
    send_at: dict[str, float] = {}
    recv_at: dict[str, float] = {}
    received_chunks: list[ActionChunk] = []
    failures: list[FailureTrigger] = []
    estops: list[Empty] = []

    domain_id = isolated_domain_id()

    node_name = f"safety_kernel_soak_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_ENVELOPE_PARAMS, node_name, domain_id)
    if True:  # preserve indent of legacy tempfile block
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_soak_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"

                def on_safe(msg: ActionChunk) -> None:
                    recv_at[msg.trace_id] = time.perf_counter()
                    received_chunks.append(msg)

                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    on_safe,
                    100,
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
                pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 100)

                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                # Discovery.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                        break
                    executor.spin_once(timeout_sec=0.05)
                assert pub.get_subscription_count() >= 1
                assert safe_sub.get_publisher_count() >= 1

                # Publish n_chunks chunks at chunk-rate; spin between each.
                start = time.perf_counter()
                for i in range(n_chunks):
                    chunk = ActionChunk()
                    chunk.control_mode = 0
                    chunk.horizon = 1
                    chunk.n_dof = 6
                    chunk.flat = [0.05 * (i % 10), 0.0, 0.0, 0.0, 0.0, 0.5]
                    chunk.rskill_id = "openral/soak-skill"
                    chunk.trace_id = f"soak-{i:04d}"
                    send_at[chunk.trace_id] = time.perf_counter()
                    pub.publish(chunk)
                    # Spin once between publishes — runs at ~30 Hz, so
                    # the executor drains any pending safe_action messages.
                    target = start + (i + 1) * publish_period_s
                    while time.perf_counter() < target:
                        executor.spin_once(timeout_sec=0.001)
                # Final drain.
                drain_deadline = time.perf_counter() + 5.0
                while time.perf_counter() < drain_deadline and len(received_chunks) < n_chunks:
                    executor.spin_once(timeout_sec=0.05)

                # Assertions.
                assert len(received_chunks) == n_chunks, (
                    f"expected {n_chunks} safe_action messages, got {len(received_chunks)}"
                )
                assert len(failures) == 0, (
                    f"unexpected violations on the safe envelope: {len(failures)}"
                )
                assert len(estops) == 0

                # Latency stats.
                latencies_ms = [(recv_at[t] - send_at[t]) * 1000.0 for t in send_at if t in recv_at]
                assert len(latencies_ms) == n_chunks
                latencies_ms.sort()
                p50 = statistics.median(latencies_ms)
                p99 = latencies_ms[int(0.99 * n_chunks)]
                p_max = latencies_ms[-1]
                print(
                    f"\nkernel-soak latency over {n_chunks} chunks: "
                    f"p50={p50:.3f} ms  p99={p99:.3f} ms  max={p_max:.3f} ms"
                )
                assert p99 < P99_BUDGET_MS, (
                    f"kernel p99 round-trip latency {p99:.3f} ms exceeds "
                    f"the {P99_BUDGET_MS:.1f} ms 30 Hz chunk budget"
                )
            finally:
                rclpy.shutdown()
        finally:
            terminate_kernel(proc)
