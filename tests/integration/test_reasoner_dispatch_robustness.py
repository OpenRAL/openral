"""Live ROS tests for the execute_rskill dispatch-path robustness fixes.

Three seams from the PR #19 xhigh code review, all on the busy-latch path:

1. **Dispatch-phase watchdog** — `_rskill_inflight` latches BEFORE the async
   send, but rclpy futures never time out on their own: an action server that
   dies (or stops spinning) after the readiness probe never sends a goal
   response, so nothing would ever release the latch and every future dispatch
   is refused as busy. The watchdog (``dispatch_watchdog_s`` param) bounds the
   send→goal-response window: on expiry it releases the latch and emits a
   ``KIND_CONTROLLER`` FailureTrigger (state=dispatch_timeout).

2. **Resident-VLA probe exemption** — the live free-VRAM probe compares free
   VRAM against the manifest's declared footprint, but after a goal ends the
   runner keeps the policy warm: its own residency is why free is low, so a
   re-dispatch of the SAME skill was falsely refused. The probe is skipped
   when ``call.rskill_id`` matches the last accepted skill.
3. **Dispatch generations** — a goal response arriving after its watchdog
   expired must be canceled as stale; it cannot cancel a newer watchdog or
   overwrite the newer goal's in-flight state.

Real reasoner node + real ``ExecuteRskill`` ActionServer + real DDS graph
(CLAUDE.md §1.11); the wedge in test 1 is produced by a real server whose node
is simply never spun — endpoints match on the graph (the readiness probe
passes) but no goal response is ever produced, exactly the died-after-probe
failure mode. The only doubles: the free-VRAM reading is pinned via
monkeypatch (process boundary — ``nvidia-smi``) and the fixture manifest is
injected at the guard's seam.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite::

    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_dispatch_robustness.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SMOLVLA = _REPO / "rskills" / "smolvla-libero" / "rskill.yaml"
_VLA_ID = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"


def _spin_until(executor: Any, predicate: Any, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_dispatch_watchdog_releases_a_wedged_busy_latch() -> None:
    """A server that never answers must not wedge the actuation path forever.

    The server node exists on the graph (so ``server_is_ready`` passes and the
    goal is actually sent) but is never added to an executor — the goal-request
    service callback never runs and the ``send_goal_async`` future never
    resolves. Before the watchdog, ``_rskill_inflight`` stayed latched forever
    and the reasoner refused every subsequent dispatch as busy.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    failures: list[Any] = []

    rclpy.init()
    try:
        # The dispatches are driven directly on the node's own method below;
        # the client only exists so on_configure passes (heartbeat ticks pick
        # the benign wait).
        from openral_core import WaitTool

        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_VLA_ID})),
            tick_hz=2.0,
        )
        reasoner.set_parameters([Parameter("dispatch_watchdog_s", value=2.0)])
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        # Real server on the graph — endpoints match, so the readiness probe
        # passes — but its node is NEVER spun: no goal response, ever.
        dead_server_node = rclpy.create_node("openral_test_vla_server_unresponsive")
        ActionServer(
            dead_server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=lambda gh: ExecuteRskill.Result(),
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        sub_node = rclpy.create_node("openral_test_watchdog_failure_sub")
        fail_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sub_node.create_subscription(
            FailureTrigger, "/openral/failure/rskill", failures.append, fail_qos
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)
        _spin_until(executor, lambda: False, 1.5)  # DDS discovery

        call = ExecuteRskillTool(rskill_id=_VLA_ID, prompt="wedge me", deadline_s=0.0)
        reasoner._dispatch_execute_rskill(call, traceparent=None)
        assert reasoner._rskill_inflight, "latch must be set for the dispatch window"

        released = _spin_until(executor, lambda: not reasoner._rskill_inflight, 6.0)
        assert released, "watchdog never released the wedged busy latch"
        assert reasoner._renderer.inflight_skill is None, "in_flight context line must clear"

        _spin_until(executor, lambda: bool(failures), 2.0)
        timeout_failures = [m for m in failures if "dispatch_timeout" in m.evidence_json]
        assert timeout_failures, (
            f"no dispatch_timeout FailureTrigger (failures={[m.evidence_json for m in failures]})"
        )

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        dead_server_node.destroy_node()
        sub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()


def test_late_goal_response_cannot_replace_a_new_dispatch() -> None:
    """A watchdog-expired goal response must be canceled as stale.

    Goal A responds after its watchdog releases the latch. Goal B starts while
    A is still pending. When A finally accepts, it must not cancel B's watchdog
    or replace B's in-flight state.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool, WaitTool
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    old_id = "openral/rskill-watchdog-old"
    new_id = "openral/rskill-watchdog-new"
    old_cancelled = threading.Event()
    new_completed = threading.Event()

    rclpy.init()
    try:
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset({old_id, new_id})),
            tick_hz=2.0,
        )
        reasoner.set_parameters([Parameter("dispatch_watchdog_s", value=0.5)])
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_vla_server_delayed_response")
        server_group = ReentrantCallbackGroup()

        def _goal(request: Any) -> GoalResponse:
            time.sleep(1.2 if request.rskill_id == old_id else 2.0)
            return GoalResponse.ACCEPT

        def _execute(goal_handle: Any) -> Any:
            result = ExecuteRskill.Result()
            if goal_handle.request.rskill_id == old_id:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not goal_handle.is_cancel_requested:
                    time.sleep(0.02)
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    old_cancelled.set()
                    result.success = False
                    result.failure_reason = "stale dispatch canceled"
                    return result
                goal_handle.abort()
                result.success = False
                result.failure_reason = "stale dispatch was not canceled"
                return result
            goal_handle.succeed()
            new_completed.set()
            result.success = True
            result.trace_id = "00-new-dispatch"
            return result

        server = ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=_goal,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=server_group,
        )

        executor = MultiThreadedExecutor(num_threads=5)
        executor.add_node(reasoner)
        executor.add_node(server_node)
        _spin_until(executor, lambda: False, 1.5)

        old_call = ExecuteRskillTool(rskill_id=old_id, prompt="old", deadline_s=0.0)
        reasoner._dispatch_execute_rskill(old_call, traceparent=None)
        assert _spin_until(executor, lambda: not reasoner._rskill_inflight, 3.0)

        reasoner.set_parameters([Parameter("dispatch_watchdog_s", value=5.0)])
        new_call = ExecuteRskillTool(rskill_id=new_id, prompt="new", deadline_s=0.0)
        reasoner._dispatch_execute_rskill(new_call, traceparent=None)
        new_generation = reasoner._active_dispatch_generation
        assert new_generation is not None

        assert _spin_until(executor, old_cancelled.is_set, 4.0)
        assert reasoner._active_dispatch_generation == new_generation
        assert reasoner._rskill_inflight
        assert reasoner._renderer.inflight_skill == new_id
        assert reasoner._dispatch_watchdog is not None

        assert _spin_until(
            executor,
            lambda: new_completed.is_set() and not reasoner._rskill_inflight,
            5.0,
        )
        assert reasoner._active_dispatch_generation is None

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.shutdown()
        server.destroy()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()


def test_resident_vla_redispatch_skips_the_free_vram_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-dispatching the warm skill must not be refused for its own VRAM.

    Goal 1 is accepted with plenty of free VRAM; the runner then holds the
    policy warm, so the probe would read low free VRAM. Goal 2 (same skill,
    probe pinned to 0.5 GB vs the manifest's >1 GB declaration) must still
    reach the server — the resident-skill exemption skips the probe. A control
    dispatch of a DIFFERENT skill id under the same pinned reading must be
    refused, proving the probe is skipped by identity, not disabled.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool, RSkillManifest
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_reasoner_ros import reasoner_node as rn_module
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    vla_manifest = RSkillManifest.from_yaml(_SMOLVLA)
    assert (vla_manifest.active_min_vram_gb() or 0.0) > 0.5, "fixture must declare > 0.5 GB"

    executed: list[str] = []
    failures: list[Any] = []
    other_id = "openral/rskill-notreal-other-vla"

    rclpy.init()
    try:
        from openral_core import WaitTool

        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_VLA_ID, other_id})),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        assert reasoner._reward_manifest is None
        reasoner._manifest_for_rskill = lambda _rskill_id: vla_manifest  # type: ignore[method-assign]  # reason: inject fixture manifest at the guard's seam

        server_node = rclpy.create_node("openral_test_vla_server_resident")

        def _execute(goal_handle: Any) -> Any:
            executed.append(goal_handle.request.rskill_id)
            result = ExecuteRskill.Result()
            result.success = True
            result.failure_reason = ""
            result.trace_id = "00-trace-resident"
            goal_handle.succeed()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        sub_node = rclpy.create_node("openral_test_resident_failure_sub")
        fail_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sub_node.create_subscription(
            FailureTrigger, "/openral/failure/rskill", failures.append, fail_qos
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)
        _spin_until(executor, lambda: False, 1.5)  # DDS discovery

        # Goal 1: plenty of free VRAM — accepted, runs, completes.
        monkeypatch.setattr(rn_module, "_detect_gpu_free_vram_gb", lambda: 8.0)
        call = ExecuteRskillTool(rskill_id=_VLA_ID, prompt="pick the teapot", deadline_s=0.0)
        reasoner._dispatch_execute_rskill(call, traceparent=None)
        done1 = _spin_until(
            executor, lambda: len(executed) == 1 and not reasoner._rskill_inflight, 6.0
        )
        assert done1, "goal 1 never completed"
        assert reasoner._resident_vla_id == _VLA_ID

        # Goal 2: the warm policy now "holds" the VRAM — probe pinned LOW.
        # Same skill id → the resident exemption must let it through.
        monkeypatch.setattr(rn_module, "_detect_gpu_free_vram_gb", lambda: 0.5)
        reasoner._dispatch_execute_rskill(call, traceparent=None)
        done2 = _spin_until(
            executor, lambda: len(executed) == 2 and not reasoner._rskill_inflight, 6.0
        )
        assert done2, (
            "re-dispatch of the resident skill never reached the server — "
            f"falsely refused? failures={[m.evidence_json for m in failures]}"
        )
        assert not any("vram_insufficient" in m.evidence_json for m in failures)

        # Control: a DIFFERENT skill under the same 0.5 GB reading must refuse.
        other = ExecuteRskillTool(rskill_id=other_id, prompt="pick the teapot", deadline_s=0.0)
        reasoner._dispatch_execute_rskill(other, traceparent=None)
        refused = _spin_until(
            executor,
            lambda: any("vram_insufficient" in m.evidence_json for m in failures),
            6.0,
        )
        assert refused, "non-resident skill was not refused under low free VRAM"
        assert executed == [_VLA_ID, _VLA_ID], "the control dispatch must not reach the server"

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.remove_node(sub_node)
        server_node.destroy_node()
        sub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
