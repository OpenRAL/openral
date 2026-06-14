"""Live ROS integration test for ADR-0050 VRAM eviction on VLA dispatch.

The reasoner, before dispatching a GPU-heavy ``execute_rskill`` (a VLA policy),
must deactivate its configured GPU lifecycle peers — the object-detector
``LifecycleNode`` is the canonical one — so their VRAM is released *before* the
policy loads, then reactivate them once the skill finishes. Without this, on an
8 GB card the detector (~1.3 GB) co-resident with a VLA (~4.5 GB) OOMs at load
(observed live 2026-06-12: ``rldx_sidecar_died_during_boot`` /
``torch.OutOfMemoryError``).

This exercises the real reasoner node + a real active ``LifecycleNode`` standing
in for the detector + a real ``ExecuteRskill`` ``ActionServer``; the only test
double is ``FakeToolUseClient`` at the LLM process boundary (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of
``tests/integration/test_reasoner_node_end_to_end.py``. Run with::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_vram_eviction.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_rskill_frees_vram_peer_before_dispatch_then_reactivates() -> None:
    """A GPU lifecycle peer is deactivated BEFORE the VLA runs and reactivated after.

    Sequence asserted (by monotonic timestamp, all on one executor):

    1. ``deactivate`` — the peer (a stand-in detector) leaves ACTIVE, releasing
       VRAM, *before* …
    2. ``execute`` — the ``ExecuteRskill`` action server's execute callback runs
       (the policy would load here), and *after the skill result* …
    3. ``activate`` — the peer is reactivated.

    The ordering ``deactivate < execute`` is the crux: it proves the detector's
    VRAM is freed before the policy loads (the fix for the 8 GB OOM). The
    trailing ``activate`` proves the detector is restored for the next perception
    cycle.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    peer_name = "openral_test_vram_peer"
    # One shared, monotonic-timestamped event log across the peer + the action
    # server, so we can assert the deactivate→execute→activate ordering.
    events: list[tuple[str, float]] = []

    class RecordingDetectorPeer(LifecycleNode):
        """Active LifecycleNode stand-in for the GPU object detector."""

        def __init__(self) -> None:
            super().__init__(peer_name)

        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            del state
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
            del state
            events.append(("activate", time.monotonic()))
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
            del state
            events.append(("deactivate", time.monotonic()))
            return TransitionCallbackReturn.SUCCESS

    reactivated = threading.Event()

    rclpy.init()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id="openral/skill-test-vla",
                    prompt="pick the bread",
                    deadline_s=0.0,
                ),
                # Absorb the post-dispatch tick(s) without erroring.
                *[
                    __import__("openral_core", fromlist=["EmitPromptTool"]).EmitPromptTool(
                        target_topic="/openral/prompt", text="standing by"
                    )
                    for _ in range(4)
                ],
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset({"openral/skill-test-vla"})),
            tick_hz=2.0,
        )
        reasoner.set_parameters(
            [Parameter("vram_lifecycle_peers", Parameter.Type.STRING_ARRAY, [peer_name])]
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        # The detector peer starts ACTIVE (the deploy state). Clear the setup
        # activate so `events` captures only the dispatch-driven transitions.
        peer = RecordingDetectorPeer()
        peer.trigger_configure()
        peer.trigger_activate()
        events.clear()

        # Action server: records when the policy would load, then succeeds.
        server_node = rclpy.create_node("openral_test_vla_server")

        def _execute(goal_handle: Any) -> Any:
            events.append(("execute", time.monotonic()))
            result = ExecuteRskill.Result()
            result.success = True
            result.failure_reason = ""
            result.trace_id = "00-trace-vla"
            goal_handle.succeed()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub_node = rclpy.create_node("openral_test_vram_prompt_pub")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(peer)
        executor.add_node(server_node)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_vram_prompt_pub"
        prompt.text = "pick the bread"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        # Spin until the reactivation (second activate) lands, or time out.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            if any(label == "activate" for label, _ in events) and any(
                label == "execute" for label, _ in events
            ):
                # reactivation observed after an execute → done
                reactivated.set()
                break

        # Drain a moment so a just-fired reactivate callback completes.
        drain = time.monotonic() + 1.0
        while time.monotonic() < drain:
            executor.spin_once(timeout_sec=0.05)

        executor.remove_node(reasoner)
        executor.remove_node(peer)
        executor.remove_node(server_node)
        peer.destroy_node()
        server_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    labels = [label for label, _ in sorted(events, key=lambda e: e[1])]
    assert "deactivate" in labels, (
        f"GPU peer was never deactivated before the VLA dispatch; events={labels}. "
        "ADR-0050 VRAM eviction on execute_rskill did not fire."
    )
    assert "execute" in labels, f"action server never ran the goal; events={labels}"
    # Crux: the peer's VRAM is freed BEFORE the policy loads.
    assert labels.index("deactivate") < labels.index("execute"), (
        f"deactivate must precede the policy load (execute); events={labels}"
    )
    # And the detector is restored after the skill result.
    assert "activate" in labels, f"GPU peer was not reactivated after the skill; events={labels}"
    assert labels.index("execute") < labels.index("activate"), (
        f"reactivation must follow the skill result; events={labels}"
    )
