"""Live ROS integration coverage for the latched ``/openral/safety_status``.

ADR-0096 / hazard-log HZ-0096-1. Two properties that only a real DDS graph
can prove, because both are about **QoS durability** rather than about the
publishing code:

1. **Late-subscriber correctness.** A subscriber that connects *after* a
   latch still receives the current value. This is what TRANSIENT_LOCAL buys
   and what ``/openral/estop`` and ``/openral/failure/safety`` — both
   VOLATILE by design — cannot deliver: a dashboard tab opened mid-mission
   or a runner reconnecting after a crash would otherwise see nothing at all
   until the next transition.
2. **The runner's consumption path.** The real ``RskillRunnerNode`` reads the
   real ``SafetyPassthroughNode``'s status through the existing
   ``safety_abort_getter`` seam (openral#115) and names the actual fault,
   instead of collapsing every abort to ``"/openral/estop"``.

Real components throughout (CLAUDE.md §1.11): the production
``SafetyPassthroughNode`` decides the violation and publishes the status
itself; the production ``RskillRunnerNode`` subscribes and answers. No
mocks, no hand-built status messages standing in for a publisher.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the sibling live tests, and listed
in ``scripts/ros_live_tests.sh``. CI runs it inside ``openral:x86`` (the
``docker-build`` workflow). Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live            # whole suite; `-k safety_status` narrows it
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)


def _status_qos() -> Any:
    """The publishers' QoS: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1.

    A subscriber whose durability does not match never matches the
    publisher at all, so a drifted profile here would make these tests pass
    vacuously (no messages, no assertions reached) rather than fail.
    """
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )


def _spin_until(executor: Any, predicate: Any, *, timeout_s: float = 5.0) -> bool:
    """Spin until ``predicate()`` is true or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return predicate()


@contextmanager
def _active_safety_node(node_name: str) -> Iterator[tuple[Any, Any]]:
    """Bring up a real, activated ``SafetyPassthroughNode``; yield (executor, node)."""
    import rclpy
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.lifecycle import TransitionCallbackReturn

    rclpy.init()
    node = SafetyPassthroughNode(node_name=node_name)
    node.set_parameters(
        [
            rclpy.parameter.Parameter("n_dof", value=6),
            rclpy.parameter.Parameter("estop_reset_cooldown_s", value=0.05),
        ]
    )
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield executor, node
    finally:
        with __import__("contextlib").suppress(Exception):
            node.trigger_deactivate()
            node.trigger_cleanup()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def test_late_subscriber_receives_the_latched_safety_status() -> None:
    """A subscriber created AFTER the latch still gets the current value.

    The whole reason ADR-0096 chose a new latched topic over reusing the
    VOLATILE ``/openral/failure/safety``: current state has to survive the
    consumer not being there when it changed.
    """
    import rclpy
    from openral_msgs.msg import SafetyStatus
    from std_msgs.msg import Empty

    with _active_safety_node("openral_safety_status_late_sub") as (executor, node):
        early = rclpy.create_node("openral_safety_status_late_sub_early")
        executor.add_node(early)
        estop_pub = early.create_publisher(Empty, "/openral/estop", 10)
        executor.spin_once(timeout_sec=0.2)

        # Latch it — and deliberately observe nothing while it happens.
        estop_pub.publish(Empty())
        assert _spin_until(executor, lambda: node._estopped is True), (
            "precondition: the supervisor must be latched before we subscribe"
        )
        # Let the latch transition finish being published before anyone looks.
        _spin_until(executor, lambda: False, timeout_s=0.3)

        # NOW the consumer shows up.
        late = rclpy.create_node("openral_safety_status_late_sub_joiner")
        received: list[Any] = []
        late.create_subscription(
            SafetyStatus, "/openral/safety_status", received.append, _status_qos()
        )
        executor.add_node(late)
        try:
            assert _spin_until(executor, lambda: len(received) >= 1), (
                "TRANSIENT_LOCAL must deliver the current value to a late joiner"
            )
            first = received[0]
            assert first.latched is True, (
                "the FIRST sample a late joiner receives must already say we are latched"
            )
            assert first.drop_reason == SafetyStatus.DROP_EXTERNAL_ESTOP
            assert first.detail, "a latched status must say why"
        finally:
            executor.remove_node(late)
            late.destroy_node()
            executor.remove_node(early)
            early.destroy_node()


def test_latched_status_heartbeat_keeps_the_stamp_fresh() -> None:
    """HZ-0096-1 mitigation 2 — ``header.stamp`` advances while a latch holds.

    A durable value that stopped being re-stamped is indistinguishable from
    a dead publisher's leftover, which is exactly what the runner's
    staleness rule fails closed on. So the publisher must keep proving it is
    alive without changing what it says.
    """
    import rclpy
    from openral_msgs.msg import SafetyStatus
    from std_msgs.msg import Empty

    with _active_safety_node("openral_safety_status_heartbeat") as (executor, _node):
        helper = rclpy.create_node("openral_safety_status_heartbeat_helper")
        received: list[Any] = []
        helper.create_subscription(
            SafetyStatus, "/openral/safety_status", received.append, _status_qos()
        )
        estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)
        executor.add_node(helper)
        try:
            executor.spin_once(timeout_sec=0.2)
            estop_pub.publish(Empty())
            assert _spin_until(executor, lambda: any(m.latched for m in received))
            latched_count = len([m for m in received if m.latched])
            # Two heartbeat periods: at least one refresh must land.
            assert _spin_until(
                executor,
                lambda: len([m for m in received if m.latched]) > latched_count,
                timeout_s=3.0,
            ), "the 1 Hz liveness refresh never re-published the latched value"
            latched_msgs = [m for m in received if m.latched]
            first_ns = (
                latched_msgs[0].header.stamp.sec * 10**9 + latched_msgs[0].header.stamp.nanosec
            )
            last_ns = (
                latched_msgs[-1].header.stamp.sec * 10**9 + latched_msgs[-1].header.stamp.nanosec
            )
            assert last_ns > first_ns, "header.stamp must advance on every refresh"
            # The VALUE must not have drifted — only the evidence of liveness.
            assert latched_msgs[-1].drop_reason == latched_msgs[0].drop_reason
            assert latched_msgs[-1].latched is True
        finally:
            executor.remove_node(helper)
            helper.destroy_node()


def test_runner_safety_abort_getter_names_the_fault_from_safety_status() -> None:
    """The runner's ``safety_abort_getter`` reports the typed fault, live.

    End-to-end through the real seam openral#115 introduced: a real
    ``SafetyPassthroughNode`` latches on a real envelope violation and
    publishes ``SafetyStatus``; the real ``RskillRunnerNode`` subscribes and
    its ``_safety_abort_reason`` — the callable handed to
    ``ROSPublishingHAL`` — names the fault instead of only the topic.
    """
    import rclpy
    from openral_msgs.msg import ActionChunk
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.lifecycle import TransitionCallbackReturn

    rclpy.init()
    runtime = compose_so100_runtime()
    safety = SafetyPassthroughNode(node_name="openral_safety_status_runner_seam")
    safety.set_parameters(
        [
            rclpy.parameter.Parameter("n_dof", value=6),
            rclpy.parameter.Parameter("min_joint", value=[-1.0] * 6),
            rclpy.parameter.Parameter("max_joint", value=[1.0] * 6),
        ]
    )
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(safety)
    executor.add_node(runtime.skill_runner_node)
    helper = rclpy.create_node("openral_safety_status_runner_seam_helper")
    executor.add_node(helper)
    chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
    runner = runtime.skill_runner_node
    try:
        for node in (safety, runner):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        executor.spin_once(timeout_sec=0.3)

        # Baseline: nothing is latched, so the seam reports no abort. The
        # runner has seen a status by now (the supervisor's activation
        # publish), so this also proves the liveness rule does not fire on a
        # freshly-published clear status.
        assert _spin_until(executor, lambda: runner._safety_status is not None), (
            "the runner must receive the supervisor's activation SafetyStatus"
        )
        assert runner._safety_abort_reason() is None

        # A real envelope violation: joint 0 at 4.0 rad, outside ±1.0.
        bad = ActionChunk()
        bad.control_mode = 0  # JOINT_POSITION
        bad.horizon = 1
        bad.n_dof = 6
        bad.flat = [4.0, 0.1, 0.1, 0.1, 0.1, 0.1]
        bad.rskill_id = "openral/test-constant-skill"
        bad.trace_id = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        # Wait for the LATCHED STATUS specifically. The runner's own
        # /openral/estop latch also fires here (defense in depth) and often
        # lands first, so gating on "any abort reason" would let the test pass
        # without the status ever arriving.
        assert _spin_until(
            executor,
            lambda: (
                (chunk_pub.publish(bad) or True)
                and runner._safety_status is not None
                and bool(runner._safety_status.latched)
            ),
        ), "the violation never reached the runner as a latched SafetyStatus"

        reason = runner._safety_abort_reason()
        assert reason is not None
        # The typed fault, named — this is the whole point of ADR-0096's
        # consumer wiring.
        assert "/openral/safety_status:" in reason, reason
        assert "kind_workspace" in reason, reason
        assert "joint[0]" in reason, reason
        # ...and the belt-and-braces /openral/estop latch is still reported,
        # because the supervisor fires that too and dropping it would hide
        # that defense-in-depth is engaged.
        assert "/openral/estop" in reason, reason
    finally:
        with __import__("contextlib").suppress(Exception):
            for node in (runner, safety):
                node.trigger_deactivate()
                node.trigger_cleanup()
        executor.shutdown()
        helper.destroy_node()
        runner.destroy_node()
        runtime.world_state_node.destroy_node()
        safety.destroy_node()
        rclpy.shutdown()
