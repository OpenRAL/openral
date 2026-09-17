#!/usr/bin/env python3
"""ROS 2 reasoner + supervisor graph spec §5 bullet 4 — deadman_watchdog_node.

Fires ``/openral/estop`` (plus a structured ``FailureTrigger``) when the
``/openral/safe_action`` stream dies while a skill goal is executing. Runs in
its own process, so a crash in the in-band ``openral_safety`` node or the C++
kernel still produces a brake event.

``/openral/safe_action`` is bursty — the runner publishes chunks only while an
``ExecuteRskill`` goal runs — so a free-running deadline would fire on every
idle boot. The node therefore gates on an **execution window** taken from the
runner's own action-status topic (``arm_status_topic``). Three properties make
that the right source, and an advisory "current task" string the wrong one:

* A dead runner cannot close the window. It closes only on a terminal goal
  status, and a process that is gone publishes nothing — so silence inside an
  open window stays evidence of a dead actuation path.
* One writer. The action server owns its own status topic, so no second node
  can close the window on the runner's behalf.
* ``TRANSIENT_LOCAL``, so a watchdog that joins mid-goal inherits the current
  status instead of waiting for the next transition.

Every guard that can *suppress* a fire is bounded: a window that never produces
a first chunk fires at ``first_chunk_deadline_s``, and the post-estop latch is
released when ``/openral/safety_status`` reports recovery.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = [
    "DEFAULT_CHECK_PERIOD_S",
    "DEFAULT_FIRST_CHUNK_DEADLINE_S",
    "DEFAULT_SAFETY_STATUS_TOPIC",
    "DEFAULT_SAFE_ACTION_DEADLINE_S",
    "DeadmanWatchdogNode",
    "main",
]

DEFAULT_SAFE_ACTION_DEADLINE_S = 0.2
"""Default deadline (seconds) for a /openral/safe_action to arrive."""

DEFAULT_CHECK_PERIOD_S = 0.05
"""Deadline-check timer period. 50 ms is responsive at a 30 Hz chunk rate."""

DEFAULT_FIRST_CHUNK_DEADLINE_S = 60.0
"""Bound on goal-accepted -> first chunk. Covers a cold VLA load; without a
bound, a runner that dies between accepting a goal and emitting its first
chunk holds the window open forever and is never braked."""

DEFAULT_SAFETY_STATUS_TOPIC = "/openral/safety_status"
"""Latched SafetyStatus (ADR-0096) — the recovery signal that releases the
latch this node sets when it observes an estop."""

# Names of the action_msgs/GoalStatus constants meaning "this goal is still
# live". Resolved to values off the generated message class at configure time
# (_live_goal_statuses) rather than hardcoded, so a renumbering upstream cannot
# silently widen or narrow the gate on a safety path.
_LIVE_GOAL_STATUS_NAMES = ("STATUS_ACCEPTED", "STATUS_EXECUTING", "STATUS_CANCELING")


def _live_goal_statuses(status_cls: Any) -> frozenset[int]:
    """The GoalStatus values that mean a goal is still running.

    Args:
        status_cls: The generated ``action_msgs.msg.GoalStatus`` class.

    Returns:
        The numeric values of ACCEPTED / EXECUTING / CANCELING.

    Example:
        >>> class _S:
        ...     STATUS_ACCEPTED = 1
        ...     STATUS_EXECUTING = 2
        ...     STATUS_CANCELING = 3
        >>> sorted(_live_goal_statuses(_S))
        [1, 2, 3]
    """
    return frozenset(getattr(status_cls, name) for name in _LIVE_GOAL_STATUS_NAMES)


class DeadmanWatchdogNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Lifecycle node monitoring ``/openral/safe_action`` for liveness.

    Parameters:
        ``safe_action_deadline_s``: Maximum age (seconds) of the newest
            ``/openral/safe_action`` inside an open window before estop fires.
        ``check_period_s``: Internal timer period.
        ``robot_name``: Tag for FailureTrigger evidence.
        ``arm_status_topic``: ``action_msgs/GoalStatusArray`` topic carrying the
            runner's goal status. Empty (the default) is **free-running**:
            armed by ``on_activate``, any silence past the deadline fires.
            Non-empty evaluates the deadline only while a goal is live.
        ``first_chunk_deadline_s``: Maximum time an open window may go without
            producing one ``/openral/safe_action`` before estop fires. Gated
            mode only.
        ``safety_status_topic``: Latched ``openral_msgs/SafetyStatus`` whose
            ``latched=False`` releases this node's post-estop latch. Empty
            disables re-arming, leaving the node one-shot per activation.

    Example:
        >>> node = DeadmanWatchdogNode(node_name="deadman_doctest")
        >>> node.get_parameter("arm_status_topic").get_parameter_value().string_value
        ''
        >>> node.destroy_node()
    """

    def __init__(self, node_name: str = "openral_deadman_watchdog") -> None:
        """Declare parameters; resources open at on_configure."""
        super().__init__(node_name)
        self.declare_parameter("safe_action_deadline_s", DEFAULT_SAFE_ACTION_DEADLINE_S)
        self.declare_parameter("check_period_s", DEFAULT_CHECK_PERIOD_S)
        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("arm_status_topic", "")
        self.declare_parameter("first_chunk_deadline_s", DEFAULT_FIRST_CHUNK_DEADLINE_S)
        self.declare_parameter("safety_status_topic", DEFAULT_SAFETY_STATUS_TOPIC)

        self._safe_sub: Any = None
        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._estop_sub: Any = None
        self._arm_sub: Any = None
        self._status_sub: Any = None
        self._timer: Any = None

        self._last_safe_ns: int = 0
        self._window_opened_ns: int = 0
        self._armed: bool = False
        self._triggered: bool = False
        self._window_open: bool = False
        self._chunk_seen_in_window: bool = False
        self._live_statuses: frozenset[int] = frozenset()

    # -- Lifecycle -----------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open subscriptions / publishers."""
        del state
        from openral_msgs.msg import ActionChunk, FailureTrigger
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import Empty

        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        failure_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )

        self._safe_sub = self.create_subscription(
            ActionChunk, "/openral/safe_action", self._on_safe_action, chunk_qos
        )
        self._estop_pub = self.create_publisher(Empty, "/openral/estop", estop_qos)
        self._failure_pub = self.create_publisher(
            FailureTrigger, "/openral/failure/safety", failure_qos
        )
        # Latch behind an estop from any source so this node never storms the
        # topic behind the kernel. Released by _on_safety_status.
        self._estop_sub = self.create_subscription(
            Empty, "/openral/estop", self._on_external_estop, estop_qos
        )

        arm_status_topic = self.get_parameter("arm_status_topic").get_parameter_value().string_value
        if arm_status_topic:
            from action_msgs.msg import GoalStatus, GoalStatusArray
            from rclpy.qos import qos_profile_action_status_default

            self._live_statuses = _live_goal_statuses(GoalStatus)

            # The action server's own QoS (RELIABLE, TRANSIENT_LOCAL, depth 1).
            # Matching it is what lets a late-joining watchdog inherit the
            # current goal status instead of waiting for the next transition.
            self._arm_sub = self.create_subscription(
                GoalStatusArray,
                arm_status_topic,
                self._on_arm_status,
                qos_profile_action_status_default,
            )
            self.get_logger().info(
                f"safety.deadman_window_gated arm_status_topic={arm_status_topic!r}"
            )

        status_topic = self.get_parameter("safety_status_topic").get_parameter_value().string_value
        if status_topic:
            from openral_msgs.msg import SafetyStatus

            status_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            self._status_sub = self.create_subscription(
                SafetyStatus, status_topic, self._on_safety_status, status_qos
            )

        period = self.get_parameter("check_period_s").get_parameter_value().double_value
        self._timer = self.create_timer(period, self._check_deadline)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Arm the deadline check."""
        del state
        now = time.time_ns()
        self._last_safe_ns = now
        self._window_opened_ns = now
        self._armed = True
        self._triggered = False
        # Free-running: open from activation. Gated: opens on the first live
        # goal status and not before.
        self._window_open = self._arm_sub is None
        self._chunk_seen_in_window = False
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Disarm — deadline checks become no-ops until next activate."""
        del state
        self._armed = False
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release all resources."""
        del state
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for attr in ("_safe_sub", "_estop_sub", "_arm_sub", "_status_sub"):
            sub = getattr(self, attr)
            if sub is not None:
                self.destroy_subscription(sub)
                setattr(self, attr, None)
        for attr in ("_estop_pub", "_failure_pub"):
            pub = getattr(self, attr)
            if pub is not None:
                self.destroy_publisher(pub)
                setattr(self, attr, None)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Force cleanup."""
        return self.on_cleanup(state)

    # -- Callbacks -----------------------------------------------------------

    def _on_safe_action(self, _msg: object) -> None:
        """Reset the deadline timer on every safe_action arrival."""
        self._last_safe_ns = time.time_ns()
        self._chunk_seen_in_window = True

    def _on_arm_status(self, msg: Any) -> None:
        """Open/close the execution window from the runner's goal status.

        ACCEPTED / EXECUTING / CANCELING mean the actuation path is supposed to
        be producing chunks; every other status is terminal.
        """
        live = any(entry.status in self._live_statuses for entry in msg.status_list)
        if live and not self._window_open:
            now = time.time_ns()
            self._last_safe_ns = now
            self._window_opened_ns = now
            self._chunk_seen_in_window = False
        elif not live:
            self._chunk_seen_in_window = False
        self._window_open = live

    def _on_external_estop(self, _msg: object) -> None:
        """Latch behind an estop from any source, to avoid double publishes."""
        self._triggered = True

    def _on_safety_status(self, msg: Any) -> None:
        """Release the latch once the safety layer reports recovery.

        Without this the node is dead after the first estop of a deploy: the
        operator resets the kernel, the graph resumes, and the only independent
        watchdog stays silent for the life of the process. Re-arming re-bases
        both deadlines, so a reset is never immediately followed by a fire on
        the silence that accumulated while stopped.
        """
        if msg.latched or not self._triggered:
            return
        now = time.time_ns()
        self._triggered = False
        self._last_safe_ns = now
        self._window_opened_ns = now
        self.get_logger().info("safety.deadman_rearmed after /openral/safety_status recovery")

    def _check_deadline(self) -> None:
        """Timer callback: fire estop if the chunk stream is overdue."""
        if not self._armed or self._triggered or not self._window_open:
            return
        now = time.time_ns()
        if self._arm_sub is not None and not self._chunk_seen_in_window:
            # Bounded: a goal that never produces a first chunk is a dead
            # actuation path once its load budget is spent.
            budget_s: float = (
                self.get_parameter("first_chunk_deadline_s").get_parameter_value().double_value
            )
            age_s = (now - self._window_opened_ns) / 1e9
            if age_s > budget_s:
                self._fire_estop(age_s, budget_s, "first_chunk")
            return
        deadline_s: float = (
            self.get_parameter("safe_action_deadline_s").get_parameter_value().double_value
        )
        age_s = (now - self._last_safe_ns) / 1e9
        if age_s > deadline_s:
            self._fire_estop(age_s, deadline_s, "safe_action")

    def _fire_estop(self, age_s: float, deadline_s: float, operation: str) -> None:
        """Publish estop + FailureTrigger; latch until safety_status recovers."""
        from openral_msgs.msg import FailureTrigger
        from std_msgs.msg import Empty

        self._triggered = True
        assert self._estop_pub is not None and self._failure_pub is not None
        self._estop_pub.publish(Empty())

        trigger = FailureTrigger()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_TIMEOUT
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        # Matches openral_core.TimeoutEvidence.
        trigger.evidence_json = json.dumps(
            {
                "kind": "timeout",
                "operation": operation,
                "deadline_s": float(deadline_s),
                "elapsed_s": float(age_s),
            }
        )
        trigger.rskill_id = ""
        trigger.trace_id = ""
        self._failure_pub.publish(trigger)
        self.get_logger().error(
            f"safety.deadman_fired operation={operation} "
            f"age_s={age_s:.3f} deadline_s={deadline_s:.3f}"
        )


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_safety_watchdog deadman_watchdog_node``."""
    rclpy.init(args=args)
    try:
        node = DeadmanWatchdogNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if already shut down
    return 0


if __name__ == "__main__":
    sys.exit(main())
