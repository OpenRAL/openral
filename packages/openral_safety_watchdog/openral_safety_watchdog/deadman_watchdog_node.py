#!/usr/bin/env python3
"""ROS 2 reasoner + supervisor graph spec §5 bullet 4 — deadman_watchdog_node.

Fires ``/openral/estop`` if no ``/openral/safe_action`` message arrives
within ``safe_action_deadline_s`` (default 0.2 s — 6 chunks at the
30 Hz baseline). Independent of the C++ safety kernel; runs in its own
process so a kernel crash still triggers a brake event.

Also publishes a ``FailureTrigger`` with
``kind=KIND_TIMEOUT, severity=SEVERITY_ABORT`` on
``/openral/failure/safety`` so the reasoner sees a structured timeout
event (TimeoutEvidence) rather than only the bare estop.

``/openral/safe_action`` is a **bursty** topic, not a continuous stream:
the runner publishes chunks only while an ``ExecuteRskill`` goal is
running, and an idle deploy publishes nothing at all. A watchdog that
armed at activation would therefore fire on every boot, so the node
carries an explicit execution-window gate (``arm_topic``, see
:class:`DeadmanWatchdogNode`). The gate only ever *narrows* when the
watchdog may fire; inside an open window the deadline is unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["DEFAULT_SAFE_ACTION_DEADLINE_S", "DeadmanWatchdogNode", "main"]

DEFAULT_SAFE_ACTION_DEADLINE_S = 0.2
"""Default deadline (seconds) for a /openral/safe_action to arrive."""

DEFAULT_CHECK_PERIOD_S = 0.05
"""How often to check for deadline expiry. 50 ms keeps the watchdog
responsive without burning CPU on a 30 Hz chunk rate."""


class DeadmanWatchdogNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Lifecycle node monitoring ``/openral/safe_action`` for liveness.

    Parameters:
        ``safe_action_deadline_s``: Maximum age (seconds) for the most
            recent ``/openral/safe_action`` before estop fires. Default
            ``DEFAULT_SAFE_ACTION_DEADLINE_S``.
        ``check_period_s``: Internal timer period. Default
            ``DEFAULT_CHECK_PERIOD_S``.
        ``robot_name``: Tag for FailureTrigger evidence.
        ``arm_topic``: ``std_msgs/String`` topic carrying the execution
            window. Empty (the default) means **free-running**: the
            deadline is armed by ``on_activate`` and every silence past
            it fires — the standalone posture, used by the unit tests and
            by any graph whose ``safe_action`` really is continuous.
            Non-empty subscribes that topic and only evaluates the
            deadline while the window is open (a non-empty payload; an
            empty payload closes it) **and** at least one
            ``/openral/safe_action`` has arrived inside the window. That
            is the posture ``deploy_e2e.launch.py`` wires
            (``/openral/reward/active_task``, which the runner publishes
            with the instruction on goal accept and ``""`` on every goal
            exit): a crash mid-goal leaves the window open with the chunk
            stream dead, which fires; an idle deploy and the seconds a
            VLA spends loading before its first chunk do not.
    """

    def __init__(self, node_name: str = "openral_deadman_watchdog") -> None:
        """Declare parameters; resources open at on_configure."""
        super().__init__(node_name)
        self.declare_parameter("safe_action_deadline_s", DEFAULT_SAFE_ACTION_DEADLINE_S)
        self.declare_parameter("check_period_s", DEFAULT_CHECK_PERIOD_S)
        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("arm_topic", "")

        self._safe_sub: Any = None
        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._estop_sub: Any = None
        self._arm_sub: Any = None
        self._timer: Any = None

        self._last_safe_ns: int = 0
        self._armed: bool = False
        self._triggered: bool = False
        # Execution window (``arm_topic``). Free-running config leaves the
        # window permanently open, which is exactly the old behaviour.
        self._window_open: bool = False
        self._chunk_seen_in_window: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

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
        # Defense-in-depth: when the safety kernel publishes /openral/estop
        # itself, the deadman should not double-publish — track external
        # estop so we suppress storm publishing.
        self._estop_sub = self.create_subscription(
            Empty, "/openral/estop", self._on_external_estop, estop_qos
        )

        # Execution-window gate. Only opened when ``arm_topic`` names one;
        # otherwise the node is free-running and the window is always open.
        arm_topic = self.get_parameter("arm_topic").get_parameter_value().string_value
        if arm_topic:
            from std_msgs.msg import String

            window_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._arm_sub = self.create_subscription(
                String, arm_topic, self._on_arm_window, window_qos
            )
            self.get_logger().info(
                f"safety.deadman_window_gated arm_topic={arm_topic!r} "
                "(deadline evaluated only inside an open execution window)"
            )

        period = self.get_parameter("check_period_s").get_parameter_value().double_value
        self._timer = self.create_timer(period, self._check_deadline)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Arm the deadline check. First chunk must arrive before deadline."""
        del state
        self._last_safe_ns = time.time_ns()
        self._armed = True
        self._triggered = False
        # Free-running (no ``arm_topic``): the window is open from activation,
        # which is the pre-gate behaviour. Gated: it opens on the first
        # non-empty window message and not before.
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
        if self._safe_sub is not None:
            self.destroy_subscription(self._safe_sub)
            self._safe_sub = None
        if self._estop_sub is not None:
            self.destroy_subscription(self._estop_sub)
            self._estop_sub = None
        if self._arm_sub is not None:
            self.destroy_subscription(self._arm_sub)
            self._arm_sub = None
        if self._estop_pub is not None:
            self.destroy_publisher(self._estop_pub)
            self._estop_pub = None
        if self._failure_pub is not None:
            self.destroy_publisher(self._failure_pub)
            self._failure_pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Force cleanup."""
        return self.on_cleanup(state)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_safe_action(self, _msg: object) -> None:
        """Reset the deadline timer on every safe_action arrival."""
        self._last_safe_ns = time.time_ns()
        self._chunk_seen_in_window = True
        # A late chunk after a trigger does NOT auto-clear the latch —
        # the kernel still owns recovery via /openral/estop_reset
        # (CLAUDE.md §10: ROSEStopRequested never auto-cleared).

    def _on_arm_window(self, msg: Any) -> None:
        """Open/close the execution window from ``arm_topic``.

        A non-empty payload means "something is executing"; an empty one
        means "nothing is". Opening re-bases the deadline so the window's
        own start is never counted as silence, and closing forgets the
        stream so a fresh window must see its own first chunk.
        """
        open_now = bool(str(msg.data))
        if open_now and not self._window_open:
            self._last_safe_ns = time.time_ns()
        if not open_now:
            self._chunk_seen_in_window = False
        self._window_open = open_now

    def _on_external_estop(self, _msg: object) -> None:
        """Mark triggered when the kernel or another source fires estop.

        Prevents this watchdog from racing the kernel into double publishes.
        """
        if not self._triggered:
            self._triggered = True

    def _check_deadline(self) -> None:
        """Timer callback: fire estop if no safe_action within the deadline."""
        if not self._armed or self._triggered:
            return
        if not self._window_open:
            return
        # Gated mode only: a window that has not yet produced a single chunk
        # is not evidence of a dead actuation path — a VLA can spend tens of
        # seconds loading between goal accept and its first chunk.
        if self._arm_sub is not None and not self._chunk_seen_in_window:
            return
        deadline_s: float = (
            self.get_parameter("safe_action_deadline_s").get_parameter_value().double_value
        )
        age_s = (time.time_ns() - self._last_safe_ns) / 1e9
        if age_s <= deadline_s:
            return
        self._fire_estop(age_s, deadline_s)

    def _fire_estop(self, age_s: float, deadline_s: float) -> None:
        """Publish estop + FailureTrigger; latch internal triggered flag."""
        from openral_msgs.msg import FailureTrigger
        from std_msgs.msg import Empty

        self._triggered = True
        assert self._estop_pub is not None and self._failure_pub is not None
        self._estop_pub.publish(Empty())

        trigger = FailureTrigger()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_TIMEOUT
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        # Build TimeoutEvidence JSON inline — matches
        # ``openral_core.TimeoutEvidence`` (kind="timeout", operation,
        # deadline_s, elapsed_s).
        evidence = {
            "kind": "timeout",
            "operation": "safe_action",
            "deadline_s": float(deadline_s),
            "elapsed_s": float(age_s),
        }
        trigger.evidence_json = json.dumps(evidence)
        trigger.rskill_id = ""  # unknown at this layer
        trigger.trace_id = ""
        self._failure_pub.publish(trigger)
        self.get_logger().error(
            f"safety.deadman_fired age_s={age_s:.3f} deadline_s={deadline_s:.3f}"
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
