#!/usr/bin/env python3
"""ADR-0018 §5 bullet 4 — deadman_watchdog_node.

Fires ``/openral/estop`` if no ``/openral/safe_action`` message arrives
within :attr:`safe_action_deadline_s` (default 0.2 s — 6 chunks at the
30 Hz baseline). Independent of the C++ safety kernel; runs in its own
process so a kernel crash still triggers a brake event.

Also publishes a ``FailureTrigger`` with
``kind=KIND_TIMEOUT, severity=SEVERITY_ABORT`` on
``/openral/failure/safety`` so the reasoner sees a structured timeout
event (TimeoutEvidence) rather than only the bare estop.
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
            :data:`DEFAULT_SAFE_ACTION_DEADLINE_S`.
        ``check_period_s``: Internal timer period. Default
            :data:`DEFAULT_CHECK_PERIOD_S`.
        ``robot_name``: Tag for FailureTrigger evidence.
    """

    def __init__(self, node_name: str = "openral_deadman_watchdog") -> None:
        """Declare parameters; resources open at on_configure."""
        super().__init__(node_name)
        self.declare_parameter("safe_action_deadline_s", DEFAULT_SAFE_ACTION_DEADLINE_S)
        self.declare_parameter("check_period_s", DEFAULT_CHECK_PERIOD_S)
        self.declare_parameter("robot_name", "robot")

        self._safe_sub: Any = None
        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._estop_sub: Any = None
        self._timer: Any = None

        self._last_safe_ns: int = 0
        self._armed: bool = False
        self._triggered: bool = False

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

        period = self.get_parameter("check_period_s").get_parameter_value().double_value
        self._timer = self.create_timer(period, self._check_deadline)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Arm the deadline check. First chunk must arrive before deadline."""
        del state
        self._last_safe_ns = time.time_ns()
        self._armed = True
        self._triggered = False
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
        # A late chunk after a trigger does NOT auto-clear the latch —
        # the kernel still owns recovery via /openral/estop_reset
        # (CLAUDE.md §10: ROSEStopRequested never auto-cleared).

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
