#!/usr/bin/env python3
"""ADR-0018 §5 bullet 2 — human estop forwarder lifecycle node.

Subscribes to ``/openral/human_estop`` (where UI / Slack / voice
adapters publish), and republishes onto ``/openral/estop`` plus emits a
``FailureTrigger(KIND_HUMAN, HumanEvidence(channel=...))`` on
``/openral/failure/safety``. The C++ safety kernel + HAL both subscribe
to ``/openral/estop`` independently (defense in depth, CLAUDE.md §1.5).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["HumanEstopForwarderNode", "main"]


class HumanEstopForwarderNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Forwards ``/openral/human_estop`` onto ``/openral/estop``.

    Parameters:
        ``channel_label``: Free-text tag on the HumanEvidence (e.g.,
            ``"dashboard"``, ``"slack"``, ``"voice"``). Default
            ``"unknown_human_channel"``.
    """

    def __init__(self, node_name: str = "openral_human_estop_forwarder") -> None:
        """Declare parameters; opens no resources until on_configure."""
        super().__init__(node_name)
        self.declare_parameter("channel_label", "unknown_human_channel")

        self._human_sub: Any = None
        self._estop_pub: Any = None
        self._failure_pub: Any = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open subscription + publishers."""
        del state
        from openral_msgs.msg import FailureTrigger
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import Empty

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
        self._estop_pub = self.create_publisher(Empty, "/openral/estop", estop_qos)
        self._failure_pub = self.create_publisher(
            FailureTrigger, "/openral/failure/safety", failure_qos
        )
        self._human_sub = self.create_subscription(
            Empty, "/openral/human_estop", self._on_human_estop, estop_qos
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """No additional resources to start."""
        del state
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """No additional resources to stop."""
        del state
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release subscription + publishers."""
        del state
        if self._human_sub is not None:
            self.destroy_subscription(self._human_sub)
            self._human_sub = None
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

    def _on_human_estop(self, _msg: object) -> None:
        """Forward to /openral/estop + emit HumanEvidence FailureTrigger."""
        from openral_msgs.msg import FailureTrigger
        from std_msgs.msg import Empty

        assert self._estop_pub is not None and self._failure_pub is not None
        self._estop_pub.publish(Empty())

        channel = (
            self.get_parameter("channel_label").get_parameter_value().string_value
            or "unknown_human_channel"
        )
        trigger = FailureTrigger()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_HUMAN
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        evidence = {"kind": "human", "channel": channel}
        trigger.evidence_json = json.dumps(evidence)
        trigger.rskill_id = ""
        trigger.trace_id = ""
        self._failure_pub.publish(trigger)
        self.get_logger().warning(f"safety.human_estop_forwarded channel={channel!r}")


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_human_estop forwarder_node``."""
    rclpy.init(args=args)
    try:
        node = HumanEstopForwarderNode()
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
