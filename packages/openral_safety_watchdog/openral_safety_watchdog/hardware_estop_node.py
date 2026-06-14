#!/usr/bin/env python3
"""ADR-0018 §5 bullet 3 — hardware_estop_node.

Bridges a hardware estop source (GPIO relay via libgpiod, or a USB HID
pendant via /dev/input) onto ``/openral/estop``. Polls the device at
:attr:`poll_rate_hz` (default 100 Hz) and publishes
``std_msgs/Empty`` + ``FailureTrigger(KIND_HUMAN, SEVERITY_ABORT,
HumanEvidence(channel="hardware_pendant"))`` on the rising edge.

The actual device driver is opaque to this node — the
:meth:`HardwareEstopNode._read_pressed` hook is overridden by per-vendor
subclasses (or by tests that simulate the device). The base class
implements the polling loop, edge detection, and ROS publication.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["DEFAULT_POLL_RATE_HZ", "HardwareEstopNode", "main"]

DEFAULT_POLL_RATE_HZ = 100.0
"""How often (Hz) to query the hardware estop state."""


class HardwareEstopNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """Lifecycle node bridging a hardware pendant onto /openral/estop.

    Subclasses override :meth:`_read_pressed` to talk to a real device.
    The base class — used in tests — reads from an injected state
    callable; this keeps the polling / publication logic exercised by
    unit tests without requiring real hardware on CI runners.
    """

    def __init__(self, node_name: str = "openral_hardware_estop") -> None:
        """Declare parameters; opens no resources until on_configure."""
        super().__init__(node_name)
        self.declare_parameter("poll_rate_hz", DEFAULT_POLL_RATE_HZ)
        self.declare_parameter("device", "")  # "" → injection mode (tests)
        self.declare_parameter("active_low", True)
        self.declare_parameter("channel_label", "hardware_pendant")

        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._timer: Any = None
        self._last_pressed: bool = False

        # Injection hook: tests assign a Callable[[], bool] here before
        # configure/activate. Production subclasses override
        # ``_read_pressed`` instead.
        self.read_pressed_hook: Any = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open publishers and start polling timer."""
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

        rate = self.get_parameter("poll_rate_hz").get_parameter_value().double_value
        period = 1.0 / max(rate, 1.0)
        self._timer = self.create_timer(period, self._poll)
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
        """Release timer + publishers."""
        del state
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
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

    # ── Device read hook ─────────────────────────────────────────────────────

    def _read_pressed(self) -> bool:
        """Return whether the hardware pendant is currently pressed.

        Default implementation uses :attr:`read_pressed_hook` if set,
        otherwise returns ``False`` (no estop). Subclasses override to
        read a real GPIO pin or HID device.
        """
        if self.read_pressed_hook is not None:
            return bool(self.read_pressed_hook())
        return False

    # ── Polling ──────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Poll the device; on rising edge publish estop + FailureTrigger."""
        pressed = self._read_pressed()
        # Rising-edge: only publish on the first press, not while held.
        if pressed and not self._last_pressed:
            self._fire_estop()
        self._last_pressed = pressed

    def _fire_estop(self) -> None:
        """Publish std_msgs/Empty + FailureTrigger(KIND_HUMAN)."""
        from openral_msgs.msg import FailureTrigger
        from std_msgs.msg import Empty

        assert self._estop_pub is not None and self._failure_pub is not None
        self._estop_pub.publish(Empty())

        channel = (
            self.get_parameter("channel_label").get_parameter_value().string_value
            or "hardware_pendant"
        )
        trigger = FailureTrigger()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_HUMAN
        trigger.severity = FailureTrigger.SEVERITY_ABORT
        # HumanEvidence layout from openral_core.HumanEvidence.
        evidence = {"kind": "human", "channel": channel}
        trigger.evidence_json = json.dumps(evidence)
        trigger.rskill_id = ""
        trigger.trace_id = ""
        self._failure_pub.publish(trigger)
        self.get_logger().warning(f"safety.hardware_estop_fired channel={channel!r}")


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_safety_watchdog hardware_estop_node``."""
    rclpy.init(args=args)
    try:
        node = HardwareEstopNode()
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
