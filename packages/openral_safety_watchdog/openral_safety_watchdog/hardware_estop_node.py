#!/usr/bin/env python3
"""ROS 2 reasoner + supervisor graph spec §5 bullet 3 — hardware_estop_node.

Bridges a hardware estop source (GPIO relay via libgpiod, or a USB HID
pendant via /dev/input) onto ``/openral/estop``. Polls the device at
``poll_rate_hz`` (default 100 Hz) and publishes
``std_msgs/Empty`` + ``FailureTrigger(KIND_HUMAN, SEVERITY_ABORT,
HumanEvidence(channel="hardware_pendant"))`` on the rising edge.

The actual device driver is opaque to this node — the
``HardwareEstopNode._read_pressed`` hook is overridden by per-vendor
subclasses (or by tests that drive a real device state source). The base
class implements the polling loop, edge detection, and ROS publication.

**This repository ships no pendant driver.** The base ``_read_pressed``
has nothing to read, so a bare ``hardware_estop_node`` is not a hardware
E-stop source and must never be mistaken for one. ``on_configure``
therefore refuses to leave UNCONFIGURED unless a read source is actually
present (a vendor subclass, or an injected ``read_pressed_hook``) *and*
the declared ``device`` path exists — see
:meth:`HardwareEstopNode._resolve_read_source`. A node that reports
not-ready is the honest outcome on a host with no pendant; a node sitting
ACTIVE and polling a stub that always answers "not pressed" is not.

A read that *raises* is also not a "not pressed" answer. An unplugged pendant
or a driver fault means the state is unknown, and for a brake source unknown
fails closed: :meth:`HardwareEstopNode._poll` catches the error, fires the
estop once, and latches so a persistently broken driver cannot storm the topic.
"""

from __future__ import annotations

import json
import os
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

    Subclasses override ``_read_pressed`` to talk to a real device.
    The base class reads from an injected ``read_pressed_hook`` state
    callable; this keeps the polling / publication logic exercised by
    tests driving a real state source without requiring a pendant on CI
    runners.

    Parameters:
        ``poll_rate_hz``: Device poll rate. Default
            ``DEFAULT_POLL_RATE_HZ``.
        ``device``: Path of the GPIO chip / HID node this bridge owns
            (e.g. ``/dev/gpiochip0``, ``/dev/input/by-id/...``). Empty —
            the default — declares **no hardware E-stop on this host**,
            and configure fails so the node reports not-ready rather than
            posing as a source that can never fire.
        ``active_low``: Electrical polarity of the pendant contact, for
            vendor subclasses. Unused by the base class.
        ``channel_label``: Tag on the emitted ``HumanEvidence``.
    """

    def __init__(self, node_name: str = "openral_hardware_estop") -> None:
        """Declare parameters; opens no resources until on_configure."""
        super().__init__(node_name)
        self.declare_parameter("poll_rate_hz", DEFAULT_POLL_RATE_HZ)
        self.declare_parameter("device", "")  # "" → no pendant declared (see _resolve_read_source)
        self.declare_parameter("active_low", True)
        self.declare_parameter("channel_label", "hardware_pendant")

        self._estop_pub: Any = None
        self._failure_pub: Any = None
        self._timer: Any = None
        self._last_pressed: bool = False
        # Set when a read raises; keeps a broken driver from storming.
        self._read_failed: bool = False

        # Injection hook: tests assign a Callable[[], bool] here before
        # configure/activate. Production subclasses override
        # ``_read_pressed`` instead.
        self.read_pressed_hook: Any = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open publishers and start polling timer, if a device is really there.

        Returns ``FAILURE`` — leaving the node UNCONFIGURED, i.e. visibly
        not-ready on the graph — whenever this host has no hardware E-stop
        this node could read. The reason is logged at ERROR every time, so
        a deploy without a pendant says so instead of implying one.
        """
        del state
        unavailable = self._resolve_read_source()
        if unavailable is not None:
            self.get_logger().error(
                f"safety.hardware_estop_unavailable reason={unavailable!r} — "
                "this graph has NO hardware E-stop source; the node stays "
                "unconfigured. Other E-stop sources (deadman watchdog, human "
                "forwarder, safety kernel) are unaffected."
            )
            return TransitionCallbackReturn.FAILURE
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
        """Clear the read-failure latch for this activation."""
        del state
        self._read_failed = False
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

    def _resolve_read_source(self) -> str | None:
        """``None`` when a real read source exists; else why it does not.

        Two independent conditions, both required:

        1. Something can actually answer "is it pressed" — an injected
           ``read_pressed_hook`` or a subclass that overrides
           ``_read_pressed``. The base implementation is not a driver; it
           answers ``False`` forever, which would read as "pendant fine".
        2. The ``device`` the operator declared exists on this host. An
           empty ``device`` declares no pendant at all.
        """
        device = self.get_parameter("device").get_parameter_value().string_value
        has_driver = (
            self.read_pressed_hook is not None
            or type(self)._read_pressed is not HardwareEstopNode._read_pressed
        )
        if not device:
            return "no 'device' parameter declared"
        if not os.path.exists(device):
            return f"declared device {device!r} is not present on this host"
        if not has_driver:
            return (
                f"device {device!r} exists but this build ships no pendant driver "
                "(HardwareEstopNode is a base class; a vendor subclass must "
                "override _read_pressed)"
            )
        return None

    def _read_pressed(self) -> bool:
        """Return whether the hardware pendant is currently pressed.

        Default implementation uses ``read_pressed_hook`` if set.
        Subclasses override to read a real GPIO pin or HID device.
        ``on_configure`` refuses to bring the node up when neither exists,
        so the ``False`` below is unreachable from an ACTIVE node — it is
        not a "pendant is fine" answer, and must never become one.
        """
        if self.read_pressed_hook is not None:
            return bool(self.read_pressed_hook())
        return False

    # ── Polling ──────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Poll the device; on rising edge publish estop + FailureTrigger.

        A read that raises is treated as "pendant state unknown", which for a
        brake source means unsafe: fire once, then latch. Letting the exception
        escape would kill the timer and take this E-stop source off the graph
        with no estop and no further diagnostic.
        """
        if self._read_failed:
            return
        try:
            pressed = self._read_pressed()
        except Exception as exc:  # reason: any driver fault -> unknown -> fail closed
            self._read_failed = True
            self.get_logger().error(
                f"safety.hardware_estop_read_failed error={exc!r} — pendant state is "
                "unknown, braking and latching; reactivate the node to retry"
            )
            self._fire_estop(reason="read_failed")
            return
        # Rising-edge: only publish on the first press, not while held.
        if pressed and not self._last_pressed:
            self._fire_estop()
        self._last_pressed = pressed

    def _fire_estop(self, reason: str = "pressed") -> None:
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
        # openral_core.HumanEvidence — actor is REQUIRED and the model forbids
        # extra keys, so the channel belongs in `actor`, not a `channel` key.
        evidence = {"kind": "human", "actor": channel, "reason": reason}
        trigger.evidence_json = json.dumps(evidence)
        trigger.rskill_id = ""
        trigger.trace_id = ""
        self._failure_pub.publish(trigger)
        self.get_logger().warning(
            f"safety.hardware_estop_fired channel={channel!r} reason={reason!r}"
        )


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
