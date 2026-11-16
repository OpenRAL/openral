"""Live ROS 2 subscriber for the latched `/openral/safety_status` (ADR-0096).

Everything else on the Safety surface is inferred from OTel `safety.check`
spans (`violation` sets the e-stop latch, `info` clears it), which only works
while chunks are flowing — a latched kernel stops publishing safe actions, so
the stream dries up and a dashboard opened mid-mission sees nothing.

ADR-0096 adds a typed, latched answer: `openral_msgs/SafetyStatus` on
`/openral/safety_status`, published RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1
by both the C++ safety kernel and `SafetyPassthroughNode`. TRANSIENT_LOCAL is
load-bearing: a dashboard opened mid-mission receives the current value on
connect without having witnessed the transition.

The dashboard's first rclpy subscriber (`estop_publisher.py` is its first
publisher); same shape — one node, created at launch, spun on a daemon
thread, inert when rclpy / `openral_msgs` is unavailable. Read-only: no
publisher, no service client, no authority over the robot.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from openral_observability.failure_bus import drop_reason_label
from openral_observability.rclpy_spin import spin_executor_until_shutdown

if TYPE_CHECKING:
    from openral_observability.dashboard.store import TelemetryStore

__all__ = [
    "DIAGNOSTICS_TOPIC",
    "SAFETY_FAILURE_TOPIC",
    "SAFETY_STATUS_TOPIC",
    "SafetyStatusSubscriber",
    "drop_reason_label",
]

SAFETY_STATUS_TOPIC = "/openral/safety_status"
# The safety kernel's typed stops (``FailureTrigger``); the console's trace shows each.
SAFETY_FAILURE_TOPIC = "/openral/failure/safety"
# The HAL lifecycle node's 1 Hz heartbeat carries its own e-stop latch (``estopped``):
# a kernel release does not release a HAL that needs a restart, or whose re-arm failed.
DIAGNOSTICS_TOPIC = "/diagnostics"


class SafetyStatusSubscriber:
    """Feeds the store from the latched `/openral/safety_status` topic."""

    def __init__(self, store: TelemetryStore) -> None:
        """Create the node + subscription now (at launch) so DDS discovers early.

        Args:
            store: The dashboard's telemetry store; every received status is
                written to it via ``TelemetryStore.set_safety_status``.
        """
        self._store = store
        self._node: Any = None
        self._subscription: Any = None
        self._diagnostics: Any = None
        self._failures: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_rclpy = False
        try:
            self._start()
        except Exception:  # reason: a ROS hiccup must never break the dashboard
            self.close()

    def _start(self) -> None:
        import rclpy
        from openral_msgs.msg import FailureTrigger, SafetyStatus
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = Node("openral_dashboard_safety_status")
        # Must MATCH the publishers' profile. A VOLATILE subscriber would
        # never match a TRANSIENT_LOCAL publisher at all — the card would sit
        # empty forever with no error anywhere to explain it.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._subscription = self._node.create_subscription(
            SafetyStatus, SAFETY_STATUS_TOPIC, self._on_status, qos
        )
        # The kernel's publisher profile (RELIABLE, VOLATILE); depth absorbs a burst.
        self._failures = self._node.create_subscription(
            FailureTrigger,
            SAFETY_FAILURE_TOPIC,
            self._on_failure,
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=50),
        )
        try:
            from diagnostic_msgs.msg import DiagnosticArray

            # The heartbeat's own profile (depth 10, RELIABLE, VOLATILE).
            self._diagnostics = self._node.create_subscription(
                DiagnosticArray, DIAGNOSTICS_TOPIC, self._on_diagnostics, 10
            )
        except ImportError:  # reason: no diagnostic_msgs: the kernel's latch alone, as before
            self._diagnostics = None
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=spin_executor_until_shutdown,
            args=(self._executor,),
            name="openral_dashboard_safety_status_spin",
            daemon=True,
        )
        self._thread.start()

    @property
    def available(self) -> bool:
        """True when the subscription is live (rclpy + `openral_msgs` present)."""
        return self._subscription is not None

    def _on_status(self, msg: Any) -> None:  # reason: ROS message is untyped
        """Write one received status into the store (runs on the spin thread)."""
        stamp = msg.header.stamp
        self._store.set_safety_status(
            latched=bool(msg.latched),
            drop_reason=int(msg.drop_reason),
            drop_reason_label=drop_reason_label(int(msg.drop_reason), type(msg)),
            detail=str(msg.detail),
            rskill_id=str(msg.rskill_id),
            trace_id=str(msg.trace_id),
            stamp_unix=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
        )

    def _on_failure(self, msg: Any) -> None:  # reason: ROS message is untyped
        """Write one kernel ``FailureTrigger`` into the store (runs on the spin thread)."""
        stamp = msg.header.stamp
        cls = type(msg)
        self._store.record_safety_failure(
            kind_label=drop_reason_label(int(msg.kind), cls, ("KIND_",)),
            severity_label=drop_reason_label(int(msg.severity), cls, ("SEVERITY_",)),
            evidence_json=str(msg.evidence_json),
            rskill_id=str(msg.rskill_id),
            trace_id=str(msg.trace_id),
            stamp_unix=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
        )

    def _on_diagnostics(self, msg: Any) -> None:  # reason: ROS message is untyped
        """Record each HAL heartbeat's ``estopped`` (runs on the spin thread)."""
        for status in msg.status:
            values = {kv.key: kv.value for kv in status.values}
            if "estopped" in values:
                self._store.set_hal_estop(str(status.name), values["estopped"] == "true")

    def close(self) -> None:
        """Tear down the node + executor; shut down rclpy only if we started it."""
        import contextlib

        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown()
        if self._node is not None:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
        if self._owns_rclpy:
            with contextlib.suppress(Exception):
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
        self._node = None
        self._subscription = None
        self._executor = None
