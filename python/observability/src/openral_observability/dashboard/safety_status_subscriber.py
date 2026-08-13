"""Live ROS 2 subscriber for the latched `/openral/safety_status` (ADR-0096).

Everything else on the dashboard's Safety surface is inferred from OTel
`safety.check` spans: a `violation` sets the e-stop latch flag, an `info` pass
clears it. That inference only works *while chunks are flowing*. A latched
kernel stops publishing safe actions, so the chunk stream dries up and the
dashboard is left showing whatever it last inferred — and an operator who
opens the page after the stop sees nothing at all, because spans are a stream,
not a state.

ADR-0096 gives the graph a typed, latched answer to "what is safety doing
right now": `openral_msgs/SafetyStatus` on `/openral/safety_status`, published
RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1 by both the C++ safety kernel and
`SafetyPassthroughNode`. TRANSIENT_LOCAL is the load-bearing part here: a
dashboard opened mid-mission receives the current value on connect, without
having witnessed the transition.

This is the dashboard's first rclpy *subscriber* (`estop_publisher.py` is its
first publisher) and it follows that module's shape exactly: one node, created
at launch, spun on a daemon thread, and inert-but-harmless when rclpy / the
`openral_msgs` overlay is unavailable — a standalone dashboard with no ROS
workspace sourced keeps working and simply shows the Safety Status card as
"waiting".

Read-only by construction: it subscribes and writes into the store. It holds
no publisher, no service client, and no authority over the robot.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_observability.dashboard.store import TelemetryStore

__all__ = ["SAFETY_STATUS_TOPIC", "SafetyStatusSubscriber", "drop_reason_label"]

SAFETY_STATUS_TOPIC = "/openral/safety_status"


def drop_reason_label(code: int, status_cls: Any) -> str:  # reason: ROS message class is untyped
    """Name a ``SafetyStatus.drop_reason`` from the IDL's own constants.

    Reverse-maps the value against the generated message class's ``KIND_*`` /
    ``DROP_*`` constants, so a constant added to ``SafetyStatus.msg`` shows up
    here with no second table to keep in sync — and an unknown value degrades
    to a readable placeholder instead of a wrong label.

    Args:
        code: The ``drop_reason`` value received on the wire.
        status_cls: The generated ``openral_msgs.msg.SafetyStatus`` class.

    Returns:
        The constant's name lower-cased (e.g. ``"kind_collision"``), or
        ``"drop_reason_<code>"`` when this build does not know the value.

    Example:
        >>> class _S:
        ...     KIND_COLLISION = 10
        ...     DROP_NONE = 255
        >>> drop_reason_label(10, _S)
        'kind_collision'
        >>> drop_reason_label(200, _S)
        'drop_reason_200'
    """
    for name in dir(status_cls):
        if not name.startswith(("KIND_", "DROP_")):
            continue
        if getattr(status_cls, name, None) == code:
            return name.lower()
    return f"drop_reason_{code}"


class SafetyStatusSubscriber:
    """Feeds the store from the latched `/openral/safety_status` topic."""

    def __init__(self, store: TelemetryStore) -> None:
        """Create the node + subscription now (at launch) so DDS discovers early.

        Args:
            store: The dashboard's telemetry store; every received status is
                written to it via :meth:`TelemetryStore.set_safety_status`.
        """
        self._store = store
        self._node: Any = None
        self._subscription: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_rclpy = False
        try:
            self._start()
        except Exception:  # reason: a ROS hiccup must never break the dashboard
            self.close()

    def _start(self) -> None:
        import rclpy
        from openral_msgs.msg import SafetyStatus
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
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, name="openral_dashboard_safety_status_spin", daemon=True
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
