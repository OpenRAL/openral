"""Persistent ROS 2 e-stop publisher for the dashboard (safety-critical).

The dashboard has no rclpy node of its own, so its e-stop originally shelled out
``ros2 topic pub`` per press. That spawns a FRESH publisher every time which must
rediscover the subscribers before it can deliver — adding ~1 s and, worse,
racing discovery so the message could be published before any subscriber is
matched and be silently LOST. An e-stop that is sometimes lost, and never
instant, is not an e-stop (observed live: the robot kept moving).

This holds ONE publisher, created at dashboard startup, so DDS discovery of the
HAL / kernel / runner subscribers happens ONCE at launch. Every later press then
publishes instantly on the already-matched publisher (sub-millisecond, no
subprocess, no discovery race).

Degrades gracefully: when rclpy / ROS is unavailable (a standalone dashboard
with no workspace sourced) construction leaves the publisher inert and callers
fall back to the shell-out path.
"""

from __future__ import annotations

import threading
from typing import Any


class EstopPublisher:
    """A launch-time-created, always-matched publisher for the e-stop topics."""

    ESTOP_TOPIC = "/openral/estop"
    CLEARED_TOPIC = "/openral/estop_cleared"

    def __init__(self) -> None:
        """Create the node + publishers now (at launch) so discovery happens early."""
        self._node: Any = None
        self._estop_pub: Any = None
        self._cleared_pub: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._empty_cls: Any = None
        self._owns_rclpy = False
        try:
            self._start()
        except Exception:  # reason: a ROS hiccup must never break the dashboard
            self.close()

    def _start(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import Empty

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._empty_cls = Empty
        self._node = Node("openral_dashboard_estop")
        # Match the HAL/kernel/runner estop subscriptions (RELIABLE, VOLATILE,
        # depth 10) so the publisher discovers + connects to them.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._estop_pub = self._node.create_publisher(Empty, self.ESTOP_TOPIC, qos)
        self._cleared_pub = self._node.create_publisher(Empty, self.CLEARED_TOPIC, qos)
        # Spin in a daemon thread so DDS discovery / liveliness run continuously
        # from launch; publishing itself happens from the request thread (rclpy
        # publishers are thread-safe), so a press never waits on the executor.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, name="openral_dashboard_estop_spin", daemon=True
        )
        self._thread.start()

    @property
    def available(self) -> bool:
        """True when the persistent publisher is live (rclpy + ROS present)."""
        return self._estop_pub is not None

    def trigger(self) -> bool:
        """Publish /openral/estop instantly. Returns False if unavailable."""
        return self._publish(self._estop_pub)

    def clear(self) -> bool:
        """Publish /openral/estop_cleared instantly. Returns False if unavailable."""
        return self._publish(self._cleared_pub)

    def _publish(self, pub: Any) -> bool:
        if pub is None or self._empty_cls is None:
            return False
        # Publish a few times: cheap insurance against a single lost datagram on
        # a safety topic. Idempotent (the estop is a latch), sub-millisecond.
        msg = self._empty_cls()
        for _ in range(3):
            pub.publish(msg)
        return True

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
        self._estop_pub = None
        self._cleared_pub = None
        self._executor = None
