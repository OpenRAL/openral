"""Persistent ROS 2 e-stop publisher for the dashboard (safety-critical).

A shell-out ``ros2 topic pub`` per press spawns a FRESH publisher that must
rediscover subscribers before delivering — ~1 s, and racing discovery, so the
message could be published before any subscriber matched and be silently
LOST (observed live: the robot kept moving).

This holds ONE publisher, created at dashboard startup, so DDS discovery of the
HAL / kernel / runner subscribers happens ONCE at launch. Every later press then
publishes instantly on the already-matched publisher (sub-millisecond, no
subprocess, no discovery race).

The operator prompt rides the same node (openral#679): a shell-out ``openral prompt``
per prompt had the same race, and a prompt it reported sent sometimes never reached
the prompt router.

Degrades gracefully: when rclpy / ROS is unavailable (a standalone dashboard
with no workspace sourced) construction leaves the publisher inert and callers
fall back to the shell-out path.
"""

from __future__ import annotations

import threading
from typing import Any

from openral_observability.rclpy_spin import spin_executor_until_shutdown


class EstopPublisher:
    """A launch-time-created, always-matched publisher for the e-stop topics."""

    ESTOP_TOPIC = "/openral/estop"
    CLEARED_TOPIC = "/openral/estop_cleared"
    PROMPT_TOPIC = "/openral/prompt_in/dashboard"

    def __init__(self) -> None:
        """Create the node + publishers now (at launch) so discovery happens early."""
        self._node: Any = None
        self._estop_pub: Any = None
        self._cleared_pub: Any = None
        self._prompt_pub: Any = None
        self._prompt_cls: Any = None
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
        # The operator prompt, with the prompt router's input QoS (RELIABLE, VOLATILE,
        # depth 10, as ``openral prompt``). Without openral_msgs the prompt falls back to
        # the shell-out; E-STOP is unaffected.
        try:
            from openral_msgs.msg import PromptStamped

            self._prompt_cls = PromptStamped
            self._prompt_pub = self._node.create_publisher(PromptStamped, self.PROMPT_TOPIC, qos)
        except ImportError:
            self._prompt_pub = None
        # Spin in a daemon thread so DDS discovery / liveliness run continuously
        # from launch; publishing itself happens from the request thread (rclpy
        # publishers are thread-safe), so a press never waits on the executor.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=spin_executor_until_shutdown,
            args=(self._executor,),
            name="openral_dashboard_estop_spin",
            daemon=True,
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

    @property
    def prompts(self) -> bool:
        """True when operator prompts go through this node, not the ``openral prompt`` CLI."""
        return self._prompt_pub is not None

    def prompt(self, text: str, metadata_json: str) -> int | None:
        """Publish one operator prompt on the dashboard's prompt-router input.

        Returns how many subscribers it reached (0: no prompt router on the graph, so nothing will
        act on it), or ``None`` when this node carries no prompts (use the shell-out).
        """
        if self._prompt_pub is None or self._prompt_cls is None or self._node is None:
            return None
        matched = int(self._prompt_pub.get_subscription_count())
        if matched == 0:
            return 0
        msg = self._prompt_cls()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_dashboard"
        msg.text = text
        msg.metadata_json = metadata_json
        self._prompt_pub.publish(msg)  # once: a prompt is not idempotent like the e-stop latch
        return matched

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
        self._prompt_pub = None
        self._executor = None
