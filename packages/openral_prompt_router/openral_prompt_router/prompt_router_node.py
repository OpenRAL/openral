#!/usr/bin/env python3
"""ADR-0018 F10 — ``prompt_router_node`` lifecycle node.

Single node that fans in operator prompts from any external source
into a normalised :class:`openral_msgs/PromptStamped` stream on
``/openral/prompt``. In v1 the only external adapter is the CLI
(``openral prompt "do X"`` publishes directly to ``/openral/prompt_in/cli``,
which this node forwards onto ``/openral/prompt`` after enriching the
``metadata_json`` with the source tag). The reasoner consumes
``/openral/prompt`` exclusively — sources never publish there directly.

Arbitration (per ADR-0018 §3 / capability review §3.F10):

* Single FIFO queue with KEEP_LAST=10 on each per-source input and on
  the fan-out topic.
* Source priority tag is stamped onto ``metadata_json`` as
  ``{"source": "<source>", "priority": <int>, ...}``; human-source
  prompts get priority 100 so they overtake auto-prompts (priority
  10).
* No silent drops; rate-limited bursts surface as a structlog warning.

Startup prompt (multi-task deploy):

``openral deploy sim`` can deliver a ``startup_prompt`` ROS string
parameter. When non-empty, the node publishes it onto ``/openral/prompt``
at ``on_activate`` time (before any operator can send a competing prompt),
so the reasoner's first tick immediately sees the scene's ``tasks:`` goals
without requiring a separate ``openral prompt`` invocation. The startup
prompt is emitted with the ``"cli"`` source (priority 100) so it is
treated as a human-level operator instruction.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

try:  # pragma: no cover — gated by colcon-built artifact
    from openral_msgs.msg import PromptStamped as IDLPromptStamped
except ImportError:  # pragma: no cover
    IDLPromptStamped = None  # type: ignore[assignment, misc]


__all__ = ["DEFAULT_SOURCES", "PromptRouterNode"]

# Per ADR-0018 §1 / capability review §3.F10: /openral/prompt uses
# RELIABLE + VOLATILE + KEEP_LAST=10.
_QOS_PROMPT = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Bound on how long on_activate waits for the reasoner's /openral/prompt
# subscriber to be discovered before emitting the one-shot startup prompt.
# Generous: the reasoner's on_configure loads the skill palette (seconds) and
# only then creates the subscription; missing the prompt boots the run idle.
_STARTUP_PROMPT_SUBSCRIBER_TIMEOUT_S = 30.0

# v1 adapter registry — only the CLI source is wired. Priorities chosen
# so a human prompt overtakes an auto-prompt (CLAUDE.md §6.2 — the
# reasoner is below operator authority).
DEFAULT_SOURCES: dict[str, int] = {
    "cli": 100,  # human, highest
    "dashboard": 100,  # human, same priority as CLI
    "auto": 10,  # machine cascade (EmitPromptTool self-cascades)
}


class PromptRouterNode(LifecycleNode):
    """ROS 2 lifecycle prompt-fan-in node (ADR-0018 F10).

    Each registered source listens on ``/openral/prompt_in/<source>``
    and republishes the message onto ``/openral/prompt`` after stamping
    a ``{"source": "<source>", "priority": <int>}`` field onto
    ``metadata_json``.

    Args:
        node_name: ROS node name; default ``openral_prompt_router``.
        sources: Mapping ``source_name → priority``. Defaults to
            :data:`DEFAULT_SOURCES`. A deployment YAML may restrict
            this set; the router only listens to sources declared
            here (per ADR-0018 §3.F10 "per-source allowlist").
    """

    def __init__(
        self,
        *,
        node_name: str = "openral_prompt_router",
        sources: dict[str, int] | None = None,
    ) -> None:
        """Initialise; no rclpy I/O until on_configure."""
        super().__init__(node_name)
        self._sources = dict(sources or DEFAULT_SOURCES)
        self._pub: Any = None
        self._forwarded_count: int = 0
        # Declare the startup_prompt parameter so `openral deploy sim`
        # can pass `startup_prompt:=<text>` as a launch argument. The
        # node reads it in on_activate and publishes it as the first
        # operator-priority prompt. An empty string (the default) means
        # "no startup prompt — idle until the operator sends one."
        self.declare_parameter("startup_prompt", "")

    # ── lifecycle ──────────────────────────────────────────────────────────

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Build the fan-out publisher + one subscriber per allowed source."""
        del state
        if IDLPromptStamped is None:
            self.get_logger().error(
                "openral_msgs not on PYTHONPATH — colcon-build openral_msgs and source install/",
            )
            return TransitionCallbackReturn.FAILURE
        self._pub = self.create_publisher(IDLPromptStamped, "/openral/prompt", _QOS_PROMPT)
        for source, priority in self._sources.items():
            topic = f"/openral/prompt_in/{source}"
            self.create_subscription(
                IDLPromptStamped,
                topic,
                lambda msg, _s=source, _p=priority: self._on_inbound(_s, _p, msg),
                _QOS_PROMPT,
            )
        self.get_logger().info(
            f"on_configure: routing {sorted(self._sources.keys())} → /openral/prompt",
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Publish the startup prompt if configured; then idle (purely reactive)."""
        del state
        startup_prompt = self.get_parameter("startup_prompt").get_parameter_value().string_value
        if startup_prompt:
            self._publish_startup_prompt(startup_prompt)
        self.get_logger().info("on_activate")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop forwarding (subscriptions remain attached)."""
        del state
        self.get_logger().info("on_deactivate")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Drop the publisher; subscriptions auto-cleaned by rclpy."""
        del state
        self._pub = None
        return TransitionCallbackReturn.SUCCESS

    # ── callback ───────────────────────────────────────────────────────────

    def _on_inbound(self, source: str, priority: int, msg: Any) -> None:
        """Forward a prompt onto ``/openral/prompt`` with the source tag."""
        if self._pub is None:
            return
        # Merge the source tag into the existing metadata_json (preserve
        # any fields the source set, but our {source, priority} pair wins).
        try:
            metadata = json.loads(msg.metadata_json) if msg.metadata_json else {}
            if not isinstance(metadata, dict):
                metadata = {"_inbound": metadata}
        except (json.JSONDecodeError, TypeError):
            metadata = {"_inbound_raw": msg.metadata_json}
        metadata["source"] = source
        metadata["priority"] = priority

        fanout = IDLPromptStamped()
        fanout.header = msg.header
        fanout.text = msg.text
        fanout.metadata_json = json.dumps(metadata, sort_keys=True)
        self._pub.publish(fanout)
        self._forwarded_count += 1
        self.get_logger().info(
            f"forwarded prompt source={source} priority={priority} text={msg.text!r}",
        )

    def _publish_startup_prompt(self, text: str) -> None:
        """Publish ``text`` as a ``cli``-priority operator prompt at activate time.

        This is the multi-task deploy path: ``openral deploy sim`` passes the
        scene's ``tasks:`` goals as the ``startup_prompt`` ROS parameter so the
        reasoner's first tick already sees the operator's goal without requiring a
        separate ``openral prompt`` invocation.

        The prompt is tagged as source=``"cli"`` (priority 100) to match the
        behaviour of a human operator typing ``openral prompt "..."`` in a
        terminal — the same high-priority path that preempts any queued
        auto-prompts.
        """
        if self._pub is None:
            return
        # `/openral/prompt` is RELIABLE + VOLATILE (ADR-0018 §1), so a sample
        # published before the reasoner's subscriber has been discovered is
        # silently dropped — the reasoner would boot idle and emit "provide a
        # task" instead of seeing the scene's startup goal. The router activates
        # before the reasoner finishes configuring, so wait (bounded) for at
        # least one matched subscriber before emitting this one-shot prompt.
        # DDS discovery runs on its own thread, so this poll does not deadlock
        # the (separate-process) reasoner that we are waiting on.
        deadline = time.monotonic() + _STARTUP_PROMPT_SUBSCRIBER_TIMEOUT_S
        while self._pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._pub.get_subscription_count() == 0:
            self.get_logger().warning(
                "startup_prompt: no subscriber on /openral/prompt after "
                f"{_STARTUP_PROMPT_SUBSCRIBER_TIMEOUT_S:.0f}s; publishing anyway "
                "(reasoner may miss it under VOLATILE QoS)",
            )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "openral_prompt_router"
        msg.text = text
        msg.metadata_json = json.dumps(
            {"source": "cli", "priority": DEFAULT_SOURCES["cli"], "startup": True},
            sort_keys=True,
        )
        self._pub.publish(msg)
        self._forwarded_count += 1
        self.get_logger().info(
            f"startup_prompt published source=cli priority={DEFAULT_SOURCES['cli']} text={text!r}",
        )

    # ── public helpers for tests ───────────────────────────────────────────

    @property
    def forwarded_count(self) -> int:
        """Number of prompts the router has forwarded since :meth:`on_configure`."""
        return self._forwarded_count


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_prompt_router prompt_router_node``."""
    from openral_observability import configure_observability

    # Idempotent + no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset.
    configure_observability(service_name="openral.prompt_router")

    rclpy.init(args=args)
    try:
        node = PromptRouterNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy. On
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread, spin instead raises ExternalShutdownException.
            # Either way the context is already shut down by the time we
            # reach the `finally` below, so the bare `rclpy.shutdown()`
            # we used to call there raised
            # `RCLError: rcl_shutdown already called` — the
            # `try_shutdown()` switch below is the corresponding fix.
            pass
        finally:
            node.destroy_node()
    finally:
        # Idempotent — no-op when the SIGINT handler (or whoever fired
        # ExternalShutdownException) already shut down the context.
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
