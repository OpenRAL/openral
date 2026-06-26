#!/usr/bin/env python3
"""Tier-C critic producer node (ADR-0064 / ADR-0074 / observability audit P1 R3).

Subscribes the generic ``/openral/critic/score`` topic (``openral_msgs/CriticScore``)
that reward models publish — the Robometer reward rSkill today (ADR-0057), a
future SARM, success classifiers — routes each self-describing
``(critic_id, score, threshold)`` sample through a
:class:`~openral_reasoner.CriticWatchdogGroup`, and on a **stall or success**
publishes a Tier-C ``FailureTrigger`` (``KIND_CRITIC`` / ``SEVERITY_FAIL``, the
emitted ``CriticEvidence``, ``trace_id`` propagated) on ``/openral/failure/critic``
via :class:`~openral_observability.failure_bus.FailureBusPublisher`. The
``reasoner_node`` maps that FAIL event onto a forced Tier-C tick
(``ReasonerCore.tick(force=True, tier="C")``).

A **stall** fires when ``stall_patience`` consecutive sub-threshold,
non-improving observations accumulate. A **success** fires the first time
``score >= threshold`` per streak (ADR-0074 reward-watcher), so the reasoner is
woken the moment an attempt is likely done — not only after a subsequent stall.

The producer keys one watchdog per ``critic_id``, so several reward models share
the single ``/openral/failure/critic`` source and each fires independently — no
producer-side config to onboard a new model.

Advisory only (CLAUDE.md §1.1): it produces a *planning* signal that drives
replanning; it never actuates, never commands the C++ safety kernel, never
touches E-stop.

Parameters:
    score_topic (str): critic-score topic to watch. Default ``/openral/critic/score``.
    stall_patience (int): consecutive stalled samples per critic before firing.
        Default 5. Must be ``>= 1``.
    min_delta (float): minimum strict improvement over a critic's running best
        counted as progress. Default 0.02 — a small positive floor so sub-threshold
        reward-model noise (e.g. Robometer progress jittering 0.620↔0.622) does not
        keep re-arming the watchdog after it has latched.
"""

from __future__ import annotations

from typing import Any

import rclpy
from openral_observability.failure_bus import (
    KIND_CRITIC,
    SEVERITY_FAIL,
    FailureBusPublisher,
    FailureSource,
)
from openral_reasoner import CriticWatchdogGroup
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


class CriticProducerNode(Node):
    """Watch ``/openral/critic/score`` and emit a Tier-C FailureTrigger on a stall or success."""

    def __init__(self) -> None:
        """Read params, build the watchdog group + failure-bus publisher, subscribe."""
        super().__init__("openral_critic_producer")
        self.declare_parameter("score_topic", "/openral/critic/score")
        self.declare_parameter("stall_patience", 5)
        self.declare_parameter("min_delta", 0.02)

        gp = self.get_parameter
        topic = gp("score_topic").get_parameter_value().string_value
        patience = gp("stall_patience").get_parameter_value().integer_value
        min_delta = gp("min_delta").get_parameter_value().double_value

        self._group = CriticWatchdogGroup(stall_patience=patience, min_delta=min_delta)
        self._bus = FailureBusPublisher(self, FailureSource.CRITIC)
        self._bus.create_publisher()
        self._bus.start()

        from openral_msgs.msg import CriticScore

        # Low-rate advisory stream: RELIABLE so no stall sample is dropped.
        score_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(CriticScore, topic, self._on_score, score_qos)
        self.get_logger().info(
            f"critic_producer: watching {topic!r} -> {self._bus.topic} "
            f"(stall_patience={patience}, min_delta={min_delta}); "
            f"fires on stall OR success (score >= threshold)"
        )

    def _on_score(self, msg: Any) -> None:
        """Route one CriticScore through the group; publish on a stall or success."""
        evidence = self._group.observe(
            critic_id=msg.critic_id,
            score=float(msg.score),
            threshold=float(msg.threshold),
        )
        if evidence is None:
            return
        # Propagate the score sample's trace_id; fall back to the active OTel
        # context inside FailureBusPublisher when the field is empty.
        published = self._bus.publish(
            kind=KIND_CRITIC,
            severity=SEVERITY_FAIL,
            evidence=evidence,
            trace_id=msg.trace_id or None,
        )
        trigger = "success" if evidence.score >= evidence.threshold else "stall"
        self.get_logger().warning(
            f"critic {trigger}: critic_id={evidence.critic_id!r} "
            f"score={evidence.score:.3f} threshold={evidence.threshold:.3f} "
            f"-> {self._bus.topic} (published={published})"
        )

    def destroy_node(self) -> None:
        """Tear down the failure-bus publisher before the node."""
        self._bus.destroy()
        super().destroy_node()


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the critic producer node, shut down cleanly."""
    rclpy.init(args=args)
    node = CriticProducerNode()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
