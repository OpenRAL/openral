"""Shared QoS profiles for this package's lifecycle estop nodes.

``deadman_watchdog_node`` and ``hardware_estop_node`` both publish
``/openral/estop`` and ``/openral/failure/safety`` with the same QoS; this
module is the single definition so the two stay byte-identical by
construction instead of by two people remembering to copy the same numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rclpy.qos import QoSProfile

__all__ = ["estop_qos", "failure_qos"]


def estop_qos() -> QoSProfile:
    """QoS for ``/openral/estop``: ``RELIABLE`` / ``VOLATILE`` / ``KEEP_LAST=10``."""
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )


def failure_qos() -> QoSProfile:
    """QoS for ``/openral/failure/safety``: ``RELIABLE`` / ``VOLATILE`` / ``KEEP_LAST=50``."""
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=50,
    )
