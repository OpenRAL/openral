"""Internal Protocol shim that mirrors ``openral_runner.SensorReader``.

The :class:`openral_sensors.ros_publisher.SensorRosPublisher` accepts any
reader satisfying the ``open / close / read_latest`` structural contract.
Importing ``openral_runner.sensor_reader.SensorReader`` directly would
create a sensors→runner→sensors dependency cycle (the runner already
imports the catalog from this package). We declare a Protocol that
matches the runner's surface verbatim — duck-typing via the Protocol
gives us mypy-level structural coverage without the cycle.

If the runner's protocol grows, mirror it here in the same PR (and add
a regression test in this package that exercises a concrete reader
through this Protocol).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openral_core import SensorFrame

__all__ = ["SensorReaderLike"]


@runtime_checkable
class SensorReaderLike(Protocol):
    """Structural alias of :class:`openral_runner.sensor_reader.SensorReader`.

    Attributes:
        sensor_id: Sensor name; lands on ``Image.header.frame_id``
            unless the publisher overrides via its ``frame_id`` kwarg.
        is_open: ``True`` between :meth:`open` and :meth:`close`.
    """

    sensor_id: str
    is_open: bool

    def open(self) -> None:
        """Acquire the capture device and start any background workers."""

    def close(self) -> None:
        """Release the capture device. Idempotent."""

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the most recent buffered :class:`SensorFrame`. Non-blocking."""
