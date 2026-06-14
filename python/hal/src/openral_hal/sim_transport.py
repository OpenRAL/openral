"""SimTransport -- in-memory simulated ros2_control transport.

Replaces ``MagicMock`` in unit tests for ``RosControlHAL`` with a typed,
stateful simulation that applies published joint trajectory commands to
internal state.  This creates a closed-loop test environment: calling
``send_action`` publishes via ``publish()``, which updates positions
that ``state()`` then exposes to ``read_state()``.

Example:
    >>> from openral_hal.sim_transport import SimTransport
    >>> transport = SimTransport(n_joints=3)
    >>> transport.publish(
    ...     "/ctrl/joint_trajectory",
    ...     {
    ...         "joint_targets": [[1.0, 2.0, 3.0]],
    ...         "control_mode": "joint_position",
    ...         "horizon": 1,
    ...         "stamp_ns": 0,
    ...     },
    ... )
    >>> transport.state()["position"]
    [1.0, 2.0, 3.0]
    >>> transport.call_count
    1
"""

from __future__ import annotations

__all__ = ["SimTransport"]


class SimTransport:
    """In-memory transport simulating a ros2_control joint trajectory controller.

    Published joint trajectory commands update internal joint positions, which
    are then returned by :meth:`state`.  All published messages are recorded
    for assertion in tests.

    Args:
        n_joints: Number of joints to simulate.  Initial positions, velocities,
            and efforts are all ``0.0``.

    Example:
        >>> transport = SimTransport(n_joints=2)
        >>> transport.publish(
        ...     "/ctrl/traj",
        ...     {
        ...         "joint_targets": [[0.5, -0.5]],
        ...         "control_mode": "joint_position",
        ...         "horizon": 1,
        ...         "stamp_ns": 0,
        ...     },
        ... )
        >>> tuple(transport.state()["position"])
        (0.5, -0.5)
        >>> transport.call_count
        1
        >>> topic, msg = transport.last_call  # type: ignore[misc]
        >>> topic
        '/ctrl/traj'
    """

    def __init__(self, n_joints: int) -> None:
        """Initialise zeroed joint state for *n_joints* joints."""
        self._n_joints = n_joints
        self._positions: list[float] = [0.0] * n_joints
        self._velocities: list[float] = [0.0] * n_joints
        self._efforts: list[float] = [0.0] * n_joints
        self._published: list[tuple[str, dict[str, object]]] = []

    # -- Transport callables (injected into RosControlHAL) --------------------

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Record *msg* and apply ``joint_targets`` to internal state.

        If ``msg["joint_targets"]`` is a list of trajectory steps, the **last
        step** is applied to positions -- mirroring how a real joint trajectory
        controller would reach the final waypoint.

        Args:
            topic: The ROS 2 topic name.
            msg: The published message dict.
        """
        self._published.append((topic, msg))
        targets = msg.get("joint_targets")
        if isinstance(targets, list) and targets:
            last_step = targets[-1]
            if isinstance(last_step, list):
                self._positions = [float(v) for v in last_step]

    def state(self) -> dict[str, object]:
        """Return the current simulated joint state.

        Returns:
            Dict with ``"position"``, ``"velocity"``, and ``"effort"`` keys,
            each containing a ``list[float]`` of length ``n_joints``.
        """
        return {
            "position": list(self._positions),
            "velocity": list(self._velocities),
            "effort": list(self._efforts),
        }

    # -- Introspection (replaces MagicMock assertions) ------------------------

    @property
    def call_count(self) -> int:
        """Number of times :meth:`publish` has been called."""
        return len(self._published)

    @property
    def last_call(self) -> tuple[str, dict[str, object]] | None:
        """The most recent ``(topic, msg)`` pair, or ``None``."""
        return self._published[-1] if self._published else None

    @property
    def calls(self) -> list[tuple[str, dict[str, object]]]:
        """All ``(topic, msg)`` pairs in chronological order."""
        return list(self._published)
