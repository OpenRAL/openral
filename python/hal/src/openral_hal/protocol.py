"""HAL Protocol — the normative interface every Hardware Abstraction Layer adapter must satisfy.

The Protocol is structural (``runtime_checkable``), so any class that implements
the required attributes and methods is a valid HAL without needing to inherit from
this class.  See RFC §8.2 for the canonical definition.

Example:
    >>> from openral_hal.protocol import HAL
    >>> from openral_hal.ros_control import RosControlHAL
    >>> import inspect
    >>> # RosControlHAL satisfies the Protocol at runtime
    >>> issubclass(RosControlHAL, HAL)  # doctest: +SKIP
    True
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core.schemas import Action, JointState, RobotDescription

__all__ = ["HAL"]


@runtime_checkable
class HAL(Protocol):
    """Structural protocol that every HAL adapter must satisfy.

    The hot path (``read_state`` / ``send_action``) is synchronous and must
    complete within the robot's control cycle budget.  Blocking I/O or memory
    allocation inside these methods is forbidden.

    Implementors must raise ``ROSEStopRequested`` (a ``ROSSafetyViolation``)
    from ``estop()`` so that the safety supervisor can catch it at the boundary
    and trigger an incident log entry.

    Attributes:
        description: Normative ``RobotDescription`` manifest for this robot.
    """

    description: RobotDescription

    def connect(self) -> None:
        """Open the connection to the robot hardware or simulator.

        Raises:
            ROSConfigError: If the URDF, controller name, or topic cannot be
                resolved.
            ROSRuntimeError: If the underlying transport fails to initialise.
        """
        ...

    def disconnect(self) -> None:
        """Close the connection and release all resources gracefully.

        Must be idempotent — calling ``disconnect()`` on an already-disconnected
        HAL must not raise.
        """
        ...

    def read_state(self) -> JointState:
        """Return the latest joint state snapshot.

        Raises:
            ROSRuntimeError: If the HAL is not connected.
            ROSPerceptionStale: If the most recent reading exceeds the staleness
                deadline configured in the ``RobotDescription``.

        Returns:
            The latest ``JointState`` for all joints listed in
            ``description.joints``.
        """
        ...

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the underlying controller.

        The HAL is responsible only for forwarding the typed action; safety
        clamping happens in the C++ safety kernel before this call is made.

        Args:
            action: The ``Action`` produced by a Skill or the safety shaper.

        Raises:
            ROSRuntimeError: If the HAL is not connected.
            ROSConfigError: If the action's ``control_mode`` is incompatible
                with the robot's ``supported_control_modes``.
        """
        ...

    def estop(self) -> None:
        """Trigger an emergency stop.

        Raises:
            ROSEStopRequested: Always.  Callers must NOT catch this silently.
                Only the safety supervisor boundary may catch it.
        """
        ...
