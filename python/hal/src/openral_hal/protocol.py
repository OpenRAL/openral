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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from openral_core.schemas import Action, JointState, RobotDescription

__all__ = [
    "HAL",
    "EStopRecovery",
    "HALHealthProvider",
    "HALHealthReport",
    "LifecycleEStopHAL",
    "ResettableLifecycleEStopHAL",
]


class EStopRecovery(StrEnum):
    """Recovery policy after a lifecycle node forwards an e-stop to a HAL."""

    RESETTABLE = "resettable"
    RESTART_REQUIRED = "restart_required"


@dataclass(frozen=True)
class HALHealthReport:
    """Cached health data exposed through the lifecycle diagnostics heartbeat.

    ``health()`` implementations must not perform device or network I/O. The
    lifecycle node calls them from its low-rate diagnostics path.
    """

    message: str
    fields: Mapping[str, str]


@runtime_checkable
class HALHealthProvider(Protocol):
    """Optional HAL extension for cached diagnostics."""

    def health(self) -> HALHealthReport:
        """Return the latest cached health report without performing I/O."""
        ...


@runtime_checkable
class LifecycleEStopHAL(Protocol):
    """Optional extension for HALs that propagate the lifecycle e-stop downstream.

    HALs that implement this protocol opt into receiving ``estop()`` when the
    generic lifecycle node latches ``/openral/estop``. Existing HALs that do not
    opt in retain the lifecycle node's local latch-only behavior.
    """

    estop_recovery: EStopRecovery

    def estop(self) -> None:
        """Stop downstream hardware or owned processes and always raise."""
        ...


@runtime_checkable
class ResettableLifecycleEStopHAL(LifecycleEStopHAL, Protocol):
    """Lifecycle e-stop extension for HALs that support in-process recovery.

    The contract every implementation keeps: the robot comes back **holding its
    present pose** (no motion toward a stale goal), and the HAL accepts commands
    only after the re-arm is confirmed by the hardware. On any failure it raises,
    leaves the hardware stopped, and the lifecycle node keeps its latch.
    """

    def reset_estop(self) -> None:
        """Re-arm the downstream controller after its safety conditions pass."""
        ...


@runtime_checkable
class EStopObservableHAL(Protocol):
    """Optional extension: the HAL can still report joint state while e-stopped.

    The lifecycle node stops polling a real HAL once it latches an e-stop, because
    many stops tear the transport down. A HAL whose stop leaves a read path open
    (servos de-energized, bus still answering reads) implements this, and the node
    keeps publishing ``/joint_states`` through the latch, so the 3D view and the
    safety kernel see the robot moved by hand or by gravity.

    ``read_state_estopped`` must never write to the hardware or energize anything.
    """

    def read_state_estopped(self) -> JointState:
        """Read the joint state during an e-stop, read-only.

        Raises:
            ROSRuntimeError: No read path is open (e.g. the stop was not proven,
                so the transport was closed). The node then skips the tick.
        """
        ...


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
