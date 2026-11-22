"""Shared HAL helpers — collapse duplicated `_require_connected` / `_validate_action` patterns.

`HALBase` is a non-ABC mixin: it owns the `_connected` flag, the two
validation helpers, and a default flag-and-log `disconnect` for adapters with
nothing else to tear down. Each adapter still implements its own `connect` /
`read_state` / `send_action` / `estop` because those bodies are genuinely
bespoke (vendor SDK calls, MJCF buffers, ROS publishers); adapters that hold a
real resource (SDK handle, MuJoCo buffers, USB port) still override
`disconnect` to release it. The goal is to delete duplication, not to invent
a uniform lifecycle that no real adapter wants.
"""

from __future__ import annotations

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription

log = structlog.get_logger(__name__)


def resolve_staleness_limit_s(description: RobotDescription, override: float | None) -> float:
    """A real HAL's joint-state freshness window: an explicit value, else the manifest's.

    The manifest's ``safety.joint_state_staleness_limit_s`` is the one place a rig
    declares it (``build_hal`` passes it as ``staleness_limit_s``); a HAL built
    directly without it reads its description. No constructor default: a window
    nobody measured must not reach the actuation path silently.

    Raises:
        ROSConfigError: Neither an override nor a manifest value exists.

    Example:
        >>> from openral_core.schemas import RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/aloha_bimanual/robot.yaml")
        >>> resolve_staleness_limit_s(desc, None)
        0.2
        >>> resolve_staleness_limit_s(desc, 0.05)
        0.05
    """
    if override is not None:
        return float(override)
    declared = description.safety.joint_state_staleness_limit_s
    if declared is None:
        raise ROSConfigError(
            f"robot {description.name!r} declares no safety.joint_state_staleness_limit_s "
            "and no staleness_limit_s was passed; declare the measured window in the "
            "manifest (tools/joint_state_staleness_probe.py --robot)."
        )
    return float(declared)


def require_connected(connected: bool, backend: str, operation: str) -> None:
    """The one "not connected" refusal every HAL (and the MuJoCo twin) raises.

    Raises:
        ROSRuntimeError: ``connected`` is false: before ``connect`` / after ``disconnect``.

    Example:
        >>> require_connected(True, "UR5eHAL", "read_state")
    """
    if not connected:
        raise ROSRuntimeError(
            f"{backend}.{operation}() called while not connected (before connect() or "
            "after disconnect()). Call connect() first."
        )


def require_supported_control_mode(description: RobotDescription, action: Action) -> None:
    """Refuse a mode outside the manifest's ``supported_control_modes`` (empty: any).

    Raises:
        ROSConfigError: ``action.control_mode`` is not supported by the robot.
    """
    supported = description.capabilities.supported_control_modes
    if supported and action.control_mode not in supported:
        raise ROSConfigError(
            f"Action control_mode '{action.control_mode}' is not in "
            f"supported_control_modes {supported} for robot '{description.name}'."
        )


def require_control_mode(action: Action, allowed: ControlMode, backend: str) -> None:
    """Refuse every mode but ``allowed`` (a HAL that drives one mode).

    Raises:
        ROSConfigError: ``action.control_mode`` is not ``allowed``.
    """
    if action.control_mode != allowed:
        raise ROSConfigError(
            f"{backend} only supports {allowed.value}; got {action.control_mode!r}."
        )


def validate_action_dims(action: Action, joint_count: int, robot: str) -> None:
    """Refuse a ``joint_targets`` row that is not one target per joint (no rows: nothing).

    Raises:
        ROSConfigError: A row of the wrong width.
    """
    if action.joint_targets is None:
        return
    for step_idx, step in enumerate(action.joint_targets):
        if len(step) != joint_count:
            raise ROSConfigError(
                f"Action joint_targets[{step_idx}] has {len(step)} values "
                f"but robot '{robot}' has {joint_count} joints."
            )


class HALBase:
    """Shared state + helpers for every HAL adapter.

    Subclasses set `self.description` and `self._connected = False` in their
    own `__init__` (the explicit assignment avoids constraining the long
    per-adapter constructor signatures). The checks are the module-level
    validators above, shared with the sim HAL shell and the MuJoCo twin.
    """

    description: RobotDescription
    _connected: bool

    def _require_connected(self, operation: str) -> None:
        require_connected(self._connected, type(self).__name__, operation)

    def disconnect(self) -> None:
        """Close the connection.  Idempotent.

        Default for adapters with no extra resource to release. Adapters
        holding a real handle (SDK connection, MuJoCo buffers, USB port)
        override this to release it before calling (or replacing) this body.
        """
        if not self._connected:
            return
        log.info("hal.disconnect", robot=self.description.name)
        self._connected = False

    def _validate_action_dims(self, action: Action, joint_count: int) -> None:
        validate_action_dims(action, joint_count, self.description.name)


def _raw_floats(raw: dict[str, object], key: str, width: int) -> list[float]:
    """Return `raw[key]` as a list of floats, or `width` zeros if missing/not a list."""
    val = raw.get(key)
    if isinstance(val, list):
        return [float(v) for v in val]
    return [0.0] * width
