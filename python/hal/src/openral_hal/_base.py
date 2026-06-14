"""Shared HAL helpers — collapse duplicated `_require_connected` / `_validate_action` patterns.

`HALBase` is a non-ABC mixin: it owns the `_connected` flag and the two
helpers that were copy-pasted across every adapter. Each adapter still
implements its own `connect` / `disconnect` / `read_state` / `send_action` /
`estop` because those bodies are genuinely bespoke (vendor SDK calls, MJCF
buffers, ROS publishers). The goal is to delete duplication, not to invent
a uniform lifecycle that no real adapter wants.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription


class HALBase:
    """Shared state + helpers for every HAL adapter.

    Subclasses set `self.description` and `self._connected = False` in their
    own `__init__` (the explicit assignment avoids constraining the long
    per-adapter constructor signatures).
    """

    description: RobotDescription
    _connected: bool

    def _require_connected(self, operation: str) -> None:
        if not self._connected:
            raise ROSRuntimeError(
                f"{type(self).__name__}.{operation}() called while not connected. "
                "Call connect() first."
            )

    def _require_control_mode(self, action: Action, allowed: ControlMode) -> None:
        if action.control_mode != allowed:
            raise ROSConfigError(
                f"{type(self).__name__} only supports {allowed.value}; got {action.control_mode!r}."
            )

    def _validate_action_dims(self, action: Action, joint_count: int) -> None:
        if action.joint_targets is None:
            return
        for step_idx, step in enumerate(action.joint_targets):
            if len(step) != joint_count:
                raise ROSConfigError(
                    f"Action joint_targets[{step_idx}] has {len(step)} values "
                    f"but robot '{self.description.name}' has {joint_count} joints."
                )
