"""SO100DigitalTwin — in-process simulator for the SO-100 follower arm.

A real lerobot ``Robot`` subclass that implements the full lerobot interface
without any serial port or physical hardware.  Joint state is stored in memory
and updated by ``send_action()``, making it suitable as a digital twin for
testing the full ``SO100FollowerHAL`` adapter code path.

Unit angles
-----------
lerobot's convention (with ``use_degrees=True``) is **degrees** for revolute
joints and [0, 100] for the gripper.  This class follows the same convention
so that ``SO100FollowerHAL`` exercises its full degrees↔radians conversion.

Example:
    >>> from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
    >>> twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    >>> twin.connect(calibrate=False)
    >>> obs = twin.get_observation()
    >>> list(obs.keys())  # doctest: +NORMALIZE_WHITESPACE
    ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
     'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    >>> twin.disconnect()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

__all__ = ["SO100DigitalTwin", "SO100DigitalTwinConfig"]

# Canonical joint order — mirrors lerobot's SO100Follower motor dict.
_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default joint positions in lerobot's native units:
# revolute joints in degrees, gripper in [0, 100].
_DEFAULT_POSITIONS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,  # mid-range (half-open)
}


@dataclass(kw_only=True)
class SO100DigitalTwinConfig(RobotConfig):  # type: ignore[misc]
    """Configuration for the SO-100 digital twin.

    Args:
        initial_positions: Initial joint positions in lerobot's native units
            (revolute joints in degrees, gripper in [0, 100]).  Defaults to
            all-zero revolute joints and gripper at 50 (half-open).

    Example:
        >>> cfg = SO100DigitalTwinConfig(initial_positions={"shoulder_pan": 90.0})
        >>> cfg.initial_positions["shoulder_pan"]
        90.0
    """

    initial_positions: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_POSITIONS))


class SO100DigitalTwin(Robot):  # type: ignore[misc]
    """In-process digital twin for the SO-100 follower arm.

    Implements the full lerobot ``Robot`` interface without any serial port.
    Joint state is initialised from ``config.initial_positions`` and updated
    each time ``send_action()`` is called, so observation sequences reflect
    sent commands.

    Args:
        config: ``SO100DigitalTwinConfig`` with optional initial positions.

    Example:
        >>> cfg = SO100DigitalTwinConfig()
        >>> twin = SO100DigitalTwin(cfg)
        >>> twin.connect(calibrate=False)
        >>> twin.is_connected
        True
        >>> twin.get_observation()["gripper.pos"]
        50.0
        >>> twin.disconnect()
    """

    config_class = SO100DigitalTwinConfig
    name = "so100_digital_twin"

    def __init__(self, config: SO100DigitalTwinConfig) -> None:
        """Initialise the digital twin; no connection is opened."""
        super().__init__(config)
        self.config = config
        self._is_connected: bool = False
        self._is_calibrated: bool = True  # pre-calibrated; no wizard needed
        # Internal state: positions in lerobot's native units
        self._positions: dict[str, float] = {
            name: config.initial_positions.get(name, _DEFAULT_POSITIONS[name])
            for name in _JOINT_NAMES
        }

    # ── lerobot.Robot required properties ────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict[str, type]:
        """Observation feature schema — one float per joint position.

        Returns:
            Dict mapping ``"<joint>.pos"`` keys to ``float`` type.
        """
        return {f"{name}.pos": float for name in _JOINT_NAMES}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action feature schema — one float per joint position target.

        Returns:
            Dict mapping ``"<joint>.pos"`` keys to ``float`` type.
        """
        return {f"{name}.pos": float for name in _JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        """True if ``connect()`` has been called and ``disconnect()`` has not."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Always True — the digital twin requires no calibration."""
        return self._is_calibrated

    # ── lerobot.Robot required methods ────────────────────────────────────────

    @check_if_already_connected  # type: ignore[untyped-decorator]
    def connect(self, calibrate: bool = True) -> None:
        """Activate the twin.  No serial port is opened.

        Args:
            calibrate: Ignored — the twin is always pre-calibrated.
        """
        self._is_connected = True

    def calibrate(self) -> None:
        """No-op — the digital twin requires no interactive calibration."""
        self._is_calibrated = True

    def configure(self) -> None:
        """No-op — no bus configuration needed."""

    @check_if_not_connected  # type: ignore[untyped-decorator]
    def get_observation(self) -> RobotObservation:
        """Return current joint positions in lerobot's native units.

        Returns:
            Dict ``{"<joint>.pos": float}`` with revolute joints in degrees
            and gripper in [0, 100].

        Raises:
            RuntimeError: If not connected (enforced by decorator).
        """
        return {f"{name}.pos": self._positions[name] for name in _JOINT_NAMES}

    @check_if_not_connected  # type: ignore[untyped-decorator]
    def send_action(self, action: RobotAction) -> RobotAction:
        """Apply a position command and update internal state.

        Args:
            action: Dict ``{"<joint>.pos": float}`` in lerobot's native units.

        Returns:
            The same action dict (lerobot convention).

        Raises:
            RuntimeError: If not connected (enforced by decorator).
        """
        for name in _JOINT_NAMES:
            key = f"{name}.pos"
            if key in action:
                self._positions[name] = float(action[key])
        return action

    @check_if_not_connected  # type: ignore[untyped-decorator]
    def disconnect(self) -> None:
        """Deactivate the twin.  Idempotent if called via the decorator."""
        self._is_connected = False
