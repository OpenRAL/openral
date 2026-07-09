"""Serial-boundary fake of ``SO100FollowerHAL`` for ``openral connect`` tests.

CLAUDE.md §1.11 boundary double — the SO-100 follower speaks Feetech
serial over a USB port, a hardware/process boundary no CI box has.
This is a real recording fake (not a ``MagicMock``): it implements the
subset of the :class:`openral_hal.HAL` Protocol that
``openral_cli.main._connect_so100`` actually drives
(``connect`` / ``read_state`` / ``disconnect``) and records the
lifecycle so tests assert on observable state, not on mock call
bookkeeping.

Failure injection mirrors the real adapter's contract: ``connect``
raises ``ROSConfigError`` / ``ROSRuntimeError`` and ``read_state``
raises ``ROSRuntimeError`` when the transport drops mid-read.
"""

from __future__ import annotations

from openral_core.schemas import JointState

# The real SO-100 follower joint layout (robots/so100/robot.yaml — the
# 5-DOF arm + gripper the Feetech bus exposes).
SO100_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class FakeSO100FollowerHAL:
    """Recording stand-in for ``openral_hal.so100_follower.SO100FollowerHAL``."""

    embodiment_tag = "so100_follower"

    def __init__(
        self,
        port: str,
        *,
        connect_error: Exception | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.port = port
        self._connect_error = connect_error
        self._read_error = read_error
        self.connected = False
        self.disconnect_count = 0
        self.read_count = 0

    def connect(self) -> None:
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    def read_state(self) -> JointState:
        self.read_count += 1
        if self._read_error is not None:
            raise self._read_error
        return JointState(
            name=SO100_JOINT_NAMES,
            position=[0.1, 0.2, 0.3, 0.0, 0.0, 0.5],
            velocity=[0.0] * 6,
            effort=[0.0] * 6,
            stamp_ns=0,
        )

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1
