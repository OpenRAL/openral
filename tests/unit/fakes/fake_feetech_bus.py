"""Serial-boundary fake of lerobot's ``FeetechMotorsBus`` for the SO-100/SO-101 e-stop.

CLAUDE.md §1.11 boundary double: the SO-100/SO-101 follower's servos sit
behind a USB-serial Feetech bus no CI box has. This fake stands in for the
bus object lerobot's ``SOFollower`` exposes as ``robot.bus`` — the register
I/O surface ``SO100FollowerHAL.estop`` drives (``write`` / ``read`` of
``Torque_Enable``, ``port_handler`` release, ``ping`` at connect). It keeps
real per-servo register state and records every transaction, so tests assert
on what reached each servo rather than on mock bookkeeping.

Failure injection mirrors what the real bus raises (lerobot 0.6.1
``MotorsBus._write`` / ``_read``): ``ConnectionError`` when a servo sends no
status packet (dark servo, unplugged cable), and a register that silently keeps
its value (``stuck_torque``) for the read-back check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The SO-100/SO-101 follower's servo names, in lerobot's bus order.
SO_FOLLOWER_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class FakePortHandler:
    """The two ``scservo_sdk.PortHandler`` members lerobot's stop path touches."""

    is_using: bool = False
    cleared: int = 0

    def clearPort(self) -> None:  # noqa: N802  # reason: mirrors scservo_sdk's camelCase API
        """Flush the serial buffers (counted)."""
        self.cleared += 1


@dataclass
class FakeFeetechBus:
    """Per-servo ``Torque_Enable`` register state behind a fake serial port.

    Args:
        dark: Servos that answer no packet: every ``write``/``read`` raises
            ``ConnectionError``, as lerobot does after its retries.
        stuck_torque: Servos that acknowledge the ``Torque_Enable`` write but
            keep reading back ``1`` (a write that did not take).
        unreadable: Servos whose write is acknowledged but whose read-back
            raises ``ConnectionError``.
    """

    dark: frozenset[str] = frozenset()
    stuck_torque: frozenset[str] = frozenset()
    unreadable: frozenset[str] = frozenset()
    port_handler: FakePortHandler = field(default_factory=FakePortHandler)
    torque: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SO_FOLLOWER_MOTORS, 1))
    writes: list[tuple[str, str, int]] = field(default_factory=list)
    reads: list[tuple[str, str]] = field(default_factory=list)

    @property
    def motors(self) -> dict[str, int]:
        """Motor name → id, the shape lerobot's ``MotorsBus.motors`` keys by."""
        return {name: i + 1 for i, name in enumerate(SO_FOLLOWER_MOTORS)}

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        """Return the STS3215 model number, or ``None`` for a dark servo."""
        del num_retry
        return None if motor in self.dark else 777

    def write(
        self,
        data_name: str,
        motor: str,
        value: int,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write one register; raises ``ConnectionError`` for a dark servo."""
        del normalize, num_retry
        if self.port_handler.is_using:
            raise ConnectionError(f"[TxRxResult] Port is in use! ({motor})")
        if motor in self.dark:
            raise ConnectionError(f"[TxRxResult] There is no status packet! ({motor})")
        self.writes.append((data_name, motor, int(value)))
        if data_name == "Torque_Enable" and motor not in self.stuck_torque:
            self.torque[motor] = int(value)

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> int:
        """Read one register; raises ``ConnectionError`` for a dark/unreadable servo."""
        del normalize, num_retry
        if motor in self.dark or motor in self.unreadable:
            raise ConnectionError(f"[TxRxResult] There is no status packet! ({motor})")
        self.reads.append((data_name, motor))
        if data_name != "Torque_Enable":
            raise KeyError(data_name)
        return self.torque[motor]
