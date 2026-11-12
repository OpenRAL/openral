"""Serial-device fake of a Feetech STS servo bus on a pseudo-terminal.

CLAUDE.md §1.11 boundary double for the out-of-process torque watcher
(``openral_hal.feetech_torque_watcher``): that watcher opens a serial *device
path* in another process, so the double has to sit at the device boundary,
not at lerobot's Python bus object (``fake_feetech_bus.py``). A thread answers
Feetech protocol-0 ``WRITE`` (0x03) and ``READ`` (0x02) packets on the master
side of a pty, keeps real per-servo register bytes, and records every write.
"""

from __future__ import annotations

import os
import pty
import threading
import tty
from dataclasses import dataclass, field

_INST_READ = 0x02
_INST_WRITE = 0x03
TORQUE_ENABLE_ADDR = 40


def _checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


@dataclass
class FakeFeetechSerial:
    """A pty that answers like a chain of STS3215 servos.

    Args:
        servo_ids: Ids that answer. Any other id stays silent (dark servo).
        initial_torque: ``Torque_Enable`` value every servo starts with.
    """

    servo_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    initial_torque: int = 1
    writes: list[tuple[int, int, int]] = field(default_factory=list)
    registers: dict[int, bytearray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Open the pty and start the responder thread."""
        for sid in self.servo_ids:
            regs = bytearray(256)
            regs[TORQUE_ENABLE_ADDR] = self.initial_torque
            self.registers[sid] = regs
        self._master, self._slave = pty.openpty()
        tty.setraw(self._slave)  # no echo / line discipline before pyserial opens it
        self.path = os.ttyname(self._slave)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def torque(self, sid: int) -> int:
        """Current ``Torque_Enable`` byte of servo ``sid``."""
        with self._lock:
            return self.registers[sid][TORQUE_ENABLE_ADDR]

    def close(self) -> None:
        """Close both pty ends (the responder thread then exits)."""
        for fd in (self._slave, self._master):
            try:
                os.close(fd)
            except OSError:  # reason: already closed
                continue

    def _serve(self) -> None:
        buf = bytearray()
        while True:
            try:
                chunk = os.read(self._master, 256)
            except OSError:  # reason: pty closed -> stop serving
                return
            if not chunk:
                return
            buf += chunk
            while True:
                start = buf.find(b"\xff\xff")
                if start < 0 or len(buf) < start + 4:
                    break
                length = buf[start + 3]
                end = start + 4 + length
                if len(buf) < end:
                    break
                packet = bytes(buf[start:end])
                del buf[:end]
                reply = self._handle(packet)
                if reply:
                    os.write(self._master, reply)

    def _handle(self, packet: bytes) -> bytes:
        sid, inst, params = packet[2], packet[4], packet[5:-1]
        if sid not in self.servo_ids:
            return b""
        data = b""
        with self._lock:
            regs = self.registers[sid]
            if inst == _INST_WRITE:
                addr = params[0]
                regs[addr : addr + len(params) - 1] = params[1:]
                self.writes.append((sid, addr, params[1]))
            elif inst == _INST_READ:
                addr, n = params[0], params[1]
                data = bytes(regs[addr : addr + n])
            else:
                return b""
        body = bytes([sid, len(data) + 2, 0]) + data
        return b"\xff\xff" + body + bytes([_checksum(body)])
