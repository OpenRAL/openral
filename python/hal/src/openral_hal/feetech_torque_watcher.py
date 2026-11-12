"""Out-of-process torque-off watcher for a Feetech (SO-100/SO-101) servo bus.

A SIGKILL or OOM kill of the HAL process skips every Python ``finally`` and
``atexit``, so the torque-off writes in ``SO100FollowerHAL.disconnect`` /
``estop`` never run and the STS3215 servos keep holding torque (issue #545,
hazard-log Entry 081; CLAUDE.md §3: no path where a Python crash leaves motors
energized). The servos have no communication watchdog of their own (see the
HAL README), so the cut has to come from another process.

``FeetechTorqueWatcher.spawn`` starts this file as a separate, lean Python
process (stdlib + ``scservo_sdk`` + ``structlog``; never the 5 s
``openral_hal`` import) in its own session, so a terminal Ctrl-C or a
process-group kill aimed at the HAL does not take it down too. It waits on
two signals:

- the HAL's PID dying (``pidfd`` on Linux; a 0.1 s ``getppid()`` poll
  elsewhere), and
- its stdin pipe from the HAL: the byte ``q`` is a clean stop (the HAL has
  already cut torque itself) and the watcher exits **without** touching the
  bus; EOF without ``q`` means the HAL dropped the pipe without saying so,
  which is treated like death (fail closed).

On death it opens the serial port (the kernel closed the dead HAL's fd), writes
``Torque_Enable=0`` to every servo id and reads each back, logs the per-servo
result and exits ``0`` only when every servo reads back torque off. The retry
budget is explicit (``_OPEN_RETRIES`` port opens, ``_TORQUE_RETRIES`` tries per
servo) and logged (CLAUDE.md §1.4). The watcher never writes anything but
``Torque_Enable=0``.

Known ceiling: a kill of the whole cgroup (``systemd stop`` with
``KillMode=control-group``, a container stop) takes the watcher down with the
HAL. That path is the hardware E-stop's, not this watcher's.

Example:
    >>> from openral_hal.feetech_torque_watcher import FeetechTorqueWatcher
    >>> FeetechTorqueWatcher.__name__
    'FeetechTorqueWatcher'
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

import structlog

__all__ = ["FeetechTorqueWatcher", "cut_torque", "main"]

log = structlog.get_logger(__name__)

# Feetech STS/SCS ``Torque_Enable`` register (same address on STS and SCS
# tables in lerobot's ``feetech/tables.py``) and the protocol-0 packet handler
# the STS3215 speaks.
_TORQUE_ENABLE_ADDR = 40
_TORQUE_OFF = 0
_PROTOCOL_VERSION = 0
# lerobot's own ``MotorsBus.disconnect`` uses 5 retries for its torque-off
# writes; the watcher matches it per servo.
_TORQUE_RETRIES = 5
# The port may still be closing when the death signal fires; try a few opens.
_OPEN_RETRIES = 10
_OPEN_RETRY_S = 0.02
_PARENT_POLL_S = 0.1
# Only delays the cut when the HAL dropped the pipe while still alive.
_EOF_NAME_S = 0.2
_STOP_BYTE = b"q"
_ARMED_LINE = b"armed\n"


def cut_torque(port: str, servo_ids: Sequence[int], baudrate: int) -> dict[int, str]:
    """Write ``Torque_Enable=0`` to every servo on ``port`` and read it back.

    Every servo is attempted even if an earlier one fails. Returns
    ``{servo_id: "torque_off" | "torque_on" | "torque_unknown: <why>"}``.

    Raises:
        OSError: If the port cannot be opened after ``_OPEN_RETRIES`` tries.
    """
    from scservo_sdk import PacketHandler, PortHandler  # noqa: PLC0415

    handler = PortHandler(port)
    last: Exception | None = None
    for _ in range(_OPEN_RETRIES):
        try:
            if handler.setBaudRate(baudrate):  # opens the port at this rate
                break
        except Exception as exc:  # reason: pyserial raises SerialException/OSError
            last = exc
        time.sleep(_OPEN_RETRY_S)
    else:
        raise OSError(f"cannot open {port!r} after {_OPEN_RETRIES} tries: {last}")
    packets = PacketHandler(_PROTOCOL_VERSION)
    result: dict[int, str] = {}
    try:
        for sid in servo_ids:
            state = "torque_unknown: no status packet"
            for _ in range(_TORQUE_RETRIES):
                comm, err = packets.write1ByteTxRx(handler, sid, _TORQUE_ENABLE_ADDR, _TORQUE_OFF)
                if comm != 0:
                    state = f"torque_unknown: write comm={comm} err={err}"
                    continue
                value, comm, err = packets.read1ByteTxRx(handler, sid, _TORQUE_ENABLE_ADDR)
                if comm != 0:
                    state = f"torque_unknown: read comm={comm} err={err}"
                    continue
                state = "torque_off" if value == _TORQUE_OFF else "torque_on"
                if state == "torque_off":
                    break
            result[sid] = state
    finally:
        handler.closePort()
    return result


def _wait_for_trigger(parent_pid: int) -> str:
    """Block until the HAL dies, drops the pipe, or sends the clean-stop byte."""
    stdin = sys.stdin.buffer.fileno()
    pidfd: int | None = None
    with contextlib.suppress(AttributeError, OSError):
        pidfd = os.pidfd_open(parent_pid)
    # A parent that died before the pidfd opened is already reparented.
    if os.getppid() != parent_pid:
        return "parent_died"
    watched = [stdin] if pidfd is None else [stdin, pidfd]
    while True:
        ready, _, _ = select.select(watched, [], [], None if pidfd is not None else _PARENT_POLL_S)
        if pidfd is not None and pidfd in ready:
            return "parent_died"
        if stdin in ready:
            data = os.read(stdin, 1)
            if data == _STOP_BYTE:
                return "clean_stop"
            if not data:
                # The kernel closes a dying HAL's pipe end a moment before its
                # pidfd turns readable; give the pidfd that moment to name it.
                if pidfd is not None and select.select([pidfd], [], [], _EOF_NAME_S)[0]:
                    return "parent_died"
                return "pipe_closed"
        if pidfd is None and os.getppid() != parent_pid:
            return "parent_died"


def main(argv: Sequence[str] | None = None) -> int:
    """Watcher process entry point (spawned by ``FeetechTorqueWatcher.spawn``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, required=True)
    parser.add_argument("--ids", required=True, help="comma-separated servo ids")
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    # Own process: route its log lines to the HAL's inherited stderr (stdout
    # carries only the armed handshake and is closed after it).
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    ids = [int(i) for i in args.ids.split(",")]
    # Only the HAL decides when the watcher goes: stray terminal signals must
    # not disarm it (it already runs in its own session).
    for sig in (signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    import scservo_sdk  # noqa: F401, PLC0415  # reason: pay the import before arming

    os.write(sys.stdout.buffer.fileno(), _ARMED_LINE)
    os.close(sys.stdout.buffer.fileno())
    trigger = _wait_for_trigger(args.parent_pid)
    if trigger == "clean_stop":
        log.info("hal.torque_watcher.clean_exit", port=args.port, parent_pid=args.parent_pid)
        return 0
    t0 = time.monotonic()
    log.critical(
        "hal.torque_watcher.triggered",
        trigger=trigger,
        port=args.port,
        servo_ids=ids,
        parent_pid=args.parent_pid,
    )
    try:
        states = cut_torque(args.port, ids, args.baudrate)
    except OSError as exc:
        log.critical("hal.torque_watcher.port_unavailable", port=args.port, error=str(exc))
        return 2
    ok = all(s == "torque_off" for s in states.values())
    log.critical(
        "hal.torque_watcher.torque_off" if ok else "hal.torque_watcher.torque_off_unproven",
        port=args.port,
        states={str(k): v for k, v in states.items()},
        elapsed_ms=round((time.monotonic() - t0) * 1000.0, 1),
    )
    return 0 if ok else 1


class FeetechTorqueWatcher:
    """Handle on a running watcher process; see the module docstring.

    Args:
        proc: The spawned watcher (use ``spawn``).
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        """Wrap an already-armed watcher process."""
        self.proc = proc

    @classmethod
    def spawn(
        cls,
        port: str,
        servo_ids: Sequence[int],
        baudrate: int,
        *,
        armed_timeout_s: float = 10.0,
    ) -> FeetechTorqueWatcher:
        """Start the watcher for this process and wait until it is armed.

        Raises:
            RuntimeError: If the watcher exits or does not report armed within
                ``armed_timeout_s`` (the caller must then refuse to energise).
        """
        cmd = [
            sys.executable,
            "-P",  # no script dir on sys.path: openral_hal's modules never shadow
            os.path.abspath(__file__),
            f"--port={port}",
            f"--baudrate={baudrate}",
            "--ids=" + ",".join(str(i) for i in servo_ids),
            f"--parent-pid={os.getpid()}",
        ]
        proc = subprocess.Popen(  # reason: fixed argv, no shell
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True
        )
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], armed_timeout_s)
        line = proc.stdout.readline() if ready else b""
        proc.stdout.close()
        if line != _ARMED_LINE:
            proc.kill()
            proc.wait()
            raise RuntimeError(
                f"Feetech torque watcher for {port!r} did not arm within {armed_timeout_s} s "
                f"(exit code {proc.returncode})"
            )
        log.info("hal.torque_watcher.armed", port=port, servo_ids=list(servo_ids), pid=proc.pid)
        return cls(proc)

    @property
    def alive(self) -> bool:
        """``True`` while the watcher process has not exited (a non-blocking ``poll``)."""
        return self.proc.poll() is None

    def stop(self, *, clean: bool, timeout_s: float = 5.0) -> int | None:
        """Stop the watcher; returns its exit code (``None`` if it had to be killed).

        Args:
            clean: ``True`` once the caller has proven torque off itself: the
                watcher exits without touching the bus. ``False`` closes the
                pipe without the stop byte, so the watcher runs its own
                torque-off pass (fail closed) — the caller must have released
                the port first.
            timeout_s: How long to wait for the watcher to exit.
        """
        assert self.proc.stdin is not None
        with contextlib.suppress(BrokenPipeError, ValueError):
            if clean:
                self.proc.stdin.write(_STOP_BYTE)
                self.proc.stdin.flush()
            self.proc.stdin.close()
        try:
            code: int | None = self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.error("hal.torque_watcher.stop_timeout", pid=self.proc.pid, clean=clean)
            self.proc.kill()
            self.proc.wait()
            code = None
        log.info("hal.torque_watcher.stopped", pid=self.proc.pid, clean=clean, exit_code=code)
        return code


if __name__ == "__main__":
    sys.exit(main())
