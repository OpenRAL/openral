"""Out-of-process Feetech torque watcher (issue #545, hazard-log Entry 081).

The watcher is a real separate process talking to a serial device; the only
double is the device itself (``FakeFeetechSerial``, a pty that answers the
Feetech protocol). The HAL process it guards is a real child that gets a real
SIGKILL.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from openral_hal import feetech_torque_watcher
from openral_hal.feetech_torque_watcher import FeetechTorqueWatcher
from openral_hal.so100_follower import SO100FollowerHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig

from tests.unit.fakes.fake_feetech_serial import TORQUE_ENABLE_ADDR, FakeFeetechSerial

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="pty + pidfd are Linux-only")

_IDS = (1, 2, 3, 4, 5, 6)
_BAUD = 1_000_000
# Generous CI bound; the measured figure is in the HAL README.
_CUT_BOUND_S = 2.0

# A stand-in HAL process: arms the watcher exactly as SO100FollowerHAL.connect
# does, reports the watcher pid, then idles until killed. The module is loaded
# by path so the child skips the multi-second ``openral_hal`` package import.
_HAL_CHILD = """
import importlib.util, sys, time
import structlog
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
spec = importlib.util.spec_from_file_location("watcher", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
w = mod.FeetechTorqueWatcher.spawn(sys.argv[2], [int(i) for i in sys.argv[3].split(",")], 1000000)
print(w.proc.pid, flush=True)
time.sleep(60)
"""


@pytest.fixture
def bus() -> Iterator[FakeFeetechSerial]:
    fake = FakeFeetechSerial(servo_ids=_IDS, initial_torque=1)
    yield fake
    fake.close()


def _is_running(pid: int) -> bool:
    """False once ``pid`` is gone or a zombie awaiting its (re)parent's reap."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):  # reason: reaped mid-read
        return False


def _read_until_eof(stream: object, timeout_s: float) -> bytes:
    fd = stream.fileno()  # type: ignore[attr-defined]  # reason: any pipe file
    out = b""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], deadline - time.monotonic())
        if not ready:
            break
        chunk = os.read(fd, 4096)
        if not chunk:
            return out
        out += chunk
    raise AssertionError(f"watcher still running after {timeout_s} s; output so far: {out!r}")


def test_sigkill_of_the_hal_process_cuts_torque_on_every_servo(bus: FakeFeetechSerial) -> None:
    hal = subprocess.Popen(  # reason: fixed argv
        [
            sys.executable,
            "-c",
            _HAL_CHILD,
            feetech_torque_watcher.__file__,
            bus.path,
            ",".join(map(str, _IDS)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert hal.stdout is not None and hal.stderr is not None
        watcher_pid = int(hal.stdout.readline())
        assert all(bus.torque(i) == 1 for i in _IDS)
        t0 = time.monotonic()
        hal.send_signal(signal.SIGKILL)
        # The watcher inherited the child's stderr: EOF there == watcher exited.
        log = _read_until_eof(hal.stderr, _CUT_BOUND_S)
        elapsed = time.monotonic() - t0
    finally:
        hal.kill()
        hal.wait()
    assert [bus.torque(i) for i in _IDS] == [0] * len(_IDS), log
    assert b"hal.torque_watcher.torque_off" in log and b"parent_died" in log, log
    assert elapsed < _CUT_BOUND_S
    assert not _is_running(watcher_pid)
    # Only Torque_Enable=0 ever reached the bus.
    assert {(addr, value) for _, addr, value in bus.writes} == {(TORQUE_ENABLE_ADDR, 0)}


def test_clean_stop_exits_without_touching_the_bus(bus: FakeFeetechSerial) -> None:
    watcher = FeetechTorqueWatcher.spawn(bus.path, _IDS, _BAUD)
    assert watcher.stop(clean=True) == 0
    assert watcher.proc.poll() == 0
    assert bus.writes == []
    assert all(bus.torque(i) == 1 for i in _IDS)


def test_a_dropped_pipe_without_the_stop_byte_fails_closed(bus: FakeFeetechSerial) -> None:
    watcher = FeetechTorqueWatcher.spawn(bus.path, _IDS, _BAUD)
    assert watcher.stop(clean=False) == 0
    assert [bus.torque(i) for i in _IDS] == [0] * len(_IDS)


def test_a_dark_servo_makes_the_watcher_exit_nonzero(bus: FakeFeetechSerial) -> None:
    watcher = FeetechTorqueWatcher.spawn(bus.path, (*_IDS, 7), _BAUD)
    assert watcher.stop(clean=False) == 1
    # Every answering servo is still cut despite the dark one.
    assert [bus.torque(i) for i in _IDS] == [0] * len(_IDS)


def test_cut_torque_refuses_a_missing_port() -> None:
    with pytest.raises(OSError, match="cannot open"):
        feetech_torque_watcher.cut_torque("/dev/openral-no-such-port", _IDS, _BAUD)


def test_hal_disconnect_retires_its_watcher_cleanly(bus: FakeFeetechSerial) -> None:
    hal = SO100FollowerHAL(robot=SO100DigitalTwin(SO100DigitalTwinConfig()))
    hal.connect()
    # The twin has no serial bus, so attach a real watcher as connect() does on hardware.
    watcher = FeetechTorqueWatcher.spawn(bus.path, _IDS, _BAUD)
    hal._torque_watcher = watcher
    hal.disconnect()
    assert hal._torque_watcher is None
    assert watcher.proc.poll() == 0  # exited, reaped: no watcher left running
    assert bus.writes == []
