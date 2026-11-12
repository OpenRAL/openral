"""HIL test: a SIGKILLed SO-101 HAL process leaves no servo energized (issue #545).

Hazard-log Entry 081 mitigation (a): ``SO100FollowerHAL.connect`` arms an
out-of-process ``FeetechTorqueWatcher`` that writes ``Torque_Enable=0`` to every
servo once the HAL's PID dies. This test proves it on the real bench arm: a
child process opens the bus, arms the watcher, enables torque the HAL's way,
and is SIGKILLed; every servo must then read ``Torque_Enable == 0``.

The child does **not** run ``SO100FollowerHAL.connect()``: that also writes
lerobot's motor configuration (return delay, acceleration, PID, the gripper
current limits) and the ``Lock`` register, none of which this test is about.
It does exactly the HAL's watcher arming (``FeetechTorqueWatcher.spawn`` with
the bus's ids and baud rate) and the HAL's torque-on guard
(``_hold_pose_on_torque_enable``: per servo ``Goal_Position := Present_Position``,
read back), then writes ``Torque_Enable=1`` and nothing else. If any joint
drifts more than ``_MAX_DRIFT_TICKS`` after torque-on, the child cuts torque
and exits, and the test fails.

Torque-on makes the arm hold its pose. Gated on ``SO101_TORQUE_HIL=1`` on top
of the bench gates, so a stray ``pytest tests/hil/`` never energizes the arm.
Run it attended, with the arm resting in a stable pose and the 12 V switch in
reach. Teardown always writes torque off and verifies it.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time

import pytest

from tests.hil.test_so101_serial_live import (
    _EXPECTED_SERVO_IDS,
    _bench_calibration_file,
    _bench_port,
    _scene_hal_defaults,
    requires_live_servos,
    requires_rig,
)

# Bound on SIGKILL -> watcher exited with every servo read back torque off.
_CUT_BOUND_S = 1.0
_MAX_DRIFT_TICKS = 20

_CHILD = r"""
import json, sys, time
from pathlib import Path
import structlog
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
from lerobot.robots.so_follower import SO100Follower, SOFollowerRobotConfig
from openral_hal.feetech_torque_watcher import FeetechTorqueWatcher, cut_torque
from openral_hal.so100_follower import _hold_pose_on_torque_enable

port, ident, cal_dir, max_drift = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
robot = SO100Follower(SOFollowerRobotConfig(port=port, id=ident, calibration_dir=Path(cal_dir)))
bus = robot.bus
ids = [m.id for m in bus.motors.values()]
baud = bus.port_handler.baudrate
watcher = FeetechTorqueWatcher.spawn(port, ids, baud)  # as SO100FollowerHAL.connect
bus.connect()

def torque_on_only(motors=None, num_retry=0):  # Torque_Enable=1, no Lock write
    for m in (list(bus.motors) if motors is None else [motors]):
        bus.write("Torque_Enable", m, 1, normalize=False, num_retry=num_retry)

bus.enable_torque = torque_on_only
_hold_pose_on_torque_enable(bus)  # the HAL's Goal := Present guard
before = {m: bus.read("Present_Position", m, normalize=False) for m in bus.motors}
bus.enable_torque()
t_end = time.monotonic() + 0.5
drift = {}
while time.monotonic() < t_end:
    now = {m: bus.read("Present_Position", m, normalize=False) for m in bus.motors}
    drift = {m: max(drift.get(m, 0), abs(now[m] - before[m])) for m in bus.motors}
torque = {m: bus.read("Torque_Enable", m, normalize=False) for m in bus.motors}
goal = {m: bus.read("Goal_Position", m, normalize=False) for m in bus.motors}
report = {"before": before, "goal": goal, "drift": drift, "torque": torque, "ids": ids}
if max(drift.values()) > max_drift:
    bus.disconnect(disable_torque=False)
    report["aborted"] = cut_torque(port, ids, baud)
    watcher.stop(clean=True)
    print(json.dumps(report), flush=True)
    sys.exit(3)
print(json.dumps(report), flush=True)
time.sleep(60)
"""


def _read_until_eof(fd: int, timeout_s: float) -> bytes | None:
    out = b""
    deadline = time.monotonic() + timeout_s
    while (left := deadline - time.monotonic()) > 0:
        if not select.select([fd], [], [], left)[0]:
            break
        chunk = os.read(fd, 4096)
        if not chunk:
            return out
        out += chunk
    return None


def _torque_off_verified(port: str) -> dict[int, str]:
    from openral_hal.feetech_torque_watcher import cut_torque

    return cut_torque(port, _EXPECTED_SERVO_IDS, 1_000_000)


@requires_rig
@requires_live_servos
@pytest.mark.skipif(
    os.environ.get("SO101_TORQUE_HIL") != "1",
    reason="energizes the SO-101 (holds pose); set SO101_TORQUE_HIL=1, attended",
)
def test_sigkill_of_the_hal_process_cuts_torque_on_every_servo() -> None:  # pragma: no cover
    from scservo_sdk import PacketHandler, PortHandler

    port = _bench_port()
    child = subprocess.Popen(  # reason: fixed argv
        [
            sys.executable,
            "-c",
            _CHILD,
            port,
            str(_scene_hal_defaults()["id"]),
            str(_bench_calibration_file().parent),
            str(_MAX_DRIFT_TICKS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log: bytes | None = b""
    elapsed = float("nan")
    try:
        assert child.stdout is not None and child.stderr is not None
        line = child.stdout.readline()
        assert line, f"child exited {child.wait()}: {child.stderr.read()!r}"
        report = json.loads(line)
        print(f"\nchild report: {report}")
        assert "aborted" not in report, f"a joint drifted after torque-on: {report}"
        assert set(report["torque"].values()) == {1}, report
        t0 = time.monotonic()
        child.send_signal(signal.SIGKILL)
        # The watcher inherited the child's stderr: EOF there == watcher exited.
        log = _read_until_eof(child.stderr.fileno(), 5.0)
        elapsed = time.monotonic() - t0
        print(f"SIGKILL -> watcher exit: {elapsed * 1000:.1f} ms\n{(log or b'').decode()}")
    finally:
        child.kill()
        child.wait()
        time.sleep(0.2)  # let a still-running watcher release the port
        # Independent read-back (not the watcher's code), then the teardown cut.
        handler = PortHandler(port)
        assert handler.setBaudRate(1_000_000)
        packets = PacketHandler(0)
        after = {i: packets.read1ByteTxRx(handler, i, 40)[:2] for i in _EXPECTED_SERVO_IDS}
        handler.closePort()
        teardown = _torque_off_verified(port)
        print(f"read-back after kill: {after}; teardown: {teardown}")
        assert set(teardown.values()) == {"torque_off"}, teardown
    assert log is not None, "watcher did not exit within 5 s"
    assert b"hal.torque_watcher.torque_off" in log and b"parent_died" in log, log
    assert after == {i: (0, 0) for i in _EXPECTED_SERVO_IDS}, after
    assert elapsed < _CUT_BOUND_S, elapsed
