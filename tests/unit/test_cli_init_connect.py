"""Unit tests for ``openral connect``.

Companion to ``tests/unit/test_cli_skill.py`` (covers ``openral rskill
install/list``) and ``tests/unit/test_doctor.py`` (covers ``openral
doctor``). ``openral detect`` (which superseded ``ral init``) is covered
by ``tests/unit/test_detect_*.py``; ``openral calibrate camera`` lives
with the rest of the sensor surface in ``tests/unit/test_sensors.py``.

The SO-100 serial port is a hardware boundary no CI box has, so the HAL
is substituted with the recording fake in
:mod:`tests.unit.fakes.fake_so100_hal` (CLAUDE.md §1.11 boundary
double). Assertions target CLI behavior (exit code, output, fake's
observable lifecycle state), never mock call bookkeeping.

Coverage
--------
- ``openral connect`` — unsupported robot type → exit 1.
- ``openral connect`` — happy path: joint summary printed, HAL left disconnected.
- ``openral connect`` — ``ROSConfigError`` from ``connect()`` → exit 1.
- ``openral connect`` — ``ROSRuntimeError`` from ``connect()`` → exit 1.
- ``openral connect`` — read failure still disconnects (finally clause).
"""

from __future__ import annotations

from unittest.mock import patch

from openral_cli.main import app
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from typer.testing import CliRunner

from tests.unit.fakes.fake_so100_hal import FakeSO100FollowerHAL

runner = CliRunner()

_HAL_IMPORT = "openral_hal.so100_follower.SO100FollowerHAL"


def test_connect_unsupported_robot_exits_1() -> None:
    result = runner.invoke(app, ["connect", "--robot", "non_existent_robot"])
    assert result.exit_code == 1
    assert "Unknown robot" in result.output


def test_connect_so100_happy_path() -> None:
    captured: list[FakeSO100FollowerHAL] = []

    def build(port: str) -> FakeSO100FollowerHAL:
        hal = FakeSO100FollowerHAL(port)
        captured.append(hal)
        return hal

    with patch(_HAL_IMPORT, new=build):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    # Real joint layout surfaces in the printed summary.
    assert "shoulder_pan=0.100 rad" in result.output
    (hal,) = captured
    assert hal.port == "/dev/null"
    assert hal.read_count == 1
    assert not hal.connected  # disconnected on the way out
    assert hal.disconnect_count == 1


def test_connect_so100_rosconfigerror_exits_1() -> None:
    def build(port: str) -> FakeSO100FollowerHAL:
        return FakeSO100FollowerHAL(port, connect_error=ROSConfigError("bad URDF"))

    with patch(_HAL_IMPORT, new=build):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_connect_so100_rosruntimeerror_exits_1() -> None:
    def build(port: str) -> FakeSO100FollowerHAL:
        return FakeSO100FollowerHAL(port, connect_error=ROSRuntimeError("transport down"))

    with patch(_HAL_IMPORT, new=build):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 1
    assert "Runtime error" in result.output


def test_connect_so100_disconnect_runs_even_after_read_failure() -> None:
    """read_state failures still trigger disconnect — finally clause is honoured."""
    captured: list[FakeSO100FollowerHAL] = []

    def build(port: str) -> FakeSO100FollowerHAL:
        hal = FakeSO100FollowerHAL(port, read_error=ROSRuntimeError("read failed"))
        captured.append(hal)
        return hal

    with patch(_HAL_IMPORT, new=build):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    # The finally branch still calls disconnect; the runtime error
    # propagates and Typer translates it into a non-zero exit.
    assert result.exit_code != 0
    (hal,) = captured
    assert not hal.connected
    assert hal.disconnect_count == 1
