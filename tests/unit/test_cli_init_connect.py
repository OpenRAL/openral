"""Unit tests for ``openral connect``, ``openral calibrate camera``.

Companion to ``tests/unit/test_cli_skill.py`` (covers ``ral skill install/list``)
and ``tests/unit/test_doctor.py`` (covers ``openral doctor``).
``openral detect`` (which superseded ``ral init``) is covered by
``tests/unit/test_detect_*.py``.

All ROS 2 / hardware / HF I/O is mocked so this file runs in <2 s on a
vanilla CI box and respects CLAUDE.md §5.4.

Coverage
--------
- ``openral connect``        — unsupported robot type → exit 1.
- ``openral connect``        — happy path against a mocked ``SO100FollowerHAL``.
- ``openral connect``        — ``ROSConfigError`` from ``connect()`` → exit 1.
- ``openral connect``        — ``ROSRuntimeError`` from ``connect()`` → exit 1.
- ``openral calibrate camera`` — invalid ``--chessboard-size`` → exit 1.
- ``openral calibrate camera`` — ``--dry-run`` prints the command and exits 0.
- ``openral calibrate camera`` — ros2 binary missing → exit 1.
- ``openral calibrate camera`` — happy path runs the subprocess and propagates rc.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openral_cli.main import app
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import JointState
from typer.testing import CliRunner

runner = CliRunner()


def test_connect_unsupported_robot_exits_1() -> None:
    result = runner.invoke(app, ["connect", "--robot", "non_existent_robot"])
    assert result.exit_code == 1
    assert "Unknown robot" in result.output


def test_connect_so100_happy_path() -> None:
    fake_state = JointState(
        name=["shoulder_pan", "shoulder_lift", "elbow_flex"],
        position=[0.1, 0.2, 0.3],
        velocity=[0.0, 0.0, 0.0],
        effort=[0.0, 0.0, 0.0],
        stamp_ns=0,
    )
    fake_hal = MagicMock()
    fake_hal.read_state.return_value = fake_state

    with patch("openral_hal.so100_follower.SO100FollowerHAL", return_value=fake_hal):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    fake_hal.connect.assert_called_once()
    fake_hal.read_state.assert_called_once()
    fake_hal.disconnect.assert_called_once()


def test_connect_so101_happy_path() -> None:
    """SO-101 routes through the shared SO100FollowerHAL (same controller)."""
    fake_state = JointState(
        name=["shoulder_pan", "shoulder_lift", "elbow_flex"],
        position=[0.1, 0.2, 0.3],
        velocity=[0.0, 0.0, 0.0],
        effort=[0.0, 0.0, 0.0],
        stamp_ns=0,
    )
    fake_hal = MagicMock()
    fake_hal.read_state.return_value = fake_state

    with patch("openral_hal.so100_follower.SO100FollowerHAL", return_value=fake_hal):
        result = runner.invoke(app, ["connect", "--robot", "so101", "--port", "/dev/null"])

    assert result.exit_code == 0, result.output
    assert "SO-101" in result.output
    assert "Connected" in result.output
    fake_hal.connect.assert_called_once()
    fake_hal.read_state.assert_called_once()
    fake_hal.disconnect.assert_called_once()


def test_connect_so100_rosconfigerror_exits_1() -> None:
    fake_hal = MagicMock()
    fake_hal.connect.side_effect = ROSConfigError("bad URDF")

    with patch("openral_hal.so100_follower.SO100FollowerHAL", return_value=fake_hal):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_connect_so100_rosruntimeerror_exits_1() -> None:
    fake_hal = MagicMock()
    fake_hal.connect.side_effect = ROSRuntimeError("transport down")

    with patch("openral_hal.so100_follower.SO100FollowerHAL", return_value=fake_hal):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    assert result.exit_code == 1
    assert "Runtime error" in result.output


def test_connect_so100_disconnect_runs_even_after_read_failure() -> None:
    """read_state failures still trigger disconnect — finally clause is honoured."""
    fake_hal = MagicMock()
    fake_hal.read_state.side_effect = ROSRuntimeError("read failed")

    with patch("openral_hal.so100_follower.SO100FollowerHAL", return_value=fake_hal):
        result = runner.invoke(app, ["connect", "--robot", "so100", "--port", "/dev/null"])

    # The finally branch still calls disconnect; the runtime error
    # propagates and Typer translates it into a non-zero exit.
    assert result.exit_code != 0
    fake_hal.disconnect.assert_called_once()


# ── openral calibrate camera ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_size", ["bad", "8x", "x6", "8X6X3", "abc x def"])
def test_calibrate_camera_invalid_chessboard_size_exits_1(bad_size: str) -> None:
    result = runner.invoke(
        app,
        ["calibrate", "camera", "--sensor", "head_color", "--chessboard-size", bad_size],
    )
    assert result.exit_code == 1
    assert "Invalid --chessboard-size" in result.output


def test_calibrate_camera_dry_run_prints_command_and_exits_0() -> None:
    result = runner.invoke(
        app,
        [
            "calibrate",
            "camera",
            "--sensor",
            "head_color",
            "--chessboard-size",
            "8x6",
            "--square-size",
            "0.025",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Topic derivation: image: /head_color/image_raw, info: /head_color/camera_info
    assert "/head_color/image_raw" in result.output
    assert "/head_color/camera_info" in result.output
    assert "ros2 run camera_calibration cameracalibrator" in result.output


def test_calibrate_camera_uses_explicit_topic_override() -> None:
    result = runner.invoke(
        app,
        [
            "calibrate",
            "camera",
            "--sensor",
            "head_color",
            "--topic",
            "/custom/image_raw",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "/custom/image_raw" in result.output
    assert "/custom/camera_info" in result.output


def test_calibrate_camera_missing_ros2_binary_exits_1() -> None:
    with patch("openral_cli.main.shutil.which", return_value=None):
        result = runner.invoke(
            app,
            ["calibrate", "camera", "--sensor", "head_color", "--chessboard-size", "8x6"],
        )
    assert result.exit_code == 1
    assert "ros2 not found" in result.output


def test_calibrate_camera_propagates_subprocess_returncode() -> None:
    fake_completed = MagicMock(returncode=42)
    with (
        patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
        patch("openral_cli.main.subprocess.run", return_value=fake_completed) as run_mock,
    ):
        result = runner.invoke(
            app,
            ["calibrate", "camera", "--sensor", "head_color", "--chessboard-size", "8x6"],
        )
    assert result.exit_code == 42
    assert run_mock.call_count == 1
    cmd = run_mock.call_args.args[0]
    # Command is built deterministically; spot-check key fragments.
    assert cmd[:4] == ["ros2", "run", "camera_calibration", "cameracalibrator"]
    assert "image:=/head_color/image_raw" in cmd


def test_calibrate_camera_runs_successfully_when_subprocess_returns_zero() -> None:
    fake_completed = MagicMock(returncode=0)
    with (
        patch("openral_cli.main.shutil.which", return_value="/usr/bin/ros2"),
        patch("openral_cli.main.subprocess.run", return_value=fake_completed),
    ):
        result = runner.invoke(
            app,
            ["calibrate", "camera", "--sensor", "head_color", "--chessboard-size", "8x6"],
        )
    assert result.exit_code == 0
