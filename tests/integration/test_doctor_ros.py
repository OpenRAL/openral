"""Real ROS 2 integration test for openral doctor ROS 2 checks.

Requires a sourced ROS 2 installation (``ROS_DISTRO`` env var set).
Skipped automatically in pure-Python CI.

Verifies that ``_check_ros2`` returns real ``ok`` status rows when
ROS 2 is actually installed and sourced, not just mocked.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — needs real ROS 2")
def test_check_ros2_returns_ok_when_installed() -> None:  # pragma: no cover
    """_check_ros2 returns ok for binary + distro when ROS 2 is sourced on host."""
    from openral_cli.main import _check_ros2

    results = _check_ros2()
    by_check = {r.check: r for r in results}

    assert "ROS 2 binary" in by_check, f"Missing 'ROS 2 binary' row; got {list(by_check)}"
    assert by_check["ROS 2 binary"].status == "ok", (
        f"Expected ok but got {by_check['ROS 2 binary'].status}: {by_check['ROS 2 binary'].details}"
    )

    assert "ROS 2 distro" in by_check
    assert by_check["ROS 2 distro"].status == "ok", (
        f"ROS_DISTRO is set but distro check returned {by_check['ROS 2 distro'].status}. "
        "Is ROS 2 actually installed and ros2 on PATH?"
    )
    assert by_check["ROS 2 distro"].details == os.environ["ROS_DISTRO"]


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — needs real ROS 2")
def test_doctor_command_exits_0_with_ros2_installed() -> None:  # pragma: no cover
    """``openral doctor`` exits 0 when ROS 2 is installed and Python ≥ 3.10."""
    import sys

    from openral_cli.main import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, (
        f"openral doctor exited {result.exit_code} with ROS 2 installed.\nOutput:\n{result.output}"
    )
    assert sys.version_info >= (3, 10)
    assert "ok" in result.output
