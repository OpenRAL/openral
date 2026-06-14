"""Unit tests for ``openral deploy run`` (ADR-0032).

`deploy run` shells the production ROS graph with `hal_mode:=real` (no
in-process runner). These tests exercise the CLI's resolution + error paths via
Typer's `CliRunner` **without** running `ros2 launch` — the happy path
monkeypatches the launch runner to capture the resolved invocation; the live
launch + real `connect()` are HIL-verified on a robot host.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_cli import deploy_sim as _deploy_sim
from openral_cli.deploy_sim import LaunchInvocation
from openral_cli.main import app
from typer.testing import CliRunner


def _write_robot_env_yaml(
    tmp_path: Path,
    *,
    robot_id: str = "so100_follower",
    port: str = "/dev/ttyUSB0",
) -> Path:
    """Write a minimal, schema-valid RobotEnvironment YAML."""
    out = tmp_path / "robot_env.yaml"
    out.write_text(
        f"robot_id: {robot_id}\n"
        "hal:\n"
        f"  adapter: {robot_id}\n"
        "  transport:\n"
        f"    port: {port}\n"
        "sensors: []\n"
        "task:\n"
        "  id: deploy/zero\n"
        "  scene_id: deploy/zero\n"
        '  instruction: "deploy run smoke"\n'
        "  max_steps: 30\n"
        "vla:\n"
        "  id: gpu_passthrough\n"
        '  weights_uri: "rskills/noop"\n'
        "rate_hz: 30.0\n"
    )
    return out


def test_help_renders() -> None:
    """``deploy run --help`` exits 0 and surfaces the new flags."""
    result = CliRunner().invoke(app, ["deploy", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in result.output
    assert "--robot" in result.output
    assert "RobotEnvironment" in result.output


def test_missing_config_path_fails() -> None:
    """A non-existent --config exits non-zero with a clear error."""
    result = CliRunner().invoke(app, ["deploy", "run", "--config", "/tmp/does-not-exist.yaml"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()


def test_sim_only_robot_fails_fast(tmp_path: Path) -> None:
    """A simulation-only robot (g1, hal.real null) fails before shelling the launch."""
    config = _write_robot_env_yaml(tmp_path, robot_id="g1", port="")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config)])
    assert result.exit_code != 0
    assert "g1" in result.output


def test_real_mode_shells_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: the CLI resolves so100 in real mode and invokes the launch runner.

    The launch runner is monkeypatched (no `ros2 launch`); we assert it received
    a real-mode invocation that opens the serial bus (no sim digital twin).
    """
    captured: dict[str, LaunchInvocation] = {}

    def _fake_run(invocation: LaunchInvocation, *, run_preflight: bool = True) -> int:
        captured["inv"] = invocation
        return 0

    monkeypatch.setattr(_deploy_sim, "run_launch_invocation", _fake_run)

    config = _write_robot_env_yaml(tmp_path, robot_id="so100_follower")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config)])

    assert result.exit_code == 0, result.output
    inv = captured["inv"]
    assert inv.robot_id == "so100_follower"
    # Real mode: no sim digital-twin injection — the so100 node opens serial.
    assert "sim_robot_yaml" not in inv.hal_params
