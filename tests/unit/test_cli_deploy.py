"""Unit tests for ``openral deploy run``."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_cli import deploy_sim as _deploy_sim
from openral_cli.deploy_sim import LaunchInvocation
from openral_cli.main import app
from typer.testing import CliRunner


def _write_deploy_scene_yaml(tmp_path: Path, *, robot_id: str = "so100_follower") -> Path:
    """Write a minimal, schema-valid DeployScene YAML."""
    out = tmp_path / "deploy_scene.yaml"
    out.write_text(
        f"scene:\n  id: deploy/zero\n  backend: mujoco\nrobot_id: {robot_id}\n",
        encoding="utf-8",
    )
    return out


def test_help_renders() -> None:
    result = CliRunner().invoke(app, ["deploy", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in result.output
    assert "--robot" in result.output
    assert "--dry-run" in result.output
    assert "DeployScene" in result.output


def test_missing_config_path_fails() -> None:
    result = CliRunner().invoke(app, ["deploy", "run", "--config", "/tmp/does-not-exist.yaml"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()


def test_sim_only_robot_fails_fast(tmp_path: Path) -> None:
    config = _write_deploy_scene_yaml(tmp_path, robot_id="g1")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config)])
    assert result.exit_code != 0
    assert "g1" in result.output


def test_real_mode_shells_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, LaunchInvocation] = {}

    def _fake_run(invocation: LaunchInvocation, *, run_preflight: bool = True) -> int:
        captured["inv"] = invocation
        return 0

    monkeypatch.setattr(_deploy_sim, "run_launch_invocation", _fake_run)

    config = _write_deploy_scene_yaml(tmp_path, robot_id="so100_follower")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config)])

    assert result.exit_code == 0, result.output
    inv = captured["inv"]
    assert inv.robot_id == "so100_follower"
    assert "sim_robot_yaml" not in inv.hal_params


def test_real_mode_dry_run_prints_launch_without_shelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_run(invocation: LaunchInvocation, *, run_preflight: bool = True) -> int:
        raise AssertionError("dry-run must not shell ros2 launch")

    monkeypatch.setattr(_deploy_sim, "run_launch_invocation", _fail_run)

    config = _write_deploy_scene_yaml(tmp_path, robot_id="so100_follower")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "hal_mode=real" in result.output
    assert "hal_params:" in result.output
    assert "argv:" in result.output


def test_real_mode_forwards_deploy_config_for_sensor_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`deploy run` forwards its --config path as the `deploy_config` launch arg.

    The runtime node opens the deploy config's `sensors:` readers from that
    path and publishes the physical cameras onto
    /openral/cameras/<sensor_id>/image — without it a real deploy has no
    camera publisher at all (the sim HAL bridge only exists in sim mode).
    """
    captured: dict[str, LaunchInvocation] = {}

    def _fake_run(invocation: LaunchInvocation, *, run_preflight: bool = True) -> int:
        captured["inv"] = invocation
        return 0

    monkeypatch.setattr(_deploy_sim, "run_launch_invocation", _fake_run)

    config = _write_deploy_scene_yaml(tmp_path, robot_id="so101_follower")
    result = CliRunner().invoke(app, ["deploy", "run", "--config", str(config)])

    assert result.exit_code == 0, result.output
    inv = captured["inv"]
    assert f"deploy_config:={config.resolve()}" in inv.argv_template


def test_deploy_validate_flags_missing_calibration(tmp_path: Path) -> None:
    """`deploy validate` errors when a serial HAL has no calibration (the exact
    gap that fails at runtime with 'has no calibration registered')."""
    config = tmp_path / "scene.yaml"
    config.write_text(
        "scene:\n  id: so101_bench\n"
        "robot_id: so101_follower\n"
        "hal:\n  defaults:\n    port: /dev/ttyACM0\n    calibrate_on_connect: false\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["deploy", "validate", "--config", str(config)])
    assert result.exit_code != 0, result.output
    assert "calibration" in result.output.lower()


def test_deploy_validate_ready_with_committed_calibration(tmp_path: Path) -> None:
    """A scene `hal:` binding with a committed calibration passes validation
    (no --hal needed); a not-attached device is a warning, not an error."""
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "so_follower.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "scene.yaml"
    config.write_text(
        "scene:\n  id: so101_bench\n"
        "robot_id: so101_follower\n"
        "hal:\n  defaults:\n    port: /dev/ttyACM0\n    id: so_follower\n"
        "    calibration_dir: calibration\n    calibrate_on_connect: false\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["deploy", "validate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "ready" in result.output.lower()
