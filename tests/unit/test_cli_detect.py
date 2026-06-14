"""Unit tests for the ``openral detect`` and ``ral skill check`` CLI commands.

Hermetic — every probe is exercised against a clean container, no
hardware required.  Larger end-to-end coverage lives in
``test_detect_probes_no_hardware.py`` / ``test_detect_assemble.py`` /
``test_detect_compatibility.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestBhDetect:
    def test_detect_no_write_prints_summary(self) -> None:
        result = runner.invoke(
            app,
            ["detect", "--no-write", "--include", "network", "--dds-timeout", "0"],
        )
        assert result.exit_code == 0, result.output
        assert "openral detect" in result.output
        # --no-write prints the assembled yaml to stdout.
        assert "name:" in result.output

    def test_detect_writes_full_robot_yaml(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        # Must be a complete RobotDescription, not the legacy stub.
        assert "name" in data
        assert "capabilities" in data
        assert "embodiment_kind" in data
        assert "safety" in data

    def test_detect_with_report_dump(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        report = tmp_path / "detection.json"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--report",
                str(report),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert report.exists()
        # Raw report is JSON.
        import json

        payload = json.loads(report.read_text())
        assert payload["schema_version"] == "0.1"


class TestBhSkillCheck:
    def test_skill_check_missing_robot_yaml_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["rskill", "check", "--robot", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_skill_check_against_assembled_yaml(self, tmp_path: Path) -> None:
        # Step 1: produce a robot.yaml via `openral detect`.
        out = tmp_path / "robot.yaml"
        runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        # Step 2: run `ral skill check` against an empty registry.
        empty_registry = tmp_path / "empty-registry.json"
        # Point --rskills-dir at a non-existent path so the default ("rskills/")
        # doesn't walk the in-tree rskills/ from the repo cwd.
        missing_rskills = tmp_path / "no-such-rskills"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", empty_registry):
            result = runner.invoke(
                app,
                [
                    "rskill",
                    "check",
                    "--robot",
                    str(out),
                    "--rskills-dir",
                    str(missing_rskills),
                    "--json",
                ],
            )
        # Empty registry → exit 0 (no incompat rows).
        assert result.exit_code == 0, result.output
        # JSON output parseable.
        import json

        payload = json.loads(result.output)
        assert payload["schema_version"] == "0.1"
        assert "rows" in payload
