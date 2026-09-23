"""``benchmark scene`` reports invalid ``backend_options`` as a config error, not a traceback."""

from __future__ import annotations

from pathlib import Path

import yaml
from openral_cli.main import app
from typer.testing import CliRunner

_REPO = Path(__file__).resolve().parents[2]


def test_invalid_backend_options_exit_cleanly(tmp_path: Path) -> None:
    data = yaml.safe_load((_REPO / "scenes/benchmark/robocasa_pnp.yaml").read_text())
    data["scene"]["backend_options"]["not_an_option"] = 1  # RoboCasaBackendOptions forbids extras
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(data))

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "scene",
            "--config",
            str(cfg),
            "--rskill",
            str(_REPO / "rskills/rldx1-ft-rc365-nf4"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "config:" in result.output
    assert "not_an_option" in result.output
