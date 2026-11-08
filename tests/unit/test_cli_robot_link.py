"""`openral robot link <code>` (ADR-0129 Interface B).

`cloudflared` is a real executable on PATH that logs its argv (a process-boundary
double, CLAUDE.md §1.11); the enrolment goes to a temp path via
`OPENRAL_ROBOT_ENROLMENT`.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from openral_cli.main import app
from openral_core import RobotEnrolment
from typer.testing import CliRunner

_TUNNEL = "eyJhIjoidHVubmVsLXNlY3JldCJ9"
_LINK: dict[str, object] = {
    "v": 1,
    "robot": "k3x9",
    "aud": "https://r-k3x9.openral.com",
    "org": "",
    "iss": "https://www.openral.com/api/auth",
    "jwks": "https://www.openral.com/api/auth/jwks",
    "tunnel": _TUNNEL,
}
_CODE = base64.urlsafe_b64encode(json.dumps(_LINK).encode()).decode().rstrip("=")


@pytest.fixture
def enrolment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config" / "robot.json"
    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(path))
    return path


def test_link_writes_the_enrolment_without_the_tunnel_token_and_installs_it(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_log = tmp_path / "cloudflared_argv.json"
    fake = tmp_path / "bin" / "cloudflared"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")

    result = CliRunner().invoke(app, ["robot", "link", _CODE])

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(enrolment.stat().st_mode) == 0o600
    stored = json.loads(enrolment.read_text())
    assert "tunnel" not in stored and _TUNNEL not in enrolment.read_text()
    assert RobotEnrolment.model_validate(stored).robot == "k3x9"
    assert json.loads(argv_log.read_text()) == ["service", "install", _TUNNEL]


def test_link_without_cloudflared_prints_how_to_install_it(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    result = CliRunner().invoke(app, ["robot", "link", _CODE])
    assert result.exit_code == 0, result.output
    assert enrolment.is_file()
    assert f"sudo cloudflared service install {_TUNNEL}" in result.output


def test_a_bad_code_writes_nothing(enrolment: Path) -> None:
    result = CliRunner().invoke(app, ["robot", "link", "not-a-code"])
    assert result.exit_code == 2
    assert not enrolment.exists()
