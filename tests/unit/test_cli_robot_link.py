"""`openral robot link <code>` and the tunnel the dashboard runs (ADR-0129 §3, Interface B).

`cloudflared`, `systemctl` and `loginctl` are real executables on PATH that log their
argv (process-boundary doubles, CLAUDE.md §1.11); no pinned download applies
(`tunnel.CLOUDFLARED` emptied); the enrolment and the user unit go to temp paths via
`OPENRAL_ROBOT_ENROLMENT` and `XDG_CONFIG_HOME`.
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
    from openral_observability.dashboard import tunnel

    monkeypatch.setattr(tunnel, "CLOUDFLARED", {})  # never download in a unit test
    path = tmp_path / "config" / "robot.json"
    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return path


def _fake_cloudflared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, systemd: bool = True, active: bool = False
) -> Path:
    """Fake cloudflared (logs argv + TUNNEL_TOKEN) and, with ``systemd``, systemctl/loginctl."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "cloudflared_call.json"
    calls = tmp_path / "systemd_calls.txt"
    tools = {
        "cloudflared": f"open({str(log)!r}, 'w').write(json.dumps("
        "{'argv': sys.argv[1:], 'token': os.environ.get('TUNNEL_TOKEN')}))",
    }
    if systemd:
        log_call = f"open({str(calls)!r}, 'a').write(' '.join(sys.argv) + '\\n')"
        tools["systemctl"] = (
            f"{log_call}\nsys.exit(0 if {active!r} or 'is-active' not in sys.argv else 3)"
        )
        tools["loginctl"] = log_call
    for name, body in tools.items():
        exe = bin_dir / name
        exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    path = str(bin_dir) if not systemd else f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    if not systemd:  # keep the real systemctl out of reach, but python on PATH
        path = f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}"
    monkeypatch.setenv("PATH", path)
    return log


def test_link_writes_the_enrolment_and_keeps_the_token_apart(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cloudflared(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["robot", "link", _CODE])

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(enrolment.stat().st_mode) == 0o600
    stored = json.loads(enrolment.read_text())
    assert "tunnel" not in stored and _TUNNEL not in enrolment.read_text()
    assert RobotEnrolment.model_validate(stored).robot == "k3x9"
    env_file = enrolment.with_name("tunnel.env")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert env_file.read_text() == f"TUNNEL_TOKEN={_TUNNEL}\n"
    assert "sudo" not in result.output and "openral dashboard" in result.output
    unit = tmp_path / "xdg" / "systemd" / "user" / "openral-tunnel.service"
    text = unit.read_text()
    assert f"EnvironmentFile={env_file}" in text and _TUNNEL not in text
    assert "tunnel --no-autoupdate run" in text and "Restart=always" in text
    calls = [line.split()[1:] for line in (tmp_path / "systemd_calls.txt").read_text().splitlines()]
    assert ["--user", "enable", "--now", "openral-tunnel.service"] in calls
    assert ["enable-linger"] in calls
    assert "after every reboot" in result.output


def test_link_without_user_systemd_leaves_the_tunnel_to_the_dashboard(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cloudflared(tmp_path, monkeypatch, systemd=False)
    result = CliRunner().invoke(app, ["robot", "link", _CODE])
    assert result.exit_code == 0, result.output
    assert "no user systemd here" in result.output and "runs it instead" in result.output


def test_no_service_skips_the_unit(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cloudflared(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "xdg" / "systemd" / "user" / "openral-tunnel.service").exists()


def test_link_without_any_cloudflared_still_links(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    result = CliRunner().invoke(app, ["robot", "link", _CODE])
    assert result.exit_code == 0, result.output
    assert enrolment.is_file() and enrolment.with_name("tunnel.env").is_file()
    assert "retries when it starts" in result.output


def test_the_dashboard_runs_the_tunnel_with_the_token_in_its_env_only(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_observability.dashboard.tunnel import TUNNEL_PORT, start_tunnel

    log = _fake_cloudflared(tmp_path, monkeypatch)
    assert start_tunnel(TUNNEL_PORT) is None  # not linked yet: no tunnel
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE]).exit_code == 0
    proc = start_tunnel(TUNNEL_PORT)
    assert proc is not None
    assert proc.wait(timeout=30) == 0
    call = json.loads(log.read_text())
    assert call == {"argv": ["tunnel", "--no-autoupdate", "run"], "token": _TUNNEL}


def test_a_bad_code_writes_nothing(enrolment: Path) -> None:
    result = CliRunner().invoke(app, ["robot", "link", "not-a-code"])
    assert result.exit_code == 2
    assert not enrolment.exists()


def test_the_dashboard_leaves_the_tunnel_to_a_running_service(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_observability.dashboard.tunnel import TUNNEL_PORT, start_tunnel

    log = _fake_cloudflared(tmp_path, monkeypatch, active=True)
    assert CliRunner().invoke(app, ["robot", "link", _CODE]).exit_code == 0
    assert start_tunnel(TUNNEL_PORT) is None  # the service holds the tunnel
    assert not log.exists()  # no second cloudflared started
