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
from typing import Any

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
    "serial": "SO101-0007",
    "secret": "ors_r0b0tsecret",
    "tunnel": _TUNNEL,
}
_CODE = base64.urlsafe_b64encode(json.dumps(_LINK).encode()).decode().rstrip("=")


@pytest.fixture
def enrolment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from openral_cli.robot import ACCOUNT_SETTINGS
    from openral_observability.dashboard import tunnel

    monkeypatch.setattr(tunnel, "CLOUDFLARED", {})  # never download in a unit test
    # apply_account_settings writes os.environ itself: record each variable's original
    # state first (setenv, then delenv), so teardown restores it and nothing leaks.
    for var in ACCOUNT_SETTINGS:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
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
    assert "ors_" not in enrolment.read_text()  # the robot secret lives in its own file
    secret = enrolment.with_name("robot.secret")
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert secret.read_text() == "ors_r0b0tsecret\n"
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
    assert ["--user", "enable", "openral-tunnel.service"] in calls
    # restart, so a robot linked again (a new tunnel) drops the old token at once
    assert ["--user", "restart", "openral-tunnel.service"] in calls
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


def test_a_tunnel_token_cannot_add_lines_to_the_service_env(enrolment: Path) -> None:
    """The token lands in a systemd EnvironmentFile: a newline would add LD_PRELOAD."""
    from openral_observability.dashboard.tunnel import write_tunnel_token

    evil = {**_LINK, "tunnel": f"{_TUNNEL}\nLD_PRELOAD=/tmp/evil.so"}
    code = base64.urlsafe_b64encode(json.dumps(evil).encode()).decode()
    result = CliRunner().invoke(app, ["robot", "link", code])
    assert result.exit_code == 2 and "malformed tunnel token" in result.output
    assert not enrolment.with_name("tunnel.env").exists()
    with pytest.raises(ValueError, match="not base64"):
        write_tunnel_token("a\nLD_PRELOAD=x")


def test_the_dashboard_leaves_the_tunnel_to_a_running_service(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_observability.dashboard.tunnel import TUNNEL_PORT, start_tunnel

    log = _fake_cloudflared(tmp_path, monkeypatch, active=True)
    assert CliRunner().invoke(app, ["robot", "link", _CODE]).exit_code == 0
    assert start_tunnel(TUNNEL_PORT) is None  # the service holds the tunnel
    assert not log.exists()  # no second cloudflared started


class _Resp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_deploy_start_applies_account_settings_env_first_cache_offline(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The site's settings fill the env (a set variable wins); offline, the 0600 cache does."""
    from openral_cli import robot

    _fake_cloudflared(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE]).exit_code == 0
    for var in ("OPENRAL_REASONER_API_KEY", "OPENRAL_REASONER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "local-wins")
    seen: list[tuple[str, str, str]] = []

    def site(req: Any, timeout: float) -> _Resp:  # the openral.com boundary
        seen.append((req.full_url, req.headers["Authorization"], req.get_header("User-agent")))
        body = {"settings": {"OPENRAL_REASONER_API_KEY": "sk-1", "OPENRAL_REASONER_MODEL": "m"}}
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(robot, "_open", site)
    assert robot.apply_account_settings(echo=lambda _: None) == ["OPENRAL_REASONER_API_KEY"]
    # Not urllib's default agent, which Cloudflare refuses.
    assert seen == [
        ("https://www.openral.com/api/robot/settings", "Bearer ors_r0b0tsecret", "openral-robot")
    ]
    assert os.environ["OPENRAL_REASONER_API_KEY"] == "sk-1"
    assert os.environ["OPENRAL_REASONER_MODEL"] == "local-wins"
    cache = enrolment.with_name("settings.json")
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600

    def offline(req: Any, timeout: float) -> _Resp:
        raise OSError("no route to host")

    monkeypatch.setattr(robot, "_open", offline)
    monkeypatch.delenv("OPENRAL_REASONER_API_KEY")
    assert robot.apply_account_settings(echo=lambda _: None) == ["OPENRAL_REASONER_API_KEY"]
    assert os.environ["OPENRAL_REASONER_API_KEY"] == "sk-1"


def test_account_settings_never_set_the_robots_own_switches(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote and write controls stay on the robot (Safety-WG), whatever the site sends."""
    from openral_cli import robot

    _fake_cloudflared(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE]).exit_code == 0
    monkeypatch.delenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", raising=False)
    body = {"OPENRAL_DASHBOARD_REMOTE_CONTROL": "1", "OPENRAL_REASONER_MODEL": "m"}
    monkeypatch.setattr(
        robot,
        "_open",
        lambda req, timeout: _Resp(json.dumps({"settings": body}).encode()),
    )
    assert robot.apply_account_settings(echo=lambda _: None) == ["OPENRAL_REASONER_MODEL"]
    assert "OPENRAL_DASHBOARD_REMOTE_CONTROL" not in os.environ


def test_a_refused_credential_drops_the_cache_and_a_new_link_drops_the_old_robot(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removed or re-keyed robot keeps no copy of its org's key; neither does a re-link."""
    import urllib.error

    from openral_cli import robot

    _fake_cloudflared(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE]).exit_code == 0
    cache = enrolment.with_name("settings.json")
    cache.write_text('{"OPENRAL_REASONER_API_KEY": "sk-1"}')
    monkeypatch.delenv("OPENRAL_REASONER_API_KEY", raising=False)

    def refused(req: Any, timeout: float) -> _Resp:
        raise urllib.error.HTTPError(req.full_url, 401, "Unknown robot credential", {}, None)  # type: ignore[arg-type]  # reason: no headers needed

    monkeypatch.setattr(robot, "_open", refused)
    assert robot.apply_account_settings(echo=lambda _: None) == []
    assert not cache.exists() and "OPENRAL_REASONER_API_KEY" not in os.environ

    # Linking another robot, from a code without a secret, keeps nothing of the first.
    cache.write_text('{"OPENRAL_REASONER_API_KEY": "sk-1"}')
    older = {k: v for k, v in _LINK.items() if k != "secret"}
    code = base64.urlsafe_b64encode(json.dumps(older).encode()).decode()
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", code]).exit_code == 0
    assert not cache.exists() and not enrolment.with_name("robot.secret").exists()


def test_the_settings_request_never_follows_a_redirect() -> None:
    """A redirect would carry the robot's credential to another host."""
    import urllib.request

    from openral_cli import robot

    req = urllib.request.Request("https://www.openral.com/api/robot/settings")
    assert robot._NoRedirect().redirect_request(req, None, 302, "Found", {}, "https://evil") is None


def test_the_robots_own_key_never_meets_an_account_endpoint(
    enrolment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Console settings chose the reasoner: no account reasoner setting is applied at all.

    Otherwise a site (or a compromised one) naming its own endpoint would receive the key
    the operator saved on the robot.
    """
    from openral_cli import robot
    from openral_observability.dashboard.robot_settings import save_setting

    _fake_cloudflared(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["robot", "link", "--no-service", _CODE]).exit_code == 0

    def site(req: Any, timeout: float) -> _Resp:  # the openral.com boundary
        body = {"settings": {"OPENRAL_REASONER_ENDPOINT": "https://evil.example/v1"}}
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(robot, "_open", site)
    save_setting("ANTHROPIC_API_KEY", "sk-ant-robot-only-1111")
    save_setting("OPENRAL_REASONER_MODEL", "claude-sonnet-5-5")
    assert robot.apply_reasoner_settings(echo=lambda _: None) == [
        "OPENRAL_REASONER_API_KEY",
        "OPENRAL_REASONER_MODEL",
    ]
    assert "OPENRAL_REASONER_ENDPOINT" not in os.environ
    # Without a console model the account's settings apply as before.
    for var in ("OPENRAL_REASONER_API_KEY", "OPENRAL_REASONER_MODEL"):
        monkeypatch.delenv(var)
    save_setting("OPENRAL_REASONER_MODEL", "")
    assert robot.apply_reasoner_settings(echo=lambda _: None) == ["OPENRAL_REASONER_ENDPOINT"]
