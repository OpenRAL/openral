"""The robot's Cloudflare Tunnel, run by the dashboard itself (ADR-0129 §3).

``openral robot link <code>`` stores the tunnel token next to the enrolment
(``tunnel.env``, mode 0600), fetches a pinned ``cloudflared`` into the cache, and
installs it as a **user** systemd service (``openral-tunnel.service``, with
``loginctl enable-linger`` so it starts at boot without a login): the tunnel stays up
across reboots, with no root. Where there is no user systemd (containers, some
minimal images) ``openral dashboard`` runs the same tunnel as its own child instead,
up exactly while the dashboard is. The tunnel's ingress is configured remotely by
openral.com and points at ``localhost:4319``, the remote listener; while the
dashboard is down, openral.com shows the robot as unreachable.

The token goes to cloudflared through ``TUNNEL_TOKEN`` in its environment, never on
its command line (``ps`` shows argv to every user).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import structlog
from openral_core import enrolment_path

from openral_observability.dashboard.ui_assets import PinnedAsset, _cache_root, _download

__all__ = [
    "CLOUDFLARED",
    "SERVICE",
    "TUNNEL_PORT",
    "TUNNEL_TOKEN_RE",
    "ensure_cloudflared",
    "install_service",
    "read_tunnel_token",
    "service_active",
    "start_tunnel",
    "tunnel_env_path",
    "write_tunnel_token",
]

_logger = structlog.get_logger(__name__)

TUNNEL_PORT = 4319
"""Where openral.com's tunnel ingress sends ``/api`` and ``/ws``: the remote listener."""

_RELEASE = "https://github.com/cloudflare/cloudflared/releases/download/2026.10.7"
CLOUDFLARED: dict[str, PinnedAsset] = {
    "x86_64": PinnedAsset(
        url=f"{_RELEASE}/cloudflared-linux-amd64",
        sha256="1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872",
        size=40274445,
    ),
    "aarch64": PinnedAsset(
        url=f"{_RELEASE}/cloudflared-linux-arm64",
        sha256="3df43aedc9ff291d1735b5cb576cb97e0b157488796db2e32b28aecabb445775",
        size=37667293,
    ),
}
"""cloudflared 2026.10.7 (2026-10-30) per ``platform.machine()``; digests from the release."""


TUNNEL_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]+")
"""A Cloudflare tunnel token: base64 of its JSON credentials."""

SERVICE = "openral-tunnel.service"
"""The user systemd unit ``openral robot link`` installs."""

_UNIT = """[Unit]
Description=OpenRAL robot tunnel to openral.com (openral robot link)
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile={env}
ExecStart={exe} tunnel --no-autoupdate run
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def tunnel_env_path() -> Path:
    """``tunnel.env`` next to the enrolment file: ``TUNNEL_TOKEN=<token>``."""
    return enrolment_path().with_name("tunnel.env")


def write_tunnel_token(token: str) -> Path:
    """Store the tunnel token (mode 0600, never through a symlink); returns the file.

    Raises:
        ValueError: ``token`` is not base64: anything else (a newline) would add lines to
            the service's EnvironmentFile, e.g. ``LD_PRELOAD``.
    """
    if not TUNNEL_TOKEN_RE.fullmatch(token):
        raise ValueError("the tunnel token is not base64")
    path = tunnel_env_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)  # an existing file keeps its old mode through O_CREAT
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"TUNNEL_TOKEN={token}\n")
    return path


def read_tunnel_token() -> str | None:
    """The stored tunnel token, or ``None`` when the robot isn't linked."""
    path = tunnel_env_path()
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "TUNNEL_TOKEN" and value.strip():
            return value.strip()
    return None


def ensure_cloudflared() -> Path | None:
    """The pinned cloudflared for this machine, fetched once; else one on ``PATH``, or ``None``."""
    asset = CLOUDFLARED.get(platform.machine())
    if asset is not None:
        exe = _cache_root() / "cloudflared" / asset.sha256 / "cloudflared"
        if exe.is_file():
            return exe
        try:
            exe.parent.mkdir(parents=True, exist_ok=True)
            _download(asset, exe)
            exe.chmod(0o755)
            return exe
        except (OSError, ValueError) as exc:
            _logger.warning("dashboard.cloudflared_fetch_failed", url=asset.url, error=str(exc))
    found = shutil.which("cloudflared")
    return Path(found) if found else None


def _unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / SERVICE


def _run(*argv: str) -> bool:
    """Run a systemd/logind tool quietly; ``False`` when it is missing or fails."""
    exe = shutil.which(argv[0])
    if exe is None:
        return False
    return subprocess.run([exe, *argv[1:]], capture_output=True, check=False).returncode == 0


def install_service(exe: Path) -> str | None:
    """Install and start ``openral-tunnel.service`` for this user, enabled at boot.

    Returns:
        ``None`` on success, else why it couldn't (the dashboard then runs the tunnel).
    """
    unit = _unit_path()
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_UNIT.format(env=tunnel_env_path(), exe=exe), encoding="utf-8")
    if not _run("systemctl", "--user", "daemon-reload"):
        return "no user systemd here"
    # restart, not just `enable --now`: a running unit (an earlier link) keeps its old token.
    if not (
        _run("systemctl", "--user", "enable", SERVICE)
        and _run("systemctl", "--user", "restart", SERVICE)
    ):
        return f"systemctl --user enable/restart {SERVICE} failed"
    # Start at boot, before anyone logs in. Allowed for yourself on stock polkit.
    if not _run("loginctl", "enable-linger"):
        return "the tunnel runs, but only while you are logged in (loginctl enable-linger failed)"
    return None


def service_active() -> bool:
    """Whether ``openral-tunnel.service`` is running for this user."""
    return _run("systemctl", "--user", "is-active", "--quiet", SERVICE)


def start_tunnel(remote_port: int) -> subprocess.Popen[bytes] | None:
    """Run the robot's tunnel as a child process, unless the user service already does.

    ``None`` when the robot isn't linked, the service runs it, or there is no binary.
    """
    token = read_tunnel_token()
    if token is None:
        return None
    if service_active():
        return None
    if remote_port != TUNNEL_PORT:
        _logger.warning(
            "dashboard.tunnel_port_mismatch", remote_port=remote_port, ingress=TUNNEL_PORT
        )
    exe = ensure_cloudflared()
    if exe is None:
        _logger.warning("dashboard.tunnel_unavailable", reason="no cloudflared for this machine")
        return None
    env = {**os.environ, "TUNNEL_TOKEN": token}
    return subprocess.Popen(  # pinned binary, fixed argv; the token rides in env
        [str(exe), "tunnel", "--no-autoupdate", "run"], env=env
    )
