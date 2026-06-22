"""Uvicorn entry point for the dashboard.

Wrapped in its own module so ``openral dashboard`` can ``import`` and call
:func:`run_dashboard` without pulling FastAPI / uvicorn into the
critical-path of every ``openral`` invocation.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openral_observability.dashboard.store import TelemetryStore

__all__ = ["run_dashboard"]

_LOG = logging.getLogger(__name__)

# Hosts that keep the (unauthenticated) dashboard reachable only from the local
# machine. Anything else exposes the OTLP receiver and POST /api/prompt — which
# injects operator prompts into the robot's reasoner — to the network.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def _exposure_warning(host: str) -> str | None:
    """Return a security warning when *host* is not loopback, else ``None``.

    The dashboard has no authentication (CLAUDE.md §1 / issue #44 — localhost
    only). Binding to ``0.0.0.0`` or any routable address lets anyone who can
    reach the port post prompts that drive the reasoner and spoof telemetry, so
    a non-loopback bind must be surfaced loudly rather than happen silently.

    Args:
        host: The bind address passed to :func:`run_dashboard`.

    Returns:
        A human-readable warning string for a non-loopback bind, or ``None``
        when the bind stays on the local machine.
    """
    if host.strip().lower() in _LOOPBACK_HOSTS:
        return None
    return (
        f"dashboard bound to {host!r}, which is NOT loopback. The dashboard has "
        "no authentication: any host that can reach this port can inject prompts "
        "into the reasoner (POST /api/prompt) and spoof telemetry. Bind 127.0.0.1 "
        "and use an SSH tunnel for remote access."
    )


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 4318,
    inprocess_cmd: list[str] | None = None,
    store: TelemetryStore | None = None,
    log_level: str = "warning",
) -> None:
    """Start the dashboard server on ``host:port`` and block until SIGINT.

    Args:
        host: Bind address; defaults to loopback (no auth, per
            CLAUDE.md §1 / issue #44 — explicitly localhost-only).
        port: HTTP port. The same port serves the UI, the SSE stream,
            and the OTLP/HTTP receiver routes. Defaults to ``4318``, the
            OTLP/HTTP standard receiver port — chosen over the historic
            ``8000`` (issue #132) because ``8000`` is the default for
            ``mkdocs serve`` (``just docs``), ``python -m http.server``,
            and most FastAPI tutorials, so the dashboard frequently
            collided with a workload already running on the dev host.
        inprocess_cmd: Optional argv to spawn as a child process with
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` + ``OTEL_EXPORTER_OTLP_PROTOCOL``
            pointed at the dashboard. Lets a user do
            ``openral dashboard --inprocess -- openral sim run
            --config foo.yaml --rskill rskills/<id>``
            in one terminal.
        store: Optional pre-built :class:`TelemetryStore`. A fresh one
            is created when ``None`` — supplying one is useful for
            tests that want to assert against the store after a run.
        log_level: uvicorn log level. ``warning`` is quiet enough for a
            demo without being silent on errors.
    """
    import uvicorn  # local import: heavy enough to defer until invocation

    from openral_observability.dashboard.app import create_app

    app = create_app(store)

    discovery = None
    try:
        from openral_observability.dashboard.discovery import Discovery

        discovery = Discovery()
        discovery.start(host=host, port=port)
        app.state.discovery = discovery
    except Exception as exc:  # discovery is best-effort; never gate the dashboard
        _LOG.warning("dashboard.discovery_start_failed error=%s", exc)

    child: subprocess.Popen[bytes] | None = None
    if inprocess_cmd:
        child = _spawn_child(inprocess_cmd, host=host, port=port)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    exposure = _exposure_warning(host)
    if exposure is not None:
        _LOG.warning("dashboard.exposed_bind host=%s", host)
        print(f"WARNING: {exposure}", file=sys.stderr, flush=True)

    def _on_signal(signum: int, _frame: object) -> None:
        server.should_exit = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    # Print the URL on a single, easy-to-grep line before uvicorn starts
    # (issue #132). If the bind subsequently fails, the user still sees
    # which URL was attempted, alongside uvicorn's error.
    link_host = "localhost" if host in {"0.0.0.0", "::", ""} else host
    print(
        f"OpenRAL dashboard: http://{link_host}:{port}/  "
        f"(OTLP endpoint: http://{link_host}:{port})",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.run()
    finally:
        if discovery is not None:
            discovery.stop()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()


def _spawn_child(cmd: list[str], *, host: str, port: int) -> subprocess.Popen[bytes]:
    """Spawn ``cmd`` with OTLP env pointing at the dashboard.

    ``host`` is the dashboard bind address; the child uses ``127.0.0.1``
    when the dashboard is bound to a wildcard or ``0.0.0.0`` so it
    always lands on the local loopback (no DNS round-trip).
    """
    exe = shutil.which(cmd[0])
    if exe is None:
        msg = f"`openral dashboard --inprocess`: command not found: {cmd[0]!r}"
        raise FileNotFoundError(msg)
    env = os.environ.copy()
    child_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://{child_host}:{port}"
    env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    _LOG.info("openral dashboard spawning child: %s", cmd)
    return subprocess.Popen([exe, *cmd[1:]], env=env, stdout=sys.stdout, stderr=sys.stderr)
