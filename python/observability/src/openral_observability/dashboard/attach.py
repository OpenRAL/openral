"""Spawn a child ``openral dashboard`` and attach the current process to it.

Inverse of ``openral dashboard --inprocess <cmd>``: there, the *dashboard*
spawns the workload as a child with OTLP env pre-set. Here, the
*workload* spawns the dashboard as a child (e.g. ``openral sim run
--dashboard``) so a user can light up the live pane without juggling
two terminals.

The helper is intentionally tolerant: if the child fails to start or
``/healthz`` never comes up, the workload continues without OTel
attached — convenience must not gate the run.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["attached_dashboard", "spawn_dashboard"]

_LOG = logging.getLogger(__name__)

_ENV_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
_ENV_PROTOCOL = "OTEL_EXPORTER_OTLP_PROTOCOL"
_HTTP_OK = 200


@contextmanager
def spawn_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 4318,
    ready_timeout_s: float = 10.0,
) -> Iterator[str | None]:
    """Spawn ``openral dashboard`` as a child, set OTLP env, yield the URL.

    On enter:
        1. ``shutil.which('openral')`` to locate the in-tree CLI. If missing,
           yield ``None`` (no dashboard attached).
        2. ``Popen(['openral', 'dashboard', '--host', host, '--port', port])``.
        3. Poll ``/healthz`` until 200 or ``ready_timeout_s`` elapses.
        4. Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` + ``OTEL_EXPORTER_OTLP_PROTOCOL``
           in ``os.environ`` so the next ``configure_observability`` call
           picks the endpoint up.
        5. Print a single-line URL banner to stderr (matches the banner
           ``openral dashboard`` itself prints — same shape so tooling can
           grep for either).

    On exit:
        SIGINT the child (BatchSpanProcessor flushes on shutdown), wait
        up to 5 s, then escalate to terminate / kill. Restore the prior
        ``OTEL_EXPORTER_OTLP_*`` env values so a nested process tree
        doesn't inherit stale config.

    Yields:
        The full dashboard URL (``http://host:port/``) when attached,
        or ``None`` if the child could not be started or never reported
        healthy — caller should continue the workload either way.
    """
    # The CLI entry point is ``openral`` per ADR-0021 (single console
    # script; bare ``ral`` was renamed). When the workload runs in an
    # environment where ``.venv/bin`` isn't on PATH (e.g. inside
    # ``ros2 launch`` whose env was built from
    # ``/opt/ros/jazzy/setup.bash``) ``shutil.which('openral')``
    # returns None, so look for the console script next to
    # ``sys.executable`` as a fallback — that's where ``uv sync``
    # installs the entry point. We deliberately do NOT fall back to
    # ``python -m openral_cli.main``: that re-imports the CLI in a
    # child interpreter and trips a pre-existing
    # ``if __name__ == "__main__":`` early-return that skips
    # registration of subcommands defined later in the file.
    exe_path = shutil.which("openral")
    if exe_path is None:
        candidate = os.path.join(os.path.dirname(sys.executable), "openral")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            exe_path = candidate
    if exe_path is None:
        _LOG.warning(
            "--dashboard requested but the 'openral' console script "
            "could not be located (neither on PATH nor next to "
            "sys.executable=%s); continuing without an attached "
            "dashboard.",
            sys.executable,
        )
        yield None
        return

    link_host = "localhost" if host in {"0.0.0.0", "::", ""} else host
    url = f"http://{link_host}:{port}/"
    healthz = f"{url}healthz"

    child = subprocess.Popen(
        [exe_path, "dashboard", "--host", host, "--port", str(port)],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    prior = {
        _ENV_ENDPOINT: os.environ.get(_ENV_ENDPOINT),
        _ENV_PROTOCOL: os.environ.get(_ENV_PROTOCOL),
    }
    try:
        if not _wait_healthy(healthz, child, timeout_s=ready_timeout_s):
            _LOG.warning(
                "dashboard child did not report healthy on %s within %.1fs; "
                "continuing without an attached dashboard.",
                healthz,
                ready_timeout_s,
            )
            yield None
            return

        os.environ[_ENV_ENDPOINT] = f"http://{link_host}:{port}"
        os.environ[_ENV_PROTOCOL] = "http/protobuf"
        print(
            f"OpenRAL dashboard attached: {url}  (child process; will exit with this command)",
            file=sys.stderr,
            flush=True,
        )
        yield url
    finally:
        _shutdown_child(child)
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def attached_dashboard(*, enabled: bool, port: int = 4318) -> Iterator[bool]:
    """Convenience wrapper for CLI commands that gate dashboard attach on a flag.

    Pattern at call sites (``openral sim run``, ``openral deploy run``,
    ``openral benchmark run``):

        with attached_dashboard(enabled=dashboard, port=dashboard_port):
            rc = _run(args)

    Behaviour:

    * ``enabled=False`` — yield ``False`` immediately, no child spawned.
    * ``enabled=True`` — delegate to :func:`spawn_dashboard`; if it
      yields a URL, re-run :func:`configure_observability` so the
      current process re-binds onto the freshly-attached endpoint;
      otherwise yield ``False`` (workload continues unattached).
    * On exit (regardless of attach success / workload outcome): if we
      were attached, drain via :func:`shutdown_observability` *before*
      the child SIGINT in :func:`spawn_dashboard`'s ``finally`` so the
      last span/metric batch lands instead of churning on
      ``Connection refused`` after the receiver is gone.

    Yields ``True`` iff a dashboard was actually attached.
    """
    if not enabled:
        yield False
        return
    # Deferred imports: keep this module importable without pulling
    # the full SDK eagerly. Same rationale as the sim-side comment.
    from openral_observability._sdk import (
        configure_observability,
        shutdown_observability,
    )

    with spawn_dashboard(port=port) as attached:
        if attached is not None:
            configure_observability(service_name="ral")
        try:
            yield attached is not None
        finally:
            if attached is not None:
                shutdown_observability()


def _wait_healthy(healthz_url: str, child: subprocess.Popen[bytes], *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(healthz_url, timeout=0.5) as resp:
                if resp.status == _HTTP_OK:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.1)
    return False


def _shutdown_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    try:
        child.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    child.terminate()
    try:
        child.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2.0)
