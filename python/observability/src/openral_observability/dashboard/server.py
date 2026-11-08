"""Uvicorn entry point for the dashboard.

Wrapped in its own module so ``openral dashboard`` can ``import`` and call
``run_dashboard`` without pulling FastAPI / uvicorn into the
critical-path of every ``openral`` invocation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from openral_core import RobotManifest

if TYPE_CHECKING:
    import uvicorn

    from openral_observability.dashboard.store import TelemetryStore

__all__ = ["run_dashboard"]

_LOG = logging.getLogger(__name__)

# Hosts that keep the (unauthenticated) local listener reachable only from this
# machine. On any other bind, other hosts get the read-only LAN view plus E-STOP.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})
_DEFAULT_REMOTE_PORT = 4319


def _exposure_warning(host: str) -> str | None:
    """Return a security warning when *host* is not loopback, else ``None``.

    The local listener has no authentication. Bound to a routable address, every
    host that can reach the port sees the robot read-only and can press E-STOP
    (ADR-0129: non-loopback clients get nothing else), so the bind is surfaced
    loudly rather than happening silently.

    Args:
        host: The bind address passed to ``run_dashboard``.

    Returns:
        A human-readable warning string for a non-loopback bind, or ``None``
        when the bind stays on the local machine.
    """
    if host.strip().lower() in _LOOPBACK_HOSTS:
        return None
    return (
        f"dashboard bound to {host!r}, which is NOT loopback. It has no login: any "
        "host that can reach this port sees the robot read-only and can press "
        "E-STOP (only this machine gets full control). For remote control, link "
        "the robot to openral.com (`openral robot link`) instead."
    )


def run_dashboard(  # noqa: PLR0915, PLR0912  # reason: linear bootstrap (app + UI assets + ROS helpers + discovery + listeners)
    *,
    host: str = "127.0.0.1",
    port: int = 4318,
    remote_port: int | None = None,
    bridge_port: int = 8765,
    manifest_path: Path | None = None,
    inprocess_cmd: list[str] | None = None,
    store: TelemetryStore | None = None,
    log_level: str = "warning",
) -> None:
    """Start the dashboard's listeners and block until SIGINT.

    Args:
        host: Local listener bind address; loopback by default. Loopback clients get
            full control; on another bind, other hosts get read-only + E-STOP.
        port: Local listener port: the UI, the API and the OTLP/HTTP receiver.
            Defaults to ``4318``, the OTLP/HTTP standard port (issue #132: ``8000``
            collides with ``mkdocs serve`` and most FastAPI demos).
        remote_port: The remote listener, on 127.0.0.1 for the robot's Cloudflare
            Tunnel (ADR-0129). ``None`` opens it on ``4319`` when the robot is linked
            (``openral robot link``) and not otherwise; a port without an enrolment
            is refused.
        bridge_port: ``foxglove_bridge``'s loopback port, proxied at ``/ws``.
        manifest_path: A robot manifest JSON file (the deploy launch writes one) served
            until the runtime sends a fresher ``openral.robot.manifest`` record, so a
            deploy without the reasoner still has one. Unreadable → logged, ignored.
        inprocess_cmd: Optional argv to spawn as a child process with
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` + ``OTEL_EXPORTER_OTLP_PROTOCOL``
            pointed at the dashboard. Lets a user do
            ``openral dashboard --inprocess -- openral sim run
            --config foo.yaml --rskill rskills/<id>``
            in one terminal.
        store: Optional pre-built ``TelemetryStore``. A fresh one
            is created when ``None`` — supplying one is useful for
            tests that want to assert against the store after a run.
        log_level: uvicorn log level. ``warning`` is quiet enough for a
            demo without being silent on errors.

    Raises:
        ValueError: ``remote_port`` was given but the robot is not linked, or the
            enrolment file is invalid.
    """
    import uvicorn  # local import: heavy enough to defer until invocation

    from openral_observability.dashboard.app import (
        _remote_control_enabled,
        _write_controls_enabled,
        create_app,
    )
    from openral_observability.dashboard.remote_auth import (
        JwksCache,
        TokenVerifier,
        jwks_cache_file,
        load_enrolment,
        remote_listener,
    )
    from openral_observability.dashboard.ui_assets import ensure_ui_assets

    enrolment = load_enrolment()
    if remote_port is not None and enrolment is None:
        msg = "--remote-port needs a linked robot: run `openral robot link <code>` first"
        raise ValueError(msg)
    if enrolment is not None and remote_port is None:
        remote_port = _DEFAULT_REMOTE_PORT
    verifier = (
        TokenVerifier(enrolment, JwksCache(enrolment.jwks, jwks_cache_file()))
        if enrolment is not None
        else None
    )
    app = create_app(store, verifier=verifier, bridge_url=f"ws://127.0.0.1:{bridge_port}")
    if manifest_path is not None:
        try:
            manifest = RobotManifest.model_validate_json(manifest_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:  # never gate the dashboard (or its E-STOP) on it
            _LOG.warning("dashboard.manifest_file_unreadable path=%s error=%s", manifest_path, exc)
        else:
            app.state.store.set_manifest(manifest)

    # The robot UI: the pinned release, fetched once into $OPENRAL_CACHE_DIR (see
    # ui_assets.py). Best-effort: offline, `/` serves the built-in fallback page.
    try:
        app.state.ui_dir = ensure_ui_assets()
        if app.state.ui_dir is None:
            _LOG.warning("dashboard.ui_assets unavailable — serving the fallback page at /")
    except Exception as exc:  # never gate the dashboard (or its E-STOP) on the UI download
        _LOG.warning("dashboard.ui_assets_start_failed error=%s", exc)

    # Persistent e-stop publisher: created ONCE here so DDS discovery of the
    # HAL/kernel/runner subscribers happens at launch, and every later E-STOP
    # press publishes instantly (no per-press shell-out / discovery race — a
    # racing shell-out once lost the message and the robot never stopped). Inert
    # + graceful when rclpy/ROS is absent; the endpoints then fall back.
    try:
        from openral_observability.dashboard.estop_publisher import EstopPublisher

        app.state.estop = EstopPublisher()
        if app.state.estop.available:
            _LOG.info("dashboard.estop_publisher ready (instant e-stop)")
        else:
            _LOG.warning("dashboard.estop_publisher inert (no ROS) — estop falls back to shell-out")
    except Exception as exc:  # never gate the dashboard on the e-stop publisher
        _LOG.warning("dashboard.estop_publisher_start_failed error=%s", exc)

    # ADR-0096 — the latched /openral/safety_status subscriber. Read-only: it
    # feeds the Safety Status card with the graph's authoritative current
    # safety state, which the span-inferred latch cannot supply once a latched
    # kernel stops the chunk flow the inference rides on. Created here so the
    # TRANSIENT_LOCAL sample lands at launch. Inert + harmless without ROS.
    try:
        from openral_observability.dashboard.safety_status_subscriber import (
            SafetyStatusSubscriber,
        )

        app.state.safety_status = SafetyStatusSubscriber(app.state.store)
        if app.state.safety_status.available:
            _LOG.info("dashboard.safety_status_subscriber ready (latched safety state)")
        else:
            _LOG.warning(
                "dashboard.safety_status_subscriber inert (no ROS) — "
                "the Safety Status card stays 'waiting'"
            )
    except Exception as exc:  # never gate the dashboard on the safety subscriber
        _LOG.warning("dashboard.safety_status_subscriber_start_failed error=%s", exc)

    # Camera perception overlays — detector boxes drawn over the camera tiles.
    # Read-only and advisory: it only decides what an operator SEES, never what
    # the robot does. Created here for the same reason as the two above, so DDS
    # discovery happens at launch rather than on first frame. Inert + harmless
    # without ROS; the camera tiles then render exactly as they did before.
    try:
        from openral_observability.dashboard.perception_overlay_subscriber import (
            PerceptionOverlaySubscriber,
        )

        app.state.perception_overlay = PerceptionOverlaySubscriber(app.state.store)
        if app.state.perception_overlay.available:
            legs = (
                "detector boxes + segmenter masks"
                if app.state.perception_overlay.masks_available
                else "detector boxes only (openral_msgs/SegmentMasks not in the overlay)"
            )
            _LOG.info("dashboard.perception_overlay_subscriber ready (%s)", legs)
        else:
            _LOG.warning(
                "dashboard.perception_overlay_subscriber inert (no ROS) — "
                "camera tiles render without perception overlays"
            )
    except Exception as exc:  # never gate the dashboard on the overlay subscriber
        _LOG.warning("dashboard.perception_overlay_subscriber_start_failed error=%s", exc)

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

    # proxy_headers off on both listeners: the client address decides local vs LAN,
    # and a forwarded-for header must never turn a LAN client into loopback.
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level,
                access_log=False,
                proxy_headers=False,
                ws_ping_interval=30.0,
            )
        )
    ]
    if remote_port is not None:
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    remote_listener(app),
                    host="127.0.0.1",
                    port=remote_port,
                    log_level=log_level,
                    access_log=False,
                    proxy_headers=False,
                    ws_ping_interval=30.0,
                    lifespan="off",
                )
            )
        )

    exposure = _exposure_warning(host)
    if exposure is not None:
        _LOG.warning("dashboard.exposed_bind host=%s", host)
        print(f"WARNING: {exposure}", file=sys.stderr, flush=True)

    if _write_controls_enabled():
        _LOG.warning("dashboard.write_controls_enabled")
        print(
            "WARNING: dashboard write-controls ENABLED (OPENRAL_DASHBOARD_WRITE_CONTROLS=1): "
            "the robot's installed rSkills can be run from the dashboard. The safety kernel "
            "still disposes all motion.",
            file=sys.stderr,
            flush=True,
        )
    if remote_port is not None and _remote_control_enabled():
        _LOG.warning("dashboard.remote_control_enabled")
        print(
            "WARNING: remote control ENABLED (OPENRAL_DASHBOARD_REMOTE_CONTROL=1): signed-in "
            "operators can reset the e-stop, prompt and run skills over the robot's tunnel, "
            "away from the work cell (IEC 60204-1 expects a reset where the hazard is seen).",
            file=sys.stderr,
            flush=True,
        )

    def _on_signal(signum: int, _frame: object) -> None:
        for server in servers:
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
    if enrolment is not None and remote_port is not None:
        print(
            f"OpenRAL dashboard remote listener: 127.0.0.1:{remote_port} "
            f"(tunnel target for {enrolment.aud})",
            file=sys.stderr,
            flush=True,
        )
    tunnel = None
    if enrolment is not None and remote_port is not None:
        from openral_observability.dashboard.tunnel import start_tunnel

        tunnel = start_tunnel(remote_port)
        if tunnel is not None:
            print(
                f"OpenRAL robot tunnel: {enrolment.aud} -> 127.0.0.1:{remote_port}",
                file=sys.stderr,
                flush=True,
            )
    try:
        asyncio.run(_serve(servers))
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        if discovery is not None:
            discovery.stop()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()


async def _serve(servers: list[uvicorn.Server]) -> None:
    """Run every listener; when one stops (signal or bind failure), stop them all."""
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for server in servers:
        server.should_exit = True
    await asyncio.gather(*tasks)


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
