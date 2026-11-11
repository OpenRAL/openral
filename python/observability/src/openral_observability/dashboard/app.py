"""FastAPI app for the robot dashboard (ADR-0129).

One ASGI app, served by up to two listeners (``remote_auth``):

* ``/`` — the robot UI (``ui_assets``), or the built-in ``static/fallback.html``
  (status + E-STOP) when no UI is installed. ``/viz/`` — the pinned Lichtblick 3D
  view (``viz_assets``).
* ``GET /api/config``, ``/api/manifest``, ``/api/state``, ``/api/stream[?images=0]``
  (SSE), ``/api/camera/{source}/snapshot.jpg``, ``/api/traces``,
  ``/api/spans/{trace_id}`` — reads.
* ``POST /api/actions/{id}`` — the manifest's actions: ``estop``, ``estop_reset``,
  ``prompt``, ``skill:<rskill id>``. Role, LAN and remote-control rules are applied
  here, every action is announced on ``/openral/operator_action`` for the compliance
  recorder, and the safety kernel still disposes all motion.
* ``POST /api/viz/ticket``, ``WS /ws`` — the 3D view's
  ticketed proxy to ``foxglove_bridge`` on loopback.
* ``POST /api/transcribe`` — operator speech to text, on the host.
* ``/v1/traces``, ``/v1/metrics``, ``/v1/logs`` — the OTLP/HTTP receiver, on the local
  listener only: loopback, plus LAN hosts with ``$OPENRAL_DASHBOARD_LAN_INGEST=1``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import http
import importlib
import io
import ipaddress
import itertools
import json
import mimetypes
import os
import re
import shutil
import socket
import time
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import jsonschema
import structlog
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from google.protobuf.message import DecodeError
from openral_core import (
    ACTOR_KEY_ENV,
    REMOTE_CONTROL_ENV,
    WRITE_CONTROLS_ENV,
    ManifestAction,
    RobotManifest,
    pseudonymous_actor_id,
)
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from openral_observability.dashboard.bridge_relay import BridgeRelay
from openral_observability.dashboard.remote_auth import (
    ROLE_RANK,
    Principal,
    TokenVerifier,
    VizTickets,
    allowed_origins,
    bearer,
    is_remote,
)
from openral_observability.dashboard.robot_settings import (
    SETTINGS,
    load_settings,
    save_setting,
    shown,
)
from openral_observability.dashboard.store import TelemetryStore
from openral_observability.dashboard.ui_assets import ui_dir

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

__all__ = ["LAN_INGEST_ENV", "create_app"]

_logger = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _write_controls_enabled() -> bool:
    """Whether skills may run from the dashboard (``$OPENRAL_DASHBOARD_WRITE_CONTROLS=1``)."""
    return os.environ.get(WRITE_CONTROLS_ENV, "") == "1"


def _remote_control_enabled() -> bool:
    """Whether reset, prompt and skills may run over the tunnel (default off; E-STOP always)."""
    return os.environ.get(REMOTE_CONTROL_ENV, "") == "1"


LAN_INGEST_ENV = "OPENRAL_DASHBOARD_LAN_INGEST"
"""Opt-in for OTLP ingest on ``/v1/*`` from LAN hosts; loopback always may (ADR-0017)."""


def _lan_ingest_enabled() -> bool:
    """Whether other hosts may export OTLP here (``$OPENRAL_DASHBOARD_LAN_INGEST=1``)."""
    return os.environ.get(LAN_INGEST_ENV, "") == "1"


# Request-body budget, raw AND gzip-decoded (OTLP batches carry camera
# thumbnails, so this is generous; it exists to stop a gzip bomb, not to tune).
_MAX_BODY_BYTES = 64 * 1024 * 1024

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# The only writes a LAN client (non-loopback, local listener) may make: E-STOP, a ticket
# to watch the 3D view, and — only with $OPENRAL_DASHBOARD_LAN_INGEST=1 — OTLP ingest on
# /v1/* (a robot computer exporting to a dashboard on the LAN, ADR-0017): off by default,
# so a LAN host cannot fill the event log or fake safety telemetry. The UI and the other
# actions are read-only from the LAN (ADR-0129 §3); the remote listener refuses /v1/*.
_LAN_WRITES = frozenset({"/api/actions/estop", "/api/viz/ticket"})
_DEFAULT_BRIDGE_URL = "ws://127.0.0.1:8765"
_WS_POLICY_VIOLATION = 1008
_WS_UNAVAILABLE = 1011

# E-STOP never waits for a manifest: it is resolvable before the runtime has sent one.
_ESTOP = ManifestAction(id="estop", kind="estop", label="E-STOP", min_role="viewer", remote=True)
# Speech to text runs under the prompt action's rules (operator; remote control on).
_TRANSCRIBE = ManifestAction(
    id="transcribe", kind="prompt", label="Transcribe", min_role="operator", remote=False
)


# Headers Cloudflare's edge adds to every request it forwards. The tunnel targets the
# remote listener only; one of them on the local listener means a tunnel (or another
# proxy) was pointed at it, and such a request must never be the local operator.
_TUNNEL_HEADERS = ("cf-ray", "cf-connecting-ip", "cf-warp-tag-id", "cf-ipcountry", "cf-visitor")


def _via_tunnel(headers: Mapping[str, str]) -> bool:
    """Whether a local-listener request carries Cloudflare-forwarded headers."""
    return any(h in headers for h in _TUNNEL_HEADERS) or (
        "cloudflare" in headers.get("cdn-loop", "").lower()
    )


def _is_loopback(host: str) -> bool:
    """Whether ``host`` (a client address) is this machine."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# DNS rebinding: a page on evil.example rebound to 127.0.0.1 is same-origin with itself
# (its Origin matches its Host), so a loopback client is the local operator only when
# Host names this machine (``app.state.local_hosts``; ``run_dashboard`` adds the bind address).
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", socket.gethostname().lower()})


def _local_host(host_header: str, allowed: set[str]) -> bool:
    """Whether a ``Host`` header names this machine (any port)."""
    host = urlsplit(f"//{host_header}").hostname or ""
    return host in allowed or host.endswith(".localhost")


def _cross_origin(request: Request) -> bool:
    """True when a browser marked ``request`` as coming from another origin.

    Browsers send ``Origin`` on every cross-origin POST (form or fetch, CORS or
    not), so an ``Origin`` whose host:port differs from ``Host`` is a page on
    another site driving the operator's browser (CSRF). No ``Origin`` means a
    non-browser client (curl, ``openral``, an OTLP exporter) and is allowed;
    ``Origin: null`` (sandboxed iframe, file://) is refused. Local listener only:
    the remote listener checks ``Origin`` against its allowlist instead.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return False
    return urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower()


# Background tasks (skill-result drains, post-E-STOP announcements) are kept
# here so the event loop does not GC them before they finish. Each task removes
# itself from this set in its own done-callback.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _in_background(coro: Awaitable[None]) -> None:
    task: asyncio.Task[None] = asyncio.ensure_future(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# Regex to extract the goal ID from `ros2 action send_goal` stdout.
_GOAL_ACCEPTED_RE = re.compile(r"Goal accepted with ID:\s*(\S+)")
_GOAL_REJECTED_PHRASE = "Goal was rejected"

# Default acceptance timeout in seconds (overridden by env).
_DEFAULT_SKILL_ACCEPT_TIMEOUT_S = 12.0

# Lichtblick's web build (served at /viz/) loads ES modules and WebAssembly:
# browsers reject an ESM import served as text/plain, and streaming wasm
# compilation wants application/wasm.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


async def _read_goal_decision(
    stdout: asyncio.StreamReader,
) -> tuple[str | None, bool, list[str]]:
    """Read stdout lines until ``Goal accepted with ID`` or ``Goal was rejected``.

    Returns ``(goal_id, rejected, captured_lines)`` where exactly one of
    ``goal_id`` (non-None) or ``rejected=True`` is set on a decision line.
    If EOF arrives before either marker, both are falsy and ``captured_lines``
    holds the full output.
    """
    captured: list[str] = []
    goal_id: str | None = None
    rejected = False
    async for raw in stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        captured.append(line)
        m = _GOAL_ACCEPTED_RE.search(line)
        if m:
            goal_id = m.group(1)
            return goal_id, rejected, captured
        if _GOAL_REJECTED_PHRASE in line:
            rejected = True
            return goal_id, rejected, captured
    return goal_id, rejected, captured


async def _drain_skill_goal_stdout(
    stdout: asyncio.StreamReader,
    proc: asyncio.subprocess.Process,
    captured_lines: list[str],
    goal_id: str,
    skill_id: str,
) -> None:
    """Drain remaining ``stdout`` after goal acceptance and audit-log the outcome.

    Runs as a background ``asyncio.Task`` — must catch all exceptions and log
    them (never crash the event loop). Started by ``_skill_execute_response``
    after the action server accepts the goal, so the eventual outcome is
    still audit-logged.
    """
    remaining: list[str] = list(captured_lines)
    try:
        async for raw in stdout:
            remaining.append(raw.decode("utf-8", errors="replace").rstrip())
        await proc.wait()
        full_out = "\n".join(remaining)
        # Parse success / failure_reason / trace_id from the output.
        success = "success: True" in full_out or "success=True" in full_out
        failure_reason: str | None = None
        trace_id: str | None = None
        for line in remaining:
            if "failure_reason:" in line:
                failure_reason = line.split("failure_reason:", 1)[-1].strip() or None
            if "trace_id:" in line:
                trace_id = line.split("trace_id:", 1)[-1].strip() or None
        _logger.warning(
            "dashboard_skill_result",
            goal_id=goal_id,
            skill_id=skill_id,
            success=success,
            trace_id=trace_id,
            failure_reason=failure_reason,
        )
    except Exception as exc:  # background task must log, never crash
        _logger.warning(
            "dashboard_skill_drain_error",
            goal_id=goal_id,
            skill_id=skill_id,
            error=str(exc),
        )


# ── Operator voice prompt (POST /api/transcribe) ──────────────────────────────
# A client POSTs captured operator speech here (voice prompts in the robot UI are
# a follow-up, ADR-0129 §2). We transcribe it with a local CPU Whisper model
# (faster-whisper / CTranslate2) and hand the text back for the prompt action.
# The audio never leaves the host. faster-whisper ships with the dashboard extra so
# this works out of the box; the endpoint still degrades to a 503 (rather than
# crashing the dashboard) on the off chance the dependency is ever stripped.
_STT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB guard — a 30 s 16 kHz mono WAV is ~1 MB.
# Decoded budget: a compressed clip under _STT_MAX_BYTES can still expand to
# hours of PCM, so decoding stops at this many seconds instead of buffering it.
_STT_MAX_SECONDS = 120
_STT_SAMPLE_RATE = 16_000  # Whisper's input rate.
_STT_MODEL_CACHE: dict[str, Any] = {}  # name → WhisperModel; loaded once, reused.


class _STTUnavailableError(RuntimeError):
    """Raised when faster-whisper is not importable in this environment."""


class _STTAudioTooLongError(ValueError):
    """Raised when the decoded clip exceeds ``_STT_MAX_SECONDS``."""


def _skip_invalid_frames(frames: Iterator[Any]) -> Iterator[Any]:
    """Yield decoded frames, skipping corrupt ones as ``faster_whisper.decode_audio`` does."""
    import av

    while True:
        try:
            yield next(frames)
        except StopIteration:
            return
        except av.error.InvalidDataError:
            continue


def _decode_audio_bounded(audio: bytes) -> Any:
    """Decode ``audio`` to 16 kHz mono float32, stopping at ``_STT_MAX_SECONDS``.

    Mirrors ``faster_whisper.decode_audio`` (PyAV + s16 resample) but counts
    samples as frames arrive, so a highly compressed upload cannot make the
    decoder buffer its full expansion.

    Raises:
        _STTAudioTooLongError: the decoded clip exceeds the duration budget.
    """
    import av  # faster-whisper's own decoder dependency
    import numpy as np

    max_samples = _STT_MAX_SECONDS * _STT_SAMPLE_RATE
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=_STT_SAMPLE_RATE
    )
    chunks: list[Any] = []
    n = 0
    with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
        # Trailing None flushes the resampler.
        for frame in itertools.chain(_skip_invalid_frames(container.decode(audio=0)), [None]):
            for out in resampler.resample(frame):
                pcm = out.to_ndarray().reshape(-1)
                n += pcm.size
                if n > max_samples:
                    raise _STTAudioTooLongError
                chunks.append(pcm)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe_sync(audio: bytes) -> tuple[str, str]:
    """Load (once, cached) the local Whisper model and transcribe ``audio``.

    Blocking CPU work — call via ``asyncio.to_thread``, never on the event
    loop. Returns ``(text, model_name)``. The model, device and compute type
    are env-selectable (``OPENRAL_STT_MODEL`` / ``_DEVICE`` / ``_COMPUTE``),
    defaulting to ``base.en`` on CPU with int8 quantization so it runs on any
    operator host. ``audio`` may be any container PyAV can decode (WAV, WebM/
    Opus, …); faster-whisper resamples to 16 kHz internally.

    Raises:
        _STTUnavailableError: faster-whisper is not importable (it ships with the
            dashboard extra, so this only fires if the dependency was stripped).
        _STTAudioTooLongError: the decoded clip exceeds ``_STT_MAX_SECONDS``.
    """
    try:
        # ships with the dashboard extra — untyped when present, ImportError if stripped.
        import faster_whisper  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as exc:  # ModuleNotFoundError + a shadowed/None sys.modules entry
        raise _STTUnavailableError from exc
    pcm = _decode_audio_bounded(audio)  # before the model load: refuse long clips cheaply
    name = os.environ.get("OPENRAL_STT_MODEL", "base.en")
    model = _STT_MODEL_CACHE.get(name)
    if model is None:
        device = os.environ.get("OPENRAL_STT_DEVICE", "cpu")
        compute = os.environ.get("OPENRAL_STT_COMPUTE", "int8")
        _logger.info("stt.model_load", model=name, device=device, compute_type=compute)
        model = faster_whisper.WhisperModel(name, device=device, compute_type=compute)
        _STT_MODEL_CACHE[name] = model
    segments, _info = model.transcribe(pcm, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip(), name


async def _transcribe_response(audio: bytes) -> JSONResponse:
    """Transcribe captured operator speech to text (see module note above)."""
    if not audio:
        return JSONResponse({"error": "empty audio body"}, status_code=400)
    if len(audio) > _STT_MAX_BYTES:
        return JSONResponse(
            {"error": f"audio too large ({len(audio)} bytes > {_STT_MAX_BYTES} limit)"},
            status_code=413,
        )
    try:
        text, model_name = await asyncio.to_thread(_transcribe_sync, audio)
    except _STTUnavailableError:
        return JSONResponse(
            {"error": "speech-to-text unavailable — faster-whisper is not importable"},
            status_code=503,
        )
    except _STTAudioTooLongError:
        return JSONResponse(
            {"error": f"audio too long (> {_STT_MAX_SECONDS} s decoded)"}, status_code=413
        )
    except Exception as exc:  # decode/runtime errors surface as 422 — logged, never swallowed
        _logger.warning("stt.transcribe_failed", error=str(exc))
        return JSONResponse({"error": f"transcription failed: {exc}"}, status_code=422)
    return JSONResponse({"status": "ok", "text": text, "model": model_name})


async def _prompt_response(text: str) -> JSONResponse:
    """Publish an operator prompt via ``openral prompt``.

    Shells out so the wire shape (PromptStamped on
    ``/openral/prompt_in/dashboard``, fanned out by ``prompt_router_node`` at
    priority 100) matches the CLI exactly. The console script is ``openral``,
    the single entry point.
    """
    openral = shutil.which("openral")
    if openral is None:
        return _error(503, "`openral` not on PATH; source the workspace install first")
    proc = await asyncio.create_subprocess_exec(
        openral,
        "prompt",
        "--topic",
        "/openral/prompt_in/dashboard",
        "--",  # the operator's text is never an option, even when it starts with "-"
        text,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _error(504, "openral prompt timed out after 10 s")
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "error": "openral prompt failed",
                "returncode": proc.returncode,
                "stderr": stderr,
            },
            status_code=502,
        )
    return JSONResponse({"ok": True, "stdout": stdout, "stderr": stderr})


async def _estop_reset_response(estop: Any = None) -> JSONResponse:
    """Clear a latched safety e-stop via the kernel's ``/openral/estop_reset``.

    The C++ safety kernel latches on a violation and drops every candidate chunk
    until this ``std_srvs/Trigger`` service is called — so after an e-stop NO
    prompt does anything until the latch is cleared. The reset itself is a
    ``ros2 service call`` (a service is request/response, so no discovery race).
    The kernel enforces a post-estop cooldown; an early call returns
    ``success=false`` (HTTP 409) so the operator can retry. On success we also
    broadcast ``/openral/estop_cleared`` so the HAL + runner un-latch too —
    instantly via ``estop`` (the persistent publisher) when present, else the
    shell-out fallback. Re-prompt once this succeeds.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return _error(503, "`ros2` not on PATH; source the workspace install first")
    proc = await asyncio.create_subprocess_exec(
        ros2,
        "service",
        "call",
        "/openral/estop_reset",
        "std_srvs/srv/Trigger",
        "{}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _error(504, "estop_reset timed out after 15 s — is the safety kernel running?")
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "error": "ros2 service call /openral/estop_reset failed",
                "returncode": proc.returncode,
                "stderr": stderr,
            },
            status_code=502,
        )
    # Trigger response renders as `success=True/False, message='…'`.
    accepted = "success=True" in stdout
    if accepted:
        # The kernel service only clears the KERNEL latch; HAL + runner latch
        # independently on /openral/estop and have no reset path of their own
        # (they'd otherwise stay latched until a node restart). Broadcast
        # /openral/estop_cleared so those clear too — instantly via the
        # persistent publisher when present, else the shell-out fallback.
        # Best effort: a publish failure doesn't undo the kernel reset.
        if estop is not None and estop.available:
            estop.clear()
        else:
            await _publish_estop_cleared()
    return JSONResponse(
        {
            "ok": accepted,
            "accepted": accepted,
            "stdout": stdout,
            **({} if accepted else {"error": "the kernel refused the reset (cooldown?)"}),
        },
        status_code=200 if accepted else 409,
    )


async def _publish_estop_cleared() -> None:
    """Broadcast /openral/estop_cleared so latching nodes (HAL, runner) resume.

    Symmetric to the /openral/estop trigger. Best-effort + swallowed errors:
    the kernel reset has already succeeded by the time we get here, so a failed
    broadcast must not turn a successful reset into an error response.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return
    with contextlib.suppress(Exception):
        proc = await asyncio.create_subprocess_exec(
            ros2,
            "topic",
            "pub",
            "-w",
            "1",
            "--times",
            "3",
            "--rate",
            "10",
            "/openral/estop_cleared",
            "std_msgs/msg/Empty",
            "{}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10.0)


async def _estop_trigger_response() -> JSONResponse:
    """Trigger a graph-wide safety e-stop by publishing to ``/openral/estop``.

    The C++ safety kernel AND every HAL subscribe to this ``std_msgs/Empty``
    topic and latch immediately, dropping every in-flight and future chunk
    until ``/openral/estop_reset``. No rclpy node here, so it shells out to
    ``ros2 topic pub``, published a few times since a freshly-spawned
    publisher must first discover the subscribers; e-stop is idempotent (a
    latch), so repeats are harmless and this beats the discovery race.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return _error(503, "`ros2` not on PATH; source the workspace install first")
    proc = await asyncio.create_subprocess_exec(
        ros2,
        "topic",
        "pub",
        # CRITICAL: a fresh publisher must WAIT for the subscribers to be
        # discovered before publishing, or the e-stop message is silently lost
        # to the discovery race and the robot never stops (observed live). -w 1
        # blocks until >=1 of the HAL/kernel/runner subscriptions is matched.
        "-w",
        "1",
        "--times",
        "3",
        "--rate",
        "10",
        "/openral/estop",
        "std_msgs/msg/Empty",
        "{}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _error(504, "estop publish timed out after 10 s — is the graph running?")
    if proc.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "error": "ros2 topic pub /openral/estop failed",
                "returncode": proc.returncode,
                "stderr": stderr_b.decode("utf-8", errors="replace").strip(),
            },
            status_code=502,
        )
    return JSONResponse({"ok": True, "accepted": True}, status_code=200)


async def _skill_execute_response(skill_id: str, goal_params_json: str, actor: str) -> JSONResponse:
    """Dispatch an ExecuteRskill action goal (guarded write-control; safety kernel disposes).

    Async accept-then-track: the endpoint returns as soon as the action server
    **accepts** the goal (HTTP 202) rather than blocking on full completion.

    - Returns **HTTP 202** with ``{"status":"accepted","goal_id":"<id>",...}``
      the moment ``Goal accepted with ID: <id>`` appears on stdout.
    - Returns **HTTP 409** if ``Goal was rejected`` appears before acceptance.
    - Returns **HTTP 504** if the action server does not accept within
      ``OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S`` seconds (default 12).
    - Returns **HTTP 502** if the subprocess exits before printing any
      acceptance line.
    - Returns **HTTP 503** if ``ros2`` is not on PATH.

    On 202 a background ``asyncio.Task`` drains the remaining stdout, awaits
    process exit, and audit-logs the eventual outcome.

    Every call is audit-logged at WARNING level before the subprocess is
    spawned; ``actor`` is the token ``sub`` or the client IP. ``goal_params_json``
    is never logged.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return _error(503, "`ros2` not on PATH; source the workspace install first")

    accept_timeout = float(
        os.environ.get(
            "OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S", str(_DEFAULT_SKILL_ACCEPT_TIMEOUT_S)
        )
    )

    goal = json.dumps(
        {
            "rskill_id": skill_id,
            "revision": "",
            "prompt": "",
            "prompt_metadata_json": "",
            "goal_params_json": goal_params_json,
            "deadline_s": 0.0,
        }
    )
    _logger.warning(
        "dashboard_write_attempt",
        operation="skill_execute",
        outcome="sent",
        skill_id=skill_id,
        actor=actor,
    )

    proc = await asyncio.create_subprocess_exec(
        ros2,
        "action",
        "send_goal",
        "/openral/execute_rskill",
        "openral_msgs/action/ExecuteRskill",
        goal,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    # Capture stdout narrowed to StreamReader — PIPE was requested so it is always set.
    if proc.stdout is None:  # pragma: no branch
        raise RuntimeError("stdout is None despite asyncio.subprocess.PIPE request")
    stdout = proc.stdout

    # ── Phase 1: read lines until accepted / rejected / timeout / EOF ─────────
    try:
        goal_id, rejected, captured_lines = await asyncio.wait_for(
            _read_goal_decision(stdout), timeout=accept_timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _error(
            504,
            f"action server did not accept the goal within {accept_timeout:.0f} s "
            "(OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S)",
        )

    # ── Phase 2a: goal rejected ───────────────────────────────────────────────
    if rejected:
        await proc.wait()
        _logger.warning(
            "dashboard_write_attempt",
            operation="skill_execute",
            outcome="rejected",
            skill_id=skill_id,
            actor=actor,
        )
        return _error(409, "action server rejected the goal")

    # ── Phase 2b: proc exited before any accepted/rejected line ──────────────
    if goal_id is None:
        # EOF without matching either marker — subprocess crashed or exited early.
        await proc.wait()
        out = "\n".join(captured_lines)
        return JSONResponse(
            {
                "ok": False,
                "error": "ros2 action send_goal exited before goal was accepted",
                "output": out,
            },
            status_code=502,
        )

    # ── Phase 3: goal accepted — log it, schedule background drain ───────────
    # Rebind to str so mypy knows it's not None inside the inner closure below.
    accepted_goal_id: str = goal_id
    _logger.warning(
        "dashboard_write_attempt",
        operation="skill_execute",
        outcome="accepted",
        skill_id=skill_id,
        goal_id=accepted_goal_id,
        actor=actor,
    )

    _in_background(
        _drain_skill_goal_stdout(stdout, proc, captured_lines, accepted_goal_id, skill_id)
    )

    return JSONResponse(
        {
            "ok": True,
            "goal_id": accepted_goal_id,
            "detail": "skill dispatched; track progress via dashboard telemetry",
        },
        status_code=202,
    )


async def _read_body(request: Request) -> bytes:
    """Read the request body, decompressing gzip if the encoding header says so.

    The OTel HTTP exporter compresses by default when ``--insecure`` is
    set; we honour both gzip and identity. Anything else is read as-is.
    Both the raw and the decoded size are capped at ``_MAX_BODY_BYTES`` while
    reading, so neither a huge upload nor a gzip bomb is buffered whole.

    Raises:
        HTTPException: 413 when the raw or decoded body exceeds the budget,
            400 when the gzip stream is corrupt.
    """
    limit = _MAX_BODY_BYTES
    too_large = HTTPException(status_code=413, detail=f"request body exceeds {limit} bytes")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_large
    if request.headers.get("content-encoding", "").lower() != "gzip":
        return bytes(body)
    out = bytearray()
    data = bytes(body)
    try:
        while data:  # gzip allows concatenated members
            d = zlib.decompressobj(wbits=31)
            out += d.decompress(data, limit + 1 - len(out))
            if len(out) > limit or d.unconsumed_tail:
                raise too_large
            if not d.eof:
                raise HTTPException(status_code=400, detail="truncated gzip body")
            data = d.unused_data
    except zlib.error as exc:
        raise HTTPException(status_code=400, detail=f"gzip decode failed: {exc}") from exc
    return bytes(out)


def _protobuf_response(message: Any) -> Response:
    return Response(
        content=message.SerializeToString(),
        media_type="application/x-protobuf",
        status_code=200,
    )


def _bad_request(detail: str) -> Response:
    return _error(400, detail)


def _camera_thumb(store: TelemetryStore, source: str) -> str | None:
    """Return the latest base64 JPEG thumbnail for ``source``, or ``None``."""
    cameras = store.snapshot().get("topics", {}).get("perception", {}).get("cameras", {})
    entry = cameras.get(source)
    if not isinstance(entry, dict):
        return None
    thumb = entry.get("thumbnail_jpeg_b64")
    return thumb if isinstance(thumb, str) and thumb else None


async def _announce_operator_action(
    action: str, principal: Principal, client_ip: str, detail: str = ""
) -> None:
    """Announce a dashboard action on ``/openral/operator_action`` (issue #468), best effort.

    The compliance recorder records the effect either way (a reset is the latch
    clearing); this says it came through the dashboard and from whom. The identity is
    the openral.com user (the token's ``sub``, its ``jti`` in ``detail``) on the remote
    listener, else the client IP; either way only its HMAC under ``$OPENRAL_ACTOR_KEY``
    is sent, never the identity in clear. The channel is ``remote`` for the tunnel.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return
    key = os.environ.get(ACTOR_KEY_ENV, "")
    identity = principal.sub or client_ip
    actor_id = pseudonymous_actor_id(identity, key) if key and identity else ""
    if principal.jti:
        detail = f"{detail} jti={principal.jti}".strip()
    payload = json.dumps(
        {
            "action": action,
            "channel_label": "remote" if principal.mode == "remote" else "dashboard",
            "actor_id": actor_id,
            "detail": detail,
        }
    )
    proc = await asyncio.create_subprocess_exec(
        ros2,
        "topic",
        "pub",
        "--once",
        "--wait-matching-subscriptions",
        "1",
        "/openral/operator_action",
        "openral_msgs/msg/OperatorAction",
        payload,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except TimeoutError:  # no recorder up: the action still happens, unattributed
        proc.kill()
        await proc.wait()


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status, headers=headers)


def _served_manifest(manifest: RobotManifest, principal: Principal) -> dict[str, Any]:
    """The manifest as this dashboard enforces it right now.

    The emitter read the write-controls / remote-control flags from its own env; this
    dashboard's flags are the ones that decide, so skills drop out with write controls
    off, ``remote`` follows the remote-control flag, and a LAN client sees only E-STOP.
    """
    remote_control = _remote_control_enabled()
    actions = [
        a.model_copy(update={"remote": a.kind == "estop" or remote_control})
        for a in manifest.actions
        if (a.kind != "skill" or _write_controls_enabled())
        and (principal.mode != "lan" or a.kind == "estop")
    ]
    served = manifest.model_copy(update={"actions": actions})
    return served.model_dump(mode="json", by_alias=True)


def _settings_body(principal: Principal) -> dict[str, object]:
    """``GET /api/settings``: openral.com's settings shape, for this robot's own file."""
    return {
        "scope": "robot",
        "canWrite": principal.mode == "local"
        and ROLE_RANK[principal.role] >= ROLE_RANK["operator"],
        "settings": SETTINGS,
        "values": shown(load_settings()),
    }


def _action_refusal(action: ManifestAction, principal: Principal) -> str | None:
    """Why ``principal`` may not run ``action`` (ADR-0129 §6), or ``None``."""
    if action.kind == "estop":
        return None  # every role, every listener: a stop is never refused
    if principal.mode == "lan":
        return "a LAN client may only press E-STOP"
    # The manifest may raise an action's role, never lower it under operator: only a stop
    # is a viewer's (a manifest is runtime data, and a viewer must not drive the robot).
    needed = max(action.min_role, "operator", key=ROLE_RANK.__getitem__)
    if ROLE_RANK[principal.role] < ROLE_RANK[needed]:
        return f"role {principal.role!r} may not run {action.id!r} (needs {needed})"
    if principal.mode == "remote" and not _remote_control_enabled():
        return f"remote control is off on this robot (set {REMOTE_CONTROL_ENV}=1 to allow it)"
    if action.kind == "skill" and not _write_controls_enabled():
        return f"write controls are off on this robot (set {WRITE_CONTROLS_ENV}=1)"
    return None


async def _run_action(  # noqa: PLR0911  # reason: one return per action kind / failure
    request: Request, action: ManifestAction
) -> JSONResponse:
    """Dispatch an authorised action; E-STOP first, before anything that could fail."""
    app_state = request.app.state
    principal: Principal = request.state.principal
    client_ip = request.client.host if request.client else ""
    if action.kind == "estop":
        # The dashboard issued the stop, so it knows the latch is set; the kernel stops
        # emitting the safety.check spans that would otherwise flip the flag.
        app_state.store.set_estopped(True)
        est = app_state.estop
        if est is not None and est.available:
            est.trigger()
            resp = JSONResponse({"ok": True, "accepted": True})
        else:
            resp = await _estop_trigger_response()
        # Announced after the stop, never before: attribution must not delay it.
        _in_background(_announce_operator_action("estop", principal, client_ip))
        return resp
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return _error(400, f"invalid json: {exc}")
    args = body.get("args", {}) if isinstance(body, dict) else None
    if not isinstance(args, dict):
        return _error(400, 'body must be {"args": {...}}')
    if action.args_schema is not None:
        try:
            jsonschema.validate(args, action.args_schema)
        except jsonschema.ValidationError as exc:
            return _error(400, f"invalid args: {exc.message}")
    if action.kind == "estop_reset":
        await _announce_operator_action("estop_reset", principal, client_ip)
        resp = await _estop_reset_response(app_state.estop)
        # Clear the latch flag only when the kernel accepted the reset; a cooldown
        # rejection (409) leaves the robot latched.
        if resp.status_code == http.HTTPStatus.OK:
            app_state.store.set_estopped(False)
        return resp
    if action.kind == "prompt":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error(400, "invalid args: 'text' must be a non-empty string")
        await _announce_operator_action("prompt", principal, client_ip)
        return await _prompt_response(text)
    rskill_id = action.id.removeprefix("skill:")
    await _announce_operator_action("skill_execute", principal, client_ip, rskill_id)
    actor = principal.sub or client_ip
    return await _skill_execute_response(rskill_id, json.dumps(args) if args else "", actor)


def _without_images(snapshot: dict[str, Any]) -> dict[str, Any]:
    """``snapshot`` minus the camera thumbnails (``?images=0``; cameras are polled as JPEGs).

    Copies the touched levels: the same snapshot object goes to every subscriber.
    """
    topics = snapshot.get("topics", {})
    cameras = topics.get("perception", {}).get("cameras")
    if not cameras:
        return snapshot
    stripped = {
        n: {k: v for k, v in c.items() if k != "thumbnail_jpeg_b64"} for n, c in cameras.items()
    }
    perception = {**topics["perception"], "cameras": stripped}
    return {**snapshot, "topics": {**topics, "perception": perception}}


# The robot UI and the fallback page: their own scripts only, styles may be inline (React
# sets element styles), images from blob: snapshots, the API and the /ws bridge on this
# origin only ('self' covers ws:/wss: to the same host, CSP Level 3); the page draws the
# 3D scene itself from /ws; /viz/ is Lichtblick, opened in its own tab; framed only here
# and by openral.com.
_UI_CSP = (
    "default-src 'self'; script-src 'self'{inline}; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; connect-src 'self'; frame-src 'self'; "
    "font-src 'self'; object-src 'none'; base-uri 'self'; "
    "frame-ancestors 'self' https://www.openral.com https://dev.openral.com"
)
# Lichtblick: WebAssembly ('wasm-unsafe-eval'), eval ('unsafe-eval': v1.30 refuses to start
# without `new Function` — canRenderApp.ts — and protobufjs decodes by generated code), blob:
# workers, and the one inline script of its index.html, allowed by hash; framed by the UI.
# connect-src is this origin only: a crafted ``?ds.url=wss://elsewhere`` link cannot point
# the viewer at a foreign bridge (or fetch a foreign file), only at this robot's /ws.
_VIZ_CSP = (
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'{inline}; "
    "worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; "
    "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'none'; frame-ancestors 'self'"
)
_INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>")


# Lichtblick's index.html default-layout slot (its Docker entrypoint fills the same one).
_LAYOUT_PLACEHOLDER = "/*LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER*/"


def _csp(policy: str, html: str = "") -> str:
    """``policy`` with ``html``'s inline scripts allowed by sha256."""
    inline = "".join(
        f" 'sha256-{base64.b64encode(hashlib.sha256(m.encode()).digest()).decode()}'"
        for m in _INLINE_SCRIPT.findall(html)
    )
    return policy.format(inline=inline)


def _with_csp(target: Path, policy: str) -> FileResponse:
    """Serve ``target`` under ``policy``, allowing an HTML file's own inline scripts by hash."""
    html = target.read_text(encoding="utf-8") if target.suffix == ".html" else ""
    return FileResponse(target, headers={"Content-Security-Policy": _csp(policy, html)})


def _robot_layout(manifest: RobotManifest | None) -> dict[str, Any] | None:
    """This robot's Lichtblick layout (its cameras, its base frame), or ``None`` without one.

    Built by the generator the deploy launch uses for Foxglove Studio, a colcon package
    importable once the workspace overlay is sourced.
    """
    if manifest is None:
        return None
    try:
        layout_mod = importlib.import_module("openral_foxglove_bringup.layout")
    except ImportError:
        _logger.warning("dashboard.viz_layout_unavailable", reason="source the workspace overlay")
        return None
    cameras = [p.source for p in manifest.panels if p.type == "camera" and p.source]
    layout: dict[str, Any] = layout_mod.build_layout(cameras, follow_frame=manifest.base_frame)
    return layout


def _viz_index(root: Path, manifest: RobotManifest | None) -> Response:
    """Lichtblick's index.html with this robot's layout in its default-layout slot.

    The layout rides in the page, not a ``layoutUrl``: one request, and the CSP's hash
    covers the inline script as served. ``<`` is escaped so no string can end the script.
    """
    html = (root / "index.html").read_text(encoding="utf-8")
    layout = _robot_layout(manifest)
    if layout is not None:
        html = html.replace(_LAYOUT_PLACEHOLDER, json.dumps(layout).replace("<", "\\u003c"), 1)
    return HTMLResponse(
        html,
        headers={"Content-Security-Policy": _csp(_VIZ_CSP, html), "Cache-Control": "no-store"},
    )


def _file_in(root: Path, rel: str, policy: str) -> Response:
    """``root/rel`` under CSP ``policy``, refusing anything outside ``root``."""
    base = root.resolve()
    target = (base / (rel or "index.html")).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return _error(404, "not found")
    return _with_csp(target, policy)


async def _gate(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Who is asking, and may they: runs before every HTTP route (WebSockets check in ``/ws``)."""
    if is_remote(request.scope):
        return await _gate_remote(request, call_next)
    client = request.client.host if request.client else ""
    if _via_tunnel(request.headers):
        _logger.warning("dashboard.tunnel_headers_on_local_listener", path=request.url.path)
        return _error(403, "a forwarded request on the local listener (tunnel misrouted?)")
    loopback = _is_loopback(client)
    if loopback and not _local_host(request.headers.get("host", ""), request.app.state.local_hosts):
        _logger.warning("dashboard.host_refused", host=request.headers.get("host"))
        return _error(403, "unexpected Host header (DNS rebinding?)")
    request.state.principal = (
        Principal(mode="local", role="operator")
        if loopback
        else Principal(mode="lan", role="viewer")
    )
    if request.method not in _SAFE_METHODS:
        # CSRF: a page on another site driving a local operator's browser. ponytail:
        # Origin check only — a DNS-rebinding page is same-origin by construction.
        refusal = (
            "cross-origin request refused"
            if _cross_origin(request)
            else None
            if loopback
            or request.url.path in _LAN_WRITES
            or (request.url.path.startswith("/v1/") and _lan_ingest_enabled())
            else f"OTLP ingest from the LAN is off (set {LAN_INGEST_ENV}=1)"
            if request.url.path.startswith("/v1/")
            else "read-only from the LAN: only E-STOP"
        )
        if refusal is not None:
            _logger.warning(
                "dashboard_write_attempt",
                operation=request.url.path,
                outcome="denied",
                reason=refusal,
                origin=request.headers.get("origin"),
                operator_ip=client,
            )
            return _error(403, refusal)
    return await call_next(request)


async def _gate_remote(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """The tunnel listener: CORS allowlist, OPTIONS before auth, ``/v1`` refused, Bearer JWT."""
    origin = request.headers.get("origin")
    cors = (
        {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
        if origin is not None and origin in allowed_origins()
        else {}
    )
    if origin is not None and not cors:
        return _error(403, "origin not allowed")
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                **cors,
                "Access-Control-Allow-Methods": "GET, POST",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Max-Age": "600",
            },
        )
    path = request.url.path
    if path == "/v1" or path.startswith("/v1/"):
        return _error(404, "not found", cors)
    verifier: TokenVerifier | None = request.app.state.verifier
    token = bearer(request.headers.get("authorization"))
    if verifier is None:
        result: Principal | str = "this robot is not enrolled"
    elif token is None:
        result = "Authorization: Bearer <token> required"
    else:
        result = await verifier.verify(token)
    # One write per token (a stop is never refused): a replayed token cannot prompt,
    # reset, run a skill or mint a 3D ticket twice. 401 makes the console mint afresh.
    if (
        not isinstance(result, str)
        and verifier is not None
        and request.method not in _SAFE_METHODS
        and path != "/api/actions/estop"
        and not verifier.spend(result)
    ):
        result = "token already used for a write"
    if isinstance(result, str):
        _logger.warning("dashboard.remote_auth_refused", path=path, reason=result)
        return _error(401, result, {**cors, "WWW-Authenticate": 'Bearer error="invalid_token"'})
    request.state.principal = result
    return _with_headers(await call_next(request), cors)


def _with_headers(resp: Response, headers: dict[str, str]) -> Response:
    resp.headers.update(headers)
    return resp


async def _relay_to_browser(
    upstream: ClientConnection, websocket: WebSocket, relay: BridgeRelay
) -> None:
    """Read the bridge into ``relay`` and send what it releases while the browser keeps up.

    While a send is in flight the reader keeps going, so newer snapshot frames replace
    older unsent ones. Ordered frames are bounded: when the relay is ``full()`` the reader
    waits for the browser, which hands the backpressure back to ``foxglove_bridge``.
    """
    ready = asyncio.Event()
    sent = asyncio.Event()

    async def read_bridge() -> None:
        async for message in upstream:
            relay.from_bridge(message)
            ready.set()
            while relay.full():
                sent.clear()
                await sent.wait()

    reader = asyncio.create_task(read_bridge())
    try:
        while True:
            waiter = asyncio.create_task(ready.wait())
            try:
                await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            ready.clear()
            for frame in relay.drain():
                if isinstance(frame, bytes):
                    await websocket.send_bytes(frame)
                else:
                    await websocket.send_text(frame)
            sent.set()
            if reader.done() and not relay.pending():
                reader.result()  # re-raise a bridge error
                return
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        if relay.dropped:
            # Explicit, not silent: how many stale snapshot frames this viewer never got.
            _logger.info("dashboard.bridge_relay_coalesced", dropped=relay.dropped)


async def _proxy_bridge(websocket: WebSocket, url: str, expires_at: float | None) -> None:
    """Pump frames between the browser and ``foxglove_bridge`` until a side or the ticket ends."""
    import websockets
    from websockets.typing import Subprotocol

    offered = [Subprotocol(p) for p in websocket.scope.get("subprotocols", [])]
    try:
        upstream = await websockets.connect(
            url, subprotocols=offered or None, ping_interval=30, max_size=None
        )
    except (OSError, websockets.InvalidHandshake, TimeoutError) as exc:
        _logger.warning("dashboard.bridge_unavailable", url=url, error=str(exc))
        await websocket.accept()
        await websocket.close(code=_WS_UNAVAILABLE, reason="foxglove_bridge is not running")
        return

    # Latest-wins for snapshot topics: a slow browser link (the tunnel) gets the
    # newest image / cloud / map instead of a growing backlog (bridge_relay).
    relay = BridgeRelay()

    async def downstream() -> None:
        await _relay_to_browser(upstream, websocket, relay)

    async def upstream_pump() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes") if message.get("bytes") is not None else message.get("text")
            if data is not None:
                relay.from_browser(data)
                await upstream.send(data)

    async with upstream:
        await websocket.accept(subprotocol=upstream.subprotocol)
        tasks = {asyncio.create_task(downstream()), asyncio.create_task(upstream_pump())}
        timeout = None if expires_at is None else max(0.0, expires_at - time.time())
        done, pending = await asyncio.wait(
            tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        # Await the cancelled pump so it never outlives the socket (or the event loop).
        await asyncio.gather(*pending, return_exceptions=True)
        expired = not done
        with contextlib.suppress(RuntimeError, OSError):
            await websocket.close(
                code=_WS_POLICY_VIOLATION if expired else 1000,
                reason="viz ticket expired" if expired else "",
            )


def create_app(  # noqa: PLR0915  # reason: flat FastAPI route-registration table; each endpoint is one statement
    store: TelemetryStore | None = None,
    *,
    verifier: TokenVerifier | None = None,
    bridge_url: str = _DEFAULT_BRIDGE_URL,
) -> FastAPI:
    """Build the FastAPI app bound to ``store`` (a fresh one if ``None``).

    The returned app is a normal ASGI application; mount it under any server
    (uvicorn, hypercorn, ...) or test transport. Wrap it in
    ``remote_auth.remote_listener`` for the tunnel listener, which then needs
    ``verifier`` (built from the robot's enrolment) to admit anyone.

    Args:
        store: The telemetry store (``app.state.store``).
        verifier: Checks openral.com tokens on the remote listener; ``None`` refuses all.
        bridge_url: ``foxglove_bridge``'s WebSocket, which ``/ws`` proxies.
    """
    store = store if store is not None else TelemetryStore()
    app = FastAPI(title="OpenRAL Dashboard", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.verifier = verifier
    app.state.viz_tickets = VizTickets()
    app.state.local_hosts = set(_LOCAL_HOSTS)
    app.state.bridge_url = bridge_url
    # The robot UI (ui_assets); run_dashboard refreshes it after fetching the pin.
    app.state.ui_dir = ui_dir()
    app.state.viz_dir = None
    app.state.viz_lock = asyncio.Lock()
    # Set by run_dashboard when mDNS discovery is wired; None in tests / when
    # the 'mdns' extra is absent. The /api/robots endpoint tolerates both.
    app.state.discovery = None
    # Persistent e-stop publisher (set by run_dashboard). None here so tests /
    # standalone use fall back to the shell-out path.
    app.state.estop = None
    # Latched /openral/safety_status subscriber (set by run_dashboard, ADR-0096).
    # None without ROS; the Safety Status panel then renders "waiting" rather
    # than claiming the robot is clear.
    app.state.safety_status = None
    # /openral/perception/objects overlay subscriber (set by run_dashboard).
    # None without ROS; the camera tiles then simply carry no overlays.
    app.state.perception_overlay = None

    app.middleware("http")(_gate)

    @app.post(
        "/v1/traces",
        response_class=Response,
        responses={200: {"content": {"application/x-protobuf": {}}}},
    )
    async def post_traces(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        body = await _read_body(request)
        try:
            req = ExportTraceServiceRequest.FromString(body)
        except DecodeError as exc:
            return _bad_request(f"trace decode failed: {exc}")
        request.app.state.store.ingest_spans(list(req.resource_spans))
        return _protobuf_response(ExportTraceServiceResponse())

    @app.post(
        "/v1/metrics",
        response_class=Response,
        responses={200: {"content": {"application/x-protobuf": {}}}},
    )
    async def post_metrics(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        body = await _read_body(request)
        try:
            req = ExportMetricsServiceRequest.FromString(body)
        except DecodeError as exc:
            return _bad_request(f"metric decode failed: {exc}")
        request.app.state.store.ingest_metrics(list(req.resource_metrics))
        return _protobuf_response(ExportMetricsServiceResponse())

    @app.post(
        "/v1/logs",
        response_class=Response,
        responses={200: {"content": {"application/x-protobuf": {}}}},
    )
    async def post_logs(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        # issue #318 — the structlog→OTel bridge feeds the event log; the robot
        # manifest (ADR-0129) arrives here too, as an `openral.robot.manifest` record.
        body = await _read_body(request)
        try:
            req = ExportLogsServiceRequest.FromString(body)
        except DecodeError as exc:
            return _bad_request(f"log decode failed: {exc}")
        client = request.client.host if request.client else ""
        request.app.state.store.ingest_logs(
            list(req.resource_logs), trust_manifest=_is_loopback(client)
        )
        return _protobuf_response(ExportLogsServiceResponse())

    @app.get("/api/traces")
    async def get_traces(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Indexed trace_ids so `openral replay` can pick a trace when `--trace` is
        # omitted. Bounded; see _TRACE_INDEX_MAX_TRACES in store.py.
        return JSONResponse({"traces": request.app.state.store.list_traces()})

    @app.get("/api/spans/{trace_id}")
    async def get_spans(  # pyright: ignore[reportUnusedFunction]
        trace_id: str, request: Request
    ) -> JSONResponse:
        # 404 when the trace has not been ingested (or was evicted from the index).
        spans = request.app.state.store.lookup_trace(trace_id)
        if spans is None:
            return JSONResponse({"error": "trace not indexed"}, status_code=404)
        return JSONResponse({"trace_id": trace_id, "spans": spans})

    @app.get("/api/state")
    async def get_state(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(request.app.state.store.snapshot())

    @app.get("/api/stream")
    async def get_stream(  # pyright: ignore[reportUnusedFunction]
        request: Request, images: bool = True
    ) -> StreamingResponse:
        return StreamingResponse(
            _sse_stream(request, images=images),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/camera/{source}/snapshot.jpg")
    async def get_camera_snapshot(  # pyright: ignore[reportUnusedFunction]
        source: str, request: Request
    ) -> Response:
        # The latest frame of one camera as a JPEG. Polled, never streamed: video
        # over a free Cloudflare Tunnel hostname is against its terms (ADR-0129 §3).
        thumb = _camera_thumb(request.app.state.store, source)
        if thumb is None:
            return _error(404, f"no frame from camera {source!r}")
        try:
            jpeg = base64.b64decode(thumb, validate=True)
        except binascii.Error:
            _logger.warning("dashboard.snapshot_decode_failed", source=source)
            return _error(404, f"no decodable frame from camera {source!r}")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/robots")
    async def get_robots(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # issue #75b — mDNS-discovered OpenRAL services on the LAN. Read-only.
        discovery = getattr(request.app.state, "discovery", None)
        if discovery is None:
            return JSONResponse({"enabled": False, "robots": []})
        return JSONResponse(
            {
                "enabled": discovery.enabled,
                "robots": [r.model_dump() for r in discovery.robots()],
            }
        )

    @app.get("/api/settings")
    async def get_settings(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # The reasoner settings kept on this robot (robot_settings): never over the tunnel
        # (openral.com has its own), secrets only as "set ...abcd".
        principal: Principal = request.state.principal
        if principal.mode == "remote":
            return _error(403, "this robot's own settings are set at the robot")
        return JSONResponse(_settings_body(principal))

    @app.put("/api/settings")
    async def put_setting(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        principal: Principal = request.state.principal
        if principal.mode != "local" or ROLE_RANK[principal.role] < ROLE_RANK["operator"]:
            _logger.warning(
                "dashboard_write_attempt",
                operation="settings",
                outcome="denied",
                reason="settings are set by the local operator",
                mode=principal.mode,
                role=principal.role,
            )
            return _error(403, "only the operator at the robot can change its settings")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "expected JSON {name, value}")
        name = body.get("name") if isinstance(body, dict) else None
        value = body.get("value", "") if isinstance(body, dict) else None
        if not isinstance(name, str) or not isinstance(value, str):
            return _error(400, "expected JSON {name, value}")
        try:
            save_setting(name, value)
        except ValueError as exc:
            return _error(400, str(exc))
        except OSError as exc:
            return _error(500, f"could not save the setting: {exc.strerror or exc}")
        # Audited without the value.
        _logger.info("dashboard.setting_written", name=name, cleared=not value.strip())
        return JSONResponse(_settings_body(principal))

    @app.get("/api/config")
    async def get_config(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Who the caller is and what this robot allows (ADR-0129 Interface A).
        principal: Principal = request.state.principal
        verifier_: TokenVerifier | None = request.app.state.verifier
        manifest: RobotManifest | None = request.app.state.store.manifest()
        robot_id = (
            verifier_.enrolment.robot
            if verifier_ is not None
            else manifest.robot_id
            if manifest is not None
            else None
        )
        return JSONResponse(
            {
                "mode": principal.mode,
                "role": principal.role,
                "robot_id": robot_id,
                # The owning org's slug from the robot's enrolment ("" for a free or
                # unlinked robot): the console's left pane names whose robot this is.
                "org": verifier_.enrolment.org if verifier_ is not None else "",
                "write_controls": _write_controls_enabled(),
                "remote_control": _remote_control_enabled(),
                "jaeger_ui_url": os.environ.get("OPENRAL_JAEGER_UI_URL", "").rstrip("/"),
            }
        )

    @app.get("/api/manifest")
    async def get_manifest(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        manifest: RobotManifest | None = request.app.state.store.manifest()
        if manifest is None:
            return _error(404, "the runtime has not sent a robot manifest yet")
        return JSONResponse(_served_manifest(manifest, request.state.principal))

    @app.post("/api/actions/{action_id:path}")
    async def post_action(  # pyright: ignore[reportUnusedFunction]
        action_id: str, request: Request
    ) -> JSONResponse:
        principal: Principal = request.state.principal
        manifest: RobotManifest | None = request.app.state.store.manifest()
        action = (
            _ESTOP
            if action_id == "estop"
            else next((a for a in manifest.actions if a.id == action_id), None)
            if manifest is not None
            else None
        )
        if action is None:
            return _error(404, f"no action {action_id!r} on this robot")
        refusal = _action_refusal(action, principal)
        if refusal is not None:
            _logger.warning(
                "dashboard_write_attempt",
                operation=action_id,
                outcome="denied",
                reason=refusal,
                mode=principal.mode,
                role=principal.role,
            )
            return _error(403, refusal)
        return await _run_action(request, action)

    @app.post("/api/transcribe")
    async def post_transcribe(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator speech to text on the host (local Whisper); the audio never leaves it.
        # A step towards a prompt, so a prompt's rules: Whisper is not for viewers to run.
        refusal = _action_refusal(_TRANSCRIBE, request.state.principal)
        if refusal is not None:
            return _error(403, refusal)
        audio = await _read_body(request)
        return await _transcribe_response(audio)

    @app.post("/api/viz/ticket")
    async def post_viz_ticket(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        tickets: VizTickets = request.app.state.viz_tickets
        ticket = tickets.issue(request.state.principal)
        expires_at = tickets.expires_at(ticket, remote=is_remote(request.scope)) or time.time()
        return JSONResponse({"ticket": ticket, "expires_in": int(expires_at - time.time())})

    @app.websocket("/ws")
    async def ws_proxy(  # pyright: ignore[reportUnusedFunction]
        websocket: WebSocket, ticket: str | None = None
    ) -> None:
        # The 3D view's link to foxglove_bridge (loopback, read-only). The remote
        # listener and LAN clients need a viz ticket; loopback may omit it.
        remote = is_remote(websocket.scope)
        if not remote and _via_tunnel(websocket.headers):
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return
        origin = websocket.headers.get("origin")
        if origin is not None and (
            origin not in allowed_origins()
            if remote
            else urlsplit(origin).netloc.lower() != websocket.headers.get("host", "").lower()
        ):
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return
        tickets: VizTickets = websocket.app.state.viz_tickets
        expires_at = tickets.expires_at(ticket, remote=remote)
        client = websocket.client.host if websocket.client else ""
        if expires_at is None and (
            remote
            or not _is_loopback(client)
            or not _local_host(websocket.headers.get("host", ""), websocket.app.state.local_hosts)
        ):
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return
        await _proxy_bridge(websocket, websocket.app.state.bridge_url, expires_at)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.get("/viz/{path:path}")
    async def get_viz(path: str, request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        # Lichtblick, fetched on first use (viz_assets); 503 when offline.
        state = request.app.state
        async with state.viz_lock:
            if state.viz_dir is None:
                from openral_observability.dashboard.viz_assets import ensure_viz_assets

                state.viz_dir = await asyncio.to_thread(ensure_viz_assets)
        if state.viz_dir is None:
            return _error(503, "the 3D view could not be downloaded (offline?)")
        # Files are read off the event loop: it also serves E-STOP.
        if path in ("", "index.html"):
            return await asyncio.to_thread(_viz_index, state.viz_dir, state.store.manifest())
        return await asyncio.to_thread(_file_in, state.viz_dir, path, _VIZ_CSP)

    @app.get("/{path:path}")
    async def get_ui(path: str, request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        # The robot UI; the built-in status + E-STOP page when none is installed.
        root: Path | None = request.app.state.ui_dir
        if root is None:
            if path in ("", "index.html"):
                return await asyncio.to_thread(_with_csp, _STATIC_DIR / "fallback.html", _UI_CSP)
            return _error(404, "not found")
        return await asyncio.to_thread(_file_in, root, path, _UI_CSP)

    return app


async def _sse_stream(request: Request, *, images: bool = True) -> AsyncIterator[bytes]:
    """Server-Sent Events generator for ``/api/stream``.

    Emits an initial snapshot so a fresh client sees the current state without
    waiting for the next ingest, then every delta from the store's subscription
    queue, with a ``: keepalive`` comment every 15 s. ``images=False`` drops the
    camera thumbnails. On the remote listener the stream ends at the token's
    ``exp``: the client reconnects with a fresh token.
    """
    store: TelemetryStore = request.app.state.store
    deadline: float | None = request.state.principal.exp

    def frame(payload: dict[str, Any]) -> bytes:
        return _sse_frame(payload if images else _without_images(payload))

    queue = store.subscribe()
    try:
        yield frame(store.snapshot())
        while True:
            if await request.is_disconnected():
                return
            wait_s = 15.0
            if deadline is not None:
                wait_s = min(wait_s, deadline - time.time())
                if wait_s <= 0:
                    return
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=wait_s)
            except TimeoutError:
                yield b": keepalive\n\n"
                continue
            yield frame(payload)
    finally:
        store.unsubscribe(queue)


def _sse_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, default=str)
    return f"data: {body}\n\n".encode()
