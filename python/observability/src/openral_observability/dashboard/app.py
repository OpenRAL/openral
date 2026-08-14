"""FastAPI app for the live dashboard.

The same ASGI app serves three things on one port:

* ``/``, ``/static/*`` — the single-page UI.
* ``/api/state`` (JSON) and ``/api/stream`` (SSE) — read endpoints
  consumed by the page's JavaScript.
* ``/v1/traces``, ``/v1/metrics``, ``/v1/logs`` — OTLP/HTTP protobuf
  receivers. Any OpenRAL workload pointed at this port via
  ``OTEL_EXPORTER_OTLP_ENDPOINT=http://<host>:<port>`` +
  ``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`` will stream live into
  the dashboard.
* ``POST /api/prompt`` — operator-driven write endpoint that shells
  out to ``openral prompt`` (the operator-prompt dispatch path) targeting
  the prompt-router's ``dashboard`` source.
* ``POST /api/estop_reset`` — operator recovery: clears a latched safety
  e-stop by calling the kernel's ``/openral/estop_reset`` (std_srvs/Trigger)
  service via ``ros2 service call``, so the robot can be re-tasked after an
  e-stop. Returns 409 when the kernel rejects it (post-estop cooldown).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import gzip
import http
import io
import json
import mimetypes
import os
import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.protobuf.message import DecodeError
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

from openral_observability.dashboard.store import TelemetryStore

__all__ = ["create_app"]

_logger = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _write_controls_enabled() -> bool:
    """Whether guarded write-controls are on (default OFF)."""
    return os.environ.get("OPENRAL_DASHBOARD_WRITE_CONTROLS", "") == "1"


# Param-name substrings that may affect safety; the dashboard refuses to tune
# these (CLAUDE.md §1.1 — never lower a velocity limit without a paper trail).
#
# Fail-closed intent: when in doubt, refuse.  Both the compact form (e.g.
# "estop") AND the ROS 2 underscored/prefixed form (e.g. "e_stop") are listed
# so that names like "e_stop_enable", "dead_man_timeout", "safe_mode", and
# "safe_zone_radius" are all caught by substring matching.  "safe" subsumes
# "safety" but both are kept for clarity; "e_stop" subsumes "estop" (which
# never appears in practice) but both are kept for the same reason.
_SAFETY_PARAM_DENYLIST = (
    "velocity",
    "accel",
    "force",
    "torque",
    "limit",
    "workspace",
    "estop",
    "e_stop",  # ROS 2 underscored form: e_stop_enable, e_stop_triggered, …
    "deadman",
    "dead_man",  # ROS 2 underscored form: dead_man_timeout, …
    "safety",
    "safe",  # catches safe_mode, safe_zone_radius, safe_dist, … (and safety)
    "watchdog",
)
_MJPEG_BOUNDARY = "frame"

# Background drain tasks created by _skill_execute_response are stored here so
# the event loop does not GC them before they finish.  Each task removes itself
# from this set in its own done-callback.
_BACKGROUND_DRAIN_TASKS: set[asyncio.Task[None]] = set()

# Regex to extract the goal ID from `ros2 action send_goal` stdout.
_GOAL_ACCEPTED_RE = re.compile(r"Goal accepted with ID:\s*(\S+)")
_GOAL_REJECTED_PHRASE = "Goal was rejected"

# Default acceptance timeout in seconds (overridden by env).
_DEFAULT_SKILL_ACCEPT_TIMEOUT_S = 12.0


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
    them (never crash the event loop). Called by ``_skill_execute_response``
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


# The vendored voice-prompt assets (static/vendor/vad/) include ESM (.mjs) and
# WebAssembly (.wasm) served by StaticFiles. Browsers reject an ESM dynamic
# import served as text/plain, and streaming wasm compilation wants
# application/wasm — register both so onnxruntime-web loads offline cleanly.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

# ── Operator voice prompt (POST /api/transcribe) ──────────────────────────────
# The dashboard's mic button records until the operator stops speaking
# (browser-side Silero VAD) and POSTs the captured audio here. We transcribe it
# with a local CPU Whisper model (faster-whisper / CTranslate2) and hand the
# text back so the page drops it into the operator-prompt box and sends it. The
# audio never leaves the host. faster-whisper ships with the dashboard extra so
# this works out of the box; the endpoint still degrades to a 503 (rather than
# crashing the dashboard) on the off chance the dependency is ever stripped.
_STT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB guard — a 30 s 16 kHz mono WAV is ~1 MB.
_STT_MODEL_CACHE: dict[str, Any] = {}  # name → WhisperModel; loaded once, reused.


class _STTUnavailableError(RuntimeError):
    """Raised when faster-whisper is not importable in this environment."""


def _transcribe_sync(audio: bytes) -> tuple[str, str]:
    """Load (once, cached) the local Whisper model and transcribe ``audio``.

    Blocking CPU work — call via :func:`asyncio.to_thread`, never on the event
    loop. Returns ``(text, model_name)``. The model, device and compute type
    are env-selectable (``OPENRAL_STT_MODEL`` / ``_DEVICE`` / ``_COMPUTE``),
    defaulting to ``base.en`` on CPU with int8 quantization so it runs on any
    operator host. ``audio`` may be any container PyAV can decode (WAV, WebM/
    Opus, …); faster-whisper resamples to 16 kHz internally.

    Raises:
        _STTUnavailableError: faster-whisper is not importable (it ships with the
            dashboard extra, so this only fires if the dependency was stripped).
    """
    try:
        # ships with the dashboard extra — untyped when present, ImportError if stripped.
        import faster_whisper  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as exc:  # ModuleNotFoundError + a shadowed/None sys.modules entry
        raise _STTUnavailableError from exc
    name = os.environ.get("OPENRAL_STT_MODEL", "base.en")
    model = _STT_MODEL_CACHE.get(name)
    if model is None:
        device = os.environ.get("OPENRAL_STT_DEVICE", "cpu")
        compute = os.environ.get("OPENRAL_STT_COMPUTE", "int8")
        _logger.info("stt.model_load", model=name, device=device, compute_type=compute)
        model = faster_whisper.WhisperModel(name, device=device, compute_type=compute)
        _STT_MODEL_CACHE[name] = model
    segments, _info = model.transcribe(io.BytesIO(audio), beam_size=1)
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
    except Exception as exc:  # decode/runtime errors surface as 422 — logged, never swallowed
        _logger.warning("stt.transcribe_failed", error=str(exc))
        return JSONResponse({"error": f"transcription failed: {exc}"}, status_code=422)
    return JSONResponse({"status": "ok", "text": text, "model": model_name})


async def _prompt_response(text: str) -> JSONResponse:
    """Publish an operator prompt via ``openral prompt``.

    Shells out so the wire shape (PromptStamped on
    ``/openral/prompt_in/dashboard``, fanned out by ``prompt_router_node`` at
    priority 100) matches the CLI exactly. The console script is ``openral``,
    the single entry point. Body lives here to keep ``create_app`` under the
    statement cap.
    """
    openral = shutil.which("openral")
    if openral is None:
        return JSONResponse(
            {"error": "`openral` not on PATH; source the workspace install first"},
            status_code=503,
        )
    proc = await asyncio.create_subprocess_exec(
        openral,
        "prompt",
        text,
        "--topic",
        "/openral/prompt_in/dashboard",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return JSONResponse({"error": "openral prompt timed out after 10 s"}, status_code=504)
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return JSONResponse(
            {"error": "openral prompt failed", "returncode": proc.returncode, "stderr": stderr},
            status_code=502,
        )
    return JSONResponse({"status": "ok", "stdout": stdout, "stderr": stderr})


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
    shell-out fallback. Re-prompt via ``/api/prompt`` once this succeeds.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return JSONResponse(
            {"error": "`ros2` not on PATH; source the workspace install first"},
            status_code=503,
        )
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
        return JSONResponse(
            {"error": "estop_reset timed out after 15 s — is the safety kernel running?"},
            status_code=504,
        )
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return JSONResponse(
            {
                "error": "ros2 service call /openral/estop_reset failed",
                "returncode": proc.returncode,
                "stderr": stderr,
            },
            status_code=502,
        )
    # Trigger response renders as `success=True/False, message='…'`.
    accepted = "success=True" in stdout
    if accepted:
        # The kernel service only clears the KERNEL latch; the HAL + runner latch
        # independently on /openral/estop and had no reset path (they stayed
        # latched until a node restart, so the robot never resumed). Now that the
        # kernel's cooldown-gated reset has succeeded, broadcast
        # /openral/estop_cleared so those nodes clear too — instantly via the
        # persistent publisher when present, else the shell-out fallback. Best
        # effort: a publish failure doesn't undo the kernel reset.
        if estop is not None and estop.available:
            estop.clear()
        else:
            await _publish_estop_cleared()
    return JSONResponse(
        {"status": "ok" if accepted else "rejected", "accepted": accepted, "stdout": stdout},
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
    topic and latch immediately — dropping every in-flight and future chunk
    until ``/openral/estop_reset``. This is the operator's "stop the robot NOW"
    control; it mirrors the physical deadman / hardware-estop publishers. Like
    the reset path the dashboard has no rclpy node, so it shells out to ``ros2
    topic pub``. Published a few times because a freshly-spawned publisher must
    first discover the already-running subscribers; e-stop is idempotent (a
    latch), so repeats are harmless and this beats the discovery race.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return JSONResponse(
            {"error": "`ros2` not on PATH; source the workspace install first"},
            status_code=503,
        )
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
        return JSONResponse(
            {"error": "estop publish timed out after 10 s — is the graph running?"},
            status_code=504,
        )
    if proc.returncode != 0:
        return JSONResponse(
            {
                "error": "ros2 topic pub /openral/estop failed",
                "returncode": proc.returncode,
                "stderr": stderr_b.decode("utf-8", errors="replace").strip(),
            },
            status_code=502,
        )
    return JSONResponse({"status": "ok", "accepted": True}, status_code=200)


def _config_response() -> JSONResponse:
    """Dashboard-level config (Jaeger UI url, write-controls flag, …) sourced from env.

    The UI fetches this once on load to decide whether to enable the
    "open in jaeger" link, whether to reveal the guarded operator
    write-controls panel, and whether the mic button's voice prompt is usable.
    Returning ``""`` (the default) leaves the Jaeger link disabled with a
    helpful tooltip — the previous behaviour of unconditionally linking to
    ``localhost:16686`` produced a broken-link click for every user who
    doesn't run Jaeger locally.
    """
    from openral_observability.dashboard.vad_assets import vad_assets_available

    jaeger_url = os.environ.get("OPENRAL_JAEGER_UI_URL", "").rstrip("/")
    return JSONResponse(
        {
            "jaeger_ui_url": jaeger_url,
            "write_controls_enabled": _write_controls_enabled(),
            "voice_prompt_enabled": vad_assets_available(),
        }
    )


async def _skill_execute_from_request(request: Request) -> JSONResponse:
    """Validate the ``POST /api/skill/execute`` payload and dispatch the action.

    Handles the flag check, JSON decode, field validation, and dispatch in one
    place so the route closure stays under the statement cap (PLR0915).
    Every attempt — permitted or denied — is audit-logged.
    """
    operator_ip = request.client.host if request.client else "unknown"
    if not _write_controls_enabled():
        _logger.warning(
            "dashboard_write_attempt",
            operation="skill_execute",
            outcome="denied_flag_off",
            operator_ip=operator_ip,
        )
        return JSONResponse(
            {
                "error": (
                    "write-controls disabled; set OPENRAL_DASHBOARD_WRITE_CONTROLS=1 "
                    "(pending safety-WG review)"
                )
            },
            status_code=403,
        )
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"invalid json: {exc}"}, status_code=400)
    skill_id = payload.get("skill_id") if isinstance(payload, dict) else None
    if not isinstance(skill_id, str) or not skill_id.strip():
        return JSONResponse({"error": "field 'skill_id' required"}, status_code=400)
    return await _skill_execute_response(
        skill_id.strip(),
        str(payload.get("revision", "") or ""),
        str(payload.get("prompt", "") or ""),
        str(payload.get("goal_params_json", "") or ""),
        operator_ip,
    )


async def _skill_execute_response(
    skill_id: str, revision: str, prompt: str, goal_params_json: str, operator_ip: str
) -> JSONResponse:
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
    process exit, and audit-logs the eventual outcome. The task reference is
    kept in ``_BACKGROUND_DRAIN_TASKS`` so the event loop cannot GC it before
    completion.

    Every call is audit-logged at WARNING level before the subprocess is
    spawned. ``prompt`` and ``goal_params_json`` are never logged.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return JSONResponse(
            {"error": "`ros2` not on PATH; source the workspace install first"},
            status_code=503,
        )

    accept_timeout = float(
        os.environ.get(
            "OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S", str(_DEFAULT_SKILL_ACCEPT_TIMEOUT_S)
        )
    )

    goal = json.dumps(
        {
            "rskill_id": skill_id,
            "revision": revision,
            "prompt": prompt,
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
        revision=revision,
        operator_ip=operator_ip,
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
        return JSONResponse(
            {
                "error": (
                    f"action server did not accept the goal within {accept_timeout:.0f} s "
                    "(OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S)"
                )
            },
            status_code=504,
        )

    # ── Phase 2a: goal rejected ───────────────────────────────────────────────
    if rejected:
        await proc.wait()
        _logger.warning(
            "dashboard_write_attempt",
            operation="skill_execute",
            outcome="rejected",
            skill_id=skill_id,
            operator_ip=operator_ip,
        )
        return JSONResponse(
            {"status": "rejected", "detail": "action server rejected the goal"},
            status_code=409,
        )

    # ── Phase 2b: proc exited before any accepted/rejected line ──────────────
    if goal_id is None:
        # EOF without matching either marker — subprocess crashed or exited early.
        await proc.wait()
        out = "\n".join(captured_lines)
        return JSONResponse(
            {"error": "ros2 action send_goal exited before goal was accepted", "output": out},
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
        operator_ip=operator_ip,
    )

    task = asyncio.create_task(
        _drain_skill_goal_stdout(stdout, proc, captured_lines, accepted_goal_id, skill_id)
    )
    _BACKGROUND_DRAIN_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_DRAIN_TASKS.discard)

    return JSONResponse(
        {
            "status": "accepted",
            "goal_id": accepted_goal_id,
            "detail": "skill dispatched; track progress via dashboard telemetry",
        },
        status_code=202,
    )


async def _param_set_from_request(request: Request) -> JSONResponse:
    """Validate the ``POST /api/param/set`` payload and dispatch the operation.

    Handles the flag check, JSON decode, field validation, denylist check, and
    dispatch in one place so the route closure stays under the statement cap
    (PLR0915). Every attempt — permitted or denied — is audit-logged.
    """
    operator_ip = request.client.host if request.client else "unknown"
    if not _write_controls_enabled():
        _logger.warning(
            "dashboard_write_attempt",
            operation="param_set",
            outcome="denied_flag_off",
            operator_ip=operator_ip,
        )
        return JSONResponse(
            {
                "error": (
                    "write-controls disabled; set OPENRAL_DASHBOARD_WRITE_CONTROLS=1 "
                    "(pending safety-WG review)"
                )
            },
            status_code=403,
        )
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"invalid json: {exc}"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "object body required"}, status_code=400)
    node = payload.get("node")
    name = payload.get("name")
    value = payload.get("value")
    if not all(isinstance(v, str) and v.strip() for v in (node, name, value)):
        return JSONResponse(
            {"error": "fields 'node', 'name', 'value' required (non-empty strings)"},
            status_code=400,
        )
    return await _param_set_response(
        node,  # type: ignore[arg-type]  # reason: narrowed by the all(isinstance) guard above
        name,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
        operator_ip,
    )


async def _param_set_response(node: str, name: str, value: str, operator_ip: str) -> JSONResponse:
    """Set a NON-safety ROS param via ``ros2 param set`` (guarded write-control).

    Checks ``name`` case-insensitively against ``_SAFETY_PARAM_DENYLIST`` before
    any shell-out. A matching name is refused with 403 and a paper-trail audit
    log (CLAUDE.md §1.1). Permitted calls are also audit-logged before the
    subprocess is spawned so there is no fire-and-forget gap.
    """
    low = name.lower()
    refused = next((tok for tok in _SAFETY_PARAM_DENYLIST if tok in low), None)
    if refused is not None:
        _logger.warning(
            "dashboard_write_attempt",
            operation="param_set",
            outcome="denied_denylist",
            operator_ip=operator_ip,
            node=node,
            name=name,
            matched=refused,
        )
        return JSONResponse(
            {
                "error": (
                    f"refused: param {name!r} looks safety-relevant (matched {refused!r}); "
                    "change it via a reviewed config/manifest, not the dashboard "
                    "(CLAUDE.md §1.1)"
                )
            },
            status_code=403,
        )
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return JSONResponse(
            {"error": "`ros2` not on PATH; source the workspace install first"},
            status_code=503,
        )
    _logger.warning(
        "dashboard_write_attempt",
        operation="param_set",
        outcome="sent",
        operator_ip=operator_ip,
        node=node,
        name=name,
        value=value,
    )
    proc = await asyncio.create_subprocess_exec(
        ros2,
        "param",
        "set",
        node,
        name,
        value,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return JSONResponse({"error": "param set timed out after 10 s"}, status_code=504)
    out = out_b.decode("utf-8", errors="replace").strip()
    err = err_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return JSONResponse(
            {"error": "ros2 param set failed", "returncode": proc.returncode, "stderr": err},
            status_code=502,
        )
    return JSONResponse({"status": "ok", "stdout": out})


def create_app(store: TelemetryStore | None = None) -> FastAPI:  # noqa: PLR0915  # reason: flat FastAPI route-registration table; each endpoint is one statement
    """Build the FastAPI app bound to ``store`` (a fresh one if ``None``).

    The returned app is a normal ASGI application; mount it under any
    server (uvicorn, hypercorn, ...) or test transport. The store is
    accessible at ``app.state.store`` so tests can introspect it.
    """
    store = store if store is not None else TelemetryStore()
    app = FastAPI(title="OpenRAL Dashboard", docs_url=None, redoc_url=None)
    app.state.store = store
    # Set by run_dashboard when mDNS discovery is wired; None in tests / when
    # the 'mdns' extra is absent. The /api/robots endpoint tolerates both.
    app.state.discovery = None
    # Persistent e-stop publisher (set by run_dashboard). None here so tests /
    # standalone use fall back to the shell-out path in the estop endpoints.
    app.state.estop = None
    # Latched /openral/safety_status subscriber (set by run_dashboard, ADR-0096).
    # None without ROS; the Safety Status card then renders "waiting" rather
    # than claiming the robot is clear.
    app.state.safety_status = None
    # /openral/perception/objects overlay subscriber (set by run_dashboard).
    # None without ROS; the camera tiles then simply carry no overlays.
    app.state.perception_overlay = None

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
        # issue #318 — route the structlog→OTel bridge into the event log.
        # Every OpenRAL log line (incl. DEBUG) ships here; the store maps
        # OTLP severity_number → debug/info/warn/error/fatal and appends a
        # TelemetryEvent per record. The UI defaults the Debug chip off.
        body = await _read_body(request)
        try:
            req = ExportLogsServiceRequest.FromString(body)
        except DecodeError as exc:
            return _bad_request(f"log decode failed: {exc}")
        request.app.state.store.ingest_logs(list(req.resource_logs))
        return _protobuf_response(ExportLogsServiceResponse())

    @app.get("/api/traces")
    async def get_traces(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # List of indexed trace_ids so `openral replay` can pick a trace
        # when the user omits `--trace`. Bounded; see
        # _TRACE_INDEX_MAX_TRACES in store.py.
        return JSONResponse({"traces": request.app.state.store.list_traces()})

    @app.get("/api/spans/{trace_id}")
    async def get_spans(  # pyright: ignore[reportUnusedFunction]
        trace_id: str, request: Request
    ) -> JSONResponse:
        # Full span list for one trace. 404 when the trace has not been
        # ingested (or evicted from the bounded index).
        spans = request.app.state.store.lookup_trace(trace_id)
        if spans is None:
            return JSONResponse({"error": "trace not indexed"}, status_code=404)
        return JSONResponse({"trace_id": trace_id, "spans": spans})

    @app.get("/api/state")
    async def get_state(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(request.app.state.store.snapshot())

    @app.get("/api/stream")
    async def get_stream(request: Request) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        return StreamingResponse(
            _sse_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/camera/{source}/stream")
    async def get_camera_stream(  # pyright: ignore[reportUnusedFunction]
        source: str, request: Request
    ) -> Response:
        # Re-serve the per-camera OTLP thumbnail as a continuous MJPEG stream
        # (issue #75a). 404 only when the source is entirely unknown — a known
        # camera with no frame yet still opens and waits.
        cameras = (
            request.app.state.store.snapshot()
            .get("topics", {})
            .get("perception", {})
            .get("cameras", {})
        )
        if source not in cameras:
            return JSONResponse({"error": f"unknown camera source {source!r}"}, status_code=404)
        return StreamingResponse(
            _mjpeg_stream(request, source),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/robots")
    async def get_robots(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # issue #75b — mDNS-discovered OpenRAL services for the "Add Robot"
        # panel. Read-only. Returns the disabled shape when discovery is off.
        discovery = getattr(request.app.state, "discovery", None)
        if discovery is None:
            return JSONResponse({"enabled": False, "robots": []})
        return JSONResponse(
            {
                "enabled": discovery.enabled,
                "robots": [r.model_dump() for r in discovery.robots()],
            }
        )

    @app.get("/api/config")
    async def get_config() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Dashboard-level config (Jaeger UI url, …) sourced from env.
        # Body in _config_response to keep create_app under the statement cap.
        return _config_response()

    @app.post("/api/prompt")
    async def post_prompt(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator prompt entry point from the dashboard. Shells out to
        # `openral prompt` so the wire shape (PromptStamped on
        # /openral/prompt_in/dashboard, fanned out by prompt_router_node
        # at priority 100) matches the CLI exactly. The console script is
        # named `openral`, not `ral`.
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            return JSONResponse({"error": f"invalid json: {exc}"}, status_code=400)
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            return JSONResponse(
                {"error": "field 'text' required (non-empty string)"}, status_code=400
            )
        return await _prompt_response(text)

    @app.post("/api/transcribe")
    async def post_transcribe(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator voice prompt — the mic button records until silence (browser
        # VAD) and POSTs the raw audio blob (audio/wav). We transcribe it on the
        # host with a local Whisper model and return the text; the page fills
        # the prompt box and reuses the normal /api/prompt send path. Body lives
        # in a module helper to keep create_app() under the statement cap.
        audio = await _read_body(request)
        return await _transcribe_response(audio)

    @app.post("/api/estop")
    async def post_estop(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator "stop the robot NOW". The persistent publisher (discovered at
        # launch) fires INSTANTLY; only when it's unavailable (standalone / no
        # ROS) do we fall back to the slow, discovery-racing shell-out.
        # Mark the latch authoritatively: the e-stop aborts the skill, so the
        # kernel stops emitting safety.check spans and the telemetry-driven flag
        # would never flip — the Reset control would never appear (observed live).
        request.app.state.store.set_estopped(True)
        est = getattr(request.app.state, "estop", None)
        if est is not None and est.available:
            est.trigger()
            return JSONResponse({"status": "ok", "accepted": True}, status_code=200)
        return await _estop_trigger_response()

    @app.post("/api/estop_reset")
    async def post_estop_reset(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator recovery from a latched safety e-stop. Body lives in a
        # module-level helper to keep create_app() under the statement cap.
        resp = await _estop_reset_response(getattr(request.app.state, "estop", None))
        # Clear the latch flag only when the kernel actually accepted the reset
        # (200); a cooldown rejection (409) leaves the robot latched, so the
        # Reset control must stay.
        if resp.status_code == http.HTTPStatus.OK:
            request.app.state.store.set_estopped(False)
        return resp

    @app.post("/api/skill/execute")
    async def post_skill_execute(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # issue #75c — guarded skill switch. Default OFF.
        # Full logic lives in _skill_execute_from_request to keep create_app
        # under the statement cap (PLR0915).
        return await _skill_execute_from_request(request)

    @app.post("/api/param/set")
    async def post_param_set(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # issue #75c — guarded param tune. Default OFF; safety params
        # refused via denylist. Full logic in _param_set_from_request to keep
        # create_app under the statement cap (PLR0915).
        return await _param_set_from_request(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.get("/")
    async def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app


async def _read_body(request: Request) -> bytes:
    """Read the request body, decompressing gzip if the encoding header says so.

    The OTel HTTP exporter compresses by default when ``--insecure`` is
    set; we honour both gzip and identity. Anything else is read as-is.
    """
    body = await request.body()
    encoding = request.headers.get("content-encoding", "").lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    return body


def _protobuf_response(message: Any) -> Response:
    return Response(
        content=message.SerializeToString(),
        media_type="application/x-protobuf",
        status_code=200,
    )


def _bad_request(detail: str) -> Response:
    return Response(
        content=json.dumps({"error": detail}),
        media_type="application/json",
        status_code=400,
    )


def _camera_thumb(store: TelemetryStore, source: str) -> str | None:
    """Return the latest base64 JPEG thumbnail for ``source``, or ``None``."""
    cameras = store.snapshot().get("topics", {}).get("perception", {}).get("cameras", {})
    entry = cameras.get(source)
    if not isinstance(entry, dict):
        return None
    thumb = entry.get("thumbnail_jpeg_b64")
    return thumb if isinstance(thumb, str) and thumb else None


def _mjpeg_part(thumb_b64: str) -> bytes:
    """Frame one base64 JPEG as a multipart/x-mixed-replace part.

    Raises:
        binascii.Error: ``thumb_b64`` is not valid base64.
    """
    jpeg = base64.b64decode(thumb_b64, validate=True)
    head = (
        f"--{_MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return head + jpeg + b"\r\n"


async def _mjpeg_stream(request: Request, source: str) -> AsyncIterator[bytes]:
    """Push the store's latest thumbnail for ``source`` as it changes.

    Subscribes to the store (same primitive as the SSE stream), emits the
    current frame immediately, then a new multipart part on each *changed*
    thumbnail. MJPEG has no keepalive frame, so on idle we just re-check
    client disconnect. Always unsubscribes on exit.
    """
    store: TelemetryStore = request.app.state.store
    queue = store.subscribe()
    last: str | None = None
    try:
        thumb = _camera_thumb(store, source)
        if thumb is not None:
            try:
                part = _mjpeg_part(thumb)
            except binascii.Error:
                _logger.warning("dashboard.mjpeg_decode_failed", source=source)
            else:
                last = thumb
                yield part
        while True:
            if await request.is_disconnected():
                return
            try:
                await asyncio.wait_for(queue.get(), timeout=15.0)
            except TimeoutError:
                continue
            thumb = _camera_thumb(store, source)
            if thumb is not None and thumb != last:
                try:
                    part = _mjpeg_part(thumb)
                except binascii.Error:
                    _logger.warning("dashboard.mjpeg_decode_failed", source=source)
                    continue
                last = thumb
                yield part
    finally:
        store.unsubscribe(queue)


async def _sse_stream(request: Request) -> AsyncIterator[bytes]:
    """Server-Sent Events generator for ``/api/stream``.

    Emits an initial snapshot so a fresh client sees the current state
    without waiting for the next ingest, then forwards every delta from
    the store's subscription queue. A heartbeat keepalive every 15 s
    prevents idle connection timeouts from intermediaries.
    """
    store: TelemetryStore = request.app.state.store
    queue = store.subscribe()
    try:
        yield _sse_frame(store.snapshot())
        while True:
            if await request.is_disconnected():
                return
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=15.0)
            except TimeoutError:
                yield b": keepalive\n\n"
                continue
            yield _sse_frame(payload)
    finally:
        store.unsubscribe(queue)


def _sse_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, default=str)
    return f"data: {body}\n\n".encode()
