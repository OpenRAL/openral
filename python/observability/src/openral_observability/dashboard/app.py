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
  out to ``openral prompt`` (ADR-0018 F10) targeting the prompt-router's
  ``dashboard`` source.
* ``POST /api/estop_reset`` — operator recovery: clears a latched safety
  e-stop by calling the kernel's ``/openral/estop_reset`` (std_srvs/Trigger)
  service via ``ros2 service call``, so the robot can be re-tasked after an
  e-stop. Returns 409 when the kernel rejects it (post-estop cooldown).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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

_STATIC_DIR = Path(__file__).parent / "static"


async def _prompt_response(text: str) -> JSONResponse:
    """Publish an operator prompt via ``openral prompt`` (ADR-0018 F10).

    Shells out so the wire shape (PromptStamped on
    ``/openral/prompt_in/dashboard``, fanned out by ``prompt_router_node`` at
    priority 100) matches the CLI exactly. The console script is ``openral``
    (ADR-0021). Body lives here to keep ``create_app`` under the statement cap.
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


async def _estop_reset_response() -> JSONResponse:
    """Clear a latched safety e-stop via the kernel's ``/openral/estop_reset``.

    The C++ safety kernel latches on a violation and drops every candidate chunk
    until this ``std_srvs/Trigger`` service is called — so after an e-stop NO
    prompt does anything until the latch is cleared. The dashboard has no rclpy
    node of its own, so (like ``POST /api/prompt``'s ``openral prompt``
    shell-out) we call ``ros2 service call``. The kernel enforces a post-estop
    cooldown; an early call returns ``success=false`` (HTTP 409) so the operator
    can retry. Re-prompt via ``/api/prompt`` once this succeeds.
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
    return JSONResponse(
        {"status": "ok" if accepted else "rejected", "accepted": accepted, "stdout": stdout},
        status_code=200 if accepted else 409,
    )


def create_app(store: TelemetryStore | None = None) -> FastAPI:
    """Build the FastAPI app bound to ``store`` (a fresh one if ``None``).

    The returned app is a normal ASGI application; mount it under any
    server (uvicorn, hypercorn, ...) or test transport. The store is
    accessible at ``app.state.store`` so tests can introspect it.
    """
    store = store if store is not None else TelemetryStore()
    app = FastAPI(title="OpenRAL Dashboard", docs_url=None, redoc_url=None)
    app.state.store = store

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
        # ADR-0018 F7 — list of indexed trace_ids so `openral replay` can
        # pick a trace when the user omits `--trace`. Bounded; see
        # _TRACE_INDEX_MAX_TRACES in store.py.
        return JSONResponse({"traces": request.app.state.store.list_traces()})

    @app.get("/api/spans/{trace_id}")
    async def get_spans(  # pyright: ignore[reportUnusedFunction]
        trace_id: str, request: Request
    ) -> JSONResponse:
        # ADR-0018 F7 — full span list for one trace. 404 when the trace
        # has not been ingested (or evicted from the bounded index).
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

    @app.get("/api/config")
    async def get_config() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Dashboard-level config (Jaeger UI url, …) sourced from env.

        The UI fetches this once on load to decide whether to enable the
        "open in jaeger" link. Returning ``""`` (the default) leaves the
        link disabled with a helpful tooltip — the previous behaviour of
        unconditionally linking to ``localhost:16686`` produced a
        broken-link click for every user who doesn't run Jaeger locally.
        """
        jaeger_url = os.environ.get("OPENRAL_JAEGER_UI_URL", "").rstrip("/")
        return JSONResponse({"jaeger_ui_url": jaeger_url})

    @app.post("/api/prompt")
    async def post_prompt(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # ADR-0018 F10 — operator prompt entry point from the dashboard.
        # Shells out to `openral prompt` so the wire shape (PromptStamped on
        # /openral/prompt_in/dashboard, fanned out by prompt_router_node
        # at priority 100) matches the CLI exactly. The console script is
        # named `openral` (ADR-0021), not `ral`.
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

    @app.post("/api/estop_reset")
    async def post_estop_reset(_request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # Operator recovery from a latched safety e-stop. Body lives in a
        # module-level helper to keep create_app() under the statement cap.
        return await _estop_reset_response()

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
