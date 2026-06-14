"""HTTP-level tests for the dashboard ASGI app.

Uses ``httpx.AsyncClient`` against an ``httpx.ASGITransport`` so the
tests exercise the full FastAPI request lifecycle — protobuf decode
on the OTLP/HTTP receiver routes, JSON serialization on /api/state,
SSE framing on /api/stream — without binding a real socket.
"""

from __future__ import annotations

import asyncio
import gzip
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from openral_observability.dashboard import TelemetryStore, create_app
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, InstrumentationScope, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import (
    LogRecord,
    ResourceLogs,
    ScopeLogs,
    SeverityNumber,
)
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
)


def _av(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    return AnyValue(string_value=str(value))


def _otlp_traces_payload() -> bytes:
    req = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(attributes=[KeyValue(key="service.name", value=_av("ral"))]),
                scope_spans=[
                    ScopeSpans(
                        spans=[
                            Span(
                                trace_id=b"\x02" * 16,
                                span_id=b"\x02" * 8,
                                name="rskill.execute",
                                start_time_unix_nano=1_000_000_000_000_000_000,
                                end_time_unix_nano=1_000_000_000_023_000_000,
                                attributes=[
                                    KeyValue(key="rskill.id", value=_av("smolvla-libero")),
                                ],
                            )
                        ]
                    )
                ],
            )
        ]
    )
    return req.SerializeToString()


@pytest.mark.asyncio
async def test_post_traces_decodes_and_updates_state() -> None:
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/traces",
            content=_otlp_traces_payload(),
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 200
        # Body must be a valid ExportTraceServiceResponse protobuf.
        ExportTraceServiceResponse.FromString(resp.content)

        state = (await client.get("/api/state")).json()
        assert state["service_name"] == "ral"
        card = state["cards"]["rskill_execute"]
        assert card["attrs"]["rskill.id"] == "smolvla-libero"
        assert card["duration_ms"] == 23.0


def _otlp_logs_payload() -> bytes:
    req = ExportLogsServiceRequest(
        resource_logs=[
            ResourceLogs(
                resource=Resource(attributes=[KeyValue(key="service.name", value=_av("ral"))]),
                scope_logs=[
                    ScopeLogs(
                        scope=InstrumentationScope(name="openral.world_state"),
                        log_records=[
                            LogRecord(
                                time_unix_nano=1_000_000_000_000_000_000,
                                severity_number=SeverityNumber.SEVERITY_NUMBER_DEBUG,
                                body=_av("world_state.detected_objects count=0"),
                            )
                        ],
                    )
                ],
            )
        ]
    )
    return req.SerializeToString()


@pytest.mark.asyncio
async def test_post_logs_ingests_debug_line_into_event_log() -> None:
    """issue #318 — /v1/logs now surfaces real log lines (incl. DEBUG)."""
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/logs",
            content=_otlp_logs_payload(),
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 200
        # Body must be a valid ExportLogsServiceResponse protobuf.
        ExportLogsServiceResponse.FromString(resp.content)

        state = (await client.get("/api/state")).json()
        debug = [ev for ev in state["events"] if ev["severity"] == "debug"]
        assert len(debug) == 1
        assert debug[0]["kind"] == "openral.world_state"
        assert debug[0]["title"] == "world_state.detected_objects count=0"


@pytest.mark.asyncio
async def test_post_logs_malformed_returns_400() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/logs",
            content=b"not-a-protobuf-at-all-\xff\xfe",
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_post_traces_accepts_gzip_encoding() -> None:
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = gzip.compress(_otlp_traces_payload())
        resp = await client.post(
            "/v1/traces",
            content=body,
            headers={"Content-Type": "application/x-protobuf", "Content-Encoding": "gzip"},
        )
        assert resp.status_code == 200
        state = (await client.get("/api/state")).json()
        assert "rskill_execute" in state["cards"]


@pytest.mark.asyncio
async def test_post_traces_malformed_returns_400() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/traces",
            content=b"not-a-protobuf",
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_index_serves_html() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "OpenRAL · Instrument Deck".encode() in resp.content
        assert "<title>OpenRAL · Instrument Deck</title>".encode() in resp.content


@pytest.mark.asyncio
async def test_healthz() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_api_config_defaults_to_empty_jaeger_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/config returns an empty Jaeger URL when OPENRAL_JAEGER_UI_URL is unset.

    The UI uses this to decide whether to enable the "open in jaeger"
    footer link — an empty string keeps the link disabled with a
    tooltip rather than producing a broken-link click against a
    guessed ``localhost:16686``.
    """
    monkeypatch.delenv("OPENRAL_JAEGER_UI_URL", raising=False)
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json() == {"jaeger_ui_url": ""}


@pytest.mark.asyncio
async def test_api_config_reflects_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured OPENRAL_JAEGER_UI_URL is surfaced via /api/config (trailing slash stripped)."""
    monkeypatch.setenv("OPENRAL_JAEGER_UI_URL", "https://jaeger.example/")
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json() == {"jaeger_ui_url": "https://jaeger.example"}


@pytest.mark.asyncio
async def test_subscriber_queue_receives_ingest_payload() -> None:
    """The SSE wiring at the store level: subscribers see ingest deltas.

    The actual ``/api/stream`` HTTP framing is exercised in the
    integration test on a real uvicorn socket
    (:mod:`tests.integration.test_dashboard_end_to_end`); httpx's
    ``ASGITransport`` buffers the full response body so a streaming
    endpoint deadlocks against it. Here we validate the store-level
    publish channel that the SSE generator awaits on.
    """
    store = TelemetryStore()
    queue = store.subscribe()
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        req = ExportTraceServiceRequest.FromString(_otlp_traces_payload())
        store.ingest_spans(list(req.resource_spans))
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert "rskill_execute" in payload["cards"]
    finally:
        store.unsubscribe(queue)


# ─────────────────────────── POST /api/prompt ────────────────────────────────
#
# These tests exercise the real subprocess-spawn path. Per CLAUDE.md §1.11 /
# §5.4 we do not mock `subprocess` / `asyncio.create_subprocess_exec`; instead
# we shadow `openral` on PATH with a tiny real script that records its argv to
# a log file. That gives us a green test only if the production code actually
# spawns a child with the expected args.


@pytest.fixture
def openral_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install an `openral` shim on PATH that logs its argv and exits 0.

    Yields the log file. Tests read the file to confirm the dashboard
    spawned `openral prompt <text> --topic /openral/prompt_in/dashboard`.
    """
    log = tmp_path / "openral_calls.txt"
    shim = tmp_path / "openral"
    # Real shim — records argv (minus script path) as JSON and prints a
    # line matching the canonical `openral prompt` stdout so the dashboard
    # endpoint surfaces it back to the operator unchanged.
    shim.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import json, sys, pathlib\n"
        + f"pathlib.Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        + "topic = sys.argv[sys.argv.index('--topic') + 1]\n"
        + "print(f'openral prompt: published on {topic} text={sys.argv[2]!r}')\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    yield log


def _read_shim_argv(log: Path) -> list[str]:
    """Sync helper so the async test stays clear of ASYNC240."""
    import json as _json

    return list(_json.loads(log.read_text()))


@pytest.mark.asyncio
async def test_post_prompt_invokes_openral_with_dashboard_topic(openral_shim: Path) -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/prompt", json={"text": "pick the red cube"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert "published on /openral/prompt_in/dashboard" in body["stdout"]
    # The shim logged the real argv — confirms we shelled out to `openral
    # prompt` with --topic pointing at the dashboard source.
    argv = _read_shim_argv(openral_shim)
    assert argv == ["prompt", "pick the red cube", "--topic", "/openral/prompt_in/dashboard"]


@pytest.mark.asyncio
async def test_post_prompt_rejects_empty_text(openral_shim: Path) -> None:
    del openral_shim  # fixture present so PATH is shimmed, but no call expected
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for payload in ({}, {"text": ""}, {"text": "   "}, {"text": 42}):
            resp = await client.post("/api/prompt", json=payload)
            assert resp.status_code == 400, payload


@pytest.mark.asyncio
async def test_post_prompt_propagates_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real shim, but exits non-zero — exercises the 502 path without mocks.
    shim = tmp_path / "openral"
    shim.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import sys\n"
        + "print('boom', file=sys.stderr)\n"
        + "sys.exit(7)\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/prompt", json={"text": "go"})
        assert resp.status_code == 502
        body = resp.json()
        assert body["returncode"] == 7
        assert "boom" in body["stderr"]


@pytest.mark.asyncio
async def test_post_prompt_returns_503_when_openral_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force an empty PATH so `shutil.which("openral")` returns None.
    monkeypatch.setenv("PATH", str(tmp_path))
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/prompt", json={"text": "go"})
        assert resp.status_code == 503
        assert "not on PATH" in resp.json()["error"]
