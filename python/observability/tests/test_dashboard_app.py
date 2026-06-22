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
        assert "OpenRAL · Live Dashboard".encode() in resp.content
        assert "<title>OpenRAL · Live Dashboard</title>".encode() in resp.content


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


# ───────────────────────── POST /api/transcribe (STT) ────────────────────────
#
# Operator voice prompt: the dashboard mic POSTs captured audio here and the
# endpoint runs a local faster-whisper model (ships with the dashboard extra).
# Per CLAUDE.md §1.11 we use the real model on a real (public-domain) speech
# clip — no mocked transcriber. The heavy end-to-end test importorskips
# faster-whisper (it is a default dep, so it runs in a normal env; the skip is
# only the CI-without-the-dashboard-extra path, §1.12); the contract tests
# (empty body, dependency-stripped 503) run everywhere with no model download.

_JFK_WAV = Path(__file__).parent / "fixtures" / "jfk.wav"


@pytest.mark.asyncio
async def test_post_transcribe_rejects_empty_body() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/transcribe", content=b"")
        assert resp.status_code == 400
        assert "empty audio" in resp.json()["error"]


@pytest.mark.asyncio
async def test_post_transcribe_returns_503_when_faster_whisper_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real import failure (not a mocked transcriber): shadow `faster_whisper`
    # with None in sys.modules so the in-function import raises ImportError,
    # exercising the graceful-degradation path exactly as an uninstalled host.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/transcribe", content=b"\x00\x01\x02\x03")
        assert resp.status_code == 503
        assert "speech-to-text unavailable" in resp.json()["error"]


@pytest.mark.asyncio
async def test_post_transcribe_real_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    # End-to-end with the real model on a real public-domain JFK clip
    # ("...ask not what your country can do for you..."). tiny.en keeps the
    # one-time model download light; it transcribes this clean 16 kHz clip
    # reliably. Skips only if faster-whisper is absent (non-dashboard install).
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("OPENRAL_STT_MODEL", "tiny.en")
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=120.0
    ) as client:
        resp = await client.post(
            "/api/transcribe",
            content=_JFK_WAV.read_bytes(),
            headers={"content-type": "audio/wav"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "tiny.en"
        assert "country" in body["text"].lower(), body["text"]


def _otlp_camera_payload(source: str, thumb_b64: str) -> bytes:
    from openral_observability import semconv

    return ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(attributes=[KeyValue(key="service.name", value=_av("ral"))]),
                scope_spans=[
                    ScopeSpans(
                        scope=InstrumentationScope(name="test"),
                        spans=[
                            Span(
                                name=semconv.SPAN_SENSORS_READ_LATEST,
                                trace_id=b"\x11" * 16,
                                span_id=b"\x22" * 8,
                                attributes=[
                                    KeyValue(key=semconv.SENSORS_SOURCE, value=_av(source)),
                                    KeyValue(
                                        key=semconv.SENSORS_THUMBNAIL_JPEG_B64,
                                        value=_av(thumb_b64),
                                    ),
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    ).SerializeToString()


@pytest.mark.asyncio
async def test_camera_stream_emits_jpeg_part() -> None:
    # httpx.ASGITransport buffers the full response body before returning,
    # so a true infinite MJPEG StreamingResponse deadlocks it (same limitation
    # noted in test_subscriber_queue_receives_ingest_payload for SSE). We
    # therefore exercise the two helpers directly — the same bytes the live
    # server would push over the wire — confirming the store ingest → b64
    # thumbnail → MJPEG part bytes round-trip produces valid framing.
    # Wire-format coverage (route response headers, Content-Type boundary,
    # 404 for unknown sources over a real socket) lives in
    # test_dashboard_mjpeg_integration.py which launches a real uvicorn server.
    import base64

    from openral_observability.dashboard.app import _camera_thumb, _mjpeg_part

    store = TelemetryStore()
    app = create_app(store)
    jpeg = b"\xff\xd8\xff\xe0jpegbytes\xff\xd9"
    thumb_b64 = base64.b64encode(jpeg).decode("ascii")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/v1/traces", content=_otlp_camera_payload("wrist", thumb_b64))

    # Helpers: round-trip through the store → b64 → MJPEG part bytes.
    thumb = _camera_thumb(store, "wrist")
    assert thumb == thumb_b64
    chunk = _mjpeg_part(thumb)
    assert b"Content-Type: image/jpeg" in chunk
    assert b"Content-Length: " in chunk
    assert jpeg in chunk


@pytest.mark.asyncio
async def test_camera_stream_unknown_source_404() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/camera/nope/stream")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_robots_disabled_when_no_discovery() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/robots")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "robots": []}


@pytest.mark.asyncio
async def test_api_robots_lists_registry() -> None:
    from openral_observability.dashboard.discovery import DiscoveredRobot, Discovery, RobotRegistry

    reg = RobotRegistry()
    reg.upsert(
        DiscoveredRobot(name="arm", addresses=["10.0.0.5"], port=4318, properties={}, last_seen=1.0)
    )
    disc = Discovery(registry=reg)
    disc.enabled = True
    app = create_app(TelemetryStore())
    app.state.discovery = disc
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/robots")
    body = resp.json()
    assert body["enabled"] is True
    assert body["robots"][0]["name"] == "arm"
    assert body["robots"][0]["port"] == 4318


# ─────────────────────── POST /api/skill/execute ─────────────────────────────
#
# issue #75c / ADR-0064 — flag-gated skill switch. DEFAULT OFF. Tests are
# written BEFORE the implementation (TDD — safety-touching per CLAUDE.md §4.2).
# The "no ros2" case reuses the empty-PATH trick from
# test_post_prompt_returns_503_when_openral_missing above.


@pytest.mark.asyncio
async def test_skill_execute_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", raising=False)
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/api/skill/execute", json={"skill_id": "x"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_skill_execute_requires_skill_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/api/skill/execute", json={"skill_id": "  "})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_skill_execute_503_without_ros2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH → shutil.which("ros2") is None
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/api/skill/execute", json={"skill_id": "openral/skill-pick"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_skill_execute_rejects_non_one_truthy_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exact string "1" enables write-controls (ADR-0064 §1).

    Common truthy strings ("true", "yes", "on", "True") must NOT unlock the
    endpoint. This locks the safety default against a future refactor that
    loosens the check from a strict equality to a broader truthiness test.
    """
    for truthy_non_one in ("true", "True", "yes", "on", "1 ", " 1", "2"):
        monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", truthy_non_one)
        app = create_app(TelemetryStore())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post("/api/skill/execute", json={"skill_id": "x"})
        assert resp.status_code == 403, (
            f"expected 403 for OPENRAL_DASHBOARD_WRITE_CONTROLS={truthy_non_one!r}, "
            f"got {resp.status_code}"
        )


@pytest.mark.asyncio
async def test_vendored_vad_assets_served_offline() -> None:
    # The voice prompt is fully offline: the VAD library, Silero model and
    # onnxruntime-web wasm are vendored under static/vendor/vad/ (no CDN). Assert
    # each asset serves 200 with a browser-loadable MIME — onnxruntime's ESM
    # (.mjs) and wasm fail to load if the dashboard mounts them as text/plain.
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    expected = {
        "bundle.min.js": "javascript",
        "vad.worklet.bundle.min.js": "javascript",
        "silero_vad_v5.onnx": None,
        "ort.wasm.min.js": "javascript",
        "ort-wasm-simd-threaded.mjs": "javascript",
        "ort-wasm-simd-threaded.wasm": "application/wasm",
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for name, ctype in expected.items():
            resp = await client.get(f"/static/vendor/vad/{name}")
            assert resp.status_code == 200, name
            if ctype is not None:
                assert ctype in resp.headers["content-type"], (name, resp.headers["content-type"])
