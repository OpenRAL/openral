"""HTTP-level tests for the dashboard ASGI app.

Uses ``httpx.AsyncClient`` against an ``httpx.ASGITransport`` so the
tests exercise the full FastAPI request lifecycle — protobuf decode
on the OTLP/HTTP receiver routes, JSON serialization on /api/state,
SSE framing on /api/stream — without binding a real socket.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


@pytest.fixture
def _otlp_traces_payload(av: Callable[[object], AnyValue]) -> Callable[[], bytes]:
    def _otlp_traces_payload() -> bytes:
        req = ExportTraceServiceRequest(
            resource_spans=[
                ResourceSpans(
                    resource=Resource(attributes=[KeyValue(key="service.name", value=av("ral"))]),
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
                                        KeyValue(key="rskill.id", value=av("smolvla-libero")),
                                    ],
                                )
                            ]
                        )
                    ],
                )
            ]
        )
        return req.SerializeToString()

    return _otlp_traces_payload


@pytest.mark.asyncio
async def test_post_traces_decodes_and_updates_state(
    _otlp_traces_payload: Callable[[], bytes],
) -> None:
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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


@pytest.fixture
def _otlp_logs_payload(av: Callable[[object], AnyValue]) -> Callable[[], bytes]:
    def _otlp_logs_payload() -> bytes:
        req = ExportLogsServiceRequest(
            resource_logs=[
                ResourceLogs(
                    resource=Resource(attributes=[KeyValue(key="service.name", value=av("ral"))]),
                    scope_logs=[
                        ScopeLogs(
                            scope=InstrumentationScope(name="openral.world_state"),
                            log_records=[
                                LogRecord(
                                    time_unix_nano=1_000_000_000_000_000_000,
                                    severity_number=SeverityNumber.SEVERITY_NUMBER_DEBUG,
                                    body=av("world_state.detected_objects count=0"),
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        return req.SerializeToString()

    return _otlp_logs_payload


@pytest.mark.asyncio
async def test_post_logs_ingests_debug_line_into_event_log(
    _otlp_logs_payload: Callable[[], bytes],
) -> None:
    """issue #318 — /v1/logs now surfaces real log lines (incl. DEBUG)."""
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.post(
            "/v1/logs",
            content=b"not-a-protobuf-at-all-\xff\xfe",
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_post_traces_accepts_gzip_encoding(
    _otlp_traces_payload: Callable[[], bytes],
) -> None:
    store = TelemetryStore()
    app = create_app(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.post(
            "/v1/traces",
            content=b"not-a-protobuf",
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_healthz() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_api_config_reflects_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured OPENRAL_JAEGER_UI_URL is surfaced via /api/config (trailing slash stripped)."""
    monkeypatch.setenv("OPENRAL_JAEGER_UI_URL", "https://jaeger.example/")
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["jaeger_ui_url"] == "https://jaeger.example"
        assert resp.json()["org"] == ""  # no enrolment: no org


@pytest.mark.asyncio
async def test_subscriber_queue_receives_ingest_payload(
    _otlp_traces_payload: Callable[[], bytes],
) -> None:
    """The SSE wiring at the store level: subscribers see ingest deltas.

    The actual ``/api/stream`` HTTP framing is exercised in the
    integration test on a real uvicorn socket
    (``tests.integration.test_dashboard_end_to_end``); httpx's
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


# ───────────────────────── POST /api/transcribe (STT) ────────────────────────
#
# Operator speech to text: a client POSTs captured audio here and the endpoint
# runs a local faster-whisper model (ships with the dashboard extra). Per
# CLAUDE.md §1.11 we use the real model on a real (public-domain) speech clip —
# no mocked transcriber. The heavy end-to-end test importorskips faster-whisper
# (it is a default dep, so it runs in a normal env; the skip is only the
# CI-without-the-dashboard-extra path, §1.12); the contract tests (empty body,
# dependency-stripped 503) run everywhere with no model download.

_JFK_WAV = Path(__file__).parent / "fixtures" / "jfk.wav"


@pytest.mark.asyncio
async def test_post_transcribe_rejects_empty_body() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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
        transport=transport, base_url="http://localhost", timeout=120.0
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


@pytest.fixture
def _otlp_camera_payload(av: Callable[[object], AnyValue]) -> Callable[[str, str], bytes]:
    def _otlp_camera_payload(source: str, thumb_b64: str) -> bytes:
        from openral_observability import semconv

        return ExportTraceServiceRequest(
            resource_spans=[
                ResourceSpans(
                    resource=Resource(attributes=[KeyValue(key="service.name", value=av("ral"))]),
                    scope_spans=[
                        ScopeSpans(
                            scope=InstrumentationScope(name="test"),
                            spans=[
                                Span(
                                    name=semconv.SPAN_SENSORS_READ_LATEST,
                                    trace_id=b"\x11" * 16,
                                    span_id=b"\x22" * 8,
                                    attributes=[
                                        KeyValue(key=semconv.SENSORS_SOURCE, value=av(source)),
                                        KeyValue(
                                            key=semconv.SENSORS_THUMBNAIL_JPEG_B64,
                                            value=av(thumb_b64),
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ]
        ).SerializeToString()

    return _otlp_camera_payload


@pytest.mark.asyncio
async def test_api_robots_disabled_when_no_discovery() -> None:
    app = create_app(TelemetryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
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
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.get("/api/robots")
    body = resp.json()
    assert body["enabled"] is True
    assert body["robots"][0]["name"] == "arm"
    assert body["robots"][0]["port"] == 4318


@pytest.mark.asyncio
async def test_camera_snapshot_serves_the_latest_jpeg(
    _otlp_camera_payload: Callable[[str, str], bytes],
) -> None:
    # Cameras are polled as JPEG snapshots, never streamed (ADR-0129 §3).
    import base64

    jpeg = b"\xff\xd8\xff\xe0jpegbytes\xff\xd9"
    transport = httpx.ASGITransport(app=create_app(TelemetryStore()))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        await client.post(
            "/v1/traces", content=_otlp_camera_payload("wrist", base64.b64encode(jpeg).decode())
        )
        resp = await client.get("/api/camera/wrist/snapshot.jpg")
        unknown = await client.get("/api/camera/nope/snapshot.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.headers["cache-control"] == "no-store"
    assert resp.content == jpeg
    assert unknown.status_code == 404


def test_stream_without_images_drops_only_camera_thumbnails() -> None:
    # `?images=0`: the SSE payload minus thumbnails, without touching the shared snapshot.
    from openral_observability.dashboard.app import _without_images

    snapshot: dict[str, Any] = {
        "topics": {
            "perception": {"cameras": {"wrist": {"thumbnail_jpeg_b64": "x", "age_ms": 3}}},
            "slam": {"png_b64": "map"},
        },
        "run_id": "r",
    }
    stripped = _without_images(snapshot)
    assert stripped["topics"]["perception"]["cameras"] == {"wrist": {"age_ms": 3}}
    assert stripped["topics"]["slam"] == {"png_b64": "map"}  # the map panel still draws
    assert snapshot["topics"]["perception"]["cameras"]["wrist"]["thumbnail_jpeg_b64"] == "x"


@pytest.mark.asyncio
async def test_root_serves_the_fallback_page_without_a_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_DASHBOARD_UI_DIR", raising=False)
    app = create_app(TelemetryStore())
    app.state.ui_dir = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.get("/")
        missing = await client.get("/assets/app.js")
    assert resp.status_code == 200
    assert b"api/actions/estop" in resp.content  # the E-STOP button
    assert missing.status_code == 404
    # A strict CSP: the page's one inline script is allowed by hash, nothing else inline.
    csp = resp.headers["content-security-policy"]
    script = re.search(rb"<script>([\s\S]*?)</script>", resp.content)
    assert script is not None
    digest = base64.b64encode(hashlib.sha256(script.group(1)).digest()).decode()
    assert f"script-src 'self' 'sha256-{digest}';" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'self' https://www.openral.com https://dev.openral.com" in csp
    # This origin only: no bare ws:/wss: that would let an injected script reach anywhere.
    assert "connect-src 'self';" in csp and "ws:" not in csp


@pytest.mark.asyncio
async def test_root_serves_a_local_ui_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>robot ui</title>")
    (tmp_path / "assets" / "app.js").write_text("console.log('ui')")
    (tmp_path.parent / "secret.txt").write_text("outside the build")
    monkeypatch.setenv("OPENRAL_DASHBOARD_UI_DIR", str(tmp_path))
    transport = httpx.ASGITransport(app=create_app(TelemetryStore()))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        index = await client.get("/")
        asset = await client.get("/assets/app.js")
        escape = await client.get("/assets/%2E%2E/%2E%2E/secret.txt")
    assert b"robot ui" in index.content
    assert asset.text == "console.log('ui')"
    assert escape.status_code == 404
    assert "script-src 'self';" in index.headers["content-security-policy"]
    # The page names the hashed bundles: revalidated on every load, or an updated UI
    # keeps serving the old one from the browser cache. The bundles themselves may cache.
    assert index.headers["cache-control"] == "no-cache"
    assert "cache-control" not in asset.headers


@pytest.mark.asyncio
async def test_viz_serves_lichtblick_under_its_own_csp(tmp_path: Path) -> None:
    # Lichtblick needs eval (it will not start without `new Function`), WebAssembly and
    # blob: workers, and has one inline script. No manifest yet: the slot stays empty.
    inline = "global = globalThis;"
    (tmp_path / "index.html").write_text(f"<html><script>{inline}</script></html>")
    app = create_app(TelemetryStore())
    app.state.viz_dir = tmp_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.get("/viz/")
    csp = resp.headers["content-security-policy"]
    digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
    assert f"script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval' 'sha256-{digest}';" in csp
    assert "worker-src 'self' blob:" in csp
    # A crafted ?ds.url=wss://elsewhere cannot point Lichtblick at a foreign bridge.
    assert "connect-src 'self';" in csp and "ws:" not in csp
    assert "frame-ancestors 'self'" in csp


@pytest.mark.asyncio
async def test_gzip_body_over_decoded_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_observability.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "_MAX_BODY_BYTES", 1024)
    bomb = gzip.compress(b"\0" * 100_000)
    assert len(bomb) < 1024
    store = TelemetryStore()
    transport = httpx.ASGITransport(app=create_app(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        resp = await client.post(
            "/v1/traces",
            content=bomb,
            headers={"content-type": "application/x-protobuf", "content-encoding": "gzip"},
        )
    assert resp.status_code == 413, resp.text


@pytest.mark.asyncio
async def test_post_transcribe_refuses_audio_over_duration_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A compressed clip can decode to far more samples than its byte size
    # suggests; the decoder must stop at the duration budget, not buffer it all.
    pytest.importorskip("faster_whisper")
    from openral_observability.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "_STT_MAX_SECONDS", 2)
    monkeypatch.setenv("OPENRAL_STT_MODEL", "tiny.en")
    transport = httpx.ASGITransport(app=create_app(TelemetryStore()))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost", timeout=120.0
    ) as c:
        resp = await c.post(
            "/api/transcribe", content=_JFK_WAV.read_bytes(), headers={"content-type": "audio/wav"}
        )
    assert resp.status_code == 413, resp.text


def test_bounded_decode_skips_corrupt_frames() -> None:
    # Like faster_whisper.decode_audio, a corrupt frame must not fail the decode (→ 422):
    # a real MP3 with a stomped mid-stream frame still yields the audio before it.
    import io

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from openral_observability.dashboard import app as app_mod

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp3") as out:
        stream = out.add_stream("mp3", rate=16_000)
        t = np.arange(16_000 * 3) / 16_000
        pcm = (np.sin(2 * np.pi * 440 * t) * 10_000).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = 16_000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    data = bytearray(buf.getvalue())
    off = len(data) // 4
    data[off : off + 64] = b"\xff" * 64

    # The raw PyAV iterator aborts on the stomped frame.
    with pytest.raises(av.error.InvalidDataError), av.open(io.BytesIO(bytes(data))) as c:
        for _ in c.decode(audio=0):
            pass
    # Same outcome as faster-whisper's own decoder: the frames before the corruption,
    # no exception (PyAV ends its decode generator once it has raised).
    pcm_out = app_mod._decode_audio_bounded(bytes(data))
    assert pcm_out.size > 0
    faster_whisper = pytest.importorskip("faster_whisper")
    assert pcm_out.size == faster_whisper.decode_audio(io.BytesIO(bytes(data))).size
