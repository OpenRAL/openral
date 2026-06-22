"""Real-server integration test for the MJPEG camera-stream endpoint (issue #75a).

``httpx.ASGITransport`` buffers the full response body before returning the
``Response`` object, so an infinite ``StreamingResponse`` deadlocks it.  This
module launches a **real uvicorn server** in a subprocess, POSTs a genuine OTLP
camera span to ``/v1/traces``, then reads ``GET /api/camera/<src>/stream``
over a raw socket to verify the wire format end-to-end.

The test is intentionally isolated from the ASGI-transport tests so it can use
blocking ``socket`` / ``urllib.request`` I/O inside a plain (non-async) test
function — no event loop needed.
"""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

# Skip the entire module if uvicorn is not importable (non-dashboard install).
uvicorn = pytest.importorskip("uvicorn", reason="uvicorn not installed (dashboard extra required)")

_HOST = "127.0.0.1"
_PORT = 4399
_BASE = f"http://{_HOST}:{_PORT}"
# PYTHONPATH must cover both observability and core sources so the subprocess
# can import openral_observability and openral_core from the worktree.
_PYTHONPATH = "python/observability/src:python/core/src"


def _wait_healthy(timeout: float = 10.0) -> bool:
    """Poll ``/healthz`` until the server responds 200 or ``timeout`` expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(f"{_BASE}/healthz", timeout=0.5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _build_otlp_camera_payload(source: str, thumb_b64: str) -> bytes:
    """Build a serialised OTLP ExportTraceServiceRequest with one camera span."""
    # Imports here so the module-level importorskip for uvicorn is the only
    # hard gate; these proto/semconv deps are present in any observability env.
    from openral_observability import semconv
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import (
        AnyValue,
        InstrumentationScope,
        KeyValue,
    )
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

    def _av(s: str) -> AnyValue:
        return AnyValue(string_value=s)

    span = Span(
        name=semconv.SPAN_SENSORS_READ_LATEST,
        trace_id=b"\x11" * 16,
        span_id=b"\x22" * 8,
        attributes=[
            KeyValue(key=semconv.SENSORS_SOURCE, value=_av(source)),
            KeyValue(key=semconv.SENSORS_THUMBNAIL_JPEG_B64, value=_av(thumb_b64)),
        ],
    )
    rs = ResourceSpans(
        resource=Resource(attributes=[KeyValue(key="service.name", value=_av("ral"))]),
        scope_spans=[ScopeSpans(scope=InstrumentationScope(name="integration-test"), spans=[span])],
    )
    return ExportTraceServiceRequest(resource_spans=[rs]).SerializeToString()


@pytest.fixture(scope="module")
def live_server() -> subprocess.Popen[bytes]:  # type: ignore[type-arg]
    """Start a real uvicorn server; yield the Popen handle; terminate on teardown."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _PYTHONPATH
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--host",
            _HOST,
            "--port",
            str(_PORT),
            "openral_observability.dashboard.app:create_app",
            "--factory",
            "--log-level",
            "error",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_healthy(timeout=10.0):
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail(f"uvicorn did not become healthy on {_BASE}/healthz within 10 s")
        yield proc  # type: ignore[misc]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_mjpeg_stream_wire_format(live_server: subprocess.Popen[bytes]) -> None:  # type: ignore[type-arg]
    """POST a real OTLP camera span then verify the MJPEG stream wire format.

    Asserts:
    - The HTTP response line contains ``multipart/x-mixed-replace``.
    - A ``Content-Type: image/jpeg`` part header is present.
    - The exact JPEG bytes ingested appear in the response body.
    """
    del live_server  # used only to ensure the server is up

    jpeg = b"\xff\xd8\xff\xe0" + b"REALSERVERjpeg" + b"\xff\xd9"
    thumb_b64 = base64.b64encode(jpeg).decode("ascii")
    payload = _build_otlp_camera_payload("wrist", thumb_b64)

    # POST the span so the store knows about camera source "wrist".
    req = urllib.request.Request(
        f"{_BASE}/v1/traces",
        data=payload,
        headers={"Content-Type": "application/x-protobuf"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    assert resp.status == 200, f"OTLP ingest failed with status {resp.status}"

    # Open a raw socket and read the first MJPEG chunk.
    sock = socket.create_connection((_HOST, _PORT), timeout=5)
    try:
        sock.sendall(b"GET /api/camera/wrist/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        sock.settimeout(5)
        buf = b""
        while len(buf) < 32_768:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            # Stop once we have the JPEG end-of-image marker and the part header.
            if b"\xff\xd9" in buf and b"image/jpeg" in buf:
                break
    finally:
        sock.close()

    assert b"multipart/x-mixed-replace" in buf, (
        "Response did not contain multipart/x-mixed-replace Content-Type"
    )
    assert b"Content-Type: image/jpeg" in buf, (
        "MJPEG part did not contain Content-Type: image/jpeg header"
    )
    assert jpeg in buf, "Exact JPEG bytes were not present in the MJPEG stream"


def test_mjpeg_stream_unknown_source_returns_404(
    live_server: subprocess.Popen[bytes],  # type: ignore[type-arg]
) -> None:
    """``GET /api/camera/<unknown>/stream`` must return HTTP 404."""
    del live_server  # used only to ensure the server is up

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{_BASE}/api/camera/no_such_camera/stream", timeout=5)
    assert exc_info.value.code == 404, (
        f"Expected 404 for unknown camera source, got {exc_info.value.code}"
    )
