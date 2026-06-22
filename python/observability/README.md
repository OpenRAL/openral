# openral-observability

OpenRAL observability — OpenTelemetry tracing + structlog→OTLP logging bridge for Skill / inference / safety spans

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
eight-layer model in the project docs.

- **Docs:** https://openral.github.io/openral/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep at `0.1.x` until the first
> public release (ADR-0021).

## Voice prompt (local speech-to-text)

The live dashboard's operator-prompt box has a mic button. Click it and the
browser listens until you stop speaking — voice-activity detection runs
client-side via [`@ricky0123/vad-web`](https://github.com/ricky0123/vad)
(Silero VAD). The captured audio is POSTed to `POST /api/transcribe`, which
runs a **local** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
model on the host, fills the prompt box with the transcript, and sends it via
the normal `/api/prompt` path. Your audio never leaves the machine.

**It works fully offline, out of the box.** `faster-whisper` ships with the
`dashboard` extra (which `openral-cli` already pulls in), and every browser
asset — the VAD library, Silero model, `onnxruntime-web` and its wasm — is
vendored under `static/vendor/vad/`. No CDN, no opt-in extra, no network at
runtime. The endpoint still degrades to HTTP 503 if `faster-whisper` is ever
stripped from the environment, but a normal install never hits that path. The
only one-time cost is the Whisper model weights, fetched from the Hugging Face
Hub on the first transcription and then cached (pre-pull to stay air-gapped).

Tuning knobs (read at first transcription; defaults run on any CPU):

| Env var | Default | Notes |
| --- | --- | --- |
| `OPENRAL_STT_MODEL` | `base.en` | any faster-whisper model id (`tiny.en`, `small`, `large-v3`, …) |
| `OPENRAL_STT_DEVICE` | `cpu` | `cuda` to use a local GPU |
| `OPENRAL_STT_COMPUTE` | `int8` | CTranslate2 compute type (`int8`, `float16`, …) |

The vendored browser assets and their versions/licenses are documented in
[`static/vendor/vad/NOTICE.md`](src/openral_observability/dashboard/static/vendor/vad/NOTICE.md),
which also records the `npm pack` command to refresh them.

## Live camera video (MJPEG stream)

The camera cards in the dashboard now show live video instead of a static
thumbnail. Each camera is served at:

```
GET /api/camera/{source}/stream
```

This endpoint re-serves the per-camera OTLP thumbnail JPEG as a continuous
`multipart/x-mixed-replace` MJPEG stream — the same thumbnails that already
flow in via the `sensors.read_latest` span attribute `thumbnail_jpeg_b64`. No
extra camera pipeline is needed. The frame rate is bounded by how often the
workload exports spans (configured via `OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS`,
default 30 ms ≈ 33 Hz). The endpoint returns 404 only when the source name is
entirely unknown to the store; a known camera that has not yet emitted a frame
opens the stream and waits.

## mDNS discovery (`mdns` extra)

Install the optional `mdns` extra to let `openral dashboard` advertise itself
on the LAN and browse for other OpenRAL services:

```
pip install openral-observability[mdns]
# or, inside the OpenRAL workspace:
uv sync --group mdns
```

This pulls in [`zeroconf>=0.131`](https://github.com/python-zeroconf/python-zeroconf)
(LGPL-2.1, TSC-approved 2026-06-21; used unmodified as an optional declared
dependency, not vendored).

When `zeroconf` is importable, `run_dashboard` starts a `Discovery` instance
that:

- **Browses** for `_openral-otlp._tcp.local.` services on the LAN (always).
- **Advertises** the dashboard's own OTLP endpoint on the LAN — but **only
  when the bind address is a non-loopback, non-wildcard IPv4 address**. A
  loopback (`127.0.0.1`) or wildcard (`0.0.0.0`) bind is browse-only and never
  advertised (advertising a loopback address to the LAN is meaningless, and
  advertising a wildcard address is ambiguous).

Discovered services are surfaced in the "Add Robot" panel via a read-only
endpoint:

```
GET /api/robots
→ {"enabled": true, "robots": [{name, addresses, port, properties, last_seen}, …]}
```

When the `mdns` extra is absent or `zeroconf` fails to start, the endpoint
returns `{"enabled": false, "robots": []}` — the dashboard runs exactly as
before; discovery is additive, never load-bearing.
