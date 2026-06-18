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
