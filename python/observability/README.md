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
> public release.

## Voice prompt (local speech-to-text)

The live dashboard's operator-prompt box has a mic button. Click it and the
browser listens until you stop speaking — voice-activity detection runs
client-side via [`@ricky0123/vad-web`](https://github.com/ricky0123/vad)
(Silero VAD). The captured audio is POSTed to `POST /api/transcribe`, which
runs a **local** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
model on the host, fills the prompt box with the transcript, and sends it via
the normal `/api/prompt` path. Your audio never leaves the machine.

`faster-whisper` ships with the `dashboard` extra (which `openral-cli` already
pulls in). The small browser glue files for VAD are vendored under
`static/vendor/vad/`; the large pinned Silero ONNX and onnxruntime WASM assets
are fetched and checksum-verified into `$OPENRAL_CACHE_DIR/dashboard_assets/vad/`
on dashboard startup, then copied into the served static directory. If those
assets are unavailable, `/api/config` reports `voice_prompt_enabled=false` and
the mic button stays disabled instead of failing at click time. The transcription
endpoint still degrades to HTTP 503 if `faster-whisper` is ever stripped from the
environment. Whisper model weights are fetched from the Hugging Face Hub on the
first transcription and then cached (pre-pull both caches to stay air-gapped).

Tuning knobs (read at first transcription; defaults run on any CPU):

| Env var | Default | Notes |
| --- | --- | --- |
| `OPENRAL_STT_MODEL` | `base.en` | any faster-whisper model id (`tiny.en`, `small`, `large-v3`, …) |
| `OPENRAL_STT_DEVICE` | `cpu` | `cuda` to use a local GPU |
| `OPENRAL_STT_COMPUTE` | `int8` | CTranslate2 compute type (`int8`, `float16`, …) |

The vendored browser assets and their versions/licenses are documented in
[`static/vendor/vad/NOTICE.md`](src/openral_observability/dashboard/static/vendor/vad/NOTICE.md),
which also records the `npm pack` command to refresh them.

## Safety · current state (latched `/openral/safety_status`)

ADR-0096. Every other card on the dashboard is fed by OTel over the embedded
OTLP receiver. This one is not: `SafetyStatusSubscriber` opens a real rclpy
subscription to the **latched** `/openral/safety_status`
(`openral_msgs/SafetyStatus`, RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1) that
the C++ safety kernel and `SafetyPassthroughNode` both publish, and writes
each status into the store via `TelemetryStore.set_safety_status`.

Why a topic and not a span: the `safety.check` span path can only *infer* a
latch from chunk flow, and a latched kernel is exactly the state in which the
chunk flow stops. Durability closes the other half — a dashboard opened
mid-mission receives the current value on connect, instead of showing
"unknown" until the next fault.

The card renders `latched` / `clear` / `stale`, the typed `drop_reason`, the
publisher's `detail`, the `rskill` in flight, and the age of the last
transition. Publishers re-stamp at 1 Hz, so an age past ~3 s flips the card to
`stale`, meaning **unknown, not safe** — the publisher may be gone (hazard-log
HZ-0096-1). Without rclpy or the `openral_msgs` overlay the subscriber stays
inert and the card reads *waiting*; it never renders "clear" from an absence
of data. Read-only: it holds no publisher, no service client, and no
authority over the robot.

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

## Perception overlays on the camera tiles

The camera tiles show what the robot *sees*; the overlays show what it *made of
it*. `kind: detector` boxes and `kind: segmenter` masks are drawn on a canvas
layered over each tile's MJPEG frame, so a mis-grasp no longer has to be
diagnosed by reading a label list beside a picture.

**Detector boxes** stream today. `PerceptionOverlaySubscriber` opens a real
rclpy subscription to `/openral/perception/objects`
(`openral_msgs/PromptStamped`, BEST_EFFORT + VOLATILE + KEEP_LAST=5 — the
sensor-class profile `ros_image_detector_node` publishes with; a RELIABLE
subscriber would never match it) and decodes the `metadata_json` payload as an
`openral_core.ObjectsMetadata`, writing it to the store via
`TelemetryStore.set_perception_detections`. Each box renders as a coloured rect
with a `label 0.87` chip.

**Segmenter masks** do not stream. `openral_msgs/srv/SegmentInView` is a
*service*: it returns plural full-frame **mono8** `sensor_msgs/Image` masks
(255 = in mask), ordered area-ascending, with a parallel
`mask_scores_advisory`, one shot per attach/detach/regrasp event. So there is
nothing to subscribe to, and this package deliberately does not invent a topic.
What ships is the seam that side lands on: `mono8_mask_to_png_b64` (mono8 →
an LA PNG whose alpha *is* the mask, tintable in one canvas composite) and
`TelemetryStore.set_perception_masks`. The renderer already draws any
`(camera, masks[], scores[])` tuple it is handed, so a mask publisher lights
the overlay up with no frontend change. **Follow-up for the vision branch:** a
small optional latest-mask debug topic on the segmenter node is the cleanest
way to feed it — better there, next to the producer, than as unreachable code
here.

The advisory scores are displayed and **never** used to rank or filter. The
service documents why: a mis-aimed prompt was measured returning a
59.8%-of-frame mask at that model's *top* score of 0.977. Masks render in the
producer's area-ascending order.

Three properties worth knowing:

* **Coordinates are source-image pixels**, and the tile shows an
  aspect-preserving 320×240 thumbnail under `object-fit: cover`. The renderer
  maps through the cover transform using the overlay's own `frame_width` /
  `frame_height`, so boxes track the displayed size rather than sitting at a
  constant offset from what they describe.
* **`OPENRAL_DASHBOARD_FLIP_180` is honoured.** Detectors consume the raw
  camera topic while that env var rotates only the dashboard's display copy;
  the flag rides on each overlay so the renderer applies the same turn. Without
  it the boxes would land point-mirrored on a picture that looks right.
* **Stale overlays fade, then clear.** Freshness runs on the dashboard's own
  receipt clock (source stamps ride sim time in a sim deploy): an overlay dims
  after 1.5 s, is gone by 4 s, and is dropped immediately if the tile's frame
  is more than 2 s newer than it. The producing rSkill is named in the tile's
  bottom-right corner, so "whose boxes are these" is never a guess.

Instance colours are the one chroma the dashboard allows outside the
safety palette, and they live in `dashboard.js` (`OVERLAY_COLORS`) because they
are painted into a canvas, never onto UI chrome. Slots are assigned in fixed
order by a stable hash of the label, so a label keeps its colour across frames
instead of repainting when the detection count changes — and every mark is also
directly labelled, so identity is never colour-alone.

Without rclpy or the `openral_msgs` overlay the subscriber stays inert and the
tiles render exactly as they did before. Read-only and advisory: it decides
only what an operator *sees*, never what the robot does.

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

## Write-controls (`OPENRAL_DASHBOARD_WRITE_CONTROLS`)

> **Default: OFF.** These endpoints are pending safety-WG review and a
> hazard-log update. Do not enable in production until the safety WG
> has signed off.

Two guarded write endpoints are available when the flag is set:

```
POST /api/skill/execute   # dispatch an ExecuteRskill action goal (returns 202 on acceptance)
POST /api/param/set       # tune a non-safety ROS 2 parameter via ros2 param set
```

`POST /api/skill/execute` returns **HTTP 202** as soon as the action server
accepts the goal — it does **not** block on skill completion. The response body
includes `goal_id` for telemetry correlation. Execution progress is tracked via
the dashboard SSE stream and OTLP telemetry. The acceptance timeout is
configurable via `OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S` (default `12` s).

Enable them by starting the dashboard with the environment variable set:

```bash
OPENRAL_DASHBOARD_WRITE_CONTROLS=1 openral dashboard
```

The dashboard prints a loud `WARNING:` banner to stderr on startup when the
flag is on. The flag is also surfaced in `GET /api/config`:

```json
{"jaeger_ui_url": "...", "write_controls_enabled": true}
```

**Safety posture:**

- Both endpoints return `403` when the flag is off (default).
- `POST /api/param/set` also refuses any param name that matches a substring in
  a hard-coded safety denylist (`velocity`, `accel`, `force`, `torque`,
  `limit`, `workspace`, `estop`, `e_stop`, `deadman`, `dead_man`, `safety`,
  `safe`, `watchdog`) — these must be changed via a reviewed config or manifest,
  never the dashboard (CLAUDE.md §1.1).
- Every attempt (permitted or denied) is audit-logged at WARNING level before
  any subprocess is spawned, providing a paper trail.
- The safety kernel (`cpp/openral_safety_kernel`) remains the sole authority on
  whether a skill action proceeds — the dashboard shells out to
  `ros2 action send_goal /openral/execute_rskill`; the kernel disposes.
