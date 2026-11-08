# openral-observability

OpenRAL observability — OpenTelemetry tracing + structlog→OTLP logging bridge for Skill / inference / safety spans

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.

## The robot dashboard

`openral dashboard` serves the robot UI, its API and an OTLP/HTTP receiver on one
port (ADR-0129; the user guide is `docs/quickstart/dashboard.md`):

- **The UI** at `/` is the pinned release of `python/observability/ui/`
  (`dashboard/ui_assets.py`: url + sha256 + size, fetched once into
  `$OPENRAL_CACHE_DIR/dashboard_assets/`), or a local build named by
  `OPENRAL_DASHBOARD_UI_DIR`. Without either, `static/fallback.html` shows the
  robot's status and an E-STOP button.
- **The manifest.** The deploy's reasoner node emits an `openral.robot.manifest`
  OTLP log record (`producer.emit_robot_manifest`, body = `openral_core.RobotManifest`
  JSON) at configure and on `/openral/skill_registry_changed`; the store caches the
  latest (`TelemetryStore.manifest()`, `manifest_rev` on every snapshot) and
  `GET /api/manifest` serves it.
- **Actions** run through `POST /api/actions/{id}` with `{"args": {...}}`, checked
  against the action's `args_schema` (`jsonschema`): `estop` (the persistent
  `EstopPublisher`, else a `ros2 topic pub` shell-out), `estop_reset` (the kernel's
  `/openral/estop_reset`; `409` during its cooldown), `prompt` (`openral prompt` on
  `/openral/prompt_in/dashboard`) and `skill:<rskill id>` (an `ExecuteRskill` goal;
  `202` + `goal_id` once accepted, `OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S`, default
  12 s). E-STOP never needs a manifest. Every action is announced on
  `/openral/operator_action` for the compliance recorder — after the stop for an
  E-STOP, before acting otherwise. The safety kernel still disposes all motion.
- **Cameras** are polled: `GET /api/camera/{source}/snapshot.jpg` re-serves the
  latest `sensors.read_latest` thumbnail; `GET /api/stream?images=0` is the SSE
  stream without those thumbnails.

### Listeners and roles

| Listener | Client | Mode / role | Allowed |
|---|---|---|---|
| `--host`:`--port` (default `127.0.0.1:4318`) | loopback | `local` / operator | everything the manifest offers |
| same, bound to a LAN address | any other host | `lan` / viewer | GETs, `estop`, `POST /api/viz/ticket` |
| `127.0.0.1:--remote-port` (default 4319 once linked) | the Cloudflare Tunnel | `remote` / the token's role | see below |

`proxy_headers` is off on both, so a forwarded-for header never makes a LAN client
loopback. The local listener refuses a state-changing request whose `Origin` names
another host:port than `Host` (CSRF; DNS rebinding is not stopped by this).

The **remote listener** opens when the robot is linked (`openral robot link <code>`
writes `~/.config/openral/robot.json`; `OPENRAL_ROBOT_ENROLMENT` overrides the path).
`dashboard/remote_auth.py` checks every request's `Authorization: Bearer <JWT>` in
ADR-0129 Interface B order: header `alg` ES256 and `typ` absent or `at+jwt`, `kid` in
the issuer's JWKS (cached on disk, refetched only for an unknown `kid`, at most once a
minute), the signature, `iss`, a single `aud` equal to this robot's origin, `exp`/`iat`
with 60 s leeway and `exp − iat ≤ 300`, the `sub`/`jti`/`org`/`role`/`robot` claims,
and `robot`/`org` equal to the enrolment. `OPTIONS` is answered before auth for the
origins in `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` (default
`https://www.openral.com,https://dev.openral.com,https://viz.openral.com`; never
`Allow-Credentials`), `/v1/*` answers 404, and `/api/stream` ends at the token's `exp`.
Authorization headers and viz tickets are never logged.

Roles: **viewer** may press E-STOP; **operator** and **admin** may also reset, prompt
and run skills — remotely only with `OPENRAL_DASHBOARD_REMOTE_CONTROL=1` (default off).
Skills need `OPENRAL_DASHBOARD_WRITE_CONTROLS=1` everywhere (default off; only the exact
value `1` counts). `GET /api/manifest` serves the actions as this dashboard enforces
them: no skills with write controls off, `remote` per the remote-control flag, only
`estop` for a LAN client. The dashboard prints a `WARNING:` banner for either flag.
**Pending safety-WG review**: the remote-control flag and its default, LAN read-only
plus E-STOP, and the removal of free-form parameter writes (ADR-0129 §6).

### The 3D view

`/viz/` serves a pinned Lichtblick web build (`dashboard/viz_assets.py`, MPL-2.0, its
source noted in `NOTICE.txt` beside it), fetched on first use without its source maps.
It reaches `foxglove_bridge` (loopback, read-only) through `WS /ws`, a proxy that pings
every 30 s and checks `Origin` on upgrade. Remote and LAN clients need a ticket from
`POST /api/viz/ticket` (256-bit, bound to the caller, reusable for 15 minutes; it also
opens `GET /api/viz/layout.json`, built by `openral_foxglove_bringup.layout`). The
bridge port is `--bridge-port` (default 8765).

### Speech to text

`POST /api/transcribe` runs a **local**
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) model on the captured audio
and returns the text; the audio never leaves the machine. Voice prompts in the robot UI
are a follow-up. `faster-whisper` ships with the `dashboard` extra (which `openral-cli`
already pulls in); the endpoint degrades to HTTP 503 if it is ever stripped. Whisper
weights are fetched from the Hugging Face Hub on first use and cached.

| Env var | Default | Notes |
| --- | --- | --- |
| `OPENRAL_STT_MODEL` | `base.en` | any faster-whisper model id (`tiny.en`, `small`, `large-v3`, …) |
| `OPENRAL_STT_DEVICE` | `cpu` | `cuda` to use a local GPU |
| `OPENRAL_STT_COMPUTE` | `int8` | CTranslate2 compute type (`int8`, `float16`, …) |

## Safety · current state (latched `/openral/safety_status`)

ADR-0096. Every other panel is fed by OTel over the embedded OTLP receiver. This one is not: `SafetyStatusSubscriber` opens a real rclpy
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

## Perception overlays on the camera tiles

The camera tiles show what the robot *sees*; the overlays show what it *made of
it*. `kind: detector` boxes and `kind: segmenter` masks are drawn on a canvas
layered over each camera snapshot, so a mis-grasp no longer has to be
diagnosed by reading a label list beside a picture.

**Detector boxes** stream today. `PerceptionOverlaySubscriber` opens a real
rclpy subscription to `/openral/perception/objects`
(`openral_msgs/PromptStamped`, BEST_EFFORT + VOLATILE + KEEP_LAST=5 — the
sensor-class profile `ros_image_detector_node` publishes with; a RELIABLE
subscriber would never match it) and decodes the `metadata_json` payload as an
`openral_core.ObjectsMetadata`, writing it to the store via
`TelemetryStore.set_perception_detections`. Each box renders as a coloured rect
with a `label 0.87` chip.

**Segmenter masks** are not a stream by nature: `openral_msgs/srv/SegmentInView`
is a *service* returning plural full-frame **mono8** `sensor_msgs/Image` masks
(255 = in mask), ordered area-ascending, with a parallel
`mask_scores_advisory`, one shot per attach/detach/regrasp event. The overlay
source is therefore a **diagnostic** re-publication of that reply, living next
to the producer:
`openral_perception_ros.segmenter_node` publishes `openral_msgs/SegmentMasks`
on `/openral/perception/masks` when its `publish_debug_masks` parameter is set
(**default false** — off, no publisher exists at all). `PerceptionOverlaySubscriber`
subscribes it at the matching sensor-class QoS (BEST_EFFORT + VOLATILE +
KEEP_LAST=1) and decodes each mask with `mono8_mask_to_png_b64` (mono8 → an LA
PNG whose alpha *is* the mask, tintable in one canvas composite) into
`TelemetryStore.set_perception_masks`.

Nothing on the robot reads that topic: the HAL's attachment producer takes its
masks from its own service reply and gates them on geometry. Dropping every
message here changes only what an operator sees.

The advisory scores are displayed and **never** used to rank or filter. The
service documents why: a mis-aimed prompt was measured returning a
59.8%-of-frame mask at that model's *top* score of 0.977. Masks render in the
producer's area-ascending order.

Rendering:

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

Instance colours are the one chroma the UI allows outside the safety palette,
painted into a canvas, never onto UI chrome. Slots are assigned in fixed order by
a stable hash of the label, so a label keeps its colour across frames instead of
repainting when the detection count changes — and every mark is also directly
labelled, so identity is never colour-alone.

Without rclpy or the `openral_msgs` overlay the subscriber stays inert and the
tiles render without overlays. Read-only and advisory: it decides only what an
operator *sees*, never what the robot does.

## mDNS discovery (`mdns` extra)

Install the optional `mdns` extra to let `openral dashboard` advertise itself
on the LAN and browse for other OpenRAL services:

```
pip install openral-observability[mdns]
# or, inside the OpenRAL workspace (an extra of openral-observability, not a group):
just sync --extra mdns
```

This pulls in [`zeroconf>=0.131`](https://github.com/python-zeroconf/python-zeroconf)
(LGPL-2.1, TSC-approved 2026-06-21; used unmodified as an optional declared
dependency, not vendored).

When `zeroconf` is importable, `run_dashboard` starts a `Discovery` instance
that:

- **Browses** for `_openral-otlp._tcp.local.` services on the LAN (always).
- **Advertises** the dashboard's own OTLP endpoint on the LAN **only when the
  bind address is a non-loopback, non-wildcard IPv4 address**. A loopback
  (`127.0.0.1`) bind is meaningless to advertise and a wildcard (`0.0.0.0`)
  one ambiguous, so both are browse-only.

Discovered services are listed by a read-only endpoint:

```
GET /api/robots
→ {"enabled": true, "robots": [{name, addresses, port, properties, last_seen}, …]}
```

When the `mdns` extra is absent or `zeroconf` fails to start, the endpoint
returns `{"enabled": false, "robots": []}`; discovery is additive, never
load-bearing.
