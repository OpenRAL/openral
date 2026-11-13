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

- **The UI** at `/` is the pinned robot console, a prebuilt bundle whose source is
  not in this repository (`dashboard/ui_assets.py`: url + sha256 + size, fetched once
  into `$OPENRAL_CACHE_DIR/dashboard_assets/`; its license and the open alternatives
  are in `docs/quickstart/dashboard.md`, "The page"), or a local build named by
  `OPENRAL_DASHBOARD_UI_DIR`. Without either, `static/fallback.html` shows the
  robot's status and an E-STOP button. Start-up logs `dashboard.ui_source`: which of
  the three `/` serves. Both are served under a strict
  `Content-Security-Policy` (own scripts only, an HTML file's inline scripts by
  sha256; framed only by this origin and openral.com); `/viz/` gets Lichtblick's own
  (`'unsafe-eval'`, `'wasm-unsafe-eval'`, blob: workers).
- **The manifest.** The deploy launch writes one and passes `--manifest <file>`, so a
  deploy without the reasoner has one too; the reasoner, when it runs, emits an
  `openral.robot.manifest` OTLP log record (`producer.emit_robot_manifest`, body =
  `openral_core.RobotManifest` JSON) at configure and on
  `/openral/skill_registry_changed`. The store keeps the latest of either
  (`TelemetryStore.manifest()`, `manifest_rev` on every snapshot) and
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
| same, bound to a LAN address | any other host | `lan` / viewer | GETs, `estop`, `POST /api/viz/ticket`; OTLP ingest on `/v1/*` (a robot computer exporting to a LAN dashboard, ADR-0017) only with `OPENRAL_DASHBOARD_LAN_INGEST=1` (default off) |
| `127.0.0.1:--remote-port` (default 4319 once linked) | the Cloudflare Tunnel | `remote` / the token's role | see below |

`proxy_headers` is off on both, so a forwarded-for header never makes a LAN client
loopback. The local listener refuses a state-changing request whose `Origin` names
another host:port than `Host` (CSRF), a loopback request whose `Host` is not this
machine (DNS rebinding), and any request or `/ws` upgrade carrying Cloudflare's
forwarding headers (`cf-ray`, `cf-connecting-ip`, `cf-warp-tag-id`, `cdn-loop: cloudflare`):
a tunnel pointed at the local port by mistake gets 403, never the local operator. The
deploy's own exporters always send to `127.0.0.1`.

The **remote listener** opens when the robot is linked (`openral robot link <code>`
writes `~/.config/openral/robot.json`, `OPENRAL_ROBOT_ENROLMENT` overrides the path, and
`tunnel.env` beside it). The robot's Cloudflare Tunnel (`dashboard/tunnel.py`, a pinned
`cloudflared`) runs as the user service `openral-tunnel.service`, which survives reboots,
or, without user systemd, as the dashboard's own child process.
`dashboard/remote_auth.py` checks every request's `Authorization: Bearer <JWT>` in
ADR-0129 Interface B order: header `alg` ES256 and `typ` absent or `at+jwt`, `kid` in
the issuer's JWKS (cached on disk, refetched only for an unknown `kid`, at most once a
minute), the signature, `iss`, a single `aud` equal to this robot's origin, `exp`/`iat`
with 60 s leeway and `exp − iat ≤ 300`, the `sub`/`jti`/`org`/`role`/`robot` claims,
and `robot`/`org` equal to the enrolment. `OPTIONS` is answered before auth for the
origins in `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` (default
`https://www.openral.com,https://dev.openral.com,https://viz.openral.com`; never
`Allow-Credentials`), `/v1/*` answers 404, and `/api/stream` ends at the token's `exp`.
A token may make any number of reads but **one write**: its `jti` is held (in memory,
until `exp` + leeway) and a second `POST`/`PUT` with it — a replay — is a 401, which
makes the console mint a fresh token; E-STOP is never refused for it. The console sends
every write but E-STOP with a fresh token (`ui/src/lib/auth.js`).
Authorization headers and viz tickets are never logged.

Roles: **viewer** may press E-STOP; **operator** and **admin** may also reset, prompt
and run skills (a manifest's `min_role` may raise an action above operator, never lower
it: only E-STOP is a viewer's) — remotely only with `OPENRAL_DASHBOARD_REMOTE_CONTROL=1` (default off).
Skills need `OPENRAL_DASHBOARD_WRITE_CONTROLS=1` everywhere (default off; only the exact
value `1` counts). `GET /api/manifest` serves the actions as this dashboard enforces
them: no skills with write controls off, `remote` per the remote-control flag, only
`estop` for a LAN client. The dashboard prints a `WARNING:` banner for either flag.
**Pending safety-WG review**: the remote-control flag and its default, LAN read-only
plus E-STOP, and the removal of free-form parameter writes (ADR-0129 §6).

### The 3D view

The robot UI draws the 3D scene itself (three.js) from
`foxglove_bridge` through `WS /ws`: the URDF (meshes through the bridge's `assets`
capability), link poses from `/tf`, the occupancy grid, a point cloud, collision and
payload markers and the world-state objects. It subscribes only to the layers that are
on. `/ws` runs every bridge→browser frame through `dashboard/bridge_relay.py`: for heavy
snapshot topics (images, point clouds, occupancy grids, laser scans) only the newest
unsent message per subscription is kept while the browser is slow; everything else is
relayed in order, and once 8 MiB of ordered frames wait unsent the proxy stops reading
the bridge, so the bridge's own `send_buffer_limit` sheds load. The number of coalesced
frames is logged when the connection ends (`dashboard.bridge_relay_coalesced`).

For raw topic inspection `/viz/` also serves a pinned Lichtblick web build (`dashboard/viz_assets.py`, MPL-2.0, its
source noted in `NOTICE.txt` beside it), fetched on first use without its source maps.
Both reach `foxglove_bridge` (loopback, read-only) through `WS /ws`, a proxy that pings
every 30 s and checks `Origin` on upgrade. Remote and LAN clients need a ticket from
`POST /api/viz/ticket` (256-bit, bound to the caller and to the listener that minted
it, reusable for 15 minutes but never past the minting token's `exp`; LAN, loopback and
remote tickets are separate pools, so a LAN host cannot evict a remote viewer's). The
`/viz/` index.html carries this robot's layout (`openral_foxglove_bringup.layout`) in
Lichtblick's default-layout slot, so it opens on the robot's cameras and base frame; its
CSP allows `'unsafe-eval'` (Lichtblick v1.30 will not start without `new Function`) and
the inline script by hash, and `connect-src 'self'` only (as the UI's), so a crafted
`?ds.url=wss://elsewhere` link cannot point the viewer at a foreign bridge. The bridge port is `--bridge-port` (default 8765).

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
