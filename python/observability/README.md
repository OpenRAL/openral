# openral-observability

OpenTelemetry tracing, metrics and a structlog-to-OTLP log bridge for OpenRAL,
plus the robot dashboard (`openral dashboard`): the web server, its API and an
OTLP/HTTP receiver on one port.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. All workspace packages
move in lockstep with the workspace SemVer version.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

## What is in the package

| Module | Purpose |
|---|---|
| `semconv`, `metrics`, `tracing` | Normative span / event / metric names and span helpers (`rskill_span`, `inference_span`, `safety_span`, `reasoner_span`). Annotated index: [telemetry reference](https://github.com/OpenRAL/openral/blob/master/docs/reference/telemetry.md) |
| `_sdk` | `configure_observability(service_name=...)`, `configure_worker_observability`, `shutdown_observability`. With no OTLP endpoint the whole SDK is a no-op |
| `lifecycle`, `diagnostics`, `failure_bus` | `@log_lifecycle_errors`, the 1 Hz `/diagnostics` heartbeat, the `FailureTrigger` bus (`/openral/failure/*`) |
| `compliance`, `compliance_log` | Typed compliance records as OTLP logs, and the on-robot hash-chained log. See the [compliance log](https://github.com/OpenRAL/openral/blob/master/docs/reference/compliance-log.md) |
| `producer`, `propagation`, `system_metrics`, `tracing_lttng`, `replay/` | Robot-manifest log record, W3C traceparent propagation, 1 Hz CPU/RAM/GPU metrics, optional LTTng tracepoints, rosbag2/mcap-to-trace join (`openral replay`) |
| `dashboard/` | The server (`app`, `server`, `store`), the remote listener (`remote_auth`, `remote_router`, `tunnel`), the 3D-view relay (`bridge_relay`), pinned UI and viewer assets (`ui_assets`, `viz_assets`), and read-only ROS subscribers (`safety_status_subscriber`, `perception_overlay_subscriber`) |

User guide for the dashboard: [Dashboard quickstart](https://github.com/OpenRAL/openral/blob/master/docs/quickstart/dashboard.md).

## The robot dashboard

- **The UI** at `/` is the pinned robot console, a prebuilt Apache-2.0 bundle
  whose source is not in this repository (`dashboard/ui_assets.py`: url,
  sha256, size; fetched once into `$OPENRAL_CACHE_DIR/dashboard_assets/`), or a
  local build named by `OPENRAL_DASHBOARD_UI_DIR`. Without either,
  `static/fallback.html` shows the robot's status and an E-STOP button. Start-up
  logs `dashboard.ui_source`. `/viz/` serves a pinned Lichtblick web build for
  raw topic inspection.
- **The manifest** is written by the deploy launch (`--manifest <file>`) and,
  when the reasoner runs, also emitted as an `openral.robot.manifest` OTLP log
  record. `GET /api/manifest` serves the latest one.
- **Actions** run through `POST /api/actions/{id}` with `{"args": {...}}`,
  checked against the action's `args_schema`: `estop`, `estop_reset` (the
  kernel's `/openral/estop_reset`; `409` during its cooldown), `prompt` and
  `skill:<rskill id>` (an `ExecuteRskill` goal; `202` + `goal_id` once accepted).
  E-STOP never needs a manifest. Every action is announced on
  `/openral/operator_action` for the compliance recorder. The safety kernel
  still disposes all motion.
- **Cameras** are polled (`GET /api/camera/{source}/snapshot.jpg`);
  `GET /api/stream?images=0` is the SSE stream without thumbnails.
- **The 3D view** is drawn by the UI from `foxglove_bridge` through `WS /ws`.
  `bridge_relay.py` keeps only the newest unsent frame of heavy snapshot topics
  while the browser is slow. The bridge port is `--bridge-port` (default 8765).
- **Speech to text:** `POST /api/transcribe` runs a local faster-whisper model;
  audio never leaves the machine. The endpoint returns 503 if the `dashboard`
  extra's `faster-whisper` is missing.

### Listeners and roles

| Listener | Client | Mode / role | Allowed |
|---|---|---|---|
| `--host`:`--port` (default `127.0.0.1:4318`) | loopback | `local` / operator | everything the manifest offers |
| same, bound to a LAN address | any other host | `lan` / viewer | GETs, `estop`, `POST /api/viz/ticket`; OTLP ingest on `/v1/*` only with `OPENRAL_DASHBOARD_LAN_INGEST=1` |
| `127.0.0.1:--remote-port` (default 4319, or `$OPENRAL_DASHBOARD_REMOTE_PORT`) | the Cloudflare Tunnel | `remote` / the token's role | a valid ES256 bearer JWT per request (`remote_auth.py`, ADR-0129) |

The local listener refuses cross-origin state-changing requests (CSRF), a
loopback request with a foreign `Host` (DNS rebinding), and any request carrying
Cloudflare forwarding headers, so a tunnel pointed at it by mistake gets 403.
The remote listener opens when the robot is linked (`openral robot link <code>`,
enrolment at `~/.config/openral/robot.json` or `OPENRAL_ROBOT_ENROLMENT`). A
token may make any number of reads but one write; a replayed write is a 401.

Roles: **viewer** may press E-STOP; **operator** and **admin** may also reset,
prompt and run skills (a manifest's `min_role` may raise an action, never lower
it; only E-STOP is a viewer's). Remote control needs
`OPENRAL_DASHBOARD_REMOTE_CONTROL=1`; running skills needs
`OPENRAL_DASHBOARD_WRITE_CONTROLS=1` (exactly `1`). Both are off on the robot
unless its linked account turns them on; an OFF on either side wins, and a
change applies at the next start (ADR-0132). The dashboard prints a `WARNING:`
banner for either flag.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S` | 12 | How long an action waits for the goal to be accepted |
| `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` | openral.com origins | Origins the remote listener answers (CORS, `/ws` upgrade) |
| `OPENRAL_DASHBOARD_FLIP_180` | off | Rotates the dashboard's display copy of camera frames (overlays follow) |
| `OPENRAL_STT_MODEL` / `_DEVICE` / `_COMPUTE` | `base.en` / `cpu` / `int8` | faster-whisper model id, `cuda` for a GPU, CTranslate2 compute type |

The full table (OTLP, sampling, consent, compliance variables) is in the
[telemetry reference](https://github.com/OpenRAL/openral/blob/master/docs/reference/telemetry.md#configuration).

## Read-only ROS subscribers

Both are inert without `rclpy` or the `openral_msgs` overlay, hold no
publisher, service client or authority over the robot, and never render
"clear" or an overlay from an absence of data.

- **Safety status.** `SafetyStatusSubscriber` subscribes to the latched
  `/openral/safety_status` (`openral_msgs/SafetyStatus`, RELIABLE +
  TRANSIENT_LOCAL + KEEP_LAST=1; ADR-0096) and writes it to the store. The card
  shows `latched` / `clear` / `stale`, the drop reason, the rSkill in flight and
  the age of the last transition. Publishers re-stamp at 1 Hz, so an age past
  about 3 s reads `stale`, meaning **unknown, not safe**.
- **Perception overlays.** `PerceptionOverlaySubscriber` draws detector boxes
  (`/openral/perception/objects`, `openral_msgs/PromptStamped`, BEST_EFFORT) and,
  when the segmenter's `publish_debug_masks` parameter is on (default off),
  masks (`/openral/perception/masks`, `openral_msgs/SegmentMasks`) over the
  camera tiles. Advisory mask scores are displayed, never used to rank or
  filter. Nothing on the robot reads these overlays.

## mDNS discovery (`mdns` extra)

```
pip install openral-observability[mdns]
# in the workspace (an extra, not a group):
just sync --extra mdns
```

This pulls in [`zeroconf>=0.131`](https://github.com/python-zeroconf/python-zeroconf)
(LGPL-2.1, TSC-approved; an optional declared dependency, not vendored). When
`zeroconf` is importable, `run_dashboard` browses for `_openral-otlp._tcp.local.`
services, and advertises its own OTLP endpoint only when bound to a non-loopback,
non-wildcard IPv4 address. `GET /api/robots` lists what it found
(`{"enabled": false, "robots": []}` when the extra is absent). Discovery is
additive, never load-bearing.
