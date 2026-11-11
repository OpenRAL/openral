# `openral dashboard` — the robot's own page

One process per robot serves the **robot UI**, its API, and an embedded OTLP/HTTP
receiver. Locally and offline you open it on the robot; online, the same UI runs on
**openral.com/dashboard** and reaches the robot through its own Cloudflare Tunnel
(ADR-0129). The UI is robot-agnostic: it draws only the panels and actions the robot
declares in its **manifest**.

## Run it

One port serves everything. The default is **4318** (the OTLP/HTTP standard; `8000`
collided with `mkdocs serve` and most FastAPI demos).

```bash
openral dashboard            # 127.0.0.1:4318
# → OpenRAL dashboard: http://localhost:4318/ …
```

`openral deploy sim|run` starts it for you (opt out with `--no-dashboard`), together
with the read-only `foxglove_bridge` for the 3D view (opt out with `--no-foxglove`).

Point any other workload at it. **Sim does not emit OTel by default**:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  openral sim run --config scenes/sim/libero_spatial.yaml --rskill smolvla-libero
```

or let either side spawn the other: `openral dashboard --inprocess "openral sim run …"`,
or `--dashboard` on `openral sim run`, `deploy run` and `benchmark run` (the child
dashboard exits with the workload).

## The page

`/` serves the robot UI, the React app in `python/observability/ui/`. The dashboard
downloads its pinned release once (url + sha256 + size in
`openral_observability.dashboard.ui_assets.ROBOT_UI`) into
`$OPENRAL_CACHE_DIR/dashboard_assets/` (default `~/.cache/openral`). To serve a local
build instead, set `OPENRAL_DASHBOARD_UI_DIR=/path/to/ui/dist`.

When no UI is available (no release pinned yet, or an offline first start) `/` serves a
built-in fallback page: the robot's status and an **E-STOP** button.

What the UI shows comes from the **robot manifest**, which the deploy runtime emits at
start and again whenever the installed rSkills change (`/openral/skill_registry_changed`):

- **Panels** — robot identity, one camera per image sensor (with its `position` in the
  camera row: `left`, `center` or `right`, from `openral_core.camera_position`), joints, then map (SLAM),
  point cloud (octomap), reasoner, reward monitor (when that leg runs), the running
  rSkill, safety status and ledger, system health, metrics, events, world state, and the
  3D view when the bridge runs. The UI lays them out as an operator console: cameras
  over the 3D scene, and the rest as tabs in a right pane (Reasoner, Events, World,
  Safety, Skill, Robot, System) that each operator can show, hide and reorder.
- **Actions** — `estop`, `estop_reset` (with a "work cell clear?" confirmation),
  `prompt` (when the reasoner runs), and one `skill:<rskill id>` per installed rSkill
  the robot can run — the same set the reasoner may dispatch — whose form is the
  rSkill's `goal_params_schema`. Skills appear only with
  `OPENRAL_DASHBOARD_WRITE_CONTROLS=1`.

The deploy launch writes the manifest and hands it to the dashboard
(`openral dashboard --manifest <file>`); the reasoner, when it runs, re-sends it as an
OTLP record whenever the installed rSkills change, and the fresher one wins. Cameras are polled JPEG
snapshots (`/api/camera/<sensor>/snapshot.jpg`), never video.

## Who may do what

| Where the request comes from | Mode | Role | May |
|---|---|---|---|
| This machine (`127.0.0.1:4318`) | `local` | operator | everything the manifest offers |
| Another host, with `--host <LAN address>` | `lan` | viewer | read, press E-STOP, and export OTLP to `/v1/*` |
| openral.com, through the tunnel (`127.0.0.1:4319`) | `remote` | the user's role | see below |

Over the tunnel, every request carries a 5-minute token openral.com mints for that user
and that robot only. **viewer** may read and press E-STOP; **operator** and **admin**
may also reset the e-stop, prompt and run skills — but only when the robot's site sets
`OPENRAL_DASHBOARD_REMOTE_CONTROL=1` (default off: IEC 60204-1 expects a reset where
the hazard can be seen). E-STOP is always allowed, from every role and listener.
**Remote E-STOP is best-effort, not a safety function**: the physical e-stop stays
authoritative.

Every action is announced on `/openral/operator_action` and recorded by the compliance
recorder as an intervention, attributed to the signed-in user (or, locally, the client
address) — pseudonymised, see [compliance log](../reference/compliance-log.md).

A request whose `Origin` names another site than the page is refused (CSRF); the tunnel
listener accepts only the origins in `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` (default
`https://www.openral.com,https://dev.openral.com,https://viz.openral.com`).

## Reaching the robot from openral.com

Sign in on openral.com and add the robot (**Dashboard → Robots → Add robot**, with its
name and serial number; a free account has one remote robot, a Pro org as many as it
needs). The serial travels in the link: every deploy stamps it on its compliance records,
so a linked robot needs no `OPENRAL_UNIT_SERIAL`. It shows a one-time
command; run it on the robot, as the user that runs OpenRAL:

```bash
openral robot link <code>
```

That is the only step on the robot. Nothing to install and no root:

- the code's issuer (`iss`) must be `https://www.openral.com` or
  `https://dev.openral.com`, and its signing keys (`jwks`) must be on the same origin;
  any other code is refused, and so is a stored `robot.json` naming another site. A
  self-hosted issuer sets `OPENRAL_TRUSTED_ISSUERS` (comma-separated origins; it
  replaces the openral.com pair, and an origin listed there may also serve another
  listed issuer's keys);
- the robot's identity (no secrets) goes to `~/.config/openral/robot.json` and its tunnel
  token to `tunnel.env` beside it, both mode 0600;
- a pinned `cloudflared` (sha256-checked) is fetched into `~/.cache/openral`;
- the tunnel is installed as the **user** systemd service `openral-tunnel.service`, enabled
  with `loginctl enable-linger`, so it is up now and after every reboot, before anyone
  logs in. `--no-service` skips this; so does a machine without user systemd (a
  container). Then `openral dashboard` runs the tunnel as its own child, up exactly while
  the dashboard is.

Start `openral dashboard` (or the deploy, which starts it): it opens the remote listener on
`127.0.0.1:4319`, the tunnel's target, and the robot shows on openral.com/dashboard. While
the dashboard is down the tunnel stays up and openral.com shows the robot as unreachable.
The listener answers only `/api/*` and `/ws`, with a valid token; the OTLP receiver is
never reachable through it.

**Account settings.** Settings on openral.com (**Account → Settings**: the reasoner's API
key, model, endpoint and dialect, set by an org admin, or by you on a free account) reach
the robot at every deploy start: the link stored the robot's own credential
(`robot.secret`, 0600), the deploy fetches its settings with it, caches them
(`settings.json`, 0600) for offline starts, and puts them in the environment. A variable
already set on the robot wins.

**Settings in the console.** The console's left pane (⋯ next to your name → **Settings**)
edits the reasoner's Anthropic and OpenAI API keys and its model in place. On openral.com
it edits your account's and your org's settings (above). On the robot it edits the robot's
own (`robot-settings.json` beside the enrolment, 0600), which only the operator at the
robot can change (read-only from the LAN, never over the tunnel); a secret only ever shows
as `set …abcd`. A deploy applies them at start: a variable already set wins, then the
robot's own settings, then the account's.

Manage the service like any user unit:

```bash
systemctl --user status openral-tunnel.service
```

To unlink, remove the robot on openral.com (its tunnel is deleted there), then
`systemctl --user disable --now openral-tunnel.service` and delete
`~/.config/openral/robot.json`, `tunnel.env`, `robot.secret` and `settings.json`.

## The 3D view

The robot UI draws the 3D scene itself (three.js): the robot from its URDF, TF, the
occupancy grid (`/map`), a point cloud, collision boxes and payload markers
(`/openral/viz/*`) and the world-state objects, each a layer the operator can switch
off. It reads the same `/ws` bridge proxy as Lichtblick and subscribes only to the
layers that are on. Links are posed from `/tf`, so the model follows whatever
`robot_state_publisher` computed. The proxy relays heavy snapshot topics (images, point
clouds, maps, laser scans) latest-wins, so a slow tunnel drops stale frames instead of
falling behind; sampled streams (joint states, IMU, odometry) are never dropped by it.
From another machine on the LAN the page asks the robot for a 3D ticket first, as it
does through the tunnel.

The dashboard also serves a pinned [Lichtblick](https://github.com/Lichtblick-Suite/lichtblick)
web build (MPL-2.0, the open Foxglove Studio fork) at `/viz/`, downloaded on first use.
It connects to `foxglove_bridge` through the dashboard's `/ws` proxy — the bridge itself
stays on loopback and read-only. Remote and LAN viewers need a ticket from
`POST /api/viz/ticket` (valid 15 minutes, reusable for reconnects). `/viz/` opens with a
layout for this robot's cameras and base frame already in place. The robot UI links to it
from the System pane for raw topic inspection.

## Endpoints

| Path | What it serves |
|---|---|
| `GET /` | The robot UI, or the fallback page |
| `GET /viz/` | Lichtblick (3D) |
| `GET /api/config` | `mode`, `role`, `robot_id`, `write_controls`, `remote_control` |
| `GET /api/manifest` | The robot manifest (404 until the runtime sends one) |
| `GET /api/state` | One-shot JSON snapshot (`manifest_rev` bumps when the manifest changes) |
| `GET /api/stream[?images=0]` | Server-Sent Events; `images=0` drops camera thumbnails |
| `GET /api/camera/{sensor}/snapshot.jpg` | Latest frame of one camera |
| `POST /api/actions/{id}` | Run an action: body `{"args": {...}}` |
| `POST /api/viz/ticket` · `WS /ws` | The 3D view's ticket and bridge proxy |
| `GET /api/traces` · `GET /api/spans/{trace_id}` | Trace index (`openral replay`) |
| `POST /api/transcribe` | Speech to text on the host (local Whisper) |
| `POST /v1/{traces,metrics,logs}` | OTLP/HTTP receiver (local listener, from this machine or the LAN; never through the tunnel) |
| `GET /healthz` | `{"status": "ok"}` |

`POST /api/actions/{id}` answers `200 {"ok": true}` (`202` with a `goal_id` for a
skill), `400` for args that fail the action's schema, `403` for a role, listener or flag
that does not allow it, `404` for an unknown action, `409` for a refused reset or goal,
and `503` when ROS is not available.

## Running alongside Jaeger + Prometheus

For post-hoc trace analysis or metric history, bring up the dev compose stack and point
the workload at the OTel Collector (`http://localhost:4317`) instead; it fans out to
Jaeger (`http://localhost:16686`) and Prometheus (`http://localhost:9090`):

```bash
docker compose -f docker-compose.dev.yml up -d otelcol jaeger prometheus
```

Set `OPENRAL_JAEGER_UI_URL` and the UI links traces to Jaeger. One OTLP endpoint per
workload: either the dashboard or the collector.
