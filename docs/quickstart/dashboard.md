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

`/` serves the robot UI (the robot console), a prebuilt web bundle the dashboard
downloads once (url + sha256 + size pinned in
`openral_observability.dashboard.ui_assets.ROBOT_UI`) into
`$OPENRAL_CACHE_DIR/dashboard_assets/` (default `~/.cache/openral`). To serve another
build instead, set `OPENRAL_DASHBOARD_UI_DIR` to a directory holding its `index.html`.
At start the dashboard logs one `dashboard.ui_source` line saying what `/` serves: the
pinned bundle with its URL, a local directory, or the fallback page.

When no console is available (an offline first start, or `OPENRAL_DASHBOARD_UI_DIR`
naming a directory without an `index.html`) `/` serves a built-in fallback page: the
robot's status and an **E-STOP** button.

**The console's license and source.**

- The console bundle is licensed Apache-2.0, like the rest of OpenRAL.
- It is prebuilt (minified), and its source is not published (ADR-0131). The
  dashboard server, the robot API and the fallback page with its E-STOP button are
  in this repository.
- To serve a different UI, point `OPENRAL_DASHBOARD_UI_DIR` at your own build, or at
  an empty directory to get the fallback page. The robot API
  ([Endpoints](#endpoints)) is the public contract for third-party UIs.

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
| Another host, with `--host <LAN address>` | `lan` | viewer | read and press E-STOP; export OTLP to `/v1/*` only with `OPENRAL_DASHBOARD_LAN_INGEST=1` |
| openral.com, through the tunnel (`127.0.0.1:4319`) | `remote` | the user's role | see below |

Over the tunnel, every request carries a 5-minute token openral.com mints for that user
and that robot only. **viewer** may read and press E-STOP; **operator** and **admin**
may also reset the e-stop, prompt and run skills — but only with
`OPENRAL_DASHBOARD_REMOTE_CONTROL=1`. A robot linked to openral.com gets it, and
`OPENRAL_DASHBOARD_WRITE_CONTROLS=1`, from its account (ADR-0132): both are on unless
turned off as **Remote control** and **Skill controls** under **Settings** on openral.com
(by an org admin, or by the owner of a free robot). An OFF on either side wins: the
account's OFF turns a switch off even when the robot sets `1`, and a site that wants every
reset done where the hazard can be seen (IEC 60204-1) sets
`OPENRAL_DASHBOARD_REMOTE_CONTROL=0` on the robot, which openral.com can't turn back on.
Only `1` turns a switch on, and a cached copy never does. Remote control fails closed: it
is off whenever openral.com couldn't confirm it at start (unreachable, an error, or a cache
that can't be written), even on a robot set to `1`. An unlinked robot keeps both off. A change is **not live**: it applies at the next start (see **At deploy start**;
restart `openral dashboard` for remote control, the deploy for skills). E-STOP is always
allowed, from every role and listener.
**Remote E-STOP is best-effort, not a safety function**: the physical e-stop stays
authoritative.

Every action is announced on `/openral/operator_action` and recorded by the compliance
recorder as an intervention, attributed to the signed-in user (or, locally, the client
address) — pseudonymised, see [compliance log](../reference/compliance-log.md).

A request whose `Origin` names another site than the page is refused (CSRF); the tunnel
listener accepts only the origins in `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` (default
`https://www.openral.com,https://dev.openral.com,https://viz.openral.com`). A token
over the tunnel is good for any number of reads but one write (a replayed token cannot
prompt or reset twice; E-STOP is never refused). A request on the local port that carries
Cloudflare's forwarding headers is refused: only the tunnel's own port takes forwarded
traffic. A manifest may raise an action's role, never lower it under operator.

**OTLP from other hosts is off by default.** A LAN host could otherwise fill the event
log or fake telemetry the operator reads. To collect from another robot computer on the
LAN (ADR-0017), start the dashboard with `OPENRAL_DASHBOARD_LAN_INGEST=1`; exporters on
this machine (the deploy sends to `127.0.0.1`) are unaffected. A dashboard bound to one
LAN address (not `0.0.0.0`) sees its own exporters arrive from that address, so it
needs the opt-in too.

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

**Several linked robots on one host** (`openral deploy sim` of a scene with several robots,
each its own openral.com robot): link each one with its own enrolment,
`OPENRAL_ROBOT_ENROLMENT=~/.config/openral/robots/<scene id>/<robot name>/robot.json openral robot
link <its code>`. openral.com points every robot's tunnel at `localhost:4319`, so the deploy
gives each robot's remote listener a port of its own (`4350 + i`, through
`OPENRAL_DASHBOARD_REMOTE_PORT`) and runs one router on `127.0.0.1:4319` that sends each
request to its robot by host (`r-<id>.openral.com`); every robot still checks its own tokens.
Each robot's dashboard runs its own tunnel. The user service keeps the robot it was linked
for: `openral robot link` never replaces a service that runs another robot's tunnel. Port
4319 must be free while the scene runs (a host robot's own dashboard holding it leaves the
scene's robots unreachable from openral.com; the router logs it to `router.log`).

**Settings levels.** openral.com holds settings at four levels. Each setting lives at exactly
one of them, and a robot reads two: its own robot settings and one account.

| Level | What is set there | Who changes it | Reaches the robot |
|---|---|---|---|
| **Robot** | its name, reasoner model, curated rSkills | an org admin (the owner, for a free robot) on openral.com's Robot settings page; the operator at the robot, in the console's ⚙ | yes; wins over the account |
| **Organisation** | the provider API keys, the remote-control and skill-control switches | an org admin, in **Account → Settings** | yes, for the org's robots |
| **Personal** | the same keys and switches | you, in **Account → Settings** | only for your own free robots, never an org's |
| **Fleet** | the fleet's super reasoner (model, workcells, steps at once) | an org admin, on the fleet's settings page | no; openral.com uses them itself |

An account's model from before models were set per robot is the fallback for a robot with
none of its own.

**Two reasoners, two models.** Don't confuse them:

- The **robot's reasoner** (`OPENRAL_REASONER_MODEL`, a robot setting) is the S2 planner of
  one robot: it runs with that robot's deploy, picks its rSkills and talks to its operator.
  The robot gets its model and the account's key for it at deploy start.
- The **fleet's super reasoner** (`OPENRAL_FLEET_MODEL`, a fleet setting, ADR-0135; a Pro
  org's) runs on no robot: its loop runs in the fleet console on openral.com, which calls its
  model with the org's key. It splits a fleet goal into steps and gives each to one robot as
  a prompt (`POST /api/actions/prompt` with `new_goal` and a `fleet-…:<n>` `correlation_id`),
  which that robot's own reasoner carries out with its own rSkills; it follows each step
  through the robot's `/api/mission`, and relays a robot's question to the person (or, with
  the fleet's "Super reasoner answers robots" on, answers it). Its settings are never sent to
  a robot, and a robot's model never sets it.

**At deploy start.** A linked robot fetches its settings from openral.com
(`<site>/api/robot/settings`, the account's with the robot's own on top) with its own
credential (`robot.secret`, 0600), caches them (`settings.json`, 0600) for offline starts,
and puts them in the environment. A variable already set on the robot wins, so a change made
on openral.com or in the console reaches the reasoner at the next deploy start, and never
while that variable is set. The dashboard's switches are the exception: openral.com's OFF
turns one off over the robot's own `1`, a cached copy (used when openral.com can't be
reached) never turns one on, and remote control stays off when openral.com couldn't confirm
it. Only a refused credential (401) deletes the cache.

**The robot's rSkills.** A robot's **Robot settings** page on openral.com lists the
rSkills it may run (`OPENRAL_RSKILLS`, comma-separated Hugging Face repo ids such as
`OpenRAL/rskill-smolvla-so101-eraser_place-bf16`): the reasoner offers only those. At
`openral deploy run` / `deploy sim` start, right after the account settings apply and
before the reasoner starts, the robot installs each listed rSkill it doesn't have yet,
one at a time, printing one line per rSkill (`installing …`, `installed`, `already
installed`, or `skipped` and why). Only Apache-2.0 / MIT / BSD rSkills install on their
own; a proprietary or non-commercial one is skipped with the `openral rskill install
<id>` to run by hand (it asks you to accept the licence). A gated OpenRAL Pro rSkill this
robot's Hugging Face token can't read (401/403) says to request access on Hugging Face.
No failure (offline, Hub down, a bad manifest) stops the deploy: it starts with what is
installed. `--no-rskill-install` or `OPENRAL_RSKILL_AUTO_INSTALL=0` turns this off.

**Settings in the console.** Robot settings have one source of truth: openral.com. The
console's header ⚙ shows the same Robot settings as openral.com's Robot settings page (its
name, its reasoner model, its curated rSkills): the robot's dashboard relays them to
`<site>/api/robot/console` with its own credential (`GET`/`PUT`/`PATCH /api/robot-settings`),
and openral.com decides who may change what. The operator at the robot may read and change
them; a LAN viewer may only read them; over the tunnel they may only be read, with remote
control on (change them on openral.com, where you are signed in). An unlinked robot, or one
whose credential openral.com refuses, answers 409, and 502 while openral.com can't be
reached. The robot keeps no reasoner settings of its own: a deploy applies the account's at
start (above), and a variable set on the robot still wins; a `robot-settings.json` left by an
older release is no longer read (the deploy says so) and can be deleted.

Manage the service like any user unit:

```bash
systemctl --user status openral-tunnel.service
```

To unlink, remove the robot on openral.com (its tunnel is deleted there), then
`systemctl --user disable --now openral-tunnel.service` and delete
`~/.config/openral/robot.json`, `tunnel.env`, `robot.secret` and `settings.json` (and
`robot-settings.json`, if an older release left one).

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
`POST /api/viz/ticket` (valid 15 minutes, reusable for reconnects, never past the
token that asked for it, and only on the port that issued it). The page may connect only
to the robot it came from, so a `?ds.url=` link cannot point it at another bridge. `/viz/` opens with a
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
| `GET /api/mission` | For programs: the mission (goal, the prompt's `correlation_id`, each task's status), the reasoner's last tick, `awaiting_operator` with the `question` it asked, the last dashboard skill run by its `goal_id` (`skill_result`: `running`, then `finished` with `success` and `failure_reason`), and `stopped` / `estopped` / `latched` with `latch_reason` and `latch_detail` (`{kind, link, other, depth_m}`: what the latching stop hit, e.g. `attached:trk:2` into `voxel_464002`, 14.5 mm deep). A robot of a scene with several robots also reports `world_pose` (`{x, y, yaw, frame: "world"}`: where its map sits in the scene's shared frame; its graph publishes the same as a static `world -> map` TF). A few hundred bytes |
| `GET /api/mission/stream` | The same as Server-Sent Events, one event per change |
| `GET /api/camera/{sensor}/snapshot.jpg` | Latest frame of one camera |
| `POST /api/actions/{id}` | Run an action: body `{"args": {...}}` |
| `POST /api/viz/ticket` · `WS /ws` | The 3D view's ticket and bridge proxy |
| `GET /api/traces` · `GET /api/spans/{trace_id}` | Trace index (`openral replay`) |
| `POST /api/transcribe` | Speech to text on the host (local Whisper) |
| `POST /v1/{traces,metrics,logs}` | OTLP/HTTP receiver (local listener, from this machine, or the LAN with `OPENRAL_DASHBOARD_LAN_INGEST=1`; never through the tunnel) |
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
