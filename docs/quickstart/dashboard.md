# `openral dashboard`: the robot's own page

Goal: open the live page for a robot, know who may do what on it, and reach it remotely if you want.

One process per robot serves the **robot UI**, its API and an embedded OTLP/HTTP receiver. Locally and offline you open it on the robot. Online, the same UI runs on [openral.com](https://openral.com) and reaches the robot through its own Cloudflare Tunnel (ADR-0129). The UI is robot-agnostic: it draws only the panels and actions the robot declares in its manifest.

## Run it

One port serves everything: **4318** by default.

```bash
uv run openral dashboard            # 127.0.0.1:4318
```

You should see `OpenRAL dashboard: http://localhost:4318/ ...`.

`openral deploy sim|run` starts it for you (opt out with `--no-dashboard`), plus the read-only `foxglove_bridge` for the 3D view (opt out with `--no-foxglove`).

Point other workloads at it. Sim does not emit OTel by default:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  uv run openral sim run --config scenes/sim/libero_spatial.yaml --rskill smolvla-libero
```

Or let one side spawn the other: `openral dashboard --inprocess "openral sim run ..."`, or pass `--dashboard` to `sim run`, `deploy run` and `benchmark run` (the child dashboard exits with the workload).

## The page

`/` serves the robot UI, a prebuilt Apache-2.0 web bundle (source private, ADR-0131). The dashboard downloads it once, pinned by url and sha256 in `openral_observability.dashboard.ui_assets.ROBOT_UI`, into `$OPENRAL_CACHE_DIR/dashboard_assets/` (default `~/.cache/openral`), and logs one `dashboard.ui_source` line saying what `/` serves. Set `OPENRAL_DASHBOARD_UI_DIR` to a directory with an `index.html` to serve another build. With no console available (an offline first start, or a directory without `index.html`), `/` serves a built-in fallback page: robot status and an **E-STOP** button.

What the UI shows comes from the **robot manifest**, which the deploy writes at start (`openral dashboard --manifest <file>`) and the reasoner re-sends whenever the installed rSkills change; the fresher one wins.

- **Panels**: robot identity, one camera per image sensor (`left`, `center` or `right`), joints, map, point cloud, reasoner, reward monitor (when it runs), the running rSkill, safety status, system health, metrics, events, world state, and the 3D view when the bridge runs. Operators can show, hide and reorder the tabs.
- **Actions**: `estop`, `estop_reset` (with a "work cell clear?" confirmation), `prompt` (when the reasoner runs), and one `skill:<rskill id>` per installed rSkill the robot can run, whose form is the rSkill's `goal_params_schema`. Skills appear only with `OPENRAL_DASHBOARD_WRITE_CONTROLS=1`.

Cameras are polled JPEG snapshots, never video.

## Who may do what

| Where the request comes from | Mode | Role | May |
|---|---|---|---|
| This machine (`127.0.0.1:4318`) | `local` | operator | everything the manifest offers |
| Another host, with `--host <LAN address>` | `lan` | viewer | read and press E-STOP; export OTLP to `/v1/*` only with `OPENRAL_DASHBOARD_LAN_INGEST=1` |
| openral.com through the tunnel (`127.0.0.1:4319`) | `remote` | the user's role | viewer: read and E-STOP. operator/admin: also reset, prompt and run skills, only with `OPENRAL_DASHBOARD_REMOTE_CONTROL=1` |

E-STOP is allowed from every role and listener. **Remote E-STOP is best-effort, not a safety function: the physical e-stop stays authoritative.** Every action is announced on `/openral/operator_action` and recorded by the compliance recorder as an intervention ([compliance log](../reference/compliance-log.md)).

Safeguards on the network paths:

| Safeguard | Behavior |
|---|---|
| Tunnel tokens | Short-lived, minted by openral.com for one user and one robot; any number of reads, but one write per token (E-STOP is never refused). |
| Origin check | A request whose `Origin` names another site than the page is refused (CSRF). The tunnel listener accepts only `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` (default `https://www.openral.com,https://dev.openral.com,https://viz.openral.com`). |
| Forwarded traffic | A request on the local port carrying Cloudflare forwarding headers is refused; only the tunnel's own port takes forwarded traffic. |
| Remote listener | Answers only `/api/*` and `/ws`, with a valid token. The OTLP receiver is never reachable through it. |
| OTLP from other hosts | Off by default, so a LAN host cannot fill the event log or fake telemetry. Set `OPENRAL_DASHBOARD_LAN_INGEST=1` to collect from another robot computer. A dashboard bound to one LAN address needs it too. |
| Remote control and skill controls | A linked robot gets them from its openral.com account (ADR-0132). OFF on either side wins; only `1` turns a switch on; remote control fails closed when openral.com cannot confirm it at start. A change applies at the next start. To require every reset on site (IEC 60204-1), set `OPENRAL_DASHBOARD_REMOTE_CONTROL=0` on the robot. An unlinked robot keeps both off. |
| Action roles | A manifest may raise an action's required role, never lower it under operator. |

## Reach the robot from openral.com

Platform behavior (accounts, settings levels, fleet features, org limits) is documented on [openral.com](https://openral.com), not here. On the robot you need one step.

1. Sign in on openral.com and add the robot (**Dashboard, Robots, Add robot**, with its name and serial number). It shows a one-time command.
2. Run it on the robot, as the user that runs OpenRAL (no root, nothing to install):

```bash
uv run openral robot link <code>
```

You should see the link succeed. Then start `openral dashboard` (or the deploy); the robot appears on openral.com/dashboard. While the dashboard is down the tunnel stays up and openral.com shows the robot as unreachable.

What `link` does:

- Accepts only codes whose issuer is `https://www.openral.com` or `https://dev.openral.com`, with signing keys on the same origin. A self-hosted issuer sets `OPENRAL_TRUSTED_ISSUERS` (comma-separated origins; replaces the pair).
- Stores the robot's identity (no secrets) in `~/.config/openral/robot.json` and its tunnel token in `tunnel.env` beside it, both mode 0600; fetches a sha256-pinned `cloudflared` into `~/.cache/openral`.
- Installs the user systemd service `openral-tunnel.service` (with `loginctl enable-linger`), so the tunnel is up after every reboot. `--no-service` skips it, as does a machine without user systemd (a container); the dashboard then runs the tunnel as its own child.
- At deploy start a linked robot fetches its settings from openral.com (cached for offline starts); a variable already set on the robot wins. A robot's rSkill list (`OPENRAL_RSKILLS`, Hugging Face repo ids) is installed at `deploy run|sim` start: Apache-2.0 / MIT / BSD rSkills install on their own, others are skipped with the `openral rskill install <id>` to run by hand. A failure never stops the deploy. `--no-rskill-install` or `OPENRAL_RSKILL_AUTO_INSTALL=0` turns this off.

**Several linked robots on one host** (a multi-robot `deploy sim` scene): link each with its own enrolment and no service: `OPENRAL_ROBOT_ENROLMENT=~/.config/openral/robots/<scene id>/<robot name>/robot.json openral robot link --no-service <code>`. The deploy gives each robot's remote listener its own port (`4350 + i`, through `OPENRAL_DASHBOARD_REMOTE_PORT`) and runs one router on `127.0.0.1:4319`; every robot still checks its own tokens. Port 4319 belongs to the router while the scene runs, so a host robot's own dashboard started then keeps its local listener and E-STOP but turns its remote listener off (it warns).

To unlink, remove the robot on openral.com, then:

```bash
systemctl --user disable --now openral-tunnel.service
rm ~/.config/openral/{robot.json,tunnel.env,robot.secret,settings.json}
```

## The 3D view

The UI draws the scene itself (three.js): the robot from its URDF and `/tf`, the occupancy grid, point cloud, collision boxes and payload markers (`/openral/viz/*`) and world-state objects, each a switchable layer. It reads the dashboard's `/ws` proxy to `foxglove_bridge`, which stays on loopback and read-only. The proxy relays heavy topics latest-wins, so a slow tunnel drops stale frames; joint states, IMU and odometry are never dropped. From another machine the page asks the robot for a 3D ticket first.

A pinned [Lichtblick](https://github.com/Lichtblick-Suite/lichtblick) web build (MPL-2.0) is served at `/viz/` for raw topic inspection, downloaded on first use. Remote and LAN viewers need a ticket from `POST /api/viz/ticket`; the page may connect only to the robot it came from.

## Endpoints

| Path | What it serves |
|---|---|
| `GET /` | The robot UI, or the fallback page |
| `GET /viz/` | Lichtblick (3D) |
| `GET /api/config` | `mode`, `role`, `robot_id`, `write_controls`, `remote_control` |
| `GET /api/manifest` | The robot manifest (404 until the runtime sends one) |
| `GET /api/state` | One-shot JSON snapshot (`manifest_rev` bumps when the manifest changes) |
| `GET /api/stream[?images=0]` | Server-Sent Events; `images=0` drops camera thumbnails |
| `GET /api/mission` | Programmatic status: the mission (goal, `correlation_id`, task statuses), the reasoner's last tick, `awaiting_operator` and its `question`, the last dashboard skill run (`skill_result`), `stopped` / `estopped` / `latched` with `latch_reason` and `latch_detail`, and `world_pose` in multi-robot scenes |
| `GET /api/mission/stream` | The same as Server-Sent Events, one event per change |
| `GET /api/camera/{sensor}/snapshot.jpg` | Latest frame of one camera |
| `POST /api/actions/{id}` | Run an action: body `{"args": {...}}` |
| `POST /api/viz/ticket` · `WS /ws` | The 3D view's ticket and bridge proxy |
| `GET /api/traces` · `GET /api/spans/{trace_id}` | Trace index (`openral replay`) |
| `POST /api/transcribe` | Speech to text on the host (local Whisper) |
| `POST /v1/{traces,metrics,logs}` | OTLP/HTTP receiver (this machine, or the LAN with `OPENRAL_DASHBOARD_LAN_INGEST=1`; never through the tunnel) |
| `GET /healthz` | `{"status": "ok"}` |

`POST /api/actions/{id}` answers `200 {"ok": true}` (`202` with a `goal_id` for a skill), `400` for args that fail the action's schema, `403` for a role, listener or flag that does not allow it, `404` for an unknown action, `409` for a refused reset or goal, `503` when ROS is not available.

## Alongside Jaeger and Prometheus

For post-hoc traces or metric history, bring up the dev stack and send the workload to the OTel Collector (`http://localhost:4317`) instead; it fans out to Jaeger (`:16686`) and Prometheus (`:9090`). Use one OTLP endpoint per workload, the dashboard or the collector.

```bash
docker compose -f docker-compose.dev.yml up -d otelcol jaeger prometheus
```

Set `OPENRAL_JAEGER_UI_URL` and the UI links traces to Jaeger.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/` shows a bare page with E-STOP | No console bundle: first start offline, or `OPENRAL_DASHBOARD_UI_DIR` has no `index.html`. Check the `dashboard.ui_source` log line. |
| No `skill:` actions in the UI | Start with `OPENRAL_DASHBOARD_WRITE_CONTROLS=1`. |
| Remote user cannot reset or prompt | Remote control is off: `OPENRAL_DASHBOARD_REMOTE_CONTROL=1` and the account's switch must both allow it. |
| Another robot's telemetry never shows | Start the dashboard with `OPENRAL_DASHBOARD_LAN_INGEST=1`. |
| Scene robots unreachable from openral.com | A host robot's dashboard holds port 4319; stop it before the scene starts. |
