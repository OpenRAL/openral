# `openral dashboard` — live debugging UI

A single-page, read-only debug pane over the OpenRAL OTel stream.
Renders the most recent `rskill.execute`, `skill.chunk_inference`,
and `safety.check` spans, rolling metric histograms, and an event
log — live, no Jaeger required.

See also [ADR-0017](../adr/0017-dashboard-otlp-receiver.md) for the
"embedded receiver, not in-process exporter" design choice.

## Run it

The dashboard is a `openral` subcommand. The HTTP port serves the UI, the
SSE event stream, **and** an embedded OTLP/HTTP receiver. The default
port is **4318** (the OTLP/HTTP standard); it used to be `8000`, but
that collided with `mkdocs serve` (`just docs`) and most FastAPI
demos — see issue #132.

```bash
openral dashboard            # binds 127.0.0.1:4318 by default
# → stderr prints: OpenRAL dashboard: http://localhost:4318/ …
```

Then point any OpenRAL workload at it. **Sim does not emit OTel by
default** — you opt in via the env vars below; an unconfigured
`openral sim run` is silent on the dashboard side:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  openral sim run --config scenes/benchmark/libero_spatial.yaml --rskill smolvla-libero
```

Open `http://localhost:4318` in a browser. The connection indicator
in the top right turns green within a few hundred ms.

## One-keystroke demo — two options

### Dashboard → sim (`--inprocess`)
The dashboard spawns the workload as a child with the OTLP env vars
already configured. Pass the whole command as **one shell-quoted
string** (shlex-tokenised):

```bash
openral dashboard \
  --inprocess "openral sim run --config scenes/benchmark/pusht.yaml --rskill diffusion-pusht"
```

### Workload → dashboard (`--dashboard` on sim / deploy / benchmark)
Inverse path: the workload spawns the dashboard as a child on `4318`
(override with `--dashboard-port`) and shuts it down on exit. Best
when you already have the workload command memorised. All three
entry points carry the same flag:

```bash
openral sim run --dashboard \
  --config scenes/benchmark/pusht.yaml \
  --rskill diffusion-pusht

openral deploy run --dashboard --config deployments/so100.yaml

openral benchmark run --dashboard \
  --suite libero_spatial \
  --rskill smolvla-libero
```

The workload prints `OpenRAL dashboard attached: http://localhost:4318/`
once the child reports healthy, then routes traces+metrics to it for
the rest of the run. The child is SIGINT'd at exit after OTel finishes
draining (no `Connection refused` retries on the way down).

## What you see

- **Top bar** — service name, run mode (`sim` / `hardware` /
  `benchmark` once PR #108 lands), short run id, connection status.
- **Reasoner · mission** — the reasoner's active task queue (ADR-0073),
  rendered from `reasoner.mission_json` as an ordered checklist: each
  subtask shows its status (done ✓ / active ▶ / verifying ? / abandoned ✗
  / pending), a reward success bar (the gate that advances the queue),
  and the attempts/cap ladder. Subdivided children from a replan
  (`#123`: a blocked task `t2` spliced into `t2.1`, `t2.2`, …) are
  indented under their parent with a dim id badge — depth is read from the
  dot-path task id, so the flat queue still reads as a hierarchy. The
  latest tick (tool / model / error) is a footer line beneath it. Empty
  until the `reasoner_node` is ACTIVE.
- **Live signal · running skill** — one card for the active `rSkill`: skill id
  (headline), role, action-applied, tick idx, and a `step Xms · forward Yms`
  latency line. `step` is the full `rskill.execute` call; `forward` is the model
  forward pass inside it (`rskill.chunk_inference`) — the gap between them is the
  pre/post-processing cost. The former separate `Inference` card was folded in
  here (its engine is already in Identity, its only unique datum was the
  forward-pass latency); the redundant one-line `Safety` live-signal card was
  removed (Safety has its own zone above). Card border turns red on
  `Status.ERROR`.
- **Safety** — running counters of `safety_violation`, `estop_requested`,
  `deadline_missed`, `sensor_stale` span events, alongside the **Safety
  check ledger** (per-check pass/fail from `safety.check` spans).
- **Embodiment** — robot joints, **World state** (now also carrying the
  durable scene-object spatial-memory table, merged from the former
  standalone `World · scene objects` card), and system health. The
  separate `Commands · next action` card was removed; the commanded joint
  values still render as the second trace on the joints card.
- **Metrics** — every histogram, counter, and gauge that comes over
  the wire, with p50/p95 (for histograms) and a sparkline of the
  last ~600 samples. Empty until a workload starts emitting; PR #108
  adds `openral.tick.duration`, `openral.inference.duration`,
  `openral.hal.*.duration`, etc. Hover any sparkline for the exact
  value + clock time of the nearest sample (a white-ringed marker
  snaps to the point); each graph carries its own min/max Y labels and
  shares one bottom time axis with dotted gridlines so every row reads
  on the same clock. Metrics whose producer emits a contractual
  threshold (`openral.metric.threshold_ms` — the runner latency budget
  on `tick.duration`, the world-state staleness deadline on
  `world_state.staleness_ms`) draw a dashed budget/deadline line and
  redden the trace once the latest sample breaches it (`*_dir` set to
  `lower` flips the test for floor-style metrics where dropping below is
  the fault). Warn/error events appear as severity-coloured vertical
  markers aligned across all graphs. **Click any point** to focus the
  whole column on that instant — a line is drawn across every graph and
  the event log scopes to a window around it (clear with the ✕).
  **Scroll** over a graph to zoom the shared time window in/out (centred
  on the cursor, bounded by the retained buffer; "reset zoom" restores
  the full view), and Y rescales to whatever is visible. The **freeze**
  toggle pauses live updates so you can inspect a moment.
- **Event log** — chronological feed of the last 60 events (spans,
  span events, and real log lines bridged from structlog over OTLP);
  ESTOP / safety violation rows render in red. Filter chips toggle the
  `info` / `warn` / `error` buckets (on by default) and a `debug` bucket
  (off by default — high-rate DEBUG such as `world_state` ~30 Hz would
  otherwise flood the bounded view; toggle it on when you need it).
  Log lines are bucketed by OTLP `severity_number`
  (DEBUG→`debug`, INFO→`info`, WARN→`warn`, ERROR/FATAL→`error`).

The main page stays focused on telemetry. Discovery (`GET /api/robots`) and the
guarded write endpoints (`POST /api/skill/execute`, `POST /api/param/set`) stay
available for operator tooling, but they are not surfaced as dashboard cards.

## Endpoints

| Path           | What it serves                                       |
|----------------|------------------------------------------------------|
| `GET /`        | Single-page UI (vanilla JS + SSE, no npm)            |
| `GET /healthz` | `{"status": "ok"}` for compose healthchecks          |
| `GET /api/state`  | One-shot JSON snapshot of the current store       |
| `GET /api/stream` | Server-Sent Events — every state update           |
| `POST /v1/traces`  | OTLP/HTTP receiver — `application/x-protobuf`    |
| `POST /v1/metrics` | OTLP/HTTP receiver — `application/x-protobuf`    |
| `POST /v1/logs`    | OTLP/HTTP receiver — log lines into the event log |

## Running alongside Jaeger + Prometheus

The dashboard is fine on its own. For post-hoc trace analysis or
Prometheus-style metric history, also bring up the dev compose stack:

```bash
docker compose -f docker-compose.dev.yml up -d otelcol jaeger prometheus
```

Point your workload at `http://localhost:4317` (the OTel Collector)
to fan out: traces → Jaeger UI at `http://localhost:16686`, metrics →
Prometheus at `http://localhost:9090`. The dashboard can still be
attached on a different port for the live pane:

```bash
openral dashboard --port 4318
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  openral sim run ...   # → otelcol → jaeger + prometheus
```

(Pointing at both the dashboard *and* otelcol simultaneously is a
v2 feature — the OTel SDK supports it via multiple exporters; v1
of the dashboard expects one endpoint at a time.)

## Why this exists

OpenRAL emits all the right OTel spans and metrics by design (see
ADR-0010 for the runner-side contract, and PR #108 for the metric
surface), and Jaeger renders them beautifully — **after** the run.
For the Day 30 demo and for on-robot debugging, the operator wants
a *live* pane that updates as the robot moves. `openral dashboard` is
that pane; it does not replace Jaeger for post-hoc analysis.
