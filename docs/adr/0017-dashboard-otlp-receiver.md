# ADR-0017: `openral dashboard` — embedded OTLP/HTTP receiver, not in-process exporter

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: CLAUDE.md §6.1 (Layer 7 — Observability), §1.11 (no mocks),
  §1.14 (docs travel with code); closes
  [issue #44](https://github.com/OpenRAL/openral/issues/44).

## Context

Issue #44 asks for a live "WebSim-style" debugging UI for OpenRAL,
served at `localhost:8000` and rendering the latest `rskill.execute`,
`skill.chunk_inference`, and `safety.check` spans plus rolling
metrics. The acceptance criteria require that the dashboard works
**without** Jaeger/Tempo running, and that it streams **live** (not
post-hoc).

OpenRAL's telemetry already exits the process via OTLP/gRPC (per the
existing `configure_observability` in `openral_observability._sdk`,
and per PR #108 which adds metrics + W3C propagation on the same wire).
Two architectures could surface that telemetry on a local web page:

A) **In-process exporter.** Inside the same Python process that runs
   `openral sim run` (or the production workload), the dashboard registers
   an additional `SpanProcessor` / `MetricReader` that pushes into an
   in-memory ring; a FastAPI route serves an HTML view of that ring.
   Pros: no network, simplest packaging. Cons: requires the dashboard
   to share a process with the workload, which is fragile when the
   workload crashes, and is impossible when the workload runs in
   another container, on another host, or as a ROS 2 node tree.

B) **Embedded OTLP/HTTP receiver.** The dashboard ships its own OTLP
   receiver on the same port that serves the HTML and SSE. Any
   OpenRAL workload — local subprocess, container, remote robot —
   that points `OTEL_EXPORTER_OTLP_ENDPOINT` at the dashboard's port
   (with `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`) lights up the
   pane live. Pros: decoupled lifecycle, single port, identical
   semantics to OM1's WebSim and to any production collector.
   Cons: one additional dependency (`opentelemetry-proto` for
   decoding); two HTTP hops in a single-host demo (negligible at
   demo cadence).

## Decision

Adopt **option B**. The dashboard is an embedded OTLP/HTTP receiver
plus an HTML/SSE renderer, served from a single FastAPI app on a
single port.

The `--inprocess` flag covers the one-keystroke demo: `openral dashboard
--port 8000 --inprocess ral --inprocess sim --inprocess run …` spawns
the workload as a child with the OTLP env vars already pointed at
the dashboard. The workload still exports via the standard OTLP/HTTP
wire — same code path, no second adapter to maintain.

We deliberately use OTLP/**HTTP**, not OTLP/gRPC, for the embedded
receiver because:

- The OTel specification requires every SDK to support HTTP/protobuf
  as the second protocol, so any workload (Python, C++, Rust) can
  point at this dashboard without a gRPC server inside FastAPI.
- It lets the dashboard share one port between UI, SSE, JSON state,
  and the receiver — no separate `--otlp-port`.
- It avoids a gRPC dependency inside the dashboard process; the
  receiver is a handful of FastAPI POST handlers that decode
  `opentelemetry.proto.collector.{trace,metrics,logs}.v1.*` protobuf
  bodies (already in `uv.lock`).

## Consequences

- New deps under `openral-observability[dashboard]`: `fastapi`,
  `uvicorn[standard]`, `httpx` (test transport).
  `opentelemetry-proto` becomes an explicit (not transitive) dep.
- The dashboard does **not** subscribe to the in-process OTel
  providers. A workload that wants both Jaeger *and* the dashboard
  configures two exporters via the standard
  `OTEL_EXPORTER_OTLP_ENDPOINT,otel-jaeger:4317` multi-endpoint
  pattern — out of scope for v1; the demo path just points at the
  dashboard alone.
- No layer boundary is crossed. The dashboard is additive in Layer 7
  (Observability); no new responsibility is moved.
- Companion to this ADR: the dev compose stack now puts
  `otel/opentelemetry-collector-contrib` in front of Jaeger so the
  metric path is testable end-to-end against a real wire (Jaeger
  alone returns `UNIMPLEMENTED` on `/v1/metrics`). See
  `docker/otelcol/config.yaml` and `docker-compose.dev.yml`.

## Verification

- Unit: `python/observability/tests/test_dashboard_store.py`,
  `test_dashboard_app.py` — real OTLP protobuf payloads, real
  FastAPI app via `httpx.ASGITransport`.
- Integration: `tests/integration/test_dashboard_end_to_end.py` —
  real uvicorn socket + real OTel SDK + real OTLP/HTTP exporter
  exporting into the dashboard. No mocks per CLAUDE.md §1.11.
- Live demo: `openral dashboard --inprocess "openral benchmark scene --config
  scenes/benchmark/pusht.yaml --rskill diffusion-pusht"`.

## Amendments

### 2026-05-19 — Default port `8000` → `4318` (issue #132)

The original Decision example bound `localhost:8000`. In practice
`8000` collided constantly with `mkdocs serve` (`just docs`),
`python -m http.server`, and the FastAPI dev-server default — a
contributor running the docs preview alongside the dashboard would
hit `[Errno 98] Address already in use` and the dashboard would not
start. The default was changed to **4318** (the OTLP/HTTP standard
receiver port, per the OTel specification), which doubles as a hint
that the port speaks OTLP/HTTP. The `--port` flag is unchanged;
explicit overrides continue to work. The Decision itself (embedded
OTLP/HTTP receiver on a single port) is unaffected. `run_dashboard`
also now emits a single stderr banner with the resolved URL
(`OpenRAL dashboard: http://localhost:4318/`) so the user does not
have to grep uvicorn's startup log to find the link.

### 2026-05-19 — `--inprocess` accepts one shell-quoted string

Originally `--inprocess` was repeated once per argv token (e.g.
`--inprocess ral --inprocess sim --inprocess run --inprocess
--config --inprocess foo.yaml`) because Typer parses each occurrence
as one list element. In practice that produced very long, mistake-
prone invocations. The flag now takes a single string parsed with
`shlex.split`, so `openral dashboard --inprocess "openral sim run --config
foo.yaml"` is the new shape. The child-spawn behaviour is unchanged.

### 2026-05-19 — `--dashboard` flag on sim / deploy / benchmark

Added the symmetric path to `--inprocess`: `openral sim run --dashboard`,
`openral deploy run --dashboard`, and `openral benchmark run --dashboard`
spawn `openral dashboard` as a child on `--dashboard-port` (default
4318), wait for `/healthz`, set
`OTEL_EXPORTER_OTLP_{ENDPOINT,PROTOCOL}`, re-run
`configure_observability`, and shut the child down on exit (after
`shutdown_observability` drains so the last batch lands instead of
churning on `Connection refused`). Helpers live in
`openral_observability.dashboard`:

- `spawn_dashboard(*, host, port, ready_timeout_s)` — low-level
  `@contextmanager` yielding the URL (or `None` on failure).
- `attached_dashboard(*, enabled, port)` — high-level `@contextmanager`
  that the CLI call sites use; a no-op when `enabled=False`, otherwise
  delegates to `spawn_dashboard` plus the configure-/-drain pair.

Failure modes (no `ral` on PATH, port busy, `/healthz` timeout) are
logged at WARNING and the workload continues without OTel attached —
convenience must not gate the run (CLAUDE.md §1.4).

### 2026-06-08 — Three-tier scene paths (ADR-0041)

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers and stripped
rSkill names from filenames. The Pusht live-demo example above moved
from `scenes/benchmarks/diffusion_pusht.yaml` to
`scenes/benchmark/pusht.yaml` (singular `benchmark/`, rSkill name
dropped); the rSkill is now passed at the CLI via `--rskill
diffusion-pusht`. Dashboard / OTLP receiver behavior is unchanged —
only the on-disk scene path is renamed. See ADR-0041 and
[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the tier hierarchy.
