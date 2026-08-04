# Telemetry reference — spans, metrics, logs, instrumentation

Everything OpenRAL reports at runtime, where it comes from, how often it fires,
and where it surfaces. If a signal is not in this file, it does not exist; if it
is here and marked **not emitted**, the name exists but nothing produces it.

Names are normative and live in
[`openral_observability.semconv`](../../python/observability/src/openral_observability/semconv.py) —
that module is the single source of truth, this page is its annotated index.

**Contents:** [How it flows](#how-it-flows) · [Spans](#spans) ·
[Span events](#span-events) · [Metrics](#metrics) · [Logs](#logs) ·
[Load-phase instrumentation](#load-phase-instrumentation) ·
[Declared but not emitted](#declared-but-not-emitted) · [Configuration](#configuration) ·
[Where it goes](#where-it-goes)

---

## How it flows

Every OpenRAL process calls `configure_observability(service_name=…)`, which
installs a `TracerProvider`, `MeterProvider` and `LoggerProvider` all pointed at
one OTLP endpoint, plus a structlog→OTel log bridge. During `openral deploy sim`
that endpoint is the **in-tree dashboard**, which is itself an OTLP receiver:

```
 nodes ──OTLP/HTTP──▶ openral dashboard :4318 ──▶ TelemetryStore ──▶ SSE ──▶ browser
   │                    /v1/traces  /v1/metrics  /v1/logs
   └──────────────▶ otel-collector ──▶ Jaeger (traces) / Prometheus (metrics)
```

So spans, metrics **and every log line** land in the same store that feeds the
operator's Event Log. That is why the banding rules below matter as much as the
signals themselves: without them the panel is unreadable.

With `--no-dashboard` and no `OTEL_EXPORTER_OTLP_ENDPOINT`, the whole SDK is a
no-op — no providers, no exporters, and structlog falls back to its console
renderer.

---

## Spans

**Event Log band.** `dashboard/store.py::_is_headline_span` decides the band:
ERROR status → `error`; else a name in `_HEADLINE_SPANS` (or with a headline
prefix) → `info`; else → `debug`. This is an **allow-list**, so a new span is
quiet until deliberately promoted. Every span is still indexed in full for
`openral replay` regardless of band — this only affects the operator's log view.

| Span | Layer | Emitted at | Frequency | Band | Surfaces in |
|---|---|---|---|---|---|
| `cli.command` | CLI | `observability/cli.py:91` | 1 / invocation | **info** | Event Log |
| `rskill.execute` | rSkill | `rskill/base.py:266`, `rskill_runner_node.py:608` | 1 / goal | **info** | Event Log, rSkill card |
| `rskill.configure` | rSkill | `rskill/base.py:159` | per lifecycle | **info** | Event Log, rSkill card |
| `rskill.activate` | rSkill | `rskill/base.py:183` | per lifecycle | **info** | Event Log, rSkill card |
| `reasoner.tick` | Reasoning | `reasoner/core.py:392` | per LLM round-trip (~0.2 Hz cap) | **info** | Reasoner card |
| `world.scene_objects` | World state | `world_state/scene_objects_span.py:94` | ~0.2 Hz | **info** | World-state card, SLAM overlay |
| `sim.run` | Sim | `sim/sim_runner.py:436` | 1 / run (held open) | **info** | Event Log |
| `detect.probe.*` | Detect | `detect/detect.py:85-108` | 1 / `openral detect` | **info** | Event Log |
| `hal.read_state` | HAL | `hal/lifecycle.py:837`, `deploy_runner.py:415` | **30 Hz** | debug | Robot-state card |
| `hal.send_action` | HAL | `hal/lifecycle.py:938`, `deploy_runner.py:518` | **30 Hz** | debug | Robot-state card |
| `sensors.read_latest` | Sensors | `deploy_runner.py:340`, `world_state_ros/lifecycle_node.py:656`, `sim_sensor_bridge.py:477` | **30 Hz × cameras** | debug | Perception card (thumbnails) |
| `world_state.snapshot` | World state | `world_state/aggregator.py:434` | **30 Hz** | debug | World-state card |
| `rskill.tick` | Runner | `runner/base.py:244` | **30 Hz** | debug | rSkill card |
| `rskill.chunk_inference` | rSkill | `rskill/_vla_core.py:527`, `rskill_runner_node.py:898`, 5 sim adapters | per chunk | debug | rSkill card (latency) |
| `safety.check` | Safety | `runner/safety.py:121`, `supervisor_node.py:283`, `lifecycle_kernel.cpp:444` | per candidate chunk | debug | Safety ledger |
| `reward.score` | Perception | `reward_monitor_node.py:314` | `score_period_s` (2 s) | debug | rSkill card reward bar |
| `slam.occupancy_grid` | SLAM | `runner/slam_bridge.py:394` | 1 Hz (throttled) | debug | SLAM map card |
| `world.pointcloud` | World state | `runner/world_cloud_bridge.py:375` | throttled | debug | Pointcloud card |
| `sim.step` / `eval.step` | Sim | `sim/sim_runner.py:626` | per step | debug | — |
| `physics.step` | Sim | `sim/sim_runner.py:671` | per step | debug | — |
| `reasoner.skill_failure` | Reasoning | `reasoner_node.py:4890` | per failure | **error** | Event Log, failure counter |

> **Why the allow-list.** It replaced a deny-list that named only the three
> obvious 30 Hz spans and missed four more ticking at the same rate. Measured on
> one second of a real 30 Hz two-camera deploy, the Event Log took **121 info
> rows/s**, cycling its 200-slot ring every **1.65 s** — so every lifecycle line
> an operator needed scrolled away before it could be read. Inverted, routine
> traffic contributes **0** info rows.

---

## Span events

Attached to whichever span is active. Counted in the dashboard's Command band
where noted.

| Event | Emitted at | Meaning | Band |
|---|---|---|---|
| `openral.event.safety_violation` | `runner/base.py:302`, `deploy_runner.py:487`, `lifecycle_kernel.cpp` | Safety check rejected an action | error, counted |
| `openral.event.deadline_missed` | `runner/base.py:427` | Tick overran its cadence budget | error, counted |
| `openral.event.skill_failure` | `reasoner_node.py:4887` | Reasoner-published skill failure | error (→ warn while e-stop-latched), counted |
| `openral.event.sensor_stale` | `deploy_runner.py:387` | Sensor older than its age budget | counted |
| `openral.event.staleness_latched` | `world_state/aggregator.py:586` | World-state component went stale | — |
| `openral.event.error_latched` | `world_state/aggregator.py:589` | World-state component latched an error | — |
| `openral.event.estop_requested` | — | **[not emitted](#declared-but-not-emitted)** | counted (always 0) |
| `openral.event.action_dropped` | — | **[not emitted](#declared-but-not-emitted)** | — |
| `openral.event.chunk_prefetch_hit` | — | **[not emitted](#declared-but-not-emitted)** | — |
| `openral.event.chunk_prefetch_miss` | — | **[not emitted](#declared-but-not-emitted)** | — |
| `openral.event.episode_closed` | — | **[not emitted](#declared-but-not-emitted)** | — |

---

## Metrics

Exported every `OPENRAL_OTEL_METRIC_INTERVAL_MS` (default **5000**). Label
vocabulary is a closed set (`semconv.py`); threshold hints ride as data-point
attributes and are stripped by the store so they cannot fragment a series.

| Metric | Type | Unit | Recorded at | Frequency |
|---|---|---|---|---|
| `openral.tick.duration` | Histogram | ms | `runner/base.py:291` | 30 Hz |
| `openral.inference.duration` | Histogram | ms | `runner/base.py:293` | 30 Hz |
| `openral.hal.read_state.duration` | Histogram | ms | `deploy_runner.py:427` | 30 Hz |
| `openral.hal.send_action.duration` | Histogram | ms | `deploy_runner.py:535` | 30 Hz |
| `openral.sensors.age_ms` | Histogram | ms | `deploy_runner.py:377` | 30 Hz × camera |
| `openral.world_state.staleness_ms` | Histogram | ms | `world_state/aggregator.py:560` | 30 Hz × component |
| `openral.tick.budget_violations` | Counter | — | `runner/base.py:352` | on breach |
| `openral.tick.deadline_misses` | Counter | — | `runner/base.py:420` | on breach — **every** miss, unlike the rate-limited log |
| `openral.safety.violations` | Counter | — | `runner/base.py:302`, `deploy_runner.py:495` | on violation |
| `openral.sensors.stale_reads` | Counter | — | `deploy_runner.py:394` | on stale read |
| `openral.sim.episode.count` | Counter | — | `sim/sim_runner.py:840` | per episode |
| `openral.sim.episode.success` | Counter | — | `sim/sim_runner.py:842` | per successful episode |
| `openral.world_state.components_stale` | UpDownCounter | — | `world_state/aggregator.py:578` | 30 Hz |
| `openral.system.cpu.utilization_pct` | UpDownCounter | % | `system_metrics.py:188` | 1 Hz |
| `openral.system.ram.used_mb` / `.total_mb` | UpDownCounter | MBy | `system_metrics.py:189-190` | 1 Hz |
| `openral.system.gpu.memory_used_mb` / `.total_mb` | UpDownCounter | MBy | `system_metrics.py:191-192` | 1 Hz |
| `openral.system.gpu.utilization_pct` | UpDownCounter | % | `system_metrics.py:193` | 1 Hz |
| `openral.inference.timeouts` | Counter | — | — | **[not recorded](#declared-but-not-emitted)** |
| `openral.safety.clamps` | Counter | — | — | **[not recorded](#declared-but-not-emitted)** |
| `openral.hal.estop.count` | Counter | — | — | **[not recorded](#declared-but-not-emitted)** |
| `openral.observability.export_failures` | Counter | — | — | **[not recorded](#declared-but-not-emitted)** |

The system metrics come from a 1 Hz daemon thread
(`start_system_metrics_collector`) started automatically by
`configure_observability`; it is a quiet no-op on hosts without `psutil` /
`pynvml`.

---

## Logs

All OpenRAL logging is `structlog`, bridged to OTLP. **Records below
`OPENRAL_LOG_LEVEL` (default `INFO`) are dropped before rendering or export.**

Rough shape of the deploy-relevant call sites: ~199 `info`, ~197 `warning`,
~73 `debug`, ~41 `error`.

### Hot-path sites worth knowing

| Site | Level | Fires at | Notes |
|---|---|---|---|
| `runner/base.py:439,451` `inference_runner.deadline_missed[.drop]` | warning | up to 30 Hz | **Rate-limited** to one line per 5 s carrying `suppressed_since_last` + `worst_tick_ms`. WARNING is the one band an operator cannot filter away, and a too-slow host misses on every tick. |
| `world_state/aggregator.py:259-381` `world_state.*.updated` (7 sites) | debug | 30 Hz | Below the default floor. |
| `rskill/base.py:278` `skill.step` | debug | 30 Hz | Below the default floor. |
| `runner/safety.py:140` `safety.null_check` | debug | 30 Hz | Below the default floor. |
| `reward_monitor_node.py:424` reward score | info | 0.5 Hz | Also on the rSkill card's reward bar. |
| `reasoner/core.py:567` `reasoner.tick.selected` | info | per LLM tick | Rare and high-value — the tool the model chose. |
| `reasoner_node.py:2424,2454` tick-suppressed reasons | debug | sub-second | Correctly demoted. |
| `ros_image_detector_node.py:501-545` "continuous leg alive" | info | per frame | Already `throttle_duration_sec=5.0`. |
| `dashboard/app.py` write-control audit trail | warning | per operator write | Audit-by-design; keep at WARNING. |
| `rskill/_diagnostics.py:132,142,166` `phase_timer` | info | start/done + 15 s heartbeat | The model-load progress signal — see below. |

`ROSSafetyViolation` is never caught except at the safety-supervisor boundary,
where it triggers E-stop plus a structured incident log (CLAUDE.md §5).

---

## Load-phase instrumentation

`openral_rskill._diagnostics.phase_timer(name, *, prefix, gpu_mb=…)` wraps every
model-load phase and emits `<prefix>_<name>_{start,heartbeat,done}` with
`elapsed_s`, plus `rss_mb` and a major-fault delta, and optionally `gpu_mb`.
Heartbeat every 15 s so a long load is visibly progressing rather than hung.

> **It has a process-global side effect.** Entering the context manager raises
> `sys.setswitchinterval` to 50 ms and restores it on exit. Load phases were
> being starved by 30 fps camera threads; peers still get the GIL ~20×/s, well
> inside the safety kernel's 1 s staleness deadline. Do not wrap a *steady-state*
> phase with it.

Standard phase names: `imports`, `from_pretrained`, `to_device`,
`processor_dir`, `make_processors`, plus family-specific quantisation/compile
phases. Per-adapter shortcuts live in the sim policies (`_smolvla_phase`,
`_pi05_phase`, `_molmoact2_phase`, `_groot_phase`).

`runtime_node` additionally emits `prewarm_vla_framework_*` for the one-time
torch + lerobot import, which **must** run before any camera reader thread
exists — see [`docs/methods/11-ros2-nodes.md`](../methods/11-ros2-nodes.md).

Offline phase breakdown for a single skill:

```bash
uv run tools/profile_policy_load.py --rskill rskills/<dir>
```

### Other instrumentation

| Mechanism | Where | Default |
|---|---|---|
| `DiagnosticsHeartbeat` → `/diagnostics` | `observability/diagnostics.py` | 1 Hz per lifecycle node |
| FailureTrigger bus → `/openral/failure/*` | `observability/failure_bus.py` | token-bucket rate-limited |
| LTTng tracepoints (8 pairs) | `observability/tracing_lttng.py` | **off**; `OPENRAL_ROS2_TRACING=1`, <300 ns when off |
| rosbag2/mcap ↔ OTel join | `observability/replay/` | `openral replay` |
| W3C traceparent propagation | `observability/propagation.py` | cross-process, parent trace preserved |

---

## Declared but not emitted

These names exist in `semconv` / `metrics` but **nothing produces them**. They
are kept because each records an intended contract, and annotated in-place so
the modules do not read as an inventory of what actually works. A source-scanning
test (`python/observability/tests/test_declared_not_emitted.py`) pins this list
both ways, so wiring one up — or adding another — fails until this table is
updated.

| Signal | Why it is not just a missing `add(1)` |
|---|---|
| `openral.event.estop_requested`<br>`openral.hal.estop.count` | ⚠️ **The dashboard renders an `e-stops` counter from the event, so that widget reads 0 no matter how many e-stops fire.** A safety indicator that cannot leave zero reads as "no e-stops have occurred". Wiring it is a safety change: CLAUDE.md §3 requires a safety-WG reviewer, a hazard-log update, and tests proving the behaviour is at least as conservative. |
| `openral.safety.clamps` | The signal exists (`safety.clamped` span attribute in the C++ kernel; gripper-width clamping in `supervisor_node`) but nothing feeds the counter, so "how often is the kernel silently correcting the policy?" is unanswerable. Also a safety-boundary change. |
| `openral.inference.timeouts` | Nothing raises `ROSInferenceTimeout` — it appears only in docstrings as a contract no backend enforces. Wiring the counter means implementing the timeout first. |
| `openral.observability.export_failures` | Requires hooking the SDK exporter failure path. Until then a collector silently dropping batches looks identical to a healthy one — the exact failure this counter exists to expose. |
| `openral.event.action_dropped` | `DeadlineOverrunPolicy.DROP` does drop the action, but reports via `deadline_missed` instead. |
| `openral.event.chunk_prefetch_hit` / `_miss` | Action-chunk prefetch runs but never reports its hit rate, so "is the prefetch helping?" cannot be answered from a trace. |
| `openral.event.episode_closed` | Intended to let a Jaeger query pivot from a skill execution to the produced dataset row; `RolloutRecorder` closes episodes without it, so that pivot does not work. |

There is a second, related gap worth recording here: the dashboard's e-stop
latch (`store.py:919-920`) clears only on `safety.severity == "ok"`, and no
emitter ever produces `"ok"` — the three `safety.check` emitters all send
`"info"`. So `estopped` latches true until an explicit `POST /api/estop_reset`.
Same safety-WG scope as the counter above.

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *unset* → **full no-op** | Where traces, metrics and logs go |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | gRPC | `http/protobuf` for the in-tree dashboard |
| `OPENRAL_LOG_LEVEL` | `INFO` | Log-record floor. Names or integers; unparseable falls back to the default rather than raising. Does **not** affect span bands. |
| `OPENRAL_OTEL_SAMPLE_RATIO` | always-on | Head-based ratio; `deploy`/`connect` set 0.1 in the CLI process |
| `OPENRAL_OTEL_METRIC_INTERVAL_MS` | 5000 | Metric export interval |
| `OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS` | 30 | `BatchSpanProcessor` flush — ~1.3× the thumbnail rate so the dashboard does not alias |
| `OPENRAL_ROS2_TRACING` | off | LTTng tracepoints |
| `OPENRAL_JAEGER_UI_URL` | unset | Enables the trace deep-link in the dashboard header |
| `OPENRAL_DASHBOARD_WRITE_CONTROLS` | off | Enables `POST /api/skill/execute` and `/api/param/set` |

Sampling note: the 0.1 hardware ratio applies to the **CLI process only**. HAL
and other nodes call `configure_observability()` without a ratio, so their spans
are always-on even on real hardware.

---

## Where it goes

| Sink | Transport | Notes |
|---|---|---|
| In-tree dashboard | OTLP/HTTP → `127.0.0.1:4318` | Default for `deploy sim`; SSE to the browser; MJPEG re-serve of thumbnails |
| otel-collector-contrib | OTLP → `:4317` | `docker-compose.dev.yml`; traces→Jaeger, metrics→Prometheus, logs→debug |
| Jaeger | via collector | UI on `:16686` |
| Prometheus | scrapes `otelcol:8889` | 5 s interval, 2 h retention |
| C++ safety kernel | its own OTLP/HTTP exporter | service `openral_safety_kernel`; no sampler configured (always-on) |
| rosbag2 / mcap | `openral replay` | joins bag ↔ `/api/traces` |

Service names on the wire: `openral` (CLI), `openral.runtime`,
`openral.hal.<robot>`, `openral.world_state`, `openral.reasoner`,
`openral.safety`, `openral.prompt_router`, `openral.reward_monitor`,
`openral_safety_kernel`.

---

## See also

- [`docs/methods/06-reasoning-wam-safety-observability.md`](../methods/06-reasoning-wam-safety-observability.md) — per-symbol observability inventory
- [`docs/methods/05-inference-runner.md`](../methods/05-inference-runner.md) — runner tick + deadline contract
- [`python/observability/README.md`](../../python/observability/README.md) — package overview
