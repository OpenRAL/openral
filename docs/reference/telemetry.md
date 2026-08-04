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
| `deploy.bringup` | all lifecycle nodes | `observability/lifecycle.py` (via `@log_lifecycle_errors`) | 1 / transition | **info** | Event Log |
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

### The three event rings

Demoting the noisy spans stopped info being *generated*, but the main ring is
one FIFO shared with the debug stream, so the rows that remained were still
evicted within seconds. The store therefore keeps two protected lanes beside it,
and `snapshot()` merges all three deduped:

| ring | size | holds |
|---|---|---|
| `_events` | 200 | everything, including debug |
| `_error_events` | 64 | errors + fatals, plus `skill_failure` even when downgraded to warn |
| `_headline_events` | 64 | the remaining non-debug rows (`deploy.bringup`, `reasoner.tick`, `world.scene_objects` …) |

**The two lanes are separate on purpose.** Mirroring all non-debug traffic into
the single error lane let routine info evict the safety events those 64 slots
exist to preserve: `world.scene_objects` alone runs at ~0.10/s, so the shared
lane fully cycled in ~11 minutes of an otherwise idle scene and a minute-one
`safety.violation` was gone by minute twelve — the exact "counter goes up, no
trace" failure the protected lane was added to prevent. One budget each means
neither class can starve the other.

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
| `openral.event.estop_requested` | `hal/lifecycle.py::_emit_estop_telemetry` | E-stop latched at the HAL | error, counted, protected lane |
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
| `openral.tick.duration` | Histogram | ms | `runner/base.py` | 30 Hz — **library-runner path only**, see below |
| `openral.inference.duration` | Histogram | ms | `observability/tracing.py::inference_span` | per inference, **every adapter**; label `kind` ∈ `foreground` \| `prefetch` \| `single` (`InferenceKind`, mypy-enforced) — a timing axis; chunk *shape* rides the span's `inference.chunk_size`, not the label |
| `openral.hal.read_state.duration` | Histogram | ms | `hal/lifecycle.py::_hal_duration_metric` + `deploy_runner.py` | 30 Hz, **both paths** |
| `openral.hal.send_action.duration` | Histogram | ms | `hal/lifecycle.py::_hal_duration_metric` + `deploy_runner.py` | per chunk, **both paths** |
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
| `openral.hal.estop.count` | Counter | — | `hal/lifecycle.py::_emit_estop_telemetry` | per e-stop (label `hal.adapter`) |
| `openral.observability.export_failures` | Counter | — | `_sdk._FailureCountingSpanExporter` | per failed span-export batch (label `signal_kind=trace`) |

> **Latency metrics are emitted by the span helpers, not their callers.**
> `inference.duration` comes from `inference_span`; the two `hal.*.duration`
> histograms from `_hal_duration_metric`, paired with the spans in the shared
> HAL lifecycle base. They used to be recorded only inside
> `openral_runner.InferenceRunnerBase` / `DeployRunner`, which the ROS deploy
> graph never instantiates — so a live `deploy run` produced the spans and no
> histograms at all. Emitting both from one seam makes them impossible to
> diverge. `openral.tick.duration` is deliberately left alone: its unit is the
> library runner's whole tick (sensors + HAL + skill + safety) and no such unit
> exists on the ROS graph, where those are four separate processes.
>
> `InferenceRunnerBase` used to record `inference.duration` **as well**, from
> `result.inference_ms`. That was removed: `inference_ms` is `Skill.step`
> wall-time — the chunk *dispatch* cost, near-zero on the ticks a
> `ChunkedExecutor` spends replaying a cached action — not the inference. Same
> instrument, two meanings, two disjoint label sets (`{rskill.id}` vs `{kind}`),
> so the eval/sim path doubled its sample count and computed p95 over a mixed
> population. The dispatch cost stays on the tick span as `rskill.inference_ms`
> and in the run summary's avg/p99, where it is unambiguous.
>
> **A single seam is only safe while it is universal.** Removing the runner's
> record exposed five sidecar adapters that never opened the span at all —
> `behavior_groot`, `internvla_n1`, `lingbot_va_a1`, `lingbot_vla2`,
> `rlbench_3dda`, all of whose inference is one `SidecarClient.call()` — so
> they had been reporting latency only through the runner's proxy. They are now
> instrumented (`engine="sidecar"`), and
> `tests/unit/test_every_adapter_is_instrumented.py` is a source-level canary
> that fails if a future adapter defines `step()` without reaching
> `inference_span` or `run_inference`.

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

These names exist in `semconv` / `metrics` but **nothing produces them** (six, after `estop_requested` / `hal.estop.count` were wired and the
`inference.timeouts` contract was deleted with `ROSInferenceTimeout`, issue #49). They
are kept because each records an intended contract, and annotated in-place so
the modules do not read as an inventory of what actually works. A source-scanning
test (`python/observability/tests/test_declared_not_emitted.py`) pins this list
both ways, so wiring one up — or adding another — fails until this table is
updated.

| Signal | Why it is not just a missing `add(1)` |
|---|---|
| `openral.event.action_dropped` | `DeadlineOverrunPolicy.DROP` does drop the action, but reports via `deadline_missed` instead. |
| `openral.event.chunk_prefetch_hit` / `_miss` | Action-chunk prefetch runs but never reports its hit rate, so "is the prefetch helping?" cannot be answered from a trace. |
| `openral.event.episode_closed` | Intended to let a Jaeger query pivot from a skill execution to the produced dataset row; `RolloutRecorder` closes episodes without it, so that pivot does not work. |

**Removed: `openral.safety.clamps` / `safety.clamped`.** OpenRAL never clamps.
The attribute was written in four places and was a literal `False` in every one;
no code path ever set it `True`. That is not an oversight — the safety layer is
deny-by-default and `compute_intersection` "rejects (never clamps)" any envelope
field that would loosen the robot ceiling. The counter therefore measured an
operation the system does not perform, and the attribute cost a constant on
every `safety.check` span at 30 Hz. (HAL adapters *do* saturate commands into an
actuator's physical range — the Franka gripper maps `[0, 1]` onto its travel —
but that is device range-mapping, not a safety correction.)

**Fixed: the e-stop latch now self-clears.** `store.py` cleared
`topics.safety.estopped` only on `safety.severity == "ok"`, which no emitter has
ever produced — the C++ kernel sends `info` on a pass, `warn` while latched and
`violation` on a drop. The latch could only be set, never cleared, so the code
comment claiming it "self-corrects after a reset" was false and `estopped` stuck
true until an explicit `POST /api/estop_reset`. It now clears on a clean pass
from a real kernel, which is proof the kernel is not latched (it returns early
with `estop_latched` while `fault_latch_` is set). The null client is excluded:
it emits `info` unconditionally without checking anything, so treating that as
evidence of a clear would let a no-op client unlatch the UI. Dashboard-side
only — no safety code changed.

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
