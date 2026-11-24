# Telemetry reference — spans, metrics, logs, instrumentation

Everything OpenRAL reports at runtime, where it comes from, how often it fires,
and where it surfaces. If a signal is not here, it does not exist; if it is marked
**not emitted**, the name exists but nothing produces it.

Names are normative in
[`openral_observability.semconv`](https://github.com/OpenRAL/openral/blob/master/python/observability/src/openral_observability/semconv.py);
this page is its annotated index.

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

So spans, metrics and every log line land in the same store that feeds the
operator's Event Log, which is why the banding rules below matter: without them
the panel is unreadable.

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
| `cli.command` | CLI | `observability/cli.py` | 1 / invocation | **info** | Event Log |
| `deploy.bringup` | all lifecycle nodes | `observability/lifecycle.py` (via `@log_lifecycle_errors`) | 1 / transition | **info** | Event Log |
| `rskill.execute` | rSkill | `rskill/base.py` (`rSkillBase.step`), `rskill_runner_node.py` | 1 / tick (`step`) + 1 / goal (runner node) | debug — not promoted while the per-tick emitter shares the name | rSkill card |
| `rskill.configure` | rSkill | `rskill/base.py` | per lifecycle | **info** | Event Log, rSkill card |
| `rskill.activate` | rSkill | `rskill/base.py` | per lifecycle | **info** | Event Log, rSkill card |
| `reasoner.tick` | Reasoning | `reasoner/core.py` | per LLM round-trip (~0.2 Hz cap) | **info** | Reasoner card |
| `world.scene_objects` | World state | `world_state/scene_objects_span.py` | on scene change (+60 s keepalive; checked at 0.2 Hz) | **info** | World-state card, SLAM overlay |
| `sim.run` | Sim | `sim/sim_runner.py` | 1 / run (held open) | **info** | Event Log |
| `detect.probe.*` | Detect | `detect/detect.py` | 1 / `openral detect` | **info** | Event Log |
| `hal.read_state` | HAL | `hal/lifecycle.py`, `deploy_runner.py` | **30 Hz** | debug | Robot-state card |
| `hal.send_action` | HAL | `hal/lifecycle.py`, `deploy_runner.py` | **30 Hz** | debug | Robot-state card |
| `sensors.read_latest` | Sensors | `deploy_runner.py`, `world_state_ros/lifecycle_node.py`, `sim_sensor_bridge.py` | **30 Hz × cameras** | debug | Perception card (thumbnails) |
| `world_state.snapshot` | World state | `world_state/aggregator.py` | **30 Hz** | debug | World-state card |
| `rskill.tick` | Runner | `runner/base.py` | **30 Hz** | debug | rSkill card |
| `rskill.chunk_inference` | rSkill | `rskill/_vla_core.py`, `rskill_runner_node.py`, 8 sim adapters | per chunk | debug | rSkill card (latency) |
| `safety.check` | Safety | `runner/safety.py`, `supervisor_node.py`, `lifecycle_kernel.cpp` | per candidate chunk | debug | Safety ledger |
| `reward.score` | Perception | `reward_monitor_node.py` | `score_period_s` (2 s) | debug | rSkill card reward bar |
| `slam.occupancy_grid` | SLAM | `runner/slam_bridge.py` | 1 Hz (throttled) | debug | SLAM map card |
| `world.pointcloud` | World state | `runner/world_cloud_bridge.py` | throttled | debug | Pointcloud card |
| `sim.step` / `eval.step` | Sim | `sim/sim_runner.py` | per step | debug | — |
| `physics.step` | Sim | `sim/sim_runner.py` | per step | debug | — |
| `reasoner.skill_failure` | Reasoning | `reasoner_node.py` | per failure, only when no `reasoner.tick` span is recording | **error** | Transient fallback carrying `openral.event.skill_failure`; normally that event rides the active `reasoner.tick` span (see below) |
| `world_state.object_query` | World state | `world_state_ros/object_tracker_node.py` | per object-query escalation | debug | Parented on the goal's trace |

> The allow-list keeps routine 30 Hz spans out of the Event Log's 200-slot ring; only promoted spans show as info rows.

### `rskill.chunk_inference` attributes

Every span opened by `inference_span` (directly, or through the
`_vla_core.run_inference` seam) carries `inference.kind` and, when known,
`inference.chunk_index` / `inference.chunk_size` / `inference.engine` /
`inference.device`, plus `inference.duration_ms` recorded on close. Those six
have `semconv.py` constants; the RTC attribute below rides the helper's generic
`inference.`-prefixed `**attrs` path and has none.

| Attribute | Recorded at | Meaning |
|---|---|---|
| `inference.rtc_delay` | `rskill/_vla_core.py::run_inference` | Real-Time Chunking only: how many actions the consumer popped out of the `ActionQueue` while the *previous* inference ran — the prefix length the guidance freezes. It is the queue's **index delta**, not a latency estimate, so it stays correct in fast-forward sim as well as wall-clock deploy. `0` on the cold-start chunk (no previous tail). The executor also logs each merge as `chunked_executor.rtc_merge` at DEBUG (`chunk_index`, `merge_tick` = actions served before the merge, measured `rtc_delay`), so switch points can be lined up with a bag of served actions (enable DEBUG on just that stdlib logger, `logging.getLogger("openral_rskill.executor").setLevel(logging.DEBUG)`, e.g. from a `sitecustomize.py`; a deploy's root logger stays at WARNING). |

The attribute is written only when the executor threads an `inference_delay`
into `run_inference(call_kwargs=…)`, so its presence in a trace is also the
"RTC is on for this skill" marker. It stays on the span and never becomes a
metric label — the `openral.inference.duration` label set is closed to `kind`.

### The three event rings

The store keeps two protected lanes beside the main FIFO so routine rows cannot
evict the ones an operator needs; `snapshot()` merges all three, deduplicated.

| ring | size | holds |
|---|---|---|
| `_events` | 200 | everything, including debug |
| `_error_events` | 64 | errors + fatals, plus `skill_failure` even when downgraded to warn |
| `_headline_events` | 64 | the remaining non-debug rows (`deploy.bringup`, `reasoner.tick`, `world.scene_objects` ...) |

The lanes are separate so that routine info cannot cycle the slots reserved for
safety events.

---


## Span events

Attached to whichever span is active. Counted in the dashboard's Command band
where noted.

| Event | Emitted at | Meaning | Band |
|---|---|---|---|
| `openral.event.safety_violation` | `runner/base.py`, `deploy_runner.py`, `lifecycle_kernel.cpp` | Safety check rejected an action | error, counted |
| `openral.event.deadline_missed` | `runner/base.py` | Tick overran its cadence budget | error, counted |
| `openral.event.skill_failure` | `reasoner_node.py` | Reasoner-published skill failure | error (→ warn while e-stop-latched), counted |
| `openral.event.sensor_stale` | `deploy_runner.py` | Sensor older than its age budget | counted |
| `openral.event.staleness_latched` | `world_state/aggregator.py` | World-state component **went** stale — only for a component that has had data. A never-received one is stale in `diag` and counted in `components_stale`, but has not latched: it never had a fresh state to fall from. Suppressing it removes a WARN that fired on every bringup (`world_state` subscribes before the HAL's first `joint_state`). | — |
| `openral.event.error_latched` | `world_state/aggregator.py` | World-state component latched an error | — |
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
| `openral.sensors.age_ms` | Histogram | ms | `deploy_runner.py` | 30 Hz × camera |
| `openral.world_state.staleness_ms` | Histogram | ms | `world_state/aggregator.py` | 30 Hz × component |
| `openral.tick.budget_violations` | Counter | — | `runner/base.py` | on breach |
| `openral.tick.deadline_misses` | Counter | — | `runner/base.py` | on breach — **every** miss, unlike the rate-limited log |
| `openral.safety.violations` | Counter | — | `runner/base.py`, `deploy_runner.py` | on violation |
| `openral.sensors.stale_reads` | Counter | — | `deploy_runner.py` | on stale read |
| `openral.failure.count` | Counter | — | `observability/compliance.py` (compliance recorder) | per `FailureTrigger`; labels `component` (bus source), `kind`, `severity` |
| `openral.sim.episode.count` | Counter | — | `sim/sim_runner.py` | per episode |
| `openral.sim.episode.success` | Counter | — | `sim/sim_runner.py` | per successful episode |
| `openral.world_state.components_stale` | UpDownCounter | — | `world_state/aggregator.py` | 30 Hz |
| `openral.system.cpu.utilization_pct` | UpDownCounter | % | `system_metrics.py` | 1 Hz |
| `openral.system.ram.used_mb` / `.total_mb` | UpDownCounter | MBy | `system_metrics.py` | 1 Hz |
| `openral.system.gpu.memory_used_mb` / `.total_mb` | UpDownCounter | MBy | `system_metrics.py` | 1 Hz |
| `openral.system.gpu.utilization_pct` | UpDownCounter | % | `system_metrics.py` | 1 Hz |
| `openral.hal.estop.count` | Counter | — | `hal/lifecycle.py::_emit_estop_telemetry` | per e-stop (label `hal.adapter`) |
| `openral.observability.export_failures` | Counter | — | `_sdk._FailureCountingSpanExporter` | per failed span-export batch (label `signal_kind=trace`) |

> **Latency metrics come from the span helpers.** `inference.duration` is
> recorded by `inference_span` and the two `hal.*.duration` histograms by
> `_hal_duration_metric`, so both the library runner and the ROS deploy graph
> report them. `openral.tick.duration` is the library runner's whole tick and
> has no ROS-graph equivalent. `tests/unit/test_every_adapter_is_instrumented.py`
> fails if an adapter defines `step()` without reaching `inference_span`,
> `run_inference` or `build_chunk_executor`.


## Compliance records

Issue #467. Typed `openral_core.ComplianceRecord`s go out as OTLP **log records**,
one per record, through `openral_observability.compliance.ComplianceEmitter`: the
body is the record's JSON, the timestamp its `ts_ns`, the trace context its
`trace_id` / `span_id`. A collector or hosted plane filters on the attributes.

| Attribute | Value |
|---|---|
| `openral.record.kind` | `failure`, `safety_stop`, `safety_decision`, `plan_decision`, `intervention`, `software_inventory`, `session`, `security_incident` |
| `openral.record.id` | the record's 32-hex id |
| `openral.schema.version` | the record kind's `schema_version` |
| `openral.retention.class` | `ai_log_6m`, `mr_safety_decision_1y`, `mr_trace_5y`, `incident_10y`, `telemetry_raw` (floors in `RETENTION_FLOOR_DAYS`) |

| Producer | Records |
|---|---|
| `openral_compliance` recorder node (`/openral_compliance_recorder`) | `failure` per `FailureTrigger` on `/openral/failure/*`; `safety_stop` per `/openral/safety_status` transition into `latched`, with `parent_record_id` → the safety failure whose kind matches the drop reason, arriving up to 2 s before or after the latch (an e-stop forwarder publishes `/openral/estop` before its trigger); a latch replayed at configure is not a new stop; nothing is recorded while the node is not ACTIVE; `session` on activate / deactivate; `software_inventory` on activate; `intervention` (`estop_reset`) per latch clearing, attributed by `/openral/operator_action` or recorded `unknown`; `intervention` (`param_set`) per announced dashboard parameter change. Publishes `/openral/estop` when a record cannot be written to the on-robot log |
| `DeployRunner` safety gate (`ROSSafetyViolation` at the supervisor boundary) | `safety_stop` with `trigger_source="supervisor"`, once per unbroken run of the same violation class (not per withheld tick) |

The machine is named by the **resource**, which the deploy launch stamps into
`OTEL_RESOURCE_ATTRIBUTES` for every node (the C++ kernel's `Resource::Create`
merges it too): `openral.unit.serial`, `openral.org.id`, `openral.site.id`,
from `openral_core.resolve_unit_identity`. A hardware deploy without a serial is
refused.

Every record is also written to the on-robot, hash-chained compliance log (always on for
`deploy run`; see [The on-robot compliance log](compliance-log.md)) before it goes out over
OTLP, and the recorder records a `software_inventory` at every start.

Each emitter (one per process) has its own `stream_id`; `seq` is gap-free within it, so a
gap is a lost record and a new `stream_id` a restarted process.

Camera thumbnails (`openral.sensors.thumbnail_jpeg_b64`) are only encoded and attached
when the OTLP endpoint is on the deployer's own network (loopback, private or link-local
address, or a dotless host such as `otelcol`) or the deployer set
`OPENRAL_TELEMETRY_IMAGE_CONSENT=1` (`compliance.images_may_leave_host`); withholding
logs `telemetry.images_withheld` once per endpoint.
The reasoner tick's free-text `reasoner.rationale` follows the same rule with its own
consent, `OPENRAL_TELEMETRY_TEXT_CONSENT=1` (`compliance.free_text_may_leave_host`; a
rationale paraphrases the operator's prompt and the scene); withholding logs
`telemetry.free_text_withheld` once per endpoint.

## Logs

All OpenRAL logging is `structlog`, bridged to OTLP. **Records below
`OPENRAL_LOG_LEVEL` (default `INFO`) are dropped before rendering or export.**

### Hot-path sites worth knowing

| Site | Level | Fires at | Notes |
|---|---|---|---|
| `runner/base.py` `inference_runner.deadline_missed[.drop]` | warning | up to 30 Hz | **Rate-limited** to one line per 5 s carrying `suppressed_since_last` + `worst_tick_ms`. WARNING is the one band an operator cannot filter away, and a too-slow host misses on every tick. |
| `world_state/aggregator.py` `world_state.*.updated` (8 sites) | debug | 30 Hz | Below the default floor. |
| `rskill/base.py` `skill.step` | debug | 30 Hz | Below the default floor. |
| `runner/safety.py` `safety.null_check` | debug | 30 Hz | Below the default floor. |
| `reward_monitor_node.py` reward score | info | 0.5 Hz | Also on the rSkill card's reward bar. |
| `reasoner/core.py` `reasoner.tick.selected` | info | per LLM tick | Rare and high-value — the tool the model chose. |
| `reasoner_node.py` tick-suppressed reasons | debug | sub-second | Correctly demoted. |
| `ros_image_detector_node.py` "continuous leg alive" | info | per frame | Already `throttle_duration_sec=5.0`. |
| `dashboard/app.py` write-control audit trail | warning | per operator write | Audit-by-design; keep at WARNING. |
| `rskill/_diagnostics.py` `phase_timer` | info | start/done + 15 s heartbeat | The model-load progress signal — see below. |
| `hal/sim_attached.py` `sim.task_success` / `sim.task_success_final` | info | per verdict change / 1 per session | Deploy-sim ground truth — see below. |
| `hal/sim_attached.py` `sim.placement_grade` / `sim.placement_grade_final` / `sim.placement_grade_unavailable` | info (unavailable: warning when `set_task` raised) | per grade change / 1 per graded goal / 1 per ungraded goal | Deploy-sim placement grade of the dispatched goal — see below. |
| `hal/sim_attached.py` `sim.time_base` (logger `openral.sim.time_base`) | info | 1 per connect | Deploy-sim time base: `sim_dt_per_tick_s`, `control_freq_hz`, `backend`, `robot` (the bare MuJoCo twin adds `physics_dt_s` / `physics_steps`), so a replayed trace knows which clock its velocities used. Companions: `hal.time_base.pinned` / `hal.time_base.unresolved` (warning) mark a twin that is not timing-faithful; `hal.time_base.timestep_shrunk` / `robosuite.time_base.timestep_shrunk` (info) record a shrunk physics step. |
| `hal/sim_attached.py` `hal.send_action.dropped_estop_latched` | warning | 1 per latch episode (`estop` → `reset_estop`) | A sim HAL's estop latch is dropping commands (`backend`, `robot`, `control_mode`, `tick_index` of the first): a latched send is never silent. |
| `hal/mujoco_twin.py` `hal.renderer_close_failed` | warning | per renderer whose GL teardown raised | The twin's `close` still releases the model; the renderer is dropped, not kept. |

### The deploy-sim task-success signal

`openral deploy sim` runs its backend continuously, which suppresses the
simulator's per-step task evaluation. `SimAttachedHAL` therefore polls the
backend's own predicate (RoboCasa's `env._check_success()`, via the optional
`task_success` extension of `openral_sim.rollout.SimRollout`) once per
`env.step`. Logger `openral.sim.task_success`; each line is also mirrored to
stdout as `<event> <json>`, because a launched HAL subprocess exports but never
prints structlog records.

| Line | Fires | Carries |
|---|---|---|
| `sim.task_success` | every change of the verdict, both edges | `success`, `previous`, `first_success`, `scene_id`, `task_id`, `sim_time_ns`, `sim_time_s`, `step` |
| `sim.task_success_final` | once, at `disconnect` (lifecycle shutdown or `HALLifecycleNodeBase.shutdown_hal` on signal teardown) | `success`, `ever_succeeded`, `first_success_sim_time_ns`, `transitions`, `scene_id`, `task_id`, `sim_time_ns`, `steps` |
| `sim.task_success_probe_failed` | once, if the predicate raises | `error`, `error_type`; polling then latches off |

Both edges are logged because RoboCasa success is not latched. A backend with
no predicate emits nothing: "unknown" is never reported as failure. An Isaac
scene with `placement_success` is also graded per dispatched goal
(see [sim environments](sim-environments.md)):

| Line | Fires | Carries |
|---|---|---|
| `sim.placement_grade` | every change of the graded prompt's grade | `prompt`, `grade`, `score`, `previous`, `goal_active`, `scene_id`, `task_id`, `sim_time_ns`, `sim_time_s`, `step` |
| `sim.placement_grade_final` | once per graded goal, or at `disconnect` | `prompt`, `grade`, `score`, `best_grade`, `best_score`, `scene_id`, `task_id`, `sim_time_ns`, `steps` |
| `sim.placement_grade_unavailable` | once per goal the scene does not grade; at WARNING if `set_task` raised | `prompt` |

These verdicts reach the log and nothing else: they never feed termination,
reset, reward or the action path (CLAUDE.md §1.4). A SIGKILLed process emits no
final line.

### The E-stop ground-truth signals

`SimSensorBridge` records every kernel stop, loaded or not. Three `error`-level
lines, joined by `stop_seq`:

| Line | Fires | Carries |
|---|---|---|
| `sim.estop_ground_truth_snapshot` | every `/openral/estop` while MuJoCo handles are live (Isaac: when the env serves `contact_ground_truth`) | contacts, `nearest_*_pairs` probes with coverage, joint state, base TF, the 3-deep `candidate_action_chunks` ring and the kernel's `CollisionEvidence` if it arrived within 0.5 s. Isaac instead records PhysX `robot_world_contacts` (`backend: "isaacsim"`, no near-miss probes) |
| `sim.estop_ground_truth_evidence` | only when the evidence lost the publish race | `stop_seq`, `collision_evidence` |
| `sim.estop_initial_configuration` | only when the stop fired before any action reached the HAL | `violation: "initial_configuration"`, `stop_seq`, `candidate_chunks_seen`, `sim_time_s`, `nearest_robot_world_pair`, `detail` |

Each pair in `nearest_*_pairs` carries its own proof: `distance_m`,
`distance_certified`, `distance_method`, witness points and, for bracketed
geometry, `distance_bracket_m`; coverage blocks add `certified_pairs`,
`uncertified_pairs` and `distance_instrument`. A verdict may rest only on a
certified pair (`tools/validation_matrix.py::probe_is_distance_certified`;
background in [collision validation evidence](collision-validation-evidence.md)).
The kernel's `CollisionEvidence` also carries `joint_positions_rad`, the
configuration forward kinematics ran on for the reported `horizon_step`.

The third line separates "spawned somewhere unsafe" from an ordinary mid-task
stop: `SimAttachedHAL.last_action_ns` is `0` until the first `send_action`. The
remedy is a scene-config one (another seed, or pinned `layout_ids` /
`style_ids`). These lines are diagnostics only (CLAUDE.md §1.4); the stop is
never suppressed, delayed or altered.

### Graded velocity scaling

Off by default (`collision_scale_proximity_m: 0.0`). When armed, an accepted
chunk near an obstacle is republished with its rate columns scaled:

| Signal | Where | Carries |
|---|---|---|
| `safety.velocity_scale` | `safety.check` span attribute | scale in `(0, 1]` applied to this chunk |
| `safety.scale_slack_m` | `safety.check` span attribute | clearance above the gate margin that produced it |
| `safety.collision_scaled` | kernel log, transition-gated | scale, slack, band, control mode, `rskill_id`; emitted when the scale moves by >= 0.1 or the band clears |
| `scaled` | `/diagnostics` key-value, 1 Hz | count of chunks republished slower (a subset of `passed`) |

### The RoboCasa scene composition

`robocasa_scene_composition` (structlog, info, once per `reset`, from
`openral_sim.backends.robocasa._RoboCasaSim`) carries `scene_id`,
`effective_layout_id` / `effective_style_id` (what RoboCasa actually composed;
style is `"custom"` for a style dict) and `pinned_layout_id` /
`pinned_style_id` (what the scene YAML pinned, or `null`). RoboCasa redraws
layout and style from its RNG on every reset, so the seed alone does not name
the kitchen. When the YAML pinned exactly one layout or style and the effective
one differs, the sim raises `ROSConfigError` instead of logging.


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
| `openral.event.action_dropped` | `DeadlineOverrunPolicy.DROP` does drop the action, but reports via `deadline_missed` instead. |
| `openral.event.chunk_prefetch_hit` / `_miss` | Action-chunk prefetch runs but never reports its hit rate, so "is the prefetch helping?" cannot be answered from a trace. |
| `openral.event.episode_closed` | Intended to let a Jaeger query pivot from a skill execution to the produced dataset row; `RolloutRecorder` closes episodes without it, so that pivot does not work. |

**Removed: `safety.clamps` / `safety.clamped`.** No code path ever set it; the
envelope layer rejects, never clamps. Distance-graded velocity scaling (see
[Graded velocity scaling](#graded-velocity-scaling)) is the one mechanism that
modifies a commanded value, and it reports through `safety.velocity_scale` and
`safety.scale_slack_m` instead.


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
| `OPENRAL_ROS2_TRACING_FALLBACK_DIR` | unset | Fallback trace directory for the LTTng session (`tracing_lttng.py`) |
| `OPENRAL_JAEGER_UI_URL` | unset | Enables the trace deep-link in the dashboard header |
| `OPENRAL_DASHBOARD_WRITE_CONTROLS` | off (linked robot: from the account at the next deploy start, on unless turned off; an OFF on either side wins) | `1` offers and runs the installed rSkills as dashboard actions (`skill:<id>`) |
| `OPENRAL_DASHBOARD_REMOTE_CONTROL` | off (linked robot: from the account at the next deploy or dashboard start, on unless turned off; an OFF on either side wins) | `1` lets signed-in operators reset, prompt and run skills over the robot's tunnel (E-STOP always may) |
| `OPENRAL_DASHBOARD_ALLOWED_ORIGINS` | openral.com's origins | Origins the tunnel listener answers (CORS + `/ws` upgrade) |
| `OPENRAL_DASHBOARD_UI_DIR` | unset | Serve a local robot-UI build at `/` instead of the pinned release |
| `OPENRAL_ROBOT_ENROLMENT` | `~/.config/openral/robot.json` | The enrolment `openral robot link` writes; opens the remote listener |
| `OPENRAL_UNIT_SERIAL` | unit file's `serial` | `openral.unit.serial`; **required** for a hardware deploy |
| `OPENRAL_ORG_ID` | unit file's `org_id` | `openral.org.id` |
| `OPENRAL_SITE_ID` | unset | `openral.site.id` |
| `OPENRAL_COMPLIANCE_DIR` | `~/.local/state/openral/compliance` on `deploy run`; unset (off) for sim | The on-robot compliance log |
| `OPENRAL_CLOCK_SOURCE` | `monotonic_unsynced` | Compliance records' `clock_source` (`ptp` / `ntp` / `monotonic_unsynced`) |
| `OPENRAL_TELEMETRY_IMAGE_CONSENT` | off | `1` lets camera thumbnails go to a public OTLP endpoint |
| `OPENRAL_TELEMETRY_TEXT_CONSENT` | off | `1` lets the reasoner tick's free-text `reasoner.rationale` go to a public OTLP endpoint |
| `OPENRAL_ACTOR_KEY` | unset | Deployer-held HMAC key for operator identity in `intervention` records; see [compliance log](compliance-log.md) |
| `OPENRAL_DASHBOARD_LAN_INGEST` | off | `1` lets other LAN hosts export OTLP to the dashboard (otherwise loopback only) |
| `OPENRAL_DASHBOARD_REMOTE_PORT` | 4319 | Port of the remote (tunnel) listener |
| `OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S` | 12 | Seconds the dashboard waits for a skill goal to be accepted |
| `OPENRAL_DASHBOARD_FLIP_180` | off | Rotates the dashboard's display copy of camera frames 180 degrees |
| `OPENRAL_STT_MODEL` / `_DEVICE` / `_COMPUTE` | `base.en` / `cpu` / `int8` | Dashboard voice-prompt speech-to-text; see the [observability README](https://github.com/OpenRAL/openral/blob/master/python/observability/README.md) |

Sampling note: the 0.1 hardware ratio applies to the **CLI process only**. HAL
and other nodes call `configure_observability()` without a ratio, so their spans
are always-on even on real hardware.

---

## Where it goes

| Sink | Transport | Notes |
|---|---|---|
| In-tree dashboard | OTLP/HTTP → `127.0.0.1:4318` | Default for `deploy sim`; SSE to the browser; camera thumbnails re-served as JPEG snapshots; caches the `openral.robot.manifest` log record |
| otel-collector-contrib | OTLP → `:4317` | `docker-compose.dev.yml`; traces→Jaeger, metrics→Prometheus, logs→debug |
| Jaeger | via collector | UI on `:16686` |
| Prometheus | scrapes `otelcol:8889` | 5 s interval, 2 h retention |
| C++ safety kernel | its own OTLP/HTTP exporter | service `openral_safety_kernel`; no sampler configured (always-on) |
| rosbag2 / mcap | `openral replay` | joins bag ↔ `/api/traces` |

Service names on the wire: `openral` (CLI), `openral.runtime`,
`openral.hal.<robot>`, `openral.world_state`, `openral.reasoner`,
`openral.safety`, `openral.prompt_router`, `openral.reward_monitor`,
`openral.compliance`, `openral_safety_kernel`.

---

## See also

- [`docs/methods/06-reasoning-wam-safety-observability.md`](../methods/06-reasoning-wam-safety-observability.md) — per-symbol observability inventory
- [`docs/methods/05-inference-runner.md`](../methods/05-inference-runner.md) — runner tick + deadline contract
- [`python/observability/README.md`](https://github.com/OpenRAL/openral/blob/master/python/observability/README.md) — package overview
