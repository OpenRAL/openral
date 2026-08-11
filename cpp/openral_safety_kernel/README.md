# openral_safety_kernel — C++ safety kernel

> Layer 6 (Safety). Separate process, real-time validator on the
> chunk-rate boundary. Replaces F5's Python pass-through
> (`packages/openral_safety/SafetyPassthroughNode`) behind the same
> topic contract. **Python proposes, C++ disposes.** (CLAUDE.md §1.5).

## Topic contract

| Direction | Topic / Service | Type | QoS |
| --- | --- | --- | --- |
| sub | `/openral/candidate_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=1 |
| sub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| pub | `/openral/safe_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=1 |
| pub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| pub | `/openral/failure/safety` | `openral_msgs/FailureTrigger` | RELIABLE, VOLATILE, KL=50 |
| pub | `/openral/safety_status` | `openral_msgs/SafetyStatus` | RELIABLE, **TRANSIENT_LOCAL**, KL=1 |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray`, 1 Hz | default |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

### `/openral/safety_status` — current state, not events (ADR-0096)

The only **latched** topic here. `/openral/estop` and
`/openral/failure/safety` are `VOLATILE` by design: they are event
streams, so a subscriber that connects after the fact sees nothing and
neither carries a notion of *current* state. `SafetyStatus` carries
exactly that — `latched`, `drop_reason`, `detail`, `rskill_id`,
`trace_id`, `header.stamp` — on the "description/static" QoS class, so a
dashboard opened mid-mission or a runner reconnecting after a crash
reads the truth immediately.

Published on **every transition**:

| Path | `latched` | `drop_reason` |
| --- | --- | --- |
| lifecycle activation | current latch | `DROP_NONE` when clear |
| envelope violation | `true` | `KIND_FORCE` / `KIND_WORKSPACE` / `KIND_CONTROLLER` |
| geometric collision | `true` | `KIND_COLLISION` |
| external `/openral/estop` | `true` | `DROP_EXTERNAL_ESTOP` |
| `envelope_unconfigured` drop | `false` | `DROP_ENVELOPE_UNCONFIGURED` |
| world/voxel/state unavailable or overflow | `false` | `DROP_{WORLD,VOXEL,STATE}_UNAVAILABLE`, `DROP_{WORLD,VOXEL}_OVERFLOW` |
| attached payload unverifiable (ADR-0092) | `false` | `DROP_ATTACHED_UNAVAILABLE`, `DROP_ATTACHED_OVERFLOW` |
| chunk accepted after a drop | `false` | `DROP_NONE` |
| `/openral/estop_reset` succeeded | `false` | `DROP_NONE` |

Two rules make the durable value trustworthy (hazard-log HZ-0096-1):

1. **Publish on every activation**, not only on the next fault — a
   restarted kernel must overwrite the stale sample a still-connected
   consumer is holding within one activation cycle.
2. **Re-stamp at 1 Hz** on the `/diagnostics` heartbeat, so
   `header.stamp` is standing evidence the publisher is alive.
   Consumers treat a status older than their liveness window as
   *unknown, not safe*.

The publication is transition-gated on the `(latched, drop_reason)`
pair, so a drop that repeats for every chunk (an unconfigured envelope,
a stale world model) publishes once and is refreshed by the heartbeat,
never per chunk on the 30-200 Hz path.

This is observability only: it adds a publisher, and changes no
enforcement decision anywhere in the kernel.

## Quickstart

```bash
# 1. Build the envelope file from the robot + skill manifests.
uv run python -m openral_safety.envelope_loader \
    --robot robots/so100_follower/robot.yaml \
    --skill rskills/smolvla-libero/rskill.yaml \
    --out /tmp/openral_safety_envelope.yaml

# 2. Build and launch the C++ kernel.
colcon build --packages-select openral_safety_kernel
source install/setup.bash
ros2 launch openral_safety_kernel kernel_only.launch.py \
    envelope_file:=/tmp/openral_safety_envelope.yaml
```

## Lifecycle

`unconfigured → configure → activate → deactivate → cleanup → shutdown`

- **configure**: reads the envelope YAML; refuses to advance on
  schema mismatch or unset path. Opens publishers / subscribers /
  service. Starts the 1 Hz `/diagnostics` timer.
- **activate**: activates managed lifecycle publishers so messages flow.
- **deactivate**: stops outbound publication; subscribers keep
  receiving (the fault latch still trips on external estop).
- **cleanup**: releases all resources; clears the fault latch and
  counters.

## Fault latch + recovery

On envelope violation OR external `/openral/estop`:

1. Drop the candidate chunk (no republish on `/openral/safe_action`).
2. Publish `FailureTrigger` on `/openral/failure/safety` with
   `kind=KIND_FORCE | KIND_WORKSPACE | KIND_CONTROLLER`,
   `severity=SEVERITY_ABORT`, `evidence_json` (Pydantic-deserialisable),
   `rskill_id` and `trace_id` from the chunk.
3. Publish `std_msgs/Empty` on `/openral/estop`.
4. Publish the latched `SafetyStatus` (`latched=true`, the same numeric
   kind the `FailureTrigger` carries) on `/openral/safety_status`.
5. Set `fault_latch=true`. All further candidates drop with reason
   `estop_latched`.

Recovery is manual: `ros2 service call /openral/estop_reset
std_srvs/srv/Trigger`. The service refuses to clear the latch until
`estop_reset_cooldown_s` (default 500 ms) has passed since the last
estop publish — `ROSEStopRequested` is never auto-cleared
(CLAUDE.md §10).

## Observability

The kernel emits one OTel `safety.check` span per candidate chunk over
OTLP/HTTP — the same wire format the in-tree dashboard
(`openral dashboard`, default port 4318) ingests on `/v1/traces`. Spans
carry:

| Attribute | Value |
| --- | --- |
| `safety.check_name` | `"envelope"` |
| `safety.kernel` | `"cpp"` (closed-set; surfaces in Identity card) |
| `safety.severity` | `"info"` (pass), `"warn"` (latched / unconfigured), `"violation"` |
| `safety.drop_reason` | `estop_latched`, `envelope_unconfigured`, `force`, `workspace`, or `controller` |
| `safety.violation_{reason,joint,value,limit}` | populated on `violation` |
| `rskill.id` | `ActionChunk.rskill_id` (short-prefix form the dashboard latches) |

On a violation the span also fires an
`openral.event.safety_violation` event so the dashboard's counted
events ledger ticks.

The W3C `traceparent` carried on `ActionChunk.trace_id` is extracted
with the stock propagator and used as the parent context, so each
`safety.check` span is a child of the producer's `rskill.tick`.

Endpoint resolution follows the standard OTel env vars:

```bash
# Point at the dashboard / Jaeger / any OTLP/HTTP collector.
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
ros2 launch openral_safety_kernel kernel_only.launch.py \
    envelope_file:=/tmp/openral_safety_envelope.yaml
```

When the env var is unset the kernel falls back to
`http://localhost:4318` (the dashboard's default bind).
`BatchSpanProcessor` ferries spans off the chunk-callback thread, so
the validator stays allocation-free
(`test_no_alloc.cpp` still pins the guarantee).

## Real-time guarantees

- Validator (`validate()` in `src/validator.cpp`) is allocation-free.
  Pinned in CI by `test_no_alloc.cpp` via a counting global
  `operator new`.
- C++17 / no exceptions across the kernel boundary — `Result<void,
  Violation>` propagation (CLAUDE.md §5.2).
- `SCHED_FIFO` + CPU pinning are opt-in via the `request_sched_fifo` /
  `cpu_affinity` parameters; the node warns when the privileges are
  unavailable rather than silently downgrading.

## Testing & verification

Three tiers, all driving the **real** `safety_kernel_node` (no mocks):

- **C++ unit** (`just safety-kernel-test`) — `test_validator.cpp`,
  `test_collision.cpp`, `test_lifecycle_kernel.cpp`, and the
  allocation pin `test_no_alloc.cpp` (10,000× runs, zero allocs).
- **Sim** (`tests/sim/safety/`) — the kernel subprocess gated against a
  MuJoCo oracle: envelope (`test_kernel_with_*_twin.py`) and geometric
  collision (`test_kernel_*_collision*.py`, `test_kernel_h1_self_collision.py`,
  `test_kernel_mjcf_lowered_self_collision.py`). These prove
  self/world/voxel `KIND_COLLISION` rejection + estop end-to-end.

> Geometric collision arms automatically in `openral deploy sim` /
> `deploy run`: the launch lowers the robot's collision model
> (`openral_safety.mjcf_lowering` preferred, manifest fallback) and the
> kernel logs `self-collision check enabled: N links`. The lowering
> assigns `dof_index` by **movable-joint order** — matching by joint
> *name* previously froze the FK at the rest pose for robots whose MJCF
> joint names differ from the manifest (issue #77).

## Related

- `cpp/opentelemetry_cpp_vendor` — ROS 2 vendor package this one
  depends on for `opentelemetry-cpp` at colcon-build time.
- `packages/openral_safety/openral_safety/envelope_loader.py` — Python
  bridge that writes the envelope YAML the kernel reads.
- `packages/openral_safety/openral_safety/supervisor_node.py` — Day-1
  Python pass-through; the kernel is the process swap behind the same
  topic contract.
- `python/observability/src/openral_observability/dashboard/store.py`
  — Dashboard's TelemetryStore; consumes the `safety.check` spans
  this kernel emits.
