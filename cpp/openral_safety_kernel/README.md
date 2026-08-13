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
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray`, 1 Hz | default |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

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
4. Set `fault_latch=true`. All further candidates drop with reason
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
| `safety.sweep_min_distance_m` | collision stops only — see below |
| `rskill.id` | `ActionChunk.rskill_id` (short-prefix form the dashboard latches) |

On a violation the span also fires an
`openral.event.safety_violation` event so the dashboard's counted
events ledger ticks.

### Collision evidence: one pair, one distance

A collision stop publishes `CollisionEvidence` whose `link_a`,
`link_b_or_object` and `min_distance_m` all describe **the same** geometry
pair — the deepest pair that actually tripped the check. `min_distance_m` is
never the sweep-wide minimum: a collision sweep also measures pairs that stayed
clear of the margin and pairs the gate deliberately exempted (an attached
payload's attach-time contact baseline, including the payload's own uncleared
occupancy residue), and quoting one of those against the cell that stopped the
robot sends diagnosis after a penetration that never happened.

The sweep-wide minimum is still useful, so it is reported **separately and
never in the evidence payload**: `sweep_min_distance_m=` in the
`safety.collision` log line, and the `safety.sweep_min_distance_m` span
attribute. `sweep_min_distance_m <= min_distance_m` always. A large gap between
the two means something deep is being tolerated on purpose — usually a grasped
payload's own occupancy residue — not that the stop was deep.

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

## Frame convention (ADR-0095)

The kernel applies **no transforms**. It FKs each link from the manifest's
per-joint `origin_xyz` / `origin_rpy` starting at the robot's `base_frame`, and
rasterizes the resulting capsules straight against `/openral/world_voxels`,
whose `header.frame_id` and `origin` are taken at face value. Everything the
kernel touches therefore lives in one frame, `base_frame`, and that frame is
whatever `/tf` says it is:

```
odom -> base_link        ← MobileBaseBridge (from the HAL's base_pose_6dof)
  ├─ base_link -> <cam>_optical_frame   ← the depth extrinsic feeding octomap
  │     …octomap_server → openral_octomap_bridge → /openral/world_voxels
  └─ manifest origin_xyz chain          ← this kernel's collision FK
```

The two branches must be measured against the **same body**. On robosuite /
RoboCasa mobile manipulators `base_link` is the arm mount at the top of the
0.70 m pedestal (`mobilebase0_support`), not the ground-level chassis root, so
`robots/panda_mobile*/robot.yaml` carries the plain Franka `panda_joint1`
origin `[0, 0, 0.333]`. PR #103 briefly set it to `1.033` to cancel a
producer-side frame bug *inside* the kernel; ADR-0095 fixes the producer at
source and reverts the root **in the same commit**, because either half alone
leaves the kernel a full pedestal away from the obstacles it is checked against
(hazard **HZ-0095-1**).

That pairing is envelope-neutral by construction — the grid's content and the
FK root move by the same −0.700 m, and a capsule-vs-voxel distance depends only
on their difference. `VoxelCollision.BaseFrameAlignmentPreservesTheProtective`
`Envelope` asserts the hit flag, the evidence cell index and `min_distance` are
identical across the pair; `VoxelCollision.HalfAppliedFrameAlignmentMovesThe`
`KernelByThePedestal` asserts a half-applied change blinds the kernel.

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
