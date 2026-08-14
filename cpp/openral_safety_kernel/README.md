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
clear of the margin and pairs the gate deliberately exempted (see the exemption
ladder below), and quoting one of those against the cell that stopped the
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

## Attached-payload exemption ladder (ADR-0092)

A grasped payload is checked as robot geometry against the occupancy map. That
is the point — the carried object must still collide with cabinets and people.
But two contacts are legitimate and neither is distinguishable from a real
penetration by depth alone, so the kernel grants exactly two bounded
exemptions inside `check_attached_voxel_collision`, and nothing else. An
exempted cell reaches `sweep_min_distance` only; it can never be the cell an
E-stop names.

**1. Support-contact witness (ADR-0092 D6).** Attachment says what the robot is
carrying, not that the payload is free of its environment: a grasped baguette
is still lying on the counter. World State attests that one contact at attach
time on the existing `attached_objects` path — a support identity, a contact
point and outward normal in the **attached object's own frame**, a patch
radius, and the **physical** contact depth. The kernel exempts a cell only when
it is inside the patch laterally *and* no higher above the attested support
plane than

```
half_resolution · (|n.x| + |n.y| + |n.z|)   +   attested depth   +   slack
└─ exact half-width of the voxel cube ──┘       └─ physical ─┘       └ 1 mm ┘
      projected on the support normal
```

That first term is what makes the witness work under coarse maps. A surface
cell's cube overshoots the true surface it discretises, so a 1 mm physical
contact reads as up to ~15 mm of cube penetration at 25 mm resolution — the
2026-08-13 baguette run E-stopped on exactly that, `-15.70 mm` reported against
six real MuJoCo contacts of `-0.87..-1.37 mm`. Measuring depth against the
attested plane instead of the cell cube accounts for the inflation *exactly*,
by geometry, rather than absorbing it into a widened tolerance that would also
license real penetration. A cell whose solid sits genuinely above the attested
support face still stops the robot, and a payload driving into its support
raises that height millimetre for millimetre until it does.

The witness is **latched, and it dies on separation**: once nothing it would
exempt is still in contact, its bit clears and the kernel never sets it again.
Only a genuinely new attestation — keyed on `(object_id, support_id,
stamp_ns)`, so the heartbeated attachment snapshot cannot resurrect it —
re-arms it. Lift and re-contact is a new violation; so is a regrasp; so is any
contact against a surface the witness does not name. No attestation at all —
which is what the vision attachment producer emits, honestly — means no
exemption.

Layer-2 does not get unbounded trust: `support_witness_max_patch_radius_m`
(0.5) and `support_witness_max_penetration_m` (0.01) cap what an attestation
may claim, and a witness beyond them fails the whole attachment message closed.

**2. Embedded attach-time residue.** The payload's own occupancy left in the
map at attach — a cell already at least half a voxel inside the payload when
the baseline was snapshotted. This is stale self-occupancy, a different
phenomenon from support contact, and the witness deliberately does not cover
it; the attach-time baseline snapshot is retained solely for this.

**What this replaced.** The non-deepening index-keyed baseline (a per-cell
attach-time distance, with any penetration no deeper than it allowed) and the
`attached_contact_allow_new_shallow` contact-phase allowance are both **gone**.
The index-keyed baseline decorrelated in exactly the situation it was meant to
cover: voxel indices shift as the base drives and the lattice re-phases, while
the physical contact persists, so the recorded cells stop matching the cells
actually touched. `allow_new_shallow` was worse — it exempted new cells no one
had attested, up to a tolerance that had been raised to the voxel size to make
supported motion pass. Both removals move cases from no-stop to stop.

`attached_contact_tolerance_m` survives with its name's meaning restored:
physical slack for FK and pose noise, defaulting to 1 mm. It is no longer
overridden to the octomap resolution in `sim_e2e.launch.py` (hazard
**HZ-0095-2**), because the quantisation it was standing in for is now handled
geometrically by the witness predicate.

With a live witness the predicted Cartesian steps are checked too — the old
path skipped them whenever attach-time contact was active, because it had no
pose-dependent way to tell the support contact apart. The skip remains only for
the unattested legacy case.

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
