# ADR-0020: C++ safety kernel

- Status: **Proposed**
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: [ADR-0018](0018-ros2-reasoner-supervisor.md) §5 (the topic
  contract this ADR completes); CLAUDE.md §1.1 (safety beats
  helpfulness), §1.5 (Python proposes, C++ disposes), §6.1 Layer 6
  (safety), §7.7 (safety working-group review), §10
  (`ROSSafetyViolation` never auto-cleared).

## Context

ADR-0018 §5 locked the topic contract for the chunk-rate safety
boundary and shipped F5 as a Python pass-through
(`packages/openral_safety/openral_safety/SafetyPassthroughNode`) that:

* Subscribes `/openral/candidate_action`.
* Validates against a stub envelope (joint position limits, n_dof
  match — first row only).
* Republishes on `/openral/safe_action` when valid; drops + publishes
  `/openral/estop` and a stderr log on violation.
* Serves `/openral/estop_reset` with a cooldown.

The Python pass-through is intentionally inert beyond the topic
contract (CLAUDE.md §1.5: Python proposes; C++ disposes). It does NOT:

* Publish a typed `FailureTrigger` on `/openral/failure/safety`.
* Open an OTel `safety.check` span or emit LTTng tracepoints.
* Enforce velocity / force / workspace AABB.
* Iterate the full `horizon` × `n_dof` payload.
* Carry a no-allocation guarantee on the hot path.

ADR-0018 §5 calls those gaps out and defers them to a follow-up ADR;
this ADR is it.

## Decision

### 1. Process model

The kernel ships as a **separate ROS 2 process**
(`cpp/openral_safety_kernel/`), built via ament_cmake. It is a
`rclcpp_lifecycle::LifecycleNode` named `openral_safety_kernel`. It
replaces the Day-1 Python `SafetyPassthroughNode` behind the same topic
contract — same publishers, same subscribers, same `/openral/estop_reset`
service. The Python skeleton is retained so the package metadata stays
stable and ament_python tooling continues to discover the supervisor
name; production deployments choose between the two via launch-file
selection.

### 2. Topic / service contract (unchanged from ADR-0018 §1)

| Direction | Topic / Service | Type | QoS |
| --- | --- | --- | --- |
| sub | `/openral/candidate_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=1 |
| sub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| pub | `/openral/safe_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=1 |
| pub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| pub | `/openral/failure/safety` | `openral_msgs/FailureTrigger` | RELIABLE, VOLATILE, KL=50 |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray`, 1 Hz | default |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

### 3. Envelope contract

The envelope intersection (ADR-0018 §5) is computed Python-side by
`openral_safety.envelope_loader` (planned: `packages/openral_safety/openral_safety/envelope_loader.py`):

* Takes a `RobotDescription` (ceiling) and an optional `RSkillManifest`
  (tighter floor).
* Per-field intersection: scalar `max_*` use `min(robot, skill)`;
  workspace AABB uses the skill box (already proven `⊆` robot);
  `deadman_required` is logical OR. Only **explicitly-set** skill fields
  participate (Pydantic `model_fields_set`) — partial skill envelopes do
  not silently overwrite with schema defaults.
* **Loosening is rejected at goal acceptance**: any skill field that
  loosens the robot's ceiling raises
  `openral_core.exceptions.ROSConfigError`. Never silently honored
  (CLAUDE.md §1.1).
* Writes a flat YAML (`schema_version: 1`) the C++ kernel slurps at
  `on_configure()` via `envelope.cpp` (yaml-cpp).

The new optional `envelope: SafetyEnvelope | None` field on
`openral_core.RSkillManifest` (PR-A) makes the skill-side declaration
type-safe. Pre-existing manifests without the field keep loading
unchanged and inherit the full robot ceiling.

### 4. Validation algorithm

`Result<void, Violation> validate(ChunkView, EnvelopeIntersection)` in
`cpp/openral_safety_kernel/src/validator.cpp`. Per chunk:

1. `n_dof == envelope.n_dof`, else `KIND_CONTROLLER /
   ControllerSubKind::kNdofMismatch`.
2. `flat.size() == horizon * n_dof`, else `KIND_CONTROLLER /
   kDimMismatch`.
3. Every element of `flat[]` is finite (no NaN/Inf), else
   `KIND_CONTROLLER / kNanInAction` with the offending index.
4. Per control_mode:
   * `JOINT_POSITION` → per-step per-joint `[min, max]` → `KIND_WORKSPACE`.
   * `JOINT_VELOCITY` → per-step per-joint `|v| ≤ joint_velocity_max[j]`
     (pre-multiplied by `max_joint_speed_factor`) → `KIND_WORKSPACE`.
   * `JOINT_TORQUE` → per-step per-joint
     `|τ| ≤ min(joint_torque_max[j], max_torque_nm)` → `KIND_FORCE`.
   * `CARTESIAN_POSE` → xyz position inside workspace AABB → `KIND_WORKSPACE`.
   * `CARTESIAN_TWIST` → `|v| ≤ max_ee_speed_m_s` → `KIND_FORCE`.
   * Unknown mode → `KIND_CONTROLLER / kDimMismatch`.

The kernel **rejects, does not clamp** (CLAUDE.md §1.4 — explicit beats
implicit). Clamping as graceful degradation is a v2 ADR.

### 5. Real-time guarantees

* `validate()` is allocation-free on the hot path. Pinned in CI by
  `test_no_alloc.cpp` (planned: `cpp/openral_safety_kernel/test/test_no_alloc.cpp`)
  via a global `operator new` counter; the test runs the validator
  10 000 times on the pass-through path and 5 000 times on the
  violation path and asserts zero allocations.
* C++17, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wnon-virtual-dtor -Wold-style-cast`.
* No exceptions across the kernel boundary — `Result<void, Violation>`
  propagation (CLAUDE.md §5.2).
* `SCHED_FIFO` + CPU affinity are opt-in via the `request_sched_fifo`
  and `cpu_affinity` parameters. The node warns when the privileges are
  unavailable rather than silently downgrading.

### 6. Failure semantics

On violation:

1. Drop the candidate (no `/openral/safe_action` publish).
2. Publish a typed `openral_msgs/FailureTrigger` on
   `/openral/failure/safety` with `kind ∈ {KIND_FORCE,
   KIND_WORKSPACE, KIND_CONTROLLER}`, `severity = SEVERITY_ABORT`,
   `evidence_json` carrying a Pydantic-deserializable
   `openral_core.FailureEvidence` discriminated union value
   (`ForceEvidence`, `WorkspaceEvidence`, or `ControllerEvidence`),
   `skill_id` and `trace_id` copied verbatim from the chunk.
3. Publish `std_msgs/Empty` on `/openral/estop`.
4. Set `fault_latch = true`. Every subsequent candidate drops with
   reason `estop_latched` until `/openral/estop_reset` succeeds.

### 7. Recovery

`/openral/estop_reset` is a `std_srvs/Trigger` service. Recovery is
manual (CLAUDE.md §10 — `ROSEStopRequested` never auto-cleared). The
service refuses to clear the latch until `estop_reset_cooldown_s`
(default 500 ms) has passed since the most recent estop publish.

### 8. Defense in depth

The kernel subscribes to `/openral/estop` itself so externally-triggered
estop sources latch the kernel. Four estop producers run alongside
(ADR-0018 §5):

| Source | Process |
| --- | --- |
| The kernel itself | `openral_safety_kernel` (this ADR) |
| Hardware pendant | `openral_safety_watchdog.hardware_estop_node` |
| Deadman timeout | `openral_safety_watchdog.deadman_watchdog_node` |
| Human channels | `openral_human_estop.HumanEstopForwarderNode` |

Each runs in its own process so the death of any one — including the
kernel — does NOT take down the whole estop surface.

### 9. Observability

* `/diagnostics` published at 1 Hz with lifecycle state, fault latch,
  pass / drop counts, last drop reason, and whether the envelope is
  loaded.
* `evidence_json` is the join key between C++ violations and the
  Python-side `openral_core.FailureEvidence` discriminated union — the
  reasoner already subscribes to `/openral/failure/safety` (ADR-0018 F4
  / PR #125) and deserializes evidence with `TypeAdapter`.
* OTel span emission via `opentelemetry-cpp` is wired (PR-F amendment
  below). LTTng tracepoints (`openral:safety_check_{begin,end}`) remain
  on the follow-up list and continue to ride the contract locked by PR
  #131's tracepoint helper (`OPENRAL_ROS2_TRACING=1` env gate).

### 10. Licensing

The kernel ships under **Apache-2.0**. Build-time dependencies:

| Dep | License | Disposition |
| --- | --- | --- |
| `opentelemetry-cpp` (planned) | Apache-2.0 | direct use |
| `yaml-cpp` | MIT | direct use |
| `nlohmann_json` (planned) | MIT | direct use |
| `gtest` | BSD-3 | test-only |
| `lttng-ust` (planned) | LGPL-2.1 | **dynamic link only**, gated on `OPENRAL_ROS2_TRACING=1` — TSC review per CLAUDE.md §1.9 |

Vendor-specific safety I/O (Franka FCI safety bits, UR cobot safety
words) stays out of this package and lives in
`contrib-closed-shims/` (CLAUDE.md §8).

## Consequences

### Positive

* Real envelope enforcement on the chunk-rate boundary (joint position
  / velocity / torque + cartesian AABB + ee-speed) with a typed
  FailureTrigger so the reasoner replans without parsing stderr logs.
* No-allocation validator pinned by CI — the hot path is bounded.
* Defense-in-depth: four estop producers, two estop subscribers
  (kernel + HAL), recovery requires explicit service call.
* The skill envelope schema (`RSkillManifest.envelope`) lets policy
  authors declare tighter limits per skill — and the loader rejects
  loosening at goal acceptance.

### Negative

* Python `openral_safety.SafetyPassthroughNode` and C++
  `openral_safety_kernel` both live in tree during the swap. The
  Python node is the Day-1 fallback; production runs the C++ kernel.
* `opentelemetry-cpp` and `lttng-ust` are non-trivial deps — the
  former for build complexity (vendored under
  `/opt/ros/<distro>/include` on Jazzy via distro packages, else
  fetched as a subproject), the latter for the LGPL TSC review.

### Neutral / out-of-scope

* FK-based workspace clamping for cartesian-pose actions is deferred to
  v2 — v1 enforces only the AABB on the encoded pose. Joint-space
  motions remain the primary validation path.
* Clamping as a graceful-degradation mode is rejected for v1.
* Cloud dispatch coupling is unchanged: cloud-dispatched skills publish
  the same `/openral/candidate_action`; the kernel does not care where
  the chunk came from.

## Rollout

The kernel landed in a sequence of small PRs (CLAUDE.md §7.2):

1. **PR-A** — `RSkillManifest.envelope` optional field.
2. **PR-B** — `openral_safety.envelope_loader` Python bridge.
3. **PR-C** — `cpp/openral_safety_kernel` bootstrap (CMake + headers).
4. **PR-D** — Pass-through lifecycle node + topic surface.
5. **PR-E** — Real validator + `FailureTrigger` emission.
6. **PR-F** — OTel integration (landed; see Amendments). LTTng split
   off as PR-F2 (planned).
7. **PR-G** — Defense-in-depth (deadman, hardware estop, human
   forwarder).
8. **PR-H / PR-I** — Sim and HIL test tiers (planned).

The full repo state map flips Layer 6 from `yellow` to `green` once
the kernel + defense-in-depth nodes are merged and the HIL test tier
passes on `lab-so100`.

## Amendments

### 2026-05-20 — PR-F: OTel `safety.check` span emission

The kernel now emits one OTel `safety.check` span per
`/openral/candidate_action` callback, matching the contract that
`python/observability/.../tracing.py:107-111` and the dashboard
TelemetryStore (`store.py:591-603`) lock. Specifically:

- `service.name="openral_safety_kernel"` resource attribute.
- Span attributes: `safety.check_name="envelope"`,
  `safety.kernel="cpp"` (closed-set value from
  `openral_observability.semconv.SAFETY_KERNEL_CPP`),
  `safety.severity` ∈ {`info`, `warn`, `violation`},
  `safety.clamped=false`, `rskill.id` (semconv.RSKILL_ID).
- On `warn` / `violation`: `safety.drop_reason` carries the latch /
  envelope kind. On `violation`: additional
  `safety.violation_{reason,joint,value,limit}` attributes plus a
  span event `openral.event.safety_violation` so the dashboard's
  `_COUNTED_EVENTS` set ticks.
- The W3C `traceparent` carried in `ActionChunk.trace_id` is extracted
  with the stock `HttpTraceContext` propagator and used as the parent
  context — so the kernel's `safety.check` is a child of the runner's
  `rskill.tick` (ADR-0018 §6: "OTel context is the truth").

Transport is OTLP/HTTP protobuf via `opentelemetry-cpp`'s
`OtlpHttpExporter` + `BatchSpanProcessor`, pointed at
`OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`, which
is the dashboard's bind port from
`python/observability/.../dashboard/server.py:29`). The processor
ferries spans off the chunk-callback thread on its background flush
worker, so the validator stays allocation-free (`test_no_alloc.cpp`
still pins the guarantee — the no-alloc scope wraps `validate()`, not
`on_candidate_action`).

Build dep: a new ROS 2 vendor package
`cpp/opentelemetry_cpp_vendor` fetches and builds upstream
`opentelemetry-cpp` v1.16.1 at colcon-build time (Ubuntu 24.04 has no
apt package). The vendor builds trace + OTLP-HTTP only — no gRPC, no
metrics, no Prometheus / Jaeger / Zipkin exporters — to keep the
first-build cost bounded.

Tested by `test/launch/test_e2e_otel.py`: a loopback FastAPI receiver
on a free port decodes `ExportTraceServiceRequest` and asserts that
one pass + one violation produce two `safety.check` spans with
`safety.kernel="cpp"`, `safety.check_name="envelope"`, the right
severities, and a `openral.event.safety_violation` event on the
violation span. The receiver mirrors the dashboard's `/v1/traces`
route shape exactly, so a green test here is a green Safety card on
the dashboard.

PR-F2 — LTTng tracepoints — remains planned per the original ADR;
no schedule change.
