# openral_safety

> Day-1 Python pass-through + Python helpers the
> C++ safety kernel uses at configure time. The real-time enforcer is
> `cpp/openral_safety_kernel/`; this package is the Python
> seam that wraps it.

## What's here

* **`SafetyPassthroughNode`** (`openral_safety/supervisor_node.py`) —
  Day-1 lifecycle node that locks the topic contract.
  Subscribes `/openral/candidate_action`, gates to
  `/openral/safe_action`, fires `/openral/estop` on stub envelope
  violation (n_dof + per-joint position), and serves
  `/openral/estop_reset` with a cooldown. Subscribes to
  `/openral/estop` itself (defense in depth, CLAUDE.md §1.5).
* **`SafetySupervisorNode`** — back-compat alias for
  `SafetyPassthroughNode`.
* **`envelope_loader`** (`openral_safety/envelope_loader.py`) — pure
  Python helper that intersects a `RobotDescription.safety` ceiling
  with an optional `RSkillManifest.envelope` floor and writes the flat
  YAML the C++ safety kernel reads at `on_configure()`. Rejects
  loosening with `ROSConfigError`.
* **`kernel_predicates`** (`openral_safety/kernel_predicates.py`) — the
  Python mirror of the C++ kernel's narrow phase
  (`cpp/openral_safety_kernel/src/collision.cpp`): the 15-axis SAT
  `box_box_distance`, `box_capsule_distance`, `capsule_distance` and the
  `shape_distance` dispatch, batched over configurations. Offline tools
  that decide **what the kernel will do** must ask the kernel's question
  with the kernel's geometry; issue #155 is what happens when they do
  not. **The C++ is normative — when it changes, this changes in the same
  PR** (`docs/methods/14-duplication-watch.md` item 12).
* **`urdf_lowering`** / **`mjcf_lowering`**
  (`openral_safety/urdf_lowering.py`, `mjcf_lowering.py`) — offline
  lowering of a URDF(+SRDF) or MJCF into a manifest's
  `collision_geometry` + `allowed_collision_pairs`. An ACM entry
  *removes* a self-collision check, so every rule fails toward **fewer**
  entries; "always-colliding" is established as a proof over each pair's
  relative-DoF subspace, never by sampling.
* **`cumotion_config`** (`openral_safety/cumotion_config.py`) — renders a
  cuRobo/cuMotion `robot_cfg` from the lowered geometry. Its plan-time
  model deliberately **over**-covers the kernel's: a planner that
  believes the arm is thinner than it is emits trajectories the kernel
  then has to E-stop.

## The collision tree must be connected

`collision_params_from_description` lowers the robot's kinematic tree for the
kernel's forward kinematics. That tree is the union of two manifest lists:

* **`joints`** — the robot's *movable* joints, and only those. It is consumed
  **unfiltered** elsewhere as the action-vector width, the sim `qpos` map, the
  reasoner's rSkill state-contract filter and the runner's joint permutation,
  so a zero-DoF entry here corrupts all four. Never add a fixed joint to it.
* **`fixed_attachments`** — the rigid, zero-DoF mounts (`FixedAttachment`):
  a hand bolted to a flange, a bimanual rig's arm pedestals. No DoF, no limits,
  never a command channel, invisible to every `joints` consumer.

Together they must describe **exactly one connected tree**. A manifest that
leaves a rigid mount out is *refused* with `ROSConfigError` naming the
disconnected roots — it is not lowered.

The loader previously treated any link no joint reached as a second base at the
identity frame. That silently placed the whole orphaned subtree on the robot's
origin, which is wrong in both directions: it fabricates contacts that cannot
happen, and — the reason this is a hazard rather than a nuisance — it leaves
that subtree's real swept volume entirely unmodelled, so a genuine
self-collision is never detected. There is no safe default for a link whose
pose is unknown, so the loader refuses instead of guessing.

Every attachment origin must be read out of the robot's real URDF/MJCF at the
zero configuration (composed, where the source model inserts intermediate
links), exactly as `joints`' own origins are. Never estimate a mount transform.
`tests/sim/safety/test_collision_fk_matches_source_model.py` enforces this: it
runs the kernel's FK over the lowered parameters and asserts every link lands
within 10 µm of the same body in the robot's **normative** model — the one its
manifest was lowered from. Which model that is matters: a robot's URDF and its
MJCF can describe the same machine with different intermediate frames (`g1`'s
`torso_link` sits 10 mm apart between the two while both agree on every link
above it), so checking against the wrong one measures a convention, not a
defect.

Per CLAUDE.md §7.7 / §1.1, any PR that **extends** enforcement here
requires:

1. Explicit reviewer assignment to the safety working group.
2. An update to the safety hazard log in the private OpenRAL/management
   repo (`safety/hazard-log.md`).
3. Tests proving the new behaviour is at least as conservative as
   the old.

This package is the Day-1 implementation of the locked topic-boundary
contract.

## Layer

CLAUDE.md §6.1 Layer 6 (Safety). The Python-side `SafetyClient`
Protocol stays at `python/runner/src/openral_runner/safety.py`
(`NullSafetyClient`) — it remains the in-process tick-time gate the
`DeployRunner` calls. This package is the **chunk-rate topic
boundary** the `rskill_runner_node` and `<robot>_hal_node` peer with.
The C++ kernel that ultimately replaces this node's internals lives
at `cpp/openral_safety_kernel/`.

## Topic surface (locked)

| Direction | Topic | Type | QoS |
|---|---|---|---|
| sub | `/openral/candidate_action` | `openral_msgs/ActionChunk` | RELIABLE · VOLATILE · KL=1 |
| pub | `/openral/safe_action` | `openral_msgs/ActionChunk` | RELIABLE · VOLATILE · KL=1 |
| pub | `/openral/estop` | `std_msgs/Empty` | RELIABLE · VOLATILE · KL=10 |
| pub | `/openral/safety_status` | `openral_msgs/SafetyStatus` | RELIABLE · **TRANSIENT_LOCAL** · KL=1 |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) | RELIABLE · VOLATILE · KL=10 |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

`/openral/safety_status` (ADR-0096) is the only **latched** topic here:
current safety state (`latched`, `drop_reason`, `detail`, `rskill_id`,
`trace_id`, `header.stamp`) rather than an event, so a dashboard opened
mid-mission or a runner reconnecting after a crash reads the truth
immediately. Published on every latch transition, every clear/recovery
transition, on **every activation** (hazard-log HZ-0096-1 mitigation 1
— a restarted node must overwrite the stale durable sample a still
connected consumer holds), and re-stamped at 1 Hz so `header.stamp` is
evidence the publisher is alive. The C++ kernel publishes the identical
contract, so this node keeps replacing/being replaced behind the same
topic surface. Before it, this node had no typed failure output at all
— it never constructed a `FailureTrigger` — so a fail-closed drop or
e-stop from here was a bare `std_msgs/Empty` and nothing else.
Observability only: no enforcement path changed.

`/openral/estop` is subscribed by **both** the HAL and the
skill_runner (defense in depth, CLAUDE.md §1.5).

## Day-1 envelope checks (stub, but real)

* `n_dof` mismatch vs the node's `n_dof` parameter
  (default `-1` ≡ "do not enforce", set to the robot's DOF in
  production launches).
* First-row joint targets vs `min_joint` / `max_joint` per-joint
  position limits (both empty ≡ "do not enforce").

Velocity, force, and workspace AABB enforcement land with the C++
kernel — those are intentionally **not** implemented in Python (§7.7
prohibits a divergent Python-side enforcer that has to be re-validated
when the kernel lands).

On envelope violation:

1. The candidate `ActionChunk` is dropped (not republished).
2. `std_msgs/Empty` is published on `/openral/estop`.
3. The node latches into an estop state; subsequent chunks are dropped
   until `/openral/estop_reset` is called and the 500 ms cooldown has
   elapsed. `ROSEStopRequested` (CLAUDE.md §10) is never
   auto-cleared.

## Production vs Day-1

The C++ safety kernel ships as a **process swap** behind the same
topic contract — same publishers, same subscribers, same
`/openral/estop_reset` service. Production deployments choose between
the Python pass-through (here) and the C++ kernel via launch-file
selection; the rest of the graph (rskill_runner_node, reasoner_node,
HAL adapters) is identical.

```python
from openral_safety.supervisor_node import SafetyPassthroughNode
# Back-compat alias still exported:
from openral_safety.supervisor_node import SafetySupervisorNode
```

`SafetyPassthroughNode` is a managed-lifecycle node with the standard
five transition callbacks; the heartbeat is wired via
`openral_observability.DiagnosticsHeartbeat`.

## Why a Python pass-through *and* a C++ kernel

CLAUDE.md operating principles forbid (§1.1, §1.5):

* Catching `ROSSafetyViolation` and continuing.
* Hidden retries or fallbacks.
* Python proposing **and** disposing — actuation-side enforcement must
  be C++ to meet the real-time guarantees.

The Day-1 Python pass-through exists so the topic contract is locked
end-to-end *before* the kernel lands. Once the C++ safety kernel is on
disk the Python node remains for digital-twin runs / pre-hardware tests;
the kernel runs in production.

## Related

* The ROS 2 reasoner + supervisor graph spec §F5 / §5 — normative spec.
* `cpp/openral_safety_kernel/` — the real-time C++ enforcer.
* `packages/openral_safety_watchdog/` — deadman + hardware-estop
  watchdog nodes (reasoner + supervisor graph spec §5 bullets 3 & 4).
* `packages/openral_human_estop/` — human estop forwarder
  (reasoner + supervisor graph spec §5 bullet 2).
* `python/runner/src/openral_runner/safety.py` — in-process
  `SafetyClient` Protocol + `NullSafetyClient` (the in-process seam
  the runner calls every tick). Independent of the topic boundary
  surfaced here.
* CLAUDE.md §1.5, §6.1, §7.7, §10 (`ROSSafetyViolation` /
  `ROSEStopRequested`).
