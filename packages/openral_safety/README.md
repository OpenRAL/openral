# openral_safety

> ADR-0018 §5 F5 — Day-1 Python pass-through + Python helpers the
> C++ safety kernel uses at configure time. The real-time enforcer is
> `cpp/openral_safety_kernel/` (ADR-0020); this package is the Python
> seam that wraps it.

## What's here

* **`SafetyPassthroughNode`** (`openral_safety/supervisor_node.py`) —
  Day-1 lifecycle node that locks the topic contract from ADR-0018 §1.
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
  loosening with `ROSConfigError` (ADR-0018 §5).

Per CLAUDE.md §7.7 / §1.1, any PR that **extends** enforcement here
requires:

1. Explicit reviewer assignment to the safety working group.
2. A hazard-log update.
3. Tests proving the new behaviour is at least as conservative as
   the old.

ADR-0018 §F5 is the normative spec; this package is its Day-1
implementation.

## Layer

CLAUDE.md §6.1 Layer 6 (Safety). The Python-side `SafetyClient`
Protocol stays at `python/runner/src/openral_runner/safety.py`
(`NullSafetyClient`) — it remains the in-process tick-time gate the
`DeployRunner` calls. This package is the **chunk-rate topic
boundary** the `rskill_runner_node` and `<robot>_hal_node` peer with.
The C++ kernel that ultimately replaces this node's internals lives
at `cpp/openral_safety_kernel/` (ADR-0020).

## Topic surface (locked)

| Direction | Topic | Type | QoS |
|---|---|---|---|
| sub | `/openral/candidate_action` | `openral_msgs/ActionChunk` | RELIABLE · VOLATILE · KL=1 |
| pub | `/openral/safe_action` | `openral_msgs/ActionChunk` | RELIABLE · VOLATILE · KL=1 |
| pub | `/openral/estop` | `std_msgs/Empty` | RELIABLE · VOLATILE · KL=10 |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) | RELIABLE · VOLATILE · KL=10 |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

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

ADR-0020 ships the C++ kernel as a **process swap** behind the same
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
end-to-end *before* the kernel lands. Once ADR-0020's kernel is on disk
the Python node remains for digital-twin runs / pre-hardware tests; the
kernel runs in production.

## Related

* ADR-0018 §F5 / §5 — normative spec.
* `cpp/openral_safety_kernel/` — the real-time C++ enforcer
  (ADR-0020).
* `packages/openral_safety_watchdog/` — deadman + hardware-estop
  watchdog nodes (ADR-0018 §5 bullets 3 & 4).
* `packages/openral_human_estop/` — human estop forwarder
  (ADR-0018 §5 bullet 2).
* `python/runner/src/openral_runner/safety.py` — in-process
  `SafetyClient` Protocol + `NullSafetyClient` (the in-process seam
  the runner calls every tick). Independent of the topic boundary
  surfaced here.
* CLAUDE.md §1.5, §6.1, §7.7, §10 (`ROSSafetyViolation` /
  `ROSEStopRequested`).
