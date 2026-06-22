# ADR-0064 — Dashboard guarded write-controls

- **Status:** Proposed — pending safety-WG review
- **Date:** 2026-06-21
- **Related:** ADR-0018 (reasoner/dispatch + operator prompt F10), issue #75
  (dashboard improvements), issue #81 (raw joint teleop — explicit non-goal),
  issue #44 (dashboard localhost-only, no auth)

## Context

The observability dashboard (`python/observability/.../dashboard/`) is an
unauthenticated, localhost-only service (issue #44). It was designed purely as a
read-only OTLP receiver and visualizer. Operators have requested the ability to
switch the active skill and tune a ROS 2 parameter directly from the console
(issue #75, sub-feature (c)).

Both operations reach actuation configuration and therefore engage several
CLAUDE.md constraints:

- **§1.1** — Safety beats helpfulness; never lower a velocity limit without a
  paper trail; never silently bypass any safety check.
- **§1.5** — Python touches motors only through a typed bridge to `ros2_control`
  with a watchdog; the hot path is C++.
- **§3** — The C++ safety kernel is authoritative; never add a path that bypasses
  the E-stop or that could leave motors energized after a Python crash.

Because the dashboard currently has no authentication layer and these write
operations are novel to its surface, they must be gated and traceable before
any merge is considered safe.

## Decision

### 1. Default-off flag

Write-controls are **disabled by default**. They are unlocked only when the
operator explicitly sets:

```
OPENRAL_DASHBOARD_WRITE_CONTROLS=1
```

When the flag is absent (or set to any value other than `1`):

- All write endpoints (`POST /api/skill/execute`, `POST /api/param/set`) return
  **HTTP 403** immediately.
- The UI hides the write-control widgets entirely (no dead buttons).

When the flag is set to `1`:

- A **loud banner** is shown in the dashboard UI.
- A **stderr/structlog warning** is emitted at startup, citing this ADR number
  and noting "pending safety-WG review."
- Write endpoints become active under the constraints below.

### 2. Skill-switch (`POST /api/skill/execute`)

The endpoint shells out to:

```
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill {...}
```

This is the **sole documented external trigger** for a skill (the
`rskill_runner_node` contract). The endpoint introduces **no new actuation
path**: the C++ safety kernel continues to inspect, rate-limit, and dispose
every motion chunk. This mirrors the existing `estop_reset` and
`openral-prompt` shell-out pattern already present in the observability layer.

**Async accept-then-track (HTTP 202):** the endpoint returns as soon as the
action server **accepts** the goal, not when the skill finishes. This avoids
504 errors on long-running skills. The response body includes `goal_id` so the
operator can correlate telemetry. Execution progress is tracked via dashboard
SSE / OTLP telemetry. Acceptance timeout is configurable via
`OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S` (default `12` s).

Response codes:

| Code | Meaning |
|------|---------|
| **202** | Goal accepted; execution in progress |
| **409** | Action server rejected the goal |
| **504** | Action server did not accept within `OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S` |
| **502** | `ros2` subprocess exited before printing any acceptance line |
| **503** | `ros2` not on PATH |

Every invocation is audit-logged via `structlog` before the shell-out is
issued, regardless of outcome. On acceptance, `goal_id` is included in the
audit record. A background task drains remaining subprocess output and
audit-logs the eventual outcome (see §4).

### 3. Param-tune (`POST /api/param/set`)

The endpoint shells out to `ros2 param set <node> <param> <value>`, but
enforces a **safety-parameter denylist** checked before any shell-out is
attempted.

A request is **refused with HTTP 403 and a paper-trail log message** if the
parameter name contains any of the following substrings (case-insensitive).
Both compact and ROS 2 underscored forms are listed so that names such as
`e_stop_enable`, `dead_man_timeout`, `safe_mode`, and `safe_zone_radius` are
caught in addition to the compact spellings. The list is **fail-closed**: when
in doubt, refuse (operator uses a reviewed config/manifest change instead).

| Substring | Rationale |
|-----------|-----------|
| `velocity` | velocity limits and gains |
| `accel` | acceleration limits |
| `force` | force/torque limits and thresholds |
| `torque` | torque limits |
| `limit` | generic limit parameters |
| `workspace` | workspace boundary constraints |
| `estop` | compact e-stop form |
| `e_stop` | ROS 2 underscored e-stop form (`e_stop_enable`, `e_stop_triggered`, …) |
| `deadman` | compact deadman-switch form |
| `dead_man` | ROS 2 underscored deadman form (`dead_man_timeout`, …) |
| `safety` | safety-supervisor parameters |
| `safe` | `safe_mode`, `safe_zone_radius`, `safe_dist`, … (also subsumes `safety`) |
| `watchdog` | watchdog timeout parameters |

Refusals are **never silent**. The response body identifies the matched
substring; the structlog audit record includes the requesting parameter name,
the matched rule, and a reference to this ADR.

Permitted param-set operations are also audit-logged before the shell-out.

### 4. Audit log

Every write attempt — whether permitted or denied — produces a structlog event
at WARNING level with the following mandatory fields:

- `event`: `"dashboard_write_attempt"`
- `operation`: `"skill_execute"` or `"param_set"`
- `outcome`: `"sent"` | `"accepted"` | `"rejected"` | `"denied_denylist"` | `"denied_flag_off"`
- `adr`: `"ADR-0064"`
- `operator_ip`: loopback address (always `127.0.0.1` given the localhost-only
  surface)
- operation-specific fields (skill ID or param name + node, `goal_id` on
  acceptance, denylist match if denied)

Additionally, after skill execution completes, a **second** structlog event
is emitted by the background drain task:

- `event`: `"dashboard_skill_result"`
- `goal_id`, `skill_id`, `success` (bool), `trace_id` (str or `None`),
  `failure_reason` (str or `None`)

`prompt` and `goal_params_json` are **never** included in any log event.

## Non-goals

- **Raw per-joint teleoperation** (the S0 cerebellar layer, 500–1000 Hz, C++
  only) is explicitly **out of scope** and tracked separately in issue #81.
  The dashboard operates only at the skill/parameter abstraction; it has no
  direct line to `ros2_control` joint controllers.
- **Authentication or authorization hardening** of the dashboard surface is
  out of scope for this ADR; it is tracked in issue #44.
- **UI configuration panels** for safety parameters; those belong in a
  safety-WG-owned tool, not the observability dashboard.

## Consequences

### Before merge

This ADR requires:

1. **Safety-WG sign-off** on the denylist completeness and the audit-log schema.
2. A **hazard-log entry** covering the residual risk that a permitted param-set
   could indirectly affect motion quality (e.g., a planner tolerance that
   widens a safety margin).

Neither (1) nor (2) is complete at time of writing. This ADR is **Proposed**,
not Accepted.

### After merge (if accepted)

- The safety kernel remains the sole authoritative gate on motion; no new
  actuation path is introduced.
- No E-stop bypass exists; a Python crash in the dashboard leaves actuation
  flowing through the existing kernel-gated action path unchanged.
- Operators gain a traceable, audit-logged way to dispatch skills and tune
  non-safety parameters from the console — without requiring a separate
  terminal session.
- The default-off flag means existing deployments are unaffected until an
  operator explicitly opts in.
- The denylist is a defense-in-depth measure, not a safety-critical boundary
  on its own; the kernel remains the primary control.
