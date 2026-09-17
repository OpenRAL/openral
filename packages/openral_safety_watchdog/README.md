# openral_safety_watchdog

ROS 2 reasoner + supervisor graph spec §5 — **defense-in-depth E-stop sources**, independent of the in-band
`openral_safety` node and the C++ safety kernel so a crash in either still
triggers a brake event. Two lifecycle nodes:

| Node | ADR | What it does |
|---|---|---|
| `deadman_watchdog_node` | §5 bullet 4 | Fires `/openral/estop` if no `/openral/safe_action` arrives within `safe_action_deadline_s` while an execution window is open, or if an open window never produces a first chunk within `first_chunk_deadline_s` (see below). Also emits a `FailureTrigger(KIND_TIMEOUT, SEVERITY_ABORT)` on `/openral/failure/safety` so the reasoner sees a structured `TimeoutEvidence` event, not just a bare estop. |
| `hardware_estop_node` | §5 bullet 3 | Bridges a hardware estop source (GPIO relay via libgpiod, or a USB-HID pendant via `/dev/input`) onto `/openral/estop`, polling at `poll_rate_hz` (default 100 Hz) and publishing `std_msgs/Empty` + `FailureTrigger(KIND_HUMAN, SEVERITY_ABORT, HumanEvidence)` on the rising edge. The per-vendor device read is an overridable hook; the base class owns the poll loop, edge detection, and publication. **This repo ships no such vendor driver** — see "Hardware pendant" below. |

```
/openral/safe_action ──(silence > deadline, window open)──▶ deadman_watchdog_node ─┐
GPIO relay / USB pendant ──(rising edge, or read failure)─▶ hardware_estop_node ────┴─▶ /openral/estop (+ /openral/failure/safety)
```

`/openral/estop` never auto-clears — `/openral/estop_reset` (`std_srvs/Trigger`,
served by `openral_safety`'s supervisor node and by the C++ kernel) is the only
recovery path (CLAUDE.md §1.5, §3).

## Bringup

Both nodes are spawned by `packages/openral_rskill_ros/launch/deploy_e2e.launch.py`,
the single launch behind `openral deploy sim` and `openral deploy run`, on **both**
`hal_mode:=sim` and `hal_mode:=real`. Node names on the graph:
`/openral_deadman_watchdog`, `/openral_hardware_estop`.

The deadman is driven to ACTIVE by `tools/lifecycle_autostart.py --required`,
and **the graph shuts down if it does not get there.** The poll-based path is
the same one the safety kernel uses, because a dropped `ACTIVATE` would leave a
brake source silently sitting in INACTIVE; `--required` additionally turns "the
node was never there at all" (an overlay built without this package) into a
non-zero exit instead of an informational one, and an `OnProcessExit` handler
turns that into a refusal to run. A deploy that cannot arm its only independent
watchdog must not come up unprotected.

The pendant is only autostarted when `hardware_estop_device` is non-empty. With
no device declared its configure failure is *expected*, and emitting that
failure on every deploy would train an operator to ignore the one log line that
also means "the deadman failed to activate".

## Deadman: the execution window

`/openral/safe_action` is **bursty**, not a continuous stream — the runner only
publishes chunks while an `ExecuteRskill` goal is running, and an idle deploy
publishes none at all. A watchdog armed at activation would therefore E-stop
every boot and latch, so the node has an explicit gate:

| Parameter | Default | Meaning |
|---|---|---|
| `safe_action_deadline_s` | `0.2` | Maximum age of the newest `/openral/safe_action` before estop fires. `deploy_e2e.launch.py` overrides this to **8.0 s (sim) / 5.0 s (real)** — one VLA inference separates two consecutive chunks in a real rollout, so 0.2 s would fire mid-rollout. That override reuses the horizon the actuation path already applies to the same stream (`action_applied_timeout_s`); it is a deliberately conservative first value, and tightening it is a safety-WG decision backed by measured per-policy chunk gaps. |
| `check_period_s` | `0.05` | Internal deadline-check timer period. |
| `arm_status_topic` | `""` | Empty = **free-running**: the deadline is armed by `on_activate` and any silence past it fires. Non-empty names an `action_msgs/GoalStatusArray` topic — the deadline is evaluated only while some goal is ACCEPTED / EXECUTING / CANCELING. `deploy_e2e.launch.py` sets `/openral/execute_rskill/_action/status`, the runner's own action-status topic. |
| `first_chunk_deadline_s` | `60.0` | Bound on goal-accepted → first chunk, covering a cold policy load. `deploy_e2e.launch.py` sets 120 s. Gated mode only. |
| `safety_status_topic` | `/openral/safety_status` | Latched `SafetyStatus` whose `latched=False` releases this node's post-estop latch. Empty makes the node one-shot per activation. |
| `robot_name` | `"robot"` | Tag carried on the `FailureTrigger` evidence. |

A runner killed mid-goal never publishes a terminal goal status, so the window
stays open with the chunk stream dead — which is exactly the case that fires. An
idle deploy, and the seconds a policy spends loading before its first chunk, do
not.

**Why the action-status topic and not an advisory task string.** Three
properties matter, and only the action server's own status topic has all three:
a dead runner cannot publish a terminal status (so it cannot close its own
window and hide its death); there is exactly one writer (an advisory
`/openral/reward/active_task` is published by *both* the runner and the
reasoner, and the reasoner's dispatch watchdog clears it precisely when the
runner dies after accepting a goal — which would silence this node for its
headline case); and it is `TRANSIENT_LOCAL`, so a watchdog that joins mid-goal
inherits the current status instead of waiting for the next transition.

**Every suppressing guard is bounded.** A window that never produces a first
chunk fires at `first_chunk_deadline_s`, rather than waiting forever for a
chunk that a dead runner will never send.

The node also **subscribes** `/openral/estop`: an estop from any other source
latches its internal `_triggered` flag so it never storms the topic behind the
kernel. That latch is released when `/openral/safety_status` reports
`latched=False` — i.e. when the operator's `/openral/estop_reset` lands on the
kernel. Without that release the node would be one-shot: the first estop of a
deploy, from any source including the dashboard stop button, would silence it
for the life of the process, leaving the graph with exactly the hole this node
exists to close and no way to notice.

## Hardware pendant

`hardware_estop_node` is a bridge, not a driver: `HardwareEstopNode._read_pressed`
is the per-vendor hook, and **no vendor subclass ships in this repository**. The
base implementation would answer "not pressed" forever, which reads exactly like
a healthy pendant, so `on_configure` refuses to leave UNCONFIGURED unless a real
read source exists:

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `""` | Path of the GPIO chip / HID node. Empty declares *no hardware E-stop on this host*. |
| `poll_rate_hz` | `100.0` | Device poll rate. |
| `active_low` | `true` | Contact polarity, for vendor subclasses; unused by the base class. |
| `channel_label` | `"hardware_pendant"` | Tag on the emitted `HumanEvidence`. |

A read that *raises* is not a "not pressed" answer either. An unplugged pendant
or a driver fault means the state is unknown, and for a brake source unknown
fails closed: `_poll` catches the error, fires the estop once with
`HumanEvidence(reason="read_failed")`, and latches so a persistently broken
driver cannot storm the topic. Letting the exception escape would kill the timer
and take this E-stop source off the graph with no estop and no diagnostic.

`on_configure` returns `FAILURE` — leaving the node visibly not-ready on the
graph, with the reason logged at ERROR — when `device` is empty, when the
declared path is absent on the host, or when a path is present but no driver
(subclass override or injected `read_pressed_hook`) can read it. The deploy
graph passes `hardware_estop_device` (default empty), so on a host with no
pendant every deploy logs `safety.hardware_estop_unavailable` and the node stays
unconfigured. That refusal cannot take the graph down and does not affect the
other E-stop sources.

## Tests

* `test/test_deadman_watchdog.py` — real lifecycle node + real `openral_msgs`
  IDL: fires on silence, stays quiet under a live chunk stream and while a
  policy is still loading, suppresses itself behind an external estop, bounds
  the arm→first-chunk wait, and re-arms when `SafetyStatus` clears (but not
  while it is still latched).
* `test/test_hardware_estop_node.py` — the not-ready refusals, the rising-edge
  publish through an injected device state source, and the fail-closed brake on
  a read that raises.
* `tests/integration/test_estop_watchdog_graph_live.py` — the whole point: a
  real multi-process graph where the in-band safety node is killed and
  `/openral/estop` still fires from this package.

See also [`openral_human_estop`](../openral_human_estop/) (the human-channel
forwarder) and `packages/openral_safety` (the in-band envelope/collision node).
