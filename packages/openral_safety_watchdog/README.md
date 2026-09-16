# openral_safety_watchdog

ROS 2 reasoner + supervisor graph spec §5 — **defense-in-depth E-stop sources**, independent of the in-band
`openral_safety` node and the C++ safety kernel so a crash in either still
triggers a brake event. Two lifecycle nodes:

| Node | ADR | What it does |
|---|---|---|
| `deadman_watchdog_node` | §5 bullet 4 | Fires `/openral/estop` if no `/openral/safe_action` arrives within `safe_action_deadline_s` while an execution window is open (see below). Also emits a `FailureTrigger(KIND_TIMEOUT, SEVERITY_ABORT)` on `/openral/failure/safety` so the reasoner sees a structured `TimeoutEvidence` event, not just a bare estop. |
| `hardware_estop_node` | §5 bullet 3 | Bridges a hardware estop source (GPIO relay via libgpiod, or a USB-HID pendant via `/dev/input`) onto `/openral/estop`, polling at `poll_rate_hz` (default 100 Hz) and publishing `std_msgs/Empty` + `FailureTrigger(KIND_HUMAN, SEVERITY_ABORT, HumanEvidence)` on the rising edge. The per-vendor device read is an overridable hook; the base class owns the poll loop, edge detection, and publication. **This repo ships no such vendor driver** — see "Hardware pendant" below. |

```
/openral/safe_action ──(silence > deadline, window open)──▶ deadman_watchdog_node ─┐
GPIO relay / USB pendant ──────(rising edge)─────────────▶ hardware_estop_node ────┴─▶ /openral/estop (+ /openral/failure/safety)
```

`/openral/estop` never auto-clears — `/openral/estop_reset` (`std_srvs/Trigger`,
served by `openral_safety`'s supervisor node and by the C++ kernel) is the only
recovery path (CLAUDE.md §1.5, §3).

## Bringup

Both nodes are spawned by `packages/openral_rskill_ros/launch/deploy_e2e.launch.py`,
the single launch behind `openral deploy sim` and `openral deploy run`, on **both**
`hal_mode:=sim` and `hal_mode:=real`. They are driven to ACTIVE by
`tools/lifecycle_autostart.py` — the same poll-based path as the safety kernel,
because a dropped `ACTIVATE` would leave an E-stop source silently sitting in
INACTIVE. Node names on the graph: `/openral_deadman_watchdog`,
`/openral_hardware_estop`.

## Deadman: the execution window

`/openral/safe_action` is **bursty**, not a continuous stream — the runner only
publishes chunks while an `ExecuteRskill` goal is running, and an idle deploy
publishes none at all. A watchdog armed at activation would therefore E-stop
every boot and latch, so the node has an explicit gate:

| Parameter | Default | Meaning |
|---|---|---|
| `safe_action_deadline_s` | `0.2` | Maximum age of the newest `/openral/safe_action` before estop fires. `deploy_e2e.launch.py` overrides this to **8.0 s (sim) / 5.0 s (real)** — one VLA inference separates two consecutive chunks in a real rollout, so 0.2 s would fire mid-rollout. That override reuses the horizon the actuation path already applies to the same stream (`action_applied_timeout_s`); it is a deliberately conservative first value, and tightening it is a safety-WG decision backed by measured per-policy chunk gaps. |
| `check_period_s` | `0.05` | Internal deadline-check timer period. |
| `arm_topic` | `""` | Empty = **free-running**: the deadline is armed by `on_activate` and any silence past it fires. Non-empty names a `std_msgs/String` topic carrying the execution window — the deadline is only evaluated while the payload is non-empty **and** at least one `/openral/safe_action` has arrived inside that window. `deploy_e2e.launch.py` sets `/openral/reward/active_task`, which the runner publishes with the instruction on goal accept and `""` on every goal exit. |
| `robot_name` | `"robot"` | Tag carried on the `FailureTrigger` evidence. |

A runner killed mid-goal never publishes the closing `""`, so the window stays
open with the chunk stream dead — which is exactly the case that fires. An idle
deploy, and the seconds a VLA spends loading before its first chunk, do not.

The node also **subscribes** `/openral/estop`: an estop from any other source
latches its internal `_triggered` flag so it never storms the topic behind the
kernel.

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
  IDL: fires on silence, stays quiet under a live chunk stream, and suppresses
  itself behind an external estop.
* `test/test_hardware_estop_node.py` — the not-ready refusals and the
  rising-edge publish through an injected device state source.
* `tests/integration/test_estop_watchdog_graph_live.py` — the whole point: a
  real multi-process graph where the in-band safety node is killed and
  `/openral/estop` still fires from this package.

See also [`openral_human_estop`](../openral_human_estop/) (the human-channel
forwarder) and `packages/openral_safety` (the in-band envelope/collision node).
