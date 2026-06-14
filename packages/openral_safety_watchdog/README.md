# openral_safety_watchdog

ADR-0018 §5 — **defense-in-depth E-stop sources**, independent of the in-band
`openral_safety` node and the C++ safety kernel so a crash in either still
triggers a brake event. Two lifecycle nodes:

| Node | ADR | What it does |
|---|---|---|
| `deadman_watchdog_node` | §5 bullet 4 | Fires `/openral/estop` if no `/openral/safe_action` arrives within `safe_action_deadline_s` (default 0.2 s ≈ 6 chunks at 30 Hz). Also emits a `FailureTrigger(KIND_TIMEOUT, SEVERITY_ABORT)` on `/openral/failure/safety` so the reasoner sees a structured `TimeoutEvidence` event, not just a bare estop. |
| `hardware_estop_node` | §5 bullet 3 | Bridges a hardware estop source (GPIO relay via libgpiod, or a USB-HID pendant via `/dev/input`) onto `/openral/estop`, polling at `poll_rate_hz` (default 100 Hz) and publishing `std_msgs/Empty` + `FailureTrigger(KIND_HUMAN, SEVERITY_ABORT, HumanEvidence)` on the rising edge. The per-vendor device read is an overridable hook; the base class owns the poll loop, edge detection, and publication. |

```
/openral/safe_action ──(silence > deadline)──▶ deadman_watchdog_node ─┐
GPIO relay / USB pendant ───(rising edge)───▶ hardware_estop_node ────┴─▶ /openral/estop (+ /openral/failure/safety)
```

`/openral/estop` never auto-clears — `/openral/estop_reset` (`std_srvs/Trigger`)
is the only recovery path (CLAUDE.md §1.5, §3). Brought up automatically by the
`openral deploy sim` / `deploy run` graph.

See also [`openral_human_estop`](../openral_human_estop/) (the human-channel
forwarder) and `packages/openral_safety` (the in-band envelope/collision node).
