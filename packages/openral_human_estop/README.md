# openral_human_estop

ADR-0018 §5 bullet 2 — **human-driven E-stop forwarder**. `forwarder_node`
subscribes to a high-level `/openral/human_estop` request (from the dashboard,
a voice/prompt channel, or any operator UI) and republishes it onto the canonical
`/openral/estop` topic, alongside a `FailureTrigger(KIND_HUMAN, SEVERITY_ABORT,
HumanEvidence)` on `/openral/failure` so the reasoner records a structured
human-abort event.

```
dashboard / operator UI ──/openral/human_estop──▶ forwarder_node ──▶ /openral/estop (+ /openral/failure)
```

It is deliberately separate from the hardware/deadman sources in
[`openral_safety_watchdog`](../openral_safety_watchdog/): a human pressing "stop"
in software and a deadman timeout are different evidence channels, and keeping
the forwarder in its own process means a stuck UI can't wedge the brake path.
`/openral/estop` is latching — recovery is only via `/openral/estop_reset`
(CLAUDE.md §1.5). Brought up automatically by the `openral deploy sim` /
`deploy run` graph.
