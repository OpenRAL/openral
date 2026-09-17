# openral_human_estop

ROS 2 reasoner + supervisor graph spec §5 bullet 2 — **human-driven E-stop forwarder**. `forwarder_node`
subscribes to a high-level `/openral/human_estop` request (from the dashboard,
a voice/prompt channel, or any operator UI) and republishes it onto the canonical
`/openral/estop` topic, alongside a `FailureTrigger(KIND_HUMAN, SEVERITY_ABORT,
HumanEvidence)` on `/openral/failure/safety` so the reasoner records a structured
human-abort event.

```
operator UI / voice / Slack adapter ──/openral/human_estop──▶ forwarder_node ──▶ /openral/estop (+ /openral/failure/safety)
```

It is deliberately separate from the hardware/deadman sources in
[`openral_safety_watchdog`](../openral_safety_watchdog/): a human pressing "stop"
in software and a deadman timeout are different evidence channels, and keeping
the forwarder in its own process means a stuck UI can't wedge the brake path.
`/openral/estop` is latching — recovery is only via `/openral/estop_reset`
(`std_srvs/Trigger`, served by `openral_safety`'s supervisor node and by the C++
kernel; CLAUDE.md §1.5).

## Bringup

Spawned by `packages/openral_rskill_ros/launch/deploy_e2e.launch.py` — the single
launch behind `openral deploy sim` and `openral deploy run` — on **both**
`hal_mode:=sim` and `hal_mode:=real`, as `/openral_human_estop_forwarder`, and
driven to ACTIVE by `tools/lifecycle_autostart.py` (the poll-based path the
safety kernel uses, so a dropped `ACTIVATE` cannot leave an E-stop source
sitting in INACTIVE).

## Producers

**No node in this repository publishes `/openral/human_estop` today.** The
dashboard's stop button publishes `/openral/estop` directly, through
`openral_observability.dashboard.estop_publisher`. This forwarder exists for
external operator adapters — a UI, a voice channel, a chat bridge — which are
out of tree; it is brought up with the graph so such an adapter can attach to a
running deploy without a relaunch. Do not read its presence as evidence that a
software stop button is wired.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `channel_label` | `"unknown_human_channel"` | Free-text tag placed on the emitted `HumanEvidence` as its **`actor`** (e.g. `"dashboard"`, `"slack"`, `"voice"`). The deploy graph leaves it at the default, because naming a channel would assert a producer that does not exist; an adapter that owns the topic should set it. |

The emitted `evidence_json` is `{"kind": "human", "actor": <channel_label>, "reason": "human_estop"}` — it validates against `openral_core.HumanEvidence`, whose `actor` is required and which forbids extra keys. (It previously emitted a `channel` key and no `actor`, so every human-channel event failed to parse.)

## Tests

* `test/test_forwarder_node.py` — real lifecycle node + real `openral_msgs` IDL:
  `/openral/human_estop` → `/openral/estop` + `FailureTrigger(KIND_HUMAN)`.
* `tests/integration/test_estop_watchdog_graph_live.py` — the forwarder in a
  real multi-process deploy-shaped graph, alongside the deadman watchdog.
