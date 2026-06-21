# ADR-0064 — Generic `CriticScore` topic + Tier-C critic producer

- **Status:** Accepted 2026-06-21. Closes the observability-audit **P1 R3**
  (default producer for the reserved `/openral/failure/critic` Tier-C source).
  The decision-core (`CriticWatchdog` / `CriticWatchdogGroup`) and the generic
  message land first; the producer node + live deploy-sim validation follow in
  the same branch.
- **Date:** 2026-06-21
- **ADR number:** `0064`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - ADR-0018 §F3 + 2026-05-25 amendment — the namespaced `FailureTrigger` bus
    and the failure-tier taxonomy (`safety→A`, `hal/sensor/rskill/wam→B`,
    **`critic→C`**, operator/perception→D). The `critic` source was *reserved*
    there with no producer; this ADR ships one.
  - ADR-0057 — `kind: reward` rSkills (Robometer-4B): the first reward model
    that produces a normalized per-frame progress/success scalar. It is the
    canonical first publisher on the new topic, but **not** privileged.
  - ADR-0047 / ADR-0056 — read-only auxiliary perception (scene VLM, on-demand
    detectors) that runs parallel to the VLA and feeds the Reasoner. The critic
    producer is the same shape, but its output is a *failure event* (drives
    replanning), not a read-only tool answer.

## Context

ADR-0018 reserved `/openral/failure/critic` for Tier-C "a critic flagged the
rollout below bar" events, but nothing produced them: a robot whose task
progress silently *stalls* (the policy keeps emitting action chunks but the
scene stops moving toward the goal) ran to a timeout instead of triggering
replanning. ADR-0057 then added the missing *signal* — Robometer emits a
normalized per-frame progress/success scalar — but routed it only through a
read-only `query_task_progress` **service** the Reasoner polls on demand. There
was still no path from "reward model sees a stall" to "Reasoner preempts".

The reasoner side is already complete: `reasoner_node` subscribes to all six
`/openral/failure/*` sources (including `critic`), maps `critic→"C"`, and forces
`ReasonerCore.tick(force=True, tier="C")` on `severity >= FAIL`. The only gap is
a **producer**: something that turns critic scores into Tier-C failure events.

The open question this ADR answers: how do **multiple, evolving** reward models
— Robometer today, a future SARM (self-assessment reward model), success
classifiers — feed that one producer without each becoming a special case?

## Decision

1. **A generic `openral_msgs/CriticScore` message on a shared topic
   `/openral/critic/score`.** Fields: `header`, `critic_id`, `score` (higher is
   better, critic's native range), `threshold` (the critic's own pass bar),
   `trace_id` (W3C traceparent). Every reward model publishes self-describing
   samples here; they are distinguished by `critic_id`. Onboarding a new reward
   model (SARM, …) is *just another publisher* — no producer-side config, no new
   topic, no IDL change. We chose a new typed message over reusing the
   `query_task_progress` service because a stall is a *stream* the producer must
   watch continuously, not a value polled on demand, and over a hard-coded
   Robometer dependency because the source must stay open (CLAUDE.md §1.4 —
   explicit, swappable).

2. **A `critic_id`-keyed decision core.** `CriticWatchdog` (pure, import-safe)
   decides *when* a single critic has stalled; `CriticWatchdogGroup` keys one
   watchdog per `critic_id` so several critics share the one
   `/openral/failure/critic` source and each fires its own `CriticEvidence`
   independently. Threshold is taken from each critic's own samples — the
   watchdog never assumes Robometer's range.

3. **A critic producer node** subscribes to `/openral/critic/score`, routes each
   sample through the group, and on a stall publishes `FailureTrigger`
   (`KIND_CRITIC`, `SEVERITY_FAIL`, the emitted `CriticEvidence`,
   `trace_id` propagated) via the existing `FailureBusPublisher`. It owns no
   actuation authority (CLAUDE.md §1.1) — it only advises the Reasoner.

The signal is **advisory**: a critic FAIL drives replanning, it never commands
the C++ safety kernel and never touches E-stop or velocity limits. No
`packages/openral_safety/` or kernel code changes; no safety-WG gate.

## Consequences

- **+** The reserved Tier-C source finally has a producer; stalls trigger
  replanning instead of timing out.
- **+** SARM and any future reward model are first-class with zero producer
  changes — they publish `CriticScore` and pick a `critic_id`.
- **+** No `schema_version` migrator needed: `CriticScore` is purely additive
  (a new message, never persisted by an older producer); `openral_msgs` stays at
  its current version.
- **−** One more topic + node in the deploy graph; gated behind an opt-in
  `--enable-critic` launch flag so existing graphs are unaffected.
- **Risk:** a mis-tuned `threshold`/`stall_patience` could over- or under-fire.
  Mitigated by the watchdog's latch (one event per stall, not per frame) and the
  failure bus's token-bucket rate limit; thresholds are per-critic and live in
  the publisher's config, surfaced in traces.
