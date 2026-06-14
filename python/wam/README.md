# openral-wam

OpenRAL World Action Model (WAM) layer — `WorldModel` Protocol +
`Rollout` schema + `NullWorldModel` stub.

> **Scaffold status (2026-05-18).** This package ships the **Protocol surface only**:
> `WorldModel`, `Rollout`, and a `NullWorldModel` stub for plumbing tests.
> Concrete adapters (Cosmos Predict, UnifoLM-WMA-0, IRASim) land in
> v0.3+ per CLAUDE.md §6.3.

## Layer

CLAUDE.md §6.1 Layer 5 — generative simulator used by the planning loop
for mental simulation (gating action chunks), failure anticipation, and
replanning.

## Three integration patterns (CLAUDE.md §6.3)

1. **Mental simulation (gating)** — sample N short rollouts before
   committing an action chunk. Threshold via `Rollout.confidence`.
2. **Failure anticipation** — continuous predicted-vs-observed
   discrepancy detector. The `WorldStateAggregator` and the WAM share a
   trace span; divergence raises a `FailureTrigger`.
3. **Replanning loop** — propose alternative subgoals as visual
   prompts; the reasoner re-prompts with the predicted-but-failed
   trajectory as context.

All three consume the same surface — `WorldModel.rollout(...)` →
`Rollout` — so the Protocol is sufficient for v0.2.

## Public surface

```python
from openral_wam import WorldModel, Rollout, NullWorldModel
```

- `WorldModel` — structural Protocol every WAM adapter satisfies.
- `Rollout` — Pydantic v2 schema for the predicted trajectory.
- `NullWorldModel` — identity stub returning the input state for
  `horizon` steps.

## ADRs

- [ADR-0003 — Pydantic v2 over `@dataclass`](../../docs/adr/0003-pydantic-over-dataclasses.md)
- CLAUDE.md §6.3 — the canonical write-up of the three integration
  patterns; a dedicated ADR will follow when the first concrete
  adapter lands.
