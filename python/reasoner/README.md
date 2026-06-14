# openral-reasoner

OpenRAL S2 reasoner — typed LLM client + `Plan` emission.

> **Scaffold status (2026-05-18).** This package ships the **Protocol surface only**:
> `Reasoner`, `LLMClient`, `Plan`, `ToolCall`, and a `NullReasoner` stub
> for plumbing tests. Concrete LLM clients (OpenAI, Anthropic) and the
> deterministic `Plan → BT.CPP v4 XML` emitter land in follow-up PRs
> tracked in [`docs/roadmap/index.md`](../../docs/roadmap/index.md) (Week-4
> Reasoner stub).

## Layer

CLAUDE.md §6.1 Layer 4 — the slow planning loop (5–10 Hz) sitting between
the `WorldStateAggregator` and the S1 skill executor.

## ADRs

- [ADR-0005 — BT.CPP v4 XML + typed LLM tool palette, not LangGraph](../../docs/adr/0005-bt-llm-not-langgraph.md)
- [ADR-0003 — Pydantic v2 over `@dataclass`](../../docs/adr/0003-pydantic-over-dataclasses.md)
- [ADR-0010 — Inference runner](../../docs/adr/0010-inference-runner.md) (downstream consumer of the BT XML)

## Public surface

```python
from openral_reasoner import LLMClient, Reasoner, Plan, ToolCall, NullReasoner
```

- `LLMClient` — wire-level Protocol; one method (`complete_structured`).
- `Reasoner` — planning-layer Protocol; one method (`plan`).
- `Plan` / `ToolCall` — Pydantic v2 structured-output schemas.
- `NullReasoner` — no-LLM stub returning a single-leaf `Plan`.

## Why no real LLM yet

The roadmap calls Reasoner a Week-4 deliverable. The Protocol surface
lands first so:

1. The runner (`openral_runner`) and the future BT executor
   (`packages/openral_skill/bt_runner.cpp`) can be wired against a
   locked signature without waiting on provider integrations.
2. `docs/METHODS.md` can carry an entry for this layer (CLAUDE.md §1.13)
   so contributors looking for a planning seam find one.
3. The first provider PR is reduced to "implement `LLMClient` for X,
   ship a concrete `Reasoner`, add the BT XML emitter" — three
   independent pieces, each reviewable in isolation.
