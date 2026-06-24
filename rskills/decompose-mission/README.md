# rskill-decompose-mission

A `kind: playbook` rSkill (ADR-0072): a symbolic S2 **decision procedure** the
Reasoner reads, not a neural policy. It carries no weights — the authored
[`PLAYBOOK.md`](./PLAYBOOK.md) *is* its runtime.

## What this skill does

Breaks a compound, multi-step instruction into an ordered list of subtasks, each
with its own verifiable done-condition (an internal TODO list), then executes and
verifies them in order. It decomposes the goal, records the subtasks to memory so
the plan survives a tick, dispatches the matching skill for each, verifies the
done-condition before advancing, and on a subtask failure replans **that subtask
only** — escalating to a human if a subtask exhausts its replan budget. Concrete
walkthrough: the stack-bowls / drawer / cookie-box example in
[`PLAYBOOK.md`](./PLAYBOOK.md).

## How it works

This playbook is **content, not code**. When installed, the reasoner injects
`PLAYBOOK.md` into its system prompt and follows the SOP, composing tools it
already has (`execute_rskill`, `query_scene`, `query_task_progress`,
`memory_write`, `memory_search`, `emit_prompt`). It is `role: s2` and is **never**
dispatched through `ExecuteSkill`. Every motion it triggers is an `execute_rskill`
→ Action chunk → C++ safety kernel — the playbook holds no actuation authority
(CLAUDE.md §1.1).

### Observation → action contract

None. A playbook emits no `Action` chunks and requires no actuators
(`actuators_required: []`, `chunk_size: 1`). Its "output" is the sequence of
tool calls the reasoner makes while following the SOP, bounded by
`playbook.max_steps`.

## How it was authored / Upstream provenance

N/A — a playbook is **hand-authored**, not trained: it has no weights and no
upstream model. Its provenance is [ADR-0072](../../docs/adr/0072-reasoner-playbooks-and-self-maintained-memory.md)
(also linked via `paper_url`). To change behaviour, edit `PLAYBOOK.md` and bump
`version`.

## Supported robots

Embodiment-agnostic — declares the explicit wildcard `embodiment_tags: ["any"]`
(never an empty list). Pure planning / orchestration, so `capabilities_required`
is empty (`{}`): it works on any robot. Each subtask it dispatches is gated at
runtime by the composed tools, not by this playbook's flags.

## Sensors required

None directly. The tools it composes declare their own sensor needs.

## Manifest summary

- `kind: playbook`, `role: s2`, `actions: [plan]`, `chunk_size: 1`.
- `playbook.trigger`: the goal is a compound, multi-step instruction.
- `playbook.done_predicate`: every subtask's verifiable goal has been confirmed met, or the mission has been handed off.
- `playbook.max_steps`: 24.

## Quick start

```python
from openral_core.schemas import RSkillManifest

m = RSkillManifest.from_yaml("rskills/decompose-mission/rskill.yaml")
assert m.kind == "playbook" and m.playbook is not None
print(m.playbook.trigger)
```

## Reproduction

Packaging-only: the manifest + SOP are validated by
`tests/unit/test_playbook_rskill_manifest.py`. There is no benchmark number to
reproduce; the playbook's behaviour is exercised by the reasoner integration
tests in later ADR-0072 phases.

## Evaluation

N/A — no `eval/*.json`; a playbook produces no benchmarkable policy output.

## License

- **Code / content:** Apache-2.0.
- **Weights:** none.

## See also

- [ADR-0072](../../docs/adr/0072-reasoner-playbooks-and-self-maintained-memory.md) — the `playbook` kind + reasoner memory.
- [`PLAYBOOK.md`](./PLAYBOOK.md) — the decision procedure itself.
