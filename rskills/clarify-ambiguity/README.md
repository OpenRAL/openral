# rskill-clarify-ambiguity

A `kind: playbook` rSkill (ADR-0072): a symbolic S2 **decision procedure** the
Reasoner reads, not a neural policy. It carries no weights — the authored
[`PLAYBOOK.md`](./PLAYBOOK.md) *is* its runtime.

## What this skill does

Resolves an underspecified or ambiguous goal **before** acting. When the request
admits more than one interpretation — two candidate bowls, a missing destination,
an unsafe-to-guess choice — it disambiguates from spatial memory, then from the
scene, and only then asks the operator a concise question; it **never** guesses on
an irreversible action (placing, pouring, opening). Concrete walkthrough: the
two-bowls example in [`PLAYBOOK.md`](./PLAYBOOK.md).

## How it works

This playbook is **content, not code**. When installed, the reasoner injects
`PLAYBOOK.md` into its system prompt and follows the SOP, composing tools it
already has (`query_scene`, `memory_search`, `recall_object`, `emit_prompt`). It
is `role: s2` and is **never** dispatched through `ExecuteSkill`. It actuates
nothing: its job is to gate the downstream `execute_rskill` → Action chunk → C++
safety kernel with a single unambiguous goal — the playbook holds no actuation
authority (CLAUDE.md §1.1).

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
(never an empty list) and an empty `capabilities_required: {}`: resolving an
ambiguous reference is operator interaction plus memory/scene queries, so it
works on **any** robot. The read-only scene/memory queries are gated at runtime by
the composed tools, not by this playbook's flags.

## Sensors required

None directly. The tools it composes declare their own sensor needs.

## Manifest summary

- `kind: playbook`, `role: s2`, `actions: [plan]`, `chunk_size: 1`.
- `playbook.trigger`: the goal is underspecified or ambiguous.
- `playbook.done_predicate`: the goal has a single unambiguous interpretation,
  confirmed from memory/scene or by the operator.
- `playbook.max_steps`: 5.

## Quick start

```python
from openral_core.schemas import RSkillManifest

m = RSkillManifest.from_yaml("rskills/clarify-ambiguity/rskill.yaml")
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
