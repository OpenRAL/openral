# rskill-find-object

A `kind: playbook` rSkill (ADR-0071): a symbolic S2 **decision procedure** the
Reasoner reads, not a neural policy. It carries no weights — the authored
[`PLAYBOOK.md`](./PLAYBOOK.md) *is* its runtime.

## What this skill does

Locates a named object the request didn't give a pose for. It recalls the object
from spatial memory; on a miss it runs a **bounded commonsense active search**
(rank likely rooms/containers → navigate → open → look) and, if the search
budget is exhausted, escalates to a human. Concrete walkthrough: the water-bottle
example in [`PLAYBOOK.md`](./PLAYBOOK.md).

## How it works

This playbook is **content, not code**. When installed, the reasoner injects
`PLAYBOOK.md` into its system prompt and follows the SOP, composing tools it
already has (`recall_object`, `resolve_place`, `locate_in_view`,
`execute_rskill`, `memory_search`). It is `role: s2` and is **never** dispatched
through `ExecuteSkill`. Every motion it triggers is an `execute_rskill` → Action
chunk → C++ safety kernel — the playbook holds no actuation authority (CLAUDE.md
§1.1).

### Observation → action contract

None. A playbook emits no `Action` chunks and requires no actuators
(`actuators_required: []`, `chunk_size: 1`). Its "output" is the sequence of
tool calls the reasoner makes while following the SOP, bounded by
`playbook.max_steps`.

## How it was authored / Upstream provenance

N/A — a playbook is **hand-authored**, not trained: it has no weights and no
upstream model. Its provenance is [ADR-0071](../../docs/adr/0071-reasoner-playbooks-and-self-maintained-memory.md)
(also linked via `paper_url`). To change behaviour, edit `PLAYBOOK.md` and bump
`version`.

## Supported robots

Embodiment-agnostic (`embodiment_tags: []`). Gated by `capabilities_required`
(`has_vision: true` — a real `RobotCapabilities` flag): the loader filters it out
on robots without a camera. Navigation / container-opening are gated at runtime by
the composed tools, not by this playbook's flags.

## Sensors required

None directly. The tools it composes declare their own sensor needs.

## Manifest summary

- `kind: playbook`, `role: s2`, `actions: [plan]`, `chunk_size: 1`.
- `playbook.trigger`: the goal names an object whose location is not given.
- `playbook.done_predicate`: the target object is confirmed in view at a known pose.
- `playbook.max_steps`: 12.

## Quick start

```python
from openral_core.schemas import RSkillManifest

m = RSkillManifest.from_yaml("rskills/find-object/rskill.yaml")
assert m.kind == "playbook" and m.playbook is not None
print(m.playbook.trigger)
```

## Reproduction

Packaging-only: the manifest + SOP are validated by
`tests/unit/test_playbook_rskill_manifest.py`. There is no benchmark number to
reproduce; the playbook's behaviour is exercised by the reasoner integration
tests in later ADR-0071 phases.

## Evaluation

N/A — no `eval/*.json`; a playbook produces no benchmarkable policy output.

## License

- **Code / content:** Apache-2.0.
- **Weights:** none.

## See also

- [ADR-0071](../../docs/adr/0071-reasoner-playbooks-and-self-maintained-memory.md) — the `playbook` kind + reasoner memory.
- [`PLAYBOOK.md`](./PLAYBOOK.md) — the decision procedure itself.
