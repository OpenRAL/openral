---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- any
inference: false
---

# rskill-playbook-find_object

A `kind: playbook` rSkill: a symbolic S2 decision procedure the Reasoner
reads, not a neural policy. It has no weights; the authored
[`PLAYBOOK.md`](./PLAYBOOK.md) is its runtime.

## What this skill does

Locates a named object the request didn't give a pose for. It recalls the object
from spatial memory; on a miss it runs a **bounded commonsense active search**
(rank likely rooms/containers → navigate → open → look) and, if the search
budget is exhausted, escalates to a human. Concrete walkthrough: the water-bottle
example in [`PLAYBOOK.md`](./PLAYBOOK.md).

## How it works

This playbook is content, not code. When installed, the reasoner injects
`PLAYBOOK.md` into its system prompt and follows the SOP, composing tools it
already has (`recall_object`, `resolve_place`, `locate_in_view`,
`execute_rskill`, `memory_search`). It is `role: s2` and is **never** dispatched
through `ExecuteRskill`. Every motion it triggers is an `execute_rskill` → Action
chunk → C++ safety kernel — the playbook holds no actuation authority (CLAUDE.md
§1.1).

### Observation → action contract

None. A playbook emits no `Action` chunks and requires no actuators
(`actuators_required: []`, `chunk_size: 1`). Its "output" is the sequence of
tool calls the reasoner makes while following the SOP, bounded by
`playbook.max_steps`.

## How it was authored / Upstream provenance

N/A: hand-authored, not trained, so no weights and no upstream model.
Provenance is the authoring decision record (linked via `paper_url`). To change
behaviour, edit `PLAYBOOK.md` and bump `version`.

## Supported robots

Embodiment-agnostic — declares the explicit wildcard `embodiment_tags: ["any"]`
(never an empty list). Gated by `capabilities_required`
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

Packaging-only: `tests/unit/test_playbook_rskill_manifest.py` validates the
manifest + SOP. No benchmark number to reproduce;
`tests/unit/test_reasoner_playbooks.py` covers the reasoner's `## PLAYBOOKS`
system-prompt injection.

## Evaluation

N/A — no `eval/*.json`; a playbook produces no benchmarkable policy output.

## License

- **Code / content:** Apache-2.0.
- **Weights:** none.

## See also

- [`PLAYBOOK.md`](./PLAYBOOK.md) — the decision procedure itself.
