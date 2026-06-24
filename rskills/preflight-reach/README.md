# rskill-preflight-reach

A `kind: playbook` rSkill (ADR-0072): a symbolic S2 **decision procedure** the
Reasoner reads, not a neural policy. It carries no weights — the authored
[`PLAYBOOK.md`](./PLAYBOOK.md) *is* its runtime.

## What this skill does

Before dispatching a manipulation skill, it checks the target is within the
robot's reachable workspace. It reads the reasoner's `## ROBOT` self-model
(locomotion, reach, payload), locates the target via spatial memory / detection,
and runs a **reach check**. If the target is reachable it allows the
manipulation; if it is out of reach and a mobile base exists it stages an
approach stand pose and re-checks; if the arm is fixed (or the target is over the
payload limit) it does **not** dispatch and hands off to a human. Concrete
walkthrough: the mug example in [`PLAYBOOK.md`](./PLAYBOOK.md).

## How it works

This playbook is **content, not code**. When installed, the reasoner injects
`PLAYBOOK.md` into its system prompt and follows the SOP, composing tools it
already has (`recall_object`, `resolve_place`, `query_scene`, `execute_rskill`,
`emit_prompt`). It is `role: s2` and is **never** dispatched through
`ExecuteSkill`. Every motion it triggers is an `execute_rskill` → Action chunk →
C++ safety kernel — the playbook holds no actuation authority (CLAUDE.md §1.1).

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
(never an empty list). Gated by `capabilities_required`
(`has_vision: true` — a real `RobotCapabilities` flag): the loader filters it out
on robots without a camera. Navigation / staging are gated at runtime by the
composed tools, not by this playbook's flags.

## Sensors required

None directly. The tools it composes declare their own sensor needs.

## Manifest summary

- `kind: playbook`, `role: s2`, `actions: [plan]`, `chunk_size: 1`.
- `playbook.trigger`: about to dispatch a manipulation skill on a target whose reachability is uncertain.
- `playbook.done_predicate`: the target is confirmed within the robot's reachable workspace (or staged so it is), or the task is handed off.
- `playbook.max_steps`: 8.

## Quick start

```python
from openral_core.schemas import RSkillManifest

m = RSkillManifest.from_yaml("rskills/preflight-reach/rskill.yaml")
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
