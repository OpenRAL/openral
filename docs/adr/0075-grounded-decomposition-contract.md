# ADR-0075 — Grounded task decomposition: the smallest actionable unit is a verb + a specific perceived object

- **Status:** Proposed 2026-06-27.
- **Date:** 2026-06-27
- **ADR number:** `0075`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - [ADR-0073](0073-reasoner-success-gating-and-task-queue.md) — the deterministic
    `MissionState` task queue and `decompose_mission` tool (#123). **This ADR
    constrains the *content* of the subtasks that tool writes**: today
    `decompose_mission.subtasks` is `list[str]`, so the LLM can write a vague
    "first batch of objects" that the queue faithfully stores and then dispatches.
  - [ADR-0072](0072-reasoner-playbooks-and-self-maintained-memory.md) — the
    `decompose-mission` playbook. This ADR makes the playbook's "split into the
    smallest actionable unit" a *typed contract* rather than only a prose SOP.
  - [ADR-0074](0074-vlm-adjudicated-completion-and-reward-driven-progress.md) — the
    completion-gate ladder. Grounded subtasks give the verifier a concrete object
    to check ("is the milk in the basket?") instead of an unverifiable set.
  - [ADR-0022](0022-reasoner-tool-palette-per-skill.md) — the per-skill tool
    palette (`RSkillToolEntry`: description + actions + objects + scenes). This ADR
    does **not** change how a skill is *selected*; it changes how the *task* the
    skill is dispatched on is grounded.
  - [ADR-0026](0026-per-skill-goal-params-schema.md) — `goal_params_schema`, the
    per-skill structured-params surface. Distinct from this ADR (see Alternatives):
    that schema describes a skill's *inputs*; this ADR describes the *task's
    object reference*.

## Context

Live `libero_object` deploy-sim runs (2026-06-26/27, `investigate/issue-triage`,
reasoner LLM = OpenRouter `z-ai/glm-5.2`) exposed a decomposition-quality wall
that is **independent of the VLA and independent of the reasoner's plumbing** —
both of which now work end-to-end:

- Operator goal: *"Put all the objects on the table into the basket."*
- The live detector publishes a `scene_objects` line into the reasoner's context
  every tick — milk, ketchup, alphabet soup, each with a 3-D position
  (`python/reasoner/src/openral_reasoner/context.py`). **The concrete object list
  was already in front of the LLM.**
- A runtime *grounding gate* (this branch) refuses `execute_rskill` while the
  active task names a collective/quantified target, and self-prompts the LLM to
  enumerate + decompose. The gate fired 5× and **correctly forced** glm-5.2 to
  call `decompose_mission`.
- glm-5.2 nonetheless decomposed "all the objects" into **5 vague "batch"
  subtasks** — `t1.1 = "Pick up the first batch of objects on the table and place
  …"`. "first batch of objects" is *still collective*; the gate re-refused it; no
  actuation ever happened.

The decisive evidence: **the model had the grounded object list and still emitted
an ungrounded subtask.** Prose in the system prompt requesting "one specific
object per subtask" (ADR-0072 playbook; the §"smallest actionable unit" block
added to `DEFAULT_SYSTEM_PROMPT` on this branch) is *necessary but demonstrably
insufficient* on a weaker model. The runtime gate is a correct backstop but it
only *blocks* a bad decomposition; it cannot *produce* a good one, so on a weak
model the loop is: vague-execute → refuse → vague-decompose → refuse → repeat.

The invariant we actually want is universal across all prompts / scenes / tasks /
robots:

> **The smallest actionable unit is a verb applied to exactly ONE specific,
> perceivable object** (optionally `verb + object + preposition + object` for a
> placement). A task is dispatchable to a skill **iff** its object reference
> resolves to exactly one entity the perception layer currently reports.

"Put all the objects in the basket" is not actionable until the robot *looks*,
*enumerates* (milk, ketchup, soup), and rewrites the goal as one grounded subtask
per object. "Bring me a glass of wine" is not actionable until it becomes
`navigate(kitchen) → open(fridge) → pick(wine bottle) → …`, each step naming a
concrete object. The failure mode is the same in both: a *collective or abstract
noun standing in for a set of concrete entities the agent has not yet bound to
perception*.

CLAUDE.md §1.3: **types are the contract.** The durable fix is to make the
invariant a type the LLM's structured-output path cannot violate, not a sentence
it can ignore.

## Decision

Make `decompose_mission` subtasks **grounded by construction**: each subtask
carries an `object_ref` that must bind to an entity currently in the reasoner's
perception (`scene_objects`) — or be explicitly marked as a non-manipulation step
(navigation / open / inspect) whose target is itself a named concrete place or
object, never a quantifier or bare plural.

### Schema (Layer 4, `openral_core`)

Replace the free-text `subtasks: list[str]` on `DecomposeMissionTool` with a
structured, validated shape (backward-incompatible → `schema_version` bump +
migrator, CLAUDE.md §1.6):

```python
class GroundedSubtask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    verb: SubtaskVerb              # enum: pick_place | navigate | open | inspect | …
    object_ref: str               # a concrete object/place name; NEVER a quantifier
    dest_ref: str | None = None   # placement destination, when the verb needs one
    # rendered to the VLA prompt as "<verb> the <object_ref> [into the <dest_ref>]"

class DecomposeMissionTool(_ReasonerToolBase):
    tool: Literal["decompose_mission"] = "decompose_mission"
    subtasks: list[GroundedSubtask] = Field(min_length=1)
    target_task_id: str = ""
```

Two layers of grounding enforcement, cheapest first:

1. **Structural (always on).** A field-validator rejects an `object_ref` /
   `dest_ref` that is a quantifier or bare generic plural (the same
   `all|every|each|both|everything|objects|items|things` vocabulary the runtime
   gate uses, factored into one shared predicate). The provider's structured-output
   path re-prompts on a `ValidationError` — the *same* mechanism that already
   re-prompts a malformed `ReasonerToolCall`. This alone defeats "first batch of
   objects": it is no longer a representable value.

2. **Perceptual (when scene is known).** When the node applies the tool
   (`_dispatch_decompose_mission`) and a live `scene_objects` set is available, an
   `object_ref` that resolves to **zero** known entities is surfaced back to the
   LLM as "no such object in view — re-enumerate" rather than silently queued.
   (Soft: navigation/exploration legitimately targets not-yet-seen objects, so
   this warns + re-prompts; it does not hard-reject, matching the existing
   seen-but-not-lifted exception in `DEFAULT_SYSTEM_PROMPT`.)

`object_ref` is deliberately a **name**, not a detector instance id: detector ids
churn frame-to-frame and across embodiments, whereas a name ("the milk") is
stable, human-legible on the dashboard/trace, and is exactly what the VLA prompt
needs. The enum over *currently-seen* names is supplied to the LLM through the
enumeration-invite prompt, not baked into the tool schema (the scene is dynamic).

### What stays the same

- **Skill selection** remains LLM-driven over the ADR-0022 palette. Grounding
  constrains the *task*, not the *tool choice*.
- **`MissionState`** is unchanged in shape — `GroundedSubtask` renders to the
  existing `TaskState.text` ("pick up the milk and put it in the basket") so the
  queue, subdivision (ADR-0073/#123), and verification (ADR-0074) are untouched.
- The **runtime grounding gate** stays as defence-in-depth (it also catches
  generic-singular phrasings — "grab the thing" — that the structural validator's
  plural-only vocabulary misses; see Alternatives). Its regex is a `v0` backstop;
  the perceptual check (1-entity resolution) is the general form.

## Consequences

- **Positive.** The "verb + one specific object" invariant becomes
  un-violable at the output type, so even a weak model (glm-5.2) cannot emit "the
  first batch of objects"; it must name milk / ketchup / soup. Generalises to all
  prompts/scenes/robots because grounding is defined against *perception*, not a
  scene-specific word list. Gives the ADR-0074 verifier a concrete done-predicate
  per subtask for free. Dashboard/trace gets legible per-object tasks.
- **Negative / cost.** Backward-incompatible schema change → `schema_version` bump
  + migrator + fixture test. Every reasoner-tool fuzz/round-trip test that builds
  a `DecomposeMissionTool` updates to the structured shape. The verb enum must be
  curated and kept in step with the rSkill action vocabulary (`RSkillAction`).
- **Neutral.** No layer boundary moves; this is entirely within Layer 4
  (Reasoning) + the Layer-4 schema in `openral_core`.

## Alternatives considered

1. **Prompt-only (system prompt + playbook).** *Rejected as sufficient, kept as
   necessary.* The receipt above: the model had the object list and ignored the
   request. Shipped on this branch as the cheap first layer, but it cannot carry
   the invariant alone.
2. **Runtime gate only (regex on collective words).** *Kept as a backstop, not the
   fix.* It blocks bad output but cannot produce good output; on a weak model it
   loops. Its English-quantifier vocabulary also misses generic-singular
   ("grab the thing") and is monolingual. The perceptual 1-entity resolution in
   this ADR is strictly more general.
3. **Enrich the per-skill input schema (`goal_params_schema`, ADR-0026).**
   *Rejected for this problem.* That schema describes what a *skill* consumes; VLAs
   genuinely consume a free-text instruction (`goal_params_schema is None`), so a
   richer input schema cannot force the *task* to be grounded. Different axis.
4. **Full affordance / precondition-effect (PDDL/STRIPS) contract per skill.**
   *Deferred to a future ADR (roadmap).* Declaring each skill's `pre:`/`eff:`
   world predicates would let the reasoner *plan* long-horizon goals (the "glass
   of wine" chain) and derive done-conditions automatically — the "real" version
   of this invariant. It is a much larger Layer-4 change (a symbolic planner over
   the LLM) and is not required to defeat the observed failure. `GroundedSubtask`
   is the ADR-light step that fixes the immediate wall and is a clean substrate for
   the affordance work later (an `object_ref` is the bound `?x` a precondition
   would quantify over).

## Rollout

1. **Shipped now (this branch, prose layer):** the §"smallest actionable unit"
   block in `DEFAULT_SYSTEM_PROMPT`, the `decompose-mission` playbook trigger for
   collective targets, and the runtime grounding gate + its unit slice. Verified
   by driving the reasoner directly on the captured `libero_object` decision
   (collective prompt + `scene_objects` = milk/ketchup/soup) — see the reasoner
   grounding test.
2. **This ADR (next PR):** `GroundedSubtask` schema + structural validator +
   perceptual re-prompt + migrator + fuzz/round-trip test updates + `docs/methods`
   + repo-state-map `SCHEMAS`.
3. **Future ADR:** affordance/precondition-effect contracts → grounded
   long-horizon planner.
