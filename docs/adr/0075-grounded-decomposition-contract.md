# ADR-0075 — Grounded task decomposition: the smallest actionable unit is a verb + a specific perceived object

- **Status:** Accepted 2026-06-27 (implemented — schema + shared predicate + node + tests).
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
  - [ADR-0022](0022-rskill-action-vocabulary.md) — the per-skill tool
    palette (`RSkillToolEntry`: description + actions + objects + scenes). This ADR
    does **not** change how a skill is *selected*; it changes how the *task* the
    skill is dispatched on is grounded.
  - [ADR-0026](0026-rskill-structured-goal-parameters.md) — `goal_params_schema`, the
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
carries an `object_ref` naming exactly one concrete object or place — for a
manipulation step an entity the reasoner can point to in perception
(`scene_objects`), for a navigation / open / inspect step a named concrete place
or object — never a quantifier or bare generic plural.

### Schema (Layer 4, `openral_core`) — as implemented

Replace the free-text `subtasks: list[str]` on `DecomposeMissionTool` with a
structured, validated `GroundedSubtask`:

```python
class GroundedSubtask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    object_ref: str = Field(min_length=1)  # the ONE concrete object/place this acts on
    text: str = Field(min_length=1)        # the instruction handed to the skill (VLA prompt)

    @model_validator(mode="after")
    def _check_grounded(self) -> "GroundedSubtask":
        # object_ref and text may not be collective (is_collective_target), and
        # text must NAME object_ref so the grounding is explicit.
        ...
    def render(self) -> str: return self.text.strip()

class DecomposeMissionTool(_ReasonerToolBase):
    tool: Literal["decompose_mission"] = "decompose_mission"
    subtasks: list[GroundedSubtask] = Field(min_length=1)
    target_task_id: str = ""
    def rendered_subtasks(self) -> list[str]: ...  # [s.render() for s in subtasks]
```

**No `verb` enum, no `dest_ref`** (the drafted shape). A verb enum couples the
schema to the open-ended `RSkillAction` vocabulary and forces grammar templating
for marginal gain; the destination lives naturally in `text`. The enforcement
that matters is `object_ref`: the LLM must commit to **one** concrete object per
subtask, so covering a set requires N subtasks each with a distinct `object_ref`.

**No on-disk migrator / `schema_version` bump.** `DecomposeMissionTool` is an LLM
*wire* contract (ephemeral structured output consumed live), never persisted to
disk — so the §1.6 on-disk migration path does not apply. The provider
regenerates the tool's `input_schema` from `model_json_schema()`, which now
nests `GroundedSubtask` (required `object_ref` + `text`); the change is to the
live tool surface only.

Grounding enforcement, structural (always on): the `@model_validator` rejects an
`object_ref` or `text` that is a quantifier / bare generic plural — via
`is_collective_target`, the **single shared predicate** in `openral_core` that the
runtime execute gate also uses — and requires `text` to name `object_ref`. The
provider's structured-output path re-prompts on the `ValidationError`, the *same*
mechanism that re-prompts a malformed `ReasonerToolCall`. This alone defeats
"first batch of objects": it is no longer a representable value.

`object_ref` is deliberately a **name**, not a detector instance id: detector ids
churn frame-to-frame and across embodiments, whereas a name ("the milk") is
stable, human-legible on the dashboard/trace, and is exactly what the VLA prompt
needs. The set of currently-seen names is supplied to the LLM through the
enumeration-invite prompt + the `scene_objects` context, not baked into the tool
schema (the scene is dynamic).

A **perceptual** check (reject an `object_ref` resolving to zero scene entities)
is a natural follow-up but is deferred: the structural check already makes the
headline failure un-representable, and a soft perceptual re-prompt needs the node
to thread live `scene_objects` into `_dispatch_decompose_mission`.

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
- **Negative / cost.** The LLM-wire tool surface changes (provider regenerates
  the `input_schema`), so the `decompose_mission` round-trip test moves to the
  structured shape. A weak model now occasionally *fails validation* (collective
  `object_ref` / `text` not naming its `object_ref`) and gets re-prompted instead
  of silently storing a bad subtask — strictly better, but it converts some bad
  outputs into a re-prompt round-trip latency. No on-disk migrator (the tool is
  not persisted; see Schema).
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

1. **Prose + runtime-gate layer (commit 80f362d):** the §"smallest actionable
   unit" block in `DEFAULT_SYSTEM_PROMPT`, the `decompose-mission` playbook trigger
   for collective targets, and the runtime grounding gate. A live 10-case x 3-run
   grounding matrix on glm-5.2 (`.goals/.../eval_reasoner_grounding.py`) passed
   8/10; the cracks (stochastic non-decompose, ambiguous singulars) motivated this
   ADR.
2. **This ADR (implemented):** the shared `is_collective_target` predicate +
   `GroundedSubtask` schema + structural validator + `DecomposeMissionTool.subtasks`
   retyped + node `rendered_subtasks()` + the enumeration-invite/system-prompt copy
   for the structured shape + `docs/methods` + repo-state-map `SCHEMAS`. Verified
   offline: unit tests pin the rejection of a collective `object_ref` (the
   "first batch of objects" failure is now a `ValidationError`), and the
   provider-facing tool schema requires nested `object_ref` + `text`. A live LLM
   re-run of the matrix against the new schema is the remaining confirmation
   (pending OpenRouter credit).
3. **Future ADR:** affordance/precondition-effect contracts → grounded
   long-horizon planner; plus the soft perceptual `object_ref` resolution check.
