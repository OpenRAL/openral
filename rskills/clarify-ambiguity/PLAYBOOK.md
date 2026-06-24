# clarify-ambiguity

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0072.

## Trigger
The instruction has more than one valid interpretation — e.g. "put the bowl in
the drawer" when **two** bowls are present, or a request with a **missing**
destination ("put it away" with no place given). Any underspecified or
unsafe-to-guess referent or parameter trips this playbook **before** acting.

## Preconditions
- A spatial-memory backend is available (`memory_search` / `recall_object` tools).
- A scene query is available (`query_scene`) to enumerate candidate referents.
- An operator channel is available (`emit_prompt`) for the disambiguation question.

## Steps
1. **Detect.** Pin down the ambiguity precisely: *which* referent or parameter
   is unresolved (the target object? the destination? a side/colour?), and *how
   many* candidates fit. If exactly one interpretation is valid, there is no
   ambiguity — exit and proceed.
2. **Resolve from memory first.** Call `memory_search(<referent>)` over the
   `MEMORY.md` Preferences log for a stated user preference that disambiguates
   ("the user means the **left** bowl"). A recorded preference beats asking again.
3. **Resolve from the scene.** Call `query_scene` / `recall_object(<referent>)`.
   If exactly **one** candidate actually matches the description in the current
   scene, the ambiguity is resolved — skip to **Verify**.
4. **Ask, don't guess** (only if STILL ambiguous **and** the action is
   irreversible — placing, pouring, opening). `emit_prompt` a short, **specific**
   disambiguation question to the operator ("There are two black bowls — the
   **left** one or the **right** one?") and **WAIT** for the answer. Do not guess.
5. **Record the resolution.** Once resolved, `memory_write(op=add,
   section="preferences", target=<referent>, content=<resolution>)` so the same
   question isn't asked again on the next task.

## Verify (done predicate)
A **single concrete** referent / parameter, sourced from memory, the scene, or
the operator. The downstream skill now has an unambiguous goal to dispatch.

## Fallbacks
- No operator response within the task budget → **hold the action** and
  `emit_prompt` that the task is blocked pending clarification. This is the
  terminal human-handoff rung; **never** proceed on a guess for an irreversible
  step.
- **Never** loop past `max_steps`. Every candidate considered, every memory/scene
  query, and the operator exchange are on the OTel trace, so the resolution is
  replayable.

## Safety
This playbook only *decides* — it resolves the goal and then **gates** the
downstream skill. Refusing to guess on an irreversible action is the safe
default (CLAUDE.md §1.1, §1.4): a wrong referent would yield a wrong placement
the C++ safety kernel cannot un-do, so the playbook asks first rather than
relaxing any check. Every motion it eventually unblocks still crosses the kernel
(ADR-0030).
