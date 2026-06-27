# decompose-mission

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0072.

## Trigger
Either of:
- **Compound goal** — a multi-step instruction: multiple verbs, a "then"/"after"
  sequence, or several sub-goals (e.g. "stack the bowls and put them in the drawer,
  then put the plate on the cookie box").
- **Collective / quantified target** — the goal names a *set* rather than one
  specific object: a quantifier (all / every / each / both / everything) or a bare
  generic plural ("put **the objects** in the basket", "clear **the items** off
  the table"). A skill acts on exactly ONE specific object, so a collective target
  is never directly actionable — it must be enumerated from the live `scene_objects`
  perception list and split into one subtask per concrete object BEFORE any
  `execute_rskill`. Each subtask names a single specific object (verb + that object
  + destination), never "the first batch of objects" or "the remaining items".

## Preconditions
- A self-maintained memory backend is available (`memory_write` / `memory_search`
  tools) so the decomposed plan survives a reasoner tick.
- A scene-query tool (`query_scene`) is available to check each subtask's
  done-condition; a reward monitor (`query_task_progress`) may also be up.

## Steps
1. **Decompose.** Break the goal into an ordered list of subtasks, each a
   `(action, verifiable done-condition)` pair — e.g. `["bowls stacked" via
   query_scene, "bowls in drawer", "plate on cookie box"]`. Each done-condition
   must be something `query_scene` (or `query_task_progress`) can confirm.
2. **Record.** Write the ordered list to memory as open tasks
   (`memory_write(op=add, section="open_tasks", ...)`) — an internal `TODO.md` so
   the plan survives a tick and the mission can resume after an interruption.
3. **Execute in order** (bounded by `playbook.max_steps`). For each subtask in
   list order:
   - Dispatch the matching skill toward its action:
     `execute_rskill(<skill>, goal=<subtask.action>)`.
4. **Verify before advancing.** Confirm the subtask's done-condition with
   `query_scene(<subtask.done_condition>)` (and `query_task_progress` if a reward
   monitor is up) **BEFORE** moving on. **Never assume success.**
5. **Mark done.** On a verified subtask, `memory_write(op=supersede,
   section="open_tasks", ...)` to mark it done and advance to the next.
6. **Per-subtask replan.** On a subtask FAILURE, replan **that subtask only**
   (retry / substitute skill) — do **not** restart the whole mission. The
   already-verified subtasks stay done.

## Verify (done predicate)
All subtasks' done-conditions confirmed — the `open_tasks` list is empty.

## Fallbacks
- A subtask that exhausts its replan budget → `emit_prompt` to the operator
  describing exactly which subtask blocked and why ("I stacked the bowls but can't
  open the drawer — the handle won't move"). This is the terminal human-handoff
  rung of the replanning ladder. The remaining `open_tasks` stay recorded so the
  mission can resume.
- **Never** loop past `max_steps`. Every subtask tried and its outcome are on the
  OTel trace, so the mission is replayable.

## Safety
This playbook only *decides* and *sequences*. Every motion it triggers is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel
(ADR-0030); a wrong decomposition yields a bad plan the kernel still vetoes, never
a relaxed check (CLAUDE.md §1.1).
