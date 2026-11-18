# decompose-mission

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here.

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
- `decompose_mission` is in the tool list. It writes the `## MISSION` task
  ledger the node sequences and the dashboard's Mission card shows.
- A scene-query tool (`query_scene`) is available to check each subtask's
  done-condition; a reward monitor (`query_task_progress`) may also be up.

## Steps
1. **Decompose before acting.** Before any `execute_rskill`, call `decompose_mission`
   with an EMPTY `target_task_id` and the ordered subtasks: one per
   skill-sized step, each naming its one object or place (e.g.
   `[{object_ref: "shelf S3", text: "navigate to the dock pose in front of shelf S3"},
   {object_ref: "box", text: "restock box x1 into lower slot 1"}]`). The node
   refuses this whole-queue write once a task has been attempted, so it must
   come before the goal's first `execute_rskill`; read-only looks may come
   first (a collective target is grounded with `locate_in_view` before it is
   split, per the Trigger above). Do NOT keep the plan only in memory or in
   your rationale: a plan outside the ledger gets no per-step verdict,
   retry budget or dashboard row.
2. **Execute the active subtask.** Dispatch the matching skill toward it:
   `execute_rskill(<skill>, prompt=<subtask text>)`. The ledger shows one
   active subtask at a time.
3. **Verify before advancing.** Confirm the subtask's done-condition with
   `query_scene(<done-condition>)` (and `query_task_progress` if a reward
   monitor is up). **Never assume success.** The node advances the ledger to
   the next subtask once the active one is verified.
4. **Per-subtask replan.** On a subtask FAILURE, replan **that subtask only**
   (retry / substitute skill, or `decompose_mission` with `target_task_id` set
   to it to split it finer) — do **not** restart the whole mission. The
   already-verified subtasks stay done.

## Verify (done predicate)
Every subtask in the `## MISSION` ledger is done.

## Fallbacks
- A subtask that exhausts its replan budget → `emit_prompt` to the operator
  describing exactly which subtask blocked and why ("I stacked the bowls but can't
  open the drawer — the handle won't move"). This is the terminal human-handoff
  rung of the replanning ladder. The remaining subtasks stay in the ledger so the
  mission can resume.
- **Never** loop past `max_steps`. Every subtask tried and its outcome are on the
  OTel trace, so the mission is replayable.

## Safety
This playbook only *decides* and *sequences*. Every motion it triggers is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel; a wrong
decomposition yields a bad plan the kernel still vetoes, never a relaxed check
(CLAUDE.md §1.1).
