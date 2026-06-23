# find-object

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0071.

## Trigger
The goal names a physical object (e.g. "bring the water bottle", "where is my
mug?") whose pose is **not** given in the request and is not in current view.

## Preconditions
- A spatial-memory backend is available (`recall_object` / `resolve_place` tools).
- The robot has a mobile base (to reach candidate places) and a gripper (to open
  occluding containers). These are declared in `capabilities_required`.

## Steps
1. **Recall.** Call `recall_object(target)`. If a `status:current` pose is
   returned, skip to **Verify**.
2. **Memory prior.** On a miss, call `memory_search(target)` over the
   `MEMORY.md` Object-Location Log for the last-seen (possibly `stale`) location.
   Use it as the top search prior — a stale sighting still beats a blind sweep.
3. **Rank candidates.** Build a ranked candidate list from (a) the scene-graph
   regions/places and (b) commonsense priors ("a water bottle is usually in the
   kitchen, then the fridge"). Containers whose contents are occluded come first.
4. **Search loop** (bounded by `playbook.max_steps`). For each candidate in rank
   order:
   - `resolve_place(candidate)` → `execute_rskill(NAVIGATE, goal=place)`.
   - If the candidate is an occluding container, `execute_rskill(OPEN, target=container)`.
   - `locate_in_view(target)`. **Stop on a hit.**

## Verify (done predicate)
`locate_in_view` confirms the target **in view at a known pose**. On success,
record it: `memory_write(op=supersede, section="object_locations", target=<object>,
content=<place>)` so the next task recalls it directly.

## Fallbacks
- Budget (`max_steps`) exhausted with no hit → `emit_prompt` to the operator
  ("I can't find the water bottle — where should I look?"). This is the terminal
  human-handoff rung of the replanning ladder.
- **Never** loop past `max_steps`. Every candidate tried and its outcome are on
  the OTel trace, so the search is replayable.

## Safety
This playbook only *decides* and *sequences*. Every motion it triggers is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel
(ADR-0030); a wrong recall yields a bad plan the kernel still vetoes, never a
relaxed check (CLAUDE.md §1.1).
