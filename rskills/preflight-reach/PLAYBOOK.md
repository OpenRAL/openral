# preflight-reach

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0072.

## Trigger
About to grasp or place an object (e.g. "pick up the mug", "put the can on the
shelf") but it is unclear the embodiment can physically reach the target from
where it currently stands — its reachability from the current base and arm pose
is uncertain.

## Preconditions
- The reasoner's `## ROBOT` self-model block is available (DOF, end-effectors,
  approximate reach, locomotion, payload limit).
- A spatial-memory / detector backend gives the target pose (`recall_object` /
  `query_scene` / `locate_in_view` tools).

## Steps
1. **Read the self-model** (`## ROBOT`). Note locomotion (a fixed arm vs a
   mobile base), the arm's approximate reach, and the payload limit. These three
   facts decide everything that follows.
2. **Locate the target.** Call `recall_object(target)` (or `query_scene` /
   `locate_in_view`) → the target's pose. No pose → defer to the find-object
   playbook before continuing.
3. **Reach check.** Is the target within the arm's workspace from the current
   base pose, *and* within the payload limit? Compare the target pose against the
   self-model's reach envelope and the payload against its limit.
4. **Reachable → done.** If the target is in the workspace and under the payload
   limit, allow the manipulation skill to dispatch.
5. **Out of reach AND a mobile base exists →** `resolve_place` an approach stand
   pose near the target, `execute_rskill(NAVIGATE, goal=stand_pose)`, then
   **re-check** (back to step 3) from the new base pose.
6. **Fixed arm and out of reach (or over payload) →** do **not** dispatch the
   manipulation skill. Hand off (see Fallbacks). A fixed arm cannot stage itself
   closer, so navigating is not an option.

## Verify (done predicate)
The target is inside the reachable workspace — directly, or after staging the
mobile base — and within the payload limit. Only then is the downstream
manipulation skill allowed to dispatch.

## Fallbacks
- Fixed-arm out-of-reach, or target over the payload limit → `emit_prompt` to the
  operator ("the mug is outside my workspace / too heavy; please move it closer
  or reduce the load"). This is the terminal human-handoff rung of the
  replanning ladder.
- Pairs with the stage-for-manipulation playbook: when a mobile base exists, that
  playbook owns the staging motion; this one owns the reach decision.
- **Never** loop past `max_steps`. Every reach check, stage, and re-check is on
  the OTel trace, so the decision is replayable.

## Safety
This playbook only *decides* and *sequences*. Every motion it triggers is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel
(ADR-0030); a wrong reach estimate yields a bad plan the kernel still vetoes,
never a relaxed check (CLAUDE.md §1.1).
