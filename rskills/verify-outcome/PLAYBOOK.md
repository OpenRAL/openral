# verify-outcome

> **Hand-authored decision procedure (SOP).** Unlike the generated `SKILL.md`
> discovery view, this file is the *content the S2 Reasoner reads and follows*.
> It is injected into the reasoner's system prompt when this playbook is
> installed. The `rskill.yaml` `playbook.body_uri` points here. ADR-0072.

## Trigger
A manipulation or navigation skill just returned and its success is **not**
self-evident: the action result (the policy reported "done") is not the same as
the world state. Examples: a pick-and-place skill returned, but did the bowl
actually land on the plate? A navigate skill returned, but is the robot really at
the goal? Whenever the action result ≠ world state, verify before proceeding.

## Preconditions
- A scene-query backend is available (`query_scene` tool) and, ideally, a reward
  monitor (`query_task_progress`).
- The robot has a camera — declared in `capabilities_required` (`has_vision: true`).

## Steps
1. **Ask a yes/no success question.** Call `query_scene` with a *specific*
   yes/no question about the expected end-state — e.g. "is the black bowl on the
   plate now?". Phrase it around the concrete success condition, not the action.
2. **Cross-check the reward monitor.** If a reward monitor is available, call
   `query_task_progress` for the success probability and the `stalled` flag. Use
   it as an independent second opinion on the scene answer.
3. **Confirm the object's pose** (optional). `locate_in_view(target)` to confirm
   the object is where the success state requires it to be — a scene answer of
   "yes" is stronger when the object is actually located at the expected place.
4. **Classify.** Declare **success only if the evidence agrees**: the scene
   answer is affirmative and (where available) `query_task_progress` is not
   stalled and reports a high success probability. Any disagreement → **FAILURE**.
5. **Record & route.**
   - On **success**, optionally `memory_write` a useful lesson for later tasks.
   - On **FAILURE**, `memory_write(op=add, section="lessons", ...)` what went
     wrong, and signal the caller to replan or hand off. **Never** silently
     report success.

## Verify (done predicate)
A success/failure classification backed by **at least one** scene or reward
observation. A failure must have triggered a replan or human handoff — the
playbook does not return "success" without evidence, and does not swallow a
failure.

## Fallbacks
- Query backends unavailable, or the scene answer and reward monitor
  **contradict** each other → `emit_prompt` asking the operator to confirm the
  outcome ("Did the bowl end up on the plate?"). **Do not assume success.** This
  is the terminal human-handoff rung of the replanning ladder.
- **Never** loop past `max_steps`. Every query and the resulting classification
  are on the OTel trace, so the verification is replayable.

## Safety
This playbook only *decides* and *sequences*. It triggers no motion of its own;
when its classification routes back to a replan, every resulting motion is an
`execute_rskill` → Action chunk that still crosses the C++ safety kernel
(ADR-0030). Reporting a false success would be a truth violation (CLAUDE.md
§1.2) — when in doubt it classifies FAILURE and escalates, never the reverse.
