# Investigation — reasoner success-gating on the critic signal + sequential multi-task handling

- **Date:** 2026-06-24
- **Status:** Investigation + proposed avenue. The decision is formalized in
  [ADR-0073](../adr/0073-reasoner-success-gating-and-task-queue.md).
- **Motivation:** A live `openral deploy sim --config scenes/deploy/libero_pnp.yaml`
  run (franka / LIBERO, two-task `pick the black bowl … | pick the butter …` goal)
  finished **crash-free but with 0/2 tasks accomplished**: task 1 executed but the
  reward plateaued at 0.729 (< 0.8 success), and task 2 was never attempted — the
  reasoner went idle ("I am ready and awaiting your instructions") after one skill.

This document maps *why* that happens against the current code, then defines a
possible avenue to fix it. It is descriptive + proposal only; no behavior changes
land here.

---

## 1. Observed behavior

From the run log (`/tmp/libero-fullrun.log`, 2026-06-24):

1. Startup prompt delivered intact: `pick the black bowl … | pick the butter …`.
2. Reasoner dispatched `execute_rskill smolvla-libero` for task 1.
3. `deadline_s=0 → resolved to 60s`; the VLA ran ~60 s.
4. Reward monitor scored continuously, peaking at **0.729** (success threshold
   0.8); the critic producer fired stall failures at 0.575 and 0.715.
5. Runner returned, reasoner logged `execute_rskill succeeded`, then
   `emit_prompt → "I am ready and awaiting your instructions"`.
6. Reasoner idled; 11× `tick error: response did not contain a tool_calls block`
   from the local OpenAI-compatible LLM.

Net: **1 dispatch, 0 verified successes, task 2 never dispatched.**

---

## 2. Root cause — three confirmed gaps, one shared theme

### Gap A — there is no task queue; the multi-task string is delegated wholesale to the LLM

- `python/cli/src/openral_cli/deploy_sim.py:683` joins `DeployScene.tasks` with
  `" | "` into one operator prompt.
- `packages/openral_prompt_router/openral_prompt_router/prompt_router_node.py`
  publishes that whole string verbatim on `/openral/prompt`; it does **not** split
  on `" | "`.
- `python/reasoner/src/openral_reasoner/context.py` stores it as a single
  `PromptRecord` and renders it as one line in the `## PROMPTS` block, then
  **drains it pull-once** after the first successful tick
  (`python/reasoner/src/openral_reasoner/core.py:320`).
- There is **no** task list, index, queue, "next task", goal stack, or mission
  decomposition anywhere in the reasoner. Sequencing is entirely the LLM's job
  within a single context, and nothing re-injects task 2.

### Gap B — "success" is not reward-gated, and is meaningless for a VLA

- The rSkill runner returns `result.success = True` and `goal_handle.succeed()`
  for **any** VLA run that reaches its deadline without crashing
  (`packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py:661-663`).
  A VLA never self-terminates, so this is the normal path — `result.success`
  means *"the policy ran for its allotted time"*, not *"the task was accomplished"*.
  Only `ROSRskillGoalSatisfied` (raised by wrapped-ROS skills, **not** VLAs) or a
  hard error changes it.
- The reasoner judges completion as `status == 4 and result.success` →
  `outcome="ok"` (`packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py:2521`).
  It **never reads the reward/critic score** on this path. So "ran the policy
  60 s" is recorded as "task done", even at reward 0.73.

### Gap C — the critic signal is advisory-only

- A critic stall fires a Tier-C **forced tick**
  (`reasoner_node.py:847`) that appends a `FailureEventRecord` to the
  `## FAILURES` buffer. Acting on it is left to the LLM; there is no
  deterministic "sub-threshold reward ⇒ retry".
- `query_task_progress` (ADR-0057) exists and returns `progress_now`,
  `success_now`, trends, and a `stalled` flag — but it is only invoked **when the
  LLM chooses to call it** (`reasoner_node.py:1644`), never automatically after a
  skill returns.

### Shared theme

The reasoner **conflates *skill-executed* with *task-accomplished*** and offloads
both *sequencing* and *success-judgement* to an LLM that (a) is never required to
verify against the reward, and (b) was empirically unreliable on the local
tool-calling model (11× `no tool_calls`). When the LLM doesn't act, the prompt is
already drained → `heartbeat_idle` → "awaiting instructions".

---

## 3. Why the design produced exactly this run

```
dispatch task-1 skill
  → VLA runs 60 s (reward peaks 0.729, < 0.8)
  → runner returns result.success=True  (deadline reached, no crash)
  → reasoner records outcome="ok"        (NOT reward-gated)
  → prompt already drained pull-once
  → no event bumps renderer.seq
  → heartbeat_idle → emit_prompt "awaiting instructions"
task-2 was never an entity the system tracked.
```

The critic stalls *did* fire, but only nudged a flaky LLM that emitted no tool
call.

---

## 4. Proposed avenue — deterministic mission scaffolding in the Reasoner (S2)

OpenRAL's reasoner is deliberately LLM-driven (S2 emits typed tool calls). But
**sequencing and success-gating are exactly the parts that should be
deterministic** — they need bookkeeping, not judgement. Keep the LLM for *skill
selection*; move *task lifecycle* into `ReasonerCore`. Three pieces, smallest
first.

### Piece 1 — typed `MissionState` (sequential task queue)

Carry `DeployScene.tasks` as a **structured list** end-to-end instead of a joined
`" | "` string (an optional structured field on the prompt path, or a small
`MissionStamped` channel). `ReasonerCore` holds:

```text
MissionState(tasks: list[TaskState], current: int)
TaskState(text: str, status: pending|active|done|abandoned, attempts: int)
```

Each tick injects **only the current active task** as the goal — not the whole
list. This removes the "remember and sequence N tasks" burden from the LLM. When a
task completes, the next task is injected as a fresh prompt, which **bumps
`renderer.seq` and breaks the idle** automatically.

### Piece 2 — reward-gated completion (deterministic gate, LLM still drives the skill)

On `execute_rskill` return, **do not** trust `result.success`. Instead, auto-issue
the *existing* `query_task_progress` assess (the machinery in
`reasoner_node._dispatch_query_task_progress` already exists — fire it
automatically as a verification step) and gate:

- `success_now ≥ success_threshold` for a small dwell (e.g. 2 consecutive
  readings) → mark the current task `done`, advance `current`, inject the next
  task.
- sub-threshold / `stalled` → keep the task `active`, increment `attempts`, and
  **deterministically drive the existing replanning ladder** (retry → param-tweak
  → substitute-skill → goal-replan → human-handoff), using `retry_cap` as the loop
  guard.
- ladder exhausted → mark `abandoned`, `emit_prompt` an honest *"couldn't complete
  task K (reward plateaued at X)"*, then advance or hand off.

When no reward monitor is present (`task_progress_available=False`), fall back to
today's behavior (runner success). The change degrades gracefully.

### Piece 3 — make the ladder fire on critic stall, not just hope

The Tier-C forced tick already exists; tie a critic-stall on the **active** task to
a deterministic ladder step (bounded by `attempts`/`retry_cap`) rather than relying
on the LLM to read `## FAILURES` and react.

### Smallest viable first cut

Piece 1 + Piece 2 (queue + reward-gated advance with a dwell) fix the observed
run; Piece 3 is hardening.

---

## 5. What this does and does not solve

- **Solves** the silent-idle and never-advances behavior: the system now either
  verifies+advances, retries, or honestly reports failure and hands off — never
  falsely "done + idle".
- **Robust to a flaky local LLM** for *lifecycle* (deterministic), while still
  using it for *skill choice*.
- **Does NOT** make a weak VLA succeed. If the policy caps at reward 0.73, gating
  correctly drives it to `abandoned` / human-handoff instead of false success.
  This **complements** the separate execution-fidelity work (making the deploy
  runner replay chunks at the benchmark's cadence); it does not replace it.
  **Honest failure > fake success.**

---

## 6. Boundaries, safety, and caveats

- **Layer:** all within Reasoning (S2). Consumes the existing critic/reward signal
  and the existing `query_task_progress` tool — no new layer crossing. It does
  amend the reasoner decision contract, hence [ADR-0073](../adr/0073-reasoner-success-gating-and-task-queue.md).
- **Safety unchanged:** this is S2 advisory bookkeeping; actuation still flows
  through the safety kernel. A *wrong* reward reading can only cause an extra retry
  or an early hand-off — never an unsafe action. The human-handoff backstop bounds
  the cost.
- **The reward model can be wrong.** The `attempts` cap + dwell requirement +
  human-handoff bound the cost of a bad reading; success-gating should require a
  short dwell (≥2 consecutive `success_now ≥ thr`) so one noisy frame can't
  trigger a false advance.

---

## 7. Touch points (for the implementing PR)

- `python/reasoner/src/openral_reasoner/core.py` — `MissionState`, the
  reward-gated completion check, the deterministic ladder driver.
- `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py` —
  auto-verify on skill return; mission ingestion; inject next task.
- The prompt / mission channel — carry `tasks: list[str]` structurally rather than
  a joined `" | "` string.
- `python/reasoner/src/openral_reasoner/tool_use.py` — system prompt: tell the LLM
  that completion-gating and task advancement are now automatic.
- Tests (pure-logic, no GPU): mission advancement, gate thresholds with dwell,
  ladder escalation, graceful fallback when no reward monitor is present.

---

## 8. Anchored evidence index

| Claim | Anchor |
|-------|--------|
| Tasks joined with `" | "` | `python/cli/src/openral_cli/deploy_sim.py:683` |
| Router publishes whole string | `packages/openral_prompt_router/openral_prompt_router/prompt_router_node.py` (`_publish_startup_prompt`) |
| Prompt drained pull-once | `python/reasoner/src/openral_reasoner/core.py:320` |
| Runner returns `success=True` at deadline | `packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py:661-663` |
| Success = `status==4 and result.success` (not reward-gated) | `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py:2521` |
| Critic stall → Tier-C forced tick (advisory) | `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py:847` |
| `query_task_progress` only when LLM calls it | `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py:1644` |
| `retry_cap` suppression | `python/reasoner/src/openral_reasoner/core.py:266-283` |
