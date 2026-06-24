# ADR-0073 — Reasoner success-gating on the reward/critic signal + a sequential task queue

- **Status:** Proposed 2026-06-24.
- **Date:** 2026-06-24
- **ADR number:** `0073`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - [ADR-0018](0018-ros2-reasoner-supervisor.md) — the ROS 2 reasoner + supervisor
    graph and its tick loop / replanning ladder. **This ADR amends it**: it makes
    *task lifecycle* (sequencing + completion) deterministic in `ReasonerCore`
    while leaving *skill selection* LLM-driven.
  - [ADR-0057](0057-robometer-reward-rskill.md) — the Robometer reward monitor and
    `query_task_progress`. This ADR consumes that signal as the completion gate
    instead of leaving it purely advisory.
  - [ADR-0064](0064-critic-score-topic-and-tier-c-producer.md) — the critic
    producer and the Tier-C `/openral/failure/critic` failure. This ADR ties a
    critic stall on the *active* task to a deterministic ladder step.
  - [ADR-0036](0036-osc-action-contracts-deploy-path-gate.md) — the deploy-path
    action gate and its continuous-deploy auto-reset amendment; unchanged, but the
    reason the runner's `result.success` is a poor completion signal for VLAs.
  - [ADR-0072](0072-reasoner-playbooks-and-self-maintained-memory.md) — the
    `kind: "playbook"` S2 decision procedures the reasoner reads into its system
    prompt. **Central to this ADR**: `decompose-mission`, `verify-outcome`, and
    `preflight-reach` already describe the *strategy* (how to split a compound
    goal, how to verify an outcome, how to check reachability). This ADR is the
    deterministic *substrate* those playbooks ride on — it does not re-implement
    them. (The decisive evidence: all six playbooks were injected into the system
    prompt in the failed run, and the reasoner still neither decomposed nor
    advanced — see Context.)
  - [ADR-0064 (vision SLAM)](0064-vision-slam-lidarless-cuvslam-nvblox-monodepth.md)
    — cuVSLAM + nvblox + Depth-Anything-3 metric depth: the map frame, robot pose,
    and 3D object/occupancy geometry that ground a subtask's target and its
    navigation feasibility (powering the `find-object` / `stage-for-manipulation`
    playbooks the mission orchestrates).
  - [ADR-0065](0065-cumotion-cuda-moveit-planning.md) — cuMotion GPU MoveIt
    planning: a fast collision-aware reachability check that makes the
    `preflight-reach` playbook a *pre-dispatch feasibility gate* (stage or hand off
    instead of burning a 60 s VLA attempt on an unreachable target).
  - Investigation that motivates this ADR:
    [`docs/investigations/reasoner-success-gating-and-multitask.md`](../investigations/reasoner-success-gating-and-multitask.md)
    — the full code map (file:line anchors) and the live run that exposed the gap.

## Context

A live `openral deploy sim` run with a two-task goal
(`pick the black bowl … | pick the butter …`) finished **crash-free but with 0/2
tasks accomplished**. Task 1's policy ran its full deadline (reward peaked at
0.729, below the 0.8 success threshold) and the reasoner then went idle; task 2
was never attempted. The investigation traced three structural gaps:

1. **No task queue.** The multi-task string is joined with `" | "`
   (`deploy_sim.py:683`), published verbatim by the prompt router, stored as a
   single `PromptRecord`, and **drained pull-once** after the first tick
   (`core.py:320`). There is no task list, index, or "next task" anywhere — the
   reasoner relies on the LLM to parse and sequence within one context, and
   nothing re-injects task 2.

2. **Success is not reward-gated, and is meaningless for a VLA.** The rSkill
   runner returns `result.success = True` for *any* VLA run that reaches its
   deadline without crashing (`rskill_runner_node.py:661-663`) — a VLA never
   self-terminates, so this is the normal path. The reasoner treats
   `status == 4 and result.success` as `outcome="ok"` (`reasoner_node.py:2521`)
   and **never reads the reward/critic score** on that path.

3. **The critic signal is advisory-only.** A critic stall fires a Tier-C forced
   tick that appends a failure record to the `## FAILURES` buffer
   (`reasoner_node.py:847`); acting on it is left to the LLM. `query_task_progress`
   (ADR-0057) is only invoked when the LLM chooses to.

The shared theme: the reasoner **conflates *skill-executed* with
*task-accomplished*** and offloads both sequencing and success-judgement to an
LLM that is never required to verify and was empirically unreliable on the local
tool-calling model (11× `no tool_calls`). When the LLM does not act, the prompt is
already drained → `heartbeat_idle` → "awaiting instructions".

This is distinct from the *execution-fidelity* problem (the deploy runner's
trimmed loop does not reproduce the benchmark's closed-loop chunk replay, so the
VLA's reward caps below threshold). That problem makes the grasp *fail*; this ADR
is about the reasoner *not noticing* and *not advancing* — it is needed regardless
of how good the policy is.

**The strategy already existed and was not enough.** The failed run logged
`playbooks: injected 6 into the system prompt` — `decompose-mission`,
`verify-outcome`, `preflight-reach`, `find-object`, `stage-for-manipulation`, and
`clarify-ambiguity` (ADR-0072) were all in front of the LLM. They *describe* how
to split a compound goal, verify an outcome, and check reachability. The reasoner
still did none of it. That is the load-bearing evidence for this ADR: a
prompt-only (playbook-only) approach cannot carry task lifecycle when the
tool-calling model is unreliable. The fix is therefore not *more* strategy text —
it is a deterministic lifecycle the strategy plugs into.

## Decision

Introduce **deterministic mission scaffolding in the Reasoner (S2)**. Keep the LLM
for *skill selection*; move *task lifecycle* (sequencing + completion) into
`ReasonerCore`, where it is bookkeeping, not judgement.

The scaffolding is the **substrate the existing ADR-0072 playbooks orchestrate**,
not a replacement for them. The division of labour:

| Concern | Strategy (LLM, ADR-0072 playbook) | Lifecycle (deterministic, this ADR) |
|---|---|---|
| Split a compound goal | `decompose-mission` | `MissionState.tasks` populated; `split_mission` fallback |
| Verify an outcome | `verify-outcome` (`query_scene` + `query_task_progress`) | auto reward gate → `complete_active` / `abandon_active` |
| Check reachability | `preflight-reach` (cuMotion + SLAM/detector) | pre-dispatch gate → stage or `abandon_active`, no wasted run |

The playbooks decide *what is true*; the `MissionState` records it and *advances
the queue* so the decision is not lost on the next pull-once drain or the next
flaky tool-call.

### 1. Typed `MissionState` (sequential task queue)

Carry the scene's `tasks` as a **structured list** end-to-end instead of a joined
`" | "` string (an optional structured field on the prompt path, or a small
`MissionStamped` channel). `ReasonerCore` owns the mission:

```text
MissionState(tasks: list[TaskState], current: int)
TaskState(
  task_id: str,
  text: str,
  status: pending | active | verifying | done | abandoned,
  attempts: int,
  last_rskill_id: str | None,
  last_trace_id: str | None,
  last_verdict: str | None,
)
```

`TaskState.status` is a strict state machine:
`pending → active → verifying → done | abandoned`. At most one task may be
`active` or `verifying` at a time. `abandoned` is terminal: the task is not
silently re-queued.

Each tick renders a compact `## MISSION` block from the live state: completed
tasks as one-line summaries, the active task id/text, attempt count, pending
task count, and the last verification verdict. It injects **only the current
active task** as the goal. When a task completes, the next task becomes active
and bumps `renderer.seq`, breaking `heartbeat_idle` automatically.

This mirrors the useful part of coding-agent harnesses (explicit todo ledger,
one active item, visible status) without importing their file/checkpoint
machinery: the robot needs a mission ledger, not a second project manager.

**Population — deterministic split, with the playbook as the upgrade.** The queue
is filled deterministically by `split_mission` (on `" | "` and `", then"`), which
handles the deploy-CLI join and simple operator phrasing. For a genuinely compound
goal the `decompose-mission` playbook (ADR-0072) is the richer path: the LLM
produces an ordered subtask list, which **populates the same `MissionState.tasks`**
rather than living only in the prompt. Either way the queue is the durable record;
the split is the floor, the playbook the ceiling. This keeps decomposition
*strategy* in the playbook and decomposition *persistence* in the lifecycle.

### 2. Reward-gated completion (deterministic gate; the LLM still drives the skill)

On `execute_rskill` return, the reasoner **does not** mark the task done on
`result.success` alone. When a reward monitor is available
(`task_progress_available`), it auto-issues the existing `query_task_progress`
assess as a verification step with `task=current_task.text` and gates on the
typed service response, not on the natural-language re-prompt path:

For VLA skills, this is the primary completion signal: VLAs are
non-self-terminating policies that run until a deadline / step budget expires,
so `runner.result.success` means "the policy ran without a controller fault,"
not "the task was accomplished."

- `success_now ≥ success_threshold` for a **dwell** of ≥2 consecutive readings →
  mark the current task `done`, advance `current`, inject the next task.
- sub-threshold or `stalled` → keep the task `active`, increment `attempts`, and
  drive the existing replanning ladder (retry → param-tweak → substitute-skill →
  goal-replan → human-handoff), bounded by `attempts` / `retry_cap`.
- ladder exhausted → mark `abandoned`, `emit_prompt` an honest *"could not complete
  task K (reward plateaued at X)"* with the `MissionState` snapshot, then advance
  or pause for human handoff. A human-priority correction may resume the same
  mission at `current` instead of starting a new one.

For wrapped ROS skills with a real terminal done predicate (for example
`ROSRskillGoalSatisfied`), runner success can still complete the task. For VLAs,
deadline success alone can only produce `last_verdict="unverified"`; if no reward
monitor or scene verifier is available, the reasoner emits a specific operator
handoff instead of marking the task done. Degrading gracefully must not mean
reintroducing fake success.

The gate appends the verdict to mission/execution feedback after the state
transition. The existing LLM-selected `query_task_progress` tool may still
publish its advisory re-prompt, but automatic completion gating is typed
node-side bookkeeping.

**The reward gate is the floor; `verify-outcome` is the ceiling.** The automatic
`query_task_progress` gate is the deterministic backstop that always runs. The
`verify-outcome` playbook (ADR-0072) is the richer semantic check — it composes
`query_scene` ("is the bowl actually in the drawer?") with `query_task_progress`
and classifies success/failure with evidence. When the LLM runs `verify-outcome`,
its conclusion drives the **same** `complete_active` / `abandon_active`
transitions as the automatic gate. The numeric reward catches the common case
without the LLM; the VLM check resolves the ambiguous case the reward can't score.
Both write to the one queue, so the verdict is never lost.

### 2b. Pre-dispatch feasibility gate (ground the honest-handoff decision)

Before `record_attempt` on a manipulation task, the reasoner may run the
`preflight-reach` playbook (ADR-0072), which is only *useful* with the new
backends: **cuMotion** (ADR-0065) answers "is this target reachable + collision
free from here?" and **vision SLAM / a detector** (ADR-0064) answers "where is the
target in the map?". If the target is unreachable, the lifecycle stages the robot
(a `stage-for-manipulation` step) or `abandon_active`s the task **without burning a
60 s VLA attempt** — turning the honest-handoff decision from a reward-timeout into
a *grounded* feasibility verdict. This piece is optional and gated on the backends
being present; it does not block Pieces 1–2.

### 3. Critic stall drives the ladder only when task-scoped

A Tier-C critic stall triggers a deterministic ladder step only when the stall is
known to score the active task. The current `CriticScore`/`CriticEvidence`
contract carries `critic_id`, score, threshold, and trace, but **not task text or
task id**, while `reward_monitor_node`'s continuous critic path scores its
startup `task` parameter by default. For multi-task missions, that is not enough
evidence to say "task 2 stalled."

So Piece 3 requires one of these before it increments the active task attempt:

- the reward monitor is retargeted to `current_task.text` for the active attempt;
- or `CriticScore` / `CriticEvidence` carries a `task_id` / task text echo that
  matches `MissionState.current`;
- or the reasoner confirms the stall with a typed
  `query_task_progress(task=current_task.text)` response.

Until then, unscoped critic failures remain advisory context and forced ticks;
they do not deterministically advance the ladder for a named task.

### Scope of the first cut

Pieces 1 + 2 (queue + reward-gated advance with a dwell) are the minimum that fixes
the observed run. Piece 3 is hardening and may land in a follow-up.

### Harness lessons folded in

- **LLM coding harnesses (Claude Code / opencode / Copilot-style agents):** keep a
  visible task ledger, act through typed tools, and mark work done only after a
  check. Applied here as `MissionState` + typed verification; skipped full
  checkpoints/rollback because a robot mission cannot be undone like a git diff.
- **Robot harnesses (OpenMind/OM1-style NL bus, Inner Monologue/SayCan/Reflexion
  lineage, π-style policy runners):** keep the policy executor separate from the
  success judge, feed compact sensor/execution history back to the planner, and
  use explicit affordance/progress gates before advancing. Applied here as
  active-task-scoped reward verification and mission feedback; skipped a second
  planner or WAM because ADR-0073 only needs bookkeeping.

## Alternatives considered

- **Prompt-only fix (improve the system prompt / write better playbooks so the LLM
  verifies + sequences).** Rejected as the *primary* mechanism, with direct
  evidence: the failed run had all six ADR-0072 playbooks — including
  `decompose-mission` and `verify-outcome` — already injected into the system
  prompt, and the local tool-calling model still neither decomposed nor advanced
  (11× `no tool_calls`). Better strategy text cannot carry task lifecycle when the
  model is unreliable. The playbooks are kept and *relied on* for strategy; this
  ADR adds the deterministic lifecycle they were missing.

- **Re-implement decomposition / verification in the lifecycle (ignore the
  playbooks).** Rejected: `decompose-mission` and `verify-outcome` already encode
  the strategy and compose the right tools (`query_scene` + `query_task_progress`).
  Duplicating that as hard-coded logic would fork the behaviour and rot. The
  lifecycle instead *consumes* the playbooks' conclusions (they call
  `complete_active` / `abandon_active` / populate `tasks`) and only owns the
  bookkeeping the LLM cannot be trusted to hold.

- **Gate on the runner's `result.success`.** Rejected: for a VLA this is `True`
  whenever the policy ran its deadline without crashing
  (`rskill_runner_node.py:661`), so it carries no task-completion information.

- **Split tasks at the prompt router / CLI into separate `/openral/prompt`
  messages.** Rejected as insufficient on its own: without a queue the reasoner
  still drains each prompt pull-once and has no notion of "current vs remaining",
  and without reward-gating it would advance on a false success. A structured
  mission channel owned by `ReasonerCore` is the durable form; naive `" | "`
  splitting is at best a partial Piece 1.

- **Subscribe the reasoner directly to `/openral/critic/score` and gate on
  `progress_now`.** Rejected as the completion signal: the critic score is
  `progress_now` (higher-is-better progress), not `success_now`. Completion needs
  `success_now ≥ threshold`, which `query_task_progress` already returns; reusing
  it avoids a second consumer of the same model with a weaker signal.

- **Let unscoped Tier-C critic stalls drive the active task ladder.** Rejected for
  multi-task missions: the current continuous critic stream may still be scoring
  the first startup task. Deterministic retries need task-scoped evidence, not just
  "some critic stalled."

## Consequences

- **Positive:** the reasoner stops conflating execution with accomplishment. It
  verifies, advances through a multi-task mission, retries deterministically on a
  sub-threshold reward, and reports honest failure + human-handoff when a task
  cannot be completed — never falsely "done + idle". Task lifecycle becomes robust
  to a flaky tool-calling LLM.

- **Honest about weak policies:** a VLA that caps below threshold is now driven to
  `abandoned` / human-handoff rather than reported as success. This *surfaces* the
  execution-fidelity gap instead of hiding it; it does not fix it.

- **Safety:** unchanged. This is S2 advisory bookkeeping; actuation still flows
  through the safety kernel. A wrong reward reading can only cause an extra retry
  or an early hand-off, never an unsafe action; the dwell + `attempts` cap +
  human-handoff bound the cost.

- **Trace completeness:** every `execute_rskill` span is stamped with
  `mission.task_id`, `mission.task_index`, `mission.task_text`, and
  `mission.attempts`; the reward gate records `reward.success_now`,
  `reward.dwell_count`, and `reward.gate_result` on the same trace. A post-mortem
  can replay why a task advanced, retried, or handed off.

- **Cost / risk:** auto-issuing `query_task_progress` on every skill return adds
  one reward-model inference per task attempt (the monitor is already resident when
  enabled). The reward model can be wrong; the dwell requirement and `attempts` cap
  bound a false advance or a premature abandon. Task-scoping the query avoids
  scoring task K with task K-1's instruction.

- **Tests:** pure-logic and unit-testable without a GPU — mission advancement,
  gate thresholds with dwell, ladder escalation, VLA-without-verifier handoff, and
  task-scoped critic handling. A deploy-sim integration check confirms task 2 is
  attempted after task 1 reaches a verified or abandoned terminal state, trace
  spans carry mission/reward attributes, and a sub-threshold task is retried then
  handed off rather than silently completed.
