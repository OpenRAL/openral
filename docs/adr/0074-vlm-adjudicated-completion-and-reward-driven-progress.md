# ADR-0074 — VLM-adjudicated task completion + reward-driven progress (ADR-0073 amendment C)

- **Status:** Proposed 2026-06-26.
- **Related:**
  - [ADR-0018](0018-ros2-reasoner-supervisor.md) — the event-driven reasoner, the heartbeat + event-tier preemption, and the bounded replanning ladder this builds on.
  - [ADR-0057](0057-robometer-reward-rskill.md) — the `kind: "reward"` rSkill + `RewardContract` (`success_threshold`, `frame_window_s`); **extended here** with progress-calibration fields.
  - [ADR-0064](0064-critic-score-topic-and-tier-c-producer.md) — the critic producer + the Tier-C `/openral/failure/critic` failure this **extends** from "stall only" to "success | plateau | patience".
  - [ADR-0073](0073-reasoner-success-gating-and-task-queue.md) — the mission queue + success-gating this amendment replaces the hardcoded `0.8`/`deadline_s` of.

## Context

Deploy task completion was gated on the robometer critic crossing a hardcoded `0.8`, and VLA execution ran for a `deadline_s` *time* budget the LLM guessed (we observed `z-ai/glm-5.2` pick `5s` — no progress — and `60s` on consecutive ticks). A clock is the wrong unit: it cannot tell "getting closer" from "stuck", and a single threshold mis-reads a physically-successful place the critic scored `0.78`. The reward stream already publishes `progress` / `success` / `progress_trend` every ~1–2 s — the signal that actually knows.

## Decision

**The reward model owns the calibrated baseline; the LLM iterates on top; the VLM adjudicates the ambiguous middle.** Authority stack:

> **system fallback  <  reward-model calibrated default  <  LLM per-task override**

### 1. The reward model ships its own progress calibration (extends `RewardContract`)
`RewardContract` already carries `success_threshold`; add three siblings, calibrated from the model's own eval distribution (robometer = loose, general; a future movement-specific SARM = tight):
- `check_floor: float ∈ [0,1]` — below this the attempt is clearly not done (no VLM call, straight to the ladder).
- `plateau_window_s: float > 0` — the trailing window over which `progress_trend ≈ 0` means "stopped getting closer".
- `plateau_tolerance: float ≥ 0` — the ε band that absorbs the critic's noise (robometer wanders ~0.55–0.78).
- `default_patience_s: float > 0` — the baseline execution ceiling (a backstop, not the usual stop).

`success_threshold` is reused as the **high-confidence auto-pass** bar.

### 2. The reward-watcher is the wake source (deterministic, fast loop)
Running alongside the VLA, it fires a **wake event** the instant any of these hits, whichever first — re-using the ADR-0064 `critic_producer → /openral/failure/critic → heartbeat-preempt` path, extended beyond stall:
- **success** — `score ≥ success_threshold` (auto-pass; no VLM),
- **plateau** — `|progress_trend| < plateau_tolerance` sustained over `plateau_window_s` (done *or* stuck),
- **patience** — the effective patience ceiling elapsed (backstop).

This replaces `deadline_s` as a goal-level knob; it survives only as the patience ceiling.

### 3. The LLM overrides per-task (optional dispatch fields)
`ExecuteRskillTool` gains optional `patience_s` / `progress_tolerance` overrides (default `None` → use the reward model's calibration). The LLM sets a *task-adaptive* patience (short for a quick grab, long for a precise insertion) and may loosen/tighten ε when a task misbehaves. The effective value (model default ⊕ LLM override) is rendered into context and stamped on the tick span.

### 4. The wake is a normal tick over the FULL palette
The wake event injects the **reward trajectory** (the shape, not the latest scalar) and, in the ambiguous band, a **VLM frame-verdict**, then runs an ordinary reasoner tick. The LLM's response is unconstrained — the entire `ReasonerToolCall` palette + playbooks: complete, retry-with-more-patience, `decompose_mission`, reposition/stage, swap skill, `query_scene`/`locate_in_view`, advance task, write memory, or human-handoff. Bounded by the existing replanning ladder (per-kind caps → human-handoff) so it terminates.

### 5. The three-tier verdict (replaces the hardcoded `0.8`)
At a wake, the node reads the critic score:
1. `score ≥ success_threshold` → `complete_active` **without** a VLM call (trust the calibrated SARM).
2. `check_floor ≤ score < success_threshold` → **`describe_image`** on the current frame ("is `<task>` complete? yes/no") — multimodal reasoner LLM preferred, local scene VLM (`query_scene`) fallback. yes → complete; no → ladder.
3. `score < check_floor` → no VLM; straight to the ladder.

### 6. No-VLM degraded path
With no multimodal LLM and no scene VLM, tier-1 (auto-pass) still completes; the **ambiguous middle** can't be adjudicated → those attempts run to the patience ceiling and escalate to human-handoff. Honest: no oracle + no VLM ⇒ never claim an uncertain success.

## Amendment (2026-06-29) — gate progress, score the whole attempt, both heads to context, per-task locate budget

In libero_object deploy the verdict consumed the reward signal wrong, producing a permanent sub-0.5 "plateau" that never let a task complete. Three coupled fixes (probe-validated against `.goals/libero-multitask-deploy/probe_vla_and_robometer.py` over a cached real-rollout set), plus a node-side loop bound:

1. **Gate the band on the PROGRESS head, not the success head.** Robometer-4B emits two heads: `progress` (closeness) and `success` (done-probability). Measured on cached LIBERO rollouts, **progress reaches 0.80–0.86 on a genuine physical success and ~0.74 on a failure (clean separation), while `success` is compressed to ~0.56–0.79 even on a real success** — so the §5 bars (0.8/0.5, calibrated against progress) made tier-1 auto-pass effectively dead over the success head, and every real success stalled at tier-2/tier-3. `evaluate_task_verdict` now gates on `progress_now`; `success_now` is kept as a secondary corroborating signal surfaced in the verdict text and available to the tier-2 VLM adjudication. The §5 score `s` is the progress head henceforth.
2. **Score the whole attempt, not a trailing 8 s slice.** Robometer (ADR-0057) scores a trajectory from its START; an 8 s trailing window often missed the completion moment and dropped the score into the ladder band (0.3–0.5). `robometer-4b/rskill.yaml` `frame_window_s` is raised `8.0 → 40.0` (the rolling-buffer retention = the attempt horizon, sized past the 30 s `default_patience_s`), and the mission-verify query requests the contract's full `frame_window_s` (the monitor clamps to its retained horizon) so it scores start→now. On the cached set this lifts a real success's scored progress from ~0.70 (trailing, vlm_check/ladder) to ~0.85 (full attempt, **auto-pass**); 6/7 real successes now reach `complete`, the 7th `vlm_check`, and the one real failure stays `vlm_check` (VLM-rejected) — vs 0/7 completing under the old success-head + trailing-window gate.
3. **Render BOTH heads to the LLM, distinctly labelled.** A new `## REWARD` context section (`ContextRenderer.set_reward_state` / `RewardStateRecord`) surfaces `progress=<v> (closeness, trend …)` and `success=<v> (done-confidence, trend …)` so the LLM uses progress for persist-vs-replan and success for done-ness. Fed from each `query_task_progress` / mission-verify response.
4. **Per-task locate budget (node-side loop bound).** The live locate-loop persisted because `locate_in_view` kept HITTING (`found=True`) and the `SearchProgress` bound only counts MISSES + resets on a hit. `TaskLocateBudget` counts locate cycles spent on the active mission task (hit or miss); after `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS=3` cycles without an `execute_rskill` dispatch the subtask is abandoned via the mission ladder with a specific reason (e.g. `could not confirm 'teapot' in view after 3 locate attempts…`) surfaced in the `## EXECUTION` buffer and as the task's `✗` ledger verdict, so the next pick proceeds. Reset on task advance and on a real execute dispatch.

**Camera orientation (verified, no change needed).** The reward monitor subscribes the same raw `/openral/cameras/<cam>/image` the VLA consumes (LIBERO publishes bottom-up; the VLA corrects via `flip_180` internally). The LIBERO backend's `env.render()` returns the *same* `_last_pixels["image"]` array as the published obs frame, so the probe scored byte-identical frames to what the deploy monitor sees — and scored real successes at progress 0.80–0.86. Robometer accepts this orientation; flipping the monitor's input would *depress* scores. Evidence, not a fix.

## Consequences
- **Reward-cancel is the primary stop (§2, implemented).** A reward-watcher wake (`critic` FAIL) while a VLA is in flight cancels the `execute_rskill` action goal — the reasoner stores the accepted goal handle (`_active_rskill_goal`) and `_on_failure` routes a reward wake to `_cancel_inflight_rskill_for_reward()` (`is_reward_wake` predicate). The runner already honors `goal_handle.is_cancel_requested` and returns `canceled`; the reasoner distinguishes a reward-driven cancel (`_rskill_cancel_reason="reward"`) from an operator/estop cancel and re-enters the three-tier verify gate (`_maybe_verify_active_mission_task`) on the former, skipping the controller-failure path (an intentional stop is not a fault). The resolved patience ceiling (below) stays armed as the backstop.
- **Patience + band edges read the active `RewardContract` (§1/§3, implemented).** The reasoner loads the same reward-model manifest the monitor uses (`reward_manifest_path` param, set by the deploy launch from `reward_monitor_manifest`) into `_reward_contract`. Dispatch resolves the patience ceiling via `resolve_patience_s` (LLM `patience_s` override > reward model's `default_patience_s` > legacy `deadline_s`) and sends it as the goal's `deadline_s` + the reasoner-side backstop timer; the verify gate resolves band edges via `resolve_band_edges` (the contract's `success_threshold`/`check_floor`, else the module-level system fallback). The `_DEFAULT_*` constants survive only as the no-reward-model fallback.
- `deadline_s` is demoted to the patience ceiling; completion is reward-shape + VLM, never a clock or a lone threshold.
- Scales to SARMs with zero re-architecting — each ships its own calibration.
- The `describe_image` multimodal primitive (landed) is the tier-2 mechanism.
- Layer: reasoner (S2) + reward (perception) contract only — no actuation/safety path touched (the signal stays advisory, CLAUDE.md §1.1).
