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

## Consequences
- `deadline_s` is demoted to the patience ceiling; completion is reward-shape + VLM, never a clock or a lone threshold.
- Scales to SARMs with zero re-architecting — each ships its own calibration.
- The `describe_image` multimodal primitive (landed) is the tier-2 mechanism.
- Layer: reasoner (S2) + reward (perception) contract only — no actuation/safety path touched (the signal stays advisory, CLAUDE.md §1.1).
