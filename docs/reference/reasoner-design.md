# Reasoner Design & Decisions

How the **reasoner (S2)** actually thinks — the connective narrative behind the
individual decisions. Where [`reasoner.md`](reasoner.md) is the *reference* (the
contract, the cadence, the env vars) and each ADR is *one decision in isolation*,
this page is organized **by the logic problem the reasoner has to solve**. For
each problem: what it is, how we solve it, why we chose that, and the governing
ADR(s) to read for the full record.

It deliberately does **not** restate mechanism detail that lives in
[`reasoner.md`](reasoner.md) or the [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md) —
it links to them.

> **One sentence:** the reasoner is an event-driven LLM supervisor that closes
> `context → LLM → one typed tool call` at ~0.2 Hz; it decides *what to do next*
> and **never drives motors** — it proposes, the C++ safety kernel disposes.

A recurring design through-line runs under every section below: **wrap an
unreliable LLM in deterministic, bounded scaffolding.** The LLM decides *what is
true* and *what to try*; typed state machines, calibrated signals, and hard caps
decide *what is recorded* and *when to stop*. Keep that in mind — it explains most
of the choices.

---

## 1. The core loop — one typed tool call per slow tick

**Problem.** A robot supervisor that calls an LLM on a fixed fast timer burns
tokens doing nothing, and a free-form LLM that emits prose or multiple
simultaneous actions can't be safely dispatched or replayed.

**Solution** ([ADR-0018](../adr/0018-ros2-reasoner-supervisor.md)). The reasoner
is **event-driven with a slow heartbeat**:

- **Heartbeat** at `tick_hz = 0.2` (one tick / 5 s). A heartbeat tick that has
  seen no new event short-circuits inside `ReasonerCore` with
  `suppressed_reason="heartbeat_idle"` — no LLM call, no span.
- **Event preemption** is the real trigger, gated by a hard `min_interval_s =
  100 ms` so nothing can thrash the LLM. Four tiers:

  | Tier | Source topic | Preempts on |
  |---|---|---|
  | **A — safety** | `/openral/failure/safety` | `severity ≥ WARN` |
  | **B — execution** | `/openral/failure/{hal,sensor,rskill,wam}` | `severity ≥ FAIL` |
  | **C — progress** | `/openral/failure/critic` | `severity ≥ FAIL` |
  | **D — operator/world** | `/openral/prompt`, `/openral/perception/*` | new prompt forces; perception is informational |

- Each tick the LLM emits **exactly one** variant of the `ReasonerToolCall`
  discriminated union (`ExecuteRskill`, `LifecycleTransition`, `ReloadGstPipeline`,
  `EmitPrompt`, plus the read-only query/memory tools below) — Pydantic-validated
  structured output, never free-form JSON.

**Authority boundary.** The reasoner holds **no actuation authority**: it never
publishes `ActionChunk`. Only `rskill_runner_node` does, and every action passes
the C++ safety kernel ([ADR-0020](../adr/0020-cpp-safety-kernel.md)) before it
reaches a motor. *Python proposes; C++ disposes.*

**Why.** Event-driven cuts idle LLM calls ~85% vs a fast timer; the heartbeat is
the deadlock insurance ("task not progressing"). One typed call per tick is what
makes a run dispatchable, traceable, and replayable from the trace alone.

---

## 2. Knowing what's in front of it — perception without 3D

**Problem.** To act on "the cup" the reasoner must know a cup is visible and be
able to refer to it across ticks. The 3D scene graph (`scene_objects`) needs
depth to lift object poses — but many deploy cameras are **RGB-only** (LIBERO),
so `scene_objects` comes up empty and the reasoner has nothing to ground against.
This is exactly what made a real run loop forever on a collective goal.

**Solution.** Two complementary perception surfaces, both **read-only**, neither
requiring depth:

- **Camera-space `in_view` enumeration** ([ADR-0076](../adr/0076-detection-identity-and-camera-space-enumeration.md)).
  The continuous detector stamps a stable per-object `det_id` (via a 2D-IoU
  `DetectionTracker2D`) and the context renders a line the LLM can refer to:
  `in_view[top]: #0 milk @px(412,233), #1 ketchup @px(388,251), …`. Pixel
  centers, not 3D poses — kept in a separate line from `scene_objects[map]:
  …@(x,y,z)` so coordinate spaces never blur. Identity exists with or without
  depth.
- **A sticky `located` line** ([ADR-0076 §4](../adr/0076-detection-identity-and-camera-space-enumeration.md)).
  The continuous detector's fixed ~230-class vocab *mislabels* the goal objects
  (a basket read as "box", ketchup as "bottle"). So every successful open-vocab
  `locate_in_view` hit is folded by `ContextRenderer.note_located()` into a
  persistent `located[<cam>]` line (latest-wins, `_LOCATED_CAP=12`). The prompt
  tells the LLM `located` is authoritative over the noisy `in_view`.

**On-demand localization** ([ADR-0043](../adr/0043-locate-in-view-reasoner-tool.md),
[ADR-0056](../adr/0056-on-demand-detectors-as-promptable-reasoner-tools.md)). The
`locate_in_view` tool asks a live detector "is X in camera Y right now?" via the
`/openral/perception/<detector>/locate_in_view` service. ADR-0056 makes detectors
**node-per-detector** so a continuous detector and one or more on-demand locators
coexist, and gives `LocateInViewTool` a `detector` selector (fast
`omdet-turbo-locator` for simple "find X", `locateanything-3b` for referring
expressions). A `recall_object` miss auto-escalates to a live `locate_in_view`
before handoff — policy in the node, not dependent on the LLM picking the tool.

> **Two hard-won usability fixes** ([ADR-0056 amendment 2026-06-29](../adr/0056-on-demand-detectors-as-promptable-reasoner-tools.md)):
> (1) `omdet-turbo-locator` is a multi-label detector — query it with concrete
> object **nouns** / a comma-list (`"cup, bowl, basket"`), never a collective
> phrase (`"the objects on the table"`), which it matches as one nonexistent
> class. (2) The launch sets `primary_camera=det_camera` so frames cache under
> the real camera name (`"top"`); otherwise every `locate_in_view(camera="top")`
> missed against a frame stored under `"default"`. Both surfaced as the same
> `found=False` loop.

**Active search** ([ADR-0039](../adr/0039-llm-task-planning-active-search.md),
*proposed*). When an object isn't in memory at all, `recall_object` /
`resolve_place` plus a bounded `SearchBudget` (max places **and** wall-clock,
no hidden default) drive a *look → navigate → re-query* loop, opening occluding
containers first, terminating in human-handoff.

**Why.** The reasoner doesn't need 3D poses to *decide* — it needs labels + a
stable id to refer. Camera-space enumeration is cheap (it already subscribes to
the detector) and depth-free. The sticky `located` line closes the real gap where
the cheap continuous detector mislabels but the open-vocab locator confirms.

---

## 3. Turning a goal into actionable subtasks — grounded decomposition

**Problem.** A collective operator goal ("put **all** the objects on the table
into the basket") has to become an ordered list of single-object subtasks. Live
testing showed that *prose* asking the LLM for "one specific object per subtask"
isn't enough — weak models emit vague "batch" subtasks even with the grounded
object list in context.

**Solution.**

- **A structural contract, not a prompt** ([ADR-0075](../adr/0075-grounded-decomposition-contract.md)).
  `DecomposeMissionTool.subtasks` is `list[GroundedSubtask]`, where
  `GroundedSubtask(object_ref, text)` carries a Pydantic `@model_validator` that
  **rejects** a collective `object_ref`/`text` (shared `is_collective_target`
  predicate) and requires `text` to name `object_ref`. The type makes a vague
  subtask un-representable on the wire.
- **A sequential task queue** ([ADR-0073](../adr/0073-reasoner-success-gating-and-task-queue.md)).
  `MissionState(tasks, current)` holds `TaskState`s with a strict lifecycle
  `pending → active → verifying → {done|abandoned}` (at most one active). The
  operator goal seeds a single task (`MissionState.from_prompt`); the LLM
  decomposes via `DecomposeMissionTool`, which populates the *same* queue. A
  blocked task can be subdivided in place (`subdivide_active`), bounded by
  `DEFAULT_MAX_SUBDIVIDE_DEPTH = 2`.

**Why.** "Types are the contract" (CLAUDE.md §1.3) — a structural invariant the
model can't violate beats prose it can ignore. The queue gives the reasoner a
*deterministic* record of where it is, so progress doesn't depend on the LLM
re-deriving the plan every tick. The decomposition itself still needs a capable
model — see **§8 (Choosing the brain)** below.

---

## 4. Knowing when a subtask is actually done — reward-gated, VLM-adjudicated completion

**Problem.** A VLA emits action chunks but **no notion of success**. The skill
runner's `result.success` only means "the policy ran to its deadline without
crashing," not "the task was accomplished." Gating completion on a clock, or on a
single hardcoded `0.8` threshold, can't tell *getting closer* from *stuck* and
misclassifies a physically-successful result scored 0.78.

**Solution** — a layered signal stack:

1. **A reward model running parallel to the VLA** ([ADR-0057](../adr/0057-robometer-reward-rskill.md)).
   `kind: reward` rSkills (default Robometer-4B, NF4, ~3.6 GB) score the shared
   camera stream every ~1–2 s and expose `progress_now` / `success_now` /
   trends through the read-only `query_task_progress` tool. Its `RewardContract`
   manifest block declares the calibration (`success_threshold`,
   `frame_window_s`, …). Advisory only.
2. **A stall watchdog that fires a stream, not a poll** ([ADR-0064](../adr/0064-critic-score-topic-and-tier-c-producer.md)).
   Every reward model publishes self-describing `CriticScore` on
   `/openral/critic/score`; a `critic_id`-keyed `CriticWatchdogGroup` watches for
   a stall and publishes a Tier-C `FailureTrigger` on `/openral/failure/critic`
   — so a plateau *preempts a tick* instead of silently running to timeout.
3. **A reward-watcher wake** ([ADR-0074](../adr/0074-vlm-adjudicated-completion-and-reward-driven-progress.md)).
   The instant the reward signal hits **success**, **plateau**, or the
   **patience ceiling**, the in-flight VLA is cancelled and a normal reasoner
   tick wakes with the reward trajectory injected. `patience_s` (an
   `ExecuteRskillTool` field, default from the contract) replaces the
   LLM-guessed `deadline_s` as the execution backstop.
4. **A three-tier verdict** ([ADR-0074](../adr/0074-vlm-adjudicated-completion-and-reward-driven-progress.md)).
   `evaluate_task_verdict` replaces the hardcoded threshold:
   - **auto-pass** (`score ≥ success_threshold`) → `complete_active`, no VLM call;
   - **vlm_check** (`check_floor ≤ score < success_threshold`) → adjudicate the
     current frame with `describe_image` ("is `<task>` complete? yes/no");
   - **ladder** (`score < check_floor`) → no VLM, straight to replanning.

> **Amendment 2026-06-29** — gate on the **progress head**, not success: progress
> separates genuine success (0.80–0.86) from failure (~0.74), while the success
> head is compressed/noisy (0.56–0.79). Both heads are rendered to the LLM in a
> `## REWARD` context block (`set_reward_state`). The reward also scores the
> **whole attempt** (start→now), not an 8 s trailing window (`frame_window_s`
> raised `8.0 → 40.0`). A per-task `TaskLocateBudget` abandons after
> `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS = 3` locate cycles that never dispatch a VLA.

**Why.** The reward signal is *already calibrated and continuous* — a clock and a
single threshold throw that away. The authority stack (system fallback < reward
contract default < LLM per-task override) scales to future per-task SARMs with no
re-architecting. Degradation is honest: with no reward oracle and no VLM, the
ambiguous middle is never claimed as success — it runs to the patience ceiling
and hands off.

---

## 5. Always running a VLA with its reward model — pairing + VRAM fit

**Problem.** §4 only works if the reward model is actually co-resident with the
VLA. Pairing used to be an implicit deploy flag decoupled from which VLA the
reasoner picks at runtime, and nothing guaranteed both fit on the GPU before
loading — you'd discover the mismatch as a mid-run CUDA OOM.

**Solution** ([ADR-0077](../adr/0077-vla-reward-pairing-and-vram-fit.md)). A VLA
manifest **names its reward model** (`reward_rskill_name`, allowed only for
`kind == "vla"`; `None` = deployment default) and declares per-dtype VRAM
(`min_vram_gb`, read by `active_min_vram_gb()`). A pure helper
`assert_vla_reward_fits(vla, reward, gpu_total_gb, margin_gb=0.5)` raises
`ROSConfigError` (undeclared) or `ROSGPUMemoryError` (won't fit). It runs at two
points: the reasoner's `_refuse_unfittable_vla` drops a non-fitting VLA from the
palette so it's never dispatched, and (defense-in-depth) the runner re-checks
before `from_pretrained`. The deploy CLI adds a **pre-launch** preflight
(`_preflight_reward_vram_fit`, torch-free `nvidia-smi` probe) that hard-exits only
when *no* capability-matched VLA fits — **`deploy sim` / `deploy run` only**, not
`benchmark` / `sim run`. (Eviction of *other* peers — detectors before the VLA —
is the complementary [ADR-0050](../adr/0050-single-resident-skill-vram-eviction.md).)

**Why.** A VLA without a reward model is blind to its own success, so it should
never run alone. Sizes are knowable from the manifests; an oversized pair should
fail before launch with an actionable message, not as an opaque OOM mid-grasp.

---

## 6. When things go wrong — the bounded replanning ladder

**Problem.** A failed or stalled step must escalate through *progressively more
disruptive* recovery, and must be guaranteed to terminate (no infinite retry
storm).

**Solution** ([ADR-0018](../adr/0018-ros2-reasoner-supervisor.md),
[ADR-0073](../adr/0073-reasoner-success-gating-and-task-queue.md)). A fixed ladder:
**retry → param-tweak → substitute-skill → goal-replan → human-handoff**. The
shipped gate is `ReasonerCore`'s per-kind retry cap (`retry_cap_per_kind`,
default 3) — consecutive same-kind selections beyond the cap are suppressed
(`suppressed_reason="retry_cap"`), and the streak resets on a material context
shift. At the mission layer, `TaskState.attempts` bounds a task's total tries;
exhaustion calls `abandon_active`, emits an honest *"could not complete task K"*
with the `MissionState` snapshot, and advances. The terminal rung is always
human-handoff.

**Why.** Bounded everything (CLAUDE.md §1.4) — every recovery path has an explicit
cap and no hidden default, so a wrong reward reading or a stuck skill costs a
finite number of tries, then surfaces honestly. *(The substitute-skill and
goal-replan rungs are partially realized; the retry cap + attempts cap + handoff
ship today — see [`reasoner.md` §Bounded replanning](reasoner.md#bounded-replanning).)*

---

## 7. Learning and reusing knowledge — playbooks + self-maintained memory

**Problem.** The reasoner has strong *mechanism* but thin *content*: decision
procedures lived as bespoke Python, it didn't learn across episodes, and it never
saw its own body or its execution outcomes.

**Solution** ([ADR-0072](../adr/0072-reasoner-playbooks-and-self-maintained-memory.md),
*proposed/phased*):

- **Playbooks** (`kind: "playbook"`, `role: "s2"`) — Markdown SOPs the LLM
  *reads and interprets*, never executes. Their `PlaybookContract` (trigger,
  `composes_tools`, `done_predicate`, `max_steps`, fallback) is selection
  metadata; the `PLAYBOOK.md` body is injected into the `## PLAYBOOKS` prompt
  block. Seven launch playbooks encode the recurring procedures
  (`find-object`, `decompose-mission`, `verify-outcome`, `preflight-reach`,
  `stage-for-manipulation`, `clarify-ambiguity`, `subtask-with-goal`).
- **Self-maintained `MEMORY.md`** — *semantic/narrative* memory (preferences,
  corrections, lessons, durable home facts), complementary to the *geometric*
  scene graph ([ADR-0038](../adr/0038-persistent-semantic-spatial-memory.md)):
  the scene graph answers "where is the mug?", `MEMORY.md` answers "how does this
  household like things done?". The LLM edits it only through typed
  `MemoryWriteTool` / `MemorySearchTool` ops (add/update/supersede/delete) — a
  traced event, never a free-form rewrite — and a periodic consolidation pass
  keeps it bounded.
- **Context grounding** — a `## ROBOT` self-model block (reach hull, FOV,
  gripper, control modes, derived from `RobotCapabilities` at configure), a
  `## EXECUTION` block (one NL line per skill outcome, success *and* failure),
  and the `## REWARD` block from §4. These close the loop so the next tick
  reasons on reality.

**Why.** Playbooks make decision procedures *authored content* (versioned,
shipped, discoverable) instead of code. Memory is split by *kind* (geometric vs
semantic) so each lands in its correct consumer. Every write is a discrete traced
call — the planner can't silently corrupt the file, and a run stays replayable
from the trace + the `MEMORY.md` snapshot. None of this adds actuation authority.

---

## 8. Choosing the brain — LLM selection & the deploy default

**Problem.** The reasoner's hardest job (grounded decomposition of a collective
goal, §3) is genuinely reasoning-heavy. Weak/cheap models follow the one-tool-
per-tick contract fine but **over-locate and never call `decompose_mission`**;
the library must stay provider-agnostic (no cloud lock-in).

**Solution.** Every provider satisfies the `ToolUseClient` protocol, selected at
`on_configure` from `OPENRAL_REASONER_LLM_*` env. The *library* factory
(`build_tool_use_client_from_env`) has **no default** and refuses to guess. The
*deploy-sim launch* does pick one when env is unset: `provider=openrouter`,
`model=openai/gpt-5.5`, with `OPENRAL_REASONER_LLM_MAX_TOKENS=16384` defaulted so
a reasoning model doesn't reserve its full window and get 402'd on a metered key.
Explicit env always wins; the default needs an API key (fails loudly at activate
otherwise).

**Why.** In live deploy testing GPT-5.5 was the only model that *reliably*
decomposed the collective goal (glm-5.2 over-located and never decomposed; Opus
4.8 worked but needed nudges; the OpenRouter `:free` tier emitted the placeholder
skill id). Simpler single-object goals run fine on the cheaper baselines in the
[README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md#baseline-llm-recommended-configurations).

---

## Design through-lines

The same principles recur across every section — if you internalize these, the
individual decisions follow:

1. **Deterministic scaffolding around an unreliable LLM.** Typed `MissionState`,
   `GroundedSubtask`, calibrated reward gates, and hard caps do the bookkeeping
   the model can't be trusted to. The LLM decides *what is true* and *what to
   try*; the scaffolding decides *what is recorded* and *when to stop*.
2. **Everything bounded, no hidden defaults** (CLAUDE.md §1.4). Heartbeat,
   min-interval, retry cap, attempts cap, subdivide depth, search budget,
   patience ceiling, locate budget — all explicit.
3. **Perception is advisory; the kernel disposes** (CLAUDE.md §1.1). Detectors,
   reward models, critics, scene graph, and memory are all read-only inputs to a
   *decision*; none commands a motor. The C++ safety kernel is the only authority.
4. **Honest degradation.** Missing a reward oracle or a VLM never produces a
   claimed-uncertain success — the attempt runs to the bound and hands off.
5. **Types are the contract** (CLAUDE.md §1.3). A vague subtask or an unpaired
   reward model is made *un-representable on the wire*, not merely discouraged.

---

## ADR index — logic problem → decision record

| Logic problem | ADR(s) |
|---|---|
| Tick loop, tiers, one-tool-per-tick, authority boundary | [0018](../adr/0018-ros2-reasoner-supervisor.md) |
| Camera-space `in_view` enumeration, `det_id`, sticky `located` | [0076](../adr/0076-detection-identity-and-camera-space-enumeration.md) |
| On-demand `locate_in_view`, node-per-detector, model selection | [0043](../adr/0043-locate-in-view-reasoner-tool.md), [0056](../adr/0056-on-demand-detectors-as-promptable-reasoner-tools.md) |
| Scene VLM Q&A (`query_scene`) | [0047](../adr/0047-vlm-rskill-kind.md) |
| Active object search over the scene graph | [0039](../adr/0039-llm-task-planning-active-search.md) |
| Grounded decomposition (`GroundedSubtask`) | [0075](../adr/0075-grounded-decomposition-contract.md) |
| Mission queue, success-gating | [0073](../adr/0073-reasoner-success-gating-and-task-queue.md) |
| Reward model (`kind: reward`, Robometer) | [0057](../adr/0057-robometer-reward-rskill.md) |
| Critic-score stall → Tier-C replan | [0064](../adr/0064-critic-score-topic-and-tier-c-producer.md) |
| Reward-watcher + three-tier VLM-adjudicated verdict | [0074](../adr/0074-vlm-adjudicated-completion-and-reward-driven-progress.md) |
| VLA↔reward pairing + VRAM fit | [0077](../adr/0077-vla-reward-pairing-and-vram-fit.md), [0050](../adr/0050-single-resident-skill-vram-eviction.md) |
| Playbooks + self-maintained `MEMORY.md` | [0072](../adr/0072-reasoner-playbooks-and-self-maintained-memory.md) |

---

## See also

- [`reasoner.md`](reasoner.md) — the reasoner **reference** (cadence, tool
  contract, palette gating, provider table, observability, how to run it).
- [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md) — ROS wrapper contract, provider presets, baseline LLM configs.
- [Architecture overview](../architecture/overview.md) · [repo state map](../architecture/repo-state-map.html).
