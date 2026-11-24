# Reasoner Design & Decisions

Why the reasoner (S2) works the way it does, one section per logic problem: the problem, the solution, the reason. The mechanism (cadence, tool contract, palette, model selection) is in [`reasoner.md`](reasoner.md) and the [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md). The decision log is private (`OpenRAL/management`).

The reasoner is an event-driven LLM supervisor: `context → LLM → one typed tool call`. It decides what to do next and never drives motors. It proposes; the C++ safety kernel disposes. The common idea is to wrap an unreliable LLM in deterministic, bounded scaffolding (see [Design through-lines](#design-through-lines)).

---

## 1. The core loop: one typed tool call per slow tick

**Problem.** An LLM on a fast timer burns tokens doing nothing. A free-form LLM that emits prose or several actions at once cannot be dispatched or replayed safely.

**Solution.** Events force a tick; a slow timer beat only runs gates (full trigger table in [`reasoner.md`](reasoner.md#cadence-event-model)). Each tick yields exactly one Pydantic-validated `ReasonerToolCall`. Only `rskill_runner_node` publishes `ActionChunk`, and every action passes the C++ safety kernel.

**Execution model.** The blocking LLM call (`select_tool`, and the VLM gate's `describe_image`) runs on its own worker, off the rclpy executor. A tick has three phases: `ReasonerCore.prepare_tick` (gates and context render) → `run_prepared_llm` (worker) → `finish_tick` (bookkeeping and dispatch, marshalled back through a guard condition). One worker means one outstanding call; ticks requested mid-flight coalesce into one replay. While the LLM thinks for 10-60 s, goal results and Tier-A preemptions still run on time. `finish_tick` drains only the prepare-time snapshot, so an operator prompt that arrives mid-flight survives for the next tick.

**Why.** Polling a running skill every beat made most calls answer `wait`, so the beat is now a backstop and deadlock insurance. One typed call per tick makes a run dispatchable, traceable and replayable from the trace alone.

---

## 2. Knowing what is in front of it: perception without 3D

**Problem.** To act on "the cup" the reasoner must know a cup is visible and refer to it across ticks. The 3D scene graph needs depth, but many deploy cameras are RGB-only (LIBERO), so `scene_objects` comes up empty and the reasoner has nothing to ground against.

**Solution.** Two read-only perception surfaces, neither needing depth:

- **Camera-space `in_view` enumeration.** The continuous detector stamps a stable per-object `det_id` (2D-IoU `DetectionTracker2D`), rendered as `in_view[top]: #0 milk @px(412,233), ...`. Pixel centres stay on a separate line from `scene_objects[map]: ...@(x,y,z)` so coordinate spaces never blur. `scene_objects` lists every instance, and the spatial-memory `memory_objects[map]` line gives each instance the id `grasp_target.object_id` takes, so "pick up the boxes" decomposes to one subtask per id.
- **A sticky `located` line.** The continuous detector's fixed vocabulary mislabels goal objects (a basket read as "box"). Every successful open-vocab `locate_in_view` hit is folded by `ContextRenderer.note_located()` into a persistent `located[<cam>]` line (latest wins, capped at 12). The prompt tells the LLM `located` outranks the noisy `in_view`.

**On-demand localisation.** `locate_in_view` asks a live detector "is X in camera Y right now?" Detectors run node-per-detector, so a continuous detector and on-demand locators coexist; `LocateInViewTool` has a `detector` selector (`omdet-turbo-locator` for simple "find X", `locateanything-3b` for referring expressions). A `recall_object` miss escalates to a live `locate_in_view` before human handoff; the node does this, not the LLM.

Two usage rules follow from the detector design. Query `omdet-turbo-locator` with concrete nouns or a comma list (`"cup, bowl, basket"`), never a collective phrase, which it matches as one nonexistent class. And the launch sets `primary_camera=det_camera` so frames cache under the real camera name.

**Active search.** The `recall_object` / `resolve_place` / `locate_in_view` miss loop is bounded by `SearchProgress(SearchBudget())` (5 candidates, 5 attempts) and ends in human handoff. `plan_active_search` (ranked frontier, occluding containers first) ships in `openral_reasoner.active_search` but the node does not call it yet.

**Why.** Deciding needs labels and a stable id, not 3D poses. Camera-space enumeration is cheap and depth-free; the `located` line closes the gap where the cheap detector mislabels and the open-vocab locator confirms.

---

## 3. Turning a goal into subtasks: grounded decomposition

**Problem.** "Put all the objects on the table into the basket" must become single-object subtasks. Asking for that in prose is not enough: weak models emit vague batch subtasks.

**Solution.**

- **A structural contract.** `DecomposeMissionTool.subtasks` is `list[GroundedSubtask]`. Its validator rejects a collective `object_ref` or `text` (`is_collective_target`) and requires `text` to name `object_ref`, so a vague subtask cannot be represented on the wire.
- **A sequential task queue.** `MissionState(tasks, current)` holds `TaskState`s with the lifecycle `pending → active → verifying → {done|abandoned}`, at most one active. The operator goal seeds one task; `decompose_mission` fills the same queue. A blocked task subdivides in place (`subdivide_active`), bounded by `DEFAULT_MAX_SUBDIVIDE_DEPTH = 2`.

**Why.** A structural invariant the model cannot violate beats prose it can ignore. The queue is a deterministic record of progress, so the LLM does not re-derive the plan every tick. Decomposition still needs a capable model (§8).

---

## 4. Knowing when a subtask is done: reward-gated, VLM-adjudicated completion

**Problem.** A VLA emits chunks but no notion of success. The runner's `result.success` only means the policy ran to its deadline. A clock or one fixed threshold cannot tell "getting closer" from "stuck".

**Solution.** A layered signal stack:

1. **A reward model beside the VLA.** `kind: reward` rSkills (default Robometer-4B, NF4) score the shared camera stream and expose `progress_now` / `success_now` / trends through `query_task_progress`. The `RewardContract` manifest block declares the calibration (`success_threshold`, `frame_window_s`, ...). Advisory only.
2. **A stall watchdog on a stream.** Reward models publish `CriticScore` on `/openral/critic/score`; a `critic_id`-keyed `CriticWatchdogGroup` turns a plateau into a Tier-C `FailureTrigger`, so a stall preempts a tick instead of running to timeout.
3. **A reward-watcher wake.** When the signal hits success, plateau or the patience ceiling, the in-flight VLA is cancelled and a tick wakes with the reward trajectory in context. `patience_s` on `ExecuteRskillTool` replaces the LLM-guessed `deadline_s` as the backstop.
4. **A three-tier verdict** (`evaluate_task_verdict`): `score >= success_threshold` auto-passes (`complete_active`, no VLM call); `check_floor <= score < success_threshold` asks the VLM to adjudicate the current frame; below `check_floor` goes straight to the replanning ladder.

The gate uses the progress head, not the success head: progress separates genuine success from failure, while the success head is compressed and noisy. Both heads are rendered in the `## REWARD` context block. The reward scores the whole attempt, not a short trailing window. A per-task `TaskLocateBudget` abandons after `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS = 3` locate cycles that never dispatch a VLA.

**Why.** The reward signal is already calibrated and continuous; a clock throws that away. Degradation is honest: with no reward oracle and no VLM, the ambiguous middle is never claimed as success. It runs to the patience ceiling and hands off.

---

## 5. Always running a VLA with its reward model: pairing and VRAM fit

**Problem.** §4 only works if the reward model is co-resident with the VLA. Nothing guaranteed both fit on the GPU before loading, so a mismatch showed up as a mid-run CUDA OOM.

**Solution.** A VLA manifest names its reward model (`reward_rskill_name`, `kind == "vla"` only) and declares per-dtype VRAM (`min_vram_gb`). `assert_vla_reward_fits(vla, reward, gpu_total_gb, margin_gb=0.5)` raises `ROSConfigError` (undeclared) or `ROSGPUMemoryError` (too big). It runs three times: a pre-launch preflight in the deploy CLI (`_preflight_reward_vram_fit`, hard exit only when no capability-matched policy and no runnable wrapped-ROS skill fits), the reasoner's `_refuse_unfittable_vla` palette filter, and the runner's re-check before `from_pretrained`. Wrapped-ROS skills (Nav2, MoveIt) need no VRAM and no reward model. The gate runs in `deploy sim` / `deploy run` only, not in `benchmark` or `sim run`.

**Why.** A VLA without a reward model is blind to its own success. Sizes are knowable from manifests, so an oversized pair should fail before launch with an actionable message.

---

## 6. When things go wrong: the bounded replanning ladder

**Problem.** A failed or stalled step must escalate through progressively more disruptive recovery, and must terminate.

**Solution.** The ladder is retry → param-tweak → substitute-skill → goal-replan → human-handoff. Shipped today: the per-kind retry cap in `ReasonerCore` ([`reasoner.md`](reasoner.md#bounded-replanning)), `TaskState.failures` capped by `DEFAULT_MAX_ATTEMPTS` (3), and `TaskState.attempts` capped by `DEFAULT_MAX_TASK_DISPATCHES` (10), so successes that never finish a task are bounded without spending its tries. Exhaustion calls `abandon_active`, emits an honest "could not complete task K" with the `MissionState` snapshot, and advances. The terminal rung is always human handoff. The substitute and goal-replan rungs are only partly built.

**Why.** Every recovery path has an explicit cap and no hidden default, so a wrong reward reading or a stuck skill costs a finite number of tries and then surfaces.

---

## 7. Reusing knowledge: playbooks and self-maintained memory

**Problem.** Decision procedures lived as bespoke Python, the reasoner did not learn across episodes, and it never saw its own body or execution outcomes.

**Solution.**

- **Playbooks** (`kind: playbook`, `role: s2`) are Markdown SOPs the LLM reads, never executes. The `PlaybookContract` (`trigger`, `body_uri`, `composes_tools`, `done_predicate`, `max_steps`) is selection metadata; the body goes into the `## PLAYBOOKS` prompt block. The six in-tree playbooks are listed in [`rskills.md`](rskills.md#playbook-rskills-kind-playbook).
- **`MEMORY.md`** is semantic memory (preferences, corrections, durable home facts), complementary to the geometric scene graph. The LLM edits it only through typed `MemoryWriteTool` / `MemorySearchTool` ops, each a traced event, and a periodic consolidation pass keeps it bounded.
- **Context grounding.** A `## ROBOT` self-model (reach hull, FOV, gripper, control modes), a `## EXECUTION` block (one line per skill outcome, success and failure) and the `## REWARD` block close the loop so the next tick reasons on reality.

**Why.** Playbooks make procedures versioned, shipped content instead of code. Memory is split by kind so each lands in the right consumer. Every write is a discrete traced call, so a run stays replayable from the trace and the `MEMORY.md` snapshot. None of this adds actuation authority.

---

## 8. Choosing the brain: LLM selection and the deploy default

**Problem.** Grounded decomposition (§3) is reasoning-heavy. Weak models follow the one-tool-per-tick contract but over-locate and skip `decompose_mission`. The library must also stay provider-agnostic.

**Solution.** Selection is model-first (ADR-0088); see [`reasoner.md`](reasoner.md#reasoner-model-selection). The library factory has no default and refuses to guess. The deploy launch defaults to `gpt-5.5` with `OPENRAL_REASONER_MAX_TOKENS=16384`, so a reasoning model does not reserve its full window and get rejected on a metered key. The default needs an API key and fails loudly without one.

**Why.** Collective goals decompose reliably only on capable models; single-object goals run on cheaper ones (see the [curated models](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md#curated-reasoner-models)).

---

## Design through-lines

1. **Deterministic scaffolding around an unreliable LLM.** Typed `MissionState`, `GroundedSubtask`, calibrated reward gates and hard caps do the bookkeeping. The LLM decides what is true and what to try; the scaffolding decides what is recorded and when to stop.
2. **Everything bounded, no hidden defaults.** Beat, min interval, retry cap, attempts cap, subdivide depth, search budget, patience ceiling and locate budget are all explicit.
3. **Perception is advisory; the kernel disposes.** Detectors, reward models, critics, scene graph and memory are read-only inputs to a decision. The C++ safety kernel is the only authority over motors.
4. **Honest degradation.** A missing reward oracle or VLM never produces a claimed success.
5. **Types are the contract.** A vague subtask or an unpaired reward model is made unrepresentable, not just discouraged.

---

## See also

- [`reasoner.md`](reasoner.md): cadence, tool contract, palette gating, model selection, how to run it.
- [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md): ROS wrapper contract and provider presets.
- [Architecture overview](../architecture/overview.md) · [repo state map](../architecture/repo-state-map.html).
