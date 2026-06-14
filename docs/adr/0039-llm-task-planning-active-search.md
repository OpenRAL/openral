# ADR-0039: LLM task planning and active object search over the scene graph

- Status: **Proposed**
- Date: 2026-06-02
- Related: [ADR-0038](0038-persistent-semantic-spatial-memory.md) (the Layer-2
  scene-graph world model + read-only `RecallObject*` / `ResolvePlace*` query
  contracts this ADR consumes and exposes to the reasoner); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  (the S2 Reasoner, the **closed** `ReasonerToolCall` palette this ADR extends,
  and the bounded replanning ladder); [ADR-0022](0022-rskill-action-vocabulary.md)
  (the `RSkillAction` verb vocabulary — NAVIGATE / OPEN / GRASP / POUR / PLACE …
  — the planner sequences); [ADR-0024](0024-ros-wrapped-rskills.md) (the wrapped
  Nav2 / MoveIt skills that execute the steps); [ADR-0030](0030-geometric-safety-collision-checking.md)
  (the kernel that still gates every motion); CLAUDE.md §1.1 (safety beats
  helpfulness), §1.4 (explicit, no hidden fallback), §3 Reasoner & dispatch
  (LLM tool calls are Pydantic structured output; bounded replanning ladder;
  no hidden default). Task-archetype coverage is grounded in a benchmark survey
  (ALFRED, BEHAVIOR-1K, Habitat HAB, Housekeep, RoboCasa, TEACh, PARTNR,
  OK-Robot/DynaMem/TidyBot).

## Context

[ADR-0038](0038-persistent-semantic-spatial-memory.md) gives the robot a
durable, queryable world model — a scene graph of objects/places/rooms/agents
with `RecallObjectQuery` / `ResolvePlaceQuery` read contracts. What it deliberately
does **not** specify is *how the S2 Reasoner uses it*: how it turns a
natural-language goal into a sequence of skill calls, how it queries the graph
mid-plan, and — critically — **what it does when the thing it needs is not in
memory.**

The driving example is "bring me a cup of wine": the robot must decompose the
request, recall the wine (in the fridge) and a glass (location unknown), open the
fridge, search likely places for the glass, pour, and deliver to the requester.
Two of those steps are not mere lookups:

- **Active search.** A glass may never have been observed. A useful robot does
  not give up — it reasons *"glasses are usually in cabinets or on the kitchen
  table,"* generates candidate locations from the scene graph's rooms/places plus
  commonsense priors, and searches them. This is the Housekeep / object-goal
  pattern, and it is the gap `ROSObjectNotInMemory` was designed to trigger.
- **Container access.** The wine is inside an *occluding* container
  (`fridge.occludes_contents`); the planner must insert an `OPEN` step before the
  grasp. The HAB "Prepare Groceries" task is exactly this.

This is squarely **Layer 4 (Reasoning)** behavior. It belongs in its own ADR
because it (a) extends the **closed** `ReasonerToolCall` palette (ADR-0018 §4),
which the palette's own contract says requires ROS-side dispatch and an authority
review, and (b) adds a new bounded behavior (active search) to the replanning
ladder.

### Task-archetype coverage (why this scope, and what's deferred)

A survey of household benchmarks yields ~11 distinct task archetypes. This ADR
targets the subset the wine task needs and that the ADR-0038 world model already
supports; the rest are named so the boundary is explicit.

| Archetype | Example | In scope here? |
|---|---|---|
| Fetch-and-deliver | "bring me a cup of wine" | **Yes** |
| Find-object-with-unknown-location (active search) | "find a glass" | **Yes** |
| Open-receptacle-to-access-contents | open the fridge | **Yes** (planner inserts `OPEN`) |
| Navigate-to-view / examine | "find the mug" (ADR-0038) | **Yes** |
| Human-handover / deliver-to-person | "bring it to me" | **Yes** (agent node) |
| Rearrange-to-goal-configuration | "tidy these 5 items" | Deferred |
| Tidy-to-canonical-locations (commonsense placement) | Housekeep / TidyBot | Deferred (needs `belongs_at` priors) |
| Set-table / multi-object assembly | HAB Set Table | Deferred |
| State-change (heat/clean/cook/slice/fill) | ALFRED Heat&Place; "pour" | Partial — pour is a skill; object *state* modeling deferred |
| Tool / appliance use | RoboCasa "turn knob" | Deferred (needs affordance nodes) |
| Long-horizon multi-agent / temporal-constrained | PARTNR | Deferred (needs ordering + per-agent capability model) |

Deferred archetypes mostly need world-model *attributes* (object `state`,
`belongs_at` priors, ordering/affordance edges) and *skills* that this ADR does
not introduce — they are future ADRs building on the same substrate.

## Decision

### 1. Expose the scene-graph queries to the reasoner as read-only tools

Two new variants are added to the `ReasonerToolCall` discriminated union
(ADR-0018 §4):

- **`RecallObjectTool`** → dispatches an ADR-0038 `RecallObjectQuery` against the
  scene-graph service; returns a `RecallObjectResult` (recall pose + camera-facing
  `ApproachViewpoint` + `inside_container_id`).
- **`ResolvePlaceTool`** → dispatches a `ResolvePlaceQuery`; returns a
  `ResolvePlaceResult` (goal pose + `traversable_to` path).

These are **read-only**: they query memory and return data to the LLM's next
reasoning step (agentic retrieval, the ReMEmbR pattern). They dispatch a
**service call** (not an action goal), produce **no actuation**, and — like every
existing variant — the reasoner **holds no authority over actuation** (CLAUDE.md
§3; ADR-0018 §4). Because the palette is closed, this ADR carries the required
extension: (a) the two variants here, (b) the matching read-only dispatch in
`openral_reasoner_ros.reasoner_node`, (c) a CLAUDE.md note that the reasoner's
**read** surface now includes spatial memory (its **actuation** authority is
unchanged — no §7 safety-authority shift).

*Alternative considered:* inject query results into the reasoner's `WorldState`
context every tick instead of tool calls. Rejected as the primary path because
active search is inherently **iterative** — the LLM must query, look, and
re-query — which the tool-call loop expresses naturally and a static context dump
does not. (A small always-on context summary may still be added later.)

### 2. Task decomposition

The reasoner decomposes a natural-language goal into an ordered sequence of
`ExecuteRskillTool` calls (ADR-0022 verbs: NAVIGATE, OPEN, GRASP, POUR, PLACE,
…) interleaved with `RecallObjectTool` / `ResolvePlaceTool` queries, emitting one
tool call per tick via the LLM's structured-output mode (no free-form JSON, §3).
Full plan-tree execution via BT v4 (`bt_executor_node`) remains the future option
ADR-0018 already reserves; this ADR uses the existing per-tick tool-call loop.

### 3. Active object search (the new behavior)

When `RecallObjectTool` yields `ROSObjectNotInMemory`, the reasoner enters a
**bounded active search**:

1. **Generate candidates.** Combine (a) scene-graph structure — rooms and
   `place` nodes, especially containers whose `occludes_contents` is true — with
   (b) the LLM's commonsense priors ("a wine glass is usually in a cabinet or on
   the kitchen table"). Rank candidate places.
2. **Search loop.** For each candidate in rank order, within budget: `OPEN` the
   container if it occludes, navigate to the place (`ResolvePlaceTool`), look,
   and let perception update the scene graph (ADR-0038 Phase 2 builder); re-issue
   `RecallObjectTool`.
3. **Terminate.** Success on a hit; otherwise stop at a **search budget** (max
   candidate places and/or wall-clock) and escalate to **human-handoff** — the
   terminal rung of the ADR-0018 replanning ladder. The budget is explicit
   reasoner config (no hidden default, §1.4) and the search is fully traced
   (every candidate + outcome on the OTel span) so the run is replayable (§1.8).

Active search slots into the existing bounded ladder (retry → param-tweak →
substitute-skill → **goal-replan / search** → human-handoff) rather than adding
an unbounded loop.

### 4. Container access

When a `RecallObjectResult` match carries `inside_container_id` (or a candidate
place is a container with `occludes_contents`), the planner inserts an
`OPEN`-verb skill step on that container before the access/grasp step, and (where
appropriate) a `CLOSE` after. This is read directly from the ADR-0038 attributes;
no new world-model field is required.

### 5. Safety and bounds (unchanged invariants)

The planner **proposes**; every emitted skill still produces an `Action` chunk
that crosses the ADR-0030 C++ safety kernel, which **disposes**. The scene graph
remains **advisory** — a wrong recall yields a bad plan the kernel still vetoes,
never a relaxed safety check (§1.1). Active search is **bounded** and terminates
in human-handoff. No tool added here actuates directly.

## Alternatives considered

- **Query-as-context instead of tools.** §1 — rejected as primary (active search
  is iterative).
- **Hard-coded search heuristics instead of LLM priors.** Brittle across homes;
  the commonsense prior is the whole point (Housekeep shows LLM priors beat
  fixed heuristics). LLM priors are used, but **grounded** by the scene graph and
  **bounded** by the search budget so they cannot run away.
- **BT v4 plan trees now.** Deferred to the ADR-0018 future option; the per-tick
  tool-call loop is sufficient for the in-scope archetypes and avoids standing up
  `bt_executor_node` here.
- **Put planning in the WAM (Layer 5).** The WAM is optional, best-effort,
  deadline-fallback mental simulation — unsuitable as the primary task planner.
  It may later *gate* candidate plans (failure anticipation), but the planner
  lives in the S2 Reasoner.

## Consequences

- The reasoner can execute long-horizon fetch/search/deliver tasks end to end,
  recovering from "not in memory" via bounded commonsense search instead of
  failing — the behavior the wine task needs.
- The `ReasonerToolCall` palette grows by two **read-only** variants; ADR-0018's
  dispatch and a CLAUDE.md read-surface note are updated in the same change. No
  actuation-authority shift.
- Deferred archetypes (rearrange/tidy, set-table, state-change, tool-use,
  multi-agent/temporal) get a clear home: world-model attribute ADRs +
  skill ADRs on top of this planner.

## Phasing

1. **This ADR + palette contracts (landed).** `RecallObjectTool` /
   `ResolvePlaceTool` read-only variants on the `ReasonerToolCall` union, with
   fuzz round-trip + discriminator-decode tests and docs. The two variants are a
   typed contract **not yet exposed in the live provider palette** — the ROS
   dispatch + result-return path needs `rclpy` (untestable off-robot) and the
   agentic result-return loop is a real design step, so both move to Phase 2.
   (Depends on ADR-0038 Phase 1 + Phase 2, landed.)
2. **Query rendering + result-return bridge (landed, pure-Python).**
   `ToolPalette.spatial_memory_available` gates rendering the two tools in the
   provider palette (`_tool_palette_to_anthropic_tools`); the decoder already
   routes their payloads. `openral_reasoner.spatial_query.run_spatial_query` maps
   a tool call → ADR-0038 query, runs it against an injected
   `SpatialMemoryQuerier` (the real `SpatialMemory` satisfies the Protocol — no
   `openral_world_state` import in Layer 4), and renders an LLM-readable result
   (a miss → "not in memory" text, never a fabricated pose). Tested against the
   real home fixture.
2b. **ROS dispatch wiring (landed).** `reasoner_node` accepts an optional
   `spatial_memory` backend; when present it sets `spatial_memory_available`,
   routes `recall_object` / `resolve_place` through `_dispatch_spatial_query`
   (→ `run_spatial_query`), and republishes the result as a `PromptStamped`
   (frame_id `"spatial_memory"`, so it is consumed not self-filtered) — the
   prompt cascade. Verified live on ROS 2 Jazzy
   (`tests/integration/test_reasoner_node_end_to_end.py::test_recall_object_query_reprompts_with_spatial_memory_result`):
   "bring me a cup of wine" → `RecallObjectTool` → re-prompt naming the wine and
   the occluding fridge.
2c. **Deployment wiring — preloaded map (landed).** `reasoner_node` declares a
   `spatial_memory_path` ROS parameter; `_maybe_load_spatial_memory` loads a
   persisted `SceneGraph` into a `SpatialMemory` at `on_configure` (when no
   backend was injected) and flips `spatial_memory_available`, so a launched node
   offers + dispatches the query tools against a preloaded map.
   `sim_e2e.launch.py` exposes it as `spatial_memory_path:=<path>`. Verified live
   (`tests/integration/...::test_spatial_memory_path_param_preloads_query_backend`).
   **Remaining for the dynamic path:** a producer that fills
   `WorldState.detected_objects` from live perception (the Layer-2 detected-object
   ingest is still planned) feeding the ADR-0038 Phase 2 builder, plus the
   active-search loop bound (Phase 4).
3. **Task decomposition.** Multi-step `ExecuteRskillTool` sequencing for
   fetch-and-deliver + open-receptacle; sim test on the home fixture.
4. **Active search — bound (landed).** `openral_reasoner.active_search`:
   `plan_active_search` builds a `SearchBudget`-bounded candidate frontier from
   the scene graph (occluding containers first, then places; LLM prioritizes
   among them via priors); `SearchProgress` is the runaway bound. `reasoner_node`
   caps consecutive query re-prompts and terminates the cascade in human-handoff
   — verified live (`...::test_active_search_cascade_is_bounded_and_hands_off`).
   **Remaining:** wire `plan_active_search`'s frontier into the miss re-prompt so
   the LLM is handed the ranked candidates (needs scene-graph access at the
   dispatch site), and the navigate→look→re-query skill loop itself.
5. **Deliver-to-agent.** Resolve the requester `agent` node as the return goal.

Each runtime phase ships sim tests against real fixtures (no mocks, §1.11) and
updates all affected docs in the same PR (§1.14).
