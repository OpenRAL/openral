# Layer 5–8 — Reasoning, WAM, Safety, Observability

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Layer 8 (Observability) is fully shipped — traces + metrics + structlog→OTLP log bridge, with W3C TraceContext propagation helpers for cross-process correlation (Python ↔ ROS 2 ↔ C++ safety kernel). Layers 5–7 ship their **Protocol surface** (landed 2026-05-18) so the rest of the runtime can compose against locked signatures; concrete implementations (LLM clients, generative WAM adapters, the C++ safety kernel) are still planned.

### `python/reasoner/src/openral_reasoner/protocol.py`
_S2 reasoner Protocols (ADR-0005)._

- `class LLMClient(Protocol)` — Wire-level Protocol for an LLM provider that supports structured output. Attribute: `model_id`. Method: `complete_structured(prompt, schema) -> Plan`. (L29)
- `class Reasoner(Protocol)` — Planning-layer Protocol every S2 reasoner satisfies. Attributes: `plan_rate_hz`, `client: LLMClient | None`. Method: `plan(world_state, goal) -> Plan` — emits a validated `Plan` for the goal given the current `WorldState`; raises `ROSReasonerInvalidPlan` / `ROSPlanningError`. (L64)

### `python/reasoner/src/openral_reasoner/plan.py`
_Pydantic v2 structured-output schemas the LLM emits (ADR-0003 + ADR-0005)._

- `class ToolCall(BaseModel)` — One leaf of a `Plan` — a single skill invocation. Fields: `rskill_id: str (min_length=1)`, `params: dict[str, Any]`, `rationale: str | None`. `extra="forbid"`. (L24)
- `class Plan(BaseModel)` — The structured LLM output the reasoner emits per planning tick. Fields: `goal: str (min_length=1)`, `tool_calls: list[ToolCall] (min_length=1)`, `confidence: float ∈ [0.0, 1.0]`, `bt_xml: str | None`. `extra="forbid"`. (L66)

### `python/reasoner/src/openral_reasoner/null_reasoner.py`
_No-LLM stub satisfying the `Reasoner` Protocol (for plumbing tests; not a production fallback)._

- `class NullReasoner` — Emits a single-leaf `Plan` calling `default_skill_id` with confidence 1.0. Attributes: `plan_rate_hz`, `client = None`, `default_skill_id`. (L30)
  - `__init__(default_skill_id="noop", *, plan_rate_hz=5.0) -> None` — Initialise. (L63)
  - `plan(world_state, goal) -> Plan` — Return a context-free single-leaf `Plan`. (L74)

### `python/reasoner/src/openral_reasoner/tool_use.py`
_ADR-0018 F4 — typed LLM tool-use clients (direct-dispatch surface). CLAUDE.md §6.2 / §7.6 amended in the same PR. The direct typed `ReasonerToolCall` surface is the sole planner output._

- module constant `DEFAULT_SYSTEM_PROMPT: str` — Factual system prompt for the S2 reasoner: one-tool-per-tick semantics, goal fidelity, robot/scene-matched skill selection, the ADR-0044 go-see-then-act ladder — recall (recall_object, honouring Phase-4a 'approach BLOCKED') → navigate-to-approach (resolve_place / Nav2) → aim (the camera-aiming/look-at skill) → verify (locate_in_view, live vs remembered) → manipulate, each rung gated on its tool/skill being in the palette — progress evaluation, observe-but-never-bypass safety/e-stop handling, and exact-field-name discipline. Concrete deployments may override. (L78)
- module constant `OPENROUTER_BASE_URL: str` — `https://openrouter.ai/api/v1`; pre-filled when `PROVIDER=openrouter` so users don't have to memorise it. (L447)
- module constant `OLLAMA_BASE_URL: str` — `http://localhost:11434/v1`; pre-filled when `PROVIDER=ollama` (no auth). (L456)
- module constant `VLLM_BASE_URL: str` — `http://localhost:8000/v1`; pre-filled when `PROVIDER=vllm` (local `vllm serve`, no auth by default). (L464)
- module constants `GEMINI_BASE_URL` / `XAI_BASE_URL` / `DEEPSEEK_BASE_URL: str` — vendor OpenAI-compatible endpoints pre-filled when `PROVIDER=gemini` / `xai` / `deepseek` (issue #74; auth-required thin presets over `OpenAICompatibleToolUseClient`). (L474)
- module constant `_OPENAI_COMPATIBLE_PRESETS: dict[str, str]` — auth-required cloud presets (`openrouter` / `gemini` / `xai` / `deepseek`) → default base URL; the factory iterates this so the vendors reuse one branch. (L482)
- module constant `_LOCAL_OPENAI_COMPATIBLE_PRESETS: dict[str, str]` — local self-hosted no-auth presets (`ollama` / `vllm`) → loopback base URL; the factory serves both in one branch and gives them the longer 60 s cold-start timeout. (L494)
- module constant `SYSTEM_PROMPT_ENV_VAR: str = "OPENRAL_REASONER_SYSTEM_PROMPT"` (L343) — env var that overrides the base operating brief; honoured by `resolve_reasoner_system_prompt`.
- `render_robot_context_prompt(capabilities: RobotCapabilities | None, *, base_prompt=DEFAULT_SYSTEM_PROMPT) -> str` (L232) — Option B: append a deterministic `## THIS ROBOT` body-awareness block (embodiment tags, locomotion + navigate/no-navigate guidance, manipulation/sensing hardware, payload, control modes) to the system prompt. `None` returns `base_prompt` unchanged.
- `resolve_reasoner_system_prompt(capabilities: RobotCapabilities | None, *, env=None) -> str` (L346) — Compose the reasoner system prompt: base brief (`OPENRAL_REASONER_SYSTEM_PROMPT` override if non-empty, else `DEFAULT_SYSTEM_PROMPT`) + the `## THIS ROBOT` block. `env` is injectable for tests. Called by `ReasonerNode.on_configure`.
- `class ToolUseClient(Protocol)` (L393) — Attribute `model_id`; method `select_tool(*, context_text, palette, system_prompt=DEFAULT_SYSTEM_PROMPT) -> ReasonerToolCall`. Raises `ROSReasonerInvalidPlan` on bad discriminator / palette mismatch, `ROSPlanningError` on transport failure.
- `build_tool_use_client_from_env() -> ToolUseClient` (L511) — Factory reading `OPENRAL_REASONER_LLM_{PROVIDER, MODEL, API_KEY, BASE_URL}`. PROVIDER ∈ {`anthropic`, `openai-compatible`, `ollama`, `vllm`, `openrouter`, `gemini`, `xai`, `deepseek`}; `openrouter` / `gemini` / `xai` / `deepseek` are auth-required shortcuts on top of `OpenAICompatibleToolUseClient` that pre-fill the vendor base URL (`_OPENAI_COMPATIBLE_PRESETS`), `ollama` / `vllm` pin a local loopback URL with no key and a 60 s cold-start timeout (`_LOCAL_OPENAI_COMPATIBLE_PRESETS`). No cloud lock-in: open-core has no default.
- `class AnthropicToolUseClient` (L1029) — Anthropic SDK-backed client. `__init__(*, model_id, api_key, max_tokens=1024, timeout_s=10.0)`. Lazy-imports `anthropic`.
- `class OpenAICompatibleToolUseClient` (L1108) — OpenAI SDK-backed client pointed at any OpenAI-protocol endpoint (cloud OpenAI, local vLLM, Ollama-OpenAI). `__init__(*, model_id, api_key=None, base_url=None, timeout_s=10.0)`. Lazy-imports `openai`.
- `_tool_palette_to_anthropic_tools(palette) -> list[dict]` — Render the palette as Anthropic `tools` JSON Schema fragments. ADR-0022: when `palette.skills` is non-empty, emits one `execute_rskill__<slug>` tool per skill carrying a real NL description + the structured action/object/scene tags (the `rskill_id` field is dropped from each per-skill `input_schema` because the tool name is the authority). Empty / legacy palettes (`execute_rskill_ids` only) fall back to the original single-`execute_skill` tool with the ids in the description.
- `_decode_tool_payload(*, tool_name, arguments, palette) -> ReasonerToolCall` — Validate provider output against the union + palette. ADR-0022: per-skill tool names (`execute_rskill__<slug>`) are resolved back to the canonical `execute_skill` discriminator via `palette.skills` lookup, and the resolved `rskill_id` overrides anything the LLM provides.
- module constant `_PER_SKILL_TOOL_PREFIX: str = "execute_rskill__"` — prefix the decoder matches on to identify per-skill tool calls. (L627)
- module constant `_LLM_TOOL_NAME_MAX_LEN: int = 64` — Anthropic + OpenAI tool-name regex limit; long HF Hub ids are sha1-suffix-truncated to fit. (L630)
- `_skill_id_to_tool_name(rskill_id: str) -> str` — Slugify a `<owner>/<repo>` HF Hub id into a 64-char-max LLM tool name. Long ids get an 8-char sha1 suffix to stay unique post-truncation. (L633)
- `_format_skill_tool_description(entry: RSkillToolEntry) -> str` — Render the skill's id + description + actions + objects + scenes into the NL string the LLM scores. (L656)
- `_drop_property(schema: dict, name: str) -> dict` — Return a copy of a JSON Schema dict with `name` stripped from both `properties` and `required`. Used to drop `rskill_id` from per-skill `ExecuteRskillTool` schemas. (L938)

### `python/reasoner/src/openral_reasoner/palette.py`
_ADR-0018 F4 / ADR-0022 — closed-set `ToolPalette` + builder. Three tool variants (reload_gst_pipeline / lifecycle_transition / emit_prompt) are always available; `execute_skill` is gated by the installed-rSkill registry filtered by `RobotCapabilities` + license posture. ADR-0022: palette carries per-skill metadata (`RSkillToolEntry`), not just opaque ids — the LLM gets one tool per skill with description + action verbs + object/scene tags._

- `class RSkillToolEntry(BaseModel)` (L135) — Frozen per-skill record surfaced to the LLM as one tool. Fields: `rskill_id: str`, `description: str`, `actions: tuple[RSkillAction, ...]`, `objects: tuple[str, ...] = ()`, `scenes: tuple[str, ...] = ()`. Mirrored from the matching `RSkillManifest` fields at palette-build time.
- `class ContinuousDetectorEntry(BaseModel)` (L44) — ADR-0051. Frozen coverage record for a `mode: continuous` detector — surfaced to the LLM as *coverage* (not a tool) so it can read world state for tracked objects and reserve `locate_in_view` for the long tail. Fields: `rskill_id: str`, `description: str`, `objects: tuple[str, ...] = ()`, `scenes: tuple[str, ...] = ()`, `num_labels: int = 0` (compact characterisation, not the full label list).
- `class OnDemandDetectorEntry(BaseModel)` — ADR-0056. Frozen record for a `mode: on_demand` open-vocab locator surfaced as a selectable `locate_in_view` option (a prompt-able **read-only** tool, never an ExecuteSkill policy). Fields: `rskill_id: str`, `alias: str` (short selector the LLM passes as `LocateInViewTool.detector`), `description: str` (capability hint).
- `detector_alias(rskill_name) -> str` — ADR-0056. Short LLM-/operator-facing detector id: strips the `OpenRAL/` org + `rskill-` kind prefixes (`"OpenRAL/rskill-omdet-turbo-locator"` → `"omdet-turbo-locator"`). Single source of truth for the alias the reasoner routes on.
- `detector_service_segment(alias) -> str` — ADR-0056. ROS-safe service-namespace segment for an alias (hyphens → underscores), so the locate service lives at `/openral/perception/<segment>/locate_in_view`.
- `locate_in_view_service(detector, *, default="") -> str` — ADR-0056. Resolves the `locate_in_view` service for a (possibly empty) selector: empty `detector` → `default`; empty resolved alias → the legacy `/openral/perception/locate_in_view` (single-detector back-compat); else the namespaced service. Shared by the reasoner dispatch and the deploy launch (which names each locator node's service).
- `class ToolPalette(BaseModel)` (L174) — Frozen palette presented to the LLM each tick. Fields: `skills: tuple[RSkillToolEntry, ...] = ()` (ADR-0022 primary surface), `execute_rskill_ids: frozenset[str] = frozenset()` (back-compat — auto-derived from `skills` via the `_derive_execute_rskill_ids` model-validator), `sensor_ids: frozenset[str] = frozenset()`, `node_ids: frozenset[str] = frozenset()`, `continuous_detectors: tuple[ContinuousDetectorEntry, ...] = ()` (ADR-0051 — `mode: continuous` detectors for the active robot; coverage, not tools), `spatial_memory_available: bool = False` (ADR-0039 — gates the two read-only `recall_object` / `resolve_place` query tools; off unless the reasoner_node has a SpatialMemory backend wired), `detector_available: bool = False` (ADR-0043 — gates `locate_in_view`), `on_demand_detectors: tuple[OnDemandDetectorEntry, ...] = ()` (ADR-0056 — selectable locator options for `locate_in_view`), `scene_query_available: bool = False` (ADR-0047 — gates `query_scene`; independent of `detector_available`), `memory_available: bool = False` (ADR-0071 §3 — gates the self-maintained-memory tools `memory_write` (write) + `memory_search` (read-only archival recall); off unless the reasoner_node has a `MEMORY.md` wired via `memory_md_path`). Cross-validator `_check_skills_match_ids` rejects callers that pass both `skills` and `execute_rskill_ids` with disagreeing ids.
- `build_tool_palette(*, installed_skills, robot_capabilities, sensor_ids=(), node_ids=(), commercial_deployment=False, spatial_memory_available=False, detector_available=False, scene_query_available=False, task_progress_available=False, memory_available=False) -> ToolPalette` — A skill is included iff role=s1, kind≠detector (detector rSkills are perception producers, not ExecuteRskill-dispatchable — ADR-0035/0037), capability flags satisfied, embodiment tags intersect, and (when commercial) license allows commercial use. A `mode: continuous` detector (ADR-0051) is instead collected into `continuous_detectors` (coverage for the LLM, never an ExecuteSkill tool); a `mode: on_demand` detector (ADR-0056) is collected into `on_demand_detectors` (selectable `locate_in_view` options via `detector_alias`, never an ExecuteSkill tool). Emits `RSkillToolEntry` records (manifest `description`/`actions`/`objects`/`scenes` mirrored in) in stable id-sorted order so the LLM tool schema is deterministic. `spatial_memory_available` forwards the read-only `recall_object`/`resolve_place` tools (ADR-0039); `detector_available` forwards the read-only `locate_in_view` tool (ADR-0043); `scene_query_available` forwards the read-only `query_scene` tool (ADR-0047); `memory_available` forwards the `memory_write` + `memory_search` tools (ADR-0071 §3). All are `ToolPalette` fields gated in `tool_use` so the LLM only sees a tool when its dispatcher is wired; `detector_available` and `scene_query_available` are independent (localization vs scene-state reasoning). The reasoner_node dispatches `query_scene` via `_dispatch_query_scene` → `/openral/perception/query_scene` and re-prompts with the answer (frame_id `scene_vlm`).

### `python/reasoner/src/openral_reasoner/spatial_query.py`
_ADR-0039 Phase 2 — read-only spatial-memory query bridge: maps a `RecallObjectTool` / `ResolvePlaceTool` to an ADR-0038 query, runs it against an injected backend, and renders an LLM-readable result for the prompt cascade. Layer-4 module; does not import `openral_world_state` (backend is duck-typed)._

- `class SpatialMemoryQuerier(Protocol)` — Read-only query surface (`recall_object(query, *, now_ns) -> RecallObjectResult`; `resolve_place(query, *, from_node_id=None) -> ResolvePlaceResult`; `to_scene_graph() -> SceneGraph` — immutable snapshot for telemetry/dashboard); structurally satisfied by `openral_world_state.SpatialMemory`.
- `SpatialQueryTool: TypeAlias` — `RecallObjectTool | ResolvePlaceTool` (the read-only ReasonerToolCall variants this bridge dispatches).
- `recall_object_tool_to_query(call) -> RecallObjectQuery` / `resolve_place_tool_to_query(call) -> ResolvePlaceQuery` — tool → ADR-0038 query mappers.
- `format_recall_object_result(query_text, result, *, blocked_node_ids=frozenset()) -> str` / `format_resolve_place_result(reference, result) -> str` — render results as LLM-readable text (misses reported as text, never a fabricated pose). `blocked_node_ids` (ADR-0044 Phase 4) renders a match whose approach failed grid refinement as "approach BLOCKED on the occupancy grid" instead of a pose.
- `run_spatial_query(call, querier, *, now_ns, from_node_id=None, refine_approach=None) -> str` — execute a read-only tool call and render the result; catches `ROSObjectNotInMemory` → "not in memory" message. `refine_approach` (ADR-0044 Phase 4, `ApproachRefiner` — duck-typed like the querier so this L4 module never imports L2) is applied to every `recall_object` match's approach viewpoint before rendering; a `None` from the refiner marks the match BLOCKED. Thin wrapper over `run_spatial_query_detailed` returning only `.text`.
- `class SpatialQueryOutcome(NamedTuple)` — `(text: str, found: bool)`. `found` is `True` when `recall_object` returned ≥1 match (in memory, even if every approach is grid-BLOCKED) or `resolve_place` resolved the reference; `False` on a miss. Drives the reasoner's recall→`locate_in_view` escalation (ADR-0043/0056).
- `run_spatial_query_detailed(call, querier, *, now_ns, from_node_id=None, refine_approach=None) -> SpatialQueryOutcome` — same as `run_spatial_query` but also reports whether the query matched, so the node can escalate a miss to a live perception check without re-parsing the rendered text.
- `ApproachRefiner` (TypeAlias = `Callable[[ApproachViewpoint, tuple[float, float, float]], ApproachViewpoint | None]`) — the occupancy-grid refinement callback contract; the reasoner node wires `refine_approach_pose` over its latched `/map` subscription. (ADR-0044)

### `python/reasoner/src/openral_reasoner/active_search.py`
_ADR-0039 §3 Phase 4 — bounded active object search over the scene graph (pure-Python, `openral_core` only)._

- `class SearchBudget(BaseModel)` — frozen; `max_candidates` (1–50), `max_attempts` (1–50). The bound.
- `class SearchCandidate(BaseModel)` — `place_node_id, goal: Pose6D, open_container_id: str | None, reason, rank ∈ [0,1]`.
- `plan_active_search(graph, *, target_text, budget) -> list[SearchCandidate]` — ranked frontier of places to check (occluding containers first, then containers, then places), truncated to `budget.max_candidates`; `[]` when nowhere to search (→ human-handoff). Semantic prioritization among candidates is the LLM's (priors).
- `class SearchProgress` — attempt counter against a `SearchBudget`: `record_attempt() -> bool` (True while budget remains), `attempts`, `exhausted`, `reset()`. The runaway bound.
- `format_search_frontier(candidates, target_text) -> str` — LLM-readable frontier text (empty → "hand off to a human").

### `python/reasoner/src/openral_reasoner/context.py`
_ADR-0018 F4 — `ContextRenderer` builds the structured **text** snapshot the LLM consumes per tick (no pixels in v1)._

- module constant `DEFAULT_BUFFER_SIZE: int = 8` — Rolling buffer capacity per category. (L50)
- module constant `DEFAULT_PROMPT_PRIORITY: int = 10` — Default operator-prompt priority; matches `openral_prompt_router.DEFAULT_SOURCES` auto-cascade priority. Human sources stamp 100 onto `metadata_json` so they drain first (ADR-0018 §3.F10).
- `class FailureEventRecord` (frozen dataclass, L48) — Failure-buffer entry; fields `source, kind, severity, evidence_json, rskill_id, trace_id, stamp_ns`.
- `class PerceptionEventRecord` (frozen dataclass, L62) — Perception-buffer entry; fields `kind, text, metadata_json, stamp_ns`.
- `class PromptRecord` (frozen dataclass, L72) — Operator-prompt-buffer entry; fields `text, metadata_json, stamp_ns, priority=DEFAULT_PROMPT_PRIORITY`. The `priority` field is filled in by `append_prompt` from `metadata_json["priority"]` when the record was constructed with the default sentinel.
- `render_robot_self_model(description: RobotDescription) -> str` (L222) — ADR-0071 Decision 2.1 (EMOS "Robot Resume"): a deterministic static self-model block — `name`/embodiment, `dof`, `end_effectors`, `locomotion`, `payload_kg`, capability flags, `cameras` (with FOV), `control_modes` — so the LLM can judge reach/view feasibility before dispatch. Set on a renderer via `ContextRenderer.set_robot_model`; rendered as the `## ROBOT` section.
- `render_playbooks_block(entries: list[tuple[str, str]]) -> str` — ADR-0071 Decision 1 / Phase 3: renders the `## PLAYBOOKS` *system-prompt* block from `(name—trigger, PLAYBOOK.md body)` entries. `reasoner_node._collect_playbooks_block` gathers installed, capability-matched `kind: playbook` rSkills and appends the block to the system prompt at seed time; returns `""` (no-op) when none match. Playbooks guide decisions only — every motion still goes through `execute_rskill` + the safety kernel.
- `class MemoryEntry` / `class MemoryStore` (`openral_reasoner.memory`) — ADR-0071 §3 / Phase 4b: the self-maintained `MEMORY.md` file model (persistent *semantic* memory — preferences, lessons, home facts, object-location log, open tasks; complementary to the ADR-0038 *geometric* scene graph). `MemoryStore.from_markdown` / `to_markdown` round-trip the human-editable file; `to_context_block(cap=None)` renders the `## MEMORY` section — Phase 5: when `cap` is set and the store exceeds it, only the top-`cap` entries by importance then recency (current over `stale`) render, with a "use memory_search to recall" footer (bounded always-on context); `apply(op, section, content, importance, target, now)` does an explicit `add`/`update`/`supersede`/`delete` (Mem0 + Zep supersession — `supersede` marks the prior `stale` but keeps it as a search hint) returning any entry to archive; `consolidate() -> list[MemoryEntry]` (Phase 5) drops exact `(section, content)` duplicates keeping the highest-ranked (`_rank`: current > stale, then importance, then recency), returning the removed copies to archive (Mem0 ADD-merge); `search(archive, query, section, limit)` ranks archived entries (MemGPT recall). Advisory only. `reasoner_node._maybe_load_memory` loads it from the `memory_md_path` param into `ContextRenderer.set_memory_block` at configure (read path) **and** loads the `<MEMORY.md>.archive.jsonl` recall log + flips `ToolPalette.memory_available` so the write/search tools are offered (Phase 4c). The `## MEMORY` block is rendered via `_render_memory_block` under the `memory_context_cap` param (Phase 5; 0 = off). Writes flow through `_dispatch_memory_write` (apply → archive the displaced entry → `consolidate()` paging duplicates to the archive → persist `MEMORY.md` → re-render → confirm); recall through `_dispatch_memory_search` (Phase 4c).
- `class ExecutionEventRecord` (frozen dataclass) — ADR-0071 §2.2 execution-feedback buffer entry; fields `rskill_id, outcome ("ok"|"failed"), summary, reflection (§2.3 hint, failures only), stamp_ns`.
- `reflect_on_failure(outcome_state, detail) -> str` / `reflect_on_retry_cap(tool, cap) -> str` — ADR-0071 §2.3 (Reflexion): deterministic one-line strategy hints (no LLM call) turning a raw failure / exhausted retry ladder into a "change approach" cue for the next tick.
- `class ContextRenderer` (L290) — Stateful renderer. Methods: `set_robot_model(robot_model: str | None)` (ADR-0071 §2.1 — sets/clears the static `## ROBOT` self-model; static config, does NOT bump `seq`), `set_memory_block(memory_block: str | None)` (ADR-0071 §3 — sets/clears the `## MEMORY` block from the MEMORY.md store; does NOT bump `seq`), `append_execution` (ADR-0071 §2.2 — success+failure outcomes into the `## EXECUTION` section, bumps `seq` so feedback wakes an idle heartbeat), `append_failure`, `append_perception`, `append_prompt` (priority-ordered insert; buffer-evicts the lowest-priority oldest entry on overflow — every append also bumps the monotonic `seq` counter), `render(*, world_state) -> str`, `drain_prompts() -> tuple[PromptRecord, ...]` (pull-once, priority-desc + arrival-asc order; does NOT bump `seq`); properties `failures`, `perception_events`, `prompts`, `seq` (mutation counter consumed by `ReasonerCore` to short-circuit a heartbeat tick when no event has arrived since the last successful tick — ADR-0018 amendment 2026-05-25 §2). The `## WORLD_STATE` block (`_render_world_state`) renders joint_state / ee_poses / battery / diagnostics and, since #14 (2026-06-12), a `scene_objects[<frame>]: label@(x,y,z), …` line from `WorldState.detected_objects` (deduped by label, first-seen pose) — so the LLM sees the lifted object labels (e.g. `bread`) and can map a goal noun (`baguette`) onto them with its own semantics rather than only learning a name is "not in memory".
- `_summarise_evidence_json(payload) -> str` — Decode the FailureEvidence discriminated union and produce a one-line summary. (L625)
- `_extract_priority(metadata_json) -> int` — Parse a top-level `priority` field out of a PromptStamped's metadata; returns `DEFAULT_PROMPT_PRIORITY` on missing / malformed / non-int payload.

### `python/reasoner/src/openral_reasoner/core.py`
_ADR-0018 F4 — `ReasonerCore`, the transport-agnostic orchestrator. The ROS-side `reasoner_node` wraps this with rclpy._

- `class ReasonerTickResult` (frozen dataclass, L26) — Tick outcome; fields `tool_call: ReasonerToolCall | None`, `error: ROSPlanningError | None`, `elapsed_s: float`, `suppressed_reason: str` (one of `""`, `"min_interval"`, `"heartbeat_idle"`, `"retry_cap"`, `"palette_empty"`), `traceparent: str | None` (W3C traceparent captured inside the active `reasoner.tick` span — `None` when no real `TracerProvider` is installed).
- `class ReasonerCore` (L72) — Orchestrator. Methods: `tick(*, world_state, renderer, palette, force=False, tier="heartbeat") -> ReasonerTickResult`. ADR-0018 §4 min-interval (100 ms) + per-kind retry cap (default 3) enforced here. Heartbeat-idle short-circuit (ADR-0018 amendment 2026-05-25 §2): when `force=False` and `renderer.seq` matches the seq at the last successful tick, the LLM call is suppressed with `suppressed_reason="heartbeat_idle"`. Palette-empty short-circuit prevents wasted LLM calls **when `force=False`** — a `force=True` tick (event preemption from `SEVERITY_FAIL` FailureTrigger, `SEVERITY_WARN` on `/openral/failure/safety`, or new operator prompt) bypasses the min-interval gate, the heartbeat-idle gate, AND the palette-empty gate so the LLM can pick `EmitPromptTool` to escalate even on a bare reasoner. The retry-cap gate still applies under `force=True`. The `tier` kwarg (`"A"`/`"B"`/`"C"`/`"D"`/`"heartbeat"`) is recorded verbatim on the span as `reasoner.tier` for dashboard filtering — observability only; per-tier preemption thresholds live in `ReasonerNode._FAILURE_TIER_FOR_SOURCE`. Wraps the per-tick work in `reasoner_span` (`openral_observability`) so the LLM call lives under a `reasoner.tick` OTel span with `reasoner.{model, tick.idx, tool, rskill_id, suppressed_reason, error_kind, force, tier}` attributes (ADR-0018 §6).

### `python/reasoner/src/openral_reasoner/critic_watchdog.py`
_Tier-C critic progress-stall watchdog — default decision core for the reserved `/openral/failure/critic` source (observability audit P1 R3). Pure logic, import-safe (no rclpy); emits the real `openral_core.CriticEvidence`. Source-agnostic: any reward model emitting a higher-is-better scalar (Robometer ADR-0057, a future SARM, a success classifier) drives the same watchdog. The critic producer node subscribes to the generic `/openral/critic/score` topic (`openral_msgs/CriticScore`), routes samples through `CriticWatchdogGroup`, and on a non-`None` return publishes via `FailureBusPublisher(node, FailureSource.CRITIC)` (`kind=KIND_CRITIC, severity=SEVERITY_FAIL`); the reasoner_node maps that FAIL event to a forced Tier-C tick._

- `class CriticWatchdog` (L80) — Progress-stall state machine. `__init__(critic_id: str, threshold: float, stall_patience: int, *, min_delta: float = 0.0)` (raises `ValueError` on `stall_patience < 1` or `min_delta < 0`). Methods: `observe(score: float) -> CriticEvidence | None` — fires one `CriticEvidence(critic_id, score, threshold)` after `stall_patience` consecutive below-threshold, non-improving (`score <= best + min_delta`) observations, then **latches** (`None`) until progress (`score > best + min_delta`) or recovery (`score >= threshold`) clears it; `reset() -> None` — clears running best, stall counter, and latch (call on reasoner context shift, mirroring `ReasonerCore.reset_kind_streak`). Read-only properties `critic_id`, `threshold`, `stall_patience`, `min_delta`.
- `class CriticWatchdogGroup` (L214) — Multiplexer keying one `CriticWatchdog` per `critic_id` so multiple/future reward models (Robometer + SARM + …) share the `/openral/failure/critic` source independently. `__init__(*, stall_patience: int, min_delta: float = 0.0)`. Methods: `observe(*, critic_id: str, score: float, threshold: float) -> CriticEvidence | None` — lazily creates a watchdog per `critic_id` (binding `threshold` on first sight, held stable) and delegates; `known_critics() -> frozenset[str]`; `reset(critic_id: str | None = None) -> None` — drop one critic's watchdog (rebinds its threshold) or all. Read-only properties `stall_patience`, `min_delta`.

### `python/wam/src/openral_wam/protocol.py`
_World Action Model Protocol (CLAUDE.md §6.3)._

- `class WorldModel(Protocol)` — Generative simulator used by the planning layer for the three integration patterns (gating / failure anticipation / replanning). Attribute: `max_horizon`. Method: `rollout(world_state, action_chunk, horizon) -> Rollout` — predict `horizon` steps of future state; raises `ROSConfigError` (horizon exceeds max) / `ROSInferenceTimeout` (budget exceeded). (L31)

### `python/wam/src/openral_wam/rollout.py`
_Pydantic v2 schema for a WAM's predicted trajectory._

- `class Rollout(BaseModel)` — Predicted trajectory from one `WorldModel.rollout` call. Fields: `predicted_states: list[WorldState] (min_length=1)`, `predicted_rewards: list[float] | None`, `horizon: int (>0)`, `latency_ms: float (≥0.0)`, `confidence: float ∈ [0.0, 1.0]`. `extra="forbid"`. (L24)

### `python/wam/src/openral_wam/null_wam.py`
_Identity stub satisfying the `WorldModel` Protocol (for plumbing tests; not a production fallback)._

- `class NullWorldModel` — Returns `horizon` copies of the input `WorldState`, no rewards, 0.0 ms latency, confidence 1.0. Attribute: `max_horizon`. (L27)
  - `__init__(max_horizon=16) -> None` — Raises `ValueError` if `max_horizon <= 0`. (L54)
  - `rollout(world_state, action_chunk, horizon) -> Rollout` — Replays the input state. Raises `ValueError` for `horizon ∉ (0, max_horizon]`. (L60)

### `packages/openral_safety/openral_safety/supervisor_node.py`
_Lifecycle node skeleton; reserves the supervisor node name and topic surface for the future C++ kernel (CLAUDE.md §6.1 Layer 6, §7.7). No enforcement logic._

- `class SafetySupervisorNode(LifecycleNode)` — Skeleton lifecycle node. Every transition callback returns `SUCCESS`. (L611)
  - `__init__(node_name="openral_safety_supervisor") -> None` — Initialise; logs a "skeleton no-op" line so the supervisor's presence in the graph is visible. (L99)
  - `on_configure(state) -> TransitionCallbackReturn.SUCCESS` (L149)
  - `on_activate(state) -> TransitionCallbackReturn.SUCCESS` (L224)
  - `on_deactivate(state) -> TransitionCallbackReturn.SUCCESS` (L232)
  - `on_cleanup(state) -> TransitionCallbackReturn.SUCCESS` (L239)
  - `on_shutdown(state) -> TransitionCallbackReturn.SUCCESS` (L262)
- `main(args=None) -> int` — Entry point for `ros2 run openral_safety supervisor_node`. (L614)

### `packages/openral_safety/openral_safety/envelope_loader.py`
_Pydantic → C++ kernel ROS-param bridge (ADR-0020 PR-K; ADR-0030 collision)._

- `compute_intersection(robot, skill=None) -> EnvelopeIntersection` — Robot ceiling ∩ optional skill envelope; rejects (never clamps) a skill that loosens the ceiling.
- `kernel_params_from_envelope(envelope) -> dict[str, object]` — Canonical scalar/AABB envelope → kernel ROS-param dict.
- `collision_params_from_description(robot, *, margin_m=0.0) -> dict[str, object]` — ADR-0030. Flatten `collision_geometry` + `allowed_collision_pairs` + the kinematic chain (joint `origin_xyz/rpy/axis`) into the kernel's per-capsule collision params (`collision_capsule_link` + parallel radius/half-length/origin arrays, link-level ACM), topologically ordered. `{"self_collision_enabled": False}` when no geometry. Manifest-source adapter.
- `ee_link_index_from_collision_params(params) -> int` — ADR-0040 Phase 3. Pick the predictive-Cartesian EE control link (the kinematically deepest collision link) for the kernel's Jacobian look-ahead; `-1` when no collision model (predictive disabled, reactive floor only). Mis-identification is bounded by the reactive check.

### `packages/openral_safety/openral_safety/mjcf_lowering.py`
_Offline MJCF → kernel collision-params lowering (ADR-0030); imports `mujoco` lazily._

- `lower_collision_params(model, joint_names, *, margin_m=0.0) -> dict[str, object]` — Lower a compiled `mujoco.MjModel` to the kernel's collision params from the full kinematic tree (fixed mounts + floating base): per-link origins from the body tree, **every** collidable primitive per body as a capsule (cylinder→capsule, box→bounding-sphere; mesh/plane skipped), `dof_index` assigned by movable-joint **order** (the i-th hinge/slide joint → manifest column i, capped at `len(joint_names)`; MJCF joint *names* are not consulted — they differ from the manifest, e.g. `Rotation` vs `shoulder_pan`, so name-matching silently froze every link's FK at rest), ACM = parent↔child + MJCF excludes + a neutral-pose overlap sweep (the MoveIt "disable always-in-collision pairs" rule under the kernel's own capsule approximation).

### `packages/openral_safety/openral_safety/urdf_lowering.py`
_Offline URDF(+SRDF) → manifest collision-model lowering tool (ADR-0030); lazy-imports `yourdfpy` / `trimesh` (the `[lowering]` group). Populates `robot.yaml`'s `collision_geometry` + `allowed_collision_pairs` (the hand-reviewable manifest path), distinct from `mjcf_lowering` (the runtime MJCF path)._

- `parse_srdf_disabled_pairs(srdf_path) -> set[frozenset[str]]` — Parse a MoveIt SRDF's `<disable_collisions>` rows into unordered link pairs (the ACM).
- `fit_capsule_to_vertices(vertices) -> tuple[CapsuleShape, tuple[float×6]]` — PCA bounding capsule (segment along +Z) containing every vertex — a conservative over-approximation so the safety check never under-covers; returns the shape + link-frame `origin_xyz_rpy` (kernel's rpy convention, inverse of `mjcf_lowering._rpy_to_mat`).
- `lower_link_geometry(urdf_path) -> list[LinkCollisionGeometry]` — One conservative capsule/sphere per URDF link with a `<collision>` (box→8 corners, cylinder→cap rims, sphere→exact `SphereShape`, mesh→`trimesh` vertices PCA-fit), vertices first transformed into the link frame by the collision `<origin>`.
- `acm_for_geometry(urdf_path, geoms, *, srdf_path=None, n_samples=2000, seed=20260610, margin_m=0.0) -> set[frozenset[str]]` — The ACM for a specific per-link capsule geometry (the geometry the kernel will actually load). `ACM = adjacent ∪ always-colliding(capsule) ∪ [SRDF-disabled if srdf_path else never-colliding(capsule)]`. The always-colliding term adds the capsule-junction pairs a mesh-based SRDF omits (e.g. a short link making skip-one neighbours' capsules overlap) — without them the capsule kernel false-E-stops every step. Deterministic under the pinned seed.
- `sample_acm_from_urdf(urdf_path, *, n_samples=2000, seed=20260610, margin_m=0.0) -> set[frozenset[str]]` — No-SRDF fallback: lowers the URDF's own collision geometry and runs `acm_for_geometry` without an SRDF. Verified conservative against URDF-lowered (mesh-bounding) capsules — its disabled set is a subset of the precise-mesh SRDF's, never false-permissive.
- `lower_robot(robot, *, srdf_path=None, acm_only=False, geometry_only=False) -> LoweredCollisionModel` — Top-level entry. ACM source precedence: explicit `srdf_path` → `robot.srdf_path` → URDF sampling fallback; ACM scoped to links carrying geometry. Generated geometry is scoped to the manifest's kinematic chain (no orphan URDF links); `joint_fk` is lowered too (unless `acm_only`). `acm_only`/`geometry_only` restrict output so hand-tuned safety geometry isn't churned. Raises `ValueError` if `robot.urdf_path` is unset/unresolvable (a `robot_descriptions:<module>` xacro form is accepted).
- `lower_joint_fk(robot, urdf_ref) -> dict[str, tuple[xyz, rpy, axis]]` — Per-manifest-joint forward kinematics (origin + axis) read from the URDF, matched to manifest joints by `child_link`. The kernel needs these to place the link capsules (ADR-0030). Unmatched joints (synthetic gripper / base DoF) are omitted.
- `lower_robot_from_mjcf(robot, *, n_samples=2000, seed=20260610, margin_m=0.0, manifest_dir=None) -> LoweredCollisionModel` — MJCF backend for robots with no URDF whose collision is meshes (`mjcf_lowering`'s primitive path skips them), e.g. bimanual `openarm`. Keeps the manifest's hand-authored capsules; lowers joint FK (the MJCF parent→child transform at rest) + the conservative ACM (mujoco-FK sweep). Manifest↔MJCF link-name divergence is reconciled via `sim_joint_name`. Lazy-imports `mujoco` + `openral_core.assets.resolve_asset` (resolves `robot.assets.mjcf`, honouring `manifest_dir` for `file:` refs; ADR-0058). `acm_source="mjcf"`.
- `select_lowering(robot, *, manifest_dir=None) -> LoweringSource` — Provenance-correct routing (ADR-0058 §5): `"srdf"` when SRDF+URDF both present (mesh-proven ACM), `"sampling"` when a URDF with usable collision meshes but no SRDF, `"mjcf"` when no usable URDF geometry but an MJCF exists (e.g. `openarm`, whose vendored URDF's `package://` collision meshes don't resolve → 0 geometry). Replaces the naive `urdf if assets.urdf else mjcf` guess that wrongly sent `openarm` to the empty URDF path. Raises `ROSConfigError` when no lowerable asset.
- `lower_robot_auto(robot, *, acm_only=False, geometry_only=False, manifest_dir=None) -> LoweredCollisionModel` — Single dispatch over `select_lowering` → `lower_robot` (srdf/sampling) or `lower_robot_from_mjcf` (mjcf). The one entry the CLI (`openral collision lower|check`) and the byte-identical regression both call, so routing can never diverge between what's committed and what's verified.
- `LoweringSource` — `Literal["srdf", "sampling", "mjcf"]`; the source `select_lowering` resolves to (matches `LoweredCollisionModel.acm_source`).
- `class LoweredCollisionModel` — Frozen dataclass result: `collision_geometry`, `allowed_collision_pairs` (sorted tuples), `acm_source` (`"srdf"`|`"sampling"`|`"mjcf"`), `srdf_path`, `joint_fk` (per-joint FK for onboarding).

### `packages/openral_safety/openral_safety/cumotion_config.py`
_ADR-0065 — derive a cuRobo (cuMotion) robot-config from the **same** lowered collision geometry the safety kernel checks (ADR-0030), so plan-time and kernel-time collision stay consistent. Pure module; reuses `urdf_lowering._capsule_segment_radius`._

- `class CuMotionSphere` — Frozen dataclass: `center` (link-frame `(x, y, z)`), `radius` — one cuRobo collision sphere.
- `capsule_to_spheres(p0, p1, radius, *, count) -> list[CuMotionSphere]` — Sample `count` spheres evenly along the segment `p0`→`p1` (endpoints inclusive for `count >= 2`; midpoint for `count == 1`).
- `spheres_for_capsule(shape) -> int` — Sphere count to tile a lowered capsule with centres ≤ one radius apart (`ceil(L/r)+1`); `1` for a sphere / zero-length capsule.
- `link_collision_spheres(geom, *, count=None) -> list[CuMotionSphere]` — Lower one `LinkCollisionGeometry` to cuRobo spheres in its link frame (reuses the kernel's capsule→segment math).
- `actuated_joint_names(robot) -> list[str]` — Single-DOF movable joint names (revolute/prismatic/continuous), in manifest order — the cuRobo `cspace.joint_names`.
- `render_cumotion_config(robot, model) -> str` — Render a cuRobo `robot_cfg` YAML fragment (base_link, collision_spheres, self_collision_ignore from the ACM, cspace.joint_names) with a generated-provenance header. `retract_config` / accel-jerk limits are planner tuning, left for ADR-0065 Phase 3.

### `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py`
_ADR-0018 F4 — `reasoner_node` lifecycle wrapper. Thin rclpy shell around `openral_reasoner.ReasonerCore`._

- module constants `_FAILURE_SOURCES`, `_PERCEPTION_KINDS` — closed sets from ADR-0018 §3 (`hal/sensor/rskill/safety/wam/critic` (the `rskill` suffix replaced `skill` on 2026-05-25 — ADR-0018 amendment §5) and `motion/objects/ocr/scene_change`). (L203)
- module constants `_KIND_TIMEOUT`, `_KIND_CONTROLLER`, `_SEVERITY_WARN`, `_SEVERITY_FAIL` — IDL-mirror constants for `openral_msgs/FailureTrigger`. Kept inline rather than importing the `openral_observability.failure_bus` helper so the reasoner emits a `FailureTrigger` without dragging the rate-limiter into the dispatch path (the reasoner publishes O(1) events per skill goal, not a stream). (L211)
- module constants `_EXECUTE_SKILL_SERVER_PROBE_S`, `_LIFECYCLE_SERVER_PROBE_S` — 100 ms `wait_for_server` / `wait_for_service` probes so an absent F1 server / lifecycle peer can't block the executor thread. (L219)
- module constant `_FAILURE_TIER_FOR_SOURCE: dict[str, str]` — ADR-0018 2026-05-25 amendment trigger taxonomy. Greppable map of each `/openral/failure/<source>` to its tier: `safety → "A"`, `hal/sensor/rskill/wam → "B"`, `critic → "C"`. Used by `_on_failure` to stamp `reasoner.tier` on the OTel span — observability only; the per-source preemption threshold (`SEVERITY_WARN` for safety, `SEVERITY_FAIL` for everything else) is decided inline in the same callback.
- (the former module-local `_SIM_EXECUTABLE_CONTROL_MODES` frozenset was removed 2026-06-04; the `hal_mode == "sim"` gate now imports the canonical `openral_core.SIM_EXECUTABLE_CONTROL_MODES`, trimmed to the six packer-implemented modes — see the Layer-0 core entry. ADR-0036 amendment 2026-06-04.)
- `def _required_control_modes(manifest: RSkillManifest) -> set[ControlMode]` (L272) — ADR-0036 pure helper for the deploy-path palette gate. Reads `action_contract` by specificity: `None → set()` (no action constraint); `representation` set → `control_modes_for_representation(...)`; `slots` set → each non-`None` slot's `control_mode`; bare `dim` (legacy) → `{JOINT_POSITION}`.
- `def _action_executable(manifest: RSkillManifest, description: RobotDescription, hal_mode: str) -> bool` (L303) — ADR-0036 pure helper. `True` when every `_required_control_modes(manifest)` is in the executable set: `openral_core.SIM_EXECUTABLE_CONTROL_MODES` for `hal_mode == "sim"`, else `description.capabilities.supported_control_modes` (coerced to `ControlMode` both sides so an enum-member or raw-`"joint_position"`-string deserialisation compares equal). Empty required set → `True`.
- `class ReasonerNode(LifecycleNode)` (L343) — Lifecycle node. `__init__(*, node_name="openral_reasoner", tick_hz=0.2, client=None, palette=None, robot_capabilities=None, commercial_deployment=False, spatial_memory=None)`. ADR-0039 Phase 2b: `spatial_memory` is an optional read-only `SpatialMemoryQuerier` backend (an ADR-0038 `SpatialMemory`); when supplied the palette's `spatial_memory_available` is set (the `recall_object` / `resolve_place` tools are offered) and the rebuild path threads it through. Deployment wiring: the `spatial_memory_path` ROS parameter (default `""`) loads a persisted scene graph as that backend at `on_configure` when no backend was injected (see `_maybe_load_spatial_memory`); the `spatial_memory_ingest` ROS parameter (default `false`) auto-creates an empty backend and folds each `WorldState.detected_objects` snapshot into it on tick (ADR-0038 live dynamic memory from the ADR-0035 producer). ADR-0071 Decision 3b — the deploy memory bundle: `sim_e2e.launch.py` forwards `memory_md_path` (loads `MEMORY.md` + enables the memory tools) and brings up a standalone `nav2_map_server` from a saved `map.yaml` when its `map_path` arg is set and SLAM is off (latches `/map`, which the reasoner consumes into its ADR-0044 `_occupancy_grid` via `occupancy_map_topic`); with SLAM on, `map_path` is ignored (SLAM owns `/map`). ADR-0036: the `hal_mode` ROS parameter (default `"sim"`) selects the action-mode palette gate (`_action_executable`) the skill-registry refresh applies. `tick_hz` is the heartbeat rate (default 0.2 Hz = one tick every 5 s; was 5.0 pre-2026-05-25 amendment to ADR-0018 — the reasoner is now event-driven with a slow heartbeat). The two refresh-kwargs (added in the F4 contract-closure follow-up) drive the `/openral/skill_registry_changed` refresh path: without `robot_capabilities` the callback logs a warning and leaves the palette alone (an empty-capabilities refresh would risk dispatching incompatible skills).
  - `on_configure` — Build `ToolUseClient` from env if not injected, attach subscribers to `/openral/world_state_slow` + 6 failure topics + 4 perception topics + `/openral/prompt` + `/openral/skill_registry_changed`, create the `/openral/prompt` publisher + `/openral/failure/rskill` publisher + `/openral/execute_rskill` action client. Reads the `vram_lifecycle_peers` ROS parameter (default `[]`) into `_vram_lifecycle_peers` — ADR-0050 GPU peers auto-deactivated before each `execute_rskill` and reactivated after (the deploy launch sets it to `openral_ros_image_detector` when `--enable-object-detector`).
  - `on_activate` — Arm the periodic tick timer at `tick_hz`.
  - `on_deactivate` — Cancel the tick timer (subscriptions remain attached).
  - `on_cleanup` — Tear down pending skill-goal deadline timers, destroy the action client, drop cached lifecycle clients.
  - `_on_failure(source, msg)` — Append a `FailureEventRecord` to the renderer; preempt the next tick per the ADR-0018 amendment 2026-05-25 trigger taxonomy — Tier A (`source == "safety"`) preempts on `severity ≥ SEVERITY_WARN`, Tier B/C (`hal`, `sensor`, `rskill`, `wam`, `critic`) preempts on `severity ≥ SEVERITY_FAIL`.
  - `_on_tick(*, force=False, tier="heartbeat")` — Invoke `ReasonerCore.tick(..., tier=tier)`; route the resulting `ReasonerToolCall` via `_dispatch(call, traceparent=result.traceparent)`. Suppressed ticks log at DEBUG (`min_interval`, `heartbeat_idle`) or WARN (`retry_cap`) per their operational signal-to-noise. The `tier` arg is passed through from the preempting callback (`A` from `_on_failure(source="safety")`, `B`/`C` from other failure sources, `D` from `_on_prompt`) and lands on the `reasoner.tick` OTel span as `reasoner.tier`.
  - `_on_skill_registry_changed(msg)` — ADR-0018 §4 palette refresh. Walks `rSkill.list_installed()`, loads each entry's `manifest_path` into a real `RSkillManifest`, runs `build_tool_palette(...)` against the active `robot_capabilities` + `commercial_deployment` flag, installs the result via `set_palette`. `openral_rskill` is lazy-imported to keep the node cheap to import.
  - `_dispatch(call, *, traceparent=None)` — Routing-only switch over the `ReasonerToolCall` variants; delegates to `_dispatch_emit_prompt` / `_dispatch_execute_skill` / `_dispatch_lifecycle_transition` / `_dispatch_spatial_query` / `_dispatch_memory_write` / `_dispatch_memory_search`. `ReloadGstPipelineTool` is the sole log-and-acknowledge stub (F6 sensor-package service IDL not yet on disk — GH-126).
  - `_dispatch_emit_prompt(call, *, traceparent)` — Publish a `PromptStamped` on `/openral/prompt`; stamps the threaded-through `traceparent` into `metadata_json` per ADR-0018 §6.
  - `_dispatch_spatial_query(call, *, traceparent)` — ADR-0039 Phase 2b/§3. Read-only: runs a `RecallObjectTool` / `ResolvePlaceTool` against the injected `SpatialMemory` via `run_spatial_query_detailed` and republishes the rendered result as a `PromptStamped` with frame_id `"spatial_memory"` (so `_on_prompt` consumes it, not filtered as a self-emit) — the prompt cascade feeds the answer into the next tick. **Bounded** by a `SearchProgress`/`SearchBudget`: consecutive queries are counted, and once `max_attempts` is hit the result is published with the reasoner's own frame_id (filtered by `_on_prompt` → no further tick), terminating the cascade in human-handoff. Reset on any non-query dispatch and on a non-cascade operator prompt. No actuation, no `FailureTrigger`. Warns + no-ops if no backend is wired. ADR-0043/0056 (#10): a `recall_object` **miss** (`SpatialQueryOutcome.found == False`) escalates to a live `locate_in_view` for the same query term — policy-driven (not LLM-chosen) — when `detector_available` and the term hasn't already been escalated this search streak (tracked in `_locate_escalated`, reset with the search bound); the open-vocab detector grounds objects the map never ingested / labelled differently before the budget reaches human-handoff. ADR-0044 Phase 4: when a latched `/map` has been received (params `occupancy_map_topic` default `/map` — empty disables; `approach_inflation_m` default 0.25), every `recall_object` approach viewpoint is refined through `refine_approach_pose` before rendering, so the LLM only sees grid-valid approach poses (BLOCKED note when none exists; grid absent → geometric pass-through).
  - `_maybe_load_spatial_memory()` — ADR-0039 deployment wiring. On `on_configure`, when no backend was injected and `spatial_memory_path` is set, lazy-imports `openral_world_state.SpatialMemory`, `SpatialMemory.load(path)`, sets it as the query backend, and flips `spatial_memory_available`. Load failure (`OSError`/`ValueError`) degrades to WARNING + no backend (tools simply not offered) — never a fabricated map. Wired in `sim_e2e.launch.py` via the `spatial_memory_path:=<path>` launch arg. Emits the loaded map once via `_emit_scene_objects_span`.
  - `_maybe_load_memory()` — ADR-0071 §3 deployment wiring. On `on_configure`, when `memory_md_path` is set: parses the (possibly absent) `MEMORY.md` into a `MemoryStore`, renders the `## MEMORY` context block, loads the `<MEMORY.md>.archive.jsonl` recall log (`_load_memory_archive`), and flips `ToolPalette.memory_available` so `memory_write` / `memory_search` are offered. Read failure degrades to WARNING + empty store — never a fabricated memory.
  - `_dispatch_memory_write(call, *, traceparent)` — ADR-0071 §3 / Phase 4c. The reasoner's first **write-capable** dispatch: applies the `MemoryWriteTool` op (`add`/`update`/`supersede`/`delete`) to the live `MemoryStore`, appends any displaced entry to the archival JSONL (`_archive_memory_entry`), persists `MEMORY.md` (`_persist_memory`), re-renders the `## MEMORY` block, and re-prompts a short confirmation (frame_id `"memory"`) so the next tick reads the update. Advisory — a persist failure logs, never raises; warns + no-ops if no backend is wired. Timestamps come from the ROS clock (`_memory_now`, sim-time-aware).
  - `_dispatch_memory_search(call, *, traceparent)` — ADR-0071 §3 / Phase 4c. Read-only: `MemoryStore.search` over the **archive** (superseded/deleted entries that left the live file — current memory is already in the `## MEMORY` block), ranked by importance then recency (MemGPT recall), re-prompted with the hits (frame_id `"memory"`). No actuation, no file write.
  - `_emit_scene_objects_span()` — ADR-0038 dashboard telemetry. When a spatial-memory backend is wired, calls `openral_world_state.emit_scene_objects_span(self._spatial_memory.to_scene_graph(), source_node=…)` to publish the `world.scene_objects` span (scene-objects card + SLAM-map overlay). Called once on load and on every heartbeat `_on_tick` (above the `_core is None` guard, so a preloaded map shows even before the tool-use client builds). Advisory only; all failures swallowed at DEBUG so telemetry never disturbs the tick.
  - `_dispatch_execute_rskill(call, *, traceparent)` — Probe the `/openral/execute_rskill` action server (100 ms `wait_for_server`); on absence emit a `KIND_CONTROLLER` `FailureTrigger` and bail. **ADR-0050 amendment 2026-06-12:** when `vram_lifecycle_peers` is non-empty it routes through `_free_vram_peers_then_send` (deactivate the GPU peers first, then send); otherwise calls `_send_execute_rskill_goal` directly.
  - `_send_execute_rskill_goal(call, traceparent)` — Build `ExecuteRskill.Goal`, send asynchronously with `feedback_callback=_on_execute_rskill_feedback`, attach `_on_execute_rskill_goal_response` to the send future. (Extracted from `_dispatch_execute_rskill` for the VRAM-eviction sequencing.)
  - `_free_vram_peers_then_send(call, peers, traceparent)` — **ADR-0050.** Deactivate each GPU lifecycle peer via `_change_state_async`, and send the goal only once **all** in-flight `change_state` responses return — so the peer's VRAM (e.g. the ~1.3 GB object detector) is released *before* the runner loads the policy on an 8 GB card. Peers whose service is absent are skipped (dispatch still proceeds); the deactivated subset is recorded in `_deactivated_vram_peers` for reactivation.
  - `_reactivate_vram_peers()` — Reactivate the peers in `_deactivated_vram_peers` (clears the set first → idempotent). Called from `_on_execute_rskill_result` (terminal) and the goal-reject/error branches of `_on_execute_rskill_goal_response`; **not** on `deadline` (the policy may still be resident).
  - `_on_reactivate_result(peer, future)` — Best-effort log of a reactivation `change_state` outcome.
  - `_change_state_async(node, transition) -> future | None` — Shared helper: lazily create + cache a `lifecycle_msgs/srv/ChangeState` client per peer node, map `"configure"`/`"activate"`/`"deactivate"`/`"cleanup"` to `Transition.TRANSITION_*`, and call asynchronously. Returns `None` if the service isn't on the graph. Used by both `_dispatch_lifecycle_transition` and the VRAM-eviction path.
  - `_dispatch_lifecycle_transition(call)` — Drive `<call.node>/change_state` via `_change_state_async`; on success attach `_on_lifecycle_response`, on an absent service log + skip.
  - `_on_execute_skill_feedback(rskill_id, feedback_msg)` — Forward action feedback to the operator log at warning level (rare event; OTel/structlog routes this to the dashboard).
  - `_on_execute_skill_goal_response(call, sent_at, future, traceparent)` — On rejection emit a `KIND_CONTROLLER` `FailureTrigger`; on acceptance arm a one-shot deadline timer (`_on_execute_skill_deadline`, only when `call.deadline_s > 0`) and attach `_on_execute_skill_result` to `get_result_async()`.
  - `_on_execute_skill_result(call, goal_id, future, traceparent)` — Cancel the deadline timer; on `STATUS_SUCCEEDED + result.success` log success; on abort/cancel/non-success emit a `KIND_CONTROLLER` `FailureTrigger` with a `ControllerEvidence` payload (state ∈ {`aborted`, `canceled`, `failed`}, `detail=result.failure_reason`).
  - `_on_execute_skill_deadline(*, call, sent_at, goal_handle, traceparent)` — Cancel the goal via `cancel_goal_async()`; emit a `KIND_TIMEOUT` `FailureTrigger` with `TimeoutEvidence(operation="skill.<rskill_id>", deadline_s, elapsed_s)`.
  - `_on_lifecycle_response(call, future)` — Log the `ChangeState` result; lifecycle failures are operator-driven and surface in the target node's own logs (no `FailureTrigger` re-emission).
  - `_publish_skill_failure(*, kind, rskill_id, evidence, traceparent, trace_id=None)` — Build + publish a `FailureTrigger` on `/openral/failure/rskill` with `severity=SEVERITY_FAIL`; `trace_id` (when propagated by the action result) takes precedence over the reasoner's active `traceparent`.
  - Properties `renderer`, `dispatched_calls`; method `set_palette(palette)` (imperative seam called from the `/openral/skill_registry_changed` refresh callback).
- `_QOS_REGISTRY_CHANGED` — RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1 so a late-subscribing reasoner sees the most recent invalidation.
- `main(args=None) -> int` — Entry point for `ros2 run openral_reasoner_ros reasoner_node`.

### `packages/openral_prompt_router/openral_prompt_router/prompt_router_node.py`
_ADR-0018 F10 — single lifecycle node that fans in operator prompts from any external source into `/openral/prompt`. CLI is the only v1 adapter; WebSocket / voice / Slack out-of-scope per ADR §"out-of-scope"._

- module constant `DEFAULT_SOURCES: dict[str, int] = {"cli": 100, "dashboard": 100, "auto": 10}` — Default source → priority registry; human sources get 100, machine cascades get 10. (L59)
- `class PromptRouterNode(LifecycleNode)` (L66) — Lifecycle node.
  - `__init__(*, node_name="openral_prompt_router", sources=None)` — Initialise with a source → priority registry. Defaults to `DEFAULT_SOURCES`.
  - `on_configure` — Build the `/openral/prompt` fan-out publisher and one `/openral/prompt_in/<source>` subscriber per allowed source.
  - `_on_inbound(source, priority, msg)` — Forward the inbound PromptStamped onto `/openral/prompt` after merging `{"source": ..., "priority": ...}` into `metadata_json` (preserving any per-source fields).
  - Property `forwarded_count` — Number of prompts forwarded since `on_configure` (for tests).
- `main(args=None) -> int` — Entry point for `ros2 run openral_prompt_router prompt_router_node`.

### `python/cli/src/openral_cli/prompt.py`
_ADR-0018 F10 — `openral prompt "do X"` CLI adapter. Publishes a one-shot `PromptStamped` onto `/openral/prompt_in/cli` for the prompt-router to fan out. `rclpy` lazy-imported so `openral --help` stays sub-second._

- `prompt_command(text, topic="/openral/prompt_in/cli", wait_s=1.0)` — Initialise rclpy, publish one PromptStamped with `metadata_json={"source_cli": true}`, wait briefly for the subscriber to be discovered, then shut down. Exits 2 if rclpy / openral_msgs are not importable (with a hint at `just ros2-build`).

### Observability (Layer 8 — fully shipped, ADR-0017)

### `python/observability/src/openral_observability/_sdk.py`
_Idempotent OTel SDK setup + flush helper._

- `configure_observability(*, service_name="openral", endpoint=None, sample_ratio=None) -> bool` — Install OTLP/gRPC tracer + meter + logger providers; reads `OTEL_EXPORTER_OTLP_ENDPOINT` when `endpoint` is None; returns `True` if exporters were installed, `False` for the no-op path. On a successful install also kicks off `start_system_metrics_collector` so the dashboard's System health card receives CPU / RAM / GPU gauges. Registers `shutdown_observability` via `atexit` on first install. Metric reader interval is configurable via `OPENRAL_OTEL_METRIC_INTERVAL_MS` (default 5 s); the `BatchSpanProcessor` flush interval via `OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS` (default 30 ms ≈ 33 Hz — set ~1.3× the 25 Hz thumbnail rate so the dashboard captures every frame without flush-aliasing; raise it for coarser production batching). `sample_ratio` selects the trace sampler — `None` / `1.0` → `ALWAYS_ON`, values in `(0, 1)` → `ParentBased(TraceIdRatioBased(ratio))`; honors `OPENRAL_OTEL_SAMPLE_RATIO` env var when arg is None. (L111)
- `configure_worker_observability(service_name, *, endpoint=None, sample_ratio=None) -> bool` — Cross-process bootstrap for a spawned worker (dispatcher, future fleet supervisor): calls `configure_observability` (OTLP pipeline + structlog bridge) then `attach_traceparent_from_env` so the worker's root context is the parent trace; returns whatever `configure_observability` returned. Parent must spawn the child with `env={**os.environ, **traceparent_env()}` (R2 multiprocess log/trace correlation). (L227)
- `_resolve_sampler(sample_ratio) -> Sampler` — Resolve the trace sampler from arg + env, defaulting to `ALWAYS_ON`. Garbage env values fall back to always-on so a typo never drops every span. (L291)
- `shutdown_observability() -> None` — Flush + shut down all three providers; idempotent and safe to call when no exporter was installed. Stops the system-metrics collector before draining the meter so the final sample lands in the export batch. (L348)

### `python/observability/src/openral_observability/tracing.py`
_Span-context-manager helpers; safe to call before `configure_observability`._

- `rskill_span(name, *, rskill_id=None, role=None, **attrs)` — Span for a Skill lifecycle phase; emits `rskill.id` / `rskill.role` from `semconv`. (L40)
- `inference_span(name="skill.chunk_inference", *, chunk_index=None, kind="foreground", **attrs)` — Span for one VLA chunk inference; emits `inference.kind` / `inference.chunk_index`. (L70)
- `safety_span(name="safety.check", *, check_name=None, severity="info", **attrs)` — Span for a safety check; the C++ kernel parents its own `safety.check` to the Python tick via the propagator. (L98)
- `reasoner_span(name="reasoner.tick", *, tick_idx=None, model=None, force=None, **attrs)` — Span for one `ReasonerCore.tick` (ADR-0018 F4). Sets `reasoner.{tick.idx, model, force}` and accepts any extra `reasoner.*` attribute via `**attrs`. Used by `openral_reasoner.core` to record `reasoner.{tool, rskill_id, suppressed_reason, error_kind}` over the LLM call. (L132)
- `traced(name=None)` — Decorator that wraps a sync function in a span named after it. (L190)

### `python/observability/src/openral_observability/cli.py`
_Root-span helper for the ``openral`` CLI._

- `cli_command_span(subcommand, *, mode=None, run_id=None, **attrs)` — Open the `cli.command` root span for one CLI invocation; records `cli.subcommand`, `openral.run.id`, optional `openral.run.mode` / `openral.run.git_sha`. (L52)

### `python/observability/src/openral_observability/diagnostics.py`
_ADR-0018 F8 — `diagnostic_msgs/DiagnosticArray` heartbeat helper, shared by every OpenRAL lifecycle node._

- `Level` — Mirror of `diagnostic_msgs/DiagnosticStatus` level constants (`OK=0`, `WARN=1`, `ERROR=2`, `STALE=3`); re-exported so `status_fn` callbacks can avoid importing `diagnostic_msgs` on pure-Python hosts. (L32)
- `DiagnosticsHeartbeat(node, *, hardware_id, component_name, status_fn, rate_hz=1.0)` — 1 Hz `/diagnostics` publisher attached to a `rclpy.lifecycle.LifecycleNode`. Drives the standard `create_publisher` (in `on_configure`) / `start` (in `on_activate`) / `stop` (in `on_deactivate`) / `destroy` (in `on_cleanup`) sequence; `publish_once()` exposes a deterministic publication for tests; an exception inside `status_fn` is converted to a synthetic ERROR-level diagnostic so the timer never crashes the node. (L49)

### `python/observability/src/openral_observability/lifecycle.py`
_Make `LifecycleNode` transition-callback failures observable — rclpy's `__execute_callback` swallows callback exceptions into `TransitionCallbackReturn.ERROR` without logging (literal `# TODO(ivanpauno): log sth here`), so a composing host reports only `exit code 4`._

- `log_lifecycle_errors(callback) -> callback` — Decorator for `on_configure` / `on_activate` / … transition callbacks. Transparent on success; on an uncaught exception it logs the callback name + full traceback via `node.get_logger().error(...)` (→ `/rosout` → launch console) and returns `TransitionCallbackReturn.FAILURE` instead of letting the exception escape into rclpy's silent `ERROR` conversion. Applied to the `on_configure`/`on_activate` of `RskillRunnerNode`, `_WorldStateLifecycleNode`, `HALLifecycleNodeBase` (covers every per-robot HAL), and `ReasonerNode`. Imports `rclpy` lazily so the module stays import-safe on pure-Python hosts. (L39)

### `python/observability/src/openral_observability/semconv.py`
_Single source of truth for OpenRAL OTel attribute / span / metric names._

`Final[str]` constants for: the legacy `rskill.*` / `skill.*` / `inference.*` / `safety.*` attribute prefixes (shipped today); the greenfield `openral.run.*` / `openral.tick.*` / `openral.skill.*` / `openral.hal.*` / `openral.sensors.*` / `openral.world_state.*` / `openral.dataset.*` namespaces; span names (`SPAN_*`, incl. `SPAN_WORLD_SCENE_OBJECTS = "world.scene_objects"`); the ADR-0038 `openral.world_state.scene_objects.*` dashboard attrs (`WORLD_SCENE_OBJECTS_LIST` / `_COUNT` / `_FRAME` / `_SOURCE_NODE`); span-event names (`EVENT_*`, incl. `EVENT_EPISODE_CLOSED` added in ADR-0019); metric instrument names (`METRIC_*`); closed-set metric label keys (`LABEL_*`); and enum values for `openral.run.mode` / `openral.safety.kernel`. ADR-0019 also adds `DATASET_EPISODE_SUCCESS` to the `openral.dataset.*` namespace; the placeholder `DATASET_REPO_ID` / `DATASET_EPISODE_IDX` / `DATASET_FRAME_IDX` constants (L143–145) are now written by `openral_dataset.RolloutRecorder`.

### `python/observability/src/openral_observability/metrics.py`
_Cached OTel meter instruments — safe to call before `configure_observability`._

- `get_meter() -> Meter` — Resolve the OpenRAL meter against the current `MeterProvider`. (L66)
- `get_tick_duration() -> Histogram` — `openral.tick.duration`, unit `ms`. (L101)
- `get_inference_duration() -> Histogram` — `openral.inference.duration`, unit `ms`. (L116)
- `get_hal_read_state_duration() -> Histogram` — `openral.hal.read_state.duration`, unit `ms`. (L128)
- `get_hal_send_action_duration() -> Histogram` — `openral.hal.send_action.duration`, unit `ms`. (L140)
- `get_sensors_age_ms() -> Histogram` — `openral.sensors.age_ms`, unit `ms`. (L152)
- `get_world_state_staleness_ms() -> Histogram` — `openral.world_state.staleness_ms`, unit `ms`. (L164)
- `get_tick_budget_violations() -> Counter` — `openral.tick.budget_violations`. (L179)
- `get_tick_deadline_misses() -> Counter` — `openral.tick.deadline_misses`. (L190)
- `get_inference_timeouts() -> Counter` — `openral.inference.timeouts`. (L201)
- `get_safety_violations() -> Counter` — `openral.safety.violations`, labels `check_name` / `severity`. (L212)
- `get_safety_clamps() -> Counter` — `openral.safety.clamps`, label `check_name`. (L226)
- `get_hal_estop_count() -> Counter` — `openral.hal.estop.count`. (L237)
- `get_sensors_stale_reads() -> Counter` — `openral.sensors.stale_reads`. (L248)
- `get_observability_export_failures() -> Counter` — `openral.observability.export_failures`, label `signal_kind`. (L281)
- `get_world_state_components_stale() -> UpDownCounter` — `openral.world_state.components_stale`. (L298)
- `record_histogram_ms(instrument, value_ms, attributes=None) -> None` — Record a millisecond value, skipping negatives and `NaN`. (L387)

### `python/observability/src/openral_observability/producer.py`
_Producer-side helpers for recording rich span attributes on OpenRAL hot-path spans. Safe to call on no-op spans; lists are truncated to `_MAX_JOINTS` / `_MAX_EE_FRAMES` and floats rounded to 3 decimals._

- `record_joint_state(span, *, names, positions, velocities=None, efforts=None, position_limits=None, velocity_limits=None, effort_limits=None, stamp_ns=None) -> None` — Attach per-joint attributes to a `hal.read_state` span. (L83)
- `record_action(span, *, next_row, dim=None, horizon=None, applied=None, gripper_position=None, gripper_force_n=None) -> None` — Attach commanded-action attributes to a `hal.send_action` span. (L131)
- `record_ee_poses(span, ee_poses) -> None` — Flatten a `name → Pose6D` mapping onto a `world_state.snapshot` span. (L162)
- `record_sensor_frame_attrs(span, *, modality=None, encoding=None, width=None, height=None, channels=None, age_ms=None, thumbnail_bytes=None, thumbnail_already_encoded_b64=False) -> None` — Attach sensor-frame attributes to a `sensors.read_latest` span. (L188)
- `encode_rgb_thumbnail(rgb) -> bytes | None` — Encode an HWC uint8 RGB ndarray to a small JPEG for OTLP; returns `None` if Pillow is unavailable. (L229)
- `encode_frame_thumbnail(frame) -> bytes | None` — Encode an `openral_core.SensorFrame` (RGB8/BGR8/MONO8/JPEG/PNG) as a small JPEG thumbnail; returns `None` for non-renderable encodings. (L254)
- `modality_for_encoding(encoding) -> str` — Map a `FrameEncoding` (or its string value) to the dashboard's modality label (`rgb` / `mono` / `depth` / `raw` / `unknown`). Reused by `DeployRunner._tick_impl` and `world_state_ros/lifecycle_node._on_image` so both surfaces produce identical modality labels for the same encoding. (L51)
- `_MODALITY_BY_ENCODING: dict[str, str]` (L39) — Canonical encoding → modality lookup table.

### `python/observability/src/openral_observability/system_metrics.py`
_Background sampler for the `openral.system.*` gauges; feeds the dashboard's System Health card via `psutil` (CPU + RAM) and optional `pynvml` (GPU memory + util)._

- `start_system_metrics_collector(*, interval_s=1.0) -> bool` — Start a daemon thread that samples host metrics every `interval_s` seconds. Returns `False` and a quiet no-op when neither `psutil` nor `pynvml` is importable. Idempotent; re-starts retune the interval. (L45)
- `stop_system_metrics_collector(*, timeout_s=2.0) -> None` — Signal the collector thread to stop and join. Safe to call when not running. (L75)

### `python/observability/src/openral_observability/propagation.py`
_W3C TraceContext inject / extract for cross-process trace correlation._

- `current_traceparent() -> str | None` — W3C `traceparent` value for the active span, or `None` outside a span. (L53)
- `inject_traceparent(carrier=None) -> dict[str, str]` — Write the active span's `traceparent` (and optional `tracestate`) into a carrier dict; used by producers of `ActionChunk.msg` / `ExecuteRskill.action` / `FailureTrigger.msg`. (L68)
- `extract_traceparent(traceparent, tracestate=None) -> Context` — Parse a wire-side `traceparent` into an OTel `Context` for `context.attach` / `trace.use_span`; consumed by the C++ safety kernel and any Python ROS consumer. (L98)
- `traceparent_env(carrier=None) -> dict[str, str]` — Env-var carrier (`OTEL_TRACEPARENT` + optional `OTEL_TRACESTATE`) for the active span, built from `inject_traceparent`; pass as `env=` to `subprocess` / `multiprocessing` so a worker joins the parent trace. `{}` when no valid span is in scope. (L134)
- `attach_traceparent_from_env(env=None) -> object | None` — Worker-side counterpart: read `OTEL_TRACEPARENT` / `OTEL_TRACESTATE` from `env` (default `os.environ`) and `context.attach` the parent context; returns the detach token, or `None` when absent/empty. (L180)
- `remote_parent_from_env(env=None)` [@contextmanager] — Scope `attach_traceparent_from_env` for a worker `main()`: attaches on enter, detaches on exit; yields the detach token (or `None` when no carrier present). (L227)

### `python/observability/src/openral_observability/failure_bus.py`
_ADR-0018 F3 — publisher helper + IDL-mirror constants for the namespaced `/openral/failure/{...}` bus._

- `class FailureSource(str, Enum)` (L118) — `HAL | SENSOR | SKILL | SAFETY | WAM | CRITIC`; the string value is the topic suffix.
- `topic_for(source: FailureSource) -> str` (L133) — Pure helper: `FailureSource → /openral/failure/<suffix>`.
- `KIND_*` / `SEVERITY_*` `int` module constants (L94–L109) — Mirror `openral_msgs/msg/FailureTrigger`; bump both when the IDL changes.
- `DEFAULT_RATE_LIMIT_HZ: dict[int, float | None]` (L150) — Per-severity defaults (INFO/WARN → 10/s, FAIL/ABORT → unlimited). `DEFAULT_SUMMARY_PERIOD_S = 1.0` (L154).
- `class _TokenBucket` (L164) — Private, lock-protected. `__init__(rate_hz, *, capacity=1.0, clock=time.monotonic)`; `try_consume() -> bool`.
- `class FailureBusPublisher` (L213) — `__init__(node, source, *, rate_limit_hz=None, summary_period_s=1.0, clock=None)`. Methods: `create_publisher()` (opens RELIABLE+VOLATILE+KL=50 publisher on `topic_for(source)`), `start()` (boots 1 Hz suppressed-summary timer), `stop()`, `destroy()`, `publish(*, kind, severity, evidence, rskill_id='', trace_id=None) -> bool` (False when rate-limited). Properties: `topic`, `source`.

### `python/observability/src/openral_observability/logging.py`
- `trace_context_processor(_logger, _method_name, event_dict)` — structlog processor that stamps `trace_id` / `span_id` on every log event. (L41)
- `install_structlog_bridge(logger_provider)` — Wire the structlog processor chain to forward records to the OTel `LoggerProvider`. (L57)

### `python/observability/src/openral_observability/dashboard/store.py`
_In-memory aggregator for `openral dashboard` — feeds the SSE stream and the `/api/state` JSON endpoint. Thread-safe, bounded (200 events, 600 metric samples per series). (ADR-0017, issue #44). Span families registered in `_HEADLINE_FAMILIES` (L772+): `rskill.execute`, `rskill.tick`, `rskill.activate`, `rskill.configure`, `skill.chunk_inference`, `safety.check`, `hal.send_action`, `hal.read_state`, `sensors.read_latest`, `world_state.snapshot`, `slam.occupancy_grid` (ADR-0025 SLAM map card), **`reasoner.tick` (ADR-0018 F4 — last LLM tool decision, rendered in the Reasoner card added alongside ADR-0025's navigate-look-pick demo)**, `sim.run`, `sim.step`, `cli.command`. Each populates one slot in `self._topics: dict[str, dict[str, Any]]` (L266+)._

- `class TelemetryEvent` — Frozen dataclass holding one event log row (`ts_unix`, `kind`, `title`, `attrs`, `severity`). `.to_json()` returns a plain dict. (L137)
- `class TelemetryStore` — Read-side aggregator over OTLP signals. (L283)
  - `ingest_spans(payload: list[ResourceSpans]) -> int` — Decode + record spans; populates headline cards, increments span-event counters, publishes a delta to every subscriber queue. Returns the number of spans recorded. Routes by span name into per-topic buckets, incl. `world.scene_objects` → `topics["scene_objects"]` (ADR-0038 — durable spatial-memory objects for the scene-objects card + SLAM-map overlay; the `world_state.scene_objects.list` JSON attr is decoded via `_parse_object_list`). (L357)
  - `ingest_metrics(payload: list[ResourceMetrics]) -> int` — Decode + record metric data points; appends per-series samples and tracks cumulative sums. (L391)
  - `ingest_logs(payload: list[ResourceLogs]) -> int` — Decode + record OTLP `ResourceLogs` (the structlog→OTel bridge) as event-log rows (issue #318): body → title, instrumentation scope (logger) name → kind, `severity_number` → `debug`/`info`/`warn`/`error`/`fatal` via `_log_level`. Records share the bounded event ring with spans/span-events; the UI defaults the Debug chip off so high-rate DEBUG stays opt-in. Returns the number of log records recorded.
  - `snapshot() -> dict[str, Any]` — One-shot view: service identity, headline cards, event ring, counters, metric series with p50/p95. (L451)
  - `subscribe() -> asyncio.Queue` — Register an SSE subscriber. The queue is bounded; on overflow the oldest payload is dropped so the producer never blocks. (L456)
  - `unsubscribe(queue) -> None` — Drop a subscriber's queue. (L472)

### `python/observability/src/openral_observability/dashboard/discovery.py`
_mDNS advertise + browse for the live dashboard (issue #75b). Optional — requires the `mdns` extra (`zeroconf>=0.131`, LGPL-2.1, TSC-approved 2026-06-21). When `zeroconf` is not importable, `Discovery` stays disabled and the dashboard runs exactly as before._

- module constant `SERVICE_TYPE: str = "_openral-otlp._tcp.local."` — mDNS service type for all OpenRAL dashboard OTLP receivers. Single source of truth for the advertiser and browser. (L30)
- `class DiscoveredRobot(BaseModel)` — One mDNS-discovered OpenRAL service; the `/api/robots` wire shape used by external operator tooling. Fields: `name: str`, `addresses: list[str]`, `port: int`, `properties: dict[str, str] = {}`, `last_seen: float`. (L37)
- `class RobotRegistry` — Thread-safe map of discovered robots (zeroconf callbacks run off-thread). Methods: `upsert(robot: DiscoveredRobot) -> None`, `remove(name: str) -> None`, `list_robots() -> list[DiscoveredRobot]` (sorted by name, snapshot). (L47)
- `class Discovery` — Owns the `Zeroconf` instance, advertiser, and browser for the dashboard. Attribute `enabled: bool`; method `robots() -> list[DiscoveredRobot]` (delegates to registry); `start(*, host: str, port: int) -> None` (browse always; advertise only on a non-loopback, non-wildcard bind — loopback/wildcard binds are browse-only); `stop() -> None` (unregister, cancel browser, close Zeroconf). Wired into `run_dashboard` in `server.py`; `app.state.discovery` holds the instance (or `None` when the `mdns` extra is absent or zeroconf failed). (L83)

### `python/observability/src/openral_observability/dashboard/app.py`
- `create_app(store: TelemetryStore | None = None) -> FastAPI` — Build the dashboard ASGI app. Routes: `/`, `/static/*`, `/healthz`, `/api/state`, `/api/stream` (SSE), the OTLP/HTTP receivers `POST /v1/traces`, `POST /v1/metrics`, `POST /v1/logs` (logs now feed the event log via `TelemetryStore.ingest_logs` — issue #318), and the operator write endpoints `POST /api/prompt` (ADR-0018 F10, shells out to `openral prompt --topic /openral/prompt_in/dashboard`), `POST /api/estop_reset`. Also: `GET /api/camera/{source}/stream` (MJPEG multipart live camera video re-serving OTLP thumbnails — issue #75a); `GET /api/robots` (mDNS-discovered OpenRAL services for external discovery tooling — issue #75b, read-only, returns `{"enabled": false, "robots": []}` when discovery is absent); `POST /api/skill/execute` and `POST /api/param/set` (guarded write-controls, default OFF — issue #75c / ADR-0064; return 403 unless `OPENRAL_DASHBOARD_WRITE_CONTROLS=1`; `skill/execute` returns **202** on action-server acceptance with async background result logging; `param/set` also refuses safety-relevant param names via `_SAFETY_PARAM_DENYLIST`). `GET /api/config` now returns `{"jaeger_ui_url": "...", "write_controls_enabled": bool}`. Honours gzip-encoded request bodies. (L673)

### `python/observability/src/openral_observability/dashboard/server.py`
- `run_dashboard(*, host="127.0.0.1", port=4318, inprocess_cmd=None, store=None, log_level="warning") -> None` — Start uvicorn on `host:port` and block until SIGINT/SIGTERM. Prints a single `OpenRAL dashboard: http://host:port/` banner to stderr before binding (issue #132) so the user always sees the URL. When `inprocess_cmd` is set, spawns the argv as a child process with `OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` pointed at the dashboard. Default port is `4318` (OTLP/HTTP standard) instead of the historic `8000` to avoid clashing with `mkdocs serve` / `python -m http.server`. (L56)
- `spawn_dashboard(*, host="127.0.0.1", port=4318, ready_timeout_s=10.0) -> Iterator[str | None]` [@contextmanager] — Inverse of `--inprocess`: spawn `openral dashboard` as a child of the current process, poll `/healthz` until ready, set `OTEL_EXPORTER_OTLP_{ENDPOINT,PROTOCOL}`, yield the URL, and SIGINT the child on exit. Yields `None` (workload continues unattached) if `openral` is not on PATH, the child died early, or `/healthz` never came back within the timeout. (L36, in `openral_observability/dashboard/attach.py`)
- `attached_dashboard(*, enabled, port=4318) -> Iterator[bool]` [@contextmanager] — High-level wrapper used by `openral sim run --dashboard`, `openral deploy run --dashboard`, and `openral benchmark run --dashboard`. When `enabled=False`, yields `False` immediately (true no-op, no FastAPI/uvicorn imports). When `enabled=True`, delegates to `spawn_dashboard`, re-runs `configure_observability` on the new endpoint, and drains via `shutdown_observability` in `finally` so the last batch lands before the child is SIGINT'd. Yields `True` iff the child reported healthy.

### `python/observability/src/openral_observability/dashboard/store.py` — F7 trace index additions
_ADR-0018 F7 — bounded per-trace_id span index for query-time bag↔OTel join._

- `class _IndexedSpan` (L175) — Frozen-ish record retained by `trace_id`: `name`, `trace_id`, `span_id`, `parent_span_id`, `start_ns`, `end_ns`, `attrs`, `status_code`, `status_message`, `events`. `.to_json()` returns a plain dict carrying `duration_ms`.
- `TelemetryStore.list_traces() -> list[dict]` — One row per indexed trace_id (`trace_id`, `span_count`, `last_seen_unix`), most-recent first. Backs `GET /api/traces`.
- `TelemetryStore.lookup_trace(trace_id: str) -> list[dict] | None` — Every indexed span for `trace_id`, sorted ascending by `start_unix_ns`. `None` when the trace is not (or no longer) in the bounded index. Backs `GET /api/spans/{trace_id}`.
- `_TRACE_INDEX_MAX_TRACES = 64` / `_TRACE_INDEX_MAX_SPANS = 2048` — Memory caps. Older trace_ids evict FIFO on insertion.

### `python/observability/src/openral_observability/dashboard/app.py` — F7 routes
- `GET /api/traces` — JSON `{"traces": [...]}` from `TelemetryStore.list_traces`.
- `GET /api/spans/{trace_id}` — JSON `{"trace_id", "spans": [...]}` from `TelemetryStore.lookup_trace`; 404 when the trace is not indexed.
- `GET /api/config` — JSON `{"jaeger_ui_url": "..."}` sourced from the `OPENRAL_JAEGER_UI_URL` env (trailing slash stripped, default `""`). The dashboard UI fetches this on load to decide whether to enable the footer "open in jaeger" link — leaving the env unset keeps the link disabled with a helpful tooltip instead of producing a broken-link click against a guessed `localhost:16686`.

### `python/observability/src/openral_observability/tracing_lttng.py`
_ADR-0018 F9 — opt-in LTTng tracepoints around the realtime hot path. No-op when `OPENRAL_ROS2_TRACING` is unset; falls back to JSONL when `lttngust` is missing._

- `ENV_TRACING_GATE = "OPENRAL_ROS2_TRACING"` — Truthy values (`1`/`true`/`yes`/`on`) enable the backend; anything else leaves every tracepoint a no-op.
- `ENV_TRACING_FALLBACK_DIR = "OPENRAL_ROS2_TRACING_FALLBACK_DIR"` — Override for the JSONL fallback directory (default `/tmp/openral-lttng-fallback`).
- `TP_RUNNER_TICK`, `TP_HAL_READ_STATE`, `TP_HAL_SEND_ACTION`, `TP_SENSORS_READ_LATEST`, `TP_WORLD_STATE_SNAPSHOT`, `TP_SKILL_STEP`, `TP_ACTION_PUBLISH`, `TP_SAFETY_VALIDATE` — Tracepoint base names; `lttng_tracepoint` appends `_begin` / `_end` suffixes.
- `is_enabled() -> bool` (L112) — Single source of truth for the gate.
- `lttng_tracepoint(name, **attrs) -> Iterator[None]` (L175) — Context manager that fires `<name>_begin` / `<name>_end` around the block. Attaches the active OTel `trace_id` as `otel_trace_id` so CTF traces can join back to OTel.
- `class LttngSession(name, output_dir)` (L93) — Identity of an active session.
- `class LttngSessionError(RuntimeError)` (L88) — Raised by the subprocess wrappers.
- `start_session(*, name, output_dir) -> LttngSession` (L321) — `lttng create / enable-event openral:* / add-context / start`.
- `stop_session(*, name) -> None` (L346) — `lttng stop` + `destroy` (flush + teardown).
- `view_session(*, output_dir) -> None` (L358) — `babeltrace2 OUTPUT_DIR`; falls back to listing files when `babeltrace2` is absent.

### `python/dataset/src/openral_dataset/recorder.py`
_ADR-0019 — in-memory per-rollout accumulator with multi-sink fan-out._

- `@dataclass class EpisodeHeader(episode_idx, task_string, fps, robot_name, stamp_ns)` — Per-episode metadata pushed to sinks at `episode_start`. (L58)
- `@dataclass class DatasetFrame(episode_idx, frame_idx, observation_state, images, action, reward, terminated, truncated, stamp_ns, trace_id="", span_id="")` — Per-tick frame pushed to sinks at `record_frame`. `trace_id` (32 hex) / `span_id` (16 hex) carry the producing `rskill.tick` span's ids (ISSUE-109 forward link); `""` when no valid span was in scope. (L81)
- `@dataclass class EpisodeSummary(episode_idx, success, n_frames, stamp_ns)` — Per-episode close-out pushed to sinks at `episode_end`. (L122)
- `class DatasetSink(Protocol)` — Fan-out target with `open_episode` / `write_frame` / `close_episode` / `finalize`. (L140)
- `class RolloutRecorder(*, robot, task_string, fps, sinks, repo_id=None)` — In-memory accumulator that fans every step out to one or more `DatasetSink` implementations and writes the OTel `openral.dataset.repo_id` / `episode_idx` / `frame_idx` attributes on the active `rskill.tick` span. (L167)
  - `episode_start(*, task_string=None) -> int` — Open a new episode; returns its idx. (L296)
  - `record_frame(*, observation_state, images, action, reward, terminated, truncated, stamp_ns, trace_id=None, span_id=None) -> int` — Append one frame. Captures the active `rskill.tick` span's `(trace_id, span_id)` onto the frame (ISSUE-109); explicit `trace_id`/`span_id` override the live capture (the offline converter replays the bag's original ids). (L341)
  - `episode_end(*, success: bool) -> EpisodeSummary` — Close the current episode. (L454)
  - `finalize() -> None` — Flush all sinks idempotently. (L487)
  - prop `fps`, `robot_name`, `repo_id`, `n_sinks`, `expected_state_shape` — Read-only views consumed by callers building the per-frame payload. (L222)
  - `expected_image_keys() -> tuple[str, ...]` — Camera keys (without `observation.images.` prefix) the sinks expect; derived from `RobotDescription.sensors[*].vla_feature_key`. (L256)

### `python/dataset/src/openral_dataset/schema_map.py`
_Pure `RobotDescription` → LeRobot v3 features dict mapping; no I/O, no lerobot import._

- `@dataclass class FeatureSpec(key, dtype, shape)` — Decoupled feature descriptor; sinks translate to lerobot's `{'dtype', 'shape', 'names'}` format. (L45)
- `features_from_robot(robot: RobotDescription, *, fps: float) -> dict[str, FeatureSpec]` — Build the LeRobot v3 features dict for the recorder. Reads `ObservationSpec.state_shape`, `ActionSpec.dim`, and `SensorSpec.vla_feature_key` (image modalities only) from the robot manifest. (L62)

### `python/dataset/src/openral_dataset/bag.py`
_ADR-0019 PR3 — mcap-backed :class:`DatasetSink` for online hardware recording._

- `Rosbag2Sink(*, bag_path, compression="zstd")` — Writes every `RolloutRecorder` event into an mcap file readable by `ros2 bag info` / Foxglove / mcap-cli. Daemon writer thread + bounded `queue.Queue` → `write_frame` enqueues only; hot path never blocks on disk I/O. JSON-schema encoding (interoperable with ROS 2's `ros2msg` encoding for the same topics). Topics: `/openral/tick` (per-tick metadata **plus inline `observation_state` + `action` arrays**), `/openral/episode` (PHASE_START / PHASE_END markers), `/openral/dataset/image` (one base64 raw-u8 frame per camera per tick). The inline arrays + image frames make the bag self-sufficient for conversion — no separate `/joint_states` / camera-topic join needed. (L190)
  - `open_episode(header) -> None` — Open the bag on first call; emit Episode(PHASE_START). (L291)
  - `write_frame(frame) -> None` — Enqueue a Tick message (incl. inline `observation_state`/`action` + the frame's `trace_id`/`span_id`, ISSUE-109) and one DatasetImage message per camera; never blocks. The off-thread mcap write reads the ids off the frame because the OTel context is gone by then. (L305)
  - `close_episode(summary) -> None` — Emit Episode(PHASE_END) with success flag. (L348)
  - `finalize() -> None` — Drain queue, stop writer thread, close mcap. Idempotent. (L358)
  - prop `bag_path`, `n_ticks_written`, `n_episode_markers_written`, `n_images_written`, `n_dropped` — Diagnostics. (L265)
- `TOPIC_TICK`, `TOPIC_EPISODE`, `TOPIC_IMAGE`, `PHASE_START`, `PHASE_END` — Module-private constants the converter imports by symbol. (L66, L67, L73, L117, L118)

### `python/dataset/src/openral_dataset/converter.py`
_ADR-0019 PR4 — offline mcap rosbag2 → LeRobotDataset v3 converter._

- `@dataclass class DatasetSummary(output_root, n_episodes, n_frames, n_success, repo_id)` — Returned by `from_bag` describing what landed on disk. (L69)
- `Rosbag2ToLeRobotConverter.from_bag(*, bag_path, robot, output_root, repo_id=None, license="CC-BY-4.0", fps=None) -> DatasetSummary` — Walk a `Rosbag2Sink`-produced mcap, group Ticks under PHASE_START / PHASE_END markers, join each tick's inline `observation_state`/`action` arrays + the per-`(episode_idx, step_idx)` camera frames from `/openral/dataset/image`, and replay each episode through a real `LeRobotDatasetSink` → produce a reloadable v3 dataset with **real** proprio/action/video (legacy metadata-only bags fall back to zero vectors of the declared shape). Each replayed tick re-injects the bag's original `(trace_id, span_id)` so the on-disk frame points at the source rollout, not the convert run (ISSUE-109). Raises `ROSConfigError` on missing bag / missing episode markers / mismatched robot. (L118)

### `python/dataset/src/openral_dataset/frame_trace.py`
_ISSUE-109 — pivot a written LeRobotDataset frame back to its OTel ids._

- `read_frame_trace(*, root, episode_idx, frame_idx) -> tuple[str, str]` — Return the `(trace_id, span_id)` stamped on a v3 frame. Reads the `root/data/**/*.parquet` correlation columns directly via `pyarrow` (no video decode), so it works without a torchcodec/ffmpeg backend. Raises `ROSConfigError` when the root has no parquet, the dataset predates the columns, or no `(episode_idx, frame_idx)` row matches. Backs `openral replay --frame`. (L28)

### `python/dataset/src/openral_dataset/sinks.py`
_LeRobotDataset v3.0 (codebase_version="3.0") writer; deferred `LeRobotDataset.create` so per-camera shapes come from the first frame._

- `class LeRobotDatasetSink(DatasetSink)` — Implementation of `DatasetSink` writing LeRobot v3 datasets via real `lerobot.datasets.LeRobotDataset.create / add_frame / save_episode / finalize`. Lazy-imports lerobot at construction. (L93)
  - `__init__(*, root, robot, fps, repo_id=None, license="CC-BY-4.0", vcodec="libsvtav1")` — Raises `ROSConfigError` if lerobot ≥ 0.5.1 is not importable. (L129)
  - `open_episode(header) -> None` — Stash the task string for per-frame tagging. (L257)
  - `write_frame(frame) -> None` — Validates per-frame shapes against the declared features, then forwards to `LeRobotDataset.add_frame`. Adds the frame's `trace_id`/`span_id` as `string` parquet columns (ISSUE-109). (L276)
  - `close_episode(summary) -> None` — Calls `LeRobotDataset.save_episode(parallel_encoding=True)` and accumulates the per-dataset success counter. (L353)
  - `finalize() -> None` — Calls `LeRobotDataset.finalize()` then appends `dataset_success_rate` / `license` / `repo_id` and the dataset-level `trace_ids` / `n_traces` (distinct OTel traces, ISSUE-109) to `meta/info.json["metadata"]`, and writes the per-episode `episode_index → trace_id` map to the `meta/openral_traces.json` sidecar. (L396)
