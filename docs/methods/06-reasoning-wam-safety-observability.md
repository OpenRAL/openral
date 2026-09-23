# Layer 4–6 — Reasoning, Safety, Observability

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Layer 6 (Observability) is fully shipped — traces + metrics + structlog→OTLP log bridge, with W3C TraceContext propagation helpers for cross-process correlation (Python ↔ ROS 2 ↔ C++ safety kernel). Layer 4 (Reasoner) ships the live `ReasonerCore` direct-dispatch loop below; Layer 5 (C++ safety kernel) is still planned.

### `python/reasoner/src/openral_reasoner/tool_use.py`
_Typed LLM tool-use clients (direct-dispatch surface); the `ReasonerToolCall` union is the sole planner output._

- module constant `DEFAULT_SYSTEM_PROMPT: str` — Base S2 system prompt: one-tool-per-tick, safe skill selection, never bypass e-stop, exact field names, `wait` while a skill runs. Deployments may override. (L91)
- module constants `ANTHROPIC_BASE_URL` / `OPENROUTER_BASE_URL` / `OLLAMA_BASE_URL` / `VLLM_BASE_URL` / `GEMINI_BASE_URL` / `XAI_BASE_URL` / `DEEPSEEK_BASE_URL` / `HUGGINGFACE_BASE_URL: str` — Named-endpoint base URLs; canonical copy now lives in `openral_core.schemas`, re-exported here unchanged.
- module constant `_ENDPOINT_PRESETS: dict[str, ReasonerEndpointPreset]` — Back-compat alias of `openral_core.REASONER_ENDPOINT_PRESETS`: per-endpoint `url`/`dialect`/`auth_required`/`timeout_s`/`tool_choice` for each `OPENRAL_REASONER_ENDPOINT` name. The curated model-first path takes only the endpoint properties from it; the uncurated path takes all five. (L626)
- module constant `SYSTEM_PROMPT_ENV_VAR: str = "OPENRAL_REASONER_SYSTEM_PROMPT"` (L430) — env var that overrides the base operating brief; honoured by `resolve_reasoner_system_prompt`.
- `render_robot_context_prompt(capabilities: RobotCapabilities | None, *, base_prompt=DEFAULT_SYSTEM_PROMPT) -> str` (L319) — Appends a deterministic `## THIS ROBOT` body-awareness block (embodiment, locomotion, manipulation/sensing, payload, control modes) to the prompt; `None` returns `base_prompt` unchanged.
- `resolve_reasoner_system_prompt(capabilities: RobotCapabilities | None, *, env=None) -> str` (L433) — Composes the full reasoner system prompt (env override or default base brief + `## THIS ROBOT` block). `env` is injectable for tests; called from `ReasonerNode.on_configure`.
- `class ToolUseClient(Protocol)` (L480) — LLM tool-use client contract: `model_id`, optional `last_prompt_tokens`. Raises `ROSReasonerInvalidPlan` on a bad/mismatched tool call, `ROSPlanningError` on transport failure.
  - `select_tool(self, *, context_text, palette, system_prompt=DEFAULT_SYSTEM_PROMPT) -> ReasonerToolCall` (L502) — Pick exactly one `ReasonerToolCall` for `context_text`.
  - `describe_image(self, *, image_jpeg, question) -> str` (L534) — Ask the model a free-text question about a single camera frame.
- `class OpenAICompatibleToolUseClient` — OpenAI-compatible tool-use client. `tool_choice='auto'` covers endpoints that reject `'required'` (retries once on a prose reply); `max_tokens` optionally caps completion tokens, else the endpoint default applies.
- module constant `REASONER_MODEL_ENV: str = "OPENRAL_REASONER_MODEL"` (L568)
- module constant `REASONER_ENDPOINT_ENV: str = "OPENRAL_REASONER_ENDPOINT"` (L569)
- module constant `REASONER_API_KEY_ENV: str = "OPENRAL_REASONER_API_KEY"` (L570)
- module constant `REASONER_DIALECT_ENV: str = "OPENRAL_REASONER_DIALECT"` (L571)
- module constant `REASONER_MAX_TOKENS_ENV: str = "OPENRAL_REASONER_MAX_TOKENS"` (L572)
- module constant `REASONER_TIMEOUT_ENV: str = "OPENRAL_REASONER_TIMEOUT_S"` (L573) — model-first env names (ADR-0088).
- `build_tool_use_client_from_env() -> ToolUseClient` — Model-first client factory: `OPENRAL_REASONER_MODEL` resolves `openral_core.REASONER_MODELS` for dialect/endpoint/auth/hosting/tool-choice/token-cap; `OPENRAL_REASONER_ENDPOINT` overrides location (a named preset, a bare URL, or `managed`); `API_KEY`/`MAX_TOKENS`/`TIMEOUT_S` override the rest. `DIALECT` is needed only for a bare URL and always wins when set. No hidden model default. (L629)
- `class AnthropicToolUseClient` (L1417) — Anthropic SDK-backed client; builds/reuses one client/HTTP pool and marks the static system/tools prefix cacheable.
  - `select_tool(self, *, context_text, palette, system_prompt=DEFAULT_SYSTEM_PROMPT) -> ReasonerToolCall` (L1484) — Call Anthropic and decode the resulting tool payload.
  - `describe_image(self, *, image_jpeg, question) -> str` (L1531) — Ask the Anthropic model a free-text question about a camera frame.
- `_anthropic_response_text(response) -> str` — Concatenate all text blocks so thinking-enabled responses do not lose an answer after a leading thinking block.
- `class OpenAICompatibleToolUseClient` (L1578) — OpenAI-compatible client with cached SDK/HTTP pool, endpoint/tool-choice/token-cap configuration, and image-description support.
  - `select_tool(self, *, context_text, palette, system_prompt=DEFAULT_SYSTEM_PROMPT) -> ReasonerToolCall` (L1653) — Call the OpenAI-compatible endpoint and decode the tool call.
  - `describe_image(self, *, image_jpeg, question) -> str` (L1728) — Ask an OpenAI-compatible model a free-text question about a frame.
- `_tool_palette_to_anthropic_tools(palette) -> list[dict]` — Render the closed palette, including `WaitTool`; per-skill names use collision-resistant `execute_rskill__<slug>_<sha1-8>`.
- `_tool_palette_to_openai_tools(palette) -> list[dict]` — Convert the same surface to OpenAI function shape without leaking the Anthropic-only `input_schema` key.
- `_decode_tool_payload(*, tool_name, arguments, palette) -> ReasonerToolCall` — Validate provider output against the union + palette; per-skill names resolve through the same hashed mapping used to render them.
- module constant `_PER_SKILL_TOOL_PREFIX: str = "execute_rskill__"` — prefix the decoder matches on to identify per-skill tool calls. (L923)
- module constant `_LLM_TOOL_NAME_MAX_LEN: int = 64` — Anthropic + OpenAI tool-name regex limit; long HF Hub ids are sha1-suffix-truncated to fit. (L926)
- `_skill_id_to_tool_name(rskill_id: str) -> str` — Map a `<owner>/<repo>` id into a collision-resistant 64-char-max `execute_rskill__<slug>_<sha1-8>` name. (L929)
- `_format_skill_tool_description(entry: RSkillToolEntry) -> str` — Render the skill's id + description + actions + objects + scenes into the NL string the LLM scores. (L946)
- `_drop_property(schema: dict, name: str) -> dict` — Return a copy of a JSON Schema dict with `name` stripped from both `properties` and `required`. Used to drop `rskill_id` from per-skill `ExecuteRskillTool` schemas. (L1301)
- module constant `_TOOL_ADAPTER: TypeAdapter[ReasonerToolCall]` (L316) — cached `TypeAdapter(ReasonerToolCall)` used to validate a decoded tool payload against the discriminated union.

### `python/reasoner/src/openral_reasoner/cosmos3.py`
_NVIDIA Cosmos 3 reasoner backend (`OPENRAL_REASONER_MODEL=cosmos3-edge`), the physical-AI-native S2 planner. Runs the on-device 4B Edge tier behind an OpenAI-compatible endpoint, so the typed tool-use contract is unchanged. Companion boot helper: `tools/cosmos3_reasoner_sidecar.py`._

- module constant `COSMOS3_BASE_URL: str` — `http://127.0.0.1:8901/v1`; dedicated port so the sidecar never collides with a user-run `vllm serve` on :8000. (L68)
- module constant `DEFAULT_COSMOS3_MODEL: str` — `nvidia/Cosmos3-Edge`, served id of the curated `cosmos3-edge` registry entry. (L74)
- module constants `AUTOSTART_ENV` / `BOOT_TIMEOUT_ENV` / `SIDECAR_SCRIPT_ENV` / `DEFAULT_BOOT_TIMEOUT_S` — env knobs: `OPENRAL_COSMOS3_AUTOSTART` (`0` disables managed spawn), `OPENRAL_COSMOS3_BOOT_TIMEOUT_S` (first boot provisions a venv + downloads weights; default 1800 s), `OPENRAL_COSMOS3_SIDECAR` (boot-helper path override). (L79)
- module constant `BOOT_TIMEOUT_ENV: str = "OPENRAL_COSMOS3_BOOT_TIMEOUT_S"` (L80)
- module constant `SIDECAR_SCRIPT_ENV: str = "OPENRAL_COSMOS3_SIDECAR"` (L81)
- module constant `DEFAULT_BOOT_TIMEOUT_S: float = 1800.0` (L82) — default boot timeout in seconds.
- module constant `_LOOPBACK_HOSTS: frozenset[str]` (L84) — `{"localhost", "127.0.0.1", "::1", "0.0.0.0"}`, hostnames `_managed_port` treats as loopback.
- `find_cosmos3_sidecar_script() -> Path` — Locate `tools/cosmos3_reasoner_sidecar.py` (env override or repo-parent walk); raises `ROSConfigError` when an explicit override path is missing or the walk finds nothing. (L87)
- `_managed_port(base_url) -> int | None` — Explicit port of a loopback base URL, or `None` (unmanaged, e.g. a portless reverse proxy). Autostart needs an explicit loopback port so spawn and readiness probe agree. (L119)
- `class Cosmos3ToolUseClient(OpenAICompatibleToolUseClient)` — Managed local sidecar client over the inherited OpenAI-compatible wire path; adds probe/spawn/readiness/teardown. `warm()` lets `ReasonerNode.on_configure` start the sidecar at bringup instead of on the first tick, overlapping the model load with other startup work. Idempotent: reuses a still-booting child instead of duplicate-spawning. (L155)
  - `warm(self) -> None` (L342) — Start the managed sidecar now instead of on the first tick.
  - `select_tool(self, *, context_text, palette, system_prompt=DEFAULT_SYSTEM_PROMPT) -> ReasonerToolCall` (L361) — Ensure the managed server is up, then run the OpenAI-compatible path.
  - `describe_image(self, *, image_jpeg, question) -> str` (L376) — Ensure the managed server is up, then run the OpenAI-compatible path.
  - `close(self) -> None` (L328) — Terminate the server child if (and only if) we spawned it.

### `python/reasoner/src/openral_reasoner/palette.py`
_Closed-set `ToolPalette` + builder. Three tool variants (reload_gst_pipeline / lifecycle_transition / emit_prompt) are always available; `execute_skill` is gated by the installed-rSkill registry filtered by `RobotCapabilities` + license posture._

- `class RSkillToolEntry(BaseModel)` (L240) — Frozen per-skill record surfaced to the LLM as one tool: `rskill_id`, `description`, `actions`, `objects`, `scenes`, mirrored from the matching `RSkillManifest` at palette-build time.
- `class ContinuousDetectorEntry(BaseModel)` (L110) — Frozen coverage record for a `mode: continuous` detector, surfaced to the LLM as coverage (not a tool) so it can read tracked objects from world state and reserve `locate_in_view` for the long tail.
- `class OnDemandDetectorEntry(BaseModel)` (L217) — Frozen record for a `mode: on_demand` open-vocab locator, surfaced as a selectable `locate_in_view` option — a read-only prompt-able tool, never an ExecuteRskill policy.
- `detector_alias(rskill_name) -> str` (L138) — Short LLM-/operator-facing detector id: strips the `OpenRAL/` org + `rskill-` kind prefixes. Single source of truth for the alias the reasoner routes on.
- `detector_service_segment(alias) -> str` (L151) — ROS-safe service-namespace segment for an alias (hyphens → underscores), so the locate service lives at `/openral/perception/<segment>/locate_in_view`.
- `locate_in_view_service(detector, *, default="") -> str` (L200) — Resolves the `locate_in_view` service for a (possibly empty) selector: empty → `default`; unresolved alias → the legacy single-detector service; else the namespaced service.
- `resolve_locate_in_view_detector(detector, *, known_aliases) -> str` (L162) — Normalizes an LLM-supplied `detector` value to `""` when it names no configured on-demand locator, so an unrecognised alias degrades to the deployment default instead of a service that never existed.
- `class ToolPalette(BaseModel)` (L279) — Frozen palette presented to the LLM each tick: primary `skills`, back-compat `execute_rskill_ids` (auto-derived), sensor/node ids, `continuous_detectors` (coverage), and availability flags gating the read-only tools — `spatial_memory_available` (`recall_object`/`resolve_place`), `detector_available` (`locate_in_view`, with `on_demand_detectors` as its options), `scene_query_available` (`query_scene`, independent of detection), `memory_available` (`memory_write`/`memory_search`). A cross-validator rejects `skills`/`execute_rskill_ids` passed with disagreeing ids.
- `build_tool_palette(*, installed_skills, robot_capabilities, sensor_ids=(), node_ids=(), commercial_deployment=False, spatial_memory_available=False, detector_available=False, scene_query_available=False, task_progress_available=False, memory_available=False) -> ToolPalette` (L412) — Builds the palette: a skill qualifies by role/kind/capability/embodiment/license checks. A `continuous` detector becomes coverage only; `on_demand` becomes a `locate_in_view` option; `segmenter` is dropped and surfaced nowhere. Output is id-sorted for a deterministic tool schema.
- `task_space_disagreement(manifest, description, hal_mode, legacy_ok) -> str | None` (L52) — Warn-only shadow gate: compares the canonical `task_space_compatible` verdict against the caller's `legacy_ok`, returning a warning string only on disagreement (`None` otherwise, including for non-actuating skills). Surfaces cross-layer mismatches without changing the drop/publish decision.

### `python/reasoner/src/openral_reasoner/spatial_query.py`
_Read-only spatial-memory query bridge: maps a `RecallObjectTool` / `ResolvePlaceTool` to a query, runs it against an injected backend, and renders an LLM-readable result. Layer-4 module; backend is duck-typed, not imported from `openral_world_state`._

- `class SpatialMemoryQuerier(Protocol)` (L49) — Read-only query surface; structurally satisfied by `openral_world_state.SpatialMemory`.
  - `recall_object(self, query, *, now_ns) -> RecallObjectResult` (L52) — Recall objects matching `query` (empty result = not found).
  - `resolve_place(self, query, *, from_node_id=None) -> ResolvePlaceResult` (L56) — Resolve a place/room/agent reference (raises `ROSObjectNotInMemory` on miss).
  - `to_scene_graph(self) -> SceneGraph` (L62) — Immutable snapshot of the current graph (for telemetry / dashboard).
- `SpatialQueryTool: TypeAlias` — `RecallObjectTool | ResolvePlaceTool` (the read-only ReasonerToolCall variants this bridge dispatches).
- `recall_object_tool_to_query(call) -> RecallObjectQuery` (L67) — tool → query mapper.
- `resolve_place_tool_to_query(call) -> ResolvePlaceQuery` (L72) — tool → query mapper.
- `format_recall_object_result(query_text, result, *, blocked_node_ids=frozenset()) -> str` (L81) — Render results as LLM-readable text; a miss is reported as text, never a fabricated pose. `blocked_node_ids` renders a match whose approach failed grid refinement as BLOCKED instead of a pose.
- `format_resolve_place_result(reference, result) -> str` (L125) — render a `ResolvePlaceResult` as an LLM-readable text block.
- `run_spatial_query_detailed(call, querier, *, now_ns, from_node_id=None, refine_approach=None) -> SpatialQueryOutcome` (L150) — Execute a read-only tool call; render text and report whether the query matched, so a miss can be escalated without re-parsing the rendered text.
- `class SpatialQueryOutcome(NamedTuple)` (L136) — `(text, found)`. `found` is `True` when `recall_object` had ≥1 match (even if grid-BLOCKED) or `resolve_place` resolved; drives the reasoner's recall→`locate_in_view` escalation.
- `run_spatial_query(call, querier, *, now_ns, from_node_id=None, refine_approach=None) -> str` (L221) — Execute a read-only tool call and render the result; `ROSObjectNotInMemory` → "not in memory". `refine_approach` (duck-typed, so this L4 module never imports L2) adjusts each `recall_object` match's approach viewpoint; `None` marks it BLOCKED. Thin wrapper returning only `.text`.
- `ApproachRefiner` (TypeAlias = `Callable[[ApproachViewpoint, tuple[float, float, float]], ApproachViewpoint | None]`) — occupancy-grid refinement callback contract; the reasoner node wires `refine_approach_pose` over its latched `/map` subscription.

### `python/reasoner/src/openral_reasoner/active_search.py`
_Bounded active object search over the scene graph (pure-Python, `openral_core` only)._

- `class SearchBudget(BaseModel)` (L25) — frozen; `max_candidates` (1–50), `max_attempts` (1–50). The bound.
- `class SearchCandidate(BaseModel)` (L39) — `place_node_id, goal: Pose6D, open_container_id: str | None, reason, rank ∈ [0,1]`.
- `plan_active_search(graph, *, target_text, budget) -> list[SearchCandidate]` (L60) — Ranked frontier of places to check (occluding containers, then containers, then places), truncated to `budget.max_candidates`; `[]` when nowhere to search (hands off to a human). The LLM prioritizes among candidates.
- `class SearchProgress` (L114) — attempt counter against a `SearchBudget`. The runaway bound.
  - `attempts(self) -> int` (L128) — Number of search steps consumed so far.
  - `exhausted(self) -> bool` (L133) — True once the attempt budget is spent (hands off to a human).
  - `record_attempt(self) -> bool` (L137) — Consume one attempt; `True` while budget remains, else `False`.
  - `reset(self) -> None` (L142) — Reset the counter (e.g. on a fresh operator goal).
- `format_search_frontier(candidates, target_text) -> str` (L147) — LLM-readable frontier text (empty → "hand off to a human").

### `python/reasoner/src/openral_reasoner/completion.py`
_VLM adjudication helpers (pure, no rclpy). Importable and testable without a ROS install._
- `COMPLETION_QUESTION: str` (L161) — prompt template for the ambiguous reward band, formatted with `task=active.text`.
- module constant `_NEGATIONS: tuple[str, ...]` (L165) — negation tokens `parse_yes_no` checks for.
- module constant `_AFFIRMATIVES: tuple[str, ...]` (L183) — affirmative tokens `parse_yes_no` checks for.
- `parse_yes_no(answer: str) -> bool` (L196) — Parse a VLM free-text answer: `True` iff it contains an affirmative token without a negation token. Token-based, not substring, to avoid false-completing on `"No. It is done."` or `"abandoned"` (contains "done"); typographic apostrophes fold to ASCII first. `False` on empty or ambiguous input.
- `image_msg_to_jpeg(*, data: bytes, height: int, width: int, encoding: str, flip_180: bool = False) -> bytes` (L253) — Convert a raw `sensor_msgs/Image` payload to JPEG via numpy + PIL (no cv_bridge); supports `"rgb8"`/`"bgr8"`. `flip_180` rotates 180° before encoding to match bottom-up LIBERO/MuJoCo frames. Raises `ValueError` on unsupported encoding.
- `is_frame_fresh(*, age_s: float, max_age_s: float) -> bool` (L22) — Whether a cached completion frame is recent enough to adjudicate: `True` iff `max_age_s <= 0` or `age_s <= max_age_s`. A stale frame degrades to "cannot adjudicate", never a false verdict.
- `is_reward_wake(*, source: str, severity: int, severity_fail: int) -> bool` (L129) — Classify a `FailureTrigger` as a reward-watcher wake: `True` iff `source == "critic"` and `severity >= severity_fail`. Used to cancel an in-flight `execute_rskill` goal on the reward signal rather than waiting for `deadline_s`.
- `resolve_band_edges(*, contract_threshold: float | None, contract_floor: float | None, fallback_threshold: float, fallback_floor: float) -> tuple[float, float]` (L50) — Three-tier verdict band edges: the active reward model's `(success_threshold, check_floor)` when both present, else the system fallback (never mixed).
- `resolve_patience_s(*, override: float | None, contract_default: float | None, legacy_deadline_s: float) -> float` (L93) — Patience ceiling for a dispatch: LLM `patience_s` override > reward model's `default_patience_s` > legacy `deadline_s` (`0.0` → runner resolves its own ceiling).

### `python/reasoner/src/openral_reasoner/mission.py`
_Typed sequential task queue for multi-task deploy goals. Reasoner-internal bookkeeping (no rclpy/Pydantic); the node drives transitions, the `ContextRenderer` renders the `## MISSION` ledger._
- `TaskStatus` (TypeAlias = `Literal["pending","active","verifying","done","abandoned"]`) — subtask lifecycle; `done`/`abandoned` terminal (never re-queued).
- `VerdictAction` (TypeAlias = `Literal["complete","abandon","retry","vlm_check"]`) — the reward-gate decision: `complete` (auto-pass), `vlm_check` (ambiguous band — caller adjudicates via `describe_image`), `abandon` (ladder exhausted), `retry`.
- `DEFAULT_MAX_ATTEMPTS: int = 3` (L49) — default per-task attempt cap before the gate abandons + hands off.
- `DEFAULT_MAX_SUBDIVIDE_DEPTH: int = 2` (L52) — max re-decomposition depth; a task already at this depth is refused subdivision, terminating the ladder in human-handoff.
- `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS: int = 3` (L63) — per-task `locate_in_view` cycle budget before abandonment. Distinct from `SearchProgress`'s miss budget, which resets on a hit and so cannot bound a hit-looping search.
- module constant `_ACTIVE_STATES: frozenset[TaskStatus]` (L240) — `{"active", "verifying"}`.
- module constant `_TERMINAL_STATES: frozenset[TaskStatus]` (L241) — `{"done", "abandoned"}`.
- `evaluate_task_verdict(*, ok: bool, progress_now: float, success_threshold: float, check_floor: float, attempts: int, max_attempts=DEFAULT_MAX_ATTEMPTS, success_now: float | None = None) -> tuple[VerdictAction, str]` (L156) — Three-tier reward gate. Gates on the PROGRESS head, not the (compressed) success head: `progress_now >= success_threshold` → `"complete"`; `check_floor <= progress_now < success_threshold` → `"vlm_check"`; below floor → attempts ladder (`"abandon"` at cap, else `"retry"`). `ok=False` always goes to the ladder. `success_now` is an optional secondary signal in the verdict text, never overriding the progress band.
- `class TaskLocateBudget` (dataclass, slots) (L75) — per-task `locate_in_view` cycle budget. Held by `reasoner_node` as `_task_locate_budget`.
  - `task_id(self) -> str | None` (L128) — the task the budget is currently tracking (for persistence).
  - `count(self) -> int` (L104) — locate cycles charged against the current task so far.
  - `reset(self) -> None` (L108) — clear the budget (new task / real progress).
  - `restore(self, task_id, count) -> None` (L113) — restore a persisted counter (reasoner restart resume).
  - `charge(self, task_id) -> bool` (L132) — Count one locate cycle for the active task (auto-resets on `task_id` change); `True` once the count exceeds `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS`.
  - `reason(self, query) -> str` (L144) — specific abandonment reason for the ledger, e.g. `could not confirm 'teapot' in view after 3 locate attempts without a skill dispatch`.
- `class TaskState` (dataclass, slots) (L245) — one subtask: fields `task_id, text, status=pending, attempts=0, last_rskill_id, last_trace_id, last_verdict, depth=0` (`depth` = re-decomposition level).
- `class MissionState` (L274) — ordered queue, ≤1 task `active`/`verifying`.
  - `from_prompt(cls, text) -> MissionState` (L306) — seeds the operator goal as a single active task; the LLM decomposes via `decompose_mission`.
  - `tasks(self) -> tuple[TaskState, ...]` (L318) — snapshot of every task, in order.
  - `active(self) -> TaskState | None` (L326) — the currently `active` or `verifying` task, or `None`.
  - `is_empty(self) -> bool` (L330) — `True` when the mission carries no tasks (empty/whitespace prompt).
  - `is_complete(self) -> bool` (L334) — `True` when every task is terminal (`done`/`abandoned`).
  - `has_started(self) -> bool` (L342) — `True` when any task is terminal or the active task has been attempted; gates a safe `decompose_mission` populate-replace.
  - `record_attempt(self, *, rskill_id, trace_id=None) -> None` (L366) — note a dispatch against the active task (increments `attempts`).
  - `mark_verifying(self) -> None` (L376) — move the active task `active → verifying` (a skill returned; gating).
  - `complete_active(self, verdict) -> TaskState | None` (L382) — mark the active task `done` and activate the next pending task.
  - `abandon_active(self, reason) -> TaskState | None` (L389) — mark the active task `abandoned` (ladder exhausted / unverifiable).
  - `rearm_active(self) -> TaskState | None` (L396) — move the active task `verifying → active` so dispatch/subdivide resumes after a subdivision offer.
  - `subdivide_active(self, subtasks, *, max_depth=DEFAULT_MAX_SUBDIVIDE_DEPTH) -> TaskState | None` (L418) — Flat-splice the blocked active task in place with finer children `t<n>.1, t<n>.2, …` at `depth+1`, activate the first; returns `None` (caller falls back to `abandon_active`) with no active task, empty subtasks, or the depth bound reached.
  - `to_state_dict(self) -> dict[str, object]` (L485) — Full round-trippable snapshot (every `TaskState` field) for crash-safe ladder persistence, unlike the lossy `to_summary`.
  - `from_state_dict(cls, state) -> MissionState` (L517) — rebuild a mission from `to_state_dict` output (exact resume); raises `ValueError` on malformed input.
  - `to_summary(self) -> dict[str, object]` (L544) — JSON-able snapshot stamped on the `reasoner.tick` span as `reasoner.mission_json` for the dashboard Mission card; children carry dotted ids `t<n>.1` for indenting under their parent.
  - `render(self) -> str` (L573) — the `## MISSION` ledger: done ✓ / active ▶ / verifying ? / abandoned ✗ + pending count; children indented by depth.

### `python/reasoner/src/openral_reasoner/node_policy.py`
_Pure prompt-handling policy for the reasoner node (no rclpy) — the two consequential `_on_prompt` decisions, single-sourced and unit-testable without a ROS install._

- module constant `CASCADE_PROMPT_SOURCES: frozenset[str]` (L29) — the six frame_ids of the reasoner's own cascade re-prompts (`spatial_memory`, `detector`, `scene_vlm`, `reward_monitor`, `memory`, `mission`). Both the search-bound/streak reset guard and the mission-rebuild guard key off this one set.
- `should_rebuild_mission(source, metadata_json, mission) -> bool` (L34) — Whether an inbound prompt replaces the mission queue: cascade → never; no/empty/finished/not-yet-started mission → yes; a genuinely in-progress mission → only with top-level `"new_goal": true` metadata (an operator reply must not silently discard the in-flight queue).

### `python/reasoner/src/openral_reasoner/persistence.py`
_Crash-safe ladder persistence (pure, no rclpy): the mission ledger + every replanning-ladder bound snapshotted after each mutation and reloaded at configure, so a restart resumes the ladder instead of resetting every cap mid-mission._

- module constant `_SCHEMA_VERSION: Literal["0.1"]` (L33) — on-disk schema version stamped into every snapshot; `load_ladder_state` refuses a mismatch.
- `class ReasonerLadderState` (Pydantic v2, extra-forbid) (L36) — versioned on-disk boundary carrying everything a restarted reasoner needs: `mission: MissionState | None`, `subdivide_offered: set[str]`, `collective_nudges: dict[str, int]`, `locate_task_id: str | None`, `locate_count: int`.
- `save_ladder_state(path, state) -> None` (L93) — atomic JSON write (tmp + `os.replace`; a crash mid-write leaves the previous snapshot intact). Versioned (`schema_version: "0.1"`).
- `load_ladder_state(path) -> ReasonerLadderState | None` (L115) — `None` on absent / corrupt / version-mismatched snapshot; caller starts fresh and logs rather than resuming from a bad snapshot.

### `python/reasoner/src/openral_reasoner/memory.py`
_The self-maintained `MEMORY.md` file model: persistent semantic memory (preferences, lessons, home facts, object-location log, open tasks). Summarized inline in `context.py`'s entry above; this is the file's own symbol inventory._

- module constant `_SECTION_TITLES: dict[MemorySection, str]` (L28) — the five section headings (`"Home Map / Places"`, `"User Preferences"`, `"Learned Lessons / Corrections"`, `"Object-Location Log"`, `"Open Tasks / Commitments"`) MEMORY.md is organized under.
- module constant `_TITLE_TO_SECTION: dict[str, MemorySection]` (L35) — reverse of `_SECTION_TITLES`, for parsing `from_markdown`.
- module constant `_ENTRY_RE` (L38) — regex matching one round-trip-stable MEMORY.md entry line (`- [imp:<float> ts:<iso> st:<status>] <content>`).
- module constant `_SCHEMA_VERSION: str = "0.1"` (L40) — on-disk MEMORY.md schema marker.
- `class MemoryEntry` (L53) — One remembered fact in a `MemoryStore`.
  - `render_line(self) -> str` (L62) — Render the entry as one round-trip-stable MEMORY.md line.
- `class MemoryStore` (L68) — Ordered set of `MemoryEntry` rendered to / parsed from `MEMORY.md`.
  - `from_markdown(cls, text) -> MemoryStore` (L78) — Parse a `MEMORY.md` body. Unrecognized lines are ignored (lenient).
  - `to_markdown(self) -> str` (L113) — The full on-disk `MEMORY.md` (title + schema marker + sections).
  - `to_context_block(self, *, cap=None) -> str` (L132) — The `## MEMORY` block the reasoner injects into its context; `cap` renders only the top-`cap` entries by importance then recency.
  - `consolidate(self) -> list[MemoryEntry]` (L157) — Merge exact-duplicate facts, keeping the best; return the removed entries for archiving.
  - `entries(self) -> tuple[MemoryEntry, ...]` (L180) — A snapshot of the current entries, in order.
  - `apply(self, *, op, section, content, importance, target, now) -> MemoryEntry | None` (L186) — Apply one explicit `add`/`update`/`supersede`/`delete` edit. Returns an entry to archive (or `None`).
  - `search(archive, *, query, section, limit) -> list[MemoryEntry]` (L232) — Rank archived entries by a keyword match.

### `python/reasoner/src/openral_reasoner/context.py`
_`ContextRenderer` builds the structured text snapshot the LLM consumes per tick (no pixels in v1)._

- module constant `DEFAULT_BUFFER_SIZE: int = 8` — Rolling buffer capacity per category. (L57)
- module constant `DEFAULT_PROMPT_PRIORITY: int = 10` (L62) — Default operator-prompt priority, matching `openral_prompt_router.DEFAULT_SOURCES`; human sources stamp 100 so they drain first.
- module constant `_FAILURE_ADAPTER: TypeAdapter[FailureEvidence]` (L64) — cached `TypeAdapter(FailureEvidence)` used to decode a failure buffer entry's `evidence_json`.
- module constant `_PERCEPTION_ADAPTER: TypeAdapter[PerceptionEventMetadata]` (L65) — cached `TypeAdapter(PerceptionEventMetadata)` used to decode a perception buffer entry's `metadata_json`.
- `class FailureEventRecord` (frozen dataclass) (L69) — Failure-buffer entry; fields `source, kind, severity, evidence_json, rskill_id, trace_id, stamp_ns`.
- `class PerceptionEventRecord` (frozen dataclass) (L86) — Perception-buffer entry; fields `kind, text, metadata_json, stamp_ns`.
- `class PromptRecord` (frozen dataclass) (L96) — Operator-prompt-buffer entry; fields `text, metadata_json, stamp_ns, priority=DEFAULT_PROMPT_PRIORITY`. `priority` is filled by `append_prompt` from `metadata_json["priority"]` when constructed with the default sentinel.
- `render_robot_self_model(description: RobotDescription) -> str` (L334) — Deterministic static self-model block (embodiment, dof, end-effectors, locomotion, payload, capability flags, cameras with FOV, control modes) so the LLM can judge reach/view feasibility before dispatch. Set via `ContextRenderer.set_robot_model`; rendered as `## ROBOT`.
- `render_playbooks_block(entries: list[tuple[str, str]]) -> str` (L295) — Renders the `## PLAYBOOKS` system-prompt block from `(name—trigger, PLAYBOOK.md body)` entries; `""` when none match. Playbooks guide decisions only — every motion still goes through `execute_rskill` + the safety kernel.
- `class MemoryEntry` / `class MemoryStore` (`openral_reasoner.memory`, own section below) — The self-maintained `MEMORY.md` semantic-memory model (complementary to the geometric scene graph), advisory only. Round-trips the file; renders a capped `## MEMORY` context block; applies explicit add/update/supersede/delete edits; consolidates duplicates; searches the archive. Loaded at configure by `reasoner_node._maybe_load_memory`.
- `class ExecutionEventRecord` (frozen dataclass) (L124) — Execution-feedback buffer entry; fields `rskill_id, outcome ("ok"|"failed"), summary, reflection (failures only), stamp_ns`.
- `class RewardStateRecord` (frozen dataclass) (L142) — Latest two-head reward assessment surfaced to the LLM: `progress` (closeness, the gated head), `success` (done-confidence, compressed, secondary), `progress_trend`, `success_trend`, `task`, `stamp_ns`. Rendered as `## REWARD` via `set_reward_state`/`_render_reward`.
- `reflect_on_failure(outcome_state, detail, *, timed_out: bool | None = None) -> str` (L176) — Deterministic one-line strategy hint (no LLM call) turning a raw failure into a "change approach" cue. `timed_out` comes from the typed `ExecuteRskill.Result.failure_kind` when the caller has it; `None` falls back to a deprecated substring probe over the prose.
- `reflect_on_reward_plateau(progress_now) -> str` (L229) — Reward-plateau hint (policy ran clean but progress says not done): the move is to change tactic, not to subdivide or re-issue the same instruction.
- `reflect_on_invalid_plan(detail) -> str` (L259) — Strategy hint feeding a decode error back so a model that emitted malformed JSON arguments fixes its call instead of re-issuing it verbatim.
- `reflect_on_retry_cap(tool, cap) -> str` (L283) — Strategy hint when `ReasonerCore`'s per-kind retry ladder is exhausted for `tool`.
- `class ContextRenderer` (L402) — Stateful renderer.
  - `set_robot_model(self, robot_model) -> None` (L491) — Sets/clears the static `## ROBOT` self-model; does NOT bump `seq`.
  - `set_memory_block(self, memory_block) -> None` (L500) — Sets/clears the `## MEMORY` block from the MEMORY.md store; does NOT bump `seq`.
  - `set_mission(self, mission) -> None` (L510) — Sets/clears the active task queue rendered as `## MISSION`; a new goal is an event so it DOES bump `seq`.
  - `set_in_view(self, objects) -> None` (L522) — Sets/clears the latest continuous-detector enumeration rendered as `in_view[<camera>]` in `## WORLD_STATE`; bumps `seq` only when the rendered enumeration actually changes.
  - `set_inflight_skill(self, rskill_id, *, stamp_ns=0, state='running') -> None` (L542) — Records (or clears) the in-flight `execute_rskill` goal with its phase (`"dispatching"` until accept, then `"running"`), rendered as the leading `in_flight:` line in `## EXECUTION`; state changes bump `seq`.
  - `inflight_skill(self) -> str | None` (L561) — `rskill_id` of the goal currently in flight, or `None`; also read by `ReasonerCore`'s heartbeat-idle gate.
  - `inflight_state(self) -> str | None` (L566) — Phase of the in-flight goal (`"dispatching"` / `"running"`), or `None`.
  - `note_located(self, objects) -> None` (L573) — Folds open-vocab `locate_in_view` hits into a sticky `located[<camera>]` line (latest-wins, capped) that survives the continuous detector's per-frame clobber, keeping a mislabelled goal noun grounded for decompose/dispatch; bumps `seq`.
  - `clear_located(self) -> None` (L617) — Drops the sticky open-vocab grounding on a new operator goal; bumps `seq` when non-empty.
  - `set_reward_state(self, reward) -> None` (L631) — Sets/clears the two-head reward assessment rendered as `## REWARD`; bumps `seq`.
  - `mission(self) -> MissionState | None` (L660) — the active `MissionState`, or `None`; the node mutates it in place for non-waking bookkeeping (`record_attempt`/`mark_verifying`).
  - `advance_mission(self, *, done, verdict) -> TaskState | None` (L670) — `complete_active`/`abandon_active` the active task + activate the next, bumps `seq`; `None` when the mission is finished or unset.
  - `append_failure(self, record) -> None` (L692) — pushes a failure event onto the rolling buffer; bumps `seq`.
  - `append_execution(self, record) -> None` (L697) — Pushes a skill execution outcome onto the rolling buffer; bumps `seq` so feedback wakes an idle heartbeat.
  - `clear_failures(self) -> None` (L711) — drops stale failure/execution context after `/openral/estop_cleared` so the next prompt is not poisoned by a resolved e-stop.
  - `append_perception(self, record) -> None` (L706) — pushes a perception event onto the rolling buffer; bumps `seq`.
  - `append_prompt(self, record) -> None` (L725) — priority-ordered insert; buffer-evicts the lowest-priority oldest entry on overflow; bumps `seq`.
  - `render(self, *, world_state) -> str` (L768) — Returns the deterministic text snapshot for one LLM tick. `## WORLD_STATE` covers joint_state/ee_poses/battery/diagnostics plus a 3D `scene_objects[<frame>]` line and a depth-free, pixel-space `in_view[<camera>]` line (so grounding still works with RGB-only / no octomap); `note_located` hits render as a sticky `located[<camera>]` line distinct from `in_view`. `## REWARD` (when set) carries both reward heads distinctly labelled so the LLM never blurs progress (persist-vs-replan) with success (done-ness).
  - `failures(self) -> tuple[FailureEventRecord, ...]` (L971) — snapshot of the failure buffer (oldest first).
  - `perception_events(self) -> tuple[PerceptionEventRecord, ...]` (L981) — snapshot of the perception buffer (oldest first).
  - `executions(self) -> tuple[ExecutionEventRecord, ...]` (L976) — snapshot of the execution-feedback buffer (oldest first).
  - `prompts(self) -> tuple[PromptRecord, ...]` (L986) — snapshot of the prompt buffer (oldest first).
  - `seq(self) -> int` (L991) — monotonic mutation counter `ReasonerCore` uses to short-circuit a heartbeat tick when nothing has changed since the last successful tick.
  - `drain_prompts(self, *, seen=None) -> tuple[PromptRecord, ...]` (L1005) — Pull-once, priority-desc + arrival-asc order; does NOT bump `seq`. `seen` drains only those records by identity, so a prompt that arrived mid-LLM-call and was never rendered survives for the next tick.
- module constant `_PROMPT_EXCLUDED_FIELDS: frozenset[str]` (L1068) — evidence fields `_summarise_evidence_json` drops by role rather than length (currently `{"joint_positions_rad"}`, useless to a planner).
- module constant `_PROMPT_FIELD_BUDGET: int = 48` (L1074) — char backstop past which `_summarise_evidence_json` drops a field and discloses it as `+<field>[<n>]`.
- `_summarise_evidence_json(payload) -> str` — Decode the FailureEvidence union and produce a one-line, 120-char-truncated summary; a field named in `_PROMPT_EXCLUDED_FIELDS` or past the `_PROMPT_FIELD_BUDGET` backstop is dropped and disclosed as `+<field>[<n>]`. Exclusion is by role, not length, so identity fields survive truncation instead of noise fields. (L1077)
- `_extract_priority(metadata_json) -> int` — Parse a top-level `priority` field out of a PromptStamped's metadata; returns `DEFAULT_PROMPT_PRIORITY` on missing / malformed / non-int payload.

### `python/reasoner/src/openral_reasoner/core.py`
_`ReasonerCore`, the transport-agnostic orchestrator. The ROS-side `reasoner_node` wraps this with rclpy._

- `class ReasonerTickResult` (frozen dataclass) (L120) — Tick outcome; fields `tool_call`, `error`, `elapsed_s`, `suppressed_reason` (`""`, `"min_interval"`, `"heartbeat_idle"`, `"mission_finished"`, `"retry_cap"`, `"palette_empty"`), `traceparent` (`None` when no real `TracerProvider` is installed).
- `_call_identity(call) -> str` — Canonical retry-cap identity of a tool call: the full argument payload minus `rationale` (rephrasing rationale between identical retries must not dodge the cap), JSON-serialised with sorted keys.
- `class PreparedTick` (dataclass) (L86) — In-flight state of a phased tick between `prepare_tick` and `finish_tick`: LLM inputs (`context_text`, `palette`, `system_prompt`) plus prepare-time snapshots (`started`, `seq`, `prompts`, an unattached OTel `span`, `renderer`, `force`, `tier`). The `seq`/`prompts` snapshots bound `finish_tick` to only what the model actually saw. `llm_s`/`prompt_tokens` are the only fields the LLM phase writes back, so tick latency can be attributed to the provider vs. reasoner-side overhead.
- module constant `_RETRY_CAP_EXEMPT_TOOLS: frozenset[str]` (L68) — `{"recall_object", "resolve_place", "locate_in_view", "wait"}`, the tool kinds transparent to the retry cap (see `ReasonerCore` below).
- `class ReasonerCore` (L150) — Orchestrator.
  - `retry_cap(self) -> int` (L237) — the configured consecutive-identical-call cap (read-only).
  - `streak_tool(self) -> str` (L242) — tool kind of the current consecutive-call streak (`""` when none).
  - `reset_kind_streak(self) -> None` (L246) — reset the consecutive-call counter used by the retry-cap gate.
  - `tick(self, *, world_state, renderer, palette, force=False, tier='heartbeat') -> ReasonerTickResult` (L257) — Synchronous composition of the phased API (prepare → off-thread LLM call → finish) so the executor is never starved. Enforces the min-interval and retry cap (search tools + `wait` are exempt); short-circuits on a finished mission or idle heartbeat. `force=True` bypasses every gate but the retry cap. Wraps the tick in a `reasoner.tick` OTel span.
  - `prepare_tick(self, *, world_state, renderer, palette, force=False, tier='heartbeat') -> PreparedTick | ReasonerTickResult` (L318) — Phase 1 of a tick: run the suppression gates and render the context; must run on the owning/executor thread.
  - `run_prepared_llm(self, prep) -> ReasonerToolCall` (L461) — Phase 2 of a tick: the blocking `select_tool` round-trip; the ONLY phase safe off-thread.
  - `finish_tick(self, prep, *, call=None, error=None) -> ReasonerTickResult` (L493) — Phase 3 of a tick: retry-cap ladder + bookkeeping + span close, back on the owning thread.

### `python/reasoner/src/openral_reasoner/critic_watchdog.py`
_Tier-C critic progress-stall / success watchdog — default decision core for the reserved `/openral/failure/critic` source. Pure logic, import-safe (no rclpy). Source-agnostic: any reward model emitting a higher-is-better scalar drives the same watchdog; the critic producer node routes samples through `CriticWatchdogGroup` and publishes to the failure bus on a fire. Fires on stall OR success, waking the reasoner promptly when an attempt is likely done._

- `class CriticWatchdog` (L62) — Progress-stall / success state machine; raises `ValueError` on `stall_patience < 1` or `min_delta < 0`. Fires one `CriticEvidence` in two mutually exclusive, one-shot-per-streak cases: success (score crosses threshold) or stall (`stall_patience` consecutive non-improving observations). Success takes precedence when both would fire on the same sample.
  - `critic_id(self) -> str` (L131) — Identifier of the upstream critic stamped onto emitted evidence.
  - `threshold(self) -> float` (L136) — Pass threshold; observations at or above it are recoveries.
  - `stall_patience(self) -> int` (L141) — Consecutive stalled observations required to fire.
  - `min_delta(self) -> float` (L146) — Minimum strict improvement over the running best to count as progress.
  - `observe(self, score) -> CriticEvidence | None` (L150) — Feed one progress/critic score and decide whether to fire.
  - `reset(self) -> None` (L220) — Clears running best, stall counter, and both latches (call on a reasoner context shift).
- `class CriticWatchdogGroup` (L234) — Multiplexer keying one `CriticWatchdog` per `critic_id` so several reward models share the `/openral/failure/critic` source independently.
  - `stall_patience(self) -> int` (L293) — Consecutive stalled observations each watchdog requires to fire.
  - `min_delta(self) -> float` (L298) — Minimum strict improvement each watchdog counts as progress.
  - `observe(self, *, critic_id, score, threshold) -> CriticEvidence | None` (L302) — Lazily creates a watchdog per `critic_id` (threshold bound on first sight) and delegates.
  - `known_critics(self) -> frozenset[str]` (L328) — Return the `critic_id` set seen since construction / last reset.
  - `reset(self, critic_id=None) -> None` (L332) — drop one critic's watchdog (rebinds its threshold) or all.

### `packages/openral_safety/openral_safety/supervisor_node.py`
_Day-1 Python safety envelope: `candidate_action` → `safe_action` pass-through with real per-control-mode envelope checks, the estop latch/reset pair, and the latched SafetyStatus topic. Reserves the node name and topic surface for the future C++ kernel; any addition of enforcement beyond this file requires safety-WG sign-off._

- `class SafetyPassthroughNode(LifecycleNode)` (L134) — Owns `/openral/candidate_action → /openral/safe_action` plus the estop latch/reset pair and the `SafetyStatus` topic.
  - `__init__(node_name="openral_safety") -> None` (L155)
  - `on_configure(state) -> TransitionCallbackReturn` (L218) — Opens the publishers, subscriptions, service, and diagnostics heartbeat.
  - `on_activate(state) -> TransitionCallbackReturn` (L308)
  - `on_deactivate(state) -> TransitionCallbackReturn` (L340)
  - `on_cleanup(state) -> TransitionCallbackReturn` (L350)
  - `on_shutdown(state) -> TransitionCallbackReturn` (L382)
  - `_on_candidate_action(msg) -> None` (L388) — Subscribes `/openral/candidate_action`. Drops the candidate and re-fires `/openral/estop` on an envelope violation; drops silently while already latched; otherwise forwards unchanged on `/openral/safe_action`. Emits a `safety.check` OTel span per candidate.
  - `_envelope_violation(msg) -> tuple[str | None, str]` (L445) — Dispatches on `control_mode`: joint modes get the position-limit check; Cartesian/twist/gripper modes each get their own bound check. Every bound parameter defaults to `-1.0` (no enforcement declared, skip).
  - `_handle_violation(msg, *, kind, reason) -> None` (L706) — Drops the chunk, latches the estop, publishes `std_msgs/Empty` on `/openral/estop`, and updates the latched `SafetyStatus`.
  - `_on_external_estop(_msg) -> None` (L735) — Subscribes `/openral/estop` (defense in depth): any external estop publication latches this node too, independent of its own checks.
  - `_on_estop_reset(request, response) -> object` (L762) — Exposes `/openral/estop_reset` (`std_srvs/Trigger`); clears the latch only once `estop_reset_cooldown_s` (default 0.5 s) has elapsed since the last estop.
- `SafetySupervisorNode` (L846) — Back-compat alias of `SafetyPassthroughNode`, not a separate skeleton.
- `SAFETY_STATUS_TOPIC: str` (L69) — `/openral/safety_status`, the latched current-safety-state topic (RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1), published alongside — never instead of — `/openral/estop`. Same contract the C++ kernel publishes.
- `SAFETY_STATUS_HEARTBEAT_S: float` (L76) — 1.0 s liveness refresh; `header.stamp` is re-stamped at this cadence even when nothing changed, so a durable value can be told apart from a dead publisher's leftover state.
- `main(args=None) -> int` (L849) — Entry point for `ros2 run openral_safety supervisor_node`.
- module constant `_KERNEL_LABEL_PASSTHROUGH: str = "passthrough"` (L54) — `safety.kernel` span/log label this Python passthrough stamps (vs. the future C++ kernel's own label).
- module constant `_WORKSPACE_VIOLATION_KINDS: frozenset[str]` (L93) — `{"workspace", "gripper_range"}`, envelope-violation kinds `_envelope_violation` can return.
- module constant `_RATE_VIOLATION_KINDS: frozenset[str]` (L94) — `{"cartesian_step", "cartesian_step_rot", "ee_linear_speed", "ee_angular_speed", "base_linear_speed", "base_angular_speed"}`, the rate/speed-bound violation kinds.
- module constant `DEFAULT_ESTOP_RESET_COOLDOWN_S: float = 0.5` (L82) — default cooldown `_on_estop_reset` enforces since the last estop before clearing the latch.

### `cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp`
_Allocation-free attached-payload contact handling, plus the staged 26-DOP → exact-convex-hull narrow phase for the arm-link-vs-world-voxel check ([`collision-hull-narrow-phase.md`](../reference/collision-hull-narrow-phase.md), [`collision-tight-geometry.md`](../reference/collision-tight-geometry.md))._

- `struct CollisionHit` — One collision check's outcome and, on a hit, its E-stop evidence. `link_a`/`link_b`/`min_distance` name the single deepest tripped pair; `sweep_min_distance` is the separate sweep-wide minimum and never supplies the evidence distance. `advisory` is the one field that is itself a decision — refuses the action without latching, for an attached payload only, never for a robot link.
- `hull_hull_distance(a, b, margin, fallback, depth_is_box_bound = nullptr) -> double` — Stage 2 for a self-collision pair: GJK on the two links' exact hulls, replacing the OBB SAT when both ship stage-2 geometry. The optional out-parameter reports whether the returned depth is the (looser but sound) box's bound rather than a true hull measurement — GJK proves an overlap but does not size it. Disclosure only; never gates the trip decision.
- `check_contact_force_gate(attached) -> ContactForceGateResult` — Declaration-scoped contact-force gate for a place, where geometric clearance is zero by design. Only ever adds a refusal, evaluated after every geometric check has passed. Arms only with a valid live declaration, a matching object/witness, and calibration. `kMaxContactForceThresholdN = 140.0` mirrors the schema's cap (ISO/TS 15066 Table A.2). Allocation-free.
- `place_approach_allowance_cap(resolution) -> double` — The place approach allowance's cap at a live voxel `resolution`: `min(1.5 × voxel, 4 cm)`, `0.0` for an unusable resolution. `constexpr`, header-inline; the single definition the geometry and the log line both quote.
- `place_approach_allowance(grid, object_index, center) -> double` — Margin reduction the live place declaration grants this payload against this occupied cell, or `0.0`. Returns the reduction only when the region is valid, the object is in `object_mask`, the box is finite/non-degenerate, and the cell centre is inside it (exact point-in-OBB). Reduces the margin only — deepening past it still stops, and arm-vs-world is untouched.
- `enum class PlaceRegionStatus` — Outcome of a place-region ingest: `kOk`, `kNoObject` (ordinary pre-grasp state), `kBadPose`, `kBadExtents`, `kDegenerate`, `kOversize`, `kOversizeVolume`, `kBadGeometry`, `kGeometryOverflow`. Every non-`kOk` value means no allowance.
- `ingest_place_region(pose, half_extents, object_mask, out) -> PlaceRegionStatus` — Validate a producer-supplied region into `out`, fail-closed toward no allowance (a bad region can only make the kernel more permissive). Rejects a non-finite pose/extent, an over-large side or volume, and an empty `object_mask`, each under its own status. Not on the hot path.
- `ingest_place_target_geometry(geometry, store, region) -> PlaceRegionStatus` — Lowers the declared target's own collision primitives into an already-validated `region`, so a payload can be adjudicated against the modelled receptacle instead of its voxel quantisation. Call only after the box validates; anything but `kOk` resets the whole region. `store` must be sized to `kMaxPlaceTargetPrimitives` (64) and outlive every check reading the region. Rejects a malformed shape or an over-cap list — a half-modelled receptacle is worse than none. Not on the hot path.
- `place_target_distance(region, prim, prim_xf) -> double` — Exact surface distance from one attached payload primitive to the declared target's own geometry; `+infinity` when the declaration ships none, collapsing to the pre-existing voxel-only path. Allocation-free.
- `place_region_status_reason(status) -> const char*` — Stable snake_case token for a `PlaceRegionStatus`, used as the `reason=` key of the node's place-region log lines. Never null.
- `support_contact_exempts(object, object_xf, center, resolution, slack) -> bool` — Whether object `i`'s attested support contact explains this occupied cell: the cell must be inside the attested patch laterally and no higher above the plane than the voxel's own projected half-width plus attested depth, slack, and one voxel of co-planar headroom. Mirrored in lockstep by `support_patch_withholds` ([03-world-state.md](03-world-state.md)). Caller must have already established the witness is live.
- `update_support_contact_witnesses(attached, scratch, grid, live_mask, margin) -> uint8` — Refreshes the per-object support-witness latch against the measured configuration; a bit clears on separation and is never re-set here, so re-contact after a lift is a new violation. Returns 0 (fail-closed) when the map is unusable. Reads occupancy, so it is partitioned against the Layer-2 payload clearing that withholds the attested patch's cells precisely so this latch keeps something to measure.
- `update_attached_voxel_contacts(attached, scratch, grid, contact_mask, contact_distance, mask_capacity, distance_capacity, snapshot) -> bool` — Snapshots the occupied voxel contacts present at attachment and refreshes them against the measured payload pose; only that embedded subset (the payload's own uncleared occupancy residue) still exempts anything, no bits added after the snapshot.
- `place_advisory_depth(resolution) -> double` — `min(0.2 × resolution, 5 mm)`, the depth past the place approach allowance within which an attached payload's world contact is refusable rather than latching. `0.0` for an unusable resolution, collapsing to today's latched stop.
- `check_voxel_collision(model, scratch, grid, margin) -> CollisionHit` — Checks every FK'd robot primitive against the occupied cells of a dense voxel `grid`. Never sets `advisory` — a robot link against the world is always a latching stop. A boxed link with tight geometry runs a staged narrow phase (26-DOP, then exact hull GJK) that only tightens, never loosens, the reported distance versus the plain OBB. Allocation-free.
- `kDopAxes` / `kDopAxis[kDopAxes][3]` — The 26-DOP's 13 unit axis directions (3 face, 4 corner, 6 edge; each a `lo`/`hi` slab) in the owning OBB's local frame; the first three are the box's own axes. Mirrored in Python as `openral_core.schemas.DOP_AXES` ([14-duplication-watch.md](14-duplication-watch.md) item 12).
- `kMaxTightHullVertices = 320` — Hard ceiling on a stage-2 hull's vertex count, enforced at configure time; a cost bound (the exhaustive-scan support function is linear in vertex count), not a geometry choice. A hull over the cap runs stage 1 only. Mirrored as `MAX_TIGHT_HULL_VERTICES`.
- `kTightContainmentEpsilonM = 1e-9` — Floating-point slack `validate_tight_geometry` allows checking a stage-2 vertex against its own DOP slabs (tangent by construction). The DOP-inside-OBB check itself takes no slack. Mirrored as `TIGHT_CONTAINMENT_EPSILON_M`.
- `kGjkMaxIterations = 24` / `kGjkTolerance = 1e-9` — Iteration ceiling and convergence tolerance for the stage-2 GJK loop; hitting the ceiling is not a failure — every iteration's result is a lower bound, so a truncated run is simply more conservative.
- `kMaxStage2PerCheck = 32` — Ceiling on stage-2 invocations in one `check_voxel_collision` call across every link; past it the check keeps refining with stage 1 and the shipped `box_box_distance`, so the cap can only make the answer more conservative. Changing it changes a real-time bound — belongs in the hazard log.
- `struct LinkHull` — Tight convex geometry refining ONE `Obb` in `CollisionModel::boxes`, in that box's local frame. Fields: `dop_lo`/`dop_hi` (stage 1, always on) and `vertex_first`/`vertex_count` (stage 2, CSR offset into `hull_vertices`; `vertex_count == 0` means stage 1 only).
- `CollisionModel::box_hull` / `hulls` / `hull_vertices` — The three `CollisionModel` members carrying tight geometry: `box_hull` (parallel to `boxes`) indexes into `hulls` for the tight representation refining that box, `-1` for none. An empty `box_hull` means no box carries tight geometry — the pre-existing behaviour every capsule-lowered robot keeps.
- `AttachedPrimitive::hull_index` / `AttachedModel::hulls` / `AttachedModel::hull_vertices` — The payload twin of the robot's tight-geometry fields, for the payload-vs-world-voxel check only. Unlike a robot's fixed geometry, a payload's hull arrives on every world-state message and is refilled by `ingest_attached_objects` without allocating. Ingest fails DOWN, not closed: an unprovable refinement is dropped and the primitive checked as the plain box (refusing the whole attachment would drop its geometry — the unsafe direction). An over-budget hull is refused whole, never truncated.
- `enum class TightGeometryStatus` — Why `validate_tight_geometry` refused a model: `kOk`, `kBadArity`, `kEscapesBox`, `kHullEscapesDop`, `kTooManyVertices`, `kDegenerate`. Anything but `kOk` is fail-closed: the kernel keeps the shipped OBB narrow phase for every link.
- `validate_tight_hull(hull, half_extents, vertices) -> TightGeometryStatus` — The containment proof for one tight representation against one box: slabs finite, `hull ⊆ 26-DOP ⊆ box`, vertex count in budget, CSR slice in range. Shared by the robot's configure-time check and the payload's per-message ingest, so both are held to the same standard. Allocation-free.
- `validate_tight_geometry(model, offending_box) -> TightGeometryStatus` — Proves, at configure time, that every declared tight representation is a subset of the shipped OBB it refines, keeping `check_voxel_collision`'s broad-phase reach correct. `offending_box` receives the offending index on refusal. Not on the hot path; allocation-free.
- `tight_geometry_status_reason(status) -> const char*` — Stable snake_case token naming a `TightGeometryStatus`, for the kernel's tight-geometry log line. Never null.
- `struct TightPose` / `tight_pose_init(hull, hull_vertices, box_xf, half_side, out) -> void` — Per-link-pose constants for the staged narrow phase, hoisted once per checked configuration outside the cell loop. Holds pointers into the caller's `CollisionModel`; must not outlive it. Allocation-free.
- `dop_cell_lower_bound(pose, center, half_side, best_axis) -> double` — Stage 1: a conservative lower bound on the surface distance between the link's 26-DOP and an occupied voxel cube, via separating-axis over 16 directions. `best_axis` (nullable) receives the winning normal, seeding stage 2's search. Allocation-free.
- `struct GjkWitness` — Stage-2 warm-start carried between adjacent cells/steps. Stores hull vertex indices and cube-corner sign codes (hints, re-evaluated per cell), never cached Minkowski-difference points; `n == 0` is a cold start.
- `hull_cell_distance(pose, center, half_side, seed_dir, margin, fallback, witness) -> double` — Stage 2: a lower bound on the surface distance between the link's exact convex hull and an occupied voxel cube, seeded from stage 1's separating axis. Every returnable value is a supporting-hyperplane lower bound, so the iteration cap and early exit can only make the answer more conservative; on overlap, `fallback` (stage 1's figure) is returned instead. Allocation-free (fixed-size stack simplex).
- `hull_hull_distance(a, b, margin, fallback) -> double` — Stage 2 for a self-collision pair: the same GJK with the cube's corners replaced by a second hull's vertices. `check_self_collision` calls it for an OBB pair within `margin` when both links ship a stage-2 hull; the OBB stays the broad phase and fallback, so a manifest without `tight_geometry` is bit-for-bit unchanged. Exists because the OBB alone cannot answer for two interleaving links whose real geometry only sometimes interpenetrates. Does not stop at the first bound clearing `margin` (its result is the reported `sweep_min_distance`). Cold-started. Allocation-free.

### `packages/openral_safety/openral_safety/envelope_loader.py`
_Pydantic → C++ kernel ROS-param bridge._

- module constant `_ACTUATED_JOINT_TYPES: frozenset[str]` (L108) — `{"revolute", "prismatic", "continuous"}`, joint types counted as actuated DOF.
- `merge_deploy_envelope(robot_env, deploy) -> SafetyEnvelope` (L249) — Apply explicit `DeployScene.safety` fields to the robot ceiling with tighten-only validation; omitted fields keep robot manifest values.
- `class EnvelopeIntersection` (L59) — The numerical product of `robot.safety ∩ skill.envelope`.
- `compute_intersection(robot, skill, *, deploy=None) -> EnvelopeIntersection` (L292) — Robot ceiling ∩ optional deploy/workcell envelope ∩ optional skill envelope; rejects (never clamps) any deploy or skill safety field that loosens the robot ceiling.
- `kernel_params_from_envelope(envelope) -> dict[str, object]` (L399) — Canonical scalar/AABB envelope → kernel ROS-param dict.
- module constant `_JOINT_KIND_CODE: dict[JointType, int]` (L462) — `{REVOLUTE: 1, CONTINUOUS: 1, PRISMATIC: 2}`, the kernel's per-joint kind code used when flattening the kinematic chain.
- `collision_params_from_description(robot, *, margin_m=None) -> dict[str, object]` (L609) — Flatten collision geometry + ACM + the kinematic chain into the kernel's collision params. Raises `ROSConfigError` unless the links form one connected tree, and again if it cannot lower a primitive's shape, rather than silently mis-approximating it. A boxed link with `tight_geometry` also lowers the staged-narrow-phase DOP/hull arrays.
- `merge_extra_allowed_pairs(params, pairs) -> dict[str, object]` (L805) — Additive deploy-scene ACM merge. Resolves link names, rejects unknown/self pairs, dedupes order-insensitively, no-ops when self-collision geometry is disabled.
- `ee_link_index_from_collision_params(params) -> int` (L849) — Picks the predictive-Cartesian EE control link (the kinematically deepest collision link) for the kernel's Jacobian look-ahead; `-1` when no collision model (predictive disabled, reactive floor only).

### `packages/openral_safety/openral_safety/mjcf_lowering.py`
_Offline MJCF → kernel collision-params lowering; imports `mujoco` lazily._

- module constant `_MJ_GEOM_PLANE: int = 0` (L35)
- module constant `_MJ_GEOM_SPHERE: int = 2` (L36)
- module constant `_MJ_GEOM_CAPSULE: int = 3` (L37)
- module constant `_MJ_GEOM_CYLINDER: int = 5` (L38)
- module constant `_MJ_GEOM_BOX: int = 6` (L39) — MJCF `geom_type` enum values `lower_collision_params` reads from the compiled model.
- module constant `_LOWERABLE_GEOM_TYPES: frozenset[int]` (L80) — `{sphere, capsule, cylinder, box}`, the geom types this lowering path handles (mesh/plane skipped).
- `lower_collision_params(model, joint_names, *, margin_m=0.0) -> dict[str, object]` (L270) — Lowers a compiled `mujoco.MjModel` to the kernel's collision params: per-link origins, every collidable primitive as a capsule, `dof_index` assigned by movable-joint order (not by MJCF joint name, which can differ from the manifest), ACM = parent↔child + MJCF excludes + a neutral-pose overlap sweep.

### `packages/openral_safety/openral_safety/kernel_predicates.py`
_The Python mirror of the C++ kernel's narrow phase (`cpp/openral_safety_kernel/src/collision.cpp`), batched over a leading configuration axis. Anything offline that decides what the kernel will do must ask the same question with the same geometry. Every dispatch is exhaustive over `CollisionShape` and ends in a typed `ROSConfigError`, never a silent fallback._

- module constant `_DEGENERATE_AXIS: float = 1e-9` (L68) — near-zero-length-axis tolerance used when normalizing a separating axis.
- module constant `_TERNARY_ITERS: int = 48` (L70) — ternary-search iteration count for `box_capsule_distance`'s convex point→segment minimization.
- `box_box_distance(a_tf, a_half, b_tf, b_half) -> NDArray` (L94) — 15-axis separating-axis surface gap between two oriented boxes (port of the C++ kernel); the max over the 15 candidates is the exact signed separation. `<= 0` means overlap.
- `box_capsule_distance(box_tf, box_half, cap_tf, cap_radius, cap_half_length) -> NDArray` (L145) — Box↔capsule gap: the capsule's segment ternary-searched against the box in its local frame, minus the radius.
- `capsule_distance(a_tf, a_radius, a_half_length, b_tf, b_radius, b_hl) -> NDArray` (L186) — Capsule↔capsule gap: segment-segment minimum distance minus both radii. A sphere is a capsule with `half_length = 0`.
- `shape_distance(a_shape, a_tf, b_shape, b_tf) -> NDArray` (L247) — Dispatch reproducing the kernel's self-collision routing (box↔box, box↔capsule either order, else capsule/sphere); the single entry point an offline sweep should use.
- `shape_max_extent_m(shape) -> float` (L288) — Farthest any point of `shape` lies from its own origin; bounds how far a link's geometry sits from a joint axis.
- `bounding_capsule_segment(shape, origin) -> tuple[p0, p1, radius]` (L303) — A capsule that contains `shape`, for consumers (today only `cumotion_config`) that can only express capsules; a box takes its longest axis as the segment and the half-diagonal of the rest as the radius.
- `eroded_shape(shape, eps_m) -> CollisionShape | None` (L363) — Minkowski erosion by an `eps_m` ball; `None` when empty, which a caller must read as "cannot certify", never as "no overlap".

### `packages/openral_safety/openral_safety/urdf_lowering.py`
_Offline URDF(+SRDF) → manifest collision-model lowering tool; lazy-imports `yourdfpy` / `trimesh` (the `[lowering]` group). Populates `robot.yaml`'s collision geometry + ACM (the hand-reviewable manifest path), distinct from `mjcf_lowering` (the runtime MJCF path)._

_An ACM entry removes a self-collision check, so every rule here fails toward fewer entries. The always-colliding proof (`_certified_always_colliding`) is a branch-and-bound Lipschitz-bounded sweep over the pair's relative-DoF subspace whose cost knobs can only withhold an entry, never certify an unsound one._

- `parse_srdf_disabled_pairs(srdf_path) -> set[frozenset[str]]` (L69) — Parse a MoveIt SRDF's `<disable_collisions>` rows into unordered link pairs (the ACM).
- `fit_capsule_to_vertices(vertices) -> tuple[CapsuleShape, tuple[float×6]]` (L112) — PCA bounding capsule containing every vertex, a conservative over-approximation; returns the shape + link-frame origin.
- `lower_link_geometry(urdf_path) -> list[LinkCollisionGeometry]` (L214) — One conservative capsule/sphere per URDF link with a `<collision>` (box→8 corners, cylinder→cap rims, sphere→exact, mesh→PCA-fit trimesh vertices).
- module constant `_CERTIFY_MAX_DOF: int = 5` (L337)
- module constant `_CERTIFY_MAX_CELLS: int = 400_000` (L341)
- module constant `_CERTIFY_MAX_LEVELS: int = 96` (L344)
- module constant `_COARSE_NODES: int = 7` (L346) — the four `_certified_always_colliding` cost knobs (see the module intro); hitting any withholds the ACM entry, never soundness.
- module constant `_MJCF_RNG_SEED: int = 20260610` (L354)
- module constant `_MJCF_N_SAMPLES: int = 2000` (L355) — defaults for `lower_robot_from_mjcf`'s seeded always-colliding sweep.
- `acm_for_geometry(urdf_path, geoms, *, srdf_path=None, margin_m=0.0) -> set[frozenset[str]]` (L626) — The ACM for the exact per-link primitives the kernel will load, decided with the kernel's own predicates at the robot's own margin. `ACM = adjacent ∪ always-colliding ∪ [SRDF-disabled if given]` — geometry can prove a pair always collides but never that it never does, so without an SRDF a sometimes-colliding pair stays checked. Deterministic, no RNG.
- `sample_acm_from_urdf(urdf_path, *, margin_m=0.0) -> set[frozenset[str]]` (L689) — No-SRDF fallback: `adjacent ∪ always-colliding` from the URDF's own geometry. Conservative against a precise-mesh SRDF (its disabled set is a subset, never false-permissive). The name is now a misnomer (nothing is sampled) but is kept as public API/provenance value.
- `class LoweredCollisionModel` (L717) — Frozen dataclass result: `collision_geometry`, `allowed_collision_pairs` (sorted tuples), `acm_source` (`"srdf"`|`"sampling"`|`"mjcf"`), `srdf_path`, `joint_fk`.
- `lower_joint_fk(robot, urdf_ref) -> dict[str, tuple[xyz, rpy, axis]]` (L793) — Per-manifest-joint forward kinematics read from the URDF, matched by `child_link`; needed to place the link capsules. Unmatched (synthetic) joints are omitted.
- `lower_robot_from_mjcf(robot, *, n_samples=2000, seed=20260610, margin_m=0.0, manifest_dir=None) -> LoweredCollisionModel` (L834) — MJCF backend for robots with no URDF collision geometry (e.g. bimanual `openarm`): keeps the manifest's hand-authored capsules, lowers joint FK from mujoco, and derives a conservative ACM from a seeded mujoco-FK overlap sweep — the one remaining sampled criterion, since the URDF joint-tree proof doesn't apply here. A manifest `assets.srdf` is unioned into the sweep's disabled set for hand-reviewed rest-pose exemptions the sweep can't prove; hand-editing the generated ACM block is forbidden. `acm_source="mjcf"`.
- module constant `_MOVABLE_JOINT_TYPES: tuple[str, ...]` (L402) — `("revolute", "continuous", "prismatic")`, joint types `lower_joint_fk`/`acm_for_geometry` treat as movable.
- `lower_robot(robot, *, srdf_path=None, acm_only=False, geometry_only=False, manifest_dir=None) -> LoweredCollisionModel` (L1005) — Top-level entry; threads `robot.safety.self_collision_margin_m` into `acm_for_geometry` so the ACM matches the kernel's live threshold. ACM source precedence: explicit `srdf_path` → manifest `assets.srdf` → URDF-only fallback. Raises `ROSConfigError` on zero fitted links, never an empty geometry/ACM the kernel would silently not check.
- `select_lowering(robot, *, manifest_dir=None) -> LoweringSource` (L1159) — Provenance-correct routing: `"srdf"` (SRDF + usable URDF geometry), `"sampling"` (URDF geometry, no SRDF), `"mjcf"` (no usable URDF geometry but an MJCF exists, e.g. `openarm`'s unresolved package mesh refs). Raises `ROSConfigError` when no lowerable asset exists.
- `lower_robot_auto(robot, *, acm_only=False, geometry_only=False, manifest_dir=None) -> LoweredCollisionModel` (L1204) — Single dispatch over `select_lowering` → `lower_robot` or `lower_robot_from_mjcf`; the one entry point the CLI and the regression test both call, so routing can never diverge.
- `LoweringSource` — `Literal["srdf", "sampling", "mjcf"]`; the source `select_lowering` resolves to (matches `LoweredCollisionModel.acm_source`).

### `packages/openral_safety/openral_safety/cumotion_config.py`
_Derive a cuRobo (cuMotion) robot-config from the lowered collision geometry. Pure module; uses `kernel_predicates.bounding_capsule_segment`. Deliberately over-covers the kernel's geometry — cuRobo can only express spheres, and a planner that believes the arm is thinner than it is emits trajectories the kernel then has to E-stop. Looser than the kernel is fine; tighter is not._

- module constant `_ACTUATED_JOINT_TYPES: frozenset[JointType]` (L50) — `{REVOLUTE, PRISMATIC, CONTINUOUS}`, joint types `actuated_joint_names` counts as single-DOF movable.
- `class CuMotionSphere` (L56) — Frozen dataclass: `center` (link-frame `(x, y, z)`), `radius` — one cuRobo collision sphere.
- `capsule_to_spheres(p0, p1, radius, *, count) -> list[CuMotionSphere]` (L68) — Sample `count` spheres evenly along the segment `p0`→`p1` (endpoints inclusive for `count >= 2`; midpoint for `count == 1`).
- `spheres_for_capsule(shape) -> int` (L110) — Sphere count to tile a lowered primitive with centres ≤ one radius apart; `1` for a sphere / zero-length capsule. Also accepts a `BoxShape`.
- `link_collision_spheres(geom, *, count=None) -> list[CuMotionSphere]` (L134) — Lower one `LinkCollisionGeometry` (capsule, sphere, or box) to cuRobo spheres in its link frame, via the containing capsule.
- `sphere_model_geometry(geoms) -> dict[str, LinkCollisionGeometry]` (L156) — Each link as the capsule cuRobo's spheres actually cover (box → containing capsule; capsules/spheres unchanged) — a strictly larger solid than the kernel's box, so it needs its own ACM proof.
- `actuated_joint_names(robot) -> list[str]` (L203) — Single-DOF movable joint names, in manifest order — the cuRobo `cspace.joint_names`.
- module constant `_SPHERE_DP: int = 6` (L212) — decimal places sphere centers/radii are rounded to when rendered into the YAML fragment.
- `render_cumotion_config(robot, model, *, urdf_path=None) -> str` (L215) — Render a cuRobo `robot_cfg` YAML fragment with a generated-provenance header. Spheres come from `robot.collision_geometry` (falling back to `model.collision_geometry` only when the manifest carries none). The ACM is not copied verbatim: given `urdf_path` it is widened with pairs proven always-colliding for the sphere model, so the planner stays looser than the kernel, never tighter. Without `urdf_path` the manifest ACM is used as-is and the header says so.

### `packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py`
_`reasoner_node` lifecycle wrapper. Thin rclpy shell around `openral_reasoner.ReasonerCore`._

- module constants `_FAILURE_SOURCES`, `_PERCEPTION_KINDS` — closed sets: `hal/sensor/rskill/safety/wam/critic` and `motion/objects/ocr/scene_change`. (L266)
- module constant `_PERCEPTION_KINDS: tuple[str, ...]` (L267) — `("motion", "objects", "ocr", "scene_change")`.
- module constant `_QOS_WORLD_STATE` (L209) — `KEEP_LAST=1, RELIABLE, VOLATILE`, the `/openral/world_state_slow` subscription QoS.
- module constant `_QOS_FAILURE` (L215) — `KEEP_LAST=50, RELIABLE, VOLATILE`, the six `/openral/failure/*` subscription QoS.
- module constant `_QOS_PERCEPTION` (L221) — `KEEP_LAST=10, BEST_EFFORT, VOLATILE`, the four perception-topic subscription QoS.
- module constant `_QOS_PROMPT` (L227) — `KEEP_LAST=10, RELIABLE, VOLATILE`, the `/openral/prompt` publisher/subscription QoS.
- module constant `_QOS_MAP` (L245) — `KEEP_LAST=1, RELIABLE, TRANSIENT_LOCAL`, the latched `/map` subscription QoS.
- module constant `_QOS_COMPLETION_CAMERA` (L256) — `KEEP_LAST=1, BEST_EFFORT, VOLATILE`, the VLM completion-camera subscription QoS.
- module constants `_KIND_TIMEOUT`, `_KIND_CONTROLLER`, `_SEVERITY_WARN`, `_SEVERITY_FAIL` — IDL-mirror constants for `openral_msgs/FailureTrigger`, kept inline (not the `failure_bus` helper) so the reasoner emits without pulling the rate-limiter into the dispatch path. (L293)
- module constant `_KIND_CONTROLLER: int = 5` (L294)
- module constant `_SEVERITY_WARN: int = 1` (L295)
- module constant `_SEVERITY_FAIL: int = 2` (L296)
- module constant `_SKILL_FAILURE_KIND_NAMES: dict[int, str]` (L302) — `{_KIND_TIMEOUT: "timeout", _KIND_CONTROLLER: "controller"}`, kind-derived failure-state names for `_emit_skill_failure_event`.
- module constants `_EXECUTE_SKILL_SERVER_PROBE_S`, `_LIFECYCLE_SERVER_PROBE_S` — 100 ms `wait_for_server` / `wait_for_service` probes so an absent server/peer can't block the executor thread. (L310)
- module constant `_LIFECYCLE_SERVER_PROBE_S: float = 0.1` (L311)
- module constant `_MISSION_VERIFY_WINDOW_S: float = 40.0` (L276) — fallback verify window when the active `RewardContract` sets no `frame_window_s`.
- module constant `_DEFAULT_SUCCESS_THRESHOLD: float = 0.8` (L285) — fallback reward-gate `success_threshold` when no `RewardContract` is loaded.
- module constant `_DEFAULT_CHECK_FLOOR: float = 0.5` (L286) — fallback reward-gate `check_floor` when no `RewardContract` is loaded.
- module constant `_FAILURE_TIER_FOR_SOURCE: dict[str, str]` (L319) — trigger taxonomy mapping each `/openral/failure/<source>` to its tier: `safety → "A"`, `hal/sensor/rskill/wam → "B"`, `critic → "C"`. Used to stamp `reasoner.tier` on the OTel span; the preemption threshold itself is decided inline in `_on_failure`.
- (the `hal_mode == "sim"` gate imports the canonical `openral_core.SIM_EXECUTABLE_CONTROL_MODES` — trimmed to the six packer-implemented modes, see the Layer-0 core entry — rather than a module-local frozenset.)
- `def _required_control_modes(manifest: RSkillManifest) -> set[ControlMode]` (L371) — Pure helper for the deploy-path palette gate. Reads `action_contract` by specificity, from `None` (no constraint) down through representation/slots/legacy `dim`.
- `def _action_executable(manifest: RSkillManifest, description: RobotDescription, hal_mode: str) -> bool` (L396) — Pure helper: `True` when every `_required_control_modes(manifest)` is in the executable set for `hal_mode` (sim's fixed set, else the robot's declared modes). Empty required set → `True`.
- `def _resets_search_episode(call) -> bool` (L434) — Pure helper for the find→re-prompt cascade bound. `True` when dispatching `call` ends the active-search episode; `False` for the three read-only search actions, so a search loop's own budget can never be reset by its own query and fail to terminate.
- `class ReasonerNode(LifecycleNode)` (L605) — Lifecycle node wrapping `ReasonerCore`. Optional injected `spatial_memory` enables the recall/resolve tools; otherwise `spatial_memory_path`/`spatial_memory_ingest` params can load or build one. `memory_md_path` enables the memory tools. `hal_mode` selects the action-mode palette gate; `tick_hz` (default 0.2 Hz) is the heartbeat rate.
  - `_submit_client_warmup(client) -> None` — At `on_configure`, kicks a managed LLM sidecar's boot onto `_llm_pool` instead of leaving it to the first tick, off the executor thread so the lifecycle transition returns promptly. No-op for clients without `warm()`; a failure here only warns, since `select_tool`'s lazy path still owns whether the server is usable.
  - `on_configure(self, state) -> TransitionCallbackReturn` (L1039) — Build `ToolUseClient` from env if not injected, attach subscribers to world-state/failure/perception/prompt/registry topics, create the prompt/failure publishers + `execute_rskill` action client. Reads `vram_lifecycle_peers` (GPU peers auto-deactivated around each `execute_rskill`), loads the reward manifest, and probes GPU total VRAM for the pre-dispatch fit check.
  - `on_activate(self, state) -> TransitionCallbackReturn` (L1333) — Arm the periodic tick timer at `tick_hz`.
  - `on_deactivate(self, state) -> TransitionCallbackReturn` (L1341) — Cancel the tick timer (subscriptions remain attached).
  - `on_cleanup(self, state) -> TransitionCallbackReturn` (L1358) — Tear down pending skill-goal deadline timers, destroy the action client and every cached per-topic emit_prompt publisher, drop cached lifecycle/service clients, and clear the in-flight goal / cancel-reason / tick-trampoline state.
  - `on_shutdown(self, state) -> TransitionCallbackReturn` (L1424) — Final shutdown.
  - `_on_failure(source, msg)` — Append a `FailureEventRecord` to the renderer and preempt the next tick per the tier taxonomy: Tier A (safety) at `severity ≥ WARN`, Tier B/C at `severity ≥ FAIL`. When a `critic` FAIL is a reward wake and an `execute_rskill` goal is in flight, cancels that goal (latched reason `"reward"`) instead of ticking — the reward signal stops the VLA now, not at the `deadline_s` clock — and the canceled result re-enters mission verification.
  - `_on_tick(*, force=False, tier="heartbeat")` — Single-flight trampoline over `_start_tick`: a tick requested mid-dispatch is coalesced (`force` wins) and replayed after, bounded by `_MAX_TICK_REPLAYS=4`. Runs the blocking LLM call on a single-worker pool so the rclpy executor is never starved, then marshals dispatch back onto the executor. A round-trip landing after deactivate/cleanup is dropped by a generation check.
  - **Mission lifecycle.** `_on_prompt` seeds a `MissionState` from the operator goal, gated by `should_rebuild_mission` (cascades never rebuild; an in-progress mission needs explicit `new_goal` metadata). `_dispatch_execute_rskill` records the attempt and the accepted goal handle a reward wake can cancel, with `deadline_s` set to the resolved patience ceiling. On skill return, `_maybe_verify_active_mission_task` issues a windowed `query_task_progress` and `_on_mission_verify_response` applies the three-tier verdict (auto-pass / vlm_check / ladder), gated on the progress head; an inconclusive `vlm_check` falls back into the retry ladder rather than retrying forever. `_emit_mission_complete` summarizes a finished queue.
  - **VLM completion adjudication.** `completion_camera_topic` param (default `"/openral/cameras/top/image"`, empty = disabled) subscribes and caches each frame as JPEG. `_adjudicate_completion(task_text) -> bool | None` asks the tool-use client's `describe_image`; `None` (no frame/client/provider error) degrades to the replanning ladder rather than a false verdict. `_complete_active_and_advance` is the shared advance path for both the native-complete and VLM-confirmed branches.
  - **Task subdivision on replan.** `_should_offer_subdivision(active, offered, max_depth) -> bool` decides whether an abandoned task gets one decomposition offer, bounded by a one-offer-per-task set and the max depth, so a task that declines still terminates in human-handoff. `_dispatch_decompose_mission` flat-splices finer children into a target task, or replaces the whole queue only when it hasn't started yet. No actuation.
  - **Per-task locate budget.** `_charge_task_locate_budget(call, *, traceparent) -> bool` (top of `_dispatch_locate_in_view`) charges one locate cycle against the active task; once `DEFAULT_MAX_TASK_LOCATE_ATTEMPTS` is exhausted without an `execute_rskill` dispatch it abandons the subtask with a specific reason and forces a Tier-C tick, without dispatching the locate. Fixes a live locate-loop the `SearchProgress` miss budget can't bound (that budget resets on a hit). Cleared on a new operator goal and on a real `execute_rskill` dispatch.
  - `_on_skill_registry_changed(msg)` — Palette refresh: walks installed rSkills, loads each manifest, rebuilds the palette against the active capabilities + commercial flag (preserving every availability flag, including `memory_available`), installs it via `set_palette`.
  - `_dispatch(call, *, traceparent=None)` — Routing-only switch over the `ReasonerToolCall` variants; delegates to the per-tool `_dispatch_*` methods. `WaitTool` is a deliberate no-op; `ReloadGstPipelineTool` is a log-and-acknowledge stub pending its service IDL.
  - `_dispatch_emit_prompt(call, *, traceparent)` — Publish a `PromptStamped` on `call.target_topic` (per-topic publisher cache; `/openral/prompt` reuses the standing cascade publisher); stamps the threaded-through `traceparent` into `metadata_json`.
  - `_dispatch_spatial_query(call, *, traceparent)` — Read-only: runs a `RecallObjectTool`/`ResolvePlaceTool` against the injected `SpatialMemory` and republishes the result as a cascade prompt. Bounded by a `SearchProgress`/`SearchBudget`; exhausting it abandons the active mission task so later heartbeats cannot restart a terminal search. A `recall_object` miss escalates policy-driven to a live `locate_in_view` for the same term when a detector is available and the term hasn't already been escalated this streak. With a latched `/map`, every approach viewpoint is refined through the occupancy grid before rendering (BLOCKED when none exists).
  - `_maybe_load_spatial_memory()` — Deployment wiring: on `on_configure`, when no backend was injected and `spatial_memory_path` is set, loads the persisted scene graph and flips `spatial_memory_available`. Load failure degrades to a WARNING + no backend, never a fabricated map.
  - `_maybe_load_memory()` — Deployment wiring: on `on_configure`, when `memory_md_path` is set, parses `MEMORY.md`, renders the `## MEMORY` block, loads the archive log, and flips `memory_available`. Read failure degrades to a WARNING + empty store.
  - `_dispatch_memory_write(call, *, traceparent)` — The reasoner's write-capable dispatch: applies the `MemoryWriteTool` op to the live `MemoryStore`, archives any displaced entry, persists `MEMORY.md`, re-renders `## MEMORY`, and re-prompts a short confirmation. Advisory — a persist failure logs, never raises.
  - `_dispatch_memory_search(call, *, traceparent)` — Read-only: `MemoryStore.search` over the archive (superseded/deleted entries; current memory is already in `## MEMORY`), re-prompted with the hits. No actuation, no file write.
  - `_emit_scene_objects_span()` — Dashboard telemetry: when a spatial-memory backend is wired, publishes the `world.scene_objects` span for the scene-objects card + SLAM-map overlay. Advisory only; failures swallowed at DEBUG.
  - `is_collective_target(text) -> bool` — Imported from `openral_core` (single source of truth, shared with the `GroundedSubtask` schema validator); true when a task text targets a set. Drives the execute grounding gate.
  - `_emit_enumeration_invite(task, *, traceparent)` — Grounding gate's self-prompt: tells the LLM the active task targets a collective set and to `decompose_mission` into one concrete subtask per object before any actuation.
  - `_dispatch_execute_rskill(call, *, traceparent)` — Probes the action server (absence emits a `FailureTrigger`). Refuses a collective-target task without recording an attempt (grounding gate), and a second goal while one is in flight (busy gate, watchdog-bounded so a dead runner can't wedge the latch forever). A two-tier VRAM/reward-fit check refuses before dispatch rather than risking a slow CUDA OOM abort. Routes GPU-peer eviction through `_free_vram_peers_then_send` when configured.
  - `_send_execute_rskill_goal(call, generation, traceparent)` — Build `ExecuteRskill.Goal`, send asynchronously with a feedback callback, attach the generation-bound goal-response handler.
  - `_free_vram_peers_then_send(call, peers, generation, traceparent)` — Deactivate each GPU lifecycle peer and send the goal only once every deactivation response returns, so a peer's VRAM is released before the runner loads the policy. Peers whose service is absent are skipped; a late deactivation from an invalidated generation is immediately reactivated instead of sending a stale goal.
  - `_reactivate_vram_peers()` — Reactivate the peers deactivated for the last dispatch (idempotent). Called on terminal result and on goal reject/error, not on a deadline (the policy may still be resident).
  - `_on_reactivate_result(peer, future)` — Best-effort log of a reactivation `change_state` outcome.
  - `_change_state_async(node, transition) -> future | None` — Shared helper: lazily cache a `ChangeState` client per peer node and call it asynchronously; `None` if the service isn't on the graph. Used by both lifecycle dispatch and VRAM eviction.
  - `_dispatch_lifecycle_transition(call)` — Drive `<call.node>/change_state` via `_change_state_async`; on success attach `_on_lifecycle_response`, on an absent service log + skip.
  - module constant `_FEEDBACK_LOG_PERIOD_S: float = 1.0` (L431) — WARNING-log throttle period for per-chunk action feedback.
  - `_on_execute_skill_feedback(rskill_id, feedback_msg)` — Forward action feedback to the operator log, throttled to one WARNING per `_FEEDBACK_LOG_PERIOD_S` (a VLA goal streams one feedback per chunk); suppressed lines go to DEBUG.
  - `_on_execute_skill_goal_response(call, generation, sent_at, future, traceparent)` — Ignore/cancel stale generations; on rejection emit a `FailureTrigger`; on acceptance arm the deadline timer and attach the result handler.
  - `_on_execute_skill_result(call, generation, goal_id, future, traceparent)` — Ignore stale generations, cancel the deadline timer; log success or emit a `FailureTrigger` with `ControllerEvidence` on abort/cancel/failure. The replanning ladder classifies on the typed `failure_kind`, not the prose: a config/capability-mismatch result drops that rSkill from the palette until rebuilt; a deadline-miss drives a "timed out" reflection; controller/runtime failures remain retryable.
  - module constant `_LEGACY_FAILURE_REASON_PREFIXES: tuple[tuple[str, str], ...]` (L478) — deprecated-in-place prose→`FAILURE_*` prefix table, consulted only when a failed result carries `FAILURE_NONE` (a pre-`failure_kind` producer).
  - `_rskill_failure_kind(result) -> int` — Typed failure kind for a terminal result; the runner-stamped uint8 is authoritative, falling back to the deprecated prose-prefix table only for a producer that predates the field.
  - `_failure_kind_value(name) -> int` — Resolve an `ExecuteRskill.Result` `FAILURE_*` constant by name off the generated IDL rather than re-declaring the values, so the reasoner cannot drift from `packages/msgs`.
  - module constant `_PERMANENT_FAILURE_KIND_NAMES: frozenset[str]` (L498) — `{"FAILURE_CONFIG_ERROR", "FAILURE_CAPABILITY_MISMATCH"}`.
  - module constant `_TIMED_OUT_FAILURE_KIND_NAMES: frozenset[str]` (L504) — `{"FAILURE_DEADLINE_MISSED"}`.
  - `_palette_after_rskill_failure(palette, rskill_id, failure_kind) -> ToolPalette` — Drop a skill after a typed, session-persistent availability failure, classified on `failure_kind` (never on prose); `FAILURE_NONE` is "unclassified", never "permanently broken".
  - `_on_execute_skill_deadline(*, call, generation, sent_at, goal_handle, traceparent)` — Ignore stale generations; latch a `"patience"` cancel reason (a patience-expired attempt still runs the reward verify gate), cancel the goal, and emit a `KIND_TIMEOUT` `FailureTrigger`.
  - **Ladder persistence.** `_persist_ladder_state()` snapshots `ReasonerLadderState` to the `ladder_state_path` param after every ledger mutation; `_maybe_restore_ladder_state()` reloads it at `on_configure`, so a restart resumes the ladder instead of resetting every cap mid-mission.
  - `_on_lifecycle_response(call, future)` — Log the `ChangeState` result; lifecycle failures surface in the target node's own logs (no `FailureTrigger` re-emission).
  - `_publish_skill_failure(*, kind, rskill_id, evidence, traceparent, trace_id=None)` — Build + publish a `FailureTrigger` on `/openral/failure/rskill`, then mirror it onto the OTLP span path via `_emit_skill_failure_event` so the OTLP-only dashboard can tally it.
  - `_emit_skill_failure_event(*, kind, rskill_id, evidence)` — Stamp an `openral.event.skill_failure` span event carrying the failure state, on the active tick span when recording or a transient span otherwise. Drives the dashboard "skill failures" counter.
  - `renderer(self) -> ContextRenderer` (L5093) — Direct read access for tests asserting buffer state.
  - `dispatched_calls(self) -> tuple[Any, ...]` (L5098) — Snapshot of tool calls the reasoner has dispatched (in order).
  - `set_palette(self, palette) -> None` (L5102) — Replace the active palette (rebuilt on `/openral/skill_registry_changed`).
- `_QOS_REGISTRY_CHANGED` (L236) — RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1 so a late-subscribing reasoner sees the most recent invalidation.
- `main(args=None) -> int` (L5107) — Entry point for `ros2 run openral_reasoner_ros reasoner_node`.

### `packages/openral_prompt_router/openral_prompt_router/prompt_router_node.py`
_Single lifecycle node that fans in operator prompts from any external source into `/openral/prompt`. CLI is the only v1 adapter; WebSocket / voice / Slack are out of scope._

- module constant `DEFAULT_SOURCES: dict[str, int] = {"cli": 100, "dashboard": 100, "auto": 10}` — Default source → priority registry; human sources get 100, machine cascades get 10. (L76)
- module constant `_QOS_PROMPT` (L60) — `KEEP_LAST=10, RELIABLE, VOLATILE`, the fan-out publisher / per-source subscriber QoS.
- module constant `_STARTUP_PROMPT_SUBSCRIBER_TIMEOUT_S: float = 30.0` (L71) — how long `on_activate` waits for a subscriber before publishing the configured startup prompt anyway.
- `class PromptRouterNode(LifecycleNode)` (L83) — Lifecycle node.
  - `__init__(*, node_name="openral_prompt_router", sources=None)` — Initialise with a source → priority registry. Defaults to `DEFAULT_SOURCES`.
  - `on_configure(self, state) -> TransitionCallbackReturn` (L119) — Build the `/openral/prompt` fan-out publisher and one `/openral/prompt_in/<source>` subscriber per allowed source.
  - `on_activate(self, state) -> TransitionCallbackReturn` (L141) — Publish the startup prompt if configured; then idle (purely reactive).
  - `on_deactivate(self, state) -> TransitionCallbackReturn` (L150) — Stop forwarding (subscriptions remain attached).
  - `on_cleanup(self, state) -> TransitionCallbackReturn` (L156) — Drop the publisher; subscriptions auto-cleaned by rclpy.
  - `_on_inbound(source, priority, msg)` — Forward the inbound PromptStamped onto `/openral/prompt` after merging `{"source": ..., "priority": ...}` into `metadata_json` (preserving any per-source fields).
  - `forwarded_count(self) -> int` (L238) — Number of prompts forwarded since `on_configure` (for tests).
- `main(args=None) -> int` (L243) — Entry point for `ros2 run openral_prompt_router prompt_router_node`.

### `python/cli/src/openral_cli/prompt.py`
_`openral prompt "do X"` CLI adapter. Publishes a one-shot `PromptStamped` onto `/openral/prompt_in/cli` for the prompt-router to fan out. `rclpy` lazy-imported so `openral --help` stays sub-second._

- `prompt_command(text, topic="/openral/prompt_in/cli", wait_s=1.0, discovery_wait_s=5.0, new_goal=False)` (L31) — Initialise rclpy, publish one PromptStamped with `metadata_json={"source_cli": true}` plus `"new_goal": true` under `--new-goal`, wait briefly for the subscriber to be discovered, then shut down. Exits 2 if rclpy / openral_msgs are not importable (with a hint at `just ros2-build`). The prompt-router preserves the mission-replacement flag when stamping `source`/`priority`.

### Observability (Layer 6 — fully shipped)

### `python/observability/src/openral_observability/_sdk.py`
_Idempotent OTel SDK setup + flush helper._

- `configure_observability(*, service_name="openral", endpoint=None, sample_ratio=None) -> bool` — Install OTLP/gRPC tracer + meter + logger providers; reads `OTEL_EXPORTER_OTLP_ENDPOINT` when `endpoint` is None; `True` if exporters were installed, `False` for the no-op path. On success also starts the system-metrics collector and registers `shutdown_observability` via `atexit`. `sample_ratio` (or `OPENRAL_OTEL_SAMPLE_RATIO`) selects the trace sampler: `None`/`1.0` → always-on, else a parent-based ratio sampler. (L115)
- `configure_worker_observability(service_name, *, endpoint=None, sample_ratio=None) -> bool` — Cross-process bootstrap for a spawned worker: calls `configure_observability` then attaches the parent trace context from env, so a child spawned with `env={**os.environ, **traceparent_env()}` joins the parent trace. (L231)
- `_resolve_sampler(sample_ratio) -> Sampler` — Resolve the trace sampler from arg + env, defaulting to always-on; a garbage env value also falls back to always-on rather than silently dropping every span. (L344)
- `shutdown_observability() -> None` — Flush + shut down all three providers; idempotent and safe with no exporter installed. Stops the system-metrics collector before draining the meter so the final sample lands in the export batch. (L401)

### `python/observability/src/openral_observability/tracing.py`
_Span-context-manager helpers; safe to call before `configure_observability`._

- `rskill_span(name, *, rskill_id=None, role=None, **attrs)` — Span for a Skill lifecycle phase; emits `rskill.id` / `rskill.role` from `semconv`. (L61)
- `inference_span(name="skill.chunk_inference", *, chunk_index=None, kind: InferenceKind="foreground", **attrs)` — Span for one VLA inference plus the `openral.inference.duration` histogram (emitted together so they cannot diverge). `InferenceKind = Literal["foreground", "prefetch", "single"]` is the closed timing-axis label set. (L91)
- `safety_span(name="safety.check", *, check_name=None, severity="info", **attrs)` — Span for a safety check; the C++ kernel parents its own `safety.check` to the Python tick via the propagator. (L145)
- `reasoner_span(name="reasoner.tick", *, tick_idx=None, model=None, force=None, **attrs)` — Span for one `ReasonerCore.tick`. Sets `reasoner.{tick.idx, model, force}` and accepts any extra `reasoner.*` attribute via `**attrs`. (L179)
- `start_reasoner_span(name="reasoner.tick", *, tick_idx=None, model=None, force=None, **attrs) -> Span` — Non-attaching variant for the phased tick: returns a started span without attaching it, so `prepare_tick` → off-thread `run_prepared_llm` → `finish_tick` can carry it between phases (each re-attaching via `use_span`) without an intermediate executor callback seeing it as current. Caller owns `span.end()`. (L250)
- module constant `P` (L52) — `ParamSpec("P")`, generic parameter type var for `traced`'s decorator signature.
- module constant `R` (L53) — `TypeVar("R")`, generic return type var for `traced`'s decorator signature.
- `traced(name=None)` — Decorator that wraps a sync function in a span named after it. (L278)

### `python/observability/src/openral_observability/cli.py`
_Root-span helper for the ``openral`` CLI._

- `cli_command_span(subcommand, *, mode=None, run_id=None, **attrs)` — Open the `cli.command` root span for one CLI invocation; records `cli.subcommand`, `openral.run.id`, optional `openral.run.mode` / `openral.run.git_sha`. (L51)

### `python/observability/src/openral_observability/diagnostics.py`
_`diagnostic_msgs/DiagnosticArray` heartbeat helper, shared by every OpenRAL lifecycle node._

- `Level` — Mirror of `diagnostic_msgs/DiagnosticStatus` level constants (`OK=0`, `WARN=1`, `ERROR=2`, `STALE=3`); re-exported so `status_fn` callbacks can avoid importing `diagnostic_msgs` on pure-Python hosts. (L31)
- `class DiagnosticsHeartbeat` (L48) — 1 Hz `/diagnostics` publisher attached to a `rclpy.lifecycle.LifecycleNode`. `__init__(node, *, hardware_id, component_name, status_fn, rate_hz=1.0)`. An exception inside `status_fn` is converted to a synthetic ERROR-level diagnostic so the timer never crashes the node.
  - `hardware_id(self) -> str` (L112) — Return the configured `hardware_id` field.
  - `component_name(self) -> str` (L117) — Return the configured `DiagnosticStatus.name` value.
  - `create_publisher(self) -> None` (L121) — Open the `/diagnostics` publisher. Call from `on_configure`.
  - `start(self) -> None` (L138) — Start the 1 Hz publish timer. Call from `on_activate`.
  - `stop(self) -> None` (L155) — Cancel the publish timer. Call from `on_deactivate`.
  - `destroy(self) -> None` (L167) — Destroy the publisher. Call from `on_cleanup` / `on_shutdown`.
  - `publish_once(self) -> None` (L179) — Publish one `DiagnosticArray` immediately.

### `python/observability/src/openral_observability/lifecycle.py`
_Make `LifecycleNode` transition-callback failures observable — rclpy silently converts an uncaught callback exception to `TransitionCallbackReturn.ERROR` without logging, so a composing host sees only `exit code 4`._

- `log_lifecycle_errors(callback) -> callback` — Decorator for `on_configure`/`on_activate`/… transition callbacks. Transparent on success; on an uncaught exception logs the callback name + traceback and returns `TransitionCallbackReturn.FAILURE` instead of letting rclpy swallow it silently. Applied to `RskillRunnerNode`, `_WorldStateLifecycleNode`, `HALLifecycleNodeBase`, and `ReasonerNode`. Lazy-imports `rclpy` so the module stays import-safe on pure-Python hosts. (L44)

### `python/observability/src/openral_observability/semconv.py`
_Single source of truth for OpenRAL OTel attribute / span / metric names. `Final[str]` constants grouped by wire namespace exactly as the module's own `# ──` section banners divide them; each constant's value is its own description._

**rskill.* — short-prefix span/metric attributes**
- `const RSKILL_ID: str = "rskill.id"` (L30)
- `const RSKILL_ROLE: str = "rskill.role"` (L31)
- `const RSKILL_TICK_MS: str = "rskill.tick_ms"` (L33)
- `const RSKILL_INFERENCE_MS: str = "rskill.inference_ms"` (L34)
- `const RSKILL_SENSORS_MS: str = "rskill.sensors_ms"` (L35)
- `const RSKILL_WORLD_STATE_MS: str = "rskill.world_state_ms"` (L36)
- `const RSKILL_SAFETY_MS: str = "rskill.safety_ms"` (L37)
- `const RSKILL_HAL_MS: str = "rskill.hal_ms"` (L38)
- `const RSKILL_ACTION_APPLIED: str = "rskill.action_applied"` (L39)
- `const RSKILL_SAFETY_VIOLATIONS: str = "rskill.safety_violations"` (L40)
- `const RSKILL_EPISODE_IDX: str = "rskill.episode_idx"` (L41)
- `const RSKILL_STEP_IDX: str = "rskill.step_idx"` (L42)
- `const RSKILL_REWARD: str = "rskill.reward"` (L43)
- `const RSKILL_TERMINATED: str = "rskill.terminated"` (L44)
- `const RSKILL_TRUNCATED: str = "rskill.truncated"` (L45)
- `const INFERENCE_KIND: str = "inference.kind"` (L47)
- `const INFERENCE_CHUNK_INDEX: str = "inference.chunk_index"` (L48)
- `const INFERENCE_CHUNK_SIZE: str = "inference.chunk_size"` (L49)
- `const INFERENCE_ENGINE: str = "inference.engine"` (L50)
- `const INFERENCE_DEVICE: str = "inference.device"` (L51)
- `const INFERENCE_DURATION_MS: str = "inference.duration_ms"` (L52)
- `const SAFETY_SEVERITY: str = "safety.severity"` (L54)
- `const SAFETY_CHECK_NAME: str = "safety.check_name"` (L55)
- `const SAFETY_KERNEL: str = "safety.kernel"` (L56)

**reward.* — task-progress reward monitor scores**
- `const REWARD_PROGRESS: str = "reward.progress"` (L59)
- `const REWARD_SUCCESS: str = "reward.success"` (L60)
- `const REWARD_STALLED: str = "reward.stalled"` (L61)
- `const REWARD_SUCCEEDED: str = "reward.succeeded"` (L62)
- `const REWARD_FRAMES: str = "reward.frames"` (L63)
- `const REWARD_TASK: str = "reward.task"` (L64)
- `const REWARD_CAMERA: str = "reward.camera"` (L65)

**openral.run.* — CLI invocation**
- `const RUN_ID: str = "openral.run.id"` (L69)
- `const RUN_MODE: str = "openral.run.mode"` (L70)
- `const RUN_GIT_SHA: str = "openral.run.git_sha"` (L71)

**openral.tick.* — runner cadence**
- `const TICK_IDX: str = "openral.tick.idx"` (L75)
- `const TICK_RATE_HZ: str = "openral.tick.rate_hz"` (L76)
- `const TICK_DEADLINE_MS: str = "openral.tick.deadline_ms"` (L77)

**openral.rskill.* — rSkill identity (namespaced)**
- `const RSKILL_ID_NS: str = "openral.rskill.id"` (L81)
- `const RSKILL_REVISION_NS: str = "openral.rskill.revision"` (L82)
- `const RSKILL_ROLE_NS: str = "openral.rskill.role"` (L83)
- `const RSKILL_ACTION_HORIZON: str = "openral.rskill.action_horizon"` (L84)

**openral.hal.***
- `const HAL_ADAPTER: str = "openral.hal.adapter"` (L88)
- `const HAL_ROBOT_MODEL: str = "openral.hal.robot.model"` (L89)
- `const HAL_CONTROL_MODE: str = "openral.hal.control_mode"` (L90)
- `const HAL_JOINT_NAMES: str = "openral.hal.joint.names"` (L95)
- `const HAL_JOINT_POSITIONS: str = "openral.hal.joint.positions"` (L96)
- `const HAL_JOINT_VELOCITIES: str = "openral.hal.joint.velocities"` (L97)
- `const HAL_JOINT_EFFORTS: str = "openral.hal.joint.efforts"` (L98)
- `const HAL_JOINT_POSITION_LIMITS_LO: str = "openral.hal.joint.position_limits_lo"` (L99)
- `const HAL_JOINT_POSITION_LIMITS_HI: str = "openral.hal.joint.position_limits_hi"` (L100)
- `const HAL_JOINT_VELOCITY_LIMITS: str = "openral.hal.joint.velocity_limits"` (L101)
- `const HAL_JOINT_EFFORT_LIMITS: str = "openral.hal.joint.effort_limits"` (L102)
- `const HAL_JOINT_STAMP_NS: str = "openral.hal.joint.stamp_ns"` (L103)
- `const HAL_ACTION_NEXT: str = "openral.hal.action.next"` (L110)
- `const HAL_ACTION_DIM: str = "openral.hal.action.dim"` (L111)
- `const HAL_ACTION_HORIZON: str = "openral.hal.action.horizon"` (L112)
- `const HAL_ACTION_APPLIED: str = "openral.hal.action.applied"` (L113)
- `const HAL_EE_NAMES: str = "openral.hal.ee.names"` (L117)
- `const HAL_EE_POSE_PREFIX: str = "openral.hal.ee.pose"` (L118)
- `const HAL_GRIPPER_POSITION: str = "openral.hal.gripper.position"` (L119)
- `const HAL_GRIPPER_FORCE_N: str = "openral.hal.gripper.force_n"` (L120)

**openral.sensors.***
- `const SENSORS_MODALITY: str = "openral.sensors.modality"` (L124)
- `const SENSORS_SOURCE: str = "openral.sensors.source"` (L125)
- `const SENSORS_AGE_MS: str = "openral.sensors.age_ms"` (L126)
- `const SENSORS_WIDTH: str = "openral.sensors.width"` (L127)
- `const SENSORS_HEIGHT: str = "openral.sensors.height"` (L128)
- `const SENSORS_CHANNELS: str = "openral.sensors.channels"` (L129)
- `const SENSORS_ENCODING: str = "openral.sensors.encoding"` (L130)
- `const SENSORS_THUMBNAIL_JPEG_B64: str = "openral.sensors.thumbnail_jpeg_b64"` (L135)

**openral.system.* — host / GPU / CPU health**
- `const SYSTEM_GPU_INDEX: str = "openral.system.gpu.index"` (L139)
- `const SYSTEM_GPU_NAME: str = "openral.system.gpu.name"` (L140)

**openral.world_state.***
- `const WORLD_STATE_STALENESS_MS: str = "openral.world_state.staleness_ms"` (L144)
- `const WORLD_STATE_COMPONENTS_STALE: str = "openral.world_state.components_stale"` (L145)
- `const WORLD_STATE_HAS_LATCHED_ERROR: str = "openral.world_state.has_latched_error"` (L146)
- `const WORLD_STATE_COMPONENT: str = "openral.world_state.component"` (L147)

**openral.world_state.scene_objects.* (durable spatial-memory scene-object graph)**
- `const WORLD_SCENE_OBJECTS_LIST: str = "openral.world_state.scene_objects.list"` (L153)
- `const WORLD_SCENE_OBJECTS_COUNT: str = "openral.world_state.scene_objects.count"` (L154)
- `const WORLD_SCENE_OBJECTS_FRAME: str = "openral.world_state.scene_objects.frame_id"` (L155)
- `const WORLD_SCENE_OBJECTS_SOURCE_NODE: str = "openral.world_state.scene_objects.source_node"` (L156)

**openral.dataset.***
- `const DATASET_REPO_ID: str = "openral.dataset.repo_id"` (L160)
- `const DATASET_EPISODE_IDX: str = "openral.dataset.episode_idx"` (L161)
- `const DATASET_FRAME_IDX: str = "openral.dataset.frame_idx"` (L162)
- `const DATASET_EPISODE_SUCCESS: str = "openral.dataset.episode.success"` (L166)

**Span names**
- `const SPAN_CLI_COMMAND: str = "cli.command"` (L170)
- `const SPAN_RSKILL_TICK: str = "rskill.tick"` (L171)
- `const SPAN_RSKILL_CONFIGURE: str = "rskill.configure"` (L172)
- `const SPAN_RSKILL_ACTIVATE: str = "rskill.activate"` (L173)
- `const SPAN_RSKILL_EXECUTE: str = "rskill.execute"` (L174)
- `const SPAN_RSKILL_CHUNK_INFERENCE: str = "rskill.chunk_inference"` (L175)
- `const SPAN_HAL_READ_STATE: str = "hal.read_state"` (L176)
- `const SPAN_HAL_SEND_ACTION: str = "hal.send_action"` (L177)
- `const SPAN_SENSORS_READ_LATEST: str = "sensors.read_latest"` (L178)
- `const SPAN_WORLD_STATE_SNAPSHOT: str = "world_state.snapshot"` (L179)
- `const SPAN_WORLD_SCENE_OBJECTS: str = "world.scene_objects"` (L180)
- `const SPAN_SAFETY_CHECK: str = "safety.check"` (L181)
- `const SPAN_REWARD_SCORE: str = "reward.score"` (L182)
- `const SPAN_REASONER_TICK: str = "reasoner.tick"` (L184)
- `const SPAN_DEPLOY_BRINGUP: str = "deploy.bringup"` (L186)
- `const BRINGUP_NODE: str = "openral.bringup.node"` (L198)
- `const BRINGUP_TRANSITION: str = "openral.bringup.transition"` (L199)
- `const BRINGUP_RESULT: str = "openral.bringup.result"` (L200)
- `const REASONER_MODEL: str = "reasoner.model"` (L203)
- `const REASONER_TICK_IDX: str = "reasoner.tick.idx"` (L204)
- `const REASONER_TOOL: str = "reasoner.tool"` (L205)
- `const REASONER_RSKILL_ID: str = "reasoner.rskill_id"` (L206)
- `const REASONER_SUPPRESSED_REASON: str = "reasoner.suppressed_reason"` (L207)
- `const REASONER_ERROR_KIND: str = "reasoner.error_kind"` (L208)
- `const REASONER_FORCE: str = "reasoner.force"` (L209)
- `const REASONER_LLM_S: str = "reasoner.llm_s"` (L214)
- `const REASONER_PROMPT_TOKENS: str = "reasoner.prompt_tokens"` (L216)
- `const REASONER_MISSION_JSON: str = "reasoner.mission_json"` (L222)
- `const SKILL_FAILURE_STATE: str = "openral.event.skill_failure.state"` (L227)
- `const REASONER_TIER: str = "reasoner.tier"` (L232)
- `const SPAN_SIM_RUN: str = "sim.run"` (L234)
- `const SPAN_SIM_STEP: str = "sim.step"` (L235)
- `const SPAN_PHYSICS_STEP: str = "physics.step"` (L236)

**Span-event names**
- `const EVENT_ESTOP_REQUESTED: str = "openral.event.estop_requested"` (L254)
- `const EVENT_SENSOR_STALE: str = "openral.event.sensor_stale"` (L255)
- `const EVENT_STALENESS_LATCHED: str = "openral.event.staleness_latched"` (L256)
- `const EVENT_ERROR_LATCHED: str = "openral.event.error_latched"` (L257)
- `const EVENT_SAFETY_VIOLATION: str = "openral.event.safety_violation"` (L258)
- `const EVENT_ACTION_DROPPED: str = "openral.event.action_dropped"` (L261)
- `const EVENT_DEADLINE_MISSED: str = "openral.event.deadline_missed"` (L262)
- `const EVENT_SKILL_FAILURE: str = "openral.event.skill_failure"` (L269)
- `const EVENT_CHUNK_PREFETCH_HIT: str = "openral.event.chunk_prefetch_hit"` (L273)
- `const EVENT_CHUNK_PREFETCH_MISS: str = "openral.event.chunk_prefetch_miss"` (L274)
- `const EVENT_EPISODE_CLOSED: str = "openral.event.episode_closed"` (L279)

**Metric instrument names**
- `const METRIC_TICK_DURATION: str = "openral.tick.duration"` (L283)
- `const METRIC_INFERENCE_DURATION: str = "openral.inference.duration"` (L284)
- `const METRIC_TICK_BUDGET_VIOLATIONS: str = "openral.tick.budget_violations"` (L285)
- `const METRIC_TICK_DEADLINE_MISSES: str = "openral.tick.deadline_misses"` (L286)
- `const METRIC_SAFETY_VIOLATIONS: str = "openral.safety.violations"` (L287)
- `const METRIC_HAL_READ_STATE_DURATION: str = "openral.hal.read_state.duration"` (L288)
- `const METRIC_HAL_SEND_ACTION_DURATION: str = "openral.hal.send_action.duration"` (L289)
- `const METRIC_HAL_ESTOP_COUNT: str = "openral.hal.estop.count"` (L290)
- `const METRIC_SENSORS_AGE_MS: str = "openral.sensors.age_ms"` (L291)
- `const METRIC_SENSORS_STALE_READS: str = "openral.sensors.stale_reads"` (L292)
- `const METRIC_WORLD_STATE_STALENESS_MS: str = "openral.world_state.staleness_ms"` (L293)
- `const METRIC_WORLD_STATE_COMPONENTS_STALE: str = "openral.world_state.components_stale"` (L294)
- `const METRIC_OBSERVABILITY_EXPORT_FAILURES: str = "openral.observability.export_failures"` (L295)
- `const METRIC_SIM_EPISODE_COUNT: str = "openral.sim.episode.count"` (L296)
- `const METRIC_SYSTEM_GPU_MEMORY_USED_MB: str = "openral.system.gpu.memory_used_mb"` (L300)
- `const METRIC_SYSTEM_GPU_MEMORY_TOTAL_MB: str = "openral.system.gpu.memory_total_mb"` (L301)
- `const METRIC_SYSTEM_GPU_UTIL_PCT: str = "openral.system.gpu.utilization_pct"` (L302)
- `const METRIC_SYSTEM_CPU_UTIL_PCT: str = "openral.system.cpu.utilization_pct"` (L303)
- `const METRIC_SYSTEM_RAM_USED_MB: str = "openral.system.ram.used_mb"` (L304)
- `const METRIC_SYSTEM_RAM_TOTAL_MB: str = "openral.system.ram.total_mb"` (L305)
- `const METRIC_SIM_EPISODE_SUCCESS: str = "openral.sim.episode.success"` (L306)

**Closed-set label vocabularies (for metric labels)**
- `const LABEL_RSKILL_ID: str = "rskill.id"` (L313)
- `const LABEL_RSKILL_REVISION: str = "rskill.revision"` (L314)
- `const LABEL_HAL_ADAPTER: str = "hal.adapter"` (L315)
- `const LABEL_ROBOT_MODEL: str = "robot.model"` (L316)
- `const LABEL_CONTROL_MODE: str = "control_mode"` (L317)
- `const LABEL_ENGINE: str = "engine"` (L318)
- `const LABEL_DEVICE: str = "device"` (L319)
- `const LABEL_KIND: str = "kind"` (L320)
- `const LABEL_MODALITY: str = "modality"` (L321)
- `const LABEL_CHECK_NAME: str = "check_name"` (L322)
- `const LABEL_SEVERITY: str = "severity"` (L323)
- `const LABEL_SIGNAL_KIND: str = "signal_kind"` (L324)
- `const LABEL_COMPONENT: str = "component"` (L325)
- `const LABEL_REASON: str = "reason"` (L326)
- `const METRIC_THRESHOLD_MS: str = "openral.metric.threshold_ms"` (L332)
- `const METRIC_THRESHOLD_DIR: str = "openral.metric.threshold_dir"` (L338)
- `const THRESHOLD_DIR_UPPER: str = "upper"` (L339)
- `const THRESHOLD_DIR_LOWER: str = "lower"` (L340)

**Run-mode enum (closed set for openral.run.mode)**
- `const RUN_MODE_SIM: str = "sim"` (L344)
- `const RUN_MODE_HARDWARE: str = "hardware"` (L345)
- `const RUN_MODE_BENCHMARK: str = "benchmark"` (L346)

**Safety kernel enum (closed set for safety.kernel — see SAFETY_KERNEL)**
- `const SAFETY_KERNEL_NULL: str = "null"` (L350)
- `const SAFETY_KERNEL_CPP: str = "cpp"` (L351)


### `python/observability/src/openral_observability/metrics.py`
_Cached OTel meter instruments — safe to call before `configure_observability`._

- `get_meter() -> Meter` — Resolve the OpenRAL meter against the current `MeterProvider`. (L64)
- `get_tick_duration() -> Histogram` — `openral.tick.duration`, unit `ms`. (L99)
- `get_inference_duration() -> Histogram` — `openral.inference.duration`, unit `ms`. (L114)
- `get_hal_read_state_duration() -> Histogram` — `openral.hal.read_state.duration`, unit `ms`. (L126)
- `get_hal_send_action_duration() -> Histogram` — `openral.hal.send_action.duration`, unit `ms`. (L138)
- `get_sensors_age_ms() -> Histogram` — `openral.sensors.age_ms`, unit `ms`. (L150)
- `get_world_state_staleness_ms() -> Histogram` — `openral.world_state.staleness_ms`, unit `ms`. (L162)
- `get_tick_budget_violations() -> Counter` — `openral.tick.budget_violations`. (L177)
- `get_tick_deadline_misses() -> Counter` — `openral.tick.deadline_misses`. (L188)
- `get_safety_violations() -> Counter` — `openral.safety.violations`, labels `check_name` / `severity`. (L199)
- `get_hal_estop_count() -> Counter` — `openral.hal.estop.count`. (L213)
- `get_sensors_stale_reads() -> Counter` — `openral.sensors.stale_reads`. (L234)
- `get_sim_episode_count() -> Counter` — `openral.sim.episode.count`; sim episodes that ran to completion (terminated or truncated). (L245)
- `get_sim_episode_success() -> Counter` — `openral.sim.episode.success`; sim episodes that hit `task.success_key` at least once. (L256)
- `get_observability_export_failures() -> Counter` — `openral.observability.export_failures`, label `signal_kind`. (L267)
- `get_world_state_components_stale() -> UpDownCounter` — `openral.world_state.components_stale`. (L292)
- `get_system_gpu_memory_used_mb() -> UpDownCounter` — `openral.system.gpu.memory_used_mb`, unit `MBy`. (L306)
- `get_system_gpu_memory_total_mb() -> UpDownCounter` — `openral.system.gpu.memory_total_mb`, unit `MBy`. (L318)
- `get_system_gpu_util_pct() -> UpDownCounter` — `openral.system.gpu.utilization_pct`, unit `%`. (L330)
- `get_system_cpu_util_pct() -> UpDownCounter` — `openral.system.cpu.utilization_pct`, unit `%`. (L342)
- `get_system_ram_used_mb() -> UpDownCounter` — `openral.system.ram.used_mb`, unit `MBy`. (L354)
- `get_system_ram_total_mb() -> UpDownCounter` — `openral.system.ram.total_mb`, unit `MBy`. (L366)
- `record_histogram_ms(instrument, value_ms, attributes=None) -> None` — Record a millisecond value, skipping negatives and `NaN`. (L381)

### `python/observability/src/openral_observability/producer.py`
_Producer-side helpers for recording rich span attributes on OpenRAL hot-path spans. Safe to call on no-op spans; lists are truncated to `_MAX_JOINTS` / `_MAX_EE_FRAMES` and floats rounded to 3 decimals._

- `record_joint_state(span, *, names, positions, velocities=None, efforts=None, position_limits=None, velocity_limits=None, effort_limits=None, stamp_ns=None) -> None` — Attach per-joint attributes to a `hal.read_state` span. (L96)
- `record_action(span, *, next_row, dim=None, horizon=None, applied=None, gripper_position=None, gripper_force_n=None) -> None` — Attach commanded-action attributes to a `hal.send_action` span. (L144)
- `record_ee_poses(span, ee_poses) -> None` — Flatten a `name → Pose6D` mapping onto a `world_state.snapshot` span. (L175)
- `record_sensor_frame_attrs(span, *, modality=None, encoding=None, width=None, height=None, channels=None, age_ms=None, thumbnail_bytes=None, thumbnail_already_encoded_b64=False) -> None` — Attach sensor-frame attributes to a `sensors.read_latest` span. (L201)
- `emit_sensor_frame_span(frame, *, sensor_name, age_ms, flip_180=False, tracer_name=…) -> None` — The shared producer of a dashboard `sensors.read_latest` span for one camera frame: optional flip of a display-only copy (the policy's actual frame is never mutated), then span + attrs + thumbnail. Both frame emitters route through this one function so they can't drift apart. (L245) `encode_frame_thumbnail` — but the span is opened first and the function returns as soon as `span.is_recording()` is false (no OTLP endpoint / dashboard off): the flip copy and the JPEG only feed the span, and at 30 Hz per camera they held ~25 % of the deploy runtime's GIL on an AGX Orin for nothing.
- `encode_rgb_thumbnail(rgb) -> bytes | None` — Encode an HWC uint8 RGB ndarray to a small JPEG for OTLP; returns `None` if Pillow is unavailable. (L308)
- `encode_frame_thumbnail(frame) -> bytes | None` — Encode an `openral_core.SensorFrame` (RGB8/BGR8/MONO8/JPEG/PNG) as a small JPEG thumbnail; returns `None` for non-renderable encodings. (L333)
- `modality_for_encoding(encoding) -> str` — Map a `FrameEncoding` to the dashboard's modality label (`rgb`/`mono`/`depth`/`raw`/`unknown`); shared by `DeployRunner` and the world-state node so both produce identical labels. (L57)
- `_MODALITY_BY_ENCODING: dict[str, str]` (L45) — Canonical encoding → modality lookup table.

### `python/observability/src/openral_observability/system_metrics.py`
_Background sampler for the `openral.system.*` gauges; feeds the dashboard's System Health card via `psutil` (CPU + RAM) and optional `pynvml` (GPU memory + util)._

- `start_system_metrics_collector(*, interval_s=1.0) -> bool` — Start a daemon thread that samples host metrics every `interval_s` seconds. Returns `False` and a quiet no-op when neither `psutil` nor `pynvml` is importable. Idempotent; re-starts retune the interval. (L46)
- `stop_system_metrics_collector(*, timeout_s=2.0) -> None` — Signal the collector thread to stop and join. Safe to call when not running. (L76)
- `_nvml_query(read, what, gpu_index) -> Any | None` — Run one NVML read, returning `None` when the device does not implement it. Each GPU metric is queried independently, so an unsupported call (e.g. no discrete VRAM pool on a unified-memory SoC) cannot also drop the other metrics. Logged once per (device, query). (L146)

### `python/observability/src/openral_observability/propagation.py`
_W3C TraceContext inject / extract for cross-process trace correlation._

- `current_traceparent() -> str | None` — W3C `traceparent` value for the active span, or `None` outside a span. (L54)
- `current_trace_id() -> str | None` — The 32-hex trace id of the active span, or `None` outside a span (including no-op observability mode). The id half of `current_traceparent`, for the fields that store a bare pointer into the trace tree — `RunResult.trace_id`, `RSkillEvalResult.trace_id`. (L69)
- `inject_traceparent(carrier=None) -> dict[str, str]` — Write the active span's `traceparent` (and optional `tracestate`) into a carrier dict; used by producers of `ActionChunk.msg` / `ExecuteRskill.action` / `FailureTrigger.msg`. (L87)
- `extract_traceparent(traceparent, tracestate=None) -> Context` — Parse a wire-side `traceparent` into an OTel `Context` for `context.attach` / `trace.use_span`; consumed by the C++ safety kernel and any Python ROS consumer. (L117)
- `traceparent_env(carrier=None) -> dict[str, str]` — Env-var carrier (`OTEL_TRACEPARENT` + optional `OTEL_TRACESTATE`) for the active span, built from `inject_traceparent`; pass as `env=` to `subprocess` / `multiprocessing` so a worker joins the parent trace. `{}` when no valid span is in scope. (L153)
- `attach_traceparent_from_env(env=None) -> object | None` — Worker-side counterpart: read `OTEL_TRACEPARENT` / `OTEL_TRACESTATE` from `env` (default `os.environ`) and `context.attach` the parent context; returns the detach token, or `None` when absent/empty. (L195)
- `remote_parent_from_env(env=None)` [@contextmanager] — Scope `attach_traceparent_from_env` for a worker `main()`: attaches on enter, detaches on exit; yields the detach token (or `None` when no carrier present). (L238)

### `python/observability/src/openral_observability/failure_bus.py`
_Publisher helper + IDL-mirror constants for the namespaced `/openral/failure/{...}` bus._

- `class FailureSource(str, Enum)` (L113) — `HAL | SENSOR | SKILL | SAFETY | WAM | CRITIC`; the string value is the topic suffix.
- `topic_for(source: FailureSource) -> str` (L128) — Pure helper: `FailureSource → /openral/failure/<suffix>`.
- `KIND_*` / `SEVERITY_*` `int` module constants (L105–L121) — Mirror `openral_msgs/msg/FailureTrigger` (incl. `KIND_COLLISION = 10`, stamped by the safety kernel on every collision stop); bump both when the IDL changes. A test pins the two sides name-for-name and value-for-value against the generated constants.
- `KIND_TIMEOUT: int = 0` (L92)
- `KIND_FORCE: int = 1` (L93)
- `KIND_WORKSPACE: int = 2` (L94)
- `KIND_PERCEPTION: int = 3` (L95)
- `KIND_CRITIC: int = 4` (L96)
- `KIND_CONTROLLER: int = 5` (L97)
- `KIND_SELFVERIFY: int = 6` (L98)
- `KIND_HUMAN: int = 7` (L99)
- `KIND_WAM: int = 8` (L100)
- `KIND_REASONER_TIMEOUT: int = 9` (L101)
- `KIND_COLLISION: int = 10` (L102)
- `KIND_SUPPRESSED_SUMMARY: int = 254` (L103)
- `SEVERITY_INFO: int = 0` (L105)
- `SEVERITY_WARN: int = 1` (L106)
- `SEVERITY_FAIL: int = 2` (L107)
- `SEVERITY_ABORT: int = 3` (L108)
- `TOPIC_PREFIX: str = "/openral/failure"` (L110)
- `DEFAULT_RATE_LIMIT_HZ: dict[int, float | None]` (L146) — Per-severity defaults (INFO/WARN → 10/s, FAIL/ABORT → unlimited).
- `DEFAULT_SUMMARY_PERIOD_S: float = 1.0` (L153)
- `class _TokenBucket` (L156) — Private, lock-protected. `__init__(rate_hz, *, capacity=1.0, clock=time.monotonic)`; `try_consume() -> bool`.
- `class FailureBusPublisher` (L205) — `__init__(node, source, *, rate_limit_hz=None, summary_period_s=1.0, clock=None)`. Configure the publisher; does not open the ROS publisher.
  - `topic(self) -> str` (L268) — The full topic this publisher writes to.
  - `source(self) -> FailureSource` (L273) — The source layer this publisher represents.
  - `create_publisher(self) -> None` (L277) — Open the ROS 2 publisher with the bus QoS profile.
  - `start(self) -> None` (L299) — Start the 1 Hz `KIND_SUPPRESSED_SUMMARY` roll-up timer.
  - `stop(self) -> None` (L312) — Cancel the suppressed-summary timer (`on_deactivate`).
  - `destroy(self) -> None` (L318) — Tear down the publisher + timer (`on_cleanup`).
  - `publish(self, *, kind, severity, evidence, rskill_id='', trace_id=None) -> bool` (L329) — Emit one `FailureTrigger`, subject to rate-limiting (`False` when rate-limited).

### `python/observability/src/openral_observability/logging.py`
- `trace_context_processor(_logger, _method_name, event_dict)` — structlog processor that stamps `trace_id` / `span_id` on every log event. (L54)
- `resolve_log_level() -> int` (L70) — Resolve the OpenRAL log floor from `OPENRAL_LOG_LEVEL` (name or int); defaults to `INFO`, not `DEBUG`; an unparseable value falls back to the default rather than raising. Below the floor the stdlib check short-circuits before rendering/shipping, which matters since several DEBUG call sites fire per control tick. Governs log records only — dashboard span rows are banded separately by `dashboard.store._is_headline_span`.
- `apply_structlog_level_floor()` — Configure structlog's stock renderer to drop events below `resolve_log_level()`; called by `configure_observability` on its no-endpoint path, where the stock config otherwise printed every DEBUG event (a deploy runtime emits one per 750 Hz joint state and per camera frame — ~10 % of the process's GIL on an AGX Orin). (L103)
- `install_structlog_bridge(logger_provider)` — Wire the structlog processor chain to forward records to the OTel `LoggerProvider`, with both the bridge logger and the `openral` root logger set to `resolve_log_level()`. (L98)

### `python/observability/src/openral_observability/dashboard/store.py`
_In-memory aggregator for `openral dashboard` — feeds the SSE stream and the `/api/state` JSON endpoint. Thread-safe, bounded (200 events, 600 metric samples per series). Registered headline span families each populate one card slot in `self._topics`._

_**Event-log severity band** (distinct from the headline-card routing above). `_is_headline_span(name)` bands a span's Event Log row: ERROR status → `error`; a small allow-list of names/prefixes (`cli.command`, `rskill.execute`/`configure`/`activate`, `reasoner.tick`, `world.scene_objects`, `sim.run`, `detect.probe.*`) → `info`; everything else → `debug`. Deliberately an allow-list rather than a deny-list, so a new high-rate span stays quiet in the event log until promoted here instead of flooding it by default; every span is still indexed in full for `openral replay`._

- `class TelemetryEvent` — Frozen dataclass holding one event log row (`ts_unix`, `kind`, `title`, `attrs`, `severity`). (L229)
  - `to_json(self) -> dict[str, Any]` (L255) — Return a plain-dict view suitable for JSON serialization.
- `class TelemetryStore` — Read-side aggregator over OTLP signals. (L381)
  - `ingest_spans(payload: list[ResourceSpans]) -> int` — Decode + record spans; populates headline cards, increments span-event counters, publishes a delta to every subscriber queue. Returns the number of spans recorded. (L474)
  - `ingest_metrics(payload: list[ResourceMetrics]) -> int` — Decode + record metric data points; appends per-series samples and tracks cumulative sums. (L508)
  - `ingest_logs(self, payload) -> int` (L529) — Decode + record OTLP `ResourceLogs` (the structlog→OTel bridge) as event-log rows: body → title, logger name → kind, severity number → level. Shares the bounded event ring with spans; the UI defaults the Debug chip off. Returns the number of log records recorded.
  - `snapshot() -> dict[str, Any]` — One-shot view: service identity, headline cards, event ring, counters, metric series with p50/p95. (L568)
  - `subscribe() -> asyncio.Queue` — Register an SSE subscriber. The queue is bounded; on overflow the oldest payload is dropped so the producer never blocks. (L746)
  - `unsubscribe(queue) -> None` — Drop a subscriber's queue. (L762)
  - `set_estopped(self, value) -> None` (L573) — Force the e-stop latch flag from an authoritative operator action.
  - `set_safety_status(*, latched, drop_reason, drop_reason_label, detail, rskill_id, trace_id, stamp_unix) -> None` — Record one `SafetyStatus` from the latched `/openral/safety_status` topic and push it to SSE subscribers. The only authoritative safety state the dashboard has, distinct from the span-inferred latch which stops updating the moment a kernel actually latches. `stamp_unix` is load-bearing: it lets the card tell a live latch from a dead publisher's leftover durable value. (L591)
  - `list_traces(self) -> list[dict[str, Any]]` (L935) — Return one record per indexed trace_id, most-recent first.
  - `lookup_trace(self, trace_id) -> list[dict[str, Any]] | None` (L953) — Return every indexed span for `trace_id` in chronological order, or `None`.

- `TelemetryStore.set_perception_detections(self, *, camera, detections, model_id, frame_width, frame_height, stamp_unix, flip_180=False) -> None` (L644) — Record one detector result as a camera overlay and push it to SSE subscribers. `camera` is the wire `sensor_id`, the join key between a frame and its boxes; `frame_width`/`frame_height` let the tile scale boxes onto its aspect-preserving thumbnail. An empty `detections` list is recorded, not dropped, so "found nothing" is expressible rather than leaving a stale overlay painted.
- `TelemetryStore.set_perception_masks(self, *, camera, masks, rskill_id, stamp_unix, flip_180=False) -> None` (L701) — Record one segmenter result as a camera overlay: plural full-frame masks in the producer's area-ascending order, with advisory scores carried for display only (never used to rank or filter). Fed from the segmenter node's diagnostic re-publication of its one-shot service reply, off by default. Display only — nothing in the safety or attachment path reads this topic.

### `python/observability/src/openral_observability/dashboard/perception_overlay_subscriber.py`
_Detector boxes + segmenter masks drawn over the camera tiles. Same shape as `safety_status_subscriber.py` — one node created at launch, spun on a daemon thread, inert-but-harmless without rclpy / the `openral_msgs` overlay. Read-only and **advisory**: it decides what an operator sees, never what the robot does._

- module constant `PERCEPTION_OBJECTS_TOPIC: str = "/openral/perception/objects"` — the detector node's `output_topic` default. (L48)
- module constant `PERCEPTION_MASKS_TOPIC: str = "/openral/perception/masks"` (L49)
- module constant `PERCEPTION_MASKS_TOPIC: str = "/openral/perception/masks"` — the segmenter node's `debug_masks_topic` default; a diagnostic re-publication of its latest reply, off by default there, so subscribing costs nothing when nobody enabled it.
- `dashboard_flip_180() -> bool` — Whether this host rotates the dashboard's display copy of camera frames, parsing `OPENRAL_DASHBOARD_FLIP_180` with the same truthy set as the sensor leg and the world-state node. Stamped onto each overlay so the renderer applies exactly the flip the image got. (L64)
- `mono8_mask_to_png_b64(data: bytes, width: int, height: int) -> str` — Encode one mono8 mask as a base64 LA PNG whose alpha channel is the mask, so the frontend tints it with one composite instead of decoding pixels in JS. Raises `ValueError` when `data` is not exactly `width * height` bytes. (L93)
- `class PerceptionOverlaySubscriber` (L141) — Opens the node + subscription immediately, at the detector's sensor-class QoS (BEST_EFFORT + VOLATILE + KEEP_LAST=5) — a RELIABLE subscriber would never match and the overlay would sit silently blank.
  - `available(self) -> bool` (L218) — True when the subscription is live (rclpy + `openral_msgs` present).
  - `masks_available(self) -> bool` (L223) — True when the mask subscription is live (needs `SegmentMasks` built).
  - `close(self) -> None` (L310) — Tear down the node/executor and shut rclpy down only if it started it. `_on_objects` and `_on_masks` each drop a malformed payload with a logged decode-failure rather than letting an exception kill the spin thread and end overlays for the session. The mask leg is optional (its own `ImportError` guard) so an older `openral_msgs` build without `SegmentMasks` keeps drawing boxes; `masks_available` reports it separately from `available`. Wired in `run_dashboard` onto `app.state.perception_overlay`.

### `python/observability/src/openral_observability/dashboard/estop_publisher.py`
_Persistent ROS 2 e-stop publisher for the dashboard (safety-critical) — the dashboard's first rclpy publisher. One publisher created at dashboard startup so DDS discovery happens once; a later press publishes instantly instead of racing a fresh subprocess's discovery._

- `class EstopPublisher` — Creates the node + `/openral/estop` + `/openral/estop_cleared` publishers immediately at RELIABLE/VOLATILE/depth-10 QoS, matching the HAL/kernel/runner subscriptions; degrades to inert (callers fall back to the shell-out path) when rclpy/ROS is unavailable. (L24)
  - `available(self) -> bool` (L76) — True when the persistent publisher is live (rclpy + ROS present).
  - `trigger(self) -> bool` (L80) — Publish `/openral/estop` instantly. Returns `False` if unavailable.
  - `clear(self) -> bool` (L84) — Publish `/openral/estop_cleared` instantly. Returns `False` if unavailable.
  - `close(self) -> None` (L98) — Tear down the node + executor; shut down rclpy only if we started it.

### `python/observability/src/openral_observability/dashboard/safety_status_subscriber.py`
_The dashboard's first rclpy subscriber: one node created at launch, spun on a daemon thread, inert-but-harmless without rclpy / the `openral_msgs` overlay. Read-only: no publisher, no service client, no authority over the robot._

- module constant `SAFETY_STATUS_TOPIC: str = "/openral/safety_status"` (L30)
- `drop_reason_label(code: int, status_cls: Any) -> str` — Name a `drop_reason` by reverse-mapping it against the generated message class's own constants, so a constant added to `SafetyStatus.msg` needs no second table here; unknown values degrade to `"drop_reason_<code>"`. (L33)
- `class SafetyStatusSubscriber` — Opens the node + subscription immediately at the publishers' QoS (RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1) — a VOLATILE subscriber would never match and the card would sit silently empty. Wired in `run_dashboard` onto `app.state.safety_status`. (L66)
  - `available(self) -> bool` (L117) — True when the subscription is live (rclpy + `openral_msgs` present).
  - `close(self) -> None` (L134) — Tear down the node/executor and shuts rclpy down only if it started it.

### `python/observability/src/openral_observability/dashboard/discovery.py`
_mDNS advertise + browse for the live dashboard. Optional, requires the `mdns` extra; when `zeroconf` is not importable, `Discovery` stays disabled and the dashboard runs exactly as before._

- module constant `SERVICE_TYPE: str = "_openral-otlp._tcp.local."` — mDNS service type for all OpenRAL dashboard OTLP receivers. Single source of truth for the advertiser and browser. (L30)
- `class DiscoveredRobot(BaseModel)` — One mDNS-discovered OpenRAL service; the `/api/robots` wire shape used by external operator tooling. Fields: `name: str`, `addresses: list[str]`, `port: int`, `properties: dict[str, str] = {}`, `last_seen: float`. (L37)
- `class RobotRegistry` — Thread-safe map of discovered robots (zeroconf callbacks run off-thread). (L47)
  - `upsert(self, robot) -> None` (L55) — Insert or replace the entry keyed by `robot.name`.
  - `remove(self, name) -> None` (L64) — Remove the entry for *name* if it exists (idempotent).
  - `list_robots(self) -> list[DiscoveredRobot]` (L73) — Return all known robots sorted by name.
- `class Discovery` — Owns the `Zeroconf` instance, advertiser, and browser for the dashboard. Attribute `enabled: bool`. Wired into `run_dashboard` in `server.py`; `app.state.discovery` holds the instance (or `None` when the `mdns` extra is absent or zeroconf failed). (L83)
  - `robots(self) -> list[DiscoveredRobot]` (L99) — Delegate to the underlying registry.
  - `start(self, *, host, port) -> None` (L107) — Start browsing always; advertise only on a non-loopback, non-wildcard bind.
  - `stop(self) -> None` (L151) — Unregister the advertised service, cancel the browser, and close Zeroconf.

### `python/observability/src/openral_observability/dashboard/app.py`
- `create_app(store: TelemetryStore | None = None) -> FastAPI` — Build the dashboard ASGI app: the SSE/state routes, the OTLP/HTTP receivers, operator write endpoints (`/api/prompt`, `/api/estop_reset`), a live MJPEG camera stream, mDNS robot discovery, and the guarded write-controls (`/api/skill/execute`, `/api/param/set`, both OFF by default). (L775)

### `python/observability/src/openral_observability/dashboard/vad_assets.py`
- `class PinnedAsset(NamedTuple)` (L52) — `url: str`, `sha256: str`, `size: int` for one pinned binary asset.
- module constant `PINNED_VAD_ASSETS: dict[str, PinnedAsset]` (L65) — The three voice-prompt binaries no longer committed to git, pinned to exact upstream URLs + sha256 recorded in `static/vendor/vad/NOTICE.md`.
- `sha256_of(path: Path) -> str` — Hex sha256 digest of the file at `path`, read in chunks. (L94)
- `ensure_vad_assets() -> bool` (L142) — Best-effort: for each pinned asset, reuse a sha256-verified cache hit or download + verify one, then serve it from `static/vendor/vad/`. Never raises; a failed asset logs a warning and does not stop the others. Called best-effort from `run_dashboard` on every start (never gates startup).
- `vad_assets_available() -> bool` (L208) — Cheap presence-only check (no re-hash) of whether every pinned asset is currently served; backs `/api/config`'s `voice_prompt_enabled`.

### `python/observability/src/openral_observability/dashboard/server.py`
- `run_dashboard(*, host="127.0.0.1", port=4318, inprocess_cmd=None, store=None, log_level="warning") -> None` — Start uvicorn on `host:port` and block until SIGINT/SIGTERM. Calls `vad_assets.ensure_vad_assets()` best-effort before binding. Prints a single URL banner to stderr before binding. When `inprocess_cmd` is set, spawns the argv as a child process pointed at this dashboard's OTLP endpoint. Default port `4318` (OTLP/HTTP standard) rather than `8000`, to avoid clashing with common dev servers. (L56)
- `spawn_dashboard` / `attached_dashboard` — Inverse of `--inprocess`; see `dashboard/attach.py` below (a separate module despite the name symmetry with this file).

### `python/observability/src/openral_observability/dashboard/attach.py`
_The `--inprocess` inverse: spawn `openral dashboard` as a child process and attach the current process's telemetry to it. Used by `openral sim run --dashboard` / `openral deploy run --dashboard` / `openral benchmark run --dashboard`._

- module constant `_LOG` (L30) — `logging.getLogger(__name__)`.
- module constant `_ENV_ENDPOINT: str = "OTEL_EXPORTER_OTLP_ENDPOINT"` (L32)
- module constant `_ENV_PROTOCOL: str = "OTEL_EXPORTER_OTLP_PROTOCOL"` (L33)
- module constant `_HTTP_OK: int = 200` (L34) — healthz poll success code.
- `spawn_dashboard(*, host="127.0.0.1", port=4318, ready_timeout_s=10.0) -> Iterator[str | None]` [@contextmanager] (L38) — Spawn `openral dashboard` as a child, set the OTLP env, poll `/healthz` until ready, yield the URL, and SIGINT the child on exit. Yields `None` if `openral` isn't on PATH, the child dies early, or the health check never passes within the timeout.
- `_wait_healthy(healthz_url, child, *, timeout_s) -> bool` (L206) — Poll `healthz_url` until it returns `_HTTP_OK` or `child` exits or `timeout_s` elapses.
- `_shutdown_child(child) -> None` (L221) — SIGINT the child and wait, escalating if needed.
- `attached_dashboard(*, enabled, port=4318, subcommand=None, mode=None) -> Iterator[bool]` [@contextmanager] (L133) — Convenience wrapper for CLI commands that gate dashboard attach on a flag. `enabled=False` yields `False` as a true no-op (no FastAPI/uvicorn imports). `enabled=True` delegates to `spawn_dashboard`, re-runs `configure_observability` on the new endpoint, and re-opens the `cli.command` root span so workload spans have a real recording ancestor instead of being orphaned under the pre-dashboard root. Drains via `shutdown_observability` in `finally`. Yields `True` iff the child reported healthy.

### `python/observability/src/openral_observability/dashboard/store.py` — F7 trace index additions
_Bounded per-trace_id span index for query-time bag↔OTel join._

- `class _IndexedSpan` (L267) — Frozen-ish record retained by `trace_id`: `name`, `trace_id`, `span_id`, `parent_span_id`, `start_ns`, `end_ns`, `attrs`, `status_code`, `status_message`, `events`. `.to_json()` returns a plain dict carrying `duration_ms`.
- `TelemetryStore.list_traces() -> list[dict]` — One row per indexed trace_id (`trace_id`, `span_count`, `last_seen_unix`), most-recent first. Backs `GET /api/traces`.
- `TelemetryStore.lookup_trace(trace_id: str) -> list[dict] | None` — Every indexed span for `trace_id`, sorted ascending by `start_unix_ns`. `None` when the trace is not (or no longer) in the bounded index. Backs `GET /api/spans/{trace_id}`.
- `_TRACE_INDEX_MAX_TRACES = 64` / `_TRACE_INDEX_MAX_SPANS = 2048` — Memory caps. Older trace_ids evict FIFO on insertion.

### `python/observability/src/openral_observability/dashboard/app.py` — F7 routes
- `GET /api/traces` — JSON `{"traces": [...]}` from `TelemetryStore.list_traces`.
- `GET /api/spans/{trace_id}` — JSON `{"trace_id", "spans": [...]}` from `TelemetryStore.lookup_trace`; 404 when the trace is not indexed.
- `GET /api/config` — JSON `{"jaeger_ui_url": "..."}` from the `OPENRAL_JAEGER_UI_URL` env (default `""`). The UI uses this to enable/disable the footer "open in jaeger" link rather than guessing a `localhost:16686` link that may be dead.

### `python/observability/src/openral_observability/tracing_lttng.py`
_Opt-in LTTng tracepoints around the realtime hot path. No-op when `OPENRAL_ROS2_TRACING` is unset; falls back to JSONL when `lttngust` is missing._

- `ENV_TRACING_GATE = "OPENRAL_ROS2_TRACING"` (L60) — Truthy values (`1`/`true`/`yes`/`on`) enable the backend; anything else leaves every tracepoint a no-op.
- `ENV_TRACING_FALLBACK_DIR = "OPENRAL_ROS2_TRACING_FALLBACK_DIR"` (L66) — Override for the JSONL fallback directory (default `/tmp/openral-lttng-fallback`).
- `TP_RUNNER_TICK = "openral:runner_tick"` (L74)
- `TP_HAL_READ_STATE = "openral:hal_read_state"` (L75)
- `TP_HAL_SEND_ACTION = "openral:hal_send_action"` (L76)
- `TP_SENSORS_READ_LATEST = "openral:sensors_read_latest"` (L77)
- `TP_WORLD_STATE_SNAPSHOT = "openral:world_state_snapshot"` (L78)
- `TP_SKILL_STEP = "openral:skill_step"` (L79)
- `TP_ACTION_PUBLISH = "openral:action_publish"` (L80)
- `TP_SAFETY_VALIDATE = "openral:safety_validate"` (L81) — Tracepoint base names; `lttng_tracepoint` appends `_begin` / `_end` suffixes.
- `is_enabled() -> bool` (L112) — Single source of truth for the gate.
- `lttng_tracepoint(name, **attrs) -> Iterator[None]` (L175) — Context manager that fires `<name>_begin` / `<name>_end` around the block. Attaches the active OTel `trace_id` as `otel_trace_id` so CTF traces can join back to OTel.
- `class LttngSession(name, output_dir)` (L93) — Identity of an active session.
- `class LttngSessionError(RuntimeError)` (L88) — Raised by the subprocess wrappers.
- module constant `_BACKEND_RESOLVED: bool = False` (L107) — whether `_resolve_backend` has run yet (cached-resolution guard).
- module constant `_BACKEND: Any = None` (L108) — the cached resolved tracer backend (`_LttngUstBackend` / `_JsonFallbackBackend` / `None`).
- module constant `_WARNED_ONCE: bool = False` (L109) — one-time fallback-warning latch consumed by `_warn_fallback`.
- module constant `_TRACEPARENT_PART_COUNT: int = 4` (L85) — expected `traceparent` dash-separated part count.
- module constant `_LOG` (L55) — `logging.getLogger(__name__)`.
- `start_session(*, name, output_dir) -> LttngSession` (L321) — `lttng create / enable-event openral:* / add-context / start`.
- `stop_session(*, name) -> None` (L346) — `lttng stop` + `destroy` (flush + teardown).
- `view_session(*, output_dir) -> None` (L358) — `babeltrace2 OUTPUT_DIR`; falls back to listing files when `babeltrace2` is absent.

### `python/dataset/src/openral_dataset/recorder.py`
_In-memory per-rollout accumulator with multi-sink fan-out._

- `@dataclass class EpisodeHeader(episode_idx, task_string, fps, robot_name, stamp_ns)` — Per-episode metadata pushed to sinks at `episode_start`. (L55)
- `@dataclass class DatasetFrame(episode_idx, frame_idx, observation_state, images, action, reward, terminated, truncated, stamp_ns, trace_id="", span_id="")` — Per-tick frame pushed to sinks at `record_frame`. `trace_id` (32 hex) / `span_id` (16 hex) carry the producing `rskill.tick` span's ids (ISSUE-109 forward link); `""` when no valid span was in scope. (L78)
- `@dataclass class EpisodeSummary(episode_idx, success, n_frames, stamp_ns)` — Per-episode close-out pushed to sinks at `episode_end`. (L119)
- `class DatasetSink(Protocol)` — Fan-out target with `open_episode` / `write_frame` / `close_episode` / `finalize`. (L137)
  - `open_episode(self, header) -> None` (L151) — Begin a new episode. Idempotent only within one episode.
  - `write_frame(self, frame) -> None` (L154) — Append one frame to the current episode.
  - `close_episode(self, summary) -> None` (L157) — Finalise the current episode (success flag, frame count).
  - `finalize(self) -> None` (L160) — Flush pending I/O and release resources. Idempotent.
- `class RolloutRecorder(*, robot, task_string, fps, sinks, repo_id=None)` — In-memory accumulator that fans every step out to one or more `DatasetSink` implementations and writes the OTel `openral.dataset.repo_id` / `episode_idx` / `frame_idx` attributes on the active `rskill.tick` span. (L164)
  - `episode_start(*, task_string=None) -> int` — Open a new episode; returns its idx. (L293)
  - `record_frame(*, observation_state, images, action, reward, terminated, truncated, stamp_ns, trace_id=None, span_id=None) -> int` — Append one frame. Captures the active `rskill.tick` span's `(trace_id, span_id)` onto the frame (ISSUE-109); explicit `trace_id`/`span_id` override the live capture (the offline converter replays the bag's original ids). (L338)
  - `episode_end(*, success: bool) -> EpisodeSummary` — Close the current episode. (L451)
  - `finalize() -> None` — Flush all sinks idempotently. (L484)
  - `prop fps, robot_name, repo_id, n_sinks, expected_state_shape` (L219–239) — Read-only views consumed by callers building the per-frame payload (`robot_name` L224, `repo_id` L229, `n_sinks` L234, `expected_state_shape` L239).
  - `expected_image_keys() -> tuple[str, ...]` — Camera keys (without `observation.images.` prefix) the sinks expect; derived from `RobotDescription.sensors[*].vla_feature_key`. (L253)

### `python/dataset/src/openral_dataset/schema_map.py`
_Pure `RobotDescription` → LeRobot v3 features dict mapping; no I/O, no lerobot import._

- module constant `_LEROBOT_IMAGE_DTYPE: str = "video"` (L31)
- module constant `_LEROBOT_STATE_DTYPE: str = "float32"` (L32)
- module constant `_LEROBOT_ACTION_DTYPE: str = "float32"` (L33) — LeRobot v3 feature dtypes `features_from_robot` assigns.
- module constant `_IMAGE_MODALITIES: frozenset[str]` (L38) — `{"rgb", "depth", "rgbd", "thermal", "ir"}`, sensor modalities mapped to image features.
- `@dataclass class FeatureSpec(key, dtype, shape)` — Decoupled feature descriptor; sinks translate to lerobot's `{'dtype', 'shape', 'names'}` format. (L42)
- `features_from_robot(robot: RobotDescription, *, fps: float) -> dict[str, FeatureSpec]` — Build the LeRobot v3 features dict for the recorder. Reads `ObservationSpec.state_shape`, `ActionSpec.dim`, and `SensorSpec.vla_feature_key` (image modalities only) from the robot manifest. (L59)

### `python/dataset/src/openral_dataset/bag.py`
_Mcap-backed ``DatasetSink`` for online hardware recording._

- module constant `TOPIC_TICK: str = "/openral/tick"` (L53)
- module constant `TOPIC_EPISODE: str = "/openral/episode"` (L54)
- module constant `TOPIC_IMAGE: str = "/openral/dataset/image"` (L59) — the three mcap topic names the converter imports by symbol.
- module constant `_TICK_SCHEMA: dict[str, Any]` (L66) — JSON Schema for the `/openral/tick` message.
- module constant `_IMAGE_SCHEMA: dict[str, Any]` (L99) — JSON Schema for the `/openral/dataset/image` message.
- module constant `_EPISODE_SCHEMA: dict[str, Any]` (L125) — JSON Schema for the `/openral/episode` message.
- module constant `PHASE_START: int = 0` (L139)
- module constant `PHASE_END: int = 1` (L140) — the two `Episode.phase` enum values.
- module constant `_NDIM_H: int = 1` (L144)
- module constant `_NDIM_HW: int = 2` (L145)
- module constant `_NDIM_HWC: int = 3` (L146) — expected `ndarray.ndim` for a mono/2D/HWC image frame.
- module constant `_QUEUE_MAXSIZE: int = 1024` (L152) — bound on the writer thread's `queue.Queue`; `write_frame` drops (counted in `n_dropped`) rather than block the hot path when full.
- module constant `_WRITER_JOIN_TIMEOUT_S: float = 5.0` (L157) — `finalize`'s join timeout on the writer thread.
- module constant `_MCAP_INSTALL_HINT: str` (L161) — install hint raised when `mcap` is not importable.
- module constant `_STOP` (L173) — sentinel `_Stop()` instance enqueued to tell the writer thread to exit.
- `class Rosbag2Sink(DatasetSink)` — Writes every `RolloutRecorder` event into an mcap file readable by `ros2 bag info` / Foxglove / mcap-cli. Daemon writer thread + bounded queue so `write_frame` never blocks on disk I/O. Topics: `/openral/tick` (metadata plus inline `observation_state`/`action` arrays), `/openral/episode` (phase markers), `/openral/dataset/image` (one frame per camera per tick) — self-sufficient for conversion, no separate topic join needed. (L176)
  - `bag_path(self) -> Path` (L251) — Output bag path (read-only).
  - `n_ticks_written(self) -> int` (L256) — Number of `/openral/tick` messages successfully written.
  - `n_episode_markers_written(self) -> int` (L261) — Number of `/openral/episode` messages (start + end) written.
  - `n_images_written(self) -> int` (L266) — Number of `/openral/dataset/image` messages successfully written.
  - `n_dropped(self) -> int` (L271) — Number of messages dropped because the writer queue was full.
  - `open_episode(self, header) -> None` (L277) — Open the bag on first call; emit Episode(PHASE_START).
  - `write_frame(self, frame) -> None` (L291) — Enqueue a Tick message (incl. inline `observation_state`/`action` + the frame's `trace_id`/`span_id`, ISSUE-109) and one DatasetImage message per camera; never blocks. The off-thread mcap write reads the ids off the frame because the OTel context is gone by then.
  - `close_episode(self, summary) -> None` (L334) — Emit Episode(PHASE_END) with success flag.
  - `finalize(self) -> None` (L344) — Drain queue, stop writer thread, close mcap. Idempotent.

### `python/dataset/src/openral_dataset/converter.py`
_Offline mcap rosbag2 → LeRobotDataset v3 converter._

- module constant `_MCAP_INSTALL_HINT: str` (L49) — install hint raised when `mcap` is not importable.
- `@dataclass class DatasetSummary(output_root, n_episodes, n_frames, n_success, repo_id)` — Returned by `from_bag` describing what landed on disk. (L56)
- `class Rosbag2ToLeRobotConverter` (L84) — Offline replay of a `Rosbag2Sink`-produced mcap into a LeRobotDataset v3.
- `Rosbag2ToLeRobotConverter.from_bag(*, bag_path, robot, output_root, repo_id=None, license="CC-BY-4.0", fps=None) -> DatasetSummary` — Walk a `Rosbag2Sink`-produced mcap, group ticks under episode markers, join each tick's inline arrays + camera frames, and replay each episode through a real `LeRobotDatasetSink` to produce a reloadable v3 dataset with real proprio/action/video. Each replayed tick re-injects the bag's original `(trace_id, span_id)` so the on-disk frame points at the source rollout, not the convert run. Raises `ROSConfigError` on a missing bag, missing episode markers, or a mismatched robot. (L104)

### `python/dataset/src/openral_dataset/frame_trace.py`
_ISSUE-109 — pivot a written LeRobotDataset frame back to its OTel ids._

- `read_frame_trace(*, root, episode_idx, frame_idx) -> tuple[str, str]` — Return the `(trace_id, span_id)` stamped on a v3 frame. Reads the `root/data/**/*.parquet` correlation columns directly via `pyarrow` (no video decode), so it works without a torchcodec/ffmpeg backend. Raises `ROSConfigError` when the root has no parquet, the dataset predates the columns, or no `(episode_idx, frame_idx)` row matches. Backs `openral replay --frame`. (L28)

### `python/dataset/src/openral_dataset/sinks.py`
_LeRobotDataset v3.0 (codebase_version="3.0") writer; deferred `LeRobotDataset.create` so per-camera shapes come from the first frame._

- module constant `DEFAULT_LICENSE: str = "CC-BY-4.0"` (L43)
- module constant `DEFAULT_VCODEC: str = "libsvtav1"` (L48) — default license / video codec for a new `LeRobotDatasetSink`.
- module constant `_FPS_INTEGER_TOLERANCE: float = 1e-6` (L51) — tolerance for treating a recorded fps as an integer.
- module constant `_TRACE_FEATURES: dict[str, dict[str, Any]]` (L59) — the `trace_id`/`span_id` string-feature specs added to every LeRobot v3 dataset (ISSUE-109).
- module constant `_LEROBOT_INSTALL_HINT: str` (L67) — install hint raised when `lerobot>=0.5.1` is not importable.
- `class LeRobotDatasetSink(DatasetSink)` — Implementation of `DatasetSink` writing LeRobot v3 datasets via real `lerobot.datasets.LeRobotDataset.create / add_frame / save_episode / finalize`. Lazy-imports lerobot at construction. (L85)
  - `__init__(*, root, robot, fps, repo_id=None, license="CC-BY-4.0", vcodec="libsvtav1")` — Raises `ROSConfigError` if lerobot ≥ 0.5.1 is not importable. (L120)
  - `open_episode(header) -> None` — Stash the task string for per-frame tagging. (L248)
  - `write_frame(frame) -> None` — Validates per-frame shapes against the declared features, then forwards to `LeRobotDataset.add_frame`. Adds the frame's `trace_id`/`span_id` as `string` parquet columns (ISSUE-109). (L267)
  - `close_episode(summary) -> None` — Calls `LeRobotDataset.save_episode(parallel_encoding=True)` and accumulates the per-dataset success counter. (L344)
  - `finalize() -> None` — Calls `LeRobotDataset.finalize()` then appends `dataset_success_rate` / `license` / `repo_id` and the dataset-level `trace_ids` / `n_traces` (distinct OTel traces, ISSUE-109) to `meta/info.json["metadata"]`, and writes the per-episode `episode_index → trace_id` map to the `meta/openral_traces.json` sidecar. (L385)


### `python/observability/src/openral_observability/replay/bag_reader.py`
_Read a rosbag2 (mcap) recording as a stream of `BagMessage` for the F7 bag↔OTel replay correlator._

- module constant `_TRACEPARENT_RE_BYTES` (L40) — `re.compile(rb"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})")`, matches a `traceparent` embedded in a raw CDR payload.
- module constant `_TRACEPARENT_RE_STR` (L43) — same pattern, `str` form, for JSON-schema-encoded payloads.
- module constant `_RAW_TRACE_ID_RE` (L48) — `re.compile(r"^[0-9a-f]{32}$")`.
- module constant `_RAW_SPAN_ID_RE` (L49) — `re.compile(r"^[0-9a-f]{16}$")`.
- `class BagMessage` (L84) — One rosbag2 record surfaced to the bag↔OTel replay correlator; fields `topic`, `log_time_ns`, `publish_time_ns`, `trace_id`, `traceparent`, `schema_name`, `payload_summary`.
- `_trace_id_from_json_payload(payload) -> tuple[str, str]` (L52) — Return `(trace_id_hex, traceparent)` from a `jsonschema` payload.
- `_resolve_mcap_path(path) -> Path` (L117) — Find the actual `.mcap` file inside a rosbag2 directory.
- `_extract_traceparent_from_ros2msg(payload) -> tuple[str, str]` (L135) — Return `(trace_id_hex, traceparent)` from a CDR payload, or empties.
- `read_bag(bag_path) -> Iterator[BagMessage]` (L144) — Yield every message in `bag_path` as a `BagMessage`.

### `python/observability/src/openral_observability/replay/correlator.py`
_Merge a rosbag2 message stream with dashboard-indexed OTel spans into one sorted timeline for `openral replay`._

- `class TimelineEntry` (L27) — One row in the correlated timeline; fields `kind: Literal["bag", "span"]`, `ts_ns`, `trace_id`, `topic`, `span_name`, `attrs`, `duration_ms`.
  - `to_json(self) -> dict[str, Any]` (L55) — Return a plain-dict view suitable for `json.dumps`.
- `list_bag_trace_ids(bag_messages) -> list[dict[str, Any]]` (L68) — Return distinct trace_ids in `bag_messages` with their per-trace counts.
- `build_timeline(bag_messages, spans, *, trace_id=None) -> list[TimelineEntry]` (L95) — Merge `bag_messages` + `spans` into one sorted timeline.

### `python/observability/src/openral_observability/replay/trace_query.py`
_Minimal read-only HTTP client for the dashboard's F7 trace-query endpoints (`GET /api/traces`, `GET /api/spans/{trace_id}`)._

- module constant `_HTTP_NOT_FOUND: int = 404` (L23)
- `class TraceQueryError(RuntimeError)` (L26) — Raised when the dashboard does not return a usable JSON body.
- `class DashboardTraceClient` (L31) — Minimal read-only HTTP client for the dashboard trace endpoints. `base_url: str = "http://127.0.0.1:8000"`, `timeout_s: float = 5.0`.
  - `list_traces(self) -> list[dict[str, Any]]` (L45) — Return `GET /api/traces` decoded, sorted server-side most-recent first.
  - `get_spans(self, trace_id) -> list[dict[str, Any]]` (L54) — Return every span for `trace_id`; empty list when not indexed.

### `python/observability/src/openral_observability/replay/cli.py`
_`openral replay` implementation: record a bag with the right topics, then join it against dashboard-indexed OTel spans by trace_id._

- module constant `ProfileName` (L35) — `Literal["slim", "full"]`, the two `ros2 bag record` topic/regex profiles.
- module constant `RECORD_PROFILES: dict[str, dict[str, list[str]]]` (L45) — the `slim` (safety/estop/world-state-slow) and `full` (adds world-state-fast, joint_states, tf, all perception/sensors) topic + regex lists.
- `build_record_command(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=()) -> list[str]` (L85) — Compose the `ros2 bag record` argv for `profile`.
- `run_record(*, profile, output_dir, storage="mcap", extra_topics=(), extra_regex=(), dry_run=False) -> tuple[list[str], subprocess.CompletedProcess[bytes] | None]` (L210) — Invoke `ros2 bag record` with the chosen profile.
- `class ReplayResult` (L133) — Output of `run_replay` — both summary + the joined timeline; fields `trace_id`, `bag_trace_ids`, `timeline`, `bag_path`.
  - `to_json(self) -> dict[str, Any]` (L152) — Return a plain-dict view suitable for `json.dumps`.
- `run_replay(*, bag_path, trace_id, dashboard_url) -> ReplayResult` (L162) — Read `bag_path`, fetch matching spans, return the joined timeline.
- `write_timeline(result, out_path) -> None` (L283) — Persist a `ReplayResult` as pretty-printed JSON to `out_path`.
