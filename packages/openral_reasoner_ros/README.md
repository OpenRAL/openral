# `openral_reasoner_ros`

ROS 2 lifecycle wrapper for the OpenRAL S2 reasoner. The cadence, tool contract, palette, mission handling and grounding rules are in [`docs/reference/reasoner.md`](../../docs/reference/reasoner.md); this page is the node's ROS surface, its LLM configuration contract and its observability.

## What it does

A thin rclpy lifecycle node around [`openral_reasoner.ReasonerCore`](../../python/reasoner/src/openral_reasoner/core.py). Prompt and failure subscriptions stay live while the node is configured but inactive; their payloads wait in the context renderer, but no tick runs and nothing is dispatched until `on_activate`.

**Subscriptions**

| Topic | Type | Use |
|---|---|---|
| `/openral/world_state_slow` | `openral_msgs/WorldStateStamped` (5 Hz) | world state in the tick context |
| `/openral/failure/{hal,sensor,rskill,safety,wam,critic,world_state}` | `openral_msgs/FailureTrigger` | tiered preemption (below) |
| `/openral/perception/{motion,objects,ocr,scene_change}` | `openral_msgs/PromptStamped` | informational context |
| `/openral/prompt` | `openral_msgs/PromptStamped` | operator goals (Tier D, always preempts) |
| `/openral/safety_status` | `openral_msgs/SafetyStatus` (latched) | rendered as `## SAFETY`; a new latch forces a Tier-A tick |

Tier A (`/openral/failure/safety`) preempts at `severity >= SEVERITY_WARN`; tiers B and C (`hal`, `sensor`, `rskill`, `wam`, `critic`) at `SEVERITY_FAIL`. A matching `KIND_OBJECT_QUERY` on `/openral/failure/world_state` cancels the active manipulation and re-grounds within the per-task dispatch cap; it never produces a human handoff or changes e-stop behaviour. The node drops the FailureTriggers it publishes itself on `/openral/failure/rskill`. A HAL grasp miss (`KIND_GRASP_OUTCOME`) rides the same ladder and renders as a `[grasp_miss]` line in `## EXECUTION`.

**Tool dispatch.** Each tick the LLM picks one [`ReasonerToolCall`](../../python/core/src/openral_core/schemas.py) variant. Effect tools:

| Tool | Dispatch | Notes |
|---|---|---|
| `ExecuteRskillTool` | `rclpy.action.ActionClient` goal on `/openral/execute_rskill` | Streams feedback; emits a `FailureTrigger` with `KIND_CONTROLLER` (rejection, abort, server unavailable) or `KIND_TIMEOUT`. One goal at a time: a second dispatch is refused and the context carries an `in_flight:` line |
| `LifecycleTransitionTool` | `lifecycle_msgs/srv/ChangeState` on `<node>/change_state` | `configure` / `activate` / `deactivate` / `cleanup` only; `node` must be one of the palette's `node_ids`, checked by the decoder and again at dispatch, so the deadman and the kernel can never be targeted |
| `EmitPromptTool` | publish on `target_topic` | Only `EMIT_PROMPT_TOPICS` (`/openral/prompt`, `/openral/prompt_in/auto`), checked in the schema and at dispatch; stamps the OTel `traceparent` into `metadata_json` |
| `WaitTool` | none | Logged with its rationale; no ROS traffic |
| `ReloadGstPipelineTool` | `/openral/sensors/<id>/reload_pipeline` | Log-and-acknowledge stub: the sensor-service IDL is not on disk yet |

The reasoner never publishes `openral_msgs/ActionChunk`; actuation lives behind the action server and the safety kernel.

**ROS parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `tick_hz` | 0.2 | timer beat; it only runs gates |
| `ladder_state_path` | empty (off) | JSON path; the mission ledger and every replanning bound are snapshotted after each mutation and restored at `on_configure` |
| `dispatch_watchdog_s` | 30 (`<= 0` off) | bounds the dispatch phase; on expiry the node releases the busy latch, reactivates evicted peers and emits `KIND_CONTROLLER` (`state="dispatch_timeout"`) |
| `execute_server_wait_s` | 30 (`0` fails on first miss) | how long a dispatch re-probes (every 0.25 s) for `/openral/execute_rskill` before emitting `KIND_CONTROLLER` (`state="unavailable"`) |
| `object_tracking` | `off` | tracking mode (the deploy's `DeployRuntime.object_tracking`); when on, grasp / place declarations also carry `ObjectQuery`s |
| `grasp_target_voxel_m` | 0.0 (unset) | search-box padding; must be set on a manifest with a gripper, else `on_configure` raises `ROSConfigError` |
| `grasp_target_extrinsic_error_m` | `MAX_PLANAR_ERR_M` | extra padding on every side of a grounded box |

**Mission handling.** An operator prompt rebuilds the mission queue only when none is in progress; mid-flight it is guidance unless `metadata_json` carries `{"new_goal": true}` (`openral prompt --new-goal "..."`). The reasoner's own cascade re-prompts never rebuild the mission or reset the search and retry budgets. A wrapped ROS rSkill whose manifest sets `ros_integration.completes_task_on_success: true` (the in-tree Nav2 skill) completes the active task when its goal succeeds, unless the LLM sent `final_step: false`. The full ladder, attempt caps and grounding rules are in [`reasoner.md`](../../docs/reference/reasoner.md#node-behaviour).

**Execution lines.** A successful goal's `## EXECUTION` line carries the runner's `outcome_summary`; a result-only wrapped action reports its terminal status, scalar result fields and sent goal, bounded to 400 characters. When `ExecuteRskill.Result.task_outcome_json` names a source, the line also carries `task_outcome=<succeeded|failed|unknown> (<source>): <reason>`; `unknown` is never shown as success.

For a direct `locate_in_view` call, a missing service, failed request or non-empty `LocateInView.failure_reason` produces a detector failure prompt that says visibility is unknown. Only a completed response with `found=false` and an empty reason reads "not visible" (see the [`LocateInView` wire contract](../msgs/README.md#services--srv)).

## Completion check (`completion_check_node`)

A VLA emits no success signal, and Robometer's success head alone is not a completion signal (it can peak mid-carry as often as at placement). With `runtime.enable_completion_check` (plus the reward monitor, the critic leg and the scene VLM) the head is a cheap trigger and a VLM the judge. On a success score of 0.6 or more, a `DETACH` grasp event or a newly released track (one check per 10 s), the node:

1. holds the runner's commands (`/openral/rskill/hold`: the policy keeps stepping, nothing is sent);
2. asks `/openral/perception/query_scene` the reasoner's `COMPLETION_QUESTION` once per RGB camera;
3. ends the goal as a success (`/openral/rskill/complete`) when every camera answers yes, and otherwise releases the hold.

It only withholds or ends motion; every command still crosses the safety kernel. Each check logs `completion_check <reason>: done=<bool>`.

## Reasoner model registry

Reasoner selection is **model-first** (ADR-0088). The primary knob is
`OPENRAL_REASONER_MODEL`, a key in the curated
[`openral_core.REASONER_MODELS`](../../python/core/src/openral_core/schemas.py)
registry. Registry membership means the model cleared OpenRAL's robotics
tool-calling contract, except the 30 `*-openrouter` and `*-cloud` entries, which are
`measured=False` (see below).
The registry resolves the wire dialect, served model id, default endpoint,
auth requirement, hosting mode, and local-compute floor.

| `OPENRAL_REASONER_MODEL` | Served model | Client | Hosting / endpoint | Auth |
|---|---|---|---|---|
| `claude-opus-4-8` | `claude-opus-4-8` | `AnthropicToolUseClient` | Anthropic cloud | required |
| `claude-opus-5-5` | `claude-opus-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `claude-sonnet-5-5` | `claude-sonnet-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `claude-haiku-5-5` | `claude-haiku-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `gpt-5.5` | `openai/gpt-5.5` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `gpt-5.6` | `openai/gpt-5.6` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `cosmos3-edge` | `nvidia/Cosmos3-Edge` | `Cosmos3ToolUseClient` | managed local `http://127.0.0.1:8901/v1` | none |
| `qwen3.5-4b` | `qwen3.5:4b` | `OpenAICompatibleToolUseClient` (`tool_choice: auto`) | your Ollama, `http://localhost:11434/v1` | none |
| `qwen3.5-9b` | `qwen3.5:9b` | `OpenAICompatibleToolUseClient` (`tool_choice: auto`) | your Ollama, `http://localhost:11434/v1` | none |
| `<name>-openrouter` (16, below) | the OpenRouter id | `OpenAICompatibleToolUseClient` | OpenRouter cloud | OpenRouter key |
| `<name>-cloud` (14, below) | the Ollama tag | `OpenAICompatibleToolUseClient` (`tool_choice: auto`) | Ollama Cloud, `https://ollama.com/v1` | ollama.com API key |

The env contract is:

- `OPENRAL_REASONER_MODEL` — required registry key, or a raw model id for the
  explicit uncurated escape hatch.
- `OPENRAL_REASONER_ENDPOINT` — optional URL override. Location lives here,
  not in a provider name. For an OpenAI-dialect cloud model, an override means
  the operator owns auth policy for that endpoint; Anthropic-compatible
  endpoints still require a key.
- `OPENRAL_REASONER_API_KEY` — required only when the resolved endpoint needs it.
- `OPENRAL_REASONER_MAX_TOKENS` / `OPENRAL_REASONER_TIMEOUT_S` — optional
  per-call overrides. The `ollama` / `vllm` named endpoints cap a call at
  **60 s**; a large local model on a long-running mission reaches that ceiling
  (`tick error: … Request timed out`), and `OPENRAL_REASONER_TIMEOUT_S` is the
  way past it. Before raising it, read `llm_s` / `prompt_tokens` on the tick
  (below) — a ceiling hit driven by a growing prompt wants context management,
  not a longer timeout. The default is 10 s per call; a curated model can set its own (`ReasonerModel.timeout_s`: `glm-5.3-flash-cloud` and `glm-flash-latest-openrouter` 60 s, `qwen3.8-flash-openrouter` 30 s).
- `OPENRAL_REASONER_FIRST_CALL_TIMEOUT_S` — optional, openai dialect only. The
  read timeout of each request while the model may be loading. A cold local model
  loads on a request, and Ollama abandons the load when the client hangs up, so
  the per-call timeout alone would time out every tick. It applies until the
  endpoint first replies, and again after a request times out. Default **120 s**
  on a bare URL, on the `ollama` / `vllm` named endpoints and for a locally hosted
  curated model; off on a cloud endpoint. A value at or below the per-call timeout
  turns it off. The SDK retries a timed-out request twice, so a server that never
  answers holds a tick for three times this long.
- `OPENRAL_REASONER_MAX_SKILL_TOOLS` — optional per-tick cap `k` on skill tools.
  The reasoner ranks the palette's skills against the active mission task and
  pending operator prompts by keyword overlap and offers at most `k`: first the
  skill in flight, then the best match for the newest operator prompt, then the
  task's last dispatched skill, then the best navigate and look-at skill, then the
  highest scores. Ties prefer skills not yet tried for the task, then higher weight
  precision, then the id. `cosmos3-edge` defaults to 4; every other curated model
  sees the whole palette. `0` removes the cap. A call to a skill outside the subset
  is refused like an off-palette id.
- `OPENRAL_REASONER_DIALECT=anthropic|openai` — required only for an uncurated
  raw model id.
- `OPENRAL_REASONER_TOOL_CHOICE=auto|required` — for an uncurated model on a
  bare-URL `openai` endpoint or on the `anthropic` dialect (where `required` is
  sent as `{"type": "any"}`). Set `auto` for a model that rejects forced tool use
  with HTTP 400 (a thinking-mode gateway; the Claude 5 generation). Case-
  insensitive, and any other value fails at client build (and in `openral
  doctor`). Curated models and named `openai` endpoints carry their own value,
  so there it is ignored with a `reasoner.tool_choice_ignored` warning.

#### Models that refuse forced tool use

The reasoner asks for exactly one tool call per tick (`tool_choice` `required`,
Anthropic `any`). Some models refuse that: the Claude 5 generation answers
`400 tool_choice: type "tool" and "any" are not supported for this model`, and
some OpenAI-compatible gateways refuse `required` in thinking mode. The clients
handle this without configuration:

- The curated Claude 5 entries run `tool_choice: auto` from the start, with a
  16000-token cap (`_CLAUDE5_MAX_TOKENS`) because they think before calling a tool and thinking counts
  against `max_tokens` (an uncurated `anthropic` model gets the same cap).
- Any other model whose 400 refuses forced tool use is retried once with
  `auto`; if that retry succeeds the client stays on `auto`, logged as
  `reasoner.tool_choice_downgraded`. An unrelated 400 that only mentions
  `tool_choice` does not downgrade it.
- Under `auto`, a reply with no tool call gets one nudge ("respond with exactly
  one tool call"). A second prose reply is a tick error (`reasoner.no_tool_call`)
  that the node feeds back into the next prompt, like any undecodable call — so a
  model stuck in prose is told to call a tool rather than stalling the mission.
- A refusal (`stop_reason: refusal` / `finish_reason: content_filter`) or a
  reply cut off at the token cap is a tick error, not a `wait`.
- OpenAI's reasoning models (`gpt-5.6-*`, `gpt-6*`) refuse function tools on
  Chat Completions while reasoning is on, and some (`gpt-6-astra`,
  `gpt-6.1-sol`) cannot turn reasoning off. So the `openai` named endpoint (or
  OpenAI's URL spelled out) uses the **Responses API**
  (`OpenAIResponsesToolUseClient`): reasoning stays at the model's default,
  requests set `store: false`, and the bare model id is sent (`gpt-5.5`, not
  OpenRouter's `openai/gpt-5.5`). Every other OpenAI-compatible endpoint stays on
  Chat Completions; one that asks for `max_completion_tokens` gets it after one
  400 (`reasoner.max_tokens_param_switched`).

```bash
# Curated cloud model
export OPENRAL_REASONER_MODEL=gpt-5.5
export OPENRAL_REASONER_API_KEY=sk-or-...

# Curated managed-local model
export OPENRAL_REASONER_MODEL=cosmos3-edge
```

For the uncurated escape hatch see "Uncurated local endpoint" below.

`OPENRAL_REASONER_MAX_TOKENS` defaults to `16384` for the curated GPT-5.x
entries so OpenRouter does not reserve their full output window and reject a
low-balance request with HTTP 402. Anthropic keeps its client default. Managed
Cosmos gets a 120 s first-call timeout for kernel compilation.

For `cosmos3-edge`, a down managed endpoint auto-starts
`tools/cosmos3_reasoner_sidecar.py`; disable with
`OPENRAL_COSMOS3_AUTOSTART=0`. `OPENRAL_COSMOS3_BOOT_TIMEOUT_S` bounds first
boot. An explicit `OPENRAL_REASONER_ENDPOINT` points the same curated model at
a self-managed vLLM/NIM endpoint.

### Legacy migration

The old `OPENRAL_REASONER_LLM_{PROVIDER,MODEL,API_KEY,BASE_URL,...}` contract
was **removed in 0.3.0** (deprecated in 0.2.0 per ADR-0088). It is no longer
read at all: an environment that sets only the old vars now fails with
"`OPENRAL_REASONER_MODEL` is unset" rather than silently selecting a model.

Each var maps across directly:

| Legacy | Model-first |
|---|---|
| `OPENRAL_REASONER_LLM_PROVIDER` + `OPENRAL_REASONER_LLM_MODEL` | `OPENRAL_REASONER_MODEL` (+ `OPENRAL_REASONER_ENDPOINT`) |
| `OPENRAL_REASONER_LLM_BASE_URL` | `OPENRAL_REASONER_ENDPOINT` |
| `OPENRAL_REASONER_LLM_API_KEY` | `OPENRAL_REASONER_API_KEY` |
| `OPENRAL_REASONER_LLM_MAX_TOKENS` | `OPENRAL_REASONER_MAX_TOKENS` |
| `OPENRAL_REASONER_LLM_TIMEOUT_S` | `OPENRAL_REASONER_TIMEOUT_S` |

`PROVIDER` split into two axes (which model, where it runs): a hosted provider becomes `ENDPOINT=<name>` (`anthropic`, `openrouter`, `gemini`, `xai`, `deepseek`, `huggingface`, `ollama`, `ollama_cloud`, `vllm`, or `openai`), `cosmos` becomes the curated `cosmos3-edge`, and `openai-compatible` becomes `ENDPOINT=<url>` + `DIALECT=openai`. A named endpoint carries its own base URL, dialect, auth posture, cold-start timeout and `tool_choice` quirk, so `DIALECT` is needed only for a bare URL.

Tests use a deterministic `FakeToolUseClient` under
[`tests/integration/fakes/`](../../tests/integration/fakes/) — the only test
double permitted at this process boundary per CLAUDE.md §1.11.

## System prompt

The base system prompt (`openral_reasoner.DEFAULT_SYSTEM_PROMPT`) is a
robot-agnostic operating brief: one-tool-per-tick semantics, faithful
adherence to the operator goal, robot/scene-matched skill selection,
locate-before-manipulate (`recall_object`), navigate-to-approach
(`resolve_place` / Nav2 navigation skills), per-tick progress evaluation,
and observe-but-never-bypass safety/e-stop handling ("Python proposes,
C++ disposes").

At `on_configure` the node calls
[`resolve_reasoner_system_prompt`](../../python/reasoner/src/openral_reasoner/tool_use.py),
which composes:

1. **Base brief** — `DEFAULT_SYSTEM_PROMPT`, unless
   `OPENRAL_REASONER_SYSTEM_PROMPT` is set to a non-empty value (whitespace-only
   counts as unset), which replaces it.
2. **`## THIS ROBOT` block** — appended by `render_robot_context_prompt` from
   the active robot's `RobotCapabilities` (`robot_yaml` ROS parameter, or the
   `robot_capabilities` constructor arg): embodiment tags, whether it can
   locomote (gates the navigate-to-approach rule), manipulation/sensing
   hardware, payload, and control modes.

With no robot wired the prompt stays at the (possibly overridden) base brief
alone.

## Curated reasoner models

The library factory has no default and refuses to guess. `openral deploy sim`
does default to `OPENRAL_REASONER_MODEL=gpt-5.5`, with
`OPENRAL_REASONER_MAX_TOKENS=16384`; it was the most reliable model in live
collective-goal decomposition tests. The default needs
`OPENRAL_REASONER_API_KEY` and fails loudly without it.

### Cloud — GPT-5.5 / GPT-5.6 via OpenRouter or OpenAI

```bash
export OPENRAL_REASONER_MODEL=gpt-5.5  # or gpt-5.6
export OPENRAL_REASONER_API_KEY=sk-or-...
# With an OpenAI key instead, point the same model at OpenAI's own API
# (Responses API; uncurated ids such as gpt-6-sol work the same way):
# export OPENRAL_REASONER_ENDPOINT=openai
# export OPENRAL_REASONER_API_KEY=sk-...
```

### Cloud — Claude Opus 4.8 / Claude 5

```bash
export OPENRAL_REASONER_MODEL=claude-opus-4-8  # or claude-opus-5-5 / claude-sonnet-5-5 / claude-haiku-5-5
export OPENRAL_REASONER_API_KEY=sk-ant-...
```

The Claude 5 entries run `tool_choice: auto` (they refuse forced tool use) and
think before they act, so a tick can take longer than on Opus 4.8; raise
`OPENRAL_REASONER_TIMEOUT_S` if ticks time out at the 10 s default.

### Managed local — NVIDIA Cosmos 3 Edge

[Cosmos 3 Edge](../../docs/reference/cosmos3-edge-reasoner.md) is the 4B on-device
tier of NVIDIA's Cosmos 3 family (OpenMDW-1.1 weights, commercial use OK). One env
var is enough; the managed vLLM sidecar auto-starts on the first tick (first boot
downloads about 8 GB):

```bash
export OPENRAL_REASONER_MODEL=cosmos3-edge
```

It needs an NVIDIA GPU (Ampere or newer; 12 GB or more recommended, 8 GB works for
small palettes). A down managed endpoint auto-starts
`tools/cosmos3_reasoner_sidecar.py`; disable with `OPENRAL_COSMOS3_AUTOSTART=0`,
bound first boot with `OPENRAL_COSMOS3_BOOT_TIMEOUT_S`, or point
`OPENRAL_REASONER_ENDPOINT` at a self-managed vLLM or NIM server. Serving defaults,
the Jetson sizing and the measured tool-call reliability are on the assessment page.

### OpenRouter and Ollama Cloud — the unmeasured `*-openrouter` / `*-cloud` models

These 30 entries have `measured=False`: they are offered because their catalog entry
declares tools and image input, not because they cleared the tool-calling bar. The
client logs `reasoner.model.unmeasured` when it builds one and `openral doctor`
warns. Latency against the 10 s cloud default is unknown, except the three models
with their own timeout (listed under `OPENRAL_REASONER_TIMEOUT_S`). An alias such as
`~google/gemini-pro-latest` resolves server-side, so the client logs
`reasoner.model.served` when the answering model changes and the tick span carries
`reasoner.served_model`.

**OpenRouter** (OpenRouter key; Claude and GPT keep their own entries):

| `OPENRAL_REASONER_MODEL` | OpenRouter id |
|---|---|
| `gemini-pro-latest-openrouter` | `~google/gemini-pro-latest` |
| `gemini-flash-latest-openrouter` | `~google/gemini-flash-latest` |
| `grok-latest-openrouter` | `~x-ai/grok-latest` |
| `deepseek-flash-latest-openrouter` | `~deepseek/deepseek-flash-latest` |
| `kimi-latest-openrouter` | `~moonshotai/kimi-latest` |
| `glm-flash-latest-openrouter` | `~z-ai/glm-flash-latest` |
| `qwen3.8-max-prime-openrouter` | `qwen/qwen3.8-max-prime` |
| `qwen3.8-flash-openrouter` | `qwen/qwen3.8-flash` (`tool_choice: auto`) |
| `glm-5.3-flashx-openrouter` | `z-ai/glm-5.3-flashx` |
| `mistral-large-4-openrouter` | `mistralai/mistral-large-4-0` |
| `mistral-small-4-openrouter` | `mistralai/mistral-small-2603` |
| `muse-wonder-openrouter` | `meta/muse-wonder` |
| `muse-ember-openrouter` | `meta/muse-ember` |
| `minimax-m3-openrouter` | `minimax/minimax-m3` |
| `minimax-m3.1-flash-openrouter` | `minimax/minimax-m3.1-flash` |
| `nemotron-3.5-super-vl-openrouter` | `nvidia/nemotron-3.5-super-vl` |

**Ollama Cloud** (ollama.com serves large models over the same OpenAI-compatible API
as a local Ollama, behind an ollama.com API key). Set `OPENRAL_REASONER_API_KEY` to
the key (the robot console's **Ollama API key** setting does this) and pick a
`*-cloud` model. Any other ollama.com tag runs uncurated with
`OPENRAL_REASONER_ENDPOINT=ollama_cloud`.

| `OPENRAL_REASONER_MODEL` | Ollama tag |
|---|---|
| `deepseek-v4.1-flash-cloud` | `deepseek-v4.1-flash` |
| `gemma4-31b-cloud` | `gemma4:31b` |
| `glm-5.3-flash-cloud` | `glm-5.3-flash` |
| `inkling-2-small-cloud` | `inkling-2-small` |
| `kimi-k2.6-cloud` | `kimi-k2.6` |
| `kimi-k2.7-code-cloud` | `kimi-k2.7-code` |
| `kimi-k3-cloud` | `kimi-k3` |
| `kimi-k3.1-pro-cloud` | `kimi-k3.1-pro` |
| `mimo-v2.6-flash-cloud` | `mimo-v2.6-flash` |
| `mimo-v2.6-pro-cloud` | `mimo-v2.6-pro` |
| `minimax-m3-cloud` | `minimax-m3` |
| `minimax-m3.1-flash-cloud` | `minimax-m3.1-flash` |
| `mistral-large-4-cloud` | `mistral-large-4` |
| `muse-spark-open-cloud` | `muse-spark-open` |

Ollama Cloud ignores `tool_choice` like a local Ollama (the entries and the
`ollama_cloud` preset send `auto`). It loads every model with its full window, so
the client's truncated-prompt check, which guards a local Ollama's 4096-token
default, is off for it.

### On the robot — Qwen 3.5 4B / 9B via Ollama

`qwen3.5-4b` and `qwen3.5-9b` run on the robot's own [Ollama](https://ollama.com)
(vision, tools and thinking; Apache-2.0 weights; the Q4_K_M `qwen3.5:4b`
(3.4 GB) and `qwen3.5:9b` (6.6 GB) tags). They need no key and nothing but the
model (`just bootstrap-ollama --model qwen3.5:9b` for the 9B):

```bash
just bootstrap-ollama            # installs Ollama, pulls qwen3.5:4b, serves a 32K window
export OPENRAL_REASONER_MODEL=qwen3.5-4b
```

The bootstrap reads its tags from `REASONER_MODELS`, so it needs a Python that
imports `openral_core` (the repo's `.venv` after `just sync`, or
`OPENRAL_PYTHON`). On Linux it installs Ollama v0.32.14 from the GitHub
release tarballs, each checked against a sha256 pinned in
`scripts/bootstrap_ollama.sh` before anything is extracted, and registers the
systemd unit under the default `/usr/local` prefix. `OLLAMA_PREFIX` installs
elsewhere (files only), and `OLLAMA_PORT` serves on another port (the script
then prints the endpoint URL to export). It always talks to the local daemon,
whatever a client `OLLAMA_HOST` says.

**The context window is the one setting that matters.** The reasoner's prompt
is 9–26K tokens (system prompt plus one tool per skill); Ollama loads a model
with a 4096-token window on a GPU under 24 GiB
([docs](https://docs.ollama.com/context-length)) and **silently truncates** a
longer prompt, and its OpenAI-compatible API has no field to raise it. Start
the daemon with `OLLAMA_CONTEXT_LENGTH=32768` (`just bootstrap-ollama` sets it on
the daemon it starts and on the systemd unit of an Ollama it installs; an
existing systemd install: `sudo systemctl edit ollama` →
`Environment=OLLAMA_CONTEXT_LENGTH=32768`). The
client compares the prompt tokens Ollama reports with the request it sent
(confirming a suspected cut against the window `/api/ps` reports, so densely
packed text that fits is not refused) and fails the tick (`ROSPlanningError: … cut the prompt to fit`) rather than act
on a truncated prompt.

Ollama ignores `tool_choice`
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)),
so the `ollama` endpoint sends `auto`: a prose reply gets the one nudge, then
counts as an invalid plan.

Both entries meet the curation bar (at least 17/20 valid calls through the real
client with a 32K window and thinking on, on `so101_follower` and on the
`franka_panda` full palette). Measured on Ollama 0.32.14, RTX 5090:

| model | `so101_follower` (9,286 tokens) | `franka_panda` full palette (20,436 tokens) | median tick |
|---|---|---|---|
| `qwen3.5-4b` | 19/20 | 18/20 | 2.5 s |
| `qwen3.5-9b` | 17-20/20 | 20/20 | 3.1-3.8 s |

With thinking off, ticks drop to about 2 s but validity falls below the bar, so
both entries keep thinking on.

### Uncurated local endpoint — explicit escape hatch

Models outside `REASONER_MODELS` are not claimed compatible. They work when the
model id, endpoint, and dialect are explicit; the factory and `openral doctor`
warn that robotics tool-calling reliability is unverified.

```bash
# Linux: installs ollama's release tarballs only when each one's sha256 matches
# its pin in scripts/bootstrap_ollama.sh; refuses otherwise.
just bootstrap-ollama --model qwen3:8b
export OPENRAL_REASONER_MODEL=qwen3:8b
export OPENRAL_REASONER_ENDPOINT=ollama
```

## Synopsis

```bash
just ros2-build      # builds openral_msgs + openral_reasoner_ros
source install/setup.bash

# One curated model, e.g.:
export OPENRAL_REASONER_MODEL=claude-opus-4-8
export OPENRAL_REASONER_API_KEY=sk-ant-...

ros2 run openral_reasoner_ros reasoner_node
ros2 lifecycle set /openral_reasoner configure
ros2 lifecycle set /openral_reasoner activate
```

## Observability — what to expect on the dashboard

Each `ReasonerCore.tick` opens an OTel span named `reasoner.tick`
(see `openral_observability.reasoner_span`) with these attributes:

| Attribute | When set | Meaning |
|---|---|---|
| `reasoner.tick.idx` | Always | Monotonic per-`ReasonerCore` tick counter. |
| `reasoner.model` | When the client has a `model_id` | LLM model identifier (e.g. `claude-opus-4-8`). |
| `reasoner.force` | Always | `True` when the tick was preempted by `FailureTrigger.severity ≥ FAIL` or a new operator prompt. |
| `reasoner.tool` | Successful + retry-cap suppressed ticks | The `tool` discriminator of the `ReasonerToolCall` variant the LLM picked (e.g. `execute_rskill`). |
| `reasoner.rskill_id` | When the call carries a `rskill_id` (`execute_rskill`) | Skill id the LLM chose. |
| `reasoner.suppressed_reason` | Suppressed ticks | `palette_empty` or `retry_cap`. The `min_interval`, `safety_latched`, `llm_error_backoff`, `retry_cap_hold`, `mission_finished`, `wait_hold`, `backstop_not_due`, `heartbeat_idle`, `no_goal` and `llm_gave_up` gates fire BEFORE the span opens (so dashboards don't show noise). |
| `reasoner.tick.trigger` | Always | What started the tick: the event of a forced tick (`prompt`, `tool_result:<source>`, `skill_result`, `skill_failure`, `failure:<source>`, `safety_latched`, `object_query`, `mission_advance`, `verdict`, `active_search`, `decompose`) or the gate that let a timer beat through (`context_changed`, `backstop`, `wait_timeout`, `llm_error_retry`). |
| `reasoner.tick.coalesced_triggers` | When requests merged | Triggers that arrived while the previous tick was in flight and were folded into this one. |
| `reasoner.tick.skipped_since_last` | Always | Timer beats suppressed since the previous LLM call. |
| `reasoner.wait.until` / `reasoner.wait.timeout_s` | The model chose `wait` | What the wait is for and its bound. |
| `reasoner.llm_s` | Ticks that reached the LLM | Wall-clock of the provider round-trip alone. `ReasonerTickResult.elapsed_s` is end-to-end, so `elapsed_s - llm_s` is the reasoner-side overhead. |
| `reasoner.prompt_tokens` | When the client reports usage | Provider-reported prompt tokens, cache reads included. Latency that grows with this is a context-size problem: window or summarize the history rather than raising `OPENRAL_REASONER_TIMEOUT_S`. |
| `reasoner.tier` | Always | Trigger tier that drove this call: `A` (safety), `B` (replan: hal/sensor/rskill/wam), `C` (critic), `D` (operator/perception), or `heartbeat`. |
| `reasoner.offered_skills` | When a skill-tool cap is set | Skill ids offered this tick (`ReasonerCore.offered_palette`). The ranking (every skill's score and the query) is on the `reasoner.skill_subset` log, emitted when the subset changes. |
| `reasoner.mission_json` | When a mission is active | `MissionState.to_summary()` JSON — the ordered task queue (id/text/status/attempts/failures/verdict/rskill_id/depth/steps, with `max_attempts` and `max_dispatches`) the live dashboard renders as the Mission card checklist, plus `goal` (the operator goal the queue serves) and `splits` (each task `decompose_mission` subdivided, by id: its text and the steps it ran; its children's dot-path ids hang under it). A task's budget is `failures` of `max_attempts` and `attempts` of `max_dispatches`. Each entry of `steps` is one dispatch at the task: `rskill_id`, `prompt` and `outcome` — `running`, `ok`, `failed` (aborted, rejected, or counted as a failure by the ladder), `cancelled` (operator / e-stop) or `interrupted` (never reported back). Absent on bare-goal deploys. |
| `reasoner.error_kind` | Provider failure | `ROSPlanningError` subclass name; an `exception` event is added to the span. |

`reasoner.llm_s` and `reasoner.prompt_tokens` also appear on the
`reasoner.tick.selected` structured log, readable without opening Jaeger.

The active W3C `traceparent` captured inside this span is threaded onto the
outbound `EmitPromptTool` `PromptStamped.metadata_json` so the F7 bag↔OTel
correlator can join the published prompt back to the producing tick.

Spans are emitted via `opentelemetry-sdk`; a no-op (cost <1 µs) when
`configure_observability` was not called (no provider installed). The
`just docker-smoke-x86-reasoner` smoke installs a real provider so the
round-trip can be observed end-to-end inside the deploy image.

## See also

- [`openral_reasoner.core`](../../python/reasoner/src/openral_reasoner/core.py) — transport-agnostic orchestrator.
- [`packages/openral_prompt_router`](../openral_prompt_router/) — F10 prompt fan-in.
