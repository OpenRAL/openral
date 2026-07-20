# NVIDIA Cosmos 3 Edge as the S2 reasoner

> Assessment + integration record for the `cosmos` reasoner provider
> (`OPENRAL_REASONER_LLM_PROVIDER=cosmos`,
> `openral_reasoner.cosmos3.Cosmos3ToolUseClient`). Last reviewed
> 2026-07-20 — the day the Edge weights shipped; recheck the
> [Validation status](#validation-status) table before relying on
> anything marked unverified.

## What it is

[Cosmos 3](https://github.com/nvidia/cosmos) is NVIDIA's omnimodal
world-model family (announced 2026-05-31; the Edge tier launched
2026-07-15 and its weights landed on Hugging Face 2026-07-20 at
[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge)). One
Mixture-of-Transformers model, two towers:

- **Reasoner tower** — autoregressive VLM: text/image/video in → text out.
  Trained for world understanding, spatial grounding (2D boxes, 2D/3D
  points), physical-plausibility judgment, task planning, action
  forecasting, and embodied decision making. Callable independently of the
  generator; up to 256K context; Qwen3-VL-compatible message conventions.
- **Generator tower** — diffusion: video / audio / **action-chunk** out
  (world simulation, policy rollout). Not used by the reasoner integration;
  it is the future WAM / policy-rSkill surface (see
  [Layer-boundary notes](#layer-boundary-notes)).

Family tiers: **Edge 4B** (on-device — Jetson Thor T2000/T3000, RTX;
trained from scratch on a Nemotron backbone), **Nano 16B** (workstation,
Qwen3-VL-initialised), **Super 64B** (datacenter). All three serve the same
API, so the tier is just `OPENRAL_REASONER_LLM_MODEL`.

## Why it fits the S2 slot

The OpenRAL S2 reasoner is event-driven at a 0.2 Hz heartbeat, sees a
structured text snapshot (no pixels on the tick path), picks exactly one
typed tool call per tick, and separately answers single-frame VQA for the
completion-adjudication gate (`describe_image`). Against that contract:

| S2 requirement | Cosmos 3 Edge |
|---|---|
| Typed tool calls via provider tool-use API (CLAUDE.md §3) | Served behind vLLM's OpenAI-compatible chat-completions API with `--enable-auto-tool-choice`; the existing `tool_choice="required"` wire path is reused unchanged |
| ~0.2 Hz tick budget | 4B AR decode on an edge GPU is comfortably sub-second per short tool call — orders of magnitude inside budget (measured numbers pending, see below) |
| `describe_image` completion gate | Native VLM — the *same local model* adjudicates completion; today's local text-only baselines (e.g. Ollama Qwen3 8B) cannot do this without a second cloud/VLM endpoint |
| Physical/embodied judgment | This is the model's actual training objective (robotics / AV / warehouse domains), vs. general-web LLMs adapted to it |
| On-robot, offline, no PII egress | Fully local; no cloud round-trip, no per-token cost, trivially satisfies the no-PII-in-cloud-logs rule |
| License compatible with commercial deployment | **OpenMDW-1.1** (Linux Foundation) — commercial + noncommercial OK. No `OPENRAL_ALLOW_NONCOMMERCIAL` guard needed. Weights-license fact per CLAUDE.md §1.9; OpenRAL code stays Apache-2.0 |

## Integration shape (what shipped)

- `openral_reasoner.cosmos3.Cosmos3ToolUseClient` — subclass of
  `OpenAICompatibleToolUseClient`; identical wire path, plus a managed
  local-server lifecycle (probe `GET /v1/models` → lazily spawn
  `tools/cosmos3_reasoner_sidecar.py` when the loopback endpoint is down →
  bounded readiness wait → terminate-only-our-child teardown).
- `tools/cosmos3_reasoner_sidecar.py` — uv-provisions an isolated Python
  3.12 venv from the hash-pinned pure-PyPI
  `tools/sidecar_requirements/cosmos3_reasoner.lock` (vllm 0.24.0 at lock
  time), then execs
  `vllm serve nvidia/Cosmos3-Edge --enable-auto-tool-choice
  --tool-call-parser hermes --max-model-len 32768` on port **8901**.
- Config surface: `OPENRAL_REASONER_LLM_{MODEL,BASE_URL,API_KEY,TIMEOUT_S,MAX_TOKENS}`
  as usual, plus `OPENRAL_COSMOS3_AUTOSTART` / `OPENRAL_COSMOS3_BOOT_TIMEOUT_S`
  / `OPENRAL_COSMOS3_SIDECAR`. Self-managed serving (your own `vllm serve`,
  or the Cosmos 3 Reasoner NIM container) is a `BASE_URL` away.
- `openral doctor` knows the provider (default base URL, no-key, no-model
  requirements; `Cosmos 3` probe row that reports `info` — not `warn` —
  when the endpoint is down, because autostart is the normal cold state).

## Expected footprint

4B parameters, BF16-only officially tested → ~8 GB weights + KV cache.
Comfortable on RTX-class (≥12 GB) and Jetson Thor (T3000 32 GB; T2000
16 GB is tight with a co-resident S1 VLA — budget VRAM deliberately, or
serve Cosmos 3 on a second GPU / box and point `BASE_URL` at it).
`--max-model-len 32768` (default) keeps the KV cache bounded; a reasoner
tick's context is a few thousand tokens.

## Risks and open questions

1. **Tool-calling reliability of a 4B world model is unproven.** Cosmos 3's
   reasoner is trained for physical reasoning and grounding, not
   function-calling agent traces. The palette's `execute_rskill__*` /
   `decompose_mission` discipline that GPT-5.5-class models handle well may
   degrade at 4B. Mitigations already in place: `tool_choice="required"`,
   Pydantic union validation with `ROSReasonerInvalidPlan` feedback into the
   next prompt, and the per-kind retry cap. Escalation path: same provider,
   `MODEL=nvidia/Cosmos3-Nano`.
2. **Tool-call parser.** Edge's reasoner is Nemotron-backbone (Nano/Super
   are Qwen3-VL-based); the family documents Qwen3-VL-compatible message
   conventions, so the sidecar defaults to vLLM's `hermes` parser. If Edge's
   emission format diverges in practice, override `--tool-call-parser`
   (the boot helper exposes it) — and if vLLM ships a dedicated `cosmos3`
   parser, switch the default.
3. **Serving-stack freshness.** Nano/Super reasoner serving is documented
   for vLLM ≥0.23; the Edge tier's dedicated vLLM integration is newer than
   the lock may assume. If `vllm serve nvidia/Cosmos3-Edge` rejects the
   architecture, bump the lock (`uv pip compile …/cosmos3_reasoner.in`) —
   Nano on a workstation is the fallback that is known-documented.
4. **Reasoning-trace latency.** Cosmos 3 supports explicit `<think>`
   reasoning; long traces would eat the 120 s call timeout on small GPUs.
   The reasoner does not request the explicit-reasoning format; if the model
   emits it anyway, cap it with `OPENRAL_REASONER_LLM_MAX_TOKENS`.
5. **No Edge-tier public benchmarks yet.** NVIDIA published family-level
   results (VANTAGE-Bench, PAI-Bench, Physics-IQ, RoboLab leads) but no
   Edge-specific numbers at launch. Treat quality claims as
   family-level until measured here.

## Validation status

Honesty ledger (CLAUDE.md §1.2) — this container has no NVIDIA GPU, so
everything model-side is pending a GPU host:

| Item | Status |
|---|---|
| Factory / client / autostart lifecycle / doctor rows | ✅ unit-tested (`tests/unit/test_reasoner_cosmos3.py`, `tests/unit/test_doctor.py`) |
| Tool-call wire path through `Cosmos3ToolUseClient` | ✅ unit-tested against the openai-SDK network-boundary double |
| Lock resolves (vllm 0.24.0, torch 2.11.0, py3.12) | ✅ compiled + hash-pinned |
| `vllm serve nvidia/Cosmos3-Edge` boots and loads the reasoner tower | ⬜ pending GPU host |
| Live `select_tool` tick + `describe_image` gate against the served model | ⬜ pending GPU host |
| Tool-call reliability vs. the deploy-sim baseline (collective-goal decomposition) | ⬜ pending eval run |
| Tick latency / VRAM on Jetson Thor + RTX reference hosts | ⬜ pending hardware |

## Layer-boundary notes

This integration touches **Layer 4 (Reasoning) only** — a new
`ToolUseClient` backend behind the existing Protocol; no layer boundary
moved. Two adjacent opportunities are intentionally *not* part of it:

- **WAM (Layer 5):** Cosmos 3's generator tower is an action-conditioned
  world model — exactly the `openral_wam` protocol's planned
  mental-simulation backend (the roadmap already names Cosmos Predict; the
  unified Cosmos 3 generator supersedes it). Separate decision + PR.
- **S1 policy rSkill:** `nvidia/Cosmos3-Edge-Policy-DROID` emits action
  chunks for DROID-format single-arm embodiments and could be packaged as
  an rSkill via the existing VLA adapter machinery. Separate decision + PR.
