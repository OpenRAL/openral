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
  time) **plus a SHA-pinned transformers overlay** for `cosmos3_edge`
  (`_TRANSFORMERS_EDGE_SHA`, no released transformers has it yet), then —
  for the Edge diffusers layout — **downloads the checkpoint and builds a
  flattened reasoner view** (`resolve_served_model` / `materialize_reasoner_view`)
  and execs `vllm serve <view> --served-model-name nvidia/Cosmos3-Edge
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192
  --gpu-memory-utilization 0.90 --enforce-eager` on port **8901**. Every one
  of those non-obvious flags was forced by a real failure on the 8 GB 4070
  (see the validation ledger).
- Config surface: `OPENRAL_REASONER_LLM_{MODEL,BASE_URL,API_KEY,TIMEOUT_S,MAX_TOKENS}`
  as usual, plus `OPENRAL_COSMOS3_AUTOSTART` / `OPENRAL_COSMOS3_BOOT_TIMEOUT_S`
  / `OPENRAL_COSMOS3_SIDECAR` / `OPENRAL_COSMOS3_GPU_MEM_UTIL`. Self-managed
  serving (your own `vllm serve`, or the Cosmos 3 Reasoner NIM container) is a
  `BASE_URL` away.
- `openral doctor` knows the provider (default base URL, no-key, no-model
  requirements; `Cosmos 3` probe row that reports `info` — not `warn` —
  when the endpoint is down, because autostart is the normal cold state).

## Expected footprint

The reasoner tower is ~3B params; at BF16 (the only officially-tested
precision) it loaded at **~6.4 GB resident on the 8 GB RTX 4070** with an
8192-token KV cache — a genuine fit, not a projection, but a *tight* one.
`--enforce-eager` (skip CUDA-graph capture) is load-bearing on 8 GB, and the
8192 window needs `OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` on this card. Note the
reasoner's own system prompt + tool schemas already run ~7.7K tokens, so a
sub-8192 window rejects a real tick — which puts **8 GB at the practical
floor**: it works, but with no room for a co-resident S1 VLA and little KV
headroom. A **≥12 GB card is the comfortable minimum**; Jetson Thor T3000
(32 GB) / T2000 (16 GB) — Edge's actual targets — have ample room, and the
sidecar's 0.90 default suits them. The model supports up to 131K context,
KV-VRAM permitting.

## Risks and open questions

0. **BLOCKER (live-confirmed): vLLM cannot yet run a forward pass on
   `cosmos3_edge`.** See the validation ledger — the served engine boots but
   500s on the first request via an upstream `get_rope_index` shape bug. This
   is the one thing standing between "boots" and "works", and it is not
   OpenRAL's to fix. Everything below assumes that clears.
1. **Tool-calling reliability of a 4B world model is unproven.** Cosmos 3's
   reasoner is trained for physical reasoning and grounding, not
   function-calling agent traces. The palette's `execute_rskill__*` /
   `decompose_mission` discipline that GPT-5.5-class models handle well may
   degrade at 4B. Mitigations already in place: `tool_choice="required"`,
   Pydantic union validation with `ROSReasonerInvalidPlan` feedback into the
   next prompt, and the per-kind retry cap. Escalation path: same provider,
   `MODEL=nvidia/Cosmos3-Nano`. (Not yet measurable — see risk 0.)
2. **Tool-call parser.** Edge's reasoner is Nemotron-backbone (Nano/Super
   are Qwen3-VL-based); the family documents Qwen3-VL-compatible message
   conventions, so the sidecar defaults to vLLM's `hermes` parser. The grammar
   builds correctly from our palette (confirmed live); whether Edge *emits* in
   that format is unmeasured (risk 0). Override `--tool-call-parser` if it
   diverges, or switch the default if vLLM ships a dedicated `cosmos3` parser.
3. **Serving-stack freshness — CONFIRMED, and handled.** No released
   transformers recognises `cosmos3_edge`; the sidecar pins the `main`-branch
   SHA that does. When a release ships it, drop the overlay (see
   `_TRANSFORMERS_EDGE_SHA`) and add `transformers>=<version>` to the `.in`.
   Nano/Super (standard layout, no view needed) are the known-documented
   fallback and are served by repo id directly.
4. **Reasoning-trace latency.** Cosmos 3 supports explicit `<think>`
   reasoning; long traces would eat the 120 s call timeout on small GPUs.
   The reasoner does not request the explicit-reasoning format; if the model
   emits it anyway, cap it with `OPENRAL_REASONER_LLM_MAX_TOKENS`.
5. **No Edge-tier public benchmarks yet.** NVIDIA published family-level
   results (VANTAGE-Bench, PAI-Bench, Physics-IQ, RoboLab leads) but no
   Edge-specific numbers at launch. Treat quality claims as
   family-level until measured here.

## Validation status

Honesty ledger (CLAUDE.md §1.2). Validated live on an **RTX 4070 Laptop
(8 GB, CUDA 13.0)**: 2026-07-20 (day the Edge weights shipped — pinned
stable stack) and 2026-07-21 (vLLM `main` nightly — first working
end-to-end tick).

| Item | Status |
|---|---|
| Factory / client / autostart lifecycle / doctor rows | ✅ unit-tested (`tests/unit/test_reasoner_cosmos3.py`, `tests/unit/test_doctor.py`) |
| Tool-call wire path through `Cosmos3ToolUseClient` | ✅ unit-tested against the openai-SDK network-boundary double |
| Sidecar view/argv/layout helpers | ✅ unit-tested (`tests/unit/test_cosmos3_sidecar.py`) |
| Lock resolves (vllm 0.24.0, torch 2.11.0, py3.12) | ✅ compiled + hash-pinned |
| `cosmos3_edge` arch recognised by the serving stack | ✅ **via SHA-pinned transformers overlay** — no *released* transformers knows `cosmos3_edge` yet (5.14.1 rejects it); `main`@`cbf4d720` accepts it. Codified in the sidecar. |
| Weights loadable by vLLM (diffusers subfolder layout) | ✅ **via the flattened reasoner view** (`materialize_reasoner_view`) — vLLM's loader can't follow the subfolder `weight_map`; the symlink view fixes it. |
| `vllm serve nvidia/Cosmos3-Edge` boots on 8 GB | ✅ "Application startup complete"; ~6.4 GB resident at BF16, 8192-token KV, `--enforce-eager`. On this 8 GB card the 8192 window needs `OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` (the default 0.90 leaves only ~0.54 GiB for KV, short of the ~0.88 GiB the window needs — 0.90 suits the ≥12 GB cards Edge targets). All knobs codified in the sidecar. |
| Reasoner prompt reaches the model | ✅ real request tokenised (7,707 tokens) and the xgrammar tool-call grammar built from the full OpenRAL palette (`execute_rskill__*`, `decompose_mission`, …). |
| Live `select_tool` tick end-to-end (pinned stable vLLM 0.24.0) | ❌ **blocked** — the first forward pass crashes in `transformers…cosmos3_edge.get_rope_index` (`IndexError`): vLLM's Transformers-multimodal fallback passes a 1-D `input_ids` where the modeling code expects 2-D. Hits every request. Not an OpenRAL bug. |
| Model itself is sound | ✅ under plain `transformers.generate` on the same 4070 the reasoner loaded in 9.5 s and produced coherent physical-reasoning text at **~46 tok/s** (BF16). |
| **Live `select_tool` tick end-to-end (vLLM `main`, 2026-07-21)** | ✅ **WORKS** — vLLM nightly (`0.23.1rc1.dev1329+g616c9bd0f`, contains the native `Cosmos3EdgeForConditionalGeneration` from [#48291](https://github.com/vllm-project/vllm/pull/48291)) + the one-line `k_norm_und_for_gen` weight-filter from still-open [#49190](https://github.com/vllm-project/vllm/pull/49190): a real `select_tool` tick through `Cosmos3ToolUseClient` returned a **validated `DecomposeMissionTool`** (correctly grounded subtask) in **1.49 s warm**; `describe_image` answered correctly in 0.65 s. The native impl also reads the diffusers layout **by repo id** — no flattened view needed on `main`. |
| Tool-call reliability (first look) | ⚠️ 1/3 ticks validated first-shot; the misses were exactly the designed retry-ladder cases (grounding-validator rejection → `ROSReasonerInvalidPlan` feedback; one prose reply). Real reliability eval vs. the deploy-sim baseline still pending. |
| 8 GB fit with the native impl | ✅ needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (now set by the sidecar) **and** `--kv-cache-dtype fp8` for the 8192-token window (flag added); ~6.75 GB resident, 15,392-token fp8 KV. |
| Tick latency | ✅ 1.5–2 s warm per tool-call tick on the 4070 — far inside the 0.2 Hz S2 budget. |
| Tool-call reliability vs. the deploy-sim baseline | ⬜ pending eval run (unblocked once the sidecar's pinned vLLM contains #48291) |
| Tick latency / VRAM on Jetson Thor | ⬜ pending hardware (Q1 2027 modules) |

### Upstream timeline & what unblocks the pinned sidecar

* **vLLM [#48291](https://github.com/vllm-project/vllm/pull/48291)** (merged 2026-07-14, **in no release** — v0.25.1 was cut 40 min after the merge without it): native Edge reasoner; bypasses the buggy fallback and loads the diffusers layout directly.
* **vLLM [#49190](https://github.com/vllm-project/vllm/pull/49190)** (open): skips the generator-tower `k_norm_und_for_gen` tensors (without it the native loader errors on weight mapping) + video-processing fixes. The weight-filter half is required for Edge; validated here as a one-line local patch.
* **transformers**: `cosmos3_edge` still release-less (5.14.1 lacks it); the SHA-pinned overlay remains required for config parsing on every path.
* **Nightly caveat**: the current vLLM nightly wheel's metadata carries a self-contradictory `torchcodec` constraint on x86 Linux (resolver-breaking); it installs only with `--no-deps` over an existing 0.24.0 dep tree — fine for validation, not lockable.

**Action when a vLLM release ships #48291 + #49190**: bump `cosmos3_reasoner.in`/`.lock`, drop the Edge branch of `materialize_reasoner_view` (native impl reads the repo layout), keep the transformers overlay until a transformers release lands, and re-run the reliability eval. Until then `provider=cosmos` on the pinned stable boots but 500s on inference; the cloud/local baselines in the reasoner README remain the working default.

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
