# NVIDIA Cosmos 3 Edge as the S2 reasoner

> Assessment + integration record for the curated `cosmos3-edge` reasoner model
> (`OPENRAL_REASONER_MODEL=cosmos3-edge`,
> `openral_reasoner.cosmos3.Cosmos3ToolUseClient`). Recheck the
> [Validation status](#validation-status) table before relying on
> anything marked unverified.

## What it is

[Cosmos 3](https://github.com/nvidia/cosmos) is NVIDIA's omnimodal
world-model family; weights are on Hugging Face at
[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge). One
Mixture-of-Transformers model, two towers:

- **Reasoner tower** — autoregressive VLM: text/image/video in → text out.
  Trained for world understanding, spatial grounding (2D boxes, 2D/3D
  points), physical-plausibility judgment, task planning, action
  forecasting, and embodied decision making. Callable independently of the
  generator; Qwen3-VL-compatible message conventions. Context: the model card
  says up to 256K tokens, but the checkpoint's own `config.json` sets
  `text_config.max_position_embeddings: 131072` (128K), which is the ceiling
  vLLM enforces. OpenRAL serves far less: the sidecar's `--max-model-len`
  default is 8192 (see [Expected footprint](#expected-footprint)).
- **Generator tower** — diffusion: video / audio / **action-chunk** out
  (world simulation, policy rollout). Not used by the reasoner integration;
  it is the future WAM / policy-rSkill surface (see
  [Layer-boundary notes](#layer-boundary-notes)).

Family tiers: **Edge 4B** (on-device — Jetson Thor T2000/T3000, RTX;
trained from scratch on a Nemotron backbone), **Nano 16B** (workstation,
Qwen3-VL-initialised), **Super 64B** (datacenter). All three serve the same
API. Edge is the curated registry entry; other tiers use the explicit
uncurated model + endpoint escape hatch until separately validated.

## Why it fits the S2 slot

The OpenRAL S2 reasoner is event-driven at a 0.2 Hz heartbeat, sees a
structured text snapshot (no pixels on the tick path), picks exactly one
typed tool call per tick, and separately answers single-frame VQA for the
completion-adjudication gate (`describe_image`). Against that contract:

| S2 requirement | Cosmos 3 Edge |
|---|---|
| Typed tool calls via provider tool-use API (CLAUDE.md §3) | Served behind vLLM's OpenAI-compatible chat-completions API with `--enable-auto-tool-choice`; the existing `tool_choice="required"` wire path is reused unchanged |
| ~0.2 Hz tick budget | Measured 1.3–2 s warm per tool-call tick (Jetson AGX Thor, RTX 4070), inside the 5 s heartbeat; the first tool call after boot is a one-off ~64 s grammar build (see [Validation status](#validation-status)) |
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
  `tools/sidecar_requirements/cosmos3_reasoner.lock` (vllm 0.28.0, torch
  2.13.0, on every platform) **plus a SHA-pinned transformers overlay** for
  `cosmos3_edge` (`_TRANSFORMERS_EDGE_SHA`). It asks the venv's vLLM whether
  it carries the native Edge model (`vllm_has_native_edge_model`; 0.28.0
  does) and serves the snapshot as-is; only a vLLM without it gets the
  **flattened reasoner view** (`resolve_served_model` /
  `materialize_reasoner_view`). It then execs `vllm serve <model>
  --served-model-name nvidia/Cosmos3-Edge --enable-auto-tool-choice
  --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.90
  --enforce-eager --max-num-seqs 4 --limit-mm-per-prompt '{"video":0,"image":1}'`
  (plus `--kv-cache-dtype` when set) on port **8901**, with
  `VLLM_USE_FLASHINFER_SAMPLER=0`. Every one of those non-obvious flags was
  forced by a real failure (see the validation ledger). Those are the dGPU
  defaults; a Jetson gets different sizing (see [Jetson defaults](#jetson-defaults)).
- Config surface: `OPENRAL_REASONER_{MODEL,ENDPOINT,API_KEY,TIMEOUT_S,MAX_TOKENS}`
  plus `OPENRAL_COSMOS3_AUTOSTART` / `OPENRAL_COSMOS3_BOOT_TIMEOUT_S`
  / `OPENRAL_COSMOS3_SIDECAR` (client) and `OPENRAL_COSMOS3_GPU_MEM_UTIL` /
  `OPENRAL_COSMOS3_MAX_MODEL_LEN` / `OPENRAL_COSMOS3_KV_CACHE_BYTES` /
  `OPENRAL_COSMOS3_QUANTIZATION` /
  `OPENRAL_COSMOS3_MAX_NUM_SEQS` / `OPENRAL_COSMOS3_LIMIT_MM_PER_PROMPT` /
  `OPENRAL_COSMOS3_SIDECAR_HOME` / `OPENRAL_COSMOS3_SIDECAR_VENV` (sidecar). Self-managed
  serving (your own `vllm serve`, or the Cosmos 3 Reasoner NIM container) is a
  `OPENRAL_REASONER_ENDPOINT` away.
- `openral doctor` knows the provider (default base URL, no-key, no-model
  requirements; `Cosmos 3` probe row that reports `info` — not `warn` —
  when the endpoint is down, because autostart is the normal cold state).

## Expected footprint

The reasoner tower is ~3B params; at BF16 (the only officially-tested
precision) it loaded at **~6.4 GB resident on an 8 GB GPU** with an
8192-token KV cache — a genuine fit, but a *tight* one. `--enforce-eager`
(skip CUDA-graph capture) is load-bearing at 8 GB, and the 8192 window needs
`OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` there. When this section was first written
(2026-08) the reasoner's system prompt plus tool schemas ran ~7.7K tokens, so
8192 was the smallest window that held a tick. The 2026-09-14 measurement in
[Validation status](#validation-status) supersedes that: real palettes now run
~10.2K (4 skills) to ~20K (14 skills) tokens, so the 8192 default 400s on any
real palette and 8 GB needs the 12288-token recipe given there. **8 GB is the
practical floor**: it works for small palettes only, with no room for a
co-resident S1 VLA and little KV headroom.

A **≥12 GB card is the comfortable minimum**; Jetson Thor T3000 (32 GB) /
T2000 (16 GB) — Edge's actual targets — have ample room, but they need the
[Jetson defaults](#jetson-defaults), not the 0.90 one. The checkpoint's limit is 131,072 tokens
(`max_position_embeddings`; the model card's "256K" is not what this
checkpoint's config allows), KV-VRAM permitting; raise `--max-model-len` to
use it.

## Jetson defaults

On a Jetson (`openral_core.gpu.is_tegra_host()`), `serve_defaults` changes
five flags. Each can still be overridden by env var or CLI flag. All of them
were measured on a Jetson AGX Thor (R38.4, 122 GiB) and an AGX Orin (R39.2,
61 GiB) with the real reasoner prompt. The full write-up, including the run
beside a π0.5 VLA, is in openral-pro `docs/reference/cosmos3-edge-jetson.md`.

| flag | dGPU | Jetson | why |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.90 | 0.15 | On unified memory this is a fraction of the whole board, checked as free at start-up. 0.90 asks a Thor for 110 GiB and refuses to boot next to a resident VLA (`Free memory on device cuda:0 (86.63/122.82 GiB) on startup is less than desired`). |
| `--kv-cache-memory-bytes` | unset | `4G` | Sizes the KV cache directly (37,440 BF16 tokens), so the utilization above only gates start-up. |
| `--max-model-len` | 8192 | 32768 | Real palettes are 17K+ tokens (`so100_follower` 17,181; with the read-only query tools 21,357). |
| `--enforce-eager` | on | off | Eager mode exists for 8 GB of VRAM. On Thor, CUDA graphs cost nothing in boot time once the compile cache is warm. On Orin they took a warm tick from 10.2 s (FP8 Marlin, eager) to 3.7 s (BF16, graphs). |
| `--quantization` | none | `fp8` on `sm_89`+ | On Thor, FP8 weights decode at 58 tok/s against 44 for BF16 at 17K context. Beside a back-to-back π0.5 loop, the median tick drops from 8.2 s to 3.5–4.7 s. Orin only gets weight-only Marlin FP8, so it stays BF16. The KV cache stays BF16 on both, because the checkpoint has no FP8 KV scales. |

The registry entry also sets `max_tokens_default=512`. Every invalid tick
measured was a runaway inside the tool-call grammar, and without a cap it
ran to the client's 120 s timeout; with the cap it stops after about 10 s.
The longest valid call measured was 503 tokens; raise the cap with
`OPENRAL_REASONER_MAX_TOKENS`.

Live check of the managed path on Thor with these defaults: the client
autostarted the sidecar (ready in 96 s from a warm cache), and real
`select_tool` ticks on the `so100_follower` palette decoded 7 of 12 through
the full Pydantic decode. Valid ticks took 1.2–4.2 s, and the five failing
ticks were capped at 9–13 s. Before this change the same path returned a 400
on every tick, because the prompt did not fit the 8192 window.

Validity across 80 ticks was 65% for FP8 weights against 73% for BF16
(n=60). That difference is not significant, but it leans the wrong way, so
the reliability eval below is the thing to run before relying on FP8. Pass
`OPENRAL_COSMOS3_QUANTIZATION=` to serve BF16.

Palettes over ~32K tokens (`franka_panda`: 44,047, or 48,223 with the query
tools) do not work on Edge at any window. With a 64K window, every serving
configuration gave 0–2 valid calls out of 8.

## Risks and open questions

0. **vLLM's Transformers fallback cannot run `cosmos3_edge`.** It boots but
   500s on the first request via an upstream `get_rope_index` shape bug. The
   pinned vLLM 0.28.0 carries the native Edge model
   ([#48291](https://github.com/vllm-project/vllm/pull/48291)) and does not
   take that path; a vLLM without it still hits the bug (the sidecar warns).
1. **Tool-calling reliability of a 4B world model is unproven.** Cosmos 3's
   reasoner is trained for physical reasoning and grounding, not
   function-calling agent traces. The palette's `execute_rskill__*` /
   `decompose_mission` discipline that GPT-5.5-class models handle well may
   degrade at 4B. Mitigations already in place: `tool_choice="required"`,
   Pydantic union validation with `ROSReasonerInvalidPlan` feedback into the
   next prompt, and the per-kind retry cap. Escalation path: same provider,
   `MODEL=nvidia/Cosmos3-Nano`. First look: 1/3 ticks validated first-shot;
   the reliability eval is pending.
2. **Tool-call parser.** Edge's reasoner is Nemotron-backbone (Nano/Super
   are Qwen3-VL-based); the family documents Qwen3-VL-compatible message
   conventions, so the sidecar defaults to vLLM's `hermes` parser. The grammar
   builds correctly from our palette (confirmed live), and Edge returned
   validated tool calls through it on Thor and x86. Override `--tool-call-parser` if it
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
   emits it anyway, cap it with `OPENRAL_REASONER_MAX_TOKENS`.
5. **No Edge-tier public benchmarks yet.** NVIDIA published family-level
   results (VANTAGE-Bench, PAI-Bench, Physics-IQ, RoboLab leads) but no
   Edge-specific numbers at launch. Treat quality claims as
   family-level until measured here.

## Validation status

Honesty ledger (CLAUDE.md §1.2). Validated live on an **RTX 4070 Laptop
(8 GB, CUDA 13.0)** on the pinned stable stack, then again on vLLM `main`
nightly for the first working end-to-end tick. The aarch64 rows are from a
**Jetson AGX Thor** (JetPack 7, 122 GiB unified), where cosmos3-edge now
serves and tool-calls for real.

**x86_64 now serves natively too.** The sidecar lock previously picked up
the workspace's `torchcodec<0.10` constraint, which capped x86 at vLLM
0.24.0 (below the release with native Edge support) while aarch64 resolved
0.28.0. Compiling the lock in isolation fixes this: it now pins
`vllm==0.28.0` universally, reproducing the aarch64-validated environment
(torch 2.13.0, torchcodec 0.16.0) on every platform. Verified live on an
8 GB x86_64 host: native architecture resolves, and **4/4 `select_tool`
ticks** through the real `Cosmos3ToolUseClient` returned a validated
`ReasonerToolCall`. The x86 ❌ rows below reflect the earlier pinned-0.24.0
configuration and no longer apply.

**Sizing on an 8 GB card.** The system prompt alone tokenizes to **2,804**
tokens, and the per-skill tool schemas add **7,154** for a 4-skill
`so100_follower` palette (~10.2 K total in flight) or **17,248** for a
14-skill `franka_panda` one (~20 K). The sidecar's `--max-model-len` default
of 8192 therefore 400s on *any* real palette.

A working 8 GB recipe is `--max-model-len 12288 --gpu-memory-utilization
0.97 --kv-cache-dtype fp8` (13,568 KV tokens), which fits the small palette
with room for the reply but not the 14-skill one. The defaults are
deliberately left alone: 12288 at the default 0.90 utilisation refuses to
start ("0.66 GiB KV cache is needed … available 0.2 GiB"), and raising
utilisation by default would be hostile to a GPU that also drives a
display.

| Item | Status |
|---|---|
| Factory / client / autostart lifecycle / doctor rows | ✅ unit-tested (`tests/unit/test_reasoner_cosmos3.py`, `tests/unit/test_doctor.py`) |
| Tool-call wire path through `Cosmos3ToolUseClient` | ✅ unit-tested against the openai-SDK network-boundary double |
| Sidecar view/argv/layout helpers | ✅ unit-tested (`tests/unit/test_cosmos3_sidecar.py`) |
| Lock resolves (vllm 0.28.0, torch 2.13.0, py3.12, every platform) | ✅ compiled + hash-pinned |
| `cosmos3_edge` arch recognised by the serving stack | ✅ **via SHA-pinned transformers overlay** — no *released* transformers knows `cosmos3_edge` yet (5.14.1 rejects it); `main`@`cbf4d720` accepts it. Codified in the sidecar. |
| Weights loadable by vLLM (diffusers subfolder layout) | ✅ **via the flattened reasoner view** (`materialize_reasoner_view`) — vLLM's loader can't follow the subfolder `weight_map`; the symlink view fixes it. |
| `vllm serve nvidia/Cosmos3-Edge` boots on 8 GB | ✅ "Application startup complete"; ~6.4 GB resident at BF16, 8192-token KV, `--enforce-eager`. On this 8 GB card the 8192 window needs `OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` (the default 0.90 leaves only ~0.54 GiB for KV, short of the ~0.88 GiB the window needs — 0.90 suits the ≥12 GB cards Edge targets). All knobs codified in the sidecar. |
| Reasoner prompt reaches the model | ✅ real request tokenised (7,707 tokens) and the xgrammar tool-call grammar built from the full OpenRAL palette (`execute_rskill__*`, `decompose_mission`, …). |
| Live `select_tool` tick end-to-end (pinned stable vLLM 0.24.0) | ❌ **blocked** — the first forward pass crashes in `transformers…cosmos3_edge.get_rope_index` (`IndexError`): vLLM's Transformers-multimodal fallback passes a 1-D `input_ids` where the modeling code expects 2-D. Hits every request. Not an OpenRAL bug. |
| Model itself is sound | ✅ under plain `transformers.generate` on the same 4070 the reasoner loaded in 9.5 s and produced coherent physical-reasoning text at **~46 tok/s** (BF16). |
| **Live `select_tool` tick end-to-end (vLLM `main`, 2026-07-21)** | ✅ **WORKS** — vLLM nightly (`0.23.1rc1.dev1329+g616c9bd0f`, contains the native `Cosmos3EdgeForConditionalGeneration` from [#48291](https://github.com/vllm-project/vllm/pull/48291)) + the one-line `k_norm_und_for_gen` weight-filter from still-open [#49190](https://github.com/vllm-project/vllm/pull/49190): a real `select_tool` tick through `Cosmos3ToolUseClient` returned a **validated `DecomposeMissionTool`** (correctly grounded subtask) in **1.49 s warm**; `describe_image` answered correctly in 0.65 s. The native impl also reads the diffusers layout **by repo id** — no flattened view needed on `main`. |
| Tool-call reliability (first look) | ⚠️ 1/3 ticks validated first-shot; the misses were exactly the designed retry-ladder cases (grounding-validator rejection → `ROSReasonerInvalidPlan` feedback; one prose reply). Real reliability eval vs. the deploy-sim baseline still pending. |
| 8 GB fit with the native impl | ✅ needs the expandable-segments allocator config (now set by the sidecar — `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` depending on the sidecar venv's torch) **and** `--kv-cache-dtype fp8` for the 8192-token window (flag added); ~6.75 GB resident, 15,392-token fp8 KV. |
| Tick latency | ✅ 1.5–2 s warm per tool-call tick on the 4070 — far inside the 0.2 Hz S2 budget. |
| Tool-call reliability vs. the deploy-sim baseline | ⬜ pending eval run (unblocked once the sidecar's pinned vLLM contains #48291) |
| Sidecar venv provisions on **linux-aarch64** (Jetson Thor) | ✅ **fixed** — the committed `--universal` lock was unsatisfiable there: its `nvidia-nccl-cu13==2.28.9 ; sys_platform == 'linux'` covered aarch64 too, where `torch==2.13.0` requires `2.29.7`. Regenerating the lock scopes it per platform (`nvidia-nccl-cu12==2.27.5` x86 / `nvidia-nccl-cu13==2.29.7` aarch64); `uv pip install --dry-run` on the Thor now resolves, selecting **vllm 0.28.0 / transformers 5.16.1 / torch 2.13.0**. |
| `cosmos3_edge` arch on aarch64 / vLLM 0.28.0 | ✅ **native, no transformers overlay needed** — `Resolved architecture: Cosmos3EdgeForConditionalGeneration`. vLLM 0.28.0 carries [#48291](https://github.com/vllm-project/vllm/pull/48291), so the x86 "blocked on the pinned stable release" rows above **do not apply to a Jetson**. The sidecar no longer assumes either way: `vllm_has_native_edge_model()` asks the venv's own `ModelRegistry`. |
| Flattened reasoner view vs. the native loader | ✅ **fixed** — the view *breaks* the native loader (`RuntimeError: Cannot find any model weights with …/Cosmos3-Edge-reasoner`); it exists for the Transformers fallback, which cannot follow the subfolder `weight_map`. `resolve_served_model(..., native_edge=True)` now serves the snapshot dir as-is: **3 shards, 4.66 GiB, 5.71 s**. |
| `vllm serve` reaches serving on Thor | ✅ **fixed** — engine init died in FlashInfer's JIT (`flashinfer/sampling.cuh:20:10: fatal error: curand.h: No such file or directory`; JetPack 7 ships the CUDA runtime, not the toolkit headers). The sidecar now defaults `VLLM_USE_FLASHINFER_SAMPLER=0` — a JIT-compiled sampler is not worth a hard boot dependency on the toolchain for a 4B model at 0.2 Hz. Server listens on 127.0.0.1:8901. (`ninja` must also be on `PATH`; it is, from the sidecar venv's own `bin`.) |
| **Live inference on Jetson AGX Thor** | ✅ **WORKS** — a real chat completion returned coherent physical-reasoning text (vllm-0.28.0, BF16, 8192-token window, `--enforce-eager`, `--gpu-memory-utilization 0.20`). This is precisely the forward pass that crashes on the x86 pinned stack. |
| **Live `select_tool`-shaped tick on Thor** | ✅ **WORKS** — `tool_choice="required"` over a `decompose_mission` schema returned a **validated tool call** with well-formed arguments and `finish_reason="tool_calls"`. |
| Tick latency on Jetson AGX Thor | ✅ **1.26 / 1.26 / 1.36 s warm** per tool-call tick — inside the 0.2 Hz S2 budget, and in line with the 1.5–2 s measured on the 4070. **First** tool-call request cost **64 s** cold (xgrammar builds the tool-call grammar); that is a one-off per grammar, not a per-tick cost, but it does mean the first reasoner tick after boot blows a 5 s budget. |
| Tool-call reliability vs. the deploy-sim baseline on Thor | ⬜ still pending — three identical ticks are a latency measurement, not a reliability eval. |

### Upstream status

* **vLLM [#48291](https://github.com/vllm-project/vllm/pull/48291)**: native Edge reasoner; shipped in vLLM 0.28.0, which the sidecar lock pins on every platform.
* **transformers**: the sidecar still installs the SHA-pinned overlay (`_TRANSFORMERS_EDGE_SHA`); drop it, and add `transformers>=<version>` to `cosmos3_reasoner.in`, once a transformers release carries `cosmos3_edge`.
* `materialize_reasoner_view` only serves a vLLM without the native model; retire it once no supported platform needs it.

Still owed: the tool-call reliability eval against the deploy-sim baseline.

## Layer-boundary notes

This integration touches **Layer 4 (Reasoning) only** — a new
`ToolUseClient` backend behind the existing Protocol; no layer boundary
moved. Two adjacent opportunities are intentionally *not* part of it:

- **WAM (OpenRAL Pro, not a public layer):** Cosmos 3's generator tower is an action-conditioned
  world model — a mental-simulation backend. WAM implementations live in
  the private OpenRAL Pro monorepo (ADR-0104), not here. Separate decision
  + PR.
- **S1 policy rSkill:** `nvidia/Cosmos3-Edge-Policy-DROID` emits action
  chunks for DROID-format single-arm embodiments and could be packaged as
  an rSkill via the existing VLA adapter machinery. Separate decision + PR.
