# NVIDIA Cosmos 3 Edge as the S2 reasoner

`cosmos3-edge` is a curated reasoner model that runs on the robot itself: `OPENRAL_REASONER_MODEL=cosmos3-edge`, served by `openral_reasoner.cosmos3.Cosmos3ToolUseClient` through a managed local vLLM sidecar. It is `measured=True` in `openral_core.REASONER_MODELS`, with `max_tokens_default=512` and `max_skill_tools=4`.

## What it is

[Cosmos 3](https://github.com/nvidia/cosmos) is NVIDIA's omnimodal world-model family. Weights: [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge), license **OpenMDW-1.1** (commercial use allowed, no `OPENRAL_ALLOW_NONCOMMERCIAL` guard). One Mixture-of-Transformers model with two towers:

- **Reasoner tower**: text, image and video in, text out. Trained for spatial grounding, physical-plausibility judgment, task planning and embodied decisions. The checkpoint's `config.json` allows 131,072 tokens (`max_position_embeddings`).
- **Generator tower**: video, audio and action-chunk out. The reasoner integration does not use it. Its LIBERO action-policy finetune runs separately as the rSkill [`cosmos3-edge-libero-bf16`](https://github.com/OpenRAL/openral/tree/master/rskills/cosmos3-edge-libero-bf16/) (in-process lerobot `cosmos3`; see [vla_compatibility.md](vla_compatibility.md)). A DROID policy checkpoint (`nvidia/Cosmos3-Edge-Policy-DROID`) exists but is not packaged. WAM implementations are OpenRAL Pro.

Family tiers are Edge 4B (on-device), Nano 16B and Super 64B. Only Edge is curated; other tiers need an explicit model id and endpoint.

## Why it fits the S2 slot

| S2 requirement | Cosmos 3 Edge |
|---|---|
| Typed tool calls via the provider tool-use API | vLLM's OpenAI-compatible API with `--enable-auto-tool-choice`; the existing `tool_choice="required"` path is reused |
| 0.2 Hz tick budget | 1.3-2 s warm per tool-call tick on Jetson AGX Thor and RTX 4070. The first tool call after boot takes about 64 s while xgrammar builds the grammar |
| `describe_image` completion gate | Native VLM, so the same local model adjudicates completion |
| On-robot, offline, no PII egress | Fully local, no per-token cost |

## Integration shape

- `Cosmos3ToolUseClient` subclasses `OpenAICompatibleToolUseClient` and adds a managed server lifecycle: probe `GET /v1/models`, spawn `tools/cosmos3_reasoner_sidecar.py` if the loopback endpoint is down, wait for readiness, and terminate only its own child.
- The sidecar provisions an isolated Python 3.12 venv from the hash-pinned lock `tools/sidecar_requirements/cosmos3_reasoner.lock` (vllm 0.28.0, torch 2.13.0, transformers 5.16.1, every platform) plus a SHA-pinned transformers overlay for `cosmos3_edge` (`_TRANSFORMERS_EDGE_SHA`). vLLM 0.28.0 carries the native Edge model and serves the snapshot as-is. A vLLM without it gets a flattened reasoner view (`materialize_reasoner_view`), because vLLM's Transformers fallback cannot run `cosmos3_edge`.
- It execs `vllm serve` on port 8901 with `--served-model-name nvidia/Cosmos3-Edge --enable-auto-tool-choice --tool-call-parser hermes --max-num-seqs 4 --limit-mm-per-prompt '{"video":0,"image":1}'` and `VLLM_USE_FLASHINFER_SAMPLER=0`. Window, memory and eager mode depend on the host (next section).
- Config: `OPENRAL_REASONER_{MODEL,ENDPOINT,API_KEY,TIMEOUT_S,FIRST_CALL_TIMEOUT_S,MAX_TOKENS}` plus the client's `OPENRAL_COSMOS3_AUTOSTART`, `_BOOT_TIMEOUT_S`, `_SIDECAR` and the sidecar's `OPENRAL_COSMOS3_GPU_MEM_UTIL`, `_MAX_MODEL_LEN`, `_KV_CACHE_BYTES`, `_ENFORCE_EAGER`, `_MAX_NUM_SEQS`, `_LIMIT_MM_PER_PROMPT`, `_SIDECAR_HOME`, `_SIDECAR_VENV`. Your own `vllm serve` or the Cosmos 3 Reasoner NIM container works through `OPENRAL_REASONER_ENDPOINT`.
- `openral doctor` has a `Cosmos 3` row that reports `info`, not `warn`, when the endpoint is down, because autostart is the normal cold state.

## Footprint and serving defaults

The reasoner tower is about 3B parameters, served in BF16. Real prompts are large: the system prompt is about 2.8K tokens and each skill tool adds roughly 1.8K, so a 4-skill palette is about 10-12K tokens and a 14-skill palette about 20-25K. The dGPU default window of 8192 therefore fits no real palette.

| Flag | dGPU | Jetson (`is_tegra_host()`) | Why |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.90 | 0.15 | On unified memory this is a fraction of the whole board, checked as free at start-up. 0.90 refuses to boot beside a resident VLA |
| `--kv-cache-memory-bytes` | unset | `4G` | Sizes the KV cache directly, so utilization only gates start-up |
| `--max-model-len` | 8192 | 32768 | Real palettes are 10-26K tokens |
| `--enforce-eager` | on | off | Eager mode exists for 8 GB cards; on Thor CUDA graphs cost nothing once the compile cache is warm |

Env vars and CLI flags override each default. **8 GB is the practical floor**: a 4-skill palette fits with `--max-model-len 12288 --gpu-memory-utilization 0.97 --kv-cache-dtype fp8` (about 6.75 GB resident), a 14-skill palette does not, and there is no room for a co-resident VLA. A card with 12 GB or more is the comfortable minimum. Jetson Thor (32 GB) and Orin have ample room.

FP8 and other quantized serving are OpenRAL Pro: the sidecar appends whatever a `cosmos3_serving` module registered in the `openral.pro_modules` entry-point group returns, and adds nothing without it.

`max_tokens_default=512` caps runaways inside the tool-call grammar, the only failure mode seen. Without it a bad tick runs to the client's 120 s timeout; with it, it stops in 9-19 s. The longest valid call was 503 tokens; raise the cap with `OPENRAL_REASONER_MAX_TOKENS`.

## Tool-calling reliability

A 4B model degrades as the number of near-identical skill tools grows, not with the token count. So the registry entry sets `max_skill_tools=4`: each tick `ReasonerCore.offered_palette` ranks the palette's skills against the active task and pending operator prompts and offers at most four. The slots go first to the skill in flight, then the best match for the newest operator prompt, then the task's last dispatched skill, then the best navigate / look-at skill, then the highest scores. The subset changes only when the task does, so vLLM's prefix cache holds across ticks. Offered ids are on the `reasoner.offered_skills` span attribute. `OPENRAL_REASONER_MAX_SKILL_TOOLS` overrides it (`0` offers every skill).

Measured on Jetson AGX Thor (vLLM 0.28.0, BF16, Jetson defaults, 20 ticks each, `franka_panda` with every query tool, mission "Put the red cube in the blue bowl."):

| Skills offered | Prompt tokens | Valid ticks | Median warm tick |
|---|---|---|---|
| 14 (full palette) | 25,322 | 2/20 | 13.4 s |
| 12 | 22,655 | 0/20 | 12.8 s |
| 8 | 17,624 | 16/20 | 1.9 s |
| **4 (default)** | 12,470 | **19/20** | 1.7 s |

The tool schemas sent to the model carry no Pydantic class docstrings or titles; each field's meaning comes from a short `Field(description=...)` in `openral_core`, and a unit test fails any reasoner tool field without one.

## Open questions

- **Tool-call parser.** The sidecar uses vLLM's `hermes` parser, since the family documents Qwen3-VL-compatible messages. Edge returned validated calls through it on Thor and x86. Override `--tool-call-parser` if it diverges.
- **Transformers overlay.** The workspace and the sidecar lock use released transformers (5.22.2 and 5.16.1), but the sidecar still installs the SHA-pinned `main` overlay for `cosmos3_edge`. Whether a release now recognises the architecture is not re-verified; drop the overlay and add `transformers>=<version>` to `cosmos3_reasoner.in` once one does.
- **Reasoning traces.** Cosmos 3 supports explicit `<think>` output, which would eat the call timeout on small GPUs. The reasoner does not request it; cap it with `OPENRAL_REASONER_MAX_TOKENS` if it appears.
- **Benchmarks.** NVIDIA published family-level results only; treat quality claims as family-level.

## Layer boundary

This touches Layer 4 (Reasoning) only: a new `ToolUseClient` backend behind the existing Protocol.
