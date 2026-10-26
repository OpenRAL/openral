---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- nf4
- 4-bit
- any
- vlm
- video-language-model
- scene-understanding
- spatial-reasoning
- qwen
- bitsandbytes
base_model:
- Qwen/Qwen3.5-4B
base_model_relation: quantized
inference: false
---

# rskill-qwen35_4b-any-general-nf4

> **Requires OpenRAL Pro.** This rSkill runs at `int4 (NF4)` (sub-16-bit weights). Under ADR-0123, open-source OpenRAL loads only fp32, fp16 and bf16 weights; quantized formats (int8, fp8, int4/NF4, nvfp4) need the OpenRAL Pro quantization package.

> **OpenRAL rSkill** — Qwen3.5-4B natively-multimodal video-language model
> packaged as an NF4 bitsandbytes `vlm` rSkill. Accepts RGB
> image or video frames plus a natural-language query; returns a text answer.
> **No actuators.** Apache-2.0.

## Quick Start

```bash
openral rskill install OpenRAL/rskill-qwen35_4b-any-general-nf4
```

```python
from openral_core.schemas import RSkillManifest

manifest = RSkillManifest.from_yaml("rskills/qwen35-4b-nf4/rskill.yaml")
assert manifest.kind == "vlm"
assert manifest.role == "s2"
assert manifest.quantization.extra["scheme"] == "nf4"
assert manifest.is_commercial_use_allowed is True
```

## What It Does

Qwen3.5-4B is a natively-multimodal foundation model trained from scratch on
interleaved text, image, and video tokens. Given an RGB image or video clip and
a natural-language question, it returns a free-form text answer grounded in the
visual content.

It is `kind: vlm`, `role: s2`: a pure perception component at S2
(slow-reasoning) rate (~0.2–1 Hz), not an S1 fast policy. It emits text answers
and **never drives `ros2_control` joints**.

Representative queries for robot scene understanding:

- *"What objects are on the table?"*
- *"Is the gripper clear of obstacles?"*
- *"Describe the relative positions of the cup and bowl."*
- *"Has the pick-and-place task completed?"*

## Why Qwen3.5-4B over Qwen2.5-VL-7B

| | Qwen3.5-4B (this skill) | Qwen2.5-VL-7B |
|---|---|---|
| Parameters | 4B | 7B |
| VideoMME (w/ subs.) | **83.5%** | ~72% |
| MLVU | **82.8%** | ~73% |
| VRAM at NF4 | **2.5 GB** declared (`min_vram_gb.int4`), ~3.3 GB measured resident | — |
| VRAM at BF16 | **~8 GB** | ~13 GB |
| Architecture | Hybrid linear-attn (3:1) | Full quadratic ViT+LLM |
| License | Apache-2.0 | Apache-2.0 |

Qwen3.5-4B beats Qwen2.5-VL-7B on every video benchmark despite being 3B
smaller, and its hybrid attention handles long video sequences (continuous robot
camera streams) far more efficiently. At NF4 it fits within 8 GB VRAM alongside
the S1 skill stack.

## Architecture

Qwen3.5 uses a 3:1 hybrid attention stack: three Gated DeltaNet
(linear-attention, O(n)) layers per full-attention layer, cutting cost on long
sequences. The vision encoder is shared with Qwen3-VL. Key features:

- **Native video support** — temporal patch embedding, second-level event
  localization, up to 256K context (extensible to 1M)
- **Spatial grounding** — RefCOCO avg ~80.6; strong for "where is X?" queries
- **201-language support**

## Runtime

This rSkill ships a **pre-quantized NF4 checkpoint** as `weights_uri`
(`hf://OpenRAL/rskill-qwen35_4b-any-general-nf4`): `model.safetensors` with an embedded
bitsandbytes `quantization_config` (nf4, double-quant, bf16 compute). The
sidecar loads it **directly as 4-bit** (~3.3 GB resident, no bf16 load spike),
so it fits an 8 GB GPU with no loader workaround. `source_repo` records the
SHA-pinned upstream Apache-2.0 model it was quantized from (provenance, CLAUDE.md §1.8).

Reproduce the checkpoint with `tools/build_qwen_vlm_nf4_checkpoint.py` (run in
the sidecar venv):

```bash
$OPENRAL_QWEN_VLM_SIDECAR_VENV/bin/python tools/build_qwen_vlm_nf4_checkpoint.py \
  --source Qwen/Qwen3.5-4B \
  --out ~/.cache/openral/qwen35-4b-nf4-ckpt
```

It loads the upstream model once (serial materialization so the bf16 pass fits
8 GB), saves the NF4 weights + processor, and verifies the checkpoint reloads as
4-bit and answers a smoke query.

The `kind: vlm` runtime is implemented as a read-only reasoner tool,
**not** an `ExecuteRskill` (a scene VLM produces text, not actions):

- **Sidecar**: `tools/qwen_vlm_sidecar.py` boots the NF4 model in its own venv
  and serves a ZMQ REQ/REP + msgpack protocol. Provision it separately and
  point at it with `OPENRAL_QWEN_VLM_SIDECAR_VENV` (or let the backend
  auto-spawn it on first query).
- **Backend**: `openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`
  is the node-side ZMQ client; `build_scene_vlm(manifest)` builds it from this
  manifest. The node-side client deps (pyzmq + msgpack) install with
  `just sync --group qwen-vlm`.
- **Service node**: `openral_perception_ros.scene_vlm_node` subscribes the
  cameras and serves `/openral/perception/query_scene`
  (`openral_msgs/srv/QueryScene`).
- **Reasoner tool**: the LLM sees the read-only `query_scene` tool when the
  reasoner is launched with `scene_query_available:=true`. It asks open-ended
  scene-state questions ("has the robot grasped the mug?", "is the task
  complete?") and the answer feeds the next reasoning tick.

### Validated live

The sidecar + backend + `query_scene` path ran end-to-end on an **NVIDIA RTX
4070 Laptop (8 GB)**: NF4 Qwen3.5-4B loads to **~3.3 GB resident**, and real
image queries return correct answers, including task verification ("Has a robot
gripper grasped any object?" → "No", grounded in the frame). Covered by the
GPU-gated `tests/unit/test_qwen_scene_vlm.py::test_e2e_query_coco_sample` (set
`OPENRAL_QWEN_VLM_SIDECAR_VENV`).

**8 GB load note.** The **pre-quantized** `weights_uri` loads 4-bit weights
directly (~3.3 GB, ~6 s) with no workaround. The workaround only matters when
*quantizing at load* from the raw upstream (the build step, or
`--model Qwen/Qwen3.5-4B`): transformers 5.x's parallel loader materializes
weights in bf16 on-GPU *before* bitsandbytes quantizes, and the 4-way-concurrent
~7.4 GB transient OOMs an 8 GB card. The sidecar then forces **serial
materialization** (`core_model_loading.GLOBAL_WORKERS = 1`) + the
expandable-segments allocator config (`PYTORCH_ALLOC_CONF` on this sidecar's
torch 2.9.1; `PYTORCH_CUDA_ALLOC_CONF` was renamed in 2.9); it auto-detects which
path applies and sets the right env var name. The Gated-DeltaNet fast kernels
(`fla` / `causal-conv1d`) are optional; without them transformers uses a slower
torch fallback. The model loads via `AutoModelForImageTextToText` (it registers
as `Qwen3_5ForConditionalGeneration`).

## Benchmark Numbers

Benchmarks below are paper-reported (Qwen team, February 2026); not
reproduced locally, and no `eval/*.json` ships with this rSkill.

| Benchmark | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|
| VideoMME (w/ subtitles) | 83.5% | 84.5% |
| VideoMME (w/o subtitles) | 76.9% | 78.4% |
| VideoMMMU | 74.1% | 78.9% |
| MLVU | 82.8% | 84.4% |
| MVBench | 71.2% | 74.4% |
| LVBench | 66.4% | 70.0% |
| MMMU | 77.6% | 78.4% |
| RefCOCO avg | 80.6% | 81.3% |
| LingoQA (driving / spatial) | 74.4% | 80.4% |

## Supported robots and embodiments

**Embodiment-agnostic**: it emits text, never actuator commands, so the only
hardware dependency is an RGB camera stream of at least 336×336.
`rskill.yaml` declares the wildcard `embodiment_tags: ["any"]`, so any robot
with a compatible camera can install it and expose the reasoner's `query_scene`
tool. It pairs with any S1 VLA policy:
the VLA acts, this VLM verifies (e.g. "did the grasp succeed?").

## Sensors and Observation Contract

| Direction | Key | Modality | Shape / format | Notes |
|---|---|---|---|---|
| in | any RGB camera | RGB image or video | min 336 × 336 | `vla_feature_key` intentionally omitted |
| in | query | text | natural language | scene question, grounding query, or task-completion check |
| out | answer | text | free-form | grounded text response; returned as `QueryScene.answer` |

The model emits no action chunks and has no proprioception contract.

## Manifest Summary

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-qwen35_4b-any-general-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s2` / `vlm` |
| `runtime` | `pytorch` |
| `quantization.dtype` | `int4` |
| `quantization.extra.scheme` | `nf4` |
| `weights_uri` | `hf://OpenRAL/rskill-qwen35_4b-any-general-nf4` (pre-quantized NF4) |
| `min_vram_gb.bf16` | 8.0 GB |
| `min_vram_gb.int4` | 2.5 GB |
| `latency_budget.per_chunk_ms` | 3000 ms |
| `actions` | `query` |

## License

The rSkill package metadata and README are OpenRAL project files under
Apache-2.0. The wrapped Qwen3.5 weights are released by the Qwen Team under
**Apache-2.0**, permitting commercial use. No `OPENRAL_ALLOW_NONCOMMERCIAL=1`
flag is needed.
