---
tags:
  - OpenRAL
  - rskill
  - vlm
  - video-language-model
  - scene-understanding
  - spatial-reasoning
  - qwen
  - nf4
  - bitsandbytes
license: apache-2.0
language:
  - en
base_model:
  - Qwen/Qwen3.5-4B
---

# rskill-qwen35-4b-nf4

> **OpenRAL rSkill** — Qwen3.5-4B natively-multimodal video-language model
> packaged as an NF4 bitsandbytes `vlm` rSkill (ADR-0047). Accepts RGB
> image or video frames plus a natural-language query; returns a text answer.
> **No actuators.** Apache-2.0.

## Quick Start

```bash
ral skill install hf://OpenRAL/rskill-qwen35-4b-nf4
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

This rSkill declares `kind: vlm` and `role: s2` because it is a pure
perception component operating at S2 (slow-reasoning) rate (~0.2–1 Hz), not
an S1 fast policy. It consumes camera frames and natural-language queries,
emits text answers, and **never drives `ros2_control` joints**.

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
| VRAM at NF4 | **~2.5 GB** | ~3.3 GB |
| VRAM at BF16 | **~8 GB** | ~13 GB |
| Architecture | Hybrid linear-attn (3:1) | Full quadratic ViT+LLM |
| License | Apache-2.0 | Apache-2.0 |

Qwen3.5-4B beats Qwen2.5-VL-7B on every video benchmark despite being 3B
smaller. The 3:1 Gated DeltaNet / full-attention hybrid processes long video
sequences far more efficiently — important for continuous robot camera streams.
At NF4 it fits well within 8 GB VRAM alongside the S1 skill stack.

## Architecture

Qwen3.5 uses a 3:1 hybrid attention stack: three Gated DeltaNet
(linear-attention, O(n)) layers for every one full-attention layer. This
reduces cost on long sequences significantly. The vision encoder is shared with
Qwen3-VL. Key features:

- **Native video support** — temporal patch embedding, second-level event
  localization, up to 256K context (extensible to 1M)
- **Spatial grounding** — RefCOCO avg ~80.6; strong for "where is X?" queries
- **201-language support**

## Runtime

This rSkill ships a **pre-quantized NF4 checkpoint** as `weights_uri`
(`hf://OpenRAL/rskill-qwen35-4b-nf4`): `model.safetensors` with an embedded
bitsandbytes `quantization_config` (nf4, double-quant, bf16 compute). The
sidecar loads it **directly as 4-bit** (~3.3 GB resident, no bf16 load spike),
so it fits an 8 GB GPU with no loader workaround. `source_repo` records the
SHA-pinned upstream Apache-2.0 model it was quantized from (provenance, §8).

Reproduce the checkpoint with `tools/build_qwen_vlm_nf4_checkpoint.py` (run in
the sidecar venv):

```bash
$OPENRAL_QWEN_VLM_SIDECAR_VENV/bin/python tools/build_qwen_vlm_nf4_checkpoint.py \
  --source Qwen/Qwen3.5-4B \
  --out ~/.cache/openral/qwen35-4b-nf4-ckpt
```

It loads the upstream model once (forcing serial materialization so the bf16
pass fits 8 GB), saves the NF4 weights + processor, then verifies the checkpoint
reloads directly as 4-bit and answers a smoke query.

The `kind: vlm` runtime is implemented (ADR-0047) as a read-only reasoner tool,
**not** an `ExecuteSkill` (a scene VLM produces text, not actions):

- **Sidecar**: `tools/qwen_vlm_sidecar.py` boots the NF4 model in its own venv
  and serves a ZMQ REQ/REP + msgpack protocol. Provision it separately and
  point at it with `OPENRAL_QWEN_VLM_SIDECAR_VENV` (or let the backend
  auto-spawn it on first query).
- **Backend**: `openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`
  is the node-side ZMQ client; `build_scene_vlm(manifest)` builds it from this
  manifest. The node-side client deps (pyzmq + msgpack) install with
  `uv sync --group qwen-vlm`.
- **Service node**: `openral_perception_ros.scene_vlm_node` subscribes the
  cameras and serves `/openral/perception/query_scene`
  (`openral_msgs/srv/QueryScene`).
- **Reasoner tool**: the LLM sees the read-only `query_scene` tool when the
  reasoner is launched with `scene_query_available:=true`. It asks open-ended
  scene-state questions ("has the robot grasped the mug?", "is the task
  complete?") and the answer feeds the next reasoning tick.

### Validated live

The sidecar + backend + `query_scene` path was run end-to-end on an **NVIDIA
RTX 4070 Laptop (8 GB)**: NF4 Qwen3.5-4B loads to **~3.3 GB resident**, and real
image queries return correct answers — including the task-verification use case
("Has a robot gripper grasped any object?" → "No", grounded in the frame).
Covered by the GPU-gated `tests/unit/test_qwen_scene_vlm.py::test_e2e_query_coco_sample`
(set `OPENRAL_QWEN_VLM_SIDECAR_VENV`), not asserted blind.

**8 GB load note.** Deploying the **pre-quantized** `weights_uri` loads the
4-bit weights directly (~3.3 GB, ~6 s) with no workaround — the clean 8 GB path.
The workaround only matters when *quantizing at load* from the raw upstream
(the build step, or `--model Qwen/Qwen3.5-4B`): transformers 5.x's parallel
loader materializes weights in bf16 on-GPU *before* bitsandbytes quantizes, and
the 4-way-concurrent ~7.4 GB transient OOMs an 8 GB card, so the sidecar forces
**serial materialization** (`core_model_loading.GLOBAL_WORKERS = 1`) +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The sidecar auto-detects
which path applies. The Gated-DeltaNet fast kernels (`fla` / `causal-conv1d`)
are optional — without them transformers uses a slower torch fallback (the model
still loads and answers). The model loads via `AutoModelForImageTextToText` (it
registers as `Qwen3_5ForConditionalGeneration`).

## Benchmark Numbers

Benchmarks below are paper-reported (Qwen team, February 2026);
`reproduced_locally: false` in the eval JSON.

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

This scene VLM is **embodiment-agnostic** — it reasons about camera frames and
emits text, never actuator commands, so it imposes no kinematic requirement.
The only hardware dependency is an RGB camera stream of at least 336×336. All
in-tree OpenRAL embodiment tags are therefore listed in `rskill.yaml` (`aloha`,
`franka_panda`, `g1`, `google_robot`, `gr1`, `h1`, `mobile_base`, `openarm`,
`panda_mobile`, `pusht`, `rizon4`, `sawyer`, `so100_follower`, `so101_follower`,
`ur10e`, `ur5e`, `widowx`) so any robot with a compatible camera can install it
and expose the reasoner's `query_scene` tool. It pairs with any S1 VLA policy:
the VLA acts, this VLM verifies (e.g. "did the grasp succeed?").

## Sensors and Observation Contract

| Direction | Key | Modality | Shape / format | Notes |
|---|---|---|---|---|
| in | any RGB camera | RGB image or video | min 336 × 336 | `vla_feature_key` intentionally omitted |
| in | query | text | natural language | scene question, grounding query, or task-completion check |
| out | answer | text | free-form | grounded text response; adapter parses to `SceneQueryResult` |

The model emits no action chunks and has no proprioception contract.

## Manifest Summary

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-qwen35-4b-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s2` / `vlm` |
| `runtime` | `pytorch` |
| `quantization.dtype` | `int4` |
| `quantization.extra.scheme` | `nf4` |
| `weights_uri` | `hf://OpenRAL/rskill-qwen35-4b-nf4` (pre-quantized NF4) |
| `min_vram_gb.bf16` | 8.0 GB |
| `min_vram_gb.int4` | 2.5 GB |
| `latency_budget.per_chunk_ms` | 3000 ms |
| `actions` | `query` |

## License

The rSkill package metadata and README are OpenRAL project files under
Apache-2.0. The wrapped Qwen3.5 weights are released by the Qwen Team under
**Apache-2.0**, permitting commercial use. No `OPENRAL_ALLOW_NONCOMMERCIAL=1`
flag is needed.
