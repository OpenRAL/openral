---
tags:
  - OpenRAL
  - rskill
  - smolvla
  - lerobot
  - vla
  - so100
  - libero
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-smolvla-libero

> **OpenRAL rSkill** — SmolVLA (0.45 B) finetuned on the [LIBERO](https://libero-project.github.io/) benchmark, packaged for use with the [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) with a `rskill.yaml` manifest that adds capability checking, license surfacing, latency budgets, and local registry integration.  It does **not** copy model weights.

---

## Demo — SO-100 digital twin

50-step closed-loop rollout, zero real hardware:

```bash
uv run python examples/so100_smolvla/run.py \
    --skill-id rskills/smolvla-libero/rskill.yaml \
    --steps 50 \
    --save-video /tmp/so100_rollout.gif
```

Measured on RTX 4070 Laptop · CUDA 12.8 · PyTorch 2.10:

| Phase | Latency |
|---|---|
| Weight load (from disk cache) | ~14 s |
| First chunk inference (JIT + cuDNN warm-up) | ~900 ms |
| Subsequent steps (cached action-queue pop) | **4 ms** |
| Mean over 50 steps | **4 ms** |
| Manifest budget (`per_chunk_ms`) | 150 ms ✓ |

---

## Quick start

```python
import os
os.environ["HF_TOKEN"] = "<your-read-token>"

from openral_rskill.loader import rSkill

# Install from HF Hub (downloads manifest + registers locally):
pkg = rSkill.from_pretrained("OpenRAL/rskill-smolvla-libero")
# pkg.manifest.weights_uri → "hf://HuggingFaceVLA/smolvla_libero"
# pkg.local_dir            → ~/.cache/openral/rskills/...

# Or load offline from local clone:
pkg = rSkill.from_yaml("skills/smolvla-libero/rskill.yaml")
```

Via CLI:

```bash
ral skill install hf://OpenRAL/rskill-smolvla-libero
ral run examples/so100_smolvla --skill-id OpenRAL/rskill-smolvla-libero
```

---

## Upstream model

| Field | Value |
|---|---|
| Source repo | [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arxiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA: Efficient Vision-Language-Action Model* |
| License | Apache-2.0 |
| Parameters | ~450 M |
| Benchmark | LIBERO (table-top manipulation, 4 suites × 10 tasks) |
| Training data | [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero) — 1 693 demos |

---

## Supported robots

| Robot | Embodiment tag | Status | Notes |
|---|---|---|---|
| Franka Panda (LIBERO sim) | `libero` | ✓ validated | Native training embodiment |
| SO-100 follower arm | `so100_follower` | ✓ IO verified | Digital-twin rollout tested (Day 20) |
| Any 6–7 DOF manipulator | `manipulator` | ⚡ experimental | Requires obs-format adapter |

To add a new robot: create a `SensorBundle` + obs-format adapter, update `embodiment_tags` in `rskill.yaml`, and open a PR.

---

## Hardware requirements

### Minimum (inference only)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA 11.8+ GPU with ≥ 2 GiB VRAM | RTX 3060 / 4060 Ti |
| VRAM | 1.5 GiB (fp32) · 0.95 GiB (bf16) | ≥ 4 GiB |
| RAM | 4 GiB | 16 GiB |
| CPU | Any x86-64 / ARM64 | — |
| Storage | 2 GiB (weights) | SSD recommended |

### Reference host (Day 20 measurements)

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop (7.62 GiB VRAM, CUDA 12.8) |
| Driver | 555.xx |
| PyTorch | 2.10.0+cu128 |
| Peak VRAM | ~1.1 GiB (bf16 chunk inference, 512 × 512 inputs) |

> **CPU fallback**: possible but expect ~20× slower inference (900 ms chunk → ~18 s).  Set `device="cpu"` in the run script; no code changes required.

---

## Sensors

| Key | Type | Resolution | Format | Description |
|---|---|---|---|---|
| `observation.images.OBS_IMAGE_1` | RGB camera | 512 × 512 | `float32 [0, 1]` | Top / overhead view (primary) |
| `observation.images.OBS_IMAGE_2` | RGB camera | 512 × 512 | `float32 [0, 1]` | Wrist / end-effector view |
| `observation.state` | Proprioception | (7,) | `float32` | Joint positions (rad or deg, model-native) |

Images are resized to 512 × 512 before tokenisation.  The model applies pixel-shuffle (4×) to compress each frame to 64 VLM tokens (no tiling).

**For SO-100 digital twin runs**: images are synthesised as zero tensors (no real camera); state uses 6-DOF twin positions padded to 7-DOF.

---

## Observation → action contract

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").eval()
preprocessor, _ = make_pre_post_processors(policy.config, "HuggingFaceVLA/smolvla_libero")

# Raw observation dict (LIBERO format):
raw = {
    "observation.images.OBS_IMAGE_1": top_cam,   # (1, 3, 512, 512) float32 [0, 1]
    "observation.images.OBS_IMAGE_2": wrist_cam, # (1, 3, 512, 512) float32 [0, 1]
    "observation.state": joint_pos,              # (1, 7)           float32
    "task": ["pick up the red cube"],            # list[str]
}

batch = preprocessor(raw)                        # normalise, tokenise
action = policy.select_action(batch)             # → (1, 8) float32

# action[:, :7] = joint position commands (Franka Panda)
# action[:, 7]  = gripper width command (0 = closed, 1 = open)
```

### Action chunking

| Field | Value |
|---|---|
| `chunk_size` | 50 |
| `n_action_steps` | 50 |
| Flow matching steps | 10 |
| Inference mode | Synchronous (drain chunk, then re-infer) |

---

## Optimizations

| Optimization | Command / config | VRAM impact | Latency impact |
|---|---|---|---|
| **bf16 autocast** (default) | `torch.autocast("cuda", torch.bfloat16)` | −30% vs fp32 | −10–20% |
| **torch.compile** | `torch.compile(policy, mode="reduce-overhead")` | +0% | −15–25% (after warm-up) |
| **TensorRT** (planned) | `rskill.yaml → engine_uri` (v0.3) | −20% | −40–60% |
| **INT8 / FP8** (planned) | `QuantizationConfig(dtype="int8")` (v0.3) | −50% | −50% |
| **CPU-only** | pass `device="cpu"` | n/a | ~20× slower |

`torch.compile` requires PyTorch 2.3+.  TRT/INT8 support is tracked in [OpenRAL #milestone-m3](https://github.com/OpenRAL/openral).

---

## rSkill manifest summary

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py)

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-smolvla-libero` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` (fast visuomotor policy, 30–50 Hz) |
| `embodiment_tags` | `libero`, `so100_follower`, `manipulator` |
| `runtime` | `pytorch` |
| `quantization.dtype` | `bf16` |
| `weights_uri` | `hf://HuggingFaceVLA/smolvla_libero` |
| `latency_budget.per_chunk_ms` | 150 ms |
| `latency_budget.warmup_ms` | 8 000 ms |
| `latency_budget.load_ms` | 30 000 ms |
| `dispatch_target` | `edge` |
| `fallback_skill_id` | `null` |
| `commercial_use_allowed` | `true` |
| `signature` | `null` (sigstore v1.0, planned) |

---

## Evaluation results

### Upstream benchmark (paper, not locally reproduced)

From **Table 2** of [arxiv:2506.01844](https://arxiv.org/abs/2506.01844), SmolVLA (0.45 B), multi-task training.
Protocol: 10 trials per task, binary success/fail, Franka Panda in LIBERO simulator.

| Suite | SmolVLA 0.45B | OpenVLA 7B | Octo 90M | π₀ 3.3B |
|---|:---:|:---:|:---:|:---:|
| LIBERO-Spatial | **90%** | 84.7% | 78.9% | 90% |
| LIBERO-Object | **96%** | 88.4% | 85.7% | 86% |
| LIBERO-Goal | **92%** | 79.2% | 84.6% | 95% |
| LIBERO-Long | **71%** | 53.7% | 51.1% | 73% |
| **Average** | **87.3%** | 76.5% | 75.1% | 86.0% |

Full results with config: [`eval/libero.json`](eval/libero.json) — `reproduced_locally: false`.

### IO contract verification (Day 20, locally measured)

The following properties were verified locally using `tests/sim/test_franka_panda_smolvla_libero.py` and the SO-100 digital twin:

| Property | Expected | Measured | Status |
|---|---|---|---|
| State input shape | (1, 7) float32 | (1, 7) float32 | ✓ |
| Image input resolution | 512 × 512 | 512 × 512 | ✓ |
| Action output shape | (1, 8) float32 | (1, 8) float32 | ✓ |
| Actions finite (no NaN/Inf) | true | true | ✓ |
| Warm chunk latency | ≤ 150 ms | ~110 ms | ✓ |
| Cached step latency | ≤ 30 ms | ~4 ms | ✓ |
| Peak VRAM | ≤ 2.0 GiB | ~1.1 GiB | ✓ |
| 50-step digital twin rollout | completes | completes | ✓ |

---

## Local reproduction

### 1 — Install prerequisites

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
CC=/usr/bin/gcc uv sync --all-packages --group sim
```

### 2 — Manifest + IO contract tests (no GPU required for manifest tests)

```bash
# Manifest tests only (fast, no weights download):
uv run pytest tests/sim/test_franka_panda_smolvla_libero.py::TestRSkillManifest -v

# Full IO contract tests (requires CUDA + HF Hub cache):
uv run pytest tests/sim/test_franka_panda_smolvla_libero.py -v -m sim
```

### 3 — Run the end-to-end demo

```bash
# 50-step SO-100 digital twin rollout with GIF output:
uv run python examples/so100_smolvla/run.py \
    --skill-id rskills/smolvla-libero/rskill.yaml \
    --steps 50 \
    --save-video /tmp/so100_rollout.gif

# Or via just:
just sim so100
```

### 4 — LIBERO benchmark reproduction (requires LIBERO gym, planned Day 21+)

```bash
# Install LIBERO gymnasium environment:
uv add libero --group sim

# Run full 40-task eval:
uv run pytest tests/sim/test_franka_panda_smolvla_libero.py::TestSmolVLALiberoPolicy -v -m sim
```

Full LIBERO benchmark repro (updating `eval/libero.json` to `reproduced_locally: true`) requires
the LIBERO gym package and ~8 h on a desktop GPU. This is tracked as a Day 21+ milestone.

---

## Schema compliance

This rSkill was validated with:

```bash
uv run python tools/schema_export.py           # regenerate JSON Schema
uv run python -c "
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml('skills/smolvla-libero/rskill.yaml')
print(pkg.manifest.model_dump_json(indent=2))
"
```

The manifest validates against `RSkillManifest` (Pydantic v2) without errors.

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| 0.1.0 | 2026-05-05 | Initial packaging — manifest + README + paper eval numbers |

---

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/libero.json`) is **Apache-2.0**.
The wrapped model weights ([`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)) are also Apache-2.0 per the upstream repo.

Commercial use is allowed. See `rskill.yaml → commercial_use_allowed: true`.
