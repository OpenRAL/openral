---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- smolvla
- lerobot
- vision-language-action
- franka_panda
- vla
- so100
- libero
- manipulation
datasets:
- HuggingFaceVLA/libero
inference: false
---

# rskill-smolvla-franka_panda-libero_spatial-bf16

> **OpenRAL rSkill** — SmolVLA (0.45 B) finetuned on the [LIBERO](https://libero-project.github.io/) benchmark, packaged for use with the [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) with a `rskill.yaml` manifest that adds capability checking, license surfacing, latency budgets, and local registry integration. It does **not** copy model weights. (`HuggingFaceVLA/smolvla_libero` was the earlier wrap; its config does not match the paper — see the `rskill.yaml` header.)

## Quick start

```python
from openral_rskill.loader import rSkill

# Install from HF Hub (downloads manifest + registers locally):
pkg = rSkill.from_pretrained("OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16")
# pkg.manifest.weights_uri → "hf://lerobot/smolvla_libero"

# Or load offline from a local clone:
pkg = rSkill.from_yaml("rskills/smolvla-libero/rskill.yaml")
```

Via CLI:

```bash
openral rskill install OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
```

## Upstream model

| Field | Value |
|---|---|
| Source repo | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arxiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA: Efficient Vision-Language-Action Model* |
| License | Apache-2.0 |
| Parameters | ~450 M |
| Benchmark | LIBERO (table-top manipulation, 4 suites × 10 tasks) |
| Training data | [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero) (manifest `dataset_uri`) |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
|---|---|---|---|
| Franka Panda (LIBERO sim) | `franka_panda` | ✓ validated | Native training embodiment |

To add a new robot: declare matching `vla_feature_key`s on its cameras in `robots/<robot_id>/robot.yaml`, add its tag to `embodiment_tags` in `rskill.yaml`, and open a PR.

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | CUDA GPU with ≥ 2 GiB VRAM | RTX 3060 / 4060 Ti |
| VRAM | `min_vram_gb.bf16: 1.2` (0.93 GB weights at load, RTX 4070) | ≥ 4 GiB, so the paired reward monitor co-resides |
| RAM | 4 GiB | 16 GiB |
| Storage | 2 GiB (weights) | SSD recommended |

The deploy also loads the paired reward monitor (`reward_rskill_name: OpenRAL/rskill-robometer_4b-any-general-nf4`), which declares its own `min_vram_gb`.

> **CPU fallback**: ~20× slower inference. Pass `--device cpu` to `openral sim run` / `openral benchmark run`.

## Sensors

| Key | Type | Min resolution | Description |
|---|---|---|---|
| `observation.images.camera1` | RGB camera | 224 × 224 | Overhead / agent view |
| `observation.images.camera2` | RGB camera | 224 × 224 | Wrist / end-effector view |
| `observation.state` | Proprioception | (8,) | `libero_eef8d` task-space state: eef position ‖ eef axis-angle ‖ gripper |

The runner matches each camera against the robot's `SensorSpec.vla_feature_key`. Images are rotated 180° (`image_preprocessing.flip_180: true`), as in LIBERO training. `openral sim run` supplies the 8-D state directly; `openral deploy sim` assembles it from TF + `JointState` (`eef_frame: panda_hand_tcp`, `world_frame: world`).

## Observation → action contract

| Direction | Shape | Notes |
|---|---|---|
| in | 2 × RGB + `(8,)` state + task string | see *Sensors* |
| out | `(7,)` per step | `representation: delta_ee_6d_plus_gripper` — 6-D OSC end-effector delta (`cartesian_delta_scale: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`) + gripper |

### Action chunking

| Field | Value |
|---|---|
| `chunk_size` | 50 |
| `n_action_steps` | 25 (half-chunk replan) |
| Flow matching steps | 10 |

## Optimizations

| Optimization | Command / config | VRAM impact | Latency impact |
|---|---|---|---|
| **bf16** (default) | `quantization.dtype: bf16` | −30% vs fp32 | −10–20% |
| **torch.compile** | `torch.compile(policy, mode="reduce-overhead")` | +0% | −15–25% (after warm-up) |
| **CPU-only** | `--device cpu` | n/a | ~20× slower |

## rSkill manifest summary

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py)

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s1` / `vla` |
| `model_family` | `smolvla` |
| `embodiment_tags` | `franka_panda` |
| `runtime` | `pytorch` |
| `quantization.dtype` | `bf16` |
| `min_vram_gb` | `bf16: 1.2` |
| `weights_uri` | `hf://lerobot/smolvla_libero` |
| `latency_budget.per_chunk_ms` | 150 ms |
| `latency_budget.max_execution_s` | 60 s |
| `reward_rskill_name` | `OpenRAL/rskill-robometer_4b-any-general-nf4` |
| `is_commercial_use_allowed` | `true` |

## Evaluation results

### Locally reproduced (`openral benchmark run`)

| Suite | Success | Episodes per task | JSON |
|---|:---:|:---:|---|
| LIBERO-Spatial | 0.90 | 1 | [`eval/libero_spatial.json`](eval/libero_spatial.json) |
| LIBERO-Object | 0.78 | 10 | [`eval/libero_object.json`](eval/libero_object.json) |
| LIBERO-Goal | 0.67 | 10 | [`eval/libero_goal.json`](eval/libero_goal.json) |
| LIBERO-Long (`libero_10`) | 0.59 | 10 | [`eval/libero_10.json`](eval/libero_10.json) |

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

## Local reproduction

```bash
git clone https://github.com/OpenRAL/openral && cd openral
CC=/usr/bin/gcc just sync --group libero

# One suite (writes rskills/smolvla-libero/eval/<suite>.json):
openral benchmark run --suite libero_spatial --rskill rskills/smolvla-libero

# Sim-tier tests (IO contract, rollout, peak VRAM; need CUDA + the HF cache):
uv run pytest tests/sim/test_franka_panda_smolvla_libero.py -v -m sim
```

## Schema compliance

```bash
uv run python -c "
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml('rskills/smolvla-libero/rskill.yaml')
print(pkg.manifest.model_dump_json(indent=2))
"
```

The manifest validates against `RSkillManifest` (Pydantic v2).

## Changelog

| Version | Date | Notes |
|---|---|---|
| 0.1.0 | 2026-05-05 | Initial packaging — manifest + README + paper eval numbers |

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/*.json`) is **Apache-2.0**.
The wrapped model weights ([`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero)) are also Apache-2.0 per the upstream repo.

Commercial use is allowed. `RSkillManifest.is_commercial_use_allowed` derives this from `license`.
