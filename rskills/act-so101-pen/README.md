---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- act
- lerobot
- vision-language-action
- so101_follower
datasets:
- gabrycina/so101-passing-pen
inference: false
---

# rskill-act-so101-pen-fp32

> **OpenRAL rSkill** — [ACT](https://arxiv.org/abs/2304.13705) (Action Chunking
> with Transformers) finetuned for **"pass the pen"** pick-and-place on a **real
> SO-101 follower arm**, packaged for `OpenRAL`.

This package wraps
[`gabrycina/so101-passing-pen-policy`](https://huggingface.co/gabrycina/so101-passing-pen-policy)
with a `rskill.yaml` manifest that adds capability checking, license surfacing,
latency budgets, the joint-units contract, a paired reward monitor, and local
registry integration. It does **not** copy the model weights.

It is the smaller, faster, ONNX/TensorRT-friendly sibling of
[`smolvla-so101-pen`](../smolvla-so101-pen): same task and embodiment, but a
plain CNN+transformer (ResNet-18 + VAE, ~52 M params) instead of a
VLM+flow-matching policy — so the **whole model exports to a single ONNX graph**
(see *ONNX / TensorRT* below).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/act-so101-pen/rskill.yaml")
```

```bash
# Real SO-101 deploy (torch baseline):
uv run openral deploy run --robot so101 --rskill rskills/act-so101-pen

# With the ONNX/TensorRT inference path (engine built + cached on first load):
OPENRAL_ACT_TRT=1 uv run openral deploy run --robot so101 --rskill rskills/act-so101-pen
```

## Upstream model / training

| Field | Value |
| --- | --- |
| Source repo | [`gabrycina/so101-passing-pen-policy`](https://huggingface.co/gabrycina/so101-passing-pen-policy) |
| Paper | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) — *Action Chunking with Transformers* |
| Training dataset | [`gabrycina/so101-passing-pen`](https://huggingface.co/datasets/gabrycina/so101-passing-pen) (~22.4k frames) |
| Architecture | ACT — ResNet-18 backbone, 4 encoder + 1 decoder layers, latent VAE, `chunk_size=100` |
| Precision | fp32 (torch); optional bf16/fp32 TensorRT engine |
| License | Apache-2.0 (code + weights) |

## Supported robots / embodiments

`so101_follower` — the 6-DoF SO-101 follower arm (5 arm joints + 1 gripper).
The checkpoint drives absolute joint positions.

> **Joint units — degrees.** This checkpoint's state and action are in **degrees**
> (verified: `observation.state` / `action` normalizer stats span ±100, MEAN_STD).
> openral's `JointState` / `Action` contract is radians, so the skill_runner
> converts deg↔rad at the policy boundary. The manifest declares
> `action_contract.joint_units: degrees` explicitly — a wrong guess would drive
> the arm into its joint limits.

## Sensors / observation contract

Two RGB streams, aliased to the checkpoint's `observation.images.*` inputs:

| Manifest key | Robot sensor | Checkpoint view |
| --- | --- | --- |
| `observation.images.camera1` | `top` | front / overview |
| `observation.images.camera2` | `wrist` | wrist |

Both views were **trained at 640×480** (checkpoint `config.json`:
`observation.images.front` and `observation.images.wrist` are both
`[3, 480, 640]`). ACT does **not** resize — lerobot's `modeling_act.py` has no
resize step and this checkpoint's `policy_preprocessor.json` is
rename→batch→device→normalize only (no resize stage), so the ResNet-18
backbone sees each frame at the sensor's native resolution. The manifest
minima reflect that: **both** `camera1` and `camera2` require the full
640×480. The `top` OAK-D Pro and the `wrist` icSpring rig each deliver 640×480
(confirmed with `v4l2-ctl --get-fmt-video`; `scenes/deploy/so101_bench.yaml`
binds both at that size). `camera2` previously declared only 256×256 because
`robots/so101_follower/robot.yaml` carried a stale 256×256 wrist intrinsics
block that a higher floor would have failed the sensor gate against; that
manifest now records the measured resolution.
Proprioception is the 6-D joint-position vector.

**In-distribution requirement (important).** ACT is a from-scratch imitation
policy (ResNet-18, no language grounding) and is overfit to its exact training
rig — it keys off camera geometry, scale, lighting, and background, not just
"a pen on a bench". A visually *similar* setup is not enough: on frames that
match the training distribution the engine reproduces the recorded expert
actions to ~1° across the whole pick, but on a different physical rig (other
camera extrinsics / FOV / lighting) it emits erratic actions and the arm
lunges. Two real training frames are checked in under
[`reference_frames/`](reference_frames/) (`train_ep0f0_front.png`,
`train_ep0f0_wrist.png`, episode 0 frame 0; recorded action
`[2.7, -99.7, 96.9, 60.1, 3.9, 0.5]` in joint degrees) as the in-distribution
reference — match your camera framing to these before expecting good behavior,
or collect a short teleop set on your own rig and fine-tune.

## Reward monitor

This VLA emits no success signal of its own, so it runs paired
with a reward / progress monitor: `reward_rskill_name:
OpenRAL/rskill-robometer_4b-any-general-nf4` (Robometer-4B, NF4). Robometer's **measured**
resident footprint is ~5.5 GB (weights + CUDA context + VLM-scoring
activations), not the ~3.6 GB packed-weight size — its manifest `min_vram_gb`
declares the measured value so the co-residency preflight budgets it honestly. ACT
(fp32, ~0.5 GB) + Robometer fit an 8 GB card for the host-path VLA; the
device-resident NVMM path plus dual-camera DeepStream buffers, however, is a
tight fit alongside Robometer on 8 GB — run reward-off for actuation there, or
use a larger GPU.

## ONNX / TensorRT

Unlike the SmolVLA sibling (whose VLM+flow-matching graph needs a bespoke
*split* export — vision encoder + unrolled flow loop, see PR #139), ACT is a
plain CNN+transformer and exports **whole-model** to one ONNX graph:

- **Export:** `tools/export_act_onnx.py` traces `ACTPolicy.select_action`
  (inputs: two RGB images + the 6-D state; output: the action chunk) to a
  single `model.onnx`.
- **Ship:** `model.onnx` is committed into this rSkill's HF repo
  (`policy_extras.act_onnx_uri`).
- **Run:** with `OPENRAL_ACT_TRT=1` and the private `openral-pro-trt` package
  installed, the ACT adapter loads `model.onnx` through the
  TensorRT backend, which builds and **caches** the engine on the host on
  first load (same delivery shape as `rtdetr-v2-r50vd`, also an OpenRAL Pro
  rSkill). Without `openral-pro-trt` the torch path runs unchanged — the
  flag is a no-op, logged, not a silent skip.

## Manifest summary

| Field | Value |
| --- | --- |
| `model_family` | `act` |
| `role` | `s1` |
| `chunk_size` | 100 (per-step replay; `temporal_ensemble_coeff=null`) |
| `action_contract` | 6-D `joint_positions`, `joint_units: degrees` |
| `reward_rskill_name` | `OpenRAL/rskill-robometer_4b-any-general-nf4` |
| `latency_budget` | 100 ms/chunk |
| Actions | pick · place · pick_and_place · transfer (object: pen) |

## License

**Apache-2.0** (code and weights). OpenRAL's packaging is Apache-2.0;
the upstream checkpoint and dataset are Apache-2.0 as published by the author.
