---
language:
- en
license: other
license_name: lfm1.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- nat_vla
- vision-language-action
- so101_follower
- rskill-kind:vla
- rskill-role:s1
- embodiment:so101_follower
- rskill-license:lfm1.0
datasets:
- makermods/eraser_place_unblurry_real
inference: false
openral:
  description: 'NaT VLA (450M, LiquidAI LFM2-VL-450M writing NAT action tokens) multi-robot
    pretrained checkpoint, run zero-shot on the SO-101 follower: 6-DoF absolute joint
    chunks (32, 16 replayed) in LeRobot servo degrees from a fixed overview and a
    wrist camera. Not fine-tuned on an OpenRAL task; LFM Open License v1.0 weights
    (commercial use only under $10M revenue).'
  model_family: nat_vla
  min_vram_gb: 1.5
  latency_ms: 500.0
  runtime: pytorch
  dtype: bf16
---

# rskill-nat_vla-so101-pretrained-bf16

> **OpenRAL rSkill** — NaT VLA's multi-robot **pretrained** checkpoint, run zero-shot
> on the SO-101 follower arm. Not fine-tuned on an OpenRAL task.

This package contains the OpenRAL manifest only. The weights are
[`ahad-j/nat-vla-450m`](https://huggingface.co/ahad-j/nat-vla-450m), pinned by commit;
the normalisation stats come from the SO-101 teleop dataset
[`makermods/eraser_place_unblurry_real`](https://huggingface.co/datasets/makermods/eraser_place_unblurry_real).

## Preview

![Zero-shot rollout in the SO-101 tube-insertion sim: start, steps 75 and 150, end](media/so101_box_zero_shot.png)

Zero-shot in the `so101_box` tube-insertion scene: the arm leaves its rest pose in the
first ~25 steps and settles short of the tube (0/1), the expected behaviour of a
pretrained model on an unseen task.

## What this skill does

A generalist vision-language-action policy for the SO-ARM it saw in pretraining (the
`single so100/so101 robotic arm in molmoact2` setup). It predicts 32-step chunks of the
6-DoF absolute joint command (5 arm joints + gripper, LeRobot servo degrees) from a fixed
overview camera, a wrist camera and a language instruction, and replays 16 steps per
chunk. Use it as the base for a fine-tune, or to probe zero-shot behaviour.

## How it works

NaT VLA is LiquidAI's LFM2-VL-450M writing actions as tokens of the NAT neural action
tokenizer (4 latents per 32-step chunk, 25,600 action tokens). The prompt names the task,
the robot setup, the binned state and the control mode; the policy generates action
tokens greedily and `select_action` streams each step as soon as its tokens are decoded.

`openral_sim.policies.nat_vla` builds the policy for this manifest: lerobot's `nat_vla`
when the installed lerobot ships it, otherwise the copy vendored at the lerobot merge
commit (lerobot 0.7.0 predates it).

### Observation → action contract

| In | Shape | Notes |
| --- | --- | --- |
| `observation.images.front` / `observation.images.wrist` | HWC uint8 RGB | slots `camera1` / `camera2` via `image_preprocessing.aliases` |
| `observation.state` | `(6,)` float32 | LeRobot servo degrees, gripper 0–100; quantile-normalised with the dataset's q01/q99 |
| `task` | str | the instruction |

Out: a 6-D absolute joint action per step in the same units (min/max-normalised in the tokenizer).

## Upstream model / training

| | |
| --- | --- |
| Model | NaT VLA 450M: LFM2-VL-450M (SigLIP2 + LFM2-350M) + the frozen 63M NAT tokenizer |
| Pretraining | 95,315 steps, batch 12, on DROID, Bridge, RT-1, BC-Z, Franka, bimanual YAM and SO-100/101 data |
| Paper | [NAT: Scaling Causal Neural Action Tokenizers Across Embodiments](https://ahadjawaid.com/nat-paper) |
| Code | lerobot `nat_vla` ([huggingface/lerobot#5416](https://github.com/huggingface/lerobot/pull/5416)), Apache-2.0 |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| SO-101 follower | `so101_follower` | sim (zero-shot) | LeRobot degrees; the `so101_box` scene's `joint_units: degrees` |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB (fixed overview) | 224 x 224 | HWC uint8 |
| `observation.images.camera2` | RGB (wrist) | 224 x 224 | HWC uint8 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-nat_vla-so101-pretrained-bf16` |
| `model_family` | `nat_vla` |
| `license` | `lfm1.0` (weights) |
| `quantization.dtype` | `bf16` |
| `min_vram_gb` | bf16 1.5 |
| `weights_uri` | `hf://ahad-j/nat-vla-450m@a5c2e13d45832050e364265962f6ad536460e955` |
| `dataset_uri` | `hf://makermods/eraser_place_unblurry_real` (normalisation stats) |
| `chunk_size` / `n_action_steps` | 32 / 16 |
| `state_contract.dim` / `action_contract.dim` | 6 / 6 |

## Quick start

```bash
just sync --group sim
MUJOCO_GL=egl openral sim run --config scenes/sim/so101_tube_insertion.yaml \
  --rskill rskills/nat-vla-so101-pretrained-bf16 --n-episodes 1 --max-steps 300
```

## Reproduction

The command above is the run recorded under [Evaluation](#evaluation).

## Evaluation

| Scene | Episodes | Success | Mean step latency | Host |
| --- | --- | --- | --- | --- |
| `so101_box/tube_insertion` (zero-shot) | 1 | 0/1 | 13.6 ms | RTX 5070 Laptop, bf16 |

A smoke check of the port, not a benchmark claim; `eval/` is empty until a fine-tune.

## License

The weights are a derivative of LiquidAI LFM2-VL-450M under the **LFM Open License
v1.0**: commercial use only for organizations under $10M annual revenue. OpenRAL treats
the posture (`lfm1.0`) as not commercial-allowed, so a commercial deploy needs
`OPENRAL_ALLOW_NONCOMMERCIAL=1`, which is logged. The vendored policy code is Apache-2.0.

## See also

- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
- [`rskill-smolvla-so101-eraser_place-bf16`](../rskill-smolvla-so101-eraser_place-bf16/) — a fine-tuned SO-101 policy on the same dataset.
