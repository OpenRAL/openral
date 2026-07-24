---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- lingbot_va_a1
- vision-language-action
- galaxea_a1
base_model:
- robbyant/lingbot-va-base
base_model_relation: finetune
datasets:
- pengyue-polaron/nyush-galaxea-a1-fruit-placement-eef-v21
inference: false
---

# rskill-lingbot_va-galaxea_a1-fruit_placement-bf16

> **OpenRAL rSkill** — a LingBot-VA fruit-placement policy for the Galaxea A1,
> deployed through OpenRAL's observation, typed-action, safety-kernel, and HAL
> contracts.

This package points to the public checkpoint at
[`pengyue-polaron/lingbot-va-galaxea-a1-fruit-placement-eef`](https://huggingface.co/pengyue-polaron/lingbot-va-galaxea-a1-fruit-placement-eef)
and does not copy model weights into the OpenRAL repository.

## Preview

![Galaxea A1 fruit-placement scene](https://huggingface.co/pengyue-polaron/lingbot-va-galaxea-a1-fruit-placement-eef/resolve/90e017bdbc6afac2e441b4634c9192776bbcb8b7/assets/fruit_placement_agent_view_labeled.png)

## What this skill does

The policy picks fruit, including a mango, from a tabletop and places it into a
bowl or plate. It consumes synchronized front and wrist RGB observations and
predicts episode-relative end-effector pose plus a continuous normalized gripper
command.

| Field | Value |
| --- | --- |
| Actions | `pick`, `place` |
| Objects | `fruit`, `mango`, `bowl`, `plate` |
| Scene | `tabletop` |
| Embodiment | `galaxea_a1` |

## Upstream model and training

The checkpoint is a full-parameter fine-tune of
[`robbyant/lingbot-va-base`](https://huggingface.co/robbyant/lingbot-va-base).
It jointly predicts video latents and robot-action channels. Training used 130
episodes and 44,824 frames at 30 FPS from the revision-pinned
[`nyush-galaxea-a1-fruit-placement-eef-v21`](https://huggingface.co/datasets/pengyue-polaron/nyush-galaxea-a1-fruit-placement-eef-v21)
dataset. The run used 1,000 optimizer steps, two NVIDIA H100 80 GB GPUs,
full-parameter FSDP, bfloat16, and an effective global batch size of 16.

The model emits 16 EEF/gripper steps per chunk. Quantile normalization and the
action-channel map `[0, 1, 2, 3, 4, 5, 6, 28]` are applied in the external
LingBot server from the checkpoint's `configs/va_a1_cfg.py`; this is not a
LeRobot `PolicyProcessorPipeline`.

OpenRAL's `lingbot_va_a1` adapter validates each physical EEF target with the A1
Runtime contract, solves IK, and emits six absolute joint targets plus one
normalized gripper target. If an IK solution is farther than the robot
manifest's per-tick joint-step limit, the adapter explicitly subdivides it
before advancing the model action or KV cache. The typed action then follows
OpenRAL's normal candidate-action, C++ safety-kernel, safe-action, and Galaxea A1
HAL path.

## Sensors and observation contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.front` | `(480, 480, 3)` RGB uint8 | Cropped D455 front view from the A1 Runtime Camera Bridge |
| in | `observation.images.wrist` | `(480, 640, 3)` RGB uint8 | D405 wrist view from the same paired Camera Bridge |
| in | `observation.state` | `(6,)` float32 | Six A1 arm joints in radians |
| out | action | `(7,)` float32 | Six absolute joint targets in radians and one normalized gripper target |

The A1 Runtime remains the sole camera-device owner. OpenRAL connects to its
public paired Camera Bridge and does not open either RealSense device itself.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Galaxea A1, original arm | `galaxea_a1` | Hardware-in-the-loop integration | Joint and gripper OpenRAL HAL round trips have passed; the combined model-to-hardware task run is the remaining validation step |

This rSkill is specific to the six-joint Galaxea A1 contract in
`robots/galaxea_a1/robot.yaml`. It is not a generic Cartesian HAL and does not
enable the vendor AnyGrasp/AnyEffector path.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-lingbot_va-galaxea_a1-fruit_placement-bf16` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `model_family` | `lingbot_va_a1` |
| `embodiment_tags` | `galaxea_a1` |
| `runtime` / precision | `pytorch` / `bf16` |
| `weights_uri` | revision-pinned public LingBot-VA A1 checkpoint |
| `state_contract.dim` / `action_contract.dim` | `6` / `7` |
| `chunk_size` / `n_action_steps` | `16` / `8` |
| `latency_budget.per_chunk_ms` | `6000` |
| `commercial_use_allowed` | `true` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

The software-only compatibility check does not initialize ROS or hardware:

```bash
uv run --group lingbot openral rskill check \
  rskills/lingbot-va-galaxea-a1-fruit-placement/rskill.yaml \
  --robot robots/galaxea_a1/robot.yaml
```

For real deployment, follow the owner-separated startup sequence in
[`docs/methods/01-hal.md`](../../docs/methods/01-hal.md): start the A1 Runtime
camera owner and LingBot server, start the isolated OpenRAL ROS1 sidecar, then
run the OpenRAL deployment scene. Do not start the A1 Runtime joint execution
bridge at the same time.

## Evaluation

No formal task-success result is shipped yet. Offline validation has exercised
the real paired cameras, real model server, manifest-to-policy construction,
EEF validation and IK, joint-step subdivision, and OpenRAL typed joint/gripper
dispatch. The OpenRAL candidate-action-to-HAL path has also passed separate
real-hardware joint and gripper round trips. One combined live rollout is still
required before claiming end-to-end OpenRAL model deployment on the A1.

## License

This rSkill package and the revision-pinned model weights are Apache-2.0. The
model repository contains the authoritative `LICENSE.txt`; the OpenRAL package
references the weights and does not redistribute them.

## See also

- [`robots/galaxea_a1/robot.yaml`](../../robots/galaxea_a1/robot.yaml)
- [`scenes/deploy/galaxea_a1_bench.yaml`](../../scenes/deploy/galaxea_a1_bench.yaml)
- [`docs/methods/01-hal.md`](../../docs/methods/01-hal.md)
