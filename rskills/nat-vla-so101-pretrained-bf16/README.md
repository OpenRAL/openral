---
language:
- en
license: other
license_name: lfm1.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- nat_vla
- lerobot
- vision-language-action
- so101_follower
- rskill-kind:vla
- rskill-role:s1
- embodiment:so101_follower
- rskill-license:lfm1.0
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
the normalisation stats are the pretraining's own pooled SO-ARM stats, the
`so100_so101_molmoact2` tag of
[`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101)'s
`norm_stats.json` (see [Joint convention](#joint-convention)).

## Preview

![Zero-shot rollout in the SO-101 tube-insertion sim: steps 0, 100, 200 and 300, overhead camera](media/so101_box_zero_shot.png)

Zero-shot in the `so101_box` tube-insertion scene (`so101_tube_insertion_lerobot.yaml`,
steps 0 / 100 / 200 / 300): the arm turns and settles short of the tube (0/1), the expected
behaviour of a pretrained model on an unseen task. See [Real SO-101](#real-so-101-zero-shot)
for the real arm.

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
| `observation.state` | `(6,)` float32 | LeRobot servo degrees, gripper 0–100; mapped to the pretraining's joint convention (`joint_affine`), then quantile-normalised with the pooled q01/q99 |
| `task` | str | the instruction |

Out: a 6-D absolute joint action per step in the same units: decoded with the pooled
action q01/q99 and mapped back to this robot's convention.

### Joint convention

NaT VLA was pretrained with MolmoAct2's data pipeline (NAT paper, arXiv 2611.28873,
Appendix F), which normalises the SO-ARM mixture by its pooled 1st/99th percentiles
(tag `so100_so101_molmoact2`, whose `setup_type` is this skill's setup string). That
mixture is mostly recorded in lerobot's **old** calibration convention: episodes start
at shoulder_lift 185–198° and elbow ~180°. A current lerobot calibration is
range-centred: the same rest pose reads about −103° / 95°, and reaching *raises*
shoulder_lift where the old convention lowers it. `policy_extras.joint_affine` maps
between them (`model = scale × robot + offset`): shoulder_lift `[-1, 87]`, elbow
`[1, 86]`, identity elsewhere. The offsets were fitted on episode start poses and
cross-checked on the pooled medians, to about ±5°. With a single new-convention
dataset's stats instead, the real arm swung backward and drove shoulder_lift into its
stop.

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
| `policy_extras.norm_stats` | `hf://allenai/MolmoAct2-SO100_101@152569fe…/norm_stats.json`, tag `so100_so101_molmoact2` |
| `policy_extras.joint_affine` | shoulder_lift `[-1, 87]`, elbow `[1, 86]`, identity elsewhere |
| `chunk_size` / `n_action_steps` | 32 / 16 |
| `state_contract.dim` / `action_contract.dim` | 6 / 6 |

## Quick start

Sim (the `so101_box` variant whose state is the real arm's joint convention; the plain
`so101_tube_insertion.yaml` applies MolmoAct2's offsets, which `joint_affine` would map a
second time):

```bash
just sync --group sim
MUJOCO_GL=egl openral sim run --config scenes/sim/so101_tube_insertion_lerobot.yaml \
  --rskill rskills/nat-vla-so101-pretrained-bf16 --n-episodes 1 --max-steps 300
```

Real SO-101 (the committed bench scene; pin the reasoner's palette to this skill and leave
the reward monitor off on an 8 GB card):

```bash
OPENRAL_RSKILLS=OpenRAL/rskill-nat_vla-so101-pretrained-bf16 \
  openral deploy run -c scenes/deploy/so101_bench.yaml --no-enable-reward-monitor
openral prompt --new-goal "pick up the yellow lego block and place it on top of the red lego block"
```

Start from the folded rest pose (shoulder_lift ≈ −100°, elbow ≈ 95°): every training
episode starts there. Motion is attended; keep the power switch within reach.

## Reproduction

The command above is the run recorded under [Evaluation](#evaluation).

## Evaluation

| Scene | Episodes | Success | Mean step latency | Host |
| --- | --- | --- | --- | --- |
| `so101_box/tube_insertion` (zero-shot, `so101_tube_insertion_lerobot.yaml`) | 1 | 0/1 | 7.6 ms | RTX 5070 Laptop, bf16 |

A smoke check of the port, not a benchmark claim; `eval/` is empty until a fine-tune.

### Real SO-101 (zero-shot)

Run through `openral deploy run` on a laptop SO-101 bench (built-in webcam as `top`,
icSpring wrist camera; RTX 5070 Laptop), prompt "pick up the yellow lego block and place it
on top of the red lego block", recorded with `openral record` (2026-11-24):

| Wiring | Motion | HAL stops |
| --- | --- | --- |
| One new-convention dataset's stats (`makermods/eraser_place`), no joint mapping | 4 s reach, then the shoulder swung back toward its stop and stalled | 2 shoulder_lift tracking-fault e-stops in 2 runs (commanded −100° against a −94° stop) |
| Pooled MolmoAct2 stats + `joint_affine` (this manifest) | ~75 s of smooth motion around the rest pose; shoulder/elbow follow the commands | none |

Offline, on a recorded bench frame from the rest pose, the old wiring's first command jumped
the elbow 44° and the shoulder 15°; this wiring's stays within 1–13° on both. Neither
completes the task: this is the pretrained checkpoint, and pan / wrist sat at the edge of the
pooled range, turning away from the blocks. A fine-tune on the bench's own demonstrations
is the way to task success.

## License

The weights are a derivative of LiquidAI LFM2-VL-450M under the **LFM Open License
v1.0**: commercial use only for organizations under $10M annual revenue. OpenRAL treats
the posture (`lfm1.0`) as not commercial-allowed, so a commercial deploy needs
`OPENRAL_ALLOW_NONCOMMERCIAL=1`, which is logged. The vendored policy code is Apache-2.0.

## See also

- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
- [`rskill-smolvla-so101-eraser_place-bf16`](../rskill-smolvla-so101-eraser_place-bf16/) — an SO-101 policy fine-tuned on `makermods/eraser_place_unblurry_real`.
