---
tags:
  - openral
  - rskill
  - openvla
  - openvla-oft
  - vla
  - widowx
  - simpler
  - maniskill3
  - manipulation
license: mit
language:
  - en
---

# rskill-openvla-oft-simpler-widowx-nf4

> OpenVLA-OFT bridge policy (RLinf, PPO-tuned on ManiSkill3 PutOnPlateInScene25),
> packaged for OpenRAL and evaluated on the SimplerEnv WidowX put-on-plate tasks.

## What this skill does

Wraps [`RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood)
— an [OpenVLA-OFT](https://openvla-oft.github.io/) (arXiv:2502.19645) policy,
RL-tuned with PPO on the ManiSkill3 `PutOnPlateInScene25` task using a WidowX
250 S — and runs it on the four canonical [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
WidowX (Bridge V2) put-on-plate tasks that share its embodiment, EE-delta
control, and `bridge_orig` normalization.

**Why WidowX and not Panda/PickCube:** this checkpoint is a *bridge* policy. The
ManiSkill3 Panda `PickCube-v1` scenes are a different embodiment and task; the
ADR-0060 task-data gate correctly refuses that pairing (it would produce a
plausible-but-unsolvable rollout). See [ADR-0061](../../docs/adr/0061-openvla-oft-policy-family.md)
for the full rationale.

## How it works

Loaded in-process by OpenRAL's `openvla` policy adapter
(`python/sim/src/openral_sim/policies/openvla.py`) as a transformers
*custom-code* model (`AutoModelForVision2Seq` + `trust_remote_code`, gated by
`OPENRAL_ALLOW_REMOTE_CODE=1`). NF4 (4-bit) quantization plus the CUDA
expandable-segments allocator bring the 7.5 B backbone within an 8 GB GPU.

### Observation → action contract

- **Input:** one 224×224 RGB frame (the SimplerEnv 3rd-view, surfaced as
  `camera1`) and the prompt
  `In: What action should the robot take to {instruction.lower()}?\nOut: `. No
  proprioception (`use_proprio=False`).
- **Output:** 256-bin discrete action tokens decoded to `[-1, 1]`, then
  de-normalized with the embedded `unnorm_key=bridge_orig` stats (BOUNDS_Q99):
  6 end-effector deltas (3 position + 3 rotation) rescaled, gripper passed
  through. Action chunk = 8 × 7-D, replayed open-loop.

## How it was trained

Upstream base `Haozhan72/Openvla-oft-SFT-libero-goal-trajall`, ManiSkill LoRA
SFT, then PPO on `PutOnPlateInScene25Main-v3` (WidowX 250 S). RLinf model-index
success: train 0.977; OOD vision 0.921 / semantic 0.648 / position 0.736. See
the upstream card for the full protocol.

## Supported robots

- `widowx` (WidowX 250 S, Bridge V2 flat-table setup).

## Sensors required

- One RGB stream, ≥224×224, mapped to `observation.images.camera1`.

## Manifest summary

See [`rskill.yaml`](./rskill.yaml). Key fields: `model_family: openvla`,
`license: mit`, `quantization.dtype: int4`, `chunk_size: 8`,
`evaluated_tasks` = the four SimplerEnv WidowX put-on-plate tasks,
`action_contract` = `delta_ee_6d_plus_gripper` (dim 7).

## Quick start

```bash
just sync --all-packages --group simpler-env
hf download RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood
OPENRAL_ALLOW_REMOTE_CODE=1 openral benchmark run \
  --suite simpler_env_widowx \
  --rskill openvla-oft-simpler-widowx-nf4
```

## Reproduction

```bash
# Single SimplerEnv WidowX scene (carrot-on-plate):
OPENRAL_ALLOW_REMOTE_CODE=1 openral benchmark run \
  --suite simpler_env_widowx --task simpler_env/widowx_carrot_on_plate \
  --rskill openvla-oft-simpler-widowx-nf4
```

## Evaluation

Per-task success rate on the SimplerEnv WidowX tasks (real-to-sim correlator —
feed into the MMRV / Pearson computation against the matching real-robot eval).
Verified success numbers are recorded in the OpenRAL benchmark tracking sheet
and the PR that introduced this skill (issue #55).

## License

MIT (upstream `RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood`). The OpenRAL
packaging is Apache-2.0. The checkpoint is a `trust_remote_code` custom-code
model; loading executes repo-shipped Python and requires
`OPENRAL_ALLOW_REMOTE_CODE=1` (provenance: rSkill signature verification is not
yet implemented — ADR-0006).

## See also

- [ADR-0061 — OpenVLA / OpenVLA-OFT policy family](../../docs/adr/0061-openvla-oft-policy-family.md)
- [ADR-0060 — benchmark task-data compatibility gate](../../docs/adr/0060-benchmark-task-data-compatibility-gate.md)
- [`rldx1-ft-simpler-widowx-nf4`](../rldx1-ft-simpler-widowx-nf4) — the sibling WidowX bridge rSkill.
