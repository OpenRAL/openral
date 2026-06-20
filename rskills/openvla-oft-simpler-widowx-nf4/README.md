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
> packaged for OpenRAL and locally verified on SimplerEnv WidowX carrot-on-plate.

## What this skill does

Wraps [`RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood)
— an [OpenVLA-OFT](https://openvla-oft.github.io/) (arXiv:2502.19645) policy,
RL-tuned with PPO on the ManiSkill3 `PutOnPlateInScene25` task using a WidowX
250 S — and runs it on the [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
WidowX (Bridge V2) carrot-on-plate task that shares its embodiment, EE-delta
control, and `bridge_orig` normalization. The sibling Bridge tasks are not
declared in `evaluated_tasks` until locally reproduced.

**Why WidowX and not Panda/PickCube:** this checkpoint is a *bridge* policy. The
ManiSkill3 Panda `PickCube-v1` scenes are a different embodiment and task; the
ADR-0060 task-data gate correctly refuses that pairing (it would produce a
plausible-but-unsolvable rollout). See [ADR-0063](../../docs/adr/0063-openvla-oft-policy-family.md)
for the full rationale.

## How it works

Loaded in-process by OpenRAL's `openvla` policy adapter
(`python/sim/src/openral_sim/policies/openvla.py`) as a transformers
*custom-code* model (`AutoModelForVision2Seq` + `trust_remote_code`, gated by
`OPENRAL_ALLOW_REMOTE_CODE=1`). NF4 (4-bit) quantization plus the CUDA
expandable-segments allocator bring the 7.5 B backbone within an 8 GB GPU.
The RLinf checkpoint currently needs a 4.40-era transformers runtime; the
default OpenRAL workspace pins transformers 5.3 for lerobot families, so keep
OpenVLA validation in a dedicated environment rather than syncing it together
with the default VLA groups.

### Observation → action contract

- **Input:** one 224×224 RGB frame (the SimplerEnv 3rd-view, surfaced as
  `camera1`) and the prompt
  `In: What action should the robot take to {instruction.lower()}?\nOut: `. No
  proprioception (`use_proprio=False`).
- **Output:** 256-bin discrete action tokens decoded to `[-1, 1]`, then
  de-normalized with the embedded `unnorm_key=bridge_orig` stats (BOUNDS_Q99):
  6 end-effector deltas (3 position + 3 rotation) rescaled, gripper passed
  through. The manifest drives RLinf's `generate_action_verl` path with
  right-padded prompts (`max_length=30`), temperature sampling (`0.6`), torch
  seed `0`, `action_scale=2.0` on the first six dimensions, and binary gripper
  threshold `0.5`. Action chunk = 8 × 7-D, replayed open-loop.

## Upstream model / training

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
`evaluated_tasks` = `simpler_env/widowx_carrot_on_plate`,
`benchmarks.simpler_env_widowx: 0.4`, `policy_extras` = the RLinf generation
and action-transform knobs, `action_contract` = `delta_ee_6d_plus_gripper`
(dim 7).

## Quick start

```bash
just sync --all-packages --group simpler-env
hf download RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood
OPENRAL_ALLOW_REMOTE_CODE=1 openral benchmark run \
  --suite simpler_env_widowx --task simpler_env/widowx_carrot_on_plate \
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

Local seeded validation on an RTX 4070 Laptop GPU (8 GB), NF4, SimplerEnv
ManiSkill3 `PutCarrotOnPlateInScene-v1`, 5 episodes, seeds 0..4, 60-step
horizon, `generate_action_verl`, torch seed 0 reapplied on each policy reset,
`action_scale=2.0`:

- `simpler_env/widowx_carrot_on_plate`: **2/5 success (40%)**.

Public `widowx_carrot_on_plate` without the RLinf action transform scored 0/5,
and the exact upstream `PutOnPlateInScene25Main-v3` registration needs RLinf
assets that were not present in the public source checkout. Those numbers are
not claimed here.

## License

MIT (upstream `RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood`). The OpenRAL
packaging is Apache-2.0. The checkpoint is a `trust_remote_code` custom-code
model; loading executes repo-shipped Python and requires
`OPENRAL_ALLOW_REMOTE_CODE=1` (provenance: rSkill signature verification is not
yet implemented — ADR-0006).

## See also

- [ADR-0063 — OpenVLA / OpenVLA-OFT policy family](../../docs/adr/0063-openvla-oft-policy-family.md)
- [ADR-0060 — benchmark task-data compatibility gate](../../docs/adr/0060-benchmark-task-data-compatibility-gate.md)
- [`rldx1-ft-simpler-widowx-nf4`](../rldx1-ft-simpler-widowx-nf4) — the sibling WidowX bridge rSkill.
