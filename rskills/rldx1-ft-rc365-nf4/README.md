---
tags:
  - openral
  - rskill
  - rldx
  - vla
  - panda_mobile
  - robocasa
  - kitchen
  - manipulation
  - non-commercial
license: other
license_name: rlwrld-model-license-v1.0
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
language:
  - en
---

# rskill-rldx1-ft-rc365-nf4

> RLDX-1 finetuned on the **RoboCasa-365** cross-task generalization
> benchmark — 365 tasks across a wide scene/skill distribution,
> PandaMobile embodiment. Upstream paper-reported success: **31.5 %**
> (vs the focused 24-task RoboCasa Kitchen finetune
> `RLDX-1-FT-ROBOCASA`, which scores higher on its narrower suite).

<!-- openral:rskill-readme-delegates-to: ../rldx1-ft-libero-nf4 -->

This is a sibling of [`rldx1-ft-libero-nf4`](https://huggingface.co/OpenRAL/rskill-rldx1-ft-libero-nf4);
that README owns the canonical architecture, license, auto-managed
sidecar lifecycle, and NF4 quantization documentation for every member
of the RLDX-1 family. Read it first. The sections below cover only
this checkpoint's RC365-specific contract.

## Run

The `rldx` adapter auto-spawns the sidecar with `--embodiment-tag GENERAL_EMBODIMENT` (the model card's own inference example) on first observation — single command:

```bash
openral sim run \
    --config scenes/sim/robocasa_pnp.yaml \
    --rskill rskills/rldx1-ft-rc365-nf4 \
    --rskill rskills/rldx1-ft-rc365-nf4 \
    --view
```

Manual boot (debug / shared host): set `OPENRAL_RLDX_AUTO_SPAWN=0` and run `python tools/rldx_sidecar.py --model RLWRLD/RLDX-1-FT-RC365 --port 5555 --quantization nf4 --embodiment-tag GENERAL_EMBODIMENT` yourself.

## Action / state contract

RLDX-1-FT-RC365 targets PandaMobile (the Franka Panda on a mobile base — RoboCasa's default robot). Per `processor_config.json`'s `general_embodiment` slot (which the model card's inference example selects via `EmbodimentTag.GENERAL_EMBODIMENT`):

**Inputs:**

| Modality | Keys | Per-key dim | Total |
|---|---|---|---|
| Video (T=4) | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand` | 256×256 RGB | 3 streams |
| State | `eef_position_relative`, `eef_rotation_relative` (quat), `gripper_qpos`, `base_position`, `base_rotation` (quat) | 3, 4, 2, 3, 4 | **16-D** |
| Language | `annotation.human.task_description` | string | — |

**Outputs** (action chunks of length 16, per-step layout):

| Key | Dim | Type |
|---|---|---|
| `end_effector_position` | 3 | delta |
| `end_effector_rotation` | 3 | delta (axis-angle) |
| `gripper_close` | 1 | absolute |
| `base_motion` | 4 | delta (dx, dy, dyaw, dz) |
| `control_mode` | 1 | absolute |
| **total** | **12** | |

The openral adapter concatenates these into a 12-D action vector. RoboCasa's PandaMobile BASIC composite consumes 11-D (`arm_osc(6) + gripper(1) + base(3) + torso(1)`); `openral_sim/backends/robocasa.py` already trims the trailing dim (the legacy "torso" slot the model treats as a control_mode flag).

## State re-slicing

The openral RoboCasa scene emits the **human300_16d** state layout: `eef_pos(3) + eef_quat(4) + base_pos(3) + base_rot(4) + grip(2)`. RC365's general_embodiment expects gripper BEFORE base; the adapter re-slices accordingly (see `_RC365_STATE_SLICES_FROM_HUMAN300` in `openral_sim.policies.rldx`):

```
human300 idx [0:3]   → state.end_effector_position_relative  (eef_pos)
human300 idx [3:7]   → state.end_effector_rotation_relative  (eef_quat)
human300 idx [14:16] → state.gripper_qpos                    (grip)
human300 idx [7:10]  → state.base_position                   (base_pos)
human300 idx [10:14] → state.base_rotation                   (base_rot)
```

Set `scene.backend_options.state_layout: human300_16d` on the SimEnvironment YAML to make sure the RoboCasa scene emits the layout the adapter expects.

Upstream: <https://huggingface.co/RLWRLD/RLDX-1-FT-RC365>
