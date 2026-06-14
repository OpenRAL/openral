---
tags:
  - OpenRAL
  - rskill
  - pi05
  - lerobot
  - vla
  - franka
  - robocasa
  - kitchen
  - manipulation
license: apache-2.0
language:
  - en
---

# pi05-robocasa365-human300-nf4

> **OpenRAL rSkill** — pre-quantized **nf4** packaging of Physical
> Intelligence's **π₀.₅** (3.4 B PaliGemma backbone) fine-tuned on the
> [RoboCasa365 Human-300](https://robocasa.ai) task suite (300 atomic +
> composite kitchen tasks, 100 demos each) against the **PandaMobile**
> embodiment.

## Upstream model

| Field | Value |
| --- | --- |
| Source | `robocasa/robocasa365_checkpoints` (multitask_learning/75000), converted via [`tools/openpi_to_lerobot_pi05.py`](../../tools/openpi_to_lerobot_pi05.py) and quantized via [`tools/quantize_rskill.py`](../../tools/quantize_rskill.py). |
| HF mirror | `OpenRAL/rskill-pi05-robocasa365-human300-nf4` |
| Training data | RoboCasa365 Human-300: 300 atomic + composite kitchen tasks, 100 demos each. |
| Architecture | π₀.₅: 3.4 B PaliGemma backbone + flow-matching action head. |
| License | Apache-2.0 (code + weights) |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| PandaMobile (Franka Panda on a mobile base) | `franka_panda` | ✓ sim | RoboCasa Kitchen default robot; state layout = `human300_16d`. |

## Sensors required

| Key | Modality | Resolution | Notes |
| --- | --- | --- | --- |
| `observation.images.robot0_agentview_left_image`  | RGB | 256 × 256 | Aliased to the policy's `camera1` key via `image_preprocessing.aliases`. |
| `observation.images.robot0_agentview_right_image` | RGB | 256 × 256 | Aliased to `camera2`. |
| `observation.images.robot0_eye_in_hand_image`     | RGB | 256 × 256 | Aliased to `camera3`. |
| `observation.state`                               | proprioception | (16,) | `human300_16d` layout: eef_pos(3) · eef_quat(4) · base_pos(3) · base_rot(4) · gripper(2). |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-robocasa365-human300-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `franka_panda` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (nf4 / bitsandbytes / bf16 compute) |
| `weights_uri` | `hf://OpenRAL/rskill-pi05-robocasa365-human300-nf4` |
| `chunk_size` | 50 |
| `state_contract` | `human300_16d` named layout |
| `commercial_use_allowed` | `true` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Why nf4?

The HF-hosted mirror ships the *already-packed* nf4 state dict plus a
`quantization_metadata.json` sentinel. The pi05 adapter detects the
sentinel, meta-initialises the policy graph (~14 s instead of the
~137 s `from_pretrained` walk), overlays the prequant state via
`install_prequantized_linears`, and skips the bf16 → nf4 conversion
entirely. Warm-up drops to ~20 s on a 4070-mobile (8 GiB).

## Running it

```bash
OPENRAL_ALLOW_ROBOCASA_ASSETS=1 \
  uv run openral sim run \
    --config scenes/sim/robocasa_pnp.yaml \
    --rskill rskills/pi05-robocasa365-human300-nf4 \
    --rskill rskills/pi05-robocasa365-human300-nf4 \
    --view --max-steps 200
```

The `robocasa` group **conflicts with `libero`** in a single venv; see
[`docs/tutorials/sim/create-a-sim-environment.md`](../../docs/tutorials/sim/create-a-sim-environment.md)
("Level 6: a custom MuJoCo environment via RoboCasa") for the one-time
setup (clone robocasa, install robosuite@master, fetch the CC-BY-4.0
kitchen assets).

## Image preprocessing

`flip_vertical: true` — the canonical openpi-robocasa eval pulls images
through `RoboCasaGymEnv.process_img` which applies `img[::-1, :, :]`.
The adapter applies the same flip before forward; the alias map routes
the three robosuite cameras into the policy's `camera1/2/3` keys.

## License

Apache-2.0 — both the wrapping rSkill package (`rskill.yaml`,
`README.md`) and the wrapped upstream checkpoint
(`robocasa/robocasa365_checkpoints`). Commercial use is allowed
(`commercial_use_allowed: true`).
