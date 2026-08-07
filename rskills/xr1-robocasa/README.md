---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- xr1
- vision-language-action
- nf4
- 4-bit
- panda_mobile
- robocasa
base_model:
- XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa
base_model_relation: quantized
inference: false
---

# XR-1 RoboCasa

> OpenRAL wrapper for Xiaomi Robotics' XR-1 RoboCasa checkpoint on the
> PandaMobile kitchen embodiment. The package points at the pinned upstream
> safetensors; it does not copy weights.

## Preview

[Clean RoboCasa rollout (fail)](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/robocasa-pick-place_xr1-nf4_fail/full.mp4)

![XR-1 RoboCasa preview](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/robocasa-pick-place_xr1-nf4_fail/poster.jpg)

## What this skill does

| Field | Value |
| --- | --- |
| Actions | pick, place, open, close |
| Objects | kitchen objects |
| Scenes | RoboCasa kitchens |
| Embodiment | `panda_mobile` |

## How it works

XR-1 combines Qwen3-VL-4B-Instruct with a 36-layer DiT action head. OpenRAL
runs the exact torch 2.8 / transformers 4.57.1 / FlashAttention stack in a
sidecar, pads the 8-D physical state to XR-1's internal 60-D state tensor, and
applies Xiaomi's 0.95 center crop before returning the checkpoint's first seven
decoded action channels.

### Observation -> action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `camera1`, `camera2`, `camera3` | three RGB 256x256 frames | left/right base and wrist |
| in | `observation.state` | `(8,)` | seven arm joints plus one gripper value |
| out | action chunk | `(T, 7)` | delta XYZ, delta axis-angle, gripper |

## Upstream model and training

| Field | Value |
| --- | --- |
| Source repo | [`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa) |
| Base model | XR-1 5B post-trained for RoboCasa |
| Paper | [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) |
| License | Apache-2.0 |
| Parameters | 5.44B |
| Training data | Xiaomi's RoboCasa benchmark post-training data |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| PandaMobile in RoboCasa | `panda_mobile` | integration-ready | Upstream evaluates RoboCasa v0.2; OpenRAL's current backend version must still be benchmark-correlated. |

## Sensors required

Three RGB streams at 256x256 or larger and an 8-D arm/gripper state.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-xr1-panda_mobile-robocasa-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `runtime` / dtype | PyTorch sidecar / NF4 |
| `weights_uri` | `hf://OpenRAL/rskill-xr1-panda_mobile-robocasa-nf4` |
| `latency_budget.per_chunk_ms` | 120000 |
| `commercial_use_allowed` | yes |

## Quick start

```bash
just sync --group sidecar-wire --group robocasa
OPENRAL_ALLOW_REMOTE_CODE=1 openral sim run \
  --config scenes/sim/xr1_robocasa_pnp.yaml \
  --rskill rskills/xr1-robocasa
```

## Reproduction

The command above validates the OpenRAL observation/action wiring. It is not a
reproduction of Xiaomi's published RoboCasa score because Xiaomi evaluates
RoboCasa v0.2 while OpenRAL currently provisions its own pinned kitchen stack.

## Evaluation

No benchmark JSON is shipped until a full upstream-version-matched run is
completed. Integration claims are limited to schema and adapter validation.

## License

The rSkill metadata and upstream XR-1 code/checkpoint are Apache-2.0. Loading
requires `OPENRAL_ALLOW_REMOTE_CODE=1` because the HF repo ships executable
custom Transformers code.

## See also

- [`scenes/sim/xr1_robocasa_pnp.yaml`](../../scenes/sim/xr1_robocasa_pnp.yaml)
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
