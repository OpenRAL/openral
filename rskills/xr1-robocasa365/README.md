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
- robocasa365
base_model:
- XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
base_model_relation: quantized
inference: false
---

# XR-1 RoboCasa365

> OpenRAL wrapper for Xiaomi Robotics' XR-1 RoboCasa365 checkpoint. The
> package references the pinned upstream checkpoint and preserves its temporal
> three-camera contract.

## Preview

[Successful RoboCasa365 OpenDrawer rollout](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/robocasa365-open-drawer_xr1-nf4_success/full.mp4)

![XR-1 RoboCasa365 OpenDrawer preview](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/robocasa365-open-drawer_xr1-nf4_success/poster.jpg)

[CloseBlenderLid fail rollout](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/robocasa365-close-blender_xr1-nf4_fail/full.mp4)

## What this skill does

| Field | Value |
| --- | --- |
| Actions | generalist kitchen manipulation |
| Objects | kitchen objects and fixtures |
| Scenes | RoboCasa365 kitchens |
| Embodiment | `panda_mobile` |

## How it works

The adapter stores seven observations, samples four frames at offsets
`[-6, -4, -2, 0]`, converts the OpenRAL 16-D EE/base quaternion state to the
14-D axis-angle order used by Xiaomi's evaluator, applies the reference 0.95
center crop, and sends the batch to the isolated XR-1 sidecar. It replays 16
actions before replanning.

### Observation -> action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | three RGB videos | `(4, H, W, 3)` each | left/right base and wrist |
| in | state history | `(4, 14)` | EE, gripper, base; raw physical units |
| out | action chunk | `(T, 12)` | EE delta, gripper, base, padding/mode |

## Upstream model and training

| Field | Value |
| --- | --- |
| Source repo | [`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365) |
| Base model | XR-1 5B |
| Paper | [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) |
| License | Apache-2.0 |
| Parameters | 5.44B |
| Training data | RoboCasa365 |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| PandaMobile in RoboCasa | `panda_mobile` | integration-ready | Xiaomi does not pin the exact RoboCasa365 revision, so benchmark reproduction remains blocked. |

## Sensors required

Three RGB streams plus the PandaMobile EE, gripper, and base state described in
the manifest's `rc365` bindings.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `runtime` / dtype | PyTorch sidecar / NF4 |
| `weights_uri` | `hf://OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4` |
| `latency_budget.per_chunk_ms` | 120000 |
| `commercial_use_allowed` | yes |

## Quick start

```bash
just sync --group sidecar-wire --group robocasa
OPENRAL_ALLOW_REMOTE_CODE=1 openral sim run \
  --config scenes/sim/xr1_robocasa365_close_blender_lid.yaml \
  --rskill rskills/xr1-robocasa365
```

## Reproduction

The scene is a 20-step wiring check matching Xiaomi's documented smoke task,
not a score reproduction. Xiaomi's exact RoboCasa365 dependency revision is
not published.

## Evaluation

No benchmark JSON is shipped until the upstream environment revision can be
pinned and the full 50-trial protocol is run.

## License

The wrapper and upstream code/weights are Apache-2.0. The checkpoint executes
custom HF code and therefore requires `OPENRAL_ALLOW_REMOTE_CODE=1`.

## See also

- [`scenes/sim/xr1_robocasa365_close_blender_lid.yaml`](../../scenes/sim/xr1_robocasa365_close_blender_lid.yaml)
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
