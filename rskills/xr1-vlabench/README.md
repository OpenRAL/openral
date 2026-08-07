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
- franka_panda
- vlabench
base_model:
- XiaomiRobotics/Xiaomi-Robotics-1-VLABench
base_model_relation: quantized
inference: false
---

# XR-1 VLABench

> OpenRAL wrapper for Xiaomi Robotics' XR-1 VLABench checkpoint on the
> Franka Panda benchmark embodiment.

## Preview

[Clean VLABench rollout (fail)](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/vlabench-select-fruit_xr1-nf4_fail/full.mp4)

![XR-1 VLABench preview](https://huggingface.co/datasets/OpenRAL/website-media/resolve/main/benchmarks/vlabench-select-fruit_xr1-nf4_fail/poster.jpg)

## What this skill does

| Field | Value |
| --- | --- |
| Actions | grasp, pick, place |
| Objects | fruit and tabletop objects |
| Scenes | VLABench |
| Embodiment | `franka_panda` |

## How it works

The sidecar receives VLABench's front, second, and wrist views at 480x480 and
the raw 7-D relative EE state. XR-1 predicts ten 7-D deltas. OpenRAL integrates
those deltas into the absolute targets consumed by VLABench and executes five
before replanning. The gripper scalar uses Xiaomi's 0.2 threshold and maps to
OpenRAL's normalized closed/open commands.

Camera order is load-bearing and matches Xiaomi's evaluator:
`camera1=front(raw index 2)`, `camera2=base/second(raw index 0)`,
`camera3=wrist(raw index 3)`. Frames are not flipped (`flip_180: false`).

### Observation -> action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `camera1`, `camera2`, `camera3` | three RGB 480x480 frames | front, second, wrist |
| in | `observation.state` | `(7,)` | XYZ, Euler, gripper |
| out | action chunk | `(10, 7)` | integrated absolute target pose plus gripper |

## Upstream model and training

| Field | Value |
| --- | --- |
| Source repo | [`XiaomiRobotics/Xiaomi-Robotics-1-VLABench`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-VLABench) |
| Base model | XR-1 5B |
| Paper | [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) |
| License | Apache-2.0 |
| Parameters | 5.44B |
| Training data | VLABench |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda in VLABench | `franka_panda` | integration-ready | Full run requires the external VLABench assets. |

## Sensors required

Three RGB streams at 480x480 and VLABench's 7-D EE/gripper state.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-xr1-franka_panda-vlabench-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `runtime` / dtype | PyTorch sidecar / NF4 |
| `weights_uri` | `hf://OpenRAL/rskill-xr1-franka_panda-vlabench-nf4` |
| `latency_budget.per_chunk_ms` | 2000 |
| `commercial_use_allowed` | yes |

## Quick start

```bash
just sync --group sidecar-wire
OPENRAL_ALLOW_REMOTE_CODE=1 openral sim run \
  --config scenes/benchmark/vlabench_select_fruit.yaml \
  --rskill rskills/xr1-vlabench
```

## Reproduction

Provision the VLABench asset bundle documented by the scene backend, then run
the command above. This package ships no score until that real simulator run
has completed.

### Persist NF4 locally

The default sidecar quantizes the pinned BF16 source at load. Export once to
avoid that repeated pack step:

```bash
OPENRAL_ALLOW_REMOTE_CODE=1 uv run python tools/xr1_sidecar.py \
  --model hf://XiaomiRobotics/Xiaomi-Robotics-1-VLABench@2dfc33b390478f71737eacb4748333e6d8638a06 \
  --profile vlabench_choice --quantization nf4 \
  --export-dir outputs/run_artifacts/xr1-vlabench-nf4-rskill/weights
```

The local unpublished package is then
`outputs/run_artifacts/xr1-vlabench-nf4-rskill`.
Its packed output was verified bit-identical to runtime NF4 on the same seeded
input (`MAE=0`, `max error=0`). On the reference host, policy initialization
dropped from about 18.0 s to 10.1 s.

## Evaluation

NF4 was loaded and executed on an RTX 4070 Laptop 8 GB against the real
VLABench `select_fruit` environment. The sidecar used 3.66 GiB process VRAM;
the first synthetic chunk took 1.35 s and the warmed real-scene chunk took
0.81 s. One finite `(10, 7)` chunk was integrated and applied successfully.
This validates execution, not task success or Xiaomi's benchmark score.

## License

The wrapper and upstream code/weights are Apache-2.0. Loading requires
`OPENRAL_ALLOW_REMOTE_CODE=1` for the checkpoint's custom Transformers code.

## See also

- [`scenes/benchmark/vlabench_select_fruit.yaml`](../../scenes/benchmark/vlabench_select_fruit.yaml)
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
