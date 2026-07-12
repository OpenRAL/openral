---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- smolvla
- lerobot
- vision-language-action
- franka_panda
- vla
- vlabench
- manipulation
inference: false
---

# rskill-smolvla-franka_panda-vlabench-bf16

> **OpenRAL rSkill — VLABench integration baseline (NOT a passing policy).**
> SmolVLA (~0.5 B) finetuned on [VLABench](https://github.com/OpenMOSS/VLABench)
> (`lerobot/vlabench_unified`), lerobot-native, runs in-process on lerobot 0.6.0
> (bf16, fits 8 GB). Wraps [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench).

## ⚠ Status: baseline, scores 0%

This rSkill exists to **exercise and validate the OpenRAL VLABench backend**, not
to score the benchmark. Measured **0/3 on six diverse primitive tasks**
(`select_fruit`, `select_drink`, `select_toy`, `select_book`, `add_condiment`,
`insert_flower`) — **identical to lerobot's own `lerobot-eval` reference**, which
confirms the OpenRAL wiring is faithful (state 7-D, action absolute-eef, cameras
`camera1/2/3`) and the 0% is the policy, not the integration.

The only VLABench policy above 50% is `VLABench/pi0-fast-ft-primitive-10task-deltachunk`
(51.2% primitive avg), which is **openpi/JAX** and would need conversion to lerobot
`PI0FAST` (+int8 for 8 GB) to run in-process — a dedicated, not-yet-done effort.
VLABench's composite/long-horizon suite is unsolved (<50%) by every known policy.

## Provisioning

The **Python side auto-installs** on first env build via the `vlabench`
`ensure_backend_deps` plan (`OPENRAL_AUTO_INSTALL_DEPS=1`, the default): it clones
`OpenMOSS/VLABench`, `uv pip install --no-deps -e`'s it, adds the numpy-2 sim deps, and
writes a raise-on-use `rrt_algorithms` stub (git-only data-gen dep, off the VLA eval path).

The **~12 GB CC-BY asset bundle is a one-time manual fetch** (a Google-Drive `gdown` pull
too flaky to drive unattended — the backend raises with this exact recipe when it is absent):

```bash
export VLABENCH_ROOT=$HOME/.cache/openral/repos/VLABench/VLABench   # the clone the plan installs
python $HOME/.cache/openral/repos/VLABench/scripts/download_assets.py   # ~12 GB obj + scene
```

## Run

```bash
MUJOCO_GL=egl VLABENCH_ROOT=$VLABENCH_ROOT \
  openral benchmark scene --config scenes/benchmark/vlabench_select_fruit.yaml \
    --rskill rskills/smolvla-vlabench
```

## Upstream model / training

Base is [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)
(SmolVLA ~0.5 B, [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)), finetuned
on [`lerobot/vlabench_unified`](https://huggingface.co/datasets/lerobot/vlabench_unified)
(VLABench, 97 tasks). The wrapped checkpoint is
[`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench)
(Apache-2.0); OpenRAL adds no weights, only packaging.

## Supported robots / embodiments

`franka_panda` (VLABench's 7-DOF Franka Panda). The manifest's `embodiment_tags`
must intersect the robot's — matched against `robots/franka_panda`.

## Sensors / observation contract

Three RGB views (`camera1/2/3`, from the env's `image`/`second_image`/`wrist_image`,
≥224×224) plus a 7-D proprio state `[pos_robot(3), euler_xyz(3), gripper(1)]`. The
checkpoint's preprocessor renames `image→camera1…` (a no-op on the already-canonical
keys) and resizes to 256.

## Manifest summary

| Field | Value |
| --- | --- |
| `model_family` | `smolvla` (bf16, ~0.5 B) |
| cameras | `camera1/2/3` (env `image`/`second_image`/`wrist_image`) |
| state | 7-D `[pos_robot(3), euler_xyz(3), gripper(1)]` |
| action | 7-D absolute eef pose → IK (`delta_ee_6d_plus_gripper` label is nominal) |
| robot | `franka_panda` (uses the manifest's 3rd `camera3`/`front` sensor) |

## License

Apache-2.0 — both this rSkill's packaging and the wrapped `lerobot/smolvla_vlabench`
checkpoint.

## See also
- `python/sim/src/openral_sim/backends/vlabench.py` — the backend.
- [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench) — upstream checkpoint.
