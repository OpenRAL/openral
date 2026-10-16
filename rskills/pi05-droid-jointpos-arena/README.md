---
language:
- en
license: other
license_name: permissive-research
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- pi05
- lerobot
- vision-language-action
- droid_franka_robotiq
- droid
inference: false
---

# rskill-pi05-droid_franka_robotiq-droid_jointpos-bf16

> **OpenRAL rSkill** — Physical Intelligence's [π0.5](https://arxiv.org/abs/2504.16054)
> DROID **joint-position** checkpoint (openpi `pi05_droid_jointpos`), the policy
> [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) evaluates on its
> DROID / RoboLab tabletop tasks, packaged for `openral sim run` / `openral deploy sim`
> on OpenRAL's Isaac Sim 6.x backend with a **Franka Panda + Robotiq 2F-85**.

This package points at
[`DAVIAN-Robotics/pi05_droid_jointpos`](https://huggingface.co/DAVIAN-Robotics/pi05_droid_jointpos)
(lerobot `PI05Policy`, pinned to commit `66f1ce6e`) and adds an `rskill.yaml`:
capability checking, camera-slot aliasing, DROID's gripper sense, the 15 Hz time base
and the delta / absolute action split the lerobot port leaves to the caller.

## What this skill does

Picks a tabletop object and places it in a container, prompted in language. The
committed scene is Arena's `banana_in_bowl` (`"pick up the banana and place it in the
bowl"`); the same checkpoint covers the other RoboLab pick-and-place tasks.

| Field | Value |
| --- | --- |
| Actions | `pick`, `place`, `pick_and_place` |
| Objects | `bowl`, `object` |
| Scenes | `tabletop` |
| Embodiment | `droid_franka_robotiq` |

## How it works

π0.5 (PaliGemma 3B + a flow-matching action expert, 10 denoising steps). One exterior
and one wrist RGB view (resized with padding to 224 × 224 by lerobot; the
checkpoint's third, right-wrist slot is padded) and the 8-D state go in; a 15-step
chunk comes out and is replayed in full before the next inference, as Arena's openpi
client does (`open_loop_horizon` 15).

### Observation → action contract

openpi `DroidInputs` / `DroidOutputs` (`src/openpi/policies/droid_policy.py` L35-81 at
`215abfb2`), as Arena's client feeds them (`isaaclab_arena_openpi/policy/droid_adapter.py`
L30-75 at `738d20a2`):

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` → `base_0_rgb` | `(1, 3, 720, 1280)` | Arena `external_camera` (`exterior_image_1_left`) |
| in | `observation.images.camera2` → `left_wrist_0_rgb` | `(1, 3, 720, 1280)` | Arena `wrist_camera` (`wrist_image_left`) |
| in | `observation.state` | `(1, 8)` | 7 joint positions (rad) + gripper (DROID: 0 open … 1 closed) |
| in | `task` | str | e.g. `"pick up the banana and place it in the bowl"` |
| out | action chunk | `(15, 8)` | 7 joint **deltas** from the observed state + absolute gripper |

**Deltas.** openpi trains joints 0-6 as deltas (`training/config.py` L403-409:
`DeltaActions` with `make_bool_mask(7, -1)`) and its server adds the state back
(`AbsoluteActions`). The lerobot port ships only unnormalize in its postprocessor
(its card: "an external adapter is required"). The manifest declares the split on
`actuators_required` (`mode: delta`, `n_dof: 7` / `mode: absolute`, `n_dof: 1`), and
the pi05 adapter adds each chunk's own observed joint state to its delta channels
right after inference (`pi05.delta_actions`). The checkpoint pads its action to 32;
the adapter keeps `action_contract.dim` 8 (`pi05.action_truncated`).

**Gripper.** DROID's `gripper_position` is 1 = closed (Arena: `finger_joint / (π/4)`);
OpenRAL's jaw channel is `[0 = closed, 1 = open]`. `action_contract.joint_calibration`
`finger_joint: {scale: -1, offset: 1}` flips it on the state in and the action out.
Arena binarises the command at > 0.5 (`actions.py` L17-22); this skill sends the
continuous value.

**Time base.** 15 Hz: openpi `examples/droid/main.py` `DROID_CONTROL_FREQUENCY = 15`;
Arena `sim.dt` 1/120 s × `decimation` 8.

## Upstream model / training

| Field | Value |
| --- | --- |
| Weights repo | [`DAVIAN-Robotics/pi05_droid_jointpos`](https://huggingface.co/DAVIAN-Robotics/pi05_droid_jointpos) @ `66f1ce6e5d199eb7f8f266bc5ed9d67644fbb6f1` (7.23 GB `model.safetensors`) |
| Converted from | `gs://openpi-assets-simeval/pi05_droid_jointpos` (card), the weights Arena's openpi server loads (`docker/run_openpi_server.sh` L51-58, config `pi05_droid_jointpos_polaris`) |
| openpi config | `pi05_droid_jointpos_polaris`: `Pi0Config(action_horizon=15, pi05=True)`, `DroidActionSpace.JOINT_POSITION`, quantile norm (`training/misc/polaris_config.py` L29-56) |
| Training data | DROID 1.0.1 (weight 0.9) + `polaris_droid_cotrain_dataset` 1.0.0 (0.1) |
| Paper | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) — *π0.5* |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda + Robotiq 2F-85 | `droid_franka_robotiq` | ⚡ experimental | Simulation only (`scenes/sim/isaac_arena_droid_banana_in_bowl.yaml`, `scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml`). Not yet run: Spark (the Isaac 6.1 host) was down when this was packaged. |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 (Arena renders 1280 × 720) | `float32` |
| `observation.images.camera2` | RGB | 224 × 224 (Arena renders 1280 × 720) | `float32` |
| `observation.state` | proprioception | `(8,)` | `float32` |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-droid_franka_robotiq-droid_jointpos-bf16` |
| `version` | `0.1.0` |
| `license` | `permissive_research` |
| `model_family` | `pi05` |
| `quantization` | `bf16` / `pytorch` (needs ≥ 8.1 GB device memory) |
| `chunk_size` / `n_action_steps` | 15 / 15 |
| `action_contract` | dim 8, `joint_positions`, radians, 15 Hz, gripper calibrated |

## Quick start

```bash
# Isaac Sim 6.1 (binary install at /opt/isaac-sim) + an RTX GPU with >= 9 GB.
OPENRAL_ISAAC_SIDECAR_PYTHON=/opt/isaac-sim/python.sh OMNI_KIT_ACCEPT_EULA=YES \
uv run openral sim run --config scenes/sim/isaac_arena_droid_banana_in_bowl.yaml \
    --rskill rskills/pi05-droid-jointpos-arena \
    --n-episodes 8 --save-video outputs/arena_droid_banana
```

## Reproduction

Card claim to reproduce: RoboLab `BananaInBowlTask` 7/8 (JAX baseline 6/8). Arena's
own reference run: `python isaaclab_arena/evaluation/experiment_runner.py
--experiment_config isaaclab_arena_environments/robolab/experiment_configs/robolab_20_tasks_pi0.yaml`
(task `banana_in_bowl_pi0`). The OpenRAL scene fixes RoboLab's layout instead of
Arena's per-reset placement and grades position-at-rest only (see the scene header).

## Evaluation

No local result yet; `eval/` is empty until the first Spark run.

## License

- **Code:** openpi Apache-2.0; this package Apache-2.0.
- **Weights:** `permissive_research` — Physical Intelligence π0.5 weights, as OpenRAL
  treats every π0.5 checkpoint (CLAUDE.md §3); the DAVIAN conversion declares no
  license of its own. Commercial use needs an agreement with Physical Intelligence.

## See also

- [`robots/droid_franka_robotiq`](../../robots/droid_franka_robotiq/) — the robot.
- [`docs/reference/isaac-reproducible-datasets-survey.md`](../../docs/reference/isaac-reproducible-datasets-survey.md) §4.1.
- [`rskills/pi05-libero-int8`](../pi05-libero-int8/) — the same π0.5 graph on LIBERO.
