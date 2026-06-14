---
tags:
  - OpenRAL
  - rskill
  - smolvla
  - lerobot
  - vla
  - franka_panda
  - maniskill
  - maniskill3
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-smolvla-maniskill-franka

> **OpenRAL rSkill** — SmolVLA (0.45 B) finetuned on a 1000-demo Franka
> LiftCube dataset in ManiSkill3 SAPIEN, packaged for use with the
> [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps
[`Calvert0921/smolvla_franka_liftcube_1000`](https://huggingface.co/Calvert0921/smolvla_franka_liftcube_1000)
with a `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does
**not** copy model weights.

## What this skill does

Picks and lifts a single cube on a tabletop with a Franka Panda arm in
the ManiSkill3 SAPIEN simulator. Action chunks of length 50; two RGB
camera views (overhead + wrist); 9-D Franka proprio state.

| Field | Value |
| --- | --- |
| Actions | `pick`, `lift` |
| Objects | `cube` |
| Scenes  | `tabletop` |
| Embodiment | `franka_panda` |

## Upstream model / training

| Field | Value |
| --- | --- |
| Source repo | [`Calvert0921/smolvla_franka_liftcube_1000`](https://huggingface.co/Calvert0921/smolvla_franka_liftcube_1000) |
| Base model  | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper       | [arxiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA: Efficient Vision-Language-Action Model* |
| License     | apache-2.0 (inherited from base; the upstream finetune ships no LICENSE / model card — assumption documented here) |
| Parameters  | ~450 M |
| Training data | [`Calvert0921/SmolVLA_LiftCube_Franka_1000`](https://huggingface.co/datasets/Calvert0921/SmolVLA_LiftCube_Franka_1000) — 1000 Franka LiftCube demos in ManiSkill3 SAPIEN |
| Backbone | SmolVLM2-500M-Video-Instruct (frozen vision encoder) |
| Action head | Flow-matching (10 denoising steps per chunk) |
| Chunk size | 50 |

The training data was collected with a `pd_joint_pos` Franka in
SAPIEN; the checkpoint therefore outputs 8-D joint commands (7 arm + 1
gripper). State inputs are the Franka qpos (9 values: 7 arm joints +
2 finger joints) — the gym observation `agent.qpos` exactly matches
this layout.

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in  | `observation.images.up`        | `(1, 3, 256, 256) uint8` | Overhead / base camera (`camera1` in-tree, aliased) |
| in  | `observation.images.wrist`     | `(1, 3, 256, 256) uint8` | Wrist / hand camera (`camera2` in-tree, aliased) |
| in  | `observation.state`            | `(1, 9) float32`         | Franka `agent.qpos` (7 arm + 2 fingers) |
| in  | `task`                         | `list[str]`              | Free-form natural language instruction |
| out | action chunk                   | `(50, 8) float32`        | Per-step joint position command (7 arm + 1 gripper) |

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (ManiSkill3 SAPIEN) | `franka_panda` | ✓ end-to-end | Manifest validates and `openral sim run --view` produces a live SAPIEN window of the policy lifting the cube; processors are auto-synthesized from the training dataset's `meta/episodes_stats.jsonl` because the upstream model repo doesn't ship `policy_*processor.json`. |

## Sensors required

Mirrors `rskill.yaml::sensors_required`:

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 256 × 256 | `uint8`, aliased to `up` at preprocessing |
| `observation.images.camera2` | RGB | 256 × 256 | `uint8`, aliased to `wrist` at preprocessing |
| `observation.state`          | proprioception | (9,) | `float32` (Franka qpos) |

## Manifest summary

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-smolvla-maniskill-franka` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` (fast visuomotor policy) |
| `model_family` | `smolvla` |
| `embodiment_tags` | `franka_panda` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://Calvert0921/smolvla_franka_liftcube_1000` |
| `chunk_size` / `n_action_steps` | `50` / `50` |
| `latency_budget.per_chunk_ms` | 200 ms |
| `state_contract.dim` / `action_contract.dim` | 9 / 8 |
| `commercial_use_allowed` | `true` (apache-2.0) |

## Quick start

```python
from openral_rskill.loader import rSkill

pkg = rSkill.from_yaml("rskills/smolvla-maniskill-franka/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.version)
print(pkg.manifest.weights_uri)
```

## Reproduction

```bash
# One-time bootstrap
just bootstrap && uv sync --all-packages --group sim --group maniskill3

# End-to-end rollout (live SAPIEN window via --view).
DISPLAY=:1 uv run --group sim --group maniskill3 \
    openral sim run --config scenes/benchmark/maniskill_pick_cube.yaml \
                --rskill rskills/smolvla-maniskill-franka \
                --view
```

The runner deferred-opens the SAPIEN window after the policy load
(~25 s) so the window manager never sees an unresponsive empty
viewer. On a warm cache the policy lifts the cube within ~200 steps
(reward accumulates from `0` to `~50` over the rollout).

## Evaluation

No benchmarks shipped yet (`eval/.gitkeep` only). The headline LiftCube
success rate will be populated by `openral benchmark run` once a paired
benchmark suite lands in `benchmarks/`:

```bash
openral benchmark run \
    --suite maniskill3_pick_place \
    --vla smolvla:rskills/smolvla-maniskill-franka
```

## How the wiring works

The model expects two RGB cameras (`up` / `wrist`), a 9-D Franka qpos
state, and emits 8-D joint position commands. The end-to-end path:

1. **ManiSkill3 backend** (`python/sim/src/openral_sim/backends/maniskill3.py`)
   surfaces every entry in `sensor_data` as `camera1` / `camera2` /
   ... in declaration order, plumbs `backend_options.robot_uids`
   through to `gym.make` (so this YAML's `panda_wristcam` brings in
   the wrist camera), and forwards `task.max_steps` to
   `max_episode_steps` so the rollout isn't silently truncated at
   MS3's default 50 steps.
2. **rSkill manifest** declares `image_preprocessing.aliases:
   {camera1: up, camera2: wrist}` so the SmolVLA preprocessor finds
   what it expects, and `state_contract.dim: 9` so the adapter
   truncates the env's `qpos+qvel` state to qpos-only.
3. **SmolVLA adapter** (`python/sim/src/openral_sim/policies/smolvla.py`)
   detects the upstream model's missing `policy_*processor.json`
   files and falls back to rebuilding the lerobot processors from
   `manifest.dataset_uri`'s `meta/episodes_stats.jsonl` — generic
   path that applies to any community finetune uploaded without
   processors.
4. **`--view`** triggers the SAPIEN `viewer_render()` hook
   (deferred-window, mirroring PR #160's simpler_env). The window
   opens lazily on the first applied step, after the policy is
   loaded.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0**.
The wrapped weights at
[`Calvert0921/smolvla_franka_liftcube_1000`](https://huggingface.co/Calvert0921/smolvla_franka_liftcube_1000)
ship without an explicit LICENSE file; the base model
`lerobot/smolvla_base` is Apache-2.0 and this derivative is treated as
Apache-2.0 here on the inheritance assumption. If the upstream author
later publishes a different posture, this manifest's `license:` field
should be updated to match.

## See also

- [`robots/franka_panda/robot.yaml`](../../robots/franka_panda/robot.yaml) — RobotDescription manifest.
- [`scenes/benchmark/maniskill_pick_cube.yaml`](../../scenes/benchmark/maniskill_pick_cube.yaml) — paired BenchmarkScene config (pass `--rskill rskills/smolvla-maniskill-franka`).
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
