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
- so101_follower
datasets:
- LightwheelAI/leisaac-pick-orange
inference: false
---

# rskill-smolvla-so101-leisaac_pick_orange-bf16

> **OpenRAL rSkill** — [SmolVLA](https://arxiv.org/abs/2506.01844) finetuned in
> [LeIsaac](https://github.com/LightwheelAI/leisaac) (Isaac Sim) to **pick three
> oranges and put them on a plate** with an **SO-101 follower arm**, packaged for
> `openral sim run` on OpenRAL's Isaac Sim 6.x backend.

This package points at
[`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange)
(model card license `apache-2.0`; pinned to commit `71cf4a9`) and adds an
`rskill.yaml` manifest: capability checking, camera-slot aliasing, LeIsaac's
per-joint unit calibration, the 60 Hz time base, and local registry integration.

## What this skill does

Picks the three oranges off LeIsaac's kitchen counter and puts them on the plate.
Single-task policy: the training dataset has exactly one instruction.

| Field | Value |
| --- | --- |
| Actions | `pick`, `place`, `pick_and_place` |
| Objects | `orange`, `plate` |
| Scenes | `kitchen` |
| Embodiment | `so101_follower` |

> **Prompt it verbatim:** `"Grab orange and place into plate"` (the dataset's only
> task string, also the manifest's `default_prompt`).

## Quick start

```bash
# Isaac Sim 6.x (e.g. a binary install at /opt/isaac-sim) + an RTX GPU.
OPENRAL_ISAAC_SIDECAR_PYTHON=/opt/isaac-sim/python.sh OMNI_KIT_ACCEPT_EULA=YES \
uv run openral sim run --config scenes/sim/isaac_leisaac_pick_orange.yaml \
    --rskill rskills/smolvla-so101-leisaac-pick-orange \
    --instruction "Grab orange and place into plate" \
    --n-episodes 10 --save-video outputs/leisaac_pick_orange
```

## How it works

SmolVLA (~0.45 B: a 16-layer SmolVLM2-500M backbone with a frozen vision encoder
plus a flow-matching action expert). Two 640×480 RGB views and the 6-D joint state
go in; one 50-step chunk of absolute 6-D joint targets comes out and is replayed
in full before the next inference, as LeIsaac's evaluation does
(`policy_action_horizon 50`).

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` → `front` | `(1, 3, 480, 640)` | LeIsaac's fixed `front` camera (the scene's `top` sensor) |
| in | `observation.images.camera2` → `wrist` | `(1, 3, 480, 640)` | gripper-mounted camera |
| in | `observation.state` | `(1, 6)` | joint positions in LeIsaac motor units |
| in | `task` | str | `"Grab orange and place into plate"` |
| out | action chunk | `(50, 6)` | absolute joint targets in LeIsaac motor units |

### Units — LeIsaac's SO-101 motor map (`joint_calibration`)

LeIsaac does not record plain degrees. `convert_leisaac_action_to_lerobot`
(`leisaac/utils/robot_utils.py`) maps each joint's USD angle onto the real motor's
normalized range, `lerobot = (deg − jl0) / (jl1 − jl0) · (ml1 − ml0) + ml0`, with the
USD limits `jl` and motor ranges `ml` below, for both the action and the
observation state. The manifest declares it as `action_contract.joint_calibration`
(`policy = scale · robot + offset`, robot = arm radians / the normalized `[0, 1]`
jaw), which `PolicyIOCodec` applies on the deploy runner and, on the `isaac_sim`
scene (`robot_unit_io`), in `openral sim run`:

| Joint | USD limits (deg) | Motor range | `scale` | `offset` |
| --- | --- | --- | ---: | ---: |
| `shoulder_pan` | −110 … 110 | −100 … 100 | 52.08707 | 0 |
| `shoulder_lift` | −100 … 100 | −100 … 100 | 57.29578 | 0 |
| `elbow_flex` | −100 … 90 | −100 … 100 | 60.31135 | 5.263158 |
| `wrist_flex` | −95 … 95 | −100 … 100 | 60.31135 | 0 |
| `wrist_roll` | −160 … 160 | −100 … 100 | 35.80986 | 0 |
| `gripper` | −10 … 100 | 0 … 100 | 90.90913 | 9.090909 |

The gripper's robot unit is the jaw fraction the Isaac scene opens over the URDF's
0 … 1.74533 rad (0 … 100.0°). `robots/so101_follower`'s URDF carries the same
`so101_new_calib` joints as LeIsaac's `so101_follower.usd` (identical limits; zero
pose = LeIsaac's init), so no sign or zero correction is folded in. The jaw's
−10 … 0° tail (training reads it closed at ≈ −7.5°) is outside the manifest's
`[0, 1]` gripper envelope, so commands there clamp to the closed end.

### Time base — 60 Hz

LeIsaac steps its env once per Isaac Lab physics dt (1/60 s, decimation 1) and
records and evaluates one action per step; the dataset's `fps: 30` label does not
change that. The manifest declares `action_contract.control_freq_hz: 60` and the
scene runs the SO-101 at 60 Hz (`backend_options.control_freq_hz`).

## Upstream model and training

| Field | Value |
| --- | --- |
| Weights repo | [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange) (pinned `@71cf4a9`) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA* |
| License | `apache-2.0` (model card) |
| Training data | [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) — 60 sim teleop episodes, 36 293 frames, 1 task |
| Training | 30 000 steps, batch 56, AdamW lr 1e-4, cosine decay, expert-only, lerobot 0.4.5 |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| SO-101 follower | `so101_follower` | ⚡ experimental | Simulation only (`scenes/sim/isaac_leisaac_pick_orange.yaml`). A real 30 Hz SO-101 is refused by the runner's trained-rate check (`control_freq_hz: 60`). |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 (trained 640 × 480) | `float32` |
| `observation.images.camera2` | RGB | 224 × 224 (trained 640 × 480) | `float32` |
| `observation.state` | proprioception | `(6,)` | `float32` |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-smolvla-so101-leisaac_pick_orange-bf16` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s1` / `vla` |
| `model_family` | `smolvla` |
| `embodiment_tags` | `so101_follower` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://edge-inference/smolvla-so101-pick-orange@71cf4a9` |
| `chunk_size` / `n_action_steps` | 50 / 50 (no prefetch, no RTC) |
| `action_contract` | 6-D `joint_positions`, `joint_calibration` (LeIsaac motor units), `control_freq_hz: 60` |
| `default_prompt` | `Grab orange and place into plate` |
| `latency_budget.per_chunk_ms` | 400 |
| `min_vram_gb.bf16` | 1.5 (same graph as the eraser skill; not re-measured) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Evaluation

No eval JSON shipped yet: no OpenRAL rollout of this checkpoint has completed.
Reference point (LeIsaac's own harness on Isaac Sim 5.1, not OpenRAL): about 12–16 %
success with LeIsaac's front-camera randomization, which random-walks the camera
across resets; with the front camera held at its nominal mount, 3/4. The OpenRAL
scene keeps the front camera at that nominal mount with no randomization.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0**. The weights stay
upstream at `edge-inference/smolvla-so101-pick-orange` under the license their
model card declares (`apache-2.0`).

## See also

- [`scenes/sim/isaac_leisaac_pick_orange.yaml`](../../scenes/sim/isaac_leisaac_pick_orange.yaml) — the paired Isaac scene.
- [`rskills/rskill-smolvla-so101-eraser_place-bf16`](../rskill-smolvla-so101-eraser_place-bf16/) — sibling SO-101 SmolVLA skill (real teleop, degrees).
- [`robots/so101_follower/robot.yaml`](../../robots/so101_follower/) — RobotDescription manifest.
