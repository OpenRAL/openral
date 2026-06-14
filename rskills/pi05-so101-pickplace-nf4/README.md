---
tags:
  - OpenRAL
  - rskill
  - pi05
  - lerobot
  - vla
  - so101
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-pi05-so101-pickplace-nf4

This package wraps `hf://HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b`
as a pre-quantized (nf4) OpenRAL rSkill. It does **not** copy model
weights — the manifest points at the mirrored nf4 pack on the Hub.

## What this skill does

A π0.5 (PaliGemma + action-expert) vision-language-action policy
finetuned for **pick-and-place** on the SO-101 follower arm. Given two
RGB views (overhead + wrist) and the 6-D joint state, it emits absolute
6-DoF joint-position action chunks of length 50.

| Field | Value |
| --- | --- |
| Actions | pick, place, pick_and_place |
| Objects | diverse tabletop objects |
| Scenes  | tabletop |
| Embodiment | so101_follower |

## How it works

π0.5 runs a frozen-ish PaliGemma 3 B backbone plus a small action
expert that decodes a flow-matching action chunk conditioned on the
image tokens, the language instruction, and the proprioceptive state.
This rSkill ships the post-quantization nf4 state dict and a
`quantization_metadata.json` sentinel; the pi05 adapter detects it,
meta-initialises the graph, overlays the packed weights, and skips the
on-line bf16→nf4 conversion, so the 4.14 B graph fits in ~4 GiB of VRAM.

### Observation → action contract

| dir | key | shape | notes |
| --- | --- | --- | --- |
| in | `observation.images.top` | `(1, 3, H, W)` float32 [0,1] | overhead view ← scene `oak_top` |
| in | `observation.images.wrist` | `(1, 3, H, W)` float32 [0,1] | gripper-mounted view ← scene `wrist` |
| in | `observation.images.front` | `(1, 3, H, W)` float32 [0,1] | optional; zero-padded via image mask when absent |
| in | `observation.state` | `(1, 6)` float32 | SO-101 joint positions (rad) |
| out | action chunk | `(50, 6)` float32 | absolute joint-position targets |

## How it was trained / provenance

| Field | Value |
| --- | --- |
| Source repo | [`HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b`](https://huggingface.co/HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b) |
| Base model  | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) (Physical Intelligence π0.5) |
| Paper       | [arxiv:2410.24164](https://arxiv.org/abs/2410.24164) — *π0.5: a VLA with Open-World Generalization* |
| License     | apache-2.0 |
| Parameters  | ~4.14 B (PaliGemma backbone + action expert) |
| Training data | [`HollyTan/so101_pick-place-v2.2-100eps`](https://huggingface.co/datasets/HollyTan/so101_pick-place-v2.2-100eps) — 100 SO-101 teleop episodes |

The nf4 pack is produced from the upstream checkpoint with
`tools/quantize_rskill.py` (see the header of `rskill.yaml` for the exact
command).

## Supported robots / embodiments

| Embodiment | Cameras | Status | Notes |
| --- | --- | --- | --- |
| `so101_follower` | overhead + wrist | ⚡ experimental | 6-DoF; validated to load + step on the `so101_box` sim scene |

SO-100 shares the 6-DoF follower IO contract but is **not** claimed —
it's an unverified transfer and the canonical SO-100 detection profile
does not advertise int4 GPU support. Add `so100_follower` to
`embodiment_tags` once an SO-100 host validates the nf4 load.

## Sensors required

| Feature key | Modality | Min size | Dtype |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 | `float32` |
| `observation.images.camera2` | RGB | 224 × 224 | `float32` |
| `observation.state`          | proprioception | (6,) | `float32` |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-so101-pickplace-nf4` |
| `model_family` | `pi05` |
| `role` | `s1` |
| `license` | `apache-2.0` |
| `embodiment_tags` | `so101_follower` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (nf4) |
| `weights_uri` | `hf://OpenRAL/rskill-pi05-so101-pickplace-nf4` |

## License

Apache-2.0 — both the upstream code and the HollyTan checkpoint weights
are Apache-2.0, so commercial use is permitted (unlike the
permissive-research `pi05-so100` base wrapper).
