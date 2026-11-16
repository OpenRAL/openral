---
language:
- en
license: other
license_name: flux-non-commercial-license
pipeline_tag: object-detection
tags:
- OpenRAL
- rskill
- detector
- object-detection
- any
- rskill-kind:detector
- rskill-role:s1
- embodiment:any
- rskill-license:flux_non_commercial
inference: false
openral:
  description: Open Vision Banana 4B (zero-shot) — FLUX.2 [klein] fine-tuned to paint
    perception answers — packaged as an on-demand referring-segmentation locator.
    Given an RGB frame and a free-text description it paints the matching region,
    which is decoded into one box per connected region. No actuators.
  model_family: null
  min_vram_gb: 17.0
  latency_ms: 30000.0
  runtime: pytorch
  dtype: bf16
---

# rskill-open_vision_banana_4b-any-refseg-bf16

> **OpenRAL rSkill** — [Open Vision Banana](https://openvisionbanana.github.io/)
> 4B zero-shot (FLUX.2 [klein] fine-tuned to paint perception answers),
> packaged as an **on-demand referring-segmentation locator**. Perception only.
> **No actuators.** The upstream weights are under the **FLUX Non-Commercial
> License**.

## Quick Start

The model runs in its own sidecar venv (`diffusers==0.39.0`,
`transformers==4.57.6`), built on first use by `tools/ovb_sidecar.py`. The
detector node only needs the ZMQ wire:

```bash
just sync --group ovb
OPENRAL_ALLOW_NONCOMMERCIAL=1 openral deploy sim --config scenes/deploy/<scene>.yaml \
  --object-detector-locator open-vision-banana-4b
```

The reasoner then reaches it through `locate_in_view` with `detector:
open-vision-banana-4b`; its query is used as the referring expression.

## Upstream model / training

- **Model:** Open Vision Banana 4B zero-shot, checkpoint
  `final_models/4B/zeroshot/checkpoint-87500-95000-w25/transformer` of
  [`FunAILab/Open-Vision-Banana`](https://huggingface.co/FunAILab/Open-Vision-Banana)
  @ `be5da66`.
- **Architecture:** a fine-tuned FLUX.2 [klein] transformer loaded into the
  base [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
  pipeline @ `a3b4f48` (Qwen3 text encoder, VAE, scheduler).
- **Training:** public data only; the zero-shot mix excludes the evaluation
  benchmarks' training splits. Code: [OpenVisionBanana/OpenVisionBanana](https://github.com/OpenVisionBanana/OpenVisionBanana).

## Supported robots / embodiments

Any (`embodiment_tags: [any]`): a perception rSkill that needs only an RGB
camera and commands no actuators.

## Sensors / observation contract

One RGB frame, at least 256 × 256. The frame's longest edge is capped at
1024 px before generation; boxes are reported in the frame's own pixel space.

## Manifest summary

| Field | Value |
| --- | --- |
| `kind` / `role` | `detector` / `s1` |
| `detector.engine` / `mode` | `ovb_sidecar` / `on_demand` |
| `runtime` / dtype | `pytorch` / `bf16` |
| `license` | `flux_non_commercial` |
| `latency_budget.per_chunk_ms` | 30000 (placeholder, not yet measured) |

## How it answers

1. The query becomes OVB's upstream referring-segmentation prompt
   (`[TASK: REFERRING_SEGMENTATION] … Paint the matching region red [220, 30, 30] …`).
2. The sidecar runs 28 denoising steps (guidance 3.5, seed 0) on the frame
   (longest edge capped at 1024 px) and returns the painted image.
3. The detector node resizes it back with nearest-neighbour, snaps each pixel
   to the nearer of the paint colour or black (upstream's evaluation rule), and
   emits one box per 4-connected region larger than 0.1 % of the frame.

Every box carries `confidence=1.0`: OVB emits no score.

## Limits

- **Slow.** One query is a full diffusion run — seconds, not milliseconds.
  It is on-demand only and never the continuous detector.
- **VRAM.** ~16 GB of bf16 weights (transformer 7.75 GB + Qwen3 text encoder
  8.0 GB). The sidecar defaults to diffusers model CPU offload. Not yet
  measured on OpenRAL hardware.
- **Depth is not used.** OVB's depth output is relative inverse depth, never
  metric; this rSkill runs only the referring-segmentation task.

## License

- **Code:** Apache-2.0 (OpenRAL and upstream OVB).
- **Weights:** FLUX Non-Commercial License (`flux_non_commercial`). The loader
  refuses a commercial deployment unless `OPENRAL_ALLOW_NONCOMMERCIAL=1`.

## Citation

Open Vision Banana: A Fully Open Recipe for Generative Visual Perception,
arXiv:2611.15844.
