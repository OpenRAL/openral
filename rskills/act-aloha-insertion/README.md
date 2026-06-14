---
tags:
  - OpenRAL
  - rskill
  - act
  - lerobot
  - aloha
  - bimanual
  - manipulation
  - insertion
license: mit
language:
  - en
---

# rskill-act-aloha-insertion

> **OpenRAL rSkill (custom example)** — ACT (Action Chunking Transformer)
> finetuned on the ALOHA bimanual **peg-insertion** task, packaged for
> `OpenRAL`.

This package wraps
[`lerobot/act_aloha_sim_insertion_human`](https://huggingface.co/lerobot/act_aloha_sim_insertion_human)
with a `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does
**not** copy model weights.

It is the harder sibling of [`rskill-act-aloha`](../act-aloha) (cube
transfer) and demonstrates how a single packaging format covers multiple
task-specific checkpoints from the same paper. The runnable demo lives at
`scenes/benchmark/aloha_insertion.yaml` and is wired into the
top-level `just sim-custom` recipe.

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/act_aloha_sim_insertion_human`](https://huggingface.co/lerobot/act_aloha_sim_insertion_human) |
| Architecture | Action Chunking Transformer (~52M params, chunk=100) |
| Task | gym-aloha `AlohaInsertion-v0` (bimanual peg-in-socket) |
| License | MIT |
| Paper | Zhao et al., 2023 — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* ([arXiv 2304.13705](https://arxiv.org/abs/2304.13705)) |

## Why no `eval/` block?

This skill is shipped as a **custom-example** package, not as a
reproduced benchmark entry. The paper's headline number for sim ALOHA
insertion is markedly lower than the cube-transfer figure (the task is
harder and the upstream protocol uses different camera intrinsics). We
deliberately omit `eval/` rather than copy paper numbers without an
internal reproduction; per CLAUDE.md §6.4 that omission must be
documented — this section is that documentation. Add `eval/aloha_insertion.json`
once a local reproduction lands.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| ALOHA bimanual (Trossen) — `gym-aloha` MuJoCo | `aloha`, `lerobot` | ✓ sim | 14-DoF (2 × 7-DoF arms with parallel grippers); MuJoCo MJX `AlohaInsertion-v0`. |

Same physical embodiment as the [`act-aloha`](../act-aloha/) sibling
(cube transfer); the only difference is the task contact dynamics — peg
insertion is harder than cube pick-and-place.

## Sensors required

| Key | Modality | Resolution | Format |
| --- | --- | --- | --- |
| `observation.images.top` | RGB camera | 640 × 480 | `float32` |
| `observation.state`      | proprioception | (14,) | `float32` (2 × 7-DoF joint positions) |

Single top-down RGB stream like the cube-transfer sibling — the
checkpoint does not consume wrist or third-person views.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-act-aloha-insertion` |
| `version` | `0.1.0` |
| `license` | `mit` |
| `role` | `s1` |
| `embodiment_tags` | `aloha`, `lerobot` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://lerobot/act_aloha_sim_insertion_human` |
| `chunk_size` | 100 |
| `commercial_use_allowed` | `true` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Run it

```bash
just sim-custom
```

…which is equivalent to:

```bash
MUJOCO_GL=egl uv run --group sim openral sim run \
  --config scenes/benchmark/aloha_insertion.yaml \
  --save-video example_videos
```

## License

This rSkill package (`rskill.yaml`, `README.md`) is **MIT** to match
the upstream weights. Commercial use is allowed
(`commercial_use_allowed: true`).
