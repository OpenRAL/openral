---
tags:
  - openral
  - rskill
  - rldx
  - vla
  - gr1
  - humanoid
  - manipulation
  - non-commercial
license: other
license_name: rlwrld-model-license-v1.0
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
language:
  - en
---

# rskill-rldx1-ft-gr1-nf4

> RLDX-1 finetuned on the [RoboCasa GR-1 tabletop tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
> (24-task suite, Fourier GR-1 ArmsAndWaistFourierHands humanoid),
> packaged for OpenRAL.

<!-- openral:rskill-readme-delegates-to: ../rldx1-ft-libero-nf4 -->

This is a sibling of [`rldx1-ft-libero-nf4`](https://huggingface.co/OpenRAL/rskill-rldx1-ft-libero-nf4);
that README owns the canonical architecture, license, auto-managed
sidecar lifecycle, and NF4 quantization documentation for every member
of the RLDX-1 family. Read it first. The sections below cover only
this checkpoint's GR-1-specific contract.

## Run

The `rldx` adapter auto-spawns the sidecar (`--embodiment-tag GENERAL_EMBODIMENT`, the model card's own inference example) on first observation — single command:

```bash
openral sim run \
    --config scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml \
    --rskill rskills/rldx1-ft-gr1-nf4 \
    --rskill rskills/rldx1-ft-gr1-nf4 \
    --view
```

Manual boot (debug / shared host): set `OPENRAL_RLDX_AUTO_SPAWN=0` and run `python tools/rldx_sidecar.py --model RLWRLD/RLDX-1-FT-GR1 --port 5555 --quantization nf4 --embodiment-tag GENERAL_EMBODIMENT` yourself.

### Sidecar runtime knobs

* **`torch.compile` is disabled by default in the sidecar** (the boot helper sets `TORCH_COMPILE_DISABLE=1` + `TORCHINDUCTOR_DISABLE=1` in the child env). The upstream `run_rldx_server.py` doesn't request compile unless `--compile` is passed, but `bitsandbytes`/bnb-4bit can trigger inductor implicitly via `torch.dispatch`, and the post-load warmup never returns on ≤8 GiB GPUs (observed on RTX 4070-class hosts: model loads, ZMQ bind never happens). Override by exporting `TORCH_COMPILE_DISABLE=0` if you have ≥12 GiB headroom and want the steady-state speedup.
* **First-boot wait** — `boot_timeout_s` defaults to 900 s. On a fresh host with no `~/.cache/openral/rldx-sidecar/source` checkout, the upstream `uv sync` + the bf16/nf4 model download can exceed that; raise via `OPENRAL_RLDX_BOOT_TIMEOUT_S=1800` (env) or `vla.extra.boot_timeout_s` (YAML) — set on the merged manifest at [`rskills/rldx1-ft-gr1-nf4/rskill.yaml`](rskill.yaml). Subsequent boots reuse the cached venv + weights and the ~3 min ceiling drops to ~30 s.

## Action / state contract

RLDX-1-FT-GR1 is native to the Fourier GR-1 — the model card is explicit: *"RLDX-1 finetuned on the GR-1 Tabletop benchmark, a 24-task humanoid manipulation suite using the Fourier GR-1 humanoid platform"*, with the action space *"arms + waist + Fourier hands"*. The deployable contract lives in the checkpoint's `general_embodiment` modality slot (the inference example uses `EmbodimentTag.GENERAL_EMBODIMENT`); its `processor_config.json` registers five state keys with per-key dims matching the Fourier BASIC composite exactly:

| Key | Dim | Source slice (39-D openral proprio) |
|---|---|---|
| `state.waist` | 3 | `joint_pos[0:3]` |
| `state.right_arm` | 7 | `joint_pos[3:10]` |
| `state.left_arm` | 7 | `joint_pos[10:17]` |
| `state.right_hand` | 6 | `right_gripper_qpos[0:6]` (first 6 of the 11-D Fourier qpos) |
| `state.left_hand` | 6 | `left_gripper_qpos[0:6]` (first 6 of the 11-D Fourier qpos) |
| **total** | **29** | matches Fourier BASIC composite |

The action chunk is the same 5-key layout; the openral adapter concatenates the per-group columns back into the 29-D BASIC vector (`[right_arm | left_arm | waist | right_hand | left_hand]`) for `openral sim run`. Camera key: `video.ego_view`. Language key: `annotation.human.coarse_action`.

(Note: the FT-GR1 processor also carries `humanoid_everyday_g1` / `humanoid_everyday_h1` modality configs as cross-embodiment slots used during pretraining — those refer to NVIDIA's Unitree-G1/H1 "Humanoid Everyday" dataset and are **not** the deployment target. The `_g1` suffix tripped up an earlier version of this README.)

See `robots/gr1/robot.yaml` for the canonical openral embodiment manifest and `python/sim/src/openral_sim/backends/robocasa.py` for the scene contract.

Upstream: <https://huggingface.co/RLWRLD/RLDX-1-FT-GR1>
