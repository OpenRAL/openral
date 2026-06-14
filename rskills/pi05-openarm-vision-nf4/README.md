---
tags:
  - OpenRAL
  - rskill
  - pi05
  - lerobot
  - vla
  - openarm
  - bimanual
  - manipulation
license: other
license_name: permissive-research
language:
  - en
---

# pi05-openarm-mddoai-vision

OpenArm v2 bimanual π0.5 rSkill. Source checkpoint: `mddoai/pi05_openarm_vision`.
Sister manifest of `pi05-openarm-mddoai-vast` —
identical dataset (`mddoai/openarm_2026-05-14_clean`, 89 episodes,
LEFT-FIRST 16-D state/action in radians) but trained with
`use_relative_actions: false`, so every action channel is an **absolute**
joint-position target instead of vast's per-arm deltas.

## Provenance

| Artefact | URI | License | Notes |
| --- | --- | --- | --- |
| Base policy | `hf://lerobot/pi05_base` | `license:gemma` (research use) | PaliGemma 3B + 300M action expert. |
| Trained checkpoint | `hf://mddoai/pi05_openarm_vision` | apache-2.0 | Full fine-tune; `use_peft: false`, `use_relative_actions: false`. |
| Pre-quantized NF4 | `hf://OpenRAL/rskill-pi05-openarm-vision-nf4` | apache-2.0 (wrapper) | Produced by `tools/quantize_rskill.py --scheme nf4`. Runtime-loadable. |
| Pre-quantized int8 | `hf://OpenRAL/rskill-pi05-openarm-vision-int8` | apache-2.0 (wrapper) | Produced by `tools/quantize_rskill.py --scheme int8`. **Upload-only artefact** — the pi05 adapter does not yet wire the int8 fast-path. |
| Training dataset | `hf://datasets/mddoai/openarm_2026-05-14_clean` | apache-2.0 | 89 episodes, 39k frames, bi_openarm, fps=15. |
| Embodiment target | `robots/openarm/robot.yaml` | apache-2.0 | OpenArm v2 MJCF; matches the dataset's 16-D state layout. |

## Supported robots / embodiments

Targets the `openarm` embodiment (OpenArm v2 bimanual; see
[`robots/openarm/robot.yaml`](../../robots/openarm/robot.yaml)). The
manifest's `embodiment_tags: ["openarm"]` ties the rSkill to that
single robot — running on a non-openarm host fails the loader's
embodiment intersection check.

## Sensors / observation contract (from on-Hub `config.json`)

- **Cameras**: 3 × RGB, `observation.images.{base,left_wrist,right_wrist}`, native 480×640, downscaled to 224×224 by the policy preprocessor (`image_resolution: [224, 224]`).
- **State**: 16-D `observation.state`, **left-first** layout = `[left_joint1..7, left_gripper, right_joint1..7, right_gripper]`.
- **Action**: 16-D **absolute joint-position targets**, same left-first layout. `use_relative_actions: false` — no delta accumulation in the postprocessor. Consumed directly by the OpenArm joint-position controller.
- **Chunk**: 50 actions, 10 flow-matching denoising steps per chunk.
- **Normalization**: QUANTILES on STATE + ACTION; IDENTITY on VISUAL. Stats live in `policy_preprocessor_step_2_normalizer_processor.safetensors`.

## Difference vs the vast sibling

| Field | `…-mddoai-vast` | `…-mddoai-vision` (this) |
| --- | --- | --- |
| `use_relative_actions` | `true` (arms deltas, grippers absolute) | `false` (all absolute) |
| `control_mode_semantics.mode` | `delta` | `absolute` |
| Postprocessor effect | `absolute_actions_processor` rolls deltas → absolutes | no-op (values already absolute) |
| Joint order | left-first | left-first |
| `chunk_size` | 50 | 50 |
| Dataset | `mddoai/openarm_2026-05-14_clean` | `mddoai/openarm_2026-05-14_clean` |
| `state_contract.dim` / `action_contract.dim` | 16 / 16 | 16 / 16 |

The reasoner sees both as `openarm`-compatible — selection is driven by the
LLM's tool-call argument (`skill_id`).

## Quantization

The on-Hub `mddoai/pi05_openarm_vision` ships bf16 weights only. Two
mirrored packs live under the OpenRAL org for faster cold-load:

```bash
# Pre-flight: HF auth (write scope) + ~30 GB free disk during the pipeline.
export HF_TOKEN=hf_xxx       # write scope; required for `--target` upload
df -h ~                      # confirm ≥30 GiB free
nvidia-smi                   # confirm a CUDA GPU is visible (bitsandbytes
                             # only packs nf4 / int8 on CUDA)

# 1. NF4 — drop-in replacement for the on-line nf4 path.
uv run python tools/quantize_rskill.py \
    --source mddoai/pi05_openarm_vision \
    --target OpenRAL/rskill-pi05-openarm-vision-nf4 \
    --scheme nf4 \
    --policy-class lerobot.policies.pi05.modeling_pi05.PI05Policy

# 2. int8 — upload-only artefact; the pi05 adapter cannot yet consume it.
uv run python tools/quantize_rskill.py \
    --source mddoai/pi05_openarm_vision \
    --target OpenRAL/rskill-pi05-openarm-vision-int8 \
    --scheme int8 \
    --policy-class lerobot.policies.pi05.modeling_pi05.PI05Policy
```

To consume the NF4 fast-path at runtime, point the manifest's
`weights_uri` at the mirrored repo:

```yaml
weights_uri: "hf://OpenRAL/rskill-pi05-openarm-vision-nf4"
```

The pi05 adapter's `load_prequantized_state_for_rskill` will then
detect the `quantization_metadata.json` sentinel and overlay the
packed `Linear4bit` state via `install_prequantized_linears`,
skipping the ~30 s on-line bf16→nf4 conversion.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-openarm-vision-nf4` |
| `model_family` | `pi05` |
| `embodiment_tags` | `openarm` |
| `quantization.dtype` | `int4` (bitsandbytes NF4, bf16 compute) |
| `weights_uri` | `hf://OpenRAL/rskill-pi05-openarm-vision-nf4` |
| `state_contract.dim` / `action_contract.dim` | 16 / 16 (left-first) |
| `chunk_size` / `n_action_steps` | 50 / 25 |

See [`rskill.yaml`](rskill.yaml) for the full manifest the loader
validates.

## License

| Component | Licence | Posture |
| --- | --- | --- |
| Wrapper (this manifest + README) | `apache-2.0` | Full commercial reuse. |
| NF4 weights re-host | `apache-2.0` | Derived from the upstream `mddoai/pi05_openarm_vision` checkpoint (apache-2.0) via lossless quantization. |
| Base policy `lerobot/pi05_base` | Gemma research licence (`license:gemma`) | Non-commercial research use only. Commercial deployment of any π0.5 derivative requires a separate Physical Intelligence agreement. The OpenRAL loader derives `is_commercial_use_allowed=False` from the manifest's effective license; commercial activation still requires `OPENRAL_ALLOW_NONCOMMERCIAL=1` plus the vendor agreement. |

## Known gaps

- `eval/` is intentionally empty until a real `openral benchmark run`
  produces a `reproduced_locally: true` result.
- `min_vram_gb.int4: 4.0` is conservative; refresh after the first
  successful chunk inference on the 4070-mobile.
- The int8 mirror is upload-only — the loader does not detect the
  `int8` scheme today, so pointing `weights_uri` at the int8 repo
  will fail with a missing-`Linear8bitLt`-installer error. Use the
  nf4 mirror for runtime work.
