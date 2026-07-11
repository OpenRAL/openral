<!-- OpenRAL rSkill README — LingBot-VLA 1.0 (4B) RoboTwin post-train -->

# rskill-lingbot-vla-4b-robotwin

**LingBot-VLA 1.0 (4 B)** — a Qwen2.5-VL-3B backbone with a *dense* Qwen2
flow-matching action expert (10 denoising steps), **post-trained on RoboTwin** for
the AgileX "Cobot Magic" dual-arm embodiment. Paper: *"A Pragmatic VLA Foundation
Model"* ([arXiv 2601.18692](https://arxiv.org/abs/2601.18692)); code + weights
Apache-2.0 ([github.com/robbyant/lingbot-vla](https://github.com/robbyant/lingbot-vla),
[robbyant/lingbot-vla-4b-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin)).

Unlike the v2 **6 B foundation** checkpoint (`rskill-lingbot-vla2-robotwin`), which
mean-collapses zero-shot, this is a **task post-train**: it reproduces ground-truth
RoboTwin actions and drives the arms to the object. See *Verified performance*.

## Preview

![lift_pot demo](media/lingbot_vla_4b_robotwin_lift_pot.gif)

*RoboTwin 2.0 `lift_pot` (SAPIEN), aloha-agilex — the model drives both arms to the
pot's side handles (bimanual approach) from three RGB views.*

## What this skill does

An **S1** vision-language-action policy: from three RGB views (head + per-wrist) and
a 14-D dual-arm proprio state it predicts a **50-step action chunk** of 14-D
absolute joint commands; the adapter replays 25 steps before re-inferring. It is the
RoboTwin specialization of the LingBot-VLA foundation model (20 k h of real dual-arm
data across 9 configs, then RoboTwin post-training).

## How it works

The upstream V1 `lingbotvla` package pins `torch==2.8.0` / `transformers==4.51.3` /
`lerobot==0.4.2` (flat layout), incompatible with the OpenRAL workspace
(`torch>=2.9` / `transformers>=5`, CLAUDE.md §3). So the model runs **out-of-process**
in an auto-provisioned Python 3.12 ZMQ sidecar, driven by the SAME `lingbot_vla2`
adapter as the v2 model via a `--variant v1` switch:

- boot helper `tools/lingbot_vla2_sidecar.py --variant v1` clones
  `github.com/robbyant/lingbot-vla @ 4eb34b7` + builds the V1 venv on first use;
- server `tools/_lingbot_vla2_server.py --variant v1` (`_LingBotV1Policy`) loads the
  upstream `LingbotVLAServer`, NF4-quantizes the Qwen2.5-VL backbone in place
  (keeping `o_proj` bf16 — see note), and answers `ping`/`reset`/`get_action`/`close`.

Three server-side patches make the flash-free (eager) path correct: (1) inject the
missing `rotate_half` into the eager vision attention; (2) force
`attn_implementation=eager` (the custom Qwen2.5-VL attention registers no `sdpa`
kernel and flash-attn is not installed); (3) skip `o_proj` in NF4 — the interleaved
attention forward reads `o_proj.weight.dtype` to cast activations, which a packed
`Linear4bit` (uint8) would corrupt to Byte.

### Observation → action contract

Verified against `configs/robot_configs/robotwin.yaml`,
`assets/norm_stats/robotwin_50.json` (bounds_99), and the shipped `lingbotvla_cli.yaml`:

| | shape / keys |
|---|---|
| cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` — HWC uint8, resized to 224×224 server-side by the model's `FeatureTransform` |
| state | 14-D `[arm_l(6) grip_l(1) arm_r(6) grip_r(1)]` (radians + 0–1 grippers) |
| action | 14-D per step (`action.arm.position` 12 + `action.effector.position` 2) |
| chunk | 50 predicted; 25 replayed before re-inference |

## Verified performance (RTX 4070 Laptop, 8 GB)

**Open-loop vs GT** (RoboTwin `lift_pot` ep0, NF4 backbone, eager, 10 denoising
steps) — the decisive check the v2 foundation checkpoint failed:

- `action[0]` tracks **state**, not the action mean: `|pred[0]−state|` = 0.004–0.038
  vs `|pred[0]−action_mean|` = 0.16–0.65 at every probed timestep. **Not mean-collapsed.**
- mean absolute error **0.023** over the chunk (per-timestep 0.013–0.028);
  within-chunk jitter ratio pred/GT = **1.24** (as smooth as the GT demo).

**Resources (measured live):**

- **VRAM 3.65 GB** resident (NF4 backbone + bf16 `o_proj`/expert). Fits an 8 GB card
  with ~4 GB free — enough to **co-reside with the SAPIEN RoboTwin sim** (~1.9 GB).
- **~0.57 s** per 50-step chunk inference on the 4070 (eager, no flash-attn).

**Closed-loop RoboTwin `lift_pot` (SAPIEN, aloha-agilex):** **1/10 = 10 %** success
over 10 episodes (seed 0, 300 steps) — run live on the 8 GB 4070 with the NF4 model
(3.65 GB) and the SAPIEN sim (~1.9 GB) **co-resident on one GPU**, ~26 ms mean step
latency. The robot drives both arms to the pot's side handles (correct bimanual
approach) on every episode and completes the lift on the successful one. Details in
`eval/scene_robotwin.json` (`reproduced_locally: true`, exact `reproduction_cli`).
lift_pot is a hard bimanual task; the open-loop MAE above confirms the policy is
sound, and this is an honest NF4 + eager + laptop-GPU + single-seed rate (not
paper-reported).

## How it was trained

Upstream (not by OpenRAL): LingBot-VLA foundation pre-training on 20 k h of
real-world data from 9 dual-arm configs, then RoboTwin post-training on the
aloha-agilex embodiment. Loss `L1_fm` (flow matching), `bounds_99` action/state
normalization. See the paper and upstream repo.

## Supported robots

`aloha_agilex` (RoboTwin 2.0 / AgileX Cobot Magic, dual-arm 14-DoF). The three scene
cameras are aligned positionally onto the model's `cam_high` / `cam_left_wrist` /
`cam_right_wrist` by the adapter.

## Manifest summary

| field | value |
|---|---|
| `model_family` | `lingbot_vla` |
| `embodiment_tags` | `aloha_agilex` |
| `quantization` | `int4` (bitsandbytes NF4, backbone; `o_proj`/expert bf16) |
| `min_vram_gb` | fp32 16.8 / bf16 8.4 / **int4 4.5** |
| `chunk_size` / `n_action_steps` | 50 / 25 |
| `latency_budget.per_chunk_ms` | 1000 (measured ~570 ms steady) |
| `weights_uri` | NF4 mirror (prequant pack) — see below |

## Quick start

```bash
# The RoboTwin scene sidecar (SAPIEN) + the LingBot V1 policy sidecar co-reside on
# one 8 GB GPU. Point the escape-hatch envs at a provisioned checkout/venv/ckpt, or
# let the boot helper provision them on first use.
openral benchmark scene \
  --config scenes/benchmark/robotwin_lift_pot.yaml \
  --rskill rskills/lingbot-vla-4b-robotwin \
  --n-episodes 10 --save-video out/ --write-eval
```

## Reproduction

`eval/robotwin.json` is produced by the command above (`reproduced_locally: true`);
the exact `reproduction_cli` is recorded in that file. Open-loop numbers come from
the GT-vs-prediction probe described under *Verified performance*.

## License

Apache-2.0 (OpenRAL wrapper). Upstream LingBot-VLA code **and** weights are
Apache-2.0 — no weight-license guard applies (CLAUDE.md §1.9).

## See also

- `rskills/lingbot-vla2-robotwin` — the v2 6 B foundation checkpoint (same adapter).
- `python/sim/src/openral_sim/policies/lingbot_vla2.py` — the shared adapter (`lingbot_vla` id).
- `tools/_lingbot_vla2_server.py` (`_LingBotV1Policy`) — the V1 loading path.
