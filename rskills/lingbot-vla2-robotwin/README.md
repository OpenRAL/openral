---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- lingbot_vla2
- vision-language-action
- nf4
- 4-bit
- aloha_agilex
- lingbot-vla2
- vla
- robotwin
- bimanual
- manipulation
inference: false
---

# rskill-lingbot-vla2-robotwin

> **OpenRAL rSkill** — Robbyant **LingBot-VLA 2.0** (6.38 B, Qwen3-VL-4B backbone +
> sparse-MoE flow-matching action expert) for dual-arm manipulation on the
> **RoboTwin 2.0** / AgileX "Cobot Magic" embodiment, packaged for use with the
> [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps
[`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
(upstream code: [`github.com/robbyant/lingbot-vla-v2`](https://github.com/robbyant/lingbot-vla-v2)
@ `69729b4`) with an `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does **not** copy
model weights.

## What this skill does

A multi-task dual-arm policy for the RoboTwin 2.0 benchmark on the AgileX
"aloha-agilex" embodiment. Predicts 50-step action chunks across three RGB views
(head + per-wrist) driving a 14-DoF dual-arm joint command, replaying 25 steps
before re-inferring (matching the upstream `--use_length 25` deploy).

| Field | Value |
| --- | --- |
| Actions | `generalist`, `pick`, `place`, `transfer`, `pour` |
| Objects | `block`, `bottle`, `cup`, `bowl` |
| Scenes  | `tabletop` |
| Embodiment | `aloha_agilex` (AgileX Cobot Magic, dual-arm) |
| Action space | 14-D joint position (2×(6 arm + 1 gripper)) |
| Cameras | `camera1` (head), `camera2` (left wrist), `camera3` (right wrist), 256×256 |

> **Preview.** A demo clip is added in Phase 3 once the 25 GB checkpoint download
> completes and the skill has been exercised end-to-end on a GPU host. Until then
> this rSkill is packaging-only (no local rollout has been run).

## Upstream model / how it works

LingBot-VLA 2.0 is a 6.38 B model: a `Qwen3-VL-4B-Instruct` vision-language
backbone feeds a sparse **mixture-of-experts** flow-matching **action expert**
(10 denoising steps) that emits a 14-D action per step. OpenRAL loads the upstream
`LingbotVLAv2Server` through the out-of-process `lingbot_vla2` adapter
(`openral_sim.policies.lingbot_vla2`): the model returns the unnormalized
`action.arm.position` (12) + `action.effector.position` (2) per step, the sidecar
flattens them to a `(50, 14)` chunk, and the adapter replays 25 of those 14-D
steps per re-inference.

The upstream `lingbotvla` package pins `torch==2.8.0` / `transformers==4.57.3` /
`triton==3.4.0` and ships custom Triton MoE kernels, which cannot coexist with the
OpenRAL `torch>=2.9` / `transformers>=5` runtime (CLAUDE.md §3). The model
therefore runs in an **auto-provisioned Python 3.12 + torch-2.8 ZMQ sidecar**
(`tools/lingbot_vla2_sidecar.py` clones the pinned upstream repo + builds the venv;
`tools/_lingbot_vla2_server.py` serves `ping`/`reset`/`get_action`/`close`). Because
`flash-attn` is not installed, the server coerces the upstream `flash_attention_2`
hardcode to `sdpa` (or `eager`).

### Sensors / observation contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` | `(256, 256, 3)` RGB uint8 | Head / overhead view → upstream `cam_high`. |
| in | `observation.images.camera2` | `(256, 256, 3)` RGB uint8 | Left wrist view → upstream `cam_left_wrist`. |
| in | `observation.images.camera3` | `(256, 256, 3)` RGB uint8 | Right wrist view → upstream `cam_right_wrist`. |
| in | `observation.state` | `(14,) float32` | aloha-agilex dual-arm joint state `[arm_l(6) grip_l(1) arm_r(6) grip_r(1)]`, radians. |
| out | action chunk | `(50, 14) float32` | Absolute dual-arm joint position commands (25 replayed per re-inference). |

## How it was trained

LingBot-VLA 2.0 is Robbyant's second-generation VLA, described in *"From
Foundation to Application: Improving VLA Models in Practice"*. This rSkill is a
thin wrapper — the weights live upstream and are not copied here.

| Field | Value |
| --- | --- |
| Source repo | [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b) |
| Upstream code | [`github.com/robbyant/lingbot-vla-v2`](https://github.com/robbyant/lingbot-vla-v2) @ `69729b4` |
| Base backbone | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Paper | [huggingface.co/papers/2607.06403](https://huggingface.co/papers/2607.06403) |
| License | apache-2.0 (code **and** weights) |
| Parameters | 6.38 B (fp32 safetensors, ~25.5 GB on disk) |
| Embodiment | RoboTwin 2.0 / AgileX Cobot Magic (dual-arm) |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| AgileX Cobot Magic (RoboTwin 2.0) | `aloha_agilex` | ⚡ experimental | Packaging validated; end-to-end rollout is Phase 3 (pending the 25 GB weight download + GPU run). |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-lingbot-vla2-robotwin` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `model_family` | `lingbot_vla2` |
| `embodiment_tags` | `aloha_agilex` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (bitsandbytes NF4 backbone, bf16 expert) |
| `weights_uri` | `hf://robbyant/lingbot-vla-v2-6b` |
| `state_contract.dim` / `action_contract.dim` | `14` / `14` |
| `chunk_size` / `n_action_steps` | `50` / `25` |
| `min_vram_gb` | fp32 `25.5`, bf16 `12.8`, int4 `7.0` (**int4 unverified — estimate**) |
| `latency_budget.per_chunk_ms` | `2000.0` (generous placeholder; ~130 ms on a 4090D reference) |
| `evaluated_tasks` | `robotwin` |
| `commercial_use_allowed` | `true` (Apache-2.0) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill

pkg = rSkill.from_yaml("rskills/lingbot-vla2-robotwin/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.version)  # OpenRAL/rskill-lingbot-vla2-robotwin 0.1.0
```

## Reproduction

The 6.38 B model runs out-of-process in an auto-provisioned Python 3.12 + torch-2.8
sidecar; the OpenRAL side only needs the ZMQ wire:

```bash
# openral-side wire (pyzmq + msgpack)
just sync --all-packages --group lingbot --inexact

# the sidecar auto-clones github.com/robbyant/lingbot-vla-v2 @ 69729b4 and builds
# its torch-2.8 venv on first use; the RoboTwin 5-task suite (Phase 3):
openral benchmark run --suite robotwin --vla lingbot_vla2:rskills/lingbot-vla2-robotwin
```

## Evaluation

No locally-reproduced numbers are shipped yet — `eval/` is empty (`.gitkeep`).
Phase 3 populates it via `openral benchmark run --suite robotwin` on the GPU eval
host.

Paper-cited results (GM-100 benchmark, **`reproduced_locally: false`**):

| Benchmark | Embodiment | Success | `reproduced_locally` | Source |
| --- | --- | --- | --- | --- |
| GM-100 | AgileX | 34.4% | false | paper ([huggingface.co/papers/2607.06403](https://huggingface.co/papers/2607.06403)) |
| GM-100 | Galaxea | 15.6% | false | paper ([huggingface.co/papers/2607.06403](https://huggingface.co/papers/2607.06403)) |

Reproduction command (Phase 3):
`openral benchmark run --suite robotwin --vla lingbot_vla2:rskills/lingbot-vla2-robotwin`.

> **8 GB fit is UNVERIFIED.** The `int4` (NF4 backbone + bf16 expert) footprint is
> an estimate (~7 GB); it has not been measured. bf16 (12.8 GB) needs a ≥16 GB
> card and fp32 (25.5 GB) a ≥32 GB card. Phase 3 measures the real NF4 peak.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0**. The wrapped
weights at `hf://robbyant/lingbot-vla-v2-6b` and the upstream `lingbotvla` code are
also **Apache-2.0** (code and weights), so commercial use is permitted. The package
does not copy weights into this repository; runtime loading still emits OpenRAL's
unverified-provenance warning until the planned signing control exists.

## See also

- `robots/aloha_agilex/robot.yaml` — RobotDescription manifest (dual-arm AgileX).
- `rskills/smolvla-robotwin/` — sibling SmolVLA rSkill on the same embodiment.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
