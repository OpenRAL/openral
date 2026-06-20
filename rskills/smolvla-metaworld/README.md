---
tags:
  - OpenRAL
  - rskill
  - smolvla
  - lerobot
  - vla
  - metaworld
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-smolvla-metaworld

> **OpenRAL rSkill** — SmolVLA (0.45 B) finetuned on the
> [MetaWorld MT50](https://meta-world.github.io/) benchmark
> (50 manipulation tasks, Rethink Sawyer arm).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/smolvla-metaworld/rskill.yaml")
```

```bash
# Single demo scene (BenchmarkScene tier, paper protocol):
openral benchmark scene --config scenes/benchmark/metaworld_push.yaml \
    --rskill rskills/smolvla-metaworld

# Full headline suites (write eval/<suite>.json with reproduced_locally=true):
openral benchmark run --suite metaworld_mt10 --rskill rskills/smolvla-metaworld
openral benchmark run --suite metaworld_mt50 --rskill rskills/smolvla-metaworld
```

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/smolvla_metaworld`](https://huggingface.co/lerobot/smolvla_metaworld) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arxiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA: Efficient Vision-Language-Action Model* |
| License | Apache-2.0 |
| Parameters | ~450 M |
| Benchmark | MetaWorld MT50 (50 tasks, Rethink Sawyer) |
| Training data | `lerobot/metaworld_mt50` |

The checkpoint is **multi-task**: a single set of weights covers the whole
MetaWorld family, so the manifest gates it with the family entry
`evaluated_tasks: ["metaworld"]` (covers every `metaworld/<task>-v3` task id
and the bare `metaworld` scene id). The 4-D proprio state, the single
`observation.images.camera1` RGB input, and the normalisation statistics are
verified against the lerobot checkpoint — see `docs/reference/vla_compatibility.md` §3.2.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Rethink Sawyer (MetaWorld sim) | `sawyer` | ✓ matches | Native training embodiment. |
| Franka Panda / SO-100 | — | does **not** match | The `libero` / `so100_follower` tags are intentionally excluded; MetaWorld uses a different task distribution and camera setup. |

## Sensors required

| Key | Modality | Min resolution | Notes |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 | Mapped from MetaWorld's corner camera (`corner2`, 480×480 native). No adapter image flip — lerobot's `MetaworldEnv` already corrects the corner camera's 180° inversion. |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-smolvla-metaworld` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://lerobot/smolvla_metaworld` |
| `latency_budget.per_chunk_ms` | 150 ms |
| `evaluated_tasks` | `["metaworld"]` (family gate) |
| `commercial_use_allowed` | `true` |

Full schema: `openral_core.RSkillManifest`.

## Evaluation

Locally reproduced on the **MetaWorld MT50** suite via
`openral benchmark run --suite metaworld_mt50` (`reproduced_locally: true`,
see [`eval/metaworld_mt50.json`](eval/metaworld_mt50.json)).

| Suite | Tasks | Protocol | Result |
| --- | --- | --- | --- |
| MT50 | 50 | 1 episode / seed 0 / `max_steps=200` | **16/50 solved · avg 0.30** |

The MT50 run also covers all 10 MT10 tasks; a dedicated
`openral benchmark run --suite metaworld_mt10` reproduction can be written to
`eval/metaworld_mt10.json` by re-running the command above. Raise `n_episodes`
for a paper-equivalent (50-goals/task) number.

Solved at seed 0 (success_rate 1.0): `assembly-v3`, `button-press-topdown-v3`,
`button-press-v3`, `coffee-button-v3`, `door-close-v3`, `door-lock-v3`,
`drawer-close-v3`, `faucet-close-v3`, `handle-press-side-v3`, `handle-press-v3`,
`pick-place-v3`, `pick-place-wall-v3`, `plate-slide-back-side-v3`,
`plate-slide-side-v3`, `push-v3`.

> Single-episode/seed-0 numbers are a cheap smoke of the headline set, not a
> paper claim — per-task success on harder tasks is seed-sensitive.

## Demo scenes

Five single-task `BenchmarkScene` entries (website-demo tier, 500-step horizon,
50 episodes) live under `scenes/benchmark/`:
`metaworld_push.yaml`, `metaworld_pick_place.yaml`, `metaworld_button_press.yaml`,
`metaworld_door_open.yaml`, `metaworld_drawer_open.yaml`. The full sweeps live in
`benchmarks/metaworld_mt10.yaml` (10 tasks) and `benchmarks/metaworld_mt50.yaml`
(50 tasks).

## License

This rSkill package (`rskill.yaml`, `README.md`,
`eval/metaworld_mt50.json`) is **Apache-2.0**. The wrapped weights are also
Apache-2.0. Commercial use is allowed.

## See also

- [`rskills/smolvla-libero/README.md`](../smolvla-libero/README.md) — gold-standard LIBERO finetune (locally verified).
- [`robots/sawyer/README.md`](../../robots/sawyer/README.md).
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.2.
