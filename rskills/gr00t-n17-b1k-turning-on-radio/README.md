---
language:
- en
license: other
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- gr00t
- vision-language-action
- r1pro
- behavior-1k
datasets:
- behavior-1k/2026-challenge-demos
inference: false
base_model:
- nvidia/GR00T-N1.7-3B
base_model_relation: finetune
---

# rskill-gr00t-n17-b1k-turning-on-radio

> **OpenRAL rSkill** — the official 2026 BEHAVIOR-1K GR00T N1.7
> `turning_on_radio` checkpoint for the simulated Galaxea R1 Pro.

This package contains the OpenRAL manifest and adapter configuration only. The
organizer checkpoint is downloaded separately from the
[BEHAVIOR baseline page](https://behavior.stanford.edu/challenge/baselines.html);
weights are not copied into this repository.

## Preview

The official task demonstration is available in the
[BEHAVIOR challenge gallery](https://behavior.stanford.edu/challenge/tasks/).
No local rollout image is shipped until the checkpoint has been reproduced on
this host.

## What this skill does

The policy navigates the R1 Pro to a household radio and manipulates its controls
to turn it on. It is task-specific to `turning_on_radio`.

| Field | Value |
| --- | --- |
| Actions | rotate, push |
| Objects | radio, dial, button |
| Scenes | household living room |
| Embodiment | BEHAVIOR R1 Pro (`r1pro`) |

## How it works

The adapter runs the pinned `wensi-ai/Isaac-GR00T` `behavior` branch in an
isolated Python 3.10 sidecar. The sidecar uses the upstream `Gr00tPolicy` and
`B1KPolicyWrapper` unchanged, including temporal ensembling and the official
R1Pro modality split.

### Observation -> action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | head RGB | HWC uint8 | Official ZED head camera |
| in | left wrist RGB | HWC uint8 | Left RealSense |
| in | right wrist RGB | HWC uint8 | Right RealSense |
| in | proprioception | `(61,)` float32 | Official `PROPRIOCEPTION_INDICES["R1Pro"]` order |
| out | action | `(23,)` float32 | Base velocity 3 + torso 4 + arms 7+7 + grippers 1+1 |

The organizer wrapper converts the 61-D vector into base, torso, arm, and
gripper state groups. It emits a 16-step GR00T action horizon and returns one
temporally-ensembled 23-D action per evaluator step.

## Upstream model / training

| Field | Value |
| --- | --- |
| Source code | [`wensi-ai/Isaac-GR00T@ace36d9`](https://github.com/wensi-ai/Isaac-GR00T/tree/ace36d935b376fbf25cd56371e23877b95407c40) |
| Checkpoint | Organizer-provided `turning_on_radio` Google Drive checkpoint |
| Base model | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Dataset | [`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) |
| Paper | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| Parameters | approximately 3.1 B |
| Weights license | Unknown: the Drive checkpoint ships without a separately published license file |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Simulated Galaxea R1 Pro | `r1pro` | evaluator + deploy sim | Official BEHAVIOR observation/action contract |

`openral deploy sim` uses `robots/r1pro/robot.yaml`, publishes the simulator's
native 61-D policy state through WorldState, validates every typed action slot
through the safety kernel, then atomically commits all six slots as one 23-D
OmniGibson step.

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.head` | RGB | 224 x 224 | HWC uint8 |
| `observation.images.left_wrist` | RGB | 224 x 224 | HWC uint8 |
| `observation.images.right_wrist` | RGB | 224 x 224 | HWC uint8 |
| `observation.state` | proprioception | `(61,)` | float32 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-gr00t-n17-b1k-turning-on-radio` |
| `version` | `0.1.0` |
| `license` | `unknown` |
| `role` | `s1` |
| `model_family` | `gr00t` |
| `runtime` | external Python 3.10 Isaac-GR00T sidecar |
| `weights_uri` | `local://checkpoints/behavior-groot-turning-on-radio` |
| `chunk_size` | 16 |
| `state_contract.dim` / `action_contract.dim` | 61 / 23 |
| `latency_budget.per_chunk_ms` | 1500 |

## Quick start

```bash
git clone https://github.com/wensi-ai/Isaac-GR00T \
  ~/.cache/openral/behavior-groot/source
git -C ~/.cache/openral/behavior-groot/source checkout \
  ace36d935b376fbf25cd56371e23877b95407c40
cd ~/.cache/openral/behavior-groot/source
uv sync --frozen --python 3.10

export OPENRAL_BEHAVIOR_GROOT_SIDECAR_PYTHON="$PWD/.venv/bin/python"
export OPENRAL_BEHAVIOR_GROOT_CHECKPOINT=/absolute/path/to/checkpoint

cd /path/to/openral
just sync --group behavior-groot
openral behavior serve \
  --rskill rskills/gr00t-n17-b1k-turning-on-radio \
  --task turning_on_radio
```

Full deploy graph:

```bash
openral deploy sim \
  --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"
```

In the BEHAVIOR environment:

```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 \
  --output-dir outputs/openral --write-video
```

## Reproduction

The command above is the canonical one-rollout reproduction. Use public
instances `0-9` for challenge reporting and keep the evaluator-generated JSON
and videos unmodified.

## Evaluation

No OpenRAL-generated score is shipped. The organizer checkpoint and evaluator
have not yet been run together on this host.

## License

The OpenRAL adapter, manifest, and documentation are Apache-2.0. The
organizer-provided fine-tuned checkpoint is marked `license: unknown` because
the Google Drive artifact has no separately published license file. The base
GR00T N1.7 model uses the NVIDIA Open Model License, but this package does not
silently assume that the fine-tune inherits identical terms.

## See also

- [BEHAVIOR evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html)
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
- [`python/cli/src/openral_cli/behavior.py`](../../python/cli/src/openral_cli/behavior.py)
