---
tags:
  - OpenRAL
  - rskill
license: other
language:
  - en
---

# rskill-TEMPLATE_ID

> **OpenRAL rSkill** — one-sentence summary of what this skill does, on
> which embodiment, trained on which dataset. Replace this blockquote
> before publishing; `tools/rskill_publisher.py` refuses to upload a
> README that still contains `TEMPLATE_ID` or `TODO:` placeholders.

This package wraps `hf://<upstream-owner>/<upstream-repo>` with a
`rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does
**not** copy model weights.

> **Scaffold note.** This file was produced by `openral rskill new
> TEMPLATE_ID` from `rskills/template/README.md`. Every `<TODO>` /
> `TEMPLATE_ID` / `TEMPLATE_ORG` marker below is a placeholder the
> documentation validator
> (`openral_cli._rskill_doc_validator.validate_rskill_docs`) treats as
> a publish-blocking error — read each section, fill it in, and delete
> this note when you are done.

## What this skill does

<!-- TODO: 1–3 sentences naming the action verbs, objects, and scenes.
     Mirror the `actions:` / `objects:` / `scenes:` keys you set in
     `rskill.yaml` so the reasoner's LLM tool palette and this README
     agree on what the skill is for. Example:

         Picks a single cube from the tabletop and places it in a bin
         on the same surface. Trained on the LIBERO-Object suite (10
         tasks, 50 demos each). Robot: Franka Panda in MuJoCo. -->

| Field | Value |
| --- | --- |
| Actions | <!-- TODO: pick, place, pick_and_place, … --> |
| Objects | <!-- TODO: cube, mug, drawer, … --> |
| Scenes  | <!-- TODO: tabletop, kitchen, … --> |
| Embodiment | <!-- TODO: franka_panda / so100_follower / aloha / … --> |

## How it works

<!-- TODO: One paragraph on the model architecture and how observations
     turn into actions. Cover: policy family (smolvla / pi05 / xvla /
     act / diffusion / rldx / custom), backbone, action head, chunk
     size, control mode. Don't paraphrase the paper — say what THIS
     packaging does (input shape, output shape, replay cadence). -->

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` | `(1, 3, H, W) float32 [0,1]` | <!-- TODO: top / wrist / overhead --> |
| in | `observation.images.camera2` | `(1, 3, H, W) float32 [0,1]` | <!-- TODO --> |
| in | `observation.state`           | `(1, D)` float32                | <!-- TODO: joint positions in rad? deg? --> |
| out | action chunk                  | `(chunk_size, A)` float32       | <!-- TODO: joint pos? delta EE? --> |

## How it was trained

<!-- TODO: Where do the weights come from? Cover all of:

       * upstream HF Hub repo (link)
       * base / pretrain checkpoint (link)
       * training dataset (link, number of demos / hours / steps)
       * hardware used for training (if known)
       * paper / blog post (link)

     If this rSkill is a thin wrapper around an upstream checkpoint
     ("does not copy weights"), state that explicitly so consumers know
     where the weights live. -->

| Field | Value |
| --- | --- |
| Source repo | <!-- TODO: [`<owner>/<repo>`](https://huggingface.co/<owner>/<repo>) --> |
| Base model  | <!-- TODO: [`<owner>/<base>`](https://huggingface.co/<owner>/<base>) --> |
| Paper       | <!-- TODO: [arxiv:NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) — *Title* --> |
| License     | <!-- TODO: apache-2.0 / mit / permissive_research / … --> |
| Parameters  | <!-- TODO: e.g. ~450 M --> |
| Training data | <!-- TODO: dataset link + size, e.g. 1 693 LIBERO demos --> |

## Supported robots

<!-- TODO: One row per embodiment you have validated. The first column
     must match an `EmbodimentTag` literal so `openral rskill check`
     accepts it. -->

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| <!-- TODO --> | <!-- TODO --> | <!-- TODO: ✓ validated / ⚡ experimental --> | <!-- TODO --> |

## Sensors required

<!-- TODO: Mirror `rskill.yaml::sensors_required`. The validator only
     checks that this heading is present; the content matters for
     consumers running `openral rskill check` against their robot. -->

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | <!-- TODO: 224 × 224 --> | `float32` |
| `observation.images.camera2` | RGB | <!-- TODO --> | `float32` |
| `observation.state`          | proprioception | <!-- TODO: (D,) --> | `float32` |

## Manifest summary

<!-- TODO: Mirror the headline fields from `rskill.yaml`. Keep in sync
     with the YAML; consumers grep this table to decide whether the
     skill fits their hardware. -->

| Field | Value |
| --- | --- |
| `name` | `TEMPLATE_ORG/rskill-TEMPLATE_ID` |
| `version` | `0.1.0` |
| `license` | <!-- TODO --> |
| `role` | `s1` |
| `embodiment_tags` | <!-- TODO --> |
| `runtime` / `quantization.dtype` | <!-- TODO: pytorch / bf16 --> |
| `weights_uri` | <!-- TODO: hf://<owner>/<repo> --> |
| `latency_budget.per_chunk_ms` | <!-- TODO --> |
| `commercial_use_allowed` | <!-- TODO: derived from license --> |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
# TODO: replace with a runnable example that loads THIS skill and
# performs one inference. Keep it copy-pasteable.
from openral_rskill.loader import rSkill

pkg = rSkill.from_yaml("rskills/TEMPLATE_ID/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.version)
```

## Reproduction

```bash
# TODO: replace with the exact `openral sim run` / `openral benchmark run`
# command that reproduces the numbers in `eval/*.json` (or, for
# packaging-only skills, validates the wiring).
just bootstrap && uv sync --all-packages

openral sim run --config scenes/<your-config>.yaml \
            --rskill rskills/TEMPLATE_ID
```

## Evaluation

<!-- TODO: One subsection per `eval/<benchmark>.json` file you ship. If
     this is a packaging-only skill with no benchmarks, replace this
     section with a one-line "No benchmarks shipped — see
     CLAUDE.md §6.4." note and the validator will still accept it. -->

| Benchmark | Score | `reproduced_locally` | Config |
| --- | --- | --- | --- |
| <!-- TODO --> | <!-- TODO --> | <!-- TODO: true / false --> | <!-- TODO --> |

## License

<!-- TODO: State the license posture explicitly. The wrapping rSkill
     package is typically Apache-2.0; the wrapped weights may differ.
     Make the consumer-visible posture (`commercial_use_allowed`) match
     `rskill.yaml::license`. -->

This rSkill package (`rskill.yaml`, `README.md`, `eval/*.json`) is
**<TODO: apache-2.0 / mit / …>**. The wrapped weights at
`<TODO: weights_uri>` are released under
**<TODO: same as above / different — link to upstream LICENSE>**.

## See also

<!-- TODO: Cross-link to the paired robot manifest, sim config, and any
     sibling rSkills. Helps consumers navigate the in-tree catalog. -->

- `robots/<embodiment>/README.md` — RobotDescription manifest.
- `scenes/<config>.yaml` — paired SimEnvironment config.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
