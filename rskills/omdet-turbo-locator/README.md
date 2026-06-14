---
tags:
  - OpenRAL
  - rskill
  - detector
  - object-detection
  - open-vocabulary
  - zero-shot
  - omdet-turbo
  - on-demand
  - locate-in-view
license: apache-2.0
language:
  - en
base_model:
  - omlab/omdet-turbo-swin-tiny-hf
---

# rskill-omdet-turbo-locator

> **OpenRAL rSkill** — OmDet-Turbo (Swin-tiny) packaged as an Apache-2.0,
> **on-demand** open-vocabulary locator (`mode: on_demand`, ADR-0051). The
> reasoner invokes it via the read-only `locate_in_view` tool — "is object X in
> view right now?" — when it needs a specific object the continuous detector
> bank does not cover. A lightweight, real-time, in-process alternative to the
> 3B NVIDIA LocateAnything VLM for simple "find X" queries. **No actuators.**

This package wraps `hf://omlab/omdet-turbo-swin-tiny-hf` with a `rskill.yaml`
manifest. It does **not** copy model weights — they are the same Apache-2.0
checkpoint as its continuous sibling [`omdet-turbo-indoor`](../omdet-turbo-indoor/).

## What this skill does

Answers on-demand open-vocabulary localization queries from the reasoner: given
a free-text object (e.g. `"the red stapler"`), it runs one detection pass on the
current frame and reports whether that object is visible and where. It is **not**
a continuous background producer — it does not stream into world state every
frame; it responds when prompted (the `locate_in_view` service / the
`detector_query` topic). It emits no action chunks and drives no actuators.

| Field | Value |
| --- | --- |
| Actions | `detect` |
| Objects | open-vocabulary queried object (any free-text class the reasoner asks for) |
| Scenes  | tabletop, kitchen, indoor, household, office |
| Embodiment | embodiment-agnostic (any RGB camera ≥ 640×480) |

## How it works

OmDet-Turbo is a real-time `transformers` open-vocabulary detector
(`AutoModelForZeroShotObjectDetection`), run **in-process** by the
[`OmDetTurboDetector`](../../python/runner/src/openral_runner/backends/gstreamer/omdet_turbo_detector.py)
backend (`DetectorTier.ZEROSHOT_HF`). The same backend serves both detector
modes; this rSkill selects `mode: on_demand`, so the detector node exposes the
`locate_in_view` service and the `detector_query` retarget topic:

- `detect_with_query(frame, …, query)` — one-shot detection for a reasoner query
  without disturbing any persistent vocabulary (the `locate_in_view` path).
- `set_query(text)` — persistently retarget the query (the `detector_query` topic).

The free-text query is parsed into OmDet's multi-label class list by
`query_to_classes` (comma / `</c>` separated; a single phrase is one class).
`labels` in the manifest is only the static default used when no query is
supplied.

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | any RGB camera | `(H, W, 3)` BGR `uint8` | latest frame cached per camera for the service; min 640×480 |
| in | query | text | object/description from the reasoner's `locate_in_view` call |
| out | `ObjectsMetadata` | list of `ObjectDetection2D` | `(label, confidence, bbox_xyxy)`; no action chunk |

## Upstream model and training

A thin wrapper around the upstream Apache-2.0 OmDet-Turbo checkpoint; weights
live upstream and are not copied here.

| Field | Value |
| --- | --- |
| Source repo | [`omlab/omdet-turbo-swin-tiny-hf`](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) |
| Base model  | OmDet-Turbo, Swin-tiny backbone |
| Paper       | [arxiv:2403.06892](https://arxiv.org/abs/2403.06892) — *Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head* |
| License     | apache-2.0 (commercial use permitted) |
| Parameters  | ~115 M |
| Training data | upstream: Objects365 / GoldG and grounding data per the OmDet-Turbo release |

## Supported robots

Embodiment-agnostic — the only requirement is an RGB camera stream. All in-tree
embodiment tags are declared in `rskill.yaml`.

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| any with an RGB camera | `franka_panda`, `so100_follower`, `aloha`, … | ⚡ experimental | camera-only |

## Sensors required

Mirrors `rskill.yaml::sensors_required`.

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| any RGB camera | RGB | 640 × 480 | `uint8` BGR frame |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-omdet-turbo-locator` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s1` / `detector` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp16` |
| `detector.engine` / `detector.mode` | `zeroshot_hf` / `on_demand` |
| `weights_uri` | `hf://omlab/omdet-turbo-swin-tiny-hf` |
| `latency_budget.per_chunk_ms` | 200 ms |
| `commercial_use_allowed` | yes (Apache-2.0 weights) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```bash
uv sync --group omdet   # torch + transformers for the in-process backend
```

```python
from openral_core.schemas import RSkillManifest, DetectorMode

manifest = RSkillManifest.from_yaml("rskills/omdet-turbo-locator/rskill.yaml")
assert manifest.detector.mode is DetectorMode.ON_DEMAND
```

## Reproduction

Packaging-only wrapper — no trained numbers to reproduce. Validate the wiring
(manifest + backend query path) without a GPU:

```bash
just bootstrap && uv sync --all-packages
uv run pytest tests/unit/test_omdet_turbo_detector.py
```

## Evaluation

No benchmarks shipped — packaging-only wrapper; see CLAUDE.md §6.4.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **apache-2.0**. The wrapped
weights at `hf://omlab/omdet-turbo-swin-tiny-hf` are also **apache-2.0**, so the
locator is fully commercial-safe (CLAUDE.md §1.9).

## See also

- [`rskills/omdet-turbo-indoor/`](../omdet-turbo-indoor/) — the continuous
  background sibling (same weights, `mode: continuous`, fixed 266-class vocab).
- [`rskills/locateanything-3b-nf4/`](../locateanything-3b-nf4/) — higher-quality
  3B open-vocab locator (NVIDIA non-commercial; `VLM_SIDECAR` tier).
- [`docs/adr/0051-detector-invocation-mode.md`](../../docs/adr/0051-detector-invocation-mode.md) — continuous vs on-demand detector mode.
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
