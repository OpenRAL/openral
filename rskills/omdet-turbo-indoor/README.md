---
tags:
  - OpenRAL
  - rskill
  - detector
  - object-detection
  - open-vocabulary
  - zero-shot
  - omdet-turbo
  - indoor
license: apache-2.0
language:
  - en
base_model:
  - omlab/omdet-turbo-swin-tiny-hf
---

# rskill-omdet-turbo-indoor

> **OpenRAL rSkill** — OmDet-Turbo (Swin-tiny) packaged as an in-process,
> Apache-2.0 open-vocabulary object detector run over a **fixed** curated indoor
> vocabulary (~266 household / kitchen / office / manipulation classes). It is an
> unprompted background perception producer: it streams `ObjectsMetadata` to
> `/openral/perception/objects` every frame, giving the world model far more
> object classes than the 80 COCO categories — without any reasoner prompting.
> **No actuators.**

This package wraps `hf://omlab/omdet-turbo-swin-tiny-hf` with a `rskill.yaml`
manifest that adds the fixed-vocabulary detector contract, capability checking,
license surfacing, and latency budgets. It does **not** copy model weights.

## What this skill does

Detects objects from a fixed curated indoor vocabulary in every RGB camera
frame and publishes 2D detections (`ObjectsMetadata`) on the perception bus. It
emits no action chunks, drives no actuators, and has no proprioception
contract — a pure ADR-0037 perception producer. Because the class list is fixed
(not query-driven), it behaves like a large closed-vocabulary detector: the
reasoner does not retarget it.

| Field | Value |
| --- | --- |
| Actions | `detect` |
| Objects | open-vocabulary indoor objects — kitchenware, tableware, appliances, furniture, tools, containers (~266 classes) |
| Scenes  | tabletop, kitchen, indoor, household, office, bathroom |
| Embodiment | embodiment-agnostic (any RGB camera ≥ 640×480) |

## How it works

OmDet-Turbo is a real-time, transformer-based open-vocabulary detector
(`AutoModelForZeroShotObjectDetection`). Unlike `locateanything-3b-nf4` — a heavy
VLM pinned to `transformers==4.57.1` that must run out-of-process in a sidecar —
OmDet-Turbo is a first-class `transformers` architecture that loads under the
OpenRAL runtime's own `transformers>=5`. It therefore runs **in process**: no
sidecar venv, no ZMQ.

The OpenRAL backend
([`OmDetTurboDetector`](../../python/runner/src/openral_runner/backends/gstreamer/omdet_turbo_detector.py))
loads the processor + model on first `detect()`, moves the model to CUDA when
available (CPU fallback otherwise), and runs the manifest's fixed `labels`
vocabulary against each frame via `processor.post_process_grounded_object_detection`.
It is selected as `DetectorTier.ZEROSHOT_HF` by `build_manifest_detector` for
manifests whose `detector.engine` is `zeroshot_hf`, and consumes the same
system-memory BGR camera-tee branch as the CPU ONNX and VLM-sidecar tiers
(ADR-0037 2026-06-12 amendment).

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | any RGB camera | `(H, W, 3)` BGR `uint8` | system-memory frame from the camera tee; min 640×480. `vla_feature_key` is intentionally omitted |
| out | `ObjectsMetadata` | list of `ObjectDetection2D` | `(label, confidence, bbox_xyxy)` per detection on `/openral/perception/objects`; no action chunk |

## Upstream model and training

This rSkill is a thin wrapper around the upstream Apache-2.0 OmDet-Turbo
checkpoint; the weights live upstream and are not copied here.

| Field | Value |
| --- | --- |
| Source repo | [`omlab/omdet-turbo-swin-tiny-hf`](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) |
| Base model  | OmDet-Turbo, Swin-tiny backbone |
| Paper       | [arxiv:2403.06892](https://arxiv.org/abs/2403.06892) — *Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head* |
| License     | apache-2.0 (commercial use permitted) |
| Parameters  | ~115 M |
| Training data | upstream: Objects365 / GoldG and grounding data per the OmDet-Turbo release |

## Supported robots

This detector is embodiment-agnostic — the only requirement is an RGB camera
stream. All in-tree embodiment tags are declared in `rskill.yaml`.

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| any with an RGB camera | `franka_panda`, `so100_follower`, `aloha`, … | ⚡ experimental | camera-only; see `rskill.yaml::embodiment_tags` for the full list |

## Sensors required

Mirrors `rskill.yaml::sensors_required`.

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| any RGB camera | RGB | 640 × 480 | `uint8` BGR frame |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-omdet-turbo-indoor` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` / `kind` | `s1` / `detector` |
| `embodiment_tags` | all in-tree embodiments (camera-only) |
| `runtime` / `quantization.dtype` | `pytorch` / `fp16` |
| `detector.engine` | `zeroshot_hf` (in-process Transformers zero-shot) |
| `weights_uri` | `hf://omlab/omdet-turbo-swin-tiny-hf` |
| `latency_budget.per_chunk_ms` | 200 ms |
| `commercial_use_allowed` | yes (Apache-2.0 weights) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```bash
uv sync --group omdet   # torch + transformers for the in-process backend
```

```python
from openral_core.schemas import RSkillManifest, DetectorEngine

manifest = RSkillManifest.from_yaml("rskills/omdet-turbo-indoor/rskill.yaml")
assert manifest.kind == "detector"
assert manifest.detector.engine is DetectorEngine.ZEROSHOT_HF
print(len(manifest.detector.labels), "fixed indoor classes")
```

Run it on the camera tee in sim (publishes `ObjectsMetadata` every frame, no
ONNX file and no prompting):

```bash
openral deploy sim \
  --object-detector-manifest rskills/omdet-turbo-indoor/rskill.yaml
```

## Reproduction

This is a packaging-only wrapper — there are no trained numbers to reproduce.
To validate the wiring (manifest + in-process dispatch) without a GPU:

```bash
just bootstrap && uv sync --all-packages
uv run pytest tests/unit/test_omdet_turbo_detector.py
```

The GPU-gated end-to-end test
(`test_e2e_detects_indoor_objects_on_coco_sample`) loads the real Apache-2.0
weights and grounds indoor classes on the `coco_sample.jpg` fixture; it skips on
GPU-less hosts (the legitimate CI skip path, CLAUDE.md §12).

## Evaluation

No benchmarks shipped — packaging-only wrapper; see CLAUDE.md §6.4.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **apache-2.0**. The wrapped
weights at `hf://omlab/omdet-turbo-swin-tiny-hf` are also released under
**apache-2.0**, so the detector is fully commercial-safe (CLAUDE.md §1.9) —
unlike the NVIDIA non-commercial `locateanything-3b-nf4` open-vocab detector.

## See also

- [`rskills/locateanything-3b-nf4/`](../locateanything-3b-nf4/) — query-driven
  open-vocab detector (NVIDIA non-commercial; `VLM_SIDECAR` tier).
- [`rskills/rtdetr-coco-r18/`](../rtdetr-coco-r18/) — fixed 80-class COCO RT-DETR
  detector (ONNX tier).
- [`docs/adr/0037-gstreamer-perception-bus-object-detection.md`](../../docs/adr/0037-gstreamer-perception-bus-object-detection.md) — detector kind + tier contract.
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
