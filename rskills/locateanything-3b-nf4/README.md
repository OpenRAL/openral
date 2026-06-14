---
tags:
  - OpenRAL
  - rskill
  - detector
  - object-detection
  - visual-grounding
  - locateanything
  - nvidia
  - nf4
  - bitsandbytes
license: other
license_name: nvidia-license
license_link: https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE
language:
  - en
base_model:
  - nvidia/LocateAnything-3B
---

# rskill-locateanything-3b-nf4

> **OpenRAL rSkill** - NVIDIA LocateAnything-3B packaged as an NF4
> bitsandbytes PyTorch detector rSkill. It localizes queried objects, text,
> GUI elements, and points from RGB images and emits perception results only.
> **No actuators.** The wrapped upstream weights are NVIDIA non-commercial
> research/evaluation weights.

## Quick Start

```bash
OPENRAL_ALLOW_NONCOMMERCIAL=1 ral skill install hf://OpenRAL/rskill-locateanything-3b-nf4
```

```python
import os

os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"
os.environ["OPENRAL_ALLOW_REMOTE_CODE"] = "1"

from openral_core.schemas import RSkillManifest

manifest = RSkillManifest.from_yaml("rskills/locateanything-3b-nf4/rskill.yaml")
assert manifest.kind == "detector"
assert manifest.quantization.extra["scheme"] == "nf4"
```

The upstream model uses Transformers custom code (`trust_remote_code=True`).
OpenRAL should execute it only after the operator has accepted the remote-code
risk for this specific package.

## What It Does

LocateAnything is an open-vocabulary visual-grounding model. Given an RGB image
and a natural-language query, it can return structured coordinate tokens for
object detection, phrase grounding, dense detection, scene text localization,
GUI element grounding, and point localization. The model card describes a hybrid
mode that combines parallel box decoding with autoregressive fallback for format
irregularity or spatial ambiguity.

This rSkill declares `kind: detector` because it is a pure perception producer:
it consumes camera frames and text/object queries, emits localization metadata,
and never drives `ros2_control` joints.

## Runtime Status

The package is marked `runtime: pytorch` with NF4 metadata: OpenRAL's original
ADR-0037 camera-tee detector runner is ONNX/TensorRT-oriented, while upstream
LocateAnything is a Transformers custom-code model needing `transformers==4.57.1`.
The HF rSkill repo therefore contains the quantized PyTorch weights and upstream
custom-code sidecars needed by `AutoModel.from_pretrained(..., trust_remote_code=True)`.

The OpenRAL adapter is **implemented and validated** (ADR-0037 2026-06-09
amendment): the `LocateAnythingDetector` backend
(`openral_runner.backends.gstreamer`) runs the model out-of-process in an isolated
`transformers==4.57.1` venv (`tools/locateanything_sidecar.py`) over a ZMQ +
msgpack link, parses its `<ref>`/`<box>` text into `ObjectsMetadata`, and is
selected as `DetectorTier.VLM_SIDECAR` for `runtime: pytorch` manifests. It is a
drop-in for the RT-DETR ONNX detector rskills in the
`openral deploy sim --object-detector-manifest …` graph, with a static default
query (manifest `labels`), a dynamic `/openral/perception/detector_query` override
for the continuous leg, and the read-only `locate_in_view` reasoner tool + service
for one-shot on-demand checks (ADR-0043).

Two venvs are involved: the **sidecar** venv (`transformers==4.57.1`, holds the
model — provision separately, point at it with `OPENRAL_LOCATEANYTHING_SIDECAR_VENV`)
and the **detector-node** venv (the deploy-sim runtime), which needs the `pyzmq`
+ `msgpack` ZMQ client from the `locateanything` dependency group
(`uv sync --group locateanything`).

## Upstream Model And Training

| Field | Value |
| --- | --- |
| Source repo | [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) |
| Source revision | `7a81d810571dc5f244b2f0b6868128f24b1cbd85` |
| Paper | [arxiv:2605.27365](https://arxiv.org/abs/2605.27365) |
| Architecture | LocateAnythingForConditionalGeneration; MoonViT vision tower plus Qwen2.5-3B text backbone |
| Runtime | Transformers custom code, BF16 upstream; this rSkill ships NF4 packed weights |
| Training scale | 12M unique images, about 140M natural-language queries, and 785M bounding boxes per upstream card |
| License | NVIDIA License, non-commercial research/evaluation use |

The upstream card lists supported tasks including general object detection,
dense object detection, referring-expression grounding, scene text detection,
layout/OCR grounding, GUI grounding, and point localization.

## Supported Robots And Embodiments

This detector is embodiment-agnostic. The only declared hardware requirement is
an RGB camera stream of at least 640 x 480. All in-tree OpenRAL embodiment tags
are listed in `rskill.yaml` so robots with a compatible RGB sensor can install
the package once a PyTorch detector adapter is available.

## Sensors And Observation Contract

| Direction | Key | Modality | Shape / format | Notes |
| --- | --- | --- | --- | --- |
| in | any RGB camera | RGB image | min 640 x 480 | `vla_feature_key` is intentionally omitted |
| in | detector query | text | natural language | object names, referring phrases, OCR/layout queries, GUI targets, point targets |
| preprocessing | LocateAnythingProcessor | image + text | dynamic visual tokens | upstream processor uses 14 px patches, 2 x 2 merge kernel, and 25,600 token input limit |
| out | localization tokens | text | boxes or points | adapter must parse coordinate tokens into OpenRAL `ObjectsMetadata` |

The model emits no action chunks and has no proprioception contract.

## Manifest Summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-locateanything-3b-nf4` |
| `version` | `0.1.0` |
| `license` | `nvidia_non_commercial` |
| `role` / `kind` | `s1` / `detector` |
| `runtime` | `pytorch` |
| `quantization.dtype` | `int4` |
| `quantization.extra.scheme` | `nf4` |
| `weights_uri` | `hf://OpenRAL/rskill-locateanything-3b-nf4` |
| `source_repo` | `hf://nvidia/LocateAnything-3B@7a81d810571dc5f244b2f0b6868128f24b1cbd85` |
| `latency_budget.per_chunk_ms` | 1000 ms |
| `actions` | `detect` |

The published HF repo includes `model.safetensors`, `quantization_metadata.json`,
upstream tokenizer/processor/config sidecars, and the upstream custom-code files
required by Transformers.

## Quantization

NF4 packing follows the OpenRAL bitsandbytes rule used by
`tools/quantize_rskill.py`: large `Linear` modules with at least 4,000,000
weight elements are rewritten to `bnb.nn.Linear4bit` with BF16 compute, while
smaller heads stay in BF16. The manifest records this as `dtype: int4` plus
`extra.scheme: nf4` because the schema enum represents storage dtype rather than
bitsandbytes' named quantization scheme.

Reproduce from this worktree:

```bash
cd <path/to/openral-checkout>
OPENRAL_TRUSTED_REMOTE_CODE_ORGS=nvidia \
  uv run python tools/quantize_rskill.py \
  --source nvidia/LocateAnything-3B \
  --target OpenRAL/rskill-locateanything-3b-nf4 \
  --loader transformers \
  --transformers-auto-class AutoModel \
  --trust-remote-code \
  --scheme nf4 \
  --device cuda \
  --skip-upload \
  --keep-temp
```

Use the private-only `tools/rskill_publisher.py` path, or an equivalent
`HfApi.create_repo(..., private=True)` plus `upload_folder`, for publication.
The quantizer's generic upload helper is not used for this package because this
rSkill must remain private in the OpenRAL organization.

## License

The rSkill package metadata and README are OpenRAL project files. The wrapped
LocateAnything weights and upstream custom-code sidecars are governed by the
NVIDIA License from `nvidia/LocateAnything-3B`; Section 3.3 limits use to
non-commercial research or evaluation except for NVIDIA and its affiliates.
OpenRAL should require explicit non-commercial acceptance before install or
activation.
