---
tags:
  - OpenRAL
  - rskill
  - detector
  - object-detection
  - rt-detr
  - onnx
  - coco
license: apache-2.0
language:
  - en
base_model:
  - PekingU/rtdetr_r18vd_coco_o365
---

# rskill-rtdetr-coco-r18

> **OpenRAL rSkill** — RT-DETR (Real-Time DEtection TRansformer) with a
> ResNet-18vd backbone (r18vd), trained on COCO and exported to ONNX. Runs
> as a perception producer on the camera tee and publishes `ObjectsMetadata`
> to `/openral/perception/objects`. **No actuators.** This skill uses
> `kind: detector` (ADR-0037); it emits no `Action` chunks and drives no
> `ros2_control` joints.

## What it does

RT-DETR r18vd detects 80 COCO-category objects in each camera frame and
publishes per-frame `ObjectsMetadata` events containing bounding boxes,
class labels, and confidence scores. The runtime `ObjectsDetector`
(in `openral_perception`) reads the `detector` manifest block at configure
time to initialise the ONNX inference session and bind the class-id →
label mapping.

## Latency

| Host           | Latency (ms) |
|----------------|-------------|
| NVIDIA GPU     | ~15–30      |
| x86 CPU        | ~35–50      |

Budget declared in manifest: `per_chunk_ms: 50.0`.

## Weights

The `model.onnx` file is not committed to the repository (binary artefact; see
`.gitignore`). Reproduce it with an **ephemeral** overlay environment (does not
mutate the project venv):

```bash
uv run --isolated --no-project \
    --with "transformers>=4.45,<5" --with "torch>=2.2" --with torchvision \
    --with onnx --with onnxscript \
    python tools/export_rtdetr_onnx.py \
    --out rskills/rtdetr-coco-r18/model.onnx \
    --model-id PekingU/rtdetr_r18vd_coco_o365
```

> **Do NOT** run this via `uv sync --group onnx-export` — `uv sync` reconciles
> the project venv to the synced group set and prunes `pydantic`/`structlog`
> (and other deps) the source-on-PYTHONPATH dev/test setup relies on, breaking
> the unit tests.
>
> `--isolated --no-project` is required: a plain `uv run --with` overlays on the
> project venv, whose `torchvision` is built against a different `torch` than the
> overlay's — importing `RTDetrForObjectDetection` then dies with
> `operator torchvision::nms does not exist`. The isolated form builds a clean
> ephemeral env (project venv untouched). `transformers<5` keeps the stable
> RTDetr `forward` (logits + pred_boxes) signature; `onnx`+`onnxscript` are
> required by the torch ≥2.7 ONNX exporter. GPU footprint: ~0.2 GB at 640² fp32 —
> runs on an 8 GB card, no quantization needed.

The torch 2.9 new exporter splits the model into two files that must be kept
together in the same directory:

| File               | Description                                  | sha256 (first 16 hex)      | Size  |
|--------------------|----------------------------------------------|----------------------------|-------|
| `model.onnx`       | ONNX graph (references `model.onnx.data`)    | `bda4dbeceff130ce...`       | 2.3 MB |
| `model.onnx.data`  | External weight data (loaded by ORT)         | `8dff132e55df1bef...`        | 78 MB |

Full sha256 values (reproduced with `transformers 4.x` + `torch 2.9` + `onnxscript`;
the new torch exporter is not bit-reproducible across toolchain versions, so treat
these as a same-host integrity check, not a cross-version guarantee):
- `model.onnx`:      `bda4dbeceff130cec050e9757c9d95e217526a00730fb5f1558f960a6b316c63`
- `model.onnx.data`: `8dff132e55df1befdf394a672a29906e38df7653be66705c47bd2a41634567b2`

The published copies on the HF Hub repo are the canonical artefacts; the local
export above must match them on the same toolchain.

| Field        | Value                                                                |
|--------------|----------------------------------------------------------------------|
| `model_id`   | `PekingU/rtdetr_r18vd_coco_o365`                                     |
| `opset`      | 18 (torch 2.9 new exporter; opset 17 target auto-bumped by exporter) |
| `input`      | `pixel_values` — shape `(batch, 3, 640, 640)`, float32, range [0,1] |
| `outputs`    | `logits (1, 300, 80)` pre-sigmoid; `pred_boxes (1, 300, 4)` cxcywh  |

## Upstream model / training

This rSkill packages an RT-DETR (Real-Time DEtection TRansformer) object
detector with a **ResNet-18vd backbone (`r18vd`)**, exported to ONNX. It
copies no PyTorch policy weights — the ONNX graph is produced from the
upstream Transformers checkpoint by `tools/export_rtdetr_onnx.py` (see the
**Weights** section above for the exact command and sha256 digests).

| Field | Value |
| --- | --- |
| Architecture | RT-DETR, `r18vd` backbone |
| Source repo | [`PekingU/rtdetr_r18vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r18vd_coco_o365) |
| Training data | COCO (80 categories), pretrained on Objects365 (`o365`) |
| Export tool | `tools/export_rtdetr_onnx.py` → `model.onnx` + `model.onnx.data` |
| Paper | [arxiv:2304.08069](https://arxiv.org/abs/2304.08069) — *DETRs Beat YOLOs on Real-time Object Detection* |
| License | apache-2.0 |

## Supported robots / embodiments

This detector is **embodiment-agnostic**: it consumes any RGB camera stream
and emits `ObjectsMetadata`. All known embodiment tags are declared in the
manifest; the `sensors_required` entry has no `vla_feature_key`, so the
loader accepts any camera key — not just `camera1`.

## Sensors / observation contract

| Direction | Key | Modality | Shape / format | Notes |
| --- | --- | --- | --- | --- |
| in | any RGB camera | RGB `sensor_msgs/Image` | min 640 × 480 | `vla_feature_key` unset — any camera name accepted |
| (preprocessing) | — | — | resized to 640 × 640, `/255` → float32 `[0,1]`, NCHW | `pixel_values` `(batch, 3, 640, 640)` |
| out | COCO-80 detections | `ObjectsMetadata` | per object: `label`, `confidence`, `bbox` | published to `/openral/perception/objects` |

The detector emits **no** `Action` chunks and has no proprioception
(`observation.state`) contract.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-rtdetr-coco-r18` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `kind` | `detector` (ADR-0037 perception producer) |
| `embodiment_tags` | all 17 canonical embodiment tags (any robot with RGB camera) |
| `runtime` / `quantization.dtype` | `onnx` / `fp32` |
| `weights_uri` | `local://rskills/rtdetr-coco-r18` |
| `latency_budget.per_chunk_ms` | `50.0` |
| `detector.labels` | 80 COCO categories |
| `detector.input_size` | `[640, 640]` |
| `detector.score_threshold` | `0.5` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## License

Weights: Apache-2.0 (PaddlePaddle RT-DETR public release).
See [arxiv:2304.08069](https://arxiv.org/abs/2304.08069) for the paper.
