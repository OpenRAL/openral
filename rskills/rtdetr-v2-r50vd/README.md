---
language:
- en
license: apache-2.0
pipeline_tag: object-detection
tags:
- OpenRAL
- rskill
- detector
- object-detection
- any
- rskill-kind:detector
- rskill-role:s1
- embodiment:any
- rskill-license:apache-2.0
- rt-detr
- rt-detr-v2
- onnx
- coco
base_model:
- PekingU/rtdetr_v2_r50vd
base_model_relation: finetune
inference: false
openral:
  description: RT-DETRv2 object detector, ResNet-50vd backbone, trained on COCO 2017
    (80 categories). ONNX runtime; perception producer that runs on the camera tee
    and publishes ObjectsMetadata to /openral/perception/objects. Improved baseline
    with bag-of-freebies and selective multi-scale features. Apache-2.0 weights mirrored
    from PekingU/rtdetr_v2_r50vd. See ADR-0037 for the detector kind contract.
  model_family: null
  min_vram_gb: 0.7
  latency_ms: 50.0
  runtime: onnx
  dtype: fp32
---

# rskill-rtdetr_v2_r50vd-any-coco-fp32

> **OpenRAL rSkill** — RT-DETRv2 (Real-Time DEtection TRansformer v2) with a
> ResNet-50vd backbone, trained on COCO 2017. Runs as a perception producer on
> the camera tee and publishes `ObjectsMetadata` to
> `/openral/perception/objects`. **No actuators.** This skill uses
> `kind: detector` (ADR-0037); it emits no `Action` chunks and drives no
> `ros2_control` joints.
>
> Weights are a direct mirror of
> [PekingU/rtdetr_v2_r50vd](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
> (Apache-2.0). This repo adds the OpenRAL `rskill.yaml` manifest.

## What it does

RT-DETRv2-R50 detects 80 COCO-category objects in each camera frame and
publishes per-frame `ObjectsMetadata` events containing bounding boxes, class
labels, and confidence scores. The runtime `ObjectsDetector` (in
`openral_runner`) reads the `detector` manifest block at configure time to
initialise the inference session and bind the class-id → label mapping.

The OpenRAL detector perception path (`ros_image_detector_node` →
`DetectorRunner` → `ObjectsDetector`) is **ONNX-based**, so this rSkill ships
an ONNX export (`model.onnx` + external-data `model.onnx.data`) produced by
`tools/export_rtdetr_onnx.py`. The manifest declares `runtime: onnx` with
fp32 weights, so `onnxruntime` runs it on any host (CPU or CUDA EP). The
detector tier is chosen by the host, not the manifest: with OpenRAL Pro
installed, the NVMM aggregator tier builds and caches a TensorRT engine from
the same `model.onnx` on first load. The `weights/` PyTorch checkpoint remains for standalone
`transformers` inference (see **Standalone inference** below).

RT-DETRv2 improves over RT-DETR v1 with selective multi-scale feature
extraction, a discrete sampling operator, and improved training strategies.

## Supported robots / embodiments

This detector is **embodiment-agnostic**: it requires only an RGB camera of at
least 640×480 and emits `ObjectsMetadata`. The manifest declares the
`embodiment_tags: ["any"]` wildcard; the `sensors_required` entry sets `modality: rgb`
with no `vla_feature_key`, so the loader accepts any RGB camera stream
regardless of its key name.

## Sensors / observation contract

| Direction | Key | Modality | Shape / format | Notes |
| --- | --- | --- | --- | --- |
| in | any RGB camera | RGB `sensor_msgs/Image` | min 640 × 480 | `vla_feature_key` unset — any camera name accepted |
| (preprocessing) | — | — | resized to 640 × 640, `/255` → float32 `[0,1]`, NCHW | `pixel_values` `(batch, 3, 640, 640)` |
| out | COCO-80 detections | `ObjectsMetadata` | per object: `label`, `confidence`, `bbox` | published to `/openral/perception/objects` |

The detector emits **no** `Action` chunks and has no proprioception
(`observation.state`) contract.

## Latency

| Host | Runtime | Median latency (ms) |
|------|---------|---------------------|
| Intel Core Ultra 7 255HX (20 threads, shared machine under load) | onnxruntime CPU, fp32 | 372 (R18 on the same run: 146) |

Measured 2026-11-19, 640×640 input, 30 runs after 5 warm-up. No GPU
measurement yet. The manifest's `per_chunk_ms: 50.0` is a GPU budget (CUDA
or the OpenRAL Pro TensorRT tier); the CPU tier does not meet it. The
detector is a background producer, so a slow tier lowers its publish rate
rather than blocking the action loop.

## VRAM

| dtype | VRAM   |
|-------|--------|
| fp32  | ~700 MB |

The shipped weights are fp32 (`dtype: fp32`), and the OpenRAL Pro TensorRT
tier builds an fp32 engine from them (it passes no reduced precision).

## Accuracy (COCO val2017)

| Metric      | Score |
|-------------|-------|
| AP@0.5:0.95 | 54.3% |
| AP@0.5      | 71.2% |
| AP@0.75     | 59.1% |

## Weights

Two artefacts ship in this rSkill:

1. **ONNX (used by the OpenRAL detector path)** — `model.onnx` +
   `model.onnx.data`. Not committed to git (binary artefact; see `.gitignore`).
   Reproduce with the same **ephemeral** overlay used for `rtdetr-coco-r18`:

   ```bash
   uv run --isolated --no-project \
       --with "transformers>=4.45,<5" --with "torch>=2.2" --with torchvision \
       --with onnx --with onnxscript \
       python tools/export_rtdetr_onnx.py \
       --out rskills/rtdetr-v2-r50vd/model.onnx \
       --model-id PekingU/rtdetr_v2_r50vd
   ```

   > Use `--isolated --no-project` — a plain `uv run --with` overlays the
   > project venv whose `torchvision` is built against a different `torch`,
   > breaking the `RTDetrForObjectDetection` import. Never `uv sync --group
   > onnx-export` (it prunes `pydantic`/`structlog` from the dev venv).

   | File              | Description                               | sha256 (first 16 hex) | Size   |
   |-------------------|-------------------------------------------|-----------------------|--------|
   | `model.onnx`      | ONNX graph (references `model.onnx.data`) | `e2c96541b7f9e110...`  | 3.9 MB |
   | `model.onnx.data` | External weight data (loaded by ORT/TRT)  | `eb70cc9cb101c445...`  | 165 MB |

   The new torch exporter is not bit-reproducible across toolchain versions;
   treat the digests as a same-host integrity check. The published copies on
   the HF Hub repo are canonical.

   | Field     | Value                                                              |
   |-----------|--------------------------------------------------------------------|
   | `model_id`| `PekingU/rtdetr_v2_r50vd`                                          |
   | `input`   | `pixel_values` — shape `(batch, 3, 640, 640)`, float32, range `[0,1]` |
   | `outputs` | `logits (1, 300, 80)` pre-sigmoid; `pred_boxes (1, 300, 4)` cxcywh |

2. **PyTorch (`weights/model.safetensors`)** — mirrored from
   [PekingU/rtdetr_v2_r50vd](https://huggingface.co/PekingU/rtdetr_v2_r50vd),
   same Apache-2.0 license, with the upstream `config.json` and
   `preprocessor_config.json`. Used only by the standalone `transformers`
   example below; the OpenRAL detector path does **not** load it.

## Upstream model / training

This rSkill packages RT-DETRv2 (Real-Time DEtection TRansformer v2) with a
**ResNet-50vd backbone (`r50vd`)**. It copies no new weights — both the ONNX
export and `weights/model.safetensors` derive from the upstream Transformers
checkpoint (see the **Weights** section above).

| Field | Value |
| --- | --- |
| Architecture | RT-DETRv2, `r50vd` backbone |
| Source repo | [`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) |
| Training data | COCO 2017 (80 categories) |
| Detector runtime | onnxruntime (fp32), or the OpenRAL Pro TensorRT tier, from `model.onnx` |
| Standalone runtime | PyTorch / `transformers` (`RTDetrV2ForObjectDetection`, `weights/`) |
| Paper | [arxiv:2407.17140](https://arxiv.org/abs/2407.17140) — *RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer* |
| License | apache-2.0 |

## Usage in OpenRAL

### Activate the skill

```bash
ral skill activate OpenRAL/rskill-rtdetr_v2_r50vd-any-coco-fp32
```

### Reference in robot manifest

```yaml
perception_producers:
  - skill_id: "hf://OpenRAL/rskill-rtdetr_v2_r50vd-any-coco-fp32"
    role: "s1"
```

### Standalone inference (Python)

```python
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

image_processor = RTDetrImageProcessor.from_pretrained(
    "OpenRAL/rskill-rtdetr_v2_r50vd-any-coco-fp32", subfolder="weights"
)
model = RTDetrV2ForObjectDetection.from_pretrained(
    "OpenRAL/rskill-rtdetr_v2_r50vd-any-coco-fp32", subfolder="weights"
).cuda()

image = Image.open("kitchen.jpg")
inputs = image_processor(images=image, return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([(image.height, image.width)], device="cuda"),
    threshold=0.5,
)
for result in results:
    for score, label_id, box in zip(
        result["scores"], result["labels"], result["boxes"]
    ):
        label = model.config.id2label[label_id.item()]
        print(f"{label}: {score:.2f} {[round(i, 2) for i in box.tolist()]}")
```

## Supported object classes (80 COCO)

**Household objects**: bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**People & animals**: person, cat, dog, bird, horse, sheep, cow, elephant,
bear, zebra, giraffe

**Outdoor / transport**: car, bicycle, motorcycle, bus, train, truck,
airplane, boat, traffic light, fire hydrant, stop sign, parking meter, bench,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports
ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-rtdetr_v2_r50vd-any-coco-fp32` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `kind` | `detector` (ADR-0037 perception producer) |
| `embodiment_tags` | `["any"]` (any robot with an RGB camera) |
| `runtime` / `quantization.dtype` | `onnx` / `fp32` |
| `weights_uri` | `local://rskills/rtdetr-v2-r50vd` |
| `latency_budget.per_chunk_ms` | `50.0` |
| `detector.labels` | 80 COCO categories |
| `detector.input_size` | `[640, 640]` |
| `detector.score_threshold` | `0.5` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Citation

```bibtex
@article{lv2024rtdetrv2,
  title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
  author={Lv, Wenyu and Zhao, Yian and Chang, Qinyao and Huang, Kui and Wang, Guanzhong and Liu, Yi},
  journal={arXiv preprint arXiv:2407.17140},
  year={2024}
}
```

## License

- **Weights** (`weights/`): Apache-2.0, mirrored from [PekingU/rtdetr_v2_r50vd](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- **rSkill manifest and packaging** (`rskill.yaml`, `README.md`): Apache-2.0
