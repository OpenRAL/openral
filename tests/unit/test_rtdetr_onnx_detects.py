"""Gate: the exported RT-DETR ONNX produces real COCO detections via ObjectsDetector.

Skips when the ONNX or onnxruntime/PIL is absent (CI without the exported weight);
runs locally where tools/export_rtdetr_onnx.py has produced model.onnx.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("PIL")

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "rskills" / "rtdetr-coco-r18" / "model.onnx"
IMG = Path(__file__).parent / "fixtures" / "coco_sample.jpg"
COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@pytest.mark.skipif(
    not ONNX.exists(), reason="model.onnx not exported (run tools/export_rtdetr_onnx.py)"
)
def test_exported_rtdetr_detects_a_real_coco_object():
    import numpy as np
    from openral_runner.backends.gstreamer.objects_detector import ObjectsDetector
    from PIL import Image as PILImage

    det = ObjectsDetector(
        str(ONNX),
        labels=COCO80,
        model_id="rtdetr-coco-r18",
        input_size=(640, 640),
        score_threshold=0.3,
    )
    rgb = np.asarray(PILImage.open(IMG).convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[..., ::-1]).tobytes()
    md = det.detect(bgr, w, h, "front_depth")
    assert md is not None and len(md.detections) >= 1
    assert md.frame_width == w and md.frame_height == h
    for d in md.detections:
        assert d.label in COCO80
        x0, y0, x1, y1 = d.bbox_xyxy
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    # The canonical image contains cats — sanity that detection is real, not noise:
    labels = {d.label for d in md.detections}
    assert labels & {"cat", "remote", "couch", "tv", "bed", "chair"}, f"unexpected labels {labels}"
