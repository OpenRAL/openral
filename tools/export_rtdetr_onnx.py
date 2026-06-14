"""Export PekingU/rtdetr_r18vd_coco_o365 to ONNX matching ObjectsDetector's contract.

Single image input, /255 preprocessing, two 3-D outputs: pre-sigmoid logits
(1,N,80) + cxcywh-normalised boxes (1,N,4). HF RTDetrForObjectDetection.forward
returns logits + pred_boxes (cxcywh); RTDetrImageProcessor rescales /255 with
do_normalize=False — matching the repo decode (ADR-0037).

Usage (isolated ephemeral env — does NOT mutate the project venv; do not
`uv sync` the onnx-export group, which prunes pydantic/structlog from the dev
venv). `--isolated --no-project` is required so the project venv's torchvision
(built against a different torch) does not shadow the overlay and break the
`RTDetrForObjectDetection` import; `onnx`+`onnxscript` are needed by the
torch >=2.7 exporter; `transformers<5` pins the stable RTDetr forward signature:
    uv run --isolated --no-project \
        --with "transformers>=4.45,<5" --with "torch>=2.2" --with torchvision \
        --with onnx --with onnxscript \
        python tools/export_rtdetr_onnx.py --out rskills/rtdetr-coco-r18/model.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import RTDetrForObjectDetection


def export(out_path: Path, model_id: str = "PekingU/rtdetr_r18vd_coco_o365") -> None:
    # transformers ships no type stubs, so from_pretrained() returns Any;
    # bind it to a typed handle so .eval() is a typed nn.Module call.
    model: torch.nn.Module = RTDetrForObjectDetection.from_pretrained(model_id)
    model.eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.m(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    wrapped = _Wrap(model).eval()
    dummy = torch.randn(1, 3, 640, 640)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_path),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={"pixel_values": {0: "batch"}},
        opset_version=17,
    )
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"exported {out_path} sha256={sha}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="PekingU/rtdetr_r18vd_coco_o365")
    ns = ap.parse_args()
    export(ns.out, ns.model_id)
