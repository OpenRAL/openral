"""Export PekingU/rtdetr_r18vd_coco_o365 to ONNX matching ObjectsDetector's contract.

Single image input, /255 preprocessing, two 3-D outputs: pre-sigmoid logits
(1,N,80) + cxcywh-normalised boxes (1,N,4). HF RTDetrForObjectDetection.forward
returns logits + pred_boxes (cxcywh); RTDetrImageProcessor rescales /255 with
do_normalize=False — matching the repo decode.

Usage (isolated ephemeral env — does NOT mutate the project venv; do not
`uv sync` the onnx-export group, which prunes pydantic/structlog from the dev
venv). `--isolated --no-project` is required so the project venv's torchvision
(built against a different torch) does not shadow the overlay and break the
`RTDetrForObjectDetection` import; `onnx`+`onnxscript` are needed by the
torch >=2.7 exporter. Under transformers>=5.22 the export recomputes the position
embedding in float32 (`_sincos_in_output_dtype`) so ONNX Runtime can load it:
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


def _sincos_in_output_dtype(
    height: int,
    width: int,
    embed_dim: int = 256,
    temperature: float = 10000.0,
    cls_token: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """transformers' ``build_2d_sinusoidal_position_embedding``, computed in ``dtype``.

    transformers 5.22 computes RT-DETR's 2-D sin-cos position embedding in float64
    (5.5 used the model dtype). The exported graph then holds float64 ``Cos`` nodes,
    which ONNX Runtime's CPU provider has no kernel for ("Could not find an
    implementation for Cos(7)"), so the model would not load. Same formula and layout
    (``[sin_h | cos_h | sin_w | cos_w]``, row-major), float32 arithmetic.
    """
    pos_dim = embed_dim // 4
    omega = 1.0 / temperature ** (torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim)
    grid_h, grid_w = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    emb_h = grid_h.flatten().outer(omega)
    emb_w = grid_w.flatten().outer(omega)
    pos = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
    if cls_token:
        pos = torch.cat([torch.zeros(1, embed_dim, dtype=dtype, device=device), pos], dim=0)
    return pos


def export(out_path: Path, model_id: str = "PekingU/rtdetr_r18vd_coco_o365") -> None:
    from transformers.models.rt_detr import modeling_rt_detr

    if hasattr(modeling_rt_detr, "build_2d_sinusoidal_position_embedding"):  # transformers>=5.22
        modeling_rt_detr.build_2d_sinusoidal_position_embedding = _sincos_in_output_dtype
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
