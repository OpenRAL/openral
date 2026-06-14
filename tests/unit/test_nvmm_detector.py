"""GPU test for NvmmObjectsDetector (ADR-0037 PR5b). No DeepStream needed:
a deterministic RT-DETR-signature ONNX + a device RGBA buffer prove
device-pointer -> ObjectsMetadata. The source RGBA buffer is allocated with
cudart (cuda-python), matching the executor's nvrtc + cuda-python path (no
pycuda). Skipped without tensorrt/cuda-python/GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import ObjectsMetadata


def _export_const_rtdetr_onnx(path: Path, *, h: int, w: int, n: int, c: int) -> None:
    """Export a model with the RT-DETR I/O signature that emits one sure 'car'.

    Output 'logits' (1,n,c): query 0 strongly class 1, rest negative.
    Output 'boxes' (1,n,4): query 0 a centred box. Tied to the input mean * 0 so the
    graph keeps its image input (not constant-folded away) yet the detection is
    deterministic regardless of pixels — it validates the NVMM plumbing.
    """
    torch = pytest.importorskip("torch", reason="torch authors the ONNX fixture")

    class _ConstRTDETR(torch.nn.Module):
        def forward(self, images: torch.Tensor):
            bias = images.mean() * 0.0
            logit = torch.full((1, n, c), -9.0) + bias
            logit[0, 0, 1] = 9.0 + bias
            box = torch.zeros(1, n, 4) + bias
            box[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4]) + bias
            return logit, box

    torch.onnx.export(
        _ConstRTDETR().eval(),
        (torch.zeros(1, 3, h, w),),
        str(path),
        input_names=["images"],
        output_names=["logits", "boxes"],
        dynamo=False,
    )


def test_nvmm_detector_devptr_to_objectsmetadata(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from cuda.bindings import runtime as cudart
    from openral_runner.backends.gstreamer.nvbufsurface import NvBufSurfaceHandle
    from openral_runner.backends.gstreamer.nvmm_detector import NvmmObjectsDetector

    # cudaSetDevice initializes the primary context; skip on a GPU-less host.
    if int(cudart.cudaSetDevice(0)[0]) != 0:
        pytest.skip("no usable CUDA device")

    h, w = 64, 64
    onnx_path = tmp_path / "const_rtdetr.onnx"
    _export_const_rtdetr_onnx(onnx_path, h=h, w=w, n=10, c=3)

    det = NvmmObjectsDetector(
        onnx_path,
        labels=["bg", "car", "person"],
        model_id="rtdetr-const",
        input_size=(h, w),
        score_threshold=0.5,
    )
    src: Any = None  # bound before the try so the finally never raises NameError
    try:
        pitch = w * 4
        rgba = np.full((h, pitch), 128, dtype=np.uint8)
        # Allocate the source buffer on-device with cudart and upload the frame.
        (src,) = cudart.cudaMalloc(rgba.nbytes)[1:]
        cudart.cudaMemcpy(
            int(src),
            rgba.ctypes.data,
            rgba.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        handle = NvBufSurfaceHandle(
            gpu_ptr=int(src),
            width=w,
            height=h,
            pitch=pitch,
            color_format=19,
            size=h * pitch,
            batch_size=1,
        )
        md = det.detect_nvmm(handle, "head_rgb")
    finally:
        det.close()
        if src is not None:
            cudart.cudaFree(int(src))

    assert isinstance(md, ObjectsMetadata)
    assert md.detections[0].label == "car"
    assert md.sensor_id == "head_rgb"
