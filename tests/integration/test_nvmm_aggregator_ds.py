"""DeepStream-container integration test for the NVMM aggregator tier (ADR-0037 PR5b).

Runs ONLY inside the ds-on (DS9) container: it needs an NVMM colour-convert element
(nvvideoconvert) + libnvbufsurface. Skipped everywhere else. Builds
videotestsrc -> tee -> [nvvideoconvert -> NVMM RGBA -> appsink] via DetectorRunner
(tier=NVMM_AGGREGATOR) and asserts the aggregator emits ObjectsMetadata from the GPU
dataPtr with no host copy on the det leg. Real schemas / real pipeline / real engine
(CLAUDE.md §11)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

gi = pytest.importorskip("gi", reason="PyGObject not installed")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_nvmm_stack() -> None:
    """Skip unless an NVMM converter element + libnvbufsurface are present (ds-on/Tegra only)."""
    from openral_runner.backends.gstreamer import nvbufsurface
    from openral_runner.backends.gstreamer.pipeline import nvmm_convert_element

    for elem in ("videotestsrc", "tee", "queue", "appsink"):
        if Gst.Registry.get().find_feature(elem, Gst.ElementFactory.__gtype__) is None:
            pytest.skip(f"core GStreamer element {elem!r} missing")
    if nvmm_convert_element() is None:
        pytest.skip("no NVMM colour-convert element (run inside the ds-on container)")
    try:
        nvbufsurface.load()
    except nvbufsurface.NvBufSurfaceLibraryError:
        pytest.skip("libnvbufsurface.so unavailable (run inside the ds-on container)")


def _export_const_rtdetr_onnx(path: Path, *, h: int, w: int, n: int, c: int) -> None:
    """Const-output RT-DETR ONNX: query 0 → class 1 ('bicycle' in COCO), deterministic."""
    torch = pytest.importorskip("torch", reason="torch authors the ONNX fixture")

    class _ConstRTDETR(torch.nn.Module):
        def forward(self, images: torch.Tensor):  # type: ignore[override]
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


def test_nvmm_buffer_flow_to_wrap_buffer() -> None:
    """Real DeepStream NVMM RGBA buffer -> wrap_buffer -> valid GPU dataPtr (container-unique).

    Proves the live DeepStream NVMM buffer flow feeds a mappable NvBufSurface whose
    dataPtr + geometry our ctypes wrapper reads correctly. Needs only gst +
    libnvbufsurface (NO TensorRT/cuda-python), so it runs in the ds-on container even
    without those wheels. The kernel + TRT-on-dataPtr half is proven on the
    host (test_trt_nvmm_executor / test_nvmm_detector); together they validate the
    full zero-copy path. Exercises the same read-only-map ``from_buffer_copy`` path
    as ``DetectorRunner._on_sample_nvmm``.
    """
    import ctypes

    _require_nvmm_stack()
    from openral_runner.backends.gstreamer import nvbufsurface
    from openral_runner.backends.gstreamer.pipeline import nvmm_convert_element

    conv = nvmm_convert_element()
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=10 ! video/x-raw,width=320,height=240,framerate=30/1 ! "
        f"{conv} ! video/x-raw(memory:NVMM),format=RGBA,width=640,height=640 ! "
        "appsink name=nvmm_sink emit-signals=false sync=false max-buffers=2 drop=true"
    )
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(Gst.SECOND)
    try:
        sample = pipeline.get_by_name("nvmm_sink").emit("try-pull-sample", 5 * Gst.SECOND)
        assert sample is not None, "no NVMM sample pulled within 5s"
        buffer = sample.get_buffer()
        ok_map, map_info = buffer.map(Gst.MapFlags.READ)
        assert ok_map, "NVMM buffer map failed"
        try:
            # NVMM maps read-only -> from_buffer_copy (copies only the metadata struct,
            # not the GPU frame); keep struct_bytes alive while wrap_buffer derefs it.
            struct_bytes = (ctypes.c_uint8 * map_info.size).from_buffer_copy(map_info.data)
            addr = ctypes.cast(struct_bytes, ctypes.c_void_p).value
            assert addr is not None, "NULL mapped base address"
            handle = nvbufsurface.wrap_buffer(addr)
        finally:
            buffer.unmap(map_info)
        assert handle.gpu_ptr > 0, "NULL GPU dataPtr from NVMM surface"
        assert (handle.width, handle.height) == (640, 640), "unexpected NVMM geometry"
        assert handle.pitch >= 640 * 4, "pitch smaller than width*4"
    finally:
        pipeline.set_state(Gst.State.NULL)


def test_nvmm_aggregator_emits_objectsmetadata(tmp_path: Path) -> None:
    """videotestsrc -> tee -> NVMM aggregator branch emits ObjectsMetadata (in-container only)."""
    _require_nvmm_stack()
    pytest.importorskip("tensorrt", reason="tensorrt group not installed in this image")
    pytest.importorskip("cuda", reason="cuda-python not installed in this image")
    import yaml
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.gstreamer.detector_runner import DetectorRunner
    from openral_runner.backends.gstreamer.objects_detector import DetectorTier
    from openral_runner.backends.gstreamer.pipeline import TEE_NAME, leaky_branch

    manifest = RSkillManifest.model_validate(
        yaml.safe_load(
            (_REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml").read_text(encoding="utf-8")
        )
    )
    assert manifest.kind == "detector" and manifest.detector is not None
    net_w, net_h = manifest.detector.input_size  # (width, height) = (640, 640)

    onnx_path = tmp_path / "const_rtdetr.onnx"
    _export_const_rtdetr_onnx(onnx_path, h=net_h, w=net_w, n=10, c=len(manifest.detector.labels))

    pipeline = Gst.parse_launch(
        "videotestsrc is-live=true ! video/x-raw,framerate=30/1,width=320,height=240 ! "
        f"tee name={TEE_NAME}  " + leaky_branch("fakesink sync=false")
    )
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(Gst.SECOND)

    collected: list = []
    runner = DetectorRunner(
        pipeline,
        manifest,
        onnx_path=onnx_path,
        sensor_id="head_rgb",
        on_detection=collected.append,
        tier=DetectorTier.NVMM_AGGREGATOR,
    )
    try:
        runner.start()
        deadline = time.time() + 15.0
        while not collected and time.time() < deadline:
            time.sleep(0.1)
        assert collected, "no ObjectsMetadata from the NVMM aggregator within 15s"
        assert collected[0].detections, "ObjectsMetadata had no detections"
    finally:
        runner.stop()
        pipeline.set_state(Gst.State.NULL)
