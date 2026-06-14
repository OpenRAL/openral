"""Unit tests for the TensorRT runtime backend (openral_rskill.runtime_tensorrt)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import QuantizationConfig
from openral_rskill.runtime import Runtime
from openral_rskill.runtime_tensorrt import _engine_cache_tag


def _export_tiny_onnx(path: Path) -> None:
    """Export a real 1-op ONNX model (y = x*2 + 1) for a genuine ONNX->TRT build."""
    torch = pytest.importorskip("torch", reason="torch needed to author the ONNX fixture")

    class _Tiny(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0 + 1.0

    dummy = torch.zeros(1, 3, 8, 8)
    torch.onnx.export(
        _Tiny().eval(),
        (dummy,),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamo=False,
    )


def _export_tiny_onnx_dynamic(path: Path) -> None:
    """Export the same y = x*2 + 1 model with a dynamic batch axis.

    Exercises TensorRT's deferred output-shape resolution: the output dim is
    unknown until the input shape is bound at inference time (as for RT-DETR's
    dynamic batch), which is what the two-pass ``infer()`` must handle.
    """
    torch = pytest.importorskip("torch", reason="torch needed to author the ONNX fixture")

    class _Tiny(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0 + 1.0

    dummy = torch.zeros(1, 3, 8, 8)
    torch.onnx.export(
        _Tiny().eval(),
        (dummy,),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        dynamo=False,
    )


def _export_tiny_onnx_external_data(path: Path) -> None:
    """Export a model whose weights live in a sidecar ``*.data`` file.

    Mirrors the RT-DETR detector rSkills: ``tools/export_rtdetr_onnx.py`` emits
    ``model.onnx`` (graph) + ``model.onnx.data`` (external weights). A weight
    large enough to clear the external-data size threshold is needed, so use a
    real Linear layer rather than the constant-only ``_export_tiny_onnx`` model.
    The TRT OnnxParser must resolve the sidecar relative to ``path``'s directory
    (regression for the ``parse_from_file`` fix — ``parse(bytes)`` has no path
    context and fails with "Failed to open file").
    """
    torch = pytest.importorskip("torch", reason="torch needed to author the ONNX fixture")
    onnx = pytest.importorskip("onnx", reason="onnx needed to externalize weights")

    class _Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(64, 64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    dummy = torch.zeros(1, 64)
    tmp_single = path.with_suffix(".single.onnx")
    torch.onnx.export(
        _Linear().eval(),
        (dummy,),
        str(tmp_single),
        input_names=["x"],
        output_names=["y"],
        dynamo=False,
    )
    model = onnx.load(str(tmp_single))
    tmp_single.unlink()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )


def test_engine_cache_tag_is_stable_and_arch_version_specific() -> None:
    tag = _engine_cache_tag((8, 9), "10.5.0")
    assert tag == "tensorrt-sm89-trt10.5.0"
    assert _engine_cache_tag((8, 9), "10.5.0") == tag
    assert _engine_cache_tag((8, 6), "10.5.0") != tag
    assert _engine_cache_tag((8, 9), "10.6.0") != tag


def test_detect_compute_capability_returns_major_minor() -> None:
    pytest.importorskip(
        "cuda.bindings", reason="cuda-python not installed (uv sync --group tensorrt)"
    )
    from openral_rskill.runtime_tensorrt import _detect_compute_capability

    cc = _detect_compute_capability(0)
    assert isinstance(cc, tuple) and len(cc) == 2
    major, minor = cc
    assert isinstance(major, int) and isinstance(minor, int)
    assert major >= 1


def test_construction_defers_heavy_imports_and_reports_state() -> None:
    import sys

    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    # A *prior* test in the same session may have already imported the heavy
    # GPU deps (e.g. _detect_compute_capability imports cuda.bindings). The
    # invariant under test is that *construction* triggers neither import — not
    # that they are globally absent — so snapshot the cache and assert the
    # delta, which is robust to test ordering.
    before = set(sys.modules)
    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-rtdetr-coco-r18")
    newly_imported = set(sys.modules) - before
    assert rt.is_loaded is False
    assert rt.device == "cuda:0"
    assert not any(name == "tensorrt" or name.startswith("cuda.") for name in newly_imported)


def test_construction_rejects_non_cuda_device() -> None:
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with pytest.raises(ROSRuntimeError, match="CUDA"):
        TensorRTRuntime(device="cpu", rskill_id="openral/rskill-rtdetr-coco-r18")


def test_construction_rejects_malformed_device_index() -> None:
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with pytest.raises(ROSRuntimeError):
        TensorRTRuntime(device="cuda:abc", rskill_id="openral/rskill-rtdetr-coco-r18")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_load_builds_then_caches_engine() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    from openral_rskill.engine_cache import EngineCache
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        onnx_path = tmp / "tiny.onnx"
        _export_tiny_onnx(onnx_path)
        cache = EngineCache(cache_dir=tmp / "engines")

        rt1 = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny", cache=cache)
        assert cache.entry_count == 0
        rt1.load(onnx_path)
        assert rt1.is_loaded is True
        assert cache.entry_count == 1

        rt2 = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny", cache=cache)
        rt2.load(onnx_path)
        assert rt2.is_loaded is True
        assert cache.entry_count == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_load_builds_engine_from_external_data_onnx() -> None:
    """ONNX with a sidecar ``*.data`` file builds (RT-DETR detector rSkill shape).

    Regression for ``_build_serialized_engine`` using ``parse_from_file`` so TRT
    resolves the external-data companion relative to the ONNX directory; the old
    ``parse(f.read())`` path failed with "Failed to open file: model.onnx.data".
    """
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "model.onnx"
        _export_tiny_onnx_external_data(onnx_path)
        assert (Path(d) / "model.onnx.data").exists(), "fixture must emit a sidecar .data file"
        rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-extdata")
        rt.load(onnx_path)
        assert rt.is_loaded is True
        rt.unload()


def test_load_missing_file_raises() -> None:
    # No importorskip: load() checks file existence before _import_trt(), so
    # this path runs on hosts without the tensorrt group installed.
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny")
    with pytest.raises(ROSRuntimeError, match="not found"):
        rt.load("/nonexistent/model.onnx")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_infer_matches_reference_math() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "tiny.onnx"
        _export_tiny_onnx(onnx_path)
        rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny")
        rt.load(onnx_path)

        x = np.ones((1, 3, 8, 8), dtype=np.float32)
        out = rt.infer({"x": x})
        assert "y" in out
        np.testing.assert_allclose(out["y"], x * 2.0 + 1.0, rtol=1e-3, atol=1e-3)


def test_infer_without_load_raises() -> None:
    # No importorskip: the no-engine guard raises before any TRT/CUDA import,
    # so this runs on hosts without the tensorrt group installed.
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny")
    with pytest.raises(ROSRuntimeError, match="no engine loaded"):
        rt.infer({"x": np.ones((1, 3, 8, 8), dtype=np.float32)})


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_infer_dynamic_batch_shape() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "tiny_dynamic.onnx"
        _export_tiny_onnx_dynamic(onnx_path)
        rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny-dyn")
        rt.load(onnx_path)

        x = np.ones((2, 3, 8, 8), dtype=np.float32)
        out = rt.infer({"x": x})
        assert out["y"].shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out["y"], x * 2.0 + 1.0, rtol=1e-3, atol=1e-3)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_bf16_build_runs_and_matches_reference() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "tiny.onnx"
        _export_tiny_onnx(onnx_path)
        rt = TensorRTRuntime(
            device="cuda:0",
            rskill_id="openral/rskill-tiny",
            quantization=QuantizationConfig(dtype="bf16"),
        )
        rt.load(onnx_path)
        assert rt.is_loaded is True

        x = np.ones((1, 3, 8, 8), dtype=np.float32)
        out = rt.infer({"x": x})
        np.testing.assert_allclose(out["y"], x * 2.0 + 1.0, rtol=1e-2, atol=0.05)


def test_satisfies_runtime_protocol() -> None:
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny")
    assert isinstance(rt, Runtime)


def test_quantize_rejects_mismatched_config() -> None:
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    rt = TensorRTRuntime(
        device="cuda:0",
        rskill_id="openral/rskill-tiny",
        quantization=QuantizationConfig(dtype="fp16"),
    )
    rt.quantize(QuantizationConfig(dtype="fp16"))
    with pytest.raises(ROSRuntimeError, match="build time"):
        rt.quantize(QuantizationConfig(dtype="int8"))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_unload_clears_state() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed (uv sync --group tensorrt)")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "tiny.onnx"
        _export_tiny_onnx(onnx_path)
        rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/rskill-tiny")
        rt.load(onnx_path)
        assert rt.is_loaded is True
        rt.unload()
        assert rt.is_loaded is False


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_serialized_engine_returns_bytes(tmp_path: Path) -> None:
    """serialized_engine builds (cache miss) and returns deserializable bytes."""
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    onnx_path = tmp_path / "tiny.onnx"
    _export_tiny_onnx(onnx_path)
    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/test-serialized")
    blob = rt.serialized_engine(onnx_path)
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    # Second call is a cache hit and returns identical bytes.
    assert rt.serialized_engine(onnx_path) == blob


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_load_self_heals_corrupt_cache_entry(tmp_path: Path) -> None:
    """A corrupt cached .engine makes load() raise AND purge it so the next load rebuilds."""
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from openral_rskill.engine_cache import EngineCache
    from openral_rskill.runtime_tensorrt import TensorRTRuntime, _import_trt

    onnx_path = tmp_path / "tiny.onnx"
    _export_tiny_onnx(onnx_path)
    cache = EngineCache(cache_dir=tmp_path / "engines")
    rt = TensorRTRuntime(device="cuda:0", rskill_id="openral/test-selfheal", cache=cache)

    # Build once to populate the cache, then corrupt the on-disk engine.
    rt.load(onnx_path)
    rt.unload()
    key = cache.cache_key(rt._rskill_id, rt._backend_tag(_import_trt()), rt._quant)
    cached_path = cache.get(key)
    assert cached_path is not None
    cached_path.write_bytes(b"not a real engine")

    # A corrupt cache HIT must raise and self-heal (invalidate the bad entry).
    with pytest.raises(ROSRuntimeError, match="failed to deserialize"):
        rt.load(onnx_path)
    assert cache.get(key) is None

    # The next load is a cache MISS that rebuilds cleanly.
    rt.load(onnx_path)
    assert rt.is_loaded is True
    assert cache.get(key) is not None
