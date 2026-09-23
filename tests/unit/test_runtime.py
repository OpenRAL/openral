"""Unit tests for the Runtime layer: Protocol, NullRuntime, and the quantization schemas.

PyTorchRuntime and ONNXRuntime tests are skipped when the respective packages
are not installed (marked with pytest.importorskip inside each test method).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from openral_core import (
    DeviceInfo,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
    ROSRuntimeError,
)
from openral_rskill import NullRuntime, Runtime

_GB = 1 << 30

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None


# Module-level model class so torch.save can pickle it (local classes cannot be pickled)
if _TORCH_AVAILABLE:
    import torch as _th

    class _DictLinear(_th.nn.Module):
        """nn.Module wrapper that accepts and returns dicts (matches VLA convention)."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = _th.nn.Linear(4, 2)

        def forward(self, inputs: dict[str, _th.Tensor]) -> dict[str, _th.Tensor]:
            return {"output": self.linear(inputs["x"])}


# ── Runtime Protocol ──────────────────────────────────────────────────────────


class TestRuntimeProtocol:
    def test_null_runtime_is_runtime(self) -> None:
        assert isinstance(NullRuntime(), Runtime)

    def test_duck_typed_class_is_runtime(self) -> None:
        """Any class with the right shape satisfies Runtime."""

        class _Duck:
            @property
            def is_loaded(self) -> bool:
                return False

            @property
            def device(self) -> str:
                return "cpu"

            def load(self, path: object) -> None: ...

            def infer(self, inputs: object) -> dict[str, object]:
                return {}

            def quantize(self, config: object) -> None: ...

            def warmup(self, inputs: object) -> None: ...

            def unload(self) -> None: ...

        assert isinstance(_Duck(), Runtime)

    def test_incomplete_class_is_not_runtime(self) -> None:
        """A class missing 'device' does not satisfy Runtime."""

        class _Incomplete:
            @property
            def is_loaded(self) -> bool:
                return False

            # missing: device, load, infer, quantize, warmup, unload

        assert not isinstance(_Incomplete(), Runtime)


# ── NullRuntime ───────────────────────────────────────────────────────────────


class TestNullRuntime:
    def test_initial_is_loaded_false(self) -> None:
        assert NullRuntime().is_loaded is False

    def test_device_default(self) -> None:
        assert NullRuntime().device == "cpu"

    def test_custom_device(self) -> None:
        assert NullRuntime(device="cuda:0").device == "cuda:0"

    def test_load_sets_is_loaded(self) -> None:
        rt = NullRuntime()
        rt.load("ignored/path.pt")
        assert rt.is_loaded is True

    def test_unload_clears_is_loaded(self) -> None:
        rt = NullRuntime()
        rt.load("any")
        rt.unload()
        assert rt.is_loaded is False

    def test_unload_before_load_is_safe(self) -> None:
        rt = NullRuntime()
        rt.unload()  # must not raise
        assert rt.is_loaded is False

    def test_infer_returns_empty_dict(self) -> None:
        rt = NullRuntime()
        rt.load("x")
        assert rt.infer({"obs": [1.0, 2.0]}) == {}

    def test_infer_before_load_returns_empty_dict(self) -> None:
        """NullRuntime does not guard against pre-load infer (it's a no-op stub)."""
        assert NullRuntime().infer({}) == {}

    def test_quantize_is_noop(self) -> None:
        rt = NullRuntime()
        rt.quantize(QuantizationConfig())  # must not raise

    def test_warmup_is_noop(self) -> None:
        rt = NullRuntime()
        rt.warmup({})  # must not raise

    def test_load_unload_cycle(self) -> None:
        rt = NullRuntime()
        for _ in range(3):
            rt.load("x")
            assert rt.is_loaded
            rt.unload()
            assert not rt.is_loaded


# ── QuantizationDtype / QuantizationBackend ───────────────────────────────────


class TestQuantizationEnums:
    def test_dtype_values(self) -> None:
        assert QuantizationDtype.FP32.value == "fp32"
        assert QuantizationDtype.FP16.value == "fp16"
        assert QuantizationDtype.BF16.value == "bf16"
        assert QuantizationDtype.INT8.value == "int8"
        assert QuantizationDtype.INT4.value == "int4"
        assert QuantizationDtype.FP4_NVFP4.value == "fp4_nvfp4"

    def test_fp8_wire_value_is_pinned_and_distinct_from_int8(self) -> None:
        # Serialized into rSkill manifests; FP8 (E4M3) must never alias INT8.
        assert QuantizationDtype("fp8") is QuantizationDtype.FP8
        assert QuantizationDtype.FP8 is not QuantizationDtype.INT8

    def test_backend_values(self) -> None:
        assert QuantizationBackend.PYTORCH.value == "pytorch"
        assert QuantizationBackend.ONNX.value == "onnx"
        assert QuantizationBackend.TENSORRT.value == "tensorrt"
        assert QuantizationBackend.GGUF.value == "gguf"
        assert QuantizationBackend.MLX.value == "mlx"


# ── QuantizationConfig ────────────────────────────────────────────────────────


class TestQuantizationConfig:
    def test_defaults(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.dtype is QuantizationDtype.FP32
        assert cfg.backend is QuantizationBackend.PYTORCH
        assert cfg.per_channel is False
        assert cfg.calibration_dataset is None
        assert cfg.extra == {}

    def test_custom_construction(self) -> None:
        cfg = QuantizationConfig(
            dtype=QuantizationDtype.INT8,
            backend=QuantizationBackend.TENSORRT,
            per_channel=True,
            calibration_dataset="lerobot/pusht",
        )
        assert cfg.dtype is QuantizationDtype.INT8
        assert cfg.backend is QuantizationBackend.TENSORRT
        assert cfg.per_channel is True
        assert cfg.calibration_dataset == "lerobot/pusht"

    def test_extra_field(self) -> None:
        cfg = QuantizationConfig(extra={"calibration_steps": 128})
        assert cfg.extra["calibration_steps"] == 128

    def test_json_round_trip(self) -> None:
        cfg = QuantizationConfig(dtype=QuantizationDtype.INT8)
        restored = QuantizationConfig.model_validate_json(cfg.model_dump_json())
        assert restored == cfg


# ── DeviceInfo ────────────────────────────────────────────────────────────────


class TestDeviceInfo:
    def test_defaults(self) -> None:
        info = DeviceInfo()
        assert info.device_str == "cpu"
        assert info.gpu_memory_bytes == 0
        assert info.cuda_compute_capability is None
        assert info.cpu_count == 1
        assert info.arch == "x86_64"

    def test_gpu_device(self) -> None:
        info = DeviceInfo(
            device_str="cuda:0",
            gpu_memory_bytes=24 * _GB,
            cuda_compute_capability=(8, 9),
        )
        assert info.cuda_compute_capability == (8, 9)


# ── PyTorchRuntime (skipped without torch) ────────────────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestPyTorchRuntime:
    """Tests that do not require real checkpoint files."""

    def _rt(self) -> object:
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        return PyTorchRuntime(device="cpu")

    def test_device_property(self) -> None:
        rt = self._rt()
        assert rt.device == "cpu"  # type: ignore[union-attr]

    def test_initial_not_loaded(self) -> None:
        rt = self._rt()
        assert rt.is_loaded is False  # type: ignore[union-attr]

    def test_load_nonexistent_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="not found"):
            rt.load("/nonexistent/path/model.pt")  # type: ignore[union-attr]

    def test_infer_before_load_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="no model loaded"):
            rt.infer({})  # type: ignore[union-attr]

    def test_quantize_before_load_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="before load"):
            rt.quantize(QuantizationConfig(dtype=QuantizationDtype.INT8))  # type: ignore[union-attr]

    def test_load_unsafe_pickle_refused_without_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        monkeypatch.delenv("OPENRAL_ALLOW_UNSAFE_PICKLE", raising=False)
        model = th.nn.Linear(2, 2)
        path = tmp_path / "model.pt"
        th.save(model, str(path))
        rt = PyTorchRuntime(device="cpu")
        with pytest.raises(ROSRuntimeError, match="remote-code-execution"):
            rt.load(path)
        assert rt.is_loaded is False

    def test_load_safetensors_works_without_pickle_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safe path loads a state_dict from .safetensors with NO unsafe-pickle
        env and round-trips weights into a fresh module (security audit C2)."""
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime
        from safetensors.torch import save_file

        # Prove the safe path needs no acknowledgement env.
        monkeypatch.delenv("OPENRAL_ALLOW_UNSAFE_PICKLE", raising=False)

        trained = _DictLinear()
        with th.no_grad():
            trained.linear.weight.fill_(0.5)
            trained.linear.bias.fill_(0.25)
        sd_path = tmp_path / "model.safetensors"
        save_file(trained.state_dict(), str(sd_path))

        rt = PyTorchRuntime(device="cpu")
        fresh = _DictLinear()  # caller supplies the architecture
        rt.load_safetensors(sd_path, model=fresh)
        assert rt.is_loaded

        x = {"x": th.ones(1, 4)}
        out = rt.infer(x)["output"]
        # 0.5*sum(ones(4)) + 0.25 == 2.25 per output unit — weights actually loaded.
        assert th.allclose(out, th.full((1, 2), 2.25))

    def test_load_safetensors_missing_file_raises(self, tmp_path: pathlib.Path) -> None:
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        rt = PyTorchRuntime(device="cpu")
        with pytest.raises(ROSRuntimeError, match="not found"):
            rt.load_safetensors(tmp_path / "nope.safetensors", model=_DictLinear())

    def test_load_safetensors_arch_mismatch_raises(self, tmp_path: pathlib.Path) -> None:
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime
        from safetensors.torch import save_file

        save_file(th.nn.Linear(8, 8).state_dict(), str(tmp_path / "wrong.safetensors"))
        rt = PyTorchRuntime(device="cpu")
        with pytest.raises(ROSRuntimeError, match="does not fit"):
            rt.load_safetensors(tmp_path / "wrong.safetensors", model=_DictLinear())

    def test_load_refuses_malicious_pickle_before_executing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint with a code-executing ``__reduce__`` must be refused
        BEFORE deserialization, so the payload never runs (security audit C2).

        This is the real RCE the guard exists to stop: ``torch.save`` pickles the
        object, and a vanilla ``torch.load(weights_only=False)`` would invoke the
        ``__reduce__`` and run the command. With the guard and the env unset, the
        load is rejected first, so the canary file is never created.
        """
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        canary = tmp_path / "pwned"

        class _Evil:
            def __reduce__(self) -> tuple[object, tuple[str]]:
                import os

                return (os.system, (f"touch {canary}",))

        payload = tmp_path / "evil.pt"
        th.save(_Evil(), str(payload))

        monkeypatch.delenv("OPENRAL_ALLOW_UNSAFE_PICKLE", raising=False)
        rt = PyTorchRuntime(device="cpu")
        with pytest.raises(ROSRuntimeError, match="remote-code-execution"):
            rt.load(payload)
        assert not canary.exists(), "malicious __reduce__ executed despite refusal"

    def test_unsupported_quant_raises_after_load(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        monkeypatch.setenv("OPENRAL_ALLOW_UNSAFE_PICKLE", "1")
        model = th.nn.Linear(2, 2)
        path = tmp_path / "model.pt"
        th.save(model, str(path))
        rt = PyTorchRuntime(device="cpu")
        rt.load(path)
        with pytest.raises(ROSRuntimeError, match="unsupported"):
            rt.quantize(QuantizationConfig(dtype=QuantizationDtype.FP4_NVFP4))

    def test_full_lifecycle_with_linear_model(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch as th
        from openral_rskill.runtime_pytorch import PyTorchRuntime

        monkeypatch.setenv("OPENRAL_ALLOW_UNSAFE_PICKLE", "1")
        model = _DictLinear()
        path = tmp_path / "linear.pt"
        th.save(model, str(path))

        rt = PyTorchRuntime(device="cpu")
        assert not rt.is_loaded
        rt.load(path)
        assert rt.is_loaded

        dummy = {"x": th.zeros(1, 4)}
        rt.warmup(dummy)
        out = rt.infer(dummy)
        assert "output" in out

        # Dynamic INT8 quantization needs a registered backend qengine
        # (fbgemm on x86, qnnpack on ARM).  Some macOS arm64 torch wheels
        # advertise an engine in `supported_engines` but still raise
        # `NoQEngine` from `quantized::linear_prepack`; treat that as a
        # platform-skip rather than a test failure, but still exercise the
        # unload leg below.
        skip_msg: str | None = None
        try:
            rt.quantize(QuantizationConfig(dtype=QuantizationDtype.INT8))
            out2 = rt.infer(dummy)
            assert "output" in out2
        except RuntimeError as exc:
            if "NoQEngine" not in str(exc) and "quantized engine" not in str(exc):
                raise
            skip_msg = f"no torch quantized engine available: {exc}"

        rt.unload()
        assert not rt.is_loaded

        if skip_msg is not None:
            pytest.skip(skip_msg)


# ── ONNXRuntime (skipped without onnxruntime) ─────────────────────────────────


@pytest.mark.skipif(not _ORT_AVAILABLE, reason="onnxruntime not installed")
class TestONNXRuntime:
    def _rt(self, device: str = "cpu") -> object:
        from openral_rskill.runtime_onnx import ONNXRuntime

        return ONNXRuntime(device=device)

    def test_device_property(self) -> None:
        rt = self._rt()
        assert rt.device == "cpu"  # type: ignore[union-attr]

    def test_initial_not_loaded(self) -> None:
        rt = self._rt()
        assert rt.is_loaded is False  # type: ignore[union-attr]

    def test_mps_raises_on_init(self) -> None:
        from openral_rskill.runtime_onnx import ONNXRuntime

        with pytest.raises(ROSRuntimeError, match="MPS"):
            ONNXRuntime(device="mps")

    def test_load_nonexistent_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="not found"):
            rt.load("/nonexistent/model.onnx")  # type: ignore[union-attr]

    def test_quantize_always_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="offline"):
            rt.quantize(QuantizationConfig())  # type: ignore[union-attr]

    def test_infer_before_load_raises(self) -> None:
        rt = self._rt()
        with pytest.raises(ROSRuntimeError, match="no model loaded"):
            rt.infer({})  # type: ignore[union-attr]

    def test_full_lifecycle_with_exported_model(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("torch", reason="torch required to export ONNX")
        import numpy as np
        import torch as th
        from openral_rskill.runtime_onnx import ONNXRuntime

        model = th.nn.Linear(3, 2)
        path = tmp_path / "model.onnx"
        th.onnx.export(
            model,
            th.zeros(1, 3),
            str(path),
            input_names=["x"],
            output_names=["y"],
            opset_version=17,
        )
        rt = ONNXRuntime(device="cpu")
        rt.load(path)
        assert rt.is_loaded

        out = rt.infer({"x": np.zeros((1, 3), dtype=np.float32)})
        assert "y" in out

        rt.unload()
        assert not rt.is_loaded
