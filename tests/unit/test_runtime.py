"""Unit tests for the Runtime layer: Protocol, NullRuntime, quantization registry,
auto_select_quant, and EngineCache.

PyTorchRuntime and ONNXRuntime tests are skipped when the respective packages
are not installed (marked with pytest.importorskip inside each test method).

Coverage
--------
- ``Runtime`` Protocol: NullRuntime satisfies it; isinstance check works.
- ``NullRuntime``: initial state; load/unload; infer returns {}; quantize/warmup
  no-ops; device property; is_loaded transitions.
- ``Runtime`` Protocol: arbitrary duck-typed class that satisfies the Protocol.
- ``QuantizationDtype`` / ``QuantizationBackend`` enum values.
- ``QuantizationConfig`` defaults and construction.
- ``DeviceInfo`` defaults.
- ``QUANT_PRESETS``: all keys present; values are QuantizationConfig instances.
- ``auto_select_quant``: CPU-only → int8; apple_silicon → bf16; GPU < 4 GB → int4;
  GPU 4-8 GB -> fp16; GPU > 8 GB CC >= 8.0 -> bf16; GPU > 8 GB CC < 8.0 -> fp16;
  no CC but large GPU → fp16.
- ``EngineCache``: cache_key deterministic; get miss → None; put persists; get hit;
  invalidate; clear; size_bytes; entry_count; put raises on missing source.
- ``PyTorchRuntime`` (skipped without torch): device property; load non-existent →
  ROSRuntimeError; unload while not loaded; quantize before load → ROSRuntimeError;
  unsupported quant → ROSRuntimeError.
- ``ONNXRuntime`` (skipped without onnxruntime): device property; MPS raises on init;
  load non-existent → ROSRuntimeError; quantize → ROSRuntimeError always.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import ClassVar

import pytest
from openral_core import (
    DeviceInfo,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
    ROSRuntimeError,
)
from openral_rskill import (
    DEFAULT_CACHE_DIR,
    QUANT_PRESETS,
    EngineCache,
    NullRuntime,
    Runtime,
    auto_select_quant,
)

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


# ── QUANT_PRESETS ─────────────────────────────────────────────────────────────


class TestQuantPresets:
    _EXPECTED_KEYS: ClassVar[set[str]] = {
        "fp32",
        "fp16",
        "bf16",
        "int8_dynamic",
        "int8_dynamic_per_channel",
        "int4",
        "fp4_nvfp4",
        "onnx_int8",
        "trt_int8",
    }

    def test_all_keys_present(self) -> None:
        assert set(QUANT_PRESETS.keys()) == self._EXPECTED_KEYS

    def test_all_values_are_quant_config(self) -> None:
        for key, val in QUANT_PRESETS.items():
            assert isinstance(val, QuantizationConfig), key

    def test_fp32_preset(self) -> None:
        assert QUANT_PRESETS["fp32"].dtype is QuantizationDtype.FP32
        assert QUANT_PRESETS["fp32"].backend is QuantizationBackend.PYTORCH

    def test_int8_dynamic_preset(self) -> None:
        p = QUANT_PRESETS["int8_dynamic"]
        assert p.dtype is QuantizationDtype.INT8
        assert p.per_channel is False

    def test_int8_per_channel_preset(self) -> None:
        p = QUANT_PRESETS["int8_dynamic_per_channel"]
        assert p.dtype is QuantizationDtype.INT8
        assert p.per_channel is True

    def test_fp4_nvfp4_uses_tensorrt_backend(self) -> None:
        assert QUANT_PRESETS["fp4_nvfp4"].backend is QuantizationBackend.TENSORRT

    def test_onnx_int8_uses_onnx_backend(self) -> None:
        assert QUANT_PRESETS["onnx_int8"].backend is QuantizationBackend.ONNX


# ── auto_select_quant ─────────────────────────────────────────────────────────


class TestAutoSelectQuant:
    def test_cpu_only_returns_int8_dynamic(self) -> None:
        info = DeviceInfo(device_str="cpu", gpu_memory_bytes=0)
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.INT8
        assert cfg.backend is QuantizationBackend.PYTORCH

    def test_apple_silicon_returns_bf16(self) -> None:
        info = DeviceInfo(device_str="mps", arch="apple_silicon", gpu_memory_bytes=0)
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.BF16

    def test_small_gpu_under_4gb_returns_int4(self) -> None:
        info = DeviceInfo(device_str="cuda:0", gpu_memory_bytes=2 * _GB)
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.INT4

    def test_mid_gpu_4_to_8gb_returns_fp16(self) -> None:
        for gb in (4, 6, 8):
            info = DeviceInfo(device_str="cuda:0", gpu_memory_bytes=gb * _GB)
            cfg = auto_select_quant(info)
            assert cfg.dtype is QuantizationDtype.FP16, f"gb={gb}"

    def test_large_gpu_ampere_plus_returns_bf16(self) -> None:
        info = DeviceInfo(
            device_str="cuda:0",
            gpu_memory_bytes=24 * _GB,
            cuda_compute_capability=(8, 6),
        )
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.BF16

    def test_large_gpu_volta_returns_fp16(self) -> None:
        info = DeviceInfo(
            device_str="cuda:0",
            gpu_memory_bytes=16 * _GB,
            cuda_compute_capability=(7, 0),
        )
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.FP16

    def test_large_gpu_no_cc_returns_fp16(self) -> None:
        info = DeviceInfo(
            device_str="cuda:0",
            gpu_memory_bytes=12 * _GB,
            cuda_compute_capability=None,
        )
        cfg = auto_select_quant(info)
        assert cfg.dtype is QuantizationDtype.FP16

    def test_exactly_4gb_boundary_is_fp16(self) -> None:
        info = DeviceInfo(device_str="cuda:0", gpu_memory_bytes=4 * _GB)
        assert auto_select_quant(info).dtype is QuantizationDtype.FP16

    def test_exactly_8gb_boundary_is_fp16(self) -> None:
        info = DeviceInfo(device_str="cuda:0", gpu_memory_bytes=8 * _GB)
        assert auto_select_quant(info).dtype is QuantizationDtype.FP16


# ── EngineCache ───────────────────────────────────────────────────────────────


@pytest.fixture()
def cache(tmp_path: pathlib.Path) -> EngineCache:
    return EngineCache(cache_dir=tmp_path / "cache")


@pytest.fixture()
def cfg() -> QuantizationConfig:
    return QuantizationConfig(dtype=QuantizationDtype.INT8)


class TestEngineCache:
    def test_default_cache_dir_is_path(self) -> None:
        assert isinstance(DEFAULT_CACHE_DIR, pathlib.Path)

    def test_init_creates_directory(self, tmp_path: pathlib.Path) -> None:
        subdir = tmp_path / "nested" / "cache"
        EngineCache(cache_dir=subdir)
        assert subdir.is_dir()

    def test_cache_key_is_deterministic(self, cache: EngineCache, cfg: QuantizationConfig) -> None:
        k1 = cache.cache_key("my/skill", "pytorch", cfg)
        k2 = cache.cache_key("my/skill", "pytorch", cfg)
        assert k1 == k2

    def test_cache_key_length(self, cache: EngineCache, cfg: QuantizationConfig) -> None:
        k = cache.cache_key("my/skill", "pytorch", cfg)
        assert len(k) == 16

    def test_cache_key_differs_by_skill_id(
        self, cache: EngineCache, cfg: QuantizationConfig
    ) -> None:
        k1 = cache.cache_key("skill_a", "pytorch", cfg)
        k2 = cache.cache_key("skill_b", "pytorch", cfg)
        assert k1 != k2

    def test_cache_key_differs_by_backend(
        self, cache: EngineCache, cfg: QuantizationConfig
    ) -> None:
        k1 = cache.cache_key("skill", "pytorch", cfg)
        k2 = cache.cache_key("skill", "tensorrt", cfg)
        assert k1 != k2

    def test_cache_key_differs_by_quant(self, cache: EngineCache) -> None:
        k1 = cache.cache_key("skill", "pytorch", QuantizationConfig(dtype=QuantizationDtype.FP16))
        k2 = cache.cache_key("skill", "pytorch", QuantizationConfig(dtype=QuantizationDtype.INT8))
        assert k1 != k2

    def test_get_miss_returns_none(self, cache: EngineCache, cfg: QuantizationConfig) -> None:
        key = cache.cache_key("skill", "pytorch", cfg)
        assert cache.get(key) is None

    def test_put_and_get_hit(
        self, cache: EngineCache, cfg: QuantizationConfig, tmp_path: pathlib.Path
    ) -> None:
        src = tmp_path / "model.engine"
        src.write_bytes(b"fake engine data")
        key = cache.cache_key("skill", "tensorrt", cfg)
        dest = cache.put(key, src)
        assert dest.exists()
        got = cache.get(key)
        assert got is not None
        assert got.read_bytes() == b"fake engine data"

    def test_put_raises_on_missing_source(
        self, cache: EngineCache, cfg: QuantizationConfig, tmp_path: pathlib.Path
    ) -> None:
        key = cache.cache_key("skill", "pytorch", cfg)
        with pytest.raises(FileNotFoundError):
            cache.put(key, tmp_path / "nonexistent.engine")

    def test_invalidate_removes_entry(
        self, cache: EngineCache, cfg: QuantizationConfig, tmp_path: pathlib.Path
    ) -> None:
        src = tmp_path / "m.engine"
        src.write_bytes(b"x")
        key = cache.cache_key("skill", "pytorch", cfg)
        cache.put(key, src)
        cache.invalidate(key)
        assert cache.get(key) is None

    def test_invalidate_miss_is_noop(self, cache: EngineCache, cfg: QuantizationConfig) -> None:
        cache.invalidate("0000000000000000")  # must not raise

    def test_clear_removes_all_entries(self, cache: EngineCache, tmp_path: pathlib.Path) -> None:
        for i in range(3):
            src = tmp_path / f"m{i}.engine"
            src.write_bytes(b"x")
            cache.put(str(i) * 16, src)
        cache.clear()
        assert cache.entry_count == 0

    def test_size_bytes_empty(self, cache: EngineCache) -> None:
        assert cache.size_bytes == 0

    def test_size_bytes_after_put(
        self, cache: EngineCache, cfg: QuantizationConfig, tmp_path: pathlib.Path
    ) -> None:
        data = b"engine" * 100
        src = tmp_path / "m.engine"
        src.write_bytes(data)
        key = cache.cache_key("skill", "pytorch", cfg)
        cache.put(key, src)
        assert cache.size_bytes == len(data)

    def test_entry_count(self, cache: EngineCache, tmp_path: pathlib.Path) -> None:
        assert cache.entry_count == 0
        for i in range(4):
            src = tmp_path / f"e{i}.engine"
            src.write_bytes(b"y")
            cache.put(str(i) * 16, src)
        assert cache.entry_count == 4


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
