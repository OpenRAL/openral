"""Pin the ``FP8`` member of ``QuantizationDtype`` and its TensorRT preset.

FP8 exists in this vocabulary for one reason: an engine backend needs a name
for "8-bit float, E4M3" that is distinct from ``INT8``. The distinction is not
cosmetic — INT8 has a single fixed scale per tensor, which clips the outlier
activations transformer attention produces, while FP8 keeps floating-point
dynamic range at the same 8 bits. A backend that collapsed the two would build
a numerically different engine than the manifest asked for.

The wire value is pinned because it is serialized into rSkill manifests and
into engine-cache keys; renaming it silently invalidates both.
"""

from __future__ import annotations

import pytest
from openral_core import DeviceInfo, QuantizationDtype
from openral_core.schemas import QuantizationBackend
from openral_rskill.quantization import QUANT_PRESETS, auto_select_quant

_GIB = 1 << 30


def test_fp8_wire_value_is_pinned() -> None:
    """``fp8`` is the serialized form — manifests and cache keys depend on it."""
    assert QuantizationDtype.FP8.value == "fp8"
    assert QuantizationDtype("fp8") is QuantizationDtype.FP8


def test_fp8_is_distinct_from_int8() -> None:
    """FP8 must not alias INT8: same bit width, different numerics."""
    assert QuantizationDtype.FP8 is not QuantizationDtype.INT8
    assert QuantizationDtype.FP8.value != QuantizationDtype.INT8.value


def test_trt_fp8_preset_targets_tensorrt() -> None:
    """FP8 reaches an engine as graph Q/DQ nodes, so its only preset is TensorRT."""
    preset = QUANT_PRESETS["trt_fp8"]
    assert preset.dtype is QuantizationDtype.FP8
    assert preset.backend is QuantizationBackend.TENSORRT


@pytest.mark.parametrize(
    ("name", "device_info"),
    [
        ("cpu_only", DeviceInfo(device_str="cpu", gpu_memory_bytes=0, arch="x86_64")),
        (
            "blackwell_128gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=128 * _GIB,
                cuda_compute_capability=(12, 1),
                arch="aarch64",
            ),
        ),
        (
            "ada_8gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=8 * _GIB,
                cuda_compute_capability=(8, 9),
                arch="x86_64",
            ),
        ),
    ],
)
def test_adding_fp8_did_not_change_auto_selection(name: str, device_info: DeviceInfo) -> None:
    """Adding the member must stay additive — the heuristic never picks FP8.

    ``auto_select_quant`` chooses a *runtime* dtype for the PyTorch path, and
    FP8 there would need an engine build the heuristic knows nothing about.
    Selecting it automatically would hand callers a config no PyTorch backend
    can honor, so FP8 stays opt-in via an explicit preset or manifest.
    """
    assert auto_select_quant(device_info).dtype is not QuantizationDtype.FP8, name
