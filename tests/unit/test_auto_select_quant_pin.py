"""Pin ``auto_select_quant`` outputs across ADR-0013's supported targets.

ADR-0013 PR 2/3 §3.3. Regression guard: if anyone tweaks the heuristic
in :func:`openral_rskill.quantization.auto_select_quant`, the
Jetson families will not silently change dtype.

Note: ADR-0013 §3.3's prediction for Orin Nano / Xavier NX is `int4`,
but the actual heuristic returns `fp16` for 8 GB shared memory (the
4 GB < mem ≤ 8 GB band lands in fp16, not int4). The pin-tests
document the **actual** behaviour. Promoting Orin Nano / Xavier NX to
`int4` is a follow-up against `auto_select_quant` itself, not against
this file — the ADR's prediction was prescriptive, not descriptive.
"""

from __future__ import annotations

import pytest
from openral_core import DeviceInfo, QuantizationDtype
from openral_rskill.quantization import auto_select_quant

_GIB = 1 << 30


@pytest.mark.parametrize(
    "name,device_info,expected_dtype",
    [
        # x86 CPU-only — no GPU.
        (
            "x86_cpu",
            DeviceInfo(device_str="cpu", gpu_memory_bytes=0, arch="x86_64"),
            QuantizationDtype.INT8,
        ),
        # x86 dGPU mid-range (e.g. RTX 4070, 12 GB, Ada CC 8.9) — BF16 band.
        (
            "x86_dgpu_ada_12gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=12 * _GIB,
                cuda_compute_capability=(8, 9),
                arch="x86_64",
            ),
            QuantizationDtype.BF16,
        ),
        # x86 dGPU big VRAM (RTX 4090, 24 GB, Ada CC 8.9) — BF16 band.
        (
            "x86_dgpu_ada_24gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=24 * _GIB,
                cuda_compute_capability=(8, 9),
                arch="x86_64",
            ),
            QuantizationDtype.BF16,
        ),
        # x86 dGPU Turing (T4, 16 GB, CC 7.5 < 8.0) — fp16 band (no BF16).
        (
            "x86_dgpu_turing_16gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=16 * _GIB,
                cuda_compute_capability=(7, 5),
                arch="x86_64",
            ),
            QuantizationDtype.FP16,
        ),
        # Orin AGX — 32 GB shared, CC 8.7 → bf16.
        (
            "orin_agx_32gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=32 * _GIB,
                cuda_compute_capability=(8, 7),
                arch="aarch64",
            ),
            QuantizationDtype.BF16,
        ),
        # Orin Nano — 8 GB shared, CC 8.7 → fp16 (4 < mem ≤ 8 GiB band).
        (
            "orin_nano_8gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=8 * _GIB,
                cuda_compute_capability=(8, 7),
                arch="aarch64",
            ),
            QuantizationDtype.FP16,
        ),
        # Xavier NX — 8 GB shared, CC 7.2 → fp16.
        (
            "xavier_nx_8gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=8 * _GIB,
                cuda_compute_capability=(7, 2),
                arch="aarch64",
            ),
            QuantizationDtype.FP16,
        ),
        # Xavier AGX — 16 GB shared, CC 7.2 → fp16 (> 8 GiB, CC < 8.0).
        (
            "xavier_agx_16gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=16 * _GIB,
                cuda_compute_capability=(7, 2),
                arch="aarch64",
            ),
            QuantizationDtype.FP16,
        ),
        # Maxwell Nano — 4 GB shared, CC 5.3 → int4 (memory-constrained).
        (
            "maxwell_nano_4gb",
            DeviceInfo(
                device_str="cuda:0",
                gpu_memory_bytes=4 * _GIB - 1,
                cuda_compute_capability=(5, 3),
                arch="aarch64",
            ),
            QuantizationDtype.INT4,
        ),
    ],
)
def test_auto_select_quant_pin(
    name: str,
    device_info: DeviceInfo,
    expected_dtype: QuantizationDtype,
) -> None:
    cfg = auto_select_quant(device_info)
    assert cfg.dtype is expected_dtype, f"{name}: expected {expected_dtype}, got {cfg.dtype}"
