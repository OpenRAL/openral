"""Quantization registry and device-aware auto-selection.

This module defines named quantization presets and a heuristic that picks the
best preset given the host's ``DeviceInfo``.  It has no ML-framework imports.

Public surface
--------------
- ``QUANT_PRESETS``: Dict of named ``QuantizationConfig`` presets.
- ``auto_select_quant``: Select the best preset for a given ``DeviceInfo``.
"""

from __future__ import annotations

from openral_core.schemas import (
    DeviceInfo,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
)

# ── Named presets ─────────────────────────────────────────────────────────────

_CC_BF16_MIN: int = 8
"""Minimum CUDA compute capability major version for hardware-accelerated BF16 (Ampere)."""

QUANT_PRESETS: dict[str, QuantizationConfig] = {
    "fp32": QuantizationConfig(
        dtype=QuantizationDtype.FP32,
        backend=QuantizationBackend.PYTORCH,
    ),
    "fp16": QuantizationConfig(
        dtype=QuantizationDtype.FP16,
        backend=QuantizationBackend.PYTORCH,
    ),
    "bf16": QuantizationConfig(
        dtype=QuantizationDtype.BF16,
        backend=QuantizationBackend.PYTORCH,
    ),
    "int8_dynamic": QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        backend=QuantizationBackend.PYTORCH,
        per_channel=False,
    ),
    "int8_dynamic_per_channel": QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        backend=QuantizationBackend.PYTORCH,
        per_channel=True,
    ),
    "int4": QuantizationConfig(
        dtype=QuantizationDtype.INT4,
        backend=QuantizationBackend.PYTORCH,
    ),
    "fp4_nvfp4": QuantizationConfig(
        dtype=QuantizationDtype.FP4_NVFP4,
        backend=QuantizationBackend.TENSORRT,
    ),
    "onnx_int8": QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        backend=QuantizationBackend.ONNX,
    ),
    "trt_int8": QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        backend=QuantizationBackend.TENSORRT,
    ),
}

# ── Auto-selection heuristic ──────────────────────────────────────────────────

_GB = 1 << 30  # bytes per GiB


def auto_select_quant(device_info: DeviceInfo) -> QuantizationConfig:
    """Return the best ``QuantizationConfig`` for *device_info*.

    Selection strategy (in priority order):

    1. **No GPU** (``gpu_memory_bytes == 0``): ``int8_dynamic`` — CPU-friendly
       dynamic quantization, halves memory footprint with minimal accuracy loss.
    2. **Apple Silicon** (``arch == "apple_silicon"``): ``bf16`` — MLX /
       PyTorch MPS favour BF16; FP16 arithmetic is not natively accelerated.
    3. **GPU < 4 GiB**: ``int4`` — memory-constrained edge GPUs (Jetson Orin 8 GB
       shared, discrete with < 4 GiB VRAM).
    4. **GPU 4-8 GiB**: ``fp16`` — standard mid-range GPU sweet spot.
    5. **GPU > 8 GiB, CUDA CC ≥ 8.0** (Ampere+): ``bf16`` — Ampere+
       hardware accelerates BF16; better numerical stability than FP16.
    6. **GPU > 8 GiB, CUDA CC < 8.0** (Volta/Turing): ``fp16`` — native FP16
       tensor cores, BF16 not accelerated.

    Args:
        device_info: Host compute snapshot.

    Returns:
        A ``QuantizationConfig`` from :data:`QUANT_PRESETS`.

    Example:
        >>> info = DeviceInfo(device_str="cpu", gpu_memory_bytes=0)
        >>> cfg = auto_select_quant(info)
        >>> cfg.dtype
        <QuantizationDtype.INT8: 'int8'>

        >>> info_big_gpu = DeviceInfo(
        ...     device_str="cuda:0",
        ...     gpu_memory_bytes=16 * (1 << 30),
        ...     cuda_compute_capability=(8, 6),
        ... )
        >>> auto_select_quant(info_big_gpu).dtype
        <QuantizationDtype.BF16: 'bf16'>
    """
    mem = device_info.gpu_memory_bytes

    # Apple Silicon always favours BF16 regardless of reported GPU memory
    if device_info.arch == "apple_silicon":
        return QUANT_PRESETS["bf16"]

    if mem == 0:
        return QUANT_PRESETS["int8_dynamic"]

    if mem < 4 * _GB:
        return QUANT_PRESETS["int4"]

    if mem <= 8 * _GB:
        return QUANT_PRESETS["fp16"]

    # > 8 GiB: check CUDA compute capability for BF16 acceleration
    cc = device_info.cuda_compute_capability
    if cc is not None and cc[0] >= _CC_BF16_MIN:  # Ampere (8.0) and newer
        return QUANT_PRESETS["bf16"]

    return QUANT_PRESETS["fp16"]
