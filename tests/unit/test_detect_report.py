"""Tests for :class:`openral_detect.DetectionReport` and helpers.

Hermetic — exercises the schema only, no probes.  Per CLAUDE.md §1.11,
all fixtures use real Pydantic models with realistic values, no mocks.
"""

from __future__ import annotations

import pytest
from openral_core.schemas import QuantizationDtype, RSkillRuntime
from openral_detect import (
    AppleSiliconInfo,
    CameraProbeResult,
    DdsTopicRecord,
    DetectionReport,
    GpuProbeResult,
    JetsonInfo,
    NvidiaGpuInfo,
    RealsenseDeviceInfo,
    Ros2TopologyResult,
)


def _empty(**overrides: object) -> DetectionReport:
    base: dict[str, object] = dict(
        detected_at="2026-05-10T12:34:56Z",
        host_os="Linux 6.18.5",
        python_version="3.12.3",
    )
    base.update(overrides)
    return DetectionReport(**base)


class TestDetectionReportRoundTrip:
    def test_empty_report_round_trips_through_json(self) -> None:
        r = _empty()
        rebuilt = DetectionReport.model_validate_json(r.model_dump_json())
        assert rebuilt == r

    def test_populated_report_round_trips(self) -> None:
        r = _empty(
            gpu=GpuProbeResult(
                nvidia=[
                    NvidiaGpuInfo(
                        index=0,
                        name="NVIDIA GeForce RTX 4090",
                        vram_total_mib=24576,
                        vram_free_mib=24000,
                        pci_bus_id="0000:01:00.0",
                        driver_version="550.78",
                        cuda_compute_capability=(8, 9),
                        cuda_toolkit_version="12.4",
                        tensorrt_version="10.5",
                        supported_dtypes=[
                            QuantizationDtype.FP16,
                            QuantizationDtype.INT8,
                        ],
                        tops_estimate=1321.0,
                    )
                ],
                backend="nvml",
            ),
            cameras=CameraProbeResult(
                realsense=[
                    RealsenseDeviceInfo(
                        serial="123456789",
                        name="Intel RealSense D435I",
                        model_id="D435I",
                    )
                ],
            ),
            ros2=Ros2TopologyResult(
                topics=[DdsTopicRecord(name="/lowstate", type_name="unitree_go/msg/LowState")],
                inferred_robot_type="unitree_g1",
            ),
            warnings=["pyrealsense2 not installed"],
        )
        rebuilt = DetectionReport.model_validate_json(r.model_dump_json())
        assert rebuilt == r

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            DetectionReport.model_validate(
                {
                    "detected_at": "2026-05-10T00:00:00Z",
                    "host_os": "Linux",
                    "python_version": "3.12.3",
                    "unknown_field": True,
                }
            )


class TestDerivedRuntimes:
    def test_no_accelerator_falls_back_to_cpu_runtimes(self) -> None:
        r = _empty()
        assert r.derived_runtimes() == [
            RSkillRuntime.GGUF,
            RSkillRuntime.ONNX,
            RSkillRuntime.PYTORCH,
        ]

    def test_nvidia_card_unlocks_tensorrt_trt_llm_vllm(self) -> None:
        r = _empty(
            gpu=GpuProbeResult(
                nvidia=[
                    NvidiaGpuInfo(
                        index=0,
                        name="RTX 4090",
                        vram_total_mib=24576,
                        vram_free_mib=24000,
                        pci_bus_id="0000:01:00.0",
                        driver_version="550",
                        cuda_compute_capability=(8, 9),
                    )
                ],
                backend="nvml",
            )
        )
        rt = r.derived_runtimes()
        assert RSkillRuntime.TENSORRT in rt
        assert RSkillRuntime.TRT_LLM in rt
        assert RSkillRuntime.VLLM in rt
        assert RSkillRuntime.GGUF not in rt  # discrete-GPU host: no CPU fallback

    def test_jetson_unlocks_tensorrt_but_not_trt_llm_vllm(self) -> None:
        # Orin Nano can run TensorRT but typically not vLLM / TRT-LLM at scale.
        r = _empty(
            gpu=GpuProbeResult(
                jetson=JetsonInfo(board="Jetson Orin Nano", tops=40.0, ram_gb=8.0),
                backend="jtop",
            )
        )
        rt = r.derived_runtimes()
        assert RSkillRuntime.TENSORRT in rt
        assert RSkillRuntime.TRT_LLM not in rt
        assert RSkillRuntime.VLLM not in rt

    def test_apple_silicon_unlocks_mlx(self) -> None:
        r = _empty(
            gpu=GpuProbeResult(
                apple_silicon=AppleSiliconInfo(chip="Apple M3 Max", gpu_cores=40),
                backend="system_profiler",
            )
        )
        rt = r.derived_runtimes()
        assert RSkillRuntime.MLX in rt
        assert RSkillRuntime.PYTORCH in rt


class TestDerivedDtypes:
    def test_empty_report_returns_empty_list(self) -> None:
        # No accelerator → empty list so the matcher skips the dtype check
        # (consistent with the empty-list "unknown" semantics in commit 3).
        assert _empty().derived_dtypes() == []

    def test_accelerator_with_no_dtypes_still_yields_fp32_baseline(self) -> None:
        r = _empty(
            gpu=GpuProbeResult(
                nvidia=[
                    NvidiaGpuInfo(
                        index=0,
                        name="legacy",
                        vram_total_mib=1,
                        vram_free_mib=1,
                        pci_bus_id="x",
                        driver_version="x",
                        cuda_compute_capability=(7, 0),
                    )
                ],
                backend="nvml",
            )
        )
        assert QuantizationDtype.FP32 in r.derived_dtypes()

    def test_union_across_accelerators(self) -> None:
        r = _empty(
            gpu=GpuProbeResult(
                nvidia=[
                    NvidiaGpuInfo(
                        index=0,
                        name="A",
                        vram_total_mib=1,
                        vram_free_mib=1,
                        pci_bus_id="x",
                        driver_version="x",
                        cuda_compute_capability=(8, 9),
                        supported_dtypes=[QuantizationDtype.FP16, QuantizationDtype.INT8],
                    )
                ],
                jetson=JetsonInfo(
                    board="x",
                    cuda_compute_capability=(8, 7),
                    supported_dtypes=[QuantizationDtype.INT4],
                ),
                backend="nvml",
            )
        )
        d = r.derived_dtypes()
        # Union across accelerators — accelerators that supplied dtypes win;
        # FP32 baseline is added only when nobody supplied anything.
        assert QuantizationDtype.FP16 in d
        assert QuantizationDtype.INT8 in d
        assert QuantizationDtype.INT4 in d
