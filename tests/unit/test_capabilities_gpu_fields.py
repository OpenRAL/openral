"""Tests for GPU/runtime/dtype fields on :class:`RobotCapabilities`.

These fields exist so ``rSkill.check_capabilities`` can match a host's
inference accelerator support against ``RSkillManifest.runtime`` and
``RSkillManifest.quantization.dtype`` during ``openral detect`` /
``ral skill check`` (see plan: auto-provisioning).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core.schemas import (
    QuantizationDtype,
    RobotCapabilities,
    RobotDescription,
    RSkillRuntime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestRobotCapabilitiesGpuFields:
    def test_defaults_are_empty_or_zero(self) -> None:
        caps = RobotCapabilities()
        assert caps.gpu_vram_gb == 0.0
        assert caps.cuda_compute_capability is None
        assert caps.cuda_toolkit_version is None
        assert caps.tensorrt_version is None
        assert caps.gpu_supported_runtimes == []
        assert caps.gpu_supported_dtypes == []

    def test_populated_fields_round_trip_through_json(self) -> None:
        caps = RobotCapabilities(
            gpu_vram_gb=24.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version="12.4",
            tensorrt_version="10.5",
            gpu_supported_runtimes=[
                RSkillRuntime.PYTORCH,
                RSkillRuntime.ONNX,
                RSkillRuntime.TENSORRT,
            ],
            gpu_supported_dtypes=[
                QuantizationDtype.FP16,
                QuantizationDtype.INT8,
            ],
        )
        rebuilt = RobotCapabilities.model_validate_json(caps.model_dump_json())
        assert rebuilt == caps

    def test_blackwell_fp4_combination(self) -> None:
        caps = RobotCapabilities(
            cuda_compute_capability=(10, 0),
            gpu_supported_dtypes=[QuantizationDtype.FP4_NVFP4],
            gpu_supported_runtimes=[RSkillRuntime.TENSORRT, RSkillRuntime.TRT_LLM],
        )
        assert caps.cuda_compute_capability == (10, 0)
        assert QuantizationDtype.FP4_NVFP4 in caps.gpu_supported_dtypes


class TestVisionSlamCapability:
    """ADR-0064 — `has_vision_slam` gates the camera-based SLAM backend
    (cuVSLAM + nvblox) for lidar-less robots, alongside `has_lidar` which
    gates the 2D lidar `slam_toolbox` backend."""

    def test_defaults_false(self) -> None:
        # Backward-compatible additive field: every existing manifest stays
        # "no vision SLAM" without a schema_version bump (CLAUDE.md §1.6).
        assert RobotCapabilities().has_vision_slam is False

    def test_independent_of_has_lidar(self) -> None:
        caps = RobotCapabilities(has_lidar=False, has_vision_slam=True)
        assert caps.has_lidar is False
        assert caps.has_vision_slam is True

    def test_round_trips_through_json(self) -> None:
        caps = RobotCapabilities(has_vision_slam=True)
        rebuilt = RobotCapabilities.model_validate_json(caps.model_dump_json())
        assert rebuilt.has_vision_slam is True
        assert rebuilt == caps


class TestExistingManifestsLoad:
    """Every committed ``robots/<name>/robot.yaml`` must still load."""

    @pytest.mark.parametrize(
        "manifest",
        sorted((REPO_ROOT / "robots").glob("*/robot.yaml")),
        ids=lambda p: p.parent.name,
    )
    def test_existing_robot_manifests_load_with_default_gpu_fields(self, manifest: Path) -> None:
        desc = RobotDescription.from_yaml(str(manifest))
        # New fields default to "no on-board accelerator" — correct for every
        # passive robot rig committed to the repo today.
        assert desc.capabilities.gpu_vram_gb == 0.0
        assert desc.capabilities.cuda_compute_capability is None
        assert desc.capabilities.gpu_supported_runtimes == []
        assert desc.capabilities.gpu_supported_dtypes == []


class TestYamlRoundTrip:
    def test_yaml_round_trip_preserves_gpu_fields(self, tmp_path: Path) -> None:
        src = REPO_ROOT / "robots" / "so100_follower" / "robot.yaml"
        desc = RobotDescription.from_yaml(str(src))
        # Inject a realistic GPU profile (Jetson Orin AGX).
        desc.capabilities = desc.capabilities.model_copy(
            update={
                "gpu_vram_gb": 64.0,
                "onboard_compute_tops": 275.0,
                "cuda_compute_capability": (8, 7),
                "cuda_toolkit_version": "12.2",
                "tensorrt_version": "8.6",
                "gpu_supported_runtimes": [
                    RSkillRuntime.PYTORCH,
                    RSkillRuntime.ONNX,
                    RSkillRuntime.TENSORRT,
                ],
                "gpu_supported_dtypes": [
                    QuantizationDtype.FP32,
                    QuantizationDtype.FP16,
                    QuantizationDtype.INT8,
                ],
            }
        )
        out = tmp_path / "robot.yaml"
        out.write_text(yaml.safe_dump(desc.model_dump(mode="json")))
        rebuilt = RobotDescription.from_yaml(str(out))
        assert rebuilt.capabilities.gpu_vram_gb == 64.0
        assert rebuilt.capabilities.cuda_compute_capability == (8, 7)
        assert rebuilt.capabilities.gpu_supported_runtimes == [
            RSkillRuntime.PYTORCH,
            RSkillRuntime.ONNX,
            RSkillRuntime.TENSORRT,
        ]
        assert QuantizationDtype.INT8 in rebuilt.capabilities.gpu_supported_dtypes
