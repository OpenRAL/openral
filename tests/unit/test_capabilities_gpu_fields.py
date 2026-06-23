"""Tests for :class:`ComputeSpec` and its integration with :class:`RobotDescription`.

After the ``ComputeSpec`` split (ADR-0013 follow-up), GPU / runtime / dtype
fields live on :class:`ComputeSpec` attached to :attr:`RobotDescription.compute`
rather than on :class:`RobotCapabilities`.

``rSkill.check_capabilities`` consumes the compute spec via the optional
``compute=robot.compute`` keyword.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core.schemas import (
    ComputeSpec,
    QuantizationDtype,
    RobotDescription,
    RSkillRuntime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestComputeSpecDefaults:
    def test_defaults_are_empty_or_zero(self) -> None:
        spec = ComputeSpec()
        assert spec.gpu_vram_gb == 0.0
        assert spec.cuda_compute_capability is None
        assert spec.cuda_toolkit_version is None
        assert spec.tensorrt_version is None
        assert spec.gpu_supported_runtimes == []
        assert spec.gpu_supported_dtypes == []
        assert spec.nvmm_available is False

    def test_populated_fields_round_trip_through_json(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=24.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version="12.4",
            tensorrt_version="10.5",
            gpu_supported_runtimes=[RSkillRuntime.PYTORCH, RSkillRuntime.ONNX],
            gpu_supported_dtypes=[QuantizationDtype.FP16, QuantizationDtype.INT8],
        )
        rebuilt = ComputeSpec.model_validate_json(spec.model_dump_json())
        assert rebuilt == spec

    def test_blackwell_fp4_combination(self) -> None:
        spec = ComputeSpec(
            cuda_compute_capability=(10, 0),
            gpu_supported_dtypes=[QuantizationDtype.FP4_NVFP4],
            gpu_supported_runtimes=[RSkillRuntime.TENSORRT, RSkillRuntime.TRT_LLM],
        )
        assert spec.cuda_compute_capability == (10, 0)
        assert QuantizationDtype.FP4_NVFP4 in spec.gpu_supported_dtypes


class TestVisionSlamCapability:
    """ADR-0064 — ``has_vision_slam`` stays on :class:`RobotCapabilities`."""

    def test_defaults_false(self) -> None:
        from openral_core.schemas import RobotCapabilities

        assert RobotCapabilities().has_vision_slam is False

    def test_independent_of_has_lidar(self) -> None:
        from openral_core.schemas import RobotCapabilities

        caps = RobotCapabilities(has_lidar=False, has_vision_slam=True)
        assert caps.has_lidar is False
        assert caps.has_vision_slam is True


class TestSupportsCuMotion:
    """``ComputeSpec.supports_cumotion()`` — the GPU gate for ADR-0065.

    cuMotion requires Ampere+ (CC >= 8.0), CUDA >= 13.0, and a nominal 8 GB
    GPU.  Nominal-8 GB cards report ~7.99 GiB, so the VRAM floor sits just
    below 8.0 GiB to avoid wrongly excluding a real 8 GB card.
    """

    def test_nominal_8gb_ada_cuda13_qualifies(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=8188 / 1024.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version="13.2",
        )
        assert spec.supports_cumotion() is True

    def test_ample_ampere_host_qualifies(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=24.0,
            cuda_compute_capability=(8, 0),
            cuda_toolkit_version="13.0",
        )
        assert spec.supports_cumotion() is True

    def test_no_gpu_does_not_qualify(self) -> None:
        assert ComputeSpec().supports_cumotion() is False

    def test_pre_ampere_does_not_qualify(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=16.0,
            cuda_compute_capability=(7, 5),
            cuda_toolkit_version="13.0",
        )
        assert spec.supports_cumotion() is False

    def test_cuda_below_13_does_not_qualify(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=24.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version="12.4",
        )
        assert spec.supports_cumotion() is False

    def test_insufficient_vram_does_not_qualify(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=6.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version="13.2",
        )
        assert spec.supports_cumotion() is False

    def test_missing_cuda_toolkit_does_not_qualify(self) -> None:
        spec = ComputeSpec(
            gpu_vram_gb=24.0,
            cuda_compute_capability=(8, 9),
            cuda_toolkit_version=None,
        )
        assert spec.supports_cumotion() is False


class TestRobotDescriptionComputeField:
    """``RobotDescription`` compute slots (ADR-0069)."""

    def test_compute_defaults_to_none(self) -> None:
        """All existing manifests load with all compute slots None (no accelerator declared)."""
        for manifest in sorted((REPO_ROOT / "robots").glob("*/robot.yaml")):
            desc = RobotDescription.from_yaml(str(manifest))
            name = manifest.parent.name
            assert desc.compute_edge is None, f"{name}: expected compute_edge=None"
            assert desc.compute_local is None, f"{name}: expected compute_local=None"
            assert desc.compute_cloud is None, f"{name}: expected compute_cloud=None"

    def test_compute_spec_round_trips_on_description(self, tmp_path: Path) -> None:
        src = REPO_ROOT / "robots" / "so100_follower" / "robot.yaml"
        desc = RobotDescription.from_yaml(str(src))
        spec = ComputeSpec(
            gpu_vram_gb=64.0,
            compute_tops=275.0,
            cuda_compute_capability=(8, 7),
            cuda_toolkit_version="12.2",
            tensorrt_version="8.6",
            gpu_supported_runtimes=[RSkillRuntime.PYTORCH, RSkillRuntime.TENSORRT],
            gpu_supported_dtypes=[QuantizationDtype.FP16, QuantizationDtype.INT8],
        )
        desc = desc.model_copy(update={"compute_edge": spec})
        out = tmp_path / "robot.yaml"
        out.write_text(yaml.safe_dump(desc.model_dump(mode="json")))
        rebuilt = RobotDescription.from_yaml(str(out))
        assert rebuilt.compute_edge is not None
        assert rebuilt.compute_local is None
        assert rebuilt.compute_edge.gpu_vram_gb == 64.0
        assert rebuilt.compute_edge.cuda_compute_capability == (8, 7)
        assert rebuilt.compute_edge.supports_cumotion() is False  # CUDA 12.2 < 13


class TestExistingManifestsLoad:
    """Every committed ``robots/<name>/robot.yaml`` still loads cleanly."""

    @pytest.mark.parametrize(
        "manifest",
        sorted((REPO_ROOT / "robots").glob("*/robot.yaml")),
        ids=lambda p: p.parent.name,
    )
    def test_existing_robot_manifests_load(self, manifest: Path) -> None:
        desc = RobotDescription.from_yaml(str(manifest))
        # All compute slots default to None — no accelerator declared.
        assert desc.compute_edge is None
        assert desc.compute_local is None
        assert desc.compute_cloud is None
