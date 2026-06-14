"""Tests for :func:`openral_detect.assemble_robot_description`.

Exercises the identify-then-enrich flow end-to-end with **real**
fixtures — the canonical ``robots/so100_follower/robot.yaml`` and the
real ``CATALOG`` populated by every vendor module on import.  No mocks.
"""

from __future__ import annotations

from pathlib import Path

from openral_core.schemas import (
    QuantizationDtype,
    RobotDescription,
    RSkillRuntime,
    SensorBundle,
)
from openral_detect import (
    CameraProbeResult,
    DetectionReport,
    GpuProbeResult,
    JetsonInfo,
    NvidiaGpuInfo,
    RealsenseDeviceInfo,
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
    V4l2CameraInfo,
    assemble_robot_description,
)
from openral_detect.report import Ros2TopologyResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _empty_report(**overrides: object) -> DetectionReport:
    base: dict[str, object] = dict(
        detected_at="2026-05-10T00:00:00Z",
        host_os="Linux 6.18",
        python_version="3.12.3",
    )
    base.update(overrides)
    return DetectionReport(**base)


class TestStandardRobot:
    """A known robot signature => canonical robot.yaml is the base."""

    def test_so100_usb_match_loads_canonical_manifest(self) -> None:
        usb_dev = UsbDeviceRecord(
            port="/dev/ttyUSB0",
            vid=0x1A86,
            pid=0x7523,
            description="CH340 Feetech bus",
        )
        report = _empty_report(
            usb=UsbProbeResult(
                devices=[usb_dev],
                matches=[
                    UsbMatchRecord(
                        device=usb_dev,
                        chip="CH340",
                        driver_hint="Feetech serial bus — SO-100",
                        embodiment_tag="so100_follower",
                        bh_robot_type="so100",
                    )
                ],
            ),
        )
        desc = assemble_robot_description(report)
        # Standard robot ⇒ canonical manifest used as-is.
        assert desc.name == "so100_follower"
        assert "so100_follower" in desc.capabilities.embodiment_tags

    def test_dds_topology_inference_picks_canonical_manifest(self) -> None:
        report = _empty_report(
            ros2=Ros2TopologyResult(
                topics=[],
                inferred_robot_type="so100",
            ),
        )
        desc = assemble_robot_description(report)
        assert desc.name == "so100_follower"


class TestUnknownRobotScaffold:
    def test_no_match_emits_unknown_scaffold(self) -> None:
        report = _empty_report()
        desc = assemble_robot_description(report)
        assert desc.name.startswith("unknown_")
        assert desc.capabilities.embodiment_tags == ["unknown"]


class TestSensorEnrichment:
    def test_realsense_d435i_detected_to_catalog_bundle(self) -> None:
        report = _empty_report(
            cameras=CameraProbeResult(
                realsense=[
                    RealsenseDeviceInfo(
                        serial="ABC123",
                        name="Intel RealSense D435I",
                        model_id="D435I",
                    )
                ]
            ),
        )
        desc = assemble_robot_description(report)
        # Last bundle is the freshly-attached RealSense.
        bundle = desc.sensor_bundles[-1]
        assert isinstance(bundle, SensorBundle)
        # Real intrinsics — pulled from the catalog factory, not invented.
        rgb = next(s for s in bundle.sensors if s.modality == "rgb")
        assert rgb.intrinsics is not None
        assert rgb.intrinsics.fx > 100.0
        # Serial threaded into metadata for downstream identification.
        assert rgb.metadata["serial_no"] == "ABC123"
        assert rgb.metadata["catalog_id"] == "intel/realsense_d435i"
        assert rgb.metadata["needs_calibration"] is True

    def test_unknown_realsense_falls_back_to_d435_with_warning(self) -> None:
        report = _empty_report(
            cameras=CameraProbeResult(
                realsense=[
                    RealsenseDeviceInfo(serial="X", name="RealSense D9999", model_id="D9999")
                ]
            ),
        )
        desc = assemble_robot_description(report)
        assert desc.sensor_bundles, "should still emit a bundle for follow-up"
        assert any("no catalog entry" in w for w in desc.onboard_compute.get("detect_warnings", []))

    def test_v4l2_logitech_c920_detected_to_catalog_spec(self) -> None:
        report = _empty_report(
            cameras=CameraProbeResult(
                v4l2=[
                    V4l2CameraInfo(
                        device_path="/dev/video0",
                        name="HD Pro Webcam C920",
                        bus_info="usb-0000:00:14.0-3",
                    )
                ]
            ),
        )
        desc = assemble_robot_description(report)
        # The C920 V4L2 name must resolve via signature_for_v4l2 → logitech/c920.
        last = desc.sensors[-1]
        assert last.metadata["catalog_id"] == "logitech/c920"
        assert last.intrinsics is not None  # real intrinsics, not invented

    def test_unknown_v4l2_camera_falls_back_to_generic_spec(self) -> None:
        report = _empty_report(
            cameras=CameraProbeResult(
                v4l2=[
                    V4l2CameraInfo(
                        device_path="/dev/video0",
                        name="Acme Cam 9000",
                        bus_info="",
                    )
                ]
            ),
        )
        desc = assemble_robot_description(report)
        last = desc.sensors[-1]
        # Generic spec — no catalog hit.
        assert last.metadata["catalog_id"] == ""
        assert last.metadata["needs_calibration"] is True
        assert any("no catalog entry" in w for w in desc.onboard_compute.get("detect_warnings", []))


class TestComputeEnrichment:
    def test_rtx_4090_promotes_caps(self) -> None:
        report = _empty_report(
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
                        supported_dtypes=[QuantizationDtype.FP16, QuantizationDtype.INT8],
                        tops_estimate=1321.0,
                    )
                ],
                backend="nvml",
            ),
        )
        desc = assemble_robot_description(report)
        assert desc.capabilities.onboard_compute_tops == 1321.0
        assert desc.capabilities.gpu_vram_gb == 24.0
        assert desc.capabilities.cuda_compute_capability == (8, 9)
        assert desc.capabilities.cuda_toolkit_version == "12.4"
        assert desc.capabilities.tensorrt_version == "10.5"
        # Discrete NVIDIA host unlocks TensorRT / TRT-LLM / vLLM.
        assert RSkillRuntime.TENSORRT in desc.capabilities.gpu_supported_runtimes
        assert RSkillRuntime.TRT_LLM in desc.capabilities.gpu_supported_runtimes
        # Onboard compute blob captures the raw probe payload.
        assert "gpu_probe" in desc.onboard_compute

    def test_jetson_orin_promotes_caps(self) -> None:
        report = _empty_report(
            gpu=GpuProbeResult(
                jetson=JetsonInfo(
                    board="Jetson AGX Orin",
                    tops=275.0,
                    ram_gb=64.0,
                    cuda_compute_capability=(8, 7),
                    supported_dtypes=[QuantizationDtype.INT8],
                ),
                backend="jtop",
            ),
        )
        desc = assemble_robot_description(report)
        assert desc.capabilities.onboard_compute_tops == 275.0
        assert desc.capabilities.onboard_memory_gb == 64.0
        assert desc.capabilities.cuda_compute_capability == (8, 7)


class TestRos2Metadata:
    def test_robot_description_topic_sets_urdf_sentinel(self) -> None:
        report = _empty_report(
            ros2=Ros2TopologyResult(
                topics=[],
                has_robot_description=True,
                rmw_implementation="rmw_cyclonedds_cpp",
            ),
        )
        desc = assemble_robot_description(report)
        assert desc.urdf_path == "ros2://robot_description"
        assert desc.middleware == "cyclonedds"


class TestIdempotence:
    def test_assemble_does_not_mutate_canonical_manifest(self) -> None:
        # Two assemblers from independent reports must yield independent
        # capability lists — proves the deep-copy semantics.
        r1 = _empty_report(
            gpu=GpuProbeResult(
                jetson=JetsonInfo(
                    board="Jetson AGX Orin",
                    tops=275.0,
                    ram_gb=64.0,
                    cuda_compute_capability=(8, 7),
                ),
                backend="jtop",
            ),
        )
        d1 = assemble_robot_description(r1)
        r2 = _empty_report()
        d2 = assemble_robot_description(r2)
        assert d1.capabilities.onboard_compute_tops == 275.0
        assert d2.capabilities.onboard_compute_tops == 0.0


class TestRoundTripAfterAssemble:
    def test_assembled_description_round_trips_through_yaml(self, tmp_path: Path) -> None:
        import yaml

        report = _empty_report(
            cameras=CameraProbeResult(
                realsense=[
                    RealsenseDeviceInfo(serial="X", name="Intel RealSense D435I", model_id="D435I")
                ]
            ),
        )
        desc = assemble_robot_description(report)
        path = tmp_path / "robot.yaml"
        path.write_text(yaml.safe_dump(desc.model_dump(mode="json")))
        rebuilt = RobotDescription.from_yaml(str(path))
        assert rebuilt.sensor_bundles[-1].bundle_name.startswith("realsense_")
