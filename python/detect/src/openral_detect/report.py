"""Pydantic models for the auto-provisioning detection report.

A :class:`DetectionReport` is a typed snapshot of everything ``openral detect``
discovered on the host: USB controllers, GPUs / Jetson / Apple Silicon,
cameras (V4L2 / RealSense / Orbbec), ROS 2 topology, and network
interfaces.

Every probe that fails (missing optional dep, no hardware, command not on
``$PATH``) appends a typed message to :attr:`DetectionReport.warnings` and
returns an empty result.  **Probes never raise** — that is the contract
that lets ``openral detect`` produce a useful report on bare hosts.

Example:
    >>> from openral_detect.report import DetectionReport, GpuProbeResult
    >>> r = DetectionReport(
    ...     detected_at="2026-05-10T00:00:00Z",
    ...     host_os="Linux 6.18.5",
    ...     python_version="3.12.3",
    ... )
    >>> isinstance(r.warnings, list)
    True
"""

from __future__ import annotations

from typing import Literal

from openral_core.schemas import QuantizationDtype, RSkillRuntime
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AppleSiliconInfo",
    "CameraProbeResult",
    "DetectionReport",
    "GpuProbeResult",
    "JetsonInfo",
    "NetworkInterfaceInfo",
    "NetworkProbeResult",
    "NvidiaGpuInfo",
    "OrbbecDeviceInfo",
    "RealsenseDeviceInfo",
    "Ros2TopologyResult",
    "UsbProbeResult",
    "V4l2CameraInfo",
]

# ── USB ────────────────────────────────────────────────────────────────────────


class UsbDeviceRecord(BaseModel):
    """One USB serial device captured for the report.

    Mirrors :class:`openral_cli.autodetect.UsbDevice` as a Pydantic
    model so the report serializes cleanly through JSON / YAML.
    """

    port: str
    vid: int
    pid: int
    description: str


class UsbMatchRecord(BaseModel):
    """A detected USB device that matched the known-device VID/PID table."""

    device: UsbDeviceRecord
    chip: str
    driver_hint: str
    embodiment_tag: str
    bh_robot_type: str


class UsbProbeResult(BaseModel):
    """USB enumeration output."""

    devices: list[UsbDeviceRecord] = Field(default_factory=list)
    matches: list[UsbMatchRecord] = Field(default_factory=list)


# ── GPU / compute ──────────────────────────────────────────────────────────────


class NvidiaGpuInfo(BaseModel):
    """One discrete NVIDIA GPU.

    Attributes hold everything ``rSkill.check_capabilities`` may need to
    decide whether the host supports a given ``RSkillManifest.runtime``
    or ``quantization.dtype``.
    """

    index: int
    name: str
    vram_total_mib: int
    vram_free_mib: int
    pci_bus_id: str
    driver_version: str
    cuda_compute_capability: tuple[int, int]
    cuda_toolkit_version: str | None = None
    tensorrt_version: str | None = None
    supported_dtypes: list[QuantizationDtype] = Field(default_factory=list)
    tops_estimate: float = 0.0


class JetsonInfo(BaseModel):
    """An NVIDIA Jetson SoC (Orin Nano / NX / AGX, Xavier NX / AGX)."""

    board: str
    soc: str = ""
    jetpack_version: str = ""
    tops: float = 0.0
    ram_gb: float = 0.0
    cuda_compute_capability: tuple[int, int] | None = None
    cuda_toolkit_version: str | None = None
    tensorrt_version: str | None = None
    supported_dtypes: list[QuantizationDtype] = Field(default_factory=list)
    power_mode: str = ""


class AppleSiliconInfo(BaseModel):
    """An Apple Silicon SoC (M-series)."""

    chip: str
    gpu_cores: int = 0
    unified_mem_gb: float = 0.0
    supported_dtypes: list[QuantizationDtype] = Field(default_factory=list)


GpuBackend = Literal[
    "nvml",
    "nvidia-smi",
    "lspci",
    "jtop",
    "tegra-release",
    "system_profiler",
    "none",
]


class GpuProbeResult(BaseModel):
    """Per-host GPU / SoC discovery output."""

    nvidia: list[NvidiaGpuInfo] = Field(default_factory=list)
    jetson: JetsonInfo | None = None
    apple_silicon: AppleSiliconInfo | None = None
    backend: GpuBackend = "none"


# ── Cameras ────────────────────────────────────────────────────────────────────


class V4l2CameraInfo(BaseModel):
    """One V4L2 camera node (Linux ``/dev/video*``)."""

    device_path: str
    name: str
    bus_info: str = ""
    formats: list[str] = Field(default_factory=list)
    max_resolution: tuple[int, int] | None = None


class RealsenseDeviceInfo(BaseModel):
    """One Intel RealSense device discovered via ``pyrealsense2``."""

    serial: str
    name: str
    model_id: str  # e.g. "D435I", "D435", "D455", "D405"
    firmware_version: str = ""
    usb_type: str = ""


class OrbbecDeviceInfo(BaseModel):
    """One Orbbec depth camera discovered via the Orbbec SDK."""

    serial: str
    name: str
    model_id: str
    firmware_version: str = ""


class CameraProbeResult(BaseModel):
    """Per-host camera discovery output (V4L2 + RealSense + Orbbec)."""

    v4l2: list[V4l2CameraInfo] = Field(default_factory=list)
    realsense: list[RealsenseDeviceInfo] = Field(default_factory=list)
    orbbec: list[OrbbecDeviceInfo] = Field(default_factory=list)


# ── ROS 2 topology ────────────────────────────────────────────────────────────


class DdsTopicRecord(BaseModel):
    """One ROS 2 topic discovered during DDS scan."""

    name: str
    type_name: str


class Ros2TopologyResult(BaseModel):
    """ROS 2 topology snapshot (topics, nodes, RMW, domain)."""

    topics: list[DdsTopicRecord] = Field(default_factory=list)
    inferred_robot_type: str | None = None
    has_robot_description: bool = False
    has_tf: bool = False
    nodes: list[str] = Field(default_factory=list)
    rmw_implementation: str = ""
    domain_id: int = 0


# ── Network ────────────────────────────────────────────────────────────────────


class NetworkInterfaceInfo(BaseModel):
    """One network interface (Ethernet, Wi-Fi, loopback, …)."""

    name: str
    mac: str = ""
    ipv4: list[str] = Field(default_factory=list)
    mtu: int = 0
    link_speed_mbps: int | None = None
    is_up: bool = False


class NetworkProbeResult(BaseModel):
    """Per-host network discovery output."""

    hostname: str = ""
    interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)
    default_route: str | None = None


# ── Top-level report ──────────────────────────────────────────────────────────


class DetectionReport(BaseModel):
    """Typed result of a single ``detect_hardware()`` invocation.

    The report is the **only** input to :func:`assemble_robot_description`
    and to :func:`check_installed_rskills`, so all subsequent stages can run
    without re-probing the host.

    Attributes:
        schema_version: Bumped on a breaking change to this report shape.
        detected_at: ISO 8601 UTC timestamp.
        host_os: ``platform.system()`` + ``platform.release()``.
        python_version: ``platform.python_version()``.
        usb: USB enumeration + matched VID/PID rows.
        gpu: NVIDIA / Jetson / Apple Silicon discovery.
        cameras: V4L2 / RealSense / Orbbec live device lists.
        ros2: DDS topology + RMW + domain.
        network: Hostname / interfaces / default route.
        warnings: Non-fatal probe issues — never error.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    detected_at: str
    host_os: str = ""
    python_version: str = ""
    usb: UsbProbeResult = Field(default_factory=UsbProbeResult)
    gpu: GpuProbeResult = Field(default_factory=GpuProbeResult)
    cameras: CameraProbeResult = Field(default_factory=CameraProbeResult)
    ros2: Ros2TopologyResult = Field(default_factory=Ros2TopologyResult)
    network: NetworkProbeResult = Field(default_factory=NetworkProbeResult)
    warnings: list[str] = Field(default_factory=list)

    def derived_runtimes(self) -> list[RSkillRuntime]:
        """Translate detected accelerators into a host-supported runtime list.

        Used by the assembler to populate
        ``RobotCapabilities.gpu_supported_runtimes``.  Conservative — only
        adds runtimes that have a clear hardware story:

        - NVIDIA discrete or Jetson present →
          ``{pytorch, onnx, tensorrt, trt_llm, vllm}``.
        - Apple Silicon present → ``{pytorch, mlx}``.
        - Otherwise (CPU only) → ``{pytorch, onnx, gguf}``.

        Returns:
            Deduplicated, sorted list (sort key = enum value).
        """
        runtimes: set[RSkillRuntime] = set()
        if self.gpu.nvidia or self.gpu.jetson is not None:
            runtimes.update(
                {
                    RSkillRuntime.PYTORCH,
                    RSkillRuntime.ONNX,
                    RSkillRuntime.TENSORRT,
                }
            )
            if self.gpu.nvidia:
                # Only discrete cards realistically host trt-llm / vLLM.
                runtimes.update({RSkillRuntime.TRT_LLM, RSkillRuntime.VLLM})
        if self.gpu.apple_silicon is not None:
            runtimes.update({RSkillRuntime.PYTORCH, RSkillRuntime.MLX})
        if not runtimes:
            runtimes.update({RSkillRuntime.PYTORCH, RSkillRuntime.ONNX, RSkillRuntime.GGUF})
        return sorted(runtimes, key=lambda r: r.value)

    def derived_dtypes(self) -> list[QuantizationDtype]:
        """Union of supported quantization dtypes across detected accelerators.

        Returns ``[]`` when no accelerator advertised any dtype — this keeps
        ``rSkill.check_capabilities`` in the "unknown — skip" branch (per
        commit 3 semantics) rather than rejecting every skill on a host
        whose probes returned nothing.  An assembled list of at least
        ``[FP32]`` is added once any accelerator is present so FP32-only
        manifests pass on every detected host.
        """
        dtypes: set[QuantizationDtype] = set()
        for gpu in self.gpu.nvidia:
            dtypes.update(gpu.supported_dtypes)
        if self.gpu.jetson is not None:
            dtypes.update(self.gpu.jetson.supported_dtypes)
        if self.gpu.apple_silicon is not None:
            dtypes.update(self.gpu.apple_silicon.supported_dtypes)
        # At least one accelerator was detected — guarantee FP32 baseline.
        if (
            self.gpu.nvidia or self.gpu.jetson is not None or self.gpu.apple_silicon is not None
        ) and not dtypes:
            dtypes.add(QuantizationDtype.FP32)
        return sorted(dtypes, key=lambda d: d.value)
