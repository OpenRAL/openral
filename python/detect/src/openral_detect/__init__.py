"""openral auto-provisioning — `openral detect` machinery."""

__version__ = "0.1.0"

from openral_detect.assemble import assemble_robot_description, build_compute_spec
from openral_detect.compatibility import (
    CompatibilityReport,
    RSkillCompatRow,
    SectionVerdict,
    check_installed_rskills,
    check_single_rskill,
)
from openral_detect.detect import PROBE_NAMES, detect_hardware
from openral_detect.report import (
    AppleSiliconInfo,
    CameraProbeResult,
    DdsTopicRecord,
    DetectionReport,
    GpuProbeResult,
    JetsonInfo,
    NetworkInterfaceInfo,
    NetworkProbeResult,
    NvidiaGpuInfo,
    OrbbecDeviceInfo,
    RealsenseDeviceInfo,
    Ros2TopologyResult,
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
    V4l2CameraInfo,
)
from openral_detect.scaffold import ScaffoldOverrides, scaffold_robot_environment

__all__ = [
    "PROBE_NAMES",
    "AppleSiliconInfo",
    "CameraProbeResult",
    "CompatibilityReport",
    "DdsTopicRecord",
    "DetectionReport",
    "GpuProbeResult",
    "JetsonInfo",
    "NetworkInterfaceInfo",
    "NetworkProbeResult",
    "NvidiaGpuInfo",
    "OrbbecDeviceInfo",
    "RSkillCompatRow",
    "RealsenseDeviceInfo",
    "Ros2TopologyResult",
    "ScaffoldOverrides",
    "SectionVerdict",
    "UsbDeviceRecord",
    "UsbMatchRecord",
    "UsbProbeResult",
    "V4l2CameraInfo",
    "assemble_robot_description",
    "build_compute_spec",
    "check_installed_rskills",
    "check_single_rskill",
    "detect_hardware",
    "scaffold_robot_environment",
]
