"""Per-domain hardware probes for ``openral detect``.

Each probe is a pure function returning a typed Pydantic record from
:mod:`openral_detect.report`.  Probes never raise on missing
optional deps or absent hardware — they return empty records and append
a one-line message to the parent ``DetectionReport.warnings``.
"""

from __future__ import annotations

from openral_detect.probes.cameras import probe_v4l2_cameras
from openral_detect.probes.dds import probe_dds
from openral_detect.probes.gpu import probe_gpus
from openral_detect.probes.network import probe_network
from openral_detect.probes.realsense import probe_realsense_devices
from openral_detect.probes.usb import probe_usb

__all__ = [
    "probe_dds",
    "probe_gpus",
    "probe_network",
    "probe_realsense_devices",
    "probe_usb",
    "probe_v4l2_cameras",
]
