"""Umbrella entry point — `detect_hardware()`.

Runs every probe sequentially and aggregates the results into a single
:class:`DetectionReport`.  Probes are wrapped in
``openral_observability.tracing.safety_span`` so the whole detection
pass shows up as a single span tree in OTel.

The function is **synchronous and never raises**.  Probe failures append
typed warnings to :attr:`DetectionReport.warnings`; missing optional
dependencies are *expected* and produce a one-line note, not an error.
"""

from __future__ import annotations

import datetime
import platform

from openral_detect.probes import (
    probe_dds,
    probe_gpus,
    probe_network,
    probe_realsense_devices,
    probe_usb,
    probe_v4l2_cameras,
)
from openral_detect.report import (
    CameraProbeResult,
    DetectionReport,
)

__all__ = ["PROBE_NAMES", "detect_hardware"]

PROBE_NAMES: frozenset[str] = frozenset(
    {"usb", "dds", "gpu", "cameras_v4l2", "cameras_realsense", "network"}
)


def detect_hardware(
    *,
    dds_timeout_s: float = 5.0,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> DetectionReport:
    """Probe every supported domain and return a typed report.

    Args:
        dds_timeout_s: Wall-clock timeout for the DDS topic scan.  Set to
            ``0`` to skip the DDS probe entirely (handy for hosts with no
            ROS 2 sourced).
        include: Optional set of probe names to run; default is "all".
            Recognized names: ``usb``, ``dds``, ``gpu``, ``cameras_v4l2``,
            ``cameras_realsense``, ``network``.  Unknown names append a
            warning but do not raise.
        exclude: Probe names to skip even when ``include`` covers them.

    Returns:
        A :class:`DetectionReport` with every requested probe populated.

    Example:
        >>> from openral_detect import detect_hardware
        >>> r = detect_hardware(include={"network"})
        >>> bool(r.network.hostname)
        True
    """
    requested = (include or set(PROBE_NAMES)) - (exclude or set())
    unknown = requested - PROBE_NAMES
    warnings: list[str] = []
    for name in sorted(unknown):
        warnings.append(f"detect: unknown probe name {name!r} ignored.")
    requested = requested & PROBE_NAMES

    try:
        from openral_observability.tracing import (  # noqa: PLC0415
            safety_span,
        )
    except ImportError:  # pragma: no cover  # reason: observability is a hard dep
        from contextlib import nullcontext  # noqa: PLC0415

        def safety_span(name: str, **_: object) -> object:  # type: ignore[no-redef]
            return nullcontext()

    report_kwargs: dict[str, object] = {}

    if "usb" in requested:
        with safety_span(name="detect.probe.usb", check_name="usb_enumeration"):
            report_kwargs["usb"] = probe_usb(warnings=warnings)

    if "dds" in requested and dds_timeout_s > 0.0:
        with safety_span(name="detect.probe.dds", check_name="dds_topology"):
            report_kwargs["ros2"] = probe_dds(timeout_s=dds_timeout_s, warnings=warnings)

    if "gpu" in requested:
        with safety_span(name="detect.probe.gpu", check_name="gpu_enumeration"):
            report_kwargs["gpu"] = probe_gpus(warnings=warnings)

    cam_v4l2 = []
    cam_realsense = []
    if "cameras_v4l2" in requested:
        with safety_span(name="detect.probe.cameras_v4l2", check_name="v4l2"):
            cam_v4l2 = probe_v4l2_cameras(warnings=warnings)
    if "cameras_realsense" in requested:
        with safety_span(name="detect.probe.cameras_realsense", check_name="realsense"):
            cam_realsense = probe_realsense_devices(warnings=warnings)
    if cam_v4l2 or cam_realsense:
        report_kwargs["cameras"] = CameraProbeResult(v4l2=cam_v4l2, realsense=cam_realsense)

    if "network" in requested:
        with safety_span(name="detect.probe.network", check_name="network"):
            report_kwargs["network"] = probe_network(warnings=warnings)

    detected_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return DetectionReport(
        detected_at=detected_at,
        host_os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        warnings=warnings,
        **report_kwargs,  # type: ignore[arg-type]  # reason: keys match optional fields
    )
