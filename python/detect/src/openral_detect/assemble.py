"""Assemble a complete :class:`RobotDescription` from a `DetectionReport`.

Strategy (per the user's clarification):

1. **Pick a base.**

   - If the host matches a known robot signature
     (``so100`` / ``aloha`` / …) and ``robots/<name>/robot.yaml`` exists,
     load the canonical manifest via
     :meth:`RobotDescription.from_yaml`.  **Standard robot ⇒ standard
     description, untouched** except for sensor / compute enrichment.
   - Otherwise synthesise a minimal scaffold the operator can hand-edit.

2. **Enrich sensors via catalog reverse-lookup.**
   Each detected camera (RealSense, V4L2 USB UVC) is translated into a
   :class:`openral_sensors.SensorSignature`.  When the signature
   resolves to a catalog entry, we call
   ``CATALOG.build(entry.id, ...)`` to materialize a *fully-populated*
   :class:`openral_core.SensorSpec` / :class:`SensorBundle` — real
   intrinsics, FOV, encoding, rate.  Detected serial numbers and
   ``needs_calibration`` flags land in ``SensorSpec.metadata``.

3. **Promote GPU caps onto :class:`RobotCapabilities`.**
   The probed accelerator records (NVIDIA, Jetson, Apple Silicon)
   populate ``onboard_compute_tops``, ``onboard_memory_gb``,
   ``gpu_vram_gb``, ``cuda_compute_capability``,
   ``cuda_toolkit_version``, ``tensorrt_version``,
   ``gpu_supported_runtimes``, ``gpu_supported_dtypes`` so the
   downstream skill compatibility check (commit 8) can match
   ``RSkillManifest.runtime`` and ``quantization.dtype`` against the
   real host.

The function never raises on a missing catalog entry — it falls back to
a generic ``SensorSpec`` and adds a warning to the description's
``onboard_compute["detect_warnings"]`` list.
"""

from __future__ import annotations

import platform
import socket
from typing import Any

from openral_core.schemas import (
    EmbodimentKind,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SensorBundle,
    SensorModality,
    SensorSpec,
    UrdfAsset,
)
from openral_sensors import CATALOG, SensorSignature

from openral_detect.probes.gpu import _probe_nvmm_available
from openral_detect.registry import (
    canonical_robot_path,
    signature_for_realsense,
    signature_for_usb_uvc,
    signature_for_v4l2,
)
from openral_detect.report import (
    DetectionReport,
    GpuProbeResult,
    RealsenseDeviceInfo,
    UsbDeviceRecord,
    V4l2CameraInfo,
)

__all__ = ["assemble_robot_description"]


def assemble_robot_description(
    detection: DetectionReport,
    *,
    base_description: RobotDescription | None = None,
) -> RobotDescription:
    """Build a :class:`RobotDescription` from a probed host.

    Args:
        detection: A :class:`DetectionReport` produced by
            :func:`openral_detect.detect_hardware`.
        base_description: Caller-supplied canonical description to use as
            the base.  When ``None``, the assembler resolves the inferred
            ``bh_robot_type`` (from USB matches or DDS inference) to
            ``robots/<name>/robot.yaml`` and loads it directly.  When no
            known robot matches, a minimal scaffold is synthesised.

    Returns:
        A fully-populated :class:`RobotDescription` ready to be written to
        disk and consumed by :func:`openral_detect.check_installed_rskills`.
    """
    base = base_description or _pick_base(detection)
    base = _enrich_sensors(base, detection)
    base = _enrich_compute(base, detection)
    base = _enrich_ros2(base, detection)
    return base


# ── 1. Pick base ──────────────────────────────────────────────────────────────


def _pick_base(detection: DetectionReport) -> RobotDescription:
    """Look up canonical manifest by inferred robot type or synthesise scaffold."""
    inferred = _infer_bh_robot_type(detection)
    if inferred is not None:
        path = canonical_robot_path(inferred)
        if path is not None:
            return RobotDescription.from_yaml(str(path))
    return _scaffold_unknown(detection)


def _infer_bh_robot_type(detection: DetectionReport) -> str | None:
    if detection.ros2.inferred_robot_type:
        return detection.ros2.inferred_robot_type
    for match in detection.usb.matches:
        if match.bh_robot_type:
            return match.bh_robot_type
    return None


def _scaffold_unknown(detection: DetectionReport) -> RobotDescription:
    hostname = detection.network.hostname or socket.gethostname() or "host"
    return RobotDescription(
        name=f"unknown_{hostname}",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[],
        capabilities=RobotCapabilities(
            embodiment_tags=["unknown"],
        ),
        safety=SafetyEnvelope(),
    )


# ── 2. Enrich sensors ────────────────────────────────────────────────────────


def _enrich_sensors(description: RobotDescription, detection: DetectionReport) -> RobotDescription:
    """Translate probed cameras into catalog-built ``SensorSpec`` / ``SensorBundle``.

    Falls back to a generic spec when no signature matches; the description's
    ``onboard_compute["detect_warnings"]`` carries a one-line note in that
    case so the operator knows which devices need calibration.
    """
    new_sensors: list[SensorSpec] = list(description.sensors)
    new_bundles: list[SensorBundle] = list(description.sensor_bundles)
    detect_warnings: list[str] = []

    realsense_seen: set[str] = {b.bundle_name for b in new_bundles}
    v4l2_seen: set[str] = {s.name for s in new_sensors}

    # ── RealSense ────────────────────────────────────────────────────────────
    for i, dev in enumerate(detection.cameras.realsense):
        bundle_name = _unique_name(f"realsense_{i}", realsense_seen)
        new_bundles.append(
            _build_realsense_bundle(
                dev,
                bundle_name=bundle_name,
                parent_frame=description.base_frame,
                detect_warnings=detect_warnings,
            )
        )
        realsense_seen.add(bundle_name)

    # ── V4L2 / USB UVC ───────────────────────────────────────────────────────
    rs_serials = {dev.serial for dev in detection.cameras.realsense}
    rs_names = {dev.name for dev in detection.cameras.realsense}
    for i, cam in enumerate(detection.cameras.v4l2):
        # Skip RealSense devices that also surfaced via V4L2 to avoid dupes.
        if any(rs_name in cam.name for rs_name in rs_names):
            continue
        spec_name = _unique_name(f"camera_{i}", v4l2_seen)
        new_sensors.append(
            _build_v4l2_camera(
                cam,
                detection_usb=detection.usb.devices,
                spec_name=spec_name,
                parent_frame=description.base_frame,
                detect_warnings=detect_warnings,
            )
        )
        v4l2_seen.add(spec_name)

    onboard = dict(description.onboard_compute)
    if detect_warnings:
        onboard["detect_warnings"] = detect_warnings

    capabilities = description.capabilities.model_copy(
        update={"has_vision": bool(new_sensors or new_bundles)}
    )

    _ = rs_serials  # reserved for future "annotate by serial" pass; keeps lint quiet

    return description.model_copy(
        update={
            "sensors": new_sensors,
            "sensor_bundles": new_bundles,
            "onboard_compute": onboard,
            "capabilities": capabilities,
        },
        deep=True,
    )


def _build_realsense_bundle(
    dev: RealsenseDeviceInfo,
    *,
    bundle_name: str,
    parent_frame: str,
    detect_warnings: list[str],
) -> SensorBundle:
    sig = signature_for_realsense(dev.model_id)
    entry = CATALOG.find_by_signature(sig)
    if entry is None:
        detect_warnings.append(
            f"realsense {dev.model_id} (serial {dev.serial}): no catalog entry; "
            "review intrinsics manually."
        )
        # Fall back to D435 — most common — so the assembler always emits a
        # bundle.  The metadata flags this for follow-up.
        entry = CATALOG.find_by_signature(SensorSignature(kind="realsense", value="D435"))
    assert entry is not None  # at minimum D435 is registered  # reason: invariant
    bundle = CATALOG.build(entry.id, name=bundle_name, parent_frame=parent_frame)
    assert isinstance(bundle, SensorBundle)
    # Annotate every sensor in the bundle with detection metadata.
    for spec in bundle.sensors:
        meta = dict(spec.metadata)
        meta.update(
            {
                "detected_by": "openral detect",
                "needs_calibration": True,
                "serial_no": dev.serial,
                "firmware_version": dev.firmware_version,
                "usb_type": dev.usb_type,
                "catalog_id": entry.id,
            }
        )
        # Pydantic models don't allow attribute mutation by default outside of
        # model_copy, so we replace each sensor in-place.
        # SensorSpec uses use_enum_values=True, but mutation is still allowed
        # because no immutability config is set.
        spec.metadata = meta  # reason: same-package mutation
    return bundle


def _build_v4l2_camera(
    cam: V4l2CameraInfo,
    *,
    detection_usb: list[UsbDeviceRecord],
    spec_name: str,
    parent_frame: str,
    detect_warnings: list[str],
) -> SensorSpec:
    # Try V4L2 product-name signature first.
    entry = None
    for token in cam.name.split():
        sig = signature_for_v4l2(token)
        entry = CATALOG.find_by_signature(sig)
        if entry is not None:
            break
    if entry is None:
        # Try USB VID/PID matching against any device on the same bus.
        for usb in detection_usb:
            if usb.vid and usb.pid:
                sig = signature_for_usb_uvc(usb.vid, usb.pid)
                entry = CATALOG.find_by_signature(sig)
                if entry is not None:
                    break
    if entry is not None:
        spec = CATALOG.build(entry.id, name=spec_name, parent_frame=parent_frame)
        assert isinstance(spec, SensorSpec)
        meta = dict(spec.metadata)
        meta.update(
            {
                "detected_by": "openral detect",
                "needs_calibration": True,
                "device_path": cam.device_path,
                "v4l2_name": cam.name,
                "bus_info": cam.bus_info,
                "catalog_id": entry.id,
            }
        )
        spec.metadata = meta
        return spec

    detect_warnings.append(
        f"v4l2 camera {cam.name!r} ({cam.device_path}): no catalog entry; "
        "wrote a generic RGB SensorSpec — please populate intrinsics."
    )
    return SensorSpec(
        name=spec_name,
        modality=SensorModality.RGB,
        frame_id=f"{spec_name}_optical_frame",
        parent_frame=parent_frame,
        rate_hz=30.0,
        encoding="rgb8",
        vendor=cam.name.split()[0] if cam.name else "unknown",
        model=cam.name,
        driver_pkg="usb_cam",
        metadata={
            "detected_by": "openral detect",
            "needs_calibration": True,
            "device_path": cam.device_path,
            "v4l2_name": cam.name,
            "bus_info": cam.bus_info,
            "catalog_id": "",
        },
    )


# ── 3. Promote GPU caps onto RobotCapabilities ───────────────────────────────


def _enrich_compute(description: RobotDescription, detection: DetectionReport) -> RobotDescription:
    gpu = detection.gpu
    onboard = dict(description.onboard_compute)
    onboard["gpu_probe"] = gpu.model_dump(mode="json")
    onboard["host_os"] = detection.host_os
    onboard["python_version"] = detection.python_version

    tops = _max_tops(gpu)
    vram_gb = _max_vram_gb(gpu)
    memory_gb = _system_memory_gb(gpu)
    cc = _highest_compute_capability(gpu)
    cuda_toolkit = _first_non_empty(
        [g.cuda_toolkit_version for g in gpu.nvidia]
        + ([gpu.jetson.cuda_toolkit_version] if gpu.jetson else [])
    )
    tensorrt_v = _first_non_empty(
        [g.tensorrt_version for g in gpu.nvidia]
        + ([gpu.jetson.tensorrt_version] if gpu.jetson else [])
    )
    runtimes = detection.derived_runtimes()
    dtypes = detection.derived_dtypes()

    capabilities = description.capabilities.model_copy(
        update={
            "onboard_compute_tops": max(description.capabilities.onboard_compute_tops, tops),
            "onboard_memory_gb": max(description.capabilities.onboard_memory_gb, memory_gb),
            "gpu_vram_gb": max(description.capabilities.gpu_vram_gb, vram_gb),
            "cuda_compute_capability": cc or description.capabilities.cuda_compute_capability,
            "cuda_toolkit_version": cuda_toolkit or description.capabilities.cuda_toolkit_version,
            "tensorrt_version": tensorrt_v or description.capabilities.tensorrt_version,
            "gpu_supported_runtimes": runtimes,
            "gpu_supported_dtypes": dtypes,
            "nvmm_available": _probe_nvmm_available(),
        }
    )
    return description.model_copy(
        update={"onboard_compute": onboard, "capabilities": capabilities},
        deep=True,
    )


def _max_tops(gpu: GpuProbeResult) -> float:
    candidates: list[float] = [g.tops_estimate for g in gpu.nvidia]
    if gpu.jetson is not None:
        candidates.append(gpu.jetson.tops)
    return max(candidates) if candidates else 0.0


def _max_vram_gb(gpu: GpuProbeResult) -> float:
    candidates: list[float] = [g.vram_total_mib / 1024.0 for g in gpu.nvidia]
    return max(candidates) if candidates else 0.0


def _system_memory_gb(gpu: GpuProbeResult) -> float:
    if gpu.jetson is not None:
        return gpu.jetson.ram_gb
    if gpu.apple_silicon is not None:
        return gpu.apple_silicon.unified_mem_gb
    return 0.0


def _highest_compute_capability(gpu: GpuProbeResult) -> tuple[int, int] | None:
    ccs: list[tuple[int, int]] = [g.cuda_compute_capability for g in gpu.nvidia]
    if gpu.jetson is not None and gpu.jetson.cuda_compute_capability is not None:
        ccs.append(gpu.jetson.cuda_compute_capability)
    return max(ccs) if ccs else None


def _first_non_empty(values: list[str | None]) -> str | None:
    for v in values:
        if v:
            return v
    return None


# ── 4. ROS 2 metadata ────────────────────────────────────────────────────────


def _enrich_ros2(description: RobotDescription, detection: DetectionReport) -> RobotDescription:
    update: dict[str, Any] = {}
    rmw = detection.ros2.rmw_implementation.lower()
    if "cyclone" in rmw:
        update["middleware"] = "cyclonedds"
    elif "fastrtps" in rmw or "fastdds" in rmw:
        update["middleware"] = "fastdds"
    elif "zenoh" in rmw:
        update["middleware"] = "zenoh"
    if detection.ros2.has_robot_description and description.assets.urdf is None:
        # The robot publishes its own URDF on /robot_description at runtime — mark
        # it with the dynamic ros2:// asset ref (ADR-0057); no static file is
        # vendored, the launch subscribes to the topic instead.
        update["assets"] = description.assets.model_copy(
            update={"urdf": UrdfAsset(ref="ros2://robot_description")}
        )
    if not update:
        return description
    return description.model_copy(update=update)


# ── helpers ──────────────────────────────────────────────────────────────────


def _unique_name(stem: str, taken: set[str]) -> str:
    if stem not in taken:
        return stem
    for i in range(1, 1000):
        candidate = f"{stem}_{i}"
        if candidate not in taken:
            return candidate
    raise RuntimeError(f"could not generate unique name from {stem!r}")


_ = platform  # imported for future host-os checks; keep stable
