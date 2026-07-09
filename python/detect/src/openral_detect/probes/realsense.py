"""Live Intel RealSense device probe via ``pyrealsense2``.

Returns one :class:`RealsenseDeviceInfo` per device the SDK can see.
Empty list when the SDK is not installed (Apple Silicon, headless CI),
or when no devices are connected — never raises.
"""

from __future__ import annotations

from openral_detect.report import RealsenseDeviceInfo

__all__ = ["probe_realsense_devices"]


def probe_realsense_devices(*, warnings: list[str] | None = None) -> list[RealsenseDeviceInfo]:
    """Enumerate connected Intel RealSense devices.

    The ``model_id`` field is the canonical key used by
    :class:`openral_sensors.SensorSignature` (kind=``"realsense"``)
    so the assembler can call ``CATALOG.find_by_signature(...)`` and
    materialize a fully-populated ``SensorBundle`` with real intrinsics.
    """
    sink = warnings if warnings is not None else []
    try:
        import pyrealsense2 as rs  # noqa: PLC0415  # reason: optional extra
    except ImportError:
        sink.append(
            "cameras.realsense: pyrealsense2 not installed — install the "
            "[realsense] extra on an x86 Linux host."
        )
        return []
    try:
        ctx = rs.context()
        devices = list(ctx.devices)
    except Exception as exc:
        sink.append(f"cameras.realsense: SDK enumeration failed: {exc!r}")
        return []
    if not devices:
        sink.append("cameras.realsense: SDK loaded but no devices connected.")
        return []

    out: list[RealsenseDeviceInfo] = []
    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            firmware = (
                dev.get_info(rs.camera_info.firmware_version)
                if dev.supports(rs.camera_info.firmware_version)
                else ""
            )
            usb = (
                dev.get_info(rs.camera_info.usb_type_descriptor)
                if dev.supports(rs.camera_info.usb_type_descriptor)
                else ""
            )
        except Exception as exc:
            sink.append(f"cameras.realsense: get_info failed for one device: {exc!r}")
            continue
        out.append(
            RealsenseDeviceInfo(
                serial=str(serial),
                name=str(name),
                model_id=_extract_model_id(str(name)),
                firmware_version=str(firmware),
                usb_type=str(usb),
            )
        )
    return out


def _extract_model_id(name: str) -> str:
    """Pull the canonical model id (e.g. ``D435I``) from the SDK name string.

    The SDK reports names like ``"Intel RealSense D435I"`` /
    ``"Intel(R) RealSense(TM) D455"``.  We pick the last whitespace-
    separated token starting with a capital ``D`` followed by digits.
    """
    for token in reversed(name.split()):
        cleaned = token.replace("(R)", "").replace("(TM)", "").strip()
        if cleaned.startswith("D") and len(cleaned) >= 4 and cleaned[1].isdigit():  # noqa: PLR2004
            return cleaned.upper()
    return name
