"""USB serial enumeration — thin wrapper around ``openral_cli.autodetect``.

Reuses the canonical ``enumerate_usb_devices`` + ``match_known_devices``
pipeline (and the ``_VID_PID_TABLE``) so the auto-detect package never
duplicates USB-probing logic.
"""

from __future__ import annotations

from openral_cli.autodetect import enumerate_usb_devices, match_known_devices

from openral_detect.report import (
    UsbDeviceRecord,
    UsbMatchRecord,
    UsbProbeResult,
)

__all__ = ["probe_usb"]


def probe_usb(*, warnings: list[str] | None = None) -> UsbProbeResult:
    """Enumerate USB serial devices and match them against the VID/PID table.

    Args:
        warnings: Optional list to append non-fatal probe issues to.  When
            ``None``, warnings are silently dropped (the umbrella probe
            owns aggregation).

    Returns:
        A populated :class:`UsbProbeResult`.  Empty when the host has no
        USB serial devices or when ``pyudev`` is unavailable on Linux.
    """
    try:
        devices = enumerate_usb_devices()
    except Exception as exc:  # reason: probe contract — never raise
        if warnings is not None:
            warnings.append(f"usb: enumerate_usb_devices failed: {exc!r}")
        return UsbProbeResult()

    device_records = [
        UsbDeviceRecord(port=d.port, vid=d.vid, pid=d.pid, description=d.description)
        for d in devices
    ]
    match_records: list[UsbMatchRecord] = []
    for m in match_known_devices(devices):
        match_records.append(
            UsbMatchRecord(
                device=UsbDeviceRecord(
                    port=m.device.port,
                    vid=m.device.vid,
                    pid=m.device.pid,
                    description=m.device.description,
                ),
                chip=m.known.chip,
                driver_hint=m.known.driver_hint,
                embodiment_tag=m.known.embodiment_tag,
                bh_robot_type=m.known.bh_robot_type,
            )
        )

    if not device_records and warnings is not None:
        warnings.append("usb: no serial devices found (pyudev / system_profiler).")

    return UsbProbeResult(devices=device_records, matches=match_records)
