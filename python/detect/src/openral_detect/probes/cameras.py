"""V4L2 camera probe (Linux ``/dev/video*``).

Shells out to ``v4l2-ctl --list-devices`` because there is no widely
deployed pure-Python v4l2 binding that we can mandate as a dependency.
On hosts without ``v4l2-ctl`` (or non-Linux), returns an empty list and
a warning.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess

from openral_detect.report import V4l2CameraInfo

__all__ = ["probe_v4l2_cameras"]


def probe_v4l2_cameras(*, warnings: list[str] | None = None) -> list[V4l2CameraInfo]:
    """Enumerate V4L2 camera nodes via ``v4l2-ctl --list-devices``.

    Args:
        warnings: Optional list to append non-fatal probe issues to.

    Returns:
        One :class:`V4l2CameraInfo` per camera *device* (a single node group;
        the lowest-numbered ``/dev/videoN`` is reported).  Empty on
        non-Linux hosts or when ``v4l2-ctl`` is not installed.
    """
    sink = warnings if warnings is not None else []
    if platform.system() != "Linux":
        sink.append("cameras.v4l2: skipped (non-Linux host).")
        return []
    cmd = shutil.which("v4l2-ctl")
    if not cmd:
        sink.append("cameras.v4l2: v4l2-ctl not on $PATH (apt install v4l-utils).")
        return []
    try:
        raw = subprocess.check_output(
            [cmd, "--list-devices"], text=True, timeout=3, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        sink.append(f"cameras.v4l2: v4l2-ctl --list-devices failed: {exc!r}")
        return []
    return _parse_v4l2_list_devices(raw)


def _parse_v4l2_list_devices(raw: str) -> list[V4l2CameraInfo]:
    """Parse the ``v4l2-ctl --list-devices`` output.

    The output is a sequence of header lines followed by indented device
    paths::

        HD Pro Webcam C920 (usb-0000:00:14.0-3):
            /dev/video0
            /dev/video1
            /dev/media0

    We take the first ``/dev/video*`` of each block as the canonical
    capture node.
    """
    out: list[V4l2CameraInfo] = []
    current_name: str | None = None
    current_bus: str = ""
    for line in raw.splitlines():
        if not line.strip():
            current_name = None
            current_bus = ""
            continue
        if not line.startswith((" ", "\t")):
            # Header — "Name (bus):"
            header = line.rstrip(":").strip()
            current_name = header
            m = re.search(r"\((.+?)\)\s*$", header)
            if m is not None:
                current_bus = m.group(1)
                current_name = header[: m.start()].strip()
            continue
        path = line.strip()
        if (
            current_name
            and path.startswith("/dev/video")
            # Only the first /dev/videoN per block becomes a row.
            and not any(
                c.device_path.startswith("/dev/video") for c in out if c.name == current_name
            )
        ):
            out.append(
                V4l2CameraInfo(
                    device_path=path,
                    name=current_name,
                    bus_info=current_bus,
                )
            )
    return out
