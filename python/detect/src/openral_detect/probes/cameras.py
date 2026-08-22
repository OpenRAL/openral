"""V4L2 camera probe (Linux ``/dev/video*``).

Two backends, tried in order:

1. **sysfs** (``/sys/class/video4linux``) — always present on Linux, needs
   no package and no root, and is the only backend that reads the camera's
   own USB descriptor.  That descriptor is what resolves a ``usb_uvc``
   catalog signature, so a camera found this way arrives with real
   intrinsics rather than a generic placeholder.
2. **``v4l2-ctl --list-devices``** — used when sysfs yields nothing (an
   unusual kernel layout), and as the enrichment source for ``bus_info``
   strings on hosts that have ``v4l-utils`` installed.

The previous implementation shelled out to ``v4l2-ctl`` only, which meant a
host without ``v4l-utils`` reported *zero* cameras while several were plugged
in — a silent false negative on exactly the provisioning path that is
supposed to find them.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess

from openral_detect.report import V4l2CameraInfo

__all__ = ["probe_v4l2_cameras"]

_SYSFS_V4L2 = "/sys/class/video4linux"


def probe_v4l2_cameras(*, warnings: list[str] | None = None) -> list[V4l2CameraInfo]:
    """Enumerate V4L2 cameras, one row per physical device.

    Args:
        warnings: Optional list to append non-fatal probe issues to.

    Returns:
        One :class:`V4l2CameraInfo` per camera *device* (a single node group;
        the lowest-numbered ``/dev/videoN`` is reported).  Empty on
        non-Linux hosts and on hosts with no camera attached.
    """
    sink = warnings if warnings is not None else []
    if platform.system() != "Linux":
        sink.append("cameras.v4l2: skipped (non-Linux host).")
        return []

    cameras = _probe_via_sysfs(warnings=sink)
    if cameras:
        return cameras

    cmd = shutil.which("v4l2-ctl")
    if not cmd:
        sink.append(
            "cameras.v4l2: no cameras in /sys/class/video4linux and v4l2-ctl "
            "not on $PATH (apt install v4l-utils)."
        )
        return []
    try:
        raw = subprocess.check_output(
            [cmd, "--list-devices"], text=True, timeout=3, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        sink.append(f"cameras.v4l2: v4l2-ctl --list-devices failed: {exc!r}")
        return []
    return _parse_v4l2_list_devices(raw)


# ── sysfs backend ─────────────────────────────────────────────────────────────


def _read(path: str) -> str:
    """Read one sysfs attribute, returning ``""`` when it is absent."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _usb_ancestor(node: str) -> str:
    """Return the sysfs path of the nearest USB device node above ``node``.

    Args:
        node: An absolute, symlink-resolved sysfs device path.

    Returns:
        The ancestor directory carrying ``idVendor`` / ``idProduct``, or
        ``""`` for a device with no USB ancestor (MIPI-CSI, virtual).
    """
    current = node
    for _ in range(8):
        if _read(f"{current}/idVendor") and _read(f"{current}/idProduct"):
            return current
        parent = os.path.dirname(current)
        if parent == current or parent == "/sys":
            break
        current = parent
    return ""


def _capture_node_index(camera: V4l2CameraInfo) -> tuple[int, str]:
    """Sort key placing ``/dev/video9`` before ``/dev/video10``."""
    m = re.search(r"(\d+)$", camera.device_path)
    return (int(m.group(1)) if m else 0, camera.device_path)


def _probe_via_sysfs(*, warnings: list[str]) -> list[V4l2CameraInfo]:
    """Enumerate cameras from ``/sys/class/video4linux``.

    Nodes are grouped by their parent device so that a UVC camera exposing a
    capture node and a metadata node yields a single row.  Two identical
    cameras (same VID/PID, same product string) stay distinct because their
    parent USB paths differ.

    Args:
        warnings: List to append non-fatal probe issues to.

    Returns:
        One row per physical camera, ordered by capture-node path.
    """
    try:
        nodes = sorted(
            os.listdir(_SYSFS_V4L2),
            # video10 must sort after video9, not between video1 and video2.
            key=lambda n: (int(m.group(1)) if (m := re.search(r"(\d+)$", n)) else 0, n),
        )
    except OSError:
        return []

    grouped: dict[str, V4l2CameraInfo] = {}
    for node in nodes:
        if not node.startswith("video"):
            continue
        sysfs_node = f"{_SYSFS_V4L2}/{node}"
        device_path = f"/dev/{node}"
        name = _read(f"{sysfs_node}/name")
        try:
            parent = os.path.dirname(os.path.realpath(f"{sysfs_node}/device"))
        except OSError:  # pragma: no cover  # reason: realpath does not raise on Linux
            continue
        if parent in grouped:
            # A second node on the same physical device — the first (lowest
            # numbered) node already represents it.
            continue

        usb = _usb_ancestor(os.path.realpath(f"{sysfs_node}/device"))
        vid = pid = 0
        serial = ""
        bus_info = ""
        if usb:
            try:
                vid = int(_read(f"{usb}/idVendor"), 16)
                pid = int(_read(f"{usb}/idProduct"), 16)
            except ValueError:
                vid = pid = 0
            serial = _read(f"{usb}/serial")
            bus_info = f"usb-{os.path.basename(usb)}"

        grouped[parent] = V4l2CameraInfo(
            device_path=device_path,
            name=name or node,
            bus_info=bus_info,
            vid=vid,
            pid=pid,
            serial=serial,
        )

    cameras = sorted(grouped.values(), key=_capture_node_index)
    unidentified = [c.name for c in cameras if not c.vid]
    if unidentified:
        warnings.append(
            f"cameras.v4l2: {unidentified} have no USB descriptor (MIPI-CSI or "
            "virtual device); catalog lookup falls back to the product name."
        )
    return cameras


# ── v4l2-ctl backend ──────────────────────────────────────────────────────────


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
