"""USB VID/PID enumeration and DDS topic discovery for ``openral detect``.

This module provides two independent probes:

1. **USB enumeration** — finds USB serial adapters attached to the host and
   matches their VID/PID against a table of known robot controllers.  On
   Linux it uses ``pyudev`` (no root required); on macOS it falls back to
   ``system_profiler``; everywhere it also falls back to ``/dev/tty*`` globs.

2. **DDS discovery** — runs ``ros2 topic list`` for a bounded timeout and
   maps observed topic name prefixes to known robot types (Unitree, ALOHA, …).

Both probes are pure functions with no global state and can be called from
tests without hardware.

Example:
    >>> from openral_cli.autodetect import enumerate_usb_devices
    >>> devs = enumerate_usb_devices()  # returns [] on machines with no USB adapters
    >>> isinstance(devs, list)
    True
"""

from __future__ import annotations

import json
import platform
import subprocess
from glob import glob
from typing import NamedTuple

__all__ = [
    "DdsTopic",
    "KnownDevice",
    "UsbDevice",
    "UsbMatch",
    "enumerate_usb_devices",
    "infer_robot_from_topics",
    "match_known_devices",
    "scan_dds_topics",
]

# ── Data types ─────────────────────────────────────────────────────────────────


class UsbDevice(NamedTuple):
    """A USB serial device detected on the host.

    Attributes:
        port: Device path, e.g. ``"/dev/ttyUSB0"``.
        vid: Vendor ID as an integer (0 if unknown).
        pid: Product ID as an integer (0 if unknown).
        description: Human-readable product string from the USB descriptor.
    """

    port: str
    vid: int
    pid: int
    description: str


class KnownDevice(NamedTuple):
    """A known USB adapter/controller from the VID/PID table.

    Attributes:
        chip: USB chip name, e.g. ``"CH340"``.
        driver_hint: Human-readable hint about the robot this adapter drives.
        embodiment_tag: openral embodiment tag, e.g. ``"so100_follower"``.
            Empty string means multiple robots are possible with this adapter.
        bh_robot_type: Short robot type string for ``openral connect --robot``.
            Empty string means ambiguous.
    """

    chip: str
    driver_hint: str
    embodiment_tag: str
    bh_robot_type: str


class UsbMatch(NamedTuple):
    """A detected USB device that matched the known-device table.

    Attributes:
        device: The detected USB device.
        known: The matching entry from the VID/PID table.
    """

    device: UsbDevice
    known: KnownDevice


class DdsTopic(NamedTuple):
    """A ROS 2 topic observed during the DDS discovery scan.

    Attributes:
        name: Full topic name, e.g. ``"/lowstate"``.
        type_name: Message type string, e.g. ``"unitree_go/msg/LowState"``.
    """

    name: str
    type_name: str


# ── VID/PID table ─────────────────────────────────────────────────────────────
# Maps (vendor_id, product_id) → KnownDevice.
# Add entries here as new robot controllers are supported.

_VID_PID_TABLE: dict[tuple[int, int], KnownDevice] = {
    # ── CH34x USB-serial chips (WCH Semiconductor) ───────────────────────────
    # Used on cheap USB-to-TTL dongles, many SO-100 debug boards, LeKiwi.
    (0x1A86, 0x7523): KnownDevice(
        "CH340",
        "Feetech serial bus — SO-100 / Koch / LeKiwi arm",
        "so100_follower",
        "so100",
    ),
    (0x1A86, 0x7522): KnownDevice(
        "CH340C/K",
        "Feetech serial bus — SO-100 / Koch / LeKiwi arm",
        "so100_follower",
        "so100",
    ),
    (0x1A86, 0x55D4): KnownDevice(
        "CH9102",
        "Feetech serial bus — SO-100 / Koch / LeKiwi arm",
        "so100_follower",
        "so100",
    ),
    # ── Silicon Labs CP210x ───────────────────────────────────────────────────
    # Used on many microcontroller dev boards and Feetech adapters.
    (0x10C4, 0xEA60): KnownDevice(
        "CP2102",
        "Feetech / Dynamixel serial bus",
        "so100_follower",
        "so100",
    ),
    (0x10C4, 0xEA6A): KnownDevice(
        "CP2104",
        "Feetech / Dynamixel serial bus",
        "so100_follower",
        "so100",
    ),
    (0x10C4, 0xEA70): KnownDevice(
        "CP2105",
        "dual-port Feetech / Dynamixel serial bus",
        "so100_follower",
        "so100",
    ),
    # ── FTDI ─────────────────────────────────────────────────────────────────
    # Used by Dynamixel USB2Dynamixel, OpenCM 9.04, ALOHA leader/follower.
    (0x0403, 0x6001): KnownDevice(
        "FT232RL",
        "Dynamixel USB2Dynamixel — ALOHA / OpenManipulator",
        "aloha",
        "",  # ambiguous until topic scan
    ),
    (0x0403, 0x6010): KnownDevice(
        "FT2232H",
        "Dynamixel / custom dual-port",
        "",
        "",
    ),
    (0x0403, 0x6014): KnownDevice(
        "FT232H",
        "Dynamixel / custom single-port",
        "",
        "",
    ),
    # ── Arduino ──────────────────────────────────────────────────────────────
    (0x2341, 0x0043): KnownDevice(
        "Arduino Uno",
        "custom firmware robot controller",
        "",
        "",
    ),
    (0x2341, 0x0042): KnownDevice(
        "Arduino Mega 2560",
        "custom firmware robot controller",
        "",
        "",
    ),
    # ── STM32 virtual COM port ────────────────────────────────────────────────
    (0x0483, 0x5740): KnownDevice(
        "STM32 VCP",
        "custom STM32 robot controller",
        "",
        "",
    ),
    # ── Unitree robotics ─────────────────────────────────────────────────────
    # Unitree G1/H1/B1 connect over Ethernet (DDS), not USB serial.
    # Their USB port (0x0483:0xDF11) is only used for DFU firmware flashing.
    (0x0483, 0xDF11): KnownDevice(
        "STM32 DFU",
        "Unitree firmware update port (not the control interface — use Ethernet)",
        "unitree_g1",
        "",
    ),
}

# ── DDS topic → robot type map ────────────────────────────────────────────────
# Maps topic name prefix (lower-cased) → ral robot type string.

_TOPIC_ROBOT_MAP: dict[str, str] = {
    "/lowstate": "unitree_g1",  # Unitree G1 / H1 / B1 / Go2
    "/lowcmd": "unitree_g1",
    "/sportmodestate": "unitree_g1",
    "/follower_arms_position_goal": "aloha",  # HiLSeRL ALOHA / ACT
    "/bimanual_stretch": "aloha",
    "/so100": "so100",  # openral SO-100 lifecycle node
    "/joint_trajectory_controller/joint_trajectory": "ros2_control",
    "/lekiwi": "lekiwi",
}


# ── USB enumeration ────────────────────────────────────────────────────────────


def _enumerate_linux_pyudev() -> list[UsbDevice]:
    """Enumerate USB serial devices via pyudev (Linux only).

    Returns:
        List of detected ``UsbDevice`` instances.  Empty if pyudev is not
        installed or no USB serial devices are present.
    """
    try:
        import pyudev  # noqa: PLC0415  # reason: Linux-only optional dep; lazy import keeps macOS clean

        ctx = pyudev.Context()
        devices: list[UsbDevice] = []
        for dev in ctx.list_devices(subsystem="tty"):
            parent = dev.find_parent("usb", "usb_interface")
            if parent is None:
                continue
            port = dev.get("DEVNAME", "")
            if not port:
                continue
            vid_str = parent.get("ID_VENDOR_ID", "0") or "0"
            pid_str = parent.get("ID_MODEL_ID", "0") or "0"
            desc = parent.get("ID_MODEL", "") or parent.get("ID_VENDOR", "") or ""
            try:
                vid = int(vid_str, 16)
                pid = int(pid_str, 16)
            except ValueError:
                vid, pid = 0, 0
            devices.append(UsbDevice(port=port, vid=vid, pid=pid, description=desc))
        return sorted(devices, key=lambda d: d.port)
    except Exception:  # reason: pyudev failure → fall through to glob
        return []


def _enumerate_macos_system_profiler() -> list[UsbDevice]:
    """Enumerate USB devices via ``system_profiler`` on macOS.

    Returns:
        List of detected ``UsbDevice`` instances.  Empty if the command fails
        or no USB serial devices are present.
    """
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPUSBDataType", "-json"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(raw)
    except Exception:  # reason: subprocess or json failure → fall through
        return []

    devices: list[UsbDevice] = []

    def _walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            # A USB device entry has vendor_id + product_id
            vid_str: str = node.get("vendor_id", "") or ""
            pid_str: str = node.get("product_id", "") or ""
            bsd_name: str = node.get("bsd_name", "") or ""
            desc: str = node.get("_name", "") or ""
            if vid_str and pid_str and bsd_name and bsd_name.startswith("cu."):
                port = f"/dev/{bsd_name}"
                try:
                    vid = int(vid_str.replace("0x", ""), 16)
                    pid = int(pid_str.replace("0x", ""), 16)
                except ValueError:
                    vid, pid = 0, 0
                devices.append(UsbDevice(port=port, vid=vid, pid=pid, description=desc))
            for v in node.values():
                _walk(v)

    _walk(data)
    return sorted(devices, key=lambda d: d.port)


def _enumerate_glob_fallback() -> list[UsbDevice]:
    """Enumerate USB serial devices via ``/dev/tty*`` glob patterns.

    This fallback provides port paths but no VID/PID information.

    Returns:
        List of ``UsbDevice`` with vid=0, pid=0.
    """
    sys = platform.system()
    if sys == "Linux":
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*"]
    elif sys == "Darwin":
        patterns = ["/dev/cu.usbserial*", "/dev/cu.usbmodem*"]
    else:
        return []

    ports: list[str] = []
    for pattern in patterns:
        ports.extend(sorted(glob(pattern)))

    return [UsbDevice(port=p, vid=0, pid=0, description="") for p in sorted(set(ports))]


def enumerate_usb_devices() -> list[UsbDevice]:
    """Enumerate USB serial adapters attached to this host.

    Tries platform-specific methods in order:

    - **Linux**: ``pyudev`` (VID/PID + port); falls back to ``/dev/tty*`` glob.
    - **macOS**: ``system_profiler SPUSBDataType -json`` (VID/PID + port);
      falls back to ``/dev/cu.*`` glob.
    - **Other**: returns empty list.

    Returns:
        List of `UsbDevice`, sorted by port path.  May be empty.

    Example:
        >>> devs = enumerate_usb_devices()
        >>> isinstance(devs, list)
        True
    """
    sys = platform.system()
    if sys == "Linux":
        devs = _enumerate_linux_pyudev()
        if not devs:
            devs = _enumerate_glob_fallback()
        return devs
    if sys == "Darwin":
        devs = _enumerate_macos_system_profiler()
        if not devs:
            devs = _enumerate_glob_fallback()
        return devs
    return []


def match_known_devices(devices: list[UsbDevice]) -> list[UsbMatch]:
    """Match detected USB devices against the known VID/PID table.

    Args:
        devices: Output of `enumerate_usb_devices`.

    Returns:
        List of `UsbMatch` for every device whose ``(vid, pid)``
        pair appears in :data:`_VID_PID_TABLE`.  Devices with unknown
        VID/PID (vid=0) are excluded.

    Example:
        >>> matches = match_known_devices([])
        >>> matches
        []
    """
    matches: list[UsbMatch] = []
    for dev in devices:
        known = _VID_PID_TABLE.get((dev.vid, dev.pid))
        if known is not None:
            matches.append(UsbMatch(device=dev, known=known))
    return matches


# ── DDS discovery ─────────────────────────────────────────────────────────────


def scan_dds_topics(timeout_s: float = 5.0) -> list[DdsTopic]:
    """Run ``ros2 topic list -t`` and return observed topics.

    Requires ``ros2`` to be on ``PATH`` and the ROS 2 environment sourced.
    Returns an empty list if ``ros2`` is not available or times out.

    Args:
        timeout_s: Maximum seconds to wait for ``ros2 topic list`` to return.
            Defaults to ``5.0``.

    Returns:
        List of `DdsTopic` sorted by topic name.  Empty if ROS 2 is
        not available or no topics are discovered within the timeout.

    Example:
        >>> topics = scan_dds_topics(timeout_s=0.1)  # too short → empty in most envs
        >>> isinstance(topics, list)
        True
    """
    import shutil  # noqa: PLC0415  # reason: keep top-level imports minimal

    if not shutil.which("ros2"):
        return []

    try:
        raw = subprocess.check_output(
            ["ros2", "topic", "list", "-t"],
            text=True,
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []

    topics: list[DdsTopic] = []
    for raw_line in raw.strip().splitlines():
        # Format: "/topic_name [pkg/msg/Type]"
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        name = parts[0]
        type_name = parts[1].strip("[]") if len(parts) > 1 else ""
        topics.append(DdsTopic(name=name, type_name=type_name))

    return sorted(topics, key=lambda t: t.name)


def infer_robot_from_topics(topics: list[DdsTopic]) -> str | None:
    """Infer a robot type from DDS topic names.

    Checks each topic name against :data:`_TOPIC_ROBOT_MAP` prefix matches.
    Returns the first match found (in topic-name alphabetical order).

    Args:
        topics: Output of `scan_dds_topics`.

    Returns:
        A robot type string (e.g. ``"unitree_g1"``) if a known topic prefix
        is found, otherwise ``None``.

    Example:
        >>> from openral_cli.autodetect import DdsTopic, infer_robot_from_topics
        >>> infer_robot_from_topics([DdsTopic("/lowstate", "unitree_go/msg/LowState")])
        'unitree_g1'
        >>> infer_robot_from_topics([]) is None
        True
    """
    for topic in topics:
        name_lower = topic.name.lower()
        for prefix, robot_type in _TOPIC_ROBOT_MAP.items():
            if name_lower.startswith(prefix.lower()):
                return robot_type
    return None
