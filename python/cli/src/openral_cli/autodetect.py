"""USB, SocketCAN, and DDS transport discovery for ``openral detect``.

This module provides three independent probes:

1. **USB enumeration** — finds USB serial adapters attached to the host and
   matches their VID/PID against a table of known robot controllers.  On
   Linux it uses ``pyudev`` (no root required); on macOS it falls back to
   ``system_profiler``; everywhere it also falls back to ``/dev/tty*`` globs.

2. **SocketCAN enumeration** — lists CAN / CAN FD network interfaces from
   ``/sys/class/net`` and maps deliberately-named ones (``openarm_left``, …)
   to known robot types.  This is the transport that USB-serial enumeration
   structurally cannot see: a CAN adapter registers a *network* device, so a
   CAN-attached arm has no ``/dev/tty*`` node to find.

3. **DDS discovery** — runs ``ros2 topic list`` for a bounded timeout and
   maps observed topic name prefixes to known robot types (Unitree, ALOHA, …).

All three probes are pure functions with no global state, never open a bus or
a device, and can be called from tests without hardware.

Example:
    >>> from openral_cli.autodetect import enumerate_usb_devices
    >>> devs = enumerate_usb_devices()  # returns [] on machines with no USB adapters
    >>> isinstance(devs, list)
    True
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from glob import glob
from typing import NamedTuple

__all__ = [
    "CanInterface",
    "CanMatch",
    "DdsTopic",
    "KnownDevice",
    "UsbDevice",
    "UsbMatch",
    "can_adapter_name",
    "enumerate_can_interfaces",
    "enumerate_usb_devices",
    "infer_robot_from_can",
    "infer_robot_from_topics",
    "match_can_interfaces",
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
        embodiment_tag: openral embodiment tag, e.g. ``"so101_follower"``.
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
    # Used on cheap USB-to-TTL dongles, the SO-100/SO-101 control boards, LeKiwi.
    # The SO-101 is electrically identical to the SO-100 over USB (same Feetech
    # controller / VID/PID), so the bus alone cannot distinguish them. The SO-101
    # is the current revision, so a bare plug-in defaults to `so101_follower`;
    # an SO-100 is selected explicitly with `openral detect --robot so100`.
    (0x1A86, 0x7523): KnownDevice(
        "CH340",
        "Feetech serial bus — SO-100 / SO-101 / Koch / LeKiwi arm",
        "so101_follower",
        "so101",
    ),
    (0x1A86, 0x7522): KnownDevice(
        "CH340C/K",
        "Feetech serial bus — SO-100 / SO-101 / Koch / LeKiwi arm",
        "so101_follower",
        "so101",
    ),
    (0x1A86, 0x55D3): KnownDevice(
        "CH343",
        "Feetech serial bus — SO-100 / SO-101 / Koch / LeKiwi arm",
        "so101_follower",
        "so101",
    ),
    (0x1A86, 0x55D4): KnownDevice(
        "CH9102",
        "Feetech serial bus — SO-100 / SO-101 / Koch / LeKiwi arm",
        "so101_follower",
        "so101",
    ),
    # ── Silicon Labs CP210x ───────────────────────────────────────────────────
    # Used on many microcontroller dev boards and Feetech adapters.
    (0x10C4, 0xEA60): KnownDevice(
        "CP2102",
        "Feetech / Dynamixel serial bus",
        "so101_follower",
        "so101",
    ),
    (0x10C4, 0xEA6A): KnownDevice(
        "CP2104",
        "Feetech / Dynamixel serial bus",
        "so101_follower",
        "so101",
    ),
    (0x10C4, 0xEA70): KnownDevice(
        "CP2105",
        "dual-port Feetech / Dynamixel serial bus",
        "so101_follower",
        "so101",
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
    "/so101": "so101",  # openral SO-101 lifecycle node (current default arm)
    "/so100": "so100",  # explicit openral SO-100 lifecycle node
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
            # VID/PID/model live on the usb_device node, not the usb_interface
            # — CDC-ACM boards (/dev/ttyACM*, e.g. the SO-101's CH343) leave them
            # unset on the interface, so read them from the usb_device parent.
            parent = dev.find_parent("usb", "usb_device")
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


# ── SocketCAN discovery ───────────────────────────────────────────────────────
# The third transport family. USB-serial and DDS both miss CAN-attached arms:
# a Damiao/CAN-FD robot such as the Enactic OpenArm exposes no `/dev/tty*` node
# (the USB adapter registers a *network* device, not a serial one) and publishes
# no DDS topics until its ROS 2 bringup is already running — which is precisely
# what `openral detect` runs *before*.


class CanInterface(NamedTuple):
    """One SocketCAN network interface discovered on the host.

    Attributes:
        name: Interface name, e.g. ``"openarm_left"`` or ``"can0"``.
        is_up: ``True`` when the link is administratively up.
        fd_enabled: ``True`` when the controller runs in CAN FD mode.
        bitrate: Nominal (arbitration-phase) bitrate in bit/s; ``0`` if unset.
        data_bitrate: CAN FD data-phase bitrate in bit/s; ``0`` when not FD.
        state: Controller state as reported by the kernel — ``"ERROR-ACTIVE"``,
            ``"ERROR-PASSIVE"``, ``"BUS-OFF"``, ``"STOPPED"``, or ``""`` when
            ``ip`` is unavailable.  ``"ERROR-PASSIVE"`` on an otherwise healthy
            link is the signature of a bus whose peers are unpowered: frames
            leave the adapter but nothing ACKs them.
        driver: Kernel bittiming-const name, e.g. ``"pcan_usb_pro_fd"``.
        mtu: Link MTU — ``16`` for classic CAN, ``72`` for CAN FD.
        vid: USB vendor ID of the parent adapter (0 for on-SoC controllers).
        pid: USB product ID of the parent adapter (0 for on-SoC controllers).
        description: USB product string of the parent adapter, if any.
    """

    name: str
    is_up: bool
    fd_enabled: bool
    bitrate: int
    data_bitrate: int
    state: str
    driver: str
    mtu: int
    vid: int
    pid: int
    description: str


class CanMatch(NamedTuple):
    """A group of CAN interfaces that together identify one known robot.

    Attributes:
        interfaces: The interfaces that matched, in name order.  A bimanual
            robot contributes one interface per arm.
        known: The matching entry describing what drives that bus.
    """

    interfaces: tuple[CanInterface, ...]
    known: KnownDevice


# ── CAN interface-name → robot type map ───────────────────────────────────────
# Keyed on interface-name *prefix*, lower-cased.
#
# Unlike USB, a CAN bus carries no vendor descriptor: the adapter identifies
# itself (PEAK, Kvaser, …) but the motors on the far side do not announce what
# machine they are bolted into.  The one durable signal is the interface name,
# which on a provisioned robot is pinned by a udev rule rather than left as the
# kernel's `canN` enumeration order.  So this table matches *deliberate* names
# only — a bare `can0` stays unmatched, because `can0` means nothing.
#
# The OpenArm's documented Jetson provisioning names its two PCAN-USB Pro FD
# channels `openarm_left` (motor `dev_id 0x1`) and `openarm_right` (`dev_id
# 0x0`) exactly so that the ROS 2 bringup does not depend on enumeration order.
_CAN_NAME_ROBOT_TABLE: dict[str, KnownDevice] = {
    "openarm": KnownDevice(
        "SocketCAN",
        "Damiao CAN FD motor bus — Enactic OpenArm (one interface per arm)",
        "openarm",
        "openarm",
    ),
}

# ── CAN adapter VID/PID table ─────────────────────────────────────────────────
# Purely descriptive: names the dongle in the report so an operator can tell a
# missing adapter from a missing robot.  Never used to infer a robot type — the
# same adapter drives any CAN machine.  Entries are added only when the
# VID/PID has been read off real hardware.
_CAN_ADAPTER_TABLE: dict[tuple[int, int], str] = {
    (0x0C72, 0x0011): "PEAK PCAN-USB Pro FD",
}

# Vendor-level fallback for an unlisted product ID from a known CAN vendor.
_CAN_VENDOR_TABLE: dict[int, str] = {
    0x0C72: "PEAK-System",
}

# ARPHRD_CAN — the `/sys/class/net/<if>/type` value that marks a CAN link.
# From the kernel's `include/uapi/linux/if_arp.h`.
_ARPHRD_CAN = 280

# Sysfs root for network devices.  A module constant rather than a literal so
# tests can point the enumeration at a real fixture tree on disk.
_SYSFS_NET = "/sys/class/net"


def _read_sysfs(path: str) -> str:
    """Read one sysfs attribute, returning ``""`` when it is absent."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _usb_parent_ids(ifname: str) -> tuple[int, int, str]:
    """Walk up from a net device to its USB parent and read its descriptor.

    Args:
        ifname: Network interface name.

    Returns:
        ``(vid, pid, product)``.  All-zero / empty when the interface has no
        USB ancestor (an on-SoC CAN controller, a vcan, …).
    """
    try:
        node = os.path.realpath(f"{_SYSFS_NET}/{ifname}/device")
    except OSError:  # pragma: no cover  # reason: realpath on Linux does not raise here
        return (0, 0, "")
    # A USB-attached CAN adapter sits a handful of levels below the USB device
    # node that carries the descriptor; bound the walk so a malformed tree
    # cannot spin.
    for _ in range(6):
        vid_raw = _read_sysfs(f"{node}/idVendor")
        pid_raw = _read_sysfs(f"{node}/idProduct")
        if vid_raw and pid_raw:
            try:
                return (int(vid_raw, 16), int(pid_raw, 16), _read_sysfs(f"{node}/product"))
            except ValueError:
                return (0, 0, "")
        parent = os.path.dirname(node)
        if parent == node or parent == "/sys":
            break
        node = parent
    return (0, 0, "")


def _ip_link_details() -> dict[str, dict[str, object]]:
    """Snapshot ``ip -details -json link show``, keyed by interface name.

    Returns:
        Mapping of interface name to its raw ``ip`` record.  Empty when
        ``ip`` is not installed or returns something unparseable — callers
        degrade to the sysfs-only view rather than failing.
    """
    import shutil  # noqa: PLC0415  # reason: keep top-level imports minimal

    exe = shutil.which("ip")
    if not exe:
        return {}
    try:
        raw = subprocess.check_output(
            [exe, "-details", "-json", "link", "show"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        parsed = json.loads(raw)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        json.JSONDecodeError,
    ):
        return {}
    if not isinstance(parsed, list):
        return {}
    out: dict[str, dict[str, object]] = {}
    for record in parsed:
        if isinstance(record, dict):
            name = record.get("ifname")
            if isinstance(name, str):
                out[name] = record
    return out


def _can_info_from_ip(record: dict[str, object]) -> tuple[bool, int, int, str, str]:
    """Extract the CAN-specific fields from one ``ip -details -json`` record.

    Returns:
        ``(fd_enabled, bitrate, data_bitrate, state, driver)`` — every field
        defaulted when the record does not carry a ``linkinfo.info_data``
        block (a non-CAN link, or an ``ip`` too old to emit one).
    """
    linkinfo = record.get("linkinfo")
    if not isinstance(linkinfo, dict):
        return (False, 0, 0, "", "")
    info = linkinfo.get("info_data")
    if not isinstance(info, dict):
        return (False, 0, 0, "", "")

    ctrlmode = info.get("ctrlmode")
    fd_enabled = isinstance(ctrlmode, list) and "FD" in ctrlmode

    def _bitrate(key: str) -> int:
        block = info.get(key)
        if isinstance(block, dict):
            value = block.get("bitrate")
            if isinstance(value, int):
                return value
        return 0

    state = info.get("state")
    const = info.get("bittiming_const")
    driver = ""
    if isinstance(const, dict):
        name = const.get("name")
        if isinstance(name, str):
            driver = name

    return (
        fd_enabled,
        _bitrate("bittiming"),
        _bitrate("data_bittiming"),
        state if isinstance(state, str) else "",
        driver,
    )


def enumerate_can_interfaces() -> list[CanInterface]:
    """Enumerate every SocketCAN interface on the host.

    Linux-only and dependency-free: the interface list comes from
    ``/sys/class/net`` (an interface whose ``type`` is ``280`` /
    ``ARPHRD_CAN``), and ``ip -details -json link show`` — when present —
    enriches each row with bitrates, FD mode, and controller state.  Neither
    step needs root, and neither opens the bus, so calling this never
    perturbs a robot that is already running.

    Returns:
        Interfaces sorted by name.  Empty on non-Linux hosts and on Linux
        hosts with no CAN controller.

    Example:
        >>> from openral_cli.autodetect import enumerate_can_interfaces
        >>> ifaces = enumerate_can_interfaces()  # [] on hosts with no CAN bus
        >>> isinstance(ifaces, list)
        True
    """
    if platform.system() != "Linux":
        return []
    try:
        names = sorted(os.listdir(_SYSFS_NET))
    except OSError:
        return []

    details = _ip_link_details()
    out: list[CanInterface] = []
    for name in names:
        if _read_sysfs(f"{_SYSFS_NET}/{name}/type") != str(_ARPHRD_CAN):
            continue
        mtu_raw = _read_sysfs(f"{_SYSFS_NET}/{name}/mtu")
        try:
            mtu = int(mtu_raw)
        except ValueError:
            mtu = 0
        record = details.get(name, {})
        fd_enabled, bitrate, data_bitrate, state, driver = _can_info_from_ip(record)
        # Without `ip`, MTU still separates classic CAN (16) from CAN FD (72).
        if not record:
            fd_enabled = mtu > 16
        vid, pid, product = _usb_parent_ids(name)
        out.append(
            CanInterface(
                name=name,
                is_up=_read_sysfs(f"{_SYSFS_NET}/{name}/operstate").lower() == "up",
                fd_enabled=fd_enabled,
                bitrate=bitrate,
                data_bitrate=data_bitrate,
                state=state,
                driver=driver,
                mtu=mtu,
                vid=vid,
                pid=pid,
                description=product,
            )
        )
    return out


def can_adapter_name(iface: CanInterface) -> str:
    """Human-readable name for the USB adapter behind a CAN interface.

    Args:
        iface: One row from :func:`enumerate_can_interfaces`.

    Returns:
        The catalogued adapter name (``"PEAK PCAN-USB Pro FD"``), else the
        USB product string, else a vendor-only label, else ``"SocketCAN"``
        for an on-SoC controller with no USB ancestor.

    Example:
        >>> from openral_cli.autodetect import CanInterface, can_adapter_name
        >>> iface = CanInterface(
        ...     "openarm_left", True, True, 1000000, 5000000,
        ...     "ERROR-ACTIVE", "pcan_usb_pro_fd", 72, 0x0C72, 0x0011, "PCAN-USB Pro FD",
        ... )
        >>> can_adapter_name(iface)
        'PEAK PCAN-USB Pro FD'
    """
    catalogued = _CAN_ADAPTER_TABLE.get((iface.vid, iface.pid))
    if catalogued is not None:
        return catalogued
    if iface.description:
        return iface.description
    vendor = _CAN_VENDOR_TABLE.get(iface.vid)
    if vendor is not None:
        return f"{vendor} (unlisted 0x{iface.pid:04x})"
    return "SocketCAN"


def match_can_interfaces(interfaces: list[CanInterface]) -> list[CanMatch]:
    """Group CAN interfaces by the robot their names declare.

    Only interfaces that are **up** are considered: a down link proves the
    kernel knows about a controller, not that a robot is wired to it.

    Args:
        interfaces: Output of :func:`enumerate_can_interfaces`.

    Returns:
        One :class:`CanMatch` per matched robot type, in robot-type order,
        each carrying every interface that matched it.  Empty when no
        interface name matches :data:`_CAN_NAME_ROBOT_TABLE`.

    Example:
        >>> from openral_cli.autodetect import match_can_interfaces
        >>> match_can_interfaces([])
        []
    """
    grouped: dict[str, list[CanInterface]] = {}
    for iface in interfaces:
        if not iface.is_up:
            continue
        lowered = iface.name.lower()
        for prefix in _CAN_NAME_ROBOT_TABLE:
            if lowered.startswith(prefix):
                grouped.setdefault(prefix, []).append(iface)
                break

    matches: list[CanMatch] = []
    for prefix in sorted(grouped):
        known = _CAN_NAME_ROBOT_TABLE[prefix]
        members = tuple(sorted(grouped[prefix], key=lambda i: i.name))
        # Name the actual adapter rather than the generic "SocketCAN" default.
        matches.append(
            CanMatch(
                interfaces=members,
                known=known._replace(chip=can_adapter_name(members[0])),
            )
        )
    return matches


def infer_robot_from_can(interfaces: list[CanInterface]) -> str | None:
    """Infer a robot type from SocketCAN interface names.

    Args:
        interfaces: Output of :func:`enumerate_can_interfaces`.

    Returns:
        A robot type string (e.g. ``"openarm"``) when an up interface's name
        matches a known prefix, otherwise ``None``.

    Example:
        >>> from openral_cli.autodetect import CanInterface, infer_robot_from_can
        >>> iface = CanInterface(
        ...     "openarm_left", True, True, 1000000, 5000000,
        ...     "ERROR-ACTIVE", "pcan_usb_pro_fd", 72, 0x0C72, 0x0011, "PCAN-USB Pro FD",
        ... )
        >>> infer_robot_from_can([iface])
        'openarm'
        >>> infer_robot_from_can([]) is None
        True
    """
    matches = match_can_interfaces(interfaces)
    if not matches:
        return None
    return matches[0].known.bh_robot_type or None
