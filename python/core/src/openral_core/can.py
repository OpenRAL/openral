"""SocketCAN transport discovery — robot-agnostic.

A CAN-bus robot is structurally invisible to USB and serial discovery: the
adapter registers a *network* device, not a ``/dev/tty*``, so nothing in the
serial or USB probe path can see it.  This module is the one place that reads
the kernel's view of those links, so every layer that needs it — hardware
detection, a HAL's connect-time preflight, an operator report — agrees on what
it saw.

Nothing here knows about any particular robot.  The mapping from an interface
name to *which machine is on the far side of the bus* is a matching concern and
lives with the other device tables in ``openral_cli.autodetect``; what lives
here is the mechanism any CAN robot needs.

Linux-only and dependency-free.  Nothing here opens a bus or needs root, so it
is safe to call against a robot that is already running.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Mapping
from typing import NamedTuple

from openral_core.exceptions import ROSConfigError

__all__ = [
    "SYSFS_NET",
    "CanInterface",
    "can_link_state",
    "enumerate_can_interfaces",
    "preflight_can_links",
]

# ARPHRD_CAN — the ``/sys/class/net/<if>/type`` value that marks a CAN link.
# From the kernel's ``include/uapi/linux/if_arp.h``.
_ARPHRD_CAN = 280

# Classic CAN frames cap at 8 data bytes (16-byte MTU); CAN FD's 64-byte
# payload needs 72.  MTU is therefore the FD indicator on hosts without `ip`.
_CLASSIC_CAN_MTU = 16

# Sysfs root for network devices.  A module constant rather than a literal so
# callers and tests can point the enumeration at a fixture tree on disk.
SYSFS_NET = "/sys/class/net"


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


def _read_sysfs(path: str) -> str:
    """Read one sysfs attribute, returning ``""`` when it is absent."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _usb_parent_ids(ifname: str, sysfs_net: str) -> tuple[int, int, str]:
    """Walk up from a net device to its USB parent and read its descriptor.

    Args:
        ifname: Network interface name.
        sysfs_net: Sysfs root for network devices.

    Returns:
        ``(vid, pid, product)``.  All-zero / empty when the interface has no
        USB ancestor (an on-SoC CAN controller, a vcan, …).
    """
    try:
        node = os.path.realpath(f"{sysfs_net}/{ifname}/device")
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
        if parent in (node, "/sys"):
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
        defaulted when the record carries no ``linkinfo``/``info_data``, which
        is what a non-CAN or `ip`-less host looks like.
    """
    linkinfo = record.get("linkinfo")
    info: dict[str, object] = {}
    if isinstance(linkinfo, dict):
        data = linkinfo.get("info_data")
        if isinstance(data, dict):
            info = data

    def _bitrate(key: str) -> int:
        block = info.get(key)
        if isinstance(block, dict):
            value = block.get("bitrate")
            if isinstance(value, int):
                return value
        return 0

    ctrlmode = info.get("ctrlmode")
    fd_enabled = isinstance(ctrlmode, list) and "FD" in ctrlmode
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


def enumerate_can_interfaces(*, sysfs_net: str | None = None) -> list[CanInterface]:
    """Enumerate every SocketCAN interface on the host.

    Linux-only and dependency-free: the interface list comes from
    ``/sys/class/net`` (an interface whose ``type`` is ``280`` /
    ``ARPHRD_CAN``), and ``ip -details -json link show`` — when present —
    enriches each row with bitrates, FD mode, and controller state.  Neither
    step needs root, and neither opens the bus, so calling this never
    perturbs a robot that is already running.

    Args:
        sysfs_net: Sysfs root for network devices.  Defaults to
            :data:`SYSFS_NET`; override it to read a recorded fixture tree.

    Returns:
        Interfaces sorted by name.  Empty on non-Linux hosts and on Linux
        hosts with no CAN controller.

    Example:
        >>> from openral_core.can import enumerate_can_interfaces
        >>> ifaces = enumerate_can_interfaces()  # [] on hosts with no CAN bus
        >>> isinstance(ifaces, list)
        True
    """
    if platform.system() != "Linux":
        return []
    root = SYSFS_NET if sysfs_net is None else sysfs_net
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []

    details = _ip_link_details()
    out: list[CanInterface] = []
    for name in names:
        if _read_sysfs(f"{root}/{name}/type") != str(_ARPHRD_CAN):
            continue
        mtu_raw = _read_sysfs(f"{root}/{name}/mtu")
        try:
            mtu = int(mtu_raw)
        except ValueError:
            mtu = 0
        record = details.get(name, {})
        fd_enabled, bitrate, data_bitrate, state, driver = _can_info_from_ip(record)
        # Without `ip`, MTU still separates classic CAN (16) from CAN FD (72).
        if not record:
            fd_enabled = mtu > _CLASSIC_CAN_MTU
        vid, pid, product = _usb_parent_ids(name, root)
        out.append(
            CanInterface(
                name=name,
                is_up=_read_sysfs(f"{root}/{name}/operstate").lower() == "up",
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


def can_link_state(interface: str, *, sysfs_net: str | None = None) -> tuple[bool, str]:
    """Report whether ``interface`` is an existing, up CAN link.

    The single-interface counterpart to :func:`enumerate_can_interfaces`, for
    the case where a caller already knows the name it needs and wants to say
    precisely what is wrong when it is unusable.

    Args:
        interface: SocketCAN interface name, e.g. ``"openarm_left"``.
        sysfs_net: Sysfs root for network devices.  Defaults to
            :data:`SYSFS_NET`.

    Returns:
        ``(is_up, reason)``.  ``reason`` is empty when the link is usable and
        otherwise names the specific problem, so a caller can put it in an
        exception rather than making the operator go look.
    """
    root = SYSFS_NET if sysfs_net is None else sysfs_net
    arphrd = _read_sysfs(f"{root}/{interface}/type")
    if not arphrd:
        return (False, f"no network interface named {interface!r}")
    if arphrd != str(_ARPHRD_CAN):
        return (False, f"{interface!r} exists but is not a CAN link (type={arphrd})")
    operstate = _read_sysfs(f"{root}/{interface}/operstate").lower()
    if operstate != "up":
        return (False, f"{interface!r} is a CAN link but is {operstate or 'in an unknown state'}")
    return (True, "")


def preflight_can_links(
    interfaces: Mapping[str, str],
    *,
    hal_label: str,
    remedy: str = "",
    sysfs_net: str | None = None,
) -> dict[str, str]:
    """Refuse to connect over a motor bus that is not up.

    Any CAN robot needs this check at ``connect()`` time, whatever its bus
    count: a single-bus arm passes one entry, a bimanual station passes two, a
    multi-limb machine passes as many as it has.  The robot-specific part —
    what the operator should *do* about it — is passed in as ``remedy`` rather
    than written into this function.

    Args:
        interfaces: Bus label to SocketCAN interface name, e.g.
            ``{"left": "openarm_left", "right": "openarm_right"}``.  The label
            is what appears in the returned health map.
        hal_label: Name of the calling HAL, used to open the error message.
        remedy: Robot-specific operator guidance appended to the error.
        sysfs_net: Sysfs root for network devices.  Defaults to
            :data:`SYSFS_NET`.

    Returns:
        A health map of ``{f"{label}_can": "<name>" or "<name> (DOWN)"}``,
        suitable for a HAL health report.

    Raises:
        ROSConfigError: If any interface is missing, is not a CAN link, or is
            down.  The message names every interface that failed and why, so
            one connect attempt reports every problem rather than the first.
    """
    problems: list[str] = []
    health: dict[str, str] = {}
    for label, interface in interfaces.items():
        is_up, reason = can_link_state(interface, sysfs_net=sysfs_net)
        health[f"{label}_can"] = interface if is_up else f"{interface} (DOWN)"
        if not is_up:
            problems.append(reason)
    if problems:
        message = f"{hal_label} cannot connect: {'; '.join(problems)}."
        if remedy:
            message = f"{message} {remedy}"
        raise ROSConfigError(message)
    return health
