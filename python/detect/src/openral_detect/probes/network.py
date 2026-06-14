"""Network interface probe via ``psutil``.

Used by the assembler to populate diagnostic metadata in the
``RobotDescription.onboard_compute`` blob and to surface a hostname /
default-route summary in ``openral detect``'s output.
"""

from __future__ import annotations

import socket

from openral_detect.report import NetworkInterfaceInfo, NetworkProbeResult

__all__ = ["probe_network"]


def probe_network(*, warnings: list[str] | None = None) -> NetworkProbeResult:
    """Capture host name + per-interface MAC / IPv4 / MTU / link state."""
    sink = warnings if warnings is not None else []
    hostname = socket.gethostname()
    try:
        import psutil  # noqa: PLC0415  # reason: optional path with stdlib fallback
    except ImportError:
        sink.append("network: psutil not installed — only hostname returned.")
        return NetworkProbeResult(hostname=hostname)

    try:
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception as exc:
        sink.append(f"network: psutil enumeration failed: {exc!r}")
        return NetworkProbeResult(hostname=hostname)

    interfaces: list[NetworkInterfaceInfo] = []
    for name, addr_list in addrs.items():
        mac = ""
        ipv4: list[str] = []
        for a in addr_list:
            family_name = getattr(a.family, "name", "") or str(a.family)
            if family_name in ("AF_PACKET", "AF_LINK") or "AF_LINK" in family_name:
                mac = a.address or ""
            elif family_name == "AF_INET":
                ipv4.append(a.address)
        st = stats.get(name)
        interfaces.append(
            NetworkInterfaceInfo(
                name=name,
                mac=mac,
                ipv4=ipv4,
                mtu=int(getattr(st, "mtu", 0) or 0),
                link_speed_mbps=int(getattr(st, "speed", 0) or 0) or None,
                is_up=bool(getattr(st, "isup", False)),
            )
        )

    default_route = _default_route()
    return NetworkProbeResult(
        hostname=hostname,
        interfaces=interfaces,
        default_route=default_route,
    )


def _default_route() -> str | None:
    """Best-effort default-route lookup that does not transmit a packet.

    Uses a UDP socket connect trick: ``getsockname`` reveals which local
    address the kernel would source from, even with a fake destination.
    Returns ``None`` on failure.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.1)
            s.connect(("10.255.255.255", 1))
            return str(s.getsockname()[0])
    except OSError:
        return None
