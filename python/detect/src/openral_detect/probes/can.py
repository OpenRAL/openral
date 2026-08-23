"""SocketCAN enumeration — thin wrapper around ``openral_cli.autodetect``.

Reuses the canonical ``enumerate_can_interfaces`` + ``match_can_interfaces``
pipeline (and the ``_CAN_NAME_ROBOT_TABLE``) so the auto-detect package never
duplicates CAN-probing logic.

This probe is read-only in the strongest sense: it reads ``/sys/class/net``
and shells out to ``ip link show``.  It never opens a CAN socket, so running
``openral detect`` against a live robot cannot perturb it.
"""

from __future__ import annotations

from openral_cli.autodetect import (
    CanInterface,
    can_adapter_name,
    enumerate_can_interfaces,
    match_can_interfaces,
)

from openral_detect.report import (
    CanInterfaceInfo,
    CanMatchRecord,
    CanProbeResult,
)

__all__ = ["diagnose_can_matches", "probe_can"]


def _to_record(iface: CanInterface) -> CanInterfaceInfo:
    """Convert one ``autodetect`` NamedTuple into its Pydantic report row."""
    return CanInterfaceInfo(
        name=iface.name,
        is_up=iface.is_up,
        fd_enabled=iface.fd_enabled,
        bitrate=iface.bitrate,
        data_bitrate=iface.data_bitrate,
        state=iface.state,
        driver=iface.driver,
        mtu=iface.mtu,
        vid=iface.vid,
        pid=iface.pid,
        adapter=can_adapter_name(iface),
    )


def diagnose_can_matches(matches: list[CanMatchRecord]) -> list[str]:
    """Turn controller states on matched buses into operator-facing warnings.

    A matched bus that is transmitting into silence is the single most common
    "detected but will not move" failure, and reading it off the raw ``state``
    field in the JSON report requires knowing what ``ERROR-PASSIVE`` means.
    This names the condition instead.

    Args:
        matches: The matched rows from a :class:`CanProbeResult`.

    Returns:
        Zero or more warning lines, in match order.

    Example:
        >>> from openral_detect.probes.can import diagnose_can_matches
        >>> diagnose_can_matches([])
        []
    """
    out: list[str] = []
    for match in matches:
        passive = [i.name for i in match.interfaces if i.state == "ERROR-PASSIVE"]
        if passive:
            out.append(
                f"can: {match.bh_robot_type} interfaces {passive} are up but in "
                "ERROR-PASSIVE — frames are leaving the adapter and nothing is "
                "ACKing them. The motors are almost certainly unpowered."
            )
        off = [i.name for i in match.interfaces if i.state == "BUS-OFF"]
        if off:
            out.append(
                f"can: {match.bh_robot_type} interfaces {off} are BUS-OFF — the "
                "controller took itself off the bus after excessive errors. "
                "Check termination and bitrate before re-running."
            )
    return out


def probe_can(*, warnings: list[str] | None = None) -> CanProbeResult:
    """Enumerate SocketCAN interfaces and match them against the name table.

    Args:
        warnings: Optional list to append non-fatal probe issues to.  When
            ``None``, warnings are silently dropped (the umbrella probe
            owns aggregation).

    Returns:
        A populated :class:`CanProbeResult`.  Empty on non-Linux hosts and
        on Linux hosts with no CAN controller.

    Example:
        >>> from openral_detect.probes import probe_can
        >>> result = probe_can()  # empty on hosts with no CAN bus
        >>> isinstance(result.interfaces, list)
        True
    """
    try:
        interfaces = enumerate_can_interfaces()
    except Exception as exc:  # reason: probe contract — never raise
        if warnings is not None:
            warnings.append(f"can: enumerate_can_interfaces failed: {exc!r}")
        return CanProbeResult()

    records = [_to_record(i) for i in interfaces]
    matches = [
        CanMatchRecord(
            interfaces=[_to_record(i) for i in m.interfaces],
            chip=m.known.chip,
            driver_hint=m.known.driver_hint,
            embodiment_tag=m.known.embodiment_tag,
            bh_robot_type=m.known.bh_robot_type,
        )
        for m in match_can_interfaces(interfaces)
    ]

    if warnings is not None:
        if not records:
            warnings.append("can: no SocketCAN interfaces found.")
        elif not matches:
            up = [r.name for r in records if r.is_up]
            warnings.append(
                "can: no interface name matched a known robot "
                f"(up: {up or 'none'}). Name the links after the robot "
                "(e.g. 'openarm_left') via a udev rule so detection is "
                "independent of kernel enumeration order."
            )
        warnings.extend(diagnose_can_matches(matches))

    return CanProbeResult(interfaces=records, matches=matches)
