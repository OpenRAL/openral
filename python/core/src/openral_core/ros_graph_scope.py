# SPDX-License-Identifier: Apache-2.0
"""Whether a process's ROS 2 graph reaches beyond this host.

One pure decision shared by the two places that start a real robot:
``openral deploy run`` (``openral_cli._dds_scope.confine_real_scope``, which
defaults a host-confined scope first) and ``deploy_e2e.launch.py`` under
``hal_mode:=real`` (which runs on whatever environment ``ros2 launch`` was
started with, so a hand-run launch is checked too). A graph that reaches other
hosts lets any of them publish ``/openral/safe_action`` (the HAL executes it),
call ``/openral/estop_reset``, or spoof ``/joint_states`` and the deadman
heartbeat. See ``docs/reference/ros-graph-security.md``.

Lives in ``openral_core`` because the launch file cannot import ``openral_cli``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from openral_core.exceptions import ROSConfigError

__all__ = ["ALLOW_NETWORK_ROS_ENV", "networked_ros_scope", "require_host_confined_ros_scope"]

#: Opt-in for a real robot's ROS graph to reach beyond this host. Only the exact
#: value ``"1"`` counts.
ALLOW_NETWORK_ROS_ENV: Final[str] = "OPENRAL_ALLOW_NETWORK_ROS"

#: Discovery ranges that keep a graph on this host.
_HOST_CONFINED_RANGES: Final[frozenset[str]] = frozenset({"LOCALHOST", "OFF"})
#: Zenoh ignores both discovery variables and its router listens on every interface.
_ZENOH_RMW: Final[str] = "rmw_zenoh_cpp"
#: Vendor XML that adds remote peers on top of the ROS_* variables (lower-cased
#: markers): Cyclone's ``<Peers>`` / ``<Peer>``, Fast DDS's ``initialPeersList``.
_VENDOR_PEER_MARKERS: Final[dict[str, str]] = {
    "CYCLONEDDS_URI": "<peer",
    "FASTRTPS_DEFAULT_PROFILES_FILE": "initialpeerslist",
    "FASTDDS_DEFAULT_PROFILES_FILE": "initialpeerslist",
}


def _vendor_peers(env: Mapping[str, str]) -> str | None:
    """Describe a DDS vendor config in ``env`` that lists peers (or can't be read), else ``None``.

    ``CYCLONEDDS_URI`` is inline XML or a comma-separated list of (``file://``)
    paths; the Fast DDS variables name one profiles file. An unreadable file is
    reported too: what it adds is unknown, so it is not called host-confined.
    """
    for var, marker in _VENDOR_PEER_MARKERS.items():
        value = env.get(var, "").strip()
        if not value:
            continue
        inline = value.startswith("<")
        for source in [value] if inline else value.split(","):
            try:
                text = (
                    source if inline else Path(source.strip().removeprefix("file://")).read_text()
                )
            except OSError:
                return f"{var}={value} (unreadable, so the peers it adds are unknown)"
            if marker in text.lower():
                return f"{var}={value} (its XML lists remote peers)"
    return None


def networked_ros_scope(env: Mapping[str, str]) -> str | None:
    """Describe the scope when ``env`` lets the graph reach other hosts, else ``None``.

    Reads ``env`` as rcl (Jazzy) does: an unset or empty
    ``ROS_AUTOMATIC_DISCOVERY_RANGE`` is ``SUBNET``; any ``ROS_STATIC_PEERS``
    reaches out; ``ROS_LOCALHOST_ONLY=1`` overrides both ("'automatic_discovery_range'
    and 'static_peers' will be ignored"). Whatever those say, a Fast DDS
    ``ROS_DISCOVERY_SERVER``, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` (which reads
    neither variable; its router listens on every interface), or a vendor XML
    config that lists peers (``CYCLONEDDS_URI`` ``<Peers>``, a Fast DDS profiles
    file's ``initialPeersList``; an unreadable one counts too) reaches other hosts.

    Example:
        >>> networked_ros_scope({"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}) is None
        True
        >>> networked_ros_scope({}) is None
        False
    """
    if env.get("RMW_IMPLEMENTATION", "").strip() == _ZENOH_RMW:
        return f"RMW_IMPLEMENTATION={_ZENOH_RMW} (no discovery-range confinement)"
    server = env.get("ROS_DISCOVERY_SERVER", "").strip()
    if server:
        return f"ROS_DISCOVERY_SERVER={server}"
    vendor = _vendor_peers(env)
    if vendor is not None:
        return vendor
    if env.get("ROS_LOCALHOST_ONLY", "").strip() == "1":
        return None
    discovery = env.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").strip().upper()
    peers = env.get("ROS_STATIC_PEERS", "").strip()
    if not peers and discovery in _HOST_CONFINED_RANGES:
        return None
    return (
        f"ROS_AUTOMATIC_DISCOVERY_RANGE={discovery or '(unset = SUBNET)'} "
        f"ROS_STATIC_PEERS={peers or '(unset)'}"
    )


def require_host_confined_ros_scope(env: Mapping[str, str]) -> str | None:
    """Refuse a networked ROS scope for a real robot unless ``env`` opts in.

    Args:
        env: The environment the robot's graph runs under. The opt-in
            (``OPENRAL_ALLOW_NETWORK_ROS=1``) is read from it too.

    Returns:
        ``None`` when the scope is host-confined; the scope description when it
        is networked and opted in, so the caller logs the hazard.

    Raises:
        ROSConfigError: A networked scope without the opt-in.

    Example:
        >>> require_host_confined_ros_scope({"ROS_LOCALHOST_ONLY": "1"}) is None
        True
    """
    if (
        env.get("ROS_LOCALHOST_ONLY", "").strip() != "1"
        and env.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").strip().upper() == "OFF"
    ):
        # Jazzy's OFF hides nodes even from each other on this host: the robot's
        # lifecycle nodes would never find one another.
        raise ROSConfigError(
            "ROS_AUTOMATIC_DISCOVERY_RANGE=OFF stops a real robot's own nodes from "
            "discovering each other; use LOCALHOST to keep the graph on this host."
        )
    scope = networked_ros_scope(env)
    if scope is None or env.get(ALLOW_NETWORK_ROS_ENV) == "1":
        return scope
    raise ROSConfigError(
        f"refusing to start a real robot on a ROS graph that reaches other hosts "
        f"({scope}). Any host that can discover this graph can publish "
        f"/openral/safe_action (the HAL executes it), call /openral/estop_reset, "
        f"and spoof /joint_states or the deadman heartbeat. Keep the graph on this "
        f"host with ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST and no ROS_STATIC_PEERS "
        f"(`openral deploy run` sets this by default), or, when the graph must span "
        f"machines on a trusted, isolated network, set {ALLOW_NETWORK_ROS_ENV}=1 and "
        f"export the same ROS_DOMAIN_ID on every host.",
    )
