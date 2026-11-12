# SPDX-License-Identifier: Apache-2.0
"""Whether a process's ROS 2 graph reaches beyond this host.

One pure decision shared by every place that starts a real robot:
``openral deploy run`` (``openral_cli._dds_scope.confine_real_scope``, which
defaults a host-confined scope first), ``deploy_e2e.launch.py`` under
``hal_mode:=real``, and every vendor ``real_bringup`` launch
(``require_host_confined_real_bringup``). The launches run on whatever
environment ``ros2 launch`` was started with, so a hand-run launch is checked
too. A graph that reaches other hosts lets any of them publish
``/openral/safe_action`` (the HAL executes it), call ``/openral/estop_reset``,
or spoof ``/joint_states`` and the deadman heartbeat. See ``docs/reference/ros-graph-security.md``.

Lives in ``openral_core`` because the launch file cannot import ``openral_cli``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import structlog

from openral_core.exceptions import ROSConfigError
from openral_core.ros_security import require_secure_ros_graph, ros_security_enabled

_log = structlog.get_logger(__name__)

__all__ = [
    "ALLOW_NETWORK_ROS_ENV",
    "discovery_range_is_ros_default",
    "networked_ros_scope",
    "require_host_confined_real_bringup",
    "require_host_confined_ros_scope",
]

#: Opt-in for a real robot's ROS graph to reach beyond this host. Only the exact
#: value ``"1"`` counts.
ALLOW_NETWORK_ROS_ENV: Final[str] = "OPENRAL_ALLOW_NETWORK_ROS"

#: rcl's default range. ``source /opt/ros/jazzy/setup.bash`` also *exports* it
#: whenever it is unset (``ros_environment``'s ``1.ros_discovery.dsv``:
#: ``set-if-unset;ROS_AUTOMATIC_DISCOVERY_RANGE;SUBNET``), so in any sourced shell
#: an exported ``SUBNET`` is indistinguishable from "nobody chose anything".
_ROS_DEFAULT_RANGE: Final[str] = "SUBNET"
#: Discovery ranges that keep a graph on this host.
_HOST_CONFINED_RANGES: Final[frozenset[str]] = frozenset({"LOCALHOST", "OFF"})
#: Zenoh ignores both discovery variables and its router listens on every interface.
_ZENOH_RMW: Final[str] = "rmw_zenoh_cpp"
#: Vendor XML that adds remote peers on top of the ROS_* variables (lower-cased
#: markers): Cyclone's ``<Peers>`` / ``<Peer>``, Fast DDS's ``initialPeersList``
#: or ``discoveryServersList`` (a discovery server, as ``ROS_DISCOVERY_SERVER``).
_FASTDDS_PEER_MARKERS: Final[tuple[str, ...]] = ("initialpeerslist", "discoveryserverslist")
_VENDOR_PEER_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "CYCLONEDDS_URI": ("<peer",),
    "FASTRTPS_DEFAULT_PROFILES_FILE": _FASTDDS_PEER_MARKERS,
    "FASTDDS_DEFAULT_PROFILES_FILE": _FASTDDS_PEER_MARKERS,
}


def _vendor_peers(env: Mapping[str, str]) -> str | None:
    """Describe a DDS vendor config in ``env`` that lists peers (or can't be read), else ``None``.

    ``CYCLONEDDS_URI`` is inline XML or a comma-separated list of (``file://``)
    paths; the Fast DDS variables name one profiles file. An unreadable file is
    reported too: what it adds is unknown, so it is not called host-confined.
    """
    for var, markers in _VENDOR_PEER_MARKERS.items():
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
            if any(m in text.lower() for m in markers):
                return f"{var}={value} (its XML lists remote peers)"
    return None


def discovery_range_is_ros_default(env: Mapping[str, str]) -> bool:
    """Whether ``ROS_AUTOMATIC_DISCOVERY_RANGE`` in ``env`` is ROS's default, not a choice.

    Unset or empty is the default to rcl. ``SUBNET`` is too unless
    ``OPENRAL_ALLOW_NETWORK_ROS=1`` is set: ``setup.bash`` exports it into every
    sourced shell, so the variable alone cannot say an operator chose the LAN —
    the OpenRAL opt-in can. Every other value (``LOCALHOST``, ``OFF``,
    ``SYSTEM_DEFAULT``) is something ROS never exports, so it is a choice.
    ``ROS_LOCALHOST_ONLY`` is not consulted; callers that honour it check it first.

    Example:
        >>> discovery_range_is_ros_default({"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"})
        True
        >>> discovery_range_is_ros_default(
        ...     {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", ALLOW_NETWORK_ROS_ENV: "1"}
        ... )
        False
        >>> discovery_range_is_ros_default({"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"})
        False
    """
    discovery = env.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").strip().upper()
    if not discovery:
        return True
    return discovery == _ROS_DEFAULT_RANGE and env.get(ALLOW_NETWORK_ROS_ENV) != "1"


def networked_ros_scope(env: Mapping[str, str]) -> str | None:
    """Describe the scope when ``env`` lets the graph reach other hosts, else ``None``.

    Reads ``env`` as rcl (Jazzy) does: an unset or empty
    ``ROS_AUTOMATIC_DISCOVERY_RANGE`` is ``SUBNET``; any ``ROS_STATIC_PEERS``
    reaches out; ``ROS_LOCALHOST_ONLY=1`` overrides both ("'automatic_discovery_range'
    and 'static_peers' will be ignored"). Whatever those say, a Fast DDS
    ``ROS_DISCOVERY_SERVER``, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` (which reads
    neither variable; its router listens on every interface), or a vendor XML
    config that lists peers (``CYCLONEDDS_URI`` ``<Peers>``, a Fast DDS profiles
    file's ``initialPeersList`` or ``discoveryServersList``; an unreadable one
    counts too) reaches other hosts.

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
    """Refuse a networked ROS scope for a real robot unless ``env`` opts in and runs SROS2.

    Args:
        env: The environment the robot's graph runs under. The opt-in
            (``OPENRAL_ALLOW_NETWORK_ROS=1``) is read from it too.

    Returns:
        ``None`` when the scope is host-confined; the scope description when it
        is networked and opted in, so the caller logs the hazard.

    Raises:
        ROSConfigError: A networked scope without the opt-in or without SROS2
            Enforce (ADR-0130), or ``ROS_SECURITY_ENABLE=true`` with a keystore
            ``require_secure_ros_graph`` refuses.

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
    if ros_security_enabled(env):
        # A secured graph is checked whatever its scope: a broken keystore must
        # fail here, not as nodes that silently never join.
        require_secure_ros_graph(env)
    scope = networked_ros_scope(env)
    if scope is None:
        return None
    if env.get(ALLOW_NETWORK_ROS_ENV) != "1":
        raise ROSConfigError(
            f"refusing to start a real robot on a ROS graph that reaches other hosts "
            f"({scope}). Any host that can discover this graph can publish "
            f"/openral/safe_action (the HAL executes it), call /openral/estop_reset, "
            f"and spoof /joint_states. Keep the graph on this "
            f"host with ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST and no ROS_STATIC_PEERS "
            f"(`openral deploy run` sets this by default; sourcing ROS's setup.bash "
            f"exports SUBNET when the range is unset, so a hand-run launch must export "
            f"LOCALHOST itself), or, when the graph must span "
            f"machines, set {ALLOW_NETWORK_ROS_ENV}=1, run it under SROS2 "
            f"(`openral security init <robot_id>`) and export the same ROS_DOMAIN_ID on "
            f"every host.",
        )
    if env.get("RMW_IMPLEMENTATION", "").strip() == _ZENOH_RMW:
        # ADR-0130: Zenoh's access control is its own session/router config, not the
        # DDS keystore, so a valid keystore says nothing about who may join a Zenoh graph.
        raise ROSConfigError(
            f"refusing to start a real robot on a networked {_ZENOH_RMW} graph: SROS2 "
            f"enclaves secure DDS only. Use rmw_cyclonedds_cpp or rmw_fastrtps_cpp with "
            f"`openral security init`, or keep the Zenoh graph on this host."
        )
    if not ros_security_enabled(env):
        # ADR-0130: the opt-in alone is not enough; there is no unsecured networked cell.
        raise ROSConfigError(
            f"refusing to start a real robot on a networked ROS graph ({scope}) without "
            f"SROS2: {ALLOW_NETWORK_ROS_ENV}=1 acknowledges the network but does not stop a "
            f"peer from publishing /openral/safe_action. Run `openral security init "
            f"<robot_id>` and export ROS_SECURITY_ENABLE=true, ROS_SECURITY_STRATEGY=Enforce "
            f"and ROS_SECURITY_KEYSTORE on every host."
        )
    return scope


def require_host_confined_real_bringup(launch: str, env: Mapping[str, str]) -> None:
    """The gate every vendor ``real_bringup`` launch runs before composing any node.

    Applies ``require_host_confined_ros_scope`` to ``env`` (the launch's own ``os.environ``: its
    nodes inherit it) and logs the hazard when the graph is networked and opted in (structlog
    ``ros_graph_scope.real_bringup_networked``). A hand-run bringup gets no host-confined default
    from ``openral deploy run``, so an unset or exported-``SUBNET`` range is refused.

    Args:
        launch: The launch's name, prefixed to the warning.
        env: The environment the bringup's nodes run under.

    Raises:
        ROSConfigError: Whatever ``require_host_confined_ros_scope`` refuses: a networked
            scope without ``OPENRAL_ALLOW_NETWORK_ROS=1`` and SROS2 Enforce (ADR-0130),
            or ``ROS_AUTOMATIC_DISCOVERY_RANGE=OFF``.

    Example:
        >>> require_host_confined_real_bringup(
        ...     "openral_hal_openarm", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
        ... )
    """
    scope = require_host_confined_ros_scope(env)
    if scope is not None:
        _log.warning(
            "ros_graph_scope.real_bringup_networked",
            launch=launch,
            scope=scope,
            hazard="every host on this network can command the robot",
        )
