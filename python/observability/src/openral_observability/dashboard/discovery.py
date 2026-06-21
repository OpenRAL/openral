"""mDNS advertise + browse for the live dashboard (issue #75b).

The dashboard stays a passive OTLP receiver: this module lets it (1) advertise
its OTLP endpoint on the LAN so a discovery-capable workload can find it
without a hand-typed endpoint, and (2) browse for other advertised OpenRAL
services and surface them in the "Add Robot" panel.

``zeroconf`` is an optional dependency (the ``mdns`` extra). When it is not
importable, :class:`Discovery` stays disabled and the dashboard runs exactly as
before — discovery is additive, never load-bearing.
"""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from zeroconf import ServiceInfo, Zeroconf

__all__ = ["SERVICE_TYPE", "DiscoveredRobot", "Discovery", "RobotRegistry"]

_logger = structlog.get_logger(__name__)

SERVICE_TYPE = "_openral-otlp._tcp.local."

# Hosts for which advertising to the LAN is meaningless (and which we never
# expose without the loud server.py warning). See server.py _LOOPBACK_HOSTS.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


class DiscoveredRobot(BaseModel):
    """One mDNS-discovered OpenRAL service (the /api/robots wire shape)."""

    name: str
    addresses: list[str]
    port: int
    properties: dict[str, str] = Field(default_factory=dict)
    last_seen: float


class RobotRegistry:
    """Thread-safe map of discovered robots (zeroconf callbacks run off-thread)."""

    def __init__(self) -> None:
        """Initialise an empty registry with its lock."""
        self._robots: dict[str, DiscoveredRobot] = {}
        self._lock = threading.Lock()

    def upsert(self, robot: DiscoveredRobot) -> None:
        """Insert or replace the entry keyed by ``robot.name``.

        Args:
            robot: The discovered robot to store or update.
        """
        with self._lock:
            self._robots[robot.name] = robot

    def remove(self, name: str) -> None:
        """Remove the entry for *name* if it exists (idempotent).

        Args:
            name: The mDNS service name to remove.
        """
        with self._lock:
            self._robots.pop(name, None)

    def list_robots(self) -> list[DiscoveredRobot]:
        """Return all known robots sorted by name.

        Returns:
            A snapshot list; safe to iterate after the lock is released.
        """
        with self._lock:
            return sorted(self._robots.values(), key=lambda r: r.name)


class Discovery:
    """Owns the Zeroconf instance, advertiser, and browser for the dashboard."""

    def __init__(self, *, registry: RobotRegistry | None = None) -> None:
        """Initialise with an optional shared registry.

        Args:
            registry: Existing :class:`RobotRegistry` to use; a fresh one is
                created when *None*.
        """
        self.registry = registry if registry is not None else RobotRegistry()
        self.enabled = False
        self._zc: Zeroconf | None = None
        self._service_info: ServiceInfo | None = None
        self._browser: Any = None

    def robots(self) -> list[DiscoveredRobot]:
        """Delegate to the underlying registry.

        Returns:
            All currently known robots, sorted by name.
        """
        return self.registry.list_robots()

    def start(self, *, host: str, port: int, ts_now: float) -> None:
        """Start browsing always; advertise only on a non-loopback bind.

        ``ts_now`` is the current unix time (passed in — scripts/tests must not
        call ``time.time`` implicitly here; the caller stamps it).
        """
        try:
            from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
        except ImportError:
            _logger.warning("discovery.zeroconf_unavailable", hint="install the 'mdns' extra")
            return
        self._zc = Zeroconf()
        self._browser = ServiceBrowser(
            self._zc,
            SERVICE_TYPE,
            _RegistryListener(self.registry, ts_now),  # type: ignore[arg-type] # reason: _RegistryListener satisfies ServiceListener structurally but mypy can't verify duck-typing across a guarded import
        )
        if host.strip().lower() not in _LOOPBACK_HOSTS:
            self._service_info = ServiceInfo(
                SERVICE_TYPE,
                f"openral-dashboard-{socket.gethostname()}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(host)],
                port=port,
                properties={b"path": b"/v1", b"role": b"dashboard"},
            )
            self._zc.register_service(self._service_info)
            _logger.info("discovery.advertising", host=host, port=port)
        else:
            _logger.info("discovery.browse_only", reason="loopback bind not advertised")
        self.enabled = True

    def stop(self) -> None:
        """Unregister the advertised service and close the Zeroconf socket."""
        if self._zc is None:
            return
        if self._service_info is not None:
            self._zc.unregister_service(self._service_info)
        self._zc.close()
        self._zc = None
        self.enabled = False


class _RegistryListener:
    """zeroconf ServiceListener → RobotRegistry adapter."""

    def __init__(self, registry: RobotRegistry, ts_now: float) -> None:
        self._registry = registry
        self._ts = ts_now

    def _resolve(self, zc: Zeroconf, name: str) -> None:
        info = zc.get_service_info(SERVICE_TYPE, name, timeout=2000)
        if info is None:
            return
        self._registry.upsert(
            DiscoveredRobot(
                name=name,
                addresses=[addr for addr in info.parsed_addresses()],
                port=info.port or 0,
                properties={
                    k.decode("utf-8", "replace"): (v or b"").decode("utf-8", "replace")
                    for k, v in (info.properties or {}).items()
                },
                last_seen=self._ts,
            )
        )

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._resolve(zc, name)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._resolve(zc, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._registry.remove(name)
