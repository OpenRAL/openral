"""Unit tests for the dashboard mDNS discovery registry (issue #75b).

The zeroconf network wiring is exercised only when zeroconf + a network are
present; these tests cover the pure registry + the JSON-facing model, which
need neither.
"""

from __future__ import annotations

from openral_observability.dashboard.discovery import (
    DiscoveredRobot,
    Discovery,
    RobotRegistry,
)


def _robot(name: str, ts: float) -> DiscoveredRobot:
    return DiscoveredRobot(
        name=name, addresses=["10.0.0.5"], port=4318, properties={"version": "0.1"}, last_seen=ts
    )


def test_registry_upsert_replaces_by_name() -> None:
    reg = RobotRegistry()
    reg.upsert(_robot("arm._openral-otlp._tcp.local.", 1.0))
    reg.upsert(_robot("arm._openral-otlp._tcp.local.", 2.0))
    robots = reg.list_robots()
    assert len(robots) == 1
    assert robots[0].last_seen == 2.0


def test_registry_remove() -> None:
    reg = RobotRegistry()
    reg.upsert(_robot("a", 1.0))
    reg.remove("a")
    assert reg.list_robots() == []
    reg.remove("a")  # idempotent — no raise


def test_discovery_robots_delegates_to_registry() -> None:
    reg = RobotRegistry()
    reg.upsert(_robot("a", 1.0))
    disc = Discovery(registry=reg)
    assert disc.enabled is False  # not started
    assert [r.name for r in disc.robots()] == ["a"]
