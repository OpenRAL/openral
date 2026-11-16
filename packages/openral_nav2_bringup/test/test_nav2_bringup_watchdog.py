"""The watchdog's teardown returns an aborted Nav2 bring-up to a state STARTUP accepts."""

from __future__ import annotations

from openral_nav2_bringup.nav2_bringup_watchdog_node import (
    TRANSITION_CLEANUP,
    TRANSITION_DEACTIVATE,
    recovery_plan,
)

# lifecycle_manager_navigation's node_names (nav2_bringup navigation_launch.py, Jazzy).
NAV2_SERVERS = [
    "controller_server",
    "smoother_server",
    "planner_server",
    "route_server",
    "behavior_server",
    "velocity_smoother",
    "collision_monitor",
    "bt_navigator",
    "waypoint_follower",
    "docking_server",
]


def _replay(states: dict[str, str], plan: list[tuple[str, int]]) -> dict[str, str]:
    """Apply the plan with lifecycle semantics; an invalid transition fails the test."""
    out = dict(states)
    for name, transition in plan:
        if transition == TRANSITION_DEACTIVATE:
            assert out[name] == "active", (name, out[name])
            out[name] = "inactive"
        else:
            assert transition == TRANSITION_CLEANUP
            assert out[name] == "inactive", (name, out[name])
            out[name] = "unconfigured"
    return out


def test_the_spark_abort_returns_every_server_to_unconfigured() -> None:
    # Spark carry_task (2026-11-16): bt_navigator failed to activate; the manager stopped,
    # leaving the servers before it active and those after it configured but inactive.
    states = {n: "active" for n in NAV2_SERVERS[:7]}
    states |= {"bt_navigator": "inactive", "waypoint_follower": "inactive"}
    states |= {"docking_server": "inactive"}
    after = _replay(states, recovery_plan(states, NAV2_SERVERS))
    assert set(after.values()) == {"unconfigured"}


def test_an_abort_during_configure_touches_only_configured_servers() -> None:
    states = {n: "unconfigured" for n in NAV2_SERVERS} | {"controller_server": "inactive"}
    assert recovery_plan(states, NAV2_SERVERS) == [("controller_server", TRANSITION_CLEANUP)]


def test_an_unreachable_server_is_left_alone() -> None:
    states = {"controller_server": "active", "bt_navigator": "unknown"}
    plan = recovery_plan(states, ["controller_server", "bt_navigator"])
    assert all(name == "controller_server" for name, _ in plan)
